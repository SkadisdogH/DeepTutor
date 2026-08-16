"""Content-hash upload dedupe for knowledge bases.

Covers the full dedupe contract:

* the hash store (``<kb>/metadata.json`` → ``file_hashes``) read/write/backfill;
* batch creation flagging files duplicate from a client-supplied sha256;
* ``complete`` skipping duplicate-flagged files (no parts required) and
  returning ``task_id: None`` when every file is a duplicate;
* save-time dedupe in ``_save_uploaded_files`` against real bytes (the
  authoritative check, covering clients that send no hash);
* lazy backfill healing KBs whose raw files predate hash recording.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest

try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover - optional dependency in lightweight envs
    FastAPI = None
    TestClient = None

pytestmark = pytest.mark.skipif(
    FastAPI is None or TestClient is None, reason="fastapi not installed"
)

if FastAPI is not None and TestClient is not None:
    knowledge_router_module = importlib.import_module("deeptutor.api.routers.knowledge")
    router = knowledge_router_module.router
    from deeptutor.knowledge import file_hashes
else:  # pragma: no cover - optional dependency in lightweight envs
    knowledge_router_module = None
    router = None
    file_hashes = None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _build_app() -> FastAPI:
    if FastAPI is None or router is None:  # pragma: no cover - guarded by pytestmark
        raise RuntimeError("fastapi is not installed")
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/knowledge")
    return app


class _FakeKBManager:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.config: dict[str, dict] = {"knowledge_bases": {}}

    def _load_config(self) -> dict:
        return self.config

    def _save_config(self) -> None:
        pass

    def list_knowledge_bases(self) -> list[str]:
        return sorted(self.config.get("knowledge_bases", {}).keys())

    def update_kb_status(self, name: str, status: str, progress: dict | None = None) -> None:
        entry = self.config.setdefault("knowledge_bases", {}).setdefault(name, {"path": name})
        entry["status"] = status
        entry["progress"] = progress or {}

    def get_default(self, *, available_names: list[str] | None = None) -> str | None:
        names = available_names if available_names is not None else self.list_knowledge_bases()
        return names[0] if names else None

    def get_knowledge_base_path(self, name: str) -> Path:
        kb_dir = self.base_dir / name
        kb_dir.mkdir(parents=True, exist_ok=True)
        return kb_dir


def _make_manager(tmp_path: Path, name: str = "ready-kb") -> _FakeKBManager:
    manager = _FakeKBManager(tmp_path / "knowledge_bases")
    manager.config["knowledge_bases"][name] = {
        "path": name,
        "rag_provider": "llamaindex",
        "needs_reindex": False,
        "status": "ready",
    }
    return manager


def _setup_router(monkeypatch, tmp_path: Path, manager: _FakeKBManager) -> None:
    monkeypatch.setattr(knowledge_router_module, "get_kb_manager", lambda: manager)
    monkeypatch.setattr(knowledge_router_module, "_kb_base_dir", tmp_path / "knowledge_bases")
    monkeypatch.setattr(
        knowledge_router_module,
        "_chunk_upload_root",
        lambda: tmp_path / "chunked_uploads",
    )


def _record_hash(manager: _FakeKBManager, kb_name: str, rel_path: str, data: bytes) -> None:
    """Seed the KB's hash store directly (as a previous upload would have)."""
    kb_dir = manager.base_dir / kb_name
    kb_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = kb_dir / "metadata.json"
    metadata = (
        json.loads(metadata_file.read_text(encoding="utf-8"))
        if metadata_file.exists()
        else {}
    )
    metadata.setdefault("file_hashes", {})[rel_path] = _sha256(data)
    metadata_file.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")


def _read_hashes(manager: _FakeKBManager, kb_name: str) -> dict:
    metadata_file = manager.base_dir / kb_name / "metadata.json"
    if not metadata_file.exists():
        return {}
    return json.loads(metadata_file.read_text(encoding="utf-8")).get("file_hashes", {})


# ---------------------------------------------------------------------------
# Hash store primitives
# ---------------------------------------------------------------------------


def test_sha256_helpers_match_hashlib(tmp_path: Path) -> None:
    data = b"the quick brown fox jumps over the lazy dog" * 1000
    path = tmp_path / "f.bin"
    path.write_bytes(data)

    assert file_hashes.sha256_file(path) == _sha256(data)
    with open(path, "rb") as fh:
        assert file_hashes.sha256_stream(fh) == _sha256(data)
    assert file_hashes.is_sha256_hex(_sha256(data))
    assert not file_hashes.is_sha256_hex("")
    assert not file_hashes.is_sha256_hex(_sha256(data).upper())
    assert not file_hashes.is_sha256_hex("abc")


def test_record_and_read_file_hashes_merge(tmp_path: Path) -> None:
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()

    file_hashes.record_file_hashes(kb_dir, {"a.md": "aa" * 32, "b.md": "bb" * 32})
    file_hashes.record_file_hashes(kb_dir, {"b.md": "bb" * 32, "c.md": "cc" * 32})

    assert file_hashes.read_file_hashes(kb_dir) == {
        "a.md": "aa" * 32,
        "b.md": "bb" * 32,
        "c.md": "cc" * 32,
    }


def test_read_file_hashes_tolerates_corruption(tmp_path: Path) -> None:
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    (kb_dir / "metadata.json").write_text("{not json", encoding="utf-8")
    assert file_hashes.read_file_hashes(kb_dir) == {}


def test_ensure_raw_hashes_backfills_and_persists(tmp_path: Path) -> None:
    kb_dir = tmp_path / "kb"
    raw_dir = kb_dir / "raw"
    (raw_dir / "sub").mkdir(parents=True)
    (raw_dir / "a.md").write_bytes(b"alpha")
    (raw_dir / "sub" / "b.md").write_bytes(b"beta")
    (raw_dir / ".hidden").write_bytes(b"secret")

    result = file_hashes.ensure_raw_hashes(kb_dir, raw_dir)

    assert result == {"a.md": _sha256(b"alpha"), "sub/b.md": _sha256(b"beta")}
    # Backfill is persisted, so a second call is a pure store read.
    assert file_hashes.read_file_hashes(kb_dir) == result
    assert file_hashes.dedupe_hash_set(kb_dir) == set(result.values())


def test_dedupe_hash_set_is_read_only_never_backfills(tmp_path: Path) -> None:
    """The dedupe query must not invent records for unindexed raw files."""
    kb_dir = tmp_path / "kb"
    raw_dir = kb_dir / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "never_indexed.pdf").write_bytes(b"orphan bytes")

    assert file_hashes.dedupe_hash_set(kb_dir) == set()
    # The query left the store untouched — the file stays re-uploadable.
    assert file_hashes.read_file_hashes(kb_dir) == {}


def test_dedupe_hash_set_returns_recorded_hashes_when_raw_missing(tmp_path: Path) -> None:
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    _record_hash_for(kb_dir, "gone.md", b"gone")

    assert file_hashes.dedupe_hash_set(kb_dir) == {_sha256(b"gone")}


def _record_hash_for(kb_dir: Path, rel_path: str, data: bytes) -> None:
    file_hashes.record_file_hashes(kb_dir, {rel_path: _sha256(data)})


# ---------------------------------------------------------------------------
# Batch creation flags duplicates from a client-supplied sha256
# ---------------------------------------------------------------------------


def test_batch_flags_duplicate_from_client_sha256(monkeypatch, tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    _setup_router(monkeypatch, tmp_path, manager)
    _record_hash(manager, "ready-kb", "book.pdf", b"already here")

    with TestClient(_build_app()) as client:
        response = client.post(
            "/api/v1/knowledge/ready-kb/upload/batch",
            json={
                "files": [
                    {"name": "book.pdf", "size": 12, "rel_path": "", "sha256": _sha256(b"already here")},
                    {"name": "new.pdf", "size": 5, "rel_path": "", "sha256": _sha256(b"new")},
                    {"name": "no-hash.pdf", "size": 5, "rel_path": ""},
                ]
            },
        )

    assert response.status_code == 200
    files = response.json()["files"]
    assert files[0]["duplicate"] is True
    assert files[1]["duplicate"] is False
    # No client hash → no verdict (authoritative check happens at complete).
    assert files[2]["duplicate"] is False


def test_batch_ignores_invalid_sha256(monkeypatch, tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    _setup_router(monkeypatch, tmp_path, manager)
    _record_hash(manager, "ready-kb", "book.pdf", b"already here")

    with TestClient(_build_app()) as client:
        response = client.post(
            "/api/v1/knowledge/ready-kb/upload/batch",
            json={
                "files": [
                    {"name": "book.pdf", "size": 12, "rel_path": "", "sha256": "not-a-hash"},
                ]
            },
        )

    assert response.status_code == 200
    assert response.json()["files"][0]["duplicate"] is False


def test_batch_does_not_backfill_unrecorded_raw_files(monkeypatch, tmp_path: Path) -> None:
    """A raw file with no hash record is NOT a known duplicate — the dedupe
    query is read-only, so its content can still be (re)uploaded."""
    manager = _make_manager(tmp_path)
    _setup_router(monkeypatch, tmp_path, manager)
    # File on disk, no hash record anywhere (e.g. its indexing failed).
    raw_dir = manager.base_dir / "ready-kb" / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "failed.pdf").write_bytes(b"legacy content")

    with TestClient(_build_app()) as client:
        response = client.post(
            "/api/v1/knowledge/ready-kb/upload/batch",
            json={
                "files": [
                    {"name": "copy.pdf", "size": 14, "rel_path": "", "sha256": _sha256(b"legacy content")},
                ]
            },
        )

    assert response.status_code == 200
    assert response.json()["files"][0]["duplicate"] is False
    # The query never wrote to the store.
    assert _read_hashes(manager, "ready-kb") == {}


# ---------------------------------------------------------------------------
# Complete: duplicate-flagged files need no parts; all-duplicate batches
# dispatch no task
# ---------------------------------------------------------------------------


def test_complete_skips_duplicate_flagged_files(monkeypatch, tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    _setup_router(monkeypatch, tmp_path, manager)
    _record_hash(manager, "ready-kb", "book.pdf", b"dup bytes")

    calls: list[dict] = []

    async def _fake_task(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(knowledge_router_module, "run_upload_processing_task", _fake_task)

    with TestClient(_build_app()) as client:
        created = client.post(
            "/api/v1/knowledge/ready-kb/upload/batch",
            json={
                "files": [
                    {"name": "book.pdf", "size": 9, "sha256": _sha256(b"dup bytes")},
                    {"name": "fresh.pdf", "size": 5},
                ]
            },
        ).json()
        batch_id = created["batch_id"]
        # Only the fresh file's parts are sent — the duplicate needs none.
        client.post(
            "/api/v1/knowledge/ready-kb/upload/chunk",
            data={"batch_id": batch_id, "file_index": "1", "chunk_index": "0"},
            files={"data": ("part.bin", b"fresh", "application/octet-stream")},
        )
        response = client.post(
            "/api/v1/knowledge/ready-kb/upload/complete", json={"batch_id": batch_id}
        )

    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "Uploaded 1 files. Processing in background."
    assert body["files"] == ["fresh.pdf"]
    assert body["skipped"] == [{"name": "book.pdf", "reason": "duplicate"}]
    assert body["task_id"]

    raw_dir = tmp_path / "knowledge_bases" / "ready-kb" / "raw"
    assert (raw_dir / "fresh.pdf").read_bytes() == b"fresh"
    assert not (raw_dir / "book.pdf").exists()
    assert calls and sorted(Path(p).name for p in calls[0]["uploaded_file_paths"]) == ["fresh.pdf"]


def test_complete_all_duplicates_dispatches_nothing(monkeypatch, tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    _setup_router(monkeypatch, tmp_path, manager)
    _record_hash(manager, "ready-kb", "book.pdf", b"dup bytes")

    calls: list[dict] = []

    async def _fake_task(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(knowledge_router_module, "run_upload_processing_task", _fake_task)

    with TestClient(_build_app()) as client:
        created = client.post(
            "/api/v1/knowledge/ready-kb/upload/batch",
            json={
                "files": [
                    {"name": "book.pdf", "size": 9, "sha256": _sha256(b"dup bytes")},
                ]
            },
        ).json()
        batch_id = created["batch_id"]
        response = client.post(
            "/api/v1/knowledge/ready-kb/upload/complete", json={"batch_id": batch_id}
        )

    assert response.status_code == 200
    body = response.json()
    assert body["task_id"] is None
    assert body["files"] == []
    assert body["skipped"] == [{"name": "book.pdf", "reason": "duplicate"}]
    assert "already exist" in body["message"]
    assert calls == []  # no background task
    # KB was never flipped to processing.
    assert manager.config["knowledge_bases"]["ready-kb"]["status"] == "ready"
    # Batch dir cleaned up like a successful complete.
    assert not (tmp_path / "chunked_uploads" / batch_id).exists()


def test_complete_dedupes_by_real_bytes_without_client_hash(monkeypatch, tmp_path: Path) -> None:
    """A client that sends no sha256 still gets deduped at complete."""
    manager = _make_manager(tmp_path)
    _setup_router(monkeypatch, tmp_path, manager)
    _record_hash(manager, "ready-kb", "book.pdf", b"same bytes")

    calls: list[dict] = []

    async def _fake_task(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(knowledge_router_module, "run_upload_processing_task", _fake_task)

    with TestClient(_build_app()) as client:
        created = client.post(
            "/api/v1/knowledge/ready-kb/upload/batch",
            json={"files": [{"name": "book.pdf", "size": 10}]},
        ).json()
        batch_id = created["batch_id"]
        client.post(
            "/api/v1/knowledge/ready-kb/upload/chunk",
            data={"batch_id": batch_id, "file_index": "0", "chunk_index": "0"},
            files={"data": ("part.bin", b"same bytes", "application/octet-stream")},
        )
        response = client.post(
            "/api/v1/knowledge/ready-kb/upload/complete", json={"batch_id": batch_id}
        )

    assert response.status_code == 200
    body = response.json()
    assert body["files"] == []
    assert body["skipped"] == [{"name": "book.pdf", "reason": "duplicate"}]
    assert body["task_id"] is None
    assert calls == []
    assert not (tmp_path / "knowledge_bases" / "ready-kb" / "raw" / "book.pdf").exists()


# ---------------------------------------------------------------------------
# Save-time dedupe in the classic multipart route (authoritative, real bytes)
# ---------------------------------------------------------------------------


def test_classic_upload_skips_upload_matching_recorded_hash(monkeypatch, tmp_path: Path) -> None:
    """Re-uploading content whose hash was RECORDED (i.e. successfully indexed
    earlier) is skipped without persisting or dispatching a task."""
    manager = _make_manager(tmp_path)
    _setup_router(monkeypatch, tmp_path, manager)
    # A previous successful indexing recorded this content's hash.
    _record_hash(manager, "ready-kb", "book.pdf", b"same content")

    calls: list[dict] = []

    async def _fake_task(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(knowledge_router_module, "run_upload_processing_task", _fake_task)

    with TestClient(_build_app()) as client:
        response = client.post(
            "/api/v1/knowledge/ready-kb/upload",
            files={"files": ("copy.pdf", b"same content", "application/pdf")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["files"] == []
    assert body["skipped"] == [{"name": "copy.pdf", "reason": "duplicate"}]
    assert body["task_id"] is None
    assert calls == []
    raw_dir = tmp_path / "knowledge_bases" / "ready-kb" / "raw"
    assert not (raw_dir / "copy.pdf").exists()
    # The dedupe pass never wrote anything to the store.
    assert _read_hashes(manager, "ready-kb") == {"book.pdf": _sha256(b"same content")}


def test_save_does_not_record_hashes_before_indexing(monkeypatch, tmp_path: Path) -> None:
    """Regression guard: saving a NEW file must NOT write its hash into the
    store — the record belongs to the post-indexing step. Recording at save
    time made the file look 'already indexed' to its own processing task,
    which then silently skipped it (Staged 0)."""
    manager = _make_manager(tmp_path)
    _setup_router(monkeypatch, tmp_path, manager)

    async def _fake_task(**kwargs):
        return None

    monkeypatch.setattr(knowledge_router_module, "run_upload_processing_task", _fake_task)

    with TestClient(_build_app()) as client:
        response = client.post(
            "/api/v1/knowledge/ready-kb/upload",
            files={"files": ("new-book.pdf", b"fresh bytes", "application/pdf")},
        )

    assert response.status_code == 200
    assert response.json()["files"] == ["new-book.pdf"]
    assert response.json()["skipped"] == []
    raw_dir = tmp_path / "knowledge_bases" / "ready-kb" / "raw"
    assert (raw_dir / "new-book.pdf").read_bytes() == b"fresh bytes"
    # The store is still empty: indexing has not happened yet.
    assert _read_hashes(manager, "ready-kb") == {}


def test_staging_does_not_skip_freshly_saved_file(tmp_path: Path) -> None:
    """End-to-end guard at the adder level: with an empty hash store, the
    processing task's staging step must return a just-saved raw file for
    indexing (no self-match), and only AFTER indexing records the hash does a
    repeat staging skip it as a duplicate."""
    from deeptutor.knowledge.add_documents import DocumentAdder

    kb_dir = tmp_path / "ready-kb"
    raw_dir = kb_dir / "raw"
    raw_dir.mkdir(parents=True)
    # Ready provider index (constructor requires it) + provider binding.
    version_dir = kb_dir / "version-1"
    version_dir.mkdir()
    (version_dir / "docstore.json").write_text("{}", encoding="utf-8")
    (version_dir / "index_store.json").write_text("{}", encoding="utf-8")
    (version_dir / "meta.json").write_text(
        json.dumps(
            {"provider": "llamaindex", "signature": "llamaindex", "version": "version-1"}
        ),
        encoding="utf-8",
    )
    (kb_dir / "metadata.json").write_text(
        json.dumps({"rag_provider": "llamaindex", "version": "1.0"}),
        encoding="utf-8",
    )

    saved = raw_dir / "new-book.pdf"
    saved.write_bytes(b"fresh bytes")

    adder = DocumentAdder(kb_name="ready-kb", base_dir=str(tmp_path))

    # First staging pass: empty store → the just-saved file is NOT a duplicate.
    staged = adder.add_documents([str(saved)], allow_duplicates=False)
    assert staged == [saved]

    # Simulation of a successful index run: only now is the hash recorded.
    adder._record_successful_hash(saved)
    assert adder.get_ingested_hashes() == {"new-book.pdf": _sha256(b"fresh bytes")}

    # A second staging pass now skips the same content.
    assert adder.add_documents([str(saved)], allow_duplicates=False) == []


def test_classic_upload_keeps_same_name_different_content(monkeypatch, tmp_path: Path) -> None:
    """Same filename with different bytes is a replacement, not a duplicate —
    and the replacement must not be recorded before it is indexed."""
    manager = _make_manager(tmp_path)
    _setup_router(monkeypatch, tmp_path, manager)

    async def _fake_task(**kwargs):
        return None

    monkeypatch.setattr(knowledge_router_module, "run_upload_processing_task", _fake_task)

    with TestClient(_build_app()) as client:
        first = client.post(
            "/api/v1/knowledge/ready-kb/upload",
            files={"files": ("book.pdf", b"version one", "application/pdf")},
        )
        second = client.post(
            "/api/v1/knowledge/ready-kb/upload",
            files={"files": ("book.pdf", b"version two", "application/pdf")},
        )

    assert first.json()["skipped"] == []
    assert second.json()["skipped"] == []
    assert second.json()["files"] == ["book.pdf"]

    raw_dir = tmp_path / "knowledge_bases" / "ready-kb" / "raw"
    assert (raw_dir / "book.pdf").read_bytes() == b"version two"
    # Nothing recorded yet — indexing never ran in this test.
    assert _read_hashes(manager, "ready-kb") == {}
