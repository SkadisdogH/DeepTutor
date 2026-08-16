from __future__ import annotations

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
else:  # pragma: no cover - optional dependency in lightweight envs
    knowledge_router_module = None
    router = None


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


# A tiny payload helper: chunked uploads hit three endpoints, so drive the
# whole flow through one helper.
def _chunk_payload(name: str, size: int, rel_path: str = "") -> dict:
    return {"name": name, "size": size, "rel_path": rel_path}


def _send_parts(
    client,
    kb_name: str,
    batch_id: str,
    files: list[tuple[str, bytes]],
    chunk_size: int | None = None,
) -> list:
    """Upload all parts of a batch the way the web client does (serial parts)."""
    responses = []
    for file_index, (_name, content) in enumerate(files):
        if chunk_size is None:
            chunk_size = 8 * 1024 * 1024
        start = 0
        chunk_index = 0
        while start < len(content) or (len(content) == 0 and chunk_index == 0):
            end = min(start + chunk_size, len(content))
            part = content[start:end]
            resp = client.post(
                f"/api/v1/knowledge/{kb_name}/upload/chunk",
                data={
                    "batch_id": batch_id,
                    "file_index": str(file_index),
                    "chunk_index": str(chunk_index),
                },
                files={"data": ("part.bin", part, "application/octet-stream")},
            )
            responses.append(resp)
            if len(content) == 0:
                break
            start = end
            chunk_index += 1
            if start >= len(content):
                break
    return responses


def test_create_batch_returns_chunk_plan(monkeypatch, tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    _setup_router(monkeypatch, tmp_path, manager)
    monkeypatch.setattr(
        knowledge_router_module,
        "_chunk_upload_root",
        lambda: tmp_path / "chunked_uploads",
    )

    with TestClient(_build_app()) as client:
        response = client.post(
            "/api/v1/knowledge/ready-kb/upload/batch",
            json={
                "files": [
                    _chunk_payload("big.txt", 9 * 1024 * 1024),
                    _chunk_payload("tiny.md", 3),
                ]
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["batch_id"]
    assert body["chunk_size"] == 8 * 1024 * 1024
    assert body["max_file_size_bytes"] == 200 * 1024 * 1024
    assert body["files"][0] == {
        "index": 0,
        "name": "big.txt",
        "rel_path": "",
        "size": 9 * 1024 * 1024,
        "chunk_count": 2,
        "duplicate": False,
    }
    assert body["files"][1]["chunk_count"] == 1

    batch_dir = tmp_path / "chunked_uploads" / body["batch_id"]
    assert batch_dir.exists()
    manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["kb_name"] == "ready-kb"
    assert len(manifest["files"]) == 2


def test_create_batch_accepts_rel_path(monkeypatch, tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    _setup_router(monkeypatch, tmp_path, manager)

    with TestClient(_build_app()) as client:
        response = client.post(
            "/api/v1/knowledge/ready-kb/upload/batch",
            json={"files": [_chunk_payload("notes.md", 10, "folder/sub/notes.md")]},
        )

    assert response.status_code == 200
    assert response.json()["files"][0]["rel_path"] == "folder/sub/notes.md"


def test_create_batch_rejects_unknown_kb(monkeypatch, tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    _setup_router(monkeypatch, tmp_path, manager)

    with TestClient(_build_app()) as client:
        response = client.post(
            "/api/v1/knowledge/nope/upload/batch",
            json={"files": [_chunk_payload("a.txt", 1)]},
        )

    assert response.status_code == 404


def test_create_batch_rejects_needs_reindex(monkeypatch, tmp_path: Path) -> None:
    manager = _make_manager(tmp_path, name="legacy-kb")
    manager.config["knowledge_bases"]["legacy-kb"]["needs_reindex"] = True
    manager.config["knowledge_bases"]["legacy-kb"]["status"] = "needs_reindex"
    _setup_router(monkeypatch, tmp_path, manager)

    with TestClient(_build_app()) as client:
        response = client.post(
            "/api/v1/knowledge/legacy-kb/upload/batch",
            json={"files": [_chunk_payload("a.txt", 1)]},
        )

    assert response.status_code == 409
    assert "reindex" in response.json()["detail"].lower()


def test_create_batch_rejects_empty_files(monkeypatch, tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    _setup_router(monkeypatch, tmp_path, manager)

    with TestClient(_build_app()) as client:
        response = client.post(
            "/api/v1/knowledge/ready-kb/upload/batch", json={"files": []}
        )

    assert response.status_code == 400


def test_create_batch_rejects_file_too_large(monkeypatch, tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    _setup_router(monkeypatch, tmp_path, manager)

    with TestClient(_build_app()) as client:
        response = client.post(
            "/api/v1/knowledge/ready-kb/upload/batch",
            json={"files": [_chunk_payload("big.txt", 200 * 1024 * 1024 + 1)]},
        )

    assert response.status_code == 400
    assert "maximum" in response.json()["detail"].lower()


def test_create_batch_rejects_unsupported_extension(monkeypatch, tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    _setup_router(monkeypatch, tmp_path, manager)

    with TestClient(_build_app()) as client:
        response = client.post(
            "/api/v1/knowledge/ready-kb/upload/batch",
            json={"files": [_chunk_payload("evil.exe", 10)]},
        )

    assert response.status_code == 400
    assert "unsupported" in response.json()["detail"].lower()


def test_create_batch_rejects_duplicate_names(monkeypatch, tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    _setup_router(monkeypatch, tmp_path, manager)

    with TestClient(_build_app()) as client:
        response = client.post(
            "/api/v1/knowledge/ready-kb/upload/batch",
            json={"files": [_chunk_payload("a.txt", 1), _chunk_payload("a.txt", 2)]},
        )

    assert response.status_code == 400
    assert "duplicate" in response.json()["detail"].lower()


def test_chunk_upload_persists_part(monkeypatch, tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    _setup_router(monkeypatch, tmp_path, manager)
    monkeypatch.setattr(
        knowledge_router_module,
        "_chunk_upload_root",
        lambda: tmp_path / "chunked_uploads",
    )

    content = b"x" * 100
    with TestClient(_build_app()) as client:
        created = client.post(
            "/api/v1/knowledge/ready-kb/upload/batch",
            json={"files": [_chunk_payload("a.txt", 100)]},
        ).json()
        responses = _send_parts(client, "ready-kb", created["batch_id"], [("a.txt", content)])

    assert all(r.status_code == 200 for r in responses)
    part = tmp_path / "chunked_uploads" / created["batch_id"] / "f0" / "part_000000"
    assert part.read_bytes() == content


def test_chunk_upload_is_idempotent(monkeypatch, tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    _setup_router(monkeypatch, tmp_path, manager)
    monkeypatch.setattr(
        knowledge_router_module,
        "_chunk_upload_root",
        lambda: tmp_path / "chunked_uploads",
    )

    with TestClient(_build_app()) as client:
        created = client.post(
            "/api/v1/knowledge/ready-kb/upload/batch",
            json={"files": [_chunk_payload("a.txt", 5)]},
        ).json()
        first = client.post(
            "/api/v1/knowledge/ready-kb/upload/chunk",
            data={"batch_id": created["batch_id"], "file_index": "0", "chunk_index": "0"},
            files={"data": ("part.bin", b"hello", "application/octet-stream")},
        )
        second = client.post(
            "/api/v1/knowledge/ready-kb/upload/chunk",
            data={"batch_id": created["batch_id"], "file_index": "0", "chunk_index": "0"},
            files={"data": ("part.bin", b"WORLD", "application/octet-stream")},
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["duplicate"] is True
    # The original part must not be overwritten.
    part = tmp_path / "chunked_uploads" / created["batch_id"] / "f0" / "part_000000"
    assert part.read_bytes() == b"hello"


def test_chunk_upload_rejects_invalid_indexes(monkeypatch, tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    _setup_router(monkeypatch, tmp_path, manager)

    with TestClient(_build_app()) as client:
        created = client.post(
            "/api/v1/knowledge/ready-kb/upload/batch",
            json={"files": [_chunk_payload("a.txt", 5)]},
        ).json()
        bad_file = client.post(
            "/api/v1/knowledge/ready-kb/upload/chunk",
            data={"batch_id": created["batch_id"], "file_index": "3", "chunk_index": "0"},
            files={"data": ("part.bin", b"x", "application/octet-stream")},
        )
        bad_chunk = client.post(
            "/api/v1/knowledge/ready-kb/upload/chunk",
            data={"batch_id": created["batch_id"], "file_index": "0", "chunk_index": "9"},
            files={"data": ("part.bin", b"x", "application/octet-stream")},
        )

    assert bad_file.status_code == 400
    assert bad_chunk.status_code == 400


def test_chunk_upload_rejects_unknown_batch(monkeypatch, tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    _setup_router(monkeypatch, tmp_path, manager)

    with TestClient(_build_app()) as client:
        response = client.post(
            "/api/v1/knowledge/ready-kb/upload/chunk",
            data={"batch_id": "deadbeef", "file_index": "0", "chunk_index": "0"},
            files={"data": ("part.bin", b"x", "application/octet-stream")},
        )

    assert response.status_code == 404


def test_chunk_upload_rejects_oversized_part(monkeypatch, tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    _setup_router(monkeypatch, tmp_path, manager)
    # Shrink the chunk size so the test does not need 8MB of memory.
    monkeypatch.setattr(knowledge_router_module, "UPLOAD_CHUNK_SIZE", 1000)

    with TestClient(_build_app()) as client:
        created = client.post(
            "/api/v1/knowledge/ready-kb/upload/batch",
            json={"files": [_chunk_payload("a.txt", 1000)]},
        ).json()
        response = client.post(
            "/api/v1/knowledge/ready-kb/upload/chunk",
            data={"batch_id": created["batch_id"], "file_index": "0", "chunk_index": "0"},
            files={"data": ("part.bin", b"x" * 1001, "application/octet-stream")},
        )

    assert response.status_code == 400
    assert "exceeds chunk size" in response.json()["detail"]


def test_complete_merges_persists_and_cleans_up(monkeypatch, tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    _setup_router(monkeypatch, tmp_path, manager)
    monkeypatch.setattr(
        knowledge_router_module,
        "_chunk_upload_root",
        lambda: tmp_path / "chunked_uploads",
    )

    calls: list[dict] = []

    async def _fake_task(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(knowledge_router_module, "run_upload_processing_task", _fake_task)

    file_a = b"A" * (8 * 1024 * 1024 + 123)  # 2 chunks
    file_b = b"B" * 1000  # 1 chunk
    with TestClient(_build_app()) as client:
        created = client.post(
            "/api/v1/knowledge/ready-kb/upload/batch",
            json={"files": [_chunk_payload("a.txt", len(file_a)), _chunk_payload("b.txt", len(file_b))]},
        ).json()
        batch_id = created["batch_id"]
        assert all(
            r.status_code == 200
            for r in _send_parts(client, "ready-kb", batch_id, [("a.txt", file_a), ("b.txt", file_b)])
        )
        response = client.post(
            "/api/v1/knowledge/ready-kb/upload/complete", json={"batch_id": batch_id}
        )

    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "Uploaded 2 files. Processing in background."
    assert sorted(body["files"]) == ["a.txt", "b.txt"]
    assert body["task_id"]

    raw_dir = tmp_path / "knowledge_bases" / "ready-kb" / "raw"
    assert (raw_dir / "a.txt").read_bytes() == file_a
    assert (raw_dir / "b.txt").read_bytes() == file_b

    # KB flipped to processing before the (no-op) task dispatch.
    assert manager.config["knowledge_bases"]["ready-kb"]["status"] == "processing"
    assert calls and calls[0]["kb_name"] == "ready-kb"
    assert sorted(Path(p).name for p in calls[0]["uploaded_file_paths"]) == ["a.txt", "b.txt"]

    # Temporary parts are gone after a successful complete.
    assert not (tmp_path / "chunked_uploads" / batch_id).exists()


def test_complete_reports_missing_chunks_and_keeps_parts(monkeypatch, tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    _setup_router(monkeypatch, tmp_path, manager)
    monkeypatch.setattr(
        knowledge_router_module,
        "_chunk_upload_root",
        lambda: tmp_path / "chunked_uploads",
    )

    async def _fake_task(**kwargs):
        return None

    monkeypatch.setattr(knowledge_router_module, "run_upload_processing_task", _fake_task)

    file_a = b"A" * (8 * 1024 * 1024 + 123)  # 2 chunks, only 1 sent
    with TestClient(_build_app()) as client:
        created = client.post(
            "/api/v1/knowledge/ready-kb/upload/batch",
            json={"files": [_chunk_payload("a.txt", len(file_a))]},
        ).json()
        batch_id = created["batch_id"]
        client.post(
            "/api/v1/knowledge/ready-kb/upload/chunk",
            data={"batch_id": batch_id, "file_index": "0", "chunk_index": "0"},
            files={"data": ("part.bin", file_a[: 8 * 1024 * 1024], "application/octet-stream")},
        )
        response = client.post(
            "/api/v1/knowledge/ready-kb/upload/complete", json={"batch_id": batch_id}
        )
        # A failed complete keeps the parts so the client can top up and retry.
        assert response.status_code == 400
        assert "incomplete" in response.json()["detail"]
        assert (tmp_path / "chunked_uploads" / batch_id).exists()
        # Topping up the missing part and completing again must succeed.
        client.post(
            "/api/v1/knowledge/ready-kb/upload/chunk",
            data={"batch_id": batch_id, "file_index": "0", "chunk_index": "1"},
            files={"data": ("part.bin", file_a[8 * 1024 * 1024 :], "application/octet-stream")},
        )
        retry = client.post(
            "/api/v1/knowledge/ready-kb/upload/complete", json={"batch_id": batch_id}
        )

    assert retry.status_code == 200
    assert not (tmp_path / "chunked_uploads" / batch_id).exists()


def test_complete_rejects_size_mismatch(monkeypatch, tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    _setup_router(monkeypatch, tmp_path, manager)
    monkeypatch.setattr(
        knowledge_router_module,
        "_chunk_upload_root",
        lambda: tmp_path / "chunked_uploads",
    )

    # Manifest claims 100 bytes but only 50 are sent.
    with TestClient(_build_app()) as client:
        created = client.post(
            "/api/v1/knowledge/ready-kb/upload/batch",
            json={"files": [_chunk_payload("a.txt", 100)]},
        ).json()
        batch_id = created["batch_id"]
        client.post(
            "/api/v1/knowledge/ready-kb/upload/chunk",
            data={"batch_id": batch_id, "file_index": "0", "chunk_index": "0"},
            files={"data": ("part.bin", b"x" * 50, "application/octet-stream")},
        )
        response = client.post(
            "/api/v1/knowledge/ready-kb/upload/complete", json={"batch_id": batch_id}
        )

    assert response.status_code == 400
    assert "size mismatch" in response.json()["detail"]


def test_complete_rejects_unknown_batch(monkeypatch, tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    _setup_router(monkeypatch, tmp_path, manager)

    with TestClient(_build_app()) as client:
        response = client.post(
            "/api/v1/knowledge/ready-kb/upload/complete", json={"batch_id": "nope"}
        )

    assert response.status_code == 404


def test_cancel_removes_batch_idempotently(monkeypatch, tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    _setup_router(monkeypatch, tmp_path, manager)
    monkeypatch.setattr(
        knowledge_router_module,
        "_chunk_upload_root",
        lambda: tmp_path / "chunked_uploads",
    )

    with TestClient(_build_app()) as client:
        created = client.post(
            "/api/v1/knowledge/ready-kb/upload/batch",
            json={"files": [_chunk_payload("a.txt", 5)]},
        ).json()
        batch_id = created["batch_id"]
        assert (tmp_path / "chunked_uploads" / batch_id).exists()

        first = client.post(
            "/api/v1/knowledge/ready-kb/upload/cancel", json={"batch_id": batch_id}
        )
        second = client.post(
            "/api/v1/knowledge/ready-kb/upload/cancel", json={"batch_id": batch_id}
        )

    assert first.status_code == 200
    assert second.status_code == 200  # idempotent
    assert not (tmp_path / "chunked_uploads" / batch_id).exists()
