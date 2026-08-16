"""Content-hash storage for knowledge-base raw files (upload dedupe).

Uploads are deduplicated by *content*, not by filename: two files with
identical bytes should only be transferred, persisted and indexed once. This
module owns the sha256 store that makes that possible:

* the store lives in ``<kb>/metadata.json`` under ``file_hashes``, a map of
  raw-relative POSIX path → sha256 hex (the same key rule the incremental-add
  path and ``remove_raw_document`` use, so removed files drop their record);
* a record means **the content was successfully indexed** — the only writers
  are ``DocumentAdder._record_successful_hash`` (after a successful
  incremental add) and :func:`ensure_raw_hashes` (called AFTER a successful
  KB creation/indexing run). Nothing records a hash before indexing has
  happened — doing so makes the newly-uploaded file look "already indexed"
  to its own processing task and silently skips it;
* :func:`dedupe_hash_set` is the read-only query the upload pipeline uses: the
  set of hashes already stored/recorded for this KB.

Why no lazy backfill on the dedupe path: a raw file whose index failed must
stay re-uploadable. Backfilling it into the store would make re-uploads of the
same bytes be treated as duplicates even though the content never made it into
retrieval. Backfill is therefore only ever an explicit migration step (e.g.
after a successful create/reindex), never a side effect of asking "is this a
duplicate?".

Hash writes are best-effort merges through ``atomic_write_json`` (temp file +
rename). A concurrent upload may lose a record to last-writer-wins, which only
means that file is not deduplicated next time — never corruption.

This module deliberately uses only plain ``Path`` I/O, so it can run on a
worker thread (:func:`asyncio.to_thread`) without holding the event loop.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import BinaryIO, Iterator

from deeptutor.services.file_io import atomic_write_json

def sha256_file(path: Path, chunk_size: int = 65536) -> str:
    """sha256 hex digest of ``path``'s bytes (chunked, constant memory)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        _update_from_stream(digest, fh, chunk_size)
    return digest.hexdigest()


def sha256_stream(readable: BinaryIO, chunk_size: int = 65536) -> str:
    """sha256 hex digest of a binary stream's remaining bytes.

    The caller is responsible for seeking to the start first; the stream is
    left at EOF.
    """
    digest = hashlib.sha256()
    _update_from_stream(digest, readable, chunk_size)
    return digest.hexdigest()


def _update_from_stream(digest: "hashlib._Hash", readable: BinaryIO, chunk_size: int) -> None:
    while True:
        block = readable.read(chunk_size)
        if not block:
            break
        digest.update(block)


def is_sha256_hex(value: str) -> bool:
    """Whether ``value`` looks like a client-supplied sha256 digest."""
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def read_file_hashes(kb_dir: Path) -> dict[str, str]:
    """The recorded ``file_hashes`` map (raw-relative path → sha256 hex).

    Returns ``{}`` when the KB has no metadata, no record, or the record is
    corrupt — never raises, so dedupe degrades to "no known hashes" instead of
    failing an upload.
    """
    metadata_file = kb_dir / "metadata.json"
    if not metadata_file.exists():
        return {}
    try:
        data = json.loads(metadata_file.read_text(encoding="utf-8"))
    except Exception:
        return {}
    hashes = data.get("file_hashes") if isinstance(data, dict) else None
    return hashes if isinstance(hashes, dict) else {}


def record_file_hashes(kb_dir: Path, records: dict[str, str]) -> None:
    """Merge ``records`` (raw-relative path → sha256 hex) into the KB's store.

    A no-op for an empty ``records`` dict. Corruption-safe: an unreadable
    metadata file is treated as empty and rewritten.
    """
    if not records:
        return
    metadata_file = kb_dir / "metadata.json"
    try:
        metadata = (
            json.loads(metadata_file.read_text(encoding="utf-8"))
            if metadata_file.exists()
            else {}
        )
    except Exception:
        metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}
    hashes = metadata.get("file_hashes")
    if not isinstance(hashes, dict):
        hashes = metadata["file_hashes"] = {}
    hashes.update(records)
    atomic_write_json(metadata_file, metadata)


def _iter_raw_files(raw_dir: Path) -> Iterator[Path]:
    """Regular files under ``raw_dir`` (recursive, hidden entries skipped).

    Mirrors :func:`deeptutor.knowledge.manifest.iter_kb_documents`: hidden
    entries are editor/OS bookkeeping and never uploadable content.
    """
    if not raw_dir.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(raw_dir):
        dirnames[:] = sorted(name for name in dirnames if not name.startswith("."))
        current = Path(dirpath)
        for filename in sorted(filenames):
            if filename.startswith("."):
                continue
            path = current / filename
            if path.is_file():
                yield path


def ensure_raw_hashes(kb_dir: Path, raw_dir: Path) -> dict[str, str]:
    """Return the raw/ hash map, backfilling + persisting any missing entries.

    Explicit-migration tool — call it only when the files under ``raw/`` are
    known to be successfully indexed (e.g. right after a successful KB
    creation/indexing run). It must NOT be called from the dedupe query path:
    a raw file whose indexing failed would otherwise be recorded, and its
    re-upload would then be wrongly treated as a duplicate.
    """
    recorded = read_file_hashes(kb_dir)
    missing: dict[str, str] = {}
    for path in _iter_raw_files(raw_dir):
        rel = path.relative_to(raw_dir).as_posix()
        if rel in recorded:
            continue
        try:
            missing[rel] = sha256_file(path)
        except OSError:
            continue
    if missing:
        record_file_hashes(kb_dir, missing)
        return {**recorded, **missing}
    return recorded


def dedupe_hash_set(kb_dir: Path) -> set[str]:
    """Set of recorded sha256 hexes whose content is already indexed in the KB.

    This is the authoritative dedupe query for the upload pipeline: a file
    whose digest is in this set is a duplicate and should not be transferred,
    persisted or indexed again.

    Read-only by design — it never backfills, so files that failed indexing
    (and have no record) stay re-uploadable.
    """
    return set(read_file_hashes(kb_dir).values())


__all__ = [
    "dedupe_hash_set",
    "ensure_raw_hashes",
    "is_sha256_hex",
    "read_file_hashes",
    "record_file_hashes",
    "sha256_file",
    "sha256_stream",
]
