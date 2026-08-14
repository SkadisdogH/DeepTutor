"""Add-document tool logic — thin wrapper around the canonical KB adder.

The chat agent downloads files into its workspace but has no way to push them
into a knowledge base (only ``rag``/``kb_files`` are exposed, and those are
read-only). This module exposes ``add_documents_to_kb`` so a new ``add_to_kb``
tool can stage + index downloaded files through the exact same path the
knowledge-center upload UI uses (:func:`deeptutor.knowledge.add_documents`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List


def _validate_paths(file_paths: Iterable[str]) -> List[str]:
    """Return only non-empty, existing, non-URL local file paths."""
    cleaned: List[str] = []
    for raw in file_paths:
        p = str(raw or "").strip()
        if not p:
            continue
        if p.lower().startswith(("http://", "https://", "data:", "ftp://")):
            raise ValueError(
                f"add_to_kb needs a local file path, not a URL: {p!r}. "
                "Download the file to the workspace first, then pass its path."
            )
        path = Path(p)
        if not path.exists():
            raise ValueError(f"File does not exist: {p}")
        if not path.is_file():
            raise ValueError(f"Not a regular file: {p}")
        cleaned.append(p)
    return cleaned


async def add_documents_to_kb(
    kb_name: str,
    file_paths: Iterable[str],
    *,
    allow_duplicates: bool = False,
) -> dict:
    """Stage and index ``file_paths`` into the writable KB ``kb_name``.

    Returns a dict with ``processed_count`` (0 means nothing new to add) and
    ``added_files`` (the staged names). Raises ValueError on invalid input or
    when indexing fails.
    """
    from deeptutor.knowledge.add_documents import add_documents as _adder
    from deeptutor.multi_user.knowledge_access import resolve_kb

    kb_name = (kb_name or "").strip()
    if not kb_name:
        raise ValueError("add_to_kb requires an explicit kb_name.")

    paths = _validate_paths(file_paths)
    if not paths:
        raise ValueError("add_to_kb requires at least one existing file to add.")

    # Resolve the KB (enforces write access; raises 403 for read-only ones).
    resource = resolve_kb(kb_name, require_write=True)

    processed = await _adder(
        kb_name=resource.name,
        source_files=paths,
        base_dir=str(resource.base_dir),
        allow_duplicates=allow_duplicates,
    )
    return {
        "processed_count": int(processed or 0),
        "added_files": [Path(p).name for p in paths],
        "kb_name": resource.name,
    }


__all__ = ["add_documents_to_kb"]
