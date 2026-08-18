"""Memoization guards for knowledge-base index probes (CPU-idle regression).

Motivation: ``inspect_kb_versions`` re-parses every provider index artifact on
the event loop. The Math KB's 88MB docstore.json costs ~0.5s per parse and the
knowledge page polls ``/list`` every 4s while it *thinks* work is active — so
without memoization a stale progress snapshot turns into a self-sustaining
~40% single-core idle burn and 10s+ API queueing (see docs/PITFALLS.md
「知识页面轮询 × inspect_kb_versions 全量解析 = CPU 空转回路」).

These tests pin the memo contract:
- unchanged files → exactly one probe, repeat calls hit the cache
  (provider names are normalized, so ``None`` and ``"llamaindex"`` share a key);
- any real file change (content write bumps mtime/size) → cache invalidates;
- provider is part of the cache key;
- returned lists are shallow copies (callers cannot poison the cache);
- ``provider_failure_summary(..., versions=...)`` never re-probes;
- ``clear_kb_inspect_cache`` is the explicit invalidation seam.
"""

from __future__ import annotations

import json

import pytest

from deeptutor.services.rag import index_probe
from deeptutor.services.rag.index_probe import (
    clear_kb_inspect_cache,
    inspect_kb_versions,
    provider_failure_summary,
)


def _make_flat_kb(root, doc_nodes: int = 2):
    """Build a minimal flat-layout KB: version-1 with a llama_index store."""
    kb_dir = root / "kb"
    ver = kb_dir / "version-1"
    ver.mkdir(parents=True)
    (ver / "meta.json").write_text(
        json.dumps({"provider": "llamaindex", "signature": "sig1"}), encoding="utf-8"
    )
    data = {
        "docstore/ref_doc_info": {"ref1": {"doc_hash": "x"}},
        "docstore/data": {f"n{i}": {"text": "hello"} for i in range(doc_nodes)},
    }
    (ver / "docstore.json").write_text(json.dumps(data), encoding="utf-8")
    (ver / "index_store.json").write_text("{}", encoding="utf-8")
    return kb_dir


@pytest.fixture(autouse=True)
def _clean_cache():
    clear_kb_inspect_cache()
    yield
    clear_kb_inspect_cache()


def _probe_counter(monkeypatch):
    calls = {"n": 0}
    real = index_probe.inspect_provider_version

    def counting(entry, provider):
        calls["n"] += 1
        return real(entry, provider)

    monkeypatch.setattr(index_probe, "inspect_provider_version", counting)
    return calls


def test_memoizes_unchanged_files_single_probe(tmp_path, monkeypatch):
    kb = _make_flat_kb(tmp_path, doc_nodes=2)
    calls = _probe_counter(monkeypatch)

    first = inspect_kb_versions(kb, None)
    second = inspect_kb_versions(kb, None)
    # Provider name is normalized: "llamaindex" hits the None-key entry.
    third = inspect_kb_versions(kb, "llamaindex")

    assert calls["n"] == 1
    assert first == second == third
    assert first[0]["version"] == "version-1"
    assert first[0]["ready"] is True
    assert first[0]["doc_count"] == 2


def test_cache_invalidates_on_file_change(tmp_path, monkeypatch):
    kb = _make_flat_kb(tmp_path, doc_nodes=2)
    calls = _probe_counter(monkeypatch)

    inspect_kb_versions(kb, None)
    assert calls["n"] == 1

    # Rewrite the docstore with different content: mtime_ns + size change.
    (kb / "version-1" / "docstore.json").write_text(
        json.dumps(
            {
                "docstore/ref_doc_info": {"ref1": {"doc_hash": "x"}},
                "docstore/data": {"n0": {"text": "a"}, "n1": {"text": "b"}, "n2": {"text": "c"}},
            }
        ),
        encoding="utf-8",
    )
    refreshed = inspect_kb_versions(kb, None)
    assert calls["n"] == 2
    assert refreshed[0]["doc_count"] == 3


def test_cache_key_includes_provider(tmp_path, monkeypatch):
    kb = _make_flat_kb(tmp_path)
    calls = _probe_counter(monkeypatch)

    inspect_kb_versions(kb, None)
    assert calls["n"] == 1
    # Different provider (normalized) → different key → fresh probe.
    inspect_kb_versions(kb, "pageindex")
    assert calls["n"] == 2


def test_returned_list_is_a_copy_callers_cannot_poison_cache(tmp_path, monkeypatch):
    kb = _make_flat_kb(tmp_path)
    _probe_counter(monkeypatch)

    first = inspect_kb_versions(kb, None)
    first[0]["ready"] = False
    first[0]["injected"] = "garbage"
    first.append({"version": "fake"})

    second = inspect_kb_versions(kb, None)
    assert second[0]["ready"] is True
    assert "injected" not in second[0]
    assert len(second) == 1


def test_provider_failure_summary_reuses_versions_no_reprobe(tmp_path, monkeypatch):
    kb = _make_flat_kb(tmp_path)
    versions = inspect_kb_versions(kb, None)

    def boom(*args, **kwargs):
        raise AssertionError("provider_failure_summary must not re-probe")

    monkeypatch.setattr(index_probe, "inspect_kb_versions", boom)
    summary = provider_failure_summary(kb, None, versions=versions)
    # Fresh, ready store has no failure summary.
    assert summary == ""


def test_provider_failure_summary_without_versions_still_probes(tmp_path, monkeypatch):
    kb = _make_flat_kb(tmp_path)
    calls = _probe_counter(monkeypatch)

    provider_failure_summary(kb, None)
    assert calls["n"] == 1


def test_clear_kb_inspect_cache_forces_recompute(tmp_path, monkeypatch):
    kb = _make_flat_kb(tmp_path)
    calls = _probe_counter(monkeypatch)

    inspect_kb_versions(kb, None)
    assert calls["n"] == 1
    clear_kb_inspect_cache(kb)
    inspect_kb_versions(kb, None)
    assert calls["n"] == 2

    # Clearing one KB leaves another KB's entry alone.
    other = _make_flat_kb(tmp_path / "other")
    inspect_kb_versions(other, None)
    clear_kb_inspect_cache(kb)
    inspect_kb_versions(other, None)
    assert calls["n"] == 3  # 'other' still memoized → no new probe