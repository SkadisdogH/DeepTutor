"""Tests for the temporary (side-chat scratch) session flag.

Covers two pieces:
1. ``_filter_temporary_sessions`` — the listing filter that keeps scratch
   conversations out of the normal session history.
2. ``TurnRuntimeManager.start_turn`` — writing ``preferences.temporary`` when
   the client sends ``temporary: true``, so the flag sticks for the life of
   the session and every listing endpoint can filter it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from deeptutor.api.routers.sessions import _filter_temporary_sessions
from deeptutor.core.stream import StreamEvent, StreamEventType
from deeptutor.services.session.sqlite_store import SQLiteSessionStore
from deeptutor.services.session.turn_runtime import TurnRuntimeManager


def _session(preferences: dict | None = None) -> dict:
    return {
        "id": "s-1",
        "session_id": "s-1",
        "title": "T",
        "preferences": preferences or {},
    }


# ---------------------------------------------------------------------------
# Listing filter
# ---------------------------------------------------------------------------


def test_keeps_normal_sessions():
    sessions = [
        _session(),
        _session({"capability": "chat"}),
        _session({"tool_setting": "x"}),
    ]
    result = _filter_temporary_sessions(sessions, include_temporary=False)
    assert len(result) == 3


def test_drops_temporary_sessions_by_default():
    sessions = [
        _session(),
        _session({"temporary": True}),
        _session({"temporary": True, "capability": "chat"}),
    ]
    result = _filter_temporary_sessions(sessions, include_temporary=False)
    assert [s["id"] for s in result] == ["s-1"]


def test_falsey_temporary_flag_is_kept():
    sessions = [
        _session({"temporary": False}),
        _session({"temporary": 0}),
        _session({}),
    ]
    result = _filter_temporary_sessions(sessions, include_temporary=False)
    assert len(result) == 3


def test_truthy_non_bool_temporary_flag_is_dropped():
    sessions = [_session(), _session({"temporary": "yes"}), _session({"temporary": 1})]
    result = _filter_temporary_sessions(sessions, include_temporary=False)
    assert [s["id"] for s in result] == ["s-1"]


def test_include_temporary_returns_everything():
    sessions = [_session(), _session({"temporary": True})]
    result = _filter_temporary_sessions(sessions, include_temporary=True)
    assert len(result) == 2


def test_handles_missing_preferences_blob():
    sessions = [
        {"id": "no-prefs", "session_id": "no-prefs", "title": "T"},
        {"id": "null-prefs", "session_id": "null-prefs", "title": "T", "preferences": None},
    ]
    result = _filter_temporary_sessions(sessions, include_temporary=False)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# start_turn writes the flag into session preferences
# ---------------------------------------------------------------------------


def _fake_skill_service() -> SimpleNamespace:
    return SimpleNamespace(
        summary_entries=lambda: [],
        load_always_for_context=lambda: "",
        load_for_context=lambda _skills: "",
        list_skills=lambda: [],
    )


def _fake_persona_service() -> SimpleNamespace:
    return SimpleNamespace(
        load_for_context=lambda name: (
            f"## Active Persona\n### Persona: {name}\n\nbody" if name else ""
        )
    )


async def _noop_async(*_args, **_kwargs):
    return None


def _install_start_turn_mocks(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    captured: dict[str, object] = {}

    class FakeContextBuilder:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        async def build(self, **kwargs):
            return SimpleNamespace(
                conversation_history=[],
                conversation_summary="",
                context_text="",
                token_count=0,
                budget=0,
            )

    class FakeOrchestrator:
        async def handle(self, context):
            yield StreamEvent(
                type=StreamEventType.CONTENT,
                source="chat",
                stage="responding",
                content="Scratch reply",
                metadata={"call_kind": "llm_final_response"},
            )
            yield StreamEvent(type=StreamEventType.DONE, source="chat")

    monkeypatch.setattr("deeptutor.services.llm.config.get_llm_config", lambda: SimpleNamespace())
    monkeypatch.setattr(
        "deeptutor.services.session.context_builder.ContextBuilder", FakeContextBuilder
    )
    monkeypatch.setattr("deeptutor.runtime.orchestrator.ChatOrchestrator", FakeOrchestrator)
    monkeypatch.setattr(
        "deeptutor.book.context.build_book_context",
        lambda *_args, **_kwargs: SimpleNamespace(text="", references=[], warnings=[]),
    )
    monkeypatch.setattr(
        "deeptutor.services.memory.get_memory_store",
        lambda: SimpleNamespace(
            read_l3_concat=lambda: "",
            emit=_noop_async,
        ),
    )
    monkeypatch.setattr(
        "deeptutor.services.skill.get_skill_service",
        _fake_skill_service,
    )
    monkeypatch.setattr(
        "deeptutor.services.persona.get_persona_service",
        _fake_persona_service,
    )
    return captured


def _base_payload(**overrides) -> dict:
    payload = {
        "type": "start_turn",
        "content": "what does this mean?",
        "session_id": None,
        "capability": "chat",
        "tools": [],
        "knowledge_bases": [],
        "attachments": [],
        "language": "en",
        "config": {},
    }
    payload.update(overrides)
    return payload


@pytest.mark.asyncio
async def test_start_turn_persists_temporary_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _install_start_turn_mocks(monkeypatch)
    store = SQLiteSessionStore(tmp_path / "chat_history.db")
    runtime = TurnRuntimeManager(store)

    session, turn = await runtime.start_turn(
        _base_payload(temporary=True, history_references=["main-session-1"])
    )
    # Drain the turn so the persisted row is terminal before the second
    # start_turn (a live "running" turn blocks the next one).
    async for _ in runtime.subscribe_turn(turn["id"], after_seq=0):
        pass

    persisted = await store.get_session(session["id"])
    assert persisted is not None
    assert persisted["preferences"]["temporary"] is True

    # The flag keeps being persisted on every turn of the same session, so a
    # reconnection that re-sends ``temporary: true`` cannot un-flag it.
    session2, turn2 = await runtime.start_turn(
        _base_payload(
            session_id=session["id"],
            temporary=True,
            history_references=["main-session-1"],
        )
    )
    async for _ in runtime.subscribe_turn(turn2["id"], after_seq=0):
        pass
    assert session2["id"] == session["id"]
    persisted2 = await store.get_session(session["id"])
    assert persisted2["preferences"]["temporary"] is True

    # A normal (non-temporary) session stays unflagged.
    normal, turn3 = await runtime.start_turn(_base_payload(content="hello"))
    async for _ in runtime.subscribe_turn(turn3["id"], after_seq=0):
        pass
    normal_persisted = await store.get_session(normal["id"])
    assert normal_persisted["preferences"].get("temporary") is None