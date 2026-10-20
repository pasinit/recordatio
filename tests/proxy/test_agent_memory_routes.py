"""Briefing preview + transcript ingest (proxy/agent_memory_routes.py).

``GET /v1/users/{user_id}/briefing`` must return exactly what the agent
briefing hook would inject, without the hook's writes (briefing-events row,
``retrieval_count`` bumps). ``POST /v1/agents/{user_id}/{slug}/transcripts``
must run the hook's extraction once per ``idempotency_key``.
"""

import asyncio
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from tpasini.recordatio.entity import Entity
from tpasini.recordatio.models import KnowledgeType, KnowledgeUnit
from tpasini.recordatio.proxy import retrieval
from tpasini.recordatio.proxy.agent_memory_routes import register_agent_memory_routes
from tpasini.recordatio.proxy.briefing_log import BriefingLog
from tpasini.recordatio.proxy.transcript_log import TranscriptLog
from tpasini.recordatio.storage.base import SQLiteStore

_USER = "sam"
_SLUG = "assistant"
_AGENT = f"{_USER}/{_SLUG}"
_QUERY = "where does Sam work?"
_AUTH = {"x-api-key": "local-key"}


def _unit(uid: str, content: str) -> KnowledgeUnit:
    now = datetime.now(UTC)
    return KnowledgeUnit(
        id=uid, type=KnowledgeType.SEMANTIC, content=content,
        asserted_at=now, valid_from=now, provenance="test",
    )


def _transcript_body(key: str, **overrides: str) -> dict[str, str]:
    body = {
        "input": "Remind me where the Riverside offsite is.",
        "output": "The Riverside offsite is at the Globex campus on Friday.",
        "occurred_at": "2026-10-19T08:00:00+00:00",
        "source": "routine",
        "idempotency_key": key,
    }
    body.update(overrides)
    return body


async def _request(app: FastAPI, method: str, path: str, **kwargs: Any) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://recordatio") as client:
        return await client.request(method, path, **kwargs)


# ---------------------------------------------------------------------------
# Wired into the real server: briefing parity and no-writes
# ---------------------------------------------------------------------------


@pytest.fixture()
def server_mod(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The server module pointed at a fresh seeded store and briefing log."""
    monkeypatch.setenv("RECORDATIO_EXTRACTION_MODEL", "test")
    monkeypatch.setenv("AGENTMEM_DB", str(tmp_path / "memories.db"))
    from tpasini.recordatio.proxy import server as server_mod

    store = SQLiteStore(tmp_path / "seeded.db")
    # Seeded as the agent's own extraction would write them: the hook retrieves
    # with source_agent_id = the agent's composite id.
    store.add(_unit("u-work", "Sam works at Globex in Riverside"), _AGENT, _AGENT)
    store.add(_unit("u-commute", "Sam cycles to the Globex office"), _AGENT, _AGENT)
    store.add(_unit("u-other", "Morgan works at Initech"), "morgan/assistant", "morgan/assistant")
    store.add_entity(Entity(id="e-globex", label="Globex", asserted_at=datetime.now(UTC)), _AGENT, _AGENT)
    monkeypatch.setattr(server_mod, "store", store)
    monkeypatch.setattr(server_mod, "briefing_log", BriefingLog(tmp_path / "briefing.db"))

    async def no_wikidata(*_: Any, **__: Any) -> list[Entity]:
        return []

    monkeypatch.setattr(retrieval, "wikidata_seed_entities", no_wikidata)
    return server_mod


async def _await_background(server_mod) -> None:
    pending = [t for t in server_mod._background_tasks if not t.done()]
    if pending:
        await asyncio.gather(*pending)


def _write_counts(server_mod) -> tuple[int, int]:
    """(briefing-event rows, total retrieval_count across units/entities/relations)."""
    events = server_mod.briefing_log._conn.execute("SELECT COUNT(*) FROM briefing_events").fetchone()[0]
    conn = server_mod.store._conn
    bumps = sum(
        conn.execute(f"SELECT COALESCE(SUM(retrieval_count), 0) FROM {table}").fetchone()[0]
        for table in ("knowledge_units", "entities", "relations")
    )
    return events, bumps


async def test_briefing_equals_what_the_hook_injects(server_mod) -> None:
    response = await _request(
        server_mod.app, "GET", f"/v1/users/{_USER}/briefing",
        params={"slug": _SLUG, "query": _QUERY}, headers=_AUTH,
    )
    assert response.status_code == 200
    preview = response.json()

    hook_text, _panel = await server_mod._agent_briefing_hook(_USER, _SLUG, _QUERY, "base system prompt")
    await _await_background(server_mod)

    assert hook_text is not None
    assert "Sam works at Globex in Riverside" in hook_text
    assert "Morgan" not in hook_text  # another agent's scope never leaks in
    assert preview["briefing"] == hook_text
    built = await server_mod._agent_briefing(_USER, _SLUG, _QUERY, "", call_id=None)
    assert built.selected_ids is not None
    assert preview["units"] == len(built.selected_ids.units) > 0


async def test_briefing_performs_no_writes(server_mod) -> None:
    before = _write_counts(server_mod)
    response = await _request(
        server_mod.app, "GET", f"/v1/users/{_USER}/briefing",
        params={"slug": _SLUG, "query": _QUERY}, headers=_AUTH,
    )
    await _await_background(server_mod)
    assert response.status_code == 200
    assert response.json()["units"] > 0
    assert _write_counts(server_mod) == before

    # Positive control: the hook itself does write both, so the counters above
    # would have moved had the preview written anything.
    await server_mod._agent_briefing_hook(_USER, _SLUG, _QUERY, "base system prompt")
    await _await_background(server_mod)
    events, bumps = _write_counts(server_mod)
    assert events == before[0] + 1
    assert bumps > before[1]


async def test_briefing_for_an_empty_scope_is_empty(server_mod) -> None:
    response = await _request(
        server_mod.app, "GET", "/v1/users/nobody/briefing",
        params={"slug": _SLUG, "query": _QUERY}, headers=_AUTH,
    )
    assert response.status_code == 200
    assert response.json() == {"briefing": "", "units": 0}


@pytest.mark.parametrize(
    ("params", "headers", "status"),
    [
        ({"slug": _SLUG, "query": _QUERY}, {}, 401),
        ({"query": _QUERY}, _AUTH, 400),
        ({"slug": _SLUG}, _AUTH, 400),
        ({"slug": "_user", "query": _QUERY}, _AUTH, 400),
    ],
)
async def test_briefing_rejects_bad_requests(server_mod, params, headers, status) -> None:
    response = await _request(server_mod.app, "GET", f"/v1/users/{_USER}/briefing", params=params, headers=headers)
    assert response.status_code == status


async def test_transcript_runs_the_agent_extraction_hook(server_mod, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[dict[str, Any]] = []

    async def spy_extract_and_store(turn, api_key, agent_id, user_id=None, parent_request_id=None,
                                    source_agent_id=None, **_: Any) -> dict:
        captured.append({
            "turn": turn, "api_key": api_key, "agent_id": agent_id,
            "user_id": user_id, "source_agent_id": source_agent_id,
        })
        return {}

    monkeypatch.setattr(server_mod, "_extract_and_store", spy_extract_and_store)
    key = f"run-{uuid.uuid4().hex}"
    path = f"/v1/agents/{_USER}/{_SLUG}/transcripts"

    first = await _request(server_mod.app, "POST", path, json=_transcript_body(key), headers=_AUTH)
    await _await_background(server_mod)
    again = await _request(server_mod.app, "POST", path, json=_transcript_body(key), headers=_AUTH)
    await _await_background(server_mod)

    assert first.status_code == 202
    assert first.json() == {"idempotency_key": key, "duplicate": False}
    assert again.status_code == 200
    assert again.json() == {"idempotency_key": key, "duplicate": True}
    assert len(captured) == 1
    call = captured[0]
    assert call["agent_id"] == _AGENT
    assert call["user_id"] == _USER
    assert call["source_agent_id"] == _AGENT
    assert call["api_key"] is None
    assert call["turn"].messages == [{"role": "user", "content": "Remind me where the Riverside offsite is."}]
    assert call["turn"].response_text == "The Riverside offsite is at the Globex campus on Friday."


# ---------------------------------------------------------------------------
# Route-level: idempotency and validation against stub callables
# ---------------------------------------------------------------------------


class _Recorder:
    def __init__(self, fail: bool = False) -> None:
        self.calls: list[tuple[str, str, dict]] = []
        self.fail = fail

    async def __call__(self, user_id: str, slug: str, transcript: dict) -> None:
        if self.fail:
            raise RuntimeError("could not schedule extraction")
        self.calls.append((user_id, slug, transcript))


def _routes_app(log: TranscriptLog, extract: _Recorder) -> FastAPI:
    async def preview(user_id: str, slug: str, query: str) -> tuple[str | None, int]:
        return None, 0

    app = FastAPI()
    register_agent_memory_routes(app, preview_briefing=preview, extract_transcript=extract, transcript_log=log)
    return app


@pytest.fixture()
def transcript_log(tmp_path: Path) -> TranscriptLog:
    return TranscriptLog(tmp_path / "transcripts.db")


async def test_transcript_idempotency_is_scoped_per_agent(transcript_log) -> None:
    extract = _Recorder()
    app = _routes_app(transcript_log, extract)

    first = await _request(app, "POST", "/v1/agents/sam/assistant/transcripts",
                           json=_transcript_body("k1"), headers=_AUTH)
    repeat = await _request(app, "POST", "/v1/agents/sam/assistant/transcripts",
                            json=_transcript_body("k1", output="a different retry body"), headers=_AUTH)
    other_agent = await _request(app, "POST", "/v1/agents/sam/planner/transcripts",
                                 json=_transcript_body("k1"), headers=_AUTH)

    assert [first.status_code, repeat.status_code, other_agent.status_code] == [202, 200, 202]
    assert repeat.json()["duplicate"] is True
    assert [(user, slug) for user, slug, _ in extract.calls] == [("sam", "assistant"), ("sam", "planner")]
    assert transcript_log.count() == 2


async def test_transcript_key_survives_a_restart(tmp_path: Path) -> None:
    extract = _Recorder()
    first = await _request(_routes_app(TranscriptLog(tmp_path / "t.db"), extract), "POST",
                           "/v1/agents/sam/assistant/transcripts", json=_transcript_body("k1"), headers=_AUTH)
    repeat = await _request(_routes_app(TranscriptLog(tmp_path / "t.db"), extract), "POST",
                            "/v1/agents/sam/assistant/transcripts", json=_transcript_body("k1"), headers=_AUTH)
    assert (first.status_code, repeat.status_code) == (202, 200)
    assert len(extract.calls) == 1


async def test_transcript_key_is_released_when_extraction_cannot_be_scheduled(transcript_log) -> None:
    with pytest.raises(RuntimeError):
        await _request(_routes_app(transcript_log, _Recorder(fail=True)), "POST",
                       "/v1/agents/sam/assistant/transcripts", json=_transcript_body("k1"), headers=_AUTH)
    assert transcript_log.count() == 0

    retry = await _request(_routes_app(transcript_log, _Recorder()), "POST",
                           "/v1/agents/sam/assistant/transcripts", json=_transcript_body("k1"), headers=_AUTH)
    assert retry.status_code == 202


@pytest.mark.parametrize(
    ("body", "headers", "status"),
    [
        (_transcript_body("k1"), {}, 401),
        ([], _AUTH, 400),
        ({k: v for k, v in _transcript_body("k1").items() if k != "source"}, _AUTH, 400),
        (_transcript_body("k1", idempotency_key=""), _AUTH, 400),
        (_transcript_body("k1", occurred_at="last Tuesday"), _AUTH, 400),
        (_transcript_body("k1", input="", output="  "), _AUTH, 400),
        ({**_transcript_body("k1"), "output": 42}, _AUTH, 400),
    ],
)
async def test_transcript_rejects_bad_requests(transcript_log, body, headers, status) -> None:
    extract = _Recorder()
    response = await _request(_routes_app(transcript_log, extract), "POST",
                              "/v1/agents/sam/assistant/transcripts", json=body, headers=headers)
    assert response.status_code == status
    assert extract.calls == []
    assert transcript_log.count() == 0


async def test_transcript_rejects_the_reserved_user_slug(transcript_log) -> None:
    response = await _request(_routes_app(transcript_log, _Recorder()), "POST",
                              "/v1/agents/sam/_user/transcripts", json=_transcript_body("k1"), headers=_AUTH)
    assert response.status_code == 400
