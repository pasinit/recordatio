"""HTTP-level tests for the new agent endpoints.

Exercises the FastAPI router directly with a stub runner / fresh config
store so we cover both the 200 round-trip and the 409 version-mismatch
path that the improver depends on, without spinning up the full
``server.app`` (which would bind the real Pontifex client).
"""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import anyio
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tpasini.recordatio.agents.config_store import AgentConfig, AgentConfigStore
from tpasini.recordatio.agents.events import TaskEvent, TaskEventKind
from tpasini.recordatio.proxy import agent_endpoints
from tpasini.recordatio.proxy.agent_endpoints import register_agent_routes


class _StubRunner:
    """Records the run-args and yields a canned sequence of events.

    No transport, no briefing hook — the endpoints router doesn't know or
    care about those; it just needs an awaitable async-generator from
    ``run()``.
    """

    def __init__(self, events: list[TaskEvent]) -> None:
        self.events = events
        self.calls: list[dict] = []

    async def run(self, **kwargs: object) -> AsyncIterator[TaskEvent]:
        self.calls.append(kwargs)
        for ev in self.events:
            yield ev


# A constant test credential. The router only checks for *presence* of an
# API key (Recordatio is a credential pass-through, not an identity-verifier),
# so any non-empty header value is accepted. We send it via the standard
# ``Authorization: Bearer …`` header on every test client by default.
_AUTH = {"authorization": "Bearer test-key"}


@pytest.fixture()
def client_and_state(tmp_path: Path) -> tuple[TestClient, AgentConfigStore, _StubRunner]:
    config_store = AgentConfigStore(tmp_path / "configs.db")
    runner = _StubRunner(events=[
        TaskEvent(TaskEventKind.STARTED, {"transport": "subscription"}),
        TaskEvent(TaskEventKind.MESSAGE, {"text": "hello"}),
        TaskEvent(TaskEventKind.COMPLETED, {"stop_reason": "end_turn"}),
    ])
    app = FastAPI()
    register_agent_routes(app, runner=runner, config_store=config_store)
    client = TestClient(app)
    # Inject the test auth header on every request so existing tests
    # don't all need to repeat it. Tests that want to probe the
    # unauthenticated path build their own ``TestClient`` against the
    # same app (see ``test_endpoints_require_auth``).
    client.headers.update(_AUTH)
    return client, config_store, runner


# ---------------------------------------------------------------------------
# GET /v1/agents/{user_id}/{slug}/config
# ---------------------------------------------------------------------------


def test_get_config_404_when_missing(
    client_and_state: tuple[TestClient, AgentConfigStore, _StubRunner],
) -> None:
    client, _, _ = client_and_state
    resp = client.get("/v1/agents/alice/weather/config")
    assert resp.status_code == 404


def test_get_config_roundtrips_after_put(
    client_and_state: tuple[TestClient, AgentConfigStore, _StubRunner],
) -> None:
    client, _, _ = client_and_state
    put_resp = client.put(
        "/v1/agents/alice/weather/config",
        json={"system_prompt": "you are weather", "model": "anthropic/claude-opus-4-7"},
    )
    assert put_resp.status_code == 200
    assert put_resp.json()["version"] == 1

    get_resp = client.get("/v1/agents/alice/weather/config")
    assert get_resp.status_code == 200
    body = get_resp.json()
    assert body["system_prompt"] == "you are weather"
    assert body["model"] == "anthropic/claude-opus-4-7"
    assert body["version"] == 1


# ---------------------------------------------------------------------------
# PUT /v1/agents/{user_id}/{slug}/config — version-mismatch (409)
# ---------------------------------------------------------------------------


def test_put_409_on_stale_version(
    client_and_state: tuple[TestClient, AgentConfigStore, _StubRunner],
) -> None:
    client, _, _ = client_and_state
    client.put("/v1/agents/alice/weather/config", json={"system_prompt": "v1"})
    # First improver — succeeds.
    client.put(
        "/v1/agents/alice/weather/config",
        json={"system_prompt": "v2", "version": 1},
    )
    # Second improver still has version=1 in hand.
    resp = client.put(
        "/v1/agents/alice/weather/config",
        json={"system_prompt": "v2-other", "version": 1},
    )
    assert resp.status_code == 409
    body = resp.json()
    assert body["expected"] == 1
    assert body["actual"] == 2


def test_put_invalid_json_returns_400(
    client_and_state: tuple[TestClient, AgentConfigStore, _StubRunner],
) -> None:
    client, _, _ = client_and_state
    resp = client.put(
        "/v1/agents/alice/weather/config",
        content="not json",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# POST /v1/agents/{user_id}/{slug}/messages
# ---------------------------------------------------------------------------


def test_post_messages_404_when_no_config(
    client_and_state: tuple[TestClient, AgentConfigStore, _StubRunner],
) -> None:
    client, _, _ = client_and_state
    resp = client.post(
        "/v1/agents/alice/weather/messages",
        json={"content": "hi"},
    )
    assert resp.status_code == 404


def test_post_messages_400_when_content_missing(
    client_and_state: tuple[TestClient, AgentConfigStore, _StubRunner],
) -> None:
    client, _, _ = client_and_state
    resp = client.post("/v1/agents/alice/weather/messages", json={})
    assert resp.status_code == 400


def test_post_messages_streams_sse(
    client_and_state: tuple[TestClient, AgentConfigStore, _StubRunner],
) -> None:
    client, config_store, runner = client_and_state
    config_store.put("alice", "weather", AgentConfig(
        system_prompt="you are weather",
        model="anthropic/claude-opus-4-7",
    ))
    resp = client.post(
        "/v1/agents/alice/weather/messages",
        json={"content": "ping"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    body = resp.text
    # All three canned events arrived in order.
    assert "event: started\n" in body
    assert "event: message\n" in body
    assert "event: completed\n" in body
    assert body.index("event: started") < body.index("event: message") < body.index("event: completed")

    # Runner received the user content and the stored config.
    assert len(runner.calls) == 1
    call = runner.calls[0]
    assert call["user_id"] == "alice"
    assert call["slug"] == "weather"
    assert call["content"] == "ping"
    assert call["config"].system_prompt == "you are weather"


def test_post_messages_inline_config_wins_over_store(
    client_and_state: tuple[TestClient, AgentConfigStore, _StubRunner],
) -> None:
    # An inline agent_config is authoritative for this turn — the runner sees
    # it verbatim, not the stored copy (per-query config; no seed drift).
    client, config_store, runner = client_and_state
    config_store.put("alice", "weather", AgentConfig(system_prompt="STALE stored prompt"))
    resp = client.post(
        "/v1/agents/alice/weather/messages",
        json={
            "content": "ping",
            "agent_config": {
                "system_prompt": "LIVE per-query prompt",
                "model": "anthropic/claude-opus-4-8",
                "allowed_tools": ["mcp__telos__web_search", "Read"],
            },
        },
    )
    assert resp.status_code == 200
    call = runner.calls[0]
    assert call["config"].system_prompt == "LIVE per-query prompt"
    assert call["config"].allowed_tools == ["mcp__telos__web_search", "Read"]


def test_post_messages_inline_config_works_without_stored_config(
    client_and_state: tuple[TestClient, AgentConfigStore, _StubRunner],
) -> None:
    # No store row at all — the inline config alone drives the turn (no 404).
    client, _, runner = client_and_state
    resp = client.post(
        "/v1/agents/alice/weather/messages",
        json={
            "content": "ping",
            "agent_config": {"system_prompt": "p", "allowed_tools": []},
        },
    )
    assert resp.status_code == 200
    assert runner.calls[0]["config"].allowed_tools == []


def test_post_messages_inline_config_requires_allowed_tools(
    client_and_state: tuple[TestClient, AgentConfigStore, _StubRunner],
) -> None:
    # Omitting allowed_tools (None) is rejected — we never fall back to the
    # SDK's wide-open default for a caller-supplied config.
    client, _, _ = client_and_state
    resp = client.post(
        "/v1/agents/alice/weather/messages",
        json={"content": "ping", "agent_config": {"system_prompt": "p"}},
    )
    assert resp.status_code == 400
    assert "allowed_tools" in resp.json()["error"]


def test_post_messages_inline_config_must_be_object(
    client_and_state: tuple[TestClient, AgentConfigStore, _StubRunner],
) -> None:
    client, _, _ = client_and_state
    resp = client.post(
        "/v1/agents/alice/weather/messages",
        json={"content": "ping", "agent_config": "not-a-dict"},
    )
    assert resp.status_code == 400


class _SlowRunner(_StubRunner):
    """Stub runner that pauses before each event — longer than the patched
    keepalive interval, so the stream wrapper has silence to fill."""

    def __init__(self, events: list[TaskEvent], gap_s: float) -> None:
        super().__init__(events)
        self._gap_s = gap_s

    async def run(self, **kwargs: object) -> AsyncIterator[TaskEvent]:
        self.calls.append(kwargs)
        for ev in self.events:
            await asyncio.sleep(self._gap_s)
            yield ev


def test_post_messages_emits_keepalive_during_event_silence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no TaskEvent arrives within the keepalive window the stream
    emits SSE comment lines, so per-read client timeouts never fire during
    a turn's silent stretches. Events still arrive, in order, around the
    comments."""
    monkeypatch.setattr(agent_endpoints, "_KEEPALIVE_INTERVAL_S", 0.02)
    config_store = AgentConfigStore(tmp_path / "configs.db")
    config_store.put("alice", "weather", AgentConfig(
        system_prompt="you are weather",
        model="anthropic/claude-opus-4-7",
    ))
    runner = _SlowRunner(
        events=[
            TaskEvent(TaskEventKind.STARTED, {"transport": "subscription"}),
            TaskEvent(TaskEventKind.MESSAGE, {"text": "hello"}),
            TaskEvent(TaskEventKind.COMPLETED, {"stop_reason": "end_turn"}),
        ],
        gap_s=0.1,
    )
    app = FastAPI()
    register_agent_routes(app, runner=runner, config_store=config_store)
    client = TestClient(app)
    client.headers.update(_AUTH)

    resp = client.post("/v1/agents/alice/weather/messages", json={"content": "ping"})

    assert resp.status_code == 200
    body = resp.text
    assert ": keepalive\n\n" in body
    assert "event: started\n" in body
    assert "event: message\n" in body
    assert "event: completed\n" in body
    assert body.index("event: started") < body.index("event: message") < body.index("event: completed")


class _CancelScopeRunner(_StubRunner):
    """Reproduces the production runner's shape: ``run()`` holds an anyio
    task group (a cancel scope) open across its yields — see
    ``AgentRunner.run``'s ``async with anyio.create_task_group()``. anyio
    requires the scope to exit in the task that entered it, so this stub
    breaks any consumer that drives the generator from more than one task."""

    def __init__(self, events: list[TaskEvent], gap_s: float) -> None:
        super().__init__(events)
        self._gap_s = gap_s

    async def run(self, **kwargs: object) -> AsyncIterator[TaskEvent]:
        self.calls.append(kwargs)
        async with anyio.create_task_group():
            for ev in self.events:
                await asyncio.sleep(self._gap_s)
                yield ev


def test_post_messages_keepalive_keeps_cancel_scope_in_one_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: with several keepalive windows elapsing per event, the
    old per-``anext()`` task wrapper entered and exited the runner's cancel
    scope in different tasks — "Attempted to exit cancel scope in a
    different task than it was entered in" killed real long turns (the
    ``sdk_exception`` routine failures of 2026-07-24/27/28/31). The stream
    must deliver the full event sequence, keepalives interleaved."""
    monkeypatch.setattr(agent_endpoints, "_KEEPALIVE_INTERVAL_S", 0.01)
    config_store = AgentConfigStore(tmp_path / "configs.db")
    config_store.put("alice", "weather", AgentConfig(
        system_prompt="you are weather",
        model="anthropic/claude-opus-4-7",
    ))
    runner = _CancelScopeRunner(
        events=[
            TaskEvent(TaskEventKind.STARTED, {"transport": "subscription"}),
            TaskEvent(TaskEventKind.MESSAGE, {"text": "hello"}),
            TaskEvent(TaskEventKind.COMPLETED, {"stop_reason": "end_turn"}),
        ],
        gap_s=0.06,
    )
    app = FastAPI()
    register_agent_routes(app, runner=runner, config_store=config_store)
    client = TestClient(app)
    client.headers.update(_AUTH)

    resp = client.post("/v1/agents/alice/weather/messages", json={"content": "ping"})

    assert resp.status_code == 200
    body = resp.text
    assert ": keepalive\n\n" in body
    assert "event: completed\n" in body
    assert "cancel scope" not in body


# ---------------------------------------------------------------------------
# Auth gate — all three endpoints require an inbound API key
# ---------------------------------------------------------------------------


def test_endpoints_require_auth(tmp_path: Path) -> None:
    """No ``x-api-key`` and no ``Authorization`` header ⇒ 401 on every endpoint."""
    config_store = AgentConfigStore(tmp_path / "configs.db")
    runner = _StubRunner(events=[])
    app = FastAPI()
    register_agent_routes(app, runner=runner, config_store=config_store)
    # Bare client — no _AUTH header injected.
    bare = TestClient(app)

    # Pre-populate a config so a successful POST would otherwise 200 —
    # this guarantees the 401 comes from the auth gate, not the
    # config-not-found check.
    config_store.put("alice", "weather", AgentConfig(
        system_prompt="x",
        model="anthropic/claude-opus-4-7",
    ))

    get_resp = bare.get("/v1/agents/alice/weather/config")
    assert get_resp.status_code == 401
    assert get_resp.json() == {"error": "unauthorized"}

    put_resp = bare.put(
        "/v1/agents/alice/weather/config",
        json={"system_prompt": "x"},
    )
    assert put_resp.status_code == 401
    assert put_resp.json() == {"error": "unauthorized"}

    post_resp = bare.post(
        "/v1/agents/alice/weather/messages",
        json={"content": "hi"},
    )
    assert post_resp.status_code == 401
    assert post_resp.json() == {"error": "unauthorized"}

    # Runner must not have been invoked when auth failed.
    assert runner.calls == []


def test_endpoints_accept_x_api_key_header(tmp_path: Path) -> None:
    """``x-api-key`` header alone (no ``Authorization``) also satisfies the gate."""
    config_store = AgentConfigStore(tmp_path / "configs.db")
    runner = _StubRunner(events=[])
    app = FastAPI()
    register_agent_routes(app, runner=runner, config_store=config_store)
    client = TestClient(app)

    resp = client.get(
        "/v1/agents/alice/weather/config",
        headers={"x-api-key": "test-key"},
    )
    # 404 because no config has been stored — proves we got *past* auth.
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Call-lineage fields — POST body carries them through to the runner
# ---------------------------------------------------------------------------


def test_post_messages_forwards_lineage_fields(
    client_and_state: tuple[TestClient, AgentConfigStore, _StubRunner],
) -> None:
    """``source_agent_id`` and ``parent_request_id`` in the body reach
    ``runner.run`` so the extraction hook can stamp provenance.
    """
    client, config_store, runner = client_and_state
    config_store.put("alice", "weather", AgentConfig(
        system_prompt="you are weather",
        model="anthropic/claude-opus-4-7",
    ))
    resp = client.post(
        "/v1/agents/alice/weather/messages",
        json={
            "content": "ping",
            "source_agent_id": "curia-leader",
            "parent_request_id": "recordatio-abc123",
        },
    )
    assert resp.status_code == 200

    assert len(runner.calls) == 1
    call = runner.calls[0]
    assert call["source_agent_id"] == "curia-leader"
    assert call["parent_request_id"] == "recordatio-abc123"


def test_post_messages_lineage_defaults_to_none(
    client_and_state: tuple[TestClient, AgentConfigStore, _StubRunner],
) -> None:
    """Absent lineage fields ⇒ runner is called with ``None`` (not a string)."""
    client, config_store, runner = client_and_state
    config_store.put("alice", "weather", AgentConfig(
        system_prompt="you are weather",
        model="anthropic/claude-opus-4-7",
    ))
    resp = client.post(
        "/v1/agents/alice/weather/messages",
        json={"content": "ping"},
    )
    assert resp.status_code == 200

    call = runner.calls[0]
    assert call["source_agent_id"] is None
    assert call["parent_request_id"] is None


# ---------------------------------------------------------------------------
# POST /messages — conversation history (``messages`` body field)
# ---------------------------------------------------------------------------


def _put_weather_config(config_store: AgentConfigStore) -> None:
    config_store.put("alice", "weather", AgentConfig(
        system_prompt="you are weather",
        model="anthropic/claude-opus-4-7",
    ))


def test_post_messages_forwards_history_to_runner(
    client_and_state: tuple[TestClient, AgentConfigStore, _StubRunner],
) -> None:
    """The full ``messages`` history off the body must reach ``runner.run``
    verbatim — ignoring it was the per-turn-amnesia P0. ``content`` stays
    the latest user message alongside."""
    client, config_store, runner = client_and_state
    _put_weather_config(config_store)
    history = [
        {"role": "user", "content": "any rain today?"},
        {"role": "assistant", "content": "no rain expected"},
        {"role": "user", "content": "and tomorrow?"},
    ]
    resp = client.post(
        "/v1/agents/alice/weather/messages",
        json={"content": "and tomorrow?", "messages": history},
    )
    assert resp.status_code == 200
    call = runner.calls[0]
    assert call["messages"] == history
    assert call["content"] == "and tomorrow?"


def test_post_messages_accepts_block_list_content_in_history(
    client_and_state: tuple[TestClient, AgentConfigStore, _StubRunner],
) -> None:
    """Image turns carry content as a block list — valid, passed through."""
    client, config_store, runner = client_and_state
    _put_weather_config(config_store)
    history = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,xx"}},
            {"type": "text", "text": "look"},
        ]},
        {"role": "user", "content": "and tomorrow?"},
    ]
    resp = client.post(
        "/v1/agents/alice/weather/messages",
        json={"content": "and tomorrow?", "messages": history},
    )
    assert resp.status_code == 200
    assert runner.calls[0]["messages"] == history


def test_post_messages_without_messages_passes_none(
    client_and_state: tuple[TestClient, AgentConfigStore, _StubRunner],
) -> None:
    client, config_store, runner = client_and_state
    _put_weather_config(config_store)
    resp = client.post(
        "/v1/agents/alice/weather/messages",
        json={"content": "ping"},
    )
    assert resp.status_code == 200
    assert runner.calls[0]["messages"] is None


@pytest.mark.parametrize(
    "bad_messages",
    [
        "not a list",
        [{"role": "user", "content": "ok"}, "garbage"],
        [{"content": "missing role"}],
        [{"role": "", "content": "empty role"}],
        [{"role": "user", "content": 42}],
        [{"role": "user", "content": ["block must be a dict"]}],
    ],
    ids=["not-list", "non-dict-item", "missing-role", "empty-role",
         "non-str-content", "non-dict-block"],
)
def test_post_messages_400_on_malformed_messages(
    client_and_state: tuple[TestClient, AgentConfigStore, _StubRunner],
    bad_messages: object,
) -> None:
    """``messages`` is optional — but present-and-malformed must 400 (a
    caller that sends history deserves to learn it was garbage rather
    than silently losing memory) and never reach the runner."""
    client, config_store, runner = client_and_state
    _put_weather_config(config_store)
    resp = client.post(
        "/v1/agents/alice/weather/messages",
        json={"content": "ping", "messages": bad_messages},
    )
    assert resp.status_code == 400
    assert runner.calls == []
