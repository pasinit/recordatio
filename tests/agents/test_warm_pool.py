"""Warm session pool tests: pool hygiene + the transport's warm/cold split.

The pool's contract (one spare per (user, slug, config-version), one turn
per session, TTL + capacity hygiene, dead-spare fallback) is exercised
against fakes — no SDK install or subprocess needed. The transport tests
prove the streaming pump, rate-limit recording, terminal-event guarantee
and disconnect-in-finally hold identically on both paths.
"""

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest

from tests.agents.test_runner import (
    _fake_options_cls,
    _make_class,
    _RecordingSDK,
    _success_result_message,
)
from tpasini.recordatio.agents.config_store import AgentConfig
from tpasini.recordatio.agents.events import TaskEvent, TaskEventKind
from tpasini.recordatio.agents.rate_limit_state import RateLimitStateStore
from tpasini.recordatio.agents.runner import SubscriptionTransport
from tpasini.recordatio.agents.warm_pool import WarmPool

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeClient:
    """Stands in for ``ClaudeSDKClient``: connect/query/receive/disconnect."""

    def __init__(
        self,
        messages: list[object] | None = None,
        *,
        fail_on_query: bool = False,
        options: Any = None,
    ) -> None:
        self.messages = messages if messages is not None else [_success_result_message()]
        self.fail_on_query = fail_on_query
        self.options = options
        self.connect_calls = 0
        self.query_prompts: list[str] = []
        self.disconnected = False

    async def connect(self) -> None:
        self.connect_calls += 1

    async def query(self, prompt: str) -> None:
        if self.fail_on_query:
            raise RuntimeError("write to dead subprocess")
        self.query_prompts.append(prompt)

    async def receive_response(self) -> AsyncIterator[object]:
        for msg in self.messages:
            yield msg

    async def disconnect(self) -> None:
        self.disconnected = True

    def mark_subprocess_dead(self) -> None:
        """Mimic the SDK internals the pool's liveness probe inspects."""
        self._transport = SimpleNamespace(_process=SimpleNamespace(returncode=1))


def _connector(client: _FakeClient):
    async def _connect() -> _FakeClient:
        await client.connect()
        return client

    return _connect


async def _warm(
    pool: WarmPool, user: str, slug: str, version: int, client: _FakeClient,
    *, config: AgentConfig | None = None,
) -> None:
    fingerprint = (
        SubscriptionTransport._session_fingerprint(config, "BASE", config.model or None)
        if config is not None else None
    )
    task = pool.schedule_warm(user, slug, version, _connector(client), fingerprint=fingerprint)
    assert task is not None
    await task


@pytest.fixture()
def rate_state(tmp_path) -> RateLimitStateStore:
    return RateLimitStateStore(tmp_path / "rate.db")


# ---------------------------------------------------------------------------
# WarmPool unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_take_on_empty_pool_returns_none() -> None:
    pool = WarmPool(min_age_s=0, max_spares=4, ttl_s=600)
    assert await pool.take("alice", "weather", 1) is None


@pytest.mark.asyncio
async def test_warm_then_take_is_one_shot() -> None:
    """One turn per session: ``take`` removes the spare permanently —
    a second take for the same key must go cold."""
    pool = WarmPool(min_age_s=0, max_spares=4, ttl_s=600)
    client = _FakeClient()
    await _warm(pool, "alice", "weather", 1, client)
    assert client.connect_calls == 1  # connect paid at warm time

    assert await pool.take("alice", "weather", 1) is client
    assert await pool.take("alice", "weather", 1) is None
    await pool.close_all()


@pytest.mark.asyncio
async def test_schedule_warm_dedupes_per_key() -> None:
    pool = WarmPool(min_age_s=0, max_spares=4, ttl_s=600)
    await _warm(pool, "alice", "weather", 1, _FakeClient())
    assert pool.schedule_warm("alice", "weather", 1, _connector(_FakeClient())) is None
    assert pool.size == 1
    await pool.close_all()


@pytest.mark.asyncio
async def test_config_version_change_invalidates_spare_at_take() -> None:
    """A spare's session config is sealed at connect — a config-version
    bump makes it useless. take() at the new version must miss AND
    disconnect the stale spare."""
    pool = WarmPool(min_age_s=0, max_spares=4, ttl_s=600)
    v1_client = _FakeClient()
    await _warm(pool, "alice", "weather", 1, v1_client)

    assert await pool.take("alice", "weather", 2) is None
    await pool.wait_idle()
    assert v1_client.disconnected is True
    assert pool.size == 0
    await pool.close_all()


@pytest.mark.asyncio
async def test_schedule_warm_supersedes_stale_version() -> None:
    pool = WarmPool(min_age_s=0, max_spares=4, ttl_s=600)
    v1_client = _FakeClient()
    await _warm(pool, "alice", "weather", 1, v1_client)
    await _warm(pool, "alice", "weather", 2, _FakeClient())
    await pool.wait_idle()
    assert v1_client.disconnected is True
    assert pool.spare_keys() == [("alice", "weather", 2)]
    await pool.close_all()


@pytest.mark.asyncio
async def test_expired_spare_is_not_handed_out() -> None:
    now = [0.0]
    pool = WarmPool(min_age_s=0, max_spares=4, ttl_s=10, clock=lambda: now[0])
    client = _FakeClient()
    await _warm(pool, "alice", "weather", 1, client)

    now[0] = 11.0  # past the TTL
    assert await pool.take("alice", "weather", 1) is None
    await pool.wait_idle()
    assert client.disconnected is True
    await pool.close_all()


@pytest.mark.asyncio
async def test_ttl_reaper_disconnects_expired_spares() -> None:
    """The background reaper — not just take() — must terminate expired
    subprocesses, otherwise idle agents would hold CLIs for ever."""
    pool = WarmPool(min_age_s=0, max_spares=4, ttl_s=0.05, reap_interval_s=0.02)
    client = _FakeClient()
    await _warm(pool, "alice", "weather", 1, client)
    assert pool.size == 1

    for _ in range(50):  # ~1s ceiling; normally done in <0.1s
        await asyncio.sleep(0.02)
        if client.disconnected:
            break
    assert client.disconnected is True
    assert pool.size == 0
    await pool.close_all()


@pytest.mark.asyncio
async def test_capacity_evicts_oldest_spare() -> None:
    now = [0.0]
    pool = WarmPool(min_age_s=0, max_spares=2, ttl_s=600, clock=lambda: now[0])
    oldest = _FakeClient()
    await _warm(pool, "alice", "weather", 1, oldest)
    now[0] = 1.0
    await _warm(pool, "bob", "news", 1, _FakeClient())
    now[0] = 2.0
    await _warm(pool, "carol", "mail", 1, _FakeClient())
    await pool.wait_idle()

    assert pool.size == 2
    assert ("alice", "weather", 1) not in pool.spare_keys()
    assert oldest.disconnected is True
    await pool.close_all()


@pytest.mark.asyncio
async def test_disabled_pool_never_warms() -> None:
    pool = WarmPool(min_age_s=0, max_spares=0, ttl_s=600)
    assert pool.schedule_warm("alice", "weather", 1, _connector(_FakeClient())) is None
    assert pool.size == 0


@pytest.mark.asyncio
async def test_dead_spare_is_not_handed_out() -> None:
    pool = WarmPool(min_age_s=0, max_spares=4, ttl_s=600)
    client = _FakeClient()
    await _warm(pool, "alice", "weather", 1, client)
    client.mark_subprocess_dead()

    assert await pool.take("alice", "weather", 1) is None
    await pool.wait_idle()
    assert client.disconnected is True
    await pool.close_all()


@pytest.mark.asyncio
async def test_close_all_disconnects_everything_and_blocks_new_warms() -> None:
    pool = WarmPool(min_age_s=0, max_spares=4, ttl_s=600)
    clients = [_FakeClient(), _FakeClient()]
    await _warm(pool, "alice", "weather", 1, clients[0])
    await _warm(pool, "bob", "news", 1, clients[1])

    await pool.close_all()
    assert all(c.disconnected for c in clients)
    assert pool.size == 0
    assert pool.schedule_warm("alice", "weather", 1, _connector(_FakeClient())) is None


@pytest.mark.asyncio
async def test_failed_warm_connect_is_contained() -> None:
    pool = WarmPool(min_age_s=0, max_spares=4, ttl_s=600)

    async def boom() -> Any:
        raise RuntimeError("CLI failed to spawn")

    task = pool.schedule_warm("alice", "weather", 1, boom)
    assert task is not None
    await task  # must not raise
    assert pool.size == 0
    # The key is released — a retry is allowed.
    assert pool.schedule_warm("alice", "weather", 1, _connector(_FakeClient())) is not None
    await pool.wait_idle()
    assert pool.size == 1
    await pool.close_all()


# ---------------------------------------------------------------------------
# SubscriptionTransport: warm/cold split
# ---------------------------------------------------------------------------


def _exploding_query_fn(**_: Any) -> AsyncIterator[object]:
    raise AssertionError("cold path must not be used on a warm hit")
    yield  # pragma: no cover


async def _run_turn(
    transport: SubscriptionTransport,
    *,
    config: AgentConfig,
    briefing: str | None = None,
) -> list[TaskEvent]:
    events: list[TaskEvent] = []
    async for ev in transport.run(
        user_id="alice", slug="weather", content="hi",
        system_prompt="BASE", config=config, briefing=briefing,
    ):
        events.append(ev)
    return events


@pytest.mark.asyncio
async def test_warm_hit_skips_connect_and_cold_query(
    rate_state: RateLimitStateStore,
) -> None:
    """A pooled spare serves the turn: no new connect, no one-shot
    query(); the prompt goes through ``client.query`` and the response
    through ``receive_response``. Afterwards the used session is
    disconnected (one turn per session) and a replacement spare is
    warmed in the background via the client factory."""
    pool = WarmPool(min_age_s=0, max_spares=4, ttl_s=600)
    spare = _FakeClient(messages=[
        _make_class("AssistantMessage", content=[{"type": "text", "text": "hello"}]),
        _success_result_message(),
    ])
    config = AgentConfig(model="claude-opus-4-7", allowed_tools=[], version=3)
    await _warm(pool, "alice", "weather", 3, spare, config=config)
    connect_calls_after_warm = spare.connect_calls

    replacements: list[_FakeClient] = []

    def client_factory(options: Any) -> _FakeClient:
        client = _FakeClient(options=options)
        replacements.append(client)
        return client

    transport = SubscriptionTransport(
        rate_state=rate_state,
        query_fn=_exploding_query_fn,  # spy: cold path must stay untouched
        options_cls=_fake_options_cls,
        warm_pool=pool,
        client_factory=client_factory,
    )
    events = await _run_turn(transport, config=config)

    # Turn served entirely by the spare.
    assert spare.query_prompts == ["hi"]
    assert spare.connect_calls == connect_calls_after_warm  # no re-connect
    kinds = [e.kind for e in events]
    assert kinds[0] is TaskEventKind.STARTED
    assert events[0].payload["session_source"] == "warm"
    assert TaskEventKind.MESSAGE in kinds
    assert kinds[-1] is TaskEventKind.COMPLETED

    # One turn per session: the used client's subprocess is gone.
    assert spare.disconnected is True

    # Predictive warm-up: a fresh spare was connected in the background
    # with the BASE system prompt sealed in its options.
    await pool.wait_idle()
    assert pool.spare_keys() == [("alice", "weather", 3)]
    assert len(replacements) == 1
    assert replacements[0].connect_calls == 1
    assert replacements[0].options.system_prompt == "BASE"
    assert replacements[0].options.env == {"ANTHROPIC_API_KEY": "", "ENABLE_TOOL_SEARCH": "false"}
    await pool.close_all()


@pytest.mark.asyncio
async def test_warm_turn_carries_briefing_in_prompt_not_system(
    rate_state: RateLimitStateStore,
) -> None:
    pool = WarmPool(min_age_s=0, max_spares=4, ttl_s=600)
    spare = _FakeClient()
    config = AgentConfig(model="claude-opus-4-7", allowed_tools=[], version=1)
    await _warm(pool, "alice", "weather", 1, spare, config=config)

    transport = SubscriptionTransport(
        rate_state=rate_state,
        query_fn=_exploding_query_fn,
        options_cls=_fake_options_cls,
        warm_pool=pool,
        client_factory=lambda options: _FakeClient(options=options),
    )
    await _run_turn(
        transport, config=config,
        briefing="<relevant_memory>\n[MEMORY] facts\n[/MEMORY]\n</relevant_memory>",
    )

    # build_system owns the block tags; the runner inserts the briefing
    # verbatim between (absent) history and the message — no extra wrapper.
    assert spare.query_prompts == [
        "<relevant_memory>\n[MEMORY] facts\n[/MEMORY]\n</relevant_memory>\n\nhi"
    ]
    await pool.close_all()


@pytest.mark.asyncio
async def test_dead_spare_at_send_falls_back_to_cold(
    rate_state: RateLimitStateStore,
) -> None:
    """A spare whose subprocess died between warm-up and take must not
    lose the turn: the failed stdin write triggers a transparent cold
    spawn, and the corpse is disconnected."""
    pool = WarmPool(min_age_s=0, max_spares=4, ttl_s=600)
    corpse = _FakeClient(fail_on_query=True)
    config = AgentConfig(model="claude-opus-4-7", allowed_tools=[], version=1)
    await _warm(pool, "alice", "weather", 1, corpse, config=config)

    cold_sdk = _RecordingSDK(messages=[
        _make_class("AssistantMessage", content=[{"type": "text", "text": "ok"}]),
        _success_result_message(),
    ])
    transport = SubscriptionTransport(
        rate_state=rate_state,
        query_fn=cold_sdk.query,
        options_cls=_fake_options_cls,
        warm_pool=pool,
        client_factory=lambda options: _FakeClient(options=options),
    )
    events = await _run_turn(transport, config=config)

    assert corpse.disconnected is True
    assert cold_sdk.last_prompt == "hi"  # cold spawn served the turn
    assert events[-1].kind is TaskEventKind.COMPLETED
    await pool.close_all()


@pytest.mark.asyncio
async def test_cold_turn_schedules_replacement_spare(
    rate_state: RateLimitStateStore,
) -> None:
    """First turn after idle is cold by design — but it must seed the
    pool so the NEXT turn is warm."""
    pool = WarmPool(min_age_s=0, max_spares=4, ttl_s=600)
    cold_sdk = _RecordingSDK(messages=[_success_result_message()])
    config = AgentConfig(model="claude-opus-4-7", allowed_tools=[], version=7)

    transport = SubscriptionTransport(
        rate_state=rate_state,
        query_fn=cold_sdk.query,
        options_cls=_fake_options_cls,
        warm_pool=pool,
        client_factory=lambda options: _FakeClient(options=options),
    )
    events = await _run_turn(transport, config=config)
    assert events[0].payload["session_source"] == "cold"
    assert cold_sdk.last_prompt == "hi"

    await pool.wait_idle()
    assert pool.spare_keys() == [("alice", "weather", 7)]
    spare = await pool.take(
        "alice", "weather", 7,
        fingerprint=transport._session_fingerprint(config, "BASE", config.model or None),
    )
    assert spare is not None
    assert spare.options.system_prompt == "BASE"
    await pool.close_all()


@pytest.mark.asyncio
async def test_config_version_bump_routes_to_cold_and_drops_spare(
    rate_state: RateLimitStateStore,
) -> None:
    pool = WarmPool(min_age_s=0, max_spares=4, ttl_s=600)
    v1_spare = _FakeClient()
    await _warm(pool, "alice", "weather", 1, v1_spare)

    cold_sdk = _RecordingSDK(messages=[_success_result_message()])
    transport = SubscriptionTransport(
        rate_state=rate_state,
        query_fn=cold_sdk.query,
        options_cls=_fake_options_cls,
        warm_pool=pool,
        client_factory=lambda options: _FakeClient(options=options),
    )
    config_v2 = AgentConfig(model="claude-opus-4-7", allowed_tools=[], version=2)
    events = await _run_turn(transport, config=config_v2)

    assert events[0].payload["session_source"] == "cold"
    assert cold_sdk.last_prompt == "hi"  # served cold
    assert v1_spare.query_prompts == []  # stale spare never used …
    await pool.wait_idle()
    assert v1_spare.disconnected is True  # … and its subprocess is gone
    await pool.close_all()


@pytest.mark.asyncio
async def test_warm_turn_records_rate_limit_and_disconnects_on_sdk_error(
    rate_state: RateLimitStateStore,
) -> None:
    """The pump behaves identically on the warm path: rate-limit events
    feed the cache, an iterator failure surfaces as a structured error,
    and the session is disconnected either way."""

    class _ExplodingClient(_FakeClient):
        async def receive_response(self) -> AsyncIterator[object]:
            yield _make_class(
                "RateLimitEvent",
                rate_limit_info={
                    "status": "allowed", "rate_limit_type": "five_hour",
                    "overage_status": "allowed", "overage_disabled_reason": None,
                    "utilization": 0.5, "resets_at": None, "overage_resets_at": None,
                },
            )
            raise RuntimeError("CLI crashed mid-turn")

    pool = WarmPool(min_age_s=0, max_spares=4, ttl_s=600)
    spare = _ExplodingClient()
    config = AgentConfig(model="claude-opus-4-7", allowed_tools=[], version=1)
    await _warm(pool, "alice", "weather", 1, spare, config=config)

    transport = SubscriptionTransport(
        rate_state=rate_state,
        query_fn=_exploding_query_fn,
        options_cls=_fake_options_cls,
        warm_pool=pool,
        client_factory=lambda options: _FakeClient(options=options),
    )
    events = await _run_turn(transport, config=config)

    assert events[-1].kind is TaskEventKind.ERROR
    assert events[-1].payload["reason"] == "sdk_exception"
    assert spare.disconnected is True
    snapshot = rate_state.get("alice", "weather")
    assert snapshot is not None and snapshot.utilization == 0.5
    await pool.close_all()


@pytest.mark.asyncio
async def test_no_pool_keeps_legacy_cold_behaviour(
    rate_state: RateLimitStateStore,
) -> None:
    """Without a pool wired (unit tests, RECORDATIO_WARM_POOL_MAX=0 setups)
    the transport is byte-for-byte the old one-shot query() path."""
    cold_sdk = _RecordingSDK(messages=[_success_result_message()])
    transport = SubscriptionTransport(
        rate_state=rate_state, query_fn=cold_sdk.query, options_cls=_fake_options_cls,
    )
    events = await _run_turn(
        transport, config=AgentConfig(model="claude-opus-4-7", allowed_tools=[]),
    )
    assert events[0].payload["session_source"] == "cold"
    assert cold_sdk.last_prompt == "hi"
    assert events[-1].kind is TaskEventKind.COMPLETED


async def test_take_skips_spare_younger_than_min_age() -> None:
    """A just-spawned spare may still be finishing its background MCP
    connect — take() leaves it to mature (turn goes cold) and only
    returns it once the minimum age has passed."""
    now = [0.0]
    pool = WarmPool(min_age_s=3, max_spares=4, ttl_s=600, clock=lambda: now[0])
    client = _FakeClient()

    async def connect() -> _FakeClient:
        return client

    task = pool.schedule_warm("u", "s", 1, connect)
    assert task is not None
    await task

    # Too young: left in the pool, caller goes cold.
    assert await pool.take("u", "s", 1) is None
    # Matured: handed out normally.
    now[0] = 3.5
    assert await pool.take("u", "s", 1) is client
    await pool.close_all()


@pytest.mark.asyncio
@pytest.mark.parametrize("header", ["X-Telos-Turn-Id", "x-telos-turn-id", "Authorization"])
async def test_equal_versions_with_changed_headers_never_reuse_spare(
    rate_state: RateLimitStateStore, header: str,
) -> None:
    pool = WarmPool(min_age_s=0)
    old = AgentConfig(mcp_servers=[
        {"name": "tools", "type": "http", "url": "https://example.test/mcp",
         "headers": {header: "first"}},
    ])
    spare = _FakeClient()
    await _warm(pool, "alice", "weather", 1, spare, config=old)
    current = AgentConfig.from_dict(old.to_dict())
    current.mcp_servers = [{**old.mcp_servers[0], "headers": {header: "second"}}]
    cold = _RecordingSDK(messages=[_success_result_message()])
    transport = SubscriptionTransport(
        rate_state=rate_state, query_fn=cold.query, options_cls=_fake_options_cls,
        warm_pool=pool, client_factory=lambda options: _FakeClient(options=options),
    )
    events = await _run_turn(transport, config=current)
    await pool.wait_idle()
    assert events[0].payload["session_source"] == "cold"
    assert spare.query_prompts == []
    assert spare.disconnected
    assert cold.last_options.mcp_servers["tools"]["headers"][header] == "second"
    if header.lower() == "x-telos-turn-id":
        assert pool.size == 0
    else:
        assert pool.size == 1
    await pool.close_all()


@pytest.mark.asyncio
async def test_predictive_warm_snapshots_inline_configuration(
    rate_state: RateLimitStateStore,
) -> None:
    pool = WarmPool(min_age_s=0)
    clients = []

    def factory(options):
        client = _FakeClient(options=options)
        clients.append(client)
        return client

    transport = SubscriptionTransport(
        rate_state=rate_state, query_fn=_exploding_query_fn,
        options_cls=_fake_options_cls, warm_pool=pool, client_factory=factory,
    )
    config = AgentConfig(mcp_servers=[{"name": "tools", "headers": {"Authorization": "first"}}])
    transport._maybe_schedule_warm(
        user_id="alice", slug="weather", system_prompt="BASE", config=config, model=None,
    )
    config.mcp_servers[0]["headers"]["Authorization"] = "second"
    await pool.wait_idle()
    assert clients[0].options.mcp_servers["tools"]["headers"]["Authorization"] == "first"
    assert await pool.take(
        "alice", "weather", 1,
        fingerprint=transport._session_fingerprint(config, "BASE", None),
    ) is None
    await pool.wait_idle()
    assert clients[0].disconnected
    await pool.close_all()


@pytest.mark.parametrize("change", [
    {"system_prompt": "different"},
    {"model": "different"},
    {"allowed_tools": []},
    {"subagents": [{"name": "helper", "prompt": "different"}]},
    {"mcp_servers": [{"name": "tools", "url": "https://different.test/mcp"}]},
])
def test_fingerprint_covers_all_session_configuration(change) -> None:
    original = AgentConfig()
    changed = AgentConfig(**change)
    fingerprint = SubscriptionTransport._session_fingerprint
    assert fingerprint(original, original.system_prompt, original.model) != fingerprint(
        changed, changed.system_prompt, changed.model,
    )
