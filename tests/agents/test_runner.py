"""Runner tests: transport selection + the manually-induced fallback.

Phase 1 acceptance criteria covered here:

- Transport selection given different cached rate-limit states.
- The "manually inducing overage_status: rejected causes the next task
  to take the ApiKeyTransport path with no loss of memory briefing or
  extraction" criterion.

The SDK and Pontifex are both stubbed so the tests can run hermetically
on any developer's box without network or credentials.
"""

import logging
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anyio
import httpx
import pytest

from tpasini.recordatio.agents.config_store import AgentConfig
from tpasini.recordatio.agents.events import TaskEvent, TaskEventKind
from tpasini.recordatio.agents.rate_limit_state import RateLimitStateStore
from tpasini.recordatio.agents.runner import (
    _HISTORY_CHAR_BUDGET,
    AgentRunner,
    ApiKeyTransport,
    SubscriptionTransport,
    TransportSelection,
    _render_prompt_with_history,
    select_transport,
)

# ---------------------------------------------------------------------------
# Fake SDK + Pontifex
# ---------------------------------------------------------------------------


def _make_class(name: str, **fields: object) -> object:
    cls = type(name, (SimpleNamespace,), {})
    return cls(**fields)


class _RecordingSDK:
    """Captures the kwargs the runner passed to ``query()`` and yields a
    canned sequence of fake messages.

    ``options`` is what we want to assert against — it must carry the
    briefing-augmented system_prompt, the clean cwd, and the agent's
    allowed_tools. The fake messages drive coverage for rate-limit
    cache updates and completion handling.

    We snapshot the cwd's contents at call time too: the runner uses a
    short-lived TemporaryDirectory so the path is gone by the time the
    test asserts; capturing inside the fake side-steps that.
    """

    def __init__(self, messages: list[object]) -> None:
        self.messages = messages
        self.last_prompt: str | None = None
        self.last_options: Any = None
        self.cwd_snapshot: dict[str, Any] | None = None

    async def query(self, *, prompt: str, options: Any) -> AsyncIterator[object]:
        self.last_prompt = prompt
        self.last_options = options
        cwd = Path(getattr(options, "cwd", "")) if getattr(options, "cwd", None) else None
        if cwd is not None:
            self.cwd_snapshot = {
                "exists": cwd.exists(),
                "contains_claude_md": (cwd / "CLAUDE.md").exists(),
                "path_str": str(cwd),
            }
        for msg in self.messages:
            yield msg


class _FakePontifex:
    """Stub for the ``Pontifex.stream()`` interface used by ApiKeyTransport."""

    def __init__(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        self.received_messages: list[dict] | None = None
        self.received_system: str | None = None
        self.received_kwargs: dict[str, Any] | None = None

    async def stream(
        self,
        messages: list[dict],
        system: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict]:
        self.received_messages = messages
        self.received_system = system
        self.received_kwargs = kwargs
        for chunk in self.chunks:
            yield chunk


def _fake_options_cls(**kwargs: Any) -> SimpleNamespace:
    """Plain bag for kwargs — we just want to assert what got passed in."""
    return SimpleNamespace(**kwargs)


# ---------------------------------------------------------------------------
# select_transport
# ---------------------------------------------------------------------------


@pytest.fixture()
def rate_state(tmp_path: Path) -> RateLimitStateStore:
    return RateLimitStateStore(tmp_path / "rate.db")


def test_select_transport_defaults_to_subscription_when_no_cache(
    rate_state: RateLimitStateStore,
) -> None:
    assert select_transport(rate_state, "alice", "weather") is TransportSelection.SUBSCRIPTION


def test_select_transport_picks_api_key_when_window_exceeded(
    rate_state: RateLimitStateStore,
) -> None:
    from datetime import UTC as _UTC
    from datetime import datetime as _dt
    future = str(_dt.now(_UTC).timestamp() + 3600)
    rate_state.upsert(
        user_id="alice", slug="weather",
        status="exceeded", rate_limit_type="five_hour",
        overage_status="rejected", overage_disabled_reason="out_of_credits",
        utilization=1.0, resets_at=future, overage_resets_at=None,
    )
    assert select_transport(rate_state, "alice", "weather") is TransportSelection.API_KEY


def test_select_transport_stays_on_subscription_for_policy_only_rejection(
    rate_state: RateLimitStateStore,
) -> None:
    """overage=rejected on a healthy snapshot is a policy descriptor —
    routing on it sent every turn to the fallback at 20% utilization
    (prod, 2026-06-10)."""
    rate_state.upsert(
        user_id="alice", slug="weather",
        status="allowed", rate_limit_type="five_hour",
        overage_status="rejected", overage_disabled_reason="out_of_credits",
        utilization=0.2, resets_at=None, overage_resets_at=None,
    )
    assert select_transport(rate_state, "alice", "weather") is TransportSelection.SUBSCRIPTION


def test_select_transport_resets_to_subscription_after_clear(
    rate_state: RateLimitStateStore,
) -> None:
    rate_state.upsert(
        user_id="alice", slug="weather",
        status="exceeded", rate_limit_type="five_hour",
        overage_status="rejected", overage_disabled_reason="out_of_credits",
        utilization=1.0, resets_at=None, overage_resets_at=None,
    )
    # When a fresh "allowed" snapshot lands, the next task should pick
    # subscription again. ``upsert`` is unconditional replace; that's
    # the migration-plan-mandated "reset" behaviour ("when a subsequent
    # subscription call succeeds with remaining > 0, clear the fallback
    # flag").
    rate_state.upsert(
        user_id="alice", slug="weather",
        status="allowed", rate_limit_type="five_hour",
        overage_status="allowed", overage_disabled_reason=None,
        utilization=0.1, resets_at=None, overage_resets_at=None,
    )
    assert select_transport(rate_state, "alice", "weather") is TransportSelection.SUBSCRIPTION


# ---------------------------------------------------------------------------
# SubscriptionTransport — direct
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subscription_transport_passes_clean_cwd_and_briefed_prompt(
    rate_state: RateLimitStateStore,
) -> None:
    sdk = _RecordingSDK(messages=[
        _make_class("SystemMessage", subtype="init", data={"session_id": "s1"}),
        _make_class("AssistantMessage", content=[{"type": "text", "text": "hi"}]),
        _make_class(
            "ResultMessage",
            subtype="success", is_error=False, stop_reason="end_turn",
            duration_ms=10, duration_api_ms=10, num_turns=1, session_id="s1",
            total_cost_usd=0.0, usage=None, result=None, structured_output=None,
            model_usage=None, permission_denials=[], deferred_tool_use=None,
            errors=[], api_error_status=None,
        ),
    ])
    transport = SubscriptionTransport(
        rate_state=rate_state,
        query_fn=sdk.query,
        options_cls=_fake_options_cls,
    )
    events: list[TaskEvent] = []
    async for ev in transport.run(
        user_id="alice", slug="weather",
        content="hi", system_prompt="SYSTEM+BRIEFING",
        config=AgentConfig(model="claude-opus-4-7", allowed_tools=[]),
    ):
        events.append(ev)

    # Sanity: the recording SDK saw the briefed prompt and a clean cwd.
    opts = sdk.last_options
    assert opts is not None
    assert opts.system_prompt == "SYSTEM+BRIEFING"
    assert opts.permission_mode == "bypassPermissions"
    assert opts.allowed_tools == []
    # Token-level streaming: partial StreamEvents must be requested so the
    # translator can emit message_delta/thinking_delta events.
    assert opts.include_partial_messages is True
    # Subscription-by-construction: the CLI prefers an env ANTHROPIC_API_KEY
    # over the OAuth token, so the subprocess env must blank it out or the
    # transport silently flips to API billing.
    assert opts.env == {"ANTHROPIC_API_KEY": "", "ENABLE_TOOL_SEARCH": "false"}
    # Interactive built-ins fail headless and make the model repeat itself.
    assert set(opts.disallowed_tools) >= {
        "AskUserQuestion", "EnterPlanMode", "ExitPlanMode",
        "CronCreate", "CronDelete", "CronList", "RemoteTrigger", "ScheduleWakeup",
    }
    # The fake snapshots the cwd state at call time because the runner
    # uses a TemporaryDirectory that's gone by the time we get here.
    assert sdk.cwd_snapshot is not None
    assert sdk.cwd_snapshot["exists"] is True
    assert sdk.cwd_snapshot["contains_claude_md"] is False

    kinds = [e.kind for e in events]
    # First event is the runner-emitted ``started`` (with transport tag),
    # then the translator emits ``started`` for the SDK's init message —
    # both are fine; the wire contract just requires *some* ``started``.
    assert kinds[0] is TaskEventKind.STARTED
    assert events[0].payload["transport"] == "subscription"
    assert TaskEventKind.MESSAGE in kinds
    assert kinds[-1] is TaskEventKind.COMPLETED


@pytest.mark.asyncio
async def test_subscription_transport_caches_rate_limit_event(
    rate_state: RateLimitStateStore,
) -> None:
    sdk = _RecordingSDK(messages=[
        _make_class(
            "RateLimitEvent",
            rate_limit_info={
                "status": "allowed",
                "rate_limit_type": "five_hour",
                "overage_status": "rejected",
                "overage_disabled_reason": "out_of_credits",
                "utilization": 0.91,
                "resets_at": None,
                "overage_resets_at": None,
            },
        ),
        _make_class(
            "ResultMessage",
            subtype="success", is_error=False, stop_reason="end_turn",
            duration_ms=1, duration_api_ms=1, num_turns=1, session_id="s",
            total_cost_usd=0.0, usage=None, result=None, structured_output=None,
            model_usage=None, permission_denials=[], deferred_tool_use=None,
            errors=[], api_error_status=None,
        ),
    ])
    transport = SubscriptionTransport(
        rate_state=rate_state,
        query_fn=sdk.query,
        options_cls=_fake_options_cls,
    )
    async for _ in transport.run(
        user_id="alice", slug="weather",
        content="hi", system_prompt="sys",
        config=AgentConfig(allowed_tools=[]),
    ):
        pass

    # The event is cached verbatim — but overage=rejected on a healthy
    # (status=allowed) snapshot is policy, not state, so the next turn
    # STAYS on the subscription.
    snap = rate_state.get("alice", "weather")
    assert snap is not None
    assert snap.overage_status == "rejected"
    assert select_transport(rate_state, "alice", "weather") is TransportSelection.SUBSCRIPTION


# ---------------------------------------------------------------------------
# SubscriptionTransport — Phase 4 subagent plumbing
# ---------------------------------------------------------------------------


class _FakeAgentDef(SimpleNamespace):
    """Stand-in for ``claude_agent_sdk.AgentDefinition`` — captures
    whatever kwargs the runner passed so the test can introspect.
    """


@pytest.mark.asyncio
async def test_subscription_transport_passes_subagents_to_options(
    rate_state: RateLimitStateStore,
) -> None:
    sdk = _RecordingSDK(messages=[
        _make_class(
            "ResultMessage",
            subtype="success", is_error=False, stop_reason="end_turn",
            duration_ms=10, duration_api_ms=10, num_turns=1, session_id="s1",
            total_cost_usd=0.0, usage=None, result=None, structured_output=None,
            model_usage=None, permission_denials=[], deferred_tool_use=None,
            errors=[], api_error_status=None,
        ),
    ])
    transport = SubscriptionTransport(
        rate_state=rate_state,
        query_fn=sdk.query,
        options_cls=_fake_options_cls,
        agent_def_cls=_FakeAgentDef,
    )
    config = AgentConfig(
        model="claude-opus-4-7",
        allowed_tools=[],
        subagents=[
            {"name": "researcher", "description": "Finds facts.",
             "prompt": "You research.", "tools": ["WebSearch"]},
            {"name": "judge", "description": "Scores candidates.",
             "prompt": "You judge.", "tools": None},
        ],
    )
    async for _ in transport.run(
        user_id="alice", slug="weather",
        content="hi", system_prompt="SYSTEM",
        config=config,
    ):
        pass

    opts = sdk.last_options
    assert opts is not None
    assert hasattr(opts, "agents"), "subagents must be threaded through to options"
    assert set(opts.agents.keys()) == {"researcher", "judge"}
    assert opts.agents["researcher"].description == "Finds facts."
    assert opts.agents["researcher"].prompt == "You research."
    assert opts.agents["researcher"].tools == ["WebSearch"]
    assert opts.agents["judge"].tools is None


@pytest.mark.asyncio
async def test_subscription_transport_omits_agents_key_when_no_subagents(
    rate_state: RateLimitStateStore,
) -> None:
    sdk = _RecordingSDK(messages=[
        _make_class(
            "ResultMessage",
            subtype="success", is_error=False, stop_reason="end_turn",
            duration_ms=10, duration_api_ms=10, num_turns=1, session_id="s1",
            total_cost_usd=0.0, usage=None, result=None, structured_output=None,
            model_usage=None, permission_denials=[], deferred_tool_use=None,
            errors=[], api_error_status=None,
        ),
    ])
    transport = SubscriptionTransport(
        rate_state=rate_state,
        query_fn=sdk.query,
        options_cls=_fake_options_cls,
        agent_def_cls=_FakeAgentDef,
    )
    async for _ in transport.run(
        user_id="alice", slug="weather",
        content="hi", system_prompt="SYSTEM",
        config=AgentConfig(allowed_tools=[]),  # subagents defaults to []
    ):
        pass

    opts = sdk.last_options
    assert opts is not None
    assert not hasattr(opts, "agents"), "no 'agents' kwarg should be passed when subagents is empty"


@pytest.mark.asyncio
async def test_subscription_transport_subagent_missing_fields_get_defaults(
    rate_state: RateLimitStateStore,
) -> None:
    sdk = _RecordingSDK(messages=[
        _make_class(
            "ResultMessage",
            subtype="success", is_error=False, stop_reason="end_turn",
            duration_ms=10, duration_api_ms=10, num_turns=1, session_id="s1",
            total_cost_usd=0.0, usage=None, result=None, structured_output=None,
            model_usage=None, permission_denials=[], deferred_tool_use=None,
            errors=[], api_error_status=None,
        ),
    ])
    transport = SubscriptionTransport(
        rate_state=rate_state,
        query_fn=sdk.query,
        options_cls=_fake_options_cls,
        agent_def_cls=_FakeAgentDef,
    )
    config = AgentConfig(
        allowed_tools=[],
        subagents=[{"name": "minimal"}],  # only name; description/prompt/tools missing
    )
    async for _ in transport.run(
        user_id="alice", slug="weather",
        content="hi", system_prompt="SYSTEM",
        config=config,
    ):
        pass

    opts = sdk.last_options
    assert opts.agents["minimal"].description == ""
    assert opts.agents["minimal"].prompt == ""
    assert opts.agents["minimal"].tools is None


# ---------------------------------------------------------------------------
# SubscriptionTransport — Phase 4 subagent VALIDATION (PR #134 review fixes)
# ---------------------------------------------------------------------------
#
# These tests cover the validation rules added by the review:
#   - missing 'name'              -> structured ERROR event (covered by
#     test_subscription_transport_malformed_subagent_yields_error_event above)
#   - non-dict items in subagents -> structured ERROR event
#   - duplicate names             -> structured ERROR event
#   - 'tools' set to a string     -> structured ERROR event
#   - all SDK fields forwarded    -> AgentDefinition kwargs match
#   - delegation-tools missing    -> warning logged (does not raise)


@pytest.mark.asyncio
async def test_subagent_non_dict_item_emits_structured_error(
    rate_state: RateLimitStateStore,
) -> None:
    """Finding #7: a bare string (or any non-dict) in subagents must be
    rejected with a structured ERROR rather than raising TypeError out
    of the generator."""

    sdk = _RecordingSDK(messages=[_success_result_message()])
    transport = SubscriptionTransport(
        rate_state=rate_state, query_fn=sdk.query,
        options_cls=_fake_options_cls, agent_def_cls=_FakeAgentDef,
    )
    config = AgentConfig(allowed_tools=[], subagents=["researcher"])  # type: ignore[list-item]
    events: list[TaskEvent] = []
    async for ev in transport.run(
        user_id="alice", slug="weather",
        content="hi", system_prompt="SYS",
        config=config,
    ):
        events.append(ev)

    err = next(e for e in events if e.kind is TaskEventKind.ERROR)
    assert err.payload["reason"] == "invalid_subagent_config"
    assert "must be a dict" in err.payload["message"]


@pytest.mark.asyncio
async def test_subagent_duplicate_names_emits_structured_error(
    rate_state: RateLimitStateStore,
) -> None:
    """Finding #2: silently dropping earlier dupes via dict-comp keying is
    a footgun. The runner must reject the config."""

    sdk = _RecordingSDK(messages=[_success_result_message()])
    transport = SubscriptionTransport(
        rate_state=rate_state, query_fn=sdk.query,
        options_cls=_fake_options_cls, agent_def_cls=_FakeAgentDef,
    )
    config = AgentConfig(
        allowed_tools=[],
        subagents=[
            {"name": "researcher", "description": "v1", "prompt": "p1"},
            {"name": "researcher", "description": "v2", "prompt": "p2"},
        ],
    )
    events: list[TaskEvent] = []
    async for ev in transport.run(
        user_id="alice", slug="weather",
        content="hi", system_prompt="SYS",
        config=config,
    ):
        events.append(ev)

    err = next(e for e in events if e.kind is TaskEventKind.ERROR)
    assert err.payload["reason"] == "invalid_subagent_config"
    assert "duplicate subagent name" in err.payload["message"]
    assert "researcher" in err.payload["message"]


@pytest.mark.asyncio
async def test_subagent_tools_as_string_emits_structured_error(
    rate_state: RateLimitStateStore,
) -> None:
    """Finding #5: ``tools='WebSearch'`` is the worst footgun — the SDK
    would iterate the string into ['W','e','b',...]. Reject it explicitly."""

    sdk = _RecordingSDK(messages=[_success_result_message()])
    transport = SubscriptionTransport(
        rate_state=rate_state, query_fn=sdk.query,
        options_cls=_fake_options_cls, agent_def_cls=_FakeAgentDef,
    )
    config = AgentConfig(
        allowed_tools=[],
        subagents=[
            {"name": "r", "description": "d", "prompt": "p", "tools": "WebSearch"},
        ],
    )
    events: list[TaskEvent] = []
    async for ev in transport.run(
        user_id="alice", slug="weather",
        content="hi", system_prompt="SYS",
        config=config,
    ):
        events.append(ev)

    err = next(e for e in events if e.kind is TaskEventKind.ERROR)
    assert err.payload["reason"] == "invalid_subagent_config"
    assert "tools" in err.payload["message"]
    assert "list" in err.payload["message"]


@pytest.mark.asyncio
async def test_subagent_forwards_all_sdk_fields(
    rate_state: RateLimitStateStore,
) -> None:
    """Findings #8 + #9: per-subagent ``model`` and every other AgentDefinition
    field (``permissionMode``, ``mcpServers``, ``disallowedTools``, ``skills``,
    ``memory``, ``maxTurns``, ``effort``, ``background``, ``initialPrompt``)
    must be forwarded to the SDK rather than silently dropped."""

    sdk = _RecordingSDK(messages=[_success_result_message()])
    transport = SubscriptionTransport(
        rate_state=rate_state, query_fn=sdk.query,
        options_cls=_fake_options_cls, agent_def_cls=_FakeAgentDef,
    )
    config = AgentConfig(
        allowed_tools=["Task"],
        subagents=[{
            "name": "full",
            "description": "exercises every field",
            "prompt": "be thorough",
            "tools": ["Read"],
            "model": "claude-opus-4-7",
            "disallowedTools": ["Write"],
            "skills": ["analysis"],
            "memory": "project",
            "maxTurns": 5,
            "effort": "high",
            "permissionMode": "acceptEdits",
            "mcpServers": ["pontifex"],
            "initialPrompt": "warm up",
            "background": True,
        }],
    )
    async for _ in transport.run(
        user_id="alice", slug="weather",
        content="hi", system_prompt="SYS",
        config=config,
    ):
        pass

    full = sdk.last_options.agents["full"]
    assert full.description == "exercises every field"
    assert full.prompt == "be thorough"
    assert full.tools == ["Read"]
    assert full.model == "claude-opus-4-7"
    assert full.disallowedTools == ["Write"]
    assert full.skills == ["analysis"]
    assert full.memory == "project"
    assert full.maxTurns == 5
    assert full.effort == "high"
    assert full.permissionMode == "acceptEdits"
    assert full.mcpServers == ["pontifex"]
    assert full.initialPrompt == "warm up"
    assert full.background is True


@pytest.mark.asyncio
async def test_subagent_accepts_snake_case_field_aliases(
    rate_state: RateLimitStateStore,
) -> None:
    """Seeders may emit snake_case for ergonomics; the runner normalises
    to the SDK's camelCase before constructing AgentDefinition."""

    sdk = _RecordingSDK(messages=[_success_result_message()])
    transport = SubscriptionTransport(
        rate_state=rate_state, query_fn=sdk.query,
        options_cls=_fake_options_cls, agent_def_cls=_FakeAgentDef,
    )
    config = AgentConfig(
        allowed_tools=["Task"],
        subagents=[{
            "name": "snake",
            "description": "d",
            "prompt": "p",
            "disallowed_tools": ["Write"],
            "max_turns": 3,
            "permission_mode": "acceptEdits",
            "mcp_servers": ["pontifex"],
            "initial_prompt": "go",
        }],
    )
    async for _ in transport.run(
        user_id="alice", slug="weather",
        content="hi", system_prompt="SYS",
        config=config,
    ):
        pass

    snake = sdk.last_options.agents["snake"]
    assert snake.disallowedTools == ["Write"]
    assert snake.maxTurns == 3
    assert snake.permissionMode == "acceptEdits"
    assert snake.mcpServers == ["pontifex"]
    assert snake.initialPrompt == "go"


@pytest.mark.asyncio
async def test_subagent_unknown_field_ignored_for_forward_compat(
    rate_state: RateLimitStateStore,
) -> None:
    """A newer seeder may emit a field this Recordatio version doesn't know
    about. Refusing to launch on such input is the wrong call; we ignore
    unknown keys so deploys can roll forward independently."""

    sdk = _RecordingSDK(messages=[_success_result_message()])
    transport = SubscriptionTransport(
        rate_state=rate_state, query_fn=sdk.query,
        options_cls=_fake_options_cls, agent_def_cls=_FakeAgentDef,
    )
    config = AgentConfig(
        allowed_tools=["Task"],
        subagents=[{
            "name": "future",
            "description": "d", "prompt": "p",
            "someFutureField": {"experimental": True},
        }],
    )
    async for _ in transport.run(
        user_id="alice", slug="weather",
        content="hi", system_prompt="SYS",
        config=config,
    ):
        pass

    future = sdk.last_options.agents["future"]
    # The unknown field is dropped on the floor — _FakeAgentDef would
    # accept it if we passed it (SimpleNamespace), so this asserts our
    # filter is doing its job.
    assert not hasattr(future, "someFutureField")


@pytest.mark.asyncio
async def test_subagent_tools_none_vs_empty_list_preserved(
    rate_state: RateLimitStateStore,
) -> None:
    """Finding #6 documentation: ``tools=None`` ≠ ``tools=[]``. The runner
    must NOT coerce one into the other — the SDK semantic is
    None=inherit, []=no tools. This test pins the behaviour so a future
    refactor can't silently swap them."""

    sdk = _RecordingSDK(messages=[_success_result_message()])
    transport = SubscriptionTransport(
        rate_state=rate_state, query_fn=sdk.query,
        options_cls=_fake_options_cls, agent_def_cls=_FakeAgentDef,
    )
    config = AgentConfig(
        allowed_tools=["Task"],
        subagents=[
            {"name": "inherits", "description": "d", "prompt": "p", "tools": None},
            {"name": "locked", "description": "d", "prompt": "p", "tools": []},
        ],
    )
    async for _ in transport.run(
        user_id="alice", slug="weather",
        content="hi", system_prompt="SYS",
        config=config,
    ):
        pass

    assert sdk.last_options.agents["inherits"].tools is None
    assert sdk.last_options.agents["locked"].tools == []


@pytest.mark.asyncio
async def test_subagent_delegation_tools_missing_logs_warning(
    rate_state: RateLimitStateStore,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Finding #10: a non-empty allowed_tools list that includes neither
    'Task' nor 'Agent' makes any configured subagents unreachable. The
    runner warns (does not raise) so the inconsistency is observable."""

    sdk = _RecordingSDK(messages=[_success_result_message()])
    transport = SubscriptionTransport(
        rate_state=rate_state, query_fn=sdk.query,
        options_cls=_fake_options_cls, agent_def_cls=_FakeAgentDef,
    )
    config = AgentConfig(
        allowed_tools=["Read", "Write"],  # no Task, no Agent
        subagents=[{"name": "stranded", "description": "d", "prompt": "p"}],
    )
    with caplog.at_level(logging.WARNING, logger="tpasini.recordatio.agents.runner"):
        async for _ in transport.run(
            user_id="alice", slug="weather",
            content="hi", system_prompt="SYS",
            config=config,
        ):
            pass

    assert any(
        "delegation impossible" in rec.message
        for rec in caplog.records
    )
    # And the run still completed (no ERROR event from a raise).
    # (We assert by checking the SDK was actually invoked.)
    assert sdk.last_options is not None


@pytest.mark.asyncio
async def test_subagent_delegation_warning_suppressed_when_task_allowed(
    rate_state: RateLimitStateStore,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Counter-test: when 'Task' is in allowed_tools, no warning fires."""

    sdk = _RecordingSDK(messages=[_success_result_message()])
    transport = SubscriptionTransport(
        rate_state=rate_state, query_fn=sdk.query,
        options_cls=_fake_options_cls, agent_def_cls=_FakeAgentDef,
    )
    config = AgentConfig(
        allowed_tools=["Task", "Read"],
        subagents=[{"name": "reachable", "description": "d", "prompt": "p"}],
    )
    with caplog.at_level(logging.WARNING, logger="tpasini.recordatio.agents.runner"):
        async for _ in transport.run(
            user_id="alice", slug="weather",
            content="hi", system_prompt="SYS",
            config=config,
        ):
            pass

    assert not any(
        "delegation impossible" in rec.message for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_subagent_delegation_warning_suppressed_when_allowlist_empty(
    rate_state: RateLimitStateStore,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Empty allowed_tools is the SDK's "allow all" — no warning either."""

    sdk = _RecordingSDK(messages=[_success_result_message()])
    transport = SubscriptionTransport(
        rate_state=rate_state, query_fn=sdk.query,
        options_cls=_fake_options_cls, agent_def_cls=_FakeAgentDef,
    )
    config = AgentConfig(
        allowed_tools=[],
        subagents=[{"name": "reachable", "description": "d", "prompt": "p"}],
    )
    with caplog.at_level(logging.WARNING, logger="tpasini.recordatio.agents.runner"):
        async for _ in transport.run(
            user_id="alice", slug="weather",
            content="hi", system_prompt="SYS",
            config=config,
        ):
            pass

    assert not any(
        "delegation impossible" in rec.message for rec in caplog.records
    )


# ---------------------------------------------------------------------------
# SubscriptionTransport — terminal-event guarantee + timeout scoping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subscription_transport_synthesizes_error_on_eof_without_result(
    rate_state: RateLimitStateStore,
) -> None:
    """Finding #9: if the SDK iterator ends without a ResultMessage, the
    stream used to fall silent — SSE consumers hung forever waiting for a
    terminal event. The runner must synthesize a structured ERROR."""

    sdk = _RecordingSDK(messages=[
        _make_class("AssistantMessage", content=[{"type": "text", "text": "partial"}]),
        # No ResultMessage — the stream just ends.
    ])
    transport = SubscriptionTransport(
        rate_state=rate_state, query_fn=sdk.query, options_cls=_fake_options_cls,
    )
    events: list[TaskEvent] = []
    async for ev in transport.run(
        user_id="alice", slug="weather",
        content="hi", system_prompt="SYS",
        config=AgentConfig(allowed_tools=[]),
    ):
        events.append(ev)

    assert events[-1].kind is TaskEventKind.ERROR
    assert events[-1].payload["reason"] == "sdk_eof_without_result"
    # The partial output still made it out before the synthetic terminal.
    assert TaskEventKind.MESSAGE in [e.kind for e in events]


@pytest.mark.asyncio
async def test_subscription_transport_no_synthetic_error_after_completed(
    rate_state: RateLimitStateStore,
) -> None:
    """Counter-test: a stream that DOES end with a ResultMessage must not
    grow a trailing synthetic error."""

    sdk = _RecordingSDK(messages=[_success_result_message()])
    transport = SubscriptionTransport(
        rate_state=rate_state, query_fn=sdk.query, options_cls=_fake_options_cls,
    )
    events: list[TaskEvent] = []
    async for ev in transport.run(
        user_id="alice", slug="weather",
        content="hi", system_prompt="SYS",
        config=AgentConfig(allowed_tools=[]),
    ):
        events.append(ev)

    assert events[-1].kind is TaskEventKind.COMPLETED
    assert TaskEventKind.ERROR not in [e.kind for e in events]


@pytest.mark.asyncio
async def test_subscription_transport_times_out_when_sdk_stalls(
    rate_state: RateLimitStateStore,
) -> None:
    """Finding #8 regression guard: after moving the timeout to the pump
    task (so consumer back-pressure can't trip it), a genuinely stalled
    SDK must still surface the structured timeout error."""

    async def stalling_query(**_: Any) -> AsyncIterator[object]:
        yield _make_class("AssistantMessage", content=[{"type": "text", "text": "partial"}])
        await anyio.sleep(30)

    transport = SubscriptionTransport(
        rate_state=rate_state,
        query_fn=stalling_query,
        options_cls=_fake_options_cls,
        timeout_s=0.1,
    )
    events: list[TaskEvent] = []
    async for ev in transport.run(
        user_id="alice", slug="weather",
        content="hi", system_prompt="SYS",
        config=AgentConfig(allowed_tools=[]),
    ):
        events.append(ev)

    # The partial message still streamed out, then the timeout error.
    assert TaskEventKind.MESSAGE in [e.kind for e in events]
    assert events[-1].kind is TaskEventKind.ERROR
    assert events[-1].payload["reason"] == "timeout"
    assert events[-1].payload["timeout_s"] == 0.1


@pytest.mark.asyncio
async def test_subscription_transport_slow_consumer_does_not_trip_timeout(
    rate_state: RateLimitStateStore,
) -> None:
    """Finding #8, the actual claim: a slow downstream consumer must NEVER
    trip the SDK-progress timeout — even past the 64-message buffer.

    Regression for the PR #139 review blocker: ``send()`` used to sit
    inside ``fail_after``, so once the buffer filled, time blocked on a
    slow consumer counted against the timeout and produced a spurious
    ``reason: timeout`` mid-stream. The SDK here yields 200 messages
    instantly; the consumer drains far slower than ``timeout_s`` allows
    in aggregate. Every event must arrive, with a clean COMPLETED and no
    ERROR."""

    n_messages = 200
    messages: list[object] = [
        _make_class("AssistantMessage", content=[{"type": "text", "text": f"m{i}"}])
        for i in range(n_messages)
    ]
    messages.append(_success_result_message())
    sdk = _RecordingSDK(messages=messages)
    transport = SubscriptionTransport(
        rate_state=rate_state,
        query_fn=sdk.query,
        options_cls=_fake_options_cls,
        timeout_s=0.05,  # total consumer drain time (~0.4s) dwarfs this
    )
    events: list[TaskEvent] = []
    async for ev in transport.run(
        user_id="alice", slug="weather",
        content="hi", system_prompt="SYS",
        config=AgentConfig(allowed_tools=[]),
    ):
        events.append(ev)
        # Slow consumer: per-event wall time well past the buffer window.
        await anyio.sleep(0.002)

    kinds = [e.kind for e in events]
    assert TaskEventKind.ERROR not in kinds
    assert kinds.count(TaskEventKind.MESSAGE) == n_messages
    assert events[-1].kind is TaskEventKind.COMPLETED


@pytest.mark.asyncio
async def test_subscription_transport_late_pump_exception_after_completed_is_logged_not_emitted(
    rate_state: RateLimitStateStore,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A pump exception that lands after the drain already yielded a
    terminal COMPLETED must not produce a second terminal ERROR — one
    terminal event per stream. The late failure is logged instead."""

    async def result_then_raise(**_: Any) -> AsyncIterator[object]:
        yield _success_result_message()
        raise RuntimeError("transport died after the result")

    transport = SubscriptionTransport(
        rate_state=rate_state,
        query_fn=result_then_raise,
        options_cls=_fake_options_cls,
    )
    events: list[TaskEvent] = []
    with caplog.at_level(logging.ERROR, logger="tpasini.recordatio.agents.runner"):
        async for ev in transport.run(
            user_id="alice", slug="weather",
            content="hi", system_prompt="SYS",
            config=AgentConfig(allowed_tools=[]),
        ):
            events.append(ev)

    kinds = [e.kind for e in events]
    assert TaskEventKind.ERROR not in kinds
    assert events[-1].kind is TaskEventKind.COMPLETED
    assert kinds.count(TaskEventKind.COMPLETED) == 1
    # The late exception is still observable in logs.
    assert any("subscription transport raised" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# AgentRunner — the manual-fallback acceptance criterion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_manually_induced_overage_falls_back_to_api_key_path(
    rate_state: RateLimitStateStore,
) -> None:
    """Phase 1 acceptance criterion verbatim: manually inducing
    ``overage_status: rejected`` in the cache causes the next task to
    take the ``ApiKeyTransport`` path, and both the memory briefing
    and the extraction hook still run."""

    # 1. Pre-seed the cache with a depleted-window snapshot (the genuine
    #    fallback trigger; a bare overage=rejected policy flag no longer
    #    routes away from the subscription).
    from datetime import UTC as _UTC
    from datetime import datetime as _dt
    rate_state.upsert(
        user_id="alice", slug="weather",
        status="exceeded", rate_limit_type="five_hour",
        overage_status="rejected", overage_disabled_reason="out_of_credits",
        utilization=1.0, resets_at=str(_dt.now(_UTC).timestamp() + 3600),
        overage_resets_at=None,
    )

    # 2. Wire a subscription transport that would explode if used.
    async def explode_query_fn(**_: Any) -> AsyncIterator[object]:
        raise AssertionError("subscription transport must not be invoked when cache says rejected")
        yield  # pragma: no cover  (unreachable, satisfies async-gen typing)

    sub = SubscriptionTransport(
        rate_state=rate_state,
        query_fn=explode_query_fn,
        options_cls=_fake_options_cls,
    )

    # 3. Wire an API-key transport with a fake Pontifex.
    pf = _FakePontifex(chunks=[
        {"choices": [{"index": 0, "delta": {"content": "hi there"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ])
    api_transport = ApiKeyTransport(pontifex_factory=lambda _model, _key: pf)

    # 4. Briefing + extraction hooks record their arguments so we can
    #    assert they were called with the right inputs on the fallback path.
    briefing_invocations: list[dict] = []

    async def briefing_hook(
        user_id: str, slug: str, content: str, base_system: str,
    ) -> tuple[str | None, dict | None]:
        briefing_invocations.append({
            "user_id": user_id, "slug": slug,
            "content": content, "base_system": base_system,
        })
        # New contract: the hook returns the briefing TEXT; composing it
        # into the system prompt (API-key path) or the rendered prompt
        # (subscription path) is the runner's job.
        return "[MEMORY] briefed", {"injected": True}

    extraction_invocations: list[dict] = []

    async def extraction_hook(user_id: str, slug: str, transcript: dict) -> None:
        extraction_invocations.append({
            "user_id": user_id, "slug": slug, "transcript": transcript,
        })

    runner = AgentRunner(
        subscription=sub,
        api_key=api_transport,
        rate_state=rate_state,
        briefing_hook=briefing_hook,
        extraction_hook=extraction_hook,
    )

    config = AgentConfig(
        system_prompt="you are weather",
        model="anthropic/claude-opus-4-7",
    )
    events: list[TaskEvent] = []
    async for ev in runner.run(
        user_id="alice", slug="weather", content="any rain today?", config=config,
    ):
        events.append(ev)

    # 5a. Transport was API key, evidence: the started event tags it,
    #     and the fake Pontifex received the briefed system prompt.
    started = next(e for e in events if e.kind is TaskEventKind.STARTED)
    assert started.payload["transport"] == "api_key"
    assert pf.received_system == "you are weather\n\n[MEMORY] briefed"
    assert pf.received_messages == [{"role": "user", "content": "any rain today?"}]

    # 5b. Briefing was invoked with the agent's base system prompt.
    assert briefing_invocations == [{
        "user_id": "alice", "slug": "weather",
        "content": "any rain today?", "base_system": "you are weather",
    }]

    # 5c. The wire stream included a ``message`` and ended in ``completed``.
    kinds = [e.kind for e in events]
    assert TaskEventKind.MESSAGE in kinds
    assert kinds[-1] is TaskEventKind.COMPLETED

    # 5d. Extraction ran with the assembled transcript.
    assert len(extraction_invocations) == 1
    extracted = extraction_invocations[0]
    assert extracted["user_id"] == "alice"
    assert extracted["slug"] == "weather"
    assert extracted["transcript"]["response_text"] == "hi there"
    assert extracted["transcript"]["messages"] == [{"role": "user", "content": "any rain today?"}]


@pytest.mark.asyncio
async def test_runner_preserves_lineage_in_extraction_transcript(
    rate_state: RateLimitStateStore,
) -> None:
    """``source_agent_id`` + ``parent_request_id`` passed to ``run`` must
    survive into the transcript dict the extraction hook receives.

    Regression guard for the P0 found in PR review #133 — the hook
    was previously hardcoding ``None`` for both lineage fields, which
    broke cross-agent call-lineage tracking on the agent path.
    """
    # Minimal subscription transport that yields a single text message
    # so the runner reaches the extraction-hook code path.
    sdk = _RecordingSDK(messages=[
        _make_class("AssistantMessage", content=[{"type": "text", "text": "ok"}]),
        _make_class(
            "ResultMessage",
            subtype="success", is_error=False, stop_reason="end_turn",
            duration_ms=1, duration_api_ms=1, num_turns=1, session_id="s",
            total_cost_usd=0.0, usage=None, result=None, structured_output=None,
            model_usage=None, permission_denials=[], deferred_tool_use=None,
            errors=[], api_error_status=None,
        ),
    ])
    sub = SubscriptionTransport(
        rate_state=rate_state, query_fn=sdk.query, options_cls=_fake_options_cls,
    )
    api_transport = ApiKeyTransport(pontifex_factory=lambda _m, _k: _FakePontifex(chunks=[]))

    captured: dict = {}

    async def extraction_hook(user_id: str, slug: str, transcript: dict) -> None:
        captured["transcript"] = transcript

    runner = AgentRunner(
        subscription=sub, api_key=api_transport, rate_state=rate_state,
        briefing_hook=None, extraction_hook=extraction_hook,
    )

    async for _ev in runner.run(
        user_id="alice", slug="weather", content="hi",
        config=AgentConfig(system_prompt="s", model="anthropic/claude-opus-4-7", allowed_tools=[]),
        source_agent_id="curia-leader",
        parent_request_id="recordatio-xyz789",
    ):
        pass

    transcript = captured["transcript"]
    assert transcript["source_agent_id"] == "curia-leader"
    assert transcript["parent_request_id"] == "recordatio-xyz789"
    # Existing fields still present.
    assert transcript["messages"] == [{"role": "user", "content": "hi"}]


@pytest.mark.asyncio
async def test_runner_omits_lineage_keys_when_not_provided(
    rate_state: RateLimitStateStore,
) -> None:
    """When the caller doesn't pass lineage fields the transcript dict
    must not carry stub ``None`` values either — keeps the hook's
    ``transcript.get(...)`` semantics unambiguous (``None`` returned only
    when truly absent, not because we wrote it).
    """
    sdk = _RecordingSDK(messages=[
        _make_class("AssistantMessage", content=[{"type": "text", "text": "ok"}]),
        _make_class(
            "ResultMessage",
            subtype="success", is_error=False, stop_reason="end_turn",
            duration_ms=1, duration_api_ms=1, num_turns=1, session_id="s",
            total_cost_usd=0.0, usage=None, result=None, structured_output=None,
            model_usage=None, permission_denials=[], deferred_tool_use=None,
            errors=[], api_error_status=None,
        ),
    ])
    sub = SubscriptionTransport(
        rate_state=rate_state, query_fn=sdk.query, options_cls=_fake_options_cls,
    )
    api_transport = ApiKeyTransport(pontifex_factory=lambda _m, _k: _FakePontifex(chunks=[]))

    captured: dict = {}

    async def extraction_hook(user_id: str, slug: str, transcript: dict) -> None:
        captured["transcript"] = transcript

    runner = AgentRunner(
        subscription=sub, api_key=api_transport, rate_state=rate_state,
        briefing_hook=None, extraction_hook=extraction_hook,
    )

    async for _ev in runner.run(
        user_id="alice", slug="weather", content="hi",
        config=AgentConfig(system_prompt="s", model="anthropic/claude-opus-4-7", allowed_tools=[]),
    ):
        pass

    transcript = captured["transcript"]
    assert "source_agent_id" not in transcript
    assert "parent_request_id" not in transcript


@pytest.mark.asyncio
async def test_runner_extraction_fires_when_client_stops_on_completed(
    rate_state: RateLimitStateStore,
) -> None:
    """Regression for the agent-path zero-extraction incident (2026-08):
    telos disconnects the instant it sees COMPLETED, which ``aclose()``s this
    generator. Extraction must still run — it is fired BEFORE the terminal
    event is yielded, so a consumer that stops on COMPLETED (and tears the
    generator down) cannot preempt it."""
    sdk = _RecordingSDK(messages=[
        _make_class("AssistantMessage", content=[{"type": "text", "text": "ok"}]),
        _make_class(
            "ResultMessage",
            subtype="success", is_error=False, stop_reason="end_turn",
            duration_ms=1, duration_api_ms=1, num_turns=1, session_id="s",
            total_cost_usd=0.0, usage=None, result=None, structured_output=None,
            model_usage=None, permission_denials=[], deferred_tool_use=None,
            errors=[], api_error_status=None,
        ),
    ])
    sub = SubscriptionTransport(
        rate_state=rate_state, query_fn=sdk.query, options_cls=_fake_options_cls,
    )
    api_transport = ApiKeyTransport(pontifex_factory=lambda _m, _k: _FakePontifex(chunks=[]))

    calls: list[dict] = []

    async def extraction_hook(user_id: str, slug: str, transcript: dict) -> None:
        calls.append({"user_id": user_id, "slug": slug, "transcript": transcript})

    runner = AgentRunner(
        subscription=sub, api_key=api_transport, rate_state=rate_state,
        briefing_hook=None, extraction_hook=extraction_hook,
    )

    gen = runner.run(
        user_id="alice", slug="weather", content="hi",
        config=AgentConfig(system_prompt="s", model="anthropic/claude-opus-4-7", allowed_tools=[]),
    )
    # Consume exactly like telos: stop the moment COMPLETED arrives, then close
    # the stream — reproducing the client disconnect that tears the generator down.
    async for ev in gen:
        if ev.kind is TaskEventKind.COMPLETED:
            break
    await gen.aclose()

    # Extraction fired despite the early stop + teardown (once, not duplicated).
    assert len(calls) == 1
    assert calls[0]["transcript"]["response_text"] == "ok"
    assert calls[0]["transcript"]["messages"] == [{"role": "user", "content": "hi"}]


@pytest.mark.asyncio
async def test_runner_picks_subscription_when_cache_healthy(
    rate_state: RateLimitStateStore,
) -> None:
    """Counter-test to the above: a healthy cache (or no cache) keeps the
    subscription path in play."""

    sdk = _RecordingSDK(messages=[
        _make_class("AssistantMessage", content=[{"type": "text", "text": "ok"}]),
        _make_class(
            "ResultMessage",
            subtype="success", is_error=False, stop_reason="end_turn",
            duration_ms=1, duration_api_ms=1, num_turns=1, session_id="s",
            total_cost_usd=0.0, usage=None, result=None, structured_output=None,
            model_usage=None, permission_denials=[], deferred_tool_use=None,
            errors=[], api_error_status=None,
        ),
    ])
    sub = SubscriptionTransport(
        rate_state=rate_state, query_fn=sdk.query, options_cls=_fake_options_cls,
    )

    async def boom_factory(_model: str, _key: str | None) -> _FakePontifex:
        raise AssertionError("api-key transport must not be invoked when cache is healthy")

    api_transport = ApiKeyTransport(pontifex_factory=boom_factory)

    runner = AgentRunner(
        subscription=sub, api_key=api_transport, rate_state=rate_state,
        briefing_hook=None, extraction_hook=None,
    )
    events: list[TaskEvent] = []
    async for ev in runner.run(
        user_id="alice", slug="weather", content="hi",
        config=AgentConfig(system_prompt="s", model="anthropic/claude-opus-4-7", allowed_tools=[]),
    ):
        events.append(ev)
    started = next(e for e in events if e.kind is TaskEventKind.STARTED)
    assert started.payload["transport"] == "subscription"


# ---------------------------------------------------------------------------
# Regression coverage: subagent error containment + allowed_tools=None inherit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subscription_transport_malformed_subagent_yields_error_event(
    rate_state: RateLimitStateStore,
) -> None:
    """A subagent dict missing the required ``name`` key used to escape
    the SSE stream as an uncaught ``KeyError`` (the dict-comprehension
    sat outside the try/except). The runner must now catch the failure
    and surface a single structured ``error`` event instead.
    """

    async def must_not_run(**_: Any) -> AsyncIterator[object]:
        raise AssertionError("query_fn must not be reached when options construction fails")
        yield  # pragma: no cover

    transport = SubscriptionTransport(
        rate_state=rate_state,
        query_fn=must_not_run,
        options_cls=_fake_options_cls,
        agent_def_cls=_FakeAgentDef,
    )
    config = AgentConfig(
        allowed_tools=[],
        subagents=[{"description": "missing name", "prompt": "p"}],
    )
    events: list[TaskEvent] = []
    async for ev in transport.run(
        user_id="alice", slug="weather",
        content="hi", system_prompt="SYSTEM",
        config=config,
    ):
        events.append(ev)

    kinds = [e.kind for e in events]
    # The runner-level ``started`` still fires before the construction
    # try-block; the failure then surfaces as a single ``error`` event
    # and the generator returns cleanly. Validation now happens in
    # ``_validate_and_build_agents`` which raises ValueError with a
    # structured message, surfaced under the dedicated
    # ``invalid_subagent_config`` reason.
    assert kinds[0] is TaskEventKind.STARTED
    assert TaskEventKind.ERROR in kinds
    err = next(e for e in events if e.kind is TaskEventKind.ERROR)
    assert err.payload["reason"] == "invalid_subagent_config"
    assert "missing 'name'" in err.payload["message"]


@pytest.mark.asyncio
async def test_subscription_transport_omits_allowed_tools_when_config_is_none(
    rate_state: RateLimitStateStore,
) -> None:
    """``AgentConfig.allowed_tools`` defaults to ``None`` ("inherit SDK
    defaults"). Passing ``allowed_tools=[]`` to ``ClaudeAgentOptions``
    instead means "block every tool" — wrong semantics. The runner
    must omit the kwarg entirely in the ``None`` case.
    """

    sdk = _RecordingSDK(messages=[
        _make_class(
            "ResultMessage",
            subtype="success", is_error=False, stop_reason="end_turn",
            duration_ms=1, duration_api_ms=1, num_turns=1, session_id="s",
            total_cost_usd=0.0, usage=None, result=None, structured_output=None,
            model_usage=None, permission_denials=[], deferred_tool_use=None,
            errors=[], api_error_status=None,
        ),
    ])

    # Spy on the options constructor: record kwargs verbatim so we can
    # assert ``allowed_tools`` was NOT among them.
    captured_kwargs: dict[str, Any] = {}

    def spy_options_cls(**kwargs: Any) -> SimpleNamespace:
        captured_kwargs.update(kwargs)
        return SimpleNamespace(**kwargs)

    transport = SubscriptionTransport(
        rate_state=rate_state,
        query_fn=sdk.query,
        options_cls=spy_options_cls,
    )
    async for _ in transport.run(
        user_id="alice", slug="weather",
        content="hi", system_prompt="SYSTEM",
        config=AgentConfig(model="claude-opus-4-7"),  # allowed_tools defaults to None
    ):
        pass

    assert "allowed_tools" not in captured_kwargs, (
        "allowed_tools must be omitted (not passed as []) when the config has None"
    )
    # And the rest of the static kwargs still made it through.
    assert captured_kwargs["system_prompt"] == "SYSTEM"
    assert captured_kwargs["permission_mode"] == "bypassPermissions"


@pytest.mark.asyncio
async def test_subscription_transport_passes_allowed_tools_when_config_specifies(
    rate_state: RateLimitStateStore,
) -> None:
    """Counterpart to the above: when the config DOES pin a tool list
    (including the empty list — explicit deny), the kwarg must be
    forwarded so the user's intent is honoured.
    """

    sdk = _RecordingSDK(messages=[
        _make_class(
            "ResultMessage",
            subtype="success", is_error=False, stop_reason="end_turn",
            duration_ms=1, duration_api_ms=1, num_turns=1, session_id="s",
            total_cost_usd=0.0, usage=None, result=None, structured_output=None,
            model_usage=None, permission_denials=[], deferred_tool_use=None,
            errors=[], api_error_status=None,
        ),
    ])
    captured_kwargs: dict[str, Any] = {}

    def spy_options_cls(**kwargs: Any) -> SimpleNamespace:
        captured_kwargs.update(kwargs)
        return SimpleNamespace(**kwargs)

    transport = SubscriptionTransport(
        rate_state=rate_state,
        query_fn=sdk.query,
        options_cls=spy_options_cls,
    )
    async for _ in transport.run(
        user_id="alice", slug="weather",
        content="hi", system_prompt="SYSTEM",
        config=AgentConfig(model="claude-opus-4-7", allowed_tools=["WebSearch", "Read"]),
    ):
        pass

    assert captured_kwargs.get("allowed_tools") == ["WebSearch", "Read"]


# ---------------------------------------------------------------------------
# mcp_servers wiring: list-of-dicts config → SDK dict keyed by server name
# ---------------------------------------------------------------------------


def _success_result_message() -> object:
    return _make_class(
        "ResultMessage",
        subtype="success", is_error=False, stop_reason="end_turn",
        duration_ms=1, duration_api_ms=1, num_turns=1, session_id="s",
        total_cost_usd=0.0, usage=None, result=None, structured_output=None,
        model_usage=None, permission_denials=[], deferred_tool_use=None,
        errors=[], api_error_status=None,
    )


@pytest.mark.asyncio
async def test_subscription_transport_passes_mcp_servers_as_dict_keyed_by_name(
    rate_state: RateLimitStateStore,
) -> None:
    """``AgentConfig.mcp_servers`` is stored as a LIST of dicts carrying a
    ``name`` field; ``ClaudeAgentOptions.mcp_servers`` expects a DICT keyed
    by server name whose values are the SDK ``McpServerConfig`` shapes
    (``name`` stripped). The runner used to drop the field entirely — the
    telos server was never registered and, because ``allowed_tools`` is a
    whitelist, the agent ran with zero tools.
    """

    sdk = _RecordingSDK(messages=[_success_result_message()])
    captured_kwargs: dict[str, Any] = {}

    def spy_options_cls(**kwargs: Any) -> SimpleNamespace:
        captured_kwargs.update(kwargs)
        return SimpleNamespace(**kwargs)

    transport = SubscriptionTransport(
        rate_state=rate_state,
        query_fn=sdk.query,
        options_cls=spy_options_cls,
    )
    config = AgentConfig(
        model="claude-opus-4-7",
        allowed_tools=["mcp__telos__list_goals"],
        mcp_servers=[
            {
                "name": "telos",
                "type": "http",
                "url": "https://telos.example.com/mcp",
                "headers": {"Authorization": "Bearer secret"},
            },
            {
                "name": "local",
                "type": "stdio",
                "command": "my-mcp",
                "args": ["--flag"],
            },
        ],
    )
    async for _ in transport.run(
        user_id="alice", slug="weather",
        content="hi", system_prompt="SYSTEM",
        config=config,
    ):
        pass

    assert captured_kwargs["mcp_servers"] == {
        "telos": {
            "type": "http",
            "url": "https://telos.example.com/mcp",
            "headers": {"Authorization": "Bearer secret"},
        },
        "local": {
            "type": "stdio",
            "command": "my-mcp",
            "args": ["--flag"],
        },
    }
    # The whitelist still rides alongside, now resolvable.
    assert captured_kwargs["allowed_tools"] == ["mcp__telos__list_goals"]


@pytest.mark.asyncio
async def test_subscription_transport_omits_mcp_servers_when_config_empty(
    rate_state: RateLimitStateStore,
) -> None:
    """No configured servers → no ``mcp_servers`` kwarg at all, so the SDK
    falls back to its own default (empty dict) rather than us passing a
    redundant empty mapping.
    """

    sdk = _RecordingSDK(messages=[_success_result_message()])
    captured_kwargs: dict[str, Any] = {}

    def spy_options_cls(**kwargs: Any) -> SimpleNamespace:
        captured_kwargs.update(kwargs)
        return SimpleNamespace(**kwargs)

    transport = SubscriptionTransport(
        rate_state=rate_state,
        query_fn=sdk.query,
        options_cls=spy_options_cls,
    )
    async for _ in transport.run(
        user_id="alice", slug="weather",
        content="hi", system_prompt="SYSTEM",
        config=AgentConfig(model="claude-opus-4-7"),  # mcp_servers defaults to []
    ):
        pass

    assert "mcp_servers" not in captured_kwargs


@pytest.mark.asyncio
async def test_subscription_transport_malformed_mcp_server_yields_error_event(
    rate_state: RateLimitStateStore,
) -> None:
    """An mcp_servers entry without a ``name`` cannot be keyed into the SDK
    dict. It must surface as a structured ``error`` event (same containment
    as malformed subagents), not an uncaught exception on the SSE stream.
    """

    async def must_not_run(**_: Any) -> AsyncIterator[object]:
        raise AssertionError("query_fn must not be reached when options construction fails")
        yield  # pragma: no cover

    transport = SubscriptionTransport(
        rate_state=rate_state,
        query_fn=must_not_run,
        options_cls=_fake_options_cls,
    )
    config = AgentConfig(
        mcp_servers=[{"type": "http", "url": "https://telos.example.com/mcp"}],
    )
    events: list[TaskEvent] = []
    async for ev in transport.run(
        user_id="alice", slug="weather",
        content="hi", system_prompt="SYSTEM",
        config=config,
    ):
        events.append(ev)

    kinds = [e.kind for e in events]
    assert kinds[0] is TaskEventKind.STARTED
    assert TaskEventKind.ERROR in kinds
    err = next(e for e in events if e.kind is TaskEventKind.ERROR)
    assert err.payload["reason"] == "sdk_exception"
    assert err.payload["exception_type"] == "ValueError"
    assert "name" in err.payload["message"]


@pytest.mark.asyncio
async def test_subscription_transport_duplicate_mcp_server_names_last_wins(
    rate_state: RateLimitStateStore,
) -> None:
    """Two entries with the same ``name``: the runner keeps the last one
    (dict semantics) and logs a warning rather than failing the task.
    """

    sdk = _RecordingSDK(messages=[_success_result_message()])
    captured_kwargs: dict[str, Any] = {}

    def spy_options_cls(**kwargs: Any) -> SimpleNamespace:
        captured_kwargs.update(kwargs)
        return SimpleNamespace(**kwargs)

    transport = SubscriptionTransport(
        rate_state=rate_state,
        query_fn=sdk.query,
        options_cls=spy_options_cls,
    )
    config = AgentConfig(
        mcp_servers=[
            {"name": "telos", "type": "http", "url": "https://old.example.com/mcp"},
            {"name": "telos", "type": "http", "url": "https://new.example.com/mcp"},
        ],
    )
    async for _ in transport.run(
        user_id="alice", slug="weather",
        content="hi", system_prompt="SYSTEM",
        config=config,
    ):
        pass

    assert captured_kwargs["mcp_servers"] == {
        "telos": {"type": "http", "url": "https://new.example.com/mcp"},
    }


# ---------------------------------------------------------------------------
# ApiKeyTransport fallback: capability degradation is surfaced, not faked
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_api_key_transport_surfaces_degraded_capabilities(
    rate_state: RateLimitStateStore,
) -> None:
    """Pontifex has no MCP client / tool executor / subagent support. When
    the config carries any of those, the fallback's ``started`` event must
    list them under ``degraded_capabilities`` so the loss is observable.
    """

    pf = _FakePontifex(chunks=[
        {"choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"}]},
    ])
    transport = ApiKeyTransport(pontifex_factory=lambda model, key: pf)
    config = AgentConfig(
        model="anthropic/claude-opus-4-7",
        mcp_servers=[{"name": "telos", "type": "http", "url": "https://t.example.com/mcp"}],
        allowed_tools=["mcp__telos__list_goals"],
        subagents=[{"name": "sub", "description": "d", "prompt": "p"}],
    )
    events: list[TaskEvent] = []
    async for ev in transport.run(
        user_id="alice", slug="weather",
        content="hi", system_prompt="SYSTEM",
        config=config,
    ):
        events.append(ev)

    started = next(e for e in events if e.kind is TaskEventKind.STARTED)
    assert started.payload["degraded_capabilities"] == [
        "mcp_servers", "allowed_tools", "subagents",
    ]


@pytest.mark.asyncio
async def test_api_key_transport_omits_degraded_capabilities_when_none(
    rate_state: RateLimitStateStore,
) -> None:
    """A config with no SDK-only capabilities loses nothing on fallback —
    the ``started`` payload must not carry the key at all.
    """

    pf = _FakePontifex(chunks=[
        {"choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"}]},
    ])
    transport = ApiKeyTransport(pontifex_factory=lambda model, key: pf)
    events: list[TaskEvent] = []
    async for ev in transport.run(
        user_id="alice", slug="weather",
        content="hi", system_prompt="SYSTEM",
        config=AgentConfig(model="anthropic/claude-opus-4-7"),
    ):
        events.append(ev)

    started = next(e for e in events if e.kind is TaskEventKind.STARTED)
    assert "degraded_capabilities" not in started.payload


@pytest.mark.asyncio
async def test_api_key_transport_attaches_openai_wire_usage_to_completed() -> None:
    """Pontifex's final chunk carries an OpenAI-wire ``usage`` block
    (prompt_tokens/completion_tokens + Anthropic cache passthrough). The
    ``completed`` event must carry it verbatim under ``usage`` — the same
    payload contract telos parses on the subscription path.
    """

    usage = {
        "prompt_tokens": 900,
        "completion_tokens": 120,
        "total_tokens": 1020,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 800,
    }
    pf = _FakePontifex(chunks=[
        {"choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": usage},
    ])
    transport = ApiKeyTransport(pontifex_factory=lambda model, key: pf)
    events: list[TaskEvent] = []
    async for ev in transport.run(
        user_id="alice", slug="weather",
        content="hi", system_prompt="SYSTEM",
        config=AgentConfig(model="anthropic/claude-opus-4-7"),
    ):
        events.append(ev)

    completed = next(e for e in events if e.kind is TaskEventKind.COMPLETED)
    assert completed.payload["usage"] == usage


@pytest.mark.asyncio
async def test_api_key_transport_omits_usage_when_upstream_never_reports_it() -> None:
    """A stream with no ``usage`` on any chunk → the ``completed`` payload
    must not carry the key at all (consumers fall back to estimates)."""

    pf = _FakePontifex(chunks=[
        {"choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"}]},
    ])
    transport = ApiKeyTransport(pontifex_factory=lambda model, key: pf)
    events: list[TaskEvent] = []
    async for ev in transport.run(
        user_id="alice", slug="weather",
        content="hi", system_prompt="SYSTEM",
        config=AgentConfig(model="anthropic/claude-opus-4-7"),
    ):
        events.append(ev)

    completed = next(e for e in events if e.kind is TaskEventKind.COMPLETED)
    assert "usage" not in completed.payload


# ---------------------------------------------------------------------------
# ApiKeyTransport: faithful finish_reason + explicit max_tokens (finding #11)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_api_key_transport_mirrors_upstream_finish_reason() -> None:
    """The ``completed`` event's ``stop_reason`` must mirror the upstream
    ``finish_reason`` verbatim (here: "length", i.e. truncation by token
    budget) rather than a hardcoded "stop"."""

    pf = _FakePontifex(chunks=[
        {"choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]},
    ])
    transport = ApiKeyTransport(pontifex_factory=lambda model, key: pf)
    events: list[TaskEvent] = []
    async for ev in transport.run(
        user_id="alice", slug="weather",
        content="hi", system_prompt="SYSTEM",
        config=AgentConfig(model="anthropic/claude-opus-4-7"),
    ):
        events.append(ev)

    completed = next(e for e in events if e.kind is TaskEventKind.COMPLETED)
    assert completed.payload["stop_reason"] == "length"


@pytest.mark.asyncio
async def test_api_key_transport_content_filter_emits_error_not_completed() -> None:
    """``finish_reason == "content_filter"`` means the provider blocked or
    censored the response. Reporting that as a clean ``completed`` would let
    consumers treat a censored answer as a full one — it must surface as a
    structured ``error`` instead."""

    pf = _FakePontifex(chunks=[
        {"choices": [{"index": 0, "delta": {"content": "par"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "content_filter"}]},
    ])
    transport = ApiKeyTransport(pontifex_factory=lambda model, key: pf)
    events: list[TaskEvent] = []
    async for ev in transport.run(
        user_id="alice", slug="weather",
        content="hi", system_prompt="SYSTEM",
        config=AgentConfig(model="anthropic/claude-opus-4-7"),
    ):
        events.append(ev)

    kinds = [e.kind for e in events]
    assert TaskEventKind.COMPLETED not in kinds
    assert events[-1].kind is TaskEventKind.ERROR
    assert events[-1].payload["reason"] == "content_filter"
    assert events[-1].payload["stop_reason"] == "content_filter"


@pytest.mark.asyncio
async def test_api_key_transport_passes_explicit_max_tokens() -> None:
    """ApiKeyTransport must pass the shared default token budget explicitly
    rather than silently relying on Pontifex's own default."""

    pf = _FakePontifex(chunks=[
        {"choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"}]},
    ])
    transport = ApiKeyTransport(pontifex_factory=lambda model, key: pf)
    async for _ in transport.run(
        user_id="alice", slug="weather",
        content="hi", system_prompt="SYSTEM",
        config=AgentConfig(model="anthropic/claude-opus-4-7"),
    ):
        pass

    assert pf.received_kwargs is not None
    assert pf.received_kwargs.get("max_tokens") == 8096


# ---------------------------------------------------------------------------
# model normalization: provider-prefixed model names on the subscription path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subscription_transport_strips_anthropic_provider_prefix(
    rate_state: RateLimitStateStore,
) -> None:
    """telos seeds ``AgentConfig.model`` verbatim from its MODEL env var,
    which uses the ``provider/model`` syntax (``anthropic/claude-…``). The
    SDK only understands bare model names — the prefixed form errors with
    "model may not exist". The transport must strip the provider segment
    before handing the model to ``ClaudeAgentOptions``.
    """

    sdk = _RecordingSDK(messages=[_success_result_message()])
    captured_kwargs: dict[str, Any] = {}

    def spy_options_cls(**kwargs: Any) -> SimpleNamespace:
        captured_kwargs.update(kwargs)
        return SimpleNamespace(**kwargs)

    transport = SubscriptionTransport(
        rate_state=rate_state,
        query_fn=sdk.query,
        options_cls=spy_options_cls,
    )
    events: list[TaskEvent] = []
    async for ev in transport.run(
        user_id="alice", slug="weather",
        content="hi", system_prompt="SYSTEM",
        config=AgentConfig(model="anthropic/claude-sonnet-4-6"),
    ):
        events.append(ev)

    assert captured_kwargs["model"] == "claude-sonnet-4-6"
    # The started event reports what's actually passed to the SDK.
    assert events[0].kind is TaskEventKind.STARTED
    assert events[0].payload["model"] == "claude-sonnet-4-6"
    assert events[-1].kind is TaskEventKind.COMPLETED


@pytest.mark.asyncio
async def test_subscription_transport_passes_bare_model_unchanged(
    rate_state: RateLimitStateStore,
) -> None:
    """A model without a provider prefix goes through verbatim — the
    normalization must only strip a leading ``provider/`` segment, never
    rewrite an already-bare name.
    """

    sdk = _RecordingSDK(messages=[_success_result_message()])
    captured_kwargs: dict[str, Any] = {}

    def spy_options_cls(**kwargs: Any) -> SimpleNamespace:
        captured_kwargs.update(kwargs)
        return SimpleNamespace(**kwargs)

    transport = SubscriptionTransport(
        rate_state=rate_state,
        query_fn=sdk.query,
        options_cls=spy_options_cls,
    )
    async for _ in transport.run(
        user_id="alice", slug="weather",
        content="hi", system_prompt="SYSTEM",
        config=AgentConfig(model="claude-sonnet-4-6"),
    ):
        pass

    assert captured_kwargs["model"] == "claude-sonnet-4-6"


@pytest.mark.asyncio
async def test_subscription_transport_rejects_non_anthropic_provider(
    rate_state: RateLimitStateStore,
) -> None:
    """The subscription path bills against Claude subscription credits —
    a non-anthropic provider prefix (per ``resolve_model``'s registry) can
    never work there. Surface a structured ``error`` event instead of
    letting the SDK fail cryptically; the SDK must never be invoked.
    """

    sdk = _RecordingSDK(messages=[_success_result_message()])
    transport = SubscriptionTransport(
        rate_state=rate_state,
        query_fn=sdk.query,
        options_cls=_fake_options_cls,
    )
    events: list[TaskEvent] = []
    async for ev in transport.run(
        user_id="alice", slug="weather",
        content="hi", system_prompt="SYSTEM",
        config=AgentConfig(model="ollama/qwen2.5:14b"),
    ):
        events.append(ev)

    assert len(events) == 1
    assert events[0].kind is TaskEventKind.ERROR
    assert events[0].payload["reason"] == "unsupported_model_provider"
    assert events[0].payload["model"] == "ollama/qwen2.5:14b"
    assert events[0].payload["transport"] == "subscription"
    # The SDK was never reached.
    assert sdk.last_options is None


# ---------------------------------------------------------------------------
# AgentRunner — BRIEFING event emission
# ---------------------------------------------------------------------------


def _single_turn_sdk() -> _RecordingSDK:
    """A subscription SDK fake that yields one text message + clean result."""
    return _RecordingSDK(messages=[
        _make_class("AssistantMessage", content=[{"type": "text", "text": "ok"}]),
        _make_class(
            "ResultMessage",
            subtype="success", is_error=False, stop_reason="end_turn",
            duration_ms=1, duration_api_ms=1, num_turns=1, session_id="s",
            total_cost_usd=0.0, usage=None, result=None, structured_output=None,
            model_usage=None, permission_denials=[], deferred_tool_use=None,
            errors=[], api_error_status=None,
        ),
    ])


def _runner_with(
    rate_state: RateLimitStateStore,
    sdk: _RecordingSDK,
    briefing_hook: Any,
) -> AgentRunner:
    sub = SubscriptionTransport(
        rate_state=rate_state, query_fn=sdk.query, options_cls=_fake_options_cls,
    )
    api_transport = ApiKeyTransport(pontifex_factory=lambda _m, _k: _FakePontifex(chunks=[]))
    return AgentRunner(
        subscription=sub, api_key=api_transport, rate_state=rate_state,
        briefing_hook=briefing_hook, extraction_hook=None,
    )


# Legacy-shaped briefing payload — the exact dict ``prompting.build_system``
# returns and the chat path serializes into its ``agentmem.briefing`` event.
_LEGACY_BRIEFING = {
    "units": [{"type": "semantic", "content": "The user prefers Python"}],
    "entities": [{"label": "Tommaso", "wikidata_id": None, "description": "the user"}],
    "relations": [{"subject": "Tommaso", "predicate": "lives_in", "object": "Springfield"}],
    "topic_names": ["preferences"],
}


@pytest.mark.asyncio
async def test_runner_emits_briefing_event_first_with_legacy_payload(
    rate_state: RateLimitStateStore,
) -> None:
    """When the briefing hook returns data, the runner yields a BRIEFING
    event *before* any transport event, carrying the same dict the legacy
    chat path puts in its ``agentmem.briefing`` SSE event."""

    async def briefing_hook(
        user_id: str, slug: str, content: str, base_system: str,
    ) -> tuple[str | None, dict | None]:
        return "[MEMORY] briefed", dict(_LEGACY_BRIEFING)

    sdk = _single_turn_sdk()
    runner = _runner_with(rate_state, sdk, briefing_hook)

    events: list[TaskEvent] = []
    async for ev in runner.run(
        user_id="alice", slug="weather", content="hi",
        config=AgentConfig(system_prompt="base", model="anthropic/claude-opus-4-7", allowed_tools=[]),
    ):
        events.append(ev)

    assert events[0].kind is TaskEventKind.BRIEFING
    assert events[0].payload == _LEGACY_BRIEFING
    # Briefing precedes every transport event (started/message/completed).
    kinds = [e.kind for e in events]
    assert kinds.index(TaskEventKind.BRIEFING) < kinds.index(TaskEventKind.STARTED)
    assert kinds[-1] is TaskEventKind.COMPLETED
    # Briefing relocation: the system prompt stays the BASE prompt (it is
    # sealed at SDK connect time, so warm spares could never carry a
    # per-turn briefing there); the briefing rides in the rendered prompt
    # instead, between the (absent here) history block and the message.
    assert sdk.last_options.system_prompt == "base"
    # build_system owns the block tags now; the runner inserts the briefing
    # verbatim (no <memory_briefing> wrapper).
    assert sdk.last_prompt == "[MEMORY] briefed\n\nhi"


@pytest.mark.asyncio
async def test_runner_no_briefing_event_when_hook_returns_none(
    rate_state: RateLimitStateStore,
) -> None:
    async def briefing_hook(
        user_id: str, slug: str, content: str, base_system: str,
    ) -> tuple[str | None, dict | None]:
        return None, None

    sdk = _single_turn_sdk()
    runner = _runner_with(rate_state, sdk, briefing_hook)

    events: list[TaskEvent] = []
    async for ev in runner.run(
        user_id="alice", slug="weather", content="hi",
        config=AgentConfig(system_prompt="base", model="anthropic/claude-opus-4-7", allowed_tools=[]),
    ):
        events.append(ev)

    kinds = [e.kind for e in events]
    assert TaskEventKind.BRIEFING not in kinds
    assert kinds[-1] is TaskEventKind.COMPLETED


@pytest.mark.asyncio
async def test_runner_briefing_hook_failure_degrades_to_no_briefing(
    rate_state: RateLimitStateStore,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A raising briefing hook must not abort the turn — mirror the legacy
    chat path, which log.exception's retrieval failures and proceeds
    without a briefing. The base system prompt is used as-is."""

    async def briefing_hook(
        user_id: str, slug: str, content: str, base_system: str,
    ) -> tuple[str | None, dict | None]:
        raise RuntimeError("retrieval store on fire")

    sdk = _single_turn_sdk()
    runner = _runner_with(rate_state, sdk, briefing_hook)

    events: list[TaskEvent] = []
    with caplog.at_level(logging.ERROR, logger="tpasini.recordatio.agents.runner"):
        async for ev in runner.run(
            user_id="alice", slug="weather", content="hi",
            config=AgentConfig(system_prompt="base", model="anthropic/claude-opus-4-7", allowed_tools=[]),
        ):
            events.append(ev)

    kinds = [e.kind for e in events]
    assert TaskEventKind.BRIEFING not in kinds
    assert TaskEventKind.ERROR not in kinds
    assert kinds[-1] is TaskEventKind.COMPLETED
    # Un-briefed base prompt reached the transport.
    assert sdk.last_options.system_prompt == "base"
    assert any("briefing hook failed" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Conversation history — the per-turn-amnesia P0 fix
# ---------------------------------------------------------------------------
#
# Telos ships the full conversation history in the POST body (``messages``,
# OpenAI-style role/content dicts, latest user message last). The runner must
# thread it to both transports: the subscription path renders prior turns
# into a transcript preamble (the SDK session is fresh every turn), the
# API-key path passes the array natively on the chat-completions wire.
# Briefing and extraction stay pinned to the LATEST user message.


def test_render_prompt_no_messages_is_identity() -> None:
    assert _render_prompt_with_history("hi", None) == "hi"
    assert _render_prompt_with_history("hi", []) == "hi"


def test_render_prompt_single_latest_message_is_identity() -> None:
    """A messages list holding only the latest user message carries no
    history — the prompt must be byte-identical to the legacy single-turn
    shape."""
    assert _render_prompt_with_history(
        "hi", [{"role": "user", "content": "hi"}],
    ) == "hi"


def test_render_prompt_wraps_history_and_appends_latest() -> None:
    prompt = _render_prompt_with_history(
        "and tomorrow?",
        [
            {"role": "user", "content": "any rain today?"},
            {"role": "assistant", "content": "no rain expected"},
            {"role": "user", "content": "and tomorrow?"},
        ],
    )
    assert prompt == (
        "<conversation_history>\n"
        "user: any rain today?\n"
        "assistant: no rain expected\n"
        "</conversation_history>\n\n"
        "and tomorrow?"
    )


def test_render_prompt_flattens_block_list_content() -> None:
    """Image turns store content as a block list (telos/agent.py). Text
    parts are kept; non-text blocks degrade to an attachment note instead
    of shipping base64 into the prompt."""
    prompt = _render_prompt_with_history(
        "what about it?",
        [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxxx"}},
                {"type": "text", "text": "look at this chart"},
            ]},
            {"role": "assistant", "content": "interesting trend"},
            {"role": "user", "content": "what about it?"},
        ],
    )
    assert "user: [image_url attached] look at this chart" in prompt
    assert "assistant: interesting trend" in prompt
    assert "base64,xxxx" not in prompt
    assert prompt.endswith("what about it?")


def test_render_prompt_caps_history_dropping_oldest_first() -> None:
    """History beyond the char budget is trimmed OLDEST-first; the newest
    prior turns survive and the latest message (outside the block) is
    never dropped."""
    filler = "x" * 500
    history = [
        {"role": "user", "content": f"turn-{i}: {filler}"} for i in range(40)
    ]
    history.append({"role": "user", "content": "latest question"})
    prompt = _render_prompt_with_history("latest question", history)

    rendered_history = prompt.split("</conversation_history>")[0]
    assert len(rendered_history) <= _HISTORY_CHAR_BUDGET + len("<conversation_history>\n") + 1
    # Newest prior turn survived; the oldest did not.
    assert "turn-39" in prompt
    assert "turn-0:" not in prompt
    assert prompt.endswith("latest question")


def test_render_prompt_truncates_oversized_newest_line_instead_of_empty() -> None:
    """A newest prior line that alone exceeds the whole budget used to end
    the walk with an EMPTY transcript — an agent whose previous reply was a
    long report (the daily-routine shape) then ran with no history at all.
    It must instead survive as a budget-sized head + truncation marker."""
    huge = "assistant reply " * 2000  # ≫ _HISTORY_CHAR_BUDGET chars
    history = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": huge},
        {"role": "user", "content": "latest"},
    ]
    prompt = _render_prompt_with_history("latest", history)

    assert "<conversation_history>" in prompt
    transcript = prompt.split("</conversation_history>")[0]
    assert len(transcript) <= _HISTORY_CHAR_BUDGET + len("<conversation_history>\n") + 1
    assert "assistant: assistant reply" in transcript
    assert transcript.rstrip().endswith("…[truncated]")
    # The budget was spent on the newest line — older lines still drop.
    assert "old question" not in transcript
    assert prompt.endswith("latest")


def test_render_prompt_briefing_only() -> None:
    """With no history the briefing (verbatim, already block-tagged by
    build_system) directly precedes the message."""
    assert _render_prompt_with_history(
        "hi", None, briefing="<relevant_memory>\n[MEMORY] facts\n[/MEMORY]\n</relevant_memory>",
    ) == (
        "<relevant_memory>\n[MEMORY] facts\n[/MEMORY]\n</relevant_memory>\n\nhi"
    )


def test_render_prompt_orders_history_then_briefing_then_message() -> None:
    """Prompt layout contract: history → memory → message. The briefing
    sits ADJACENT to the question it was retrieved for (lost-in-the-middle
    avoidance; long-context guidance puts task-relevant material near the
    end), and the latest message keeps the final slot."""
    prompt = _render_prompt_with_history(
        "and tomorrow?",
        [
            {"role": "user", "content": "any rain today?"},
            {"role": "assistant", "content": "no rain expected"},
            {"role": "user", "content": "and tomorrow?"},
        ],
        briefing="<user_profile>\nuser lives in Sunnydale\n</user_profile>",
    )
    assert prompt == (
        "<conversation_history>\n"
        "user: any rain today?\n"
        "assistant: no rain expected\n"
        "</conversation_history>\n\n"
        "<user_profile>\n"
        "user lives in Sunnydale\n"
        "</user_profile>\n\n"
        "and tomorrow?"
    )


def test_render_prompt_briefing_does_not_consume_history_budget() -> None:
    """The 8K char cap applies to the history block ONLY — a huge briefing
    must not push history turns out (and vice versa the history cap never
    trims the briefing)."""
    filler = "x" * 500
    history = [{"role": "user", "content": f"turn-{i}: {filler}"} for i in range(40)]
    history.append({"role": "user", "content": "latest question"})
    briefing = "[MEMORY] " + "m" * (_HISTORY_CHAR_BUDGET * 2)

    with_briefing = _render_prompt_with_history("latest question", history, briefing)
    without_briefing = _render_prompt_with_history("latest question", history)

    # Identical history survives in both renderings.
    history_block_with = with_briefing.split("</conversation_history>")[0]
    history_block_without = without_briefing.split("</conversation_history>")[0]
    assert history_block_with == history_block_without
    # The briefing itself is intact and in the middle slot.
    assert briefing in with_briefing
    assert with_briefing.index("</conversation_history>") < with_briefing.index(briefing)
    assert with_briefing.endswith("latest question")


@pytest.mark.asyncio
async def test_subscription_transport_renders_history_into_sdk_prompt(
    rate_state: RateLimitStateStore,
) -> None:
    sdk = _RecordingSDK(messages=[_success_result_message()])
    transport = SubscriptionTransport(
        rate_state=rate_state, query_fn=sdk.query, options_cls=_fake_options_cls,
    )
    async for _ in transport.run(
        user_id="alice", slug="weather",
        content="and tomorrow?", system_prompt="SYS+BRIEFING",
        config=AgentConfig(allowed_tools=[]),
        messages=[
            {"role": "user", "content": "any rain today?"},
            {"role": "assistant", "content": "no rain expected"},
            {"role": "user", "content": "and tomorrow?"},
        ],
    ):
        pass

    assert sdk.last_prompt is not None
    assert sdk.last_prompt.startswith("<conversation_history>\n")
    assert "user: any rain today?" in sdk.last_prompt
    assert "assistant: no rain expected" in sdk.last_prompt
    assert sdk.last_prompt.endswith("and tomorrow?")
    # The system prompt is NOT duplicated into the prompt body.
    assert "SYS+BRIEFING" not in sdk.last_prompt
    assert sdk.last_options.system_prompt == "SYS+BRIEFING"


@pytest.mark.asyncio
async def test_subscription_transport_without_messages_prompt_unchanged(
    rate_state: RateLimitStateStore,
) -> None:
    """No ``messages`` ⇒ the legacy single-prompt behaviour, byte-identical."""
    sdk = _RecordingSDK(messages=[_success_result_message()])
    transport = SubscriptionTransport(
        rate_state=rate_state, query_fn=sdk.query, options_cls=_fake_options_cls,
    )
    async for _ in transport.run(
        user_id="alice", slug="weather",
        content="hi", system_prompt="SYS",
        config=AgentConfig(allowed_tools=[]),
    ):
        pass

    assert sdk.last_prompt == "hi"


@pytest.mark.asyncio
async def test_api_key_transport_passes_messages_natively_on_the_wire() -> None:
    """Pontifex speaks chat-completions: the history array rides the wire
    as-is — block-list content (image turns) included, no transcript
    rendering."""
    pf = _FakePontifex(chunks=[
        {"choices": [{"index": 0, "delta": {"content": "sunny"}, "finish_reason": "stop"}]},
    ])
    transport = ApiKeyTransport(pontifex_factory=lambda _m, _k: pf)
    history = [
        {"role": "user", "content": "any rain today?"},
        {"role": "assistant", "content": "no rain expected"},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxxx"}},
            {"type": "text", "text": "and this?"},
        ]},
    ]
    async for _ in transport.run(
        user_id="alice", slug="weather",
        content="and this?", system_prompt="SYS",
        config=AgentConfig(model="anthropic/claude-opus-4-7"),
        messages=history,
    ):
        pass

    assert pf.received_messages == history
    assert pf.received_system == "SYS"


@pytest.mark.asyncio
async def test_api_key_transport_without_messages_keeps_single_turn_shape() -> None:
    pf = _FakePontifex(chunks=[
        {"choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]},
    ])
    transport = ApiKeyTransport(pontifex_factory=lambda _m, _k: pf)
    async for _ in transport.run(
        user_id="alice", slug="weather",
        content="hi", system_prompt="SYS",
        config=AgentConfig(model="anthropic/claude-opus-4-7"),
    ):
        pass

    assert pf.received_messages == [{"role": "user", "content": "hi"}]


@pytest.mark.asyncio
async def test_runner_threads_messages_but_briefing_and_extraction_stay_latest_only(
    rate_state: RateLimitStateStore,
) -> None:
    """The full-history thread-through must NOT leak into briefing retrieval
    (query stays the latest user message) or extraction (re-extracting the
    whole history every turn would duplicate memories already captured on
    earlier turns)."""
    sdk = _single_turn_sdk()
    sub = SubscriptionTransport(
        rate_state=rate_state, query_fn=sdk.query, options_cls=_fake_options_cls,
    )
    api_transport = ApiKeyTransport(pontifex_factory=lambda _m, _k: _FakePontifex(chunks=[]))

    briefing_contents: list[str] = []

    async def briefing_hook(
        user_id: str, slug: str, content: str, base_system: str,
    ) -> tuple[str, dict | None]:
        briefing_contents.append(content)
        return base_system + "\n\nbriefed", None

    captured: dict = {}

    async def extraction_hook(user_id: str, slug: str, transcript: dict) -> None:
        captured["transcript"] = transcript

    runner = AgentRunner(
        subscription=sub, api_key=api_transport, rate_state=rate_state,
        briefing_hook=briefing_hook, extraction_hook=extraction_hook,
    )
    history = [
        {"role": "user", "content": "any rain today?"},
        {"role": "assistant", "content": "no rain expected"},
        {"role": "user", "content": "and tomorrow?"},
    ]
    async for _ in runner.run(
        user_id="alice", slug="weather", content="and tomorrow?",
        config=AgentConfig(system_prompt="s", model="anthropic/claude-opus-4-7", allowed_tools=[]),
        messages=history,
    ):
        pass

    # Transport saw the rendered history.
    assert sdk.last_prompt is not None
    assert "user: any rain today?" in sdk.last_prompt
    # Briefing retrieval query = latest user message only.
    assert briefing_contents == ["and tomorrow?"]
    # Extraction transcript = latest user message only (no history re-extraction).
    assert captured["transcript"]["messages"] == [
        {"role": "user", "content": "and tomorrow?"},
    ]


# ---------------------------------------------------------------------------
# Transient-transport retry (RemoteProtocolError before any output)
#
# Production evidence: unattended daily routines (scout 08-11, helper 08-12)
# terminated with a lone ``recordatio transport failure: RemoteProtocolError``
# and delivered nothing — a single transient httpx hiccup lost the whole run.
# ---------------------------------------------------------------------------


class _FlakySDK:
    """query() raises a (configurable) exception on the first ``fail_times``
    calls, *before* yielding anything, then streams ``tail`` on the next call.
    """

    def __init__(
        self,
        *,
        fail_times: int,
        tail: list[object],
        exc: BaseException | None = None,
    ) -> None:
        self.calls = 0
        self._fail_times = fail_times
        self._tail = tail
        self._exc = exc or httpx.RemoteProtocolError("Server disconnected without response")

    async def query(self, *, prompt: str, options: Any) -> AsyncIterator[object]:
        self.calls += 1
        if self.calls <= self._fail_times:
            raise self._exc
        for msg in self._tail:
            yield msg


@pytest.mark.asyncio
async def test_subscription_retries_transient_transport_error_before_output(
    rate_state: RateLimitStateStore,
) -> None:
    """A RemoteProtocolError that fires before any assistant output must be
    retried — it's a network blip, not a task failure. The run that used to
    die with a lone ``sdk_exception`` error now salvages to COMPLETED, with an
    observable ``retry`` event in between (telos renders ``kind == 'retry'``).
    """

    sdk = _FlakySDK(fail_times=1, tail=[_success_result_message()])
    transport = SubscriptionTransport(
        rate_state=rate_state, query_fn=sdk.query, options_cls=_fake_options_cls,
        transient_backoff_s=0.0,
    )
    events: list[TaskEvent] = []
    async for ev in transport.run(
        user_id="alice", slug="weather", content="hi", system_prompt="SYS",
        config=AgentConfig(allowed_tools=[]),
    ):
        events.append(ev)

    kinds = [e.kind for e in events]
    assert sdk.calls == 2  # first attempt failed, retried once and succeeded
    assert TaskEventKind.RETRY in kinds
    retry = next(e for e in events if e.kind is TaskEventKind.RETRY)
    assert retry.payload["reason"] == "transport"
    assert retry.payload["exception_type"] == "RemoteProtocolError"
    assert events[-1].kind is TaskEventKind.COMPLETED
    assert TaskEventKind.ERROR not in kinds


@pytest.mark.asyncio
async def test_subscription_does_not_retry_after_output_streamed(
    rate_state: RateLimitStateStore,
) -> None:
    """Once real assistant output has streamed, a mid-stream transport error
    can NOT be safely restarted (a restart would duplicate output / re-fire
    tool side effects): surface the terminal error as before, no retry."""

    async def flaky_query(*, prompt: str, options: Any) -> AsyncIterator[object]:
        yield _make_class("AssistantMessage", content=[{"type": "text", "text": "partial"}])
        raise httpx.RemoteProtocolError("dropped mid-stream")

    transport = SubscriptionTransport(
        rate_state=rate_state, query_fn=flaky_query, options_cls=_fake_options_cls,
        transient_backoff_s=0.0,
    )
    events: list[TaskEvent] = []
    async for ev in transport.run(
        user_id="alice", slug="weather", content="hi", system_prompt="SYS",
        config=AgentConfig(allowed_tools=[]),
    ):
        events.append(ev)

    kinds = [e.kind for e in events]
    assert TaskEventKind.MESSAGE in kinds  # the partial output made it out
    assert TaskEventKind.RETRY not in kinds
    assert events[-1].kind is TaskEventKind.ERROR
    assert events[-1].payload["reason"] == "sdk_exception"


@pytest.mark.asyncio
async def test_subscription_does_not_retry_non_transient_error(
    rate_state: RateLimitStateStore,
) -> None:
    """A non-transport exception (e.g. a ValueError from a malformed config)
    is a real failure and must surface immediately, never retried."""

    sdk = _FlakySDK(fail_times=1, tail=[_success_result_message()], exc=ValueError("boom"))
    transport = SubscriptionTransport(
        rate_state=rate_state, query_fn=sdk.query, options_cls=_fake_options_cls,
        transient_backoff_s=0.0,
    )
    events: list[TaskEvent] = []
    async for ev in transport.run(
        user_id="alice", slug="weather", content="hi", system_prompt="SYS",
        config=AgentConfig(allowed_tools=[]),
    ):
        events.append(ev)

    assert sdk.calls == 1  # no retry
    kinds = [e.kind for e in events]
    assert TaskEventKind.RETRY not in kinds
    assert events[-1].kind is TaskEventKind.ERROR
    assert events[-1].payload["reason"] == "sdk_exception"


@pytest.mark.asyncio
async def test_subscription_transient_retry_is_bounded(
    rate_state: RateLimitStateStore,
) -> None:
    """Retries are bounded: when every attempt fails transiently the terminal
    error still surfaces (we don't loop forever). With ``transient_retries=1``
    that's one initial attempt + one retry, then the error."""

    sdk = _FlakySDK(fail_times=99, tail=[_success_result_message()])
    transport = SubscriptionTransport(
        rate_state=rate_state, query_fn=sdk.query, options_cls=_fake_options_cls,
        transient_retries=1, transient_backoff_s=0.0,
    )
    events: list[TaskEvent] = []
    async for ev in transport.run(
        user_id="alice", slug="weather", content="hi", system_prompt="SYS",
        config=AgentConfig(allowed_tools=[]),
    ):
        events.append(ev)

    assert sdk.calls == 2  # initial + one retry, then give up
    assert events[-1].kind is TaskEventKind.ERROR
    assert events[-1].payload["reason"] == "sdk_exception"


class _FlakyPontifex:
    """Stub whose ``stream`` raises a transient error on the first
    ``fail_times`` calls, then yields ``chunks``."""

    def __init__(self, *, fail_times: int, chunks: list[dict]) -> None:
        self.calls = 0
        self._fail_times = fail_times
        self._chunks = chunks

    async def stream(
        self,
        *,
        messages: list[dict],
        system: str = "",
        max_tokens: int = 0,
        **kwargs: Any,
    ) -> AsyncIterator[dict]:
        self.calls += 1
        if self.calls <= self._fail_times:
            raise httpx.RemoteProtocolError("Server disconnected without response")
        for chunk in self._chunks:
            yield chunk


@pytest.mark.asyncio
async def test_api_key_retries_transient_transport_error_before_output(
    rate_state: RateLimitStateStore,
) -> None:
    """The API-key fallback path has the same bug/fix: a transient transport
    error before any output is retried instead of surfacing a lone
    ``pontifex_exception``."""

    pf = _FlakyPontifex(
        fail_times=1,
        chunks=[
            {"choices": [{"index": 0, "delta": {"content": "hi there"}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ],
    )
    transport = ApiKeyTransport(
        pontifex_factory=lambda _model, _key: pf,
        transient_backoff_s=0.0,
    )
    events: list[TaskEvent] = []
    async for ev in transport.run(
        user_id="alice", slug="weather", content="hi", system_prompt="SYS",
        config=AgentConfig(model="anthropic/claude-opus-4-7"),
    ):
        events.append(ev)

    kinds = [e.kind for e in events]
    assert pf.calls == 2
    assert TaskEventKind.RETRY in kinds
    assert events[-1].kind is TaskEventKind.COMPLETED
    assert TaskEventKind.ERROR not in kinds
