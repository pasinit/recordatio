import asyncio
import tomllib

import pytest

from tpasini.recordatio.agents.codex_client import CodexClient, codex_environment
from tpasini.recordatio.agents.codex_transport import (
    CodexEventTranslator,
    CodexSubscriptionTransport,
    codex_model,
    thread_params,
    tool_config,
)
from tpasini.recordatio.agents.config_store import AgentConfig
from tpasini.recordatio.agents.events import TaskEventKind, collect_assistant_text
from tpasini.recordatio.agents.rate_limit_state import RateLimitStateStore
from tpasini.recordatio.agents.warm_pool import WarmPool


@pytest.fixture
def state(tmp_path):
    store = RateLimitStateStore(tmp_path / "rate.db")
    yield store
    store.close()


def notification(method, **params):
    return {"method": method, "params": {"threadId": "parent", "turnId": "turn", **params}}


class FakeClient:
    def __init__(self, messages=(), *, failure=None):
        self.messages = list(messages)
        self.failure = failure
        self.requests = []
        self.disconnected = False
        self.alive = True

    async def connect(self):
        pass

    async def request(self, method, params):
        self.requests.append((method, params))
        if self.failure:
            raise self.failure
        if method == "thread/start":
            return {"thread": {"id": "parent"}, "model": "test-model"}
        return {"turn": {"id": "turn"}}

    async def next_notification(self):
        if not self.messages:
            raise EOFError("no terminal event")
        message = self.messages.pop(0)
        if isinstance(message, BaseException):
            raise message
        return message

    async def disconnect(self):
        self.disconnected = True
        self.alive = False


def make_transport(state, fake, **kwargs):
    return CodexSubscriptionTransport(rate_state=state, client_factory=lambda **kw: fake, **kwargs)


async def run(transport, config=None):
    return [
        event
        async for event in transport.run(
            user_id="u",
            slug="s",
            content="new question",
            system_prompt="base",
            config=config or AgentConfig(model="openai/test-model", allowed_tools=[]),
            messages=[{"role": "user", "content": "old question"}, {"role": "assistant", "content": "old answer"}],
            briefing="memory briefing",
        )
    ]


def test_auth_isolation(monkeypatch):
    for key in ["OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_API_KEY"]:
        monkeypatch.setenv(key, "not-for-child")
    monkeypatch.setenv("RECORDATIO_CODEX_HOME", "/service/codex")
    env = codex_environment()
    assert env["CODEX_HOME"] == "/service/codex"
    assert not any(k in env for k in ["OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_API_KEY"])


def test_model_requires_migration(monkeypatch):
    monkeypatch.delenv("RECORDATIO_SUBSCRIPTION_MODEL", raising=False)
    with pytest.raises(ValueError, match="Codex model"):
        codex_model("anthropic/claude-opus-4-7")
    assert codex_model("openai/test-model") == "test-model"
    assert codex_model("") is None
    monkeypatch.setenv("RECORDATIO_SUBSCRIPTION_MODEL", "openai/replacement")
    assert codex_model("anthropic/claude-opus-4-7") == "replacement"


def test_mcp_permissions_and_subagent_config(tmp_path):
    config = AgentConfig(
        allowed_tools=["mcp__telos__read", "Task"],
        mcp_servers=[
            {
                "name": "telos",
                "type": "http",
                "url": "http://localhost/mcp",
                "headers": {"Authorization": "Bearer per-agent"},
            }
        ],
        subagents=[{"name": "researcher", "prompt": "Research", "tools": ["mcp__telos__search"]}],
    )
    params = thread_params(config, "base", str(tmp_path))
    server = params["config"]["mcp_servers"]["telos"]
    assert server["enabled_tools"] == ["read"]
    assert server["http_headers"] == {"Authorization": "Bearer per-agent"}
    assert "type" not in server and "headers" not in server
    assert params["config"]["features.shell_tool"] is False
    role = tomllib.loads((tmp_path / "role-researcher.toml").read_text())
    assert role["mcp_servers"]["telos"]["enabled_tools"] == ["search"]
    assert role["developer_instructions"] == "Research"
    assert role["features"]["multi_agent"] is False
    assert (tmp_path / "role-researcher.toml").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("allowed,enabled", [(None, True), ([], False), (["mcp__telos__*"], True)])
def test_mcp_none_empty_wildcard(allowed, enabled):
    settings = tool_config(
        AgentConfig(
            allowed_tools=allowed,
            mcp_servers=[{"name": "telos", "command": "mcp-server"}],
        )
    )
    server = settings["mcp_servers"]["telos"]
    assert server.get("enabled", True) is enabled
    assert server["required"] is enabled


@pytest.mark.parametrize("tools", [["Bash"], ["Read"], ["mcp__absent__read"]])
def test_unsupported_tools_fail_closed(tools):
    with pytest.raises(ValueError):
        tool_config(AgentConfig(allowed_tools=tools))


async def test_streaming_history_usage_and_terminal_cleanup(state):
    item = {"id": "message", "type": "agentMessage", "text": "answer"}
    fake = FakeClient(
        [
            notification("item/agentMessage/delta", delta="ans"),
            notification("item/completed", threadId="child", item={**item, "text": "child answer"}),
            notification("item/completed", item=item),
            notification(
                "thread/tokenUsage/updated",
                tokenUsage={
                    "total": {
                        "inputTokens": 100,
                        "cachedInputTokens": 40,
                        "outputTokens": 20,
                    }
                },
            ),
            notification("turn/completed", turn={"id": "turn", "status": "completed", "items": [item]}),
        ]
    )
    events = await run(make_transport(state, fake))
    assert collect_assistant_text(events) == "answer"
    assert events[0].kind is TaskEventKind.STARTED
    assert events[-1].kind is TaskEventKind.COMPLETED
    assert events[-1].payload["usage"] == {
        "input_tokens": 60,
        "output_tokens": 20,
        "cache_read_input_tokens": 40,
        "cache_creation_input_tokens": 0,
    }
    assert events[-1].payload["cost_usd"] == 0
    assert events[-1].payload["billing_mode"] == "subscription"
    prompt = fake.requests[1][1]["input"][0]["text"]
    assert "old answer" in prompt and "memory briefing" in prompt and "new question" in prompt
    assert fake.requests[0][1]["baseInstructions"] == "base"
    assert fake.disconnected


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [TimeoutError()],
        [
            notification(
                "turn/completed",
                turn={"id": "turn", "status": "failed", "items": [], "error": {"message": "quota"}},
            )
        ],
    ],
)
async def test_failure_never_completes_and_disconnects(state, messages):
    fake = FakeClient(messages)
    events = await run(make_transport(state, fake))
    assert [e.kind for e in events].count(TaskEventKind.ERROR) == 1
    assert all(e.kind is not TaskEventKind.COMPLETED for e in events)
    assert fake.disconnected


async def test_initialization_failure_disconnects(state):
    fake = FakeClient(failure=RuntimeError("login required"))
    events = await run(make_transport(state, fake))
    assert events[-1].kind is TaskEventKind.ERROR
    assert fake.disconnected


async def test_cancelled_turn_disconnects(state):
    fake = FakeClient([asyncio.CancelledError()])
    with pytest.raises(asyncio.CancelledError):
        await run(make_transport(state, fake))
    assert fake.disconnected


async def test_close_after_started_disconnects(state):
    fake = FakeClient()
    stream = make_transport(state, fake).run(
        user_id="u",
        slug="s",
        content="hello",
        system_prompt="base",
        config=AgentConfig(allowed_tools=[]),
    )
    await anext(stream)
    await stream.aclose()
    assert fake.disconnected


def test_mcp_event_pairing():
    translator = CodexEventTranslator()
    item = {"id": "tool", "type": "mcpToolCall", "server": "telos", "tool": "search", "arguments": {"q": "x"}}
    call = translator.translate("item/started", {"item": item})[0]
    result = translator.translate(
        "item/completed", {"item": {**item, "status": "failed", "error": {"message": "bad"}}}
    )[0]
    assert call.payload["name"] == "mcp__telos__search"
    assert call.payload["call_id"] == result.payload["call_id"] == "tool"
    assert result.payload["is_error"]


async def test_dead_codex_spare_is_discarded():
    pool = WarmPool(min_age_s=0)
    fake = FakeClient()

    async def connect():
        return fake

    await pool.schedule_warm("u", "s", 1, connect)
    fake.alive = False
    assert await pool.take("u", "s", 1) is None
    await pool.wait_idle()
    await pool.close_all()
    assert fake.disconnected


async def test_client_real_stdio_handshake_and_api_auth_rejection(tmp_path, monkeypatch):
    # Exercise framing and interleaved notifications through an actual child process.
    script = tmp_path / "fake-codex"
    script.write_text("""#!/usr/bin/env python3
import json, sys
for line in sys.stdin:
    request = json.loads(line)
    if "id" not in request:
        continue
    result = {"account": {"type": "apiKey"}} if request["method"] == "account/read" else {}
    print(json.dumps({"method": "notice", "params": {}}), flush=True)
    print(json.dumps({"id": request["id"], "result": result}), flush=True)
""")
    script.chmod(0o700)
    monkeypatch.setenv("RECORDATIO_CODEX_BIN", str(script))
    monkeypatch.setenv("RECORDATIO_CODEX_HOME", str(tmp_path / "home"))
    client = CodexClient(cwd=str(tmp_path))
    try:
        with pytest.raises(RuntimeError, match="ChatGPT login"):
            await client.connect()
        assert (await client.next_notification())["method"] == "notice"
    finally:
        process = client.process
        await client.disconnect()
    assert process.returncode is not None


def test_quota_missing_or_other_bucket_cannot_clear_exhaustion(state):
    transport = make_transport(state, FakeClient())
    transport._rate_limit(
        {
            "rateLimits": {
                "limitId": "codex",
                "primary": {"usedPercent": 10, "resetsAt": 4000000000},
                "secondary": {"usedPercent": 100, "resetsAt": 4000000100},
            }
        },
        "u",
        "s",
    )
    assert state.get("u", "s").should_fall_back
    transport._rate_limit({"rateLimits": {}}, "u", "s")
    transport._rate_limit(
        {
            "rateLimits": {
                "limitId": "spark",
                "primary": {"usedPercent": 1, "resetsAt": 4000000000},
            }
        },
        "u",
        "s",
    )
    assert state.get("u", "s").should_fall_back
    assert state.get("u", "s").resets_at == "4000000100"


async def test_subscription_only_never_selects_paid_fallback(state):
    from tpasini.recordatio.agents.runner import AgentRunner

    fake = FakeClient(
        [
            notification("turn/completed", turn={"id": "turn", "status": "completed", "items": []}),
        ]
    )
    transport = make_transport(state, fake)
    transport._rate_limit(
        {
            "rateLimits": {
                "primary": {"usedPercent": 100, "resetsAt": 4000000000},
            }
        },
        "u",
        "s",
    )
    runner = AgentRunner(subscription=transport, api_key=None, rate_state=state, allow_api_fallback=False)
    events = [e async for e in runner.run(user_id="u", slug="s", content="hello", config=AgentConfig())]
    assert events[0].payload["transport"] == "subscription"
    assert events[-1].kind is TaskEventKind.COMPLETED


async def test_identity_backend_supports_async_persistence_caller(monkeypatch):
    from tpasini.recordatio.agents import codex_transport
    from tpasini.recordatio.entity_identity import CodexCLIBackend

    async def complete(prompt, model, timeout):
        return f"{prompt}:{model}:{timeout}"

    monkeypatch.setattr(codex_transport, "complete_text", complete)
    assert CodexCLIBackend(model="model", timeout=3).complete("judge") == "judge:model:3"


def test_subscription_timeout_respects_deployment_override(state, monkeypatch):
    monkeypatch.setenv("RECORDATIO_AGENT_TIMEOUT_S", "900")
    assert make_transport(state, FakeClient())._timeout_s == 900
    assert make_transport(state, FakeClient(), timeout_s=15)._timeout_s == 15


def test_granted_mcp_tools_are_preapproved_without_widening_allowlist():
    config = AgentConfig(model="", mcp_servers=[
        {"name": "granted", "type": "http", "url": "http://localhost/granted"},
        {"name": "disabled", "type": "http", "url": "http://localhost/disabled"},
    ], allowed_tools=["mcp__granted__echo"])
    servers = tool_config(config)["mcp_servers"]
    assert servers["granted"]["enabled_tools"] == ["echo"]
    assert servers["granted"]["default_tools_approval_mode"] == "approve"
    assert servers["disabled"]["enabled"] is False
    assert servers["disabled"]["enabled_tools"] == []
    config.mcp_servers[0]["default_tools_approval_mode"] = "prompt"
    config.mcp_servers[0]["tools"] = {"echo": {"approval_mode": "writes"}}
    server = tool_config(config)["mcp_servers"]["granted"]
    assert server["default_tools_approval_mode"] == "prompt"
    assert server["tools"]["echo"]["approval_mode"] == "writes"
