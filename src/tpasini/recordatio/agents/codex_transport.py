"""ChatGPT subscription transport using Codex app-server, preserving TaskEvent SSE."""

import asyncio
import json
import os
import re
import tempfile
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import anyio

from tpasini.recordatio.agents.codex_client import CodexClient
from tpasini.recordatio.agents.config_store import AgentConfig
from tpasini.recordatio.agents.events import TaskEvent, TaskEventKind
from tpasini.recordatio.agents.rate_limit_state import RateLimitStateStore
from tpasini.recordatio.agents.runner import _process_clean_cwd, _render_prompt_with_history
from tpasini.recordatio.agents.warm_pool import WarmPool


def codex_model(model: str) -> str | None:
    """Accept Codex names; reject stale Claude configurations before execution."""
    model = os.environ.get("RECORDATIO_SUBSCRIPTION_MODEL") or model
    if model.startswith("openai/"):
        model = model.removeprefix("openai/")
    if "/" in model or model.startswith("claude-"):
        raise ValueError("Set the agent model (or RECORDATIO_SUBSCRIPTION_MODEL) to a Codex model")
    return model or None


def tool_config(config: AgentConfig) -> dict[str, Any]:
    """Translate MCP allowlists, retaining the distinction between None and [].

    Claude's filesystem primitives have no exact Codex equivalents. Reject
    those names instead of silently granting shell access. MCP tools keep
    their existing mcp__server__tool identifiers in Recordatio events.
    """
    allowed = config.allowed_tools
    if allowed is not None and (not isinstance(allowed, list) or any(not isinstance(t, str) for t in allowed)):
        raise ValueError("allowed_tools must be a list of strings or null")
    supported = {"WebSearch", "Task"}
    if allowed is not None:
        unknown = [t for t in allowed if t not in supported and not t.startswith("mcp__")]
        if unknown:
            raise ValueError(f"Unsupported Claude tools; expose them through MCP: {', '.join(unknown)}")
    settings: dict[str, Any] = {
        "project_doc_max_bytes": 0,
        "web_search": "live" if allowed is None or "WebSearch" in allowed else "disabled",
        "features.shell_tool": False,
        "features.unified_exec": False,
        "features.view_image": False,
        "features.apps": False,
        "features.plugins": False,
        "features.hooks": False,
        "features.memories": False,
        "features.browser_use": False,
        "features.computer_use": False,
        "features.code_mode": False,
        "features.code_mode_host": False,
        "features.image_generation": False,
        "features.multi_agent": bool(config.subagents) and (allowed is None or "Task" in allowed),
    }
    servers: dict[str, Any] = {}
    for server in config.mcp_servers:
        name = server.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[\w-]+", name) or name in servers:
            raise ValueError("MCP servers require unique names containing letters, digits, underscores or hyphens")
        entry = {k: v for k, v in server.items() if k not in {"name", "type", "headers"}}
        if server.get("type") == "sse":
            raise ValueError("Codex requires streamable HTTP MCP; migrate the SSE endpoint")
        if "headers" in server:
            entry["http_headers"] = server["headers"]
        if allowed is not None:
            prefix = f"mcp__{name}__"
            names = [t.removeprefix(prefix) for t in allowed if t.startswith(prefix)]
            entry["enabled"] = bool(names)
            if "*" not in names:
                entry["enabled_tools"] = names
        # AgentConfig already grants the filtered MCP tools. Codex's default
        # annotation-based approval would otherwise reject granted write tools
        # under approvalPolicy=never. Keep explicit server/tool overrides.
        entry.setdefault("default_tools_approval_mode", "approve")
        entry["required"] = entry.get("enabled", True)
        servers[name] = entry
    if allowed is not None:
        for tool in allowed:
            if tool.startswith("mcp__") and not any(tool.startswith(f"mcp__{name}__") for name in servers):
                raise ValueError(f"No configured MCP server for {tool}")
    settings["mcp_servers"] = servers
    return settings


def _toml_lines(values: dict[str, Any], prefix: str = "") -> list[str]:
    """Render our JSON-compatible role settings as dotted TOML assignments."""
    lines = []
    for key, value in values.items():
        # Top-level settings use dotted keys; MCP server/tool names below are literal.
        parts = key.split(".") if not prefix else [key]
        path = prefix + ".".join(json.dumps(part) for part in parts)
        if isinstance(value, dict):
            lines.extend(_toml_lines(value, path + "."))
        elif value is not None:
            lines.append(f"{path} = {json.dumps(value, ensure_ascii=False)}")
    return lines


def thread_params(config: AgentConfig, system_prompt: str, cwd: str) -> dict[str, Any]:
    settings = tool_config(config)
    roles = {}
    for sub in config.subagents:
        name = sub.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[\w-]+", name) or name in roles:
            raise ValueError("Subagents require unique names containing letters, digits, underscores or hyphens")
        extra = set(sub) - {"name", "description", "prompt", "tools", "model"}
        if extra:
            raise ValueError(f"Unsupported subagent options for {name}: {sorted(extra)}")
        child = replace(config, subagents=[], allowed_tools=sub.get("tools", config.allowed_tools))
        if child.allowed_tools is None:
            child.allowed_tools = config.allowed_tools
        role = tool_config(child)
        role["developer_instructions"] = sub.get("prompt", "")
        if sub.get("model"):
            role["model"] = codex_model(sub["model"])
        path = Path(cwd) / f"role-{name}.toml"
        path.write_text("\n".join(_toml_lines(role)) + "\n")
        path.chmod(0o600)
        roles[name] = {"description": sub.get("description", ""), "config_file": str(path)}
    if roles:
        settings["agents"] = roles
    return {
        "model": codex_model(config.model),
        "modelProvider": "openai",
        "cwd": cwd,
        "ephemeral": True,
        "approvalPolicy": "never",
        "sandbox": "read-only",
        "baseInstructions": system_prompt,
        "developerInstructions": "",
        "config": settings,
    }


class CodexEventTranslator:
    def __init__(self) -> None:
        self.usage: dict[str, Any] = {}
        self.seen: set[str] = set()

    def translate(self, method: str, data: dict[str, Any]) -> list[TaskEvent]:
        if method == "item/agentMessage/delta":
            return [TaskEvent(TaskEventKind.MESSAGE_DELTA, {"text": data.get("delta", "")})]
        if method == "item/reasoning/summaryTextDelta":
            return [TaskEvent(TaskEventKind.THINKING_DELTA, {"text": data.get("delta", "")})]
        if method == "thread/tokenUsage/updated":
            total = (data.get("tokenUsage") or {}).get("total") or {}
            if total:
                # Codex inputTokens includes cached input. The SSE contract uses
                # Anthropic's disjoint fresh/cache buckets; avoid double counting.
                cached = total.get("cachedInputTokens", 0)
                self.usage = {
                    "input_tokens": max(0, total.get("inputTokens", 0) - cached),
                    "output_tokens": total.get("outputTokens", 0),
                    "cache_read_input_tokens": cached,
                    "cache_creation_input_tokens": total.get("cacheWriteInputTokens", 0),
                }
        if method == "error" and data.get("willRetry"):
            return [TaskEvent(TaskEventKind.RETRY, {"reason": "transport", "message": data.get("error")})]
        if method not in {"item/started", "item/completed"}:
            return []
        item = data.get("item") or {}
        kind, item_id = item.get("type"), item.get("id")
        completed = method == "item/completed"
        if completed:
            if item_id in self.seen:
                return []
            self.seen.add(item_id)
        if kind == "agentMessage" and completed:
            return [TaskEvent(TaskEventKind.MESSAGE, {"text": item.get("text", "")})]
        if kind == "reasoning" and completed:
            return [TaskEvent(TaskEventKind.THINKING, {"text": "\n".join(item.get("summary") or [])})]
        if kind == "mcpToolCall":
            if completed:
                return [
                    TaskEvent(
                        TaskEventKind.TOOL_RESULT,
                        {
                            "call_id": item_id,
                            "result": item.get("result") or item.get("error"),
                            "is_error": bool(item.get("error")) or item.get("status") == "failed",
                        },
                    )
                ]
            return [
                TaskEvent(
                    TaskEventKind.TOOL_CALL,
                    {
                        "call_id": item_id,
                        "name": f"mcp__{item['server']}__{item['tool']}",
                        "input": item.get("arguments"),
                    },
                )
            ]
        if kind == "webSearch":
            return [
                TaskEvent(
                    TaskEventKind.TOOL_RESULT if completed else TaskEventKind.TOOL_CALL,
                    {
                        "call_id": item_id,
                        "name": "WebSearch",
                        "input": {"query": item.get("query")},
                        "result": item.get("action") if completed else None,
                        "is_error": False,
                    },
                )
            ]
        if kind == "collabAgentToolCall":
            return [
                TaskEvent(
                    TaskEventKind.SUBAGENT_COMPLETED if completed else TaskEventKind.SUBAGENT_STARTED,
                    {
                        "task_id": item_id,
                        "description": item.get("prompt"),
                        "name": item.get("agentRole") or "subagent",
                        "is_error": item.get("status") == "failed",
                        "status": item.get("status"),
                        "agents": item.get("agentsStates"),
                    },
                )
            ]
        return []


class CodexSubscriptionTransport:
    """Execute subscription turns with ChatGPT authentication and no API-key billing."""

    def __init__(
        self,
        *,
        rate_state: RateLimitStateStore,
        warm_pool: WarmPool | None = None,
        client_factory: Callable[..., Any] = CodexClient,
        timeout_s: float | None = None,
    ) -> None:
        self._rate_state = rate_state
        self._warm_pool = warm_pool
        self._client_factory = client_factory
        self._timeout_s = (
            timeout_s if timeout_s is not None else float(os.environ.get("RECORDATIO_AGENT_TIMEOUT_S", "300"))
        )

    async def _connect(self, config: AgentConfig, system_prompt: str) -> Any:
        cwd = tempfile.mkdtemp(prefix="codex-", dir=_process_clean_cwd())
        params = thread_params(config, system_prompt, cwd)
        client = self._client_factory(cwd=cwd, timeout_s=self._timeout_s)
        try:
            await client.connect()
            result = await client.request("thread/start", params)
            client.thread_id = result["thread"]["id"]
            client.model = result.get("model") or params["model"]
            return client
        except BaseException:
            with anyio.CancelScope(shield=True):
                await client.disconnect()
            raise

    def _rate_limit(self, data: dict[str, Any], user_id: str, slug: str) -> TaskEvent:
        limits = data.get("rateLimits") or {}
        windows = [limits[k] for k in ("primary", "secondary") if limits.get(k)]
        exhausted = [w for w in windows if w.get("usedPercent", 0) >= 100]
        window = max(exhausted or windows, key=lambda w: w.get("resetsAt") or 0, default={})
        payload = {
            "status": ("exceeded" if exhausted else "allowed") if windows else None,
            "rate_limit_type": "codex",
            "overage_status": None,
            "overage_disabled_reason": None,
            "utilization": window["usedPercent"] / 100 if "usedPercent" in window else None,
            "resets_at": str(window["resetsAt"]) if window.get("resetsAt") else None,
        }
        # Missing quota data is not proof of recovery. Likewise a separate
        # model bucket (e.g. Spark) cannot clear the main subscription window.
        if windows and limits.get("limitId") in (None, "codex"):
            self._rate_state.upsert(user_id=user_id, slug=slug, overage_resets_at=None, **payload)
        return TaskEvent(TaskEventKind.RATE_LIMIT, payload)

    async def run(
        self,
        *,
        user_id: str,
        slug: str,
        content: str,
        system_prompt: str,
        config: AgentConfig,
        messages: list[dict[str, Any]] | None = None,
        briefing: str | None = None,
    ) -> AsyncIterator[TaskEvent]:
        client = None
        successful = False
        try:
            # Check before taking a spare, including a host-level model override.
            codex_model(config.model)
            if self._warm_pool is not None:
                client = await self._warm_pool.take(user_id, slug, config.version)
            source = "warm" if client is not None else "cold"
            if client is None:
                client = await self._connect(config, system_prompt)
            yield TaskEvent(
                TaskEventKind.STARTED,
                {
                    "transport": "subscription",
                    "provider": "openai",
                    "model": client.model,
                    "session_id": client.thread_id,
                    "session_source": source,
                },
            )
            result = await client.request(
                "turn/start",
                {
                    "threadId": client.thread_id,
                    "input": [{"type": "text", "text": _render_prompt_with_history(content, messages, briefing)}],
                },
            )
            turn_id = result["turn"]["id"]
            translator = CodexEventTranslator()
            while True:
                notification = await client.next_notification()
                if "id" in notification and "method" in notification:
                    await client.reject_request(notification)
                method = notification.get("method", "")
                data = notification.get("params") or {}
                if method == "account/rateLimits/updated":
                    yield self._rate_limit(data, user_id, slug)
                    continue
                # Child threads must never contaminate the parent transcript.
                if data.get("threadId") != client.thread_id or data.get("turnId", turn_id) != turn_id:
                    continue
                for event in translator.translate(method, data):
                    yield event
                if method != "turn/completed":
                    continue
                turn = data["turn"]
                if turn.get("id") != turn_id:
                    continue
                # Some app-server versions include final items only on completion.
                for item in turn.get("items") or []:
                    for event in translator.translate("item/completed", {"item": item}):
                        yield event
                if turn["status"] != "completed":
                    yield TaskEvent(
                        TaskEventKind.ERROR,
                        {
                            "reason": "codex_turn_failed",
                            "status": turn["status"],
                            "error": turn.get("error"),
                            "message": str(turn.get("error") or turn["status"]),
                        },
                    )
                    return
                successful = True
                # Disconnect BEFORE the terminal yield; SSE clients close on it.
                await client.disconnect()
                yield TaskEvent(
                    TaskEventKind.COMPLETED,
                    {
                        "session_id": client.thread_id,
                        "model": client.model,
                        "stop_reason": "end_turn",
                        "provider": "openai",
                        "billing_mode": "subscription",
                        "cost_usd": 0.0,
                        "duration_ms": turn.get("durationMs"),
                        "usage": translator.usage,
                    },
                )
                return
        except Exception as exc:
            yield TaskEvent(
                TaskEventKind.ERROR,
                {
                    "reason": "timeout" if isinstance(exc, TimeoutError) else "codex_exception",
                    "message": str(exc),
                    "exception_type": type(exc).__name__,
                },
            )
        finally:
            if client is not None:
                with anyio.CancelScope(shield=True):
                    await client.disconnect()
            if successful and self._warm_pool is not None:
                self._warm_pool.schedule_warm(
                    user_id,
                    slug,
                    config.version,
                    lambda: self._connect(config, system_prompt),
                )


async def complete_text(prompt: str, model: str = "", timeout_s: float = 120) -> str:
    """Tool-free completion for offline identity judging, using the same protocol."""
    with tempfile.TemporaryDirectory(prefix="recordatio-judge-") as cwd:
        client = CodexClient(cwd=cwd, timeout_s=timeout_s)
        try:
            async with asyncio.timeout(timeout_s):
                await client.connect()
                thread = await client.request(
                    "thread/start",
                    thread_params(
                        AgentConfig(model=model, allowed_tools=[]),
                        "Answer the user's classification request. Do not use tools.",
                        cwd,
                    ),
                )
                thread_id = thread["thread"]["id"]
                await client.request(
                    "turn/start",
                    {
                        "threadId": thread_id,
                        "input": [{"type": "text", "text": prompt}],
                    },
                )
                translator = CodexEventTranslator()
                parts = []
                while True:
                    message = await client.next_notification()
                    if "id" in message and "method" in message:
                        await client.reject_request(message)
                    data = message.get("params") or {}
                    if data.get("threadId") != thread_id:
                        continue
                    method = message.get("method", "")
                    for event in translator.translate(method, data):
                        if event.kind is TaskEventKind.MESSAGE:
                            parts.append(event.payload["text"])
                    if method == "turn/completed":
                        turn = data["turn"]
                        if turn["status"] != "completed":
                            raise RuntimeError(f"Codex judge failed: {turn.get('error') or turn['status']}")
                        for item in turn.get("items") or []:
                            for event in translator.translate("item/completed", {"item": item}):
                                if event.kind is TaskEventKind.MESSAGE:
                                    parts.append(event.payload["text"])
                        if not parts:
                            raise RuntimeError("Codex judge returned no text")
                        return "\n".join(parts)
        finally:
            with anyio.CancelScope(shield=True):
                await client.disconnect()
