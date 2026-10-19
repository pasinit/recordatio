"""Agent runner: orchestrates one task through one of two transports.

``SubscriptionTransport`` wraps ``claude_agent_sdk.query()`` and bills against
the user's Claude subscription credits. ``ApiKeyTransport`` wraps the existing
Pontifex client (the one ``/v1/chat/completions`` uses) and bills at API-token
rates. The runner picks one per task based on the cached ``RateLimitSnapshot``
and Phase 0's hard constraint: once the 5-hour window exhausts, the SDK
rejects with ``overage_status="rejected", overage_disabled_reason="out_of_credits"``;
the next task must take the API-key path automatically.

Mid-flight rate-limit rejection cannot be salvaged (the current task is lost);
the runner surfaces an ``error`` TaskEvent and the cached state is updated so
the next task falls back. This is the behaviour the migration plan calls for
("this task cannot be split").

Transcript assembly: after the task completes, the runner stitches together a
transcript dict and hands it to the existing extractor via the new
``extract_knowledge_from_transcript`` shim. The dict shape is the same
``ConversationTurn``-compatible structure the OpenAI path produces.
"""

import hashlib
import json
import logging
import os
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import anyio

from tpasini.recordatio.agents.config_store import AgentConfig
from tpasini.recordatio.agents.events import (
    TaskEvent,
    TaskEventKind,
    collect_assistant_text,
    extract_openai_finish_reason,
    translate_openai_chunk,
    translate_sdk_message,
)
from tpasini.recordatio.agents.rate_limit_state import RateLimitStateStore
from tpasini.recordatio.agents.warm_pool import WarmPool, disconnect_quietly
from tpasini.recordatio.proxy.config import resolve_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Transport selection
# ---------------------------------------------------------------------------


class TransportSelection(StrEnum):
    """Which transport the runner picked for a given task.

    Surfaced on the ``started`` event payload so the caller can log it
    and so tests can assert against a stable enum rather than a magic
    string.
    """

    SUBSCRIPTION = "subscription"
    API_KEY = "api_key"


def select_transport(rate_state: RateLimitStateStore, user_id: str, slug: str) -> TransportSelection:
    """Return the transport to use for the next task.

    Driven entirely by the cached ``RateLimitSnapshot``: if the snapshot
    is missing (we've never run this user+slug) or says the subscription
    is healthy, prefer the subscription. Otherwise fall back.
    """
    snapshot = rate_state.get(user_id, slug)
    if snapshot is None:
        return TransportSelection.SUBSCRIPTION
    if snapshot.should_fall_back:
        return TransportSelection.API_KEY
    return TransportSelection.SUBSCRIPTION


# ---------------------------------------------------------------------------
# Subscription transport
# ---------------------------------------------------------------------------


# Default per-message-gap timeout: how long the SDK may stay silent between
# messages before the turn is killed. Must accommodate the longest single
# tool execution (web search, browser automation) and extended thinking —
# the first prod turns died at 60s mid-tool-run. Overridable via
# RECORDATIO_AGENT_TIMEOUT_S for operators who prefer snappier failures: the
# Phase 0 finding (a dead `ANTHROPIC_BASE_URL` hangs the SDK ~3 minutes)
# now surfaces as a slow first-message timeout instead of a fast one.
_DEFAULT_TIMEOUT_S = float(os.environ.get("RECORDATIO_AGENT_TIMEOUT_S", "300"))


# Mirrors the default ``/v1/chat/completions`` budget in ``_stream_sse``.
# ApiKeyTransport used to omit this kwarg entirely, leaving Pontifex on its
# own 8096 default; making it explicit means both transports share a single
# knob if/when we tune the budget.
_API_KEY_DEFAULT_MAX_TOKENS = 8096


# A transient httpx transport failure (a dropped connection, a half-read
# response) is not a task-level error — the SDK/Pontifex subprocess hit a
# network hiccup mid-connect/mid-read. When one fires *before any assistant
# output has streamed*, the turn can be safely restarted from scratch; a
# single occurrence used to lose the entire run (e.g. an unattended 6am daily
# routine terminating with ``recordatio transport failure: RemoteProtocolError``
# and delivering nothing). We retry a bounded number of times with a small
# linear backoff. Overridable for operators who want a different policy.
_DEFAULT_TRANSIENT_RETRIES = int(os.environ.get("RECORDATIO_AGENT_TRANSIENT_RETRIES", "1"))
_DEFAULT_TRANSIENT_BACKOFF_S = float(os.environ.get("RECORDATIO_AGENT_TRANSIENT_BACKOFF_S", "0.5"))

# httpx exception class names treated as transient transport hiccups. Matched
# by name (walking the ``__cause__``/``__context__`` chain) rather than
# ``isinstance`` so a ``RemoteProtocolError`` re-wrapped by the Agent SDK still
# classifies without coupling this module to the SDK's private exception
# hierarchy or importing httpx at module load.
_TRANSIENT_TRANSPORT_EXC_NAMES = frozenset(
    {
        "RemoteProtocolError",
        "ConnectError",
        "ConnectTimeout",
        "ReadError",
        "ReadTimeout",
        "WriteError",
        "PoolTimeout",
    }
)


def _is_transient_transport_error(exc: BaseException) -> bool:
    """True when ``exc`` (or something in its cause chain) is a transient
    httpx transport failure rather than a task-level error.

    Only these should be retried: a task that genuinely failed (a bad config,
    a content filter, an exhausted context) must surface, not loop.
    """
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if type(cur).__name__ in _TRANSIENT_TRANSPORT_EXC_NAMES:
            return True
        cur = cur.__cause__ or cur.__context__
    return False


# Event kinds that represent real, already-streamed assistant progress. Once
# any of these has been emitted a mid-stream transport failure can no longer
# be safely restarted (re-running would duplicate output or re-fire tool side
# effects), so we surface the terminal error instead of retrying. STARTED,
# RETRY and RATE_LIMIT are pre-content signals and do NOT block a restart.
_RESTART_SAFE_EVENT_KINDS = frozenset(
    {TaskEventKind.STARTED, TaskEventKind.RETRY, TaskEventKind.RATE_LIMIT}
)


# Character budget for the rendered ``<conversation_history>`` block on the
# subscription path. ~8k chars ≈ ~2k tokens — enough for the last dozen-ish
# chat turns without crowding out the system prompt + briefing. History is
# trimmed oldest-first: we walk the prior messages newest-first and stop
# adding once the budget is exhausted, so the most recent context always
# survives. The latest user message is never part of this block (it IS the
# prompt body) and therefore can never be dropped by trimming.
_HISTORY_CHAR_BUDGET = 8_000

# Appended to the newest history line when it alone exceeds the budget and
# is cut to fit rather than dropped (see the oversized-line branch in
# ``_render_prompt_with_history``).
_TRUNCATION_MARKER = " …[truncated]"


def _content_to_text(content: Any) -> str:
    """Flatten an OpenAI-style message ``content`` field to plain text.

    Telos sends ``content`` either as a plain string or — on image turns —
    as a block list (``[{"type": "image_url", ...}, {"type": "text",
    "text": ...}]``; see ``telos/agent.py`` where the user history entry is
    built). For history rendering we keep the text parts and replace
    non-text blocks with a short attachment note so the model knows
    something was there without us shipping base64 into the prompt.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
            else:
                kind = block.get("type") or "attachment"
                parts.append(f"[{kind} attached]")
        return " ".join(p for p in parts if p)
    return ""


def _render_prompt_with_history(
    content: str,
    messages: list[dict[str, Any]] | None,
    briefing: str | None = None,
) -> str:
    """Build the SDK prompt: history block + memory briefing + latest message.

    ``messages`` is the full per-turn history telos ships in the POST body,
    ending with the latest user message (which equals ``content``). The SDK
    gets a fresh session every turn, so without this preamble the agent
    greets every message as the first (the per-turn-amnesia P0).

    Rendering order — history → memory → message:

    1. ``<conversation_history>`` — prior turns as a ``role: text``
       transcript, capped at ``_HISTORY_CHAR_BUDGET`` chars dropping
       oldest turns first (bulk context at the top).
    2. The memory briefing — the per-turn ``<user_profile>`` +
       ``<relevant_memory>`` two-block text built (and fully tagged) by
       ``prompting.build_system``. It used to ride in the system prompt,
       but that is sealed at SDK connect time, which made pre-warmed
       sessions impossible (the briefing changes every turn). It now rides
       here, adjacent to the question it was retrieved for ("lost in the
       middle" avoidance; long-context guidance puts task-relevant material
       near the end). It is inserted VERBATIM — ``build_system`` owns the
       block tags, so the runner does not add its own wrapper. It does NOT
       count against the history char budget.
    3. The latest user message, verbatim — always last, never trimmed.

    When there is no history and no briefing the prompt is just
    ``content`` — identical to the pre-history behaviour.
    """
    parts: list[str] = []

    # The trailing latest-user-message is the prompt body itself — don't
    # repeat it inside the transcript. Anything else (including a trailing
    # assistant message, which telos shouldn't send but we tolerate) is
    # history.
    history = list(messages) if messages else []
    if history and history[-1].get("role") == "user":
        history = history[:-1]

    if history:
        # Walk newest-first so trimming drops the OLDEST turns.
        lines: list[str] = []
        budget = _HISTORY_CHAR_BUDGET
        for msg in reversed(history):
            role = msg.get("role") or "user"
            text = _content_to_text(msg.get("content"))
            if not text:
                continue
            line = f"{role}: {text}"
            if len(line) > budget:
                # An oversized NEWEST line used to end the walk before
                # anything was collected, rendering an EMPTY transcript —
                # an agent whose previous reply exceeded the whole budget
                # (e.g. a daily-brief routine) then ran with no history at
                # all. Keep a budget-sized head of that line instead;
                # older lines are still dropped (budget is spent).
                if not lines and budget > len(_TRUNCATION_MARKER):
                    cut = budget - len(_TRUNCATION_MARKER)
                    lines.append(line[:cut] + _TRUNCATION_MARKER)
                break
            lines.append(line)
            budget -= len(line) + 1  # +1 for the joining newline
        if lines:
            lines.reverse()
            transcript = "\n".join(lines)
            parts.append(f"<conversation_history>\n{transcript}\n</conversation_history>")

    if briefing:
        # build_system already wraps the briefing in <user_profile> /
        # <relevant_memory> tags; insert it verbatim (no double-wrapping).
        parts.append(briefing)

    parts.append(content)
    return "\n\n".join(parts)


# Maps each seeder-supplied subagent key (camelCase per the SDK, or a
# snake_case alias for ergonomics) to the ``AgentDefinition`` constructor
# parameter name. ``description``, ``prompt`` and ``tools`` are handled
# separately in ``_validate_and_build_agents`` (defaults / type validation).
_SUBAGENT_FIELD_MAP: dict[str, str] = {
    "model": "model",
    "disallowedTools": "disallowedTools",
    "disallowed_tools": "disallowedTools",
    "skills": "skills",
    "memory": "memory",
    "maxTurns": "maxTurns",
    "max_turns": "maxTurns",
    "effort": "effort",
    "permissionMode": "permissionMode",
    "permission_mode": "permissionMode",
    "mcpServers": "mcpServers",
    "mcp_servers": "mcpServers",
    "initialPrompt": "initialPrompt",
    "initial_prompt": "initialPrompt",
    "background": "background",
}


# Tool names that — when present in ``allowed_tools`` — permit the parent agent
# to delegate to a subagent. If a config has subagents but ``allowed_tools``
# excludes both of these AND is non-empty (empty list = allow all), the
# subagents are unreachable. We log a warning rather than raise because some
# migrations land subagents and allowed_tools in separate PRs.
_SUBAGENT_DELEGATION_TOOLS: frozenset[str] = frozenset({"Task", "Agent"})

# Claude Code built-ins that need an interactive user. Under
# ``bypassPermissions`` the model can still reach them, and in a headless
# service they fail: the model then repeats itself as plain text (once
# in production: AskUserQuestion twice, then the same questions again, all
# streamed into one reply). Always blocked, on top of any per-agent list.
# The scheduling built-ins are session-scoped or need claude.ai access, so
# nothing they create outlives the session; agents that reach for them
# (observed in production: RemoteTrigger, then CronCreate) bypass the host's own
# durable scheduler.
_HEADLESS_DISALLOWED_TOOLS: tuple[str, ...] = (
    "AskUserQuestion",
    "EnterPlanMode",
    "ExitPlanMode",
    "CronCreate",
    "CronDelete",
    "CronList",
    "RemoteTrigger",
    "ScheduleWakeup",
)


def _validate_and_build_agents(
    config_subagents: list[dict[str, Any]],
    agent_def_cls: Any,
) -> dict[str, Any]:
    """Validate seeder-supplied subagent dicts and construct ``AgentDefinition``s.

    Raises ``ValueError`` (with a structured message) on:
    - non-dict items
    - missing ``name`` key
    - duplicate ``name`` across items
    - ``tools`` set to a non-list, non-None value (catches the
      ``tools="WebSearch"`` footgun where the SDK would iterate the string).

    All other SDK-supported fields (``model``, ``disallowedTools``, ``skills``,
    ``memory``, ``maxTurns``, ``effort``, ``permissionMode``, ``mcpServers``,
    ``initialPrompt``, ``background``) are forwarded verbatim when present.
    Seeders may use either camelCase or snake_case for these keys; both are
    normalised to the SDK's camelCase before construction.

    Note on ``tools``: the SDK semantic is
    - ``None``  → subagent inherits the parent's ``allowed_tools``
    - ``[]``    → subagent has no tools
    These mean different things; this function does NOT coerce one into the
    other. Seeders that want "lock down" must use ``[]``, not ``None``.
    """
    agents: dict[str, Any] = {}
    for idx, sub in enumerate(config_subagents):
        if not isinstance(sub, dict):
            raise ValueError(
                f"subagent at index {idx} must be a dict, got {type(sub).__name__}"
            )
        name = sub.get("name")
        if not name:
            raise ValueError(f"subagent at index {idx} missing 'name'")
        if not isinstance(name, str):
            raise ValueError(
                f"subagent at index {idx} 'name' must be a string, got {type(name).__name__}"
            )
        if name in agents:
            raise ValueError(f"duplicate subagent name: {name!r}")

        tools = sub.get("tools")
        # tools must be a list[str] or None. Strings are a particularly nasty
        # footgun because the SDK would iterate them character-by-character.
        if tools is not None and not isinstance(tools, list):
            raise ValueError(
                f"subagent {name!r} 'tools' must be a list[str] or None, "
                f"got {type(tools).__name__}"
            )

        kwargs: dict[str, Any] = {
            "description": sub.get("description", ""),
            "prompt": sub.get("prompt", ""),
            "tools": tools,
        }
        # Forward all other SDK-supported fields. We accept either the SDK's
        # camelCase spelling or a snake_case alias; the map normalises both
        # to the ``AgentDefinition`` constructor parameter name.
        for key, value in sub.items():
            if key in {"name", "description", "prompt", "tools"}:
                continue
            sdk_kwarg = _SUBAGENT_FIELD_MAP.get(key)
            if sdk_kwarg is None:
                # Unknown key — silently ignore. Forward-compat: a newer
                # seeder may emit fields a Recordatio running older code doesn't
                # know about, and we don't want to refuse to launch.
                continue
            kwargs[sdk_kwarg] = value

        agents[name] = agent_def_cls(**kwargs)
    return agents


def _process_clean_cwd() -> Path:
    """Return the process-wide clean-cwd PARENT dir, created lazily on first call.

    Per-request temp dirs (the previous design) raced the SDK's child
    subprocess on client disconnect: the request handler would tear the
    tmpdir down via ``TemporaryDirectory.__exit__`` while the still-alive
    SDK subprocess was holding the dir as its ``cwd``, occasionally tripping
    "directory not empty" errors and (worse) blew away the subprocess's
    working dir mid-flight. Finding #12 on PR #133. Hence: no teardown, ever.

    This dir is only the parent; each run gets a fresh ``mkdtemp``
    subdirectory underneath (see ``_clean_cwd``). Agents run with
    ``bypassPermissions`` and their tools resolve relative paths against
    the cwd, so a single shared dir could be contaminated by one agent
    (e.g. writing a ``CLAUDE.md`` that the SDK would then auto-load for
    every later run — the very thing this dir exists to avoid, see
    ``MIGRATION_NOTES`` gotcha #6). Per-run subdirs restore isolation
    while keeping the no-teardown disconnect-race fix.
    """
    global _PROCESS_CLEAN_CWD  # noqa: PLW0603 — module-level cache, intentional
    if _PROCESS_CLEAN_CWD is None:
        # ``mkdtemp`` (not ``TemporaryDirectory``) — we want the dir to
        # outlive any context manager and survive for the whole process.
        # No cleanup needed: the OS reaps ``/tmp`` on reboot, and the
        # subdirs are empty-ish (only whatever an agent chose to write),
        # so this isn't a leak worth tracking.
        _PROCESS_CLEAN_CWD = Path(tempfile.mkdtemp(prefix="recordatio-agent-clean-cwd-"))
    return _PROCESS_CLEAN_CWD


# Module-level cache for the process-wide clean cwd. Initialised lazily on
# first use so unit tests that don't touch the subscription path don't pay
# the syscall cost.
_PROCESS_CLEAN_CWD: Path | None = None


@asynccontextmanager
async def _clean_cwd() -> AsyncIterator[Path]:
    """Yield a fresh per-run clean cwd under the process-wide parent dir.

    Each run gets its own ``mkdtemp`` subdirectory so an agent that writes
    into its cwd (they run with ``bypassPermissions``) can't poison later
    runs — notably via a stray ``CLAUDE.md`` the SDK would auto-load.

    Kept as an ``asynccontextmanager`` so the call site in
    ``SubscriptionTransport.run`` stays a clean ``async with`` block.
    There is intentionally no cleanup in ``__aexit__`` — teardown raced
    the SDK subprocess on client disconnect (see ``_process_clean_cwd``).
    The OS reaps ``/tmp``; leaking empty-ish dirs is acceptable.
    """
    yield Path(tempfile.mkdtemp(prefix="run-", dir=_process_clean_cwd()))


class _SubagentConfigError(ValueError):
    """Subagent validation failure — surfaced as ``invalid_subagent_config``.

    Distinguished from the plain ``ValueError`` a malformed
    ``mcp_servers`` entry raises (which keeps its historical
    ``sdk_exception`` error shape) now that both are raised from the
    shared ``_build_options_kwargs`` helper.
    """


class SubscriptionTransport:
    """Runs a single agent turn against the user's Claude subscription.

    Two execution paths, identical event semantics:

    - **Warm**: a pre-connected ``ClaudeSDKClient`` taken from the
      ``WarmPool`` (keyed by user, slug, config version) — the
      subprocess spawn + control init + MCP handshakes were paid in the
      background after the previous turn. One turn per session: the
      client is disconnected in a ``finally`` once the turn ends.
    - **Cold**: the one-shot ``claude_agent_sdk.query()`` exactly as
      before — first turn after idle, pool miss, or dead spare.

    Each call is fully self-contained: we pick a clean ``cwd``, set
    ``system_prompt`` explicitly (so the SDK doesn't auto-load anything),
    run the turn under a per-message-gap timeout, and stream translated
    events out as they arrive. Rate-limit observations are written back
    to the cache so the next task can pre-flight against fresh state.
    After the turn a replacement spare is scheduled in the background.

    The class itself is a thin adapter — ``query_fn`` (cold path),
    ``client_factory`` (warm path) and ``options_cls`` are injectable so
    tests can substitute fakes without needing a working SDK install.
    """

    def __init__(
        self,
        *,
        rate_state: RateLimitStateStore,
        query_fn: Callable[..., AsyncIterator[Any]] | None = None,
        options_cls: Any = None,
        agent_def_cls: Any = None,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        warm_pool: WarmPool | None = None,
        client_factory: Callable[..., Any] | None = None,
        transient_retries: int = _DEFAULT_TRANSIENT_RETRIES,
        transient_backoff_s: float = _DEFAULT_TRANSIENT_BACKOFF_S,
    ) -> None:
        self._rate_state = rate_state
        self._query_fn = query_fn
        self._options_cls = options_cls
        self._agent_def_cls = agent_def_cls
        self._timeout_s = timeout_s
        self._warm_pool = warm_pool
        self._client_factory = client_factory
        self._transient_retries = transient_retries
        self._transient_backoff_s = transient_backoff_s

    def _resolve_sdk(self) -> tuple[Callable[..., AsyncIterator[Any]], Any]:
        """Lazily import the SDK so unit tests can run without it installed.

        Returning ``(query_fn, ClaudeAgentOptions)`` keeps the call site
        below clean: we don't import at module load time because the SDK
        ships a 60+MB native CLI binary that we don't want to load when
        the runner is exercised in tests that don't touch the real path.
        """
        if self._query_fn is not None and self._options_cls is not None:
            return self._query_fn, self._options_cls
        # Local import — see docstring.
        from claude_agent_sdk import ClaudeAgentOptions, query  # noqa: PLC0415

        return query, ClaudeAgentOptions

    def _resolve_agent_def(self) -> Any:
        """Lazily resolve ``AgentDefinition`` for the same reason as the
        SDK import above. Tests inject a fake via ``agent_def_cls``; the
        real path imports from the SDK only when a task actually carries
        subagents.
        """
        if self._agent_def_cls is not None:
            return self._agent_def_cls
        from claude_agent_sdk import AgentDefinition  # noqa: PLC0415

        return AgentDefinition

    def _resolve_client_factory(self) -> Callable[..., Any]:
        """Lazily resolve ``ClaudeSDKClient`` for the warm path — same
        rationale as ``_resolve_sdk``. Tests inject a fake via
        ``client_factory``."""
        if self._client_factory is not None:
            return self._client_factory
        from claude_agent_sdk import ClaudeSDKClient  # noqa: PLC0415

        return ClaudeSDKClient

    def _build_options_kwargs(
        self,
        *,
        user_id: str,
        slug: str,
        system_prompt: str,
        config: AgentConfig,
        model: str | None,
        cwd: str,
    ) -> dict[str, Any]:
        """Build the ``ClaudeAgentOptions`` kwargs for one SDK session.

        Shared by the cold path (one-shot ``query()``) and the warm path
        (``WarmPool`` spares connected ahead of time), so both seal the
        exact same session config — notably the env isolation and the
        BASE system prompt (the per-turn briefing rides in the rendered
        prompt, not here; see ``_render_prompt_with_history``).

        Raises:
            ValueError: malformed ``mcp_servers`` entry (missing
                ``name``) — surfaces as the historical ``sdk_exception``
                error shape.
            _SubagentConfigError: malformed subagent config — surfaces
                as ``invalid_subagent_config``.
            ImportError: SDK unavailable when ``config.subagents`` forces
                the lazy ``AgentDefinition`` import.
        """
        options_kwargs: dict[str, Any] = {
            "system_prompt": system_prompt,
            "cwd": cwd,
            "permission_mode": "bypassPermissions",
            # Token-level streaming: the SDK emits ``StreamEvent``s
            # (raw Anthropic stream events) alongside the complete
            # ``AssistantMessage``s. ``translate_sdk_message`` maps
            # top-level text/thinking deltas to ``message_delta`` /
            # ``thinking_delta`` TaskEvents so clients can render
            # tokens as they arrive (the legacy chat-path UX); the
            # full MESSAGE/THINKING events still follow as the
            # authoritative text. Side effect: every delta is SDK
            # forward progress, so the per-gap ``fail_after`` in
            # ``_pump`` resets far more often mid-generation.
            "include_partial_messages": True,
            # The CLI prefers an env ANTHROPIC_API_KEY over the
            # subscription OAuth token, silently flipping this
            # transport to API billing (first prod deploy failed with
            # "Credit balance is too low"). The Recordatio host needs
            # the key for extraction, but the SDK subprocess must not
            # see it: options.env merges over the inherited env, and
            # an empty value makes the CLI fall back to
            # CLAUDE_CODE_OAUTH_TOKEN (verified against the bundled
            # CLI). This is what makes SubscriptionTransport
            # subscription-by-construction.
            #
            # ENABLE_TOOL_SEARCH=false loads MCP tool definitions up
            # front instead of behind the deferred ToolSearch step, which
            # otherwise adds a "loading tool schemas" detour (and a
            # second round of narration) to every tool-using turn.
            "env": {"ANTHROPIC_API_KEY": "", "ENABLE_TOOL_SEARCH": "false"},
            "disallowed_tools": list(_HEADLESS_DISALLOWED_TOOLS),
        }
        # Only pass ``allowed_tools`` when the config actually
        # specifies a list. Passing ``[]`` to the SDK means
        # "block every tool"; passing nothing means "use the
        # SDK's defaults" — which is what the migration plan
        # calls for when an agent hasn't pinned a tool list.
        if config.allowed_tools is not None:
            options_kwargs["allowed_tools"] = list(config.allowed_tools)
        if model:
            options_kwargs["model"] = model
        if config.mcp_servers:
            # ``AgentConfig.mcp_servers`` is a list of dicts with a
            # ``name`` field; ``ClaudeAgentOptions.mcp_servers`` is a
            # dict keyed by server name whose values are the SDK's
            # ``McpServerConfig`` shapes (e.g. http: ``{"type",
            # "url", "headers"}``; stdio: ``{"command", "args",
            # "env"}``). Strip ``name`` and pass the rest through
            # verbatim. Without this mapping the server is never
            # registered, every ``mcp__<server>__*`` entry in
            # ``allowed_tools`` resolves to nothing, and — because
            # ``allowed_tools`` is a whitelist — the agent runs
            # with zero tools.
            mcp_servers: dict[str, dict[str, Any]] = {}
            for server in config.mcp_servers:
                name = server.get("name")
                if not name:
                    raise ValueError(
                        "mcp_servers entry is missing the required 'name' field"
                    )
                if name in mcp_servers:
                    logger.warning(
                        "duplicate MCP server name %r for %s/%s; last entry wins",
                        name, user_id, slug,
                    )
                mcp_servers[name] = {k: v for k, v in server.items() if k != "name"}
            options_kwargs["mcp_servers"] = mcp_servers
        if config.subagents:
            # Lazy SDK import + validation:
            # - ImportError when the SDK is unavailable (the caller's
            #   ``except ImportError`` branch).
            # - _SubagentConfigError on malformed subagent dicts
            #   (non-dict items, missing name, duplicate names, bad
            #   tools type) — kept distinct so the mcp_servers
            #   ValueError above keeps its "sdk_exception" error shape.
            # - TypeError if the SDK's AgentDefinition rejects a kwarg
            #   (the caller's generic ``except Exception``).
            agent_def_cls = self._resolve_agent_def()
            try:
                options_kwargs["agents"] = _validate_and_build_agents(
                    config.subagents, agent_def_cls,
                )
            except ValueError as exc:
                raise _SubagentConfigError(str(exc)) from exc
        return options_kwargs

    @staticmethod
    def _session_fingerprint(config: AgentConfig, system_prompt: str, model: str | None) -> str:
        """Cover all variable option inputs, not just the persisted version.

        cwd is intentionally fresh per session; fixed transport options are
        identical on both paths. Hash rather than retain credential headers.
        """
        payload = {**config.to_dict(), "system_prompt": system_prompt, "model": model}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    @staticmethod
    def _has_turn_headers(config: AgentConfig) -> bool:
        return any(
            name.lower() == "x-telos-turn-id"
            for server in config.mcp_servers
            for name in (server.get("headers") or {})
        )

    def _maybe_schedule_warm(
        self,
        *,
        user_id: str,
        slug: str,
        system_prompt: str,
        config: AgentConfig,
        model: str | None,
    ) -> None:
        """Schedule a background warm-up of the NEXT spare for this agent.

        Called after a turn was served (warm or cold): an active
        conversation predicts another turn soon, and the spare pays the
        subprocess spawn + control init + MCP handshakes ahead of it.
        ``system_prompt`` must be the BASE prompt — the spare's session
        is sealed at connect, and the per-turn briefing rides in the
        rendered prompt instead. No-op when no pool is wired (tests, or
        a pool disabled via RECORDATIO_WARM_POOL_MAX=0 ignores the call).
        """
        pool = self._warm_pool
        if pool is None or self._has_turn_headers(config):
            return
        # Freeze nested inline config before the background task can run.
        config = deepcopy(config)
        fingerprint = self._session_fingerprint(config, system_prompt, model)

        async def _connect() -> Any:
            client_factory = self._resolve_client_factory()
            _, options_cls = self._resolve_sdk()
            # Each spare gets its own fresh per-run cwd, exactly like a
            # cold run (the cwd is sealed at connect too).
            cwd = tempfile.mkdtemp(prefix="run-", dir=_process_clean_cwd())
            options_kwargs = self._build_options_kwargs(
                user_id=user_id, slug=slug, system_prompt=system_prompt,
                config=config, model=model, cwd=cwd,
            )
            client = client_factory(options_cls(**options_kwargs))
            await client.connect()
            return client

        pool.schedule_warm(
            user_id, slug, config.version, _connect, fingerprint=fingerprint,
        )

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
        """Stream TaskEvents for one task.

        Yields a ``started`` event first (with transport tag and
        ``session_source`` warm/cold), then each translated SDK event,
        then either ``completed`` or ``error``. On timeout we emit
        ``error`` and return — the SDK process is torn down by the
        surrounding cancel scope (cold) or the ``finally`` disconnect
        (warm).

        ``messages`` is the optional full conversation history from the
        request body. The SDK session is fresh every turn, so prior turns
        are rendered into the prompt as a transcript preamble (see
        ``_render_prompt_with_history``); without it the agent has
        per-turn amnesia. ``content`` stays the latest user message.

        ``system_prompt`` must be the agent's BASE prompt and
        ``briefing`` the per-turn memory briefing text (if any): the
        briefing rides in the rendered prompt between the history block
        and the latest message, NOT in the system prompt — the system
        prompt is sealed at SDK connect time, and pre-warmed spares
        connect before the briefing for the next turn exists.
        """
        query_fn, options_cls = self._resolve_sdk()
        prompt = _render_prompt_with_history(content, messages, briefing)

        # ``config.model`` may arrive provider-prefixed ("anthropic/claude-…")
        # — telos seeds it verbatim from its MODEL env var. The SDK only
        # understands bare Anthropic model names, so strip a single leading
        # provider segment using the same parser the API-key path relies on
        # (``resolve_model`` splits on the first "/"; anything after it is
        # the model name and passes through untouched). A prefix that
        # resolves to a non-anthropic backend can never work on the
        # subscription path — surface that as a structured error instead of
        # letting the SDK fail with a cryptic "model may not exist".
        model: str | None = config.model or None
        if model and "/" in model:
            bare_model, backend, _ = resolve_model(model)
            if backend != "anthropic":
                yield TaskEvent(
                    TaskEventKind.ERROR,
                    {
                        "reason": "unsupported_model_provider",
                        "model": model,
                        "transport": TransportSelection.SUBSCRIPTION.value,
                    },
                )
                return
            model = bare_model

        # Warm-spare lookup happens before STARTED so the event can tag
        # the session source. ``take`` is a dict pop + liveness probe —
        # the spare's connect cost was paid in the background after the
        # previous turn. If the spare turns out dead at send time we fall
        # back to a cold spawn below (the tag then reflects the attempt;
        # the fallback is logged).
        warm_client: Any | None = None
        if self._warm_pool is not None:
            warm_client = await self._warm_pool.take(
                user_id, slug, config.version,
                fingerprint=self._session_fingerprint(config, system_prompt, model),
            )

        session_source = "warm" if warm_client is not None else "cold"
        # INFO, not debug: warm/cold is the first thing to correlate when a
        # turn misbehaves (the two paths drive the SDK differently), and
        # until now neither side logged it above debug.
        logger.info(
            "agent turn %s/%s: session_source=%s model=%s prompt_chars=%d",
            user_id, slug, session_source, model, len(prompt),
        )

        yield TaskEvent(
            TaskEventKind.STARTED,
            {
                "transport": TransportSelection.SUBSCRIPTION.value,
                "user_id": user_id,
                "slug": slug,
                "model": model,
                "session_source": session_source,
            },
        )

        # Finding #10: warn (don't raise) when subagents are configured but
        # the parent's allowed_tools list excludes both ``Task`` and ``Agent``
        # — delegation will be impossible. We don't raise because some
        # migrations land subagents and allowed_tools in separate PRs; the
        # warning makes the inconsistency observable in logs.
        if (
            config.subagents
            and config.allowed_tools  # non-empty list (None = inherit all)
            and not _SUBAGENT_DELEGATION_TOOLS.intersection(config.allowed_tools)
        ):
            logger.warning(
                "subagents configured for %s/%s but neither 'Task' nor 'Agent' "
                "is in allowed_tools (%s); delegation impossible",
                user_id, slug, list(config.allowed_tools),
            )

        async with _clean_cwd() as cwd:
            terminal_emitted = False
            # Whether any real assistant output has streamed yet. Gates the
            # transient-transport retry below: once content is out, a restart
            # would duplicate it / re-fire tool side effects.
            streamed_content = False
            try:
                # Options are built even on the warm path: a dead spare
                # falls back to a cold spawn that needs them. Built inside
                # the try so a malformed config surfaces as a structured
                # ``error`` event rather than an uncaught exception that
                # escapes the SSE stream as a 500.
                try:
                    options_kwargs = self._build_options_kwargs(
                        user_id=user_id, slug=slug, system_prompt=system_prompt,
                        config=config, model=model, cwd=str(cwd),
                    )
                except _SubagentConfigError as exc:
                    logger.warning(
                        "subagent validation failed for %s/%s: %s",
                        user_id, slug, exc,
                    )
                    yield TaskEvent(
                        TaskEventKind.ERROR,
                        {
                            "reason": "invalid_subagent_config",
                            "message": str(exc),
                        },
                    )
                    return
                options = options_cls(**options_kwargs)

                # Warm path: one ``query()`` write to the spare's stdin —
                # the only send this session will ever see (one turn per
                # session, statelessness is the product). A failed write
                # means the subprocess died in the pool; disconnect and
                # fall back to the cold spawn transparently.
                if warm_client is not None:
                    try:
                        await warm_client.query(prompt)
                    except Exception:
                        logger.warning(
                            "warm spare for %s/%s was dead at send; falling back to cold spawn",
                            user_id, slug, exc_info=True,
                        )
                        await disconnect_quietly(warm_client)
                        warm_client = None
                # The stream attempt (spawn → pump → drain) is wrapped in a
                # bounded retry: a transient httpx transport hiccup
                # (RemoteProtocolError / ConnectError / ReadError) that fires
                # *before any assistant output has streamed* is a network
                # blip, not a task-level failure, and a single occurrence used
                # to lose the whole run (an unattended 6am daily routine
                # terminating with ``recordatio transport failure:
                # RemoteProtocolError`` and delivering nothing). We restart —
                # always from a cold spawn, since a warm spare is single-use —
                # up to ``self._transient_retries`` times, emitting an
                # observable ``retry`` event each time. Once real content has
                # streamed a restart is unsafe (duplicate output / re-fired
                # tool side effects), so the error surfaces as before.
                transient_attempt = 0
                while True:
                    if warm_client is not None:
                        # ``receive_response()`` terminates after yielding the
                        # ResultMessage — same end-of-turn shape as ``query()``.
                        message_iter = aiter(warm_client.receive_response())
                    else:
                        message_iter = aiter(query_fn(prompt=prompt, options=options))

                    # Finding #8: the timeout used to wrap the entire ``yield
                    # event`` loop, which meant slow downstream consumers (e.g.
                    # an SSE client applying back-pressure) could trip the
                    # timeout even though the SDK had plenty of life left.
                    # Fix: pump the SDK's async iterator into a bounded queue,
                    # with ``fail_after`` scoped to ONLY the ``anext()`` call —
                    # i.e. the timeout bounds SDK forward progress per message.
                    # ``send()`` happens OUTSIDE the cancel scope, so time spent
                    # blocked on a full buffer (a slow consumer beyond the
                    # 64-message window) never counts against the timeout.
                    send_stream, receive_stream = anyio.create_memory_object_stream[Any](
                        max_buffer_size=64,
                    )

                    # The pump runs as a task-group child; anyio 4 wraps child
                    # exceptions in an ExceptionGroup, which would slip past the
                    # ``except TimeoutError`` / ``except Exception`` branches
                    # below. Record the exception instead and re-raise it from
                    # the generator's own task after the drain loop, preserving
                    # the existing error-translation behaviour verbatim. Loop
                    # vars are bound as defaults so each attempt's pump closes
                    # over its own iterator/stream/error-list.
                    pump_exc: list[Exception] = []

                    async def _pump(
                        it: Any = message_iter,
                        sink: Any = send_stream,
                        errors: list[Exception] = pump_exc,
                    ) -> None:
                        try:
                            while True:
                                with anyio.fail_after(self._timeout_s):
                                    try:
                                        msg = await anext(it)
                                    except StopAsyncIteration:
                                        break
                                await sink.send(msg)
                        except Exception as exc:  # noqa: BLE001 — re-raised below
                            errors.append(exc)
                        finally:
                            await sink.aclose()

                    async with anyio.create_task_group() as tg:
                        tg.start_soon(_pump)
                        async with receive_stream:
                            async for msg in receive_stream:
                                # Record any rate-limit observation before we
                                # translate so the cache is up-to-date even if
                                # the translator decides not to surface it.
                                self._maybe_record_rate_limit(user_id, slug, msg)
                                for event in translate_sdk_message(msg):
                                    if event.kind in (TaskEventKind.COMPLETED, TaskEventKind.ERROR):
                                        terminal_emitted = True
                                    if event.kind not in _RESTART_SAFE_EVENT_KINDS:
                                        streamed_content = True
                                    yield event

                    if pump_exc:
                        stream_exc = pump_exc[0]
                        if (
                            not terminal_emitted
                            and not streamed_content
                            and transient_attempt < self._transient_retries
                            and _is_transient_transport_error(stream_exc)
                        ):
                            transient_attempt += 1
                            logger.warning(
                                "transient transport error for %s/%s before any "
                                "output (%s); retrying attempt %d/%d",
                                user_id, slug, type(stream_exc).__name__,
                                transient_attempt, self._transient_retries,
                                exc_info=True,
                            )
                            # Observable retry — telos logs ``kind == 'retry'``;
                            # mirrors the SDK's own ``api_retry`` surfacing.
                            yield TaskEvent(
                                TaskEventKind.RETRY,
                                {
                                    "attempt": transient_attempt,
                                    "max_retries": self._transient_retries,
                                    "reason": "transport",
                                    "exception_type": type(stream_exc).__name__,
                                },
                            )
                            # A warm spare is single-use and already drained;
                            # every retry is a fresh cold spawn.
                            if warm_client is not None:
                                await disconnect_quietly(warm_client)
                                warm_client = None
                            if self._transient_backoff_s > 0:
                                await anyio.sleep(self._transient_backoff_s * transient_attempt)
                            continue
                        raise stream_exc
                    break

                # Finding #9: if the SDK iterator finished without emitting
                # a ``ResultMessage``, the runner used to fall silent — no
                # terminal event ever reached the client and SSE consumers
                # hung waiting for ``completed``/``error``. Synthesize one.
                if not terminal_emitted:
                    yield TaskEvent(
                        TaskEventKind.ERROR,
                        {
                            "reason": "sdk_eof_without_result",
                            "message": "SDK iterator exhausted without ResultMessage",
                        },
                    )
                    terminal_emitted = True
            except ImportError as exc:
                # Raised when the SDK isn't installed and the lazy import
                # finally fires (only on the subagent path; the main SDK
                # import already happened in _resolve_sdk above and would
                # have raised earlier if missing).
                logger.exception("SDK import failed for %s/%s", user_id, slug)
                yield TaskEvent(
                    TaskEventKind.ERROR,
                    {
                        "reason": "sdk_import_error",
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
                return
            except TimeoutError:
                logger.warning(
                    "subscription transport timed out after %.1fs for %s/%s",
                    self._timeout_s, user_id, slug,
                )
                # A pump exception can arrive after the drain loop already
                # yielded a terminal event (e.g. the SDK sent a
                # ResultMessage and then its iterator raised). One terminal
                # event per stream: if we've already emitted one, log the
                # late failure instead of yielding a second.
                if not terminal_emitted:
                    yield TaskEvent(
                        TaskEventKind.ERROR,
                        {
                            "reason": "timeout",
                            "timeout_s": self._timeout_s,
                        },
                    )
            except Exception as exc:
                # The SDK can raise on hard transport failures (Phase 0
                # showed e.g. ``Exception: Claude Code returned an error
                # result: success`` when the upstream URL is dead). We
                # log the type+message and surface a structured error to
                # the caller.
                logger.exception("subscription transport raised for %s/%s", user_id, slug)
                # See the TimeoutError branch: never emit a second terminal
                # event after a drained COMPLETED/ERROR.
                if not terminal_emitted:
                    yield TaskEvent(
                        TaskEventKind.ERROR,
                        {"reason": "sdk_exception", "exception_type": type(exc).__name__, "message": str(exc)},
                    )
            finally:
                # One turn per session — a used warm client is NEVER
                # reused (the pool already forgot it at take()) and its
                # subprocess must not outlive the turn, on any exit path:
                # completion, timeout, SDK error, or the consumer closing
                # this generator mid-stream (GeneratorExit runs this too).
                if warm_client is not None:
                    await disconnect_quietly(warm_client)

        # Predictive warm-up: an active conversation predicts another turn
        # soon, so pre-pay the next session's cold start in the background.
        # Reached on normal completion and handled-error paths; NOT reached
        # when the consumer closed the generator early (GeneratorExit
        # propagates past the except branches above) or when the SDK is
        # unusable (the ImportError branch returns) — both are cases where
        # spawning background subprocesses would be unwise.
        self._maybe_schedule_warm(
            user_id=user_id, slug=slug, system_prompt=system_prompt,
            config=config, model=model,
        )

    def _maybe_record_rate_limit(self, user_id: str, slug: str, msg: Any) -> None:
        if type(msg).__name__ != "RateLimitEvent":
            return
        info = getattr(msg, "rate_limit_info", None)
        if info is None and isinstance(msg, dict):
            info = msg.get("rate_limit_info")
        if info is None:
            return
        # ``info`` may be a dataclass or a dict.
        def _g(key: str) -> Any:
            if isinstance(info, dict):
                return info.get(key)
            return getattr(info, key, None)

        self._rate_state.upsert(
            user_id=user_id,
            slug=slug,
            status=_g("status"),
            rate_limit_type=_g("rate_limit_type"),
            overage_status=_g("overage_status"),
            overage_disabled_reason=_g("overage_disabled_reason"),
            utilization=_g("utilization"),
            resets_at=_g("resets_at"),
            overage_resets_at=_g("overage_resets_at"),
        )


# ---------------------------------------------------------------------------
# API-key transport (fallback)
# ---------------------------------------------------------------------------


# Type alias for "something Pontifex-shaped". Spelled this way so tests can
# substitute a stub without importing pontifex.
PontifexLike = Any


class ApiKeyTransport:
    """Wraps the existing Pontifex client used by ``/v1/chat/completions``.

    Translates ``AgentConfig`` → an OpenAI-shaped ``messages`` array and
    streams the response back, fanning each chunk out as TaskEvents.

    The Pontifex factory is injected so the runner can share the same
    construction logic the chat-completions endpoint uses (provider
    routing via ``resolve_model`` etc.) without this module having to
    know about that infrastructure.

    Capability degradation vs the subscription path
    -----------------------------------------------
    Pontifex is a single-completion streaming bridge: it has no agentic
    loop, no tool executor, no MCP client, and no subagent concept. The
    SDK-side config knobs therefore **cannot** be honoured here:

    - ``mcp_servers`` — there is no MCP client to connect with; the
      servers (and their ``mcp__<server>__*`` tools) are unavailable.
    - ``allowed_tools`` — SDK tool names ("Bash", "Read",
      "mcp__telos__*") have no OpenAI function-schema equivalents and
      no executor; forwarding them (or fabricated definitions) would
      produce tool calls nothing can run, so we deliberately don't.
    - ``subagents`` — an SDK-only orchestration feature.

    Rather than fake any of this, the transport logs a warning at run
    time and surfaces the dropped capabilities on the ``started`` event
    payload (``degraded_capabilities``) so callers can see exactly what
    a fallback task lost.
    """

    def __init__(
        self,
        *,
        pontifex_factory: Callable[[str, str | None], PontifexLike],
        transient_retries: int = _DEFAULT_TRANSIENT_RETRIES,
        transient_backoff_s: float = _DEFAULT_TRANSIENT_BACKOFF_S,
    ) -> None:
        self._factory = pontifex_factory
        self._transient_retries = transient_retries
        self._transient_backoff_s = transient_backoff_s

    async def run(
        self,
        *,
        user_id: str,
        slug: str,
        content: str,
        system_prompt: str,
        config: AgentConfig,
        api_key: str | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[TaskEvent]:
        # ``config.model`` is provider-prefixed (e.g. "anthropic/claude-…")
        # — the factory takes care of resolving it; we just pass it through.
        if not config.model:
            yield TaskEvent(
                TaskEventKind.ERROR,
                {"reason": "missing_model", "transport": TransportSelection.API_KEY.value},
            )
            return

        # See the class docstring: these AgentConfig capabilities have no
        # Pontifex equivalent and are dropped on the fallback path. Warn
        # loudly (one task losing its tools silently is exactly the bug
        # this guards against) and surface the list on the started event.
        degraded = [
            cap
            for cap, present in (
                ("mcp_servers", bool(config.mcp_servers)),
                ("allowed_tools", config.allowed_tools is not None),
                ("subagents", bool(config.subagents)),
            )
            if present
        ]
        if degraded:
            logger.warning(
                "api-key fallback for %s/%s drops unsupported capabilities: %s "
                "(Pontifex has no agent loop/MCP client; the task runs as a "
                "plain completion)",
                user_id, slug, ", ".join(degraded),
            )

        started_payload: dict[str, Any] = {
            "transport": TransportSelection.API_KEY.value,
            "user_id": user_id,
            "slug": slug,
            "model": config.model,
        }
        if degraded:
            started_payload["degraded_capabilities"] = degraded
        yield TaskEvent(TaskEventKind.STARTED, started_payload)

        pf = self._factory(config.model, api_key)
        # Pontifex speaks OpenAI chat-completions natively, so the inbound
        # conversation history rides the wire as-is (block-list content for
        # image turns included) — no transcript rendering needed on this
        # path. No history → the legacy single-turn shape, unchanged.
        wire_messages = (
            [dict(m) for m in messages]
            if messages
            else [{"role": "user", "content": content}]
        )

        # Pontifex's final chunk carries an OpenAI-wire ``usage`` block
        # (prompt_tokens/completion_tokens, plus cache fields on the
        # Anthropic backend). Keep the last non-empty one we see so the
        # ``completed`` event can carry it for downstream cost recording
        # (telos's ``_usage_from_completed`` reads a nested ``usage``
        # dict off the completed payload — same contract as the
        # subscription path's ResultMessage translation).
        usage: dict[str, Any] | None = None
        # Finding #11: keep the upstream ``finish_reason`` so the terminal
        # event is faithful — ``stop_reason`` mirrors what the provider said
        # rather than a hardcoded "stop", and ``content_filter`` surfaces as
        # an ERROR instead of masquerading as a clean completion.
        finish_reason: str | None = None
        # Whether any assistant output has streamed yet — gates the transient
        # retry below (a mid-stream restart would duplicate output).
        streamed_content = False
        # A transient httpx transport hiccup before any output is a network
        # blip, not a task failure: restart the completion up to
        # ``self._transient_retries`` times rather than losing the whole run.
        transient_attempt = 0
        while True:
            try:
                async for chunk in pf.stream(
                    messages=wire_messages,
                    system=system_prompt,
                    max_tokens=_API_KEY_DEFAULT_MAX_TOKENS,
                ):
                    chunk_usage = chunk.get("usage")
                    if isinstance(chunk_usage, dict) and chunk_usage:
                        usage = chunk_usage
                    chunk_finish_reason = extract_openai_finish_reason(chunk)
                    if chunk_finish_reason:
                        finish_reason = chunk_finish_reason
                    for event in translate_openai_chunk(chunk):
                        if event.kind not in _RESTART_SAFE_EVENT_KINDS:
                            streamed_content = True
                        yield event
            except Exception as exc:
                if (
                    not streamed_content
                    and transient_attempt < self._transient_retries
                    and _is_transient_transport_error(exc)
                ):
                    transient_attempt += 1
                    logger.warning(
                        "transient transport error for %s/%s before any output "
                        "(%s); retrying attempt %d/%d",
                        user_id, slug, type(exc).__name__,
                        transient_attempt, self._transient_retries,
                        exc_info=True,
                    )
                    yield TaskEvent(
                        TaskEventKind.RETRY,
                        {
                            "attempt": transient_attempt,
                            "max_retries": self._transient_retries,
                            "reason": "transport",
                            "exception_type": type(exc).__name__,
                        },
                    )
                    # Reset per-attempt accumulators and re-create the client
                    # so the retry streams a fresh completion.
                    usage = None
                    finish_reason = None
                    if self._transient_backoff_s > 0:
                        await anyio.sleep(self._transient_backoff_s * transient_attempt)
                    pf = self._factory(config.model, api_key)
                    continue
                logger.exception("api-key transport raised for %s/%s", user_id, slug)
                yield TaskEvent(
                    TaskEventKind.ERROR,
                    {"reason": "pontifex_exception", "exception_type": type(exc).__name__, "message": str(exc)},
                )
                return
            break

        if finish_reason == "content_filter":
            # The provider truncated/blocked the response. Reporting it as
            # ``completed`` would let consumers treat a censored answer as a
            # clean one; surface an error instead (finding #11).
            yield TaskEvent(
                TaskEventKind.ERROR,
                {
                    "reason": "content_filter",
                    "stop_reason": finish_reason,
                    "transport": TransportSelection.API_KEY.value,
                },
            )
            return

        completed_payload: dict[str, Any] = {
            # Mirror the upstream finish_reason verbatim ("stop", "length",
            # "tool_calls", …); fall back to "stop" only when the stream
            # never carried one.
            "stop_reason": finish_reason or "stop",
            "transport": TransportSelection.API_KEY.value,
        }
        if usage:
            # Omit the key entirely when the upstream never reported
            # usage — consumers fall back to their own estimates.
            completed_payload["usage"] = usage
        yield TaskEvent(TaskEventKind.COMPLETED, completed_payload)


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------


@dataclass
class _RunOutput:
    """Internal accumulator passed to the extraction callback."""

    events: list[TaskEvent] = field(default_factory=list)
    transport: TransportSelection = TransportSelection.SUBSCRIPTION


# Briefing-retrieval callback: takes (user_id, slug, content, system_prompt_base)
# and returns the briefing TEXT (or None when nothing was recalled) plus an
# optional dict that describes the briefing (so the caller can persist it to
# the briefing log, matching the existing chat-completions flow). The hook
# used to return the briefing-augmented system prompt; it now returns the
# briefing alone so the runner can keep the system prompt static (pre-warmed
# SDK sessions seal it at connect) and thread the briefing into the rendered
# prompt instead. The hook's side contract is unchanged: it still stamps
# ``request_id`` into the briefing dict and persists the briefing-events row.
BriefingHook = Callable[
    [str, str, str, str],
    Awaitable[tuple[str | None, dict[str, Any] | None]],
]


# Extraction-completion callback: takes (user_id, slug, transcript) where the
# transcript is a dict shaped like the OpenAI response for backward compat
# with the existing extractor.
ExtractionHook = Callable[
    [str, str, dict[str, Any]],
    Awaitable[None],
]


class AgentRunner:
    """Composes config lookup + briefing + transport selection + extraction.

    A single instance per Recordatio process; held by the FastAPI app and
    invoked from the agent-endpoints router per request.
    """

    def __init__(
        self,
        *,
        subscription: SubscriptionTransport,
        api_key: ApiKeyTransport,
        rate_state: RateLimitStateStore,
        briefing_hook: BriefingHook | None = None,
        extraction_hook: ExtractionHook | None = None,
        allow_api_fallback: bool = True,
    ) -> None:
        self._sub = subscription
        self._api = api_key
        self._rate_state = rate_state
        self._briefing_hook = briefing_hook
        self._extraction_hook = extraction_hook
        self._allow_api_fallback = allow_api_fallback

    async def run(
        self,
        *,
        user_id: str,
        slug: str,
        content: str,
        config: AgentConfig,
        api_key: str | None = None,
        source_agent_id: str | None = None,
        parent_request_id: str | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[TaskEvent]:
        """Stream TaskEvents for one task. Selects transport, hydrates the
        briefing, runs the chosen transport, then schedules extraction.

        ``messages`` is the optional full conversation history off the
        request body (OpenAI-style role/content dicts, latest user message
        last). It is threaded to both transports so multi-turn context
        survives the per-turn fresh SDK session / Pontifex completion.
        ``content`` remains the latest user message and is what the
        briefing hook receives as its retrieval query (briefing on the
        whole history would dilute retrieval) and what the extraction
        hook sees as the user side of the transcript (extracting the
        whole history every turn would re-extract — and duplicate —
        memories already captured on earlier turns).

        Briefing injection: the subscription path carries the briefing in
        the rendered prompt (the ``<user_profile>`` + ``<relevant_memory>``
        blocks built by ``build_system``, inserted verbatim between the
        history and the latest message — keeps the system prompt static
        for warm SDK spares); the API-key path composes
        ``system_prompt = <agent system_prompt> + "\\n\\n" + <briefing>``
        as before. The briefing-event SQLite log gets one entry per
        request as a side effect of the briefing hook (it owns the log).

        ``source_agent_id`` and ``parent_request_id`` are call-lineage
        fields lifted off the inbound request body (mirrors the
        ``x-recordatio-source-agent-id`` header on the chat path) and
        surfaced to the extraction hook via the transcript dict so
        cross-agent provenance survives the agent-runner path.
        """
        selection = (
            select_transport(self._rate_state, user_id, slug)
            if self._allow_api_fallback else TransportSelection.SUBSCRIPTION
        )

        # Retrieve the per-turn memory briefing. The hook owns all the
        # briefing-log machinery; if it isn't wired the turn runs without
        # a briefing. A hook failure degrades to no-briefing rather than
        # aborting the turn — mirrors the legacy chat-completions path,
        # where retrieval/persist failures are ``logger.exception``-ed
        # internally and the request proceeds without a briefing (see
        # ``retrieval.retrieve`` and ``prompting.build_system``).
        #
        # The briefing is delivered transport-specifically (see
        # ``_dispatch``): the subscription path threads it into the
        # rendered prompt (history → memory → message) so the system
        # prompt stays static and warm SDK spares — whose system prompt
        # is sealed at connect — stay valid; the API-key path keeps the
        # legacy system-prompt composition.
        briefing_text: str | None = None
        briefing_data: dict[str, Any] | None = None
        if self._briefing_hook is not None:
            try:
                briefing_text, briefing_data = await self._briefing_hook(
                    user_id, slug, content, config.system_prompt,
                )
            except Exception:
                logger.exception(
                    "briefing hook failed for %s/%s; continuing without briefing",
                    user_id, slug,
                )
                briefing_text = None
                briefing_data = None

        # Surface the briefing to the client BEFORE any transport events,
        # mirroring the legacy ordering where ``agentmem.briefing``
        # arrives before the first completion chunk (telos renders the
        # provenance panel off this; see telos/agent.py briefing handler).
        # The payload is the exact ``briefing_data`` dict the chat path
        # serializes into its ``agentmem.briefing`` SSE event
        # (units/entities/relations/topic_names).
        if briefing_data is not None:
            yield TaskEvent(TaskEventKind.BRIEFING, briefing_data)

        collected: list[TaskEvent] = []
        extraction_fired = False

        async def _fire_extraction() -> None:
            """Trigger the extraction hook once, with a ``ConversationTurn``-shaped
            transcript. The hook enqueues its own background work, so this returns
            fast and does not itself await extraction.

            MUST be called BEFORE the terminal (COMPLETED/ERROR) event is yielded:
            the client (telos) disconnects the SSE stream the moment it sees that
            event, which ``aclose()``s this generator — so anything placed after
            the yield-loop never runs. That teardown race silently killed all
            agent-path extraction (see the zero-extraction incident). Firing here,
            while still inside the loop, schedules the background task on the event
            loop before the client can tear us down."""
            nonlocal extraction_fired
            if extraction_fired or self._extraction_hook is None:
                return
            extraction_fired = True
            transcript: dict[str, Any] = {
                "messages": [{"role": "user", "content": content}],
                "response_text": collect_assistant_text(collected),
            }
            # Carry call-lineage fields through so cross-agent provenance
            # survives. ``transcript_to_turn`` only consumes ``messages`` +
            # ``response_text``; extra keys are read directly by the hook.
            if source_agent_id is not None:
                transcript["source_agent_id"] = source_agent_id
            if parent_request_id is not None:
                transcript["parent_request_id"] = parent_request_id
            try:
                await self._extraction_hook(user_id, slug, transcript)
            except Exception:
                logger.exception("extraction hook failed for %s/%s", user_id, slug)

        async for event in self._dispatch(
            selection=selection,
            user_id=user_id,
            slug=slug,
            content=content,
            base_system_prompt=config.system_prompt,
            briefing=briefing_text,
            config=config,
            api_key=api_key,
            messages=messages,
        ):
            collected.append(event)
            # Fire extraction the instant the turn's content is complete —
            # BEFORE yielding the terminal event the client disconnects on.
            if event.kind is TaskEventKind.COMPLETED:
                await _fire_extraction()
            yield event

        # Fallback for turns that ended the dispatch loop without a COMPLETED
        # event and without a client disconnect (e.g. an ERROR-terminated or
        # empty turn drained in full). Idempotent — a no-op if already fired.
        await _fire_extraction()

    async def _dispatch(
        self,
        *,
        selection: TransportSelection,
        user_id: str,
        slug: str,
        content: str,
        base_system_prompt: str,
        briefing: str | None,
        config: AgentConfig,
        api_key: str | None,
        messages: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[TaskEvent]:
        if selection is TransportSelection.API_KEY:
            # Pontifex is a plain completion bridge with no rendered
            # prompt preamble to relocate into, and no warm sessions to
            # keep the system prompt static for — the briefing keeps the
            # legacy system-prompt composition here (identical wire shape
            # to the chat-completions path).
            if briefing and base_system_prompt:
                system_prompt = f"{base_system_prompt}\n\n{briefing}"
            else:
                system_prompt = briefing or base_system_prompt
            async for event in self._api.run(
                user_id=user_id,
                slug=slug,
                content=content,
                system_prompt=system_prompt,
                config=config,
                api_key=api_key,
                messages=messages,
            ):
                yield event
        else:
            # Subscription path: BASE system prompt only (sealed at SDK
            # connect, shared verbatim with warm spares); the briefing
            # rides in the rendered prompt between history and the latest
            # message.
            async for event in self._sub.run(
                user_id=user_id,
                slug=slug,
                content=content,
                system_prompt=base_system_prompt,
                config=config,
                messages=messages,
                briefing=briefing,
            ):
                yield event
