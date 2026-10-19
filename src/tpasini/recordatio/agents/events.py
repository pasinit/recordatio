"""TaskEvent stream — Recordatio's wire format for an agent turn.

Per ``MIGRATION_PLAN.md`` Phase 1, Recordatio emits a stream of small typed
events over SSE during a chat turn. This module owns:

- The ``TaskEvent`` dataclass + ``TaskEventKind`` enum (the wire types).
- ``translate_sdk_message()`` — fan-out from a single ``claude_agent_sdk``
  message dataclass into zero or more ``TaskEvent``s.
- ``translate_openai_chunk()`` — fan-out from one OpenAI-SSE chunk
  (the existing ``Pontifex.stream()`` output) into zero or more
  ``TaskEvent``s, used by ``ApiKeyTransport``.
- ``serialize_sse()`` — pure helper that renders a ``TaskEvent`` to the
  ``event:\\ndata:\\n\\n`` SSE wire format.

The translator is intentionally permissive: any field the SDK leaves
absent on a message is mirrored as ``None``/``[]`` rather than raising,
because the SDK evolves and we don't want to crash on an unknown subtype.
"""

import json
import logging
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)

# Tracks SDK message class names that we've already logged as "unknown" so
# we don't spam the logs once per message. Process-wide, intentionally
# mutated from ``translate_sdk_message`` — see finding #13 on PR #133.
_UNKNOWN_SDK_TYPES_LOGGED: set[str] = set()

# Runtime-internal meta-tools of the bundled Claude Code CLI. These are
# harness plumbing the runtime can invoke on its own — they are not part of
# any seeded agent config, and surfacing them as ``tool_call`` events makes
# the UI render mystery tools (observed in prod: a ``ToolSearch`` call —
# the CLI's deferred-tool-schema loader — leaked into a user's chat).
#
# Be conservative here: only names that are unambiguously plumbing belong in
# this set. Tools a seeded config could legitimately expose (``Skill``,
# ``ExitPlanMode``, ``ListMcpResources``/``ReadMcpResource``, MCP tools, …)
# must NOT be filtered.
#
# - ``ToolSearch``       — loads deferred tool schemas for the runtime.
# - ``StructuredOutput`` — the mechanism the runtime uses to capture a
#   structured-output payload when an output format is configured.
# - ``TodoWrite``        — the CLI's internal task-list bookkeeping.
_INTERNAL_TOOLS = frozenset({"ToolSearch", "StructuredOutput", "TodoWrite"})

# Tool names we've already logged as filtered, once per (tool, process).
_INTERNAL_TOOLS_LOGGED: set[str] = set()

# call_ids of filtered internal tool_use blocks, so the matching
# ``tool_result`` (which arrives in a *separate* ``UserMessage``) can be
# suppressed too. Bounded LRU (insertion-ordered dict, oldest evicted) so a
# long-lived process can't grow it without bound; entries are popped as soon
# as their result is seen, so 256 in-flight internal calls is generous.
_FILTERED_CALL_IDS: OrderedDict[str, None] = OrderedDict()
_FILTERED_CALL_IDS_MAX = 256


class TaskEventKind(StrEnum):
    """The TaskEvent ``kind`` enum from the migration plan.

    ``retry`` is added beyond the plan's listing — Phase 0 found the SDK
    emits ``SystemMessage(subtype="api_retry")`` events during outages, and
    surfacing them so the UI can render "reconnecting" rather than appear
    frozen was called out as desirable (see MIGRATION_NOTES.md gotcha #8).

    ``briefing`` carries the hydrated memory briefing payload (matches the
    legacy chat-completions ``agentmem.briefing`` SSE event), so the SPA can
    render the same provenance UI when consuming the new agent stream.
    """

    STARTED = "started"
    THINKING = "thinking"
    THINKING_DELTA = "thinking_delta"
    MESSAGE_DELTA = "message_delta"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    SUBAGENT_STARTED = "subagent_started"
    SUBAGENT_COMPLETED = "subagent_completed"
    MESSAGE = "message"
    RETRY = "retry"
    RATE_LIMIT = "rate_limit"
    BRIEFING = "briefing"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class TaskEvent:
    """One event in the SSE stream of an agent turn.

    The ``payload`` shape is kind-specific. We don't enforce a per-kind
    schema at this layer because clients only need to consume what they
    care about — keeping the wire JSON open-ended makes downstream
    evolution cheaper. The Python dataclass is just a convenient carrier.
    """

    kind: TaskEventKind
    payload: dict[str, Any] = field(default_factory=dict)


def serialize_sse(event: TaskEvent) -> bytes:
    """Render a ``TaskEvent`` as an SSE wire frame."""
    body = {"kind": event.kind.value, **event.payload}
    return f"event: {event.kind.value}\ndata: {json.dumps(body, default=str)}\n\n".encode()


# ---------------------------------------------------------------------------
# SDK message → TaskEvent
# ---------------------------------------------------------------------------


def _sdk_message_to_dict(msg: Any) -> dict[str, Any]:
    """Best-effort dataclass → dict conversion that survives SDK shape drift.

    The SDK returns dataclasses; ``asdict`` works on them but may also be
    handed a plain dict in tests. We just normalise to dict and rely on the
    callers to ``.get()`` keys defensively.

    We use ``dataclasses.is_dataclass`` to decide whether to call
    ``asdict``: tests substitute SimpleNamespace-backed fakes that have
    a ``__dict__`` but are not real dataclasses, and ``asdict`` would
    raise on them. The ``vars()`` fallback covers both that case and
    any future SDK message type that ships as a plain class.
    """
    if isinstance(msg, dict):
        return msg
    if is_dataclass(msg) and not isinstance(msg, type):
        return asdict(msg)
    return dict(vars(msg)) if hasattr(msg, "__dict__") else {}


def _remember_filtered_call_id(call_id: Any) -> None:
    """Record the call_id of a suppressed internal tool_use (bounded LRU)."""
    if not isinstance(call_id, str) or not call_id:
        return
    _FILTERED_CALL_IDS[call_id] = None
    _FILTERED_CALL_IDS.move_to_end(call_id)
    while len(_FILTERED_CALL_IDS) > _FILTERED_CALL_IDS_MAX:
        _FILTERED_CALL_IDS.popitem(last=False)


def _tool_call_event(block_dict: dict[str, Any]) -> TaskEvent | None:
    """Build a TOOL_CALL event for a tool_use block, or ``None`` if filtered.

    Internal meta-tools (``_INTERNAL_TOOLS``) emit no event; their call_id is
    remembered so the matching tool_result is suppressed as well, and the
    filtering is logged once per (tool, process) at debug level.
    """
    name = block_dict.get("name")
    if name in _INTERNAL_TOOLS:
        _remember_filtered_call_id(block_dict.get("id"))
        if name not in _INTERNAL_TOOLS_LOGGED:
            _INTERNAL_TOOLS_LOGGED.add(name)
            logger.debug(
                "translate_sdk_message: suppressing runtime-internal tool %r "
                "from the event stream (logged once per tool per process)",
                name,
            )
        return None
    return TaskEvent(
        TaskEventKind.TOOL_CALL,
        {
            "call_id": block_dict.get("id"),
            "name": name,
            "input": block_dict.get("input") or {},
        },
    )


def translate_sdk_message(msg: Any) -> list[TaskEvent]:
    """Fan-out one SDK message into zero or more TaskEvents.

    Multiple events can come out of a single SDK message because an
    ``AssistantMessage`` may carry text, tool-use, and thinking blocks
    at once. We translate them in declaration order so streaming clients
    render them in the same order the model produced them.
    """
    cls_name = type(msg).__name__
    data = _sdk_message_to_dict(msg)
    events: list[TaskEvent] = []

    if cls_name == "SystemMessage":
        subtype = data.get("subtype")
        if subtype == "init":
            # The init message also carries the session_id; surface it
            # so a UI can correlate with later events.
            events.append(TaskEvent(
                TaskEventKind.STARTED,
                {"session_id": (data.get("data") or {}).get("session_id")},
            ))
        elif subtype == "api_retry":
            payload = data.get("data") or {}
            events.append(TaskEvent(
                TaskEventKind.RETRY,
                {
                    "attempt": payload.get("attempt"),
                    "max_retries": payload.get("max_retries"),
                    "retry_delay_ms": payload.get("retry_delay_ms"),
                },
            ))
        # Other system subtypes are not surfaced — they're internal noise
        # (e.g. `compact`, `tool_use_id_correction`) that doesn't add
        # signal for a chat UI.
        return events

    if cls_name == "AssistantMessage":
        for block in data.get("content") or []:
            # Prefer typed-class dispatch over key-presence heuristics: the
            # SDK ships block dataclasses (e.g. ``ThinkingBlock``,
            # ``ToolUseBlock``, ``TextBlock``) and their class names are
            # the cleanest signal. Falls through to dict-based dispatch
            # for the asdict()-flattened case and for raw dict blocks
            # (which is what the SDK actually hands us via ``asdict``
            # at this layer — and what tests inject).
            block_cls = type(block).__name__
            block_dict: dict[str, Any] = block if isinstance(block, dict) else _sdk_message_to_dict(block)
            block_type = block_dict.get("type")

            # Class-name dispatch first.
            if block_cls == "ThinkingBlock":
                events.append(TaskEvent(
                    TaskEventKind.THINKING,
                    {"text": block_dict.get("thinking") or block_dict.get("text") or ""},
                ))
                continue
            if block_cls == "ToolUseBlock":
                tool_event = _tool_call_event(block_dict)
                if tool_event is not None:
                    events.append(tool_event)
                continue
            if block_cls == "TextBlock":
                events.append(TaskEvent(
                    TaskEventKind.MESSAGE,
                    {"text": block_dict.get("text", "")},
                ))
                continue

            # Dict-based dispatch. Prefer the explicit ``type`` discriminator
            # over key-presence heuristics; the heuristics remain only as a
            # last-resort fallback for ``asdict()``-flattened blocks that
            # have lost their ``type`` field.
            if block_type == "thinking":
                events.append(TaskEvent(
                    TaskEventKind.THINKING,
                    {"text": block_dict.get("thinking") or block_dict.get("text") or ""},
                ))
            elif block_type == "tool_use":
                tool_event = _tool_call_event(block_dict)
                if tool_event is not None:
                    events.append(tool_event)
            elif block_type == "text":
                events.append(TaskEvent(
                    TaskEventKind.MESSAGE,
                    {"text": block_dict.get("text", "")},
                ))
            # Fallback heuristics for dataclass blocks that lost ``type``
            # under ``asdict``. The order below matters: "thinking" must be
            # checked before "name"/"input" because a thinking block also
            # carries a ``text`` field on some SDK versions.
            elif "thinking" in block_dict:
                events.append(TaskEvent(
                    TaskEventKind.THINKING,
                    {"text": block_dict.get("thinking") or block_dict.get("text") or ""},
                ))
            elif "name" in block_dict and "input" in block_dict:
                tool_event = _tool_call_event(block_dict)
                if tool_event is not None:
                    events.append(tool_event)
            elif "text" in block_dict:
                events.append(TaskEvent(
                    TaskEventKind.MESSAGE,
                    {"text": block_dict.get("text", "")},
                ))
        return events

    if cls_name == "StreamEvent":
        # Partial-message events, emitted when ``include_partial_messages``
        # is on (SubscriptionTransport sets it). ``StreamEvent.event`` is
        # the raw Anthropic API stream event dict; we surface ONLY text and
        # thinking deltas of the *top-level* agent as ``message_delta`` /
        # ``thinking_delta``. The complete ``MESSAGE``/``THINKING`` event
        # (from the subsequent AssistantMessage) remains the authoritative
        # text — delta consumers reconcile against it, delta-unaware
        # consumers ignore the unknown kinds and keep working unchanged.
        #
        # Nested deltas: the SDK stamps ``parent_tool_use_id`` on stream
        # events that belong to a subagent (the Task tool-use that spawned
        # it); top-level agent events carry ``None``. Forwarding subagent
        # deltas would interleave two token streams into one visible
        # message, so anything with a parent is dropped.
        if data.get("parent_tool_use_id") is not None:
            return events
        raw = data.get("event") or {}
        if raw.get("type") != "content_block_delta":
            # message_start/stop, content_block_start/stop, the raw
            # ``message_delta`` (stop_reason/usage carrier) and any
            # future subtypes carry no renderable text — no events.
            return events
        delta = raw.get("delta") or {}
        delta_type = delta.get("type")
        if delta_type == "text_delta":
            text = delta.get("text")
            if isinstance(text, str) and text:
                events.append(TaskEvent(TaskEventKind.MESSAGE_DELTA, {"text": text}))
        elif delta_type == "thinking_delta":
            text = delta.get("thinking")
            if isinstance(text, str) and text:
                events.append(TaskEvent(TaskEventKind.THINKING_DELTA, {"text": text}))
        # input_json_delta / signature_delta: tool inputs already stream
        # nowhere today (TOOL_CALL carries the complete input) — ignored.
        return events

    if cls_name == "UserMessage":
        # User messages from the SDK in agent flow typically carry tool
        # results back to the model. Translate those to tool_result so
        # they show up on the wire (matters once we wire MCP tools in
        # Phase 3).
        for block in data.get("content") or []:
            if isinstance(block, dict) and (block.get("type") == "tool_result" or "tool_use_id" in block):
                call_id = block.get("tool_use_id")
                if isinstance(call_id, str) and call_id in _FILTERED_CALL_IDS:
                    # Result of a suppressed internal tool_use (see
                    # ``_INTERNAL_TOOLS``). Pop the id — each call has
                    # exactly one result, so the LRU stays small.
                    _FILTERED_CALL_IDS.pop(call_id, None)
                    continue
                events.append(TaskEvent(
                    TaskEventKind.TOOL_RESULT,
                    {
                        "call_id": call_id,
                        "result": block.get("content"),
                        "is_error": bool(block.get("is_error")),
                    },
                ))
        return events

    if cls_name == "RateLimitEvent":
        info = data.get("rate_limit_info") or {}
        events.append(TaskEvent(
            TaskEventKind.RATE_LIMIT,
            {
                "status": info.get("status"),
                "rate_limit_type": info.get("rate_limit_type"),
                "overage_status": info.get("overage_status"),
                "overage_disabled_reason": info.get("overage_disabled_reason"),
                "utilization": info.get("utilization"),
                "resets_at": info.get("resets_at"),
            },
        ))
        return events

    if cls_name == "ResultMessage":
        # is_error is the source of truth — Phase 0 found ``subtype`` can
        # be "success" even when ``is_error=True`` (synthetic <synthetic>
        # assistant message during a connection failure). See
        # MIGRATION_NOTES.md gotcha #7.
        if data.get("is_error"):
            events.append(TaskEvent(
                TaskEventKind.ERROR,
                {
                    "stop_reason": data.get("stop_reason"),
                    "errors": data.get("errors") or [],
                    "api_error_status": data.get("api_error_status"),
                },
            ))
        else:
            payload: dict[str, Any] = {
                "stop_reason": data.get("stop_reason"),
                "duration_ms": data.get("duration_ms"),
                "num_turns": data.get("num_turns"),
                "session_id": data.get("session_id"),
            }
            # Usage/cost for downstream cost recording — telos's
            # ``_usage_from_completed`` plucks a nested ``usage`` dict and
            # a flat ``cost_usd`` off this payload. The SDK's
            # ``ResultMessage`` carries ``usage`` as a dict
            # (input_tokens/output_tokens plus cache fields) and
            # ``total_cost_usd`` as a float; both are optional, and we
            # omit the keys entirely when absent so consumers can
            # distinguish "no usage reported" from zeros.
            usage = data.get("usage")
            if isinstance(usage, dict) and usage:
                payload["usage"] = usage
            cost_usd = data.get("total_cost_usd")
            if cost_usd is not None:
                payload["cost_usd"] = cost_usd
            events.append(TaskEvent(TaskEventKind.COMPLETED, payload))
        return events

    # Unknown SDK message type — log once per type per process so we
    # notice schema drift without flooding logs (finding #13 on PR #133).
    if cls_name not in _UNKNOWN_SDK_TYPES_LOGGED:
        _UNKNOWN_SDK_TYPES_LOGGED.add(cls_name)
        logger.warning(
            "translate_sdk_message: ignoring unknown SDK message type %r "
            "(this is logged only once per type per process)",
            cls_name,
        )
    return events


# ---------------------------------------------------------------------------
# OpenAI-SSE chunk → TaskEvent (fallback path)
# ---------------------------------------------------------------------------


def translate_openai_chunk(chunk: dict[str, Any]) -> list[TaskEvent]:
    """Fan-out one ``Pontifex.stream()`` chunk into zero or more TaskEvents.

    Used by ``ApiKeyTransport`` to render the same wire format as the
    subscription path. We only emit ``MESSAGE`` for text deltas and
    ``TOOL_CALL`` for tool-call deltas; the runner emits ``COMPLETED``
    once the stream is exhausted because OpenAI streams don't carry a
    terminal message of their own (``finish_reason`` is on a choice).
    """
    events: list[TaskEvent] = []
    for choice in chunk.get("choices") or []:
        delta = choice.get("delta") or {}
        text = delta.get("content")
        if text:
            events.append(TaskEvent(TaskEventKind.MESSAGE, {"text": text}))
        for tc in delta.get("tool_calls") or []:
            # OpenAI streams tool calls in fragments — the call_id and
            # name come first, then the arguments stream as JSON
            # fragments. Downstream consumers will need to reassemble;
            # for the v1 wire format we just forward the fragment.
            events.append(TaskEvent(
                TaskEventKind.TOOL_CALL,
                {
                    "call_id": tc.get("id"),
                    "name": (tc.get("function") or {}).get("name"),
                    "input_fragment": (tc.get("function") or {}).get("arguments"),
                    "index": tc.get("index"),
                },
            ))
    return events


def extract_openai_finish_reason(chunk: dict[str, Any]) -> str | None:
    """Pull ``finish_reason`` off the first choice of a terminal chunk.

    OpenAI-style streams put ``finish_reason`` on a choice in the *last*
    chunk (most providers send a final delta-less chunk for this).
    Returns ``None`` when the chunk has no terminal finish reason.

    Used by ``ApiKeyTransport`` to emit a faithful ``COMPLETED`` event
    (stop_reason mirrors the upstream finish_reason) and to surface
    ``content_filter`` as an ``ERROR`` rather than a clean stop —
    finding #11 on PR #133.
    """
    for choice in chunk.get("choices") or []:
        finish_reason = choice.get("finish_reason")
        if finish_reason:
            return str(finish_reason)
    return None


def collect_assistant_text(events: Iterable[TaskEvent]) -> str:
    """Concatenate all ``MESSAGE`` event texts in order.

    Used by the runner to assemble the transcript handed to the
    knowledge-extraction step.
    """
    parts: list[str] = []
    for event in events:
        if event.kind is TaskEventKind.MESSAGE:
            text = event.payload.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)
