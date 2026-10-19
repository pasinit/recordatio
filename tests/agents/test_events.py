"""TaskEvent translation tests.

Covers both SDK-message translation and OpenAI-chunk translation. We use
plain dicts in place of the real SDK dataclasses so the tests don't depend
on the SDK being installed (and so we can poke specific edge cases like
``is_error=True`` with a "success" subtype, which Phase 0 found in the
wild).
"""

import json
from types import SimpleNamespace

from tpasini.recordatio.agents.events import (
    TaskEvent,
    TaskEventKind,
    collect_assistant_text,
    serialize_sse,
    translate_openai_chunk,
    translate_sdk_message,
)

# ---------------------------------------------------------------------------
# Helpers — fake SDK message types
# ---------------------------------------------------------------------------


class _FakeMsg:
    """A SimpleNamespace whose ``type(...).__name__`` is configurable.

    The translator dispatches on ``type(msg).__name__`` so we can't just
    use SimpleNamespace directly — we need the class name to match.
    """

    @classmethod
    def make(cls, name: str, **fields: object) -> object:
        new_cls = type(name, (SimpleNamespace,), {})
        return new_cls(**fields)


def _assistant(*blocks: dict) -> object:
    return _FakeMsg.make("AssistantMessage", content=list(blocks))


# ---------------------------------------------------------------------------
# SystemMessage
# ---------------------------------------------------------------------------


def test_init_system_message_emits_started() -> None:
    msg = _FakeMsg.make("SystemMessage", subtype="init", data={"session_id": "sess-1"})
    events = translate_sdk_message(msg)
    assert len(events) == 1
    assert events[0].kind is TaskEventKind.STARTED
    assert events[0].payload["session_id"] == "sess-1"


def test_api_retry_emits_retry() -> None:
    msg = _FakeMsg.make(
        "SystemMessage",
        subtype="api_retry",
        data={"attempt": 3, "max_retries": 10, "retry_delay_ms": 1500},
    )
    events = translate_sdk_message(msg)
    assert len(events) == 1
    assert events[0].kind is TaskEventKind.RETRY
    assert events[0].payload == {"attempt": 3, "max_retries": 10, "retry_delay_ms": 1500}


def test_other_system_subtypes_emit_nothing() -> None:
    msg = _FakeMsg.make("SystemMessage", subtype="compact", data={})
    assert translate_sdk_message(msg) == []


# ---------------------------------------------------------------------------
# AssistantMessage content blocks
# ---------------------------------------------------------------------------


def test_text_block_emits_message_event() -> None:
    events = translate_sdk_message(_assistant({"type": "text", "text": "hi there"}))
    assert [e.kind for e in events] == [TaskEventKind.MESSAGE]
    assert events[0].payload == {"text": "hi there"}


def test_thinking_block_emits_thinking_event() -> None:
    events = translate_sdk_message(_assistant({"type": "thinking", "thinking": "let me see…"}))
    assert [e.kind for e in events] == [TaskEventKind.THINKING]
    assert events[0].payload == {"text": "let me see…"}


def test_tool_use_block_emits_tool_call_event() -> None:
    events = translate_sdk_message(_assistant(
        {"type": "tool_use", "id": "call-1", "name": "search", "input": {"q": "foo"}},
    ))
    assert [e.kind for e in events] == [TaskEventKind.TOOL_CALL]
    assert events[0].payload == {"call_id": "call-1", "name": "search", "input": {"q": "foo"}}


def test_mixed_blocks_preserve_order() -> None:
    events = translate_sdk_message(_assistant(
        {"type": "thinking", "thinking": "hmm"},
        {"type": "tool_use", "id": "c1", "name": "n", "input": {}},
        {"type": "text", "text": "done"},
    ))
    assert [e.kind for e in events] == [
        TaskEventKind.THINKING,
        TaskEventKind.TOOL_CALL,
        TaskEventKind.MESSAGE,
    ]


# ---------------------------------------------------------------------------
# Runtime-internal meta-tool filtering
# ---------------------------------------------------------------------------


def _user_with_results(*blocks: dict) -> object:
    return _FakeMsg.make("UserMessage", content=list(blocks))


def test_internal_tool_use_emits_no_event() -> None:
    events = translate_sdk_message(_assistant(
        {"type": "tool_use", "id": "call-ts", "name": "ToolSearch", "input": {"query": "select:Foo"}},
    ))
    assert events == []


def test_internal_tool_result_is_suppressed_too() -> None:
    # tool_use and its tool_result arrive in two separate SDK messages.
    translate_sdk_message(_assistant(
        {"type": "tool_use", "id": "call-ts-2", "name": "ToolSearch", "input": {}},
    ))
    events = translate_sdk_message(_user_with_results(
        {"type": "tool_result", "tool_use_id": "call-ts-2", "content": "schemas loaded"},
    ))
    assert events == []


def test_real_tools_are_unaffected_by_internal_filtering() -> None:
    # An internal call interleaved with a real tool call: only the real
    # tool's call and result reach the wire, in order.
    call_events = translate_sdk_message(_assistant(
        {"type": "tool_use", "id": "call-ts-3", "name": "ToolSearch", "input": {}},
        {"type": "tool_use", "id": "call-real", "name": "telos_search", "input": {"q": "x"}},
    ))
    assert [e.kind for e in call_events] == [TaskEventKind.TOOL_CALL]
    assert call_events[0].payload["name"] == "telos_search"

    result_events = translate_sdk_message(_user_with_results(
        {"type": "tool_result", "tool_use_id": "call-ts-3", "content": "internal"},
        {"type": "tool_result", "tool_use_id": "call-real", "content": "hit"},
    ))
    assert [e.kind for e in result_events] == [TaskEventKind.TOOL_RESULT]
    assert result_events[0].payload == {"call_id": "call-real", "result": "hit", "is_error": False}


def test_filtered_call_id_set_is_bounded_and_drained() -> None:
    from tpasini.recordatio.agents import events as events_mod

    # Many internal calls whose results never arrive must not grow the
    # tracking set without bound.
    for i in range(events_mod._FILTERED_CALL_IDS_MAX * 2):
        translate_sdk_message(_assistant(
            {"type": "tool_use", "id": f"call-bound-{i}", "name": "ToolSearch", "input": {}},
        ))
    assert len(events_mod._FILTERED_CALL_IDS) <= events_mod._FILTERED_CALL_IDS_MAX

    # A matching result pops its id back out of the set.
    translate_sdk_message(_user_with_results(
        {"type": "tool_result", "tool_use_id": "call-bound-511", "content": "ok"},
    ))
    assert "call-bound-511" not in events_mod._FILTERED_CALL_IDS


# ---------------------------------------------------------------------------
# RateLimitEvent
# ---------------------------------------------------------------------------


def test_rate_limit_event_passes_through_info_fields() -> None:
    msg = _FakeMsg.make(
        "RateLimitEvent",
        rate_limit_info={
            "status": "allowed",
            "rate_limit_type": "five_hour",
            "overage_status": "rejected",
            "overage_disabled_reason": "out_of_credits",
            "utilization": 0.42,
            "resets_at": "2026-06-07T20:35:00Z",
        },
    )
    events = translate_sdk_message(msg)
    assert [e.kind for e in events] == [TaskEventKind.RATE_LIMIT]
    payload = events[0].payload
    assert payload["overage_status"] == "rejected"
    assert payload["overage_disabled_reason"] == "out_of_credits"
    assert payload["utilization"] == 0.42


# ---------------------------------------------------------------------------
# ResultMessage — Phase 0 gotcha #7: subtype="success" with is_error=True
# ---------------------------------------------------------------------------


def test_result_with_is_error_emits_error_not_completed() -> None:
    """Phase 0 found this exact combination during a connection failure:
    ``subtype="success"`` but ``is_error=True``. The translator must
    treat ``is_error`` as the source of truth, not ``subtype``."""
    msg = _FakeMsg.make(
        "ResultMessage",
        subtype="success", is_error=True, stop_reason="stop_sequence",
        duration_ms=0, duration_api_ms=0, num_turns=1, session_id="s",
        total_cost_usd=0.0, usage=None, result=None, structured_output=None,
        model_usage=None, permission_denials=[], deferred_tool_use=None,
        errors=[{"message": "connection refused"}], api_error_status=None,
    )
    events = translate_sdk_message(msg)
    assert [e.kind for e in events] == [TaskEventKind.ERROR]
    assert events[0].payload["errors"] == [{"message": "connection refused"}]


def test_result_with_is_error_false_emits_completed() -> None:
    msg = _FakeMsg.make(
        "ResultMessage",
        subtype="success", is_error=False, stop_reason="end_turn",
        duration_ms=1234, duration_api_ms=1100, num_turns=1, session_id="s",
        total_cost_usd=0.0, usage=None, result=None, structured_output=None,
        model_usage=None, permission_denials=[], deferred_tool_use=None,
        errors=[], api_error_status=None,
    )
    events = translate_sdk_message(msg)
    assert [e.kind for e in events] == [TaskEventKind.COMPLETED]
    assert events[0].payload["stop_reason"] == "end_turn"
    assert events[0].payload["duration_ms"] == 1234


def test_completed_carries_usage_and_cost() -> None:
    """The ``completed`` payload must carry the ResultMessage's ``usage``
    dict verbatim plus ``total_cost_usd`` as flat ``cost_usd`` — the
    contract telos's ``_usage_from_completed`` parses for cost recording
    (nested ``usage`` with input_tokens/output_tokens, flat ``cost_usd``).
    """
    usage = {
        "input_tokens": 1200,
        "output_tokens": 340,
        "cache_creation_input_tokens": 100,
        "cache_read_input_tokens": 5000,
    }
    msg = _FakeMsg.make(
        "ResultMessage",
        subtype="success", is_error=False, stop_reason="end_turn",
        duration_ms=1234, duration_api_ms=1100, num_turns=1, session_id="s",
        total_cost_usd=0.0421, usage=usage, result=None, structured_output=None,
        model_usage=None, permission_denials=[], deferred_tool_use=None,
        errors=[], api_error_status=None,
    )
    events = translate_sdk_message(msg)
    assert [e.kind for e in events] == [TaskEventKind.COMPLETED]
    assert events[0].payload["usage"] == usage
    assert events[0].payload["cost_usd"] == 0.0421


def test_completed_omits_usage_and_cost_when_absent() -> None:
    """No usage on the ResultMessage → the keys must be absent (not None,
    not zeros) so consumers can fall back to their own estimates."""
    msg = _FakeMsg.make(
        "ResultMessage",
        subtype="success", is_error=False, stop_reason="end_turn",
        duration_ms=1234, duration_api_ms=1100, num_turns=1, session_id="s",
        total_cost_usd=None, usage=None, result=None, structured_output=None,
        model_usage=None, permission_denials=[], deferred_tool_use=None,
        errors=[], api_error_status=None,
    )
    events = translate_sdk_message(msg)
    assert [e.kind for e in events] == [TaskEventKind.COMPLETED]
    assert "usage" not in events[0].payload
    assert "cost_usd" not in events[0].payload


# ---------------------------------------------------------------------------
# OpenAI-chunk translation (fallback path)
# ---------------------------------------------------------------------------


def test_openai_chunk_text_delta_emits_message() -> None:
    chunk = {"choices": [{"index": 0, "delta": {"content": "hello "}, "finish_reason": None}]}
    events = translate_openai_chunk(chunk)
    assert [e.kind for e in events] == [TaskEventKind.MESSAGE]
    assert events[0].payload == {"text": "hello "}


def test_openai_chunk_no_content_emits_nothing() -> None:
    assert translate_openai_chunk({"choices": [{"delta": {}}]}) == []


def test_openai_chunk_tool_call_delta_emits_tool_call() -> None:
    chunk = {"choices": [{
        "index": 0,
        "delta": {"tool_calls": [
            {"index": 0, "id": "call-1", "function": {"name": "search", "arguments": "{\"q\": "}},
        ]},
    }]}
    events = translate_openai_chunk(chunk)
    assert [e.kind for e in events] == [TaskEventKind.TOOL_CALL]
    assert events[0].payload == {
        "call_id": "call-1",
        "name": "search",
        "input_fragment": '{"q": ',
        "index": 0,
    }


# ---------------------------------------------------------------------------
# StreamEvent (include_partial_messages) → message_delta / thinking_delta
# ---------------------------------------------------------------------------


def _stream_event(event: dict, parent_tool_use_id: str | None = None) -> object:
    """Build a fake SDK ``StreamEvent`` (uuid/session_id/event/parent_tool_use_id)."""
    return _FakeMsg.make(
        "StreamEvent",
        uuid="u-1",
        session_id="sess-1",
        event=event,
        parent_tool_use_id=parent_tool_use_id,
    )


def test_stream_event_text_deltas_emit_message_delta_in_order() -> None:
    chunks = ["Hel", "lo ", "world"]
    events: list[TaskEvent] = []
    for chunk in chunks:
        events.extend(translate_sdk_message(_stream_event(
            {"type": "content_block_delta", "index": 0,
             "delta": {"type": "text_delta", "text": chunk}},
        )))
    assert [e.kind for e in events] == [TaskEventKind.MESSAGE_DELTA] * 3
    assert [e.payload["text"] for e in events] == chunks


def test_full_message_still_follows_deltas() -> None:
    # The wire contract: deltas stream first, then the complete
    # AssistantMessage still emits the authoritative MESSAGE event.
    events = translate_sdk_message(_stream_event(
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": "Hello world"}},
    ))
    events.extend(translate_sdk_message(
        _assistant({"type": "text", "text": "Hello world"}),
    ))
    assert [e.kind for e in events] == [
        TaskEventKind.MESSAGE_DELTA,
        TaskEventKind.MESSAGE,
    ]
    assert events[1].payload["text"] == "Hello world"


def test_stream_event_thinking_delta_emits_thinking_delta() -> None:
    events = translate_sdk_message(_stream_event(
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "thinking_delta", "thinking": "hmm"}},
    ))
    assert [e.kind for e in events] == [TaskEventKind.THINKING_DELTA]
    assert events[0].payload["text"] == "hmm"


def test_subagent_stream_events_are_excluded() -> None:
    # parent_tool_use_id set → the delta belongs to a subagent spawned via
    # the Task tool; forwarding it would interleave token streams.
    events = translate_sdk_message(_stream_event(
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": "nested"}},
        parent_tool_use_id="toolu_123",
    ))
    assert events == []


def test_other_stream_event_subtypes_emit_nothing() -> None:
    for raw in (
        {"type": "message_start", "message": {"id": "msg_1"}},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_stop", "index": 0},
        # Raw Anthropic "message_delta" (stop_reason/usage carrier) — NOT
        # our message_delta TaskEvent; must not be confused with text.
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
         "usage": {"output_tokens": 5}},
        {"type": "message_stop"},
        # Non-text deltas on a content_block_delta.
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "input_json_delta", "partial_json": '{"a"'}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "signature_delta", "signature": "abc"}},
    ):
        assert translate_sdk_message(_stream_event(raw)) == []


def test_collect_assistant_text_ignores_deltas() -> None:
    # Transcript assembly must keep using authoritative MESSAGE events
    # only — counting deltas too would double every streamed character.
    events = [
        TaskEvent(TaskEventKind.MESSAGE_DELTA, {"text": "Hel"}),
        TaskEvent(TaskEventKind.MESSAGE_DELTA, {"text": "lo"}),
        TaskEvent(TaskEventKind.THINKING_DELTA, {"text": "hmm"}),
        TaskEvent(TaskEventKind.MESSAGE, {"text": "Hello"}),
    ]
    assert collect_assistant_text(events) == "Hello"


# ---------------------------------------------------------------------------
# Wire serialisation
# ---------------------------------------------------------------------------


def test_serialize_sse_message_delta_is_kind_generic() -> None:
    raw = serialize_sse(TaskEvent(TaskEventKind.MESSAGE_DELTA, {"text": "He"})).decode()
    assert raw.startswith("event: message_delta\n")
    body_line = next(line for line in raw.splitlines() if line.startswith("data: "))
    assert json.loads(body_line[len("data: "):]) == {"kind": "message_delta", "text": "He"}


def test_serialize_sse_produces_event_and_data_lines() -> None:
    raw = serialize_sse(TaskEvent(TaskEventKind.MESSAGE, {"text": "hi"})).decode()
    assert raw.startswith("event: message\n")
    body_line = next(line for line in raw.splitlines() if line.startswith("data: "))
    payload = json.loads(body_line[len("data: "):])
    assert payload == {"kind": "message", "text": "hi"}
    assert raw.endswith("\n\n")
