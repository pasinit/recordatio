import json

from tpasini.recordatio.proxy.stream_parser import StreamAccumulator


def _data_line(event: dict) -> bytes:
    return f"data: {json.dumps(event)}\n".encode()


def test_empty_stream():
    acc = StreamAccumulator()
    assert acc.text == ""


def test_single_delta():
    acc = StreamAccumulator()
    acc.feed(_data_line({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hello"}}))
    assert acc.text == "hello"


def test_multiple_deltas_same_block():
    acc = StreamAccumulator()
    for word in ("foo", " ", "bar"):
        acc.feed(_data_line({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": word}}))
    assert acc.text == "foo bar"


def test_multiple_blocks_ordered():
    acc = StreamAccumulator()
    acc.feed(_data_line({"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "world"}}))
    acc.feed(_data_line({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hello "}}))
    assert acc.text == "hello world"


def test_ignores_non_text_delta():
    acc = StreamAccumulator()
    delta = {"type": "input_json_delta", "partial_json": "{}"}
    acc.feed(_data_line({"type": "content_block_delta", "index": 0, "delta": delta}))
    assert acc.text == ""


def test_ignores_other_event_types():
    acc = StreamAccumulator()
    acc.feed(_data_line({"type": "message_start", "message": {}}))
    acc.feed(b"data: [DONE]\n")
    acc.feed(b"\n")
    assert acc.text == ""


def test_stop_reason_captured():
    acc = StreamAccumulator()
    assert acc.stop_reason is None
    acc.feed(_data_line({"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None}}))
    assert acc.stop_reason == "tool_use"


def test_stop_reason_end_turn():
    acc = StreamAccumulator()
    acc.feed(_data_line({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}}))
    acc.feed(_data_line({"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}}))
    assert acc.stop_reason == "end_turn"
    assert acc.text == "hi"


def test_content_blocks_text_only():
    acc = StreamAccumulator()
    acc.feed(_data_line({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}))
    acc.feed(_data_line({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hello"}}))
    assert acc.content_blocks == [{"type": "text", "text": "hello"}]


def test_content_blocks_tool_use_only():
    acc = StreamAccumulator()
    acc.feed(_data_line({
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "tool_use", "id": "tu_1", "name": "memory_search"},
    }))
    acc.feed(_data_line({
        "type": "content_block_delta", "index": 0,
        "delta": {"type": "input_json_delta", "partial_json": '{"query": "foo"}'},
    }))
    assert acc.content_blocks == [
        {"type": "tool_use", "id": "tu_1", "name": "memory_search", "input": {"query": "foo"}}
    ]


def test_content_blocks_text_then_tool_use():
    acc = StreamAccumulator()
    acc.feed(_data_line({
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""},
    }))
    acc.feed(_data_line({
        "type": "content_block_delta", "index": 0,
        "delta": {"type": "text_delta", "text": "Let me search."},
    }))
    acc.feed(_data_line({
        "type": "content_block_start", "index": 1,
        "content_block": {"type": "tool_use", "id": "tu_1", "name": "web_search"},
    }))
    acc.feed(_data_line({
        "type": "content_block_delta", "index": 1,
        "delta": {"type": "input_json_delta", "partial_json": '{"query": "bar"}'},
    }))
    blocks = acc.content_blocks
    assert blocks == [
        {"type": "text", "text": "Let me search."},
        {"type": "tool_use", "id": "tu_1", "name": "web_search", "input": {"query": "bar"}},
    ]


def test_content_blocks_skips_empty_text():
    acc = StreamAccumulator()
    # tool_use at index 0, no text blocks at all
    acc.feed(_data_line({
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "tool_use", "id": "tu_1", "name": "memory_search"},
    }))
    acc.feed(_data_line({
        "type": "content_block_delta", "index": 0,
        "delta": {"type": "input_json_delta", "partial_json": '{"query": "x"}'},
    }))
    blocks = acc.content_blocks
    assert all(b["type"] == "tool_use" for b in blocks)


def test_ignores_malformed_lines():
    acc = StreamAccumulator()
    acc.feed(b"data: not-json\n")
    acc.feed(b"not-an-sse-line\n")
    assert acc.text == ""
