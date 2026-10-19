"""``transcript_to_turn`` — minimal coverage for the SDK-side input shape.

The migration plan asks for a minimal refactor of ``extraction/extractor.py``
that lets the extractor accept either an OpenAI response body (current) or
a transcript dict assembled from SDK events. The actual change is a thin
adapter that returns a ``ConversationTurn``; we verify shape and edge cases
without invoking the extractor's LLM calls.
"""

from tpasini.recordatio.extraction.extractor import ConversationTurn, transcript_to_turn


def test_transcript_to_turn_basic_roundtrip() -> None:
    transcript = {
        "messages": [{"role": "user", "content": "ping"}],
        "response_text": "pong",
    }
    turn = transcript_to_turn(transcript)
    assert isinstance(turn, ConversationTurn)
    assert turn.messages == [{"role": "user", "content": "ping"}]
    assert turn.response_text == "pong"


def test_transcript_to_turn_defaults_missing_fields() -> None:
    turn = transcript_to_turn({})
    assert turn.messages == []
    assert turn.response_text == ""


def test_transcript_to_turn_coerces_non_string_response() -> None:
    """The runner stitches ``response_text`` from streamed MESSAGE events
    — if for any reason a non-string slips through (bug, future SDK
    change), we don't want the adapter to raise."""
    turn = transcript_to_turn({"response_text": 42})
    assert turn.response_text == "42"


def test_transcript_to_turn_ignores_non_list_messages() -> None:
    turn = transcript_to_turn({"messages": "not a list"})
    assert turn.messages == []
