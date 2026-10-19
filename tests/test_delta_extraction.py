"""Tests for delta extraction via x-recordatio-extract-from hint."""

from tpasini.recordatio.proxy.server import _extraction_messages


def test_no_hint_returns_full_messages() -> None:
    messages = [
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "reply 1"},
        {"role": "user", "content": "turn 2"},
    ]
    assert _extraction_messages(messages, None) is messages


def test_hint_returns_only_last_user_message() -> None:
    messages = [
        {"role": "user", "content": "old turn"},
        {"role": "assistant", "content": "old reply"},
        {"role": "user", "content": "new turn"},
    ]
    hint = {"last_user_message": "new turn"}
    result = _extraction_messages(messages, hint)
    assert result == [{"role": "user", "content": "new turn"}]


def test_hint_missing_last_user_message_falls_back() -> None:
    messages = [{"role": "user", "content": "turn 1"}]
    # hint present but key absent — fall back to full history
    assert _extraction_messages(messages, {}) is messages


def test_hint_non_string_last_user_message_falls_back() -> None:
    messages = [{"role": "user", "content": "turn 1"}]
    assert _extraction_messages(messages, {"last_user_message": 42}) is messages


def test_empty_hint_string_falls_back() -> None:
    messages = [{"role": "user", "content": "turn 1"}]
    # empty string is falsy — fall back
    assert _extraction_messages(messages, {"last_user_message": ""}) is messages
