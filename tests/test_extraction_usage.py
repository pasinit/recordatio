"""Tests for extraction-cost capture on the extractor side.

Mocks Pontifex to feed deterministic chunk streams into ``extract_knowledge``
and asserts the returned ``ExtractionResult`` carries token usage and model.
"""

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from tpasini.recordatio.extraction import extractor as ext_mod
from tpasini.recordatio.extraction.extractor import (
    ConversationTurn,
    ExtractionSide,
    extract_knowledge,
)
from tpasini.recordatio.models import AttributeType, EntityType, RelationType
from tpasini.recordatio.storage.base import SQLiteStore


@pytest.fixture()
def store(tmp_path: Path) -> SQLiteStore:
    return SQLiteStore(tmp_path / "test.db")


def _text_chunk(text: str) -> dict[str, Any]:
    return {"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}


def _final_with_usage(
    prompt_tokens: int,
    completion_tokens: int,
    *,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> dict[str, Any]:
    return {
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cache_creation": {"ephemeral_5m_input_tokens": cache_creation_tokens},
            "prompt_tokens_details": {"cached_tokens": cache_read_tokens},
        },
    }


class _FakePontifex:
    """Drop-in replacement for ``pontifex.Pontifex`` for tests.

    Each instance yields the chunks it was given; the constructor accepts
    arbitrary keyword arguments to match the real Pontifex signature.
    """

    def __init__(self, *_: Any, **__: Any) -> None:
        self._chunks_by_call: list[list[dict[str, Any]]] = []

    async def stream(self, *_: Any, **__: Any) -> AsyncIterator[dict[str, Any]]:
        chunks = self._chunks_by_call.pop(0) if self._chunks_by_call else [_text_chunk("{}")]
        for chunk in chunks:
            yield chunk


def _empty_pass_chunks() -> list[dict[str, Any]]:
    return [_text_chunk("{}"), _final_with_usage(prompt_tokens=0, completion_tokens=0)]


def _patch_pontifex(
    monkeypatch: pytest.MonkeyPatch,
    chunks: list[dict[str, Any]],
    *,
    ontology_chunks: list[dict[str, Any]] | None = None,
    schema_chunks: list[dict[str, Any]] | None = None,
) -> None:
    """Patch the Pontifex constructor used inside extractor.py to return a fake."""

    def factory(*args: Any, **kwargs: Any) -> _FakePontifex:
        fake = _FakePontifex()
        fake._chunks_by_call = [
            chunks,
            ontology_chunks if ontology_chunks is not None else _empty_pass_chunks(),
            schema_chunks if schema_chunks is not None else _empty_pass_chunks(),
        ]
        return fake

    monkeypatch.setattr(ext_mod, "Pontifex", factory)


def _valid_extraction_payload() -> str:
    return json.dumps(
        {
            "units": [
                {"type": "semantic", "content": "Alice loves Python", "provenance": "extraction"}
            ],
            "entity_mentions": [{"label": "Alice"}],
            "relations": [],
            "topics": [],
        }
    )


async def test_extract_knowledge_captures_upstream_usage(
    store: SQLiteStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _valid_extraction_payload()
    chunks = [
        _text_chunk(payload[: len(payload) // 2]),
        _text_chunk(payload[len(payload) // 2 :]),
        _final_with_usage(
            prompt_tokens=137,
            completion_tokens=42,
            cache_creation_tokens=11,
            cache_read_tokens=12,
        ),
    ]
    _patch_pontifex(monkeypatch, chunks)

    turn = ConversationTurn(
        messages=[{"role": "user", "content": "I love Python"}],
        response_text="Nice — what do you build with it?",
    )
    result = await extract_knowledge(
        turn, api_key="dummy", store=store,
        side=ExtractionSide.OUTPUT,
        agent_id="test-agent",
        model="claude-haiku-4-5-20251001",
    )

    assert result.input_tokens == 137
    assert result.output_tokens == 42
    assert result.cache_creation_tokens == 11
    assert result.cache_read_tokens == 12
    assert result.model == "claude-haiku-4-5-20251001"
    assert result.side is ExtractionSide.OUTPUT
    assert len(result.units) == 1


async def test_extract_knowledge_falls_back_when_no_usage(
    store: SQLiteStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _valid_extraction_payload()
    # No final usage chunk — extractor should estimate from text length.
    chunks = [_text_chunk(payload)]
    _patch_pontifex(monkeypatch, chunks)

    turn = ConversationTurn(
        messages=[{"role": "user", "content": "Some user message"}],
        response_text="Some assistant response",
    )
    result = await extract_knowledge(
        turn, api_key=None, store=store, side=ExtractionSide.OUTPUT, agent_id="test-agent",
    )

    # Estimator yields max(1, len(text)//4); both inputs and output should be > 0.
    assert result.input_tokens > 0
    assert result.output_tokens > 0
    # And less than the characters they were derived from — sanity.
    assert result.output_tokens <= max(1, len(payload) // 4)
    # Model is whatever was passed (default).
    assert result.model


async def test_extract_knowledge_returns_usage_even_when_parse_fails(
    store: SQLiteStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Garbage output — the LLM call still completed, parsing fails locally.
    chunks = [
        _text_chunk("not valid json at all"),
        _final_with_usage(prompt_tokens=10, completion_tokens=5),
    ]
    _patch_pontifex(monkeypatch, chunks)

    turn = ConversationTurn(messages=[{"role": "user", "content": "x"}], response_text="y")
    result = await extract_knowledge(
        turn, api_key=None, store=store,
        side=ExtractionSide.OUTPUT,
        agent_id="test-agent",
        model="claude-haiku-4-5-20251001",
    )

    # Empty extraction but usage and model are populated.
    assert result.units == []
    assert result.entity_mentions == []
    assert result.input_tokens == 10
    assert result.output_tokens == 5
    assert result.model == "claude-haiku-4-5-20251001"


async def test_extract_knowledge_propagates_api_errors(
    store: SQLiteStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _BoomPontifex:
        def __init__(self, *_: Any, **__: Any) -> None: ...

        async def stream(self, *_: Any, **__: Any) -> AsyncIterator[dict[str, Any]]:
            raise RuntimeError("upstream boom")
            yield  # pragma: no cover  -- generator-shape only

    monkeypatch.setattr(ext_mod, "Pontifex", _BoomPontifex)

    turn = ConversationTurn(messages=[{"role": "user", "content": "x"}], response_text="y")
    with pytest.raises(RuntimeError, match="upstream boom"):
        await extract_knowledge(
            turn, api_key=None, store=store, side=ExtractionSide.OUTPUT, agent_id="test-agent",
        )


async def test_extract_knowledge_input_side_renders_only_messages(
    store: SQLiteStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _valid_extraction_payload()
    chunks = [_text_chunk(payload), _final_with_usage(prompt_tokens=20, completion_tokens=5)]

    captured: dict[str, Any] = {}

    class _CapturingPontifex:
        def __init__(self, *_: Any, **__: Any) -> None:
            self._chunks = chunks

        async def stream(self, *_: Any, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
            captured["messages"] = kwargs.get("messages")
            for chunk in self._chunks:
                yield chunk

    monkeypatch.setattr(ext_mod, "Pontifex", _CapturingPontifex)

    turn = ConversationTurn(
        messages=[{"role": "user", "content": "INPUT-MARKER"}],
        response_text="OUTPUT-MARKER",
    )
    result = await extract_knowledge(
        turn, api_key=None, store=store, side=ExtractionSide.INPUT, agent_id="test-agent",
    )

    rendered = captured["messages"][0]["content"]
    assert "INPUT-MARKER" in rendered
    assert "OUTPUT-MARKER" not in rendered
    assert result.side is ExtractionSide.INPUT


async def test_extract_knowledge_runs_narrow_passes_with_ontology_briefing(
    store: SQLiteStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = datetime(2026, 1, 2, tzinfo=UTC)
    store.upsert_entity_type(
        EntityType(name="person", description="Human user or contact", source="seed", created_at=now),
        agent_id="agent-1",
    )
    store.upsert_attribute_type(
        AttributeType(
            name="timezone",
            description="Preferred or current timezone",
            value_type="string",
            source="seed",
            created_at=now,
        ),
        agent_id="agent-1",
    )
    store.add_relation_type(
        RelationType(
            id="rt-lives-in",
            name="lives_in",
            description="Residential location",
            source="seed",
            created_at=now,
        )
    )

    atomic_payload = json.dumps(
        {
            "units": [
                {"type": "semantic", "content": "User lives in Sunnydale."},
                {"type": "semantic", "content": "User lives in Sunnydale."},
            ]
        }
    )
    ontology_payload = json.dumps(
        {
            "entity_mentions": [
                {
                    "id": "entity:user",
                    "label": "User",
                    "type": "person",
                    "aliases": ["Tommaso"],
                    "identifiers": {"email": "tommaso@example.com"},
                    "confidence": 0.9,
                },
                {"id": "entity:sunnydale", "label": "Sunnydale", "type": "place"},
            ],
            "attributes": [
                {
                    "subject": "User",
                    "subject_id": "entity:user",
                    "attribute": "timezone",
                    "value": "Europe/Paris",
                },
                {
                    "subject": "User",
                    "subject_id": "entity:user",
                    "attribute": "timezone",
                    "value": "Europe/Paris",
                },
            ],
            "relations": [
                {
                    "subject": "User",
                    "subject_id": "entity:user",
                    "predicate": "lives_in",
                    "object": "Sunnydale",
                    "object_id": "entity:sunnydale",
                },
                {
                    "subject": "User",
                    "subject_id": "entity:user",
                    "predicate": "lives_in",
                    "object": "Sunnydale",
                    "object_id": "entity:sunnydale",
                },
            ],
        }
    )
    schema_payload = json.dumps(
        {
            "schema_proposals": [
                {
                    "kind": "entity_type",
                    "name": "place",
                    "description": "Geographic location referenced as an entity",
                },
                {
                    "kind": "entity_type",
                    "name": "place",
                    "description": "Geographic location referenced as an entity",
                },
                {
                    "kind": "topic",
                    "name": "personal_profile",
                    "description": "User profile facts",
                },
                {
                    "kind": "topic",
                    "name": "personal_profile",
                    "description": "User profile facts",
                },
            ],
            "topics": [],
        }
    )
    chunks_by_call = [
        [_text_chunk(atomic_payload), _final_with_usage(prompt_tokens=10, completion_tokens=1)],
        [_text_chunk(ontology_payload), _final_with_usage(prompt_tokens=20, completion_tokens=2)],
        [_text_chunk(schema_payload), _final_with_usage(prompt_tokens=30, completion_tokens=3)],
    ]
    captured_systems: list[str] = []

    class _CapturingPontifex:
        def __init__(self, *_: Any, **__: Any) -> None:
            self._chunks_by_call = list(chunks_by_call)

        async def stream(self, *_: Any, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
            captured_systems.append(kwargs["system"])
            for chunk in self._chunks_by_call.pop(0):
                yield chunk

    monkeypatch.setattr(ext_mod, "Pontifex", _CapturingPontifex)

    turn = ConversationTurn(
        messages=[{"role": "user", "content": "I am Tommaso and I live in Sunnydale."}],
        response_text="Noted.",
    )
    result = await extract_knowledge(
        turn,
        api_key=None,
        store=store,
        side=ExtractionSide.INPUT,
        agent_id="agent-1",
    )

    assert [unit.content for unit in result.memory_units] == ["User lives in Sunnydale."]
    assert result.entity_mentions[0].type == "person"
    assert result.entity_mentions[0].aliases == ["Tommaso"]
    assert result.entity_mentions[0].identifiers == {"email": "tommaso@example.com"}
    assert len(result.memory_units) == 1
    assert result.attributes[0].attribute == "timezone"
    assert len(result.attributes) == 1
    assert result.relation_candidates[0].predicate == "lives_in"
    assert len(result.relation_candidates) == 1
    assert [proposal.name for proposal in result.schema_proposals] == ["place", "personal_profile"]
    assert result.schema_proposals[1].kind.value == "topic"
    assert (result.input_tokens, result.output_tokens) == (60, 6)
    assert len(captured_systems) == 3
    assert "Do not extract entity mentions, attributes, relations" in captured_systems[0]
    assert "- person: Human user or contact" in captured_systems[1]
    assert "- timezone (string): Preferred or current timezone" in captured_systems[1]
    assert "- lives_in:" in captured_systems[1]
    assert "Do not return accepted facts" in captured_systems[2]
    assert "schema_proposals" in captured_systems[2]


async def test_extract_knowledge_short_circuits_on_empty_input(
    store: SQLiteStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = {"n": 0}

    class _ShouldNotRun:
        def __init__(self, *_: Any, **__: Any) -> None: ...

        async def stream(self, *_: Any, **__: Any) -> AsyncIterator[dict[str, Any]]:
            called["n"] += 1
            yield {}

    monkeypatch.setattr(ext_mod, "Pontifex", _ShouldNotRun)

    # No user-side messages → input side has nothing to extract.
    turn = ConversationTurn(messages=[], response_text="non-empty response")
    result = await extract_knowledge(
        turn, api_key=None, store=store, side=ExtractionSide.INPUT, agent_id="test-agent",
    )

    assert called["n"] == 0
    assert result.units == []
    assert result.input_tokens == 0
    assert result.side is ExtractionSide.INPUT


async def test_codex_extraction_never_constructs_api_gateway(store, monkeypatch):
    from pontifex.codex_completion import CodexCompletion

    def forbidden(**kwargs):
        raise AssertionError("API gateway must not be constructed")

    calls = []

    async def stream(self, **kwargs):
        calls.append(kwargs)
        yield _text_chunk(_valid_extraction_payload())
        yield _final_with_usage(prompt_tokens=5, completion_tokens=2)

    monkeypatch.setattr(ext_mod, "Pontifex", forbidden)
    monkeypatch.setattr(CodexCompletion, "stream", stream)
    result = await extract_knowledge(
        ConversationTurn(messages=[{"role": "user", "content": "I love Python"}], response_text="Nice"),
        api_key="must-not-be-used", store=store, side=ExtractionSide.OUTPUT,
        agent_id="test", model="test-codex", backend="codex",
    )
    assert len(calls) == 3 and len(result.units) == 1
