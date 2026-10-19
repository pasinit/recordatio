"""Tests for proxy/server.py — extraction-cost piggybacking + prompt assembly.

Two parallel feature areas land in this proxy:

1. Per-agent extraction-cost buffering / drain. A `_unreported_extractions`
   buffer accumulates costs from background extraction calls, and is drained
   onto the *next* chat response for the same agent (pop-on-read). Tests poke
   `_drain_extraction_usage` and the dict directly, plus exercise the
   FastAPI endpoint via TestClient (with `pontifex.Pontifex` mocked).

2. Prompt assembly helpers — `_build_system` combines the stable system prompt
   with retrieved briefing text, and `_count_messages_tokens` estimates input
   tokens for the assembled request.

The stable-before-briefing ordering test is the most regression-prone bit:
the previous code prepended the briefing in front of the stable system, which
made retrieval less relevant and was hard to reason about in tests.
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from tpasini.recordatio.entity import Entity
from tpasini.recordatio.extraction.contracts import (
    AtomicMemoryUnit,
    AttributeExtraction,
    EvidenceReference,
    RelationExtraction,
    SchemaProposal,
    SchemaProposalKind,
)
from tpasini.recordatio.extraction.contracts import (
    EntityMention as ExtractionEntityMention,
)
from tpasini.recordatio.extraction.extractor import ExtractionResult, ExtractionSide
from tpasini.recordatio.identity import get_principal, mirror_entity_id, user_scope
from tpasini.recordatio.models import (
    AttributeType,
    EntityAlias,
    EntityAttribute,
    EntityIdentifier,
    EntityMention,
    KnowledgeType,
    KnowledgeUnit,
    Relation,
    RelationType,
    RenderedRelation,
    Topic,
)
from tpasini.recordatio.proxy.briefing_log import BriefingLog
from tpasini.recordatio.proxy.cost_log import CostLog
from tpasini.recordatio.proxy.retrieval import RetrievalResult
from tpasini.recordatio.storage.base import SQLiteStore


@pytest.fixture()
def server_mod(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Return the (cached) ``server`` module with an empty extraction buffer.

    The module imports a heavy FastAPI app + Pontifex SDK, so we don't
    re-import per test. We just clear the module-level extraction buffer at
    the start of each test — that's the only piece of mutable state the
    tests in this file actually mutate.
    """
    monkeypatch.setenv("RECORDATIO_EXTRACTION_MODEL", "anthropic/claude-haiku-4-5-20251001")
    monkeypatch.setenv("AGENTMEM_DB", str(tmp_path / "memories.db"))

    from tpasini.recordatio.proxy import server as server_mod

    server_mod._unreported_extractions.clear()
    return server_mod


# ---------------------------------------------------------------------------
# Direct buffer-helper tests
# ---------------------------------------------------------------------------


def test_drain_returns_none_when_buffer_empty(server_mod) -> None:
    result = asyncio.run(server_mod._drain_extraction_usage("alice"))
    assert result is None


def test_drain_returns_none_when_agent_id_none(server_mod) -> None:
    # Pre-load the buffer to make sure it's the agent_id check, not emptiness,
    # that returns None.
    server_mod._unreported_extractions["alice"].append(
        server_mod._ExtractionUsage(model="m", input_tokens=1, output_tokens=1, cost_usd=0.001)
    )
    result = asyncio.run(server_mod._drain_extraction_usage(None))
    assert result is None
    # Buffer untouched.
    assert len(server_mod._unreported_extractions["alice"]) == 1


def test_drain_single_event(server_mod) -> None:
    server_mod._unreported_extractions["alice"].append(
        server_mod._ExtractionUsage(
            model="claude-haiku-4-5-20251001",
            input_tokens=800,
            output_tokens=200,
            cost_usd=0.0007,
        )
    )
    result = asyncio.run(server_mod._drain_extraction_usage("alice"))
    assert result == {
        "cost_usd_extraction": 0.0007,
        "extraction_input_tokens": 800,
        "extraction_output_tokens": 200,
        "extraction_calls": 1,
        "extraction_model": "claude-haiku-4-5-20251001",
    }


def test_drain_sums_multiple_events(server_mod) -> None:
    server_mod._unreported_extractions["alice"].extend([
        server_mod._ExtractionUsage(model="claude-haiku-4-5-20251001",
                                      input_tokens=100, output_tokens=20, cost_usd=0.001),
        server_mod._ExtractionUsage(model="claude-haiku-4-5-20251001",
                                      input_tokens=200, output_tokens=40, cost_usd=0.002),
    ])
    result = asyncio.run(server_mod._drain_extraction_usage("alice"))
    assert result["extraction_calls"] == 2
    assert result["extraction_input_tokens"] == 300
    assert result["extraction_output_tokens"] == 60
    assert result["cost_usd_extraction"] == pytest.approx(0.003)
    assert result["extraction_model"] == "claude-haiku-4-5-20251001"


def test_drain_picks_most_recent_model_when_mixed(server_mod) -> None:
    server_mod._unreported_extractions["alice"].extend([
        server_mod._ExtractionUsage(model="gpt-4o-mini",
                                      input_tokens=10, output_tokens=2, cost_usd=0.001),
        server_mod._ExtractionUsage(model="claude-haiku-4-5-20251001",
                                      input_tokens=20, output_tokens=4, cost_usd=0.002),
    ])
    result = asyncio.run(server_mod._drain_extraction_usage("alice"))
    # Most recent (last appended) wins.
    assert result["extraction_model"] == "claude-haiku-4-5-20251001"
    assert result["extraction_calls"] == 2


def test_drain_pops_on_read(server_mod) -> None:
    server_mod._unreported_extractions["alice"].append(
        server_mod._ExtractionUsage(model="m", input_tokens=10, output_tokens=2, cost_usd=0.001)
    )
    first = asyncio.run(server_mod._drain_extraction_usage("alice"))
    assert first is not None
    # Second drain returns nothing — pop-on-read.
    second = asyncio.run(server_mod._drain_extraction_usage("alice"))
    assert second is None


def test_drain_isolates_agents(server_mod) -> None:
    server_mod._unreported_extractions["alice"].append(
        server_mod._ExtractionUsage(model="m", input_tokens=10, output_tokens=2, cost_usd=0.001)
    )
    server_mod._unreported_extractions["bob"].append(
        server_mod._ExtractionUsage(model="m", input_tokens=99, output_tokens=99, cost_usd=99.0)
    )

    alice_result = asyncio.run(server_mod._drain_extraction_usage("alice"))
    assert alice_result["extraction_input_tokens"] == 10
    # Bob's buffer untouched.
    assert len(server_mod._unreported_extractions["bob"]) == 1

    bob_result = asyncio.run(server_mod._drain_extraction_usage("bob"))
    assert bob_result["extraction_input_tokens"] == 99


def test_concurrent_appends_dont_cross_contaminate(server_mod) -> None:
    """Append from N coroutines for two different agents; verify the buffers
    end up with the right counts and no cross-contamination."""

    async def runner() -> None:
        async def append_for(agent: str, count: int) -> None:
            for i in range(count):
                async with server_mod._unreported_extractions_lock:
                    server_mod._unreported_extractions[agent].append(
                        server_mod._ExtractionUsage(
                            model="m", input_tokens=i, output_tokens=i, cost_usd=0.001 * i,
                        )
                    )
                # Yield to other coroutines.
                await asyncio.sleep(0)

        await asyncio.gather(append_for("alice", 20), append_for("bob", 20))

    asyncio.run(runner())
    assert len(server_mod._unreported_extractions["alice"]) == 20
    assert len(server_mod._unreported_extractions["bob"]) == 20


# ---------------------------------------------------------------------------
# Chat-completions integration tests (mocked Pontifex)
# ---------------------------------------------------------------------------


class _FakePontifex:
    """Stand-in for ``pontifex.Pontifex`` that yields preset chunks."""

    _next_chunks: list[dict] = []

    def __init__(self, *_: Any, model: str = "test-model", **__: Any) -> None:
        self.model = model
        self._chunks = list(self._next_chunks)

    async def stream(self, *_: Any, **__: Any) -> AsyncIterator[dict]:
        for chunk in self._chunks:
            yield chunk


def _patch_pontifex(monkeypatch: pytest.MonkeyPatch, server_mod, chunks: list[dict]) -> None:
    _FakePontifex._next_chunks = chunks
    monkeypatch.setattr(server_mod, "Pontifex", _FakePontifex)


def _suppress_extraction(monkeypatch: pytest.MonkeyPatch, server_mod) -> None:
    """Replace ``_extract_and_store`` with a no-op so background extraction
    doesn't kick in during chat tests — we'll inject buffer entries directly."""

    async def noop(*_: Any, **__: Any) -> None:  # pragma: no cover  (noop)
        return None

    monkeypatch.setattr(server_mod, "_extract_and_store", noop)


def _suppress_briefing(monkeypatch: pytest.MonkeyPatch, server_mod) -> None:
    # `backend` and `call_id` are irrelevant to these buffer tests, so
    # accept-and-ignore them.
    async def passthrough(
        existing_system,
        _query,
        _agent_id,
        *,
        backend=None,
        call_id=None,
        source_agent_id=None,
    ):
        return server_mod.prompting.SystemBuildResult(system=existing_system, briefing_data=None)

    monkeypatch.setattr(server_mod, "_build_system", passthrough)


def _basic_chunks(text: str = "hi there") -> list[dict]:
    return [
        {"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
    ]


def test_non_streaming_response_omits_extraction_fields_when_buffer_empty(
    server_mod, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_pontifex(monkeypatch, server_mod, _basic_chunks())
    _suppress_extraction(monkeypatch, server_mod)
    _suppress_briefing(monkeypatch, server_mod)

    client = TestClient(server_mod.app)
    resp = client.post(
        "/v1/chat/completions",
        headers={"x-agent-id": "alice"},
        json={"model": "openai/gpt-4o-mini",
              "messages": [{"role": "user", "content": "ping"}]},
    )
    assert resp.status_code == 200
    usage = resp.json()["usage"]
    assert "cost_usd_extraction" not in usage
    assert "extraction_calls" not in usage
    assert "extraction_model" not in usage
    assert "extraction_input_tokens" not in usage
    assert "extraction_output_tokens" not in usage


def test_non_streaming_response_drains_buffer_for_agent(
    server_mod, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_pontifex(monkeypatch, server_mod, _basic_chunks())
    _suppress_extraction(monkeypatch, server_mod)
    _suppress_briefing(monkeypatch, server_mod)

    # Pre-load alice's buffer with one extraction event.
    server_mod._unreported_extractions["alice"].append(
        server_mod._ExtractionUsage(
            model="claude-haiku-4-5-20251001",
            input_tokens=800, output_tokens=200, cost_usd=0.0007,
        )
    )

    client = TestClient(server_mod.app)
    resp = client.post(
        "/v1/chat/completions",
        headers={"x-agent-id": "alice"},
        json={"model": "openai/gpt-4o-mini",
              "messages": [{"role": "user", "content": "ping"}]},
    )
    assert resp.status_code == 200
    usage = resp.json()["usage"]
    assert usage["cost_usd_extraction"] == pytest.approx(0.0007)
    assert usage["extraction_input_tokens"] == 800
    assert usage["extraction_output_tokens"] == 200
    assert usage["extraction_calls"] == 1
    assert usage["extraction_model"] == "claude-haiku-4-5-20251001"

    # A *second* call, no new extractions added — buffer is empty after pop.
    resp2 = client.post(
        "/v1/chat/completions",
        headers={"x-agent-id": "alice"},
        json={"model": "openai/gpt-4o-mini",
              "messages": [{"role": "user", "content": "ping"}]},
    )
    usage2 = resp2.json()["usage"]
    assert "cost_usd_extraction" not in usage2
    assert "extraction_calls" not in usage2


def test_chat_for_agent_a_does_not_drain_agent_b_buffer(
    server_mod, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_pontifex(monkeypatch, server_mod, _basic_chunks())
    _suppress_extraction(monkeypatch, server_mod)
    _suppress_briefing(monkeypatch, server_mod)

    server_mod._unreported_extractions["bob"].append(
        server_mod._ExtractionUsage(
            model="claude-haiku-4-5-20251001",
            input_tokens=500, output_tokens=100, cost_usd=0.0005,
        )
    )

    client = TestClient(server_mod.app)
    # Alice fires a chat — bob's buffer must remain.
    resp = client.post(
        "/v1/chat/completions",
        headers={"x-agent-id": "alice"},
        json={"model": "openai/gpt-4o-mini",
              "messages": [{"role": "user", "content": "ping"}]},
    )
    assert resp.status_code == 200
    assert "cost_usd_extraction" not in resp.json()["usage"]
    # Bob's buffer untouched.
    assert len(server_mod._unreported_extractions["bob"]) == 1


def test_chat_without_agent_id_skips_drain(
    server_mod, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_pontifex(monkeypatch, server_mod, _basic_chunks())
    _suppress_extraction(monkeypatch, server_mod)
    _suppress_briefing(monkeypatch, server_mod)

    # Agent-less request shouldn't blow up and shouldn't add extraction fields.
    client = TestClient(server_mod.app)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "openai/gpt-4o-mini",
              "messages": [{"role": "user", "content": "ping"}]},
    )
    assert resp.status_code == 200
    usage = resp.json()["usage"]
    assert "cost_usd_extraction" not in usage


def test_streaming_response_drains_buffer_for_agent(
    server_mod, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The streaming SSE path also has to drain. The agentmem.usage event
    should carry the extraction fields inside its ``usage`` block."""
    _patch_pontifex(monkeypatch, server_mod, _basic_chunks())
    _suppress_extraction(monkeypatch, server_mod)
    _suppress_briefing(monkeypatch, server_mod)

    server_mod._unreported_extractions["alice"].extend([
        server_mod._ExtractionUsage(
            model="claude-haiku-4-5-20251001",
            input_tokens=800, output_tokens=200, cost_usd=0.0007,
        ),
        server_mod._ExtractionUsage(
            model="claude-haiku-4-5-20251001",
            input_tokens=400, output_tokens=100, cost_usd=0.0003,
        ),
    ])

    client = TestClient(server_mod.app)
    with client.stream(
        "POST", "/v1/chat/completions",
        headers={"x-agent-id": "alice"},
        json={"model": "openai/gpt-4o-mini", "stream": True,
              "messages": [{"role": "user", "content": "ping"}]},
    ) as resp:
        body = b"".join(resp.iter_bytes()).decode()

    # Find the agentmem.usage line and parse it.
    usage_payload: dict | None = None
    in_usage_event = False
    for line in body.splitlines():
        if line.startswith("event: agentmem.usage"):
            in_usage_event = True
            continue
        if in_usage_event and line.startswith("data: "):
            usage_payload = json.loads(line[len("data: "):])
            break

    assert usage_payload is not None
    inner = usage_payload["usage"]
    assert inner["extraction_calls"] == 2
    assert inner["cost_usd_extraction"] == pytest.approx(0.001)
    assert inner["extraction_input_tokens"] == 1200
    assert inner["extraction_output_tokens"] == 300
    assert inner["extraction_model"] == "claude-haiku-4-5-20251001"

    # And after the drain, the buffer is empty.
    assert "alice" not in server_mod._unreported_extractions or \
           server_mod._unreported_extractions["alice"] == []


# ---------------------------------------------------------------------------
# Cache-aware cost calculation (#35)
# ---------------------------------------------------------------------------


def _cache_chunks_anthropic(text: str = "hi") -> list[dict]:
    """Final usage chunk uses Anthropic-shaped fields.

    Pontifex's Anthropic backend will emit OAI-shaped envelopes around an
    Anthropic-shaped usage payload after the bump from #34's Pontifex PR.
    The parser tolerates either shape — we mix prompt_tokens with the
    Anthropic-only ``cache_creation_input_tokens`` / ``cache_read_input_tokens``
    fields here to match what we expect post-bump.
    """
    return [
        {"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
         "usage": {
             "prompt_tokens": 1000,
             "completion_tokens": 100,
             "total_tokens": 1100,
             "cache_creation_input_tokens": 200,
             "cache_read_input_tokens": 600,
         }},
    ]


def _cache_chunks_openai(text: str = "hi") -> list[dict]:
    """Final usage chunk uses OpenAI-shaped fields with nested cache details."""
    return [
        {"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
         "usage": {
             "prompt_tokens": 1000,
             "completion_tokens": 100,
             "total_tokens": 1100,
             "prompt_tokens_details": {"cached_tokens": 400},
         }},
    ]


def test_non_streaming_anthropic_cache_usage_reflects_discount(
    server_mod, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anthropic-shaped usage with cache fields → cost reflects 1.25x write
    + 0.10x read multipliers and the response surfaces cache token counts."""
    _patch_pontifex(monkeypatch, server_mod, _cache_chunks_anthropic())
    _suppress_extraction(monkeypatch, server_mod)
    _suppress_briefing(monkeypatch, server_mod)

    client = TestClient(server_mod.app)
    resp = client.post(
        "/v1/chat/completions",
        headers={"x-agent-id": "alice"},
        json={"model": "anthropic/claude-opus-4-6",
              "messages": [{"role": "user", "content": "ping"}]},
    )
    assert resp.status_code == 200
    usage = resp.json()["usage"]
    # uncached = 1000 - 200 - 600 = 200
    # 200*15 + 200*15*1.25 + 600*15*0.10 + 100*75 = 15150 / 1e6 = 0.01515
    assert usage["cost_usd"] == pytest.approx(0.01515, abs=1e-6)
    assert usage["cache_creation_tokens"] == 200
    assert usage["cache_read_tokens"] == 600
    assert usage["prompt_tokens"] == 1000
    assert usage["completion_tokens"] == 100


def test_non_streaming_openai_cache_usage_reflects_discount(
    server_mod, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OpenAI-shaped usage with nested cached_tokens → 0.50x cache_read."""
    _patch_pontifex(monkeypatch, server_mod, _cache_chunks_openai())
    _suppress_extraction(monkeypatch, server_mod)
    _suppress_briefing(monkeypatch, server_mod)

    client = TestClient(server_mod.app)
    resp = client.post(
        "/v1/chat/completions",
        headers={"x-agent-id": "alice"},
        json={"model": "openai/gpt-4o",
              "messages": [{"role": "user", "content": "ping"}]},
    )
    assert resp.status_code == 200
    usage = resp.json()["usage"]
    # uncached = 1000 - 0 - 400 = 600
    # 600*2.50 + 400*2.50*0.50 + 100*10 = 3000 / 1e6 = 0.003
    assert usage["cost_usd"] == pytest.approx(0.003, abs=1e-6)
    assert "cache_creation_tokens" not in usage  # 0 → omitted
    assert usage["cache_read_tokens"] == 400


def test_streaming_anthropic_cache_usage_reflects_discount(
    server_mod, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Streaming SSE path emits cache fields + discounted cost in usage block."""
    _patch_pontifex(monkeypatch, server_mod, _cache_chunks_anthropic())
    _suppress_extraction(monkeypatch, server_mod)
    _suppress_briefing(monkeypatch, server_mod)

    client = TestClient(server_mod.app)
    with client.stream(
        "POST", "/v1/chat/completions",
        headers={"x-agent-id": "alice"},
        json={"model": "anthropic/claude-opus-4-6", "stream": True,
              "messages": [{"role": "user", "content": "ping"}]},
    ) as resp:
        body = b"".join(resp.iter_bytes()).decode()

    usage_payload: dict | None = None
    in_usage_event = False
    for line in body.splitlines():
        if line.startswith("event: agentmem.usage"):
            in_usage_event = True
            continue
        if in_usage_event and line.startswith("data: "):
            usage_payload = json.loads(line[len("data: "):])
            break

    assert usage_payload is not None
    inner = usage_payload["usage"]
    assert inner["cache_creation_tokens"] == 200
    assert inner["cache_read_tokens"] == 600
    assert inner["cost_usd"] == pytest.approx(0.01515, abs=1e-6)


def test_parse_upstream_usage_handles_all_shapes(server_mod) -> None:
    """Direct unit test for the helper — covers None, empty, Anthropic,
    OpenAI flat, and OpenAI nested ``prompt_tokens_details`` shapes."""
    parse = server_mod._parse_upstream_usage

    assert parse(None) == (0, 0, 0, 0)
    assert parse({}) == (0, 0, 0, 0)

    # Anthropic-native (input_tokens/output_tokens + flat cache fields)
    assert parse({
        "input_tokens": 100, "output_tokens": 50,
        "cache_creation_input_tokens": 20,
        "cache_read_input_tokens": 30,
    }) == (100, 50, 20, 30)

    # OpenAI flat (prompt_tokens/completion_tokens + nested cached_tokens)
    assert parse({
        "prompt_tokens": 1000, "completion_tokens": 100,
        "prompt_tokens_details": {"cached_tokens": 400},
    }) == (1000, 100, 0, 400)

    # Nested anthropic cache_creation
    assert parse({
        "prompt_tokens": 100, "completion_tokens": 50,
        "cache_creation": {"ephemeral_5m_input_tokens": 25},
    }) == (100, 50, 25, 0)


# ---------------------------------------------------------------------------
# _build_system — end-to-end with retrieval mocked
# ---------------------------------------------------------------------------


async def _fake_retrieve_with_briefing(
    query: str,
    agent_id: str,
    source_agent_id: str | None = None,
):
    """Simulate retrieval that returns a single semantic unit."""
    now = datetime.now(UTC)
    unit = KnowledgeUnit(
        id="u1",
        type=KnowledgeType.SEMANTIC,
        content="The user prefers Python",
        asserted_at=now,
        valid_from=now,
        provenance="test",
    )
    return RetrievalResult(
        units=[unit],
        entities=[],
        rendered_relations=[],
        raw_relations=[],
        topic_hint=None,
    )


async def _fake_retrieve_empty(
    query: str,
    agent_id: str,
    source_agent_id: str | None = None,
):
    """Simulate retrieval that returns nothing."""
    return RetrievalResult(
        units=[],
        entities=[],
        rendered_relations=[],
        raw_relations=[],
        topic_hint=None,
    )


@pytest.mark.asyncio
async def test_build_system_anthropic_returns_string_with_briefing(server_mod) -> None:
    """Recordatio returns provider-neutral system text; Pontifex owns provider adaptation."""
    big_system = "STABLE_SYSTEM"
    with patch.object(server_mod, "_retrieve", _fake_retrieve_with_briefing):
        result = await server_mod._build_system(
            big_system, "what do I prefer?", "alice", backend="anthropic"
        )

    assert isinstance(result.system, str)
    assert result.system.startswith(big_system)
    assert "Python" in result.system

    # briefing_data shape unchanged — UI clients depend on this.
    assert result.briefing_data is not None
    assert "units" in result.briefing_data
    assert "entities" in result.briefing_data
    assert "relations" in result.briefing_data
    assert "topic_names" in result.briefing_data
    assert result.briefing_data["units"] == [{"type": "semantic", "content": "The user prefers Python"}]


@pytest.mark.asyncio
async def test_build_system_anthropic_briefing_no_existing_system(server_mod) -> None:
    """Anthropic + briefing only (no stable): provider-neutral system text."""
    with patch.object(server_mod, "_retrieve", _fake_retrieve_with_briefing):
        result = await server_mod._build_system(
            None, "query", "alice", backend="anthropic"
        )

    assert isinstance(result.system, str)
    assert "Python" in result.system
    assert result.briefing_data is not None


@pytest.mark.asyncio
async def test_build_system_anthropic_no_briefing_existing_system(server_mod) -> None:
    """Anthropic + no briefing + existing system returns the original string."""
    big_system = "lorem ipsum dolor sit amet " * 250
    with patch.object(server_mod, "_retrieve", _fake_retrieve_empty):
        result = await server_mod._build_system(
            big_system, "query", "alice", backend="anthropic"
        )

    assert result.system == big_system
    assert result.briefing_data is None


@pytest.mark.asyncio
async def test_build_system_logs_budgeted_counts_and_chars(
    server_mod, caplog: pytest.LogCaptureFixture,
) -> None:
    """One compact line reports the BUDGETED profile/recall counts + chars so
    prod logs show the cap working — not one line per entity/relation."""
    now = datetime.now(UTC)
    unit = KnowledgeUnit(
        id="u1", type=KnowledgeType.SEMANTIC, content="The user prefers Python",
        asserted_at=now, valid_from=now, provenance="test",
    )
    entities = [
        Entity(label="Tommaso", asserted_at=now),
        Entity(label="Springfield", asserted_at=now),
        Entity(label="Globex", asserted_at=now),
    ]
    relations = [
        RenderedRelation(subject="Tommaso", predicate="lives_in", object="Springfield"),
        RenderedRelation(subject="Tommaso", predicate="works_at", object="Globex"),
    ]

    async def fake_retrieve(query, agent_id, source_agent_id=None):
        return RetrievalResult(
            units=[unit], entities=entities,
            rendered_relations=relations, raw_relations=[],
            topic_hint=None,
        )

    with (
        patch.object(server_mod, "_retrieve", fake_retrieve),
        caplog.at_level(logging.INFO, logger="tpasini.recordatio.proxy.server"),
    ):
        result = await server_mod._build_system(
            "STABLE", "where does the user work?", "alice", backend="anthropic"
        )

    assert result.briefing_data is not None
    lines = [r.getMessage() for r in caplog.records if "injected briefing" in r.getMessage()]
    assert len(lines) == 1
    line = lines[0]
    assert "profile:" in line
    # Counts are selected/candidate.
    assert "recall: units=1/1 entities=3/3 relations=0/0" in line
    assert "total_chars=" in line


@pytest.mark.asyncio
async def test_build_system_non_anthropic_returns_string(server_mod) -> None:
    """Non-Anthropic backends still get a plain string — no cache_control.

    Order is ``stable + "\\n\\n" + briefing`` (stable first, matching the
    new Anthropic ordering — the old briefing-first ordering was always
    wrong).
    """
    big_system = "STABLE_SYSTEM"
    with patch.object(server_mod, "_retrieve", _fake_retrieve_with_briefing):
        result = await server_mod._build_system(
            big_system, "query", "alice", backend="openai"
        )

    assert isinstance(result.system, str)
    # Stable first.
    assert result.system.startswith(big_system)
    # Briefing somewhere after.
    assert "Python" in result.system
    # And specifically: stable comes BEFORE briefing.
    stable_idx = result.system.index(big_system)
    briefing_idx = result.system.index("Python")
    assert stable_idx < briefing_idx
    assert result.briefing_data is not None


@pytest.mark.asyncio
async def test_build_system_no_briefing_no_existing_returns_none(server_mod) -> None:
    """Original no-op behaviour preserved: no system, no briefing → None."""
    with patch.object(server_mod, "_retrieve", _fake_retrieve_empty):
        result = await server_mod._build_system(
            None, "query", "alice", backend="openai"
        )

    assert result.system is None
    assert result.briefing_data is None


@pytest.mark.asyncio
async def test_build_system_openai_no_briefing_returns_existing(server_mod) -> None:
    """OpenAI backend, no briefing produced → return the original string."""
    with patch.object(server_mod, "_retrieve", _fake_retrieve_empty):
        result = await server_mod._build_system(
            "old prompt", "query", "alice", backend="openai"
        )

    assert result.system == "old prompt"
    assert result.briefing_data is None


# ---------------------------------------------------------------------------
# _count_messages_tokens — accepts both string and list-of-blocks
# ---------------------------------------------------------------------------


def test_count_messages_tokens_accepts_string_system(server_mod) -> None:
    n = server_mod._count_messages_tokens(
        [{"role": "user", "content": "hi"}],
        "you are a helpful assistant",
    )
    assert n > 0


def test_count_messages_tokens_accepts_list_system(server_mod) -> None:
    """Anthropic-format system blocks are walked and text is counted."""
    blocks = [
        {"type": "text", "text": "you are a helpful assistant", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "[MEMORY] briefing"},
    ]
    n_list = server_mod._count_messages_tokens(
        [{"role": "user", "content": "hi"}],
        blocks,
    )
    n_str = server_mod._count_messages_tokens(
        [{"role": "user", "content": "hi"}],
        "you are a helpful assistant\n[MEMORY] briefing",
    )
    # Both should produce roughly the same count (off-by-one on join is fine).
    assert abs(n_list - n_str) <= 2


def test_count_messages_tokens_none_system(server_mod) -> None:
    n = server_mod._count_messages_tokens(
        [{"role": "user", "content": "hi"}], None,
    )
    assert n > 0


# ---------------------------------------------------------------------------
# Cost-events persistence (#36)
# ---------------------------------------------------------------------------


@pytest.fixture()
def fresh_cost_log(server_mod, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Replace ``server_mod.cost_log`` with a fresh CostLog on tmp_path.

    Server module is cached across tests; the persistent global cost_log
    would otherwise carry rows between tests. Each test that asserts
    rows uses this fixture for an isolated DB.
    """
    log = CostLog(tmp_path / "cost_isolated.db")
    monkeypatch.setattr(server_mod, "cost_log", log)
    return log


def test_chat_records_one_cost_event_with_kind_chat(
    server_mod, fresh_cost_log, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_pontifex(monkeypatch, server_mod, _cache_chunks_anthropic())
    _suppress_extraction(monkeypatch, server_mod)
    _suppress_briefing(monkeypatch, server_mod)

    client = TestClient(server_mod.app)
    resp = client.post(
        "/v1/chat/completions",
        headers={"x-agent-id": "alice"},
        json={"model": "anthropic/claude-opus-4-6",
              "messages": [{"role": "user", "content": "ping"}]},
    )
    assert resp.status_code == 200

    events = fresh_cost_log.query_events()
    assert len(events) == 1
    ev = events[0]
    assert ev["kind"] == "chat"
    assert ev["model"] == "claude-opus-4-6"
    assert ev["agent_id"] == "alice"
    # Cache fields populated from the Anthropic-shaped usage chunk.
    assert ev["cache_creation_tokens"] == 200
    assert ev["cache_read_tokens"] == 600
    assert ev["input_tokens"] == 1000
    assert ev["output_tokens"] == 100
    assert ev["request_id"] is not None
    assert ev["request_id"].startswith("recordatio-")
    assert ev["ok"] is True
    assert ev["provider"] == "anthropic"


def test_chat_records_event_streaming_path(
    server_mod, fresh_cost_log, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_pontifex(monkeypatch, server_mod, _cache_chunks_anthropic())
    _suppress_extraction(monkeypatch, server_mod)
    _suppress_briefing(monkeypatch, server_mod)

    client = TestClient(server_mod.app)
    with client.stream(
        "POST", "/v1/chat/completions",
        headers={"x-agent-id": "alice"},
        json={"model": "anthropic/claude-opus-4-6", "stream": True,
              "messages": [{"role": "user", "content": "ping"}]},
    ) as resp:
        # Drain the body so the generator runs to completion (the cost-log
        # write happens before [DONE] is yielded but after agentmem.usage).
        b"".join(resp.iter_bytes())

    events = fresh_cost_log.query_events()
    assert len(events) == 1
    assert events[0]["kind"] == "chat"
    assert events[0]["agent_id"] == "alice"
    assert events[0]["cache_creation_tokens"] == 200
    assert events[0]["cache_read_tokens"] == 600


def test_extraction_records_kind_extraction_with_same_request_id(
    server_mod, fresh_cost_log, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The chat row and the extraction row from the same turn share a
    request_id. ``kind`` distinguishes them."""
    _patch_pontifex(monkeypatch, server_mod, _basic_chunks())
    _suppress_briefing(monkeypatch, server_mod)

    # Stub extract_knowledge to return a non-empty result without hitting any LLM.
    async def fake_extract_knowledge(*_: Any, **__: Any) -> ExtractionResult:
        return ExtractionResult(
            input_tokens=400, output_tokens=80,
            cache_creation_tokens=0, cache_read_tokens=0,
            model="claude-haiku-4-5-20251001",
        )

    monkeypatch.setattr(server_mod, "extract_knowledge", fake_extract_knowledge)
    # Skip the user-entity bootstrap to keep the SQLite store untouched.
    monkeypatch.setattr(server_mod, "_ensure_user_entity", lambda _: None)

    client = TestClient(server_mod.app)
    resp = client.post(
        "/v1/chat/completions",
        headers={"x-agent-id": "alice"},
        json={"model": "openai/gpt-4o-mini",
              "messages": [{"role": "user", "content": "ping"}]},
    )
    assert resp.status_code == 200

    # Background extraction fires after the response. Drain pending tasks.
    async def _drain() -> None:
        await asyncio.gather(*server_mod._background_tasks, return_exceptions=True)

    asyncio.run(_drain())

    events = fresh_cost_log.query_events()
    chat_rows = [e for e in events if e["kind"] == "chat"]
    extraction_rows = [e for e in events if e["kind"] == "extraction"]

    assert len(chat_rows) == 1
    # Two extraction rows: one for input side, one for output side.
    assert len(extraction_rows) == 2

    chat_row = chat_rows[0]
    assert chat_row["request_id"] is not None
    for row in extraction_rows:
        assert row["request_id"] == chat_row["request_id"]
        assert row["model"] == "claude-haiku-4-5-20251001"
        assert row["ok"] is True
        assert row["skipped"] is False
        assert row["input_tokens"] == 400
        assert row["output_tokens"] == 80

    sides = {row["extraction_side"] for row in extraction_rows}
    assert sides == {ExtractionSide.INPUT, ExtractionSide.OUTPUT}


def test_extraction_error_records_ok_false(
    server_mod, fresh_cost_log, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_pontifex(monkeypatch, server_mod, _basic_chunks())
    _suppress_briefing(monkeypatch, server_mod)

    async def boom(*_: Any, **__: Any) -> None:
        raise RuntimeError("upstream extraction blew up")

    monkeypatch.setattr(server_mod, "extract_knowledge", boom)
    monkeypatch.setattr(server_mod, "_ensure_user_entity", lambda _: None)

    client = TestClient(server_mod.app)
    resp = client.post(
        "/v1/chat/completions",
        headers={"x-agent-id": "alice"},
        json={"model": "openai/gpt-4o-mini",
              "messages": [{"role": "user", "content": "ping"}]},
    )
    assert resp.status_code == 200

    async def _drain() -> None:
        await asyncio.gather(*server_mod._background_tasks, return_exceptions=True)

    asyncio.run(_drain())

    events = fresh_cost_log.query_events()
    extraction_rows = [e for e in events if e["kind"] == "extraction"]
    # Both sides errored — one ok=False row per side.
    assert len(extraction_rows) == 2
    assert all(row["ok"] is False for row in extraction_rows)
    sides = {row["extraction_side"] for row in extraction_rows}
    assert sides == {ExtractionSide.INPUT, ExtractionSide.OUTPUT}


def _run_chat_with_skip_flags(
    server_mod, fresh_cost_log, monkeypatch: pytest.MonkeyPatch,
    *, skip_input: bool, skip_output: bool,
) -> tuple[list[Any], list[Any]]:
    """Drive one chat turn with the given skip flags and return the
    (extract_knowledge call args, recorded extraction rows)."""
    _patch_pontifex(monkeypatch, server_mod, _basic_chunks())
    _suppress_briefing(monkeypatch, server_mod)

    calls: list[Any] = []

    async def fake_extract_knowledge(*args: Any, **kwargs: Any) -> ExtractionResult:
        calls.append(kwargs)
        return ExtractionResult(
            input_tokens=100, output_tokens=20,
            cache_creation_tokens=0, cache_read_tokens=0,
            model="claude-haiku-4-5-20251001",
            side=kwargs.get("side"),
        )

    monkeypatch.setattr(server_mod, "extract_knowledge", fake_extract_knowledge)
    monkeypatch.setattr(server_mod, "_ensure_user_entity", lambda _: None)

    body: dict[str, Any] = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "ping"}],
    }
    if skip_input:
        body["x-recordatio-skip-input-extraction"] = True
    if skip_output:
        body["x-recordatio-skip-output-extraction"] = True

    client = TestClient(server_mod.app)
    resp = client.post("/v1/chat/completions", headers={"x-agent-id": "alice"}, json=body)
    assert resp.status_code == 200

    async def _drain() -> None:
        await asyncio.gather(*server_mod._background_tasks, return_exceptions=True)

    asyncio.run(_drain())

    extraction_rows = [e for e in fresh_cost_log.query_events() if e["kind"] == "extraction"]
    return calls, extraction_rows


def test_skip_input_extraction_only_runs_output_side(
    server_mod, fresh_cost_log, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, rows = _run_chat_with_skip_flags(
        server_mod, fresh_cost_log, monkeypatch,
        skip_input=True, skip_output=False,
    )

    assert [c["side"] for c in calls] == [ExtractionSide.OUTPUT]
    assert len(rows) == 2

    by_side = {r["extraction_side"]: r for r in rows}
    assert by_side[ExtractionSide.INPUT]["skipped"] is True
    assert by_side[ExtractionSide.INPUT]["input_tokens"] == 0
    assert by_side[ExtractionSide.INPUT]["cost_usd"] == 0.0
    assert by_side[ExtractionSide.OUTPUT]["skipped"] is False
    assert by_side[ExtractionSide.OUTPUT]["input_tokens"] == 100


def test_skip_output_extraction_only_runs_input_side(
    server_mod, fresh_cost_log, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, rows = _run_chat_with_skip_flags(
        server_mod, fresh_cost_log, monkeypatch,
        skip_input=False, skip_output=True,
    )

    assert [c["side"] for c in calls] == [ExtractionSide.INPUT]
    by_side = {r["extraction_side"]: r for r in rows}
    assert by_side[ExtractionSide.OUTPUT]["skipped"] is True
    assert by_side[ExtractionSide.INPUT]["skipped"] is False


def test_skip_both_extractions_records_two_skipped_rows(
    server_mod, fresh_cost_log, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, rows = _run_chat_with_skip_flags(
        server_mod, fresh_cost_log, monkeypatch,
        skip_input=True, skip_output=True,
    )

    assert calls == []  # extract_knowledge never invoked
    assert len(rows) == 2
    assert all(r["skipped"] is True for r in rows)
    assert {r["extraction_side"] for r in rows} == {
        ExtractionSide.INPUT, ExtractionSide.OUTPUT,
    }


def _run_chat_with_content(
    server_mod, fresh_cost_log, monkeypatch: pytest.MonkeyPatch,
    *, user_content: str, response_text: str,
) -> tuple[list[Any], list[Any]]:
    """Drive one chat turn with NO explicit skip flags, using the given
    user-message text and upstream response text. Returns the
    (extract_knowledge call kwargs, recorded extraction rows)."""
    _patch_pontifex(monkeypatch, server_mod, _basic_chunks(response_text))
    _suppress_briefing(monkeypatch, server_mod)

    calls: list[Any] = []

    async def fake_extract_knowledge(*args: Any, **kwargs: Any) -> ExtractionResult:
        calls.append(kwargs)
        return ExtractionResult(
            input_tokens=100, output_tokens=20,
            cache_creation_tokens=0, cache_read_tokens=0,
            model="claude-haiku-4-5-20251001",
            side=kwargs.get("side"),
        )

    monkeypatch.setattr(server_mod, "extract_knowledge", fake_extract_knowledge)
    monkeypatch.setattr(server_mod, "_ensure_user_entity", lambda _: None)

    body: dict[str, Any] = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": user_content}],
    }

    client = TestClient(server_mod.app)
    resp = client.post("/v1/chat/completions", headers={"x-agent-id": "alice"}, json=body)
    assert resp.status_code == 200

    async def _drain() -> None:
        await asyncio.gather(*server_mod._background_tasks, return_exceptions=True)

    asyncio.run(_drain())

    extraction_rows = [e for e in fresh_cost_log.query_events() if e["kind"] == "extraction"]
    return calls, extraction_rows


def test_ignore_for_memory_tag_in_user_turn_skips_both_sides(
    server_mod, fresh_cost_log, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, rows = _run_chat_with_content(
        server_mod, fresh_cost_log, monkeypatch,
        user_content="[please ignore for memory] just a synthetic capability test",
        response_text="sure, noted",
    )

    # A tag on the USER turn marks the whole exchange ephemeral: NEITHER side is
    # mined. The assistant's reply answers an ignored prompt and does not echo
    # the tag, so without whole-exchange suppression the OUTPUT pass would still
    # persist it (the bug this guards against). Both sides record a zero-cost
    # skipped row, exactly like the x-recordatio-skip-* paths.
    assert calls == []
    assert len(rows) == 2
    by_side = {r["extraction_side"]: r for r in rows}
    assert by_side[ExtractionSide.INPUT]["skipped"] is True
    assert by_side[ExtractionSide.INPUT]["input_tokens"] == 0
    assert by_side[ExtractionSide.INPUT]["cost_usd"] == 0.0
    assert by_side[ExtractionSide.OUTPUT]["skipped"] is True
    assert by_side[ExtractionSide.OUTPUT]["input_tokens"] == 0
    assert by_side[ExtractionSide.OUTPUT]["cost_usd"] == 0.0


def test_ignore_for_memory_tag_in_response_skips_output_side(
    server_mod, fresh_cost_log, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, rows = _run_chat_with_content(
        server_mod, fresh_cost_log, monkeypatch,
        user_content="ping",
        response_text="[please ignore for memory] ephemeral reply",
    )

    assert [c["side"] for c in calls] == [ExtractionSide.INPUT]
    by_side = {r["extraction_side"]: r for r in rows}
    assert by_side[ExtractionSide.OUTPUT]["skipped"] is True
    assert by_side[ExtractionSide.INPUT]["skipped"] is False


def test_ignore_for_memory_tag_is_case_insensitive(
    server_mod, fresh_cost_log, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, _rows = _run_chat_with_content(
        server_mod, fresh_cost_log, monkeypatch,
        user_content="[Please Ignore For Memory] mixed-case tag",
        response_text="ok",
    )

    # Case-insensitive match on the user turn still suppresses the whole
    # exchange: neither side runs.
    assert calls == []


def test_no_ignore_for_memory_tag_runs_both_sides(
    server_mod, fresh_cost_log, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, rows = _run_chat_with_content(
        server_mod, fresh_cost_log, monkeypatch,
        user_content="ping", response_text="hi there",
    )

    assert {c["side"] for c in calls} == {ExtractionSide.INPUT, ExtractionSide.OUTPUT}
    assert all(r["skipped"] is False for r in rows)


def test_x_user_id_header_captured(
    server_mod, fresh_cost_log, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_pontifex(monkeypatch, server_mod, _basic_chunks())
    _suppress_extraction(monkeypatch, server_mod)
    _suppress_briefing(monkeypatch, server_mod)

    client = TestClient(server_mod.app)
    resp = client.post(
        "/v1/chat/completions",
        headers={"x-agent-id": "alice", "x-user-id": "user-42"},
        json={"model": "openai/gpt-4o-mini",
              "messages": [{"role": "user", "content": "ping"}]},
    )
    assert resp.status_code == 200

    events = fresh_cost_log.query_events()
    assert len(events) == 1
    assert events[0]["user_id"] == "user-42"
    assert events[0]["agent_id"] == "alice"


def test_chat_response_succeeds_when_cost_log_raises(
    server_mod, fresh_cost_log, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cost-log write failure must NOT break the chat response."""
    _patch_pontifex(monkeypatch, server_mod, _basic_chunks())
    _suppress_extraction(monkeypatch, server_mod)
    _suppress_briefing(monkeypatch, server_mod)

    def boom(**_: Any) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(server_mod.cost_log, "record_chat", boom)

    client = TestClient(server_mod.app)
    resp = client.post(
        "/v1/chat/completions",
        headers={"x-agent-id": "alice"},
        json={"model": "openai/gpt-4o-mini",
              "messages": [{"role": "user", "content": "ping"}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    # Body intact — usage block carries the upstream tokens.
    assert body["usage"]["prompt_tokens"] == 10
    assert body["usage"]["completion_tokens"] == 5


def test_chat_streaming_succeeds_when_cost_log_raises(
    server_mod, fresh_cost_log, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_pontifex(monkeypatch, server_mod, _basic_chunks())
    _suppress_extraction(monkeypatch, server_mod)
    _suppress_briefing(monkeypatch, server_mod)

    def boom(**_: Any) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(server_mod.cost_log, "record_chat", boom)

    client = TestClient(server_mod.app)
    with client.stream(
        "POST", "/v1/chat/completions",
        headers={"x-agent-id": "alice"},
        json={"model": "openai/gpt-4o-mini", "stream": True,
              "messages": [{"role": "user", "content": "ping"}]},
    ) as resp:
        body = b"".join(resp.iter_bytes()).decode()

    # The stream still ended cleanly with [DONE].
    assert "[DONE]" in body
    # And the agentmem.usage event was emitted before the cost-log write attempt.
    assert "event: agentmem.usage" in body


# ---------------------------------------------------------------------------
# /costs dashboard route + JSON endpoint
# ---------------------------------------------------------------------------


def test_costs_html_route_serves_costs_html(server_mod) -> None:
    """`/costs` returns the dashboard HTML, and the page uses relative URLs
    so it works behind a /recordatio/* path prefix."""
    client = TestClient(server_mod.app)
    resp = client.get("/costs")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    body = resp.text
    # Sanity: it's the costs dashboard, not memories or anything else.
    assert "agentmem" in body
    assert "costs" in body.lower()
    # Regression: the page must use a relative fetch, not /costs/data
    # (root-absolute would break behind a path prefix).
    assert "fetch(`costs/data" in body


def test_dashboard_html_routes_disable_browser_cache(server_mod) -> None:
    client = TestClient(server_mod.app)

    for path in ["/costs", "/memories"]:
        resp = client.get(path)

        assert resp.status_code == 200
        assert resp.headers["cache-control"] == "no-store, no-cache, must-revalidate, max-age=0"
        assert resp.headers["pragma"] == "no-cache"
        assert resp.headers["expires"] == "0"


def test_costs_route_does_not_exist(server_mod) -> None:
    """`/ui` was removed — the costs dashboard replaces it."""
    client = TestClient(server_mod.app)
    resp = client.get("/ui")
    assert resp.status_code == 404


def _clear_cost_log(server_mod) -> None:
    """Drop all rows from the module-level cost_log so per-test state is
    deterministic. The cost_log is a singleton that survives across tests."""
    with server_mod.cost_log._lock:
        server_mod.cost_log._conn.execute("DELETE FROM cost_events")
        server_mod.cost_log._conn.commit()


def test_costs_data_returns_aggregates(server_mod) -> None:
    """Empty-DB happy path: totals + grouping arrays present."""
    _clear_cost_log(server_mod)
    client = TestClient(server_mod.app)
    resp = client.get("/costs/data?range=7d")
    assert resp.status_code == 200
    data = resp.json()
    assert data["range"] == "7d"
    assert "totals" in data
    assert "by_kind" in data and isinstance(data["by_kind"], list)
    assert "by_model" in data and isinstance(data["by_model"], list)
    assert "by_agent" in data and isinstance(data["by_agent"], list)
    # Empty DB: zero totals, zero rows.
    assert data["totals"]["cost_usd"] == 0.0
    assert data["totals"]["event_count"] == 0


def test_costs_data_rejects_unknown_range(server_mod) -> None:
    client = TestClient(server_mod.app)
    resp = client.get("/costs/data?range=banana")
    assert resp.status_code == 400


def test_costs_data_aggregates_recorded_events(server_mod) -> None:
    """End-to-end: write a few cost rows, then verify the dashboard surfaces
    them. Uses the existing module-level cost_log so we exercise the same
    code path the chat handler uses."""
    _clear_cost_log(server_mod)
    server_mod.cost_log.record_chat(
        model="claude-sonnet-4-6", provider="anthropic",
        input_tokens=1000, output_tokens=200,
        cache_creation_tokens=0, cache_read_tokens=0,
        cost_usd=0.05,
        agent_id="alice", user_id="u1", request_id="req1",
    )
    server_mod.cost_log.record_extraction(
        model="claude-haiku-4-5", provider="anthropic",
        input_tokens=400, output_tokens=80,
        cache_creation_tokens=0, cache_read_tokens=0,
        cost_usd=0.01,
        agent_id="alice", user_id="u1", request_id="req1",
        ok=True,
        extraction_side=ExtractionSide.OUTPUT,
    )

    client = TestClient(server_mod.app)
    resp = client.get("/costs/data?range=all")
    assert resp.status_code == 200
    data = resp.json()
    assert data["totals"]["cost_usd"] == pytest.approx(0.06)
    assert data["totals"]["event_count"] == 2
    kinds = {row["key"]: row for row in data["by_kind"]}
    assert kinds["chat"]["cost_usd"] == pytest.approx(0.05)
    assert kinds["extraction"]["cost_usd"] == pytest.approx(0.01)
    agent_rows = {row["key"]: row for row in data["by_agent"]}
    assert agent_rows["alice"]["cost_usd"] == pytest.approx(0.06)
    # Top child for the agent grouping is the most-expensive model.
    assert agent_rows["alice"]["top_child"] == "claude-sonnet-4-6"


def test_memories_data_exposes_structured_ontology_rows(
    server_mod, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server_mod, "store", SQLiteStore(tmp_path / "memories.db"))
    now = datetime.now(UTC)
    agent_id = "alice"
    _seed_entity(server_mod, entity_id="entity-user", label="User", agent_id=agent_id)
    _seed_unit(server_mod, unit_id="unit-1", content="User works at Acme.", agent_id=agent_id)

    server_mod.store.upsert_attribute_type(
        AttributeType(
            name="works_at",
            description="Employment relationship as a scalar value",
            value_type="text",
            source="test",
            created_at=now,
        ),
        agent_id,
    )
    server_mod.store.upsert_entity_alias(
        EntityAlias(entity_id="entity-user", alias="Tommaso", source="test", created_at=now),
        agent_id,
    )
    server_mod.store.upsert_entity_identifier(
        EntityIdentifier(
            entity_id="entity-user",
            namespace="email",
            value="tommaso@example.com",
            source="test",
            created_at=now,
        ),
        agent_id,
    )
    server_mod.store.upsert_entity_attribute(
        EntityAttribute(
            entity_id="entity-user",
            attribute_type="works_at",
            value="Acme",
            asserted_at=now,
            valid_from=now,
            provenance="test",
        ),
        agent_id,
    )
    server_mod.store.upsert_entity_mention(
        EntityMention(
            entity_id="entity-user",
            source_kind="knowledge_unit",
            source_id="unit-1",
            snippet="User works at Acme",
            asserted_at=now,
        ),
        agent_id,
    )
    server_mod.store.add_relation_type(
        RelationType(
            id="rt-mentions",
            name="mentions",
            description="Knowledge unit mentions entity",
            source="test",
            created_at=now,
        )
    )
    server_mod.store.add_relation(
        Relation(
            id="relation-1",
            subject_id="unit-1",
            predicate="mentions",
            object_id="entity-user",
            asserted_at=now,
            valid_from=now,
            valid_until=None,
            provenance="test",
        ),
        agent_id,
    )

    class _Request:
        query_params = {"agent_id": "alice"}

    payload = asyncio.run(server_mod.memories(_Request()))
    assert payload["agent_id"] == "alice"
    assert payload["counts"]["attributes"] == 1
    assert payload["counts"]["mentions"] == 1
    assert payload["counts"]["aliases"] == 1
    assert payload["counts"]["identifiers"] == 1
    assert payload["entity_attributes"][0]["attribute_type"] == "works_at"
    assert payload["entity_attributes"][0]["value"] == "Acme"
    assert payload["entity_mentions"][0]["source_label"] == "User works at Acme."
    assert payload["entity_aliases"][0]["alias"] == "Tommaso"
    assert payload["entity_identifiers"][0]["namespace"] == "email"
    assert any(item["name"] == "works_at" for item in payload["schema"]["attribute_types"])
    assert any(node["kind"] == "attribute" for node in payload["nodes"])
    assert any(edge["kind"] == "attribute" for edge in payload["edges"])
    assert any(edge["kind"] == "mention" for edge in payload["edges"])


def test_memories_data_filters_topics_to_selected_agent(
    server_mod, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server_mod, "store", SQLiteStore(tmp_path / "memories.db"))
    now = datetime.now(UTC)
    _seed_unit(server_mod, unit_id="unit-alice", content="Alice likes tea.", agent_id="alice")
    _seed_unit(server_mod, unit_id="unit-bob", content="Bob likes coffee.", agent_id="bob")
    server_mod.store.add_topic(
        Topic(
            id="topic-tea",
            name="tea",
            description="Tea preferences",
            source="test",
            created_at=now,
        )
    )
    server_mod.store.add_topic(
        Topic(
            id="topic-coffee",
            name="coffee",
            description="Coffee preferences",
            source="test",
            created_at=now,
        )
    )
    server_mod.store.tag_unit("unit-alice", "topic-tea", "alice")
    server_mod.store.tag_unit("unit-bob", "topic-coffee", "bob")

    class _Request:
        query_params = {"agent_id": "alice"}

    payload = asyncio.run(server_mod.memories(_Request()))
    topic_nodes = {node["id"] for node in payload["nodes"] if node["kind"] == "topic"}
    topic_edges = [edge for edge in payload["edges"] if edge["kind"] == "topic"]

    assert topic_nodes == {"topic-tea"}
    assert topic_edges == [
        {"source": "unit-alice", "target": "topic-tea", "label": "tagged_with", "kind": "topic"}
    ]


def test_memories_agents_lists_all_non_default_agent_namespaces(
    server_mod, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server_mod, "store", SQLiteStore(tmp_path / "memories.db"))
    now = datetime.now(UTC)
    helper_agent = "tommaso/helper"
    research_agent = "sam/research"

    _seed_entity(server_mod, entity_id="entity-user", label="User", agent_id=helper_agent)
    _seed_unit(server_mod, unit_id="unit-helper", content="User works at Acme.", agent_id=helper_agent)
    _seed_unit(
        server_mod,
        unit_id="unit-research",
        content="Research agent tracks Initech filings.",
        agent_id=research_agent,
    )
    _seed_unit(
        server_mod,
        unit_id="unit-default",
        content="Default namespace should not appear.",
        agent_id="default",
    )

    server_mod.store.upsert_entity_alias(
        EntityAlias(entity_id="entity-user", alias="Tommaso", source="test", created_at=now),
        helper_agent,
    )
    server_mod.store.upsert_entity_identifier(
        EntityIdentifier(
            entity_id="entity-user",
            namespace="email",
            value="tommaso@example.com",
            source="test",
            created_at=now,
        ),
        helper_agent,
    )
    server_mod.store.upsert_entity_attribute(
        EntityAttribute(
            entity_id="entity-user",
            attribute_type="employer",
            value="Acme",
            asserted_at=now,
            valid_from=now,
            provenance="test",
        ),
        helper_agent,
    )
    server_mod.store.upsert_entity_mention(
        EntityMention(
            entity_id="entity-user",
            source_kind="knowledge_unit",
            source_id="unit-helper",
            snippet="User works at Acme",
            asserted_at=now,
        ),
        helper_agent,
    )
    server_mod.store.add_relation(
        Relation(
            id="relation-helper",
            subject_id="unit-helper",
            predicate="mentions",
            object_id="entity-user",
            asserted_at=now,
            valid_from=now,
            valid_until=None,
            provenance="test",
        ),
        helper_agent,
    )

    payload = asyncio.run(server_mod.memories_agents())
    agents = {item["agent_id"]: item for item in payload["agents"]}

    assert set(agents) == {research_agent, helper_agent}
    assert payload["users"] == {"sam": 1, "tommaso": 1}
    assert payload["totals"]["agents"] == 2
    assert payload["totals"]["users"] == 2
    assert payload["totals"]["units"] == 2
    assert agents[helper_agent]["user"] == "tommaso"
    assert agents[helper_agent]["name"] == "helper"
    assert agents[helper_agent]["counts"] == {
        "units": 1,
        "entities": 1,
        "relations": 1,
        "attributes": 1,
        "mentions": 1,
        "aliases": 1,
        "identifiers": 1,
    }
    assert agents[research_agent]["counts"]["units"] == 1
    assert agents[research_agent]["counts"]["entities"] == 0


def test_provider_tag_helper(server_mod) -> None:
    """Direct unit test for the provider-tag sniffing helper."""
    tag = server_mod._provider_tag
    assert tag("anthropic", None) == "anthropic"
    assert tag("openai", None) == "openai"
    assert tag("openai", "http://localhost:11434/v1") == "openai"
    assert tag("openai", "http://ollama.local/v1") == "openai"
    assert tag("openai", "http://localhost:1234/v1") == "openai"
    assert tag("openai", "http://lmstudio.local/v1") == "openai"
    assert tag("openai", "https://generativelanguage.googleapis.com/v1beta/openai/") == "openai"
    assert tag("openai", "https://api.example.com/v1") == "openai"


def test_merge_cross_side_deduplicates_structured_extraction_fields(server_mod) -> None:
    now = datetime.now(UTC)
    primary_unit = KnowledgeUnit(
        type=KnowledgeType.SEMANTIC,
        content="User lives in Sunnydale.",
        asserted_at=now,
        valid_from=now,
        provenance="input",
    )
    secondary_unit = KnowledgeUnit(
        type=KnowledgeType.SEMANTIC,
        content="User works at Acme.",
        asserted_at=now,
        valid_from=now,
        provenance="output",
    )
    primary = ExtractionResult(
        units=[primary_unit],
        memory_units=[
            AtomicMemoryUnit(id="unit:home", type=KnowledgeType.SEMANTIC, content="User lives in Sunnydale.")
        ],
        entity_mentions=[ExtractionEntityMention(id="entity:user", label="User", type="person")],
        attributes=[AttributeExtraction(subject="User", attribute="home_city", value="Sunnydale")],
        relation_candidates=[
            RelationExtraction(subject="User", predicate="lives_in", object="Sunnydale")
        ],
    )
    secondary = ExtractionResult(
        units=[primary_unit, secondary_unit],
        memory_units=[
            AtomicMemoryUnit(id="unit:home-duplicate", type=KnowledgeType.SEMANTIC, content="User lives in Sunnydale."),
            AtomicMemoryUnit(id="unit:work", type=KnowledgeType.SEMANTIC, content="User works at Acme."),
        ],
        entity_mentions=[
            ExtractionEntityMention(id="entity:user-duplicate", label="User", type="person"),
            ExtractionEntityMention(id="entity:acme", label="Acme", type="organization"),
        ],
        attributes=[
            AttributeExtraction(subject="User", attribute="home_city", value="Sunnydale"),
            AttributeExtraction(subject="User", attribute="employer", value="Acme"),
        ],
        relation_candidates=[
            RelationExtraction(subject="User", predicate="lives_in", object="Sunnydale"),
            RelationExtraction(subject="User", predicate="works_at", object="Acme"),
        ],
        schema_proposals=[
            SchemaProposal(kind=SchemaProposalKind.RELATION_TYPE, name="works_at", description="employment")
        ],
    )

    merged = server_mod._merge_cross_side(primary, secondary)

    assert [unit.content for unit in merged.units] == ["User lives in Sunnydale.", "User works at Acme."]
    assert [unit.id for unit in merged.memory_units] == ["unit:home", "unit:work"]
    assert [mention.label for mention in merged.entity_mentions] == ["User", "Acme"]
    assert [(attribute.attribute, attribute.value) for attribute in merged.attributes] == [
        ("home_city", "Sunnydale"),
        ("employer", "Acme"),
    ]
    assert [relation.predicate for relation in merged.relation_candidates] == ["lives_in", "works_at"]
    assert [proposal.name for proposal in merged.schema_proposals] == ["works_at"]


async def test_persist_side_does_not_resolve_relations_through_unit_content(
    server_mod, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server_mod, "store", SQLiteStore(tmp_path / "persist.db"))
    now = datetime.now(UTC)
    result = ExtractionResult(
        units=[
            KnowledgeUnit(
                type=KnowledgeType.SEMANTIC,
                content="User lives in Sunnydale",
                asserted_at=now,
                valid_from=now,
                provenance="test",
            )
        ],
    )

    await server_mod._persist_side(result, "alice")

    rows = server_mod.store._conn.execute(
        "SELECT predicate FROM relations WHERE agent_id = ? ORDER BY predicate",
        ("alice",),
    ).fetchall()
    predicates = [row["predicate"] for row in rows]
    assert predicates == []
    relation_type_names = {rt.name for rt in server_mod.store.list_relation_types()}
    assert "about_fact" not in relation_type_names


async def test_persist_side_stores_only_used_relation_type_schema_proposals(
    server_mod, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server_mod, "store", SQLiteStore(tmp_path / "persist.db"))
    result = ExtractionResult(
        entity_mentions=[
            ExtractionEntityMention(id="entity:user", label="User", type="person"),
            ExtractionEntityMention(id="entity:sunnydale", label="Sunnydale", type="place"),
        ],
        relation_candidates=[
            RelationExtraction(
                subject="User",
                subject_id="entity:user",
                predicate="visited",
                object="Sunnydale",
                object_id="entity:sunnydale",
            )
        ],
        schema_proposals=[
            SchemaProposal(
                kind=SchemaProposalKind.RELATION_TYPE,
                name="visited",
                description="entity visited a place",
                confidence=0.95,
                evidence=[EvidenceReference(source="target", quote="User visited Sunnydale.")],
            ),
            SchemaProposal(
                kind=SchemaProposalKind.RELATION_TYPE,
                name="unused_predicate",
                description="not referenced by a valid relation",
                confidence=0.95,
                evidence=[EvidenceReference(source="target", quote="User visited Sunnydale.")],
            ),
        ],
    )

    await server_mod._persist_side(result, "alice")

    relation_type_names = {rt.name for rt in server_mod.store.list_relation_types()}
    assert "visited" in relation_type_names
    assert "unused_predicate" not in relation_type_names


async def test_persist_side_rejects_low_quality_schema_topic_proposals(
    server_mod, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server_mod, "store", SQLiteStore(tmp_path / "persist.db"))
    result = ExtractionResult(
        schema_proposals=[
            SchemaProposal(
                kind=SchemaProposalKind.TOPIC,
                name="housing",
                description="housing preferences",
                confidence=0.4,
                evidence=[EvidenceReference(source="target", quote="User wants a flat.")],
            ),
            SchemaProposal(
                kind=SchemaProposalKind.TOPIC,
                name="commute",
                description="commute preferences",
                confidence=0.99,
            ),
            SchemaProposal(
                kind=SchemaProposalKind.TOPIC,
                name="schools",
                description="school preferences",
                confidence=0.99,
                evidence=[EvidenceReference(source="target", quote="near the school")],
            ),
        ],
    )

    await server_mod._persist_side(result, "alice")

    topic_names = {topic.name for topic in server_mod.store.list_topics()}
    assert "housing" not in topic_names
    assert "commute" not in topic_names
    assert "schools" in topic_names


async def test_persist_side_stores_structured_entity_metadata(
    server_mod, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server_mod, "store", SQLiteStore(tmp_path / "persist.db"))
    now = datetime.now(UTC)
    unit = KnowledgeUnit(
        id="unit-user",
        type=KnowledgeType.SEMANTIC,
        content="User works at Acme and prefers green.",
        asserted_at=now,
        valid_from=now,
        provenance="test",
    )
    result = ExtractionResult(
        units=[unit],
        entity_mentions=[
            ExtractionEntityMention(
                id="entity:user",
                label="User",
                type="person",
                aliases=["Tommaso"],
                identifiers={"email": "tommaso@example.com"},
                canonical_key="person:user",
            ),
            ExtractionEntityMention(id="entity:acme", label="Acme", type="organization"),
        ],
        attributes=[
            AttributeExtraction(
                subject="User",
                subject_id="entity:user",
                attribute="favorite_color",
                value="green",
            )
        ],
        relation_candidates=[
            RelationExtraction(
                subject="User",
                subject_id="entity:user",
                predicate="works_at",
                object="Acme",
                object_id="entity:acme",
            )
        ],
        schema_proposals=[
            SchemaProposal(
                kind=SchemaProposalKind.RELATION_TYPE,
                name="works_at",
                description="employment",
            )
        ],
    )

    await server_mod._persist_side(result, "alice")

    # IDENTITY_DESIGN.md §2: the "User" mention binds to the principal in the
    # user-level store; its aliases/identifiers/attributes/mentions live there.
    uscope = user_scope("alice")
    user = get_principal(server_mod.store, uscope)
    acme = server_mod.store.get_entity_by_label("Acme", "alice")
    assert user is not None
    assert acme is not None
    assert server_mod.store.get_entity_by_label("User", "alice") is None  # no agent-scope user referent
    assert [alias.alias for alias in server_mod.store.list_entity_aliases(user.id, uscope)] == ["Tommaso"]
    assert [
        (identifier.namespace, identifier.value)
        for identifier in server_mod.store.list_entity_identifiers(user.id, uscope)
    ] == [("email", "tommaso@example.com")]
    assert [
        (attribute.attribute_type, attribute.value)
        for attribute in server_mod.store.list_entity_attributes(user.id, uscope, active_only=True)
    ] == [("favorite_color", "green")]
    assert {
        (mention.entity_id, mention.source_kind, mention.source_id)
        for mention in server_mod.store.find_entity_mentions_by_source("knowledge_unit", "unit-user", uscope)
    } == {(user.id, "knowledge_unit", "unit-user")}
    assert {
        (mention.entity_id, mention.source_kind, mention.source_id)
        for mention in server_mod.store.find_entity_mentions_by_source("knowledge_unit", "unit-user", "alice")
    } == {(acme.id, "knowledge_unit", "unit-user")}
    # The user-anchored relation lives in the user store with the non-principal
    # endpoint mirrored (§2.3); Acme itself stays agent-scoped.
    acme_mirror = mirror_entity_id(uscope, acme.type, acme.label)
    relation_rows = server_mod.store._conn.execute(
        """
        SELECT agent_id, subject_id, predicate, object_id
        FROM relations
        WHERE predicate = ? AND valid_until IS NULL
        """,
        ("works_at",),
    ).fetchall()
    assert [(row["agent_id"], row["subject_id"], row["object_id"]) for row in relation_rows] == [
        (uscope, user.id, acme_mirror)
    ]


async def test_persist_side_drops_low_confidence_unresolved_relation_object(
    server_mod, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server_mod, "store", SQLiteStore(tmp_path / "persist.db"))
    result = ExtractionResult(
        entity_mentions=[
            ExtractionEntityMention(id="entity:user", label="User", type="person"),
        ],
        relation_candidates=[
            RelationExtraction(
                subject="User",
                subject_id="entity:user",
                predicate="works_at",
                object="Acme Labs",
            )
        ],
    )

    await server_mod._persist_side(result, "alice")

    user = get_principal(server_mod.store, user_scope("alice"))
    assert user is not None
    attributes = server_mod.store.list_entity_attributes(user.id, user_scope("alice"), "works_at", active_only=True)
    assert attributes == []
    rows = server_mod.store._conn.execute(
        "SELECT id FROM relations WHERE predicate = ?",
        ("works_at",),
    ).fetchall()
    assert rows == []


async def test_persist_side_allows_high_confidence_unresolved_scalar_attribute(
    server_mod, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server_mod, "store", SQLiteStore(tmp_path / "persist.db"))
    result = ExtractionResult(
        entity_mentions=[
            ExtractionEntityMention(id="entity:user", label="User", type="person"),
        ],
        relation_candidates=[
            RelationExtraction(
                subject="User",
                subject_id="entity:user",
                predicate="works_at",
                object="Acme Labs",
                confidence=0.96,
                evidence=[EvidenceReference(source="target", quote="User works at Acme Labs.")],
            )
        ],
    )

    await server_mod._persist_side(result, "alice")

    user = get_principal(server_mod.store, user_scope("alice"))
    assert user is not None
    attributes = server_mod.store.list_entity_attributes(user.id, user_scope("alice"), "works_at", active_only=True)
    assert [(attribute.attribute_type, attribute.value) for attribute in attributes] == [("works_at", "Acme Labs")]


async def test_persist_side_supersedes_functional_scalar_attributes(
    server_mod, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server_mod, "store", SQLiteStore(tmp_path / "persist.db"))
    first = ExtractionResult(
        entity_mentions=[ExtractionEntityMention(id="entity:user", label="User", type="person")],
        relation_candidates=[
            RelationExtraction(
                subject="User",
                subject_id="entity:user",
                predicate="works_at",
                object="Acme",
                confidence=0.96,
                evidence=[EvidenceReference(source="target", quote="User works at Acme.")],
            )
        ],
    )
    second = ExtractionResult(
        entity_mentions=[ExtractionEntityMention(id="entity:user", label="User", type="person")],
        relation_candidates=[
            RelationExtraction(
                subject="User",
                subject_id="entity:user",
                predicate="works_at",
                object="Globex",
                confidence=0.96,
                evidence=[EvidenceReference(source="target", quote="User works at Globex.")],
            )
        ],
    )

    await server_mod._persist_side(first, "alice")
    await server_mod._persist_side(second, "alice")

    user = get_principal(server_mod.store, user_scope("alice"))
    assert user is not None
    active = server_mod.store.list_entity_attributes(user.id, user_scope("alice"), "works_at", active_only=True)
    all_attrs = server_mod.store.list_entity_attributes(user.id, user_scope("alice"), "works_at")
    assert [attribute.value for attribute in active] == ["Globex"]
    assert {attribute.value for attribute in all_attrs} == {"Acme", "Globex"}


async def test_persist_side_filters_low_quality_extraction_candidates(
    server_mod, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server_mod, "store", SQLiteStore(tmp_path / "persist.db"))
    now = datetime.now(UTC)
    result = ExtractionResult(
        units=[
            KnowledgeUnit(
                type=KnowledgeType.SEMANTIC,
                content="User wants a flat near Riverside Park.",
                asserted_at=now,
                valid_from=now,
                provenance="extraction",
            )
        ],
        memory_units=[
            AtomicMemoryUnit(
                id="unit:flat",
                type=KnowledgeType.SEMANTIC,
                content="User wants a flat near Riverside Park.",
                confidence=0.4,
                evidence=[EvidenceReference(source="target", quote="User wants a flat near Riverside Park.")],
            )
        ],
        entity_mentions=[
            ExtractionEntityMention(
                id="entity:hyde-park",
                label="Riverside Park",
                confidence=0.4,
                evidence=[EvidenceReference(source="target", quote="Riverside Park")],
            ),
            ExtractionEntityMention(
                id="entity:user",
                label="User",
                type="person",
                confidence=0.4,
                evidence=[EvidenceReference(source="target", quote="User")],
            ),
        ],
        attributes=[
            AttributeExtraction(
                subject="User",
                subject_id="entity:user",
                attribute="preferred_location",
                value="Riverside Park",
                confidence=0.4,
                evidence=[EvidenceReference(source="target", quote="near Riverside Park")],
            )
        ],
        relation_candidates=[
            RelationExtraction(
                subject="User",
                subject_id="entity:user",
                predicate="near",
                object="Riverside Park",
                confidence=0.4,
                evidence=[],
            )
        ],
    )

    await server_mod._persist_side(result, "alice")

    rows = server_mod.store._conn.execute(
        "SELECT content FROM knowledge_units WHERE agent_id = ?",
        ("alice",),
    ).fetchall()
    assert rows == []
    assert server_mod.store.get_entity_by_label("Riverside Park", "alice") is None
    user = get_principal(server_mod.store, user_scope("alice"))
    assert user is not None
    assert server_mod.store.list_entity_attributes(user.id, user_scope("alice")) == []
    relation_rows = server_mod.store._conn.execute(
        "SELECT predicate FROM relations",
    ).fetchall()
    assert relation_rows == []


async def test_persist_side_reuses_existing_unit_for_duplicate_content(
    server_mod, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server_mod, "store", SQLiteStore(tmp_path / "persist.db"))
    now = datetime.now(UTC)
    server_mod.store.add(
        KnowledgeUnit(
            id="unit-existing",
            type=KnowledgeType.SEMANTIC,
            content="User lives in Springfield.",
            asserted_at=now,
            valid_from=now,
            provenance="test",
        ),
        "alice",
    )
    result = ExtractionResult(
        units=[
            KnowledgeUnit(
                id="unit-new",
                type=KnowledgeType.SEMANTIC,
                content="User lives in Springfield.",
                asserted_at=now,
                valid_from=now,
                provenance="extraction",
            )
        ],
        memory_units=[
            AtomicMemoryUnit(
                id="unit:home",
                type=KnowledgeType.SEMANTIC,
                content="User lives in Springfield.",
                confidence=0.95,
                evidence=[EvidenceReference(source="target", quote="User lives in Springfield.")],
            )
        ],
        entity_mentions=[
            ExtractionEntityMention(
                id="entity:user",
                label="User",
                type="person",
                confidence=0.95,
                evidence=[EvidenceReference(source="unit", source_id="unit:home", quote="User")],
            )
        ],
    )

    await server_mod._persist_side(result, "alice")

    units = server_mod.store._conn.execute(
        "SELECT id, content FROM knowledge_units WHERE agent_id = ? AND valid_until IS NULL",
        ("alice",),
    ).fetchall()
    assert [(row["id"], row["content"]) for row in units] == [("unit-existing", "User lives in Springfield.")]
    # The "User" mention binds to the principal; its mention evidence is
    # user-store material referencing the (still agent-scoped) deduped unit.
    user = get_principal(server_mod.store, user_scope("alice"))
    assert user is not None
    mentions = server_mod.store.find_entity_mentions_by_source("knowledge_unit", "unit-existing", user_scope("alice"))
    assert [(mention.entity_id, mention.source_id, mention.snippet) for mention in mentions] == [
        (user.id, "unit-existing", "User")
    ]


async def test_persist_side_skips_duplicate_active_scalar_attribute(
    server_mod, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server_mod, "store", SQLiteStore(tmp_path / "persist.db"))
    result = ExtractionResult(
        entity_mentions=[
            ExtractionEntityMention(id="entity:user", label="User", type="person"),
        ],
        attributes=[
            AttributeExtraction(
                subject="User",
                subject_id="entity:user",
                attribute="favorite_color",
                value="green",
            )
        ],
    )

    await server_mod._persist_side(result, "alice")
    user = get_principal(server_mod.store, user_scope("alice"))
    assert user is not None
    first_rows = server_mod.store._conn.execute(
        """
        SELECT id
        FROM entity_attributes
        WHERE agent_id = ? AND entity_id = ? AND attribute_type = ? AND value = ?
        """,
        (user_scope("alice"), user.id, "favorite_color", "green"),
    ).fetchall()
    assert len(first_rows) == 1

    await server_mod._persist_side(result, "alice")

    second_rows = server_mod.store._conn.execute(
        """
        SELECT id
        FROM entity_attributes
        WHERE agent_id = ? AND entity_id = ? AND attribute_type = ? AND value = ?
        """,
        (user_scope("alice"), user.id, "favorite_color", "green"),
    ).fetchall()
    assert [row["id"] for row in second_rows] == [row["id"] for row in first_rows]


# ---------------------------------------------------------------------------
# Extraction reinforcement: re-asserting a fact bumps extraction_count
# ---------------------------------------------------------------------------


def _reinforcement_fact_result(server_mod) -> ExtractionResult:
    """A fresh ExtractionResult re-asserting "User lives in Sunnydale.".

    Built fresh per call because ``_persist_side`` mutates unit ids / resolves
    entity mentions in place, so the same instance can't be persisted twice.
    """
    now = datetime.now(UTC)
    return ExtractionResult(
        units=[
            KnowledgeUnit(
                type=KnowledgeType.SEMANTIC,
                content="User lives in Sunnydale.",
                asserted_at=now,
                valid_from=now,
                provenance="extraction",
            )
        ],
        memory_units=[
            AtomicMemoryUnit(
                id="unit:home",
                type=KnowledgeType.SEMANTIC,
                content="User lives in Sunnydale.",
                confidence=0.95,
                evidence=[EvidenceReference(source="target", quote="User lives in Sunnydale.")],
            )
        ],
        entity_mentions=[
            ExtractionEntityMention(
                id="entity:user",
                label="User",
                type="person",
                canonical_key="person:user",
                confidence=0.95,
                evidence=[EvidenceReference(source="unit", source_id="unit:home", quote="User")],
            ),
            ExtractionEntityMention(
                id="entity:sunnydale",
                label="Sunnydale",
                type="place",
                canonical_key="place:sunnydale",
                confidence=0.95,
                evidence=[EvidenceReference(source="unit", source_id="unit:home", quote="Sunnydale")],
            ),
        ],
        relation_candidates=[
            RelationExtraction(
                subject="User",
                subject_id="entity:user",
                predicate="lives_in",
                object="Sunnydale",
                object_id="entity:sunnydale",
                confidence=0.95,
                evidence=[EvidenceReference(source="target", quote="User lives in Sunnydale.")],
            )
        ],
    )


def _entity_count(server_mod, canonical_key: str, agent_id: str) -> int:
    entity = server_mod.store.get_entity_by_canonical_key(canonical_key, agent_id, "place")
    assert entity is not None
    return entity.extraction_count


def _unit_count(server_mod, agent_id: str) -> tuple[str, int]:
    """The single active unit row: user-anchored semantic units live in the
    agent's USER scope under identity routing (IDENTITY_DESIGN.md §2.2)."""
    rows = server_mod.store._conn.execute(
        "SELECT id, extraction_count FROM knowledge_units WHERE agent_id = ? AND valid_until IS NULL",
        (user_scope(agent_id),),
    ).fetchall()
    assert len(rows) == 1, [r["id"] for r in rows]
    return rows[0]["id"], rows[0]["extraction_count"]


def _relation_count(server_mod, agent_id: str) -> int:
    """The single active lives_in row: principal-endpoint relations live in
    the agent's USER scope under identity routing (IDENTITY_DESIGN.md §2.2)."""
    rows = server_mod.store._conn.execute(
        "SELECT extraction_count FROM relations "
        "WHERE agent_id = ? AND predicate = 'lives_in' AND valid_until IS NULL",
        (user_scope(agent_id),),
    ).fetchall()
    assert len(rows) == 1
    return rows[0]["extraction_count"]


async def test_reasserting_fact_bumps_extraction_counts(
    server_mod, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Extract a fact, then extract the SAME fact again via the real persistence
    path: the existing entity/relation/unit rows go 1 -> 2 (no duplicates)."""
    monkeypatch.setattr(server_mod, "store", SQLiteStore(tmp_path / "reinforce.db"))

    await server_mod._persist_side(_reinforcement_fact_result(server_mod), "alice")

    # First insert: defaults of 1.
    unit_id, unit_count = _unit_count(server_mod, "alice")
    assert unit_count == 1
    assert _entity_count(server_mod, "place:sunnydale", "alice") == 1
    assert _relation_count(server_mod, "alice") == 1

    await server_mod._persist_side(_reinforcement_fact_result(server_mod), "alice")

    # Re-assertion: same rows, counts bumped to 2, no duplicate rows inserted.
    same_unit_id, unit_count = _unit_count(server_mod, "alice")
    assert same_unit_id == unit_id  # reused, not duplicated
    assert unit_count == 2
    assert _entity_count(server_mod, "place:sunnydale", "alice") == 2
    assert _relation_count(server_mod, "alice") == 2

    relation_rows = server_mod.store._conn.execute(
        "SELECT id FROM relations WHERE agent_id = ? AND predicate = 'lives_in'",
        (user_scope("alice"),),
    ).fetchall()
    assert len(relation_rows) == 1  # the unique-index conflict was reinforced, not re-inserted


async def test_new_fact_inserts_with_count_one(
    server_mod, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A genuinely new fact inserts entity/relation/unit at the default count 1."""
    monkeypatch.setattr(server_mod, "store", SQLiteStore(tmp_path / "fresh.db"))

    await server_mod._persist_side(_reinforcement_fact_result(server_mod), "alice")

    _, unit_count = _unit_count(server_mod, "alice")
    assert unit_count == 1
    assert _entity_count(server_mod, "place:sunnydale", "alice") == 1
    assert _relation_count(server_mod, "alice") == 1


async def test_reinforcement_is_scoped_by_agent_id(
    server_mod, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One agent re-asserting a fact must not bump another agent's identical fact."""
    monkeypatch.setattr(server_mod, "store", SQLiteStore(tmp_path / "scoped.db"))

    # Seed the identical fact for both agents (each at count 1).
    await server_mod._persist_side(_reinforcement_fact_result(server_mod), "alice")
    await server_mod._persist_side(_reinforcement_fact_result(server_mod), "bob")

    # Re-assert only for alice.
    await server_mod._persist_side(_reinforcement_fact_result(server_mod), "alice")

    assert _entity_count(server_mod, "place:sunnydale", "alice") == 2
    assert _relation_count(server_mod, "alice") == 2
    assert _unit_count(server_mod, "alice")[1] == 2

    # bob's identical fact is independent — still at 1.
    assert _entity_count(server_mod, "place:sunnydale", "bob") == 1
    assert _relation_count(server_mod, "bob") == 1
    assert _unit_count(server_mod, "bob")[1] == 1


# ---------------------------------------------------------------------------
# Briefing-events persistence + API (#45)
# ---------------------------------------------------------------------------


@pytest.fixture()
def fresh_briefing_log(server_mod, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Replace ``server_mod.briefing_log`` with a fresh BriefingLog on tmp_path.

    Same isolation pattern as ``fresh_cost_log``: the module-level
    ``briefing_log`` outlives individual tests, and we don't want rows from
    one test leaking into another's assertions.
    """
    log = BriefingLog(tmp_path / "briefing_isolated.db")
    monkeypatch.setattr(server_mod, "briefing_log", log)
    return log


def _seed_unit(server_mod, *, unit_id: str, content: str, agent_id: str) -> None:
    """Insert a knowledge unit directly so hydration tests have something
    to join against. Uses the test's per-tmp_path store."""
    now = datetime.now(UTC)
    server_mod.store.add(
        KnowledgeUnit(
            id=unit_id, type=KnowledgeType.SEMANTIC, content=content,
            asserted_at=now, valid_from=now, provenance="test",
        ),
        agent_id,
    )


def _seed_entity(server_mod, *, entity_id: str, label: str, agent_id: str) -> None:
    server_mod.store.add_entity(
        Entity(id=entity_id, label=label, asserted_at=datetime.now(UTC)),
        agent_id,
    )


@pytest.mark.asyncio
async def test_build_system_persists_briefing_row_with_call_id(
    server_mod, fresh_briefing_log,
) -> None:
    """`_build_system` writes a briefing row keyed by call_id when retrieval
    yielded something. The row carries the right agent/query/IDs."""
    with patch.object(server_mod, "_retrieve", _fake_retrieve_with_briefing):
        await server_mod._build_system(
            None, "what do I prefer?", "alice",
            backend="openai", call_id="recordatio-test123",
        )

    row = fresh_briefing_log.get_by_request_id("recordatio-test123")
    assert row is not None
    assert row["request_id"] == "recordatio-test123"
    assert row["agent_id"] == "alice"
    assert row["query_text"] == "what do I prefer?"
    assert row["recalled_unit_ids"] == ["u1"]
    assert row["recalled_entity_ids"] == []
    assert row["recalled_relation_ids"] == []
    assert row["count_units"] == 1


@pytest.mark.asyncio
async def test_build_system_does_not_persist_when_briefing_empty(
    server_mod, fresh_briefing_log,
) -> None:
    """No row should be written when retrieval found nothing — empty rows
    just clutter the table and the eventual dashboard view."""
    with patch.object(server_mod, "_retrieve", _fake_retrieve_empty):
        await server_mod._build_system(
            "stable system", "query", "alice",
            backend="openai", call_id="recordatio-empty",
        )

    assert fresh_briefing_log.get_by_request_id("recordatio-empty") is None
    assert fresh_briefing_log.list_by_agent(agent_id="alice") == []


@pytest.mark.asyncio
async def test_build_system_skips_persistence_when_call_id_missing(
    server_mod, fresh_briefing_log,
) -> None:
    """Backwards-compat: callers that don't pass a call_id (e.g. internal
    helpers) still get a briefing back, just without persistence."""
    with patch.object(server_mod, "_retrieve", _fake_retrieve_with_briefing):
        result = await server_mod._build_system(
            None, "q", "alice", backend="openai",
        )
    assert result.briefing_data is not None
    # Nothing recorded — no call_id, no key.
    assert fresh_briefing_log.list_by_agent() == []


def test_get_briefing_event_hydrates_unit_content(
    server_mod, fresh_briefing_log,
) -> None:
    """GET /v1/briefing-events/{request_id} returns full unit content,
    not just IDs — that's the whole point of the endpoint."""
    _seed_unit(server_mod, unit_id="u-A", content="Tommaso likes oat milk", agent_id="alice")
    fresh_briefing_log.record_briefing(
        request_id="recordatio-A", agent_id="alice",
        query_text="what does tommaso like?", topic_hint=None,
        recalled_unit_ids=["u-A"], recalled_entity_ids=[], recalled_relation_ids=[],
    )

    client = TestClient(server_mod.app)
    resp = client.get("/v1/briefing-events/recordatio-A")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["request_id"] == "recordatio-A"
    assert payload["agent_id"] == "alice"
    assert payload["query_text"] == "what does tommaso like?"
    assert payload["count_units"] == 1
    assert len(payload["units"]) == 1
    unit = payload["units"][0]
    assert unit["id"] == "u-A"
    assert unit["content"] == "Tommaso likes oat milk"
    assert unit["type"] == "semantic"


def test_get_briefing_event_hydrates_entity_label(
    server_mod, fresh_briefing_log,
) -> None:
    _seed_entity(server_mod, entity_id="e-1", label="Tommaso", agent_id="alice")
    fresh_briefing_log.record_briefing(
        request_id="recordatio-E", agent_id="alice",
        query_text="who?", topic_hint=None,
        recalled_unit_ids=[], recalled_entity_ids=["e-1"], recalled_relation_ids=[],
    )

    client = TestClient(server_mod.app)
    resp = client.get("/v1/briefing-events/recordatio-E")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["count_entities"] == 1
    assert payload["entities"][0]["id"] == "e-1"
    assert payload["entities"][0]["label"] == "Tommaso"


def test_get_briefing_event_reports_missing_ids(
    server_mod, fresh_briefing_log,
) -> None:
    """Deleted units remain visible in accounting, without stale content."""
    _seed_unit(server_mod, unit_id="u-real", content="real fact", agent_id="alice")
    fresh_briefing_log.record_briefing(
        request_id="recordatio-mix", agent_id="alice",
        query_text="q", topic_hint=None,
        recalled_unit_ids=["u-real", "u-deleted"],
        recalled_entity_ids=[], recalled_relation_ids=[],
    )

    client = TestClient(server_mod.app)
    resp = client.get("/v1/briefing-events/recordatio-mix")
    assert resp.status_code == 200
    payload = resp.json()
    # count_* reflects what was *recalled*, not what hydrated.
    assert payload["count_units"] == 2
    assert len(payload["units"]) == 1
    assert payload["units"][0]["id"] == "u-real"
    assert payload["recalled_unit_ids"] == ["u-real", "u-deleted"]
    assert payload["content_source"] == "current_store"
    assert payload["hydration_complete"] is False
    assert payload["hydration"]["units"] == {
        "hydrated_count": 1, "unresolved_ids": ["u-deleted"],
        "errors": [], "complete": False,
    }


def test_get_briefing_event_returns_404_for_unknown_request(
    server_mod, fresh_briefing_log,
) -> None:
    client = TestClient(server_mod.app)
    resp = client.get("/v1/briefing-events/recordatio-does-not-exist")
    assert resp.status_code == 404
    assert "error" in resp.json()


def test_list_briefing_events_filters_by_agent(
    server_mod, fresh_briefing_log,
) -> None:
    """The list endpoint scopes to the given agent_id and ignores others."""
    fresh_briefing_log.record_briefing(
        request_id="recordatio-alice-1", agent_id="alice",
        query_text="q1", topic_hint=None,
        recalled_unit_ids=[], recalled_entity_ids=[], recalled_relation_ids=[],
    )
    fresh_briefing_log.record_briefing(
        request_id="recordatio-bob-1", agent_id="bob",
        query_text="q2", topic_hint=None,
        recalled_unit_ids=[], recalled_entity_ids=[], recalled_relation_ids=[],
    )
    fresh_briefing_log.record_briefing(
        request_id="recordatio-alice-2", agent_id="alice",
        query_text="q3", topic_hint=None,
        recalled_unit_ids=[], recalled_entity_ids=[], recalled_relation_ids=[],
    )

    client = TestClient(server_mod.app)
    resp = client.get("/v1/briefing-events?agent_id=alice")
    assert resp.status_code == 200
    body = resp.json()
    assert body["limit"] == 50
    assert body["offset"] == 0
    request_ids = [e["request_id"] for e in body["events"]]
    # Newest-first ordering.
    assert request_ids == ["recordatio-alice-2", "recordatio-alice-1"]
    # Bob's row is absent.
    assert "recordatio-bob-1" not in request_ids


def test_list_briefing_events_respects_limit_and_offset(
    server_mod, fresh_briefing_log,
) -> None:
    for i in range(5):
        fresh_briefing_log.record_briefing(
            request_id=f"recordatio-{i}", agent_id="alice",
            query_text=f"q{i}", topic_hint=None,
            recalled_unit_ids=[], recalled_entity_ids=[], recalled_relation_ids=[],
        )

    client = TestClient(server_mod.app)
    resp = client.get("/v1/briefing-events?agent_id=alice&limit=2&offset=1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["limit"] == 2
    assert body["offset"] == 1
    # Newest-first → recordatio-4 is at offset 0; offset=1, limit=2 → 3, 2.
    assert [e["request_id"] for e in body["events"]] == ["recordatio-3", "recordatio-2"]


# ---------------------------------------------------------------------------
# _last_user_text — strips Telos's inline <system-reminder> blocks so the
# timestamp doesn't bias retrieval or leak into the briefing UI.
# ---------------------------------------------------------------------------


def test_last_user_text_strips_system_reminder_string_content(server_mod) -> None:
    msg = (
        "<system-reminder>Current date and time: 2026-01-01 09:00</system-reminder> "
        "let's get inspiration from the reference designs"
    )
    out = server_mod._last_user_text([{"role": "user", "content": msg}])
    assert out == "let's get inspiration from the reference designs"


def test_last_user_text_strips_system_reminder_list_content(server_mod) -> None:
    msg = {"role": "user", "content": [
        {"type": "text", "text": "<system-reminder>tick</system-reminder>"},
        {"type": "text", "text": "actual question"},
    ]}
    out = server_mod._last_user_text([msg])
    assert out == "actual question"


def test_last_user_text_returns_none_when_only_reminder(server_mod) -> None:
    msg = "<system-reminder>only this</system-reminder>"
    assert server_mod._last_user_text([{"role": "user", "content": msg}]) is None


def test_last_user_text_passes_through_when_no_reminder(server_mod) -> None:
    msg = "what's the weather"
    assert server_mod._last_user_text([{"role": "user", "content": msg}]) == "what's the weather"


# ---------------------------------------------------------------------------
# Agent-runner hooks — memory scoping. The hooks must use the composite
# ``{user_id}/{slug}`` agent_id (the same value telos sends in ``x-agent-id``
# on the chat path) so the agent path shares the legacy memory scope. A bare
# slug would both orphan existing per-user memories (agent amnesia) and pool
# every user's memories into one shared scope (privacy bug).
# ---------------------------------------------------------------------------


def test_agent_briefing_hook_uses_composite_agent_id(
    server_mod, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    async def spy_build_system(
        existing_system,
        query,
        agent_id,
        *,
        backend=None,
        call_id=None,
        source_agent_id=None,
    ):
        captured["agent_id"] = agent_id
        captured["source_agent_id"] = source_agent_id
        return server_mod.prompting.SystemBuildResult(
            system="augmented", briefing_data=None, briefing_text="[BRIEFING]"
        )

    monkeypatch.setattr(server_mod, "_build_system", spy_build_system)

    briefing_text, _briefing = asyncio.run(
        server_mod._agent_briefing_hook("tommaso", "_maintenance", "hi", "base prompt")
    )

    assert captured["agent_id"] == "tommaso/_maintenance"
    # Direct user turn: mirrors the chat path's fallback of
    # x-recordatio-source-agent-id → agent_id.
    assert captured["source_agent_id"] == "tommaso/_maintenance"
    # The hook returns the bare briefing text (the runner relocates it
    # into the rendered prompt), not the combined system prompt.
    assert briefing_text == "[BRIEFING]"


def test_agent_briefing_hook_stamps_request_id(
    server_mod, monkeypatch: pytest.MonkeyPatch
) -> None:
    """telos keys the recalled-memories panel off ``request_id`` in the
    briefing payload (no chunk ids on the agent path), and hydrates from
    /v1/briefing-events/{request_id} — so the id must match the one the
    hook used to persist the briefing-events row."""
    captured: dict[str, Any] = {}

    async def spy_build_system(
        existing_system,
        query,
        agent_id,
        *,
        backend=None,
        call_id=None,
        source_agent_id=None,
    ):
        captured["call_id"] = call_id
        return server_mod.prompting.SystemBuildResult(
            system="augmented",
            briefing_data={"units": [], "entities": [{"label": "Tommaso"}]},
        )

    monkeypatch.setattr(server_mod, "_build_system", spy_build_system)

    _system, briefing = asyncio.run(
        server_mod._agent_briefing_hook("tommaso", "_maintenance", "hi", "base prompt")
    )

    assert briefing is not None
    assert briefing["request_id"] == captured["call_id"]
    assert briefing["request_id"].startswith("recordatio-agent-")
    # Original payload keys survive the stamp.
    assert briefing["entities"] == [{"label": "Tommaso"}]


def test_agent_extraction_hook_uses_composite_agent_id(
    server_mod, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    async def spy_extract_and_store(
        turn,
        api_key,
        agent_id,
        user_id=None,
        parent_request_id=None,
        source_agent_id=None,
        **_: Any,
    ):
        captured["agent_id"] = agent_id
        captured["user_id"] = user_id
        captured["source_agent_id"] = source_agent_id
        return {}

    scheduled: list[Any] = []
    monkeypatch.setattr(server_mod, "_extract_and_store", spy_extract_and_store)
    monkeypatch.setattr(server_mod, "_background", scheduled.append)

    transcript = {
        "messages": [{"role": "user", "content": "hi"}],
        "response_text": "hello",
    }

    async def run() -> None:
        await server_mod._agent_extraction_hook("tommaso", "_maintenance", transcript)
        for coro in scheduled:
            await coro

    asyncio.run(run())

    assert captured["agent_id"] == "tommaso/_maintenance"
    assert captured["user_id"] == "tommaso"
    # No lineage on the transcript → source defaults to the agent's own
    # composite id, matching the chat path.
    assert captured["source_agent_id"] == "tommaso/_maintenance"


def test_agent_extraction_hook_preserves_explicit_source_agent_id(
    server_mod, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    async def spy_extract_and_store(
        turn,
        api_key,
        agent_id,
        user_id=None,
        parent_request_id=None,
        source_agent_id=None,
        **_: Any,
    ):
        captured["agent_id"] = agent_id
        captured["source_agent_id"] = source_agent_id
        captured["parent_request_id"] = parent_request_id
        return {}

    scheduled: list[Any] = []
    monkeypatch.setattr(server_mod, "_extract_and_store", spy_extract_and_store)
    monkeypatch.setattr(server_mod, "_background", scheduled.append)

    transcript = {
        "messages": [{"role": "user", "content": "hi"}],
        "response_text": "hello",
        "source_agent_id": "tommaso/curia",
        "parent_request_id": "recordatio-abc123",
    }

    async def run() -> None:
        await server_mod._agent_extraction_hook("tommaso", "_maintenance", transcript)
        for coro in scheduled:
            await coro

    asyncio.run(run())

    assert captured["agent_id"] == "tommaso/_maintenance"
    # Caller-provided lineage passes through verbatim.
    assert captured["source_agent_id"] == "tommaso/curia"
    assert captured["parent_request_id"] == "recordatio-abc123"
