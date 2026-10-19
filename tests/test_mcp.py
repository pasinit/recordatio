"""Tests for the MCP server (proxy/mcp_server.py).

Two tools are exercised end-to-end via direct invocation:

- ``recollect`` — seeds the store, calls the tool, asserts the unit appears in
    the typed response; also asserts the missing-header guard.
- ``extract`` — mocks ``Pontifex.stream`` (same pattern as test_proxy.py),
  calls the tool in sync mode, asserts the response shape AND that the
  store/cost_log received writes. Background mode is verified separately by
  draining ``_background_tasks``.
"""

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tpasini.recordatio.models import KnowledgeType, KnowledgeUnit


@pytest.fixture()
def server_mod(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Fresh-store fixture mirroring tests/test_proxy.py."""
    monkeypatch.setenv("RECORDATIO_EXTRACTION_MODEL", "anthropic/claude-haiku-4-5-20251001")
    monkeypatch.setenv("AGENTMEM_DB", str(tmp_path / "memories.db"))
    from tpasini.recordatio.proxy import server as server_mod
    server_mod._unreported_extractions.clear()
    return server_mod


def _fake_ctx(headers: dict[str, str] | None = None) -> Any:
    """Build a minimal Context-shaped object the MCP tools accept.

    The tools only touch ``ctx.request_context.request.headers.get(...)``, so
    a SimpleNamespace chain is enough. Header keys are lowercased to match
    the lookup keys (``"x-agent-id"``, ``"x-recordatio-api-key"``, ``"x-user-id"``,
    ``"x-recordatio-source-agent-id"``).
    """
    hdrs = {k.lower(): v for k, v in (headers or {}).items()}
    request = SimpleNamespace(headers=hdrs)
    return SimpleNamespace(request_context=SimpleNamespace(request=request))


def _seed_unit(server_mod, agent_id: str, content: str) -> KnowledgeUnit:
    now = datetime.now(UTC)
    unit = KnowledgeUnit(
        type=KnowledgeType.SEMANTIC,
        content=content,
        asserted_at=now,
        valid_from=now,
        provenance="seed",
    )
    server_mod.store.add(unit, agent_id)
    return unit


# ---------------------------------------------------------------------------
# recollect
# ---------------------------------------------------------------------------


async def test_recollect_returns_seeded_unit(server_mod) -> None:
    agent_id = "test-agent"
    _seed_unit(server_mod, agent_id, "AAPL Q3 2024 revenue was $94.9B")

    tool = server_mod.mcp._tool_manager._tools["recollect"].fn
    resp = await tool("AAPL", _fake_ctx({"X-Agent-Id": agent_id}))

    assert any("AAPL" in u.content for u in resp.units), resp


async def test_recollect_requires_agent_id_header(server_mod) -> None:
    tool = server_mod.mcp._tool_manager._tools["recollect"].fn
    with pytest.raises(ValueError, match="X-Agent-Id"):
        await tool("anything", _fake_ctx({}))


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------


_EXTRACTOR_PAYLOAD = {
    "units": [{
        "type": "semantic",
        "content": "AAPL Q3 2024 revenue was $94.9B",
    }],
    "entity_mentions": [{"label": "AAPL", "type": "organization"}],
    "relations": [],
    "topics": [],
}


class _FakePontifex:
    """Returns one chunk with the extractor's expected JSON payload."""

    _chunks: list[dict] = []

    def __init__(self, *_: Any, model: str = "claude-haiku-4-5-20251001", **__: Any) -> None:
        self.model = model

    async def stream(self, *_: Any, **__: Any) -> AsyncIterator[dict]:
        for chunk in self._chunks:
            yield chunk


def _patch_extractor_pontifex(monkeypatch: pytest.MonkeyPatch, payload: dict) -> None:
    chunks = [
        {"choices": [{"delta": {"content": json.dumps(payload)}}]},
        {"usage": {"prompt_tokens": 100, "completion_tokens": 50}},
    ]
    _FakePontifex._chunks = chunks
    from tpasini.recordatio.extraction import extractor
    monkeypatch.setattr(extractor, "Pontifex", _FakePontifex)


async def test_extract_sync_persists_and_returns_summary(
    server_mod, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_extractor_pontifex(monkeypatch, _EXTRACTOR_PAYLOAD)

    agent_id = "test-agent"
    tool = server_mod.mcp._tool_manager._tools["extract"].fn
    resp = await tool(
        [{"role": "user", "content": "How much revenue did AAPL report in Q3 2024?"}],
        "AAPL reported $94.9B in revenue in Q3 2024.",
        _fake_ctx({"X-Agent-Id": agent_id}),
    )

    assert resp.persisted is True
    assert resp.queued is False
    assert resp.input is not None
    assert resp.output is not None
    assert any("AAPL" in u.content for u in resp.input.units + resp.output.units)

    from tpasini.recordatio.models import RetrievalContext
    stored = server_mod.store.search(RetrievalContext(query="AAPL"), agent_id)
    assert any("AAPL" in u.content for u in stored), "extract did not persist units to the store"


async def test_extract_sync_records_source_agent_header(
    server_mod: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_extractor_pontifex(monkeypatch, _EXTRACTOR_PAYLOAD)

    tool = server_mod.mcp._tool_manager._tools["extract"].fn
    resp = await tool(
        [{"role": "user", "content": "How much revenue did AAPL report in Q3 2024?"}],
        "AAPL reported $94.9B in revenue in Q3 2024.",
        _fake_ctx({
            "X-Agent-Id": "mcp-source-agent",
            "X-Recordatio-Source-Agent-Id": "mcp-source-agent/finance",
        }),
    )

    assert resp.persisted is True
    rows = server_mod.store._conn.execute(
        "SELECT DISTINCT source_agent_id FROM knowledge_units WHERE agent_id = ?",
        ("mcp-source-agent",),
    ).fetchall()
    assert {row["source_agent_id"] for row in rows} == {"mcp-source-agent/finance"}


async def test_extract_background_returns_queued_and_persists_later(
    server_mod, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_extractor_pontifex(monkeypatch, _EXTRACTOR_PAYLOAD)

    agent_id = "bg-agent"
    tool = server_mod.mcp._tool_manager._tools["extract"].fn
    resp = await tool(
        [{"role": "user", "content": "AAPL revenue?"}],
        "$94.9B in Q3 2024.",
        _fake_ctx({"X-Agent-Id": agent_id}),
        background=True,
    )

    assert resp.persisted is False
    assert resp.queued is True
    assert resp.input is None and resp.output is None

    # Drain pending background tasks the proxy spawned.
    pending = list(server_mod._background_tasks)
    assert pending, "background mode should have scheduled a task"
    for task in pending:
        await task

    from tpasini.recordatio.models import RetrievalContext
    stored = server_mod.store.search(RetrievalContext(query="AAPL"), agent_id)
    assert any("AAPL" in u.content for u in stored), "background extract did not persist"


async def test_extract_requires_agent_id_header(server_mod) -> None:
    tool = server_mod.mcp._tool_manager._tools["extract"].fn
    with pytest.raises(ValueError, match="X-Agent-Id"):
        await tool([{"role": "user", "content": "hi"}], "hello", _fake_ctx({}))
