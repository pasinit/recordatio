"""F2 fusion-wiring tests: retrieve() populates RetrievalResult.relevance with
BM25-only by default and RRF(bm25 ⊕ embedding) when the flag is on, and the
embedded relation surface matches the eval generator's recipe.

Offline: the bi-encoder is never loaded — embed_query / load_item_embeddings are
monkeypatched so CI does not need weights.
"""

import logging
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tpasini.recordatio.models import KnowledgeType, KnowledgeUnit, Relation
from tpasini.recordatio.proxy import embeddings
from tpasini.recordatio.proxy import retrieval as ret
from tpasini.recordatio.storage.base import SQLiteStore


@pytest.fixture
def store() -> SQLiteStore:
    db = Path(tempfile.mkdtemp()) / "m.db"
    s = SQLiteStore(db, ephemeral_fts=True)
    yield s
    s._conn.close()


def _now() -> datetime:
    return datetime(2026, 6, 15, tzinfo=UTC)


def _seed(store: SQLiteStore) -> None:
    for i in range(3):
        store.add(
            KnowledgeUnit(
                type=KnowledgeType.SEMANTIC,
                content=f"fact about widgets number {i}",
                asserted_at=_now(),
                valid_from=_now(),
                provenance="test",
                id=f"u{i}",
            ),
            "agent",
        )


def test_relevance_is_bm25_when_flag_off(store: SQLiteStore, monkeypatch):
    monkeypatch.setenv("RECORDATIO_F2_EMBEDDINGS", "0")
    relevance = ret._fuse_relevance(
        "widgets",
        "agent",
        store,
        units=[],
        entities=[],
        raw_relations=[],
        entity_by_id={},
        bm25_pairs=[("u0", 1.0)],
        keyword_relevance={"u0": 1.0},
        logger=logging.getLogger("t"),
    )
    assert relevance == {"u0": 1.0}


def test_relevance_falls_back_to_bm25_when_no_embeddings(store: SQLiteStore, monkeypatch):
    """Flag on but store has no embeddings → embedding list empty → BM25 dict."""
    monkeypatch.setenv("RECORDATIO_F2_EMBEDDINGS", "1")
    monkeypatch.setattr(embeddings, "embed_query", lambda q: [1.0] * embeddings.EMBED_DIM)
    monkeypatch.setattr(embeddings, "embed_documents", lambda s: [])
    u = KnowledgeUnit(
        type=KnowledgeType.SEMANTIC, content="x", asserted_at=_now(), valid_from=_now(), provenance="t", id="u0"
    )
    relevance = ret._fuse_relevance(
        "widgets",
        "agent",
        store,
        units=[u],
        entities=[],
        raw_relations=[],
        entity_by_id={},
        bm25_pairs=[("u0", 1.0)],
        keyword_relevance={"u0": 1.0},
        logger=logging.getLogger("t"),
    )
    assert relevance == {"u0": 1.0}


def test_relevance_fuses_when_embeddings_present(store: SQLiteStore, monkeypatch):
    """Flag on + stored embeddings → RRF union over bm25 and embedding ranks."""
    monkeypatch.setenv("RECORDATIO_F2_EMBEDDINGS", "1")
    qvec = [1.0, 0.0] + [0.0] * (embeddings.EMBED_DIM - 2)
    monkeypatch.setattr(embeddings, "embed_query", lambda q: qvec)
    monkeypatch.setattr(embeddings, "embed_documents", lambda s: [])
    # Store embeddings: u1 closest to query, u0 orthogonal.
    store.upsert_item_embedding(
        item_id="u0",
        agent_id="agent",
        kind="unit",
        surface_hash="h0",
        embedding=embeddings.pack_embedding([0.0, 1.0] + [0.0] * (embeddings.EMBED_DIM - 2)),
        model="m",
        asserted_at="t",
    )
    store.upsert_item_embedding(
        item_id="u1",
        agent_id="agent",
        kind="unit",
        surface_hash="h1",
        embedding=embeddings.pack_embedding(qvec),
        model="m",
        asserted_at="t",
    )
    u0 = KnowledgeUnit(
        type=KnowledgeType.SEMANTIC, content="a", asserted_at=_now(), valid_from=_now(), provenance="t", id="u0"
    )
    u1 = KnowledgeUnit(
        type=KnowledgeType.SEMANTIC, content="b", asserted_at=_now(), valid_from=_now(), provenance="t", id="u1"
    )
    # BM25 ranks u0 above u1; embedding ranks u1 above u0 — RRF should lift u1.
    relevance = ret._fuse_relevance(
        "widgets",
        "agent",
        store,
        units=[u0, u1],
        entities=[],
        raw_relations=[],
        entity_by_id={},
        bm25_pairs=[("u0", 2.0), ("u1", 1.0)],
        keyword_relevance={"u0": 1.0, "u1": 0.5},
        logger=logging.getLogger("t"),
    )
    # Both present, normalized to [0,1]; u1 gained from the embedding rank.
    assert set(relevance) == {"u0", "u1"}
    assert relevance["u1"] == pytest.approx(relevance["u0"])  # symmetric ranks → tie


def test_relation_embed_surface_matches_eval_recipe():
    """Relation surface == 'subject predicate object' with labels resolved from
    the label map, raw-id fallback for endpoints with no label."""
    rel = Relation(
        id="r1",
        subject_id="e1",
        predicate="works_at",
        object_id="e2",
        asserted_at=_now(),
        valid_from=_now(),
        valid_until=None,
        provenance="t",
    )
    # e1 resolves to label; e2 missing → raw id fallback.
    surface = ret._relation_embed_surface(rel, {"e1": "Acme"})
    assert surface == "Acme works_at e2"


def test_retrieve_populates_relevance_field(store: SQLiteStore, monkeypatch):
    """End-to-end through retrieve(): RetrievalResult.relevance is set (flag off
    → equals keyword_relevance)."""
    import asyncio

    monkeypatch.setenv("RECORDATIO_F2_EMBEDDINGS", "0")

    async def _no_wikidata(*a, **k):
        return []

    monkeypatch.setattr(ret, "wikidata_seed_entities", _no_wikidata)
    _seed(store)
    res = asyncio.run(ret.retrieve("widgets", "agent", store, logging.getLogger("t"), "Tommaso"))
    assert res.relevance == res.keyword_relevance
