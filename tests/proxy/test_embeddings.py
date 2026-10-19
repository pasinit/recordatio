"""Unit tests for the F2 production embedding source (proxy/embeddings.py) and
its storage cache (storage.base item_embeddings).

Properties, not implementation, and ALL offline — the bi-encoder is never
loaded: tests inject a fake ``load_stored`` / monkeypatch the model so CI does
not need the weights. Covered:
  - storage upsert + surface_hash invalidation (no-op vs re-embed)
  - surface construction parity with the budgeter / eval generator (RR4)
  - query-source cosine ranking over the candidate pool
  - graceful no-embedding fallback (item absent from embedding list)
  - the F2 enable flag gates behaviour
  - backfill idempotency (re-run skips matching surface_hash)
"""

import tempfile
from pathlib import Path

import pytest

from tpasini.recordatio.proxy import embeddings
from tpasini.recordatio.storage.base import SQLiteStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store() -> SQLiteStore:
    db = Path(tempfile.mkdtemp()) / "m.db"
    s = SQLiteStore(db, ephemeral_fts=True)
    yield s
    s._conn.close()


def _vec(seed: float) -> list[float]:
    """A deterministic EMBED_DIM vector (not normalized — fine for storage tests)."""
    return [(seed + i) % 7 / 7.0 for i in range(embeddings.EMBED_DIM)]


# ---------------------------------------------------------------------------
# Surface construction — RR4 parity with the budgeter / eval generator.
# ---------------------------------------------------------------------------


def test_unit_surface_is_content():
    assert embeddings.unit_surface("hello world") == "hello world"
    assert embeddings.unit_surface(None) == ""


def test_entity_surface_is_label_space_description():
    assert embeddings.entity_surface("Tommaso", "a person") == "Tommaso a person"
    assert embeddings.entity_surface("Tommaso", None) == "Tommaso"
    assert embeddings.entity_surface("Tommaso", "") == "Tommaso"


def test_relation_surface_is_triple():
    assert embeddings.relation_surface("A", "works_at", "B") == "A works_at B"


def test_surface_hash_stable_and_distinct():
    h1 = embeddings.surface_hash("hello")
    assert h1 == embeddings.surface_hash("hello")
    assert h1 != embeddings.surface_hash("hello ")  # whitespace is significant
    assert h1 != embeddings.surface_hash("world")


# ---------------------------------------------------------------------------
# BLOB round-trip.
# ---------------------------------------------------------------------------


def test_pack_unpack_round_trip():
    vec = _vec(0.3)
    out = embeddings.unpack_embedding(embeddings.pack_embedding(vec))
    assert len(out) == embeddings.EMBED_DIM
    for a, b in zip(vec, out, strict=True):
        assert a == pytest.approx(b, abs=1e-6)


# ---------------------------------------------------------------------------
# Storage upsert + surface_hash invalidation.
# ---------------------------------------------------------------------------


def test_upsert_and_load(store: SQLiteStore):
    h = embeddings.surface_hash("s1")
    store.upsert_item_embedding(
        item_id="u1",
        agent_id="a",
        kind="unit",
        surface_hash=h,
        embedding=embeddings.pack_embedding(_vec(1.0)),
        model="m",
        asserted_at="t",
    )
    assert store.get_item_embedding_hash("u1", "a") == h
    loaded = store.load_item_embeddings(["u1"], ("a",))
    assert "u1" in loaded
    assert len(embeddings.unpack_embedding(loaded["u1"])) == embeddings.EMBED_DIM


def test_upsert_overwrites_on_surface_change(store: SQLiteStore):
    store.upsert_item_embedding(
        item_id="u1",
        agent_id="a",
        kind="unit",
        surface_hash=embeddings.surface_hash("old"),
        embedding=embeddings.pack_embedding(_vec(1.0)),
        model="m",
        asserted_at="t1",
    )
    store.upsert_item_embedding(
        item_id="u1",
        agent_id="a",
        kind="unit",
        surface_hash=embeddings.surface_hash("new"),
        embedding=embeddings.pack_embedding(_vec(2.0)),
        model="m",
        asserted_at="t2",
    )
    # Single row per (item_id, agent_id): the new surface_hash wins.
    assert store.get_item_embedding_hash("u1", "a") == embeddings.surface_hash("new")
    rows = store._conn.execute("SELECT COUNT(*) c FROM item_embeddings WHERE item_id='u1' AND agent_id='a'").fetchone()
    assert rows["c"] == 1


def test_load_absent_item_omitted(store: SQLiteStore):
    """An item with no stored embedding is simply absent (graceful)."""
    assert store.load_item_embeddings(["missing"], ("a",)) == {}


def test_load_scopes_filter(store: SQLiteStore):
    store.upsert_item_embedding(
        item_id="u1",
        agent_id="scopeA",
        kind="unit",
        surface_hash="h",
        embedding=embeddings.pack_embedding(_vec(1.0)),
        model="m",
        asserted_at="t",
    )
    assert store.load_item_embeddings(["u1"], ("scopeB",)) == {}
    assert "u1" in store.load_item_embeddings(["u1"], ("scopeA", "scopeB"))


def test_migration_idempotent_on_reopen():
    """Re-opening the same DB keeps the embedding cache (the `.emb` sidecar
    persists across opens).  Non-ephemeral so the sidecar is a file, not :memory:."""
    db = Path(tempfile.mkdtemp()) / "persist.db"
    store = SQLiteStore(db)  # non-ephemeral → `<db>.emb` sidecar file
    store.upsert_item_embedding(
        item_id="u1",
        agent_id="a",
        kind="unit",
        surface_hash="h",
        embedding=embeddings.pack_embedding(_vec(1.0)),
        model="m",
        asserted_at="t",
    )
    store._conn.close()
    reopened = SQLiteStore(db)
    try:
        assert reopened.get_item_embedding_hash("u1", "a") == "h"
    finally:
        reopened._conn.close()


def test_canonical_db_not_mutated_by_embedding_cache():
    """The embedding cache lives in the `<db>.emb` sidecar — opening and writing
    embeddings must NOT change the canonical DB bytes (the frozen-DB sha gate)."""
    import hashlib

    db = Path(tempfile.mkdtemp()) / "canon.db"
    SQLiteStore(db)._conn.close()
    # Drop sidecars so we hash a pristine canonical file.
    for ext in (".emb", ".fts"):
        p = Path(str(db) + ext)
        if p.exists():
            p.unlink()
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    s = SQLiteStore(db)
    s.upsert_item_embedding(
        item_id="x", agent_id="a", kind="unit", surface_hash="h",
        embedding=embeddings.pack_embedding(_vec(1.0)), model="m", asserted_at="t",
    )
    s._conn.close()
    after = hashlib.sha256(db.read_bytes()).hexdigest()
    assert before == after  # canonical DB untouched; cache went to the sidecar


def test_migration_on_table_less_store_degrades_gracefully():
    """An old store (no `.emb` sidecar) gets the cache table on open and returns
    empty loads until backfilled — the BM25 fallback path."""
    db = Path(tempfile.mkdtemp()) / "legacy.db"
    # Build a store, then delete the `.emb` sidecar to simulate a pre-F2 store.
    SQLiteStore(db)._conn.close()
    sidecar = Path(str(db) + ".emb")
    if sidecar.exists():
        sidecar.unlink()
    # Re-open: the _ensure migration recreates the sidecar table idempotently.
    s = SQLiteStore(db)
    try:
        assert s.load_item_embeddings(["anything"], ("a",)) == {}  # empty cache → graceful
    finally:
        s._conn.close()


# ---------------------------------------------------------------------------
# Query-source cosine + graceful fallback (no model loaded — inject vectors).
# ---------------------------------------------------------------------------


def _monkeypatch_query_embed(monkeypatch, qvec: list[float]):
    monkeypatch.setattr(embeddings, "embed_query", lambda q: qvec)
    # No relations embedded in these tests; doc-embed not exercised here.
    monkeypatch.setattr(embeddings, "embed_documents", lambda s: [])


def test_embedding_pairs_ranks_by_cosine(monkeypatch):
    # Query close to vec_a, far from vec_b.
    q = [1.0, 0.0] + [0.0] * (embeddings.EMBED_DIM - 2)
    a = [1.0, 0.0] + [0.0] * (embeddings.EMBED_DIM - 2)
    b = [0.0, 1.0] + [0.0] * (embeddings.EMBED_DIM - 2)
    _monkeypatch_query_embed(monkeypatch, q)

    def load_stored(item_ids):
        return {"a": embeddings.pack_embedding(a), "b": embeddings.pack_embedding(b)}

    pairs = embeddings.embedding_pairs(
        "query",
        unit_ids=["a", "b"],
        entity_ids=[],
        relation_surfaces=[],
        load_stored=load_stored,
    )
    # Sorted cosine-desc: a (cos 1) before b (cos 0).
    assert [iid for iid, _ in pairs] == ["a", "b"]
    assert pairs[0][1] == pytest.approx(1.0)
    assert pairs[1][1] == pytest.approx(0.0)


def test_embedding_pairs_omits_unembedded_items(monkeypatch):
    """A candidate with no stored embedding contributes nothing (RRF degrades
    it to BM25) — graceful no-embedding fallback."""
    q = [1.0] + [0.0] * (embeddings.EMBED_DIM - 1)
    _monkeypatch_query_embed(monkeypatch, q)

    def load_stored(item_ids):
        return {"a": embeddings.pack_embedding(q)}  # only 'a' embedded; 'b' absent

    pairs = embeddings.embedding_pairs(
        "query",
        unit_ids=["a", "b"],
        entity_ids=[],
        relation_surfaces=[],
        load_stored=load_stored,
    )
    assert [iid for iid, _ in pairs] == ["a"]


def test_embedding_pairs_empty_when_no_model(monkeypatch):
    monkeypatch.setattr(embeddings, "embed_query", lambda q: None)
    pairs = embeddings.embedding_pairs(
        "query",
        unit_ids=["a"],
        entity_ids=[],
        relation_surfaces=[],
        load_stored=lambda ids: {},
    )
    assert pairs == []


# ---------------------------------------------------------------------------
# Enable flag.
# ---------------------------------------------------------------------------


def test_enabled_default_off(monkeypatch):
    monkeypatch.delenv("RECORDATIO_F2_EMBEDDINGS", raising=False)
    assert embeddings.enabled() is False


@pytest.mark.parametrize("val,expected", [("1", True), ("true", True), ("0", False), ("", False)])
def test_enabled_flag_values(monkeypatch, val, expected):
    monkeypatch.setenv("RECORDATIO_F2_EMBEDDINGS", val)
    assert embeddings.enabled() is expected


# ---------------------------------------------------------------------------
# embed_and_store: surface_hash guard + flag gating (model monkeypatched).
# ---------------------------------------------------------------------------


def test_embed_and_store_skips_when_disabled(store: SQLiteStore, monkeypatch):
    monkeypatch.setenv("RECORDATIO_F2_EMBEDDINGS", "0")
    n = embeddings.embed_and_store(store, [("u1", "a", "unit", "hi")], now_iso="t")
    assert n == 0


def test_embed_and_store_writes_and_guards(store: SQLiteStore, monkeypatch):
    monkeypatch.setenv("RECORDATIO_F2_EMBEDDINGS", "1")
    monkeypatch.setattr(embeddings, "get_model", lambda: object())
    monkeypatch.setattr(embeddings, "embed_documents", lambda surfaces: [_vec(float(i)) for i in range(len(surfaces))])

    n1 = embeddings.embed_and_store(store, [("u1", "a", "unit", "hello")], now_iso="t1")
    assert n1 == 1
    assert store.get_item_embedding_hash("u1", "a") == embeddings.surface_hash("hello")

    # Re-run with the SAME surface → no-op (surface_hash guard).
    n2 = embeddings.embed_and_store(store, [("u1", "a", "unit", "hello")], now_iso="t2")
    assert n2 == 0

    # Changed surface → re-embed.
    n3 = embeddings.embed_and_store(store, [("u1", "a", "unit", "changed")], now_iso="t3")
    assert n3 == 1
    assert store.get_item_embedding_hash("u1", "a") == embeddings.surface_hash("changed")


def test_embed_and_store_skips_empty_surface(store: SQLiteStore, monkeypatch):
    monkeypatch.setenv("RECORDATIO_F2_EMBEDDINGS", "1")
    monkeypatch.setattr(embeddings, "get_model", lambda: object())
    monkeypatch.setattr(embeddings, "embed_documents", lambda surfaces: [_vec(0.0) for _ in surfaces])
    n = embeddings.embed_and_store(store, [("u1", "a", "unit", "")], now_iso="t")
    assert n == 0
