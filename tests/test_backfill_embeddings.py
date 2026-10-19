"""Backfill-script tests (scripts/backfill_embeddings.py): idempotency and the
surface_hash skip — offline (the model is monkeypatched).
"""

import importlib.util
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tpasini.recordatio.models import KnowledgeType, KnowledgeUnit
from tpasini.recordatio.proxy import embeddings
from tpasini.recordatio.storage.base import SQLiteStore

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "backfill_embeddings.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("backfill_embeddings", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _now() -> datetime:
    return datetime(2026, 6, 15, tzinfo=UTC)


@pytest.fixture
def store_with_units() -> SQLiteStore:
    db = Path(tempfile.mkdtemp()) / "m.db"
    s = SQLiteStore(db, ephemeral_fts=True)
    for i in range(5):
        s.add(
            KnowledgeUnit(
                type=KnowledgeType.SEMANTIC,
                content=f"fact {i}",
                asserted_at=_now(),
                valid_from=_now(),
                provenance="t",
                id=f"u{i}",
            ),
            "agent",
        )
    yield s
    s._conn.close()


def test_surfaces_to_embed_lists_active_units(store_with_units: SQLiteStore):
    mod = _load_module()
    unit_ids = mod._active_unit_ids(store_with_units)
    entity_ids = mod._entity_ids(store_with_units)
    to_embed, skipped = mod._surfaces_to_embed(store_with_units, unit_ids, entity_ids)
    assert skipped == 0
    assert {iid for iid, _a, k, _s in to_embed if k == "unit"} == {f"u{i}" for i in range(5)}


def test_backfill_idempotent_skips_current(store_with_units: SQLiteStore, monkeypatch):
    """After embedding once, a second pass skips every matching surface_hash."""
    monkeypatch.setattr(
        embeddings,
        "embed_documents",
        lambda surfaces: [[0.01 * i] * embeddings.EMBED_DIM for i in range(len(surfaces))],
    )
    mod = _load_module()
    unit_ids = mod._active_unit_ids(store_with_units)
    entity_ids = mod._entity_ids(store_with_units)

    # First pass: write all.
    to_embed, skipped = mod._surfaces_to_embed(store_with_units, unit_ids, entity_ids)
    assert skipped == 0
    now_iso = datetime.now(UTC).isoformat()
    for item_id, agent_id, kind, surface in to_embed:
        vec = embeddings.embed_documents([surface])[0]
        store_with_units.upsert_item_embedding(
            item_id=item_id,
            agent_id=agent_id,
            kind=kind,
            surface_hash=embeddings.surface_hash(surface),
            embedding=embeddings.pack_embedding(vec),
            model="m",
            asserted_at=now_iso,
        )

    # Second pass: everything is current → all skipped, nothing to embed.
    to_embed2, skipped2 = mod._surfaces_to_embed(store_with_units, unit_ids, entity_ids)
    assert to_embed2 == []
    assert skipped2 == len(unit_ids)


def test_backfill_reembeds_changed_surface(store_with_units: SQLiteStore, monkeypatch):
    monkeypatch.setattr(
        embeddings,
        "embed_documents",
        lambda surfaces: [[0.0] * embeddings.EMBED_DIM for _ in surfaces],
    )
    mod = _load_module()
    # Embed u0 with a stale hash (simulating an old surface).
    store_with_units.upsert_item_embedding(
        item_id="u0",
        agent_id="agent",
        kind="unit",
        surface_hash="STALE",
        embedding=embeddings.pack_embedding([0.0] * embeddings.EMBED_DIM),
        model="m",
        asserted_at="t",
    )
    unit_ids = mod._active_unit_ids(store_with_units)
    to_embed, _skipped = mod._surfaces_to_embed(store_with_units, unit_ids, [])
    # u0's current surface ("fact 0") != STALE → it must be queued for re-embed.
    assert any(iid == "u0" for iid, _a, _k, _s in to_embed)


def test_emb_sidecar_persists_across_reopen(tmp_path):
    """Regression: the backfill must PERSIST embeddings to <db>.emb.

    The backfill opens with ephemeral_fts=True (avoid FTS lock) but
    ephemeral_emb=False so the embedding sidecar is a real file, not :memory:.
    Earlier the two were coupled, so the backfill wrote 30k vectors to memory
    and lost them on exit.
    """
    db = tmp_path / "s.db"
    store = SQLiteStore(db, ephemeral_fts=True, ephemeral_emb=False)
    store.upsert_item_embedding(
        item_id="u1", agent_id="a/x", kind="unit", surface_hash="h1",
        embedding=b"\x00\x00\x80?", model="bge-small", asserted_at=datetime.now(UTC).isoformat(),
    )
    store._conn.close()
    assert (tmp_path / "s.db.emb").exists(), "embedding sidecar file must be written"

    reopened = SQLiteStore(db, ephemeral_fts=True, ephemeral_emb=False)
    assert reopened.get_item_embedding_hash("u1", "a/x") == "h1"
    reopened._conn.close()


def test_emb_sidecar_ephemeral_when_coupled(tmp_path):
    """Default (ephemeral_emb follows ephemeral_fts) leaves NO <db>.emb file —
    the frozen-artifact guarantee the backfill fix must not break."""
    db = tmp_path / "s.db"
    store = SQLiteStore(db, ephemeral_fts=True)  # ephemeral_emb defaults to True here
    store.upsert_item_embedding(
        item_id="u1", agent_id="a/x", kind="unit", surface_hash="h1",
        embedding=b"\x00\x00\x80?", model="bge-small", asserted_at=datetime.now(UTC).isoformat(),
    )
    store._conn.close()
    assert not (tmp_path / "s.db.emb").exists(), "coupled-ephemeral open must leave no sidecar file"
