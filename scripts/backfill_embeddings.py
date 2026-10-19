"""Backfill F2 candidate embeddings for every active item in a store.

One-time (and idempotent) population of the ``item_embeddings`` cache the
production embedding source reads (``proxy/embeddings.py`` /
``storage.base.item_embeddings``).  Embeds the searchable surface (RR4: store
text only) of every active unit and every entity with the pinned bi-encoder,
keyed by ``(item_id, agent_id, surface_hash)``; a row whose stored
``surface_hash`` already matches is skipped (idempotent — a re-run after a
partial run, or after only some surfaces changed, re-embeds only the delta).

Relations are NOT backfilled: their searchable surface (rendered triple with
endpoint labels) is render-time only and is embedded on the fly at query time
(few per turn).

Default is a DRY RUN that reports counts without loading the model or writing;
``--apply`` loads the model and writes.  Schema is brought current first (old
stores lack the table), reusing the SQLiteStore-open pattern.

    PYTHONPATH=src python scripts/backfill_embeddings.py --db PATH            # dry run
    PYTHONPATH=src python scripts/backfill_embeddings.py --db PATH --apply
"""

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from tpasini.recordatio.proxy import embeddings  # noqa: E402
from tpasini.recordatio.storage.base import SQLiteStore  # noqa: E402

DEFAULT_DB = Path.home() / ".agentmem" / "memories.db"

# Encode in batches so a large store doesn't build one giant input list.
_BATCH = 256


def _active_unit_ids(store: SQLiteStore) -> list[str]:
    rows = store._conn.execute(
        "SELECT id FROM knowledge_units WHERE valid_until IS NULL ORDER BY rowid"
    ).fetchall()
    return [r["id"] for r in rows]


def _entity_ids(store: SQLiteStore) -> list[str]:
    rows = store._conn.execute("SELECT id FROM entities ORDER BY rowid").fetchall()
    return [r["id"] for r in rows]


def _relation_ids(store: SQLiteStore) -> list[str]:
    """Active relation ids (superseded rows excluded — they never surface)."""
    rows = store._conn.execute(
        "SELECT id FROM relations WHERE valid_until IS NULL ORDER BY rowid"
    ).fetchall()
    return [r["id"] for r in rows]


def _surfaces_to_embed(
    store: SQLiteStore,
    unit_ids: list[str],
    entity_ids: list[str],
    relation_ids: list[str] | None = None,
) -> tuple[list[tuple[str, str, str, str]], int]:
    """Return ``[(item_id, agent_id, kind, surface)]`` needing (re)embedding and
    the count skipped because their stored ``surface_hash`` already matches."""
    items = list(store.item_surfaces_for_embedding(unit_ids, entity_ids))
    if relation_ids:
        items += list(store.relation_surfaces_for_embedding(relation_ids))
    to_embed: list[tuple[str, str, str, str]] = []
    skipped = 0
    for item_id, agent_id, kind, surface in items:
        if not surface:
            continue
        if store.get_item_embedding_hash(item_id, agent_id) == embeddings.surface_hash(surface):
            skipped += 1
            continue
        to_embed.append((item_id, agent_id, kind, surface))
    return to_embed, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill F2 candidate embeddings.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="memories.db to backfill")
    parser.add_argument("--apply", action="store_true", help="load model + write (default: dry run)")
    args = parser.parse_args()

    db: Path = args.db.expanduser()
    if not db.exists():
        print(f"ABORT: {db} not found", file=sys.stderr)
        return 2

    # Brings the schema current (creates item_embeddings on old stores).
    # ephemeral_emb=False PERSISTS embeddings into the <db>.emb sidecar (the
    # whole point of the backfill); ephemeral_fts=True avoids rebuilding/locking
    # the live FTS sidecar while the proxy keeps serving.
    store = SQLiteStore(db, ephemeral_fts=True, ephemeral_emb=False)
    try:
        unit_ids = _active_unit_ids(store)
        entity_ids = _entity_ids(store)
        relation_ids = _relation_ids(store)
        print(f"Store: {db}")
        print(f"  active units     : {len(unit_ids)}")
        print(f"  entities         : {len(entity_ids)}")
        print(f"  active relations : {len(relation_ids)}")

        to_embed, skipped = _surfaces_to_embed(store, unit_ids, entity_ids, relation_ids)
        n_units = sum(1 for _i, _a, k, _s in to_embed if k == "unit")
        n_entities = sum(1 for _i, _a, k, _s in to_embed if k == "entity")
        n_relations = sum(1 for _i, _a, k, _s in to_embed if k == "relation")
        print(f"  already current (skipped): {skipped}")
        print(f"  to embed: {len(to_embed)} (units={n_units}, entities={n_entities}, relations={n_relations})")

        if not args.apply:
            print("\n[DRY RUN] no model loaded, nothing written. Re-run with --apply.")
            return 0

        if not to_embed:
            print("\nNothing to embed — store is already current.")
            return 0

        print(f"\nLoading bi-encoder {embeddings._MODEL_NAME} ...")
        t0 = time.perf_counter()
        if embeddings.get_model() is None:
            print("ABORT: bi-encoder unavailable (sentence-transformers/weights missing)", file=sys.stderr)
            return 1
        print(f"  loaded in {time.perf_counter() - t0:.1f}s")

        now_iso = datetime.now(UTC).isoformat()
        written = 0
        t1 = time.perf_counter()
        for start in range(0, len(to_embed), _BATCH):
            batch = to_embed[start : start + _BATCH]
            vecs = embeddings.embed_documents([s for _i, _a, _k, s in batch])
            if vecs is None:
                print("ABORT: model returned no embeddings", file=sys.stderr)
                return 1
            for (item_id, agent_id, kind, surface), vec in zip(batch, vecs, strict=True):
                store.upsert_item_embedding(
                    item_id=item_id,
                    agent_id=agent_id,
                    kind=kind,
                    surface_hash=embeddings.surface_hash(surface),
                    embedding=embeddings.pack_embedding(vec),
                    model=embeddings._MODEL_NAME,
                    asserted_at=now_iso,
                )
                written += 1
            done = min(start + _BATCH, len(to_embed))
            print(f"  embedded {done}/{len(to_embed)} "
                  f"({(time.perf_counter() - t1):.1f}s elapsed)", flush=True)

        print(f"\nDone: wrote {written} embedding rows in {time.perf_counter() - t1:.1f}s.")
        return 0
    finally:
        store._conn.close()


if __name__ == "__main__":
    sys.exit(main())
