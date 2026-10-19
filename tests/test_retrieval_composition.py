"""Composed read visibility {S, user_scope(S)} at retrieval time (gate I-12).

Query recall reaches the user store through the existing keyword seam (one
FTS query, one BM25 pool); the composition NEVER crosses users or reads
another agent's private scope — tested in both directions.
"""

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tpasini.recordatio.entity import Entity
from tpasini.recordatio.identity import ensure_principal, visibility
from tpasini.recordatio.models import KnowledgeType, KnowledgeUnit, Relation
from tpasini.recordatio.proxy.candidate_source import KeywordSource
from tpasini.recordatio.proxy.retrieval import RetrievalResult, retrieve, retrieve_profile
from tpasini.recordatio.storage.base import SQLiteStore

NOW = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)
_logger = logging.getLogger("test-retrieval-composition")


def _result_ids(result: RetrievalResult) -> set[str]:
    return {u.id for u in result.units} | {e.id for e in result.entities} | {r.id for r in result.raw_relations}


async def _full_dump_ids(
    store: SQLiteStore, query: str, agent_scope: str, *, now: datetime, user_label: str
) -> set[str]:
    """Unbudgeted union of both retrieval sources (query recall + profile)."""
    query_res = await retrieve(query, agent_scope, store, _logger, user_label)
    profile_res = await retrieve_profile(agent_scope, store, user_label, _logger, now=now)
    return _result_ids(query_res) | _result_ids(profile_res)


def _scope_item_ids(store: SQLiteStore, agent_scope: str) -> set[str]:
    """Every row id in the scope's composed visibility ``{S, user_scope(S)}``
    (a raw full-scope scan, deliberately not part of the production Store API)."""
    scopes = visibility(agent_scope)
    marks = ",".join("?" for _ in scopes)
    ids: set[str] = set()
    for table in ("knowledge_units", "entities", "relations"):
        rows = store._conn.execute(f"SELECT id FROM {table} WHERE agent_id IN ({marks})", scopes)  # noqa: S608
        ids.update(r["id"] for r in rows)
    return ids


def _unit(store: SQLiteStore, scope: str, uid: str, content: str) -> None:
    store.add(
        KnowledgeUnit(id=uid, type=KnowledgeType.SEMANTIC, content=content, asserted_at=NOW,
                      valid_from=NOW, provenance="test"),
        scope,
    )


@pytest.fixture
def store(tmp_path: Path) -> SQLiteStore:
    store = SQLiteStore(tmp_path / "composed.db")
    tommaso, _ = ensure_principal(store, "tommaso", display_name="Tommaso", now=NOW)
    sam, _ = ensure_principal(store, "sam", now=NOW)
    # User-store material for tommaso.
    _unit(store, "tommaso/_user", "u-pension", "User's pension pot is with ProviderA index funds.")
    providera = Entity(id="e-providera", label="ProviderA", type="organization", asserted_at=NOW)
    store.add_entity(providera, "tommaso/_user")
    store.add_relation(
        Relation(id="r-invests", subject_id=tommaso.id, predicate="invests_with", object_id="e-providera",
                 asserted_at=NOW, valid_from=NOW, valid_until=None, provenance="test"),
        "tommaso/_user",
    )
    # Identically-worded distractor in ANOTHER USER's store: must never leak.
    _unit(store, "sam/_user", "u-pension-sam", "User's pension pot is with ProviderA index funds.")
    # Agent-private rows.
    _unit(store, "tommaso/scout", "u-scout", "User applied to the Acme staff engineer role.")
    _unit(store, "tommaso/helper", "u-helper", "User's boiler pension paperwork is in the hallway drawer.")
    return store


def _retrieve_ids(store: SQLiteStore, query: str, scope: str) -> set[str]:
    return _result_ids(asyncio.run(retrieve(query, scope, store, _logger, "User")))


def test_query_recall_reaches_user_store_through_keyword_seam(store: SQLiteStore) -> None:
    ids = _retrieve_ids(store, "which fund is my pension with?", "tommaso/scout")
    assert "u-pension" in ids  # user-store row surfaced under the agent scope


def test_composition_never_crosses_users(store: SQLiteStore) -> None:
    """Gate 6: sam's visibility cannot read tommaso/_user (and vice versa)."""
    sam_ids = _retrieve_ids(store, "which fund is my pension with?", "sam/scout")
    assert "u-pension" not in sam_ids
    assert "r-invests" not in sam_ids
    assert "u-pension-sam" in sam_ids  # her own user store IS visible
    tommaso_ids = _retrieve_ids(store, "which fund is my pension with?", "tommaso/scout")
    assert "u-pension-sam" not in tommaso_ids


def test_composition_never_reads_another_agents_private_scope(store: SQLiteStore) -> None:
    ids = _retrieve_ids(store, "where is the pension paperwork?", "tommaso/scout")
    assert "u-helper" not in ids  # helper's private row is not scout-visible


def test_keyword_source_single_pool_over_scope_set(store: SQLiteStore) -> None:
    source = KeywordSource(store)
    pairs = source.candidates("pension providera", ("tommaso/scout", "tommaso/_user"), 50)
    ids = {item_id for item_id, _score in pairs}
    assert "u-pension" in ids
    assert "u-pension-sam" not in ids


def test_principal_alias_seeds_from_query_name(store: SQLiteStore) -> None:
    result = asyncio.run(retrieve("What does Tommaso invest with?", "tommaso/scout", store, _logger, "User"))
    assert "r-invests" in {r.id for r in result.raw_relations}  # alias seeded the principal, traversal walked it


def test_profile_falls_back_to_legacy_label_path_pre_migration(tmp_path: Path) -> None:
    """v1-artifact comparability: with no principal anywhere, profile behaves
    exactly as before (label lookup in the agent scope)."""
    store = SQLiteStore(tmp_path / "legacy.db")
    user = Entity(id="e-user-legacy", label="Tommaso", type="person", asserted_at=NOW)
    store.add_entity(user, "tommaso/scout")
    springfield = Entity(id="e-springfield-legacy", label="Springfield", type="place", asserted_at=NOW)
    store.add_entity(springfield, "tommaso/scout")
    store.add_relation(
        Relation(id="r-legacy", subject_id="e-user-legacy", predicate="lives_in", object_id="e-springfield-legacy",
                 asserted_at=NOW, valid_from=NOW, valid_until=None, provenance="test"),
        "tommaso/scout",
    )
    profile = asyncio.run(retrieve_profile("tommaso/scout", store, "Tommaso", _logger, now=NOW))
    assert "e-user-legacy" in {e.id for e in profile.entities}
    assert "r-legacy" in {r.id for r in profile.raw_relations}


def test_context_window_scan_includes_user_scope(store: SQLiteStore) -> None:
    ids = _scope_item_ids(store, "tommaso/scout")
    assert {"u-pension", "u-scout", "r-invests"} <= ids
    assert "u-helper" not in ids
    assert "u-pension-sam" not in ids


def test_full_dump_strategy_sees_composed_visibility(store: SQLiteStore) -> None:
    ids = asyncio.run(
        _full_dump_ids(store, "which fund is my pension with?", "tommaso/scout", now=NOW, user_label="User")
    )
    assert "u-pension" in ids
    assert "u-pension-sam" not in ids
