"""C3: read-side retrieval_count wiring.

After a briefing is built, ``retrieval_count`` must be bumped for exactly the
items that ACTUALLY made it into the rendered prompt — the budgeter's final
SELECTED set, AFTER the knapsack and AFTER relation pruning (no-dangling +
top-K). Retrieval candidates dropped for budget, relations pruned as dangling
or beyond top-K, and anything in the pre-budget RetrievalResult but not in the
final BriefingSelection must NOT be bumped.

These tests drive the REAL W1a budgeter (via ``prompting.build_system`` with
fake retrieve callables so the drops are deterministic) and a REAL seeded
``SQLiteStore``, then schedule the bump through the server's ``_background``
mechanism and await it — proving the bump is off the critical path yet runs.
"""

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tpasini.recordatio.entity import Entity
from tpasini.recordatio.models import KnowledgeType, KnowledgeUnit, Relation
from tpasini.recordatio.proxy import prompting
from tpasini.recordatio.proxy.briefing_budget import RECALL_CHAR_BUDGET
from tpasini.recordatio.proxy.briefing_log import BriefingLog
from tpasini.recordatio.proxy.retrieval import RetrievalResult
from tpasini.recordatio.storage.base import SQLiteStore

_logger = logging.getLogger("test")
_AGENT = "alice/assistant"
_OTHER = "bob/assistant"


def _now() -> datetime:
    return datetime.now(UTC)


def _unit(uid: str, content: str) -> KnowledgeUnit:
    now = _now()
    return KnowledgeUnit(
        id=uid, type=KnowledgeType.SEMANTIC, content=content,
        asserted_at=now, valid_from=now, provenance="test",
    )


def _entity(eid: str, label: str, *, desc: str | None = None) -> Entity:
    return Entity(id=eid, label=label, description=desc, asserted_at=_now())


def _relation(rid: str, subj: str, pred: str, obj: str) -> Relation:
    now = _now()
    return Relation(
        id=rid, subject_id=subj, predicate=pred, object_id=obj,
        asserted_at=now, valid_from=now, valid_until=None, provenance="test",
    )


def _empty() -> RetrievalResult:
    return RetrievalResult(units=[], entities=[], rendered_relations=[], raw_relations=[], topic_hint=None)


def _make_retrieve(result: RetrievalResult):
    async def _retrieve(query, agent_id, source_agent_id=None):
        return result
    return _retrieve


def _make_profile(result: RetrievalResult):
    async def _retrieve_profile(agent_id, *, now=None):
        return result
    return _retrieve_profile


async def _build(*, query_result, profile_result, briefing_log, query="where do I work and live?"):
    return await prompting.build_system(
        existing_system="BASE",
        query=query,
        agent_id=_AGENT,
        backend="anthropic",
        call_id="req-1",
        retrieve=_make_retrieve(query_result),
        retrieve_profile=_make_profile(profile_result),
        briefing_log=briefing_log,
        logger=_logger,
    )


@pytest.fixture()
def briefing_log(tmp_path: Path) -> BriefingLog:
    return BriefingLog(tmp_path / "briefing.db")


@pytest.fixture()
def server_mod(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Import the server module wired to a fresh tmp store.

    Used for ``_bump_retrieval_counts`` + ``_background`` (the real scheduling
    machinery the chat / agent paths use).
    """
    monkeypatch.setenv("RECORDATIO_EXTRACTION_MODEL", "test")
    monkeypatch.setenv("AGENTMEM_DB", str(tmp_path / "memories.db"))
    from tpasini.recordatio.proxy import server as server_mod

    return server_mod


def _rc(store: SQLiteStore, table: str, id_: str) -> int:
    row = store._conn.execute(
        f"SELECT retrieval_count FROM {table} WHERE id = ?", (id_,)
    ).fetchone()
    return row["retrieval_count"] if row else -1


async def _await_background(server_mod) -> None:
    """Let scheduled background tasks run to completion."""
    pending = [t for t in server_mod._background_tasks if not t.done()]
    if pending:
        await asyncio.gather(*pending)


# ---------------------------------------------------------------------------
# Selected-id collection (real budgeter)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_selected_ids_are_only_the_rendered_items(briefing_log) -> None:
    """A distractor entity too large for the recall budget is DROPPED by the
    knapsack; a relation whose endpoints aren't both selected is pruned as
    dangling. Neither id appears in selected_ids."""
    user = _entity("u", "Tommaso")
    globex = _entity("b", "Globex")
    huge = "x" * (RECALL_CHAR_BUDGET + 1000)
    distractor = _entity("d", "Distractor", desc=huge)
    query = RetrievalResult(
        units=[],
        entities=[user, globex, distractor],
        rendered_relations=[],
        raw_relations=[
            _relation("r_keep", user.id, "works_at", globex.id),
            # endpoints: user + distractor; distractor is dropped → dangling.
            _relation("r_dangle", user.id, "knows", distractor.id),
        ],
        topic_hint=None,
    )

    result = await _build(query_result=query, profile_result=_empty(), briefing_log=briefing_log)

    sel = result.selected_ids
    assert sel is not None
    assert set(sel.entities) == {"u", "b"}  # distractor dropped
    assert "d" not in sel.entities
    # Only the non-dangling relation survives; relation ROW ids (not entity ids).
    assert sel.relations == ["r_keep"]
    assert "r_dangle" not in sel.relations


@pytest.mark.asyncio
async def test_selected_ids_dedup_across_blocks(briefing_log) -> None:
    """An item chosen in BOTH profile and recall is listed once."""
    user = _entity("u", "Tommaso")
    springfield = _entity("l", "Springfield")
    rel = _relation("r", user.id, "lives_in", springfield.id)
    both = RetrievalResult(
        units=[_unit("fact", "Tommaso lives in Springfield")],
        entities=[user, springfield], rendered_relations=[], raw_relations=[rel], topic_hint=None,
    )
    result = await _build(query_result=both, profile_result=both, briefing_log=briefing_log)

    sel = result.selected_ids
    assert sel is not None
    assert sel.units.count("fact") == 1
    assert sel.entities.count("u") == 1
    assert sel.entities.count("l") == 1


@pytest.mark.asyncio
async def test_no_briefing_yields_no_selected_ids(briefing_log) -> None:
    result = await _build(query_result=_empty(), profile_result=_empty(), briefing_log=briefing_log)
    assert result.selected_ids is None


# ---------------------------------------------------------------------------
# Background bump against a real seeded store
# ---------------------------------------------------------------------------


def _seed(store: SQLiteStore, agent_id: str) -> None:
    store.add(_unit("fact", "Tommaso lives in Springfield"), agent_id)
    store.add(_unit("dropped_fact", "irrelevant trivia"), agent_id)
    for e in (_entity("u", "Tommaso"), _entity("b", "Globex"), _entity("l", "Springfield")):
        store.add_entity(e, agent_id)
    store.add_relation(_relation("r_keep", "u", "works_at", "b"), agent_id)
    store.add_relation(_relation("r_dangle", "u", "knows", "l"), agent_id)


@pytest.mark.asyncio
async def test_background_bump_only_rendered_items(briefing_log, server_mod, tmp_path) -> None:
    store = SQLiteStore(tmp_path / "seeded.db")
    server_mod.store = store
    _seed(store, _AGENT)

    user = _entity("u", "Tommaso")
    globex = _entity("b", "Globex")
    huge = "x" * (RECALL_CHAR_BUDGET + 1000)
    # Reuse "l" id for an oversized distractor so it's dropped from the knapsack
    # and its relation r_dangle becomes dangling.
    distractor = _entity("l", "Springfield", desc=huge)
    query = RetrievalResult(
        units=[_unit("fact", "Tommaso lives in Springfield")],
        entities=[user, globex, distractor],
        rendered_relations=[],
        raw_relations=[
            _relation("r_keep", "u", "works_at", "b"),
            _relation("r_dangle", "u", "knows", "l"),
        ],
        topic_hint=None,
    )
    result = await _build(query_result=query, profile_result=_empty(), briefing_log=briefing_log)
    sel = result.selected_ids
    assert sel is not None
    assert "l" not in sel.entities  # oversized distractor dropped
    assert "r_dangle" not in sel.relations

    # Schedule the bump in the BACKGROUND (the same mechanism the chat/agent
    # paths use) and await it.
    server_mod._background(server_mod._bump_retrieval_counts(sel, _AGENT))
    await _await_background(server_mod)

    # Rendered items bumped exactly once.
    assert _rc(store, "knowledge_units", "fact") == 1
    assert _rc(store, "entities", "u") == 1
    assert _rc(store, "entities", "b") == 1
    assert _rc(store, "relations", "r_keep") == 1
    # Dropped candidate / pruned relation / never-retrieved item stay at 0.
    assert _rc(store, "entities", "l") == 0
    assert _rc(store, "relations", "r_dangle") == 0
    assert _rc(store, "knowledge_units", "dropped_fact") == 0


@pytest.mark.asyncio
async def test_bump_scoped_by_agent_id(briefing_log, server_mod, tmp_path) -> None:
    store = SQLiteStore(tmp_path / "scoped.db")
    server_mod.store = store
    _seed(store, _AGENT)
    # _OTHER owns a distinct-id unit (PK is id alone, so ids can't collide
    # across agents). The bump is scoped to _AGENT only.
    store.add(_unit("fact_other", "Bob lives in Paris"), _OTHER)

    user = _entity("u", "Tommaso")
    query = RetrievalResult(
        units=[_unit("fact", "Tommaso lives in Springfield")],
        entities=[user], rendered_relations=[], raw_relations=[], topic_hint=None,
    )
    result = await _build(query_result=query, profile_result=_empty(), briefing_log=briefing_log)
    server_mod._background(server_mod._bump_retrieval_counts(result.selected_ids, _AGENT))
    await _await_background(server_mod)

    # _AGENT's selected row moved; _OTHER's row is untouched.
    assert _rc(store, "knowledge_units", "fact") == 1
    assert _rc(store, "knowledge_units", "fact_other") == 0

    # Bumping the SAME id but scoped to the WRONG agent moves nothing — proving
    # the agent_id scoping on the increment itself.
    await server_mod._bump_retrieval_counts(
        prompting.SelectedIds(units=["fact"], entities=[], relations=[]), _OTHER,
    )
    assert _rc(store, "knowledge_units", "fact") == 1


@pytest.mark.asyncio
async def test_bump_noop_when_no_selection(server_mod, tmp_path) -> None:
    store = SQLiteStore(tmp_path / "noop.db")
    server_mod.store = store
    _seed(store, _AGENT)
    # None selected_ids → no-op, no exception.
    await server_mod._bump_retrieval_counts(None, _AGENT)
    assert _rc(store, "knowledge_units", "fact") == 0
