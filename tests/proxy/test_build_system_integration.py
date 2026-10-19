"""Integration tests: build_system → budgeter → two-block render.

Exercises the live briefing path with the real W1a budgeter and rendering,
feeding fake retrieve / retrieve_profile callables (the store I/O is W1a/W1b's
own concern and tested elsewhere). Asserts the two budgeted blocks, tag
omission, both-empty no-briefing, the SELECTED briefing_data panel payload, the
chat-path .system composition, and budget caps.
"""

import logging
from datetime import UTC, datetime

import pytest

from tpasini.recordatio.entity import Entity
from tpasini.recordatio.models import KnowledgeType, KnowledgeUnit, Relation
from tpasini.recordatio.proxy import prompting
from tpasini.recordatio.proxy.briefing_log import BriefingLog
from tpasini.recordatio.proxy.retrieval import RetrievalResult

_logger = logging.getLogger("test")


@pytest.fixture()
def briefing_log(tmp_path) -> BriefingLog:
    return BriefingLog(tmp_path / "briefing.db")


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


async def _build(*, query_result, profile_result, briefing_log, existing="BASE"):
    return await prompting.build_system(
        existing_system=existing,
        query="where do I work and live?",
        agent_id="alice",
        backend="anthropic",
        call_id="req-1",
        retrieve=_make_retrieve(query_result),
        retrieve_profile=_make_profile(profile_result),
        briefing_log=briefing_log,
        logger=_logger,
    )


@pytest.mark.asyncio
async def test_two_budgeted_blocks_within_budget(briefing_log) -> None:
    user = _entity("u", "Tommaso")
    springfield = _entity("l", "Springfield")
    globex = _entity("b", "Globex")
    profile = RetrievalResult(
        units=[_unit("pu", "Tommaso is searching for a flat")],
        entities=[user, springfield, globex],
        rendered_relations=[],
        raw_relations=[
            _relation("r1", user.id, "lives_in", springfield.id),
            _relation("r2", user.id, "works_at", globex.id),
        ],
        topic_hint=None,
    )
    query = RetrievalResult(
        units=[_unit("qu", "the user works at Globex in Springfield")],
        entities=[user, springfield, globex],
        rendered_relations=[],
        raw_relations=[_relation("r3", user.id, "works_at", globex.id)],
        topic_hint="work",
    )

    result = await _build(query_result=query, profile_result=profile, briefing_log=briefing_log)

    assert result.briefing_text is not None
    text = result.briefing_text
    # Two distinct blocks, profile first then recall.
    assert "<user_profile>" in text and "</user_profile>" in text
    assert "<relevant_memory>" in text and "</relevant_memory>" in text
    assert text.index("<user_profile>") < text.index("<relevant_memory>")
    # The model-facing memory_search note survives inside relevant_memory.
    assert "memory_search" in text
    assert "[MEMORY]" in text
    # Within the (default) budgets.
    from tpasini.recordatio.proxy.briefing_budget import PROFILE_CHAR_BUDGET, RECALL_CHAR_BUDGET

    profile_part = text[text.index("<user_profile>"):text.index("</user_profile>")]
    recall_part = text[text.index("<relevant_memory>"):text.index("</relevant_memory>")]
    assert len(profile_part) <= PROFILE_CHAR_BUDGET + 200  # + tag/framing overhead
    assert len(recall_part) <= RECALL_CHAR_BUDGET + 200


@pytest.mark.asyncio
async def test_empty_profile_omits_user_profile_block(briefing_log) -> None:
    user = _entity("u", "Tommaso")
    query = RetrievalResult(
        units=[_unit("qu", "the user prefers Python")],
        entities=[user],
        rendered_relations=[],
        raw_relations=[],
        topic_hint=None,
    )
    result = await _build(query_result=query, profile_result=_empty(), briefing_log=briefing_log)

    assert result.briefing_text is not None
    assert "<user_profile>" not in result.briefing_text
    assert "<relevant_memory>" in result.briefing_text


@pytest.mark.asyncio
async def test_empty_recall_omits_relevant_memory_block(briefing_log) -> None:
    user = _entity("u", "Tommaso")
    profile = RetrievalResult(
        units=[_unit("pu", "Tommaso lives in Springfield")],
        entities=[user],
        rendered_relations=[],
        raw_relations=[],
        topic_hint=None,
    )
    result = await _build(query_result=_empty(), profile_result=profile, briefing_log=briefing_log)

    assert result.briefing_text is not None
    assert "<user_profile>" in result.briefing_text
    assert "<relevant_memory>" not in result.briefing_text


@pytest.mark.asyncio
async def test_both_empty_yields_no_briefing(briefing_log) -> None:
    result = await _build(query_result=_empty(), profile_result=_empty(), briefing_log=briefing_log)

    assert result.briefing_text is None
    assert result.briefing_data is None
    # The chat path falls back to the bare existing system prompt.
    assert result.system == "BASE"


@pytest.mark.asyncio
async def test_briefing_data_reflects_selected_not_full_graph(briefing_log) -> None:
    """A distractor entity in retrieval that gets budgeted out must NOT appear
    in the panel payload — the panel shows what the agent actually saw."""
    from tpasini.recordatio.proxy.briefing_budget import RECALL_CHAR_BUDGET

    user = _entity("u", "Tommaso")
    globex = _entity("b", "Globex")
    # A relation between user and globex makes both selectable; the
    # distractor entity is made larger than the WHOLE recall budget so the
    # knapsack can never admit it — proving briefing_data is built from the
    # SELECTED set, not the full retrieval graph.
    huge_desc = "x" * (RECALL_CHAR_BUDGET + 1000)
    distractor = _entity("d", "Distractor", desc=huge_desc)
    query = RetrievalResult(
        units=[],
        entities=[user, globex, distractor],
        rendered_relations=[],
        raw_relations=[_relation("r", user.id, "works_at", globex.id)],
        topic_hint=None,
    )

    result = await _build(query_result=query, profile_result=_empty(), briefing_log=briefing_log)

    assert result.briefing_data is not None
    labels = {e["label"] for e in result.briefing_data["entities"]}
    # The over-budget distractor cannot fit; it must be absent from the panel.
    assert "Distractor" not in labels
    # And the panel reflects only selected items (a subset of the full graph).
    assert labels <= {"Tommaso", "Globex"}


@pytest.mark.asyncio
async def test_chat_path_system_composes_base_plus_briefing(briefing_log) -> None:
    user = _entity("u", "Tommaso")
    query = RetrievalResult(
        units=[_unit("qu", "the user prefers Python")],
        entities=[user],
        rendered_relations=[],
        raw_relations=[],
        topic_hint=None,
    )
    result = await _build(
        query_result=query, profile_result=_empty(), briefing_log=briefing_log, existing="STABLE_SYSTEM",
    )

    assert result.system is not None
    assert result.system.startswith("STABLE_SYSTEM")
    assert result.briefing_text in result.system


@pytest.mark.asyncio
async def test_briefing_data_dedups_union(briefing_log) -> None:
    """An entity selected in BOTH the profile and recall blocks appears once."""
    user = _entity("u", "Tommaso")
    springfield = _entity("l", "Springfield")
    rel = _relation("r", user.id, "lives_in", springfield.id)
    profile = RetrievalResult(
        units=[], entities=[user, springfield], rendered_relations=[],
        raw_relations=[rel], topic_hint=None,
    )
    query = RetrievalResult(
        units=[], entities=[user, springfield], rendered_relations=[],
        raw_relations=[rel], topic_hint=None,
    )
    result = await _build(query_result=query, profile_result=profile, briefing_log=briefing_log)

    assert result.briefing_data is not None
    labels = [e["label"] for e in result.briefing_data["entities"]]
    assert sorted(labels) == sorted(set(labels))  # no dupes
