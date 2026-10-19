"""Write-routing + E-RULE recognition through the real persistence path.

Implementer gates I-8 (recognition), I-11 (routing table + §2.4 edge cases),
I-14 (cross-scope edges render under the composed read) from
IDENTITY_DESIGN.md §9.1. Every test drives ``_persist_side`` — the same code
path production extraction uses.
"""

import asyncio
import logging
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tpasini.recordatio.entity import Entity
from tpasini.recordatio.extraction.contracts import (
    EntityMention as ExtractionEntityMention,
)
from tpasini.recordatio.extraction.contracts import (
    EvidenceReference,
    RelationExtraction,
)
from tpasini.recordatio.extraction.extractor import ExtractionResult
from tpasini.recordatio.identity import (
    ensure_principal,
    get_principal,
    mirror_entity_id,
    user_scope,
)
from tpasini.recordatio.models import KnowledgeType, KnowledgeUnit
from tpasini.recordatio.proxy.cost_log import CostLog
from tpasini.recordatio.proxy.extraction_pipeline import ExtractionPipelineContext, _persist_side
from tpasini.recordatio.proxy.retrieval import render_relations_from, retrieve_profile, traverse
from tpasini.recordatio.storage.base import SQLiteStore

NOW = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)
AGENT = "tommaso/scout"
USCOPE = user_scope(AGENT)
_logger = logging.getLogger("test-identity-routing")


@pytest.fixture
def ctx(tmp_path: Path) -> ExtractionPipelineContext:
    store = SQLiteStore(tmp_path / "routing.db")
    return ExtractionPipelineContext(
        store=store,
        cost_log=CostLog(tmp_path / "cost.db"),
        extraction_sem=asyncio.Semaphore(1),
        unreported_extractions=defaultdict(list),
        unreported_extractions_lock=asyncio.Lock(),
        extraction_bare_model="test",
        extraction_backend="test",
        extraction_base_url=None,
    )


def _persist(ctx: ExtractionPipelineContext, result: ExtractionResult, agent_id: str = AGENT) -> None:
    asyncio.run(_persist_side(ctx, result, agent_id))


def _unit(content: str, type_: KnowledgeType = KnowledgeType.SEMANTIC) -> KnowledgeUnit:
    return KnowledgeUnit(
        type=type_, content=content, asserted_at=NOW, valid_from=NOW, provenance="extraction"
    )


def _user_mention(label: str = "User", type_: str | None = "person") -> ExtractionEntityMention:
    return ExtractionEntityMention(
        id=f"entity:{label.lower().replace(' ', '-')}",
        label=label,
        type=type_,
        confidence=0.95,
        evidence=[EvidenceReference(source="target", quote=label)],
    )


def _scope_of_units(ctx: ExtractionPipelineContext) -> dict[str, str]:
    return {
        r["content"]: r["agent_id"]
        for r in ctx.store._conn.execute("SELECT content, agent_id FROM knowledge_units")
    }


# --- recognition (I-8) ----------------------------------------------------------


def test_user_mention_binds_via_user_scope_canonical_key(ctx: ExtractionPipelineContext) -> None:
    _persist(ctx, ExtractionResult(entity_mentions=[_user_mention("User")]))
    principal = get_principal(ctx.store, USCOPE)
    assert principal is not None
    # No agent-scope user entity is ever created (I-8: extraction on a
    # migrated/fresh store creates no agent-scope user referent).
    assert ctx.store.get_entity_by_label("User", AGENT) is None


def test_typed_and_substring_user_mentions_do_not_bind(ctx: ExtractionPipelineContext) -> None:
    _persist(ctx, ExtractionResult(entity_mentions=[
        _user_mention("user", type_="concept"),       # type guard
        _user_mention("user account", type_="person"),  # full-label equality only
        _user_mention("users", type_="person"),
    ]))
    principal = get_principal(ctx.store, USCOPE)
    assert principal is not None  # ensure-on-write created it, nothing bound to it
    assert ctx.store.get_entity_by_label("user", AGENT) is not None
    assert ctx.store.get_entity_by_label("user account", AGENT) is not None
    assert ctx.store.get_entity_by_label("users", AGENT) is not None
    user_scope_entities = ctx.store.list_entities(USCOPE)
    assert [e.id for e in user_scope_entities] == [principal.id]


def test_name_mention_binds_through_standing_alias(ctx: ExtractionPipelineContext) -> None:
    ensure_principal(ctx.store, "tommaso", display_name="Tommaso", now=NOW)
    _persist(ctx, ExtractionResult(entity_mentions=[_user_mention("Tommaso")]))
    principal = get_principal(ctx.store, USCOPE)
    assert principal is not None
    assert ctx.store.get_entity_by_label("Tommaso", AGENT) is None  # bound, not created


def test_third_party_namesake_not_swallowed(ctx: ExtractionPipelineContext) -> None:
    """The sam/coach case: a genuine third-party 'Tommaso' exists in
    the asserting scope, so has_name must not alias it onto the principal and
    later 'Tommaso' mentions must resolve to the third party."""
    agent = "sam/coach"
    ensure_principal(ctx.store, "sam", now=NOW)
    store_tp = Entity(label="Tommaso", type="person", asserted_at=NOW)
    ctx.store.add_entity(store_tp, agent)

    _persist(
        ctx,
        ExtractionResult(
            entity_mentions=[_user_mention("User"), _user_mention("Tommaso")],
            relation_candidates=[
                RelationExtraction(
                    subject="User", subject_id="entity:user", predicate="has_name",
                    object="Tommaso", object_id="entity:tommaso",
                    confidence=0.95,
                    evidence=[EvidenceReference(source="target", quote="my name is Tommaso")],
                )
            ],
        ),
        agent,
    )
    principal = get_principal(ctx.store, "sam/_user")
    assert principal is not None
    assert ctx.store.list_entity_aliases(principal.id, "sam/_user") == []  # guard blocked
    # The third-party entity is intact and still owns its label.
    assert ctx.store.get_entity_by_label("Tommaso", agent) is not None


# --- routing table (I-11) --------------------------------------------------------


def test_principal_relation_routes_user_level_with_mirror(ctx: ExtractionPipelineContext) -> None:
    """§2.4.1 scout case: `User -[applied_to]-> Acme role` routes user-level;
    the accompanying episodic narrative stays agent-level."""
    _persist(
        ctx,
        ExtractionResult(
            units=[_unit("User applied to the Acme role on Tuesday and felt nervous.", KnowledgeType.EPISODIC)],
            entity_mentions=[_user_mention("User"), _user_mention("Acme role", type_="concept")],
            relation_candidates=[
                RelationExtraction(
                    subject="User", subject_id="entity:user", predicate="applied_to",
                    object="Acme role", object_id="entity:acme-role",
                    confidence=0.95,
                    evidence=[EvidenceReference(source="target", quote="applied to Acme")],
                )
            ],
        ),
    )
    principal = get_principal(ctx.store, USCOPE)
    assert principal is not None
    mirror = mirror_entity_id(USCOPE, "concept", "Acme role")
    rows = ctx.store._conn.execute(
        "SELECT agent_id, subject_id, object_id FROM relations WHERE predicate = 'applied_to'"
    ).fetchall()
    assert [(r["agent_id"], r["subject_id"], r["object_id"]) for r in rows] == [(USCOPE, principal.id, mirror)]
    # The narrative is episodic: the agent that lived it owns it.
    assert _scope_of_units(ctx) == {"User applied to the Acme role on Tuesday and felt nervous.": AGENT}
    # The original endpoint entity stays agent-scoped (nodes are referents).
    assert ctx.store.get_entity_by_label("Acme role", AGENT) is not None


def test_semantic_principal_unit_routes_user_level_whole(ctx: ExtractionPipelineContext) -> None:
    """§2.4.3: units are atomic — a mixed semantic unit routes whole."""
    _persist(
        ctx,
        ExtractionResult(
            units=[_unit("User prefers index funds; the scout agent tracks three boards.")],
            entity_mentions=[_user_mention("User")],
        ),
    )
    assert _scope_of_units(ctx) == {
        "User prefers index funds; the scout agent tracks three boards.": USCOPE
    }


def test_misclassified_type_stays_agent_scoped(ctx: ExtractionPipelineContext) -> None:
    """§2.4.4: routing trusts the extractor's type label."""
    _persist(
        ctx,
        ExtractionResult(
            units=[_unit("User lives in Springfield X1.", KnowledgeType.EPISODIC)],
            entity_mentions=[_user_mention("User")],
        ),
    )
    assert _scope_of_units(ctx) == {"User lives in Springfield X1.": AGENT}


def test_third_party_facts_stay_agent_scoped(ctx: ExtractionPipelineContext) -> None:
    """§2.4.2: `User -[married_to]-> Sam` is user-level with Sam
    mirrored; `Sam -[works_at]-> Acme` stays in the scope that learned it."""
    _persist(
        ctx,
        ExtractionResult(
            entity_mentions=[
                _user_mention("User"),
                _user_mention("Sam"),
                _user_mention("Acme", type_="organization"),
            ],
            relation_candidates=[
                RelationExtraction(
                    subject="User", subject_id="entity:user", predicate="married_to",
                    object="Sam", object_id="entity:sam",
                    confidence=0.95, evidence=[EvidenceReference(source="target", quote="married to Sam")],
                ),
                RelationExtraction(
                    subject="Sam", subject_id="entity:sam", predicate="works_at",
                    object="Acme", object_id="entity:acme",
                    confidence=0.95, evidence=[EvidenceReference(source="target", quote="Sam works at Acme")],
                ),
            ],
        ),
    )
    principal = get_principal(ctx.store, USCOPE)
    sam = ctx.store.get_entity_by_label("Sam", AGENT)
    assert principal is not None and sam is not None
    married = ctx.store._conn.execute(
        "SELECT agent_id, object_id FROM relations WHERE predicate = 'married_to'"
    ).fetchone()
    assert married["agent_id"] == USCOPE
    assert married["object_id"] == mirror_entity_id(USCOPE, "person", "Sam")
    works = ctx.store._conn.execute(
        "SELECT agent_id, subject_id FROM relations WHERE predicate = 'works_at'"
    ).fetchone()
    assert works["agent_id"] == AGENT
    assert works["subject_id"] == sam.id


def test_cross_agent_supersession_on_live_writes(ctx: ExtractionPipelineContext) -> None:
    """§3.3 live regime: any agent's functional assertion closes ALL active
    rows in the user scope — including rows another agent asserted."""

    def lives_in(city: str) -> ExtractionResult:
        return ExtractionResult(
            entity_mentions=[_user_mention("User"), _user_mention(city, type_="place")],
            relation_candidates=[
                RelationExtraction(
                    subject="User", subject_id="entity:user", predicate="lives_in",
                    object=city, object_id=f"entity:{city.lower()}",
                    confidence=0.95, evidence=[EvidenceReference(source="target", quote=f"lives in {city}")],
                )
            ],
        )

    _persist(ctx, lives_in("Springfield"), "tommaso/scout")
    _persist(ctx, lives_in("Paris"), "tommaso/helper")

    rows = ctx.store._conn.execute(
        "SELECT object_id, valid_until, source_agent_id FROM relations "
        "WHERE predicate = 'lives_in' AND agent_id = ? ORDER BY asserted_at",
        (USCOPE,),
    ).fetchall()
    assert len(rows) == 2
    springfield, paris = rows
    assert springfield["valid_until"] is not None  # closed by helper's assertion
    assert springfield["source_agent_id"] == "tommaso/scout"
    assert paris["valid_until"] is None
    assert paris["source_agent_id"] == "tommaso/helper"


def test_cross_agent_reassertion_reinforces_one_user_store_row(ctx: ExtractionPipelineContext) -> None:
    """§3.1: extraction_count on a user-level row counts assertions across all
    of the user's agents."""
    result = ExtractionResult(
        units=[_unit("User prefers window seats.")],
        entity_mentions=[_user_mention("User")],
    )
    _persist(ctx, result, "tommaso/scout")
    result2 = ExtractionResult(
        units=[_unit("User prefers window seats.")],
        entity_mentions=[_user_mention("User")],
    )
    _persist(ctx, result2, "tommaso/trippy")
    rows = ctx.store._conn.execute(
        "SELECT agent_id, extraction_count, source_agent_id FROM knowledge_units WHERE valid_until IS NULL"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["agent_id"] == USCOPE
    assert rows[0]["extraction_count"] == 2
    assert rows[0]["source_agent_id"] == "tommaso/scout"  # earliest asserter's attribution


# --- cross-scope edges render (I-14) ----------------------------------------------


def test_episodic_unit_cross_scope_mention_renders_under_composed_read(
    ctx: ExtractionPipelineContext,
) -> None:
    """An agent-scope episodic unit's `mentions` relation may point at the
    user-scope principal id — the composed read resolves it (§2.2)."""
    _persist(
        ctx,
        ExtractionResult(
            units=[_unit("User arrived early for the interview.", KnowledgeType.EPISODIC)],
            entity_mentions=[_user_mention("User")],
        ),
    )
    unit_row = ctx.store._conn.execute("SELECT id, agent_id FROM knowledge_units").fetchone()
    assert unit_row["agent_id"] == AGENT
    principal = get_principal(ctx.store, USCOPE)
    link = ctx.store._conn.execute(
        "SELECT agent_id, object_id FROM relations WHERE predicate = 'mentions'"
    ).fetchone()
    assert link is not None
    assert link["agent_id"] == AGENT  # the link follows the unit's home
    assert link["object_id"] == principal.id  # legal cross-scope edge

    units = [ctx.store._row_to_unit(r) for r in ctx.store._conn.execute(
        "SELECT id, type, content, asserted_at, valid_from, valid_until, provenance, "
        "extraction_count, retrieval_count FROM knowledge_units"
    )]
    traversal = traverse([], units, AGENT, ctx.store, _logger)
    assert principal.id in traversal.entity_by_id  # endpoint resolved across scopes
    rendered = render_relations_from(traversal.relations, traversal.entity_by_id, units, AGENT, ctx.store)
    assert ("User arrived early for the interview.", "mentions", "User") in {
        (r.subject, r.predicate, r.object) for r in rendered
    }


def test_profile_reads_user_store(ctx: ExtractionPipelineContext) -> None:
    _persist(
        ctx,
        ExtractionResult(
            units=[_unit("User prefers index funds for retirement.")],
            entity_mentions=[_user_mention("User"), _user_mention("Globex", type_="organization")],
            relation_candidates=[
                RelationExtraction(
                    subject="User", subject_id="entity:user", predicate="works_at",
                    object="Globex", object_id="entity:globex",
                    confidence=0.95, evidence=[EvidenceReference(source="target", quote="works at Globex")],
                )
            ],
        ),
    )
    profile = asyncio.run(retrieve_profile(AGENT, ctx.store, "User", _logger, now=NOW))
    principal = get_principal(ctx.store, USCOPE)
    assert principal.id in {e.id for e in profile.entities}
    assert {u.content for u in profile.units} == {"User prefers index funds for retirement."}
    assert {(r.subject, r.predicate, r.object) for r in profile.rendered_relations} >= {
        ("User", "works_at", "Globex")
    }
