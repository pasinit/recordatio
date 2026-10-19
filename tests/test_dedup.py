"""Tests for the mechanical dedup fixes in #57:

1. Cross-side merge in proxy.server._dedup_cross_side.
2. Partial UNIQUE index on active relations in storage.base.SQLiteStore.
3. Context/target split in extraction.extractor._render_input / _render_output.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tpasini.recordatio.entity import Entity
from tpasini.recordatio.extraction.contracts import (
    EntityMention,
    ExtractionPayload,
    RelationExtraction,
    SchemaProposal,
    SchemaProposalKind,
)
from tpasini.recordatio.extraction.extractor import (
    ConversationTurn,
    ExtractionResult,
    _render_input,
    _render_output,
)
from tpasini.recordatio.models import KnowledgeType, KnowledgeUnit, Relation
from tpasini.recordatio.proxy.server import _merge_cross_side, _norm_content
from tpasini.recordatio.storage.base import SQLiteStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit(content: str) -> KnowledgeUnit:
    now = datetime.now(UTC)
    return KnowledgeUnit(
        id="",
        type=KnowledgeType.SEMANTIC,
        content=content,
        asserted_at=now,
        valid_from=now,
        provenance="test",
    )


def _entity(label: str) -> Entity:
    return Entity(label=label, asserted_at=datetime.now(UTC))


def _make_relation(subject_id: str, predicate: str, object_id: str, at: datetime) -> Relation:
    return Relation(
        id="",
        subject_id=subject_id,
        predicate=predicate,
        object_id=object_id,
        asserted_at=at,
        valid_from=at,
        valid_until=None,
        provenance="test",
    )


@pytest.fixture()
def store(tmp_path: Path) -> SQLiteStore:
    return SQLiteStore(tmp_path / "test.db")


# ---------------------------------------------------------------------------
# Change 1 — cross-side dedup
# ---------------------------------------------------------------------------


def test_norm_content_lowercases_and_collapses_whitespace() -> None:
    assert _norm_content("  Alice   Loves\t\nPython ") == "alice loves python"


def test_merge_cross_side_drops_duplicate_unit() -> None:
    primary = ExtractionResult(units=[_unit("User lives in Sunnydale")])
    secondary = ExtractionResult(
        units=[
            _unit("user  lives in SUNNYDALE"),  # normalises to same as primary
            _unit("User works at Acme"),
        ]
    )

    merged = _merge_cross_side(primary, secondary)

    assert len(merged.units) == 2  # one from primary + one novel from secondary
    contents = [u.content for u in merged.units]
    assert "User lives in Sunnydale" in contents
    assert "User works at Acme" in contents
    # Inputs untouched (pure function).
    assert len(primary.units) == 1
    assert len(secondary.units) == 2


def test_merge_cross_side_drops_duplicate_entity_mention_label() -> None:
    primary = ExtractionResult(entity_mentions=[EntityMention(label="Sunnydale")])
    secondary = ExtractionResult(
        entity_mentions=[EntityMention(label="sunnydale"), EntityMention(label="Paris")]
    )

    merged = _merge_cross_side(primary, secondary)

    assert [mention.label for mention in merged.entity_mentions] == ["Sunnydale", "Paris"]
    assert [mention.label for mention in secondary.entity_mentions] == ["sunnydale", "Paris"]  # untouched


def test_merge_cross_side_drops_duplicate_relation_candidate_triple() -> None:
    primary = ExtractionResult(
        relation_candidates=[RelationExtraction(subject="User", predicate="lives_in", object="Sunnydale")]
    )
    secondary = ExtractionResult(
        relation_candidates=[
            RelationExtraction(subject="user", predicate="lives_in", object="SUNNYDALE"),  # dup
            RelationExtraction(subject="User", predicate="works_at", object="Acme"),  # novel
        ]
    )

    merged = _merge_cross_side(primary, secondary)

    assert len(merged.relation_candidates) == 2
    predicates = [relation.predicate for relation in merged.relation_candidates]
    assert predicates == ["lives_in", "works_at"]  # primary first, novel appended


def test_merge_cross_side_keeps_relation_candidates_with_different_predicates() -> None:
    primary = ExtractionResult(
        relation_candidates=[RelationExtraction(subject="User", predicate="lives_in", object="Sunnydale")]
    )
    secondary = ExtractionResult(
        relation_candidates=[RelationExtraction(subject="User", predicate="visited", object="Sunnydale")]
    )

    merged = _merge_cross_side(primary, secondary)

    assert len(merged.relation_candidates) == 2  # different predicates, not dups


def test_merge_cross_side_concatenates_when_primary_empty() -> None:
    primary = ExtractionResult()
    secondary = ExtractionResult(
        units=[_unit("foo")],
        entity_mentions=[EntityMention(label="bar")],
        relation_candidates=[RelationExtraction(subject="a", predicate="p", object="b")],
    )

    merged = _merge_cross_side(primary, secondary)

    assert len(merged.units) == 1
    assert [mention.label for mention in merged.entity_mentions] == ["bar"]
    assert len(merged.relation_candidates) == 1


def test_merge_cross_side_dedupes_schema_proposals_and_topics() -> None:
    primary = ExtractionResult(
        schema_proposals=[
            SchemaProposal(kind=SchemaProposalKind.RELATION_TYPE, name="hates", description="...")
        ],
        topics=["technology"],
    )
    secondary = ExtractionResult(
        schema_proposals=[
            SchemaProposal(kind=SchemaProposalKind.RELATION_TYPE, name="hates", description="..."),
            SchemaProposal(kind=SchemaProposalKind.RELATION_TYPE, name="loves", description="..."),
            SchemaProposal(kind=SchemaProposalKind.TOPIC, name="music", description="..."),
        ],
        topics=["technology", "people"],
    )

    merged = _merge_cross_side(primary, secondary)

    assert [proposal.name for proposal in merged.schema_proposals] == ["hates", "loves", "music"]
    assert merged.topics == ["technology", "people"]


def test_extraction_payload_rejects_context_dump_relation_endpoints() -> None:
    long_endpoint = (
        "Flat hunting workflow: search PortalA and PortalB, fetch full listing details, "
        "inspect photos, skip basement units, and send a daily recap with notes."
    )

    payload = ExtractionPayload.model_validate(
        {
            "relations": [
                {"subject": "User", "predicate": "lives_in", "object": "Sunnydale"},
                {"subject": long_endpoint, "predicate": "about", "object": "User"},
                {"subject": "User", "predicate": "has invalid spaces", "object": "Sunnydale"},
            ]
        }
    )

    assert [(relation.subject, relation.predicate, relation.object) for relation in payload.relations] == [
        ("User", "lives_in", "Sunnydale")
    ]


# ---------------------------------------------------------------------------
# Change 2 — partial UNIQUE on active relations
# ---------------------------------------------------------------------------


def test_add_relation_same_triple_twice_keeps_one_row(store: SQLiteStore) -> None:
    now = datetime.now(UTC)
    store.add_relation(_make_relation("s1", "p1", "o1", now), "alice")
    store.add_relation(_make_relation("s1", "p1", "o1", now), "alice")

    rows = store._conn.execute(
        "SELECT id FROM relations WHERE agent_id = ?", ("alice",)
    ).fetchall()
    assert len(rows) == 1


def test_add_relation_same_triple_different_agents_both_kept(store: SQLiteStore) -> None:
    now = datetime.now(UTC)
    store.add_relation(_make_relation("s1", "p1", "o1", now), "alice")
    store.add_relation(_make_relation("s1", "p1", "o1", now), "bob")

    rows = store._conn.execute("SELECT agent_id FROM relations").fetchall()
    agents = {r["agent_id"] for r in rows}
    assert agents == {"alice", "bob"}


def test_supersede_then_reinsert_same_triple_creates_active_row(store: SQLiteStore) -> None:
    """Functional-predicate move-and-back: X→Y, X→Z, X→Y. The third insert must
    succeed because the original (now-superseded) row is outside the partial index.
    """
    t0 = datetime.now(UTC)
    t1 = t0 + timedelta(seconds=10)
    t2 = t1 + timedelta(seconds=10)

    # t0: X located_in Y
    store.add_relation(_make_relation("X", "located_in", "Y", t0), "alice")

    # t1: supersede + insert X located_in Z
    store.supersede_relations("X", "located_in", t1, "alice")
    store.add_relation(_make_relation("X", "located_in", "Z", t1), "alice")

    # t2: supersede + insert X located_in Y (back to Y)
    store.supersede_relations("X", "located_in", t2, "alice")
    store.add_relation(_make_relation("X", "located_in", "Y", t2), "alice")

    rows = store._conn.execute(
        "SELECT object_id, valid_until FROM relations WHERE agent_id = ? ORDER BY asserted_at",
        ("alice",),
    ).fetchall()
    # Three rows: Y (superseded), Z (superseded), Y (active).
    assert len(rows) == 3
    active = [r for r in rows if r["valid_until"] is None]
    assert len(active) == 1
    assert active[0]["object_id"] == "Y"


def test_partial_index_allows_superseded_duplicate(store: SQLiteStore) -> None:
    """An active row plus a superseded row with the same triple is allowed.
    Verifies the unique index is partial (WHERE valid_until IS NULL), not full.
    """
    t0 = datetime.now(UTC)
    t1 = t0 + timedelta(seconds=10)

    store.add_relation(_make_relation("s1", "p1", "o1", t0), "alice")
    store.supersede_relations("s1", "p1", t1, "alice")
    # Now insert the same triple as an active row again.
    store.add_relation(_make_relation("s1", "p1", "o1", t1), "alice")

    rows = store._conn.execute(
        "SELECT valid_until FROM relations WHERE agent_id = ?", ("alice",)
    ).fetchall()
    assert len(rows) == 2
    assert sum(1 for r in rows if r["valid_until"] is None) == 1


def test_migration_dedups_active_duplicates(tmp_path: Path) -> None:
    """Seed a DB with duplicate active rows via raw SQL (bypassing add_relation),
    re-open via SQLiteStore, expect dedup to one active row per group.
    """
    db_path = tmp_path / "legacy.db"
    # Create a fresh store so the schema exists, then drop the partial index
    # to simulate a pre-migration DB and insert duplicates directly.
    store = SQLiteStore(db_path)
    store._conn.execute("DROP INDEX IF EXISTS idx_relations_unique_active")
    store._conn.commit()

    now = datetime.now(UTC)
    earliest = now - timedelta(seconds=30)
    middle = now - timedelta(seconds=20)
    latest = now - timedelta(seconds=10)

    def _raw_insert(rel_id: str, at: datetime, valid_until: datetime | None) -> None:
        store._conn.execute(
            """
            INSERT INTO relations (id, subject_id, predicate, object_id,
                                   asserted_at, valid_from, valid_until,
                                   provenance, agent_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rel_id, "subj", "pred", "obj",
                at.isoformat(), at.isoformat(),
                valid_until.isoformat() if valid_until else None,
                "test", "alice",
            ),
        )

    # Three duplicate ACTIVE rows for (alice, subj, pred, obj).
    _raw_insert("active-earliest", earliest, None)
    _raw_insert("active-middle", middle, None)
    _raw_insert("active-latest", latest, None)
    # Two superseded rows that should be left alone (history).
    _raw_insert("superseded-1", earliest, now)
    _raw_insert("superseded-2", middle, now)
    store._conn.commit()
    store._conn.close()

    # Re-open: migration runs in __init__.
    store2 = SQLiteStore(db_path)
    rows = store2._conn.execute(
        "SELECT id, valid_until FROM relations WHERE agent_id = ? ORDER BY asserted_at",
        ("alice",),
    ).fetchall()
    ids = [r["id"] for r in rows]
    # Earliest active kept; two later active duplicates dropped.
    assert "active-earliest" in ids
    assert "active-middle" not in ids
    assert "active-latest" not in ids
    # Superseded rows are intentional history — left untouched.
    assert "superseded-1" in ids
    assert "superseded-2" in ids


def test_migration_is_idempotent(tmp_path: Path) -> None:
    """Opening the store twice should not error or change state on the second open."""
    db_path = tmp_path / "idem.db"
    store = SQLiteStore(db_path)
    now = datetime.now(UTC)
    store.add_relation(_make_relation("s", "p", "o", now), "alice")
    store._conn.close()

    store2 = SQLiteStore(db_path)
    rows = store2._conn.execute("SELECT id FROM relations").fetchall()
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# Change 3 — context/target split in extractor rendering
# ---------------------------------------------------------------------------


def _msg(role: str, text: str) -> dict:
    return {"role": role, "content": text}


def test_render_input_emits_context_and_target_sections() -> None:
    turn = ConversationTurn(
        messages=[
            _msg("user", "I'm Alice"),
            _msg("assistant", "Hi Alice"),
            _msg("user", "Where do I live?"),
        ],
        response_text="",
    )
    rendered = _render_input(turn)

    assert "[context — for disambiguation only, do NOT extract from these]" in rendered
    assert "[target — extract knowledge from this section only]" in rendered
    # Older messages are in [context], the latest in [target].
    context_section, target_section = rendered.split("[target")
    assert "I'm Alice" in context_section
    assert "Hi Alice" in context_section
    assert "Where do I live?" in target_section
    assert "Where do I live?" not in context_section
    assert "I'm Alice" not in target_section


def test_render_input_single_message_renders_only_target() -> None:
    turn = ConversationTurn(messages=[_msg("user", "hello")], response_text="")
    rendered = _render_input(turn)

    assert "[context" not in rendered
    assert "[target" in rendered
    assert "hello" in rendered


def test_render_input_empty_messages_returns_empty_string() -> None:
    turn = ConversationTurn(messages=[], response_text="")
    assert _render_input(turn) == ""


def test_render_output_emits_response_as_target_and_messages_as_context() -> None:
    turn = ConversationTurn(
        messages=[
            _msg("user", "Hi"),
            _msg("assistant", "Hello!"),
            _msg("user", "Where do I live?"),
        ],
        response_text="You live in Sunnydale.",
    )
    rendered = _render_output(turn)

    assert "[context — for disambiguation only, do NOT extract from these]" in rendered
    assert "[target — extract knowledge from this section only]" in rendered
    context_section, target_section = rendered.split("[target")
    # Prior turn messages are in context.
    assert "Where do I live?" in context_section
    # The assistant final response is in target.
    assert "You live in Sunnydale." in target_section
    assert "[assistant:final]" in target_section


def test_render_output_no_prior_messages_renders_only_target() -> None:
    turn = ConversationTurn(messages=[], response_text="hello")
    rendered = _render_output(turn)

    assert "[context" not in rendered
    assert "[target" in rendered
    assert "hello" in rendered
