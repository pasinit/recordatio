"""Tests for the entity backfill/audit tooling."""

from datetime import UTC, datetime
from pathlib import Path

from tpasini.recordatio.backfill import apply_existing_schema_backfill, audit_db
from tpasini.recordatio.entity import Entity
from tpasini.recordatio.models import KnowledgeType, KnowledgeUnit, Relation
from tpasini.recordatio.storage.base import SQLiteStore


def _unit(content: str, unit_id: str) -> KnowledgeUnit:
    now = datetime.now(UTC)
    return KnowledgeUnit(
        id=unit_id,
        type=KnowledgeType.SEMANTIC,
        content=content,
        asserted_at=now,
        valid_from=now,
        provenance="test",
    )


def _entity(label: str, entity_id: str) -> Entity:
    return Entity(id=entity_id, label=label, asserted_at=datetime.now(UTC))


def _relation(subject_id: str, predicate: str, object_id: str, relation_id: str) -> Relation:
    now = datetime.now(UTC)
    return Relation(
        id=relation_id,
        subject_id=subject_id,
        predicate=predicate,
        object_id=object_id,
        asserted_at=now,
        valid_from=now,
        valid_until=None,
        provenance="test",
    )


def test_audit_clusters_existing_entities_identity_aliases_and_repeated_text_candidates(tmp_path: Path) -> None:
    db_path = tmp_path / "memories.db"
    store = SQLiteStore(db_path)
    store.add_entity(_entity("User", "ent-user"), "agent-a")
    store.add(_unit("User's name is Tommaso.", "unit-name"), "agent-a")
    store.add(_unit("Studio Alpha has a south-facing balcony.", "unit-studio-1"), "agent-a")
    store.add(_unit("Studio Alpha is close to the station.", "unit-studio-2"), "agent-a")
    store._conn.close()

    report = audit_db(db_path, "agent-a", generated_at=datetime(2026, 1, 1, tzinfo=UTC))

    proposals = {proposal.canonical_label: proposal for proposal in report.proposed_entities}
    assert "User" in proposals
    assert any(alias.label == "Tommaso" for alias in proposals["User"].aliases)

    assert "Studio Alpha" in proposals
    assert proposals["Studio Alpha"].confidence == 0.55
    assert proposals["Studio Alpha"].source_unit_ids == ["unit-studio-1", "unit-studio-2"]


def test_audit_reports_functional_attribute_conflicts(tmp_path: Path) -> None:
    db_path = tmp_path / "memories.db"
    store = SQLiteStore(db_path)
    store.add_entity(_entity("User", "ent-user"), "agent-a")
    store.add_entity(_entity("Springfield", "ent-springfield"), "agent-a")
    store.add_entity(_entity("Sunnydale", "ent-sunnydale"), "agent-a")
    store.add_relation(_relation("ent-user", "lives_in", "ent-springfield", "rel-springfield"), "agent-a")
    store.add_relation(_relation("ent-user", "lives_in", "ent-sunnydale", "rel-sunnydale"), "agent-a")
    store._conn.close()

    report = audit_db(db_path, "agent-a", generated_at=datetime(2026, 1, 1, tzinfo=UTC))

    assert any(
        conflict.kind == "functional_attribute_conflict"
        and "multiple active values" in conflict.message
        and conflict.evidence == ["Springfield", "Sunnydale"]
        for conflict in report.conflicts
    )


def test_dry_run_is_read_only(tmp_path: Path) -> None:
    db_path = tmp_path / "memories.db"
    store = SQLiteStore(db_path)
    store.add_entity(_entity("User", "ent-user"), "agent-a")
    store.add(_unit("User likes Python.", "unit-user"), "agent-a")
    store._conn.close()
    before = db_path.read_bytes()

    audit_db(db_path, "agent-a", generated_at=datetime(2026, 1, 1, tzinfo=UTC))

    assert db_path.read_bytes() == before


def test_apply_existing_schema_backfill_is_idempotent_and_records_audit(tmp_path: Path) -> None:
    db_path = tmp_path / "memories.db"
    store = SQLiteStore(db_path)
    store.add_entity(_entity("User", "ent-user"), "agent-a")
    store.add(_unit("User likes Python.", "unit-user"), "agent-a")
    store._conn.close()

    report = audit_db(db_path, "agent-a", generated_at=datetime(2026, 1, 1, tzinfo=UTC))
    first = apply_existing_schema_backfill(
        db_path,
        "agent-a",
        report,
        audit_path=tmp_path / "audit-1.json",
        backup_path=tmp_path / "backup-1.db",
    )

    second_report = audit_db(db_path, "agent-a", generated_at=datetime(2026, 1, 1, tzinfo=UTC))
    second = apply_existing_schema_backfill(
        db_path,
        "agent-a",
        second_report,
        audit_path=tmp_path / "audit-2.json",
        backup_path=tmp_path / "backup-2.db",
    )

    assert first.inserted_mentions == 1
    assert first.inserted_entity_mentions == 1
    assert second.inserted_mentions == 0
    assert second.inserted_entity_mentions == 0
    assert Path(first.audit_path).exists()
    assert Path(first.backup_path).exists()

    conn = SQLiteStore(db_path)._conn
    try:
        rows = conn.execute(
            """
            SELECT id
            FROM relations
            WHERE agent_id = 'agent-a'
              AND subject_id = 'unit-user'
              AND predicate = 'mentions'
              AND object_id = 'ent-user'
              AND valid_until IS NULL
            """
        ).fetchall()
        assert len(rows) == 1
    finally:
        conn.close()


def test_apply_existing_schema_backfill_writes_ontology_metadata(tmp_path: Path) -> None:
    db_path = tmp_path / "memories.db"
    store = SQLiteStore(db_path)
    store.add_entity(_entity("User", "ent-user"), "agent-a")
    store.add_entity(_entity("Acme", "ent-acme"), "agent-a")
    store.add(_unit("User's name is Tommaso. User email is tommaso@example.com.", "unit-user"), "agent-a")
    store.add_relation(_relation("ent-user", "works_at", "ent-acme", "rel-work"), "agent-a")
    store._conn.close()

    report = audit_db(db_path, "agent-a", generated_at=datetime(2026, 1, 1, tzinfo=UTC))
    result = apply_existing_schema_backfill(
        db_path,
        "agent-a",
        report,
        audit_path=tmp_path / "audit.json",
        backup_path=tmp_path / "backup.db",
    )

    assert result.inserted_aliases == 1
    assert result.inserted_identifiers == 1
    assert result.inserted_attribute_types == 1
    assert result.inserted_attributes == 1
    assert result.inserted_entity_mentions == 1

    store = SQLiteStore(db_path)
    try:
        aliases = store.list_entity_aliases("ent-user", "agent-a")
        identifiers = store.list_entity_identifiers("ent-user", "agent-a")
        attributes = store.list_entity_attributes("ent-user", "agent-a", "works_at", active_only=True)
        mentions = store.find_entity_mentions_by_source("knowledge_unit", "unit-user", "agent-a")

        assert [alias.alias for alias in aliases] == ["Tommaso"]
        assert [(identifier.namespace, identifier.value) for identifier in identifiers] == [
            ("email", "tommaso@example.com")
        ]
        assert [(attribute.attribute_type, attribute.value) for attribute in attributes] == [("works_at", "Acme")]
        assert [(mention.entity_id, mention.source_kind) for mention in mentions] == [
            ("ent-user", "knowledge_unit")
        ]
    finally:
        store._conn.close()


def test_apply_existing_schema_backfill_skips_long_attribute_values(tmp_path: Path) -> None:
    db_path = tmp_path / "memories.db"
    long_label = "A" * 200
    store = SQLiteStore(db_path)
    store.add_entity(_entity("User", "ent-user"), "agent-a")
    store.add_entity(_entity(long_label, "ent-long"), "agent-a")
    store.add_relation(_relation("ent-user", "works_at", "ent-long", "rel-long"), "agent-a")
    store._conn.close()

    report = audit_db(db_path, "agent-a", generated_at=datetime(2026, 1, 1, tzinfo=UTC))
    result = apply_existing_schema_backfill(
        db_path,
        "agent-a",
        report,
        audit_path=tmp_path / "audit.json",
        backup_path=tmp_path / "backup.db",
    )

    assert result.inserted_attributes == 0
    assert result.skipped_attributes == 1
