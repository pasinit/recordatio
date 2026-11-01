"""Storage backends for knowledge units."""

import contextlib
import sqlite3
import uuid
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path

from tpasini.recordatio.entity import Entity
from tpasini.recordatio.models import (
    AttributeType,
    EntityAlias,
    EntityAttribute,
    EntityIdentifier,
    EntityMention,
    EntityType,
    KnowledgeType,
    KnowledgeUnit,
    Relation,
    RelationType,
    RetrievalContext,
    Topic,
)
from tpasini.recordatio.retrieval_text import STOPWORDS, content_words, lemmatize, normalize_for_index, tokenize

# FTS index format version: the index stores pipeline-normalized text, so any
# change to the §3.1 pipeline (including a simplemma bump — its lemma data is
# part of the output) is an index-format change. A mismatch on open triggers a
# full rebuild (RETRIEVAL_DESIGN.md §4 G-2).
_FTS_PIPELINE_VERSION = "1/simplemma-1.2.0"

# The keyword index ALWAYS lives in an attached database under this schema
# name — a `<db>.fts` sidecar file (prod default) or `:memory:` (ephemeral,
# the frozen-artifact mode). It is NEVER created inside the canonical DB file:
# the index is derived data, and keeping it out of the canonical file means no
# open can ever change a frozen artifact's bytes (§4 G-1), no matter which
# flags the caller passes.
_FTS_SCHEMA = "memfts"

# The F2 candidate-embedding cache ALSO lives in an attached sidecar database
# (`<db>.emb` in prod, `:memory:` when ephemeral) under this schema name — for
# the SAME reason as the FTS sidecar: the canonical DB file's bytes must never
# change on open, or the frozen eval-artifact sha gate (G12) breaks. Unlike FTS
# the cache is NOT rebuildable from canonical tables (it needs the model), so
# the `<db>.emb` sidecar is persistent and is what the backfill populates; the
# eval/frozen path replays the JSON artifact and never reads this sidecar.
_EMB_SCHEMA = "mememb"


def _fts5_available() -> bool:
    """Probe FTS5 support in a throwaway connection (RETRIEVAL_DESIGN.md §4 G-3)."""
    probe = sqlite3.connect(":memory:")
    try:
        probe.execute("CREATE VIRTUAL TABLE fts_probe USING fts5(x)")
        return True
    except sqlite3.OperationalError:
        return False
    finally:
        probe.close()


class Store(ABC):
    """Abstract store for knowledge units."""

    @abstractmethod
    def add(self, unit: KnowledgeUnit, agent_id: str, source_agent_id: str | None = None) -> None: ...

    @abstractmethod
    def search(
        self,
        ctx: RetrievalContext,
        agent_id: str,
        source_agent_id: str | None = None,
    ) -> list[KnowledgeUnit]: ...

    @abstractmethod
    def invalidate(self, unit_id: str, at: datetime, agent_id: str) -> None: ...

    @abstractmethod
    def add_entity(self, entity: Entity, agent_id: str, source_agent_id: str | None = None) -> None: ...

    @abstractmethod
    def get_entity_by_label(self, label: str, agent_id: str) -> Entity | None: ...

    @abstractmethod
    def get_entity_by_wikidata_id(self, wikidata_id: str, agent_id: str) -> Entity | None: ...

    @abstractmethod
    def get_entity_by_id(self, entity_id: str, agent_id: str) -> Entity | None: ...

    @abstractmethod
    def get_entity_by_canonical_key(
        self, canonical_key: str, agent_id: str, entity_type: str | None = None
    ) -> Entity | None: ...

    @abstractmethod
    def list_entities(self, agent_id: str, entity_type: str | None = None) -> list[Entity]: ...

    @abstractmethod
    def upsert_entity_alias(self, alias: EntityAlias, agent_id: str) -> None: ...

    @abstractmethod
    def get_entity_alias(self, alias_id: str, agent_id: str) -> EntityAlias | None: ...

    @abstractmethod
    def list_entity_aliases(self, entity_id: str, agent_id: str) -> list[EntityAlias]: ...

    @abstractmethod
    def find_entity_aliases(self, alias: str, agent_id: str, entity_type: str | None = None) -> list[EntityAlias]: ...

    @abstractmethod
    def delete_entity_alias(self, alias_id: str, agent_id: str) -> None: ...

    @abstractmethod
    def upsert_entity_identifier(self, identifier: EntityIdentifier, agent_id: str) -> None: ...

    @abstractmethod
    def get_entity_identifier(self, identifier_id: str, agent_id: str) -> EntityIdentifier | None: ...

    @abstractmethod
    def list_entity_identifiers(self, entity_id: str, agent_id: str) -> list[EntityIdentifier]: ...

    @abstractmethod
    def find_entity_identifiers(
        self, namespace: str, value: str, agent_id: str, entity_type: str | None = None
    ) -> list[EntityIdentifier]: ...

    @abstractmethod
    def delete_entity_identifier(self, identifier_id: str, agent_id: str) -> None: ...

    @abstractmethod
    def upsert_entity_attribute(self, attribute: EntityAttribute, agent_id: str) -> None: ...

    @abstractmethod
    def supersede_entity_attributes(self, entity_id: str, attribute_type: str, at: datetime, agent_id: str) -> int: ...

    @abstractmethod
    def get_entity_attribute(self, attribute_id: str, agent_id: str) -> EntityAttribute | None: ...

    @abstractmethod
    def list_entity_attributes(
        self,
        entity_id: str,
        agent_id: str,
        attribute_type: str | None = None,
        active_only: bool = False,
    ) -> list[EntityAttribute]: ...

    @abstractmethod
    def find_entity_attributes(
        self,
        attribute_type: str,
        agent_id: str,
        value: str | None = None,
        active_only: bool = True,
    ) -> list[EntityAttribute]: ...

    @abstractmethod
    def delete_entity_attribute(self, attribute_id: str, agent_id: str) -> None: ...

    @abstractmethod
    def upsert_entity_mention(self, mention: EntityMention, agent_id: str) -> None: ...

    @abstractmethod
    def get_entity_mention(self, mention_id: str, agent_id: str) -> EntityMention | None: ...

    @abstractmethod
    def list_entity_mentions(self, entity_id: str, agent_id: str) -> list[EntityMention]: ...

    @abstractmethod
    def find_entity_mentions_by_source(
        self, source_kind: str, source_id: str, agent_id: str
    ) -> list[EntityMention]: ...

    @abstractmethod
    def delete_entity_mention(self, mention_id: str, agent_id: str) -> None: ...

    @abstractmethod
    def upsert_entity_type(self, entity_type: EntityType, agent_id: str) -> None: ...

    @abstractmethod
    def get_entity_type(self, entity_type_id: str, agent_id: str) -> EntityType | None: ...

    @abstractmethod
    def get_entity_type_by_name(self, name: str, agent_id: str) -> EntityType | None: ...

    @abstractmethod
    def list_entity_types(self, agent_id: str) -> list[EntityType]: ...

    @abstractmethod
    def delete_entity_type(self, entity_type_id: str, agent_id: str) -> None: ...

    @abstractmethod
    def upsert_attribute_type(self, attribute_type: AttributeType, agent_id: str) -> None: ...

    @abstractmethod
    def get_attribute_type(self, attribute_type_id: str, agent_id: str) -> AttributeType | None: ...

    @abstractmethod
    def get_attribute_type_by_name(self, name: str, agent_id: str) -> AttributeType | None: ...

    @abstractmethod
    def list_attribute_types(self, agent_id: str) -> list[AttributeType]: ...

    @abstractmethod
    def delete_attribute_type(self, attribute_type_id: str, agent_id: str) -> None: ...

    @abstractmethod
    def add_relation(self, relation: Relation, agent_id: str, source_agent_id: str | None = None) -> None: ...

    @abstractmethod
    def get_relations(self, subject_id: str, agent_id: str) -> list[Relation]: ...

    @abstractmethod
    def invalidate_relation(self, relation_id: str, at: datetime, agent_id: str) -> None: ...

    @abstractmethod
    def supersede_relations(self, subject_id: str, predicate: str, at: datetime, agent_id: str) -> int: ...

    @abstractmethod
    def add_relation_type(self, rt: RelationType) -> None: ...

    @abstractmethod
    def list_relation_types(self) -> list[RelationType]: ...

    @abstractmethod
    def add_topic(self, topic: Topic) -> None: ...

    @abstractmethod
    def list_topics(self) -> list[Topic]: ...

    @abstractmethod
    def tag_unit(self, unit_id: str, topic_id: str, agent_id: str) -> None: ...

    @abstractmethod
    def tag_entity(self, entity_id: str, topic_id: str, agent_id: str) -> None: ...


_SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge_units (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    content TEXT NOT NULL,
    asserted_at TEXT NOT NULL,
    valid_from TEXT NOT NULL,
    valid_until TEXT,
    provenance TEXT NOT NULL,
    agent_id TEXT NOT NULL DEFAULT 'default',
    source_agent_id TEXT,
    extraction_count INTEGER NOT NULL DEFAULT 1,
    retrieval_count INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_knowledge_units_valid_until
    ON knowledge_units(valid_until);

CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    type TEXT NOT NULL DEFAULT 'concept',
    subtype TEXT,
    canonical_label TEXT,
    canonical_key TEXT,
    wikidata_id TEXT,
    description TEXT,
    wikipedia_url TEXT,
    asserted_at TEXT NOT NULL,
    agent_id TEXT NOT NULL DEFAULT 'default',
    source_agent_id TEXT,
    extraction_count INTEGER NOT NULL DEFAULT 1,
    retrieval_count INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_entities_wikidata_id ON entities(wikidata_id);

CREATE TABLE IF NOT EXISTS entity_aliases (
    id TEXT PRIMARY KEY,
    entity_id TEXT NOT NULL,
    alias TEXT NOT NULL,
    alias_key TEXT NOT NULL,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL,
    agent_id TEXT NOT NULL DEFAULT 'default'
);

CREATE TABLE IF NOT EXISTS entity_identifiers (
    id TEXT PRIMARY KEY,
    entity_id TEXT NOT NULL,
    namespace TEXT NOT NULL,
    value TEXT NOT NULL,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL,
    agent_id TEXT NOT NULL DEFAULT 'default'
);

CREATE TABLE IF NOT EXISTS entity_attributes (
    id TEXT PRIMARY KEY,
    entity_id TEXT NOT NULL,
    attribute_type TEXT NOT NULL,
    value TEXT NOT NULL,
    asserted_at TEXT NOT NULL,
    valid_from TEXT NOT NULL,
    valid_until TEXT,
    provenance TEXT NOT NULL,
    agent_id TEXT NOT NULL DEFAULT 'default'
);

CREATE TABLE IF NOT EXISTS entity_mentions (
    id TEXT PRIMARY KEY,
    entity_id TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    source_id TEXT NOT NULL,
    snippet TEXT,
    asserted_at TEXT NOT NULL,
    agent_id TEXT NOT NULL DEFAULT 'default'
);

CREATE TABLE IF NOT EXISTS entity_types (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    name_key TEXT NOT NULL,
    description TEXT NOT NULL,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL,
    agent_id TEXT NOT NULL DEFAULT 'default'
);

CREATE TABLE IF NOT EXISTS attribute_types (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    name_key TEXT NOT NULL,
    description TEXT NOT NULL,
    value_type TEXT NOT NULL,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL,
    agent_id TEXT NOT NULL DEFAULT 'default'
);

CREATE TABLE IF NOT EXISTS relation_types (
    id TEXT PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    description TEXT NOT NULL,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL,
    is_functional INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS relations (
    id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object_id TEXT NOT NULL,
    asserted_at TEXT NOT NULL,
    valid_from TEXT NOT NULL,
    valid_until TEXT,
    provenance TEXT NOT NULL,
    agent_id TEXT NOT NULL DEFAULT 'default',
    source_agent_id TEXT,
    extraction_count INTEGER NOT NULL DEFAULT 1,
    retrieval_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS topics (
    id TEXT PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    description TEXT NOT NULL,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS unit_topics (
    agent_id TEXT NOT NULL DEFAULT 'default',
    unit_id TEXT NOT NULL,
    topic_id TEXT NOT NULL,
    PRIMARY KEY (agent_id, unit_id, topic_id)
);

CREATE TABLE IF NOT EXISTS entity_topics (
    agent_id TEXT NOT NULL DEFAULT 'default',
    entity_id TEXT NOT NULL,
    topic_id TEXT NOT NULL,
    PRIMARY KEY (agent_id, entity_id, topic_id)
);
"""

_SEED_RELATION_TYPES: tuple[tuple[str, str, bool], ...] = (
    ("mentions", "unit references an entity in passing", False),
    ("about", "unit is primarily about an entity", False),
    ("instance_of", "entity is an instance of another (mirrors Wikidata P31)", False),
    ("part_of", "entity is part of another entity (mirrors Wikidata P361)", False),
    ("supersedes", "this unit replaces a prior unit", False),
    ("derived_from", "this unit was consolidated from other units", False),
    # Functional predicates: only one active value per subject is valid at a time.
    ("lives_in", "entity currently lives or resides in a location", True),
    ("works_at", "entity is currently employed by or works at an organization", True),
    ("has_role", "entity currently holds a specific role or job title", True),
    ("has_name", "canonical name of the entity", True),
    ("located_in", "entity is currently located in a place", True),
    ("has_nationality", "entity's current nationality or citizenship", True),
)

_SEED_TOPICS: tuple[tuple[str, str], ...] = (
    ("technology", "software, hardware, tools, APIs"),
    ("finance", "markets, companies, revenue, accounting"),
    ("science", "research, experiments, findings"),
    ("people", "individuals, roles, biographical facts"),
    ("organizations", "companies, institutions, teams"),
    ("locations", "places, geographies"),
    ("events", "occurrences at a point in time"),
    ("procedures", "how-to knowledge, workflows"),
)

_SEED_ENTITY_TYPES: tuple[tuple[str, str], ...] = (
    ("person", "individual people and personal profiles"),
    ("organization", "companies, institutions, teams, and groups"),
    ("place", "physical and administrative locations"),
    ("listing", "marketplace, real estate, travel, or catalog listings"),
    ("asset", "owned or tracked resources and objects"),
    ("task", "work items, todos, and operational actions"),
    ("preference_profile", "stable preference sets for a person or agent"),
    ("event", "occurrences anchored in time"),
    ("document", "files, records, messages, and written artifacts"),
    ("concept", "abstract topics, categories, and ideas"),
)


class SQLiteStore(Store):
    """SQLite-backed knowledge store."""

    def __init__(
        self, db_path: Path, *, ephemeral_fts: bool = False, ephemeral_emb: bool | None = None
    ) -> None:
        """``ephemeral_fts=True`` keeps the keyword index in ``:memory:``
        instead of the persistent ``<db>.fts`` sidecar — for frozen artifacts
        (RETRIEVAL_DESIGN.md §4 G-1): no sidecar file appears next to the
        artifact, at the cost of a full index build on every open (~0.5 s on
        the 73K-row eval store). Either way the canonical DB file itself is
        never written by index code.

        ``ephemeral_emb`` controls the F2 embedding sidecar independently;
        ``None`` (default) follows ``ephemeral_fts`` so frozen-artifact opens
        leave no ``<db>.emb`` file. The backfill sets ``ephemeral_emb=False``
        with ``ephemeral_fts=True`` to PERSIST embeddings into ``<db>.emb``
        without rebuilding/locking the live FTS sidecar."""
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._ensure_agent_id_columns()
        self._ensure_entity_ontology_columns()
        self._ensure_source_agent_id_columns()
        self._ensure_reinforcement_count_columns()
        self._dedup_entities()
        self._ensure_entity_label_index()
        self._ensure_entity_indexes()
        self._ensure_relation_types_functional_column()
        self._dedup_active_relations()
        self._ensure_relations_active_unique_index()
        self._seed()
        self._conn.commit()
        self._fts_schema = _FTS_SCHEMA
        # F2 embedding cache sidecar — attached like FTS so the canonical DB is
        # never mutated on open (preserves the frozen-DB sha gate G12).
        self._emb_schema = _EMB_SCHEMA
        if ephemeral_emb is None:
            ephemeral_emb = ephemeral_fts
        emb_ephemeral = ephemeral_emb or str(db_path) == ":memory:"
        emb_target = ":memory:" if emb_ephemeral else f"{db_path}.emb"
        self._conn.execute(f"ATTACH DATABASE ? AS {_EMB_SCHEMA}", (emb_target,))
        self._ensure_item_embeddings_table()
        self.fts_enabled = _fts5_available()
        if self.fts_enabled:
            ephemeral = ephemeral_fts or str(db_path) == ":memory:"
            target = ":memory:" if ephemeral else f"{db_path}.fts"
            self._conn.execute(f"ATTACH DATABASE ? AS {_FTS_SCHEMA}", (target,))
            self._ensure_fts_index()

    @staticmethod
    def _identity_key(value: str) -> str:
        return " ".join(value.casefold().split())

    def _ensure_source_agent_id_columns(self) -> None:
        """Add optional source-agent provenance columns for agent-scoped memory rows."""
        migrations = [
            "ALTER TABLE knowledge_units ADD COLUMN source_agent_id TEXT",
            "ALTER TABLE entities        ADD COLUMN source_agent_id TEXT",
            "ALTER TABLE relations       ADD COLUMN source_agent_id TEXT",
        ]
        for stmt in migrations:
            with contextlib.suppress(sqlite3.OperationalError):
                self._conn.execute(stmt)
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_knowledge_units_source_agent ON knowledge_units(agent_id, source_agent_id)",
            "CREATE INDEX IF NOT EXISTS idx_entities_source_agent ON entities(agent_id, source_agent_id)",
            "CREATE INDEX IF NOT EXISTS idx_relations_source_agent ON relations(agent_id, source_agent_id)",
        ]
        for stmt in indexes:
            with contextlib.suppress(sqlite3.OperationalError):
                self._conn.execute(stmt)
        self._conn.commit()

    # Tables carrying reinforcement signals, in (table, model-friendly) order.
    _REINFORCEMENT_TABLES: tuple[str, ...] = ("knowledge_units", "entities", "relations")

    def _ensure_reinforcement_count_columns(self) -> None:
        """Add reinforcement counters to legacy tables (idempotent schema migration).

        Two integer signals per fact:
          - extraction_count: times asserted/re-asserted by extraction (write-side). Default 1.
          - retrieval_count:  times surfaced in a briefing (read-side). Default 0.
        Existing rows get the defaults. PRAGMA-guarded so it is safe to run repeatedly.
        """
        for table in self._REINFORCEMENT_TABLES:
            existing = {row["name"] for row in self._conn.execute(f"PRAGMA table_info({table})").fetchall()}
            if "extraction_count" not in existing:
                with contextlib.suppress(sqlite3.OperationalError):
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN extraction_count INTEGER NOT NULL DEFAULT 1")
            if "retrieval_count" not in existing:
                with contextlib.suppress(sqlite3.OperationalError):
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN retrieval_count INTEGER NOT NULL DEFAULT 0")
        self._conn.commit()

    def _dedup_entities(self) -> None:
        """Remove duplicate entity rows, keeping the one with the lowest rowid per (agent_id, label)."""
        self._conn.execute(
            """
            DELETE FROM entities
            WHERE rowid NOT IN (
                SELECT MIN(rowid) FROM entities GROUP BY agent_id, LOWER(label)
            )
            """
        )
        self._conn.commit()

    def _ensure_entity_label_index(self) -> None:
        """Replace global label unique index with a per-agent one."""
        # Drop the old global unique index so it doesn't block cross-agent inserts.
        self._conn.execute("DROP INDEX IF EXISTS idx_entities_label_unique")
        try:
            self._conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_entities_agent_label_unique ON entities(agent_id, LOWER(label))"
            )
            self._conn.commit()
        except sqlite3.OperationalError:
            pass  # index already exists

    def _ensure_entity_ontology_columns(self) -> None:
        """Add generic ontology columns to legacy entities tables."""
        migrations = [
            "ALTER TABLE entities ADD COLUMN type TEXT NOT NULL DEFAULT 'concept'",
            "ALTER TABLE entities ADD COLUMN subtype TEXT",
            "ALTER TABLE entities ADD COLUMN canonical_label TEXT",
            "ALTER TABLE entities ADD COLUMN canonical_key TEXT",
        ]
        for stmt in migrations:
            with contextlib.suppress(sqlite3.OperationalError):
                self._conn.execute(stmt)
        self._conn.execute("UPDATE entities SET type = 'concept' WHERE type IS NULL OR type = ''")
        self._conn.execute("UPDATE entities SET canonical_label = label WHERE canonical_label IS NULL")
        self._conn.commit()

    def _ensure_entity_indexes(self) -> None:
        indexes = [
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_entities_agent_type_canonical_key_unique
            ON entities(agent_id, type, canonical_key)
            WHERE canonical_key IS NOT NULL
            """,
            "CREATE INDEX IF NOT EXISTS idx_entities_agent_type ON entities(agent_id, type)",
            "CREATE INDEX IF NOT EXISTS idx_entity_aliases_agent_entity ON entity_aliases(agent_id, entity_id)",
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_entity_aliases_agent_alias ON entity_aliases(agent_id, alias_key)",
            "CREATE INDEX IF NOT EXISTS idx_entity_identifiers_agent_entity ON entity_identifiers(agent_id, entity_id)",
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_entity_identifiers_agent_namespace_value
            ON entity_identifiers(agent_id, namespace, value)
            """,
            "CREATE INDEX IF NOT EXISTS idx_entity_attributes_agent_entity ON entity_attributes(agent_id, entity_id)",
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_entity_attributes_unique_active
            ON entity_attributes(agent_id, entity_id, attribute_type, value)
            WHERE valid_until IS NULL
            """,
            "CREATE INDEX IF NOT EXISTS idx_entity_mentions_agent_entity ON entity_mentions(agent_id, entity_id)",
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_entity_mentions_agent_source
            ON entity_mentions(agent_id, entity_id, source_kind, source_id)
            """,
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_entity_types_agent_name ON entity_types(agent_id, name_key)",
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_attribute_types_agent_name ON attribute_types(agent_id, name_key)",
        ]
        for stmt in indexes:
            with contextlib.suppress(sqlite3.OperationalError):
                self._conn.execute(stmt)
        self._conn.commit()

    def _ensure_relation_types_functional_column(self) -> None:
        """Add is_functional column to relation_types if missing (schema migration)."""
        try:
            self._conn.execute("ALTER TABLE relation_types ADD COLUMN is_functional INTEGER NOT NULL DEFAULT 0")
            self._conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists

    def _dedup_active_relations(self) -> None:
        """Collapse duplicate *active* relation rows per (agent_id, subject_id, predicate, object_id),
        keeping the earliest asserted_at. Superseded rows (valid_until IS NOT NULL) are intentional
        history (functional-predicate move-and-back) and are left alone.
        """
        self._conn.execute(
            """
            DELETE FROM relations
            WHERE valid_until IS NULL
              AND rowid NOT IN (
                  SELECT MIN(rowid) FROM relations
                  WHERE valid_until IS NULL
                  GROUP BY agent_id, subject_id, predicate, object_id
              )
            """
        )
        self._conn.commit()

    def _ensure_relations_active_unique_index(self) -> None:
        """Partial unique index: at most one *active* relation per
        (agent_id, subject_id, predicate, object_id). Superseded rows are out of scope,
        so functional-predicate history (X moves Y→Z→Y) keeps working.
        """
        with contextlib.suppress(sqlite3.OperationalError):
            self._conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_relations_unique_active
                ON relations(agent_id, subject_id, predicate, object_id)
                WHERE valid_until IS NULL
                """
            )
            self._conn.commit()

    def _ensure_item_embeddings_table(self) -> None:
        """Idempotent migration for the F2 candidate-embedding cache.

        One row per (item_id, agent_id): the bi-encoder embedding of the item's
        searchable surface, keyed by a ``surface_hash`` so re-embedding is a
        no-op while the surface is unchanged (F2_EMBEDDINGS_DESIGN.md).  Unlike
        the FTS index this is NOT cheaply rebuildable (it needs the model), so
        the ``<db>.emb`` sidecar is persistent and is backfilled explicitly
        (``scripts/backfill_embeddings.py``).

        Old prod stores (no sidecar) upgrade cleanly here and run with an empty
        cache — retrieval degrades to BM25 for every item until backfilled
        (the graceful path).  ``CREATE TABLE IF NOT EXISTS`` makes re-open a
        no-op.  Lives in the attached sidecar so the canonical DB is never
        mutated (the frozen eval-artifact sha gate G12 stays intact).
        """
        e = self._emb_schema
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {e}.item_embeddings (
                item_id      TEXT NOT NULL,
                agent_id     TEXT NOT NULL,
                kind         TEXT NOT NULL,
                surface_hash TEXT NOT NULL,
                embedding    BLOB NOT NULL,
                model        TEXT NOT NULL,
                asserted_at  TEXT NOT NULL,
                PRIMARY KEY (item_id, agent_id)
            )
            """
        )
        with contextlib.suppress(sqlite3.OperationalError):
            self._conn.execute(
                f"CREATE INDEX IF NOT EXISTS {e}.idx_item_embeddings_agent ON item_embeddings(agent_id)"
            )
        self._conn.commit()

    def get_item_embedding_hash(self, item_id: str, agent_id: str) -> str | None:
        """Stored ``surface_hash`` for an item, or None if not embedded.

        The write-side guard: a caller embeds only when this differs from the
        item's current surface hash (no-op re-embed avoidance).
        """
        row = self._conn.execute(
            f"SELECT surface_hash FROM {self._emb_schema}.item_embeddings WHERE item_id = ? AND agent_id = ?",
            (item_id, agent_id),
        ).fetchone()
        return row["surface_hash"] if row else None

    def upsert_item_embedding(
        self,
        *,
        item_id: str,
        agent_id: str,
        kind: str,
        surface_hash: str,
        embedding: bytes,
        model: str,
        asserted_at: str,
    ) -> None:
        """Insert or replace one item's embedding row (write-time hook target).

        ``embedding`` is a packed float32 BLOB; ``surface_hash`` keys
        invalidation.  Single-row upsert keyed by (item_id, agent_id) — a
        changed surface overwrites the stale vector in place.
        """
        self._conn.execute(
            f"""
            INSERT INTO {self._emb_schema}.item_embeddings
                (item_id, agent_id, kind, surface_hash, embedding, model, asserted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(item_id, agent_id) DO UPDATE SET
                kind = excluded.kind,
                surface_hash = excluded.surface_hash,
                embedding = excluded.embedding,
                model = excluded.model,
                asserted_at = excluded.asserted_at
            """,
            (item_id, agent_id, kind, surface_hash, embedding, model, asserted_at),
        )
        self._conn.commit()

    def item_surfaces_for_embedding(
        self, unit_ids: list[str], entity_ids: list[str]
    ) -> list[tuple[str, str, str, str]]:
        """Read back ``(item_id, agent_id, kind, surface)`` from canonical rows.

        The surface is the EXACT stored text the budgeter/retrieval embed (RR4):
        ``content`` for units, ``"label description"`` for entities.  Reading
        from the canonical tables (rather than the extractor's in-memory objects)
        guarantees the embedded text matches what retrieval will surface and the
        scope is the real home scope, whatever the routing decided.  Used by the
        write-time embedding hook and the backfill.
        """
        out: list[tuple[str, str, str, str]] = []
        chunk = 800
        for start in range(0, len(unit_ids), chunk):
            batch = unit_ids[start : start + chunk]
            marks = ",".join("?" for _ in batch)
            rows = self._conn.execute(
                f"SELECT id, agent_id, content FROM knowledge_units WHERE id IN ({marks})",  # noqa: S608
                tuple(batch),
            ).fetchall()
            for row in rows:
                out.append((row["id"], row["agent_id"], "unit", row["content"] or ""))
        for start in range(0, len(entity_ids), chunk):
            batch = entity_ids[start : start + chunk]
            marks = ",".join("?" for _ in batch)
            rows = self._conn.execute(
                f"SELECT id, agent_id, label, description FROM entities WHERE id IN ({marks})",  # noqa: S608
                tuple(batch),
            ).fetchall()
            for row in rows:
                surface = f"{row['label'] or ''} {row['description'] or ''}".strip()
                out.append((row["id"], row["agent_id"], "entity", surface))
        return out

    def relation_surfaces_for_embedding(self, relation_ids: list[str]) -> list[tuple[str, str, str, str]]:
        """Read back ``(relation_id, agent_id, "relation", surface)`` for relations.

        Surface is the rendered triple ``"subject predicate object"`` with
        endpoint labels resolved from the ``entities`` table (raw id fallback
        when an endpoint is a unit / missing entity) — the SAME recipe the eval
        generator and the budgeter's ``_relation_text`` use, but resolved against
        canonical labels at write time so relation embeddings can be cached like
        units/entities (avoiding the per-query on-the-fly embed of thousands of
        relations).  Endpoint labels resolve within the relation's own scope.
        """
        out: list[tuple[str, str, str, str]] = []
        chunk = 800
        for start in range(0, len(relation_ids), chunk):
            batch = relation_ids[start : start + chunk]
            marks = ",".join("?" for _ in batch)
            rows = self._conn.execute(
                f"""
                SELECT r.id AS id, r.agent_id AS agent_id, r.predicate AS predicate,
                       r.subject_id AS subject_id, r.object_id AS object_id,
                       s.label AS subject_label, o.label AS object_label
                FROM relations r
                LEFT JOIN entities s ON s.id = r.subject_id AND s.agent_id = r.agent_id
                LEFT JOIN entities o ON o.id = r.object_id AND o.agent_id = r.agent_id
                WHERE r.id IN ({marks})
                """,  # noqa: S608 — marks are placeholders; ids bound below
                tuple(batch),
            ).fetchall()
            for row in rows:
                subj = row["subject_label"] or row["subject_id"]
                obj = row["object_label"] or row["object_id"]
                surface = f"{subj} {row['predicate']} {obj}"
                out.append((row["id"], row["agent_id"], "relation", surface))
        return out

    def relation_ids_referencing(self, item_ids: list[str]) -> list[str]:
        """Ids of relations whose subject or object is in ``item_ids``.

        Used by the write-time embedding hook to (re)embed the relations whose
        rendered surface may have changed when a touched entity/unit was
        written.  Deterministic order (id-asc) for reproducibility.
        """
        if not item_ids:
            return []
        out: set[str] = set()
        chunk = 800
        for start in range(0, len(item_ids), chunk):
            batch = item_ids[start : start + chunk]
            marks = ",".join("?" for _ in batch)
            rows = self._conn.execute(
                f"""
                SELECT id FROM relations
                WHERE subject_id IN ({marks}) OR object_id IN ({marks})
                """,  # noqa: S608 — placeholders only; ids bound below
                (*batch, *batch),
            ).fetchall()
            out.update(r["id"] for r in rows)
        return sorted(out)

    def load_item_embeddings(self, item_ids: list[str], scopes: tuple[str, ...]) -> dict[str, bytes]:
        """Return ``{item_id: embedding BLOB}`` for the given ids across scopes.

        Items without a stored embedding are simply absent from the result
        (the caller's RRF union degrades them to BM25 — graceful, F2 source
        contract).  Ids are globally unique, so the first scope holding the row
        wins.
        """
        if not item_ids:
            return {}
        scope_marks = ",".join("?" for _ in scopes)
        out: dict[str, bytes] = {}
        # Chunk the id list to stay under SQLite's parameter limit on large pools.
        chunk = 800
        for start in range(0, len(item_ids), chunk):
            batch = item_ids[start : start + chunk]
            id_marks = ",".join("?" for _ in batch)
            rows = self._conn.execute(
                f"""
                SELECT item_id, embedding FROM {self._emb_schema}.item_embeddings
                WHERE agent_id IN ({scope_marks}) AND item_id IN ({id_marks})
                """,  # noqa: S608 — marks are placeholder-only, values bound below
                (*scopes, *batch),
            ).fetchall()
            for row in rows:
                if row["item_id"] not in out:
                    out[row["item_id"]] = row["embedding"]
        return out

    def _ensure_fts_index(self) -> None:
        """Idempotent FTS5 keyword-index migration (RETRIEVAL_DESIGN.md §3.2, §4 G-2).

        One virtual table over §3.1-pipeline-normalized text: unit rows fill
        ``content``, entity rows fill ``label`` + ``description``. The index is
        derived data — never a rendering source, always rebuildable from the
        canonical tables. Bitemporal validity is NOT indexed: superseded rows
        stay (``invalidate`` makes no index write); consumers re-check validity
        at materialization. Safe to run on every open: re-runs are no-ops, and
        a pipeline-version mismatch, an empty/fresh index (ephemeral opens), or
        a row-count drift vs the canonical tables (e.g. writes made by an
        FTS5-less build through the LIKE fallback) triggers a full rebuild.
        """
        s = self._fts_schema
        self._conn.execute(
            f"CREATE TABLE IF NOT EXISTS {s}.memento_fts_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self._conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {s}.memento_fts USING fts5(
                item_id UNINDEXED, kind UNINDEXED, agent_id UNINDEXED,
                content, label, description,
                tokenize = 'unicode61'
            )
            """
        )
        row = self._conn.execute(f"SELECT value FROM {s}.memento_fts_meta WHERE key = 'pipeline_version'").fetchone()
        if row is None or row["value"] != _FTS_PIPELINE_VERSION or self._fts_count_drift():
            self.rebuild_fts_index()
        self._conn.commit()

    def _fts_count_drift(self) -> bool:
        """Cheap staleness heuristic: indexed rows vs canonical rows.

        Catches missed inserts/deletes (foreign writers, FTS5-less builds);
        equal-count content edits are not detected — ``rebuild_fts_index`` is
        the universal recovery (G-2).
        """
        canonical = (
            self._conn.execute("SELECT COUNT(*) FROM knowledge_units").fetchone()[0]
            + self._conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        )
        indexed = self._conn.execute(f"SELECT COUNT(*) FROM {self._fts_schema}.memento_fts").fetchone()[0]
        return canonical != indexed

    def rebuild_fts_index(self) -> None:
        """Rebuild the keyword index from scratch off the canonical tables.

        The recovery path for a corrupt/stale index (G-2: rebuild is never a
        data loss — the index stores only derived, normalized text). Indexes
        ALL rows including superseded ones, matching what the write-path hooks
        accumulate; deterministic order (rowid) for reproducibility.
        """
        if not self.fts_enabled:
            return
        s = self._fts_schema
        self._conn.execute(f"DELETE FROM {s}.memento_fts")
        units = self._conn.execute("SELECT id, agent_id, content FROM knowledge_units ORDER BY rowid").fetchall()
        for row in units:
            self._conn.execute(
                f"INSERT INTO {s}.memento_fts (item_id, kind, agent_id, content, label, description) "
                "VALUES (?, 'unit', ?, ?, '', '')",
                (row["id"], row["agent_id"], normalize_for_index(row["content"])),
            )
        entities = self._conn.execute("SELECT id, agent_id, label, description FROM entities ORDER BY rowid").fetchall()
        for row in entities:
            self._conn.execute(
                f"INSERT INTO {s}.memento_fts (item_id, kind, agent_id, content, label, description) "
                "VALUES (?, 'entity', ?, '', ?, ?)",
                (row["id"], row["agent_id"], normalize_for_index(row["label"]),
                 normalize_for_index(row["description"] or "")),
            )
        self._conn.execute(
            f"INSERT INTO {s}.memento_fts_meta (key, value) VALUES ('pipeline_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (_FTS_PIPELINE_VERSION,),
        )
        self._conn.commit()

    def _fts_sync_unit(self, unit_id: str) -> None:
        """Write-path hook: mirror one unit's canonical row into the index.

        Re-reads the canonical row (so INSERT OR REPLACE upserts stay faithful)
        and runs before the caller's commit — the index write is atomic with
        the canonical write.
        """
        if not self.fts_enabled:
            return
        s = self._fts_schema
        self._conn.execute(f"DELETE FROM {s}.memento_fts WHERE item_id = ?", (unit_id,))
        row = self._conn.execute("SELECT agent_id, content FROM knowledge_units WHERE id = ?", (unit_id,)).fetchone()
        if row:
            self._conn.execute(
                f"INSERT INTO {s}.memento_fts (item_id, kind, agent_id, content, label, description) "
                "VALUES (?, 'unit', ?, ?, '', '')",
                (unit_id, row["agent_id"], normalize_for_index(row["content"])),
            )

    def _fts_sync_entity(self, entity_id: str) -> None:
        """Write-path hook: mirror one entity's canonical row into the index.

        Reading back the canonical row also handles INSERT OR IGNORE correctly:
        an ignored insert (label collision under a different id) finds no row
        and writes nothing.
        """
        if not self.fts_enabled:
            return
        s = self._fts_schema
        self._conn.execute(f"DELETE FROM {s}.memento_fts WHERE item_id = ?", (entity_id,))
        row = self._conn.execute(
            "SELECT agent_id, label, description FROM entities WHERE id = ?", (entity_id,)
        ).fetchone()
        if row:
            self._conn.execute(
                f"INSERT INTO {s}.memento_fts (item_id, kind, agent_id, content, label, description) "
                "VALUES (?, 'entity', ?, '', ?, ?)",
                (entity_id, row["agent_id"], normalize_for_index(row["label"]),
                 normalize_for_index(row["description"] or "")),
            )

    def _ensure_agent_id_columns(self) -> None:
        """Add agent_id column to all agent-scoped tables if missing (schema migration)."""
        migrations = [
            "ALTER TABLE knowledge_units ADD COLUMN agent_id TEXT NOT NULL DEFAULT 'default'",
            "ALTER TABLE entities        ADD COLUMN agent_id TEXT NOT NULL DEFAULT 'default'",
            "ALTER TABLE relations       ADD COLUMN agent_id TEXT NOT NULL DEFAULT 'default'",
            "ALTER TABLE unit_topics     ADD COLUMN agent_id TEXT NOT NULL DEFAULT 'default'",
            "ALTER TABLE entity_topics   ADD COLUMN agent_id TEXT NOT NULL DEFAULT 'default'",
            "ALTER TABLE entity_aliases  ADD COLUMN agent_id TEXT NOT NULL DEFAULT 'default'",
            "ALTER TABLE entity_identifiers ADD COLUMN agent_id TEXT NOT NULL DEFAULT 'default'",
            "ALTER TABLE entity_attributes ADD COLUMN agent_id TEXT NOT NULL DEFAULT 'default'",
            "ALTER TABLE entity_mentions ADD COLUMN agent_id TEXT NOT NULL DEFAULT 'default'",
            "ALTER TABLE entity_types    ADD COLUMN agent_id TEXT NOT NULL DEFAULT 'default'",
            "ALTER TABLE attribute_types ADD COLUMN agent_id TEXT NOT NULL DEFAULT 'default'",
        ]
        for stmt in migrations:
            with contextlib.suppress(sqlite3.OperationalError):
                self._conn.execute(stmt)
        # agent_id indexes are created here (not in _SCHEMA) so they run after
        # the columns are guaranteed to exist on legacy databases.
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_knowledge_units_agent ON knowledge_units(agent_id)",
            "CREATE INDEX IF NOT EXISTS idx_entities_agent ON entities(agent_id)",
            "CREATE INDEX IF NOT EXISTS idx_relations_agent ON relations(agent_id)",
        ]
        for stmt in indexes:
            with contextlib.suppress(sqlite3.OperationalError):
                self._conn.execute(stmt)
        self._conn.commit()

    def _seed(self) -> None:
        now_iso = datetime.now(UTC).isoformat()
        for name, description, is_functional in _SEED_RELATION_TYPES:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO relation_types
                    (id, name, description, source, created_at, is_functional)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (str(uuid.uuid4()), name, description, "seed", now_iso, 1 if is_functional else 0),
            )
            # Ensure is_functional is up-to-date on existing rows (idempotent).
            if is_functional:
                self._conn.execute(
                    "UPDATE relation_types SET is_functional = 1 WHERE name = ?",
                    (name,),
                )
        for name, description in _SEED_TOPICS:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO topics
                    (id, name, description, source, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (str(uuid.uuid4()), name, description, "seed", now_iso),
            )
        for name, description in _SEED_ENTITY_TYPES:
            self._conn.execute(
                """
                INSERT INTO entity_types
                    (id, name, name_key, description, source, created_at, agent_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(agent_id, name_key) DO UPDATE SET
                    name = excluded.name,
                    description = excluded.description,
                    source = excluded.source
                """,
                (str(uuid.uuid4()), name, self._identity_key(name), description, "seed", now_iso, "default"),
            )

    def add(self, unit: KnowledgeUnit, agent_id: str, source_agent_id: str | None = None) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO knowledge_units
                (id, type, content, asserted_at, valid_from, valid_until, provenance, agent_id, source_agent_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                unit.id,
                unit.type.value,
                unit.content,
                unit.asserted_at.isoformat(),
                unit.valid_from.isoformat(),
                unit.valid_until.isoformat() if unit.valid_until is not None else None,
                unit.provenance,
                agent_id,
                source_agent_id,
            ),
        )
        self._fts_sync_unit(unit.id)
        self._conn.commit()

    def search(self, ctx: RetrievalContext, agent_id: str, source_agent_id: str | None = None) -> list[KnowledgeUnit]:
        now_iso = datetime.now().isoformat()
        pattern = f"%{ctx.query}%"
        if ctx.topic_hint:
            source_clause = "AND ku.source_agent_id = ?" if source_agent_id is not None else ""
            params = [ctx.topic_hint, agent_id, now_iso, pattern]
            if source_agent_id is not None:
                params.append(source_agent_id)
            params.append(ctx.top_k)
            rows = self._conn.execute(
                f"""
                SELECT ku.id, ku.type, ku.content, ku.asserted_at,
                       ku.valid_from, ku.valid_until, ku.provenance,
                       ku.extraction_count, ku.retrieval_count,
                       CASE WHEN EXISTS (
                           SELECT 1 FROM unit_topics ut
                           JOIN topics t ON t.id = ut.topic_id
                           WHERE ut.unit_id = ku.id AND LOWER(t.name) = LOWER(?)
                       ) THEN 1 ELSE 0 END AS topic_match
                FROM knowledge_units ku
                WHERE ku.agent_id = ?
                  AND (ku.valid_until IS NULL OR ku.valid_until > ?)
                  AND LOWER(ku.content) LIKE LOWER(?)
                  {source_clause}
                ORDER BY topic_match DESC, ku.asserted_at DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        else:
            source_clause = "AND source_agent_id = ?" if source_agent_id is not None else ""
            params = [agent_id, now_iso, pattern]
            if source_agent_id is not None:
                params.append(source_agent_id)
            params.append(ctx.top_k)
            rows = self._conn.execute(
                f"""
                SELECT id, type, content, asserted_at, valid_from, valid_until, provenance,
                       extraction_count, retrieval_count
                FROM knowledge_units
                WHERE agent_id = ?
                  AND (valid_until IS NULL OR valid_until > ?)
                  AND LOWER(content) LIKE LOWER(?)
                  {source_clause}
                ORDER BY asserted_at DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        return [self._row_to_unit(r) for r in rows]

    # ------------------------------------------------------------------
    # User-scoped reads for the nightly feedback analyst (read-only; span
    # every scope a user owns — ``{user}/_user`` plus each ``{user}/{slug}``).
    # A GLOB prefix (not LIKE) so ``_`` in a user id is a literal, not a
    # single-char wildcard.
    # ------------------------------------------------------------------

    def list_units_for_user(
        self,
        user_id: str,
        *,
        asserted_after: str | None = None,
        asserted_before: str | None = None,
        limit: int = 200,
    ) -> list[tuple[KnowledgeUnit, str]]:
        """Active units across all of ``user_id``'s scopes, newest first.

        ``asserted_after``/``asserted_before`` are ISO-8601 bounds compared
        lexically against ``asserted_at`` (same format ⇒ same order). Returns
        ``(unit, agent_id)`` pairs so callers can see which agent's scope each
        unit belongs to.
        """
        clauses = ["(agent_id = ? OR agent_id GLOB ?)", "(valid_until IS NULL OR valid_until > ?)"]
        params: list[object] = [user_id, f"{user_id}/*", datetime.now().isoformat()]
        if asserted_after is not None:
            clauses.append("asserted_at >= ?")
            params.append(asserted_after)
        if asserted_before is not None:
            clauses.append("asserted_at < ?")
            params.append(asserted_before)
        params.append(limit)
        rows = self._conn.execute(
            f"""
            SELECT id, type, content, asserted_at, valid_from, valid_until, provenance,
                   extraction_count, retrieval_count, agent_id
            FROM knowledge_units
            WHERE {" AND ".join(clauses)}
            ORDER BY asserted_at DESC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
        return [(self._row_to_unit(r), r["agent_id"]) for r in rows]

    def get_unit_for_user(self, user_id: str, unit_id: str) -> tuple[KnowledgeUnit, str] | None:
        """One unit by id, only if it lives in one of ``user_id``'s scopes."""
        row = self._conn.execute(
            """
            SELECT id, type, content, asserted_at, valid_from, valid_until, provenance,
                   extraction_count, retrieval_count, agent_id
            FROM knowledge_units
            WHERE id = ? AND (agent_id = ? OR agent_id GLOB ?)
            LIMIT 1
            """,
            (unit_id, user_id, f"{user_id}/*"),
        ).fetchone()
        return (self._row_to_unit(row), row["agent_id"]) if row else None

    def active_correction_ids(self, source_id: str, agent_id: str) -> list[str]:
        """Ids of active units that were recorded as a correction of ``source_id``.

        A correction unit carries ``provenance = "correction of {source_id}"``
        (see proxy/memory_ops.apply_memory_op). Returns the still-active ones
        (``valid_until IS NULL``) in the given scope so the correction path can
        supersede any prior active correction of the same source and keep at
        most one active descendant per source.
        """
        rows = self._conn.execute(
            """
            SELECT id FROM knowledge_units
            WHERE agent_id = ? AND provenance = ? AND valid_until IS NULL
            """,
            (agent_id, f"correction of {source_id}"),
        ).fetchall()
        return [row["id"] for row in rows]

    def raw_search_units_for_user(
        self,
        user_id: str,
        query: str,
        *,
        limit: int = 10,
        as_of: str | None = None,
    ) -> list[tuple[KnowledgeUnit, str]]:
        """Tokenized LIKE search over a user's active units — no traversal, no
        budget. Answers "does this fact exist in the store at all?".

        A unit surfaces when it contains ANY of the query's content words
        (ranked overlap, accepting each term's surface form or its lemma),
        ordered by how many distinct content terms it matches — NOT a
        conjunction of every term, and NOT the whole query as one contiguous
        substring. Requiring EVERY word made any natural-language probe
        carrying an extra token (e.g. "applications submitted Acme Foundation",
        "Example Terrace tracking") return 0 whenever one token wasn't
        literally present, hiding units that plainly exist and forcing the
        nightly analyst to fall back to production replay on every probe.
        Deliberately scans the raw ``content`` column instead of the FTS
        index: this is the analyst's ground-truth existence probe, so it must
        not depend on the index's coverage or ranking.

        ``as_of`` (ISO-8601) excludes units asserted after that instant so a
        counterfactual never surfaces a memory that post-dates the turn under
        analysis.
        """
        base_clauses = [
            "(agent_id = ? OR agent_id GLOB ?)",
            "(valid_until IS NULL OR valid_until > ?)",
        ]
        base_params: list[object] = [
            user_id,
            f"{user_id}/*",
            datetime.now().isoformat(),
        ]
        if as_of is not None:
            base_clauses.append("asserted_at <= ?")
            base_params.append(as_of)

        terms = list(dict.fromkeys(t.lower() for t in tokenize(query) if t.lower() not in STOPWORDS))
        if terms:
            # One substring fragment per content term (surface form, plus the
            # lemma as an alternative when it differs — "applications" misses a
            # unit saying "application", while the lemma usually doesn't).
            term_frags: list[str] = []
            term_params: list[object] = []
            for surface in terms:
                lemma = lemmatize(surface)
                if lemma != surface:
                    term_frags.append("(LOWER(content) LIKE ? OR LOWER(content) LIKE ?)")
                    term_params.extend((f"%{surface}%", f"%{lemma}%"))
                else:
                    term_frags.append("LOWER(content) LIKE ?")
                    term_params.append(f"%{surface}%")
            # WHERE keeps any unit matching at least one term (OR), then ORDER
            # BY the number of matched terms DESC so the best overlap ranks
            # first. Both the OR filter and the score reference the same
            # fragments, so their params are appended in that textual order.
            where_terms = "(" + " OR ".join(term_frags) + ")"
            score_sql = " + ".join(f"(CASE WHEN {f} THEN 1 ELSE 0 END)" for f in term_frags)
            where_sql = " AND ".join([*base_clauses, where_terms])
            order_sql = f"({score_sql}) DESC, asserted_at DESC"
            params: list[object] = [*base_params, *term_params, *term_params, limit]
        else:
            # All-stopword or token-free queries keep the whole-string match.
            where_sql = " AND ".join([*base_clauses, "LOWER(content) LIKE LOWER(?)"])
            order_sql = "asserted_at DESC"
            params = [*base_params, f"%{query}%", limit]

        rows = self._conn.execute(
            f"""
            SELECT id, type, content, asserted_at, valid_from, valid_until, provenance,
                   extraction_count, retrieval_count, agent_id
            FROM knowledge_units
            WHERE {where_sql}
            ORDER BY {order_sql}
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
        return [(self._row_to_unit(r), r["agent_id"]) for r in rows]

    def invalidate(self, unit_id: str, at: datetime, agent_id: str) -> None:
        self._conn.execute(
            "UPDATE knowledge_units SET valid_until = ? WHERE id = ? AND agent_id = ?",
            (at.isoformat(), unit_id, agent_id),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Reconcile / compaction support (reconcile/runner.py). The nightly
    # reconcile physically deletes superseded units; these helpers keep the
    # cascade in one place so no FTS / embedding / topic / mention / relation
    # row can outlive the unit it describes.
    # ------------------------------------------------------------------

    def list_user_ids(self) -> list[str]:
        """Every user id that owns at least one unit (the scope prefix before
        ``/``; a legacy bare scope like ``default`` is its own user)."""
        rows = self._conn.execute("SELECT DISTINCT agent_id FROM knowledge_units").fetchall()
        return sorted({str(r["agent_id"]).split("/", 1)[0] for r in rows})

    def user_scopes(self, user_id: str) -> list[str]:
        """The distinct unit scopes (``agent_id`` values) ``user_id`` owns."""
        rows = self._conn.execute(
            "SELECT DISTINCT agent_id FROM knowledge_units WHERE agent_id = ? OR agent_id GLOB ? ORDER BY agent_id",
            (user_id, f"{user_id}/*"),
        ).fetchall()
        return [r["agent_id"] for r in rows]

    def scope_units(self, agent_id: str, *, active: bool) -> list[tuple[KnowledgeUnit, int]]:
        """``(unit, rowid)`` for one scope, oldest row first.

        ``active=True`` → units valid now (``valid_until`` unset or in the
        future); ``active=False`` → units whose ``valid_until`` has passed.
        The rowid is the incremental-scan cursor the reconcile job advances.
        """
        if active:
            validity = "(valid_until IS NULL OR valid_until > ?)"
        else:
            validity = "(valid_until IS NOT NULL AND valid_until <= ?)"
        rows = self._conn.execute(
            f"""
            SELECT rowid AS rid, id, type, content, asserted_at, valid_from, valid_until, provenance,
                   extraction_count, retrieval_count
            FROM knowledge_units
            WHERE agent_id = ? AND {validity}
            ORDER BY rowid
            """,  # noqa: S608 — validity is one of two fixed literals
            (agent_id, datetime.now().isoformat()),
        ).fetchall()
        return [(self._row_to_unit(r), int(r["rid"])) for r in rows]

    def units_by_provenance(self, agent_id: str, provenance: str) -> list[KnowledgeUnit]:
        """Every unit (active or not) in ``agent_id`` with exactly ``provenance``."""
        rows = self._conn.execute(
            """
            SELECT id, type, content, asserted_at, valid_from, valid_until, provenance,
                   extraction_count, retrieval_count
            FROM knowledge_units WHERE agent_id = ? AND provenance = ?
            ORDER BY asserted_at, rowid
            """,
            (agent_id, provenance),
        ).fetchall()
        return [self._row_to_unit(r) for r in rows]

    def scope_unit_embeddings(self, agent_id: str, unit_ids: list[str]) -> dict[str, bytes]:
        """Stored unit embeddings for ``unit_ids`` in exactly one scope."""
        return self.load_item_embeddings(unit_ids, (agent_id,))

    def fts_unit_neighbours(self, content: str, agent_id: str, limit: int) -> list[str]:
        """Unit ids in ``agent_id`` whose indexed text best matches ``content``
        (BM25 over the content words, OR-joined). Empty when FTS is off."""
        if not self.fts_enabled or limit <= 0:
            return []
        words = sorted(content_words(content))
        if not words:
            return []
        expression = " OR ".join(f'"{w}"' for w in words)
        try:
            rows = self._conn.execute(
                f"""
                SELECT item_id FROM {self._fts_schema}.memento_fts
                WHERE memento_fts MATCH ? AND agent_id = ? AND kind = 'unit'
                ORDER BY bm25(memento_fts) ASC, item_id ASC
                LIMIT ?
                """,  # noqa: S608 — schema name is a module constant
                (expression, agent_id, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []  # malformed MATCH expression — the other signals still run
        return [r["item_id"] for r in rows]

    def purge_units(self, unit_ids: list[str], agent_id: str) -> dict[str, int]:
        """PHYSICALLY delete units of one scope and every row that hangs off them.

        Cascade (one transaction across the canonical DB and both sidecars):
        - ``knowledge_units`` rows (scope-checked: an id outside ``agent_id``
          is left alone);
        - their FTS rows and stored embeddings;
        - ``unit_topics`` tags and ``entity_mentions`` whose source is the unit;
        - relations with the unit as subject or object (``mentions`` /
          ``about`` / ``supersedes`` links — they describe the unit, so they
          go with it) plus those relations' stored embeddings.
        Entities are never deleted: they are standalone knowledge, and any
        entity still referenced by another unit or relation stays intact.
        Returns per-table delete counts for the audit log.
        """
        counts = {"units": 0, "fts": 0, "embeddings": 0, "topics": 0, "mentions": 0, "relations": 0}
        if not unit_ids:
            return counts
        marks = ",".join("?" for _ in unit_ids)
        with self._conn:
            present = [
                r["id"]
                for r in self._conn.execute(
                    f"SELECT id FROM knowledge_units WHERE agent_id = ? AND id IN ({marks})",  # noqa: S608
                    (agent_id, *unit_ids),
                ).fetchall()
            ]
            if not present:
                return counts
            pm = ",".join("?" for _ in present)
            relation_ids = [
                r["id"]
                for r in self._conn.execute(
                    f"SELECT id FROM relations WHERE subject_id IN ({pm}) OR object_id IN ({pm})",  # noqa: S608
                    (*present, *present),
                ).fetchall()
            ]
            counts["units"] = self._conn.execute(
                f"DELETE FROM knowledge_units WHERE agent_id = ? AND id IN ({pm})",  # noqa: S608
                (agent_id, *present),
            ).rowcount
            counts["topics"] = self._conn.execute(
                f"DELETE FROM unit_topics WHERE unit_id IN ({pm})",  # noqa: S608
                tuple(present),
            ).rowcount
            counts["mentions"] = self._conn.execute(
                f"DELETE FROM entity_mentions WHERE source_kind IN ('knowledge_unit', 'unit') "  # noqa: S608
                f"AND source_id IN ({pm})",
                tuple(present),
            ).rowcount
            if relation_ids:
                rm = ",".join("?" for _ in relation_ids)
                counts["relations"] = self._conn.execute(
                    f"DELETE FROM relations WHERE id IN ({rm})",  # noqa: S608
                    tuple(relation_ids),
                ).rowcount
            embedded = [*present, *relation_ids]
            em = ",".join("?" for _ in embedded)
            counts["embeddings"] = self._conn.execute(
                f"DELETE FROM {self._emb_schema}.item_embeddings WHERE item_id IN ({em})",  # noqa: S608
                tuple(embedded),
            ).rowcount
            if self.fts_enabled:
                counts["fts"] = self._conn.execute(
                    f"DELETE FROM {self._fts_schema}.memento_fts WHERE item_id IN ({pm})",  # noqa: S608
                    tuple(present),
                ).rowcount
        return counts

    def vacuum(self) -> None:
        """Reclaim the space freed by ``purge_units`` in every attached DB.

        Must run outside a transaction. The FTS index is ``optimize``d first
        so its segment b-trees are merged before the file is rewritten.
        """
        self._conn.commit()
        if self.fts_enabled:
            self._conn.execute(f"INSERT INTO {self._fts_schema}.memento_fts(memento_fts) VALUES ('optimize')")
            self._conn.commit()
            self._conn.execute(f"VACUUM {self._fts_schema}")
        self._conn.execute(f"VACUUM {self._emb_schema}")
        self._conn.execute("VACUUM main")

    def backup_to(self, directory: Path) -> list[Path]:
        """Consistent online snapshot of the canonical DB and its embedding
        sidecar via the sqlite backup API (never a raw file copy — a copy
        taken mid-write can be torn). The FTS sidecar is derived data and is
        rebuilt on open, so it is not snapshotted. Returns the written paths.
        """
        directory.mkdir(parents=True, exist_ok=True)
        self._conn.commit()
        written: list[Path] = []
        for schema, name in (("main", self.db_path.name), (self._emb_schema, f"{self.db_path.name}.emb")):
            target = directory / name
            dest = sqlite3.connect(str(target))
            try:
                self._conn.backup(dest, name=schema)
            finally:
                dest.close()
            written.append(target)
        return written

    def add_entity(self, entity: Entity, agent_id: str, source_agent_id: str | None = None) -> None:
        self._conn.execute(
            """
            INSERT OR IGNORE INTO entities
                (id, label, type, subtype, canonical_label, canonical_key,
                 wikidata_id, description, wikipedia_url, asserted_at, agent_id, source_agent_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entity.id,
                entity.label,
                entity.type,
                entity.subtype,
                entity.canonical_label or entity.label,
                entity.canonical_key,
                entity.wikidata_id,
                entity.description,
                entity.wikipedia_url,
                entity.asserted_at.isoformat(),
                agent_id,
                source_agent_id,
            ),
        )
        self._fts_sync_entity(entity.id)
        self._conn.commit()

    def get_entity_by_label(self, label: str, agent_id: str) -> Entity | None:
        row = self._conn.execute(
            """
            SELECT id, label, type, subtype, canonical_label, canonical_key,
                   wikidata_id, description, wikipedia_url, asserted_at,
                   extraction_count, retrieval_count
            FROM entities WHERE agent_id = ? AND LOWER(label) = LOWER(?) LIMIT 1
            """,
            (agent_id, label),
        ).fetchone()
        return self._row_to_entity(row) if row else None

    def get_entity_by_wikidata_id(self, wikidata_id: str, agent_id: str) -> Entity | None:
        row = self._conn.execute(
            """
            SELECT id, label, type, subtype, canonical_label, canonical_key,
                   wikidata_id, description, wikipedia_url, asserted_at,
                   extraction_count, retrieval_count
            FROM entities WHERE agent_id = ? AND wikidata_id = ? LIMIT 1
            """,
            (agent_id, wikidata_id),
        ).fetchone()
        return self._row_to_entity(row) if row else None

    def get_entity_by_id(self, entity_id: str, agent_id: str) -> Entity | None:
        row = self._conn.execute(
            """
            SELECT id, label, type, subtype, canonical_label, canonical_key,
                   wikidata_id, description, wikipedia_url, asserted_at,
                   extraction_count, retrieval_count
            FROM entities WHERE agent_id = ? AND id = ? LIMIT 1
            """,
            (agent_id, entity_id),
        ).fetchone()
        return self._row_to_entity(row) if row else None

    def get_entity_by_canonical_key(
        self, canonical_key: str, agent_id: str, entity_type: str | None = None
    ) -> Entity | None:
        if entity_type is None:
            row = self._conn.execute(
                """
                SELECT id, label, type, subtype, canonical_label, canonical_key,
                       wikidata_id, description, wikipedia_url, asserted_at,
                       extraction_count, retrieval_count
                FROM entities
                WHERE agent_id = ? AND canonical_key = ?
                LIMIT 1
                """,
                (agent_id, canonical_key),
            ).fetchone()
        else:
            row = self._conn.execute(
                """
                SELECT id, label, type, subtype, canonical_label, canonical_key,
                       wikidata_id, description, wikipedia_url, asserted_at,
                       extraction_count, retrieval_count
                FROM entities
                WHERE agent_id = ? AND type = ? AND canonical_key = ?
                LIMIT 1
                """,
                (agent_id, entity_type, canonical_key),
            ).fetchone()
        return self._row_to_entity(row) if row else None

    def list_entities(self, agent_id: str, entity_type: str | None = None) -> list[Entity]:
        if entity_type is None:
            rows = self._conn.execute(
                """
                SELECT id, label, type, subtype, canonical_label, canonical_key,
                       wikidata_id, description, wikipedia_url, asserted_at,
                       extraction_count, retrieval_count
                FROM entities
                WHERE agent_id = ?
                ORDER BY label
                """,
                (agent_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT id, label, type, subtype, canonical_label, canonical_key,
                       wikidata_id, description, wikipedia_url, asserted_at,
                       extraction_count, retrieval_count
                FROM entities
                WHERE agent_id = ? AND type = ?
                ORDER BY label
                """,
                (agent_id, entity_type),
            ).fetchall()
        return [self._row_to_entity(r) for r in rows]

    def upsert_entity_alias(self, alias: EntityAlias, agent_id: str) -> None:
        if not alias.id:
            alias.id = str(uuid.uuid4())
        self._conn.execute(
            """
            INSERT INTO entity_aliases
                (id, entity_id, alias, alias_key, source, created_at, agent_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(agent_id, alias_key) DO UPDATE SET
                entity_id = excluded.entity_id,
                alias = excluded.alias,
                source = excluded.source,
                created_at = excluded.created_at
            """,
            (
                alias.id,
                alias.entity_id,
                alias.alias,
                self._identity_key(alias.alias),
                alias.source,
                alias.created_at.isoformat(),
                agent_id,
            ),
        )
        self._conn.commit()

    def get_entity_alias(self, alias_id: str, agent_id: str) -> EntityAlias | None:
        row = self._conn.execute(
            """
            SELECT id, entity_id, alias, source, created_at
            FROM entity_aliases
            WHERE agent_id = ? AND id = ?
            LIMIT 1
            """,
            (agent_id, alias_id),
        ).fetchone()
        return self._row_to_entity_alias(row) if row else None

    def list_entity_aliases(self, entity_id: str, agent_id: str) -> list[EntityAlias]:
        rows = self._conn.execute(
            """
            SELECT id, entity_id, alias, source, created_at
            FROM entity_aliases
            WHERE agent_id = ? AND entity_id = ?
            ORDER BY alias
            """,
            (agent_id, entity_id),
        ).fetchall()
        return [self._row_to_entity_alias(r) for r in rows]

    def find_entity_aliases(self, alias: str, agent_id: str, entity_type: str | None = None) -> list[EntityAlias]:
        if entity_type is None:
            rows = self._conn.execute(
                """
                SELECT id, entity_id, alias, source, created_at
                FROM entity_aliases
                WHERE agent_id = ? AND alias_key = ?
                ORDER BY alias
                """,
                (agent_id, self._identity_key(alias)),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT ea.id, ea.entity_id, ea.alias, ea.source, ea.created_at
                FROM entity_aliases ea
                JOIN entities e ON e.id = ea.entity_id AND e.agent_id = ea.agent_id
                WHERE ea.agent_id = ? AND ea.alias_key = ? AND e.type = ?
                ORDER BY ea.alias
                """,
                (agent_id, self._identity_key(alias), entity_type),
            ).fetchall()
        return [self._row_to_entity_alias(r) for r in rows]

    def delete_entity_alias(self, alias_id: str, agent_id: str) -> None:
        self._conn.execute("DELETE FROM entity_aliases WHERE id = ? AND agent_id = ?", (alias_id, agent_id))
        self._conn.commit()

    def upsert_entity_identifier(self, identifier: EntityIdentifier, agent_id: str) -> None:
        if not identifier.id:
            identifier.id = str(uuid.uuid4())
        self._conn.execute(
            """
            INSERT INTO entity_identifiers
                (id, entity_id, namespace, value, source, created_at, agent_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(agent_id, namespace, value) DO UPDATE SET
                entity_id = excluded.entity_id,
                source = excluded.source,
                created_at = excluded.created_at
            """,
            (
                identifier.id,
                identifier.entity_id,
                identifier.namespace,
                identifier.value,
                identifier.source,
                identifier.created_at.isoformat(),
                agent_id,
            ),
        )
        self._conn.commit()

    def get_entity_identifier(self, identifier_id: str, agent_id: str) -> EntityIdentifier | None:
        row = self._conn.execute(
            """
            SELECT id, entity_id, namespace, value, source, created_at
            FROM entity_identifiers
            WHERE agent_id = ? AND id = ?
            LIMIT 1
            """,
            (agent_id, identifier_id),
        ).fetchone()
        return self._row_to_entity_identifier(row) if row else None

    def list_entity_identifiers(self, entity_id: str, agent_id: str) -> list[EntityIdentifier]:
        rows = self._conn.execute(
            """
            SELECT id, entity_id, namespace, value, source, created_at
            FROM entity_identifiers
            WHERE agent_id = ? AND entity_id = ?
            ORDER BY namespace, value
            """,
            (agent_id, entity_id),
        ).fetchall()
        return [self._row_to_entity_identifier(r) for r in rows]

    def find_entity_identifiers(
        self, namespace: str, value: str, agent_id: str, entity_type: str | None = None
    ) -> list[EntityIdentifier]:
        if entity_type is None:
            rows = self._conn.execute(
                """
                SELECT id, entity_id, namespace, value, source, created_at
                FROM entity_identifiers
                WHERE agent_id = ? AND namespace = ? AND value = ?
                ORDER BY namespace, value
                """,
                (agent_id, namespace, value),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT ei.id, ei.entity_id, ei.namespace, ei.value, ei.source, ei.created_at
                FROM entity_identifiers ei
                JOIN entities e ON e.id = ei.entity_id AND e.agent_id = ei.agent_id
                WHERE ei.agent_id = ? AND ei.namespace = ? AND ei.value = ? AND e.type = ?
                ORDER BY ei.namespace, ei.value
                """,
                (agent_id, namespace, value, entity_type),
            ).fetchall()
        return [self._row_to_entity_identifier(r) for r in rows]

    def delete_entity_identifier(self, identifier_id: str, agent_id: str) -> None:
        self._conn.execute(
            "DELETE FROM entity_identifiers WHERE id = ? AND agent_id = ?",
            (identifier_id, agent_id),
        )
        self._conn.commit()

    def upsert_entity_attribute(self, attribute: EntityAttribute, agent_id: str) -> None:
        if not attribute.id:
            attribute.id = str(uuid.uuid4())
        self._conn.execute(
            """
            INSERT OR REPLACE INTO entity_attributes
                (id, entity_id, attribute_type, value, asserted_at,
                 valid_from, valid_until, provenance, agent_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                attribute.id,
                attribute.entity_id,
                attribute.attribute_type,
                attribute.value,
                attribute.asserted_at.isoformat(),
                attribute.valid_from.isoformat(),
                attribute.valid_until.isoformat() if attribute.valid_until else None,
                attribute.provenance,
                agent_id,
            ),
        )
        self._conn.commit()

    def supersede_entity_attributes(self, entity_id: str, attribute_type: str, at: datetime, agent_id: str) -> int:
        cur = self._conn.execute(
            """
            UPDATE entity_attributes
            SET valid_until = ?
            WHERE agent_id = ? AND entity_id = ? AND attribute_type = ? AND valid_until IS NULL
            """,
            (at.isoformat(), agent_id, entity_id, attribute_type),
        )
        self._conn.commit()
        return cur.rowcount

    def get_entity_attribute(self, attribute_id: str, agent_id: str) -> EntityAttribute | None:
        row = self._conn.execute(
            """
            SELECT id, entity_id, attribute_type, value, asserted_at,
                   valid_from, valid_until, provenance
            FROM entity_attributes
            WHERE agent_id = ? AND id = ?
            LIMIT 1
            """,
            (agent_id, attribute_id),
        ).fetchone()
        return self._row_to_entity_attribute(row) if row else None

    def list_entity_attributes(
        self,
        entity_id: str,
        agent_id: str,
        attribute_type: str | None = None,
        active_only: bool = False,
    ) -> list[EntityAttribute]:
        conditions = ["agent_id = ?", "entity_id = ?"]
        params: list[str] = [agent_id, entity_id]
        if attribute_type is not None:
            conditions.append("attribute_type = ?")
            params.append(attribute_type)
        if active_only:
            conditions.append("valid_until IS NULL")
        where_clause = " AND ".join(conditions)
        rows = self._conn.execute(
            f"""
            SELECT id, entity_id, attribute_type, value, asserted_at,
                   valid_from, valid_until, provenance
            FROM entity_attributes
            WHERE {where_clause}
            ORDER BY asserted_at DESC
            """,
            params,
        ).fetchall()
        return [self._row_to_entity_attribute(r) for r in rows]

    def find_entity_attributes(
        self,
        attribute_type: str,
        agent_id: str,
        value: str | None = None,
        active_only: bool = True,
    ) -> list[EntityAttribute]:
        conditions = ["agent_id = ?", "attribute_type = ?"]
        params: list[str] = [agent_id, attribute_type]
        if value is not None:
            conditions.append("value = ?")
            params.append(value)
        if active_only:
            conditions.append("valid_until IS NULL")
        where_clause = " AND ".join(conditions)
        rows = self._conn.execute(
            f"""
            SELECT id, entity_id, attribute_type, value, asserted_at,
                   valid_from, valid_until, provenance
            FROM entity_attributes
            WHERE {where_clause}
            ORDER BY asserted_at DESC
            """,
            params,
        ).fetchall()
        return [self._row_to_entity_attribute(r) for r in rows]

    def delete_entity_attribute(self, attribute_id: str, agent_id: str) -> None:
        self._conn.execute(
            "DELETE FROM entity_attributes WHERE id = ? AND agent_id = ?",
            (attribute_id, agent_id),
        )
        self._conn.commit()

    def upsert_entity_mention(self, mention: EntityMention, agent_id: str) -> None:
        if not mention.id:
            mention.id = str(uuid.uuid4())
        self._conn.execute(
            """
            INSERT INTO entity_mentions
                (id, entity_id, source_kind, source_id, snippet, asserted_at, agent_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(agent_id, entity_id, source_kind, source_id) DO UPDATE SET
                snippet = excluded.snippet,
                asserted_at = excluded.asserted_at
            """,
            (
                mention.id,
                mention.entity_id,
                mention.source_kind,
                mention.source_id,
                mention.snippet,
                mention.asserted_at.isoformat(),
                agent_id,
            ),
        )
        self._conn.commit()

    def get_entity_mention(self, mention_id: str, agent_id: str) -> EntityMention | None:
        row = self._conn.execute(
            """
            SELECT id, entity_id, source_kind, source_id, snippet, asserted_at
            FROM entity_mentions
            WHERE agent_id = ? AND id = ?
            LIMIT 1
            """,
            (agent_id, mention_id),
        ).fetchone()
        return self._row_to_entity_mention(row) if row else None

    def list_entity_mentions(self, entity_id: str, agent_id: str) -> list[EntityMention]:
        rows = self._conn.execute(
            """
            SELECT id, entity_id, source_kind, source_id, snippet, asserted_at
            FROM entity_mentions
            WHERE agent_id = ? AND entity_id = ?
            ORDER BY asserted_at DESC
            """,
            (agent_id, entity_id),
        ).fetchall()
        return [self._row_to_entity_mention(r) for r in rows]

    def find_entity_mentions_by_source(self, source_kind: str, source_id: str, agent_id: str) -> list[EntityMention]:
        rows = self._conn.execute(
            """
            SELECT id, entity_id, source_kind, source_id, snippet, asserted_at
            FROM entity_mentions
            WHERE agent_id = ? AND source_kind = ? AND source_id = ?
            ORDER BY asserted_at DESC
            """,
            (agent_id, source_kind, source_id),
        ).fetchall()
        return [self._row_to_entity_mention(r) for r in rows]

    def delete_entity_mention(self, mention_id: str, agent_id: str) -> None:
        self._conn.execute("DELETE FROM entity_mentions WHERE id = ? AND agent_id = ?", (mention_id, agent_id))
        self._conn.commit()

    def upsert_entity_type(self, entity_type: EntityType, agent_id: str) -> None:
        if not entity_type.id:
            entity_type.id = str(uuid.uuid4())
        self._conn.execute(
            """
            INSERT INTO entity_types
                (id, name, name_key, description, source, created_at, agent_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(agent_id, name_key) DO UPDATE SET
                name = excluded.name,
                description = excluded.description,
                source = excluded.source,
                created_at = excluded.created_at
            """,
            (
                entity_type.id,
                entity_type.name,
                self._identity_key(entity_type.name),
                entity_type.description,
                entity_type.source,
                entity_type.created_at.isoformat(),
                agent_id,
            ),
        )
        self._conn.commit()

    def get_entity_type(self, entity_type_id: str, agent_id: str) -> EntityType | None:
        row = self._conn.execute(
            """
            SELECT id, name, description, source, created_at
            FROM entity_types
            WHERE id = ? AND agent_id IN ('default', ?)
            LIMIT 1
            """,
            (entity_type_id, agent_id),
        ).fetchone()
        return self._row_to_entity_type(row) if row else None

    def get_entity_type_by_name(self, name: str, agent_id: str) -> EntityType | None:
        row = self._conn.execute(
            """
            SELECT id, name, description, source, created_at
            FROM entity_types
            WHERE name_key = ? AND agent_id IN ('default', ?)
            ORDER BY CASE WHEN agent_id = ? THEN 0 ELSE 1 END
            LIMIT 1
            """,
            (self._identity_key(name), agent_id, agent_id),
        ).fetchone()
        return self._row_to_entity_type(row) if row else None

    def list_entity_types(self, agent_id: str) -> list[EntityType]:
        rows = self._conn.execute(
            """
            SELECT id, name, description, source, created_at
            FROM entity_types
            WHERE agent_id IN ('default', ?)
            ORDER BY name
            """,
            (agent_id,),
        ).fetchall()
        return [self._row_to_entity_type(r) for r in rows]

    def delete_entity_type(self, entity_type_id: str, agent_id: str) -> None:
        self._conn.execute("DELETE FROM entity_types WHERE id = ? AND agent_id = ?", (entity_type_id, agent_id))
        self._conn.commit()

    def upsert_attribute_type(self, attribute_type: AttributeType, agent_id: str) -> None:
        if not attribute_type.id:
            attribute_type.id = str(uuid.uuid4())
        self._conn.execute(
            """
            INSERT INTO attribute_types
                (id, name, name_key, description, value_type, source, created_at, agent_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(agent_id, name_key) DO UPDATE SET
                name = excluded.name,
                description = excluded.description,
                value_type = excluded.value_type,
                source = excluded.source,
                created_at = excluded.created_at
            """,
            (
                attribute_type.id,
                attribute_type.name,
                self._identity_key(attribute_type.name),
                attribute_type.description,
                attribute_type.value_type,
                attribute_type.source,
                attribute_type.created_at.isoformat(),
                agent_id,
            ),
        )
        self._conn.commit()

    def get_attribute_type(self, attribute_type_id: str, agent_id: str) -> AttributeType | None:
        row = self._conn.execute(
            """
            SELECT id, name, description, value_type, source, created_at
            FROM attribute_types
            WHERE id = ? AND agent_id IN ('default', ?)
            LIMIT 1
            """,
            (attribute_type_id, agent_id),
        ).fetchone()
        return self._row_to_attribute_type(row) if row else None

    def get_attribute_type_by_name(self, name: str, agent_id: str) -> AttributeType | None:
        row = self._conn.execute(
            """
            SELECT id, name, description, value_type, source, created_at
            FROM attribute_types
            WHERE name_key = ? AND agent_id IN ('default', ?)
            ORDER BY CASE WHEN agent_id = ? THEN 0 ELSE 1 END
            LIMIT 1
            """,
            (self._identity_key(name), agent_id, agent_id),
        ).fetchone()
        return self._row_to_attribute_type(row) if row else None

    def list_attribute_types(self, agent_id: str) -> list[AttributeType]:
        rows = self._conn.execute(
            """
            SELECT id, name, description, value_type, source, created_at
            FROM attribute_types
            WHERE agent_id IN ('default', ?)
            ORDER BY name
            """,
            (agent_id,),
        ).fetchall()
        return [self._row_to_attribute_type(r) for r in rows]

    def delete_attribute_type(self, attribute_type_id: str, agent_id: str) -> None:
        self._conn.execute(
            "DELETE FROM attribute_types WHERE id = ? AND agent_id = ?",
            (attribute_type_id, agent_id),
        )
        self._conn.commit()

    def add_relation(self, relation: Relation, agent_id: str, source_agent_id: str | None = None) -> None:
        if not relation.id:
            relation.id = str(uuid.uuid4())
        # INSERT OR IGNORE so the partial unique index on active rows
        # (idx_relations_unique_active) silently drops re-inserts of the same
        # active (agent_id, subject_id, predicate, object_id) triple.
        self._conn.execute(
            """
            INSERT OR IGNORE INTO relations
                (id, subject_id, predicate, object_id, asserted_at,
                 valid_from, valid_until, provenance, agent_id, source_agent_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                relation.id,
                relation.subject_id,
                relation.predicate,
                relation.object_id,
                relation.asserted_at.isoformat(),
                relation.valid_from.isoformat(),
                relation.valid_until.isoformat() if relation.valid_until else None,
                relation.provenance,
                agent_id,
                source_agent_id,
            ),
        )
        self._conn.commit()

    # Maps a caller-friendly "kind" to the physical table carrying reinforcement counters.
    _COUNTER_TABLES: dict[str, str] = {
        "unit": "knowledge_units",
        "knowledge_unit": "knowledge_units",
        "knowledge_units": "knowledge_units",
        "entity": "entities",
        "entities": "entities",
        "relation": "relations",
        "relations": "relations",
    }
    # SQLite caps variables per statement at 999; leave headroom for the trailing
    # amount/agent_id params, so chunk id batches well under the limit.
    _COUNTER_ID_CHUNK = 900

    def _increment_counter(self, table_kind: str, column: str, ids: list[str], agent_id: str, amount: int) -> None:
        """Batched, agent-scoped UPDATE that bumps `column` for the given ids.

        One parameterised `UPDATE ... WHERE id IN (...) AND agent_id = ?` per chunk.
        """
        table = self._COUNTER_TABLES.get(table_kind)
        if table is None:
            raise ValueError(f"unknown table_kind {table_kind!r}; expected one of {sorted(set(self._COUNTER_TABLES))}")
        if not ids or amount == 0:
            return
        for start in range(0, len(ids), self._COUNTER_ID_CHUNK):
            chunk = ids[start : start + self._COUNTER_ID_CHUNK]
            placeholders = ",".join("?" for _ in chunk)
            self._conn.execute(
                f"UPDATE {table} SET {column} = {column} + ? "  # noqa: S608 - table/column are fixed internal values
                f"WHERE id IN ({placeholders}) AND agent_id = ?",
                (amount, *chunk, agent_id),
            )
        self._conn.commit()

    def increment_extraction_count(self, table_kind: str, ids: list[str], agent_id: str, *, amount: int = 1) -> None:
        """Bump the write-side reinforcement counter for N ids (asserted/re-asserted by extraction)."""
        self._increment_counter(table_kind, "extraction_count", ids, agent_id, amount)

    def increment_retrieval_count(self, table_kind: str, ids: list[str], agent_id: str, *, amount: int = 1) -> None:
        """Bump the read-side usage counter for N ids (surfaced in a briefing)."""
        self._increment_counter(table_kind, "retrieval_count", ids, agent_id, amount)

    def get_relations(self, subject_id: str, agent_id: str) -> list[Relation]:
        rows = self._conn.execute(
            """
            SELECT id, subject_id, predicate, object_id, asserted_at,
                   valid_from, valid_until, provenance,
                   extraction_count, retrieval_count
            FROM relations WHERE subject_id = ? AND agent_id = ?
            ORDER BY asserted_at DESC
            """,
            (subject_id, agent_id),
        ).fetchall()
        return [self._row_to_relation(r) for r in rows]

    def invalidate_relation(self, relation_id: str, at: datetime, agent_id: str) -> None:
        self._conn.execute(
            "UPDATE relations SET valid_until = ? WHERE id = ? AND agent_id = ?",
            (at.isoformat(), relation_id, agent_id),
        )
        self._conn.commit()

    def add_relation_type(self, rt: RelationType) -> None:
        if not rt.id:
            rt.id = str(uuid.uuid4())
        self._conn.execute(
            """
            INSERT OR IGNORE INTO relation_types
                (id, name, description, source, created_at, is_functional)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (rt.id, rt.name, rt.description, rt.source, rt.created_at.isoformat(), 1 if rt.is_functional else 0),
        )
        self._conn.commit()

    def list_relation_types(self) -> list[RelationType]:
        rows = self._conn.execute(
            "SELECT id, name, description, source, created_at, is_functional FROM relation_types"
        ).fetchall()
        return [
            RelationType(
                id=r["id"],
                name=r["name"],
                description=r["description"],
                source=r["source"],
                created_at=datetime.fromisoformat(r["created_at"]),
                is_functional=bool(r["is_functional"]),
            )
            for r in rows
        ]

    def supersede_relations(self, subject_id: str, predicate: str, at: datetime, agent_id: str) -> int:
        """Invalidate all active (subject_id, predicate, agent_id) relations. Returns count invalidated."""
        cursor = self._conn.execute(
            """
            UPDATE relations SET valid_until = ?
            WHERE subject_id = ? AND predicate = ? AND agent_id = ?
              AND (valid_until IS NULL OR valid_until > ?)
            """,
            (at.isoformat(), subject_id, predicate, agent_id, at.isoformat()),
        )
        self._conn.commit()
        return cursor.rowcount

    def add_topic(self, topic: Topic) -> None:
        if not topic.id:
            topic.id = str(uuid.uuid4())
        self._conn.execute(
            """
            INSERT OR IGNORE INTO topics
                (id, name, description, source, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (topic.id, topic.name, topic.description, topic.source, topic.created_at.isoformat()),
        )
        self._conn.commit()

    def list_topics(self) -> list[Topic]:
        rows = self._conn.execute("SELECT id, name, description, source, created_at FROM topics").fetchall()
        return [
            Topic(
                id=r["id"],
                name=r["name"],
                description=r["description"],
                source=r["source"],
                created_at=datetime.fromisoformat(r["created_at"]),
            )
            for r in rows
        ]

    def tag_unit(self, unit_id: str, topic_id: str, agent_id: str) -> None:
        resolved = self._resolve_topic_id(topic_id)
        if resolved is None:
            return
        self._conn.execute(
            "INSERT OR IGNORE INTO unit_topics (agent_id, unit_id, topic_id) VALUES (?, ?, ?)",
            (agent_id, unit_id, resolved),
        )
        self._conn.commit()

    def tag_entity(self, entity_id: str, topic_id: str, agent_id: str) -> None:
        resolved = self._resolve_topic_id(topic_id)
        if resolved is None:
            return
        self._conn.execute(
            "INSERT OR IGNORE INTO entity_topics (agent_id, entity_id, topic_id) VALUES (?, ?, ?)",
            (agent_id, entity_id, resolved),
        )
        self._conn.commit()

    def _resolve_topic_id(self, topic_ref: str) -> str | None:
        """Accept either a topic id or a topic name; return the id."""
        row = self._conn.execute(
            "SELECT id FROM topics WHERE id = ? OR LOWER(name) = LOWER(?) LIMIT 1",
            (topic_ref, topic_ref),
        ).fetchone()
        return row["id"] if row else None

    @staticmethod
    def _row_to_entity(row: sqlite3.Row) -> Entity:
        return Entity(
            id=row["id"],
            label=row["label"],
            type=row["type"],
            subtype=row["subtype"],
            canonical_label=row["canonical_label"],
            canonical_key=row["canonical_key"],
            wikidata_id=row["wikidata_id"],
            description=row["description"],
            wikipedia_url=row["wikipedia_url"],
            asserted_at=datetime.fromisoformat(row["asserted_at"]),
            extraction_count=row["extraction_count"],
            retrieval_count=row["retrieval_count"],
        )

    @staticmethod
    def _row_to_entity_alias(row: sqlite3.Row) -> EntityAlias:
        return EntityAlias(
            id=row["id"],
            entity_id=row["entity_id"],
            alias=row["alias"],
            source=row["source"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    @staticmethod
    def _row_to_entity_identifier(row: sqlite3.Row) -> EntityIdentifier:
        return EntityIdentifier(
            id=row["id"],
            entity_id=row["entity_id"],
            namespace=row["namespace"],
            value=row["value"],
            source=row["source"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    @staticmethod
    def _row_to_entity_attribute(row: sqlite3.Row) -> EntityAttribute:
        vu = row["valid_until"]
        return EntityAttribute(
            id=row["id"],
            entity_id=row["entity_id"],
            attribute_type=row["attribute_type"],
            value=row["value"],
            asserted_at=datetime.fromisoformat(row["asserted_at"]),
            valid_from=datetime.fromisoformat(row["valid_from"]),
            valid_until=datetime.fromisoformat(vu) if vu else None,
            provenance=row["provenance"],
        )

    @staticmethod
    def _row_to_entity_mention(row: sqlite3.Row) -> EntityMention:
        return EntityMention(
            id=row["id"],
            entity_id=row["entity_id"],
            source_kind=row["source_kind"],
            source_id=row["source_id"],
            snippet=row["snippet"],
            asserted_at=datetime.fromisoformat(row["asserted_at"]),
        )

    @staticmethod
    def _row_to_entity_type(row: sqlite3.Row) -> EntityType:
        return EntityType(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            source=row["source"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    @staticmethod
    def _row_to_attribute_type(row: sqlite3.Row) -> AttributeType:
        return AttributeType(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            value_type=row["value_type"],
            source=row["source"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    @staticmethod
    def _row_to_relation(row: sqlite3.Row) -> Relation:
        vu = row["valid_until"]
        return Relation(
            id=row["id"],
            subject_id=row["subject_id"],
            predicate=row["predicate"],
            object_id=row["object_id"],
            asserted_at=datetime.fromisoformat(row["asserted_at"]),
            valid_from=datetime.fromisoformat(row["valid_from"]),
            valid_until=datetime.fromisoformat(vu) if vu else None,
            provenance=row["provenance"],
            extraction_count=row["extraction_count"],
            retrieval_count=row["retrieval_count"],
        )

    @staticmethod
    def _row_to_unit(row: sqlite3.Row) -> KnowledgeUnit:
        valid_until_raw = row["valid_until"]
        return KnowledgeUnit(
            id=row["id"],
            type=KnowledgeType(row["type"]),
            content=row["content"],
            asserted_at=datetime.fromisoformat(row["asserted_at"]),
            valid_from=datetime.fromisoformat(row["valid_from"]),
            valid_until=datetime.fromisoformat(valid_until_raw) if valid_until_raw else None,
            provenance=row["provenance"],
            extraction_count=row["extraction_count"],
            retrieval_count=row["retrieval_count"],
        )
