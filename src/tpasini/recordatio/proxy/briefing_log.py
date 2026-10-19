"""Sidecar SQLite store for briefing events.

Append-only log of every chat-turn briefing — i.e. the set of memories
Recordatio injected into the system prompt for that turn. Lives in a
dedicated database file (``briefing_events.db`` in the same directory as
``AGENTMEM_DB``) so historical briefing queries don't compete with the
cost log or the memory store for the same write lock.

Schema mirrors the proposal in issue #45: a single ``briefing_events``
table keyed by ``request_id`` (the per-turn ``call_id`` Recordatio generates)
with the recalled IDs stored as JSON arrays. The IDs are hydrated against
the main memory store at read time by the API endpoints, not duplicated
here. Returned content is current-store hydration, not an immutable
snapshot of what the agent saw. The API retains recorded IDs and counts
and reports hydration counts, unresolved IDs, errors and completeness.

Single connection with ``check_same_thread=False`` + a ``threading.Lock``
around writes. WAL mode lets concurrent readers proceed lockless. This is
the same threading model as ``cost_log.CostLog``.
"""

import contextlib
import json
import sqlite3
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS briefing_events (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                       TEXT    NOT NULL,
    request_id               TEXT    NOT NULL,
    agent_id                 TEXT,
    source_agent_id          TEXT,
    query_text               TEXT,
    topic_hint               TEXT,
    recalled_unit_ids        TEXT    NOT NULL DEFAULT '[]',
    recalled_entity_ids      TEXT    NOT NULL DEFAULT '[]',
    recalled_relation_ids    TEXT    NOT NULL DEFAULT '[]',
    count_units              INTEGER NOT NULL DEFAULT 0,
    count_entities           INTEGER NOT NULL DEFAULT 0,
    count_relations          INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS briefing_events_request ON briefing_events(request_id);
CREATE INDEX IF NOT EXISTS briefing_events_agent   ON briefing_events(agent_id, ts);
"""


@dataclass(frozen=True)
class BriefingEvent:
    """One row in the ``briefing_events`` table."""

    ts: datetime
    request_id: str
    agent_id: str | None = None
    source_agent_id: str | None = None
    query_text: str | None = None
    topic_hint: str | None = None
    recalled_unit_ids: list[str] = field(default_factory=list)
    recalled_entity_ids: list[str] = field(default_factory=list)
    recalled_relation_ids: list[str] = field(default_factory=list)


class BriefingLog:
    """Append-only SQLite-backed log of per-turn briefing events."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._ensure_source_agent_id_column()
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _ensure_source_agent_id_column(self) -> None:
        with self._lock:
            with contextlib.suppress(sqlite3.OperationalError):
                self._conn.execute("ALTER TABLE briefing_events ADD COLUMN source_agent_id TEXT")
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS briefing_events_source_agent "
                "ON briefing_events(agent_id, source_agent_id, ts)"
            )

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def record(self, event: BriefingEvent) -> None:
        ts_iso = event.ts.isoformat() if isinstance(event.ts, datetime) else str(event.ts)
        unit_ids = list(event.recalled_unit_ids or [])
        entity_ids = list(event.recalled_entity_ids or [])
        relation_ids = list(event.recalled_relation_ids or [])
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO briefing_events
                    (ts, request_id, agent_id, source_agent_id, query_text, topic_hint,
                     recalled_unit_ids, recalled_entity_ids, recalled_relation_ids,
                     count_units, count_entities, count_relations)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts_iso,
                    event.request_id,
                    event.agent_id,
                    event.source_agent_id,
                    event.query_text,
                    event.topic_hint,
                    json.dumps(unit_ids),
                    json.dumps(entity_ids),
                    json.dumps(relation_ids),
                    len(unit_ids),
                    len(entity_ids),
                    len(relation_ids),
                ),
            )
            self._conn.commit()

    def record_briefing(
        self,
        *,
        request_id: str,
        agent_id: str | None,
        query_text: str | None,
        topic_hint: str | None,
        recalled_unit_ids: Iterable[str],
        recalled_entity_ids: Iterable[str],
        recalled_relation_ids: Iterable[str],
        source_agent_id: str | None = None,
    ) -> None:
        self.record(BriefingEvent(
            ts=datetime.now(UTC),
            request_id=request_id,
            agent_id=agent_id,
            source_agent_id=source_agent_id,
            query_text=query_text,
            topic_hint=topic_hint,
            recalled_unit_ids=list(recalled_unit_ids),
            recalled_entity_ids=list(recalled_entity_ids),
            recalled_relation_ids=list(recalled_relation_ids),
        ))

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_by_request_id(self, request_id: str) -> dict | None:
        """Return the most recent briefing row for ``request_id`` (or None).

        ``request_id`` is the per-turn ``call_id``; under normal flow it's
        unique, but if a caller ever reused one we want the latest row,
        not a random one. ``ORDER BY id DESC`` is a stable, monotonic
        proxy for "latest" without relying on timestamp tie-breaks.
        """
        row = self._conn.execute(
            "SELECT * FROM briefing_events WHERE request_id = ? "
            "ORDER BY id DESC LIMIT 1",
            (request_id,),
        ).fetchone()
        return None if row is None else self._row_to_dict(row)

    def list_by_agent(
        self,
        *,
        agent_id: str | None = None,
        source_agent_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        """List briefing rows, optionally filtered by ``agent_id``.

        Returned newest-first. ``agent_id=None`` returns rows across all
        agents — useful for an admin dashboard, less useful for a per-user
        view; callers should pass ``agent_id`` when they have one.
        """
        params: list[Any] = []
        sql = "SELECT * FROM briefing_events"
        clauses: list[str] = []
        if agent_id is not None:
            clauses.append("agent_id = ?")
            params.append(agent_id)
        if source_agent_id is not None:
            clauses.append("source_agent_id = ?")
            params.append(source_agent_id)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([int(limit), int(offset)])
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        return {
            "id": int(row["id"]),
            "ts": row["ts"],
            "request_id": row["request_id"],
            "agent_id": row["agent_id"],
            "source_agent_id": row["source_agent_id"],
            "query_text": row["query_text"],
            "topic_hint": row["topic_hint"],
            "recalled_unit_ids": json.loads(row["recalled_unit_ids"] or "[]"),
            "recalled_entity_ids": json.loads(row["recalled_entity_ids"] or "[]"),
            "recalled_relation_ids": json.loads(row["recalled_relation_ids"] or "[]"),
            "count_units": int(row["count_units"] or 0),
            "count_entities": int(row["count_entities"] or 0),
            "count_relations": int(row["count_relations"] or 0),
        }
