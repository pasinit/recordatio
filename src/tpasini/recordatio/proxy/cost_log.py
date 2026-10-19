"""Sidecar SQLite store for cost + cache events.

Append-only log of every chat turn and every extraction call. Lives in a
separate database file from the knowledge store (``cost_events.db`` in the
same directory as ``AGENTMEM_DB``) so historical cost queries don't lock
the memory store.

Single connection with ``check_same_thread=False`` + a ``threading.Lock``
around writes. WAL mode lets concurrent readers proceed lockless.
"""

import sqlite3
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tpasini.recordatio.extraction import ExtractionSide

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cost_events (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                       TEXT    NOT NULL,
    agent_id                 TEXT,
    source_agent_id          TEXT,
    user_id                  TEXT,
    kind                     TEXT    NOT NULL CHECK (kind IN ('chat','extraction')),
    model                    TEXT    NOT NULL,
    provider                 TEXT,
    input_tokens             INTEGER NOT NULL DEFAULT 0,
    output_tokens            INTEGER NOT NULL DEFAULT 0,
    cache_creation_tokens    INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens        INTEGER NOT NULL DEFAULT 0,
    cost_usd                 REAL,
    request_id               TEXT,
    ok                       INTEGER NOT NULL DEFAULT 1,
    extraction_side          TEXT             CHECK (extraction_side IN ('input','output') OR extraction_side IS NULL),
    skipped                  INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS cost_events_ts        ON cost_events(ts);
CREATE INDEX IF NOT EXISTS cost_events_agent_ts  ON cost_events(agent_id, ts);
CREATE INDEX IF NOT EXISTS cost_events_user_ts   ON cost_events(user_id, ts);
CREATE INDEX IF NOT EXISTS cost_events_kind_ts   ON cost_events(kind, ts);
CREATE INDEX IF NOT EXISTS cost_events_model_ts  ON cost_events(model, ts);
CREATE INDEX IF NOT EXISTS cost_events_request   ON cost_events(request_id);
"""

# Columns added after the original schema shipped. Applied via ALTER TABLE on
# startup so existing databases pick them up without manual migration. SQLite
# doesn't allow ADD COLUMN to introduce a CHECK constraint, so historical
# databases enforce ``extraction_side`` validity via the Python ``ExtractionSide``
# enum at write time instead.
_ADDITIVE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("source_agent_id", "TEXT"),
    ("extraction_side", "TEXT"),
    ("skipped", "INTEGER NOT NULL DEFAULT 0"),
)


_VALID_KINDS = ("chat", "extraction")
_GROUP_BY_COLUMNS = {
    "kind": "kind",
    "model": "model",
    "agent": "agent_id",
    "source_agent": "source_agent_id",
    "user": "user_id",
    "provider": "provider",
}
_DISTINCT_COLUMNS = {"agent_id", "source_agent_id", "user_id", "model", "provider"}


@dataclass(frozen=True)
class CostEvent:
    """One row in the cost_events table."""

    ts: datetime
    kind: str
    model: str
    provider: str | None
    input_tokens: int
    output_tokens: int
    cache_creation_tokens: int
    cache_read_tokens: int
    cost_usd: float | None
    agent_id: str | None = None
    source_agent_id: str | None = None
    user_id: str | None = None
    request_id: str | None = None
    ok: bool = True
    # Populated only for ``kind='extraction'`` rows.
    extraction_side: ExtractionSide | None = None
    skipped: bool = False


class CostLog:
    """Append-only SQLite-backed cost & cache event log."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # WAL + NORMAL gives concurrent reads while writes are serialised under
        # the in-process lock; survives `--workers > 1` without corruption.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._apply_additive_migrations()
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS cost_events_source_agent_ts "
            "ON cost_events(agent_id, source_agent_id, ts)"
        )
        self._conn.commit()

    def _apply_additive_migrations(self) -> None:
        existing = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(cost_events)").fetchall()
        }
        for name, decl in _ADDITIVE_COLUMNS:
            if name not in existing:
                self._conn.execute(f"ALTER TABLE cost_events ADD COLUMN {name} {decl}")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def record(self, event: CostEvent) -> None:
        if event.kind not in _VALID_KINDS:
            raise ValueError(f"invalid kind: {event.kind!r}")
        ts_iso = event.ts.isoformat() if isinstance(event.ts, datetime) else str(event.ts)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO cost_events
                    (ts, agent_id, source_agent_id, user_id, kind, model, provider,
                     input_tokens, output_tokens,
                     cache_creation_tokens, cache_read_tokens,
                     cost_usd, request_id, ok,
                     extraction_side, skipped)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts_iso,
                    event.agent_id,
                    event.source_agent_id,
                    event.user_id,
                    event.kind,
                    event.model,
                    event.provider,
                    int(event.input_tokens or 0),
                    int(event.output_tokens or 0),
                    int(event.cache_creation_tokens or 0),
                    int(event.cache_read_tokens or 0),
                    None if event.cost_usd is None else float(event.cost_usd),
                    event.request_id,
                    1 if event.ok else 0,
                    event.extraction_side.value if event.extraction_side else None,
                    1 if event.skipped else 0,
                ),
            )
            self._conn.commit()

    def record_chat(
        self,
        *,
        model: str,
        provider: str | None,
        input_tokens: int,
        output_tokens: int,
        cache_creation_tokens: int,
        cache_read_tokens: int,
        cost_usd: float | None,
        agent_id: str | None,
        user_id: str | None,
        request_id: str | None,
        source_agent_id: str | None = None,
    ) -> None:
        self.record(CostEvent(
            ts=datetime.now(UTC),
            kind="chat",
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_tokens=cache_creation_tokens,
            cache_read_tokens=cache_read_tokens,
            cost_usd=cost_usd,
            agent_id=agent_id,
            source_agent_id=source_agent_id,
            user_id=user_id,
            request_id=request_id,
            ok=True,
        ))

    def record_extraction(
        self,
        *,
        model: str,
        provider: str | None,
        input_tokens: int,
        output_tokens: int,
        cache_creation_tokens: int,
        cache_read_tokens: int,
        cost_usd: float | None,
        agent_id: str | None,
        user_id: str | None,
        request_id: str | None,
        ok: bool,
        extraction_side: ExtractionSide,
        skipped: bool = False,
        source_agent_id: str | None = None,
    ) -> None:
        self.record(CostEvent(
            ts=datetime.now(UTC),
            kind="extraction",
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_tokens=cache_creation_tokens,
            cache_read_tokens=cache_read_tokens,
            cost_usd=cost_usd,
            agent_id=agent_id,
            source_agent_id=source_agent_id,
            user_id=user_id,
            request_id=request_id,
            ok=ok,
            extraction_side=extraction_side,
            skipped=skipped,
        ))

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def query_events(
        self,
        *,
        since: datetime | str | None = None,
        until: datetime | str | None = None,
        agent_id: str | None = None,
        source_agent_id: str | None = None,
        user_id: str | None = None,
        models: Iterable[str] | None = None,
        kinds: Iterable[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        where, params = self._build_filters(
            since=since, until=until, agent_id=agent_id, source_agent_id=source_agent_id, user_id=user_id,
            models=models, kinds=kinds,
        )
        sql = (
            "SELECT id, ts, agent_id, source_agent_id, user_id, kind, model, provider, "
            "       input_tokens, output_tokens, "
            "       cache_creation_tokens, cache_read_tokens, "
            "       cost_usd, request_id, ok, "
            "       extraction_side, skipped "
            "FROM cost_events"
        )
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY ts DESC, id DESC LIMIT ? OFFSET ?"
        params.extend([int(limit), int(offset)])
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_event_dict(r) for r in rows]

    def query_totals(
        self,
        *,
        since: datetime | str | None = None,
        until: datetime | str | None = None,
        agent_id: str | None = None,
        source_agent_id: str | None = None,
        user_id: str | None = None,
        models: Iterable[str] | None = None,
        kinds: Iterable[str] | None = None,
    ) -> dict:
        where, params = self._build_filters(
            since=since, until=until, agent_id=agent_id, source_agent_id=source_agent_id, user_id=user_id,
            models=models, kinds=kinds,
        )
        sql = (
            "SELECT "
            "  COALESCE(SUM(cost_usd), 0)              AS cost_usd, "
            "  COALESCE(SUM(input_tokens), 0)          AS input_tokens, "
            "  COALESCE(SUM(output_tokens), 0)         AS output_tokens, "
            "  COALESCE(SUM(cache_creation_tokens), 0) AS cache_creation_tokens, "
            "  COALESCE(SUM(cache_read_tokens), 0)     AS cache_read_tokens, "
            "  COUNT(*) AS event_count "
            "FROM cost_events"
        )
        if where:
            sql += " WHERE " + " AND ".join(where)
        row = self._conn.execute(sql, params).fetchone()
        return {
            "cost_usd": float(row["cost_usd"] or 0.0),
            "input_tokens": int(row["input_tokens"] or 0),
            "output_tokens": int(row["output_tokens"] or 0),
            "cache_creation_tokens": int(row["cache_creation_tokens"] or 0),
            "cache_read_tokens": int(row["cache_read_tokens"] or 0),
            "event_count": int(row["event_count"] or 0),
        }

    def query_series(
        self,
        *,
        granularity: str,
        since: datetime | str | None = None,
        until: datetime | str | None = None,
        agent_id: str | None = None,
        source_agent_id: str | None = None,
        user_id: str | None = None,
        models: Iterable[str] | None = None,
        kinds: Iterable[str] | None = None,
    ) -> list[dict]:
        if granularity == "hour":
            bucket_expr = "substr(ts, 1, 13) || ':00:00Z'"
        elif granularity == "day":
            bucket_expr = "substr(ts, 1, 10) || 'T00:00:00Z'"
        else:
            raise ValueError(f"granularity must be 'hour' or 'day', got {granularity!r}")

        where, params = self._build_filters(
            since=since, until=until, agent_id=agent_id, source_agent_id=source_agent_id, user_id=user_id,
            models=models, kinds=kinds,
        )
        sql = (
            f"SELECT {bucket_expr} AS bucket, "
            "  COALESCE(SUM(cost_usd), 0) AS cost_usd, "
            "  COALESCE(SUM(CASE WHEN kind='chat'       THEN cost_usd ELSE 0 END), 0) AS chat_cost_usd, "
            "  COALESCE(SUM(CASE WHEN kind='extraction' THEN cost_usd ELSE 0 END), 0) AS extraction_cost_usd, "
            "  COALESCE(SUM(input_tokens), 0)          AS input_tokens, "
            "  COALESCE(SUM(output_tokens), 0)         AS output_tokens, "
            "  COALESCE(SUM(cache_creation_tokens), 0) AS cache_creation_tokens, "
            "  COALESCE(SUM(cache_read_tokens), 0)     AS cache_read_tokens, "
            "  COUNT(*) AS event_count "
            "FROM cost_events"
        )
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " GROUP BY bucket ORDER BY bucket"
        rows = self._conn.execute(sql, params).fetchall()
        return [
            {
                "bucket": r["bucket"],
                "cost_usd": float(r["cost_usd"] or 0.0),
                "chat_cost_usd": float(r["chat_cost_usd"] or 0.0),
                "extraction_cost_usd": float(r["extraction_cost_usd"] or 0.0),
                "input_tokens": int(r["input_tokens"] or 0),
                "output_tokens": int(r["output_tokens"] or 0),
                "cache_creation_tokens": int(r["cache_creation_tokens"] or 0),
                "cache_read_tokens": int(r["cache_read_tokens"] or 0),
                "event_count": int(r["event_count"] or 0),
            }
            for r in rows
        ]

    def query_aggregate(
        self,
        *,
        group_by: str,
        limit: int = 20,
        since: datetime | str | None = None,
        until: datetime | str | None = None,
        agent_id: str | None = None,
        source_agent_id: str | None = None,
        user_id: str | None = None,
        models: Iterable[str] | None = None,
        kinds: Iterable[str] | None = None,
    ) -> list[dict]:
        column = _GROUP_BY_COLUMNS.get(group_by)
        if column is None:
            raise ValueError(
                f"group_by must be one of {sorted(_GROUP_BY_COLUMNS)}, got {group_by!r}"
            )
        where, params = self._build_filters(
            since=since, until=until, agent_id=agent_id, source_agent_id=source_agent_id, user_id=user_id,
            models=models, kinds=kinds,
        )
        sql = (
            f"SELECT {column} AS key, "
            "  COALESCE(SUM(cost_usd), 0)              AS cost_usd, "
            "  COALESCE(SUM(input_tokens), 0)          AS input_tokens, "
            "  COALESCE(SUM(output_tokens), 0)         AS output_tokens, "
            "  COALESCE(SUM(cache_creation_tokens), 0) AS cache_creation_tokens, "
            "  COALESCE(SUM(cache_read_tokens), 0)     AS cache_read_tokens, "
            "  COUNT(*) AS event_count "
            "FROM cost_events"
        )
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += f" GROUP BY {column} ORDER BY cost_usd DESC LIMIT ?"
        rows = self._conn.execute(sql, [*params, int(limit)]).fetchall()

        results: list[dict] = []
        for r in rows:
            entry = {
                "key": r["key"],
                "cost_usd": float(r["cost_usd"] or 0.0),
                "input_tokens": int(r["input_tokens"] or 0),
                "output_tokens": int(r["output_tokens"] or 0),
                "cache_creation_tokens": int(r["cache_creation_tokens"] or 0),
                "cache_read_tokens": int(r["cache_read_tokens"] or 0),
                "event_count": int(r["event_count"] or 0),
                "top_child": None,
            }
            if group_by == "agent" and r["key"] is not None:
                entry["top_child"] = self._top_model_for_agent(
                    r["key"],
                    since=since, until=until, user_id=user_id,
                    models=models, kinds=kinds,
                )
            results.append(entry)
        return results

    def list_distinct(self, column: str) -> list[str]:
        if column not in _DISTINCT_COLUMNS:
            raise ValueError(
                f"column must be one of {sorted(_DISTINCT_COLUMNS)}, got {column!r}"
            )
        rows = self._conn.execute(
            f"SELECT DISTINCT {column} FROM cost_events "
            f"WHERE {column} IS NOT NULL AND {column} != '' "
            f"ORDER BY {column}"
        ).fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _top_model_for_agent(
        self,
        agent_id: str,
        *,
        since: datetime | str | None = None,
        until: datetime | str | None = None,
        user_id: str | None = None,
        source_agent_id: str | None = None,
        models: Iterable[str] | None = None,
        kinds: Iterable[str] | None = None,
    ) -> str | None:
        where, params = self._build_filters(
            since=since, until=until, agent_id=agent_id, source_agent_id=source_agent_id, user_id=user_id,
            models=models, kinds=kinds,
        )
        sql = "SELECT model, COALESCE(SUM(cost_usd), 0) AS c FROM cost_events"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " GROUP BY model ORDER BY c DESC LIMIT 1"
        row = self._conn.execute(sql, params).fetchone()
        return row["model"] if row else None

    @staticmethod
    def _build_filters(
        *,
        since: datetime | str | None = None,
        until: datetime | str | None = None,
        agent_id: str | None = None,
        source_agent_id: str | None = None,
        user_id: str | None = None,
        models: Iterable[str] | None = None,
        kinds: Iterable[str] | None = None,
    ) -> tuple[list[str], list[Any]]:
        where: list[str] = []
        params: list[Any] = []
        if since is not None:
            where.append("ts >= ?")
            params.append(since.isoformat() if isinstance(since, datetime) else str(since))
        if until is not None:
            where.append("ts < ?")
            params.append(until.isoformat() if isinstance(until, datetime) else str(until))
        if agent_id is not None:
            where.append("agent_id = ?")
            params.append(agent_id)
        if source_agent_id is not None:
            where.append("source_agent_id = ?")
            params.append(source_agent_id)
        if user_id is not None:
            where.append("user_id = ?")
            params.append(user_id)
        models_list = [m for m in (models or []) if m]
        if models_list:
            placeholders = ",".join("?" * len(models_list))
            where.append(f"model IN ({placeholders})")
            params.extend(models_list)
        kinds_list = [k for k in (kinds or []) if k]
        if kinds_list:
            for k in kinds_list:
                if k not in _VALID_KINDS:
                    raise ValueError(f"invalid kind: {k!r}")
            placeholders = ",".join("?" * len(kinds_list))
            where.append(f"kind IN ({placeholders})")
            params.extend(kinds_list)
        return where, params

    @staticmethod
    def _row_to_event_dict(row: sqlite3.Row) -> dict:
        side_raw = row["extraction_side"]
        return {
            "id": int(row["id"]),
            "ts": row["ts"],
            "agent_id": row["agent_id"],
            "source_agent_id": row["source_agent_id"],
            "user_id": row["user_id"],
            "kind": row["kind"],
            "model": row["model"],
            "provider": row["provider"],
            "input_tokens": int(row["input_tokens"] or 0),
            "output_tokens": int(row["output_tokens"] or 0),
            "cache_creation_tokens": int(row["cache_creation_tokens"] or 0),
            "cache_read_tokens": int(row["cache_read_tokens"] or 0),
            "cost_usd": None if row["cost_usd"] is None else float(row["cost_usd"]),
            "request_id": row["request_id"],
            "ok": bool(row["ok"]),
            "extraction_side": ExtractionSide(side_raw) if side_raw else None,
            "skipped": bool(row["skipped"] or 0),
        }
