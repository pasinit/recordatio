"""SQLite-backed AgentConfig CRUD with monotonic versioning.

One row per (user_id, slug); writes carry an expected ``version`` and are
rejected with ``AgentConfigVersionMismatch`` if the stored version has moved
on. This is the optimistic-concurrency contract the improver relies on
(``PUT`` returns 409 on version mismatch, see Phase 1 acceptance criteria
and MIGRATION_PLAN.md §"Phase 5 — Improver migration").

Threading model matches the rest of Recordatio's SQLite stores (briefing_log,
cost_log): a single connection opened with ``check_same_thread=False`` plus a
``threading.Lock`` around writes, WAL journaling so reads can proceed without
contending on the write lock.
"""

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_configs (
    user_id        TEXT    NOT NULL,
    slug           TEXT    NOT NULL,
    system_prompt  TEXT    NOT NULL DEFAULT '',
    model          TEXT    NOT NULL DEFAULT '',
    mcp_servers    TEXT    NOT NULL DEFAULT '[]',
    subagents      TEXT    NOT NULL DEFAULT '[]',
    allowed_tools  TEXT,
    version        INTEGER NOT NULL DEFAULT 1,
    updated_at     TEXT    NOT NULL,
    PRIMARY KEY (user_id, slug)
);
"""


class AgentConfigVersionMismatch(Exception):
    """Raised when a PUT carries a ``version`` that does not match the row.

    Carries ``expected`` (the caller's version) and ``actual`` (what the
    store currently has) so the caller — typically the agent endpoints
    router — can render a useful 409 body.
    """

    def __init__(self, expected: int, actual: int) -> None:
        super().__init__(f"agent config version mismatch: expected {expected}, found {actual}")
        self.expected = expected
        self.actual = actual


@dataclass
class AgentConfig:
    """The schema from MIGRATION_PLAN.md §"AgentConfig schema".

    Mutable so the runner can `replace`-like patch fields when consuming it.
    Lists are not shared across instances by default.
    """

    system_prompt: str = ""
    model: str = ""
    mcp_servers: list[dict[str, Any]] = field(default_factory=list)
    subagents: list[dict[str, Any]] = field(default_factory=list)
    allowed_tools: list[str] | None = None
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_prompt": self.system_prompt,
            "model": self.model,
            "mcp_servers": list(self.mcp_servers),
            "subagents": list(self.subagents),
            "allowed_tools": list(self.allowed_tools) if self.allowed_tools is not None else None,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentConfig":
        # Defensive: some fields are optional in the wire schema. We default
        # to empty rather than raise so a partial PUT from a newer client
        # still round-trips.
        return cls(
            system_prompt=str(data.get("system_prompt", "")),
            model=str(data.get("model", "")),
            mcp_servers=list(data.get("mcp_servers") or []),
            subagents=list(data.get("subagents") or []),
            allowed_tools=list(data["allowed_tools"]) if data.get("allowed_tools") is not None else None,
            version=int(data.get("version", 1)),
        )


class AgentConfigStore:
    """Per-(user_id, slug) AgentConfig CRUD with monotonic versioning."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get(self, user_id: str, slug: str) -> AgentConfig | None:
        row = self._conn.execute(
            "SELECT * FROM agent_configs WHERE user_id = ? AND slug = ?",
            (user_id, slug),
        ).fetchone()
        return None if row is None else self._row_to_config(row)

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def put(self, user_id: str, slug: str, config: AgentConfig) -> AgentConfig:
        """Insert or update with optimistic concurrency control.

        Semantics:
        - If no row exists, insert with ``version=1`` regardless of what
          the caller passed (the caller has nothing to compare against).
        - If a row exists, ``config.version`` must equal the stored
          version; otherwise raise ``AgentConfigVersionMismatch``. On
          success the stored version is incremented by 1 and returned.
        """
        from datetime import UTC, datetime

        now = datetime.now(UTC).isoformat()
        with self._lock:
            row = self._conn.execute(
                "SELECT version FROM agent_configs WHERE user_id = ? AND slug = ?",
                (user_id, slug),
            ).fetchone()
            if row is None:
                new_version = 1
                self._conn.execute(
                    """
                    INSERT INTO agent_configs
                        (user_id, slug, system_prompt, model, mcp_servers, subagents,
                         allowed_tools, version, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        slug,
                        config.system_prompt,
                        config.model,
                        json.dumps(config.mcp_servers),
                        json.dumps(config.subagents),
                        json.dumps(config.allowed_tools) if config.allowed_tools is not None else None,
                        new_version,
                        now,
                    ),
                )
            else:
                actual = int(row["version"])
                if config.version != actual:
                    raise AgentConfigVersionMismatch(expected=config.version, actual=actual)
                new_version = actual + 1
                self._conn.execute(
                    """
                    UPDATE agent_configs
                       SET system_prompt = ?, model = ?, mcp_servers = ?, subagents = ?,
                           allowed_tools = ?, version = ?, updated_at = ?
                     WHERE user_id = ? AND slug = ?
                    """,
                    (
                        config.system_prompt,
                        config.model,
                        json.dumps(config.mcp_servers),
                        json.dumps(config.subagents),
                        json.dumps(config.allowed_tools) if config.allowed_tools is not None else None,
                        new_version,
                        now,
                        user_id,
                        slug,
                    ),
                )
            self._conn.commit()

        # Re-read so the returned object reflects exactly what's on disk.
        stored = self.get(user_id, slug)
        # Stored cannot be None — we just wrote it under the same lock.
        assert stored is not None
        return stored

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_config(row: sqlite3.Row) -> AgentConfig:
        return AgentConfig(
            system_prompt=row["system_prompt"] or "",
            model=row["model"] or "",
            mcp_servers=json.loads(row["mcp_servers"] or "[]"),
            subagents=json.loads(row["subagents"] or "[]"),
            allowed_tools=json.loads(row["allowed_tools"]) if row["allowed_tools"] is not None else None,
            version=int(row["version"]),
        )
