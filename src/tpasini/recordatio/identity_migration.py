"""Two-stage identity migration (IDENTITY_DESIGN.md rev. 2 §6).

One library shared verbatim by the prod CLI (``scripts/migrate_identity.py``)
and the eval-v2 builder (``scripts/build_eval_db_v2.py``), parameterized by
(connection, user filter, timestamp policy).

Stage 1 — per-scope unification (rev. 1 algorithm, carried intact): merge a
scope's user-referent twins ("User" + name-labelled duplicates of the same
person) into the ``User``-labelled survivor, in place, fully audited.

Stage 2 — cross-scope lift: per user (the segment before ``/``), re-home the
principal into the reserved scope ``{user_id}/_user``, lift principal-touching
relations and principal-linked active semantic units (R-RULE applied to
existing rows), mirror non-principal endpoints (uuid5), fold cross-agent
duplicates, and emit a conflict sheet — NO functional supersession is
synthesized at migration time (§3.3): conflicting active rows coexist, each
with provenance.

Determinism: no RNG; iteration ordered by ``(agent_id, id)``; uuid5 mirrors;
eval mode (``now=None``) writes no wall-clock value (every timestamp is a
function of existing rows); prod mode permits wall clock for alias
``created_at`` only. Idempotence: a second run of either stage is a no-op
(zero mutations). Conservation equations are asserted in code
(``rows_before + inserted - deleted == rows_after``, per table, per stage).
The library never commits — transaction policy belongs to the caller.
"""

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from tpasini.recordatio.entity_resolver import normalize_label
from tpasini.recordatio.identity import (
    RESERVED_USER_SLUG,
    mirror_entity_id,
    principal_canonical_key,
)

USER_NORM = "user"

# Unit-subject link predicates that follow their unit's home (§2.2).
UNIT_LINK_PREDICATES = ("mentions", "about")

_WIKIDATA_NAMESPACES = ("wikidata", "wikidata_id")

# Per-entity rows that move with the entity on merge/re-home, with their
# uniqueness key columns (used for collision-aware moves).
_ATTENDANT_UNIQUE: dict[str, tuple[str, ...]] = {
    "entity_aliases": ("agent_id", "alias_key"),
    "entity_identifiers": ("agent_id", "namespace", "value"),
    "entity_attributes": ("agent_id", "entity_id", "attribute_type", "value"),  # active rows only
    "entity_mentions": ("agent_id", "entity_id", "source_kind", "source_id"),
    "entity_topics": ("agent_id", "entity_id", "topic_id"),
}

_CONSERVED_TABLES = (
    "knowledge_units",
    "entities",
    "relations",
    "entity_aliases",
    "entity_identifiers",
    "entity_attributes",
    "entity_mentions",
    "entity_topics",
    "unit_topics",
)


class MigrationError(RuntimeError):
    """A migration invariant failed; the surrounding transaction must roll back."""


@dataclass
class MigrationReport:
    """One JSON report per run (sorted keys when serialized)."""

    stage1: dict[str, dict] = field(default_factory=dict)  # scope -> audit section
    stage2: dict[str, dict] = field(default_factory=dict)  # user -> audit section
    skipped_bare_scopes: list[str] = field(default_factory=list)
    aborted_users: dict[str, str] = field(default_factory=dict)  # user -> reason
    id_map: dict[str, str] = field(default_factory=dict)  # removed/merged id -> surviving id
    mutations: int = 0

    def conflict_sheet(self) -> dict[str, list[dict]]:
        return {user: section.get("conflict_sheet", []) for user, section in self.stage2.items()}

    def to_payload(self) -> dict:
        return {
            "stage1": self.stage1,
            "stage2": self.stage2,
            "skipped_bare_scopes": sorted(self.skipped_bare_scopes),
            "aborted_users": self.aborted_users,
            "id_map": self.id_map,
            "mutations": self.mutations,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_payload(), indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Row + mutation helpers
# ---------------------------------------------------------------------------


def _rows(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[dict]:
    cur = conn.execute(sql, params)
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]


def all_scopes(conn: sqlite3.Connection) -> list[str]:
    return [
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT agent_id FROM entities UNION SELECT DISTINCT agent_id FROM knowledge_units "
            "UNION SELECT DISTINCT agent_id FROM relations ORDER BY 1"
        )
    ]


def _scope_counts(conn: sqlite3.Connection, scopes: list[str]) -> dict[str, int]:
    marks = ",".join("?" for _ in scopes)
    return {
        table: conn.execute(
            f'SELECT COUNT(*) FROM "{table}" WHERE agent_id IN ({marks})',  # noqa: S608
            tuple(scopes),
        ).fetchone()[0]
        for table in _CONSERVED_TABLES
    }


def _min_key(row: dict) -> tuple[str, str]:
    return (row["asserted_at"], row["id"])


def _norm_content(text: str) -> str:
    return " ".join(text.lower().split())


def _alias_key(value: str) -> str:
    return " ".join(value.casefold().split())


@dataclass
class _Mutator:
    """Executes mutations and tallies inserts/deletes per table so
    conservation can be asserted in code."""

    conn: sqlite3.Connection
    mutations: int = 0
    inserted: dict[str, int] = field(default_factory=dict)
    deleted: dict[str, int] = field(default_factory=dict)

    def update(self, table: str, sql: str, params: tuple) -> int:
        n = max(self.conn.execute(sql, params).rowcount, 0)
        self.mutations += n
        return n

    def insert(self, table: str, sql: str, params: tuple) -> None:
        self.conn.execute(sql, params)
        self.mutations += 1
        self.inserted[table] = self.inserted.get(table, 0) + 1

    def delete(self, table: str, sql: str, params: tuple) -> int:
        n = max(self.conn.execute(sql, params).rowcount, 0)
        self.mutations += n
        self.deleted[table] = self.deleted.get(table, 0) + n
        return n

    def assert_conservation(self, before: dict[str, int], after: dict[str, int], where: str) -> dict:
        for table in _CONSERVED_TABLES:
            expected = before[table] + self.inserted.get(table, 0) - self.deleted.get(table, 0)
            if after[table] != expected:
                raise MigrationError(
                    f"{where}: conservation failed for {table}: "
                    f"{before[table]} + {self.inserted.get(table, 0)} - {self.deleted.get(table, 0)} "
                    f"!= {after[table]}"
                )
        return {
            "before": before,
            "after": after,
            "inserted": dict(sorted(self.inserted.items())),
            "deleted": dict(sorted(self.deleted.items())),
        }


def _fold_active_groups(
    rows: list[dict], key_of: Callable[[dict], tuple]
) -> tuple[list[dict], dict[str, dict], list[dict]]:
    """Group rows by ``key_of``; per multi-row group pick the survivor
    (min ``(asserted_at, id)``) and merge timestamps/counters into it.

    Returns ``(deletes, updates_by_id, fold_groups)``.
    """
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(key_of(row), []).append(row)
    deletes: list[dict] = []
    updates: dict[str, dict] = {}
    fold_groups: list[dict] = []
    for key in sorted(groups, key=repr):
        group = sorted(groups[key], key=_min_key)
        if len(group) == 1:
            continue
        survivor, rest = group[0], group[1:]
        merged: dict = {"asserted_at": min(r["asserted_at"] for r in group)}
        if "valid_from" in survivor:
            merged["valid_from"] = min(r["valid_from"] for r in group)
        if "extraction_count" in survivor:
            merged["extraction_count"] = sum(r["extraction_count"] or 0 for r in group)
        if "retrieval_count" in survivor:
            merged["retrieval_count"] = sum(r["retrieval_count"] or 0 for r in group)
        updates[survivor["id"]] = merged
        deletes.extend(rest)
        fold_groups.append({"survivor": survivor["id"], "folded": [r["id"] for r in rest], "rows": group})
    return deletes, updates, fold_groups


def _move_attendant_rows(
    mut: _Mutator,
    table: str,
    *,
    where: str,
    params: tuple,
    set_entity_id: str | None = None,
    set_agent_id: str | None = None,
    scope_guard: list[str] | None = None,
) -> None:
    """Re-point/re-home rows of an attendant table, collision-aware: a row
    whose new key already exists is an exact functional duplicate of what the
    target already carries — it is deleted (a fold), never duplicated.

    ``scope_guard``: when given, only rows whose current ``agent_id`` is in
    this list are processed. Required for cross-scope entity-id merges
    (``set_entity_id`` only) to prevent picking up rows from OTHER users'
    scopes that happen to reference the same entity id (conservation invariant:
    foreign-scope rows are NOT in the before-count for this user's migration).
    """
    keys = _ATTENDANT_UNIQUE[table]
    if scope_guard is not None:
        marks = ",".join("?" for _ in scope_guard)
        full_where = f"({where}) AND agent_id IN ({marks})"
        full_params = params + tuple(scope_guard)
    else:
        full_where = where
        full_params = params
    rows = _rows(mut.conn, f"SELECT rowid AS rid, * FROM {table} WHERE {full_where} ORDER BY rowid", full_params)  # noqa: S608
    for row in rows:
        new = dict(row)
        if set_entity_id is not None:
            new["entity_id"] = set_entity_id
        if set_agent_id is not None:
            new["agent_id"] = set_agent_id
        key_clause = " AND ".join(f"{c} = ?" for c in keys)
        # entity_attributes' unique index is partial (active rows only):
        # closed rows move freely and never collide.
        if table == "entity_attributes":
            if row["valid_until"] is not None:
                clash = None
            else:
                clash = mut.conn.execute(
                    f"SELECT rowid FROM {table} WHERE {key_clause} AND valid_until IS NULL AND rowid != ?",  # noqa: S608
                    tuple(new[c] for c in keys) + (row["rid"],),
                ).fetchone()
        else:
            clash = mut.conn.execute(
                f"SELECT rowid FROM {table} WHERE {key_clause} AND rowid != ?",  # noqa: S608
                tuple(new[c] for c in keys) + (row["rid"],),
            ).fetchone()
        if clash is not None:
            mut.delete(table, f"DELETE FROM {table} WHERE rowid = ?", (row["rid"],))  # noqa: S608
        else:
            sets, vals = [], []
            if set_entity_id is not None:
                sets.append("entity_id = ?")
                vals.append(set_entity_id)
            if set_agent_id is not None:
                sets.append("agent_id = ?")
                vals.append(set_agent_id)
            mut.update(table, f"UPDATE {table} SET {', '.join(sets)} WHERE rowid = ?", (*vals, row["rid"]))  # noqa: S608


# ---------------------------------------------------------------------------
# Stage 1 — per-scope unification
# ---------------------------------------------------------------------------


def _principal_token(scope: str) -> str:
    return normalize_label(scope.partition("/")[0])


def stage1_merge_set(conn: sqlite3.Connection, scope: str) -> tuple[dict | None, list[dict], list[dict]]:
    """(survivor, twins, adjudications) for one scope.

    Merge set M: ``norm(label) ∈ {"user", principal_token}`` or multi-token
    labels whose first token == principal_token. Third parties fail the token
    test. Namesake ambiguity bound: ≥2 distinct multi-token labels ⇒ all
    multi-token members dropped to adjudication. The survivor is the
    ``User``-labelled node (tie: min ``(asserted_at, id)``); without one the
    scope has no anchor and nothing merges.
    """
    token = _principal_token(scope)
    rows = _rows(conn, "SELECT * FROM entities WHERE agent_id = ? ORDER BY asserted_at, id", (scope,))
    user_nodes = [r for r in rows if normalize_label(r["label"]) == USER_NORM]
    if not user_nodes:
        return None, [], []
    survivor = sorted(user_nodes, key=_min_key)[0]
    single: list[dict] = []
    multi: list[dict] = []
    for row in rows:
        if row["id"] == survivor["id"]:
            continue
        norm = normalize_label(row["label"])
        if not norm:
            continue
        if norm in (USER_NORM, token):
            single.append(row)
            continue
        parts = norm.split(" ")
        if len(parts) > 1 and parts[0] == token:
            multi.append(row)
    adjudications: list[dict] = []
    if len({normalize_label(r["label"]) for r in multi}) >= 2:
        adjudications = [
            {"entity_id": r["id"], "label": r["label"], "reason": "namesake-ambiguity"} for r in multi
        ]
        multi = []
    return survivor, sorted(single + multi, key=_min_key), adjudications


def _alias_conflict(
    conn: sqlite3.Connection, scopes: list[str], alias_norm: str, allowed_ids: set[str]
) -> dict | None:
    """First entity/alias in ``scopes`` carrying ``alias_norm`` outside
    ``allowed_ids`` (the §2.1 collision guard), or ``None``."""
    marks = ",".join("?" for _ in scopes)
    for row in _rows(
        conn,
        f"SELECT id, label, agent_id FROM entities WHERE agent_id IN ({marks}) ORDER BY agent_id, id",  # noqa: S608
        tuple(scopes),
    ):
        if row["id"] not in allowed_ids and normalize_label(row["label"]) == alias_norm:
            return {"kind": "entity-label", "entity_id": row["id"], "label": row["label"], "scope": row["agent_id"]}
    for row in _rows(
        conn,
        f"SELECT entity_id, alias, agent_id FROM entity_aliases WHERE agent_id IN ({marks}) ORDER BY agent_id, id",  # noqa: S608
        tuple(scopes),
    ):
        if row["entity_id"] not in allowed_ids and normalize_label(row["alias"]) == alias_norm:
            return {"kind": "alias", "entity_id": row["entity_id"], "label": row["alias"], "scope": row["agent_id"]}
    return None


def _merge_entities_into(
    mut: _Mutator,
    *,
    target_id: str,
    merged: list[dict],
    guard_scopes: list[str],
    alias_target_scope: str,
    now: datetime | None,
    report: MigrationReport,
    section: dict,
    attendant_scope_guard: list[str] | None = None,
) -> None:
    """Re-point every attendant reference from each merged entity to
    ``target_id`` (relations are the caller's fold pass), add merged labels as
    guarded standing aliases, drop merged-in wikidata identifiers
    (misattributions of a private person), delete the merged entity rows.
    Shared by stage 1 (twins → survivor) and stage 2 (survivors → principal).

    ``attendant_scope_guard``: when given, only attendant rows whose
    ``agent_id`` is in this list are re-pointed. Required when the merged
    entities may have attendant rows in other users' scopes (cross-scope
    entity_topics references in the eval corpus) so foreign-scope rows are
    not included in this user's conservation equation.
    """
    conn = mut.conn
    merge_ids = {m["id"] for m in merged} | {target_id}
    for row in merged:
        dropped = _rows(
            conn,
            "SELECT * FROM entity_identifiers WHERE entity_id = ? AND namespace IN (?, ?)",
            (row["id"], *_WIKIDATA_NAMESPACES),
        )
        if dropped:
            section.setdefault("dropped_wikidata_identifiers", []).extend(dropped)
            mut.delete(
                "entity_identifiers",
                "DELETE FROM entity_identifiers WHERE entity_id = ? AND namespace IN (?, ?)",
                (row["id"], *_WIKIDATA_NAMESPACES),
            )
        if row["wikidata_id"]:
            section.setdefault("dropped_wikidata_ids", []).append(
                {"entity_id": row["id"], "wikidata_id": row["wikidata_id"]}
            )
        alias_norm = normalize_label(row["label"])
        if alias_norm and alias_norm != USER_NORM:
            conflict = _alias_conflict(conn, guard_scopes, alias_norm, merge_ids)
            if conflict is not None:
                section.setdefault("alias_conflicts", []).append({"alias": row["label"], "blocked_by": conflict})
            else:
                exists = conn.execute(
                    "SELECT 1 FROM entity_aliases WHERE agent_id = ? AND alias_key = ?",
                    (alias_target_scope, _alias_key(row["label"])),
                ).fetchone()
                if exists is None:
                    created_at = now.isoformat() if now is not None else row["asserted_at"]
                    mut.insert(
                        "entity_aliases",
                        "INSERT INTO entity_aliases (id, entity_id, alias, alias_key, source, created_at, agent_id) "
                        "VALUES (?, ?, ?, ?, 'identity-migration', ?, ?)",
                        (
                            f"alias-{target_id}-{alias_norm.replace(' ', '-')}",
                            target_id,
                            row["label"],
                            _alias_key(row["label"]),
                            created_at,
                            alias_target_scope,
                        ),
                    )
                    section.setdefault("alias_additions", []).append(row["label"])
        for table in _ATTENDANT_UNIQUE:
            _move_attendant_rows(
                mut, table, where="entity_id = ?", params=(row["id"],), set_entity_id=target_id,
                scope_guard=attendant_scope_guard,
            )
        mut.delete("entities", "DELETE FROM entities WHERE id = ? AND agent_id = ?", (row["id"], row["agent_id"]))
        section.setdefault("merged_entities", []).append(
            {k: row[k] for k in ("id", "label", "type", "agent_id", "asserted_at", "wikidata_id")}
        )
        report.id_map[row["id"]] = target_id


def _repoint_and_fold_relations(
    mut: _Mutator,
    *,
    scopes: list[str],
    final_of: Callable[[dict], tuple],
    section: dict,
    report: MigrationReport,
) -> None:
    """Row-wise relation rewrite + fold.

    ``final_of(row) -> (agent_id, subject_id, object_id, source_agent_id)``
    computes each row's final placement. Active duplicates on the unique
    active key fold (survivor min ``(asserted_at, id)``, min
    ``valid_from``/``asserted_at``, summed counters; folded rows DELETED — not
    closed, closing would assert a falsehood — and logged whole). Closed rows
    are kept except exact duplicates. Self-loops are kept and audited.
    Rewrites are applied delete-then-reinsert so no transient state can
    violate the unique active index.
    """
    conn = mut.conn
    marks = ",".join("?" for _ in scopes)
    rows = _rows(
        conn, f"SELECT * FROM relations WHERE agent_id IN ({marks}) ORDER BY agent_id, id", tuple(scopes)  # noqa: S608
    )
    finals = {row["id"]: final_of(row) for row in rows}

    active = [r for r in rows if r["valid_until"] is None]
    closed = [r for r in rows if r["valid_until"] is not None]

    def active_key(row: dict) -> tuple:
        agent, subj, obj, _src = finals[row["id"]]
        return (agent, subj, row["predicate"], obj)

    deletes, updates, fold_groups = _fold_active_groups(active, active_key)
    if fold_groups:
        section.setdefault("relation_folds", []).extend(fold_groups)
        for group in fold_groups:
            for folded_id in group["folded"]:
                report.id_map[folded_id] = group["survivor"]

    def closed_exact_key(row: dict) -> tuple:
        agent, subj, obj, _src = finals[row["id"]]
        return (agent, subj, row["predicate"], obj, row["asserted_at"], row["valid_from"],
                row["valid_until"], row["provenance"])

    closed_groups: dict[tuple, list[dict]] = {}
    for row in closed:
        closed_groups.setdefault(closed_exact_key(row), []).append(row)
    for ckey in sorted(closed_groups, key=repr):
        group = sorted(closed_groups[ckey], key=lambda r: r["id"])
        if len(group) > 1:
            section.setdefault("closed_exact_duplicates_deleted", []).extend(group[1:])
            for row in group[1:]:
                report.id_map[row["id"]] = group[0]["id"]
            deletes.extend(group[1:])

    delete_ids = {r["id"] for r in deletes}
    for row in deletes:
        mut.delete("relations", "DELETE FROM relations WHERE id = ?", (row["id"],))

    changed: list[tuple[dict, tuple, dict]] = []
    self_loops: list[str] = []
    for row in rows:
        if row["id"] in delete_ids:
            continue
        final = finals[row["id"]]
        merged = updates.get(row["id"], {})
        if final[1] == final[2]:
            self_loops.append(row["id"])
        if final != (row["agent_id"], row["subject_id"], row["object_id"], row["source_agent_id"]) or merged:
            changed.append((row, final, merged))
    if self_loops:
        section["self_loops"] = sorted(set(section.get("self_loops", [])) | set(self_loops))

    for row, _final, _merged in changed:
        mut.delete("relations", "DELETE FROM relations WHERE id = ?", (row["id"],))
    for row, final, merged in changed:
        agent, subj, obj, src = final
        mut.insert(
            "relations",
            "INSERT INTO relations (id, subject_id, predicate, object_id, asserted_at, valid_from, valid_until, "
            "provenance, agent_id, source_agent_id, extraction_count, retrieval_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row["id"], subj, row["predicate"], obj,
                merged.get("asserted_at", row["asserted_at"]),
                merged.get("valid_from", row["valid_from"]),
                row["valid_until"], row["provenance"], agent, src,
                merged.get("extraction_count", row["extraction_count"]),
                merged.get("retrieval_count", row["retrieval_count"]),
            ),
        )


def run_stage1_scope(
    conn: sqlite3.Connection, scope: str, report: MigrationReport, *, now: datetime | None = None
) -> dict | None:
    """Stage 1 on one scope; returns the audit section (``None`` when the
    scope has no user anchor)."""
    mut = _Mutator(conn)
    survivor, twins, adjudications = stage1_merge_set(conn, scope)
    if survivor is None:
        return None
    before = _scope_counts(conn, [scope])
    section: dict = {
        "survivor": survivor["id"],
        "adjudications": adjudications,
        "twins": [t["id"] for t in twins],
    }
    mapping = {t["id"]: survivor["id"] for t in twins}

    if twins:
        _merge_entities_into(
            mut,
            target_id=survivor["id"],
            merged=twins,
            guard_scopes=[scope],
            alias_target_scope=scope,
            now=now,
            report=report,
            section=section,
            attendant_scope_guard=[scope],
        )

        def final_of(row: dict) -> tuple:
            return (
                row["agent_id"],
                mapping.get(row["subject_id"], row["subject_id"]),
                mapping.get(row["object_id"], row["object_id"]),
                row["source_agent_id"],
            )

        _repoint_and_fold_relations(mut, scopes=[scope], final_of=final_of, section=section, report=report)

    # Transitional stage-1 key (re-keyed by stage 2 on lift).
    key = f"agent_user:{scope}"
    if survivor["canonical_key"] != key or survivor["type"] != "person":
        mut.update(
            "entities",
            "UPDATE entities SET canonical_key = ?, type = 'person' WHERE id = ? AND agent_id = ?",
            (key, survivor["id"], scope),
        )

    if mut.mutations == 0:
        return {"survivor": survivor["id"], "adjudications": adjudications, "twins": [], "mutations": 0} \
            if adjudications else None
    after = _scope_counts(conn, [scope])
    section["conservation"] = mut.assert_conservation(before, after, f"stage1 {scope}")
    section["mutations"] = mut.mutations
    report.mutations += mut.mutations
    return section


# ---------------------------------------------------------------------------
# Stage 2 — cross-scope lift
# ---------------------------------------------------------------------------


def _stage2_user(
    conn: sqlite3.Connection,
    user: str,
    agent_scopes: list[str],
    report: MigrationReport,
    *,
    now: datetime | None = None,
) -> dict | None:
    uscope = f"{user}/{RESERVED_USER_SLUG}"
    key = principal_canonical_key(uscope)
    mut = _Mutator(conn)

    # A pre-existing foreign ``*/_user`` scope aborts this user (I-13): the
    # reserved grammar found rows it did not write.
    existing_user_scope = _rows(conn, "SELECT * FROM entities WHERE agent_id = ? ORDER BY id", (uscope,))
    principal_rows = [r for r in existing_user_scope if r["canonical_key"] == key]
    if existing_user_scope and not principal_rows:
        report.aborted_users[user] = f"foreign scope {uscope} already exists without a principal row"
        return None

    survivors: list[dict] = []
    for scope in agent_scopes:
        rows = _rows(conn, "SELECT * FROM entities WHERE agent_id = ? ORDER BY asserted_at, id", (scope,))
        survivors.extend(r for r in rows if normalize_label(r["label"]) == USER_NORM)
    if not survivors and not principal_rows:
        return None

    scopes_of_user = agent_scopes + [uscope]
    before = _scope_counts(conn, scopes_of_user)
    section: dict = {"user_scope": uscope, "scopes": agent_scopes}

    # Principal placement: min (asserted_at, id) across the user's stage-1
    # survivors — unless a prior run already homed one (idempotent rerun).
    if principal_rows:
        principal_id = principal_rows[0]["id"]
        section["principal"] = {"id": principal_id, "origin_scope": uscope}
    else:
        principal = sorted(survivors, key=_min_key)[0]
        principal_id = principal["id"]
        # Re-home (id stable), re-key, strip misattributed wikidata ids.
        mut.update(
            "entities",
            "UPDATE entities SET agent_id = ?, canonical_key = ?, type = 'person', wikidata_id = NULL "
            "WHERE id = ? AND agent_id = ?",
            (uscope, key, principal_id, principal["agent_id"]),
        )
        dropped = _rows(
            conn,
            "SELECT * FROM entity_identifiers WHERE entity_id = ? AND namespace IN (?, ?)",
            (principal_id, *_WIKIDATA_NAMESPACES),
        )
        if dropped:
            section.setdefault("dropped_wikidata_identifiers", []).extend(dropped)
            mut.delete(
                "entity_identifiers",
                "DELETE FROM entity_identifiers WHERE entity_id = ? AND namespace IN (?, ?)",
                (principal_id, *_WIKIDATA_NAMESPACES),
            )
        if principal["wikidata_id"]:
            section.setdefault("dropped_wikidata_ids", []).append(
                {"entity_id": principal_id, "wikidata_id": principal["wikidata_id"]}
            )
        section["principal"] = {"id": principal_id, "origin_scope": principal["agent_id"]}

    merged_survivors = [s for s in survivors if s["id"] != principal_id]
    survivor_map = {s["id"]: principal_id for s in merged_survivors}

    # Merge the other scopes' survivors into the principal. Alias union is
    # twin *labels* only (stage-1 standing aliases ride along as attendant
    # rows), each addition guarded against every scope of this user.
    if merged_survivors:
        _merge_entities_into(
            mut,
            target_id=principal_id,
            merged=merged_survivors,
            guard_scopes=scopes_of_user,
            alias_target_scope=uscope,
            now=now,
            report=report,
            section=section,
            attendant_scope_guard=scopes_of_user,
        )

    # Re-home the principal's attendant rows into the user scope; alias rows
    # re-pass the collision guard against every scope of this user.
    alias_rows = _rows(
        conn,
        "SELECT rowid AS rid, * FROM entity_aliases WHERE entity_id = ? AND agent_id != ? ORDER BY rowid",
        (principal_id, uscope),
    )
    for row in alias_rows:
        conflict = _alias_conflict(conn, scopes_of_user, normalize_label(row["alias"]), {principal_id})
        if conflict is not None:
            section.setdefault("alias_conflicts", []).append({"alias": row["alias"], "blocked_by": conflict})
            mut.delete("entity_aliases", "DELETE FROM entity_aliases WHERE rowid = ?", (row["rid"],))
    for table in _ATTENDANT_UNIQUE:
        _move_attendant_rows(
            mut, table, where="entity_id = ? AND agent_id != ?", params=(principal_id, uscope),
            set_agent_id=uscope, scope_guard=scopes_of_user,
        )

    # ---- units: the R-RULE applied to existing rows ---------------------------
    survivor_ids = tuple(survivor_map) + (principal_id,)
    smarks = ",".join("?" for _ in survivor_ids)
    lift_units: list[dict] = []
    if agent_scopes:
        marks = ",".join("?" for _ in agent_scopes)
        lift_units = _rows(
            conn,
            f"""
            SELECT DISTINCT ku.* FROM knowledge_units ku
            JOIN relations r ON r.subject_id = ku.id AND r.valid_until IS NULL
                AND r.predicate IN ('mentions', 'about') AND r.object_id IN ({smarks})
            WHERE ku.agent_id IN ({marks}) AND ku.type = 'semantic' AND ku.valid_until IS NULL
            ORDER BY ku.agent_id, ku.id
            """,  # noqa: S608
            survivor_ids + tuple(agent_scopes),
        )
    lifted_unit_ids = {u["id"] for u in lift_units}
    section["lifted_units"] = len(lift_units)

    # Cross-agent exact-content fold among lifted + already-homed active units.
    homed_units = _rows(
        conn, "SELECT * FROM knowledge_units WHERE agent_id = ? AND valid_until IS NULL ORDER BY id", (uscope,)
    )
    unit_pool = lift_units + [u for u in homed_units if u["id"] not in lifted_unit_ids]
    unit_deletes, unit_updates, unit_fold_groups = _fold_active_groups(
        unit_pool, lambda u: (u["type"], _norm_content(u["content"]))
    )
    unit_fold_map: dict[str, str] = {}
    if unit_fold_groups:
        section["unit_folds"] = unit_fold_groups
        for group in unit_fold_groups:
            for folded_id in group["folded"]:
                unit_fold_map[folded_id] = group["survivor"]
                report.id_map[folded_id] = group["survivor"]
    folded_unit_ids = {u["id"] for u in unit_deletes}
    for row in unit_deletes:
        # Re-point the folded unit's topic tags (PK-collision-aware), drop the row.
        for tag in _rows(
            mut.conn, "SELECT rowid AS rid, * FROM unit_topics WHERE unit_id = ? ORDER BY rowid", (row["id"],)
        ):
            clash = mut.conn.execute(
                "SELECT 1 FROM unit_topics WHERE agent_id = ? AND unit_id = ? AND topic_id = ?",
                (uscope, unit_fold_map[row["id"]], tag["topic_id"]),
            ).fetchone()
            if clash is not None:
                mut.delete("unit_topics", "DELETE FROM unit_topics WHERE rowid = ?", (tag["rid"],))
            else:
                mut.update(
                    "unit_topics",
                    "UPDATE unit_topics SET unit_id = ?, agent_id = ? WHERE rowid = ?",
                    (unit_fold_map[row["id"]], uscope, tag["rid"]),
                )
        mut.delete("knowledge_units", "DELETE FROM knowledge_units WHERE id = ?", (row["id"],))
    user_unit_ids = {u["id"] for u in unit_pool} - folded_unit_ids
    for row in unit_pool:
        if row["id"] in folded_unit_ids:
            continue
        merged = unit_updates.get(row["id"], {})
        src = row["source_agent_id"] or (row["agent_id"] if row["agent_id"] != uscope else None)
        final = (
            uscope, src,
            merged.get("asserted_at", row["asserted_at"]),
            merged.get("valid_from", row["valid_from"]),
            merged.get("extraction_count", row["extraction_count"]),
            merged.get("retrieval_count", row["retrieval_count"]),
        )
        current = (
            row["agent_id"], row["source_agent_id"], row["asserted_at"], row["valid_from"],
            row["extraction_count"], row["retrieval_count"],
        )
        if final != current:
            mut.update(
                "knowledge_units",
                "UPDATE knowledge_units SET agent_id = ?, source_agent_id = ?, asserted_at = ?, valid_from = ?, "
                "extraction_count = ?, retrieval_count = ? WHERE id = ?",
                (*final, row["id"]),
            )
            if row["agent_id"] != uscope:
                for tag in _rows(
                    mut.conn,
                    "SELECT rowid AS rid, topic_id FROM unit_topics WHERE unit_id = ? AND agent_id = ? ORDER BY rowid",
                    (row["id"], row["agent_id"]),
                ):
                    clash = mut.conn.execute(
                        "SELECT 1 FROM unit_topics WHERE agent_id = ? AND unit_id = ? AND topic_id = ?",
                        (uscope, row["id"], tag["topic_id"]),
                    ).fetchone()
                    if clash is not None:
                        mut.delete("unit_topics", "DELETE FROM unit_topics WHERE rowid = ?", (tag["rid"],))
                    else:
                        mut.update(
                            "unit_topics", "UPDATE unit_topics SET agent_id = ? WHERE rowid = ?", (uscope, tag["rid"])
                        )

    # ---- relations: final placement + mirrors + cross-agent fold --------------
    entity_scope_by_id: dict[str, str] = {}
    entity_by_id: dict[str, dict] = {}
    for scope in scopes_of_user:
        for row in _rows(conn, "SELECT * FROM entities WHERE agent_id = ? ORDER BY id", (scope,)):
            entity_scope_by_id[row["id"]] = row["agent_id"]
            entity_by_id[row["id"]] = row

    def resolve_id(raw: str) -> str:
        return unit_fold_map.get(raw, survivor_map.get(raw, raw))

    def routes_user_level(row: dict, subj: str, obj: str) -> bool:
        if row["agent_id"] == uscope:
            return True
        if principal_id in (subj, obj) and subj in entity_scope_by_id and obj in entity_scope_by_id:
            return True
        return subj in user_unit_ids and row["predicate"] in UNIT_LINK_PREDICATES

    allmarks = ",".join("?" for _ in scopes_of_user)
    relation_rows = _rows(
        conn,
        f"SELECT * FROM relations WHERE agent_id IN ({allmarks}) ORDER BY agent_id, id",  # noqa: S608
        tuple(scopes_of_user),
    )

    # Pass 1: every agent-scope entity endpoint of a user-homed relation needs
    # a uuid5 mirror (§2.3) — cross-agent dedup by construction.
    pending_mirrors: dict[str, list[dict]] = {}
    for row in relation_rows:
        subj, obj = resolve_id(row["subject_id"]), resolve_id(row["object_id"])
        if not routes_user_level(row, subj, obj):
            continue
        for endpoint in (subj, obj):
            if endpoint == principal_id or endpoint not in entity_scope_by_id:
                continue
            if entity_scope_by_id[endpoint] == uscope:
                continue
            source = entity_by_id[endpoint]
            mid = mirror_entity_id(uscope, source["type"], source["label"])
            pending_mirrors.setdefault(mid, []).append(source)

    mirror_target: dict[str, str] = {}  # source entity id -> user-scope id to point at
    mirror_sources: dict[str, list[str]] = {}
    for mid in sorted(pending_mirrors):
        sources = sorted(pending_mirrors[mid], key=_min_key)
        exists = conn.execute("SELECT 1 FROM entities WHERE id = ? AND agent_id = ?", (mid, uscope)).fetchone()
        target = mid
        if exists is None:
            template = sources[0]
            clash = conn.execute(
                "SELECT id FROM entities WHERE agent_id = ? AND LOWER(label) = LOWER(?)",
                (uscope, template["label"]),
            ).fetchone()
            if clash is not None:
                target = clash[0]  # label-unique index: reuse the user-scope row
            else:
                mut.insert(
                    "entities",
                    "INSERT INTO entities (id, label, type, subtype, canonical_label, canonical_key, wikidata_id, "
                    "description, wikipedia_url, asserted_at, agent_id, source_agent_id, extraction_count, "
                    "retrieval_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        mid, template["label"], template["type"], template["subtype"],
                        template["canonical_label"] or template["label"], template["canonical_key"],
                        template["wikidata_id"], template["description"], template["wikipedia_url"],
                        template["asserted_at"], uscope, template["agent_id"],
                        template["extraction_count"], template["retrieval_count"],
                    ),
                )
        for s in sources:
            mirror_target[s["id"]] = target
        mirror_sources.setdefault(target, []).extend(s["id"] for s in sources)
    if mirror_sources:
        section["mirrors"] = {mid: sorted(set(ids)) for mid, ids in sorted(mirror_sources.items())}

    lift_counter = {"n": 0}

    def final_of(row: dict) -> tuple:
        subj, obj = resolve_id(row["subject_id"]), resolve_id(row["object_id"])
        if routes_user_level(row, subj, obj):
            subj = mirror_target.get(subj, subj)
            obj = mirror_target.get(obj, obj)
            src = row["source_agent_id"] or (row["agent_id"] if row["agent_id"] != uscope else None)
            if row["agent_id"] != uscope:
                lift_counter["n"] += 1
            return (uscope, subj, obj, src)
        return (row["agent_id"], subj, obj, row["source_agent_id"])

    _repoint_and_fold_relations(mut, scopes=scopes_of_user, final_of=final_of, section=section, report=report)
    section["lifted_relations"] = lift_counter["n"]

    # ---- conflict sheet (§3.3: coexistence; the owner adjudicates) ------------
    functional = [r[0] for r in conn.execute("SELECT name FROM relation_types WHERE is_functional = 1 ORDER BY name")]
    conflict_sheet: list[dict] = []
    for predicate in functional:
        rows = _rows(
            conn,
            """
            SELECT r.id, r.object_id, r.asserted_at, r.source_agent_id, e.label AS object_label
            FROM relations r LEFT JOIN entities e ON e.id = r.object_id AND e.agent_id = r.agent_id
            WHERE r.agent_id = ? AND r.subject_id = ? AND r.predicate = ? AND r.valid_until IS NULL
            ORDER BY r.asserted_at, r.id
            """,
            (uscope, principal_id, predicate),
        )
        if len({r["object_id"] for r in rows}) >= 2:
            conflict_sheet.append({
                "predicate": predicate,
                "values": [
                    {
                        "relation_id": r["id"],
                        "object_id": r["object_id"],
                        "object_label": r["object_label"],
                        "source_scope": r["source_agent_id"],
                        "asserted_at": r["asserted_at"],
                    }
                    for r in rows
                ],
            })
    section["conflict_sheet"] = conflict_sheet

    # ---- invariants + conservation --------------------------------------------
    for scope in agent_scopes:
        leftover, _t, _a = stage1_merge_set(conn, scope)
        if leftover is not None:
            raise MigrationError(f"stage2 {user}: agent scope {scope} still holds a user-referent entity")
    principals = _rows(conn, "SELECT id FROM entities WHERE agent_id = ? AND canonical_key = ?", (uscope, key))
    if len(principals) != 1:
        raise MigrationError(f"stage2 {user}: expected exactly one principal, found {len(principals)}")

    if mut.mutations == 0:
        return None
    after = _scope_counts(conn, scopes_of_user)
    section["conservation"] = mut.assert_conservation(before, after, f"stage2 {user}")
    section["mutations"] = mut.mutations
    report.mutations += mut.mutations
    return section


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def group_scopes(scopes: list[str]) -> tuple[dict[str, list[str]], list[str]]:
    """``user -> [agent scopes]`` (slash-bearing scopes only; the user's own
    ``_user`` scope excluded — stage 2 reads it directly) plus the skipped
    bare scopes (§6.2.1: inventing a user namespace for them would be the rule
    fabricating identity)."""
    users: dict[str, list[str]] = {}
    bare: list[str] = []
    for scope in sorted(scopes):
        user, sep, slug = scope.partition("/")
        if not sep:
            bare.append(scope)
            continue
        if slug == RESERVED_USER_SLUG:
            users.setdefault(user, [])
            continue
        users.setdefault(user, []).append(scope)
    return users, bare


def migrate_user(
    conn: sqlite3.Connection,
    user: str,
    agent_scopes: list[str],
    report: MigrationReport,
    *,
    now: datetime | None = None,
) -> None:
    """Both stages for ONE user (the prod CLI wraps this in
    ``BEGIN IMMEDIATE`` so a failed user rolls back alone)."""
    for scope in agent_scopes:
        section = run_stage1_scope(conn, scope, report, now=now)
        if section is not None:
            report.stage1[scope] = section
    section2 = _stage2_user(conn, user, agent_scopes, report, now=now)
    if section2 is not None:
        report.stage2[user] = section2


def merge_reports(master: MigrationReport, part: MigrationReport) -> None:
    """Fold a per-user report into the master (the CLI's per-user-transaction
    bookkeeping; key spaces are disjoint by construction)."""
    master.stage1.update(part.stage1)
    master.stage2.update(part.stage2)
    master.aborted_users.update(part.aborted_users)
    master.id_map.update(part.id_map)
    master.skipped_bare_scopes = sorted(set(master.skipped_bare_scopes) | set(part.skipped_bare_scopes))
    master.mutations += part.mutations


def migrate(
    conn: sqlite3.Connection,
    *,
    now: datetime | None = None,
    users: list[str] | None = None,
) -> MigrationReport:
    """Run both stages over every eligible user. No transaction management —
    the caller owns commit/rollback (the prod CLI wraps each user in
    ``BEGIN IMMEDIATE``; the eval builder runs one deterministic pass).
    ``now`` is the prod-mode wall clock used for alias ``created_at`` only;
    eval mode passes ``None`` (every timestamp derives from existing rows)."""
    report = MigrationReport()
    by_user, bare = group_scopes(all_scopes(conn))
    report.skipped_bare_scopes = bare
    # Stage 1 also runs on bare scopes: the per-scope merge is scope-local and
    # safe; only the lift needs a user segment.
    for scope in bare:
        section = run_stage1_scope(conn, scope, report, now=now)
        if section is not None:
            report.stage1[scope] = section
    for user in sorted(by_user):
        if users is not None and user not in users:
            continue
        migrate_user(conn, user, by_user[user], report, now=now)
    return report
