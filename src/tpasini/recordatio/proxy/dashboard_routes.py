"""Dashboard and graph-inspection routes for the proxy app."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse

from tpasini.recordatio.models import RetrievalContext
from tpasini.recordatio.proxy.cost_log import CostLog
from tpasini.recordatio.proxy.memory_ops import InvalidOpError, MemoryOpError, apply_memory_op
from tpasini.recordatio.storage.base import SQLiteStore
from tpasini.recordatio.storage.memory_feedback import MemoryFeedbackStore

_MEMORIES_HTML = (Path(__file__).parent.parent / "ui" / "memories.html").read_text()
_COSTS_HTML = (Path(__file__).parent.parent / "ui" / "costs.html").read_text()
_FEEDBACK_HTML = (Path(__file__).parent.parent / "ui" / "feedback.html").read_text()
_HTML_NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}
_RANGE_TO_HOURS: dict[str, int | None] = {
    "24h": 24,
    "7d": 7 * 24,
    "30d": 30 * 24,
    "all": None,
}


@dataclass(frozen=True)
class DashboardRouteHandlers:
    memories_ui: Callable[[], Awaitable[HTMLResponse]]
    costs_ui: Callable[[], Awaitable[HTMLResponse]]
    costs_data: Callable[[Request], Awaitable[dict]]
    memories_agents: Callable[[], Awaitable[dict]]
    memories: Callable[[Request], Awaitable[dict]]


@dataclass(frozen=True)
class CostsDataRoute:
    get_cost_log: Callable[[], CostLog]

    async def __call__(self, request: Request) -> dict:
        return await costs_data(request, get_cost_log=self.get_cost_log)


@dataclass(frozen=True)
class MemoriesAgentsRoute:
    get_store: Callable[[], SQLiteStore]

    async def __call__(self) -> dict:
        return await memories_agents(get_store=self.get_store)


@dataclass(frozen=True)
class MemoriesDataRoute:
    get_store: Callable[[], SQLiteStore]

    async def __call__(self, request: Request) -> dict:
        return await memories(request, get_store=self.get_store)


@dataclass(frozen=True)
class FeedbackDataRoute:
    get_feedback_store: Callable[[], MemoryFeedbackStore]

    async def __call__(self) -> dict:
        return await feedback_data(get_feedback_store=self.get_feedback_store)


@dataclass(frozen=True)
class FeedbackOpRoute:
    action: str  # "apply" | "dismiss"
    get_store: Callable[[], SQLiteStore]
    get_feedback_store: Callable[[], MemoryFeedbackStore]

    async def __call__(self, op_id: int, request: Request) -> Response:
        return await _decide_op(
            op_id, self.action, request,
            get_store=self.get_store, get_feedback_store=self.get_feedback_store,
        )


@dataclass(frozen=True)
class FeedbackApplyAllRoute:
    get_store: Callable[[], SQLiteStore]
    get_feedback_store: Callable[[], MemoryFeedbackStore]

    async def __call__(self) -> dict:
        return await apply_all_pending(
            get_store=self.get_store, get_feedback_store=self.get_feedback_store,
        )


async def memories_ui() -> HTMLResponse:
    return HTMLResponse(_MEMORIES_HTML, headers=_HTML_NO_CACHE_HEADERS)


async def costs_ui() -> HTMLResponse:
    return HTMLResponse(_COSTS_HTML, headers=_HTML_NO_CACHE_HEADERS)


async def feedback_ui() -> HTMLResponse:
    return HTMLResponse(_FEEDBACK_HTML, headers=_HTML_NO_CACHE_HEADERS)


def _feedback_record_json(rec: object, ops: list) -> dict:
    return {
        "id": rec.id,
        "user_id": rec.user_id,
        "day": rec.day,
        "created_at": rec.created_at,
        "summary": rec.summary,
        "findings": rec.findings,
        "ops": [
            {
                "id": op.id,
                "op_type": op.op_type,
                "target": op.target,
                "reason": op.reason,
                "status": op.status,
                "decided_at": op.decided_at,
                "decided_result": op.decided_result,
            }
            for op in ops
        ],
    }


async def feedback_data(*, get_feedback_store: Callable[[], MemoryFeedbackStore]) -> dict:
    """Every user's nightly memory-feedback records + their ops (operator view)."""
    store = get_feedback_store()
    records = store.list_all()
    return {"records": [_feedback_record_json(rec, store.list_ops(rec.id)) for rec in records]}


async def _decide_op(
    op_id: int,
    action: str,
    request: Request,
    *,
    get_store: Callable[[], SQLiteStore],
    get_feedback_store: Callable[[], MemoryFeedbackStore],
) -> Response:
    """Shared apply/dismiss handler for a dashboard op. The dashboard sits
    behind Caddy basic-auth, so there is no bearer gate here (mirrors the
    other ``/…/data`` routes); the op's owning user is derived from its
    feedback record, never trusted from the client."""
    fb = get_feedback_store()
    op = fb.get_op(op_id)
    if op is None:
        return JSONResponse({"error": "op not found"}, status_code=404)
    rec = fb.get_feedback(op.feedback_id)
    if rec is None:
        return JSONResponse({"error": "op not found"}, status_code=404)
    if op.status != "pending":
        return JSONResponse({"error": f"op already {op.status}", "status": op.status}, status_code=409)
    if action == "dismiss":
        reason = None
        try:
            body = await request.json()
            if isinstance(body, dict) and isinstance(body.get("reason"), str):
                reason = body["reason"]
        except Exception:
            pass
        fb.mark_op(op_id, status="dismissed", result=reason)
        return JSONResponse({"ok": True, "op_id": op_id, "status": "dismissed"})
    try:
        result = apply_memory_op(get_store(), rec.user_id, op.op_type, op.target)
    except InvalidOpError as exc:
        # A malformed/incomplete suggestion (e.g. a truncated uuid) can never
        # be applied — auto-reject it so it doesn't linger pending forever.
        fb.mark_op(op_id, status="dismissed", result=f"auto-rejected: {exc}")
        return JSONResponse({"ok": True, "op_id": op_id, "status": "rejected", "reason": str(exc)})
    except MemoryOpError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    fb.mark_op(op_id, status="applied", result=result)
    return JSONResponse({"ok": True, "op_id": op_id, "status": "applied", "result": result})


async def apply_all_pending(
    *,
    get_store: Callable[[], SQLiteStore],
    get_feedback_store: Callable[[], MemoryFeedbackStore],
) -> dict:
    """Apply every pending op across all users: valid ones are applied,
    invalid ones (incomplete uuid, not found, malformed) are auto-rejected.
    Returns a summary the dashboard toasts."""
    store = get_store()
    fb = get_feedback_store()
    applied = 0
    rejected = 0
    failed = 0
    for rec in fb.list_all():
        for op in fb.list_ops(rec.id):
            if op.status != "pending":
                continue
            try:
                result = apply_memory_op(store, rec.user_id, op.op_type, op.target)
            except InvalidOpError as exc:
                fb.mark_op(op.id, status="dismissed", result=f"auto-rejected: {exc}")
                rejected += 1
                continue
            except MemoryOpError:
                failed += 1
                continue
            fb.mark_op(op.id, status="applied", result=result)
            applied += 1
    return {"ok": True, "applied": applied, "rejected": rejected, "failed": failed}


async def costs_data(request: Request, *, get_cost_log: Callable[[], CostLog]) -> dict:
    """Aggregated cost summary for the dashboard."""
    rng = request.query_params.get("range", "7d")
    hours = _RANGE_TO_HOURS.get(rng)
    if hours is None and rng != "all":
        return JSONResponse(
            {"error": f"range must be one of {sorted(_RANGE_TO_HOURS)}"},
            status_code=400,
        )
    since = (datetime.now(UTC) - timedelta(hours=hours)).isoformat() if hours else None
    cost_log = get_cost_log()

    return {
        "range": rng,
        "totals": cost_log.query_totals(since=since),
        "by_kind": cost_log.query_aggregate(group_by="kind", since=since),
        "by_model": cost_log.query_aggregate(group_by="model", since=since, limit=15),
        "by_agent": cost_log.query_aggregate(group_by="agent", since=since, limit=15),
    }


async def memories_agents(*, get_store: Callable[[], SQLiteStore]) -> dict:
    store = get_store()
    rows = store._conn.execute(
        """
        WITH agent_ids AS (
            SELECT agent_id FROM knowledge_units
            UNION SELECT agent_id FROM entities
            UNION SELECT agent_id FROM relations
            UNION SELECT agent_id FROM entity_aliases
            UNION SELECT agent_id FROM entity_identifiers
            UNION SELECT agent_id FROM entity_attributes
            UNION SELECT agent_id FROM entity_mentions
            UNION SELECT agent_id FROM unit_topics
            UNION SELECT agent_id FROM entity_topics
        ),
        ku AS (
            SELECT agent_id, COUNT(*) AS count_units, MAX(asserted_at) AS last_seen
            FROM knowledge_units
            GROUP BY agent_id
        ),
        en AS (
            SELECT agent_id, COUNT(*) AS count_entities, MAX(asserted_at) AS last_seen
            FROM entities
            GROUP BY agent_id
        ),
        rel AS (
            SELECT agent_id, COUNT(*) AS count_relations, MAX(asserted_at) AS last_seen
            FROM relations
            WHERE valid_until IS NULL
            GROUP BY agent_id
        ),
        attr AS (
            SELECT agent_id, COUNT(*) AS count_attributes, MAX(asserted_at) AS last_seen
            FROM entity_attributes
            WHERE valid_until IS NULL
            GROUP BY agent_id
        ),
        mention AS (
            SELECT agent_id, COUNT(*) AS count_mentions, MAX(asserted_at) AS last_seen
            FROM entity_mentions
            GROUP BY agent_id
        ),
        alias AS (
            SELECT agent_id, COUNT(*) AS count_aliases, MAX(created_at) AS last_seen
            FROM entity_aliases
            GROUP BY agent_id
        ),
        ident AS (
            SELECT agent_id, COUNT(*) AS count_identifiers, MAX(created_at) AS last_seen
            FROM entity_identifiers
            GROUP BY agent_id
        )
        SELECT
            a.agent_id,
            COALESCE(ku.count_units, 0) AS count_units,
            COALESCE(en.count_entities, 0) AS count_entities,
            COALESCE(rel.count_relations, 0) AS count_relations,
            COALESCE(attr.count_attributes, 0) AS count_attributes,
            COALESCE(mention.count_mentions, 0) AS count_mentions,
            COALESCE(alias.count_aliases, 0) AS count_aliases,
            COALESCE(ident.count_identifiers, 0) AS count_identifiers,
            MAX(
                COALESCE(ku.last_seen, ''),
                COALESCE(en.last_seen, ''),
                COALESCE(rel.last_seen, ''),
                COALESCE(attr.last_seen, ''),
                COALESCE(mention.last_seen, ''),
                COALESCE(alias.last_seen, ''),
                COALESCE(ident.last_seen, '')
            ) AS last_seen
        FROM agent_ids a
        LEFT JOIN ku ON ku.agent_id = a.agent_id
        LEFT JOIN en ON en.agent_id = a.agent_id
        LEFT JOIN rel ON rel.agent_id = a.agent_id
        LEFT JOIN attr ON attr.agent_id = a.agent_id
        LEFT JOIN mention ON mention.agent_id = a.agent_id
        LEFT JOIN alias ON alias.agent_id = a.agent_id
        LEFT JOIN ident ON ident.agent_id = a.agent_id
        WHERE a.agent_id != 'default'
        ORDER BY a.agent_id
        """
    ).fetchall()

    agents: list[dict] = []
    users: dict[str, int] = {}
    totals = {
        "agents": 0,
        "users": 0,
        "units": 0,
        "entities": 0,
        "relations": 0,
        "attributes": 0,
        "mentions": 0,
        "aliases": 0,
        "identifiers": 0,
    }
    for row in rows:
        agent_id = row["agent_id"]
        user, _, name = agent_id.partition("/")
        user_label = user or "(none)"
        agent_label = name if name else "(root)"
        item = {
            "agent_id": agent_id,
            "user": user_label,
            "name": agent_label,
            "last_seen": row["last_seen"] or None,
            "counts": {
                "units": row["count_units"],
                "entities": row["count_entities"],
                "relations": row["count_relations"],
                "attributes": row["count_attributes"],
                "mentions": row["count_mentions"],
                "aliases": row["count_aliases"],
                "identifiers": row["count_identifiers"],
            },
        }
        agents.append(item)
        users[user_label] = users.get(user_label, 0) + 1
        totals["agents"] += 1
        for key, value in item["counts"].items():
            totals[key] += value
    totals["users"] = len(users)

    return {"agents": agents, "users": users, "totals": totals}

async def memories(request: Request, *, get_store: Callable[[], SQLiteStore]) -> dict:
    agent_id = request.query_params.get("agent_id")
    if not agent_id:
        return Response(status_code=400, content='{"error": "agent_id query param required"}')
    source_agent_id = request.query_params.get("source_agent_id") or None

    store = get_store()
    units = store.search(RetrievalContext(query="", top_k=200), agent_id, source_agent_id=source_agent_id)

    entity_rows = store._conn.execute(
        """
        SELECT id, label, type, subtype, canonical_label, canonical_key,
               wikidata_id, description, source_agent_id
        FROM entities
        WHERE agent_id = ? AND (? IS NULL OR source_agent_id = ?)
        ORDER BY rowid DESC
        LIMIT 200
        """,
        (agent_id, source_agent_id, source_agent_id),
    ).fetchall()

    alias_rows = store._conn.execute(
        """
        SELECT ea.id, ea.entity_id, e.label AS entity_label, ea.alias, ea.source, ea.created_at
        FROM entity_aliases ea
        JOIN entities e ON e.id = ea.entity_id AND e.agent_id = ea.agent_id
        WHERE ea.agent_id = ?
        ORDER BY ea.created_at DESC, ea.rowid DESC
        LIMIT 500
        """,
        (agent_id,),
    ).fetchall()

    identifier_rows = store._conn.execute(
        """
        SELECT ei.id, ei.entity_id, e.label AS entity_label, ei.namespace,
               ei.value, ei.source, ei.created_at
        FROM entity_identifiers ei
        JOIN entities e ON e.id = ei.entity_id AND e.agent_id = ei.agent_id
        WHERE ei.agent_id = ?
        ORDER BY ei.created_at DESC, ei.rowid DESC
        LIMIT 500
        """,
        (agent_id,),
    ).fetchall()

    attribute_rows = store._conn.execute(
        """
        SELECT ea.id, ea.entity_id, e.label AS entity_label, ea.attribute_type,
               ea.value, ea.asserted_at, ea.valid_from, ea.valid_until, ea.provenance
        FROM entity_attributes ea
        JOIN entities e ON e.id = ea.entity_id AND e.agent_id = ea.agent_id
        WHERE ea.agent_id = ?
        ORDER BY CASE WHEN ea.valid_until IS NULL THEN 0 ELSE 1 END, ea.asserted_at DESC, ea.rowid DESC
        LIMIT 500
        """,
        (agent_id,),
    ).fetchall()

    mention_rows = store._conn.execute(
        """
        SELECT em.id, em.entity_id, e.label AS entity_label, em.source_kind,
               em.source_id, ku.content AS source_label, em.snippet, em.asserted_at
        FROM entity_mentions em
        JOIN entities e ON e.id = em.entity_id AND e.agent_id = em.agent_id
        LEFT JOIN knowledge_units ku ON ku.id = em.source_id AND ku.agent_id = em.agent_id
        WHERE em.agent_id = ?
        ORDER BY em.asserted_at DESC, em.rowid DESC
        LIMIT 500
        """,
        (agent_id,),
    ).fetchall()

    now_iso = datetime.now(UTC).isoformat()
    rel_rows = store._conn.execute(
        """
        SELECT id, subject_id, predicate, object_id, asserted_at, valid_from, valid_until, provenance, source_agent_id
        FROM relations
        WHERE agent_id = ?
          AND (valid_until IS NULL OR valid_until > ?)
          AND (? IS NULL OR source_agent_id = ?)
        ORDER BY rowid DESC
        LIMIT 500
        """,
        (agent_id, now_iso, source_agent_id, source_agent_id),
    ).fetchall()

    unit_topic_rows = store._conn.execute(
        "SELECT unit_id, topic_id FROM unit_topics WHERE agent_id = ?", (agent_id,)
    ).fetchall()

    entity_topic_rows = store._conn.execute(
        "SELECT entity_id, topic_id FROM entity_topics WHERE agent_id = ?", (agent_id,)
    ).fetchall()

    topic_rows = store._conn.execute(
        """
        SELECT DISTINCT t.id, t.name, t.description
        FROM topics t
        WHERE t.id IN (
            SELECT topic_id FROM unit_topics WHERE agent_id = ?
            UNION
            SELECT topic_id FROM entity_topics WHERE agent_id = ?
        )
        ORDER BY t.name
        LIMIT 200
        """,
        (agent_id, agent_id),
    ).fetchall()

    entity_type_rows = store._conn.execute(
        """
        SELECT id, name, description, source, created_at
        FROM entity_types
        WHERE agent_id IN ('default', ?)
        ORDER BY name
        """,
        (agent_id,),
    ).fetchall()

    attribute_type_rows = store._conn.execute(
        """
        SELECT id, name, description, value_type, source, created_at
        FROM attribute_types
        WHERE agent_id IN ('default', ?)
        ORDER BY name
        """,
        (agent_id,),
    ).fetchall()

    relation_type_rows = store._conn.execute(
        """
        SELECT id, name, description, source, created_at, is_functional
        FROM relation_types
        ORDER BY name
        """
    ).fetchall()

    node_map: dict[str, dict] = {}
    for row in entity_rows:
        node_map[row["id"]] = {
            "id": row["id"],
            "label": row["label"],
            "kind": "entity",
            "type": row["type"],
            "subtype": row["subtype"],
            "canonical_label": row["canonical_label"],
            "canonical_key": row["canonical_key"],
            "wikidata_id": row["wikidata_id"],
            "description": row["description"],
            "source_agent_id": row["source_agent_id"],
        }
    for unit in units:
        node_map[unit.id] = {"id": unit.id, "label": unit.content[:80], "kind": unit.type.value}
    for row in topic_rows:
        node_map[row["id"]] = {
            "id": row["id"],
            "label": row["name"],
            "kind": "topic",
            "description": row["description"],
        }
    for row in attribute_rows[:300]:
        node_id = f"attribute:{row['id']}"
        label = f"{row['attribute_type']}: {row['value']}"
        node_map[node_id] = {
            "id": node_id,
            "label": label[:120],
            "kind": "attribute",
            "entity_id": row["entity_id"],
            "attribute_type": row["attribute_type"],
            "value": row["value"],
            "active": row["valid_until"] is None,
            "provenance": row["provenance"],
        }

    edges: list[dict] = []
    seen_edges: set[tuple[str, str, str]] = set()
    for row in rel_rows:
        if row["subject_id"] in node_map and row["object_id"] in node_map:
            key = (row["subject_id"], row["predicate"], row["object_id"])
            if key not in seen_edges:
                seen_edges.add(key)
                edges.append({
                    "id": row["id"],
                    "source": row["subject_id"],
                    "target": row["object_id"],
                    "label": row["predicate"],
                    "kind": "relation",
                    "provenance": row["provenance"],
                    "source_agent_id": row["source_agent_id"],
                })
    for row in attribute_rows[:300]:
        node_id = f"attribute:{row['id']}"
        if row["entity_id"] in node_map and node_id in node_map:
            key = (row["entity_id"], row["attribute_type"], node_id)
            if key not in seen_edges:
                seen_edges.add(key)
                edges.append({
                    "source": row["entity_id"],
                    "target": node_id,
                    "label": row["attribute_type"],
                    "kind": "attribute",
                })
    for row in mention_rows[:300]:
        if row["entity_id"] in node_map and row["source_id"] in node_map:
            key = (row["entity_id"], "mentioned_in", row["source_id"])
            if key not in seen_edges:
                seen_edges.add(key)
                edges.append({
                    "source": row["entity_id"],
                    "target": row["source_id"],
                    "label": "mentioned_in",
                    "kind": "mention",
                })
    for row in unit_topic_rows:
        uid, tid = row["unit_id"], row["topic_id"]
        if uid in node_map and tid in node_map:
            key = (uid, "tagged_with", tid)
            if key not in seen_edges:
                seen_edges.add(key)
                edges.append({"source": uid, "target": tid, "label": "tagged_with", "kind": "topic"})
    for row in entity_topic_rows:
        eid, tid = row["entity_id"], row["topic_id"]
        if eid in node_map and tid in node_map:
            key = (eid, "tagged_with", tid)
            if key not in seen_edges:
                seen_edges.add(key)
                edges.append({"source": eid, "target": tid, "label": "tagged_with", "kind": "topic"})

    nodes = list(node_map.values())
    id_to_label = {node["id"]: node["label"] for node in node_map.values()}
    relations_flat = [
        {
            "id": edge.get("id"),
            "subject": id_to_label[edge["source"]],
            "predicate": edge["label"],
            "object": id_to_label[edge["target"]],
            "kind": edge.get("kind", "relation"),
            "provenance": edge.get("provenance"),
            "source_agent_id": edge.get("source_agent_id"),
        }
        for edge in edges
        if edge.get("kind") == "relation"
    ]
    return {
        "agent_id": agent_id,
        "source_agent_id": source_agent_id,
        "counts": {
            "nodes": len(node_map),
            "edges": len(edges),
            "units": len(units),
            "entities": len(entity_rows),
            "relations": len(relations_flat),
            "attributes": len(attribute_rows),
            "mentions": len(mention_rows),
            "aliases": len(alias_rows),
            "identifiers": len(identifier_rows),
            "entity_types": len(entity_type_rows),
            "attribute_types": len(attribute_type_rows),
            "relation_types": len(relation_type_rows),
        },
        "nodes": nodes,
        "edges": edges,
        "units": [
            {
                "id": unit.id,
                "type": unit.type.value,
                "content": unit.content,
                "asserted_at": unit.asserted_at.isoformat(),
                "source_agent_id": source_agent_id,
            }
            for unit in units
        ],
        "entities": [
            {
                "id": row["id"],
                "label": row["label"],
                "type": row["type"],
                "subtype": row["subtype"],
                "canonical_label": row["canonical_label"],
                "canonical_key": row["canonical_key"],
                "wikidata_id": row["wikidata_id"],
                "description": row["description"],
                "source_agent_id": row["source_agent_id"],
            }
            for row in entity_rows
        ],
        "entity_aliases": [
            {
                "id": row["id"],
                "entity_id": row["entity_id"],
                "entity_label": row["entity_label"],
                "alias": row["alias"],
                "source": row["source"],
                "created_at": row["created_at"],
            }
            for row in alias_rows
        ],
        "entity_identifiers": [
            {
                "id": row["id"],
                "entity_id": row["entity_id"],
                "entity_label": row["entity_label"],
                "namespace": row["namespace"],
                "value": row["value"],
                "source": row["source"],
                "created_at": row["created_at"],
            }
            for row in identifier_rows
        ],
        "entity_attributes": [
            {
                "id": row["id"],
                "entity_id": row["entity_id"],
                "entity_label": row["entity_label"],
                "attribute_type": row["attribute_type"],
                "value": row["value"],
                "asserted_at": row["asserted_at"],
                "valid_from": row["valid_from"],
                "valid_until": row["valid_until"],
                "active": row["valid_until"] is None,
                "provenance": row["provenance"],
            }
            for row in attribute_rows
        ],
        "entity_mentions": [
            {
                "id": row["id"],
                "entity_id": row["entity_id"],
                "entity_label": row["entity_label"],
                "source_kind": row["source_kind"],
                "source_id": row["source_id"],
                "source_label": row["source_label"],
                "snippet": row["snippet"],
                "asserted_at": row["asserted_at"],
            }
            for row in mention_rows
        ],
        "schema": {
            "entity_types": [
                {
                    "id": row["id"],
                    "name": row["name"],
                    "description": row["description"],
                    "source": row["source"],
                    "created_at": row["created_at"],
                }
                for row in entity_type_rows
            ],
            "attribute_types": [
                {
                    "id": row["id"],
                    "name": row["name"],
                    "description": row["description"],
                    "value_type": row["value_type"],
                    "source": row["source"],
                    "created_at": row["created_at"],
                }
                for row in attribute_type_rows
            ],
            "relation_types": [
                {
                    "id": row["id"],
                    "name": row["name"],
                    "description": row["description"],
                    "source": row["source"],
                    "created_at": row["created_at"],
                    "is_functional": bool(row["is_functional"]),
                }
                for row in relation_type_rows
            ],
        },
        "relations": relations_flat,
    }


def register_dashboard_routes(
    app: FastAPI,
    *,
    get_store: Callable[[], SQLiteStore],
    get_cost_log: Callable[[], CostLog],
    get_feedback_store: Callable[[], MemoryFeedbackStore] | None = None,
) -> DashboardRouteHandlers:
    costs_data_handler = CostsDataRoute(get_cost_log)
    memories_agents_handler = MemoriesAgentsRoute(get_store)
    memories_handler = MemoriesDataRoute(get_store)

    app.add_api_route(
        "/memories",
        memories_ui,
        response_class=HTMLResponse,
        include_in_schema=False,
        methods=["GET"],
    )
    app.add_api_route(
        "/costs",
        costs_ui,
        response_class=HTMLResponse,
        include_in_schema=False,
        methods=["GET"],
    )
    app.add_api_route("/costs/data", costs_data_handler, include_in_schema=False, methods=["GET"])
    app.add_api_route("/memories/agents", memories_agents_handler, include_in_schema=False, methods=["GET"])
    app.add_api_route("/memories/data", memories_handler, include_in_schema=False, methods=["GET"])

    # Nightly memory-feedback dashboard (apply/dismiss the analyst's proposed
    # ops). Registered only when a feedback store is supplied so tests/embeds
    # without one keep the prior surface.
    if get_feedback_store is not None:
        app.add_api_route(
            "/feedback", feedback_ui, response_class=HTMLResponse,
            include_in_schema=False, methods=["GET"],
        )
        app.add_api_route(
            "/feedback/data", FeedbackDataRoute(get_feedback_store),
            include_in_schema=False, methods=["GET"],
        )
        app.add_api_route(
            "/feedback/apply-all",
            FeedbackApplyAllRoute(get_store, get_feedback_store),
            include_in_schema=False, methods=["POST"],
        )
        app.add_api_route(
            "/feedback/ops/{op_id}/apply",
            FeedbackOpRoute("apply", get_store, get_feedback_store),
            include_in_schema=False, methods=["POST"],
        )
        app.add_api_route(
            "/feedback/ops/{op_id}/dismiss",
            FeedbackOpRoute("dismiss", get_store, get_feedback_store),
            include_in_schema=False, methods=["POST"],
        )

    return DashboardRouteHandlers(
        memories_ui=memories_ui,
        costs_ui=costs_ui,
        costs_data=costs_data_handler,
        memories_agents=memories_agents_handler,
        memories=memories_handler,
    )
