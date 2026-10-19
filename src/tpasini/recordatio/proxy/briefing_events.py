"""Briefing-event hydration helpers."""

import logging
import sqlite3

from fastapi import Request, Response
from fastapi.responses import JSONResponse

from tpasini.recordatio.proxy.briefing_log import BriefingLog
from tpasini.recordatio.storage.base import SQLiteStore


def hydrate_briefing_row(row: dict, store: SQLiteStore, logger: logging.Logger) -> dict:
    """Hydrate a briefing-event row into the API payload shape."""
    conn = sqlite3.connect(str(store.db_path))
    conn.row_factory = sqlite3.Row
    try:
        return hydrate_briefing_row_with_conn(row, conn, logger)
    finally:
        conn.close()


def hydrate_briefing_row_with_conn(row: dict, conn: sqlite3.Connection, logger: logging.Logger) -> dict:
    """Hydrate current-store content, not an immutable snapshot of the briefing.

    Unresolved IDs were not returned; they do not prove deletion or absence.
    Error codes distinguish failed queries (including endpoint lookups) from
    successful queries that could not resolve a recorded ID.
    """
    agent_id = row.get("agent_id")
    unit_ids: list[str] = list(row.get("recalled_unit_ids") or [])
    entity_ids: list[str] = list(row.get("recalled_entity_ids") or [])
    relation_ids: list[str] = list(row.get("recalled_relation_ids") or [])

    errors: dict[str, list[str]] = {"units": [], "entities": [], "relations": []}
    units_out = _hydrate_units(row, conn, agent_id, unit_ids, logger, errors["units"])
    entities_out = _hydrate_entities(row, conn, agent_id, entity_ids, logger, errors["entities"])
    relations_out = _hydrate_relations(row, conn, agent_id, relation_ids, logger, errors["relations"])
    hydration = {}
    for kind, ids, records in (
        ("units", unit_ids, units_out),
        ("entities", entity_ids, entities_out),
        ("relations", relation_ids, relations_out),
    ):
        if ids and not agent_id:
            errors[kind].append("missing_agent_id")
        hydrated_ids = {record["id"] for record in records}
        unresolved = [id_ for id_ in ids if id_ not in hydrated_ids]
        hydration[kind] = {
            "hydrated_count": len(records),
            "unresolved_ids": unresolved,
            "errors": errors[kind],
            "complete": (
                not errors[kind] and not unresolved
                and len(records) == int(row.get(f"count_{kind}") or 0)
            ),
        }

    return {
        "id": row["id"],
        "ts": row["ts"],
        "request_id": row["request_id"],
        "agent_id": agent_id,
        "source_agent_id": row.get("source_agent_id"),
        "query_text": row.get("query_text"),
        "topic_hint": row.get("topic_hint"),
        "count_units": int(row.get("count_units") or 0),
        "count_entities": int(row.get("count_entities") or 0),
        "count_relations": int(row.get("count_relations") or 0),
        "recalled_unit_ids": unit_ids,
        "recalled_entity_ids": entity_ids,
        "recalled_relation_ids": relation_ids,
        "content_source": "current_store",
        "hydration": hydration,
        "hydration_complete": all(status["complete"] for status in hydration.values()),
        "units": units_out,
        "entities": entities_out,
        "relations": relations_out,
    }


def _hydrate_units(
    row: dict,
    conn: sqlite3.Connection,
    agent_id: str | None,
    unit_ids: list[str],
    logger: logging.Logger,
    errors: list[str],
) -> list[dict]:
    units_out: list[dict] = []
    if not unit_ids or not agent_id:
        return units_out
    placeholders = ",".join("?" * len(unit_ids))
    try:
        unit_rows = conn.execute(
            f"SELECT id, type, content, asserted_at FROM knowledge_units "
            f"WHERE agent_id = ? AND id IN ({placeholders})",
            (agent_id, *unit_ids),
        ).fetchall()
    except Exception:
        logger.exception("failed to hydrate units for briefing %s", row.get("request_id"))
        errors.append("query_failed")
        unit_rows = []
    by_id = {r["id"]: r for r in unit_rows}
    for unit_id in unit_ids:
        record = by_id.get(unit_id)
        if record is None:
            continue
        units_out.append({
            "id": record["id"],
            "type": record["type"],
            "content": record["content"],
            "asserted_at": record["asserted_at"],
        })
    return units_out


def _hydrate_entities(
    row: dict,
    conn: sqlite3.Connection,
    agent_id: str | None,
    entity_ids: list[str],
    logger: logging.Logger,
    errors: list[str],
) -> list[dict]:
    entities_out: list[dict] = []
    if not entity_ids or not agent_id:
        return entities_out
    placeholders = ",".join("?" * len(entity_ids))
    try:
        ent_rows = conn.execute(
            f"SELECT id, label, wikidata_id, description FROM entities "
            f"WHERE agent_id = ? AND id IN ({placeholders})",
            (agent_id, *entity_ids),
        ).fetchall()
    except Exception:
        logger.exception("failed to hydrate entities for briefing %s", row.get("request_id"))
        errors.append("query_failed")
        ent_rows = []
    by_id = {r["id"]: r for r in ent_rows}
    for entity_id in entity_ids:
        record = by_id.get(entity_id)
        if record is None:
            continue
        entities_out.append({
            "id": record["id"],
            "label": record["label"],
            "wikidata_id": record["wikidata_id"],
            "description": record["description"],
        })
    return entities_out


def _hydrate_relations(
    row: dict,
    conn: sqlite3.Connection,
    agent_id: str | None,
    relation_ids: list[str],
    logger: logging.Logger,
    errors: list[str],
) -> list[dict]:
    relations_out: list[dict] = []
    if not relation_ids or not agent_id:
        return relations_out
    placeholders = ",".join("?" * len(relation_ids))
    try:
        rel_rows = conn.execute(
            f"SELECT id, subject_id, predicate, object_id FROM relations "
            f"WHERE agent_id = ? AND id IN ({placeholders})",
            (agent_id, *relation_ids),
        ).fetchall()
    except Exception:
        logger.exception("failed to hydrate relations for briefing %s", row.get("request_id"))
        errors.append("query_failed")
        rel_rows = []
    label_by_id = _relation_endpoint_labels(row, conn, agent_id, rel_rows, logger, errors)
    rel_by_id = {r["id"]: r for r in rel_rows}
    for relation_id in relation_ids:
        record = rel_by_id.get(relation_id)
        if record is None:
            continue
        subject = label_by_id.get(record["subject_id"])
        object_ = label_by_id.get(record["object_id"])
        if not subject or not object_:
            continue
        relations_out.append({
            "id": record["id"],
            "subject_id": record["subject_id"],
            "predicate": record["predicate"],
            "object_id": record["object_id"],
            "subject": subject,
            "object": object_,
        })
    return relations_out


def _relation_endpoint_labels(
    row: dict,
    conn: sqlite3.Connection,
    agent_id: str,
    relation_rows: list[sqlite3.Row],
    logger: logging.Logger,
    errors: list[str],
) -> dict[str, str]:
    endpoint_ids: set[str] = set()
    for relation in relation_rows:
        endpoint_ids.add(relation["subject_id"])
        endpoint_ids.add(relation["object_id"])
    label_by_id: dict[str, str] = {}
    if not endpoint_ids:
        return label_by_id
    placeholders = ",".join("?" * len(endpoint_ids))
    ids_list = list(endpoint_ids)
    try:
        ent_rows = conn.execute(
            f"SELECT id, label FROM entities WHERE agent_id = ? AND id IN ({placeholders})",
            (agent_id, *ids_list),
        ).fetchall()
        for entity in ent_rows:
            label_by_id[entity["id"]] = entity["label"]
        missing = [id_ for id_ in ids_list if id_ not in label_by_id]
        if missing:
            missing_placeholders = ",".join("?" * len(missing))
            unit_rows = conn.execute(
                f"SELECT id, content FROM knowledge_units "
                f"WHERE agent_id = ? AND id IN ({missing_placeholders})",
                (agent_id, *missing),
            ).fetchall()
            for unit in unit_rows:
                label_by_id[unit["id"]] = unit["content"]
    except Exception:
        errors.append("endpoint_query_failed")
        logger.exception("failed to hydrate relation endpoint labels for briefing %s", row.get("request_id"))
    return label_by_id


def get_briefing_event(
    request_id: str,
    briefing_log: BriefingLog,
    store: SQLiteStore,
    logger: logging.Logger,
) -> Response:
    """Return one hydrated briefing event by request id."""
    row = briefing_log.get_by_request_id(request_id)
    if row is None:
        return JSONResponse({"error": "briefing event not found"}, status_code=404)
    return JSONResponse(hydrate_briefing_row(row, store, logger))


def list_briefing_events(
    request: Request,
    briefing_log: BriefingLog,
    store: SQLiteStore,
    logger: logging.Logger,
) -> Response:
    """Return a paginated list of hydrated briefing events."""
    query_params = request.query_params
    agent_id = query_params.get("agent_id") or None
    source_agent_id = query_params.get("source_agent_id") or None
    try:
        limit = int(query_params.get("limit", "50"))
    except ValueError:
        return JSONResponse({"error": "limit must be an integer"}, status_code=400)
    try:
        offset = int(query_params.get("offset", "0"))
    except ValueError:
        return JSONResponse({"error": "offset must be an integer"}, status_code=400)
    limit = max(1, min(limit, 500))
    offset = max(0, offset)

    rows = briefing_log.list_by_agent(
        agent_id=agent_id,
        source_agent_id=source_agent_id,
        limit=limit,
        offset=offset,
    )
    conn = sqlite3.connect(str(store.db_path))
    conn.row_factory = sqlite3.Row
    try:
        events = [hydrate_briefing_row_with_conn(row, conn, logger) for row in rows]
    finally:
        conn.close()
    return JSONResponse({"events": events, "limit": limit, "offset": offset})
