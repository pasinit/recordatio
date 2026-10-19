"""Wikidata resolution service for entities."""

import logging
from datetime import UTC, datetime

import httpx

from tpasini.recordatio.entity import Entity

logger = logging.getLogger(__name__)

_WIKIDATA_API = "https://www.wikidata.org/w/api.php"
_cache: dict[str, Entity] = {}


async def resolve_entity(label: str) -> Entity:
    """Resolve a label to an `Entity`, best-effort via Wikidata.

    Returns an `Entity` with `wikidata_id=None` on any failure or empty result.
    Results are cached by lowercased label.
    """
    key = label.lower()
    if key in _cache:
        return _cache[key]

    entity = Entity(label=label, asserted_at=datetime.now(UTC))
    params = {
        "action": "wbsearchentities",
        "search": label,
        "language": "en",
        "format": "json",
    }
    try:
        headers = {"User-Agent": "recordatio/0.1 (https://github.com/pasinit/recordatio)"}
        async with httpx.AsyncClient(timeout=5.0, headers=headers) as client:
            resp = await client.get(_WIKIDATA_API, params=params)
            resp.raise_for_status()
            data = resp.json()
        results = data.get("search") or []
        if results:
            top = results[0]
            entity.wikidata_id = top.get("id")
            entity.description = top.get("description")
            url = top.get("url")
            if url and url.startswith("//"):
                url = f"https:{url}"
            entity.wikipedia_url = url
    except Exception as e:
        logger.warning("Wikidata resolution failed for %r: %s", label, e)

    _cache[key] = entity
    return entity
