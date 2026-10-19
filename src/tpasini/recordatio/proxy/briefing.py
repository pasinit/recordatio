"""Compatibility facade for briefing prompt and retrieval helpers."""

from tpasini.recordatio.proxy.prompting import (
    ExtractedSystem,
    SystemBuildResult,
    build_system,
    extract_system,
    last_user_text,
    strip_reminders,
)
from tpasini.recordatio.proxy.retrieval import (
    RetrievalResult,
    TraversalResult,
    clean_query,
    get_entity_by_id,
    get_unit_content,
    infer_topic_hint,
    render_relations_from,
    retrieve,
    seed_entities_local,
    traverse,
    wikidata_seed_entities,
)

__all__ = [
    "ExtractedSystem",
    "RetrievalResult",
    "SystemBuildResult",
    "TraversalResult",
    "build_system",
    "clean_query",
    "extract_system",
    "get_entity_by_id",
    "get_unit_content",
    "infer_topic_hint",
    "last_user_text",
    "render_relations_from",
    "retrieve",
    "seed_entities_local",
    "strip_reminders",
    "traverse",
    "wikidata_seed_entities",
]
