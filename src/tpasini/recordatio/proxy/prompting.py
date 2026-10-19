"""Prompt extraction and memory briefing construction."""

import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from tpasini.recordatio.proxy.briefing_budget import BriefingBlock, BriefingSelection, select_briefing
from tpasini.recordatio.proxy.briefing_log import BriefingLog
from tpasini.recordatio.proxy.retrieval import RetrievalResult

_SYSTEM_REMINDER_RE = re.compile(
    r"<system-reminder>.*?</system-reminder>",
    re.DOTALL | re.IGNORECASE,
)


@dataclass(frozen=True)
class SelectedIds:
    """Row ids of items that ACTUALLY made it into the rendered briefing.

    The deduped union of the profile and recall blocks' SELECTED items, after
    the knapsack and after relation pruning (no-dangling + top-K). Callers use
    this to bump the read-side ``retrieval_count`` for exactly what the model
    saw — never the dropped retrieval candidates or pruned relations.

    ``relations`` are the actual relation ROW ids (``SelectedRelation.relation.id``),
    not entity ids.
    """

    units: list[str]
    entities: list[str]
    relations: list[str]

    @property
    def is_empty(self) -> bool:
        return not (self.units or self.entities or self.relations)


@dataclass(frozen=True)
class SystemBuildResult:
    """System prompt plus optional briefing metadata for the caller.

    ``briefing_text`` is the combined, fully block-tagged two-block briefing
    (``<user_profile>`` + ``<relevant_memory>``) without the caller's system
    prompt — the agent-runner path inserts it VERBATIM into the rendered prompt
    (it owns the tags; the runner adds no wrapper) rather than the system prompt
    (which warm SDK sessions seal at connect time), while the chat path keeps
    using the combined ``system`` (base + briefing_text).

    ``selected_ids`` carries the deduped row ids of the items rendered into the
    two blocks so the caller can schedule a background ``retrieval_count`` bump
    (off the turn's critical path). It is ``None`` only when no briefing was
    produced (both blocks empty).
    """

    system: str | None
    briefing_data: dict | None
    briefing_text: str | None = None
    selected_ids: SelectedIds | None = None


@dataclass(frozen=True)
class ExtractedSystem:
    """System text separated from non-system conversation messages."""

    system: str | None
    messages: list[dict]


def extract_system(messages: list[dict]) -> ExtractedSystem:
    """Separate system messages from the conversation."""
    system_parts: list[str] = []
    rest: list[dict] = []
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                system_parts.extend(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
        else:
            rest.append(msg)
    return ExtractedSystem(system="\n\n".join(system_parts) if system_parts else None, messages=rest)


def strip_reminders(text: str) -> str:
    """Remove transient system-reminder blocks before retrieval or extraction."""
    return _SYSTEM_REMINDER_RE.sub("", text).strip()


def last_user_text(messages: list[dict]) -> str | None:
    """Return the latest user text with inline system-reminder blocks removed."""
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return strip_reminders(content) or None
        if isinstance(content, list):
            parts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            if parts:
                return strip_reminders("\n".join(parts)) or None
        return None
    return None


# The model-facing note that must survive into the recall block: it tells the
# agent the memory was pre-fetched so it doesn't redundantly call memory_search.
_RECALL_NOTE = (
    "Note: memory has been pre-fetched for the current query. "
    "Only call memory_search for substantively different topics not covered above."
)


def _render_profile_block(block: BriefingBlock) -> str | None:
    """Render the selected profile items as a ``<user_profile>`` block.

    Returns ``None`` when the block is empty (so the tag is omitted entirely
    rather than emitted empty).
    """
    sections = _render_item_sections(block, topic_names=[])
    if not sections:
        return None
    body = "\n\n".join(sections)
    return f"<user_profile>\n{body}\n</user_profile>"


def _render_recall_block(block: BriefingBlock, topic_names: list[str]) -> str | None:
    """Render the selected query-recall items as a ``<relevant_memory>`` block.

    Keeps the legacy ``[MEMORY]`` note/Topics framing inside so the model-facing
    "only call memory_search for substantively different topics" guidance
    survives. Returns ``None`` when the block is empty.
    """
    sections = _render_item_sections(block, topic_names=topic_names)
    if not sections:
        return None
    body = "\n\n".join(sections)
    return f"<relevant_memory>\n[MEMORY]\n{_RECALL_NOTE}\n\n{body}\n[/MEMORY]\n</relevant_memory>"


def _render_item_sections(block: BriefingBlock, *, topic_names: list[str]) -> list[str]:
    """Render Topics/Entities/Facts/Relations sections from selected items.

    Faithful to ``Briefing.format`` per-item line shapes (the budgeter's char
    estimator mirrors these), but driven off the budgeted ``BriefingBlock``
    rather than the full retrieval graph.
    """
    sections: list[str] = []

    if topic_names:
        sections.append("Topics: " + ", ".join(topic_names))

    if block.entities:
        entity_lines = ["Entities:"]
        for sel in block.entities:
            e = sel.entity
            qid = f" ({e.wikidata_id})" if e.wikidata_id else ""
            desc = f": {e.description}" if e.description else ""
            entity_lines.append(f"- {e.label}{qid}{desc}")
        sections.append("\n".join(entity_lines))

    if block.units:
        fact_lines = ["Facts:"]
        for sel in block.units:
            u = sel.unit
            fact_lines.append(f"- ({u.type.value}) {u.content}")
        sections.append("\n".join(fact_lines))

    if block.relations:
        rel_lines = ["Relations:"]
        for sel in block.relations:
            r = sel.rendered
            rel_lines.append(f"- {r.subject} -[{r.predicate}]-> {r.object}")
        sections.append("\n".join(rel_lines))

    return sections


def _briefing_data_from_selection(
    selection: BriefingSelection,
    topic_names: list[str],
) -> dict:
    """Build the telos recalled-memories panel payload from SELECTED items.

    The panel must reflect what the agent actually saw (the budgeted union of
    the profile and query blocks), not the full pre-budget retrieval. Deduped
    by content/label/triple so an item chosen in both blocks appears once.
    """
    units: list[dict] = []
    seen_units: set[tuple[str, str]] = set()
    entities: list[dict] = []
    seen_entities: set[str] = set()
    relations: list[dict] = []
    seen_relations: set[tuple[str, str, str]] = set()

    for block in (selection.profile_items, selection.query_items):
        for sel in block.units:
            u = sel.unit
            key = (u.type.value, u.content)
            if key in seen_units:
                continue
            seen_units.add(key)
            units.append({"type": u.type.value, "content": u.content})
        for sel in block.entities:
            e = sel.entity
            if e.id in seen_entities:
                continue
            seen_entities.add(e.id)
            entities.append(
                {"label": e.label, "wikidata_id": e.wikidata_id, "description": e.description}
            )
        for sel in block.relations:
            r = sel.rendered
            key = (r.subject, r.predicate, r.object)
            if key in seen_relations:
                continue
            seen_relations.add(key)
            relations.append({"subject": r.subject, "predicate": r.predicate, "object": r.object})

    return {
        "units": units,
        "entities": entities,
        "relations": relations,
        "topic_names": topic_names,
    }


async def build_system(
    *,
    existing_system: str | None,
    query: str,
    agent_id: str,
    backend: str,
    call_id: str | None,
    source_agent_id: str | None = None,
    retrieve: Callable[
        [str, str, str | None],
        Awaitable[RetrievalResult],
    ],
    retrieve_profile: Callable[..., Awaitable[RetrievalResult]],
    briefing_log: BriefingLog,
    logger: logging.Logger,
) -> SystemBuildResult:
    """Retrieve memories, budget them, and build a two-block briefing.

    Runs the per-query retrieval and the query-independent profile retrieval,
    feeds both candidate sets through the W1a budgeter, and renders two
    prompt-body blocks: ``<user_profile>`` (identity facts) and
    ``<relevant_memory>`` (query recall). ``briefing_text`` is the combined,
    fully-tagged two-block string (the runner inserts it verbatim — it no longer
    wraps it). ``system`` keeps the legacy ``base + briefing`` composition for
    the chat / ApiKeyTransport path. ``briefing_data`` is the panel payload built
    from the SELECTED (budgeted) union, not the full retrieval graph.
    """
    # Single clock read for the whole build — the budgeter is pure and forbids
    # reading the clock itself, so ``now`` is injected from here.
    now = datetime.now(UTC)

    retrieval = await retrieve(query, agent_id, source_agent_id)
    profile = await retrieve_profile(agent_id, now=now)

    topic_names = [retrieval.topic_hint] if retrieval.topic_hint else []

    selection = select_briefing(
        query=query,
        now=now,
        query_units=retrieval.units,
        query_entities=retrieval.entities,
        query_raw_relations=retrieval.raw_relations,
        query_rendered_relations=retrieval.rendered_relations,
        profile_units=profile.units,
        profile_entities=profile.entities,
        profile_raw_relations=profile.raw_relations,
        profile_rendered_relations=profile.rendered_relations,
        # F2: ``retrieval.relevance`` is BM25-only by default and
        # RRF(bm25 ⊕ embedding) when RECORDATIO_F2_EMBEDDINGS is on — same shipped
        # budgeter, only the relevance dict changes; never pure-semantic.
        relevance_by_id=retrieval.relevance or retrieval.keyword_relevance,
    )

    profile_text = _render_profile_block(selection.profile_items)
    recall_text = _render_recall_block(selection.query_items, topic_names)

    blocks = [b for b in (profile_text, recall_text) if b]
    if not blocks:
        # Both empty → no briefing at all (same as today's no-memory case).
        return SystemBuildResult(system=existing_system, briefing_data=None)

    briefing_text = "\n\n".join(blocks)
    system = f"{existing_system}\n\n{briefing_text}" if existing_system else briefing_text

    p, q = selection.profile_items, selection.query_items
    # Counts are reported as selected/candidate so budget/ranking misses are
    # measurable from the logs: when selected ≪ candidate the budgeter dropped
    # recall the agent could have seen for this turn.
    logger.info(
        "injected briefing (profile: units=%d/%d entities=%d/%d relations=%d/%d chars=%d | "
        "recall: units=%d/%d entities=%d/%d relations=%d/%d chars=%d | "
        "total_chars=%d topic_hint=%s backend=%s) [counts are selected/candidate]",
        len(p.units), len(profile.units), len(p.entities), len(profile.entities),
        len(p.relations), len(profile.raw_relations), p.chars,
        len(q.units), len(retrieval.units), len(q.entities), len(retrieval.entities),
        len(q.relations), len(retrieval.raw_relations), q.chars,
        len(briefing_text), retrieval.topic_hint, backend,
    )

    if call_id:
        try:
            briefing_log.record_briefing(
                request_id=call_id,
                agent_id=agent_id,
                source_agent_id=source_agent_id,
                query_text=query,
                topic_hint=retrieval.topic_hint,
                recalled_unit_ids=_selected_unit_ids(selection),
                recalled_entity_ids=_selected_entity_ids(selection),
                recalled_relation_ids=_selected_relation_ids(selection),
            )
        except Exception:
            logger.exception("failed to persist briefing event for %s", call_id)

    briefing_data = _briefing_data_from_selection(selection, topic_names)
    return SystemBuildResult(
        system=system,
        briefing_data=briefing_data,
        briefing_text=briefing_text,
        selected_ids=_selected_ids(selection),
    )


def _selected_ids(selection: BriefingSelection) -> SelectedIds:
    """Deduped row ids of the SELECTED (rendered) items, partitioned by type.

    Drawn from the union of the profile and recall blocks AFTER the knapsack
    and relation pruning — exactly the items rendered into ``<user_profile>`` /
    ``<relevant_memory>``. Relation ids are the raw relation ROW ids.
    """
    return SelectedIds(
        units=_selected_unit_ids(selection),
        entities=_selected_entity_ids(selection),
        relations=_selected_relation_ids(selection),
    )


def _selected_unit_ids(selection: BriefingSelection) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for block in (selection.profile_items, selection.query_items):
        for sel in block.units:
            if sel.unit.id not in seen:
                seen.add(sel.unit.id)
                ids.append(sel.unit.id)
    return ids


def _selected_entity_ids(selection: BriefingSelection) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for block in (selection.profile_items, selection.query_items):
        for sel in block.entities:
            if sel.entity.id not in seen:
                seen.add(sel.entity.id)
                ids.append(sel.entity.id)
    return ids


def _selected_relation_ids(selection: BriefingSelection) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for block in (selection.profile_items, selection.query_items):
        for sel in block.relations:
            if sel.relation.id not in seen:
                seen.add(sel.relation.id)
                ids.append(sel.relation.id)
    return ids
