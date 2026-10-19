"""Memory retrieval and relation rendering for prompt briefings."""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime

import httpx

from tpasini.recordatio.entity import Entity
from tpasini.recordatio.identity import get_principal, visibility
from tpasini.recordatio.models import KnowledgeType, KnowledgeUnit, Relation, RenderedRelation, RetrievalContext
from tpasini.recordatio.proxy import embeddings
from tpasini.recordatio.proxy.candidate_source import _CAP_LOW, KeywordSource, candidate_cap
from tpasini.recordatio.proxy.rrf_fusion import fuse_rrf
from tpasini.recordatio.retrieval_text import content_words
from tpasini.recordatio.storage.base import SQLiteStore

_MAX_WIKIDATA_QUERY_LEN = 200

# G-3 fallback warning is logged once per process, not per query.
_like_fallback_warned = False


def _profile_unit_cap(user_store_scope: str | None, store: SQLiteStore) -> int:
    """Population-relative cap for the profile block's user-unit candidates.

    Measures N_active over the user scope alone (CAP_SCALING_DESIGN.md §4):
    the profile cap and the keyword cap measure different populations and gate
    different blocks, but share the same K/LOW/HIGH constants.

    Legacy pre-migration path (user_store_scope=None): the label-matched
    lookup has no user scope → returns LOW (the safe floor, equiv. to the old
    static 50 for typical small scopes, uncapped for large ones).
    """
    if user_store_scope is None:
        return _CAP_LOW
    return candidate_cap(store, (user_store_scope,))


@dataclass(frozen=True)
class TraversalResult:
    """Entity and relation neighborhood discovered during retrieval."""

    entity_by_id: dict[str, Entity]
    relations: list[Relation]


@dataclass(frozen=True)
class RetrievalResult:
    """Complete retrieval payload used to build a prompt briefing.

    ``keyword_relevance`` carries the keyword source's min–max-normalized BM25
    scores (∈ [0,1]) for the candidates it produced (RETRIEVAL_DESIGN.md §3.3,
    decision D): the budgeter uses it as the relevance term for those ids and
    falls back to lemmatized coverage for everything else (relations always;
    seeded/traversed/profile items).
    """

    units: list
    entities: list[Entity]
    rendered_relations: list[RenderedRelation]
    raw_relations: list[Relation]
    topic_hint: str | None
    keyword_relevance: dict[str, float] = field(default_factory=dict)
    # BM25 pairs in rank order (score-desc) for RRF fusion; the same pool
    # KeywordSource.candidates returned before materialization and normalization.
    # Empty when FTS is unavailable (LIKE-fallback path) or retrieval failed.
    bm25_pairs: list[tuple[str, float]] = field(default_factory=list)
    # The relevance dict the budgeter should consume: BM25-only by default
    # (== keyword_relevance), or RRF(bm25 ⊕ embedding-cosine) when F2 is on
    # (RECORDATIO_F2_EMBEDDINGS). Always populated so prompting reads one field.
    relevance: dict[str, float] = field(default_factory=dict)


def infer_topic_hint(query: str, store: SQLiteStore) -> str | None:
    """Infer a topic filter from query text using known topic names and descriptions.

    Scores every topic by whole-word (lemma) overlap between the query and the
    topic's name + description, using the same content-word pipeline as the
    keyword index (``retrieval_text.content_words``) so matching happens on
    lemmatized tokens, not raw substrings. Returns the name of the single
    best-scoring topic when it clears the threshold (at least one shared
    content word), or ``None`` when nothing does. Ties break by descending
    score then topic name, so the result is order-independent — an early broad
    topic no longer shadows a better later one.
    """
    try:
        topics = store.list_topics()
    except Exception:
        return None
    query_words = content_words(query)
    if not query_words:
        return None
    best_name: str | None = None
    best_score = 0
    for topic in topics:
        topic_words = content_words(f"{topic.name} {topic.description}")
        score = len(query_words & topic_words)
        if score == 0:
            continue
        if score > best_score or (score == best_score and (best_name is None or topic.name < best_name)):
            best_score = score
            best_name = topic.name
    return best_name


def get_entity_by_id(entity_id: str, agent_id: str, store: SQLiteStore) -> Entity | None:
    """Load an entity by stable id for the given agent."""
    try:
        row = store._conn.execute(
            "SELECT id, label, wikidata_id, description, wikipedia_url, asserted_at, "
            "extraction_count, retrieval_count "
            "FROM entities WHERE id = ? AND agent_id = ? LIMIT 1",
            (entity_id, agent_id),
        ).fetchone()
    except Exception:
        return None
    if not row:
        return None
    return Entity(
        id=row["id"],
        label=row["label"],
        wikidata_id=row["wikidata_id"],
        description=row["description"],
        wikipedia_url=row["wikipedia_url"],
        asserted_at=datetime.fromisoformat(row["asserted_at"]),
        extraction_count=row["extraction_count"],
        retrieval_count=row["retrieval_count"],
    )


def get_unit_content(unit_id: str, fallback_units: list, agent_id: str, store: SQLiteStore) -> str | None:
    """Return a knowledge unit's content from retrieved units or persistent storage."""
    for unit in fallback_units:
        if unit.id == unit_id:
            return unit.content
    try:
        row = store._conn.execute(
            "SELECT content FROM knowledge_units WHERE id = ? AND agent_id = ? LIMIT 1",
            (unit_id, agent_id),
        ).fetchone()
    except Exception:
        return None
    return row["content"] if row else None


def _principal_entity(agent_id: str, store: SQLiteStore, user_label: str) -> Entity | None:
    """The user's principal from the user-level store, falling back to the
    legacy label-lookup in the agent scope on pre-migration stores (keeps v1
    artifacts and un-migrated prod behaving exactly as before)."""
    scopes = visibility(agent_id)
    principal = get_principal(store, scopes[-1])
    if principal is not None:
        return principal
    return store.get_entity_by_label(user_label, agent_id)


def seed_entities_local(query: str, agent_id: str, store: SQLiteStore, user_label: str) -> list[Entity]:
    """Find stored entities directly mentioned by the query, across the
    composed visibility ``{S, user_scope(S)}`` (IDENTITY_DESIGN.md §4.3)."""
    seeds: list[Entity] = []
    seen_ids: set[str] = set()
    scopes = visibility(agent_id)
    query_lower = query.lower()
    if any(token in query_lower.split() for token in ("my", "i", "me", "mine", "myself", "i'm", "i've", "i'd")):
        user_entity = _principal_entity(agent_id, store, user_label)
        if user_entity:
            seeds.append(user_entity)
            seen_ids.add(user_entity.id)
    try:
        marks = ",".join("?" for _ in scopes)
        entity_rows = store._conn.execute(
            "SELECT id, label, wikidata_id, description, wikipedia_url, asserted_at, "
            f"extraction_count, retrieval_count FROM entities WHERE agent_id IN ({marks})",  # noqa: S608
            tuple(scopes),
        ).fetchall()
    except Exception:
        entity_rows = []
    for row in entity_rows:
        if row["label"].lower() in query_lower and row["id"] not in seen_ids:
            seeds.append(Entity(
                id=row["id"],
                label=row["label"],
                wikidata_id=row["wikidata_id"],
                description=row["description"],
                wikipedia_url=row["wikipedia_url"],
                asserted_at=datetime.fromisoformat(row["asserted_at"]),
                extraction_count=row["extraction_count"],
                retrieval_count=row["retrieval_count"],
            ))
            seen_ids.add(row["id"])
    # A query naming the user ("Tommaso") seeds the principal through its
    # standing alias set (the gazetteer of names lives in the user scope).
    principal = get_principal(store, scopes[-1])
    if principal is not None and principal.id not in seen_ids:
        try:
            aliases = store.list_entity_aliases(principal.id, scopes[-1])
        except Exception:
            aliases = []
        if any(alias.alias.lower() in query_lower for alias in aliases):
            seeds.append(principal)
            seen_ids.add(principal.id)
    return seeds


def clean_query(query: str) -> str:
    """Strip markup and return the first usable line for external entity lookup."""
    cleaned = re.sub(r"<[a-zA-Z][^>]*>.*?</[a-zA-Z][^>]*>", "", query, flags=re.DOTALL)
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    for line in cleaned.splitlines():
        line = line.strip()
        if line:
            return line[:_MAX_WIKIDATA_QUERY_LEN]
    return ""


async def wikidata_seed_entities(
    query: str,
    exclude_ids: set[str],
    agent_id: str,
    store: SQLiteStore,
    logger: logging.Logger,
) -> list[Entity]:
    """Resolve additional seed entities from Wikidata when they already exist locally."""
    params = {"action": "wbsearchentities", "search": query, "language": "en", "format": "json", "limit": 5}
    seeds: list[Entity] = []
    try:
        headers = {"User-Agent": "recordatio/0.1 (https://github.com/pasinit/recordatio)"}
        async with httpx.AsyncClient(timeout=5.0, headers=headers) as client:
            resp = await client.get("https://www.wikidata.org/w/api.php", params=params)
            resp.raise_for_status()
            data = resp.json()
        for item in data.get("search") or []:
            wikidata_id = item.get("id")
            label = item.get("label", "")
            entity: Entity | None = None
            if wikidata_id:
                entity = store.get_entity_by_wikidata_id(wikidata_id, agent_id)
            if entity is None and label:
                entity = store.get_entity_by_label(label, agent_id)
            if entity is not None and entity.id not in exclude_ids:
                seeds.append(entity)
                exclude_ids.add(entity.id)
    except Exception as exc:
        logger.warning("Wikidata entity seeding failed for %r: %s", query, exc)
    return seeds


def _get_entity_composed(entity_id: str, scopes: tuple[str, ...], store: SQLiteStore) -> Entity | None:
    """Entity lookup across the composed visibility (ids are globally unique,
    so the first scope that holds the row wins)."""
    for scope in scopes:
        entity = get_entity_by_id(entity_id, scope, store)
        if entity is not None:
            return entity
    return None


def traverse(
    seed_entities: list[Entity],
    units: list,
    agent_id: str,
    store: SQLiteStore,
    logger: logging.Logger,
) -> TraversalResult:
    """Traverse relation neighborhoods for seed entities and retrieved units.

    Relations are walked across the composed visibility ``{S, user_scope(S)}``
    (IDENTITY_DESIGN.md §4.3): a user-scope seed walks user-scope relations,
    an agent-scope seed walks agent-scope relations, and cross-scope edges
    resolve through the composed entity/unit lookup.
    """
    scopes = visibility(agent_id)
    entity_by_id: dict[str, Entity] = {}
    relations: list[Relation] = []
    for seed in seed_entities:
        entity_by_id[seed.id] = seed
        try:
            for scope in scopes:
                for relation in store.get_relations(seed.id, scope):
                    relations.append(relation)
                    for entity_id in (relation.subject_id, relation.object_id):
                        if entity_id not in entity_by_id:
                            entity = _get_entity_composed(entity_id, scopes, store)
                            if entity is not None:
                                entity_by_id[entity.id] = entity
        except Exception:
            logger.exception("failed to traverse relations for seed %s", seed.id)
    for unit in units:
        try:
            for scope in scopes:
                relations.extend(store.get_relations(unit.id, scope))
        except Exception:
            logger.exception("failed to load relations for unit %s", unit.id)
    for relation in relations:
        for entity_id in (relation.subject_id, relation.object_id):
            if entity_id not in entity_by_id:
                entity = _get_entity_composed(entity_id, scopes, store)
                if entity is not None:
                    entity_by_id[entity.id] = entity
    return TraversalResult(entity_by_id=entity_by_id, relations=relations)


def render_relations_from(
    relations: list[Relation],
    entity_by_id: dict[str, Entity],
    units: list,
    agent_id: str,
    store: SQLiteStore,
) -> list[RenderedRelation]:
    """Render relation ids into labels suitable for briefing text.

    Unit-content fallbacks resolve across the composed visibility: a
    user-scope relation may legally reference an agent-scope unit and vice
    versa (IDENTITY_DESIGN.md §2.2) — neither may dangle in the render.
    """
    scopes = visibility(agent_id)

    def _unit_content(item_id: str) -> str | None:
        for scope in scopes:
            content = get_unit_content(item_id, units, scope, store)
            if content is not None:
                return content
        return None

    rendered: list[RenderedRelation] = []
    seen: set[tuple[str, str, str]] = set()
    for relation in relations:
        subject_label = (
            entity_by_id[relation.subject_id].label
            if relation.subject_id in entity_by_id
            else _unit_content(relation.subject_id)
        )
        object_label = (
            entity_by_id[relation.object_id].label
            if relation.object_id in entity_by_id
            else _unit_content(relation.object_id)
        )
        if not subject_label or not object_label:
            continue
        key = (subject_label, relation.predicate, object_label)
        if key in seen:
            continue
        seen.add(key)
        rendered.append(RenderedRelation(subject=subject_label, predicate=relation.predicate, object=object_label))
    return rendered


def _materialize_keyword_candidates(
    pairs: list[tuple[str, float]],
    agent_id: str,
    store: SQLiteStore,
    source_agent_id: str | None,
) -> tuple[list[KnowledgeUnit], list[Entity], dict[str, float]]:
    """Resolve keyword-source ids against the canonical tables (§2).

    The index is never trusted for row-level validity: bitemporal and scope
    filters are re-applied here — units active-only with the same rules as
    ``store.search`` (incl. the source-agent filter), entities scope-only (no
    bitemporal columns, matching ``seed_entities_local``). Survivors get
    min–max-normalized relevance over the surviving set (§3.3, decision D);
    the degenerate one-score case is 1.0 (it matched — budget arbitration
    falls to salience/recency/density). Output ordered score-desc, id-asc.
    """
    if not pairs:
        return [], [], {}
    scopes = visibility(agent_id)
    scope_marks = ",".join("?" for _ in scopes)
    score_by_id = dict(pairs)
    placeholders = ",".join("?" for _ in pairs)
    ids = list(score_by_id)
    now_iso = datetime.now().isoformat()
    source_clause = "AND source_agent_id = ?" if source_agent_id is not None else ""
    unit_params = [*scopes, now_iso, *ids] + ([source_agent_id] if source_agent_id is not None else [])
    unit_rows = store._conn.execute(
        f"""
        SELECT id, type, content, asserted_at, valid_from, valid_until, provenance,
               extraction_count, retrieval_count
        FROM knowledge_units
        WHERE agent_id IN ({scope_marks}) AND (valid_until IS NULL OR valid_until > ?)
          AND id IN ({placeholders}) {source_clause}
        """,
        unit_params,
    ).fetchall()
    entity_rows = store._conn.execute(
        f"""
        SELECT id, label, type, subtype, canonical_label, canonical_key,
               wikidata_id, description, wikipedia_url, asserted_at,
               extraction_count, retrieval_count
        FROM entities
        WHERE agent_id IN ({scope_marks}) AND id IN ({placeholders})
        """,
        [*scopes, *ids],
    ).fetchall()
    units = sorted((store._row_to_unit(r) for r in unit_rows), key=lambda u: (-score_by_id[u.id], u.id))
    entities = sorted((store._row_to_entity(r) for r in entity_rows), key=lambda e: (-score_by_id[e.id], e.id))
    surviving = [score_by_id[item.id] for item in (*units, *entities)]
    if not surviving:
        return [], [], {}
    t_min, t_max = min(surviving), max(surviving)
    if t_max == t_min:
        relevance = {item.id: 1.0 for item in (*units, *entities)}
    else:
        relevance = {item.id: (score_by_id[item.id] - t_min) / (t_max - t_min) for item in (*units, *entities)}
    return units, entities, relevance


def _keyword_or_like_units(
    query: str,
    agent_id: str,
    store: SQLiteStore,
    logger: logging.Logger,
    topic_hint: str | None,
    source_agent_id: str | None,
) -> tuple[list[KnowledgeUnit], list[Entity], dict[str, float], list[tuple[str, float]]]:
    """Unit/entity candidates via the keyword source, or today's LIKE path (G-3).

    The keyword source searches the composed visibility ``{S, user_scope(S)}``
    as one candidate pool behind the existing seam (IDENTITY_DESIGN.md §4.2):
    one FTS query, one BM25 pool, min–max normalized over the union.

    Fallback (no FTS5 in this SQLite build): exactly the pre-F1 behavior —
    ``store.search`` with the whole query string — with one warning per
    process; the user scope is searched the same way and deduplicated.

    Returns ``(units, entities, keyword_relevance, bm25_pairs)`` where
    ``bm25_pairs`` carries the raw (item_id, score) list in rank order for
    RRF fusion; empty on the LIKE-fallback path.
    """
    global _like_fallback_warned
    scopes = visibility(agent_id)
    source = KeywordSource(store)
    if not source.available:
        if not _like_fallback_warned:
            logger.warning("SQLite build lacks FTS5; keyword retrieval falling back to LIKE substring search")
            _like_fallback_warned = True
        ctx = RetrievalContext(query=query, topic_hint=topic_hint)
        units = store.search(ctx, agent_id, source_agent_id)
        seen = {u.id for u in units}
        for scope in scopes[1:]:
            units.extend(u for u in store.search(ctx, scope, source_agent_id) if u.id not in seen)
        return units, [], {}, []
    # Population-relative cap: clamp(K × N_active, LOW, HIGH) over the composed
    # scope {S, user_scope(S)} — passes store+scopes for the count (C1/C2).
    cap = candidate_cap(store, scopes)
    pairs = source.candidates(query, scopes, cap)
    units, entities, relevance = _materialize_keyword_candidates(pairs, agent_id, store, source_agent_id)
    return units, entities, relevance, pairs


async def retrieve(
    query: str,
    agent_id: str,
    store: SQLiteStore,
    logger: logging.Logger,
    user_label: str,
    source_agent_id: str | None = None,
) -> RetrievalResult:
    """Retrieve relevant units, entities, relations, and topic context for a query."""
    topic_hint = infer_topic_hint(query, store)
    bm25_pairs: list[tuple[str, float]] = []
    try:
        units, keyword_entities, keyword_relevance, bm25_pairs = _keyword_or_like_units(
            query, agent_id, store, logger, topic_hint, source_agent_id
        )
    except Exception:
        logger.exception("briefing retrieval failed")
        units, keyword_entities, keyword_relevance = [], [], {}
    seeds = seed_entities_local(query, agent_id, store, user_label)
    seen_ids = {entity.id for entity in seeds}
    wikidata_seeds = await wikidata_seed_entities(clean_query(query), seen_ids, agent_id, store, logger)
    seeds.extend(wikidata_seeds)
    # Traversal is unchanged in F1 (§2): keyword-sourced entities are budgeter
    # candidates, not traversal seeds — entity expansion stays the seeding
    # paths' job until F3.
    traversal = traverse(seeds, units, agent_id, store, logger)
    rendered_relations = render_relations_from(
        traversal.relations,
        traversal.entity_by_id,
        units,
        agent_id,
        store,
    )
    entities = list(traversal.entity_by_id.values())
    entities.extend(e for e in keyword_entities if e.id not in traversal.entity_by_id)
    relevance = _fuse_relevance(
        query,
        agent_id,
        store,
        units=units,
        entities=entities,
        raw_relations=traversal.relations,
        entity_by_id=traversal.entity_by_id,
        bm25_pairs=bm25_pairs,
        keyword_relevance=keyword_relevance,
        logger=logger,
    )
    return RetrievalResult(
        units=units,
        entities=entities,
        rendered_relations=rendered_relations,
        raw_relations=traversal.relations,
        topic_hint=topic_hint,
        keyword_relevance=keyword_relevance,
        bm25_pairs=bm25_pairs,
        relevance=relevance,
    )


def _relation_embed_surface(relation: Relation, label_by_id: dict[str, str]) -> str:
    """Relation surface for embedding — ``"subject predicate object"`` with
    endpoint labels resolved from ``label_by_id`` (raw id fallback when an
    endpoint has no label, matching ``store.relation_surfaces_for_embedding``
    and the eval generator's recipe).  ``label_by_id`` is the union of the
    retrieved-entity labels and any store-resolved endpoint labels, so a
    cached relation embedding (canonical labels) and an on-the-fly one agree."""
    subj = label_by_id.get(relation.subject_id, relation.subject_id)
    obj = label_by_id.get(relation.object_id, relation.object_id)
    return embeddings.relation_surface(subj, relation.predicate, obj)


def _relation_label_map(
    raw_relations: list[Relation],
    entity_by_id: dict[str, Entity],
    scopes: tuple[str, ...],
    store: SQLiteStore,
) -> dict[str, str]:
    """Endpoint label map for relation surfaces, resolved consistently with the
    write-time cache: retrieved-entity labels first, then the canonical
    ``entities`` table across scopes for endpoints not in the result.  Matches
    ``store.relation_surfaces_for_embedding`` so a cache hit and an on-the-fly
    embed produce the same surface."""
    label_by_id: dict[str, str] = {eid: e.label for eid, e in entity_by_id.items() if e.label}
    missing = {eid for r in raw_relations for eid in (r.subject_id, r.object_id) if eid not in label_by_id}
    for eid in missing:
        entity = _get_entity_composed(eid, scopes, store)
        if entity is not None and entity.label:
            label_by_id[eid] = entity.label
    return label_by_id


def _fuse_relevance(
    query: str,
    agent_id: str,
    store: SQLiteStore,
    *,
    units: list,
    entities: list[Entity],
    raw_relations: list[Relation],
    entity_by_id: dict[str, Entity],
    bm25_pairs: list[tuple[str, float]],
    keyword_relevance: dict[str, float],
    logger: logging.Logger,
) -> dict[str, float]:
    """Budgeter relevance dict: BM25-only by default, RRF(bm25 ⊕ embedding) when
    F2 is on.  NEVER pure-semantic — RRF is a union, so an item absent from the
    embedding list still contributes its BM25 rank and vice versa.  An item with
    no stored/computable embedding contributes nothing to the embedding list and
    degrades to BM25 for that item (graceful).

    Falls back to ``keyword_relevance`` (the shipped BM25 behaviour) when F2 is
    off, the model is unavailable, or anything errors — so enabling/disabling
    the flag is the only behavioural switch.
    """
    if not embeddings.enabled():
        return keyword_relevance
    try:
        scopes = visibility(agent_id)
        unit_ids = [u.id for u in units]
        entity_ids = [e.id for e in entities]
        label_by_id = _relation_label_map(raw_relations, entity_by_id, scopes, store)
        rel_surfaces = [(r.id, _relation_embed_surface(r, label_by_id)) for r in raw_relations]

        def _load_stored(item_ids: list[str]) -> dict[str, bytes]:
            return store.load_item_embeddings(item_ids, scopes)

        embed_pairs = embeddings.embedding_pairs(
            query,
            unit_ids=unit_ids,
            entity_ids=entity_ids,
            relation_surfaces=rel_surfaces,
            load_stored=_load_stored,
        )
        if not embed_pairs:
            return keyword_relevance
        # Same RRF the eval path uses (proxy/rrf_fusion.fuse_rrf, k=60); BM25
        # rank order from bm25_pairs, embedding rank order from embed_pairs.
        _bm25 = bm25_pairs if bm25_pairs else list(keyword_relevance.items())
        return fuse_rrf(_bm25, dict(embed_pairs))
    except Exception:
        logger.exception("F2: embedding fusion failed; falling back to BM25 relevance")
        return keyword_relevance


def _active_user_units(
    user_label: str,
    agent_id: str,
    store: SQLiteStore,
    logger: logging.Logger,
    *,
    user_store_scope: str | None = None,
) -> list[KnowledgeUnit]:
    """Load active semantic knowledge units for the profile block.

    With a user-level store (``user_store_scope`` set), every active semantic
    unit there is user-anchored by routing (IDENTITY_DESIGN.md §2.2) — no
    label matching needed. The legacy pre-migration path matches on the
    user's label within the agent scope, exactly as before.

    The cap is population-relative (CAP_SCALING_DESIGN.md §4):
    ``clamp(K · N_active, LOW, HIGH)`` measured over the user scope alone so
    a large user store is never truncated at generation time.
    """
    cap = _profile_unit_cap(user_store_scope, store)
    try:
        if user_store_scope is not None:
            rows = store._conn.execute(
                """
                SELECT id, type, content, asserted_at, valid_from, valid_until, provenance,
                       extraction_count, retrieval_count
                FROM knowledge_units
                WHERE agent_id = ?
                  AND type = ?
                  AND valid_until IS NULL
                ORDER BY asserted_at DESC
                LIMIT ?
                """,
                (user_store_scope, KnowledgeType.SEMANTIC.value, cap),
            ).fetchall()
        else:
            rows = store._conn.execute(
                """
                SELECT id, type, content, asserted_at, valid_from, valid_until, provenance,
                       extraction_count, retrieval_count
                FROM knowledge_units
                WHERE agent_id = ?
                  AND type = ?
                  AND valid_until IS NULL
                  AND LOWER(content) LIKE LOWER(?)
                ORDER BY asserted_at DESC
                LIMIT ?
                """,
                (agent_id, KnowledgeType.SEMANTIC.value, f"%{user_label}%", cap),
            ).fetchall()
    except Exception:
        logger.exception("profile unit lookup failed for user %r", user_label)
        return []
    return [store._row_to_unit(row) for row in rows]


async def retrieve_profile(
    agent_id: str,
    store: SQLiteStore,
    user_label: str,
    logger: logging.Logger,
    *,
    now: datetime | None = None,  # noqa: ARG001 — reserved for future recency anchoring
) -> RetrievalResult:
    """Gather the user's query-independent identity neighbourhood as candidates.

    Returns, all active-only (valid_until IS NULL) and deduplicated:
      - the user entity,
      - its active 1-hop relations,
      - the endpoint entities of those relations (so no relation dangles),
      - high-salience semantic knowledge units that mention the user.

    This does NOT rank or trim — that is the budgeter's (W1a) job under B_P. It is
    purely a store lookup: no Wikidata, no query-term matching.

    Profile is THE user-anchored read (IDENTITY_DESIGN.md §4.1): the principal
    resolves by canonical key in the user-level store, its 1-hop neighborhood
    and the user-mentioning semantic units live there by routing. On a
    pre-migration store (no principal) the legacy label-lookup path runs
    unchanged; if neither exists, an empty result is returned gracefully.
    """
    scopes = visibility(agent_id)
    user_store_scope: str | None = scopes[-1]
    user_entity = None
    try:
        user_entity = get_principal(store, scopes[-1])
    except Exception:
        logger.exception("profile principal lookup failed for %s", scopes[-1])
    if user_entity is None:
        # Legacy pre-migration fallback: the per-scope user entity by label.
        user_store_scope = None
        try:
            user_entity = store.get_entity_by_label(user_label, agent_id)
        except Exception:
            logger.exception("profile user-entity lookup failed for %r", user_label)
    if user_entity is None:
        return RetrievalResult(
            units=[],
            entities=[],
            rendered_relations=[],
            raw_relations=[],
            topic_hint=None,
        )

    relation_scope = user_store_scope if user_store_scope is not None else agent_id
    entity_by_id: dict[str, Entity] = {user_entity.id: user_entity}

    raw_relations: list[Relation] = []
    seen_relation_ids: set[str] = set()
    try:
        for relation in store.get_relations(user_entity.id, relation_scope):
            # Active only: get_relations does not filter, so drop superseded facts here.
            if relation.valid_until is not None:
                continue
            if relation.id in seen_relation_ids:
                continue
            seen_relation_ids.add(relation.id)
            raw_relations.append(relation)
            # Pull endpoint entities so relations don't dangle in the render
            # (user-scope rows by §2.2/§2.3 closure; composed lookup for the
            # legal cross-scope edges).
            for endpoint_id in (relation.subject_id, relation.object_id):
                if endpoint_id not in entity_by_id:
                    endpoint = _get_entity_composed(endpoint_id, (relation_scope, *scopes), store)
                    if endpoint is not None:
                        entity_by_id[endpoint.id] = endpoint
    except Exception:
        logger.exception("profile relation traversal failed for user %s", user_entity.id)

    units = _active_user_units(user_label, agent_id, store, logger, user_store_scope=user_store_scope)

    rendered_relations = render_relations_from(raw_relations, entity_by_id, units, agent_id, store)

    return RetrievalResult(
        units=units,
        entities=list(entity_by_id.values()),
        rendered_relations=rendered_relations,
        raw_relations=raw_relations,
        topic_hint=None,
    )
