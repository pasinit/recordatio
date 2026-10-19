"""Bounded briefing selection — length-dampened budgeter with relevance floor (W1a).

Pure, no-I/O module implementing the structural item-value model from
``BRIEFING_DESIGN.md``: a value function ``v(item | q)``, a two-pass knapsack
with a hard char budget and graceful drop, and two-stage relation pruning
(no-dangling + top-K cap).

The knapsack objective is updated per RERANKING_DESIGN.md §2 (ruling 4):

1. **Length-dampened key** ``v / chars^λ`` (λ=0.5, pre-registered) replaces
   the old value-per-char density ``v / chars``.  λ=0.5 (square-root
   normalisation, standard IR "pivoted length normalisation") halves the density
   advantage short fragments enjoy, preventing the swarm-vs-fact pathology
   (q-0008-class). λ=1 reproduces the old behaviour; λ=0 is pure value.

2. **Relevance-floor reserve pass** (φ=0.30, ρ=0.6, pre-registered): before
   the general length-dampened fill, a fraction φ of each block's budget is
   reserved for items whose reranked context-relevance ≥ ρ; those items are
   admitted value-first (not density-first) — a genuine answer fact beats the
   swarm regardless of length. The remaining 1−φ fills by length-dampened key.

Both constants are pre-registered from first principles (RERANKING_DESIGN.md §7):
λ=0.5 = standard pivoted normalisation; φ≤0.30 caps reserve-pass damage;
ρ=0.6 = "more confident than midpoint". Governed by F8 — never tuned on frozen
annotations.

Given a ``RetrievalResult``-shaped candidate set (units, entities, raw +
rendered relations), a query, a separate profile-candidate set, and char
budgets, it produces a :class:`BriefingSelection` with a profile block and a
query-recall block. ``now`` is injected for testability — this module never
reads the clock.
"""

import math
import os
from dataclasses import dataclass, field
from datetime import datetime

from tpasini.recordatio.entity import Entity
from tpasini.recordatio.models import KnowledgeType, Relation, RenderedRelation
from tpasini.recordatio.retrieval_text import content_words as _content_words


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# --- Tunable constants (env-overridable; the eval/improver loop's knobs) ---

# Type priors: curated units beat entities beat raw triples.
W_UNIT = _env_float("RECORDATIO_BRIEFING_W_UNIT", 1.0)
W_ENTITY = _env_float("RECORDATIO_BRIEFING_W_ENTITY", 0.6)
W_RELATION = _env_float("RECORDATIO_BRIEFING_W_RELATION", 0.4)

# Value-component coefficients.
ALPHA = _env_float("RECORDATIO_BRIEFING_ALPHA", 0.5)  # relevance weight
BETA = _env_float("RECORDATIO_BRIEFING_BETA", 0.3)  # salience weight
GAMMA = _env_float("RECORDATIO_BRIEFING_GAMMA", 0.2)  # recency weight

# Recency decay half-life, in days. The default (stable SEMANTIC/PROCEDURAL
# profile facts) is deliberately slow: a fact about who the user is does not go
# stale in weeks.
HALF_LIFE_DAYS = _env_float("RECORDATIO_BRIEFING_HALFLIFE_DAYS", 90.0)

# Type-conditioned half-life for volatile EPISODIC units (status snapshots:
# property listings, application-status records). A uniform 90-day half-life is
# far too slow for these: a fresh same-day record scores 0.5**(1/90)=0.992 while
# a 90-day-old snapshot still scores 0.5 — a <0.1 gap that, after the GAMMA
# recency weight, is easily overturned by the BETA salience term accruing on the
# older, oft-rewritten unit, so a stale snapshot outranks a same-day
# authoritative record (the recurring recall-miss pathology). The un-tombstoned
# snapshots that trigger this never get a ``valid_until``, so ``_validity_factor``
# never fires for them. A much shorter half-life makes EPISODIC recency decay on
# the ~fortnightly cadence these statuses actually change at: a 90-day-old
# snapshot decays to 0.5**(90/14)≈0.012, restoring fresh-record-wins ordering.
# This is a NEW type-conditioned constant, NOT a re-tune of the shared knob
# (F8: never tuned on frozen annotations); env-overridable.
EPISODIC_HALF_LIFE_DAYS = _env_float("RECORDATIO_BRIEFING_EPISODIC_HALFLIFE_DAYS", 14.0)

# Down-weight for items whose validity window has closed (``valid_until`` in the
# past). The value function otherwise never reads ``valid_until``, so an
# explicitly-expired fact ("no longer available as of …") keeps its full
# salience/recency score and can outrank the fresher unit that superseded it —
# the stale-status-vs-resolution pathology. A closed window multiplies the
# item's value by this factor: a heavy down-weight rather than a hard drop, so
# an expired fact still surfaces when it is the only evidence. Env-overridable.
EXPIRED_VALIDITY_FACTOR = _env_float("RECORDATIO_BRIEFING_EXPIRED_VALIDITY_FACTOR", 0.1)

# Salience-blend weights (C4). Salience is a weighted blend of three signals,
# each normalised to [0, 1] against the per-candidate-set maximum:
#
#   salience = w_ext · norm(log1p(extraction_count − 1))
#            + w_ret · norm(log1p(retrieval_count))
#            + w_deg · norm(degree)
#
# Feedback-loop rationale for the weighting: extraction_count is *externally
# driven* — it only grows when the user re-asserts a fact, so it is an honest
# reinforcement signal. retrieval_count is *self-reinforcing* — an item that
# gets selected into a briefing is counted, which raises its salience, which
# makes it more likely to be selected again (selected → counted → more salient
# → selected). To keep that loop from locking in early winners, the retrieval
# signal is BOTH log-dampened and weighted strictly below extraction.
W_EXTRACTION = _env_float("RECORDATIO_BRIEFING_W_EXTRACTION", 0.5)
W_RETRIEVAL = _env_float("RECORDATIO_BRIEFING_W_RETRIEVAL", 0.2)
W_DEGREE = _env_float("RECORDATIO_BRIEFING_W_DEGREE", 0.3)

# Char budgets (≈4 chars/token), matching the _HISTORY_CHAR_BUDGET convention.
PROFILE_CHAR_BUDGET = _env_int("RECORDATIO_PROFILE_CHAR_BUDGET", 3200)
RECALL_CHAR_BUDGET = _env_int("RECORDATIO_RECALL_CHAR_BUDGET", 6800)

# Relation cap (after the no-dangling filter).
MAX_RELATIONS = _env_int("RECORDATIO_BRIEFING_MAX_RELATIONS", 20)

# Floor salience for units whose counters are still at baseline
# (extraction_count == 1, retrieval_count == 0). Units have no degree signal,
# so a never-reinforced, never-retrieved unit would otherwise score 0; the
# floor keeps it competitive on relevance + recency. Units with real counter
# evidence get real salience from the blend above (which can exceed the floor).
UNIT_SALIENCE = _env_float("RECORDATIO_BRIEFING_UNIT_SALIENCE", 0.1)

# Minimum char estimate guards against div-by-zero in the length-dampened key.
_MIN_CHARS = 1

# --- Objective-change constants (RERANKING_DESIGN.md §2, pre-registered F8-strict) ---

# Length-dampening exponent λ: key = v / chars^λ.
# λ=0.5 is standard "pivoted length normalisation" (square-root dampening);
# λ=1 reproduces old value-per-char; λ=0 is pure value. Pre-registered; env-overridable
# for ablation studies only (env name from the spec's knob table).
LENGTH_LAMBDA = _env_float("RECORDATIO_BRIEFING_LENGTH_LAMBDA", 0.5)

# Relevance-floor reserve pass:
#   φ = fraction of block budget reserved for high-relevance items.
#   ρ = minimum reranked context-relevance to qualify for the reserve.
# Pre-registered from first principles (φ≤0.30 caps reserve-pass damage;
# ρ=0.6 = "more confident than midpoint"). Env names from the spec's knob table.
RELEVANCE_FLOOR_FRACTION = _env_float("RECORDATIO_BRIEFING_RELEVANCE_FLOOR_FRACTION", 0.30)
RELEVANCE_FLOOR_THRESHOLD = _env_float("RECORDATIO_BRIEFING_RELEVANCE_FLOOR_THRESHOLD", 0.60)

# Content words now come from the shared §3.1 pipeline (retrieval_text):
# lemmatized + number-normalized, same frozen stopword list — so the coverage
# relevance below and the keyword source's loose terms share one tokenization
# (RETRIEVAL_DESIGN.md decision D, amended).


# --- Selected-item wrappers (carry enough to render and to build the panel) ---


@dataclass(frozen=True)
class SelectedUnit:
    unit: object  # KnowledgeUnit
    value: float


@dataclass(frozen=True)
class SelectedEntity:
    entity: Entity
    value: float


@dataclass(frozen=True)
class SelectedRelation:
    relation: Relation
    rendered: RenderedRelation
    value: float


@dataclass(frozen=True)
class BriefingBlock:
    """One rendered block's worth of chosen items (profile or query)."""

    units: list[SelectedUnit] = field(default_factory=list)
    entities: list[SelectedEntity] = field(default_factory=list)
    relations: list[SelectedRelation] = field(default_factory=list)

    @property
    def chars(self) -> int:
        total = 0
        for s in self.units:
            total += char_unit(s.unit)
        for s in self.entities:
            total += char_entity(s.entity)
        for s in self.relations:
            total += char_relation(s.rendered)
        return total


@dataclass(frozen=True)
class RankedCandidate:
    """One scored candidate in a block's value-ordered pre-budget ranking.

    Exposed for evaluation (the benchmark's mean-rank / MRR metrics — see
    BENCHMARK_DESIGN.md §3): the rank of an item *before* the budget cut is a
    finer-grained ranking-quality signal than binary inclusion. Selection
    behaviour is unaffected by this bookkeeping.
    """

    item_id: str
    kind: str  # "unit" | "entity" | "relation"
    value: float
    chars: int


@dataclass(frozen=True)
class BriefingSelection:
    """Two-block selection: query-independent profile + query recall.

    ``profile_items`` and ``query_items`` are disjoint by item id (query recall
    excludes anything already chosen for the profile). The union of both blocks
    is the panel payload (telos's recalled-memories panel).

    ``profile_ranking`` / ``query_ranking`` are each block's full candidate
    list ordered by descending value — the PRE-BUDGET ranking the knapsack
    drew from (relations are ranked even when the no-dangling rule or the
    top-K cap later dropped them, so an unselected needle still has a rank).
    Purely observational: the selection itself is unchanged by their presence.
    """

    profile_items: BriefingBlock
    query_items: BriefingBlock
    profile_ranking: tuple[RankedCandidate, ...] = ()
    query_ranking: tuple[RankedCandidate, ...] = ()


# --- Char estimation (faithful to models.py Briefing.format per-item lines) ---


def char_unit(unit: object) -> int:
    """Estimate the rendered length of a unit line: ``- ({type}) {content}``."""
    type_val = getattr(getattr(unit, "type", ""), "value", getattr(unit, "type", ""))
    content = getattr(unit, "content", "") or ""
    return max(_MIN_CHARS, len(f"- ({type_val}) {content}"))


def char_entity(entity: Entity) -> int:
    """Estimate an entity line: ``- {label} ({wikidata}): {desc}``."""
    qid = f" ({entity.wikidata_id})" if entity.wikidata_id else ""
    desc = f": {entity.description}" if entity.description else ""
    return max(_MIN_CHARS, len(f"- {entity.label}{qid}{desc}"))


def char_relation(rendered: RenderedRelation) -> int:
    """Estimate a relation line: ``- {subj} -[{pred}]-> {obj}``."""
    return max(
        _MIN_CHARS,
        len(f"- {rendered.subject} -[{rendered.predicate}]-> {rendered.object}"),
    )


# --- Value components ---


def _recency(
    asserted_at: datetime | None, now: datetime, half_life_days: float = HALF_LIFE_DAYS
) -> float:
    """Exponential decay: ``0.5 ** (age_days / half_life_days)`` ∈ [0, 1].

    ``half_life_days`` defaults to the global :data:`HALF_LIFE_DAYS` (stable
    profile facts, and entities/relations which carry no KnowledgeType). Unit
    scoring overrides it per type via :func:`_unit_half_life` so volatile
    EPISODIC status snapshots decay far faster than stable SEMANTIC facts.
    """
    if asserted_at is None:
        return 0.0
    age_days = max(0.0, (now - asserted_at).total_seconds() / 86_400.0)
    return 0.5 ** (age_days / half_life_days)


def _unit_half_life(unit: object) -> float:
    """Recency half-life for a knowledge unit, conditioned on its KnowledgeType.

    EPISODIC units are volatile status snapshots that a later run supersedes
    within days; they decay on :data:`EPISODIC_HALF_LIFE_DAYS`. SEMANTIC and
    PROCEDURAL facts are stable and keep the global :data:`HALF_LIFE_DAYS`.
    ``getattr`` keeps it safe for unit-shaped objects that omit ``type``.
    """
    if getattr(unit, "type", None) == KnowledgeType.EPISODIC:
        return EPISODIC_HALF_LIFE_DAYS
    return HALF_LIFE_DAYS


def _validity_factor(item: object, now: datetime) -> float:
    """Multiplicative penalty for items whose validity window has closed.

    Returns ``EXPIRED_VALIDITY_FACTOR`` when ``valid_until`` is set and not in
    the future (the fact is explicitly expired), else ``1.0``. ``getattr`` keeps
    it safe for candidate types that carry no ``valid_until`` (e.g. entities),
    which are always treated as unexpired."""
    valid_until = getattr(item, "valid_until", None)
    if valid_until is not None and valid_until <= now:
        return EXPIRED_VALIDITY_FACTOR
    return 1.0


def _relevance(text: str, query_words: set[str]) -> float:
    """Fraction of query content-words present in ``text`` ∈ [0, 1]."""
    if not query_words:
        return 0.0
    item_words = _content_words(text)
    if not item_words:
        return 0.0
    hits = sum(1 for w in query_words if w in item_words)
    return hits / len(query_words)


def _entity_degrees(raw_relations: list[Relation]) -> dict[str, int]:
    """Active-relation participation count per entity id (degree centrality)."""
    degree: dict[str, int] = {}
    for rel in raw_relations:
        for eid in (rel.subject_id, rel.object_id):
            degree[eid] = degree.get(eid, 0) + 1
    return degree


# --- Salience (reinforcement counters + degree centrality) -----------------


def _log_extraction(item: object) -> float:
    """Dampened write-side reinforcement. Baseline extraction_count is 1
    (every fact starts asserted once), so a never-re-asserted fact contributes
    exactly 0 — only *re*-assertions count as reinforcement."""
    return math.log1p(max(getattr(item, "extraction_count", 1) - 1, 0))


def _log_retrieval(item: object) -> float:
    """Dampened read-side usage. log1p tames the self-reinforcing
    selected→counted→more-salient feedback loop (see W_RETRIEVAL above)."""
    return math.log1p(max(getattr(item, "retrieval_count", 0), 0))


@dataclass(frozen=True)
class _SalienceNorms:
    """Per-candidate-set maxima used to normalise each salience signal to [0, 1]."""

    max_log_extraction: float
    max_log_retrieval: float
    max_degree: int


def _salience_norms(
    units: list, entities: list[Entity], raw_relations: list[Relation], degrees: dict[str, int]
) -> _SalienceNorms:
    items = [*units, *entities, *raw_relations]
    return _SalienceNorms(
        max_log_extraction=max((_log_extraction(i) for i in items), default=0.0),
        max_log_retrieval=max((_log_retrieval(i) for i in items), default=0.0),
        max_degree=max(degrees.values(), default=0),
    )


def _blend_salience(
    item: object, degree_norm: float | None, norms: _SalienceNorms
) -> float:
    """Weighted blend of the three salience signals, normalised per candidate set.

    A signal with no evidence anywhere in the candidate set (max 0 — e.g. a
    store predating the counter columns, or zeroed weights) is dropped and the
    remaining weights are renormalised, instead of silently shrinking the
    salience scale relative to relevance/recency. This also means a counter-less
    candidate set degrades exactly to the pre-C4 behaviour (pure degree
    centrality). ``degree_norm`` is ``None`` for units, which have no degree.
    """
    parts: list[tuple[float, float]] = []  # (weight, normalised signal)
    if W_EXTRACTION > 0 and norms.max_log_extraction > 0:
        parts.append((W_EXTRACTION, _log_extraction(item) / norms.max_log_extraction))
    if W_RETRIEVAL > 0 and norms.max_log_retrieval > 0:
        parts.append((W_RETRIEVAL, _log_retrieval(item) / norms.max_log_retrieval))
    if W_DEGREE > 0 and norms.max_degree > 0 and degree_norm is not None:
        parts.append((W_DEGREE, degree_norm))
    total_weight = sum(w for w, _ in parts)
    if total_weight <= 0:
        return 0.0
    return sum(w * s for w, s in parts) / total_weight


def _unit_salience(unit: object, norms: _SalienceNorms) -> float:
    """Units get real salience from their own counters; the UNIT_SALIENCE floor
    remains only for units with no reinforcement evidence (both counters at
    baseline the blend is 0, so ``max`` keeps the floor)."""
    return max(_blend_salience(unit, None, norms), UNIT_SALIENCE)


# --- Candidate scoring ---


@dataclass(frozen=True)
class _Candidate:
    """An item scored and ready for the knapsack."""

    item_id: str
    value: float
    chars: int
    kind: str  # "unit" | "entity" | "relation"
    payload: object  # the selected-item wrapper

    @property
    def density(self) -> float:
        """Legacy value-per-char (λ=1).  Used only for reference / λ=1 ablation."""
        return self.value / self.chars

    def dampened_key(self, lam: float) -> float:
        """Length-dampened ordering key: v / chars^λ (RERANKING_DESIGN.md §2.2).

        λ=0.5 (default): square-root dampening, halves short-fragment density advantage.
        λ=1: pure value-per-char (old behaviour). λ=0: pure value.
        Tie-break on item_id is handled at sort site.
        """
        return self.value / (max(self.chars, _MIN_CHARS) ** lam)


def _unit_text(unit: object) -> str:
    return getattr(unit, "content", "") or ""


def _entity_text(entity: Entity) -> str:
    label = entity.label or ""
    desc = entity.description or ""
    return f"{label} {desc}".strip()


def _relation_text(rendered: RenderedRelation) -> str:
    return f"{rendered.subject} {rendered.predicate} {rendered.object}"


def _keyword_or_coverage_relevance(
    item_id: str, text: str, query_words: set[str], relevance_by_id: dict[str, float]
) -> float:
    """Source-aware relevance (decision D, amended): a candidate carrying a
    keyword-source score uses its normalized BM25; everything else (relations
    always; seeded/traversed/profile items) keeps the lemmatized-coverage
    measure. Both flavours share the [0,1] scale and the §3.1 tokenization."""
    keyword = relevance_by_id.get(item_id)
    if keyword is not None:
        return keyword
    return _relevance(text, query_words)


def _score_unit(
    unit: object, query_words: set[str], now: datetime, norms: _SalienceNorms, alpha: float,
    relevance_by_id: dict[str, float],
) -> float:
    relevance = (
        _keyword_or_coverage_relevance(unit.id, _unit_text(unit), query_words, relevance_by_id) if alpha else 0.0
    )
    salience = _unit_salience(unit, norms)
    recency = _recency(getattr(unit, "asserted_at", None), now, _unit_half_life(unit))
    return W_UNIT * (alpha * relevance + BETA * salience + GAMMA * recency) * _validity_factor(unit, now)


def _score_entity(
    entity: Entity,
    query_words: set[str],
    now: datetime,
    degrees: dict[str, int],
    norms: _SalienceNorms,
    alpha: float,
    relevance_by_id: dict[str, float],
) -> float:
    relevance = (
        _keyword_or_coverage_relevance(entity.id, _entity_text(entity), query_words, relevance_by_id)
        if alpha
        else 0.0
    )
    degree_norm = degrees.get(entity.id, 0) / norms.max_degree if norms.max_degree > 0 else 0.0
    salience = _blend_salience(entity, degree_norm, norms)
    recency = _recency(getattr(entity, "asserted_at", None), now)
    return W_ENTITY * (alpha * relevance + BETA * salience + GAMMA * recency)


def _score_relation(
    relation: Relation,
    rendered: RenderedRelation,
    query_words: set[str],
    now: datetime,
    degrees: dict[str, int],
    norms: _SalienceNorms,
    alpha: float,
    relevance_by_id: dict[str, float],
) -> float:
    # Relations never carry a keyword-source score; the shared helper keeps the
    # coverage fallback uniform anyway (absence from the dict = coverage).
    relevance = (
        _keyword_or_coverage_relevance(relation.id, _relation_text(rendered), query_words, relevance_by_id)
        if alpha
        else 0.0
    )
    # Counters are per-item (the relation's own extraction/retrieval counts);
    # only the degree component is inherited from the max endpoint, since a
    # relation's centrality is its endpoints'.
    if norms.max_degree > 0:
        degree_norm = (
            max(
                degrees.get(relation.subject_id, 0),
                degrees.get(relation.object_id, 0),
            )
            / norms.max_degree
        )
    else:
        degree_norm = 0.0
    salience = _blend_salience(relation, degree_norm, norms)
    recency = _recency(getattr(relation, "asserted_at", None), now)
    return W_RELATION * (alpha * relevance + BETA * salience + GAMMA * recency) * _validity_factor(relation, now)


# --- Two-pass knapsack (RERANKING_DESIGN.md §2.2) ---


def _knapsack(
    candidates: list[_Candidate],
    budget: int,
    relevance_by_id: dict[str, float] | None = None,
) -> list[_Candidate]:
    """Two-pass greedy fill: relevance-floor reserve then length-dampened general.

    Pass 1 — relevance-floor reserve (§2.2 point 2):
        Reserve φ=RELEVANCE_FLOOR_FRACTION of the budget for items whose
        reranked context-relevance ≥ ρ=RELEVANCE_FLOOR_THRESHOLD.  Admitted
        value-first (NOT density-first) so a genuine answer fact beats the
        swarm regardless of length.  Items that don't fit (char cost > reserve
        budget) are deferred to the general pass, not discarded.

    Pass 2 — length-dampened general fill (§2.2 point 1):
        Remaining budget filled by descending ``dampened_key(λ)`` with graceful
        skip (an item that doesn't fit is skipped, not a stopping condition).
        Items chosen in pass 1 are excluded via their ids (no double-spend).

    With ``relevance_by_id=None`` (profile block, α=0), pass 1 is skipped:
    no relevance signal → no reserve allocation.  The profile path stays
    byte-identical to the pre-reranker behaviour (only λ changes the key).
    """
    if not candidates:
        return []

    lam = LENGTH_LAMBDA
    chosen: list[_Candidate] = []
    chosen_ids: set[str] = set()
    used = 0

    # Pass 1: relevance-floor reserve.
    if relevance_by_id:
        reserve_budget = int(budget * RELEVANCE_FLOOR_FRACTION)
        floor_candidates = [
            c for c in candidates
            if relevance_by_id.get(c.item_id, 0.0) >= RELEVANCE_FLOOR_THRESHOLD
        ]
        # Value-first within the reserve (not density-first — spec §2.2).
        floor_candidates.sort(key=lambda c: (-c.value, c.item_id))
        reserve_used = 0
        for cand in floor_candidates:
            if reserve_used + cand.chars <= reserve_budget:
                chosen.append(cand)
                chosen_ids.add(cand.item_id)
                reserve_used += cand.chars
                used += cand.chars

    # Pass 2: length-dampened general fill on remaining budget.
    remaining = budget - used
    general = [c for c in candidates if c.item_id not in chosen_ids]
    for cand in sorted(
        general,
        key=lambda c: (-c.dampened_key(lam), -c.value, c.item_id),
    ):
        if remaining <= 0:
            break
        if cand.chars <= remaining:
            chosen.append(cand)
            chosen_ids.add(cand.item_id)
            remaining -= cand.chars
    return chosen


def _build_block(chosen: list[_Candidate]) -> BriefingBlock:
    units = [c.payload for c in chosen if c.kind == "unit"]
    entities = [c.payload for c in chosen if c.kind == "entity"]
    relations = [c.payload for c in chosen if c.kind == "relation"]
    return BriefingBlock(units=units, entities=entities, relations=relations)


def _pair_relations(
    raw_relations: list[Relation],
    rendered_relations: list[RenderedRelation],
    entity_by_id: dict[str, Entity],
) -> list[tuple[Relation, RenderedRelation]]:
    """Align raw relations to a rendered line using resolved endpoint labels.

    ``RenderedRelation`` carries no ids, so we re-render each raw relation from
    the entity set (matching ``render_relations_from``'s label resolution).
    Relations whose endpoints aren't both known entities can't be rendered here
    and are dropped (they'd be dangling anyway). ``rendered_relations`` is
    accepted for contract symmetry with ``RetrievalResult`` but the faithful
    re-render is authoritative since it preserves the raw-relation id.
    """
    del rendered_relations  # re-rendered from entity_by_id below (id-preserving)
    pairs: list[tuple[Relation, RenderedRelation]] = []
    seen: set[str] = set()
    for rel in raw_relations:
        if rel.id in seen:
            continue
        subj = entity_by_id.get(rel.subject_id)
        obj = entity_by_id.get(rel.object_id)
        if subj is None or obj is None:
            continue
        rendered = RenderedRelation(subject=subj.label, predicate=rel.predicate, object=obj.label)
        pairs.append((rel, rendered))
        seen.add(rel.id)
    return pairs


def _select_block(
    *,
    units: list,
    entities: list[Entity],
    raw_relations: list[Relation],
    rendered_relations: list[RenderedRelation],
    query_words: set[str],
    now: datetime,
    budget: int,
    alpha: float,
    exclude_ids: set[str],
    relevance_by_id: dict[str, float],
) -> tuple[BriefingBlock, set[str], tuple[RankedCandidate, ...]]:
    """Score, prune relations, knapsack-fill one block.

    Returns ``(block, chosen_ids, ranking)`` where ``ranking`` is every scored
    candidate ordered by descending value — the pre-budget ranked list (see
    :class:`RankedCandidate`). Relations are scored for the ranking *before*
    the no-dangling filter (scoring does not depend on which entities were
    selected), so the ranking covers all candidates; the selection path is
    byte-identical to before this bookkeeping existed.
    """
    entity_by_id = {e.id: e for e in entities}
    degrees = _entity_degrees(raw_relations)
    norms = _salience_norms(units, entities, raw_relations, degrees)

    # Pass 1: knapsack units + entities. Relations can only reference entities
    # that "made it into the selected set" (the no-dangling rule), so they must
    # be admitted after we know which entities were selected.
    item_candidates: list[_Candidate] = []

    for unit in units:
        if unit.id in exclude_ids:
            continue
        value = _score_unit(unit, query_words, now, norms, alpha, relevance_by_id)
        item_candidates.append(
            _Candidate(unit.id, value, char_unit(unit), "unit", SelectedUnit(unit, value))
        )

    for entity in entities:
        if entity.id in exclude_ids:
            continue
        value = _score_entity(entity, query_words, now, degrees, norms, alpha, relevance_by_id)
        item_candidates.append(
            _Candidate(entity.id, value, char_entity(entity), "entity", SelectedEntity(entity, value))
        )

    # Pass relevance_by_id to enable the reserve pass (only when alpha > 0;
    # profile block passes alpha=0 and relevance_by_id stays empty there).
    chosen = _knapsack(item_candidates, budget, relevance_by_id if alpha else None)
    used = sum(c.chars for c in chosen)
    selected_entity_ids = {c.item_id for c in chosen if c.kind == "entity"}

    # Score every paired relation up front: scoring is independent of which
    # entities were selected, so doing it pre-filter changes nothing about the
    # selection while giving the pre-budget ranking full relation coverage.
    pairs = _pair_relations(raw_relations, rendered_relations, entity_by_id)
    all_relation_candidates: list[_Candidate] = []
    for rel, rendered in pairs:
        if rel.id in exclude_ids:
            continue
        value = _score_relation(rel, rendered, query_words, now, degrees, norms, alpha, relevance_by_id)
        all_relation_candidates.append(
            _Candidate(
                rel.id, value, char_relation(rendered), "relation",
                SelectedRelation(rel, rendered, value),
            )
        )

    # Relation pruning stage 1: no dangling — both endpoints must be entities
    # that were actually selected for this block.
    non_dangling_ids = {
        rel.id
        for rel, _rendered in pairs
        if rel.subject_id in selected_entity_ids and rel.object_id in selected_entity_ids
    }
    relation_candidates = [c for c in all_relation_candidates if c.item_id in non_dangling_ids]

    # Stage 2: top-K cap by the relation's OWN scored value. (Pre-C4 this
    # ranked by max endpoint-entity value — a proxy needed when relations had
    # no signal of their own. On a dense hub node every relation shares the hub
    # endpoint, so that proxy tied them all and the cap degenerated to
    # insertion order; per-item counters give each relation a real rank.)
    relation_candidates.sort(key=lambda c: c.value, reverse=True)
    capped = relation_candidates[:MAX_RELATIONS]

    # Relations then compete in the knapsack for the remaining char budget.
    # Relations never carry a keyword-source score (coverage fallback only), so
    # the reserve pass (requiring reranked relevance ≥ ρ) is not meaningful here;
    # pass relevance_by_id anyway so any future relation-reranking signal flows in.
    chosen += _knapsack(capped, budget - used, relevance_by_id if alpha else None)

    block = _build_block(chosen)
    chosen_ids = {c.item_id for c in chosen}
    ranking = tuple(
        RankedCandidate(item_id=c.item_id, kind=c.kind, value=c.value, chars=c.chars)
        for c in sorted(
            item_candidates + all_relation_candidates,
            key=lambda c: (-c.value, c.item_id),  # value-ordered; id tie-break for determinism
        )
    )
    return block, chosen_ids, ranking


def select_briefing(
    *,
    query: str,
    now: datetime,
    query_units: list,
    query_entities: list[Entity],
    query_raw_relations: list[Relation],
    query_rendered_relations: list[RenderedRelation],
    profile_units: list,
    profile_entities: list[Entity],
    profile_raw_relations: list[Relation],
    profile_rendered_relations: list[RenderedRelation],
    profile_budget: int = PROFILE_CHAR_BUDGET,
    recall_budget: int = RECALL_CHAR_BUDGET,
    relevance_by_id: dict[str, float] | None = None,
) -> BriefingSelection:
    """Produce a two-block :class:`BriefingSelection`.

    Profile is query-independent (``alpha=0``): ranked by salience + recency.
    Query recall is ranked with the full value function and **excludes** any
    item already chosen for the profile (dedup by id).

    Both candidate sets are ``RetrievalResult``-shaped; the profile set is
    supplied separately (W1b produces it; here it is just another input so the
    module is testable standalone).

    ``relevance_by_id`` carries the keyword source's normalized BM25 scores
    (RetrievalResult.keyword_relevance); ids absent from it fall back to the
    lemmatized-coverage relevance. Profile scoring (``alpha=0``) never reads
    relevance, so the profile path is unaffected.
    """
    query_words = _content_words(query)
    relevance_by_id = relevance_by_id or {}

    profile_block, profile_ids, profile_ranking = _select_block(
        units=profile_units,
        entities=profile_entities,
        raw_relations=profile_raw_relations,
        rendered_relations=profile_rendered_relations,
        query_words=query_words,
        now=now,
        budget=profile_budget,
        alpha=0.0,  # profile is query-independent
        exclude_ids=set(),
        relevance_by_id=relevance_by_id,
    )

    query_block, _, query_ranking = _select_block(
        units=query_units,
        entities=query_entities,
        raw_relations=query_raw_relations,
        rendered_relations=query_rendered_relations,
        query_words=query_words,
        now=now,
        budget=recall_budget,
        alpha=ALPHA,
        exclude_ids=profile_ids,  # dedup against the profile
        relevance_by_id=relevance_by_id,
    )

    return BriefingSelection(
        profile_items=profile_block,
        query_items=query_block,
        profile_ranking=profile_ranking,
        query_ranking=query_ranking,
    )
