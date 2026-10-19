"""Candidate sources for query-time retrieval (RETRIEVAL_DESIGN.md §2–§3).

F1 enables exactly one source: FTS5 keyword search with BM25 over the
pipeline-normalized index built by :class:`SQLiteStore`. The
:class:`CandidateSource` protocol is the seam F2 (embeddings, hybrid fusion)
and F3 (entity-link JOINs) plug into later; no fusion rule exists yet because
F1's consumer takes the keyword source alone.

Query construction (§3.3): gazetteer multi-word-entity matches and the
capitalization heuristic become exact FTS5 phrase queries (precision-first);
leftover non-stopword words become loose terms; everything is OR-joined and
BM25's more-matches-score-higher does the conjunction work. Every phrase and
term passes through the same §3.1 pipeline the index stores (gates R6/R7).

Boosting does NOT live here: column weights below express index-internal
priority only; item-kind/salience/recency trade-offs stay in the budgeter
(decision A).

Cap-scaling (CAP_SCALING_DESIGN.md §3 / task #41): the candidate cap is
population-relative — ``cap(scope_set) = clamp(K · N_active, LOW, HIGH)`` —
so it never truncates a scope whose active-unit population exceeds the old
static default, while remaining bounded above (compute) and below (recall).
K=1, LOW=256, HIGH=2000 are pre-registered from first principles (§3):
  * LOW = ⌈max(RECALL, PROFILE) / min_item_size⌉, rounded to 256 — the cap
    can never sit below the budget's own selection ceiling (§3.1).
  * HIGH = 2000 — above the current largest scope with headroom, far below
    any latency concern; a safety rail, not a quality knob (§3.2).
  * K=1 — the cap equals the active-unit count, removing truncation by
    construction while the budget does all ranking (§3.3).
The companion RRF fusion (see rrf_fusion.py) now owns in-pool ranking quality;
the cap's only job is "generate enough for the fusion to choose from."
"""

import os
import re
from collections.abc import Sequence
from typing import Protocol

from tpasini.recordatio.retrieval_text import STOPWORDS, normalize_for_index, pipeline_tokens
from tpasini.recordatio.storage.base import SQLiteStore

type ScopeSpec = str | Sequence[str]


def _scope_tuple(scopes: ScopeSpec) -> tuple[str, ...]:
    return (scopes,) if isinstance(scopes, str) else tuple(scopes)

# Per-source candidate cap — env override (hard ceiling, back-compat with F1
# experiments).  When RECORDATIO_RETRIEVAL_CANDIDATE_CAP is set it pins the cap;
# unset, the population-relative rule below runs (CAP_SCALING_DESIGN.md §4).
_CANDIDATE_CAP_ENV = "RECORDATIO_RETRIEVAL_CANDIDATE_CAP"

# Cap-scaling constants (CAP_SCALING_DESIGN.md §3, pre-registered, F8-strict):
# never tuned against the frozen annotations.
_CAP_K = 1          # slope: cap = K × N_active (K=1 removes truncation by construction)
_CAP_LOW = 256      # floor: budget can admit at most 6800/27≈252 items; 256 ≥ that ceiling
_CAP_HIGH = 2000    # ceiling: safety rail, above current largest scope (784), below latency risk

# §3.3 step 2: the unseen-MWE capitalization heuristic, optional, default on.
_CAP_HEURISTIC_ENV = "RECORDATIO_RETRIEVAL_CAP_HEURISTIC"

# BM25 column weights (§3.2): index-internal priority only — content 1.0,
# label 0.6, description 0.4; UNINDEXED columns 0. Module constants; changing
# them is an experiment (F8 rules apply).
_BM25_WEIGHTS = "0.0, 0.0, 0.0, 1.0, 0.6, 0.4"

# Gazetteer scope: labels/aliases shorter than this match too promiscuously.
_MIN_LABEL_LEN = 4

# Word tokens with positions, for gazetteer spans and the capitalization scan.
_WORD_SPAN_RE = re.compile(r"\w[\w'’]*")

# Sentence-terminating punctuation: a capitalized token right after one of
# these (or at text start) may owe its capital to sentence position.
_SENTENCE_END = (".", "!", "?", "\n")


def _count_active_units(store: SQLiteStore, scopes: tuple[str, ...]) -> int:
    """Count active (valid_until IS NULL) knowledge units across the scope set.

    Used by the population-relative cap rule (CAP_SCALING_DESIGN.md §3.3).
    Returns 0 on any error so the rule degrades safely to LOW.
    """
    try:
        marks = ",".join("?" for _ in scopes)
        row = store._conn.execute(
            f"SELECT COUNT(*) FROM knowledge_units WHERE agent_id IN ({marks}) AND valid_until IS NULL",  # noqa: S608
            scopes,
        ).fetchone()
        return int(row[0]) if row else 0
    except Exception:
        return 0


def candidate_cap(store: SQLiteStore | None = None, scopes: tuple[str, ...] | None = None) -> int:
    """Per-source candidate cap: population-relative rule, env-overridable.

    Population-relative rule (CAP_SCALING_DESIGN.md §3):
        cap = clamp(K × N_active, LOW, HIGH)
    where N_active is the active-unit count across the scope set.

    If RECORDATIO_RETRIEVAL_CANDIDATE_CAP is set, it pins the cap (hard ceiling
    override for operational escape or F8 experiments — back-compat with F1).
    If store/scopes are not provided (legacy callers), returns LOW (conservative
    floor — equivalent to the old static default for small scopes).
    """
    env_val = os.environ.get(_CANDIDATE_CAP_ENV)
    if env_val is not None:
        try:
            return int(env_val)
        except (TypeError, ValueError):
            pass
    if store is None or scopes is None:
        return _CAP_LOW
    n_active = _count_active_units(store, scopes)
    return max(_CAP_LOW, min(_CAP_HIGH, _CAP_K * n_active))


def _cap_heuristic_enabled() -> bool:
    return os.environ.get(_CAP_HEURISTIC_ENV, "1") not in ("0", "false", "no")


class CandidateSource(Protocol):
    def candidates(self, query: str, agent_id: ScopeSpec, limit: int) -> list[tuple[str, float]]:
        """Return up to `limit` (item_id, score) pairs for this scope set.

        ``agent_id`` is one scope or a composed visibility set (the F1 seam
        with scope-set support, IDENTITY_DESIGN.md §4.2): one query, one
        candidate pool over the union. Scores are source-local (higher =
        better); cross-source comparability is NOT promised by this contract.
        """
        ...


def _overlaps(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(start < e and s < end for s, e in spans)


class KeywordSource:
    """FTS5 + BM25 keyword candidates over units and entities (§3).

    Requires the store's FTS index (``store.fts_enabled``); the consumer is
    responsible for the LIKE fallback when FTS5 is unavailable (§4 G-3) — this
    class then returns no candidates.
    """

    def __init__(self, store: SQLiteStore) -> None:
        self._store = store

    @property
    def available(self) -> bool:
        return self._store.fts_enabled

    # -- gazetteer ------------------------------------------------------------

    def _gazetteer_terms(self, agent_id: ScopeSpec) -> list[str]:
        """The scope set's entity labels and aliases, length ≥ 4 (§3.3 step 1).

        The composed visibility's user scope contributes the principal's
        standing aliases, gating query-name matching (IDENTITY_DESIGN.md
        §4.2). Aliases were empty on the v1 artifacts — no v1 expectation
        rests on them.
        """
        scopes = _scope_tuple(agent_id)
        marks = ",".join("?" for _ in scopes)
        conn = self._store._conn
        labels = [
            r["label"]
            for r in conn.execute(f"SELECT label FROM entities WHERE agent_id IN ({marks})", scopes)  # noqa: S608
        ]
        labels += [
            r["alias"]
            for r in conn.execute(f"SELECT alias FROM entity_aliases WHERE agent_id IN ({marks})", scopes)  # noqa: S608
        ]
        return [t for t in labels if len(t) >= _MIN_LABEL_LEN]

    def _gazetteer_spans(self, query: str, agent_id: ScopeSpec) -> list[tuple[int, int]]:
        """Non-overlapping, longest-match-first, case-insensitive label spans.

        Word-boundary matching (not bare substring): precision-first — a label
        inside another word is noise, and inflected forms still match later via
        the lemmatized phrase (gate R7).
        """
        query_lower = query.lower()
        matches: list[tuple[int, int]] = []
        for term in self._gazetteer_terms(agent_id):
            if term.lower() not in query_lower:
                continue
            pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)
            matches.extend(m.span() for m in pattern.finditer(query))
        # Longest first, then leftmost; greedy non-overlapping selection.
        chosen: list[tuple[int, int]] = []
        for start, end in sorted(matches, key=lambda s: (-(s[1] - s[0]), s[0])):
            if not _overlaps(start, end, chosen):
                chosen.append((start, end))
        return sorted(chosen)

    # -- capitalization heuristic (§3.3 step 2) --------------------------------

    def _capitalized_spans(self, query: str, consumed: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Maximal runs of ≥ 2 capitalized tokens, outside consumed spans.

        A run may not begin at sentence start: a sentence-initial capital is
        positional, so the run's first token is dropped and the remainder kept
        if still ≥ 2 tokens ("What Springfield Bridge flat…" → "Springfield Bridge").
        Tokens count as adjacent only across pure whitespace (punctuation
        breaks the run).
        """
        tokens = [(m.start(), m.end(), m.group()) for m in _WORD_SPAN_RE.finditer(query)]
        runs: list[list[tuple[int, int, bool]]] = []  # token = (start, end, sentence_initial)
        current: list[tuple[int, int, bool]] = []
        prev_end: int | None = None
        for start, end, text in tokens:
            gap = query[prev_end:start] if prev_end is not None else None
            sentence_initial = prev_end is None or any(p in gap for p in _SENTENCE_END)
            capitalized = text[0].isupper() and not _overlaps(start, end, consumed)
            adjacent = current and gap is not None and gap.strip() == "" and not sentence_initial
            if capitalized and adjacent:
                current.append((start, end, False))
            elif capitalized:
                if len(current) >= 2:
                    runs.append(current)
                current = [(start, end, sentence_initial)]
            else:
                if len(current) >= 2:
                    runs.append(current)
                current = []
            prev_end = end
        if len(current) >= 2:
            runs.append(current)
        spans: list[tuple[int, int]] = []
        for run in runs:
            if run[0][2]:  # sentence-initial first token: positional capital
                run = run[1:]
            if len(run) >= 2:
                spans.append((run[0][0], run[-1][1]))
        return spans

    # -- match expression (§3.3 steps 3–4) -------------------------------------

    def build_match_expression(self, query: str, agent_id: ScopeSpec) -> str:
        """All phrases and loose terms, pipeline-normalized, OR-joined."""
        consumed = self._gazetteer_spans(query, agent_id)
        phrase_spans = list(consumed)
        if _cap_heuristic_enabled():
            phrase_spans += self._capitalized_spans(query, consumed)
        phrase_spans.sort()

        parts: list[str] = []
        seen: set[str] = set()
        for start, end in phrase_spans:
            phrase = normalize_for_index(query[start:end])
            if phrase and phrase not in seen:
                seen.add(phrase)
                parts.append(f'"{phrase}"')

        # Leftover words: the unconsumed segments through the shared pipeline,
        # stopwords dropped (same frozen list the coverage relevance uses).
        cursor = 0
        segments: list[str] = []
        for start, end in phrase_spans:
            segments.append(query[cursor:start])
            cursor = end
        segments.append(query[cursor:])
        for segment in segments:
            for token in pipeline_tokens(segment):
                if token not in STOPWORDS and token not in seen:
                    seen.add(token)
                    parts.append(f'"{token}"')
        return " OR ".join(parts)

    # -- execution (§3.3 step 5) ------------------------------------------------

    def candidates(self, query: str, agent_id: ScopeSpec, limit: int) -> list[tuple[str, float]]:
        """Up to `limit` (item_id, score) pairs; score = −bm25 (higher = better).

        ``agent_id`` may be a composed scope set: one FTS query filters
        ``agent_id IN (...)`` so the union is ONE BM25 pool (scores from one
        index on one query are comparable; separate per-scope normalization
        would fabricate parity between pools of different sizes —
        IDENTITY_DESIGN.md §4.2). Deterministic (G-4): ties broken by item_id.
        The index is never trusted for row-level validity — the consumer
        re-applies bitemporal and scope filters at materialization (§3.2).
        """
        if not self.available:
            return []
        scopes = _scope_tuple(agent_id)
        expression = self.build_match_expression(query, scopes)
        if not expression:
            return []
        # NB: FTS5 forbids aliasing the table in MATCH/bm25(); the unqualified
        # name resolves correctly even with a schema-qualified FROM.
        schema = self._store._fts_schema
        marks = ",".join("?" for _ in scopes)
        rows = self._store._conn.execute(
            f"""
            SELECT item_id, bm25(memento_fts, {_BM25_WEIGHTS}) AS score
            FROM {schema}.memento_fts
            WHERE memento_fts MATCH ? AND agent_id IN ({marks})
            ORDER BY score ASC, item_id ASC
            LIMIT ?
            """,
            (expression, *scopes, limit),
        ).fetchall()
        return [(r["item_id"], -r["score"]) for r in rows]
