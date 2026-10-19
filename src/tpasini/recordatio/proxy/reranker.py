"""Cross-encoder context reranker (RERANKING_DESIGN.md §3–§6).

Sits between candidate generation and the knapsack:

    candidates (capped pool) ──► reranker ──► relevance_by_id{id: ctx_rel ∈ [0,1]}
                                                     │
                                                     ▼
                              briefing_budget.select_briefing(relevance_by_id=…)

The reranker *replaces* the keyword-BM25 entry of ``relevance_by_id`` with a
context-relevance score for the items it scores; items it does not score keep
the lemmatized-coverage / BM25 fallback (§1.1).

Determinism (§6): CI never loads the model. The benchmark loads a frozen
``tests/benchmark/rerank-scores-v{1,2}.json`` artifact — a map
``{(query_id, item_id): score}`` produced once offline by the pinned checkpoint.
The capability check is::

    1. frozen-scores artifact available  → replay scores (CI + benchmark)
    2. model present (warm pool loaded)  → live cross-encoder scoring (prod)
    3. neither                           → lexical fallback with one logged warning

Context assembly (§4 / ruling 3): last user message + last k=3 user turns'
content words (bounded thread). Skill/topic hook reserved as ``active_skill``
arg; v1 always passes ``None``.

Cap companion (§9 / ruling 7): pool gate ``RERANK_MAX_PAIRS`` (default 256)
caps the cross-encoder; above that, the pre-ranked BM25/coverage head is
cross-encoded and the tail keeps lexical scores (G-R1). This is where the
cap-scaling companion rule lands: the reranker owns in-pool ranking, and
``RERANK_MAX_PAIRS ≤ cap`` is the interaction rule.
"""

import logging
import os
import threading
import time

_logger = logging.getLogger("recordatio.reranker")

# ---------------------------------------------------------------------------
# Pre-registered constants (RERANKING_DESIGN.md §2, §5, §7 — F8-strict;
# never swept against the frozen annotations).
# ---------------------------------------------------------------------------

# Pool gate G-R1: max pairs fed to the cross-encoder per turn.
# Matches CAP_SCALING_DESIGN.md LOW (the cap floor) so RERANK_MAX_PAIRS ≤ cap.
RERANK_MAX_PAIRS: int = int(os.environ.get("RECORDATIO_RERANK_MAX_PAIRS", 256))

# Bounded thread context: last k user turns prepended to last user message.
# Pre-registered k=3 (ruling 3); env-overridable for ablation studies only.
RERANK_CONTEXT_TURNS: int = int(os.environ.get("RECORDATIO_RERANK_CONTEXT_TURNS", 3))

# Pinned cross-encoder checkpoint (BAAI/bge-reranker-base: XLM-RoBERTa short-text
# reranker, multilingual, domain-agnostic; sanity-gate passed 2026-06-14;
# never fine-tuned on eval; fully offline; deterministic weights → frozen
# scores in CI).  Replaces cross-encoder/ms-marco-MiniLM-L-6-v2 which scored
# median 0.000 on memory-fact pairs (model-domain mismatch; see findings.md).
# bge-reranker-base applies sigmoid internally; predict() returns scores ∈ [0,1];
# no additional sigmoid step is needed in _score_live.
_MODEL_NAME = "BAAI/bge-reranker-base"

# ---------------------------------------------------------------------------
# Warm-pool: one model load per process (G-R3).
# ---------------------------------------------------------------------------

_model_lock = threading.Lock()
_model = None          # the loaded CrossEncoder, or None if unavailable
_model_load_tried = False
_fallback_warned = False  # G-R4: warn once


def _load_model() -> object | None:
    """Load the pinned cross-encoder once (warm-pool, G-R3).

    Returns the model or None if the package or weights are unavailable.
    Sets ``_model_load_tried`` so the attempt is made at most once per
    process regardless of outcome.
    """
    global _model, _model_load_tried
    if _model_load_tried:
        return _model
    with _model_lock:
        if _model_load_tried:
            return _model
        try:
            from sentence_transformers import CrossEncoder  # noqa: PLC0415
            _model = CrossEncoder(_MODEL_NAME)
            _logger.info("reranker: loaded cross-encoder %s (warm pool)", _MODEL_NAME)
        except Exception as exc:
            _logger.warning(
                "reranker: could not load cross-encoder %s (%s); falling back to lexical",
                _MODEL_NAME, exc,
            )
            _model = None
        finally:
            _model_load_tried = True
    return _model


def get_model() -> object | None:
    """Return the warm-pool model, loading it on first call (G-R3)."""
    return _load_model()


# ---------------------------------------------------------------------------
# Context assembly (ruling 3 / §4).
# ---------------------------------------------------------------------------


def build_rerank_context(
    last_user_message: str,
    thread: list[str] | None = None,
    active_skill: str | None = None,  # reserved; v1 always None
) -> str:
    """Assemble the reranker's context string.

    v1: last user message + up to RERANK_CONTEXT_TURNS prior user turns
    (space-joined), bounded. Full thread verbatim is rejected: dilutes
    attention and re-introduces the cost the bounded-context thesis avoids.

    ``active_skill`` is reserved (future skill-scoping, ruling 3 deferral);
    v1 ignores it.
    """
    parts: list[str] = []
    if thread:
        for turn in thread[-(RERANK_CONTEXT_TURNS):]:
            if turn.strip():
                parts.append(turn.strip())
    parts.append(last_user_message)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Score replay from frozen artifact (§6 / ruling 6).
# ---------------------------------------------------------------------------


def _replay_scores(
    context: str,
    candidates: list[tuple[str, str]],  # (item_id, text)
    frozen_scores: dict[tuple[str, str], float],
    query_id: str | None,
) -> dict[str, float]:
    """Return scores from the frozen artifact for the matched (query_id, item_id) pairs.

    Items without a frozen score are absent from the returned dict; the caller
    applies the lexical fallback for those (G-R4 principle extended to partial
    coverage).
    """
    result: dict[str, float] = {}
    if query_id is None:
        return result
    for item_id, _text in candidates:
        key = (query_id, item_id)
        if key in frozen_scores:
            result[item_id] = frozen_scores[key]
    return result


# ---------------------------------------------------------------------------
# Live cross-encoder scoring.
# ---------------------------------------------------------------------------


def _score_live(
    context: str,
    candidates: list[tuple[str, str]],  # (item_id, text)
    model: object,
) -> dict[str, float]:
    """Score candidates via the live cross-encoder.

    Returns context-relevance ∈ [0, 1] per item_id, bounded to
    RERANK_MAX_PAIRS (G-R1: pre-ranked head; tail keeps lexical).

    bge-reranker-base applies sigmoid internally (activation_fn=Sigmoid() is
    set in the checkpoint config); model.predict() already returns scores in
    [0, 1].  No additional sigmoid is applied here.
    """
    if not candidates:
        return {}
    pairs = [(context, text) for _item_id, text in candidates[:RERANK_MAX_PAIRS]]
    t0 = time.perf_counter()
    raw_scores = model.predict(pairs)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    _logger.debug("reranker: scored %d pairs in %.1f ms", len(pairs), elapsed_ms)
    result: dict[str, float] = {}
    for (item_id, _text), score in zip(candidates[:RERANK_MAX_PAIRS], raw_scores, strict=True):
        # Clamp to [0, 1]: bge-reranker-base outputs sigmoid-activated scores;
        # clamp guards against float rounding at boundaries.
        result[item_id] = float(max(0.0, min(1.0, float(score))))
    return result


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def get_rerank_scores(
    context: str,
    candidates: list[tuple[str, str]],  # (item_id, text) — pre-ranked BM25/coverage order
    existing_relevance: dict[str, float],  # used for RERANK_MAX_PAIRS pre-rank head selection
    *,
    frozen_scores: dict[tuple[str, str], float] | None = None,
    query_id: str | None = None,
) -> dict[str, float]:
    """Return raw cross-encoder scores only (no merge with existing_relevance).

    Same capability-check order as ``rerank()``:
    1. Frozen-scores artifact → replay.
    2. Live model → live scoring.
    3. Neither → empty dict (caller applies lexical fallback via existing_relevance).

    This is the seam for RRF fusion: the caller computes
    ``fuse_rrf(bm25_pairs, get_rerank_scores(...))`` to produce a fused
    relevance dict that combines rank information from both lists.  The merged
    dict from ``rerank()`` is still available for non-RRF usage paths.
    """
    global _fallback_warned

    if len(candidates) > RERANK_MAX_PAIRS:
        scored_head = sorted(
            candidates,
            key=lambda t: (-existing_relevance.get(t[0], 0.0), t[0]),
        )[:RERANK_MAX_PAIRS]
    else:
        scored_head = candidates

    # Attempt 1: frozen artifact.
    if frozen_scores is not None and query_id is not None:
        new_scores = _replay_scores(context, scored_head, frozen_scores, query_id)
        if new_scores:
            return new_scores
        # No match for this query_id → fall through.

    # Attempt 2: live model.
    model = get_model()
    if model is not None:
        return _score_live(context, scored_head, model)

    # Attempt 3: lexical fallback — return empty; caller uses BM25-only RRF.
    if not _fallback_warned:
        _logger.warning(
            "reranker: cross-encoder unavailable and no frozen scores; "
            "RRF fusion will use BM25-only ranking (G-R4). "
            "This is the expected path in CI when neither artifact nor model is present."
        )
        _fallback_warned = True
    return {}


def rerank(
    context: str,
    candidates: list[tuple[str, str]],  # (item_id, text) — pre-ranked BM25/coverage order
    existing_relevance: dict[str, float],  # current keyword_relevance (BM25/coverage)
    *,
    frozen_scores: dict[tuple[str, str], float] | None = None,
    query_id: str | None = None,
) -> dict[str, float]:
    """Produce a merged ``relevance_by_id`` for ``select_briefing``.

    Priority order (§6 capability check):
    1. Frozen-scores artifact (``frozen_scores`` is non-None and covers the pair)
       → replay; covers CI and benchmark.
    2. Live cross-encoder (model loaded) → live scoring.
    3. Neither → lexical fallback (G-R4): return ``existing_relevance`` unchanged,
       log one warning.

    The returned dict merges:
    - cross-encoder context-relevance for scored items (replaces their BM25 entry)
    - existing_relevance for items not scored (relations always; items outside the
      RERANK_MAX_PAIRS head)

    The merge is copy-on-write: ``existing_relevance`` is not mutated.

    For RRF fusion usage, prefer ``get_rerank_scores()`` which returns only the
    cross-encoder scores without merging.
    """
    global _fallback_warned

    # Pre-rank by existing relevance for the RERANK_MAX_PAIRS head (G-R1):
    # if there are more candidates than RERANK_MAX_PAIRS, cross-encode only
    # the top-N by current score; the tail keeps its lexical score.
    if len(candidates) > RERANK_MAX_PAIRS:
        # Sort by descending existing relevance; fall back to item_id for ties.
        scored_head = sorted(
            candidates,
            key=lambda t: (-existing_relevance.get(t[0], 0.0), t[0]),
        )[:RERANK_MAX_PAIRS]
    else:
        scored_head = candidates

    new_scores: dict[str, float] | None = None

    # Attempt 1: frozen artifact.
    if frozen_scores is not None and query_id is not None:
        new_scores = _replay_scores(context, scored_head, frozen_scores, query_id)
        if new_scores:
            # Partial coverage OK: items not in the frozen artifact keep lexical.
            result = dict(existing_relevance)
            result.update(new_scores)
            return result
        # Frozen artifact present but no match for this query_id → fall through.

    # Attempt 2: live model.
    model = get_model()
    if model is not None:
        new_scores = _score_live(context, scored_head, model)
        result = dict(existing_relevance)
        result.update(new_scores)
        return result

    # Attempt 3: lexical fallback (G-R4).
    if not _fallback_warned:
        _logger.warning(
            "reranker: cross-encoder unavailable and no frozen scores; "
            "falling back to lexical relevance (G-R4). "
            "This is the expected path in CI when neither artifact nor model is present."
        )
        _fallback_warned = True
    return existing_relevance
