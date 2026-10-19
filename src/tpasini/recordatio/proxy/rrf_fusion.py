"""RRF (Reciprocal Rank Fusion) relevance source for production briefing retrieval.

This module makes RRF fusion a first-class PRODUCTION relevance source,
complementing the existing BM25 keyword source (F1).  The fused relevance dict
is passed to ``select_briefing(relevance_by_id=...)`` via the existing port —
no new contract.

Architecture (RERANKING_DESIGN.md §1.1 extended):

    BM25 ranking (KeywordSource.candidates)  ─┐
                                               ├─► _fuse_rrf ─► relevance_by_id
    Reranker ranking (frozen/live/fallback)   ─┘

The reranker stays behind its own frozen-scores / live-model / lexical-fallback
path (see ``reranker.py``).  RRF combines the *rank* of BM25 and reranker lists
— it does NOT pick a winner between them, so the reranker is not a replacement
for BM25, it is a complementary signal fused by rank.

RRF constant k=60 (standard; RETRIEVAL_DESIGN.md decision E / F8-strict):
1 / (k + rank), rank 1-based.  Pre-registered; changing it is an experiment
governed by F8 rules.

Determinism: RRF depends only on rank order.  Given the same BM25 pairs and
reranker scores, the fused dict is byte-identical across runs.  The reranker
uses frozen scores in CI/bench (rrf_fusion.py is silent about how those are
produced — see reranker.py §6).

Fallback: if no reranker scores are available (empty dict), RRF degrades
gracefully to BM25-only (each item's score is its BM25 RRF contribution only),
which is then re-normalized to [0,1].  The budgeter's existing lexical-coverage
fallback for unscored items is unaffected.
"""

# RRF constant k (standard; pre-registered F8-strict).
RRF_K = 60


def rrf_score(rank: int, k: int = RRF_K) -> float:
    """Reciprocal rank fusion score for one list: 1/(k + rank), rank 1-based."""
    return 1.0 / (k + rank)


def fuse_rrf(
    bm25_pairs: list[tuple[str, float]],
    rerank_scores: dict[str, float],
) -> dict[str, float]:
    """Reciprocal Rank Fusion of BM25 and reranker ranked lists → relevance_by_id.

    BM25 pairs are already sorted score-desc (higher = better) by KeywordSource.
    Reranker scores are ∈ [0,1] (higher = better); their rank order drives the
    second list.  Items absent from one list contribute only from the other list.

    The fused score is normalized to [0,1] over the union so it enters the
    budgeter's relevance port on the same scale as BM25 and the coverage fallback.

    With an empty ``rerank_scores`` dict (fallback path), the result is equivalent
    to BM25-only RRF scores, re-normalized — strictly BM25-rank-order-preserving.
    """
    bm25_rank = {item_id: i + 1 for i, (item_id, _) in enumerate(bm25_pairs)}
    rerank_sorted = sorted(rerank_scores.items(), key=lambda kv: -kv[1])
    rerank_rank = {item_id: i + 1 for i, (item_id, _) in enumerate(rerank_sorted)}
    all_ids = set(bm25_rank) | set(rerank_rank)
    raw: dict[str, float] = {}
    for item_id in all_ids:
        score = 0.0
        if item_id in bm25_rank:
            score += rrf_score(bm25_rank[item_id])
        if item_id in rerank_rank:
            score += rrf_score(rerank_rank[item_id])
        raw[item_id] = score
    if not raw:
        return {}
    min_s, max_s = min(raw.values()), max(raw.values())
    if max_s == min_s:
        return {k: 1.0 for k in raw}
    return {k: (v - min_s) / (max_s - min_s) for k, v in raw.items()}
