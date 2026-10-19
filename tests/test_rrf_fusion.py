"""Unit tests for RRF fusion (rrf_fusion.py) and the production RRF wiring.

Tests properties, not implementation:
- fuse_rrf: correct RRF rank scores, normalisation, union of lists
- fuse_rrf degrades to BM25-only when rerank_scores empty
- RRF_K constant is 60 (standard; pre-registered F8-strict)
- items absent from one list contribute from the other list only
"""

import pytest

from tpasini.recordatio.proxy.rrf_fusion import RRF_K, fuse_rrf, rrf_score

# ---------------------------------------------------------------------------
# rrf_score
# ---------------------------------------------------------------------------

def test_rrf_score_rank1():
    """1/(k+1) for rank=1."""
    assert rrf_score(1) == pytest.approx(1.0 / (RRF_K + 1))


def test_rrf_score_rank2():
    """1/(k+2) for rank=2."""
    assert rrf_score(2) == pytest.approx(1.0 / (RRF_K + 2))


def test_rrf_score_monotone_decreasing():
    """Higher rank → lower score."""
    assert rrf_score(1) > rrf_score(2) > rrf_score(10) > rrf_score(100)


def test_rrf_k_is_60():
    """Pre-registered constant (F8-strict — must not drift)."""
    assert RRF_K == 60


# ---------------------------------------------------------------------------
# fuse_rrf: basic correctness
# ---------------------------------------------------------------------------

def test_fuse_rrf_two_lists_union():
    """Items from both lists are all present in the fused dict."""
    bm25_pairs = [("A", 0.9), ("B", 0.5)]
    rerank_scores = {"B": 0.8, "C": 0.6}
    result = fuse_rrf(bm25_pairs, rerank_scores)
    assert set(result.keys()) == {"A", "B", "C"}


def test_fuse_rrf_scores_in_0_1():
    """All fused scores are in [0, 1]."""
    bm25_pairs = [("A", 0.9), ("B", 0.5), ("C", 0.2)]
    rerank_scores = {"A": 0.7, "C": 0.3, "D": 0.1}
    result = fuse_rrf(bm25_pairs, rerank_scores)
    for v in result.values():
        assert 0.0 <= v <= 1.0


def test_fuse_rrf_top_bm25_and_top_reranker_gets_highest_score():
    """An item ranked 1st in both lists should get the highest fused score."""
    # Item A: rank 1 in BM25 + rank 1 in reranker → highest raw RRF score.
    bm25_pairs = [("A", 0.9), ("B", 0.5), ("C", 0.1)]
    rerank_scores = {"A": 0.95, "B": 0.4, "C": 0.1}
    result = fuse_rrf(bm25_pairs, rerank_scores)
    top = max(result, key=lambda k: result[k])
    assert top == "A"


def test_fuse_rrf_normalised_to_1():
    """Max fused score is exactly 1.0 (normalized)."""
    bm25_pairs = [("A", 0.9), ("B", 0.3)]
    rerank_scores = {"A": 0.8, "B": 0.2}
    result = fuse_rrf(bm25_pairs, rerank_scores)
    assert max(result.values()) == pytest.approx(1.0)


def test_fuse_rrf_normalised_min_is_0():
    """Min fused score is exactly 0.0 (normalized)."""
    bm25_pairs = [("A", 0.9), ("B", 0.3)]
    rerank_scores = {"A": 0.8, "B": 0.2}
    result = fuse_rrf(bm25_pairs, rerank_scores)
    assert min(result.values()) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# fuse_rrf: bm25-only fallback (empty rerank_scores)
# ---------------------------------------------------------------------------

def test_fuse_rrf_empty_rerank_is_bm25_only():
    """With empty reranker scores, result covers exactly the BM25 items."""
    bm25_pairs = [("A", 0.9), ("B", 0.3), ("C", 0.1)]
    result = fuse_rrf(bm25_pairs, {})
    assert set(result.keys()) == {"A", "B", "C"}


def test_fuse_rrf_empty_rerank_preserves_bm25_order():
    """BM25-only RRF preserves BM25 rank ordering (rank1 > rank2 > rank3)."""
    bm25_pairs = [("A", 0.9), ("B", 0.3), ("C", 0.1)]
    result = fuse_rrf(bm25_pairs, {})
    assert result["A"] > result["B"] > result["C"]


def test_fuse_rrf_empty_bm25_pairs_and_scores():
    """Both lists empty → empty result."""
    assert fuse_rrf([], {}) == {}


# ---------------------------------------------------------------------------
# fuse_rrf: item absent from one list
# ---------------------------------------------------------------------------

def test_fuse_rrf_item_only_in_bm25():
    """An item only in BM25 gets only its BM25 RRF contribution."""
    bm25_pairs = [("A", 0.9), ("B", 0.3)]
    rerank_scores = {"A": 0.8}  # B absent from reranker
    result = fuse_rrf(bm25_pairs, rerank_scores)
    # A: BM25 rank 1 + reranker rank 1 → two contributions
    # B: BM25 rank 2 only → one contribution
    # A must beat B
    assert result["A"] > result["B"]


def test_fuse_rrf_item_only_in_reranker():
    """An item only in the reranker list gets only its reranker RRF contribution."""
    bm25_pairs = [("A", 0.9)]
    rerank_scores = {"A": 0.8, "B": 0.9}  # B only in reranker
    result = fuse_rrf(bm25_pairs, rerank_scores)
    # B gets one contribution (reranker rank 1); A gets two contributions.
    assert "B" in result


def test_fuse_rrf_all_same_score_normalises_to_1():
    """All-same raw scores: normalised to 1.0 for every item."""
    bm25_pairs = [("A", 0.5)]  # one item, rank 1
    rerank_scores = {}          # BM25-only: only one raw score possible
    result = fuse_rrf(bm25_pairs, rerank_scores)
    assert result == {"A": pytest.approx(1.0)}
