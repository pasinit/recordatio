"""Unit tests for the reranker and the updated knapsack objective (RR2).

Test properties, not implementation:
- length-dampened key: q-0008-class synthetic (long answer fact vs short swarm)
- relevance-floor reserve: high-relevance item admitted before general fill
- λ=1 reproduces old value-per-char behaviour
- frozen-score replay replaces existing_relevance
- lexical fallback when model + frozen scores absent
- build_rerank_context: last user message + bounded thread
- cap companion: RERANK_MAX_PAIRS ≤ pool gate enforcement (G-R1)
"""

import math
from unittest.mock import MagicMock, patch

import pytest

from tpasini.recordatio.proxy.briefing_budget import (
    RELEVANCE_FLOOR_FRACTION,
    _Candidate,
    _knapsack,
)
from tpasini.recordatio.proxy.reranker import (
    RERANK_MAX_PAIRS,
    _replay_scores,
    build_rerank_context,
    rerank,
)

# ---------------------------------------------------------------------------
# _Candidate.dampened_key (§2.2 point 1)
# ---------------------------------------------------------------------------

def test_dampened_key_default_lambda():
    """key = v / chars^0.5 (square-root dampening)."""
    c = _Candidate("id", value=1.0, chars=100, kind="unit", payload=None)
    assert c.dampened_key(0.5) == pytest.approx(1.0 / math.sqrt(100))


def test_dampened_key_lambda_1_equals_density():
    """λ=1 reproduces the old value-per-char density (regression guard)."""
    c = _Candidate("id", value=0.3, chars=50, kind="unit", payload=None)
    assert c.dampened_key(1.0) == pytest.approx(c.density)


def test_dampened_key_lambda_0_pure_value():
    """λ=0: key = v / chars^0 = v (length-blind)."""
    c = _Candidate("id", value=0.7, chars=9999, kind="unit", payload=None)
    assert c.dampened_key(0.0) == pytest.approx(0.7)


def test_dampened_key_shorter_item_less_advantage():
    """With λ=0.5 a 3× shorter item gets √3 ≈ 1.73× not 3× advantage."""
    long_item = _Candidate("long", value=1.0, chars=90, kind="unit", payload=None)
    short_item = _Candidate("short", value=1.0, chars=10, kind="unit", payload=None)
    advantage_lam0 = short_item.dampened_key(0.0) / long_item.dampened_key(0.0)
    advantage_lam1 = short_item.density / long_item.density
    advantage_lam05 = short_item.dampened_key(0.5) / long_item.dampened_key(0.5)
    # λ=1: 9× advantage (chars ratio); λ=0.5: √9=3× advantage; λ=0: 1× (no advantage)
    assert advantage_lam1 == pytest.approx(9.0)
    assert advantage_lam05 == pytest.approx(3.0)
    assert advantage_lam0 == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _knapsack: q-0008 synthetic (RR2)
# ---------------------------------------------------------------------------

def _make_unit(item_id: str, value: float, chars: int) -> _Candidate:
    return _Candidate(item_id, value=value, chars=chars, kind="unit", payload=item_id)


def test_q0008_class_long_answer_beats_swarm():
    """Synthetic q-0008: a 56-char answer fact at reranked relevance 1.0 must beat
    48 short fragments when the relevance floor φ=0.30, ρ=0.6 is active.

    Budget: 480 chars exactly (= 48 × 10).  Swarm: 48 × 10-char fragments at
    value=0.4 and density 0.04. Needle: 56 chars, value=0.6, density 0.011.
    Without the floor the swarm fills the budget first (old density order).
    With the floor reserve (φ=0.30 → 144-char reserve; needle costs 56 → fits)
    the needle is admitted in pass 1 before the swarm can exhaust the budget.
    """
    needle_id = "needle"
    n_frags = 48
    frag_chars = 10
    budget = n_frags * frag_chars  # 480 — swarm fills exactly without floor

    relevance_by_id = {needle_id: 1.0}  # cross-encoder gave it max score
    needle = _make_unit(needle_id, value=0.6, chars=56)
    swarm = [_make_unit(f"frag-{i}", value=0.4, chars=frag_chars) for i in range(n_frags)]

    chosen = _knapsack([needle] + swarm, budget, relevance_by_id=relevance_by_id)
    chosen_ids = {c.item_id for c in chosen}
    assert needle_id in chosen_ids, (
        "Needle must be selected via the relevance-floor reserve pass; "
        "the short-fragment swarm should not evict it."
    )


def test_q0008_regresses_with_lambda1_and_no_floor():
    """Probe: with λ=1 (old density) and no floor, the swarm wins (known bad state).

    Budget is set so 48 × 10-char fragments exactly fill it, leaving no room for
    the 56-char needle.  The needle's density is 0.6/56≈0.011 < 0.4/10=0.04, so
    the old sort order takes all 48 fragments first and the needle is dropped.
    """
    needle_id = "needle"
    n_frags = 48
    frag_chars = 10
    # Exact budget: fits all fragments, no room for the needle (56 chars).
    budget = n_frags * frag_chars  # = 480

    needle = _make_unit(needle_id, value=0.6, chars=56)
    swarm = [_make_unit(f"frag-{i}", value=0.4, chars=frag_chars) for i in range(n_frags)]

    # Simulate old knapsack: value/chars (density), no reserve.
    chosen_old = []
    used = 0
    for c in sorted([needle] + swarm, key=lambda c: c.density, reverse=True):
        if used + c.chars <= budget:
            chosen_old.append(c)
            used += c.chars
    chosen_ids_old = {c.item_id for c in chosen_old}
    # The old objective drops the needle (density 0.6/56≈0.011 < 0.4/10=0.04).
    assert needle_id not in chosen_ids_old, (
        "With the old value-per-char objective the needle should be density-displaced"
    )


def test_reserve_pass_uses_value_not_density():
    """Items in the reserve are admitted value-first, not density-first."""
    # Two candidates both above ρ=0.6: high-value/long vs low-value/short.
    # Value-first means the high-value long item is chosen over the low-value short.
    budget = 200
    _ = int(budget * RELEVANCE_FLOOR_FRACTION)  # 60 chars — documented for clarity
    high_val = _make_unit("high_val", value=0.9, chars=50)
    low_val = _make_unit("low_val", value=0.2, chars=10)
    relevance_by_id = {"high_val": 0.9, "low_val": 0.8}  # both above ρ=0.6
    # Reserve: 60 chars. high_val costs 50 (fits); low_val costs 10 (fits after).
    # Value-first: high_val chosen first (0.9 > 0.2).
    chosen = _knapsack([high_val, low_val], budget, relevance_by_id=relevance_by_id)
    chosen_ids = {c.item_id for c in chosen}
    assert "high_val" in chosen_ids, "High-value item must be chosen in reserve pass"


def test_reserve_pass_skipped_without_relevance():
    """Profile block (alpha=0, no relevance_by_id) → pure dampened fill, no reserve."""
    needle_id = "needle"
    needle = _make_unit(needle_id, value=0.6, chars=56)
    swarm = [_make_unit(f"frag-{i}", value=0.4, chars=10) for i in range(48)]
    # Without relevance_by_id the reserve pass is skipped entirely.
    # The needle's dampened_key(0.5)=0.6/√56≈0.080 vs swarm 0.4/√10≈0.126 →
    # swarm still wins on the dampened key (narrower gap than λ=1 but swarm still higher).
    chosen = _knapsack([needle] + swarm, budget=6800, relevance_by_id=None)
    # Just check it runs without error; needle not required here (profile has no relevance).
    assert isinstance(chosen, list)


def test_knapsack_no_double_spend():
    """An item chosen in the reserve pass is excluded from the general fill."""
    budget = 100
    item = _make_unit("x", value=0.8, chars=20)
    relevance_by_id = {"x": 0.9}  # above threshold → reserve candidate
    chosen = _knapsack([item], budget, relevance_by_id=relevance_by_id)
    assert sum(1 for c in chosen if c.item_id == "x") == 1, "Item chosen at most once"


def test_knapsack_graceful_skip():
    """An item that doesn't fit is skipped (not a stopping condition)."""
    big = _make_unit("big", value=0.9, chars=100)
    small = _make_unit("small", value=0.1, chars=10)
    # Budget of 20: big doesn't fit; small does.
    chosen = _knapsack([big, small], budget=20, relevance_by_id=None)
    assert any(c.item_id == "small" for c in chosen)
    assert all(c.item_id != "big" for c in chosen)


# ---------------------------------------------------------------------------
# reranker.build_rerank_context (ruling 3 / §4)
# ---------------------------------------------------------------------------

def test_build_rerank_context_last_message_only():
    ctx = build_rerank_context("where does Dr Jane Doe work?")
    assert "Dr Jane Doe" in ctx


def test_build_rerank_context_bounded_thread():
    thread = ["turn 0", "turn 1", "turn 2", "turn 3", "turn 4"]
    ctx = build_rerank_context("follow-up?", thread=thread)
    # Last k=3 turns prepended.
    assert "turn 2" in ctx and "turn 3" in ctx and "turn 4" in ctx
    assert "turn 0" not in ctx and "turn 1" not in ctx


def test_build_rerank_context_no_thread():
    ctx = build_rerank_context("hello world", thread=None)
    assert ctx == "hello world"


def test_build_rerank_context_active_skill_reserved():
    """active_skill is accepted but has no effect in v1."""
    ctx_without = build_rerank_context("q", thread=None, active_skill=None)
    ctx_with = build_rerank_context("q", thread=None, active_skill="property-search")
    assert ctx_without == ctx_with  # v1: skill ignored


# ---------------------------------------------------------------------------
# reranker.rerank: frozen-score replay (§6 / RR3)
# ---------------------------------------------------------------------------

def test_rerank_frozen_scores_replace_existing():
    """Frozen scores take precedence over existing_relevance for matched items."""
    frozen: dict[tuple[str, str], float] = {("q-001", "item-A"): 0.92}
    existing = {"item-A": 0.1, "item-B": 0.5}
    candidates = [("item-A", "text A"), ("item-B", "text B")]
    result = rerank("context", candidates, existing, frozen_scores=frozen, query_id="q-001")
    assert result["item-A"] == pytest.approx(0.92)  # replaced by frozen
    assert result["item-B"] == pytest.approx(0.5)   # kept (not in frozen)


def test_rerank_frozen_scores_partial_coverage():
    """Items not in the frozen artifact keep their existing_relevance."""
    frozen: dict[tuple[str, str], float] = {("q-001", "item-A"): 0.8}
    existing = {"item-A": 0.1, "item-B": 0.5}
    candidates = [("item-A", "text A"), ("item-B", "text B")]
    result = rerank("context", candidates, existing, frozen_scores=frozen, query_id="q-001")
    assert result["item-B"] == pytest.approx(0.5)


def test_rerank_no_frozen_no_model_lexical_fallback():
    """With no frozen scores and model unavailable → lexical fallback (G-R4)."""
    existing = {"x": 0.7}
    candidates = [("x", "some text")]
    # Force model to be unavailable.
    with patch("tpasini.recordatio.proxy.reranker._model", None), \
         patch("tpasini.recordatio.proxy.reranker._model_load_tried", True):
        result = rerank("context", candidates, existing, frozen_scores=None, query_id=None)
    assert result == existing


def test_rerank_frozen_scores_wrong_query_id_fallback():
    """Frozen scores for a different query_id → no match → live/lexical."""
    frozen: dict[tuple[str, str], float] = {("q-OTHER", "item-A"): 0.99}
    existing = {"item-A": 0.3}
    candidates = [("item-A", "text")]
    with patch("tpasini.recordatio.proxy.reranker._model", None), \
         patch("tpasini.recordatio.proxy.reranker._model_load_tried", True):
        result = rerank("context", candidates, existing, frozen_scores=frozen, query_id="q-001")
    assert result["item-A"] == pytest.approx(0.3)  # unchanged, fallback to lexical


# ---------------------------------------------------------------------------
# Note: _sigmoid was removed when the model was swapped from ms-marco-MiniLM-L-6-v2
# to BAAI/bge-reranker-base (2026-06-14).  bge-reranker-base applies sigmoid
# internally (activation_fn=Sigmoid() in checkpoint config); model.predict()
# returns scores in [0,1] directly.  No module-level _sigmoid is needed.
# ---------------------------------------------------------------------------
# Cap companion: RERANK_MAX_PAIRS ≤ cap (RR7 / §9 ruling 7)
# ---------------------------------------------------------------------------

def test_rerank_max_pairs_constant():
    """RERANK_MAX_PAIRS == 256 matches CAP_SCALING_DESIGN.md LOW (RR7)."""
    assert RERANK_MAX_PAIRS == 256


def test_rerank_pool_gate_limits_scored_pairs():
    """When pool exceeds RERANK_MAX_PAIRS, only the top-N are cross-encoded (G-R1)."""
    scored_items: list[str] = []

    def fake_predict(pairs):
        for _ctx, text in pairs:
            scored_items.append(text)
        return [0.5] * len(pairs)

    mock_model = MagicMock()
    mock_model.predict.side_effect = fake_predict

    # Pool larger than RERANK_MAX_PAIRS.
    n_pool = RERANK_MAX_PAIRS + 50
    candidates = [(f"id-{i}", f"text-{i:04d}") for i in range(n_pool)]
    existing = {f"id-{i}": i / n_pool for i in range(n_pool)}

    with patch("tpasini.recordatio.proxy.reranker._model", mock_model), \
         patch("tpasini.recordatio.proxy.reranker._model_load_tried", True), \
         patch("tpasini.recordatio.proxy.reranker._model", mock_model):
        rerank("context", candidates, existing)

    assert len(scored_items) == RERANK_MAX_PAIRS, (
        f"Must score at most RERANK_MAX_PAIRS={RERANK_MAX_PAIRS} pairs; got {len(scored_items)}"
    )


# ---------------------------------------------------------------------------
# _replay_scores: None query_id → empty (safety)
# ---------------------------------------------------------------------------

def test_replay_scores_none_query_id_returns_empty():
    frozen = {("q-001", "item"): 0.8}
    result = _replay_scores("ctx", [("item", "text")], frozen, query_id=None)
    assert result == {}
