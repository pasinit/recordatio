"""Tests for cache-aware ``lookup_cost``.

Covers the eight cases from issue #34 §1d. Expected values are hand-computed
from the formula:

    uncached     = max(0, input_tokens - cache_creation_tokens - cache_read_tokens)
    input_cost   = uncached              * input_price
    write_cost   = cache_creation_tokens * input_price * cache_write_multiplier
    read_cost    = cache_read_tokens     * input_price * cache_read_multiplier
    output_cost  = output_tokens         * output_price
    total = (input_cost + write_cost + read_cost + output_cost) / 1_000_000

For Claude Opus 4.6 the per-1M rates are input=15, output=75, with anthropic
defaults cache_write=1.25, cache_read=0.10. For GPT-4o input=2.50, output=10,
openai defaults cache_write=1.00, cache_read=0.50.
"""

from pathlib import Path

import pytest

from tpasini.recordatio.proxy.pricing import lookup_cost

_TOL = 1e-6


def test_no_cache() -> None:
    # 1000*15 + 100*75 = 22500 / 1e6 = 0.0225
    cost = lookup_cost("claude-opus-4-6", 1000, 100, 0, 0)
    assert cost == pytest.approx(0.0225, abs=_TOL)


def test_all_cache_read() -> None:
    # 1000 * 15 * 0.10 = 1500 / 1e6 = 0.0015
    cost = lookup_cost("claude-opus-4-6", 1000, 0, 0, 1000)
    assert cost == pytest.approx(0.0015, abs=_TOL)


def test_all_cache_write() -> None:
    # 1000 * 15 * 1.25 = 18750 / 1e6 = 0.01875
    cost = lookup_cost("claude-opus-4-6", 1000, 0, 1000, 0)
    assert cost == pytest.approx(0.01875, abs=_TOL)


def test_mixed() -> None:
    # uncached = 1000 - 200 - 600 = 200
    # 200*15 + 200*15*1.25 + 600*15*0.10 + 100*75
    # = 3000 + 3750 + 900 + 7500 = 15150 / 1e6 = 0.01515
    cost = lookup_cost("claude-opus-4-6", 1000, 100, 200, 600)
    assert cost == pytest.approx(0.01515, abs=_TOL)


def test_openai_cached() -> None:
    # uncached = 1000 - 0 - 400 = 600
    # 600*2.50 + 400*2.50*0.50 + 100*10
    # = 1500 + 500 + 1000 = 3000 / 1e6 = 0.003
    cost = lookup_cost("gpt-4o", 1000, 100, 0, 400)
    assert cost == pytest.approx(0.003, abs=_TOL)


def test_unknown_model_returns_none() -> None:
    assert lookup_cost("not-a-real-model", 1000, 100) is None
    assert lookup_cost("not-a-real-model", 1000, 100, 50, 50) is None


def test_inputs_smaller_than_cache_clamps_to_zero() -> None:
    # input=100 < cache_creation+cache_read=300 → uncached clamps to 0,
    # so input_cost = 0; only cache and output costs contribute.
    # 200*15*1.25 + 100*15*0.10 + 0*75 = 3750 + 150 + 0 = 3900 / 1e6 = 0.0039
    cost = lookup_cost("claude-opus-4-6", 100, 0, 200, 100)
    assert cost == pytest.approx(0.0039, abs=_TOL)


def test_pricing_module_has_no_future_annotations() -> None:
    """Regression guard: CLAUDE.md forbids ``from __future__ import annotations``."""
    src = (
        Path(__file__).resolve().parent.parent
        / "src" / "tpasini" / "recordatio" / "proxy" / "pricing.py"
    ).read_text()
    assert "from __future__" not in src
