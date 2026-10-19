"""Shared pricing table and cost lookup for upstream LLM calls.

Used by both the chat-completions path and the background extraction path
so that estimated costs in usage events and aggregated extraction summaries
are derived from the same source of truth.

Prices are in USD per 1M tokens. Each model entry carries an ``input`` /
``output`` per-1M price plus optional ``cache_write`` / ``cache_read``
multipliers applied to the input rate. Where the per-model multipliers are
absent, provider-family defaults from ``_PROVIDER_DEFAULTS`` apply.
"""

import logging
from decimal import Decimal
from typing import TypedDict

logger = logging.getLogger(__name__)

# Already-warned-about models so we log once per process, not per request.
_warned_unknown_models: set[str] = set()


class _ModelPricing(TypedDict, total=False):
    input: float          # USD per 1M uncached input tokens
    output: float         # USD per 1M output tokens
    cache_write: float    # multiplier on `input`
    cache_read: float     # multiplier on `input`


# Per-provider-family defaults applied when a model entry doesn't override
# its own cache multipliers. Anthropic 5-min ephemeral writes bill at 1.25x;
# OpenAI doesn't bill cache writes at a premium (1.0x); Gemini ditto.
_PROVIDER_DEFAULTS: dict[str, _ModelPricing] = {
    "anthropic": {"cache_write": 1.25, "cache_read": 0.10},
    "openai":    {"cache_write": 1.00, "cache_read": 0.50},
    "gemini":    {"cache_write": 1.00, "cache_read": 0.25},
}


_PRICE_TABLE: dict[str, _ModelPricing] = {
    # Anthropic
    "claude-opus-4-6":              {"input": 15.0, "output": 75.0},
    "claude-sonnet-4-6":            {"input": 3.0,  "output": 15.0},
    "claude-haiku-5-5":             {"input": 0.10, "output": 0.50},
    "claude-haiku-4-5-20251001":    {"input": 0.80, "output": 4.0},
    "claude-haiku-4-5":             {"input": 0.80, "output": 4.0},
    # OpenAI
    "gpt-4o":                       {"input": 2.50, "output": 10.0},
    "gpt-4o-mini":                  {"input": 0.15, "output": 0.60},
    "gpt-4.1":                      {"input": 2.0,  "output": 8.0},
    "gpt-4.1-mini":                 {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano":                 {"input": 0.10, "output": 0.40},
    "o3":                           {"input": 2.0,  "output": 8.0},
    "o3-mini":                      {"input": 1.10, "output": 4.40},
    "o4-mini":                      {"input": 1.10, "output": 4.40},
    # Gemini
    "gemini-2.5-flash":             {"input": 0.15, "output": 0.60},
    "gemini-2.5-pro":               {"input": 1.25, "output": 10.0},
    "gemini-2.0-flash":             {"input": 0.10, "output": 0.40},
}


def _provider_family(model: str) -> str:
    """Determine provider family from a model name.

    Unknown names fall back to the openai family but with a no-discount
    cache_read of 1.0 — see ``_resolve_cache_multipliers``.
    """
    name = model.lower()
    if name.startswith("claude-"):
        return "anthropic"
    if name.startswith("gpt-") or name.startswith("o3") or name.startswith("o4"):
        return "openai"
    if name.startswith("gemini-"):
        return "gemini"
    return "unknown"


def _resolve_cache_multipliers(model: str, entry: _ModelPricing) -> tuple[float, float]:
    """Return (cache_write_mult, cache_read_mult) for *model*.

    Per-model overrides win; otherwise the provider-family default applies;
    otherwise (genuinely unknown family) cache-write is 1.0 and cache-read
    is 1.0 (no discount, treat cached tokens as full input).
    """
    family = _provider_family(model)
    defaults = _PROVIDER_DEFAULTS.get(family, {"cache_write": 1.0, "cache_read": 1.0})
    write_mult = entry.get("cache_write", defaults.get("cache_write", 1.0))
    read_mult = entry.get("cache_read", defaults.get("cache_read", 1.0))
    return float(write_mult), float(read_mult)


def lookup_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float | None:
    """Return estimated cost in USD, or None if the model is not priced.

    ``input_tokens`` is the TOTAL prompt-token count as reported by the
    upstream (Anthropic ``input_tokens`` / OpenAI ``prompt_tokens``). The
    function subtracts ``cache_creation_tokens`` and ``cache_read_tokens``
    internally to derive the uncached remainder — both Anthropic and OpenAI
    report cached counts as a *breakdown* of prompt_tokens, not in addition.
    """
    # Strip a `provider/` routing prefix if present — `_resolve_model` does
    # this before constructing Pontifex, but defensive callers may still
    # pass the raw routed name.
    bare = model.split("/", 1)[1] if "/" in model else model
    entry = _PRICE_TABLE.get(bare)
    if entry is None:
        if model not in _warned_unknown_models:
            _warned_unknown_models.add(model)
            logger.warning(
                "no pricing entry for model %r — cost_usd will be omitted. "
                "Add it to proxy/pricing.py:_PRICE_TABLE.",
                model,
            )
        return None
    model = bare
    input_price = entry.get("input", 0.0)
    output_price = entry.get("output", 0.0)
    write_mult, read_mult = _resolve_cache_multipliers(model, entry)

    uncached = max(0, input_tokens - cache_creation_tokens - cache_read_tokens)
    input_cost = uncached * input_price
    write_cost = cache_creation_tokens * input_price * write_mult
    read_cost = cache_read_tokens * input_price * read_mult
    output_cost = output_tokens * output_price
    total = (input_cost + write_cost + read_cost + output_cost) / 1_000_000
    return float(Decimal(str(total)).quantize(Decimal("0.000001")))
