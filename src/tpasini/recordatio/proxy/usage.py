"""Token counting, provider tags, and extraction usage buffering."""

import asyncio
from collections import defaultdict
from dataclasses import dataclass

try:
    import tiktoken

    _tiktoken_available = True
except ImportError:
    _tiktoken_available = False


def count_tokens(text: str, model: str = "") -> int:
    """Count tokens in *text*. Uses tiktoken when available, else estimates."""
    if _tiktoken_available:
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    # Rough estimate: ~4 chars per token for English text.
    return max(1, len(text) // 4)


def parse_upstream_usage(usage: dict | None) -> tuple[int, int, int, int]:
    """Return ``(input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens)``."""
    if not usage:
        return (0, 0, 0, 0)

    def _int(value: object) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    input_tokens = _int(usage.get("prompt_tokens") or usage.get("input_tokens"))
    output_tokens = _int(usage.get("completion_tokens") or usage.get("output_tokens"))

    cache_creation = usage.get("cache_creation_input_tokens")
    if not cache_creation:
        nested = usage.get("cache_creation")
        if isinstance(nested, dict):
            cache_creation = nested.get("ephemeral_5m_input_tokens")
    cache_creation_tokens = _int(cache_creation)

    cache_read = usage.get("cache_read_input_tokens")
    if not cache_read:
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict):
            cache_read = details.get("cached_tokens")
    cache_read_tokens = _int(cache_read)

    return (input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens)


def count_messages_tokens(
    messages: list[dict],
    system: str | list[dict] | None,
    model: str = "",
) -> int:
    """Estimate input tokens from messages plus system prompt."""
    parts: list[str] = []
    if isinstance(system, str):
        parts.append(system)
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
    return count_tokens("\n".join(parts), model)


def provider_tag(backend: str, base_url: str | None) -> str:
    """Return the configured backend used for cost-event aggregation."""
    _ = base_url
    return backend


@dataclass(frozen=True)
class ExtractionUsage:
    """Usage and cost summary for one completed extraction call."""

    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0


class ExtractionUsageBuffer:
    """In-memory buffer used to piggyback completed extraction costs."""

    def __init__(self) -> None:
        self.pending: dict[str, list[ExtractionUsage]] = defaultdict(list)
        self.lock = asyncio.Lock()

    async def drain(self, agent_id: str | None) -> dict | None:
        """Return and clear buffered extraction usage for one agent."""
        if not agent_id:
            return None
        async with self.lock:
            pending = self.pending.pop(agent_id, [])
        if not pending:
            return None
        fragment: dict = {
            "cost_usd_extraction": sum(p.cost_usd for p in pending),
            "extraction_input_tokens": sum(p.input_tokens for p in pending),
            "extraction_output_tokens": sum(p.output_tokens for p in pending),
            "extraction_calls": len(pending),
            "extraction_model": pending[-1].model,
        }
        cache_creation_total = sum(p.cache_creation_tokens for p in pending)
        cache_read_total = sum(p.cache_read_tokens for p in pending)
        if cache_creation_total:
            fragment["extraction_cache_creation_tokens"] = cache_creation_total
        if cache_read_total:
            fragment["extraction_cache_read_tokens"] = cache_read_total
        return fragment
