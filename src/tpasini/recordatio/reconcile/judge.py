"""The reconcile judge: a structured verdict on one pair of memories, plus the
deterministic checks that must pass before that verdict may delete anything.

The judge NEVER writes memory text. It sees two stored units (labelled A/B)
and answers with a relation, which one to keep, and verbatim quotes from
both units as evidence. Code then verifies the quotes against the stored
content — a verdict whose evidence doesn't check out keeps both units.

Contract (JSON object, nothing else)::

    {
      "relation": "duplicate_subsumed" | "supersedes" | "compatible" | "unrelated" | "unsure",
      "keep": "A" | "B" | null,
      "drop": "A" | "B" | null,
      "facts": [  # one entry per fact of the DROPPED unit
        {"dropped_fact": "<verbatim span of drop>",
         "kept_span": "<verbatim span of keep>",
         "status": "same" | "replaced"}
      ],
      "evidence": "<one sentence>"
    }

Verification (``verify_verdict``):
- quotes must be verbatim (case/whitespace-insensitive) spans of their unit;
- the dropped-fact quotes must together cover every content word of the
  dropped unit — an uncovered word means an unaccounted fact, so keep both;
- ``duplicate_subsumed``: every fact is ``same``; each kept span shares a
  content word and every number with its dropped fact; episodic units must
  describe the same day (two identical events on different days are two
  memories);
- ``supersedes``: at least one fact ``replaced`` (kept and dropped claims
  differ), the rest ``same``; the kept unit must be strictly newer by
  (valid_from, asserted_at).
Anything else — unrelated, compatible, unsure, a malformed reply, a timeout —
deletes nothing.
"""
import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

from tpasini.recordatio.models import KnowledgeType, KnowledgeUnit
from tpasini.recordatio.retrieval_text import content_words, normalize_numbers

logger = logging.getLogger(__name__)

type Relation = Literal["duplicate_subsumed", "supersedes", "compatible", "unrelated", "unsure"]
DELETING_RELATIONS: frozenset[str] = frozenset({"duplicate_subsumed", "supersedes"})


class JudgeFact(BaseModel):
    model_config = ConfigDict(extra="ignore")

    dropped_fact: str
    kept_span: str
    status: Literal["same", "replaced"]


class JudgeVerdict(BaseModel):
    model_config = ConfigDict(extra="ignore")

    relation: Relation
    keep: Literal["A", "B"] | None = None
    drop: Literal["A", "B"] | None = None
    facts: list[JudgeFact] = []
    evidence: str = ""


@dataclass
class JudgeCall:
    """One judge invocation: the parsed verdict (``None`` on any failure)."""

    verdict: JudgeVerdict | None
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None


class PairJudge(Protocol):
    async def judge(self, a: KnowledgeUnit, b: KnowledgeUnit) -> JudgeCall: ...


JUDGE_SYSTEM_PROMPT = """\
You compare two memories (A and B) stored for the same user and decide whether \
one of them is redundant. You never write, merge, fix or summarise memory text: \
you only classify the pair and quote the stored text verbatim.

Relations:
- "duplicate_subsumed": every fact in one memory is also stated by the other. \
The one to DROP is the one whose facts are all contained in the one to KEEP. \
If the would-be-dropped memory has ANY detail the other lacks (a name, number, \
date, place, condition, reason, qualifier), it is NOT subsumed — answer "compatible".
- "supersedes": both memories make a claim about the same thing (same subject, \
same attribute) and the claims conflict, so the newer one replaces the older one. \
KEEP the memory with the later valid_from/asserted_at, DROP the older one. Every \
other fact of the dropped memory must be stated by the kept one too.
- "compatible": related but both carry information worth keeping (different \
facts, additional detail, or claims that can both be true).
- "unrelated": about different things.
- "unsure": anything you are not certain about.

Episodic memories describe events. The same kind of event on different days is \
two different events: "compatible", never a duplicate.

For "duplicate_subsumed" and "supersedes" list, in "facts", EVERY fact of the \
dropped memory: "dropped_fact" is an exact substring copied from the dropped \
memory, "kept_span" is an exact substring copied from the kept memory that states \
the same fact ("status": "same") or the conflicting newer claim ("status": \
"replaced"). Together the dropped_fact quotes must cover the whole dropped memory.

When in doubt, answer "unsure". Deleting a memory that held something unique is \
far worse than keeping a duplicate.

Reply with ONLY a JSON object:
{"relation": "...", "keep": "A"|"B"|null, "drop": "A"|"B"|null, \
"facts": [{"dropped_fact": "...", "kept_span": "...", "status": "same"|"replaced"}], \
"evidence": "one sentence"}
"""


def render_pair(a: KnowledgeUnit, b: KnowledgeUnit) -> str:
    def block(label: str, u: KnowledgeUnit) -> str:
        return (
            f"Memory {label}\n"
            f"type: {u.type.value}\n"
            f"valid_from: {u.valid_from.isoformat()}\n"
            f"asserted_at: {u.asserted_at.isoformat()}\n"
            f"content: {u.content}"
        )

    return f"{block('A', a)}\n\n{block('B', b)}"


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_verdict(raw: str) -> JudgeVerdict:
    """Parse the judge's reply; tolerates code fences / prose around the object.

    Raises:
        ValueError: no JSON object, or one that doesn't fit the contract.
    """
    match = _JSON_OBJECT_RE.search(raw or "")
    if match is None:
        raise ValueError("judge reply has no JSON object")
    try:
        return JudgeVerdict.model_validate(json.loads(match.group(0)))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"judge reply does not fit the verdict contract: {exc}") from exc


class LLMJudge:
    """Judge backed by the same pontifex gateway the extractor uses."""

    def __init__(
        self,
        *,
        model: str,
        backend: str,
        api_key: str | None,
        base_url: str | None,
        max_tokens: int,
        timeout_s: float,
    ) -> None:
        self.model = model
        self._backend = backend
        self._max_tokens = max_tokens
        self._timeout_s = timeout_s
        if backend == "codex":
            from pontifex.codex import CodexSession
            from pontifex.codex_completion import CodexCompletion

            from tpasini.recordatio.agents.codex_client import CodexClient

            self._gw = CodexCompletion(model=model, session=CodexSession(client_factory=CodexClient))
        else:
            from pontifex import Pontifex

            self._gw = Pontifex(backend=backend, model=model, api_key=api_key, base_url=base_url)

    async def _stream(self, rendered: str) -> tuple[str, int, int]:
        from tpasini.recordatio.extraction.extractor import _estimate_tokens, _ExtractionStreamChunk

        chunks: list[str] = []
        usage = None
        extra_body = {"stream_options": {"include_usage": True}} if self._backend == "openai" else None
        async for chunk in self._gw.stream(
            messages=[{"role": "user", "content": rendered}],
            system=JUDGE_SYSTEM_PROMPT,
            max_tokens=self._max_tokens,
            extra_body=extra_body,
        ):
            parsed = _ExtractionStreamChunk.model_validate(chunk)
            for choice in parsed.choices:
                if choice.delta.content:
                    chunks.append(choice.delta.content)
            if parsed.usage is not None:
                usage = parsed.usage
        raw = "".join(chunks)
        if usage is not None:
            input_tokens, output_tokens, _cw, _cr = usage.token_counts()
        else:
            input_tokens = _estimate_tokens(rendered) + _estimate_tokens(JUDGE_SYSTEM_PROMPT)
            output_tokens = _estimate_tokens(raw)
        return raw, input_tokens, output_tokens

    async def judge(self, a: KnowledgeUnit, b: KnowledgeUnit) -> JudgeCall:
        from tpasini.recordatio.proxy.pricing import lookup_cost

        try:
            raw, input_tokens, output_tokens = await asyncio.wait_for(
                self._stream(render_pair(a, b)), timeout=self._timeout_s
            )
        except TimeoutError:
            return JudgeCall(verdict=None, error=f"judge timed out after {self._timeout_s}s")
        except Exception as exc:  # transport/auth/rate-limit: keep both, retry next night
            logger.warning("reconcile judge call failed: %s", exc)
            return JudgeCall(verdict=None, error=f"{exc.__class__.__name__}: {exc}")
        cost = lookup_cost(self.model, input_tokens, output_tokens)
        try:
            verdict = parse_verdict(raw)
        except ValueError as exc:
            return JudgeCall(
                verdict=None, error=str(exc), input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost
            )
        return JudgeCall(verdict=verdict, input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost)


# ---------------------------------------------------------------------------
# Deterministic verification
# ---------------------------------------------------------------------------


def _norm(text: str) -> str:
    return " ".join(text.casefold().split())


def _quote_in(quote: str, content: str) -> bool:
    q = _norm(quote).strip(" .;,")
    return bool(q) and q in _norm(content)


_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


def _numbers(text: str) -> set[str]:
    return set(_NUMBER_RE.findall(normalize_numbers(text)))


def recency(unit: KnowledgeUnit) -> tuple[datetime, datetime]:
    """``(valid_from, asserted_at)`` as UTC-aware datetimes (legacy rows may be
    naive; they are UTC by convention) so units always compare."""

    def aware(dt: datetime) -> datetime:
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)

    return aware(unit.valid_from), aware(unit.asserted_at)


@dataclass(frozen=True)
class VerifiedDrop:
    """A verdict that passed every check: ``drop`` may be deleted, ``keep`` stays verbatim."""

    keep: KnowledgeUnit
    drop: KnowledgeUnit
    relation: str
    evidence: dict[str, object] = field(default_factory=dict)


def verify_verdict(
    verdict: JudgeVerdict, a: KnowledgeUnit, b: KnowledgeUnit
) -> tuple[VerifiedDrop | None, str]:
    """Return ``(VerifiedDrop, "")`` when the verdict may delete, else
    ``(None, reason)`` — the reason is why both units are kept."""
    if verdict.relation not in DELETING_RELATIONS:
        return None, verdict.relation
    units = {"A": a, "B": b}
    if verdict.keep is None or verdict.drop is None or verdict.keep == verdict.drop:
        return None, "rejected: keep/drop must name the two different memories"
    keep, drop = units[verdict.keep], units[verdict.drop]
    if not verdict.facts:
        return None, "rejected: no facts quoted"

    covered: set[str] = set()
    replaced = 0
    for fact in verdict.facts:
        if not _quote_in(fact.dropped_fact, drop.content):
            return None, "rejected: dropped_fact is not a verbatim span of the dropped memory"
        if not _quote_in(fact.kept_span, keep.content):
            return None, "rejected: kept_span is not a verbatim span of the kept memory"
        covered |= content_words(fact.dropped_fact)
        if fact.status == "same":
            if not content_words(fact.dropped_fact) & content_words(fact.kept_span):
                return None, "rejected: a 'same' fact shares no content word with its kept span"
            if not _numbers(fact.dropped_fact) <= _numbers(fact.kept_span):
                return None, "rejected: a 'same' fact has a number the kept span lacks"
        else:
            if _norm(fact.dropped_fact) == _norm(fact.kept_span):
                return None, "rejected: a 'replaced' claim is identical to its replacement"
            replaced += 1

    uncovered = content_words(drop.content) - covered
    if uncovered:
        return None, f"rejected: {len(uncovered)} word(s) of the dropped memory not accounted for by any fact"

    if verdict.relation == "duplicate_subsumed":
        if replaced:
            return None, "rejected: duplicate_subsumed with a 'replaced' fact"
        if (
            KnowledgeType.EPISODIC in (keep.type, drop.type)
            and recency(keep)[0].date() != recency(drop)[0].date()
        ):
            return None, "rejected: episodic memories from different days are distinct events"
    else:  # supersedes
        if not replaced:
            return None, "rejected: supersedes without a replaced claim"
        if recency(keep) <= recency(drop):
            return None, "rejected: the kept memory is not newer than the dropped one"

    evidence: dict[str, object] = {
        "facts": [f.model_dump() for f in verdict.facts],
        "judge_evidence": verdict.evidence,
    }
    return VerifiedDrop(keep=keep, drop=drop, relation=verdict.relation, evidence=evidence), ""


__all__ = [
    "DELETING_RELATIONS",
    "JUDGE_SYSTEM_PROMPT",
    "JudgeCall",
    "JudgeFact",
    "JudgeVerdict",
    "LLMJudge",
    "PairJudge",
    "VerifiedDrop",
    "parse_verdict",
    "recency",
    "render_pair",
    "verify_verdict",
]
