"""The reconcile verdict contract and the deterministic checks that must pass
before a verdict may delete anything.

Recordatio never calls a model for this. The judge is an agent (the nightly
Cardea routine, or the ``agentmem reconcile`` harness) that sees two stored
units labelled A/B and answers with a structured verdict quoting both units
verbatim. It never writes memory text. Code then verifies the quotes against
the stored content — a verdict whose evidence doesn't check out keeps both.

Contract (JSON object)::

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
Anything else — unrelated, compatible, unsure, a malformed verdict — deletes
nothing.
"""
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from tpasini.recordatio.models import KnowledgeType, KnowledgeUnit
from tpasini.recordatio.retrieval_text import content_words, normalize_numbers

type Relation = Literal["duplicate_subsumed", "supersedes", "compatible", "unrelated", "unsure"]
DELETING_RELATIONS: frozenset[str] = frozenset({"duplicate_subsumed", "supersedes"})


class JudgeFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dropped_fact: str
    kept_span: str
    status: Literal["same", "replaced"]


class JudgeVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relation: Relation
    keep: Literal["A", "B"] | None = None
    drop: Literal["A", "B"] | None = None
    facts: list[JudgeFact] = []
    evidence: str = ""


_JSON_RE = re.compile(r"[\[{].*[\]}]", re.DOTALL)


def extract_json(raw: str) -> object:
    """The outermost JSON object/array in model text (tolerates fences and prose).

    Raises:
        ValueError: no parseable JSON.
    """
    match = _JSON_RE.search(raw or "")
    if match is None:
        raise ValueError("no JSON in the reply")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(f"reply is not valid JSON: {exc}") from exc


def parse_verdict(raw: object) -> JudgeVerdict:
    """Validate a verdict (a dict, or model text holding one).

    Raises:
        ValueError: it doesn't fit the contract.
    """
    data = extract_json(raw) if isinstance(raw, str) else raw
    try:
        return JudgeVerdict.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"verdict does not fit the contract: {exc}") from exc


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
    "JudgeFact",
    "JudgeVerdict",
    "VerifiedDrop",
    "extract_json",
    "parse_verdict",
    "recency",
    "verify_verdict",
]
