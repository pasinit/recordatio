"""Zero-shot entity-identity judgment (#190).

Replaces the hand-tuned ``canonical_dedup_key`` — which encoded Springfield/postcode
specific rules — with two GENERAL signals that carry no domain tailoring:

  1. **Wikidata grounding.** Each surface label is best-effort linked to a
     Wikidata QID + one-line description. A confident link gives the judge
     real-world context (``EU`` = the political union; ``EU institutions`` = the
     bodies *of* the union) and, when two labels link to two *different*
     high-confidence QIDs, vetoes a merge outright — cheap precision, no LLM
     call.
  2. **A zero-shot LLM judge.** The workhorse. Given the two labels, their type,
     and any Wikidata descriptions, it answers "same real-world entity?" against
     a *general* rubric (qualifier-variant vs content-distinct), never the
     specific examples.

Composition is precision-first: a Wikidata veto wins; otherwise the LLM decides;
absent any usable signal (offline, no LLM, Wikidata down) the judge **abstains**
(``UNKNOWN``) so callers never merge on no evidence. Verdicts and Wikidata links
are cached, so the judge is a pure function of its inputs given a fixed backend.

The judge is deliberately ignorant of *how* candidate pairs are chosen — blocking
(who to compare) lives in the caller; the judge only answers *are these two the
same*. That separation is what lets the same judge serve the batch merge
migration and the extraction-time resolver.
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import threading
import time
import unicodedata
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import StrEnum
from functools import lru_cache
from typing import Protocol

import httpx

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[^\w]+")
_WIKIDATA_API = "https://www.wikidata.org/w/api.php"


def _norm(value: str | None) -> str:
    """Casefold + NFKC + collapse non-word runs — the resolver's normal form."""
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    return _WORD_RE.sub(" ", normalized).strip()


class IdentityRelation(StrEnum):
    SAME = "same"
    DIFFERENT = "different"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class EntityRef:
    """A surface form to be judged. ``type`` and ``aliases`` are context only;
    ``wikidata_id`` is used when already known (skips a lookup)."""

    label: str
    type: str | None = None
    aliases: tuple[str, ...] = ()
    wikidata_id: str | None = None


@dataclass(frozen=True)
class IdentityVerdict:
    relation: IdentityRelation
    confidence: float
    reason: str
    source: str  # "wikidata" | "llm" | "abstain"

    @property
    def same(self) -> bool:
        return self.relation is IdentityRelation.SAME


_ABSTAIN = IdentityVerdict(IdentityRelation.UNKNOWN, 0.0, "no usable signal", "abstain")


class EntityIdentityJudge(Protocol):
    """The minimal surface the entity resolver depends on: decide whether two
    surface forms are the same real-world entity. ``CompositeIdentityJudge``
    implements it; tests can inject a stub."""

    def same_entity(self, a: EntityRef, b: EntityRef) -> IdentityVerdict: ...


# ---------------------------------------------------------------------------
# LLM transport
# ---------------------------------------------------------------------------


class LLMBackend(Protocol):
    """A synchronous text-in/text-out completion. The judge owns prompt + parse;
    a backend only ships a string and returns one."""

    def complete(self, prompt: str) -> str: ...


class StubBackend:
    """Deterministic backend for tests: a callable that maps prompt -> raw text.

    Lets a suite assert the judge's parsing/composition without a live model.
    """

    def __init__(self, responder: Callable[[str], str]) -> None:
        self._responder = responder

    def complete(self, prompt: str) -> str:
        return self._responder(prompt)


class ClaudeCLIBackend:
    """Subscription-OAuth ``claude -p`` shell-out.

    The repo's ambient ``ANTHROPIC_API_KEY`` is a dead/low-balance key (it
    shadows the working subscription auth), so we strip it from the child env —
    the same lesson the dataset generator learned. JSON output mode wraps the
    model's text in a ``result`` field, sometimes fenced; we unwrap both.
    """

    def __init__(self, model: str = "claude-haiku-4-5-20251001", timeout: float = 120.0) -> None:
        self.model = model
        self.timeout = timeout

    def complete(self, prompt: str) -> str:
        child_env = {k: v for k, v in os.environ.items() if k not in {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"}}
        proc = subprocess.run(
            ["claude", "-p", "--output-format", "json", "--model", self.model, prompt],
            capture_output=True,
            text=True,
            timeout=self.timeout,
            env=child_env,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"claude -p failed ({proc.returncode}): {proc.stderr.strip()[:300]}")
        outer = json.loads(proc.stdout)
        if outer.get("is_error"):
            raise RuntimeError(f"claude -p error: {outer.get('result', '')[:300]}")
        return str(outer.get("result", ""))



class CodexCLIBackend:
    """Subscription-authenticated Codex CLI completion for the identity judge."""

    def __init__(self, model: str = "", timeout: float = 120.0) -> None:
        self.model = model
        self.timeout = timeout

    def complete(self, prompt: str) -> str:
        from concurrent.futures import ThreadPoolExecutor

        from tpasini.recordatio.agents.codex_transport import complete_text

        def run() -> str:
            return asyncio.run(complete_text(prompt, self.model, self.timeout))

        # The existing resolver contract is synchronous, including its use
        # inside async extraction persistence. Own the CLI loop in a worker
        # when the caller already has an event loop; never nest asyncio.run.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return run()
        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(run).result()


def cli_backend(provider: str = "claude", *, model: str | None = None, timeout: float = 120.0) -> LLMBackend:
    """Select a judge CLI without changing the original Claude defaults."""
    if provider == "claude":
        return ClaudeCLIBackend(model=model if model is not None else "claude-haiku-4-5-20251001", timeout=timeout)
    if provider == "codex":
        return CodexCLIBackend(model=model or "", timeout=timeout)
    raise ValueError(f"Unknown entity judge provider: {provider!r}; choose claude or codex")


def _extract_json_object(text: str) -> dict:
    """Pull the first ``{...}`` JSON object out of a model reply (handles ```json
    fences and leading prose). Raises ValueError if none parses."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"no JSON object in reply: {text[:200]!r}")
    return json.loads(text[start : end + 1])


def _extract_json_array(text: str) -> list[dict]:
    """Pull the first ``[...]`` JSON array out of a model reply."""
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"no JSON array in reply: {text[:200]!r}")
    return json.loads(text[start : end + 1])


# ---------------------------------------------------------------------------
# Wikidata grounding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WikidataLink:
    qid: str
    label: str
    description: str
    confident: bool


class WikidataLinker:
    """Best-effort label -> (QID, description). A link is *confident* when the
    returned entity's own label closely matches the query (so we don't trust a
    fuzzy top hit for a misspelt or qualifier-laden surface). Cached by
    normalized label; failures cache as ``None`` so a flaky network never blocks
    the judge twice."""

    def __init__(
        self,
        *,
        timeout: float = 5.0,
        confident_ratio: float = 0.9,
        min_interval: float = 0.25,
        max_retries: int = 3,
    ) -> None:
        self.timeout = timeout
        self.confident_ratio = confident_ratio
        self.min_interval = min_interval  # serialize requests ~4/s (anon WDQS limit)
        self.max_retries = max_retries
        self._cache: dict[str, WikidataLink | None] = {}
        self._lock = threading.Lock()
        self._last_request = 0.0

    def link(self, label: str) -> WikidataLink | None:
        key = _norm(label)
        if not key:
            return None
        with self._lock:  # cache + throttle are shared across judge worker threads
            if key in self._cache:
                return self._cache[key]
            link = self._lookup(label, key)
            self._cache[key] = link
            return link

    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def _lookup(self, label: str, key: str) -> WikidataLink | None:
        params = {
            "action": "wbsearchentities",
            "search": label,
            "language": "en",
            "format": "json",
            "limit": "1",
        }
        headers = {"User-Agent": "recordatio/0.1 (https://github.com/pasinit/recordatio)"}
        results = None
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                with httpx.Client(timeout=self.timeout, headers=headers) as client:
                    resp = client.get(_WIKIDATA_API, params=params)
                if resp.status_code == 429:  # rate-limited — exponential backoff
                    time.sleep(self.min_interval * (2 ** (attempt + 2)))
                    continue
                resp.raise_for_status()
                results = resp.json().get("search") or []
                break
            except Exception as exc:  # network / parse — degrade to "no link"
                logger.warning("wikidata link failed for %r: %s", label, exc)
                return None
        if results is None:  # exhausted retries on 429
            logger.warning("wikidata rate-limited for %r after %d tries", label, self.max_retries)
            return None
        if not results:
            return None
        top = results[0]
        hit_label = top.get("label") or ""
        ratio = SequenceMatcher(a=key, b=_norm(hit_label)).ratio()
        return WikidataLink(
            qid=top.get("id") or "",
            label=hit_label,
            description=top.get("description") or "",
            confident=ratio >= self.confident_ratio,
        )


# ---------------------------------------------------------------------------
# The judge
# ---------------------------------------------------------------------------


_SYSTEM_RUBRIC = (
    "You decide whether two short surface labels denote the SAME real-world entity.\n"
    "They are the SAME entity when one merely adds a redundant LOCATING, TEMPORAL, "
    "or DISAMBIGUATING qualifier to the other — a postal code, a containing place "
    "or region, a parenthetical gloss, a date, or a punctuation/spelling variant — "
    "so both still point at one and the same thing.\n"
    "They are DIFFERENT when the extra words name a DISTINCT thing: a sub-part or "
    "sub-organisation, a derived or named-after work, a related-but-separate "
    "concept, or simply two different entities that happen to share a prefix.\n"
    "When you are not sure, answer DIFFERENT — a wrong merge silently corrupts a "
    "person's memory, while a missed merge is harmless.\n"
    'Reply with ONLY a JSON object: {"same": <true|false>, "confidence": <0..1>, '
    '"reason": "<at most 12 words>"}.'
)

_BATCH_RUBRIC = (
    _SYSTEM_RUBRIC.rsplit("\n", 1)[0]
    + "\n"
    + "Below is a numbered list of label pairs (A vs B). Judge EACH independently.\n"
    + 'Reply with ONLY a JSON array, one object per pair: '
    + '[{"n": <pair number>, "same": <true|false>, "confidence": <0..1>, "reason": "<≤12 words>"}].'
)


@dataclass
class CompositeIdentityJudge:
    """Wikidata-grounded zero-shot identity judge.

    ``llm`` is required to ever return SAME (Wikidata alone only *vetoes*). With
    no backend the judge still runs but can only veto or abstain — safe for
    offline/dry contexts. ``min_confidence`` is the floor below which an LLM
    SAME is downgraded to UNKNOWN (precision guard)."""

    llm: LLMBackend | None = None
    wikidata: WikidataLinker | None = None
    min_confidence: float = 0.6
    _cache: dict[tuple[str, str, str | None], IdentityVerdict] = field(default_factory=dict, repr=False)

    def same_entity(self, a: EntityRef, b: EntityRef) -> IdentityVerdict:
        local = self._local_only(a, b)
        if local is not None:
            return local
        link_a, link_b = self._link(a), self._link(b)
        veto = self._wikidata_veto(link_a, link_b)
        verdict = veto if veto is not None else (
            self._llm_verdict(a, b, link_a, link_b) if self.llm is not None else _ABSTAIN
        )
        self._cache[self._ckey(a, b)] = verdict
        return verdict

    def same_entity_batch(
        self, pairs: list[tuple[EntityRef, EntityRef]], *, batch_size: int = 24, workers: int = 8
    ) -> list[IdentityVerdict]:
        """Judge many pairs, packing several per LLM call. Trivial-same, cache
        hits, and Wikidata vetoes are resolved with no model call; the remainder
        are chunked and the chunks judged concurrently. Returns one verdict per
        input pair, in order."""
        results: list[IdentityVerdict | None] = [None] * len(pairs)
        to_llm: list[tuple[int, EntityRef, EntityRef, WikidataLink | None, WikidataLink | None]] = []
        for i, (a, b) in enumerate(pairs):
            local = self._local_only(a, b)
            if local is not None:
                results[i] = local
                continue
            link_a, link_b = self._link(a), self._link(b)
            veto = self._wikidata_veto(link_a, link_b)
            if veto is not None:
                results[i] = self._cache[self._ckey(a, b)] = veto
            elif self.llm is None:
                results[i] = _ABSTAIN
            else:
                to_llm.append((i, a, b, link_a, link_b))

        chunks = [to_llm[s : s + batch_size] for s in range(0, len(to_llm), batch_size)]
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for chunk, verdicts in zip(chunks, pool.map(self._llm_batch, chunks), strict=True):
                for (i, a, b, _la, _lb), v in zip(chunk, verdicts, strict=True):
                    results[i] = self._cache[self._ckey(a, b)] = v
        return [v if v is not None else _ABSTAIN for v in results]

    def _ckey(self, a: EntityRef, b: EntityRef) -> tuple[str, str, str | None]:
        ka, kb = _norm(a.label), _norm(b.label)
        return (min(ka, kb), max(ka, kb), a.type if a.type == b.type else None)

    def _local_only(self, a: EntityRef, b: EntityRef) -> IdentityVerdict | None:
        """Resolution needing no network/model: identical normalized label, or a
        cached prior verdict. Returns None when a real judgment is required."""
        ka, kb = _norm(a.label), _norm(b.label)
        if ka and ka == kb:
            return IdentityVerdict(IdentityRelation.SAME, 1.0, "identical normalized label", "trivial")
        return self._cache.get(self._ckey(a, b))

    def _link(self, ref: EntityRef) -> WikidataLink | None:
        if self.wikidata is None:
            return None
        return self.wikidata.link(ref.label)

    @staticmethod
    def _wikidata_veto(link_a: WikidataLink | None, link_b: WikidataLink | None) -> IdentityVerdict | None:
        """Two confident links to DIFFERENT QIDs ⇒ different entities, no LLM
        needed. (Confident links to the *same* QID are left to the LLM as well,
        because a shared QID can still be a coarse match; we only use Wikidata to
        rule merges OUT, never to force them in.)"""
        if (
            link_a is not None
            and link_b is not None
            and link_a.confident
            and link_b.confident
            and link_a.qid
            and link_b.qid
            and link_a.qid != link_b.qid
        ):
            return IdentityVerdict(
                IdentityRelation.DIFFERENT,
                0.95,
                f"distinct Wikidata entities ({link_a.qid} ≠ {link_b.qid})",
                "wikidata",
            )
        return None

    def _llm_verdict(
        self, a: EntityRef, b: EntityRef, link_a: WikidataLink | None, link_b: WikidataLink | None
    ) -> IdentityVerdict:
        prompt = f"{_SYSTEM_RUBRIC}\n\n{_render_pair(a, b, link_a, link_b)}"
        try:
            obj = _extract_json_object(self.llm.complete(prompt))
        except Exception as exc:
            logger.warning("identity judge LLM failed for %r/%r: %s", a.label, b.label, exc)
            return _ABSTAIN
        return self._verdict_from_obj(obj)

    def _llm_batch(
        self,
        items: list[tuple[int, EntityRef, EntityRef, WikidataLink | None, WikidataLink | None]],
    ) -> list[IdentityVerdict]:
        """One LLM call for a chunk of pairs; verdicts mapped back by ordinal so a
        dropped/reordered row degrades to abstain rather than misaligning."""
        lines = [
            f"{n}. A: {_render_ref(a, la)}  ||  B: {_render_ref(b, lb)}"
            for n, (_i, a, b, la, lb) in enumerate(items, 1)
        ]
        prompt = f"{_BATCH_RUBRIC}\n\n" + "\n".join(lines)
        try:
            arr = _extract_json_array(self.llm.complete(prompt))
        except Exception as exc:
            logger.warning("identity judge batch failed (%d pairs): %s", len(items), exc)
            return [_ABSTAIN] * len(items)
        by_n: dict[int, dict] = {}
        for obj in arr:
            try:
                by_n[int(obj["n"])] = obj
            except (KeyError, TypeError, ValueError):
                continue
        return [self._verdict_from_obj(by_n[n]) if n in by_n else _ABSTAIN for n in range(1, len(items) + 1)]

    def _verdict_from_obj(self, obj: dict) -> IdentityVerdict:
        same = bool(obj.get("same"))
        confidence = float(obj.get("confidence", 0.0) or 0.0)
        reason = str(obj.get("reason", ""))[:120]
        if same and confidence < self.min_confidence:
            return IdentityVerdict(IdentityRelation.UNKNOWN, confidence, f"low-confidence same: {reason}", "llm")
        relation = IdentityRelation.SAME if same else IdentityRelation.DIFFERENT
        return IdentityVerdict(relation, confidence, reason, "llm")


def _render_pair(
    a: EntityRef, b: EntityRef, link_a: WikidataLink | None, link_b: WikidataLink | None
) -> str:
    return (
        f"Label A: {_render_ref(a, link_a)}\n"
        f"Label B: {_render_ref(b, link_b)}\n"
        "Same real-world entity?"
    )


def _render_ref(ref: EntityRef, link: WikidataLink | None) -> str:
    parts = [f'"{ref.label}"']
    if ref.type:
        parts.append(f"(type: {ref.type})")
    if link is not None and link.confident and (link.label or link.description):
        gloss = " — ".join(p for p in (link.label, link.description) if p)
        parts.append(f"[Wikidata: {gloss}]")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Runtime wiring
# ---------------------------------------------------------------------------

_ENV_FLAG = "RECORDATIO_ENTITY_JUDGE"


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def judge_from_env() -> "CompositeIdentityJudge | None":
    """The resolver's runtime judge, constructed from env — or ``None`` (disabled).

    OFF by default: a judge is built only when ``RECORDATIO_ENTITY_JUDGE`` is truthy,
    so putting an LLM call into every extraction is a deliberate opt-in, never a
    silent default. ``RECORDATIO_ENTITY_JUDGE_PROVIDER`` selects ``claude`` (default)
    or ``codex``; ``RECORDATIO_ENTITY_JUDGE_MODEL`` overrides the model and
    ``RECORDATIO_ENTITY_JUDGE_WIKIDATA`` adds grounding.

    CAVEAT: the CLI backend shells out and BLOCKS; an async-native transport for
    the in-proxy path is a follow-up before enabling this under load.
    """
    if not _truthy(os.environ.get(_ENV_FLAG)):
        return None
    model = os.environ.get("RECORDATIO_ENTITY_JUDGE_MODEL")
    provider = os.environ.get("RECORDATIO_ENTITY_JUDGE_PROVIDER", "claude")
    wikidata = WikidataLinker() if _truthy(os.environ.get("RECORDATIO_ENTITY_JUDGE_WIKIDATA")) else None
    return CompositeIdentityJudge(llm=cli_backend(provider, model=model), wikidata=wikidata)
