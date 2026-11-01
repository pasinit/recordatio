"""Cheap, deterministic candidate finding for the reconcile job.

Per scope, three signals propose pairs; none of them decides anything on its
own except exact duplicates:

1. normalised exact match (casefold, whitespace + edge punctuation collapsed)
   — identical text, resolved without the judge;
2. embedding cosine over the vectors already stored in the ``.emb`` sidecar
   (no model load) — the focus units against every active unit;
3. FTS/BM25 neighbours of each focus unit's content words.

Pairs are always same-type and same-user. Across scopes, only the user's
shared scope (``{user}/_user``) pairs with each of that user's agent scopes,
and the shared copy is the one that may be kept (enforced again at apply).
Two agent scopes never pair with each other.
"""
import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass

from tpasini.recordatio.models import KnowledgeType, KnowledgeUnit
from tpasini.recordatio.reconcile.judge import recency
from tpasini.recordatio.retrieval_text import content_words
from tpasini.recordatio.storage.base import SQLiteStore

_EDGE_PUNCT_RE = re.compile(r"^[\s\"'`.,;:!?()\[\]-]+|[\s\"'`.,;:!?()\[\]-]+$")


def normalise_content(text: str) -> str:
    return _EDGE_PUNCT_RE.sub("", " ".join(text.casefold().split()))


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pair_key(a: KnowledgeUnit, b: KnowledgeUnit) -> str:
    """Stable key for a judged pair: both ids + both contents, order-free.
    A content change on either side invalidates any cached verdict."""
    parts = sorted(f"{u.id}:{content_hash(u.content)}" for u in (a, b))
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CandidatePair:
    a: KnowledgeUnit
    b: KnowledgeUnit
    signal: str  # "embedding" | "fts"
    score: float


@dataclass(frozen=True)
class ExactDuplicate:
    keep: KnowledgeUnit
    keep_scope: str
    drop: KnowledgeUnit
    drop_scope: str


def _exact_keep_order(u: KnowledgeUnit) -> tuple[int, object, str]:
    # Most-reinforced first, then the earliest assertion, then id — so the
    # surviving row is the one retrieval has leaned on most.
    return (-(u.extraction_count + u.retrieval_count), recency(u)[1], u.id)


def exact_key(u: KnowledgeUnit) -> tuple[str, str, str] | None:
    """Grouping key for identical text. Episodic units only group within the
    same valid_from day (the same event recorded twice); the same text on
    another day is another event."""
    norm = normalise_content(u.content)
    if not norm:
        return None
    day = recency(u)[0].date().isoformat() if u.type is KnowledgeType.EPISODIC else ""
    return (u.type.value, day, norm)


def exact_duplicates(scoped: list[tuple[KnowledgeUnit, str]], user_scope: str) -> list[ExactDuplicate]:
    """Units with identical normalised text, resolved without a judge.

    A group holding a copy in ``user_scope`` keeps the best shared copy and
    drops every other member (shared or agent-scoped). Otherwise each scope
    keeps its own best copy: two agent scopes never resolve against each other.
    """
    groups: dict[tuple[str, str, str], list[tuple[KnowledgeUnit, str]]] = defaultdict(list)
    for u, scope in scoped:
        if (key := exact_key(u)) is not None:
            groups[key].append((u, scope))
    out: list[ExactDuplicate] = []
    for members in groups.values():
        if len(members) < 2:
            continue
        shared = sorted((m for m in members if m[1] == user_scope), key=lambda m: _exact_keep_order(m[0]))
        if shared:
            keep, keep_scope = shared[0]
            out.extend(
                ExactDuplicate(keep=keep, keep_scope=keep_scope, drop=u, drop_scope=scope)
                for u, scope in members if u.id != keep.id
            )
            continue
        by_scope: dict[str, list[KnowledgeUnit]] = defaultdict(list)
        for u, scope in members:
            by_scope[scope].append(u)
        for scope, units in by_scope.items():
            ordered = sorted(units, key=_exact_keep_order)
            out.extend(ExactDuplicate(keep=ordered[0], keep_scope=scope, drop=u, drop_scope=scope) for u in ordered[1:])
    return out


@dataclass(frozen=True)
class PairTuning:
    """Candidate-volume knobs (``ReconcileSettings`` feeds them from env).

    - ``cosine_threshold``: minimum embedding cosine for an embedding pair.
    - ``max_per_unit``: at most this many neighbours per focus unit, best first
      (embedding and FTS pairs together), so one generic unit cannot flood a run.
    - ``fts_neighbours``: BM25 neighbours looked up per focus unit.
    - ``fts_min_overlap``: an FTS-only pair is kept only when the two units'
      content words overlap at least this much (Jaccard). BM25 alone surfaces
      topical neighbours, not duplicates; set above 1 to drop FTS-only pairs.
    """

    cosine_threshold: float = 0.90
    max_per_unit: int = 3
    fts_neighbours: int = 3
    fts_min_overlap: float = 0.5


def word_overlap(a: str, b: str) -> float:
    """Jaccard overlap of the two texts' content words (pipeline-normalised)."""
    wa, wb = content_words(a), content_words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _embedding_neighbours(
    store: SQLiteStore,
    focus: list[KnowledgeUnit],
    focus_scope: str,
    pool: dict[str, KnowledgeUnit],
    pool_scope: str,
    tuning: PairTuning,
) -> dict[str, list[CandidatePair]]:
    """Per focus unit: its pool neighbours at or above the cosine threshold, best first."""
    focus_blobs = store.scope_unit_embeddings(focus_scope, [u.id for u in focus])
    pool_blobs = store.scope_unit_embeddings(pool_scope, list(pool))
    if not focus_blobs or not pool_blobs:
        return {}
    import numpy as np

    def matrix(blobs: dict[str, bytes]) -> tuple[list[str], np.ndarray]:
        ids = list(blobs)
        mat = np.stack([np.frombuffer(blobs[i], dtype="<f4") for i in ids]).astype(np.float32)
        norms = np.linalg.norm(mat, axis=1)
        norms[norms == 0.0] = 1.0
        return ids, mat / norms[:, None]

    focus_ids, fmat = matrix(focus_blobs)
    pool_ids, pmat = matrix(pool_blobs)
    by_id = {u.id: u for u in focus}
    sims = fmat @ pmat.T
    out: dict[str, list[CandidatePair]] = {}
    for row, fid in enumerate(focus_ids):
        cols = np.nonzero(sims[row] >= tuning.cosine_threshold)[0]
        found = []
        for col in cols:
            other = pool[pool_ids[int(col)]]
            if other.id == fid or other.type is not by_id[fid].type:
                continue
            found.append(CandidatePair(a=by_id[fid], b=other, signal="embedding", score=float(sims[row, col])))
        found.sort(key=lambda p: (-p.score, p.b.id))
        out[fid] = found
    return out


def candidate_pairs(
    store: SQLiteStore,
    focus: list[KnowledgeUnit],
    pool: list[KnowledgeUnit],
    *,
    focus_scope: str,
    pool_scope: str,
    tuning: PairTuning,
    exclude: set[str] | None = None,
) -> list[CandidatePair]:
    """Deduplicated pairs (focus unit, pool unit), most similar first.

    ``focus_scope == pool_scope`` pairs within one scope; otherwise across two
    scopes of the same user. Each focus unit contributes at most
    ``tuning.max_per_unit`` pairs: its embedding neighbours by cosine, then
    FTS neighbours that pass the word-overlap bar (scored by that overlap).
    The result is sorted by score, then ids, so the order is stable from one
    call to the next. Units in ``exclude`` never pair.
    """
    exclude = exclude or set()
    by_id = {u.id: u for u in pool if u.id not in exclude}
    focus = [u for u in focus if u.id not in exclude]
    embedded = _embedding_neighbours(store, focus, focus_scope, by_id, pool_scope, tuning)
    seen: set[frozenset[str]] = set()
    out: list[CandidatePair] = []
    for u in focus:
        mine: list[CandidatePair] = []
        for p in embedded.get(u.id, []):
            key = frozenset((p.a.id, p.b.id))
            if key not in seen and len(mine) < tuning.max_per_unit:
                seen.add(key)
                mine.append(p)
        if len(mine) < tuning.max_per_unit and tuning.fts_neighbours > 0 and tuning.fts_min_overlap <= 1:
            fts: list[CandidatePair] = []
            for other_id in store.fts_unit_neighbours(u.content, pool_scope, tuning.fts_neighbours + 1):
                other = by_id.get(other_id)
                if other is None or other.id == u.id or other.type is not u.type:
                    continue
                if frozenset((u.id, other.id)) in seen:
                    continue
                overlap = word_overlap(u.content, other.content)
                if overlap >= tuning.fts_min_overlap:
                    fts.append(CandidatePair(a=u, b=other, signal="fts", score=overlap))
            for p in sorted(fts, key=lambda p: (-p.score, p.b.id))[: tuning.max_per_unit - len(mine)]:
                seen.add(frozenset((p.a.id, p.b.id)))
                mine.append(p)
        out += mine
    out.sort(key=lambda p: (-p.score, p.a.id, p.b.id))
    return out


__all__ = [
    "CandidatePair",
    "PairTuning",
    "ExactDuplicate",
    "candidate_pairs",
    "content_hash",
    "exact_duplicates",
    "exact_key",
    "normalise_content",
    "pair_key",
    "word_overlap",
]
