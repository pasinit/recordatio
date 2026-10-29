"""Cheap, deterministic candidate finding for the reconcile job.

Per scope, three signals propose pairs; none of them decides anything on its
own except exact duplicates:

1. normalised exact match (casefold, whitespace + edge punctuation collapsed)
   — identical text, resolved without the judge;
2. embedding cosine over the vectors already stored in the ``.emb`` sidecar
   (no model load) — the focus units against every active unit;
3. FTS/BM25 neighbours of each focus unit's content words.

Pairs are always same-scope and same-type: a unit in one agent's scope is
never dropped because another scope holds a similar one.
"""
import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass

from tpasini.recordatio.models import KnowledgeType, KnowledgeUnit
from tpasini.recordatio.reconcile.judge import recency
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
    drop: KnowledgeUnit


def _exact_keep_order(u: KnowledgeUnit) -> tuple[int, object, str]:
    # Most-reinforced first, then the earliest assertion, then id — so the
    # surviving row is the one retrieval has leaned on most.
    return (-(u.extraction_count + u.retrieval_count), recency(u)[1], u.id)


def exact_duplicates(units: list[KnowledgeUnit]) -> list[ExactDuplicate]:
    """Same-type units with identical normalised text. Episodic units only
    group within the same valid_from day (the same event recorded twice);
    the same text on another day is another event."""
    groups: dict[tuple[str, str, str], list[KnowledgeUnit]] = defaultdict(list)
    for u in units:
        norm = normalise_content(u.content)
        if not norm:
            continue
        day = recency(u)[0].date().isoformat() if u.type is KnowledgeType.EPISODIC else ""
        groups[(u.type.value, day, norm)].append(u)
    out: list[ExactDuplicate] = []
    for members in groups.values():
        if len(members) < 2:
            continue
        ordered = sorted(members, key=_exact_keep_order)
        out.extend(ExactDuplicate(keep=ordered[0], drop=other) for other in ordered[1:])
    return out


def _embedding_pairs(
    store: SQLiteStore,
    agent_id: str,
    units: dict[str, KnowledgeUnit],
    focus: list[KnowledgeUnit],
    threshold: float,
) -> list[CandidatePair]:
    blobs = store.scope_unit_embeddings(agent_id, list(units))
    focus_ids = [u.id for u in focus if u.id in blobs]
    if not focus_ids or len(blobs) < 2:
        return []
    import numpy as np

    ids = list(blobs)
    mat = np.stack([np.frombuffer(blobs[i], dtype="<f4") for i in ids]).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1)
    norms[norms == 0.0] = 1.0
    mat = mat / norms[:, None]
    index = {iid: n for n, iid in enumerate(ids)}
    sims = mat[[index[i] for i in focus_ids]] @ mat.T
    out: list[CandidatePair] = []
    for row, fid in enumerate(focus_ids):
        for col in np.nonzero(sims[row] >= threshold)[0]:
            other = ids[int(col)]
            if other == fid or units[other].type is not units[fid].type:
                continue
            out.append(CandidatePair(a=units[fid], b=units[other], signal="embedding", score=float(sims[row, col])))
    return out


def candidate_pairs(
    store: SQLiteStore,
    agent_id: str,
    units: list[KnowledgeUnit],
    focus: list[KnowledgeUnit],
    *,
    cosine_threshold: float,
    fts_neighbours: int,
    exclude: set[str] | None = None,
) -> list[CandidatePair]:
    """Deduplicated candidate pairs touching ``focus``, best first.

    Embedding pairs rank by cosine (desc); FTS-only pairs follow in BM25
    rank order. Units in ``exclude`` (already resolved this run) never pair.
    """
    exclude = exclude or set()
    by_id = {u.id: u for u in units if u.id not in exclude}
    focus = [u for u in focus if u.id in by_id]
    seen: set[frozenset[str]] = set()
    out: list[CandidatePair] = []

    for p in sorted(
        _embedding_pairs(store, agent_id, by_id, focus, cosine_threshold), key=lambda p: (-p.score, p.a.id, p.b.id)
    ):
        key = frozenset((p.a.id, p.b.id))
        if key not in seen:
            seen.add(key)
            out.append(p)

    fts_pairs: list[CandidatePair] = []
    for u in focus:
        for rank, other_id in enumerate(store.fts_unit_neighbours(u.content, agent_id, fts_neighbours + 1)):
            other = by_id.get(other_id)
            if other is None or other.id == u.id or other.type is not u.type:
                continue
            key = frozenset((u.id, other.id))
            if key in seen:
                continue
            seen.add(key)
            fts_pairs.append(CandidatePair(a=u, b=other, signal="fts", score=float(-rank)))
    fts_pairs.sort(key=lambda p: (-p.score, p.a.id, p.b.id))
    return out + fts_pairs


__all__ = [
    "CandidatePair",
    "ExactDuplicate",
    "candidate_pairs",
    "content_hash",
    "exact_duplicates",
    "normalise_content",
    "pair_key",
]
