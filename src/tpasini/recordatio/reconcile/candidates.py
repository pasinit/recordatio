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


def _embedding_pairs(
    store: SQLiteStore,
    focus: list[KnowledgeUnit],
    focus_scope: str,
    pool: dict[str, KnowledgeUnit],
    pool_scope: str,
    threshold: float,
) -> list[CandidatePair]:
    focus_blobs = store.scope_unit_embeddings(focus_scope, [u.id for u in focus])
    pool_blobs = store.scope_unit_embeddings(pool_scope, list(pool))
    if not focus_blobs or not pool_blobs:
        return []
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
    out: list[CandidatePair] = []
    for row, fid in enumerate(focus_ids):
        for col in np.nonzero(sims[row] >= threshold)[0]:
            other = pool[pool_ids[int(col)]]
            if other.id == fid or other.type is not by_id[fid].type:
                continue
            out.append(CandidatePair(a=by_id[fid], b=other, signal="embedding", score=float(sims[row, col])))
    return out


def candidate_pairs(
    store: SQLiteStore,
    focus: list[KnowledgeUnit],
    pool: list[KnowledgeUnit],
    *,
    focus_scope: str,
    pool_scope: str,
    cosine_threshold: float,
    fts_neighbours: int,
    exclude: set[str] | None = None,
) -> list[CandidatePair]:
    """Deduplicated pairs (focus unit, pool unit), best first.

    ``focus_scope == pool_scope`` pairs within one scope; otherwise across two
    scopes of the same user. Embedding pairs rank by cosine (desc); FTS-only
    pairs follow in BM25 rank order. Units in ``exclude`` never pair.
    """
    exclude = exclude or set()
    by_id = {u.id: u for u in pool if u.id not in exclude}
    focus = [u for u in focus if u.id not in exclude]
    seen: set[frozenset[str]] = set()
    out: list[CandidatePair] = []

    embedded = _embedding_pairs(store, focus, focus_scope, by_id, pool_scope, cosine_threshold)
    for p in sorted(embedded, key=lambda p: (-p.score, p.a.id, p.b.id)):
        key = frozenset((p.a.id, p.b.id))
        if key not in seen:
            seen.add(key)
            out.append(p)

    fts_pairs: list[CandidatePair] = []
    for u in focus:
        for rank, other_id in enumerate(store.fts_unit_neighbours(u.content, pool_scope, fts_neighbours + 1)):
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
    "exact_key",
    "normalise_content",
    "pair_key",
]
