"""Production embedding source for hybrid bm25⊕embedding retrieval (F2).

The live counterpart of ``scripts/generate_embed_scores.py``: it computes the
SAME bi-encoder cosines the frozen eval artifact replays, but from embeddings
stored at write time (units/entities) plus on-the-fly embedding of relations,
rather than from a precomputed JSON.  Parity with the eval recipe is a hard
constraint (F2_EMBEDDINGS_DESIGN.md "Production source MUST match"):

  * model      — ``BAAI/bge-small-en-v1.5`` (fixed by the F2-0 separability
                 gate, NOT inclusion; F8).
  * query side — BGE instruction prefix on the QUERY only; documents bare.
  * surface    — store text only (RR4), built EXACTLY as the budgeter does:
                   unit     -> content
                   entity   -> "label description"
                   relation -> "subject predicate object" (rendered labels)
                 mirrors ``briefing_budget._unit_text/_entity_text/_relation_text``
                 and the eval ``_candidate_surfaces``.
  * cosine     — embeddings L2-normalized; cosine == dot product.

Warm-pool singleton (G-R3, mirrors ``reranker.py``): one model load per
process, lazy, thread-safe, and graceful when sentence-transformers / the
weights are unavailable (returns no scores → the budgeter degrades to BM25 for
those items, exactly the documented fallback).

Persistence: unit and entity embeddings are precomputed at write time and
stored keyed by ``(item_id, surface_hash)`` (see ``storage.base`` /
``item_embeddings``); ``surface_hash`` guards no-op re-embeds.  Relations are
embedded on the fly at query time — their searchable surface (rendered triple
with endpoint labels resolved) only exists at render time, and there are few
per turn, so the one-shot cost is negligible.  A unit/entity with no stored
embedding (un-backfilled store) simply contributes nothing to the embedding
list; RRF's union handles the absence (graceful degrade to BM25 for that item).

Off by default behind ``RECORDATIO_F2_EMBEDDINGS`` so the deploy can ship dark and
flip the relevance change on (or roll it back) without a rebuild.
"""

import hashlib
import logging
import os
import struct
import threading
import time

_logger = logging.getLogger("recordatio.embeddings")

# Pinned bi-encoder — fixed by the F2-0 separability sanity gate (NOT inclusion;
# F8).  MUST match scripts/generate_embed_scores.py or eval and prod diverge.
_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384  # bge-small-en-v1.5 hidden size; stored BLOBs are EMBED_DIM float32.

# BGE retrieval recipe: instruction prefix on the QUERY side only (documents get
# no prefix).  Identical string to the eval generator.
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Surface-hash version: bump only if the surface-construction RULE changes (it
# must stay byte-faithful to the budgeter / eval generator).  Mixed into the
# hash so a rule change invalidates every stored row on next write.
_SURFACE_HASH_VERSION = "1"


def enabled() -> bool:
    """Whether F2 embedding fusion is active (deploy flag, default off).

    Read per call so the flag can be flipped via env without a rebuild; the
    cost is one ``os.environ.get`` — negligible next to the query embed.
    """
    return os.environ.get("RECORDATIO_F2_EMBEDDINGS", "0") not in ("0", "false", "no", "")


# ---------------------------------------------------------------------------
# Surface construction (RR4 — store text only, byte-faithful to the budgeter).
# ---------------------------------------------------------------------------


def unit_surface(content: str | None) -> str:
    """Searchable surface for a unit — mirrors ``briefing_budget._unit_text``."""
    return content or ""


def entity_surface(label: str | None, description: str | None) -> str:
    """Searchable surface for an entity — mirrors ``_entity_text`` /
    the eval generator's ``"label description"``."""
    return f"{label or ''} {description or ''}".strip()


def relation_surface(subject: str, predicate: str, obj: str) -> str:
    """Searchable surface for a rendered relation — mirrors ``_relation_text``."""
    return f"{subject} {predicate} {obj}"


def surface_hash(surface: str) -> str:
    """Content hash keying the stored embedding (re-embed only on surface change).

    Versioned so a surface-rule change invalidates all rows; sha256 of the
    exact embedded text (no normalization — the model sees this string).
    """
    h = hashlib.sha256()
    h.update(_SURFACE_HASH_VERSION.encode("utf-8"))
    h.update(b"\x00")
    h.update(surface.encode("utf-8"))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# BLOB (de)serialization — packed little-endian float32, EMBED_DIM wide.
# ---------------------------------------------------------------------------


def pack_embedding(vec: list[float]) -> bytes:
    """Serialize an embedding to a compact float32 BLOB (EMBED_DIM × 4 bytes)."""
    return struct.pack(f"<{len(vec)}f", *vec)


def unpack_embedding(blob: bytes) -> list[float]:
    """Deserialize a float32 BLOB back into a Python float list."""
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


# ---------------------------------------------------------------------------
# Warm-pool model (one load per process, G-R3 — mirrors reranker.py).
# ---------------------------------------------------------------------------

_model_lock = threading.Lock()
_model = None
_model_load_tried = False
_unavailable_warned = False


def _load_model() -> object | None:
    """Load the pinned bi-encoder once (warm pool).  None if unavailable."""
    global _model, _model_load_tried
    if _model_load_tried:
        return _model
    with _model_lock:
        if _model_load_tried:
            return _model
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            _model = SentenceTransformer(_MODEL_NAME)
            _logger.info("embeddings: loaded bi-encoder %s (warm pool)", _MODEL_NAME)
        except Exception as exc:
            _logger.warning(
                "embeddings: could not load bi-encoder %s (%s); F2 fusion will degrade to BM25-only",
                _MODEL_NAME,
                exc,
            )
            _model = None
        finally:
            _model_load_tried = True
    return _model


def get_model() -> object | None:
    """Return the warm-pool model, loading it on first call."""
    return _load_model()


def available() -> bool:
    """Whether the embedding model is loadable in this process."""
    return get_model() is not None


# ---------------------------------------------------------------------------
# Encoding.
# ---------------------------------------------------------------------------


def embed_documents(surfaces: list[str]) -> list[list[float]] | None:
    """Embed candidate surfaces (no prefix; L2-normalized).  None if no model.

    Used at write time (unit/entity persist) and at query time (relations).
    """
    model = get_model()
    if model is None:
        return None
    if not surfaces:
        return []
    vecs = model.encode(surfaces, normalize_embeddings=True, show_progress_bar=False)
    return [list(map(float, v)) for v in vecs]


def embed_query(query: str) -> list[float] | None:
    """Embed a query with the BGE instruction prefix (L2-normalized).  None if
    no model.  MATCHES the eval generator's query side exactly."""
    model = get_model()
    if model is None:
        return None
    vec = model.encode([_BGE_QUERY_INSTRUCTION + query], normalize_embeddings=True, show_progress_bar=False)[0]
    return list(map(float, vec))


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine of two L2-normalized vectors == dot product (matches the eval).

    Falls back to a defensive normalized dot if the inputs are not unit-norm.
    Scalar path (used by tests / single pairs); the query source uses the
    vectorized ``_cosine_blobs`` over the candidate matrix.
    """
    n = min(len(a), len(b))
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(n):
        x = a[i]
        y = b[i]
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    # Inputs are L2-normalized at encode time (na≈nb≈1); the divisor is a
    # no-op then and a safety net for any non-normalized caller.
    return dot / ((na**0.5) * (nb**0.5))


def _cosine_blobs(q_emb: list[float], id_blobs: list[tuple[str, bytes]]) -> list[tuple[str, float]]:
    """Vectorized cosine of one query vs many stored float32 BLOBs.

    Brute-force cosine over the candidate pool as a single numpy matmul — the
    documented scaling approach (ANN is the drop-in if stores grow). Avoids the
    per-row Python unpack + scalar loop that dominated query latency. Stored
    vectors are L2-normalized at write time and the query at embed time, so the
    dot product IS the cosine; a defensive renorm guards non-unit rows.
    """
    if not id_blobs:
        return []
    import numpy as np

    ids = [iid for iid, _ in id_blobs]
    # frombuffer is zero-copy; stack into one (N, dim) matrix.
    mat = np.stack([np.frombuffer(b, dtype="<f4") for _iid, b in id_blobs]).astype(np.float32)
    q = np.asarray(q_emb, dtype=np.float32)
    dots = mat @ q
    norms = np.linalg.norm(mat, axis=1) * float(np.linalg.norm(q))
    norms[norms == 0.0] = 1.0
    cosines = dots / norms
    return list(zip(ids, (float(c) for c in cosines), strict=True))


# ---------------------------------------------------------------------------
# Production embedding source.
# ---------------------------------------------------------------------------


def embedding_pairs(
    query: str,
    *,
    unit_ids: list[str],
    entity_ids: list[str],
    relation_surfaces: list[tuple[str, str]],
    load_stored: "object",
) -> list[tuple[str, float]]:
    """Compute ``(item_id, cosine)`` pairs for the candidate pool.

    ``load_stored(item_ids) -> {item_id: bytes}`` returns the stored,
    surface-hash-validated embedding BLOBs for ANY item kind (units, entities,
    AND relations — all cached at write time; caller-bound to the store + scope).
    ``relation_surfaces`` is ``[(relation_id, rendered_surface)]``; a relation
    whose embedding is NOT cached (just-written, pre-backfill) is embedded on the
    fly so the candidate is never silently dropped — but the common path is a
    cache hit (cached relations avoid the per-query embed of thousands of them,
    the query-latency hot spot).

    Returns pairs sorted cosine-desc, item_id-asc (deterministic), over only the
    candidates that HAD an embedding.  A candidate with no stored/computable
    embedding is simply omitted (RRF union handles the absence → BM25-only for
    that item — graceful).  Returns ``[]`` when the model is unavailable.
    """
    q_emb = embed_query(query)
    if q_emb is None:
        return []

    rel_ids = [rid for rid, _s in relation_surfaces]
    stored: dict[str, bytes] = load_stored(  # type: ignore[operator]
        [*unit_ids, *entity_ids, *rel_ids]
    )
    id_blobs: list[tuple[str, bytes]] = list(stored.items())

    # Embed on the fly ONLY relations missing from the cache (e.g. written this
    # turn, before the next backfill / write-hook embed has caught up).
    missing = [(rid, s) for rid, s in relation_surfaces if rid not in stored]
    if missing:
        rel_vecs = embed_documents([s for _rid, s in missing])
        if rel_vecs:
            for (rid, _s), vec in zip(missing, rel_vecs, strict=True):
                id_blobs.append((rid, pack_embedding(vec)))

    pairs = _cosine_blobs(q_emb, id_blobs)
    pairs.sort(key=lambda p: (-p[1], p[0]))
    return pairs


def embed_and_store(
    store: "object",
    items: list[tuple[str, str, str, str]],
    *,
    now_iso: str,
) -> int:
    """Write-time hook: embed + upsert items whose surface changed (no-op else).

    ``items`` is ``[(item_id, agent_id, kind, surface)]`` — units and entities
    persisted this turn (relations are query-time only).  For each item, the
    surface hash is compared against the stored row; only changed/new surfaces
    are re-embedded (``surface_hash`` guard).  Empty surfaces are skipped.
    Returns the number of rows (re)written.  No-op (returns 0) when the model
    or F2 is unavailable — the store simply stays un-embedded and retrieval
    degrades to BM25 (graceful).

    Synchronous: extraction already does LLM calls off the turn's critical path
    (``_persist_side`` runs in the background), so the one batched encode here
    is acceptable.
    """
    if not enabled() or get_model() is None or not items:
        return 0

    to_embed: list[tuple[str, str, str, str]] = []  # (item_id, agent_id, kind, surface)
    for item_id, agent_id, kind, surface in items:
        if not surface:
            continue
        new_hash = surface_hash(surface)
        try:
            current = store.get_item_embedding_hash(item_id, agent_id)  # type: ignore[attr-defined]
        except Exception:
            current = None
        if current == new_hash:
            continue  # surface unchanged → no-op re-embed
        to_embed.append((item_id, agent_id, kind, surface))

    if not to_embed:
        return 0

    vecs = embed_documents([s for _i, _a, _k, s in to_embed])
    if vecs is None:
        return 0

    written = 0
    for (item_id, agent_id, kind, surface), vec in zip(to_embed, vecs, strict=True):
        try:
            store.upsert_item_embedding(  # type: ignore[attr-defined]
                item_id=item_id,
                agent_id=agent_id,
                kind=kind,
                surface_hash=surface_hash(surface),
                embedding=pack_embedding(vec),
                model=_MODEL_NAME,
                asserted_at=now_iso,
            )
            written += 1
        except Exception:
            _logger.exception("embeddings: failed to upsert embedding for %s", item_id)
    return written


def timed_embedding_pairs(
    query: str,
    *,
    unit_ids: list[str],
    entity_ids: list[str],
    relation_surfaces: list[tuple[str, str]],
    load_stored: "object",
) -> tuple[list[tuple[str, float]], float]:
    """``embedding_pairs`` plus the wall-clock ms it took (latency probe)."""
    t0 = time.perf_counter()
    pairs = embedding_pairs(
        query,
        unit_ids=unit_ids,
        entity_ids=entity_ids,
        relation_surfaces=relation_surfaces,
        load_stored=load_stored,
    )
    return pairs, (time.perf_counter() - t0) * 1000.0
