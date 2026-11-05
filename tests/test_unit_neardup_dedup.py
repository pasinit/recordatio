"""Regression tests for embedding near-duplicate dedup on unit write.

Exact-string dedup (``_norm_content`` equality) never matches a paraphrased
re-extraction of the same standing fact, so each re-encounter inserts a fresh
unit (extraction_count=1) and near-identical clusters accumulate. The
embedding-cosine fallback in ``_active_unit_id_for_content`` /
``_near_duplicate_unit_id`` reinforces the canonical instead.

These tests inject deterministic vectors (the real bi-encoder is a heavy ML
dependency and is never loaded here): ``embeddings.embed_documents`` is
monkeypatched to embed the incoming content, and the canonical unit's stored
embedding is written directly via ``store.upsert_item_embedding``. So they
exercise the dedup control flow without downloading model weights.

A second group of tests covers the flag-independent lexical fallback
(``_lexical_near_duplicate_unit_id``): the embedding near-dup path is gated on
``RECORDATIO_F2_EMBEDDINGS`` (default off), so in a deploy where F2 is dark the
lexical difflib-ratio fallback is what keeps write-side paraphrase dedup from
going dark. These run with ``embeddings.enabled`` monkeypatched to ``False`` and
no model loaded at all.
"""

import logging
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from tpasini.recordatio.extraction.contracts import ExtractionResult
from tpasini.recordatio.models import KnowledgeType, KnowledgeUnit
from tpasini.recordatio.proxy import embeddings
from tpasini.recordatio.proxy.extraction_pipeline import (
    _active_unit_id_for_content,
    _persist_units_and_topics,
)
from tpasini.recordatio.storage.base import SQLiteStore

_AGENT = "sam"
_DIM = embeddings.EMBED_DIM


def _unit(content: str, unit_type: KnowledgeType = KnowledgeType.PROCEDURAL) -> KnowledgeUnit:
    now = datetime.now(UTC)
    return KnowledgeUnit(
        type=unit_type,
        content=content,
        asserted_at=now,
        valid_from=now,
        provenance="test",
    )


def _onehot(index: int) -> list[float]:
    """An L2-normalized one-hot vector (unit norm, like the real encoder)."""
    vec = [0.0] * _DIM
    vec[index] = 1.0
    return vec


def _vec_with_cosine(target: float) -> list[float]:
    """An L2-normalized vector whose cosine with ``_onehot(0)`` == ``target``.

    Lets a test pin a paraphrase's similarity to a precise value that straddles
    the default 0.97 cutoff, so a per-type threshold override is what decides the
    merge.
    """
    vec = [0.0] * _DIM
    vec[0] = target
    vec[1] = (1.0 - target * target) ** 0.5
    return vec


@pytest.fixture()
def store(tmp_path: Path) -> SQLiteStore:
    return SQLiteStore(tmp_path / "neardup.db")


def _store_embedding(store: SQLiteStore, unit_id: str, vec: list[float]) -> None:
    store.upsert_item_embedding(
        item_id=unit_id,
        agent_id=_AGENT,
        kind="unit",
        surface_hash="test-hash",
        embedding=embeddings.pack_embedding(vec),
        model="test",
        asserted_at=datetime.now(UTC).isoformat(),
    )


def test_paraphrase_reinforces_canonical_when_embeddings_on(
    store: SQLiteStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical = _unit("Scout's daily brief must list each tracked role once, newest first.")
    store.add(canonical, _AGENT)
    _store_embedding(store, canonical.id, _onehot(0))

    # A paraphrase of the same standing rule: no byte-for-byte match, but the
    # (injected) embedding is essentially identical to the canonical's.
    paraphrase = _unit("For the Scout daily brief, enter every tracked role a single time, most recent first.")

    monkeypatch.setattr(embeddings, "enabled", lambda: True)
    monkeypatch.setattr(embeddings, "embed_documents", lambda surfaces: [_onehot(0)])

    ctx = SimpleNamespace(store=store)
    assert _active_unit_id_for_content(ctx, paraphrase, _AGENT) == canonical.id


def test_distinct_fact_is_not_merged(store: SQLiteStore, monkeypatch: pytest.MonkeyPatch) -> None:
    canonical = _unit("Scout's daily brief lists tracked roles newest first.")
    store.add(canonical, _AGENT)
    _store_embedding(store, canonical.id, _onehot(0))

    # A genuinely different fact — orthogonal embedding, cosine 0 < threshold.
    other = _unit("The weekly report should be sent every Friday afternoon.")

    monkeypatch.setattr(embeddings, "enabled", lambda: True)
    monkeypatch.setattr(embeddings, "embed_documents", lambda surfaces: [_onehot(1)])

    ctx = SimpleNamespace(store=store)
    assert _active_unit_id_for_content(ctx, other, _AGENT) is None


def test_lexical_paraphrase_reinforces_when_embeddings_disabled(
    store: SQLiteStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The headline regression: with F2 dark (the actual deploy config — the
    flag is set nowhere), a light paraphrase of a standing durable fact now
    reinforces the canonical via the flag-independent lexical fallback instead of
    inserting a near-identical twin.

    Fails on the old code, where ``_near_duplicate_unit_id`` short-circuits to
    ``None`` whenever ``embeddings.enabled()`` is false, so the only dedup that
    could fire was the exact ``_norm_content`` byte match — which this
    punctuation/word-order paraphrase does not satisfy. No embedding model is
    loaded or monkeypatched here: the lexical path uses stored text only.
    """
    canonical = _unit("Scout's daily brief lists tracked roles newest first.")
    store.add(canonical, _AGENT)
    paraphrase = _unit("Scout daily brief: tracked roles, newest first.")

    monkeypatch.setattr(embeddings, "enabled", lambda: False)

    ctx = SimpleNamespace(store=store)
    assert _active_unit_id_for_content(ctx, paraphrase, _AGENT) == canonical.id


def test_lexical_fallback_does_not_merge_heavy_rewrite_when_disabled(
    store: SQLiteStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lexical fallback is conservative: a heavy semantic rewrite (difflib
    ratio well under the cutoff) still inserts fresh. Catching those is the
    embedding path's / nightly reconcile's job — a lexical merge there would risk
    an irreversible wrong collapse."""
    canonical = _unit("Scout's daily brief must list each tracked role once, newest first.")
    store.add(canonical, _AGENT)
    heavy_rewrite = _unit(
        "For the Scout daily brief, enter every tracked role a single time, most recent first."
    )

    monkeypatch.setattr(embeddings, "enabled", lambda: False)

    ctx = SimpleNamespace(store=store)
    assert _active_unit_id_for_content(ctx, heavy_rewrite, _AGENT) is None


def test_lexical_fallback_does_not_merge_distinct_fact_when_disabled(
    store: SQLiteStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A genuinely different fact is nowhere near the cutoff and inserts fresh."""
    canonical = _unit("Scout's daily brief lists tracked roles newest first.")
    store.add(canonical, _AGENT)
    other = _unit("The weekly report should be sent every Friday afternoon.")

    monkeypatch.setattr(embeddings, "enabled", lambda: False)

    ctx = SimpleNamespace(store=store)
    assert _active_unit_id_for_content(ctx, other, _AGENT) is None


def test_lexical_fallback_episodic_stays_conservative_when_disabled(
    store: SQLiteStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """EPISODIC keeps the strict lexical default (0.95), not the looser durable
    0.90: a dated status snapshot whose only change is the status word
    ('submitted' -> 'received') is a DISTINCT event and must not be merged. The
    same-ratio durable paraphrase in the next test does merge — proving the
    per-type split is real."""
    canonical = _unit(
        "On 2026-09-14 the HRW grants officer application was marked submitted.",
        unit_type=KnowledgeType.EPISODIC,
    )
    store.add(canonical, _AGENT)
    distinct_status = _unit(
        "On 2026-09-14 the HRW grants officer application was marked received.",
        unit_type=KnowledgeType.EPISODIC,
    )

    monkeypatch.setattr(embeddings, "enabled", lambda: False)

    ctx = SimpleNamespace(store=store)
    # ratio ~0.92: below the strict episodic default 0.95 → inserts fresh.
    assert _active_unit_id_for_content(ctx, distinct_status, _AGENT) is None


def test_lexical_fallback_durable_merges_in_per_type_band_when_disabled(
    store: SQLiteStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A durable paraphrase at ratio ~0.94 — between the durable (0.90) and
    strict/episodic (0.95) cutoffs — merges for SEMANTIC/PROCEDURAL, confirming
    the looser built-in default is what reinforces the paraphrase-prone types."""
    canonical = _unit("Scout's daily brief lists tracked roles newest first.")
    store.add(canonical, _AGENT)
    paraphrase = _unit("Scout's daily brief shows tracked roles newest first.")

    monkeypatch.setattr(embeddings, "enabled", lambda: False)

    ctx = SimpleNamespace(store=store)
    assert _active_unit_id_for_content(ctx, paraphrase, _AGENT) == canonical.id


def test_lexical_fallback_env_override_when_disabled(
    store: SQLiteStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lexical cutoff is env-overridable without a rebuild: raising the
    procedural knob above the paraphrase's ratio stops the merge."""
    canonical = _unit("Scout's daily brief lists tracked roles newest first.")
    store.add(canonical, _AGENT)
    paraphrase = _unit("Scout's daily brief shows tracked roles newest first.")  # ratio ~0.94

    monkeypatch.setattr(embeddings, "enabled", lambda: False)
    monkeypatch.setenv("RECORDATIO_UNIT_NEARDUP_LEXICAL_RATIO_PROCEDURAL", "0.99")

    ctx = SimpleNamespace(store=store)
    assert _active_unit_id_for_content(ctx, paraphrase, _AGENT) is None


def test_exact_match_still_short_circuits_before_embeddings(
    store: SQLiteStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exact (normalized) match returns without ever embedding."""
    canonical = _unit("Scout's daily brief lists tracked roles newest first.")
    store.add(canonical, _AGENT)

    def _boom(_surfaces: list[str]) -> list[list[float]]:
        raise AssertionError("embed_documents must not be called on an exact match")

    monkeypatch.setattr(embeddings, "enabled", lambda: True)
    monkeypatch.setattr(embeddings, "embed_documents", _boom)

    # Same content, different whitespace/case → normalizes equal.
    dup = _unit("  Scout's daily brief   lists tracked roles NEWEST first. ")
    ctx = SimpleNamespace(store=store)
    assert _active_unit_id_for_content(ctx, dup, _AGENT) == canonical.id


@pytest.mark.parametrize("unit_type", [KnowledgeType.PROCEDURAL, KnowledgeType.SEMANTIC])
def test_durable_paraphrase_reinforces_by_builtin_default(
    store: SQLiteStore, monkeypatch: pytest.MonkeyPatch, unit_type: KnowledgeType
) -> None:
    """A durable (procedural/semantic) paraphrase just under the old conservative
    0.97 now reinforces the canonical by the looser per-type BUILT-IN default —
    no env override required.

    This is the recurring production complaint: durable facts the LLM paraphrases
    slightly each run landed below 0.97 and twinned instead of reinforcing, which
    diluted retrieval. It fails on the old code (built-in default was a flat 0.97).
    """
    canonical = _unit("Scout's daily brief lists each tracked role once, newest first.", unit_type=unit_type)
    store.add(canonical, _AGENT)
    _store_embedding(store, canonical.id, _onehot(0))

    paraphrase = _unit(
        "For Scout's brief, each tracked role appears a single time, most recent first.",
        unit_type=unit_type,
    )
    monkeypatch.setattr(embeddings, "enabled", lambda: True)
    monkeypatch.setattr(embeddings, "embed_documents", lambda surfaces: [_vec_with_cosine(0.95)])
    # No RECORDATIO_UNIT_NEARDUP_* env set: the looser cutoff is the built-in default.

    ctx = SimpleNamespace(store=store)
    assert _active_unit_id_for_content(ctx, paraphrase, _AGENT) == canonical.id


@pytest.mark.parametrize("unit_type", [KnowledgeType.PROCEDURAL, KnowledgeType.SEMANTIC])
def test_durable_paraphrase_below_builtin_default_still_not_merged(
    store: SQLiteStore, monkeypatch: pytest.MonkeyPatch, unit_type: KnowledgeType
) -> None:
    """The looser durable default is still a cutoff, not a free-for-all: a
    paraphrase well below it (cosine 0.90 < 0.94 default) inserts fresh."""
    canonical = _unit("Scout's daily brief lists each tracked role once, newest first.", unit_type=unit_type)
    store.add(canonical, _AGENT)
    _store_embedding(store, canonical.id, _onehot(0))

    paraphrase = _unit("Send the weekly wrap-up to the team every Friday.", unit_type=unit_type)
    monkeypatch.setattr(embeddings, "enabled", lambda: True)
    monkeypatch.setattr(embeddings, "embed_documents", lambda surfaces: [_vec_with_cosine(0.90)])

    ctx = SimpleNamespace(store=store)
    assert _active_unit_id_for_content(ctx, paraphrase, _AGENT) is None


def test_episodic_paraphrase_keeps_conservative_default(
    store: SQLiteStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """EPISODIC units are deliberately excluded from the looser durable default:
    their exact wording is load-bearing (dated snapshots, per-application tracker
    records), so a 0.95 paraphrase still lands below the conservative 0.97 and
    inserts fresh — protecting those records from being merged away."""
    canonical = _unit("On 2026-08-21 the owner asked to permanently exclude helper.", unit_type=KnowledgeType.EPISODIC)
    store.add(canonical, _AGENT)
    _store_embedding(store, canonical.id, _onehot(0))

    paraphrase = _unit(
        "The owner requested helper be permanently excluded, 2026-08-21.",
        unit_type=KnowledgeType.EPISODIC,
    )
    monkeypatch.setattr(embeddings, "enabled", lambda: True)
    monkeypatch.setattr(embeddings, "embed_documents", lambda surfaces: [_vec_with_cosine(0.95)])

    ctx = SimpleNamespace(store=store)
    assert _active_unit_id_for_content(ctx, paraphrase, _AGENT) is None


def test_per_type_override_loosens_durable_type(store: SQLiteStore, monkeypatch: pytest.MonkeyPatch) -> None:
    """The per-type knob loosens the cutoff for a paraphrase-prone durable type,
    so a 0.95 paraphrase now reinforces the canonical instead of twinning."""
    canonical = _unit(
        "Scout's daily brief lists each tracked role once, newest first.",
        unit_type=KnowledgeType.PROCEDURAL,
    )
    store.add(canonical, _AGENT)
    _store_embedding(store, canonical.id, _onehot(0))

    paraphrase = _unit(
        "For Scout's brief, each tracked role appears a single time, most recent first.",
        unit_type=KnowledgeType.PROCEDURAL,
    )
    monkeypatch.setattr(embeddings, "enabled", lambda: True)
    monkeypatch.setattr(embeddings, "embed_documents", lambda surfaces: [_vec_with_cosine(0.95)])
    monkeypatch.setenv("RECORDATIO_UNIT_NEARDUP_COSINE_PROCEDURAL", "0.90")

    ctx = SimpleNamespace(store=store)
    assert _active_unit_id_for_content(ctx, paraphrase, _AGENT) == canonical.id


def test_persist_logs_insert_vs_reinforce_tally(
    store: SQLiteStore, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The persistence pass emits a per-turn insert-vs-reinforce tally at INFO so
    a silent near-dup no-op (e.g. F2 embeddings off → every paraphrase twins) is
    visible in the logs without hand-auditing the store.

    Fails on the old code: only per-unit DEBUG lines existed, with no aggregate
    outcome tally and no record of whether the embedding near-dup path was even
    available.
    """
    # One already-stored canonical that the incoming batch re-asserts verbatim
    # (exact-string reinforce path, no embeddings needed) plus one brand-new fact.
    canonical = _unit("Scout's daily brief lists tracked roles newest first.", KnowledgeType.SEMANTIC)
    store.add(canonical, _AGENT)

    dup = _unit("Scout's daily brief lists tracked roles newest first.", KnowledgeType.SEMANTIC)
    fresh = _unit("Tommaso is tracking three properties this month.", KnowledgeType.SEMANTIC)
    result = ExtractionResult(units=[dup, fresh])

    monkeypatch.setattr(embeddings, "enabled", lambda: False)
    ctx = SimpleNamespace(store=store)

    with caplog.at_level(logging.INFO, logger="tpasini.recordatio.proxy.extraction_pipeline"):
        _persist_units_and_topics(ctx, result, datetime.now(UTC), _AGENT)

    summaries = [r.getMessage() for r in caplog.records if "persisted knowledge units" in r.getMessage()]
    assert len(summaries) == 1
    assert "1 inserted" in summaries[0]
    assert "1 reinforced" in summaries[0]
    # The gating condition for the near-dup fallback is recorded so an all-insert
    # run with embeddings off is distinguishable from a genuine all-new batch.
    assert "enabled=False" in summaries[0]


def test_persist_logs_nothing_when_no_units(
    store: SQLiteStore, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """No units persisted → no summary line (the tally is not spurious noise)."""
    monkeypatch.setattr(embeddings, "enabled", lambda: False)
    ctx = SimpleNamespace(store=store)
    with caplog.at_level(logging.INFO, logger="tpasini.recordatio.proxy.extraction_pipeline"):
        _persist_units_and_topics(ctx, ExtractionResult(units=[]), datetime.now(UTC), _AGENT)
    assert not [r for r in caplog.records if "persisted knowledge units" in r.getMessage()]


def test_per_type_override_does_not_bleed_across_types(
    store: SQLiteStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Loosening the procedural knob must NOT loosen episodic units, whose exact
    wording is more load-bearing — they keep the conservative default."""
    canonical = _unit("On 2026-08-21 the owner asked to permanently exclude helper.", unit_type=KnowledgeType.EPISODIC)
    store.add(canonical, _AGENT)
    _store_embedding(store, canonical.id, _onehot(0))

    paraphrase = _unit(
        "The owner requested helper be permanently excluded, 2026-08-21.",
        unit_type=KnowledgeType.EPISODIC,
    )
    monkeypatch.setattr(embeddings, "enabled", lambda: True)
    monkeypatch.setattr(embeddings, "embed_documents", lambda surfaces: [_vec_with_cosine(0.95)])
    # Only the procedural knob is loosened; episodic still resolves to the 0.97 default.
    monkeypatch.setenv("RECORDATIO_UNIT_NEARDUP_COSINE_PROCEDURAL", "0.90")

    ctx = SimpleNamespace(store=store)
    assert _active_unit_id_for_content(ctx, paraphrase, _AGENT) is None


def test_global_knob_still_applies_to_all_types(store: SQLiteStore, monkeypatch: pytest.MonkeyPatch) -> None:
    """The pre-existing global knob remains the fallback for every type."""
    canonical = _unit("Tommaso is tracking three properties.", unit_type=KnowledgeType.SEMANTIC)
    store.add(canonical, _AGENT)
    _store_embedding(store, canonical.id, _onehot(0))

    paraphrase = _unit("Tommaso currently tracks three separate properties.", unit_type=KnowledgeType.SEMANTIC)
    monkeypatch.setattr(embeddings, "enabled", lambda: True)
    monkeypatch.setattr(embeddings, "embed_documents", lambda surfaces: [_vec_with_cosine(0.95)])
    monkeypatch.setenv("RECORDATIO_UNIT_NEARDUP_COSINE", "0.90")

    ctx = SimpleNamespace(store=store)
    assert _active_unit_id_for_content(ctx, paraphrase, _AGENT) == canonical.id
