"""Tests for the W1a value-per-char briefing budgeter (pure, synthetic data)."""

import math
from datetime import datetime, timedelta

from tpasini.recordatio.entity import Entity
from tpasini.recordatio.models import KnowledgeType, KnowledgeUnit, Relation, RenderedRelation
from tpasini.recordatio.proxy import briefing_budget as bb

NOW = datetime(2026, 6, 10, 12, 0, 0)


def _unit(uid: str, content: str, *, days_old: float = 0.0, type_=KnowledgeType.SEMANTIC) -> KnowledgeUnit:
    asserted = NOW - timedelta(days=days_old)
    return KnowledgeUnit(
        type=type_,
        content=content,
        asserted_at=asserted,
        valid_from=asserted,
        provenance="test",
        id=uid,
    )


def _entity(
    eid: str, label: str, *, desc: str | None = None, days_old: float = 0.0, wikidata: str | None = None
) -> Entity:
    return Entity(
        id=eid,
        label=label,
        description=desc,
        wikidata_id=wikidata,
        asserted_at=NOW - timedelta(days=days_old),
    )


def _relation(rid: str, subj: str, pred: str, obj: str, *, days_old: float = 0.0) -> Relation:
    asserted = NOW - timedelta(days=days_old)
    return Relation(
        id=rid,
        subject_id=subj,
        predicate=pred,
        object_id=obj,
        asserted_at=asserted,
        valid_from=asserted,
        valid_until=None,
        provenance="test",
    )


_NO_NORMS = bb._SalienceNorms(max_log_extraction=0.0, max_log_retrieval=0.0, max_degree=0)


def _norms(units=(), entities=(), relations=()):
    rels = list(relations)
    return bb._salience_norms(list(units), list(entities), rels, bb._entity_degrees(rels))


def _select(**over):
    base = dict(
        query="",
        now=NOW,
        query_units=[],
        query_entities=[],
        query_raw_relations=[],
        query_rendered_relations=[],
        profile_units=[],
        profile_entities=[],
        profile_raw_relations=[],
        profile_rendered_relations=[],
    )
    base.update(over)
    return bb.select_briefing(**base)


# --- value monotonicity --------------------------------------------------


def test_relevance_monotonic():
    """A unit matching more query terms ranks higher (greater value)."""
    q = "tell me about springfield property budget"
    matching = _unit("u1", "springfield property budget search flat area")
    other = _unit("u2", "completely unrelated trivia about penguins")
    words = bb._content_words(q)
    v_match = bb._score_unit(matching, words, NOW, _NO_NORMS, bb.ALPHA, {})
    v_other = bb._score_unit(other, words, NOW, _NO_NORMS, bb.ALPHA, {})
    assert v_match > v_other


def test_recency_monotonic():
    fresh = _unit("u1", "same text", days_old=1)
    stale = _unit("u2", "same text", days_old=400)
    words = bb._content_words("same text")
    assert bb._score_unit(fresh, words, NOW, _NO_NORMS, bb.ALPHA, {}) > bb._score_unit(
        stale, words, NOW, _NO_NORMS, bb.ALPHA, {}
    )


def test_recency_halflife():
    """At exactly HALF_LIFE_DAYS recency is ~0.5."""
    assert abs(bb._recency(NOW - timedelta(days=bb.HALF_LIFE_DAYS), NOW) - 0.5) < 1e-9
    assert bb._recency(NOW, NOW) == 1.0


def test_episodic_half_life_is_shorter_than_semantic():
    """EPISODIC units decay on a much shorter half-life than stable facts."""
    episodic = _unit("e", "status snapshot", type_=KnowledgeType.EPISODIC)
    semantic = _unit("s", "profile fact", type_=KnowledgeType.SEMANTIC)
    assert bb._unit_half_life(episodic) == bb.EPISODIC_HALF_LIFE_DAYS
    assert bb._unit_half_life(semantic) == bb.HALF_LIFE_DAYS
    assert bb.EPISODIC_HALF_LIFE_DAYS < bb.HALF_LIFE_DAYS


def test_fresh_episodic_beats_stale_salient_episodic():
    """Regression: a fresh same-day EPISODIC status snapshot outranks a stale
    (90-day-old) but more-salient EPISODIC snapshot.

    Under the old uniform 90-day half-life the stale unit's recency was still
    ~0.5, so its accumulated extraction/retrieval salience flipped the ranking
    and a stale snapshot outscored a same-day authoritative record — the
    recurring cross-user recall-miss pathology (tommaso 2026-08-16, sam
    2026-08-18/19). The type-conditioned EPISODIC half-life restores
    fresh-record-wins. This assertion fails on the uniform-half-life code.
    """
    fresh = _unit(
        "fresh", "oak terrace flat live today", days_old=0, type_=KnowledgeType.EPISODIC
    )
    stale = KnowledgeUnit(
        type=KnowledgeType.EPISODIC,
        content="elm gardens flat stale snapshot",
        asserted_at=NOW - timedelta(days=90),
        valid_from=NOW - timedelta(days=90),
        provenance="test",
        id="stale",
        extraction_count=10,  # re-asserted often → high write-side salience
        retrieval_count=9,  # surfaced often → high read-side salience
    )
    # Norms chosen so the stale unit's blended salience lands at a moderate ~0.5
    # (its own log-counters sit at half of the per-set maxima): high enough to
    # win under the old uniform half-life, not so high it swamps recency.
    norms = bb._SalienceNorms(
        max_log_extraction=2 * math.log1p(9),
        max_log_retrieval=2 * math.log1p(9),
        max_degree=0,
    )
    v_fresh = bb._score_unit(fresh, set(), NOW, norms, bb.ALPHA, {})
    v_stale = bb._score_unit(stale, set(), NOW, norms, bb.ALPHA, {})
    assert v_fresh > v_stale


def test_salience_monotonic():
    """An entity with higher degree centrality scores higher."""
    hub = _entity("hub", "Hub", days_old=0)
    leaf = _entity("leaf", "Leaf", days_old=0)
    other = _entity("o1", "O1")
    rels = [
        _relation("r1", "hub", "p", "o1"),
        _relation("r2", "hub", "p", "leaf"),
        _relation("r3", "o1", "p", "hub"),
    ]
    degrees = bb._entity_degrees(rels)
    norms = _norms(entities=[hub, leaf, other], relations=rels)
    v_hub = bb._score_entity(hub, set(), NOW, degrees, norms, alpha=0.0, relevance_by_id={})
    v_leaf = bb._score_entity(leaf, set(), NOW, degrees, norms, alpha=0.0, relevance_by_id={})
    assert v_hub > v_leaf
    _ = other


# --- reinforcement-counter salience (C4) ----------------------------------


def _counted_unit(uid: str, *, ext: int = 1, ret: int = 0, days_old: float = 0.0) -> KnowledgeUnit:
    u = _unit(uid, "same text either way", days_old=days_old)
    u.extraction_count = ext
    u.retrieval_count = ret
    return u


def test_unit_salience_monotone_in_extraction_count():
    low = _counted_unit("low", ext=5)
    high = _counted_unit("high", ext=20)
    norms = _norms(units=[low, high])
    assert bb._unit_salience(high, norms) > bb._unit_salience(low, norms)


def test_unit_salience_monotone_in_retrieval_count():
    low = _counted_unit("low", ret=5)
    high = _counted_unit("high", ret=40)
    norms = _norms(units=[low, high])
    assert bb._unit_salience(high, norms) > bb._unit_salience(low, norms)


def test_retrieval_contribution_below_extraction_for_equal_counts():
    """retrieval_count is a self-reinforcing feedback loop (selected → counted →
    more salient → selected), so for the same effective count it must move
    salience LESS than the externally-driven extraction_count."""
    # extraction baseline is 1, so ext=1+n and ret=n are equal effective counts.
    n = 12
    ext_only = _counted_unit("ext", ext=1 + n)
    ret_only = _counted_unit("ret", ret=n)
    norms = _norms(units=[ext_only, ret_only])
    baseline = _counted_unit("base")
    sal_base = bb._unit_salience(baseline, norms)
    sal_ext = bb._unit_salience(ext_only, norms)
    sal_ret = bb._unit_salience(ret_only, norms)
    assert sal_ext - sal_base > sal_ret - sal_base > 0


def test_fresh_unit_keeps_floor_salience():
    """Both counters at baseline -> the 0.1 floor still applies."""
    fresh = _counted_unit("fresh")
    reinforced = _counted_unit("hot", ext=10, ret=20)
    norms = _norms(units=[fresh, reinforced])
    assert bb._unit_salience(fresh, norms) == bb.UNIT_SALIENCE
    assert bb._unit_salience(reinforced, norms) > bb.UNIT_SALIENCE


def test_extraction_baseline_contributes_zero():
    """extraction_count=1 (never re-asserted) contributes 0 to the blend."""
    assert bb._log_extraction(_counted_unit("u", ext=1)) == 0.0
    assert bb._log_extraction(_counted_unit("u", ext=2)) > 0.0


def test_entity_salience_monotone_in_counters():
    a = _entity("a", "Alpha")
    b = _entity("b", "Beta")
    a.extraction_count, a.retrieval_count = 15, 30
    rels = [_relation("r1", "a", "p", "b")]
    degrees = bb._entity_degrees(rels)
    norms = _norms(entities=[a, b], relations=rels)
    # same degree (1 each), counters differ -> a wins
    assert bb._score_entity(a, set(), NOW, degrees, norms, alpha=0.0, relevance_by_id={}) > bb._score_entity(
        b, set(), NOW, degrees, norms, alpha=0.0, relevance_by_id={}
    )


def test_relation_uses_its_own_counters():
    """Two relations sharing endpoints differ only by their own counters."""
    e1, e2 = _entity("e1", "A"), _entity("e2", "B")
    hot = _relation("hot", "e1", "p", "e2")
    cold = _relation("cold", "e1", "q", "e2")
    hot.extraction_count, hot.retrieval_count = 12, 30
    rels = [hot, cold]
    degrees = bb._entity_degrees(rels)
    norms = _norms(entities=[e1, e2], relations=rels)
    rr = RenderedRelation("A", "p", "B")
    v_hot = bb._score_relation(hot, rr, set(), NOW, degrees, norms, alpha=0.0, relevance_by_id={})
    v_cold = bb._score_relation(cold, rr, set(), NOW, degrees, norms, alpha=0.0, relevance_by_id={})
    assert v_hot > v_cold


def test_counterless_candidate_set_degrades_to_degree_salience():
    """With every counter at baseline the blend renormalises to pure degree
    centrality — pre-C4 behaviour, so legacy stores rank identically."""
    hub, leaf = _entity("hub", "Hub"), _entity("leaf", "Leaf")
    rels = [_relation("r1", "hub", "p", "leaf"), _relation("r2", "hub", "p", "leaf")]
    norms = _norms(entities=[hub, leaf], relations=rels)
    assert bb._blend_salience(hub, 1.0, norms) == 1.0
    assert bb._blend_salience(leaf, 0.5, norms) == 0.5


# --- budget respected ----------------------------------------------------


def test_budget_never_exceeded():
    units = [_unit(f"u{i}", "word " * 40, days_old=i) for i in range(50)]
    sel = _select(
        query="word",
        query_units=units,
        recall_budget=500,
        profile_budget=300,
    )
    assert sel.query_items.chars <= 500
    assert sel.profile_items.chars <= 300


def test_profile_budget_respected():
    units = [_unit(f"p{i}", "identity fact " * 10, days_old=i) for i in range(30)]
    sel = _select(profile_units=units, profile_budget=400)
    assert sel.profile_items.chars <= 400
    assert len(sel.profile_items.units) >= 1


# --- graceful drop -------------------------------------------------------


def test_graceful_drop_skips_oversized_keeps_smaller():
    """A too-big high-value item is skipped; smaller items still fill."""
    huge = _unit("huge", "relevant " * 200, days_old=0)  # high value, too big
    small_a = _unit("small_a", "relevant small", days_old=0)
    small_b = _unit("small_b", "relevant tiny", days_old=0)
    budget = bb.char_unit(small_a) + bb.char_unit(small_b) + 5
    assert bb.char_unit(huge) > budget
    sel = _select(
        query="relevant",
        query_units=[huge, small_a, small_b],
        recall_budget=budget,
    )
    ids = {s.unit.id for s in sel.query_items.units}
    assert "huge" not in ids
    assert {"small_a", "small_b"} <= ids


# --- relation pruning: no dangling --------------------------------------


def test_dangling_relations_pruned():
    """A relation to an unselected entity is dropped."""
    e1 = _entity("e1", "Alice", days_old=0)
    e2 = _entity("e2", "Bob", days_old=0)
    # e3 has a huge description so it won't be selected under a tight budget.
    e3 = _entity("e3", "Ghost", desc="x" * 5000, days_old=500)
    rel_good = _relation("rg", "e1", "knows", "e2")
    rel_dangle = _relation("rd", "e1", "haunted_by", "e3")
    budget = bb.char_entity(e1) + bb.char_entity(e2) + bb.char_relation(
        RenderedRelation("Alice", "knows", "Bob")
    ) + 5
    sel = _select(
        query="alice bob",
        query_entities=[e1, e2, e3],
        query_raw_relations=[rel_good, rel_dangle],
        recall_budget=budget,
    )
    rel_ids = {s.relation.id for s in sel.query_items.relations}
    assert "rd" not in rel_ids  # e3 not selected -> dangling -> pruned
    sel_entity_ids = {s.entity.id for s in sel.query_items.entities}
    assert "e3" not in sel_entity_ids


# --- relation pruning: top-K cap ----------------------------------------


def test_top_k_cap(monkeypatch):
    monkeypatch.setattr(bb, "MAX_RELATIONS", 3)
    entities = [_entity(f"e{i}", f"Ent{i}") for i in range(8)]
    # star graph: e0 connected to everyone -> all non-dangling
    rels = [_relation(f"r{i}", "e0", "rel", f"e{i}") for i in range(1, 8)]
    sel = _select(
        query="",
        query_entities=entities,
        query_raw_relations=rels,
        recall_budget=100_000,  # generous so cap, not budget, limits
    )
    assert len(sel.query_items.relations) <= 3


# --- profile alpha = 0 (query independence) -----------------------------


def test_profile_query_independent():
    """Profile ordering is identical regardless of the query (alpha forced 0)."""
    units = [
        _unit("a", "springfield property budget", days_old=10),
        _unit("b", "penguins antarctica ice", days_old=10),
    ]
    sel_q1 = _select(query="tell me about springfield property", profile_units=list(units), profile_budget=10_000)
    sel_q2 = _select(query="tell me about penguins", profile_units=list(units), profile_budget=10_000)
    order1 = [s.unit.id for s in sel_q1.profile_items.units]
    order2 = [s.unit.id for s in sel_q2.profile_items.units]
    assert order1 == order2


def test_profile_relevance_dropped_in_score():
    u = _unit("u", "springfield property budget")
    words = bb._content_words("springfield property budget")
    # alpha=0 -> score independent of query word overlap
    assert bb._score_unit(u, words, NOW, _NO_NORMS, alpha=0.0, relevance_by_id={}) == bb._score_unit(
        u, set(), NOW, _NO_NORMS, alpha=0.0, relevance_by_id={}
    )


# --- dedup between profile and query ------------------------------------


def test_dedup_profile_excluded_from_query():
    shared = _unit("shared", "shared identity fact about springfield")
    qonly = _unit("qonly", "springfield specific query thing")
    sel = _select(
        query="springfield",
        profile_units=[shared],
        query_units=[shared, qonly],
        profile_budget=10_000,
        recall_budget=10_000,
    )
    profile_ids = {s.unit.id for s in sel.profile_items.units}
    query_ids = {s.unit.id for s in sel.query_items.units}
    assert "shared" in profile_ids
    assert "shared" not in query_ids  # deduped out of query recall
    assert "qonly" in query_ids


def test_dedup_entity_across_blocks():
    e = _entity("e1", "Springfield")
    sel = _select(
        query="springfield",
        profile_entities=[e],
        query_entities=[e],
        profile_budget=10_000,
        recall_budget=10_000,
    )
    assert any(s.entity.id == "e1" for s in sel.profile_items.entities)
    assert all(s.entity.id != "e1" for s in sel.query_items.entities)


# --- char estimation matches the renderer shape -------------------------


def test_char_estimation_matches_renderer_lines():
    u = _unit("u", "hello world")
    e = _entity("e", "Springfield", desc="capital", wikidata="Q0000002")
    rr = RenderedRelation("Alice", "knows", "Bob")
    assert bb.char_unit(u) == len("- (semantic) hello world")
    assert bb.char_entity(e) == len("- Springfield (Q0000002): capital")
    assert bb.char_relation(rr) == len("- Alice -[knows]-> Bob")


# --- selection shape -----------------------------------------------------


def test_selection_shape_carries_payload():
    e = _entity("e1", "Alice")
    e2 = _entity("e2", "Bob")
    u = _unit("u1", "a fact")
    r = _relation("r1", "e1", "knows", "e2")
    sel = _select(
        query="alice",
        query_units=[u],
        query_entities=[e, e2],
        query_raw_relations=[r],
        recall_budget=10_000,
    )
    assert isinstance(sel, bb.BriefingSelection)
    assert sel.query_items.units[0].unit is u
    labels = {s.entity.label for s in sel.query_items.entities}
    assert {"Alice", "Bob"} <= labels
    assert sel.query_items.relations[0].rendered.subject == "Alice"


# --- validity window (valid_until) ---------------------------------------


def test_expired_valid_until_downweights_unit():
    """A unit whose validity window has closed (valid_until in the past) scores
    below an otherwise-identical still-valid unit. Without this an explicitly
    expired 'no longer available' fact keeps its full salience/recency score and
    outranks the fresher unit that superseded it (the stale-status pathology)."""
    stale = _unit("u1", "tracking three properties")
    stale.valid_until = NOW - timedelta(days=1)
    fresh = _unit("u2", "tracking three properties")  # identical text, no expiry
    v_stale = bb._score_unit(stale, set(), NOW, _NO_NORMS, 0.0, {})
    v_fresh = bb._score_unit(fresh, set(), NOW, _NO_NORMS, 0.0, {})
    assert v_stale < v_fresh
    assert v_stale == v_fresh * bb.EXPIRED_VALIDITY_FACTOR


def test_future_valid_until_not_penalised():
    """An open validity window (valid_until in the future) is scored normally."""
    open_window = _unit("u1", "still tracking")
    open_window.valid_until = NOW + timedelta(days=30)
    no_window = _unit("u2", "still tracking")
    assert bb._score_unit(open_window, set(), NOW, _NO_NORMS, 0.0, {}) == bb._score_unit(
        no_window, set(), NOW, _NO_NORMS, 0.0, {}
    )


def test_expired_valid_until_downweights_relation():
    """The same expiry penalty applies to relations (valid_until on Relation)."""
    stale = _relation("r1", "e1", "tracking", "e2")
    stale.valid_until = NOW - timedelta(days=1)
    fresh = _relation("r2", "e1", "tracking", "e2")
    rr = RenderedRelation(subject="A", predicate="tracking", object="B")
    v_stale = bb._score_relation(stale, rr, set(), NOW, {}, _NO_NORMS, 0.0, {})
    v_fresh = bb._score_relation(fresh, rr, set(), NOW, {}, _NO_NORMS, 0.0, {})
    assert v_stale < v_fresh
    assert v_stale == v_fresh * bb.EXPIRED_VALIDITY_FACTOR
