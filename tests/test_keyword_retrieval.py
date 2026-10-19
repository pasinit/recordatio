"""F1 keyword retrieval — implementer gates R1–R10.

Each test names the gate it proves. Offline, deterministic, fixture-only.
"""

import hashlib
import logging
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from tpasini.recordatio.entity import Entity
from tpasini.recordatio.models import KnowledgeType, KnowledgeUnit
from tpasini.recordatio.proxy import retrieval
from tpasini.recordatio.proxy.briefing_budget import _keyword_or_coverage_relevance
from tpasini.recordatio.proxy.candidate_source import KeywordSource, candidate_cap
from tpasini.recordatio.proxy.retrieval import _materialize_keyword_candidates, retrieve
from tpasini.recordatio.retrieval_text import content_words, normalize_for_index, normalize_numbers, pipeline_tokens
from tpasini.recordatio.storage import base as storage_base
from tpasini.recordatio.storage.base import SQLiteStore

AGENT = "kw-agent"
NOW = datetime(2026, 6, 10, 12, 0, 0, tzinfo=UTC)
_logger = logging.getLogger("test-keyword")


def _unit(store: SQLiteStore, content: str, *, uid: str | None = None, valid_until: datetime | None = None) -> str:
    uid = uid or str(uuid.uuid4())
    store.add(
        KnowledgeUnit(
            id=uid, type=KnowledgeType.SEMANTIC, content=content,
            asserted_at=NOW, valid_from=NOW, valid_until=valid_until, provenance="t",
        ),
        AGENT,
    )
    return uid


def _entity(store: SQLiteStore, label: str, *, desc: str | None = None) -> str:
    eid = str(uuid.uuid4())
    store.add_entity(Entity(id=eid, label=label, description=desc, asserted_at=NOW), AGENT)
    return eid


# --- R6: pipeline parity — every §3.1 number format + EN/IT lemma cases --------


def test_r6_number_formats():
    assert normalize_numbers("£2,000 pcm") == "2000 gbp pcm"
    assert normalize_numbers("rent of 150,000") == "rent of 150000"
    assert normalize_numbers("£150k") == "150000 gbp"
    assert normalize_numbers("$12K") == "12000 usd"
    assert normalize_numbers("€2,500") == "2500 eur"
    # Ranges: each side normalizes; the dash is a tokenizer separator (both dashes).
    assert pipeline_tokens("£10–£20/hour")[:4] == ["10", "gbp", "20", "gbp"]
    assert pipeline_tokens("£150k-£200k") == ["150000", "gbp", "200000", "gbp"]
    # Percentages, dates, times pass through untouched (then tokenize normally).
    assert normalize_numbers("10% on 2026-06-10 at 12:30") == "10% on 2026-06-10 at 12:30"


def test_r6_lemmas_en_it():
    assert pipeline_tokens("properties viewings agencies") == ["property", "viewing", "agency"]
    assert pipeline_tokens("libri gatti") == ["libro", "gatto"]


def test_r6_index_and_query_share_one_code_path():
    # The index stores normalize_for_index(); the query side builds phrases with
    # normalize_for_index() and terms with pipeline_tokens() — same module, same
    # functions (this test pins the contract that they are not duplicated).
    text = "Viewings at £2,000 pcm"
    assert normalize_for_index(text) == " ".join(pipeline_tokens(text))
    assert content_words(text) <= set(pipeline_tokens(text))


# --- R7: phrase parity — inflected surface forms still match -------------------


def test_r7_gazetteer_phrase_matches_inflected_content(tmp_path):
    store = SQLiteStore(tmp_path / "r7.db")
    _entity(store, "Mortgage Advisers")
    uid = _unit(store, "The mortgage adviser recommended a fixed rate.")  # singular in content
    source = KeywordSource(store)
    # Label occurs verbatim (plural) in the query; content is singular: the
    # shared lemmatized pipeline must bridge the inflection on BOTH sides.
    hits = source.candidates("What did the Mortgage Advisers say?", AGENT, 50)
    assert uid in {item_id for item_id, _ in hits}
    expr = source.build_match_expression("What did the Mortgage Advisers say?", AGENT)
    assert '"mortgage adviser"' in expr  # the phrase, lemmatized


def test_r7_phrase_with_number_normalization(tmp_path):
    store = SQLiteStore(tmp_path / "r7b.db")
    uid = _unit(store, "Budget is £150k for the flat.")
    hits = KeywordSource(store).candidates("Is £150,000 enough?", AGENT, 50)
    assert uid in {item_id for item_id, _ in hits}  # 150000 gbp on both sides


# --- R8: cap honored, env override effective, ties broken by item_id ----------


def test_r8_cap_and_env_override(tmp_path, monkeypatch):
    store = SQLiteStore(tmp_path / "r8.db")
    ids = sorted(_unit(store, f"zebra fact number {i}") for i in range(12))
    source = KeywordSource(store)
    # Without store/scopes, candidate_cap() returns _CAP_LOW (256) — the
    # population-relative rule's safe floor (CAP_SCALING_DESIGN.md §3).
    assert candidate_cap() == 256
    # With the store, the rule counts active units (12 here) → clamp(12,256,2000)=256.
    scopes = (AGENT,)
    assert candidate_cap(store, scopes) == 256
    # Env override pins the cap (back-compat with F1 experiments).
    monkeypatch.setenv("RECORDATIO_RETRIEVAL_CANDIDATE_CAP", "5")
    assert candidate_cap() == 5
    hits = source.candidates("tell me about zebra", AGENT, candidate_cap())
    assert len(hits) == 5
    assert ids  # cap respected with deterministic content below


def test_r8_ties_broken_by_item_id(tmp_path):
    store = SQLiteStore(tmp_path / "r8b.db")
    # Identical content -> identical BM25; order must be item_id ascending.
    ids = sorted(_unit(store, "identical aardvark sentence") for _ in range(6))
    hits = KeywordSource(store).candidates("aardvark", AGENT, 4)
    assert [item_id for item_id, _ in hits] == ids[:4]


# --- R10: relevance normalization + coverage fallback ---------------------------


def test_r10_minmax_normalization(tmp_path):
    store = SQLiteStore(tmp_path / "r10.db")
    u_best = _unit(store, "puffin colony on the cliff with many puffin nests puffin")
    u_mid = _unit(store, "a single puffin was seen")
    _unit(store, "nothing relevant at all")
    pairs = KeywordSource(store).candidates("puffin", AGENT, 50)
    units, entities, relevance = _materialize_keyword_candidates(pairs, AGENT, store, None)
    assert entities == []
    assert set(relevance) == {u_best, u_mid}
    assert relevance[u_best] == 1.0 and relevance[u_mid] == 0.0  # min-max over the set
    assert all(0.0 <= v <= 1.0 for v in relevance.values())


def test_r10_degenerate_equal_scores_all_one(tmp_path):
    store = SQLiteStore(tmp_path / "r10b.db")
    a = _unit(store, "walrus sentence equal")
    b = _unit(store, "walrus sentence equal")
    pairs = KeywordSource(store).candidates("walrus", AGENT, 50)
    _units, _entities, relevance = _materialize_keyword_candidates(pairs, AGENT, store, None)
    assert relevance == {a: 1.0, b: 1.0}  # t_max == t_min -> every candidate 1.0


def test_r10_coverage_fallback_only_for_absent_ids():
    rel = {"in-dict": 0.7}
    assert _keyword_or_coverage_relevance("in-dict", "no overlap here", {"x"}, rel) == 0.7
    # Absent id -> lemmatized coverage on the shared pipeline.
    cov = _keyword_or_coverage_relevance("absent", "springfield property", content_words("springfield property"), rel)
    assert cov == 1.0


# --- R4: graceful LIKE fallback (FTS5 unavailable) ------------------------------


async def test_r4_fallback_matches_like_path_and_warns_once(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(storage_base, "_fts5_available", lambda: False)
    monkeypatch.setattr(retrieval, "_like_fallback_warned", False)
    store = SQLiteStore(tmp_path / "r4.db")
    assert store.fts_enabled is False
    matching = _unit(store, "the daily standup moved to 9am")
    _unit(store, "unrelated fact")

    async def _no_wikidata(*_a, **_k):
        return []

    monkeypatch.setattr(retrieval, "wikidata_seed_entities", _no_wikidata)
    with caplog.at_level(logging.WARNING):
        res1 = await retrieve("standup moved", AGENT, store, _logger, "Tommaso")
        res2 = await retrieve("standup moved", AGENT, store, _logger, "Tommaso")
    # Exactly today's behavior: LIKE %query% substring match, no keyword scores.
    assert [u.id for u in res1.units] == [matching]
    assert res1.keyword_relevance == {}
    warnings = [r for r in caplog.records if "FTS5" in r.message]
    assert len(warnings) == 1  # logged once per process, not per query
    assert [u.id for u in res2.units] == [matching]


# --- R5: migration idempotence + DROP/rebuild reproduces the index --------------


def _dump_index(store: SQLiteStore) -> list[tuple]:
    return store._conn.execute(
        f"SELECT item_id, kind, agent_id, content, label, description "
        f"FROM {store._fts_schema}.memento_fts ORDER BY item_id"
    ).fetchall()


def test_r5_open_n_times_one_schema_and_rebuild_reproduces(tmp_path):
    db = tmp_path / "r5.db"
    store = SQLiteStore(db)
    _unit(store, "heron stands in the £1,200 pond")
    _entity(store, "Heron Pond", desc="a pond")
    before = [tuple(r) for r in _dump_index(store)]
    sidecar_schema = [
        tuple(r)
        for r in store._conn.execute("SELECT type, name FROM memfts.sqlite_master ORDER BY name").fetchall()
    ]
    store._conn.close()
    for _ in range(3):
        reopened = SQLiteStore(db)
        schema_now = [
            tuple(r)
            for r in reopened._conn.execute("SELECT type, name FROM memfts.sqlite_master ORDER BY name").fetchall()
        ]
        assert schema_now == sidecar_schema  # one schema, no duplicates/drift
        reopened._conn.close()
    final = SQLiteStore(db)
    assert [tuple(r) for r in _dump_index(final)] == before  # persisted, not rebuilt away
    final.rebuild_fts_index()
    assert [tuple(r) for r in _dump_index(final)] == before  # DROP+rebuild reproduces


def test_r5_write_path_sync_and_canonical_faithfulness(tmp_path):
    store = SQLiteStore(tmp_path / "r5b.db")
    uid = _unit(store, "the first version mentions osprey")
    assert any("osprey" in r["content"] for r in _dump_index(store))
    # INSERT OR REPLACE upsert: index follows the canonical row.
    _unit(store, "the second version mentions kestrel", uid=uid)
    rows = [r for r in _dump_index(store) if r["item_id"] == uid]
    assert len(rows) == 1 and "kestrel" in rows[0]["content"]
    # invalidate() makes no index write; materialization re-applies validity.
    store.invalidate(uid, NOW + timedelta(days=1), AGENT)
    assert any(r["item_id"] == uid for r in _dump_index(store))
    pairs = KeywordSource(store).candidates("kestrel", AGENT, 50)
    assert uid in {i for i, _ in pairs}  # index still serves the id…
    units, _e, _r = _materialize_keyword_candidates(pairs, AGENT, store, None)
    assert units == []  # …but the consumer's bitemporal re-check drops it


# --- R1/R2: artifact hygiene + determinism (fixture-scale) ----------------------


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def test_r1_ephemeral_open_and_query_leave_file_bytes_unchanged(tmp_path, monkeypatch):
    db = tmp_path / "frozen.db"
    build = SQLiteStore(db)
    _unit(build, "Tommaso viewed a flat near Harbour Point for £2,000 pcm")
    _entity(build, "Harbour Point", desc="business district")
    build._conn.close()
    db.with_name(db.name + ".fts").unlink()  # drop the build-time sidecar; the frozen file stands alone
    sha_before = _sha(db)

    async def _no_wikidata(*_a, **_k):
        return []

    monkeypatch.setattr(retrieval, "wikidata_seed_entities", _no_wikidata)
    store = SQLiteStore(db, ephemeral_fts=True)
    res = await retrieve("What flat did I view near Harbour Point?", AGENT, store, _logger, "Tommaso")
    assert res.units  # the keyword source actually ran
    store._conn.close()
    assert _sha(db) == sha_before  # G-1: no byte of the frozen file changed
    assert not db.with_name(db.name + ".fts").exists()  # and no sidecar litter


def test_r2_same_store_same_query_same_candidates(tmp_path):
    store = SQLiteStore(tmp_path / "r2.db")
    for i in range(8):
        _unit(store, f"flamingo item {i} about the lagoon")
    source = KeywordSource(store)
    first = source.candidates("flamingo lagoon", AGENT, 50)
    for _ in range(3):
        assert source.candidates("flamingo lagoon", AGENT, 50) == first


# --- query construction details (§3.3) ------------------------------------------


def test_gazetteer_longest_match_first_non_overlapping(tmp_path):
    store = SQLiteStore(tmp_path / "gz.db")
    _entity(store, "Harbour Point")
    _entity(store, "Harbour")  # shorter label inside the longer one
    source = KeywordSource(store)
    expr = source.build_match_expression("flats in Harbour Point please", AGENT)
    assert '"harbour point"' in expr
    assert '"harbour" OR' not in expr  # consumed by the longer match


def test_capitalization_heuristic_flag(tmp_path, monkeypatch):
    store = SQLiteStore(tmp_path / "cap.db")
    source = KeywordSource(store)
    q = "What about Jane Roe then?"
    assert '"jane roe"' in source.build_match_expression(q, AGENT)
    monkeypatch.setenv("RECORDATIO_RETRIEVAL_CAP_HEURISTIC", "0")
    assert '"jane roe"' not in source.build_match_expression(q, AGENT)


def test_sentence_initial_capital_not_a_phrase(tmp_path):
    store = SQLiteStore(tmp_path / "cap2.db")
    expr = KeywordSource(store).build_match_expression("Springfield Bridge flats. What Jane Roe said.", AGENT)
    # "Springfield Bridge" starts the text -> first token positional; remainder <2 tokens.
    assert '"springfield bridge"' not in expr and '"bridge"' in expr
    assert '"jane roe"' in expr  # "What" absorbed the sentence-initial slot


def test_stopword_only_query_yields_no_candidates(tmp_path):
    store = SQLiteStore(tmp_path / "stop.db")
    _unit(store, "anything at all")
    assert KeywordSource(store).build_match_expression("what is it", AGENT) == ""
    assert KeywordSource(store).candidates("what is it", AGENT, 50) == []
