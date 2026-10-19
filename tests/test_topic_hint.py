"""`infer_topic_hint` — whole-word scoring across all topics.

Regression coverage for the substring/first-3-keyword/first-hit bug: a
property-search query used to be tagged with a diplomacy topic because an
incidental long word from the diplomacy description occurred as a substring of
the query. Matching now happens on lemmatized content words (the same pipeline
as the keyword index) and the best-scoring topic wins, so incidental overlaps
no longer misclassify and an early broad topic no longer shadows a better one.
"""

import uuid
from datetime import UTC, datetime

from tpasini.recordatio.models import Topic
from tpasini.recordatio.proxy.retrieval import infer_topic_hint
from tpasini.recordatio.storage.base import SQLiteStore

NOW = datetime(2026, 6, 10, 12, 0, 0, tzinfo=UTC)


def _store(tmp_path) -> SQLiteStore:
    """A store whose only topics are the ones the test adds (drop the seeds)."""
    store = SQLiteStore(tmp_path / "hint.db")
    store._conn.execute("DELETE FROM topics")
    store._conn.commit()
    return store


def _topic(store: SQLiteStore, name: str, description: str) -> None:
    store.add_topic(Topic(id=str(uuid.uuid4()), name=name, description=description, source="seed", created_at=NOW))


def test_property_query_does_not_return_diplomacy_topic(tmp_path):
    """The reported misclassification: a housing query must not pick a
    diplomacy topic just because one of its first-3 description words happens
    to be an incidental match. Diplomacy is listed first (old first-hit logic
    would return it on the "central" overlap); scoring picks the topic that
    actually shares most of the query's words."""
    store = _store(tmp_path)
    _topic(store, "multilateral_diplomacy", "Central diplomatic relations and treaties between states")
    _topic(store, "property_search", "Rental flats and apartments to let near metro stations in Springfield")
    hint = infer_topic_hint("Where can I find a rental flat near the metro in downtown Springfield", store)
    assert hint == "property_search"


def test_clearly_matching_query_returns_its_topic(tmp_path):
    store = _store(tmp_path)
    _topic(store, "multilateral_diplomacy", "Coordination between states on treaties and international relations")
    hint = infer_topic_hint("What is the latest on the multilateral diplomacy talks?", store)
    assert hint == "multilateral_diplomacy"


def test_no_topic_clears_threshold_returns_none(tmp_path):
    store = _store(tmp_path)
    _topic(store, "property_search", "Rental and property listings to rent in Springfield")
    assert infer_topic_hint("What time is the dentist appointment tomorrow?", store) is None


def test_best_scoring_topic_wins_over_earlier_weaker_match(tmp_path):
    """Two topics share one word with the query; the one sharing more wins,
    regardless of iteration order (early-topic-wins would pick the first)."""
    store = _store(tmp_path)
    _topic(store, "general_news", "Daily news and current Springfield events")
    _topic(store, "property_search", "Property rental listings and flats to rent in Springfield")
    hint = infer_topic_hint("property rental listings in Springfield", store)
    assert hint == "property_search"


def test_topic_name_match_beyond_first_three_description_words(tmp_path):
    """The old code only scanned the first 3 long description words; a match on
    a later description word (or the name) must still be found."""
    store = _store(tmp_path)
    _topic(
        store,
        "logistics",
        "Warehousing packaging inventory shipping and international freight forwarding",
    )
    # "freight" is the 7th long word — past the old [:3] window.
    assert infer_topic_hint("track the freight shipment", store) == "logistics"


def test_no_topics_returns_none(tmp_path):
    store = _store(tmp_path)
    assert infer_topic_hint("anything at all", store) is None
