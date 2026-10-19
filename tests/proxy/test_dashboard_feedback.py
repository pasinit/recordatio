"""Tests for the dashboard memory-feedback card routes.

The dashboard sits behind Caddy basic-auth, so these routes carry no bearer
gate (like the other ``/…/data`` routes). Covered: the operator data view
across users, apply (mutates + marks), dismiss, the decided/unknown guards,
and that the surface is absent when no feedback store is wired.
"""
import uuid
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from tpasini.recordatio.models import KnowledgeType, KnowledgeUnit
from tpasini.recordatio.proxy.cost_log import CostLog
from tpasini.recordatio.proxy.dashboard_routes import register_dashboard_routes
from tpasini.recordatio.storage.base import SQLiteStore
from tpasini.recordatio.storage.memory_feedback import MemoryFeedbackStore

U1 = str(uuid.uuid4())
U2 = str(uuid.uuid4())


def _unit(uid: str, content: str) -> KnowledgeUnit:
    at = datetime(2026, 7, 31, 9, 0, tzinfo=UTC)
    return KnowledgeUnit(id=uid, type=KnowledgeType.SEMANTIC, content=content,
                         asserted_at=at, valid_from=at, provenance="test")


@pytest.fixture()
def env(tmp_path: Path):
    store = SQLiteStore(tmp_path / "mem.db")
    store.add(_unit(U1, "Sam is a analyst"), "sam/scout")
    store.add(_unit(U2, "dup"), "sam/scout")
    fb = MemoryFeedbackStore(tmp_path / "fb.db")
    fid = fb.record(user_id="sam", day="2026-07-31", summary="a dup",
                    findings=[{"kind": "duplicate", "detail": "u2 dups u1"}],
                    ops=[{"op_type": "tombstone", "target": {"unit_id": U2}, "reason": "dup of u1"}])
    op_id = fb.list_ops(fid)[0].id
    app = FastAPI()
    register_dashboard_routes(
        app, get_store=lambda: store, get_cost_log=lambda: CostLog(tmp_path / "cost.db"),
        get_feedback_store=lambda: fb,
    )
    return app, store, fb, op_id


def _active(store: SQLiteStore) -> set[str]:
    return {u.id for u, _ in store.list_units_for_user("sam")}


async def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


async def test_feedback_page_and_data(env) -> None:
    app, store, fb, op_id = env
    async with await _client(app) as c:
        page = await c.get("/feedback")
        assert page.status_code == 200 and "memory feedback" in page.text
        data = await c.get("/feedback/data")
    assert data.status_code == 200
    recs = data.json()["records"]
    assert len(recs) == 1
    assert recs[0]["user_id"] == "sam"
    assert recs[0]["ops"][0]["op_type"] == "tombstone"
    assert recs[0]["ops"][0]["status"] == "pending"


async def test_apply_button_mutates_and_marks(env) -> None:
    app, store, fb, op_id = env
    assert U2 in _active(store)
    async with await _client(app) as c:
        resp = await c.post(f"/feedback/ops/{op_id}/apply", json={})
    assert resp.status_code == 200 and resp.json()["status"] == "applied"
    assert U2 not in _active(store)  # tombstoned
    assert fb.get_op(op_id).status == "applied"


async def test_dismiss_button_marks_without_mutating(env) -> None:
    app, store, fb, op_id = env
    async with await _client(app) as c:
        resp = await c.post(f"/feedback/ops/{op_id}/dismiss", json={"reason": "not a dup"})
    assert resp.status_code == 200 and resp.json()["status"] == "dismissed"
    assert U2 in _active(store)  # untouched
    assert fb.get_op(op_id).decided_result == "not a dup"


async def test_apply_all_applies_valid_and_auto_rejects_invalid(env) -> None:
    app, store, fb, op_id = env
    # Add a second valid op (correct) + an invalid one (truncated uuid).
    fb.record(user_id="sam", day="2026-07-30", summary="fixes", findings=[], ops=[
        {"op_type": "correct", "target": {"unit_id": U1, "new_content": "Sam, Senior Analyst"}, "reason": "title"},
        {"op_type": "tombstone", "target": {"unit_id": U2[:8]}, "reason": "truncated id"},
    ])
    async with await _client(app) as c:
        resp = await c.post("/feedback/apply-all", json={})
    assert resp.status_code == 200
    body = resp.json()
    assert body["applied"] == 2  # the tombstone (env) + the correct
    assert body["rejected"] == 1  # the truncated-uuid op
    statuses = {o.status for rec in fb.list_all() for o in fb.list_ops(rec.id)}
    assert statuses == {"applied", "dismissed"}  # nothing left pending


async def test_apply_all_absent_without_store(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "m.db")
    app = FastAPI()
    register_dashboard_routes(app, get_store=lambda: store, get_cost_log=lambda: CostLog(tmp_path / "c.db"))
    async with await _client(app) as c:
        assert (await c.post("/feedback/apply-all", json={})).status_code == 404


def test_list_all_not_capped_at_60(tmp_path: Path) -> None:
    # Regression: a 60-row default once undercounted a 99-day backfill.
    from datetime import date, timedelta
    fb = MemoryFeedbackStore(tmp_path / "fb.db")
    d0 = date(2026, 1, 1)
    for i in range(65):
        fb.record(user_id="u", day=(d0 + timedelta(days=i)).isoformat(), summary="s", findings=[], ops=[])
    assert len(fb.list_all()) == 65


async def test_decided_and_unknown_guards(env) -> None:
    app, store, fb, op_id = env
    async with await _client(app) as c:
        assert (await c.post(f"/feedback/ops/{op_id}/apply", json={})).status_code == 200
        # second apply on the now-decided op → 409
        assert (await c.post(f"/feedback/ops/{op_id}/apply", json={})).status_code == 409
        assert (await c.post("/feedback/ops/9999/apply", json={})).status_code == 404


async def test_routes_absent_without_feedback_store(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "m.db")
    app = FastAPI()
    register_dashboard_routes(app, get_store=lambda: store, get_cost_log=lambda: CostLog(tmp_path / "c.db"))
    async with await _client(app) as c:
        assert (await c.get("/feedback")).status_code == 404
        assert (await c.get("/feedback/data")).status_code == 404
