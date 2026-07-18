"""Per-block feedback: stable ids, upsert semantics, and the export join (Feature 4).

The stable-id tests carry the weight. A drifting id orphans a rating silently —
the star just stops showing up, with no error anywhere — so the invariants that
prevent drift (mint-once, per-kind ordinals, no ordinal burned by a card that
never renders) each get their own test.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

# The SSE serializer lives under ui-pro/, which is not a package root.
_UI_PRO = str(Path(__file__).resolve().parents[2] / "ui-pro")
if _UI_PRO not in sys.path:
    sys.path.insert(0, _UI_PRO)

from api.serializers.data_card import _build_data_card_event  # noqa: E402

from services import analytics_service as analytics_module  # noqa: E402
from services.analytics_service import AnalyticsService  # noqa: E402
from services.block_identity import (  # noqa: E402
    RATEABLE_BLOCK_KINDS,
    BlockIdAllocator,
    compute_block_id,
)
from services.eval_export_service import build_eval_export, to_jsonl  # noqa: E402


@pytest.fixture
def analytics(monkeypatch, tmp_path):
    monkeypatch.setattr(analytics_module, "_LOCAL_DB", str(tmp_path / "analytics.db"))
    return AnalyticsService()


# ── Stable block identity ────────────────────────────────────────


def test_block_id_is_pure():
    """Same coordinates -> same id, across processes and forever. This is what
    lets a rating cast today still resolve after a redeploy."""
    a = compute_block_id("conv-1", "run-1", "data", 0)
    b = compute_block_id("conv-1", "run-1", "data", 0)
    assert a == b
    assert len(a) == 40  # sha1 hex


def test_block_id_varies_on_every_coordinate():
    base = compute_block_id("conv-1", "run-1", "data", 0)
    assert compute_block_id("conv-2", "run-1", "data", 0) != base
    assert compute_block_id("conv-1", "run-2", "data", 0) != base
    assert compute_block_id("conv-1", "run-1", "plotly", 0) != base
    assert compute_block_id("conv-1", "run-1", "data", 1) != base


def test_allocator_counts_per_kind_not_globally():
    """A figure emitted mid-stream must not renumber the table that lands at the
    end of the turn — otherwise interleaving changes ids and orphans ratings."""
    alloc = BlockIdAllocator("conv-1", "run-1")
    first_data = alloc.next("data")
    alloc.next("image")          # an unrelated kind lands between the two tables
    alloc.next("plotly")
    second_data = alloc.next("data")

    assert first_data == compute_block_id("conv-1", "run-1", "data", 0)
    assert second_data == compute_block_id("conv-1", "run-1", "data", 1)


def test_allocator_ids_are_unique_within_a_turn():
    alloc = BlockIdAllocator("conv-1", "run-1")
    ids = [alloc.next(kind) for kind in RATEABLE_BLOCK_KINDS for _ in range(3)]
    assert len(ids) == len(set(ids))


def test_two_turns_never_collide():
    """run_id is uuid4 per turn, so the same ordinal in two turns is distinct."""
    a = BlockIdAllocator("conv-1", "run-A")
    b = BlockIdAllocator("conv-1", "run-B")
    assert a.next("data") != b.next("data")


# ── Emission paths: eager vs. done ───────────────────────────────


def _result(rows=2):
    return {
        "type": "data",
        "data": pd.DataFrame({"ra": [1.0] * rows, "dec": [2.0] * rows}),
        "tool_name": "datalab_query",
    }


def test_emitted_event_and_persisted_dict_carry_the_same_id():
    """The crux round-trip: the id on the wire IS the id in rich_meta. Replay
    reads the stored value, so these agreeing is what makes a reload show the
    same stars."""
    alloc = BlockIdAllocator("conv-1", "run-1")
    event_str, rich = _build_data_card_event(
        _result(), "user-1", block_id_factory=lambda: alloc.next("data")
    )
    payload = json.loads(event_str[len("data: "):])

    assert payload["blockId"] == rich["blockId"]
    assert payload["blockId"] == compute_block_id("conv-1", "run-1", "data", 0)
    assert rich["blockKind"] == "data"


def test_data_cards_number_in_emission_order_across_both_paths():
    """sse.py shares ONE allocator between the eager (mid-stream) and done
    (end-of-turn) emission sites. Whichever site emits first gets data-0, and
    the id is persisted either way — so the two paths cannot disagree."""
    alloc = BlockIdAllocator("conv-1", "run-1")

    # eager site
    eager_event, eager_rich = _build_data_card_event(
        _result(), "user-1", block_id_factory=lambda: alloc.next("data")
    )
    # done site, same turn, same allocator
    done_event, done_rich = _build_data_card_event(
        _result(), "user-1", block_id_factory=lambda: alloc.next("data")
    )

    assert eager_rich["blockId"] == compute_block_id("conv-1", "run-1", "data", 0)
    assert done_rich["blockId"] == compute_block_id("conv-1", "run-1", "data", 1)
    assert eager_rich["blockId"] != done_rich["blockId"]
    assert json.loads(eager_event[len("data: "):])["blockId"] == eager_rich["blockId"]
    assert json.loads(done_event[len("data: "):])["blockId"] == done_rich["blockId"]


def test_card_that_never_renders_does_not_burn_an_ordinal():
    """The factory is called at the stamp site, not at call time. A result the
    serializer rejects must leave the counter untouched, or the next real card
    would be numbered around a card the user never saw."""
    alloc = BlockIdAllocator("conv-1", "run-1")

    # Not a data result — the serializer bails before stamping.
    assert _build_data_card_event(
        {"type": "papers", "papers": []}, "user-1",
        block_id_factory=lambda: alloc.next("data"),
    ) is None
    assert alloc.peek("data") == 0

    _, rich = _build_data_card_event(
        _result(), "user-1", block_id_factory=lambda: alloc.next("data")
    )
    assert rich["blockId"] == compute_block_id("conv-1", "run-1", "data", 0)


def test_no_factory_means_no_block_fields():
    """Back-compat: callers that don't opt in emit exactly what they did before."""
    event_str, rich = _build_data_card_event(_result(), "user-1")
    payload = json.loads(event_str[len("data: "):])
    assert "blockId" not in payload
    assert "blockId" not in rich


# ── log_block_feedback: upsert + validation ──────────────────────


def test_rerating_replaces_rather_than_appends(analytics):
    analytics.log_block_feedback(
        block_id="b1", rating=4, run_id="run-1", conversation_id="conv-1",
        user_id="u1", block_kind="data",
    )
    analytics.log_block_feedback(
        block_id="b1", rating=2, run_id="run-1", conversation_id="conv-1",
        user_id="u1", block_kind="data", comment="actually the units are wrong",
    )

    rows = analytics.export_block_feedback()
    assert len(rows) == 1
    assert rows[0]["rating"] == 2
    assert rows[0]["comment"] == "actually the units are wrong"


def test_two_users_rate_the_same_block_independently(analytics):
    analytics.log_block_feedback(block_id="b1", rating=5, user_id="u1", conversation_id="c1")
    analytics.log_block_feedback(block_id="b1", rating=1, user_id="u2", conversation_id="c1")

    rows = analytics.export_block_feedback()
    assert len(rows) == 2
    assert {r["rating"] for r in rows} == {1, 5}
    assert analytics.export_block_feedback(user_id="u2")[0]["rating"] == 1


@pytest.mark.parametrize("bad", [0, 6, -1, 99])
def test_out_of_range_rating_is_rejected(analytics, bad):
    """A 0 or an 11 in the label set is worse than a missing label."""
    with pytest.raises(ValueError):
        analytics.log_block_feedback(block_id="b1", rating=bad, user_id="u1")
    assert analytics.export_block_feedback() == []


@pytest.mark.parametrize("bad", ["", None])
def test_missing_block_id_is_rejected(analytics, bad):
    with pytest.raises(ValueError):
        analytics.log_block_feedback(block_id=bad, rating=3, user_id="u1")


@pytest.mark.parametrize("bad", ["great", "4", 4.9, 4.0, True, False, None])
def test_non_integer_rating_is_rejected(analytics, bad):
    """CX-05: int() would coerce 4.9 -> 4 and JSON true -> 1, landing a human
    judgement in the label set that nobody made. bool subclasses int, so it has
    to be excluded explicitly."""
    with pytest.raises(ValueError):
        analytics.log_block_feedback(block_id="b1", rating=bad, user_id="u1")
    assert analytics.export_block_feedback() == []


def test_get_block_ratings_is_scoped_to_conversation_and_user(analytics):
    analytics.log_block_feedback(block_id="b1", rating=4, conversation_id="c1", user_id="u1")
    analytics.log_block_feedback(block_id="b2", rating=5, conversation_id="c2", user_id="u1")
    analytics.log_block_feedback(block_id="b3", rating=1, conversation_id="c1", user_id="u2")

    ratings = analytics.get_block_ratings("c1", "u1")
    assert set(ratings) == {"b1"}
    assert ratings["b1"]["rating"] == 4
    assert analytics.get_block_ratings("", "u1") == {}


def test_existing_like_dislike_feedback_still_works(analytics):
    """block_feedback is additive — /api/feedback must not regress."""
    analytics.log_feedback(message_id="m1", feedback="like", user_id="u1")
    analytics.log_block_feedback(block_id="b1", rating=4, user_id="u1")

    assert len(analytics.export_feedback_json()) == 1
    assert len(analytics.export_block_feedback()) == 1


# ── Export join ──────────────────────────────────────────────────


class _FakeConversations:
    def __init__(self, messages):
        self._messages = messages

    def get_conversation_messages(self, conversation_id):
        return self._messages.get(conversation_id, [])


class _FakeRuns:
    def __init__(self, runs):
        self._runs = runs

    def get_run(self, run_id, *, user_id=None):
        return self._runs.get(run_id)


def _turn_messages(text_block_id="tb", data_block_id="db"):
    return [
        {"role": "user", "content": "cone search gaia_dr3 around M13"},
        {
            "role": "assistant",
            "content": "Here are 2 sources near M13.",
            "metadata": {
                "runMeta": {"run_id": "run-1", "model": "gpt-5.4", "text_block_id": text_block_id},
                "dataTables": [{
                    "blockId": data_block_id,
                    "columns": ["ra", "dec"],
                    "sourceName": "gaia_dr3",
                    "totalRows": 2,
                    "displayedRows": 2,
                    "truncated": False,
                    "request": {"kind": "adql", "text": "SELECT ra, dec FROM gaia_dr3"},
                }],
            },
        },
    ]


def _run_row():
    return {
        "id": "run-1", "model": "gpt-5.4", "provider": "openai", "status": "completed",
        "input_tokens": 1200, "output_tokens": 300, "total_tokens": 1500,
        "cost_usd": 0.0042, "tools_called": ["datalab_query"],
    }


def test_export_joins_rating_to_prompt_query_tokens_and_cost(analytics):
    analytics.log_block_feedback(
        block_id="db", rating=4, run_id="run-1", conversation_id="c1",
        user_id="u1", block_kind="data", comment="right table, right units",
    )

    rows = build_eval_export(
        analytics,
        _FakeConversations({"c1": _turn_messages()}),
        _FakeRuns({"run-1": _run_row()}),
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["rating"] == 4
    assert row["comment"] == "right table, right units"
    assert row["prompt"] == "cone search gaia_dr3 around M13"
    assert row["block_kind"] == "data"
    assert row["block_content"]["source_name"] == "gaia_dr3"
    assert row["block_content"]["columns"] == ["ra", "dec"]
    # The exact query behind the rated block (Feature 1's provenance)
    assert row["block_query"]["text"] == "SELECT ra, dec FROM gaia_dr3"
    # Phase 3's cost accounting, joined in
    assert row["model"] == "gpt-5.4"
    assert row["total_tokens"] == 1500
    assert row["cost_usd"] == 0.0042
    assert row["block_resolved"] is True


def test_export_resolves_a_rated_text_block(analytics):
    analytics.log_block_feedback(
        block_id="tb", rating=5, run_id="run-1", conversation_id="c1",
        user_id="u1", block_kind="text",
    )

    rows = build_eval_export(
        analytics, _FakeConversations({"c1": _turn_messages()}), _FakeRuns({"run-1": _run_row()})
    )
    assert rows[0]["block_kind"] == "text"
    assert rows[0]["block_content"]["text"] == "Here are 2 sources near M13."
    assert rows[0]["prompt"] == "cone search gaia_dr3 around M13"


def test_export_survives_a_deleted_conversation(analytics):
    """The label is the artifact worth keeping — a rating whose turn is gone
    still exports, flagged unresolved, rather than vanishing or throwing."""
    analytics.log_block_feedback(
        block_id="gone", rating=3, run_id="run-x", conversation_id="deleted",
        user_id="u1", block_kind="data",
    )

    rows = build_eval_export(analytics, _FakeConversations({}), _FakeRuns({}))
    assert len(rows) == 1
    assert rows[0]["block_resolved"] is False
    assert rows[0]["rating"] == 3
    assert rows[0]["total_tokens"] is None


def test_export_resolves_every_grid_of_a_two_search_turn(analytics):
    """CX-03: a turn that searches papers twice used to overwrite the singular
    rich_meta field, so the FIRST grid's rating could never be resolved again.
    papersGroups keeps them all."""
    messages = [
        {"role": "user", "content": "papers on NGC 253, then on M13"},
        {
            "role": "assistant",
            "content": "Two searches.",
            "metadata": {
                "runMeta": {"run_id": "run-1", "text_block_id": "tb"},
                # singular fields = the LAST grid only (back-compat)
                "papers": [{"title": "M13 paper", "bibcode": "B2"}],
                "papersBlockId": "pp1",
                "papersGroups": [
                    {"papers": [{"title": "NGC 253 paper", "bibcode": "B1"}],
                     "blockId": "pp0", "request": {"kind": "ads", "q": "NGC 253"}},
                    {"papers": [{"title": "M13 paper", "bibcode": "B2"}],
                     "blockId": "pp1", "request": {"kind": "ads", "q": "M13"}},
                ],
            },
        },
    ]
    for block in ("pp0", "pp1"):
        analytics.log_block_feedback(
            block_id=block, rating=4, run_id="run-1", conversation_id="c1",
            user_id="u1", block_kind="papers",
        )

    rows = build_eval_export(
        analytics, _FakeConversations({"c1": messages}), _FakeRuns({"run-1": _run_row()})
    )
    by_block = {r["block_id"]: r for r in rows}
    assert by_block["pp0"]["block_resolved"] is True
    assert by_block["pp1"]["block_resolved"] is True
    # Each grid keeps its OWN query and content, not the survivor's.
    assert by_block["pp0"]["block_query"]["q"] == "NGC 253"
    assert by_block["pp1"]["block_query"]["q"] == "M13"
    assert by_block["pp0"]["block_content"]["papers"][0]["bibcode"] == "B1"


def test_export_resolves_every_notebook_of_a_conductor_turn(analytics):
    """CX-03, notebook half: Conductor emits its own notebook alongside the
    normal path, and the singular field only kept the last."""
    messages = [
        {"role": "user", "content": "analyze"},
        {
            "role": "assistant",
            "content": "Done.",
            "metadata": {
                "runMeta": {"run_id": "run-1", "text_block_id": "tb"},
                "notebook": {"title": "Second", "data": {"cells": [1, 2]}, "blockId": "nb1"},
                "notebooks": [
                    {"title": "First", "data": {"cells": [1]}, "blockId": "nb0"},
                    {"title": "Second", "data": {"cells": [1, 2]}, "blockId": "nb1"},
                ],
            },
        },
    ]
    analytics.log_block_feedback(block_id="nb0", rating=2, run_id="run-1",
                                conversation_id="c1", user_id="u1", block_kind="notebook")

    rows = build_eval_export(
        analytics, _FakeConversations({"c1": messages}), _FakeRuns({"run-1": _run_row()})
    )
    assert rows[0]["block_resolved"] is True
    assert rows[0]["block_content"]["title"] == "First"


def test_export_scopes_to_one_rater(analytics):
    analytics.log_block_feedback(block_id="db", rating=4, conversation_id="c1", user_id="u1")
    analytics.log_block_feedback(block_id="db", rating=1, conversation_id="c1", user_id="u2")

    rows = build_eval_export(
        analytics, _FakeConversations({"c1": _turn_messages()}), _FakeRuns({}), user_id="u1"
    )
    assert len(rows) == 1
    assert rows[0]["rated_by"] == "u1"


def test_export_reads_each_conversation_and_run_once(analytics):
    """N ratings on one turn must not mean N history reads — the export is an
    admin endpoint over the whole label set."""
    for i, block in enumerate(["tb", "db"]):
        analytics.log_block_feedback(
            block_id=block, rating=i + 3, run_id="run-1", conversation_id="c1", user_id="u1",
        )

    conversations = _FakeConversations({"c1": _turn_messages()})
    runs = _FakeRuns({"run-1": _run_row()})
    calls = {"conv": 0, "run": 0}
    inner_conv, inner_run = conversations.get_conversation_messages, runs.get_run
    conversations.get_conversation_messages = lambda cid: (calls.__setitem__("conv", calls["conv"] + 1), inner_conv(cid))[1]
    runs.get_run = lambda rid, **kw: (calls.__setitem__("run", calls["run"] + 1), inner_run(rid, **kw))[1]

    rows = build_eval_export(analytics, conversations, runs)
    assert len(rows) == 2
    assert calls == {"conv": 1, "run": 1}


def test_jsonl_is_one_object_per_line(analytics):
    analytics.log_block_feedback(block_id="tb", rating=5, run_id="run-1", conversation_id="c1", user_id="u1")
    analytics.log_block_feedback(block_id="db", rating=4, run_id="run-1", conversation_id="c1", user_id="u1")

    rows = build_eval_export(
        analytics, _FakeConversations({"c1": _turn_messages()}), _FakeRuns({"run-1": _run_row()})
    )
    lines = to_jsonl(rows).splitlines()
    assert len(lines) == 2
    for line in lines:
        assert json.loads(line)["rating"] in (4, 5)


# ── Router auth (UIAPI-12) ───────────────────────────────────────
# The write endpoints ARE the intake for the human-label eval dataset:
# anonymous writers polluted the export and collided on one shared user_id.
# These tests pin the fix at the HTTP surface — unauthenticated POSTs are
# rejected, authenticated ones persist under the caller's real user id.


@pytest.fixture
def api_client(monkeypatch):
    # Hermetic: importing api.* load_dotenv()s the owner's real env; the Data
    # Lab token must never leak into an api-importing test (repo convention —
    # see tests/unit/test_datalab_p0.py).
    monkeypatch.delenv("DATALAB_TOKEN", raising=False)

    import os
    import sys as _sys

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for _p in (repo, os.path.join(repo, "ui-pro")):
        if _p not in _sys.path:
            _sys.path.insert(0, _p)

    from fastapi.testclient import TestClient

    import api.main as main_mod
    from api.deps import get_current_user

    client = TestClient(main_mod.app)
    try:
        yield client, main_mod, get_current_user
    finally:
        main_mod.app.dependency_overrides.pop(get_current_user, None)


def test_block_feedback_post_requires_auth(api_client):
    client, _, _ = api_client
    resp = client.post("/api/block-feedback", json={"block_id": "b1", "rating": 4})
    assert resp.status_code == 401


def test_feedback_post_requires_auth(api_client):
    client, _, _ = api_client
    resp = client.post("/api/feedback", json={"message_id": "m1", "feedback": "like"})
    assert resp.status_code == 401


def test_block_feedback_post_persists_the_authenticated_caller(api_client, monkeypatch):
    client, main_mod, get_current_user = api_client
    main_mod.app.dependency_overrides[get_current_user] = lambda: {
        "sub": "user-9", "email": "nine@example.com"
    }

    from api.routers import analytics as analytics_router

    recorded = {}
    monkeypatch.setattr(
        analytics_router.analytics_service,
        "log_block_feedback",
        lambda **kwargs: recorded.update(kwargs),
    )

    resp = client.post(
        "/api/block-feedback",
        json={"block_id": "b1", "rating": 4, "conversation_id": "c1", "block_kind": "data"},
    )

    assert resp.status_code == 200
    assert resp.json()["rating"] == 4
    # The row is attributed to the real caller — never 'anonymous'.
    assert recorded["user_id"] == "user-9"
    assert recorded["block_id"] == "b1"


def test_plan_feedback_post_requires_auth(api_client):
    # UIAPI-04 sibling: plan approval was the other unauthenticated writer.
    client, _, _ = api_client
    resp = client.post("/api/plan-feedback", json={"conversation_id": "c1", "approve": True})
    assert resp.status_code == 401


def test_plan_feedback_rejects_a_non_owner(api_client, monkeypatch):
    """UIAPI-04: an authenticated caller must still OWN the conversation —
    a leaked UUID must not let one user approve another user's plan."""
    client, main_mod, get_current_user = api_client
    main_mod.app.dependency_overrides[get_current_user] = lambda: {
        "sub": "attacker", "email": "a@example.com"
    }

    from api.routers import general as general_router

    monkeypatch.setattr(
        general_router.conversation_service,
        "conversation_belongs_to_user",
        lambda conv_id, user_id: False,
    )

    resp = client.post("/api/plan-feedback", json={"conversation_id": "victim-conv", "approve": True})
    # 404, not 403 — do not confirm the conversation exists to a non-owner.
    assert resp.status_code == 404


def test_plan_feedback_owner_reaches_the_run_scoped_queue(api_client, monkeypatch):
    """UIAPI-08: the registry is keyed by (conversation, run_id); a request
    naming the run addresses exactly that run's queue."""
    import queue as stdlib_queue

    client, main_mod, get_current_user = api_client
    main_mod.app.dependency_overrides[get_current_user] = lambda: {
        "sub": "owner", "email": "o@example.com"
    }

    from api.routers import general as general_router

    monkeypatch.setattr(
        general_router.conversation_service,
        "conversation_belongs_to_user",
        lambda conv_id, user_id: True,
    )
    run_a, run_b = stdlib_queue.Queue(), stdlib_queue.Queue()
    with general_router._plan_feedback_lock:
        general_router._plan_feedback_queues[("c1", "run-a")] = run_a
        general_router._plan_feedback_queues[("c1", "run-b")] = run_b
    try:
        resp = client.post(
            "/api/plan-feedback",
            json={"conversation_id": "c1", "run_id": "run-a", "approve": True},
        )
        assert resp.status_code == 200
        assert run_a.get_nowait()["approve"] is True
        assert run_b.empty()  # the concurrent run's queue is untouched
    finally:
        with general_router._plan_feedback_lock:
            general_router._plan_feedback_queues.pop(("c1", "run-a"), None)
            general_router._plan_feedback_queues.pop(("c1", "run-b"), None)
