"""Full-result export integrity (Feature 2).

The property under test is a data-integrity one: an export is either the
COMPLETE result or a loud failure. There is no code path that quietly hands
back the 10k-row display preview in place of the real data, so these tests
assert both halves — that a live id streams every row and every column, and
that a dead id produces an error rather than a plausible-looking short file.
"""

import csv
import io
import os
import sys
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app_client(monkeypatch):
    # `api` lives under ui-pro/. Import inside the fixture so this file has no
    # collection-time side effects — mirrors tests/unit/test_usage_endpoints.py.
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for _p in (repo, os.path.join(repo, "ui-pro")):
        if _p not in sys.path:
            sys.path.insert(0, _p)

    import api.main as main_mod
    from api.deps import get_current_user

    client = TestClient(main_mod.app)

    def _override(user):
        main_mod.app.dependency_overrides[get_current_user] = lambda: user

    _override({"sub": "user-1", "email": "one@example.com"})
    try:
        yield client, main_mod, _override
    finally:
        main_mod.app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture
def store(tmp_path):
    from services.datalab_result_store import DatalabResultStore

    return DatalabResultStore(cache_dir=tmp_path / "results", enable_disk_cache=False)


@pytest.fixture
def use_store(monkeypatch, store):
    """Point both the router and the serializer at an isolated store."""
    import services.datalab_result_store as mod

    monkeypatch.setattr(mod, "default_result_store", lambda: store)
    return store


def _wide_frame(rows: int = 12_000) -> pd.DataFrame:
    """A frame bigger than the card's 10k preview cap, with columns the card's
    display map would drop and a value longer than its 60-char clip."""
    return pd.DataFrame({
        "ra": [10.0 + i * 1e-5 for i in range(rows)],
        "dec": [-20.0 + i * 1e-5 for i in range(rows)],
        "obscure_column_the_card_never_displays": ["x" * 80] * rows,
        "flux_with_full_precision": [1.234567891234 + i for i in range(rows)],
    })


# ── Complete export ───────────────────────────────────────────────────────────

def test_export_streams_every_row(app_client, use_store):
    client, _, _ = app_client
    df = _wide_frame(12_000)
    rid = use_store.put(df, {"source": "test_catalog", "owner_id": "user-1"})

    resp = client.get(f"/api/results/{rid}/export.csv")

    assert resp.status_code == 200
    lines = resp.text.strip().split("\n")
    assert len(lines) == 12_001, "CSV must carry every row plus one header row"
    assert resp.headers["X-Quasar-Rowcount"] == "12000"


def test_export_carries_all_columns_not_the_display_subset(app_client, use_store):
    client, _, _ = app_client
    df = _wide_frame(10)
    rid = use_store.put(df, {"source": "test_catalog", "owner_id": "user-1"})

    resp = client.get(f"/api/results/{rid}/export.csv")

    assert resp.status_code == 200
    header = resp.text.split("\n")[0].strip()
    assert header.split(",") == list(df.columns)
    # The column the card's display map would have dropped is present...
    assert "obscure_column_the_card_never_displays" in header
    # ...and its value is not clipped to the card's 60-char display limit.
    assert "x" * 80 in resp.text


def test_export_does_not_clip_float_precision(app_client, use_store):
    client, _, _ = app_client
    # The card renders floats via _fmt as f"{v:.3f}" — the export must not.
    df = pd.DataFrame({"flux": [1.234567891234]})
    rid = use_store.put(df, {"source": "s", "owner_id": "user-1"})

    resp = client.get(f"/api/results/{rid}/export.csv")

    assert resp.status_code == 200
    assert "1.234567891234" in resp.text


# ── CSV quoting / encoding (f2-CX-12, ties to f2-CX-07) ───────────────────────

def test_export_quotes_hostile_cells_and_headers_losslessly(app_client, use_store):
    """Commas, quotes, embedded newlines, and unicode — in HEADERS and cells —
    must survive a csv.reader round-trip byte-for-byte."""
    client, _, _ = app_client
    hostile_header = 'col,"quoted"\nnewline ✓'
    hostile_cells = ["a,b", 'she said "hi"', "line1\nline2", "ünïcode — ✓"]
    df = pd.DataFrame({hostile_header: hostile_cells, "plain": ["1", "2", "3", "4"]})
    rid = use_store.put(df, {"source": "s", "owner_id": "user-1"})

    resp = client.get(f"/api/results/{rid}/export.csv")

    assert resp.status_code == 200
    parsed = list(csv.reader(io.StringIO(resp.text)))
    assert parsed[0] == [hostile_header, "plain"]
    assert [row[0] for row in parsed[1:]] == hostile_cells


def test_export_renders_every_missing_flavour_as_empty_field(app_client, use_store):
    """pd.NA, pd.NaT, None, and float('nan') all export as EMPTY CSV fields —
    never as the literal '<NA>' / 'NaT' / 'nan' (f2-CX-07)."""
    client, _, _ = app_client
    df = pd.DataFrame({
        "int_na": pd.array([1, pd.NA], dtype="Int64"),
        "when": [pd.Timestamp("2026-01-01"), pd.NaT],
        "obj": ["x", None],
        "flt": [1.5, float("nan")],
    })
    rid = use_store.put(df, {"source": "s", "owner_id": "user-1"})

    resp = client.get(f"/api/results/{rid}/export.csv")

    assert resp.status_code == 200
    parsed = list(csv.reader(io.StringIO(resp.text)))
    assert parsed[2] == ["", "", "", ""]
    assert "<NA>" not in resp.text and "NaT" not in resp.text


# ── Loud failure ──────────────────────────────────────────────────────────────

def test_expired_result_returns_410_not_partial_data(app_client, use_store):
    client, _, _ = app_client
    rid = use_store.put(_wide_frame(20), {"source": "s", "owner_id": "user-1"})

    # NOTE: monkeypatching ttl_seconds=0 would NOT work — _is_expired() treats
    # a falsy/negative TTL as "no expiry" and returns False. Age the payload
    # instead, which is what actually happens in production.
    use_store.ttl_seconds = 3600
    use_store._memory[rid]["created_at"] = time.time() - 7200

    resp = client.get(f"/api/results/{rid}/export.csv")

    assert resp.status_code == 410
    assert "re-run" in resp.json()["detail"].lower()


def test_unknown_id_returns_404(app_client, use_store):
    client, _, _ = app_client
    resp = client.get("/api/results/not-a-result-id/export.csv")
    assert resp.status_code == 404


def test_evicted_id_returns_410(app_client, use_store):
    client, _, _ = app_client
    resp = client.get("/api/results/dlr_deadbeefdeadbeefdeadbeefdeadbeef/export.csv")
    # Well-formed but resolvable nowhere: still a hard failure, never a partial.
    assert resp.status_code == 410


def test_export_requires_auth(app_client, use_store):
    client, main_mod, _ = app_client
    from api.deps import get_current_user

    main_mod.app.dependency_overrides.pop(get_current_user, None)
    rid = use_store.put(_wide_frame(5), {"source": "s"})

    resp = client.get(f"/api/results/{rid}/export.csv")

    assert resp.status_code in (401, 403)


def test_export_refuses_another_users_result(app_client, use_store):
    client, _, override = app_client
    rid = use_store.put(_wide_frame(5), {"source": "s", "owner_id": "user-1"})

    override({"sub": "user-2", "email": "two@example.com"})
    resp = client.get(f"/api/results/{rid}/export.csv")

    # 404 rather than 403: don't confirm the id exists to a non-owner.
    assert resp.status_code == 404


# ── Ownership: legacy ids, stamping, producers (f2-CX-01 / f2-CX-13) ──────────

def test_legacy_ownerless_result_is_not_exportable(app_client, use_store):
    """An id whose meta carries no owner_id is refused for EVERY authenticated
    user — including the one whose session produced it. Card-served ids are
    stamped at serialization; anything still ownerless is agent-internal."""
    client, _, override = app_client
    rid = use_store.put(_wide_frame(5), {"source": "s"})  # no owner_id

    assert client.get(f"/api/results/{rid}/export.csv").status_code == 404
    assert client.get(f"/api/results/{rid}/meta").status_code == 404

    override({"sub": "user-2", "email": "two@example.com"})
    assert client.get(f"/api/results/{rid}/export.csv").status_code == 404
    assert client.get(f"/api/results/{rid}/meta").status_code == 404


def test_data_card_reuse_stamps_serializing_user_as_owner(app_client, use_store):
    """Reusing a capability-minted ownerless id through the card serializer
    adopts it for the requesting user: they export 200, anyone else 404."""
    client, _, override = app_client
    import json

    from api.serializers.data_card import _build_data_card_event

    df = _wide_frame(5)
    rid = use_store.put(df, {"source": "s", "tool_name": "datalab_sia_search"})

    event_str, _ = _build_data_card_event(
        {"type": "data", "data": df, "source": "test", "result_id": rid}, "user-1"
    )
    payload = json.loads(event_str.split("data: ", 1)[1])

    assert payload["resultId"] == rid
    assert use_store.lookup(rid)[1]["owner_id"] == "user-1"
    assert client.get(f"/api/results/{rid}/export.csv").status_code == 200

    override({"sub": "user-2", "email": "two@example.com"})
    assert client.get(f"/api/results/{rid}/export.csv").status_code == 404


def test_stamp_owner_never_reassigns_an_existing_owner(store):
    rid = store.put(pd.DataFrame({"a": [1]}), {"owner_id": "user-1"})

    assert store.stamp_owner(rid, "user-2") is False
    assert store.lookup(rid)[1]["owner_id"] == "user-1"
    # Re-stamping the same owner is an idempotent success.
    assert store.stamp_owner(rid, "user-1") is True
    # dlt_ my-table keys and unresolvable ids are refused outright.
    assert store.stamp_owner("dlt_some_table", "user-1") is False
    assert store.stamp_owner("dlr_" + "0" * 32, "user-1") is False


def test_serializer_mints_no_id_without_an_owner(use_store):
    """With no owner_id and no existing id the serializer returns None — an
    ownerless minted id could never be exported, so the button must degrade
    honestly instead (f2-CX-01)."""
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for _p in (repo, os.path.join(repo, "ui-pro")):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    import json

    from api.serializers.data_card import _build_data_card_event

    event_str, _ = _build_data_card_event(
        {"type": "data", "data": _wide_frame(5), "source": "test"}, None
    )
    payload = json.loads(event_str.split("data: ", 1)[1])

    assert payload["resultId"] is None
    assert not use_store._memory  # nothing was stored


def test_sparcl_producer_put_stamps_ctx_user_as_owner(use_store):
    """Producer-side stamping (f2-CX-01): spectra._store_rows carries
    ctx.user_id into the stored meta so SPARCL results are export-owned."""
    import types

    from capabilities.spectra import _store_rows

    ctx = types.SimpleNamespace(result_store=use_store, user_id="user-7")
    rid = _store_rows(ctx, [{"a": 1}], {"tool_name": "sparcl_search_spectra"})

    assert rid
    assert use_store.lookup(rid)[1]["owner_id"] == "user-7"

    # An anonymous ctx stays ownerless rather than stamping a junk value.
    ctx_anon = types.SimpleNamespace(result_store=use_store, user_id=None)
    rid2 = _store_rows(ctx_anon, [{"a": 1}], {"tool_name": "sparcl_search_spectra"})
    assert "owner_id" not in use_store.lookup(rid2)[1]


# ── Ceiling ───────────────────────────────────────────────────────────────────

def test_export_max_rows_ceiling_is_flagged_as_truncated(app_client, use_store, monkeypatch):
    client, _, _ = app_client
    monkeypatch.setenv("QUASAR_RESULT_EXPORT_MAX_ROWS", "100")
    rid = use_store.put(_wide_frame(500), {"source": "s", "owner_id": "user-1"})

    resp = client.get(f"/api/results/{rid}/export.csv")

    assert resp.status_code == 200
    assert len(resp.text.strip().split("\n")) == 101
    assert resp.headers["X-Quasar-Rowcount"] == "100"
    assert resp.headers["X-Quasar-Total-Rows"] == "500"
    # A ceiling-clipped export is partial — it must say so in-band.
    assert resp.headers["X-Quasar-Truncated"] == "1"


# ── Scan-campaign regressions (2026-07-17) ────────────────────────────────────

def test_export_of_empty_result_still_carries_header(app_client, use_store):
    """UIAPI-13 / dl-export-empty-frame-no-header: a zero-row result must
    export its column schema, not a 0-byte file."""
    client, _, _ = app_client
    df = pd.DataFrame({"ra": pd.Series(dtype=float), "dec": pd.Series(dtype=float)})
    rid = use_store.put(df, {"source": "s", "owner_id": "user-1"})

    resp = client.get(f"/api/results/{rid}/export.csv")

    assert resp.status_code == 200
    assert resp.text == "ra,dec\n"


def test_export_rejects_my_table_keys(app_client, use_store):
    """UIAPI-05 / dl-export-owner-gap: dlt_ keys carry no owner_id and legacy
    shared-namespace names are guessable slugs — this route must not serve
    them (my-tables have their own owner-scoped route)."""
    client, _, override = app_client
    rid = use_store.put(_wide_frame(5), {"source": "s", "owner_id": "user-1"})
    use_store.save_result(rid, "secret_table")

    override({"sub": "user-2", "email": "two@example.com"})
    resp = client.get("/api/results/dlt_secret_table/export.csv")
    assert resp.status_code == 404

    # Even the saving user goes through the my-tables route, not this one.
    override({"sub": "user-1", "email": "one@example.com"})
    assert client.get("/api/results/dlt_secret_table/export.csv").status_code == 404
    assert client.get("/api/results/dlt_secret_table/meta").status_code == 404


# ── Serializer → HTTP integration (f2-CX-14) ──────────────────────────────────

def test_serializer_minted_id_streams_full_frame_over_http(app_client, use_store):
    """The id the card serializer mints must be downloadable end-to-end: all
    12k rows through the actual /export.csv route, not just store.lookup."""
    client, _, _ = app_client
    import json

    from api.serializers.data_card import _build_data_card_event

    df = _wide_frame(12_000)
    event_str, _ = _build_data_card_event(
        {"type": "data", "data": df, "source": "test"}, "user-1"
    )
    payload = json.loads(event_str.split("data: ", 1)[1])
    assert payload["resultId"]

    resp = client.get(f"/api/results/{payload['resultId']}/export.csv")

    assert resp.status_code == 200
    lines = resp.text.strip().split("\n")
    assert len(lines) == 12_001, "CSV must carry every row plus one header row"
    assert lines[0].split(",") == list(df.columns)


# ── Disk tier: memory eviction and worker restart (f2-CX-16) ──────────────────

def test_export_survives_memory_eviction_and_worker_restart(app_client, monkeypatch, tmp_path):
    from services.datalab_result_store import DatalabResultStore
    import services.datalab_result_store as mod

    cache_dir = tmp_path / "disk_results"
    store = DatalabResultStore(cache_dir=cache_dir, enable_disk_cache=True)
    if store._cache is None:
        pytest.skip("diskcache unavailable")
    monkeypatch.setattr(mod, "default_result_store", lambda: store)
    client, _, _ = app_client
    store2 = None
    try:
        rid = store.put(_wide_frame(50), {"source": "s", "owner_id": "user-1"})

        # (a) Evicted from the memory tier: the export resolves from disk.
        store._memory.clear()
        resp = client.get(f"/api/results/{rid}/export.csv")
        assert resp.status_code == 200
        assert len(resp.text.strip().split("\n")) == 51

        # (b) Worker restart: a NEW store on the same cache dir resolves it.
        store._cache.close()
        store2 = DatalabResultStore(cache_dir=cache_dir, enable_disk_cache=True)
        monkeypatch.setattr(mod, "default_result_store", lambda: store2)
        resp = client.get(f"/api/results/{rid}/export.csv")
        assert resp.status_code == 200
        assert len(resp.text.strip().split("\n")) == 51
    finally:
        # Windows: release the DiskCache sqlite handles so tmp_path cleanup
        # does not trip over locked files.
        try:
            store._cache.close()
        except Exception:
            pass
        if store2 is not None and store2._cache is not None:
            try:
                store2._cache.close()
            except Exception:
                pass


# ── Meta ──────────────────────────────────────────────────────────────────────

def test_meta_reports_full_shape(app_client, use_store):
    client, _, _ = app_client
    df = _wide_frame(12_000)
    rid = use_store.put(df, {"source": "test_catalog", "owner_id": "user-1"})

    resp = client.get(f"/api/results/{rid}/meta")

    assert resp.status_code == 200
    body = resp.json()
    assert body["totalRows"] == 12_000
    assert body["columns"] == list(df.columns)
    # Expiry disclosure (f2-CX-10): createdAt/ttlSeconds/expiresAt, with the
    # store-internal "_created_at" key kept out of the response.
    assert body["ttlSeconds"] == use_store.ttl_seconds
    assert body["createdAt"] and body["createdAt"].endswith("+00:00")
    assert body["expiresAt"] > body["createdAt"]
    assert "_created_at" not in resp.text


def test_meta_on_expired_id_returns_410(app_client, use_store):
    client, _, _ = app_client
    rid = use_store.put(_wide_frame(5), {"source": "s", "owner_id": "user-1"})
    use_store.ttl_seconds = 3600
    use_store._memory[rid]["created_at"] = time.time() - 7200

    resp = client.get(f"/api/results/{rid}/meta")

    assert resp.status_code == 410


# ── Store lookup semantics ────────────────────────────────────────────────────

def test_lookup_distinguishes_expired_from_unknown(store):
    rid = store.put(pd.DataFrame({"a": [1]}), {})
    assert store.lookup(rid)[2] == "ok"
    assert store.lookup("garbage")[2] == "unknown"
    assert store.lookup("dlr_" + "0" * 32)[2] == "gone"

    store.ttl_seconds = 3600
    store._memory[rid]["created_at"] = time.time() - 7200
    assert store.lookup(rid)[2] == "expired"


# ── Serializer honesty ────────────────────────────────────────────────────────

def test_data_card_reports_truncation_and_mints_an_export_id(use_store):
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for _p in (repo, os.path.join(repo, "ui-pro")):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    import json

    from api.serializers.data_card import _build_data_card_event

    df = _wide_frame(12_000)
    event_str, rich = _build_data_card_event(
        {"type": "data", "data": df, "source": "test"}, "user-1"
    )
    payload = json.loads(event_str.split("data: ", 1)[1])

    assert payload["totalRows"] == 12_000
    assert payload["displayedRows"] == 10_000
    assert payload["truncated"] is True
    # No result_id on the run_result → the serializer must mint one, or the card
    # would have no way to offer a complete download.
    assert payload["resultId"]
    # The size fields must survive into history, not just the live event.
    assert rich["totalRows"] == 12_000 and rich["truncated"] is True

    # And that minted id must resolve to the FULL frame, not the preview.
    frame, meta, status = use_store.lookup(payload["resultId"])
    assert status == "ok"
    assert len(frame) == 12_000
    assert meta["owner_id"] == "user-1"


def test_data_card_marks_a_complete_table_as_not_truncated(use_store):
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for _p in (repo, os.path.join(repo, "ui-pro")):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    import json

    from api.serializers.data_card import _build_data_card_event

    event_str, _ = _build_data_card_event(
        {"type": "data", "data": _wide_frame(42), "source": "test"}, "user-1"
    )
    payload = json.loads(event_str.split("data: ", 1)[1])

    assert payload["totalRows"] == 42
    assert payload["displayedRows"] == 42
    assert payload["truncated"] is False


def test_data_card_reuses_an_existing_result_id(use_store):
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for _p in (repo, os.path.join(repo, "ui-pro")):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    import json

    from api.serializers.data_card import _build_data_card_event

    # Data Lab tools already stash the frame and stamp result_id; don't re-put.
    event_str, _ = _build_data_card_event(
        {"type": "data", "data": _wide_frame(5), "source": "test", "result_id": "dlr_preexisting"},
        "user-1",
    )
    payload = json.loads(event_str.split("data: ", 1)[1])

    assert payload["resultId"] == "dlr_preexisting"


# ── Disk-tier TTL (f2-CX-18) ──────────────────────────────────────────────────

def test_disk_tier_lookup_honors_ttl(tmp_path, app_client, monkeypatch):
    """f2-CX-18: an aged payload resurrected from the DiskCache must read as
    expired — never as a live result — and the stale disk copy is purged."""
    pytest.importorskip("diskcache")
    import services.datalab_result_store as mod
    from services.datalab_result_store import DatalabResultStore

    client, _, _ = app_client
    store = DatalabResultStore(cache_dir=tmp_path / "results", enable_disk_cache=True)
    try:
        monkeypatch.setattr(mod, "default_result_store", lambda: store)
        rid = store.put(_wide_frame(5), {"source": "s", "owner_id": "user-1"})

        # Age the DISK copy out-of-band (models a legacy payload written with no
        # expire, or an out-of-band cache write) and evict the memory tier — the
        # exact resurrection path the in-memory TTL check never sees.
        aged = dict(store._cache.get(rid))
        aged["created_at"] = time.time() - 7200
        store._cache.set(rid, aged)
        store._memory.clear()

        assert store.lookup(rid)[2] == "expired"
        assert store._cache.get(rid) is None  # stale copy purged, cannot resurrect
        with pytest.raises(KeyError):
            store.get(rid)

        # And through the route: a hard 410, never data.
        resp = client.get(f"/api/results/{rid}/export.csv")
        assert resp.status_code == 410
    finally:
        store._cache.close()


def test_export_headers_are_cors_exposed(app_client, use_store):
    """f2-CX-20: a cross-origin browser can only read the X-Quasar-* verdict
    headers (and the export filename) if CORS explicitly exposes them —
    without Access-Control-Expose-Headers the cross-site production topology
    reads null for every one of them and a ceiling-clipped CSV would pass as
    full, the exact silent-partial Feature 2 forbids."""
    client, _, _ = app_client
    rid = use_store.put(_wide_frame(5), {"source": "s", "owner_id": "user-1"})

    resp = client.get(
        f"/api/results/{rid}/export.csv",
        headers={"Origin": "http://localhost:3001"},
    )

    assert resp.status_code == 200
    exposed = {
        h.strip().lower()
        for h in resp.headers.get("Access-Control-Expose-Headers", "").split(",")
        if h.strip()
    }
    for header in (
        "x-quasar-truncated",
        "x-quasar-rowcount",
        "x-quasar-total-rows",
        "content-disposition",
    ):
        assert header in exposed, f"{header} missing from Access-Control-Expose-Headers"


# ── Upstream-partial honesty (f2-CX-21) ───────────────────────────────────────

def test_data_card_surfaces_upstream_truncation(use_store):
    """f2-CX-21: a structural upstream_truncated stamp (run result or stored
    provenance) reaches the card payload; the client then refuses to label the
    export complete."""
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for _p in (repo, os.path.join(repo, "ui-pro")):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    import json

    from api.serializers.data_card import _build_data_card_event

    event_str, rich = _build_data_card_event(
        {"type": "data", "data": _wide_frame(10), "source": "test",
         "provenance": {"upstream_truncated": True, "upstream_total": 3000}},
        "user-1",
    )
    payload = json.loads(event_str.split("data: ", 1)[1])

    assert payload["upstreamPartial"] is True
    assert payload["upstreamTotal"] == 3000
    # And it survives into history with the rest of the card.
    assert rich["upstreamPartial"] is True and rich["upstreamTotal"] == 3000


def test_data_card_warnings_alone_do_not_mark_upstream_partial(use_store):
    """f2-CX-21: bare warning strings must NOT flip the export verdict — they
    stay a visible card caveat. Only the structural stamp counts."""
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for _p in (repo, os.path.join(repo, "ui-pro")):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    import json

    from api.serializers.data_card import _build_data_card_event

    event_str, _ = _build_data_card_event(
        {"type": "data", "data": _wide_frame(5), "source": "test",
         "warnings": ["band substituted g->z"]},
        "user-1",
    )
    payload = json.loads(event_str.split("data: ", 1)[1])

    assert payload["upstreamPartial"] is False
    assert payload["upstreamTotal"] is None
    assert payload["partial"] is True  # the existing warnings caveat is untouched


def test_data_card_reads_upstream_stamp_from_run_result_root(use_store):
    """f2-CX-22: the structural stamp can arrive at the run-result ROOT (the
    _external_catalog_table_result seam used by lightkurve and any future
    capped producer), not only via stored provenance."""
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for _p in (repo, os.path.join(repo, "ui-pro")):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    import json

    from api.serializers.data_card import _build_data_card_event

    event_str, rich = _build_data_card_event(
        {"type": "data", "data": _wide_frame(5), "source": "test",
         "upstream_truncated": True, "upstream_total": 50},
        "user-1",
    )
    payload = json.loads(event_str.split("data: ", 1)[1])

    assert payload["upstreamPartial"] is True and payload["upstreamTotal"] == 50
    assert rich["upstreamPartial"] is True
