"""datalab_stream_selection sub-cone tiling (L09: tile-level 502s on the
10-degree Gaia x NSC join). Every tile is bounded by the PARENT cone, carries
the PM window and CMD mask, rows are merged on source_id, and a partial run
says which sub-cones are unknown. Network is faked."""
from __future__ import annotations

import itertools
from types import SimpleNamespace as NS

import pandas as pd

from capabilities.base import CallContext
from capabilities.datalab_tools import StreamSelection

PM = {"pmra_min": -3.5, "pmra_max": -2.0, "pmdec_min": -3.3, "pmdec_max": -2.1}
MASK = {"color_min": 0.2, "color_max": 0.6, "g_min": 17.0, "g_max": 21.5}


class _Store:
    def __init__(self):
        self.saved = []

    def put(self, df, meta):
        self.saved.append((df, meta))
        return f"dlr_{len(self.saved)}"


class _Client:
    timeout = 60.0

    def __init__(self, fail_every=0):
        self.sql = []
        self.fail_every = fail_every
        self._ids = itertools.count()

    def query(self, sql, fmt, async_fallback, timeout):
        self.sql.append(sql)
        if self.fail_every and len(self.sql) % self.fail_every == 0:
            raise RuntimeError("HTTP 502 Bad Gateway")
        # overlapping tiles return a shared star (source_id 1) plus one of their own
        return NS(dataframe=pd.DataFrame({"source_id": [1, 100 + next(self._ids)], "ra": [229.0, 229.1], "dec": [-0.1, -0.2]}))


def _ctx(client, store):
    return CallContext(services={"datalab_client": client}, result_store=store)


def test_every_tile_is_bounded_by_the_parent_cone_and_carries_every_cut():
    client, store = _Client(), _Store()
    out = StreamSelection()._run_tiled(_ctx(client, store), 229.018, -0.124, 5.0, pm=PM, mask=MASK,
                                       match_arcsec=1.0, small_limit=50000, limit=5000, meta={})
    assert out["success"] and out["tiles_total"] == len(client.sql) > 1 and not out["partial"]
    for sql in client.sql:
        assert "q3c_radial_query(ra, dec, 229.018, -0.124, 5)" in sql  # parent bound
        assert "pmra BETWEEN -3.5000 AND -2.0000" in sql and "pmdec BETWEEN -3.3000 AND -2.1000" in sql
    df, meta = store.saved[0]
    assert list(df["source_id"]).count(1) == 1, "rows merged on source_id"
    assert out["rowcount"] == len(df) == 1 + len(client.sql)
    assert meta["provenance"]["tiles_completed"] == out["tiles_total"]


def test_failed_tiles_make_the_result_partial_and_say_so():
    client, store = _Client(fail_every=2), _Store()
    out = StreamSelection()._run_tiled(_ctx(client, store), 229.018, -0.124, 5.0, pm=PM, mask=MASK,
                                       match_arcsec=1.0, small_limit=50000, limit=5000, meta={})
    assert out["success"] and out["partial"] and out["tiles_completed"] < out["tiles_total"]
    assert "PARTIAL: unqueried sub-cones are unknown, not empty." in out["warnings"][0]


def test_all_tiles_failing_is_an_infrastructure_failure():
    client, store = _Client(fail_every=1), _Store()
    out = StreamSelection()._run_tiled(_ctx(client, store), 229.018, -0.124, 5.0, pm=PM, mask=MASK,
                                       match_arcsec=1.0, small_limit=50000, limit=5000, meta={})
    assert out["success"] is False and out["status"] == "infrastructure_failure" and "502" in out["error"]
    assert not store.saved


class _AsyncStreamClient(_Client):
    token = "real.user.token"

    def __init__(self, states):
        super().__init__()
        self.states = list(states)
        self.submitted = []

    def submit(self, *, sql=None, adql=None, timeout=None):
        self.submitted.append(sql)
        return "jobS"

    def status(self, jobid):
        return self.states.pop(0) if self.states else "EXECUTING"

    def results(self, jobid, *, fmt="pandas", query_text=None):
        from integrations.datalab_client import DatalabResult
        return DatalabResult.from_dataframe(pd.DataFrame({"source_id": [1, 2, 3], "ra": [229.0] * 3, "dec": [-0.1] * 3}), {})

    def abort(self, jobid):
        return "ABORTED"


def test_whole_field_join_runs_as_one_async_job_when_a_login_token_exists(monkeypatch):
    import services.datalab_orchestration as orch

    monkeypatch.setenv("DATALAB_STREAM_ASYNC", "1")  # opt-in: the async queue was too slow live (2026-09-23)
    monkeypatch.setattr(orch.time, "sleep", lambda s: None)
    sql, meta = StreamSelection.build_sql(229.018, -0.124, 5.0, pm=PM, cmd_mask=MASK, match_arcsec=1.0, small_limit=50000, limit=5000)
    client, store = _AsyncStreamClient(["EXECUTING", "COMPLETED"]), _Store()
    out = StreamSelection()._run_async(_ctx(client, store), sql, meta)
    assert out["success"] and out["rowcount"] == 3 and "async job (jobS" in out["warnings"][0]
    assert "pmra BETWEEN -3.5000 AND -2.0000" in client.submitted[0] and not client.sql, "no sync tiles"
    assert store.saved[0][1]["provenance"]["jobid"] == "jobS"
    # anonymous token -> the caller falls back to sub-cones
    anon = StreamSelection()._run_async(_ctx(_Client(), _Store()), sql, meta)
    assert anon.get("async_unavailable") is True
