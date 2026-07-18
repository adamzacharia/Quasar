"""xmatch_user_list provenance carryover (scan-campaign 2026-07-17).

dl-xmatch-truncation-provenance-drop: a LIMIT-truncated source result is a
storage-order, spatially biased slice. When xmatch_user_list mints a NEW result
from it, the truncation stamp must survive into the derived result's provenance
(and a warning must reach the model) — otherwise downstream plots of the match
table lose the TRUNCATED SAMPLE caption and a biased corner slice renders as
the full on-sky distribution (the live-P7/P9 failure mode).
"""

from __future__ import annotations

import pandas as pd

import capabilities.datalab as dl
from capabilities.base import CallContext
from services import cds_xmatch
from services.datalab_result_store import DatalabResultStore


class _Resp:
    def __init__(self, status_code=200, text=""):
        self.status_code = status_code
        self.text = text


def _fake_xmatch(monkeypatch, response_csv="angDist,id,ra,dec,Jmag\n0.2,obj1,10.0,20.0,12.3\n"):
    def fake_post(url, data=None, files=None, timeout=None):
        return _Resp(200, response_csv)

    monkeypatch.setattr(cds_xmatch.requests, "post", fake_post)


def test_xmatch_carries_source_truncation_into_new_result(monkeypatch):
    _fake_xmatch(monkeypatch)
    store = DatalabResultStore(enable_disk_cache=False)
    source_id = store.put(
        pd.DataFrame({"ra": [10.0], "dec": [20.0]}),
        {"provenance": {"catalog": "nsc_dr2", "table": "object",
                        "limit_truncated": True, "row_limit": 500}},
    )
    ctx = CallContext(services={}, result_store=store, user_id="alice")
    out = dl.XmatchUserList().run(
        dl.XmatchUserListInput(catalog="gaia_dr3", result_id=source_id, radius_arcsec=2.0),
        ctx,
    ).to_native()
    assert out["success"] is True
    # The model sees the bias warning on the tool output itself.
    assert any("truncated" in w.lower() for w in out.get("warnings", []))

    stored = store.get(out["result_id"])
    prov = stored.provenance
    assert prov["limit_truncated"] is True and prov["row_limit"] == 500
    assert prov["source_result_id"] == source_id
    assert prov["source_provenance"]["catalog"] == "nsc_dr2"
    # The plot layer's stamp logic fires on the derived result.
    from services.datalab_analysis import _truncation_warnings
    assert _truncation_warnings(prov)
    # Owner scoping for the export route (dl-export-owner-gap).
    frame, meta, status = store.lookup(out["result_id"])
    assert status == "ok" and meta["owner_id"] == "alice"


def test_xmatch_untruncated_source_carries_no_stamp(monkeypatch):
    _fake_xmatch(monkeypatch)
    store = DatalabResultStore(enable_disk_cache=False)
    source_id = store.put(
        pd.DataFrame({"ra": [10.0], "dec": [20.0]}),
        {"provenance": {"catalog": "nsc_dr2", "table": "object"}},
    )
    ctx = CallContext(services={}, result_store=store)
    out = dl.XmatchUserList().run(
        dl.XmatchUserListInput(catalog="gaia_dr3", result_id=source_id, radius_arcsec=2.0),
        ctx,
    ).to_native()
    assert out["success"] is True
    assert "warnings" not in out or not any(
        "truncated" in w.lower() for w in out.get("warnings", [])
    )
    prov = store.get(out["result_id"]).provenance
    assert "limit_truncated" not in prov
