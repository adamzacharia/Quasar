"""Data Lab SIA/transport regressions (scan-campaign 2026-07-17).

- dl-sia-timeout-never-applied: pyvo calls session.get(); __getattr__ resolved
  it to the REAL requests.Session bound method, so the request()-only override
  never ran and DATALAB_SIA_TIMEOUT_SECONDS was dead config.
- CX-06: SIA provenance must carry the exact wire params (POS/SIZE) so the
  provenance surface can render a reproducible request line.
- dl-results-download-timeout-loses-jobid: a results download that blows the
  wall clock must re-raise WITH the jobid so the completed server job stays
  reachable via datalab_job_results.
"""

from __future__ import annotations

import pytest

from integrations.datalab_sia_client import DatalabSiaClient, _TimeoutSession


def test_timeout_session_injects_timeout_on_get_and_post():
    ts = _TimeoutSession(42.0)
    captured = {}

    class _FakeSession:
        def get(self, url, **kwargs):
            captured["get"] = kwargs
            return "ok"

        def post(self, url, **kwargs):
            captured["post"] = kwargs
            return "ok"

        def request(self, method, url, **kwargs):
            captured["request"] = kwargs
            return "ok"

    ts._session = _FakeSession()
    # Exactly what pyvo DALQuery.submit does: session.get(url, stream=True).
    ts.get("https://datalab.noirlab.edu/sia", params={}, stream=True)
    assert captured["get"]["timeout"] == 42.0
    ts.post("https://datalab.noirlab.edu/sia", data={})
    assert captured["post"]["timeout"] == 42.0
    ts.request("GET", "https://datalab.noirlab.edu/sia")
    assert captured["request"]["timeout"] == 42.0
    # An explicit caller timeout is never overridden.
    ts.get("https://datalab.noirlab.edu/sia", timeout=5.0)
    assert captured["get"]["timeout"] == 5.0


def test_sia_provenance_carries_wire_params(monkeypatch):
    """CX-06: the descriptive scalars (ra/dec/fov_deg) are not what went over
    the wire — SIZE is cos(dec)-stretched. The provenance must include the
    literal POS/SIZE request parameters."""
    client = DatalabSiaClient(timeout=10.0)
    monkeypatch.setattr(
        DatalabSiaClient,
        "_search_endpoint",
        lambda self, ep, ra, dec, size: [{"access_url": "http://x/fits"}],
    )
    out = client.search(150.0, -60.0, 0.2, endpoint="https://example/sia")
    prov = out["provenance"]
    assert prov["ra"] == 150.0 and prov["dec"] == -60.0 and prov["fov_deg"] == 0.2
    expected_size = DatalabSiaClient._sia_size(0.2, -60.0)
    assert prov["params"]["POS"] == "150.0,-60.0"
    assert prov["params"]["SIZE"] == f"{expected_size[0]},{expected_size[1]}"
    # SIZE really is the stretched request, not the descriptive fov.
    assert expected_size[0] > 0.2


def test_results_download_timeout_keeps_jobid():
    """dl-results-download-timeout-loses-jobid: when the async-fallback job
    COMPLETED but the results download blew the wall clock, the raised error
    carried no jobid and the finished server-side result became unreachable."""
    from integrations import datalab_client as dc

    client = dc.DatalabClient(token="real.login.token")
    client.query = lambda **kw: "job-123"
    client.status = lambda jid: "COMPLETED"

    def _slow_results(jid, **kw):
        raise dc.DatalabClientError(
            "Data Lab request to /results failed: wall-clock deadline of 90s "
            "exceeded while the response was still streaming (request timed out)"
        )

    client.results = _slow_results
    with pytest.raises(dc.DatalabClientError) as excinfo:
        client._query_via_async_job("SELECT 1", "sql")
    assert excinfo.value.jobid == "job-123"
    assert "job-123" in str(excinfo.value)
    assert "datalab_job_results" in str(excinfo.value)
