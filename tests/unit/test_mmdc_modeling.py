"""MMDC emission modeling (services/mmdc_modeling.py): fit-input building,
validation, redshift resolution, the fit job lifecycle and the overlay figure.
Offline: a fake SDK client and the recorded batch result fixture."""

from __future__ import annotations

import io
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from services import mmdc_modeling as mm
from services import mmdc_service as ms

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "mmdc"
SSC = {"log_B": -1.5, "log_electron_luminosity": 44.0, "log_gamma_cut": 5.0, "log_gamma_min": 2.0,
       "log_radius": 16.0, "lorentz_factor": 20.0, "spectral_index": 2.2}


def _sed(name="sed_mkn421_trim.csv"):
    return ms.normalize_sed_csv((FIX / name).read_text(encoding="utf-8"))


# ── validation ──────────────────────────────────────────────────────────────
def test_model_type_normalization():
    assert mm.normalize_model_type("ssc") == "SSC" and mm.normalize_model_type("lepto-hadronic") == "HADRONIC"
    with pytest.raises(ValueError):
        mm.normalize_model_type("synchrotron")


def test_parameter_names_and_training_ranges_are_enforced():
    assert mm.validate_parameters("SSC", SSC, require_all=True) == SSC
    with pytest.raises(ValueError, match="unknown SSC parameter"):
        mm.validate_parameters("SSC", {**SSC, "bogus": 1}, require_all=True)   # MMDC itself accepts this silently
    with pytest.raises(ValueError, match="training range"):
        mm.validate_parameters("SSC", {**SSC, "log_B": 5.0}, require_all=True)  # MMDC itself accepts this silently
    with pytest.raises(ValueError, match="missing"):
        mm.validate_parameters("SSC", {"log_B": -1}, require_all=True)
    assert mm.validate_parameters("EIC", {"log_MBH": 8.5}, require_all=False) == {"log_MBH": 8.5}
    with pytest.raises(ValueError, match="unknown HADRONIC"):
        mm.validate_parameters("HADRONIC", {"log_electron_luminosity": 44}, require_all=False)


def test_neutrino_likelihood_requires_explicit_inputs():
    assert mm.validate_neutrino("HADRONIC", None) == {}
    p = mm.validate_neutrino("HADRONIC", "poisson", n_icecube=1, dt=12)
    assert p["n_icecube"] == 1 and p["dt"] == 12.0 and "1 IceCube event(s) in a 12-month window" in p["description"]
    c = mm.validate_neutrino("HADRONIC", "chi2", x1=100, x2=1000, y=-11.5)
    assert c["likelihood_type"] == "chi2" and "between 100 and 1000 TeV" in c["description"]
    for kwargs, msg in [
        (dict(likelihood_type="poisson", n_icecube=1), "missing"),
        (dict(likelihood_type="poisson", n_icecube=1.5, dt=6), "whole number"),
        (dict(likelihood_type="poisson", n_icecube=1, dt=500), "outside"),
        (dict(likelihood_type="poisson", n_icecube=1, dt=6, y=-11), "do not apply"),
        (dict(likelihood_type="chi2", x1=900, x2=500, y=-11), "below"),
        (dict(likelihood_type=None, n_icecube=1), "without likelihood_type"),
    ]:
        with pytest.raises(ValueError, match=msg):
            mm.validate_neutrino("HADRONIC", **kwargs)
    with pytest.raises(ValueError, match="only to model_type='HADRONIC'"):
        mm.validate_neutrino("SSC", "poisson", n_icecube=1, dt=6)


def test_z_validation():
    assert mm.validate_z(0.03) == 0.03
    for bad in (0, -1, 11, "x", float("nan")):
        with pytest.raises(ValueError):
            mm.validate_z(bad)


# ── fit input ───────────────────────────────────────────────────────────────
def test_fit_points_drop_upper_limits_and_bin():
    df = _sed()
    fit, rep = mm.fit_points_from_frame(df)
    assert rep["n_upper_limits_dropped"] == int(df["upper_limit"].sum()) > 0
    assert rep["n_rows"] == len(df) and rep["n_fit_points"] == len(fit) >= mm.MIN_FIT_POINTS
    assert list(fit.columns) == ["frequency", "flux", "flux_err", "n_rows"] and (fit["flux_err"] > 0).all()
    assert fit["n_rows"].sum() == len(df) - rep["n_upper_limits_dropped"] - rep["n_nonpositive_or_missing_dropped"]
    assert "time-averaged" in rep["method"]
    text = mm.fit_csv_text(fit)
    assert text.splitlines()[0] == "frequency,flux,flux_err" and "UL" not in text   # a UL flag column crashes MMDC


def test_fit_bin_error_is_max_of_scatter_and_measurement():
    rows = pd.DataFrame({"frequency_Hz": [1.00e17, 1.01e17, 1.02e17] + [10 ** e for e in (9, 11, 13, 15, 19, 23)],
                         "nuFnu_erg_cm2_s": [1e-11, 3e-11, 2e-11] + [1e-12] * 6,
                         "nuFnu_err_erg_cm2_s": [1e-13] * 3 + [1e-13] * 6, "upper_limit": False})
    fit, rep = mm.fit_points_from_frame(rows)
    xbin = fit.iloc[(fit["frequency"] - 1.01e17).abs().argmin()]
    assert xbin["n_rows"] == 3 and xbin["flux"] == pytest.approx(2e-11)
    assert xbin["flux_err"] == pytest.approx(np.std([1e-11, 3e-11, 2e-11], ddof=1))
    assert rep["ranges_covered"] and "coverage_warnings" not in rep or "gamma" in rep["ranges_covered"]


def test_too_few_points_and_bad_tables_raise():
    tiny = pd.DataFrame({"frequency": [1e9, 1e10], "flux": [1e-12, 1e-12], "flux_err": [1e-13, 1e-13]})
    with pytest.raises(ValueError, match="only 2 fit points"):
        mm.fit_points_from_frame(tiny)
    with pytest.raises(ValueError, match="frequency and nuFnu"):
        mm.fit_points_from_frame(pd.DataFrame({"a": [1]}))
    with pytest.raises(ValueError, match="csv_text needs"):
        mm.parse_uploaded_csv("a,b\n1,2\n")


def test_coverage_warning_without_gamma_rays():
    df = _sed()
    fit, rep = mm.fit_points_from_frame(df[df["range"] != "gamma"])
    assert any("gamma" in w for w in rep["coverage_warnings"])


# ── results ─────────────────────────────────────────────────────────────────
def test_bound_flags_on_the_recorded_mkn421_fit():
    raw = json.loads((FIX / "batch_result_ssc_mkn421.json").read_text())
    rows = mm.flag_bounds("SSC", raw["best_parameters"], raw["fixed_parameters"])
    by = {r["parameter"]: r for r in rows}
    assert by["log_B"]["at_bound"] == "at lower bound"                # -2.93 +- 0.07 vs floor -3
    assert by["lorentz_factor"]["at_bound"] == "outside training range"  # 54.0 vs web-UI range 3 to 50
    assert by["log_radius"]["at_bound"] is None
    wide = mm.flag_bounds("HADRONIC", {"log_B": {"value": -1.66, "error": 1.46}, "pe": {"value": 1.80, "error": 0.1}}, {})
    wb = {r["parameter"]: r["at_bound"] for r in wide}
    assert wb["log_B"] == "1-sigma interval reaches the lower bound"   # interior value, wide error: not an edge value
    assert wb["pe"] == "at lower bound"                                 # 1.80 on [1.75, 5]: within 2 % of the range
    assert "Doppler factor" in by["lorentz_factor"]["meaning"]
    fixed = mm.flag_bounds("SSC", raw["best_parameters"], {"log_radius": {"value": 16.0, "error": None}})
    assert next(r for r in fixed if r["parameter"] == "log_radius")["fixed"] is True


def test_curve_peaks_find_the_two_humps():
    inf = json.loads((FIX / "infer_ssc_mkn421.json").read_text())
    peaks = mm.curve_peaks(inf["nu"], inf["nuFnu"])
    assert 1 <= len(peaks) <= 3 and all(p["nuFnu"] > 0 for p in peaks)


def test_copy_artifacts_only_from_mmdc_media(monkeypatch, tmp_path):
    import services.plotting as plotting

    monkeypatch.setattr(plotting, "PLOT_OUTPUT_DIR", str(tmp_path))
    links = {"pdf_report": f"{ms.MMDC_BASE_URL}/media/plots/abc.pdf", "best_model_csv": "https://evil.example/x.csv",
             "best_parameters_csv": None}
    out = mm.copy_artifacts(links, "abc", lambda url: b"%PDF-1.4")
    assert out["pdf_report"]["copied"] and out["pdf_report"]["url"] == "/plots/mmdc_fit_abc_pdf_report.pdf"
    assert (tmp_path / "mmdc_fit_abc_pdf_report.pdf").read_bytes() == b"%PDF-1.4"
    assert out["best_model_csv"]["copied"] is False and "best_parameters_csv" not in out


# ── service with a fake SDK ─────────────────────────────────────────────────
class _Plot:
    def _apply_style(self, dark=False):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt

    def _save_and_encode(self, fig, name):
        import matplotlib.pyplot as plt

        plt.close(fig)
        return {"web_url": f"/plots/{name}.png", "base64_png": "AAAA"}


class _Jobs:
    def __init__(self):
        from services.datalab_job_service import DatalabJobService

        self.inner = DatalabJobService(max_workers=1)

    def __getattr__(self, item):
        return getattr(self.inner, item)


class _FakeModeling:
    def __init__(self):
        self.raw = json.loads((FIX / "batch_result_ssc_mkn421.json").read_text())
        self.polls = 0
        self.done_after = 2
        self.submitted = {}
        self.validated = None

    def infer(self, **kw):
        inf = json.loads((FIX / "infer_ssc_mkn421.json").read_text())
        self.infer_kw = kw
        return SimpleNamespace(**inf)

    def validate_csv(self, fobj):
        self.validated = (getattr(fobj, "name", None), fobj.read().decode())
        return SimpleNamespace(model_dump=lambda: {"success": True, "message": "ok", "data_points": 9})

    def submit_batch(self, fobj, **kw):
        self.submitted = {"name": getattr(fobj, "name", None), **kw}
        return SimpleNamespace(batch_result_id="batch-1")

    def get_batch_result(self, bid):
        from astro_mmdc.models.modeling import BatchResult

        self.polls += 1
        if self.polls < self.done_after:
            return BatchResult(status="processing", model_type="SSC")
        return BatchResult.model_validate(self.raw)


def _services(monkeypatch, tmp_path, fake=None):
    import services.plotting as plotting
    from services import datalab_result_store as store_mod

    monkeypatch.setattr(plotting, "PLOT_OUTPUT_DIR", str(tmp_path))
    store = store_mod.DatalabResultStore(enable_disk_cache=False)
    monkeypatch.setattr(store_mod, "default_result_store", lambda: store)
    fake = fake or _FakeModeling()
    client = SimpleNamespace(modeling=fake)
    mmdc = ms.MmdcService(plotting_service=_Plot(), client_factory=lambda: client,
                          known_sources_loader=lambda: ([], []))
    ned_calls = []

    def ned(name):
        ned_calls.append(name)
        return 0.0308

    svc = mm.MmdcModelingService(mmdc, job_service=_Jobs(), fetch_bytes=lambda url: b"x", ned_lookup=ned)
    df = _sed()
    rid = store.put(df, {"owner_id": "u1", "source": "MMDC", "mmdc": {"kind": "sed", "target": "Mkn 421",
                                                                      "redshift": 0.03179,
                                                                      "window": {"rule_applied": "contained"}}})
    return svc, fake, store, rid, ned_calls


def test_z_resolution_order(monkeypatch, tmp_path):
    svc, _, _, _, ned_calls = _services(monkeypatch, tmp_path)
    assert svc.resolve_z(0.1, {}, None) == (0.1, "given by the caller")
    z, src = svc.resolve_z(None, {"mmdc": {"redshift": 0.047}}, None)
    assert z == 0.047 and src.startswith("MMDC") and ned_calls == []
    z, src = svc.resolve_z(None, {}, "Mkn 421")
    assert z == 0.0308 and src.startswith("NED") and ned_calls == ["Mkn 421"]
    svc._ned_lookup = lambda name: None
    with pytest.raises(ValueError, match="do not guess"):
        svc.resolve_z(None, {}, "Nowhere")


def test_spectrum_mode_overlays_model_on_the_stored_sed(monkeypatch, tmp_path):
    svc, fake, _, rid, _ = _services(monkeypatch, tmp_path)
    out = svc.spectrum(model_type="SSC", z=None, parameters=SSC, result_id=rid, owner_id="u1")
    assert out["success"] and out["z"] == 0.03179 and out["z_source"].startswith("MMDC")
    assert fake.infer_kw["model_type"] == "SSC" and fake.infer_kw["parameters"] == SSC
    names = [t["name"] for t in out["_figure"]["plotly_spec"]["data"]]
    assert "SSC model (fixed parameters)" in names and "upper limits" in names
    assert out["attribution"]["modeling_citations"][0]["bibcode"] == "2024ApJ...963...71B"
    with pytest.raises(KeyError):
        svc.spectrum(model_type="SSC", z=0.03, parameters=SSC, result_id=rid, owner_id="intruder")


def test_fit_lifecycle_submit_pending_then_done(monkeypatch, tmp_path):
    svc, fake, _, rid, _ = _services(monkeypatch, tmp_path)
    fake.done_after = 3
    mm.FIT_POLL_INTERVAL_S, old = 0.0, mm.FIT_POLL_INTERVAL_S
    try:
        first = svc.submit_fit(model_type="SSC", result_id=rid, owner_id="u1", wait_seconds=0)
        assert first["pending"] and first["job_id"] == "batch-1" and "mmdc_model_job" in first["error"]
        assert fake.validated[0].endswith(".csv") and fake.submitted["name"].endswith(".csv")
        assert fake.validated[1].splitlines()[0] == "frequency,flux,flux_err"
        assert fake.submitted["z"] == 0.03179 and fake.submitted["model_type"] == "SSC"
        assert "email" not in fake.submitted  # never send a user's email; the SDK default stays
        assert first["params"]["source"]["window"]["rule_applied"] == "contained"
        rec = svc.jobs.status("batch-1", owner_id="u1")
        assert rec["kind"] == "mmdc_fit" and rec["status"] == "running"
        done = svc.poll("batch-1", owner_id="u1", wait_seconds=5)
    finally:
        mm.FIT_POLL_INTERVAL_S = old
    assert done["success"] and done["job_status"] == "succeeded"
    by = {r["parameter"]: r for r in done["best_fit_parameters"]}
    assert by["log_B"]["value"] == pytest.approx(-2.9312, abs=1e-3) and by["log_B"]["error"] == pytest.approx(0.0682, abs=1e-3)
    assert "log_B" in done["parameters_at_bound"] and done["warnings"]
    assert done["fit_statistics"]["logZ"] == pytest.approx(-175.189, abs=1e-2)
    assert done["artifacts"]["pdf_report"]["url"].startswith("/plots/mmdc_fit_batch-1")
    names = [t["name"] for t in done["_figure"]["plotly_spec"]["data"]]
    assert "SSC best fit" in names and "fit input (binned)" in names and "posterior samples" in names
    assert svc.jobs.status("batch-1", owner_id="u1")["status"] == "succeeded"
    again = svc.poll("batch-1", owner_id="u1")   # served from the job record, no heavy figure
    assert again["success"] and "_figure" not in again
    with pytest.raises(KeyError):
        svc.poll("batch-1", owner_id="someone-else")


def test_poll_with_no_turn_budget_left_says_stop(monkeypatch, tmp_path):
    """UI 2026-09-26 Q3b: with the turn budget spent, poll returned at once and
    the model re-called it every 2 s. Now it is marked budget_exhausted."""
    import services.tool_budgets as tb

    svc, fake, _, rid, _ = _services(monkeypatch, tmp_path)
    fake.done_after = 99
    monkeypatch.setattr(tb, "remaining_seconds", lambda: 5.0)
    out = svc.submit_fit(model_type="SSC", result_id=rid, owner_id="u1", turn_seconds_left=5.0)
    assert out["pending"] and out["budget_exhausted"] and "Do NOT call mmdc_model_job again this turn" in out["error"]
    # review CX-B06: a tight TOOL deadline with plenty of TURN left is an ordinary pending result
    out2 = svc.poll(out["job_id"], owner_id="u1", wait_seconds=0, turn_seconds_left=250.0)
    assert out2["pending"] and not out2.get("budget_exhausted") and "Call mmdc_model_job" in out2["error"]
    monkeypatch.setattr(tb, "remaining_seconds", lambda: None)
    out3 = svc.poll(out["job_id"], owner_id="u1", wait_seconds=0)
    assert out3["pending"] and not out3.get("budget_exhausted")


def test_hadronic_fit_passes_the_neutrino_likelihood(monkeypatch, tmp_path):
    svc, fake, _, rid, _ = _services(monkeypatch, tmp_path)
    out = svc.submit_fit(model_type="HADRONIC", result_id=rid, owner_id="u1", likelihood_type="poisson",
                         n_icecube=1, dt=6, wait_seconds=0)
    assert fake.submitted["model_type"] == "hadronic" and fake.submitted["likelihood_type"] == "poisson"
    assert fake.submitted["n_icecube"] == 1 and fake.submitted["dt"] == 6.0
    assert out["params"]["neutrino_likelihood"]["description"].startswith("Poisson likelihood on 1 IceCube")


def test_fit_input_errors(monkeypatch, tmp_path):
    svc, _, _, rid, _ = _services(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="exactly one"):
        svc.submit_fit(model_type="SSC", owner_id="u1")
    with pytest.raises(ValueError, match="every parameter is fixed"):
        svc.submit_fit(model_type="SSC", result_id=rid, owner_id="u1", fixed_parameters=SSC)
    with pytest.raises(KeyError, match="no longer available"):
        svc.submit_fit(model_type="SSC", result_id="dlr_nope", owner_id="u1")
    with pytest.raises(KeyError, match="not a stored result"):
        svc.submit_fit(model_type="SSC", result_id="not-an-id", owner_id="u1")


def test_done_without_pdf_link_still_finishes(monkeypatch, tmp_path):
    """Live 3C 279: MMDC said 'done' with best parameters but no pdf_link for 90+ s."""
    fake = _FakeModeling()
    fake.raw = {**fake.raw, "status": "done", "pdf_link": None}
    fake.done_after = 1
    svc, _, _, rid, _ = _services(monkeypatch, tmp_path, fake)
    out = svc.submit_fit(model_type="SSC", result_id=rid, owner_id="u1", wait_seconds=0)
    assert out["success"] and out["job_status"] == "succeeded" and "pdf_report" not in out["artifacts"]
    assert any("had not produced its PDF" in w for w in out["warnings"])


def test_cx08_validation_success_false_blocks_submission(monkeypatch, tmp_path):
    fake = _FakeModeling()
    fake.validate_csv = lambda fobj: SimpleNamespace(model_dump=lambda: {"success": False, "message": "too variable",
                                                                         "data_points": 3})
    svc, _, _, rid, _ = _services(monkeypatch, tmp_path, fake)
    with pytest.raises(ValueError, match="MMDC rejected the fit input: too variable"):
        svc.submit_fit(model_type="SSC", result_id=rid, owner_id="u1", wait_seconds=0)
    assert fake.submitted == {}   # nothing was submitted


def test_failed_fit_is_reported(monkeypatch, tmp_path):
    fake = _FakeModeling()
    fake.raw = {**fake.raw, "status": "error", "pdf_link": None}
    fake.done_after = 1
    svc, _, _, rid, _ = _services(monkeypatch, tmp_path, fake)
    out = svc.submit_fit(model_type="SSC", result_id=rid, owner_id="u1", wait_seconds=0)
    assert out["success"] is False and out["job_status"] == "failed"
    assert svc.jobs.status("batch-1", owner_id="u1")["status"] == "failed"


# ── agent tools ─────────────────────────────────────────────────────────────
def test_agent_model_tools_end_to_end(monkeypatch, tmp_path):
    from tests.integration.test_agent_archive_tools import _load_agent_module

    svc, fake, store, rid, _ = _services(monkeypatch, tmp_path)
    module = _load_agent_module()
    agent = module.QuasarAgent.__new__(module.QuasarAgent)
    agent._tls = threading.local()
    agent._tls.current_user_id = "u1"
    agent.last_run_result = None
    agent._mmdc_modeling_service_instance = svc
    fake.done_after = 1
    out = agent._mmdc_model(mode="fit", model_type="SSC", result_id=rid)
    assert out["success"] and out["figure_attached"] and "_figure" not in out
    assert [c["type"] for c in agent._accumulated_run_results] == ["image"]   # no table card: the SED table stays last
    bad = agent._mmdc_model(mode="spectrum", model_type="SSC", parameters={**SSC, "bogus": 1})
    assert bad["success"] is False and "unknown SSC parameter" in bad["error"]
    missing = agent._mmdc_model_job(job_id="nope")
    assert missing["success"] is False and "Unknown MMDC fit job" in missing["error"]


def test_answer_verifier_accepts_fit_redshift_and_logz():
    """UI 2026-09-26 Q3-f: 'z = 0.5354184' and 'logZ = -34.056' were flagged as
    selection cuts although the fit result carries z and multinest logZ."""
    from core.answer_verifier import build_trace_summary, verify_answer

    result = {"success": True, "mode": "fit", "z": 0.5354184, "fit_statistics": {"logZ": -34.056, "logZ_err": 0.056}}
    ts = build_trace_summary([{"output": json.dumps(result)}], [], [])
    ts.query_arg_texts.append('{"job_id": "9c32e04d"}')   # a data tool ran, so cuts are policed
    report = verify_answer("The fit used z = 0.5354184 and reached logZ = -34.056 ± 0.056.", ts)
    assert not report.by_kind("cut"), [c.text for c in report.by_kind("cut")]
    bad = verify_answer("The sample was cut at z > 0.9.", ts)
    assert bad.by_kind("cut")   # a genuine unsupported cut is still flagged


def test_budgets_declared_for_model_tools():
    from services import tool_budgets as tb

    for name in ("mmdc_model", "mmdc_model_job", "variability_analysis"):
        assert name in tb.INNER_BUDGETS
    assert not tb.check_hierarchy(["mmdc_model", "mmdc_model_job", "variability_analysis", "mmdc_sed"])


# ── Round B independent review (reviewB ledger) regressions ─────────────────
def test_b07_propagated_error_uses_only_points_with_errors():
    rows = pd.DataFrame({"frequency_Hz": [1.0e17, 1.01e17, 1.02e17, 1.03e17] + [10 ** e for e in (9, 11, 13, 15, 19, 23)],
                         "nuFnu_erg_cm2_s": [1e-11] * 4 + [1e-12] * 6,
                         "nuFnu_err_erg_cm2_s": [1e-12, np.nan, np.nan, np.nan] + [1e-13] * 6, "upper_limit": False})
    fit, _ = mm.fit_points_from_frame(rows)
    xbin = fit.iloc[(fit["frequency"] - 1.015e17).abs().argmin()]
    assert xbin["n_rows"] == 4 and xbin["flux_err"] == pytest.approx(1e-12)   # not 2.5e-13


def test_b08_fixed_parameters_are_never_flagged_at_bound():
    rows = mm.flag_bounds("HADRONIC", {"log_B": {"value": 0.0, "error": 0.1}},
                          {"log_gamma_e_min": {"value": 1.5, "error": None}})
    by = {r["parameter"]: r for r in rows}
    assert by["log_gamma_e_min"]["fixed"] and by["log_gamma_e_min"]["at_bound"] is None


def test_b11_ownerless_caller_cannot_poll_an_owned_fit(monkeypatch, tmp_path):
    svc, fake, _, rid, _ = _services(monkeypatch, tmp_path)
    fake.done_after = 99
    out = svc.submit_fit(model_type="SSC", result_id=rid, owner_id="u1", wait_seconds=0)
    with pytest.raises(KeyError):
        svc.poll(out["job_id"], owner_id=None, wait_seconds=0)
