"""MMDC blazar emission modeling (SSC / EIC / hadronic CNN surrogates) for Quasar.

Two modes, both on the ``astro-mmdc`` SDK (services/mmdc_service.py owns the
client, the host breaker and the tool deadline):

* ``spectrum`` -- ``modeling.infer``: a fixed-parameter model spectrum, synchronous (~2 s).
* ``fit`` -- ``modeling.submit_batch`` + ``get_batch_result``: a MultiNest fit on
  MMDC's server (~3 min for SSC), tracked as an external job in
  services/datalab_job_service.py so it shows in the Jobs panel and can be
  polled with ``mmdc_model_job``.

Facts from the 2026-09-26 live probe (tmp/mmdc-integration-2026-09-26/PHASE0.md):
the server silently accepts unknown parameter names and out-of-range values, so
names and ranges are validated here against the ranges the mmdc.am web UI
enforces (its bundle, fetched 2026-09-26). Raw multi-epoch data are rejected as
"highly variable" (validation_type data_variability) and a ``flag`` column with
UL crashes the server (HTTP 500), so fits send three columns, drop upper
limits and average detections in log-frequency bins. The PDF / CSV links are
public with a 30-day cache header and no documented expiry, so they are copied
into Quasar's own plot storage.
"""

from __future__ import annotations

import io
import math
import os
import re
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from services import mmdc_service as ms

MODEL_TYPES = ("SSC", "EIC", "HADRONIC")
_API_MODEL_TYPE = {"SSC": "SSC", "EIC": "EIC", "HADRONIC": "hadronic"}

# Parameter ranges enforced by the mmdc.am web UI (assets bundle, 2026-09-26).
# These are the surrogate's validity ranges: outside them the CNN extrapolates.
PARAM_RANGES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "SSC": {
        "log_B": (-3.0, 2.0), "log_electron_luminosity": (42.0, 48.0), "log_gamma_cut": (2.0, 8.0),
        "log_gamma_min": (1.5, 5.0), "log_radius": (15.0, 18.0), "lorentz_factor": (3.0, 50.0),
        "spectral_index": (1.8, 5.0),
    },
    "EIC": {
        "log_B": (-3.0, 2.5), "log_electron_luminosity": (42.0, 48.0), "log_gamma_cut": (2.0, 6.0),
        "log_gamma_min": (1.5, 5.0), "log_radius": (15.0, 18.0), "lorentz_factor": (3.0, 50.0),
        "spectral_index": (1.8, 5.0), "log_Ld": (43.5, 47.0), "log_MBH": (7.0, 10.0),
        "log_nu_BLR": (14.5, 16.0), "log_nu_DT": (12.5, 14.0),
    },
    "HADRONIC": {
        "log_B": (-3.0, 3.5), "log_Le": (42.5, 48.5), "log_gamma_e_min": (1.5, 2.0), "log_gamma_e_cut": (2.0, 8.0),
        "log_gamma_p_cut": (3.0, 11.0), "log_Lp": (42.0, 52.0), "log_R": (14.5, 18.0), "lorentz_factor": (3.5, 80.0),
        "pe": (1.75, 5.0), "pp": (1.65, 3.45),
    },
}
Z_RANGE = (0.0, 10.0)
NEUTRINO_RANGES = {"n_icecube": (1, 100), "dt": (1.0, 120.0), "x1": (1.0, 1000.0), "x2": (100.0, 1.0e4),
                   "y": (-16.0, -9.0)}
PARAM_LABELS = {
    "log_B": "log10 B [G] (magnetic field)",
    "log_electron_luminosity": "log10 Le [erg/s] (electron luminosity)",
    "log_Le": "log10 Le [erg/s] (electron luminosity)",
    "log_gamma_cut": "log10 gamma_max (electron cut-off Lorentz factor)",
    "log_gamma_e_cut": "log10 gamma_e,max (electron cut-off)",
    "log_gamma_min": "log10 gamma_min (minimum electron Lorentz factor)",
    "log_gamma_e_min": "log10 gamma_e,min (minimum electron Lorentz factor)",
    "log_gamma_p_cut": "log10 gamma_p,max (proton cut-off)",
    "log_radius": "log10 R [cm] (emitting region radius)",
    "log_R": "log10 R [cm] (emitting region radius)",
    "lorentz_factor": "delta (Doppler factor; 'lorentz_factor' in the MMDC API)",
    "spectral_index": "p (electron spectral index)",
    "pe": "p_e (electron spectral index)",
    "pp": "p_p (proton spectral index)",
    "log_Lp": "log10 Lp [erg/s] (proton luminosity)",
    "log_Ld": "log10 Ld [erg/s] (accretion-disk luminosity)",
    "log_MBH": "log10 M_BH [M_sun] (black-hole mass)",
    "log_nu_BLR": "log10 nu_BLR [Hz] (BLR photon frequency)",
    "log_nu_DT": "log10 nu_DT [Hz] (dusty-torus photon frequency)",
}

MODEL_VALIDATE_TIMEOUT_S = ms._env_f("MMDC_MODEL_VALIDATE_TIMEOUT_SECONDS", 30.0)
MODEL_SUBMIT_TIMEOUT_S = ms._env_f("MMDC_MODEL_SUBMIT_TIMEOUT_SECONDS", 30.0)
MODEL_INFER_TIMEOUT_S = ms._env_f("MMDC_MODEL_INFER_TIMEOUT_SECONDS", 30.0)
FIT_SUBMIT_WAIT_S = ms._env_f("MMDC_FIT_SUBMIT_WAIT_SECONDS", 120.0)  # fewer LLM round trips per fit
# A finished fit returns ~8.5k posterior samples and ~280 model curves: 10 s and once 45 s were not enough live.
FIT_STATUS_TIMEOUT_S = ms._env_f("MMDC_FIT_STATUS_TIMEOUT_SECONDS", 60.0)
FIT_JOB_WAIT_S = ms._env_f("MMDC_FIT_JOB_WAIT_SECONDS", 90.0)
FIT_POLL_INTERVAL_S = 10.0
ARTIFACT_TIMEOUT_S = ms._env_f("MMDC_ARTIFACT_TIMEOUT_SECONDS", 10.0)
NED_TIMEOUT_S = ms._env_f("MMDC_NED_TIMEOUT_SECONDS", 20.0)
DEFAULT_BIN_DEX = 0.1
MAX_CONTEXTS = 50
MIN_FIT_POINTS = 6
SURROGATE_CAVEAT = (
    "The MMDC fit uses convolutional-neural-network surrogates of the SSC / EIC / hadronic models, valid only inside "
    "their training ranges (the ranges listed in parameter_ranges). A best-fit value flagged at_bound sits at the edge of "
    "that range: the posterior is truncated there and the true value may lie outside it.")


# ── validation ──────────────────────────────────────────────────────────────
def normalize_model_type(model_type: Any) -> str:
    key = str(model_type or "SSC").strip().upper().replace("-", "").replace(" ", "")
    if key in ("LEPTOHADRONIC", "HADRON", "PROTON"):
        key = "HADRONIC"
    if key not in MODEL_TYPES:
        raise ValueError(f"model_type must be one of {list(MODEL_TYPES)} (got {model_type!r})")
    return key


def validate_parameters(model: str, params: Optional[Dict[str, Any]], *, require_all: bool) -> Dict[str, float]:
    ranges = PARAM_RANGES[model]
    params = dict(params or {})
    unknown = sorted(set(params) - set(ranges))
    if unknown:
        raise ValueError(f"unknown {model} parameter(s) {unknown}; valid names: {sorted(ranges)}")
    out: Dict[str, float] = {}
    bad = []
    for name, value in params.items():
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a number (got {value!r})") from None
        lo, hi = ranges[name]
        if not (math.isfinite(v) and lo <= v <= hi):
            bad.append(f"{name}={value} (valid {lo} to {hi})")
        out[name] = v
    if bad:
        raise ValueError("parameter(s) outside the surrogate's training range: " + "; ".join(bad))
    if require_all:
        missing = sorted(set(ranges) - set(out))
        if missing:
            raise ValueError(f"mode='spectrum' needs every {model} parameter; missing {missing}")
    return out


def validate_z(z: Any) -> float:
    try:
        v = float(z)
    except (TypeError, ValueError):
        raise ValueError(f"z must be a number (got {z!r})") from None
    if not (math.isfinite(v) and Z_RANGE[0] < v <= Z_RANGE[1]):
        raise ValueError(f"z must lie in ({Z_RANGE[0]}, {Z_RANGE[1]}] (got {z!r})")
    return v


def validate_neutrino(model: str, likelihood_type: Optional[str], n_icecube: Any = None, dt: Any = None,
                      x1: Any = None, x2: Any = None, y: Any = None) -> Dict[str, Any]:
    """The hadronic neutrino likelihood: Poisson on an IceCube event count in a
    time window, or chi2 on a neutrino flux in an energy band. Every input is
    required explicitly for the chosen type (no defaults are invented)."""
    given = {k: v for k, v in dict(n_icecube=n_icecube, dt=dt, x1=x1, x2=x2, y=y).items() if v is not None}
    lt = str(likelihood_type).strip().lower() if likelihood_type else None
    if lt is None:
        if given:
            raise ValueError("neutrino inputs were given without likelihood_type ('poisson' or 'chi2')")
        return {}
    if model != "HADRONIC":
        raise ValueError("a neutrino likelihood applies only to model_type='HADRONIC'")
    if lt not in ("poisson", "chi2"):
        raise ValueError("likelihood_type must be 'poisson' or 'chi2'")
    need = ("n_icecube", "dt") if lt == "poisson" else ("x1", "x2", "y")
    other = ("x1", "x2", "y") if lt == "poisson" else ("n_icecube", "dt")
    missing = [k for k in need if k not in given]
    if missing:
        raise ValueError(f"likelihood_type='{lt}' needs {list(need)}; missing {missing}")
    stray = [k for k in other if k in given]
    if stray:
        raise ValueError(f"{stray} do not apply to likelihood_type='{lt}'")
    out: Dict[str, Any] = {"likelihood_type": lt}
    for k in need:
        lo, hi = NEUTRINO_RANGES[k]
        try:
            v = float(given[k])
        except (TypeError, ValueError):
            raise ValueError(f"{k} must be a number (got {given[k]!r})") from None
        if not (math.isfinite(v) and lo <= v <= hi):
            raise ValueError(f"{k}={given[k]} is outside the MMDC range {lo} to {hi}")
        if k == "n_icecube":
            if not float(v).is_integer():
                raise ValueError("n_icecube must be a whole number of events")
            v = int(v)
        out[k] = v
    if lt == "chi2" and out["x1"] >= out["x2"]:
        raise ValueError("x1 (E1, TeV) must be below x2 (E2, TeV)")
    out["description"] = (
        f"Poisson likelihood on {out['n_icecube']} IceCube event(s) in a {out['dt']:g}-month window"
        if lt == "poisson" else
        f"chi2 likelihood on a neutrino flux log10(F) = {out['y']:g} erg cm^-2 s^-1 between {out['x1']:g} and {out['x2']:g} TeV")
    return out


# ── fit input ───────────────────────────────────────────────────────────────
def _pick(df: pd.DataFrame, *names: str) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None


def fit_points_from_frame(df: pd.DataFrame, bin_dex: float = DEFAULT_BIN_DEX) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """SED rows -> the three-column table MMDC's fitter accepts.

    Upper limits and non-positive fluxes are dropped (MMDC's fitter takes
    detections only: a UL flag column makes the server fail). Detections are
    averaged in ``bin_dex`` log10-frequency bins: flux = mean, error = the larger
    of the in-bin scatter and the propagated measurement error. That is what
    MMDC's validator asks for with multi-epoch data, and it is a time-averaged
    SED, so the answer must say so.
    """
    fcol = _pick(df, "frequency_Hz", "frequency", "freq", "nu")
    ycol = _pick(df, "nuFnu_erg_cm2_s", "flux", "nufnu")
    ecol = _pick(df, "nuFnu_err_erg_cm2_s", "flux_err", "err_flux", "nufnu_err")
    if not fcol or not ycol:
        raise ValueError(f"the table needs frequency and nuFnu columns (got {list(df.columns)})")
    rep: Dict[str, Any] = {"n_rows": int(len(df)), "bin_width_dex": float(bin_dex)}
    work = pd.DataFrame({
        "frequency": pd.to_numeric(df[fcol], errors="coerce"),
        "flux": pd.to_numeric(df[ycol], errors="coerce"),
        "flux_err": pd.to_numeric(df[ecol], errors="coerce") if ecol else np.nan,
    })
    ulcol = _pick(df, "upper_limit", "is_upper_limit")
    if ulcol is None and "flag" in df.columns:
        ul = df["flag"].map(ms._is_ul).astype(bool)
    else:
        ul = df[ulcol].astype(bool) if ulcol else pd.Series(False, index=df.index)
    rep["n_upper_limits_dropped"] = int(ul.sum())
    work = work[~ul.values]
    bad = ~(work["frequency"] > 0) | ~(work["flux"] > 0)
    rep["n_nonpositive_or_missing_dropped"] = int(bad.sum())
    work = work[~bad]
    if not len(work):
        raise ValueError("no detections left to fit after dropping upper limits and non-positive fluxes")
    zero_err = ~(work["flux_err"] > 0)
    rep["n_rows_without_error"] = int(zero_err.sum())
    if bin_dex and bin_dex > 0:
        work = work.assign(_bin=np.floor(np.log10(work["frequency"].values) / float(bin_dex)))
        out = []
        for _, g in work.groupby("_bin"):
            f = g["flux"].values
            e = g["flux_err"].where(g["flux_err"] > 0).values
            nu = float(10 ** np.mean(np.log10(g["frequency"].values)))
            mean = float(np.mean(f))
            n_e = int(np.isfinite(e).sum())
            # propagated error of the mean over the points that HAVE an error (review CX-B07)
            stat = float(np.sqrt(np.nansum(e ** 2)) / n_e) if n_e else 0.0
            spread = float(np.std(f, ddof=1)) if len(f) > 1 else 0.0
            err = max(stat, spread)
            if err <= 0:
                err = 0.1 * mean  # a lone point with no quoted error: 10 %, reported below
                rep["n_bins_with_assumed_10pct_error"] = rep.get("n_bins_with_assumed_10pct_error", 0) + 1
            out.append((nu, mean, err, len(g)))
        fit = pd.DataFrame(out, columns=["frequency", "flux", "flux_err", "n_rows"])
        rep["method"] = (f"detections averaged in {bin_dex:g}-dex log10(frequency) bins; flux = mean, error = max(in-bin "
                         "scatter, propagated measurement error); a time-averaged SED")
    else:
        fit = work.assign(n_rows=1)[["frequency", "flux", "flux_err", "n_rows"]].copy()
        missing = ~(fit["flux_err"] > 0)
        fit.loc[missing, "flux_err"] = 0.1 * fit.loc[missing, "flux"]
        rep["n_bins_with_assumed_10pct_error"] = int(missing.sum())
        rep["method"] = "unbinned detections"
    fit = fit.sort_values("frequency").reset_index(drop=True)
    rep["n_fit_points"] = int(len(fit))
    rep["frequency_range_Hz"] = [float(f"{fit['frequency'].min():.4g}"), float(f"{fit['frequency'].max():.4g}")]
    ranges = sorted({ms.classify_range(v) for v in fit["frequency"]})
    rep["ranges_covered"] = ranges
    warn = []
    if "gamma" not in ranges:
        warn.append("no gamma-ray points: the inverse-Compton component is unconstrained")
    if "xray" not in ranges:
        warn.append("no X-ray points")
    if warn:
        rep["coverage_warnings"] = warn
    if len(fit) < MIN_FIT_POINTS:
        raise ValueError(f"only {len(fit)} fit points after binning (need >= {MIN_FIT_POINTS}); widen the window or "
                         "use window_mode='overlap' / include_undated=true in mmdc_sed")
    return fit, rep


def _named_bytes(text: str, name: str = "quasar_sed_fit.csv") -> io.BytesIO:
    # The SDK takes the upload filename from the file object's .name.
    bio = io.BytesIO(text.encode("utf-8"))
    bio.name = name
    return bio


def fit_csv_text(fit: pd.DataFrame) -> str:
    buf = io.StringIO()
    fit[["frequency", "flux", "flux_err"]].to_csv(buf, index=False, float_format="%.6e")
    return buf.getvalue()


def parse_uploaded_csv(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(str(text)))
    if not {"frequency", "flux"} <= {c.strip().lower() for c in df.columns}:
        raise ValueError("csv_text needs columns frequency, flux, flux_err (nuFnu in erg cm^-2 s^-1, frequency in Hz)")
    df.columns = [c.strip() for c in df.columns]
    return df


# ── results ─────────────────────────────────────────────────────────────────
def flag_bounds(model: str, best: Dict[str, Any], fixed: Dict[str, Any]) -> List[Dict[str, Any]]:
    ranges = PARAM_RANGES[model]
    rows = []
    for name in list(ranges):
        src = fixed if name in fixed else best
        if name not in src:
            continue
        pv = src[name]
        val = float(pv["value"] if isinstance(pv, dict) else getattr(pv, "value", pv))
        err = pv.get("error") if isinstance(pv, dict) else getattr(pv, "error", None)
        err = float(err) if err is not None else None
        lo, hi = ranges[name]
        edge = 0.02 * (hi - lo)
        at_bound = None
        if name in fixed:
            pass  # a fixed value has no posterior to truncate (review CX-B08)
        elif val < lo or val > hi:
            at_bound = "outside training range"
        elif val - lo <= edge:
            at_bound = "at lower bound"
        elif hi - val <= edge:
            at_bound = "at upper bound"
        elif err and val - err <= lo:
            # The value is interior but its 1-sigma interval reaches the edge (UI 2026-09-26 Q4:
            # log_B = -1.66 +- 1.46 on [-3, 3.5] was mislabelled "at lower bound").
            at_bound = "1-sigma interval reaches the lower bound"
        elif err and val + err >= hi:
            at_bound = "1-sigma interval reaches the upper bound"
        rows.append({
            "parameter": name,
            "meaning": PARAM_LABELS.get(name, name),
            "value": float(f"{val:.5g}"),
            "error": None if err is None else float(f"{err:.3g}"),
            "fixed": name in fixed,
            "range": [lo, hi],
            "at_bound": at_bound,
        })
    return rows


def curve_peaks(nu: Sequence[float], nufnu: Sequence[float]) -> List[Dict[str, float]]:
    """Local maxima of log nuFnu (synchrotron / high-energy humps), highest first."""
    x = np.log10(np.asarray(nu, dtype=float))
    y = np.asarray(nufnu, dtype=float)
    ok = np.isfinite(x) & (y > 1e-19)
    x, y = x[ok], np.log10(y[ok])
    peaks = []
    for i in range(1, len(y) - 1):
        if y[i] >= y[i - 1] and y[i] > y[i + 1] and (y[i] - y.min()) > 0.5:
            peaks.append({"nu_Hz": float(f"{10 ** x[i]:.4g}"), "nuFnu": float(f"{10 ** y[i]:.4g}")})
    peaks.sort(key=lambda p: -p["nuFnu"])
    return peaks[:3]


def _curves_from_data(data: Optional[Dict[str, Any]], n_samples: int = 20) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    data = data or {}
    best = data.get("best")
    samples = [v for k, v in data.items() if k != "best" and isinstance(v, dict) and v.get("nu")]
    step = max(1, len(samples) // n_samples) if samples else 1
    return best, samples[::step][:n_samples]


# ── artifacts ───────────────────────────────────────────────────────────────
def copy_artifacts(links: Dict[str, Optional[str]], job_id: str, fetch: Callable[[str], bytes]) -> Dict[str, Any]:
    """Copy MMDC's PDF / CSV outputs into /plots so links outlive MMDC's cache."""
    from services.plotting import PLOT_OUTPUT_DIR

    out: Dict[str, Any] = {}
    safe = re.sub(r"[^A-Za-z0-9_-]", "", job_id)[:40] or uuid.uuid4().hex[:12]
    for key, url in links.items():
        if not url:
            continue
        if not str(url).startswith(f"{ms.MMDC_BASE_URL}/media/"):
            out[key] = {"mmdc_url": url, "copied": False, "error": "unexpected host; not copied"}
            continue
        ext = ".pdf" if str(url).lower().endswith(".pdf") else ".csv"
        name = f"mmdc_fit_{safe}_{key}{ext}"
        try:
            content = fetch(url)
            os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
            with open(os.path.join(PLOT_OUTPUT_DIR, name), "wb") as fh:
                fh.write(content)
            out[key] = {"url": f"/plots/{name}", "mmdc_url": url, "copied": True, "bytes": len(content)}
        except Exception as exc:  # noqa: BLE001 - a missing copy leaves the MMDC link
            out[key] = {"mmdc_url": url, "copied": False, "error": str(exc)[:200]}
    return out


# ── the service ─────────────────────────────────────────────────────────────
class MmdcModelingService:
    def __init__(self, mmdc: Optional[ms.MmdcService] = None, *, job_service: Any = None,
                 fetch_bytes: Optional[Callable[[str], bytes]] = None, ned_lookup: Optional[Callable[[str], Optional[float]]] = None):
        self.mmdc = mmdc or ms.default_mmdc_service()
        self._job_service = job_service
        self._fetch_bytes = fetch_bytes
        self._ned_lookup = ned_lookup
        # batch id -> context the poller needs (source table, fit points, params)
        self._contexts: Dict[str, Dict[str, Any]] = {}

    @property
    def jobs(self):
        if self._job_service is None:
            from services.datalab_job_service import default_job_service

            self._job_service = default_job_service()
        return self._job_service

    def _fetch(self, url: str) -> bytes:
        if self._fetch_bytes is not None:
            return self._fetch_bytes(url)
        import httpx

        def _get():
            r = httpx.get(url, timeout=ARTIFACT_TIMEOUT_S, follow_redirects=False)
            r.raise_for_status()
            return r.content

        return self.mmdc.call("fit artifact", _get, ARTIFACT_TIMEOUT_S)

    # redshift ----------------------------------------------------------------
    def ned_redshift(self, name: str) -> Optional[float]:
        if self._ned_lookup is not None:
            return self._ned_lookup(name)
        from services.tool_budgets import bounded_timeout, call_bounded

        def _q():
            from astroquery.ipac.ned import Ned

            t = Ned.query_object(name)
            if len(t) and "Redshift" in t.colnames:
                v = t["Redshift"][0]
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    return None
                return v if math.isfinite(v) and v > 0 else None
            return None

        try:
            return call_bounded(_q, bounded_timeout(NED_TIMEOUT_S, label="NED redshift"), label="NED redshift")
        except Exception:  # noqa: BLE001 - reported as "not found" by the caller
            return None

    def resolve_z(self, z: Any, meta: Dict[str, Any], target: Optional[str]) -> Tuple[float, str]:
        """Explicit z, else MMDC's redshift for the source, else NED. Never a guess."""
        if z is not None:
            zc = validate_z(z)
            mz = ((meta or {}).get("mmdc") or {}).get("redshift")
            if mz is not None and abs(float(mz) - zc) < 1e-9:
                return zc, "MMDC source info (get_info), passed through by the caller"
            return zc, "given by the caller"
        mm = (meta or {}).get("mmdc") or {}
        if mm.get("redshift") is not None:
            return validate_z(mm["redshift"]), "MMDC source info (get_info) for this source"
        name = target or mm.get("target")
        if name:
            v = self.ned_redshift(str(name))
            if v is not None:
                return validate_z(v), f"NED (query_object '{name}')"
        raise ValueError("no redshift available: MMDC has none for this source and NED returned none. Ask the user "
                         "for z (do not guess), then call again with z=...")

    # spectrum ------------------------------------------------------------------
    def spectrum(self, *, model_type: str, z: Any, parameters: Dict[str, Any], ebl: bool = True,
                 result_id: Optional[str] = None, target_name: Optional[str] = None, owner_id: Optional[str] = None,
                 user_id: Optional[str] = None) -> Dict[str, Any]:
        model = normalize_model_type(model_type)
        params = validate_parameters(model, parameters, require_all=True)
        frame, meta = (None, {})
        if result_id:
            frame, meta = ms.load_result(result_id, owner_id)
        zval, zsrc = self.resolve_z(z, meta, target_name)
        client = self.mmdc.client(user_id)
        res = self.mmdc.call("model inference", lambda: client.modeling.infer(
            z=zval, ebl=bool(ebl), model_type=_API_MODEL_TYPE[model], parameters=params), MODEL_INFER_TIMEOUT_S)
        nu, nufnu = list(res.nu), list(res.nuFnu)
        out: Dict[str, Any] = {
            "success": True, "mode": "spectrum", "model_type": model, "z": zval, "z_source": zsrc, "ebl": bool(ebl),
            "parameters": [{"parameter": k, "meaning": PARAM_LABELS.get(k, k), "value": v,
                            "range": list(PARAM_RANGES[model][k])} for k, v in params.items()],
            "model_points": len(nu),
            "nu_range_Hz": [float(f"{min(nu):.4g}"), float(f"{max(nu):.4g}")] if nu else None,
            "spectral_peaks": curve_peaks(nu, nufnu),
            "attribution": ms.attribution(modeling=True),
            "caveat": SURROGATE_CAVEAT,
        }
        if getattr(res, "neutrino_energy", None) is not None and getattr(res, "eFe_nu_tot", None) is not None:
            ne, nf = list(res.neutrino_energy), list(res.eFe_nu_tot)
            if ne:
                i = int(np.argmax(nf))
                out["neutrino_spectrum"] = {"points": len(ne), "peak_energy": float(f"{ne[i]:.4g}"),
                                            "peak_E2F": float(f"{nf[i]:.4g}"),
                                            "units_note": "units as returned by MMDC (not documented in the SDK)"}
        curve = {"name": f"{model} model (fixed parameters)", "nu": nu, "nuFnu": nufnu}
        self._attach_plot(out, frame, [curve], f"{meta.get('mmdc', {}).get('target') or target_name or ''} "
                                               f"{model} model spectrum".strip())
        return out

    # fit -----------------------------------------------------------------------
    def submit_fit(self, *, model_type: str, z: Any = None, ebl: bool = True, result_id: Optional[str] = None,
                   csv_text: Optional[str] = None, target_name: Optional[str] = None,
                   fixed_parameters: Optional[Dict[str, Any]] = None, likelihood_type: Optional[str] = None,
                   n_icecube: Any = None, dt: Any = None, x1: Any = None, x2: Any = None, y: Any = None,
                   bin_dex: float = DEFAULT_BIN_DEX, owner_id: Optional[str] = None, user_id: Optional[str] = None,
                   wait_seconds: Optional[float] = None, turn_seconds_left: Optional[float] = None) -> Dict[str, Any]:
        model = normalize_model_type(model_type)
        fixed = validate_parameters(model, fixed_parameters, require_all=False)
        if len(fixed) >= len(PARAM_RANGES[model]):
            raise ValueError("every parameter is fixed; use mode='spectrum' instead")
        nu_lik = validate_neutrino(model, likelihood_type, n_icecube, dt, x1, x2, y)
        if bool(result_id) == bool(csv_text):
            raise ValueError("give exactly one of result_id (an mmdc_sed table) or csv_text")
        meta: Dict[str, Any] = {}
        if result_id:
            frame, meta = ms.load_result(result_id, owner_id)
            source = {"result_id": result_id, "table_rows": int(len(frame))}
            mm = meta.get("mmdc") or {}
            if mm.get("window"):
                w = mm["window"]
                source["window"] = {k: w.get(k) for k in ("rule_applied", "window_start_mjd", "window_end_mjd",
                                                          "window_start_date", "window_end_date",
                                                          "undated_included")}
        else:
            frame = parse_uploaded_csv(csv_text)
            source = {"csv_text_rows": int(len(frame))}
        zval, zsrc = self.resolve_z(z, meta, target_name)
        fit, prep = fit_points_from_frame(frame, bin_dex)
        text = fit_csv_text(fit)
        client = self.mmdc.client(user_id)
        val = self.mmdc.call("CSV validation", lambda: client.modeling.validate_csv(_named_bytes(text)),
                             MODEL_VALIDATE_TIMEOUT_S)
        vd = val.model_dump() if hasattr(val, "model_dump") else dict(val)
        if vd.get("success") is False:
            # A 200 carrying success=false is a rejection too: never submit past it (guard CX-08).
            raise ValueError(f"MMDC rejected the fit input: {vd.get('message') or 'validation failed'} "
                             f"({prep['n_fit_points']} binned points from {prep['n_rows']} rows)")
        kw = {k: v for k, v in nu_lik.items() if k != "description"}
        sub = self.mmdc.call("fit submission", lambda: client.modeling.submit_batch(
            _named_bytes(text), z=zval, ebl=bool(ebl), model_type=_API_MODEL_TYPE[model],
            fixed_parameters=fixed or None, **kw), MODEL_SUBMIT_TIMEOUT_S)
        job_id = str(sub.batch_result_id)
        params = {"model_type": model, "z": zval, "z_source": zsrc, "ebl": bool(ebl), "fixed_parameters": fixed,
                  "neutrino_likelihood": nu_lik or None, "source": source,
                  "target": target_name or (meta.get("mmdc") or {}).get("target")}
        self.jobs.register_external(job_id, kind="mmdc_fit", params=params, status="submitted", owner_id=owner_id)
        self._contexts[job_id] = {"frame": frame, "fit_points": fit, "prep": prep, "params": params,
                                  "owner": str(owner_id) if owner_id else None, "created": time.time(),
                                  "validation": {k: vd.get(k) for k in ("success", "message", "data_points")}}
        for old in sorted(self._contexts, key=lambda k: self._contexts[k].get("created", 0))[:-MAX_CONTEXTS]:
            self._contexts.pop(old, None)  # bounded: each keeps an SED frame (review CX-B16)
        return self.poll(job_id, owner_id=owner_id, user_id=user_id,
                         wait_seconds=FIT_SUBMIT_WAIT_S if wait_seconds is None else wait_seconds,
                         turn_seconds_left=turn_seconds_left)

    def poll(self, job_id: str, *, owner_id: Optional[str] = None, user_id: Optional[str] = None,
             wait_seconds: Optional[float] = None, turn_seconds_left: Optional[float] = None) -> Dict[str, Any]:
        """``turn_seconds_left``: what remains of the runner's per-turn tool budget (the
        tool's own deadline is always tight in fit mode, so it cannot tell "this call is
        out of time" from "this TURN is out of time"; review CX-B06)."""
        from services.tool_budgets import remaining_seconds

        rec = self.jobs.status(job_id, owner_id=owner_id)  # KeyError for another user's job
        ctx_owner = (self._contexts.get(job_id) or {}).get("owner", "unknown")
        if ctx_owner != "unknown" and (ctx_owner or None) != (str(owner_id) if owner_id else None):
            # DatalabJobService skips its owner check for owner_id=None (review CX-B11)
            raise KeyError(f"Unknown Data Lab job: {job_id}")
        if rec.get("status") == "succeeded" and isinstance(rec.get("result"), dict):
            return rec["result"]
        ctx = self._contexts.get(job_id) or {}
        params = rec.get("params") or ctx.get("params") or {}
        client = self.mmdc.client(user_id)
        limit = FIT_JOB_WAIT_S if wait_seconds is None else float(wait_seconds)
        t0 = time.monotonic()
        while True:
            br = self.mmdc.call("fit status", lambda: client.modeling.get_batch_result(job_id), FIT_STATUS_TIMEOUT_S)
            status = str(getattr(br, "status", None) or "unknown")
            if status in ("error", "failed", "cancelled"):
                self.jobs.update_external(job_id, owner_id=owner_id, status="failed", error=f"MMDC status {status}")
                return {"success": False, "job_id": job_id, "job_status": "failed",
                        "error": f"The MMDC fit ended with status '{status}'.", "params": params}
            # Finished = the PDF exists, or MMDC says done and the parameters are in
            # (live 3C 279, 2026-09-26: status 'done' with no pdf_link for 90+ s).
            if getattr(br, "pdf_link", None) or (status == "done" and getattr(br, "best_parameters", None)):
                result = self._finish(job_id, br, ctx, params)
                # the record keeps the model-facing result; the figure is shown once
                self.jobs.update_external(job_id, owner_id=owner_id, status="succeeded",
                                          result={k: v for k, v in result.items() if not k.startswith("_")})
                return result
            self.jobs.update_external(job_id, owner_id=owner_id, status="running")
            left = limit - (time.monotonic() - t0)
            rem = remaining_seconds()
            if rem is not None and rem - (3 * ARTIFACT_TIMEOUT_S + FIT_STATUS_TIMEOUT_S + 10) < left:
                left = rem - (3 * ARTIFACT_TIMEOUT_S + FIT_STATUS_TIMEOUT_S + 10)
            turn_out = turn_seconds_left is not None and (
                turn_seconds_left - (time.monotonic() - t0) < 3 * ARTIFACT_TIMEOUT_S + FIT_STATUS_TIMEOUT_S + 10 + FIT_POLL_INTERVAL_S + 20)
            if left <= FIT_POLL_INTERVAL_S:
                if turn_out:
                    # No time left in this turn: re-polling now returns instantly (UI 2026-09-26 Q3b looped).
                    return {"success": False, "pending": True, "budget_exhausted": True, "job_id": job_id,
                            "job_status": "running", "mmdc_status": status, "params": params,
                            "fit_input": ctx.get("prep"),
                            "error": (f"The MMDC {params.get('model_type', '')} fit is still running and this turn has "
                                      "no time left to wait. Do NOT call mmdc_model_job again this turn: tell the user "
                                      f"the fit is still running with job_id '{job_id}' and that asking again later "
                                      "collects it.")}
                return {"success": False, "pending": True, "job_id": job_id, "job_status": "running",
                        "mmdc_status": status, "waited_s": round(time.monotonic() - t0, 1), "retry_after_s": 30,
                        "params": params,
                        "fit_input": ctx.get("prep"),
                        "error": (f"The MMDC {params.get('model_type', '')} fit is still running on MMDC's server (an SSC "
                                  "fit takes about 3 minutes). Call mmdc_model_job with job_id="
                                  f"'{job_id}' to collect it. This is not a failure.")}
            time.sleep(FIT_POLL_INTERVAL_S)

    def _finish(self, job_id: str, br: Any, ctx: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        model = params.get("model_type") or normalize_model_type(getattr(br, "model_type", "SSC"))
        best = {k: (v.model_dump() if hasattr(v, "model_dump") else v) for k, v in (br.best_parameters or {}).items()}
        fixed = {k: (v.model_dump() if hasattr(v, "model_dump") else v) for k, v in (br.fixed_parameters or {}).items()}
        for k, v in (params.get("fixed_parameters") or {}).items():
            fixed.setdefault(k, {"value": v, "error": None})
        table = flag_bounds(model, best, fixed)
        flagged = [r["parameter"] for r in table if r["at_bound"]]
        best_curve, samples = _curves_from_data(br.data)
        stats = br.multinest_stats or {}
        artifacts = copy_artifacts({"pdf_report": br.pdf_link, "best_parameters_csv": br.csv_best_parameters_link,
                                    "best_model_csv": br.csv_best_model_link}, job_id, self._fetch)
        out: Dict[str, Any] = {
            "success": True, "mode": "fit", "job_id": job_id, "job_status": "succeeded",
            "model_type": model, "z": params.get("z"), "z_source": params.get("z_source"), "ebl": params.get("ebl"),
            "best_fit_parameters": table,
            "parameters_at_bound": flagged,
            "fit_statistics": {k: (float(f"{float(v):.6g}") if isinstance(v, (int, float)) else v) for k, v in stats.items()},
            "posterior_samples": len(br.equal_weighted_posterior or []),
            "neutrino_likelihood": params.get("neutrino_likelihood"),
            "fit_input": ctx.get("prep"),
            "fit_source": params.get("source"),
            "artifacts": artifacts,
            "spectral_peaks": curve_peaks(best_curve["nu"], best_curve["nuFnu"]) if best_curve else [],
            "parameter_ranges_source": "ranges enforced by the mmdc.am web UI (2026-09-26); the surrogate's training ranges",
            "caveat": SURROGATE_CAVEAT,
            "attribution": ms.attribution(modeling=True),
            "answer_requirements": [
                "Quote each best-fit parameter with its error from best_fit_parameters; mark fixed ones as fixed.",
                "Say the fit used the binned, time-averaged SED in fit_input (how many rows, upper limits dropped, bins).",
                "Flag every parameter in parameters_at_bound and give the surrogate-validity caveat.",
                "State z and z_source.",
                "Cite the MMDC and the two MMDC modeling papers from attribution.",
            ],
        }
        warns = []
        edge = [r["parameter"] for r in table if r["at_bound"] in ("at lower bound", "at upper bound", "outside training range")]
        wide = [r["parameter"] for r in table if r["at_bound"] and r["parameter"] not in edge]
        if edge:
            warns.append(f"{', '.join(edge)}: best-fit value at or beyond the edge of the surrogate's training range; "
                         "the posterior is truncated there and the true value may lie outside it.")
        if wide:
            warns.append(f"{', '.join(wide)}: the value is inside the training range but its 1-sigma interval "
                         "reaches a range edge, so the error is truncated (poorly constrained); do not call it an edge value.")
        if not br.pdf_link:
            warns.append("MMDC reported the fit done but had not produced its PDF report (corner plot) yet; the "
                         "parameters are final, the report link is absent.")
        if warns:
            out["warnings"] = warns
        curves = []
        for i, s in enumerate(samples):
            curves.append({"name": "posterior samples", "nu": s["nu"], "nuFnu": s["nuFnu"], "color": "#f28e2b",
                           "width": 1, "opacity": 0.25, "showlegend": i == 0})
        if best_curve:
            curves.append({"name": f"{model} best fit", "nu": best_curve["nu"], "nuFnu": best_curve["nuFnu"],
                           "color": "#d62728", "width": 2.5})
        self._attach_plot(out, ctx.get("frame"), curves, f"{params.get('target') or ''} {model} fit".strip(),
                          fit_points=ctx.get("fit_points"))
        return out

    # plot ----------------------------------------------------------------------
    def _attach_plot(self, out: Dict[str, Any], frame: Optional[pd.DataFrame], curves: List[Dict[str, Any]],
                     title: str, fit_points: Optional[pd.DataFrame] = None) -> None:
        sed = None
        if isinstance(frame, pd.DataFrame) and len(frame):
            try:
                sed = frame if "upper_limit" in frame.columns and "undated" in frame.columns else None
                if sed is None:
                    sed = ms.normalize_sed_csv(pd.DataFrame({
                        "frequency": frame[_pick(frame, "frequency_Hz", "frequency")],
                        "flux": frame[_pick(frame, "nuFnu_erg_cm2_s", "flux")],
                        "flux_err": frame[_pick(frame, "nuFnu_err_erg_cm2_s", "flux_err")] if _pick(frame, "nuFnu_err_erg_cm2_s", "flux_err") else np.nan,
                        "MJD_start": ms.UNDATED_MJD, "MJD_end": ms.UNDATED_MJD, "flag": "", "catalog": "uploaded"}))
            except Exception:  # noqa: BLE001 - the model curve alone still plots
                sed = None
        empty = pd.DataFrame(columns=ms.SED_COLUMNS).astype({"upper_limit": bool, "undated": bool})
        base = sed if sed is not None else empty
        plot_df, _ = ms.downsample_sed(base, 3000) if len(base) else (base, {})
        spec = ms.sed_plotly_spec(plot_df, title, color_by="epoch", model_curves=curves)
        if isinstance(fit_points, pd.DataFrame) and len(fit_points):
            spec["data"].append({
                "type": "scatter", "mode": "markers", "name": "fit input (binned)",
                "x": fit_points["frequency"].tolist(), "y": fit_points["flux"].tolist(),
                "error_y": {"type": "data", "array": fit_points["flux_err"].tolist(), "visible": True, "color": "#111"},
                "marker": {"symbol": "diamond-open", "size": 9, "color": "#111", "line": {"width": 1.5}},
                "hovertemplate": "binned fit point<br>ν = %{x:.3e} Hz<br>νFν = %{y:.3e}<extra></extra>"})
        png = ms.sed_png(plot_df, title, self.mmdc.plotting_service, model_curves=curves, name="mmdc_model")
        out["_figure"] = {**png, "plotly_spec": spec, "caption": title}


_DEFAULT: Optional[MmdcModelingService] = None


def default_modeling_service() -> MmdcModelingService:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = MmdcModelingService()
    return _DEFAULT
