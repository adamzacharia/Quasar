"""Generic multiwavelength variability analysis for Quasar.

Runs on any light curve Quasar can fetch or has stored, through small adapters
to one table contract (the MMDC light-curve contract in services/mmdc_service.py:
``time_mjd, value, error, band, is_upper_limit, spectral_index,
spectral_index_err``):

* MMDC light curves (mmdc_lightcurve result_id, or fetched here by target name);
* ZTF alert light curves by ALeRCE oid (services/alerce_client.py);
* TESS / Kepler / K2 light curves by target (services/lightcurve_suite.py);
* Data Lab SMASH star light curves (datalab_star_lightcurve result_id:
  mjd, filter, cmag, cerr);
* any stored table with recognisable time / value / error / band columns.

Magnitudes are converted to relative flux, F = 10^(-0.4 m), before any
statistic. Upper limits are never used as measurements.

Methods
-------
* Fractional variability Fvar with the analytic error of Vaughan et al. 2003
  (MNRAS 345, 1271, eqs. 10 and B2) and a bootstrap (random-subset) error.
  Bands with too few points or a non-positive excess variance are skipped
  with the reason.
* Flares: Bayesian Blocks (Scargle et al. 2013, ApJ 764, 167) via
  ``astropy.stats.bayesian_blocks(fitness="measures", p0=...)``. The
  quiescent level is the median of the block fluxes; a flare is a run of
  consecutive blocks above median + k * 1.4826 * MAD(block fluxes).
* Cross-band lags: nearest-peak matching of flares within a window, and the
  discrete correlation function (Edelson & Krolik 1988, ApJ 333, 646) with
  FR/RSS bootstrap uncertainties (Peterson et al. 1998, PASP 110, 660) on the
  peak and centroid lags. A positive lag means the second band lags the first.
  Rest-frame lag = observed / (1 + z) when z is known.
* Spectral trend: photon index vs log10(flux) with a least-squares slope,
  Pearson and Spearman correlations; "harder when brighter" (index falls as
  flux rises) or "softer when brighter" is stated only when both p < 0.05.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

CONTRACT = ["time_mjd", "value", "error", "band", "is_upper_limit", "spectral_index", "spectral_index_err"]
LN10_04 = 0.4 * math.log(10.0)
BB_MAX_POINTS = 4000       # O(N^2) blocks: time-bin denser light curves first
DCF_MAX_POINTS = 800       # per band, per DCF (pairs grow as N_a * N_b)
BAND_ALIASES = {
    "gamma": ["MMDCGR"], "gamma-ray": ["MMDCGR"], "fermi": ["MMDCGR"],
    "xray": ["MMDCXRT", "MMDCXRT_ORBIT"], "x-ray": ["MMDCXRT", "MMDCXRT_ORBIT"], "xrt": ["MMDCXRT"],
    "hard-xray": ["MMDCNuX"], "nustar": ["MMDCNuX"],
    "uv": ["MMDCOUV:W1", "MMDCOUV:M2", "MMDCOUV:W2"],
    "optical": ["ZTF:R", "ZTF:G", "ASAS-SN:G", "ASAS-SN:V", "MMDCOUV:V", "MMDCOUV:B", "MMDCOUV:U"],
}


# ── adapters ────────────────────────────────────────────────────────────────
def _col(df: pd.DataFrame, *names: str) -> Optional[str]:
    lower = {str(c).lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None


def as_bool_series(values: Any, index: Any = None) -> pd.Series:
    """Upper-limit flags from any table: True/False, 1/0, "true"/"false"/"yes"/"no"; NaN and
    unknown strings are False (review CX-B14: .astype(bool) made "False" and NaN True)."""
    s = pd.Series(values, index=index)
    if s.dtype == bool:
        return s

    def one(v: Any) -> bool:
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return False
        if isinstance(v, (int, float, np.integer, np.floating)):
            return bool(v)
        return str(v).strip().lower() in ("true", "t", "1", "yes", "y", "ul", "upper", "upper_limit")

    return s.map(one).astype(bool)


def from_mags(time: Sequence[float], mag: Sequence[float], magerr: Optional[Sequence[float]], band: Sequence[str],
              upper: Optional[Sequence[bool]] = None) -> pd.DataFrame:
    m = np.asarray(mag, dtype=float)
    flux = 10.0 ** (-0.4 * m)
    err = flux * LN10_04 * np.asarray(magerr, dtype=float) if magerr is not None else np.full(m.shape, np.nan)
    return pd.DataFrame({"time_mjd": np.asarray(time, dtype=float), "value": flux, "error": err,
                         "band": list(band), "is_upper_limit": list(upper) if upper is not None else False,
                         "spectral_index": np.nan, "spectral_index_err": np.nan})


def adapt_table(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Any stored light-curve table -> the contract. Returns (table, provenance note)."""
    if {"time_mjd", "value", "band"} <= set(df.columns):
        out = df.copy()
        for c in CONTRACT:
            if c not in out.columns:
                out[c] = False if c == "is_upper_limit" else np.nan
        out["is_upper_limit"] = as_bool_series(out["is_upper_limit"], out.index)
        return out[CONTRACT].copy(), {"adapter": "light-curve contract (MMDC or equivalent)", "value_kind": "flux"}
    t = _col(df, "mjd", "time_mjd", "mjd_mid", "time", "jd", "btjd")
    if t is None:
        raise ValueError(f"no time column in the table (columns: {list(df.columns)[:20]})")
    band_col = _col(df, "band", "filter", "fid", "catalog")
    bands = df[band_col].astype(str) if band_col else pd.Series(["all"] * len(df), index=df.index)
    times = pd.to_numeric(df[t], errors="coerce")
    if t.lower() == "jd":
        times = times - 2400000.5
    ul = as_bool_series(df[_col(df, "is_upper_limit", "upper_limit")]) if _col(df, "is_upper_limit", "upper_limit") else None
    mag = _col(df, "cmag", "mag", "magpsf", "magnitude", "mag_auto", "psfmag")
    if mag is not None:
        err = _col(df, "cerr", "magerr", "sigmapsf", "mag_err", "err", "error")
        out = from_mags(times, pd.to_numeric(df[mag], errors="coerce"),
                        pd.to_numeric(df[err], errors="coerce") if err else None, bands, ul)
        return out, {"adapter": f"magnitude table ({mag}); converted to relative flux 10^(-0.4 m)", "value_kind": "mag->flux"}
    val = _col(df, "value", "flux", "nufnu", "rate", "counts", "flux_normalized")
    if val is None:
        raise ValueError(f"no flux or magnitude column in the table (columns: {list(df.columns)[:20]})")
    err = _col(df, "error", "flux_err", "err", "sigma", "value_err", "fluxerr")
    out = pd.DataFrame({"time_mjd": times, "value": pd.to_numeric(df[val], errors="coerce"),
                        "error": pd.to_numeric(df[err], errors="coerce") if err else np.nan, "band": bands.values,
                        "is_upper_limit": ul.values if ul is not None else False,
                        "spectral_index": pd.to_numeric(df[_col(df, "spectral_index")], errors="coerce")
                        if _col(df, "spectral_index") else np.nan,
                        "spectral_index_err": pd.to_numeric(df[_col(df, "spectral_index_err")], errors="coerce")
                        if _col(df, "spectral_index_err") else np.nan})
    return out, {"adapter": f"flux table ({val})", "value_kind": "flux"}


def from_alerce(lc: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """ALeRCE detections -> contract rows, using the REFERENCE-CORRECTED magnitude
    (magpsf_corr, sigmapsf_corr). ``magpsf`` is a difference-image magnitude,
    |F - F_ref| with the sign in ``isdiffpos``: for any source with flux in the
    reference image (AGN, blazars, variable stars) its variability statistics are
    wrong, so detections without a corrected magnitude are dropped and counted
    (review CX-B02). Non-detection limits are difference-image limits and stay
    upper limits only on a pure-transient (reference-free) light curve."""
    names = {1: "ZTF:g", 2: "ZTF:r", 3: "ZTF:i"}
    det_all = [r for r in lc.get("detections") or [] if r.get("mjd") is not None]
    det = [r for r in det_all if r.get("magpsf_corr") is not None]
    n_uncorrected = len(det_all) - len(det)
    a = from_mags([r["mjd"] for r in det], [r["magpsf_corr"] for r in det],
                  [r.get("sigmapsf_corr") if r.get("sigmapsf_corr") is not None else np.nan for r in det],
                  [names.get(r.get("fid"), f"ZTF:fid{r.get('fid')}") for r in det], [False] * len(det))
    info = {"n_detections": len(det_all), "n_without_corrected_magnitude": n_uncorrected,
            "magnitude": "magpsf_corr (reference-corrected source magnitude)",
            "non_detections": "not used: they are difference-image limits, not limits on the source flux"}
    return a, info


def from_arrays(time: Sequence[float], value: Sequence[float], band: str) -> pd.DataFrame:
    return pd.DataFrame({"time_mjd": np.asarray(time, dtype=float), "value": np.asarray(value, dtype=float),
                         "error": np.nan, "band": band, "is_upper_limit": False, "spectral_index": np.nan,
                         "spectral_index_err": np.nan})


def clean(df: pd.DataFrame, merge_duplicates: bool = True) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Finite rows, sorted; exact same-band same-time duplicates merged by an
    inverse-variance mean (multiple catalogue entries at one instant are not
    independent epochs, and Bayesian Blocks rejects repeated times)."""
    d = df.copy()
    d = d[np.isfinite(d["time_mjd"]) & np.isfinite(d["value"])]
    stats = {"n_rows": int(len(df)), "n_non_finite_dropped": int(len(df) - len(d)), "n_duplicates_merged": 0}
    if merge_duplicates and len(d):
        key = d["time_mjd"].round(5)
        # upper limits never merge with detections at the same instant
        d = d.assign(is_upper_limit=as_bool_series(d["is_upper_limit"], d.index))
        dup = d.assign(_k=key).duplicated(subset=["band", "is_upper_limit", "_k"], keep=False)
        if dup.any():
            keep = d[~dup]
            groups = []
            for _, g in d[dup].assign(_k=key[dup]).groupby(["band", "is_upper_limit", "_k"]):
                e = g["error"].to_numpy(dtype=float)
                w = np.where(np.isfinite(e) & (e > 0), 1.0 / np.square(e), np.nan)
                row = g.iloc[0].copy()
                if np.isfinite(w).all():
                    row["value"] = float(np.sum(w * g["value"]) / np.sum(w))
                    row["error"] = float(1.0 / math.sqrt(np.sum(w)))
                else:
                    row["value"] = float(g["value"].mean())
                    row["error"] = float(np.nanmean(e)) if np.isfinite(e).any() else np.nan
                row["is_upper_limit"] = bool(g["is_upper_limit"].all())
                groups.append(row)
                stats["n_duplicates_merged"] += len(g) - 1
            d = pd.concat([keep, pd.DataFrame(groups)[keep.columns]], ignore_index=True)
    d = d.sort_values(["band", "time_mjd"]).drop(columns=[c for c in ("_k",) if c in d.columns]).reset_index(drop=True)
    return d, stats


def resolve_bands(requested: Optional[Sequence[str]], available: Sequence[str]) -> List[str]:
    if not requested:
        return list(available)
    out: List[str] = []
    avail_l = {b.lower(): b for b in available}
    for r in requested:
        key = str(r).strip()
        low = key.lower().replace(" ", "")
        if low in avail_l:
            cand = [avail_l[low]]
        elif low in BAND_ALIASES:
            cand = [b for b in BAND_ALIASES[low] if b in available][:1 if low == "optical" else None]
        else:
            cand = [b for b in available if b.lower().split(":")[-1] == low or b.lower().split(":")[0] == low]
        out.extend(c for c in cand if c not in out)
    return out


# ── statistics ──────────────────────────────────────────────────────────────
def fvar(flux: np.ndarray, err: np.ndarray) -> Dict[str, Any]:
    """Vaughan et al. 2003: excess variance and Fvar with its analytic error."""
    x = np.asarray(flux, dtype=float)
    e = np.asarray(err, dtype=float)
    n = len(x)
    mean = float(np.mean(x))
    s2 = float(np.var(x, ddof=1)) if n > 1 else 0.0
    mse = float(np.mean(np.square(e))) if np.isfinite(e).all() else float("nan")
    out: Dict[str, Any] = {"n": n, "mean_flux": mean, "sample_variance": s2, "mean_square_error": mse}
    if not math.isfinite(mse):
        out["skipped"] = "flux errors are missing, so the measurement noise cannot be removed"
        return out
    xs = s2 - mse
    out["excess_variance"] = xs
    if mean <= 0:
        out["skipped"] = "mean flux is not positive"
        return out
    if xs <= 0:
        out["skipped"] = "excess variance <= 0: no variability detected above the measurement noise"
        return out
    fv = math.sqrt(xs / mean ** 2)
    term1 = math.sqrt(1.0 / (2 * n)) * mse / (mean ** 2 * fv)
    term2 = math.sqrt(mse / n) / mean
    out["Fvar"] = fv
    out["Fvar_err"] = math.sqrt(term1 ** 2 + term2 ** 2)
    return out


def fvar_bootstrap(flux: np.ndarray, err: np.ndarray, n_boot: int = 1000, seed: int = 0) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    x = np.asarray(flux, dtype=float)
    e = np.asarray(err, dtype=float)
    n = len(x)
    idx = rng.integers(0, n, size=(n_boot, n))
    xs = x[idx]
    es = e[idx]
    mean = xs.mean(axis=1)
    s2 = xs.var(axis=1, ddof=1)
    mse = np.mean(es ** 2, axis=1)
    exc = s2 - mse
    ok = (exc > 0) & (mean > 0)
    vals = np.sqrt(exc[ok] / mean[ok] ** 2)
    if len(vals) < 10:
        return {"n_boot": n_boot, "valid": int(len(vals))}
    lo, med, hi = np.percentile(vals, [16, 50, 84])
    return {"n_boot": n_boot, "valid": int(len(vals)), "Fvar_boot_std": float(np.std(vals, ddof=1)),
            "Fvar_boot_median": float(med), "Fvar_boot_16_84": [float(lo), float(hi)]}


def _time_bin(t: np.ndarray, x: np.ndarray, e: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[float]]:
    """Uniform time bins (inverse-variance means) so O(N^2) methods stay bounded."""
    if len(t) <= max_points or not (t.max() > t.min()):
        return t, x, e, None
    width = (t.max() - t.min()) / max_points
    edges = np.arange(t.min(), t.max() + width, width)
    k = np.clip(np.digitize(t, edges) - 1, 0, len(edges) - 1)
    tt, xx, ee = [], [], []
    for b in np.unique(k):
        m = k == b
        w = 1.0 / np.square(e[m])
        tt.append(float(np.sum(w * t[m]) / np.sum(w)))
        xx.append(float(np.sum(w * x[m]) / np.sum(w)))
        ee.append(float(1.0 / math.sqrt(np.sum(w))))
    return np.asarray(tt), np.asarray(xx), np.asarray(ee), float(width)


def _fill_errors(x: np.ndarray, e: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
    if np.isfinite(e).all() and (e > 0).all():
        return e, None
    good = np.isfinite(e) & (e > 0)
    if good.any():
        e2 = np.where(good, e, np.median(e[good]))
        return e2, f"{int((~good).sum())} missing errors set to the band's median error"
    # point-to-point scatter as a noise estimate (flat-fielded space photometry)
    scatter = float(np.median(np.abs(np.diff(x))) / (0.6745 * math.sqrt(2))) if len(x) > 2 else float(np.std(x))
    return np.full_like(x, max(scatter, 1e-12 * max(1.0, abs(float(np.mean(x)))))), (
        "no flux errors: noise estimated from the point-to-point scatter")


def bayesian_block_flares(t: np.ndarray, x: np.ndarray, e: np.ndarray, *, p0: float = 0.05, k: float = 3.0) -> Dict[str, Any]:
    from astropy.stats import bayesian_blocks

    notes = []
    e, note = _fill_errors(x, e)
    if note:
        notes.append(note)
    tb, xb, eb, width = _time_bin(t, x, e, BB_MAX_POINTS)
    if width is not None:
        notes.append(f"{len(t)} points time-binned to {len(tb)} ({width:.3g} d bins) before Bayesian Blocks")
    order = np.argsort(tb, kind="stable")
    tb, xb, eb = tb[order], xb[order], eb[order]
    uniq, inv = np.unique(tb, return_inverse=True)
    if len(uniq) < len(tb):
        # astropy rejects repeated times: merge them by inverse-variance mean for the
        # segmentation only (review CX-B09: merge_duplicate_epochs=false crashed here)
        w = 1.0 / np.square(eb)
        sw = np.bincount(inv, weights=w)
        xb = np.bincount(inv, weights=w * xb) / sw
        eb = 1.0 / np.sqrt(sw)
        notes.append(f"{len(tb) - len(uniq)} repeated times merged for the Bayesian-block segmentation")
        tb = uniq
    edges = bayesian_blocks(tb, xb, eb, fitness="measures", p0=p0)
    blocks = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (tb >= lo) & (tb <= hi) if hi == edges[-1] else (tb >= lo) & (tb < hi)
        if not m.any():
            continue
        w = 1.0 / np.square(eb[m])
        blocks.append({"start": float(lo), "end": float(hi), "n": int(m.sum()),
                       "flux": float(np.sum(w * xb[m]) / np.sum(w)), "flux_err": float(1.0 / math.sqrt(np.sum(w)))})
    out: Dict[str, Any] = {"p0": p0, "k": k, "n_blocks": len(blocks), "blocks": blocks, "notes": notes, "flares": []}
    if len(blocks) < 3:
        out["flare_note"] = (f"only {len(blocks)} Bayesian block(s): no significant change points, so no flares "
                             "(quiescent level undefined)" if len(blocks) < 2 else
                             "2 Bayesian blocks: a single change point; too few blocks for a MAD threshold")
        return out
    bf = np.array([b["flux"] for b in blocks])
    med = float(np.median(bf))
    mad = float(np.median(np.abs(bf - med))) * 1.4826
    if mad <= 0:
        out["flare_note"] = "block fluxes have zero MAD; threshold undefined"
        return out
    thr = med + k * mad
    out.update({"quiescent_flux": med, "robust_sigma": mad, "threshold": thr})
    flares = []
    cur: List[int] = []
    for i, b in enumerate(blocks + [None]):
        if b is not None and b["flux"] > thr:
            cur.append(i)
            continue
        if cur:
            fb = [blocks[j] for j in cur]
            peak_block = max(fb, key=lambda z: z["flux"])
            m = (t >= fb[0]["start"]) & (t <= fb[-1]["end"])
            ip = int(np.argmax(np.where(m, x, -np.inf))) if m.any() else None
            flares.append({
                "start_mjd": round(fb[0]["start"], 4), "end_mjd": round(fb[-1]["end"], 4),
                "duration_d": round(fb[-1]["end"] - fb[0]["start"], 4),
                "peak_mjd": round(float(t[ip]), 4) if ip is not None else round((peak_block["start"] + peak_block["end"]) / 2, 4),
                "peak_block_mid_mjd": round((peak_block["start"] + peak_block["end"]) / 2, 4),
                "peak_block_flux": peak_block["flux"],
                "peak_point_flux": float(x[ip]) if ip is not None else None,
                "blocks": len(fb),
                "significance_sigma": round((peak_block["flux"] - med) / mad, 2),
            })
            cur = []
    out["flares"] = flares
    return out


def _dcf_core(ta, xa, ea, tb, xb, eb, bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    da = xa - xa.mean()
    db = xb - xb.mean()
    va = np.var(xa, ddof=1) - np.mean(ea ** 2)
    vb = np.var(xb, ddof=1) - np.mean(eb ** 2)
    if va <= 0:
        va = np.var(xa, ddof=1)
    if vb <= 0:
        vb = np.var(xb, ddof=1)
    udcf = np.outer(da, db) / math.sqrt(va * vb)
    dt = tb[None, :] - ta[:, None]
    k = np.digitize(dt.ravel(), bins) - 1
    u = udcf.ravel()
    nb = len(bins) - 1
    ok = (k >= 0) & (k < nb)
    k, u = k[ok], u[ok]
    cnt = np.bincount(k, minlength=nb).astype(float)
    s = np.bincount(k, weights=u, minlength=nb)
    mean = np.divide(s, cnt, out=np.full(nb, np.nan), where=cnt > 0)
    s2 = np.bincount(k, weights=(u - mean[k]) ** 2, minlength=nb)
    err = np.divide(np.sqrt(s2), cnt - 1, out=np.full(nb, np.nan), where=cnt > 1)
    return mean, err, cnt


def _peak_and_centroid(centers: np.ndarray, dcf: np.ndarray, frac: float = 0.8) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not np.isfinite(dcf).any():
        return None, None, None
    i = int(np.nanargmax(dcf))
    peak = float(dcf[i])
    if peak <= 0:
        return float(centers[i]), None, peak
    # centroid over the contiguous run around the peak with DCF >= frac * peak
    lo = i
    while lo - 1 >= 0 and np.isfinite(dcf[lo - 1]) and dcf[lo - 1] >= frac * peak:
        lo -= 1
    hi = i
    while hi + 1 < len(dcf) and np.isfinite(dcf[hi + 1]) and dcf[hi + 1] >= frac * peak:
        hi += 1
    w = dcf[lo:hi + 1]
    cen = float(np.sum(w * centers[lo:hi + 1]) / np.sum(w))
    return float(centers[i]), cen, peak


def dcf_lag(ta, xa, ea, tb, xb, eb, *, max_lag: float, bin_width: float, n_boot: int = 200, seed: int = 0) -> Dict[str, Any]:
    """Edelson & Krolik DCF with FR/RSS bootstrap errors on peak and centroid lags."""
    rng = np.random.default_rng(seed)
    notes = []
    ea, n1 = _fill_errors(xa, ea)
    eb, n2 = _fill_errors(xb, eb)
    notes += [n for n in (n1, n2) if n]
    ta, xa, ea, w1 = _time_bin(ta, xa, ea, DCF_MAX_POINTS)
    tb, xb, eb, w2 = _time_bin(tb, xb, eb, DCF_MAX_POINTS)
    for w in (w1, w2):
        if w is not None:
            notes.append(f"a band was time-binned ({w:.3g} d) to <= {DCF_MAX_POINTS} points for the DCF")
    # Bins are CENTRED on zero lag (review CX-B01: edges from -max_lag put 0 on an
    # edge, so a zero-lag correlation read as a peak lag of +bin_width/2).
    half = int(np.ceil(max_lag / bin_width - 0.5))
    bins = (np.arange(-half, half + 2) - 0.5) * bin_width
    centers = (bins[:-1] + bins[1:]) / 2
    dcf, err, cnt = _dcf_core(ta, xa, ea, tb, xb, eb, bins)
    dcf = np.where(cnt >= 3, dcf, np.nan)
    peak_lag, cen_lag, peak = _peak_and_centroid(centers, dcf)
    peaks, cens = [], []
    for _ in range(int(n_boot)):
        ia = np.unique(rng.integers(0, len(ta), len(ta)))
        ib = np.unique(rng.integers(0, len(tb), len(tb)))
        if len(ia) < 5 or len(ib) < 5:
            continue
        xa_r = xa[ia] + rng.normal(0, ea[ia])
        xb_r = xb[ib] + rng.normal(0, eb[ib])
        d, _, c = _dcf_core(ta[ia], xa_r, ea[ia], tb[ib], xb_r, eb[ib], bins)
        d = np.where(c >= 3, d, np.nan)
        p, ce, _ = _peak_and_centroid(centers, d)
        if p is not None:
            peaks.append(p)
        if ce is not None:
            cens.append(ce)

    def _ci(v):
        if len(v) < 20:
            return None
        lo, med, hi = np.percentile(v, [16, 50, 84])
        return {"median": float(med), "minus": float(med - lo), "plus": float(hi - med), "n": len(v)}

    return {"lag_bins_d": centers.tolist(), "dcf": [None if not np.isfinite(v) else float(v) for v in dcf],
            "dcf_err": [None if not np.isfinite(v) else float(v) for v in err], "pairs_per_bin": cnt.astype(int).tolist(),
            "peak_dcf": peak, "peak_lag_d": peak_lag, "centroid_lag_d": cen_lag,
            "peak_lag_bootstrap": _ci(peaks), "centroid_lag_bootstrap": _ci(cens),
            "bin_width_d": bin_width, "max_lag_d": max_lag, "notes": notes}


def match_flare_lags(flares_a: List[Dict[str, Any]], flares_b: List[Dict[str, Any]], window_d: float) -> List[Dict[str, Any]]:
    out = []
    for fa in flares_a:
        best = None
        for fb in flares_b:
            d = fb["peak_mjd"] - fa["peak_mjd"]
            if abs(d) <= window_d and (best is None or abs(d) < abs(best[1])):
                best = (fb, d)
        if best:
            out.append({"peak_mjd_a": fa["peak_mjd"], "peak_mjd_b": best[0]["peak_mjd"], "lag_d": round(best[1], 4)})
    return out


def index_flux_trend(flux: np.ndarray, index: np.ndarray, *, alpha: float = 0.05) -> Dict[str, Any]:
    from scipy import stats

    ok = np.isfinite(flux) & np.isfinite(index) & (flux > 0)
    f, g = np.log10(flux[ok]), index[ok]
    out: Dict[str, Any] = {"n": int(ok.sum())}
    if out["n"] < 5:
        out["skipped"] = "fewer than 5 points with both flux and spectral index"
        return out
    if np.ptp(f) == 0 or np.ptp(g) == 0:
        out["skipped"] = "flux or index is constant"
        return out
    lr = stats.linregress(f, g)
    pr = stats.pearsonr(f, g)
    sr = stats.spearmanr(f, g)
    out.update({
        "slope": float(lr.slope), "slope_err": float(lr.stderr), "intercept": float(lr.intercept),
        "intercept_err": float(lr.intercept_stderr), "pearson_r": float(pr.statistic), "pearson_p": float(pr.pvalue),
        "spearman_rho": float(sr.statistic), "spearman_p": float(sr.pvalue),
        "label_rule": f"a label is given only when both Pearson and Spearman p < {alpha}",
    })
    if pr.pvalue < alpha and sr.pvalue < alpha and np.sign(pr.statistic) == np.sign(sr.statistic):
        out["trend"] = "harder when brighter" if pr.statistic < 0 else "softer when brighter"
    else:
        out["trend"] = "no significant trend"
    return out


# ── orchestration ───────────────────────────────────────────────────────────
def analyze(df: pd.DataFrame, *, bands: Optional[Sequence[str]] = None, min_points: int = 5, p0: float = 0.05,
            k: float = 3.0, lag_bands: Optional[Sequence[str]] = None, lag_window_d: float = 30.0,
            max_lag_d: Optional[float] = None, dcf_bin_d: Optional[float] = None, z: Optional[float] = None,
            n_boot: int = 1000, n_boot_dcf: int = 200, merge_duplicates: bool = True, seed: int = 0) -> Dict[str, Any]:
    """Run every analysis on a contract table. Pure: no network, no plotting."""
    data, cstats = clean(df, merge_duplicates)
    available = list(dict.fromkeys(data["band"]))
    chosen = resolve_bands(bands, available)
    if bands and not chosen:
        raise ValueError(f"none of the requested bands {list(bands)} are present; available: {available}")
    per_band: List[Dict[str, Any]] = []
    flares_by_band: Dict[str, Dict[str, Any]] = {}
    for b in chosen:
        g = data[data["band"] == b]
        det = g[~g["is_upper_limit"].astype(bool)]
        row: Dict[str, Any] = {"band": b, "n_rows": int(len(g)), "n_upper_limits_excluded": int(len(g) - len(det)),
                               "n_used": int(len(det))}
        if len(det):
            row["MJD_range"] = [round(float(det["time_mjd"].min()), 3), round(float(det["time_mjd"].max()), 3)]
        if len(det) < max(2, int(min_points)):
            row["skipped"] = f"only {len(det)} detections (need >= {max(2, int(min_points))})"
            per_band.append(row)
            continue
        t = det["time_mjd"].to_numpy(dtype=float)
        x = det["value"].to_numpy(dtype=float)
        e = det["error"].to_numpy(dtype=float)
        fv = fvar(x, e)
        row.update({"mean_flux": fv["mean_flux"], "excess_variance": fv.get("excess_variance")})
        if "Fvar" in fv:
            row["Fvar"] = fv["Fvar"]
            row["Fvar_err"] = fv["Fvar_err"]
            bs = fvar_bootstrap(x, e, n_boot=n_boot, seed=seed)
            row["Fvar_err_bootstrap"] = bs.get("Fvar_boot_std")
            row["Fvar_bootstrap_16_84"] = bs.get("Fvar_boot_16_84")
        else:
            row["Fvar_skipped"] = fv.get("skipped")
        bb = bayesian_block_flares(t, x, e, p0=p0, k=k)
        row["n_blocks"] = bb["n_blocks"]
        row["n_flares"] = len(bb["flares"])
        if bb.get("flare_note"):
            row["flare_note"] = bb["flare_note"]
        if bb.get("notes"):
            row["notes"] = bb["notes"]
        flares_by_band[b] = bb
        idx = det["spectral_index"].to_numpy(dtype=float)
        if np.isfinite(idx).sum() >= 5:
            row["index_flux"] = index_flux_trend(x, idx)
        per_band.append(row)
    result: Dict[str, Any] = {"cleaning": cstats, "bands_available": available, "bands_analysed": chosen,
                              "per_band": per_band,
                              "flares": {b: {k2: v for k2, v in bb.items() if k2 != "blocks"} for b, bb in flares_by_band.items()},
                              "_blocks": flares_by_band}
    # lags
    pair = _lag_pair(lag_bands, flares_by_band, available, data)
    if pair:
        a, b = pair
        ga = data[(data["band"] == a) & ~data["is_upper_limit"].astype(bool)]
        gb = data[(data["band"] == b) & ~data["is_upper_limit"].astype(bool)]
        span = float(max(ga["time_mjd"].max(), gb["time_mjd"].max()) - min(ga["time_mjd"].min(), gb["time_mjd"].min()))
        if max_lag_d is not None and not (0 < float(max_lag_d) <= 5000):
            raise ValueError("max_lag_days must lie in (0, 5000]")
        if dcf_bin_d is not None and not (0 < float(dcf_bin_d)):
            raise ValueError("dcf_bin_days must be positive")
        ml = float(max_lag_d) if max_lag_d else min(100.0, max(5.0, span / 4))
        cad = np.median(np.diff(np.sort(np.concatenate([ga["time_mjd"].to_numpy(), gb["time_mjd"].to_numpy()]))))
        bw = float(dcf_bin_d) if dcf_bin_d else float(max(1.0, min(ml / 10, 5 * cad)))
        if 2 * ml / bw > 2000:
            raise ValueError(f"max_lag_days / dcf_bin_days gives {int(2 * ml / bw)} DCF bins (limit 2000); widen the bins")
        lag: Dict[str, Any] = {"band_a": a, "band_b": b, "sign_convention": f"positive lag = {b} lags {a}"}
        if len(ga) >= 5 and len(gb) >= 5:
            lag["dcf"] = dcf_lag(ga["time_mjd"].to_numpy(float), ga["value"].to_numpy(float), ga["error"].to_numpy(float),
                                 gb["time_mjd"].to_numpy(float), gb["value"].to_numpy(float), gb["error"].to_numpy(float),
                                 max_lag=ml, bin_width=bw, n_boot=n_boot_dcf, seed=seed)
        else:
            lag["dcf_skipped"] = "each band needs >= 5 detections"
        fa = (flares_by_band.get(a) or {}).get("flares") or []
        fb = (flares_by_band.get(b) or {}).get("flares") or []
        lag["flare_matches"] = match_flare_lags(fa, fb, lag_window_d)
        lag["flare_match_window_d"] = lag_window_d
        if lag["flare_matches"]:
            lag["flare_match_median_lag_d"] = float(np.median([m["lag_d"] for m in lag["flare_matches"]]))
        if z is not None and z > -1:
            lag["z"] = z
            d = lag.get("dcf") or {}
            for key in ("peak_lag_d", "centroid_lag_d"):
                if d.get(key) is not None:
                    lag[f"rest_frame_{key}"] = d[key] / (1 + z)
            if lag.get("flare_match_median_lag_d") is not None:
                lag["rest_frame_flare_match_median_lag_d"] = lag["flare_match_median_lag_d"] / (1 + z)
        result["lag"] = lag
    return result


def _lag_pair(lag_bands, flares_by_band, available, data) -> Optional[Tuple[str, str]]:
    if lag_bands:
        resolved = [resolve_bands([lb], available) for lb in lag_bands]
        firsts = [r[0] for r in resolved if r]
        if len(firsts) >= 2 and firsts[0] != firsts[1]:
            return firsts[0], firsts[1]
        raise ValueError(f"lag_bands {list(lag_bands)} must name two different available bands; available: {available}")
    if "MMDCXRT" in available and "MMDCGR" in available:
        return "MMDCXRT", "MMDCGR"
    return None


# ── plots ───────────────────────────────────────────────────────────────────
_COLORS = ["#4e79a7", "#e15759", "#59a14f", "#f28e2b", "#b07aa1", "#76b7b2"]


def blocks_plotly_spec(data: pd.DataFrame, analysis: Dict[str, Any], flares_full: Dict[str, Dict[str, Any]],
                       title: str, max_panels: int = 4, cap: int = 5000) -> Dict[str, Any]:
    from services.mmdc_service import downsample_trace

    bands = [b for b in analysis["bands_analysed"] if b in flares_full][:max_panels]
    n = max(1, len(bands))
    gap = 0.05
    h = (1.0 - gap * (n - 1)) / n
    layout: Dict[str, Any] = {"title": {"text": title}, "height": max(360, 220 * n + 80), "hovermode": "closest",
                              "showlegend": True, "legend": {"orientation": "h", "y": -0.12}, "shapes": []}
    traces = []
    for i, b in enumerate(bands):
        top = 1.0 - i * (h + gap)
        yname = "yaxis" if i == 0 else f"yaxis{i + 1}"
        yref = "y" if i == 0 else f"y{i + 1}"
        layout[yname] = {"domain": [max(0.0, top - h), top], "title": {"text": b}, "anchor": "x", "exponentformat": "power"}
        g = data[(data["band"] == b)]
        g = downsample_trace(g, max(100, cap // n))
        det, ul = g[~g["is_upper_limit"].astype(bool)], g[g["is_upper_limit"].astype(bool)]
        c = _COLORS[i % len(_COLORS)]
        traces.append({"type": "scatter", "mode": "markers", "name": b, "x": det["time_mjd"].tolist(), "y": det["value"].tolist(),
                       "yaxis": yref, "marker": {"size": 4, "color": c, "opacity": 0.7},
                       "error_y": {"type": "data", "array": det["error"].fillna(0).tolist(), "visible": True, "thickness": 0.6, "width": 0},
                       "hovertemplate": b + "<br>MJD %{x:.3f}<br>flux %{y:.3e}<extra></extra>"})
        if len(ul):
            traces.append({"type": "scatter", "mode": "markers", "name": f"{b} limits", "x": ul["time_mjd"].tolist(),
                           "y": ul["value"].tolist(), "yaxis": yref,
                           "error_y": {"type": "data", "symmetric": False, "array": [0.0] * len(ul),
                                       "arrayminus": [abs(float(v)) * 0.3 for v in ul["value"]], "visible": True,
                                       "thickness": 1.0, "width": 0, "color": c},
                           "marker": {"symbol": "triangle-down", "size": 6, "color": c, "opacity": 0.6}})
        bb = flares_full[b]
        xs, ys = [], []
        for blk in bb["blocks"]:
            xs += [blk["start"], blk["end"], None]
            ys += [blk["flux"], blk["flux"], None]
        traces.append({"type": "scatter", "mode": "lines", "name": f"{b} Bayesian blocks", "x": xs, "y": ys, "yaxis": yref,
                       "line": {"color": "#222", "width": 1.6}, "hoverinfo": "skip"})
        if bb.get("threshold") is not None:
            layout["shapes"].append({"type": "line", "xref": "paper", "x0": 0, "x1": 1, "yref": yref,
                                     "y0": bb["threshold"], "y1": bb["threshold"],
                                     "line": {"dash": "dot", "width": 1, "color": "#d62728"}})
        for f in bb["flares"]:
            layout["shapes"].append({"type": "rect", "xref": "x", "x0": f["start_mjd"], "x1": f["end_mjd"],
                                     "yref": f"{yref} domain", "y0": 0, "y1": 1, "fillcolor": "rgba(214,39,40,0.15)",
                                     "line": {"width": 0}, "layer": "below"})
    layout["xaxis"] = {"title": {"text": "MJD"}, "anchor": ("y" if n == 1 else f"y{n}")}
    return {"data": traces, "layout": layout}


def dcf_plotly_spec(lag: Dict[str, Any], title: str) -> Dict[str, Any]:
    d = lag["dcf"]
    x = [c for c, v in zip(d["lag_bins_d"], d["dcf"]) if v is not None]
    y = [v for v in d["dcf"] if v is not None]
    e = [(er or 0) for er, v in zip(d["dcf_err"], d["dcf"]) if v is not None]
    shapes = []
    for key, dash in (("peak_lag_d", "solid"), ("centroid_lag_d", "dash")):
        if d.get(key) is not None:
            shapes.append({"type": "line", "x0": d[key], "x1": d[key], "yref": "paper", "y0": 0, "y1": 1,
                           "line": {"color": "#d62728", "dash": dash, "width": 1.2}})
    return {"data": [{"type": "scatter", "mode": "markers+lines", "name": "DCF", "x": x, "y": y,
                      "error_y": {"type": "data", "array": e, "visible": True}, "marker": {"size": 6}}],
            "layout": {"title": {"text": title}, "height": 380, "shapes": shapes,
                       "xaxis": {"title": {"text": f"lag [d] (positive = {lag['band_b']} lags {lag['band_a']})"}, "zeroline": True},
                       "yaxis": {"title": {"text": "DCF"}}}}


def index_plotly_spec(data: pd.DataFrame, per_band: List[Dict[str, Any]], title: str) -> Optional[Dict[str, Any]]:
    traces = []
    for i, row in enumerate(r for r in per_band if r.get("index_flux") and "slope" in r["index_flux"]):
        b = row["band"]
        g = data[(data["band"] == b) & ~data["is_upper_limit"].astype(bool)]
        g = g[np.isfinite(g["spectral_index"]) & (g["value"] > 0)]
        c = _COLORS[i % len(_COLORS)]
        lf = np.log10(g["value"].to_numpy(float))
        tr = row["index_flux"]
        traces.append({"type": "scatter", "mode": "markers", "name": f"{b} ({tr['trend']})", "x": lf.tolist(),
                       "y": g["spectral_index"].tolist(), "marker": {"size": 5, "color": c, "opacity": 0.7},
                       "error_y": {"type": "data", "array": g["spectral_index_err"].fillna(0).tolist(), "visible": True,
                                   "thickness": 0.6, "width": 0}})
        xx = [float(lf.min()), float(lf.max())]
        traces.append({"type": "scatter", "mode": "lines", "name": f"{b} fit: slope {tr['slope']:.3f} ± {tr['slope_err']:.3f}",
                       "x": xx, "y": [tr["intercept"] + tr["slope"] * v for v in xx], "line": {"color": c, "width": 2}})
    if not traces:
        return None
    return {"data": traces, "layout": {"title": {"text": title}, "height": 400,
                                       "xaxis": {"title": {"text": "log10 flux"}},
                                       "yaxis": {"title": {"text": "photon index (higher = softer)"}}}}


# ── service: fetch / adapt, analyse, figures ─────────────────────────────────
METHOD_REFERENCES = {
    "Fvar": ("Vaughan et al. 2003, MNRAS 345, 1271. The analytic Fvar_err covers the measurement noise only; "
             "Fvar_err_bootstrap (random resampling of the points) also reflects the sampling of the intrinsic "
             "variability and is usually larger. Quote both."),
    "Bayesian Blocks": "Scargle et al. 2013, ApJ 764, 167 (astropy.stats.bayesian_blocks, fitness='measures')",
    "DCF": "Edelson & Krolik 1988, ApJ 333, 646; lag errors by FR/RSS bootstrap (Peterson et al. 1998, PASP 110, 660)",
}


def _r(v: Any, sig: int = 4, key: str = "") -> Any:
    """Round for the model-facing output: times (MJD, days) to 3 decimals,
    everything else to ``sig`` significant figures; recurses into containers."""
    if isinstance(v, dict):
        return {k: _r(x, sig, str(k)) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_r(x, sig, key) for x in v]
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (float, np.floating)) and math.isfinite(float(v)):
        k = key.lower()
        if "mjd" in k or k.endswith("_d") or k in ("median", "minus", "plus", "duration_d"):
            return round(float(v), 3)
        return float(f"{float(v):.{sig}g}")
    return v


class VariabilityService:
    def __init__(self, mmdc: Any = None, *, alerce: Any = None, lightcurves: Any = None, plotting_service: Any = None):
        self._mmdc = mmdc
        self._alerce = alerce
        self._lightcurves = lightcurves
        self._plotting = plotting_service

    @property
    def mmdc(self):
        if self._mmdc is None:
            from services.mmdc_service import default_mmdc_service

            self._mmdc = default_mmdc_service()
        return self._mmdc

    @property
    def plotting(self):
        if self._plotting is None:
            self._plotting = self.mmdc.plotting_service
        return self._plotting

    def load(self, *, result_id: Optional[str], source: Optional[str], target_name: Optional[str], ra: Optional[float],
             dec: Optional[float], identifier: Optional[str], index: int, start_date: Any, end_date: Any,
             owner_id: Optional[str], user_id: Optional[str], resolver: Any) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        from services import mmdc_service as ms

        if result_id:
            frame, meta = ms.load_result(result_id, owner_id)
            table, prov = adapt_table(frame)
            prov.update({"input": f"stored result {result_id}", "tool_name": meta.get("tool_name"),
                         "is_mmdc": str(meta.get("source") or "").upper() == "MMDC", "mmdc": meta.get("mmdc")})
            return table, prov
        src = str(source or ("mmdc" if (target_name or ra is not None) and not identifier else "")).strip().lower()
        if src == "mmdc":
            res = self.mmdc.lightcurve(target_name=target_name, ra=ra, dec=dec, start_date=start_date, end_date=end_date,
                                       resolver=resolver, user_id=user_id, make_plot=False)
            if not res.get("success"):
                raise LookupError(res.get("error") or "MMDC light curve unavailable")
            if "_table" not in res:
                raise LookupError(res.get("note") or "no MMDC light-curve rows in this cone and window")
            table, _ = res.pop("_table")
            return table, {"input": "MMDC light curves (same data as mmdc_lightcurve)", "is_mmdc": True,
                           "adapter": "MMDC light-curve contract", "value_kind": "flux",
                           "mmdc": {"target": res.get("target"), "window": res.get("window"),
                                    "resolved_position": res.get("resolved_position")},
                           "units_note": res.get("units_note")}
        if src == "ztf":
            if not identifier:
                raise ValueError("source='ztf' needs identifier (the ALeRCE oid, e.g. ZTF18abcdefg)")
            if self._alerce is None:
                from services.alerce_client import AlerceClient

                self._alerce = AlerceClient()
            lc = self._alerce.light_curve(identifier)
            if not lc.get("success"):
                raise LookupError(lc.get("error") or f"ALeRCE light curve for {identifier} unavailable")
            table, zinfo = from_alerce(lc)
            if table.empty:
                raise LookupError(
                    f"ALeRCE returned no reference-corrected magnitudes (magpsf_corr) for {identifier}: its magpsf values "
                    "are difference-image magnitudes, which do not measure the source's variability. Fvar and flares "
                    "cannot be computed from them.")
            return table, {"input": f"ALeRCE ZTF light curve {identifier}", "is_mmdc": False,
                           "adapter": "ZTF magpsf_corr (reference-corrected) converted to relative flux", "ztf": zinfo,
                           "value_kind": "mag->flux"}
        if src in ("tess", "kepler", "k2"):
            if not (identifier or target_name):
                raise ValueError(f"source='{src}' needs identifier or target_name")
            if self._lightcurves is None:
                from services.lightcurve_suite import LightcurveSuite

                self._lightcurves = LightcurveSuite()
            mission = "K2" if src == "k2" else src.upper()
            t, v, _, meta = self._lightcurves.fetch_lightcurve_arrays(identifier or target_name, mission=mission, index=index)
            band = f"{mission}:{(meta or {}).get('sector') or (meta or {}).get('quarter') or index}"
            return from_arrays(t, v, band), {
                "input": f"{mission} light curve {identifier or target_name} (index {index})", "is_mmdc": False,
                "value_kind": "normalized flux", "adapter": "lightkurve normalized flux (no per-point errors)",
                "time_note": "times are mission days (BTJD/BKJD), not MJD; flare times and lags use that scale"}
        raise ValueError("give result_id (from mmdc_lightcurve or datalab_star_lightcurve), or source='mmdc' with "
                         "target_name/ra/dec, 'ztf' with identifier (oid), or 'tess'/'kepler'/'k2' with identifier")

    def resolve_z(self, z: Any, prov: Dict[str, Any], user_id: Optional[str]) -> Tuple[Optional[float], str]:
        if z is not None:
            v = float(z)
            if not (math.isfinite(v) and v >= 0):
                raise ValueError("z must be a non-negative number")
            return v, "given by the caller"
        mm = prov.get("mmdc") or {}
        pos = mm.get("resolved_position") or {}
        if prov.get("is_mmdc") and pos.get("ra") is not None:
            try:
                job = self.mmdc.prepare(pos["ra"], pos["dec"], mm.get("target") or "source", user_id=user_id,
                                        poll_seconds=15)
                if not job.get("pending") and job.get("status") == "done":
                    info = self.mmdc.source_info(job["uuid"], user_id=user_id)
                    if info.get("redshift") is not None:
                        return float(info["redshift"]), "MMDC source info (get_info)"
            except Exception:  # noqa: BLE001 - z is optional; reported as unknown
                pass
        return None, "unknown (no z given and none from MMDC): lags are observed-frame only"

    def run(self, **kw: Any) -> Dict[str, Any]:
        from services import mmdc_service as ms

        owner_id, user_id, resolver = kw.pop("owner_id", None), kw.pop("user_id", None), kw.pop("resolver", None)
        merge = kw.get("merge_duplicate_epochs", True) is not False
        table, prov = self.load(result_id=kw.get("result_id"), source=kw.get("source"), target_name=kw.get("target_name"),
                                ra=kw.get("ra"), dec=kw.get("dec"), identifier=kw.get("identifier"),
                                index=int(kw.get("index") or 0), start_date=kw.get("start_date"),
                                end_date=kw.get("end_date"), owner_id=owner_id, user_id=user_id, resolver=resolver)
        window: Dict[str, Any] = {"applied": False}
        if kw.get("start_date") or kw.get("end_date"):
            # Every source, not only result_id (review CX-B03: ztf/tess silently ignored the
            # window). An expanded year/month/day end is exclusive, as in mmdc_service (CX-02/B05).
            s, _ = ms.parse_window_bound(kw.get("start_date"), is_end=False)
            e, e_note = ms.parse_window_bound(kw.get("end_date"), is_end=True)
            if prov.get("value_kind") == "normalized flux":
                raise ValueError("start_date/end_date are MJD-based; TESS/Kepler times are mission days (BTJD/BKJD), "
                                 "so a date window cannot be applied to this source")
            n0 = len(table)
            if s is not None:
                table = table[table["time_mjd"] >= s]
            if e is not None:
                table = table[(table["time_mjd"] < e) if "expanded" in e_note else (table["time_mjd"] <= e)]
            window = {"applied": True, "start_mjd": s, "end_mjd": e,
                      "start_date": ms.mjd_to_date(s) if s is not None else None,
                      "end_date": ms.mjd_to_date(e - 1e-6) if (e is not None and "expanded" in e_note) else
                      (ms.mjd_to_date(e) if e is not None else None),
                      "end_exclusive": bool(e is not None and "expanded" in e_note),
                      "rule": "a measurement is kept when its time lies inside the window",
                      "rows_before": int(n0), "rows_kept": int(len(table))}
        if table.empty:
            raise LookupError("the light curve has no rows in this window")
        will_lag = bool(kw.get("lag_bands")) or {"MMDCXRT", "MMDCGR"} <= set(table["band"])
        if will_lag:
            z, zsrc = self.resolve_z(kw.get("z"), prov, user_id)
        else:
            z, zsrc = (float(kw["z"]) if kw.get("z") is not None else None), "not needed (no lag computed)"
        res = analyze(table, bands=kw.get("bands"), min_points=int(kw.get("min_points") or 5),
                      p0=float(kw.get("p0") or 0.05), k=float(kw.get("flare_k") or 3.0), lag_bands=kw.get("lag_bands"),
                      lag_window_d=float(kw.get("lag_window_days") or 30.0), max_lag_d=kw.get("max_lag_days"),
                      dcf_bin_d=kw.get("dcf_bin_days"), z=z, merge_duplicates=merge)
        clean_tbl, _ = clean(table, merge)
        label = (prov.get("mmdc") or {}).get("target") or kw.get("target_name") or kw.get("identifier") or kw.get("result_id")
        per_band, rows = [], []
        for r in res["per_band"]:
            c = {k: _r(v, key=k) for k, v in r.items() if k != "index_flux"}
            if r.get("index_flux"):
                c["index_flux"] = _r(r["index_flux"])
            per_band.append(c)
            rows.append({"band": r["band"], "n_used": r.get("n_used"), "n_upper_limits_excluded": r.get("n_upper_limits_excluded"),
                         "MJD_start": (r.get("MJD_range") or [None, None])[0], "MJD_end": (r.get("MJD_range") or [None, None])[1],
                         "mean_flux": r.get("mean_flux"), "Fvar": r.get("Fvar"), "Fvar_err": r.get("Fvar_err"),
                         "Fvar_err_bootstrap": r.get("Fvar_err_bootstrap"), "n_blocks": r.get("n_blocks"),
                         "n_flares": r.get("n_flares"), "index_flux_trend": (r.get("index_flux") or {}).get("trend"),
                         "index_flux_slope": (r.get("index_flux") or {}).get("slope"),
                         "skipped": r.get("skipped") or r.get("Fvar_skipped")})
        mjd_scale = prov.get("value_kind") != "normalized flux"
        flares: Dict[str, Any] = {}
        for b, bb in res["flares"].items():
            fl = sorted(bb.get("flares") or [], key=lambda f: -f["significance_sigma"])
            entry: Dict[str, Any] = {"n_flares": len(fl), "quiescent_flux": _r(bb.get("quiescent_flux")),
                                     "threshold": _r(bb.get("threshold")), "p0": bb.get("p0"), "k": bb.get("k"),
                                     "n_blocks": bb.get("n_blocks"),
                                     "flares": [{**_r(f),
                                                 **({"peak_date": ms.mjd_to_date(f["peak_mjd"]),
                                                     "start_date": ms.mjd_to_date(f["start_mjd"]),
                                                     "end_date": ms.mjd_to_date(f["end_mjd"])} if mjd_scale else {})}
                                                for f in fl[:10]]}
            if bb.get("flare_note"):
                entry["note"] = bb["flare_note"]
            if len(fl) > 10:
                entry["more_flares_not_listed"] = len(fl) - 10
            flares[b] = entry
        out: Dict[str, Any] = {
            "success": True, "target": label, "input": prov.get("input"), "adapter": prov.get("adapter"),
            "cleaning": res["cleaning"], "bands_available": res["bands_available"], "bands_analysed": res["bands_analysed"],
            "per_band": per_band, "flares": flares, "z": z, "z_source": zsrc, "methods": METHOD_REFERENCES,
            "window": window if window.get("applied") else ((prov.get("mmdc") or {}).get("window") or {"applied": False}),
            "answer_requirements": [
                "Quote Fvar with its analytic and bootstrap errors per band; list skipped bands with the reason.",
                "Flares: start, peak and end (MJD and date), peak flux and significance, with the p0 and k used.",
                "Lags: DCF peak and centroid with their bootstrap errors and the sign convention; rest-frame only if z is known.",
                "Index-flux trends: slope, Pearson and Spearman with p-values; a harder/softer label only as the tool states it.",
            ],
        }
        for key in ("units_note", "time_note", "ztf"):
            if prov.get(key):
                out[key] = prov[key]
        if "lag" in res:
            lag = dict(res["lag"])
            d = lag.pop("dcf", None)
            if d:
                lag["dcf_summary"] = _r({k: v for k, v in d.items()
                                         if k not in ("lag_bins_d", "dcf", "dcf_err", "pairs_per_bin")})
            out["lag"] = _r(lag)
        if prov.get("is_mmdc"):
            out["attribution"] = ms.attribution()
        out["_table"] = (pd.DataFrame(rows), f"Variability of {label} ({len(rows)} band(s))")
        figs = []
        title = f"{label}: light curves with Bayesian blocks (flares shaded)"
        figs.append({**self._png_blocks(clean_tbl, res, title),
                     "plotly_spec": blocks_plotly_spec(clean_tbl, res, res["_blocks"], title), "caption": title})
        if res.get("lag", {}).get("dcf"):
            t2 = f"{label}: DCF {res['lag']['band_a']} vs {res['lag']['band_b']}"
            figs.append({**self._png_dcf(res["lag"], t2), "plotly_spec": dcf_plotly_spec(res["lag"], t2), "caption": t2})
        t3 = f"{label}: photon index vs flux"
        ispec = index_plotly_spec(clean_tbl, res["per_band"], t3)
        if ispec:
            figs.append({**self._png_index(clean_tbl, res["per_band"], t3), "plotly_spec": ispec, "caption": t3})
        out["_figures"] = figs
        return out

    # static PNG fallbacks -----------------------------------------------------
    def _save(self, fig, name: str) -> Dict[str, Any]:
        import uuid

        render = self.plotting._save_and_encode(fig, f"{name}_{uuid.uuid4().hex[:10]}")
        return {"image_base64": render.get("base64_png"), "path": render.get("web_url")}

    def _png_blocks(self, data: pd.DataFrame, res: Dict[str, Any], title: str) -> Dict[str, Any]:
        plt = self.plotting._apply_style(dark=False)
        bands = [b for b in res["bands_analysed"] if b in res["_blocks"]][:4] or res["bands_analysed"][:1]
        fig, axes = plt.subplots(len(bands), 1, figsize=(9, 2.3 * len(bands) + 0.8), sharex=True, squeeze=False)
        for i, b in enumerate(bands):
            ax = axes[i][0]
            g = data[(data["band"] == b) & ~data["is_upper_limit"].astype(bool)]
            ax.errorbar(g["time_mjd"], g["value"], yerr=g["error"].fillna(0), fmt="o", ms=2, lw=0.5, color=_COLORS[i % 6])
            bb = res["_blocks"].get(b) or {}
            for blk in bb.get("blocks") or []:
                ax.hlines(blk["flux"], blk["start"], blk["end"], color="k", lw=1.3)
            for f in bb.get("flares") or []:
                ax.axvspan(f["start_mjd"], f["end_mjd"], color="red", alpha=0.15)
            if bb.get("threshold") is not None:
                ax.axhline(bb["threshold"], ls=":", color="red", lw=0.8)
            ax.set_ylabel(b, fontsize=8)
        axes[-1][0].set_xlabel("MJD")
        axes[0][0].set_title(title, fontsize=10)
        fig.tight_layout()
        return self._save(fig, "variability_blocks")

    def _png_dcf(self, lag: Dict[str, Any], title: str) -> Dict[str, Any]:
        plt = self.plotting._apply_style(dark=False)
        d = lag["dcf"]
        fig, ax = plt.subplots(figsize=(7, 4))
        xs = [c for c, v in zip(d["lag_bins_d"], d["dcf"]) if v is not None]
        ys = [v for v in d["dcf"] if v is not None]
        es = [(e or 0) for e, v in zip(d["dcf_err"], d["dcf"]) if v is not None]
        ax.errorbar(xs, ys, yerr=es, fmt="o-", ms=3)
        if d.get("peak_lag_d") is not None:
            ax.axvline(d["peak_lag_d"], color="red", lw=1)
        ax.set_xlabel(f"lag [d] (positive = {lag['band_b']} lags {lag['band_a']})")
        ax.set_ylabel("DCF")
        ax.set_title(title, fontsize=10)
        fig.tight_layout()
        return self._save(fig, "variability_dcf")

    def _png_index(self, data: pd.DataFrame, per_band: List[Dict[str, Any]], title: str) -> Dict[str, Any]:
        plt = self.plotting._apply_style(dark=False)
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, row in enumerate(r for r in per_band if r.get("index_flux") and "slope" in r["index_flux"]):
            g = data[(data["band"] == row["band"]) & ~data["is_upper_limit"].astype(bool)]
            g = g[np.isfinite(g["spectral_index"]) & (g["value"] > 0)]
            lf = np.log10(g["value"].to_numpy(float))
            tr = row["index_flux"]
            ax.scatter(lf, g["spectral_index"], s=8, color=_COLORS[i % 6], label=f"{row['band']} ({tr['trend']})")
            xx = np.array([lf.min(), lf.max()])
            ax.plot(xx, tr["intercept"] + tr["slope"] * xx, color=_COLORS[i % 6])
        ax.set_xlabel("log10 flux")
        ax.set_ylabel("photon index")
        ax.legend(fontsize=7)
        ax.set_title(title, fontsize=10)
        fig.tight_layout()
        return self._save(fig, "variability_index")
