"""Server-side Data Lab catalog analysis and plotting tools."""

from __future__ import annotations

import ast
import difflib
import html
import math
import os
import re
import uuid
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from services.datalab_result_store import default_result_store
from services.plotting import PlottingService


# Whitelisted math functions usable inside plot expressions (evaluated on
# pandas Series / numpy arrays). Needed for derived quantities like absolute
# magnitude: phot_g_mean_mag + 5*log10(parallax/100).
_EXPR_FUNCS: Dict[str, Any] = {
    "log10": np.log10,
    "log": np.log,
    "log2": np.log2,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "exp": np.exp,
    "power": np.power,
    "arcsinh": np.arcsinh,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "arctan2": np.arctan2,
    "deg2rad": np.deg2rad,
    "rad2deg": np.rad2deg,
    "minimum": np.minimum,
    "maximum": np.maximum,
}

_EXPR_GRAMMAR_NOTE = (
    "Expressions may combine result columns with + - * / ** %, parentheses, numeric "
    "literals, and the functions: " + ", ".join(sorted(_EXPR_FUNCS)) + "."
)

_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)
_ALLOWED_UNARYOPS = (ast.USub, ast.UAdd)


def _clean_label(text: Any) -> str:
    """Sanitize a model-supplied display string (plot title/axis label).

    Models sometimes HTML-escape angle brackets in tool arguments (e.g. a title
    "NSC density g&lt;18" instead of "g<18"); matplotlib then renders the literal
    entity. Unescape once and trim so the figure reads correctly.
    """
    if text is None:
        return ""
    return html.unescape(str(text)).strip()


def _truncation_warnings(provenance: Any) -> List[str]:
    """Warnings when the source result was LIMIT-truncated (spatially biased).

    Rows come back in storage order, which is spatially clustered — a plot of a
    capped result covers one corner of the field (live P7 excluded Hydra II;
    live P9's map excluded Pal 5 itself)."""
    prov = dict(provenance or {})
    if not prov.get("limit_truncated"):
        return []
    limit = prov.get("row_limit")
    return [
        f"TRUNCATED SAMPLE: the source result hit its row cap (LIMIT {limit}) — the rows "
        "are a storage-order, spatially clustered slice, NOT the full selection. Do NOT "
        "present this plot as the on-sky distribution; rebuild it from "
        "datalab_density_aggregate / datalab_density_vetting (server-side GROUP BY), and "
        "state the truncation in the answer."
    ]


def _stamp_truncation_caption(fig, warnings: List[str]) -> None:
    """Draw a visible truncation caveat on the figure itself so a biased map can
    never be presented as the full sky distribution without the reader seeing it."""
    if not warnings:
        return
    fig.text(
        0.5, 0.005,
        "⚠ TRUNCATED SAMPLE (hit row cap) — spatially biased; not the full distribution",
        ha="center", va="bottom", fontsize=8, color="crimson",
    )


def _plotly_points_spec(
    x,
    y,
    *,
    x_label: str,
    y_label: str,
    title: str,
    color=None,
    color_label: Optional[str] = None,
    text=None,
    invert_x: bool = False,
    invert_y: bool = False,
    log_x: bool = False,
    mode: str = "markers",
    marker_symbol: Optional[str] = None,
    error_y=None,
    max_points: int = 4000,
    sort_by_color_desc: bool = False,
) -> Dict[str, Any]:
    """JSON-safe single-trace Plotly spec.

    The frontend ships the BASIC Plotly bundle (scatter/bar/pie only — no
    heatmap, no WebGL, no 3D), so every interactive Data Lab figure must be
    expressed as an SVG 'scatter' trace. Density maps become square markers
    colored by count; wedges become the 2D cartesian projection.
    """
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    cv = np.asarray(color, dtype=float) if color is not None else None
    tv = list(text) if text is not None else None
    ev = np.asarray(error_y, dtype=float) if error_y is not None else None
    if len(xv) > max_points:
        if sort_by_color_desc and cv is not None:
            # Keep the highest-count cells so density structure survives the cap.
            sel = np.argsort(cv)[::-1][:max_points]
        else:
            sel = np.random.default_rng(0).choice(len(xv), size=max_points, replace=False)
        xv, yv = xv[sel], yv[sel]
        cv = cv[sel] if cv is not None else None
        ev = ev[sel] if ev is not None else None
        tv = [tv[i] for i in sel] if tv is not None else None
    marker: Dict[str, Any] = {"size": 4, "opacity": 0.7}
    if marker_symbol:
        marker["symbol"] = marker_symbol
    if cv is not None:
        marker["color"] = [round(float(v), 4) for v in cv]
        marker["colorscale"] = "Viridis"
        marker["showscale"] = True
        if color_label:
            marker["colorbar"] = {"title": {"text": color_label}}
    trace: Dict[str, Any] = {
        "type": "scatter",
        "mode": mode,
        "x": [round(float(v), 4) for v in xv],
        "y": [round(float(v), 4) for v in yv],
        "marker": marker,
    }
    if tv is not None:
        trace["text"] = [str(t) for t in tv]
    if ev is not None:
        trace["error_y"] = {"type": "data", "array": [round(float(v), 4) for v in ev], "visible": True}
    layout: Dict[str, Any] = {
        "title": {"text": title},
        "xaxis": {"title": {"text": x_label}},
        "yaxis": {"title": {"text": y_label}},
        "showlegend": False,
        "margin": {"l": 55, "r": 15, "t": 45, "b": 45},
    }
    if invert_x:
        layout["xaxis"]["autorange"] = "reversed"
    if invert_y:
        layout["yaxis"]["autorange"] = "reversed"
    if log_x:
        layout["xaxis"]["type"] = "log"
    return {"data": [trace], "layout": layout}


def catalog_scatter(
    result_id: str,
    x_expr: str,
    y_expr: str,
    *,
    color_by: Optional[str] = None,
    invert_y: bool = False,
    invert_x: bool = False,
    title: str = "Data Lab catalog scatter",
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    overlay_locus: Optional[str] = None,
    result_store: Any = None,
    plotting_service: Optional[PlottingService] = None,
) -> Dict[str, Any]:
    title = _clean_label(title)
    res = _get_result(result_id, result_store)
    frame = res.dataframe
    x = _eval_expression(frame, x_expr)
    y = _eval_expression(frame, y_expr)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    plotting = plotting_service or PlottingService()
    plt = plotting._apply_style(dark=False)
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    if color_by and color_by in frame.columns:
        c_raw = frame.loc[mask, color_by]
        c_num = pd.to_numeric(c_raw, errors="coerce")
        if c_num.notna().any():
            sc = ax.scatter(x, y, c=c_num, s=18, alpha=0.8, edgecolors="none", cmap="viridis")
            fig.colorbar(sc, ax=ax, label=color_by)
        else:
            cats = c_raw.astype(str)
            for cat, group_idx in cats.groupby(cats).groups.items():
                ax.scatter(x.loc[group_idx], y.loc[group_idx], s=18, alpha=0.8, label=str(cat), edgecolors="none")
            ax.legend(title=color_by, fontsize=8)
    else:
        ax.scatter(x, y, s=18, alpha=0.8, edgecolors="none")
    _locus_info: Optional[Dict[str, Any]] = None
    if overlay_locus:
        _locus_info = _overlay_locus(ax, str(overlay_locus), x, y)
    ax.set_xlabel(x_label or x_expr)
    ax.set_ylabel(y_label or y_expr)
    ax.set_title(title)
    if invert_x:
        ax.invert_xaxis()
    if invert_y:
        ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    trunc = _truncation_warnings(res.provenance)
    _stamp_truncation_caption(fig, trunc)
    extra: Dict[str, Any] = {"points": int(mask.sum())}
    if _locus_info:
        extra.update(_locus_info)
    if trunc:
        extra["warnings"] = trunc
    spec_color = None
    if color_by and color_by in frame.columns:
        spec_color_series = pd.to_numeric(frame.loc[mask, color_by], errors="coerce")
        if spec_color_series.notna().any():
            spec_color = np.asarray(spec_color_series, dtype=float)
    extra["plotly_spec"] = _plotly_points_spec(
        np.asarray(x, dtype=float), np.asarray(y, dtype=float),
        x_label=x_label or x_expr, y_label=y_label or y_expr, title=title,
        color=spec_color, color_label=color_by if spec_color is not None else None,
        invert_x=invert_x, invert_y=invert_y,
    )
    return _plot_result(plotting, fig, "datalab_scatter", result_id, res.provenance, extra=extra)


def sky_density_map(
    result_id: str,
    *,
    mode: str = "hist2d",
    ra_col: Optional[str] = None,
    dec_col: Optional[str] = None,
    count_col: str = "source_count",
    bins: int = 80,
    healpix_col: str = "healpix",
    nside: Optional[int] = None,
    order: str = "nested",
    matched_filter: bool = False,
    sigma_small: float = 1.0,
    sigma_large: float = 3.0,
    peak_threshold: float = 3.0,
    max_peaks: int = 10,
    log_scale: bool = True,
    title: str = "Data Lab sky density map",
    result_store: Any = None,
    plotting_service: Optional[PlottingService] = None,
) -> Dict[str, Any]:
    title = _clean_label(title)
    res = _get_result(result_id, result_store)
    frame = res.dataframe.copy()
    plotting = plotting_service or PlottingService()
    plt = plotting._apply_style(dark=False)
    peaks: List[Dict[str, float]] = []

    mode_key = str(mode or "hist2d").lower()
    if mode_key == "healpix" or (healpix_col in frame.columns and ra_col is None and dec_col is None):
        if healpix_col not in frame.columns:
            raise ValueError(f"HEALPix column {healpix_col!r} not found")
        # The pixelization stored with the result is authoritative — decoding
        # RING pixels as NESTED (the old default) scattered a 2° cone across
        # 100° of sky (live DS-P8), and _infer_nside underestimates nside for
        # partial-sky maps. Caller values only apply when the result carries
        # no pixelization metadata.
        try:
            hp_meta = dict((res.provenance or {}).get("healpix") or {})
        except Exception:
            hp_meta = {}
        if hp_meta.get("scheme"):
            order = "ring" if str(hp_meta["scheme"]).strip().upper().startswith("RING") else "nested"
        if hp_meta.get("nside"):
            nside = int(hp_meta["nside"])
        elif nside is None:
            nside = _infer_nside(frame[healpix_col])
        # Decide log-scaling once from the actual counts so every render path
        # (healpy mollview, sparse scatter) agrees and extra["log_scale"]
        # reports what was ACTUALLY applied — a map with no positive finite
        # counts can't be log-scaled, so we fall back to linear.
        pix_values = _values(frame, count_col)
        applied_log = bool(log_scale) and _log_norm(pix_values) is not None
        fig = plt.figure(figsize=(6.0, 4.0))
        used_healpy = _try_healpy_plot(fig, frame, healpix_col, count_col, int(nside), order, title, applied_log)
        if not used_healpy:
            ax = fig.add_subplot(111)
            ra, dec = _healpix_centers(frame[healpix_col], int(nside), order)
            values = pix_values
            norm = _log_norm(values) if applied_log else None
            if norm is not None:
                # Zero/NaN/negative pixels have no log — mask them to the
                # background rather than flooring them to the min color, which
                # would paint surveyed-but-empty pixels as low-density sources.
                values = np.where(np.isfinite(values) & (values > 0), values, np.nan)
            sc = ax.scatter(ra, dec, c=values, s=20, cmap="viridis", edgecolors="none", norm=norm)
            cb_label = count_col if count_col in frame.columns else "count"
            if norm is not None:
                cb_label += " (log scale)"
            fig.colorbar(sc, ax=ax, label=cb_label)
            ax.set_xlabel("RA (deg)")
            ax.set_ylabel("Dec (deg)")
            ax.invert_xaxis()
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        extra = {"mode": "healpix", "nside": int(nside), "order": order, "log_scale": applied_log, "peaks": peaks}
        trunc = _truncation_warnings(res.provenance)
        if trunc:
            _stamp_truncation_caption(fig, trunc)
            extra["warnings"] = trunc
        spec_ra, spec_dec = _healpix_centers(frame[healpix_col], int(nside), order)
        spec_vals = np.asarray(pix_values, dtype=float)
        spec_ok = np.isfinite(spec_vals) & ((spec_vals > 0) if applied_log else np.ones(len(spec_vals), bool))
        extra["plotly_spec"] = _plotly_points_spec(
            np.asarray(spec_ra)[spec_ok], np.asarray(spec_dec)[spec_ok],
            x_label="RA (deg)", y_label="Dec (deg)", title=title,
            color=(np.log10(spec_vals[spec_ok]) if applied_log else spec_vals[spec_ok]),
            color_label=(f"log10 {count_col}" if applied_log else count_col),
            invert_x=True, marker_symbol="square", sort_by_color_desc=True,
        )
        return _plot_result(plotting, fig, "datalab_density", result_id, res.provenance, extra=extra)

    ra_name = _pick_column(frame, ra_col, ["ra", "ra_bin", "mean_fiber_ra", "s_ra"])
    dec_name = _pick_column(frame, dec_col, ["dec", "dec_bin", "mean_fiber_dec", "s_dec"])
    ra = pd.to_numeric(frame[ra_name], errors="coerce")
    dec = pd.to_numeric(frame[dec_name], errors="coerce")
    weights = _values(frame, count_col) if count_col in frame.columns else None
    mask = np.isfinite(ra) & np.isfinite(dec)
    ra_v = np.asarray(ra[mask], dtype=float)
    dec_v = np.asarray(dec[mask], dtype=float)
    weight_v = np.asarray(weights[mask], dtype=float) if weights is not None else None

    hist, xedges, yedges = np.histogram2d(ra_v, dec_v, bins=int(bins), weights=weight_v)
    if matched_filter:
        peaks = _matched_filter_peaks(hist, xedges, yedges, sigma_small, sigma_large, peak_threshold, max_peaks)

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    # Peak detection runs on the raw counts above; only the rendered color scale
    # is log-transformed. Empty bins (count 0) fall below LogNorm's vmin floor and
    # render as the background — an honest "no data here" rather than "low density".
    norm = _log_norm(hist) if log_scale else None
    mesh = ax.pcolormesh(xedges, yedges, hist.T, shading="auto", cmap="viridis", norm=norm)
    cb_label = count_col if count_col in frame.columns else "count"
    if norm is not None:
        cb_label += " (log scale)"
    fig.colorbar(mesh, ax=ax, label=cb_label)
    if peaks:
        ax.scatter([p["ra"] for p in peaks], [p["dec"] for p in peaks], marker="x", c="red", s=60, label="Peaks")
        ax.legend(fontsize=8)
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.invert_xaxis()
    ax.set_title(title)
    fig.tight_layout()
    trunc = _truncation_warnings(res.provenance)
    _stamp_truncation_caption(fig, trunc)
    extra: Dict[str, Any] = {"mode": "hist2d", "log_scale": norm is not None, "peaks": peaks}
    if trunc:
        extra["warnings"] = trunc
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    gx, gy = np.meshgrid(xc, yc, indexing="ij")
    nz = hist > 0
    cell_vals = hist[nz]
    extra["plotly_spec"] = _plotly_points_spec(
        gx[nz], gy[nz],
        x_label="RA (deg)", y_label="Dec (deg)", title=title,
        color=(np.log10(cell_vals) if norm is not None else cell_vals),
        color_label=(f"log10 {cb_label.replace(' (log scale)', '')}" if norm is not None else cb_label),
        invert_x=True, marker_symbol="square", sort_by_color_desc=True,
    )
    return _plot_result(plotting, fig, "datalab_density", result_id, res.provenance, extra=extra)


def period_fold(
    result_id: str,
    *,
    time_col: str = "mjd",
    mag_col: str = "cmag",
    error_col: Optional[str] = "cerr",
    band: Optional[str] = None,
    band_col: str = "filter",
    min_frequency: float = 1.0,
    max_frequency: float = 10.0,
    title: str = "Data Lab period-folded light curve",
    result_store: Any = None,
    plotting_service: Optional[PlottingService] = None,
) -> Dict[str, Any]:
    try:
        from astropy.timeseries import LombScargle
    except ImportError as exc:
        raise ImportError("astropy.timeseries is required for period folding") from exc

    title = _clean_label(title)
    res = _get_result(result_id, result_store)
    frame = res.dataframe
    for col in (time_col, mag_col):
        if col not in frame.columns:
            raise ValueError(f"Column {col!r} not found")
    # Single-band selection: NSC/DES/SMASH light curves interleave g/r/i/z epochs
    # in one table, and folding mixed bands smears the phased light curve (each
    # band has its own zero-point and amplitude). When a band is requested and the
    # frame carries a band/filter column, restrict to it; otherwise leave the data
    # untouched and report band_applied=None so we never pretend to have filtered.
    band_applied: Optional[str] = None
    fold_warnings: List[str] = []
    if band is not None and str(band).strip():
        want = str(band).strip()
        if band_col not in frame.columns:
            # The caller explicitly asked to fold ONE band, but the frame has no
            # such column. Folding all bands anyway would silently smear the curve
            # (mixed zero-points/amplitudes) and look successful — fail loudly so
            # the caller fixes the column name instead of trusting a wrong result.
            raise ValueError(
                f"band={want!r} requested but band column {band_col!r} not found; "
                f"pass band_col=<your filter column> or omit band"
            )
        band_series = frame[band_col].astype(str).str.strip().str.casefold()
        frame = frame[band_series == want.casefold()]
        band_applied = want
        if frame.empty:
            raise ValueError(f"No rows with {band_col}={want!r} to fold")
    elif band_col in frame.columns:
        # No band requested but the light curve mixes filters: folding them
        # together smears the phased curve (per-band zero-points/amplitudes).
        # Default to the best-sampled band and SAY so, instead of silently
        # producing a mixed-band fold (live P14).
        band_series = frame[band_col].astype(str).str.strip().str.casefold()
        counts = band_series.value_counts()
        if len(counts) > 1:
            pick = str(counts.idxmax())
            frame = frame[band_series == pick]
            band_applied = pick
            fold_warnings.append(
                f"Light curve mixes {len(counts)} bands; folded only the best-sampled "
                f"band '{pick}' ({int(counts.max())} epochs). Pass band=<name> to fold another."
            )
    t = pd.to_numeric(frame[time_col], errors="coerce")
    mag = pd.to_numeric(frame[mag_col], errors="coerce")
    # Reject non-finite AND sentinel magnitudes: surveys pad missing photometry
    # with |mag| >= 90 (e.g. 99.99 / -99), which would inject a spurious flat
    # baseline into the periodogram if folded.
    mask = np.isfinite(t) & np.isfinite(mag) & (np.abs(mag) < 90.0)
    # Only weight by uncertainties when enough epochs have FINITE, POSITIVE errors;
    # otherwise fall back to an unweighted periodogram. Passing NaN/zero dy to
    # LombScargle yields all-NaN power and crashes nanargmax.
    err = None
    use_errors = False
    if error_col and error_col in frame.columns:
        err = pd.to_numeric(frame[error_col], errors="coerce")
        err_ok = np.isfinite(err) & (err > 0)
        if int((mask & err_ok).sum()) >= 5:
            mask = mask & err_ok
            use_errors = True
    t_v = np.asarray(t[mask], dtype=float)
    mag_v = np.asarray(mag[mask], dtype=float)
    dy = np.asarray(err[mask], dtype=float) if use_errors else None
    if len(t_v) < 5:
        raise ValueError("period_fold requires at least five finite observations")

    if float(min_frequency) <= 0 or float(max_frequency) <= float(min_frequency):
        raise ValueError("period_fold requires 0 < min_frequency < max_frequency")
    ls = LombScargle(t_v, mag_v, dy=dy)
    freq, power = ls.autopower(minimum_frequency=float(min_frequency), maximum_frequency=float(max_frequency))
    best_idx = int(np.nanargmax(power))
    best_frequency = float(freq[best_idx])
    best_period = 1.0 / best_frequency
    phase = ((t_v - np.nanmin(t_v)) / best_period) % 1.0
    order = np.argsort(phase)
    time_span = float(np.nanmax(t_v) - np.nanmin(t_v))

    # False-alarm probability of the best peak (Baluev approximation — the
    # astropy default). A tiny FAP means "not noise", NOT "the period is right":
    # aliases carry low FAPs too, which is why the alternates are reported.
    # The searched frequency range MUST be passed explicitly: astropy otherwise
    # derives its own default ceiling and the FAP would describe a different
    # trial range than the periodogram actually searched. (guard CX-05)
    fap_kwargs = {
        "minimum_frequency": float(min_frequency),
        "maximum_frequency": float(max_frequency),
    }
    fap: Optional[float] = None
    try:
        fap = float(ls.false_alarm_probability(power[best_idx], **fap_kwargs))
    except Exception:
        fap = None

    # Honesty verdict — the tool must declare a null result itself: the model
    # demonstrably won't (live P14 claimed "consistent with an RR Lyrae" for a
    # grid-edge best period with FAP=0.28; the post-fix run got FAP=1.0 at the
    # 1-day alias). A best peak within one resolution element (Δf ≈ 1/T) of the
    # searched-window boundary is a window artifact, and a large FAP is noise.
    freq_res = (1.0 / time_span) if time_span > 0 else 0.0
    grid_edge = bool(
        best_idx <= 0
        or best_idx >= len(freq) - 1
        or (best_frequency - float(freq[0])) < freq_res
        or (float(freq[-1]) - best_frequency) < freq_res
    )
    significance_notes: List[str] = []
    if grid_edge:
        significance_notes.append(
            f"best period {best_period:.5g} d lies at the edge of the searched window "
            f"({1.0 / float(max_frequency):.5g}-{1.0 / float(min_frequency):.5g} d) - likely a "
            "window artifact; widen min_frequency/max_frequency and re-run"
        )
    if fap is None:
        significance_notes.append("false-alarm probability could not be computed")
    elif fap > 0.1:
        significance_notes.append(
            f"false-alarm probability {fap:.2g} > 0.1 - the peak is consistent with noise"
        )
    period_significant = bool(fap is not None and fap <= 0.1 and not grid_edge)
    if not period_significant:
        fold_warnings.append(
            "NO SIGNIFICANT PERIOD detected: " + "; ".join(significance_notes) + ". Do not "
            "present the folded curve as a real periodicity."
        )

    # Top alternate peaks: greedy pick by power, excluding a few natural
    # resolution elements (Δf ≈ 1/T) around already-accepted peaks so one broad
    # peak is not reported three times. ±1/day aliases survive by design.
    exclusion = 3.0 / time_span if time_span > 0 else 0.0
    alternates: List[Dict[str, Any]] = []
    if exclusion > 0.0:
        accepted = [best_frequency]
        finite_power = np.where(np.isfinite(power), power, -np.inf)
        for idx in np.argsort(finite_power)[::-1]:
            if len(alternates) >= 3:
                break
            f_i = float(freq[idx])
            if not np.isfinite(finite_power[idx]):
                break
            if any(abs(f_i - f_acc) < exclusion for f_acc in accepted):
                continue
            accepted.append(f_i)
            entry = {
                "period_days": 1.0 / f_i,
                "frequency_per_day": f_i,
                "power": float(power[idx]),
            }
            try:
                entry["false_alarm_probability"] = float(ls.false_alarm_probability(power[idx], **fap_kwargs))
            except Exception:
                pass
            alternates.append(entry)

    plotting = plotting_service or PlottingService()
    plt = plotting._apply_style(dark=False)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.4))
    ax1.plot(freq, power, color="#0072B2", lw=1.1)
    ax1.axvline(best_frequency, color="#D55E00", ls="--", lw=1.0)
    ax1.set_xlabel("Frequency (1/day)")
    ax1.set_ylabel("Lomb-Scargle power")
    fap_label = f", FAP={fap:.2g}" if fap is not None else ""
    if period_significant:
        ax1.set_title(f"Best period = {best_period:.5g} d{fap_label}")
    else:
        # The figure itself must carry the null verdict — captions get dropped.
        ax1.set_title(f"NO significant period (best {best_period:.5g} d{fap_label})", color="crimson")
    if dy is not None:
        ax2.errorbar(phase[order], mag_v[order], yerr=dy[order], fmt="o", ms=3, alpha=0.8)
        ax2.errorbar(phase[order] + 1.0, mag_v[order], yerr=dy[order], fmt="o", ms=3, alpha=0.8)
    else:
        ax2.scatter(phase[order], mag_v[order], s=18, alpha=0.8)
        ax2.scatter(phase[order] + 1.0, mag_v[order], s=18, alpha=0.8)
    ax2.set_xlabel("Phase")
    ax2.set_ylabel(mag_col)
    ax2.invert_yaxis()
    ax2.set_title(title)
    fig.tight_layout()
    return _plot_result(
        plotting,
        fig,
        "datalab_period",
        result_id,
        res.provenance,
        extra={
            "best_period_days": best_period,
            "best_frequency_per_day": best_frequency,
            "false_alarm_probability": fap,
            "period_significant": period_significant,
            "significance": "significant" if period_significant else "no_significant_period",
            "significance_notes": significance_notes,
            "searched_period_days": [1.0 / float(max_frequency), 1.0 / float(min_frequency)],
            "use_errors": use_errors,
            "warnings": fold_warnings,
            "alternate_periods": alternates,
            "plotly_spec": _period_fold_plotly_spec(
                freq, power, best_frequency, phase, mag_v, dy, order, mag_col,
                title=(f"Best period = {best_period:.5g} d{fap_label}"
                       if period_significant
                       else f"NO significant period (best {best_period:.5g} d{fap_label})"),
            ),
            "points": int(len(t_v)),
            "band": band_applied,
        },
    )


def _period_fold_plotly_spec(freq, power, best_frequency, phase, mag_v, dy, order, mag_col, *, title):
    """Two-panel interactive fold: periodogram (left) + phase-folded curve (right)."""
    f = np.asarray(freq, dtype=float)
    p = np.asarray(power, dtype=float)
    if len(f) > 2000:
        stride = int(np.ceil(len(f) / 2000.0))
        keep = np.zeros(len(f), dtype=bool)
        keep[::stride] = True
        keep[int(np.nanargmax(p))] = True  # never drop the peak itself
        f, p = f[keep], p[keep]
    ph = np.asarray(phase, dtype=float)[order]
    mg = np.asarray(mag_v, dtype=float)[order]
    er = np.asarray(dy, dtype=float)[order] if dy is not None else None
    fold_trace: Dict[str, Any] = {
        "type": "scatter",
        "mode": "markers",
        "name": "folded",
        "x": [round(float(v), 4) for v in np.concatenate([ph, ph + 1.0])],
        "y": [round(float(v), 4) for v in np.concatenate([mg, mg])],
        "marker": {"size": 4, "opacity": 0.75},
        "xaxis": "x2",
        "yaxis": "y2",
    }
    if er is not None:
        fold_trace["error_y"] = {
            "type": "data",
            "array": [round(float(v), 4) for v in np.concatenate([er, er])],
            "visible": True,
        }
    return {
        "data": [
            {
                "type": "scatter",
                "mode": "lines",
                "name": "periodogram",
                "x": [round(float(v), 5) for v in f],
                "y": [round(float(v), 5) for v in p],
                "line": {"width": 1},
            },
            fold_trace,
        ],
        "layout": {
            "title": {"text": title},
            "grid": {"rows": 1, "columns": 2, "pattern": "independent"},
            "xaxis": {"title": {"text": "Frequency (1/day)"}},
            "yaxis": {"title": {"text": "Lomb-Scargle power"}},
            "xaxis2": {"title": {"text": "Phase"}},
            "yaxis2": {"title": {"text": mag_col}, "autorange": "reversed"},
            "showlegend": False,
            "margin": {"l": 55, "r": 15, "t": 45, "b": 45},
            "shapes": [{
                "type": "line", "xref": "x", "yref": "paper",
                "x0": round(float(best_frequency), 5), "x1": round(float(best_frequency), 5),
                "y0": 0, "y1": 1, "line": {"dash": "dash", "width": 1},
            }],
        },
    }


_SED_MAX_OBJECTS = 300


def sed_plot(
    result_id: str,
    *,
    row_index: Optional[int] = None,
    row_indices: Optional[Sequence[int]] = None,
    sample_n: Optional[int] = None,
    filter_columns: Optional[Mapping[str, str]] = None,
    title: str = "Data Lab SED",
    result_store: Any = None,
    plotting_service: Optional[PlottingService] = None,
    svo_client: Any = None,
) -> Dict[str, Any]:
    """SED(s) from stored photometry rows.

    ``row_indices`` selects specific rows; ``sample_n`` overlays the first N
    rows — each object a faint line with the per-band MEDIAN highlighted,
    capped at 300 objects; an explicit ``row_index`` draws one object. With NO
    selector at all, multi-row results default to the SAMPLE overlay: two live
    P10 runs asked for "a few hundred" SEDs and the model called with no
    selector both times despite the tool description — description-level
    nudges demonstrably don't reach the call, so the default must.
    """
    title = _clean_label(title)
    res = _get_result(result_id, result_store)
    frame = res.dataframe
    if frame.empty:
        raise ValueError("sed_plot requires a non-empty result")
    mapping = dict(filter_columns or {
        "g": "dered_mag_g",
        "r": "dered_mag_r",
        "z": "dered_mag_z",
        "w1": "dered_mag_w1",
        "w2": "dered_mag_w2",
    })

    auto_sampled = False
    if row_indices is not None:
        indices = [int(i) for i in row_indices]
    elif sample_n is not None and int(sample_n) > 1:
        indices = list(range(min(int(sample_n), len(frame))))
    elif row_index is not None:
        indices = [int(row_index)]
    elif len(frame) > 1:
        # No selector on a multi-row result: sample-first (see docstring).
        indices = list(range(min(_SED_MAX_OBJECTS, len(frame))))
        auto_sampled = True
    else:
        indices = [0]
    capped = len(indices) > _SED_MAX_OBJECTS
    indices = indices[:_SED_MAX_OBJECTS]
    multi = len(indices) > 1

    present = {filt: col for filt, col in mapping.items() if col in frame.columns}
    if not present:
        raise ValueError("No SED filter columns found in the result")
    # Per-object SEDs: filters must be shared across the figure, so collect the
    # union of filters with at least one finite magnitude among selected rows.
    seds: List[tuple] = []  # (index, {filter: mag})
    used_filters: List[str] = []
    for idx in indices:
        row = frame.iloc[idx]
        sed: Dict[str, float] = {}
        for filt, col in present.items():
            mag = _as_float(row[col])
            if mag is None or not np.isfinite(mag) or mag >= 90:
                continue
            sed[str(filt)] = float(mag)
            if filt not in used_filters:
                used_filters.append(str(filt))
        if sed:
            seds.append((idx, sed))
    if not seds:
        raise ValueError("No finite SED magnitudes found")

    if svo_client is None:
        from integrations.svo_fps_client import SvoFpsClient

        svo_client = SvoFpsClient()
    wavelengths = svo_client.wavelengths(used_filters)
    wl = {f: float(wavelengths[f]["effective_micron"]) for f in used_filters}
    filt_order = sorted(used_filters, key=lambda f: wl[f])

    plotting = plotting_service or PlottingService()
    plt = plotting._apply_style(dark=False)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    spec_traces: List[Dict[str, Any]] = []
    if multi:
        for _idx, sed in seds:
            xs = [wl[f] for f in filt_order if f in sed]
            ys = [sed[f] for f in filt_order if f in sed]
            ax.plot(xs, ys, lw=0.6, alpha=0.18, color="#0072B2")
            spec_traces.append({
                "type": "scatter", "mode": "lines",
                "x": [round(v, 4) for v in xs], "y": [round(v, 4) for v in ys],
                "line": {"width": 1, "color": "rgba(0,114,178,0.18)"},
                "hoverinfo": "skip", "showlegend": False,
            })
        med_x = [wl[f] for f in filt_order]
        med_y = [float(np.median([sed[f] for _i, sed in seds if f in sed])) for f in filt_order]
        ax.plot(med_x, med_y, marker="o", lw=2.0, color="#D55E00", label=f"median of {len(seds)}")
        ax.legend(fontsize=8)
        spec_traces.append({
            "type": "scatter", "mode": "lines+markers", "name": f"median of {len(seds)}",
            "x": [round(v, 4) for v in med_x], "y": [round(v, 4) for v in med_y],
            "line": {"width": 2.5, "color": "#D55E00"}, "marker": {"size": 6},
        })
    else:
        _idx, sed = seds[0]
        xs = [wl[f] for f in filt_order if f in sed]
        ys = [sed[f] for f in filt_order if f in sed]
        labels = [f for f in filt_order if f in sed]
        ax.plot(xs, ys, marker="o", lw=1.2)
        for xx, yy, filt in zip(xs, ys, labels):
            ax.text(xx, yy, filt, fontsize=8, ha="left", va="bottom")
        spec_traces.append({
            "type": "scatter", "mode": "lines+markers+text", "text": labels,
            "textposition": "top center",
            "x": [round(v, 4) for v in xs], "y": [round(v, 4) for v in ys],
            "marker": {"size": 7},
        })
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("Wavelength (micron)")
    ax.set_ylabel("Magnitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    extra: Dict[str, Any] = {
        "filters": used_filters,
        "wavelengths": wavelengths,
        "objects_plotted": len(seds),
    }
    notes: List[str] = []
    if capped:
        notes.append(f"SED overlay capped at {_SED_MAX_OBJECTS} objects.")
    if auto_sampled:
        extra["auto_sampled"] = True
        notes.append(
            f"No row selector given — rendered the sample overlay ({len(seds)} of "
            f"{len(frame)} rows with the per-band median). Pass row_index for a single object."
        )
    if notes:
        extra["warnings"] = notes
    extra["plotly_spec"] = {
        "data": spec_traces,
        "layout": {
            "title": {"text": title},
            "xaxis": {"title": {"text": "Wavelength (micron)"}, "type": "log"},
            "yaxis": {"title": {"text": "Magnitude"}, "autorange": "reversed"},
            "showlegend": multi,
            "margin": {"l": 55, "r": 15, "t": 45, "b": 45},
        },
    }
    return _plot_result(plotting, fig, "datalab_sed", result_id, res.provenance, extra=extra)


def lss_wedge(
    result_id: str,
    *,
    ra_col: Optional[str] = None,
    dec_col: Optional[str] = None,
    z_col: str = "z",
    class_col: Optional[str] = None,
    pie_slice: bool = False,
    title: str = "Data Lab large-scale structure wedge",
    result_store: Any = None,
    plotting_service: Optional[PlottingService] = None,
) -> Dict[str, Any]:
    try:
        from astropy.cosmology import Planck18 as cosmology
    except ImportError as exc:
        raise ImportError("astropy.cosmology is required for lss_wedge") from exc

    title = _clean_label(title)
    res = _get_result(result_id, result_store)
    frame = res.dataframe
    ra_name = _pick_column(frame, ra_col, ["ra", "mean_fiber_ra"])
    dec_name = _pick_column(frame, dec_col, ["dec", "mean_fiber_dec"])
    if z_col not in frame.columns:
        raise ValueError(f"Column {z_col!r} not found")
    ra = pd.to_numeric(frame[ra_name], errors="coerce")
    dec = pd.to_numeric(frame[dec_name], errors="coerce")
    z = pd.to_numeric(frame[z_col], errors="coerce")
    mask = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(z) & (z > 0)
    ra_rad = np.deg2rad(np.asarray(ra[mask], dtype=float))
    dec_rad = np.deg2rad(np.asarray(dec[mask], dtype=float))
    z_v = np.asarray(z[mask], dtype=float)
    dist = np.asarray(cosmology.comoving_distance(z_v).value, dtype=float)
    x = dist * np.cos(dec_rad) * np.cos(ra_rad)
    y = dist * np.cos(dec_rad) * np.sin(ra_rad)
    z_cart = dist * np.sin(dec_rad)

    plotting = plotting_service or PlottingService()
    plt = plotting._apply_style(dark=False)
    fig = plt.figure(figsize=(6.0, 4.6))
    colors = None
    if class_col and class_col in frame.columns:
        colors = pd.Categorical(frame.loc[mask, class_col].astype(str)).codes
    if pie_slice:
        ax = fig.add_subplot(111)
        sc = ax.scatter(x, y, c=colors if colors is not None else z_v, s=15, alpha=0.75, cmap="viridis")
        ax.set_xlabel("X (Mpc)")
        ax.set_ylabel("Y (Mpc)")
        ax.set_aspect("equal", adjustable="box")
    else:
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(x, y, z_cart, c=colors if colors is not None else z_v, s=15, alpha=0.75, cmap="viridis")
        ax.set_xlabel("X (Mpc)")
        ax.set_ylabel("Y (Mpc)")
        ax.set_zlabel("Z (Mpc)")
        _set_axes_equal(ax, x, y, z_cart)
    fig.colorbar(sc, ax=ax, label=class_col if colors is not None else "redshift")
    ax.set_title(title)
    fig.tight_layout()
    # Interactive counterpart: the basic Plotly bundle has no 3D traces, so the
    # spec is the 2D comoving-plane projection (X-Y), colored by redshift.
    spec = _plotly_points_spec(
        x, y,
        x_label="X (Mpc, comoving)", y_label="Y (Mpc, comoving)", title=title,
        color=(np.asarray(colors, dtype=float) if colors is not None else z_v),
        color_label=(class_col if colors is not None else "redshift"),
    )
    spec["layout"]["yaxis"]["scaleanchor"] = "x"
    return _plot_result(
        plotting, fig, "datalab_lss", result_id, res.provenance,
        extra={"points": int(mask.sum()), "pie_slice": bool(pie_slice), "plotly_spec": spec},
    )


def _get_result(result_id: str, result_store: Any = None):
    store = result_store or default_result_store()
    return store.get(result_id)


def _resolve_expr_column(frame: pd.DataFrame, name: str, expr: str) -> pd.Series:
    if name in frame.columns:
        return pd.to_numeric(frame[name], errors="coerce")
    # Postgres folds unquoted SELECT aliases to lowercase (M_G -> m_g), so a
    # case-insensitive fallback lets the model reference the alias it wrote.
    lowered = {str(col).lower(): col for col in frame.columns}
    match = lowered.get(name.lower())
    if match is not None:
        return pd.to_numeric(frame[match], errors="coerce")
    available = [str(col) for col in frame.columns][:40]
    close = difflib.get_close_matches(name, [str(col) for col in frame.columns], n=3, cutoff=0.6)
    hint = f" Did you mean {', '.join(repr(c) for c in close)}?" if close else ""
    raise ValueError(
        f"Column {name!r} not found for expression {expr!r}.{hint} "
        f"Available columns: {', '.join(available)}. {_EXPR_GRAMMAR_NOTE}"
    )


def _validate_expr_node(node: ast.AST, expr: str) -> None:
    if isinstance(node, ast.Expression):
        _validate_expr_node(node.body, expr)
    elif isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINOPS):
        _validate_expr_node(node.left, expr)
        _validate_expr_node(node.right, expr)
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, _ALLOWED_UNARYOPS):
        _validate_expr_node(node.operand, expr)
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _EXPR_FUNCS or node.keywords:
            raise ValueError(f"Unsafe or unsupported expression: {expr!r}. {_EXPR_GRAMMAR_NOTE}")
        for arg in node.args:
            _validate_expr_node(arg, expr)
    elif isinstance(node, ast.Name):
        pass
    elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        pass
    else:
        raise ValueError(f"Unsafe or unsupported expression: {expr!r}. {_EXPR_GRAMMAR_NOTE}")


def _eval_expression(frame: pd.DataFrame, expr: str) -> pd.Series:
    text = str(expr or "").strip()
    if not text:
        raise ValueError("expression is required")
    if text in frame.columns:
        return pd.to_numeric(frame[text], errors="coerce")
    if "__" in text:
        raise ValueError(f"Unsafe or unsupported expression: {expr!r}. {_EXPR_GRAMMAR_NOTE}")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        raise ValueError(f"Unsafe or unsupported expression: {expr!r}. {_EXPR_GRAMMAR_NOTE}") from None
    _validate_expr_node(tree, expr)
    local_dict: Dict[str, Any] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in _EXPR_FUNCS and node.id not in local_dict:
            local_dict[node.id] = _resolve_expr_column(frame, node.id, expr)
    if not local_dict:
        raise ValueError(f"Expression {expr!r} must reference at least one result column. {_EXPR_GRAMMAR_NOTE}")
    env = {"__builtins__": {}}
    env.update(_EXPR_FUNCS)
    try:
        with np.errstate(all="ignore"):
            value = eval(compile(tree, "<datalab_expr>", "eval"), env, local_dict)  # noqa: S307 - AST-validated above
    except Exception as exc:
        raise ValueError(f"Failed to evaluate expression {expr!r}: {exc}. {_EXPR_GRAMMAR_NOTE}") from None
    return pd.Series(value, index=frame.index)


def _overlay_locus(ax: Any, name: str, x: pd.Series, y: pd.Series) -> Optional[Dict[str, Any]]:
    if name.lower() not in {"wd", "wd_sequence", "white_dwarf"} or x.empty:
        return None
    xs = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 80)
    ys = 11.5 + 5.0 * xs
    yrange = (float(np.nanmin(y)), float(np.nanmax(y)))
    if yrange[0] > yrange[1]:
        yrange = (yrange[1], yrange[0])
    mask = (ys >= yrange[0] - abs(yrange[1] - yrange[0])) & (ys <= yrange[1] + abs(yrange[1] - yrange[0]))
    if mask.any():
        ax.plot(xs[mask], ys[mask], color="#D55E00", lw=1.2, ls="--", label="WD locus")
        ax.legend(fontsize=8)
    # Side-of-line classification (DECISION-03 default). The drawn line IS the WD
    # locus y = 11.5 + 5.0*x; finite points on the FAINT side (larger magnitude,
    # y > line) are white-dwarf candidates, the rest are main-sequence/other.
    # Returned so the tool result can quote real counts and a model cannot call
    # the lower main sequence a "WD cooling track".
    line_at_x = 11.5 + 5.0 * np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    finite = np.isfinite(y_arr) & np.isfinite(line_at_x)
    is_wd = finite & (y_arr > line_at_x)
    n_wd = int(is_wd.sum())
    n_other = int((finite & ~is_wd).sum())
    return {
        "n_wd_candidates": n_wd,
        "n_other": n_other,
        "wd_locus_rule": "y = 11.5 + 5.0*x; faint side (y > line) = WD candidates",
        "wd_locus_note": (
            f"{n_wd} point(s) lie on the faint side of the white-dwarf locus "
            f"(y = 11.5 + 5.0*x) and are WD candidates; {n_other} point(s) lie on "
            f"the bright/main-sequence side. Do NOT label the main-sequence points "
            f"a white-dwarf cooling track."
        ),
    }


def _pick_column(frame: pd.DataFrame, preferred: Optional[str], candidates: Sequence[str]) -> str:
    if preferred:
        if preferred not in frame.columns:
            raise ValueError(f"Column {preferred!r} not found")
        return preferred
    for col in candidates:
        if col in frame.columns:
            return col
    raise ValueError(f"None of the columns are available: {list(candidates)}")


def _values(frame: pd.DataFrame, count_col: str) -> np.ndarray:
    if count_col in frame.columns:
        return np.asarray(pd.to_numeric(frame[count_col], errors="coerce").fillna(0), dtype=float)
    return np.ones(len(frame), dtype=float)


def _log_norm(values: np.ndarray) -> Any:
    """LogNorm spanning the positive finite values, or None when a log scale is
    impossible (no positive data). vmin is floored to the smallest positive
    value so zero/NaN cells never drive log(0)/log(negative)."""
    from matplotlib.colors import LogNorm

    arr = np.asarray(values, dtype=float)
    finite_pos = arr[np.isfinite(arr) & (arr > 0)]
    if finite_pos.size == 0:
        return None
    vmin = float(finite_pos.min())
    vmax = float(finite_pos.max())
    if not (vmax > vmin):
        vmax = vmin * 10.0  # single-valued map — give the colorbar a decade
    return LogNorm(vmin=vmin, vmax=vmax)


def _matched_filter_peaks(
    hist: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray,
    sigma_small: float,
    sigma_large: float,
    threshold: float,
    max_peaks: int,
) -> List[Dict[str, float]]:
    from scipy.ndimage import gaussian_filter, maximum_filter

    smooth_small = gaussian_filter(hist, sigma=float(sigma_small))
    smooth_large = gaussian_filter(hist, sigma=float(sigma_large))
    dog = smooth_small - smooth_large
    finite = dog[np.isfinite(dog)]
    if finite.size == 0:
        return []
    med = float(np.nanmedian(finite))
    std = float(np.nanstd(finite)) or 1.0
    sig = (dog - med) / std
    local_max = sig == maximum_filter(sig, size=3)
    candidates = np.argwhere(local_max & (sig >= float(threshold)))
    if candidates.size == 0:
        # No local maximum clears the threshold → no overdensity. Do NOT fall back to
        # the global max: on a flat/noisy field that would report a false peak and feed
        # a bogus coordinate into the cutout-grid / candidate-ranking workflow.
        return []
    ranked = sorted(candidates, key=lambda ij: sig[int(ij[0]), int(ij[1])], reverse=True)
    peaks: List[Dict[str, float]] = []
    for i, j in ranked[: int(max_peaks)]:
        i_i, j_i = int(i), int(j)
        peaks.append({
            "ra": float((xedges[i_i] + xedges[i_i + 1]) / 2.0),
            "dec": float((yedges[j_i] + yedges[j_i + 1]) / 2.0),
            "significance": float(sig[i_i, j_i]),
        })
    return peaks


def _try_healpy_plot(fig: Any, frame: pd.DataFrame, healpix_col: str, count_col: str, nside: int, order: str, title: str, log_scale: bool = True) -> bool:
    # Guard against a dense all-sky allocation for high nside: hp.nside2npix(4096)
    # is ~2e8 floats and would OOM the worker for a sparse map. Above the cap, fall
    # back to the sparse scatter path (renders only the returned pixels).
    max_dense = int(os.getenv("DATALAB_HEALPY_MAX_NSIDE", "512"))
    if int(nside) > max_dense:
        return False
    try:
        import healpy as hp
    except ImportError:
        return False
    values = np.full(hp.nside2npix(nside), np.nan)
    pix = pd.to_numeric(frame[healpix_col], errors="coerce")
    counts = _values(frame, count_col)
    for p, c in zip(pix, counts):
        if np.isfinite(p):
            p_i = int(p)
            if 0 <= p_i < len(values):
                values[p_i] = c
    # Guard the log scale the same way the other render paths do: healpy derives
    # vmin from ALL finite pixels (zeros included), so a bare norm="log" raises
    # "values must all be positive" on any map containing a zero-count pixel.
    # Mask non-positive pixels to the background and pin min/max to the positive
    # range so mollview never sees a non-positive vmin.
    norm = _log_norm(values) if log_scale else None
    kwargs: Dict[str, Any] = {}
    if norm is not None:
        values = np.where(np.isfinite(values) & (values > 0), values, np.nan)
        kwargs = {"norm": "log", "min": float(norm.vmin), "max": float(norm.vmax)}
    hp.mollview(
        values,
        nest=str(order).lower().startswith("nest"),
        title=title,
        fig=fig.number,
        **kwargs,
    )
    return True


def _healpix_centers(pixels: Iterable[Any], nside: int, order: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        from astropy_healpix import HEALPix
        from astropy.coordinates import ICRS
        import astropy.units as u
    except ImportError as exc:
        raise ImportError("astropy-healpix is required for HEALPix density rendering") from exc
    hp = HEALPix(nside=nside, order="nested" if str(order).lower().startswith("nest") else "ring", frame=ICRS())
    pix = np.asarray(pd.to_numeric(pd.Series(pixels), errors="coerce"), dtype=float)
    lon, lat = hp.healpix_to_lonlat(pix.astype(int))
    return lon.to_value(u.deg), lat.to_value(u.deg)


def _infer_nside(pixels: Iterable[Any]) -> int:
    max_pix = int(np.nanmax(pd.to_numeric(pd.Series(pixels), errors="coerce")))
    nside = 1
    while 12 * nside * nside <= max_pix:
        nside *= 2
    return nside


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _set_axes_equal(ax: Any, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    ranges = [np.nanmax(vals) - np.nanmin(vals) for vals in (x, y, z) if len(vals)]
    radius = max(ranges) / 2.0 if ranges else 1.0
    for setter, vals in ((ax.set_xlim, x), (ax.set_ylim, y), (ax.set_zlim, z)):
        center = (float(np.nanmax(vals)) + float(np.nanmin(vals))) / 2.0 if len(vals) else 0.0
        setter(center - radius, center + radius)


def _plot_result(
    plotting: PlottingService,
    fig: Any,
    prefix: str,
    result_id: str,
    provenance: Mapping[str, Any],
    *,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    saved = plotting._save_and_encode(fig, f"{prefix}_{uuid.uuid4().hex[:10]}")
    payload = {
        "success": True,
        "result_id": result_id,
        "image_base64": saved.get("base64_png"),
        "path": saved.get("web_url"),
        "png_path": saved.get("png_path"),
        "pdf_path": saved.get("pdf_path"),
        "provenance": dict(provenance or {}),
    }
    payload.update(dict(extra or {}))
    return payload


__all__ = ["catalog_scatter", "sky_density_map", "period_fold", "sed_plot", "lss_wedge"]
