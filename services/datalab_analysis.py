"""Server-side Data Lab catalog analysis and plotting tools."""

from __future__ import annotations

import ast
import difflib
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
    if overlay_locus:
        _overlay_locus(ax, str(overlay_locus), x, y)
    ax.set_xlabel(x_label or x_expr)
    ax.set_ylabel(y_label or y_expr)
    ax.set_title(title)
    if invert_x:
        ax.invert_xaxis()
    if invert_y:
        ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _plot_result(plotting, fig, "datalab_scatter", result_id, res.provenance, extra={"points": int(mask.sum())})


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
        fig = plt.figure(figsize=(6.0, 4.0))
        used_healpy = _try_healpy_plot(fig, frame, healpix_col, count_col, int(nside), order, title, log_scale)
        if not used_healpy:
            ax = fig.add_subplot(111)
            ra, dec = _healpix_centers(frame[healpix_col], int(nside), order)
            values = _values(frame, count_col)
            norm = _log_norm(values) if log_scale else None
            if norm is not None:
                # Floor zero/NaN pixels to vmin so returned cells stay visible
                # instead of being masked out by LogNorm.
                values = _floor_nonpositive(values, float(norm.vmin))
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
        extra = {"mode": "healpix", "nside": int(nside), "order": order, "log_scale": bool(log_scale), "peaks": peaks}
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
    return _plot_result(plotting, fig, "datalab_density", result_id, res.provenance, extra={"mode": "hist2d", "log_scale": norm is not None, "peaks": peaks})


def period_fold(
    result_id: str,
    *,
    time_col: str = "mjd",
    mag_col: str = "cmag",
    error_col: Optional[str] = "cerr",
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

    res = _get_result(result_id, result_store)
    frame = res.dataframe
    for col in (time_col, mag_col):
        if col not in frame.columns:
            raise ValueError(f"Column {col!r} not found")
    t = pd.to_numeric(frame[time_col], errors="coerce")
    mag = pd.to_numeric(frame[mag_col], errors="coerce")
    mask = np.isfinite(t) & np.isfinite(mag)
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

    ls = LombScargle(t_v, mag_v, dy=dy)
    freq, power = ls.autopower(minimum_frequency=float(min_frequency), maximum_frequency=float(max_frequency))
    best_idx = int(np.nanargmax(power))
    best_frequency = float(freq[best_idx])
    best_period = 1.0 / best_frequency
    phase = ((t_v - np.nanmin(t_v)) / best_period) % 1.0
    order = np.argsort(phase)

    plotting = plotting_service or PlottingService()
    plt = plotting._apply_style(dark=False)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.4))
    ax1.plot(freq, power, color="#0072B2", lw=1.1)
    ax1.axvline(best_frequency, color="#D55E00", ls="--", lw=1.0)
    ax1.set_xlabel("Frequency (1/day)")
    ax1.set_ylabel("Lomb-Scargle power")
    ax1.set_title(f"Best period = {best_period:.5g} d")
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
        extra={"best_period_days": best_period, "best_frequency_per_day": best_frequency, "points": int(len(t_v))},
    )


def sed_plot(
    result_id: str,
    *,
    row_index: int = 0,
    filter_columns: Optional[Mapping[str, str]] = None,
    title: str = "Data Lab SED",
    result_store: Any = None,
    plotting_service: Optional[PlottingService] = None,
    svo_client: Any = None,
) -> Dict[str, Any]:
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
    row = frame.iloc[int(row_index)]
    filters: List[str] = []
    mags: List[float] = []
    for filt, col in mapping.items():
        if col not in frame.columns:
            continue
        mag = _as_float(row[col])
        if mag is None or not np.isfinite(mag) or mag >= 90:
            continue
        filters.append(str(filt))
        mags.append(float(mag))
    if not filters:
        raise ValueError("No finite SED magnitudes found")

    if svo_client is None:
        from integrations.svo_fps_client import SvoFpsClient

        svo_client = SvoFpsClient()
    wavelengths = svo_client.wavelengths(filters)
    x = [float(wavelengths[f]["effective_micron"]) for f in filters]

    plotting = plotting_service or PlottingService()
    plt = plotting._apply_style(dark=False)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(x, mags, marker="o", lw=1.2)
    for xx, yy, filt in zip(x, mags, filters):
        ax.text(xx, yy, filt, fontsize=8, ha="left", va="bottom")
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("Wavelength (micron)")
    ax.set_ylabel("Magnitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _plot_result(plotting, fig, "datalab_sed", result_id, res.provenance, extra={"filters": filters, "wavelengths": wavelengths})


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
    return _plot_result(plotting, fig, "datalab_lss", result_id, res.provenance, extra={"points": int(mask.sum()), "pie_slice": bool(pie_slice)})


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


def _overlay_locus(ax: Any, name: str, x: pd.Series, y: pd.Series) -> None:
    if name.lower() not in {"wd", "wd_sequence", "white_dwarf"} or x.empty:
        return
    xs = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 80)
    ys = 11.5 + 5.0 * xs
    yrange = (float(np.nanmin(y)), float(np.nanmax(y)))
    if yrange[0] > yrange[1]:
        yrange = (yrange[1], yrange[0])
    mask = (ys >= yrange[0] - abs(yrange[1] - yrange[0])) & (ys <= yrange[1] + abs(yrange[1] - yrange[0]))
    if mask.any():
        ax.plot(xs[mask], ys[mask], color="#D55E00", lw=1.2, ls="--", label="WD locus")
        ax.legend(fontsize=8)


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


def _floor_nonpositive(values: np.ndarray, floor: float) -> np.ndarray:
    """Replace zero/NaN/negative cells with ``floor`` so they survive LogNorm."""
    arr = np.asarray(values, dtype=float)
    return np.where(np.isfinite(arr) & (arr > 0), arr, floor)


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
    hp.mollview(
        values,
        nest=str(order).lower().startswith("nest"),
        title=title,
        fig=fig.number,
        norm="log" if log_scale else None,
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
