"""CMD population test for stellar overdensity candidates (DataLabBench L15, L03).

A satellite candidate is judged by its colour-magnitude diagram: a real dwarf
galaxy adds an OLD, metal-poor population (main-sequence turnoff / blue main
sequence, red-giant branch, sometimes a blue horizontal branch) on top of the
Milky Way field. The test here is the classic Hess difference:

* ONE server-side aggregate counts point sources per (g-r, g) cell inside the
  candidate aperture (``n_in``) and in a background annulus (``n_out``), so no
  row cap can truncate either sample;
* the annulus counts are scaled by the area ratio and subtracted;
* the excess is summed in distance-agnostic CMD regions (below) and a Poisson
  significance is computed per region.

The regions are generic colour windows for an old, metal-poor population in
g-r (no isochrone and no assumed distance), so the verdict is a screening
statement, not a membership analysis:

* ``bhb``: -0.4 <= g-r < 0.0 (blue horizontal branch);
* ``msto``: 0.0 <= g-r < 0.45, g >= 19 (turnoff and upper main sequence);
* ``rgb``: 0.45 <= g-r < 1.0, g < 24 (red-giant branch);
* ``red_field`` (control): 1.0 <= g-r < 1.8 (disc M dwarfs, compact red
  galaxies). An excess concentrated here is NOT an old population.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

REGIONS: Dict[str, Dict[str, Any]] = {
    "bhb": {"label": "blue horizontal branch", "color": (-0.4, 0.0), "mag": (17.0, 23.0)},
    "msto": {"label": "main-sequence turnoff / blue main sequence", "color": (0.0, 0.45), "mag": (19.0, 30.0)},
    "rgb": {"label": "red-giant branch", "color": (0.45, 1.0), "mag": (15.0, 24.0)},
    "red_field": {"label": "red field control (disc dwarfs / red galaxies)", "color": (1.0, 1.8), "mag": (15.0, 30.0)},
}
OLD_REGIONS = ("bhb", "msto", "rgb")

DEFAULT_APERTURE_DEG = 0.1
DEFAULT_ANNULUS_DEG = (0.25, 0.5)
COLOR_BIN = 0.1
MAG_BIN = 0.25
COLOR_WINDOW = (-1.0, 3.0)
MAG_WINDOW = (14.0, 27.0)
HESS_ROW_LIMIT = 5000

# Verdict thresholds (Poisson sigma of the area-scaled excess).
OLD_SIGMA_COHERENT = 4.0
FEATURE_SIGMA = 2.0
OLD_SIGMA_FIELD = 2.5


def _num(x: float) -> str:
    return f"{float(x):.8g}"


def build_hess_aggregate(
    catalog: str,
    table: str,
    ra: float,
    dec: float,
    *,
    aperture_deg: float = DEFAULT_APERTURE_DEG,
    annulus_deg: Sequence[float] = DEFAULT_ANNULUS_DEG,
    predicates: Optional[Sequence[str]] = None,
    color_bin: float = COLOR_BIN,
    mag_bin: float = MAG_BIN,
):
    """(sql, meta) for ONE aggregate of point-source counts per (g-r, g) cell,
    split into aperture (``n_in``) and annulus (``n_out``). Every predicate
    must come from the registry-validated builders."""
    from services import datalab_query_builders as builders
    from services import datalab_registry as reg

    info = reg.describe_table(catalog, table)
    ra_col, dec_col = info["ra_column"], info["dec_column"]
    g_col, r_col = reg.mag_column(catalog, table, "g"), reg.mag_column(catalog, table, "r")
    a_in = float(aperture_deg)
    a0, a1 = float(annulus_deg[0]), float(annulus_deg[1])
    if not (0 < a_in < a0 < a1 <= 2.0):
        raise ValueError("need 0 < aperture < annulus inner < annulus outer <= 2 deg")
    dist = f"q3c_dist({ra_col}, {dec_col}, {_num(ra)}, {_num(dec)})"
    clauses = [f"q3c_radial_query({ra_col}, {dec_col}, {_num(ra)}, {_num(dec)}, {_num(a1)})"]
    clauses += [f"({p})" for p in (predicates or []) if str(p).strip()]
    clauses += [f"({p})" for p in builders._sentinel_mag_predicates(catalog, table, [g_col, r_col])]
    # Bound the CMD plane so the aggregate has at most 40 x 52 = 2080 cells,
    # well under its 5000-row LIMIT (guard CX-05): nothing can be truncated.
    clauses += [f"({g_col} - {r_col}) BETWEEN {_num(COLOR_WINDOW[0])} AND {_num(COLOR_WINDOW[1])}",
                f"{g_col} BETWEEN {_num(MAG_WINDOW[0])} AND {_num(MAG_WINDOW[1])}"]
    sql = (
        f"SELECT FLOOR(({g_col} - {r_col}) / {_num(color_bin)}) AS cbin,\n"
        f"       FLOOR({g_col} / {_num(mag_bin)}) AS mbin,\n"
        f"       SUM(CASE WHEN {dist} <= {_num(a_in)} THEN 1 ELSE 0 END) AS n_in,\n"
        f"       SUM(CASE WHEN {dist} >= {_num(a0)} THEN 1 ELSE 0 END) AS n_out\n"
        f"FROM {info['qualified_name']}\n"
        "WHERE " + "\n  AND ".join(clauses) + "\n"
        "GROUP BY cbin, mbin\n"
        f"LIMIT {HESS_ROW_LIMIT}"
    )
    meta = builders._meta("cmd_hess_aggregate", info, aggregate=True, spatial_bound=True, row_limit=HESS_ROW_LIMIT)
    meta.update({"aperture_deg": a_in, "annulus_deg": [a0, a1], "color_bin": color_bin, "mag_bin": mag_bin,
                 "g_col": g_col, "r_col": r_col})
    return sql, meta


def area_ratio(aperture_deg: float, annulus_deg: Sequence[float]) -> float:
    """Aperture area / annulus area (flat-sky; fine at these radii)."""
    a0, a1 = float(annulus_deg[0]), float(annulus_deg[1])
    return float(aperture_deg) ** 2 / (a1 ** 2 - a0 ** 2)


def _cells(frame: Optional[pd.DataFrame], color_bin: float, mag_bin: float) -> pd.DataFrame:
    if frame is None or frame.empty or not {"cbin", "mbin", "n_in", "n_out"} <= set(frame.columns):
        return pd.DataFrame(columns=["color", "mag", "n_in", "n_out"])
    df = frame[["cbin", "mbin", "n_in", "n_out"]].apply(pd.to_numeric, errors="coerce").dropna()
    return pd.DataFrame({
        "color": (df["cbin"] + 0.5) * color_bin,
        "mag": (df["mbin"] + 0.5) * mag_bin,
        "n_in": df["n_in"].astype(float),
        "n_out": df["n_out"].astype(float),
    })


def population_test(
    frame: Optional[pd.DataFrame],
    *,
    aperture_deg: float = DEFAULT_APERTURE_DEG,
    annulus_deg: Sequence[float] = DEFAULT_ANNULUS_DEG,
    color_bin: float = COLOR_BIN,
    mag_bin: float = MAG_BIN,
) -> Dict[str, Any]:
    """Region excesses + significance + verdict from a Hess aggregate frame
    (columns cbin, mbin, n_in, n_out)."""
    ratio = area_ratio(aperture_deg, annulus_deg)
    cells = _cells(frame, color_bin, mag_bin)
    regions: Dict[str, Dict[str, Any]] = {}
    for key, reg_def in REGIONS.items():
        c0, c1 = reg_def["color"]
        m0, m1 = reg_def["mag"]
        sel = (cells["color"] >= c0) & (cells["color"] < c1) & (cells["mag"] >= m0) & (cells["mag"] < m1)
        n_in = float(cells.loc[sel, "n_in"].sum())
        n_out = float(cells.loc[sel, "n_out"].sum())
        expected = n_out * ratio
        excess = n_in - expected
        sigma = math.sqrt(max(n_in + ratio ** 2 * n_out, 1.0))
        regions[key] = {"label": reg_def["label"], "color_range": [c0, c1], "mag_range": [m0, m1],
                        "n_aperture": int(n_in), "n_background_scaled": round(expected, 1),
                        "excess": round(excess, 1), "significance": round(excess / sigma, 1)}
    old_in = sum(regions[k]["n_aperture"] for k in OLD_REGIONS)
    old_exp = sum(regions[k]["n_background_scaled"] for k in OLD_REGIONS)
    old_out = old_exp / ratio if ratio else 0.0
    old_excess = old_in - old_exp
    old_sig = old_excess / math.sqrt(max(old_in + ratio ** 2 * old_out, 1.0))
    features = [k for k in OLD_REGIONS if regions[k]["significance"] >= FEATURE_SIGMA and regions[k]["excess"] > 0]
    red = regions["red_field"]
    red_dominant = red["significance"] >= 3.0 and red["excess"] > max(old_excess, 0.0)
    n_total_in = int(cells["n_in"].sum()) if not cells.empty else 0
    if n_total_in == 0:
        verdict, reason = "inconclusive", "no point sources in the aperture (coverage gap or empty query)"
    elif red_dominant:
        verdict = "field-like"
        reason = (f"the excess is red (g-r >= 1.0: {red['excess']:+.0f} stars, {red['significance']:.1f} sigma), "
                  "as expected for disc dwarfs or a compact galaxy group, not an old metal-poor population")
    elif old_sig >= OLD_SIGMA_COHERENT and len(features) >= 2:
        verdict = "coherent old population"
        reason = (f"{old_excess:+.0f} excess stars in the old-population windows ({old_sig:.1f} sigma), with "
                  + ", ".join(f"{REGIONS[k]['label']} {regions[k]['excess']:+.0f} ({regions[k]['significance']:.1f} sigma)" for k in features))
    elif old_sig < OLD_SIGMA_FIELD:
        verdict = "field-like"
        reason = f"no significant old-population excess ({old_excess:+.0f} stars, {old_sig:.1f} sigma over the scaled annulus)"
    else:
        verdict = "inconclusive"
        reason = (f"{old_excess:+.0f} excess stars ({old_sig:.1f} sigma) but "
                  + (f"only {len(features)} CMD feature above {FEATURE_SIGMA:g} sigma" if len(features) < 2
                     else f"below the {OLD_SIGMA_COHERENT:g} sigma bar for a coherent population"))
    return {
        "verdict": verdict,
        "reason": reason,
        "old_population_excess": round(old_excess, 1),
        "old_population_significance": round(old_sig, 1),
        "features_detected": [REGIONS[k]["label"] for k in features],
        "regions": regions,
        "n_aperture": n_total_in,
        "aperture_deg": float(aperture_deg),
        "annulus_deg": [float(annulus_deg[0]), float(annulus_deg[1])],
        "area_ratio": round(ratio, 5),
        "method": ("Hess difference: point sources in the aperture minus the area-scaled background annulus, summed in "
                   "generic old-population g-r windows (BHB -0.4..0.0; turnoff/blue MS 0.0..0.45 at g >= 19; RGB 0.45..1.0) "
                   "plus a red control window (1.0..1.8); Poisson significance. No isochrone or distance is assumed."),
    }


def depth_summary(frame: Optional[pd.DataFrame], *, mag_bin: float = MAG_BIN, which: str = "n_in") -> Dict[str, Any]:
    """Turnover (peak) of the g-magnitude histogram: the practical depth where
    completeness starts to fall."""
    cells = _cells(frame, COLOR_BIN, mag_bin)
    if cells.empty or float(cells[which].sum()) <= 0:
        return {"turnover_mag": None, "note": "no sources to measure the depth"}
    hist = cells.groupby("mag")[which].sum().sort_index()
    peak_mag = float(hist.idxmax())
    return {"turnover_mag": round(peak_mag, 2), "bin_mag": mag_bin,
            "note": f"g-band counts peak at g ~ {peak_mag:.2f} and fall fainter: the catalogue is incomplete beyond it."}


def hess_difference_figure(frame: Optional[pd.DataFrame], test: Dict[str, Any], *, plotting, title: str,
                           color_bin: float = COLOR_BIN, mag_bin: float = MAG_BIN):
    """Two panels: aperture counts and the Hess difference (aperture minus the
    scaled annulus), with the CMD windows outlined. Returns a matplotlib figure."""
    plt = plotting._apply_style(dark=False)
    cells = _cells(frame, color_bin, mag_bin)
    ratio = float(test.get("area_ratio") or 0.0)
    cmin, cmax, mmin, mmax = -0.6, 1.8, 16.0, 25.5
    ce = np.arange(cmin, cmax + color_bin / 2, color_bin)
    me = np.arange(mmin, mmax + mag_bin / 2, mag_bin)
    h_in, _, _ = np.histogram2d(cells["color"], cells["mag"], bins=[ce, me], weights=cells["n_in"]) if not cells.empty else (np.zeros((len(ce) - 1, len(me) - 1)), None, None)
    h_out, _, _ = np.histogram2d(cells["color"], cells["mag"], bins=[ce, me], weights=cells["n_out"]) if not cells.empty else (np.zeros((len(ce) - 1, len(me) - 1)), None, None)
    diff = h_in - ratio * h_out
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.6), sharey=True)
    axes[0].pcolormesh(ce, me, h_in.T, cmap="Greys", shading="flat")
    lim = max(1.0, float(np.nanpercentile(np.abs(diff), 99)) if diff.size else 1.0)
    im = axes[1].pcolormesh(ce, me, diff.T, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="flat")
    for ax, sub in zip(axes, ("aperture", "aperture - scaled annulus")):
        ax.set_xlabel("g - r")
        ax.set_title(sub, fontsize=9)
        for key, reg_def in REGIONS.items():
            c0, c1 = reg_def["color"]
            m0, m1 = max(reg_def["mag"][0], mmin), min(reg_def["mag"][1], mmax)
            ax.add_patch(plt.Rectangle((c0, m0), c1 - c0, m1 - m0, fill=False, lw=0.8,
                                       ls="--" if key == "red_field" else "-", ec="tab:green" if key != "red_field" else "tab:gray"))
        ax.set_xlim(cmin, cmax)
        ax.set_ylim(mmax, mmin)
    axes[0].set_ylabel("g")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.03, label="excess stars per cell")
    fig.suptitle(f"{title}\n{test.get('verdict')}: old-population excess {test.get('old_population_excess')} "
                 f"({test.get('old_population_significance')} sigma)", fontsize=9)
    fig.tight_layout()
    return fig


def run_population_test(
    catalog: str,
    table: str,
    ra: float,
    dec: float,
    *,
    aperture_deg: float = DEFAULT_APERTURE_DEG,
    annulus_deg: Sequence[float] = DEFAULT_ANNULUS_DEG,
    predicates: Optional[Sequence[str]] = None,
    client: Any = None,
    result_store: Any = None,
    owner_id: Optional[str] = None,
    plotting: Any = None,
    title: Optional[str] = None,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """Run the Hess aggregate, the population test and (optionally) render the
    Hess-difference panel. Returns the test dict plus ``result_id``,
    ``validated_sql`` and, when plotting is given, ``image_base64`` / ``path``."""
    from services import datalab_orchestration as orch

    sql, meta = build_hess_aggregate(catalog, table, ra, dec, aperture_deg=aperture_deg,
                                     annulus_deg=annulus_deg, predicates=predicates)
    rid, result = orch._run_builder_sql(sql, meta, client=client, result_store=result_store,
                                        owner_id=owner_id, async_fallback=False, timeout=timeout)
    frame = getattr(result, "dataframe", None)
    test = population_test(frame, aperture_deg=aperture_deg, annulus_deg=annulus_deg)
    if frame is not None and len(frame) >= HESS_ROW_LIMIT:
        # Defensive: the bounded plane cannot reach the cap, but a truncated
        # aggregate must never carry a verdict.
        test.update({"verdict": "inconclusive", "reason": f"the Hess aggregate filled its {HESS_ROW_LIMIT}-cell cap (truncated)"})
    test.update({"result_id": rid, "validated_sql": sql, "catalog": catalog, "table": table,
                 "ra": float(ra), "dec": float(dec), "depth": depth_summary(frame)})
    if plotting is not None:
        import uuid

        fig = hess_difference_figure(frame, test, plotting=plotting,
                                     title=title or f"{catalog}.{table} Hess difference at ({ra:.3f}, {dec:.3f})")
        saved = plotting._save_and_encode(fig, f"datalab_hess_{uuid.uuid4().hex[:10]}")
        test.update({"image_base64": saved.get("base64_png"), "path": saved.get("web_url")})
    return test


__all__: List[str] = [
    "REGIONS",
    "area_ratio",
    "build_hess_aggregate",
    "depth_summary",
    "hess_difference_figure",
    "population_test",
    "run_population_test",
]
