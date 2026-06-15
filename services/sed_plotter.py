# services/sed_plotter.py
"""
Photometric SED Plotter (FEATURES.md I4).

Plots multi-band photometry as a spectral energy distribution (SED) in
log-log space, with optional blackbody / modified-blackbody ("greybody")
model overlays fitted to the data.

Input photometry points accept any of:
    {"wavelength_um": 870, "flux_mjy": 12.1, "flux_err_mjy": 0.9, "label": "ALMA B7"}
    {"frequency_ghz": 345, "flux_mjy": 12.1}

Conventions match services/fits_service.py: PNG saved to RENDERED_DIR,
result dict carries {"success", "image_path", "caption", ...}.
"""

import os
import uuid
import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

RENDERED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "rendered_images")
os.makedirs(RENDERED_DIR, exist_ok=True)

C_UM_GHZ = 299792.458  # c in micron*GHz (lambda_um = C / nu_GHz)

# Physical constants (SI)
_H = 6.62607015e-34
_C = 2.99792458e8
_KB = 1.380649e-23


def _normalize_points(photometry: List[Dict[str, Any]]):
    """Convert input points to arrays of (wavelength_um, flux_mjy, err_mjy, label)."""
    wl, flux, err, labels = [], [], [], []
    for p in photometry:
        if not isinstance(p, dict):
            continue
        lam = p.get("wavelength_um")
        nu = p.get("frequency_ghz")
        if lam is None and nu is None:
            continue
        if lam is None:
            nu = float(nu)
            if nu <= 0:
                continue
            lam = C_UM_GHZ / nu
        lam = float(lam)
        f = p.get("flux_mjy")
        if lam <= 0 or f is None:
            continue
        f = float(f)
        if f <= 0:
            continue
        wl.append(lam)
        flux.append(f)
        e = p.get("flux_err_mjy")
        err.append(float(e) if e is not None and float(e) > 0 else np.nan)
        labels.append(str(p.get("label") or ""))

    order = np.argsort(wl)
    return (np.array(wl)[order], np.array(flux)[order],
            np.array(err)[order], [labels[i] for i in order])


def modified_blackbody_mjy(wavelength_um: np.ndarray, temperature_k: float,
                           beta: float, scale: float) -> np.ndarray:
    """
    Optically-thin modified blackbody (greybody):
        S_nu ∝ nu^beta * B_nu(T)
    `scale` is a free normalization in mJy at the model's peak.
    beta=0 reduces to a pure blackbody shape.
    """
    nu = _C / (np.asarray(wavelength_um, dtype=float) * 1e-6)  # Hz
    x = _H * nu / (_KB * max(temperature_k, 1.0))
    # Planck B_nu up to constants, with overflow-safe exponent
    with np.errstate(over="ignore"):
        planck = nu ** 3 / np.expm1(np.clip(x, 1e-12, 700))
    model = nu ** beta * planck
    peak = model.max() if model.size and np.isfinite(model).any() else 1.0
    return scale * model / peak


def _fit_greybody(wl_um: np.ndarray, flux_mjy: np.ndarray, beta: float):
    """
    Coarse-but-robust greybody fit: grid search over temperature, with the
    normalization solved analytically in log space. Returns (T, scale, chi).
    Suited for the few-point SEDs typical of photometric tables.
    """
    temps = np.geomspace(5, 3000, 220)
    best = None
    log_data = np.log10(flux_mjy)
    for t in temps:
        shape = modified_blackbody_mjy(wl_um, t, beta, 1.0)
        good = shape > 0
        if not good.all():
            continue
        offset = np.mean(log_data - np.log10(shape))
        resid = float(np.sum((log_data - (np.log10(shape) + offset)) ** 2))
        if best is None or resid < best[2]:
            best = (float(t), float(10 ** offset), resid)
    return best  # may be None


def plot_sed(
    photometry: List[Dict[str, Any]],
    title: str = "",
    target: str = "",
    overlay_model: str = "none",
    temperature_k: Optional[float] = None,
    beta: float = 1.8,
) -> Dict[str, Any]:
    """
    Render a photometric SED (flux vs wavelength, log-log).

    Args:
        photometry: list of photometric points (see module docstring)
        title: figure title (defaults to "<target> SED")
        target: source name used in default title/caption
        overlay_model: "none" | "blackbody" | "modified_blackbody"
            - "blackbody": beta forced to 0
            - "modified_blackbody": dust greybody with the given beta
        temperature_k: fix the model temperature; if omitted the
            temperature is fitted to the points
        beta: dust emissivity index for the modified blackbody (default 1.8)

    Returns:
        {"success": True, "image_path": ..., "caption": ...,
         "n_points": ..., "model": {...} | None}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        if not photometry or not isinstance(photometry, list):
            return {"success": False, "error": "Provide a non-empty list of photometric points"}

        wl, flux, err, labels = _normalize_points(photometry)
        if wl.size == 0:
            return {"success": False,
                    "error": "No valid points — each needs wavelength_um or frequency_ghz plus positive flux_mjy"}

        # ── Optional model overlay ────────────────────────────────────
        model_info = None
        model_curve = None
        overlay_model = (overlay_model or "none").strip().lower()
        if overlay_model in ("blackbody", "modified_blackbody"):
            use_beta = 0.0 if overlay_model == "blackbody" else float(beta)
            if temperature_k:
                t_fit = float(temperature_k)
                shape = modified_blackbody_mjy(wl, t_fit, use_beta, 1.0)
                offset = np.mean(np.log10(flux) - np.log10(np.clip(shape, 1e-300, None)))
                scale = float(10 ** offset)
            else:
                if wl.size < 2:
                    return {"success": False,
                            "error": "Model fitting needs at least 2 photometric points "
                                     "(or pass temperature_k explicitly)"}
                fit = _fit_greybody(wl, flux, use_beta)
                if fit is None:
                    return {"success": False, "error": "Model fit failed for these points"}
                t_fit, scale, _ = fit

            wl_model = np.geomspace(wl.min() / 3, wl.max() * 3, 400)
            model_curve = (wl_model, modified_blackbody_mjy(wl_model, t_fit, use_beta, scale))
            model_info = {
                "type": overlay_model,
                "temperature_k": round(t_fit, 1),
                "beta": use_beta,
                "temperature_fitted": temperature_k is None,
            }

        # ── Plot ──────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")

        has_err = np.isfinite(err)
        ax.errorbar(
            wl[has_err], flux[has_err], yerr=err[has_err],
            fmt="o", color="#06b6d4", ecolor="#38bdf8", elinewidth=1.2,
            capsize=3, markersize=7, markeredgecolor="white",
            markeredgewidth=0.6, zorder=5,
        )
        if (~has_err).any():
            ax.plot(wl[~has_err], flux[~has_err], "o", color="#06b6d4",
                    markersize=7, markeredgecolor="white",
                    markeredgewidth=0.6, zorder=5)

        # Per-point labels
        for x, y, lab in zip(wl, flux, labels):
            if lab:
                ax.annotate(lab, (x, y), textcoords="offset points",
                            xytext=(6, 8), color="#cbd5e1", fontsize=8)

        if model_curve is not None:
            label = (f"{'Blackbody' if model_info['beta'] == 0 else 'Mod. blackbody'} "
                     f"T={model_info['temperature_k']:.0f} K"
                     + (f", beta={model_info['beta']:.1f}" if model_info['beta'] else ""))
            ax.plot(model_curve[0], model_curve[1], color="#f59e0b",
                    linewidth=1.6, linestyle="--", label=label, zorder=4)
            legend = ax.legend(loc="best", framealpha=0.25, fontsize=9)
            for text in legend.get_texts():
                text.set_color("white")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Wavelength (micron)", color="white", fontsize=11)
        ax.set_ylabel("Flux density (mJy)", color="white", fontsize=11)
        ax.tick_params(which="both", colors="white", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.grid(which="both", color="#334155", alpha=0.3, linewidth=0.5)

        # Secondary frequency axis
        def um_to_ghz(x):
            return C_UM_GHZ / np.clip(x, 1e-12, None)

        sec = ax.secondary_xaxis("top", functions=(um_to_ghz, um_to_ghz))
        sec.set_xlabel("Frequency (GHz)", color="white", fontsize=10)
        sec.tick_params(colors="white", labelsize=8)

        full_title = title or (f"{target} — Spectral Energy Distribution" if target
                               else "Spectral Energy Distribution")
        ax.set_title(full_title, color="white", fontsize=13, pad=14)

        img_name = f"sed_{uuid.uuid4().hex[:10]}.png"
        img_path = os.path.join(RENDERED_DIR, img_name)
        fig.savefig(img_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)

        return {
            "success": True,
            "image_path": f"/api/images/{img_name}",
            "caption": full_title,
            "n_points": int(wl.size),
            "wavelength_range_um": [round(float(wl.min()), 4), round(float(wl.max()), 4)],
            "model": model_info,
        }

    except Exception as e:
        logger.error(f"[SED] plot_sed failed: {e}")
        return {"success": False, "error": str(e)}
