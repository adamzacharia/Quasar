"""
PlottingService — Publication-quality astronomical figure generator for Quasar AI
Produces ApJ/MNRAS-compliant matplotlib figures with LaTeX labels,
colorblind-safe palettes, and dual PNG/PDF export.

Registered agent tools:
    - plot_alma_results(style, x_column, y_column, color_by)
    - plot_sky_map(ra_col, dec_col, color_by)
    - plot_spectrum(freqs, fluxes, title)
"""

import os
import io
import base64
import json
from typing import Optional, List, Dict, Any
from datetime import datetime

# Publication style rcParams matching AASTeX / MNRAS requirements
PUB_RCPARAMS = {
    "figure.dpi": 300,
    "figure.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.fontsize": 10,
    "legend.frameon": True,
    "legend.framealpha": 0.8,
    "lines.linewidth": 1.5,
    "axes.linewidth": 1.0,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
}

# Colorblind-safe palette (Wong 2011)
WONG_PALETTE = [
    "#E69F00", "#56B4E9", "#009E73",
    "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"
]

PLOT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "ui-pro", "public", "plots")


class PlottingService:
    """Generate publication-quality astronomical figures."""

    def __init__(self):
        os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

    def _apply_style(self, dark: bool = False):
        """Apply publication or dark-mode style."""
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
        if dark:
            plt.style.use("dark_background")
        else:
            import matplotlib as mpl
            mpl.rcParams.update(PUB_RCPARAMS)
        return plt

    def _save_and_encode(self, fig, filename: str) -> Dict[str, Any]:
        """Save figure to disk + return base64 PNG for inline display."""
        import matplotlib.pyplot as plt
        png_path = os.path.join(PLOT_OUTPUT_DIR, f"{filename}.png")
        pdf_path = os.path.join(PLOT_OUTPUT_DIR, f"{filename}.pdf")
        fig.savefig(png_path, format="png")
        fig.savefig(pdf_path, format="pdf")
        plt.close(fig)

        # Base64-encode PNG for inline UI embedding
        with open(png_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        web_url = f"/plots/{filename}.png"
        return {
            "success": True,
            "png_path": png_path,
            "pdf_path": pdf_path,
            "web_url": web_url,
            "base64_png": b64,
            "filename": filename,
        }

    def plot_alma_results(
        self,
        data_records: List[Dict],
        x_column: str = "frequency",
        y_column: str = "spatial_resolution",
        color_by: str = "band",
        title: str = "ALMA Observation Summary",
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        dark_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate a scatter plot from ALMA search results.
        
        Args:
            data_records: List of dicts (DataFrame.to_dict('records'))
            x_column: Column for x-axis
            y_column: Column for y-axis
            color_by: Column to color-code points by
            title: Plot title (LaTeX supported)
            x_label: X-axis label (LaTeX supported, auto-derived if None)
            y_label: Y-axis label (LaTeX supported, auto-derived if None)
            dark_mode: Use dark background instead of publication white
        """
        try:
            import pandas as pd
            plt = self._apply_style(dark=dark_mode)
            import matplotlib.cm as cm
            import numpy as np

            df = pd.DataFrame(data_records)
            if x_column not in df.columns or y_column not in df.columns:
                available = list(df.columns)
                return {"success": False, "error": f"Column not found. Available: {available}"}

            fig, ax = plt.subplots(figsize=(3.5, 3.0))  # ApJ single-column width

            # Color coding
            if color_by in df.columns:
                categories = df[color_by].astype(str).unique()
                color_map = {c: WONG_PALETTE[i % len(WONG_PALETTE)] for i, c in enumerate(sorted(categories))}
                for cat, group in df.groupby(df[color_by].astype(str)):
                    ax.scatter(group[x_column], group[y_column],
                               c=color_map[cat], label=str(cat), s=20, alpha=0.85, edgecolors='none')
                ax.legend(title=color_by.replace("_", " ").title(), markerscale=1.5)
            else:
                ax.scatter(df[x_column], df[y_column], s=20, color=WONG_PALETTE[1], alpha=0.85, edgecolors='none')

            ax.set_xlabel(x_label or x_column.replace("_", " ").title())
            ax.set_ylabel(y_label or y_column.replace("_", " ").title())
            ax.set_title(title, pad=8)
            ax.grid(True)
            fig.tight_layout()

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            return self._save_and_encode(fig, f"alma_scatter_{ts}")

        except Exception as e:
            return {"success": False, "error": str(e)}

    def plot_sky_map(
        self,
        data_records: List[Dict],
        ra_col: str = "s_ra",
        dec_col: str = "s_dec",
        color_by: Optional[str] = "band",
        title: str = "Source Sky Distribution",
        dark_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate a RA/Dec sky scatter plot (equatorial projection).
        """
        try:
            import pandas as pd
            plt = self._apply_style(dark=dark_mode)

            df = pd.DataFrame(data_records)
            if ra_col not in df.columns or dec_col not in df.columns:
                return {"success": False, "error": f"RA ({ra_col}) or Dec ({dec_col}) column not found."}

            fig, ax = plt.subplots(figsize=(5.0, 3.0))

            if color_by and color_by in df.columns:
                # Numeric color columns (mags, counts — live P9: big_gmag over
                # 5,000 rows) must map to a colorbar: the categorical branch
                # builds one legend entry per unique value, which exploded one
                # figure to 1585x236,838 px. Discrete numeric codes with few
                # values (band numbers, CCD ids) still read best as categories.
                numeric_vals = pd.to_numeric(df[color_by], errors="coerce")
                parseable = numeric_vals.notna().sum() >= max(1, int(0.95 * df[color_by].notna().sum()))
                if parseable and numeric_vals.nunique(dropna=True) > 8:
                    # Sentinel values (NSC gmag=99.99 for missing photometry)
                    # stretch a naive colorbar until real values are one color.
                    # Percentile clipping fails once sentinels exceed the tail
                    # fraction (live Pal 5 field: 2.2% at 99.99 puts p99 at the
                    # sentinel), so clip to a 3xIQR fence instead.
                    finite = numeric_vals.dropna()
                    vmin = vmax = None
                    if len(finite) >= 20:
                        q1, q3 = finite.quantile([0.25, 0.75])
                        iqr = float(q3 - q1)
                        if iqr > 0:
                            lo, hi = float(q1 - 3 * iqr), float(q3 + 3 * iqr)
                            if finite.min() < lo or finite.max() > hi:
                                vmin = max(float(finite.min()), lo)
                                vmax = min(float(finite.max()), hi)
                    sc = ax.scatter(df[ra_col], df[dec_col], c=numeric_vals,
                                    cmap="viridis", vmin=vmin, vmax=vmax,
                                    s=12, alpha=0.7, edgecolors='none')
                    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
                    cbar.set_label(color_by.replace("_", " ").title())
                else:
                    cats = df[color_by].astype(str)
                    n_unique = cats.nunique()
                    if n_unique > 20:  # keep string legends readable too
                        top = set(cats.value_counts().index[:20])
                        cats = cats.where(cats.isin(top), other=f"other ({n_unique - 20} values)")
                    color_map = {c: WONG_PALETTE[i % len(WONG_PALETTE)] for i, c in enumerate(sorted(cats.unique()))}
                    for cat, group in df.groupby(cats):
                        ax.scatter(group[ra_col], group[dec_col], c=color_map[cat],
                                   label=str(cat), s=12, alpha=0.7, edgecolors='none')
                    ax.legend(title=color_by.replace("_", " ").title(), markerscale=1.5, fontsize=8)
            else:
                ax.scatter(df[ra_col], df[dec_col], s=12, color=WONG_PALETTE[1], alpha=0.7)

            ax.set_xlabel("Right Ascension (deg)")
            ax.set_ylabel("Declination (deg)")
            ax.set_title(title, pad=8)
            ax.invert_xaxis()  # Astronomical convention: RA increases right-to-left
            ax.grid(True)
            fig.tight_layout()

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            return self._save_and_encode(fig, f"sky_map_{ts}")

        except Exception as e:
            return {"success": False, "error": str(e)}

    def plot_spectrum(
        self,
        frequencies: List[float],
        fluxes: List[float],
        title: str = "Spectral Line Profile",
        x_label: str = "Frequency (GHz)",
        y_label: str = r"Flux Density (Jy)",
        errors: Optional[List[float]] = None,
        line_labels: Optional[Dict[float, str]] = None,
        dark_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        Plot a 1D spectral line profile with optional error bars and line ID labels.
        
        Args:
            frequencies: List of frequency values (GHz)
            fluxes: List of flux density values (Jy)
            errors: Optional flux error bars (same length as fluxes)
            line_labels: Dict of {frequency_ghz: "Line Name"} for vertical label markers
        """
        try:
            plt = self._apply_style(dark=dark_mode)

            fig, ax = plt.subplots(figsize=(5.0, 3.0))

            if errors:
                ax.errorbar(frequencies, fluxes, yerr=errors,
                            fmt="-o", ms=3, color=WONG_PALETTE[1],
                            ecolor="gray", elinewidth=0.8, capsize=2, lw=1.2)
            else:
                ax.plot(frequencies, fluxes, color=WONG_PALETTE[1], lw=1.2)

            ax.axhline(0, color="gray", lw=0.6, linestyle="--")

            # Annotate known spectral lines
            if line_labels:
                for freq, name in line_labels.items():
                    ax.axvline(freq, color=WONG_PALETTE[5], lw=0.8, linestyle=":")
                    ax.text(freq, ax.get_ylim()[1] * 0.92, name,
                            rotation=90, va="top", ha="right", fontsize=7, color=WONG_PALETTE[5])

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(title, pad=8)
            ax.grid(True)
            fig.tight_layout()

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            return self._save_and_encode(fig, f"spectrum_{ts}")

        except Exception as e:
            return {"success": False, "error": str(e)}

    def plot_band_histogram(
        self,
        data_records: List[Dict],
        column: str = "band",
        title: str = "Observation Count by Band",
        dark_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate a histogram / bar chart of observation counts by band or year.
        """
        try:
            import pandas as pd
            plt = self._apply_style(dark=dark_mode)

            df = pd.DataFrame(data_records)
            if column not in df.columns:
                return {"success": False, "error": f"Column '{column}' not found."}

            counts = df[column].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(3.5, 2.8))
            bars = ax.bar(counts.index.astype(str), counts.values,
                          color=WONG_PALETTE[:len(counts)], edgecolor="black", linewidth=0.5)

            # Add count labels on top of each bar
            for bar, val in zip(bars, counts.values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        str(val), ha="center", va="bottom", fontsize=8)

            ax.set_xlabel(column.replace("_", " ").title())
            ax.set_ylabel("Number of Observations")
            ax.set_title(title, pad=8)
            ax.grid(axis="y", alpha=0.4)
            fig.tight_layout()

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            return self._save_and_encode(fig, f"histogram_{ts}")

        except Exception as e:
            return {"success": False, "error": str(e)}
