"""NED photometry SED plotting for Quasar live imagery tools."""

from __future__ import annotations

import math
import uuid
from typing import Any, Dict, List, Optional

import numpy as np

from services.plotting import PlottingService


class NedPhotometryService:
    """Lazy astroquery-backed NED photometry plotter."""

    def __init__(self, *, plotting_service: Optional[PlottingService] = None):
        self.plotting_service = plotting_service or PlottingService()

    def sed_plot(self, target_name: str) -> Dict[str, Any]:
        target = str(target_name or "").strip()
        if not target:
            return {"success": False, "error": "target_name is required."}
        try:
            from astroquery.ipac.ned import Ned
        except ImportError as exc:
            return {"success": False, "error": f"NED photometry requires astroquery.ipac.ned: {exc}"}
        try:
            table = Ned.get_table(target, table="photometry")
            rows = self._clean_rows(table)
            if not rows:
                return {"success": False, "error": f"No positive NED photometry points found for {target}."}
            freq = np.asarray([row["frequency"] for row in rows], dtype=float)
            flux = np.asarray([row["flux_jy"] for row in rows], dtype=float)
            nu_fnu = freq * flux * 1e-23

            plt = self.plotting_service._apply_style(dark=False)
            fig, ax = plt.subplots(figsize=(7.5, 5.0))
            ax.scatter(freq, nu_fnu, s=18, alpha=0.75)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Frequency [Hz]")
            ax.set_ylabel("nu Fnu [erg s^-1 cm^-2]")
            ax.set_title(f"NED SED: {target} ({len(rows)} photometry points)")
            fig.tight_layout()
            render = self.plotting_service._save_and_encode(fig, f"ned_sed_{uuid.uuid4().hex[:10]}")
            return {
                "success": True,
                "image_base64": render.get("base64_png"),
                "path": render.get("web_url"),
                "png_path": render.get("png_path"),
                "pdf_path": render.get("pdf_path"),
                "target_name": target,
                "rowcount": len(rows),
                "preview": rows[:10],
                "provenance": {"service": "NASA/IPAC Extragalactic Database", "table": "photometry"},
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    @staticmethod
    def _clean_rows(table: Any) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        colnames = list(getattr(table, "colnames", []))

        def get_cell(row: Any, column: str) -> Any:
            if column not in colnames:
                return None
            try:
                return row[column]
            except Exception:
                return None

        for row in table:
            frequency = _positive_float(get_cell(row, "Frequency"))
            flux_jy = _positive_float(get_cell(row, "Flux Density"))
            if frequency is None or flux_jy is None:
                continue
            rows.append(
                {
                    "passband": _safe_str(get_cell(row, "Observed Passband")),
                    "frequency": frequency,
                    "flux_jy": flux_jy,
                    "refcode": _safe_str(get_cell(row, "Refcode")),
                }
            )
        return rows


def _positive_float(value: Any) -> Optional[float]:
    if np.ma.is_masked(value):
        return None
    try:
        if hasattr(value, "to_value"):
            value = value.to_value()
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out) or out <= 0:
        return None
    return out


def _safe_str(value: Any) -> Optional[str]:
    if value is None or np.ma.is_masked(value):
        return None
    text = str(value).strip()
    return text if text and text.lower() != "nan" else None


__all__ = ["NedPhotometryService"]
