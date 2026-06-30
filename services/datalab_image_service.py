"""Data Lab SIA image selection, FITS loading, and rendering services."""

from __future__ import annotations

import math
import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from integrations.datalab_sia_client import DatalabSiaClient
from services.plotting import PlottingService


@dataclass
class FitsImage:
    data: Any
    wcs: Any
    header: Any
    path: str
    source_url: str


class DatalabImageService:
    """Build image products from Data Lab SIA rows."""

    def __init__(self, *, sia_client: Optional[DatalabSiaClient] = None, plotting_service: Optional[PlottingService] = None):
        self.sia_client = sia_client or DatalabSiaClient()
        self.plotting_service = plotting_service or PlottingService()
        self.download_timeout = float(os.getenv("DATALAB_IMAGE_DOWNLOAD_TIMEOUT_SECONDS", "180"))

    def search(self, ra: float, dec: float, fov_deg: float, *, catalog: Optional[str] = None, endpoint: Optional[str] = None) -> Dict[str, Any]:
        return self.sia_client.search(ra, dec, fov_deg, catalog=catalog, endpoint=endpoint)

    def deepest_by_band(self, rows: Sequence[Mapping[str, Any]], bands: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        wanted = [str(b).strip().lower() for b in bands]
        best: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            if str(_row_get(row, "proctype") or "").strip().lower() != "stack":
                continue
            if str(_row_get(row, "prodtype") or "").strip().lower() != "image":
                continue
            exptime = _to_positive_float(_row_get(row, "exptime"))
            if exptime is None:
                continue
            bandpass = str(_row_get(row, "obs_bandpass") or "").strip().lower()
            for band in wanted:
                if not bandpass.startswith(band):
                    continue
                current = best.get(band)
                if current is None or exptime > float(current["exptime"]):
                    best[band] = {"row": dict(row), "exptime": exptime}
        return best

    def cutout(
        self,
        ra: float,
        dec: float,
        fov_deg: float,
        *,
        band: str = "g",
        catalog: Optional[str] = None,
        endpoint: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        search = self.search(ra, dec, fov_deg, catalog=catalog, endpoint=endpoint)
        if search.get("coverage_gap"):
            return self._gap_result(search, bands=[band], provenance_extra={"ra": ra, "dec": dec, "fov_deg": fov_deg})
        chosen = self.deepest_by_band(search.get("rows") or [], [band])
        if str(band).lower() not in chosen:
            return self._gap_result(search, bands=[], provenance_extra={"missing_band": band})

        row = chosen[str(band).lower()]["row"]
        image = self._load_image(row, ra=ra, dec=dec, fov_deg=fov_deg)
        cut = self._cutout_image(image, ra=ra, dec=dec, fov_deg=fov_deg)
        render = self._render_single_band(cut.data, cut.wcs, title or f"Data Lab {band}-band cutout")
        self._cleanup_paths([image.path])
        return self._image_result(
            render,
            search,
            bands=[str(band).lower()],
            provenance_extra={"selected_rows": {str(band).lower(): self._row_provenance(row)}, "source_url": image.source_url},
        )

    def color_image(
        self,
        ra: float,
        dec: float,
        fov_deg: float,
        *,
        catalog: Optional[str] = None,
        endpoint: Optional[str] = None,
        bands: Optional[Sequence[str]] = None,
        q: float = 8.0,
        stretch: float = 0.5,
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        search = self.search(ra, dec, fov_deg, catalog=catalog, endpoint=endpoint)
        if search.get("coverage_gap"):
            return self._gap_result(search, bands=[], provenance_extra={"ra": ra, "dec": dec, "fov_deg": fov_deg})
        rows = search.get("rows") or []
        # RGB band selection (red, green, blue). Default: blue=g, green=r, and red=i
        # when present else z — LS DR9's registered bands are g/r/z, so i is often
        # absent and a fixed i/r/g triplet would wrongly report a coverage gap.
        if bands:
            rgb_bands = [str(b).strip().lower() for b in bands]
            if len(rgb_bands) != 3:
                raise ValueError("bands must be a 3-item sequence (red, green, blue)")
        else:
            avail = self.deepest_by_band(rows, ["g", "r", "i", "z"])
            red = "i" if "i" in avail else ("z" if "z" in avail else "i")
            rgb_bands = [red, "r", "g"]
        red_b, green_b, blue_b = rgb_bands
        chosen = self.deepest_by_band(rows, rgb_bands)
        missing = [band for band in rgb_bands if band not in chosen]
        if missing:
            return self._gap_result(search, bands=sorted(chosen), provenance_extra={"missing_bands": missing, "requested_bands": rgb_bands})

        cutouts: Dict[str, FitsImage] = {}
        for band in rgb_bands:
            image = self._load_image(chosen[band]["row"], ra=ra, dec=dec, fov_deg=fov_deg)
            cutouts[band] = self._cutout_image(image, ra=ra, dec=dec, fov_deg=fov_deg)

        try:
            from reproject import reproject_interp
        except ImportError as exc:
            raise ImportError("reproject is required for Data Lab color images") from exc

        ref = cutouts[green_b]
        reproj: Dict[str, Any] = {}
        for band in rgb_bands:
            arr, _ = reproject_interp((cutouts[band].data, cutouts[band].wcs), ref.wcs, shape_out=ref.data.shape)
            reproj[band] = arr

        try:
            from astropy.visualization import make_lupton_rgb
        except ImportError as exc:
            raise ImportError("astropy.visualization is required for Data Lab color images") from exc

        rgb = make_lupton_rgb(reproj[red_b], reproj[green_b], reproj[blue_b], Q=q, stretch=stretch)
        render = self._render_rgb(rgb, ref.wcs, title or f"Data Lab {blue_b}{green_b}{red_b} color image")
        self._cleanup_paths([img.path for img in cutouts.values()])
        return self._image_result(
            render,
            search,
            bands=list(rgb_bands),
            provenance_extra={
                "selected_rows": {band: self._row_provenance(chosen[band]["row"]) for band in rgb_bands},
                "reprojected_to": green_b,
                "lupton_rgb_order": f"{red_b},{green_b},{blue_b}",
            },
        )

    def cutout_grid(
        self,
        peaks: Sequence[Any],
        fov_deg: float,
        *,
        band: str = "g",
        catalog: Optional[str] = None,
        endpoint: Optional[str] = None,
        title: str = "Data Lab cutout grid",
    ) -> Dict[str, Any]:
        parsed = [_parse_peak(item) for item in peaks]
        if not parsed:
            raise ValueError("cutout_grid requires at least one peak")

        plt = self.plotting_service._apply_style(dark=False)
        from astropy.visualization import AsinhStretch, ImageNormalize, ZScaleInterval

        n = len(parsed)
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows), squeeze=False)
        panels: List[Dict[str, Any]] = []
        any_image = False

        for idx, peak in enumerate(parsed):
            ax = axes[idx // cols][idx % cols]
            label = peak.get("label") or f"{peak['ra']:.4f}, {peak['dec']:.4f}"
            panel = {"label": label, "ra": peak["ra"], "dec": peak["dec"], "coverage_gap": False}
            try:
                search = self.search(peak["ra"], peak["dec"], fov_deg, catalog=catalog, endpoint=endpoint)
                chosen = self.deepest_by_band(search.get("rows") or [], [band]) if not search.get("coverage_gap") else {}
                if search.get("coverage_gap") or str(band).lower() not in chosen:
                    panel["coverage_gap"] = True
                    ax.text(0.5, 0.5, "No coverage", ha="center", va="center", transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    image = self._load_image(chosen[str(band).lower()]["row"], ra=peak["ra"], dec=peak["dec"], fov_deg=fov_deg)
                    cut = self._cutout_image(image, ra=peak["ra"], dec=peak["dec"], fov_deg=fov_deg)
                    norm = ImageNormalize(cut.data, interval=ZScaleInterval(), stretch=AsinhStretch())
                    ax.imshow(cut.data, origin="lower", cmap="gray", norm=norm)
                    any_image = True
                    panel["selected_row"] = self._row_provenance(chosen[str(band).lower()]["row"])
                    self._cleanup_paths([image.path])
                panel["used_endpoint"] = search.get("used_endpoint")
            except Exception as exc:  # noqa: BLE001 - one bad panel should not kill the grid
                panel["coverage_gap"] = True
                panel["error"] = str(exc)
                ax.text(0.5, 0.5, "No coverage", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
            ax.set_title(str(label), fontsize=9)
            panels.append(panel)

        for idx in range(n, rows * cols):
            axes[idx // cols][idx % cols].axis("off")
        fig.suptitle(title)
        fig.tight_layout()
        render = self.plotting_service._save_and_encode(fig, f"datalab_cutout_grid_{uuid.uuid4().hex[:10]}")
        return {
            "success": True,
            "image_base64": render.get("base64_png"),
            "path": render.get("web_url"),
            "png_path": render.get("png_path"),
            "used_endpoint": endpoint,
            "bands_used": [str(band).lower()] if any_image else [],
            "coverage_gap": not any_image,
            "panels": panels,
            "provenance": {"fov_deg": fov_deg, "band": band, "catalog": catalog},
        }

    def _load_image(self, row: Mapping[str, Any], *, ra: float, dec: float, fov_deg: float) -> FitsImage:
        from astropy.io import fits as afits
        from astropy.wcs import WCS
        from services import fits_service

        url = _url_from_row(row)
        if not url:
            raise ValueError("SIA image row has no FITS access URL")
        path = fits_service._download_fits(url, label="Data Lab SIA image")
        try:
            try:
                data, header = afits.getdata(path, header=True)
                if data is None:
                    raise ValueError("primary HDU has no data")
                while getattr(data, "ndim", 0) > 2:
                    data = data[data.shape[0] // 2]
            except Exception:
                with afits.open(path, memmap=False) as hdul:
                    data, hdu_idx = fits_service._pick_science_hdu(hdul)
                    header = hdul[hdu_idx].header.copy()
            wcs = WCS(header, naxis=2)
            return FitsImage(data=np.asarray(data, dtype=float), wcs=wcs, header=header, path=path, source_url=url)
        except Exception:
            try:
                os.unlink(path)
            except OSError:
                pass
            raise

    def _cutout_image(self, image: FitsImage, *, ra: float, dec: float, fov_deg: float) -> FitsImage:
        try:
            from astropy.coordinates import SkyCoord
            from astropy.nddata import Cutout2D
            import astropy.units as u

            pos = SkyCoord(ra=float(ra) * u.deg, dec=float(dec) * u.deg, frame="icrs")
            cut = Cutout2D(image.data, pos, (float(fov_deg) * u.deg, float(fov_deg) * u.deg), wcs=image.wcs, mode="trim")
            return FitsImage(data=np.asarray(cut.data, dtype=float), wcs=cut.wcs, header=image.header, path=image.path, source_url=image.source_url)
        except Exception:
            return image

    def _render_single_band(self, data: Any, wcs: Any, title: str) -> Dict[str, Any]:
        plt = self.plotting_service._apply_style(dark=False)
        from astropy.visualization import AsinhStretch, ImageNormalize, ZScaleInterval

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection=wcs) if wcs is not None else fig.add_subplot(111)
        norm = ImageNormalize(data, interval=ZScaleInterval(), stretch=AsinhStretch())
        ax.imshow(data, origin="lower", cmap="gray", norm=norm)
        ax.set_title(title)
        fig.tight_layout()
        return self.plotting_service._save_and_encode(fig, f"datalab_cutout_{uuid.uuid4().hex[:10]}")

    def _render_rgb(self, rgb: Any, wcs: Any, title: str) -> Dict[str, Any]:
        plt = self.plotting_service._apply_style(dark=False)
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection=wcs) if wcs is not None else fig.add_subplot(111)
        ax.imshow(rgb, origin="lower")
        ax.set_title(title)
        fig.tight_layout()
        return self.plotting_service._save_and_encode(fig, f"datalab_color_{uuid.uuid4().hex[:10]}")

    @staticmethod
    def _cleanup_paths(paths: Iterable[str]) -> None:
        seen = set()
        for path in paths:
            if not path or path in seen:
                continue
            seen.add(path)
            try:
                os.unlink(path)
            except OSError:
                pass
    @staticmethod
    def _row_provenance(row: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "obs_bandpass": _row_get(row, "obs_bandpass"),
            "exptime": _row_get(row, "exptime"),
            "access_url": _url_from_row(row),
            "proctype": _row_get(row, "proctype"),
            "prodtype": _row_get(row, "prodtype"),
        }

    @staticmethod
    def _gap_result(search: Mapping[str, Any], *, bands: Sequence[str], provenance_extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        provenance = dict(search.get("provenance") or {})
        provenance.update(dict(provenance_extra or {}))
        return {
            "success": True,
            "image_base64": None,
            "path": None,
            "used_endpoint": search.get("used_endpoint"),
            "bands_used": list(bands),
            "coverage_gap": True,
            "provenance": provenance,
        }

    @staticmethod
    def _image_result(render: Mapping[str, Any], search: Mapping[str, Any], *, bands: Sequence[str], provenance_extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        provenance = dict(search.get("provenance") or {})
        provenance.update(dict(provenance_extra or {}))
        return {
            "success": True,
            "image_base64": render.get("base64_png"),
            "path": render.get("web_url"),
            "png_path": render.get("png_path"),
            "pdf_path": render.get("pdf_path"),
            "used_endpoint": search.get("used_endpoint"),
            "bands_used": list(bands),
            "coverage_gap": False,
            "provenance": provenance,
        }


def _row_get(row: Mapping[str, Any], name: str) -> Any:
    wanted = str(name).lower()
    for key, value in row.items():
        if str(key).lower() == wanted:
            return value
    return None


def _url_from_row(row: Mapping[str, Any]) -> Optional[str]:
    for name in ("access_url", "access", "url", "accref", "access_reference", "download_url"):
        value = _row_get(row, name)
        if value:
            return str(value)
    return None


def _to_positive_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number) or number <= 0:
        return None
    return number


def _parse_peak(item: Any) -> Dict[str, Any]:
    if isinstance(item, Mapping):
        ra = item.get("ra", item.get("ra_deg"))
        dec = item.get("dec", item.get("dec_deg"))
        label = item.get("label")
    else:
        seq = list(item)
        if len(seq) < 2:
            raise ValueError("Each peak must have at least ra and dec")
        ra, dec = seq[0], seq[1]
        label = seq[2] if len(seq) > 2 else None
    return {"ra": float(ra), "dec": float(dec), "label": label}


__all__ = ["DatalabImageService"]
