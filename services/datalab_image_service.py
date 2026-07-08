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
        candidates = self.candidates_by_band(rows, bands)
        return {band: items[0] for band, items in candidates.items() if items}

    def candidates_by_band(
        self,
        rows: Sequence[Mapping[str, Any]],
        bands: Iterable[str],
        *,
        max_candidates: int = 4,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Per band, ALL matching Stack/image rows sorted deepest-first (capped).

        coadd_all is cross-survey and some rows carry broken cutout refs (observed: Local Group
        Survey tiles with an empty ``col=`` in access_url that 500 server-side). Callers should
        try candidates in order and fall through on download failure rather than dying on the
        single deepest row.
        """
        wanted = [str(b).strip().lower() for b in bands]
        found: Dict[str, List[Dict[str, Any]]] = {b: [] for b in wanted}
        for row in rows:
            if str(_row_get(row, "proctype") or "").strip().lower() != "stack":
                continue
            if str(_row_get(row, "prodtype") or "").strip().lower() != "image":
                continue
            # Missing exptime does NOT exclude a row: Legacy Surveys coadd rows often carry
            # no exptime, and dropping them left only broken LGS/Mosaic refs at e.g. M31.
            # Treat missing depth as 0 so these rank last but remain usable fallbacks.
            exptime = _to_positive_float(_row_get(row, "exptime")) or 0.0
            bandpass = str(_row_get(row, "obs_bandpass") or "").strip().lower()
            for band in wanted:
                if bandpass.startswith(band):
                    found[band].append({"row": dict(row), "exptime": exptime})
        def _rank(item: Dict[str, Any]):
            # Deprioritize rows whose cutout URL has an empty col= (known-broken pattern),
            # then prefer greater depth.
            url = str(_row_get(item["row"], "access_url") or "")
            broken_hint = 1 if "col=&" in url or url.endswith("col=") else 0
            return (broken_hint, -float(item["exptime"]))
        return {b: sorted(items, key=_rank)[: max_candidates] for b, items in found.items() if items}

    def _bands_with_healthy_refs(self, rows: Sequence[Mapping[str, Any]], *, exclude: str = "") -> List[str]:
        """Bands (g/r/i/z/u/y) that have at least one candidate with a non-broken cutout ref."""
        out: List[str] = []
        common = [b for b in ("g", "r", "i", "z", "u", "y") if b != str(exclude).lower()]
        for band, cands in self.candidates_by_band(rows, common).items():
            for cand in cands:
                url = str(_row_get(cand["row"], "access_url") or "")
                if "col=&" not in url and not url.endswith("col="):
                    out.append(band)
                    break
        return sorted(out)

    def _load_first_working(
        self,
        candidates: Sequence[Dict[str, Any]],
        *,
        ra: float,
        dec: float,
        fov_deg: float,
    ):
        """Try candidate rows deepest-first; return (row, FitsImage, errors). All-fail -> (None, None, errors)."""
        errors: List[str] = []
        for cand in candidates:
            row = cand["row"]
            try:
                image = self._load_image(row, ra=ra, dec=dec, fov_deg=fov_deg)
                # A tile can download with a structurally valid FITS yet be an
                # all-zero / all-NaN placeholder at THIS position (observed at M31
                # in coadd_all). That renders as a solid black image because ZScale
                # collapses to vmin==vmax. Treat a degenerate cutout region like a
                # failed tile and fall through to the next candidate.
                probe = self._cutout_image(image, ra=ra, dec=dec, fov_deg=fov_deg)
                if self._is_degenerate(probe.data):
                    self._cleanup_paths([image.path])
                    errors.append(
                        f"{str(_row_get(row, 'access_url'))[:120]} -> empty/degenerate tile "
                        "(all-NaN or single-valued at this position)"
                    )
                    continue
                return row, image, errors
            except Exception as exc:  # noqa: BLE001 - fall through to the next candidate tile
                errors.append(f"{str(_row_get(row, 'access_url'))[:120]} -> {exc}")
        return None, None, errors

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
        band_key = str(band).lower()
        candidates = self.candidates_by_band(search.get("rows") or [], [band_key]).get(band_key) or []
        if not candidates:
            # Tell the model which bands DO have healthy tiles so it can retry instead of
            # giving up (e.g. M31 has no g/r/i but a working MzLS z tile).
            suggested = self._bands_with_healthy_refs(search.get("rows") or [], exclude=band_key)
            gap = self._gap_result(search, bands=suggested, provenance_extra={"missing_band": band})
            gap["suggested_bands"] = suggested
            if suggested:
                gap["note"] = (
                    f"No {band_key}-band coverage at this position, but band(s) {suggested} have usable "
                    f"tiles here — RETRY datalab_image_cutout with band='{suggested[0]}' instead of "
                    "reporting failure. Do not describe an image that was not rendered."
                )
            return gap

        # Try candidates deepest-first: individual coadd_all tiles can 500 on the on-demand
        # cutout service (observed for Local Group Survey refs at M31) — fall through instead
        # of failing the whole tool on one broken tile.
        row, image, dl_errors = self._load_first_working(candidates, ra=ra, dec=dec, fov_deg=fov_deg)
        if image is None:
            suggested = self._bands_with_healthy_refs(search.get("rows") or [], exclude=band_key)
            hint = (
                f" Bands with healthy-looking tiles at this position: {suggested} — retry with one of those."
                if suggested else " No other band shows healthy tile refs here — suggest a different survey (e.g. get_sky_image)."
            )
            return {
                "success": False,
                "coverage_gap": False,
                "image_base64": None,
                "path": None,
                "bands_used": [],
                "error": (
                    f"All {len(candidates)} matching {band_key}-band tiles failed to download from the "
                    "Data Lab cutout service (server-side errors on those tile refs). This is a service/"
                    "tile issue, not missing coverage." + hint
                ),
                "suggested_bands": suggested,
                "download_errors": dl_errors,
                "used_endpoint": search.get("used_endpoint"),
                "provenance": dict(search.get("provenance") or {}),
            }
        cut = self._cutout_image(image, ra=ra, dec=dec, fov_deg=fov_deg)
        render = self._render_single_band(cut.data, cut.wcs, title or f"Data Lab {band}-band cutout")
        self._cleanup_paths([image.path])
        extra = {"selected_rows": {band_key: self._row_provenance(row)}, "source_url": image.source_url}
        if dl_errors:
            extra["skipped_broken_tiles"] = dl_errors
        return self._image_result(render, search, bands=[band_key], provenance_extra=extra)

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
        # RGB band selection (red, green, blue). Choose from bands that have a
        # NON-broken cutout ref: coadd_all indexes broken cross-survey tiles
        # (e.g. Local Group Survey i/r refs at M31 that 500 with an empty col=),
        # and advertising those as "available" made the model chase dead bands or
        # refuse. LS DR9's real bands are g/r/z (NO i), so we order by wavelength
        # and build blue->red from the healthy set.
        _WL_ORDER = ["u", "g", "r", "i", "z", "y"]  # blue -> red
        if bands:
            rgb_bands = [str(b).strip().lower() for b in bands]
            if len(rgb_bands) != 3:
                raise ValueError("bands must be a 3-item sequence (red, green, blue)")
        else:
            healthy = [b for b in _WL_ORDER if b in set(self._bands_with_healthy_refs(rows))]
            if len(healthy) < 3:
                # Fewer than 3 healthy bands -> a true 3-color composite is impossible
                # here (e.g. only z is healthy at the exact M31 center). Report the
                # honest healthy-band list so the model can offer a single-band cutout,
                # a slightly offset/wider field, or a HiPS/DSS2 optical color panel.
                return self._gap_result(
                    search, bands=healthy,
                    provenance_extra={"healthy_bands": healthy, "reason": "fewer than 3 bands with usable tiles"},
                )
            blue_b, red_b = healthy[0], healthy[-1]
            green_b = healthy[len(healthy) // 2]
            rgb_bands = [red_b, green_b, blue_b]
        red_b, green_b, blue_b = rgb_bands
        band_candidates = self.candidates_by_band(rows, rgb_bands)
        missing = [band for band in rgb_bands if not band_candidates.get(band)]
        if missing:
            return self._gap_result(search, bands=sorted(band_candidates), provenance_extra={"missing_bands": missing, "requested_bands": rgb_bands})

        cutouts: Dict[str, FitsImage] = {}
        chosen: Dict[str, Dict[str, Any]] = {}
        for band in rgb_bands:
            # Fall through broken tile refs per band (see cutout()).
            row, image, dl_errors = self._load_first_working(band_candidates[band], ra=ra, dec=dec, fov_deg=fov_deg)
            if image is None:
                return {
                    "success": False,
                    "coverage_gap": False,
                    "image_base64": None,
                    "path": None,
                    "bands_used": [],
                    "error": (
                        f"All matching {band}-band tiles failed to download from the Data Lab cutout "
                        "service (server-side tile errors) — cannot build the color composite."
                    ),
                    "download_errors": dl_errors,
                    "used_endpoint": search.get("used_endpoint"),
                    "provenance": dict(search.get("provenance") or {}),
                }
            chosen[band] = {"row": row}
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
                    if self._is_degenerate(cut.data):
                        self._cleanup_paths([image.path])
                        raise ValueError("empty/degenerate tile at this peak (all-NaN or constant)")
                    finite = np.asarray(cut.data, dtype=float)
                    finite = finite[np.isfinite(finite)]
                    norm = ImageNormalize(finite, interval=ZScaleInterval(), stretch=AsinhStretch())
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

    @staticmethod
    def _is_degenerate(data: Any) -> bool:
        """True when the array has no usable dynamic range — all-NaN, empty, or a
        single constant value. Such a tile renders as a solid black image because
        ZScale collapses to vmin==vmax; callers should skip it."""
        arr = np.asarray(data, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return True
        return float(np.max(finite)) == float(np.min(finite))

    def _render_single_band(self, data: Any, wcs: Any, title: str) -> Dict[str, Any]:
        plt = self.plotting_service._apply_style(dark=False)
        from astropy.visualization import AsinhStretch, ImageNormalize, ZScaleInterval

        # Backstop: never render a degenerate tile as a (misleading) black image.
        if self._is_degenerate(data):
            raise ValueError("cutout tile has no dynamic range (empty/all-NaN/constant) — not a valid image")
        arr = np.asarray(data, dtype=float)
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection=wcs) if wcs is not None else fig.add_subplot(111)
        finite = arr[np.isfinite(arr)]
        norm = ImageNormalize(finite, interval=ZScaleInterval(), stretch=AsinhStretch())
        ax.imshow(arr, origin="lower", cmap="gray", norm=norm)
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
        missing = list((provenance_extra or {}).get("missing_bands") or [])
        retry = ""
        if bands and len(bands) >= 3:
            retry = (
                f" However, band(s) {list(bands)} DO have usable tiles here — call "
                "datalab_color_image (it auto-selects an available g/r/z triplet; LS DR9 uses "
                "g/r/z, NOT i) to build the composite from those before reporting a gap."
            )
        elif bands:
            retry = (
                f" Band(s) {list(bands)} have usable tiles here — RETRY datalab_image_cutout with "
                f"band='{list(bands)[0]}' (a color image needs 3 bands, unavailable here)."
            )
        note = (
            "No image produced (coverage gap): insufficient coverage for the requested band(s) at this position"
            + (f" — missing band(s) {missing}; only {list(bands)} available." if missing
               else f" — only {list(bands)} band(s) available." if bands
               else " — the SIA search returned 0 rows.")
            + retry
            + " Only report a coverage gap if no retry above is possible; NEVER describe an image or"
            " claim bands were rendered when none were."
        )
        return {
            "success": True,
            "image_base64": None,
            "path": None,
            "used_endpoint": search.get("used_endpoint"),
            "bands_used": [],            # no image was rendered, so no bands were used
            "available_bands": list(bands),
            "coverage_gap": True,
            "note": note,
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
