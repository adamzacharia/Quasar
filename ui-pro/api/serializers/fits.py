"""Matplotlib FITS-preview rendering, extracted verbatim from ``api/main.py``.

``render_fits_preview`` takes already-downloaded FITS ``bytes`` and produces the
workbench-ready preview payload (base64 PNG image + channel maps + PV slice +
spectrum + metadata + export scripts). The HTTP fetch and host allowlisting stay
in the router; this module owns only the (heavy) rendering. Heavy libraries
(matplotlib / numpy / astropy) are imported lazily inside the function so merely
importing this module stays cheap.
"""

from pathlib import Path
from typing import Optional

from fastapi import HTTPException


def render_fits_preview(
    fits_bytes: bytes,
    *,
    download_url: str,
    filename: Optional[str],
    parsed_path: str,
    hostname: Optional[str],
) -> dict:
    """Render a downloaded ALMA FITS product into workbench-ready previews.

    Mirrors the exact rendering behaviour of the former ``preview_fits`` endpoint
    body. Raises ``HTTPException(422)`` if the bytes cannot be rendered.
    """
    import base64
    import io
    import math

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    import numpy as np
    from astropy.io import fits
    from astropy.visualization import simple_norm

    try:
        def _finite_float(value):
            try:
                number = float(value)
                return number if math.isfinite(number) else None
            except (TypeError, ValueError):
                return None

        def _figure_data_url(fig) -> str:
            png_buf = io.BytesIO()
            fig.savefig(png_buf, format="png", bbox_inches="tight")
            plt.close(fig)
            return "data:image/png;base64," + base64.b64encode(png_buf.getvalue()).decode("ascii")

        def _robust_rms(values) -> Optional[float]:
            arr = np.asarray(values, dtype=float).ravel()
            if arr.size > 250000:
                step = int(math.ceil(arr.size / 250000))
                arr = arr[::step]
            arr = arr[np.isfinite(arr)]
            if arr.size < 10:
                return None
            median = float(np.nanmedian(arr))
            mad = float(np.nanmedian(np.abs(arr - median)))
            if mad > 0:
                return 1.4826 * mad
            std = float(np.nanstd(arr))
            return std if math.isfinite(std) and std > 0 else None

        def _safe_minmax(values) -> tuple[Optional[float], Optional[float]]:
            arr = np.asarray(values, dtype=float)
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                return None, None
            return float(np.nanmin(finite)), float(np.nanmax(finite))

        def _spectral_axis(header, nchan: int):
            crval = _finite_float(header.get("CRVAL3"))
            cdelt = _finite_float(header.get("CDELT3"))
            crpix = _finite_float(header.get("CRPIX3")) or 1.0
            ctype = str(header.get("CTYPE3") or "Channel").strip()
            cunit = str(header.get("CUNIT3") or "").strip()
            if crval is None or cdelt is None:
                return np.arange(nchan, dtype=float), "Channel", "channel", None

            axis = crval + ((np.arange(nchan, dtype=float) + 1.0 - crpix) * cdelt)
            unit_lower = cunit.lower()
            ctype_upper = ctype.upper()
            overlay_unit = cunit or ""
            if "hz" in unit_lower or ("freq" in ctype_upper and np.nanmedian(np.abs(axis)) > 1e5):
                if unit_lower == "ghz":
                    converted = axis
                elif unit_lower == "mhz":
                    converted = axis / 1000.0
                elif unit_lower == "khz":
                    converted = axis / 1e6
                else:
                    converted = axis / 1e9
                return converted, f"{ctype or 'Frequency'} (GHz)", "GHz", "frequency"
            if unit_lower in {"m/s", "ms-1", "meter/s", "metre/s"}:
                return axis / 1000.0, f"{ctype or 'Velocity'} (km/s)", "km/s", "velocity"
            if unit_lower in {"km/s", "kms-1"}:
                return axis, f"{ctype or 'Velocity'} (km/s)", "km/s", "velocity"
            return axis, f"{ctype or 'Axis 3'}{f' ({overlay_unit})' if overlay_unit else ''}", overlay_unit, None

        def _render_image(image, title: str, unit: str, header, rms_value: Optional[float], add_contours: bool):
            fig, ax = plt.subplots(figsize=(6.2, 6.0), dpi=150)
            try:
                norm = simple_norm(image, "asinh", percent=99.5)
            except Exception:
                norm = None
            im = ax.imshow(image, origin="lower", cmap="inferno", norm=norm)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Pixel")
            ax.set_ylabel("Pixel")
            contour_levels = []
            if add_contours and rms_value and rms_value > 0:
                image_max = _safe_minmax(image)[1]
                contour_levels = [
                    level for level in (3 * rms_value, 5 * rms_value, 10 * rms_value)
                    if image_max is not None and image_max > level
                ]
                if contour_levels:
                    ax.contour(image, levels=contour_levels, colors="white", linewidths=0.45, alpha=0.6)

            bmaj = _finite_float(header.get("BMAJ"))
            bmin = _finite_float(header.get("BMIN"))
            bpa = _finite_float(header.get("BPA")) or 0.0
            cdelt1 = abs(_finite_float(header.get("CDELT1")) or 0)
            cdelt2 = abs(_finite_float(header.get("CDELT2")) or 0)
            if bmaj and bmin and cdelt1 and cdelt2:
                ny, nx = image.shape[-2], image.shape[-1]
                width_px = max(1.0, (bmaj * 3600.0) / (cdelt1 * 3600.0))
                height_px = max(1.0, (bmin * 3600.0) / (cdelt2 * 3600.0))
                beam = Ellipse(
                    (max(10, nx * 0.12), max(10, ny * 0.1)),
                    width=width_px,
                    height=height_px,
                    angle=bpa,
                    facecolor="none",
                    edgecolor="white",
                    linewidth=1.0,
                    alpha=0.9,
                )
                ax.add_patch(beam)
                ax.text(max(10, nx * 0.12), max(6, ny * 0.1 - height_px * 0.8), "beam",
                        color="white", fontsize=6, ha="center", alpha=0.85)

            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(unit or "value", fontsize=8)
            fig.tight_layout()
            return _figure_data_url(fig), [float(level) for level in contour_levels]

        def _render_channel_maps(cube, spectral_values, spectral_label: str, unit: str):
            nchan = int(cube.shape[0])
            sample_count = min(9, nchan)
            indices = np.unique(np.linspace(0, nchan - 1, sample_count, dtype=int))
            cols = 3 if len(indices) > 2 else len(indices)
            rows_n = int(math.ceil(len(indices) / cols))
            fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 2.4, rows_n * 2.25), dpi=145)
            axes_arr = np.asarray(axes).reshape(-1)
            for ax, idx in zip(axes_arr, indices):
                plane = cube[int(idx)]
                try:
                    norm = simple_norm(plane, "asinh", percent=99.3)
                except Exception:
                    norm = None
                ax.imshow(plane, origin="lower", cmap="inferno", norm=norm)
                if spectral_values is not None and len(spectral_values) > int(idx):
                    ax.set_title(f"ch {int(idx)} | {float(spectral_values[int(idx)]):.4g}", fontsize=7)
                else:
                    ax.set_title(f"ch {int(idx)}", fontsize=7)
                ax.set_xticks([])
                ax.set_yticks([])
            for ax in axes_arr[len(indices):]:
                ax.axis("off")
            fig.suptitle(f"Channel maps | {spectral_label}", fontsize=9)
            fig.text(0.5, 0.02, unit or "image value", ha="center", fontsize=7)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            return _figure_data_url(fig)

        def _render_pv_slice(cube, y_index: int, spectral_label: str, unit: str):
            pv = cube[:, int(y_index), :]
            fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)
            try:
                norm = simple_norm(pv, "asinh", percent=99.5)
            except Exception:
                norm = None
            im = ax.imshow(pv, origin="lower", aspect="auto", cmap="magma", norm=norm)
            ax.set_title("Central PV slice through peak row", fontsize=10)
            ax.set_xlabel("Spatial pixel X")
            ax.set_ylabel(spectral_label)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(unit or "value", fontsize=8)
            fig.tight_layout()
            return _figure_data_url(fig)

        with fits.open(io.BytesIO(fits_bytes), memmap=False) as hdul:
            hdu = next((item for item in hdul if getattr(item, "data", None) is not None), None)
            if hdu is None:
                raise ValueError("No image data found in FITS file.")
            raw_data = np.asarray(hdu.data, dtype=float)
            header = hdu.header

            squeezed = np.squeeze(raw_data)
            if squeezed.ndim < 2:
                raise ValueError("FITS data is not image-like.")
            cube = None
            cube_shape = None
            if squeezed.ndim == 2:
                image = squeezed
                primary_label = "Image preview with beam/contour overlay"
            else:
                if squeezed.ndim > 3:
                    squeezed = squeezed.reshape((-1, squeezed.shape[-2], squeezed.shape[-1]))
                cube = squeezed
                cube_shape = list(cube.shape)
                image = np.nansum(cube, axis=0)
                primary_label = "Moment-0 style cube collapse with beam/contour overlay"

            if not np.isfinite(image).any():
                raise ValueError("Image data contains no finite values.")

            object_name = str(header.get("OBJECT") or filename or "ALMA FITS")
            unit = str(header.get("BUNIT") or "")
            rms_value = _robust_rms(cube if cube is not None else image)
            image_data_url, contour_levels = _render_image(
                image,
                f"{object_name} | {unit}".strip(" |"),
                unit,
                header,
                rms_value,
                add_contours=True,
            )

            bmaj = _finite_float(header.get("BMAJ"))
            bmin = _finite_float(header.get("BMIN"))
            bpa = _finite_float(header.get("BPA"))
            restfreq = _finite_float(header.get("RESTFRQ"))
            cdelt1 = abs(_finite_float(header.get("CDELT1")) or 0) or None
            cdelt2 = abs(_finite_float(header.get("CDELT2")) or 0) or None
            min_value, max_value = _safe_minmax(image)

            spectral_values = None
            spectral_label = "Channel"
            spectral_unit = "channel"
            spectral_kind = None
            spectrum_payload = None
            channel_maps_data_url = None
            pv_slice_data_url = None
            line_overlays = []
            assumptions = [
                "Preview renders a downloaded copy limited to 50 MB.",
                "RMS is a robust MAD estimate from previewed pixel values.",
            ]
            confidence = "medium"

            if cube is not None:
                nchan = int(cube.shape[0])
                spectral_values, spectral_label, spectral_unit, spectral_kind = _spectral_axis(header, nchan)
                assumptions.append("For cube-like FITS files, the first non-spatial axis after squeezing is treated as the channel/spectral axis.")

                peak_source = np.asarray(image, dtype=float)
                if np.isfinite(peak_source).any():
                    peak_y, peak_x = np.unravel_index(np.nanargmax(np.abs(peak_source)), peak_source.shape)
                    spectrum_values = cube[:, int(peak_y), int(peak_x)]
                    extraction = f"peak pixel x={int(peak_x)}, y={int(peak_y)}"
                else:
                    peak_y = cube.shape[1] // 2
                    peak_x = cube.shape[2] // 2
                    spectrum_values = np.nanmedian(cube, axis=(1, 2))
                    extraction = "spatial median"

                finite_spec = np.isfinite(spectrum_values)
                if finite_spec.any():
                    max_points = 360
                    step = max(1, int(math.ceil(len(spectrum_values) / max_points)))
                    xs = spectral_values[::step]
                    ys = spectrum_values[::step]
                    spec_pairs = [
                        (float(x_val), float(y_val))
                        for x_val, y_val in zip(xs, ys)
                        if math.isfinite(float(x_val)) and math.isfinite(float(y_val))
                    ]
                    spectrum_payload = {
                        "x": [round(x_val, 8) for x_val, _ in spec_pairs],
                        "y": [round(y_val, 8) for _, y_val in spec_pairs],
                        "xLabel": spectral_label,
                        "yLabel": unit or "Value",
                        "extraction": extraction,
                    }
                    confidence = "high" if restfreq or str(header.get("CTYPE3") or "").strip() else "medium"

                channel_maps_data_url = _render_channel_maps(cube, spectral_values, spectral_label, unit)
                pv_slice_data_url = _render_pv_slice(cube, int(peak_y), spectral_label, unit)

                if restfreq and spectral_kind == "frequency":
                    rest_ghz = restfreq / 1e9
                    axis_min, axis_max = float(np.nanmin(spectral_values)), float(np.nanmax(spectral_values))
                    line_overlays.append({
                        "label": "Header RESTFRQ",
                        "value": round(rest_ghz, 8),
                        "unit": "GHz",
                        "source": "FITS header",
                        "inRange": bool(min(axis_min, axis_max) <= rest_ghz <= max(axis_min, axis_max)),
                    })

            if bmaj and bmin and cdelt1 and cdelt2:
                confidence = "high" if confidence == "medium" else confidence

            resolved_filename = filename or Path(parsed_path).name or "alma_product.fits"
            casa_script = (
                f"from casatasks import importfits\n"
                f"importfits(fitsimage='{resolved_filename}', imagename='{Path(resolved_filename).stem}.image', overwrite=True)\n"
                f"# Then inspect in CASA viewer or run imstat/imhead on {Path(resolved_filename).stem}.image"
            )
            if cube is not None:
                casa_script += "\n# For cube products, use immoments or specfit after checking the spectral axis."

            metadata = {
                "object": object_name,
                "unit": unit,
                "shape": list(image.shape),
                "rawShape": list(raw_data.shape),
                "cubeShape": cube_shape,
                "isCube": cube is not None,
                "channelCount": int(cube.shape[0]) if cube is not None else None,
                "beamMajorArcsec": bmaj * 3600 if bmaj is not None else None,
                "beamMinorArcsec": bmin * 3600 if bmin is not None else None,
                "beamPaDeg": bpa,
                "restFreqGhz": restfreq / 1e9 if restfreq is not None else None,
                "pixelScaleArcsec": {
                    "x": cdelt1 * 3600 if cdelt1 is not None else None,
                    "y": cdelt2 * 3600 if cdelt2 is not None else None,
                },
                "spectralAxisLabel": spectral_label if cube is not None else None,
                "rms": rms_value,
                "rmsUnit": unit or None,
                "min": min_value,
                "max": max_value,
                "sizeBytes": len(fits_bytes),
            }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not render FITS preview: {exc}")

    return {
        "imageDataUrl": image_data_url,
        "metadata": metadata,
        "downloadUrl": download_url,
        "filename": resolved_filename,
        "workbench": {
            "kind": "cube" if metadata.get("isCube") else "image",
            "primaryImageLabel": primary_label,
            "channelMapsDataUrl": channel_maps_data_url,
            "pvSliceDataUrl": pv_slice_data_url,
            "spectrum": spectrum_payload,
            "rms": rms_value,
            "rmsUnit": unit or None,
            "contourLevels": [round(float(level), 8) for level in contour_levels],
            "lineOverlays": line_overlays,
            "exports": [
                {"label": "CASA", "command": casa_script},
                {"label": "CARTA", "command": f"carta '{resolved_filename}'"},
                {"label": "DS9", "command": f"ds9 '{resolved_filename}' -scale zscale -cmap inferno"},
            ],
            "evidence": {
                "dataAccess": f"Fetched {round(len(fits_bytes) / (1024 * 1024), 2)} MB from {hostname}",
                "headersInspected": [
                    key for key in ["OBJECT", "BUNIT", "BMAJ", "BMIN", "BPA", "RESTFRQ", "CTYPE3", "CUNIT3", "CRVAL3", "CDELT3"]
                    if header.get(key) is not None
                ],
                "assumptions": assumptions,
                "confidence": confidence,
            },
        },
        "suggestedActions": [
            "Measure peak flux and RMS noise",
            "Inspect channel maps and the spectrum before choosing line windows",
            "Export/open the FITS in CASA, CARTA, DS9, or Python",
        ],
    }
