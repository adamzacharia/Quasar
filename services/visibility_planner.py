# services/visibility_planner.py
"""
Target Visibility Planner (FEATURES.md O2).

Given an observing site, a UT date, and one or more targets, computes
altitude curves across the night, astronomical darkness windows, moon
separation, and per-target observability summaries, and renders a
publication-style altitude plot.

Implemented with pure astropy (no astroplan dependency):
  - Site resolution: built-in observatory catalog -> astropy site registry
  - Target resolution: explicit RA/Dec or SIMBAD name resolution
  - Sun/Moon ephemerides: astropy.coordinates.get_sun / get_body

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

# ── Built-in observatory catalog (lon_deg_east, lat_deg, height_m) ──────
# Offline-safe: avoids astropy's downloadable site registry for the most
# common facilities. Keys are upper-cased aliases.
OBSERVATORY_SITES: Dict[str, Dict[str, Any]] = {
    "ALMA":          {"lon": -67.7550, "lat": -23.0292, "height": 5058.7, "name": "ALMA (Chajnantor)"},
    "VLA":           {"lon": -107.6184, "lat": 34.0784,  "height": 2124.0, "name": "Very Large Array"},
    "GBT":           {"lon": -79.8398, "lat": 38.4331,  "height": 807.0,  "name": "Green Bank Telescope"},
    "GREEN BANK":    {"lon": -79.8398, "lat": 38.4331,  "height": 807.0,  "name": "Green Bank Telescope"},
    "MAUNA KEA":     {"lon": -155.4681, "lat": 19.8208, "height": 4205.0, "name": "Mauna Kea"},
    "KECK":          {"lon": -155.4747, "lat": 19.8260, "height": 4159.6, "name": "Keck Observatory"},
    "SUBARU":        {"lon": -155.4761, "lat": 19.8255, "height": 4163.0, "name": "Subaru Telescope"},
    "PARANAL":       {"lon": -70.4042, "lat": -24.6272, "height": 2635.4, "name": "Paranal (VLT)"},
    "VLT":           {"lon": -70.4042, "lat": -24.6272, "height": 2635.4, "name": "Paranal (VLT)"},
    "LA SILLA":      {"lon": -70.7375, "lat": -29.2567, "height": 2347.0, "name": "La Silla"},
    "CTIO":          {"lon": -70.8150, "lat": -30.1653, "height": 2215.0, "name": "Cerro Tololo (CTIO)"},
    "LAS CAMPANAS":  {"lon": -70.7017, "lat": -29.0083, "height": 2380.0, "name": "Las Campanas"},
    "KITT PEAK":     {"lon": -111.6003, "lat": 31.9583, "height": 2120.0, "name": "Kitt Peak"},
    "PALOMAR":       {"lon": -116.8650, "lat": 33.3564, "height": 1712.0, "name": "Palomar"},
    "APACHE POINT":  {"lon": -105.8203, "lat": 32.7803, "height": 2788.0, "name": "Apache Point"},
    "EFFELSBERG":    {"lon": 6.8836,   "lat": 50.5247,  "height": 319.0,  "name": "Effelsberg 100m"},
    "PARKES":        {"lon": 148.2636, "lat": -32.9980, "height": 414.8,  "name": "Parkes (Murriyang)"},
    "MEERKAT":       {"lon": 21.4431,  "lat": -30.7130, "height": 1086.6, "name": "MeerKAT"},
    "ASKAP":         {"lon": 116.6314, "lat": -26.6970, "height": 377.0,  "name": "ASKAP"},
    "ATCA":          {"lon": 149.5501, "lat": -30.3128, "height": 237.0,  "name": "ATCA (Narrabri)"},
    "NOEMA":         {"lon": 5.9069,   "lat": 44.6339,  "height": 2552.0, "name": "NOEMA (Plateau de Bure)"},
    "IRAM 30M":      {"lon": -3.3925,  "lat": 37.0662,  "height": 2850.0, "name": "IRAM 30m (Pico Veleta)"},
    "PICO VELETA":   {"lon": -3.3925,  "lat": 37.0662,  "height": 2850.0, "name": "IRAM 30m (Pico Veleta)"},
    "SMA":           {"lon": -155.4775, "lat": 19.8243, "height": 4080.0, "name": "Submillimeter Array"},
    "JCMT":          {"lon": -155.4770, "lat": 19.8228, "height": 4120.0, "name": "JCMT"},
    "LOFAR":         {"lon": 6.8670,   "lat": 52.9089,  "height": 6.0,    "name": "LOFAR core"},
    "GMRT":          {"lon": 74.0497,  "lat": 19.0931,  "height": 650.0,  "name": "GMRT"},
    "FAST":          {"lon": 106.8567, "lat": 25.6529,  "height": 1110.0, "name": "FAST 500m"},
    "ROQUE":         {"lon": -17.8792, "lat": 28.7606,  "height": 2396.0, "name": "Roque de los Muchachos"},
    "LA PALMA":      {"lon": -17.8792, "lat": 28.7606,  "height": 2396.0, "name": "Roque de los Muchachos"},
    "GTC":           {"lon": -17.8920, "lat": 28.7567,  "height": 2267.0, "name": "Gran Telescopio Canarias"},
    "CALAR ALTO":    {"lon": -2.5461,  "lat": 37.2236,  "height": 2168.0, "name": "Calar Alto"},
    "SIDING SPRING": {"lon": 149.0661, "lat": -31.2733, "height": 1164.0, "name": "Siding Spring"},
    "MCDONALD":      {"lon": -104.0247, "lat": 30.6717, "height": 2075.0, "name": "McDonald Observatory"},
    "LICK":          {"lon": -121.6428, "lat": 37.3414, "height": 1283.0, "name": "Lick Observatory"},
    "GEMINI NORTH":  {"lon": -155.4690, "lat": 19.8238, "height": 4213.0, "name": "Gemini North"},
    "GEMINI SOUTH":  {"lon": -70.7234, "lat": -30.2408, "height": 2722.0, "name": "Gemini South"},
    "SALT":          {"lon": 20.8107,  "lat": -32.3760, "height": 1798.0, "name": "SALT (Sutherland)"},
    "VERA RUBIN":    {"lon": -70.7494, "lat": -30.2446, "height": 2663.0, "name": "Vera C. Rubin Observatory"},
    "RUBIN":         {"lon": -70.7494, "lat": -30.2446, "height": 2663.0, "name": "Vera C. Rubin Observatory"},
}


def resolve_site(site: str = None,
                 latitude_deg: float = None,
                 longitude_deg: float = None,
                 height_m: float = 0.0):
    """
    Resolve an observing site to an astropy EarthLocation.

    Priority: explicit lat/lon -> built-in catalog -> astropy site registry.
    Returns (EarthLocation, display_name). Raises ValueError when unresolvable.
    """
    from astropy.coordinates import EarthLocation
    import astropy.units as u

    if latitude_deg is not None and longitude_deg is not None:
        loc = EarthLocation(lat=latitude_deg * u.deg, lon=longitude_deg * u.deg,
                            height=(height_m or 0.0) * u.m)
        label = site or f"lat={latitude_deg:.4f}, lon={longitude_deg:.4f}"
        return loc, label

    if not site:
        raise ValueError("Provide a site name or explicit latitude/longitude")

    key = site.strip().upper()
    if key in OBSERVATORY_SITES:
        info = OBSERVATORY_SITES[key]
        loc = EarthLocation(lat=info["lat"] * u.deg, lon=info["lon"] * u.deg,
                            height=info["height"] * u.m)
        return loc, info["name"]

    # Last resort: astropy's site registry (may need network on first use)
    try:
        loc = EarthLocation.of_site(site)
        return loc, site
    except Exception as e:
        known = ", ".join(sorted({v["name"] for v in OBSERVATORY_SITES.values()}))
        raise ValueError(
            f"Unknown site '{site}' ({e}). Provide latitude/longitude, or use one of: {known}"
        )


def _resolve_targets(targets: List[Dict[str, Any]]):
    """
    Resolve a list of target dicts to SkyCoords.

    Each entry: {"name": str} (SIMBAD lookup) or
                {"name": str, "ra": deg, "dec": deg} (explicit, offline).
    Returns (resolved list of (name, SkyCoord), list of failures)
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    resolved, failed = [], []
    for t in targets:
        name = str(t.get("name") or t.get("target") or "").strip()
        ra, dec = t.get("ra"), t.get("dec")
        try:
            if ra is not None and dec is not None:
                coord = SkyCoord(ra=float(ra) * u.deg, dec=float(dec) * u.deg, frame="icrs")
            elif name:
                coord = SkyCoord.from_name(name)
            else:
                raise ValueError("target needs a name or ra/dec")
            resolved.append((name or f"({ra:.3f}, {dec:.3f})", coord))
        except Exception as e:
            failed.append({"target": name or str(t), "error": str(e)})
    return resolved, failed


def plan_visibility(
    targets: List[Dict[str, Any]],
    date: str,
    site: str = None,
    latitude_deg: float = None,
    longitude_deg: float = None,
    height_m: float = 0.0,
    min_altitude_deg: float = 30.0,
    utc_offset_hours: float = 0.0,
) -> Dict[str, Any]:
    """
    Compute and plot target observability for one UT night.

    Args:
        targets: list of {"name": ...} and/or {"name", "ra", "dec"} dicts
        date: UT date "YYYY-MM-DD" (night spans date 12:00 UT -> date+1 12:00 UT)
        site: observatory name (see OBSERVATORY_SITES) — or pass lat/lon
        latitude_deg / longitude_deg / height_m: explicit site coordinates
        min_altitude_deg: observability threshold (default 30 deg)
        utc_offset_hours: shift the time axis labels to local time

    Returns:
        {"success": True, "image_path": ..., "caption": ...,
         "site": ..., "night_window_utc": ..., "targets": [per-target summary]}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import astropy.units as u
    from astropy.time import Time
    from astropy.coordinates import AltAz, get_sun, get_body

    try:
        if not targets:
            return {"success": False, "error": "No targets provided"}
        if isinstance(targets, dict):
            targets = [targets]
        if isinstance(targets, str):
            targets = [{"name": n.strip()} for n in targets.split(",") if n.strip()]

        location, site_label = resolve_site(site, latitude_deg, longitude_deg, height_m)

        resolved, failed = _resolve_targets(targets)
        if not resolved:
            return {"success": False,
                    "error": f"Could not resolve any target: {failed}"}

        # ── Time grid: local noon-to-noon centred on the requested night ──
        t0 = Time(f"{date} 12:00:00", scale="utc") - utc_offset_hours * u.hour
        hours = np.linspace(0, 24, 24 * 12 + 1)  # 5-minute sampling
        times = t0 + hours * u.hour
        frame = AltAz(obstime=times, location=location)

        sun_alt = get_sun(times).transform_to(frame).alt.deg
        try:
            moon = get_body("moon", times, location)
            moon_alt = moon.transform_to(frame).alt.deg
        except Exception:
            moon, moon_alt = None, None

        is_night = sun_alt < -0.833            # geometric sunset/sunrise
        is_astro_dark = sun_alt < -18.0        # astronomical darkness

        step_h = hours[1] - hours[0]

        # ── Per-target curves + summaries ─────────────────────────────
        summaries = []
        curves = []
        for name, coord in resolved:
            alt = coord.transform_to(frame).alt.deg
            curves.append((name, alt))

            observable = (alt >= min_altitude_deg) & is_astro_dark
            obs_hours = float(observable.sum() * step_h)

            best_i = int(np.argmax(np.where(is_night, alt, -90.0))) if is_night.any() else int(np.argmax(alt))
            transit_i = int(np.argmax(alt))

            moon_sep = None
            if moon is not None:
                try:
                    moon_sep = float(coord.separation(moon[transit_i]).deg)
                except Exception:
                    moon_sep = None

            summaries.append({
                "target": name,
                "ra_deg": round(float(coord.icrs.ra.deg), 5),
                "dec_deg": round(float(coord.icrs.dec.deg), 5),
                "max_altitude_deg": round(float(alt.max()), 1),
                "transit_utc": times[transit_i].iso[:16],
                "best_night_altitude_deg": round(float(alt[best_i]), 1) if is_night.any() else None,
                "dark_hours_above_min_alt": round(obs_hours, 1),
                "observable": bool(obs_hours > 0.0),
                "moon_separation_deg": round(moon_sep, 1) if moon_sep is not None else None,
            })

        # ── Plot ──────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(12, 6), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")

        # Darkness shading
        ax.fill_between(hours, 0, 90, where=is_night, color="#1e293b", alpha=0.9, zorder=0)
        ax.fill_between(hours, 0, 90, where=is_astro_dark, color="#0b1120", alpha=1.0, zorder=0)

        palette = ["#06b6d4", "#f59e0b", "#a78bfa", "#4ade80", "#f87171",
                   "#38bdf8", "#facc15", "#fb923c", "#34d399", "#e879f9"]
        for i, (name, alt) in enumerate(curves):
            ax.plot(hours, alt, color=palette[i % len(palette)], linewidth=1.8, label=name)

        if moon_alt is not None:
            ax.plot(hours, moon_alt, color="#94a3b8", linewidth=1.2,
                    linestyle=":", label="Moon")

        ax.axhline(min_altitude_deg, color="#f87171", linewidth=0.9,
                   linestyle="--", alpha=0.8)
        ax.text(0.15, min_altitude_deg + 1, f"min alt {min_altitude_deg:.0f} deg",
                color="#f87171", fontsize=8)

        tz_label = "UTC" if not utc_offset_hours else f"UTC{utc_offset_hours:+.0f}"
        ax.set_xlim(0, 24)
        ax.set_ylim(0, 90)
        ax.set_xticks(np.arange(0, 25, 2))
        ax.set_xticklabels([f"{int((12 + h) % 24):02d}" for h in np.arange(0, 25, 2)])
        ax.set_xlabel(f"Time ({tz_label}, starting {date} 12:00)", color="white", fontsize=11)
        ax.set_ylabel("Altitude (deg)", color="white", fontsize=11)
        ax.tick_params(colors="white", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.grid(color="#334155", alpha=0.3, linewidth=0.5)

        title = f"Visibility from {site_label} — night of {date}"
        ax.set_title(title, color="white", fontsize=13, pad=12)
        legend = ax.legend(loc="upper right", framealpha=0.25, fontsize=9)
        for text in legend.get_texts():
            text.set_color("white")

        img_name = f"visibility_{uuid.uuid4().hex[:10]}.png"
        img_path = os.path.join(RENDERED_DIR, img_name)
        fig.savefig(img_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)

        dark_idx = np.where(is_astro_dark)[0]
        night_window = (
            f"{times[dark_idx[0]].iso[:16]} to {times[dark_idx[-1]].iso[:16]} UTC"
            if dark_idx.size else "Sun never reaches -18 deg (no astronomical darkness)"
        )

        return {
            "success": True,
            "image_path": f"/api/images/{img_name}",
            "caption": title,
            "site": site_label,
            "date": date,
            "min_altitude_deg": min_altitude_deg,
            "astronomical_darkness_utc": night_window,
            "targets": summaries,
            "failed_targets": failed,
        }

    except Exception as e:
        logger.error(f"[VISIBILITY] plan_visibility failed: {e}")
        return {"success": False, "error": str(e)}
