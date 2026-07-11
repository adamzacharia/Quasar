"""IRSA DUST SFD/SF11 Galactic reddening and extinction coefficients."""

from __future__ import annotations

import math
import os
import re
import xml.etree.ElementTree as ET
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import requests


IRSA_DUST_DEFAULT_BASE_URL = "https://irsa.ipac.caltech.edu/cgi-bin/DUST/nph-dust"
IRSA_DUST_REG_SIZE = "2.0"
LOW_GALACTIC_LATITUDE_WARNING = "low Galactic latitude: SFD values unreliable in the plane"
SF11_SCALE = 0.86
FLOAT_RE = re.compile(r"[-+]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][-+]?\d+)?")

# A_lambda / E(B-V), R_V=3.1, Schlafly & Finkbeiner (2011), Table 6.
EXTINCTION_COEFF: Dict[str, float] = {
    "U": 4.334,
    "B": 3.626,
    "V": 2.742,
    "R": 2.169,
    "I": 1.505,
    "sdss_u": 4.239,
    "sdss_g": 3.303,
    "sdss_r": 2.285,
    "sdss_i": 1.698,
    "sdss_z": 1.263,
    "ps1_g": 3.172,
    "ps1_r": 2.271,
    "ps1_i": 1.682,
    "ps1_z": 1.322,
    "ps1_y": 1.087,
    "J": 0.723,
    "H": 0.460,
    "Ks": 0.310,
    "W1": 0.189,
    "W2": 0.146,
}

_BAND_ALIASES = {
    "landoltu": "U",
    "landoltb": "B",
    "landoltv": "V",
    "landoltr": "R",
    "landolti": "I",
    # SF11 Table 6 "Landolt" UBVRI are on the Johnson-Cousins system
    # (Johnson UBV + Cousins RI; Bessell filters reproduce it)
    "johnsonu": "U",
    "johnsonb": "B",
    "johnsonv": "V",
    "johnsonr": "R",
    "johnsoni": "I",
    "cousinsr": "R",
    "cousinsi": "I",
    "bessellu": "U",
    "bessellb": "B",
    "bessellv": "V",
    "bessellr": "R",
    "besselli": "I",
    "johnsoncousinsu": "U",
    "johnsoncousinsb": "B",
    "johnsoncousinsv": "V",
    "johnsoncousinsr": "R",
    "johnsoncousinsi": "I",
    "sdssu": "sdss_u",
    "sdssg": "sdss_g",
    "sdssr": "sdss_r",
    "sdssi": "sdss_i",
    "sdssz": "sdss_z",
    "ps1g": "ps1_g",
    "ps1r": "ps1_r",
    "ps1i": "ps1_i",
    "ps1z": "ps1_z",
    "ps1y": "ps1_y",
    "panstarrsg": "ps1_g",
    "panstarrsr": "ps1_r",
    "panstarrsi": "ps1_i",
    "panstarrsz": "ps1_z",
    "panstarrsy": "ps1_y",
    "panstarrs1g": "ps1_g",
    "panstarrs1r": "ps1_r",
    "panstarrs1i": "ps1_i",
    "panstarrs1z": "ps1_z",
    "panstarrs1y": "ps1_y",
    "2massj": "J",
    "2massh": "H",
    "2massks": "Ks",
    "2massk": "Ks",
    "wisew1": "W1",
    "wisew2": "W2",
    # single letters that belong to exactly one photometric system
    # (u, r, i, g, z are multi-system and stay ambiguous on purpose)
    "b": "B",
    "v": "V",
    "y": "ps1_y",
    "j": "J",
    "h": "H",
    "k": "Ks",
    "ks": "Ks",
    "w1": "W1",
    "w2": "W2",
}


class DustExtinctionService:
    """Timeout-bound client for IRSA DUST reddening and band extinction tables."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        http_get: Optional[Callable[..., Any]] = None,
    ):
        self.base_url = str(base_url or os.getenv("IRSA_DUST_BASE_URL") or IRSA_DUST_DEFAULT_BASE_URL)
        self.timeout = float(timeout if timeout is not None else _env_float("IRSA_DUST_TIMEOUT", 30.0))
        self.http_get = http_get or requests.get

    def ebv(self, ra: Any, dec: Any) -> Dict[str, Any]:
        """Return SFD98 and SF11 E(B-V) at an ICRS sky position."""
        try:
            warnings: List[str] = []
            ra_f, dec_f = _validate_coords(ra, dec)
            galactic_b = _galactic_latitude_deg(ra_f, dec_f)
            if abs(galactic_b) < 5.0:
                warnings.append(LOW_GALACTIC_LATITUDE_WARNING)

            params = {"locstr": f"{ra_f} {dec_f} equ j2000", "regSize": IRSA_DUST_REG_SIZE}
            xml_text = self._get_xml(params)
            values, parse_warnings = _parse_ebv_xml(xml_text)
            warnings.extend(parse_warnings)

            ebv_sfd, ebv_sfd_mean, sfd_warnings = _headline_value(
                values.get("sfd_ref"),
                values.get("sfd_mean"),
                "SFD",
            )
            warnings.extend(sfd_warnings)

            ebv_sf11, ebv_sf11_mean, sf11_warnings = _sf11_values(
                values.get("sf11_ref"),
                values.get("sf11_mean"),
                ebv_sfd,
                ebv_sfd_mean,
            )
            warnings.extend(sf11_warnings)

            return {
                "success": True,
                "ebv_sfd": ebv_sfd,
                "ebv_sf11": ebv_sf11,
                "ebv_sfd_mean": ebv_sfd_mean,
                "ebv_sf11_mean": ebv_sf11_mean,
                "warnings": warnings,
                "provenance": {
                    "service": "IRSA DUST",
                    "base_url": self.base_url,
                    "params": dict(params),
                    "ra": ra_f,
                    "dec": dec_f,
                    "galactic_b_deg": galactic_b,
                    "sf11_scale": SF11_SCALE,
                },
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def extinction_table(self, ra: Any, dec: Any, bands: Optional[Any] = None) -> Dict[str, Any]:
        """Return A_lambda rows using SF11 E(B-V) and the embedded coefficient table."""
        ebv_result = self.ebv(ra, dec)
        if not ebv_result.get("success"):
            return ebv_result
        try:
            warnings = list(ebv_result.get("warnings") or [])
            selected_bands, requested_bands, band_warnings = _normalize_bands(bands)
            warnings.extend(band_warnings)
            ebv_sf11 = float(ebv_result["ebv_sf11"])
            rows = [
                {"band": band, "A_lambda": float(coeff * ebv_sf11), "coeff": float(coeff)}
                for band, coeff in ((band, EXTINCTION_COEFF[band]) for band in selected_bands)
            ]

            provenance = dict(ebv_result.get("provenance") or {})
            provenance.update(
                {
                    "service": "IRSA DUST extinction table",
                    "coefficients_source": "Schlafly & Finkbeiner 2011 Table 6, R_V=3.1",
                    "ebv_used": "ebv_sf11",
                    "requested_bands": requested_bands,
                    "bands": list(selected_bands),
                }
            )
            return {
                "success": True,
                "rows": rows,
                "count": len(rows),
                "ebv_sfd": ebv_result.get("ebv_sfd"),
                "ebv_sf11": ebv_result.get("ebv_sf11"),
                "ebv_sfd_mean": ebv_result.get("ebv_sfd_mean"),
                "ebv_sf11_mean": ebv_result.get("ebv_sf11_mean"),
                "warnings": warnings,
                "provenance": provenance,
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def _get_xml(self, params: Dict[str, Any]) -> str:
        try:
            response = self.http_get(self.base_url, params=params, timeout=self.timeout)
        except requests.RequestException as exc:
            raise RuntimeError(f"IRSA DUST request failed: {exc}") from exc
        status = int(getattr(response, "status_code", 0) or 0)
        if status != 200:
            text = str(getattr(response, "text", "") or "")[:200]
            raise RuntimeError(f"IRSA DUST returned HTTP {status}: {text}")
        text = str(getattr(response, "text", "") or "")
        if not text and getattr(response, "content", None):
            text = bytes(response.content).decode("utf-8", errors="replace")
        if not text.strip():
            raise RuntimeError("IRSA DUST returned an empty XML response.")
        return text


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) and value > 0 else default


def _validate_coords(ra: Any, dec: Any) -> Tuple[float, float]:
    try:
        ra_f = float(ra)
        dec_f = float(dec)
    except (TypeError, ValueError) as exc:
        raise ValueError("ra and dec must be numeric ICRS degrees.") from exc
    if not math.isfinite(ra_f) or not math.isfinite(dec_f):
        raise ValueError("ra and dec must be finite ICRS degrees.")
    if not 0.0 <= ra_f < 360.0:
        raise ValueError("ra must satisfy 0 <= ra < 360 degrees.")
    if not -90.0 <= dec_f <= 90.0:
        raise ValueError("dec must satisfy -90 <= dec <= 90 degrees.")
    return ra_f, dec_f


def _galactic_latitude_deg(ra: float, dec: float) -> float:
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    return float(coord.galactic.b.deg)


def _parse_ebv_xml(xml_text: str) -> Tuple[Dict[str, Optional[float]], List[str]]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise RuntimeError("IRSA DUST returned invalid XML.") from exc

    warnings: List[str] = []
    stats_nodes = [elem for elem in root.iter() if _local_name(elem.tag).lower() == "statistics"]
    if not stats_nodes:
        stats_nodes = [root]
        warnings.append("IRSA DUST XML did not include a statistics section; searched all leaf tags.")

    values: Dict[str, Optional[float]] = {"sfd_ref": None, "sfd_mean": None, "sf11_ref": None, "sf11_mean": None}
    for stats in stats_nodes:
        for elem in stats.iter():
            if list(elem):
                continue
            tag = _compact_name(_local_name(elem.tag))
            text = str(elem.text or "").strip()
            if not text:
                continue
            key = _stat_key(tag)
            if not key or values.get(key) is not None:
                continue
            values[key] = _parse_float_text(text, _local_name(elem.tag))

    if all(value is None for value in values.values()):
        raise RuntimeError("IRSA DUST XML did not include SFD/SandF E(B-V) statistics.")
    return values, warnings


def _local_name(tag: Any) -> str:
    text = str(tag)
    return text.rsplit("}", 1)[-1]


def _compact_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _stat_key(compact_tag: str) -> Optional[str]:
    if "refpixelvaluesandf" in compact_tag:
        return "sf11_ref"
    if "meanvaluesandf" in compact_tag:
        return "sf11_mean"
    if "refpixelvaluesfd" in compact_tag:
        return "sfd_ref"
    if "meanvaluesfd" in compact_tag:
        return "sfd_mean"
    return None


def _parse_float_text(text: str, field: str) -> float:
    match = FLOAT_RE.search(text)
    if not match:
        raise ValueError(f"IRSA DUST field {field} did not contain a numeric value.")
    return float(match.group(0))


def _headline_value(ref_value: Optional[float], mean_value: Optional[float], label: str) -> Tuple[float, Optional[float], List[str]]:
    warnings: List[str] = []
    if ref_value is not None:
        return float(ref_value), _float_or_none(mean_value), warnings
    if mean_value is not None:
        warnings.append(f"IRSA DUST refPixelValue{label} missing; using meanValue{label} as the headline E(B-V).")
        return float(mean_value), float(mean_value), warnings
    raise RuntimeError(f"IRSA DUST XML did not include a {label} E(B-V) value.")


def _sf11_values(
    sf11_ref: Optional[float],
    sf11_mean: Optional[float],
    sfd_ref: float,
    sfd_mean: Optional[float],
) -> Tuple[float, Optional[float], List[str]]:
    warnings: List[str] = []
    if sf11_ref is not None:
        headline = float(sf11_ref)
    elif sf11_mean is not None:
        warnings.append("IRSA DUST refPixelValueSandF missing; using meanValueSandF as the headline SF11 E(B-V).")
        headline = float(sf11_mean)
    else:
        warnings.append("IRSA DUST SandF value missing; computed SF11 E(B-V) as 0.86 x SFD.")
        headline = float(SF11_SCALE * sfd_ref)

    if sf11_mean is not None:
        mean = float(sf11_mean)
    elif sfd_mean is not None:
        warnings.append("IRSA DUST SandF mean missing; computed SF11 mean E(B-V) as 0.86 x SFD mean.")
        mean = float(SF11_SCALE * sfd_mean)
    else:
        mean = None
    return headline, mean, warnings


def _normalize_bands(bands: Optional[Any]) -> Tuple[List[str], Optional[List[str]], List[str]]:
    if bands is None:
        return list(EXTINCTION_COEFF.keys()), None, []
    raw_bands = _coerce_band_list(bands)
    selected: List[str] = []
    warnings: List[str] = []
    for raw in raw_bands:
        band, warning = _normalize_band(raw)
        if warning:
            warnings.append(warning)
        if band and band not in selected:
            selected.append(band)
    return selected, raw_bands, warnings


def _coerce_band_list(bands: Any) -> List[str]:
    if isinstance(bands, str):
        parts = [part.strip() for part in bands.split(",")]
        return [part for part in parts if part]
    if isinstance(bands, Iterable):
        return [str(part).strip() for part in bands if str(part).strip()]
    return [str(bands).strip()]


def _normalize_band(value: Any) -> Tuple[Optional[str], Optional[str]]:
    raw = str(value or "").strip()
    if not raw:
        return None, None
    if raw in EXTINCTION_COEFF:
        return raw, None
    tokens = [token for token in re.split(r"[^a-z0-9]+", raw.lower()) if token and token not in {"band", "filter"}]
    compact = "".join(tokens)
    if compact in _BAND_ALIASES:
        return _BAND_ALIASES[compact], None
    # Only letters shared by several photometric systems are truly ambiguous
    # (u: Landolt/SDSS; r,i: Landolt/SDSS/PS1; g,z: SDSS/PS1). B, V, y, J, H,
    # Ks, W1, W2 each belong to exactly one system and resolve below.
    if len(compact) == 1 and compact in {"u", "r", "i", "g", "z"}:
        return None, f"Band {raw!r} is ambiguous; use an exact key such as 'R', 'sdss_r', or 'ps1_r'."
    matches = [band for band in EXTINCTION_COEFF if band.lower() == compact]
    if len(matches) == 1:
        return matches[0], None
    return None, f"Unknown extinction band {raw!r}; skipped."


def _float_or_none(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    out = float(value)
    return out if math.isfinite(out) else None


__all__ = ["DustExtinctionService", "EXTINCTION_COEFF"]
