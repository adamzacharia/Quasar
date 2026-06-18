"""Splatalogue spectral-line queries for Quasar.

The primary client is ``astroquery.splatalogue``, which wraps Splatalogue's
current JSON search service.  If that service or client fails, Quasar falls
back to Splatalogue's standards-based IVOA Simple Line Access Protocol (SLAP)
endpoint through PyVO.

Registered agent tools:
    - identify_spectral_line(frequency_ghz, tolerance_ghz)
    - search_lines_by_molecule(molecule_name)
    - search_spectral_lines(freq_min_ghz, freq_max_ghz, ...)
"""

from __future__ import annotations

import html
import hashlib
import json
import math
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import requests


SPLATALOGUE_SLAP_URL = "https://splatalogue.online/splata-slap/slap"
SPLATALOGUE_THREADED_URL = (
    "https://splatalogue.online/splata-slap/advancededthread/false"
)
SPLATALOGUE_ADVANCED_URL = "https://splatalogue.online/splata-slap/advanceded/false"
SPLATALOGUE_POLL_URL = (
    "https://splatalogue.online/splata-slap/basicsearchgetdata/{request_number}"
)
SPLATALOGUE_SPECIES_URL = (
    "https://splatalogue.online/splata-slap/advancdedgetspecies"
)
DEFAULT_LINE_LISTS = (
    "LovasNIST",
    "SLAIM",
    "JPL",
    "CDMS",
    "ToyaMA",
    "OSU",
    "TopModel",
    "Recombination",
    "RFI",
)
VALID_ENERGY_TYPES = {"el_cm1", "eu_cm1", "el_k", "eu_k"}
VALID_INTENSITY_TYPES = {"CDMS/JPL (log)", "Sij-mu2", "Aij (log)"}
VALID_VERSIONS = {"v1.0", "v2.0", "v3.0", "vall"}
VALID_EXCLUSIONS = {"atmospheric", "potential", "probable", "known", "none"}

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_NUMBER_WITH_ERROR_RE = re.compile(
    r"(?P<frequency>[+-]?\d+(?:\.\d+)?)"
    r"(?:\s*\(\s*(?P<error>[+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s*\))?"
)


@dataclass(frozen=True)
class SpectralWindow:
    """One normalized frequency interval, represented internally in GHz."""

    minimum_ghz: float
    maximum_ghz: float


@dataclass
class SpectralLineQuery:
    """Canonical, transport-independent Splatalogue query."""

    windows: List[SpectralWindow]
    species_ids: List[int] = field(default_factory=list)
    species_names: List[str] = field(default_factory=list)
    alma_bands: List[int] = field(default_factory=list)
    frame: str = "rest"
    redshift: Optional[float] = None
    radial_velocity_kms: Optional[float] = None
    velocity_convention: str = "radio"
    transition: Optional[str] = None
    energy_min: Optional[float] = None
    energy_max: Optional[float] = None
    energy_type: str = "eu_k"
    intensity_lower_limit: Optional[float] = None
    intensity_type: Optional[str] = None
    version: str = "v3.0"
    line_lists: List[str] = field(
        default_factory=lambda: ["JPL", "CDMS", "SLAIM", "LovasNIST"]
    )
    exclude_categories: List[str] = field(
        default_factory=lambda: ["atmospheric", "potential", "probable"]
    )
    only_astronomically_observed: bool = False
    only_nrao_recommended: bool = False
    maximum_frequency_uncertainty_mhz: Optional[float] = None
    output_mode: str = "merged"
    display_fields: List[str] = field(default_factory=list)
    sort_order: str = "frequency_asc"
    page_size: int = 100

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SpectralLineQuery":
        windows = normalize_spectral_windows(
            payload.get("windows") or [],
            default_unit=str(payload.get("window_unit") or "GHz"),
        )
        if not windows:
            bands = [int(value) for value in payload.get("alma_bands") or []]
            windows = [
                SpectralWindow(*ALMA_BAND_LIMITS_GHZ[band])
                for band in bands
                if band in ALMA_BAND_LIMITS_GHZ
            ]
            windows = merge_spectral_windows(windows)
        if not windows:
            raise ValueError("At least one frequency/wavelength window or ALMA band is required")

        species_ids = [int(value) for value in payload.get("species_ids") or []]
        species_names = [
            str(value).strip()
            for value in payload.get("species_names") or []
            if str(value).strip()
        ]
        query = cls(
            windows=windows,
            species_ids=species_ids,
            species_names=species_names,
            alma_bands=[int(value) for value in payload.get("alma_bands") or []],
            frame=str(payload.get("frame") or "rest").lower(),
            redshift=_optional_float(payload.get("redshift")),
            radial_velocity_kms=_optional_float(payload.get("radial_velocity_kms")),
            velocity_convention=str(
                payload.get("velocity_convention") or "radio"
            ).lower(),
            transition=str(payload.get("transition") or "").strip() or None,
            energy_min=_optional_float(payload.get("energy_min")),
            energy_max=_optional_float(payload.get("energy_max")),
            energy_type=str(payload.get("energy_type") or "eu_k"),
            intensity_lower_limit=_optional_float(
                payload.get("intensity_lower_limit")
            ),
            intensity_type=(
                str(payload.get("intensity_type")).strip()
                if payload.get("intensity_type")
                else None
            ),
            version=str(payload.get("version") or "v3.0"),
            line_lists=list(
                payload.get("line_lists")
                or ["JPL", "CDMS", "SLAIM", "LovasNIST"]
            ),
            exclude_categories=list(
                payload.get("exclude_categories")
                or ["atmospheric", "potential", "probable"]
            ),
            only_astronomically_observed=bool(
                payload.get("only_astronomically_observed", False)
            ),
            only_nrao_recommended=bool(
                payload.get("only_nrao_recommended", False)
            ),
            maximum_frequency_uncertainty_mhz=_optional_float(
                payload.get("maximum_frequency_uncertainty_mhz")
            ),
            output_mode=str(payload.get("output_mode") or "merged").lower(),
            display_fields=list(payload.get("display_fields") or []),
            sort_order=str(payload.get("sort_order") or "frequency_asc"),
            page_size=int(payload.get("page_size") or 100),
        )
        query.validate()
        return query

    @classmethod
    def basic(cls, payload: Mapping[str, Any]) -> "SpectralLineQuery":
        values = dict(payload)
        values.setdefault("version", "v3.0")
        values.setdefault("line_lists", ["JPL", "CDMS", "SLAIM", "LovasNIST"])
        values.setdefault(
            "exclude_categories", ["atmospheric", "potential", "probable"]
        )
        values.setdefault("only_nrao_recommended", True)
        values.setdefault("output_mode", "merged")
        values.setdefault("sort_order", "frequency_asc")
        values.setdefault("page_size", 100)
        return cls.from_payload(values)

    def validate(self) -> None:
        if len(self.species_ids) > 50 or len(self.species_names) > 50:
            raise ValueError("A maximum of 50 species may be selected")
        if len(self.windows) > 30:
            raise ValueError("A maximum of 30 windows may be selected")
        if self.frame not in {"rest", "observed"}:
            raise ValueError("frame must be rest or observed")
        if self.redshift is not None and self.radial_velocity_kms is not None:
            raise ValueError("redshift and radial_velocity_kms are mutually exclusive")
        if self.redshift is not None and self.redshift <= -1:
            raise ValueError("redshift must be greater than -1")
        if self.velocity_convention not in {"radio", "optical", "relativistic"}:
            raise ValueError("Unsupported velocity convention")
        if self.output_mode not in {"raw", "merged"}:
            raise ValueError("output_mode must be raw or merged")
        if self.page_size < 1 or self.page_size > 500:
            raise ValueError("page_size must be between 1 and 500")
        SplatalogueTool._validate_line_lists(self.line_lists)
        SplatalogueTool._validate_advanced_filters(
            energy_min=self.energy_min,
            energy_max=self.energy_max,
            energy_type=self.energy_type,
            intensity_lower_limit=self.intensity_lower_limit,
            intensity_type=self.intensity_type,
        )

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["windows"] = [asdict(window) for window in self.windows]
        return result

    def effective_redshift(self) -> Optional[float]:
        if self.redshift is not None:
            return self.redshift
        if self.radial_velocity_kms is None:
            return None
        beta = self.radial_velocity_kms / 299792.458
        if self.velocity_convention == "radio":
            if beta >= 1:
                raise ValueError("Radio velocity must be below the speed of light")
            return beta / (1 - beta)
        if self.velocity_convention == "optical":
            return beta
        if abs(beta) >= 1:
            raise ValueError("Relativistic velocity magnitude must be below c")
        return math.sqrt((1 + beta) / (1 - beta)) - 1

    def rest_windows(self) -> List[SpectralWindow]:
        if self.frame != "observed":
            return list(self.windows)
        redshift = self.effective_redshift()
        if redshift is None:
            raise ValueError("Observed-frame searches require redshift or radial velocity")
        return [
            SpectralWindow(
                window.minimum_ghz * (1 + redshift),
                window.maximum_ghz * (1 + redshift),
            )
            for window in self.windows
        ]


ALMA_BAND_LIMITS_GHZ: Dict[int, tuple[float, float]] = {
    1: (35.0, 50.0),
    2: (67.0, 116.0),
    3: (84.0, 116.0),
    4: (125.0, 163.0),
    5: (158.0, 211.0),
    6: (211.0, 275.0),
    7: (275.0, 373.0),
    8: (385.0, 500.0),
    9: (602.0, 720.0),
    10: (787.0, 950.0),
}


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Numeric query values must be finite")
    return number


def normalize_spectral_windows(
    values: Sequence[Any],
    *,
    default_unit: str = "GHz",
) -> List[SpectralWindow]:
    """Convert frequency or wavelength windows to GHz and merge overlaps."""
    from astropy import units as u

    windows: List[SpectralWindow] = []
    for value in values:
        if isinstance(value, Mapping):
            low = value.get("minimum", value.get("min", value.get("from")))
            high = value.get("maximum", value.get("max", value.get("to")))
            unit_name = str(value.get("unit") or default_unit)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) < 2:
                raise ValueError("Each spectral window requires two endpoints")
            low, high = value[0], value[1]
            unit_name = default_unit
        else:
            raise ValueError("Invalid spectral window")
        unit = u.Unit(unit_name)
        converted = (
            [float(low), float(high)] * unit
        ).to(u.GHz, equivalencies=u.spectral())
        minimum, maximum = sorted(float(item.value) for item in converted)
        if minimum <= 0 or maximum <= minimum:
            raise ValueError("Spectral windows must have positive, distinct endpoints")
        windows.append(SpectralWindow(minimum, maximum))
    return merge_spectral_windows(windows)


def merge_spectral_windows(
    windows: Sequence[SpectralWindow],
) -> List[SpectralWindow]:
    merged: List[SpectralWindow] = []
    for window in sorted(windows, key=lambda item: item.minimum_ghz):
        if merged and window.minimum_ghz <= merged[-1].maximum_ghz:
            prior = merged[-1]
            merged[-1] = SpectralWindow(
                prior.minimum_ghz,
                max(prior.maximum_ghz, window.maximum_ghz),
            )
        else:
            merged.append(window)
    if len(merged) > 30:
        raise ValueError("A maximum of 30 normalized windows may be queried")
    return merged


class SplatalogueQueryCancelled(RuntimeError):
    """Raised when a running threaded query is canceled."""


class SplatalogueClient:
    """Network transport for current Splatalogue Advanced and SLAP services."""

    def __init__(
        self,
        *,
        session: Optional[requests.Session] = None,
        timeout_seconds: float = 240.0,
    ):
        self.session = session or requests.Session()
        self.timeout_seconds = timeout_seconds

    def query(
        self,
        query: SpectralLineQuery,
        *,
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        query.validate()
        warnings: List[str] = []
        errors: List[str] = []
        try:
            rows = self._query_threaded(query, cancel_event=cancel_event)
            return self._response(
                rows,
                query=query,
                backend="splatalogue_threaded_advanced",
                warnings=warnings,
            )
        except SplatalogueQueryCancelled:
            raise
        except Exception as exc:
            errors.append(f"threaded advanced: {exc}")

        if self._small_compatibility_query(query):
            try:
                rows = self._query_non_threaded(query)
                warnings.append(
                    "The threaded Advanced endpoint failed; the small query was retried "
                    "against the non-threaded Advanced endpoint."
                )
                return self._response(
                    rows,
                    query=query,
                    backend="splatalogue_advanced",
                    warnings=warnings,
                )
            except Exception as exc:
                errors.append(f"non-threaded advanced: {exc}")

        unsupported = self._slap_unsupported_filters(query)
        if unsupported:
            raise RuntimeError(
                "Splatalogue Advanced query failed and SLAP cannot verify filters: "
                + ", ".join(unsupported)
                + ". "
                + " | ".join(errors)
            )
        try:
            rows = SplatalogueTool._query_slap(
                query.windows[0].minimum_ghz,
                query.windows[0].maximum_ghz,
            )
            warnings.append(
                "Results came from the IVOA SLAP fallback after the Advanced service "
                "failed. Only constraints represented in the returned SLAP data were used."
            )
            return self._response(
                rows,
                query=query,
                backend="slap",
                degraded=True,
                warnings=warnings,
                errors=errors,
            )
        except Exception as exc:
            errors.append(f"SLAP: {exc}")
            raise RuntimeError("Splatalogue query failed: " + " | ".join(errors)) from exc

    def _query_threaded(
        self,
        query: SpectralLineQuery,
        *,
        cancel_event: Optional[threading.Event],
    ) -> List[Mapping[str, Any]]:
        payload = self._advanced_payload(query)
        response = self.session.post(
            SPLATALOGUE_THREADED_URL,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        body = response.json()
        item = body[0] if isinstance(body, list) and body else body
        request_number = (
            item.get("requestnumber") if isinstance(item, Mapping) else None
        )
        if request_number is None:
            raise RuntimeError(f"Thread submission did not return a request number: {body}")

        started = time.monotonic()
        delay = 1.0
        while True:
            if cancel_event and cancel_event.is_set():
                raise SplatalogueQueryCancelled("Splatalogue query canceled")
            elapsed = time.monotonic() - started
            if elapsed > self.timeout_seconds:
                raise TimeoutError("Splatalogue threaded query exceeded 240 seconds")
            time.sleep(delay)
            response = self.session.get(
                SPLATALOGUE_POLL_URL.format(request_number=request_number),
                timeout=30,
            )
            response.raise_for_status()
            rows = response.json()
            if not isinstance(rows, list):
                raise RuntimeError("Unexpected threaded query response")
            error_messages = []
            for row in rows:
                message = str(row.get("searchErrorMessage") or "").strip()
                normalized = message.lower()
                if (
                    message
                    and normalized not in {"no error", "none", "null"}
                    and "using thread" not in normalized
                    and "processing" not in normalized
                ):
                    error_messages.append(message)
            if error_messages and not any(
                row.get("species_id") is not None for row in rows
            ):
                raise RuntimeError("; ".join(dict.fromkeys(error_messages)))
            if self._poll_complete(rows):
                return self._strip_sentinel(rows)
            delay = 2.5

    def _query_non_threaded(
        self, query: SpectralLineQuery
    ) -> List[Mapping[str, Any]]:
        response = self.session.post(
            SPLATALOGUE_ADVANCED_URL,
            json=self._advanced_payload(query),
            timeout=60,
        )
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, list):
            raise RuntimeError("Unexpected Advanced query response")
        return self._strip_sentinel(body)

    @staticmethod
    def _advanced_payload(query: SpectralLineQuery) -> Dict[str, Any]:
        """Build the current Advanced JSON payload using frequenciesUnit."""
        from astropy import units as u
        from astroquery.splatalogue import Splatalogue

        rest_windows = query.rest_windows()
        first = rest_windows[0]
        chemical_name = None
        if len(query.species_names) == 1:
            chemical_name = SplatalogueTool._chemical_query(query.species_names[0])
        kwargs: Dict[str, Any] = {
            "chemical_name": chemical_name,
            "transition": query.transition,
            "energy_min": query.energy_min,
            "energy_max": query.energy_max,
            "energy_type": query.energy_type,
            "intensity_lower_limit": query.intensity_lower_limit,
            "intensity_type": query.intensity_type,
            "version": query.version,
            "exclude": query.exclude_categories,
            "line_lists": query.line_lists,
            "only_astronomically_observed": query.only_astronomically_observed,
            "only_NRAO_recommended": query.only_nrao_recommended,
            "export_limit": 50000,
            "get_query_payload": True,
        }
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        payload = Splatalogue.query_lines(
            first.minimum_ghz * u.GHz,
            first.maximum_ghz * u.GHz,
            **kwargs,
        )
        if "body" in payload:
            body = json.loads(payload["body"]) if isinstance(payload.get("body"), str) else dict(payload["body"])
        else:
            body = dict(payload)
        body.pop("userInputFrequenciesUnit", None)
        body["frequenciesUnit"] = "GHz"
        body["userInputFrequenciesFrom"] = [
            window.minimum_ghz for window in rest_windows
        ]
        body["userInputFrequenciesTo"] = [
            window.maximum_ghz for window in rest_windows
        ]
        body.update(
            {
                "lineStrengthDisplayCDMSJPL": True,
                "lineStrengthDisplaySijMu2": True,
                "lineStrengthDisplaySij": True,
                "lineStrengthDisplayAij": True,
                "lineStrengthDisplayLovasAST": True,
                "energyLevelOne": True,
                "energyLevelTwo": True,
                "energyLevelThree": True,
                "energyLevelFour": True,
                "displayObservationReference": True,
                "displayObservationSource": True,
                "displayTelescopeLovasNIST": True,
                "displayHFSIntensity": True,
                "displayUnresolvedQuantumNumbers": True,
                "displayUpperStateDegeneracy": True,
                "displayMoleculeTag": True,
                "displayQuantumNumberCode": True,
                "displayLabRef": True,
                "displayNRAORecommendedFrequencies": True,
                "displayUniqueSpeciesTag": True,
                "displayUniqueLineIDNumber": True,
            }
        )
        redshift = query.effective_redshift()
        if redshift is not None:
            body["frequencyRedshift"] = redshift
        if query.species_ids:
            body["speciesSelectBox"] = [str(value) for value in query.species_ids]
        if query.maximum_frequency_uncertainty_mhz is not None:
            body["frequencyErrorLimit"] = True
            body["frequencyErrorLimitValue"] = (
                query.maximum_frequency_uncertainty_mhz
            )
        return {
            "body": json.dumps(body),
            "headers": payload.get("headers") or {},
        }

    @staticmethod
    def _poll_complete(rows: Sequence[Mapping[str, Any]]) -> bool:
        if not rows:
            return False
        for row in rows:
            message = str(row.get("searchErrorMessage") or "").strip().lower()
            if "using thread" in message or "processing" in message:
                return False
        return any(
            row.get("species_id") is not None
            or row.get("lineid") is not None
            or row.get("sqlquery") is not None
            for row in rows
        )

    @staticmethod
    def _strip_sentinel(
        rows: Sequence[Mapping[str, Any]],
    ) -> List[Mapping[str, Any]]:
        return [
            row
            for row in rows
            if row.get("species_id") is not None or row.get("lineid") is not None
        ]

    @staticmethod
    def _small_compatibility_query(query: SpectralLineQuery) -> bool:
        total_width = sum(
            window.maximum_ghz - window.minimum_ghz for window in query.windows
        )
        return len(query.windows) == 1 and len(query.species_ids) <= 5 and total_width <= 2

    @staticmethod
    def _slap_unsupported_filters(query: SpectralLineQuery) -> List[str]:
        unsupported: List[str] = []
        if len(query.windows) != 1:
            unsupported.append("multiple windows")
        if query.species_ids:
            unsupported.append("species IDs")
        if query.exclude_categories:
            unsupported.append("species-category exclusions")
        if query.version not in {"vall", ""}:
            unsupported.append("data version")
        if query.maximum_frequency_uncertainty_mhz is not None:
            unsupported.append("maximum frequency uncertainty")
        return unsupported

    @staticmethod
    def _response(
        rows: Iterable[Any],
        *,
        query: SpectralLineQuery,
        backend: str,
        degraded: bool = False,
        warnings: Optional[List[str]] = None,
        errors: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return {
            "rows": list(rows),
            "backend": backend,
            "degraded": degraded,
            "unsupported_filters": [],
            "warnings": list(warnings or []),
            "errors": list(errors or []),
            "query_provenance": {
                "query": query.to_dict(),
                "service": "Splatalogue",
                "backend": backend,
                "queried_at_unix": time.time(),
            },
        }


class SplatalogueTool:
    """Query and normalize molecular spectral-line data from Splatalogue."""

    def __init__(self, client: Optional[SplatalogueClient] = None):
        self.client = client or SplatalogueClient()

    def query_catalog(
        self,
        query: SpectralLineQuery,
        *,
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        """Run a canonical query and return raw or deterministically merged rows."""
        response = self.client.query(query, cancel_event=cancel_event)
        center = None
        if len(query.windows) == 1:
            center = (
                query.windows[0].minimum_ghz + query.windows[0].maximum_ghz
            ) / 2
        lines = [
            line
            for row in response.pop("rows", [])
            if (line := self._normalize_row(row, query_frequency_ghz=center))
            is not None
        ]
        if response["backend"] == "slap":
            lines = self._filter_slap_results(
                lines,
                molecule_name=(
                    query.species_names[0]
                    if len(query.species_names) == 1
                    else None
                ),
                transition=query.transition,
                energy_min=query.energy_min,
                energy_max=query.energy_max,
                energy_type=query.energy_type,
                intensity_lower_limit=query.intensity_lower_limit,
                intensity_type=query.intensity_type,
                version=query.version,
                exclude=query.exclude_categories,
                line_lists=query.line_lists,
                only_astronomically_observed=query.only_astronomically_observed,
                only_nrao_recommended=query.only_nrao_recommended,
            )
        if query.transition:
            transition_token = self._search_token(query.transition)
            lines = [
                line
                for line in lines
                if transition_token
                in self._search_token(
                    line.get("transition")
                    or line.get("resolved_quantum_numbers")
                    or line.get("unresolved_quantum_numbers")
                )
            ]
        if query.maximum_frequency_uncertainty_mhz is not None:
            lines = [
                line
                for line in lines
                if line.get("frequency_uncertainty_mhz") is not None
                and float(line["frequency_uncertainty_mhz"])
                <= query.maximum_frequency_uncertainty_mhz
            ]
        raw_lines = sorted(
            lines,
            key=lambda line: (
                float(line["frequency_ghz"]),
                str(line.get("species") or ""),
                str(line.get("transition") or ""),
            ),
        )
        output_lines = (
            self._deduplicate(raw_lines)
            if query.output_mode == "merged"
            else raw_lines
        )
        if query.sort_order == "frequency_desc":
            output_lines.reverse()
        response.update(
            {
                "lines": output_lines,
                "raw_lines": raw_lines,
                "total_matches": len(output_lines),
                "raw_total_matches": len(raw_lines),
                "query": query.to_dict(),
            }
        )
        return response

    def identify_spectral_line(
        self,
        frequency_ghz: float,
        tolerance_ghz: float = 0.01,
        top_n: int = 5,
        *,
        molecule_name: Optional[str] = None,
        transition: Optional[str] = None,
        line_lists: Optional[Sequence[str]] = None,
        only_astronomically_observed: bool = False,
        only_nrao_recommended: bool = False,
    ) -> Dict[str, Any]:
        """Identify the closest cataloged transitions to a rest frequency."""
        frequency = self._positive_float(frequency_ghz, "frequency_ghz")
        tolerance = self._nonnegative_float(tolerance_ghz, "tolerance_ghz")
        if tolerance == 0:
            raise ValueError("tolerance_ghz must be greater than zero")

        result = self.search_spectral_lines(
            freq_min_ghz=frequency - tolerance,
            freq_max_ghz=frequency + tolerance,
            molecule_name=molecule_name,
            transition=transition,
            line_lists=line_lists,
            only_astronomically_observed=only_astronomically_observed,
            only_nrao_recommended=only_nrao_recommended,
            top_n=top_n,
            query_frequency_ghz=frequency,
        )
        result["query_frequency_ghz"] = frequency
        result["tolerance_ghz"] = tolerance
        return result

    def search_lines_by_molecule(
        self,
        molecule_name: str,
        freq_min_ghz: Optional[float] = None,
        freq_max_ghz: Optional[float] = None,
        top_n: int = 10,
        *,
        transition: Optional[str] = None,
        energy_min: Optional[float] = None,
        energy_max: Optional[float] = None,
        energy_type: str = "eu_k",
        intensity_lower_limit: Optional[float] = None,
        intensity_type: Optional[str] = None,
        line_lists: Optional[Sequence[str]] = None,
        only_astronomically_observed: bool = False,
        only_nrao_recommended: bool = False,
    ) -> Dict[str, Any]:
        """Search for transitions associated with a molecule or species name."""
        molecule = str(molecule_name or "").strip()
        if not molecule:
            raise ValueError("molecule_name is required")

        result = self.search_spectral_lines(
            freq_min_ghz=1.0 if freq_min_ghz is None else freq_min_ghz,
            freq_max_ghz=1000.0 if freq_max_ghz is None else freq_max_ghz,
            molecule_name=molecule,
            transition=transition,
            energy_min=energy_min,
            energy_max=energy_max,
            energy_type=energy_type,
            intensity_lower_limit=intensity_lower_limit,
            intensity_type=intensity_type,
            line_lists=line_lists,
            only_astronomically_observed=only_astronomically_observed,
            only_nrao_recommended=only_nrao_recommended,
            top_n=top_n,
        )
        result["molecule"] = molecule
        return result

    def search_spectral_lines(
        self,
        freq_min_ghz: float,
        freq_max_ghz: float,
        *,
        molecule_name: Optional[str] = None,
        transition: Optional[str] = None,
        energy_min: Optional[float] = None,
        energy_max: Optional[float] = None,
        energy_type: str = "eu_k",
        intensity_lower_limit: Optional[float] = None,
        intensity_type: Optional[str] = None,
        version: str = "v3.0",
        exclude: Optional[Sequence[str]] = None,
        line_lists: Optional[Sequence[str]] = None,
        only_astronomically_observed: bool = False,
        only_nrao_recommended: bool = False,
        top_n: int = 25,
        query_frequency_ghz: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run an advanced Splatalogue query and return normalized line records.

        Frequencies are always expressed in GHz in Quasar responses. Splatalogue
        catalog names, line strengths, energy levels, and identifiers are kept
        when available. Duplicate transitions from multiple line lists are
        merged while retaining all catalog names in ``catalogs``.
        """
        low = self._positive_float(freq_min_ghz, "freq_min_ghz")
        high = self._positive_float(freq_max_ghz, "freq_max_ghz")
        if low >= high:
            raise ValueError("freq_min_ghz must be less than freq_max_ghz")

        limit = self._positive_int(top_n, "top_n")
        center = (
            self._positive_float(query_frequency_ghz, "query_frequency_ghz")
            if query_frequency_ghz is not None
            else None
        )
        molecule = str(molecule_name or "").strip() or None
        transition_filter = str(transition or "").strip() or None
        selected_lists = self._validate_line_lists(line_lists)
        self._validate_advanced_filters(
            energy_min=energy_min,
            energy_max=energy_max,
            energy_type=energy_type,
            intensity_lower_limit=intensity_lower_limit,
            intensity_type=intensity_type,
        )
        if version not in VALID_VERSIONS:
            raise ValueError(f"version must be one of {sorted(VALID_VERSIONS)}")
        if exclude is not None:
            invalid_exclusions = sorted(set(exclude) - VALID_EXCLUSIONS)
            if invalid_exclusions:
                raise ValueError(
                    "Unsupported exclusion(s): " + ", ".join(invalid_exclusions)
                )

        query_options = {
            "molecule_name": molecule,
            "transition": transition_filter,
            "energy_min": energy_min,
            "energy_max": energy_max,
            "energy_type": energy_type,
            "intensity_lower_limit": intensity_lower_limit,
            "intensity_type": intensity_type,
            "version": version,
            "exclude": list(exclude) if exclude is not None else None,
            "line_lists": selected_lists,
            "only_astronomically_observed": bool(only_astronomically_observed),
            "only_nrao_recommended": bool(only_nrao_recommended),
        }

        errors: List[str] = []
        backend = "astroquery"
        try:
            rows = self._query_astroquery(
                low,
                high,
                export_stop=min(max(limit * 4, 250), 1000),
                **query_options,
            )
        except Exception as exc:
            errors.append(f"astroquery: {exc}")
            backend = "slap"
            try:
                rows = self._query_slap(low, high)
            except Exception as slap_exc:
                errors.append(f"SLAP: {slap_exc}")
                return {
                    "query_frequency_min_ghz": low,
                    "query_frequency_max_ghz": high,
                    "n_matches": 0,
                    "total_matches": 0,
                    "lines": [],
                    "backend": "unavailable",
                    "error": "Splatalogue query failed",
                    "details": errors,
                }

        lines = [
            line
            for row in rows
            if (line := self._normalize_row(row, query_frequency_ghz=center)) is not None
        ]
        if backend == "slap":
            lines = self._filter_slap_results(lines, **query_options)

        lines = self._deduplicate(lines)
        if center is not None:
            lines.sort(
                key=lambda line: (
                    abs(float(line.get("offset_mhz") or 0.0)),
                    -self._ranking_strength(line),
                    float(line["frequency_ghz"]),
                )
            )
        else:
            lines.sort(key=lambda line: float(line["frequency_ghz"]))

        total_matches = len(lines)
        returned = lines[:limit]
        response: Dict[str, Any] = {
            "query_frequency_min_ghz": low,
            "query_frequency_max_ghz": high,
            "n_matches": len(returned),
            "total_matches": total_matches,
            "lines": returned,
            "backend": backend,
            "degraded": backend == "slap",
            "unsupported_filters": [],
            "warnings": [],
            "query_provenance": {
                "backend": backend,
                "queried_at_unix": time.time(),
                "legacy_method": "search_spectral_lines",
            },
        }
        if errors:
            unsupported_filters = []
            if version not in {"vall", ""}:
                unsupported_filters.append("data version")
            if exclude:
                unsupported_filters.append("species-category exclusions")
            response["unsupported_filters"] = unsupported_filters
            response["note"] = (
                "The Astroquery request failed, so results were obtained from "
                "Splatalogue's IVOA SLAP endpoint."
            )
            response["details"] = errors
            response["warnings"] = [
                "Legacy compatibility fallback was used."
                + (
                    " The fallback cannot verify: "
                    + ", ".join(unsupported_filters)
                    + "."
                    if unsupported_filters
                    else ""
                )
            ]
        return response

    @staticmethod
    def _query_astroquery(
        freq_min_ghz: float,
        freq_max_ghz: float,
        *,
        molecule_name: Optional[str],
        transition: Optional[str],
        energy_min: Optional[float],
        energy_max: Optional[float],
        energy_type: str,
        intensity_lower_limit: Optional[float],
        intensity_type: Optional[str],
        version: str,
        exclude: Optional[Sequence[str]],
        line_lists: Sequence[str],
        only_astronomically_observed: bool,
        only_nrao_recommended: bool,
        export_stop: int,
    ) -> Iterable[Any]:
        del export_stop
        query = SpectralLineQuery(
            windows=[SpectralWindow(freq_min_ghz, freq_max_ghz)],
            species_names=[molecule_name] if molecule_name else [],
            transition=transition,
            energy_min=energy_min,
            energy_max=energy_max,
            energy_type=energy_type,
            intensity_lower_limit=intensity_lower_limit,
            intensity_type=intensity_type,
            version=version,
            exclude_categories=list(exclude or []),
            line_lists=list(line_lists),
            only_astronomically_observed=only_astronomically_observed,
            only_nrao_recommended=only_nrao_recommended,
            output_mode="raw",
        )
        return SplatalogueClient().query(query)["rows"]

    @staticmethod
    def _query_slap(freq_min_ghz: float, freq_max_ghz: float) -> Iterable[Any]:
        from astropy import units as u
        from pyvo.dal import SLAService

        # Wavelength is inverse to frequency, so the high-frequency endpoint
        # is the lower wavelength bound expected by SLAP.
        wavelength = (
            [freq_max_ghz, freq_min_ghz] * u.GHz
        ).to(u.m, equivalencies=u.spectral())
        return SLAService(SPLATALOGUE_SLAP_URL).search(
            wavelength=wavelength
        ).to_table()

    def _normalize_row(
        self,
        row: Any,
        *,
        query_frequency_ghz: Optional[float],
    ) -> Optional[Dict[str, Any]]:
        species = self._clean_text(
            self._row_value(row, "name", "Species", "molecular formula", "title")
        )
        chemical_name = self._clean_text(
            self._row_value(row, "chemical_name", "Chemical Name", "chemicalname")
        )
        transition = self._clean_text(
            self._row_value(
                row,
                "resolved_QNs",
                "Quantum Numbers",
                "quantum_numbers",
                "quantum numbers",
            )
        )
        frequency_ghz = self._frequency_ghz(row)
        if frequency_ghz is None or frequency_ghz <= 0:
            return None

        catalog = self._clean_text(
            self._row_value(row, "linelist", "Line List", "catalog name", "source")
        )
        if not species:
            species = chemical_name or "Unknown species"
        molecule = chemical_name or species
        ordered_raw = self._clean_text(self._row_value(row, "orderedFreq"))
        measured_raw = self._clean_text(self._row_value(row, "measFreq"))
        predicted_frequency, predicted_error, ordered_observed = (
            self._parse_frequency_display(ordered_raw)
        )
        measured_frequency, measured_error, measured_observed = (
            self._parse_frequency_display(measured_raw)
        )
        if predicted_frequency is None and self._row_value(row, "orderedfreq") is not None:
            predicted_frequency = frequency_ghz
        uncertainty = (
            measured_error
            if measured_frequency is not None and measured_error is not None
            else predicted_error
        )
        observed_frequency = ordered_observed or measured_observed
        bands = self._alma_bands_for_frequency(
            observed_frequency if observed_frequency is not None else frequency_ghz
        )

        line: Dict[str, Any] = {
            "species": species,
            "name": species,
            "formula": species,
            "molecule": molecule,
            "chemical_name": chemical_name or molecule,
            "species_status": (
                "known"
                if self._bool_or_none(
                    self._row_value(row, "transition_in_space", "known_interstellar")
                )
                else "unknown"
            ),
            "transition": transition,
            "resolved_quantum_numbers": transition,
            "unresolved_quantum_numbers": self._clean_text(
                self._row_value(row, "unres_quantum_numbers")
            ),
            "quantum_number_code": self._clean_text(
                self._row_value(row, "qnCode")
            ),
            "molecule_tag": self._clean_text(
                self._row_value(row, "moleculeTag")
            ),
            "frequency_ghz": round(frequency_ghz, 9),
            "catalog_frequency_ghz": round(frequency_ghz, 9),
            "predicted_frequency_ghz": predicted_frequency,
            "predicted_frequency_uncertainty_mhz": predicted_error,
            "measured_frequency_ghz": measured_frequency,
            "measured_frequency_uncertainty_mhz": measured_error,
            "observed_frequency_ghz": observed_frequency,
            "frequency_uncertainty_mhz": uncertainty,
            "frequency_basis": (
                "NRAO recommended"
                if self._bool_or_none(self._row_value(row, "Lovas_NRAO"))
                else "catalog ordered frequency"
            ),
            "frequency_selection_reason": "raw catalog entry",
            "ordered_frequency_raw": ordered_raw,
            "measured_frequency_raw": measured_raw,
            "source": catalog or "Splatalogue",
            "catalogs": [catalog] if catalog else [],
            "log_intensity": self._number_or_none(
                self._row_value(
                    row,
                    "intintensity",
                    "CDMS/JPL Intensity",
                    "log_intensity",
                )
            ),
            "sijmu2": self._number_or_none(
                self._row_value(row, "sijmu2", "Sij-mu2")
            ),
            "sij": self._number_or_none(self._row_value(row, "sij", "Sij")),
            "aij_log": self._number_or_none(
                self._row_value(row, "aij", "Log<sub>10</sub> (A<sub>ij</sub>)")
            ),
            "lovas_astronomical_intensity": self._number_or_none(
                self._row_value(row, "LovasASTIntensity")
            ),
            "lower_energy_k": self._number_or_none(
                self._row_value(
                    row,
                    "lower_state_energy_K",
                    "lowerstateenergyK",
                    "E_L (K)",
                )
            ),
            "upper_energy_k": self._number_or_none(
                self._row_value(
                    row,
                    "upper_state_energy_K",
                    "upperstateenergyK",
                    "E_U (K)",
                )
            ),
            "lower_energy_cm1": self._number_or_none(
                self._row_value(
                    row,
                    "lower_state_energy",
                    "lowerstateenergy",
                    "E_L (cm^-1)",
                )
            ),
            "upper_energy_cm1": self._number_or_none(
                self._row_value(
                    row,
                    "upper_state_energy",
                    "upperstateenergy",
                    "E_U (cm^-1)",
                )
            ),
            "line_id": self._clean_text(
                self._row_value(row, "lineid", "line_id", "Line ID")
            ),
            "raw_line_ids": [
                self._clean_text(
                    self._row_value(row, "lineid", "line_id", "Line ID")
                )
            ],
            "species_id": self._clean_text(
                self._row_value(row, "species_id", "Species ID")
            ),
            "astronomically_observed": self._bool_or_none(
                self._row_value(
                    row,
                    "transition_in_space",
                    "known_interstellar",
                )
            ),
            "nrao_recommended": self._bool_or_none(
                self._row_value(
                    row,
                    "Lovas_NRAO",
                    "frequency recommended",
                )
            ),
            "hfs_relative_intensity": self._number_or_none(
                self._row_value(row, "rel_int_HFS_Lovas")
            ),
            "upper_state_degeneracy": self._number_or_none(
                self._row_value(row, "upperStateDegen")
            ),
            "laboratory_reference": self._clean_text(
                self._row_value(row, "labref_Lovas_NIST")
            ),
            "astronomical_reference": self._clean_text(
                self._row_value(row, "obsref_Lovas_NIST")
            ),
            "observation_source": self._clean_text(
                self._row_value(row, "source_Lovas_NIST")
            ),
            "observation_telescope": self._clean_text(
                self._row_value(row, "telescope_Lovas_NIST")
            ),
            "alma_bands": bands,
            "alma_preferred_band": self._preferred_alma_band(
                observed_frequency if observed_frequency is not None else frequency_ghz,
                bands,
            ),
            "alma_band_edge_warning": self._band_edge_warning(
                observed_frequency if observed_frequency is not None else frequency_ghz
            ),
            "backend_provenance": "Splatalogue",
        }
        line["unique_line_id"] = (
            f"{line['species_id']}:{line['line_id']}"
            if line["species_id"] and line["line_id"]
            else hashlib.sha1(
                (
                    f"{species}|{transition}|{frequency_ghz:.9f}|{catalog}"
                ).encode("utf-8")
            ).hexdigest()[:20]
        )
        if query_frequency_ghz is not None:
            line["offset_mhz"] = round(
                (frequency_ghz - query_frequency_ghz) * 1000.0,
                6,
            )
        return line

    @staticmethod
    def _parse_frequency_display(
        value: str,
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """Parse Splatalogue frequency HTML after tags have been removed.

        Display values are GHz. Parenthesized errors are MHz in the current UI.
        A second number, when present, is the redshifted/observed frequency.
        """
        if not value:
            return None, None, None
        matches = list(_NUMBER_WITH_ERROR_RE.finditer(value))
        if not matches:
            return None, None, None
        rest = SplatalogueTool._number_or_none(matches[0].group("frequency"))
        error = SplatalogueTool._number_or_none(matches[0].group("error"))
        observed = None
        for match in matches[1:]:
            candidate = SplatalogueTool._number_or_none(match.group("frequency"))
            if candidate is not None and candidate > 0:
                observed = candidate
                break
        return rest, error, observed

    @staticmethod
    def _alma_bands_for_frequency(frequency_ghz: float) -> List[int]:
        return [
            band
            for band, (low, high) in ALMA_BAND_LIMITS_GHZ.items()
            if low <= frequency_ghz <= high
        ]

    @staticmethod
    def _preferred_alma_band(
        frequency_ghz: float, bands: Sequence[int]
    ) -> Optional[int]:
        if 3 in bands and 2 in bands and 84 <= frequency_ghz <= 116:
            return 3
        if not bands:
            return None
        return min(
            bands,
            key=lambda band: abs(
                frequency_ghz
                - sum(ALMA_BAND_LIMITS_GHZ[band]) / 2
            ),
        )

    @staticmethod
    def _band_edge_warning(frequency_ghz: float) -> Optional[str]:
        nearby = []
        for band, (low, high) in ALMA_BAND_LIMITS_GHZ.items():
            if min(abs(frequency_ghz - low), abs(frequency_ghz - high)) <= 0.2:
                nearby.append(str(band))
        if not nearby:
            return None
        return (
            "Within 0.2 GHz of the nominal receiver edge for ALMA Band"
            + ("s " if len(nearby) > 1 else " ")
            + ", ".join(nearby)
            + "; verify the actual FDM setup."
        )

    def _filter_slap_results(
        self,
        lines: List[Dict[str, Any]],
        *,
        molecule_name: Optional[str],
        transition: Optional[str],
        energy_min: Optional[float],
        energy_max: Optional[float],
        energy_type: str,
        intensity_lower_limit: Optional[float],
        intensity_type: Optional[str],
        version: str,
        exclude: Optional[Sequence[str]],
        line_lists: Sequence[str],
        only_astronomically_observed: bool,
        only_nrao_recommended: bool,
    ) -> List[Dict[str, Any]]:
        """Apply filters locally when the standards-based fallback is used."""
        del version, exclude  # These advanced website concepts are not in SLAP.

        molecule_query = self._search_token(molecule_name)
        formula_query = (
            str(molecule_name).strip()
            if self._looks_like_formula(molecule_name)
            else None
        )
        transition_query = self._search_token(transition)
        allowed_catalogs = {self._catalog_token(value) for value in line_lists}
        energy_field = {
            "el_k": "lower_energy_k",
            "eu_k": "upper_energy_k",
            "el_cm1": "lower_energy_cm1",
            "eu_cm1": "upper_energy_cm1",
        }.get(energy_type)
        intensity_field = {
            "CDMS/JPL (log)": "log_intensity",
            "Sij-mu2": "sijmu2",
            "Aij (log)": "aij_log",
        }.get(intensity_type or "")

        filtered: List[Dict[str, Any]] = []
        for line in lines:
            searchable_name = self._search_token(
                " ".join(
                    str(line.get(key) or "")
                    for key in ("species", "molecule", "chemical_name")
                )
            )
            if molecule_query:
                if formula_query:
                    species = str(line.get("species") or "").strip()
                    if re.match(
                        rf"^{re.escape(formula_query)}(?![A-Za-z0-9+])",
                        species,
                        flags=re.IGNORECASE,
                    ) is None:
                        continue
                elif molecule_query not in searchable_name:
                    continue
            if transition_query and transition_query not in self._search_token(
                line.get("transition")
            ):
                continue
            if allowed_catalogs and self._catalog_token(line.get("source")) not in allowed_catalogs:
                continue
            if only_astronomically_observed and line.get("astronomically_observed") is not True:
                continue
            if only_nrao_recommended and line.get("nrao_recommended") is not True:
                continue

            if energy_field:
                energy = self._number_or_none(line.get(energy_field))
                if energy_min is not None and (energy is None or energy < float(energy_min)):
                    continue
                if energy_max is not None and (energy is None or energy > float(energy_max)):
                    continue
            if intensity_lower_limit is not None and intensity_field:
                intensity = self._number_or_none(line.get(intensity_field))
                if intensity is None or intensity < float(intensity_lower_limit):
                    continue
            filtered.append(line)
        return filtered

    @staticmethod
    def _deduplicate(lines: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        groups: Dict[tuple, List[Dict[str, Any]]] = {}
        for line in lines:
            key = (
                SplatalogueTool._search_token(
                    line.get("species") or line.get("molecule")
                ),
                SplatalogueTool._search_token(line.get("transition")),
                SplatalogueTool._search_token(
                    line.get("unresolved_quantum_numbers")
                ),
                SplatalogueTool._number_or_none(
                    line.get("hfs_relative_intensity")
                ),
                round(float(line["frequency_ghz"]), 6),
            )
            groups.setdefault(key, []).append(line)

        merged: List[Dict[str, Any]] = []
        for entries in groups.values():
            ranked = sorted(entries, key=SplatalogueTool._merge_priority)
            selected = dict(ranked[0])
            catalogs: List[str] = []
            raw_ids: List[str] = []
            for entry in entries:
                for catalog in entry.get("catalogs") or []:
                    if catalog and catalog not in catalogs:
                        catalogs.append(catalog)
                for line_id in entry.get("raw_line_ids") or [entry.get("line_id")]:
                    if line_id and line_id not in raw_ids:
                        raw_ids.append(line_id)
                for field, value in entry.items():
                    if selected.get(field) in (None, "", [], "N/A") and value not in (
                        None,
                        "",
                        [],
                        "N/A",
                    ):
                        selected[field] = value
            selected["catalogs"] = catalogs
            selected["source"] = ", ".join(catalogs) if catalogs else "Splatalogue"
            selected["raw_line_ids"] = raw_ids
            if selected.get("nrao_recommended") is True:
                reason = "NRAO-recommended frequency"
            elif selected.get("frequency_uncertainty_mhz") is not None:
                reason = "smallest catalog frequency uncertainty"
            else:
                reason = (
                    f"catalog precedence ({selected.get('catalogs', ['Splatalogue'])[0]})"
                )
            selected["frequency_selection_reason"] = reason
            selected["catalog_agreement_count"] = len(catalogs)
            selected["unique_line_id"] = "merged:" + hashlib.sha1(
                repr(key).encode("utf-8")
            ).hexdigest()[:20]
            merged.append(selected)
        return merged

    @staticmethod
    def _merge_priority(line: Mapping[str, Any]) -> tuple:
        catalog_priority = {
            "cdms": 0,
            "jpl": 1,
            "slaim": 2,
            "lovasnist": 3,
        }
        catalog = SplatalogueTool._catalog_token(line.get("source"))
        uncertainty = SplatalogueTool._number_or_none(
            line.get("frequency_uncertainty_mhz")
        )
        return (
            0 if line.get("nrao_recommended") is True else 1,
            uncertainty if uncertainty is not None else float("inf"),
            catalog_priority.get(catalog, 99),
            str(line.get("line_id") or ""),
        )

    @staticmethod
    def _row_value(row: Any, *names: str) -> Any:
        colnames = getattr(row, "colnames", None)
        if colnames is None:
            table = getattr(row, "_table", None)
            colnames = getattr(table, "colnames", None)
        available = set(colnames or [])

        for name in names:
            try:
                if available and name not in available:
                    continue
                if isinstance(row, Mapping):
                    value = row.get(name)
                else:
                    value = row[name]
                if value is not None and not getattr(value, "mask", False):
                    return value
            except (KeyError, TypeError, ValueError, IndexError):
                continue
        return None

    def _frequency_ghz(self, row: Any) -> Optional[float]:
        for name in ("orderedfreq", "frequency"):
            value = self._number_or_none(self._row_value(row, name))
            if value is not None:
                return value / 1000.0
        for name in (
            "orderedFreq",
            "Freq-GHz(rest frame,redshifted)",
            "measFreq",
        ):
            value = self._number_or_none(self._row_value(row, name))
            if value is not None:
                return value
        return None

    @staticmethod
    def _clean_text(value: Any) -> str:
        if value is None or getattr(value, "mask", False):
            return ""
        text = html.unescape(_HTML_TAG_RE.sub("", str(value)))
        return _WHITESPACE_RE.sub(" ", text).strip()

    @staticmethod
    def _search_token(value: Any) -> str:
        return re.sub(r"[^a-z0-9+]+", "", str(value or "").lower())

    @staticmethod
    def _catalog_token(value: Any) -> str:
        token = SplatalogueTool._search_token(value)
        return {
            "lovas": "lovasnist",
            "recomb": "recombination",
        }.get(token, token)

    @staticmethod
    def _looks_like_formula(value: Any) -> bool:
        text = str(value or "").strip()
        return bool(
            text
            and re.fullmatch(
                r"(?:[a-z]+-)?"
                r"(?:\d*[A-Z][a-z]?\d*|\([A-Za-z0-9]+\)\d*)+"
                r"[+-]?",
                text,
            )
        )

    @staticmethod
    def _chemical_query(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        # Astroquery documents surrounding spaces as the way to request one
        # exact molecular formula instead of every species containing it.
        return f" {text} " if SplatalogueTool._looks_like_formula(text) else text

    @staticmethod
    def _number_or_none(value: Any) -> Optional[float]:
        if value is None or getattr(value, "mask", False):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    @staticmethod
    def _bool_or_none(value: Any) -> Optional[bool]:
        if value is None or getattr(value, "mask", False):
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        normalized = str(value).strip().lower()
        if normalized in {"true", "yes", "1", "on"}:
            return True
        if normalized in {"false", "no", "0", "off", ""}:
            return False
        return None

    @staticmethod
    def _ranking_strength(line: Mapping[str, Any]) -> float:
        for field in ("sijmu2", "log_intensity", "aij_log"):
            value = SplatalogueTool._number_or_none(line.get(field))
            if value is not None:
                return value
        return float("-inf")

    @staticmethod
    def _positive_float(value: Any, field: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be a number") from exc
        if not math.isfinite(result) or result <= 0:
            raise ValueError(f"{field} must be greater than zero")
        return result

    @staticmethod
    def _nonnegative_float(value: Any, field: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be a number") from exc
        if not math.isfinite(result) or result < 0:
            raise ValueError(f"{field} must be zero or greater")
        return result

    @staticmethod
    def _positive_int(value: Any, field: str) -> int:
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be an integer") from exc
        if result <= 0:
            raise ValueError(f"{field} must be greater than zero")
        return result

    @staticmethod
    def _validate_line_lists(line_lists: Optional[Sequence[str]]) -> List[str]:
        selected = list(DEFAULT_LINE_LISTS if line_lists is None else line_lists)
        invalid = sorted(set(selected) - set(DEFAULT_LINE_LISTS))
        if invalid:
            raise ValueError(
                "Unsupported line list(s): "
                + ", ".join(invalid)
                + ". Valid values: "
                + ", ".join(DEFAULT_LINE_LISTS)
            )
        return selected

    @staticmethod
    def _validate_advanced_filters(
        *,
        energy_min: Optional[float],
        energy_max: Optional[float],
        energy_type: str,
        intensity_lower_limit: Optional[float],
        intensity_type: Optional[str],
    ) -> None:
        if energy_type not in VALID_ENERGY_TYPES:
            raise ValueError(
                f"energy_type must be one of {sorted(VALID_ENERGY_TYPES)}"
            )
        if intensity_type is not None and intensity_type not in VALID_INTENSITY_TYPES:
            raise ValueError(
                f"intensity_type must be one of {sorted(VALID_INTENSITY_TYPES)}"
            )
        if intensity_lower_limit is not None and intensity_type is None:
            raise ValueError(
                "intensity_type is required when intensity_lower_limit is set"
            )
        for field, value in (
            ("energy_min", energy_min),
            ("energy_max", energy_max),
            ("intensity_lower_limit", intensity_lower_limit),
        ):
            if value is not None:
                try:
                    number = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{field} must be a number") from exc
                if not math.isfinite(number):
                    raise ValueError(f"{field} must be finite")
        if (
            energy_min is not None
            and energy_max is not None
            and float(energy_min) > float(energy_max)
        ):
            raise ValueError("energy_min must not exceed energy_max")
