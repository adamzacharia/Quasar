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
import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


SPLATALOGUE_SLAP_URL = "https://splatalogue.online/splata-slap/slap"
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


class SplatalogueTool:
    """Query and normalize molecular spectral-line data from Splatalogue."""

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
        }
        if errors:
            response["note"] = (
                "The Astroquery request failed, so results were obtained from "
                "Splatalogue's IVOA SLAP endpoint."
            )
            response["details"] = errors
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
        from astropy import units as u
        from astroquery.splatalogue import Splatalogue

        kwargs: Dict[str, Any] = {
            "chemical_name": SplatalogueTool._chemical_query(molecule_name),
            "transition": transition,
            "energy_min": energy_min,
            "energy_max": energy_max,
            "energy_type": energy_type,
            "intensity_lower_limit": intensity_lower_limit,
            "intensity_type": intensity_type,
            "version": version,
            "exclude": exclude,
            "line_lists": list(line_lists),
            "only_astronomically_observed": only_astronomically_observed,
            "only_NRAO_recommended": only_nrao_recommended,
            "export_stop": export_stop,
        }
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        return Splatalogue.query_lines(
            freq_min_ghz * u.GHz,
            freq_max_ghz * u.GHz,
            **kwargs,
        )

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

        line: Dict[str, Any] = {
            "species": species,
            "name": species,
            "molecule": molecule,
            "chemical_name": chemical_name or molecule,
            "transition": transition,
            "frequency_ghz": round(frequency_ghz, 9),
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
            "aij_log": self._number_or_none(
                self._row_value(row, "aij", "Log<sub>10</sub> (A<sub>ij</sub>)")
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
        }
        if query_frequency_ghz is not None:
            line["offset_mhz"] = round(
                (frequency_ghz - query_frequency_ghz) * 1000.0,
                6,
            )
        return line

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
        merged: Dict[tuple, Dict[str, Any]] = {}
        for line in lines:
            key = (
                SplatalogueTool._search_token(
                    line.get("species") or line.get("molecule")
                ),
                SplatalogueTool._search_token(line.get("transition")),
                round(float(line["frequency_ghz"]), 6),
            )
            existing = merged.get(key)
            if existing is None:
                merged[key] = dict(line)
                continue

            catalogs = list(existing.get("catalogs") or [])
            for catalog in line.get("catalogs") or []:
                if catalog and catalog not in catalogs:
                    catalogs.append(catalog)
            existing["catalogs"] = catalogs
            existing["source"] = ", ".join(catalogs) if catalogs else "Splatalogue"
            for field, value in line.items():
                if existing.get(field) in (None, "", [], "N/A") and value not in (
                    None,
                    "",
                    [],
                    "N/A",
                ):
                    existing[field] = value
        return list(merged.values())

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
