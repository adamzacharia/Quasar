"""Cycle-labelled official ALMA reference tables, without network or LLM recall."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


REFERENCE_PATH = Path(__file__).resolve().parents[1] / "alma-data-skill-main" / "references" / "reference-tables.json"


def load_reference_tables() -> dict[str, Any]:
    """Keep values in the documented source artifact, never in prompt templates."""
    with REFERENCE_PATH.open(encoding="utf-8") as stream:
        return json.load(stream)


def alma_reference_table(topic: str, *, cycle: int | None = None,
                         frequency_ghz: float | None = None) -> dict[str, Any]:
    """Return a dated table; unsupported eras fail instead of borrowing new values."""
    if topic not in {"configurations", "bands", "cycles"}:
        raise ValueError("topic must be configurations, bands or cycles")
    if frequency_ghz is not None and (not math.isfinite(frequency_ghz) or frequency_ghz <= 0):
        raise ValueError("frequency_ghz must be finite and positive")
    if frequency_ghz is not None and topic != "configurations":
        raise ValueError("frequency_ghz is only valid for configurations")
    tables = load_reference_tables()
    source = tables["sources"][topic]
    result = {"topic": topic, "verified_on": tables["verified_on"],
              "sources": [source], "archive_query_executed": False}
    if topic == "configurations":
        table_cycle = tables["configuration_cycle"]
        if cycle is not None and cycle != table_cycle:
            raise ValueError(f"Configuration table is verified for Cycle {table_cycle} only; retrieve the requested cycle's handbook")
        reference_frequency = tables["configuration_reference_frequency_ghz"]
        frequency = frequency_ghz if frequency_ghz is not None else reference_frequency
        if not any(row["frequency_min_ghz"] <= frequency <= row["frequency_max_ghz"] for row in tables["bands"]):
            raise ValueError("frequency_ghz is outside the documented nominal ALMA receiver bands")
        factor = reference_frequency / frequency
        result.update(cycle=table_cycle, frequency_ghz=frequency,
                      reference_frequency_ghz=reference_frequency,
                      resolution_method="tabulated" if frequency == reference_frequency else "inverse-frequency scaling of the tabulated values",
                      configurations=[{**row,
                          "resolution_min_arcsec": round(row["resolution_min_arcsec"] * factor, 6),
                          "resolution_max_arcsec": round(row["resolution_max_arcsec"] * factor, 6)}
                          for row in tables["configuration_rows"]],
                      aca_7m_resolution_arcsec=round(tables["aca_7m_resolution_arcsec_at_100ghz"] * factor, 6),
                      total_power="Single-dish array for recovering large angular scales; not a C-configuration",
                      notes=["Smaller beams mean finer angular resolution. Compact configurations give coarser resolution than extended configurations.",
                             "These are Cycle 13 guideline ranges at source declination -23 degrees. Actual beams depend on declination, hour angle, weighting and the operational antenna distribution.",
                             "Angular resolution scales inversely with observing frequency; maximum baseline alone does not specify the synthesized beam.",
                             "A scaled value is an estimate, not confirmation that an observing mode is offered. Check the matching-cycle constraints."])
    elif topic == "bands":
        if cycle is not None and str(cycle) not in tables["offered_bands"]:
            raise ValueError("Band offering tables are verified for Cycles 12 and 13 only")
        result.update(receiver_band_count=len(tables["bands"]), bands=tables["bands"],
                      definition="Nominal receiver frequency ranges; existence does not establish commissioning, installation on every antenna, or public archive holdings.",
                      offered_bands_by_cycle=tables["offered_bands"] if cycle is None else {str(cycle): tables["offered_bands"][str(cycle)]},
                      public_archive_band_count=None,
                      public_archive_verification="Not measured by this reference tool. Query the archive for public rows, and use release-date metadata for an as-of question. Never infer public holdings from offered bands.",
                      notes=["Cycle 12 observations were scheduled for October 2025 to September 2026; Cycle 13 observations were anticipated from October 2026.",
                             "Band 2 is offered in Cycle 13 on the 12-m Array only. An accepted proposal capability is not proof of existing observations.",
                             "Near band edges, spectral-line tuning restrictions apply; the table gives nominal continuum ranges."])
        result["sources"].extend([tables["sources"]["cycle12"], tables["sources"]["cycle13"]])
    else:
        rows = [row for row in tables["cycles"] if cycle is None or row["cycle"] == cycle]
        if not rows:
            raise ValueError("No verified reference entry for that cycle")
        result.update(cycles=rows, notes=["Project-code years are not a simple cycle-plus-year formula: no 2014 cycle code and no 2020.1 code.",
                                          "A cycle/project code does not establish processing version, data release date, or public holdings."])
        result["sources"].extend([tables["sources"]["cycle12"], tables["sources"]["cycle13"]])
    return result
