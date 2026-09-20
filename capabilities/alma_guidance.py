"""
capabilities/alma_guidance.py — on-demand ALMA reference guidance (INT-6/INT-7).

``browse_alma_guidance(topic)`` serves a named section of the vendored
"Working with ALMA data" skill (``third_party/alma-data-skill``, reviewed
2026-07-18) with its review date attached and a size cap. This is the
progressive-disclosure pattern the skill is written for: the always-on
system prompt carries only a ~300-token guardrail kernel (core/prompts), the
1,400 lines of references are never pasted into the prompt or chunked into
the RAG store, and the model pulls the section it needs when it needs it
(mosaics, moving targets, Total Power, polarization, band-to-band, VLBI,
restore, pipeline history, ...).

``get_alma_qa2_status(mous_uid)`` resolves the three-state QA2 disposition
(PASS / SEMIPASS / FAIL) from the archive's QA2 report PDF, which the ObsCore
``qa2_passed`` flag cannot encode (INT-4).

Transport-pure: file reads + one optional HTTP fetch; no thread-locals.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict

from capabilities.base import BaseCapability, Provenance, ToolResult

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILL_ROOT = Path(os.getenv("QUASAR_ALMA_SKILL_DIR", REPO_ROOT / "third_party" / "alma-data-skill"))
MAX_SECTION_CHARS = int(os.getenv("QUASAR_ALMA_GUIDANCE_MAX_CHARS", "6000"))

# Closed topic list -> (file, section headings to include; [] = whole file).
TOPICS: Dict[str, Dict[str, Any]] = {
    "hierarchy-identifiers": {
        "summary": "Project / SG-OUS / GOUS / MOUS / EB hierarchy, UID and project-code grammar",
        "file": "SKILL.md",
        "sections": ["The data hierarchy (memorize this)", "UID and project-code grammar"],
    },
    "guardrails": {
        "summary": "The eleven non-negotiable guardrails for ALMA archive and data work",
        "file": "SKILL.md",
        "sections": ["Non-negotiable guardrails"],
    },
    "query-units-footprints": {
        "summary": "ObsCore row grain, column units/traps, s_region footprint cones, ADQL patterns",
        "file": "references/archive-query.md",
        "sections": ["Row granularity — the #1 query mistake", "High-value columns", "ADQL patterns"],
    },
    "frequency-support": {
        "summary": "frequency_support grammar, spectral coverage, derived quantities",
        "file": "references/archive-query.md",
        "sections": ["`frequency_support` grammar", "Derived quantities — label them honestly"],
    },
    "datalink-packaging": {
        "summary": "What DataLink returns per MOUS, packaging eras, tar nesting trap, package tree",
        "file": "references/identifiers-and-packaging.md",
        "sections": ["What DataLink offers per MOUS", "Packaging eras — separate tar grouping from the logical tree",
                     "The nesting trap", "The logical Cycle-5+ full QA2 package"],
    },
    "download-preflight": {
        "summary": "DataLink states (empty vs not found), byte preflight, endpoints and mirrors",
        "file": "references/archive-query.md",
        "sections": ["Endpoints", "Download preflight and DataLink states"],
    },
    "products-qa-restore": {
        "summary": "FITS product naming, product completeness, QA ladder, QA artifacts, calibrated-MS restore (scriptForPI)",
        "file": "references/products-and-qa.md",
        "sections": ["Product FITS naming token families", "Products are not the complete science content",
                     "The QA ladder", "Getting a calibrated MeasurementSet", "Cross-checking \"which EBs made it\""],
    },
    "pipeline-artifacts": {
        "summary": "Weblog, AQUA report, PPR, manifest, recipe and run artifacts",
        "file": "references/products-and-qa.md",
        "sections": ["Machine-readable QA artifacts", "Pipeline recipes and run artifacts"],
    },
    "asdm-ms-spectral-frames": {
        "summary": "ASDM raw format, SPW identities, MS directories, DATA/CORRECTED_DATA, TOPO Doppler setting vs LSRK",
        "file": "references/asdm-and-ms.md",
        "sections": [],
    },
    "listobs-intents-spws": {
        "summary": "listobs parsing, MS views per EB, scan intents, SPW name grammar, array inference from antenna prefixes",
        "file": "references/listobs-and-intents.md",
        "sections": [],
    },
    "mosaic-moving-tp": {
        "summary": "FIELD vs SOURCE, mosaics, ephemeris/moving targets, solar, Total Power, 12m+7m+TP combination",
        "file": "references/mosaics-ephemeris-and-tp.md",
        "sections": [],
    },
    "polarization-b2b-vlbi": {
        "summary": "Receiver bands, band-to-band, arrays/configurations, correlator modes, polarization and VLBI capability notes",
        "file": "references/cycle-capabilities.md",
        "sections": ["Receiver bands", "Arrays and configurations", "Correlator / spectral setup"],
    },
    "cycle-capabilities": {
        "summary": "Cycle <-> project-code year table, receiver bands, arrays, correlator, pipeline/CASA coupling",
        "file": "references/cycle-capabilities.md",
        "sections": [],
    },
    "pipeline-history": {
        "summary": "Pipeline/CASA operations release matrix, capability milestones, package traps by release",
        "file": "references/pipeline-history.md",
        "sections": [],
    },
}

_REVIEW_DATE_RE = re.compile(r"\b(?:Reviewed|reviewed|Systematically reviewed)\s+(\d{4}-\d{2}-\d{2})")
_H2_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


def skill_available() -> bool:
    return (SKILL_ROOT / "SKILL.md").exists()


def upstream_metadata() -> Dict[str, Any]:
    path = SKILL_ROOT / "UPSTREAM.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _review_date(text: str) -> Optional[str]:
    match = _REVIEW_DATE_RE.search(text)
    return match.group(1) if match else None


def _split_sections(text: str) -> List[Tuple[str, str]]:
    """[(heading, body)] for every '## ' section; the preamble has heading ''."""
    parts: List[Tuple[str, str]] = []
    matches = list(_H2_RE.finditer(text))
    if not matches:
        return [("", text)]
    parts.append(("", text[: matches[0].start()]))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        parts.append((match.group(1).strip(), text[match.start():end]))
    return parts


def _norm_heading(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def load_topic(topic: str, max_chars: int = MAX_SECTION_CHARS) -> Dict[str, Any]:
    """Return the guidance payload for one topic (pure function; raises KeyError
    for an unknown topic, FileNotFoundError when the vendored skill is missing)."""
    key = str(topic or "").strip().lower().replace("_", "-").replace(" ", "-")
    if key not in TOPICS:
        raise KeyError(key)
    spec = TOPICS[key]
    path = SKILL_ROOT / spec["file"]
    if not path.exists():
        raise FileNotFoundError(str(path))
    text = path.read_text(encoding="utf-8")
    wanted = [_norm_heading(h) for h in spec["sections"]]
    if wanted:
        chosen = [body for heading, body in _split_sections(text) if _norm_heading(heading) in wanted]
        content = "\n".join(chosen).strip()
        if not content:  # heading drift upstream: fall back to the whole file, disclosed
            content = text
            drift = True
        else:
            drift = False
    else:
        content = text
        drift = False
    truncated = len(content) > max_chars
    if truncated:
        content = content[:max_chars].rstrip() + "\n\n[... truncated at size cap; ask for a narrower topic ...]"
    meta = upstream_metadata()
    return {
        "topic": key,
        "summary": spec["summary"],
        "source_file": f"third_party/alma-data-skill/{spec['file']}",
        "review_date": _review_date(text) or meta.get("review_date"),
        "upstream": {k: meta.get(k) for k in ("source", "revision", "vendored_on") if meta.get(k)},
        "sections": spec["sections"] or ["(whole file)"],
        "section_heading_drift": drift,
        "truncated": truncated,
        "chars": len(content),
        "content": content,
        "caveat": (
            "Dated empirical guidance (live-service checks as of the review date), not an IVOA/ALMA "
            "interface guarantee. Re-verify time-sensitive claims against the current archive."
        ),
    }


class BrowseAlmaGuidanceInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    topic: str


class BrowseAlmaGuidance(BaseCapability):
    name = "browse_alma_guidance"
    description = (
        "Read one section of the reviewed 'Working with ALMA data' reference (dated, vendored) "
        "BEFORE answering a question that needs ALMA data-model detail beyond the prompt kernel: "
        "mosaics, moving/solar targets, Total Power, polarization, band-to-band, VLBI, DataLink "
        "packaging and tar nesting, QA2/restore (scriptForPI, CASA version), spectral frames "
        "(TOPO vs LSRK), listobs/intents/SPW names, cycle/band/array capabilities, pipeline history. "
        "Topics: " + ", ".join(TOPICS.keys()) + "."
    )
    category = "archive"
    InputModel = BrowseAlmaGuidanceInput
    annotations = {"read_only": True, "cost": "cheap"}
    json_schema = {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "enum": list(TOPICS.keys()),
                "description": "Guidance topic: " + "; ".join(f"{k} = {v['summary']}" for k, v in TOPICS.items()),
            },
        },
        "required": ["topic"],
    }

    def run(self, inp: BrowseAlmaGuidanceInput, ctx) -> ToolResult:
        provenance = Provenance(
            service="alma_guidance",
            tool_name=self.name,
            query=f"browse_alma_guidance({inp.topic!r})",
            endpoint=str(SKILL_ROOT),
            retrieved_at=getattr(ctx, "now", None),
        )
        try:
            payload = load_topic(inp.topic)
        except KeyError:
            return ToolResult.fail(
                f"Unknown topic {inp.topic!r}. Topics: {', '.join(TOPICS.keys())}"
            )
        except FileNotFoundError as exc:
            return ToolResult.fail(
                f"The vendored ALMA data skill is not available ({exc}); run "
                "scripts/sync_alma_skill.py or set QUASAR_ALMA_SKILL_DIR."
            )
        return ToolResult(success=True, data=payload, provenance=provenance)


class GetAlmaQa2StatusInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    mous_uid: str


class GetAlmaQa2Status(BaseCapability):
    name = "get_alma_qa2_status"
    description = (
        "Resolve the THREE-STATE QA2 disposition (PASS / SEMIPASS / FAIL) of one ALMA MOUS from its "
        "archived QA2 report PDF. The ObsCore qa2_passed flag is only T/F and cannot distinguish "
        "SEMIPASS from PASS or FAIL — call this when a user asks whether a dataset passed QA2, or "
        "before recommending a MOUS as science-ready. Returns Unknown (with the report URL) when the "
        "report is unreachable; never guesses."
    )
    category = "archive"
    InputModel = GetAlmaQa2StatusInput
    annotations = {"read_only": True, "cost": "network"}
    json_schema = {
        "type": "object",
        "properties": {
            "mous_uid": {"type": "string", "description": "Member OUS UID, e.g. uid://A001/X12a3/X407"},
        },
        "required": ["mous_uid"],
    }

    def run(self, inp: GetAlmaQa2StatusInput, ctx) -> ToolResult:
        from services import alma_qa2

        normalized = alma_qa2.normalize_mous_uid(inp.mous_uid)
        if not normalized:
            return ToolResult.fail(f"Not a MOUS UID: {inp.mous_uid!r}")
        url = alma_qa2.qa2_report_url(normalized)
        provenance = Provenance(service="alma_qa2", tool_name=self.name, endpoint=url,
                                query=f"GET {url}", retrieved_at=getattr(ctx, "now", None))
        cached = alma_qa2.cached_statuses([normalized])
        if cached.get(normalized) in alma_qa2.QA2_REPORT_LABELS:
            status = cached[normalized]
            source = "qa2_report (cached)"
            error = ""
        else:
            result = alma_qa2.fetch_qa2_status(normalized)
            status = result.status or alma_qa2.QA2_UNKNOWN
            error = result.error
            source = "qa2_report"
            if status in alma_qa2.QA2_REPORT_LABELS and not error:
                alma_qa2._write_cached_statuses([result])
        payload = {
            "mous_uid": inp.mous_uid,
            "normalized_uid": normalized,
            "qa2_status": status,
            "source": source if status != alma_qa2.QA2_UNKNOWN else "unresolved",
            "report_url": url,
            "error": error or None,
            "meaning": {
                "Pass": "QA2 PASS: the MOUS met the PI's requirements.",
                "SemiPass": "QA2 SEMIPASS: delivered with documented shortfalls; read the QA2 report and README before science use.",
                "Fail": "QA2 FAIL: not accepted; usually not delivered as a science package.",
                "Unknown": "Could not read the QA2 report; do NOT infer the disposition from the qa2_passed flag.",
            }.get(status, ""),
            "flag_caveat": "ObsCore qa2_passed (T/F) cannot encode SEMIPASS; only the QA2 report can.",
        }
        return ToolResult(success=True, data=payload, provenance=provenance,
                          degraded=(status == alma_qa2.QA2_UNKNOWN))


CAPABILITIES = [BrowseAlmaGuidance(), GetAlmaQa2Status()]

__all__ = ["CAPABILITIES", "BrowseAlmaGuidance", "GetAlmaQa2Status", "TOPICS", "load_topic",
           "skill_available", "upstream_metadata", "SKILL_ROOT"]
