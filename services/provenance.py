"""Claim provenance ledger (increment 1).

Collects the concrete source pointers already present in Conductor sub-agent
results — ALMA project codes, ADS bibcodes, Splatalogue line rows, RAG chunks,
and web sources — dedupes them, and renders a verifiable ``## Sources`` appendix.

This is the safe, additive foundation for per-claim provenance: it does not alter
the synthesis model's text, only appends a grouped source list. Per-sentence claim
binding and SSE/UI wiring are intentionally deferred to a later increment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ALMA project/proposal code, e.g. 2021.1.00123.S
_PROJECT_CODE_RE = re.compile(r"\b\d{4}\.\d\.\d{5}\.[A-Z]\b")

# Order in which source kinds are rendered, with human-readable section titles.
_KIND_ORDER: Tuple[Tuple[str, str], ...] = (
    ("project_code", "ALMA projects"),
    ("ads_bibcode", "Literature (ADS)"),
    ("splatalogue", "Spectral lines (Splatalogue)"),
    ("rag_chunk", "Documentation"),
    ("web", "Web sources"),
    ("notebook_cell", "Computation"),
)
_KNOWN_KINDS = {kind for kind, _ in _KIND_ORDER}


@dataclass(frozen=True)
class SourceRecord:
    kind: str
    id: str
    label: str
    ref: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class ProvenanceLedger:
    """Collects and de-duplicates source pointers harvested from tool results."""

    def __init__(self) -> None:
        self._records: List[SourceRecord] = []
        self._seen: set[Tuple[str, str]] = set()

    # -- collection ----------------------------------------------------
    def add(self, record: Optional[SourceRecord]) -> None:
        if record is None or not record.id:
            return
        key = (record.kind, record.id.lower())
        if key in self._seen:
            return
        self._seen.add(key)
        self._records.append(record)

    def add_many(self, records: Iterable[SourceRecord]) -> None:
        for record in records:
            self.add(record)

    def extract_from_results(self, results: Any) -> None:
        """Walk an arbitrary Conductor results structure, harvesting sources.

        Defensive by design: tolerates dicts, lists, strings, and arbitrary
        nesting; never raises on malformed input.
        """
        try:
            self._walk(results, depth=0)
        except Exception:  # pragma: no cover - belt-and-suspenders
            return

    # -- rendering -----------------------------------------------------
    def is_empty(self) -> bool:
        return not self._records

    def to_list(self) -> List[Dict[str, Any]]:
        return [
            {
                "kind": r.kind,
                "id": r.id,
                "label": r.label,
                "ref": r.ref,
                **({"extra": r.extra} if r.extra else {}),
            }
            for r in self._records
        ]

    def render_markdown(self) -> str:
        if not self._records:
            return ""
        by_kind: Dict[str, List[SourceRecord]] = {}
        for record in self._records:
            by_kind.setdefault(record.kind, []).append(record)

        lines: List[str] = ["## Sources"]
        for kind, title in _KIND_ORDER:
            group = by_kind.get(kind)
            if not group:
                continue
            lines.append("")
            lines.append(f"**{title}**")
            for record in group:
                if record.ref:
                    lines.append(f"- {record.label} ({record.ref})")
                else:
                    lines.append(f"- {record.label}")
        return "\n".join(lines)

    # -- internals -----------------------------------------------------
    def _walk(self, node: Any, depth: int) -> None:
        if depth > 12:
            return
        if isinstance(node, str):
            self._harvest_text(node)
            return
        if isinstance(node, dict):
            self._harvest_dict(node)
            for value in node.values():
                self._walk(value, depth + 1)
            return
        if isinstance(node, (list, tuple, set)):
            for value in node:
                self._walk(value, depth + 1)
            return

    def _harvest_text(self, text: str) -> None:
        for code in _PROJECT_CODE_RE.findall(text):
            self.add(
                SourceRecord(
                    kind="project_code",
                    id=code,
                    label=f"ALMA {code}",
                    ref=None,
                )
            )
        for bibcode in _extract_bibcodes(text):
            self.add(
                SourceRecord(
                    kind="ads_bibcode",
                    id=bibcode,
                    label=bibcode,
                    ref=f"https://ui.adsabs.harvard.edu/abs/{bibcode}",
                )
            )

    def _harvest_dict(self, node: Dict[str, Any]) -> None:
        # RAG chunk: a document chunk carrying a source filename.
        source_file = node.get("source_file") or node.get("source")
        if isinstance(source_file, str) and source_file.strip():
            name = source_file.replace("\\", "/").split("/")[-1]
            year = node.get("doc_year")
            label = f"{name} ({year})" if year not in (None, "", "?") else name
            self.add(
                SourceRecord(
                    kind="rag_chunk",
                    id=f"{name}|{year}" if year not in (None, "", "?") else name,
                    label=label,
                    ref=None,
                    extra={"doc_year": year} if year not in (None, "") else {},
                )
            )

        # Splatalogue line row: a species/formula plus a frequency.
        species = node.get("species") or node.get("formula")
        freq = (
            node.get("frequency_ghz")
            or node.get("rest_frequency_ghz")
            or node.get("frequency")
        )
        if isinstance(species, str) and species.strip() and freq not in (None, ""):
            transition = node.get("transition")
            label = f"{species}"
            if transition:
                label += f" {transition}"
            label += f" @ {freq} GHz"
            self.add(
                SourceRecord(
                    kind="splatalogue",
                    id=f"{species}|{transition}|{freq}",
                    label=label,
                    ref=None,
                )
            )

        # Web source: a dict carrying a URL.
        url = node.get("url")
        if isinstance(url, str) and url.strip():
            title = node.get("title") or node.get("snippet") or url
            title = str(title).strip()
            if len(title) > 120:
                title = title[:117] + "..."
            self.add(
                SourceRecord(
                    kind="web",
                    id=url,
                    label=title,
                    ref=url,
                )
            )


def _extract_bibcodes(text: str) -> List[str]:
    """Best-effort bibcode extraction, reusing the citation verifier if present."""
    try:
        from services.citation_verifier import extract_citations

        return list(extract_citations(text).get("bibcodes", []))
    except Exception:
        # Item 5 must not depend on Item 2 being present.
        return []


def build_sources_appendix(results: Any) -> str:
    """Convenience: harvest sources from results and render the appendix."""
    ledger = ProvenanceLedger()
    ledger.extract_from_results(results)
    return ledger.render_markdown()
