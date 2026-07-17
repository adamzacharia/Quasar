"""
services/archive_profiles/schema.py — the ArchiveProfile contract.

Converged design from codex-bridge duel task-81b3b73-2145 (2 rounds + a
two-model arbiter panel; see tmp/codex/tasks/task-81b3b73-2145/final.md).
The load-bearing decisions:

* Profiles are CODE (strict frozen pydantic v2 models) — no DB.
* Non-tabular archives (ADS/OpenAlex, SIA/HiPS, VO, Splatalogue) set
  ``tables=None`` (omitted when serialized), and describe their query surface
  through ``query_surfaces`` instead of a forced empty table dict.
* ``QuerySurface.parameters`` is a curated SEMANTIC SUBSET of the registered
  tool's JSON schema (units/conventions/traps only) — the model already
  receives every tool's full schema, so duplicating it would only drift.
* Units are a flattened four-state contract (``unit``/``unit_state``/
  ``unit_note``): known / dimensionless ("1") / not_applicable / unknown.
  Numeric columns and parameters must resolve a state; ``unknown`` demands a
  dedicated ``unit_note`` so the test suite has a checkable invariant.
* Golden examples MUST carry a registered tool invocation — DataLabBench
  scores executed tool arguments and SQL, not prose. Raw SQL/ADQL/URL rides
  along as typed supplemental evidence.
* Profiles are phrased as canonical/common, never exhaustive (pinned
  ``scope_note``), so grounding does not bias the model away from valid
  unlisted columns.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Tuple

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, model_validator

ArchiveSlug = Literal["datalab", "alma", "ads_openalex", "sia_hips", "vo", "splatalogue"]

ScalarType = Literal[
    "integer", "float", "decimal", "string", "boolean", "datetime", "array", "object"
]

UnitState = Literal["known", "dimensionless", "not_applicable", "unknown"]

RequestKind = Literal[
    "structured_args", "natural_language", "sql", "adql", "http_get", "parameter_service"
]

_NUMERIC_SCALARS = {"integer", "float", "decimal"}
_NUMERIC_JSON_TYPES = {"number", "integer"}

# Per archive, at most this many pitfalls may carry a prompt_rank — the
# prompt index budget (one line per archive) is bounded by construction.
PROMPT_RANKED_MAX = 2


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class UnitFields(StrictModel):
    """Flattened four-state unit contract shared by columns and parameters.

    Authoring stays one token in the common case (``unit="mag"``); the states
    exist so a missing unit is always a CONSCIOUS decision, never an omission.
    """

    unit: Optional[str] = None
    unit_state: Optional[UnitState] = None
    unit_note: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_units(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        unit = data.get("unit")
        state = data.get("unit_state")
        note = data.get("unit_note")

        if isinstance(unit, str):
            unit = unit.strip()
            data["unit"] = unit

        inferred: Optional[str] = None
        if unit:
            inferred = "dimensionless" if unit == "1" else "known"
        elif state == "dimensionless":
            # Authored state without the symbol — canonicalize to "1".
            data["unit"] = "1"
            inferred = "dimensionless"

        if state is not None and inferred is not None and state != inferred:
            raise ValueError(
                f"unit_state={state!r} contradicts the state {inferred!r} inferred from unit={unit!r}"
            )
        state = state or inferred
        data["unit_state"] = state

        if state == "known":
            from astropy.units import Unit  # runtime dep (requirements.txt)

            try:
                Unit(unit)
            except Exception as exc:
                raise ValueError(f"unit {unit!r} does not parse as an astropy/VOUnit symbol: {exc}")
        if state == "not_applicable" and unit:
            raise ValueError("unit_state='not_applicable' forbids a unit symbol")
        if state == "unknown":
            if unit:
                raise ValueError("unit_state='unknown' forbids a unit symbol")
            if not (isinstance(note, str) and note.strip()):
                raise ValueError("unit_state='unknown' requires a nonblank unit_note")
        elif note is not None:
            raise ValueError("unit_note is only allowed when unit_state='unknown'")
        return data


class CitationSpec(StrictModel):
    id: str
    text: str
    url: Optional[AnyHttpUrl] = None
    doi: Optional[str] = None


class EndpointSpec(StrictModel):
    id: str
    description: str
    url: AnyHttpUrl
    protocol: Literal["tap", "sia", "hips2fits", "rest", "line_list"]
    citation_ids: Tuple[str, ...] = ()


class ParameterSpec(UnitFields):
    """One tool parameter that carries semantic grounding the JSON schema
    cannot (a unit, an omission rule, a trap). Never a schema copy."""

    name: str
    json_type: Literal["string", "number", "integer", "boolean", "array", "object"]
    description: str
    allowed_values: Optional[Tuple[Any, ...]] = None
    omit_when_unspecified: Optional[bool] = None

    @model_validator(mode="after")
    def _numeric_requires_unit_state(self) -> "ParameterSpec":
        if self.json_type in _NUMERIC_JSON_TYPES and self.unit_state is None:
            raise ValueError(
                f"numeric parameter {self.name!r} must resolve a unit_state "
                "(unit=..., unit='1', not_applicable, or unknown+unit_note)"
            )
        return self


class QuerySurface(StrictModel):
    id: str
    purpose: str
    tool: str
    request_kind: RequestKind
    query_argument: Optional[str] = None
    parameters: Tuple[ParameterSpec, ...] = ()
    endpoint_ids: Tuple[str, ...] = ()
    citation_ids: Tuple[str, ...] = ()

    @model_validator(mode="after")
    def _query_argument_required(self) -> "QuerySurface":
        if self.request_kind in ("natural_language", "sql", "adql") and not self.query_argument:
            raise ValueError(
                f"surface {self.id!r}: request_kind={self.request_kind!r} requires query_argument"
            )
        return self


class ColumnSpec(UnitFields):
    name: str
    dtype: ScalarType
    description: str
    role: Optional[
        Literal[
            "identifier", "ra", "dec", "time", "measurement",
            "uncertainty", "quality", "category", "healpix", "provenance",
        ]
    ] = None

    @model_validator(mode="after")
    def _numeric_requires_unit_state(self) -> "ColumnSpec":
        if self.dtype in _NUMERIC_SCALARS and self.unit_state is None:
            raise ValueError(
                f"numeric column {self.name!r} must resolve a unit_state "
                "(unit=..., unit='1', not_applicable, or unknown+unit_note)"
            )
        return self


class HealpixColumnSpec(StrictModel):
    name: str
    nside: int
    scheme: Literal["RING", "NEST"]


class TableQueryHints(StrictModel):
    footprint: Optional[str] = None
    region_strategy: Optional[str] = None
    healpix_columns: Tuple[HealpixColumnSpec, ...] = ()
    morphology: Dict[str, str] = Field(default_factory=dict)
    bitmasks: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    aggregate_safe: Optional[bool] = None
    endpoint_ids: Tuple[str, ...] = ()


class TableProfile(StrictModel):
    purpose: str
    grain: str
    columns: Tuple[ColumnSpec, ...] = Field(min_length=1)
    ra_column: Optional[str] = None
    dec_column: Optional[str] = None
    hints: TableQueryHints = TableQueryHints()
    citation_ids: Tuple[str, ...] = ()

    @model_validator(mode="after")
    def _radec_reference_columns(self) -> "TableProfile":
        if (self.ra_column is None) != (self.dec_column is None):
            raise ValueError("ra_column and dec_column must be authored together or not at all")
        names = {c.name for c in self.columns}
        for coord in (self.ra_column, self.dec_column):
            if coord is not None and coord not in names:
                raise ValueError(f"coordinate column {coord!r} is not among the listed columns")
        return self


class ProfileRef(StrictModel):
    """Typed scope reference. ``column`` refs use "<table_key>:<column>";
    ``parameter`` refs use "<surface_id>:<parameter>"."""

    kind: Literal["archive", "surface", "table", "column", "parameter"]
    ref: str


class Pitfall(StrictModel):
    id: str
    # Single-line prompt wording. The 160-char ceiling is load-bearing: with
    # at most PROMPT_RANKED_MAX ranked pitfalls per archive (see
    # ArchiveProfile) it bounds every prompt-index line under ~500 chars by
    # construction, so the injected block cannot bloat (guard CX-07).
    summary: str = Field(max_length=160)
    detail: Optional[str] = None      # fuller browse_schema explanation
    applies_to: Tuple[ProfileRef, ...] = Field(min_length=1)
    prompt_rank: Optional[int] = None

    @model_validator(mode="after")
    def _rank_positive(self) -> "Pitfall":
        if self.prompt_rank is not None and self.prompt_rank < 1:
            raise ValueError("prompt_rank must be a positive integer")
        return self


class ToolInvocation(StrictModel):
    tool: str
    arguments: Dict[str, Any]


class SqlRequestEvidence(StrictModel):
    kind: Literal["sql", "adql"]
    argument: str  # key in ToolInvocation.arguments holding the query text


class UrlRequestEvidence(StrictModel):
    kind: Literal["url"]
    url: AnyHttpUrl  # fully resolved example URL, never a template


RequestEvidence = SqlRequestEvidence | UrlRequestEvidence


class GoldenExample(StrictModel):
    id: str
    intent: str
    invocation: ToolInvocation        # mandatory — the benchmark scores executed args
    request: Optional[RequestEvidence] = Field(default=None, discriminator="kind")
    note: Optional[str] = None

    @model_validator(mode="after")
    def _sql_evidence_resolves(self) -> "GoldenExample":
        if isinstance(self.request, SqlRequestEvidence):
            value = self.invocation.arguments.get(self.request.argument)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"golden {self.id!r}: request.argument {self.request.argument!r} must name "
                    "a nonblank string in invocation.arguments"
                )
        return self


class UnitConvention(StrictModel):
    id: str
    statement: str
    applies_to: Tuple[ProfileRef, ...] = ()


class ArchiveProfile(StrictModel):
    archive: ArchiveSlug
    aliases: Tuple[str, ...] = ()
    description: str
    scope_note: Literal["Canonical/common fields and parameters; not exhaustive."] = (
        "Canonical/common fields and parameters; not exhaustive."
    )

    endpoints: Tuple[EndpointSpec, ...] = ()
    query_surfaces: Tuple[QuerySurface, ...] = Field(min_length=1)
    tables: Optional[Dict[str, TableProfile]] = None  # None for non-tabular archives

    pitfalls: Tuple[Pitfall, ...] = Field(min_length=1)
    golden_examples: Tuple[GoldenExample, ...] = Field(min_length=3)
    unit_conventions: Tuple[UnitConvention, ...] = ()
    citations: Tuple[CitationSpec, ...] = ()

    @model_validator(mode="after")
    def _cross_references_resolve(self) -> "ArchiveProfile":
        def _unique(kind: str, ids: list) -> set:
            dupes = {i for i in ids if ids.count(i) > 1}
            if dupes:
                raise ValueError(f"duplicate {kind} ids: {sorted(dupes)}")
            return set(ids)

        endpoint_ids = _unique("endpoint", [e.id for e in self.endpoints])
        citation_ids = _unique("citation", [c.id for c in self.citations])
        surface_ids = _unique("surface", [s.id for s in self.query_surfaces])
        _unique("pitfall", [p.id for p in self.pitfalls])
        _unique("golden example", [g.id for g in self.golden_examples])
        ranks = [p.prompt_rank for p in self.pitfalls if p.prompt_rank is not None]
        _unique("prompt_rank", ranks)
        if len(ranks) > PROMPT_RANKED_MAX:
            raise ValueError(
                f"at most {PROMPT_RANKED_MAX} pitfalls may carry a prompt_rank "
                f"(got {len(ranks)}) — the prompt-index line budget is bounded by construction"
            )

        table_keys = set((self.tables or {}).keys())

        def _check_ids(owner: str, kind: str, refs: Tuple[str, ...], pool: set) -> None:
            for ref in refs:
                if ref not in pool:
                    raise ValueError(f"{owner}: unresolved {kind} id {ref!r}")

        for surface in self.query_surfaces:
            _check_ids(f"surface {surface.id!r}", "endpoint", surface.endpoint_ids, endpoint_ids)
            _check_ids(f"surface {surface.id!r}", "citation", surface.citation_ids, citation_ids)
        for endpoint in self.endpoints:
            _check_ids(f"endpoint {endpoint.id!r}", "citation", endpoint.citation_ids, citation_ids)
        for key, table in (self.tables or {}).items():
            _check_ids(f"table {key!r}", "citation", table.citation_ids, citation_ids)
            _check_ids(f"table {key!r}", "endpoint", table.hints.endpoint_ids, endpoint_ids)

        param_refs = {
            f"{s.id}:{p.name}" for s in self.query_surfaces for p in s.parameters
        }
        column_refs = {
            f"{key}:{c.name}" for key, t in (self.tables or {}).items() for c in t.columns
        }

        def _check_profile_refs(owner: str, refs: Tuple[ProfileRef, ...]) -> None:
            for r in refs:
                ok = (
                    (r.kind == "archive" and r.ref == self.archive)
                    or (r.kind == "surface" and r.ref in surface_ids)
                    or (r.kind == "table" and r.ref in table_keys)
                    or (r.kind == "column" and r.ref in column_refs)
                    or (r.kind == "parameter" and r.ref in param_refs)
                )
                if not ok:
                    raise ValueError(f"{owner}: unresolved {r.kind} ref {r.ref!r}")

        for pitfall in self.pitfalls:
            _check_profile_refs(f"pitfall {pitfall.id!r}", pitfall.applies_to)
        for convention in self.unit_conventions:
            _check_profile_refs(f"unit_convention {convention.id!r}", convention.applies_to)
        return self

    def dump(self) -> Dict[str, Any]:
        """Canonical serialization: non-tabular archives carry no ``tables`` key."""
        return self.model_dump(mode="json", exclude_none=True)

    def prompt_pitfalls(self) -> list:
        """Prompt-ranked pitfall summaries, ascending rank."""
        ranked = [p for p in self.pitfalls if p.prompt_rank is not None]
        return [p.summary for p in sorted(ranked, key=lambda p: p.prompt_rank)]
