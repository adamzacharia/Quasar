# tests/unit/test_archive_profiles.py
"""Author-time gates for the per-archive grounding profiles (Feature 3).

Validation contract from the codex-bridge duel task-81b3b73-2145 final.md:
exactly six curated profiles; every curated column authored (dtype + resolved
unit state, no floor); surface tools exist with subset-compatible parameters;
>=3 golden examples per archive whose invocations validate against the LIVE
registered tool JSON schemas; SQL/ADQL goldens parse as one read-only
statement via sqlglot (Data Lab SQL additionally passes the governor); URLs
well-formed; projections under their byte caps; prompt index gated by
QUASAR_SCHEMA_GROUNDING.

All offline. The "live registry" is the real agent tool registry built the
same way tests/unit/test_vo_capability.py does.
"""

from __future__ import annotations

import json
import threading
import urllib.parse
from typing import get_args

import jsonschema
import pytest
import sqlglot
from sqlglot import expressions as sqlexp

from services import archive_profiles
from services import datalab_registry
from services import datalab_sql_policy
from services.archive_profiles import projections
from services.archive_profiles.schema import (
    ArchiveProfile,
    ArchiveSlug,
    SqlRequestEvidence,
    UrlRequestEvidence,
)

# The ArchiveSlug Literal IS the roster contract — deriving the expectation
# from it means adding an archive touches exactly two places (the profile
# module + the slug), never this test (guard CX-01 verify).
EXPECTED_ARCHIVES = set(get_args(ArchiveSlug))
NUMERIC_DTYPES = {"integer", "float", "decimal"}
NUMERIC_JSON_TYPES = {"number", "integer"}


# ── the live tool registry (real agent wiring) ────────────────────────────────
@pytest.fixture(scope="module")
def tool_registry():
    from core.tools import ToolRegistry
    from tests.integration.test_agent_archive_tools import _load_agent_module

    module = _load_agent_module()
    agent = module.QuasarAgent.__new__(module.QuasarAgent)
    agent._tls = threading.local()
    agent.tool_registry = ToolRegistry()
    agent.last_search_results = None
    agent.last_run_result = None
    agent.ads_client = None
    agent.openalex_client = None
    agent._register_tools()
    return agent.tool_registry


def _profiles():
    return [archive_profiles.get_profile(slug) for slug in sorted(EXPECTED_ARCHIVES)]


# ── 1. registry completeness ─────────────────────────────────────────────────
def test_every_declared_archive_has_a_profile():
    # Auto-discovery must surface a profile for every slug the schema
    # declares, and nothing else (guard CX-01).
    assert set(archive_profiles.PROFILES) == EXPECTED_ARCHIVES
    assert len(archive_profiles.PROFILES) >= 6  # the first-pass roster never shrinks
    for profile in _profiles():
        assert isinstance(profile, ArchiveProfile)


@pytest.mark.parametrize(
    "alias,slug",
    [
        ("ads", "ads_openalex"),
        ("openalex", "ads_openalex"),
        ("papers", "ads_openalex"),
        ("hips", "sia_hips"),
        ("vizier", "vo"),
        ("noirlab", "datalab"),
        ("DataLab", "datalab"),
    ],
)
def test_aliases_resolve(alias, slug):
    assert archive_profiles.canonical_slug(alias) == slug


def test_unknown_archive_raises_with_known_list():
    with pytest.raises(KeyError) as exc:
        archive_profiles.get_profile("hubble")
    assert "datalab" in str(exc.value)


# ── 2. column/parameter unit discipline (no floor, no exemptions) ─────────────
def test_every_numeric_column_resolves_a_unit_state():
    for profile in _profiles():
        for table_key, table in (profile.tables or {}).items():
            for col in table.columns:
                assert col.dtype, f"{profile.archive}:{table_key}:{col.name} missing dtype"
                if col.dtype in NUMERIC_DTYPES:
                    assert col.unit_state is not None, (
                        f"{profile.archive}:{table_key}:{col.name} numeric without unit_state"
                    )
                    if col.unit_state == "unknown":
                        assert col.unit_note and col.unit_note.strip()


def test_every_numeric_surface_parameter_resolves_a_unit_state():
    for profile in _profiles():
        for surface in profile.query_surfaces:
            for param in surface.parameters:
                if param.json_type in NUMERIC_JSON_TYPES:
                    assert param.unit_state is not None, (
                        f"{profile.archive}:{surface.id}:{param.name} numeric without unit_state"
                    )


def test_datalab_profile_covers_every_curated_registry_column():
    """The authored metadata and the AUTHORITATIVE registry column lists must
    cover each other exactly, for every curated table — no drift, no floor."""
    profile = archive_profiles.get_profile("datalab")
    curated = datalab_registry.registered_qualified_tables()
    assert set(profile.tables) == set(curated)
    for qualified in curated:
        catalog, table = qualified.split(".", 1)
        registry_cols = datalab_registry.describe_table(catalog, table)["columns"]
        authored_cols = [c.name for c in profile.tables[qualified].columns]
        assert authored_cols == registry_cols, f"{qualified} column drift"


def test_expansion_catalogs_stay_outside_the_curated_gate():
    profile = archive_profiles.get_profile("datalab")
    for qualified in datalab_registry.expansion_qualified_tables():
        assert qualified not in profile.tables


# ── 3. query surfaces validate against the LIVE registry ─────────────────────
_JSON_TYPE_COMPAT = {
    "number": {"number", "integer"},
    "integer": {"integer"},
    "string": {"string"},
    "boolean": {"boolean"},
    "array": {"array"},
    "object": {"object"},
}


def test_surface_tools_exist_and_parameters_are_subsets(tool_registry):
    for profile in _profiles():
        for surface in profile.query_surfaces:
            tool = tool_registry.get_tool(surface.tool)
            assert tool is not None, f"{profile.archive}:{surface.id} unknown tool {surface.tool}"
            props = tool.parameters.get("properties", {})
            if surface.query_argument:
                assert surface.query_argument in props, (
                    f"{profile.archive}:{surface.id} query_argument not in tool schema"
                )
            for param in surface.parameters:
                assert param.name in props, (
                    f"{profile.archive}:{surface.id}:{param.name} not a registered property"
                )
                registered_type = props[param.name].get("type")
                if registered_type:
                    assert registered_type in _JSON_TYPE_COMPAT[param.json_type], (
                        f"{profile.archive}:{surface.id}:{param.name} type "
                        f"{param.json_type} contradicts registered {registered_type}"
                    )
                registered_enum = props[param.name].get("enum") or (
                    props[param.name].get("items", {}).get("enum")
                    if isinstance(props[param.name].get("items"), dict) else None
                )
                if param.allowed_values and registered_enum:
                    assert set(param.allowed_values) <= set(registered_enum), (
                        f"{profile.archive}:{surface.id}:{param.name} allowed_values "
                        "not a subset of the registered enum"
                    )


# ── 4. golden examples ────────────────────────────────────────────────────────
def test_every_archive_has_at_least_three_goldens():
    for profile in _profiles():
        assert len(profile.golden_examples) >= 3, profile.archive


def test_golden_invocations_validate_against_registered_schemas(tool_registry):
    for profile in _profiles():
        for golden in profile.golden_examples:
            tool = tool_registry.get_tool(golden.invocation.tool)
            assert tool is not None, f"{profile.archive}:{golden.id} unknown tool"
            jsonschema.validate(golden.invocation.arguments, tool.parameters)


def test_sql_goldens_parse_as_one_readonly_statement():
    seen_sql = 0
    for profile in _profiles():
        for golden in profile.golden_examples:
            if not isinstance(golden.request, SqlRequestEvidence):
                continue
            seen_sql += 1
            sql = golden.invocation.arguments[golden.request.argument]
            statements = sqlglot.parse(sql)
            assert len(statements) == 1, f"{profile.archive}:{golden.id} multiple statements"
            tree = statements[0]
            assert isinstance(tree, sqlexp.Select), f"{profile.archive}:{golden.id} not a SELECT"
            forbidden = (
                sqlexp.Insert, sqlexp.Update, sqlexp.Delete,
                sqlexp.Create, sqlexp.Drop, sqlexp.Alter, sqlexp.Merge,
                # SELECT ... INTO writes a table while the root stays a
                # Select node — without this a "read-only" golden could
                # smuggle a write (guard CX-05).
                sqlexp.Into,
            )
            assert not list(tree.find_all(*forbidden)), (
                f"{profile.archive}:{golden.id} contains DDL/DML"
            )
            if golden.invocation.tool.startswith("datalab_"):
                validated = datalab_sql_policy.validate(sql, source="expert")
                assert validated.sql
    assert seen_sql >= 2  # datalab expert SQL + at least one ADQL golden exist


def test_url_goldens_and_endpoints_are_wellformed():
    for profile in _profiles():
        urls = [str(e.url) for e in profile.endpoints]
        urls += [
            str(g.request.url)
            for g in profile.golden_examples
            if isinstance(g.request, UrlRequestEvidence)
        ]
        for url in urls:
            parsed = urllib.parse.urlparse(url)
            assert parsed.scheme in ("http", "https"), url
            assert parsed.netloc, url
            assert "{" not in url and "}" not in url, url
            urllib.parse.parse_qs(parsed.query)  # must not raise


# ── 5. projections: deterministic and size-capped ─────────────────────────────
def _utf8_len(payload) -> int:
    return len(json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"))


def test_archive_projections_fit_the_cap_without_truncation():
    for profile in _profiles():
        payload = projections.archive_projection(profile)
        assert _utf8_len(payload) <= projections.ARCHIVE_PROJECTION_MAX_BYTES
        assert payload["truncated"] is False, f"{profile.archive} authored content too large"
        assert len(payload["golden_examples"]) == min(3, len(profile.golden_examples))
        assert len(payload["pitfalls"]) <= 3
        if profile.tables is not None:
            assert payload["table_count"] == len(profile.tables)


def test_every_curated_table_slice_fits_the_cap():
    for profile in _profiles():
        for table_key in (profile.tables or {}):
            payload = projections.table_projection(profile, table_key)
            assert payload is not None
            assert _utf8_len(payload) <= projections.TABLE_PROJECTION_MAX_BYTES, table_key
            assert payload["truncated"] is False, table_key
            assert payload["schema_source"] == "curated_profile"
            assert len(payload["golden_examples"]) <= 3


def test_oversized_mandatory_fields_fail_loudly_not_oversize():
    """An authored profile whose UN-droppable fields exceed the cap must raise
    (author-time failure), never ship an oversize payload (guard CX-02)."""
    from services.archive_profiles.schema import (
        GoldenExample, Pitfall, ProfileRef, QuerySurface, ToolInvocation,
    )

    big = ArchiveProfile(
        archive="vo",
        description="x" * 7000,  # mandatory field alone busts the 6 KiB cap
        query_surfaces=(QuerySurface(id="s", purpose="p", tool="vo_adql_query",
                                     request_kind="structured_args"),),
        pitfalls=(Pitfall(id="p", summary="s",
                          applies_to=(ProfileRef(kind="archive", ref="vo"),)),),
        golden_examples=tuple(
            GoldenExample(id=f"g{i}", intent="i",
                          invocation=ToolInvocation(tool="t", arguments={"a": 1}))
            for i in range(3)
        ),
    )
    with pytest.raises(ValueError, match="exceeds the"):
        projections.archive_projection(big)


def test_live_table_projection_caps_huge_column_lists():
    """Live-TAP payloads (uncurated, potentially hundreds of columns) obey the
    same slice byte budget with an honest truncation flag (guard CX-03)."""
    payload = {
        "kind": "table", "archive": "datalab", "table": "big.live",
        "schema_source": "live_tap_schema", "hint": "live",
        "columns": [f"column_name_{i:04d}" for i in range(3000)],
    }
    out = projections.live_table_projection(payload)
    assert _utf8_len(out) <= projections.TABLE_PROJECTION_MAX_BYTES
    assert out["truncated"] is True
    assert out["column_count"] == 3000
    assert out["returned_column_count"] < 3000
    assert "datalab_describe_table" in out["hint"]


# ── 6. the browse_schema capability ───────────────────────────────────────────
def _run_browse(**kwargs):
    from capabilities.base import CallContext
    from capabilities.schema import BrowseSchema

    return BrowseSchema().execute(kwargs, CallContext())


def test_browse_schema_registered_next_to_describe_table(tool_registry):
    tool = tool_registry.get_tool("browse_schema")
    assert tool is not None
    assert tool.category == "archive"
    assert "archive" in tool.parameters["properties"]


def test_browse_schema_tool_schema_stays_plain_typed(tool_registry):
    """The hand-tuned json_schema override is load-bearing: the pydantic-derived
    schema rendered Optional[str] as anyOf[string,null] and stalled the live
    anthropic tool path. Keep every property a plain type (guard CX-06)."""
    params = tool_registry.get_tool("browse_schema").parameters
    assert params["properties"]["archive"]["type"] == "string"
    assert params["properties"]["table"]["type"] == "string"
    assert params["required"] == ["archive"]
    assert "anyOf" not in json.dumps(params)


def test_browse_schema_archive_projection():
    result = _run_browse(archive="datalab")
    assert result.success
    assert result.data["kind"] == "archive"
    assert result.data["live_table_keys"] == datalab_registry.expansion_qualified_tables()
    assert result.provenance is not None and result.provenance.service == "archive_profiles"


def test_browse_schema_table_slice_curated():
    result = _run_browse(archive="datalab", table="des_dr1.main")
    assert result.success
    names = [c["name"] for c in result.data["columns"]]
    assert "mag_auto_r" in names and "class_star_r" in names
    assert "mag_auto" not in names and "class_star" not in names
    assert any(p["id"] == "per_band_columns" for p in result.data["pitfalls"])


def test_browse_schema_alias_and_bare_catalog():
    result = _run_browse(archive="noirlab", table="gaia_dr3")
    assert result.success
    assert result.data["table"] == "gaia_dr3.gaia_source"


def test_browse_schema_typed_failures():
    no_tables = _run_browse(archive="splatalogue", table="lines")
    assert not no_tables.success
    assert "search_lines_by_molecule" in no_tables.error

    unknown_archive = _run_browse(archive="hubble")
    assert not unknown_archive.success
    assert "datalab" in unknown_archive.error

    unknown_table = _run_browse(archive="datalab", table="nope.missing")
    assert not unknown_table.success
    # The hint must route to what actually populates the live cache (a Data
    # Lab query) — datalab_describe_table reads the SAME cache and would fail
    # identically on an uncached table (guard CX-08).
    assert "refreshes the cache" in unknown_table.error
    assert "datalab_list_catalogs" in unknown_table.error


# ── 7. prompt index + flag gating ─────────────────────────────────────────────
def test_prompt_index_lines_shape():
    lines = archive_profiles.prompt_index_lines()
    assert len(lines) == len(EXPECTED_ARCHIVES)
    for line in lines:
        assert "browse_schema('" in line
        assert len(line) < 500, f"prompt line too long ({len(line)}): {line[:80]}"
    assert any("mag_auto" in line for line in lines)          # DES per-band trap surfaced
    assert any("Infinity" in line for line in lines)          # NaN guard surfaced


def test_prompt_injection_is_flag_gated(monkeypatch):
    from tests.integration.test_agent_archive_tools import _load_agent_module

    module = _load_agent_module()
    agent = module.QuasarAgent.__new__(module.QuasarAgent)

    monkeypatch.setenv("QUASAR_SCHEMA_GROUNDING", "0")
    off_prompt = agent._build_system_prompt()
    assert "ARCHIVE SCHEMA GROUNDING" not in off_prompt

    monkeypatch.setenv("QUASAR_SCHEMA_GROUNDING", "1")
    on_prompt = agent._build_system_prompt()
    assert "ARCHIVE SCHEMA GROUNDING" in on_prompt
    assert "browse_schema('datalab')" in on_prompt

    monkeypatch.delenv("QUASAR_SCHEMA_GROUNDING", raising=False)
    default_prompt = agent._build_system_prompt()
    assert "ARCHIVE SCHEMA GROUNDING" in default_prompt  # default ON
