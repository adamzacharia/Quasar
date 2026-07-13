# tests/unit/test_datalab_schema_grounding.py
"""Live TAP-schema grounding: the nightly cache, the SQL governor's column
check, the survey-breadth live describe_table fallback, and the four new
curated catalogs (unwise_dr1, twomass, allwise, splus_dr4).

All tests are offline: the "live" schema is a hand-built payload seeded into a
tmp_path DiskCache via the same _cache_schema the refresher uses.
"""

from __future__ import annotations

import pandas as pd
import pytest

from services import datalab_query_builders as builders
from services import datalab_registry as reg
from services import datalab_sql_policy as policy


# ── fakes ─────────────────────────────────────────────────────────────────────
class _Result:
    def __init__(self, dataframe):
        self.dataframe = dataframe


class _FakeSchemaClient:
    """Records every SQL and serves canned tap_schema.tables / .columns frames."""

    def __init__(self, tables_df, columns_df):
        self.calls = []
        self._tables_df = tables_df
        self._columns_df = columns_df

    def query(self, *, sql=None, adql=None, fmt="pandas", **kwargs):
        self.calls.append(sql or adql)
        text = (sql or adql or "").lower()
        if "tap_schema.tables" in text:
            return _Result(self._tables_df.copy())
        if "tap_schema.columns" in text:
            return _Result(self._columns_df.copy())
        raise AssertionError(f"unexpected SQL: {sql!r}")


def _schema_frames(qualified: str, columns: dict[str, bool]):
    tables_df = pd.DataFrame(
        [{"table_name": qualified, "table_type": "table", "description": "test table"}]
    )
    columns_df = pd.DataFrame(
        [
            {
                "table_name": qualified,
                "column_name": name,
                "datatype": "adql:DOUBLE",
                "unit": None,
                "indexed": 1 if indexed else 0,
            }
            for name, indexed in columns.items()
        ]
    )
    return tables_df, columns_df


def _seed_live_payload(cache_dir, table_columns: dict[str, list[str]]):
    payload = {
        "tables": {name: {"description": "seeded"} for name in table_columns},
        "columns": {
            name: {col: {"datatype": "adql:DOUBLE", "unit": None, "indexed": False} for col in cols}
            for name, cols in table_columns.items()
        },
        "refreshed_at": 4e12,  # far future => always fresh
    }
    reg._cache_schema(payload, cache_dir=cache_dir)
    return payload


@pytest.fixture()
def isolated_cache(tmp_path, monkeypatch):
    """Point the default cache dir at a per-test directory."""
    monkeypatch.setenv("DATALAB_TAP_SCHEMA_CACHE_DIR", str(tmp_path / "tap-schema"))
    return tmp_path / "tap-schema"


# ── refresh_tap_schema / cached_tap_schema ────────────────────────────────────
def test_refresh_tap_schema_keys_on_qualified_table_name(isolated_cache):
    tables_df, columns_df = _schema_frames(
        "gaia_dr3.gaia_source", {"ra": True, "dec": True, "parallax": False}
    )
    client = _FakeSchemaClient(tables_df, columns_df)
    # Scope to one table so the fetch is a single chunk (2 queries); the default
    # scope is chunked across several queries (see the scope test below).
    payload = reg.refresh_tap_schema(client, tables=["gaia_dr3.gaia_source"])

    # Live-verified contract: tap_schema.columns has NO schema_name column —
    # both fetches must filter on the fully-qualified table_name.
    assert len(client.calls) == 2
    for sql in client.calls:
        assert "schema_name" not in sql.lower()
        assert "table_name in (" in sql.lower()
        assert "'gaia_dr3.gaia_source'" in sql.lower()

    cols = payload["columns"]["gaia_dr3.gaia_source"]
    assert set(cols) == {"ra", "dec", "parallax"}
    assert cols["ra"]["indexed"] is True and cols["parallax"]["indexed"] is False
    assert payload["tables"]["gaia_dr3.gaia_source"]["description"] == "test table"

    cached = reg.cached_tap_schema()
    assert cached is not None
    assert set(cached["columns"]) == {"gaia_dr3.gaia_source"}


def test_refresh_scope_defaults_to_registered_tables(isolated_cache):
    tables_df, columns_df = _schema_frames("gaia_dr3.gaia_source", {"ra": True})
    client = _FakeSchemaClient(tables_df, columns_df)
    reg.refresh_tap_schema(client)
    # Every curated table (including the four new catalogs) is fetched — the scope
    # is now chunked across several queries, so search all of them.
    all_sql = " ".join(client.calls)
    for name in ("unwise_dr1.object", "twomass.psc", "allwise.source", "splus_dr4.dual"):
        assert f"'{name}'" in all_sql


def test_default_refresh_scope_includes_expansion_tables(isolated_cache):
    tables_df, columns_df = _schema_frames("gaia_dr3.gaia_source", {"ra": True})
    client = _FakeSchemaClient(tables_df, columns_df)
    reg.refresh_tap_schema(client)
    # The verified expansion catalogs join the default fetch scope so they are
    # cached (and thus queryable) after the nightly refresh. Scope is chunked
    # across several queries, so search all of them.
    all_sql = " ".join(client.calls)
    for name in ("delve_dr2.objects", "catwise2020.main", "ls_dr10.tractor"):
        assert f"'{name}'" in all_sql
    # Curated tables stay in scope alongside them.
    assert "'gaia_dr3.gaia_source'" in all_sql


def test_expansion_table_describes_via_live_fallback_with_overlay(isolated_cache):
    # delve_dr2 is NOT a curated catalog; once its schema is cached it must be
    # describable via the live fallback, with the curated footprint/citation.
    _seed_live_payload(
        isolated_cache,
        {"delve_dr2.objects": ["ra", "dec", "mag_auto_g", "mag_auto_r", "ring256"]},
    )
    entry = reg.describe_table("delve_dr2", "objects")
    assert entry["ra_column"] == "ra" and entry["dec_column"] == "dec"
    assert entry["region_strategy"] == "q3c"
    assert entry["source"] == "live_tap_schema"
    assert "DECam Local Volume" in (entry["footprint"] or "")
    assert "DELVE" in entry["citation"].get("text", "")
    assert entry["citation"].get("doi") == "10.3847/1538-4365/ac78eb"


def test_expansion_catalogs_listed_and_queryable(isolated_cache):
    # Advertised in list_catalogs so the agent can discover them.
    listed = {row["catalog"] for row in reg.list_catalogs()}
    assert {"delve_dr2", "catwise2020", "ls_dr10"} <= listed
    # And a governed q3c cone query on a cached expansion table is accepted.
    _seed_live_payload(
        isolated_cache,
        {"ls_dr10.tractor": ["ra", "dec", "dered_flux_g", "dered_flux_r"]},
    )
    vq = policy.validate(
        "SELECT ra, dec FROM ls_dr10.tractor "
        "WHERE q3c_radial_query(ra, dec, 10.0, 41.0, 0.01)",
        source="builder",
    )
    assert "ls_dr10.tractor" in vq.sql.lower()


def test_refresh_extra_tables_and_bad_names_rejected(isolated_cache):
    tables_df, columns_df = _schema_frames("catwise2020.main", {"ra": True, "dec": True})
    client = _FakeSchemaClient(tables_df, columns_df)
    reg.refresh_tap_schema(client, tables=["catwise2020.main"])
    assert "'catwise2020.main'" in client.calls[0]
    with pytest.raises(ValueError):
        reg.refresh_tap_schema(client, tables=["bad'; DROP TABLE x--"])


# ── ensure_tap_schema_fresh ───────────────────────────────────────────────────
def test_ensure_fresh_disabled_by_env(isolated_cache):
    # conftest sets DATALAB_TAP_SCHEMA_AUTOREFRESH=0 for the whole suite.
    assert reg.ensure_tap_schema_fresh(object()) == "disabled"


def test_ensure_fresh_lifecycle(isolated_cache, monkeypatch):
    monkeypatch.setenv("DATALAB_TAP_SCHEMA_AUTOREFRESH", "1")
    reg._reset_refresh_state_for_tests()
    tables_df, columns_df = _schema_frames("gaia_dr3.gaia_source", {"ra": True, "dec": True})
    client = _FakeSchemaClient(tables_df, columns_df)

    # No cache yet -> a foreground run refreshes and populates the cache.
    assert reg.ensure_tap_schema_fresh(client, background=False) == "refreshed"
    assert reg.cached_tap_schema() is not None
    # Fresh cache -> no new attempt, client untouched.
    calls_before = len(client.calls)
    assert reg.ensure_tap_schema_fresh(client, background=False) == "fresh"
    assert len(client.calls) == calls_before


def test_ensure_fresh_throttles_retries_and_survives_failure(isolated_cache, monkeypatch):
    monkeypatch.setenv("DATALAB_TAP_SCHEMA_AUTOREFRESH", "1")
    reg._reset_refresh_state_for_tests()

    class _Boom:
        def query(self, **kwargs):
            raise RuntimeError("service down")

    # A failing refresh must not raise, and the next call inside the retry
    # window is skipped instead of hammering the dead service.
    assert reg.ensure_tap_schema_fresh(_Boom(), background=False) == "refreshed"
    assert reg.cached_tap_schema() is None
    assert reg.ensure_tap_schema_fresh(_Boom(), background=False) == "skipped"
    reg._reset_refresh_state_for_tests()


def test_ensure_fresh_accepts_factory(isolated_cache, monkeypatch):
    monkeypatch.setenv("DATALAB_TAP_SCHEMA_AUTOREFRESH", "1")
    reg._reset_refresh_state_for_tests()
    tables_df, columns_df = _schema_frames("gaia_dr3.gaia_source", {"ra": True})
    made = []

    def factory():
        client = _FakeSchemaClient(tables_df, columns_df)
        made.append(client)
        return client

    assert reg.ensure_tap_schema_fresh(factory, background=False) == "refreshed"
    # The factory client ran the (chunked) refresh — an even number of queries,
    # in tables/columns pairs, at least one chunk.
    assert made and len(made[0].calls) >= 2 and len(made[0].calls) % 2 == 0


# ── known_columns ─────────────────────────────────────────────────────────────
def test_known_columns_curated_only_is_not_authoritative(isolated_cache):
    info = reg.known_columns("gaia_dr3", "gaia_source")
    assert info is not None and info["authoritative"] is False
    assert {"source_id", "ra", "dec", "bp_rp"} <= info["columns"]


def test_known_columns_union_with_live_is_authoritative(isolated_cache):
    _seed_live_payload(None, {"gaia_dr3.gaia_source": ["ra", "dec", "grvs_mag"]})
    info = reg.known_columns("gaia_dr3", "gaia_source")
    assert info["authoritative"] is True
    # union: curated bp_rp AND live-only grvs_mag both known
    assert "bp_rp" in info["columns"] and "grvs_mag" in info["columns"]


def test_known_columns_unknown_table_is_none(isolated_cache):
    assert reg.known_columns("nope_dr9", "objects") is None


# ── live_table_entry / describe_table fallback ────────────────────────────────
def test_live_table_entry_autodetects_coords_and_healpix(isolated_cache):
    _seed_live_payload(
        None,
        {"vista_x.cat": ["sourceid", "ra2000", "dec2000", "jmag", "ring256", "nest4096", "nest4095"]},
    )
    entry = reg.live_table_entry("vista_x", "cat")
    assert entry["ra_column"] == "ra2000" and entry["dec_column"] == "dec2000"
    # nest4095 is not a power-of-two nside -> rejected
    assert [h["name"] for h in entry["healpix_columns"]] == ["ring256", "nest4096"]
    assert entry["region_strategy"] == "q3c" and entry["aggregate_safe"] is False
    assert entry["source"] == "live_tap_schema"


def test_live_table_entry_requires_coordinates(isolated_cache):
    _seed_live_payload(None, {"weird.table": ["flux", "mjd"]})
    assert reg.live_table_entry("weird", "table") is None


def test_describe_table_prefers_curated_then_live(isolated_cache):
    _seed_live_payload(None, {"catwise2020.main": ["ra", "dec", "w1mpro_pm"]})
    live = reg.describe_table("catwise2020", "main")
    assert live["source"] == "live_tap_schema"
    curated = reg.describe_table("gaia_dr3", "gaia_source")
    assert "source" not in curated  # curated entries keep the legacy shape
    with pytest.raises(ValueError):
        reg.describe_table("never_heard_of_it", "at_all")


def test_describe_table_checks_live_before_single_curated_remap(isolated_cache):
    _seed_live_payload(None, {"gaia_dr3.extra_source": ["ra", "dec", "flux"]})
    live = reg.describe_table("gaia_dr3", "extra_source")
    assert live["qualified_name"] == "gaia_dr3.extra_source"
    assert live["source"] == "live_tap_schema"


def test_policy_accepts_live_fallback_table(isolated_cache):
    _seed_live_payload(None, {"catwise2020.main": ["ra", "dec", "w1mpro_pm"]})
    sql = (
        "SELECT ra, dec, w1mpro_pm FROM catwise2020.main "
        "WHERE q3c_radial_query(ra, dec, 150.0, 2.2, 0.05)"
    )
    validated = policy.validate(sql, source="expert")
    assert "LIMIT 500" in validated.sql


# ── the governor's column check ───────────────────────────────────────────────
_CONE = "WHERE q3c_radial_query(ra, dec, 150.0, 2.2, 0.1)"


def test_unknown_column_warns_when_only_curated(isolated_cache):
    validated = policy.validate(
        f"SELECT ra, dec, fake_col FROM gaia_dr3.gaia_source {_CONE}", source="expert"
    )
    assert any("fake_col" in w and "curated registry" in w for w in validated.warnings)


def test_unknown_column_hard_error_when_authoritative(isolated_cache):
    _seed_live_payload(None, {"gaia_dr3.gaia_source": ["source_id", "ra", "dec", "parallax"]})
    with pytest.raises(policy.DatalabPolicyError) as excinfo:
        policy.validate(
            f"SELECT ra, dec, parallax_over_err FROM gaia_dr3.gaia_source {_CONE}",
            source="expert",
        )
    message = str(excinfo.value)
    assert "parallax_over_err" in message
    assert "parallax_over_error" in message  # did-you-mean from the curated union
    assert "datalab_describe_table" in excinfo.value.fix_hint


def test_real_columns_pass_authoritative(isolated_cache):
    _seed_live_payload(
        None, {"gaia_dr3.gaia_source": ["source_id", "ra", "dec", "parallax", "grvs_mag"]}
    )
    validated = policy.validate(
        f"SELECT source_id, grvs_mag, parallax/1000.0 AS plx_arcsec "
        f"FROM gaia_dr3.gaia_source {_CONE} ORDER BY plx_arcsec DESC LIMIT 20",
        source="expert",
    )
    assert validated.warnings == []


def test_qualified_alias_references_checked(isolated_cache):
    _seed_live_payload(None, {"gaia_dr3.gaia_source": ["source_id", "ra", "dec", "pmra"]})
    with pytest.raises(policy.DatalabPolicyError):
        policy.validate(
            f"SELECT g.source_id, g.pm_ra FROM gaia_dr3.gaia_source AS g {_CONE}",
            source="expert",
        )


def test_with_query_skips_bare_identifier_check(isolated_cache):
    _seed_live_payload(None, {"gaia_dr3.gaia_source": ["source_id", "ra", "dec"]})
    # CTE output columns are not resolvable statically -> bare idents unchecked.
    sql = (
        "WITH cone AS (SELECT source_id, ra, dec FROM gaia_dr3.gaia_source "
        "WHERE q3c_radial_query(ra, dec, 150.0, 2.2, 0.05)) "
        "SELECT source_id FROM cone LIMIT 10"
    )
    validated = policy.validate(sql, source="expert")
    assert "LIMIT" in validated.sql


def test_functions_and_aliases_not_flagged(isolated_cache):
    _seed_live_payload(None, {"gaia_dr3.gaia_source": ["source_id", "ra", "dec", "parallax"]})
    sql = (
        "SELECT COUNT(*) AS n, AVG(parallax) AS mean_plx "
        f"FROM gaia_dr3.gaia_source {_CONE} GROUP BY source_id ORDER BY n DESC LIMIT 100"
    )
    validated = policy.validate(sql, source="expert")
    assert validated.warnings == []


def test_column_check_env_off(isolated_cache, monkeypatch):
    monkeypatch.setenv("DATALAB_COLUMN_CHECK", "0")
    _seed_live_payload(None, {"gaia_dr3.gaia_source": ["ra", "dec"]})
    validated = policy.validate(
        f"SELECT nonsense_col FROM gaia_dr3.gaia_source {_CONE}", source="expert"
    )
    assert not any("nonsense_col" in w for w in validated.warnings)


def test_string_literals_never_flagged(isolated_cache):
    _seed_live_payload(None, {"sdss_dr17.specobj": ["specobjid", "ra", "dec", "class", "z"]})
    validated = policy.validate(
        "SELECT specobjid, z FROM sdss_dr17.specobj "
        "WHERE ra BETWEEN 150.0 AND 150.5 AND dec BETWEEN 2.0 AND 2.5 "
        "AND class = 'GALAXY_FAKE_TOKEN'",
        source="expert",
    )
    assert validated.warnings == [] or all("fake" not in w.lower() for w in validated.warnings)


# ── the four new curated catalogs ─────────────────────────────────────────────
@pytest.mark.parametrize(
    "catalog,table,band,expected_mag,id_col",
    [
        ("unwise_dr1", "object", "w1", "mag_w1_vg", "unwise_objid"),
        ("twomass", "psc", "k", "k_m", "designation"),
        ("allwise", "source", "w2", "w2mpro", "cntr"),
        ("splus_dr4", "dual", "r", "r_auto", "id"),
    ],
)
def test_new_catalogs_registered(catalog, table, band, expected_mag, id_col):
    info = reg.describe_table(catalog, table)
    assert expected_mag in info["columns"]
    assert reg.mag_column(catalog, table, band) == expected_mag
    assert id_col in reg.indexed_bound_columns(catalog, table)
    assert info["aggregate_safe"] is True
    assert info["footprint"]
    assert reg.citation(catalog).get("text")


def test_new_catalogs_in_catalog_listing():
    names = {row["catalog"] for row in reg.list_catalogs()}
    assert {"unwise_dr1", "twomass", "allwise", "splus_dr4"} <= names


@pytest.mark.parametrize(
    "catalog,table",
    [("unwise_dr1", "object"), ("twomass", "psc"), ("allwise", "source"), ("splus_dr4", "dual")],
)
def test_new_catalogs_builder_roundtrip(isolated_cache, catalog, table):
    sql, meta = builders.build_cone_count(catalog=catalog, table=table, ra=80.9, dec=-69.75, radius_deg=0.1)
    validated = policy.validate(sql, source="builder", meta=meta)
    assert f"{catalog}.{table}" in validated.sql.lower()


def test_new_catalog_point_source_cuts():
    assert reg.point_source_cut("allwise", "source") == {"column": "ext_flg", "op": "=", "value": 0}
    assert reg.point_source_cut("splus_dr4", "dual") == {"column": "class_star", "op": ">", "value": 0.9}
    assert reg.point_source_cut("twomass", "psc") is None
    assert reg.point_source_cut("unwise_dr1", "object") is None


def test_twomass_nest_is_256_not_4096():
    # Live-verified: 2MASS PSC carries nest256 (nside 256), not nest4096.
    healpix = {h["name"]: h["nside"] for h in reg.describe_table("twomass", "psc")["healpix_columns"]}
    assert healpix == {"ring256": 256, "nest256": 256}


# ── guard-review regressions (task-097819d-1643) ─────────────────────────────
def test_array_poly_query_passes_authoritative_check(isolated_cache):
    """CX-01: the rectangular builder emits q3c_poly_query(..., ARRAY[...]);
    ARRAY must never be flagged as an unknown column under a live cache."""
    _seed_live_payload(None, {"gaia_dr3.gaia_source": ["source_id", "ra", "dec", "parallax"]})
    sql, meta = builders.build_rectangular_region_select(
        "gaia_dr3", "gaia_source", ra_min=150.0, ra_max=150.5, dec_min=2.0, dec_max=2.5, limit=50,
    )
    validated = policy.validate(sql, source="builder", meta=meta)
    assert "q3c_poly_query" in validated.sql.lower()


def test_qualified_ref_must_belong_to_the_alias_table(isolated_cache):
    """CX-02: a column that exists only on ANOTHER joined table is still wrong
    when referenced through this alias."""
    _seed_live_payload(None, {
        "gaia_dr3.gaia_source": ["source_id", "ra", "dec", "parallax"],
        "sdss_dr17.specobj": ["specobjid", "ra", "dec", "z"],
    })
    sql = (
        "SELECT s.parallax FROM sdss_dr17.specobj AS s "
        "JOIN gaia_dr3.gaia_source AS g ON s.specobjid = g.source_id "
        "WHERE q3c_radial_query(s.ra, s.dec, 150.0, 2.2, 0.05)"
    )
    with pytest.raises(policy.DatalabPolicyError) as excinfo:
        policy.validate(sql, source="expert")
    assert "parallax" in str(excinfo.value)
    # ...while the same column through the RIGHT alias is fine.
    ok_sql = sql.replace("SELECT s.parallax", "SELECT g.parallax")
    policy.validate(ok_sql.replace("JOIN gaia_dr3.gaia_source AS g", "JOIN gaia_dr3.gaia_source AS g"), source="expert")


def test_scoped_refresh_merges_with_existing_cache(isolated_cache):
    """CX-03: a scoped refresh (tables=[...]) must not wipe the registered
    tables' schema, and a later registered refresh must not wipe the seed."""
    tables_df, columns_df = _schema_frames("gaia_dr3.gaia_source", {"ra": True, "dec": True})
    reg.refresh_tap_schema(_FakeSchemaClient(tables_df, columns_df))
    extra_tables, extra_columns = _schema_frames("catwise2020.main", {"ra": True, "dec": True, "w1mpro_pm": False})
    reg.refresh_tap_schema(_FakeSchemaClient(extra_tables, extra_columns), tables=["catwise2020.main"])
    payload = reg.cached_tap_schema()
    assert {"gaia_dr3.gaia_source", "catwise2020.main"} <= set(payload["columns"])
    # Registered refresh again -> the seeded live table survives.
    reg.refresh_tap_schema(_FakeSchemaClient(tables_df, columns_df))
    payload = reg.cached_tap_schema()
    assert "catwise2020.main" in payload["columns"]


def test_live_sibling_wins_over_single_table_tolerance(isolated_cache):
    """CX-03: a catalog with ONE curated table must still expose a seeded live
    sibling instead of silently resolving to the curated table."""
    _seed_live_payload(None, {"gaia_dr3.gaia_source_lite": ["ra", "dec", "phot_g_mean_mag"]})
    entry = reg.describe_table("gaia_dr3", "gaia_source_lite")
    assert entry.get("source") == "live_tap_schema"
    assert entry["qualified_name"] == "gaia_dr3.gaia_source_lite"
    # The tolerance still works for names with no live sibling.
    assert reg.describe_table("gaia_dr3", "object")["qualified_name"] == "gaia_dr3.gaia_source"
