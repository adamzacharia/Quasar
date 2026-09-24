"""Offline tests for the archive catalogue tools (integrations/archive_tap.py,
services/archive_catalogs.py, capabilities/catalogs.py).

No network: the TAP client gets a fake HTTP function, the service gets a
scripted fake client. Targets and tables here are generic examples, not
benchmark questions.
"""
from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from integrations.archive_tap import (
    ArchiveTapClient,
    KNOWN_SERVICES,
    TapQueryError,
    TapResult,
    quote_ident,
    quote_table,
    radec_columns,
    sql_literal,
)
from services import archive_catalogs as ac
from services.archive_catalogs import ArchiveCatalogService, build_ads_query, parse_cuts, parse_tns_fields


# ── VOTable fixtures ────────────────────────────────────────────────────────

def _votable(fields, rows, status="OK", extra_info=""):
    head = "".join(f'<FIELD name="{n}" ID="col_{i}" datatype="{t}"{" arraysize=" + chr(34) + "*" + chr(34) if t == "char" else ""}/>'
                   for i, (n, t) in enumerate(fields))
    body = "".join("<TR>" + "".join(f"<TD>{v}</TD>" for v in r) + "</TR>" for r in rows)
    return (
        '<?xml version="1.0"?><VOTABLE version="1.3" xmlns="http://www.ivoa.net/xml/VOTable/v1.3">'
        f'<RESOURCE type="results"><INFO name="QUERY_STATUS" value="{status}"/>'
        f"<TABLE>{head}<DATA><TABLEDATA>{body}</TABLEDATA></DATA></TABLE>{extra_info}</RESOURCE></VOTABLE>"
    ).encode()


def _error_votable(msg):
    return (
        '<?xml version="1.0"?><VOTABLE version="1.3" xmlns="http://www.ivoa.net/xml/VOTable/v1.3">'
        f'<RESOURCE type="results"><INFO name="QUERY_STATUS" value="ERROR">{msg}</INFO></RESOURCE></VOTABLE>'
    ).encode()


class _Resp:
    def __init__(self, content=b"", status=200, text=None, payload=None):
        self.content = content
        self.status_code = status
        self.text = text if text is not None else (content.decode("utf-8", "replace") if content else "")
        self._payload = payload

    def json(self):
        return self._payload


# ── transport ───────────────────────────────────────────────────────────────

def test_tap_client_parses_rows_and_uses_field_names_not_ids():
    seen = {}

    def post(url, data, timeout):
        seen.update(url=url, data=data, timeout=timeout)
        return _Resp(_votable([("obsid", "int"), ("name", "char")], [(101, "alpha"), (102, "beta")]))

    res = ArchiveTapClient(http_post=post).query("heasarc", "SELECT obsid, name FROM t", maxrec=10)
    assert seen["url"].endswith("/xamin/vo/tap/sync")
    assert seen["data"]["QUERY"] == "SELECT obsid, name FROM t" and seen["data"]["MAXREC"] == 10
    assert res.columns == ["obsid", "name"]
    assert res.rows == [{"obsid": 101, "name": "alpha"}, {"obsid": 102, "name": "beta"}]
    assert res.truncated is False


def test_tap_client_raises_server_error_text_instead_of_empty_result():
    post = lambda url, data, timeout: _Resp(_error_votable("Unknown table foo"), status=400)  # noqa: E731
    with pytest.raises(TapQueryError) as err:
        ArchiveTapClient(http_post=post).query("irsa", "SELECT * FROM foo")
    assert "Unknown table foo" in err.value.message
    assert err.value.query == "SELECT * FROM foo"


def test_tap_client_flags_overflow_and_cap_as_truncated():
    over = '<INFO name="QUERY_STATUS" value="OVERFLOW"/>'
    post = lambda url, data, timeout: _Resp(_votable([("a", "int")], [(1,), (2,)], extra_info=over))  # noqa: E731
    assert ArchiveTapClient(http_post=post).query("gaia", "SELECT a FROM t", maxrec=5).truncated
    post2 = lambda url, data, timeout: _Resp(_votable([("a", "int")], [(1,), (2,)]))  # noqa: E731
    assert ArchiveTapClient(http_post=post2).query("gaia", "SELECT a FROM t", maxrec=2).truncated


def test_tap_client_transport_failure_is_typed():
    def post(url, data, timeout):
        raise ConnectionError("reset by peer")

    with pytest.raises(TapQueryError, match="reset by peer"):
        ArchiveTapClient(http_post=post).query("simbad", "SELECT 1")


def test_unknown_service_is_rejected():
    with pytest.raises(TapQueryError, match="unknown TAP service"):
        ArchiveTapClient(http_post=lambda *a: None).query("nowhere", "SELECT 1")


def test_quoting_and_literals():
    assert quote_ident("parallax") == "parallax"
    assert quote_ident("time") == '"time"' and quote_ident("pi") == '"pi"'
    assert quote_ident("Jmag", force=True) == '"Jmag"'
    assert quote_table("I/259/tyc2") == '"I/259/tyc2"'
    assert quote_table("gaiadr3.gaia_source") == "gaiadr3.gaia_source"
    with pytest.raises(ValueError):
        quote_ident('bad"name')
    with pytest.raises(ValueError):
        quote_table("x; DROP TABLE y")
    assert sql_literal("O'Brien") == "'O''Brien'"
    assert sql_literal(3) == "3" and sql_literal(2.5) == "2.5"


def test_radec_columns_prefers_main_ucd_then_names():
    cols = [{"name": "RAJ2000", "ucd": "pos.eq.ra;meta.main"}, {"name": "DEJ2000", "ucd": "pos.eq.dec;meta.main"},
            {"name": "ra_other", "ucd": "pos.eq.ra"}]
    assert radec_columns(cols) == ("RAJ2000", "DEJ2000")
    assert radec_columns([{"name": "ra", "ucd": ""}, {"name": "dec", "ucd": ""}]) == ("ra", "dec")


def test_parse_cuts_accepts_strings_dicts_between_and_in():
    cuts = parse_cuts(["parallax BETWEEN 19 AND 24", {"column": "Jmag", "op": "<", "value": 14},
                       "otype = 'QSO'", "flag in (1, 2)", "x is not null"])
    assert cuts == [("parallax", "BETWEEN", [19, 24]), ("Jmag", "<", 14), ("otype", "=", "QSO"),
                    ("flag", "IN", [1, 2]), ("x", "IS NOT NULL", None)]
    assert parse_cuts("a > 1 and b < 2") == [("a", ">", 1), ("b", "<", 2)]
    with pytest.raises(ValueError):
        parse_cuts(["nonsense"])


def test_mjd_iso_round_trip():
    assert ac.mjd_to_iso(51544.5) == "2000-01-01T12:00:00Z"
    assert abs(ac.iso_to_mjd("2000-01-01") - 51544.0) < 1e-9
    assert ac.iso_to_mjd("not a date") is None


# ── scripted fake client ────────────────────────────────────────────────────

class FakeClient:
    """Answer queries from (regex, rows) rules; record every ADQL."""

    def __init__(self, rules, columns=None):
        self.rules = rules
        self.columns = columns or {}
        self.queries: List[str] = []
        self.services = dict(KNOWN_SERVICES)

    def service(self, key):
        return ArchiveTapClient(http_post=lambda *a: None).service(key)

    def table_columns(self, service, table):
        return self.columns.get((service, table.strip('"')), [])

    def query(self, service, adql, maxrec=2000, timeout=None):
        self.queries.append(adql)
        for pattern, rows in self.rules:
            if re.search(pattern, adql, re.S):
                if isinstance(rows, Exception):
                    raise rows
                rows = rows(adql) if callable(rows) else rows
                return TapResult(rows=[dict(r) for r in rows], columns=list(rows[0].keys()) if rows else [],
                                 truncated=False, elapsed_s=0.1, query=adql, endpoint=f"https://{service}/sync",
                                 service=service)
        raise AssertionError(f"unexpected query: {adql}")


def _cols(*names, ra=None, dec=None):
    out = [{"name": n, "ucd": "", "unit": "", "datatype": ""} for n in names]
    if ra:
        out.append({"name": ra, "ucd": "pos.eq.ra;meta.main", "unit": "deg", "datatype": "double"})
    if dec:
        out.append({"name": dec, "ucd": "pos.eq.dec;meta.main", "unit": "deg", "datatype": "double"})
    return out


# ── HEASARC ─────────────────────────────────────────────────────────────────

def test_heasarc_counts_on_server_and_lists_obsids():
    client = FakeClient([
        (r"COUNT\(\*\) AS n_obs", [{"n_obs": 3, "total_exposure_s": 90000.0, "first_mjd": 51544.0, "last_mjd": 60000.0}]),
        (r"COUNT\(\*\) AS n_public", [{"n_public": 2}]),
        (r"SELECT TOP", [{"DataLinkID": 7, "name": "SRC", "ra": 10.0, "dec": 20.0, "time": 51544.0, "exposure": 30000.0,
                          "public_date": 51900.0, "status": "archived", "pi": "Doe", "detector": "ACIS-S",
                          "grating": "HETG"}]),
    ])
    svc = ArchiveCatalogService(client, now_mjd=lambda: 61000.0)
    out = svc.heasarc_observations("Chandra", 10.0, 20.0, instrument="ACIS-S/HETG", limit=1)
    assert out["success"]
    s = out["summary"]
    assert s["n_observations"] == 3 and s["total_exposure_s"] == 90000.0 and s["n_public"] == 2
    assert s["obsids"] == ["7"] and s["list_truncated"] is True
    assert s["first_observation_utc"].startswith("2000-01-01")
    q = client.queries[0]
    assert "CONTAINS(POINT('ICRS', ra, dec)" in q and "grating = 'HETG'" in q and "detector LIKE 'ACIS-S%'" in q
    assert '"time"' in q  # reserved column name quoted
    assert out["rows"][0]["is_public"] is True
    assert "obsid" in client.queries[1] and "AS obs_id" in client.queries[1]


def test_heasarc_swift_instrument_uses_its_exposure_column():
    client = FakeClient([(r"COUNT\(\*\) AS n_obs", [{"n_obs": 0, "total_exposure_s": None, "first_mjd": None,
                                                    "last_mjd": None}]),
                         (r"COUNT\(\*\) AS n_public", [{"n_public": 0}]),
                         (r"SELECT TOP", [])])
    out = ArchiveCatalogService(client, now_mjd=lambda: 61000.0).heasarc_observations("swift", 1.0, 2.0,
                                                                                      instrument="xrt")
    assert out["success"] and out["summary"]["n_observations"] == 0
    assert "xrt_exposure > 0" in client.queries[0] and "SUM(xrt_exposure)" in client.queries[0]
    assert any("real zero" in w for w in out["warnings"])


def test_heasarc_rejects_unknown_mission_and_instrument():
    svc = ArchiveCatalogService(FakeClient([]))
    assert not svc.heasarc_observations("hubble", 1, 2)["success"]
    bad = svc.heasarc_observations("chandra", 1, 2, instrument="MIRI")
    assert not bad["success"] and "not recognised" in bad["error"]


def test_heasarc_archive_error_is_an_error_not_zero():
    client = FakeClient([(r".*", TapQueryError("NASA HEASARC rejected the query: boom", query="Q"))])
    out = ArchiveCatalogService(client).heasarc_observations("xmm", 1, 2)
    assert not out["success"] and "boom" in out["error"]


# ── Exoplanet Archive ───────────────────────────────────────────────────────

_EXO_COLS = {("exoplanet", "pscomppars"): _cols(*ac.EXO_PLANET_COLUMNS),
             ("exoplanet", "ps"): _cols(*ac.EXO_DEFAULT_SOLUTION_COLUMNS)}


def test_exoplanet_lookup_returns_composite_default_solution_and_provenance():
    client = FakeClient([
        (r"FROM pscomppars WHERE pl_name = 'Test-1 b'", [{"pl_name": "Test-1 b", "pl_orbper": 3.5, "pl_bmassprov": "Mass"}]),
        (r"FROM ps WHERE pl_name IN .* default_flag = 1", [{"pl_name": "Test-1 b", "pl_refname": "<a href=x>Doe et al. 2020</a>",
                                                           "pl_bmassprov": "Msini", "pl_msinie": 10.0}]),
        (r"COUNT\(\*\) AS n_solutions", [{"pl_name": "Test-1 b", "n_solutions": 4}]),
    ], _EXO_COLS)
    out = ArchiveCatalogService(client).exoplanet_lookup(planet_name="Test-1 b")
    p = out["summary"]["planets"][0]
    assert p["default_reference"] == "Doe et al. 2020" and p["n_published_solutions"] == 4
    assert p["mass_provenance"] == "Msini" and p["msini_earth"] == 10.0


def test_exoplanet_lookup_not_found_is_explicit():
    client = FakeClient([(r"pscomppars", [])], _EXO_COLS)
    out = ArchiveCatalogService(client).exoplanet_lookup(planet_name="Nope b")
    assert out["success"] and out["summary"]["found"] is False
    assert any("LIKE" in q for q in client.queries)


def test_exoplanet_census_counts_server_side_with_typed_filters():
    client = FakeClient([(r"COUNT\(\*\) AS n FROM pscomppars", [{"n": 2}]),
                         (r"SELECT TOP", [{"pl_name": "A b"}, {"pl_name": "B c"}])], _EXO_COLS)
    out = ArchiveCatalogService(client).exoplanet_census(filters={"max_distance_pc": 20, "max_radius_earth": 2},
                                                         discovery_facility="Survey X")
    assert out["summary"]["n_planets"] == 2 and out["summary"]["planets"] == ["A b", "B c"]
    q = client.queries[0]
    assert "sy_dist <= 20.0" in q and "pl_rade <= 2.0" in q and "LOWER(disc_facility) LIKE '%survey x%'" in q
    bad =ArchiveCatalogService(client).exoplanet_census(filters={"bogus": 1})
    assert not bad["success"]


def test_exoplanet_census_unknown_cut_column_suggests_close_match():
    out = ArchiveCatalogService(FakeClient([], _EXO_COLS)).exoplanet_census(cuts=["pl_orbpr > 3"])
    assert not out["success"] and "pl_orbper" in out["error"]


# ── SIMBAD ──────────────────────────────────────────────────────────────────

def _simbad_rules(found: Dict[str, Dict[str, Any]]):
    def one(adql):
        m = re.search(r"WHERE i\.id = '([^']*)'", adql)
        rec = found.get(m.group(1)) if m else None
        return [rec] if rec else []
    return [(r"WHERE i\.id = ", one),
            (r"FROM ident WHERE oidref IN", [{"oidref": 1, "id": "Gaia DR3 123"}, {"oidref": 1, "id": "NAME Test Star"}]),
            (r"FROM otypedef", [{"otype": "*", "description": "Star"}])]


def test_simbad_lookup_found_and_not_found_one_query_each():
    rec = {"oid": 1, "main_id": "HD   1", "otype": "*", "ra": 1.0, "dec": 2.0, "rvz_redshift": 0.0001,
           "rvz_bibcode": "2000A&A...1..1X", "plx_value": 10.0, "pmra": 1.0, "pmdec": 2.0}
    client = FakeClient(_simbad_rules({"Test Star": rec}))
    out = ArchiveCatalogService(client).simbad_lookup(["Test Star", "Nonexistent Thing"])
    objs = out["summary"]["objects"]
    assert objs[0]["found"] and objs[0]["main_id"] == "HD 1" and objs[0]["gaia_dr3_id"] == "123"
    assert objs[0]["otype_description"] == "Star"
    assert objs[1]["found"] is False
    assert sum(1 for q in client.queries if "WHERE i.id = " in q) == 2


def test_simbad_positions_reports_nearest_and_misses():
    def cone(adql):
        return [{"main_id": "X", "otype": "G", "ra": 10.0, "dec": 10.0, "rvz_redshift": None, "sep_arcsec": 1.2}] \
            if "10.0, 10.0" in adql else []
    client = FakeClient([(r"FROM basic WHERE", cone), (r"otypedef", [{"otype": "G", "description": "Galaxy"}])])
    out = ArchiveCatalogService(client).simbad_positions(["10.0, 10.0", [20.0, -30.0]], radius_arcsec=5)
    m = out["summary"]["matches"]
    assert m[0]["found"] and m[0]["main_id"] == "X" and m[0]["otype_description"] == "Galaxy"
    assert m[1]["found"] is False and out["summary"]["n_with_match"] == 1


def test_simbad_positions_rejects_bad_position():
    assert not ArchiveCatalogService(FakeClient([])).simbad_positions(["abc"])["success"]


# ── Gaia ────────────────────────────────────────────────────────────────────

def test_gaia_source_by_name_uses_simbad_gaia_id():
    rec = {"oid": 1, "main_id": "V* T", "otype": "RR*", "ra": 1.0, "dec": 2.0}
    rules = _simbad_rules({"T Star": rec}) + [
        (r"FROM gaiadr3.gaia_source WHERE source_id = 123",
         [{"SOURCE_ID": 123, "parallax": 2.0, "phot_g_mean_mag": 12.0, "phot_g_mean_flux_over_error": 1000.0}])]
    out = ArchiveCatalogService(FakeClient(rules)).gaia_source(identifier="T Star")
    src = out["summary"]["source"]
    assert src["source_id"] == 123 and "cross-identification" in out["summary"]["method"]
    assert 0.002 < src["phot_g_mean_mag_error"] < 0.004


def test_gaia_source_without_crossid_propagates_proper_motion():
    rec = {"oid": 2, "main_id": "Fast", "otype": "*", "ra": 100.0, "dec": 0.0, "pmra": 3600.0, "pmdec": 0.0}
    rules = [(r"WHERE i\.id = ", [rec]), (r"FROM ident WHERE oidref IN", []), (r"otypedef", []),
             (r"FROM gaiadr3.gaia_source WHERE 1=CONTAINS", [{"source_id": 9, "sep_arcsec": 0.1}])]
    client = FakeClient(rules)
    out = ArchiveCatalogService(client).gaia_source(identifier="Fast")
    q = [q for q in client.queries if "gaia_source" in q][0]
    # 3.6 arcsec/yr for 16 yr = 57.6 arcsec in RA at the equator -> 100.016 deg
    assert "100.016" in q
    assert any("propagated" in w for w in out["warnings"])


def test_gaia_variability_table_whitelist():
    svc = ArchiveCatalogService(FakeClient([]))
    assert not svc.gaia_variability("made_up", 1)["success"]
    client = FakeClient([(r"vari_rrlyrae", [{"source_id": 5, "pf": 0.5}]),
                         (r"vari_summary", [{"source_id": 5, "mean_mag_g_fov": 10.0}])])
    out = ArchiveCatalogService(client).gaia_variability("rrlyrae", 5)
    assert out["summary"]["record"]["pf"] == 0.5 and out["summary"]["vari_summary"]["mean_mag_g_fov"] == 10.0


# ── catalogue query / find / cross-match ────────────────────────────────────

_VIZ = {("vizier", "II/999/cat"): _cols("Name", "Jmag", ra="RAJ2000", dec="DEJ2000")}


def test_catalog_query_quotes_vizier_names_orders_by_distance_and_counts():
    client = FakeClient([(r"COUNT\(\*\) AS n", [{"n": 42}]),
                         (r"SELECT TOP 2", [{"Name": "a", "Jmag": 12.0, "RAJ2000": 10.0, "DEJ2000": 10.0}])], _VIZ)
    out = ArchiveCatalogService(client).catalog_query("vizier", "II/999/cat", ra=10.0, dec=10.0, radius_arcsec=60,
                                                      cuts=["Jmag < 14"], limit=2)
    assert out["summary"]["n_rows_matching"] == 42 and out["summary"]["list_truncated"]
    assert 'FROM "II/999/cat"' in client.queries[0] and '"Jmag" < 14' in client.queries[0]
    assert "ORDER BY DISTANCE" in client.queries[1]
    assert out["rows"][0]["sep_arcsec"] == 0.0


def test_catalog_query_missing_table_and_bad_column():
    svc = ArchiveCatalogService(FakeClient([], _VIZ))
    miss = svc.catalog_query("vizier", "II/000/none")
    assert not miss["success"] and "catalog_find" in miss["error"]
    bad = svc.catalog_query("vizier", "II/999/cat", cuts=["Jmgg < 1"])
    assert not bad["success"] and "Jmag" in bad["error"]


def test_catalog_query_retries_without_distance_order_when_refused():
    calls = {"n": 0}

    def rows(adql):
        if "ORDER BY DISTANCE" in adql:
            raise TapQueryError("ORDER BY expression not supported", query=adql)
        return [{"Name": "a", "Jmag": 1.0, "RAJ2000": 1.0, "DEJ2000": 1.0}]
    client = FakeClient([(r"COUNT", [{"n": 5}]), (r"SELECT TOP", rows)], _VIZ)
    out = ArchiveCatalogService(client).catalog_query("vizier", "II/999/cat", ra=1.0, dec=1.0, radius_arcsec=10,
                                                      limit=1)
    assert out["success"] and any("not necessarily the nearest" in w for w in out["warnings"])


def test_catalog_crossmatch_counts_distinct_sources():
    cols = {("vizier", "L/1/a"): _cols(ra="RAJ2000", dec="DEJ2000"), ("irsa", "rtab"): _cols(ra="ra", dec="dec")}
    left = [{"RAJ2000": 10.0, "DEJ2000": 10.0}, {"RAJ2000": 10.01, "DEJ2000": 10.0}, {"RAJ2000": 10.05, "DEJ2000": 10.05}]
    right = [{"ra": 10.0 + 0.5 / 3600, "dec": 10.0}, {"ra": 10.0, "dec": 10.0 + 0.4 / 3600},
             {"ra": 10.01, "dec": 10.0 + 1.0 / 3600}]
    client = FakeClient([(r'COUNT.*"L/1/a"', [{"n": 3}]), (r'SELECT TOP \d+ "RAJ2000"', left),
                         (r"COUNT.*rtab", [{"n": 3}]), (r"SELECT TOP \d+ ra, dec", right)], cols)
    out = ArchiveCatalogService(client).catalog_crossmatch({"service": "vizier", "table": "L/1/a"},
                                                           {"service": "irsa", "table": "rtab"},
                                                           ra=10.0, dec=10.0, radius_arcsec=600, match_radius_arcsec=2)
    s = out["summary"]
    assert s["n_left_with_match"] == 2 and s["n_right_with_match"] == 3 and s["fraction_left_matched"] == round(2 / 3, 4)
    assert s["complete"] is True and s["median_sep_arcsec"] <= 1.0


def test_catalog_crossmatch_flags_truncated_side():
    cols = {("gaia", "gaiadr3.gaia_source"): _cols(ra="ra", dec="dec"), ("irsa", "rtab"): _cols(ra="ra", dec="dec")}
    client = FakeClient([(r"COUNT.*gaia_source", [{"n": 500}]), (r"SELECT TOP \d+ ra, dec FROM gaiadr3", [{"ra": 1.0, "dec": 1.0}]),
                         (r"COUNT.*rtab", [{"n": 1}]), (r"SELECT TOP \d+ ra, dec FROM rtab", [{"ra": 1.0, "dec": 1.0}])], cols)
    out = ArchiveCatalogService(client).catalog_crossmatch({"service": "gaia", "table": "gaiadr3.gaia_source"},
                                                           {"service": "irsa", "table": "rtab"}, ra=1.0, dec=1.0,
                                                           radius_arcsec=60, match_radius_arcsec=1)
    assert out["summary"]["complete"] is False
    assert any("only 1 were pulled" in w for w in out["warnings"]) and any("epoch 2016.0" in w for w in out["warnings"])


def test_catalog_find_uses_vizier_ranking_then_lists_tables():
    client = FakeClient([(r"TAP_SCHEMA.tables WHERE table_name LIKE", [
        {"table_name": '"X/10/main"', "description": "main table"}, {"table_name": '"Y/2"', "description": "single"}])])
    finder = lambda text, lim: [{"catalog": "X/10", "description": "Survey X"}, {"catalog": "Y/2", "description": "Y"}]  # noqa: E731
    out = ArchiveCatalogService(client, vizier_find=finder).catalog_find("survey x", services=["vizier"])
    tables = [c["table"] for c in out["summary"]["catalogues"]]
    assert tables == ["X/10/main", "Y/2"]


# ── TNS / ALeRCE / ADS ──────────────────────────────────────────────────────

_TNS_PAGE = (
    '<div class="field field-iauname"><span class="name">IAU name</span><div class="value"><b>sn2099zz </b></div></div>'
    '<div class="field field-radec"><span class="name">RA/DEC (2000)</span><b><div class="value">01:00:00.000 +10:00:00.00</b>'
    '</div><div class="alter-value">15.00000 +10.00000</div></div>'
    '<div class="field field-type"><span class="name">Type</span><div class="value"><b>SN II</b></div></div>'
    '<div class="field field-redshift"><span class="name">Redshift</span><div class="value"><b>0.012</b></div></div>'
    '<div class="field field-hostname"><span class="name">Host Name</span><div class="value"><b>NGC 0001</b></div></div>'
    '<div class="field field-discoverydate"><span class="name">Discovery Date</span><div class="value"><b>2099-01-02 03:04:05</b></div></div>'
)


def test_tns_fields_parse_and_summary():
    fields = parse_tns_fields(_TNS_PAGE)
    assert fields["type"] == "SN II" and fields["hostname"] == "NGC 0001"
    get = lambda url, params=None, timeout=None, headers=None: _Resp(text=_TNS_PAGE, status=200)  # noqa: E731
    out = ArchiveCatalogService(FakeClient([]), http_get=get).tns_object("SN 2099zz")
    s = out["summary"]
    assert s["type"] == "SN II" and s["redshift"] == "0.012" and s["ra_deg"] == 15.0 and s["dec_deg"] == 10.0
    assert out["provenance"]["query"].endswith("/object/2099zz")


def test_tns_rejects_non_designation_and_reports_missing():
    svc = ArchiveCatalogService(FakeClient([]), http_get=lambda *a, **k: _Resp(text="Page not found", status=404))
    assert not svc.tns_object("M31")["success"]
    assert svc.tns_object("2099aa")["summary"]["found"] is False


def test_ztf_object_by_position_picks_nearest_and_gives_iso_dates():
    def get(url, params=None, timeout=None, headers=None):
        if url.endswith("/objects"):
            return _Resp(payload={"items": [{"oid": "ZTF20aaaaaab", "meanra": 1.001, "meandec": 1.0},
                                            {"oid": "ZTF20aaaaaaa", "meanra": 1.0, "meandec": 1.0}]})
        if url.endswith("/probabilities"):
            return _Resp(payload=[{"ranking": 1, "classifier_name": "stamp", "class_name": "SN", "probability": 0.9}])
        return _Resp(payload={"oid": "ZTF20aaaaaaa", "ndet": 12, "firstmjd": 51544.5, "lastmjd": 51545.5})
    out = ArchiveCatalogService(FakeClient([]), http_get=get).ztf_object(ra=1.0, dec=1.0, radius_arcsec=10)
    s = out["summary"]
    assert s["oid"] == "ZTF20aaaaaaa" and s["n_detections"] == 12
    assert s["first_detection_utc"] == "2000-01-01T12:00:00Z" and s["top_classes"]["stamp"]["class"] == "SN"
    assert out["warnings"]  # the second object is mentioned


def test_build_ads_query_fields_and_words():
    q, fq = build_ads_query(author="Doe, Jane", bibstem=["ApJ", "ApJL"], year_from=2001, year_to=2003, refereed=True)
    assert q == 'author:"Doe, Jane" AND bibstem:("ApJ" OR "ApJL") AND year:[2001 TO 2003]' and fq == ["property:refereed"]
    q2, _ = build_ads_query(title_words='"dark matter" halo', year_from=2010)
    assert q2 == 'year:[2010 TO *] AND title:("dark matter" AND halo)'
    with pytest.raises(ValueError):
        build_ads_query()
    with pytest.raises(ValueError):
        build_ads_query(year_from="20x0")


# ── capability layer ────────────────────────────────────────────────────────

def _ctx(service, ads=None):
    from capabilities.base import CallContext

    def table_result(rows, *, columns, source, filter_label, tool_name, warnings=None, provenance=None):
        return {"success": True, "rowcount": len(rows), "warnings": list(warnings or []),
                "provenance": provenance or {}, "source": source}

    return CallContext(services={
        "get_archive_catalog_service": lambda: service,
        "external_catalog_table_result": table_result,
        "live_imagery_coordinates": lambda target_name=None, ra=None, dec=None: (10.0, 20.0, str(target_name)),
        "ads_client": ads,
    })


def test_capability_returns_summary_and_query_provenance():
    from capabilities.catalogs import HeasarcObservations

    client = FakeClient([(r"n_obs", [{"n_obs": 1, "total_exposure_s": 5.0, "first_mjd": 51544.0, "last_mjd": 51544.0}]),
                         (r"n_public", [{"n_public": 1}]), (r"SELECT TOP", [])])
    res = HeasarcObservations().execute({"mission": "nustar", "target_name": "Some Source"},
                                        _ctx(ArchiveCatalogService(client, now_mjd=lambda: 61000.0)))
    out = res.native
    assert out["success"] and out["summary"]["n_observations"] == 1
    assert out["provenance"]["query"].startswith("SELECT COUNT(*)") and "numaster" in out["provenance"]["query"]
    assert "state them" in out["note"]


def test_capability_error_passes_through():
    from capabilities.catalogs import CatalogQuery

    res = CatalogQuery().execute({"service": "vizier", "table": "II/000/none"}, _ctx(ArchiveCatalogService(FakeClient([]))))
    assert res.success is False and "not found" in res.native["error"]


def test_ads_capability_reports_num_found():
    from capabilities.catalogs import AdsSearch

    calls = {}

    def perform_get(endpoint, params):
        calls.update(endpoint=endpoint, params=params)
        return {"response": {"numFound": 57, "docs": [{"bibcode": "2020X", "title": ["T"], "author": ["Doe, J."],
                                                       "year": "2020", "doi": ["10.1/x"], "property": ["REFEREED"]}]}}

    res = AdsSearch().execute({"author": "Doe, Jane", "bibstem": "ApJ", "year_from": 2020, "year_to": 2020,
                               "refereed": True}, _ctx(None, ads=SimpleNamespace(_perform_get=perform_get)))
    out = res.native
    assert out["summary"]["num_found"] == 57 and out["summary"]["papers"][0]["doi"] == "10.1/x"
    assert calls["params"]["fq"] == ["property:refereed"] and "bibstem" in calls["params"]["q"]


def test_tools_are_registered_with_schemas():
    from capabilities.catalogs import CAPABILITIES
    from core.tool_registrations import _ARCHIVE_CATALOG_SCHEMAS
    from services.tool_budgets import INNER_BUDGETS, TOOL_HOSTS, check_hierarchy

    names = {c.name for c in CAPABILITIES}
    assert names == set(_ARCHIVE_CATALOG_SCHEMAS)
    for name in names:
        assert name in INNER_BUDGETS and TOOL_HOSTS.get(name)
    assert check_hierarchy(sorted(names)) == []
