"""Offline unit tests for F01 VO registry discovery (all pyvo fakes injected)."""

from __future__ import annotations

import numpy as np

from services.vo_registry import VoRegistryService


# ── fakes ───────────────────────────────────────────────────────────────────
class FakeInterface:
    def __init__(self, url, type_="vs:paramhttp"):
        self.access_url = url
        self.type = type_


class FakeResource:
    def __init__(self, ivoid, short_name=None, title=None, res_type=None,
                 access_url=None, waveband=None, description=None,
                 service=None, interfaces=None):
        self.ivoid = ivoid
        self.short_name = short_name
        self.res_title = title
        self.res_type = res_type
        self.waveband = waveband
        self.res_description = description
        self._service = service
        self._interfaces = interfaces or []
        if access_url is not None:
            self.access_url = access_url

    def get_service(self, service_type, lax=False):
        if self._service is None:
            raise ValueError("no service")
        return self._service

    def list_interfaces(self):
        return self._interfaces


class FakeSvcHandle:
    def __init__(self, baseurl):
        self.baseurl = baseurl


class FakeColumn:
    def __init__(self, name, datatype="char", unit=None, ucd=None, description=None):
        self.name = name
        self.datatype = datatype
        self.unit = unit
        self.ucd = ucd
        self.description = description


class FakeTableMeta:
    def __init__(self, description=None, columns=None):
        self.description = description
        self.columns = columns or []


class FakeDALTable:
    """to_table()-compatible result with colnames + row mapping access."""

    def __init__(self, colnames, rows):
        self.colnames = colnames
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def to_table(self):
        return self


class FakeTAP:
    def __init__(self, tables=None, result=None, error=None):
        self.tables = tables or {}
        self._result = result
        self._error = error
        self.run_sync_calls = []

    def run_sync(self, query, maxrec=None):
        self.run_sync_calls.append({"query": query, "maxrec": maxrec})
        if self._error is not None:
            raise self._error
        return self._result


class FakeSCS:
    def __init__(self, result):
        self._result = result
        self.calls = []

    def search(self, pos=None, radius=None):
        self.calls.append({"pos": pos, "radius": radius})
        return self._result


# ── 1. registry search ──────────────────────────────────────────────────────
def test_registry_search_dedupe_skip_and_schema():
    resources = [
        FakeResource("ivo://a", short_name="A", title="Full record",
                     res_type="vs:catalogservice",
                     service=FakeSvcHandle("https://tap.example/a"),
                     waveband=["radio", "optical"], description="x" * 300),
        FakeResource("ivo://b", short_name="B", title="No URL at all"),
        FakeResource("ivo://c", short_name="C", title="Duplicate of A",
                     service=FakeSvcHandle("https://tap.example/a")),
        FakeResource("ivo://d", short_name="D", title="Interface fallback",
                     interfaces=[FakeInterface("https://scs.example/d")]),
    ]
    captured = {}

    def fake_regsearch(keywords=None, servicetype=None, waveband=None):
        captured.update(keywords=keywords, servicetype=servicetype)
        return resources

    svc = VoRegistryService(regsearch_fn=fake_regsearch)
    out = svc.registry_search("GLEAM survey", service_type="tap")

    assert out["success"] is True
    assert captured["keywords"] == ["GLEAM", "survey"]  # str coerced to list
    assert captured["servicetype"] == "tap"
    assert out["count"] == 2  # A + D (B skipped, C deduped)
    a = out["rows"][0]
    assert a["ivoid"] == "ivo://a" and a["access_url"] == "https://tap.example/a"
    assert a["waveband"] == "radio,optical"
    assert len(a["description"]) <= 200
    assert out["rows"][1]["access_url"] == "https://scs.example/d"
    assert any("no resolvable access URL" in w for w in out["warnings"])


def test_registry_search_clamps_and_scs_alias():
    resources = [
        FakeResource(f"ivo://r{i}", service=FakeSvcHandle(f"https://tap.example/{i}"))
        for i in range(150)
    ]
    captured = {}

    def fake_regsearch(keywords=None, servicetype=None, waveband=None):
        captured["servicetype"] = servicetype
        return resources

    out = VoRegistryService(regsearch_fn=fake_regsearch).registry_search(
        ["HI"], service_type="scs", max_rows=500
    )
    assert captured["servicetype"] == "conesearch"  # alias normalized
    assert out["count"] == 100  # MAX_REGISTRY_ROWS
    assert any("clamped" in w for w in out["warnings"])
    assert VoRegistryService(regsearch_fn=lambda **k: []).registry_search("")["success"] is False


# ── 2. tables ───────────────────────────────────────────────────────────────
def _tap_with_tables():
    tables = {}
    for i in range(60):
        tables[f"cat/table{i}"] = FakeTableMeta(description=f"catalog number {i}",
                                                columns=[FakeColumn("ra"), FakeColumn("dec")])
    tables["special/nvss"] = FakeTableMeta(description="NVSS 1.4 GHz source catalog",
                                           columns=[FakeColumn("S1.4", unit="mJy")])
    return FakeTAP(tables=tables)


def test_list_tables_keyword_filter_and_truncation():
    svc = VoRegistryService(tap_factory=lambda url: _tap_with_tables())
    filtered = svc.list_tables("https://tap.example", keyword="nvss")
    assert filtered["success"] is True and filtered["count"] == 1
    assert filtered["rows"][0]["table_name"] == "special/nvss"
    assert filtered["rows"][0]["n_columns"] == 1

    unfiltered = svc.list_tables("https://tap.example", max_tables=10)
    assert unfiltered["count"] == 10
    assert any("keyword" in w for w in unfiltered["warnings"])


def test_describe_table_case_insensitive_and_missing():
    svc = VoRegistryService(tap_factory=lambda url: _tap_with_tables())
    out = svc.describe_table("https://tap.example", "SPECIAL/NVSS")
    assert out["success"] is True
    assert out["rows"][0]["name"] == "S1.4" and out["rows"][0]["unit"] == "mJy"

    missing = svc.describe_table("https://tap.example", "nope")
    assert missing["success"] is False and "not found" in missing["error"]


# ── 3. guarded ADQL ─────────────────────────────────────────────────────────
def test_run_adql_select_only_guard():
    tap = FakeTAP(result=FakeDALTable(["x"], []))
    svc = VoRegistryService(tap_factory=lambda url: tap)

    out = svc.run_adql("https://tap.example", "DROP TABLE users")
    assert out["success"] is False and "SELECT" in out["error"]
    assert tap.run_sync_calls == []  # never reached the network

    ok = svc.run_adql(
        "https://tap.example",
        "-- a comment\n/* block\ncomment */  select TOP 5 * FROM \"VIII/65/nvss\"",
    )
    assert ok["success"] is True
    assert len(tap.run_sync_calls) == 1


def test_run_adql_maxrec_truncation_and_normalization():
    rows = [
        {"name": b"NGC 253", "flux": np.int64(42), "note": np.ma.masked},
        {"name": "M83", "flux": np.float64(1.5), "note": "ok"},
    ]
    tap = FakeTAP(result=FakeDALTable(["name", "flux", "note"], rows))
    svc = VoRegistryService(tap_factory=lambda url: tap, maxrec_cap=1000)

    out = svc.run_adql("https://tap.example", "SELECT * FROM x", max_rows=2)
    assert tap.run_sync_calls[0]["maxrec"] == 2
    assert out["truncated"] is True
    assert any("row cap" in w for w in out["warnings"])
    r0, r1 = out["rows"]
    assert r0["name"] == "NGC 253"          # bytes -> str
    assert r0["flux"] == 42 and isinstance(r0["flux"], int)  # np.int64 -> int
    assert r0["note"] is None               # masked -> None
    assert r1["flux"] == 1.5

    capped = svc.run_adql("https://tap.example", "SELECT * FROM x", max_rows=99999)
    assert tap.run_sync_calls[-1]["maxrec"] == 1000  # cap enforced
    assert any("clamped" in w for w in capped["warnings"])


def test_run_adql_server_error_text_preserved():
    tap = FakeTAP(error=RuntimeError('Query error: 1064 near "FORM": syntax error'))
    out = VoRegistryService(tap_factory=lambda url: tap).run_adql(
        "https://tap.example", "SELECT * FORM x"
    )
    assert out["success"] is False
    assert 'near "FORM"' in out["error"]


def test_url_validation():
    svc = VoRegistryService(tap_factory=lambda url: FakeTAP())
    assert svc.run_adql("ftp://nope", "SELECT 1")["success"] is False
    assert svc.list_tables("")["success"] is False


# ── 4. cone search ──────────────────────────────────────────────────────────
def test_cone_search_clamp_and_rows():
    scs = FakeSCS(FakeDALTable(["id", "ra"], [{"id": "s1", "ra": 10.0}]))
    svc = VoRegistryService(scs_factory=lambda url: scs)

    out = svc.cone_search("https://scs.example", 10.0, -5.0, radius_deg=30.0)
    assert out["success"] is True and out["count"] == 1
    assert scs.calls[0]["radius"] == 5.0  # clamped
    assert any("clamped" in w for w in out["warnings"])
    assert out["rows"][0]["id"] == "s1"

    bad = svc.cone_search("https://scs.example", 400.0, 0.0)
    assert bad["success"] is False


def test_list_and_describe_prefer_tap_schema():
    # VizieR fix (2026-07-04): svc.tables downloads a ~60k-table XML and
    # aborts; TAP_SCHEMA queries must be preferred when available.
    class SchemaTAP(FakeTAP):
        def run_sync(self, query, maxrec=None):
            self.run_sync_calls.append({"query": query, "maxrec": maxrec})
            q = query.upper()
            if "TAP_SCHEMA.TABLES" in q:
                return FakeDALTable(
                    ["table_name", "description"],
                    [{"table_name": "VIII/65/nvss", "description": "NVSS catalog"}],
                )
            if "TAP_SCHEMA.COLUMNS" in q:
                return FakeDALTable(
                    ["column_name", "datatype", "unit", "ucd", "description"],
                    [{"column_name": "S1.4", "datatype": "float", "unit": "mJy",
                      "ucd": "phot.flux", "description": "1.4 GHz flux"}],
                )
            return FakeDALTable(["x"], [])

    tap = SchemaTAP(tables={"should/not/be/walked": FakeTableMeta()})
    svc = VoRegistryService(tap_factory=lambda url: tap)

    listed = svc.list_tables("https://tap.example", keyword="nvss")
    assert listed["success"] is True and listed["count"] == 1
    assert listed["rows"][0]["table_name"] == "VIII/65/nvss"
    assert "TAP_SCHEMA" in listed["provenance"]["service"]
    assert "LIKE '%nvss%'" in tap.run_sync_calls[0]["query"]

    desc = svc.describe_table("https://tap.example", "VIII/65/nvss")
    assert desc["success"] is True
    assert desc["rows"][0]["name"] == "S1.4" and desc["rows"][0]["unit"] == "mJy"
    assert "TAP_SCHEMA" in desc["provenance"]["service"]


def test_run_adql_rejects_multi_statement():
    # guard review P2: "SELECT 1; DELETE ..." must not reach the network.
    tap = FakeTAP(result=FakeDALTable(["x"], []))
    svc = VoRegistryService(tap_factory=lambda url: tap)
    out = svc.run_adql("https://tap.example", "SELECT 1; DELETE FROM users")
    assert out["success"] is False and "single SELECT" in out["error"]
    assert tap.run_sync_calls == []
    # semicolons inside string literals are fine
    ok = svc.run_adql("https://tap.example", "SELECT * FROM t WHERE note = 'a;b'")
    assert ok["success"] is True
    # a bare trailing semicolon is fine
    ok2 = svc.run_adql("https://tap.example", "SELECT TOP 1 * FROM t;")
    assert ok2["success"] is True


def test_cone_search_passes_maxrec_when_supported():
    # guard review P2: push the cap to the service when it accepts maxrec.
    class MaxrecSCS(FakeSCS):
        def search(self, pos=None, radius=None, maxrec=None):
            self.calls.append({"pos": pos, "radius": radius, "maxrec": maxrec})
            return self._result

    scs = MaxrecSCS(FakeDALTable(["id"], [{"id": "s1"}]))
    out = VoRegistryService(scs_factory=lambda url: scs).cone_search(
        "https://scs.example", 10.0, 0.0, radius_deg=0.5, max_rows=25
    )
    assert out["success"] is True
    assert scs.calls[0]["maxrec"] == 25
    # legacy signature (no maxrec kwarg) still works via fallback
    legacy = FakeSCS(FakeDALTable(["id"], [{"id": "s1"}]))
    out2 = VoRegistryService(scs_factory=lambda url: legacy).cone_search(
        "https://scs.example", 10.0, 0.0
    )
    assert out2["success"] is True


# ── total-deadline guard (live acceptance found VizieR TAP_SCHEMA stalling a
#    whole chat turn: per-read socket timeouts never fire on a trickling
#    response, so list_tables now enforces a wall-clock budget) ──────────────
def test_list_tables_slow_service_fails_fast_not_hangs():
    import time as _time

    class SlowTAP:
        def run_sync(self, query, maxrec=None):
            _time.sleep(5)  # far beyond the 0.3 s budget below
            raise AssertionError("should never complete")

        @property
        def tables(self):  # fallback path must NOT be consulted on deadline
            raise AssertionError("tableset fallback must not run after a deadline stall")

    svc = VoRegistryService(tap_factory=lambda url: SlowTAP(), timeout=0.3)
    t0 = _time.perf_counter()
    out = svc.list_tables("https://slow.example", keyword="hi")
    elapsed = _time.perf_counter() - t0

    assert out["success"] is False
    assert "slow" in out["error"].lower()
    assert elapsed < 3.0  # fails fast; never waits out the 5 s worker


def test_list_tables_retries_case_sensitive_when_service_rejects_lower(monkeypatch):
    """scan-L7 (live 2026-07-18): TAPVizieR's ADQL parser rejects LOWER()
    ('Encountered "("...'), which used to dump every VizieR listing onto the
    full-tableset download and die in pyvo's VOSI parse. The TAP_SCHEMA path
    must retry case-SENSITIVE and disclose it."""
    import services.vo_registry as vr

    class _Result:
        def __iter__(self):
            return iter([])

        def to_table(self):
            import astropy.table
            return astropy.table.Table(
                rows=[("ivoa.demo", "A demo table")],
                names=("table_name", "description"),
            )

    class _Svc:
        def __init__(self):
            self.queries = []

        def run_sync(self, adql, maxrec=None):
            self.queries.append(adql)
            if "LOWER(" in adql:
                raise vr.requests.RequestException(
                    'Incorrect ADQL query:  Encountered "(". Was expecting one of: "."'
                )
            return _Result()

    svc = _Svc()
    service = vr.VoRegistryService()
    monkeypatch.setattr(service, "_tap_service", lambda url: svc)

    out = service.list_tables("https://example.org/tap", keyword="gaia", max_tables=5)

    assert out["success"] is True
    assert len(svc.queries) == 2  # LOWER() attempt, then the case-sensitive retry
    assert "LOWER(" not in svc.queries[1]
    assert any("case-sensitive" in w for w in out["warnings"])
    assert out["rows"][0]["table_name"] == "ivoa.demo"
