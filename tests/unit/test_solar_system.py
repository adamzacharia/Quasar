"""Offline unit tests for F11 solar-system tools (Horizons + SkyBoT)."""

from __future__ import annotations

import math

import pytest

from services.solar_system import SolarSystemService


# ── fakes ───────────────────────────────────────────────────────────────────
class FakeTable:
    """Minimal astropy-Table-like object: dict of column -> list."""

    def __init__(self, columns):
        self._cols = columns
        self.colnames = list(columns.keys())

    def __len__(self):
        return len(next(iter(self._cols.values())))

    def __getitem__(self, key):
        return self._cols[key]


class FakeHorizons:
    def __init__(self, table, capture=None, **kwargs):
        self._table = table
        if capture is not None:
            capture.update(kwargs)

    def ephemerides(self):
        return self._table


class FakeResponse:
    def __init__(self, payload=None, status_code=200, raise_json=False):
        self._payload = payload
        self.status_code = status_code
        self._raise_json = raise_json

    def json(self):
        if self._raise_json:
            raise ValueError("not json")
        return self._payload


# ── Part A: Horizons ────────────────────────────────────────────────────────
def test_horizons_normalizes_and_uses_tmag_when_v_missing():
    table = FakeTable(
        {
            "datetime_str": ["2026-Jul-03 00:00", "2026-Jul-04 00:00"],
            "RA": [187.1, 187.4],
            "DEC": [2.0, 2.1],
            "delta": [3.1, 3.2],
            "r": [2.9, 2.95],
            "Tmag": [8.4, 8.5],  # comet: no V column, has Tmag
            "elong": [120.0, 121.0],
            "alpha": [15.0, 15.2],
        }
    )
    svc = SolarSystemService(horizons_factory=lambda **kw: FakeHorizons(table, **kw))
    out = svc.horizons_ephemeris("Ceres", "2026-07-03", "2026-07-05")

    assert out["success"] is True
    assert out["count"] == 2
    r0 = out["rows"][0]
    assert r0["ra_deg"] == 187.1 and r0["dec_deg"] == 2.0
    assert r0["delta_au"] == 3.1 and r0["r_au"] == 2.9
    assert r0["v_mag"] == 8.4  # from Tmag
    assert r0["datetime"] == "2026-Jul-03 00:00"


def test_horizons_thins_large_tables():
    n = 900
    table = FakeTable(
        {
            "datetime_str": [f"t{i}" for i in range(n)],
            "RA": [float(i) % 360 for i in range(n)],
            "DEC": [0.0] * n,
        }
    )
    svc = SolarSystemService(horizons_factory=lambda **kw: FakeHorizons(table, **kw))
    out = svc.horizons_ephemeris("2026 XY", "2026-01-01", "2027-01-01")
    assert out["success"] is True
    assert out["count"] <= 400
    assert any("thinned" in w for w in out["warnings"])


def test_horizons_ambiguous_target_error_preserved():
    def boom(**kwargs):
        raise ValueError("Ambiguous target name; matches: 1;2;3")

    svc = SolarSystemService(horizons_factory=boom)
    out = svc.horizons_ephemeris("Io", "2026-07-03", "2026-07-04")
    assert out["success"] is False
    assert "Ambiguous target" in out["error"]


def test_horizons_requires_target_and_dates():
    svc = SolarSystemService(horizons_factory=lambda **kw: FakeHorizons(FakeTable({"RA": []})))
    assert svc.horizons_ephemeris("", "a", "b")["success"] is False
    assert svc.horizons_ephemeris("Ceres", "", "b")["success"] is False


# ── Part B: SkyBoT ──────────────────────────────────────────────────────────
def test_skybot_epoch_is_jd_and_params_correct(monkeypatch):
    calls = {}

    def fake_get(url, params=None, timeout=None):
        calls["params"] = dict(params or {})
        calls["timeout"] = timeout
        return FakeResponse(
            [
                {
                    "Num": 211473,
                    "Name": "Herin",
                    "RA (hms)": "12 00 07.02",
                    "DEC (dms)": "-00 00 47.7",
                    "Class": "MB>Middle",
                    "VMag (mag)": 21.8,
                    "Err (arcsec)": 0.027,
                    "d (arcsec)": 115.6,
                }
            ]
        )

    from services import solar_system

    monkeypatch.setattr(solar_system.requests, "get", fake_get)
    out = SolarSystemService(timeout=12).skybot_cone(180.0, 0.0, radius_deg=30.0, epoch="2026-07-03T00:00:00")

    assert out["success"] is True
    assert out["count"] == 1
    # epoch MUST be a JD (float-parseable), never an ISO string with a 'T'
    float(calls["params"]["-ep"])
    assert "T" not in calls["params"]["-ep"]
    assert calls["params"]["-rd"] == 10.0  # clamped from 30
    assert any("clamped" in w for w in out["warnings"])
    assert calls["timeout"] == 12
    row = out["rows"][0]
    assert row["number"] == 211473 and row["name"] == "Herin"
    assert abs(row["ra_deg"] - 180.03) < 0.05  # 12h00m07s -> ~180.03 deg
    assert row["dec_deg"] < 0
    assert row["sep_arcsec"] == 115.6 and row["pos_err_arcsec"] == 0.027


def test_skybot_decimal_degree_variant(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        return FakeResponse([{"Name": "X", "RA": "180.5", "DEC": "-1.25", "Class": "NEA"}])

    from services import solar_system

    monkeypatch.setattr(solar_system.requests, "get", fake_get)
    out = SolarSystemService().skybot_cone(180.5, -1.25, radius_deg=0.1)
    assert out["success"] is True
    row = out["rows"][0]
    assert row["ra_deg"] == 180.5 and row["dec_deg"] == -1.25
    assert row["number"] is None  # unnumbered


def test_skybot_no_solution_and_empty(monkeypatch):
    from services import solar_system

    # flag -1 object form
    monkeypatch.setattr(
        solar_system.requests, "get",
        lambda *a, **k: FakeResponse({"flag": -1, "message": "no solution"}),
    )
    out = SolarSystemService().skybot_cone(0.0, 0.0, radius_deg=1.0)
    assert out["success"] is True and out["count"] == 0
    assert any("no solution" in w for w in out["warnings"])

    # empty array
    monkeypatch.setattr(solar_system.requests, "get", lambda *a, **k: FakeResponse([]))
    out2 = SolarSystemService().skybot_cone(0.0, 0.0, radius_deg=1.0)
    assert out2["success"] is True and out2["count"] == 0


def test_skybot_http_500(monkeypatch):
    from services import solar_system

    monkeypatch.setattr(
        solar_system.requests, "get", lambda *a, **k: FakeResponse(status_code=500)
    )
    out = SolarSystemService().skybot_cone(10.0, 10.0)
    assert out["success"] is False
    assert "HTTP 500" in out["error"]


def test_skybot_default_epoch_is_jd(monkeypatch):
    calls = {}

    def fake_get(url, params=None, timeout=None):
        calls["params"] = dict(params or {})
        return FakeResponse([])

    from services import solar_system

    monkeypatch.setattr(solar_system.requests, "get", fake_get)
    out = SolarSystemService().skybot_cone(10.0, 10.0, epoch=None)
    assert out["success"] is True
    float(calls["params"]["-ep"])  # parses as JD
    assert out["provenance"]["epoch_iso"]  # ISO recorded for readability


def test_horizons_major_body_fallback():
    # codex review P2 (f06-lc): 'Mars' fails the small-body resolver; the
    # service must retry with auto resolution and note it in warnings.
    calls = []

    def factory(**kwargs):
        calls.append(kwargs.get("id_type"))
        if kwargs.get("id_type") == "smallbody":
            raise ValueError("Unknown target Mars in small-body database")
        return FakeHorizons(
            FakeTable({"datetime_str": ["t0"], "RA": [10.0], "DEC": [1.0], "V": [1.3]})
        )

    svc = SolarSystemService(horizons_factory=factory)
    out = svc.horizons_ephemeris("Mars", "2026-07-03", "2026-07-05")

    assert out["success"] is True
    assert calls == ["smallbody", None]
    assert out["provenance"]["id_type"] is None
    assert any("major-body" in w for w in out["warnings"])
    assert out["rows"][0]["v_mag"] == 1.3


def test_horizons_range_guard_truncates_and_rejects():
    # codex review P2 (f06-lc): bound the request BEFORE fetching.
    captured = {}

    def factory(**kwargs):
        captured.update(kwargs)
        return FakeHorizons(
            FakeTable({"datetime_str": ["t0"], "RA": [1.0], "DEC": [1.0]})
        )

    svc = SolarSystemService(horizons_factory=factory)

    # span > 370 d -> stop truncated, warning, factory sees truncated stop
    out = svc.horizons_ephemeris("Ceres", "2020-01-01", "2026-01-01", step="30d")
    assert out["success"] is True
    assert any("truncated" in w for w in out["warnings"])
    assert captured["epochs"]["stop"].startswith("2021-01")

    # absurd row count -> rejected before any fetch
    calls_before = dict(captured)
    out2 = svc.horizons_ephemeris("Ceres", "2026-01-01", "2026-12-01", step="1m")
    assert out2["success"] is False
    assert "coarser step" in out2["error"]
    assert captured == calls_before  # factory NOT called again

    # stop before start -> clear error
    out3 = svc.horizons_ephemeris("Ceres", "2026-07-05", "2026-07-03")
    assert out3["success"] is False
    assert "must be after" in out3["error"]
