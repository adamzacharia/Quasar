"""Tests for ALMA coverage dependability: frequency_support resolution parsing,
ObsCore data-quality filtering, edge-margin / resolution classification, public
flagging, and Doppler-frame labeling."""

import pandas as pd

from services.alma_science_queries import (
    parse_frequency_support_intervals,
    parse_frequency_support_windows,
)
from services.spectral_line_explorer import ALMACoverageService


# -- frequency_support parsing (with channel resolution) ----------------------

def test_parse_windows_captures_channel_resolution_multi_spw():
    support = "[86.24..88.11GHz, 976.56kHz, XX] U [88.10..89.98GHz, 488.28kHz, YY]"
    windows = parse_frequency_support_windows(support)
    assert len(windows) == 2
    assert windows[0]["low_ghz"] == 86.24
    assert windows[0]["high_ghz"] == 88.11
    assert windows[0]["resolution_khz"] == 976.56
    assert windows[1]["resolution_khz"] == 488.28


def test_parse_windows_handles_mhz_resolution_and_missing_resolution():
    with_mhz = parse_frequency_support_windows("[218.0..220.0GHz, 1.953MHz, ZZ]")
    assert with_mhz[0]["resolution_khz"] == 1953.0  # 1.953 MHz -> 1953 kHz

    no_res = parse_frequency_support_windows("218.0..220.0GHz")
    assert no_res[0]["resolution_khz"] is None


def test_parse_intervals_compat_shim_matches_window_ranges():
    support = "[218.0..220.0GHz, 1.953MHz] U [230.0..232.0GHz, 0.976MHz]"
    intervals = parse_frequency_support_intervals(support)
    assert intervals == [(218.0, 220.0), (230.0, 232.0)]


# -- ObsCore ADQL data-quality filters ---------------------------------------

def _intervals():
    return [{"minimum_ghz": 229.5, "maximum_ghz": 229.6}]


def test_adql_includes_calib_level_filter_by_default():
    adql = ALMACoverageService._build_obscore_adql(
        ra_deg=10.0, dec_deg=20.0, radius_arcsec=60.0, intervals=_intervals()
    )
    assert "calib_level >= 2" in adql
    assert "obs_release_date <=" not in adql  # public filter off by default


def test_adql_public_filter_uses_now():
    adql = ALMACoverageService._build_obscore_adql(
        ra_deg=10.0,
        dec_deg=20.0,
        radius_arcsec=60.0,
        intervals=_intervals(),
        public_only=True,
        now_iso="2026-06-22T12:00:00+00:00",
    )
    assert "obs_release_date <= '2026-06-22T12:00:00'" in adql


def test_adql_can_disable_calib_filter():
    adql = ALMACoverageService._build_obscore_adql(
        ra_deg=10.0, dec_deg=20.0, radius_arcsec=60.0, intervals=_intervals(),
        min_calib_level=0,
    )
    assert "calib_level >=" not in adql  # no calib predicate in the WHERE clause


# -- SPW classification: edge margin + resolution -> usable -------------------

def _line():
    return {"minimum_ghz": 230.49, "maximum_ghz": 230.51, "observed_frequency_ghz": 230.5}


def test_classify_full_with_no_margin():
    window = {"low_ghz": 230.0, "high_ghz": 231.0, "resolution_khz": 488.28}
    result = ALMACoverageService._classify_spw(window, _line())
    assert result["classification"] == "full"
    assert result["usable"] is True  # no resolution requirement -> ok


def test_classify_downgrades_to_edge_within_margin():
    # Line interval is inside the band, but within the edge safety margin.
    window = {"low_ghz": 230.485, "high_ghz": 231.0, "resolution_khz": 488.28}
    result = ALMACoverageService._classify_spw(
        window, _line(), edge_margin_ghz=0.02
    )
    assert result["classification"] == "edge"
    assert result["usable"] is False


def test_classify_edge_via_channel_count_and_resolution():
    # 100 channels * 488.28 kHz ~= 48.8 MHz margin -> line lands in the edge zone.
    window = {"low_ghz": 230.45, "high_ghz": 231.0, "resolution_khz": 488.28}
    result = ALMACoverageService._classify_spw(
        window, _line(), edge_channels=100
    )
    assert result["classification"] == "edge"


def test_classify_resolution_requirement_marks_coarse_window_unusable():
    coarse = {"low_ghz": 230.0, "high_ghz": 231.0, "resolution_khz": 31250.0}
    result = ALMACoverageService._classify_spw(
        coarse, _line(), max_channel_width_khz=1000.0
    )
    assert result["classification"] == "full"
    assert result["resolution_ok"] is False
    assert result["usable"] is False


def test_classify_unknown_resolution_is_not_usable_when_requirement_set():
    unknown = {"low_ghz": 230.0, "high_ghz": 231.0, "resolution_khz": None}
    result = ALMACoverageService._classify_spw(
        unknown, _line(), max_channel_width_khz=1000.0
    )
    assert result["resolution_ok"] is False


def test_classify_center_only_and_partial_and_none():
    line = _line()
    center_only = ALMACoverageService._classify_spw(
        {"low_ghz": 230.5, "high_ghz": 230.505, "resolution_khz": None}, line
    )
    assert center_only["classification"] == "center_only"
    partial = ALMACoverageService._classify_spw(
        {"low_ghz": 230.50, "high_ghz": 230.7, "resolution_khz": None}, line
    )
    assert partial["classification"] in {"center_only", "partial"}
    none = ALMACoverageService._classify_spw(
        {"low_ghz": 100.0, "high_ghz": 101.0, "resolution_khz": None}, line
    )
    assert none["classification"] == "none"


# -- public flagging ----------------------------------------------------------

def test_observation_is_public():
    now = "2026-06-22T00:00:00+00:00"
    assert ALMACoverageService._observation_is_public("2023-01-01T00:00:00", now) is True
    assert ALMACoverageService._observation_is_public("2030-01-01T00:00:00", now) is False
    assert ALMACoverageService._observation_is_public("", now) is None
    assert ALMACoverageService._observation_is_public(None, now) is None


# -- query() integration: response carries dependability metadata -------------

def test_query_reports_filters_doppler_public_and_usability(monkeypatch):
    service = ALMACoverageService()
    frame = pd.DataFrame(
        [
            {
                "proposal_id": "P-FINE",
                "member_ous_uid": "uid://A",
                "frequency_support": "[230.0..231.0GHz, 488.28kHz, XX]",
                "s_ra": 10.0,
                "s_dec": 20.0,
                "t_exptime": 100,
                "s_resolution": 0.2,
                "obs_release_date": "2022-01-01T00:00:00",
            },
            {
                "proposal_id": "P-COARSE",
                "member_ous_uid": "uid://B",
                "frequency_support": "[230.0..231.0GHz, 31250.0kHz, YY]",
                "s_ra": 10.0,
                "s_dec": 20.0,
                "t_exptime": 50,
                "s_resolution": 0.1,
                "obs_release_date": "2099-01-01T00:00:00",
            },
        ]
    )
    monkeypatch.setattr(service, "_query_obscore", lambda **kwargs: frame)
    target = {"ra_deg": 10.0, "dec_deg": 20.0, "redshift": 0.0}
    lines = [
        {
            "line_id": "co21",
            "species": "CO",
            "transition": "2-1",
            "frequency_ghz": 230.5,
            "observed_frequency_ghz": 230.5,
        }
    ]
    result = service.query(
        target=target,
        lines=lines,
        tolerance_mhz=2,
        max_channel_width_khz=1000.0,
        coverage_mode="any",
    )

    # filter + frame metadata present
    assert result["coverage_filters"]["min_calib_level"] == 2
    assert result["coverage_filters"]["max_channel_width_khz"] == 1000.0
    assert result["doppler"]["archive_frame"] == "TOPO"
    assert any("topocentric" in w.lower() for w in result["warnings"])  # frame margin not applied

    by_id = {p["proposal_id"]: p for p in result["projects"]}
    # Fine-resolution project: full + usable + public
    fine = by_id["P-FINE"]
    assert fine["all_lines_full"] is True
    assert fine["all_lines_usable"] is True
    assert fine["observations"][0]["is_public"] is True
    # Coarse project: full coverage but NOT usable (channel too wide), proprietary
    coarse = by_id["P-COARSE"]
    assert coarse["all_lines_full"] is True
    assert coarse["all_lines_usable"] is False
    assert coarse["observations"][0]["is_public"] is False
    # Usable project sorts ahead of the merely-full one.
    assert result["projects"][0]["proposal_id"] == "P-FINE"
