# tests/unit/test_facility_routing.py
"""
Unit tests for VLA/VLBA/GBT facility routing in SearchService.

Verifies that:
  - is_nrao_facility correctly classifies facilities
  - search_by_target / cone_search / search_by_frequency route NRAO
    facilities to the NRAO TAP client (not ALminer)
  - ALMA (or no facility) routes to ALminer
  - facility -> instrument mapping is correct

All tests are offline: clients are replaced with fakes.
"""

import pandas as pd
import pytest

from services.search import SearchService, NRAO_FACILITIES
from integrations.tap import NRAOTapClient

pytestmark = pytest.mark.unit


class FakeNRAOClient:
    """Records calls; returns a distinctive DataFrame."""

    def __init__(self):
        self.calls = []

    def search_vla_vlba(self, target_name, max_results=100, instruments=None):
        self.calls.append(("target", target_name, tuple(instruments or [])))
        return pd.DataFrame({"target_name": [target_name], "instrument_name": [instruments[0] if instruments else "VLA"]})

    def search_by_position(self, ra, dec, radius=0.5, instruments=None, max_results=100):
        self.calls.append(("position", ra, dec, tuple(instruments or [])))
        return pd.DataFrame({"s_ra": [ra], "s_dec": [dec]})

    def search_by_frequency_range(self, min_freq_ghz, max_freq_ghz, instruments=None, max_results=100):
        self.calls.append(("frequency", min_freq_ghz, max_freq_ghz, tuple(instruments or [])))
        return pd.DataFrame({"freq_min_ghz": [min_freq_ghz], "freq_max_ghz": [max_freq_ghz]})


class FakeALminerClient:
    """Records calls; returns a distinctive DataFrame."""

    def __init__(self):
        self.calls = []

    def search_by_target(self, target_name):
        self.calls.append(("target", target_name))
        return pd.DataFrame({"target_name": [target_name], "band_list": ["6"]})

    def search_by_position(self, ra, dec, radius):
        self.calls.append(("position", ra, dec))
        return pd.DataFrame({"RAJ2000": [ra]})

    def search_by_frequency(self, min_freq_ghz, max_freq_ghz):
        self.calls.append(("frequency", min_freq_ghz, max_freq_ghz))
        return pd.DataFrame({"frequency": [min_freq_ghz]})


@pytest.fixture
def service():
    svc = SearchService.__new__(SearchService)  # skip __init__ network-adjacent setup
    svc.alminer_client = FakeALminerClient()
    svc._nrao_client = FakeNRAOClient()
    return svc


# ── Facility classification ─────────────────────────────────────────


def test_nrao_facilities_set():
    assert {"VLA", "VLBA", "GBT", "EVLA", "JVLA"} == NRAO_FACILITIES


@pytest.mark.parametrize("facility,expected", [
    ("VLA", True),
    ("vla", True),
    (" Vlba ", True),
    ("GBT", True),
    ("EVLA", True),
    ("ALMA", False),
    ("", False),
    (None, False),
    ("JWST", False),
])
def test_is_nrao_facility(facility, expected):
    assert SearchService.is_nrao_facility(facility) is expected


# ── Instrument mapping ───────────────────────────────────────────────


def test_instruments_for_facility_vla():
    assert NRAOTapClient.instruments_for_facility("VLA") == ["VLA", "EVLA", "JVLA"]


def test_instruments_for_facility_vlba():
    assert NRAOTapClient.instruments_for_facility("VLBA") == ["VLBA"]


def test_instruments_for_facility_gbt():
    assert NRAOTapClient.instruments_for_facility("GBT") == ["GBT"]


def test_instruments_for_facility_default():
    assert NRAOTapClient.instruments_for_facility(None) == ["VLA", "VLBA", "EVLA", "JVLA"]


# ── Routing: search_by_target ────────────────────────────────────────


def test_target_search_routes_vla_to_nrao(service):
    df = service.search_by_target("3C 273", facility="VLA")
    assert not df.empty
    assert service._nrao_client.calls == [("target", "3C 273", ("VLA", "EVLA", "JVLA"))]
    assert service.alminer_client.calls == []


def test_target_search_routes_gbt_instruments(service):
    service.search_by_target("M31", facility="GBT")
    assert service._nrao_client.calls == [("target", "M31", ("GBT",))]


def test_target_search_default_routes_to_alminer(service):
    df = service.search_by_target("M87")
    assert not df.empty
    assert service.alminer_client.calls == [("target", "M87")]
    assert service._nrao_client.calls == []


def test_target_search_alma_routes_to_alminer(service):
    service.search_by_target("M87", facility="ALMA")
    assert service.alminer_client.calls == [("target", "M87")]
    assert service._nrao_client.calls == []


# ── Routing: cone_search ─────────────────────────────────────────────


def test_cone_search_routes_vlba_to_nrao(service):
    df = service.cone_search(187.7, 12.39, 0.5, facility="VLBA")
    assert not df.empty
    assert service._nrao_client.calls == [("position", 187.7, 12.39, ("VLBA",))]
    assert service.alminer_client.calls == []


def test_cone_search_default_routes_to_alminer(service):
    service.cone_search(187.7, 12.39, 0.5)
    assert service.alminer_client.calls == [("position", 187.7, 12.39)]
    assert service._nrao_client.calls == []


# ── Routing: search_by_frequency ─────────────────────────────────────


def test_frequency_search_routes_vla_to_nrao(service):
    df = service.search_by_frequency(1.0, 2.0, facility="VLA")
    assert not df.empty
    assert service._nrao_client.calls == [("frequency", 1.0, 2.0, ("VLA", "EVLA", "JVLA"))]


def test_frequency_search_default_routes_to_alminer(service):
    service.search_by_frequency(90.0, 110.0)
    assert service.alminer_client.calls == [("frequency", 90.0, 110.0)]


# ── Degradation: NRAO client unavailable ─────────────────────────────


def test_unavailable_nrao_client_returns_empty(service):
    service._nrao_client = False  # sentinel set on construction failure
    df = service.search_by_target("3C 273", facility="VLA")
    assert df.empty
    assert service.alminer_client.calls == []  # must NOT silently fall back to ALMA
