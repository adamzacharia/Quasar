"""Unit tests for capabilities/calc.py (P1 family migration #7 — calculators + resolve)."""

import threading

import pytest

import capabilities.calc as calc
from capabilities.calc import (
    CAPABILITIES,
    CalculateAlmaSensitivity,
    CalculateBeam,
    CalculateRedshift,
    ConvertCoordinates,
    ResolveTarget,
    resolve_target,
)
from capabilities.base import CallContext


def _ctx():
    return CallContext(services={})


def _run(cap, **kwargs):
    return cap.run(cap.InputModel(**kwargs), _ctx()).to_native()


# ─────────────────────────────────────────────────────────────────────────────
# The four calculators delegate to services.astro_calculators verbatim
# ─────────────────────────────────────────────────────────────────────────────
def test_calculate_redshift_delegates(monkeypatch):
    monkeypatch.setattr(calc, "calculate_redshift", lambda z: {"success": True, "z": z, "dL": 42})
    out = _run(CalculateRedshift(), z=2.0)
    assert out == {"success": True, "z": 2.0, "dL": 42}


def test_calculate_redshift_wraps_exception():
    # z=None → the real astro_calculators raises → typed error dict.
    out = _run(CalculateRedshift(), z=None)
    assert out["success"] is False and "error" in out


def test_convert_coordinates_passes_all_params(monkeypatch):
    seen = {}

    def _fake(**kw):
        seen.update(kw)
        return {"success": True}

    monkeypatch.setattr(calc, "convert_coordinates", _fake)
    _run(ConvertCoordinates(), l=120.0, b=30.0, input_frame="galactic")
    assert seen == {"ra": None, "dec": None, "l": 120.0, "b": 30.0,
                    "input_frame": "galactic", "output_frame": "galactic"}


def test_calculate_beam_delegates(monkeypatch):
    monkeypatch.setattr(calc, "calculate_beam",
                        lambda **kw: {"success": True, "beam_arcsec": 0.1, "kw": kw})
    out = _run(CalculateBeam(), frequency_ghz=230.0, array_config="C-6")
    assert out["success"] is True
    assert out["kw"] == {"frequency_ghz": 230.0, "max_baseline_m": None, "array_config": "C-6"}


def test_calculate_alma_sensitivity_defaults(monkeypatch):
    seen = {}

    def _fake(**kw):
        seen.update(kw)
        return {"success": True}

    monkeypatch.setattr(calc, "calculate_alma_sensitivity", _fake)
    _run(CalculateAlmaSensitivity(), band=6)
    assert seen["band"] == 6 and seen["bandwidth_ghz"] == 7.5
    assert seen["t_integration_s"] == 60.0 and seen["n_polarizations"] == 2
    assert seen["pwv_mm"] == 1.0 and seen["n_antennas"] is None
    assert seen["array"] == "12m" and seen["robust_weighting_factor"] == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# resolve_target (module function + capability)
# ─────────────────────────────────────────────────────────────────────────────
def _no_sesame(monkeypatch):
    """Make the Sesame step fail offline so the SIMBAD fallback path is exercised."""
    from astropy.coordinates import SkyCoord
    from astropy.coordinates.name_resolve import NameResolveError

    def _unavailable(name, *args, **kwargs):
        raise NameResolveError("sesame unavailable in tests")

    monkeypatch.setattr(SkyCoord, "from_name", classmethod(lambda cls, name, *a, **k: _unavailable(name)))


class _FakeSimbadResult:
    def __init__(self, ra, dec, colnames=("ra", "dec")):
        self._data = {colnames[0]: [ra], colnames[1]: [dec]}
        self.colnames = list(colnames)

    def __len__(self):
        return 1

    def __getitem__(self, key):
        return self._data[key]


def test_resolve_target_success(monkeypatch):
    _no_sesame(monkeypatch)
    import astroquery.simbad as sq
    monkeypatch.setattr(sq.Simbad, "query_object",
                        staticmethod(lambda name: _FakeSimbadResult(187.7059, 12.3911)))
    out = resolve_target("M87")
    assert out["success"] is True
    assert out["ra_deg"] == 187.7059 and out["dec_deg"] == 12.3911
    assert out["target_name"] == "M87"


def test_resolve_target_not_found(monkeypatch):
    _no_sesame(monkeypatch)
    import astroquery.simbad as sq
    monkeypatch.setattr(sq.Simbad, "query_object", staticmethod(lambda name: None))
    out = resolve_target("Nowhere")
    assert out["success"] is False and "SIMBAD could not resolve 'Nowhere'" in out["error"]


def test_resolve_target_capability_wraps_module_fn(monkeypatch):
    _no_sesame(monkeypatch)
    import astroquery.simbad as sq
    monkeypatch.setattr(sq.Simbad, "query_object",
                        staticmethod(lambda name: _FakeSimbadResult(10.0, -5.0)))
    out = _run(ResolveTarget(), target_name="X")
    assert out["success"] is True and out["ra_deg"] == 10.0


def test_resolve_target_null_name_is_caught(monkeypatch):
    _no_sesame(monkeypatch)
    import astroquery.simbad as sq

    def _boom(name):
        raise ValueError("bad name")

    monkeypatch.setattr(sq.Simbad, "query_object", staticmethod(_boom))
    out = resolve_target(None)
    assert out["success"] is False and out["error"].startswith("Resolution failed:")


# ─────────────────────────────────────────────────────────────────────────────
# agent wiring
# ─────────────────────────────────────────────────────────────────────────────
def _wiring_agent():
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
    return agent


def test_calc_registrations_keep_their_legacy_surface():
    agent = _wiring_agent()
    agent._register_tools()

    rt = agent.tool_registry.get_tool("resolve_target")
    assert rt is not None and rt.parameters["required"] == ["target_name"]

    cz = agent.tool_registry.get_tool("calculate_redshift")
    assert cz.parameters["required"] == ["z"] and cz.category == "analysis"

    cb = agent.tool_registry.get_tool("calculate_beam")
    assert cb.parameters["required"] == ["frequency_ghz"]

    cs = agent.tool_registry.get_tool("calculate_alma_sensitivity")
    assert cs.parameters["required"] == ["band"]


def test_agent_resolve_target_delegates_to_module_fn(monkeypatch):
    # The thin agent method other families receive by injection still works.
    agent = _wiring_agent()
    _no_sesame(monkeypatch)
    import astroquery.simbad as sq
    monkeypatch.setattr(sq.Simbad, "query_object",
                        staticmethod(lambda name: _FakeSimbadResult(1.0, 2.0)))
    out = agent._resolve_target("X")
    assert out["success"] is True and out["ra_deg"] == 1.0


def test_calc_tool_fn_end_to_end(monkeypatch):
    agent = _wiring_agent()
    monkeypatch.setattr(calc, "calculate_beam", lambda **kw: {"success": True, "beam_arcsec": 0.05})
    out = agent._calc_tool_fn("calculate_beam")(frequency_ghz=100.0)
    assert out["success"] is True and out["beam_arcsec"] == 0.05


def test_calc_unknown_capability_name_raises():
    agent = _wiring_agent()
    with pytest.raises(KeyError):
        agent._calc_tool_fn("not_a_tool")


def test_every_calc_capability_is_registered():
    agent = _wiring_agent()
    agent._register_tools()
    for cap in CAPABILITIES:
        assert agent.tool_registry.get_tool(cap.name) is not None, cap.name


# ─────────────────────────────────────────────────────────────────────────────
# S22 gap-fill (masterplan): numeric parity against the real
# services.astro_calculators — no monkeypatching, real math.
# ─────────────────────────────────────────────────────────────────────────────
def test_redshift_z2_planck18_numeric_parity():
    # Planck18 (H0=67.66, Om0=0.3111) at z=2 — values pinned from the legacy
    # inline implementation (byte-parity migration).
    out = _run(CalculateRedshift(), z=2.0)
    assert out["success"] is True
    assert "Planck18" in out["cosmology"]
    assert out["luminosity_distance_mpc"] == pytest.approx(15924.57, rel=1e-3)
    assert out["angular_diameter_distance_mpc"] == pytest.approx(1769.4, rel=1e-3)
    assert out["comoving_distance_mpc"] == pytest.approx(5308.19, rel=1e-3)
    assert out["lookback_time_gyr"] == pytest.approx(10.51, rel=1e-2)
    assert out["age_of_universe_at_z_gyr"] == pytest.approx(3.277, rel=1e-2)
    assert out["physical_scale_kpc_per_arcsec"] == pytest.approx(8.5783, rel=1e-3)
    assert out["distance_modulus_mag"] == pytest.approx(46.01, abs=0.02)


def test_beam_230ghz_16km_numeric_parity():
    # theta = 1.22 * lambda / B_max: 230 GHz → λ=1.303 mm; 16 km baseline →
    # 0.0205 arcsec synthesized beam.
    out = _run(CalculateBeam(), frequency_ghz=230.0, max_baseline_m=16000)
    assert out["success"] is True
    assert out["wavelength_mm"] == pytest.approx(1.303, abs=0.002)
    assert out["synthesized_beam_arcsec"] == pytest.approx(0.0205, abs=0.0005)
    assert out["synthesized_beam_mas"] == pytest.approx(20.5, abs=0.5)


def test_alma_sensitivity_band6_numeric_parity():
    # Technical Handbook (Cycle 13) eq. 9.8 with the ASC defaults:
    #   dS = 2 k Tsys / (eta_q eta_c A_eff sqrt(N(N-1) n_pol dnu t))
    # Band 6: Tsys 120 K (representative), eta_ap 0.68 (Table 9.3) x 113.1 m^2
    # = 76.91 m^2, eta_q 0.96, eta_c 0.88, N = 43, n_pol = 2, dnu = 7.5 GHz,
    # t = 60 s:
    #   sqrt(43*42*2*7.5e9*60) = sqrt(1.6254e15) = 4.0316e7
    #   denominator = 0.96*0.88*76.91*4.0316e7 = 2.6194e9
    #   dS = 2*1.380649e-23*120 / 2.6194e9 = 1.265e-30 W m^-2 Hz^-1 = 126.5 uJy
    # The former pin (135.7 uJy with 50 antennas, eta 0.65, no efficiencies)
    # used 2kT with N(N-1)/2 under the root: sqrt(2) = 1.41x too pessimistic
    # (report G-12); with eta = 1 and 50 antennas the standard form gives 96 uJy.
    out = _run(CalculateAlmaSensitivity(), band=6)
    assert out["success"] is True
    assert out["frequency_ghz"] == pytest.approx(230.0)
    assert out["tsys_k"] == 120
    assert out["n_antennas"] == 43 and out["n_baselines"] == 903
    assert out["quantization_efficiency"] == 0.96 and out["correlator_efficiency"] == 0.88
    assert out["continuum_sensitivity_ujy_beam"] == pytest.approx(126.5, rel=1e-2)
    assert "N*(N-1)" in out["formula"] and "/2" not in out["formula"]
    # eta = 1, 50 antennas, Tsys 120: the textbook 2kT/(A sqrt(N(N-1) n_pol dnu t))
    # = 96 uJy scaled by 1/(0.96*0.88) and eta_ap 0.68/0.65 -> 108.6 uJy.
    out50 = _run(CalculateAlmaSensitivity(), band=6, n_antennas=50, tsys_k=120)
    assert out50["continuum_sensitivity_ujy_beam"] == pytest.approx(108.6, rel=1e-2)


def test_convert_coordinates_m31_frames_numeric_parity():
    # M31 (ICRS 10.68458, +41.26917) → canonical Galactic l=121.174, b=−21.573
    # and ecliptic lon=27.849, lat=33.349. Independent truth: SIMBAD/astropy.
    out = _run(ConvertCoordinates(), ra=10.68458, dec=41.26917,
               input_frame="icrs", output_frame="galactic")
    assert out["success"] is True
    assert out["galactic"]["l_deg"] == pytest.approx(121.174242, abs=1e-3)
    assert out["galactic"]["b_deg"] == pytest.approx(-21.572886, abs=1e-3)
    assert out["ecliptic"]["lon_deg"] == pytest.approx(27.849273, abs=1e-3)
    assert out["ecliptic"]["lat_deg"] == pytest.approx(33.349023, abs=1e-3)
    assert out["icrs"]["ra_hms"].startswith("0:42:44")
