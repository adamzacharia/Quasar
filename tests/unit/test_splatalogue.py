from astropy.table import Table
import pytest

from services.cube_workbench import CubeWorkbenchService
from services.splatalogue import SplatalogueClient, SplatalogueTool


def _astroquery_table():
    return Table(
        rows=[
            {
                "species_id": 204,
                "name": 'CO <font color="red">v = 0</font>',
                "chemical_name": "Carbon Monoxide",
                "resolved_QNs": "2-1",
                "linelist": "CDMS",
                "intintensity": -4.1197,
                "sijmu2": 0.02423,
                "aij": -6.1605,
                "orderedfreq": 230538.0,
                "lower_state_energy_K": 5.53207,
                "upper_state_energy_K": 16.59608,
                "lineid": 363614,
                "transition_in_space": 1,
                "Lovas_NRAO": 1,
            },
            {
                "species_id": 204,
                "name": 'CO <font color="red">v = 0</font>',
                "chemical_name": "Carbon Monoxide",
                "resolved_QNs": "2-1",
                "linelist": "JPL",
                "intintensity": -4.1197,
                "sijmu2": 0.02423,
                "aij": -6.1605,
                "orderedfreq": 230538.0,
                "lower_state_energy_K": 5.53207,
                "upper_state_energy_K": 16.59608,
                "lineid": 363615,
                "transition_in_space": 1,
                "Lovas_NRAO": 1,
            },
            {
                "species_id": 999,
                "name": "Nearby molecule",
                "chemical_name": "Nearby",
                "resolved_QNs": "10-9",
                "linelist": "CDMS",
                "intintensity": -3.0,
                "sijmu2": 12.0,
                "aij": -4.0,
                "orderedfreq": 230537.5,
                "lower_state_energy_K": 20.0,
                "upper_state_energy_K": 30.0,
                "lineid": 999001,
                "transition_in_space": 0,
                "Lovas_NRAO": 0,
            },
        ]
    )


def _slap_table():
    return Table(
        rows=[
            {
                "line_id": "slap-co",
                "title": "CDMS: CO 2-1",
                "catalog name": "CDMS",
                "frequency": 230538.0,
                "molecular formula": "CO",
                "chemicalname": "Carbon Monoxide",
                "sijmu2": 0.02423,
                "aij": -6.1605,
                "lowerstateenergyK": 5.53207,
                "upperstateenergyK": 16.59608,
                "frequency recommended": 1,
                "quantum numbers": "2-1",
                "known_interstellar": True,
            },
            {
                "line_id": "slap-other",
                "title": "CDMS: Other 2-1",
                "catalog name": "CDMS",
                "frequency": 230538.2,
                "molecular formula": "OTHER",
                "chemicalname": "Other molecule",
                "sijmu2": 10.0,
                "aij": -3.0,
                "lowerstateenergyK": 200.0,
                "upperstateenergyK": 250.0,
                "frequency recommended": 0,
                "quantum numbers": "2-1",
                "known_interstellar": False,
            },
        ]
    )


def test_identify_normalizes_current_columns_ranks_and_deduplicates(monkeypatch):
    monkeypatch.setattr(
        SplatalogueTool,
        "_query_astroquery",
        staticmethod(lambda *args, **kwargs: _astroquery_table()),
    )

    result = SplatalogueTool().identify_spectral_line(
        frequency_ghz=230.538,
        tolerance_ghz=0.002,
        top_n=2,
    )

    assert result["backend"] == "astroquery"
    assert result["total_matches"] == 2
    assert result["n_matches"] == 2
    assert result["lines"][0]["species"] == "CO v = 0"
    assert result["lines"][0]["molecule"] == "Carbon Monoxide"
    assert result["lines"][0]["transition"] == "2-1"
    assert result["lines"][0]["frequency_ghz"] == pytest.approx(230.538)
    assert result["lines"][0]["offset_mhz"] == pytest.approx(0.0)
    assert result["lines"][0]["catalogs"] == ["CDMS", "JPL"]
    assert result["lines"][0]["source"] == "CDMS, JPL"
    assert result["lines"][0]["upper_energy_k"] == pytest.approx(16.59608)
    assert result["lines"][0]["astronomically_observed"] is True
    assert result["lines"][0]["nrao_recommended"] is True


def test_slap_fallback_applies_molecule_and_physical_filters(monkeypatch):
    def fail_astroquery(*args, **kwargs):
        raise RuntimeError("primary unavailable")

    monkeypatch.setattr(
        SplatalogueTool,
        "_query_astroquery",
        staticmethod(fail_astroquery),
    )
    monkeypatch.setattr(
        SplatalogueTool,
        "_query_slap",
        staticmethod(lambda *args, **kwargs: _slap_table()),
    )

    result = SplatalogueTool().search_spectral_lines(
        230.53,
        230.55,
        molecule_name="CO",
        energy_max=100,
        energy_type="eu_k",
        only_astronomically_observed=True,
        line_lists=["CDMS"],
    )

    assert result["backend"] == "slap"
    assert result["n_matches"] == 1
    assert result["lines"][0]["species"] == "CO"
    assert result["lines"][0]["frequency_ghz"] == pytest.approx(230.538)
    assert "Astroquery request failed" in result["note"]
    assert result["details"] == ["astroquery: primary unavailable"]


def test_internal_slap_fallback_is_labeled_slap_not_astroquery(monkeypatch):
    # slap-fallback-masquerades-as-advanced: when both Advanced endpoints are
    # down, SplatalogueClient.query must NOT silently hand raw SLAP rows to
    # _query_astroquery — the outer search_spectral_lines SLAP path must run so
    # results are filtered and labeled backend='slap', degraded=True.
    def advanced_down(self, *args, **kwargs):
        raise RuntimeError("advanced unavailable")

    monkeypatch.setattr(SplatalogueClient, "_query_threaded", advanced_down)
    monkeypatch.setattr(SplatalogueClient, "_query_non_threaded", advanced_down)
    monkeypatch.setattr(
        SplatalogueTool,
        "_query_slap",
        staticmethod(lambda *args, **kwargs: _slap_table()),
    )

    # version='vall' passes _slap_unsupported_filters, so before the fix the
    # client's INTERNAL fallback would have succeeded and been mislabeled.
    result = SplatalogueTool().search_spectral_lines(
        230.53,
        230.55,
        molecule_name="CO",
        version="vall",
        exclude=[],
    )

    assert result["backend"] == "slap"
    assert result["degraded"] is True
    assert result["n_matches"] == 1  # CO filter applied via _filter_slap_results
    assert result["lines"][0]["species"] == "CO"
    assert "Astroquery request failed" in result["note"]
    assert any("fallback" in warning.lower() for warning in result["warnings"])


def test_search_by_molecule_delegates_advanced_filters(monkeypatch):
    captured = {}

    def fake_search(self, **kwargs):
        captured.update(kwargs)
        return {"n_matches": 0, "total_matches": 0, "lines": [], "backend": "astroquery"}

    monkeypatch.setattr(SplatalogueTool, "search_spectral_lines", fake_search)

    result = SplatalogueTool().search_lines_by_molecule(
        "CH3OH",
        freq_min_ghz=90,
        freq_max_ghz=110,
        transition="2-1",
        energy_max=150,
        intensity_lower_limit=-5,
        intensity_type="CDMS/JPL (log)",
        line_lists=["CDMS"],
        top_n=7,
    )

    assert result["molecule"] == "CH3OH"
    assert captured == {
        "freq_min_ghz": 90,
        "freq_max_ghz": 110,
        "molecule_name": "CH3OH",
        "transition": "2-1",
        "energy_min": None,
        "energy_max": 150,
        "energy_type": "eu_k",
        "intensity_lower_limit": -5,
        "intensity_type": "CDMS/JPL (log)",
        "line_lists": ["CDMS"],
        "only_astronomically_observed": False,
        "only_nrao_recommended": False,
        "top_n": 7,
    }


def test_both_query_backends_fail_without_returning_fake_catalog_data(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(SplatalogueTool, "_query_astroquery", staticmethod(fail))
    monkeypatch.setattr(SplatalogueTool, "_query_slap", staticmethod(fail))

    result = SplatalogueTool().identify_spectral_line(230.538)

    assert result["backend"] == "unavailable"
    assert result["n_matches"] == 0
    assert result["lines"] == []
    assert result["error"] == "Splatalogue query failed"
    assert result["details"] == ["astroquery: offline", "SLAP: offline"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"freq_min_ghz": 10, "freq_max_ghz": 5}, "freq_min_ghz"),
        (
            {
                "freq_min_ghz": 5,
                "freq_max_ghz": 10,
                "intensity_lower_limit": -5,
            },
            "intensity_type",
        ),
        (
            {
                "freq_min_ghz": 5,
                "freq_max_ghz": 10,
                "line_lists": ["not-a-catalog"],
            },
            "Unsupported line list",
        ),
    ],
)
def test_advanced_query_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SplatalogueTool().search_spectral_lines(**kwargs)


def test_formula_queries_use_astroquery_exact_species_syntax():
    assert SplatalogueTool._chemical_query("CO") == " CO "
    assert SplatalogueTool._chemical_query("CH3OH") == " CH3OH "
    assert SplatalogueTool._chemical_query("SiO") == " SiO "
    assert SplatalogueTool._chemical_query("Methanol") == "Methanol"
    assert SplatalogueTool._chemical_query("Carbon Monoxide") == "Carbon Monoxide"


def test_workbench_propagates_splatalogue_backend_and_redshift(tmp_path):
    service = CubeWorkbenchService(
        session_dir=tmp_path / "sessions",
        cache_dir=tmp_path / "cache",
    )
    service._write_session(
        {
            "session_id": "line-session",
            "user_id": "user-1",
            "metadata": {},
            "state": {},
        }
    )
    captured = {}

    def fake_identify(**kwargs):
        captured.update(kwargs)
        return {
            "backend": "slap",
            "note": "Primary unavailable; used SLAP.",
            "lines": [
                {
                    "species": "CO v=0",
                    "transition": "2-1",
                    "frequency_ghz": 230.538,
                    "source": "CDMS",
                }
            ],
        }

    service.splatalogue.identify_spectral_line = fake_identify
    try:
        result = service.line_overlays(
            session_id="line-session",
            user_id="user-1",
            observed_frequency_ghz=115.269,
            redshift=1.0,
            tolerance_ghz=0.002,
        )
    finally:
        service.shutdown()

    assert captured["frequency_ghz"] == pytest.approx(230.538)
    assert result["backend"] == "slap"
    assert result["query_note"] == "Primary unavailable; used SLAP."
    assert result["lines"][0]["observed_frequency_ghz"] == pytest.approx(115.269)
    assert result["evidence"]["source"] == "Splatalogue via IVOA SLAP"
