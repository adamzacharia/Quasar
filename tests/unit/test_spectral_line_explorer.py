import threading
import time
import json

import pandas as pd
import pytest

from services.spectral_line_explorer import (
    ALMACoverageService,
    SpeciesMetadataCache,
    SpectralLineJobService,
    TargetResolver,
    analyze_confusion,
    confusion_score,
    radial_velocity_to_redshift,
)
from services.splatalogue import (
    SpectralLineQuery,
    SpectralWindow,
    SplatalogueClient,
    SplatalogueQueryCancelled,
    SplatalogueTool,
    normalize_spectral_windows,
)


pytestmark = pytest.mark.slow


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, polls):
        self.polls = list(polls)
        self.posts = []
        self.gets = []

    def post(self, url, json, timeout):
        self.posts.append((url, json, timeout))
        return FakeResponse([{"searchErrorMessage": "using thread", "requestnumber": 42}])

    def get(self, url, timeout):
        self.gets.append((url, timeout))
        return FakeResponse(self.polls.pop(0))


def test_window_units_overlap_and_observed_frame_conversion():
    windows = normalize_spectral_windows(
        [
            {"minimum": 1.0, "maximum": 2.0, "unit": "mm"},
            {"minimum": 149.8, "maximum": 310.0, "unit": "GHz"},
        ]
    )
    assert len(windows) == 1
    assert windows[0].minimum_ghz == pytest.approx(149.8)
    assert windows[0].maximum_ghz > 299.0

    query = SpectralLineQuery.from_payload(
        {
            "windows": [{"minimum": 100, "maximum": 101, "unit": "GHz"}],
            "frame": "observed",
            "redshift": 1,
        }
    )
    assert query.rest_windows()[0] == SpectralWindow(200, 202)


@pytest.mark.parametrize(
    ("convention", "velocity", "expected"),
    [
        ("optical", 2997.92458, 0.01),
        ("radio", 2997.92458, 0.0101010101),
    ],
)
def test_velocity_conversions(convention, velocity, expected):
    assert radial_velocity_to_redshift(velocity, convention) == pytest.approx(expected)


def test_threaded_submit_poll_and_sentinel_removal(monkeypatch):
    session = FakeSession(
        [
            [{"searchErrorMessage": "using thread"}],
            [
                {"species_id": 204, "lineid": 1, "orderedfreq": 230538.0},
                {"sqlquery": "select ...", "requestnumber": 42},
            ],
        ]
    )
    client = SplatalogueClient(session=session, timeout_seconds=5)
    monkeypatch.setattr(
        client,
        "_advanced_payload",
        lambda query: {"body": "{}", "headers": {}},
    )
    monkeypatch.setattr(time, "sleep", lambda value: None)

    rows = client._query_threaded(
        SpectralLineQuery(windows=[SpectralWindow(230.53, 230.55)]),
        cancel_event=None,
    )

    assert rows == [{"species_id": 204, "lineid": 1, "orderedfreq": 230538.0}]
    assert session.posts[0][1] == {"body": "{}", "headers": {}}
    assert session.gets[-1][0].endswith("/42")


def test_current_advanced_payload_uses_frequencies_unit():
    query = SpectralLineQuery(
        windows=[
            SpectralWindow(88.5, 89.2),
            SpectralWindow(230.53, 230.55),
        ],
        species_names=["CO"],
        maximum_frequency_uncertainty_mhz=50,
    )
    payload = SplatalogueClient._advanced_payload(query)
    body = json.loads(payload["body"])

    assert body["frequenciesUnit"] == "GHz"
    assert "userInputFrequenciesUnit" not in body
    assert body["userInputFrequenciesFrom"] == [88.5, 230.53]
    assert body["userInputFrequenciesTo"] == [89.2, 230.55]
    assert body["frequencyErrorLimit"] is True
    assert body["displayUnresolvedQuantumNumbers"] is True
    assert body["displayUniqueLineIDNumber"] is True
    assert body["displayObservationReference"] is True


def test_threaded_poll_honors_cancellation(monkeypatch):
    event = threading.Event()
    event.set()
    client = SplatalogueClient(session=FakeSession([]), timeout_seconds=5)
    monkeypatch.setattr(
        client,
        "_advanced_payload",
        lambda query: {"body": "{}", "headers": {}},
    )
    with pytest.raises(SplatalogueQueryCancelled):
        client._query_threaded(
            SpectralLineQuery(windows=[SpectralWindow(1, 2)]),
            cancel_event=event,
        )


def test_threaded_poll_surfaces_remote_error(monkeypatch):
    client = SplatalogueClient(
        session=FakeSession([[{"searchErrorMessage": "database unavailable"}]]),
        timeout_seconds=5,
    )
    monkeypatch.setattr(
        client,
        "_advanced_payload",
        lambda query: {"body": "{}", "headers": {}},
    )
    monkeypatch.setattr(time, "sleep", lambda value: None)
    with pytest.raises(RuntimeError, match="database unavailable"):
        client._query_threaded(
            SpectralLineQuery(windows=[SpectralWindow(1, 2)]),
            cancel_event=None,
        )


def test_frequency_parsing_merging_and_hfs_separation():
    tool = SplatalogueTool(client=object())
    first = tool._normalize_row(
        {
            "species_id": 204,
            "lineid": 1,
            "name": "CO",
            "resolved_QNs": "2-1",
            "unres_quantum_numbers": "F=2-1",
            "linelist": "JPL",
            "orderedfreq": 230538.0,
            "orderedFreq": "230.5380 (0.003), 229.5372",
            "measFreq": "230.5381 (0.001)",
            "Lovas_NRAO": 0,
        },
        query_frequency_ghz=None,
    )
    recommended = tool._normalize_row(
        {
            "species_id": 204,
            "lineid": 2,
            "name": "CO",
            "resolved_QNs": "2-1",
            "unres_quantum_numbers": "F=2-1",
            "linelist": "CDMS",
            "orderedfreq": 230538.0,
            "orderedFreq": "230.5380 (0.005), 229.5372",
            "Lovas_NRAO": 1,
        },
        query_frequency_ghz=None,
    )
    other_hfs = {
        **recommended,
        "line_id": "3",
        "raw_line_ids": ["3"],
        "unresolved_quantum_numbers": "F=1-0",
    }

    merged = tool._deduplicate([first, recommended, other_hfs])

    assert len(merged) == 2
    selected = next(item for item in merged if item["unresolved_quantum_numbers"] == "F=2-1")
    assert selected["source"] == "JPL, CDMS"
    assert selected["line_id"] == "2"
    assert selected["frequency_selection_reason"] == "NRAO-recommended frequency"
    assert first["predicted_frequency_uncertainty_mhz"] == pytest.approx(0.003)
    assert first["measured_frequency_ghz"] == pytest.approx(230.5381)
    assert first["observed_frequency_ghz"] == pytest.approx(229.5372)


def test_band_overlap_and_edge_warning():
    assert SplatalogueTool._alma_bands_for_frequency(100) == [2, 3]
    assert SplatalogueTool._preferred_alma_band(100, [2, 3]) == 3
    warning = SplatalogueTool._band_edge_warning(211.1)
    assert warning and "5, 6" in warning


def test_species_autocomplete_prioritizes_exact_formula(tmp_path):
    cache = SpeciesMetadataCache(cache_dir=tmp_path / "species")
    cache.cache.set(
        cache.CACHE_KEY,
        [
            {"species_id": 238, "formula": "CoC", "chemical_name": "Cobalt carbide", "tag": "06301", "label": "CoC"},
            {"species_id": 204, "formula": "CO v=0", "chemical_name": "Carbon Monoxide", "tag": "02801", "label": "CO v=0"},
            {"species_id": 1154, "formula": "H(alpha)", "chemical_name": "Hydrogen Recombination Line", "tag": "00103", "label": "H(alpha)"},
        ],
    )
    cache.cache.set(f"{cache.CACHE_KEY}:refreshed_at", time.time())
    try:
        assert cache.search("CO", limit=3)[0]["species_id"] == 204
    finally:
        cache.cache.close()


def test_target_resolver_agreement_conflict_and_missing(monkeypatch):
    resolver = TargetResolver()
    monkeypatch.setattr(
        resolver,
        "_query_simbad",
        lambda target: {"service": "SIMBAD", "ra_deg": 1, "dec_deg": 2, "redshift": 0.0043},
    )
    monkeypatch.setattr(
        resolver,
        "_query_ned",
        lambda target: {"service": "NED", "ra_deg": 1.1, "dec_deg": 2.1, "redshift": 0.0044},
    )
    result = resolver.resolve("M87")
    assert result["state"] == "resolved"
    assert result["coordinate_source"] == "SIMBAD"
    assert result["redshift_source"] == "SIMBAD (agrees with NED)"

    monkeypatch.setattr(
        resolver,
        "_query_ned",
        lambda target: {"service": "NED", "redshift": 0.1},
    )
    assert resolver.resolve("M87")["state"] == "needs_input"

    monkeypatch.setattr(resolver, "_query_simbad", lambda target: {"service": "SIMBAD"})
    monkeypatch.setattr(resolver, "_query_ned", lambda target: {"service": "NED"})
    assert resolver.resolve("Unknown")["state"] == "needs_input"


def test_exact_spw_classification_and_any_all_grouping(monkeypatch):
    service = ALMACoverageService()
    frame = pd.DataFrame(
        [
            {
                "proposal_id": "P1",
                "member_ous_uid": "uid://A",
                "frequency_support": "229.52..229.56GHz",
                "s_ra": 10.0,
                "s_dec": 20.0,
                "t_exptime": 100,
                "s_resolution": 0.2,
            },
            {
                "proposal_id": "P2",
                "member_ous_uid": "uid://B",
                "frequency_support": "229.539..229.545GHz",
                "s_ra": 10.0,
                "s_dec": 20.0,
                "t_exptime": 50,
                "s_resolution": 0.1,
            },
        ]
    )
    monkeypatch.setattr(service, "_query_obscore", lambda **kwargs: frame)
    target = {"ra_deg": 10, "dec_deg": 20, "redshift": 0.00436}
    lines = [
        {
            "line_id": "co21",
            "species": "CO",
            "transition": "2-1",
            "frequency_ghz": 230.538,
            "observed_frequency_ghz": 229.5372,
            "frequency_uncertainty_mhz": 0.1,
        }
    ]
    result = service.query(
        target=target,
        lines=lines,
        tolerance_mhz=2,
        coverage_mode="any",
    )
    assert result["project_count"] == 2
    assert result["projects"][0]["proposal_id"] == "P1"
    assert result["projects"][0]["all_lines_full"] is True
    p2_match = result["projects"][1]["observations"][0]["matching_lines"][0]
    assert p2_match["best_classification"] == "partial"


def test_confusion_scoring_boundaries_and_exclusion():
    strong = {
        "line_id": "other",
        "species": "HCN",
        "transition": "3-2",
        "frequency_ghz": 100.00001,
        "observed_frequency_ghz": 100.00001,
        "frequency_uncertainty_mhz": 0.01,
        "nrao_recommended": True,
        "astronomically_observed": True,
        "catalogs": ["CDMS", "JPL", "SLAIM"],
    }
    scored = confusion_score(strong, target_frequency_ghz=100, window_mhz=10)
    assert scored["classification"] == "High"
    assert scored["score"] > 90

    result = analyze_confusion(
        {"line_id": "selected", "raw_line_ids": ["selected"], "species": "CO", "transition": "1-0", "frequency_ghz": 100},
        [
            {"line_id": "selected", "raw_line_ids": ["selected"], "species": "CO", "transition": "1-0", "frequency_ghz": 100},
            strong,
        ],
        window_mhz=10,
    )
    assert result["candidate_count"] == 1
    assert result["candidates"][0]["line_id"] == "other"


def test_confusion_velocity_offset_sign_is_optical_convention():
    # confusion-velocity-offset-sign: a candidate at HIGHER observed frequency
    # is BLUESHIFTED relative to the target -> NEGATIVE velocity offset.
    higher = confusion_score(
        {"observed_frequency_ghz": 100.010},
        target_frequency_ghz=100.0,
        window_mhz=20,
    )
    assert higher["velocity_offset_kms"] == pytest.approx(-29.979, abs=0.01)
    lower = confusion_score(
        {"observed_frequency_ghz": 99.990},
        target_frequency_ghz=100.0,
        window_mhz=20,
    )
    assert lower["velocity_offset_kms"] == pytest.approx(29.979, abs=0.01)


def test_project_ranking_zero_separation_and_zero_margin_not_penalized():
    # project-ranking-falsy-zero-inversion: 0.0 separation must rank the
    # on-target project first, not be treated as "missing".
    line = {
        "line_id": "L",
        "species": "CO",
        "transition": "2-1",
        "observed_frequency_ghz": 100.0,
    }

    def row(project, separation):
        return {
            "proposal_id": project,
            "angular_separation_arcsec": separation,
            "t_exptime": 100,
            "s_resolution": 1.0,
            "matching_lines": [
                {
                    "line": dict(line),
                    "best_classification": "full",
                    "usable": True,
                    "spws": [
                        {
                            "classification": "full",
                            "usable": True,
                            "edge_margin_mhz": 5.0,
                        }
                    ],
                }
            ],
        }

    projects = ALMACoverageService._group_projects(
        [row("OFFSET-45", 45.0), row("ON-TARGET", 0.0)],
        [dict(line)],
        "any",
    )
    assert [item["proposal_id"] for item in projects] == ["ON-TARGET", "OFFSET-45"]


def _make_confusion_service(tmp_path, monkeypatch, captured):
    service = SpectralLineJobService(
        cache_dir=tmp_path / "jobs",
        ttl_seconds=60,
        max_workers=1,
        max_rows=10,
    )

    def fake_query_catalog(query, cancel_event=None):
        captured.append(query)
        lines = [
            {
                "line_id": "cand",
                "raw_line_ids": ["cand"],
                "species": "HCN",
                "transition": "3-2",
                "frequency_ghz": 100.010,
                "catalogs": ["CDMS"],
            }
        ]
        return {
            "lines": [dict(item) for item in lines],
            "raw_lines": [dict(item) for item in lines],
            "backend": "test",
            "degraded": False,
            "warnings": [],
            "query_provenance": {"backend": "test"},
        }

    monkeypatch.setattr(service.splatalogue, "query_catalog", fake_query_catalog)
    return service


def _wait_terminal(service, job):
    deadline = time.time() + 3
    current = job
    while current["status"] not in {"succeeded", "partial", "failed"} and time.time() < deadline:
        time.sleep(0.02)
        current = service.get_job(user_id="u1", job_id=job["job_id"])
    return current


def test_confusion_without_redshift_runs_at_z0_and_discloses(tmp_path, monkeypatch):
    # confusion-no-redshift-always-fails + confusion-velocity-width-doubled
    captured = []
    service = _make_confusion_service(tmp_path, monkeypatch, captured)
    try:
        job = service.create_job(
            user_id="u1",
            operation="confusion",
            payload={
                "selected_line": {
                    "line_id": "sel",
                    "raw_line_ids": ["sel"],
                    "species": "CO",
                    "transition": "1-0",
                    "frequency_ghz": 100.0,
                },
                "velocity_width_kms": 100,
            },
        )
        current = _wait_terminal(service, job)
        assert current["status"] == "partial"  # disclosure warning present
        context = current["result_context"]
        assert context["assumed_redshift_zero"] is True
        assert any("assumed z=0" in warning for warning in current["warnings"])
        # FULL velocity width: half-width = (dv/2)/c * f = 16.678 MHz at 100 GHz
        assert context["window_mhz"] == pytest.approx(16.678, abs=0.01)
        query = captured[0]
        assert query.redshift == pytest.approx(0.0)
        assert query.windows[0].minimum_ghz == pytest.approx(100.0 - 0.016678, abs=1e-5)
        assert query.windows[0].maximum_ghz == pytest.approx(100.0 + 0.016678, abs=1e-5)
        # Candidate at higher observed frequency reports a negative
        # (blueshifted) velocity offset in the returned rows.
        assert current["rows"][0]["velocity_offset_kms"] == pytest.approx(
            -29.979, abs=0.01
        )
    finally:
        service.shutdown()


def test_confusion_rest_only_line_without_redshift_derives_observed_at_z0(
    tmp_path, monkeypatch
):
    # confusion-no-redshift-always-fails RESIDUAL (found live 2026-07-18): a
    # selected_line carrying ONLY rest_frequency_ghz with no redshift used to
    # die on "selected_line requires an observed frequency" because the z=0
    # default was applied AFTER the frequency-frame derivation. Observed ==
    # rest at z=0, so this must run and disclose, like any other no-redshift
    # confusion search.
    captured = []
    service = _make_confusion_service(tmp_path, monkeypatch, captured)
    try:
        job = service.create_job(
            user_id="u1",
            operation="confusion",
            payload={
                "selected_line": {
                    "line_id": "sel",
                    "raw_line_ids": ["sel"],
                    "species": "CO",
                    "transition": "1-0",
                    "rest_frequency_ghz": 100.0,
                },
                "window_mhz": 50,
            },
        )
        current = _wait_terminal(service, job)
        assert current["status"] == "partial"  # ran + disclosure, not an error
        assert current.get("error") is None
        context = current["result_context"]
        assert context["assumed_redshift_zero"] is True
        assert any("assumed z=0" in warning for warning in current["warnings"])
        query = captured[0]
        assert query.redshift == pytest.approx(0.0)
        # Window centered on rest==observed at z=0.
        assert query.windows[0].minimum_ghz == pytest.approx(100.0 - 0.05, abs=1e-6)
        assert query.windows[0].maximum_ghz == pytest.approx(100.0 + 0.05, abs=1e-6)
    finally:
        service.shutdown()


def test_cancel_never_downgrades_terminal_job_and_flag_survives_worker_write(
    tmp_path,
):
    # sle-cancel-clobbers-completed-job: locked read-modify-write on both sides.
    service = SpectralLineJobService(
        cache_dir=tmp_path / "jobs",
        ttl_seconds=60,
        max_workers=1,
        max_rows=10,
    )
    try:
        running = {
            "job_id": "job-1",
            "user_id": "u1",
            "operation": "catalog_search",
            "status": "running",
            "phase": "querying",
            "progress": 50,
            "payload": {},
            "created_at": "now",
            "updated_at": "now",
            "warnings": [],
            "summary": {},
            "result": None,
            "error": None,
            "cancel_requested": False,
        }
        service._save(dict(running))
        # User cancels while the job is running.
        public = service.cancel_job(user_id="u1", job_id="job-1")
        assert public["cancel_requested"] is True
        # Worker then finishes from its STALE local snapshot: the persisted
        # cancel flag must be merged, not clobbered, and the result kept.
        stale_worker_copy = dict(running)
        service._update(
            stale_worker_copy,
            status="succeeded",
            result={"lines": [{"line_id": "1"}]},
        )
        stored = service.cache.get("job-1")
        assert stored["status"] == "succeeded"
        assert stored["result"] == {"lines": [{"line_id": "1"}]}
        assert stored["cancel_requested"] is True
        # A late cancel against the terminal record must not downgrade it.
        public = service.cancel_job(user_id="u1", job_id="job-1")
        assert public["status"] == "succeeded"
        stored = service.cache.get("job-1")
        assert stored["status"] == "succeeded"
        assert stored["result"] == {"lines": [{"line_id": "1"}]}
    finally:
        service.shutdown()


def test_create_job_enforces_per_user_limit_under_lock(tmp_path, monkeypatch):
    # sle-double-submit-check-then-act: the limit check still rejects a third
    # job (now performed atomically with the initial save under self._lock).
    service = SpectralLineJobService(
        cache_dir=tmp_path / "jobs",
        ttl_seconds=60,
        max_workers=1,
        max_rows=10,
    )
    try:
        monkeypatch.setattr(service, "_active_jobs_for_user", lambda user_id: 2)
        with pytest.raises(ValueError, match="two spectral-line jobs"):
            service.create_job(user_id="u1", operation="catalog_search", payload={})
    finally:
        service.shutdown()


def test_job_pagination_cancellation_and_complete_export(tmp_path, monkeypatch):
    service = SpectralLineJobService(
        cache_dir=tmp_path / "jobs",
        ttl_seconds=60,
        max_workers=1,
        max_rows=10,
    )
    monkeypatch.setattr(
        service.splatalogue,
        "query_catalog",
        lambda query, cancel_event=None: {
            "lines": [
                {
                    "line_id": str(index),
                    "species": "CO",
                    "transition": "2-1",
                    "frequency_ghz": 230.538 + index * 1e-6,
                    "catalogs": ["CDMS"],
                }
                for index in range(3)
            ],
            "raw_lines": [
                {
                    "line_id": str(index),
                    "species": "CO",
                    "transition": "2-1",
                    "frequency_ghz": 230.538 + index * 1e-6,
                    "catalogs": ["CDMS"],
                }
                for index in range(3)
            ],
            "backend": "test",
            "degraded": False,
            "warnings": [],
            "query_provenance": {"backend": "test"},
        },
    )
    try:
        job = service.create_job(
            user_id="u1",
            operation="catalog_search",
            payload={
                "mode": "advanced",
                "query": {
                    "windows": [{"minimum": 230.53, "maximum": 230.55, "unit": "GHz"}],
                    "line_lists": ["CDMS"],
                    "exclude_categories": [],
                    "version": "vall",
                },
            },
        )
        deadline = time.time() + 3
        current = job
        while current["status"] not in {"succeeded", "partial", "failed"} and time.time() < deadline:
            time.sleep(0.02)
            current = service.get_job(user_id="u1", job_id=job["job_id"], page=1, page_size=2)
        assert current["status"] == "succeeded"
        assert current["pagination"]["total_rows"] == 3
        assert len(current["rows"]) == 2
        raw_page = service.get_job(
            user_id="u1",
            job_id=job["job_id"],
            page=1,
            page_size=10,
            dataset="raw_lines",
        )
        assert raw_page["pagination"]["dataset"] == "raw_lines"
        assert len(raw_page["rows"]) == 3
        with pytest.raises(PermissionError):
            service.get_job(
                user_id="another-user",
                job_id=job["job_id"],
                page=1,
                page_size=2,
            )

        filename, media_type, content = service.export(
            user_id="u1",
            job_id=job["job_id"],
            dataset="lines",
            format_name="json",
        )
        assert filename.endswith(".json")
        assert media_type == "application/json"
        assert len(__import__("json").loads(content)["lines"]) == 3
    finally:
        service.shutdown()
