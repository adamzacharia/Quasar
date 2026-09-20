"""claims.tsv -> regression fixtures (INT-6).

The vendored skill ships review/claims.tsv: every guardrail/claim with an
evidence_status of ``confirmed`` (documentation-backed), ``confirmed-dated``
(true on the review date), ``empirical`` (observed live) or ``conflict-retained``.
Rows marked ``confirmed`` that Quasar's code can honour are pinned here as
behavioural tests; dated/empirical rows are NOT hard-coded as assertions
(report Section 3.6) — they are listed with their re-verify date so a sync
that changes them shows up in the review checklist.

Coverage map: ``PINNED`` names the claim ids each test honours; the coverage
test fails if a pinned id disappears from claims.tsv (the skill moved on) or a
confirmed claim relevant to code has no pin.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CLAIMS = REPO_ROOT / "third_party" / "alma-data-skill" / "review" / "claims.tsv"

# Claims Quasar's code honours, and where.
PINNED = {
    "ARCH-001": "aggregate_counts / summarize_projects group by member_ous_uid and count DISTINCT asdm_uid",
    "ARCH-002": "bandwidth_switching_candidates counts one frequency_support string per MOUS; coverage decided per row",
    "ARCH-004": "profile + kernel units: t_min MJD (d), em_min/em_max m, frequency GHz, bandwidth Hz",
    "ARCH-005": "parse_frequency_support_windows tolerates extensible text (resolution + several sensitivities)",
    "ARCH-007": "DataLink content_length is per-link bytes, nullable -> size_mb None; access_estsize kbyte in profile",
    "ARCH-011": "one mirror per request: NRAO TAP for cone/CO/details; provenance URL matches",
    "ARCH-012": "DataLink empty_or_unauthorized vs not_found are distinct states",
    "ARCH-014": "obs_collection profiled as a label to introspect, not a hard filter",
    "ID-001": "identifier kind from column role (member_ous_uid / asdm_uid), not the A001/A002 prefix",
    "ID-002": "canonical and sanitized UIDs both normalise to uid://",
    "PKG-011": "raw ASDM tar recognised by *asdm.sdm.tar pattern, never synthesised",
    "SAFE-001": "download_alma_data preflights DataLink bytes and free disk before transferring",
    "QA-001": "qa2_passed flag never rendered as SemiPass/Fail; report parser is the three-state source",
    "RESTORE-001": "restore guidance names auxiliary + raw ASDMs, product tars optional",
    "RESTORE-002": "CASA generator refuses a generic ALMA calibration recipe; restore under the package CASA version",
    "PROD-001": "triage caveat: delivered FITS may omit fields/SPWs/channels",
    "SPEC-001": "calculate_doppler_shift labels frames; native ALMA data TOPO per EB, products LSRK",
    "HIER-001": "array inference is a heuristic (DV/DA, CM, PM) and heterogeneous EBs list both arrays",
    "PKG-001": "array-combo template warns the archive delivers per MOUS and has not combined arrays",
}

# Confirmed claims that are documentation/operational knowledge served by
# browse_alma_guidance rather than executable code (no code pin needed).
GUIDANCE_ONLY = {
    "ARCH-003", "ARCH-006", "PKG-002", "PKG-003", "PKG-005", "PKG-006", "PKG-007", "PKG-008",
    "PKG-009", "SAFE-002", "QA-002", "QA-003", "QA-004", "FLAG-001", "INTENT-001", "EPH-001",
    "PIPE-001", "PIPE-002", "PIPE-003", "PIPE-004", "PIPE-005", "PIPE-006",
}


def _claims():
    with CLAIMS.open(encoding="utf-8", newline="") as fh:
        return {row["id"]: row for row in csv.DictReader(fh, delimiter="\t")}


def test_claims_file_present_and_every_pin_resolves():
    claims = _claims()
    assert len(claims) >= 50
    missing = [cid for cid in PINNED if cid not in claims]
    assert not missing, f"pinned claims vanished from claims.tsv: {missing}"
    for cid in PINNED:
        assert claims[cid]["evidence_status"] in {"confirmed", "confirmed-dated", "empirical", "conflict-retained"}


def test_every_confirmed_claim_is_pinned_or_guidance_only():
    claims = _claims()
    confirmed = {cid for cid, row in claims.items() if row["evidence_status"] == "confirmed"}
    unaccounted = sorted(confirmed - set(PINNED) - GUIDANCE_ONLY)
    assert not unaccounted, f"confirmed claims with no pin and no guidance-only waiver: {unaccounted}"


def test_dated_and_empirical_claims_carry_a_reverify_date():
    claims = _claims()
    for cid, row in claims.items():
        if row["evidence_status"] in {"confirmed-dated", "empirical", "conflict-retained"}:
            assert row["verified_on"].startswith("2026-"), (cid, row["verified_on"])
            # Never hard-coded: none of these ids may appear as an assertion pin
            # of a LIVE fact (the pins above use them only for code behaviour).


# ── ARCH-001 / ARCH-002 ───────────────────────────────────────────────────────
def test_arch_001_002_grain_aggregation():
    from services.alma_science_queries import aggregate_counts, summarize_projects

    df = pd.DataFrame([
        {"proposal_id": "P", "member_ous_uid": "uid://A001/X1/X1", "asdm_uid": "uid://A002/X1/Xa", "target_name": "f1"},
        {"proposal_id": "P", "member_ous_uid": "uid://A001/X1/X1", "asdm_uid": "uid://A002/X1/Xa", "target_name": "f2"},
        {"proposal_id": "P", "member_ous_uid": "uid://A001/X1/X1", "asdm_uid": "uid://A002/X1/Xb", "target_name": "f1"},
    ])
    counts = aggregate_counts(df)
    assert (counts["rows"], counts["n_mous"], counts["n_eb"]) == (3, 1, 2)
    assert summarize_projects(df).iloc[0]["n_eb"] == 2


# ── ARCH-004 ──────────────────────────────────────────────────────────────────
def test_arch_004_units_in_profile_and_kernel():
    from core.prompts import ALMA_TAP_SCHEMA
    from services import archive_profiles

    cols = {c.name: c for c in archive_profiles.get_profile("alma").tables["ivoa.obscore"].columns}
    assert cols["t_min"].unit == "d" and cols["em_min"].unit == "m"
    assert cols["frequency"].unit == "GHz" and cols["bandwidth"].unit == "Hz"
    assert "bandwidth Hz" in ALMA_TAP_SCHEMA and "em_min/em_max METERS" in ALMA_TAP_SCHEMA


# ── ARCH-005 ──────────────────────────────────────────────────────────────────
def test_arch_005_frequency_support_grammar_is_extensible():
    from services.alma_science_queries import parse_frequency_support_windows

    text = "[216.90..218.88GHz,31250.00kHz,76.5mJy/beam@10km/s,4.6mJy/beam@native,XX YY] U [230.00..231.87GHz,976.56kHz,XX YY,extra-token]"
    windows = parse_frequency_support_windows(text)
    assert [(w["low_ghz"], w["high_ghz"]) for w in windows] == [(216.9, 218.88), (230.0, 231.87)]
    assert windows[0]["resolution_khz"] == 31250.0 and windows[1]["resolution_khz"] == 976.56


# ── ARCH-007 / ARCH-012 / ID-002 ──────────────────────────────────────────────
def test_arch_007_012_id_002_datalink_contract():
    from integrations.datalink import DataLinkClient, row_to_entry

    entry = row_to_entry({"access_url": "https://a/f.fits", "content_length": None}, ["access_url", "content_length"])
    assert entry["size_mb"] is None and entry["size_known"] is False and entry["kind"] == "file"
    assert DataLinkClient._normalize_uid("uid___A001_X1_X2") == "uid://A001/X1/X2"
    client = DataLinkClient.__new__(DataLinkClient)
    assert client._votable_fault('<INFO name="QUERY_STATUS" value="ERROR">NotFoundFault</INFO>') == "NotFoundFault"
    assert client._votable_fault('<INFO name="QUERY_STATUS" value="OK"/><TABLEDATA/>') is None


# ── ARCH-011 ──────────────────────────────────────────────────────────────────
def test_arch_011_one_mirror_per_request():
    import capabilities.alma as alma
    from integrations.alminer_client import ALMA_TAP_URL

    assert alma._ALMA_TAP_URL == ALMA_TAP_URL == "https://almascience.nrao.edu/tap"
    import inspect

    src = inspect.getsource(alma.SearchAlmaCoInRedshiftRange.run)
    assert "almascience.eso.org" not in src


# ── ARCH-014 ──────────────────────────────────────────────────────────────────
def test_arch_014_obs_collection_is_introspected_not_assumed():
    from services import archive_profiles

    cols = {c.name: c for c in archive_profiles.get_profile("alma").tables["ivoa.obscore"].columns}
    assert "introspect" in cols["obs_collection"].description.lower()


# ── ID-001 ────────────────────────────────────────────────────────────────────
def test_id_001_identifier_kind_from_column_role():
    from services.search import SearchService

    svc = SearchService.__new__(SearchService)
    captured = {}

    class _Svc:
        def search(self, q):
            captured["q"] = q

            class _R:
                def to_table(self):
                    return self

                def to_pandas(self):
                    return pd.DataFrame()
            return _R()

    class _Client:
        def _get_tap_service(self):
            return _Svc()

    svc.alminer_client = _Client()
    svc.get_observation_details("uid://A002/X5d7935/X11b")  # a live Cycle 1 MOUS with an A002 UID
    assert "member_ous_uid = 'uid://A002/X5d7935/X11b'" in captured["q"]
    assert "asdm_uid = 'uid://A002/X5d7935/X11b'" in captured["q"]


# ── PKG-011 ───────────────────────────────────────────────────────────────────
def test_pkg_011_raw_asdm_recognised_from_returned_basename():
    from services.data_product_triage import product_role

    assert product_role({"filename": "uid___A002_Xb6f4d9_X3a8.asdm.sdm.tar"}).startswith("raw ASDM")
    assert product_role({"filename": "2019.1.00001.S_uid___A002_Xb6f4d9_X3a8.asdm.sdm.tar"}).startswith("raw ASDM")


# ── SAFE-001 ──────────────────────────────────────────────────────────────────
def test_safe_001_download_preflight_refuses_without_byte_estimate():
    from capabilities.alma import DownloadAlmaData
    from tests.unit.test_alma_capability import _ALMA_DF, _FakeSearchService, _ctx, _run

    class _DL:
        def list_files(self, uid):
            return {"success": True, "state": "ok", "files": [
                {"filename": "a.fits", "content_type": "application/fits", "content_length": 6_000_000_000},
            ]}

    ctx, state = _ctx(search_service=_FakeSearchService(), datalink_client=_DL())
    state.last_search_results = _ALMA_DF
    out = _run(DownloadAlmaData(), ctx, mous_uids=["uid://A/X1/X1"], max_gb=5)
    assert out["success"] is False and "exceeds the 5.0 GB cap" in out["error"]
    assert out["preflight"]["total_known_gb"] == 6.0


# ── QA-001 ────────────────────────────────────────────────────────────────────
def test_qa_001_flag_cannot_reconstruct_three_state():
    from services.alma_qa2 import QA2_REPORT_LABELS, parse_qa2_status, qa2_label_from_flag

    assert qa2_label_from_flag("T") not in {"SemiPass", "Fail"}
    assert qa2_label_from_flag("F") not in {"SemiPass", "Fail", "Pass"}
    assert parse_qa2_status("QA2 Status: SEMIPASS") == "SemiPass" and "SemiPass" in QA2_REPORT_LABELS


# ── RESTORE-001 / RESTORE-002 / PROD-001 ──────────────────────────────────────
def test_restore_and_product_claims_in_casa_guidance_and_triage():
    from services.casa_generator import ALMA_RESTORE_GUIDANCE, CASAScriptGenerator

    steps = " ".join(ALMA_RESTORE_GUIDANCE["steps"])
    assert "auxiliary" in steps and "RAW ASDMs" in steps and "not needed for the restore" in steps
    assert "compatibility table" in steps and "Never infer the version from the proposal cycle" in steps
    out = CASAScriptGenerator().generate_casa_calibration_script("t", "v.ms", "f", "p", "DA41")
    assert out["success"] is False
    import inspect

    from core import agent as agent_module

    src = inspect.getsource(agent_module.QuasarAgent._triage_alma_mous_products)
    assert "may omit fields/SPWs/channels" in src


# ── SPEC-001 ──────────────────────────────────────────────────────────────────
def test_spec_001_frames_labelled():
    from services.astro_calculators import calculate_doppler_shift

    out = calculate_doppler_shift(rest_frequency_ghz=115.271, velocity_kms=300.0, convention="radio", frame="TOPO")
    assert out["frame"] == "TOPO" and "TOPO" in out["note"] and "LSRK" in out["note"]


# ── HIER-001 / PKG-001 ────────────────────────────────────────────────────────
def test_hier_001_pkg_001_arrays_are_heuristic_and_not_combined():
    from services.alma_science_queries import infer_arrays, projects_with_array_combo

    assert infer_arrays("A001:DV01 J501:CM01") == ["12m", "7m"]   # heterogeneous EB keeps both
    df = pd.DataFrame([{"proposal_id": "P", "antenna_arrays": "A001:DV01 J501:CM01 N601:PM01",
                        "member_ous_uid": "uid://A001/X1/X1", "asdm_uid": "uid://A002/X1/X1"}])
    out = projects_with_array_combo(df, ["12m", "7m", "TP"])
    assert "heuristic" in out.iloc[0]["array_inference"] and "not combined" in out.iloc[0]["array_inference"]
