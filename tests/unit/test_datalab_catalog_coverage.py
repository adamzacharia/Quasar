"""
RE-B5: structured coverage/band metadata on the curated Data Lab registry and
server-side curation (target/ra/dec/band filters) on datalab_list_catalogs.

Offline throughout — the registered scope never touches the network. Guards:
  * every registry row carries bands / wavelength_regime / coverage (shape);
  * an LMC position EXCLUDES the known non-covering catalogs (SDSS, DESI,
    LS DR9, DES — the exact rows the NOIRLab evaluator was shown as
    "covers the LMC: NO"), keeps known-covering ones (NSC, SMASH, ...) ranked
    first, and LABELS unknown coverage (DELVE, VHS, LS DR10) instead of
    dropping it — DELVE DR3 is deliberately UNVERIFIED at the LMC (CX-10:
    its prose asserts only Magellanic periphery coverage);
  * region/dec/hemisphere boundary behavior (CX-18);
  * target-vs-explicit-coordinate mismatch is a typed error (CX-12) and
    resolver output is range-validated like explicit input (CX-13);
  * band filtering (incl. the ks/k alias between VHS and 2MASS);
  * scope='all' curation interplay: schemas stay unfiltered (CX-19);
  * no-args back-compat: the full curated listing, untouched;
  * the note states the registered-subset semantics and names scope='all'.
"""

import pandas as pd

from capabilities.base import CallContext
from capabilities import datalab as dl
from integrations.datalab_client import DatalabResult
from services import datalab_registry as reg


LMC_RA, LMC_DEC = 80.894, -69.756
SMC_RA, SMC_DEC = 13.187, -72.829
GC_RA, GC_DEC = 266.417, -29.008  # Galactic center: |b| ~ 0

# The prompt-rule quartet the model wrongly tabulated at the evaluator.
KNOWN_NON_COVERING_LMC = {"sdss_dr17", "desi_dr1", "ls_dr9", "des_dr1"}
KNOWN_COVERING_LMC = {"gaia_dr3", "nsc_dr2", "smash_dr1", "smash_dr2"}
# Prose asserts periphery/incomplete/unknown coverage only — never "covered".
UNVERIFIED_AT_LMC = {"delve_dr3", "delve_dr2", "vhs_dr5", "ls_dr10"}


def _ctx(**services):
    return CallContext(services=services)


def _run(**kwargs):
    return dl.ListCatalogs().run(dl.ListCatalogsInput(**kwargs), _ctx()).to_native()


# ── shape: structured metadata on every registry row ─────────────────────────

def test_every_registry_row_carries_structured_coverage_fields():
    rows = reg.list_catalogs()
    assert len(rows) == 17  # curated 14 + 3 expansion
    for row in rows:
        name = row["catalog"]
        assert isinstance(row["bands"], list), name
        assert all(isinstance(b, str) and b == b.lower() for b in row["bands"]), name
        regime = row["wavelength_regime"]
        assert isinstance(regime, list) and regime, name
        assert set(regime) <= {"optical", "near-ir", "mid-ir"}, name
        cov = row["coverage"]
        assert isinstance(cov, dict) and cov, name
        assert cov.get("hemisphere") in {"north", "south", "both"}, name
        # notes mirror the footprint prose so the dict is self-contained.
        assert cov.get("notes") == row["footprint"], name
        for key in ("all_sky", "covers_lmc", "covers_smc", "covers_galactic_plane"):
            if key in cov:
                assert isinstance(cov[key], bool), (name, key)
        for key in ("dec_min", "dec_max"):
            if key in cov:
                assert isinstance(cov[key], float), (name, key)


def test_photometric_bands_match_mag_template_bands_where_registered():
    by_name = {row["catalog"]: row for row in reg.list_catalogs()}
    assert by_name["gaia_dr3"]["bands"] == ["g", "bp", "rp"]
    assert by_name["unwise_dr1"]["bands"] == ["w1", "w2"]
    assert by_name["allwise"]["bands"] == ["w1", "w2", "w3", "w4"]
    assert by_name["twomass"]["bands"] == ["j", "h", "k"]
    assert by_name["vhs_dr5"]["bands"] == ["j", "h", "ks"]
    # Spectroscopic catalogs list no photometric bandpasses.
    assert by_name["desi_dr1"]["bands"] == []
    assert by_name["sdss_dr17"]["bands"] == []


# ── registry-level classification ─────────────────────────────────────────────

def test_coverage_status_negative_booleans_win_at_lmc():
    by_name = {row["catalog"]: row for row in reg.list_catalogs()}
    for name in KNOWN_NON_COVERING_LMC:
        status, reason = reg.coverage_status_for_position(
            by_name[name]["coverage"], LMC_RA, LMC_DEC
        )
        assert status == "not_covering", name
        assert "does not cover" in reason, name
    for name in KNOWN_COVERING_LMC:
        status, _ = reg.coverage_status_for_position(
            by_name[name]["coverage"], LMC_RA, LMC_DEC
        )
        assert status == "covered", name
    # Periphery/incomplete prose (DELVE DR3/DR2, VHS, LS DR10) must classify
    # UNVERIFIED — never covered (CX-10), never silently dropped.
    for name in UNVERIFIED_AT_LMC:
        status, reason = reg.coverage_status_for_position(
            by_name[name]["coverage"], LMC_RA, LMC_DEC
        )
        assert status == "unverified", name
        assert reason == "coverage unverified for this position", name


def test_coverage_status_dec_bounds_exclude_but_never_prove():
    by_name = {row["catalog"]: row for row in reg.list_catalogs()}
    # Deep-southern non-Magellanic position: SDSS/DESI are dec-bound excluded.
    status, reason = reg.coverage_status_for_position(by_name["sdss_dr17"]["coverage"], 40.0, -45.0)
    assert status == "not_covering" and "declination limit" in reason
    status, _ = reg.coverage_status_for_position(by_name["desi_dr1"]["coverage"], 40.0, -45.0)
    assert status == "not_covering"
    # Being INSIDE a survey's dec range proves nothing: DES at a northern
    # equatorial point inside [-65, +5] is still unverified, not covered.
    status, _ = reg.coverage_status_for_position(by_name["des_dr1"]["coverage"], 150.0, 2.0)
    assert status == "unverified"
    # All-sky catalogs are covered anywhere.
    status, reason = reg.coverage_status_for_position(by_name["gaia_dr3"]["coverage"], 150.0, 2.0)
    assert status == "covered" and reason == "all-sky catalog"


def test_coverage_status_galactic_plane_booleans():
    by_name = {row["catalog"]: row for row in reg.list_catalogs()}
    assert abs(reg.galactic_latitude_deg(GC_RA, GC_DEC)) < 1.0  # sanity: it IS the plane
    status, reason = reg.coverage_status_for_position(by_name["des_dr1"]["coverage"], GC_RA, GC_DEC)
    assert status == "not_covering" and "Galactic plane" in reason
    status, _ = reg.coverage_status_for_position(by_name["twomass"]["coverage"], GC_RA, GC_DEC)
    assert status == "covered"
    # NSC's "much of the Galactic plane" is patchy, not pointwise — no boolean,
    # so a plane position classifies unverified (CX-11).
    status, _ = reg.coverage_status_for_position(by_name["nsc_dr2"]["coverage"], GC_RA, GC_DEC)
    assert status == "unverified"
    # DELVE DR3 only claims to avoid the INNER plane — no boolean, so unverified.
    status, _ = reg.coverage_status_for_position(by_name["delve_dr3"]["coverage"], GC_RA, GC_DEC)
    assert status == "unverified"


# ── capability: position curation ─────────────────────────────────────────────

def test_lmc_position_excludes_known_non_covering_and_ranks_covering_first():
    out = _run(ra=LMC_RA, dec=LMC_DEC)
    assert out["success"] is True and out["scope"] == "registered"

    excluded = {row["catalog"] for row in out["excluded_by_coverage"]}
    assert excluded == KNOWN_NON_COVERING_LMC
    for row in out["excluded_by_coverage"]:
        assert row["reason"]  # every exclusion is justified

    kept = [row["catalog"] for row in out["catalogs"]]
    assert KNOWN_COVERING_LMC <= set(kept)
    assert not (KNOWN_NON_COVERING_LMC & set(kept))
    assert out["count"] == len(out["catalogs"]) == 17 - len(excluded)
    assert out["total_registered"] == 17

    statuses = [row["coverage_status"] for row in out["catalogs"]]
    # Known-covering first, then the labeled-unverified tail — never interleaved.
    first_unverified = statuses.index("coverage unverified for this position")
    assert all(s == "covered" for s in statuses[:first_unverified])
    assert all(s == "coverage unverified for this position" for s in statuses[first_unverified:])
    by_name = {row["catalog"]: row for row in out["catalogs"]}
    for name in KNOWN_COVERING_LMC:
        assert by_name[name]["coverage_status"] == "covered", name
    # Periphery/incomplete-prose catalogs ride along, labeled — never covered
    # (CX-10: delve_dr3), never dropped.
    for name in UNVERIFIED_AT_LMC:
        assert by_name[name]["coverage_status"] == "coverage unverified for this position", name

    assert out["filters"]["position"]["ra"] == LMC_RA
    assert "excluded_by_coverage" in out["note"]


def test_target_name_resolves_through_ctx_coordinate_resolver():
    calls = {}

    def fake_resolver(target_name=None, ra=None, dec=None):
        calls["target"] = target_name
        return LMC_RA, LMC_DEC, target_name

    ctx = _ctx(resolve_coordinates=fake_resolver)
    out = dl.ListCatalogs().run(dl.ListCatalogsInput(target="LMC"), ctx).to_native()
    assert calls["target"] == "LMC"
    assert out["filters"]["position"]["label"] == "LMC"
    assert {row["catalog"] for row in out["excluded_by_coverage"]} == KNOWN_NON_COVERING_LMC


def test_position_input_validation():
    out = _run(ra=LMC_RA)  # dec missing
    assert out["success"] is False and "BOTH ra and dec" in out["error"]
    out = _run(ra=400.0, dec=0.0)
    assert out["success"] is False and "ICRS" in out["error"]


def test_target_plus_disagreeing_coords_is_a_typed_error():
    # CX-12: 'LMC' resolving ~1 deg away from the explicit ra/dec must FAIL
    # loudly — never silently label unrelated coordinates with the name.
    def fake_resolver(target_name=None, ra=None, dec=None):
        return LMC_RA, LMC_DEC, target_name

    ctx = _ctx(resolve_coordinates=fake_resolver)
    out = dl.ListCatalogs().run(
        dl.ListCatalogsInput(target="LMC", ra=LMC_RA, dec=LMC_DEC + 1.0), ctx
    ).to_native()
    assert out["success"] is False
    assert "target/coordinate mismatch" in out["error"]
    assert "'LMC'" in out["error"] and "arcmin" in out["error"]


def test_target_plus_agreeing_coords_proceeds_with_target_label():
    # CX-12: within the 1-arcmin tolerance the call proceeds, labeled by name.
    def fake_resolver(target_name=None, ra=None, dec=None):
        return LMC_RA, LMC_DEC, target_name

    ctx = _ctx(resolve_coordinates=fake_resolver)
    out = dl.ListCatalogs().run(
        dl.ListCatalogsInput(target="LMC", ra=LMC_RA, dec=LMC_DEC + 0.005), ctx
    ).to_native()
    assert out["success"] is True
    assert out["filters"]["position"]["label"] == "LMC"
    assert {row["catalog"] for row in out["excluded_by_coverage"]} == KNOWN_NON_COVERING_LMC


def test_resolver_returned_coordinates_are_range_validated():
    # CX-13: resolver output gets the same ICRS range check as explicit input.
    def broken_resolver(target_name=None, ra=None, dec=None):
        return 999.0, 12.0, target_name

    ctx = _ctx(resolve_coordinates=broken_resolver)
    out = dl.ListCatalogs().run(dl.ListCatalogsInput(target="LMC"), ctx).to_native()
    assert out["success"] is False
    assert "resolved for target 'LMC'" in out["error"] and "ICRS" in out["error"]


# ── classification boundaries (CX-18) ─────────────────────────────────────────

def _dec_at_galactic_latitude(target_b, ra=GC_RA, lo=GC_DEC, hi=-65.0):
    """Dec on the ra meridian where b == target_b (b decreases from ~0 at the
    Galactic center toward the south Galactic pole along this meridian)."""
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if reg.galactic_latitude_deg(ra, mid) > target_b:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def test_lmc_cone_boundary_five_degrees():
    by_name = {row["catalog"]: row for row in reg.list_catalogs()}
    smash, des = by_name["smash_dr1"]["coverage"], by_name["des_dr1"]["coverage"]
    inside_dec, outside_dec = LMC_DEC + 4.9, LMC_DEC + 5.1  # pure-dec offsets = exact separation
    status, reason = reg.coverage_status_for_position(smash, LMC_RA, inside_dec)
    assert status == "covered" and "LMC" in reason
    status, _ = reg.coverage_status_for_position(smash, LMC_RA, outside_dec)
    assert status == "unverified"  # outside the cone the boolean no longer applies
    status, reason = reg.coverage_status_for_position(des, LMC_RA, inside_dec)
    assert status == "not_covering" and "LMC" in reason
    # Just outside the cone, DES's covers_lmc=False no longer fires, and the
    # position is inside its stated dec range → unverified, not excluded.
    status, _ = reg.coverage_status_for_position(des, LMC_RA, outside_dec)
    assert status == "unverified"


def test_smc_cone_boundary_three_degrees():
    by_name = {row["catalog"]: row for row in reg.list_catalogs()}
    smash = by_name["smash_dr1"]["coverage"]
    status, reason = reg.coverage_status_for_position(smash, SMC_RA, SMC_DEC + 2.9)
    assert status == "covered" and "SMC" in reason
    status, _ = reg.coverage_status_for_position(smash, SMC_RA, SMC_DEC + 3.1)
    assert status == "unverified"


def test_galactic_plane_boundary_abs_b_ten():
    by_name = {row["catalog"]: row for row in reg.list_catalogs()}
    des, twomass = by_name["des_dr1"]["coverage"], by_name["twomass"]["coverage"]
    inside_dec = _dec_at_galactic_latitude(-9.8)
    outside_dec = _dec_at_galactic_latitude(-10.2)
    assert abs(reg.galactic_latitude_deg(GC_RA, inside_dec) + 9.8) < 0.01  # bisection sanity
    status, reason = reg.coverage_status_for_position(des, GC_RA, inside_dec)
    assert status == "not_covering" and "Galactic plane" in reason
    status, _ = reg.coverage_status_for_position(des, GC_RA, outside_dec)
    assert status == "unverified"  # off-plane, inside DES's dec range
    status, _ = reg.coverage_status_for_position(twomass, GC_RA, inside_dec)
    assert status == "covered"


def test_hemisphere_gate_boundary_at_dec_twenty():
    by_name = {row["catalog"]: row for row in reg.list_catalogs()}
    splus = by_name["splus_dr4"]["coverage"]  # south, no dec bounds encoded
    # |dec| <= 20 is the equatorial tolerance band: no hemisphere exclusion.
    status, _ = reg.coverage_status_for_position(splus, 150.0, 19.9)
    assert status == "unverified"
    status, reason = reg.coverage_status_for_position(splus, 150.0, 20.1)
    assert status == "not_covering" and "southern-hemisphere survey" in reason


# ── capability: band curation ─────────────────────────────────────────────────

def test_band_filter_keeps_only_catalogs_listing_the_band():
    out = _run(band="w1")
    assert {row["catalog"] for row in out["catalogs"]} == {
        "allwise", "unwise_dr1", "catwise2020", "ls_dr9", "ls_dr10",
    }
    assert "gaia_dr3" in out["excluded_by_band"]
    assert out["filters"]["band"] == "w1"
    assert out["total_registered"] == 17


def test_band_filter_normalizes_the_ks_k_alias():
    # 2MASS spells Ks as 'k'; VHS spells it 'ks' — both tokens match both.
    for token in ("ks", "Ks", "k"):
        out = _run(band=token)
        assert {row["catalog"] for row in out["catalogs"]} == {"twomass", "vhs_dr5"}, token


def test_band_and_position_filters_compose():
    out = _run(band="g", ra=LMC_RA, dec=LMC_DEC)
    kept = {row["catalog"] for row in out["catalogs"]}
    assert KNOWN_COVERING_LMC <= kept
    assert "allwise" in out["excluded_by_band"]
    # des_dr1 lists g but is coverage-excluded at the LMC.
    assert "des_dr1" in {row["catalog"] for row in out["excluded_by_coverage"]}


# ── scope='all' curation interplay (CX-19) ────────────────────────────────────

class _FakeSchemasClient:
    """Serves a fixed tap_schema.schemas frame — never touches the network."""

    def __init__(self, df):
        self.df = df

    def query(self, sql, fmt="pandas"):
        assert "tap_schema.schemas" in sql
        return DatalabResult.from_dataframe(self.df.copy(), {})


def test_scope_all_with_filters_curates_catalogs_but_never_schemas(tmp_path, monkeypatch):
    monkeypatch.setenv("DATALAB_TAP_SCHEMA_CACHE_DIR", str(tmp_path / "tap-schema"))
    schemas_df = pd.DataFrame({
        "schema_name": ["gaia_dr3", "buzzard_dr1"],
        "description": ["Gaia DR3", "Buzzard simulation"],
    })
    ctx = CallContext(services={"datalab_client": _FakeSchemasClient(schemas_df)})
    out = dl.ListCatalogs().run(
        dl.ListCatalogsInput(scope="all", ra=LMC_RA, dec=LMC_DEC, band="g"), ctx
    ).to_native()
    assert out["success"] is True and out["scope"] == "all"
    # DEFINED interplay: the live schema inventory is NEVER filtered — both
    # schemas survive even though 'buzzard_dr1' has no coverage/band metadata.
    assert {row["schema"] for row in out["schemas"]} == {"gaia_dr3", "buzzard_dr1"}
    assert out["schema_count"] == 2
    # ...while the curated rows ARE curated (g band + LMC coverage ranking).
    kept = {row["catalog"] for row in out["catalogs"]}
    assert KNOWN_COVERING_LMC <= kept
    assert "allwise" in out["excluded_by_band"]  # no g band
    assert "des_dr1" in {row["catalog"] for row in out["excluded_by_coverage"]}
    assert out["filters"] == {
        "band": "g",
        "position": {"ra": LMC_RA, "dec": LMC_DEC, "label": f"RA={LMC_RA:.5f}, Dec={LMC_DEC:.5f}"},
    }
    # The note spells the interplay out.
    assert "never filtered" in out["note"]


# ── back-compat + registered-subset honesty ───────────────────────────────────

def test_no_args_returns_the_full_curated_listing_unchanged():
    out = _run()
    assert out["success"] is True and out["scope"] == "registered"
    assert out["count"] == len(out["catalogs"]) == out["total_registered"] == 17
    assert [row["catalog"] for row in out["catalogs"]] == [
        row["catalog"] for row in reg.list_catalogs()
    ]
    # No curation applied → no filter/exclusion keys and no per-row rank labels.
    assert "filters" not in out
    assert "excluded_by_band" not in out and "excluded_by_coverage" not in out
    assert all("coverage_status" not in row for row in out["catalogs"])


def test_note_states_registered_subset_and_names_scope_all():
    out = _run()
    note = out["note"]
    assert "curated" in note.lower()
    assert "17 registered catalogs" in note
    assert "NOT the full Data Lab schema set" in note
    assert "scope='all'" in note


def test_lmc_coverage_summary_keeps_unverified_near_ir_catalogs_visible():
    # UI bench 2026-09-23 L01: the answer dropped VHS (unverified at the LMC).
    out = _run(ra=LMC_RA, dec=LMC_DEC)
    summary = out["coverage_summary"]
    assert "twomass" in summary["covered"]
    unverified = {row["catalog"]: row for row in summary["coverage_unverified"]}
    assert "vhs_dr5" in unverified and "VMC" in unverified["vhs_dr5"]["footprint"]
    near_ir = summary["by_wavelength_regime"]["near-ir"]
    assert "twomass" in near_ir and "vhs_dr5 (coverage unverified here)" in near_ir
    assert "never drop" in summary["instruction"]
    # DLB-100 F11: why each COVERED catalog covers the position.
    feet = {row["catalog"]: row for row in summary["covered_footprints"]}
    assert set(feet) == set(summary["covered"])
    assert all(row["footprint"] or row["coverage_reason"] for row in feet.values())
    assert "say WHY" in summary["instruction"]
    # unfiltered listings carry no summary
    assert "coverage_summary" not in _run()
