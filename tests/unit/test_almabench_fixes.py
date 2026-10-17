"""ALMAbench 2026-10-15 defects (tmp/almabench-2026-10-15/RESULTS.md).

1. Code refusal: code_recipe returned "no curated recipe" for every code
   question (the benchmark snapshots had no docs/recipes) and the models then
   refused to write any code. Now: same-library nearest recipes, a loud
   infrastructure failure when the recipe library is missing, and a standard-API
   fallback; the reference tables live in a tracked path.
2. HUDF overlay: the ALMA product had to cover the field, not just be in a MOUS
   with a pointing nearby.
3. Archive cones: a named extended source gets a cone sized from its extent.
4. Concept answers: the documentation leads but every part is answered.
Network is faked throughout.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List

import pandas as pd
import pytest

from capabilities import alma_tools
from capabilities.base import CallContext

REPO = Path(__file__).resolve().parents[2]


class _Tap:
    def __init__(self, answer):
        self.answer = answer
        self.queries: List[str] = []

    def search(self, q, maxrec=None):
        self.queries.append(q)
        df = self.answer(q)

        class _Res:
            quasar_tap_url = "https://almascience.nrao.edu/tap"
            query_status = "OK"

            def to_table(self_inner):
                return self_inner

            def to_pandas(self_inner):
                return df.copy()

        return _Res()


def _ctx(answer, **extra):
    tap = _Tap(answer)
    svc = type("_Svc", (), {})()
    svc.alminer_client = type("_Alminer", (), {"_get_tap_service": lambda self: tap})()
    services = {
        "search_service": svc,
        "alma_tap_provenance": {"query": None, "url": None},
        "resolve_target": lambda name: {"success": True, "ra_deg": 204.25383, "dec_deg": -29.86576},
        "set_last_run_result": lambda v: None,
        "get_last_run_result": lambda: None,
        "console_log": lambda *a, **k: None,
    }
    services.update(extra)
    return CallContext(services=services), tap


def _run(cap, ctx, **kwargs):
    return cap.run(cap.InputModel(**kwargs), ctx).to_native()


# ── 1. code recipes and reference tables ─────────────────────────────────


def test_reference_tables_are_tracked_and_load():
    from services import alma_reference

    path = alma_reference.REFERENCE_PATH
    assert path.is_file() and REPO / "services" in path.parents
    ignored = subprocess.run(["git", "check-ignore", "-q", str(path)], cwd=REPO).returncode
    assert ignored == 1, f"{path} is git-ignored, so deployments would not have it"
    table = alma_reference.alma_reference_table("bands")
    assert table["receiver_band_count"] == 10


def test_recipe_files_are_not_git_ignored():
    # The round-1 failure was an ignored path (it never shipped); new recipes may be uncommitted but never ignored.
    on_disk = sorted(p.relative_to(REPO).as_posix() for p in alma_tools.RECIPES_DIR.glob("*.py"))
    ignored = subprocess.run(["git", "check-ignore", *on_disk], cwd=REPO, capture_output=True, text=True).stdout.split()
    assert on_disk and ignored == []


def test_code_recipe_gives_the_nearest_same_library_recipe_when_the_task_has_none():
    out = _run(alma_tools.CodeRecipe(), CallContext(), task="query by object name", library="pyvo")
    assert out["success"] and out["nearest_recipe"] is True
    assert out["recipes"][0]["file"].startswith("docs/recipes/pyvo_")
    assert "pyvo.dal.TAPService(" in out["recipes"][0]["code"] and "adapted from the tested recipe" in out["note"]
    # An exact match is not marked nearest and gets no extra recipe.
    exact = _run(alma_tools.CodeRecipe(), CallContext(), task="cone search", library="astroquery")
    assert "nearest_recipe" not in exact and [r["file"] for r in exact["recipes"]] == ["docs/recipes/astroquery_cone_search.py"]
    # ALMiner maps every task to its one recipe, listed once.
    alm = _run(alma_tools.CodeRecipe(), CallContext(), task="ADQL via TAP conesearch", library="alminer")
    assert [r["file"] for r in alm["recipes"]] == ["docs/recipes/alminer_alminer.py"]


def test_missing_recipe_library_is_an_infrastructure_failure_with_a_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(alma_tools, "RECIPES_DIR", tmp_path / "recipes")
    out = _run(alma_tools.CodeRecipe(), CallContext(), task="query by object name", library="astroquery")
    assert out["success"] is False and out["status"] == "infrastructure_failure"
    fb = out["fallback"]
    assert "Alma.query_object(name, public=...)" in fb["standard_calls"] and "target_name" in fb["obscore_columns"]
    assert "SHOULD still give working code" in fb["instruction"]


def test_coverage_gap_still_carries_the_standard_api_fallback(monkeypatch, tmp_path):
    (tmp_path / "astroquery_object_search.py").write_text('"""x\nlibrary: a\n"""\nprint(1)\n', encoding="utf-8")
    monkeypatch.setattr(alma_tools, "RECIPES_DIR", tmp_path)
    out = _run(alma_tools.CodeRecipe(), CallContext(), task="ADQL via TAP", library="pyvo")
    assert out["status"] == "coverage_gap" and out["available"] == ["astroquery_object_search.py"]
    assert out["fallback"]["standard_calls"][0].startswith("pyvo.dal.TAPService(")


def test_standard_api_fallback_names_only_real_calls_and_columns():
    from astroquery.alma import Alma

    for call in alma_tools._STANDARD_API_CALLS["astroquery"]:
        assert hasattr(Alma, call.split("(")[0].split(".")[1]), call
    alminer = pytest.importorskip("alminer")
    for call in alma_tools._STANDARD_API_CALLS["alminer"]:
        assert hasattr(alminer, call.split("(")[0].split(".")[1]), call
    from services.alma_science_queries import OBSCORE_BASE_COLUMNS

    known = set(OBSCORE_BASE_COLUMNS) | {"s_ra", "s_dec", "s_fov", "frequency", "pi_name", "scientific_category", "antenna_arrays",
                    "science_observation", "t_exptime", "obs_release_date", "data_rights"}
    assert set(alma_tools.OBSCORE_STANDARD_COLUMNS) <= known


def test_code_directive_never_leaves_a_code_request_without_code():
    from core.oneshot_routing import detect_oneshot_intent

    intent = detect_oneshot_intent("How do I use Astroquery to find all ALMA observations of the galaxy M83?")
    assert intent and intent["tool"] == "code_recipe"
    assert "Never answer a code request without code" in intent["directive"]
    assert "`fallback`" in intent["directive"] and "Do NOT invent API calls" in intent["directive"]


# ── 2. overlay: the ALMA image must cover the field ──────────────────────


class _DL:
    def __init__(self, listings):
        self.listings = listings
        self.calls: List[str] = []

    def list_files(self, mous_uid, pattern=None):
        self.calls.append(mous_uid)
        return {"success": True, "files": self.listings.get(mous_uid, [])}


def _f(uid, target, kind="cont.I.pbcor", mb=5.0):
    stem = uid.replace("uid://", "uid___").replace("/", "_")
    return {"filename": f"member.{stem}.ari_l.{target}_sci.spw0_1_2_3_343498MHz.12m.{kind}.fits", "size_mb": mb,
            "access_url": f"https://almascience.org/{target}.fits"}


def test_overlay_takes_the_covering_target_not_the_first_file_of_a_many_target_mous():
    # 2016.1.01079.S: one MOUS, 64 single pointings; only scuba2-084 covers the centre.
    many = "uid://A001/X8a6/X35"
    rows = [{"member_ous_uid": many, "proposal_id": "2016.1.01079.S", "target_name": "scuba2-084", "s_ra": 53.1305,
             "s_dec": -27.7736, "s_fov": 0.0047, "spatial_resolution": 0.2, "band_list": "7"}]
    dl = _DL({many: [_f(many, "scuba2-035"), _f(many, "scuba2-084"), _f(many, "scuba2-101")]})
    ctx, tap = _ctx(lambda q: pd.DataFrame(rows), datalink_client=dl)
    out = alma_tools.ArchiveOverlay._find_alma_contour(ctx, 53.1305, -27.7736, 3.0, 150.0, "field")
    assert out["product"]["filename"].endswith(".scuba2-084_sci.spw0_1_2_3_343498MHz.12m.cont.I.pbcor.fits")
    assert out["product"]["covers_field_centre"] is True and out["product"]["target_name"] == "scuba2-084"
    q = tap.queries[0]
    assert "s_region) = 1" in q and "CONTAINS(POINT('ICRS', 53.13050000, -27.77360000), s_region)" in q
    assert "ORDER BY" not in q and "science_observation = 'T'" in q


def test_overlay_prefers_the_named_field_mosaic_and_skips_mous_without_products():
    rows = [
        {"member_ous_uid": "uid://A001/X2fb/X21d", "proposal_id": "2015.1.00543.S", "target_name": "GOODS-S", "s_ra": 53.16,
         "s_dec": -27.79, "s_fov": 0.121, "spatial_resolution": 0.22, "band_list": "6"},
        {"member_ous_uid": "uid://A002/X5a9a13/X7e0", "proposal_id": "2012.1.00173.S", "target_name": "HUDF", "s_ra": 53.16,
         "s_dec": -27.79, "s_fov": 0.051, "spatial_resolution": 0.25, "band_list": "6"},
        {"member_ous_uid": "uid://A001/X87c/X21e", "proposal_id": "2016.1.00324.L", "target_name": "UDF_mosaic_1mm",
         "s_ra": 53.16, "s_dec": -27.78, "s_fov": 0.0402, "spatial_resolution": 0.68, "band_list": "6"},
    ]
    dl = _DL({"uid://A002/X5a9a13/X7e0": [], "uid://A001/X87c/X21e": [_f("uid://A001/X87c/X21e", "UDF_mosaic_1mm")]})
    ctx, _ = _ctx(lambda q: pd.DataFrame(rows), datalink_client=dl)
    out = alma_tools.ArchiveOverlay._find_alma_contour(ctx, 53.1625, -27.7914, 3.0, 150.0, "HUDF")
    assert out["product"]["proposal_id"] == "2016.1.00324.L"
    assert dl.calls == ["uid://A002/X5a9a13/X7e0", "uid://A001/X87c/X21e"], "name matches (HUDF, UDF) come before GOODS-S"


def test_overlay_point_cone_fallback_rejects_pointings_that_miss_the_centre():
    def answer(q):
        if "s_region" in q:
            raise RuntimeError("footprint predicate not supported")
        # scuba2-035 is ~6 arcmin from the HUDF with a 17-arcsec field.
        return pd.DataFrame([{"member_ous_uid": "uid://A001/X8a6/X35", "proposal_id": "2016.1.01079.S",
                              "target_name": "scuba2-035", "s_ra": 53.063875, "s_dec": -27.843611, "s_fov": 0.0047,
                              "spatial_resolution": 0.2, "band_list": "7"}])

    dl = _DL({"uid://A001/X8a6/X35": [_f("uid://A001/X8a6/X35", "scuba2-035")]})
    ctx, tap = _ctx(answer, datalink_client=dl)
    out = alma_tools.ArchiveOverlay._find_alma_contour(ctx, 53.1625, -27.7914, 3.0, 150.0, "HUDF")
    assert out["product"] is None and out["unknown"] is False and dl.calls == []
    assert any("none covers its centre" in n for n in out["notes"]) and len(tap.queries) == 2


def test_file_target_parses_alma_product_names():
    assert alma_tools.ArchiveOverlay._file_target("member.uid___A001_X1_X1.HUDF_sci.spw25.cont.I.pbcor.fits") == "hudf"
    assert alma_tools.ArchiveOverlay._file_target(
        "member.uid___A001_X2fe_Xb2a.ari_l.HUDF-JVLA-ALMA_sci.spw0_1_2_3_262521MHz.12m.cont.I.pbcor.fits") == "hudf-jvla-alma"
    assert alma_tools.ArchiveOverlay._file_target("member.uid___A001_X1_X1.J0336-2644_chk.spw11.mfs.I.pbcor.fits") is None


# ── 3. archive cones sized from the source extent ────────────────────────


def test_extent_radius_rules(monkeypatch):
    from services import target_extent

    sizes = {"M83": 13.8, "HH 212": 0.01, "M31": 199.53}
    monkeypatch.setattr(target_extent, "major_axis_lookup", lambda name: ("ok", sizes.get(name)))
    assert target_extent.extent_radius_arcsec("M83", 60.0)["radius_arcsec"] == pytest.approx(517.5)
    assert target_extent.extent_radius_arcsec("HH 212", 60.0)["radius_arcsec"] == 60.0
    assert target_extent.extent_radius_arcsec("TW Hya", 60.0)["radius_arcsec"] == 60.0
    big = target_extent.extent_radius_arcsec("M31", 60.0)
    assert big["radius_arcsec"] == target_extent.EXTENT_CAP_ARCSEC and "capped" in big["basis"]


def test_lookup_switch_and_failures_are_not_cached(monkeypatch):
    from services import target_extent

    monkeypatch.setenv("QUASAR_TARGET_EXTENT_LOOKUP", "0")
    assert target_extent.catalogued_major_axis_arcmin("M83") is None
    monkeypatch.setenv("QUASAR_TARGET_EXTENT_LOOKUP", "1")
    import astroquery.simbad as simbad_mod

    class _Boom:
        def add_votable_fields(self, *a):
            pass

        def query_object(self, name):
            raise ConnectionError("SIMBAD down")

    monkeypatch.setattr(simbad_mod, "Simbad", _Boom)
    target_extent._MAJOR_AXIS_CACHE.pop("NGC 9999", None)
    assert target_extent.major_axis_lookup("NGC 9999") == ("unavailable", None)
    assert "NGC 9999" not in target_extent._MAJOR_AXIS_CACHE
    basis = target_extent.extent_radius_arcsec("NGC 9999", 60.0)["basis"]
    assert "did not complete" in basis and "no catalogued size" not in basis  # guard CX-09


def test_source_summary_sizes_the_cone_for_an_extended_named_target(monkeypatch):
    from services import target_extent

    monkeypatch.setattr(target_extent, "major_axis_lookup", lambda name: ("ok", 13.8 if name == "M83" else None))
    row = {"member_ous_uid": "uid://A001/X1/X1", "proposal_id": "2019.1.00001.S", "target_name": "M83", "band_list": "6",
           "spatial_resolution": 0.5, "obs_release_date": "2021-01-01", "data_rights": "Public", "science_observation": "T",
           "bandwidth": 1.875e9, "frequency": 230.0, "frequency_support": "[229.00..231.00GHz,976.56kHz,XX YY]",
           "cont_sensitivity_bandwidth": 0.05, "t_exptime": 600.0}
    ctx, tap = _ctx(lambda q: pd.DataFrame([row]))
    out = _run(alma_tools.AlmaSourceSummary(), ctx, target="M83", bands=[6])
    assert out["radius_arcsec"] == pytest.approx(517.5, abs=0.1) and "SIMBAD size" in out["radius_basis"]
    assert f"{519.5 / 3600:.8f} + s_fov / 2.0" in tap.queries[0], "extent cone with field overlap at the server (CX-04)"
    # An explicit radius wins, and a compact source keeps 60 arcsec.
    ctx, tap = _ctx(lambda q: pd.DataFrame([row]))
    out = _run(alma_tools.AlmaSourceSummary(), ctx, target="M83", radius_arcsec=30)
    assert out["radius_arcsec"] == 30 and "radius_basis" not in out
    ctx, tap = _ctx(lambda q: pd.DataFrame([row]))
    out = _run(alma_tools.AlmaSourceSummary(), ctx, target="HH 212")
    assert out["radius_arcsec"] == 60 and "default 60 arcsec" in out["radius_basis"]


def test_search_by_target_cone_uses_the_extent(monkeypatch):
    from integrations import alminer_client as ac
    from services import target_extent

    monkeypatch.setattr(target_extent, "major_axis_lookup", lambda name: ("ok", 13.8))
    monkeypatch.setattr(ac, "_resolve_simbad_cached", lambda name: (204.25383, -29.86576))
    seen = {}

    def fake_parallel(self, ra, dec, radius=0.05, **kw):
        seen["radius"] = radius
        df = pd.DataFrame([{"x": 1}])
        df.attrs["quasar_cone"] = {"radius_deg": radius}
        return df

    monkeypatch.setattr(ac.ALminerClient, "_parallel_search", fake_parallel)
    client = ac.ALminerClient.__new__(ac.ALminerClient)
    df = client.search_by_target("M83")
    assert seen["radius"] == pytest.approx(517.5 / 3600)
    assert "SIMBAD size" in df.attrs["radius_basis"] and df.attrs["quasar_cone"]["radius_basis"] == df.attrs["radius_basis"]


# ── 4. concept answers ───────────────────────────────────────────────────


def test_knowledge_directive_answers_every_part_without_inventing_archive_names():
    from core import rag_routing as rr

    d = rr.KNOWLEDGE_DIRECTIVE
    assert "answer EVERY part of the question" in d and "standard radio-astronomy and interferometry knowledge" in d
    assert "Never invent archive-specific names" in d and "live archive fact" in d
    assert "Answer it from the DOCUMENTATION CONTEXT above" not in d


def test_census_directive_scopes_no_memory_to_archive_facts():
    from core.oneshot_routing import census_directive

    route = {"query_type": "redshifted_line_projects", "redshift_min": 1.0, "redshift_max": 2.0, "rest_species": "CO"}
    for text in (census_directive(route), census_directive(None)):
        assert "Do NOT answer from memory" in text
        assert "archive counts, lists and identifiers only" in text and "answer that part fully" in text


# ── guard task-3ed2659-27967 regressions (CX-01..CX-10) ──────────────────


def _row(uid, target, fov_arcmin=None, res=0.5, ra=53.1625, dec=-27.7914, pid="2099.1.00001.S"):
    return {"member_ous_uid": uid, "proposal_id": pid, "target_name": target, "s_ra": ra, "s_dec": dec,
            "s_fov": None if fov_arcmin is None else fov_arcmin / 60.0, "spatial_resolution": res, "band_list": "6"}


def _contour(rows, listings, label="field", answer=None):
    dl = _DL(listings)
    ctx, tap = _ctx(answer or (lambda q: pd.DataFrame(rows)), datalink_client=dl)
    return alma_tools.ArchiveOverlay._find_alma_contour(ctx, 53.1625, -27.7914, 3.0, 150.0, label), dl, tap


def test_cx02_a_listing_without_the_covering_source_is_never_accepted():
    uid = "uid://A001/X8a6/X35"
    out, _, _ = _contour([_row(uid, "scuba2-084", 0.3)], {uid: [_f(uid, "scuba2-035")]})
    assert out["product"] is None


def test_cx02_pointings_without_a_source_name_are_unknown_not_matched():
    uid = "uid://A001/X1/X1"
    out, dl, _ = _contour([_row(uid, "", 3.0)], {uid: [_f(uid, "anything")]})
    assert out["product"] is None and out["unknown"] is True and dl.calls == []


def test_cx03_the_best_ranked_source_in_a_mous_wins_and_its_metadata_follows_the_file():
    uid = "uid://A001/X1/X1"
    rows = [_row(uid, "other", 0.3, res=0.1), _row(uid, "HUDF", 3.0, res=0.5)]
    out, _, _ = _contour(rows, {uid: [_f(uid, "other", mb=1.0), _f(uid, "HUDF", mb=40.0)]}, label="HUDF")
    p = out["product"]
    assert ".HUDF_sci." in p["filename"] and p["target_name"] == "HUDF" and p["field_of_view_arcmin"] == 3.0


def test_cx04_unknown_field_of_view_ranks_below_a_known_adequate_one():
    a, b = "uid://A001/X1/X1", "uid://A001/X2/X2"
    rows = [_row(a, "srcA", None, res=0.1), _row(b, "srcB", 3.0, res=0.9)]
    out, dl, _ = _contour(rows, {a: [_f(a, "srcA")], b: [_f(b, "srcB")]})
    assert out["product"]["target_name"] == "srcB" and dl.calls[0] == b


def test_cx05_dotted_source_names_are_parsed_whole():
    uid = "uid://A001/X1/X1"
    name = "member.uid___A001_X1_X1.G31.41+0.31_sci.spw25.cont.I.pbcor.fits"
    assert alma_tools.ArchiveOverlay._file_target(name) == "g31.41+0.31"
    files = [{"filename": "member.uid___A001_X1_X1.G10.62-0.38_sci.spw25.cont.I.pbcor.fits", "size_mb": 1, "access_url": "u1"},
             {"filename": name, "size_mb": 5, "access_url": "u2"}]
    out, _, _ = _contour([_row(uid, "G31.41+0.31", 0.5)], {uid: files})
    assert out["product"]["access_url"] == "u2"


def test_cx06_sign_distinct_identifiers_do_not_collide():
    uid = "uid://A001/X1/X1"
    files = [{"filename": "member.uid___A001_X1_X1.J1234-0100_sci.spw25.cont.I.pbcor.fits", "size_mb": 1, "access_url": "neg"},
             {"filename": "member.uid___A001_X1_X1.J1234+0100_sci.spw25.cont.I.pbcor.fits", "size_mb": 9, "access_url": "pos"}]
    out, _, _ = _contour([_row(uid, "J1234+0100", 0.5)], {uid: files})
    assert out["product"]["access_url"] == "pos"


def test_cx07_cone_fallback_with_unknown_fov_is_unknown_not_absent():
    def answer(q):
        if "s_region" in q:
            raise RuntimeError("footprint predicate not supported")
        return pd.DataFrame([_row("uid://A001/X1/X1", "HUDF", None)])

    out, _, _ = _contour([], {}, answer=answer)
    assert out["product"] is None and out["unknown"] is True and any("UNKNOWN" in n for n in out["notes"])


def test_cx01_cx11_the_simbad_worker_is_bounded_and_carries_a_cancellable_deadline(monkeypatch):
    import time

    from services import target_extent, tool_budgets

    monkeypatch.setenv("QUASAR_TARGET_EXTENT_LOOKUP", "1")
    monkeypatch.setattr(target_extent, "LOOKUP_WALL_S", 1.2)
    seen = {"calls": 0}

    def slow(name):
        seen["calls"] += 1
        seen["deadline"] = tool_budgets.current_deadline()
        time.sleep(3)
        return 13.8

    monkeypatch.setattr(target_extent, "_query_major_axis", slow)
    target_extent._MAJOR_AXIS_CACHE.pop("Slow Galaxy", None)
    t0 = time.perf_counter()
    assert target_extent.major_axis_lookup("Slow Galaxy") == ("unavailable", None)
    assert time.perf_counter() - t0 < 2.5 and seen["calls"] == 1, "the worker really ran and was abandoned"
    assert seen["deadline"] is not None and seen["deadline"].cancelled(), "the abandoned worker's deadline is cancelled"
    assert "Slow Galaxy" not in target_extent._MAJOR_AXIS_CACHE
    # Inside a tool the worker gets a CHILD of the tool deadline.
    parent = tool_budgets.Deadline(60.0, label="tool")
    tool_budgets.adopt_deadline(parent)
    try:
        monkeypatch.setattr(target_extent, "_query_major_axis", lambda name: seen.update(child=tool_budgets.current_deadline()) or 13.8)
        target_extent._MAJOR_AXIS_CACHE.pop("Fast Galaxy", None)
        assert target_extent.major_axis_lookup("Fast Galaxy") == ("ok", 13.8)
        assert seen["child"] is not None and seen["child"] is not parent and seen["child"].token is parent.token
        # And never into the reserve the archive search needs.
        monkeypatch.setattr(tool_budgets, "remaining_seconds", lambda: 15.0)
        target_extent._MAJOR_AXIS_CACHE.pop("Other Galaxy", None)
        assert target_extent.major_axis_lookup("Other Galaxy") == ("unavailable", None)
    finally:
        tool_budgets.adopt_deadline(None)


def test_cx08_search_by_target_returns_the_radius_and_its_basis():
    from capabilities.alma import SearchByTarget

    df = pd.DataFrame([{"member_ous_uid": "uid://A001/X1/X1", "proposal_id": "2019.1.00001.S", "target_name": "M83",
                        "band_list": "6", "access_url": "https://almascience.org/datalink?ID=x", "s_ra": 204.25, "s_dec": -29.87}])
    df.attrs["quasar_cone"] = {"ra": 204.25383, "dec": -29.86576, "radius_deg": 517.5 / 3600, "adql": "SELECT 1",
                               "radius_basis": "518 arcsec cone = 1.25 x the semi-major axis of the SIMBAD size"}
    df.attrs["radius_basis"] = df.attrs["quasar_cone"]["radius_basis"]
    svc = type("_S", (), {"search_by_target": lambda self, *a, **k: df.copy()})()
    ctx, _ = _ctx(lambda q: pd.DataFrame(), search_service=svc, set_last_search_results=lambda v: None)
    out = _run(SearchByTarget(), ctx, target_name="M83")
    assert out["success"], out
    assert out["search_radius_arcsec"] == 517.5 and "SIMBAD size" in out["radius_basis"]


def test_cx10_a_task_without_keywords_is_flagged_as_nearest():
    out = _run(alma_tools.CodeRecipe(), CallContext(), task="aggregate exposure by band", library="astroquery")
    assert out["nearest_recipe"] is True and "verbatim first" in out["note"]


def _target_frame(name, radius_arcsec, empty=False):
    df = pd.DataFrame() if empty else pd.DataFrame([{"member_ous_uid": f"uid://A001/X1/{name}", "proposal_id": "2019.1.00001.S",
                                                     "target_name": name, "band_list": "6", "s_ra": 1.0, "s_dec": 1.0,
                                                     "access_url": "https://almascience.org/datalink?ID=x"}])
    df.attrs["quasar_cone"] = {"ra": 1.0, "dec": 1.0, "radius_deg": radius_arcsec / 3600, "adql": "SELECT 1"}
    df.attrs["radius_basis"] = f"basis for {name}"
    return df


def _sbt_ctx(frames):
    svc = type("_S", (), {"search_by_target": lambda self, name, *a, **k: frames[name].copy()})()
    ctx, _ = _ctx(lambda q: pd.DataFrame(), search_service=svc, set_last_search_results=lambda v: None)
    return ctx


def test_cx08_empty_and_multi_target_results_report_each_radius():
    from capabilities.alma import SearchByTarget

    out = _run(SearchByTarget(), _sbt_ctx({"M83": _target_frame("M83", 517.5, empty=True)}), target_name="M83")
    assert out["total_results"] == 0 and out["search_radius_arcsec"] == 517.5 and out["radius_basis"] == "basis for M83"
    frames = {"M83": _target_frame("M83", 517.5), "HH212": _target_frame("HH212", 60.0)}
    out = _run(SearchByTarget(), _sbt_ctx(frames), target_name="M83 and HH212")
    assert out["success"], out
    assert out["radius_by_target"] == [
        {"target": "M83", "search_radius_arcsec": 517.5, "radius_basis": "basis for M83"},
        {"target": "HH212", "search_radius_arcsec": 60.0, "radius_basis": "basis for HH212"}]


def test_cx12_filename_spelling_without_spaces_still_matches():
    uid = "uid://A001/X1/X1"
    files = [{"filename": "member.uid___A001_X1_X1.NGC1068_sci.spw25.cont.I.pbcor.fits", "size_mb": 5, "access_url": "ngc"}]
    out, _, _ = _contour([_row(uid, "NGC 1068", 0.5)], {uid: files})
    assert out["product"]["access_url"] == "ngc"
    assert alma_tools._source_key("NGC 1068") == alma_tools._source_key("NGC_1068") == "ngc1068"
    assert alma_tools._source_key("J1234+0100") != alma_tools._source_key("J1234-0100")


def test_cx13_a_better_source_in_another_mous_beats_a_worse_one_in_a_listed_mous():
    a, b = "uid://A001/X1/X1", "uid://A001/X2/X2"
    rows = [_row(a, "HUDF", 3.0, res=0.2), _row(b, "HUDF", 3.0, res=0.4), _row(a, "other", 0.3, res=0.1)]
    out, dl, _ = _contour(rows, {a: [_f(a, "other")], b: [_f(b, "HUDF")]}, label="HUDF")
    assert out["product"]["member_ous_uid"] == b and out["product"]["target_name"] == "HUDF"
    assert dl.calls == [a, b], "each MOUS is listed once"


# ── round 3 (2026-10-16): generic fixes from the alma2 item-level analysis ─────


def test_r3_band_status_explains_bands_without_public_data():
    from services.alma_reference import alma_reference_table

    ref = alma_reference_table("bands")
    out = alma_tools.AlmaPublicBandStatus._bands_without_public_data(ref, {1, 3, 4, 5, 6, 7, 8, 9, 10})
    assert [o["band"] for o in out] == [2]
    assert out[0]["offered_in_tabulated_cycles"] == [13] and "first offered in Cycle 13" in out[0]["explanation"]


def test_r3_redshift_census_maps_lines_to_bands_and_explains_the_method():
    from services.alma_server_side import redshift_windows

    m = {r["transition"]: r for r in alma_tools._redshift_band_mapping(redshift_windows("CO", 1.0, 2.0), 1.0, 2.0)}
    b3 = next(b for b in m["CO(2-1)"]["bands"] if b["band"] == 3)
    assert b3["z_min"] == 1.0 and abs(b3["z_max"] - 1.744) < 0.01  # CO 2-1 in Band 3 up to z ~ 1.74
    assert any(b["band"] == 4 for b in m["CO(3-2)"]["bands"])
    note = alma_tools._REDSHIFT_METHOD_NOTE
    assert "no redshift column" in note and "spectroscopic" in note and "frequency_support" in note


def test_r3_overlay_checks_report_levels_beam_registration_and_headers():
    result = {"contours": {"rms": 2.1e-5, "sigmas": [3, 5, 8], "negative_contours": True},
              "contour_beam": {"bmaj_arcsec": 0.6, "bmin_arcsec": 0.5, "bpa_deg": 80.0}, "base_pixel_arcsec": 0.03,
              "contour_product": {"filename": "member.uid___A001_X1_X1.HUDF_sci.spw25.cont.I.pbcor.fits"}}
    checks = " ".join(alma_tools.ArchiveOverlay._overlay_checks(result, "JWST"))
    assert "3, 5, 8 x the ALMA rms" in checks and "-3 rms" in checks and "primary-beam corrected" in checks
    assert "0.6 x 0.5 arcsec" in checks and "Gaia" in checks and "Stokes" in checks


def test_r3_source_summary_lists_every_mous():
    rows = [{"member_ous_uid": f"uid://A001/X1/X{i}", "proposal_id": "2019.1.00001.S", "target_name": "M83", "band_list": "6",
             "spatial_resolution": 0.5, "obs_release_date": "2021-01-01", "data_rights": "Public", "science_observation": "T",
             "bandwidth": 1.875e9, "frequency": 230.0, "frequency_support": "[229.00..231.00GHz,976.56kHz,XX YY]",
             "cont_sensitivity_bandwidth": 0.05, "t_exptime": 600.0} for i in range(44)]
    ctx, _ = _ctx(lambda q: pd.DataFrame(rows))
    out = _run(alma_tools.AlmaSourceSummary(), ctx, target="M83", bands=[6])
    assert out["results_returned"] == 40 and len(out["mous_list"]) == 44
    assert out["mous_list"][0].startswith("uid://A001/X1/X") and "list every MOUS" in out["note"]


def test_r3_perseus_catalogue_is_the_cited_vandam_sample():
    from services import cross_archive_matcher as cam

    rows, citation = cam.perseus_protostars()
    assert len(rows) == 97 and "Tobin et al. 2016" in citation
    name, cat = cam.normalize_source_catalog("perseus_protostars")
    assert name == "perseus_protostars" and len(cat) == 97
    tracked = subprocess.run(["git", "check-ignore", "-q", str(cam.VANDAM_PATH)], cwd=REPO).returncode
    assert tracked == 1


def test_r3_coverage_matching_counts_mosaics_for_every_source_they_contain():
    from services import cross_archive_matcher as cam

    sources = [{"source_name": "a", "ra": 52.0, "dec": 31.0}, {"source_name": "b", "ra": 52.01, "dec": 31.0},
               {"source_name": "c", "ra": 53.0, "dec": 31.0}]
    obs = pd.DataFrame([{"s_ra": 52.005, "s_dec": 31.0, "s_fov": 0.02, "proposal_id": "P1", "member_ous_uid": "m1"},
                        {"s_ra": 53.0, "s_dec": 31.0005, "s_fov": None, "proposal_id": "P2", "member_ous_uid": "m2"}])
    hits = cam.attach_covering_sources(obs, sources, radius_arcsec=5.0)
    assert sorted(hits["source_name"]) == ["a", "b", "c"]  # one 72-arcsec mosaic covers a and b; c is 1.8" off m2
    assert cam.catalog_box(sources) is not None and cam.catalog_box([{"ra": 1, "dec": 0}, {"ra": 300, "dec": 0}]) is None
    q = cam.alma_box_adql(cam.catalog_box(sources))
    assert q.startswith("SELECT DISTINCT member_ous_uid") and "s_fov" in q and "science_observation = 'T'" in q


def test_r3_archive_link_uses_the_asa_filter_ids(monkeypatch):
    import capabilities.alma_science_helpers as helpers

    monkeypatch.setattr(helpers, "head_verify", lambda url, timeout=6.0: (True, "HTTP 200"))
    ctx, _ = _ctx(lambda q: pd.DataFrame())
    out = _run(alma_tools.AlmaArchiveLink(), ctx, target="M83", band=[6, 7], public_only=True)
    url = out["asa_url"]
    assert "sourceNameResolver=M83" in url and "band=6,7" in url and "publicData=true" in url
    assert "bandList" not in url and "resultView" not in url
    out = _run(alma_tools.AlmaArchiveLink(), ctx, ra=204.2538, dec=-29.8658, radius_arcmin=8)
    assert out["asa_parameters"]["raDec"] == "204.253800 -29.865800, 8"  # trailing radius is in ARCMIN


def test_r4_archive_link_release_cut_name_match_link_and_notes(monkeypatch):
    import capabilities.alma_science_helpers as helpers

    monkeypatch.setattr(helpers, "head_verify", lambda url, timeout=6.0: (True, "HTTP 200"))
    ctx, _ = _ctx(lambda q: pd.DataFrame())
    out = _run(alma_tools.AlmaArchiveLink(), ctx, target="NGC 253", band=[3, 7], released_before="2025-01-01")
    assert out["asa_url"].endswith("?sourceNameResolver=NGC+253&band=3,7&observationsReleaseDate=%3C2025-01-01")
    assert out["asa_name_match_url"].endswith("?sourceName=NGC+253&band=3,7&observationsReleaseDate=%3C2025-01-01")
    assert out["asa_name_match_url"] in out["links"]
    notes = out["url_notes"]
    assert "%3C" in notes["url_encoding"] and "comma" in notes["url_encoding"]
    assert "position" in notes["name_vs_resolver"] and "PI entered" in notes["name_vs_resolver"]
    assert "re-runs the query live" in notes["sharing"] and "redirects" in notes["url_structure"]
    assert "obs_release_date < '2025-01-01'" in out["tap_adql"]
    bad = _run(alma_tools.AlmaArchiveLink(), ctx, target="NGC 253", released_before="soon")
    assert bad["success"] is False and bad["status"] == "invalid_input"


def test_r3_source_summary_date_cut_in_adql_and_point_cone_for_sized_radii(monkeypatch):
    from services import target_extent

    monkeypatch.setattr(target_extent, "major_axis_lookup", lambda name: ("ok", 13.8 if name == "M83" else None))
    row = {"member_ous_uid": "uid://A001/X1/X1", "proposal_id": "2019.1.00001.S", "target_name": "M83", "band_list": "6",
           "spatial_resolution": 0.5, "obs_release_date": "2021-01-01", "data_rights": "Public", "science_observation": "T",
           "bandwidth": 1.875e9, "frequency": 230.0, "frequency_support": "[229.00..231.00GHz,976.56kHz,XX YY]",
           "cont_sensitivity_bandwidth": 0.05, "t_exptime": 600.0}
    ctx, tap = _ctx(lambda q: pd.DataFrame([row]))
    out = _run(alma_tools.AlmaSourceSummary(), ctx, target="M83", bands=[6], released_before="2026-01-01")
    q = tap.queries[0]
    assert "obs_release_date < '2026-01-01'" in q and "INTERSECTS" not in q, "sized cone: pointing centres + date cut"
    assert "CONTAINS(POINT('ICRS', 204.25383000, -29.86576000), s_region) = 1" in q, "footprints containing the source (CX-04)"
    assert out["filters_in_adql"]["released_before"] == "2026-01-01" and out["cone_mode"].startswith("fields overlapping the cone")
    ctx, tap = _ctx(lambda q: pd.DataFrame([row]))
    _run(alma_tools.AlmaSourceSummary(), ctx, target="M83", radius_arcsec=30)
    assert "INTERSECTS" in tap.queries[0], "an explicit radius keeps the footprint cone"
    with pytest.raises(ValueError):
        from services.alma_science_queries import alma_cone_filters_where
        alma_cone_filters_where(1.0, 1.0, 0.01, released_before="soon")



# ── guard task-3ed2659-40703 (round 3) regressions ───────────────────────


def _xmatch(sources, answer):
    from capabilities.alma import _match_cross_archive_sources_impl

    class _Res:
        def __init__(self, df, status):
            self.df, self.query_status = df, status

        def to_table(self):
            return self

        def to_pandas(self):
            return self.df.copy()

    class _Tap2:
        def __init__(self):
            self.queries = []

        def search(self, q, maxrec=None):
            self.queries.append(q)
            df, status = answer(q)
            return _Res(df, status)

    tap = _Tap2()
    svc = type("_Svc", (), {})()
    svc.alminer_client = type("_A", (), {"_get_tap_service": lambda self: tap})()
    state = {}
    ctx = CallContext(services={"search_service": svc, "alma_tap_provenance": {"query": None, "url": None},
                                "set_last_search_results": lambda v: None, "set_last_run_result": lambda v: state.update(card=v),
                                "get_last_run_result": lambda: state.get("card"), "console_log": lambda *a, **k: None})
    out = _match_cross_archive_sources_impl(ctx, catalog_name="inline", sources=sources, archives=["ALMA"],
                                            radius_arcsec=5.0, max_sources=100, alma_mode="point_only")
    return out, tap, state


def test_cx01_box_margin_keeps_a_covering_mosaic_centred_outside_the_catalogue():
    from services import cross_archive_matcher as cam

    sources = [{"source_name": f"s{i}", "ra": 52.0 + i * 0.001, "dec": 31.0} for i in range(9)]
    box = cam.catalog_box(sources)
    assert box[1] >= 52.08, "a 0.2-deg mosaic centred at RA 52.08 must be inside the query box"
    mosaic = pd.DataFrame([{"s_ra": 52.08, "s_dec": 31.0, "s_fov": 0.2, "proposal_id": "P", "member_ous_uid": "m"}])
    assert len(cam.attach_covering_sources(mosaic, sources, radius_arcsec=5.0)) == 9


def test_cx02_cx05_box_overflow_is_partial_and_labels_follow_the_box_mode():
    sources = [{"source_name": f"s{i}", "ra": 52.0 + i * 0.01, "dec": 31.0} for i in range(9)]
    rows = pd.DataFrame([{"s_ra": 52.0, "s_dec": 31.0, "s_fov": 0.01, "proposal_id": "P", "member_ous_uid": "m",
                          "target_name": "t"}])
    out, tap, state = _xmatch(sources, lambda q: (rows, "OVERFLOW"))
    assert len(tap.queries) == 1 and tap.queries[0].startswith("SELECT DISTINCT member_ous_uid")
    assert out["footprint_mode"] == "box_fov" and out["partial"] is True
    assert any("row cap" in e and "lower bound" in e for e in out["archive_errors"])
    assert "field of view" in state["card"]["filter_label"]
    out, _, _ = _xmatch(sources, lambda q: (rows, "OK"))
    assert out["partial"] is False and out["archive_errors"] == []


def test_cx01_reopen_no_rabox_near_the_pole():
    from services import cross_archive_matcher as cam

    sources = [{"source_name": f"s{i}", "ra": 52.0 + i * 0.001, "dec": -89.0} for i in range(9)]
    assert cam.catalog_box(sources) is None, "near the pole the box cannot hold the margin: per-source cones instead"
    mid = [{"source_name": f"s{i}", "ra": 52.0 + i * 0.001, "dec": -60.0} for i in range(9)]
    box = cam.catalog_box(mid)
    assert box[1] - 52.008 >= 0.25 / 0.5 - 1e-6, "the RA margin is the exact 0.25 deg / cos(dec), no floor"


def test_cx02_reopen_truncated_box_zero_counts_are_unknown_not_absent(monkeypatch):
    import capabilities.alma as alma_mod

    monkeypatch.setattr(alma_mod, "_match_cross_archive_sources_impl", lambda ctx, **kw: {
        "success": True, "sources_tested": 3, "matched_sources": 1, "footprint_mode": "box_fov",
        "results": [{"source_name": "a", "alma_observations": 2}, {"source_name": "b", "alma_observations": 0},
                    {"source_name": "c", "alma_observations": 0}],
        "archive_errors": ["ALMA box query hit the 50000-row cap: pointings beyond it are missing, so ALMA matches are a lower bound"]})
    out = _run(alma_tools.CrossArchiveMatch(), CallContext(services={"get_last_run_result": lambda: None,
                                                                       "set_last_run_result": lambda v: None}),
               catalog_name="perseus_protostars", archives=["ALMA"])
    statuses = [r["alma_status"] for r in out["results"]]
    assert statuses == ["ok", "unknown_truncated", "unknown_truncated"]
    assert out["phase_status"]["ALMA"] == "truncated" and out["alma_status_unknown_for"] == 2
    assert out["status"] == "partial"


def test_cx04_cx09_extent_cone_overlap_is_computed_at_the_server():
    from services.alma_science_queries import EXTENT_CENTRE_PAD_DEG, alma_cone_adql, alma_cone_count_adql, alma_cone_where

    where = alma_cone_where(52.0, 31.0, 180 / 3600, footprint="contains")
    assert EXTENT_CENTRE_PAD_DEG == 0.5 and f"{180 / 3600 + 0.5:.8f}" in where
    assert "s_fov / 2.0" in where and "INTERSECTS" not in where
    assert "CONTAINS(POINT('ICRS', 52.00000000, 31.00000000), s_region) = 1" in where
    assert "CONTAINS(POINT('ICRS', s_ra, s_dec), CIRCLE('ICRS', 52.00000000, 31.00000000, 0.05000000)) = 1" in where
    # The data query and the count query share the predicate, so totals and caps refer to the same set (CX-09).
    q = alma_cone_adql(52.0, 31.0, 0.05, top=100, footprint="contains", band=[6])
    c = alma_cone_count_adql(52.0, 31.0, 0.05, footprint="contains", band=[6])
    assert q.split(" WHERE ", 1)[1] == c.split(" WHERE ", 1)[1]
    # Codex's counterexamples, evaluated with the same flat-sky test the server runs.
    import math

    def reaches(ra, dec, fov_deg, r=180 / 3600, ra0=52.0, dec0=31.0):
        dra, ddec = (ra - ra0) * math.cos(math.radians(dec0)), dec - dec0
        return math.hypot(dra, ddec) <= r + fov_deg / 2 and math.hypot(dra, ddec) <= r + EXTENT_CENTRE_PAD_DEG
    assert reaches(52.065, 31.0, 288 / 3600) and reaches(52.19, 31.0, 0.25)
    assert not reaches(52.1, 31.0, 20 / 3600)
    # Near a pole or across RA 0 the flat test is not used.
    assert "INTERSECTS" in alma_cone_where(10.0, -85.0, 0.1, footprint="contains")
    assert "INTERSECTS" in alma_cone_where(0.2, 10.0, 0.1, footprint="contains")


def test_cx10_only_the_final_sb_suffix_names_the_array():
    from services.alma_science_queries import classify_array_row, infer_arrays_from_schedblock

    assert infer_arrays_from_schedblock("Target_TP-cal_a_06_TM1") == ["12m"]
    assert infer_arrays_from_schedblock("X_a_06_TP") == ["TP"] and infer_arrays_from_schedblock("X_a_06_INT") == []
    assert classify_array_row("A001:DV01 A002:DV02", "Target_TP-cal_a_06_TM1") == ["12m"]


def test_cx11_missing_group_ids_are_not_one_science_goal():
    from services import alma_server_side as ss

    rows = [("P", None, "m1", "X_TM1", "A001:DV01"), ("P", None, "m2", "X_7M", "J501:CM01"), ("P", "", "m3", "X_TP", "N601:PM01")]

    def run(q):
        if "SELECT DISTINCT proposal_id, group_ous_uid" in q:
            return pd.DataFrame(rows, columns=["proposal_id", "group_ous_uid", "member_ous_uid", "schedblock_name", "antenna_arrays"])
        return pd.DataFrame([{"proposal_id": "P", "n_mous": 3, "n_eb": 3, "n_rows": 3, "target_name": "t", "pi_name": "p",
                              "obs_title": "o", "band_list": "6"}])
    out = ss.array_combo_projects_server_side(run, 9, ["12m", "7m", "TP"])
    assert out.extras["matching_projects"] == 1
    assert out.extras["alternative_counts"]["all_arrays_in_one_group_ous"] == 0
    assert list(out.frame["all_in_one_group_ous"]) == [False]


def test_cx02_truncated_box_with_require_all_counts_dropped_sources_as_unknown(monkeypatch):
    import capabilities.alma as alma_mod

    monkeypatch.setattr(alma_mod, "_match_cross_archive_sources_impl", lambda ctx, **kw: {
        "success": True, "sources_tested": 9, "matched_sources": 1, "footprint_mode": "box_fov",
        "results": [{"source_name": "a", "alma_observations": 2, "jwst_observations": 1}],
        "archive_errors": ["ALMA box query hit the 50000-row cap: pointings beyond it are missing, so ALMA matches are a lower bound"]})
    out = _run(alma_tools.CrossArchiveMatch(), CallContext(services={"get_last_run_result": lambda: None,
                                                                       "set_last_run_result": lambda v: None}),
               catalog_name="perseus_protostars", archives=["ALMA", "JWST"], require_all_archives=True)
    assert out["alma_status_unknown_for"] == 8 and out["status"] == "partial"
    assert "up to 8" in out["headline"]


def test_cx05_cross_archive_match_reports_the_box_method(monkeypatch):
    import capabilities.alma as alma_mod

    monkeypatch.setattr(alma_mod, "_match_cross_archive_sources_impl", lambda ctx, **kw: {
        "success": True, "sources_tested": 9, "matched_sources": 1, "footprint_mode": "box_fov", "results": [],
        "archive_errors": []})
    out = _run(alma_tools.CrossArchiveMatch(), CallContext(services={"get_last_run_result": lambda: None,
                                                                       "set_last_run_result": lambda v: None}),
               catalog_name="perseus_protostars", archives=["ALMA"])
    assert out["alma_query_mode"].startswith("one box query")


def test_cx03_multi_line_recipe_notes_are_served_whole():
    out = _run(alma_tools.CodeRecipe(), CallContext(), task="query by object name", library="astroquery")
    notes = out["recipes"][0]["notes"]
    assert "10 arcmin" in notes and "NGC_nnnn" in notes and "query_region" in notes and "public is three-state" in notes


def test_cx08_recipe_task_ids_survive_metadata_parsing():
    recipes = alma_tools.CodeRecipe._load_recipes("astroquery", ["object_search", "adql"])
    assert [r["task"] for r in recipes][:1] == ["object_search"]
    assert all(r["task"] not in ("notes", "library", "last_tested", "tested_against") for r in recipes)


def test_cx06_overlay_checks_do_not_claim_alignment_without_reprojection():
    checks = " ".join(alma_tools.ArchiveOverlay._overlay_checks({"reprojected": False}, "JWST"))
    assert "could NOT be reprojected" in checks and "reprojected onto the base image's WCS" not in checks
    checks = " ".join(alma_tools.ArchiveOverlay._overlay_checks({"reprojected": True}, "JWST"))
    assert "reprojected onto the base image's WCS" in checks


def test_cx07_invalid_release_date_is_caller_input_and_sends_no_query():
    ctx, tap = _ctx(lambda q: pd.DataFrame())
    out = _run(alma_tools.AlmaSourceSummary(), ctx, target="M83", released_before="soon")
    assert out["success"] is False and out["status"] == "invalid_input" and tap.queries == []


def test_r4_census_results_carry_method_caveats_and_a_reproducible_query():
    def answer(q):
        if "GROUP BY proposal_id" in q:
            return pd.DataFrame([{"proposal_id": "2021.1.01247.S", "n_mous": 2, "n_eb": 2, "n_rows": 8, "target_name": "Sun",
                                  "pi_name": "p", "obs_title": "t", "band_list": "3"}])
        return pd.DataFrame([{"proposal_id": "2021.1.01247.S", "target_name": "Sun", "band_list": "3",
                              "antenna_arrays": "A001:DV01", "schedblock_name": "x_TM1", "data_rights": "Public"}])

    from tests.unit.test_new_oneshot_tools import _alma_ctx

    ctx, _, _ = _alma_ctx(answer)
    out = _run(alma_tools.AlmaProjectCensus(), ctx, cycle=8, constraint={"type": "solar"})
    caveats = " ".join(out["method_caveats"])
    assert "ephemeris" in caveats and "Sunyaev" in caveats and "member_ous_uid" in caveats
    assert out["reproduce"]["simple_adql"].endswith(
        "scientific_category = 'Sun' AND (proposal_id LIKE '2021.1.%' OR proposal_id LIKE '2021.2.%' OR proposal_id LIKE '2021.A.%')"
        " AND science_observation = 'T'")
    assert "projectCode=2021.1.*%7C2021.2.*%7C2021.A.*&observationsScientificCategory=Sun" in out["reproduce"]["asa_web"]
    pub = _run(alma_tools.AlmaProjectCensus(), ctx, cycle=8, constraint={"type": "solar"}, public_only=True)
    assert pub["reproduce"]["simple_adql"].endswith("AND data_rights = 'Public'") and "publicData=true" in pub["reproduce"]["asa_web"]


def test_cx04_overlap_predicate_matches_exact_spherical_geometry():
    """Evaluate the generated ADQL arithmetic itself and compare with great-circle separations (independent check)."""
    import math
    import random

    from services.alma_science_queries import alma_cone_where

    def exact_sep(ra1, de1, ra2, de2):
        r1, d1, r2, d2 = map(math.radians, (ra1, de1, ra2, de2))
        c = math.sin(d1) * math.sin(d2) + math.cos(d1) * math.cos(d2) * math.cos(r1 - r2)
        return math.degrees(math.acos(max(-1.0, min(1.0, c))))

    rng = random.Random(7)
    cases = [(52.0, -70.0, 0.05, 52.7309511, -70.167, 0.5),       # Codex probes: separations just inside the reach
             (52.0, -75.0, 0.3, 54.89789338996636, -75.0, 0.9), (52.0, -75.0, 0.5, 54.9, -75.0, 0.5)]
    for _ in range(2000):
        ra0, de0, r = rng.uniform(30, 330), rng.uniform(-78, 78), rng.uniform(0.01, 0.6)
        fov = rng.uniform(0.005, 0.9)
        ang, dist = rng.uniform(0, 2 * math.pi), rng.uniform(0, r + 0.5)
        de = de0 + dist * math.sin(ang)
        ra = ra0 + dist * math.cos(ang) / math.cos(math.radians(de0))
        cases.append((ra0, de0, r, ra, de, fov))
    checked = 0
    for ra0, de0, r, ra, de, fov in cases:
        where = alma_cone_where(ra0, de0, r, footprint="contains")
        if "INTERSECTS" in where:
            continue  # near-pole fallback, tested elsewhere
        start, end = where.index("AND ((s_ra") + 4, where.rindex("s_fov / 2.0)") + len("s_fov / 2.0)")
        expr = where[start:end].replace("COS", "math.cos")
        got = eval(expr, {"math": math, "s_ra": ra, "s_dec": de, "s_fov": fov})  # noqa: S307 - our own generated text
        sep = exact_sep(ra0, de0, ra, de)
        if sep > r + 0.5:
            continue  # outside the indexed candidate circle (a separate CONTAINS clause)
        reach = r + fov / 2
        if sep <= reach:
            assert got, ("a true overlap was rejected", ra0, de0, r, ra, de, fov, sep)
        elif not got:
            pass
        else:
            assert sep - reach <= 2.0 / 3600, ("accepted a field more than 2 arcsec short", sep - reach)
        checked += 1
    assert checked > 1500


def test_r4_every_archive_oneshot_directive_asks_for_items_method_and_caveats():
    from core import oneshot_routing as orr

    route = orr.detect_oneshot_intent("Give me the ALMA archive URL for NGC 253 in Band 3")
    assert route and route["directive"].endswith(orr.RESULT_ANSWER_NOTE)
    for text in (orr.census_directive(None), orr.cross_archive_directive(None), orr.overlay_directive("ALMA contours on JWST")):
        assert orr.RESULT_ANSWER_NOTE in text
    assert orr.with_result_note(orr.with_result_note("x")).count(orr.RESULT_ANSWER_NOTE) == 1
    assert orr.RESULT_ANSWER_NOTE in orr.dispatched_directive(route["directive"])


def test_r4_vizier_catalogue_loader():
    from astropy.table import Table

    from services import cross_archive_matcher as cam

    t2 = Table({"_RAJ2000": [87.894, 87.844, float("nan")], "_DEJ2000": [2.946, 2.935, 1.0], "FieldA": ["HH270VLA1", "", "x"]})
    t2.meta["description"] = "ALMA fields"
    t4 = Table({"_RAJ2000": [1.0], "_DEJ2000": [2.0]})
    fake = lambda cat_id, limit: {"J/ApJ/890/130/table2": t2, "J/ApJ/890/130/table4": t4}  # noqa: E731
    label, rows = cam.load_vizier_catalog("J/ApJ/890/130", fetch=fake)
    assert label == "vizier:J/ApJ/890/130/table2 (ALMA fields); other tables: J/ApJ/890/130/table4"
    assert rows == [{"source_name": "HH270VLA1", "ra": 87.894, "dec": 2.946}, {"source_name": "table2_2", "ra": 87.844, "dec": 2.935}]
    label, rows = cam.load_vizier_catalog("J/ApJ/890/130/table4", fetch=fake)
    assert label.startswith("vizier:J/ApJ/890/130/table4") and rows[0]["source_name"] == "table4_1"
    import pytest
    with pytest.raises(ValueError):
        cam.load_vizier_catalog("J/ApJ; DROP", fetch=fake)


def test_r4_line_census_counts_objects_by_position():
    from services import alma_server_side as ss

    frame = pd.DataFrame([
        {"member_ous_uid": "m1", "s_ra": 10.0, "s_dec": -20.0},
        {"member_ous_uid": "m2", "s_ra": 10.0 + 20 / 3600 / 0.9397, "s_dec": -20.0},   # 20" away: same object
        {"member_ous_uid": "m3", "s_ra": 10.1, "s_dec": -20.0},                        # 5.6' away: another
        {"member_ous_uid": "m4", "s_ra": None, "s_dec": None},
    ])
    groups = ss.position_groups(frame)
    assert groups["m1"] == groups["m2"] != groups["m3"] and "m4" not in groups

    def answer(q):
        rows = [{"member_ous_uid": "m1", "proposal_id": "P1", "target_name": "DiskA", "s_ra": 10.0, "s_dec": -20.0},
                {"member_ous_uid": "m2", "proposal_id": "P2", "target_name": "Disk_A", "s_ra": 10.0 + 20 / 3600 / 0.9397, "s_dec": -20.0},
                {"member_ous_uid": "m3", "proposal_id": "P2", "target_name": "DiskB", "s_ra": 10.1, "s_dec": -20.0}]
        return pd.DataFrame(rows)
    out = ss.line_set_projects_server_side(answer, ["12CO", "13CO"])
    assert "SELECT member_ous_uid" in out.queries[0] and "MIN(s_ra) AS s_ra" in out.queries[0]
    assert out.extras["distinct_objects_by_position"] == 2
    assert dict(zip(out.frame["proposal_id"], out.frame["n_objects_by_position"])) == {"P1": 1, "P2": 2}
    assert any("not a detection" in n for n in out.notes) and any("different names" in n for n in out.notes)


def test_cx07_cx08_position_groups_at_the_pole_and_missing_positions():
    from services import alma_server_side as ss

    pole = pd.DataFrame([{"member_ous_uid": "a", "s_ra": 0.0, "s_dec": -89.996}, {"member_ous_uid": "b", "s_ra": 180.0, "s_dec": -89.996}])
    g = ss.position_groups(pole)
    assert g["a"] == g["b"], "28.8 arcsec apart across the pole: one object"

    def answer(q):
        return pd.DataFrame([{"member_ous_uid": "m1", "proposal_id": "P1", "target_name": "x", "s_ra": None, "s_dec": None}])
    out = ss.line_set_projects_server_side(answer, ["12CO", "13CO"])
    assert out.extras["distinct_objects_by_position"] is None and any("unknown" in n for n in out.notes)
