"""The budget hierarchy rule (services/tool_budgets.py).

RULE: every inner network budget is STRICTLY BELOW the outer tool guard, with
GUARD_HEADROOM_SECONDS to spare; the guard is the last resort. Before
2026-09-21 the Data Lab image download was 180 s (wall 270 s) under a 150 s
guard, the ALMA TAP client 3 x 120 s under 150 s, and MAST queries 600 s each
— so the guard ALWAYS fired first and its blunt "timed out after 150s"
replaced every service's own partial/structured result.

This module fails the build when:
  * any declared tool's inner chain exceeds guard - headroom;
  * a tool with an elevated guard (GUARD_OVERRIDES) has no declaration;
  * a registered tool in a network-facing category is neither declared,
    listed as a known unbounded transport, nor listed as local-only.
It also pins the dynamic half of the rule: Deadline / bounded_timeout /
call_bounded semantics.
"""
from __future__ import annotations

import re
import threading
import time
from pathlib import Path

import pytest

from services import tool_budgets as tb

REPO = Path(__file__).resolve().parents[2]

# Registered tools that never touch the network (pure compute / local state).
LOCAL_TOOLS = {
    "calculate_redshift", "convert_coordinates", "calculate_beam", "calculate_alma_sensitivity",
    "calculate_doppler_shift", "check_line_coverage", "check_co_lines", "filter_results",
    "plot_alma_results", "plot_spectrum", "plot_sky_map", "datalab_get_result", "datalab_save_result",
    "datalab_list_my_tables", "datalab_load_my_table", "datalab_confirm_sky_area", "datalab_job_status",
    "datalab_job_cancel", "datalab_export_notebook", "datalab_catalog_scatter", "datalab_sky_density_map",
    "datalab_period_fold", "datalab_lss_wedge", "datalab_sed_plot", "generate_casa_imaging_script",
    "generate_casa_calibration_script", "generate_jupyter_notebook", "analyze_uv_coverage", "download_data",
    "browse_schema", "browse_alma_guidance", "monitor_add_target", "monitor_list_targets",
    "monitor_remove_target", "list_mmu_hats_catalogs",
    "alma_reference_table",  # bundled Handbook reference tables (services/alma_reference.py), no network
    "code_recipe",  # curated snippets from docs/recipes/*.py, no network
}


def _registered_tool_names():
    """Tool names from the registration sources.

    register_tools() needs a fully constructed agent (ADS client, services),
    so the registry is read statically: literal Tool(name=...) registrations
    in core/, plus capability classes registered by iterating a module's
    CAPABILITIES list (their ``name = "..."`` class attribute).
    """
    names = set()
    literal = re.compile(r'Tool\(\s*name\s*=\s*"([A-Za-z0-9_]+)"', re.S)
    for rel in ("core/tool_registrations.py", "core/agent.py"):
        src = (REPO / rel).read_text(encoding="utf-8", errors="replace")
        names.update(literal.findall(src))
    class_attr = re.compile(r'^\s+name\s*=\s*"([A-Za-z0-9_]+)"\s*$', re.M)
    for path in sorted((REPO / "capabilities").glob("*.py")):
        names.update(class_attr.findall(path.read_text(encoding="utf-8", errors="replace")))
    assert len(names) > 100, f"expected the full registry, found {len(names)} names"
    return names


# ── static rule ──────────────────────────────────────────────────────────


def test_every_declared_inner_budget_is_strictly_below_its_guard(monkeypatch):
    for var in ("QUASAR_TOOL_TIMEOUT_SECONDS", "QUASAR_TOOL_TIMEOUT_OVERRIDES", "ALMA_TAP_TIMEOUT_SECONDS",
                "ALMA_TAP_MAX_ATTEMPTS", "HIPS2FITS_TIMEOUT", "DATALAB_TIMEOUT_SECONDS", "DATALAB_SIA_TIMEOUT_SECONDS",
                "DATALAB_IMAGE_DOWNLOAD_TIMEOUT_SECONDS", "SPARCL_READ_TIMEOUT_SECONDS", "MAST_QUERY_TIMEOUT_SECONDS",
                "SIMBAD_TIMEOUT_SECONDS"):
        monkeypatch.delenv(var, raising=False)
    violations = tb.check_hierarchy()
    assert violations == [], "inner budget >= guard - headroom:\n  " + "\n  ".join(violations)
    for name, getter in tb.INNER_BUDGETS.items():
        if name in tb.DEADLINE_ONLY_TOOLS:
            continue  # no static constant exists; the runtime clamp is the bound
        guard = tb.guard_seconds(name)
        parts = getter()
        if name in tb.LOOP_TOOLS:
            worst = max(parts.values())
            assert worst < guard, f"{name}: a single inner call of {worst}s is not strictly below its guard {guard}s"
        else:
            total = sum(parts.values())
            assert total < guard, f"{name}: inner {total}s is not strictly below its guard {guard}s"


def test_every_elevated_guard_has_a_declared_inner_budget():
    missing = sorted(t for t in tb.GUARD_OVERRIDES if t not in tb.INNER_BUDGETS)
    assert missing == [], f"tools with an elevated guard but no inner-budget declaration: {missing}"


def test_every_registered_network_tool_is_accounted_for():
    registered = _registered_tool_names()
    accounted = set(tb.INNER_BUDGETS) | set(tb.UNBOUNDED_TRANSPORT_TOOLS) | LOCAL_TOOLS
    unknown = sorted(registered - accounted)
    assert unknown == [], (
        "registered tools with no inner-budget declaration — add them to services/tool_budgets.py "
        f"(_declare_defaults) or to LOCAL_TOOLS / UNBOUNDED_TRANSPORT_TOOLS with a reason: {unknown}"
    )
    stale = sorted(set(tb.INNER_BUDGETS) - registered)
    assert stale == [], f"declared budgets for tools that are no longer registered: {stale}"


def test_a_regression_is_caught_on_a_real_tool(monkeypatch):
    # The exact 2026-09-20 inversion: a 180 s tile download under a 150 s guard.
    monkeypatch.setenv("DATALAB_IMAGE_DOWNLOAD_TIMEOUT_SECONDS", "180")
    violations = tb.check_hierarchy(["datalab_color_image"])
    assert violations and "datalab_color_image" in violations[0], violations


def test_env_overrides_are_part_of_the_check(monkeypatch):
    # Operators can re-invert the hierarchy from the environment; the check
    # reads the live values, so the suite run with such an env fails loudly.
    monkeypatch.setenv("HIPS2FITS_TIMEOUT", "200")
    assert tb.check_hierarchy(["hips_cutout"])
    monkeypatch.setenv("HIPS2FITS_TIMEOUT", "30")
    assert not tb.check_hierarchy(["hips_cutout"])


def test_guard_overrides_only_went_down():
    # Constraint of the 2026-09-21 fix: no timeout was raised to make the hierarchy hold.
    assert tb.GUARD_DEFAULT_SECONDS == 150.0
    assert tb.GUARD_OVERRIDES["search_by_target"] <= 150.0
    assert tb.GUARD_OVERRIDES["search_by_position"] <= 150.0
    assert tb.GUARD_OVERRIDES["query_alma_science_archive"] <= 180.0
    assert max(tb.GUARD_OVERRIDES.values()) <= 600.0


def test_hosts_are_declared_for_archive_tools():
    from services.host_breaker import host_of

    for tool, host in (("hips_cutout", "alasky.cds.unistra.fr"), ("datalab_color_image", "datalab.noirlab.edu"),
                       ("search_by_target", "almascience.nrao.edu"), ("match_cross_archive_sources", "mast.stsci.edu")):
        assert host in {host_of(h) for h in tb.hosts_for(tool)}, (tool, tb.hosts_for(tool))
    assert "datalab_color_image" in tb.tools_for_host("datalab.noirlab.edu")


def test_datalab_tools_are_declared_per_service_path():
    """SOFT breaker circuits are per service path: SQL tools on /query, cutout
    tools on /sia, density vetting on both (UI benchmark 2026-09-22, L07-L10)."""
    assert tb.hosts_for("datalab_cone_count") == ("datalab.noirlab.edu/query",)
    assert tb.hosts_for("datalab_image_cutout") == ("datalab.noirlab.edu/sia",)
    assert set(tb.hosts_for("datalab_density_vetting")) == {"datalab.noirlab.edu/query", "datalab.noirlab.edu/sia"}
    sia_tools = set(tb.tools_for_host("datalab.noirlab.edu/sia"))
    assert {"datalab_image_cutout", "datalab_color_image", "datalab_cutout_grid", "datalab_density_vetting"} <= sia_tools
    assert "datalab_cone_count" not in sia_tools, "an SIA outage never lists the SQL tools as affected"
    assert "datalab_cone_count" in tb.tools_for_host("datalab.noirlab.edu"), "a host-level trip affects every service"


# ── dynamic rule: Deadline / bounded_timeout / call_bounded ──────────────


@pytest.fixture(autouse=True)
def _no_deadline():
    tb.end_tool_deadline()
    yield
    tb.end_tool_deadline()


def test_bounded_timeout_is_identity_outside_a_tool():
    assert tb.bounded_timeout(600.0) == 600.0
    assert tb.remaining_seconds() is None


def test_begin_tool_deadline_subtracts_headroom_and_clamps():
    d = tb.begin_tool_deadline("t", 150.0)
    assert d is not None and 134.0 < d.remaining() <= 135.0
    assert tb.bounded_timeout(600.0) <= 135.0
    assert tb.bounded_timeout(10.0) == 10.0
    assert tb.begin_tool_deadline("t", None) is None and tb.current_deadline() is None


def test_exhausted_budget_raises_before_the_call():
    tb.begin_tool_deadline("t", 0.2 + tb.GUARD_HEADROOM_SECONDS)
    time.sleep(0.25)
    with pytest.raises(tb.BudgetExhausted) as info:
        tb.bounded_timeout(30.0, minimum=1.0, label="MAST query")
    assert "MAST query" in str(info.value)
    assert isinstance(info.value, TimeoutError), "classified as a timeout by existing _is_timeout_error checks"


def test_call_bounded_abandons_a_hung_worker_and_adopts_the_deadline():
    d = tb.begin_tool_deadline("t", 5.0 + tb.GUARD_HEADROOM_SECONDS)
    seen = {}

    def hung():
        seen["deadline"] = tb.current_deadline()
        time.sleep(2.0)
        return "late"

    t0 = time.perf_counter()
    with pytest.raises(TimeoutError):
        tb.call_bounded(hung, 0.2, label="probe")
    assert time.perf_counter() - t0 < 1.0
    time.sleep(0.05)
    # Helper threads inherit the caller's deadline as a CHILD: same budget and
    # call token (one tool call for the host breaker), own cancel flag -- and the
    # abandoned helper was cancelled so it cannot start new requests.
    child = seen["deadline"]
    assert child is not d and child.token is d.token and abs(child.remaining() - d.remaining()) < 0.5
    assert child.cancelled() and not d.cancelled()
    assert tb.call_bounded(lambda: "ok", 1.0) == "ok"


def test_deadlines_are_thread_local():
    tb.begin_tool_deadline("t", 100.0)
    other = {}

    def probe():
        other["deadline"] = tb.current_deadline()

    th = threading.Thread(target=probe)
    th.start()
    th.join()
    assert other["deadline"] is None
