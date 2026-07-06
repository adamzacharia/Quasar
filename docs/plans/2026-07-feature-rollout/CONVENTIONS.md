# CONVENTIONS — shared contract for all 2026-07 feature-rollout tasks

Every implementing model (Codex `@cx-build`, Claude, or other) MUST read this
file plus the two exemplar files before writing code:

- `services/alerce_client.py` — the canonical service-client pattern
- `tests/unit/test_alerce_client.py` — the canonical unit-test pattern

## Environment facts

- Windows 11; venv at `.venv` (Python 3.13). Run everything with
  `.venv/Scripts/python.exe`.
- Test command: `.venv/Scripts/python.exe -m pytest tests/unit/<file> -q`
  No `pytest-timeout` plugin exists — do not use `--timeout`.
- Unit tests must be FULLY OFFLINE (monkeypatch all network; no live calls).
- New deps allowed ONLY if the feature spec lists them (Phase 0 pre-installs
  `lightkurve` and `psrqpy`; everything else must use already-installed
  packages: requests, astropy, astroquery, pyvo, numpy, pandas, matplotlib).
- NEVER read or print generated/build artifacts (`.next/`, `dist/`,
  `node_modules/`, compiled CSS). Reason on source only.
- **NEVER run `git commit`, `git push`, or any other git state-changing
  command. Leave ALL changes uncommitted in the working tree.** (Added after
  the f03-rsed incident 2026-07-04: a Codex build committed and pushed to
  both remotes unprompted — the Windows sandbox does not block this, so it
  is a hard prompt-level rule.)

## Service-module contract (mirror `services/alerce_client.py`)

1. One module per feature under `services/` with a module docstring.
2. `from __future__ import annotations` at top.
3. One client/service class:
   - Constructor takes keyword-only overrides:
     `def __init__(self, *, base_url=None, timeout=None, <injectables>)`.
   - Env-var overrides: `<FEATURE>_BASE_URL`, `<FEATURE>_TIMEOUT`
     (see `ALERCE_BASE_URL` / `ALERCE_TIMEOUT` for the pattern).
   - Default timeout 30 s; EVERY network call passes `timeout=`.
   - For testability, accept injectable callables/clients in the constructor
     (e.g., `table_loader=...`, `alerce_client=...`) instead of requiring
     tests to patch deep internals.
4. Every public method returns a dict and NEVER raises:
   - Success: `{"success": True, "rows": [...], "count": int,
     "warnings": [str, ...], "provenance": {...}}` (keys vary per method but
     `success`, `warnings`, `provenance` are mandatory).
   - Failure: `{"success": False, "error": str(exc)}` via a broad
     `except Exception as exc:`.
   - `provenance` records service name, endpoint/base URL, and the exact
     query parameters used.
   - `warnings` records every clamp, fallback, or skipped sub-source.
5. Row values must be plain JSON types: float/int/str/bool/None. Convert
   numpy scalars with `float()`/`int()`; map NaN/masked values to `None`.
6. Plot-producing methods follow `AlerceClient.plot_light_curve`:
   - `from services.plotting import PlottingService`; use
     `self.plotting_service._apply_style(dark=False)`; save PNG under
     `services.plotting.PLOT_OUTPUT_DIR` (module attribute — tests
     monkeypatch it); return `{"success": True, "path": <png path>, ...}`.
7. Heavy imports (`astroquery.*`, `pyvo`, `lightkurve`, `psrqpy`) must be
   LAZY — imported inside the method or a module-level `_import_x()` helper,
   never at module import time. `import services.<module>` must be cheap.
8. Input validation helpers: clamp radii/row counts to sane maxima and append
   a warning instead of erroring (see `_normalize_radius` in alerce_client).

## Unit-test contract (mirror `tests/unit/test_alerce_client.py`)

- File: `tests/unit/test_<service_module>.py`.
- Use a local `FakeResponse` class + `monkeypatch.setattr(<module>.requests, "get", fake_get)`
  (or patch the injected loader callable) — never hit the network.
- Required test cases for every feature (plus feature-specific ones in spec):
  1. Happy path: request params correctly built (assert URL + params +
     timeout), rows normalized to expected schema.
  2. Clamping/validation produces `warnings` and still succeeds.
  3. HTTP/parse failure returns `{"success": False, "error": ...}` and the
     error string is informative (e.g., contains `HTTP 404`).
  4. Empty result set: `success: True, count: 0` (NOT an error).
- Plot tests: `monkeypatch.setattr(plotting, "PLOT_OUTPUT_DIR", _plot_dir("<feature>"))`
  writing under `test_results/<feature>/`, assert `out["path"].endswith(".png")`.

## Smoke-script contract

Each task creates `scripts/smoke/smoke_<feature>.py` (create `scripts/smoke/`
if missing): a standalone script that makes 1-3 LIVE calls and prints a
compact PASS/FAIL summary with the key numbers. The implementing model does
NOT run it (may lack network); the reviewer runs it at the review gate:
`.venv/Scripts/python.exe scripts/smoke/smoke_<feature>.py`
Exit code 0 on pass, 1 on fail. Keep runtime < 120 s.

## core/agent.py wiring recipe (EXACT anchors — insertions only)

`core/agent.py` is ~10,850 lines. Make MINIMAL insertions at these anchors;
never reorder or reformat existing code.

1. **Lazy getter** — insert AFTER the `_get_alerce_client` method
   (search: `def _get_alerce_client`):
   ```python
   def _get_<feature>_service(self):
       if not hasattr(self, "_<feature>_service_instance"):
           from services.<module> import <Class>
           self._<feature>_service_instance = <Class>()
       return self._<feature>_service_instance
   ```
2. **Tool wrapper methods** — insert near the `_search_ztf_alerts` /
   `_ztf_light_curve` block (search: `def _search_ztf_alerts`). Template:
   ```python
   def _<tool_name>(self, ...typed args with defaults...) -> Dict[str, Any]:
       self.last_run_result = None
       try:
           # position-taking tools resolve names like this:
           ra_f, dec_f, label = self._live_imagery_coordinates(target_name=target_name, ra=ra, dec=dec)
           result = self._get_<feature>_service().<method>(...)
           if not result.get("success"):
               return result
           # TABLE results:
           return self._external_catalog_table_result(
               result["rows"], columns=[...], source="<human source>",
               filter_label=f"<what/where>", tool_name="<tool_name>",
               warnings=result.get("warnings", []),
               provenance=result.get("provenance", {}),
           )
           # or PLOT/IMAGE results (result must carry a PNG "path"):
           # return self._datalab_attach_image_result(result, f"<caption>", meta=None)
       except Exception as e:
           return {"success": False, "error": str(e)}
   ```
   For `columns=`, list the useful columns but keep only those present:
   `present = [c for c in columns if any(c in r for r in rows)] or columns[:6]`
   (see `_search_ztf_alerts`).
3. **Registration** — inside `_register_tools()`, append new
   `self.tool_registry.register(Tool(...))` blocks immediately after the LAST
   existing registration in that method. Parameters use full JSON schema:
   ```python
   self.tool_registry.register(Tool(
       name="<tool_name>",
       description="<one sentence: what it returns + when to use it>",
       function=self._<tool_name>,
       parameters={"type": "object", "properties": {...}, "required": [...]},
       category="archive",   # external queries; use "analysis" for plots/computation
   ))
   ```
4. **Status labels** — find the dict containing
   `"search_ztf_alerts": "Searching ALeRCE/ZTF alerts"` (~line 4279) and add
   one `"<tool_name>": "<Gerund phrase>"` entry per tool.
5. **System-prompt guidance** — find the `**LIVE IMAGERY RULE**` bullet
   (~line 754). Add ONE new bullet line nearby (same list) telling the agent
   when to use the new tools. One sentence per feature, mention tool names in
   backticks.
6. Wrapper args that take a sky position must accept
   `target_name: Optional[str] = None, ra: Optional[float] = None,
   dec: Optional[float] = None` and resolve via `_live_imagery_coordinates`.

## Definition of done (every task)

- [ ] Service module + unit tests + smoke script created; agent.py wired
      (getter, wrappers, registrations, status labels, prompt bullet).
- [ ] `.venv/Scripts/python.exe -m pytest tests/unit/test_<module>.py -q` green.
- [ ] `.venv/Scripts/python.exe -m pytest tests/unit -q` — no NEW failures
      vs the Phase 0 baseline recorded in PROGRESS.md.
- [ ] `.venv/Scripts/python.exe -c "import services.<module>"` returns instantly
      (proves lazy imports).
- [ ] No edits outside: `services/<module>.py`, `tests/unit/test_<module>.py`,
      `scripts/smoke/smoke_<feature>.py`, `core/agent.py` (4 anchor insertions).
