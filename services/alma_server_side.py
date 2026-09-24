"""Server-side ALMA archive aggregation for the named science query types.

UI benchmark 2026-09-22 (D09, D11, D21, D22): every named query type of
``query_alma_science_archive`` executed ``SELECT TOP 5000 ... FROM ivoa.obscore
WHERE <weak or no filter> ORDER BY proposal_id`` and post-filtered in Python,
so every listed project was a 2011.0.* / 2012.1.* sliver and the headline
counts were meaningless. ALMA TAP supports ADQL aggregates -- live-probed on
2026-09-22 against almascience.nrao.edu: ``GROUP BY`` + ``COUNT(DISTINCT ...)``
(1.6 s for a project prefix), ``HAVING`` (1.7 s for one cycle), per-SPW
arithmetic ``frequency +/- bandwidth/2e9`` (rows are one SPW each: ``bandwidth``
is the SPW width in Hz, ``frequency`` its centre in GHz), a whole-archive line
coverage ``GROUP BY member_ous_uid`` in 17 s when a ``frequency BETWEEN``
range narrows the scan, per-cycle redshift aggregates in ~1 s. Queries that
return one row per SPW for the whole archive (DISTINCT with target_name) hit
the NRAO proxy's 60 s limit -- so the unit of work is a cycle or a
``GROUP BY`` aggregate, never a raw row pull.

Every function here takes ``run_query(adql) -> DataFrame`` (the caller's
budgeted TAP adapter, which records provenance) and returns a
:class:`ServerSideResult` with the per-project table, the exact ADQL list,
and an HONEST completeness statement (``complete`` / ``partial: cycles X-Y
scanned``). Rest frequencies and coverage windows ride along so the answer
can state them.
"""
from __future__ import annotations

import datetime as _dt
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

from services.alma_science_queries import (
    LINE_REST_FREQ_GHZ,
    as_text,
    band_tokens,
    band_token_where,
    cycle_periods_disclosure,
    escape_adql,
    first_nonempty,
    infer_arrays,
    infer_arrays_from_schedblock,
    line_names_for_input,
    line_names_for_species,
    parse_frequency_support_intervals,
    project_prefix_where,
    requested_bands,
)

__all__ = [
    "BWSW_THRESHOLD_MHZ",
    "ServerSideResult",
    "array_combo_projects_server_side",
    "bandwidth_switching_server_side",
    "coverage_where",
    "latest_cycle",
    "line_set_projects_server_side",
    "redshifted_line_projects_server_side",
    "solar_projects_server_side",
]

RunQuery = Callable[[str], pd.DataFrame]

# ALMA Proposer's Guide (Cycle 13, "Phase calibrator"): the OT resorts to
# Bandwidth Switching when the total bandwidth of the spectral setup is
# narrower than 937.5 MHz; Technical Handbook §10.5.4: "If all the spws are
# narrow bandwidth (< 1 GHz aggregate bandwidth) ... Bandwidth Switching".
BWSW_THRESHOLD_MHZ = 937.5
BWSW_CITATION = (
    "ALMA Proposer's Guide (Cycle 13), phase-calibrator search: BWSW when the total bandwidth of the "
    "spectral setup is < 937.5 MHz; ALMA Technical Handbook §10.5.4 Bandwidth Switching: all spws "
    "narrow (< 1 GHz aggregate bandwidth)."
)

# GROUP BY output cap (maxrec); a page at this size means "truncated".
AGG_MAXREC = 20000
# Widen the indexable frequency range by this much on each side of a line /
# window; an SPW is at most ~2 GHz wide, so its centre lies within 1 GHz of
# any frequency it covers.
RANGE_PAD_GHZ = 1.2


@dataclass
class ServerSideResult:
    frame: pd.DataFrame
    queries: List[str] = field(default_factory=list)
    complete: bool = True
    scanned: str = "whole archive"
    unscanned: List[str] = field(default_factory=list)
    n_units: int = 0
    unit: str = "MOUS"
    notes: List[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return "ok" if self.complete else "partial"

    def completeness_note(self) -> str:
        if self.complete:
            return f"complete: {self.scanned} scanned server-side ({self.n_units} {self.unit})."
        tail = f"; not scanned: {', '.join(self.unscanned)}" if self.unscanned else ""
        return f"partial: {self.scanned} scanned ({self.n_units} {self.unit}){tail}."


# ── helpers ──────────────────────────────────────────────────────────────

def latest_cycle(today: Optional[_dt.date] = None) -> int:
    """The newest cycle whose proposal year has started (Cycle N proposals
    carry year N + 2013 from Cycle 8 on; Cycle 13 = 2026)."""
    today = today or _dt.date.today()
    return max(0, min(30, today.year - 2013))


def all_cycles(newest_first: bool = True) -> List[int]:
    cycles = list(range(0, latest_cycle() + 1))
    return cycles[::-1] if newest_first else cycles


def coverage_where(nu_ghz: float, *, pad_ghz: float = RANGE_PAD_GHZ) -> str:
    """ADQL: this SPW row covers ``nu_ghz``. The BETWEEN range is the indexable
    part; the arithmetic is exact (bandwidth in Hz, frequency in GHz)."""
    nu = float(nu_ghz)
    return (
        f"(frequency BETWEEN {nu - pad_ghz:.4f} AND {nu + pad_ghz:.4f} "
        f"AND (frequency - bandwidth/2.0e9) <= {nu:.6f} AND (frequency + bandwidth/2.0e9) >= {nu:.6f})"
    )


def window_where(nu_lo: float, nu_hi: float, *, pad_ghz: float = RANGE_PAD_GHZ) -> str:
    """ADQL: this SPW row overlaps the observed window [nu_lo, nu_hi] GHz."""
    lo, hi = sorted((float(nu_lo), float(nu_hi)))
    return (
        f"(frequency BETWEEN {lo - pad_ghz:.4f} AND {hi + pad_ghz:.4f} "
        f"AND (frequency - bandwidth/2.0e9) <= {hi:.6f} AND (frequency + bandwidth/2.0e9) >= {lo:.6f})"
    )


def _science_where(science_only: bool) -> str:
    return "science_observation = 'T'" if science_only else "1=1"


def _band_where(band: Any) -> str:
    bands = requested_bands(band) if band is not None else []
    if band is not None and not bands:
        raise ValueError("band must contain ALMA band numbers from 1 to 10")
    if not bands:
        return "1=1"
    return "(" + " OR ".join(band_token_where(b) for b in bands) + ")"


def _in_list(values: Iterable[str]) -> str:
    return ", ".join(f"'{escape_adql(as_text(v))}'" for v in values)


def _truncated(df: pd.DataFrame) -> bool:
    return bool(df is not None and (df.attrs.get("truncated") or len(df) >= AGG_MAXREC))


def _remaining_seconds() -> Optional[float]:
    try:
        from services.tool_budgets import remaining_seconds

        return remaining_seconds()
    except Exception:  # pragma: no cover
        return None


def _budget_allows(min_seconds: float) -> bool:
    left = _remaining_seconds()
    return left is None or left >= min_seconds


def run_concurrently(fns: Sequence[Callable[[], Any]], *, wall_seconds: float = 120.0) -> List[Any]:
    """Run ``fns`` on daemon threads under CHILD deadlines of the current tool
    deadline (services/tool_budgets.py) and wait at most ``wall_seconds``.
    Each slot holds the return value or the exception raised; a slot that
    did not finish holds a TimeoutError."""
    from services.tool_budgets import adopt_deadline, current_deadline

    from services.tool_budgets import Deadline

    parent = current_deadline()
    if parent is not None:
        wall_seconds = min(wall_seconds, max(1.0, parent.remaining()))
    results: List[Any] = [TimeoutError("not finished within the budget")] * len(fns)
    # One child deadline per worker, created here so an abandoned worker can
    # be stopped individually: its next bounded request / is_cancelled() check
    # sees the cancel (guard CX-02, 2026-09-24: timed-out cutout panels kept
    # fetching). Outside a tool a standalone deadline plays the same role.
    # Each worker's deadline ENDS at the wall clock (+2 s grace): the requests
    # hook clamps every request's timeout to it, so an abandoned worker's
    # in-flight request ends by then as well.
    children = [parent.child_until(float(wall_seconds) + 2.0, label=f"concurrent-{i}") if parent is not None
                else Deadline(max(1.0, float(wall_seconds)) + 2.0, label=f"concurrent-{i}") for i in range(len(fns))]

    def _worker(i: int, fn: Callable[[], Any]) -> None:
        adopt_deadline(children[i])
        try:
            results[i] = fn()
        except BaseException as exc:  # noqa: BLE001 - reported per slot
            results[i] = exc

    threads = [threading.Thread(target=_worker, args=(i, fn), daemon=True, name=f"alma-agg-{i}") for i, fn in enumerate(fns)]
    for t in threads:
        t.start()
    end = time.monotonic() + wall_seconds
    for t in threads:
        t.join(max(0.0, end - time.monotonic()))
    for i, t in enumerate(threads):
        if t.is_alive():
            children[i].cancel("concurrent worker abandoned after its wall-clock budget")
    return results


def _scan_cycles(
    run_query: RunQuery,
    cycles: Sequence[int],
    make_query: Callable[[int], str],
    *,
    min_seconds_per_cycle: float,
    batch: int = 4,
) -> Tuple[List[str], List[pd.DataFrame], List[int], List[int], bool]:
    """Run one aggregate query per cycle, ``batch`` cycles concurrently, newest
    first, until every cycle is done or the tool budget runs low. Returns
    (queries, frames-with-a-cycle-column, scanned, unscanned, truncated).
    The first batch must succeed (a wholesale failure raises); a later
    failing cycle is recorded as unscanned."""
    queries: List[str] = []
    frames: List[pd.DataFrame] = []
    scanned: List[int] = []
    unscanned: List[int] = []
    truncated = False
    todo = list(cycles)
    first = True
    while todo:
        if not first and not _budget_allows(min_seconds_per_cycle):
            unscanned.extend(todo)
            break
        chunk, todo = todo[:batch], todo[batch:]
        qs = [make_query(c) for c in chunk]
        queries.extend(qs)
        outs = run_concurrently([(lambda q=q: run_query(q)) for q in qs], wall_seconds=150.0)
        for c, out in zip(chunk, outs):
            if isinstance(out, BaseException):
                if first and not scanned:
                    raise RuntimeError(f"cycle {c} aggregate failed: {type(out).__name__}: {str(out)[:160]}")
                unscanned.append(c)
                queries.append(f"-- cycle {c}: query failed ({type(out).__name__}); not scanned")
                continue
            scanned.append(c)
            if out is not None and not out.empty:
                out = out.copy()
                out["cycle"] = c
                frames.append(out)
                truncated |= _truncated(out)
        first = False
    return queries, frames, sorted(scanned), sorted(unscanned), truncated


def _agg_int(value: Any) -> Optional[int]:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


# ── 1. Cycle solar projects (D10) ────────────────────────────────────────

def solar_projects_server_side(run_query: RunQuery, cycle: int, *, science_only: bool = True, extra_where: str = "") -> ServerSideResult:
    """Per-project aggregate of the cycle's observations OF THE SUN, computed
    at the server, science scans only (calibrators such as J0238+1636 are
    excluded by ``science_observation = 'T'``; D10)."""
    from services.alma_science_queries import solar_where

    t0 = time.perf_counter()
    where = f"{project_prefix_where(int(cycle))} AND {solar_where()} AND {_science_where(science_only)}" + (f" AND {extra_where}" if extra_where else "")
    agg = (
        "SELECT proposal_id, COUNT(DISTINCT member_ous_uid) AS n_mous, COUNT(DISTINCT asdm_uid) AS n_eb, "
        "COUNT(*) AS n_rows, MIN(target_name) AS target_name, MIN(pi_name) AS pi_name, MIN(obs_title) AS obs_title, "
        "MIN(band_list) AS band_list, MIN(obs_release_date) AS obs_release_date_min, MAX(obs_release_date) AS obs_release_date_max "
        f"FROM ivoa.obscore WHERE {where} GROUP BY proposal_id"
    )
    queries = [agg]
    df = run_query(agg)
    frame = pd.DataFrame(columns=["proposal_id", "n_mous", "n_eb", "n_rows", "target_name", "pi_name", "obs_title", "band_list"])
    notes: List[str] = [cycle_periods_disclosure(int(cycle))]
    if df is not None and not df.empty:
        frame = df.copy()
        projects = frame["proposal_id"].map(as_text).tolist()
        # Distinct science targets, bands and arrays per project (bounded: the
        # matched project set is small).
        detail = (
            "SELECT DISTINCT proposal_id, target_name, band_list, antenna_arrays, schedblock_name, data_rights "
            f"FROM ivoa.obscore WHERE proposal_id IN ({_in_list(projects[:200])}) AND {_science_where(science_only)}"
        )
        queries.append(detail)
        try:
            det = run_query(detail)
        except Exception as exc:  # detail is optional
            det = pd.DataFrame()
            notes.append(f"per-project detail query failed ({type(exc).__name__}); counts are still exact.")
        if det is not None and not det.empty:
            targets, bands, arrays, rights = {}, {}, {}, {}
            for _, row in det.iterrows():
                pid = as_text(row.get("proposal_id"))
                targets.setdefault(pid, [])
                t = as_text(row.get("target_name"))
                if t and t not in targets[pid]:
                    targets[pid].append(t)
                bands.setdefault(pid, set()).update(band_tokens(row.get("band_list")))
                arr = arrays.setdefault(pid, [])
                for label in infer_arrays(row.get("antenna_arrays")) + infer_arrays_from_schedblock(row.get("schedblock_name")):
                    if label not in arr:
                        arr.append(label)
                r = as_text(row.get("data_rights")).lower()
                if r:
                    rights.setdefault(pid, set()).add(r)
            frame["science_targets"] = frame["proposal_id"].map(lambda p: ", ".join(targets.get(as_text(p), [])[:12]))
            frame["target_name"] = frame.apply(lambda r: (targets.get(as_text(r["proposal_id"])) or [as_text(r.get("target_name"))])[0], axis=1)
            frame["band_list"] = frame["proposal_id"].map(lambda p: " ".join(sorted(bands.get(as_text(p), set()), key=lambda b: (not b.isdigit(), int(b) if b.isdigit() else 0))))
            frame["arrays"] = frame["proposal_id"].map(lambda p: ", ".join(arrays.get(as_text(p), [])))
            frame["data_rights"] = frame["proposal_id"].map(lambda p: "/".join(sorted(rights.get(as_text(p), set()))))
        frame = frame.sort_values(["n_mous", "proposal_id"], ascending=[False, True]).reset_index(drop=True)
    return ServerSideResult(
        frame=frame, queries=queries, complete=not _truncated(df), scanned=f"Cycle {int(cycle)}",
        n_units=int(len(frame)), unit="projects", notes=notes, elapsed_s=time.perf_counter() - t0,
        extras={"method": "server-side GROUP BY proposal_id; science scans only" if science_only else "server-side GROUP BY proposal_id"},
    )


# ── 2. Cycle array-combination projects (D11) ────────────────────────────

_ARRAY_PATTERNS: Dict[str, str] = {
    # antenna_arrays is 'Pad:Antenna Pad:Antenna ...'; DV/DA = 12-m, CM = 7-m, PM = Total Power.
    "12m": "(antenna_arrays LIKE '%:DV%' OR antenna_arrays LIKE '%:DA%' OR schedblock_name LIKE '%_TM1' OR schedblock_name LIKE '%_TM2')",
    "7m": "(antenna_arrays LIKE '%:CM%' OR schedblock_name LIKE '%_7M')",
    "TP": "(antenna_arrays LIKE '%:PM%' OR schedblock_name LIKE '%_TP')",
}


def _normalize_array(label: str) -> str:
    text = str(label or "").strip().upper().replace("TOTAL POWER", "TP").replace("-M", "M").replace(" ", "")
    if text in ("12M", "12"):
        return "12m"
    if text in ("7M", "7", "ACA"):
        return "7m"
    if text in ("TP", "TOTALPOWER", "SD"):
        return "TP"
    return text


def array_combo_projects_server_side(
    run_query: RunQuery, cycle: int, arrays: Sequence[str], *, science_only: bool = True, extra_where: str = ""
) -> ServerSideResult:
    """Projects of one cycle that used EVERY requested array: one DISTINCT
    proposal_id query per array (antenna-name prefixes are the archive's own
    array signature), intersected, plus one per-project aggregate for the
    counts. Complete for the cycle unless a page truncated."""
    t0 = time.perf_counter()
    wanted = [_normalize_array(a) for a in (arrays or ["12m", "7m", "TP"])]
    unknown = [a for a in wanted if a not in _ARRAY_PATTERNS]
    if unknown:
        raise ValueError(f"unknown array label(s): {', '.join(unknown)}; use 12m, 7m, TP")
    cyc = project_prefix_where(int(cycle))
    sci = _science_where(science_only) + (f" AND {extra_where}" if extra_where else "")
    per_array_queries = {
        a: f"SELECT DISTINCT proposal_id FROM ivoa.obscore WHERE {cyc} AND {sci} AND {_ARRAY_PATTERNS[a]}" for a in wanted
    }
    agg = (
        "SELECT proposal_id, COUNT(DISTINCT member_ous_uid) AS n_mous, COUNT(DISTINCT asdm_uid) AS n_eb, COUNT(*) AS n_rows, "
        "MIN(target_name) AS target_name, MIN(pi_name) AS pi_name, MIN(obs_title) AS obs_title, MIN(band_list) AS band_list "
        f"FROM ivoa.obscore WHERE {cyc} AND {sci} GROUP BY proposal_id"
    )
    fns = [(lambda q=q: run_query(q)) for q in per_array_queries.values()] + [lambda: run_query(agg)]
    outputs = run_concurrently(fns, wall_seconds=150.0)
    queries = list(per_array_queries.values()) + [agg]
    sets: Dict[str, Set[str]] = {}
    truncated = False
    failures: List[str] = []
    for a, out in zip(wanted, outputs[:-1]):
        if isinstance(out, BaseException):
            failures.append(f"{a}: {type(out).__name__}: {str(out)[:160]}")
            continue
        sets[a] = set(out["proposal_id"].map(as_text)) if out is not None and not out.empty else set()
        truncated |= _truncated(out)
    if failures:
        raise RuntimeError("array-membership query failed: " + "; ".join(failures))
    agg_out = outputs[-1]
    if isinstance(agg_out, BaseException):
        raise RuntimeError(f"per-project aggregate failed: {type(agg_out).__name__}: {str(agg_out)[:160]}")
    truncated |= _truncated(agg_out)
    matched = set.intersection(*sets.values()) if sets else set()
    frame = pd.DataFrame(columns=["proposal_id", "arrays_found", "n_mous", "n_eb", "n_rows", "target_name", "pi_name", "obs_title"])
    if matched:
        base = agg_out[agg_out["proposal_id"].map(as_text).isin(matched)].copy() if agg_out is not None and not agg_out.empty else pd.DataFrame({"proposal_id": sorted(matched)})
        base["arrays_found"] = ", ".join(wanted)
        base["array_inference"] = "server-side: antenna names DV/DA = 12 m, CM = 7 m, PM = Total Power (+ SB-name suffix)"
        frame = base.sort_values(["proposal_id"]).reset_index(drop=True)
    per_array_counts = {a: len(s) for a, s in sets.items()}
    n_cycle_projects = int(len(agg_out)) if agg_out is not None else 0
    notes = [
        cycle_periods_disclosure(int(cycle)),
        "Array membership from the archive's antenna names (DV/DA = 12-m, CM = 7-m, PM = Total Power); "
        "the archive delivers per-MOUS data and has not combined the arrays.",
        f"Projects in the cycle window: {n_cycle_projects}; using each array: "
        + ", ".join(f"{a}={n}" for a, n in per_array_counts.items()) + f"; using ALL of {', '.join(wanted)}: {len(matched)}.",
    ]
    return ServerSideResult(
        frame=frame, queries=queries, complete=not truncated, scanned=f"Cycle {int(cycle)}", n_units=n_cycle_projects,
        unit="projects", notes=notes, elapsed_s=time.perf_counter() - t0,
        extras={"projects_in_window": n_cycle_projects, "per_array_project_counts": per_array_counts,
                "matching_projects": len(matched), "method": "server-side DISTINCT per array + GROUP BY proposal_id"},
    )


# ── 3. Line-set projects (D21) ────────────────────────────────────────────

def line_set_projects_server_side(
    run_query: RunQuery,
    lines: Sequence[str],
    *,
    band: Any = None,
    science_only: bool = True,
    cycle: Optional[int] = None,
    topic_filter: str = "",
    extra_where: str = "",
) -> ServerSideResult:
    """MOUS whose spectral windows cover EVERY requested rest-frame line
    (server-side per line: ``frequency +/- bandwidth/2`` on the SPW rows,
    ``GROUP BY member_ous_uid``), intersected, then aggregated per project.
    Rest frequencies are returned so the answer can state them."""
    t0 = time.perf_counter()
    names = line_names_for_input(lines)
    if not names:
        raise ValueError("lines must contain recognised molecular lines (e.g. 12CO, 13CO, C18O)")
    parts = [_science_where(science_only), _band_where(band)]
    if extra_where:
        parts.append(extra_where)
    if cycle is not None:
        parts.append(project_prefix_where(int(cycle)))
    topic = str(topic_filter or "").strip()
    if topic:
        safe_topic = escape_adql(topic.lower())
        parts.append(
            f"(LOWER(science_keyword) LIKE '%{safe_topic}%' OR LOWER(scientific_category) LIKE '%{safe_topic}%' "
            f"OR LOWER(obs_title) LIKE '%{safe_topic}%')"
        )
    base_where = " AND ".join(p for p in parts if p and p != "1=1") or "1=1"
    per_line: Dict[str, str] = {}
    for name in names:
        nu = LINE_REST_FREQ_GHZ[name]
        per_line[name] = (
            "SELECT member_ous_uid, proposal_id, MIN(target_name) AS target_name, MIN(band_list) AS band_list, "
            "MIN(pi_name) AS pi_name, MIN(obs_title) AS obs_title, MIN(science_keyword) AS science_keyword, "
            "COUNT(DISTINCT frequency) AS n_spw_hits "
            f"FROM ivoa.obscore WHERE {base_where} AND {coverage_where(nu)} GROUP BY member_ous_uid, proposal_id"
        )
    outputs = run_concurrently([(lambda q=q: run_query(q)) for q in per_line.values()], wall_seconds=150.0)
    frames: Dict[str, pd.DataFrame] = {}
    truncated = False
    failures = []
    for name, out in zip(names, outputs):
        if isinstance(out, BaseException):
            failures.append(f"{name}: {type(out).__name__}: {str(out)[:160]}")
            continue
        frames[name] = out if out is not None else pd.DataFrame()
        truncated |= _truncated(out)
    if failures:
        raise RuntimeError("line-coverage query failed: " + "; ".join(failures))
    mous_sets = {name: set(df["member_ous_uid"].map(as_text)) if not df.empty else set() for name, df in frames.items()}
    covered_all = set.intersection(*mous_sets.values()) if mous_sets else set()
    rows: List[Dict[str, Any]] = []
    if covered_all:
        merged = pd.concat([df for df in frames.values() if not df.empty], ignore_index=True)
        merged["member_ous_uid"] = merged["member_ous_uid"].map(as_text)
        merged = merged[merged["member_ous_uid"].isin(covered_all)]
        for pid, group in merged.groupby(merged["proposal_id"].map(as_text)):
            mous = sorted(set(group["member_ous_uid"]))
            rows.append({
                "proposal_id": pid,
                "n_mous_covering_all_lines": len(mous),
                "covered_lines": ", ".join(names),
                "rest_frequencies_ghz": ", ".join(f"{n} {LINE_REST_FREQ_GHZ[n]:.3f}" for n in names),
                "target_name": first_nonempty(group["target_name"].tolist()) if "target_name" in group else "",
                "targets": ", ".join(sorted({as_text(t) for t in group.get("target_name", pd.Series(dtype=str)) if as_text(t)})[:10]),
                "band_list": first_nonempty(group["band_list"].tolist()) if "band_list" in group else "",
                "pi_name": first_nonempty(group["pi_name"].tolist()) if "pi_name" in group else "",
                "obs_title": first_nonempty(group["obs_title"].tolist()) if "obs_title" in group else "",
                "science_keyword": first_nonempty(group["science_keyword"].tolist()) if "science_keyword" in group else "",
                "member_ous_uids": "; ".join(mous[:6]) + (" …" if len(mous) > 6 else ""),
                "coverage_basis": "all requested lines within the SAME MOUS (server-side SPW coverage)",
            })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["n_mous_covering_all_lines", "proposal_id"], ascending=[False, True]).reset_index(drop=True)
    per_line_counts = {name: len(s) for name, s in mous_sets.items()}
    notes = [
        "Coverage is computed per spectral window at the server (frequency ± bandwidth/2 contains the rest frequency); "
        "a MOUS must cover every requested line to count.",
        "MOUS covering each line: " + ", ".join(f"{n}={c}" for n, c in per_line_counts.items()) + f"; all lines: {len(covered_all)}.",
    ]
    if cycle is not None:
        notes.append(cycle_periods_disclosure(int(cycle)))
    return ServerSideResult(
        frame=frame, queries=list(per_line.values()), complete=not truncated,
        scanned=(f"Cycle {int(cycle)}" if cycle is not None else "whole archive"), n_units=int(len(covered_all)),
        unit="MOUS covering all lines", notes=notes, elapsed_s=time.perf_counter() - t0,
        extras={
            "lines": names,
            "rest_frequencies_ghz": {n: LINE_REST_FREQ_GHZ[n] for n in names},
            "mous_per_line": per_line_counts,
            "method": "server-side GROUP BY member_ous_uid per line; intersection; per-project aggregation",
        },
    )


# ── 4. Redshifted-line projects (D22) ────────────────────────────────────

def redshift_windows(rest_species: Sequence[str] | str, z_min: float, z_max: float) -> List[Dict[str, Any]]:
    z_lo, z_hi = sorted((float(z_min), float(z_max)))
    if not all(math.isfinite(z) and z > -1 for z in (z_lo, z_hi)):
        raise ValueError("Redshift bounds must be finite and greater than -1")
    out = []
    for name in line_names_for_species(rest_species):
        nu = LINE_REST_FREQ_GHZ[name]
        out.append({"transition": name, "rest_frequency_ghz": nu, "observed_min_ghz": nu / (1 + z_hi), "observed_max_ghz": nu / (1 + z_lo)})
    return out


def _extragalactic_where(science_category: str) -> str:
    if science_category:
        safe = escape_adql(str(science_category).lower())
        return f"LOWER(scientific_category) LIKE '%{safe}%'"
    return (
        "(LOWER(scientific_category) LIKE '%galax%' OR LOWER(scientific_category) LIKE '%cosmolog%' "
        "OR LOWER(scientific_category) LIKE '%active%' OR LOWER(science_keyword) LIKE '%galax%' "
        "OR LOWER(science_keyword) LIKE '%agn%' OR LOWER(science_keyword) LIKE '%quasar%') "
        "AND (scientific_category IS NULL OR LOWER(scientific_category) NOT LIKE '%solar system%')"
    )


def redshifted_line_projects_server_side(
    run_query: RunQuery,
    *,
    rest_species: Sequence[str] | str = "CO",
    z_min: float,
    z_max: float,
    science_category: str = "",
    cycle: Optional[int] = None,
    detail_projects: int = 40,
    min_seconds_per_cycle: float = 12.0,
    extra_where: str = "",
    skip_category: bool = False,
) -> ServerSideResult:
    """Per-project counts of SCIENCE observations whose SPWs overlap the
    observed window of any requested transition for z in [z_min, z_max] --
    one ``GROUP BY proposal_id`` aggregate per cycle (newest first; ~1 s each
    live) until every cycle is scanned or the tool budget runs low, then a
    bounded detail pull for the top projects that says WHICH transition and
    which coverage-compatible z range matched. Never a target redshift."""
    t0 = time.perf_counter()
    windows = redshift_windows(rest_species, z_min, z_max)
    win_where = "(" + " OR ".join(window_where(w["observed_min_ghz"], w["observed_max_ghz"]) for w in windows) + ")"
    # A cone on one named target (extra_where) makes the extragalactic
    # category filter unnecessary and risky (a category mislabel drops the
    # target's own projects), so callers may skip it.
    category = "" if skip_category and not science_category else f"{_extragalactic_where(science_category)} AND "
    base = f"science_observation = 'T' AND {category}{win_where}" + (f" AND {extra_where}" if extra_where else "")
    cycles = [int(cycle)] if cycle is not None else all_cycles(newest_first=True)

    def _cycle_query(c: int) -> str:
        return (
            "SELECT proposal_id, COUNT(DISTINCT member_ous_uid) AS n_mous, COUNT(DISTINCT asdm_uid) AS n_eb, COUNT(*) AS n_rows, "
            "MIN(target_name) AS target_name, MIN(pi_name) AS pi_name, MIN(obs_title) AS obs_title, "
            "MIN(scientific_category) AS scientific_category, MIN(science_keyword) AS science_keyword, MIN(band_list) AS band_list "
            f"FROM ivoa.obscore WHERE {project_prefix_where(c)} AND {base} GROUP BY proposal_id"
        )

    queries, frames, scanned, unscanned, truncated = _scan_cycles(
        run_query, cycles, _cycle_query, min_seconds_per_cycle=min_seconds_per_cycle
    )
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["proposal_id", "n_mous", "n_eb", "n_rows", "target_name", "pi_name", "obs_title", "cycle"]
    )
    if not frame.empty:
        frame = frame.sort_values(["n_mous", "proposal_id"], ascending=[False, True]).reset_index(drop=True)
        # Detail: which transition / coverage-compatible z range per project.
        top = frame["proposal_id"].map(as_text).tolist()[:max(1, int(detail_projects))]
        detail_q = (
            "SELECT proposal_id, member_ous_uid, target_name, frequency, bandwidth, band_list "
            f"FROM ivoa.obscore WHERE proposal_id IN ({_in_list(top)}) AND {base}"
        )
        queries.append(detail_q)
        transitions: Dict[str, Set[str]] = {}
        # (project, transition) -> [z_lo, z_hi] envelope over every matching SPW
        zenv: Dict[Tuple[str, str], List[float]] = {}
        targets: Dict[str, Set[str]] = {}
        try:
            det = run_query(detail_q)
        except Exception as exc:
            det = pd.DataFrame()
            queries.append(f"-- detail query failed ({type(exc).__name__}); transitions not annotated")
        z_req_lo, z_req_hi = min(z_min, z_max), max(z_min, z_max)
        if det is not None and not det.empty:
            for _, row in det.iterrows():
                pid = as_text(row.get("proposal_id"))
                try:
                    f = float(row.get("frequency"))
                    bw = float(row.get("bandwidth") or 0.0)
                except (TypeError, ValueError):
                    continue
                bw_ghz = bw / 1e9 if bw > 1e5 else bw
                lo, hi = f - bw_ghz / 2.0, f + bw_ghz / 2.0
                for w in windows:
                    inter_lo, inter_hi = max(lo, w["observed_min_ghz"]), min(hi, w["observed_max_ghz"])
                    if inter_lo > inter_hi:
                        continue
                    nu = w["rest_frequency_ghz"]
                    z_a, z_b = max(nu / inter_hi - 1.0, z_req_lo), min(nu / inter_lo - 1.0, z_req_hi)
                    transitions.setdefault(pid, set()).add(w["transition"])
                    env = zenv.setdefault((pid, w["transition"]), [z_a, z_b])
                    env[0], env[1] = min(env[0], z_a), max(env[1], z_b)
                t = as_text(row.get("target_name"))
                if t:
                    targets.setdefault(pid, set()).add(t)

        def _zrange_text(p: Any) -> str:
            pid = as_text(p)
            parts = [f"{tr}: z {env[0]:.2f}–{env[1]:.2f}" for (pp, tr), env in sorted(zenv.items()) if pp == pid]
            return "; ".join(parts)

        frame["transitions"] = frame["proposal_id"].map(lambda p: ", ".join(sorted(transitions.get(as_text(p), set()))))
        frame["coverage_compatible_z_range"] = frame["proposal_id"].map(_zrange_text)
        # A spectral setup is coverage-compatible with high-z CO whenever a
        # lower-J CO line of a NEARBY galaxy falls in the same window (PHANGS
        # CO(2-1) at 230 GHz is "CO(4-3) at z≈1"). Say which category the
        # proposal itself declared so the answer does not present Local
        # Universe surveys as z=1-2 galaxies (D22).
        def _category_hint(cat: Any) -> str:
            c = as_text(cat).lower()
            if "local universe" in c or "ism" in c or "star formation" in c:
                return "proposal category is nearby-Universe science: coverage-compatible, but an unlikely z=1-2 target"
            if any(k in c for k in ("cosmolog", "high-z", "high redshift", "galaxy evolution", "active")):
                return "proposal category is consistent with distant-galaxy science"
            return ""

        if "scientific_category" in frame.columns:
            frame["target_category_hint"] = frame["scientific_category"].map(_category_hint)
        frame["science_targets"] = frame["proposal_id"].map(lambda p: ", ".join(sorted(targets.get(as_text(p), set()))[:8]))
        frame["redshift_interpretation"] = (
            "coverage-compatible z range only (which redshift the SPW setup COULD detect the line at); "
            "ObsCore carries no target redshift"
        )
    scanned_label = (f"Cycle {cycles[0]}" if cycle is not None else
                     (f"cycles {min(scanned)}–{max(scanned)}" if scanned else "no cycle"))
    notes = [
        "Science observations only (science_observation = 'T'; calibrators, planets and other calibration scans excluded); "
        "extragalactic categories/keywords unless science_category is given.",
        "Windows: " + "; ".join(f"{w['transition']} rest {w['rest_frequency_ghz']:.3f} GHz → observed {w['observed_min_ghz']:.2f}–{w['observed_max_ghz']:.2f} GHz" for w in windows),
        "The z range per project is COVERAGE-COMPATIBLE, not a measured redshift.",
    ]
    return ServerSideResult(
        frame=frame, queries=queries, complete=(not truncated and not unscanned),
        scanned=scanned_label, unscanned=[f"Cycle {c}" for c in unscanned], n_units=int(len(frame)), unit="projects",
        notes=notes, elapsed_s=time.perf_counter() - t0,
        extras={"windows": windows, "cycles_scanned": scanned, "method": "server-side GROUP BY proposal_id per cycle + bounded detail pull"},
    )


# ── 5. Bandwidth-switching candidates (D09) ─────────────────────────────

def _aggregate_bandwidth_mhz(frequency_support: Any, fallback_max_bw_hz: Any, n_spw: Any) -> Optional[float]:
    intervals = parse_frequency_support_intervals(frequency_support)
    if intervals:
        seen = set()
        total = 0.0
        for lo, hi in intervals:
            key = (round(lo, 4), round(hi, 4))
            if key in seen:
                continue
            seen.add(key)
            total += max(0.0, hi - lo)
        return total * 1e3  # GHz -> MHz
    try:
        max_bw = float(fallback_max_bw_hz)
        n = int(n_spw or 1)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(max_bw):
        return None
    # Upper bound when the SPW list is not parseable: every SPW at most max_bw wide.
    return max_bw * n / 1e6


def bandwidth_switching_server_side(
    run_query: RunQuery,
    *,
    cycle: Optional[int] = None,
    threshold_mhz: float = BWSW_THRESHOLD_MHZ,
    min_seconds_per_cycle: float = 8.0,
    extra_where: str = "",
) -> ServerSideResult:
    """MOUS whose science spectral setup is narrower than the Handbook BWSW
    threshold: per cycle a ``GROUP BY member_ous_uid HAVING MAX(bandwidth) <
    threshold`` pre-filter at the server (no SPW wider than the threshold can
    belong to a narrow setup), then the exact aggregate bandwidth from the
    MOUS's spectral-window list; aggregated per project."""
    t0 = time.perf_counter()
    threshold_hz = float(threshold_mhz) * 1e6
    cycles = [int(cycle)] if cycle is not None else all_cycles(newest_first=True)

    # Numeric aggregates only: string MIN() columns made the per-cycle HAVING
    # scan ~4x slower live (9.3 s vs 2.5 s, Cycle 11); the candidate MOUS are
    # few, so their text metadata comes from a bounded second query.
    def _cycle_query(c: int) -> str:
        return (
            "SELECT member_ous_uid, proposal_id, COUNT(DISTINCT frequency) AS n_spw, MAX(bandwidth) AS max_bw_hz "
            f"FROM ivoa.obscore WHERE {project_prefix_where(c)} AND science_observation = 'T'" + (f" AND {extra_where}" if extra_where else "") + " "
            f"GROUP BY member_ous_uid, proposal_id HAVING MAX(bandwidth) < {threshold_hz:.0f}"
        )

    queries, frames, scanned, unscanned, truncated = _scan_cycles(
        run_query, cycles, _cycle_query, min_seconds_per_cycle=min_seconds_per_cycle
    )
    mous = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not mous.empty:
        # Text metadata + the SPW list for the candidate MOUS only.
        mous["member_ous_uid"] = mous["member_ous_uid"].map(as_text)
        detail_frames = []
        uids = mous["member_ous_uid"].tolist()
        chunks = [uids[i:i + 150] for i in range(0, len(uids), 150)]
        detail_queries = [
            "SELECT DISTINCT member_ous_uid, target_name, band_list, pi_name, obs_title, frequency_support "
            f"FROM ivoa.obscore WHERE member_ous_uid IN ({_in_list(chunk)}) AND science_observation = 'T'"
            for chunk in chunks
        ]
        queries.extend(detail_queries)
        for start in range(0, len(detail_queries), 4):
            batch_q = detail_queries[start:start + 4]
            if start and not _budget_allows(min_seconds_per_cycle):
                queries.append(f"-- detail metadata skipped for {sum(len(c) for c in chunks[start:])} MOUS (budget)")
                break
            outs = run_concurrently([(lambda q=q: run_query(q)) for q in batch_q], wall_seconds=120.0)
            for q, out in zip(batch_q, outs):
                if isinstance(out, BaseException):
                    queries.append(f"-- detail chunk failed ({type(out).__name__}); text metadata missing for some MOUS")
                elif out is not None and not out.empty:
                    detail_frames.append(out)
        if detail_frames:
            det = pd.concat(detail_frames, ignore_index=True)
            det["member_ous_uid"] = det["member_ous_uid"].map(as_text)
            agg = det.groupby("member_ous_uid").agg(
                target_name=("target_name", lambda s: first_nonempty(s.tolist())),
                band_list=("band_list", lambda s: " ".join(sorted({t for v in s for t in band_tokens(v)}))),
                pi_name=("pi_name", lambda s: first_nonempty(s.tolist())),
                obs_title=("obs_title", lambda s: first_nonempty(s.tolist())),
                frequency_support=("frequency_support", lambda s: first_nonempty(s.tolist())),
            ).reset_index()
            mous = mous.merge(agg, on="member_ous_uid", how="left")
        for col_name in ("target_name", "band_list", "pi_name", "obs_title", "frequency_support"):
            if col_name not in mous.columns:
                mous[col_name] = ""
    rows: List[Dict[str, Any]] = []
    n_candidate_mous = 0
    if not mous.empty:
        mous["aggregate_bandwidth_mhz"] = mous.apply(
            lambda r: _aggregate_bandwidth_mhz(r.get("frequency_support"), r.get("max_bw_hz"), r.get("n_spw")), axis=1
        )
        cand = mous[mous["aggregate_bandwidth_mhz"].notna() & (mous["aggregate_bandwidth_mhz"] < float(threshold_mhz))].copy()
        n_candidate_mous = int(len(cand))
        for pid, group in cand.groupby(cand["proposal_id"].map(as_text)):
            rows.append({
                "proposal_id": pid,
                "cycle": _agg_int(group["cycle"].iloc[0]),
                "n_mous_narrow_setup": int(len(group)),
                "min_aggregate_bandwidth_mhz": round(float(group["aggregate_bandwidth_mhz"].min()), 1),
                "max_spw_bandwidth_mhz": round(float(pd.to_numeric(group["max_bw_hz"], errors="coerce").max()) / 1e6, 1),
                "n_spw": _agg_int(group["n_spw"].max()),
                "band_list": " ".join(sorted({t for v in group["band_list"] for t in band_tokens(v)}, key=lambda b: (not b.isdigit(), int(b) if b.isdigit() else 0))),
                "target_name": first_nonempty(group["target_name"].tolist()),
                "pi_name": first_nonempty(group["pi_name"].tolist()),
                "obs_title": first_nonempty(group["obs_title"].tolist()),
                "criterion": f"aggregate science-SPW bandwidth < {threshold_mhz:g} MHz (narrow FDM-only setup) → BWSW likely needed for phase calibration",
                "member_ous_uids": "; ".join(sorted(group["member_ous_uid"].map(as_text))[:5]),
            })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["min_aggregate_bandwidth_mhz", "n_mous_narrow_setup", "proposal_id"], ascending=[True, False, True]).reset_index(drop=True)
    scanned_label = (f"Cycle {cycles[0]}" if cycle is not None else
                     (f"cycles {min(scanned)}–{max(scanned)}" if scanned else "no cycle"))
    notes = [
        f"Criterion: {BWSW_CITATION}",
        "Pre-filter at the server: MOUS with no science SPW wider than the threshold (HAVING MAX(bandwidth) < threshold); "
        "exact aggregate bandwidth from the MOUS spectral-window list; calibration intent itself is not recorded in ObsCore.",
    ]
    return ServerSideResult(
        frame=frame, queries=queries, complete=(not truncated and not unscanned), scanned=scanned_label,
        unscanned=[f"Cycle {c}" for c in unscanned], n_units=n_candidate_mous, unit="candidate MOUS", notes=notes,
        elapsed_s=time.perf_counter() - t0,
        extras={"threshold_mhz": float(threshold_mhz), "cycles_scanned": scanned, "citation": BWSW_CITATION,
                "method": "server-side GROUP BY member_ous_uid HAVING MAX(bandwidth) < threshold, exact aggregate client-side"},
    )
