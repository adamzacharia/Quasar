"""Live audit of archive-profile pitfalls: is each curated claim still true?

Model-free. Runs one probe per pitfall that carries a ``PitfallAudit``
(services/archive_profiles/schema.py) against the real archive and reports:

  STILL-TRUE     the probe behaved as the claim says
  STALE          the archive answered, but not as claimed: fix the pitfall
  ENDPOINT-DEAD  the endpoint answered with an HTTP 404/410
  UNREACHABLE    network / timeout / 5xx: says nothing about the claim
  MANUAL         no single probe can check it (listed for hand review)

Exit code 1 when anything is STALE or ENDPOINT-DEAD; exit code 2 when the run
is incomplete: any UNREACHABLE probe, nothing verified at all, or an unknown
--archive. A (partial) outage never reads as green (cron/CI friendly).
Idea from MANNA's evals/audit.py (NSF-Simons CosmicAI, MIT).

Usage:
  python scripts/audit_archive_profiles.py --list
  python scripts/audit_archive_profiles.py [--archive gaia] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from services.archive_profiles import PROFILES  # noqa: E402
from services.archive_profiles.schema import ArchiveProfile, Pitfall  # noqa: E402

STILL_TRUE, STALE, DEAD, UNREACHABLE, MANUAL = (
    "STILL-TRUE", "STALE", "ENDPOINT-DEAD", "UNREACHABLE", "MANUAL")


@dataclass
class AuditRow:
    archive: str
    pitfall: str
    expect: str
    status: str
    detail: str
    seconds: float = 0.0


def columns_query(table: str, columns) -> str:
    inlist = ", ".join(f"'{c}'" for c in columns)
    return (f"SELECT column_name FROM tap_schema.columns "
            f"WHERE table_name = '{table}' AND column_name IN ({inlist})")


def _http_status(exc: BaseException) -> Optional[int]:
    seen = set()
    stack: List[Any] = [exc]
    while stack:
        cur = stack.pop()
        if cur is None or id(cur) in seen or not isinstance(cur, BaseException):
            continue
        seen.add(id(cur))
        status = getattr(getattr(cur, "response", None), "status_code", None)
        if status is not None:
            return int(status)
        stack.extend([getattr(cur, "cause", None), cur.__cause__, cur.__context__])
    return None


def _is_query_error(exc: BaseException) -> bool:
    """A server-side ADQL/query rejection (the archive answered), as opposed
    to a transport failure."""
    try:
        from pyvo.dal.exceptions import DALQueryError
    except Exception:  # pragma: no cover - pyvo is a runtime dependency
        DALQueryError = ()  # type: ignore[assignment]
    return isinstance(exc, DALQueryError) if DALQueryError else False


PROBE_WALL_SECONDS = 120.0


def default_runner(timeout: float = 60.0, wall_seconds: float = PROBE_WALL_SECONDS) -> Callable[[str, str], Any]:
    """(endpoint_url, adql) -> list of row dicts, through the same guarded,
    timeout-bound session the vo_* tools use. ``timeout`` bounds each socket
    read; ``wall_seconds`` bounds the whole probe, so a server that trickles
    bytes cannot stall the audit (guard CX-08)."""
    import threading

    import pyvo

    from services.vo_registry import _normalize_dal_result, _TimeoutHTTPSession

    def run(url: str, adql: str):
        box: dict = {}

        def _probe():
            try:
                svc = pyvo.dal.TAPService(url, session=_TimeoutHTTPSession(timeout))
                box["rows"] = _normalize_dal_result(svc.run_sync(adql, maxrec=10000))[0]
            except BaseException as exc:  # handed to the caller below
                box["exc"] = exc

        # A DAEMON thread, not a ThreadPoolExecutor worker: executor threads
        # are joined at interpreter exit, so one trickling probe could still
        # hang the process after the audit gave up on it (guard CX-08).
        worker = threading.Thread(target=_probe, name="audit-probe", daemon=True)
        worker.start()
        worker.join(wall_seconds)
        if worker.is_alive():
            raise TimeoutError(f"audit probe {url} exceeded {wall_seconds:g}s total; abandoned")
        if "exc" in box:
            raise box["exc"]
        return box.get("rows", [])

    return run


def audit_pitfall(profile: ArchiveProfile, pitfall: Pitfall,
                  run: Callable[[str, str], Any], *,
                  is_query_error: Callable[[BaseException], bool] = _is_query_error) -> AuditRow:
    audit = pitfall.audit
    assert audit is not None
    row = AuditRow(profile.archive, pitfall.id, audit.expect, MANUAL, audit.reason or "")
    if audit.expect == "manual":
        return row
    endpoint = next(e for e in profile.endpoints if e.id == audit.endpoint_id)
    url = str(endpoint.url).rstrip("/")
    if audit.expect == "columns_present":
        adql = columns_query(audit.table, audit.columns)
    elif audit.expect == "columns_absent":
        adql = columns_query(audit.table, audit.columns)
    elif audit.expect == "column_units":
        inlist = ", ".join(f"'{c}'" for c in audit.units)
        adql = (f"SELECT column_name, unit FROM tap_schema.columns "
                f"WHERE table_name = '{audit.table}' AND column_name IN ({inlist})")
    else:
        adql = audit.adql
    started = time.time()
    try:
        rows = run(url, adql)
        if audit.expect == "columns_absent":
            # Absent columns only count if the TABLE is still listed; checked
            # separately so a wide table's column list is never truncated
            # into a false "absent" (guard CX-16).
            table_rows = run(url, f"SELECT table_name FROM tap_schema.tables WHERE table_name = '{audit.table}'")
            if not table_rows:
                row.seconds = round(time.time() - started, 1)
                row.status, row.detail = STALE, f"table {audit.table!r} not found in TAP_SCHEMA"
                return row
    except Exception as exc:  # noqa: BLE001 - classify every failure
        row.seconds = round(time.time() - started, 1)
        message = str(exc)
        status = _http_status(exc)
        if status in (404, 410):
            row.status, row.detail = DEAD, f"HTTP {status} from {url}"
        elif audit.expect == "error" and is_query_error(exc):
            ok = audit.error_contains.lower() in message.lower()
            row.status = STILL_TRUE if ok else STALE
            row.detail = (f"failed as claimed ({audit.error_contains!r})" if ok else
                          f"failed, but without {audit.error_contains!r}: {message[:200]}")
        elif is_query_error(exc):
            row.status, row.detail = STALE, f"query rejected: {message[:200]}"
        else:
            row.status, row.detail = UNREACHABLE, f"{type(exc).__name__}: {message[:200]}"
        return row
    row.seconds = round(time.time() - started, 1)
    n = len(rows)
    if audit.expect == "error":
        row.status, row.detail = STALE, f"expected a query error, got {n} row(s)"
    elif audit.expect == "ok":
        row.status, row.detail = STILL_TRUE, f"query ran ({n} row(s))"
    elif audit.expect == "empty":
        row.status = STILL_TRUE if n == 0 else STALE
        row.detail = f"{n} row(s)"
    elif audit.expect == "nonempty":
        row.status = STILL_TRUE if n > 0 else STALE
        row.detail = f"{n} row(s)"
    elif audit.expect == "column_units":
        got = {str(r.get("column_name", "")).lower(): str(r.get("unit") or "") for r in rows}
        wrong = {c: got.get(c.lower(), "<missing>") for c, u in audit.units.items()
                 if got.get(c.lower()) != u}
        row.status = STALE if wrong else STILL_TRUE
        row.detail = (f"unit mismatch {wrong}" if wrong else f"all {len(audit.units)} units as claimed")
    else:
        found = {str(r.get("column_name", "")).lower() for r in rows}
        wanted = [c.lower() for c in audit.columns]
        if audit.expect == "columns_present":
            missing = [c for c in wanted if c not in found]
            row.status = STALE if missing else STILL_TRUE
            row.detail = f"missing {missing}" if missing else f"all {len(wanted)} present"
        else:
            present = [c for c in wanted if c in found]
            row.status = STALE if present else STILL_TRUE
            row.detail = (f"unexpectedly present {present}" if present
                          else "all absent (table listed)")
    return row


def run_audits(archives: Optional[List[str]] = None,
               run: Optional[Callable[[str, str], Any]] = None) -> List[AuditRow]:
    run = run or default_runner()
    rows: List[AuditRow] = []
    for slug in sorted(PROFILES):
        if archives and slug not in archives:
            continue
        profile = PROFILES[slug]
        for pitfall in profile.pitfalls:
            if pitfall.audit is not None:
                rows.append(audit_pitfall(profile, pitfall, run))
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--archive", action="append", help="limit to this archive slug (repeatable)")
    ap.add_argument("--list", action="store_true", help="list audited pitfalls; no probes")
    ap.add_argument("--json", help="also write the rows as JSON to this path")
    args = ap.parse_args(argv)

    if args.list:
        for slug in sorted(PROFILES):
            if args.archive and slug not in args.archive:
                continue
            for p in PROFILES[slug].pitfalls:
                kind = p.audit.expect if p.audit else "(no audit)"
                print(f"{slug:14s} {p.id:34s} {kind}")
        return 0

    unknown = sorted(set(args.archive or []) - set(PROFILES))
    if unknown:
        print(f"unknown archive(s): {unknown}; known: {sorted(PROFILES)}")
        return 2
    rows = run_audits(args.archive)
    for r in rows:
        print(f"{r.status:13s} {r.archive:10s} {r.pitfall:34s} {r.seconds:5.1f}s  {r.detail}")
    counts = {s: sum(r.status == s for r in rows) for s in (STILL_TRUE, STALE, DEAD, UNREACHABLE, MANUAL)}
    print("summary:", ", ".join(f"{k}={v}" for k, v in counts.items()))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump([asdict(r) for r in rows], fh, indent=2)
    if counts[STALE] or counts[DEAD]:
        return 1
    if counts[UNREACHABLE] or not counts[STILL_TRUE]:
        print("incomplete: some claims could not be probed (UNREACHABLE) or nothing was verified")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
