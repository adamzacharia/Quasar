from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.moc_coverage import MocCoverageService


def _compact(result):
    if not result.get("success"):
        return result
    out = {key: result.get(key) for key in ("success", "count", "total_matches", "covered", "matched_count", "matches", "warnings") if key in result}
    rows = result.get("rows") or []
    if rows:
        out["sample_ids"] = [row.get("id") for row in rows[:5]]
    return out


def main() -> int:
    svc = MocCoverageService()
    coverage = svc.coverage_at(187.2779, 2.0524, 0.1)
    # NB: do NOT use 3C 273 for the VLASS positive check — the VLASS
    # Quicklook MedianStack MOC has a real hole at that position (bright-
    # source tiles excluded), verified live 2026-07-03. MOC answers are
    # exact, holes included.
    vlass_true = svc.survey_covers("VLASS", 180.0, 30.0)
    vlass_false = svc.survey_covers("VLASS", 10.0, -75.0)

    checks = [
        (
            "3C 273 coverage",
            coverage.get("success") is True
            and int(coverage.get("total_matches") or 0) > 100
            and any("CDS/" in str(row.get("id") or "") for row in coverage.get("rows", [])),
            coverage,
        ),
        ("VLASS covers (180,+30)", vlass_true.get("success") is True and vlass_true.get("covered") is True, vlass_true),
        ("VLASS does not cover Dec -75", vlass_false.get("success") is True and vlass_false.get("covered") is False, vlass_false),
    ]

    all_ok = True
    for label, ok, result in checks:
        all_ok = all_ok and ok
        status = "PASS" if ok else "FAIL"
        print(f"{status} {label}: {json.dumps(_compact(result), sort_keys=True)}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())