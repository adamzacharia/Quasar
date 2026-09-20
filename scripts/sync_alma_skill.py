#!/usr/bin/env python3
"""
sync_alma_skill.py — vendor / refresh the "Working with ALMA data" skill.

The skill (SKILL.md + references/*.md + review/claims.tsv, reviewed
2026-07-18; author confirmed free to use) is vendored UNCHANGED under
``third_party/alma-data-skill/`` with an ``UPSTREAM.json`` (source, revision,
review date, vendored date, per-file SHA-256). Application-specific summaries
(the prompt kernel, the archive profile) live OUTSIDE the vendored folder and
are regenerated from it.

Usage
-----
    python scripts/sync_alma_skill.py --write-manifest          # (re)write UPSTREAM.json for the vendored tree
    python scripts/sync_alma_skill.py --check                   # exit 1 if any vendored file drifted from UPSTREAM.json
    python scripts/sync_alma_skill.py --from <upstream-dir>     # diff upstream vs vendored, copy, rewrite manifest,
                                                                # and list changed claims.tsv rows for human review

After a sync, run the ALMA unit tests and the profile validation:
    .venv/Scripts/python -m pytest tests/unit/test_alma_skill_claims.py tests/unit/test_archive_profiles.py -q
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENDORED = REPO_ROOT / "third_party" / "alma-data-skill"
MANIFEST = VENDORED / "UPSTREAM.json"
VENDORED_FILES = ("SKILL.md", "agents/openai.yaml", "review/claims.tsv")
VENDORED_GLOBS = ("references/*.md", "review/*.md", "review/*.tsv")


def _files(root: Path) -> list:
    out = set()
    for rel in VENDORED_FILES:
        if (root / rel).exists():
            out.add(rel)
    for pattern in VENDORED_GLOBS:
        for path in root.glob(pattern):
            out.add(path.relative_to(root).as_posix())
    return sorted(out)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _review_date(root: Path) -> str:
    text = (root / "SKILL.md").read_text(encoding="utf-8")
    import re

    match = re.search(r"reviewed\s+(\d{4}-\d{2}-\d{2})", text)
    return match.group(1) if match else ""


def write_manifest(source: str = "", revision: str = "") -> dict:
    existing = {}
    if MANIFEST.exists():
        try:
            existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    manifest = {
        "name": "working-with-alma-data",
        "source": source or existing.get("source") or "alma-data-skill-main (author drop, 2026-09; author confirmed free to use)",
        "revision": revision or existing.get("revision") or "2026-07-18-review",
        "review_date": _review_date(VENDORED),
        "vendored_on": dt.date.today().isoformat(),
        "license_note": "Author confirmed the skill is free to use; no LICENSE file ships upstream.",
        "vendored_unchanged": True,
        "files": {rel: _sha256(VENDORED / rel) for rel in _files(VENDORED)},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def check() -> int:
    if not MANIFEST.exists():
        print("UPSTREAM.json missing; run --write-manifest", file=sys.stderr)
        return 1
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    drift = []
    for rel, digest in manifest.get("files", {}).items():
        path = VENDORED / rel
        if not path.exists():
            drift.append(f"missing: {rel}")
        elif _sha256(path) != digest:
            drift.append(f"changed: {rel}")
    for rel in _files(VENDORED):
        if rel not in manifest.get("files", {}):
            drift.append(f"unlisted: {rel}")
    if drift:
        print("vendored skill drifted from UPSTREAM.json:\n  " + "\n  ".join(drift), file=sys.stderr)
        return 1
    print(f"vendored skill matches UPSTREAM.json ({len(manifest.get('files', {}))} files, review {manifest.get('review_date')})")
    return 0


def _claims(path: Path) -> dict:
    rows = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            rows[row.get("id", "")] = row
    return rows


def sync_from(upstream: Path) -> int:
    if not (upstream / "SKILL.md").exists():
        print(f"{upstream} has no SKILL.md", file=sys.stderr)
        return 1
    old_claims = _claims(VENDORED / "review" / "claims.tsv")
    changed, added, removed = [], [], []
    for rel in _files(upstream):
        src, dst = upstream / rel, VENDORED / rel
        if not dst.exists():
            added.append(rel)
        elif _sha256(src) != _sha256(dst):
            changed.append(rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    for rel in _files(VENDORED):
        if not (upstream / rel).exists():
            removed.append(rel)
            (VENDORED / rel).unlink()
    write_manifest(source=str(upstream))
    new_claims = _claims(VENDORED / "review" / "claims.tsv")
    print(f"synced from {upstream}: {len(changed)} changed, {len(added)} added, {len(removed)} removed")
    for rel in changed:
        print(f"  changed: {rel}")
    claim_deltas = []
    for cid, row in new_claims.items():
        if cid not in old_claims:
            claim_deltas.append(f"  NEW   {cid}: {row.get('claim_or_guardrail', '')[:100]}")
        elif row != old_claims[cid]:
            claim_deltas.append(f"  EDIT  {cid}: {row.get('claim_or_guardrail', '')[:100]}")
    for cid in old_claims:
        if cid not in new_claims:
            claim_deltas.append(f"  GONE  {cid}")
    if claim_deltas:
        print("claims.tsv rows needing human review:")
        print("\n".join(claim_deltas))
    print("Now run: python scripts/gen_alma_kernel.py && pytest tests/unit/test_alma_skill_claims.py tests/unit/test_archive_profiles.py -q")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--from", dest="upstream", default=None)
    args = parser.parse_args()
    if args.upstream:
        return sync_from(Path(args.upstream))
    if args.check:
        return check()
    if args.write_manifest:
        manifest = write_manifest()
        print(f"wrote {MANIFEST} ({len(manifest['files'])} files, review {manifest['review_date']})")
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
