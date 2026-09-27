"""
Inventory of the RAG vector collections (every non-user collection).

For each collection: point count, and per source_file the chunk count,
category, cycle, document date, status and ingest time range. Writes JSON
(machine diffable) and optionally a Markdown table.

Usage:
    python scripts/rag_inventory.py --json out.json --md out.md
    python scripts/rag_inventory.py --diff before.json after.json   # source-level diff
    python scripts/rag_inventory.py --delete-source alma_general Binder_badge.pdf
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def inventory() -> Dict[str, object]:
    from services.vector_db import backend_label, get_qdrant_client
    client = get_qdrant_client()
    out: Dict[str, object] = {"backend": backend_label(), "collections": {}}
    for c in sorted(col.name for col in client.get_collections().collections):
        if c.startswith("user_"):
            continue
        sources: Dict[str, Dict[str, object]] = {}
        cats: Dict[str, int] = defaultdict(int)
        offset, n = None, 0
        while True:
            points, offset = client.scroll(collection_name=c, limit=512, offset=offset,
                                           with_payload=True, with_vectors=False)
            for p in points:
                pay = p.payload or {}
                n += 1
                sf = str(pay.get("source_file") or pay.get("source") or "(none)")
                s = sources.setdefault(sf, {
                    "chunks": 0, "category": pay.get("doc_category", pay.get("type", "")),
                    "cycle": pay.get("alma_cycle", ""), "year": pay.get("doc_year"),
                    "month": pay.get("doc_month"), "status": pay.get("doc_status", ""),
                    "source_url": pay.get("source_url", ""), "ingested_min": None, "ingested_max": None,
                })
                s["chunks"] += 1
                ts = pay.get("ingested_at")
                if ts:
                    s["ingested_min"] = min(filter(None, [s["ingested_min"], ts]))
                    s["ingested_max"] = max(filter(None, [s["ingested_max"], ts]))
                cats[str(s["category"])] += 1
            if offset is None or not points:
                break
        out["collections"][c] = {"points": n, "sources": sources, "categories": dict(sorted(cats.items()))}
    return out


def to_markdown(inv: Dict[str, object]) -> str:
    lines = [f"Backend: `{inv['backend']}`", ""]
    for c, info in inv["collections"].items():
        lines += [f"## {c}: {info['points']} points, {len(info['sources'])} sources", "",
                  "Categories: " + ", ".join(f"{k} {v}" for k, v in info["categories"].items()), "",
                  "| Source | Chunks | Category | Cycle | Date | Status | Ingested |",
                  "|---|---:|---|---|---|---|---|"]
        for sf, s in sorted(info["sources"].items(), key=lambda kv: (str(kv[1]["category"]), kv[0])):
            date = f"{s['year']}-{s['month']:02d}" if isinstance(s.get("month"), int) else str(s.get("year") or "")
            ing = (s.get("ingested_max") or "")[:10]
            lines.append(f"| {sf} | {s['chunks']} | {s['category']} | {s['cycle'] or ''} | {date} | {s['status'] or ''} | {ing} |")
        lines.append("")
    return "\n".join(lines)


def diff(before: Dict[str, object], after: Dict[str, object]) -> str:
    lines = []
    for c in sorted(set(before["collections"]) | set(after["collections"])):
        b = before["collections"].get(c, {"points": 0, "sources": {}})
        a = after["collections"].get(c, {"points": 0, "sources": {}})
        lines.append(f"## {c}: {b['points']} -> {a['points']} points")
        bs, as_ = b["sources"], a["sources"]
        for sf in sorted(set(bs) - set(as_)):
            lines.append(f"- REMOVED {sf} ({bs[sf]['chunks']} chunks)")
        for sf in sorted(set(as_) - set(bs)):
            lines.append(f"- ADDED {sf} ({as_[sf]['chunks']} chunks, {as_[sf]['category']})")
        for sf in sorted(set(bs) & set(as_)):
            # Chunk count alone misses a refresh with the same count (CX-22):
            # compare the metadata and the ingest time too.
            changes = [
                f"{k} {bs[sf].get(k)!r} -> {as_[sf].get(k)!r}"
                for k in ("chunks", "category", "cycle", "year", "month", "status", "source_url", "ingested_max")
                if bs[sf].get(k) != as_[sf].get(k)
            ]
            if changes:
                lines.append(f"- CHANGED {sf}: " + "; ".join(changes))
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--md")
    ap.add_argument("--diff", nargs=2, metavar=("BEFORE", "AFTER"))
    ap.add_argument("--delete-source", nargs=2, metavar=("COLLECTION", "SOURCE_FILE"))
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    from dotenv import load_dotenv
    load_dotenv()
    if args.diff:
        with open(args.diff[0], encoding="utf-8") as f1, open(args.diff[1], encoding="utf-8") as f2:
            print(diff(json.load(f1), json.load(f2)))
        return
    if args.delete_source:
        from services.vector_db import delete_by_filter
        print(delete_by_filter(args.delete_source[0], {"source_file": args.delete_source[1]}))
        return
    inv = inventory()
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(inv, f, indent=1, default=str)
    md = to_markdown(inv)
    if args.md:
        with open(args.md, "w", encoding="utf-8") as f:
            f.write(md)
    for c, info in inv["collections"].items():
        print(f"{c}: {info['points']} points, {len(info['sources'])} sources; {info['categories']}")


if __name__ == "__main__":
    main()
