#!/usr/bin/env python3
"""
Render Log Fetcher — Pulls recent logs from your Render service.

Usage:
    python scripts/render_logs.py                  # last 30 minutes
    python scripts/render_logs.py --minutes 60     # last 60 minutes
    python scripts/render_logs.py --errors         # errors only
    python scripts/render_logs.py --tail           # live tail (polls every 5s)

Requires RENDER_API_KEY in your .env file.
On first run, it will auto-detect your service ID and cache it.
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Load .env from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env", override=True)
except ImportError:
    pass

API_KEY = os.getenv("RENDER_API_KEY", "")
BASE_URL = "https://api.render.com/v1"
CACHE_FILE = PROJECT_ROOT / ".render_service_cache.json"


def _headers():
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json",
    }


def _get(url, params=None):
    """Simple GET request using urllib (no extra dependencies)."""
    import urllib.request
    import urllib.parse
    if params:
        url += "?" + urllib.parse.urlencode(params, doseq=True)
    req = urllib.request.Request(url, headers=_headers())
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def discover_service():
    """Find the Quasar web service on Render."""
    if CACHE_FILE.exists():
        cache = json.loads(CACHE_FILE.read_text())
        if cache.get("service_id") and cache.get("owner_id"):
            return cache["service_id"], cache["owner_id"]

    print("[*] Discovering Render services...")
    data = _get(f"{BASE_URL}/services", {"type": ["web_service"], "limit": "20"})

    services = []
    for item in data:
        svc = item.get("service", item)
        sid = svc.get("id", "")
        name = svc.get("name", "")
        owner_id = svc.get("ownerId", "")
        services.append((sid, name, owner_id))

    if not services:
        print("[ERROR] No services found on your Render account.")
        sys.exit(1)

    # Try to auto-detect Quasar service
    quasar = None
    for sid, name, oid in services:
        if "quasar" in name.lower():
            quasar = (sid, name, oid)
            break

    if not quasar:
        print("Available services:")
        for i, (sid, name, oid) in enumerate(services):
            print(f"  [{i}] {name} ({sid})")
        idx = int(input("Select service number: "))
        quasar = services[idx]

    service_id, service_name, owner_id = quasar
    print(f"[*] Using service: {service_name} ({service_id})")

    # Cache for next time
    CACHE_FILE.write_text(json.dumps({
        "service_id": service_id,
        "service_name": service_name,
        "owner_id": owner_id,
    }))
    return service_id, owner_id


def fetch_logs(service_id, owner_id, minutes=30, errors_only=False):
    """Fetch recent logs from Render."""
    now = datetime.now(timezone.utc)
    start = now - timedelta(minutes=minutes)

    params = {
        "resource": [service_id],
        "ownerId": owner_id,
        "startTime": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "endTime": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "direction": "forward",
    }

    if errors_only:
        params["level"] = ["error", "critical"]

    try:
        data = _get(f"{BASE_URL}/logs", params)
    except Exception as e:
        print(f"[ERROR] Failed to fetch logs: {e}")
        return []

    logs = data if isinstance(data, list) else data.get("logs", [])
    return logs


def print_logs(logs):
    """Pretty-print log entries."""
    for entry in logs:
        ts = entry.get("timestamp", "")
        msg = entry.get("message", "")
        level = entry.get("level", "info")

        # Color based on level
        if level in ("error", "critical"):
            color = "\033[91m"  # red
        elif level == "warning":
            color = "\033[93m"  # yellow
        elif "TOOL CALL" in msg or "FILTER" in msg:
            color = "\033[96m"  # cyan
        else:
            color = "\033[90m"  # grey

        # Shorten timestamp
        short_ts = ts[11:19] if len(ts) > 19 else ts
        # Handle Windows console encoding
        try:
            print(f"{color}{short_ts} | {msg}\033[0m")
        except UnicodeEncodeError:
            safe_msg = msg.encode("ascii", errors="replace").decode()
            print(f"{color}{short_ts} | {safe_msg}\033[0m")


def tail_logs(service_id, owner_id, interval=5):
    """Continuously poll for new logs."""
    print(f"[*] Tailing logs (every {interval}s)... Press Ctrl+C to stop.\n")
    last_ts = datetime.now(timezone.utc) - timedelta(seconds=30)

    try:
        while True:
            now = datetime.now(timezone.utc)
            params = {
                "resource": [service_id],
                "ownerId": owner_id,
                "startTime": last_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "endTime": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "direction": "forward",
            }
            try:
                data = _get(f"{BASE_URL}/logs", params)
                logs = data if isinstance(data, list) else data.get("logs", [])
                if logs:
                    print_logs(logs)
                    last_ts = now
            except Exception as e:
                print(f"\033[91m[ERROR] {e}\033[0m")

            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[*] Stopped.")


def main():
    parser = argparse.ArgumentParser(description="Fetch Render logs for Quasar")
    parser.add_argument("--minutes", type=int, default=30, help="How many minutes of logs to fetch (default: 30)")
    parser.add_argument("--errors", action="store_true", help="Show only errors")
    parser.add_argument("--tail", action="store_true", help="Live tail mode (polls every 5s)")
    parser.add_argument("--raw", action="store_true", help="Print raw JSON")
    args = parser.parse_args()

    if not API_KEY:
        print("[ERROR] RENDER_API_KEY not set. Add it to your .env file:")
        print("  RENDER_API_KEY=rnd_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        sys.exit(1)

    service_id, owner_id = discover_service()

    if args.tail:
        tail_logs(service_id, owner_id)
        return

    print(f"[*] Fetching last {args.minutes} minutes of logs...\n")
    logs = fetch_logs(service_id, owner_id, args.minutes, args.errors)

    if not logs:
        print("[*] No logs found for this time period.")
        return

    if args.raw:
        print(json.dumps(logs, indent=2))
    else:
        print_logs(logs)
        print(f"\n[*] {len(logs)} log entries")


if __name__ == "__main__":
    main()
