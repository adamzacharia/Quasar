import os
import sys
import json
import urllib.request
from dotenv import load_dotenv

# Load env vars
load_dotenv()

API_KEY = os.getenv("RENDER_API_KEY", "")
BASE_URL = "https://api.render.com/v1"

def check_deployments():
    if not API_KEY:
        print("RENDER_API_KEY not found!")
        return

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json",
    }
    
    # 1. Discover service
    req = urllib.request.Request(f"{BASE_URL}/services?type=web_service&limit=20", headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            services = json.loads(resp.read().decode())
    except Exception as e:
        print("Failed to list services:", e)
        return

    quasar_service = None
    for item in services:
        svc = item.get("service", item)
        if "quasar" in svc.get("name", "").lower():
            quasar_service = svc
            break

    if not quasar_service:
        print("Quasar service not found on Render!")
        return

    service_id = quasar_service["id"]
    name = quasar_service["name"]
    print(f"Found service: {name} ({service_id})")

    # 2. Get recent deployments
    req2 = urllib.request.Request(f"{BASE_URL}/services/{service_id}/deploys?limit=5", headers=headers)
    try:
        with urllib.request.urlopen(req2) as resp:
            deploys = json.loads(resp.read().decode())
    except Exception as e:
        print("Failed to list deploys:", e)
        return

    print("\nRecent Deployments:")
    for d in deploys:
        dep = d.get("deploy", d)
        status = dep.get("status")
        commit = dep.get("commit", {})
        commit_msg = commit.get("message", "N/A")
        commit_id = commit.get("id", "N/A")
        created_at = dep.get("createdAt")
        print(f"- Time: {created_at} | Status: {status} | Commit: {commit_id[:8]} - {commit_msg.strip()}")

if __name__ == "__main__":
    check_deployments()
