import time
import requests
import datetime
import argparse

# The URLs to keep alive
URLS = [
    "https://quasar-oi14.onrender.com",
    "https://quasar-alpha.vercel.app"
]

def ping_urls():
    print(f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting ping cycle...")
    for url in URLS:
        try:
            # We use a GET request. For APIs, hitting a lightweight endpoint like / or /health is best.
            response = requests.get(url, timeout=30)
            print(f" ✅ [{response.status_code}] Successfully pinged {url}")
        except requests.exceptions.RequestException as e:
            print(f" ❌ Failed to ping {url}. Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Keep websites alive by pinging them periodically.")
    parser.add_argument("--interval", type=int, default=10, help="Ping interval in minutes (default: 10)")
    args = parser.parse_args()

    interval_seconds = args.interval * 60
    
    print(f"🚀 Keep-alive script started.")
    print(f"⏱️  Pinging every {args.interval} minutes ({args.interval / 60:.2f} hours).")
    print(f"⚠️  Note: Render free tier spins down after 15 minutes of inactivity. This script will keep it awake by pinging every {args.interval} minutes.")
    
    try:
        ping_urls()
        print("\n✅ Keep-alive ping complete.")
    except Exception as e:
        print(f"\n❌ Keep-alive ping failed: {e}")
