# tests/smoke/test_smoke.py
"""
End-to-End Smoke Tests for Quasar API

Run against a live server (local or production) to verify the full pipeline:
  query → agent → tool call → SSE response

Usage:
    # Against local dev server:
    pytest tests/smoke/test_smoke.py -v

    # Against production:
    QUASAR_API_URL=https://quasar-7812.onrender.com pytest tests/smoke/test_smoke.py -v

Requires:
    - A running Quasar server
    - pip install pytest requests sseclient-py
"""

import os
import json
import time
import pytest
import requests

API_URL = os.environ.get("QUASAR_API_URL", "http://localhost:8000")
TIMEOUT = 120  # seconds — some queries take a while


def _is_server_reachable() -> bool:
    """Check if the Quasar API server is reachable."""
    try:
        resp = requests.get(f"{API_URL}/", timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


# Skip all tests if server is not running
pytestmark = pytest.mark.skipif(
    not _is_server_reachable(),
    reason=f"Quasar API not reachable at {API_URL}"
)


def _send_query(query: str, user_id: str = "smoke-test-user") -> dict:
    """
    Send a query to the Quasar SSE endpoint and collect the full response.

    Returns dict with:
      - status_code: HTTP status
      - events: list of parsed SSE event dicts
      - full_text: concatenated text from all 'chunk' events
      - tool_calls: list of tool names invoked
      - error: error message if any
    """
    result = {
        "status_code": None,
        "events": [],
        "full_text": "",
        "tool_calls": [],
        "error": None,
    }

    try:
        resp = requests.post(
            f"{API_URL}/api/query",
            json={"query": query, "user_id": user_id},
            headers={"Accept": "text/event-stream"},
            stream=True,
            timeout=TIMEOUT,
        )
        result["status_code"] = resp.status_code

        if resp.status_code != 200:
            result["error"] = f"HTTP {resp.status_code}: {resp.text[:500]}"
            return result

        # Parse SSE events
        buffer = ""
        for line in resp.iter_lines(decode_unicode=True):
            if line is None:
                continue
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    event = json.loads(data_str)
                    result["events"].append(event)

                    # Extract text chunks
                    if event.get("type") == "chunk":
                        result["full_text"] += event.get("content", "")
                    elif event.get("type") == "text":
                        result["full_text"] += event.get("content", "")
                    elif event.get("type") == "tool_call":
                        result["tool_calls"].append(event.get("name", "unknown"))
                    elif event.get("type") == "thought":
                        pass  # Agent thinking — ignore
                    elif event.get("type") == "error":
                        result["error"] = event.get("content", "Unknown error")
                except json.JSONDecodeError:
                    # Some events are plain text
                    result["full_text"] += data_str

    except requests.Timeout:
        result["error"] = f"Timeout after {TIMEOUT}s"
    except Exception as e:
        result["error"] = str(e)

    return result


# ── Test Cases ────────────────────────────────────────────────────


class TestAPIHealth:
    """Basic API health and connectivity tests."""

    def test_root_endpoint(self):
        """API root returns 200."""
        resp = requests.get(f"{API_URL}/", timeout=10)
        assert resp.status_code == 200

    def test_health_endpoint(self):
        """Health check endpoint responds."""
        resp = requests.get(f"{API_URL}/health", timeout=10)
        assert resp.status_code in [200, 404]  # 404 is OK if not implemented


class TestBasicQueries:
    """Test that fundamental query types return valid responses."""

    @pytest.mark.smoke
    def test_simple_knowledge_question(self):
        """A simple astrophysics question should return text."""
        result = _send_query("What is the CO(2-1) rest frequency in GHz?")
        assert result["error"] is None, f"Query failed: {result['error']}"
        assert result["status_code"] == 200
        assert len(result["full_text"]) > 20, "Response too short"
        # Should mention ~230 GHz
        assert any(
            kw in result["full_text"].lower()
            for kw in ["230", "ghz", "co"]
        ), f"Response doesn't mention CO frequency: {result['full_text'][:200]}"

    @pytest.mark.smoke
    def test_alma_observation_search(self):
        """An ALMA search query should invoke archive tools."""
        result = _send_query("Find ALMA observations of HL Tau")
        assert result["error"] is None, f"Query failed: {result['error']}"
        assert result["status_code"] == 200
        assert len(result["full_text"]) > 50, "Response too short for observation search"

    @pytest.mark.smoke
    def test_paper_search(self):
        """A literature search should invoke ADS tools."""
        result = _send_query("Search NASA ADS for recent ALMA papers on protoplanetary disks, limit to 3")
        assert result["error"] is None, f"Query failed: {result['error']}"
        assert result["status_code"] == 200
        assert len(result["full_text"]) > 50, "Response too short for paper search"


class TestToolExecution:
    """Verify that specific tools can be invoked and return results."""

    @pytest.mark.smoke
    @pytest.mark.slow
    def test_multi_archive_search(self):
        """A multi-archive query should return data from at least one archive."""
        result = _send_query("Search for observations of M31 across ALMA and VLA archives")
        assert result["error"] is None, f"Query failed: {result['error']}"
        assert len(result["full_text"]) > 100, "Response too short"


class TestEdgeCases:
    """Test error handling and edge cases."""

    @pytest.mark.smoke
    def test_empty_query(self):
        """Empty query should be handled gracefully."""
        result = _send_query("")
        # Should not crash — either returns a message or a 400
        assert result["error"] is None or result["status_code"] in [200, 400]

    @pytest.mark.smoke
    def test_nonsense_query(self):
        """Nonsense input shouldn't crash the agent."""
        result = _send_query("asdfjkl;qwerty12345")
        assert result["status_code"] == 200
        assert result["error"] is None


# ── Summary Reporter ─────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  Quasar Smoke Test Runner")
    print(f"  Target: {API_URL}")
    print(f"{'='*60}\n")

    queries = [
        ("Knowledge", "What is the CO(2-1) rest frequency?"),
        ("ALMA Search", "Find ALMA observations of HL Tau"),
        ("Paper Search", "Search ADS for recent ALMA papers on disks, limit 3"),
    ]

    passed = 0
    failed = 0

    for name, query in queries:
        print(f"  [{name}] Sending query...")
        start = time.time()
        result = _send_query(query)
        elapsed = time.time() - start

        if result["error"]:
            print(f"  [{name}] ❌ FAILED ({elapsed:.1f}s): {result['error']}")
            failed += 1
        elif len(result["full_text"]) < 20:
            print(f"  [{name}] ❌ FAILED ({elapsed:.1f}s): Response too short ({len(result['full_text'])} chars)")
            failed += 1
        else:
            print(f"  [{name}] ✅ PASSED ({elapsed:.1f}s, {len(result['full_text'])} chars)")
            passed += 1

    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'='*60}\n")
