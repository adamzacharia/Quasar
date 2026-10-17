"""Heartbeats while the post-answer passes run (round 5: the inactivity watchdog ended two finished answers)."""
import time

from core.runner import _PostPassHeartbeat


def test_heartbeat_emits_now_and_every_interval_then_stops():
    seen = []
    with _PostPassHeartbeat(lambda step, state: seen.append((step, state)), "code_check", "Checking", interval=0.05):
        time.sleep(0.18)
    n = len(seen)
    assert n >= 3
    assert seen[0] == ("__tool_heartbeat__code_check::Checking", "meta")
    time.sleep(0.15)
    assert len(seen) == n  # stopped on exit


def test_heartbeat_without_callback_is_a_no_op():
    with _PostPassHeartbeat(None, "code_check", "Checking", interval=0.01):
        pass
