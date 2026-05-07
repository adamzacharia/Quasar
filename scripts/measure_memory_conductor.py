"""
Memory stress test — CONDUCTOR PARALLEL AGENTS edition.
Forces the multi-agent Conductor with parallel task execution and monitors peak RSS.
Run:  conda run -n quasar --no-capture-output python measure_memory_conductor.py
"""
import os, sys, gc, time, threading, asyncio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if sys.platform == "win32" and "pwd" not in sys.modules:
    import types
    _pwd = types.ModuleType("pwd")
    _pwd.getpwuid = lambda uid: type("pw", (), {"pw_name": "user"})()
    sys.modules["pwd"] = _pwd

if "cgi" not in sys.modules:
    import email.message, types
    _cgi = types.ModuleType("cgi")
    def _ph(line):
        m = email.message.EmailMessage()
        m['content-type'] = line
        return m.get_content_type(), m.get_params() or {}
    _cgi.parse_header = _ph
    sys.modules["cgi"] = _cgi

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

import psutil
proc = psutil.Process(os.getpid())

# ── Memory sampler ──────────────────────────────────────────────
memory_log = []
_sampling = False
_current_label = "idle"
_start_time = None

def _sampler():
    global _sampling
    while _sampling:
        elapsed = time.time() - _start_time
        rss = proc.memory_info().rss / (1024 * 1024)
        memory_log.append((elapsed, rss, _current_label))
        time.sleep(0.25)  # 250ms resolution for catching spikes

def start_sampling():
    global _sampling, _start_time
    _sampling = True
    _start_time = time.time()
    threading.Thread(target=_sampler, daemon=True).start()

def stop_sampling():
    global _sampling
    _sampling = False
    time.sleep(0.3)

def set_label(label):
    global _current_label
    _current_label = label

def mb():
    return proc.memory_info().rss / (1024 * 1024)


# ── The query that FORCES the Conductor ─────────────────────────
# This must score above 0.55 complexity to trigger multi-agent orchestration.
# We use a genuinely complex research question that requires multiple tools.

CONDUCTOR_QUERY = (
    "Compare ALMA observations of the protostellar systems HH 212 and L1527. "
    "For each source, search the ALMA archive for Band 6 and Band 7 data, "
    "find the latest 5 papers about each on NASA ADS, "
    "and identify what molecular lines have been detected. "
    "Then produce a comparison table with columns: Source, Bands Available, "
    "Number of ALMA Projects, Key Molecules Detected, and Notable Findings."
)

# ── Simulated concurrent users (2 users hitting the API at once) ──
def simulate_concurrent_queries(agent, n_users=2):
    """Simulate n_users hitting the streaming API concurrently, like the real
    ThreadPoolExecutor(4) in main.py does."""
    
    # Simpler queries for concurrent test
    user_queries = [
        "Search ALMA for observations of Sgr A* and summarize what frequency bands are available",
        "Find all ALMA Band 3 observations of NGC 1068 and check for CO line coverage",
    ]
    
    results = [None] * n_users
    errors = [None] * n_users
    
    def _run_user(idx):
        query = user_queries[idx % len(user_queries)]
        try:
            agent.stream_response_api(
                query=query,
                user_id=f"concurrent-user-{idx}",
                on_token=lambda t: None,
                on_status=lambda s, st: None,
            )
            results[idx] = "ok"
        except Exception as e:
            errors[idx] = str(e)
    
    threads = [threading.Thread(target=_run_user, args=(i,)) for i in range(n_users)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    
    return results, errors


def main():
    print(f"\n{'='*75}")
    print(f"  QUASAR CONDUCTOR MEMORY STRESS TEST")
    print(f"  Testing parallel multi-agent execution + concurrent users")
    print(f"  Python {sys.version.split()[0]}  |  PID {os.getpid()}")
    print(f"{'='*75}")
    print(f"\n  Baseline: {mb():.1f} MB\n")

    # ── Load agent ──
    set_label("agent_init")
    start_sampling()
    
    print("  Loading QuasarAgent...", end="", flush=True)
    from core.agent import QuasarAgent, AgentConfig
    agent = QuasarAgent(AgentConfig())
    agent_mb = mb()
    print(f" done ({agent_mb:.1f} MB)\n")

    # ══════════════════════════════════════════════════════════════
    # TEST 1: Single complex query forcing the Conductor
    # ══════════════════════════════════════════════════════════════
    print("  ══ TEST 1: Complex Conductor query (parallel agents) ══")
    print(f"     \"{CONDUCTOR_QUERY[:90]}…\"")
    print(f"     Before: {mb():.1f} MB", flush=True)

    set_label("T1: Conductor")
    before = mb()
    t0 = time.time()

    collected_tokens = []
    status_log = []
    event_log = []

    def on_token(tok):
        collected_tokens.append(tok)

    def on_status(step, state):
        status_log.append((step, state))

    try:
        agent.stream_response_api(
            query=CONDUCTOR_QUERY,
            user_id="conductor-stress-test",
            on_token=on_token,
            on_status=on_status,
        )
    except Exception as e:
        collected_tokens.append(f"[ERROR: {e}]")

    elapsed_t1 = time.time() - t0
    after = mb()
    t1_samples = [e for e in memory_log if e[2] == "T1: Conductor"]
    t1_peak = max(s[1] for s in t1_samples) if t1_samples else after

    response = "".join(collected_tokens)
    
    # Check if Conductor actually ran
    conductor_ran = any("multi-agent" in s[0].lower() or "conductor" in s[0].lower() 
                        for s in status_log)
    
    print(f"     After:  {after:.1f} MB  (Δ{after-before:+.1f})  peak={t1_peak:.1f} MB  [{elapsed_t1:.1f}s]")
    print(f"     Conductor activated: {'✅ YES' if conductor_ran else '❌ NO (fell back to standard)'}")
    print(f"     Status steps: {len(status_log)}")
    print(f"     Response: {response[:120]}{'…' if len(response)>120 else ''}")
    print()

    gc.collect()
    time.sleep(2)

    # ══════════════════════════════════════════════════════════════
    # TEST 2: Two concurrent users sending ALMA queries simultaneously
    # ══════════════════════════════════════════════════════════════
    print("  ══ TEST 2: 2 concurrent users (simulated parallel API calls) ══")
    print(f"     Before: {mb():.1f} MB", flush=True)

    set_label("T2: 2 concurrent users")
    before_t2 = mb()
    t0 = time.time()

    results, errors = simulate_concurrent_queries(agent, n_users=2)

    elapsed_t2 = time.time() - t0
    after_t2 = mb()
    t2_samples = [e for e in memory_log if e[2] == "T2: 2 concurrent users"]
    t2_peak = max(s[1] for s in t2_samples) if t2_samples else after_t2

    print(f"     After:  {after_t2:.1f} MB  (Δ{after_t2-before_t2:+.1f})  peak={t2_peak:.1f} MB  [{elapsed_t2:.1f}s]")
    print(f"     Results: {results}")
    print(f"     Errors:  {errors}")
    print()

    gc.collect()
    time.sleep(2)

    stop_sampling()

    # ══════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════
    overall_peak = max(e[1] for e in memory_log)
    peak_entry = [e for e in memory_log if e[1] == overall_peak][0]

    print(f"\n{'='*75}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*75}")
    print(f"  Agent baseline:        {agent_mb:.1f} MB")
    print(f"  T1 Conductor peak:     {t1_peak:.1f} MB  ({t1_peak/agent_mb:.2f}x baseline)")
    print(f"  T2 Concurrent peak:    {t2_peak:.1f} MB  ({t2_peak/agent_mb:.2f}x baseline)")
    print(f"  Overall peak:          {overall_peak:.1f} MB  (at {peak_entry[0]:.1f}s during '{peak_entry[2]}')")
    print(f"  Final memory:          {mb():.1f} MB")
    print()

    for limit in [512, 768, 1024]:
        headroom = limit - overall_peak
        if headroom > 50:
            verdict = f"✅  {headroom:.0f} MB headroom — safe"
        elif headroom > 0:
            verdict = f"⚠️  Only {headroom:.0f} MB headroom — risky"
        else:
            verdict = f"❌  EXCEEDS by {-headroom:.0f} MB — OOM!"
        print(f"  [{limit:4d} MB plan]  {verdict}")

    # ── Condensed timeline ──
    print(f"\n{'='*75}")
    print(f"  MEMORY TIMELINE (peak per phase)")
    print(f"{'='*75}")
    
    from itertools import groupby
    for label, grp in groupby(memory_log, key=lambda e: e[2]):
        entries = list(grp)
        phase_peak = max(e[1] for e in entries)
        phase_start = entries[0][0]
        phase_end = entries[-1][0]
        print(f"  {phase_start:6.1f}s – {phase_end:6.1f}s  peak={phase_peak:7.1f} MB  {label}")

    print(f"{'='*75}\n")


if __name__ == "__main__":
    main()
