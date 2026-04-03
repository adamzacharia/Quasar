"""
Memory stress test — monitors RSS while running real queries through QuasarAgent.
Samples memory every 0.5s in a background thread and prints a timeline + peak.
Run:  conda run -n quasar --no-capture-output python measure_memory_stress.py
"""
import os, sys, gc, time, threading

# Project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Windows / Python 3.13 stubs
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

# ── Memory sampler (background thread) ──────────────────────────
memory_log = []       # list of (elapsed_sec, rss_mb, label)
_sampling = False
_current_label = "idle"
_start_time = None

def _sampler():
    global _sampling
    while _sampling:
        elapsed = time.time() - _start_time
        rss = proc.memory_info().rss / (1024 * 1024)
        memory_log.append((elapsed, rss, _current_label))
        time.sleep(0.5)

def start_sampling():
    global _sampling, _start_time
    _sampling = True
    _start_time = time.time()
    t = threading.Thread(target=_sampler, daemon=True)
    t.start()

def stop_sampling():
    global _sampling
    _sampling = False
    time.sleep(0.6)

def set_label(label):
    global _current_label
    _current_label = label

def mb():
    return proc.memory_info().rss / (1024 * 1024)


# ── Queries to test (simple → complex) ──────────────────────────

QUERIES = [
    {
        "label": "Q1: Simple greeting",
        "query": "Hello, what can you help me with?",
    },
    {
        "label": "Q2: ALMA target search",
        "query": "Search ALMA for observations of TW Hya in Band 6",
    },
    {
        "label": "Q3: Paper search",
        "query": "Find recent papers about protoplanetary disk kinematics",
    },
    {
        "label": "Q4: Complex multi-step",
        "query": (
            "I need a comprehensive analysis of HH 212. "
            "Search for all ALMA Band 7 observations, check what spectral lines "
            "are covered, look up the latest papers about this source, and "
            "summarize the key findings about the protostellar jet and disk."
        ),
    },
]


def run_query(agent, query_text: str) -> str:
    """Run a query through the agent's streaming API and collect the response."""
    collected = []

    def on_token(tok):
        collected.append(tok)

    def on_status(step, state):
        pass  # silent

    try:
        agent.stream_response_api(
            query=query_text,
            user_id="mem-stress-test",
            on_token=on_token,
            on_status=on_status,
        )
    except Exception as e:
        collected.append(f"[ERROR: {e}]")

    return "".join(collected)


# ── Main ────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*75}")
    print(f"  QUASAR MEMORY STRESS TEST")
    print(f"  Python {sys.version.split()[0]}  |  PID {os.getpid()}")
    print(f"{'='*75}")
    print(f"\n  Baseline (Python only):  {mb():.1f} MB\n")

    # ── Load agent ──
    set_label("agent_init")
    start_sampling()

    print("  Loading QuasarAgent...", end="", flush=True)
    from core.agent import QuasarAgent, AgentConfig
    config = AgentConfig()
    agent = QuasarAgent(config)
    agent_mb = mb()
    print(f" done  ({agent_mb:.1f} MB)\n")

    # ── Run each query ──
    for i, q in enumerate(QUERIES):
        label = q["label"]
        query = q["query"]

        print(f"  ── {label}")
        print(f"     \"{query[:90]}{'…' if len(query)>90 else ''}\"")
        print(f"     Before: {mb():.1f} MB", flush=True)

        set_label(label)
        before = mb()
        t0 = time.time()

        response = run_query(agent, query)

        elapsed = time.time() - t0
        after = mb()
        delta = after - before

        # Get peak during this query from the log
        query_samples = [e for e in memory_log if e[2] == label]
        peak_during = max(s[1] for s in query_samples) if query_samples else after

        print(f"     After:  {after:.1f} MB  (Δ{delta:+.1f})  peak={peak_during:.1f} MB  [{elapsed:.1f}s]")
        preview = response[:150].replace("\n", " ")
        print(f"     → {preview}{'…' if len(response)>150 else ''}")
        print()

        # GC between queries
        gc.collect()
        time.sleep(1)

    stop_sampling()

    # ── Summary ──
    peak_rss = max(e[1] for e in memory_log)
    peak_entry = [e for e in memory_log if e[1] == peak_rss][0]

    print(f"\n{'='*75}")
    print(f"  RESULTS")
    print(f"{'='*75}")
    print(f"  Agent baseline:      {agent_mb:.1f} MB")
    print(f"  Peak memory:         {peak_rss:.1f} MB  (at {peak_entry[0]:.1f}s during '{peak_entry[2]}')")
    print(f"  Final memory:        {mb():.1f} MB")
    print(f"  Peak ÷ baseline:     {peak_rss/agent_mb:.2f}x")
    print()

    limit = 512
    headroom = limit - peak_rss
    print(f"  [{limit} MB PLAN]")
    if headroom > 50:
        print(f"     ✅  {headroom:.0f} MB headroom at peak — fits comfortably")
    elif headroom > 0:
        print(f"     ⚠️  Only {headroom:.0f} MB headroom — tight, may OOM under concurrent load")
    else:
        print(f"     ❌  EXCEEDS {limit} MB by {-headroom:.0f} MB — OOM guaranteed!")

    # ── Timeline (condensed) ──
    print(f"\n{'='*75}")
    print(f"  MEMORY TIMELINE")
    print(f"{'='*75}")
    print(f"  {'Time':>7s}  {'RSS MB':>8s}  Phase")
    print(f"  {'─'*7}  {'─'*8}  {'─'*40}")

    prev_label = None
    for elapsed, rss, label in memory_log:
        marker = " ◄─── PHASE CHANGE" if label != prev_label else ""
        print(f"  {elapsed:7.1f}s  {rss:8.1f}  {label}{marker}")
        prev_label = label

    print(f"{'='*75}\n")


if __name__ == "__main__":
    main()
