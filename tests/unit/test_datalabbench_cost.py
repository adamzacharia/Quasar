"""DataLabBench cost resolution (`_resolve_cost`).

The bench must trust a cost-aware backend's honest "unpriced" (explicit null)
and only derive a local price for old servers that predate cost accounting.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_BENCH = _REPO / "Benchmark" / "datalabbench" / "run_datalabbench.py"


@pytest.fixture(scope="module")
def bench():
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))
    spec = importlib.util.spec_from_file_location("run_datalabbench", _BENCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_backend_priced_cost_is_trusted(bench):
    out = bench._resolve_cost(
        {"input": 100, "output": 50, "total": 150, "cost_usd": 0.25,
         "unpriced_tokens": 0, "cost_aware": True},
        "deepseek-v4-flash",
    )
    assert out["cost_usd"] == 0.25
    assert out["cost_source"] == "backend"


def test_cost_aware_null_is_not_overridden_with_a_derived_price(bench):
    """The key fix: a cost-aware backend saying costUsd=null is honest
    "unpriced" — the bench must NOT invent a main-model price for it."""
    out = bench._resolve_cost(
        {"input": 1_000_000, "output": 0, "total": 1_000_000, "cost_usd": None,
         "unpriced_tokens": 1_000_000, "cost_aware": True},
        "gpt-4o-mini",  # a priced model — deriving WOULD have produced $0.15
    )
    assert out["cost_usd"] is None
    assert out["cost_source"] == "backend"
    assert out["unpriced_tokens"] == 1_000_000


def test_old_server_without_cost_field_gets_a_derived_price(bench):
    """An old server (no costUsd key at all → cost_aware False) is the only
    case the bench derives a local price."""
    out = bench._resolve_cost(
        {"input": 1_000_000, "output": 0, "total": 1_000_000, "cost_aware": False},
        "gpt-4o-mini",
    )
    assert out["cost_usd"] == pytest.approx(0.15)
    assert out["cost_source"] == "derived"
    assert out["unpriced_tokens"] == 0


def test_old_server_unpriced_model_stays_unpriced(bench):
    out = bench._resolve_cost(
        {"input": 1_000_000, "output": 0, "total": 1_000_000, "cost_aware": False},
        "gpt-oss-120b",  # TACC — no price
    )
    assert out["cost_usd"] is None
    assert out["cost_source"] == "unpriced"
    assert out["unpriced_tokens"] == 1_000_000


def test_generate_report_renders_cost_column(bench, tmp_path):
    """The report must contain a per-question cost column, an unpriced-tokens
    metric, and a telemetry-missing count (CX-03/CX-10/CX-15)."""
    import types

    priced = bench.QuestionResult(id="q1", tier=1, title="Priced", prompt="p1")
    priced.usage = {"total": 1000, "cost_usd": 0.0123, "unpriced_tokens": 0,
                    "cost_source": "backend"}
    unpriced = bench.QuestionResult(id="q2", tier=2, title="TACC", prompt="p2")
    unpriced.usage = {"total": 5000, "cost_usd": None, "unpriced_tokens": 5000,
                      "cost_source": "unpriced"}
    no_tel = bench.QuestionResult(id="q3", tier=1, title="Dropped", prompt="p3")
    no_tel.usage = {}  # no usage event at all

    results = [priced, unpriced, no_tel]
    rollup = bench.overall_rollup(results)
    args = types.SimpleNamespace(
        model="deepseek-v4-flash", api_url="http://x", judge_model="j",
        skip_judge=True,
    )

    bench.generate_report(results, tmp_path, args, rollup)
    report = (tmp_path / "DataLabBench_report.md").read_text(encoding="utf-8")

    assert "## Cost per question" in report
    assert "| Question | Tier | Tokens | Cost (est.) | Unpriced tokens | Source |" in report
    assert "$0.0123" in report          # priced question's cost
    assert "no telemetry" in report     # the dropped question's source
    assert "Unpriced tokens (excluded from cost)" in report
    assert "Questions with no telemetry" in report
    # The no-telemetry question's DETAIL line must agree with its table label —
    # it says "no telemetry", not "0 tokens · cost unpriced" (CX-10). Locate
    # the q3 detail line (the one that follows the "### q3" header).
    report_lines = report.splitlines()
    q3_idx = next(i for i, ln in enumerate(report_lines) if ln.startswith("### q3"))
    q3_detail = next(ln for ln in report_lines[q3_idx:] if ln.startswith("Response time"))
    assert "no telemetry" in q3_detail
    assert "cost unpriced" not in q3_detail
    assert "0 tokens" not in q3_detail
