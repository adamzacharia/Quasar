"""core/runner._finalize_answer_text -- one post-processor for every final
answer (guard CX-24: the Conductor return used to bypass the link guard, prose
hygiene, the figure correction and the answer-versus-trace verifier)."""
from __future__ import annotations

import inspect
import json
import threading
from types import SimpleNamespace as NS

import core.runner as runner
from core.agent import QuasarAgent


def _agent(trace):
    agent = QuasarAgent.__new__(QuasarAgent)
    agent._tls = threading.local()
    agent.tool_registry = NS(list_tools=lambda: [NS(name="datalab_sql_query"), NS(name="plot_sky_map")])
    agent._accumulated_tool_trace = trace
    agent._accumulated_run_results = []
    agent.last_run_result = None
    return agent


def test_finalizer_applies_link_guard_hygiene_figure_correction_and_verifier():
    sql = "SELECT ra, dec FROM gaia_dr3.gaia_source WHERE parallax > 5 LIMIT 100"
    trace = [{"name": "datalab_sql_query", "arguments": {"sql": sql}, "sql": sql, "ok": True,
              "output": json.dumps({"success": True, "rowcount": 100})}]
    agent = _agent(trace)
    streamed = []
    text = ("I ran `datalab_sql_query` and kept parallax > 5 and class_star > 0.9 (100 rows). "
            "The sky map is shown above. See https://made-up.example/blog/post for details.")
    out = runner._finalize_answer_text(agent, text, on_token=streamed.append, user_query="find nearby stars",
                                       url_sources=[json.dumps(trace)], all_tool_results=[], had_tool_calls=True)
    assert "made-up.example" not in out.split("> 🔗")[0]           # link guard
    assert "`datalab_sql_query`" not in out                         # prose hygiene
    assert "Correction: no plot, image, or data card" in out        # figure claim without a card
    assert "Verification" in out and "class_star > 0.9" in out     # verifier
    assert streamed, "appended notes are streamed too"


def test_conductor_return_goes_through_the_finalizer():
    src = inspect.getsource(runner._stream_response_api_impl)
    conductor = src[src.index("elif conductor_answer is not None:"):src.index("return safe_assistant_text(conductor_answer)")]
    assert "_finalize_answer_text(" in conductor
    standard = src[src.index("# 7a. Append parallel web search results"):]
    assert "_finalize_answer_text(" in standard
