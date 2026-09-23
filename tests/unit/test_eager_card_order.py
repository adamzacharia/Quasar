"""Eager figure cards: the runner must stash ``__eager_data__`` BEFORE the
``__data_ready__`` that consumes it (ui-pro/api/sse.py processes the status
queue in FIFO order). UI benchmark 2026-09-23: a single-card one-shot tool
(L06 HR diagram) rendered BELOW the prose that said "shown above", and a
multi-card tool (L07) emitted each card one slot late with the last below."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace as NS

from core.llm_client import LLMClient
from tests.unit.test_discovery_recovery import _tool_agent


def test_single_card_tool_stashes_its_data_before_the_ready_signal():
    calls = []

    def create(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        if len(calls) == 1:
            delta = NS(content=None, tool_calls=[NS(index=0, id="c1", function=NS(name="datalab_selection_diagram", arguments=json.dumps({"catalog": "gaia_dr3"})))])
        else:
            delta = NS(content="The HR diagram is shown above.", tool_calls=None)
        return iter([NS(choices=[NS(delta=delta, finish_reason="stop")])])

    client = LLMClient(model="gpt-oss-120b")
    client._get_tacc_client = lambda: NS(chat=NS(completions=NS(create=create)))
    agent, executed = _tool_agent(client.responses, tool_name="datalab_selection_diagram")

    def execute(*a, **kw):
        executed.append(kw)
        agent.last_run_result = {"type": "image", "image_url": "/plots/hr.png", "caption": "HR diagram"}
        return {"success": True, "image_attached": True, "rowcount": 4848}

    agent._execute_tool_with_progress = execute
    statuses = []
    agent.stream_response_api("Give me the HR diagram of WD candidates", conversation_id="eager-order",
                              on_status=lambda m, s: statuses.append(m))
    tags = [m.split("{", 1)[0] for m in statuses if isinstance(m, str) and m.startswith(("__eager_data__", "__data_ready__"))]
    assert tags == ["__eager_data__", "__data_ready__"], tags
