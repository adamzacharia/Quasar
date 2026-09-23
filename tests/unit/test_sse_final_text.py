"""The ``final_text`` SSE event (WP1 1.7; guard CX-31 / CX-32).

The UI shows streamed tokens; backend post-processing (link guard, verifier,
prose hygiene) changes the answer after streaming. ``final_text`` hands the
client the persisted copy. Pinned here: every case of the decision
(``api.sse.final_text_event``), including the empty-final case CX-31 found,
and the stream order -- the event is emitted before persistence and before
``[DONE]`` from the same ``response_text`` that is persisted.
"""
from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

_UI_PRO = str(Path(__file__).resolve().parents[2] / "ui-pro")
if _UI_PRO not in sys.path:
    sys.path.insert(0, _UI_PRO)

import api.sse as sse  # noqa: E402
from api.sse import final_text_event  # noqa: E402


def test_nothing_streamed_sends_the_whole_answer_as_one_token():
    assert final_text_event("Answer.", "", first_token=True) == {"type": "token", "content": "Answer."}
    assert final_text_event("", "", first_token=True) is None


def test_post_processed_answer_replaces_the_streamed_text():
    streamed = "See https://made-up.example/x for more."
    final = "See for more.\n\n> 🔗 Removed 1 external link ..."
    assert final_text_event(final, streamed, first_token=False) == {"type": "final_text", "content": final}


def test_identical_text_sends_nothing():
    assert final_text_event("Same text. ", "Same text.", first_token=False) is None


def test_post_processing_that_empties_the_answer_still_replaces_it():
    """CX-31: the persisted answer is empty, so the client copy must be too."""
    assert final_text_event("", "streamed words", first_token=False) == {"type": "final_text", "content": ""}
    assert final_text_event(None, "streamed words", first_token=False) == {"type": "final_text", "content": ""}


def test_final_text_is_emitted_before_persistence_and_done_from_the_persisted_variable():
    src = inspect.getsource(sse)
    i_event = src.index("_tail_event = final_text_event(response_text, streamed_text, first_token=first_token)")
    i_persist = src.index("await asyncio.to_thread(_persist_assistant_turn)", i_event)
    i_done = src.index('yield "data: [DONE]\\n\\n"', i_persist)
    assert i_event < i_persist < i_done
    # The persisted record is built from the same response_text the event carries.
    persist_fn = src[src.index("def _persist_assistant_turn"):src.index("def _persist_assistant_turn") + 4000]
    assert re.search(r"\bresponse_text\b", persist_fn)
