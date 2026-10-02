"""Rebuild lost provider history from the saved chat (core/history_rebuild.py)."""
from types import SimpleNamespace

from core.history_rebuild import RECAP_HEADER, as_recap_text, build_seed, prior_turns
from core.llm_client import ResponsesShim


def _hist(n_turns, long=False):
    h = []
    for i in range(n_turns):
        h.append({"role": "user", "content": f"question {i} " + ("x" * 9000 if long else "")})
        h.append({"role": "assistant", "content": f"answer {i} " + ("y" * 9000 if long else "")})
    return h


def test_prior_turns_drops_current_and_leading_assistant():
    h = [{"role": "assistant", "content": "welcome"}] + _hist(2) + [{"role": "user", "content": "now?"}]
    msgs = prior_turns(h, "now?")
    assert msgs[0]["content"].startswith("question 0") and msgs[-1]["content"].startswith("answer 1")


def test_seed_keeps_recent_verbatim_and_abridges_older():
    seed, tok = build_seed(_hist(6, long=True), "next", budget_tokens=6000)
    assert seed[0]["role"] == "user"
    assert len(seed[-1]["content"]) > 5000          # newest assistant turn kept (clipped at 6k chars)
    assert len(seed[0]["content"]) <= 310          # oldest abridged
    assert tok <= 6000


def test_seed_respects_budget_and_empty_cases():
    assert build_seed([], "q") == ([], 0)
    assert build_seed(_hist(3), "q", budget_tokens=0) == ([], 0)
    seed, tok = build_seed(_hist(50), "q", budget_tokens=200)
    assert tok <= 200 and seed and seed[0]["role"] == "user"


def test_recap_text_for_other_providers():
    seed, _ = build_seed(_hist(1), "q")
    text = as_recap_text(seed)
    assert text.startswith(RECAP_HEADER) and "User: question 0" in text and "Assistant: answer 0" in text


def test_shim_builds_role_preserving_messages_from_seed():
    shim = ResponsesShim(SimpleNamespace())
    seed, _ = build_seed(_hist(2), "q")
    msgs = shim._chat_messages_for_input(None, "SYS", [*seed, {"role": "user", "content": "follow-up"}])
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert msgs[-1]["content"] == "follow-up"
