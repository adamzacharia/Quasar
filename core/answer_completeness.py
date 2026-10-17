"""Expert completeness pass for how-to / concept answers (ALMAbench round 5, 2026-10-16).

Answers grounded in documentation excerpts tend to follow the excerpts and leave out points an expert would add from
general knowledge: what biases or breaks a result, how it depends on frequency, array configuration, cycle or
software version, and how to check it. Round-4 grading showed the plain model often covered those points while the
same model inside Quasar did not.

This pass gives the SAME model that wrote the answer one short review call: the question and the draft, and the
request to list only the important points the draft leaves out, or to reply NONE. The reply is appended as a short
"Additional points" section. Nothing in the prompt is specific to any question; it asks for missing caveats,
dependencies and checks in general.

``QUASAR_COMPLETENESS_PASS`` (default on) turns it off; ``QUASAR_COMPLETENESS_BUDGET`` (default 60 s) bounds the call.
"""
from __future__ import annotations

import os
import re
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

MAX_OUTPUT_TOKENS = 1800
MIN_ANSWER_CHARS = 300
HEADING = "### Additional points"

REVIEW_PROMPT = """You are an expert radio astronomer reviewing a draft answer before it is shown to a user.

User question:
{question}

Draft answer:
<<<DRAFT
{draft}
DRAFT>>>

List the important points that an expert answer to THIS question would include but the draft leaves out or covers
only partly. Look in particular for:
- caveats and common mistakes: what can bias or break the result, and how to check for it;
- how the answer depends on observing frequency or band, array or configuration, cycle, or software version;
- practical steps the user still needs: where the information or file is found, what to check before use;
- a part of the question the draft does not answer.

Rules:
- Only well-established points that any expert would agree with. Do not repeat anything the draft already says, and
  do not contradict it: the draft is grounded in the official documentation and you are not.
- Keep the points general. Give a number (a typical value, a frequency, a size) only when it is a standard textbook
  or documentation value you are certain of. Do not state software versions, cycle numbers or dates when a default
  or behaviour changed unless the draft already gives them.
- Do not invent URLs, document section numbers, archive column names, project codes, counts or names of specific
  datasets. No citations.
- Plain language, one to three sentences per point, usually 2 to 5 points, most important first. Do not use em
  dashes.
- Reply with the points as a bulleted list and nothing else. If nothing important is missing, reply with exactly NONE.
"""


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def completeness_enabled() -> bool:
    return _flag("QUASAR_COMPLETENESS_PASS", True)


def budget_seconds() -> float:
    try:
        val = float(os.getenv("QUASAR_COMPLETENESS_BUDGET", "") or 60.0)
    except ValueError:
        return 60.0
    return val if val > 0 else 60.0


def _default_call(prompt: str, model: str, max_tokens: int) -> str:
    from core.llm_client import LLMClient

    client = LLMClient(model=model)
    resp = client.responses.create(model=model, input=prompt, max_output_tokens=max_tokens)
    return str(getattr(resp, "output_text", "") or "")


def _strip_markup(raw: str) -> str:
    try:
        from core.harmony_filter import strip_harmony_markup

        raw = strip_harmony_markup(raw)
    except Exception:
        pass
    return str(raw or "").strip()


_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+\S", re.MULTILINE)


def clean_points(raw: str) -> Optional[str]:
    """The bulleted list from the review reply, or None for NONE, an empty or an unusable reply."""
    text = _strip_markup(raw)
    if not text or re.fullmatch(r"\W*none\W*", text, re.IGNORECASE):
        return None
    lines = text.splitlines()
    # drop a heading or preamble the model added before the list
    while lines and not _BULLET_RE.match(lines[0]):
        lines.pop(0)
    body = "\n".join(lines).strip()
    if not body or not _BULLET_RE.search(body):
        return None
    body = body.replace("—", ", ").replace(" – ", ", ")
    bullets = [m.start() for m in _BULLET_RE.finditer(body)]
    if len(bullets) > 6:
        body = body[:bullets[6]].rstrip()
    return body[:6000]


def complete_answer(
    text: str,
    *,
    question: str,
    model: str,
    call: Optional[Callable[[str, str, int], str]] = None,
    deadline: Optional[float] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Return ``(text, info)``; text gains an "Additional points" section when the review finds missing points."""
    info: Dict[str, Any] = {"status": "skipped"}
    if not completeness_enabled():
        info["skipped"] = "QUASAR_COMPLETENESS_PASS off"
        return text, info
    if not model or len((text or "").strip()) < MIN_ANSWER_CHARS or HEADING in (text or ""):
        info["skipped"] = "no model, short answer or already reviewed"
        return text, info
    budget = budget_seconds()
    if deadline is not None:
        budget = min(budget, deadline - time.monotonic() - 30.0)
    if budget < 10.0:
        info["skipped"] = "not enough time left in the turn"
        return text, info

    from core.llm_client import get_llm_request_context, reinstall_llm_request_context

    prompt = REVIEW_PROMPT.format(question=(question or "").strip()[:3000], draft=(text or "")[:24000])
    fn = call or _default_call
    ctx = get_llm_request_context()
    box: Dict[str, Any] = {}

    def body() -> None:
        try:
            with reinstall_llm_request_context(ctx):
                box["raw"] = fn(prompt, model, MAX_OUTPUT_TOKENS)
        except BaseException as exc:  # noqa: BLE001 - best effort
            box["error"] = f"{type(exc).__name__}: {exc}"[:200]

    t0 = time.monotonic()
    worker = threading.Thread(target=body, name="completeness-pass", daemon=True)
    worker.start()
    worker.join(timeout=budget)
    info["elapsed_s"] = round(time.monotonic() - t0, 1)
    if worker.is_alive():
        info["skipped"] = f"timeout after {budget:.0f}s"
        return text, info
    if "error" in box:
        info["skipped"] = box["error"]
        return text, info
    points = clean_points(box.get("raw") or "")
    if not points:
        info["status"] = "none"
        return text, info
    try:
        from services.content_safety import looks_explicit_text

        if looks_explicit_text(points) and not looks_explicit_text(text):
            # The final-answer guard replaces a WHOLE answer that trips it (round-5 A25: a correct answer became
            # the filter notice after this pass added text). An addition must never cost the answer: drop it.
            info["status"] = "dropped (safety filter)"
            return text, info
    except Exception:
        pass
    info["status"] = "added"
    info["points"] = len(_BULLET_RE.findall(points))
    return text.rstrip() + f"\n\n{HEADING}\n\n{points}", info
