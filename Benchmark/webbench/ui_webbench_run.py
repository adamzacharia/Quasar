"""WebBench UI driver: every turn is typed into the real Quasar UI and captured there.

Based on tmp/ui-bench-2026-09-24/ui_bench_run.py (same capture conventions), plus:
  * sets the "Web Search" switch in the + menu per question (web_toggle: on|off)
  * multi-turn chats (all turns in one chat; the LAST turn is graded)
  * polls while the graded turn streams to time when the web sources card and
    the first answer text appear (time_to_sources_s / time_to_first_text_s)
  * saves the web sources card (title, domain, href, citation badge if any)
  * saves the backend log lines about web search for the turn (supporting
    evidence only; the UI capture is the evidence of record)

The only API call is POST /api/auth/login for the disposable local test account
(token injected into localStorage; the password is never typed into the UI).

usage: ui_webbench_run.py OUT_ROOT [questions.yaml] [timeout_s] [only_id,only_id,...]
env:   WEBBENCH_BACKEND_LOG  backend stdout log to slice (default: traced launcher log)
       WEBBENCH_FRONTEND     default http://localhost:3001
       WEBBENCH_BACKEND      default http://localhost:8000
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import requests
import yaml
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
QFILE = Path(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else HERE / "questions.yaml"
TIMEOUT_S = float(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else 300.0
ONLY = set(sys.argv[4].split(",")) if len(sys.argv) > 4 and sys.argv[4] else None

QUESTIONS = yaml.safe_load(QFILE.read_text(encoding="utf-8"))["questions"]
if ONLY:
    QUESTIONS = [q for q in QUESTIONS if q["id"] in ONLY]

FRONTEND = os.getenv("WEBBENCH_FRONTEND", "http://localhost:3001")
BACKEND = os.getenv("WEBBENCH_BACKEND", "http://localhost:8000")
REPO = HERE.parents[1]
BACKEND_LOG = Path(os.getenv("WEBBENCH_BACKEND_LOG", str(REPO / "tmp" / "latency-repro-2026-09-21" / "backend-stdout.log")))
RESULTS = OUT / "live_results.jsonl"
VIEWPORT = {"width": 1400, "height": 1900}
MAX_TALL = 14000
WEB_LOG_RE = re.compile(r"\[(?:WEB SEARCH|SEARCH ROUTER|EMAIL SEARCH|WEB EVIDENCE|WEB CITE|WEB DEEP READ|WEB PLANNER|WEB REVISE|WEB )|web_search|web_extract|Tavily|Brave|Exa\b|\[VERIFY\] web_citation", re.I)
# Phase 2: the verifier's unsupported-claim counts (before / after the revise pass)
VERIFY_BEFORE_RE = re.compile(r"\[VERIFY\] web_citation unsupported=(\d+)")
VERIFY_AFTER_RE = re.compile(r"\[VERIFY\] web_citation unsupported after revise=(\d+)")
PLANNER_RE = re.compile(r"\[WEB PLANNER\] (\d+\.\d+)s need_web=(\w+)")


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def get_login() -> dict:
    r = requests.post(f"{BACKEND}/api/auth/login", json={"username": "1@1", "password": "1"}, timeout=20)
    r.raise_for_status()
    d = r.json()
    log("login: fresh token from /api/auth/login")
    return d


def launch(p):
    for kw in ({"channel": "msedge"}, {}, {"channel": "chrome"}):
        try:
            return p.chromium.launch(headless=True, **kw)
        except Exception:
            continue
    raise RuntimeError("no browser available")


def backend_log_size() -> int:
    try:
        return BACKEND_LOG.stat().st_size
    except Exception:
        return -1


def backend_log_slice(start: int, end: int) -> str:
    if start < 0 or end <= start:
        return ""
    try:
        with BACKEND_LOG.open("rb") as fh:
            fh.seek(start)
            raw = fh.read(end - start)
        return raw.decode("utf-8", errors="replace")
    except Exception as exc:
        return f"(log slice failed: {exc})"


# ---- page helpers -----------------------------------------------------------------

JS_SET_DETAILS = """(open) => {
  const main = document.querySelector('main') || document.body;
  const ds = main.querySelectorAll('details');
  ds.forEach(d => { d.open = open; });
  return ds.length;
}"""

JS_SCROLL_HEIGHT = """() => {
  const main = document.querySelector('main') || document.body;
  let best = null;
  main.querySelectorAll('*').forEach(el => {
    const cs = getComputedStyle(el);
    if ((cs.overflowY === 'auto' || cs.overflowY === 'scroll') && el.scrollHeight > el.clientHeight + 20) {
      if (!best || el.scrollHeight > best.scrollHeight) best = el;
    }
  });
  if (!best) return {has: false, need: document.documentElement.scrollHeight};
  return {has: true, need: best.scrollHeight + (window.innerHeight - best.clientHeight)};
}"""

JS_SCROLL_TOP = """() => {
  const main = document.querySelector('main') || document.body;
  main.querySelectorAll('*').forEach(el => {
    const cs = getComputedStyle(el);
    if ((cs.overflowY === 'auto' || cs.overflowY === 'scroll') && el.scrollHeight > el.clientHeight) el.scrollTop = 0;
  });
  window.scrollTo(0, 0);
}"""

# Last assistant message only (multi-turn chats have several).
JS_ANSWER = """() => {
  const main = document.querySelector('main') || document.body;
  const blocks = Array.from(main.querySelectorAll('.prose-answer'));
  const texts = blocks.map(b => b.innerText);
  const htmls = blocks.map(b => b.innerHTML);
  const tables = main.querySelectorAll('table').length;
  const errs = Array.from(main.querySelectorAll('.prose-answer, [class*="error"], [role="alert"]'))
     .map(e => (e.innerText || '').trim())
     .filter(t => /^(?:Error:|Request failed|Something went wrong)/i.test(t));
  const chips = Array.from(main.querySelectorAll('[data-web-cite]')).map(c => c.getAttribute('data-web-cite'));
  return {texts, htmls, tables, errs, chips};
}"""

JS_THOUGHT = """() => {
  const main = document.querySelector('main') || document.body;
  const out = [];
  main.querySelectorAll('details').forEach(d => {
    const s = d.querySelector('summary');
    if (s && /Thought|Thinking/.test(s.innerText)) out.push(d.innerText);
  });
  return out;
}"""

JS_QUERIES = """() => {
  const main = document.querySelector('main') || document.body;
  const btns = Array.from(main.querySelectorAll('button[aria-expanded]'))
     .filter(b => /^(Show|Hide) query/.test((b.innerText || '').trim()));
  return btns.length;
}"""

JS_QUERY_TEXT = """() => {
  const main = document.querySelector('main') || document.body;
  const btns = Array.from(main.querySelectorAll('button[aria-expanded="true"]'))
     .filter(b => /^(Show|Hide) query/.test((b.innerText || '').trim()));
  return btns.map(b => { const wrap = b.parentElement; return wrap ? wrap.innerText : ''; });
}"""

# Web sources card: the redesigned card carries data-testid="web-sources-card";
# the legacy card is found by its "Evidence-ranked sources" heading.
JS_SOURCES = """() => {
  const main = document.querySelector('main') || document.body;
  let cards = Array.from(main.querySelectorAll('[data-testid="web-sources-card"]'));
  if (!cards.length) {
    const heads = Array.from(main.querySelectorAll('span')).filter(s => /Evidence-ranked sources/i.test(s.innerText || ''));
    cards = heads.map(h => h.closest('.space-y-4') || h.parentElement.parentElement);
  }
  return cards.filter(Boolean).map(card => ({
    text: card.innerText,
    links: Array.from(card.querySelectorAll('a[href]')).map(a => ({
      href: a.getAttribute('href'), text: (a.innerText || '').replace(/\\s+/g, ' ').trim().slice(0, 200),
      target: a.getAttribute('target'), rel: a.getAttribute('rel'), id: a.getAttribute('data-source-id') || null,
    })),
  }));
}"""

JS_STREAM_PROBE = """() => {
  const main = document.querySelector('main') || document.body;
  const nCards = main.querySelectorAll('[data-testid="web-sources-card"], [data-testid="web-sources-strip"]').length
     + Array.from(main.querySelectorAll('span')).filter(s => /^Evidence-ranked sources$/i.test((s.innerText || '').trim())).length;
  const hasSources = nCards > 0;
  const blocks = main.querySelectorAll('.prose-answer');
  const last = blocks.length ? blocks[blocks.length - 1] : null;
  return {hasSources, nCards, textLen: last ? (last.innerText || '').length : 0, nBlocks: blocks.length};
}"""


def screenshot_tall(page, path: Path) -> dict:
    info = {"viewport_h": VIEWPORT["height"]}
    try:
        page.evaluate(JS_SCROLL_TOP)
        m = page.evaluate(JS_SCROLL_HEIGHT)
        need = int(m.get("need") or 0)
        if m.get("has") and need > VIEWPORT["height"]:
            h = min(need + 120, MAX_TALL)
            page.set_viewport_size({"width": VIEWPORT["width"], "height": h})
            time.sleep(0.6)
            page.evaluate(JS_SCROLL_TOP)
            info["viewport_h"] = h
            info["truncated"] = need + 120 > MAX_TALL
        page.screenshot(path=str(path), full_page=True)
    finally:
        page.set_viewport_size(VIEWPORT)
        time.sleep(0.3)
    return info


def thinking_active(page) -> bool:
    return page.get_by_placeholder("QUASAR is thinking", exact=False).count() > 0


def thought_label(page) -> str:
    try:
        labels = []
        for s in page.locator("main details > summary").all():
            t = " ".join(s.inner_text(timeout=1500).split())
            if t.startswith("Thought") or t.startswith("Thinking"):
                labels.append(t)
        return labels[-1] if labels else ""
    except Exception:
        return ""


def set_web_toggle(page, want_on: bool, mode: str = None) -> str:
    """Open the + menu, set the Web Search mode, close the menu. Returns the
    final state. Phase 2 UI: a segmented control (menuitemradio off|auto|always,
    data-mode); "on" maps to auto. The Phase 1 on/off switch (menuitemcheckbox)
    is still handled so the driver runs against the older UI."""
    want_mode = (mode or ("auto" if want_on else "off")).lower()
    btn = page.locator("button[title='Add files or modes']").first
    btn.click(timeout=10000)
    time.sleep(0.4)
    radios = page.locator("[data-testid='web-search-mode'] [role='menuitemradio']")
    if radios.count() > 0:
        target = page.locator(f"[data-testid='web-search-mode'] [role='menuitemradio'][data-mode='{want_mode}']").first
        if target.get_attribute("aria-checked") != "true":
            target.click(timeout=5000)
            time.sleep(0.3)
        state = None
        for i in range(radios.count()):
            r = radios.nth(i)
            if r.get_attribute("aria-checked") == "true":
                state = r.get_attribute("data-mode")
        page.keyboard.press("Escape")
        time.sleep(0.2)
        if page.locator("[data-testid='web-search-mode']").count() > 0:
            btn.click(timeout=5000)
            time.sleep(0.2)
        return state or want_mode
    item = page.get_by_role("menuitemcheckbox", name=re.compile("Web Search")).first
    state = item.get_attribute("aria-checked") == "true"
    if state != want_on:
        item.click(timeout=5000)
        time.sleep(0.3)
        state = item.get_attribute("aria-checked") == "true"
    page.keyboard.press("Escape")
    time.sleep(0.2)
    if page.get_by_role("menuitemcheckbox", name=re.compile("Web Search")).count() > 0:
        btn.click(timeout=5000)  # toggle the menu closed
        time.sleep(0.2)
    return "on" if state else "off"


def send_turn(page, text: str, probe: bool) -> dict:
    """Type one turn, send it, wait for completion. With probe=True, time the
    first appearance of the sources card and of answer text for THIS turn."""
    rec = {"text": text}
    before = page.evaluate(JS_STREAM_PROBE)
    box = page.get_by_placeholder("Ask QUASAR", exact=False).first
    box.click(timeout=15000)
    box.fill(text)
    rec["typed_matches"] = box.input_value() == text
    t0 = time.perf_counter()
    page.locator("button[type=submit]").first.click(timeout=15000)
    rec["send_clicked"] = time.strftime("%Y-%m-%d %H:%M:%S")
    deadline = t0 + TIMEOUT_S
    finished_at = None
    t_src = t_txt = None
    time.sleep(1.0)
    while time.perf_counter() < deadline:
        if probe:
            try:
                pr = page.evaluate(JS_STREAM_PROBE)
                grew = pr.get("nBlocks", 0) > before.get("nBlocks", 0)
                if t_txt is None and grew and pr.get("textLen", 0) > 0:
                    t_txt = time.perf_counter() - t0
                # A sources card from an earlier turn does not count: the graded
                # turn must add a card of its own (card count grows).
                if t_src is None and pr.get("nCards", 0) > before.get("nCards", 0):
                    t_src = time.perf_counter() - t0
            except Exception:
                pass
        if not thinking_active(page):
            cand = time.perf_counter()
            time.sleep(2.0)
            if not thinking_active(page):
                finished_at = cand
                break
        time.sleep(0.5)
    rec["ui_stopwatch_s"] = round((finished_at or time.perf_counter()) - t0, 1)
    rec["timed_out"] = finished_at is None
    rec["time_to_sources_s"] = round(t_src, 1) if t_src is not None else None
    rec["time_to_first_text_s"] = round(t_txt, 1) if t_txt is not None else None
    if rec["timed_out"]:
        try:
            page.locator("button[type=submit]").first.click(timeout=3000)  # Stop
        except Exception:
            pass
        time.sleep(1.0)
    return rec


def run_one(page, q: dict) -> dict:
    qid = q["id"]
    d = OUT / qid
    d.mkdir(parents=True, exist_ok=True)
    turns = q["turns"]
    (d / "question.txt").write_text("\n---\n".join(turns), encoding="utf-8")
    rec = {"id": qid, "category": q["category"], "expect_web": q["expect_web"], "turns": turns,
           "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    page.get_by_role("button", name="New chat").first.click(timeout=15000)
    time.sleep(1.5)
    # YAML 1.1 reads a bare `on`/`off` as a boolean; accept both spellings.
    # Phase 2: web_toggle may also be a mode (off | auto | always).
    _wt = q.get("web_toggle", "on")
    _wt_s = str(_wt).lower()
    _mode = _wt_s if _wt_s in ("off", "auto", "always") else None
    rec["web_toggle"] = set_web_toggle(page, _wt is True or _wt_s in ("on", "true", "auto", "always"), mode=_mode)
    turn_recs = []
    for i, text in enumerate(turns):
        last = i == len(turns) - 1
        off0 = backend_log_size()
        tr = send_turn(page, text, probe=True)
        tr["cards_after"] = page.evaluate(JS_STREAM_PROBE).get("nCards", 0)
        tr["backend_log_offsets"] = [off0, backend_log_size()]
        turn_recs.append(tr)
        if not last:
            a = page.evaluate(JS_ANSWER)
            (d / f"turn{i + 1}_answer.md").write_text("\n\n---\n\n".join(a["texts"]), encoding="utf-8")
            time.sleep(1.5)
    rec["turn_records"] = turn_recs
    g = turn_recs[-1]
    time.sleep(3.0)  # late cards
    rec["ui_thought_label"] = thought_label(page)
    rec["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")

    a = page.evaluate(JS_ANSWER)
    texts = [t for t in a["texts"] if t.strip()]
    graded = texts[-1] if texts else "(no answer prose rendered in the UI)"
    (d / "answer.md").write_text(graded, encoding="utf-8")
    (d / "answer_all_turns.md").write_text("\n\n---\n\n".join(texts), encoding="utf-8")
    (d / "answer.html").write_text(a["htmls"][-1] if a["htmls"] else "", encoding="utf-8")
    rec["answer_chars"] = len(graded)
    rec["ui_errors"] = a["errs"][:5]
    rec["citation_chips"] = a.get("chips") or []
    rec["from_the_web_appendix"] = bool(re.search(r"From the Web|Updated Information from the Web", graded))
    rec["harmony_leak"] = any(m in graded for m in ("<|channel|>", "<|message|>", "commentary to=functions"))
    rec["verification_block"] = bool(re.search(r"Verification", graded))
    # Phase 2: the web_decision badge under the LAST answer, and the image tiles
    try:
        badges = page.locator("main [data-testid='web-decision-badge']")
        rec["web_decision_badge"] = badges.last.inner_text(timeout=2000) if badges.count() else None
        rec["web_decision_title"] = badges.last.get_attribute("title") if badges.count() else None
    except Exception as exc:
        rec["web_decision_badge"] = f"(probe failed: {type(exc).__name__})"
        rec["web_decision_title"] = None
    try:
        rec["web_image_tiles"] = page.evaluate("() => document.querySelectorAll('main [data-testid=\"web-sources-card\"] img[alt], main .web-image-grid img, main [data-web-image]').length")
    except Exception:
        rec["web_image_tiles"] = None

    # sources card(s): expand "View all" first
    try:
        for b in page.locator("main button").filter(has_text=re.compile(r"^View all \d+ sources|Also consulted")).all():
            b.click(timeout=2000)
            time.sleep(0.3)
    except Exception:
        pass
    cards = page.evaluate(JS_SOURCES)
    rec["source_cards"] = len(cards)
    last_card = cards[-1] if cards else None
    rec["sources"] = last_card["links"] if last_card else []
    (d / "sources.json").write_text(json.dumps(cards, indent=2, ensure_ascii=False), encoding="utf-8")

    page.evaluate(JS_SET_DETAILS, False)
    time.sleep(0.5)
    rec["answer_png"] = screenshot_tall(page, d / "answer.png")

    # Citation chip probe (grounded answers): hover the first chip of the LAST
    # answer (hover card text + screenshot), then click it (the matching source
    # card must be highlighted in the grid).
    rec["chip_probe"] = None
    try:
        chips = page.locator("main .prose-answer").last.locator("[data-web-cite]")
        if chips.count() > 0:
            chip = chips.first
            cid = chip.get_attribute("data-web-cite")
            chip.scroll_into_view_if_needed(timeout=3000)
            chip.hover(timeout=3000)
            time.sleep(0.6)
            tip = page.locator("[role=tooltip]").last
            tip_text = tip.inner_text(timeout=2000) if tip.count() else ""
            open_link = tip.locator("a[href]").first if tip.count() else None
            page.screenshot(path=str(d / "chip_hover.png"), full_page=False)
            chip.click(timeout=3000)
            time.sleep(0.4)
            highlighted = page.evaluate("() => Array.from(document.querySelectorAll('[data-highlight=true]')).map(e => e.getAttribute('data-source-id'))")
            # keyboard: focus the chip, Tab into the popup's Open link, wait past
            # the close delay; the popup must still be there with focus on Open
            page.mouse.move(5, 5)
            page.evaluate("() => document.activeElement && document.activeElement.blur()")
            time.sleep(0.4)          # hover-out close delay passes; popup closed
            chip.focus()             # a real keyboard entry into the chip
            time.sleep(0.3)
            page.keyboard.press("Tab")
            time.sleep(0.5)
            keyboard = page.evaluate("""() => {
              const a = document.activeElement;
              const tip = a && a.closest('[role=tooltip]');
              return {focused_tag: a ? a.tagName : null, focused_text: a ? (a.innerText || '').trim().slice(0, 20) : null,
                      in_popup: !!tip, popup_open: document.querySelectorAll('[role=tooltip]').length};
            }""")
            rec["chip_probe"] = {
                "id": cid, "hover_text": tip_text[:400],
                "open_href": open_link.get_attribute("href") if open_link is not None and open_link.count() else None,
                "open_target": open_link.get_attribute("target") if open_link is not None and open_link.count() else None,
                "highlighted_after_click": highlighted,
                "keyboard_tab_into_popup": keyboard,
            }
            (d / "chip_probe.json").write_text(json.dumps(rec["chip_probe"], indent=2, ensure_ascii=False), encoding="utf-8")
            page.mouse.move(5, 5)
            time.sleep(0.3)
    except Exception as exc:
        rec["chip_probe"] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    rec["details_panels"] = page.evaluate(JS_SET_DETAILS, True)
    time.sleep(0.8)
    rec["steps_png"] = screenshot_tall(page, d / "steps.png")
    thoughts = page.evaluate(JS_THOUGHT)
    (d / "steps.md").write_text("\n\n===\n\n".join(thoughts) if thoughts else "(no Thought panel rendered)", encoding="utf-8")
    steps_last = thoughts[-1] if thoughts else ""
    rec["web_steps"] = [l.strip() for l in steps_last.splitlines() if re.search(r"web|Web|Searching|Reading", l)][:12]

    n_q = page.evaluate(JS_QUERIES)
    rec["show_query_buttons"] = n_q
    if n_q:
        for _ in range(3):
            btns = page.locator("main button[aria-expanded='false']").filter(has_text=re.compile(r"^Show query"))
            n = btns.count()
            if n == 0:
                break
            for _i in range(n):
                try:
                    btns.nth(0).click(timeout=3000)
                    time.sleep(0.25)
                except Exception:
                    break
        time.sleep(0.8)
        rec["queries_png"] = screenshot_tall(page, d / "queries.png")
        qtexts = page.evaluate(JS_QUERY_TEXT)
        (d / "queries.md").write_text("\n\n===\n\n".join(qtexts) if qtexts else "(query panels did not expand)", encoding="utf-8")
    else:
        (d / "queries.md").write_text('(no "Show query" panels rendered for this turn)', encoding="utf-8")

    off = g["backend_log_offsets"]
    sl = backend_log_slice(off[0], off[1])
    web_lines = [l for l in sl.splitlines() if WEB_LOG_RE.search(l)]
    (d / "backend_web_log.txt").write_text("\n".join(web_lines[:400]) or "(no web lines in the backend log slice)", encoding="utf-8")
    rec["backend_web_lines"] = len(web_lines)
    rec["backend_searched"] = any("[SEARCH ROUTER] Rout" in l or "Routing to Tavily" in l for l in web_lines)
    # Phase 2 side metrics from the backend log slice (supporting evidence)
    _before = [int(m.group(1)) for m in VERIFY_BEFORE_RE.finditer(sl)]
    _after = [int(m.group(1)) for m in VERIFY_AFTER_RE.finditer(sl)]
    rec["unsupported_before"] = _before[-1] if _before else None
    rec["unsupported_after"] = (_after[-1] if _after else (_before[-1] if _before else None))
    _pl = PLANNER_RE.findall(sl)
    rec["planner_s"] = float(_pl[-1][0]) if _pl else None
    rec["planner_need_web"] = _pl[-1][1] if _pl else None
    # "official source cited first": the first CITED source card (or the first
    # card) is on one of the question's official domains
    off = [d.lower() for d in (q.get("official_domains") or [])]
    first_href = ""
    for l in rec["sources"]:
        if l.get("id"):
            first_href = l.get("href") or ""
            break
    if not first_href and rec["sources"]:
        first_href = rec["sources"][0].get("href") or ""
    host = re.sub(r"^https?://", "", first_href).split("/")[0].lower().replace("www.", "")
    rec["first_source"] = first_href
    rec["official_first"] = (any(host == d or host.endswith("." + d) for d in off) if (off and first_href) else None)

    timing = {k: rec.get(k) for k in ("id", "web_toggle", "ui_thought_label", "started", "finished", "answer_chars",
                                      "ui_errors", "harmony_leak", "show_query_buttons", "source_cards",
                                      "citation_chips", "from_the_web_appendix", "backend_searched", "chip_probe",
                                      "web_decision_badge", "web_decision_title", "web_image_tiles", "verification_block",
                                      "unsupported_before", "unsupported_after", "planner_s", "planner_need_web",
                                      "first_source", "official_first")}
    timing["turns"] = [{k: t.get(k) for k in ("text", "ui_stopwatch_s", "timed_out", "time_to_sources_s",
                                              "time_to_first_text_s", "cards_after", "backend_log_offsets")} for t in turn_recs]
    timing["ui_stopwatch_s"] = g["ui_stopwatch_s"]
    timing["time_to_sources_s"] = g["time_to_sources_s"]
    timing["n_sources"] = len(rec["sources"])
    (d / "timing.json").write_text(json.dumps(timing, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"{qid:7s} {g['ui_stopwatch_s']:6.1f}s web={rec['web_toggle']} src_card={rec['source_cards']} "
        f"n_src={len(rec['sources'])} t_src={g['time_to_sources_s']} chips={len(rec['citation_chips'])} "
        f"appendix={rec['from_the_web_appendix']} searched(log)={rec['backend_searched']} timeout={g['timed_out']} "
        f"badge={rec.get('web_decision_badge')!r} unsupported={rec.get('unsupported_before')}->{rec.get('unsupported_after')} "
        f"official_first={rec.get('official_first')} planner={rec.get('planner_s')}s")
    return rec


def main():
    login = get_login()
    auth = json.dumps({"state": {"user": login["user"], "token": login["token"], "isAuthenticated": True,
                                 "isInitialized": True, "isAuthModalOpen": False}, "version": 2})
    with sync_playwright() as p:
        browser = launch(p)
        ctx = browser.new_context(viewport=VIEWPORT, device_scale_factor=1)
        ctx.add_init_script(
            "localStorage.setItem('quasar-auth', %s); localStorage.setItem('quasar_onboarded','true');" % json.dumps(auth)
        )
        page = ctx.new_page()
        page.goto(FRONTEND, wait_until="networkidle", timeout=120000)
        time.sleep(2.5)
        (OUT / "run-meta.json").write_text(json.dumps({
            "started": time.strftime("%Y-%m-%d %H:%M:%S"), "frontend": FRONTEND, "backend": BACKEND,
            "backend_log": str(BACKEND_LOG), "questions_file": str(QFILE), "timeout_s": TIMEOUT_S,
        }, indent=2), encoding="utf-8")
        for q in QUESTIONS:
            d = OUT / q["id"]
            if (d / "timing.json").exists() and not ONLY:
                log(f"{q['id']}: already captured, skipping")
                continue
            try:
                rec = run_one(page, q)
            except Exception as exc:
                rec = {"id": q["id"], "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                       "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
                log(f"{q['id']}: DRIVER ERROR {rec['error']}")
                try:
                    d.mkdir(parents=True, exist_ok=True)
                    page.screenshot(path=str(d / "driver-error.png"), full_page=True)
                    (d / "driver-error.txt").write_text(rec["error"], encoding="utf-8")
                    page.goto(FRONTEND, wait_until="networkidle", timeout=120000)
                    time.sleep(2.5)
                except Exception:
                    pass
            with RESULTS.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        browser.close()
    log("done")


if __name__ == "__main__":
    main()
