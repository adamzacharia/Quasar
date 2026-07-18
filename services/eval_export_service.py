"""Labeled-dataset export: block ratings joined to their turn (Feature 4).

Shape: a HUMAN-LABEL SIDECAR. Each row is one human rating of one block,
carrying enough of its turn (prompt, the block's own content and the exact
query behind it, model, tokens, cost) to score a DataLabBench run against
human judgement — rather than proposing new benchmark questions.

Why the join is in Python and not in SQL
----------------------------------------
The three tables live in three DIFFERENT databases in development:
`analytics_service` opens data/analytics.db, `conversation_service` opens
data/conversations.db, `issue_report_service` opens data/issue_reports.db.
They only collapse onto one database when TURSO_DATABASE_URL is set (see
services/db.py::get_connection). A `JOIN block_feedback ... chat_runs` would
therefore pass in production and fail with "no such table" locally — including
under the unit tests. Joining in Python is correct on both.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

# A rated block's content is a SUMMARY, not the payload: a 10k-row table or a
# base64 PNG would make the export unusable as a label file.
_MAX_PAPERS_LISTED = 20
_MAX_TEXT_CHARS = 4000


def _find_block(metadata: Dict[str, Any], block_id: str) -> Optional[Tuple[str, Dict[str, Any], Any]]:
    """Locate `block_id` inside one assistant message's rich_meta.

    Returns (block_kind, content_summary, request) or None. Mirrors exactly
    where sse.py stamps each id, so the two must change together.
    """
    if not isinstance(metadata, dict):
        return None

    # text — the prose block, whose id rides on runMeta
    run_meta = metadata.get("runMeta")
    if isinstance(run_meta, dict) and run_meta.get("text_block_id") == block_id:
        return ("text", {}, None)

    # data — dataTables is the full list; dataTable is the back-compat primary
    tables = metadata.get("dataTables")
    if not isinstance(tables, list) or not tables:
        single = metadata.get("dataTable")
        tables = [single] if isinstance(single, dict) else []
    for table in tables:
        if isinstance(table, dict) and table.get("blockId") == block_id:
            content = {
                "columns": table.get("columns") or [],
                "source_name": table.get("sourceName") or "",
                "total_rows": table.get("totalRows"),
                "displayed_rows": table.get("displayedRows"),
                "truncated": bool(table.get("truncated")),
                "result_id": table.get("resultId"),
            }
            return ("data", content, table.get("request"))

    # papers — one id per grid. papersGroups holds every grid of a turn that
    # searched twice; papers/papersBlockId are only the back-compat primary.
    paper_groups = metadata.get("papersGroups")
    if not isinstance(paper_groups, list) or not paper_groups:
        paper_groups = [{
            "papers": metadata.get("papers"),
            "request": metadata.get("papersRequest"),
            "blockId": metadata.get("papersBlockId"),
        }] if metadata.get("papersBlockId") else []
    for group in paper_groups:
        if not isinstance(group, dict) or group.get("blockId") != block_id:
            continue
        papers = group.get("papers")
        listed = papers[:_MAX_PAPERS_LISTED] if isinstance(papers, list) else []
        content = {
            "paper_count": len(papers) if isinstance(papers, list) else 0,
            "papers": [
                {
                    "title": p.get("title"),
                    "bibcode": p.get("bibcode"),
                    "year": p.get("year"),
                }
                for p in listed
                if isinstance(p, dict)
            ],
        }
        return ("papers", content, group.get("request"))

    # image / plotly — figures, keyed by url in the images list
    images = metadata.get("images")
    if not isinstance(images, list) or not images:
        single_img = metadata.get("image")
        images = [single_img] if isinstance(single_img, dict) else []
    for img in images:
        if isinstance(img, dict) and img.get("blockId") == block_id:
            content = {"caption": img.get("caption") or "", "url": img.get("url") or ""}
            return (img.get("blockKind") or "image", content, img.get("request"))

    # notebook — `notebooks` holds every one of a multi-notebook turn (Conductor
    # emits its own alongside the normal path); `notebook` is the last of them.
    notebooks = metadata.get("notebooks")
    if not isinstance(notebooks, list) or not notebooks:
        single_nb = metadata.get("notebook")
        notebooks = [single_nb] if isinstance(single_nb, dict) else []
    for notebook in notebooks:
        if not isinstance(notebook, dict) or notebook.get("blockId") != block_id:
            continue
        nb_data = notebook.get("data")
        cells = nb_data.get("cells") if isinstance(nb_data, dict) else None
        content = {
            "title": notebook.get("title") or "",
            "cell_count": len(cells) if isinstance(cells, list) else 0,
        }
        return ("notebook", content, None)

    return None


def _message_metadata(msg: Dict[str, Any]) -> Dict[str, Any]:
    meta = msg.get("metadata")
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except (TypeError, ValueError):
            return {}
    return meta if isinstance(meta, dict) else {}


def _locate(messages: List[Dict[str, Any]], block_id: str):
    """Find the block and the user prompt that provoked it.

    The prompt is the nearest preceding user message — conversation_service
    returns messages ordered by created_at, so "nearest preceding" is just a
    backward scan.
    """
    for idx, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        found = _find_block(_message_metadata(msg), block_id)
        if not found:
            continue
        kind, content, request = found
        if kind == "text":
            content = {"text": (msg.get("content") or "")[:_MAX_TEXT_CHARS]}
        prompt = ""
        for prev in range(idx - 1, -1, -1):
            if messages[prev].get("role") == "user":
                prompt = messages[prev].get("content") or ""
                break
        return kind, content, request, prompt, (msg.get("content") or "")[:_MAX_TEXT_CHARS]
    return None


def build_eval_export(
    analytics_service,
    conversation_service,
    issue_report_service,
    user_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Join block ratings -> turn record -> persisted block content.

    `user_id` scopes to a single rater; None exports every rater's labels.
    A rating whose conversation or run has since been deleted still exports —
    with nulls for the missing side — because the label itself is the artifact
    worth keeping.
    """
    ratings = analytics_service.export_block_feedback(user_id=user_id)

    messages_cache: Dict[str, List[Dict[str, Any]]] = {}
    runs_cache: Dict[str, Optional[Dict[str, Any]]] = {}
    rows: List[Dict[str, Any]] = []

    for rating in ratings:
        conv_id = rating.get("conversation_id") or ""
        run_id = rating.get("run_id") or ""
        block_id = rating.get("block_id") or ""

        if conv_id and conv_id not in messages_cache:
            try:
                messages_cache[conv_id] = conversation_service.get_conversation_messages(conv_id)
            except Exception:
                messages_cache[conv_id] = []
        messages = messages_cache.get(conv_id) or []

        if run_id and run_id not in runs_cache:
            try:
                runs_cache[run_id] = issue_report_service.get_run(run_id)
            except Exception:
                runs_cache[run_id] = None
        run = runs_cache.get(run_id) or {}

        located = _locate(messages, block_id) if messages else None
        if located:
            kind, content, request, prompt, answer = located
        else:
            # The block's message is gone (deleted conversation, or a turn that
            # predates stable ids). Fall back to what the rating itself knows.
            kind, content, request, prompt, answer = (
                rating.get("block_kind") or "", {}, None, "", "",
            )

        rows.append({
            "block_id": block_id,
            "rating": rating.get("rating"),
            "comment": rating.get("comment") or "",
            "rated_at": rating.get("timestamp"),
            "rated_by": rating.get("user_id"),
            "conversation_id": conv_id,
            "run_id": run_id,
            "prompt": prompt,
            "block_kind": kind or rating.get("block_kind") or "",
            "block_content": content,
            "block_query": request,
            "answer_preview": answer,
            "model": run.get("model") or rating.get("model") or "",
            "provider": run.get("provider") or "",
            "run_status": run.get("status") or "",
            "tools_called": run.get("tools_called") or [],
            "input_tokens": run.get("input_tokens"),
            "output_tokens": run.get("output_tokens"),
            "total_tokens": run.get("total_tokens"),
            "cost_usd": run.get("cost_usd"),
            "block_resolved": bool(located),
        })

    return rows


def to_jsonl(rows: List[Dict[str, Any]]) -> str:
    """One JSON object per line — the format DataLabBench reads."""
    return "\n".join(json.dumps(r, default=str) for r in rows)
