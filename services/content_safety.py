"""Strict safety filtering for untrusted general-web search content.

The policy is intentionally conservative:
- known adult-content hosts are always blocked;
- results with strong explicit-content signals are blocked;
- when one result establishes an explicit search context, all source cards and
  web image tiles for that search are withheld;
- general-web image tiles are enabled by default, filtered by metadata, and can
  be disabled with QUASAR_WEB_IMAGES_ENABLED=false.

Archive images, uploaded images, and locally rendered scientific images do not
pass through this module.
"""

from __future__ import annotations

import copy
import os
import re
from typing import Any, Dict, Iterable, List
from urllib.parse import urlparse


FILTER_NOTICE = "Some web results were withheld by the safety filter."

_FALSE_VALUES = {"0", "false", "no", "off", "disabled"}

# Host matching is performed on DNS-label boundaries, not arbitrary URL text.
BLOCKED_HOSTS = frozenset(
    {
        "4chan.org",
        "adultfriendfinder.com",
        "bangbros.com",
        "brazzers.com",
        "chaturbate.com",
        "erome.com",
        "fansly.com",
        "hclips.com",
        "hentaihaven.xxx",
        "livejasmin.com",
        "manyvids.com",
        "motherless.com",
        "nhentai.net",
        "naughtyamerica.com",
        "onlyfans.com",
        "playboy.com",
        "pornhub.com",
        "porntrex.com",
        "redgifs.com",
        "redtube.com",
        "rule34.xxx",
        "sex.com",
        "spankbang.com",
        "stripchat.com",
        "theporndude.com",
        "xhamster.com",
        "xnxx.com",
        "xvideos.com",
        "youporn.com",
    }
)

_EXPLICIT_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:hardcore|softcore)\s+(?:porn|video|scene|content)\b",
        r"\b(?:porn|pornographic|pornography)\b",
        r"\bxxx\b",
        r"\bnsfw\b",
        r"\bhentai\b",
        r"\brule\s*34\b",
        r"\b(?:adult|porn)\s+(?:actor|actress|star|performer|film|movie|video|site|website|content|entertainment)\b",
        r"\b(?:nude|nudity|naked)\s+(?:photo|image|picture|video|scene|content|model)\b",
        r"\b(?:explicit|graphic)\s+sexual\s+(?:content|image|video|material|scene)\b",
        r"\bsexual(?:ly)?\s+explicit\b",
        r"\berotic(?:a)?\b",
        r"\bsex\s+(?:tape|video|scene|cam|chat|show|site|website)\b",
        r"\b(?:camgirl|cam-girl|webcam model)\b",
        r"\b(?:escort|hookup)\s+(?:site|service|directory)\b",
        r"\b(?:onlyfans|pornhub|xvideos|xnxx|xhamster|redtube|youporn|spankbang)\b",
    )
)

_SOURCE_TEXT_FIELDS = (
    "title",
    "name",
    "snippet",
    "content",
    "text",
    "description",
    "raw_content",
)
_IMAGE_TEXT_FIELDS = (
    "description",
    "title",
    "alt",
    "sourceTitle",
    "source_title",
    "sourcePageTitle",
    "source_page_title",
)
_TEXT_RESPONSE_FIELDS = ("answer", "content", "raw_text", "text", "summary")


def strict_web_filter_enabled() -> bool:
    return os.getenv("QUASAR_STRICT_WEB_FILTER", "true").strip().lower() not in _FALSE_VALUES


def web_images_enabled() -> bool:
    """General-web image tiles are on by default; false/0/no/off disables them."""
    return os.getenv("QUASAR_WEB_IMAGES_ENABLED", "true").strip().lower() not in _FALSE_VALUES


def _normalize_host(url: Any) -> str:
    value = str(url or "").strip()
    if not value:
        return ""
    if not value.startswith(("http://", "https://")):
        value = f"https://{value.lstrip('/')}"
    try:
        return (urlparse(value).hostname or "").lower().removeprefix("www.")
    except ValueError:
        return ""


def is_blocked_url(url: Any) -> bool:
    if not strict_web_filter_enabled():
        return False
    host = _normalize_host(url)
    if not host:
        return False
    return any(host == blocked or host.endswith(f".{blocked}") for blocked in BLOCKED_HOSTS)


def looks_explicit_text(value: Any) -> bool:
    if not strict_web_filter_enabled():
        return False
    text = str(value or "")
    if not text:
        return False
    return any(pattern.search(text) for pattern in _EXPLICIT_PATTERNS)


def is_explicit_query(query: Any) -> bool:
    """Block only strong explicit phrases; ordinary words such as 'adult' remain valid."""
    return looks_explicit_text(query)


def _combined_text(item: Dict[str, Any], fields: Iterable[str]) -> str:
    return " ".join(str(item.get(field) or "") for field in fields)


def is_safe_web_source(item: Any) -> bool:
    if not strict_web_filter_enabled():
        return True
    if isinstance(item, str):
        return not is_blocked_url(item) and not looks_explicit_text(item)
    if not isinstance(item, dict):
        return False
    url = item.get("url") or item.get("link") or item.get("href") or item.get("source_url")
    return not is_blocked_url(url) and not looks_explicit_text(_combined_text(item, _SOURCE_TEXT_FIELDS))


def is_safe_web_image(item: Any) -> bool:
    if not strict_web_filter_enabled():
        return True
    if not web_images_enabled():
        return False
    if isinstance(item, str):
        return not is_blocked_url(item) and not looks_explicit_text(item)
    if not isinstance(item, dict):
        return False
    image_url = item.get("url") or item.get("src") or item.get("image_url")
    source_url = (
        item.get("sourceUrl")
        or item.get("source_url")
        or item.get("sourcePageUrl")
        or item.get("source_page_url")
        or item.get("pageUrl")
        or item.get("page_url")
        or item.get("source")
    )
    return (
        not is_blocked_url(image_url)
        and not is_blocked_url(source_url)
        and not looks_explicit_text(_combined_text(item, _IMAGE_TEXT_FIELDS))
    )


def blocked_query_result(query: Any, provider: str = "Safety filter") -> Dict[str, Any]:
    return {
        "success": True,
        "provider": provider,
        "query": str(query or ""),
        "answer": FILTER_NOTICE,
        "results": [],
        "sources": [],
        "images": [],
        "content_filter": {
            "enabled": True,
            "filtered": True,
            "blocked_query": True,
            "blocked_results": 0,
            "blocked_images": 0,
            "blocked_text_fields": 0,
            "notice": FILTER_NOTICE,
        },
    }


def _filter_item_list(items: Any) -> tuple[Any, int, bool]:
    if not isinstance(items, list):
        return items, 0, False
    safe_items: List[Any] = []
    blocked = 0
    explicit_context = False
    for item in items:
        if is_safe_web_source(item):
            safe_items.append(item)
        else:
            blocked += 1
            explicit_context = True
    return safe_items, blocked, explicit_context


def _sanitize_mapping(mapping: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, int], bool]:
    clean = dict(mapping)
    counts = {"blocked_results": 0, "blocked_images": 0, "blocked_text_fields": 0}
    explicit_context = False

    for key in ("results", "sources", "urls"):
        if key not in clean:
            continue
        filtered, blocked, found_explicit = _filter_item_list(clean.get(key))
        clean[key] = filtered
        counts["blocked_results"] += blocked
        explicit_context = explicit_context or found_explicit

    images = clean.get("images")
    if isinstance(images, list):
        safe_images = [item for item in images if is_safe_web_image(item)]
        counts["blocked_images"] += len(images) - len(safe_images)
        clean["images"] = safe_images

    for key in _TEXT_RESPONSE_FIELDS:
        value = clean.get(key)
        if isinstance(value, str) and looks_explicit_text(value):
            clean[key] = ""
            counts["blocked_text_fields"] += 1
            explicit_context = True

    nested = clean.get("response")
    if isinstance(nested, dict):
        nested_clean, nested_counts, nested_explicit = _sanitize_mapping(nested)
        clean["response"] = nested_clean
        for key, value in nested_counts.items():
            counts[key] += value
        explicit_context = explicit_context or nested_explicit

    return clean, counts, explicit_context


def sanitize_web_payload(payload: Any) -> Any:
    """Return a filtered copy of a provider/tool payload."""
    if not strict_web_filter_enabled() or not isinstance(payload, dict):
        return payload

    prior_filter = payload.get("content_filter")
    clean, counts, explicit_context = _sanitize_mapping(copy.deepcopy(payload))

    # If any result establishes an explicit context, metadata cannot prove the
    # remaining cards or images are safe for this query. Withhold the whole set.
    if explicit_context:
        for key in _TEXT_RESPONSE_FIELDS:
            if isinstance(clean.get(key), str) and clean[key]:
                clean[key] = ""
                counts["blocked_text_fields"] += 1
        for key in ("results", "sources", "urls", "images"):
            value = clean.get(key)
            if isinstance(value, list):
                if key == "images":
                    counts["blocked_images"] += len(value)
                else:
                    counts["blocked_results"] += len(value)
                clean[key] = []
        nested = clean.get("response")
        if isinstance(nested, dict):
            for key in ("results", "sources", "urls", "images"):
                value = nested.get(key)
                if isinstance(value, list):
                    if key == "images":
                        counts["blocked_images"] += len(value)
                    else:
                        counts["blocked_results"] += len(value)
                    nested[key] = []
            for key in _TEXT_RESPONSE_FIELDS:
                if isinstance(nested.get(key), str) and nested[key]:
                    nested[key] = ""
                    counts["blocked_text_fields"] += 1

    prior_filtered = isinstance(prior_filter, dict) and bool(prior_filter.get("filtered"))
    if isinstance(prior_filter, dict):
        for key in counts:
            try:
                counts[key] = max(counts[key], int(prior_filter.get(key) or 0))
            except (TypeError, ValueError):
                pass

    filtered = prior_filtered or explicit_context or any(counts.values())
    if filtered:
        has_safe_material = any(
            isinstance(clean.get(key), list) and clean[key]
            for key in ("results", "sources", "images")
        ) or any(
            isinstance(clean.get(key), str) and clean[key].strip()
            for key in _TEXT_RESPONSE_FIELDS
        )
        if not has_safe_material:
            clean["answer"] = FILTER_NOTICE

    clean["content_filter"] = {
        "enabled": True,
        "filtered": filtered,
        "blocked_query": bool(prior_filter.get("blocked_query")) if isinstance(prior_filter, dict) else False,
        **counts,
        "notice": (
            str(prior_filter.get("notice") or FILTER_NOTICE)
            if filtered and isinstance(prior_filter, dict)
            else FILTER_NOTICE if filtered else ""
        ),
        "web_images_enabled": web_images_enabled(),
    }
    return clean


def safe_assistant_text(text: Any) -> str:
    """Last-resort guard for complete, non-streaming generated text."""
    value = str(text or "")
    return FILTER_NOTICE if looks_explicit_text(value) else value
