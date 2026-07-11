"""CADC DataLink preview-URL fetching + VOTable XML parsing.

Extracted verbatim from ``api/main.py``. Pure network+XML helper: given a batch
of ``obs_publisher_did`` values, query the CADC DataLink service and parse the
VOTable response for ``#preview`` / ``#thumbnail`` access URLs. Non-blocking:
returns an empty dict on any error.
"""

from typing import Dict, List

# ── CADC DataLink preview URL fetcher ─────────────────────────
_CADC_DATALINK_URL = "https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/caom2ops/datalink"


def _fetch_cadc_preview_urls(obs_publisher_dids: List[str], max_ids: int = 200) -> Dict[str, str]:
    """Batch-query CADC DataLink for preview image URLs.

    Sends a single HTTP request with multiple IDs and parses the VOTable
    response to extract rows with semantics=#preview or #thumbnail.

    Returns {obs_publisher_did: preview_access_url} mapping.
    Falls back to empty dict on any error (non-blocking).
    """
    import urllib.request
    import urllib.parse
    import xml.etree.ElementTree as ET

    ids = obs_publisher_dids[:max_ids]
    if not ids:
        return {}

    try:
        params = "&".join(f"ID={urllib.parse.quote(oid, safe='')}" for oid in ids)
        url = f"{_CADC_DATALINK_URL}?{params}"
        req = urllib.request.Request(url, headers={"Accept": "application/x-votable+xml"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = resp.read()

        # Parse VOTable XML — extract ID, semantics, access_url columns
        root = ET.fromstring(data)
        ns = {"vo": "http://www.ivoa.net/xml/VOTable/v1.3"}

        # Find FIELD column indices
        table = root.find(".//vo:TABLE", ns) or root.find(".//TABLE")
        if table is None:
            # Try without namespace
            ns = {}
            table = root.find(".//TABLE")
        if table is None:
            return {}

        fields = table.findall("vo:FIELD", ns) if ns else table.findall("FIELD")
        col_names = [f.get("name", "").lower() for f in fields]

        id_idx = next((i for i, n in enumerate(col_names) if n == "id"), None)
        sem_idx = next((i for i, n in enumerate(col_names) if n == "semantics"), None)
        url_idx = next((i for i, n in enumerate(col_names) if n == "access_url"), None)

        if id_idx is None or sem_idx is None or url_idx is None:
            return {}

        preview_map: Dict[str, str] = {}
        data_el = table.find("vo:DATA", ns) if ns else table.find("DATA")
        if data_el is None:
            return {}
        tabledata = data_el.find("vo:TABLEDATA", ns) if ns else data_el.find("TABLEDATA")
        if tabledata is None:
            return {}

        for tr in (tabledata.findall("vo:TR", ns) if ns else tabledata.findall("TR")):
            tds = tr.findall("vo:TD", ns) if ns else tr.findall("TD")
            if len(tds) <= max(id_idx, sem_idx, url_idx):
                continue
            obs_id = (tds[id_idx].text or "").strip()
            semantics = (tds[sem_idx].text or "").strip().lower()
            access = (tds[url_idx].text or "").strip()

            if semantics in ("#preview", "#thumbnail") and access:
                # Prefer #thumbnail (smaller), but #preview is fine too
                if obs_id not in preview_map or semantics == "#thumbnail":
                    preview_map[obs_id] = access

        return preview_map

    except Exception as e:
        print(f"[INFO] CADC DataLink preview fetch skipped: {e}")
        return {}
