"""UI presentation / serialization helpers extracted from ``api/main.py``.

These modules turn agent/tool results and archive responses into the exact
JSON payload shapes the frontend expects. They hold the hardcoded per-archive
column maps (``data_card``), the CADC VOTable/DataLink parsing (``cadc``), and
the matplotlib FITS preview rendering (``fits``). Kept transport-agnostic:
they take plain data in and return plain dicts/tuples out.
"""
