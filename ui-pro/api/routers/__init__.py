"""FastAPI route modules for the QUASAR API.

Each module defines an ``APIRouter`` grouped by domain; ``api/main.py`` mounts
them all with ``app.include_router(...)``. Extracted from the former monolithic
``main.py`` during the P1 split — endpoint paths and behaviour are unchanged.
"""
