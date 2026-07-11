"""Import-time environment bootstrap for the QUASAR API package.

This module centralises the process-level fixups that used to sit at the very
top of ``api/main.py``. It MUST be importable and runnable before any
``from services.*`` / ``from core.*`` import happens, because some of those
transitively import Unix-only modules (``pwd``) or the removed stdlib ``cgi``
module on Python 3.13.

Importing this module is idempotent — every guard checks before mutating global
state — so ``from api import bootstrap`` can appear at the top of every module in
the package (main, deps, sse, serializers) without side effects on repeat imports.

It also makes the ``api`` package self-locating: it puts both the repository root
(for ``core`` / ``services`` / ``utils``) and the ``ui-pro`` dir (for the ``api``
namespace package itself) on ``sys.path`` so the modules are importable no matter
the current working directory or how ``main.py`` was loaded (uvicorn, or a test
loading it via ``importlib.util.spec_from_file_location``).
"""

import os
import sys
from pathlib import Path

# ── Windows fix: cp1252 stdout/stderr crash the agent loop when a print()
# carries non-Latin-1 text (live 2026-07-05: a tool-call log with U+2212 from
# the user's prompt raised UnicodeEncodeError mid-turn and killed the round).
for _stream in (sys.stdout, sys.stderr):
    try:
        # line_buffering: stdout redirected to a log file is otherwise 8KB
        # block-buffered, so diagnostic prints lag minutes behind reality.
        _stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

# ── Windows fix: langchain_community.document_loaders.pebblo imports 'pwd' (Unix-only) ──
if sys.platform == "win32" and "pwd" not in sys.modules:
    import types
    _pwd_stub = types.ModuleType("pwd")
    _pwd_stub.getpwuid = lambda uid: type("pw", (), {"pw_name": "user"})()
    sys.modules["pwd"] = _pwd_stub

# ── Python 3.13 fix for 'cgi' module missing (needed by older pyvo) ──
if "cgi" not in sys.modules:
    import email.message
    import types
    _cgi_stub = types.ModuleType("cgi")

    def _parse_header(line):
        m = email.message.EmailMessage()
        m['content-type'] = line
        return m.get_content_type(), m.get_params() or {}

    _cgi_stub.parse_header = _parse_header
    sys.modules["cgi"] = _cgi_stub

# ── sys.path: make imports resolve regardless of CWD / how we were loaded ──
# ``bootstrap.py`` lives at ``<repo>/ui-pro/api/bootstrap.py``.
_API_DIR = Path(__file__).resolve().parent          # <repo>/ui-pro/api
UI_PRO_DIR = str(_API_DIR.parent)                    # <repo>/ui-pro
PROJECT_ROOT = str(_API_DIR.parent.parent)           # <repo>

# Repo root goes first so ``core`` / ``services`` / ``utils`` / ``tests`` win.
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# ui-pro is only needed so the ``api`` namespace package is discoverable; it is
# appended (low priority) so it can never shadow a repo-root package.
if UI_PRO_DIR not in sys.path:
    sys.path.append(UI_PRO_DIR)

# ── Load .env from the project root so OPENAI_API_KEY etc. are available ──
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
except Exception:
    pass
