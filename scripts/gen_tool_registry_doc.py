#!/usr/bin/env python3
"""
gen_tool_registry_doc.py — generate ``docs/TOOLS.md`` from the LIVE tool registry.

The tool count has historically drifted across hand-written docs (README said
"75+", the old ARCHITECTURE.md said "35+", implementation_plan.md said "27+"),
while the code actually registers ~140. This script kills that drift by making
the tool reference a pure function of the code: it constructs a ``QuasarAgent``,
reads ``agent.tool_registry.list_tools()``, and renders a deterministic Markdown
table grouped by category.

Usage
-----
    python scripts/gen_tool_registry_doc.py            # (re)write docs/TOOLS.md
    python scripts/gen_tool_registry_doc.py --check    # CI: fail if docs/TOOLS.md is stale
    python scripts/gen_tool_registry_doc.py --stdout    # print to stdout, write nothing

Determinism
-----------
The rendered document depends ONLY on each tool's name / category / description /
parameter schema — all hard-coded in ``core/agent.py``. Before importing the
agent we pin a canonical, offline environment (in-memory Qdrant, dummy keys) so
construction never touches the network and produces the same 140 tools in CI as
on a dev box. No timestamp is written, so re-running is a no-op when nothing
changed — which is exactly what ``--check`` relies on.
"""

import argparse
import contextlib
import io
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOC_PATH = REPO_ROOT / "docs" / "TOOLS.md"
REL_DOC_PATH = "docs/TOOLS.md"
GEN_CMD = "python scripts/gen_tool_registry_doc.py"

# A construction that yields far fewer tools than this almost certainly means a
# broken/partial environment (a service import silently dropped a family). Fail
# loudly instead of writing a truncated doc or passing a bogus --check.
MIN_EXPECTED_TOOLS = 100


def _pin_offline_env() -> None:
    """Pin a canonical, network-free environment so the registry is reproducible
    everywhere. Must run BEFORE ``core.agent`` (and its service singletons) are
    imported."""
    # Force services/vector_db.py onto its in-memory Qdrant fallback (it goes to
    # the cloud only when BOTH of these are set) — no network, no hang. We SET
    # them empty rather than pop, so a later ``load_dotenv()`` in the import
    # chain (config/__init__.py, override=False) can't re-populate them from a
    # dev ``.env`` and drag us back onto the network.
    os.environ["QDRANT_URL"] = ""
    os.environ["QDRANT_API_KEY"] = ""
    # A non-empty OpenAI key + an OpenAI model satisfy the constructor's provider
    # guard without ever calling out (tool *registration* never hits the API).
    os.environ.setdefault("OPENAI_API_KEY", "sk-doc-gen-dummy")
    os.environ.setdefault("NASA_ADS_API_KEY", "doc-gen-dummy")
    os.environ["DEFAULT_LLM_MODEL"] = "gpt-4.1"
    # Keep the native inline path. The capability adapter only swaps an
    # implementation (never a tool's name/schema/category), so the rendered doc
    # is invariant to it — pin it off anyway for a clean, reproducible run.
    os.environ["QUASAR_USE_CAPABILITY_DATALAB"] = ""


def _build_registry():
    """Construct a QuasarAgent and return its populated ToolRegistry.

    Agent construction is chatty (many ``DEBUG:`` prints); swallow stdout so the
    generator's own output stays clean. Warnings/errors still go to stderr.
    """
    _pin_offline_env()
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        from core.agent import QuasarAgent, AgentConfig
        agent = QuasarAgent(AgentConfig())
    return agent.tool_registry


def _fmt_params(parameters: dict) -> str:
    """One-line parameter summary: ``name:type*`` (``*`` = required)."""
    if not isinstance(parameters, dict):
        return "—"
    props = parameters.get("properties") or {}
    if not props:
        return "—"
    required = set(parameters.get("required") or [])
    parts = []
    for name in props:  # preserve schema order (mirrors the source definition)
        spec = props[name] if isinstance(props[name], dict) else {}
        ptype = spec.get("type")
        if isinstance(ptype, list):
            ptype = "|".join(str(t) for t in ptype)
        ptype = ptype or "any"
        parts.append(f"`{name}`:{ptype}" + ("\\*" if name in required else ""))
    return ", ".join(parts)


def _clean(text: str) -> str:
    """Collapse a description to a single escaped table-cell line."""
    text = " ".join((text or "").split())
    return text.replace("|", "\\|")


def render(registry) -> str:
    tools = registry.list_tools()
    # Deterministic ordering: category, then tool name.
    tools = sorted(tools, key=lambda t: (t.category or "", t.name))

    by_cat: dict[str, list] = {}
    for t in tools:
        by_cat.setdefault(t.category or "general", []).append(t)

    lines: list[str] = []
    lines.append("# Quasar Tool Registry")
    lines.append("")
    lines.append(
        f"> **Auto-generated — do not edit by hand.** Regenerate with "
        f"`{GEN_CMD}`.  \n"
        f"> This file is the single source of truth for the tool count; CI fails "
        f"if it drifts from the code (`{GEN_CMD} --check`)."
    )
    lines.append("")
    lines.append(
        f"Quasar registers **{len(tools)} tools** across **{len(by_cat)} "
        f"categories** on the native in-process path (per-user MCP and custom "
        f"tools are added at runtime and are not listed here)."
    )
    lines.append("")

    # Summary table.
    lines.append("| Category | Tools |")
    lines.append("|---|---|")
    for cat in sorted(by_cat):
        lines.append(f"| `{cat}` | {len(by_cat[cat])} |")
    lines.append(f"| **Total** | **{len(tools)}** |")
    lines.append("")

    # Per-category detail.
    for cat in sorted(by_cat):
        cat_tools = by_cat[cat]
        lines.append(f"## `{cat}` ({len(cat_tools)})")
        lines.append("")
        lines.append("| Tool | Parameters | Description |")
        lines.append("|---|---|---|")
        for t in cat_tools:
            lines.append(
                f"| `{t.name}` | {_fmt_params(t.parameters)} | "
                f"{_clean(t.description)} |"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Exit non-zero if docs/TOOLS.md is missing or stale "
                             "(does not write). For CI.")
    parser.add_argument("--stdout", action="store_true",
                        help="Print the document to stdout and write nothing.")
    args = parser.parse_args()

    registry = _build_registry()
    n = len(registry.list_tools())
    if n < MIN_EXPECTED_TOOLS:
        print(f"[gen_tool_registry_doc] ERROR: only {n} tools registered "
              f"(expected >= {MIN_EXPECTED_TOOLS}); refusing to write a "
              f"truncated doc. Environment likely broke agent construction.",
              file=sys.stderr)
        return 2

    content = render(registry)

    if args.stdout:
        sys.stdout.write(content)
        return 0

    if args.check:
        if not DOC_PATH.exists():
            print(f"[gen_tool_registry_doc] {REL_DOC_PATH} is missing. "
                  f"Run `{GEN_CMD}` and commit it.", file=sys.stderr)
            return 1
        current = DOC_PATH.read_text(encoding="utf-8")
        if current != content:
            import difflib
            diff = difflib.unified_diff(
                current.splitlines(keepends=True),
                content.splitlines(keepends=True),
                fromfile=f"{REL_DOC_PATH} (committed)",
                tofile=f"{REL_DOC_PATH} (from code)",
            )
            sys.stderr.writelines(list(diff)[:60])
            print(f"\n[gen_tool_registry_doc] {REL_DOC_PATH} is STALE "
                  f"({n} tools in code). Run `{GEN_CMD}` and commit the result.",
                  file=sys.stderr)
            return 1
        print(f"[gen_tool_registry_doc] {REL_DOC_PATH} is up to date "
              f"({n} tools).")
        return 0

    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text(content, encoding="utf-8")
    print(f"[gen_tool_registry_doc] wrote {REL_DOC_PATH} ({n} tools).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
