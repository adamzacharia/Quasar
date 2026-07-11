# Quasar V2 — Documentation Set (START HERE)

This folder is the **self-contained handoff** for Quasar V2 implementation. It was written so a fresh session/agent can continue the work **without any memory of the previous discussion**. Read `NEXT_SESSION_START_HERE.md` first, then the rest in the order below.

## The one decision that governs everything

> **Build the MCP path first, but keep BOTH paths.** The final system supports MCP **and** the existing direct-tool approach. MCP comes first because it is the lever that breaks the monolith bottleneck. The tool-based path stays available and gets improved later. **Highest priority = solving the monolith bottleneck.**

**Key reframing (adopted this session):** the monolith split and the "shared core both paths use" are the **same work**. Extracting capability logic out of `core/agent.py` (12,714 lines) into a shared **capabilities layer** *is* the de-monolith, and it is exactly what lets MCP and native tools share logic instead of duplicating it. So Priority #1 (monolith) and the dual-path requirement are one coherent effort — see `SHARED_CORE_ARCHITECTURE.md`.

## Files in this set (reading order)

| # | File | What it gives you |
|---|---|---|
| −1 | **`HANDOFF_NEXT_SESSION.md`** | ⚠️ **2026-07-09 — OPEN THIS FIRST.** Current state (P0 done, P1 in progress), what all 4 parallel sessions did + handed off, the proven capability-migration recipe, and the exact ordered next steps. Supersedes Step 3 of the file below. |
| 0 | **`NEXT_SESSION_START_HERE.md`** | Practical starting guide: what to read, what to inspect, first tasks, exact order. |
| 1 | **`SESSION_CONTEXT.md`** | Full overview of the previous session: project goal, current state, all major decisions, why MCP-first-but-keep-tools. |
| 2 | **`IMPLEMENTATION_PRIORITIES.md`** | The ordered priority list. Monolith is #1. Then MCP, tool compatibility, testing, cleanup, docs. |
| 3 | **`MONOLITH_BOTTLENECK_ANALYSIS.md`** | The monolith problem in detail: which files, what responsibilities are tangled, target module structure. |
| 4 | **`SHARED_CORE_ARCHITECTURE.md`** | The ports-and-adapters design: one capability = one function, exposed through a native adapter AND an MCP adapter. What is shared / MCP-specific / tool-specific. |
| 5 | **`MCP_IMPLEMENTATION_PLAN.md`** | How to build the MCP path first: server structure, registration, schemas, request/response flow, error handling, logging, testing, the bridge fix. |
| 6 | **`TOOLS_IMPLEMENTATION_PLAN.md`** | How the direct-tool path is preserved and improved on top of the shared core (no duplication). |
| 7 | **`FILE_AND_CODE_MAP.md`** | Every relevant file/folder/script/log/doc: what it does, why it matters, whether to modify. |
| 8 | **`OPEN_ISSUES_AND_TODOS.md`** | Unresolved issues, bugs, design questions, missing tests, uncertain assumptions + how to verify each. |
| 9 | **`STATUS.md`** | **Living** progress tracker. Update this after every meaningful change. |

## Backing evidence (the deep analysis behind these docs)

These docs distill — they do **not** replace — the exhaustive source material. When you need more depth than a doc gives:

- **`../QUASAR_V2_DESIGN.md`** — the full architecture design doc (10 questions, diagrams, roadmap).
- **`../QUASAR_V2_EXECUTION_PLAN.md`** — the 7-phase step-by-step execution plan with acceptance criteria.
- **`C:/Users/adama/Desktop/France/.research/data/400_quasar_*.md` … `409_*.md`** — the 10 raw whole-repo subsystem analyses (backend, core, services, integrations, rag/memory, frontend, agents/notebook, benchmark, docs, ALMA_MCP). These contain the exhaustive tool lists, exact weaknesses, and file:line references. **This is the ground truth.**
- **`C:/Users/adama/Desktop/France/.research/`** — 281-file Astro Data Lab knowledge base (for MANNA/archive work).

## Verified facts (checked against live code 2026-07-08)

- `core/agent.py` = **12,714 lines**; `ui-pro/api/main.py` = **4,368 lines**.
- **145** `Tool(`/register call sites in `agent.py` (README claims "75+", paper "35+" — both stale; treat ~142 as the real number).
- `_filter_results` is **defined twice** (`agent.py:8397` and `10147`) — a concrete symptom of the monolith.
- MCP bridge uses the broken per-call `asyncio.new_event_loop()` pattern at `agent.py:522–523` and `11671–11674`.
- `agents/` has 7 sub-agent files (`archive_agent.py`, etc.) but is **imported nowhere** outside itself → dead code.
- MCP plumbing already exists: `services/mcp_server_service.py` + `/api/mcp-servers` routes (`main.py:2683–2702`).
- `services/` = **74** modules; `integrations/` = **15**; **179** `os.getenv/environ` calls; `config/settings.py` imported **nowhere** (dead).

Line numbers not in this verified list are marked "**(approx — verify)**" throughout; verify with `grep -n`.
