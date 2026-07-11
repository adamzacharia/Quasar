# SESSION_CONTEXT.md — Full context from the previous session

**Purpose:** everything the next session needs to know about *why we are here*. Assume the reader remembers nothing.

## 1. What Quasar is

Quasar is an open-source, source-available AI research assistant for astronomy archive workflows. V1 is ALMA-focused and live at **quasarassistant.com** (repo `github.com/adamzacharia/Quasar`, gitlab mirror `gitlab.com/adamzacharia/Quasar`). Local working copy: **`C:/Users/adama/Desktop/Quasar-main`**. There is a submitted PAI26 paper ("Quasar: An AI-Powered Research Assistant for Astronomy Archive Workflows") and a poster (NSF-Simons AI Institute for Cosmic Origins).

**Architecture (V1, as it actually runs):** a `QuasarAgent` (`core/agent.py`) assembles context from a shared RAG store (ALMA Technical Handbook + Proposer's Guide in Qdrant), per-user document collections, and preference memory; a two-stage complexity detector routes **simple** queries to a direct Agent Tool Loop and **complex** queries to a **Conductor** that decomposes them into a DAG dispatched to specialist sub-agents (Archive/Literature/Analysis); a sandboxed Python REPL handles precise computation; every workflow is meant to export a reproducible Jupyter notebook. Frontend is Next.js; backend is FastAPI with SSE streaming, user accounts, BYOK provider keys, and weekly quotas.

## 2. Related project: MANNA

The same broader team (STABLE / NRAO+NOIRLab; lead Dan Gause) is building **MANNA**, a **FastMCP** server that exposes NOIRLab Astro Data Lab + NRAO/ALMA archives to LLMs over **IVOA standards (TAP/SIA/SCS)**, with async jobs and a per-archive "quirks" knowledge base, intended to run behind a self-hosted LLM (vLLM on the on-prem GPU box `dlai1`) and surfaced via `jupyter-ai` on the Data Lab notebook server `gp13`. **MANNA is the intended first/canonical MCP server Quasar will consume.** A 281-file Astro Data Lab knowledge base was built and lives at `C:/Users/adama/Desktop/France/.research/`. A standalone proof-of-concept `ALMA_MCP` FastMCP server also exists at `C:/Users/adama/Desktop/ALMA_MCP`.

## 3. What the previous session actually did

1. **Whole-repo analysis of Quasar V1** — 10 parallel readers analyzed every subsystem (backend, core/agent/conductor, services, integrations, RAG/memory, frontend, agents/notebook, benchmark, docs, ALMA_MCP). Raw output saved to `France/.research/data/400_quasar_*.md` … `409_*.md`. **This is the evidence base for all V2 planning.**
2. **Wrote `docs/QUASAR_V2_DESIGN.md`** — a full architecture design doc answering 10 strategic questions (tools to upgrade, new tools, MCP-vs-native, existing MCP servers to integrate, custom MCP ecosystem, agent-architecture changes, RAG expansion, missing capabilities, long-term architecture, prioritized roadmap) with mermaid diagrams.
3. **Wrote `docs/QUASAR_V2_EXECUTION_PLAN.md`** — a 7-phase step-by-step plan with per-step acceptance criteria.
4. **This session** — produced this `docs/v2/` handoff set and locked the dual-path decision below.

## 4. The governing decision (this session's refinement)

Earlier design leaned "MCP-first." The refined, **final** decision is:

- **Both paths ship.** The end system supports MCP tools **and** the existing direct/native tool approach. Neither is removed.
- **MCP is built first**, because it is the cleanest lever to break the monolith and because MANNA is already being built.
- **The native-tool path is preserved and improved later**, sharing logic with MCP rather than duplicating it.
- **#1 priority is the monolith bottleneck.** Everything else waits on, or is structured around, that split.

**Why MCP first (the reasoning, so it's not re-litigated):**
- Maintainability: adding a native tool today means editing a 12,714-line file at "4 anchors" (its own `CONVENTIONS.md` says so). MCP servers are small, independently tested/deployed repos.
- Security: the two worst liabilities — `exec()`-based user tools and the `/api/mcp-servers` stdio spawner — are arbitrary code/command execution *in the shared process*. Out-of-process MCP removes that class of risk.
- Scale: native tools all compete for one ~2 GB API process; MCP servers scale independently.
- Reproducibility: an MCP server can return the exact executed query per call, making the notebook honest (today it's fabricated).
- Leverage: Quasar already ships ~80% of the MCP plumbing (`services/mcp_server_service.py` + `/api/mcp-servers` + a registry that maps `mcp_tool.inputSchema`). The bridge is *broken*, not absent.

**Why keep tools (the reasoning):**
- Latency: in-process tools have no network hop — right for sub-second, chatty, or UI-coupled calls (thumbnails, deep-links, data-card shaping).
- Trust boundary: safety governor, provenance, citation verification, quotas, auth must run on Quasar's side, not a remote server.
- Continuity: quasarassistant.com is live; the native path must keep working through the whole migration.
- Offline/local dev and non-MCP LLM providers still need a native path.

## 5. Current state of the implementation (honest)

- V1 is **live and functional** but has significant drift between docs and reality and several **dead or broken flagship features** (see `OPEN_ISSUES_AND_TODOS.md`). Nothing in V2's new architecture (capabilities layer, adapters, fixed bridge) has been built yet — **this handoff is the plan, not code.**
- Confirmed-good code to preserve: `core/task_dag.py`, `core/complexity.py`, `core/tools.py` (registry shape), the Data Lab triad (`services/datalab_sql_policy.py` + `datalab_query_builders.py` + `datalab_registry.py`), `services/splatalogue.py`, `services/vo_registry.py`, `services/citation_verifier.py`, `services/provenance.py`, `integrations/datalab_client.py`, `integrations/datalab_sia_client.py`, `integrations/svo_fps_client.py`.
- Confirmed dead/broken to delete or fix: `agents/` package (imported nowhere), `core/agent_pool.py`, `core/workflow_store.py`, `core/result_cache.py`, `core/context_manager.py`, `core/health_monitor.py`, `core/model_council.py`, `core/session_memory.py` (the memory stack), `config/settings.py` (dead), the MCP bridge (cross-loop asyncio), the notebook generator (fabricated), the RAG score-gate (bug).

## 6. Non-negotiable constraints carried from the discussion

1. **Never fabricate scientific data.** Several V1 tools return mock/invented data on failure — this must become typed errors. A science tool that invents data is worse than one that errors.
2. **No arbitrary code execution in the shared process.** The `exec()` user tools and stdio MCP spawner must be isolated/removed.
3. **Keep quasarassistant.com working throughout.** Migrate behind feature flags, per tool family, with the benchmark as the regression gate.
4. **The benchmark is the gate.** No refactor merges if it regresses the DataLabBench/ALMA baseline. (Baseline currently only exists in hand-written `tmp/*.md` — capturing a real one is an early task.)
