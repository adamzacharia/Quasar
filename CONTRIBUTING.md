# Contributing to Quasar

Thanks for your interest in improving Quasar — a domain-specialized agent
framework for radio astronomy (ALMA archive search, catalog science, literature,
FITS analysis, and more). This guide is for **human contributors**: how to set up
a dev environment, run the app and the checks, and get a change merged.

> **License note.** Quasar is released under the
> [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0).
> Contributions are accepted under the same terms. Commercial use is prohibited.

---

## 1. Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Python | **3.11** | CI pins 3.11; 3.9+ generally works locally. |
| Node.js + npm | **22** | Needed for the `ui-pro/` frontend. |
| Git | any recent | — |

You also need at least one LLM provider key (`OPENAI_API_KEY`, `DEEPSEEK_API_KEY`,
or `TACC_API_KEY`) to actually run the agent. The automated test suite does **not**
call real APIs, so a key is not required just to run tests.

---

## 2. Getting started

```bash
git clone https://github.com/adamzacharia/Quasar.git
cd Quasar

# Python environment
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt

# Environment file (fill in at least one provider key)
cp .env.example .env          # macOS / Linux
Copy-Item .env.example .env   # Windows PowerShell

# Frontend dependencies
cd ui-pro && npm install && cd ..
```

The backend reads `.env` from the repository root. See the
[README](README.md#configuration) for the full list of environment variables.

---

## 3. Running the app

**Backend (FastAPI + SSE, port 8000):**

```bash
cd ui-pro
uvicorn api.main:app --reload --port 8000
```

**Frontend (Next.js, port 3000):**

```bash
cd ui-pro
npm run dev
```

**One command (Windows, SQLite auth + throwaway `1@1` / `1` login):**

```powershell
.\scripts\local\start_quasar.ps1 -Restart
```

This starts the backend on `http://localhost:8000` and the frontend on
`http://localhost:3001` with Turso disabled for the local process.

---

## 4. Repository layout

| Path | What lives here |
|---|---|
| `core/` | Agent runtime, Conductor DAG orchestration, tool registry (`core/tools.py`) |
| `capabilities/`, `adapters/` | V2 shared-core layer (pure capabilities + native/MCP adapters) — see `docs/v2/` |
| `services/` | Domain logic (search, RAG, ADS, plotting, FITS, CASA, auth, storage) |
| `integrations/` | External API clients (ALMA, ADS, DataLink, TAP, MAST, ESO, IRSA, SkyView) |
| `ui-pro/src/` | Next.js frontend |
| `ui-pro/api/` | FastAPI backend + SSE endpoints |
| `tests/` | `unit/` (offline), `integration/` (opt-in, live), `smoke/` |
| `Benchmark/datalabbench/` | DataLabBench scorer + dataset |
| `docs/` | Documentation. `docs/TOOLS.md` is **auto-generated**; `docs/v2/` is the V2 design set. |

---

## 5. Tests and checks

CI (`.github/workflows/ci.yml`) runs the following on every push/PR to `beta`.
Please run the relevant ones locally before opening a PR.

### Python

```bash
# Full offline unit suite (no network, no API keys)
pytest tests/unit/ -v --tb=short

# DataLabBench scorer self-test (offline scorer/penalty/rollup integrity)
python Benchmark/datalabbench/run_datalabbench.py --self-test

# Lint (ruff) — CI currently lints a focused file set
ruff check --select E,F,W --ignore E501,E402 <files>
```

> On Windows, run these with the venv interpreter (`.venv\Scripts\python.exe`) so
> you pick up the pinned dependencies.

### Frontend (`ui-pro/`)

```bash
npm run lint     # eslint
npm test         # node --test regression suite (thinking-vs-answer separation, ...)
npm run build    # production build must succeed
```

### Generated-docs drift check

`docs/TOOLS.md` is generated from the live tool registry. CI fails if it is out
of date, so **regenerate and commit it whenever you add, remove, or rename a
tool** (see §6):

```bash
python scripts/gen_tool_registry_doc.py --check   # what CI runs (read-only)
```

---

## 6. Adding or changing a tool

Quasar exposes ~140 tools to the LLM through a single **`ToolRegistry`**
(`core/tools.py`). Each tool is a `name` + `description` + JSON-schema
`parameters` + a Python `function`.

- **Today:** most tools are registered inline in
  `core/agent.py::_register_tools()`:

  ```python
  self.tool_registry.register(Tool(
      name="my_new_tool",
      description="What the tool does and when the LLM should call it.",
      function=self._my_new_tool,
      parameters={"type": "object", "properties": {...}, "required": [...]},
      category="analysis",
  ))
  ```

- **Going forward (preferred):** the V2 refactor moves tool *implementations*
  into pure functions under `capabilities/<domain>.py` that return a
  `ToolResult`, registered through `adapters/native`. See `docs/v2/` (start with
  `INDEX.md` → `SHARED_CORE_ARCHITECTURE.md`) before adding a new family.

Whichever path you use, after the change:

1. Regenerate the tool reference and commit it:
   ```bash
   python scripts/gen_tool_registry_doc.py
   ```
2. Add/adjust offline unit tests under `tests/unit/`.
3. Confirm `python scripts/gen_tool_registry_doc.py --check` passes (it will
   after step 1).

Do **not** hand-edit `docs/TOOLS.md` — it is overwritten by the generator, and CI
compares it against the code.

---

## 7. Documentation

- **Generated, not written:** `docs/TOOLS.md` comes from
  `scripts/gen_tool_registry_doc.py`; architecture diagrams come from
  `scripts/generate_mermaid_graphs.py`. Regenerate these instead of editing the
  output by hand.
- **Runtime source of truth:** `docs/QUASAR_SYSTEM_DOCUMENTATION.md`.
- **V2 design set:** `docs/v2/` — keep the relevant file (and `docs/v2/STATUS.md`)
  updated when a change alters architecture or scope.
- Keep tool-count claims out of prose; link to `docs/TOOLS.md` instead so numbers
  cannot drift.

---

## 8. Code style

- **Python:** `ruff` with `E,F,W` selected (`E501` line-length and `E402`
  import-order are ignored). Match the surrounding code's naming and structure.
- **TypeScript/React:** `eslint` (`npm run lint`), Next.js 15 conventions.
- Keep changes focused; avoid unrelated refactors in the same PR.

---

## 9. Branching and pull requests

- The base/default branch is **`beta`**. Branch off `beta` and target your PR
  back at `beta`.
- CI must be green (lint, unit tests, benchmark self-test, tool-doc drift check,
  import + app-start smoke tests, frontend lint/test/build) before review.
- Write a clear PR description: what changed, why, and how you verified it.
- Include tests for new behavior and update any affected docs.

---

## 10. Getting help

- **App usage & configuration:** [README.md](README.md)
- **How the running system works:** `docs/QUASAR_SYSTEM_DOCUMENTATION.md`
- **Where the project is headed (V2):** `docs/v2/INDEX.md`
- **The full tool list:** [docs/TOOLS.md](docs/TOOLS.md)

Welcome aboard, and clear skies. 🔭
