# Quasar

![Quasar UI](assets/UI.png)

**Quasar** is a free, source-available AI research assistant that makes ALMA Science Archive search and data retrieval as simple as asking a question in natural language. It is a *domain-specialized agent framework* that wraps any general-purpose LLM with the tools, knowledge, and orchestration needed to perform real radio astronomy tasks.

Quasar pairs a registry of **75+ domain-specific tools** covering archive data search, retrieval, and analysis with a **Conductor** orchestration engine that decomposes complex, multi-step research queries into Directed Acyclic task Graphs (DAGs). By externalizing task planning into the Conductor, Quasar enables even smaller or non-reasoning LLMs to reliably execute sophisticated archive workflows through structured tool composition.

> **Key insight:** A general-purpose LLM becomes a capable scientific assistant not through model fine-tuning alone, but in combination with careful domain engineering.

- **Live application:** [quasarassistant.com](https://www.quasarassistant.com/)
- **Repository:** [adamzacharia/Quasar](https://github.com/adamzacharia/Quasar)

## Current Capabilities

- Natural-language ALMA archive search by target, position, frequency, and metadata filters
- Automatic target name resolution via SIMBAD with fallback cone search
- NASA ADS literature search, paper retrieval, and full-text PDF extraction from arXiv
- Sandboxed Python REPL for precise scientific computation (redshift calculations, spectral line matching, etc.)
- Conductor-driven DAG orchestration for complex multi-step workflows
- Retrieval-Augmented Generation over ALMA technical documentation
- Per-user personal document collection and preference memory
- ALMA DataLink file listing and remote FITS header inspection
- CASA imaging and calibration script generation
- Downloadable Jupyter Notebook reproducibility for every workflow
- Streamed task execution updates via Server-Sent Events

## Architecture

Incoming queries enter the **QuasarAgent** via a web interface, which assembles context from three sources:

1. **Shared RAG store** — Indexes ALMA technical documentation (Technical Handbook, Proposer's Guide) and optionally live web search results.
2. **Per-user personal vector collection** — Researchers can upload their own documents (observing proposals, scripts, notes), retrieved alongside system knowledge.
3. **Preference memory** — Built by summarizing past conversations, learning each user's observational focus and preferences over time.

The query then passes through a **two-stage complexity detector**:
- **Simple queries** → routed to the **Agent Tool Loop**, where the backbone LLM iterates over tool selection, execution, and observation.
- **Complex queries** → escalated to the **Conductor**, which decomposes the request into a dependency-aware task graph and dispatches subtasks to specialist sub-agents (Archive, Literature, Analysis) sequentially or in parallel.

The Conductor's sub-agents have access to a **Sandboxed Python REPL** for running scripts in a restricted environment with `numpy`, `astropy`, `pandas`, and other scientific dependencies. The sandbox connects back to the full tool registry via a `call_tool()` interface, enabling hybrid workflows that combine programmatic computation with live archive access. Responses are accompanied by a downloadable **Jupyter Notebook** for full transparency and reproducibility.

```mermaid
flowchart LR
    U["User"]
    FE["Next.js UI<br/>ui-pro/src"]
    API["FastAPI SSE API<br/>ui-pro/api/main.py"]
    AG["QuasarAgent<br/>core/agent.py"]
    CTX["Context Assembly<br/>RAG · Personal Docs · Memory"]
    CD["Two-Stage Complexity<br/>Detector"]
    DIRECT["Agent Tool Loop<br/>(Simple Queries)"]
    COND["Conductor<br/>DAG Orchestration"]
    REPL["Sandboxed REPL<br/>numpy · astropy · pandas"]
    SVC["75+ Domain Tools<br/>Search · ADS · DataLink · FITS · CASA"]
    EXT["External Systems<br/>ALMA Archive · NASA ADS · CADC · SIMBAD"]
    SSE["SSE Events<br/>token · status · tool · task"]

    U --> FE --> API --> AG
    AG --> CTX
    AG --> CD
    CD --> DIRECT
    CD --> COND
    DIRECT --> SVC
    COND --> SVC
    COND --> REPL
    REPL --> SVC
    SVC --> EXT
    AG --> SSE --> FE
```

## Repository Layout

| Path | Purpose |
|---|---|
| `core/` | Agent runtime, Conductor orchestration, tool registry, complexity detector, and sandboxed REPL |
| `services/` | Domain logic for search, retrieval, FITS processing, plotting, CASA, auth, and storage |
| `integrations/` | External system adapters for ALMA, NASA ADS, DataLink, TAP, MAST, ESO, IRSA, SkyView |
| `ui-pro/src/` | Next.js frontend |
| `ui-pro/api/` | FastAPI backend and SSE endpoints |
| `tests/` | Integration, evaluation, and verification scripts |

## Requirements

- Python 3.9+
- Node.js and npm
- `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, or `TACC_API_KEY`

Optional configuration:

- `NASA_ADS_API_KEY`
- `QDRANT_URL`
- `QDRANT_API_KEY`
- `TURSO_DATABASE_URL`
- `TURSO_AUTH_TOKEN`
- `JWT_SECRET`
- `NEXT_PUBLIC_API_URL`
- `DEFAULT_LLM_MODEL`

## Configuration

Create a repository root `.env` file before starting the backend.

| Variable | Required | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | One model provider required | OpenAI model access for the agent runtime and optional embeddings |
| `DEEPSEEK_API_KEY` | One model provider required | DeepSeek model access |
| `TACC_API_KEY` | One model provider required | Texas Advanced Computing Center / Tejas model access |
| `TACC_BASE_URL` | No | TACC OpenAI-compatible endpoint; defaults to `https://ai.tejas.tacc.utexas.edu/v1` |
| `NASA_ADS_API_KEY` | No | Literature search via NASA ADS |
| `QDRANT_URL` | No | Persistent vector storage for RAG and memory |
| `QDRANT_API_KEY` | No | Authentication for Qdrant Cloud |
| `TURSO_DATABASE_URL` | No | Cloud SQL storage |
| `TURSO_AUTH_TOKEN` | No | Authentication for Turso |
| `JWT_SECRET` | Production | JWT signing secret for authentication; must be a private random value when `QUASAR_ENV=production` |
| `NEXT_PUBLIC_API_URL` | No | Frontend API base URL override |
| `DEFAULT_LLM_MODEL` | No | Backbone LLM model |
| `QUASAR_FAST_MODEL` | No | Fast internal model for routing, memory, and classifiers |
| `QUASAR_REASONING_MODEL` | No | Strong internal model for scientific reasoning subtasks |
| `QUASAR_CONDUCTOR_MODEL` | No | Planning model used by Conductor |
| `QUASAR_SYNTHESIS_MODEL` | No | Final synthesis model used by Conductor |

### TACC / Tejas configuration

Quasar supports these TACC model IDs:

```text
gpt-oss-120b
Llama-4-Maverick-17B-128E-Instruct
gemma-4-31B-it
MiniMax-M2.7
Qwen3-32B
Meta-Llama-3.2-1B-Instruct
Meta-Llama-3.1-8B-Instruct
Meta-Llama-3.3-70B-Instruct
Mistral-Large-3-675B-Instruct-2512
E5-Mistral-7B-Instruct
```

For a TACC-first Render deployment, set these environment variables in the Render service dashboard:

```bash
TACC_API_KEY=your-tacc-key
TACC_BASE_URL=https://ai.tejas.tacc.utexas.edu/v1
DEFAULT_LLM_MODEL=gpt-oss-120b
QUASAR_FAST_MODEL=gpt-oss-120b
QUASAR_REASONING_MODEL=gpt-oss-120b
QUASAR_CONDUCTOR_MODEL=gpt-oss-120b
QUASAR_SYNTHESIS_MODEL=gpt-oss-120b
QUASAR_COMPLEXITY_MODEL=gpt-oss-120b
QUASAR_WEB_INTENT_MODEL=gpt-oss-120b
QUASAR_PAPER_INTENT_MODEL=gpt-oss-120b
QUASAR_WEB_SYNTHESIS_MODEL=gpt-oss-120b
QUASAR_SUMMARY_MODEL=gpt-oss-120b
QUASAR_MEMORY_EXTRACTION_MODEL=gpt-oss-120b
QUASAR_PERSONAL_MEMORY_MODEL=gpt-oss-120b
QUASAR_PDF_MODEL=gpt-oss-120b
QUASAR_PROPOSAL_CRITIC_MODEL=gpt-oss-120b
TACC_KEY_TEST_MODEL=Meta-Llama-3.2-1B-Instruct
TACC_ENABLE_RESPONSE_FORMAT=false
TACC_STREAM_INCLUDE_USAGE=true
QUASAR_TACC_FALLBACK_MODEL=none
```

The TACC key is a platform key configured in Render, not an end-user BYOK key. Keep `OPENAI_API_KEY` only if you want OpenAI embeddings, image prepass, or OpenAI fallback behavior. For a strict TACC-only model path, leave OpenAI/DeepSeek keys unset and keep `QUASAR_TACC_FALLBACK_MODEL=none`.

## Local Development

### 1. Clone the repository and install Python dependencies

```bash
git clone https://github.com/adamzacharia/Quasar.git
cd Quasar
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS or Linux
source .venv/bin/activate

pip install -r requirements.txt

# macOS or Linux
cp .env.example .env

# Windows PowerShell
Copy-Item .env.example .env
```

### 2. Start the backend

Run the FastAPI backend from the `ui-pro` directory:

```bash
cd ui-pro
uvicorn api.main:app --reload --port 8000
```

The backend loads `.env` from the repository root and exposes SSE endpoints on `http://localhost:8000`.

### 3. Start the frontend

Open a second terminal and run:

```bash
cd ui-pro
npm install
npm run dev
```

The frontend runs on `http://localhost:3000`. If `NEXT_PUBLIC_API_URL` is not set, it defaults to `http://localhost:8000`.

### One-command local testing on Windows

For local UI testing with SQLite auth and the disposable `1@1` / `1` login:

```powershell
.\scripts\local\start_quasar.ps1 -Restart
```

This starts the backend on `http://localhost:8000` and the frontend on `http://localhost:3001` with Turso disabled for that local process.

### Regression checks

The frontend has a focused regression test to ensure streamed thinking stays separate from the final answer:

```bash
cd ui-pro
npm run test:thinking
```

### 4. Run the CLI

From the repository root:

```bash
python quasar.py cli
```

You can also run a single query from the command line:

```bash
python quasar.py query "Find ALMA observations of HL Tau in Band 6"
```

## Example Queries

- Find ALMA observations of HL Tau in Band 6.
- Check CO(2-1) line coverage for M87.
- Find the 2018 DSHARP survey overview paper by Andrews et al. and extract the angular resolution for AS 209.
- Search NASA ADS for recent ALMA papers on protoplanetary disks.
- List available data products for the best matching MOUS and inspect the FITS headers.
- Generate a CASA imaging script for a calibrated measurement set.

## Verification

1. Check the backend health endpoint at `http://localhost:8000/health`.
2. Open `http://localhost:3000`.
3. Submit a sample ALMA query.
4. Confirm that the UI receives streamed text, status updates, and task execution events.

## Notes

- The primary runtime surface is the Next.js frontend plus FastAPI backend under `ui-pro/`.
- The default LLM model is `gpt-oss-120b`. Override via `DEFAULT_LLM_MODEL` in `.env` or the Render service environment.
- CASA and CARTA integrations are disabled by default due to high RAM requirements. Enable via `ENABLE_CASA_PIPELINE=true` and `ENABLE_CARTA_INTEGRATION=true`.
- CI runs automatically on pushes to `main`/`beta` and all PRs via GitHub Actions.
- The repository contains broader astronomy modules, but the strongest supported workflow is ALMA archive search and analysis support.
- The implementation reference is maintained in `docs/QUASAR_SYSTEM_DOCUMENTATION.md`.

## License

Quasar is licensed under the [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0). See `LICENSE` for the full terms.

**In short:** anyone may freely use, modify, and share Quasar for any **noncommercial** purpose — including personal use, research, education, and use by nonprofits, public research organizations, and government institutions. **Commercial use of any kind is strictly prohibited**, including selling Quasar or any modified version of it, or offering it as part of a paid product or service. For commercial licensing inquiries, contact the maintainers.
