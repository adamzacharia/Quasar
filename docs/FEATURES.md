# Quasar Feature Roadmap -- Complete Astronomer Toolset

A living document tracking every feature in Quasar, ranked by implementation
priority. Features are organized into four tiers based on impact, feasibility,
and dependency ordering. Checkboxes track completion status.

Last audited against the codebase: 2026-04-25

---

## Status Summary

| Category               | Total | Built | In Progress | Planned | % Done |
|------------------------|-------|-------|-------------|---------|--------|
| Universal              | 10    | 6     | 1           | 3       | 60%    |
| Radio                  | 10    | 6     | 0           | 4       | 60%    |
| Gravitational Wave     | 7     | 2     | 0           | 5       | 29%    |
| Infrared               | 6     | 2     | 0           | 4       | 33%    |
| High-Energy            | 5     | 0     | 0           | 5       | 0%     |
| Optical / UV           | 7     | 1     | 0           | 6       | 14%    |
| Platform               | 6     | 5     | 0           | 1       | 83%    |
| Next-Gen               | 12    | 9     | 1           | 2       | 75%    |
| Literature             | 2     | 2     | 0           | 0       | 100%   |
| Polish & Scale         | 28    | 11    | 0           | 17      | 39%    |
| **Total**              |**95** |**44** | **2**       | **49**  |**46%** |

---

## Tier 1 -- Critical (High Impact, High Feasibility)

These features are either pure computation wrappers, thin astropy tool wrappers,
or direct extensions of existing infrastructure. Each can be built in under 2
hours and adds immediate user value.

- [x] **U1** Publication-Quality Plotting -- ApJ/MNRAS-style figures at 300 DPI, LaTeX labels, colorblind-safe palettes, PDF/PNG export (services/plotting.py)
- [x] **U2** Multi-archive Cross-matcher -- Query ALMA, Chandra, MAST, VizieR, NED, Simbad simultaneously (services/multi_archive.py + 4 clients)
- [x] **U3** Literature-to-Code -- Read arXiv paper methods, output Python/CASA/CIAO code (core/prompts/lit_to_code.py + services/code_generator.py)
- [x] **U6** Splatalogue Line ID -- Query molecular line database by frequency (services/splatalogue.py)
- [x] **R1** ALMA Archive Search -- Search by target, position, frequency, PI, project code (integrations/alminer_client.py + tap.py)
- [x] **R2** CASA Script Generator -- Generate calibration and imaging scripts (services/casa_generator.py)
- [x] **R7** Moment Map Generator -- From FITS cube, produce moment 0/1/2 maps (services/fits_service.py compute_moment_map)
- [x] **G1** GCN Alert Monitor -- Parse GW/GRB/neutrino alerts, extract key parameters (services/gcn_monitor.py)
- [x] **G2** GWTC Catalog Search -- Query Gravitational Wave Transient Catalog by mass, distance, type (services/gcn_monitor.py search_gwtc_catalog)
- [x] **I1** MAST Archive Search -- Query JWST, HST, TESS, Kepler by instrument/filter/target (integrations/mast_client.py)
- [x] **I6** WISE / Spitzer Catalog Query -- Cone search and cross-match with WISE/Spitzer catalogs (integrations/irsa_client.py)
- [x] **P1** Session Pruning -- Auto-reset context window at 90k tokens, save summary (core/context_manager.py + session_memory.py)
- [x] **P2** Telegram Multi-channel -- Receive and respond to queries via Telegram bot
- [x] **P3** Browser Control -- Autonomous web browsing via Playwright (services/browser.py)
- [x] **P4** Skills Plugin System -- Install new tools as self-contained skill packages at runtime (services/user_tools_service.py + MCP bridge)
- [x] **P5** Model Failover -- Auto-fallback across providers if primary model fails (core/health_monitor.py + model_router.py)
- [x] **W1** Red Team TAC (Proposal Critic) -- Multi-agent committee reviewing proposal drafts (services/proposal_critic.py)
- [x] **W2** Dynamic Jupyter Notebook Generation -- Generate .ipynb from DAG execution (services/notebook_gen.py)
- [x] **W3** Chat with Data Cube (VLM Integration) -- FITS cube parsing, rendering, spectrum extraction, moment maps (services/fits_service.py + fits_processing.py)
- [x] **L1** Smart Paper Search (ADS QueryBuilder) -- LLM translates NL queries into optimal ADS syntax using keyword:, bibgroup:, object:, trending(), useful(), similar() (integrations/ads_client.py ADSQueryBuilder)
- [x] **L2** Consensus Evaluation -- Search top-cited papers on a question, read all abstracts, produce structured agreement/disagreement analysis with paper-level citations (core/agent.py evaluate_consensus)
- [x] **W7** Context-Aware Recovery Engine -- RecoveryEngine receives predecessor results/errors for root-cause analysis; predecessor error propagation replaces generic "predecessor failed" messages (core/recovery.py + core/conductor.py + core/task_dag.py)
- [x] **W8** Tiered Complexity DAG Control -- 3-tier complexity classification (moderate/complex/expert) with per-tier max subtask caps to prevent over-decomposition (core/conductor.py + core/agent.py)
- [x] **W9** Hybrid BM25 Reranking -- Client-side BM25 keyword reranking over Qdrant semantic results using Reciprocal Rank Fusion; catches exact acronym/keyword matches without re-ingestion (services/rag_service.py)
- [x] **W10** Human-in-the-Loop Plan Review -- Conductor blocks on user approval before executing DAG; users can approve or provide feedback to trigger LLM re-decomposition with labeled prompt sections; max 3 revision iterations (core/conductor.py + ui-pro/api/main.py + ui-pro/src/components/PlanReviewWidget.tsx)
- [x] **W12** Real-Time Token-by-Token Reasoning Streaming -- Smooth, real-time token-by-token streaming of deep reasoning Chain of Thought (CoT) directly inside a premium collapsible ThoughtProcessWidget, replacing discrete line-by-line step bubbles; auto-detects thinking models (o-series, DeepSeek Pro, GPT-5.x) and persists thoughts in history (core/agent.py + ui-pro/api/main.py + ui-pro/src/components/ChatMessage.tsx)
- [x] **W11** OpenAlex Researcher & Bibliometrics -- Researcher profile lookup (h-index, institution, ORCID, topics, publication history); bibliometric trend aggregation (papers-per-year); silent batch DOI enrichment of ADS paper results (FWCI, citation percentiles, funding tags, OA PDFs); dual parallel web search for researcher profiles (general context + dedicated email/contact lookup); additive routing (OpenAlex runs alongside RAG, never exclusive); branding hidden from user-facing output (integrations/openalex_client.py + core/agent.py lookup_researcher + get_research_trends)
- [x] **U9** Redshift Calculator -- Convert between z, Mpc, lookback time, physical scale, angular size (services/astro_calculators.py)
- [x] **U10** Coordinate Converter -- RA/Dec to Galactic to Ecliptic, epoch conversions J2000/B1950 (services/astro_calculators.py)
- [x] **R6** Beam Calculator -- Given array config and frequency, compute synthesized beam size (services/astro_calculators.py)
- [x] **R4** ALMA Sensitivity Calculator -- Radiometer equation per band with Tsys and PWV scaling (services/astro_calculators.py)
- [x] **O6** Finding Chart Generator -- DSS/PanSTARRS chart with WCS, crosshair, N/E arrows, scale bar (integrations/skyview_client.py)
- [x] **R3** Spectral Line Profile Plotter -- Extract spectrum from cube, fit Gaussian, report FWHM and integrated flux (services/fits_service.py)

---

## Tier 2 -- High Priority (Builds on Existing Infrastructure)

These features require moderate new code but leverage existing services (RAG,
archive clients, plotting). Each takes 2-6 hours and significantly widens the
user audience.

- [/] **U7** NED / Simbad Natural Language Search -- Basic SIMBAD/NED object lookups work via resolve_target and cross_match_source tools; NL criteria queries not yet built (services/multi_archive.py, partial)
- [ ] **R5** ALMA Proposal Advisor -- RAG over Cycle call and proposer guide (extends rag_service.py)
- [ ] **U4** Grant / Proposal Writer -- RAG over NSF/NASA calls, successful funded abstracts, reviewer guidelines
- [ ] **R8** Channel Map Viewer -- Grid of velocity channel maps from 3D cubes (extends fits_service.py)
- [ ] **O2** Target Visibility Planner -- Given site, date, target list, generate observability chart (astroplan)
- [ ] **O1** Transient Classifier -- Given ZTF/LSST alert, classify as SN Ia, AGN, CV, asteroid (LLM-powered)
- [ ] **O3** Spectral Redshift Assistant -- Upload 1D spectrum, AI identifies lines and estimates redshift
- [ ] **I2** JWST Pipeline Advisor -- Recommend CRDS context, pipeline stages, parameters for obs mode (RAG pattern)
- [ ] **I3** JWST Proposal Advisor -- RAG over successful JWST proposals and Cycle call
- [ ] **G3** Multi-messenger Sky Matcher -- Cross-match GW HEALPix sky map against telescope archives
- [ ] **I4** Photometric SED Plotter -- Plot multi-band photometry with model SED overlay

---

## Tier 3 -- Medium Priority (New Services Required)

These features require new service modules or external library integration.
Each takes 4-8 hours and serves a more specialized user base.

- [ ] **H1** CIAO Script Generator -- Generate CIAO data reduction scripts for Chandra (mirrors CASA pattern)
- [ ] **H2** XSPEC Model Advisor -- Recommend spectral models for given source type / energy range (LLM-based)
- [ ] **H3** Fermi LAT 4FGL Search -- Query Fermi 4FGL catalog for sources near a target (astroquery.fermi)
- [ ] **O5** Extinction Calculator -- Query Schlafly and Finkbeiner dust maps for E(B-V) at any position
- [ ] **O4** Color-Magnitude Diagram Builder -- Generate CMD from catalog photometry
- [ ] **G4** GW Parameter Estimator -- Given event params (M_chirp, distance), estimate expected SNR
- [ ] **U5** Observing Run Log Analyzer -- Upload telescope night log, AI flags anomalies (PDF parsing + LLM)
- [ ] **U8** SED Fitter Integration -- Wrap CIGALE/Bagpipes for photometric redshift and galaxy SED fitting
- [ ] **R9** uv-Coverage Plotter -- Visualize baseline coverage for array configuration
- [ ] **R10** Primary Beam Correction Explainer -- Guide through PB correction for mosaics (RAG + LLM)
- [ ] **I5** Dust Temperature Estimator -- Modified blackbody fit to far-IR photometry

---

## Tier 4 -- Future (Ambitious / Specialized)

These features require significant new infrastructure, external system
integration, or are highly specialized. They are stretch goals for future
releases.

- [ ] **H4** X-ray Image Display -- Display Chandra/XMM event images with smoothing
- [ ] **H5** Timing Analysis Tool -- Power spectra, lightcurves, periodograms from event files
- [ ] **G5** Bilby / LALInference Config Generator -- Generate PE config files from event parameters
- [ ] **G6** GW Strain Plotter -- Plot GW timeseries with bandpass filter and matched filter
- [ ] **G7** Corner Plot Generator -- Plot posterior distributions from MCMC output (corner.py)
- [ ] **O7** Transient Name Server Search -- Query IAU TNS for registered transients
- [ ] **P6** WhatsApp Channel -- Route queries via Meta WhatsApp Cloud API
- [ ] **W4** Autonomous ToO Agent -- Trigger off GCN alerts, cross-match, pre-generate scripts, approve via Telegram
- [ ] **W5** "Find Similar" Morphological Search -- Embedding-based image search against ALMA archive
- [ ] **W6** Theory vs Reality Bridge -- Sandboxed simalma or Mirage runs converting simulations to telescope data

---

## Tier 5 -- Polish & Scale (Ongoing)

These items make Quasar production-grade, trustworthy, and pleasant to use.
Without them, even great features feel broken.

### Error Handling & Reliability

- [ ] **PS1** Replace bare `except: pass` -- Specific exception handling + logging for all 8 instances
- [x] **PS2** Structured Logging -- Langfuse trace spans across Conductor DAG execution + thread-local LLM call parenting; Sentry breadcrumbs per DAG node (core/langfuse_integration.py + core/conductor.py)
- [ ] **PS3** Rate Limiting -- Add rate limits to public API endpoints
- [ ] **PS4** Input Validation -- Sanitize all tool parameters before execution
- [ ] **PS5** Graceful Degradation -- Fallback behavior when external services (ADS, ALMA, MAST) are down

### Performance & Scalability

- [ ] **PS6** Response Caching -- Cache repeat archive queries (TTL-based)
- [ ] **PS7** Connection Pooling -- Pool HTTP connections for external APIs
- [ ] **PS8** Request Queuing -- Queue expensive operations (FITS downloads, CASA scripts)
- [ ] **PS9** Token Budget Optimization -- Better context window budgeting (infrastructure exists but underutilized)

### User Experience

- [ ] **PS10** Mobile-Responsive UI -- Currently desktop-only
- [x] **PS11** Onboarding Tutorial -- 5-step guided first-visit overlay with glassmorphism cards, step dots, localStorage persistence, and re-trigger from Settings (ui-pro/src/components/OnboardingOverlay.tsx)
- [ ] **PS12** Example Query Gallery -- Show example queries on the empty state
- [x] **PS13** Download Progress Indicator -- Streaming progress bar with filename, speed, percentage via SSE download_progress events; auto-dismiss after completion (services/download_utils.py + ui-pro/src/components/DownloadProgress.tsx)
- [ ] **PS14** Export Conversation -- Export chat as PDF or Markdown
- [x] **PS15** Dark/Light Mode Toggle -- CSS custom property-based theming via data-theme attribute, Zustand store, Sun/Moon toggle in sidebar, flash-free initialization (ui-pro/src/lib/theme-store.ts + ui-pro/src/app/globals.css)

### Documentation & Community

- [ ] **PS16** API Reference Docs -- Complete OpenAPI/Swagger documentation (FastAPI auto-gen exists partially)
- [ ] **PS17** Architecture Guide -- Contributor-facing system architecture documentation
- [ ] **PS18** Plugin Development Guide -- How to build and package Quasar skill plugins
- [ ] **PS19** Tutorial Notebooks -- Common workflow tutorials as Jupyter notebooks

### Security & Operations

- [x] **PS20** Guest Gate, Auth Lockout, & COPPA Compliance -- Lock down the entire interface and backend endpoints to registered investigators, adding a mandatory Terms & Conditions checkbox agreement during login/signup certifying users are at least 13 years of age to protect children's data privacy under COPPA regulations, with a beautiful cosmic /terms policy page detailing Langfuse & Turso data collection (ui-pro/src/components/AuthModal.tsx + ui-pro/src/app/terms/page.tsx + ui-pro/api/main.py + docs/legal/terms.md)
- [ ] **PS21** Remove Hardcoded JWT Secret -- Require env var for JWT signing key
- [ ] **PS22** API Key Rotation -- Mechanism to rotate API keys without downtime
- [ ] **PS23** Usage Quotas -- Per-user usage limits
- [ ] **PS24** Abuse Detection -- Detect prompt injection, excessive API calls
- [x] **PS25** Error Monitoring -- Sentry trace correlation with quasar.trace_id and quasar.tool tags; breadcrumbs for DAG node failures; tool tagging in @log_tool decorator (core/logger.py + core/conductor.py)
- [x] **PS26** Astronomy Acronym Expansion -- Expand ALMA/VLA/JWST/AGN etc. to full forms in web search queries so generic engines return astronomy results (core/agent.py _expand_astro_query)
- [x] **PS27** Response Feedback Collection -- Like/dislike buttons on assistant messages, persisted to analytics DB for RLHF/improvement (services/analytics_service.py + ui-pro/src/components/ChatMessage.tsx)
- [x] **PS28** LLM Cost Analytics & Tracing -- Helicone OpenAI proxy for automatic cost dashboards; Langfuse generation logging per LLM call with token usage and latency (core/llm_client.py + core/langfuse_integration.py)
- [x] **PS29** OpenAI Responses API Unification -- Purged all legacy chat completions and unified the entire codebase (vision uploads, text reasoning, literature searching, proposal critiques) under the modern OpenAI Responses API via LLMClient (ui-pro/api/main.py)
- [x] **PS30** DeepSeek Thinking Mode Integration -- Added full support for the new 'deepseek-v4-pro' model in the backend and frontend; maps the native real-time 'reasoning_content' chain-of-thought tokens directly to SSE events to display raw reasoning steps in the UI Thought widget; configures reasoning effort controls and thinking parameter shimming (core/llm_client.py + ui-pro/src/components/Sidebar.tsx)
- [x] **PS31** DeepSeek Tool Calling Parity -- Fixed three bugs preventing data cards, tables, and charts from appearing when using DeepSeek models: (1) tool_choice parameter was not forwarded to the DeepSeek Chat Completions API, so archive fetch queries did not force tool execution; (2) function call argument deltas used an unreachable call_id lookup path, leaving tool arguments empty; (3) response.completed handler did not backfill missing arguments/names for already-tracked function calls. All non-OpenAI providers now produce identical data card and table output as OpenAI. (core/llm_client.py + core/agent.py)
- [x] **PS32** DeepSeek V4 Flash Full Integration -- Added 'deepseek-v4-flash' as a selectable model across the entire stack: backend model list, frontend fallback/default lists, temperature-stripping set, and thinking-models set; Flash runs with thinking enabled at max reasoning effort, identical to V4 Pro; pricing and observability cost tracking were already in place (ui-pro/api/main.py + ui-pro/src/lib/api.ts + ui-pro/src/lib/store.ts + core/agent.py)
- [x] **PS33** Langfuse Token & Cost Tracking Parity -- Fixed a major issue where DeepSeek, Anthropic, Google, and Local models were missing token usage and cost metrics in the Langfuse dashboard. (1) Defined standardized LLMUsage and updated LLMResponse to store usage; (2) Populated usage in non-streaming mapping functions; (3) Added stream_options to DeepSeek and Local streaming, and extracted final message/chunk usage in Anthropic and Google streams to yield usage metadata on completion. (core/llm_client.py)
- [x] **PS34** Conversation Model Persistence -- Fixed a bug where a conversation's selected model would reset to default ("gpt-5.4-mini") on page reload or chat switching by adding model persistence across the entire stack: (1) added a nullable model column to the conversations SQLite/Turso database table with startup schema migration; (2) updated `/api/chat` and `/api/conversations` backend routes to save the active model on new messages; (3) updated frontend Zustand store and types to restore and sync the active model on conversation switch. (services/conversation_service.py + ui-pro/api/main.py + ui-pro/src/lib/api.ts + ui-pro/src/lib/store.ts)

---

## Implementation Order (Remaining Features)

Phase 1 -- Quick Wins (Tier 1 remaining, estimated 1 day):
  U9, U10, R6, R4, O6, R3

Phase 2 -- Archive and Search Expansion (Tier 2 first half, estimated 2 days):
  U7, R5, U4, R8, O2

Phase 3 -- AI-Powered Analysis (Tier 2 second half, estimated 2 days):
  O1, O3, I2, I3, G3, I4

Phase 4 -- X-ray and Specialized (Tier 3, estimated 3 days):
  H1, H2, H3, O5, O4, G4, U5, U8, R9, R10, I5

Phase 5 -- Stretch Goals (Tier 4, ongoing):
  H4, H5, G5, G6, G7, O7, P6, W4, W5, W6

Phase 6 -- Polish & Scale (Ongoing, continuous):
  PS1-PS25 (prioritize PS13 download progress, PS1 error handling, PS21 JWT security)

---

*Last updated: 2026-05-26 -- Quasar v3.3.7 (Langfuse token and cost tracking parity)*
