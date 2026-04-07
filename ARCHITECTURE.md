# Quasar Architecture

> **Quasar v3.0** — AI-Powered Radio Astronomy Research Assistant
> This document maps every module, how they connect, and the data flows through the system.

---

## System Overview

```mermaid
graph TB
    subgraph Entry["Entry Points"]
        CLI["cli.py<br/>Terminal REPL"]
        WebUI["ui/app.py<br/>Streamlit Web UI"]
        NextUI["ui-pro/<br/>Next.js Chat UI"]
        TG["telegram.py<br/>Telegram Bot"]
    end

    subgraph Core["Core — Orchestration"]
        Agent["agent.py<br/>QuasarAgent"]
        RLM["rlm.py<br/>RecursiveLanguageModel"]
        RLMEnv["rlm_environment.py<br/>REPL Sandbox"]
        Memory["memory.py<br/>ConversationMemory"]
        Tools["tools.py<br/>ToolRegistry"]
        Prompts["prompts.py<br/>LLM Prompts"]
    end

    subgraph Services["Services — Business Logic"]
        Search["search.py<br/>SearchService"]
        ADS["ads_service.py<br/>ADSService"]
        RAG["rag_service.py<br/>RAGService"]
        MemSvc["memory_service.py<br/>MemoryService"]
        Browser["browser.py<br/>BrowserService"]
        Plotting["plotting.py<br/>PlottingService"]
        Splat["splatalogue.py<br/>SplatalogueTool"]
        MultiArch["multi_archive.py<br/>MultiArchiveMatcher"]
        CASA_Gen["casa_generator.py<br/>CASAScriptGenerator"]
        GCN["gcn_monitor.py<br/>GCNAlertMonitor"]
        Auth["auth.py<br/>AuthService"]
        ConvSvc["conversation_service.py<br/>ConversationService"]
        Analysis["analysis.py<br/>RadioAnalysisService"]
        DataProc["data_processor.py<br/>DataProcessor"]
        CodeGen["code_generator.py<br/>CodeGenerator"]
        VisSvc["visualization_service.py<br/>VisualizationService"]
    end

    subgraph Integrations["Integrations — External API Clients"]
        ALminer["alminer_client.py<br/>ALminerClient"]
        ADSClient["ads_client.py<br/>ADSClient"]
        TAP["tap.py<br/>NRAOTapClient"]
        CASAInt["casa.py<br/>CASA Integration"]
        CARTA["carta.py<br/>CARTA Integration"]
        DL["datalink.py<br/>DataLinkClient"]
    end

    subgraph External["External Services"]
        OpenAI["OpenAI API<br/>GPT-4o / GPT-4o-mini"]
        ALMA_Archive["ALMA Science Archive<br/>NRAO"]
        NASA_ADS["NASA ADS<br/>Paper Search"]
        Mem0["mem0<br/>Long-Term Memory"]
        ChromaDB["ChromaDB<br/>Vector Store"]
        SIMBAD["SIMBAD/NED/MAST<br/>Catalogs"]
        Splatalogue_DB["Splatalogue<br/>Line Database"]
        DuckDuckGo["DuckDuckGo<br/>Web Search"]
    end

    %% Entry → Core
    CLI --> Agent
    WebUI --> Agent
    NextUI -->|"SSE /api/chat"| Agent
    TG -->|"Webhook"| Agent

    %% Core internal
    Agent --> Memory
    Agent --> Tools
    Agent --> Prompts
    Agent --> RLM
    RLM --> RLMEnv
    RLM --> Prompts

    %% Core → Services
    Agent --> Search
    Agent --> ADS
    Agent --> RAG
    Agent --> MemSvc
    Agent --> Browser
    Agent --> Plotting
    Agent --> Splat
    Agent --> MultiArch
    Agent --> CASA_Gen
    Agent --> GCN
    Agent --> Analysis

    %% Services → Integrations
    Search --> ALminer
    ADS --> ADSClient

    %% Services/Integrations → External
    Agent --> OpenAI
    RLM --> OpenAI
    ALminer --> ALMA_Archive
    ADSClient --> NASA_ADS
    TAP --> ALMA_Archive
    Agent --> Mem0
    RAG --> ChromaDB
    MemSvc --> ChromaDB
    MultiArch --> SIMBAD
    Splat --> Splatalogue_DB
    Browser --> DuckDuckGo

    classDef entry fill:#4A90D9,stroke:#2C5F8A,color:#fff
    classDef core fill:#E8475F,stroke:#B33044,color:#fff
    classDef service fill:#F5A623,stroke:#C17D12,color:#fff
    classDef integration fill:#7ED321,stroke:#5CA018,color:#fff
    classDef external fill:#9B59B6,stroke:#7D3C98,color:#fff

    class CLI,WebUI,NextUI,TG entry
    class Agent,RLM,RLMEnv,Memory,Tools,Prompts core
    class Search,ADS,RAG,MemSvc,Browser,Plotting,Splat,MultiArch,CASA_Gen,GCN,Auth,ConvSvc,Analysis,DataProc,CodeGen,VisSvc service
    class ALminer,ADSClient,TAP,CASAInt,CARTA,DL integration
    class OpenAI,ALMA_Archive,NASA_ADS,Mem0,ChromaDB,SIMBAD,Splatalogue_DB,DuckDuckGo external
```

---

## Layered Architecture

Quasar follows a strict **4-layer** design. Each layer only calls the layer below it.

```mermaid
graph LR
    subgraph L1["Layer 1: Entry Points"]
        direction TB
        A1["Streamlit UI"]
        A2["Next.js UI-Pro"]
        A3["CLI"]
        A4["Telegram Bot"]
    end

    subgraph L2["Layer 2: Core Orchestration"]
        direction TB
        B1["QuasarAgent"]
        B2["RLM Engine"]
        B3["ToolRegistry"]
        B4["ConversationMemory"]
    end

    subgraph L3["Layer 3: Services"]
        direction TB
        C1["SearchService"]
        C2["ADSService"]
        C3["RAGService"]
        C4["BrowserService"]
        C5["PlottingService"]
        C6["+ 11 more services"]
    end

    subgraph L4["Layer 4: Integrations"]
        direction TB
        D1["ALminerClient"]
        D2["ADSClient"]
        D3["NRAOTapClient"]
        D4["CASA / CARTA"]
    end

    L1 --> L2
    L2 --> L3
    L3 --> L4

    style L1 fill:#4A90D9,stroke:#2C5F8A,color:#fff
    style L2 fill:#E8475F,stroke:#B33044,color:#fff
    style L3 fill:#F5A623,stroke:#C17D12,color:#fff
    style L4 fill:#7ED321,stroke:#5CA018,color:#fff
```

---

## Query Processing Flow (Main Pipeline)

```mermaid
sequenceDiagram
    participant U as User
    participant FE as Frontend (UI/CLI)
    participant API as api/main.py
    participant AG as QuasarAgent
    participant RLM as RLM Engine
    participant LLM as OpenAI GPT-4o
    participant TR as ToolRegistry
    participant SVC as Services Layer
    participant EXT as External APIs

    U->>FE: Submit query
    FE->>API: POST /api/chat (SSE)
    API->>AG: stream_response_api(query)

    Note over AG: Step 1: Session pruning check
    AG->>AG: _prune_session_if_needed()

    Note over AG: Step 2: RLM complexity check
    AG->>RLM: should_use_rlm(query)
    RLM->>RLM: Heuristic keyword scan

    alt Complex query (score ≥ 0.55)
        RLM->>LLM: Decompose into subtasks
        LLM-->>RLM: [subtask1, subtask2, ...]
        loop Each subtask
            RLM->>AG: _rlm_tool_executor(subtask)
            AG->>SVC: Execute tool
            SVC->>EXT: Fetch data
            EXT-->>SVC: Results
            SVC-->>AG: Formatted result
            AG-->>RLM: Subtask answer
        end
        RLM->>LLM: Aggregate all results
        LLM-->>RLM: Final synthesis
        RLM-->>AG: Complete answer
    else Simple query
        Note over AG: Step 3: RAG context retrieval
        AG->>SVC: RAGService.search(query)
        SVC-->>AG: ALMA Manual chunks

        Note over AG: Step 4: Long-term memory
        AG->>SVC: MemoryService.search(query)
        SVC-->>AG: User facts/preferences

        Note over AG: Step 5: LLM call with tools
        AG->>LLM: messages + tools + context
        
        alt LLM decides to use a tool
            LLM-->>AG: tool_call(name, args)
            AG->>TR: get_tool(name)
            TR-->>AG: Tool function
            AG->>SVC: tool.execute(**args)
            SVC->>EXT: API call
            EXT-->>SVC: Raw data
            SVC-->>AG: Formatted result
            AG->>LLM: Tool result as context
            LLM-->>AG: Streamed response
        else LLM answers directly
            LLM-->>AG: Streamed response
        end
    end

    AG-->>API: SSE stream chunks
    API-->>FE: SSE events (text, data, tool_call)
    FE-->>U: Rendered response + data tables + plots
```

---

## Agent Initialization — What Connects to What

```mermaid
graph TD
    Agent["QuasarAgent.__init__()"]

    Agent --> M1["ConversationMemory"]
    Agent --> M2["ToolRegistry"]
    Agent --> M3["SearchService"]
    Agent --> M4["RadioAnalysisService"]
    Agent --> M5["RAGService → ChromaDB"]
    Agent --> M6["MemoryService → ChromaDB"]
    Agent --> M7["ADSService → NASA ADS API"]
    Agent --> M8["BrowserService → Playwright"]
    Agent --> M9["PlottingService → Matplotlib"]
    Agent --> M10["SplatalogueTool → Splatalogue API"]
    Agent --> M11["MultiArchiveMatcher → SIMBAD/NED/MAST"]
    Agent --> M12["CASAScriptGenerator"]
    Agent --> M13["GCNAlertMonitor → GWOSC"]
    Agent --> M14["RecursiveLanguageModel"]
    Agent --> M15["mem0 Long-Term Memory"]

    M3 --> I1["ALminerClient → ALMA Archive"]
    M14 --> M16["RLMREPLExecutor → Python Sandbox"]
    M14 --> M17["ComplexityDetector → GPT-4o-mini"]

    style Agent fill:#E8475F,stroke:#B33044,color:#fff
    style M1 fill:#F5A623,stroke:#C17D12,color:#fff
    style M2 fill:#F5A623,stroke:#C17D12,color:#fff
    style M14 fill:#E8475F,stroke:#B33044,color:#fff
```

---

## Registered Tools (27 total)

```mermaid
graph LR
    subgraph Archive["Archive Search (6)"]
        T1["search_by_position"]
        T2["search_by_target"]
        T3["search_by_frequency"]
        T4["search_alma_with_keywords"]
        T5["advanced_search"]
        T6["search_catalog"]
    end

    subgraph Data["Data Operations (5)"]
        T7["get_observation_details"]
        T8["download_data"]
        T9["download_alma_data"]
        T10["filter_results"]
        T11["resolve_target"]
    end

    subgraph Viz["Visualization (3)"]
        T12["plot_alma_results"]
        T13["plot_sky_map"]
        T14["plot_spectrum"]
    end

    subgraph Lines["Spectral Lines (4)"]
        T15["check_line_coverage"]
        T16["check_co_lines"]
        T17["identify_spectral_line"]
        T18["search_lines_by_molecule"]
    end

    subgraph Browse["Web Browsing (4)"]
        T19["web_search"]
        T20["navigate_to_url"]
        T21["read_page"]
        T22["click_element"]
    end

    subgraph Papers["Literature (1)"]
        T23["search_papers"]
    end

    subgraph Multi["Multi-Messenger (4)"]
        T24["cross_match_source"]
        T25["generate_casa_imaging_script"]
        T26["generate_casa_calibration_script"]
        T27["get_latest_gw_events"]
    end
```

---

## Data Storage

```mermaid
graph LR
    subgraph Storage["Data Stores"]
        ChromaDB["chroma_db/<br/>Vector embeddings<br/>(ALMA Manual + memories)"]
        UsersDB["users.db<br/>SQLite<br/>(Auth credentials)"]
        ConvDB["conversations.db<br/>SQLite<br/>(Chat history)"]
        Downloads["downloads/<br/>FITS files"]
        UserUploads["user_uploads/<br/>User data"]
    end

    RAG["RAGService"] --> ChromaDB
    MemSvc["MemoryService"] --> ChromaDB
    Auth["AuthService"] --> UsersDB
    ConvSvc["ConversationService"] --> ConvDB
    Agent["QuasarAgent"] --> Downloads

    style ChromaDB fill:#9B59B6,stroke:#7D3C98,color:#fff
    style UsersDB fill:#9B59B6,stroke:#7D3C98,color:#fff
    style ConvDB fill:#9B59B6,stroke:#7D3C98,color:#fff
```

---

## RLM (Recursive Language Model) Pipeline

```mermaid
flowchart TD
    Q["User Query"] --> CD["ComplexityDetector"]

    CD -->|"Score ≤ 0.2"| SIMPLE["Simple → Standard Agent"]
    CD -->|"0.2 < Score < 0.8"| LLM_CHECK["LLM Assessment<br/>(GPT-4o-mini)"]
    CD -->|"Score ≥ 0.8"| COMPLEX["Complex → RLM Engine"]

    LLM_CHECK -->|"Score < 0.55"| SIMPLE
    LLM_CHECK -->|"Score ≥ 0.55"| COMPLEX

    COMPLEX --> CTX_CHECK{"Context > 2000 chars?"}

    CTX_CHECK -->|Yes| REPL["REPL Mode<br/>RLMREPLExecutor"]
    CTX_CHECK -->|No| DECOMPOSE["LLM Decomposition<br/>_execute_decompose()"]

    REPL --> ENV["RLMEnvironment<br/>(Sandboxed Python)"]
    ENV --> |"Code execution loop"| ENV
    ENV --> ANSWER["answer() called"]

    DECOMPOSE --> SUB["Execute Subtasks<br/>sequentially"]
    SUB --> AGG["Aggregate Results<br/>(LLM synthesis)"]

    ANSWER --> FINAL["Final Response"]
    AGG --> FINAL

    style COMPLEX fill:#E8475F,stroke:#B33044,color:#fff
    style REPL fill:#F5A623,stroke:#C17D12,color:#fff
    style ENV fill:#7ED321,stroke:#5CA018,color:#fff
```

---

## File Index

| File | Layer | One-Line Purpose |
|------|-------|-----------------|
| `quasar.py` | Entry | Main entrypoint, initializes agent & launches UI |
| **core/** | | |
| `agent.py` | Core | Central orchestrator — routes queries, manages tools, streams responses |
| `rlm.py` | Core | Recursive Language Model — decomposes complex multi-hop queries |
| `rlm_environment.py` | Core | Sandboxed Python REPL for RLM context processing |
| `memory.py` | Core | Short-term conversation memory with topic detection |
| `tools.py` | Core | Tool/function registry for OpenAI function calling |
| `prompts.py` | Core | All LLM prompt templates (intent, entity, response, RLM) |
| `cli.py` | Entry | Rich terminal interface for interactive use |
| **services/** | | |
| `search.py` | Service | Facade over ALminerClient for all archive searches |
| `ads_service.py` | Service | NASA ADS paper search with NLP query building |
| `rag_service.py` | Service | RAG via ChromaDB — ingests PDFs, semantic search |
| `memory_service.py` | Service | Long-term user memory using ChromaDB vectors |
| `browser.py` | Service | Headless Chromium web browsing via Playwright |
| `plotting.py` | Service | Publication-quality plots (ApJ/MNRAS style) |
| `splatalogue.py` | Service | Molecular spectral line identification |
| `multi_archive.py` | Service | Cross-match sources across SIMBAD/NED/MAST/VizieR |
| `casa_generator.py` | Service | Generate CASA imaging & calibration scripts |
| `gcn_monitor.py` | Service | GW event monitoring via GWOSC catalog |
| `auth.py` | Service | User registration/login with PBKDF2 hashing |
| `conversation_service.py` | Service | Persistent chat history in SQLite |
| `analysis.py` | Service | UV coverage analysis, imaging parameters |
| `data_processor.py` | Service | FITS file processing and format conversion |
| `code_generator.py` | Service | Python/CASA script generation for analysis |
| `visualization_service.py` | Service | Advanced visualization pipelines |
| `paper_search.py` | Service | Alternative paper search utilities |
| **integrations/** | | |
| `alminer_client.py` | Integration | ALminer library wrapper for ALMA archive |
| `ads_client.py` | Integration | Direct NASA ADS API client |
| `tap.py` | Integration | TAP/VO protocol client for VLA/VLBA/ALMA |
| `tap_astroquery.py` | Integration | Astroquery-based TAP alternative |
| `casa.py` | Integration | CASA software integration for calibration |
| `carta.py` | Integration | CARTA visualization tool integration |
| `datalink.py` | Integration | DataLink protocol client for data access |
| **config/** | | |
| `settings.py` | Config | Centralized configuration (API keys, paths, limits) |
| **utils/** | | |
| `formatters.py` | Utils | Output formatting, unit conversion, coordinates |

---



## Configuration & Environment

```
.env
├── OPENAI_API_KEY          → GPT-4o / GPT-4o-mini
├── NASA_ADS_API_KEY        → NASA ADS paper search
├── TELEGRAM_BOT_TOKEN      → Telegram channel
├── MEM0_API_KEY            → mem0 long-term memory (optional)
└── CHROMA_PERSIST_DIR      → ChromaDB vector store path

config/settings.py
├── Environment             → dev / test / production
├── APIConfig               → Keys and endpoints
├── PathConfig              → Data, cache, download directories
├── SearchConfig            → Max results, timeouts
├── ProcessingConfig        → CASA pipeline parameters
├── UIConfig                → Streamlit settings
├── PerformanceConfig       → Workers, cache TTL, memory limits
└── FeatureFlags            → Enable/disable features
```

---

*Generated: 2026-02-24 | Quasar v3.0*
