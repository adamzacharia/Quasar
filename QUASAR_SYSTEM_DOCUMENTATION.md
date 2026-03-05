# 🔭 QUASAR System Documentation

> **QUASAR** - AI-Powered Radio Astronomy Research Assistant  
> A comprehensive guide to the system architecture, directory structure, and query processing flow.

---

## 📁 Directory Structure Overview

```
Quasar-main/
├── 📂 core/                    # Core AI agent and orchestration logic
├── 📂 services/                # Business logic and external service handlers
├── 📂 integrations/            # External API clients (ALMA, NASA ADS, TAP)
├── 📂 ui/                      # Streamlit web interface + assets
│   └── 📂 assets/              # Logo and static files
├── 📂 config/                  # Configuration management
├── 📂 utils/                   # Utility functions and formatters
├── 📂 tests/                   # Test suites and verification scripts
├── 📂 docs/                    # Documentation and PDFs
│   ├── 📂 pdfs/                # ALMA manuals and guides
│   └── 📂 references/          # Text references and notes
├── 📂 scripts/                 # Utility scripts (ingestion, extraction)
├── 📂 data/                    # Databases (users.db, conversations.db)
├── 📂 test_results/            # Test outputs and logs
├── 📂 chroma_db/               # Vector database for RAG
├── 📂 ALMA_MCP/                # MCP server for Claude/Responses API
├── 📄 requirements.txt         # Python dependencies
├── 📄 setup.sh                 # Setup script
├── 📄 .env                     # Environment variables (API keys)
└── 📄 README.md                # Project readme
```

---

## 📂 Core Directory (`core/`)

The brain of Quasar. Contains the main AI agent, memory management, tool registry, and prompts.

### 📄 `agent.py` - The Main AI Agent (1000 lines)

This is the **heart of QUASAR**. The `QuasarAgent` class orchestrates all interactions between the user, LLM, tools, and external services.

#### Key Classes:

| Class | Description |
|-------|-------------|
| `AgentConfig` | Dataclass holding configuration (API keys, model name, temperature, tokens, verbosity) |
| `QuasarAgent` | Main agent class - handles NLP, tool execution, and response generation |

#### Key Methods in `QuasarAgent`:

| Method | Purpose |
|--------|:--------|
| `__init__()` | Initializes all services (SearchService, ADSService, RAGService, mem0), registers tools, builds system prompt |
| `_build_system_prompt()` | Creates the persona prompt for the LLM |
| `_register_tools()` | Registers all available tools with OpenAI-compatible schemas |
| `determine_intent()` | Uses LLM to classify user intent (SEARCH, QUESTION, PAPERS, CLARIFICATION) |
| `extract_entities()` | Extracts structured entities from natural language |
| `process_query()` | Main query processing - determines intent, extracts entities, routes to appropriate handler |
| `stream_general_response()` | **Chat Completions API** - handles RAG context, tool execution, memory updates |
| `stream_response_api()` | **NEW: Responses API** - uses native MCP support and mem0 long-term memory |
| `set_model()` | Dynamically switch LLM model from UI |
| `generate_summary()` | Creates natural language summaries of search results |
| `_update_memory()` | Extracts and stores user facts/preferences for long-term memory |

#### Registered Tools (13 total):

| Tool Name | Description |
|-----------|-------------|
| `search_by_position` | Search archives by RA/Dec coordinates |
| `search_by_target` | Search by astronomical object name |
| `search_by_frequency` | Search by frequency range (GHz) |
| `get_observation_details` | Get details for a specific observation ID |
| `download_data` | Download observation data |
| `analyze_uv_coverage` | Analyze UV coverage of measurement sets |
| `search_alma_with_keywords` | Search ALMA with specific keywords (PI, project code) |
| `advanced_search` | Execute custom ADQL/TAP queries |
| `plot_alma_results` | Generate visualization (sky, frequency, overview plots) |
| `download_alma_data` | Download ALMA data with dry-run option |
| `search_papers` | Search NASA ADS for research papers |
| `check_line_coverage` | Check if spectral lines are covered in search results |
| `check_co_lines` | Check for CO, 13CO, C18O lines |
| `search_catalog` | Search for a catalog of objects |

---

### 📄 `prompts.py` - LLM Prompts

Contains structured prompts for the LLM to perform specific tasks.

| Prompt | Purpose |
|--------|---------|
| `INTENT_CLASSIFICATION_PROMPT` | Classifies user query into SEARCH, QUESTION, or CLARIFICATION |
| `ENTITY_EXTRACTION_PROMPT` | Extracts astronomical entities (source_name, band, project_code, science_keyword, etc.) |
| `RESPONSE_GENERATION_PROMPT` | Guides the LLM to respond as a helpful PhD student with RAG context |

---

### 📄 `memory.py` - Conversation Memory

Short-term conversation memory using a sliding window approach.

| Class | Purpose |
|-------|---------|
| `ConversationMemory` | Manages conversation history with configurable max turns |

| Method | Purpose |
|--------|---------|
| `add_message()` | Adds a message (user/assistant/system) with timestamp |
| `get_history()` | Returns all messages in conversation |
| `get_last_n_turns()` | Returns last N conversation turns |
| `clear()` | Clears all conversation memory |
| `get_context_summary()` | Generates a topic summary (VLA, pulsars, downloads) |

---

### 📄 `tools.py` - Tool Registry

Defines the tool infrastructure for OpenAI function calling.

| Class | Purpose |
|-------|---------|
| `Tool` | Dataclass representing a single tool (name, description, function, parameters) |
| `ToolRegistry` | Registry managing all available tools |

| Method | Purpose |
|--------|---------|
| `register()` | Adds a tool to the registry |
| `get_tool()` | Retrieves a tool by name |
| `get_openai_tools()` | Converts all tools to OpenAI function-calling format |

---

### 📄 `cli.py` - Command Line Interface

Provides a Rich-based terminal interface for interacting with Quasar.

---

## 📂 Services Directory (`services/`)

Business logic layer that handles specific domains.

### 📄 `ads_service.py` - NASA ADS Integration (529 lines)

Handles paper searches via NASA ADS API.

| Class | Purpose |
|-------|---------|
| `ADSService` | Main service for ADS API interaction |
| `ADSQueryBuilder` | LLM-backed helper that converts natural language to ADS queries |

| Method | Purpose |
|--------|---------|
| `search_papers()` | Search with raw ADS query syntax |
| `search_natural_language()` | **Smart search**: LLM converts question to optimal ADS query |
| `search_by_target()` | Search papers about a specific astronomical target |
| `search_by_facility()` | Search papers mentioning a radio facility (ALMA, VLA, etc.) |
| `search_by_frequency()` | Search papers about observations in a frequency range |
| `get_paper_details()` | Get full details for a specific bibcode |
| `_heuristic_query()` | Fallback heuristic query builder when LLM is unavailable |

---

### 📄 `rag_service.py` - Retrieval Augmented Generation (67 lines)

Vector database for storing and searching documentation (ALMA Manual).

| Class | Purpose |
|-------|---------|
| `RAGService` | Manages document ingestion and semantic search |

| Method | Purpose |
|--------|---------|
| `ingest_document()` | Load PDF, chunk into 1000-char segments, create embeddings, store in ChromaDB |
| `search()` | Semantic similarity search returning top-k relevant chunks |

**Storage**: Uses ChromaDB at `chroma_db/` with OpenAI embeddings.

---

### 📄 `memory_service.py` - Long-Term Memory (86 lines)

Mem0-inspired service for storing user facts/preferences across sessions.

| Class | Purpose |
|-------|---------|
| `MemoryService` | Long-term user memory storage |

| Method | Purpose |
|--------|---------|
| `add_memory()` | Store a user fact with metadata (user_id, timestamp) |
| `search_memories()` | Semantic search for relevant memories by user |
| `get_all_memories()` | Retrieve all memories for a user |

**How it works**: Agent extracts implicit facts from conversations (e.g., "User is interested in protoplanetary disks") and stores them for future context.

---

### 📄 `search.py` - Search Service (131 lines)

High-level search orchestration. Acts as a facade over ALminer client.

| Class | Purpose |
|-------|---------|
| `SearchService` | Coordinates all archive searches |

| Method | Purpose |
|--------|---------|
| `cone_search()` | Search by RA/Dec position |
| `search_by_target()` | Search by object name |
| `search_by_frequency()` | Search by frequency range |
| `advanced_search()` | Execute raw ADQL queries |
| `search_alma_with_keywords()` | Search with specific ALMA keywords |
| `plot_alma_results()` | Generate visualizations |
| `download_alma_data()` | Download data files |
| `check_line_coverage_on_last()` | Check spectral line coverage on cached results |
| `check_co_lines_on_last()` | Check CO isotope lines on cached results |

---

### 📄 `auth.py` - Authentication Service (104 lines)

User registration and login with secure password hashing.

| Class | Purpose |
|-------|---------|
| `AuthService` | Handles user management |

| Method | Purpose |
|--------|---------|
| `register_user()` | Create new user with PBKDF2-hashed password |
| `login_user()` | Verify credentials and return user_id |

**Security**: PBKDF2-HMAC-SHA256 with 100,000 iterations, random 32-byte salt.  
**Storage**: SQLite database (`users.db`).

---

### 📄 `analysis.py` - Radio Analysis Service

Provides analysis functions for radio astronomy data.

| Class | Purpose |
|-------|---------|
| `RadioAnalysisService` | UV coverage analysis, imaging parameters |

---

### 📄 `data_processor.py` - Data Processing

Handles FITS file processing, data extraction, and format conversion.

---

### 📄 `code_generator.py` - Script Generation

Generates CASA scripts, Python snippets, and analysis code.

---

## 📂 Integrations Directory (`integrations/`)

External API clients connecting to real astronomical services.

### 📄 `alminer_client.py` - ALminer Integration (316 lines)

Wrapper around the `alminer` library for ALMA archive access.

| Class | Purpose |
|-------|---------|
| `ALminerClient` | ALMA archive search and data retrieval |

| Method | Purpose |
|--------|---------|
| `search_by_target()` | Query ALMA by source name using `alminer.target()` |
| `search_by_position()` | Cone search using `alminer.conesearch()` |
| `search_by_keywords()` | Search with PI name, project code via `alminer.keysearch()` |
| `search_by_sql()` | Execute custom TAP query via `alminer.query()` |
| `search_by_frequency()` | Search by frequency range |
| `plot_sky_distribution()` | Generate RA/Dec sky map |
| `plot_freq_coverage()` | Generate frequency coverage plot |
| `plot_overview()` | Generate integration time vs sensitivity plot |
| `download_data()` | Download FITS files from archive |
| `get_line_coverage()` | Check spectral line coverage |
| `get_co_lines()` | Check CO isotope lines |
| `_standardize_columns()` | Normalize column names for consistent UI display |

---

### 📄 `ads_client.py` - NASA ADS Client (529 lines)

Alternative ADS client with natural language query support.

---

### 📄 `tap.py` - TAP Protocol Client (460 lines)

Table Access Protocol client for VLA/VLBA data.

| Class | Purpose |
|-------|---------|
| `NRAOTapClient` | TAP-based search for VLA/VLBA archives |

| Method | Purpose |
|--------|---------|
| `search_by_source_name()` | Search across all NRAO facilities |
| `search_vla_vlba()` | Search VLA/VLBA specific data |
| `search_alma()` | Search ALMA via astroquery |
| `test_connection()` | Verify TAP service connectivity |

---

### 📄 `casa.py` - CASA Integration

Integration with CASA (Common Astronomy Software Applications) for data calibration and imaging.

---

### 📄 `carta.py` - CARTA Integration

Integration with CARTA (Cube Analysis and Rendering Tool for Astronomy) for visualization.

---

## 📂 UI Directory (`ui/`)

Streamlit-based web interface with modern design.

### 📄 `app.py` - Main Web Application

The Streamlit interface with:
- **New Q Logo** (circle + diagonal jet representing quasar)
- **Model Selector** dropdown (gpt-4o, gpt-4o-mini, gpt-4.1, gpt-3.5-turbo)
- Animated dark-theme cosmic background
- Chat interface with streaming responses
- User authentication (login/register)
- Data visualization (tables, plots)
- Paper display with arXiv/ADS badges
- **External Links** (GitHub, Documentation, Contributors)
- **Custom Tools** panel with template
- **API Settings** toggle (Responses API vs Chat Completions)

| Function | Purpose |
|----------|:--------|
| `render_header()` | Renders QUASAR logo (base64 encoded) |
| `render_model_selector()` | Model dropdown with "Read more" link |
| `format_message_with_tags()` | Converts @commands to visual badges |
| `display_papers()` | Enhanced paper display with links and badges |
| `display_results_with_summary()` | Data table with AI-generated summary |
| `render_chat_interface()` | Main chat UI with streaming support |
| `render_login_page()` | Authentication UI |

---

### 📄 `components.py` - UI Components

Reusable Streamlit components for:
- Data tables
- Plot displays
- Paper cards
- Progress indicators

---

### 📄 `enhanced_functions.py` - Enhanced UI Functions

Additional UI enhancements:
- Advanced filtering
- Export options
- Interactive plot controls

---

## 📂 Config Directory (`config/`)

### 📄 `settings.py` - Configuration Management (299 lines)

Centralized configuration using dataclasses.

| Class | Purpose |
|-------|---------|
| `Environment` | Enum for dev/test/production |
| `LogLevel` | Logging configuration |
| `APIConfig` | API keys and endpoints |
| `PathConfig` | File paths (data, cache, downloads) |
| `SearchConfig` | Search defaults (max results, timeouts) |
| `ProcessingConfig` | CASA pipeline parameters |
| `UIConfig` | Streamlit settings |
| `PerformanceConfig` | Workers, cache TTL, memory limits |
| `FeatureFlags` | Enable/disable features |
| `Settings` | Main settings aggregator |

---

## 📂 Utils Directory (`utils/`)

### 📄 `formatters.py`

Output formatting utilities for:
- Table formatting
- Unit conversion
- Coordinate formatting

---

## 📂 Tests Directory (`tests/`)

Comprehensive test suite with 31 test files:

| Test File | Purpose |
|-----------|---------|
| `test_quasar.py` | Main agent tests |
| `test_auth_flow.py` | Authentication flow tests |
| `test_alminer.py` | ALminer integration tests |
| `test_rag.py` | RAG service tests |
| `test_memory.py` | Memory service tests |
| `run_regression_suite.py` | Automated regression testing |
| `run_real_evaluation.py` | Live evaluation with real queries |
| `verify_*.py` | Verification scripts for specific features |

---

# 🔄 Query Processing Flow

## Overview Flow Diagram

```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                    UI (app.py)                              │
│  1. Detect command tags (@archive, @search, @paper)         │
│  2. Display "thinking" indicator                            │
│  3. Call agent.stream_general_response()                    │
└────────┬────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                QuasarAgent.stream_general_response()         │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ STEP 1: Check for Simple Confirmations                │  │
│  │ Skip RAG for "yes", "ok", "proceed"                   │  │
│  └───────────────────────────────────────────────────────┘  │
│                          │                                   │
│                          ▼                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ STEP 2: RAG Context Retrieval                         │  │
│  │ - Query ChromaDB for relevant ALMA Manual chunks      │  │
│  │ - Extract source and page number for citations        │  │
│  │ - Display "📘 Consulting ALMA Manual..."              │  │
│  └───────────────────────────────────────────────────────┘  │
│                          │                                   │
│                          ▼                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ STEP 3: Long-Term Memory Retrieval                    │  │
│  │ - Search user's stored facts/preferences              │  │
│  │ - Add as context: "User is interested in..."          │  │
│  └───────────────────────────────────────────────────────┘  │
│                          │                                   │
│                          ▼                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ STEP 4: Command Handling                              │  │
│  │                                                        │  │
│  │ @archive → tool_choice="required", force search tool  │  │
│  │ @search  → tool_choice="none", pure RAG response      │  │
│  │ @paper   → tool_choice="required", force search_papers│  │
│  │ (none)   → tool_choice="auto", LLM decides            │  │
│  └───────────────────────────────────────────────────────┘  │
│                          │                                   │
│                          ▼                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ STEP 5: Build Messages Array                          │  │
│  │ - System prompt (persona)                              │  │
│  │ - RAG context + Long-term memory                       │  │
│  │ - Conversation history (if not @search)                │  │
│  │ - Command-specific instructions                        │  │
│  └───────────────────────────────────────────────────────┘  │
│                          │                                   │
│                          ▼                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ STEP 6: OpenAI API Call with Streaming                │  │
│  │ - model: gpt-4o                                        │  │
│  │ - tools: registered tool schemas (13 tools)            │  │
│  │ - tool_choice: auto/required/none                      │  │
│  │ - stream: true                                          │  │
│  └───────────────────────────────────────────────────────┘  │
│                          │                                   │
│                          ▼                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ STEP 7: Stream Processing Loop                        │  │
│  │ FOR each chunk in stream:                              │  │
│  │   IF content → append to response, update UI          │  │
│  │   IF tool_calls → accumulate function name + args     │  │
│  └───────────────────────────────────────────────────────┘  │
│                          │                                   │
│                          ▼                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ STEP 8: Tool Execution                                │  │
│  │ FOR each tool_call:                                    │  │
│  │   1. Parse arguments from JSON                         │  │
│  │   2. Look up tool in registry                          │  │
│  │   3. Execute: tool.function(**params)                  │  │
│  │   4. Summarize result for context efficiency           │  │
│  │   5. Store result in self.last_run_result             │  │
│  │   6. Add summary to memory                             │  │
│  └───────────────────────────────────────────────────────┘  │
│                          │                                   │
│                          ▼                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ STEP 9: Memory Update                                 │  │
│  │ - Extract user facts from query using GPT-4o-mini     │  │
│  │ - Store in MemoryService for future sessions          │  │
│  └───────────────────────────────────────────────────────┘  │
│                          │                                   │
│                          ▼                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ STEP 10: Return Response                              │  │
│  │ - full_response: streamed text                         │  │
│  │ - self.last_run_result: structured data for UI         │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                    UI (app.py)                              │
│  1. Display streamed response                                │
│  2. Check last_run_result type                               │
│     - "data" → display_results_with_summary()               │
│     - "papers" → display_papers()                            │
│     - "image" → st.image(path)                               │
│  3. Add to chat history                                      │
└─────────────────────────────────────────────────────────────┘
```

---

# 📝 Query Types and Processing

## 1. `@archive` - Archive Search

**Purpose**: Find astronomical observations in ALMA/NRAO archives.

**Processing Flow**:
```
@archive Find Band 6 data for Sz65
         │
         ▼
┌────────────────────────────────────┐
│ Command Detection: @archive        │
│ → tool_choice = "required"        │
│ → System message: MUST use search │
└────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│ LLM forced to use a tool           │
│ Selected: search_by_target         │
│ Arguments: {target_name: "Sz65"}  │
└────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│ Tool Execution                     │
│ → SearchService.search_by_target() │
│ → ALminerClient.search_by_target() │
│ → alminer.target("Sz65")          │
└────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│ Result: DataFrame with columns:    │
│ - target_name, project_code, Band  │
│ - freq_min, freq_max, sensitivity  │
│ - beam_major, beam_minor, PI       │
└────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│ UI displays:                       │
│ 1. Interactive data table          │
│ 2. AI-generated summary            │
│ 3. Plot options (sky, freq)        │
└────────────────────────────────────┘
```

**Available Search Tools**:
- `search_by_target` - By object name (resolves via SIMBAD)
- `search_by_position` - By RA/Dec coordinates
- `search_by_frequency` - By frequency range
- `search_alma_with_keywords` - By PI name, project code
- `advanced_search` - Custom ADQL query

---

## 2. `@search` - Knowledge Search (Pure RAG)

**Purpose**: Answer technical questions using ALMA Manual and documentation.

**Processing Flow**:
```
@search What is the proprietary period for ALMA data?
         │
         ▼
┌────────────────────────────────────┐
│ Command Detection: @search         │
│ → tool_choice = "none"            │
│ → Tools disabled completely        │
│ → Conversation history excluded    │
└────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│ RAG Search                         │
│ → RAGService.search(query, k=3)   │
│ → ChromaDB similarity search       │
│ → Returns ALMA Manual chunks       │
└────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│ Context Assembly                   │
│ "Content: The proprietary period..."│
│ "[Citation: ALMA Manual, Page 42]" │
└────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│ LLM Response (no tools)            │
│ → Uses only RAG context            │
│ → Must cite source and page        │
│ → Pure knowledge answer            │
└────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│ Example Response:                  │
│ "The proprietary period for ALMA   │
│  PI data is 12 months from the     │
│  date of delivery (ALMA Cycle 10   │
│  Handbook, Page 42)."              │
└────────────────────────────────────┘
```

**Key Difference**: 
- No tool execution
- No conversation history (prevents data search contamination)
- Forces citations from RAG context

---

## 3. `@paper` - Paper Search

**Purpose**: Find research papers via NASA ADS.

**Processing Flow**:
```
@paper Find papers on protoplanetary disks around Sz65
         │
         ▼
┌────────────────────────────────────┐
│ Command Detection: @paper          │
│ → tool_choice = "required"        │
│ → System message: MUST use         │
│   search_papers tool               │
└────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│ LLM selects: search_papers         │
│ Arguments: {query: "Sz65           │
│   protoplanetary disk"}           │
└────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────┐
│ ADSService.search_natural_language()                       │
│                                                            │
│ 1. ADSQueryBuilder uses LLM to convert                    │
│    "protoplanetary disks around Sz65"                     │
│    → "object:Sz65 AND (protoplanetary OR disk)"           │
│                                                            │
│ 2. Execute query against NASA ADS API                     │
│    Endpoint: https://api.adsabs.harvard.edu/v1/search     │
│                                                            │
│ 3. Return fields: title, author, year, abstract,          │
│    citation_count, bibcode, doi, pub                       │
└────────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│ UI displays:                       │
│ - Paper cards with badges          │
│   (arXiv preprint vs published)    │
│ - Clickable ADS/DOI links          │
│ - Citation counts                  │
│ - First author + year              │
└────────────────────────────────────┘
```

**Natural Language to ADS Query Examples**:
| Natural Language | ADS Query |
|-----------------|-----------|
| "Papers about Sz65" | `object:"Sz65"` |
| "ALMA papers on Black holes" | `bibstem:ALMA AND (black hole)` |
| "Recent papers on CO emission" | `abs:"CO emission" year:2023-2024` |

---

## 4. General Queries (No Command Tag)

**Purpose**: General conversation, clarification, or mixed queries.

**Processing Flow**:
```
How can I analyze the UV coverage of my data?
         │
         ▼
┌────────────────────────────────────┐
│ No command detected                │
│ → tool_choice = "auto"            │
│ → Include conversation history     │
│ → Include RAG context              │
│ → Include long-term memory         │
└────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│ LLM decides autonomously:          │
│ - If question → answer from RAG    │
│ - If request → may use tools       │
│ - If clarification → ask follow-up │
└────────────────────────────────────┘
```

---

# 🧠 Intent Classification

The agent uses LLM-based intent classification via `determine_intent()`:

```
┌─────────────────────────────────────────────────────────────┐
│                 INTENT CLASSIFICATION                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  SEARCH                                                      │
│  ├─ "Find ALMA data for Sz65"                               │
│  ├─ "Show me Band 6 observations"                           │
│  └─ "Search for VLA data on 3C273"                          │
│                                                              │
│  QUESTION                                                    │
│  ├─ "How do I calibrate data?"                              │
│  ├─ "What is the sensitivity of Band 6?"                    │
│  └─ "Explain the difference between Bands"                  │
│                                                              │
│  PAPERS                                                      │
│  ├─ "Find papers on protoplanetary disks"                   │
│  ├─ "Literature review for Sz65"                            │
│  └─ "What are recent publications about..."                 │
│                                                              │
│  CLARIFICATION                                               │
│  ├─ "Sz65" (after agent asks for target)                    │
│  ├─ "Band 6"                                                 │
│  └─ "Yes, high resolution"                                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Output Format**:
```json
{
  "intent": "SEARCH",
  "confidence": 0.95,
  "reasoning": "User explicitly asked to 'Find' data, indicating search intent."
}
```

---

# 🔍 Entity Extraction

The agent extracts structured entities via `extract_entities()`:

```
Query: "Find Band 6 observations of Sz65 with high sensitivity"
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 ENTITY EXTRACTION                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  {                                                           │
│    "source_name": "Sz65",                                   │
│    "band": 6,                                                │
│    "project_code": null,                                     │
│    "science_keyword": null,                                  │
│    "angular_resolution": null,                               │
│    "sensitivity": "high"                                     │
│  }                                                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

# 💾 Memory Systems

QUASAR has two distinct memory systems:

## Short-Term Memory (Conversation)

```
┌─────────────────────────────────────────────────────────────┐
│              ConversationMemory (core/memory.py)            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Purpose: Track current conversation context                 │
│  Storage: In-memory deque with sliding window                │
│  Capacity: max_turns * 2 messages (default: 20)             │
│                                                              │
│  Example:                                                    │
│  [                                                           │
│    {"role": "user", "content": "Find data for Sz65"},       │
│    {"role": "assistant", "content": "Found 42 obs..."},     │
│    {"role": "user", "content": "Show Band 6 only"},         │
│    ...                                                       │
│  ]                                                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Long-Term Memory (User Profile)

```
┌─────────────────────────────────────────────────────────────┐
│            MemoryService (services/memory_service.py)        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Purpose: Store user facts/preferences across sessions       │
│  Storage: ChromaDB vector store (chroma_db/)                 │
│  Collection: "user_memory"                                   │
│                                                              │
│  How it works:                                               │
│  1. After each query, agent extracts facts:                  │
│     _update_memory() uses GPT-4o-mini to find implicit facts │
│                                                              │
│  2. Example extraction:                                      │
│     Query: "I'm studying the disk around Sz65 for my thesis" │
│     → Fact: "User is studying protoplanetary disk of Sz65"  │
│     → Fact: "User is working on a thesis"                    │
│                                                              │
│  3. Future queries retrieve relevant memories                │
│     Query: "Show me more Band 6 data"                        │
│     → Memory: "User is interested in Sz65"                   │
│     → Adds context for better recommendations                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

# 📊 Data Flow Summary

```
┌──────────┐    Query     ┌───────────┐    API    ┌──────────────┐
│   User   │──────────────│  Streamlit │──────────│  QuasarAgent │
└──────────┘              │   UI      │           └───────┬──────┘
                          └───────────┘                   │
                                                          │
                           ┌──────────────────────────────┼────────┐
                           │                              │        │
                           ▼                              ▼        ▼
                    ┌─────────────┐              ┌─────────┐  ┌─────────┐
                    │ RAGService  │              │ OpenAI  │  │ Memory  │
                    │ (ChromaDB)  │              │  GPT-4o │  │ Service │
                    └─────────────┘              └────┬────┘  └─────────┘
                           │                         │
                           │ context                 │ tool calls
                           │                         │
                           └─────────────────────────┼─────────────────┐
                                                     │                 │
                                                     ▼                 ▼
                                       ┌─────────────────────┐  ┌─────────────┐
                                       │   SearchService     │  │ ADSService  │
                                       └──────────┬──────────┘  └──────┬──────┘
                                                  │                    │
                                                  ▼                    ▼
                                       ┌─────────────────────┐  ┌─────────────┐
                                       │  ALminerClient      │  │ NASA ADS    │
                                       │  (ALMA Archive)     │  │ API         │
                                       └─────────────────────┘  └─────────────┘
```

---

# 🔧 Environment Configuration

Required environment variables in `.env`:

```bash
# Core LLM
OPENAI_API_KEY=sk-...          # Required for GPT-4o

# Paper Search
NASA_ADS_API_KEY=...           # Required for @paper command

# Optional
DEFAULT_LLM_MODEL=gpt-4o       # Model selection
MAX_SEARCH_RESULTS=1000        # Search limit
CACHE_TTL=3600                 # Cache lifetime in seconds
```

---

# 🚀 Running the System

## Start the UI
```bash
cd Quasar-main
streamlit run ui/app.py
```

## Run Tests
```bash
python -m pytest tests/
```

## CLI Mode
```bash
python -m core.cli
```

---

*This documentation was auto-generated by analyzing the Quasar codebase.*
