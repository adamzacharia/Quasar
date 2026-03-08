# Error and Correction Log

This document serves as a persistent record of errors encountered, changes made, files affected, causes, and solutions during our development sessions. 

---

## Template for New Entries

### [Date] - [Short Issue Title]
- **Error/Issue**: (Describe the error or issue encountered)
- **Files Affected**: (List the related or modified files)
- **Cause**: (What was the root cause?)
- **Solution/Changes Made**: (How was it fixed? What code was changed?)
- **Additional Details / Screenshots**: (Any relevant logs, tracebacks, or screenshot references)

---

## Recent Corrections Log

### 2026-03-06 - Fixing ADS Paper Search Empty Arguments in UI-Pro (Responses API)
- **Error/Issue**: Paper search via the UI-Pro chat returned "persistent issue with accessing the paper search function." Backend logs showed `[TOOL CALL] search_papers({})` — the tool was called with empty arguments.
- **Files Affected**: `core/agent.py` (lines 1520–1555, Responses API streaming handler)
- **Cause**: The OpenAI Responses API uses two different IDs for function calls during streaming: `call_id` (on `response.output_item.added`) and `item_id` (on `response.function_call_arguments.delta`). The code only indexed function calls by `call_id`, but the argument delta events arrived keyed by `item_id`. When `call_id ≠ item_id`, argument deltas were silently dropped and tools were invoked with `{}`.
- **Solution/Changes Made**: Added a bidirectional `item_id_to_call_id` mapping dictionary. When a function call item is added, both its `call_id` and `item.id` are mapped to the canonical key. When argument deltas arrive, the raw ID is resolved through this mapping before looking up the function call entry.
- **Additional Details / Screenshots**: Verified in browser — query "Find recent papers about ALMA observations of molecular clouds" now returns 7 live papers from NASA ADS (e.g., "ALMA Survey of Protostellar Outflow-Envelope Interactions and Evolution in Orion-A", 2026).

### 2026-02-27 - Fixing ADS Tool Call
- **Error/Issue**: The agent was not correctly calling the NASA ADS `search_papers` tool, causing it to return fallback data or fail to find relevant papers.
- **Files Affected**: Code related to `ADSQueryBuilder` and tool definitions (Agent / Tool interfaces).
- **Cause**: The `ADSQueryBuilder` was failing to correctly execute live API calls, causing the system to retreat to fallback data.
- **Solution/Changes Made**: Diagnosed the failing query builder and corrected the tool call mechanism to ensure the agent retrieves and uses live API data successfully.
- **Additional Details / Screenshots**: N/A

### 2026-02-24 - Refining Chat UI & Streaming
- **Error/Issue**: Needs finalization of the Thinking Process UI and true real-time streaming from the Responses API.
- **Files Affected**: RAG implementation files, frontend UI components (Thinking Process widget).
- **Cause**: Streaming tokens needed to be seamlessly displayed and RAG citations were missing.
- **Solution/Changes Made**: Verified backend streaming of tokens, ensured frontend displays them seamlessly, auto-collapsing the Thinking widget once streaming starts, and finalized RAG citations.
- **Additional Details / Screenshots**: UI correctly displays the token stream and RAG citations.

### 2026-02-23 - Connecting UI-Pro Backend
- **Error/Issue**: Frontend `ChatArea.tsx` contained hardcoded data and SSE streaming wasn't fully functional.
- **Files Affected**: `ChatArea.tsx`, Store files, FastAPI backend code.
- **Cause**: The UI-Pro frontend components were not integrated with the Quasar backend API.
- **Solution/Changes Made**: Integrated frontend with Quasar backend. Replaced mock responses with live API calls. Fixed the backend's SSE streaming implementation and improved message persistence in the store.
- **Additional Details / Screenshots**: N/A

