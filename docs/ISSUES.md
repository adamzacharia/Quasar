# QUASAR Project Issues & Inconsistencies Audit

**Date:** 2026-03-08
**Branch:** `beta`
**Auditor:** Automated deep-scan across all Python backend, Next.js frontend, configuration, and documentation files.

---

## Table of Contents

1. [CRITICAL — Runtime Crashes & Broken Features](#1-critical--runtime-crashes--broken-features)
   - [1.1 Undefined Method `search_radio_papers()`](#11-undefined-method-search_radio_papers)
   - [1.2 Empty `LIT_TO_CODE_PROMPT` Imported by Agent](#12-empty-lit_to_code_prompt-imported-by-agent)
   - [1.3 Duplicate Tool Registration `plot_alma_results`](#13-duplicate-tool-registration-plot_alma_results)
   - [1.4 Hardcoded JWT Secret in Production Code](#14-hardcoded-jwt-secret-in-production-code)
2. [HIGH — Will Break in Non-Local / Multi-Dev Environments](#2-high--will-break-in-non-local--multi-dev-environments)
   - [2.1 Triple `NRAOTapClient` Class Name Collision](#21-triple-nraotapclient-class-name-collision)
   - [2.2 Hardcoded API URLs in AuthModal.tsx](#22-hardcoded-api-urls-in-authmodaltsx)
   - [2.3 Model Name Mismatch Across Configuration Files](#23-model-name-mismatch-across-configuration-files)
   - [2.4 Duplicate ADS Client Implementations](#24-duplicate-ads-client-implementations)
3. [MEDIUM — Silent Failures & Maintainability Hazards](#3-medium--silent-failures--maintainability-hazards)
   - [3.1 Bare `except: pass` Swallowing All Errors](#31-bare-except-pass-swallowing-all-errors)
   - [3.2 Incomplete `services/__init__.py` Exports](#32-incomplete-services__init__py-exports)
   - [3.3 `integrations/__init__.py` Import Conflict](#33-integrations__init__py-import-conflict)
   - [3.4 FEATURES.md Mislabels Implemented Features](#34-featuresmd-mislabels-implemented-features)
   - [3.5 RLM Engine (862 Lines) Completely Undocumented](#35-rlm-engine-862-lines-completely-undocumented)
   - [3.6 DataLinkClient is a Non-Functional Stub](#36-datalinkclient-is-a-non-functional-stub)
4. [LOW — Code Quality & Minor Gaps](#4-low--code-quality--minor-gaps)
   - [4.1 CARTA Integration Uses Unix-Only Commands on Windows](#41-carta-integration-uses-unix-only-commands-on-windows)
   - [4.2 Duplicate Prompt Definitions](#42-duplicate-prompt-definitions)
   - [4.3 Missing `PyJWT` in requirements.txt](#43-missing-pyjwt-in-requirementstxt)
   - [4.4 Private `_collection` Attribute Access in RAGService](#44-private-_collection-attribute-access-in-ragservice)
   - [4.5 No Transaction Handling in ConversationService](#45-no-transaction-handling-in-conversationservice)
   - [4.6 Hardcoded Google Client ID Fallback](#46-hardcoded-google-client-id-fallback)
5. [Summary Table](#5-summary-table)

---

## 1. CRITICAL — Runtime Crashes & Broken Features

These issues cause immediate exceptions or silently broken features in production.

---

### 1.1 Undefined Method `search_radio_papers()`

| Field | Detail |
|-------|--------|
| **File** | `core/agent.py` line 1899 |
| **Severity** | CRITICAL |
| **Impact** | Runtime `AttributeError` crash when searching papers by source name |

#### What Happens

```python
# core/agent.py:1892-1903
def search_papers(self, source_name: str) -> Optional[pd.DataFrame]:
    """Search for papers using NASA ADS or arXiv fallback"""
    if not source_name:
        return None

    # Use NASA ADS if available
    if self.ads_client:
        papers = self.ads_client.search_radio_papers(source_name)  # <-- THIS METHOD DOES NOT EXIST
        if papers:
            return pd.DataFrame(papers)

    return None
```

#### Why It's an Issue

The `ADSService` class defined in `integrations/ads_client.py` exposes these public methods:

- `search_papers(query, max_results, sort, fields, filters, start)` (line 73)
- `search_natural_language(query, max_results, sort)` (line 114)
- `search_by_target(target_name, max_results)` (line 154)

There is **no** `search_radio_papers()` method. When the agent's internal `search_papers()` helper is called (e.g., after an archive search to find related literature), Python will raise:

```
AttributeError: 'ADSService' object has no attribute 'search_radio_papers'
```

This silently prevents the "related papers" feature from ever working.

#### How to Fix

**Option A** — Call the correct existing method:

```python
# core/agent.py:1899 — replace:
papers = self.ads_client.search_radio_papers(source_name)
# with:
papers = self.ads_client.search_by_target(source_name)
```

**Option B** — If radio-specific filtering is desired, add the method to `ADSService`:

```python
# integrations/ads_client.py — add to ADSService class:
def search_radio_papers(self, target: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """Search ADS for radio-astronomy papers about a target."""
    query = f'object:"{target}" AND (ALMA OR VLA OR radio OR millimeter OR submillimeter)'
    return self.search_papers(query, max_results=max_results)
```

---

### 1.2 Empty `LIT_TO_CODE_PROMPT` Imported by Agent

| Field | Detail |
|-------|--------|
| **File** | `core/prompts.py` line 2, imported at `core/agent.py` line 51 |
| **Severity** | CRITICAL |
| **Impact** | Literature-to-Code feature silently produces empty/garbage output |

#### What Happens

```python
# core/prompts.py:1-2
# core/prompts.py
LIT_TO_CODE_PROMPT = ""          # <-- EMPTY STRING
```

```python
# core/agent.py:51
from core.prompts import LIT_TO_CODE_PROMPT   # imports ""
```

Meanwhile, the **real** prompt with actual content lives in a different file:

```python
# core/prompts/lit_to_code.py:1-15
LIT_TO_CODE_PROMPT = """You are an expert radio astronomy data reduction specialist.
Your task is to read the methodology or data reduction section from a published paper
(provided below) and generate a functional Python/CASA script that replicates their
data processing steps.

Focus on extracting and applying key parameters mentioned in the text:
- Calibration steps (bandpass, phase, flux calibrators)
- Imaging parameters (cell size, image size, weighting scheme like Briggs robust, UV tapering)
- Cleaning thresholds (RMS noise)
...
"""
```

#### Why It's an Issue

The agent imports the empty string from `core/prompts.py` (a flat module), not from the `core/prompts/` package directory. When the Literature-to-Code pipeline runs, it sends an empty system prompt to the LLM, meaning:

1. The LLM has no instructions on how to extract methodology or generate CASA scripts.
2. Output will be generic/useless rather than structured code.
3. Test file `test_lit_to_code_end_to_end.py` may appear to "pass" but produces low-quality output.

#### How to Fix

**Option A** — Fix the import in `core/agent.py` to use the correct source:

```python
# core/agent.py:51 — replace:
from core.prompts import LIT_TO_CODE_PROMPT
# with:
from core.prompts.lit_to_code import LIT_TO_CODE_PROMPT
```

**Option B** — Copy the real prompt content into `core/prompts.py` to keep all prompts centralized:

```python
# core/prompts.py:2 — replace:
LIT_TO_CODE_PROMPT = ""
# with the full prompt text from core/prompts/lit_to_code.py
```

**Option C** — Make `core/prompts.py` re-export from the submodule:

```python
# core/prompts.py:2 — replace:
LIT_TO_CODE_PROMPT = ""
# with:
from core.prompts.lit_to_code import LIT_TO_CODE_PROMPT
```

> Note: This requires careful handling since `core/prompts.py` (a file) and `core/prompts/` (a directory) coexist. Python can only import from one. The cleanest approach is Option A.

---

### 1.3 Duplicate Tool Registration `plot_alma_results`

| Field | Detail |
|-------|--------|
| **File** | `core/agent.py` lines 356-367 and lines 533-550 |
| **Severity** | CRITICAL |
| **Impact** | Only one implementation runs; the other is silently overwritten |

#### What Happens

**First registration (line 356):**

```python
self.tool_registry.register(Tool(
    name="plot_alma_results",
    description="Generate visualization for valid ALMA search results. Must have searched first.",
    function=self._plot_alma_results,     # Uses search_service.plot_alma_results()
    parameters={
        "type": "object",
        "properties": {
            "plot_type": {"type": "string", "enum": ["sky", "frequency", "overview"], ...}
        },
        "required": ["plot_type"]
    }
))
```

**Second registration (line 533):**

```python
self.tool_registry.register(Tool(
    name="plot_alma_results",
    description="Generate a publication-quality scatter plot (ApJ/MNRAS style, 300 DPI, colorblind-safe)...",
    function=lambda **kw: self.plotting_service.plot_alma_results(
        data_records=(...),
        **kw
    ),
    parameters={
        "type": "object",
        "properties": {
            "x_column": {"type": "string", ...},
            "y_column": {"type": "string", ...},
        },
        ...
    }
))
```

#### Why It's an Issue

The `ToolRegistry` is a dictionary keyed by tool name. The second registration overwrites the first. This means:

1. The first implementation (`self._plot_alma_results` which uses `search_service`) is silently discarded.
2. The OpenAI function schema sent to the LLM describes `x_column`/`y_column` parameters, but the LLM may also see the old description in cached prompts.
3. If the intended behavior was `_plot_alma_results` (which has `plot_type` parameter), the LLM will call with `x_column`/`y_column` and the handler won't understand.
4. Two different implementations means two different behaviors — it's unclear which one is "correct."

#### How to Fix

**Option A** — Remove the duplicate and keep the publication-quality version (line 533):

```python
# Delete lines 356-367 entirely (the first registration)
```

**Option B** — Rename the second to a different tool:

```python
# Line 534 — change name:
name="plot_alma_publication",
```

**Option C** — Merge both into one handler that supports both parameter sets:

```python
def _plot_alma_results(self, plot_type: str = None, x_column: str = None, y_column: str = None, **kw):
    if x_column and y_column:
        return self.plotting_service.plot_alma_results(data_records=..., x_column=x_column, y_column=y_column, **kw)
    else:
        return self._generate_basic_plot(plot_type or "sky")
```

---

### 1.4 Hardcoded JWT Secret in Production Code

| Field | Detail |
|-------|--------|
| **File** | `services/auth.py` line 17 |
| **Severity** | CRITICAL (Security) |
| **Impact** | Anyone with source code access can forge authentication tokens |

#### What Happens

```python
# services/auth.py:17
JWT_SECRET = os.environ.get("JWT_SECRET", "quasar-research-assistant-super-secret-key-2026")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24 * 7  # 1 week
```

#### Why It's an Issue

If the `JWT_SECRET` environment variable is not set (which is the default since it's not documented in `.env.example` or anywhere), the application falls back to a hardcoded secret string visible in the public source code. An attacker can:

1. Read the secret from the repository.
2. Generate valid JWT tokens for any user ID.
3. Authenticate as any user, access their personalized RAG documents, conversation history, and any protected endpoints.

The tokens last 1 week, giving a wide window for exploitation.

#### How to Fix

**Step 1** — Generate a strong secret and add to `.env`:

```bash
# .env
JWT_SECRET=<generate with: python -c "import secrets; print(secrets.token_urlsafe(64))">
```

**Step 2** — Refuse to start if the secret is not configured:

```python
# services/auth.py:17 — replace:
JWT_SECRET = os.environ.get("JWT_SECRET", "quasar-research-assistant-super-secret-key-2026")
# with:
JWT_SECRET = os.environ.get("JWT_SECRET")
if not JWT_SECRET:
    raise RuntimeError(
        "JWT_SECRET environment variable is required. "
        "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(64))\""
    )
```

**Step 3** — Add `JWT_SECRET` to `.env.example` as a required variable.

---

## 2. HIGH — Will Break in Non-Local / Multi-Dev Environments

These issues work on localhost but break when deploying, sharing, or scaling.

---

### 2.1 Triple `NRAOTapClient` Class Name Collision

| Field | Detail |
|-------|--------|
| **Files** | `integrations/tap.py:25`, `integrations/tap_astroquery.py:19`, `integrations/tap_fixed.py:18` |
| **Severity** | HIGH |
| **Impact** | Import shadowing, wrong implementation used silently |

#### What Happens

Three separate files each define a class with the identical name `NRAOTapClient`:

```python
# integrations/tap.py:25
class NRAOTapClient:
    """Fixed client for interacting with NRAO's TAP service - Version 2"""

# integrations/tap_astroquery.py:19
class NRAOTapClient:
    """..."""

# integrations/tap_fixed.py:18
class NRAOTapClient:
    """Fixed client for interacting with NRAO's TAP service - Version 2"""
```

Additionally, `integrations/__init__.py` exports from `tap.py`:

```python
# integrations/__init__.py:7
from .tap import NRAOTapClient
```

#### Why It's an Issue

1. **Silent shadowing**: If any code does `from integrations.tap_fixed import NRAOTapClient` after importing from `integrations`, the second import overwrites the first.
2. **Developer confusion**: It's impossible to know which implementation is "correct" without reading all three files.
3. **Maintenance nightmare**: Bug fixes applied to one file won't apply to the others.
4. **`__init__.py` imports from `tap.py`** but the agent has that import commented out (`# from integrations.tap import NRAOTapClient`), suggesting `tap.py` may be broken or deprecated.

#### How to Fix

**Step 1** — Decide on the canonical implementation (likely `tap_astroquery.py` since the agent comment says TAP is "broken"):

```
integrations/tap_astroquery.py  → KEEP (rename class to ALMAQueryClient or keep NRAOTapClient)
integrations/tap.py             → DELETE or rename to tap_legacy.py
integrations/tap_fixed.py       → DELETE or rename to tap_fixed_legacy.py
```

**Step 2** — Update `integrations/__init__.py`:

```python
from .tap_astroquery import NRAOTapClient
```

**Step 3** — If legacy versions must be kept, rename their classes:

```python
# integrations/tap_fixed.py
class NRAOTapClientFixed:  # or NRAOTapClientLegacy
```

---

### 2.2 Hardcoded API URLs in AuthModal.tsx

| Field | Detail |
|-------|--------|
| **File** | `ui-pro/src/components/AuthModal.tsx` lines 62 and 90 |
| **Severity** | HIGH |
| **Impact** | Authentication completely breaks when deployed to any non-localhost environment |

#### What Happens

```typescript
// AuthModal.tsx:62
const res = await fetch("http://localhost:8000/api/auth/google", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ credential: response.credential }),
});

// AuthModal.tsx:90
const res = await fetch(`http://localhost:8000${endpoint}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
});
```

Meanwhile, every other component in the app correctly uses environment variables:

```typescript
// SettingsModal.tsx:7
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// lib/api.ts
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
```

#### Why It's an Issue

When the app is deployed to a server (e.g., `https://quasar.example.com`):

1. Every feature works except login/registration.
2. Auth requests still go to `http://localhost:8000` on the user's machine, which either fails silently or connects to a completely wrong service.
3. Google OAuth will also fail because the callback URL won't match.
4. This is especially insidious because it works perfectly during development.

#### How to Fix

```typescript
// AuthModal.tsx — add at top of file (after imports):
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Line 62 — replace:
const res = await fetch("http://localhost:8000/api/auth/google", {
// with:
const res = await fetch(`${API_BASE}/api/auth/google`, {

// Line 90 — replace:
const res = await fetch(`http://localhost:8000${endpoint}`, {
// with:
const res = await fetch(`${API_BASE}${endpoint}`, {
```

---

### 2.3 Model Name Mismatch Across Configuration Files

| Field | Detail |
|-------|--------|
| **Files** | `config/settings.py:36`, `config/__init__.py:41`, `integrations/ads_client.py:446`, `.env` |
| **Severity** | HIGH |
| **Impact** | Different parts of the app may use different (possibly deprecated) LLM models |

#### What Happens

| Location | Default Model | Used When |
|----------|---------------|-----------|
| `config/settings.py:36` | `"gpt-4-turbo-preview"` | Settings object used by old Streamlit UI |
| `config/__init__.py:41` | `"gpt-4o"` | Legacy Config wrapper |
| `integrations/ads_client.py:446` | `"gpt-4o-mini"` | ADS natural-language query builder |
| `services/ads_service.py:598` | `"gpt-4o-mini"` | ADS query builder (services copy) |
| `.env` (actual) | `DEFAULT_LLM_MODEL=gpt-4o-mini` | Runtime override |
| System docs | `gpt-4o` | Documented default |

```python
# config/settings.py:36 — the worst offender:
openai_model: str = field(
    default_factory=lambda: os.getenv("DEFAULT_LLM_MODEL", "gpt-4-turbo-preview")
)
# "gpt-4-turbo-preview" is a deprecated model alias from early 2024
```

#### Why It's an Issue

1. **`gpt-4-turbo-preview`** is a deprecated model name. OpenAI may remove it, causing API errors.
2. If `DEFAULT_LLM_MODEL` env var is unset, different components silently pick different models, leading to inconsistent behavior and costs.
3. Documentation says one thing, code does another — confusing for contributors.

#### How to Fix

**Step 1** — Standardize the fallback across all files to `"gpt-4o-mini"` (the current `.env` value):

```python
# config/settings.py:36 — replace "gpt-4-turbo-preview":
openai_model: str = field(default_factory=lambda: os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini"))

# config/__init__.py:41 — replace "gpt-4o":
return os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")
```

**Step 2** — Add a comment documenting the canonical default:

```python
# All model fallbacks should use "gpt-4o-mini" to match .env convention
```

---

### 2.4 Duplicate ADS Client Implementations

| Field | Detail |
|-------|--------|
| **Files** | `integrations/ads_client.py` (557 lines) and `services/ads_service_old.py` (267 lines) |
| **Severity** | HIGH |
| **Impact** | Maintenance confusion, potential wrong import, diverging bug fixes |

#### What Happens

Two files implement NASA ADS integration, both defining a class called `ADSService`:

```python
# integrations/ads_client.py:29
class ADSService:
    """Service for interacting with the NASA ADS API."""
    # 557 lines, modern, with retry logic, query builder, LLM-powered NL search

# services/ads_service_old.py (filename literally says "old")
class ADSService:
    # 267 lines, simpler, no retry logic, no NL query builder
```

The agent uses `integrations/ads_client.py`:

```python
# core/agent.py:37
from integrations.ads_client import ADSService
```

But `services/ads_service_old.py` still exists and could be imported by accident.

#### Why It's an Issue

1. Same class name in two locations — IDE auto-imports may pick the wrong one.
2. Bug fixes to the active version won't propagate to the old one.
3. New developers won't know which is canonical.
4. The old file still passes import checks, so nothing flags it as dead code.

#### How to Fix

**Option A** — Delete the old file:

```bash
git rm services/ads_service_old.py
```

**Option B** — If keeping for reference, rename and add deprecation:

```python
# services/ads_service_old.py → services/_ads_service_deprecated.py
# Add at top:
"""DEPRECATED: Use integrations.ads_client.ADSService instead. Kept for reference only."""
import warnings
warnings.warn("ads_service_old is deprecated, use integrations.ads_client", DeprecationWarning)
```

---

## 3. MEDIUM — Silent Failures & Maintainability Hazards

These issues don't crash immediately but cause hard-to-debug problems.

---

### 3.1 Bare `except: pass` Swallowing All Errors

| Field | Detail |
|-------|--------|
| **Files** | `services/rag_service.py` (6 instances), `services/search.py` (1), `services/visualization_service.py` (1) |
| **Severity** | MEDIUM |
| **Impact** | Silent failures make debugging nearly impossible |

#### What Happens

```python
# services/rag_service.py:72-73
try:
    self.general_store = Chroma(
        persist_directory=self.general_dir,
        embedding_function=self.embeddings
    )
except:        # <-- catches EVERYTHING: KeyboardInterrupt, SystemExit, MemoryError...
    pass       # <-- silently continues as if nothing happened

# Same pattern at lines 82-83, 92-93, 300, 309, 328
```

```python
# services/search.py:104
try:
    return self.alminer_client.search_by_target(source_name)
except:
    return pd.DataFrame()    # Returns empty — user never knows WHY the search failed
```

#### Why It's an Issue

1. **Catches fatal exceptions**: `KeyboardInterrupt`, `SystemExit`, `MemoryError`, `GeneratorExit` should never be silently caught.
2. **No logging**: When ChromaDB fails to load (corrupt database, version mismatch, missing files), the error is silently eaten. The user sees no documents in RAG with zero indication of why.
3. **Debugging black hole**: When something doesn't work, there's no traceback, no log, no error message — just silence.
4. **Masking real bugs**: A typo in a column name, a missing dependency, or a network error all look identical: "it just doesn't work."

#### Locations

| File | Line(s) | Context |
|------|---------|---------|
| `services/rag_service.py` | 72, 82, 92 | ChromaDB store initialization |
| `services/rag_service.py` | 300, 309 | Collection stats retrieval |
| `services/rag_service.py` | 328 | Personal document listing |
| `services/search.py` | 104 | ALMA target search |
| `services/visualization_service.py` | 167 | Temp file cleanup |

#### How to Fix

Replace every bare `except: pass` with specific exception handling and logging:

```python
import logging
logger = logging.getLogger(__name__)

# services/rag_service.py:72 — replace:
except:
    pass
# with:
except Exception as e:
    logger.warning(f"Failed to load general ChromaDB store from {self.general_dir}: {e}")

# services/search.py:104 — replace:
except:
    return pd.DataFrame()
# with:
except Exception as e:
    logger.error(f"ALMA target search failed for '{source_name}': {e}")
    return pd.DataFrame()

# services/visualization_service.py:167 — replace (cleanup is OK to be lenient):
except:
    pass
# with:
except OSError:
    pass
```

---

### 3.2 Incomplete `services/__init__.py` Exports

| Field | Detail |
|-------|--------|
| **File** | `services/__init__.py` |
| **Severity** | MEDIUM |
| **Impact** | `from services import RAGService` fails; inconsistent import patterns across codebase |

#### What Happens

```python
# services/__init__.py (complete file):
"""
Quasar Services Module
Business logic and data processing
"""

from .search import SearchService
from .analysis import RadioAnalysisService

__all__ = [
    'SearchService',
    'RadioAnalysisService'
]
```

The module has 10+ services but only exports 2.

#### Why It's an Issue

Services actually defined in the `services/` directory:

| Service | File | Exported? |
|---------|------|-----------|
| `SearchService` | `search.py` | Yes |
| `RadioAnalysisService` | `analysis.py` | Yes |
| `RAGService` | `rag_service.py` | **No** |
| `ConversationService` | `conversation_service.py` | **No** |
| `AuthService` | `auth.py` | **No** |
| `PlottingService` | `plotting.py` | **No** |
| `BrowserService` | `browser.py` | **No** |
| `MemoryService` | `memory_service.py` | **No** |
| `ProposalCriticService` | `proposal_critic.py` | **No** |
| `NotebookGenerator` | `notebook_gen.py` | **No** |
| `PDFProcessingService` | `pdf_processing.py` | **No** |
| `FITSProcessingService` | `fits_processing.py` | **No** |

This forces each consumer to use long-form imports like `from services.rag_service import RAGService` instead of `from services import RAGService`.

#### How to Fix

Update `services/__init__.py` to export all public services:

```python
from .search import SearchService
from .analysis import RadioAnalysisService
from .rag_service import RAGService
from .memory_service import MemoryService
from .plotting import PlottingService
from .browser import BrowserService
from .proposal_critic import ProposalCriticService
from .notebook_gen import generate_analysis_notebook
from .pdf_processing import PDFProcessingService
from .fits_processing import FITSProcessingService

__all__ = [
    'SearchService', 'RadioAnalysisService', 'RAGService',
    'MemoryService', 'PlottingService', 'BrowserService',
    'ProposalCriticService', 'generate_analysis_notebook',
    'PDFProcessingService', 'FITSProcessingService',
]
```

---

### 3.3 `integrations/__init__.py` Import Conflict

| Field | Detail |
|-------|--------|
| **File** | `integrations/__init__.py` line 7 |
| **Severity** | MEDIUM |
| **Impact** | `from integrations import NRAOTapClient` may crash or import wrong class |

#### What Happens

```python
# integrations/__init__.py:7
from .tap import NRAOTapClient       # imports from tap.py
```

But the agent has this import commented out:

```python
# core/agent.py:35
# from integrations.tap import NRAOTapClient    # <-- DISABLED
```

This suggests `tap.py` may be broken. Yet `__init__.py` still tries to import from it, meaning `import integrations` itself could fail.

#### How to Fix

Align `__init__.py` with the actual working implementation:

```python
# If tap_astroquery.py is the working version:
from .tap_astroquery import NRAOTapClient
from .datalink import DataLinkClient
from .ads_client import ADSService

__all__ = ['NRAOTapClient', 'DataLinkClient', 'ADSService']
```

---

### 3.4 FEATURES.md Mislabels Implemented Features

| Field | Detail |
|-------|--------|
| **File** | `FEATURES.md` |
| **Severity** | MEDIUM |
| **Impact** | Misleading documentation; contributors may re-implement existing features |

#### Mislabeled Features

| Feature | Marked As | Actual Status | Evidence |
|---------|-----------|---------------|----------|
| Literature-to-Code (U3) | Planned | **Implemented** | `core/prompts/lit_to_code.py`, `test_lit_to_code_end_to_end.py` exist |
| Chat with Data Cube (W3) | Planned | **Partially implemented** | `services/fits_processing.py` has metadata extraction and preview |
| RLM Engine | Not mentioned | **Implemented** | `core/rlm.py` (402 lines), `core/rlm_environment.py` (462 lines) |
| Gemini Model Support | Not mentioned | **Implemented** | `ui-pro/api/main.py` lists 9 Gemini model variants |

#### How to Fix

Update `FEATURES.md` to accurately reflect the current state. Mark Literature-to-Code as built, add RLM and Gemini support as features, and note the partial status of FITS/data cube support.

---

### 3.5 RLM Engine (862 Lines) Completely Undocumented

| Field | Detail |
|-------|--------|
| **Files** | `core/rlm.py` (402 lines), `core/rlm_environment.py` (462 lines) |
| **Severity** | MEDIUM |
| **Impact** | Major architectural component invisible to contributors and users |

#### What Happens

The Recursive Language Model (RLM) engine is a significant feature that:

- Decomposes complex queries into sub-tasks
- Provides a REPL sandbox environment for code execution
- Routes between "simple" and "complex" query paths in the API
- Is actively used in `ui-pro/api/main.py` for complexity analysis

Yet it appears in zero documentation files: not in FEATURES.md, README.md, or any architecture docs.

#### How to Fix

Add a section to FEATURES.md or create `docs/rlm.md` describing:

1. What RLM is and why it exists
2. How it routes queries (simple vs. complex)
3. The decomposition and aggregation prompts
4. The sandbox environment capabilities

---

### 3.6 DataLinkClient is a Non-Functional Stub

| Field | Detail |
|-------|--------|
| **File** | `integrations/datalink.py` (26 lines) |
| **Severity** | MEDIUM |
| **Impact** | Users think they can download data but the feature does nothing |

#### What Happens

```python
# integrations/datalink.py:16-26
def download_observation(self, obs_id: str, output_dir: str = "./data") -> Dict[str, Any]:
    """Download observation data"""
    # Placeholder implementation
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    return {
        "status": "download_queued",
        "obs_id": obs_id,
        "output_dir": output_dir,
        "message": "Download functionality requires authentication setup"
    }
```

The function creates a directory but **never actually downloads anything**. It returns `"status": "download_queued"` which suggests to the caller that a download is in progress.

#### Why It's an Issue

1. The agent registers a `download_alma_data` tool (line 369-380 of `agent.py`) that calls this stub.
2. The LLM will tell users "download has been queued" — a lie.
3. Users will wait indefinitely for data that never arrives.

#### How to Fix

**Option A** — Implement actual download using NRAO DataLink API (requires auth integration).

**Option B** — Make the stub honest:

```python
def download_observation(self, obs_id: str, output_dir: str = "./data") -> Dict[str, Any]:
    return {
        "status": "not_implemented",
        "error": "ALMA data download is not yet implemented. "
                 "Please download manually from https://almascience.nrao.edu/aq/",
        "obs_id": obs_id,
    }
```

---

## 4. LOW — Code Quality & Minor Gaps

These are minor issues that reduce code quality but don't break functionality.

---

### 4.1 CARTA Integration Uses Unix-Only Commands on Windows

| Field | Detail |
|-------|--------|
| **File** | `integrations/carta.py` line 64 |
| **Severity** | LOW |
| **Impact** | CARTA detection always fails on Windows |

#### What Happens

```python
# integrations/carta.py:61-69
def _command_exists(self, command: str) -> bool:
    """Check if command exists in PATH"""
    try:
        subprocess.run(["which", command],     # <-- Unix-only command
                     capture_output=True,
                     check=True)
        return True
    except:
        return False
```

The project is running on Windows 11, where `which` does not exist (the equivalent is `where`).

#### How to Fix

```python
import shutil

def _command_exists(self, command: str) -> bool:
    """Check if command exists in PATH (cross-platform)"""
    return shutil.which(command) is not None
```

`shutil.which()` is a cross-platform Python standard library function that works on both Windows and Unix.

---

### 4.2 Duplicate Prompt Definitions

| Field | Detail |
|-------|--------|
| **Files** | `core/prompts.py:2` and `core/prompts/lit_to_code.py:1` |
| **Severity** | LOW |
| **Impact** | Confusing — two definitions of the same variable with different values |

`LIT_TO_CODE_PROMPT` is defined as `""` in `core/prompts.py` and as a full 15-line prompt in `core/prompts/lit_to_code.py`. This is the root cause of issue 1.2.

#### How to Fix

Remove the empty definition from `core/prompts.py`:

```python
# core/prompts.py:2 — DELETE this line:
LIT_TO_CODE_PROMPT = ""
```

And ensure all imports point to `core/prompts/lit_to_code.py`.

---

### 4.3 Missing `PyJWT` in requirements.txt

| Field | Detail |
|-------|--------|
| **File** | `requirements.txt` |
| **Severity** | LOW |
| **Impact** | Fresh installs may fail with `ModuleNotFoundError: No module named 'jwt'` |

#### What Happens

```python
# services/auth.py:14
import jwt    # This is the PyJWT package
```

But `requirements.txt` (159 lines) does not include `PyJWT`. It may be installed as a transitive dependency of another package, but that's not guaranteed.

#### How to Fix

Add to `requirements.txt`:

```
PyJWT>=2.8.0
```

---

### 4.4 Private `_collection` Attribute Access in RAGService

| Field | Detail |
|-------|--------|
| **File** | `services/rag_service.py` lines 298 and 307 |
| **Severity** | LOW |
| **Impact** | Will break if ChromaDB/LangChain changes internal API |

#### What Happens

```python
# services/rag_service.py:298
stats["general"] = {
    "status": "ready",
    "count": self.general_store._collection.count()    # <-- private attribute
}
```

`_collection` is a private/internal attribute of LangChain's `Chroma` class. It could be renamed or removed in any minor version update.

#### How to Fix

Use the public API instead. LangChain's `Chroma` wrapper exposes a `_collection` but the more stable approach is:

```python
# Option A — use the wrapper's own method if available:
docs = self.general_store.get()
count = len(docs.get('ids', []))

# Option B — if count() is truly needed, pin the chromadb version and add a comment:
# NOTE: Accessing _collection.count() — fragile, depends on chromadb internals.
# Pin chromadb version in requirements.txt to avoid breakage.
```

---

### 4.5 No Transaction Handling in ConversationService

| Field | Detail |
|-------|--------|
| **File** | `services/conversation_service.py` lines 120-144 |
| **Severity** | LOW |
| **Impact** | Interrupted saves could leave database in inconsistent state |

#### What Happens

```python
# services/conversation_service.py:120-124
conn = sqlite3.connect(self.db_path)
cursor = conn.cursor()

# Delete existing messages
cursor.execute('DELETE FROM messages WHERE conversation_id = ?', (conversation_id,))

# Insert all messages (loop follows)
# ...
conn.commit()
```

If the process crashes between the DELETE and the final `conn.commit()`, all messages for that conversation are permanently lost with nothing to replace them.

#### How to Fix

Wrap in an explicit transaction:

```python
conn = sqlite3.connect(self.db_path)
try:
    cursor = conn.cursor()
    cursor.execute('BEGIN TRANSACTION')
    cursor.execute('DELETE FROM messages WHERE conversation_id = ?', (conversation_id,))
    # ... insert messages ...
    conn.commit()
except Exception:
    conn.rollback()
    raise
finally:
    conn.close()
```

Or use Python's context manager:

```python
with sqlite3.connect(self.db_path) as conn:
    cursor = conn.cursor()
    cursor.execute('DELETE FROM messages WHERE conversation_id = ?', (conversation_id,))
    # ... insert messages ...
    # commit is automatic on success, rollback on exception
```

---

### 4.6 Hardcoded Google Client ID Fallback

| Field | Detail |
|-------|--------|
| **File** | `ui-pro/src/components/AuthModal.tsx` line ~32 |
| **Severity** | LOW |
| **Impact** | Exposes a Google OAuth client ID in source code; may confuse deployments |

#### What Happens

A Google OAuth client ID is hardcoded as a fallback when `NEXT_PUBLIC_GOOGLE_CLIENT_ID` is not set. While client IDs are not secrets (they're embedded in HTML), hardcoding it means:

1. Forks will accidentally use the original project's OAuth client.
2. OAuth consent screens may show wrong branding.

#### How to Fix

Remove the fallback and require the env var:

```typescript
const googleClientId = process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID;
// Only render Google login button if client ID is configured
{googleClientId && (
    <div id="google-signin-button" />
)}
```

---

## 5. Summary Table

| # | Issue | Severity | File(s) | Type |
|---|-------|----------|---------|------|
| 1.1 | `search_radio_papers()` undefined | **CRITICAL** | `core/agent.py:1899` | Runtime crash |
| 1.2 | Empty `LIT_TO_CODE_PROMPT` imported | **CRITICAL** | `core/prompts.py:2`, `core/agent.py:51` | Silent failure |
| 1.3 | Duplicate `plot_alma_results` tool | **CRITICAL** | `core/agent.py:357,533` | Overwritten handler |
| 1.4 | Hardcoded JWT secret | **CRITICAL** | `services/auth.py:17` | Security vulnerability |
| 2.1 | Triple `NRAOTapClient` collision | **HIGH** | `integrations/tap*.py` (3 files) | Import shadowing |
| 2.2 | Hardcoded URLs in AuthModal | **HIGH** | `ui-pro/.../AuthModal.tsx:62,90` | Deployment breakage |
| 2.3 | Model name mismatch | **HIGH** | `config/settings.py:36` + 3 others | Config inconsistency |
| 2.4 | Duplicate ADS implementations | **HIGH** | `integrations/ads_client.py`, `services/ads_service_old.py` | Maintenance hazard |
| 3.1 | Bare `except: pass` (8 instances) | **MEDIUM** | `services/rag_service.py` + 2 others | Silent failures |
| 3.2 | Incomplete `__init__.py` exports | **MEDIUM** | `services/__init__.py` | Missing exports |
| 3.3 | `integrations/__init__.py` conflict | **MEDIUM** | `integrations/__init__.py:7` | Import may crash |
| 3.4 | FEATURES.md mislabeled | **MEDIUM** | `FEATURES.md` | Misleading docs |
| 3.5 | RLM engine undocumented | **MEDIUM** | `core/rlm.py`, `core/rlm_environment.py` | Missing docs |
| 3.6 | DataLinkClient is a stub | **MEDIUM** | `integrations/datalink.py` | Fake functionality |
| 4.1 | Unix-only `which` on Windows | **LOW** | `integrations/carta.py:64` | Platform compat |
| 4.2 | Duplicate prompt definitions | **LOW** | `core/prompts.py`, `core/prompts/lit_to_code.py` | Code duplication |
| 4.3 | Missing `PyJWT` in requirements | **LOW** | `requirements.txt` | Install breakage |
| 4.4 | Private `_collection` access | **LOW** | `services/rag_service.py:298,307` | Fragile API use |
| 4.5 | No SQLite transactions | **LOW** | `services/conversation_service.py` | Data integrity |
| 4.6 | Hardcoded Google client ID | **LOW** | `ui-pro/.../AuthModal.tsx` | Config hygiene |

**Total: 4 Critical, 4 High, 6 Medium, 6 Low = 20 issues**
