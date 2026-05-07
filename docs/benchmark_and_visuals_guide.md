# PAI26 Paper: Benchmarks, Visuals & Conference Context

## Prior Conference Research

> [!IMPORTANT]
> PAI26 is the **first edition** of this specific conference. It is co-organized by the same team behind the NeurIPS Machine Learning and Physical Sciences (ML4PS) workshop, which has run since 2017.

### NeurIPS ML4PS Workshop (PAI26's predecessor)
- **ML4PS 2024** (NeurIPS 2024, Vancouver) — ~100 accepted 4-page papers
- **ML4PS 2023** (NeurIPS 2023, New Orleans) — similar scale
- Papers list: [ml4physicalsciences.github.io/2024](https://ml4physicalsciences.github.io/2024/)
- Format: identical to PAI26 (4 pages, NeurIPS style, double-blind)

### Key Observations from Past Papers
- **Dominant topics**: simulation-based inference, neural PDEs, equivariant networks, generative models for physics
- **Astronomy papers**: gravitational lensing, photodissociation regions, hybrid summary statistics
- **NO tool-augmented LLM agent papers** have appeared yet — **this is a gap we can fill**
- The workshop explicitly called out "the emerging role of foundation models" in 2024

### Closely Related Work (to cite and position against)

| Paper | Year | Relevance |
|-------|------|-----------|
| CosmoPaperQA (SciRag) | 2025 | RAG benchmark for astro, 105 QA pairs |
| ResearchBench | 2025 | End-to-end astro paper replication agent benchmark |
| Astro-QA | 2025 | 3082 question benchmark for LLMs in astronomy |
| AstroMLab / AstroBench | 2025 | LLM performance benchmark, factual retrieval |
| AstroLLaMA | 2023 | Domain-finetuned LLM for astronomy |
| AstroSage-LLaMA | 2025 | Domain-specialized astro assistant |
| Gemini as astro expert | 2025 | Nature Astronomy, autonomous classification |

> [!TIP]
> **Our differentiator**: None of these are *tool-augmented archive interaction agents*. They focus on QA, text generation, or classification. Quasar is unique in combining **live archive search + RAG + multi-step orchestration + script generation**.

---

## Benchmarking Strategy

### Category 1: Archive Search Accuracy (20 tasks)

**What to measure**: Can the agent correctly translate natural-language queries into structured ALMA archive searches and return relevant results?

| Task Type | Example Query | Gold Standard |
|-----------|--------------|---------------|
| Target search | "Find ALMA observations of HL Tau" | Expert-verified result set from ASA |
| Position search | "Search within 5 arcsec of RA=12h30m, Dec=-45d" | Manual TAP query result |
| Frequency search | "Band 6 observations covering 230 GHz" | Verified frequency match |
| Keyword search | "Protoplanetary disk observations at <0.1 arcsec" | Filtered archive results |
| Combined | "Band 7 observations of Orion KL with >1hr integration" | Multi-parameter query |

**Metrics**:
- **Precision@k** — fraction of returned results that are relevant
- **Recall** — fraction of relevant results that were returned
- **Parameter accuracy** — did the agent extract correct RA/Dec/frequency/band?

### Category 2: Multi-Step Workflows (15 tasks)

**What to measure**: Can the agent correctly chain tools to complete end-to-end research tasks?

| Task Type | Steps Involved | Evaluation |
|-----------|---------------|------------|
| Search → Inspect | Search archive, then read FITS headers | Correct headers extracted |
| Search → Script | Find data, generate CASA imaging script | Valid CASA script syntax |
| Search → Literature | Find observations, cross-reference ADS | Relevant papers returned |
| Search → Plot | Query results → sky map or frequency coverage | Correct plot output |
| Full pipeline | Search → DataLink → FITS → CASA → Plot | All steps complete & correct |

**Metrics**:
- **Task completion rate** — did all steps execute successfully?
- **Step accuracy** — fraction of steps with correct outputs
- **Tool selection precision** — did the agent pick the right tool at each step?
- **Total latency** — end-to-end time vs. expert manual time

### Category 3: Knowledge + Retrieval (15 tasks)

**What to measure**: Can the agent answer technical questions using RAG vs. hallucinating?

| Task Type | Example Question | Source |
|-----------|-----------------|--------|
| Technical spec | "Max recoverable scale for 12m array at 230 GHz?" | ALMA Technical Handbook |
| Proposal guidance | "What is the Phase 1 proposal deadline for Cycle 11?" | ALMA Proposer's Guide |
| Calibration | "How does bandpass calibration work in ALMA?" | ALMA documentation |
| Instrument | "What are the ALMA Band 6 receiver specifications?" | Technical Handbook |

**Metrics**:
- **Answer accuracy** — expert-graded correctness (1-5 scale)
- **Factual grounding** — does the answer cite correct source?
- **Hallucination rate** — fraction of claims not in source docs

### Ablation Studies

Run these to strengthen the benchmarking section:

1. **RAG vs. No-RAG** — same queries with and without document retrieval
2. **Conductor vs. Direct** — complex queries with and without task decomposition
3. **Tool-augmented vs. Text-only** — agent with vs. without tool access
4. **GPT-4 vs. GPT-4o-mini** — model size impact on accuracy

---

## Figures & Diagrams to Include

### Figure 1: System Architecture (Already in paper ✅)
TikZ diagram showing the 4-layer architecture. Currently in [main.tex](file:///c:/Users/adama/Desktop/Quasar-main/paper/main.tex).

### Figure 2: Workflow Trace Example (NEW — recommended)
Show a concrete end-to-end execution trace:
```
User: "Find Band 6 observations of HL Tau and generate a CASA imaging script"
  ↓
[Tool: search_by_target("HL Tau", band=6)]  →  12 results
  ↓
[Tool: get_observation_details(mous_id)]  →  metadata table
  ↓
[Tool: generate_casa_imaging_script(params)]  →  CASA script
  ↓
Agent: "I found 12 observations. Here's the imaging script for..."
```
**How to create**: TikZ sequence/flowchart diagram or `listings` code block.

### Figure 3: Benchmark Results Bar Chart (NEW)
Bar chart comparing accuracy across the 3 task categories:
- X-axis: task categories
- Y-axis: accuracy (%)
- Grouped bars: Full system vs. ablation variants (no-RAG, no-tools, etc.)

**How to create**: `pgfplots` bar chart in LaTeX, or generate via matplotlib and include as PDF.

### Figure 4: RAG Pipeline Detail (OPTIONAL)
Diagram showing: Documents → Chunking → Embedding → Qdrant → Query → Top-k retrieval → LLM context injection.

### Figure 5: Conductor DAG Example (OPTIONAL)
Show a task DAG for a complex query with parallel and sequential subtasks.

### Figure 6: UI Screenshot (OPTIONAL — for poster, not paper)
Screenshot of the Quasar web UI showing a live query with streaming results.

> [!NOTE]
> For a 4-page paper, **Figures 1-3 are essential**. Figures 4-6 are better suited for the poster or supplementary material.

---

## Compiling the LaTeX Paper

### Option 1: Overleaf (Recommended — no local install needed)
1. Go to [overleaf.com](https://overleaf.com)
2. Create a new project → Upload Project
3. Upload the entire `paper/` folder ([main.tex](file:///c:/Users/adama/Desktop/Quasar-main/paper/main.tex), [references.bib](file:///c:/Users/adama/Desktop/Quasar-main/paper/references.bib), [neurips_2025.sty](file:///c:/Users/adama/Desktop/Quasar-main/paper/neurips_2025.sty))
4. Compile and download the PDF

### Option 2: Install LaTeX Locally
```powershell
# Install MiKTeX (Windows)
winget install MiKTeX.MiKTeX

# Then compile:
cd paper
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

### Option 3: Docker
```bash
docker run --rm -v ${PWD}/paper:/workdir danteev/texlive \
  sh -c "cd /workdir && pdflatex main && bibtex main && pdflatex main && pdflatex main"
```
