# Quasar Benchmark Suite

A structured set of benchmark questions for evaluating the Quasar system's ability to answer ALMA Science Archive queries. Questions are organized by **topic** and **difficulty level**.

## Structure

```
Benchmark/
├── README.md                          # This file
├── run_benchmark.py                   # Automated benchmark runner
├── scoring_rubric.md                  # How to evaluate responses
├── 01_general_knowledge/              # ALMA fundamentals & archive navigation
│   ├── easy.md
│   ├── medium.md
│   └── hard.md
├── 02_data_specific/                  # Queries about specific projects & observations
│   ├── easy.md
│   ├── medium.md
│   └── hard.md
├── 03_analysis_and_methods/           # Programmatic access, queries, visualization
│   ├── easy.md
│   ├── medium.md
│   └── hard.md
├── 04_scientific_questions/           # Science-driven archive exploration
│   ├── easy.md
│   ├── medium.md
│   └── hard.md
└── results/                           # Auto-generated results (gitignored)
    └── <timestamp>_<model>/
        ├── results.json
        ├── benchmark_report.md
        ├── scores_by_question.png
        ├── radar_dimensions.png
        ├── scores_by_category.png
        ├── scores_by_difficulty.png
        ├── response_times.png
        └── grade_distribution.png
```

## Difficulty Levels

| Level    | Weight | Description |
|----------|--------|-------------|
| **Easy** | 1.0x   | Factual recall, straightforward lookups, single-step queries |
| **Medium** | 1.5x | Multi-step reasoning, combining filters, moderate domain knowledge |
| **Hard** | 2.0x   | Complex analysis, cross-archive queries, advanced visualization |

## Question Summary

| Category | Easy | Medium | Hard | Total |
|----------|------|--------|------|-------|
| General Knowledge | 2 | 2 | 1 | 5 |
| Data-Specific | 1 | 1 | 1 | 3 |
| Analysis & Methods | 1 | 4 | 2 | 7 |
| Scientific Questions | 1 | 1 | 1 | 3 |
| **Total** | **5** | **8** | **5** | **18** |

## Quick Start

### Run the full benchmark
```bash
python Benchmark/run_benchmark.py --api-url https://quasar-oi14.onrender.com --model gpt-4.1
```

### Run specific questions only
```bash
python Benchmark/run_benchmark.py --questions GK-E-01 AM-M-03 DS-H-01
```

### Dry run (preview questions without executing)
```bash
python Benchmark/run_benchmark.py --dry-run
```

### Custom judge model
```bash
python Benchmark/run_benchmark.py --judge-model gpt-4o-mini
```

## Requirements

```bash
pip install httpx openai matplotlib numpy
```

## What the Runner Does

1. **Sends each question** to the Quasar API as a chat message
2. **Collects the SSE response** (full streaming response)
3. **Scores via LLM judge** (GPT-4o by default) on 5 dimensions:
   - Correctness (0-5)
   - Completeness (0-5)
   - Code Quality (0-5 or N/A)
   - Presentation (0-5)
   - Tool Usage (0-5)
4. **Generates 6 charts**:
   - Scores by question (bar chart)
   - Radar chart of average dimension scores
   - Scores by category (horizontal bar)
   - Scores by difficulty (bar chart)
   - Response times (bar chart)
   - Grade distribution (pie chart)
5. **Produces a markdown report** with summary, tables, embedded charts, and full response details

## Output

Results are saved to `Benchmark/results/<timestamp>_<model>/` containing:
- `results.json` — Raw machine-readable results
- `benchmark_report.md` — Full human-readable report
- `*.png` — Visualization charts
