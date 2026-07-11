# Benchmark Scoring Rubric

This rubric defines how to evaluate Quasar system responses to benchmark questions.

---

## Scoring Dimensions

Each response is evaluated across **5 dimensions**, each scored 0–5:

### 1. Correctness (0–5)
Does the answer contain factually accurate information?

| Score | Description |
|-------|-------------|
| 0 | Completely incorrect or fabricated information |
| 1 | Major factual errors that would mislead the user |
| 2 | Some correct elements, but significant errors present |
| 3 | Mostly correct, with minor inaccuracies |
| 4 | Correct with only trivial issues |
| 5 | Fully correct, verified against ground truth |

### 2. Completeness (0–5)
Does the answer address all parts of the question?

| Score | Description |
|-------|-------------|
| 0 | Does not address the question at all |
| 1 | Addresses only a small fraction of the question |
| 2 | Addresses some parts but misses major components |
| 3 | Addresses most parts, missing minor details |
| 4 | Nearly complete, all major points covered |
| 5 | Comprehensive, covers all aspects including edge cases |

### 3. Code Quality (0–5)
*Only applicable for questions requiring code. Score N/A otherwise.*

| Score | Description |
|-------|-------------|
| 0 | No code provided when expected |
| 1 | Code present but non-functional (syntax errors, wrong API) |
| 2 | Code runs but produces incorrect results |
| 3 | Code works but has minor issues (deprecated API, poor style) |
| 4 | Code works correctly with good style |
| 5 | Code is correct, idiomatic, well-documented, and handles edge cases |

### 4. Presentation (0–5)
Is the answer well-structured and easy to understand?

| Score | Description |
|-------|-------------|
| 0 | Unreadable or incoherent |
| 1 | Poorly organized, hard to follow |
| 2 | Basic structure but lacks clarity |
| 3 | Reasonably well-organized |
| 4 | Well-structured with clear sections and formatting |
| 5 | Excellent presentation with tables, code blocks, and visual aids |

### 5. Tool Usage (0–5)
Did the system use appropriate tools and agents effectively?

| Score | Description |
|-------|-------------|
| 0 | No tools used when required |
| 1 | Tools used but incorrectly or ineffectively |
| 2 | Some appropriate tool usage with significant gaps |
| 3 | Generally appropriate tool usage |
| 4 | Good tool selection and execution |
| 5 | Optimal tool orchestration with multi-step reasoning where needed |

---

## Composite Score

**Total Score** = Sum of all applicable dimensions

| Difficulty | Max Possible (with code) | Max Possible (no code) |
|------------|--------------------------|------------------------|
| Easy       | 25                       | 20                     |
| Medium     | 25                       | 20                     |
| Hard       | 25                       | 20                     |

### Grade Thresholds

| Grade | Percentage | Description |
|-------|-----------|-------------|
| A     | ≥ 90%     | Excellent — production-ready quality |
| B     | 75–89%    | Good — minor improvements needed |
| C     | 60–74%    | Acceptable — functional but needs work |
| D     | 40–59%    | Below expectations — significant issues |
| F     | < 40%     | Failing — fundamental problems |

---

## Difficulty Weighting

When computing an **overall system score**, apply difficulty weights:

| Difficulty | Weight |
|------------|--------|
| Easy       | 1.0x   |
| Medium     | 1.5x   |
| Hard       | 2.0x   |

**Weighted Score** = Σ (question_score × difficulty_weight) / Σ (max_score × difficulty_weight)

---

## Recording Results

For each benchmark run, record:

| Field | Description |
|-------|-------------|
| **Question ID** | e.g., GK-E-01 |
| **Date** | When the test was run |
| **System Version** | Quasar version / commit hash |
| **Response Time** | Time from query submission to final answer (seconds) |
| **Correctness** | 0–5 |
| **Completeness** | 0–5 |
| **Code Quality** | 0–5 or N/A |
| **Presentation** | 0–5 |
| **Tool Usage** | 0–5 |
| **Total Score** | Sum |
| **Grade** | A/B/C/D/F |
| **Notes** | Any observations, errors, or noteworthy behaviors |

---

## Special Evaluation Notes

### For "Live Query" Questions
Questions requiring actual archive queries (DS-E-01, DS-M-01, etc.) should be evaluated based on:
- Did the system **attempt** the query?
- Was the query **syntactically valid**?
- Did it return **plausible results**?
- Were results **correctly interpreted**?

### For Visualization Questions
Questions requiring plots/images (AM-H-01, AM-H-02) should be evaluated based on:
- Was a visualization **produced**?
- Is it **scientifically accurate**?
- Is it **visually clear and well-labeled**?
- Does it meet the **specific requirements** (contours, colorscale, etc.)?

### For Multi-Step Questions
Hard questions often require multi-step reasoning. Evaluate:
- Was the problem **correctly decomposed**?
- Were intermediate steps **logically sound**?
- Was the **final synthesis** coherent?
- Did the system use **appropriate orchestration** (Conductor DAG)?
