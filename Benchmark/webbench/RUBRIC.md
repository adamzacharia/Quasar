# WebBench v0 grading contract

Graded only from the saved UI capture of the LAST turn of each question
(answer.png, steps.png, queries.png, answer.md, sources.json, timing.json).
`backend_web_log.txt` is supporting evidence (it can explain what the UI shows,
it never replaces it). Every score line in an evaluation.md quotes the evidence.

## Score per question (100 points)

### Questions with `expect_web: true`

| Code | Points | What earns it |
|---|---|---|
| T | 20 | Web search ran for this turn (web step in the Thought panel, or a sources card on this answer). |
| S | 15 | Source relevance: at least 2 of the first 4 shown sources are about the asked subject (8); at least one is the official or primary source named in the question rubric (7). |
| G | 25 | Grounding: the key facts in the answer body come from the shown sources and each is attributed. 25 = supported AND attributed per claim (inline citation that opens the right source); 15 = supported but only attributed as a block (appendix paragraph, or one source link for the whole answer); 5 = supported by a shown source but not attributed at all; 0 = not supported or contradicted by the sources. |
| C | 30 | Correctness against the question rubric's key facts (split as listed there). |
| P | 10 | Presentation: sources visible before the answer finished streaming (5; `timing.json` `time_to_sources_s` < `ui_stopwatch_s` minus 2 s); no truncation, raw markup, removed-link notices or an empty web section (5). |

### Questions with `expect_web: false`

| Code | Points | What earns it |
|---|---|---|
| T | 40 | No web search for this turn: no web step, no sources card, no "From the Web" text, and (supporting) no provider call in the backend log slice. |
| C | 50 | Correctness against the question rubric. |
| P | 10 | Clean answer: no web clutter, no claim of web verification, no truncation. |

### Questions with `expect_images: true` (images category, Phase 2)

Scored as `expect_web: true`, with T requiring image tiles on the answer and S / G judging
the tiles' relevance and attribution as the question rubric says. An `images` question with
`expect_web: false` (an imaging WORD that is not a picture request) is scored as
`expect_web: false`.

## Side metrics (reported, not scored)

- `triggered`: y/n versus expected (for trigger precision and recall).
- `queries`: the search strings shown in the Thought panel or backend log.
- `n_sources`, `cited_ids`, `time_to_sources_s`, `ui_stopwatch_s`.
- `citation_precision`: cited sentences whose cited source supports them, divided by all cited sentences. N/A when the answer has no inline web citations.
- `from_the_web_appendix`: y/n.

### Graded per question since Phase 2

- `official_first`: y/n/n/a. y when the FIRST cited source (the source card with the lowest
  citation number that the answer actually cites; the first card when nothing is cited) is on
  one of the question's `official_domains` (questions.yaml) AND, for a question that names a
  cycle or document version, that page is about the asked version (a Cycle 7 guide cited first
  for a Cycle 13 question is n). n/a for questions without official domains or without web.
- `unsupported_cited_claims`: the number of cited sentences whose numbers or dates the cited
  source does not show, as the grader reads the answer against sources.json (strict, the same
  rule as `citation_precision`). The backend's own count (`[VERIFY] web_citation unsupported=`,
  and `... after revise=` when the revise pass ran) is recorded in timing.json as supporting
  evidence; the graded number is the grader's.
- `web_decision_badge`: the badge text under the answer ("Searched the web (2 queries)" /
  "Web skipped: ..."), checked against `triggered`.
- `planner_s`: the planner call time from the backend log (for the p95 acceptance).

## Ground truth

Each rubric lists key facts with the official page they were checked against
and the date of the check. Time-sensitive facts (VLA configuration) are judged
as of the run date.
