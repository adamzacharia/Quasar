# CLAUDE.md — working agreement for the Quasar repo

## Collaboration protocol: consult Codex (GPT‑5.5) on ALL changes

For **every** feature implementation, code change, edit, refactor, bug fix, or
design decision anywhere inside this repository, **consult Codex running GPT‑5.5
at `xhigh` reasoning effort** to get an independent second view *before and/or
during* the work. This is mandatory, not optional — deliberately mix-and-match
models so each plan is pressure-tested by the other before code lands.

Use Codex to: challenge the plan, surface alternatives and edge cases, sanity-check
scientific/astronomy logic, and review diffs. Treat its feedback as input to
weigh, not gospel — reconcile disagreements explicitly and explain the final call.

### How to invoke Codex

- **Discuss / plan (read‑only second opinion):**
  ```bash
  codex exec -m gpt-5.5 -c model_reasoning_effort="xhigh" -s read-only "<plan or question + relevant file paths>"
  ```
- **Have Codex implement (it edits files):**
  ```bash
  codex -m gpt-5.5 -c model_reasoning_effort="xhigh"          # interactive
  codex exec -m gpt-5.5 -c model_reasoning_effort="xhigh" -s workspace-write "<self-contained spec>"
  ```

### Automated bridge (no manual copy-paste)

Claude drives Codex headlessly via `scripts/codex_bridge.sh` (low-level) and
`scripts/codex_loop.sh` (the full hands-off orchestrator). Both run without the
user pasting anything:

```bash
# Low-level single calls:
bash scripts/codex_bridge.sh consult < plan.md       # read-only second opinion (fresh thread)
bash scripts/codex_bridge.sh build   < spec.md       # Codex edits files (workspace-write, fresh thread)
bash scripts/codex_bridge.sh consult --resume < reply.md  # CONTINUE the same thread (Codex remembers)
bash scripts/codex_bridge.sh build   --full   < spec.md   # + network/full access when needed
```

`--resume` now works: the bridge captures each fresh run's session id to
`tmp/codex/session_id` and resumes THAT id (sandbox is passed via
`-c sandbox_mode=`, since `codex exec resume` rejects `-s`). This makes cheap
multi-round consult↔build↔fix discussion possible — use it instead of one-shots.

Codex's final message prints to stdout (and `tmp/codex/last_message.md`); the
full transcript goes to `tmp/codex/run.log`. **Never let Codex read or print
generated files** (`.next/`, `dist/`, `build/`, `node_modules/`, compiled CSS) —
it wastes 100k+ tokens; reason on source only, never `cat` build output.

### Mode keywords (Claude picks the loop from a trailing keyword)

The user selects how heavily to lean on Codex by ending their prompt with one of
these. Claude maps the keyword to `scripts/codex_loop.sh --mode …` and runs it
hands-off. No keyword → Claude picks (`build` for non-trivial work, inline for
trivial) and states which it chose.

| Keyword | Mode | Who implements / who reviews |
|---|---|---|
| `@cx-build` | **build** | Codex implements (consult→build→self-review→fix); **Claude reviews** the diff. Maximizes Codex's share of the work. |
| `@cx-duel`  | **duel**  | **Both** implement — Codex in a `git worktree`, Claude in `beta`; diff both vs base and pick/merge. |
| `@cx-guard` | **guard** | **Claude implements**; Codex reviews the diff over multiple resume rounds; Claude reconciles findings. |

```bash
bash scripts/codex_loop.sh --mode build  --test 'cmd' < spec.md   # default
bash scripts/codex_loop.sh --mode guard  --base beta             # after Claude edits
bash scripts/codex_loop.sh --mode duel   < spec.md               # + --teardown to clean up worktree
```

Artifacts per run land in `tmp/codex/tasks/<id>/` (consult.md, build.md,
review.md, findings.md, status.md). Codex findings are persisted so they can't be
silently skipped — fix or explicitly waive each before declaring done.

### Workflow for any change

1. **Claude** drafts the plan/spec (exact files, behavior, acceptance criteria, tests).
2. **Consult Codex (GPT‑5.5 @ xhigh)** on that plan; incorporate or explicitly rebut its feedback.
3. **Implement** per the mode keyword — default to handing the heavy editing to Codex (`build`) so Codex spends its own tokens while Claude plans and reviews; the *other* model always reviews the diff.
4. **Verify:** run the relevant tests and adversarially check the risky logic before declaring done.
5. Hand specs to Codex as self-contained prompts (paste-ready). Prefer threaded `--resume` rounds over fresh one-shots so the two models actually discuss.

## Git

- Default working branch: **`beta`**.
- **Always push to BOTH remotes** after committing: `origin` → GitHub
  (`github.com/adamzacharia/Quasar`) and `gitlab` → GitLab
  (`gitlab.com/adamzacharia/Quasar`). Run `git push origin beta && git push gitlab beta`.
- Only commit/push when the user has asked.

## Running the app (Windows / PowerShell)

- One command: `./scripts/local/start_quasar.ps1 -Restart` (backend on :8000 via `ui-pro/launch.py`, frontend on :3001).
- Backend only: `cd ui-pro; uvicorn api.main:app --reload --port 8000`. Frontend only: `cd ui-pro; npm run dev`.
- Spectral Line Explorer backend is gated by `ENABLE_SPECTRAL_LINE_EXPLORER` (now defaults enabled). Live science tests are gated by `RUN_LIVE_SPECTRAL_LINE_TESTS=1`.
