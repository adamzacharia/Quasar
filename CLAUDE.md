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

Claude drives Codex headlessly via `scripts/codex_bridge.sh`, so the
consult → implement → review loop runs without the user pasting anything:

```bash
bash scripts/codex_bridge.sh consult < plan.md   # read-only second opinion
bash scripts/codex_bridge.sh build   < spec.md   # Codex edits files (workspace-write)
bash scripts/codex_bridge.sh build --full < spec.md  # + network/full access when needed
```

Codex's final message prints to stdout (and `tmp/codex/last_message.md`); the
full transcript goes to `tmp/codex/run.log` so it never floods Claude's context.
Claude then reviews the diff (`git diff`) and runs the relevant tests, and
re-specs a follow-up if needed. This loop runs automatically for code work in
this repo — do not stop to ask permission for each Codex round.

### Workflow for any change

1. **Claude** drafts the plan/spec (exact files, behavior, acceptance criteria, tests).
2. **Consult Codex (GPT‑5.5 @ xhigh)** on that plan; incorporate or explicitly rebut its feedback.
3. **Implement** — either Codex (per a self-contained spec) or Claude — and the *other* model reviews the diff.
4. **Verify:** run the relevant tests and adversarially check the risky logic before declaring done.
5. Hand specs to Codex as self-contained prompts (paste-ready), so Codex spends its own tokens on the heavy editing while Claude plans and reviews.

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
