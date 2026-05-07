---
description: Weekly system audit - Verify all documentation matches the actual codebase and update stale references
---

# Weekly System Audit

**Run this audit at least once per week, or whenever the user invokes `/audit`.**

This workflow ensures all documentation, issue tracking, and architecture references
stay synchronized with the actual codebase. Stale docs are one of the biggest
risks to project credibility, especially for the paper.

## Files to Audit

Audit each of these files against the live codebase. For every file, read the
current content, verify claims against actual code, and fix any inaccuracies.

### 1. docs/FEATURES.md -- Feature Roadmap

- [ ] Verify every `[x]` (Built) feature actually exists in the codebase. Search for the listed file path.
- [ ] Check if any `[ ]` (Planned) features have been implemented but not marked. Grep for relevant class/function names.
- [ ] Verify the Status Summary table math is correct (totals, percentages).
- [ ] Update the "Last updated" date.

### 2. docs/ARCHITECTURE.md -- System Architecture

- [ ] Verify every file and module listed actually exists at the stated path.
- [ ] Check that listed line counts are roughly accurate (within 20%).
- [ ] Verify the component diagram matches the current import graph.
- [ ] Check that listed tool counts match `core/agent.py` `_register_tools()`.
- [ ] Remove references to deleted files (e.g., old service stubs).

### 3. docs/ISSUES.md -- Issue Tracker

- [ ] Check every open issue: is it still reproducible? Search the codebase for the buggy pattern.
- [ ] Mark resolved issues as fixed with today's date.
- [ ] Add any new issues discovered during the audit.

### 4. docs/QUASAR_SYSTEM_DOCUMENTATION.md -- System Reference

- [ ] Verify API endpoints match `ui-pro/api/main.py` routes.
- [ ] Verify listed environment variables match `.env.example` and actual usage.
- [ ] Check that database references are current (Qdrant, Turso, etc.).
- [ ] Verify model names match `config/settings.py` and `core/llm_client.py`.

### 5. README.md -- Project README

- [ ] Verify setup instructions still work (dependencies, env vars).
- [ ] Check that listed features match `docs/FEATURES.md` built count.
- [ ] Verify deployment URLs are current and accessible.
- [ ] Check that listed environment variables are complete and accurate.

### 6. docs/fixes.md -- Applied Fixes Log

- [ ] Verify every fix listed has actually been applied in the codebase.
- [ ] Remove entries for fixes that are no longer relevant.
- [ ] Add entries for any fixes applied since last audit.

### 7. .env.example -- Environment Template

- [ ] Cross-reference against every `os.getenv()` / `os.environ.get()` call in the codebase.
- [ ] Ensure all required variables are listed with clear descriptions.
- [ ] Verify no real secrets or API keys are committed.

### 8. requirements.txt -- Python Dependencies

- [ ] Verify every imported package is listed.
- [ ] Check for unused dependencies that can be removed.
- [ ] Verify version pins are not blocking security updates.

## How to Run the Audit

For each file above:

1. **Read the file** in full.
2. **Grep the codebase** to verify each claim (file paths, function names, class names, tool counts).
3. **Fix inaccuracies** directly -- do not just flag them, correct them.
4. **Log changes** in `docs/ISSUES.md` under a new section: `### [DATE] Weekly Audit`.

## Audit Report

After completing all checks, add an entry at the bottom of `docs/ISSUES.md`:

```markdown
### [YYYY-MM-DD] Weekly System Audit

**Files audited:** FEATURES.md, ARCHITECTURE.md, ISSUES.md, QUASAR_SYSTEM_DOCUMENTATION.md, README.md, fixes.md, .env.example, requirements.txt

**Corrections made:**
- [list each correction with file name and what was changed]

**New issues found:**
- [list any new issues discovered]

**Status:** All documentation verified against codebase as of [date].
```

## Important Notes

- This audit should take 15-30 minutes if docs are mostly current.
- If docs are severely stale (>50% inaccurate), flag it to the user and propose a full rewrite.
- Never silently skip a file. If a file cannot be verified, note it in the report.
- The goal is documentation that a reviewer or new contributor can trust completely.
