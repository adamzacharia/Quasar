---
description: MANDATORY - Update docs/FEATURES.md every time a feature is added, modified, or completed
---

# Feature Tracking -- MANDATORY

**This workflow MUST be followed whenever you implement, modify, or complete any feature in the Quasar project.**

## When This Applies

- You add a new tool to `core/agent.py`
- You create or modify a service in `services/` or `integrations/`
- You complete a feature listed in `docs/FEATURES.md`
- You add new functionality that qualifies as a "feature" (user-facing capability)

## Steps

1. **Open `docs/FEATURES.md`** and locate the relevant feature entry.

2. **If the feature already exists in the list:**
   - Change `[ ]` to `[x]` to mark it as completed.
   - Add the file path in parentheses at the end of the description, e.g., `(services/astro_calculators.py)`.
   - Update the Status Summary table at the top (increment the Built count, decrement Planned, recalculate % Done).

3. **If the feature is NEW and not yet listed:**
   - Add it to the correct Tier section based on its complexity and impact.
   - Use the next available ID in its category (e.g., if the last Universal feature is U10, the new one is U11).
   - Follow the existing format: `- [x] **ID** Feature Name -- Description (file_path.py)`
   - Update the Status Summary table.

4. **Update the `Last updated` date** at the bottom of the file to today's date.

5. **Also update these related files if the change is significant:**
   - `docs/ARCHITECTURE.md` -- if new modules or services were added
   - `docs/ISSUES.md` -- if the feature fixes a known issue, mark the issue as resolved
   - `README.md` -- if the feature changes the setup, API, or user-facing capabilities

## Format Reference

```markdown
- [x] **U9** Redshift Calculator -- Convert between z, Mpc, lookback time (services/astro_calculators.py)
- [ ] **U11** New Feature Name -- Brief description of what it does
```

## Status Summary Table Format

Keep the table at the top of FEATURES.md accurate:

```markdown
| Category               | Total | Built | In Progress | Planned | % Done |
|------------------------|-------|-------|-------------|---------|--------|
| Universal              | 10    | 5     | 0           | 5       | 50%    |
```

## Important

- Never remove a feature entry, only change its status.
- If a feature is partially built, use `[/]` and add "(partial)" to the status.
- Always include the implementing file path so future audits can verify.
