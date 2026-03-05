---
description: MANDATORY - Always use the Quasar conda environment for ALL Python operations in this project
---

# Quasar Conda Environment — MANDATORY

**This rule applies to EVERY interaction with this project. No exceptions.**

## Rules

1. **ALL Python commands** must be run inside the `quasar` conda environment.
2. **ALL pip installs** must target the `quasar` conda environment.
3. **ALL uvicorn/python/pytest commands** must use `conda run -n quasar`.

## How to Run Commands

// turbo-all

### Install packages:
```powershell
conda run -n quasar pip install <package-name>
```

### Start the backend:
```powershell
conda run -n quasar --no-capture-output python -m uvicorn ui-pro.api.main:app --port 8000
```

### Run tests:
```powershell
conda run -n quasar python -m pytest tests/
```

### Run any Python script:
```powershell
conda run -n quasar python <script.py>
```

## Important Notes
- The conda env is named **quasar** (lowercase q)
- Use `conda run -n quasar` (not `conda activate`) because activate doesn't work in all PowerShell sessions
- Add `--no-capture-output` for long-running/streaming commands (uvicorn, etc.)
- The frontend (npm/node) does NOT need conda — only Python commands do

