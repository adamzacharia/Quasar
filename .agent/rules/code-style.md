# Code Style Rules

1. **No emojis in code**: Never use emoji characters in source code (comments, docstrings, string literals, logging) unless the user explicitly asks for them. Use plain ASCII text descriptions instead.

2. **Output emoji convention**: When generating user-facing output (UI text, status messages, chat responses), use the Quasar project's own emoji set rather than generic provider defaults. Refer to the frontend's emoji/icon components for the canonical set.

3. **No dry-run tools or half-baked features**: All features, tool actions, and data operations (e.g., data downloading, calculation engines, and file generation) must always default to executing for real using actual, live datasets and real files. Do not use dry-run, simulation, or stub execution modes as defaults. When a tool has a dry-run flag, it must default to `False` (always execute live/real data by default).
