# Model response fixtures

Checked 2026-09-20. These contain no account data or credentials.

- `openai.json`: public example response from https://developers.openai.com/api/reference/resources/models/methods/list (omits optional shutdown_date).
- `anthropic.json`: reduced public example from https://platform.claude.com/docs/en/api/models/list. Keeps image/thinking capabilities; pagination identifiers have been made consistent and has_more set false for the single-page normalization fixture. Zero token limits are deliberately unknown, not guessed.
- `deepseek.json`: representative documented OpenAI-compatible shape, based on the user-specified endpoint and existing DeepSeek IDs. The documentation endpoint could not be fetched during this run; this is NOT a captured live response.
- `google.json`: representative fixture based on the fields documented at https://ai.google.dev/api/models; NOT a captured account response. Includes an embedding model to test generation-method filtering.

Additional synthetic model IDs in tests cover exclusion rules, alias snapshots, new families, malformed pagination, auth errors, and cache races. Live authenticated response captures require test credentials; no real user keys are used by the tests.
