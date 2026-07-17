"""Per-model token pricing and cost estimation.

Every number here is an **estimate**. Prices are hand-maintained from public
provider pricing pages and drift without notice; treat `cost_usd` anywhere in
Quasar as "what this turn would have cost at the rates on
``PRICING_LAST_VERIFIED``", not as a bill. Tokens are the source of truth —
cost is derived from them.

Rates are USD per **million** tokens.

Provenance of the tables below (re-verify before quoting cost in a paper):
  - anthropic: from the vendored claude-api reference, 2026-07-16.
  - openai / deepseek / google: best-effort from public pricing pages; these
    are the least trustworthy rows.

`tacc` and `local` are deliberately absent. TACC is grant-funded and `local/`
is self-hosted, so neither has a per-token price we can honestly quote —
`estimate_cost` returns None for them rather than a wrong $0.00. None means
"unknown", and callers must render it as such (an unpriced turn is not a free
turn).
"""

from __future__ import annotations

import os
import re
import threading
from typing import Dict, Optional

# A prefix match is only accepted when the text after the base model name looks
# like a date/version/variant suffix — a separator (-, @, :, _, .) immediately
# followed by a digit (e.g. "-20251001", "@20251101", "-2024-08-06"). This is
# what lets "claude-haiku-4-5-20251001" resolve to "claude-haiku-4-5" while
# "gpt-4o-imaginary" (separator + letter) stays UNKNOWN instead of borrowing
# gpt-4o's price. A word-suffix model is a different model, not a variant.
_VERSION_SUFFIX = re.compile(r"^[-@:_.]\d")

# Date the tables below were last checked against provider pricing pages.
PRICING_LAST_VERIFIED = "2026-07-16"

# {provider: {model: {input_per_mtok, output_per_mtok}}} in USD per 1M tokens.
MODEL_PRICING: Dict[str, Dict[str, Dict[str, float]]] = {
    "anthropic": {
        "claude-fable-5": {"input_per_mtok": 10.00, "output_per_mtok": 50.00},
        "claude-mythos-5": {"input_per_mtok": 10.00, "output_per_mtok": 50.00},
        "claude-opus-4-8": {"input_per_mtok": 5.00, "output_per_mtok": 25.00},
        "claude-opus-4-7": {"input_per_mtok": 5.00, "output_per_mtok": 25.00},
        "claude-opus-4-6": {"input_per_mtok": 5.00, "output_per_mtok": 25.00},
        "claude-opus-4-5": {"input_per_mtok": 5.00, "output_per_mtok": 25.00},
        # Sonnet 5 list price; an introductory $2/$10 runs through 2026-08-31,
        # so this over-estimates until then.
        "claude-sonnet-5": {"input_per_mtok": 3.00, "output_per_mtok": 15.00},
        "claude-sonnet-4-6": {"input_per_mtok": 3.00, "output_per_mtok": 15.00},
        "claude-sonnet-4-5": {"input_per_mtok": 3.00, "output_per_mtok": 15.00},
        "claude-haiku-4-5": {"input_per_mtok": 1.00, "output_per_mtok": 5.00},
    },
    "openai": {
        "gpt-4o": {"input_per_mtok": 2.50, "output_per_mtok": 10.00},
        "gpt-4o-mini": {"input_per_mtok": 0.15, "output_per_mtok": 0.60},
        "gpt-4.1": {"input_per_mtok": 2.00, "output_per_mtok": 8.00},
        "gpt-4.1-mini": {"input_per_mtok": 0.40, "output_per_mtok": 1.60},
        "gpt-4.1-nano": {"input_per_mtok": 0.10, "output_per_mtok": 0.40},
        # Quasar deployment alias. Rate INFERRED from OpenAI's mini tier;
        # unverified against the deployment's actual contract — correct before
        # quoting in a publication.
        "gpt-5.4-mini": {"input_per_mtok": 0.40, "output_per_mtok": 1.60},
        "o3": {"input_per_mtok": 2.00, "output_per_mtok": 8.00},
        "o3-mini": {"input_per_mtok": 1.10, "output_per_mtok": 4.40},
        "o4-mini": {"input_per_mtok": 1.10, "output_per_mtok": 4.40},
        # Embeddings: output tokens are always 0, so only the input rate bites.
        # ada-002 is langchain_openai's default and is what personal-RAG
        # actually calls today (sse.py) — do not drop it.
        "text-embedding-ada-002": {"input_per_mtok": 0.10, "output_per_mtok": 0.0},
        "text-embedding-3-small": {"input_per_mtok": 0.02, "output_per_mtok": 0.0},
        "text-embedding-3-large": {"input_per_mtok": 0.13, "output_per_mtok": 0.0},
    },
    "deepseek": {
        "deepseek-chat": {"input_per_mtok": 0.27, "output_per_mtok": 1.10},
        "deepseek-reasoner": {"input_per_mtok": 0.55, "output_per_mtok": 2.19},
        # Quasar deployment aliases (the IDs /api/models actually serves).
        # Rates INFERRED from DeepSeek's public chat/reasoner tiers — the exact
        # rate for this deployment's contract is unverified. Correct these to
        # the real numbers before quoting cost in a publication.
        "deepseek-v4-flash": {"input_per_mtok": 0.27, "output_per_mtok": 1.10},
        "deepseek-v4-pro": {"input_per_mtok": 0.55, "output_per_mtok": 2.19},
    },
    "google": {
        "gemini-2.5-pro": {"input_per_mtok": 1.25, "output_per_mtok": 10.00},
        "gemini-2.5-flash": {"input_per_mtok": 0.30, "output_per_mtok": 2.50},
        "gemini-2.0-flash": {"input_per_mtok": 0.10, "output_per_mtok": 0.40},
    },
}


def cost_accounting_enabled() -> bool:
    """Whether to compute $ cost. Tokens are recorded either way.

    Defaults ON: the cost column is additive and non-destructive. Set
    QUASAR_ENABLE_COST_ACCOUNTING=0 to persist tokens only.
    """
    raw = os.getenv("QUASAR_ENABLE_COST_ACCOUNTING")
    if raw is None or not raw.strip():
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def normalize_provider(provider: str) -> str:
    """Mirror of UsageQuotaService.normalize_provider.

    Duplicated rather than imported to keep this module dependency-free (the
    quota service imports pricing, not the other way round). Keep in sync.
    """
    value = (provider or "").strip().lower()
    if value == "gemini":
        return "google"
    if value in {"tejas", "texas", "texas_ai"}:
        return "tacc"
    return value


def get_model_pricing(provider: str, model: str) -> Optional[Dict[str, float]]:
    """Return the rate card for a model, or None when we have no price.

    Matching is exact first, then longest-prefix — provider model IDs commonly
    carry a date or variant suffix (``claude-haiku-4-5-20251001``,
    ``gpt-4o-2024-08-06``) that shares a prefix with the base ID.
    """
    table = MODEL_PRICING.get(normalize_provider(provider))
    if not table:
        return None

    key = (model or "").strip()
    if not key:
        return None
    if key in table:
        return dict(table[key])

    lowered = key.lower()
    matches = [
        name
        for name in table
        if lowered.startswith(name.lower())
        and _VERSION_SUFFIX.match(lowered[len(name):])
    ]
    if not matches:
        return None
    # Longest prefix wins: "gpt-4o-mini-2024" must match "gpt-4o-mini", not
    # "gpt-4o".
    return dict(table[max(matches, key=len)])


def estimate_cost(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> Optional[float]:
    """Estimated USD for one turn, or None when the model has no known price.

    None is a real answer — it means "we cannot price this" (TACC, local, or a
    model missing from the table). Never coerce it to 0.0: a free turn and an
    unpriced turn are different facts, and conflating them silently under-reports
    spend.
    """
    if not cost_accounting_enabled():
        return None

    rates = get_model_pricing(provider, model)
    if rates is None:
        return None

    tokens_in = max(0, int(input_tokens or 0))
    tokens_out = max(0, int(output_tokens or 0))
    # Deliberately UNROUNDED. A single one-token call can cost ~1e-7; rounding
    # here (to any display precision) drops it to 0 and, when many such
    # auxiliary calls are summed, the whole turn rounds to $0.00. Callers round
    # once, at the persist/display boundary (turn_cost_usd, cost_breakdown,
    # the rollup's stored per-run values), after accumulation.
    return (
        tokens_in * rates["input_per_mtok"] + tokens_out * rates["output_per_mtok"]
    ) / 1_000_000


class TurnCostAccumulator:
    """One chat turn's tokens and cost, attributed to the route that paid.

    A turn is neither one call nor one route. A single turn can run an image
    pre-pass on gpt-4o-mini, personal-RAG embeddings on OpenAI, and the main
    loop on the selected model — each priced differently, and each potentially
    billed to a DIFFERENT key. Pricing the turn's token total at the main
    model's rate, or filing the whole cost under the turn's selected
    key_source, both produce a number that is simply wrong (CX-28).

    So every call is recorded individually via ``record`` and folded into both
    a turn-wide total and a per-key_source bucket.

    Thread-safe: one turn's calls can be reported from several worker threads
    (the agent worker plus auxiliary calls), and ``record`` is a read-modify-
    write across several fields that must move together. Today's GIL happens to
    make the individual ``+=``s hard to interleave badly, but that is a CPython
    implementation detail, not a guarantee — it does not hold on free-threaded
    builds, and it says nothing about the multi-field update staying consistent.
    The lock makes correctness independent of it; the cost is a few ns per call.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.input_tokens = 0
        self.output_tokens = 0
        self.usd = 0.0
        self.priced_tokens = 0
        self.unpriced_tokens = 0
        # key_source -> {"usd", "priced_tokens", "unpriced_tokens"}
        self.by_source: Dict[str, Dict[str, float]] = {}

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def record(
        self,
        provider: str,
        model: str,
        key_source: str,
        input_tokens: int,
        output_tokens: int,
    ) -> Optional[float]:
        """Fold one LLM call in. Returns the call's cost, or None if unpriced."""
        tokens_in = int(input_tokens or 0)
        tokens_out = int(output_tokens or 0)
        # Priced outside the lock: it is pure, and holding a lock across it would
        # serialise every worker thread on a table lookup.
        try:
            call_cost = estimate_cost(provider, model, tokens_in, tokens_out)
        except Exception:  # pricing must never break a chat turn
            call_cost = None

        source = (key_source or "platform").strip().lower() or "platform"
        with self._lock:
            self.input_tokens += tokens_in
            self.output_tokens += tokens_out
            bucket = self.by_source.setdefault(
                source, {"usd": 0.0, "priced_tokens": 0, "unpriced_tokens": 0}
            )
            if call_cost is None:
                self.unpriced_tokens += tokens_in + tokens_out
                bucket["unpriced_tokens"] += tokens_in + tokens_out
            else:
                self.usd += call_cost
                self.priced_tokens += tokens_in + tokens_out
                bucket["usd"] += call_cost
                bucket["priced_tokens"] += tokens_in + tokens_out
        return call_cost

    def turn_cost_usd(self) -> Optional[float]:
        """Estimated $ to produce this turn, whoever paid — the
        cost-efficiency number.

        None (not 0.0) when nothing could be priced: an all-TACC/local turn has
        no per-token price, and $0.00 would claim we know it was free.

        UNROUNDED — this is persisted and summed across turns, so rounding here
        would zero a sub-cent turn and 1,000 tiny turns would sum to $0. Display
        surfaces round once, at presentation.
        """
        if self.priced_tokens <= 0:
            return None
        return self.usd

    def platform_cost_usd(self) -> Optional[float]:
        """The subset Quasar actually paid — what admin/billing totals must sum.

        Distinct from turn_cost_usd: a turn run entirely on the user's own BYOK
        key costs Quasar nothing however expensive it was.

        Three cases, deliberately distinguished:
          - no platform call at all -> 0.0. We KNOW Quasar paid nothing.
          - platform calls ran but none could be priced -> None (unknown).
            0.0 would claim they were free.
          - otherwise -> the priced platform subset, unrounded.
        """
        bucket = self.by_source.get("platform")
        if bucket is None:
            return 0.0
        if bucket["priced_tokens"] <= 0:
            return None
        return bucket["usd"]
