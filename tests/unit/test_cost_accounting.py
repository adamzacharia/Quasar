"""Per-turn token + cost accounting and the daily platform token cap.

The cap tests carry the weight here: a bypass is free money, so each exempt
route (BYOK, quota-exempt email) and each ordering hazard gets its own test.
"""

import threading

import pytest

from services import model_pricing
from services import usage_quota_service as uqs
from services.issue_report_service import IssueReportService
from services.model_pricing import TurnCostAccumulator, estimate_cost
from services.usage_quota_service import (
    QuotaExceededError,
    UsageQuotaService,
    UsageRecord,
)


@pytest.fixture
def quota_store(monkeypatch, tmp_path):
    monkeypatch.setattr(uqs, "_LOCAL_DB", str(tmp_path / "usage_quota.db"))
    return UsageQuotaService()


@pytest.fixture
def runs(tmp_path):
    return IssueReportService(str(tmp_path / "reports.db"))


def _spend(store: UsageQuotaService, tokens: int, *, provider="openai",
           key_source="platform", user_id="u1", model="gpt-4o-mini"):
    store.record_usage(
        UsageRecord(
            user_id=user_id,
            provider=provider,
            model=model,
            key_source=key_source,
            input_tokens=tokens,
            output_tokens=0,
        )
    )


# ── estimate_cost ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "provider, model, tokens_in, tokens_out, expected",
    [
        # 1M in + 1M out at $5/$25.
        ("anthropic", "claude-opus-4-8", 1_000_000, 1_000_000, 30.0),
        ("anthropic", "claude-haiku-4-5", 1_000_000, 0, 1.0),
        ("openai", "gpt-4o", 1_000_000, 0, 2.5),
        ("deepseek", "deepseek-chat", 1_000_000, 0, 0.27),
        ("google", "gemini-2.5-pro", 1_000_000, 0, 1.25),
    ],
)
def test_estimate_cost_per_provider(provider, model, tokens_in, tokens_out, expected):
    assert estimate_cost(provider, model, tokens_in, tokens_out) == pytest.approx(expected)


def test_unpriced_providers_return_none_not_zero():
    """TACC is grant-funded and local/ is self-hosted: we have no per-token
    price for either. None means "unknown" — returning 0.0 would assert the
    turn was free, which is a different (and unsupported) claim."""
    assert estimate_cost("tacc", "gpt-oss-120b", 1_000_000, 1_000_000) is None
    assert estimate_cost("local", "local/llama", 1_000_000, 1_000_000) is None
    # Unknown model on a priced provider is also unknown, not free.
    assert estimate_cost("openai", "gpt-9-imaginary", 1_000_000, 0) is None


def test_model_matching_prefers_longest_prefix():
    """Dated/variant suffixes must resolve to the right base model —
    "gpt-4o-mini-2024-07-18" must not match the pricier "gpt-4o"."""
    assert estimate_cost("openai", "gpt-4o-mini-2024-07-18", 1_000_000, 0) == pytest.approx(0.15)
    assert estimate_cost("anthropic", "claude-haiku-4-5-20251001", 1_000_000, 0) == pytest.approx(1.0)


def test_prefix_match_only_accepts_version_suffixes():
    """A word-suffix is a DIFFERENT model, not a variant of a priced one.

    Only a date/version continuation (separator + digit) may borrow a base
    model's price. "gpt-4o-imaginary" must be unknown (None), not $2.50.
    """
    # Dated / version suffixes: priced.
    assert estimate_cost("openai", "gpt-4o-2024-08-06", 1_000_000, 0) == pytest.approx(2.5)
    assert estimate_cost("anthropic", "claude-opus-4-5@20251101", 1_000_000, 0) == pytest.approx(5.0)
    # Word suffixes: unknown, never the base model's price.
    assert estimate_cost("openai", "gpt-4o-imaginary", 1_000_000, 0) is None
    assert estimate_cost("openai", "gpt-4o-preview", 1_000_000, 0) is None
    assert estimate_cost("deepseek", "deepseek-chatty", 1_000_000, 0) is None


def test_small_calls_do_not_round_to_zero():
    """estimate_cost must not lose a single tiny call to rounding — otherwise
    many auxiliary calls each round to 0 and the turn totals $0.00 (CX-05)."""
    one_token = estimate_cost("openai", "gpt-4o-mini", 1, 0)
    assert one_token is not None and one_token > 0
    # A thousand one-token calls, accumulated the way sse.py does, is measurable.
    accumulated = sum(estimate_cost("openai", "gpt-4o-mini", 1, 0) for _ in range(1000))
    assert round(accumulated, 6) > 0


@pytest.mark.parametrize(
    "provider, model, expected",
    [
        # deepseek-v4-pro is the repo's DEFAULT_LLM_MODEL — the model most real
        # turns actually run on, so an unpriced entry here unprices the common
        # path, not an edge case.
        ("deepseek", "deepseek-v4-pro", 0.55),
        ("deepseek", "deepseek-v4-flash", 0.27),
        ("openai", "gpt-5.4-mini", 0.40),
    ],
)
def test_configured_deployment_aliases_are_priced(provider, model, expected):
    """The IDs /api/models serves are deployment aliases, not the vendors' own
    model names, so they do not fall out of the public pricing tables for free —
    each needs an explicit row or every real turn reports as unpriced (CX-27).
    """
    assert estimate_cost(provider, model, 1_000_000, 0) == pytest.approx(expected)


def test_gemini_alias_normalizes_to_google():
    assert estimate_cost("gemini", "gemini-2.5-pro", 1_000_000, 0) == pytest.approx(1.25)


def test_cost_accounting_flag_disables_pricing(monkeypatch):
    monkeypatch.setenv("QUASAR_ENABLE_COST_ACCOUNTING", "0")
    assert estimate_cost("openai", "gpt-4o", 1_000_000, 0) is None
    monkeypatch.setenv("QUASAR_ENABLE_COST_ACCOUNTING", "1")
    assert estimate_cost("openai", "gpt-4o", 1_000_000, 0) == pytest.approx(2.5)


# ── TurnCostAccumulator (per-call attribution) ──────────────────────────────


def test_accumulator_prices_each_call_at_its_own_model_rate():
    """A turn spans models. Pricing its token TOTAL at the main model's rate
    would bill the cheap embedding at the expensive model's price."""
    usage = TurnCostAccumulator()
    usage.record("openai", "gpt-4o", "platform", 1_000_000, 0)            # $2.50
    usage.record("openai", "text-embedding-ada-002", "platform", 1_000_000, 0)  # $0.10

    assert usage.total_tokens == 2_000_000
    assert usage.turn_cost_usd() == pytest.approx(2.60)
    # Not 2 * $2.50 (main-model rate on the total) and not 2 * $0.10.


def test_accumulator_excludes_byok_spend_from_platform_cost():
    """The crux of CX-28: a BYOK main loop plus a platform embedding. The user
    paid $2.50; Quasar paid only the $0.10 embedding."""
    usage = TurnCostAccumulator()
    usage.record("anthropic", "claude-opus-4-8", "byok", 1_000_000, 0)          # user pays $5
    usage.record("openai", "text-embedding-ada-002", "platform", 1_000_000, 0)  # Quasar pays $0.10

    assert usage.turn_cost_usd() == pytest.approx(5.10)   # cost to produce
    assert usage.platform_cost_usd() == pytest.approx(0.10)  # Quasar's bill only


def test_accumulator_platform_cost_is_zero_when_turn_is_all_byok():
    """No platform call ran, so Quasar paid nothing — and we KNOW that. This is
    a real 0.0, not the "unknown" None."""
    usage = TurnCostAccumulator()
    usage.record("anthropic", "claude-opus-4-8", "byok", 1_000_000, 1_000_000)

    assert usage.turn_cost_usd() == pytest.approx(30.0)
    assert usage.platform_cost_usd() == 0.0


def test_accumulator_platform_cost_unknown_when_platform_calls_are_unpriced():
    """Platform calls ran but none could be priced (all TACC). That is unknown,
    not free — 0.0 here would under-report real platform spend."""
    usage = TurnCostAccumulator()
    usage.record("tacc", "gpt-oss-120b", "platform", 1_000_000, 0)

    assert usage.turn_cost_usd() is None
    assert usage.platform_cost_usd() is None
    assert usage.unpriced_tokens == 1_000_000


def test_accumulator_key_source_defaults_to_platform():
    """An unset/blank key_source must fall back to platform, never vanish from
    attribution — the fail-safe direction is counting spend as Quasar's."""
    usage = TurnCostAccumulator()
    usage.record("openai", "gpt-4o", "", 1_000_000, 0)
    usage.record("openai", "gpt-4o", None, 1_000_000, 0)
    usage.record("openai", "gpt-4o", "PLATFORM", 1_000_000, 0)  # case-insensitive

    assert usage.platform_cost_usd() == pytest.approx(7.5)


def test_accumulator_totals_are_correct_under_concurrent_recording():
    """One turn's calls can be reported from several worker threads, so exercise
    that path: totals add up and the lock does not deadlock.

    Honest scope: this does NOT prove the lock is load-bearing. Under CPython's
    GIL the unlocked version loses nothing even at 40x2000 increments (measured),
    so this test passes with or without it. It guards the totals and the locking
    protocol, not a reproducible race.
    """
    usage = TurnCostAccumulator()
    n = 40
    barrier = threading.Barrier(n)

    def record_one():
        barrier.wait()  # maximise the overlap
        usage.record("openai", "gpt-4o", "platform", 1_000, 0)

    threads = [threading.Thread(target=record_one) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert usage.input_tokens == n * 1_000       # no lost tokens
    assert usage.priced_tokens == n * 1_000
    expected = n * 1_000 * 2.5 / 1_000_000       # $2.50/M
    assert usage.turn_cost_usd() == pytest.approx(expected)
    assert usage.platform_cost_usd() == pytest.approx(expected)


def test_accumulator_survives_pricing_failure():
    """Pricing must never break a chat turn: a raising estimate_cost degrades to
    "unpriced", and the tokens are still counted."""
    usage = TurnCostAccumulator()

    def _boom(*args, **kwargs):
        raise RuntimeError("pricing table exploded")

    import services.model_pricing as mp

    original, mp.estimate_cost = mp.estimate_cost, _boom
    try:
        usage.record("openai", "gpt-4o", "platform", 100, 50)
    finally:
        mp.estimate_cost = original

    assert usage.total_tokens == 150          # tokens survive
    assert usage.unpriced_tokens == 150       # but are honestly unpriced
    assert usage.turn_cost_usd() is None


# ── finalize_run persistence ────────────────────────────────────────────────


def test_finalize_run_persists_tokens_and_cost(runs):
    runs.start_run(
        run_id="r1", user_id="u1", conversation_id="c1", trace_id="t1",
        model="gpt-4o", provider="openai", key_source="platform",
    )
    runs.finalize_run(
        "r1", status="completed", duration_ms=1200, tools_called=[],
        input_tokens=1000, output_tokens=500,
        cost_usd=estimate_cost("openai", "gpt-4o", 1000, 500),
    )

    row = runs.get_run("r1", user_id="u1")
    assert row["input_tokens"] == 1000
    assert row["output_tokens"] == 500
    assert row["total_tokens"] == 1500
    # 1000 * $2.50/M + 500 * $10/M
    assert row["cost_usd"] == pytest.approx(0.0075)


def test_finalize_run_derives_total_from_parts(runs):
    """total_tokens is derived, so it can never disagree with its own parts."""
    runs.start_run(
        run_id="r2", user_id="u1", conversation_id="c1", trace_id="t1",
        model="gpt-4o", provider="openai", key_source="platform",
    )
    runs.finalize_run("r2", status="completed", duration_ms=1, input_tokens=7,
                      output_tokens=3, cost_usd=None)
    row = runs.get_run("r2", user_id="u1")
    assert row["total_tokens"] == 10
    assert row["cost_usd"] is None


def test_migration_preserves_legacy_chat_runs(tmp_path):
    """The new accounting columns must be added to a PRE-EXISTING chat_runs
    table without dropping legacy rows (the non-destructive ALTER idiom)."""
    from services.db import get_connection

    db_path = str(tmp_path / "legacy.db")
    # Stand up an old-schema chat_runs (no accounting columns) and seed a row.
    conn = get_connection(db_path)
    conn.execute(
        """
        CREATE TABLE chat_runs (
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL, conversation_id TEXT,
            trace_id TEXT, model TEXT NOT NULL, provider TEXT NOT NULL,
            key_source TEXT, status TEXT NOT NULL, started_at TEXT NOT NULL,
            updated_at TEXT NOT NULL, completed_at TEXT, duration_ms INTEGER,
            tools_called TEXT, last_status TEXT, error_code TEXT,
            error_message TEXT, client_ip TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO chat_runs (id, user_id, model, provider, status, started_at, updated_at)"
        " VALUES ('old', 'u1', 'gpt-4o', 'openai', 'completed', '2020-01-01T00:00:00+00:00', '2020-01-01T00:00:00+00:00')"
    )
    conn.commit()
    conn.close()

    # Running the service migrates the table in place.
    service = IssueReportService(db_path)
    legacy = service.get_run("old")
    assert legacy is not None                     # row survived
    assert legacy["model"] == "gpt-4o"
    assert legacy["input_tokens"] is None         # new column present, NULL for legacy
    assert legacy["cost_usd"] is None
    assert legacy["platform_cost_usd"] is None
    assert legacy["unpriced_tokens"] is None

    # And the migrated table accepts new accounting writes.
    service.finalize_run("old", status="completed", duration_ms=1, input_tokens=10,
                         output_tokens=5, cost_usd=0.001, unpriced_tokens=0)
    assert service.get_run("old")["total_tokens"] == 15


def test_finalize_run_without_usage_leaves_columns_null(runs):
    """Existing callers that pass no usage must not write a bogus 0."""
    runs.start_run(
        run_id="r3", user_id="u1", conversation_id="c1", trace_id="t1",
        model="gpt-oss-120b", provider="tacc", key_source="platform",
    )
    runs.finalize_run("r3", status="completed", duration_ms=1)
    row = runs.get_run("r3", user_id="u1")
    assert row["input_tokens"] is None
    assert row["total_tokens"] is None
    assert row["cost_usd"] is None


def test_usage_rollup_separates_unpriced_runs(runs):
    # (model, provider, cost, unpriced_tokens) — an all-TACC turn has all its
    # tokens unpriced, which is what sse.py persists.
    for i, (model, provider, cost, unpriced) in enumerate([
        ("gpt-4o", "openai", 0.01, 0),
        ("gpt-4o", "openai", 0.02, 0),
        ("gpt-oss-120b", "tacc", None, 100),  # fully unpriced
    ]):
        runs.start_run(
            run_id=f"x{i}", user_id="u1", conversation_id="c1", trace_id="t1",
            model=model, provider=provider, key_source="platform",
        )
        runs.finalize_run(f"x{i}", status="completed", duration_ms=1,
                          input_tokens=100, output_tokens=0, cost_usd=cost,
                          unpriced_tokens=unpriced)

    rollup = runs.usage_rollup(group_by="provider")
    assert rollup["totals"]["runs"] == 3
    assert rollup["totals"]["total_tokens"] == 300
    assert rollup["totals"]["cost_usd"] == pytest.approx(0.03)
    # The TACC run contributes tokens but no cost — the rollup must say so
    # rather than let $0.03 read as the full cost of all three runs.
    assert rollup["totals"]["unpriced_runs"] == 1
    assert rollup["totals"]["unpriced_tokens"] == 100


def test_usage_rollup_flags_partly_priced_run_as_unpriced(runs):
    """A turn priced in part (a TACC main model plus a priced OpenAI embedding)
    has a non-null cost_usd AND unpriced tokens. It must NOT read as fully
    priced — unpriced_tokens/unpriced_runs carry the honest signal (CX-01)."""
    runs.start_run(
        run_id="mix", user_id="u1", conversation_id="c1", trace_id="t1",
        model="gpt-oss-120b", provider="tacc", key_source="platform",
    )
    # 5000 total tokens, small priced cost, but 4900 tokens unpriced (TACC).
    runs.finalize_run("mix", status="completed", duration_ms=1, input_tokens=5000,
                      output_tokens=0, cost_usd=0.00002, unpriced_tokens=4900)

    rollup = runs.usage_rollup(group_by="provider")
    assert rollup["totals"]["cost_usd"] == pytest.approx(0.00002)
    assert rollup["totals"]["unpriced_tokens"] == 4900
    assert rollup["totals"]["unpriced_runs"] == 1  # flagged despite non-null cost


def test_usage_rollup_excludes_zero_token_failures_from_unpriced(runs):
    """A failed turn that spent nothing (0 tokens, NULL cost) must not inflate
    the unpriced-run count — it had nothing to price (CX-01)."""
    runs.start_run(
        run_id="fail", user_id="u1", conversation_id="c1", trace_id="t1",
        model="gpt-oss-120b", provider="tacc", key_source="platform",
    )
    runs.finalize_run("fail", status="failed", duration_ms=1, input_tokens=0,
                      output_tokens=0, cost_usd=None, unpriced_tokens=0)

    rollup = runs.usage_rollup(group_by="provider")
    assert rollup["totals"]["unpriced_runs"] == 0
    assert rollup["totals"]["unpriced_tokens"] == 0


def test_usage_rollup_since_normalizes_offset_timestamps(runs):
    """`since` must compare as an instant, not lexically — an offset timestamp
    that resolves to a LATER UTC instant must exclude an earlier UTC row
    (CX-06)."""
    runs.start_run(
        run_id="ts", user_id="u1", conversation_id="c1", trace_id="t1",
        model="gpt-4o", provider="openai", key_source="platform",
    )
    runs.finalize_run("ts", status="completed", duration_ms=1, input_tokens=100,
                      output_tokens=0, cost_usd=0.00025)
    row = runs.get_run("ts")
    # Force a known started_at at 14:00Z.
    conn = runs._conn()
    conn.execute("UPDATE chat_runs SET started_at = ? WHERE id = 'ts'",
                 ("2026-01-01T14:00:00+00:00",))
    conn.commit(); conn.close()

    # since = 10:00-05:00 == 15:00Z, which is AFTER 14:00Z → row excluded.
    # A naive lexical compare of "10:00" < "14:00" would wrongly include it.
    excluded = runs.usage_rollup(group_by="provider", since="2026-01-01T10:00:00-05:00")
    assert excluded["totals"]["runs"] == 0
    # since = 10:00Z is genuinely before 14:00Z → row included.
    included = runs.usage_rollup(group_by="provider", since="2026-01-01T10:00:00Z")
    assert included["totals"]["runs"] == 1


def test_usage_rollup_rejects_malformed_since(runs):
    with pytest.raises(ValueError):
        runs.usage_rollup(group_by="provider", since="not-a-timestamp")


def test_usage_rollup_rejects_unknown_group_by(runs):
    """group_by is interpolated into SQL, so it must be allowlisted."""
    with pytest.raises(ValueError):
        runs.usage_rollup(group_by="user_id; DROP TABLE chat_runs")


def test_tiny_turn_cost_persisted_unrounded(runs):
    """A sub-6dp turn cost must persist unrounded so many tiny turns don't each
    zero out and sum to $0 (CX-08)."""
    tiny = estimate_cost("openai", "gpt-4o-mini", 1, 0)  # ~1.5e-7
    assert tiny is not None and 0 < tiny < 1e-6  # below the 6dp display floor
    for i in range(20):
        runs.start_run(
            run_id=f"tiny{i}", user_id="u1", conversation_id="c1", trace_id="t1",
            model="gpt-4o-mini", provider="openai", key_source="platform",
        )
        runs.finalize_run(f"tiny{i}", status="completed", duration_ms=1,
                          input_tokens=1, output_tokens=0, cost_usd=tiny,
                          unpriced_tokens=0)
    # The persisted per-turn value is the unrounded tiny cost, not 0.
    assert runs.get_run("tiny0")["cost_usd"] == pytest.approx(tiny)
    # And 20 of them sum to a measurable, non-zero total.
    assert runs.usage_rollup(group_by="provider")["totals"]["cost_usd"] > 0


def test_turn_rollup_separates_platform_cost_from_byok(runs):
    """chat_runs.cost_usd is the turn's FULL cost whoever paid, so summing it
    puts user-funded BYOK spend in Quasar's bill. platform_cost_usd carries the
    per-call-attributed platform subset, and the rollup must report both (CX-28).
    """
    # A BYOK turn: $2.50 of the user's own money, plus a $0.01 platform embedding.
    runs.start_run(
        run_id="byok", user_id="u1", conversation_id="c1", trace_id="t1",
        model="claude-opus-4-8", provider="anthropic", key_source="byok",
    )
    runs.finalize_run("byok", status="completed", duration_ms=1,
                      input_tokens=1_000_000, output_tokens=0,
                      cost_usd=2.51, platform_cost_usd=0.01, unpriced_tokens=0)
    # A wholly platform turn.
    runs.start_run(
        run_id="plat", user_id="u1", conversation_id="c1", trace_id="t1",
        model="gpt-4o", provider="openai", key_source="platform",
    )
    runs.finalize_run("plat", status="completed", duration_ms=1,
                      input_tokens=1_000_000, output_tokens=0,
                      cost_usd=2.5, platform_cost_usd=2.5, unpriced_tokens=0)

    totals = runs.usage_rollup(group_by="provider")["totals"]
    # Full cost of producing both answers: the cost-efficiency number.
    assert totals["cost_usd"] == pytest.approx(5.01)
    # What Quasar actually owes: the platform turn plus only the BYOK turn's
    # platform-paid embedding — NOT the user's $2.50.
    assert totals["platform_cost_usd"] == pytest.approx(2.51)


def test_platform_cost_survives_legacy_and_unset_rows(runs):
    """platform_cost_usd is NULL for rows written before the column existed (and
    by callers that pass no usage). Those must not poison the SUM."""
    runs.start_run(
        run_id="legacy", user_id="u1", conversation_id="c1", trace_id="t1",
        model="gpt-4o", provider="openai", key_source="platform",
    )
    runs.finalize_run("legacy", status="completed", duration_ms=1,
                      input_tokens=100, output_tokens=0, cost_usd=0.00025)

    assert runs.get_run("legacy")["platform_cost_usd"] is None
    totals = runs.usage_rollup(group_by="provider")["totals"]
    assert totals["cost_usd"] == pytest.approx(0.00025)
    assert totals["platform_cost_usd"] == 0.0


# ── global (per-call) admin rollup ──────────────────────────────────────────


def test_global_rollup_attributes_spend_per_call_not_per_turn(quota_store):
    """The admin rollup must attribute each call to the route that ran it. A
    platform embedding inside a BYOK turn must NOT be hidden under BYOK (CX-04).
    """
    # Same logical turn: BYOK anthropic main call + platform OpenAI embedding.
    _spend(quota_store, 1000, provider="anthropic", model="claude-opus-4-8",
           key_source="byok")
    _spend(quota_store, 100, provider="openai", model="text-embedding-ada-002",
           key_source="platform")

    by_source = {
        g["key_source"]: g
        for g in quota_store.global_usage_rollup(group_by="key_source")["groups"]
    }
    assert by_source["platform"]["total_tokens"] == 100   # embedding NOT hidden
    assert by_source["byok"]["total_tokens"] == 1000
    # The platform embedding carries a real (priced) cost attributed to platform.
    assert by_source["platform"]["cost_usd"] > 0


def test_global_rollup_prices_per_provider_model(quota_store):
    _spend(quota_store, 1_000_000, provider="openai", model="gpt-4o",
           key_source="platform")
    _spend(quota_store, 1_000_000, provider="tacc", model="gpt-oss-120b",
           key_source="platform")

    rollup = quota_store.global_usage_rollup(group_by="provider")
    by_provider = {g["provider"]: g for g in rollup["groups"]}
    assert by_provider["openai"]["cost_usd"] == pytest.approx(2.5)
    # TACC has no price → tokens counted as unpriced, cost stays 0.
    assert by_provider["tacc"]["cost_usd"] == 0.0
    assert by_provider["tacc"]["unpriced_tokens"] == 1_000_000
    assert rollup["totals"]["unpriced_tokens"] == 1_000_000


def test_global_rollup_rejects_unknown_group_by(quota_store):
    with pytest.raises(ValueError):
        quota_store.global_usage_rollup(group_by="user_id; DROP")


def test_global_rollup_total_survives_many_subcent_groups(quota_store):
    """The total must sum RAW group costs, not per-group-rounded ones — 20 users
    each making one one-token call each round to $0 individually but total a
    measurable ~$0.000003 (CX-08 aggregate case)."""
    for i in range(20):
        _spend(quota_store, 1, provider="openai", model="gpt-4o-mini",
               user_id=f"u{i}")
    rollup = quota_store.global_usage_rollup(group_by="user")
    # Each per-user group is sub-cent and rounds to $0 for display...
    assert all(g["cost_usd"] == 0.0 for g in rollup["groups"])
    # ...but the summed total is raw, so it is non-zero.
    assert rollup["totals"]["cost_usd"] > 0


# ── daily cap ───────────────────────────────────────────────────────────────


def test_daily_cap_disabled_by_default(monkeypatch, quota_store):
    """Merging this feature must not start capping any existing deployment.

    Spend stays under TACC's 1M weekly quota so this isolates the daily cap
    rather than tripping the pre-existing weekly one.
    """
    monkeypatch.delenv("QUASAR_DAILY_TOKEN_LIMIT", raising=False)
    _spend(quota_store, 900_000, provider="tacc", model="gpt-oss-120b")
    quota_store.ensure_allowed(
        user_id="u1", user_email="person@example.test",
        provider="tacc", key_source="platform",
    )


@pytest.mark.parametrize("raw", ["0", "", "-5", "not-a-number"])
def test_daily_cap_disabled_for_non_positive_or_malformed(monkeypatch, quota_store, raw):
    monkeypatch.setenv("QUASAR_DAILY_TOKEN_LIMIT", raw)
    assert uqs.daily_token_limit() is None


def test_daily_cap_raises_at_threshold(monkeypatch, quota_store):
    monkeypatch.setenv("QUASAR_DAILY_TOKEN_LIMIT", "1000")

    _spend(quota_store, 999)
    # Under the cap: still allowed.
    quota_store.ensure_allowed(
        user_id="u1", user_email="person@example.test",
        provider="openai", key_source="platform",
    )

    _spend(quota_store, 1)  # now exactly at 1000
    with pytest.raises(QuotaExceededError):
        quota_store.ensure_allowed(
            user_id="u1", user_email="person@example.test",
            provider="openai", key_source="platform",
        )


def test_daily_cap_is_aggregate_across_providers(monkeypatch, quota_store):
    """The cap is a single total, so spend cannot be laundered by hopping
    providers."""
    monkeypatch.setenv("QUASAR_DAILY_TOKEN_LIMIT", "1000")
    _spend(quota_store, 600, provider="openai")
    _spend(quota_store, 400, provider="deepseek")

    with pytest.raises(QuotaExceededError):
        quota_store.ensure_allowed(
            user_id="u1", user_email="person@example.test",
            provider="deepseek", key_source="platform",
        )


def test_daily_cap_applies_to_providers_with_no_weekly_quota(monkeypatch, quota_store):
    """Regression guard for the ordering hazard in ensure_allowed.

    anthropic/google are absent from PLATFORM_TOKEN_LIMITS, so the weekly block
    early-returns for them. If the daily check ran after that return, picking
    such a provider would bypass the cap outright.
    """
    monkeypatch.setenv("QUASAR_DAILY_TOKEN_LIMIT", "1000")
    _spend(quota_store, 1000, provider="anthropic", model="claude-opus-4-8")

    assert "anthropic" not in uqs.PLATFORM_TOKEN_LIMITS  # premise of this test
    with pytest.raises(QuotaExceededError):
        quota_store.ensure_allowed(
            user_id="u1", user_email="person@example.test",
            provider="anthropic", key_source="platform",
        )


def test_daily_cap_stacks_with_weekly_quota(monkeypatch, quota_store):
    """Adding the daily cap must not loosen the weekly quota."""
    monkeypatch.setenv("QUASAR_DAILY_TOKEN_LIMIT", "10_000_000")  # far above weekly
    _spend(quota_store, 100_000)  # exhausts the 100k/week openai allowance

    with pytest.raises(QuotaExceededError):
        quota_store.ensure_allowed(
            user_id="u1", user_email="person@example.test",
            provider="openai", key_source="platform",
        )


def test_daily_cap_is_per_user(monkeypatch, quota_store):
    monkeypatch.setenv("QUASAR_DAILY_TOKEN_LIMIT", "1000")
    _spend(quota_store, 1000, user_id="u1")

    # u2 has spent nothing and must be unaffected by u1's spend.
    quota_store.ensure_allowed(
        user_id="u2", user_email="other@example.test",
        provider="openai", key_source="platform",
    )


def test_daily_cap_only_counts_the_rolling_window(monkeypatch, quota_store):
    """Spend older than the window must not count against today."""
    monkeypatch.setenv("QUASAR_DAILY_TOKEN_LIMIT", "1000")
    _spend(quota_store, 5000)

    conn = quota_store._conn()
    conn.execute(
        "UPDATE llm_usage_events SET created_at = ?",
        ("2020-01-01T00:00:00+00:00",),
    )
    conn.commit()
    conn.close()

    quota_store.ensure_allowed(
        user_id="u1", user_email="person@example.test",
        provider="openai", key_source="platform",
    )


def test_per_provider_daily_cap_when_configured(monkeypatch, quota_store):
    monkeypatch.delenv("QUASAR_DAILY_TOKEN_LIMIT", raising=False)
    monkeypatch.setattr(uqs, "PLATFORM_DAILY_TOKEN_LIMITS", {"openai": 500})
    _spend(quota_store, 500, provider="openai")

    with pytest.raises(QuotaExceededError):
        quota_store.ensure_allowed(
            user_id="u1", user_email="person@example.test",
            provider="openai", key_source="platform",
        )
    # A different provider is untouched by an openai-only daily cap.
    quota_store.ensure_allowed(
        user_id="u1", user_email="person@example.test",
        provider="deepseek", key_source="platform",
    )


# ── daily cap bypasses (the security-critical cohorts) ──────────────────────


def test_byok_bypasses_daily_cap(monkeypatch, quota_store):
    """BYOK spend is on the user's own key, so Quasar has no cost to cap."""
    monkeypatch.setenv("QUASAR_DAILY_TOKEN_LIMIT", "1000")
    _spend(quota_store, 10_000_000, key_source="byok")

    quota_store.ensure_allowed(
        user_id="u1", user_email="person@example.test",
        provider="openai", key_source="byok", byok_token_limit=None,
    )


def test_byok_daily_bypass_survives_platform_spend(monkeypatch, quota_store):
    """A user over the platform daily cap can still use their own key."""
    monkeypatch.setenv("QUASAR_DAILY_TOKEN_LIMIT", "1000")
    _spend(quota_store, 5000, key_source="platform")

    with pytest.raises(QuotaExceededError):
        quota_store.ensure_allowed(
            user_id="u1", user_email="person@example.test",
            provider="openai", key_source="platform",
        )
    quota_store.ensure_allowed(
        user_id="u1", user_email="person@example.test",
        provider="openai", key_source="byok", byok_token_limit=None,
    )


def test_quota_exempt_email_bypasses_daily_cap(monkeypatch, quota_store):
    """The NRAO/admin exempt cohort must stay uncapped — a regression here
    silently throttles the people the exemption exists for."""
    monkeypatch.setenv("QUASAR_DAILY_TOKEN_LIMIT", "1000")
    monkeypatch.setenv("QUASAR_TOKEN_LIMIT_EXEMPT_EMAILS", "exempt@example.test")
    _spend(quota_store, 10_000_000, user_id="exempt-user")

    quota_store.ensure_allowed(
        user_id="exempt-user", user_email="exempt@example.test",
        provider="openai", key_source="platform",
    )


def test_admin_email_bypasses_daily_cap(monkeypatch, quota_store):
    monkeypatch.setenv("QUASAR_DAILY_TOKEN_LIMIT", "1000")
    monkeypatch.setenv("ADMIN_EMAILS", "boss@example.test")
    _spend(quota_store, 10_000_000, user_id="admin-user")

    quota_store.ensure_allowed(
        user_id="admin-user", user_email="boss@example.test",
        provider="openai", key_source="platform",
    )


# ── usage_summary ───────────────────────────────────────────────────────────


def test_usage_summary_reports_daily_and_cost(monkeypatch, quota_store):
    monkeypatch.setenv("QUASAR_DAILY_TOKEN_LIMIT", "1000")
    _spend(quota_store, 400, provider="openai", model="gpt-4o")

    summary = quota_store.usage_summary(user_id="u1", user_email="person@example.test")

    assert summary["daily"]["used_tokens"] == 400
    assert summary["daily"]["limit_tokens"] == 1000
    assert summary["daily"]["remaining_tokens"] == 600
    assert summary["daily"]["unlimited"] is False
    assert summary["daily"]["exhausted"] is False

    # 400 input tokens at $2.50/M.
    assert summary["cost"]["platform_today"]["cost_usd"] == pytest.approx(0.001)
    assert summary["cost"]["platform_today"]["unpriced_tokens"] == 0
    assert summary["cost"]["is_estimate"] is True
    assert summary["cost"]["pricing_last_verified"] == model_pricing.PRICING_LAST_VERIFIED


def test_usage_summary_counts_unpriced_tokens_separately(quota_store):
    """A TACC-only user must not read as "$0.00 spent, nothing to see"."""
    _spend(quota_store, 5000, provider="tacc", model="gpt-oss-120b")

    summary = quota_store.usage_summary(user_id="u1", user_email="person@example.test")
    today = summary["cost"]["platform_today"]
    assert today["cost_usd"] == 0.0
    assert today["total_tokens"] == 5000
    assert today["unpriced_tokens"] == 5000


def test_usage_summary_reports_exempt_user_as_uncapped(monkeypatch, quota_store):
    """ensure_allowed returns before any cap for exempt users, so the summary
    must not advertise a limit that is never enforced."""
    monkeypatch.setenv("QUASAR_DAILY_TOKEN_LIMIT", "1000")
    monkeypatch.setenv("QUASAR_TOKEN_LIMIT_EXEMPT_EMAILS", "exempt@example.test")

    summary = quota_store.usage_summary(
        user_id="exempt-user", user_email="exempt@example.test"
    )
    assert summary["daily"]["unlimited"] is True
    assert summary["daily"]["limit_tokens"] is None
    assert summary["daily"]["exhausted"] is False


def test_usage_summary_splits_platform_and_byok_cost(quota_store):
    """Platform spend is Quasar's bill; BYOK spend is the user's. Never merged."""
    _spend(quota_store, 1_000_000, provider="openai", model="gpt-4o",
           key_source="platform")
    _spend(quota_store, 1_000_000, provider="openai", model="gpt-4o",
           key_source="byok")

    summary = quota_store.usage_summary(user_id="u1", user_email="person@example.test")
    assert summary["cost"]["platform_today"]["cost_usd"] == pytest.approx(2.5)
    assert summary["cost"]["byok_today"]["cost_usd"] == pytest.approx(2.5)


# ── concurrent admission (CX-29) ────────────────────────────────────────────
#
# The cap was a read-then-decide check with nothing recording the decision, so
# concurrent callers each read the same under-limit total and every one of them
# passed. These tests drive ensure_allowed from real threads released together,
# because that race is invisible to sequential calls — which is exactly why the
# original cost tests missed it (CX-33).


def _admit_concurrently(store, n, *, reserve, spend_each=0, user_id="u1",
                        user_email="person@example.test", provider="openai"):
    """Fire `n` ensure_allowed calls at once; return how many were admitted.

    A barrier releases every thread on the same instant so they overlap inside
    the check, rather than accidentally serialising and passing a broken cap.
    """
    admitted = []
    errors = []
    barrier = threading.Barrier(n)

    def attempt():
        try:
            barrier.wait()
            handle = store.ensure_allowed(
                user_id=user_id, user_email=user_email, provider=provider,
                key_source="platform", reserve=reserve,
            )
            admitted.append(handle)
            if spend_each:
                store.record_usage(
                    UsageRecord(user_id=user_id, provider=provider, model="gpt-4o",
                                key_source="platform", input_tokens=spend_each,
                                output_tokens=0),
                    reservation_id=handle,
                )
        except QuotaExceededError:
            pass
        except Exception as exc:  # surface real breakage, don't read as "denied"
            errors.append(exc)

    threads = [threading.Thread(target=attempt) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, f"unexpected errors: {errors!r}"
    return len(admitted)


def test_concurrent_calls_cannot_all_pass_one_call_of_headroom(monkeypatch, quota_store):
    """THE CX-29 regression. With room for one call, exactly one is admitted.

    Without an atomic reserve every thread reads 999 < 1000 and passes, so all
    20 run and collectively spend ~11x the cap.
    """
    monkeypatch.setenv("QUASAR_DAILY_TOKEN_LIMIT", "1000")
    monkeypatch.setenv("QUASAR_CALL_TOKEN_RESERVATION", "500")
    _spend(quota_store, 999)  # one token of headroom

    admitted = _admit_concurrently(quota_store, 20, reserve=True, spend_each=500)
    assert admitted == 1

    used = quota_store.get_used_tokens_all_providers(
        "u1", "platform", since=quota_store.daily_window_start()
    )
    # The one admitted call's real spend is the only overshoot; a turn cannot be
    # gated on tokens it has not requested yet.
    assert used == 1499


def test_concurrent_admission_does_not_falsely_reject_a_user_with_headroom(
    monkeypatch, quota_store
):
    """Reservations must not turn into a covert concurrency limit.

    Admission stays `used + outstanding < limit`, so allowed concurrency scales
    with remaining headroom. A user at zero usage against a large cap must not
    be told their allowance is exhausted.
    """
    monkeypatch.setenv("QUASAR_DAILY_TOKEN_LIMIT", "1000000")
    monkeypatch.setenv("QUASAR_CALL_TOKEN_RESERVATION", "4000")

    assert _admit_concurrently(quota_store, 20, reserve=True) == 20


def test_concurrent_refusal_reports_in_flight_not_exhausted(monkeypatch, quota_store):
    """A caller refused because its OWN other calls hold the headroom must not be
    told its allowance is gone — it may be nowhere near the cap."""
    monkeypatch.setenv("QUASAR_DAILY_TOKEN_LIMIT", "1000")
    # One in-flight call reserves the whole cap, so the next is refused with
    # nothing on the ledger to blame it on.
    monkeypatch.setenv("QUASAR_CALL_TOKEN_RESERVATION", "1000")
    # Nothing spent at all: the only thing consuming headroom is the reservation.
    first = quota_store.ensure_allowed(
        user_id="u1", user_email="person@example.test", provider="openai",
        key_source="platform", reserve=True,
    )
    assert first is not None
    assert quota_store.get_used_tokens("u1", "openai", "platform") == 0  # no spend

    with pytest.raises(QuotaExceededError, match="in flight"):
        quota_store.ensure_allowed(
            user_id="u1", user_email="person@example.test", provider="openai",
            key_source="platform", reserve=True,
        )


def test_exhausted_cap_still_reports_the_specific_cap(monkeypatch, quota_store):
    """The in-flight message must not swallow the real 'allowance exhausted'
    case — a genuinely capped user gets the specific weekly-quota message."""
    monkeypatch.delenv("QUASAR_DAILY_TOKEN_LIMIT", raising=False)
    monkeypatch.setenv("QUASAR_CALL_TOKEN_RESERVATION", "4000")
    _spend(quota_store, 100_000)  # exhausts the openai weekly allowance

    with pytest.raises(QuotaExceededError, match="per week"):
        quota_store.ensure_allowed(
            user_id="u1", user_email="person@example.test", provider="openai",
            key_source="platform", reserve=True,
        )


def test_recording_usage_settles_the_reservation(monkeypatch, quota_store):
    """A settled call must stop holding tokens, or a user's own completed calls
    would eat their headroom until the TTL."""
    monkeypatch.setenv("QUASAR_DAILY_TOKEN_LIMIT", "1000")
    monkeypatch.setenv("QUASAR_CALL_TOKEN_RESERVATION", "500")

    handle = quota_store.ensure_allowed(
        user_id="u1", user_email="person@example.test", provider="openai",
        key_source="platform", reserve=True,
    )
    assert quota_store.outstanding_reservation_tokens("u1") == 500

    quota_store.record_usage(
        UsageRecord(user_id="u1", provider="openai", model="gpt-4o",
                    key_source="platform", input_tokens=10, output_tokens=0),
        reservation_id=handle,
    )
    assert quota_store.outstanding_reservation_tokens("u1") == 0
    assert quota_store.get_used_tokens("u1", "openai", "platform") == 10


def test_reservation_released_when_call_records_nothing(monkeypatch, quota_store):
    """A failed call reports no tokens. Its hold must come off anyway."""
    monkeypatch.setenv("QUASAR_DAILY_TOKEN_LIMIT", "1000")
    monkeypatch.setenv("QUASAR_CALL_TOKEN_RESERVATION", "500")

    handle = quota_store.ensure_allowed(
        user_id="u1", user_email="person@example.test", provider="openai",
        key_source="platform", reserve=True,
    )
    quota_store.release_reservation(handle)
    assert quota_store.outstanding_reservation_tokens("u1") == 0
    # Idempotent: the error path may release a reservation twice.
    quota_store.release_reservation(handle)
    assert quota_store.outstanding_reservation_tokens("u1") == 0


def test_zero_token_usage_still_releases_the_reservation(monkeypatch, quota_store):
    """record_usage short-circuits on a zero-token record; it must not leak the
    hold on the way out."""
    monkeypatch.setenv("QUASAR_DAILY_TOKEN_LIMIT", "1000")
    monkeypatch.setenv("QUASAR_CALL_TOKEN_RESERVATION", "500")

    handle = quota_store.ensure_allowed(
        user_id="u1", user_email="person@example.test", provider="openai",
        key_source="platform", reserve=True,
    )
    quota_store.record_usage(
        UsageRecord(user_id="u1", provider="openai", model="gpt-4o",
                    key_source="platform", input_tokens=0, output_tokens=0),
        reservation_id=handle,
    )
    assert quota_store.outstanding_reservation_tokens("u1") == 0


def test_abandoned_reservation_expires(monkeypatch, quota_store):
    """A crashed turn never settles. Its hold must not lock the user out past
    the TTL."""
    monkeypatch.setenv("QUASAR_DAILY_TOKEN_LIMIT", "1000")
    monkeypatch.setenv("QUASAR_CALL_TOKEN_RESERVATION", "900")
    # Born already expired, standing in for a hold whose TTL has passed.
    monkeypatch.setattr(uqs, "CALL_RESERVATION_TTL_SECONDS", -1)

    assert quota_store.ensure_allowed(
        user_id="u1", user_email="person@example.test", provider="openai",
        key_source="platform", reserve=True,
    ) is not None
    # Expired holds count for nothing, so the next call is still admitted.
    assert quota_store.outstanding_reservation_tokens("u1") == 0
    assert quota_store.ensure_allowed(
        user_id="u1", user_email="person@example.test", provider="openai",
        key_source="platform", reserve=True,
    ) is not None


def test_reservations_are_never_reported_as_spend(monkeypatch, quota_store):
    """Reservations are estimates. If one reached the usage ledger it would be
    billed as a fact and inflate every cost report."""
    monkeypatch.setenv("QUASAR_DAILY_TOKEN_LIMIT", "1000")
    monkeypatch.setenv("QUASAR_CALL_TOKEN_RESERVATION", "500")
    quota_store.ensure_allowed(
        user_id="u1", user_email="person@example.test", provider="openai",
        key_source="platform", reserve=True,
    )

    summary = quota_store.usage_summary(user_id="u1", user_email="person@example.test")
    assert summary["daily"]["used_tokens"] == 0
    assert summary["cost"]["platform_today"]["cost_usd"] == 0.0
    assert summary["cost"]["platform_today"]["total_tokens"] == 0
    assert quota_store.global_usage_rollup(group_by="provider")["groups"] == []


def test_reservations_disabled_restores_pure_read_check(monkeypatch, quota_store):
    """E<=0 opts out entirely: no holds, and the historical check still caps."""
    monkeypatch.setenv("QUASAR_DAILY_TOKEN_LIMIT", "1000")
    monkeypatch.setenv("QUASAR_CALL_TOKEN_RESERVATION", "0")

    assert quota_store.ensure_allowed(
        user_id="u1", user_email="person@example.test", provider="openai",
        key_source="platform", reserve=True,
    ) is None
    assert quota_store.outstanding_reservation_tokens("u1") == 0

    _spend(quota_store, 1000)
    with pytest.raises(QuotaExceededError):
        quota_store.ensure_allowed(
            user_id="u1", user_email="person@example.test", provider="openai",
            key_source="platform", reserve=True,
        )


def test_sequential_admission_is_unchanged_by_reservations(monkeypatch, quota_store):
    """The predicate stays `used < limit`, so a settled sequence admits exactly
    as it did before reservations existed — nobody is cut off early."""
    monkeypatch.setenv("QUASAR_DAILY_TOKEN_LIMIT", "1000")
    monkeypatch.setenv("QUASAR_CALL_TOKEN_RESERVATION", "4000")  # far above the cap

    # 999 spent, one token of headroom, no calls in flight: still admitted, even
    # though the reservation alone dwarfs the cap.
    _spend(quota_store, 999)
    handle = quota_store.ensure_allowed(
        user_id="u1", user_email="person@example.test", provider="openai",
        key_source="platform", reserve=True,
    )
    assert handle is not None
    quota_store.record_usage(
        UsageRecord(user_id="u1", provider="openai", model="gpt-4o",
                    key_source="platform", input_tokens=1, output_tokens=0),
        reservation_id=handle,
    )
    # Now at exactly 1000/1000 — the cap trips, as it always did.
    with pytest.raises(QuotaExceededError):
        quota_store.ensure_allowed(
            user_id="u1", user_email="person@example.test", provider="openai",
            key_source="platform", reserve=True,
        )


# ── concurrent bypasses: the exempt cohorts must stay uncapped ──────────────


def test_byok_is_never_reserved_under_concurrency(monkeypatch, quota_store):
    """BYOK spends the user's own key, so it takes no hold and no platform cap
    applies — even with many calls in flight."""
    monkeypatch.setenv("QUASAR_DAILY_TOKEN_LIMIT", "1000")
    monkeypatch.setenv("QUASAR_CALL_TOKEN_RESERVATION", "900")
    _spend(quota_store, 5000, key_source="platform")  # platform is over its cap

    admitted, errors = [], []
    barrier = threading.Barrier(10)

    def attempt():
        try:
            barrier.wait()
            admitted.append(
                quota_store.ensure_allowed(
                    user_id="u1", user_email="person@example.test",
                    provider="openai", key_source="byok", byok_token_limit=None,
                    reserve=True,
                )
            )
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=attempt) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(admitted) == 10
    assert all(h is None for h in admitted)  # no hold taken
    assert quota_store.outstanding_reservation_tokens("u1", "byok") == 0


def test_quota_exempt_email_is_never_reserved_under_concurrency(monkeypatch, quota_store):
    """The NRAO/admin exempt cohort must stay uncapped. A hold here would
    throttle exactly the people the exemption exists for."""
    monkeypatch.setenv("QUASAR_DAILY_TOKEN_LIMIT", "1000")
    monkeypatch.setenv("QUASAR_CALL_TOKEN_RESERVATION", "900")
    monkeypatch.setenv("QUASAR_TOKEN_LIMIT_EXEMPT_EMAILS", "exempt@example.test")
    _spend(quota_store, 10_000_000, user_id="exempt-user")

    admitted = _admit_concurrently(
        quota_store, 10, reserve=True,
        user_id="exempt-user", user_email="exempt@example.test",
    )
    assert admitted == 10
    assert quota_store.outstanding_reservation_tokens("exempt-user") == 0


def test_uncapped_provider_takes_no_reservation(monkeypatch, quota_store):
    """anthropic/google have no weekly quota. With the daily cap off, no cap
    applies at all, so they must keep their zero-write fast path."""
    monkeypatch.delenv("QUASAR_DAILY_TOKEN_LIMIT", raising=False)
    monkeypatch.setenv("QUASAR_CALL_TOKEN_RESERVATION", "4000")

    assert "anthropic" not in uqs.PLATFORM_TOKEN_LIMITS  # premise
    assert quota_store.ensure_allowed(
        user_id="u1", user_email="person@example.test", provider="anthropic",
        key_source="platform", reserve=True,
    ) is None
    assert quota_store.outstanding_reservation_tokens("u1") == 0


def test_concurrent_mixed_byok_and_platform_calls_attribute_separately(
    monkeypatch, quota_store
):
    """Mixed-route concurrency (CX-33): platform calls are capped and held while
    BYOK calls run free, and each lands under its own key source."""
    monkeypatch.setenv("QUASAR_DAILY_TOKEN_LIMIT", "1000")
    monkeypatch.setenv("QUASAR_CALL_TOKEN_RESERVATION", "500")
    _spend(quota_store, 999)  # platform: one token of headroom

    platform_admitted, byok_admitted, errors = [], [], []
    barrier = threading.Barrier(16)

    def attempt(key_source):
        try:
            barrier.wait()
            handle = quota_store.ensure_allowed(
                user_id="u1", user_email="person@example.test", provider="openai",
                key_source=key_source, byok_token_limit=None, reserve=True,
            )
            (platform_admitted if key_source == "platform" else byok_admitted).append(1)
            quota_store.record_usage(
                UsageRecord(user_id="u1", provider="openai", model="gpt-4o",
                            key_source=key_source, input_tokens=100, output_tokens=0),
                reservation_id=handle,
            )
        except QuotaExceededError:
            pass
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=attempt, args=("platform" if i % 2 else "byok",))
        for i in range(16)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(byok_admitted) == 8      # BYOK is uncapped
    assert len(platform_admitted) == 1  # platform had room for exactly one
    # Each call is booked to the source that actually paid for it.
    assert quota_store.get_used_tokens("u1", "openai", "byok") == 800
    assert quota_store.get_used_tokens("u1", "openai", "platform") == 999 + 100


# ---------------------------------------------------------------------------
# Conductor-path accounting (A2 CX-01 / CX-13): executor threads must carry
# the request's usage/quota context.
# ---------------------------------------------------------------------------
def test_conductor_executor_threads_carry_the_usage_context():
    """Drive the REAL _execute_node submission path with a spy tool executor.

    The accounting context lives in a threading.local; run_in_executor threads
    never inherit it, so before the fix every conductor subtask's LLM spend
    was unrecorded and unmetered. This asserts the context (and its
    usage_recorder) is visible inside the executor thread — written to fail
    on the pre-fix code.
    """
    import asyncio

    from core.conductor import Conductor, OrchestrationRun
    from core.llm_client import get_llm_request_context, llm_request_context
    from core.task_dag import TaskDAG
    from core.workflow_memory import WorkflowMemory

    recorded = []
    seen = {}

    def spy_tool_executor(task_description, dep_context="", subtask_model="", user_id=""):
        ctx = get_llm_request_context()
        seen["ctx_visible"] = ctx is not None
        if ctx is not None and ctx.usage_recorder:
            ctx.usage_recorder(model="fake-model", input_tokens=3, output_tokens=5)
        return "subtask done"

    conductor = Conductor(client=object(), tool_executor=spy_tool_executor)
    conductor.model_router = None
    conductor.sandbox_executor = None
    conductor.recovery = None

    dag = TaskDAG()
    dag.build_from_subtasks(
        [{"id": "t1", "description": "measure", "agent_type": "general", "depends_on": []}]
    )
    run = OrchestrationRun(dag=dag, workflow_memory=WorkflowMemory())
    node = dag.nodes["t1"]

    with llm_request_context(
        user_id="u-conductor",
        usage_recorder=lambda **kw: recorded.append(kw),
    ):
        result = asyncio.run(conductor._execute_node(run, node, user_id="u-conductor"))

    assert result == "subtask done"
    assert seen["ctx_visible"], (
        "conductor executor thread did not see the request's LLM accounting "
        "context — subtask LLM calls would go unrecorded and unmetered"
    )
    assert recorded and recorded[0]["model"] == "fake-model"


def test_reinstall_llm_request_context_propagates_to_a_bare_thread():
    """The runner's conductor thread is a bare threading.Thread (runner.py):
    the captured context must survive the hop and restore cleanly after."""
    import threading

    from core.llm_client import (
        get_llm_request_context,
        llm_request_context,
        reinstall_llm_request_context,
    )

    calls = []
    with llm_request_context(user_id="u2", usage_recorder=lambda **kw: calls.append(kw)):
        captured = get_llm_request_context()
        inner_seen = {}

        def worker():
            inner_seen["before"] = get_llm_request_context()
            with reinstall_llm_request_context(captured):
                ctx = get_llm_request_context()
                inner_seen["inside"] = ctx
                ctx.usage_recorder(model="m", input_tokens=1, output_tokens=1)
            inner_seen["after"] = get_llm_request_context()

        t = threading.Thread(target=worker)
        t.start()
        t.join()

    assert inner_seen["before"] is None          # threads do NOT inherit — the bug
    assert inner_seen["inside"] is captured      # reinstall carries it across
    assert inner_seen["after"] is None           # and restores cleanly
    assert calls and calls[0]["model"] == "m"


def test_llm_client_create_records_usage_and_settles_reservation():  # A2 CX-13
    """Drive the REAL LLMClient.responses.create path — provider transport
    stubbed at the _call_openai seam, everything else real: admission through
    the context quota_checker, _record_usage reading the provider-reported
    usage block, the reservation settled INTO the recorder, and the failure
    path releasing it. This is the standard-path accounting flow end-to-end
    minus only the wire."""
    import types

    from core.llm_client import LLMClient, llm_request_context

    client = LLMClient(model="gpt-4o-mini")
    shim = client.responses

    fake_result = types.SimpleNamespace(
        usage=types.SimpleNamespace(input_tokens=11, output_tokens=7),
        output_text="ok",
    )
    shim._call_openai = lambda kwargs, attachments=None: fake_result

    recorded = []
    granted = []
    released = []

    def quota_checker(**kw):
        granted.append(kw)
        return f"res-{len(granted)}"

    with llm_request_context(
        user_id="u-e2e",
        usage_recorder=lambda **kw: recorded.append(kw),
        quota_checker=quota_checker,
        quota_releaser=lambda rid: released.append(rid),
    ):
        out = shim.create(model="gpt-4o-mini", input="hi", stream=False)

    assert out is fake_result
    assert granted and granted[0]["provider"] == "openai"
    assert recorded, "the real create() path never reached the usage recorder"
    assert recorded[0]["input_tokens"] == 11 and recorded[0]["output_tokens"] == 7
    assert recorded[0]["reservation_id"] == "res-1"   # settled, not leaked
    assert released == []                              # success path releases nothing

    # Failure path: the call never produced usage → the reservation is RELEASED.
    def _boom(kwargs, attachments=None):
        raise RuntimeError("provider down")

    shim._call_openai = _boom
    with llm_request_context(
        user_id="u-e2e",
        usage_recorder=lambda **kw: recorded.append(kw),
        quota_checker=quota_checker,
        quota_releaser=lambda rid: released.append(rid),
    ):
        try:
            shim.create(model="gpt-4o-mini", input="hi", stream=False)
        except RuntimeError:
            pass

    assert released == ["res-2"]


def test_turn_accumulator_freeze_drops_late_calls():  # A2 CX-17
    """Late conductor-thread calls arriving AFTER the usage snapshot must not
    mutate the reported turn totals; they stay counted in the per-call ledger
    and are surfaced via late_calls_dropped."""
    from services.model_pricing import TurnCostAccumulator

    acc = TurnCostAccumulator()
    acc.record("openai", "gpt-4o-mini", "platform", 100, 50)
    before_total = acc.total_tokens
    before_cost = acc.turn_cost_usd()

    acc.freeze()
    late_cost = acc.record("openai", "gpt-4o-mini", "platform", 999, 999)

    assert acc.total_tokens == before_total          # snapshot unchanged
    assert acc.turn_cost_usd() == before_cost
    assert acc.late_calls_dropped == 1
    assert late_cost is not None                     # the CALL was still priced
