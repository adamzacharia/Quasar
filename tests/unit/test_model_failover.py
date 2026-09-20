"""RE-A2 failover machinery — HealthMonitor wiring, refreshed fallbacks, runner switch.

Covers the three pieces that used to be dead code:
  1. HealthMonitor threshold/reset semantics and the refreshed FALLBACK_MODELS map.
  2. The ResponsesShim choke point: successes/failures recorded per provider;
     quota trips and cancellations exempt; mid-stream deaths counted.
  3. The runner's turn-start consult: OFF by default (QUASAR_MODEL_FAILOVER),
     switches with a visible status note only when every gate condition holds.
"""

import threading
from types import SimpleNamespace

import pytest

import core.health_monitor as health_monitor_module
from core.agent import QuasarAgent
from core.health_monitor import (
    FAILURE_THRESHOLD,
    FALLBACK_MODELS,
    RECOVERY_CHECK_INTERVAL,
    HealthMonitor,
    get_health_monitor,
    health_failure_exempt,
    reset_health_monitor,
)
from core.llm_client import (
    ATTACHMENT_KIND_PROVIDERS,
    LLMClient,
    ResponsesShim,
    detect_provider,
    llm_request_context,
    provider_has_key_path,
    providers_accepting_attachments,
)
from core.model_router import ModelRouter
from core.runner import _maybe_failover_model
from services.usage_quota_service import QuotaExceededError


def _drive_unhealthy(monitor, provider, error="boom"):
    for _ in range(FAILURE_THRESHOLD):
        monitor.record_failure(provider, error)


def _backdate_unhealthy(monitor, provider, seconds):
    """Age the unhealthy stamp so the recovery interval has 'elapsed'."""
    with monitor._lock:
        monitor._providers[provider]._marked_unhealthy_at -= seconds


# ---------------------------------------------------------------------------
# 1. HealthMonitor semantics + refreshed FALLBACK_MODELS
# ---------------------------------------------------------------------------


def test_failure_threshold_flips_is_healthy():
    monitor = HealthMonitor()
    for _ in range(FAILURE_THRESHOLD - 1):
        monitor.record_failure("tacc", "timeout")
        assert monitor.is_healthy("tacc")
    monitor.record_failure("tacc", "timeout")
    assert not monitor.is_healthy("tacc")


def test_success_resets_consecutive_failures():
    monitor = HealthMonitor()
    _drive_unhealthy(monitor, "tacc")
    assert not monitor.is_healthy("tacc")
    monitor.record_success("tacc")
    assert monitor.is_healthy("tacc")
    # ...and the streak restarts from zero, not from the old count.
    monitor.record_failure("tacc", "timeout")
    assert monitor.is_healthy("tacc")


def test_unknown_provider_assumed_healthy():
    assert HealthMonitor().is_healthy("never-seen")


@pytest.mark.parametrize(
    "provider,primary,expected",
    [
        ("openai", "gpt-5.4-mini", "deepseek-v4-pro"),
        ("anthropic", "claude-sonnet-5", "deepseek-v4-pro"),
        ("google", "gemini-2.5-flash", "deepseek-v4-pro"),
        ("tacc", "gpt-oss-120b", "deepseek-v4-flash"),
        ("deepseek", "deepseek-v4-pro", "gpt-oss-120b"),
        ("local", "local/llama3", "gpt-oss-120b"),
    ],
)
def test_get_fallback_model_returns_refreshed_mappings(
    monkeypatch, provider, primary, expected
):
    monkeypatch.delenv(f"QUASAR_{provider.upper()}_FALLBACK_MODEL", raising=False)
    monitor = HealthMonitor()
    # Healthy → no fallback needed.
    assert monitor.get_fallback_model(primary, provider) is None
    _drive_unhealthy(monitor, provider)
    assert monitor.get_fallback_model(primary, provider) == expected


def test_every_fallback_is_a_different_provider():
    for provider, fallback in FALLBACK_MODELS.items():
        assert detect_provider(fallback) != provider, (
            f"{provider} falls back to {fallback}, which is the SAME provider"
        )


def test_get_fallback_model_env_override(monkeypatch):
    monkeypatch.setenv("QUASAR_TACC_FALLBACK_MODEL", "gpt-4.1")
    monitor = HealthMonitor()
    _drive_unhealthy(monitor, "tacc")
    assert monitor.get_fallback_model("gpt-oss-120b", "tacc") == "gpt-4.1"
    monkeypatch.setenv("QUASAR_TACC_FALLBACK_MODEL", "none")
    assert monitor.get_fallback_model("gpt-oss-120b", "tacc") is None


def test_get_health_monitor_is_process_shared():
    reset_health_monitor()
    try:
        a = get_health_monitor()
        b = get_health_monitor()
        assert a is b
    finally:
        reset_health_monitor()


def test_recovery_probe_single_flight_and_failed_probe_restamps():
    """CX-04 + CX-05: one probe at a time; a failed probe restarts the clock."""
    monitor = HealthMonitor()
    _drive_unhealthy(monitor, "tacc")
    assert not monitor.is_healthy("tacc")

    _backdate_unhealthy(monitor, "tacc", RECOVERY_CHECK_INTERVAL + 1)
    assert monitor.is_healthy("tacc")        # first caller claims THE probe
    assert not monitor.is_healthy("tacc")    # CX-05: concurrent turn keeps failing over

    monitor.record_failure("tacc", "still down")  # the probe's call failed
    # CX-04: fresh unhealthy stamp — not probe-eligible again until a full
    # interval passes (previously the stale stamp made it permanently healthy).
    assert not monitor.is_healthy("tacc")

    _backdate_unhealthy(monitor, "tacc", RECOVERY_CHECK_INTERVAL + 1)
    assert monitor.is_healthy("tacc")        # next interval → next probe
    monitor.record_success("tacc")           # probe wins
    assert monitor.is_healthy("tacc")
    assert monitor.get_status()["tacc"]["healthy"]


def test_get_status_is_pure_and_never_claims_the_probe():
    monitor = HealthMonitor()
    _drive_unhealthy(monitor, "tacc")
    _backdate_unhealthy(monitor, "tacc", RECOVERY_CHECK_INTERVAL + 1)
    # Reporting shows degraded and must NOT consume the probe slot...
    assert monitor.get_status()["tacc"]["healthy"] is False
    # ...so the routing check can still claim it.
    assert monitor.is_healthy("tacc")


def test_concurrent_recording_conserves_counts_and_never_raises():
    """CX-01/CX-02/CX-15: threads hammering the shared monitor lose nothing."""
    monitor = HealthMonitor()
    errors = []
    ops = 200
    threads_per_provider = 2
    providers = 3

    def worker(i):
        provider = f"p{i % providers}"
        try:
            for _ in range(ops):
                monitor.record_failure(provider, "boom")
                monitor.record_success(provider)
                monitor.is_healthy(provider)
                monitor.get_status()
        except Exception as exc:  # pragma: no cover - the assertion target
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(i,))
        for i in range(providers * threads_per_provider)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    status = monitor.get_status()
    for i in range(providers):
        entry = status[f"p{i}"]
        assert entry["total_failures"] == ops * threads_per_provider
        assert entry["total_successes"] == ops * threads_per_provider


def test_concurrent_failures_flip_unhealthy_without_lost_updates():
    """CX-15: failure-only contention actually drives the provider unhealthy
    and no increment is lost (consecutive == total == threads * ops)."""
    monitor = HealthMonitor()
    errors = []
    ops = 100
    nthreads = 8
    barrier = threading.Barrier(nthreads)

    def worker():
        try:
            barrier.wait()
            for _ in range(ops):
                monitor.record_failure("tacc", "boom")
                monitor.get_status()
        except Exception as exc:  # pragma: no cover - the assertion target
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(nthreads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    status = monitor.get_status()["tacc"]
    assert status["total_failures"] == ops * nthreads
    assert status["consecutive_failures"] == ops * nthreads
    assert status["healthy"] is False


def test_concurrent_recovery_probe_claims_exactly_one():
    """CX-15: N threads racing into the recovery window get exactly ONE probe
    claim; the losers keep failing over, and a failed probe re-arms cleanly."""
    monitor = HealthMonitor()
    _drive_unhealthy(monitor, "tacc")
    _backdate_unhealthy(monitor, "tacc", RECOVERY_CHECK_INTERVAL + 1)

    nthreads = 16
    barrier = threading.Barrier(nthreads)
    results = []
    results_lock = threading.Lock()
    errors = []

    def worker():
        try:
            barrier.wait()
            verdict = monitor.is_healthy("tacc")
            with results_lock:
                results.append(verdict)
        except Exception as exc:  # pragma: no cover - the assertion target
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(nthreads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert results.count(True) == 1, f"expected exactly one probe claim, got {results}"
    assert results.count(False) == nthreads - 1

    # The winner's call fails → fresh unhealthy clock, probe slot released:
    monitor.record_failure("tacc", "still down")
    assert not monitor.is_healthy("tacc")
    # ...and only after another full interval is the next probe claimable.
    _backdate_unhealthy(monitor, "tacc", RECOVERY_CHECK_INTERVAL + 1)
    assert monitor.is_healthy("tacc")


def test_env_override_whitespace_is_stripped(monkeypatch):
    """CX-08: the returned override is trimmed, not just the emptiness check."""
    monkeypatch.setenv("QUASAR_TACC_FALLBACK_MODEL", "  gpt-4.1  ")
    monitor = HealthMonitor()
    _drive_unhealthy(monitor, "tacc")
    assert monitor.get_fallback_model("gpt-oss-120b", "tacc") == "gpt-4.1"


# ---------------------------------------------------------------------------
# 2. Choke point: ResponsesShim health recording + exemptions
# ---------------------------------------------------------------------------


class _NamedQuota(Exception):
    """Same NAME as the services class — the exemption matches by MRO name."""


_NamedQuota.__name__ = "QuotaExceededError"


def test_health_failure_exempt_classifier():
    assert health_failure_exempt(QuotaExceededError("weekly cap reached"))
    assert health_failure_exempt(_NamedQuota("mid-turn trip"))
    insufficient = RuntimeError("Error code: 429 - insufficient_quota")
    assert health_failure_exempt(insufficient)
    coded = RuntimeError("429")
    coded.code = "insufficient_quota"
    assert health_failure_exempt(coded)
    assert health_failure_exempt(ValueError("OPENAI_API_KEY is required"))
    assert health_failure_exempt(TypeError("bad kwarg"))
    assert health_failure_exempt(KeyError("model"))
    assert health_failure_exempt(GeneratorExit())
    # Real provider ill-health IS counted.
    assert not health_failure_exempt(RuntimeError("connection reset by peer"))
    assert not health_failure_exempt(TimeoutError("read timed out"))


@pytest.fixture()
def fresh_monitor(monkeypatch):
    monitor = HealthMonitor()
    monkeypatch.setattr(health_monitor_module, "_GLOBAL_MONITOR", monitor)
    return monitor


def _shim():
    return ResponsesShim(SimpleNamespace())


def test_shim_records_success_and_failure(fresh_monitor):
    shim = _shim()
    shim._record_provider_health("tacc", "gpt-oss-120b")
    status = fresh_monitor.get_status()["tacc"]
    assert status["total_successes"] == 1 and status["total_failures"] == 0

    shim._record_provider_health(
        "tacc", "gpt-oss-120b", error=RuntimeError("connection reset")
    )
    status = fresh_monitor.get_status()["tacc"]
    assert status["total_failures"] == 1
    assert "gpt-oss-120b" in status["last_error"]


def test_shim_does_not_record_quota_errors_as_failures(fresh_monitor):
    shim = _shim()
    shim._record_provider_health(
        "openai", "gpt-4.1", error=QuotaExceededError("headroom exhausted")
    )
    shim._record_provider_health(
        "openai", "gpt-4.1", error=RuntimeError("429 insufficient_quota billing")
    )
    # Neither quota flavor created a failure entry.
    assert fresh_monitor.get_status().get("openai") is None
    assert fresh_monitor.is_healthy("openai")


def test_stream_wrapper_counts_mid_stream_death(fresh_monitor):
    shim = _shim()

    def dying():
        yield SimpleNamespace(type="response.output_text.delta", delta="hi")
        raise RuntimeError("connection reset by peer")

    wrapped = shim._wrap_usage_stream(dying(), "tacc", "gpt-oss-120b")
    with pytest.raises(RuntimeError):
        list(wrapped)
    assert fresh_monitor.get_status()["tacc"]["total_failures"] == 1


def test_stream_wrapper_counts_clean_exhaustion_as_success(fresh_monitor):
    shim = _shim()
    events = [SimpleNamespace(type="response.output_text.delta", delta="hi")]
    assert list(shim._wrap_usage_stream(iter(events), "tacc", "gpt-oss-120b")) == events
    status = fresh_monitor.get_status()["tacc"]
    assert status["total_successes"] == 1 and status["total_failures"] == 0


def test_stream_wrapper_ignores_consumer_abandonment(fresh_monitor):
    """User cancellation (close() → GeneratorExit at the yield) is neither a
    success nor a failure — the provider was fine when we walked away."""
    shim = _shim()
    events = [
        SimpleNamespace(type="a", delta="1"),
        SimpleNamespace(type="b", delta="2"),
    ]
    wrapped = shim._wrap_usage_stream(iter(events), "tacc", "gpt-oss-120b")
    next(wrapped)
    wrapped.close()
    assert fresh_monitor.get_status() == {}


def test_stream_wrapper_ignores_consumer_throw(fresh_monitor):
    """CX-17: the CONSUMER throwing into the generator surfaces at the yield,
    not at next(it) — it must count as neither success nor provider failure."""
    shim = _shim()
    events = [
        SimpleNamespace(type="a", delta="1"),
        SimpleNamespace(type="b", delta="2"),
    ]
    wrapped = shim._wrap_usage_stream(iter(events), "tacc", "gpt-oss-120b")
    next(wrapped)
    with pytest.raises(RuntimeError, match="consumer-side abort"):
        wrapped.throw(RuntimeError("consumer-side abort"))
    assert fresh_monitor.get_status() == {}


def test_byok_failure_is_isolated_from_platform_health(fresh_monitor):
    """CX-03/CX-16: a failure on a user's own key never degrades the provider
    for platform-key requests; a BYOK success still counts as evidence-of-up."""
    client = LLMClient(model="gpt-oss-120b")
    shim = client.responses
    err = RuntimeError("connection reset by peer")

    with llm_request_context(
        provider_api_keys={"tacc": "user-key"},
        key_source_by_provider={"tacc": "byok"},
    ):
        for _ in range(FAILURE_THRESHOLD + 1):
            shim._record_provider_health("tacc", "gpt-oss-120b", error=err)
        shim._record_provider_health("tacc", "gpt-oss-120b")  # BYOK success

    # No BYOK failure reached the shared monitor; the success did.
    assert fresh_monitor.is_healthy("tacc")
    status = fresh_monitor.get_status()["tacc"]
    assert status["total_failures"] == 0
    assert status["total_successes"] == 1

    # A platform-key failure afterwards still counts normally.
    shim._record_provider_health("tacc", "gpt-oss-120b", error=err)
    assert fresh_monitor.get_status()["tacc"]["total_failures"] == 1


def test_shim_create_end_to_end_records_health(fresh_monitor):
    """CX-14: success/failure recording through the REAL ResponsesShim.create
    wiring (transport stubbed at the provider-dispatch boundary)."""
    client = LLMClient(model="gpt-oss-120b")
    shim = client.responses

    ok = SimpleNamespace(usage=None, output_text="ok")
    shim._call_tacc = lambda kwargs, attachments=None: ok  # instance attr shadows
    assert shim.create(model="gpt-oss-120b", input="hi") is ok
    status = fresh_monitor.get_status()["tacc"]
    assert status["total_successes"] == 1 and status["total_failures"] == 0

    def boom(kwargs, attachments=None):
        raise RuntimeError("connection reset by peer")

    shim._call_tacc = boom
    with pytest.raises(RuntimeError):
        shim.create(model="gpt-oss-120b", input="hi")
    status = fresh_monitor.get_status()["tacc"]
    assert status["total_failures"] == 1 and status["total_successes"] == 1


def test_shim_create_streaming_end_to_end_records_success(fresh_monitor):
    client = LLMClient(model="gpt-oss-120b")
    shim = client.responses
    events = [
        SimpleNamespace(type="response.output_text.delta", delta="hi", response=None)
    ]
    shim._stream_tacc = lambda kwargs, attachments=None: iter(events)

    out = list(shim.create(model="gpt-oss-120b", input="hi", stream=True))

    assert out == events
    status = fresh_monitor.get_status()["tacc"]
    assert status["total_successes"] == 1 and status["total_failures"] == 0


# ---------------------------------------------------------------------------
# 3. Runner turn-start consult (pattern: test_response_api_error_recovery)
# ---------------------------------------------------------------------------


class _FakeResponses:
    """Mimics LLMClient.responses — immediately streams a text answer."""

    def __init__(self, answer="fine answer"):
        self.answer = answer
        self.create_calls = []
        self.cleared = []

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return iter(
            [
                SimpleNamespace(
                    type="response.created",
                    response=SimpleNamespace(id=f"resp_{len(self.create_calls)}"),
                ),
                SimpleNamespace(type="response.output_text.delta", delta=self.answer),
            ]
        )

    def clear_history(self, response_id=None):
        self.cleared.append(response_id)


def _make_agent(fake_responses, monitor):
    agent = object.__new__(QuasarAgent)
    agent._tls = threading.local()
    agent._conv_response_ids = {}
    agent._conv_run_tokens = {}
    agent._conv_ids_lock = threading.Lock()
    agent.config = SimpleNamespace(
        model="gpt-oss-120b", temperature=0.2, max_tokens=2048, verbose=False
    )
    agent.system_prompt = "system prompt"
    agent.ads_client = None
    agent.long_term_memory = None
    agent.client = SimpleNamespace(responses=fake_responses)
    agent.health_monitor = monitor
    # Stub preamble services not under test
    agent._prune_session_if_needed = lambda *a, **k: None
    agent._build_tools_for_responses_api = lambda: []
    return agent


DEGRADED_NOTE = "Model gpt-oss-120b is degraded — using deepseek-v4-flash for this turn"


def _run_turn(agent, conv):
    statuses = []
    result = agent.stream_response_api(
        "hello there",
        conversation_id=conv,
        on_status=lambda msg, state: statuses.append((msg, state)),
    )
    return result, statuses


def test_flag_off_no_switch_even_when_unhealthy(monkeypatch):
    monkeypatch.delenv("QUASAR_MODEL_FAILOVER", raising=False)  # default OFF
    monkeypatch.delenv("QUASAR_TACC_FALLBACK_MODEL", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monitor = HealthMonitor()
    _drive_unhealthy(monitor, "tacc")
    fake = _FakeResponses()
    agent = _make_agent(fake, monitor)

    result, statuses = _run_turn(agent, "conv-off")

    assert result == fake.answer
    assert fake.create_calls[0]["model"] == "gpt-oss-120b"
    assert not any("degraded" in msg for msg, _ in statuses)


def test_flag_on_unhealthy_switches_with_status_note(monkeypatch):
    monkeypatch.setenv("QUASAR_MODEL_FAILOVER", "1")
    monkeypatch.delenv("QUASAR_TACC_FALLBACK_MODEL", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monitor = HealthMonitor()
    _drive_unhealthy(monitor, "tacc")
    fake = _FakeResponses()
    agent = _make_agent(fake, monitor)

    result, statuses = _run_turn(agent, "conv-on")

    assert result == fake.answer
    assert fake.create_calls[0]["model"] == "deepseek-v4-flash"
    assert (DEGRADED_NOTE, "completed") in statuses


def test_flag_on_but_no_fallback_key_path_keeps_primary(monkeypatch):
    """BYOK/user-pinned protection: without a usable key path on the fallback's
    provider, keep the primary and let the normal retry/error path speak."""
    monkeypatch.setenv("QUASAR_MODEL_FAILOVER", "1")
    monkeypatch.delenv("QUASAR_TACC_FALLBACK_MODEL", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monitor = HealthMonitor()
    _drive_unhealthy(monitor, "tacc")
    fake = _FakeResponses()
    agent = _make_agent(fake, monitor)

    result, statuses = _run_turn(agent, "conv-nokey")

    assert result == fake.answer
    assert fake.create_calls[0]["model"] == "gpt-oss-120b"
    assert not any("degraded" in msg for msg, _ in statuses)


def test_flag_on_healthy_primary_untouched(monkeypatch):
    monkeypatch.setenv("QUASAR_MODEL_FAILOVER", "1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    fake = _FakeResponses()
    agent = _make_agent(fake, HealthMonitor())

    result, statuses = _run_turn(agent, "conv-healthy")

    assert result == fake.answer
    assert fake.create_calls[0]["model"] == "gpt-oss-120b"
    assert not any("degraded" in msg for msg, _ in statuses)


def test_flag_on_unhealthy_fallback_provider_keeps_primary(monkeypatch):
    monkeypatch.setenv("QUASAR_MODEL_FAILOVER", "1")
    monkeypatch.delenv("QUASAR_TACC_FALLBACK_MODEL", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monitor = HealthMonitor()
    _drive_unhealthy(monitor, "tacc")
    _drive_unhealthy(monitor, "deepseek")  # fallback target is down too
    fake = _FakeResponses()
    agent = _make_agent(fake, monitor)

    result, statuses = _run_turn(agent, "conv-bothdown")

    assert result == fake.answer
    assert fake.create_calls[0]["model"] == "gpt-oss-120b"
    assert not any("degraded" in msg for msg, _ in statuses)


def test_provider_has_key_path(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert not provider_has_key_path("deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    assert provider_has_key_path("deepseek")
    # Client-scoped BYOK key counts even without env.
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    client = SimpleNamespace(provider_api_keys={"deepseek": "byok"})
    assert provider_has_key_path("deepseek", client)
    # Unknown providers never claim a key path.
    assert not provider_has_key_path("not-a-provider")


def test_env_override_same_provider_is_rejected(monkeypatch):
    """CX-07: an env override pointing at ANOTHER model on the SAME unhealthy
    provider gets no bypass — the turn keeps the primary."""
    monkeypatch.setenv("QUASAR_MODEL_FAILOVER", "1")
    monkeypatch.setenv("QUASAR_TACC_FALLBACK_MODEL", "Qwen3-32B")  # also tacc
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monitor = HealthMonitor()
    _drive_unhealthy(monitor, "tacc")
    fake = _FakeResponses()
    agent = _make_agent(fake, monitor)

    result, statuses = _run_turn(agent, "conv-sameprov")

    assert result == fake.answer
    assert fake.create_calls[0]["model"] == "gpt-oss-120b"
    assert not any("degraded" in msg for msg, _ in statuses)


def test_attachment_turn_skips_attachment_dropping_fallback(monkeypatch):
    """CX-09: tacc→deepseek failover would silently drop uploads (DeepSeek
    strips image attachments) — an attachment-carrying turn keeps the primary."""
    monkeypatch.setenv("QUASAR_MODEL_FAILOVER", "1")
    monkeypatch.delenv("QUASAR_TACC_FALLBACK_MODEL", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    assert "deepseek" not in providers_accepting_attachments(
        [{"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}}]
    )
    monitor = HealthMonitor()
    _drive_unhealthy(monitor, "tacc")
    fake = _FakeResponses()
    agent = _make_agent(fake, monitor)
    statuses = []

    result = agent.stream_response_api(
        "hello there",
        conversation_id="conv-attach",
        on_status=lambda msg, state: statuses.append((msg, state)),
        attachments=[{"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}}],
    )

    assert result == fake.answer
    assert fake.create_calls[0]["model"] == "gpt-oss-120b"
    assert not any("degraded" in msg for msg, _ in statuses)


def test_model_router_health_reroute_gated_by_flag(monkeypatch):
    """CX-06: live health data must not reroute Conductor helper models unless
    QUASAR_MODEL_FAILOVER is enabled."""
    monkeypatch.delenv("QUASAR_FAST_MODEL", raising=False)
    monkeypatch.delenv("QUASAR_DEEPSEEK_FALLBACK_MODEL", raising=False)
    monkeypatch.setenv("TACC_API_KEY", "test-key")  # fallback key path
    monitor = HealthMonitor()
    _drive_unhealthy(monitor, "deepseek")
    router = ModelRouter(default_model="gpt-oss-120b")
    router.health_monitor = monitor

    monkeypatch.delenv("QUASAR_MODEL_FAILOVER", raising=False)  # default OFF
    assert router._apply_health_check("deepseek-v4-flash") == "deepseek-v4-flash"
    monkeypatch.setenv("QUASAR_MODEL_FAILOVER", "0")
    assert router._apply_health_check("deepseek-v4-flash") == "deepseek-v4-flash"

    monkeypatch.setenv("QUASAR_MODEL_FAILOVER", "1")
    assert router._apply_health_check("deepseek-v4-flash") == "gpt-oss-120b"


def test_model_router_fallback_passes_eligibility_gates(monkeypatch):
    """CX-06 round 2: the router's mapped fallback passes the SAME gates as
    the runner — key path and fallback-provider health — not just the flag."""
    monkeypatch.setenv("QUASAR_MODEL_FAILOVER", "1")
    monkeypatch.delenv("QUASAR_FAST_MODEL", raising=False)
    monkeypatch.delenv("QUASAR_DEEPSEEK_FALLBACK_MODEL", raising=False)
    monitor = HealthMonitor()
    _drive_unhealthy(monitor, "deepseek")
    router = ModelRouter(default_model="gpt-oss-120b")
    router.health_monitor = monitor

    # No key path for the tacc fallback → keep the primary (trading one error
    # for another helps nobody).
    for var in ("TACC_API_KEY", "TEJAS_API_KEY", "TEXAS_AI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert router._apply_health_check("deepseek-v4-flash") == "deepseek-v4-flash"

    # Key present but the fallback provider is itself unhealthy → keep primary.
    monkeypatch.setenv("TACC_API_KEY", "test-key")
    _drive_unhealthy(monitor, "tacc")
    assert router._apply_health_check("deepseek-v4-flash") == "deepseek-v4-flash"

    # Fallback healthy again + key path → reroute engages.
    monitor.record_success("tacc")
    assert router._apply_health_check("deepseek-v4-flash") == "gpt-oss-120b"

    # A same-provider env override is rejected, exactly like the runner path.
    monkeypatch.setenv("QUASAR_DEEPSEEK_FALLBACK_MODEL", "deepseek-v4-pro")
    assert router._apply_health_check("deepseek-v4-flash") == "deepseek-v4-flash"


def test_attachment_kind_provider_mapping():
    """CX-21: acceptance sets mirror the builder kind checks exactly."""
    assert ATTACHMENT_KIND_PROVIDERS["openai_input_file"] == frozenset({"openai"})
    assert ATTACHMENT_KIND_PROVIDERS["anthropic_document_file"] == frozenset({"anthropic"})
    assert ATTACHMENT_KIND_PROVIDERS["gemini_file"] == frozenset({"google"})
    assert ATTACHMENT_KIND_PROVIDERS["image_url"] == frozenset({"openai", "tacc"})

    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}}
    assert providers_accepting_attachments([image]) == frozenset({"openai", "tacc"})
    # Mixed kinds intersect: an OpenAI file ref + an image → only openai.
    mixed = [{"kind": "openai_input_file", "file_id": "f"}, image]
    assert providers_accepting_attachments(mixed) == frozenset({"openai"})
    # Unknown kinds are conservative: nobody is known to forward them.
    assert providers_accepting_attachments([{"weird": True}]) == frozenset()


def test_attachment_kind_gating_cross_vs_same_kind(monkeypatch):
    """CX-21: a document ref of one provider's kind blocks failover to a
    DIFFERENT 'capable' provider (its builder silently drops foreign kinds),
    while a ref of the fallback's OWN kind — and a kindless turn — switch."""
    monkeypatch.setenv("QUASAR_MODEL_FAILOVER", "1")
    monkeypatch.setenv("QUASAR_OPENAI_FALLBACK_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monitor = HealthMonitor()
    _drive_unhealthy(monitor, "openai")
    agent = SimpleNamespace(health_monitor=monitor, client=None)

    # Cross-kind: OpenAI file ref is invisible to the Anthropic builder → skip.
    openai_doc = {"kind": "openai_input_file", "file_id": "f_1"}
    assert _maybe_failover_model(agent, "gpt-4.1", None, [openai_doc]) == "gpt-4.1"

    # Same-kind: an anthropic_document_file IS accepted by the fallback → switch.
    anthropic_doc = {"kind": "anthropic_document_file", "file_id": "f_2"}
    assert (
        _maybe_failover_model(agent, "gpt-4.1", None, [anthropic_doc])
        == "claude-sonnet-5"
    )

    # Kindless (no attachments): unaffected by the attachment gate → switch.
    assert _maybe_failover_model(agent, "gpt-4.1", None, None) == "claude-sonnet-5"


def test_rejected_attachment_failover_does_not_claim_fallback_probe(monkeypatch):
    """CX-20: a switch rejected by a cheap gate (attachments here) must not
    have consumed the fallback provider's single-flight recovery probe."""
    monkeypatch.setenv("QUASAR_MODEL_FAILOVER", "1")
    monkeypatch.delenv("QUASAR_TACC_FALLBACK_MODEL", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monitor = HealthMonitor()
    _drive_unhealthy(monitor, "tacc")
    _drive_unhealthy(monitor, "deepseek")
    # The deepseek fallback is probe-ELIGIBLE (recovery window open).
    _backdate_unhealthy(monitor, "deepseek", RECOVERY_CHECK_INTERVAL + 1)
    agent = SimpleNamespace(health_monitor=monitor, client=None)

    # tacc→deepseek failover is rejected by the attachment gate (DeepSeek
    # forwards no attachment kind) BEFORE any deepseek health check runs.
    doc = {"kind": "openai_input_file", "file_id": "f_1"}
    out = _maybe_failover_model(agent, "gpt-oss-120b", None, [doc])
    assert out == "gpt-oss-120b"

    # deepseek's probe was NOT consumed by the rejected switch: the next
    # health check can still claim it.
    assert monitor.is_healthy("deepseek")
