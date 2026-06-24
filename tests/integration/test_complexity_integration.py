# tests/integration/test_complexity_integration.py
"""
Integration / E2E tests for the Complexity Detector and Sandbox.

Tests marked with @pytest.mark.integration make LIVE calls to OpenAI's API
and require a valid OPENAI_API_KEY environment variable.
Skip them with: pytest -m "not integration"

Run only these tests:
    pytest tests/integration/test_complexity_integration.py -v
"""

import os
import pytest
from unittest.mock import MagicMock

# Load .env if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
HAS_KEY = bool(OPENAI_KEY)

skip_no_key = pytest.mark.skipif(
    not HAS_KEY, reason="OPENAI_API_KEY not set — skipping live tests"
)
integration = pytest.mark.integration


# ---------------------------------------------------------------------------
# 1.  Sandbox — numpy / scipy calculations (no OpenAI needed)
# ---------------------------------------------------------------------------

class TestSandboxScientificComputing:
    """Verify that the Sandbox can execute numpy/scipy code for astronomy calcs."""

    def _make_env(self, ctx: str = "placeholder context"):
        from core.sandbox import SandboxEnvironment
        return SandboxEnvironment(ctx)

    def test_numpy_frequency_conversion(self):
        """Convert CO(2-1) rest freq to observed freq at M87 redshift."""
        env = self._make_env()
        result = env.execute(
            "import numpy as np\n"
            "rest_freq = 230.538   # GHz, CO(2-1)\n"
            "z = 0.00428           # M87 redshift\n"
            "obs_freq = rest_freq / (1 + z)\n"
            "print(f'Observed freq: {obs_freq:.3f} GHz')\n"
            "results.append(obs_freq)\n"
        )
        assert result.success, f"Failed: {result.error}"
        assert "229" in result.stdout  # ~229.55 GHz
        assert len(env.namespace["results"]) == 1

    def test_numpy_sensitivity_calc(self):
        """Radiometer equation: sigma = SEFD / sqrt(2 * BW * t_int * N*(N-1)/2)."""
        env = self._make_env()
        result = env.execute(
            "import numpy as np\n"
            "sefd = 70.0       # Jy, Band 6 ALMA\n"
            "bw = 7.5e9        # Hz (7.5 GHz)\n"
            "t_int = 3600.0    # seconds\n"
            "n_ant = 50\n"
            "n_baselines = n_ant * (n_ant - 1) / 2\n"
            "sigma = sefd / np.sqrt(2 * bw * t_int * n_baselines)\n"
            "sigma_uJy = sigma * 1e6\n"
            "print(f'Sensitivity: {sigma_uJy:.2f} uJy/beam')\n"
            "results.append(sigma_uJy)\n"
        )
        assert result.success, f"Failed: {result.error}"
        # Should be a few µJy — sanity check it's a reasonable number
        sigma = env.namespace["results"][0]
        assert 0.1 < sigma < 100.0, f"Unreasonable sensitivity: {sigma} uJy"

    def test_math_import(self):
        """Basic math calculations work."""
        env = self._make_env()
        result = env.execute(
            "import math\n"
            "wavelength_m = 3e8 / (230.538e9)  # speed of light / freq\n"
            "wavelength_mm = wavelength_m * 1000\n"
            "print(f'Wavelength: {wavelength_mm:.3f} mm')\n"
        )
        assert result.success, f"Failed: {result.error}"
        assert "1.3" in result.stdout  # ~1.3 mm

    def test_results_accumulate_across_steps(self):
        """Verify persistent namespace across execution steps."""
        env = self._make_env()
        env.execute("rest_freq = 230.538")
        env.execute("z = 0.00428")
        result = env.execute(
            "obs_freq = rest_freq / (1 + z)\n"
            "results.append(f'{obs_freq:.3f}')\n"
            "print(results)\n"
        )
        assert result.success
        assert "229" in result.stdout

    def test_blocked_module_still_blocked(self):
        """Ensure the expanded allowlist didn't break security."""
        env = self._make_env()
        result = env.execute("import subprocess")
        assert result.success is False


# ---------------------------------------------------------------------------
# 2.  ComplexityDetector — heuristic (no OpenAI needed)
# ---------------------------------------------------------------------------

class TestComplexityDetectorHeuristic:
    """Heuristic scoring should correctly classify queries."""

    def _score(self, query: str) -> float:
        from core.complexity import ComplexityDetector
        return ComplexityDetector._heuristic_score(query)

    def test_simple_search(self):
        assert self._score("Find ALMA data for M87") < 0.4

    def test_simple_question(self):
        assert self._score("What is Band 6?") < 0.4

    def test_multi_hop_line_coverage(self):
        s = self._score(
            "Do any M87 observations cover the CO(2-1) line at the "
            "correct redshift and overlap in frequency?"
        )
        assert s >= 0.55

    def test_multi_hop_comparison(self):
        s = self._score(
            "Compare sensitivity across multiple bands and correlate "
            "with observed frequency for redshift 0.5"
        )
        assert s >= 0.8


# ---------------------------------------------------------------------------
# 3.  Live OpenAI tests — ComplexityDetector LLM probe
# ---------------------------------------------------------------------------

@skip_no_key
@integration
class TestComplexityDetectorLive:
    """Test LLM-based complexity assessment with real API calls."""

    def _make_detector(self):
        from openai import OpenAI
        from core.complexity import ComplexityDetector
        client = OpenAI(api_key=OPENAI_KEY)
        return ComplexityDetector(client, model="gpt-4o-mini")

    def test_llm_classifies_simple(self):
        det = self._make_detector()
        result = det.assess("What bands does ALMA have?")
        assert result.score < 0.6, f"Expected simple, got score={result.score}"

    def test_llm_classifies_complex(self):
        det = self._make_detector()
        result = det.assess(
            "Check if any ALMA observations of NGC 1068 cover the HCN(4-3) "
            "line at the galaxy's redshift, and estimate the expected line "
            "sensitivity based on the observation parameters"
        )
        assert result.is_complex, f"Expected complex, got score={result.score}"
        assert len(result.suggested_subtasks) > 0


# ---------------------------------------------------------------------------
# 4.  Live E2E — Sandbox REPL executor
# ---------------------------------------------------------------------------

@skip_no_key
@integration
class TestSandboxExecutorLive:
    """Test the full Sandbox executor loop with a large context."""

    def test_sandbox_processes_large_context(self):
        from openai import OpenAI
        from core.sandbox import SandboxExecutor

        client = OpenAI(api_key=OPENAI_KEY)
        executor = SandboxExecutor(
            client=client,
            model="gpt-4o-mini",
            sub_model="gpt-4o-mini",
            verbose=True,
        )

        # Create a context with multiple observation records
        records = []
        for i in range(50):
            band = [3, 6, 7][i % 3]
            freq = {3: 100.0, 6: 230.0, 7: 345.0}[band]
            records.append(
                f"Observation {i+1}: Target=Source_{i+1}, Band={band}, "
                f"Freq={freq + i*0.1:.1f} GHz, Resolution={0.02 + i*0.001:.3f} arcsec"
            )
        context = "\n".join(records)

        answer = executor.run(
            query="How many Band 6 observations are in this list? "
                  "What is the average frequency of Band 6 observations?",
            context=context,
            max_iterations=6,
        )
        assert len(answer) > 10, f"Answer too short: {answer}"
        # There should be ~17 Band 6 observations (every 3rd of 50)
        # We just check it returns something reasonable
        print(f"Sandbox answer: {answer}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
