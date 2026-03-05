# tests/test_rlm_integration.py
"""
Integration / E2E tests for the RLM system.

These tests make LIVE calls to OpenAI's API and require a valid
OPENAI_API_KEY environment variable.  They are marked with
@pytest.mark.integration so you can skip them with:
    pytest -m "not integration"

Run only these tests:
    pytest tests/test_rlm_integration.py -v
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
# 1.  REPL sandbox — numpy / scipy calculations (no OpenAI needed)
# ---------------------------------------------------------------------------

class TestREPLScientificComputing:
    """Verify that the REPL can execute numpy/scipy code for astronomy calcs."""

    def _make_env(self, ctx: str = "placeholder context"):
        from core.rlm_environment import RLMEnvironment
        return RLMEnvironment(ctx)

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
        from core.rlm import ComplexityDetector
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
        from core.rlm import ComplexityDetector
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
# 4.  Live OpenAI tests — RLM decomposition
# ---------------------------------------------------------------------------

@skip_no_key
@integration
class TestRLMDecompositionLive:
    """Test that the RLM can decompose a multi-hop query into subtasks."""

    def _make_rlm(self):
        from openai import OpenAI
        from core.rlm import RecursiveLanguageModel
        client = OpenAI(api_key=OPENAI_KEY)
        return RecursiveLanguageModel(
            client=client,
            model="gpt-4o-mini",  # cheaper for tests
            verbose=True,
            use_repl=False,  # test LLM-only path
        )

    def test_decompose_multi_hop(self):
        rlm = self._make_rlm()
        subtasks = rlm._decompose(
            "Do any M87 observations cover the CO(2-1) line?",
            context="",
        )
        assert len(subtasks) >= 2, f"Expected >=2 subtasks, got {subtasks}"
        # Should mention something about frequency or redshift
        combined = " ".join(subtasks).lower()
        assert any(kw in combined for kw in ["frequency", "redshift", "co", "m87"]), \
            f"Subtasks don't mention relevant concepts: {subtasks}"

    def test_direct_answer_works(self):
        rlm = self._make_rlm()
        answer = rlm._direct_answer(
            "What is the rest frequency of CO(2-1)?",
            context="",
        )
        assert "230" in answer, f"Expected ~230 GHz in answer: {answer}"


# ---------------------------------------------------------------------------
# 5.  Live E2E — full RLM execute pipeline
# ---------------------------------------------------------------------------

@skip_no_key
@integration
class TestRLME2ELive:
    """End-to-end test: full decompose → execute → aggregate pipeline."""

    def test_full_pipeline(self):
        from openai import OpenAI
        from core.rlm import RecursiveLanguageModel
        client = OpenAI(api_key=OPENAI_KEY)

        rlm = RecursiveLanguageModel(
            client=client,
            model="gpt-4o-mini",
            verbose=True,
            use_repl=False,
        )

        answer = rlm.execute(
            "What is the observed frequency of the CO(2-1) line "
            "for a source at redshift z=0.1?"
        )
        assert len(answer) > 20, "Answer too short"
        # Should contain a reasonable frequency (~209 GHz)
        assert any(s in answer for s in ["209", "210", "GHz"]), \
            f"Answer doesn't mention expected frequency: {answer}"


# ---------------------------------------------------------------------------
# 6.  Live E2E — REPL executor
# ---------------------------------------------------------------------------

@skip_no_key
@integration
class TestREPLExecutorLive:
    """Test the full REPL executor loop with a large context."""

    def test_repl_processes_large_context(self):
        from openai import OpenAI
        from core.rlm_environment import RLMREPLExecutor

        client = OpenAI(api_key=OPENAI_KEY)
        executor = RLMREPLExecutor(
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
        print(f"REPL answer: {answer}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
