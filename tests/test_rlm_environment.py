# tests/test_rlm_environment.py
"""
Unit tests for the RLM REPL Environment.

Tests sandbox safety, context helpers, and code execution.
"""

import pytest
from core.rlm_environment import RLMEnvironment, ExecutionResult, _is_safe


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

SAMPLE_CONTEXT = """\
ALMA Observation Summary
========================
Target: M87
RA: 187.7059
Dec: 12.3911
Band: 6
Frequency: 230.5 GHz
Resolution: 0.025 arcsec
Integration time: 3600 seconds
PI: Event Horizon Telescope Collaboration

Target: Sgr A*
RA: 266.4168
Dec: -29.0078
Band: 7
Frequency: 345.8 GHz
Resolution: 0.018 arcsec
Integration time: 7200 seconds
PI: Event Horizon Telescope Collaboration

Target: HL Tau
RA: 67.4620
Dec: 18.2328
Band: 3
Frequency: 100.0 GHz
Resolution: 0.035 arcsec
Integration time: 1800 seconds
PI: ALMA Partnership
"""


# ---------------------------------------------------------------------------
# Safety checks
# ---------------------------------------------------------------------------

class TestSafety:
    def test_safe_code(self):
        ok, msg = _is_safe("x = len(context)")
        assert ok is True

    def test_blocks_os_import(self):
        ok, msg = _is_safe("import os; os.system('rm -rf /')")
        assert ok is False
        assert "Blocked" in msg

    def test_blocks_subprocess(self):
        ok, msg = _is_safe("import subprocess")
        assert ok is False

    def test_blocks_open(self):
        ok, msg = _is_safe("f = open('/etc/passwd')")
        assert ok is False

    def test_blocks_dunder(self):
        ok, msg = _is_safe("x.__class__.__bases__")
        assert ok is False

    def test_allows_safe_imports(self):
        ok, msg = _is_safe("import re")
        assert ok is True

    def test_allows_math(self):
        ok, msg = _is_safe("import math; x = math.pi")
        assert ok is True


# ---------------------------------------------------------------------------
# Environment execution
# ---------------------------------------------------------------------------

class TestRLMEnvironment:
    def setup_method(self):
        self.env = RLMEnvironment(SAMPLE_CONTEXT)

    def test_context_loaded(self):
        assert self.env.namespace["context"] == SAMPLE_CONTEXT
        assert self.env.namespace["num_lines"] == len(SAMPLE_CONTEXT.splitlines())
        assert self.env.namespace["num_chars"] == len(SAMPLE_CONTEXT)

    def test_simple_print(self):
        result = self.env.execute('print("hello")')
        assert result.success is True
        assert "hello" in result.stdout

    def test_variable_persistence(self):
        self.env.execute("x = 42")
        result = self.env.execute("print(x)")
        assert result.success is True
        assert "42" in result.stdout

    def test_search_function(self):
        result = self.env.execute('matches = search("M87")\nprint(matches)')
        assert result.success is True
        assert "M87" in result.stdout

    def test_head_function(self):
        result = self.env.execute("print(head(5))")
        assert result.success is True
        assert "ALMA" in result.stdout

    def test_tail_function(self):
        result = self.env.execute("print(tail(3))")
        assert result.success is True

    def test_slice_lines(self):
        result = self.env.execute("print(slice_lines(1, 3))")
        assert result.success is True
        assert "ALMA" in result.stdout

    def test_count_matches(self):
        result = self.env.execute('c = count_matches("Target:")\nprint(c)')
        assert result.success is True
        assert "3" in result.stdout

    def test_unsafe_code_blocked(self):
        result = self.env.execute("import os; os.system('ls')")
        assert result.success is False
        assert "Blocked" in result.error

    def test_regex_import_allowed(self):
        result = self.env.execute('import re\nm = re.findall(r"Band: (\\d+)", context)\nprint(m)')
        assert result.success is True
        assert "6" in result.stdout

    def test_trajectory_tracking(self):
        self.env.execute("x = 1")
        self.env.execute("y = 2")
        assert len(self.env.trajectory) == 2
        assert all(t["success"] for t in self.env.trajectory)

    def test_state_summary(self):
        self.env.execute("targets_found = 3")
        summary = self.env.get_state_summary()
        assert "chars" in summary
        assert "lines" in summary

    def test_error_handling(self):
        result = self.env.execute("1 / 0")
        assert result.success is False
        assert "ZeroDivisionError" in result.error

    def test_blocked_builtins(self):
        result = self.env.execute("eval('1+1')")
        assert result.success is False  # eval is blocked


# ---------------------------------------------------------------------------
# Complexity detection (from existing rlm.py)
# ---------------------------------------------------------------------------

class TestComplexityDetection:
    """Quick sanity check that heuristic detection works."""

    def test_simple_query(self):
        from core.rlm import ComplexityDetector
        # Heuristic-only (no LLM call)
        score = ComplexityDetector._heuristic_score("What is ALMA?")
        assert score < 0.5

    def test_complex_query(self):
        from core.rlm import ComplexityDetector
        score = ComplexityDetector._heuristic_score(
            "Find observations that cover the CO(2-1) line at the redshift of M87 "
            "and compare their sensitivity across multiple bands"
        )
        assert score >= 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
