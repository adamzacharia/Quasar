"""Tests for the datalab_analysis plot expression engine.

Covers the gpt-oss-120b failure mode where an HR-diagram request died with
"Column 'M_G' not found for expression 'M_G'": derived quantities like absolute
magnitude need log10(), case-insensitive alias lookup, and errors that teach the
model how to fix the call.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from services.datalab_analysis import _eval_expression, catalog_scatter
from services.datalab_result_store import DatalabResultStore


class _MemoryPlottingService:
    def _apply_style(self, dark=False):
        from services.plotting import PlottingService

        return PlottingService()._apply_style(dark=dark)

    def _save_and_encode(self, fig, filename):
        import base64
        import io
        import matplotlib.pyplot as plt

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        return {
            "success": True,
            "base64_png": base64.b64encode(buf.getvalue()).decode(),
            "web_url": f"/plots/{filename}.png",
            "png_path": None,
            "pdf_path": None,
        }


def _gaia_frame():
    return pd.DataFrame(
        {
            "parallax": [10.0, 5.0, 2.0, 0.0, 20.0],
            "phot_g_mean_mag": [15.0, 16.0, 17.0, 18.0, 14.0],
            "bp_rp": [0.5, 0.7, 1.0, 1.2, 0.3],
        }
    )


def test_log10_absolute_magnitude_expression():
    frame = _gaia_frame()
    result = _eval_expression(frame, "phot_g_mean_mag + 5*log10(parallax/100)")
    expected = frame["phot_g_mean_mag"] + 5 * np.log10(frame["parallax"] / 100.0)
    finite = np.isfinite(expected)
    assert np.allclose(result[finite], expected[finite])
    # parallax == 0 row becomes -inf, later masked by the plot; no crash.
    assert not np.isfinite(result[3])


def test_simple_arithmetic_and_parentheses():
    frame = pd.DataFrame({"g": [1.0, 2.0], "r": [0.5, 1.0]})
    assert np.allclose(_eval_expression(frame, "g - r"), [0.5, 1.0])
    assert np.allclose(_eval_expression(frame, "(g - r) * 2"), [1.0, 2.0])
    assert np.allclose(_eval_expression(frame, "2.5 * g"), [2.5, 5.0])
    assert np.allclose(_eval_expression(frame, "-g"), [-1.0, -2.0])


def test_case_insensitive_column_resolution():
    # Postgres folds unquoted SELECT aliases to lowercase: "... AS M_G" -> m_g.
    frame = pd.DataFrame({"m_g": [11.0, 12.0], "bp_rp": [0.1, 0.2]})
    assert np.allclose(_eval_expression(frame, "M_G"), [11.0, 12.0])
    assert np.allclose(_eval_expression(frame, "M_G + 1"), [12.0, 13.0])


def test_missing_column_error_lists_columns_and_suggestion():
    frame = _gaia_frame()
    with pytest.raises(ValueError) as exc:
        _eval_expression(frame, "parallx + 1")
    msg = str(exc.value)
    assert "Available columns:" in msg
    assert "parallax" in msg
    assert "Did you mean" in msg
    assert "log10" in msg  # grammar note teaches the supported functions


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os')",
        "frame.attr",
        "g[0]",
        "g > 5",
        "median(g)",
        "log10(g, base=2)",
        "lambda x: x",
        "'text'",
        "g; r",
    ],
)
def test_unsafe_expressions_rejected(expr):
    frame = pd.DataFrame({"g": [1.0], "r": [2.0]})
    with pytest.raises(ValueError):
        _eval_expression(frame, expr)


def test_constant_only_expression_rejected():
    frame = pd.DataFrame({"g": [1.0]})
    with pytest.raises(ValueError, match="at least one result column"):
        _eval_expression(frame, "5 + 3")


@pytest.mark.parametrize(
    "expr",
    [
        "dec + 9**9**9**9",       # chained constant Pow (right-assoc bigint bomb)
        "g + 10**10**10",         # the dossier's literal repro
        "g + 2**1000",            # single huge constant exponent
        "g + ((9**64)**64)**64",  # exponent-capped but base-nested explosion
    ],
)
def test_constant_pow_bomb_rejected_before_compile(expr):
    """expr-const-pow-dos / dl-expr-pow-bigint-dos: constant ** chains must be
    rejected during AST validation — compile()'s constant folding would
    otherwise build an astronomically large exact integer and hang the worker."""
    frame = pd.DataFrame({"g": [1.0], "dec": [2.0]})
    with pytest.raises(ValueError, match="[Uu]nsafe|exponent"):
        _eval_expression(frame, expr)


def test_reasonable_pow_still_allowed():
    frame = pd.DataFrame({"g": [2.0, 3.0], "flux": [100.0, 1000.0]})
    assert np.allclose(_eval_expression(frame, "g**2"), [4.0, 9.0])
    assert np.allclose(_eval_expression(frame, "g + 10**2"), [102.0, 103.0])
    # Data-driven exponents evaluate in numpy floats — no bigint path.
    assert np.allclose(_eval_expression(frame, "2**g"), [4.0, 8.0])
    # Scientific-notation-style constants stay usable (10**38 is finite).
    assert np.isfinite(_eval_expression(frame, "flux / 10**8")).all()


@pytest.mark.parametrize("expr", ["minimum(g)", "arctan2(g)", "log10()"])
def test_wrong_function_arity_raises_valueerror_with_grammar_note(expr):
    # Arity errors surface as numpy TypeErrors inside eval; they must come back
    # as ValueError carrying the grammar note so the model can self-correct.
    frame = pd.DataFrame({"g": [1.0, 2.0]})
    with pytest.raises(ValueError, match="log10"):
        _eval_expression(frame, expr)


def test_catalog_scatter_renders_hr_diagram_with_log10():
    store = DatalabResultStore()
    result_id = store.put(_gaia_frame(), {"catalog": "gaia_dr3", "table": "gaia_source"})
    out = catalog_scatter(
        result_id,
        "bp_rp",
        "phot_g_mean_mag + 5*log10(parallax/100)",
        invert_y=True,
        overlay_locus="wd",
        result_store=store,
        plotting_service=_MemoryPlottingService(),
    )
    assert out["success"] is True
    # The parallax == 0 row is non-finite and must be masked out.
    assert out["points"] == 4
