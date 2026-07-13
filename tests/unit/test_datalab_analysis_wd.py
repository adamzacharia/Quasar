"""S33: WD-locus side-of-line classification (synthetic points, offline)."""
import pandas as pd
from services.datalab_analysis import _overlay_locus


class _FakeAx:
    def plot(self, *a, **k):
        pass

    def legend(self, *a, **k):
        pass


def test_wd_locus_classifies_both_sides():
    # line y = 11.5 + 5*x. At x=0 line=11.5; at x=1 line=16.5.
    x = pd.Series([0.0, 0.0, 0.0, 1.0])
    y = pd.Series([13.0, 15.0, 10.0, 20.0])  # WD: 13,15 (>11.5), 20 (>16.5) = 3; other: 10 = 1
    info = _overlay_locus(_FakeAx(), "wd", x, y)
    assert info is not None
    assert info["n_wd_candidates"] == 3
    assert info["n_other"] == 1
    assert "faint side" in info["wd_locus_rule"]
    assert "cooling track" in info["wd_locus_note"]


def test_non_wd_locus_returns_none():
    x = pd.Series([0.0, 1.0])
    y = pd.Series([1.0, 2.0])
    assert _overlay_locus(_FakeAx(), "main_sequence", x, y) is None


def test_empty_returns_none():
    x = pd.Series([], dtype=float)
    y = pd.Series([], dtype=float)
    assert _overlay_locus(_FakeAx(), "wd", x, y) is None


def test_ignores_nonfinite_points():
    x = pd.Series([0.0, 0.0])
    y = pd.Series([13.0, float("nan")])  # one WD, one NaN (ignored)
    info = _overlay_locus(_FakeAx(), "white_dwarf", x, y)
    assert info["n_wd_candidates"] == 1
    assert info["n_other"] == 0
