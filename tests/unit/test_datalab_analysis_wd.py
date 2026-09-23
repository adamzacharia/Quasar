"""S33: WD-locus side-of-line classification (synthetic points, offline)."""
import pandas as pd
from services.datalab_analysis import _overlay_locus


class _FakeAx:
    def plot(self, *a, **k):
        pass

    def legend(self, *a, **k):
        pass


def test_wd_locus_classifies_both_sides():
    # line y = 9.625 + 3.25*x. At x=0 line=9.625; at x=1 line=12.875.
    x = pd.Series([0.0, 0.0, 0.0, 1.0])
    y = pd.Series([13.0, 15.0, 9.0, 20.0])  # WD: 13,15 (>9.625), 20 (>12.875) = 3; other: 9 = 1
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



def test_realistic_hr_diagram_points():
    # A 0.3 BP-RP white dwarf at M_G 12.5 is a candidate; a K dwarf (1.0, 6.0) and
    # an M dwarf (3.0, 12.0) are main sequence (the old line rejected the white
    # dwarf too: L06, UI benchmark 2026-09-23).
    x = pd.Series([0.3, 1.0, 3.0])
    y = pd.Series([12.5, 6.0, 12.0])
    info = _overlay_locus(_FakeAx(), "wd", x, y)
    assert info["n_wd_candidates"] == 1 and info["n_other"] == 2
