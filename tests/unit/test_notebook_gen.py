"""Unit tests for services.notebook_gen code-cell templates."""

from services.notebook_gen import _build_code_cell


def test_viz_code_cell_uses_light_style():
    # Exported notebooks are read outside the dark app (docs, print), so the
    # generated matplotlib code must not force dark_background — beta eval:
    # dark figures pasted into white documents were unreadable.
    code = _build_code_cell("viz", "Plot the FITS image", None)
    assert "dark_background" not in code
    assert "plt.style.use('default')" in code


def test_viz_code_cell_keeps_figure_conventions():
    code = _build_code_cell("viz", "Render the continuum map", None)
    assert "figsize=(10, 10)" in code
    assert "ZScaleInterval" in code
