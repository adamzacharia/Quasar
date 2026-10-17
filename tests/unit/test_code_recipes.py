"""docs/recipes/*.py — the curated snippets code_recipe serves must be real
Python that names ONLY APIs that exist in the installed libraries.

Offline: every recipe compiles and every ``Alma.<attr>`` / ``alminer.<attr>`` /
``pyvo.dal.<attr>`` it names exists. Live (RUN_LIVE_RECIPE_TESTS=1): the
astroquery and pyvo recipes execute against the ALMA archive.
"""
from __future__ import annotations

import ast
import os
import re
from pathlib import Path

import pytest

RECIPES = sorted((Path(__file__).resolve().parents[2] / "docs" / "recipes").glob("*.py"))


@pytest.mark.parametrize("path", RECIPES, ids=lambda p: p.name)
def test_recipe_compiles_and_carries_metadata(path):
    src = path.read_text(encoding="utf-8")
    compile(src, str(path), "exec")
    head = ast.get_docstring(ast.parse(src)) or ""
    for key in ("library:", "last_tested:", "tested_against:"):
        assert key in head, f"{path.name} docstring lacks {key}"


def _attrs(src: str, root: str) -> set:
    return set(re.findall(rf"\b{re.escape(root)}\.([A-Za-z_][A-Za-z0-9_]*)\(", src))


def test_astroquery_recipes_only_name_real_alma_methods():
    from astroquery.alma import Alma

    real = {m for m in dir(Alma) if not m.startswith("_")}
    for path in RECIPES:
        if not path.name.startswith("astroquery_"):
            continue
        used = _attrs(path.read_text(encoding="utf-8"), "alma")
        missing = sorted(u for u in used if u not in real)
        assert missing == [], f"{path.name} names non-existent Alma methods: {missing}"
        code = path.read_text(encoding="utf-8")
        doc = ast.get_docstring(ast.parse(code), clean=False) or ""
        code_only = code.replace(doc, "")  # the docstring WARNS against the names; the code must not USE them
        assert not re.search(r"get_data_links|query_sql|get_data_products", code_only), \
            "the invented API names from the 2026-09-22 benchmark must never appear in recipe code"


def test_alminer_recipe_only_names_real_functions():
    alminer = pytest.importorskip("alminer")
    path = next(p for p in RECIPES if p.name.startswith("alminer_"))
    used = _attrs(path.read_text(encoding="utf-8"), "alminer")
    missing = sorted(u for u in used if not hasattr(alminer, u))
    assert missing == [], f"alminer recipe names non-existent functions: {missing}"


def test_recipe_keyword_arguments_exist_in_the_installed_libraries():
    """2026-10-16: the ALMiner recipe called run_query(adql, print_targets=False), which the installed alminer
    rejects (TypeError), so the 'tested' snippet stopped at step 3. Every keyword a recipe passes to an alminer.* or
    Alma method must be in that function's signature."""
    import inspect

    alminer = pytest.importorskip("alminer")
    from astroquery.alma import Alma

    bad = []
    for path in RECIPES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)):
                continue
            owner, attr = node.func.value.id, node.func.attr
            fn = getattr(alminer, attr, None) if owner == "alminer" else getattr(Alma, attr, None) if owner == "alma" else None
            if fn is None or not callable(fn):
                continue
            params = inspect.signature(fn).parameters
            if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
                continue
            bad += [f"{path.name}: {owner}.{attr}({kw.arg}=...)" for kw in node.keywords if kw.arg and kw.arg not in params]
    assert bad == [], bad


def test_pyvo_recipes_use_tapservice_search():
    for path in RECIPES:
        if path.name.startswith("pyvo_"):
            src = path.read_text(encoding="utf-8")
            assert "pyvo.dal.TAPService(" in src and ".search(" in src


def test_code_recipe_tool_serves_the_files():
    from capabilities.alma_tools import CodeRecipe
    from capabilities.base import CallContext

    cap = CodeRecipe()
    out = cap.run(cap.InputModel(task="how do I query ALMA for an object by name", library="astroquery"), CallContext()).to_native()
    assert out["success"] and out["recipes"][0]["file"].endswith("astroquery_object_search.py")
    assert "query_object" in out["recipes"][0]["code"] and out["recipes"][0]["last_tested"]
    out = cap.run(cap.InputModel(task="list and download DataLink files", library="astroquery"), CallContext()).to_native()
    assert out["recipes"][0]["file"].endswith("astroquery_datalink.py")
    out = cap.run(cap.InputModel(task="conesearch with run_query", library="alminer"), CallContext()).to_native()
    assert out["recipes"][0]["file"].endswith("alminer_alminer.py")
    out = cap.run(cap.InputModel(task="ADQL via TAP", library="pyvo"), CallContext()).to_native()
    assert out["recipes"][0]["file"].endswith("pyvo_adql.py")
    out = cap.run(cap.InputModel(task="list all observations of a galaxy in Band 3 released before a date", library="pyvo"),
                  CallContext()).to_native()
    assert out["recipes"][0]["file"].endswith("pyvo_dataset_list.py") and out["recipes"][0]["task"] == "dataset_list"
    out = cap.run(cap.InputModel(task="list every MOUS released before 2025", library="astroquery"), CallContext()).to_native()
    assert out["recipes"][0]["file"].endswith("astroquery_dataset_list.py")
    bad = cap.run(cap.InputModel(task="anything", library="fortran"), CallContext()).to_native()
    assert bad["success"] is False


@pytest.mark.skipif(os.getenv("RUN_LIVE_RECIPE_TESTS") != "1", reason="live archive; set RUN_LIVE_RECIPE_TESTS=1")
@pytest.mark.parametrize("name", ["astroquery_object_search.py", "astroquery_adql.py", "pyvo_adql.py", "pyvo_cone_search.py",
                                  "pyvo_dataset_list.py", "astroquery_dataset_list.py"])
def test_recipes_run_live(name):
    path = next(p for p in RECIPES if p.name == name)
    ns: dict = {}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), ns)  # noqa: S102 - the recipes ARE the code under test


def test_code_recipe_returns_runnable_code_without_the_metadata_docstring():
    """Live 2026-09-23 D15: the model re-typed the recipe's metadata docstring
    and dropped its closing quotes, so the displayed snippet did not parse. The
    tool now returns only runnable code (provenance in separate fields)."""
    import ast

    from capabilities.alma_tools import CodeRecipe
    from capabilities.base import CallContext

    cap = CodeRecipe()
    for lib, task in (("astroquery", "query by object name"), ("pyvo", "ADQL via TAP"), ("alminer", "target search")):
        out = cap.run(cap.InputModel(library=lib, task=task), CallContext())
        out = out.to_native() if hasattr(out, "to_native") else out
        assert out["success"], (lib, out)
        for r in out["recipes"]:
            ast.parse(r["code"])
            assert '"""' not in r["code"].split("\n", 3)[1] and "tests/unit" not in r["code"]
            assert r["last_tested"]


@pytest.mark.parametrize("name", ["pyvo_dataset_list.py", "astroquery_dataset_list.py"])
def test_dataset_list_recipes_refuse_an_overflowed_answer(name, monkeypatch):
    """Guard CX-12: pyvo stays silent when the answer is exactly the requested cap; the recipe must check OVERFLOW."""
    import astropy.coordinates
    import pyvo
    from astropy.coordinates import SkyCoord
    from astropy.table import Table

    class _Res:
        query_status = "OVERFLOW"

        def to_table(self):
            return Table({"member_ous_uid": ["uid://A001/X1/X1"], "proposal_id": ["2019.1.00001.S"]})

    class _Svc:
        def __init__(self, *a, **k):
            pass

        def search(self, *a, **k):
            return _Res()

    monkeypatch.setattr(astropy.coordinates.SkyCoord, "from_name", classmethod(lambda cls, name: SkyCoord(10.0, -20.0, unit="deg")))
    monkeypatch.setattr(pyvo.dal, "TAPService", _Svc)
    from astroquery.alma.core import AlmaClass  # `Alma` is an instance; Alma() returns a fresh AlmaClass

    monkeypatch.setattr(AlmaClass, "query_tap", lambda self, q, **k: _Res())
    path = next(p for p in RECIPES if p.name == name)
    with pytest.raises(SystemExit, match="row cap reached"):
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), {})  # noqa: S102
