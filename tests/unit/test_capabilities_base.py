"""
Unit tests for the P1 shared-core foundation:
  * capabilities/base.py — ToolResult, CallContext, BaseCapability
  * core/tools.py        — extended ToolRegistry (namespacing/validation/collision)
  * adapters/native      — capability → Tool bridge

All offline; a toy capability stands in for a real family.
"""

import pytest
from pydantic import BaseModel

from capabilities.base import (
    BaseCapability,
    CallContext,
    Provenance,
    ToolResult,
)
from core.tools import Tool, ToolRegistry
from adapters.native import build_tool, register_capabilities


# ── ToolResult ───────────────────────────────────────────────────────────────
def test_toolresult_to_native_canonical():
    tr = ToolResult(
        success=True,
        result_id="dlr_abc",
        warnings=["w1"],
        meta={"rowcount": 3, "catalog": "gaia_dr3"},
    )
    native = tr.to_native()
    assert native == {
        "success": True,
        "result_id": "dlr_abc",
        "warnings": ["w1"],
        "rowcount": 3,
        "catalog": "gaia_dr3",
    }


def test_toolresult_native_passthrough_is_verbatim():
    legacy = {"success": True, "result_id": "x", "rowcount": 9, "preview": [{"a": 1}], "note": "hi"}
    tr = ToolResult(success=True, result_id="x", native=legacy)
    assert tr.to_native() == legacy  # value parity for migrated tools (pydantic copies)


def test_toolresult_fail_carries_no_data():
    tr = ToolResult.fail("CASA not installed", meta={"fix_hint": "install casatools"})
    assert tr.success is False
    native = tr.to_native()
    assert native["success"] is False
    assert native["error"] == "CASA not installed"
    assert native["fix_hint"] == "install casatools"
    assert "data" not in native  # a failure never fabricates a payload


def test_toolresult_to_mcp_drops_parity_hatch():
    tr = ToolResult(success=True, result_id="x", native={"legacy": True},
                    provenance=Provenance(service="datalab", query="SELECT 1"))
    mcp = tr.to_mcp()
    assert "native" not in mcp
    assert mcp["success"] is True and mcp["result_id"] == "x"
    assert mcp["provenance"]["service"] == "datalab"


# ── CallContext ──────────────────────────────────────────────────────────────
def test_callcontext_service_injection_and_missing():
    client = object()
    ctx = CallContext(services={"datalab_client": client}, tokens={"ads": "k"})
    assert ctx.service("datalab_client") is client
    assert ctx.token("ads") == "k"
    assert ctx.token("missing", "d") == "d"
    with pytest.raises(KeyError):
        ctx.service("not_injected")


def test_callcontext_status_emit_and_noop():
    events = []
    ctx = CallContext(emit=lambda e: events.append(e))
    ctx.status("working", "running")
    assert events == [{"type": "status", "message": "working", "state": "running"}]
    # No emitter → silent no-op, never raises.
    CallContext().status("silent")


# ── ToolRegistry ─────────────────────────────────────────────────────────────
def _tool(name, fn=None, category="general"):
    return Tool(name=name, description="d", function=(fn or (lambda **k: {"ok": True})),
                parameters={"type": "object", "properties": {}}, category=category)


def test_registry_register_get_has_unregister():
    reg = ToolRegistry()
    reg.register(_tool("alpha"))
    assert reg.has("alpha")
    assert reg.get_tool("alpha").name == "alpha"
    assert reg.unregister("alpha") is True
    assert reg.has("alpha") is False
    assert reg.unregister("alpha") is False


def test_registry_collision_protection():
    reg = ToolRegistry()
    reg.register(_tool("dup"))
    with pytest.raises(ValueError):
        reg.register(_tool("dup"), replace=False)
    # default replace=True overwrites cleanly
    reg.register(_tool("dup"))
    assert reg.has("dup")


def test_registry_category_dedup_on_reregister():
    reg = ToolRegistry()
    reg.register(_tool("t", category="datalab"))
    reg.register(_tool("t", category="datalab"))  # re-register same name
    assert reg.names("datalab").count("t") == 1  # no duplicate category entry


def test_registry_namespace_prefix():
    reg = ToolRegistry()
    reg.register(_tool("query"), namespace="manna")
    assert reg.has("manna__query")
    assert not reg.has("query")


def test_registry_openai_tools_subsetting():
    reg = ToolRegistry()
    reg.register(_tool("a"))
    reg.register(_tool("b"))
    reg.register(_tool("c"))
    subset = reg.get_openai_tools(names=["a", "c"])
    got = {t["function"]["name"] for t in subset}
    assert got == {"a", "c"}


# ── Native adapter + BaseCapability ──────────────────────────────────────────
class _EchoInput(BaseModel):
    x: int
    label: str = "hi"


class _EchoCap(BaseCapability):
    name = "echo_tool"
    description = "Echo x back as a result."
    category = "toy"
    InputModel = _EchoInput

    def run(self, inp, ctx):
        mult = ctx.service("multiplier")
        return ToolResult(success=True, data={"x": inp.x * mult, "label": inp.label},
                          meta={"echoed": True})


def _ctx_provider():
    return CallContext(services={"multiplier": 10})


def test_native_adapter_runs_capability():
    tool = build_tool(_EchoCap(), _ctx_provider)
    assert tool.name == "echo_tool" and tool.category == "toy"
    out = tool.function(x=5)
    assert out["success"] is True
    assert out["data"] == {"x": 50, "label": "hi"}
    assert out["echoed"] is True


def test_native_adapter_typed_error_on_bad_args():
    tool = build_tool(_EchoCap(), _ctx_provider)
    out = tool.function(label="no-x-provided")  # missing required 'x'
    assert out["success"] is False
    assert "Invalid arguments" in out["error"]


def test_register_capabilities_into_registry():
    reg = ToolRegistry()
    names = register_capabilities(reg, [_EchoCap()], _ctx_provider)
    assert names == ["echo_tool"]
    assert reg.has("echo_tool")
    assert reg.get_tool("echo_tool").function(x=2)["data"]["x"] == 20


def test_basecapability_json_schema_override():
    explicit = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}

    class _C(_EchoCap):
        json_schema = explicit

    cap = _C()
    assert cap.parameters_schema() is explicit  # verbatim, not pydantic-derived
    # Without an override, falls back to the pydantic schema.
    assert "properties" in _EchoCap().parameters_schema()
