"""
capabilities/base.py — the three shared contracts (P1 step 1).

``ToolResult``   — the canonical envelope every capability returns.
``CallContext``  — per-request state passed EXPLICITLY (kills thread-locals and
                   ``os.environ`` token reads inside business logic).
``Capability``   — the protocol every capability satisfies.

Plus ``ResultStore`` (the ``result_id`` handle store), ``ServiceClient`` (a thin
base for injected archive clients), and ``BaseCapability`` (an optional concrete
base that removes boilerplate).

Golden rules these types enforce:
  * A capability is transport-pure: no thread-locals, no ``os.environ`` token
    reads, no ``print()``, no SSE emission. Everything it needs arrives via the
    typed input model or the ``CallContext``.
  * Failures are typed errors on ``ToolResult`` — never fabricated data.
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Type,
    runtime_checkable,
)

from pydantic import BaseModel, ConfigDict, Field


# ─────────────────────────────────────────────────────────────────────────────
# ToolResult — the canonical envelope
# ─────────────────────────────────────────────────────────────────────────────
# The key under which the native adapter smuggles canonical provenance past
# `to_native()`. `to_native()` is a FROZEN byte-parity contract (migrated tools
# return their legacy dict verbatim via `native`), so provenance/reproducible_
# snippet cannot travel inside it — yet the agent's tool loop only ever sees
# that dict. The adapter attaches this key; the agent pops it off BEFORE the
# result is serialized to the model, so the LLM-facing payload is unchanged.
# See core/provenance.py and adapters/native/__init__.py.
PROVENANCE_SIDECAR_KEY = "__quasar_provenance__"


class Provenance(BaseModel):
    """Where a result came from and how it was produced (for honest notebooks)."""

    model_config = ConfigDict(extra="allow")

    service: str                      # "datalab" | "alma" | "ads" | ...
    endpoint: Optional[str] = None    # actual URL / TAP endpoint hit
    query: Optional[str] = None       # the exact ADQL/SQL/param string executed
    retrieved_at: Optional[str] = None  # ISO timestamp (from CallContext.now)
    rowcount: Optional[int] = None
    tool_name: Optional[str] = None


class ToolResult(BaseModel):
    """The value every capability returns. Replaces bare pandas DataFrames and
    the ``__eager_data__`` / ``last_run_result`` thread-local hacks.

    ``native`` is a migration aid: when a capability is extracted from a legacy
    in-process tool whose exact output dict must be preserved for benchmark
    parity, it can set ``native`` to that dict and :meth:`to_native` returns it
    verbatim. New capabilities leave ``native`` unset and rely on the canonical
    fields.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    success: bool
    data: Any = None                                  # records / scalar / artifact ref
    columns: Optional[List[Any]] = None               # [{name,dtype,role}] or [str]
    warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = None                       # typed error; NEVER fabricated data
    degraded: bool = False                            # served a fallback / partial result
    result_id: Optional[str] = None                   # handle for chained follow-up tools
    provenance: Optional[Provenance] = None
    reproducible_snippet: Optional[str] = None        # e.g. the astroquery/pyvo line
    pagination_cursor: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)  # tool-specific structured extras

    # Byte-parity escape hatch for migrated legacy tools (see class docstring).
    native: Optional[Dict[str, Any]] = None

    def to_native(self) -> Dict[str, Any]:
        """Serialize to the dict a native (in-process) tool call returns to the
        agent's tool loop. Preserves legacy output verbatim when ``native`` is set.
        """
        if self.native is not None:
            return self.native
        out: Dict[str, Any] = {"success": self.success}
        if self.error is not None:
            out["error"] = self.error
        if self.result_id is not None:
            out["result_id"] = self.result_id
        if self.columns is not None:
            out["columns"] = self.columns
        if self.warnings:
            out["warnings"] = self.warnings
        if self.degraded:
            out["degraded"] = True
        if self.data is not None:
            out["data"] = self.data
        # Structured extras are flattened last so a capability can shape the
        # exact top-level keys the agent/benchmark expect.
        out.update(self.meta)
        return out

    def provenance_sidecar(self) -> Optional[Dict[str, Any]]:
        """The provenance :meth:`to_native` structurally cannot carry.

        Returns ``None`` when the capability declared none, so the adapter adds
        nothing to the result dict in that case. See
        :data:`PROVENANCE_SIDECAR_KEY`.
        """
        if self.provenance is None and self.reproducible_snippet is None:
            return None
        out: Dict[str, Any] = {}
        if self.provenance is not None:
            out["provenance"] = self.provenance.model_dump(exclude_none=True)
        if self.reproducible_snippet is not None:
            out["reproducible_snippet"] = self.reproducible_snippet
        return out

    def to_mcp(self) -> Dict[str, Any]:
        """Serialize for the MCP adapter (structured content). Drops the
        parity escape hatch; exposes the canonical envelope."""
        return self.model_dump(exclude={"native"}, exclude_none=True)

    @classmethod
    def fail(cls, error: str, *, degraded: bool = False, **kw: Any) -> "ToolResult":
        """Construct a typed failure. Never carries fabricated data."""
        return cls(success=False, error=error, degraded=degraded, **kw)


# ─────────────────────────────────────────────────────────────────────────────
# CallContext — per-request state, passed explicitly
# ─────────────────────────────────────────────────────────────────────────────
class Budget(BaseModel):
    """Row / time / token ceilings a capability must respect."""

    max_rows: Optional[int] = None
    max_seconds: Optional[float] = None
    max_tokens: Optional[int] = None


@runtime_checkable
class ResultStore(Protocol):
    """The ``result_id`` handle store. A capability that produces a table calls
    ``put`` and returns the handle; a follow-up capability calls ``get``.
    Implemented today by ``services.datalab_result_store.DatalabResultStore``.
    """

    def put(self, dataframe: Any, meta: Optional[Dict[str, Any]] = None) -> str: ...

    def get(self, result_id: str) -> Any: ...


class CallContext(BaseModel):
    """Everything a capability needs for one call, passed EXPLICITLY.

    This is what makes a capability transport-pure and unit-testable: it never
    reaches for a thread-local, ``self.*`` on the agent, or ``os.environ`` — it
    reads ``tokens`` / ``services`` / ``result_store`` from here.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_id: Optional[str] = None
    # Per-session archive tokens (datalab, ads, alma). NOT process env.
    tokens: Dict[str, str] = Field(default_factory=dict)
    # result_id → prior result (a ResultStore). Optional for stateless tools.
    result_store: Optional[Any] = None
    # Injected clients/services keyed by name (e.g. "datalab_client").
    services: Dict[str, Any] = Field(default_factory=dict)
    # Optional progress/status emitter (SSE on native, MCP progress on MCP).
    emit: Optional[Callable[[dict], None]] = None
    budget: Budget = Field(default_factory=Budget)
    trace_id: Optional[str] = None
    now: Optional[str] = None  # injected ISO timestamp for reproducibility

    def service(self, key: str) -> Any:
        """Fetch an injected service/client, or raise a clear error."""
        svc = self.services.get(key)
        if svc is None:
            raise KeyError(
                f"CallContext is missing the '{key}' service. The adapter must "
                f"inject it before calling this capability."
            )
        return svc

    def token(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Per-session archive token (never read from os.environ here)."""
        return self.tokens.get(key, default)

    def status(self, message: str, state: str = "running") -> None:
        """Emit a progress/status event if an emitter is wired; a no-op otherwise."""
        if self.emit is not None:
            try:
                self.emit({"type": "status", "message": message, "state": state})
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Capability protocol + optional concrete base
# ─────────────────────────────────────────────────────────────────────────────
@runtime_checkable
class Capability(Protocol):
    """The contract every capability satisfies. One typed input model → one JSON
    schema for BOTH adapters; one ``run`` → the logic."""

    name: str            # canonical tool name (no transport prefix)
    description: str     # LLM-facing description
    category: str        # for intent-subsetting + planner grouping
    InputModel: Type[BaseModel]  # pydantic model → JSON schema for both adapters
    annotations: Dict[str, Any]  # SLA hint, read-only flag, cost class → MCP annotations

    def run(self, inp: BaseModel, ctx: CallContext) -> ToolResult: ...


class BaseCapability:
    """Optional concrete base that removes boilerplate. Subclasses set the class
    attributes and implement :meth:`run`.

    Example::

        class ConeCount(BaseCapability):
            name = "datalab_cone_count"
            description = "Count catalog sources in a cone (server-side)."
            category = "datalab"
            InputModel = ConeCountInput
            def run(self, inp, ctx):
                ...
                return ToolResult(success=True, result_id=rid, meta={...})
    """

    name: str = ""
    description: str = ""
    category: str = "general"
    InputModel: Type[BaseModel]
    annotations: Dict[str, Any] = {}
    # Optional: an EXACT hand-tuned JSON schema the LLM sees. When set, native/MCP
    # adapters use it verbatim instead of ``InputModel.model_json_schema()``. This
    # preserves the carefully-worded legacy tool schemas during migration so tool
    # selection (and the benchmark) don't drift. New capabilities can leave it
    # None and rely on the pydantic-derived schema.
    json_schema: Optional[Dict[str, Any]] = None

    def parameters_schema(self) -> Dict[str, Any]:
        """The JSON schema an adapter exposes to the model."""
        if self.json_schema is not None:
            return self.json_schema
        return self.InputModel.model_json_schema()

    def run(self, inp: BaseModel, ctx: CallContext) -> ToolResult:  # pragma: no cover
        raise NotImplementedError

    # Convenience so an adapter can call cap.execute(args_dict, ctx) uniformly.
    def execute(self, args: Dict[str, Any], ctx: CallContext) -> ToolResult:
        return self.run(self.InputModel(**(args or {})), ctx)


# ─────────────────────────────────────────────────────────────────────────────
# ServiceClient — thin base for injected archive clients
# ─────────────────────────────────────────────────────────────────────────────
class ServiceClient:
    """Marker/base for injected archive clients (Data Lab, ADS, TAP, …).

    Capabilities receive concrete clients via ``CallContext.services`` — they do
    not import or construct them. Keeping a shared base lets adapters treat all
    clients uniformly and attach provenance consistently. Intentionally minimal
    for P1; grows as families migrate.
    """

    service_name: str = "service"

    def provenance(self, **extra: Any) -> Provenance:
        return Provenance(service=self.service_name, **extra)
