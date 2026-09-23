"""Transport-pure deterministic ALMA documentation capability."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from capabilities.base import BaseCapability, CallContext, Provenance, ToolResult
from services.alma_reference import alma_reference_table


class AlmaReferenceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    topic: Literal["configurations", "bands", "cycles"]
    cycle: int | None = Field(default=None, ge=0, description="Requested cycle; unsupported tables fail rather than borrowing values from another cycle.")
    frequency_ghz: float | None = Field(default=None, gt=0, allow_inf_nan=False,
        description="Configuration resolution at this frequency, scaled inversely from the documented 100 GHz table. Omit for the exact table.")


class AlmaReferenceTable(BaseCapability):
    name = "alma_reference_table"
    description = ("Official, cycle-labelled ALMA configurations/baselines/angular resolutions, receiver band frequency ranges, "
                   "and cycle/project-code mapping. Cite the returned sources. Does not query public archive holdings.")
    category = "alma"
    InputModel = AlmaReferenceInput
    annotations = {"readOnlyHint": True, "idempotentHint": True}

    def run(self, inp: AlmaReferenceInput, ctx: CallContext) -> ToolResult:
        try:
            data = alma_reference_table(inp.topic, cycle=inp.cycle, frequency_ghz=inp.frequency_ghz)
        except (OSError, ValueError, KeyError) as exc:
            return ToolResult.fail(f"ALMA reference unavailable: {exc}")
        return ToolResult(success=True, data=data,
            provenance=Provenance(service="alma_documentation", endpoint=data["sources"][0]["url"],
                                  tool_name=self.name, retrieved_at=data["verified_on"]))
