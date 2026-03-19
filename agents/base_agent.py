# agents/base_agent.py
"""
BaseSubAgent — Abstract base for all Quasar specialist sub-agents.

Each sub-agent:
  1. Gets a domain-specific system prompt
  2. Has access to only its allowed subset of tools
  3. Uses the tool-calling loop to execute tasks
  4. Writes results to WorkflowMemory for other agents
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import OpenAI

from core.workflow_memory import WorkflowMemory

logger = logging.getLogger(__name__)


@dataclass
class SubTaskResult:
    """Result of a sub-agent executing a task."""
    task: str
    answer: Any
    agent_type: str
    success: bool = True
    error: Optional[str] = None
    tools_used: List[str] = field(default_factory=list)


class BaseSubAgent:
    """
    Base class for specialist sub-agents.

    Subclasses should define:
      - AGENT_TYPE: str (e.g. "archive", "literature")
      - ALLOWED_TOOLS: List[str] (tool names this agent can use)
      - SYSTEM_PROMPT: str (domain-specific instructions)
    """

    AGENT_TYPE: str = "general"
    ALLOWED_TOOLS: List[str] = []
    SYSTEM_PROMPT: str = "You are a helpful research assistant."
    MAX_TOOL_ROUNDS: int = 8

    def __init__(
        self,
        client: OpenAI,
        model: str,
        tool_registry: Any,
        verbose: bool = False,
    ):
        self.client = client
        self.model = model
        self.tool_registry = tool_registry
        self.verbose = verbose
        # Filter tools to only those allowed for this agent
        self._tools = self._build_tool_list()

    def _build_tool_list(self) -> List[dict]:
        """Build OpenAI-format tools list filtered to ALLOWED_TOOLS."""
        tools = []
        for tool_name in self.ALLOWED_TOOLS:
            tool = self.tool_registry.get_tool(tool_name)
            if tool:
                tools.append({
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                })
        return tools

    async def run(
        self,
        task: str,
        context: str = "",
        workflow_memory: Optional[WorkflowMemory] = None,
    ) -> SubTaskResult:
        """
        Execute a task using this agent's allowed tools.

        Parameters
        ----------
        task : str
            The task description to execute.
        context : str
            Context from dependency results.
        workflow_memory : WorkflowMemory, optional
            Shared memory to read/write results.

        Returns
        -------
        SubTaskResult
        """
        # Build input with context
        full_input = task
        if context:
            full_input = f"Context from prior steps:\n{context}\n\nTask: {task}"

        # Add workflow memory context if available
        if workflow_memory and len(workflow_memory) > 0:
            mem_summary = workflow_memory.get_context_summary(max_chars=3000)
            full_input += f"\n\nShared Memory:\n{mem_summary}"

        try:
            tools_used = []
            output_text = ""

            # Tool-calling loop (same pattern as agent.py stream_response_api)
            for _round in range(self.MAX_TOOL_ROUNDS):
                response = self.client.responses.create(
                    model=self.model,
                    input=full_input if _round == 0 else tool_results,
                    instructions=self.SYSTEM_PROMPT,
                    tools=self._tools if self._tools else None,
                    temperature=0.3,
                    max_output_tokens=1500,
                )

                # Check for function calls
                function_calls = []
                for item in response.output:
                    if getattr(item, "type", None) == "function_call":
                        function_calls.append(item)
                    elif getattr(item, "type", None) == "message":
                        for content in getattr(item, "content", []):
                            if getattr(content, "type", None) == "output_text":
                                output_text += content.text

                if not function_calls:
                    break

                # Execute tool calls
                tool_results = []
                for fc in function_calls:
                    tool_name = fc.name
                    try:
                        args = json.loads(fc.arguments) if fc.arguments else {}
                    except json.JSONDecodeError:
                        args = {}

                    logger.info("[%s] Tool call: %s(%s)", self.AGENT_TYPE, tool_name, args)
                    tools_used.append(tool_name)

                    tool = self.tool_registry.get_tool(tool_name)
                    if tool:
                        try:
                            result = tool.execute(**args)
                            result_str = json.dumps(result, default=str)[:6000]
                        except Exception as te:
                            result_str = json.dumps({"error": str(te)})
                    else:
                        result_str = json.dumps({"error": f"Unknown tool: {tool_name}"})

                    tool_results.append({
                        "type": "function_call_output",
                        "call_id": fc.call_id,
                        "output": result_str,
                    })

            # Write result to workflow memory
            if workflow_memory:
                workflow_memory.write(
                    self.AGENT_TYPE,
                    f"result_{task[:30].replace(' ', '_')}",
                    output_text or "(no text output)",
                )

            return SubTaskResult(
                task=task,
                answer=output_text or "(no response generated)",
                agent_type=self.AGENT_TYPE,
                success=True,
                tools_used=tools_used,
            )

        except Exception as e:
            logger.error("[%s] Failed: %s", self.AGENT_TYPE, e)
            return SubTaskResult(
                task=task,
                answer=f"Error: {e}",
                agent_type=self.AGENT_TYPE,
                success=False,
                error=str(e),
            )
