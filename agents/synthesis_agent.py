# agents/synthesis_agent.py
"""SynthesisAgent — Final answer assembly from all sub-agent results."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from openai import OpenAI

from agents.base_agent import BaseSubAgent, SubTaskResult
from core.workflow_memory import WorkflowMemory

logger = logging.getLogger(__name__)


class SynthesisAgent(BaseSubAgent):
    """
    Synthesizes final answers from all sub-agent results stored in WorkflowMemory.
    Has no tools of its own — reads from shared memory and generates text.
    """

    AGENT_TYPE = "synthesis"
    ALLOWED_TOOLS = []  # No tools — reads from workflow memory
    SYSTEM_PROMPT = """\
You are the Synthesis Agent for Quasar, responsible for combining results from
multiple specialist sub-agents into a single, coherent, expert-level answer.

Guidelines:
- Read all results from the shared workflow memory.
- Build comparison tables when the user asks to compare datasets.
- Cite specific values (beam sizes, RMS, frequencies, observation counts) from the data.
- Highlight the best dataset(s) based on user criteria (e.g. highest resolution, best sensitivity).
- Use markdown formatting: tables, bullet points, bold for key values.
- TABLES: ALWAYS use proper Markdown table syntax with | pipes and | --- | separators.
  Never output space-aligned text — it does not render correctly in the UI.
- If any sub-agent failed, note what data is missing and suggest alternatives.
- Be concise but thorough — this is the final answer the user sees.
"""

    async def run(
        self,
        task: str,
        context: str = "",
        workflow_memory: Optional[WorkflowMemory] = None,
    ) -> SubTaskResult:
        """Override run to read from workflow memory instead of using tools."""
        # Build rich context from workflow memory
        memory_context = ""
        if workflow_memory and len(workflow_memory) > 0:
            memory_context = workflow_memory.get_context_summary(max_chars=6000)

        full_input = (
            f"Task: {task}\n\n"
            f"Context from prior steps:\n{context}\n\n"
            f"All sub-agent results from shared memory:\n{memory_context}"
        )

        try:
            response = self.client.responses.create(
                model=self.model,
                input=full_input,
                instructions=self.SYSTEM_PROMPT,
                temperature=0.3,
                max_output_tokens=2000,
            )

            output_text = response.output_text.strip()

            return SubTaskResult(
                task=task,
                answer=output_text,
                agent_type=self.AGENT_TYPE,
                success=True,
            )

        except Exception as e:
            logger.error("[synthesis] Failed: %s", e)
            return SubTaskResult(
                task=task,
                answer=f"Synthesis failed: {e}",
                agent_type=self.AGENT_TYPE,
                success=False,
                error=str(e),
            )
