# core/model_council.py
"""
Model Council — Multi-model consensus for critical scientific judgments.

When a scientific decision requires high confidence (e.g. spectral line ID,
best dataset selection), the Council queries 2–3 models in parallel and
synthesizes the best answer with a confidence score.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ModelCouncil:
    """Ask 2–3 models the same question, synthesize the best answer."""

    DEFAULT_MODELS = ["gpt-4o", "gemini-2.0-pro", "claude-sonnet"]

    def __init__(
        self,
        clients: Optional[Dict[str, Any]] = None,
        conductor_model: str = "gpt-4o",
        verbose: bool = False,
    ):
        """
        Parameters
        ----------
        clients : dict
            Mapping of model_name → client instance (OpenAI, Anthropic, Google).
            If None, council is disabled and falls back to single-model answer.
        conductor_model : str
            Model used to synthesize the council's deliberation.
        """
        self.clients = clients or {}
        self.conductor_model = conductor_model
        self.verbose = verbose

    async def deliberate(
        self,
        question: str,
        context: str = "",
        models: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Query multiple models with the same question and synthesize.

        Returns
        -------
        dict
            {answer, confidence, model_responses, reasoning}
        """
        target_models = models or list(self.clients.keys())[:3]

        if len(target_models) < 2:
            # Not enough models for council — single model answer
            if target_models and target_models[0] in self.clients:
                answer = await self._query_model(
                    target_models[0], question, context
                )
                return {
                    "answer": answer,
                    "confidence": 0.7,
                    "model_responses": {target_models[0]: answer},
                    "reasoning": "Single model — no consensus possible.",
                }
            return {
                "answer": "(Council disabled — no model clients configured)",
                "confidence": 0.0,
                "model_responses": {},
                "reasoning": "No models available.",
            }

        # Query all models in parallel
        tasks = {
            model: self._query_model(model, question, context)
            for model in target_models
            if model in self.clients
        }

        responses = {}
        for model, coro in tasks.items():
            try:
                responses[model] = await coro
            except Exception as e:
                responses[model] = f"[Error: {e}]"
                logger.warning("Council: %s failed: %s", model, e)

        # Synthesize
        synthesis = await self._synthesize(question, responses)
        return synthesis

    async def _query_model(self, model: str, question: str, context: str) -> str:
        """Query a single model."""
        client = self.clients.get(model)
        if not client:
            return f"[No client for {model}]"

        prompt = question
        if context:
            prompt = f"Context:\n{context}\n\nQuestion: {question}"

        try:
            # Assume OpenAI-compatible interface
            resp = client.responses.create(
                model=model,
                input=prompt,
                instructions="You are an expert astronomer. Answer concisely and precisely.",
                temperature=0.2,
                max_output_tokens=500,
            )
            return resp.output_text.strip()
        except Exception as e:
            return f"[Error: {e}]"

    async def _synthesize(
        self, question: str, responses: Dict[str, str]
    ) -> Dict[str, Any]:
        """Synthesize multiple model responses into a consensus answer."""
        responses_text = "\n\n".join(
            f"### {model}\n{answer}" for model, answer in responses.items()
        )

        # Check for agreement
        # Simple heuristic: if all responses are similar length and content, high confidence
        response_texts = [r for r in responses.values() if not r.startswith("[Error")]
        n_successful = len(response_texts)

        if n_successful == 0:
            return {
                "answer": "(All models failed)",
                "confidence": 0.0,
                "model_responses": responses,
                "reasoning": "All models returned errors.",
            }

        if n_successful == 1:
            return {
                "answer": response_texts[0],
                "confidence": 0.6,
                "model_responses": responses,
                "reasoning": "Only one model responded successfully.",
            }

        # Use conductor model to synthesize
        conductor_client = self.clients.get(self.conductor_model)
        if not conductor_client:
            # Fallback: return first successful response
            return {
                "answer": response_texts[0],
                "confidence": 0.7,
                "model_responses": responses,
                "reasoning": "No conductor model available for synthesis.",
            }

        try:
            synth = conductor_client.responses.create(
                model=self.conductor_model,
                input=(
                    f"Question: {question}\n\n"
                    f"Multiple expert models answered this question:\n\n"
                    f"{responses_text}\n\n"
                    f"Synthesize the BEST answer. If models agree, state high confidence. "
                    f"If they disagree, note the disagreement and choose the most supported answer. "
                    f"Return JSON: {{\"answer\": \"...\", \"confidence\": 0.0-1.0, \"reasoning\": \"...\"}}"
                ),
                temperature=0.1,
                max_output_tokens=600,
                text={"format": {"type": "json_object"}},
            )
            data = json.loads(synth.output_text)
            return {
                "answer": data.get("answer", response_texts[0]),
                "confidence": data.get("confidence", 0.8),
                "model_responses": responses,
                "reasoning": data.get("reasoning", ""),
            }
        except Exception as e:
            return {
                "answer": response_texts[0],
                "confidence": 0.7,
                "model_responses": responses,
                "reasoning": f"Synthesis failed: {e}. Returning first response.",
            }
