# core/llm_client.py
"""
LLM Client — Unified, provider-agnostic interface for all LLM calls.

Every Quasar module (agent, Conductor, RLM, Recovery, sub-agents) calls
`client.responses.create(...)`.  This module provides a drop-in replacement
that routes to the correct provider based on the model name:

    gpt-* / o1* / o3* / o4*  →  OpenAI Responses API  (native)
    claude-*                  →  Anthropic Messages API (tool_use)
    gemini-*                  →  Google GenAI API       (function_calling)
    local/*                   →  OpenAI-compat Chat Completions
                                 (Ollama / LM Studio via base_url)

The key insight is the `ResponsesShim` — a property on `LLMClient` named
`responses` that exposes `.create(**kwargs)`.  This means existing code like
`self.client.responses.create(...)` works unchanged.

Usage:
    # In agent.py (replaces `self.client = OpenAI(...)`)
    from core.llm_client import LLMClient
    self.client = LLMClient(model="gpt-4o")       # OpenAI
    self.client = LLMClient(model="local/deepseek-r1:7b")  # Ollama

    # All existing calls work:
    resp = self.client.responses.create(
        model="gpt-4o", input="Hello", instructions="...",
    )
    print(resp.output_text)
"""

from __future__ import annotations

import json
import os
import uuid
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

def detect_provider(model: str) -> str:
    """Detect the LLM provider from the model name."""
    if model.startswith("local/"):
        return "local"
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith("gemini-") or model.startswith("gemma-"):
        return "google"
    # Default: OpenAI (gpt-*, o1*, o3*, o4*, etc.)
    return "openai"


# ---------------------------------------------------------------------------
# Response objects — mimic OpenAI Responses API output shapes
# ---------------------------------------------------------------------------

@dataclass
class FunctionCallItem:
    """Mimics an OpenAI function_call output item."""
    type: str = "function_call"
    name: str = ""
    arguments: str = ""
    call_id: str = ""
    id: str = ""

    def __post_init__(self):
        if not self.call_id:
            self.call_id = f"call_{uuid.uuid4().hex[:16]}"
        if not self.id:
            self.id = self.call_id


@dataclass
class TextContentItem:
    """Mimics an output_text content block."""
    type: str = "output_text"
    text: str = ""


@dataclass
class MessageOutputItem:
    """Mimics a message output item containing content blocks."""
    type: str = "message"
    content: List[TextContentItem] = field(default_factory=list)


@dataclass
class LLMResponse:
    """
    Mimics OpenAI's Response object so all existing code works.

    Key attributes used by existing code:
      - resp.output_text  → combined text
      - resp.id           → response ID (for previous_response_id chaining)
      - resp.output       → list of items (function_call / message items)
    """
    output_text: str = ""
    id: str = ""
    output: List[Any] = field(default_factory=list)

    def __post_init__(self):
        if not self.id:
            self.id = f"resp_{uuid.uuid4().hex[:16]}"


# ---------------------------------------------------------------------------
# Streaming event objects — mimic OpenAI SSE events
# ---------------------------------------------------------------------------

@dataclass
class StreamEvent:
    """Mimics a single SSE event from OpenAI's streaming Responses API."""
    type: str = ""
    # For response.created
    response: Optional[LLMResponse] = None
    # For response.output_text.delta
    delta: str = ""
    # For response.output_item.added
    item: Optional[Any] = None


# ---------------------------------------------------------------------------
# Responses Shim — makes `client.responses.create(...)` work
# ---------------------------------------------------------------------------

class ResponsesShim:
    """
    Provides `client.responses.create(...)` interface.
    Routes to the correct provider implementation.
    """

    def __init__(self, llm_client: "LLMClient"):
        self._llm = llm_client

    def create(self, **kwargs) -> Any:
        """
        Drop-in replacement for OpenAI's `client.responses.create(...)`.

        Supported kwargs (translated per provider):
          - model: str
          - input: str | list  (user message or tool results)
          - instructions: str  (system prompt)
          - temperature: float
          - max_output_tokens: int
          - tools: list        (function tool definitions)
          - stream: bool
          - text: dict         (e.g. {"format": {"type": "json_object"}})
          - previous_response_id: str  (conversation continuity — OpenAI only)
        """
        model = kwargs.get("model", self._llm.default_model)
        provider = detect_provider(model)
        stream = kwargs.get("stream", False)
        attachments = kwargs.pop("attachments", None)

        if provider == "openai":
            return self._call_openai(kwargs, attachments=attachments)
        elif provider == "anthropic":
            if stream:
                return self._stream_anthropic(kwargs, attachments=attachments)
            return self._call_anthropic(kwargs, attachments=attachments)
        elif provider == "google":
            if stream:
                return self._stream_google(kwargs, attachments=attachments)
            return self._call_google(kwargs, attachments=attachments)
        elif provider == "local":
            if attachments:
                raise ValueError(
                    "Document attachments are not supported for local models in this path."
                )
            if stream:
                return self._stream_local(kwargs)
            return self._call_local(kwargs)
        else:
            raise ValueError(f"Unknown provider for model: {model}")

    # ── OpenAI (passthrough — native Responses API) ──────────────────────

    def _call_openai(self, kwargs: dict, attachments: Optional[List[Dict[str, Any]]] = None) -> Any:
        """Direct passthrough to OpenAI Responses API."""
        client = self._llm._get_openai_client()
        if attachments:
            kwargs = dict(kwargs)
            kwargs["input"] = self._build_openai_input(kwargs.get("input", ""), attachments)
        return client.responses.create(**kwargs)

    def _build_openai_input(
        self,
        input_data,
        attachments: Optional[List[Dict[str, Any]]] = None,
    ):
        """Build a Responses API input payload with native input_file items."""
        if isinstance(input_data, list):
            return input_data

        content = []
        prompt_text = input_data if isinstance(input_data, str) else json.dumps(input_data, default=str)
        if prompt_text:
            content.append({"type": "input_text", "text": prompt_text})

        for attachment in attachments or []:
            if attachment.get("kind") != "openai_input_file":
                continue
            file_id = attachment.get("file_id")
            if not file_id:
                raise ValueError("OpenAI attachment is missing file_id.")
            content.append({"type": "input_file", "file_id": file_id})

        if not content:
            content.append({"type": "input_text", "text": ""})

        return [{"role": "user", "content": content}]

    # ── Anthropic (Claude) ───────────────────────────────────────────────

    def _call_anthropic(
        self,
        kwargs: dict,
        attachments: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Translate responses.create() to Anthropic Messages API."""
        client = self._llm._get_anthropic_client()
        model = kwargs.get("model", self._llm.default_model)
        instructions = kwargs.get("instructions", "")
        input_data = kwargs.get("input", "")
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_output_tokens", 2000)
        tools_raw = kwargs.get("tools", None)
        json_mode = False
        text_opt = kwargs.get("text", None)
        if text_opt and isinstance(text_opt, dict):
            fmt = text_opt.get("format", {})
            if fmt.get("type") == "json_object":
                json_mode = True

        # Build messages
        messages = self._build_anthropic_messages(input_data, attachments=attachments)

        # Translate tool schemas
        anthropic_tools = None
        if tools_raw:
            anthropic_tools = self._translate_tools_for_anthropic(tools_raw)

        # System prompt
        system_text = instructions or ""
        if json_mode:
            system_text += "\n\nYou MUST respond with valid JSON only. No extra text."

        call_kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system_text:
            call_kwargs["system"] = system_text
        if anthropic_tools:
            call_kwargs["tools"] = anthropic_tools

        resp = client.messages.create(**call_kwargs)

        # Convert to LLMResponse
        return self._anthropic_to_llm_response(resp)

    def _stream_anthropic(
        self,
        kwargs: dict,
        attachments: Optional[List[Dict[str, Any]]] = None,
    ):
        """Streaming Anthropic call — returns an iterator of StreamEvents."""
        client = self._llm._get_anthropic_client()
        model = kwargs.get("model", self._llm.default_model)
        instructions = kwargs.get("instructions", "")
        input_data = kwargs.get("input", "")
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_output_tokens", 2000)
        tools_raw = kwargs.get("tools", None)

        messages = self._build_anthropic_messages(input_data, attachments=attachments)
        anthropic_tools = self._translate_tools_for_anthropic(tools_raw) if tools_raw else None

        call_kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if instructions:
            call_kwargs["system"] = instructions
        if anthropic_tools:
            call_kwargs["tools"] = anthropic_tools

        # Return a generator that yields StreamEvent objects
        return self._anthropic_stream_generator(client, call_kwargs)

    def _anthropic_stream_generator(self, client, call_kwargs):
        """Generate StreamEvent objects from Anthropic streaming."""
        resp_id = f"resp_{uuid.uuid4().hex[:16]}"

        # Emit response.created
        yield StreamEvent(
            type="response.created",
            response=LLMResponse(id=resp_id),
        )

        function_calls = {}  # index -> {name, arguments}
        current_tool_index = None

        with client.messages.stream(**call_kwargs) as stream:
            for event in stream:
                event_type = getattr(event, 'type', '')

                if event_type == 'content_block_start':
                    block = getattr(event, 'content_block', None)
                    if block and getattr(block, 'type', '') == 'tool_use':
                        idx = getattr(event, 'index', 0)
                        current_tool_index = idx
                        fc = FunctionCallItem(
                            name=block.name,
                            call_id=block.id or f"call_{uuid.uuid4().hex[:16]}",
                        )
                        function_calls[idx] = fc
                        yield StreamEvent(
                            type="response.output_item.added",
                            item=fc,
                        )

                elif event_type == 'content_block_delta':
                    delta = getattr(event, 'delta', None)
                    if delta:
                        delta_type = getattr(delta, 'type', '')
                        if delta_type == 'text_delta':
                            yield StreamEvent(
                                type="response.output_text.delta",
                                delta=delta.text,
                            )
                        elif delta_type == 'input_json_delta':
                            idx = getattr(event, 'index', current_tool_index or 0)
                            fc = function_calls.get(idx)
                            if fc:
                                fc.arguments += delta.partial_json
                                yield StreamEvent(
                                    type="response.function_call_arguments.delta",
                                    delta=delta.partial_json,
                                    item=fc,
                                )

    def _build_anthropic_messages(self, input_data, attachments: Optional[List[Dict[str, Any]]] = None) -> list:
        """Convert responses.create() input to Anthropic messages format."""
        if isinstance(input_data, str):
            if attachments:
                content_blocks = []
                for attachment in attachments:
                    if attachment.get("kind") != "anthropic_document_file":
                        continue
                    file_id = attachment.get("file_id")
                    if not file_id:
                        raise ValueError("Anthropic attachment is missing file_id.")
                    content_blocks.append({
                        "type": "document",
                        "title": attachment.get("filename") or "Attached document",
                        "source": {
                            "type": "file",
                            "file_id": file_id,
                        },
                    })
                if input_data:
                    content_blocks.append({"type": "text", "text": input_data})
                if not content_blocks:
                    content_blocks.append({"type": "text", "text": input_data or ""})
                return [{"role": "user", "content": content_blocks}]
            return [{"role": "user", "content": input_data}]
        elif isinstance(input_data, list):
            # Tool results format: [{"type": "function_call_output", "call_id": ..., "output": ...}]
            messages = []
            # Build assistant message with tool_use blocks (required by Anthropic)
            tool_use_blocks = []
            tool_results = []
            for item in input_data:
                if isinstance(item, dict) and item.get("type") == "function_call_output":
                    call_id = item.get("call_id", "")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": call_id,
                        "content": item.get("output", ""),
                    })
            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            return messages
        return [{"role": "user", "content": str(input_data)}]

    def _translate_tools_for_anthropic(self, tools: list) -> list:
        """Convert OpenAI tool format to Anthropic tool format."""
        anthropic_tools = []
        for t in tools:
            if t.get("type") == "function" or "name" in t:
                anthropic_tools.append({
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "input_schema": t.get("parameters", {"type": "object", "properties": {}}),
                })
        return anthropic_tools

    def _anthropic_to_llm_response(self, resp) -> LLMResponse:
        """Convert Anthropic response to LLMResponse."""
        output_text = ""
        output_items = []

        for block in resp.content:
            if block.type == "text":
                output_text += block.text
            elif block.type == "tool_use":
                fc = FunctionCallItem(
                    name=block.name,
                    arguments=json.dumps(block.input) if isinstance(block.input, dict) else str(block.input),
                    call_id=block.id,
                )
                output_items.append(fc)

        return LLMResponse(
            output_text=output_text,
            id=resp.id if hasattr(resp, 'id') else f"resp_{uuid.uuid4().hex[:16]}",
            output=output_items or [MessageOutputItem(content=[TextContentItem(text=output_text)])],
        )

    # ── Google Gemini ────────────────────────────────────────────────────

    def _call_google(
        self,
        kwargs: dict,
        attachments: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Translate responses.create() to Google GenAI API."""
        client = self._llm._get_google_client()
        model = kwargs.get("model", self._llm.default_model)
        instructions = kwargs.get("instructions", "")
        input_data = kwargs.get("input", "")
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_output_tokens", 2000)
        tools_raw = kwargs.get("tools", None)
        json_mode = False
        text_opt = kwargs.get("text", None)
        if text_opt and isinstance(text_opt, dict):
            fmt = text_opt.get("format", {})
            if fmt.get("type") == "json_object":
                json_mode = True

        # Build content
        user_text = input_data if isinstance(input_data, str) else json.dumps(input_data, default=str)
        full_prompt = f"{instructions}\n\n{user_text}" if instructions else user_text
        if json_mode:
            full_prompt += "\n\nRespond with valid JSON only."

        # Translate tools for Gemini
        gemini_tools = None
        if tools_raw:
            gemini_tools = self._translate_tools_for_google(tools_raw)

        gen_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if json_mode:
            gen_config["response_mime_type"] = "application/json"

        call_kwargs = {
            "model": model,
            "contents": self._build_google_contents(full_prompt, attachments=attachments),
            "config": gen_config,
        }
        if gemini_tools:
            call_kwargs["config"]["tools"] = gemini_tools

        resp = client.models.generate_content(**call_kwargs)

        return self._google_to_llm_response(resp)

    def _stream_google(
        self,
        kwargs: dict,
        attachments: Optional[List[Dict[str, Any]]] = None,
    ):
        """Streaming Google GenAI call — returns an iterator of StreamEvents."""
        client = self._llm._get_google_client()
        model = kwargs.get("model", self._llm.default_model)
        instructions = kwargs.get("instructions", "")
        input_data = kwargs.get("input", "")
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_output_tokens", 2000)
        tools_raw = kwargs.get("tools", None)

        user_text = input_data if isinstance(input_data, str) else json.dumps(input_data, default=str)
        full_prompt = f"{instructions}\n\n{user_text}" if instructions else user_text

        gemini_tools = self._translate_tools_for_google(tools_raw) if tools_raw else None

        gen_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        call_kwargs = {
            "model": model,
            "contents": self._build_google_contents(full_prompt, attachments=attachments),
            "config": gen_config,
        }
        if gemini_tools:
            call_kwargs["config"]["tools"] = gemini_tools

        # Emit response.created
        resp_id = f"resp_{uuid.uuid4().hex[:16]}"
        yield StreamEvent(type="response.created", response=LLMResponse(id=resp_id))

        stream = client.models.generate_content_stream(**call_kwargs)
        for chunk in stream:
            text = chunk.text if hasattr(chunk, "text") and chunk.text else ""
            if text:
                yield StreamEvent(type="response.output_text.delta", delta=text)

            # Handle function calls in streaming
            if hasattr(chunk, 'candidates'):
                for candidate in chunk.candidates:
                    if hasattr(candidate, 'content') and candidate.content:
                        for part in candidate.content.parts:
                            if hasattr(part, 'function_call') and part.function_call:
                                fc = FunctionCallItem(
                                    name=part.function_call.name,
                                    arguments=json.dumps(dict(part.function_call.args)) if part.function_call.args else "{}",
                                )
                                yield StreamEvent(type="response.output_item.added", item=fc)

    def _translate_tools_for_google(self, tools: list) -> list:
        """Convert OpenAI tool format to Google GenAI function declarations."""
        from google.genai import types as genai_types

        function_declarations = []
        for t in tools:
            if t.get("type") == "function" or "name" in t:
                params = t.get("parameters", {})
                function_declarations.append(
                    genai_types.FunctionDeclaration(
                        name=t.get("name", ""),
                        description=t.get("description", ""),
                        parameters=params,
                    )
                )
        if function_declarations:
            return [genai_types.Tool(function_declarations=function_declarations)]
        return []

    def _google_to_llm_response(self, resp) -> LLMResponse:
        """Convert Google GenAI response to LLMResponse."""
        output_text = ""
        output_items = []

        if hasattr(resp, 'text') and resp.text:
            output_text = resp.text

        # Handle function calls
        if hasattr(resp, 'candidates'):
            for candidate in resp.candidates:
                if hasattr(candidate, 'content') and candidate.content:
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            if not output_text:
                                output_text = part.text
                        if hasattr(part, 'function_call') and part.function_call:
                            fc = FunctionCallItem(
                                name=part.function_call.name,
                                arguments=json.dumps(dict(part.function_call.args)) if part.function_call.args else "{}",
                            )
                            output_items.append(fc)

        return LLMResponse(
            output_text=output_text,
            output=output_items or [MessageOutputItem(content=[TextContentItem(text=output_text)])],
        )

    def _build_google_contents(
        self,
        prompt_text: str,
        attachments: Optional[List[Dict[str, Any]]] = None,
    ):
        """Build Gemini contents with file_data parts for uploaded documents."""
        if not attachments:
            return prompt_text

        parts = []
        for attachment in attachments:
            if attachment.get("kind") != "gemini_file":
                continue
            file_uri = attachment.get("file_uri")
            mime_type = attachment.get("mime_type")
            if not file_uri or not mime_type:
                raise ValueError("Gemini attachment is missing file_uri or mime_type.")
            parts.append({
                "file_data": {
                    "mime_type": mime_type,
                    "file_uri": file_uri,
                }
            })

        if prompt_text:
            parts.append({"text": prompt_text})

        if not parts:
            return prompt_text

        return [{"role": "user", "parts": parts}]

    # ── Local LLM (Ollama / LM Studio via OpenAI-compat API) ────────────

    def _call_local(self, kwargs: dict) -> LLMResponse:
        """Translate responses.create() to OpenAI Chat Completions (local)."""
        client = self._llm._get_local_client()
        model = kwargs.get("model", self._llm.default_model)
        actual_model = model.removeprefix("local/")

        instructions = kwargs.get("instructions", "")
        input_data = kwargs.get("input", "")
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_output_tokens", 2000)
        tools_raw = kwargs.get("tools", None)
        json_mode = False
        text_opt = kwargs.get("text", None)
        if text_opt and isinstance(text_opt, dict):
            fmt = text_opt.get("format", {})
            if fmt.get("type") == "json_object":
                json_mode = True

        # Build messages
        messages = self._build_chat_messages(instructions, input_data, json_mode)

        # Translate tools
        openai_tools = None
        if tools_raw:
            openai_tools = self._translate_tools_for_chat_completions(tools_raw)

        call_kwargs = {
            "model": actual_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if openai_tools:
            call_kwargs["tools"] = openai_tools
        if json_mode:
            call_kwargs["response_format"] = {"type": "json_object"}

        resp = client.chat.completions.create(**call_kwargs)
        return self._chat_completion_to_llm_response(resp)

    def _stream_local(self, kwargs: dict):
        """Streaming local LLM call — returns an iterator of StreamEvents."""
        client = self._llm._get_local_client()
        model = kwargs.get("model", self._llm.default_model)
        actual_model = model.removeprefix("local/")

        instructions = kwargs.get("instructions", "")
        input_data = kwargs.get("input", "")
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_output_tokens", 2000)
        tools_raw = kwargs.get("tools", None)

        messages = self._build_chat_messages(instructions, input_data)
        openai_tools = self._translate_tools_for_chat_completions(tools_raw) if tools_raw else None

        call_kwargs = {
            "model": actual_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if openai_tools:
            call_kwargs["tools"] = openai_tools

        # Emit response.created
        resp_id = f"resp_{uuid.uuid4().hex[:16]}"
        yield StreamEvent(type="response.created", response=LLMResponse(id=resp_id))

        # Track function calls across chunks
        function_calls = {}  # index -> FunctionCallItem

        stream = client.chat.completions.create(**call_kwargs)
        for chunk in stream:
            choice = chunk.choices[0] if chunk.choices else None
            if not choice:
                continue

            delta = choice.delta

            # Text content
            if delta.content:
                yield StreamEvent(type="response.output_text.delta", delta=delta.content)

            # Tool calls
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in function_calls:
                        fc = FunctionCallItem(
                            name=tc.function.name or "",
                            call_id=tc.id or f"call_{uuid.uuid4().hex[:16]}",
                        )
                        function_calls[idx] = fc
                        yield StreamEvent(type="response.output_item.added", item=fc)

                    fc = function_calls[idx]
                    if tc.function.name:
                        fc.name = tc.function.name
                    if tc.function.arguments:
                        fc.arguments += tc.function.arguments
                        yield StreamEvent(
                            type="response.function_call_arguments.delta",
                            delta=tc.function.arguments,
                            item=fc,
                        )

    def _build_chat_messages(self, instructions: str, input_data, json_mode: bool = False) -> list:
        """Build Chat Completions messages from responses.create() args."""
        messages = []

        if instructions:
            sys_content = instructions
            if json_mode:
                sys_content += "\n\nYou MUST respond with valid JSON only. No extra text."
            messages.append({"role": "system", "content": sys_content})

        if isinstance(input_data, str):
            messages.append({"role": "user", "content": input_data})
        elif isinstance(input_data, list):
            # Tool results: [{"type": "function_call_output", "call_id": ..., "output": ...}]
            # In Chat Completions, these are "tool" role messages
            for item in input_data:
                if isinstance(item, dict) and item.get("type") == "function_call_output":
                    messages.append({
                        "role": "tool",
                        "tool_call_id": item.get("call_id", ""),
                        "content": item.get("output", ""),
                    })
        else:
            messages.append({"role": "user", "content": str(input_data)})

        return messages

    def _translate_tools_for_chat_completions(self, tools: list) -> list:
        """Convert Responses API flat tool format to Chat Completions nested format."""
        cc_tools = []
        for t in tools:
            if t.get("type") == "mcp":
                continue  # Skip MCP tools for non-OpenAI providers
            cc_tools.append({
                "type": "function",
                "function": {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {"type": "object", "properties": {}}),
                }
            })
        return cc_tools

    def _chat_completion_to_llm_response(self, resp) -> LLMResponse:
        """Convert Chat Completions response to LLMResponse."""
        choice = resp.choices[0] if resp.choices else None
        if not choice:
            return LLMResponse(output_text="")

        msg = choice.message
        output_text = msg.content or ""
        output_items = []

        # Handle tool calls
        if msg.tool_calls:
            for tc in msg.tool_calls:
                fc = FunctionCallItem(
                    name=tc.function.name,
                    arguments=tc.function.arguments or "",
                    call_id=tc.id,
                )
                output_items.append(fc)

        if not output_items:
            output_items = [MessageOutputItem(content=[TextContentItem(text=output_text)])]

        return LLMResponse(
            output_text=output_text,
            id=resp.id if hasattr(resp, 'id') else f"resp_{uuid.uuid4().hex[:16]}",
            output=output_items,
        )


# ---------------------------------------------------------------------------
# Main LLM Client
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Unified LLM client. Drop-in replacement for OpenAI() client.

    Exposes `client.responses.create(...)` via the ResponsesShim so all
    existing code works unchanged.  Routes to the correct provider based
    on model name.

    Usage:
        client = LLMClient(model="gpt-4o")
        resp = client.responses.create(model="gpt-4o", input="Hello", ...)
        print(resp.output_text)
    """

    def __init__(self, model: str = "gpt-4o"):
        self.default_model = model
        self._responses_shim = ResponsesShim(self)

        # Lazy-loaded provider clients
        self._openai_client = None
        self._anthropic_client = None
        self._google_client = None
        self._local_client = None

    @property
    def responses(self) -> ResponsesShim:
        """Provides `client.responses.create(...)` interface."""
        return self._responses_shim

    # ── Provider client getters (lazy init) ─────────────────────────────

    def _get_openai_client(self):
        """Get or create OpenAI client."""
        if self._openai_client is None:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY is required for OpenAI models. "
                    "Set it in .env or use a local model (prefix with 'local/')."
                )
            self._openai_client = OpenAI(api_key=api_key)
        return self._openai_client

    def _get_anthropic_client(self):
        """Get or create Anthropic client."""
        if self._anthropic_client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError("Install 'anthropic' package: pip install anthropic")
            api_key = os.getenv("ANTHROPIC_API_KEY", "")
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY is required for Claude models. "
                    "Set it in .env or use a local model (prefix with 'local/')."
                )
            self._anthropic_client = anthropic.Anthropic(api_key=api_key)
        return self._anthropic_client

    def _get_google_client(self):
        """Get or create Google GenAI client."""
        if self._google_client is None:
            try:
                from google import genai as ggenai
            except ImportError:
                raise ImportError("Install 'google-genai' package: pip install google-genai")
            api_key = os.getenv("GEMINI_API_KEY", "")
            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY is required for Gemini models. "
                    "Set it in .env or use a local model (prefix with 'local/')."
                )
            self._google_client = ggenai.Client(api_key=api_key)
        return self._google_client

    def _get_local_client(self):
        """Get or create local LLM client (Ollama / LM Studio)."""
        if self._local_client is None:
            from openai import OpenAI
            base_url = os.getenv("LOCAL_LLM_BASE_URL", "")
            if not base_url:
                raise ValueError(
                    "LOCAL_LLM_BASE_URL is required for local models. "
                    "Set it in .env (e.g. http://localhost:11434/v1 for Ollama)."
                )
            self._local_client = OpenAI(
                base_url=base_url,
                api_key="not-needed",  # Local servers don't require API keys
            )
        return self._local_client
