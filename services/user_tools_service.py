"""
UserToolsService — per-user custom tool definitions and secrets.

Storage layout (all inside the project root):
    user_tools/
        <user_id>/
            tools.json    — list of tool definitions (name, desc, code, params, api_key_name)
            secrets.json  — {key_name: key_value}  (never committed — .gitignore)
"""

import json
import os
import inspect
import re
import types
from pathlib import Path
from typing import Any, Dict, List, Optional


# ─── Helpers ──────────────────────────────────────────────────────────────────

_BASE = Path(__file__).resolve().parent.parent / "user_tools"

def _user_dir(user_id: str) -> Path:
    p = _BASE / user_id
    p.mkdir(parents=True, exist_ok=True)
    return p

def _tools_file(user_id: str) -> Path:
    return _user_dir(user_id) / "tools.json"

def _secrets_file(user_id: str) -> Path:
    return _user_dir(user_id) / "secrets.json"


# ─── Validation ───────────────────────────────────────────────────────────────

TOOL_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,39}$")

def _validate_name(name: str):
    if not TOOL_NAME_RE.match(name):
        raise ValueError(
            f"Tool name '{name}' is invalid. Use lowercase letters, digits, and underscores only (max 40 chars, must start with a letter)."
        )

def _derive_schema(code: str, func_name: str) -> Dict[str, Any]:
    """
    Execute the function code in a sandbox and introspect its signature
    to auto-build an OpenAI-compatible JSON schema.

    Falls back to a permissive schema if introspection fails.
    """
    _PYTHON_TYPE_MAP = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "list": "array",
        "dict": "object",
        "NoneType": "null",
    }

    try:
        ns: Dict[str, Any] = {}
        exec(compile(code, "<user_tool>", "exec"), ns)  # noqa: S102
        fn = ns.get(func_name)
        if fn is None or not callable(fn):
            raise ValueError(f"Function '{func_name}' not found in code.")

        sig = inspect.signature(fn)
        properties: Dict[str, Any] = {}
        required: List[str] = []

        for param_name, param in sig.parameters.items():
            ann = param.annotation
            if ann is inspect.Parameter.empty:
                json_type = "string"
            else:
                type_name = ann.__name__ if hasattr(ann, "__name__") else str(ann)
                json_type = _PYTHON_TYPE_MAP.get(type_name, "string")

            properties[param_name] = {"type": json_type, "description": f"Parameter: {param_name}"}

            # Required if no default value
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    except Exception as e:  # noqa: BLE001
        # Fallback: minimal schema
        return {
            "type": "object",
            "properties": {},
            "required": [],
            "_schema_error": str(e),
        }


# ─── Service class ────────────────────────────────────────────────────────────

class UserToolsService:
    """CRUD operations for per-user custom tool definitions and secrets."""

    # ── Tools ──────────────────────────────────────────────────────────────

    def load_tools(self, user_id: str) -> List[Dict[str, Any]]:
        """Return all saved tool definitions for a user."""
        fp = _tools_file(user_id)
        if not fp.exists():
            return []
        try:
            return json.loads(fp.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []

    def save_tool(
        self,
        user_id: str,
        name: str,
        description: str,
        code: str,
        api_key_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Validate, derive schema, and persist a new (or updated) tool.

        Returns the saved tool definition dict.
        Raises ValueError on validation errors.
        """
        _validate_name(name)

        if not description.strip():
            raise ValueError("Description cannot be empty.")
        if not code.strip():
            raise ValueError("Code cannot be empty.")

        # Ensure the code actually defines the declared function name
        fn_pattern = re.compile(rf"^\s*def\s+{re.escape(name)}\s*\(", re.MULTILINE)
        if not fn_pattern.search(code):
            raise ValueError(
                f"Your code must define a function named exactly '{name}'. "
                "Make sure the function name matches the Tool name field."
            )

        schema = _derive_schema(code, name)

        tool_def: Dict[str, Any] = {
            "name": name,
            "description": description,
            "code": code,
            "parameters": schema,
            "api_key_name": api_key_name or "",
        }

        tools = self.load_tools(user_id)
        # Replace if exists
        tools = [t for t in tools if t.get("name") != name]
        tools.append(tool_def)
        _tools_file(user_id).write_text(
            json.dumps(tools, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return tool_def

    def delete_tool(self, user_id: str, tool_name: str) -> bool:
        """Remove a tool. Returns True if it was found and removed."""
        tools = self.load_tools(user_id)
        original_len = len(tools)
        tool = next((t for t in tools if t.get("name") == tool_name), None)

        # Remove associated secret if present
        if tool and tool.get("api_key_name"):
            self.delete_secret(user_id, tool["api_key_name"])

        tools = [t for t in tools if t.get("name") != tool_name]
        if len(tools) == original_len:
            return False

        _tools_file(user_id).write_text(
            json.dumps(tools, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return True

    # ── Secrets ────────────────────────────────────────────────────────────

    def load_secrets(self, user_id: str) -> Dict[str, str]:
        """Return all stored secrets for a user as {key_name: key_value}."""
        fp = _secrets_file(user_id)
        if not fp.exists():
            return {}
        try:
            return json.loads(fp.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def save_secret(self, user_id: str, key_name: str, key_value: str):
        """Persist an API key secret for a user."""
        secrets = self.load_secrets(user_id)
        secrets[key_name] = key_value
        _secrets_file(user_id).write_text(
            json.dumps(secrets, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def delete_secret(self, user_id: str, key_name: str) -> bool:
        """Remove a single secret. Returns True if found."""
        secrets = self.load_secrets(user_id)
        if key_name not in secrets:
            return False
        del secrets[key_name]
        _secrets_file(user_id).write_text(
            json.dumps(secrets, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return True

    def inject_secrets_to_env(self, user_id: str):
        """
        Inject user secrets into os.environ for the current process.
        Call this at agent startup so tool functions can read them via
        os.environ.get("MY_KEY").
        """
        for key, value in self.load_secrets(user_id).items():
            if key and value:
                os.environ.setdefault(key, value)

    # ── Runtime loading ────────────────────────────────────────────────────

    def build_callable(self, tool_def: Dict[str, Any]):
        """
        Execute tool code in a sandboxed namespace and return the callable.
        Raises RuntimeError if execution or lookup fails.
        """
        name = tool_def["name"]
        code = tool_def.get("code", "")
        ns: Dict[str, Any] = {
            "__builtins__": __builtins__,
            "os": os,
        }
        try:
            exec(compile(code, f"<user_tool:{name}>", "exec"), ns)  # noqa: S102
        except Exception as exc:
            raise RuntimeError(f"Failed to compile tool '{name}': {exc}") from exc

        fn = ns.get(name)
        if fn is None or not callable(fn):
            raise RuntimeError(
                f"Tool '{name}' compiled successfully but callable not found in namespace. "
                "Double-check the function name matches the Tool name exactly."
            )
        return fn
