"""
Smoke test for services/user_tools_service.py
Run with: python scripts/test_user_tools.py
"""
import sys
import os
import importlib.util

# Import directly from the module file to bypass services/__init__.py
# (which drags in pyvo/cgi that may not be available in all envs)
_root = os.path.join(os.path.dirname(__file__), "..")
_spec = importlib.util.spec_from_file_location(
    "user_tools_service",
    os.path.join(_root, "services", "user_tools_service.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
UserToolsService = _mod.UserToolsService

svc = UserToolsService()
TEST_USER = "_test_smoke_user_"

# --- Cleanup from any previous run ---
try:
    import shutil
    from pathlib import Path
    shutil.rmtree(Path(__file__).resolve().parent.parent / "user_tools" / TEST_USER, ignore_errors=True)
except Exception:
    pass

print("=== UserToolsService smoke test ===\n")

# 1. Empty state
tools = svc.load_tools(TEST_USER)
assert tools == [], f"Expected empty list, got {tools}"
print("[PASS] 1. Empty tool list on fresh user")

# 2. Save a simple tool
SIMPLE_CODE = '''
def say_hello(name: str) -> dict:
    """Says hello to the given name."""
    return {"greeting": f"Hello, {name}!"}
'''
saved = svc.save_tool(TEST_USER, name="say_hello", description="Says hello.", code=SIMPLE_CODE)
assert saved["name"] == "say_hello"
tools = svc.load_tools(TEST_USER)
assert len(tools) == 1 and tools[0]["name"] == "say_hello"
print("[PASS] 2. Save and reload a simple tool")

# 3. Auto-derived schema has correct parameter
schema = saved["parameters"]
assert "name" in schema.get("properties", {}), f"Expected 'name' param, got {schema}"
assert "name" in schema.get("required", [])
print("[PASS] 3. Auto-schema derivation from type hints")

# 4. Build callable and run it
fn = svc.build_callable(saved)
result = fn(name="Astronomer")
assert result == {"greeting": "Hello, Astronomer!"}, f"Unexpected result: {result}"
print("[PASS] 4. Build callable and execute")

# 5. Save and load secrets
svc.save_secret(TEST_USER, "MY_TEST_KEY", "supersecret")
secrets = svc.load_secrets(TEST_USER)
assert secrets.get("MY_TEST_KEY") == "supersecret"
print("[PASS] 5. Save and load API key secret")

# 6. Inject secrets into env
svc.inject_secrets_to_env(TEST_USER)
assert os.environ.get("MY_TEST_KEY") == "supersecret"
print("[PASS] 6. Inject secrets into os.environ")

# 7. Delete tool also removes secret
tool2_code = '''
def tool_with_key(query: str) -> dict:
    """Requires MY_TEST_KEY."""
    return {"ok": True}
'''
svc.save_tool(TEST_USER, name="tool_with_key", description="test", code=tool2_code, api_key_name="MY_TEST_KEY")
svc.save_secret(TEST_USER, "MY_TEST_KEY", "supersecret")
removed = svc.delete_tool(TEST_USER, "tool_with_key")
assert removed is True
secrets_after = svc.load_secrets(TEST_USER)
assert "MY_TEST_KEY" not in secrets_after, f"Secret should have been deleted, got {secrets_after}"
print("[PASS] 7. Delete tool removes associated API key")

# 8. Validation rejects bad names
try:
    svc.save_tool(TEST_USER, name="Bad Name!", description="x", code="def Bad Name(): pass")
    assert False, "Should have raised ValueError"
except ValueError:
    pass
print("[PASS] 8. Tool name validation rejects invalid names")

# 9. Validation rejects mismatched function name
try:
    svc.save_tool(TEST_USER, name="my_tool", description="x", code="def other_name(): pass")
    assert False, "Should have raised ValueError"
except ValueError:
    pass
print("[PASS] 9. Validation catches function name mismatch")

# 10. Delete remaining tool
svc.delete_tool(TEST_USER, "say_hello")
assert svc.load_tools(TEST_USER) == []
print("[PASS] 10. Delete last tool — list is empty again")

# Cleanup
import shutil
from pathlib import Path
shutil.rmtree(Path(__file__).resolve().parent.parent / "user_tools" / TEST_USER, ignore_errors=True)

print("\n=== All 10 tests passed! ===")
