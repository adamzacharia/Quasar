import pytest
from core.prompts.lit_to_code import LIT_TO_CODE_PROMPT

def test_lit_to_code_prompt_content():
    """Verify that the literature-to-code prompt is defined and has substantive content."""
    assert LIT_TO_CODE_PROMPT is not None, "LIT_TO_CODE_PROMPT should not be None"
    assert len(LIT_TO_CODE_PROMPT) > 100, "LIT_TO_CODE_PROMPT should have substantive content"
    assert "{methodology_text}" in LIT_TO_CODE_PROMPT, "LIT_TO_CODE_PROMPT should contain the {methodology_text} placeholder"
    assert "Python script" in LIT_TO_CODE_PROMPT or "CASA" in LIT_TO_CODE_PROMPT, "Prompt should ask for a Python or CASA script"
