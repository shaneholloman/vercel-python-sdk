import json
from typing import Any, Dict, List

try:
    import pytest  # type: ignore
except Exception:  # pragma: no cover

    class pytest:  # type: ignore
        @staticmethod
        def raises(_exc):
            class _Ctx:
                def __enter__(self):
                    return None

                def __exit__(self, *args):
                    return True

            return _Ctx()


from pydantic import BaseModel, Field

from ai_sdk import generate_text, tool
from ai_sdk.providers.language_model import LanguageModel


# ---------------------------------------------------------------------------
# Dummy provider – emulates tool calling behaviour without external network
# ---------------------------------------------------------------------------


class DummyModel(LanguageModel):
    """A minimalistic provider that triggers exactly **one** tool call and then
    returns the tool result on the next invocation.  Suitable for unit tests
    without touching real endpoints."""

    def __init__(self):
        self._call_count = 0

    # The interface purposefully ignores *system* / *prompt* – the control
    # flow is driven exclusively by the number of invocations.
    def generate_text(
        self,
        *,
        prompt: str | None = None,
        system: str | None = None,
        messages: List[Dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        self._call_count += 1

        # 1st call → ask for the ``double`` tool.
        if self._call_count == 1:
            return {
                "text": "",  # no text yet – model wants to call a tool
                "finish_reason": "tool",
                "tool_calls": [
                    {
                        "tool_call_id": "call-1",
                        "tool_name": "double",
                        "args": {"x": 5},
                    }
                ],
                "usage": {},
            }

        # 2nd call → model has received tool result, now finishes.
        elif self._call_count == 2:
            # The dummy model *echoes* whatever the tool result was.  In a real
            # conversation the LLM would continue reasoning here.
            last_tool_msg = (
                messages[-1] if messages else {"role": "tool", "content": []}
            )
            # Handle both dict and object access for robust testing
            content = last_tool_msg.get("content") if isinstance(last_tool_msg, dict) else last_tool_msg.content
            
            # Content might be a list of ToolResult objects or dicts
            first_content = content[0]
            if hasattr(first_content, "result"):
                result_val = first_content.result
            elif isinstance(first_content, dict):
                result_val = first_content["result"]
            else:
                # Fallback if it's already the result value (simpler mocks)
                result_val = first_content
                
            # If result is json string, load it
            if isinstance(result_val, str):
                try:
                    result_value = json.loads(result_val)
                except Exception:
                    result_value = result_val
            else:
                result_value = result_val

            return {
                "text": str(result_value),
                "finish_reason": "stop",
                "usage": {},
            }

        # Should never be reached in the current test setup.
        raise RuntimeError("DummyModel called too many times")

    # Streaming not required for these tests.
    async def stream_text(self, **kwargs):  # type: ignore[override]
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Local tool implementation with Pydantic model
# ---------------------------------------------------------------------------


class DoubleParams(BaseModel):
    x: int = Field(description="Integer to double")


def _double_fn(x: int) -> int:
    return x * 2


double_tool = tool(
    name="double",
    description="Double an integer.",
    parameters=DoubleParams,
    execute=_double_fn,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_generate_text_tool_call_happy_path():
    model = DummyModel()
    res = generate_text(model=model, prompt="ignored", tools=[double_tool])
    assert res.text == "10"
    # Ensure the tool result got recorded on the response object.
    assert res.tool_results and res.tool_results[0].result == 10


def test_on_step_invocation():
    model = DummyModel()
    step_types: List[str] = []

    def _cb(info):
        step_types.append(info.step_type)

    _ = generate_text(
        model=model, prompt="irrelevant", tools=[double_tool], on_step=_cb
    )
    # Expected sequence: initial -> tool-result -> continue
    assert step_types == ["initial", "tool-result", "continue"]


def test_max_steps_guard():
    class BrokenModel(DummyModel):
        """Model that *keeps* requesting a tool each time → should hit limit."""

        def generate_text(self, **kwargs):  # type: ignore[override]
            self._call_count += 1
            return {
                "text": "",
                "finish_reason": "tool",
                "tool_calls": [
                    {
                        "tool_call_id": f"call-{self._call_count}",
                        "tool_name": "double",
                        "args": {"x": 1},
                    }
                ],
                "usage": {},
            }

    broken = BrokenModel()
    with pytest.raises(RuntimeError):
        generate_text(
            model=broken, prompt="irrelevant", tools=[double_tool], max_steps=3
        )


def test_tool_with_pydantic_model():
    """Test that tool with Pydantic model works correctly."""
    assert double_tool.name == "double"
    assert double_tool.description == "Double an integer."
    assert double_tool._pydantic_model == DoubleParams

    # Test that the JSON schema was generated correctly
    schema = double_tool.parameters
    assert schema["type"] == "object"
    assert "x" in schema["properties"]
    assert "x" in schema["required"]


@pytest.mark.asyncio
async def test_tool_execution_with_validation():
    """Test that tool execution validates inputs against Pydantic model."""
    # Valid input
    result = await double_tool.run(x=5)
    assert result == 10

    # Invalid input should raise validation error
    with pytest.raises(Exception):  # Pydantic validation error
        await double_tool.run(x="not an integer")
