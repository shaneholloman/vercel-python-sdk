import os
import pytest
from ai_sdk import gemini, generate_text, stream_text, generate_object
from dotenv import load_dotenv
from ai_sdk.tool import tool
from pydantic import BaseModel, Field

load_dotenv()

# Skip all tests if GEMINI_API_KEY is missing
pytestmark = pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY environment variable not set",
)

MODEL_ID = "gemini-2.5-flash"


@pytest.fixture
def model():
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set")
    return gemini(MODEL_ID)


@pytest.mark.asyncio
async def test_gemini_generate_text(model):
    """Basic prompt-only generation."""
    res = generate_text(model=model, prompt="Say 'hello' and nothing else.")
    assert "hello" in res.text.lower()
    assert res.finish_reason is not None
    assert res.usage is not None


@pytest.mark.asyncio
async def test_gemini_stream_text(model):
    """Streaming generation."""
    result = stream_text(model=model, prompt="Count to 3.")
    collected = []
    async for delta in result.text_stream:
        collected.append(delta)
    full_text = await result.text()
    assert len(collected) > 0
    assert len(full_text) > 0


class MathResponse(BaseModel):
    answer: int
    steps: str


@pytest.mark.asyncio
async def test_gemini_generate_object(model):
    """Structured output generation."""
    res = generate_object(
        model=model,
        schema=MathResponse,
        prompt="What is 2 + 2?",
    )
    assert isinstance(res.object, MathResponse)
    assert res.object.answer == 4


class AddParams(BaseModel):
    a: float = Field(description="First number")
    b: float = Field(description="Second number")


@tool(name="add", description="Add two numbers.", parameters=AddParams)
def add_tool(a: float, b: float) -> float:
    return a + b


@pytest.mark.asyncio
async def test_gemini_tool_call(model):
    """Tool call generation."""
    res = generate_text(
        model=model,
        prompt="What is 3+3?",
        tools=[
            add_tool,
        ],
    )
    assert "6" in res.text
