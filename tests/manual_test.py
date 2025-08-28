# ---------------------------------------------------------------------------
# Manual demo script – extended with *tool calling* showcase
# ---------------------------------------------------------------------------

import asyncio
import os
from typing import List, Optional

# Optional dependency – only used to load .env during local development.
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except ModuleNotFoundError:
    pass  # run fine without python-dotenv

from ai_sdk import (
    generate_text,
    stream_text,
    generate_object,
    stream_object,
    openai,
    tool,
    embed_many,
    cosine_similarity,
    anthropic,
    Agent,
)
from ai_sdk.types import CoreSystemMessage, CoreUserMessage, TextPart, AnyMessage
from ai_sdk.ui_stream import (
    UIStreamStartPart,
    UITextStartPart,
    UITextDeltaPart,
    UITextEndPart,
    UIFinishMessagePart,
    UIErrorPart,
    StartStepPart,
    FinishStepPart,
    ToolInputStartPart,
    ToolInputAvailablePart,
    ToolOutputAvailablePart,
)
from pydantic import BaseModel, Field


MODEL_ID = os.getenv("AI_SDK_TEST_MODEL", "claude-3-5-haiku-latest")


# ---------------------------------------------------------------------------
# Tool definition with Pydantic model – doubles an integer
# ---------------------------------------------------------------------------


class DoubleParams(BaseModel):
    x: int = Field(description="Integer to double")


def _double_exec(x: int) -> int:  # noqa: D401
    print(f"[DOUBLE_TOOL]Double called with {x}")
    return x * 2


double_tool = tool(
    name="double",
    description="Double the given integer.",
    parameters=DoubleParams,
    execute=_double_exec,
)


class FindCapitalParams(BaseModel):
    country: str = Field(description="Country to find the capital of")


def _find_capital_exec(country: str) -> str:  # noqa: D401
    print(f"[FIND_CAPITAL_TOOL] Find capital called with {country}")
    if country == "France":
        return "The capital of France is Paris."
    elif country == "Germany":
        return "The capital of Germany is Berlin."
    elif country == "Italy":
        return "The capital of Italy is Rome."
    elif country == "Spain":
        return "The capital of Spain is Madrid."
    else:
        return "I don't know the capital of that country."


find_capital_tool = tool(
    name="find_capital",
    description="Find the capital of the given country.",
    parameters=FindCapitalParams,
    execute=_find_capital_exec,
)


class CountLettersParams(BaseModel):
    text: str = Field(description="Text to count the letters of")


def _count_letters_exec(text: str) -> int:  # noqa: D401
    print(f"[COUNT_LETTERS_TOOL] Count letters called with {text}")
    return len(text)


count_letters_tool = tool(
    name="count_letters",
    description="Count the number of letters in the given text.",
    parameters=CountLettersParams,
    execute=_count_letters_exec,
)


async def demo_generate_prompt(model):
    print("\n-- Prompt-only generate_text --")
    res = generate_text(model=model, prompt="Say hello from the Python AI SDK port.")
    print("Text:", res.text)
    print("Usage:", res.usage)


async def demo_generate_messages(model):
    print("\n-- Message-based generate_text --")
    messages: List[AnyMessage] = [
        CoreSystemMessage(content="You are a helpful assistant."),
        CoreUserMessage(content=[TextPart(text="Respond with the single word 'ack'.")]),
    ]
    res = generate_text(model=model, messages=messages)
    print("Text:", res.text)


async def demo_stream(model):
    print("\n-- Streaming example --")
    result = stream_text(model=model, prompt="Tell a short Python joke.")
    collected = []
    async for delta in result.text_stream:
        print(delta, end="", flush=True)
        collected.append(delta)
    print()
    full = await result.text()
    print("Full:", full)
    assert full == "".join(collected)


async def demo_full_stream(model):
    print("\n-- Full stream (typed data stream) example --")
    result = stream_text(model=model, prompt="Tell a short Python joke.")

    if result.fullStream is None:
        print("fullStream not available for this result")
        return

    # Print only text deltas from the full, typed stream
    try:
        async for part in result.fullStream:
            if isinstance(part, UIStreamStartPart):
                pass  # could initialize UI state
            elif isinstance(part, StartStepPart):
                print("[start-step]")
            elif isinstance(part, ToolInputStartPart):
                print(f"[tool-input-start] {part.tool_name} {part.tool_call_id}")
            elif isinstance(part, ToolInputAvailablePart):
                print(
                    f"[tool-input-available] {part.tool_name} {part.tool_call_id} {part.input}"
                )
            elif isinstance(part, UITextStartPart):
                pass
            elif isinstance(part, UITextDeltaPart):
                print(part.delta, end="", flush=True)
            elif isinstance(part, UITextEndPart):
                print()
            elif isinstance(part, ToolOutputAvailablePart):
                print(f"[tool-output-available] {part.tool_call_id} {part.output}")
            elif isinstance(part, FinishStepPart):
                print("[finish-step]")
            elif isinstance(part, UIFinishMessagePart):
                pass
            elif isinstance(part, UIErrorPart):
                print(f"[error] {part.error_text}")
    except Exception as exc:  # noqa: BLE001
        print(f"[stream error] {exc}")
        return

    # Show assembled full text as well
    full = await result.text()
    print("Full:", full)


async def demo_tool_call(model):
    print("\n-- Tool calling example --")

    # Track iteration steps via callback for demonstration purposes.
    step_types = []

    def on_step(info):
        step_types.append(info.step_type)

    res = generate_text(
        model=model,
        prompt="Please double 7 using the tool.",
        tools=[double_tool],
        on_step=on_step,
    )

    print("Assistant response:", res.text)
    print("Tool steps executed:", step_types)


async def demo_tool_call_streaming(model):
    print("\n-- Tool calling example --")

    result = stream_text(
        model=model, prompt="Please double 7 using the tool.", tools=[double_tool]
    )
    collected = []
    async for delta in result.text_stream:
        print(delta, end="", flush=True)
        collected.append(delta)

    full = await result.text()
    print("Full:", full)
    assert full == "".join(collected)


class RandomNumberDetails(BaseModel):
    number: int
    is_even: bool
    factors: List[int]
    description: Optional[str] = None


# ---------------------------------------------------------------------------
# Object generation demos (complex schema)
# ---------------------------------------------------------------------------


async def demo_generate_object(model):
    print("\n-- Generate object example --")
    prompt = (
        'Respond with JSON like {"number": 57, "is_even": false, '
        '"factors": [1, 3, 19, 57], "description": "57 is an odd number. Its factors are 1, 3, 19, and 57."} (no markdown).'
    )
    res = generate_object(model=model, schema=RandomNumberDetails, prompt=prompt)
    print("Object:", res.object)


async def demo_stream_object(model):
    print("\n-- Stream object example --")
    prompt = (
        'Respond with JSON like {"number": 42, "is_even": true, '
        '"factors": [1, 2, 3, 6, 7, 14, 21, 42], "description": "42 is an even number. Its factors are 1, 2, 3, 6, 7, 14, 21, and 42."} (no markdown).'
    )

    def on_partial(obj):
        print("Partial:", obj)

    result = stream_object(
        model=model, schema=RandomNumberDetails, prompt=prompt, on_partial=on_partial
    )
    async for delta in result.object_stream:
        pass

    obj = await result.object(RandomNumberDetails)
    print("\nObject:", obj)


async def demo_embed(_):
    """Demonstrate embedding generation + cosine similarity."""
    print("\n-- Embedding example --")

    # Use a separate *embedding* model (text-embedding-3-small by default)
    EMBED_MODEL_ID = os.getenv("AI_SDK_EMBED_MODEL", "text-embedding-3-small")
    embedding_model = openai.embedding(EMBED_MODEL_ID)  # type: ignore

    values = [
        "cat",
        "dog",
        "aerospace engineer",
        "astronaut",
    ]

    res = embed_many(model=embedding_model, values=values)
    print("Embeddings lengths:", [len(e) for e in res.embeddings])

    sim = cosine_similarity(res.embeddings[0], res.embeddings[1])
    print("Cosine similarity of first two:", sim)
    sim2 = cosine_similarity(res.embeddings[0], res.embeddings[2])
    print("Cosine similarity of first and third:", sim2)
    sim3 = cosine_similarity(res.embeddings[2], res.embeddings[3])
    print("Cosine similarity of third and fourth:", sim3)


async def demo_agent(model):
    print("\n-- Agent example --")
    agent = Agent(
        name="Test Agent",
        model=model,
        tools=[find_capital_tool, count_letters_tool, double_tool],
    )

    print(
        agent.run(
            "Find the capitals of France, Germany, Italy, and Spain and return the double of the number of letters in each capital."
        )
    )


async def demo_full_stream_with_tools(model):
    print("\n-- Full stream with tools (typed data stream) --")
    result = stream_text(
        model=model,
        prompt="Please double 7 using the tool and explain briefly.",
        tools=[double_tool],
    )

    if result.fullStream is None:
        print("fullStream not available for this result")
        return

    try:
        async for part in result.fullStream:
            if isinstance(part, UIStreamStartPart):
                pass
            elif isinstance(part, StartStepPart):
                print("[start-step]")
            elif isinstance(part, ToolInputStartPart):
                print(f"[tool-input-start] {part.tool_name} {part.tool_call_id}")
            elif isinstance(part, ToolInputAvailablePart):
                print(
                    f"[tool-input-available] {part.tool_name} {part.tool_call_id} {part.input}"
                )
            elif isinstance(part, UITextStartPart):
                pass
            elif isinstance(part, UITextDeltaPart):
                print(part.delta, end="", flush=True)
            elif isinstance(part, UITextEndPart):
                print()
            elif isinstance(part, ToolOutputAvailablePart):
                print(f"[tool-output-available] {part.tool_call_id} {part.output}")
            elif isinstance(part, FinishStepPart):
                print("[finish-step]")
            elif isinstance(part, UIFinishMessagePart):
                pass
            elif isinstance(part, UIErrorPart):
                print(f"[error] {part.error_text}")
    except Exception as exc:  # noqa: BLE001
        print(f"[stream error] {exc}")
        return
    full = await result.text()
    print("Full:", full)


async def main():
    model = openai(MODEL_ID)
    # model = anthropic(MODEL_ID, api_key=os.getenv("ANTHROPIC_API_KEY"))
    # await demo_generate_prompt(model)
    # await demo_generate_messages(model)
    # await demo_stream(model)
    # await demo_tool_call(model)
    # await demo_tool_call_streaming(model)
    # await demo_generate_object(model)
    # await demo_stream_object(model)
    # await demo_embed(model)
    # await demo_agent(model)
    await demo_full_stream(model)
    await demo_full_stream_with_tools(model)


if __name__ == "__main__":
    asyncio.run(main())
