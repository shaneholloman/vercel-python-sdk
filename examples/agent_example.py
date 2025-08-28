#!/usr/bin/env python3
"""
Agent Example - Demonstrates the Agent class with tool calling capabilities.

This example shows how to:
1. Create an agent with multiple tools
2. Use both synchronous and streaming interfaces
3. Monitor agent steps with callbacks
4. Handle different types of tools (Pydantic and JSON schema)
"""

import asyncio
from pydantic import BaseModel, Field
from ai_sdk import Agent, openai, tool
from ai_sdk.agent import print_step

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover

    def load_dotenv() -> None:  # type: ignore
        return None


from ai_sdk.tool import Tool

# Load environment variables from .env file
load_dotenv()


# Define tool parameters using Pydantic (recommended approach)
class CalculatorParams(BaseModel):
    operation: str = Field(
        description="Mathematical operation: add, subtract, multiply, divide"
    )
    a: float = Field(description="First number")
    b: float = Field(description="Second number")


def calculator(operation: str, a: float, b: float) -> float | str:
    """Perform basic mathematical operations."""
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            return "Error: Division by zero"
        return a / b
    else:
        return f"Error: Invalid operation '{operation}'"


calculator: Tool = tool(
    name="calculator",
    description="Perform basic mathematical operations",
    parameters=CalculatorParams,
    execute=calculator,
)


# Define tool using JSON schema (legacy approach)
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    # Simulate weather data
    weather_data = {
        "San Francisco": "Sunny and 65°F",
        "New York": "Cloudy and 72°F",
        "Tokyo": "Rainy and 68°F",
        "London": "Foggy and 55°F",
        "Sydney": "Clear and 75°F",
    }
    return weather_data.get(location, f"Weather data not available for {location}")


get_weather: Tool = tool(
    name="get_weather",
    description="Get weather information for a location",
    parameters={
        "type": "object",
        "properties": {"location": {"type": "string", "description": "City name"}},
        "required": ["location"],
    },
    execute=get_weather,
)


def get_time(timezone: str) -> str:
    """Get current time in a specific timezone."""
    import datetime
    from datetime import timezone as tz, timedelta

    # Simple timezone mapping
    tz_map = {
        "UTC": tz.utc,
        "America/New_York": tz(timedelta(hours=-5)),
        "America/Los_Angeles": tz(timedelta(hours=-8)),
        "Europe/London": tz(timedelta(hours=0)),
        "Asia/Tokyo": tz(timedelta(hours=9)),
    }

    target_tz = tz_map.get(timezone, tz.utc)
    current_time = datetime.datetime.now(target_tz)
    return f"Current time in {timezone}: {current_time.strftime('%Y-%m-%d %H:%M:%S')}"


get_time: Tool = tool(
    name="get_time",
    description="Get the current time in a specific timezone",
    parameters={
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "Timezone (e.g., 'UTC', 'America/New_York')",
            }
        },
        "required": ["timezone"],
    },
    execute=get_time,
)


def main():
    """Demonstrate basic agent usage."""
    print("🚀 Agent Example - Basic Usage")
    print("=" * 50)

    # Create an agent with multiple tools
    model = openai("gpt-4o-mini")
    agent = Agent(
        name="Multi-Tool Assistant",
        model=model,
        system="""You are a helpful assistant that can perform calculations, 
        provide weather information, and tell the time. Always be polite and 
        provide clear explanations for your responses.""",
        tools=[calculator, get_weather, get_time],
        on_step=print_step,
        max_steps=10,
    )

    # Example 1: Simple calculation
    print("\n📊 Example 1: Mathematical calculation")
    print("-" * 30)
    response = agent.run("What is 15 * 7?")
    print(f"Response: {response}")

    # Example 2: Weather information
    print("\n🌤️ Example 2: Weather information")
    print("-" * 30)
    response = agent.run("What's the weather like in Tokyo?")
    print(f"Response: {response}")

    # Example 3: Complex multi-tool query
    print("\n🔍 Example 3: Complex query with multiple tools")
    print("-" * 30)
    response = agent.run(
        "What's 25 * 4, and what's the weather like in San Francisco? Also, what time is it in UTC?"
    )
    print(f"Response: {response}")


async def streaming_example():
    """Demonstrate streaming with the agent."""
    print("\n\n🌊 Agent Example - Streaming")
    print("=" * 50)

    model = openai("gpt-4o-mini")
    agent = Agent(
        name="Streaming Assistant",
        model=model,
        system="You are a helpful assistant that can perform calculations and provide weather information.",
        tools=[calculator, get_weather],
        max_steps=5,
    )

    print("Streaming response to: 'What's 12 * 8 and what's the weather in London?'")
    print("-" * 60)

    stream = agent.stream("What's 12 * 8 and what's the weather in London?")

    print("Response: ", end="", flush=True)
    async for chunk in stream.text_stream:
        print(chunk, end="", flush=True)
    print()  # New line after streaming


def error_handling_example():
    """Demonstrate error handling with tools."""
    print("\n\n⚠️ Agent Example - Error Handling")
    print("=" * 50)

    model = openai("gpt-4o-mini")
    agent = Agent(
        name="Error Handling Assistant",
        model=model,
        system="You are a helpful assistant. If a tool returns an error, explain what went wrong.",
        tools=[calculator],
        max_steps=3,
    )

    # This should trigger division by zero error
    response = agent.run("What is 10 divided by 0?")
    print(f"Response: {response}")


if __name__ == "__main__":
    # Run basic examples
    main()

    # Run streaming example
    asyncio.run(streaming_example())

    # Run error handling example
    error_handling_example()

    print("\n✅ All examples completed!")
