import asyncio
import os
from ai_sdk import gemini, generate_text, stream_text

from dotenv import load_dotenv

load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    print("Please set GEMINI_API_KEY environment variable.")
    exit(1)

async def main():
    model = gemini("gemini-2.5-flash")

    print("--- Generating Text ---")
    response = generate_text(
        model=model,
        prompt="Tell me a joke about programming.",
    )
    print(response.text)

    print("\n--- Streaming Text ---")
    result = stream_text(
        model=model,
        prompt="Write a haiku about Python. Make it LONG.",
    )
    async for delta in result.text_stream:
        print(delta, end="", flush=True)
    print()

if __name__ == "__main__":
    asyncio.run(main())
