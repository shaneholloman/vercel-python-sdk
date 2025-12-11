from __future__ import annotations

import os
from typing import Any, AsyncIterator, Dict, List, Optional
from openai import OpenAI  # type: ignore[import]

from .language_model import LanguageModel
from .openai import _build_chat_messages


class GeminiModel(LanguageModel):
    """OpenAI SDK compatibility provider for Google Gemini models."""

    def __init__(
        self,
        model: str,
        *,
        api_key: Optional[str] = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/",
        **default_kwargs: Any,
    ) -> None:
        # Use OpenAI SDK to talk to Gemini endpoint
        # Resolve API key from argument or environment variable
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._default_kwargs = default_kwargs

    def generate_text(
        self,
        *,
        prompt: str | None = None,
        system: str | None = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate a completion via the OpenAI SDK compatibility layer against Gemini."""
        if prompt is None and not messages:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

        # Build chat messages using OpenAI helper
        chat_messages = _build_chat_messages(
            prompt=prompt, system=system, messages=messages
        )

        # Merge default kwargs with call-site overrides
        request_kwargs: Dict[str, Any] = {**self._default_kwargs, **kwargs}

        # Call via OpenAI SDK
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=chat_messages,
            **request_kwargs,
        )

        choice = resp.choices[0]
        text = choice.message.content or ""
        finish_reason = choice.finish_reason or "unknown"

        # Extract tool_calls if present
        tool_calls = []
        if getattr(choice.message, "tool_calls", None):
            import json as _json

            for call in choice.message.tool_calls:  # type: ignore[attr-defined]
                try:
                    args = _json.loads(call.function.arguments)
                except Exception:
                    args = {"raw": call.function.arguments}
                tool_calls.append(
                    {
                        "tool_call_id": call.id,
                        "tool_name": call.function.name,
                        "args": args,
                    }
                )
            finish_reason = "tool"

        # Usage if available
        usage = resp.usage.model_dump() if hasattr(resp, "usage") else None
        return {
            "text": text,
            "finish_reason": finish_reason,
            "usage": usage,
            "raw_response": resp,
            "tool_calls": tool_calls or None,
        }

    def stream_text(
        self,
        *,
        prompt: str | None = None,
        system: str | None = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream deltas via OpenAI SDK compatibility."""
        if prompt is None and not messages:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

        # Build messages and merge kwargs
        chat_messages = _build_chat_messages(
            prompt=prompt, system=system, messages=messages
        )
        request_kwargs: Dict[str, Any] = {**self._default_kwargs, **kwargs}

        # Use AsyncOpenAI for streaming to avoid threading issues
        from openai import AsyncOpenAI

        async_client = AsyncOpenAI(
            api_key=self._client.api_key, base_url=self._client.base_url
        )

        async def _generator() -> AsyncIterator[str]:
            stream = await async_client.chat.completions.create(
                model=self._model,
                messages=chat_messages,
                stream=True,
                **request_kwargs,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    yield content

            await async_client.close()

        return _generator()


# Public factory helper
def gemini(
    model: str,
    *,
    api_key: Optional[str] = None,
    base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/",
    **default_kwargs: Any,
) -> GeminiModel:
    """Return a configured GeminiModel instance using OpenAI SDK compatibility."""
    return GeminiModel(model, api_key=api_key, base_url=base_url, **default_kwargs)
