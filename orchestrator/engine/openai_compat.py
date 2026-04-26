"""OpenAI-compatible Engine.

Covers:
- OpenAI itself (default base_url)
- Ollama, LM Studio (point at http://localhost:11434/v1 with any model)
- NVIDIA NIM (https://integrate.api.nvidia.com/v1, e.g. Nemotron)
- Together, Groq, Replicate, etc.
- Any private endpoint that speaks OpenAI's chat-completions API

Translates between our Anthropic-style tool specs (name + input_schema)
and OpenAI's function-calling format inside the Engine; the agent layer
doesn't see the difference.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

from orchestrator.engine.base import Engine, ToolCall, ToolResult, TurnResponse


class OpenAICompatEngine(Engine):
    """Engine backed by any OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        provider_name: str | None = None,
    ):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "OpenAI-compatible engine requires the `openai` package. "
                "Run: pip install -r requirements.txt  (or pip install 'openai>=1.50.0'). "
                f"Original error: {e}"
            ) from e
        # Many local servers (Ollama) ignore the key; pass a placeholder if missing.
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY") or "not-needed"
        self._client = OpenAI(base_url=base_url, api_key=resolved_key)
        self._model = model
        self._base_url = base_url
        self._provider = provider_name or _infer_provider(base_url)

        self._system: str = ""
        self._tools: list[dict[str, Any]] = []
        self._max_tokens: int = 4096
        self._messages: list[dict[str, Any]] = []

    @property
    def provider_name(self) -> str:
        return self._provider

    @property
    def model_name(self) -> str:
        return self._model

    def reset(self, *, system: str, tools: list[dict[str, Any]], max_tokens: int = 4096) -> None:
        self._system = system
        self._tools = [_to_openai_tool(t) for t in tools]
        self._max_tokens = max_tokens
        self._messages = [{"role": "system", "content": system}] if system else []

    def send_user_message(self, content: str) -> TurnResponse:
        self._messages.append({"role": "user", "content": content})
        return self._run_turn()

    def send_tool_results(self, results: list[ToolResult]) -> TurnResponse:
        for r in results:
            self._messages.append({
                "role": "tool",
                "tool_call_id": r.call_id,
                "content": r.content,
            })
        return self._run_turn()

    # ------------------------------------------------------------------

    def _run_turn(self) -> TurnResponse:
        t0 = time.monotonic()
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": self._messages,
            "max_tokens": self._max_tokens,
        }
        if self._tools:
            kwargs["tools"] = self._tools
            kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**kwargs)
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        choice = response.choices[0]
        msg = choice.message
        text = msg.content or ""

        tool_calls: list[ToolCall] = []
        raw_tool_calls = msg.tool_calls or []
        for tc in raw_tool_calls:
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=args,
            ))

        # Append the assistant turn so subsequent send_tool_results can attach.
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": text}
        if raw_tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in raw_tool_calls
            ]
        self._messages.append(assistant_msg)

        usage: dict[str, int] = {}
        if response.usage:
            usage["input_tokens"] = response.usage.prompt_tokens
            usage["output_tokens"] = response.usage.completion_tokens

        return TurnResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=choice.finish_reason or "",
            usage=usage,
            elapsed_ms=elapsed_ms,
            raw=response,
        )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _to_openai_tool(t: dict[str, Any]) -> dict[str, Any]:
    """Convert one Anthropic-style tool spec into OpenAI function-calling format."""
    return {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
        },
    }


def _infer_provider(base_url: str | None) -> str:
    if not base_url:
        return "OpenAI"
    bu = base_url.lower()
    if "openai.com" in bu:
        return "OpenAI"
    if "nvidia" in bu:
        return "Nemotron"
    if "11434" in bu or "ollama" in bu:
        return "Ollama"
    if "lmstudio" in bu or "1234" in bu:
        return "LM Studio"
    if "together" in bu:
        return "Together"
    if "groq" in bu:
        return "Groq"
    return "OpenAI-compatible"
