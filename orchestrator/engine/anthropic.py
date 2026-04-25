"""Anthropic Claude implementation of the Engine interface."""
from __future__ import annotations

import os
import time
from typing import Any

from orchestrator.engine.base import Engine, ToolCall, ToolResult, TurnResponse


DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
RETRY_BASE_WAIT_S = 15
MAX_RETRIES = 3


class AnthropicEngine(Engine):
    """Engine backed by Anthropic's Messages API (tool_use content blocks)."""

    def __init__(self, model: str | None = None):
        import anthropic
        self._client = anthropic.Anthropic()
        self._model = model or DEFAULT_MODEL
        self._system: str = ""
        self._tools: list[dict[str, Any]] = []
        self._max_tokens: int = 4096
        self._messages: list[dict[str, Any]] = []

    @property
    def provider_name(self) -> str:
        return "Anthropic"

    @property
    def model_name(self) -> str:
        return self._model

    def reset(self, *, system: str, tools: list[dict[str, Any]], max_tokens: int = 4096) -> None:
        self._system = system
        self._tools = tools
        self._max_tokens = max_tokens
        self._messages = []

    def send_user_message(self, content: str) -> TurnResponse:
        self._messages.append({"role": "user", "content": content})
        return self._run_turn()

    def send_tool_results(self, results: list[ToolResult]) -> TurnResponse:
        tool_result_blocks = [
            {"type": "tool_result", "tool_use_id": r.call_id, "content": r.content}
            for r in results
        ]
        self._messages.append({"role": "user", "content": tool_result_blocks})
        return self._run_turn()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_turn(self) -> TurnResponse:
        t0 = time.monotonic()
        response = self._call_with_retry()
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        # Append assistant turn to history (provider-native content blocks)
        self._messages.append({"role": "assistant", "content": response.content})

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            btype = getattr(block, "type", "")
            if btype == "text":
                txt = getattr(block, "text", "")
                if txt:
                    text_parts.append(txt)
            elif btype == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input or {},
                ))

        usage = {}
        if response.usage:
            usage["input_tokens"] = response.usage.input_tokens
            usage["output_tokens"] = response.usage.output_tokens

        return TurnResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=response.stop_reason or "",
            usage=usage,
            elapsed_ms=elapsed_ms,
            raw=response,
        )

    def _call_with_retry(self):
        import anthropic
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                return self._client.messages.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    system=self._system,
                    messages=self._messages,
                    tools=self._tools,
                )
            except (anthropic.RateLimitError, anthropic.InternalServerError) as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    wait = RETRY_BASE_WAIT_S * (2 ** attempt)
                    status = getattr(e, "status_code", "?")
                    print(f"  ⏳ Rate limit ({status}), retrying in {wait}s "
                          f"(attempt {attempt + 1}/{MAX_RETRIES})...")
                    time.sleep(wait)
                else:
                    raise
        if last_error:
            raise last_error
        raise RuntimeError("call_with_retry exhausted without exception (unreachable)")
