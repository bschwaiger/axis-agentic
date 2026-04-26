"""Engine ABC + value types for the LLM-with-tools abstraction.

The Engine owns conversation state internally; agents see a normalized
turn-based view (text + tool_calls + stop_reason). Each provider
(Anthropic, OpenAI-compatible, ...) gets its own Engine subclass that
translates provider-native message formats to/from this view.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """A tool invocation requested by the model."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """A tool execution result to send back to the model."""
    call_id: str
    content: str  # JSON string


@dataclass
class TurnResponse:
    """One turn of model output."""
    text: str
    tool_calls: list[ToolCall]
    stop_reason: str
    usage: dict[str, int] = field(default_factory=dict)
    elapsed_ms: int = 0
    raw: Any = None


class Engine(ABC):
    """Abstraction over an LLM provider with tool-use.

    Lifecycle: reset(system, tools) -> send_user_message(...) ->
    send_tool_results(...) (repeated) until TurnResponse.tool_calls is empty.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    @abstractmethod
    def reset(self, *, system: str, tools: list[dict[str, Any]], max_tokens: int = 4096) -> None:
        """Start a fresh conversation. Subsequent send_* calls use this state."""
        ...

    @abstractmethod
    def send_user_message(self, content: str) -> TurnResponse:
        """Append a user text message and run one model turn."""
        ...

    @abstractmethod
    def send_tool_results(self, results: list[ToolResult]) -> TurnResponse:
        """Append tool execution results and run one model turn."""
        ...
