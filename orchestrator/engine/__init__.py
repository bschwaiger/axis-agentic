"""LLM Engine abstraction for the agentic pipeline.

An Engine wraps a provider (Anthropic, OpenAI-compatible, ...) and exposes
a uniform turn-based conversation interface. Agents are written against
the Engine contract and stay provider-agnostic.
"""
from orchestrator.engine.base import Engine, ToolCall, ToolResult, TurnResponse
from orchestrator.engine.anthropic import AnthropicEngine

__all__ = [
    "Engine",
    "ToolCall",
    "ToolResult",
    "TurnResponse",
    "AnthropicEngine",
]


def get_default_engine() -> Engine:
    """Return the default engine based on environment.

    Order of preference:
    1. ANTHROPIC_API_KEY set -> AnthropicEngine
    (More providers will be added in PR2: OpenAI, Nemotron, Ollama.)
    """
    import os
    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicEngine()
    raise RuntimeError(
        "No engine credentials found. Set ANTHROPIC_API_KEY in .env."
    )
