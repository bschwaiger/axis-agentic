"""LLM Engine abstraction for the agentic pipeline.

An Engine wraps a provider (Anthropic, OpenAI-compatible, ...) and exposes
a uniform turn-based conversation interface. Agents are written against
the Engine contract and stay provider-agnostic.

Selection (in order of priority):
  1. orchestrator.engine.set_default_engine(engine)         — explicit
  2. AXIS_ENGINE env var ("anthropic" | "openai-compat")    — picks type
  3. Auto-detect from available API keys / base URLs

Built-in engines:
- AnthropicEngine
- OpenAICompatEngine (OpenAI / Ollama / Nemotron / LM Studio / Together / Groq)
"""
from __future__ import annotations

import os

from orchestrator.engine.base import Engine, ToolCall, ToolResult, TurnResponse
from orchestrator.engine.anthropic import AnthropicEngine
from orchestrator.engine.openai_compat import OpenAICompatEngine

__all__ = [
    "Engine",
    "ToolCall",
    "ToolResult",
    "TurnResponse",
    "AnthropicEngine",
    "OpenAICompatEngine",
    "get_default_engine",
    "set_default_engine",
    "build_engine",
]


_default_engine: Engine | None = None


def set_default_engine(engine: Engine) -> None:
    """Override the auto-selected engine — for tests or explicit configs."""
    global _default_engine
    _default_engine = engine


def get_default_engine() -> Engine:
    """Return the active engine; build one from env if none set."""
    global _default_engine
    if _default_engine is None:
        _default_engine = build_engine()
    return _default_engine


def build_engine(kind: str | None = None) -> Engine:
    """Construct an engine from env vars.

    kind: "anthropic" | "openai-compat". Defaults to AXIS_ENGINE; if unset,
    auto-detects from credentials.
    """
    kind = (kind or os.environ.get("AXIS_ENGINE") or "").lower()

    if not kind:
        kind = _auto_detect_kind()

    if kind == "anthropic":
        return AnthropicEngine()

    if kind in {"openai", "openai-compat", "openai_compat"}:
        base_url = os.environ.get("AXIS_ENGINE_OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        api_key = os.environ.get("AXIS_ENGINE_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        model = os.environ.get("AXIS_ENGINE_OPENAI_MODEL") or os.environ.get("OPENAI_MODEL")
        if not model:
            raise RuntimeError(
                "OpenAI-compatible engine requires AXIS_ENGINE_OPENAI_MODEL "
                "(or OPENAI_MODEL). Example: 'gpt-4o', 'qwen2.5', 'llama3.1'."
            )
        return OpenAICompatEngine(model=model, base_url=base_url, api_key=api_key)

    raise ValueError(f"Unknown engine kind '{kind}'. Use one of: anthropic, openai-compat.")


def _auto_detect_kind() -> str:
    """Pick an engine kind from available credentials. Anthropic preferred."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("AXIS_ENGINE_OPENAI_BASE_URL"):
        return "openai-compat"
    raise RuntimeError(
        "No engine credentials found. Set ANTHROPIC_API_KEY in .env, "
        "or set AXIS_ENGINE=openai-compat with OPENAI_API_KEY (or AXIS_ENGINE_OPENAI_BASE_URL "
        "for a local server like Ollama)."
    )
