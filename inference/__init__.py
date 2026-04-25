"""Inference adapter layer — wraps the model under test.

Adapters expose a uniform predict(image_path) -> Prediction interface so
that the run_inference tool stays adapter-agnostic. The active adapter is
selected via:

  1. inference.set_default_adapter(adapter)           — explicit, programmatic
  2. AXIS_ADAPTER environment variable                 — picks adapter type
  3. AXIS_ADAPTER_<TYPE>_MODEL / _URL env vars         — per-adapter config

Falling through everything: HTTPAdapter pointed at localhost:8321.

Built-in adapters:
- HTTPAdapter: any OpenAI-compatible / custom HTTP endpoint
- MLXAdapter: in-process MLX (Apple Silicon)
- TransformersAdapter: HuggingFace transformers (PyTorch)
- AnthropicAdapter: Claude vision via Anthropic SDK
"""
from __future__ import annotations

import os

from inference.base import InferenceAdapter, Prediction
from inference.http import HTTPAdapter

__all__ = [
    "InferenceAdapter",
    "Prediction",
    "HTTPAdapter",
    "get_default_adapter",
    "set_default_adapter",
    "build_adapter",
]


_default_adapter: InferenceAdapter | None = None


def set_default_adapter(adapter: InferenceAdapter) -> None:
    """Register the adapter used by run_inference. Called from coordinator/CLI."""
    global _default_adapter
    _default_adapter = adapter


def get_default_adapter() -> InferenceAdapter:
    """Return the active adapter; build one from env if none set."""
    global _default_adapter
    if _default_adapter is None:
        _default_adapter = build_adapter()
    return _default_adapter


def build_adapter(kind: str | None = None) -> InferenceAdapter:
    """Construct an adapter from env vars.

    kind: "http" | "mlx" | "transformers" | "anthropic". Defaults to
    AXIS_ADAPTER, then "http".
    """
    kind = (kind or os.environ.get("AXIS_ADAPTER") or "http").lower()

    if kind == "http":
        url = os.environ.get("AXIS_ADAPTER_HTTP_URL", "http://localhost:8321/predict")
        return HTTPAdapter(url=url)

    if kind == "mlx":
        from inference.mlx import MLXAdapter
        model = os.environ.get("AXIS_ADAPTER_MLX_MODEL", "models/axis-mura-v1-4bit")
        return MLXAdapter(model_path=model)

    if kind == "transformers":
        from inference.transformers import TransformersAdapter
        model = os.environ.get("AXIS_ADAPTER_TRANSFORMERS_MODEL", "google/medgemma-1.5-4b-it")
        return TransformersAdapter(model_id=model)

    if kind == "anthropic":
        from inference.anthropic import AnthropicAdapter
        model = os.environ.get("AXIS_ADAPTER_ANTHROPIC_MODEL")
        return AnthropicAdapter(model=model)

    raise ValueError(
        f"Unknown adapter kind '{kind}'. Use one of: http, mlx, transformers, anthropic."
    )
