"""Inference adapter layer — wraps the model under test.

Adapters expose a uniform predict(image_path) -> Prediction interface so
that the run_inference tool stays adapter-agnostic. The active adapter is
selected at coordinator startup; tools look it up via get_default_adapter().

Built-in adapters (more in PR2):
- HTTPAdapter: OpenAI-compatible / custom HTTP endpoint
"""
from inference.base import InferenceAdapter, Prediction
from inference.http import HTTPAdapter

__all__ = [
    "InferenceAdapter",
    "Prediction",
    "HTTPAdapter",
    "get_default_adapter",
    "set_default_adapter",
]


_default_adapter: InferenceAdapter | None = None


def set_default_adapter(adapter: InferenceAdapter) -> None:
    """Register the adapter used by run_inference. Called from coordinator/CLI."""
    global _default_adapter
    _default_adapter = adapter


def get_default_adapter() -> InferenceAdapter:
    """Return the active adapter, defaulting to HTTPAdapter on localhost:8321."""
    global _default_adapter
    if _default_adapter is None:
        _default_adapter = HTTPAdapter(url="http://localhost:8321/predict")
    return _default_adapter
