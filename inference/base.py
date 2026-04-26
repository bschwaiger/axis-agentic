"""InferenceAdapter ABC + Prediction value type."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Prediction:
    """One model output for a single image."""
    label: int           # 0/1 for binary; class index for multi-class
    confidence: float    # [0.0, 1.0]
    findings: str | None = None
    elapsed_ms: int | None = None


class InferenceAdapter(ABC):
    """Adapter for the model under test.

    Built-in adapters cover the common cases (HTTP, MLX local, transformers
    local, Anthropic SDK). Custom adapters subclass this and register via
    inference.set_default_adapter() before pipeline start.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def predict(self, image_path: str) -> Prediction:
        """Run the model on a single image. May raise on error."""
        ...

    def warmup(self) -> None:
        """Optional: load weights / open connection. Default: no-op."""
        return

    def close(self) -> None:
        """Optional: release resources. Default: no-op."""
        return
