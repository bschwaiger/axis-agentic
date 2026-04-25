"""MLX adapter — in-process inference on Apple Silicon via mlx-vlm.

Wraps the existing axis_detector pathway so MURA-style binary
classification works out of the box. Loading is lazy and the model is
cached per-process by axis_detector's singletons.

Requires: pip install mlx-vlm
"""
from __future__ import annotations

import sys
from pathlib import Path

from inference.base import InferenceAdapter, Prediction

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class MLXAdapter(InferenceAdapter):
    """Run a vision-language model locally via mlx-vlm."""

    def __init__(self, model_path: str, prompt_version: int = 4):
        self._model_path = model_path
        self._prompt_version = prompt_version

    @property
    def name(self) -> str:
        return f"mlx({self._model_path})"

    def warmup(self) -> None:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from axis_detector import set_backend, _load_mlx
        set_backend("mlx", self._model_path)
        _load_mlx(self._model_path)

    def predict(self, image_path: str) -> Prediction:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from axis_detector import query_model, set_backend
        set_backend("mlx", self._model_path)

        p = Path(image_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p

        result = query_model(str(p), prompt_version=self._prompt_version)

        abnormal = result.get("abnormal")
        if abnormal is None:
            abnormal = result.get("fracture")
        label = 1 if abnormal else 0

        return Prediction(
            label=label,
            confidence=float(result.get("confidence", 0.0)),
            findings=result.get("findings"),
            elapsed_ms=int(result.get("_meta", {}).get("inference_time_s", 0) * 1000) or None,
        )
