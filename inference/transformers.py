"""HuggingFace Transformers adapter — local PyTorch inference.

Loads any vision-language model that AutoModelForImageTextToText supports
(MedGemma, etc.). Model identifier can be a HF hub ID or a local path.

Requires: pip install torch transformers accelerate pillow
"""
from __future__ import annotations

import sys
from pathlib import Path

from inference.base import InferenceAdapter, Prediction

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TransformersAdapter(InferenceAdapter):
    """Run a vision-language model locally via HuggingFace Transformers."""

    def __init__(self, model_id: str, prompt_version: int = 4):
        self._model_id = model_id
        self._prompt_version = prompt_version

    @property
    def name(self) -> str:
        return f"transformers({self._model_id})"

    def warmup(self) -> None:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from axis_detector import set_backend, _load_transformers
        set_backend("transformers", self._model_id)
        _load_transformers(self._model_id)

    def predict(self, image_path: str) -> Prediction:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from axis_detector import query_model, set_backend
        set_backend("transformers", self._model_id)

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
