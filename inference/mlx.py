"""MLX adapter — in-process inference on Apple Silicon via mlx-vlm.

Loads any vision-language model mlx-vlm can serve, runs the configured
prompt against each image, parses the JSON response into a Prediction.
Model loading is lazy and cached per-process across predict() calls.

Requires: pip install mlx-vlm
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from inference.base import InferenceAdapter, Prediction

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_PROMPT = """\
Classify the image. Respond ONLY with valid JSON, no other text:
{
  "label": 0 or 1,
  "confidence": 0.0 to 1.0,
  "findings": "one-sentence summary"
}
"""


class MLXAdapter(InferenceAdapter):
    """Run a vision-language model locally via mlx-vlm."""

    def __init__(self, model_path: str, prompt: str = DEFAULT_PROMPT):
        self._model_path = model_path
        self._prompt = prompt
        self._model = None
        self._processor = None

    @property
    def name(self) -> str:
        return f"mlx({self._model_path})"

    def warmup(self) -> None:
        if self._model is not None:
            return
        try:
            from mlx_vlm import load
        except ImportError as e:
            raise RuntimeError(
                "MLX adapter requires `mlx-vlm`. Install with: pip install mlx-vlm"
            ) from e
        path = str(Path(self._model_path).expanduser())
        self._model, self._processor = load(path)

    def predict(self, image_path: str) -> Prediction:
        if self._model is None:
            self.warmup()
        from mlx_vlm import generate

        p = Path(image_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p

        t0 = time.monotonic()
        response = generate(
            self._model, self._processor,
            image=[str(p)],
            prompt=self._prompt,
            max_tokens=256,
            verbose=False,
        )
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        text = response if isinstance(response, str) else getattr(response, "text", str(response))
        parsed = _parse_json(text)
        label = int(parsed.get("label", 0))
        return Prediction(
            label=label,
            confidence=float(parsed.get("confidence", 0.5)),
            findings=parsed.get("findings"),
            elapsed_ms=elapsed_ms,
        )


def _parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
