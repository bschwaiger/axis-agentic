"""Anthropic vision adapter — Claude as the model under test.

Calls the Anthropic Messages API with an image content block. The system
prompt asks for a strict JSON binary classification so output parses
into a Prediction. Model ID defaults to claude-sonnet-4-6.

Useful for benchmarking your own model against a frontier vision-language
baseline as part of a comparative evaluation run.

Requires ANTHROPIC_API_KEY in env.
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
from pathlib import Path

from inference.base import InferenceAdapter, Prediction

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL = os.environ.get("ANTHROPIC_VISION_MODEL", "claude-sonnet-4-6")

SYSTEM_PROMPT = """\
You are an image classifier. For each image, decide whether the target \
class is present (label 1) or absent (label 0). Provide a confidence \
score and a one-sentence finding.

Respond ONLY with valid JSON, no other text:
{
  "label": 0 or 1,
  "confidence": 0.0 to 1.0,
  "findings": "one-sentence summary"
}
"""


class AnthropicAdapter(InferenceAdapter):
    """Run Claude vision as the model under test."""

    def __init__(self, model: str | None = None):
        import anthropic
        self._client = anthropic.Anthropic()
        self._model = model or DEFAULT_MODEL

    @property
    def name(self) -> str:
        return f"anthropic({self._model})"

    def predict(self, image_path: str) -> Prediction:
        p = Path(image_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p

        media_type = self._media_type(p.suffix.lower())
        with open(p, "rb") as f:
            data = base64.standard_b64encode(f.read()).decode("utf-8")

        t0 = time.monotonic()
        response = self._client.messages.create(
            model=self._model,
            max_tokens=400,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": media_type, "data": data,
                    }},
                    {"type": "text", "text": "Analyze this X-ray."},
                ],
            }],
        )
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
        parsed = self._parse_json(text)
        label = int(parsed.get("label", 0))

        return Prediction(
            label=label,
            confidence=float(parsed.get("confidence", 0.5)),
            findings=parsed.get("findings"),
            elapsed_ms=elapsed_ms,
        )

    @staticmethod
    def _media_type(suffix: str) -> str:
        return {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(suffix, "image/png")

    @staticmethod
    def _parse_json(text: str) -> dict:
        # Models sometimes wrap JSON in fences or add prose; extract the first {...} block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
