"""HuggingFace Transformers adapter — local PyTorch inference.

Loads any vision-language model that AutoModelForImageTextToText
supports. Model identifier can be a HF Hub ID or a local checkpoint path.
The configured prompt is sent with each image; the JSON response is
parsed into a Prediction. Model loading is lazy and cached.

Requires: pip install torch transformers accelerate pillow
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


class TransformersAdapter(InferenceAdapter):
    """Run a vision-language model locally via HuggingFace transformers."""

    def __init__(self, model_id: str, prompt: str = DEFAULT_PROMPT):
        self._model_id = model_id
        self._prompt = prompt
        self._model = None
        self._processor = None

    @property
    def name(self) -> str:
        return f"transformers({self._model_id})"

    def warmup(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForImageTextToText
        except ImportError as e:
            raise RuntimeError(
                "Transformers adapter requires torch + transformers. "
                "Install with: pip install torch transformers accelerate pillow"
            ) from e

        if torch.cuda.is_available():
            dtype = torch.float16
        elif torch.backends.mps.is_available():
            dtype = torch.float32  # MPS + float16 produces empty output for some VLMs
        else:
            dtype = torch.float32

        self._processor = AutoProcessor.from_pretrained(self._model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self._model_id,
            dtype=dtype,
            device_map="auto",
        )

    def predict(self, image_path: str) -> Prediction:
        if self._model is None:
            self.warmup()
        import torch
        from PIL import Image

        p = Path(image_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        image = Image.open(p).convert("RGB")

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": self._prompt},
            ],
        }]

        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)

        t0 = time.monotonic()
        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, max_new_tokens=256)
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        # Skip the prompt tokens
        prompt_len = inputs["input_ids"].shape[-1]
        new_tokens = output_ids[0][prompt_len:]
        text = self._processor.decode(new_tokens, skip_special_tokens=True)

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
