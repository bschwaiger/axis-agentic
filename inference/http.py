"""HTTP adapter — calls a remote inference endpoint.

Wraps the existing AXIS inference_server contract:
  POST {url}  body: {"image_path": str}
  -> {"prediction": int, "confidence": float, "findings": str?}

Same shape works for any OpenAI-compatible vision endpoint that follows
this minimal request/response contract; in PR2 a more general
OpenAIVisionAdapter will subclass this for true OpenAI / Ollama support.
"""
from __future__ import annotations

import time

import httpx

from inference.base import InferenceAdapter, Prediction


class HTTPAdapter(InferenceAdapter):
    """Persistent-connection HTTP client for an inference endpoint."""

    def __init__(self, url: str, timeout: float = 60.0):
        self._url = url
        self._timeout = timeout
        self._client: httpx.Client | None = None

    @property
    def name(self) -> str:
        return f"http({self._url})"

    def warmup(self) -> None:
        if self._client is None:
            self._client = httpx.Client(timeout=self._timeout)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def predict(self, image_path: str) -> Prediction:
        if self._client is None:
            self.warmup()
        assert self._client is not None
        t0 = time.monotonic()
        resp = self._client.post(self._url, json={"image_path": image_path})
        resp.raise_for_status()
        body = resp.json()
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        return Prediction(
            label=int(body["prediction"]),
            confidence=float(body.get("confidence", 0.0)),
            findings=body.get("findings"),
            elapsed_ms=elapsed_ms,
        )
