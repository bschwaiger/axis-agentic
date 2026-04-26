"""Service auto-start — bring up local dependencies before the pipeline.

The framework's core promise is "you click Run and the agents handle the
rest". That includes starting any local services the configured engine
depends on:

- Ollama (engine kind=openai-compat, base_url at localhost:11434):
    `ollama serve` if the daemon is down, `ollama pull <model>` if the
    requested model isn't present.

For HTTP adapters pointing at a localhost endpoint that isn't responding,
the framework fails fast with a clear hint rather than trying to spawn a
server it doesn't know about — bring your own server.

Each step emits cockpit events so the UI can show "Starting Ollama…",
"Pulling qwen2.5…", etc. Cloud engines/adapters (Anthropic, OpenAI proper,
NIM, …) are no-ops.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx

from cockpit import events

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# How long to wait for a service to come up after we start it
DEFAULT_STARTUP_TIMEOUT_S = 30
DEFAULT_PULL_TIMEOUT_S = 600  # model pulls can be slow


def _human_bytes(n: int) -> str:
    """1234567 -> '1.2 MB'. Used in progress lines."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n = n / 1024
    return f"{n:.1f} PB"


# ----------------------------------------------------------------------
# Ollama
# ----------------------------------------------------------------------

class OllamaService:
    """Manage a local Ollama daemon + ensure a specific model is pulled."""

    def __init__(self, base_url: str, model: str):
        # Strip trailing /v1 etc. — Ollama's native API lives at the root
        self._base_url = base_url.rstrip("/")
        self._native_root = self._base_url.rsplit("/v1", 1)[0]
        self._model = model

    @property
    def label(self) -> str:
        return f"Ollama ({self._native_root})"

    def is_running(self) -> bool:
        try:
            r = httpx.get(f"{self._native_root}/api/tags", timeout=2.0)
            return r.status_code == 200
        except (httpx.RequestError, httpx.HTTPError):
            return False

    def has_model(self) -> bool:
        try:
            r = httpx.get(f"{self._native_root}/api/tags", timeout=4.0)
            r.raise_for_status()
            tags = r.json().get("models", [])
            names = {m.get("name", "").split(":")[0] for m in tags}
            # Match by base name (qwen2.5 == qwen2.5:latest)
            return self._model.split(":")[0] in names
        except Exception:
            return False

    def start_daemon(self) -> bool:
        """Spawn `ollama serve` in the background. Returns True if dispatched."""
        if not shutil.which("ollama"):
            events.emit(
                "service_failed", service=self.label,
                error="`ollama` binary not found. Install from https://ollama.com or run "
                       "`brew install ollama`.",
            )
            return False
        try:
            log = open("/tmp/axis_agentic_ollama.log", "ab")
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            return True
        except OSError as e:
            events.emit("service_failed", service=self.label, error=f"Failed to start: {e}")
            return False

    def pull_model(self, timeout_s: int = DEFAULT_PULL_TIMEOUT_S) -> bool:
        """Pull the configured model. Blocks until pull completes or fails.

        Throttles per-status emits to once per second so the cockpit feed
        doesn't get flooded with thousands of byte-progress lines.
        """
        events.emit("service_starting", service=self.label, action=f"pull {self._model}")
        try:
            with httpx.stream(
                "POST", f"{self._native_root}/api/pull",
                json={"name": self._model, "stream": True},
                timeout=timeout_s,
            ) as r:
                r.raise_for_status()
                last_status = ""
                last_emit = 0.0
                for line in r.iter_lines():
                    if not line:
                        continue
                    import json as _json
                    try:
                        data = _json.loads(line)
                    except Exception:
                        continue
                    status = data.get("status", "")
                    total = data.get("total")
                    completed = data.get("completed")
                    text = status
                    if total and completed:
                        pct = (completed / total) * 100
                        text = f"{status} — {_human_bytes(completed)}/{_human_bytes(total)} ({pct:.0f}%)"
                    now = time.monotonic()
                    # Emit on status change (immediate) OR ~once per second for byte-progress
                    if (status and status != last_status) or (now - last_emit > 1.0):
                        events.emit("service_progress", service=self.label, text=text)
                        last_status = status
                        last_emit = now
                    if data.get("error"):
                        events.emit("service_failed", service=self.label, error=data["error"])
                        return False
            return True
        except Exception as e:
            events.emit("service_failed", service=self.label, error=f"Pull failed: {e}")
            return False

    def ensure_ready(self, startup_timeout_s: int = DEFAULT_STARTUP_TIMEOUT_S) -> bool:
        """Bring Ollama + the configured model to ready, emit progress."""
        events.emit("service_check", service=self.label)

        if not self.is_running():
            events.emit("service_starting", service=self.label, action="serve")
            if not self.start_daemon():
                return False
            deadline = time.time() + startup_timeout_s
            while time.time() < deadline:
                if self.is_running():
                    break
                time.sleep(0.5)
            else:
                events.emit("service_failed", service=self.label,
                             error=f"Daemon did not respond within {startup_timeout_s}s")
                return False

        if not self.has_model():
            if not self.pull_model():
                return False

        events.emit("service_ready", service=self.label, detail=f"model {self._model}")
        return True


# ----------------------------------------------------------------------
# Generic HTTP-endpoint health check (for HTTPAdapter)
# ----------------------------------------------------------------------

class HTTPEndpointService:
    """Health-check a user-managed HTTP inference endpoint.

    The framework does not bundle a server, so it can't auto-start one.
    What this class does is: check if the endpoint is up, and if not,
    fail fast with a clear hint to start it manually.
    """

    def __init__(self, url: str):
        self._url = url.rstrip("/")
        from urllib.parse import urlparse
        parsed = urlparse(self._url)
        self._host = parsed.hostname or "localhost"
        self._port = parsed.port or 80
        # Try a /health endpoint first; fall back to the prediction URL
        self._health = f"http://{self._host}:{self._port}/health"

    @property
    def label(self) -> str:
        return f"HTTP inference endpoint ({self._host}:{self._port})"

    def is_running(self) -> bool:
        # Try /health, then a HEAD on the predict URL
        for url in (self._health, self._url):
            try:
                r = httpx.get(url, timeout=2.0)
                if r.status_code < 500:
                    return True
            except Exception:
                continue
        return False

    def ensure_ready(self) -> bool:
        events.emit("service_check", service=self.label)
        if self.is_running():
            events.emit("service_ready", service=self.label, detail="reachable")
            return True
        events.emit(
            "service_failed", service=self.label,
            error=(f"Not reachable. Start your inference server, then retry. "
                   f"Expected at {self._url}."),
        )
        return False


# ----------------------------------------------------------------------
# Top-level: bring up engine + adapter dependencies
# ----------------------------------------------------------------------

def ensure_engine_ready(engine) -> bool:
    """Bring up any local service the engine depends on. No-op for cloud."""
    provider = (getattr(engine, "provider_name", "") or "").lower()
    base_url = getattr(engine, "_base_url", None)

    # Cloud providers — nothing to start
    if provider in {"anthropic", "openai", "nemotron"}:
        return True
    if base_url and (
        "openai.com" in base_url or
        "anthropic.com" in base_url or
        "nvidia.com" in base_url or
        "together" in base_url or
        "groq" in base_url
    ):
        return True

    # Local Ollama / LM Studio
    if base_url and ("11434" in base_url or "ollama" in base_url.lower()):
        ollama = OllamaService(base_url=base_url, model=engine.model_name)
        return ollama.ensure_ready()

    # Unknown local — best effort
    return True


def ensure_adapter_ready(adapter) -> bool:
    """Bring up any local service the adapter depends on. No-op for in-process / cloud."""
    name = (getattr(adapter, "name", "") or "").lower()

    # In-process adapters — nothing to start (model loading happens lazily inside)
    if name.startswith("mlx(") or name.startswith("transformers(") or name.startswith("anthropic("):
        events.emit("service_ready", service=name, detail="in-process")
        return True

    # HTTP adapter — extract URL
    if name.startswith("http("):
        url = name[5:-1]  # strip "http(" and ")"
        srv = HTTPEndpointService(url=url)
        return srv.ensure_ready()

    return True
