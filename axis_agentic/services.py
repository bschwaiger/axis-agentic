"""Service auto-start — bring up local dependencies before the pipeline.

The framework's core promise is "you click Run and the agents handle the
rest". That includes starting any local services the configured engine
or adapter depends on:

- Ollama (engine kind=openai-compat, base_url at localhost:11434):
    `ollama serve` if the daemon is down, `ollama pull <model>` if the
    requested model isn't present.
- AXIS inference server (adapter kind=http pointed at localhost:8321):
    spawn `python server/inference_server.py` if it isn't responding.

Each step emits cockpit events so the UI can show "Starting Ollama…",
"Pulling qwen2.5…", etc. Cloud engines/adapters (Anthropic, OpenAI proper,
NIM) are no-ops — they're someone else's problem.
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
        """Pull the configured model. Blocks until pull completes or fails."""
        events.emit("service_starting", service=self.label, action=f"pull {self._model}")
        try:
            with httpx.stream(
                "POST", f"{self._native_root}/api/pull",
                json={"name": self._model, "stream": True},
                timeout=timeout_s,
            ) as r:
                r.raise_for_status()
                last_status = ""
                for line in r.iter_lines():
                    if not line:
                        continue
                    import json as _json
                    try:
                        data = _json.loads(line)
                    except Exception:
                        continue
                    status = data.get("status", "")
                    if status and status != last_status:
                        events.emit("service_progress", service=self.label, text=status)
                        last_status = status
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
# Local AXIS inference server (FastAPI on :8321)
# ----------------------------------------------------------------------

class AxisInferenceServerService:
    """Manage server/inference_server.py — the MedGemma-on-MLX HTTP wrapper."""

    def __init__(self, url: str):
        # url is something like http://localhost:8321/predict
        self._url = url.rstrip("/")
        from urllib.parse import urlparse
        parsed = urlparse(self._url)
        self._host = parsed.hostname or "localhost"
        self._port = parsed.port or 8321
        self._health = f"http://{self._host}:{self._port}/health"

    @property
    def label(self) -> str:
        return f"AXIS inference server ({self._host}:{self._port})"

    @property
    def is_local(self) -> bool:
        return self._host in {"localhost", "127.0.0.1", "::1"}

    def is_running(self) -> bool:
        try:
            r = httpx.get(self._health, timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False

    def start(self) -> bool:
        server_script = PROJECT_ROOT / "server" / "inference_server.py"
        if not server_script.exists():
            events.emit("service_failed", service=self.label,
                         error=f"Script not found: {server_script}")
            return False
        try:
            log = open("/tmp/axis_inference_server.log", "ab")
            subprocess.Popen(
                [sys.executable, str(server_script)],
                stdout=log, stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
                start_new_session=True,
            )
            return True
        except OSError as e:
            events.emit("service_failed", service=self.label, error=str(e))
            return False

    def ensure_ready(self, startup_timeout_s: int = 60) -> bool:
        events.emit("service_check", service=self.label)
        if self.is_running():
            events.emit("service_ready", service=self.label, detail="already running")
            return True
        if not self.is_local:
            events.emit("service_failed", service=self.label,
                         error=f"Not reachable and not local — start it manually at {self._url}")
            return False

        events.emit("service_starting", service=self.label, action="spawn server/inference_server.py")
        if not self.start():
            return False
        deadline = time.time() + startup_timeout_s
        while time.time() < deadline:
            if self.is_running():
                events.emit("service_ready", service=self.label, detail=f"port {self._port}")
                return True
            time.sleep(0.5)
        events.emit("service_failed", service=self.label,
                     error=f"Did not respond within {startup_timeout_s}s. "
                           f"Check /tmp/axis_inference_server.log.")
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
        srv = AxisInferenceServerService(url=url)
        return srv.ensure_ready()

    return True
