"""Persistent user profile at ~/.config/axis-agentic/profile.yaml.

Contains:
- Engine choice (Anthropic / OpenAI-compatible) and config
- Adapter default (HTTP/MLX/transformers/Anthropic) and config
- A registry of named models the user can pick from in the cockpit
- A registry of named datasets

The profile is the single source of truth for both the CLI (`axis-agentic
eval`) and the cockpit. If no profile exists, fall back to a built-in
default that preserves the previous hardcoded behavior so first-time
users see a working pipeline.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

PROFILE_DIR = Path(os.environ.get("AXIS_PROFILE_DIR") or
                   (Path.home() / ".config" / "axis-agentic"))
PROFILE_PATH = PROFILE_DIR / "profile.yaml"

PROFILE_VERSION = 1


# ----------------------------------------------------------------------
# Default profile — preserves pre-CLI behavior when nothing is configured
# ----------------------------------------------------------------------

DEFAULT_PROFILE: dict[str, Any] = {
    "version": PROFILE_VERSION,
    "engine": {
        "kind": "anthropic",
        "model": "claude-sonnet-4-6",
    },
    "adapter": {
        "kind": "http",
        "url": "http://localhost:8321/predict",
    },
    "models": [
        {
            "name": "AXIS-MURA-v1 4-bit",
            "id": "axis-mura-v1-4bit",
            "description": "LoRA fine-tuned MedGemma 1.5 4B, MLX 4-bit quantized",
            "adapter": {"kind": "http", "url": "http://localhost:8321/predict"},
            "baseline_values": {
                "mcc": 0.637,
                "sensitivity": 0.743,
                "specificity": 0.920,
            },
            "baseline_source": "AXIS-MURA-v1 on full MURA valid set (n=896)",
        },
        {
            "name": "MedGemma 1.5 4B base",
            "id": "medgemma-1.5-4b-base",
            "description": "Unmodified base model (no LoRA), MLX 4-bit",
            "adapter": {"kind": "http", "url": "http://localhost:8321/predict"},
            "baseline_values": {},
            "baseline_source": "",
        },
    ],
    "datasets": [
        {
            "name": "eval-018",
            "description": "Demo subset (18 studies, ~2 min)",
            "dataset_path": "data/eval-018",
            "manifest_path": "data/eval-018/manifest.csv",
        },
        {
            "name": "eval-020",
            "description": "Debug subset (20 studies, ~2 min)",
            "dataset_path": "data/eval-020",
            "manifest_path": "data/eval-020/manifest.csv",
        },
        {
            "name": "eval-100",
            "description": "Demo subset (100 studies, ~10 min)",
            "dataset_path": "data/eval-100",
            "manifest_path": "data/eval-100/manifest.csv",
        },
        {
            "name": "eval-300",
            "description": "Full evaluation (300 studies, ~30 min)",
            "dataset_path": "data/eval-300",
            "manifest_path": "data/eval-300/manifest.csv",
        },
    ],
}


# ----------------------------------------------------------------------
# Load / save
# ----------------------------------------------------------------------

def load_profile(path: Path | None = None) -> dict[str, Any]:
    """Return the user profile, or DEFAULT_PROFILE if none exists."""
    p = Path(path) if path else PROFILE_PATH
    if not p.exists():
        return _deepcopy(DEFAULT_PROFILE)
    with open(p) as f:
        data = yaml.safe_load(f) or {}
    if data.get("version") != PROFILE_VERSION:
        # Future: migrate. For now: warn and use what's there.
        pass
    return _merge_defaults(data)


def save_profile(profile: dict[str, Any], path: Path | None = None) -> Path:
    """Write profile YAML; ensure parent dir exists."""
    p = Path(path) if path else PROFILE_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        yaml.safe_dump(profile, f, sort_keys=False, default_flow_style=False)
    return p


def profile_exists(path: Path | None = None) -> bool:
    p = Path(path) if path else PROFILE_PATH
    return p.exists()


# ----------------------------------------------------------------------
# Apply profile to engine/adapter selection
# ----------------------------------------------------------------------

def apply_engine_env(profile: dict[str, Any]) -> None:
    """Set AXIS_ENGINE_* env vars from profile so build_engine() sees them.

    Only sets vars that aren't already in the environment (env wins).
    """
    eng = profile.get("engine", {})
    kind = eng.get("kind", "anthropic")
    os.environ.setdefault("AXIS_ENGINE", kind)

    if kind in {"openai", "openai-compat", "openai_compat"}:
        if eng.get("base_url"):
            os.environ.setdefault("AXIS_ENGINE_OPENAI_BASE_URL", eng["base_url"])
        if eng.get("model"):
            os.environ.setdefault("AXIS_ENGINE_OPENAI_MODEL", eng["model"])
        if eng.get("api_key"):
            os.environ.setdefault("AXIS_ENGINE_OPENAI_API_KEY", eng["api_key"])
    elif kind == "anthropic":
        if eng.get("model"):
            os.environ.setdefault("ANTHROPIC_MODEL", eng["model"])


def build_adapter_for_model(model_entry: dict[str, Any]):
    """Construct the right InferenceAdapter for a profile model entry.

    Falls back to the profile-level default adapter if the model entry
    doesn't specify one.
    """
    from inference import HTTPAdapter

    spec = model_entry.get("adapter") or {}
    kind = (spec.get("kind") or "http").lower()

    if kind == "http":
        return HTTPAdapter(url=spec.get("url", "http://localhost:8321/predict"))
    if kind == "mlx":
        from inference.mlx import MLXAdapter
        return MLXAdapter(model_path=spec.get("model_path") or spec.get("model", ""))
    if kind == "transformers":
        from inference.transformers import TransformersAdapter
        return TransformersAdapter(model_id=spec.get("model_id") or spec.get("model", ""))
    if kind == "anthropic":
        from inference.anthropic import AnthropicAdapter
        return AnthropicAdapter(model=spec.get("model"))
    raise ValueError(f"Unknown adapter kind in profile: {kind!r}")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _deepcopy(obj: Any) -> Any:
    import copy
    return copy.deepcopy(obj)


def _merge_defaults(loaded: dict[str, Any]) -> dict[str, Any]:
    """Fill in any missing top-level keys from DEFAULT_PROFILE."""
    merged = _deepcopy(DEFAULT_PROFILE)
    for key, value in loaded.items():
        merged[key] = value
    return merged
