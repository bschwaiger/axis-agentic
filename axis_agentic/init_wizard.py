"""Interactive setup wizard — `axis-agentic init`.

Walks the user through:
1. Engine selection (Anthropic | OpenAI | Ollama / local | NVIDIA NIM)
2. Default adapter (HTTP / MLX / transformers / Anthropic) + per-adapter config
3. Optionally seeding a model registry entry
4. Optionally seeding a dataset registry entry

Writes the result to ~/.config/axis-agentic/profile.yaml. If a profile
already exists, the wizard offers to start fresh or amend in place
(amend = edit single sections without resetting everything).
"""
from __future__ import annotations

import os
from pathlib import Path

from axis_agentic.profile import (
    DEFAULT_PROFILE, PROFILE_PATH, load_profile, save_profile, profile_exists,
)


def run_wizard() -> int:
    print()
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║              axis-agentic — Setup Wizard                     ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")
    print()

    if profile_exists():
        print(f"  Existing profile at {PROFILE_PATH}.")
        choice = _ask("  [k]eep & edit  |  [r]eset to defaults  |  [q]uit", "k")
        if choice == "q":
            return 0
        if choice == "r":
            profile = _deepcopy(DEFAULT_PROFILE)
        else:
            profile = load_profile()
    else:
        profile = _deepcopy(DEFAULT_PROFILE)

    print()
    profile["engine"] = _setup_engine(profile.get("engine", {}))
    print()
    profile["adapter"] = _setup_default_adapter(profile.get("adapter", {}))
    print()
    if _ask_yn("  Add a model to the registry now?", default=False):
        new_model = _setup_model_entry()
        if new_model:
            profile.setdefault("models", []).append(new_model)
    print()
    if _ask_yn("  Add a dataset to the registry now?", default=False):
        new_ds = _setup_dataset_entry()
        if new_ds:
            profile.setdefault("datasets", []).append(new_ds)

    print()
    saved_path = save_profile(profile)
    print(f"  ✓ Profile written to {saved_path}")
    print()
    print(f"  Next: 'axis-agentic status' to inspect, "
          f"'axis-agentic cockpit' to launch the UI.")
    return 0


# ----------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------

ENGINE_PRESETS = {
    "1": {"kind": "anthropic", "model": "claude-sonnet-4-6", "label": "Anthropic Claude"},
    "2": {"kind": "openai-compat", "model": "gpt-4o", "base_url": None, "label": "OpenAI"},
    "3": {"kind": "openai-compat", "model": "qwen2.5",
          "base_url": "http://localhost:11434/v1", "label": "Ollama (local)"},
    "4": {"kind": "openai-compat", "model": "nvidia/llama-3.3-nemotron-super-49b-v1",
          "base_url": "https://integrate.api.nvidia.com/v1", "label": "NVIDIA NIM (Nemotron)"},
}


def _setup_engine(current: dict) -> dict:
    print("  ── Engine (the LLM driving the agents) ──")
    for k, v in ENGINE_PRESETS.items():
        print(f"    [{k}] {v['label']}  ({v['kind']}: {v['model']})")
    print(f"    [c] Custom OpenAI-compatible endpoint")
    print()
    if current:
        print(f"    Current: {current.get('kind')} / {current.get('model', '?')}")

    choice = _ask("  Pick (1-4 or c)", "1").lower()

    if choice == "c":
        kind = "openai-compat"
        base_url = _ask("  base_url (e.g. https://api.together.xyz/v1)", "")
        model = _ask("  model", "")
        eng = {"kind": kind, "model": model, "base_url": base_url}
    else:
        preset = ENGINE_PRESETS.get(choice, ENGINE_PRESETS["1"])
        eng = {"kind": preset["kind"], "model": preset["model"]}
        if preset.get("base_url"):
            eng["base_url"] = preset["base_url"]

    # Key handling — only check, don't store keys in profile
    if eng["kind"] == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("  ⚠  ANTHROPIC_API_KEY not in environment. Add it to .env before running.")
    elif eng["kind"] == "openai-compat":
        if "ollama" not in (eng.get("base_url") or "").lower() and "11434" not in (eng.get("base_url") or ""):
            if not os.environ.get("OPENAI_API_KEY"):
                print("  ⚠  OPENAI_API_KEY not in environment. Local Ollama doesn't need one.")

    return eng


# ----------------------------------------------------------------------
# Default adapter
# ----------------------------------------------------------------------

ADAPTER_PRESETS = {
    "1": {"kind": "http", "label": "HTTP endpoint (e.g. local inference server)"},
    "2": {"kind": "mlx", "label": "MLX (Apple Silicon, in-process)"},
    "3": {"kind": "transformers", "label": "HuggingFace transformers (PyTorch)"},
    "4": {"kind": "anthropic", "label": "Anthropic Claude vision"},
}


def _setup_default_adapter(current: dict) -> dict:
    print("  ── Default adapter (the model under test) ──")
    for k, v in ADAPTER_PRESETS.items():
        print(f"    [{k}] {v['label']}")
    print()
    if current:
        cur_target = current.get("url") or current.get("model_path") or current.get("model_id") or "?"
        print(f"    Current: {current.get('kind')} ({cur_target})")

    choice = _ask("  Pick (1-4)", "1")
    preset = ADAPTER_PRESETS.get(choice, ADAPTER_PRESETS["1"])
    kind = preset["kind"]

    if kind == "http":
        url = _ask("  URL", current.get("url", "http://localhost:8321/predict"))
        return {"kind": "http", "url": url}
    if kind == "mlx":
        path = _ask("  Model path (folder)",
                    current.get("model_path", ""))
        return {"kind": "mlx", "model_path": path}
    if kind == "transformers":
        model_id = _ask("  HF model ID or local path",
                        current.get("model_id", ""))
        return {"kind": "transformers", "model_id": model_id}
    if kind == "anthropic":
        model = _ask("  Anthropic vision model",
                     current.get("model", "claude-sonnet-4-6"))
        return {"kind": "anthropic", "model": model}
    return current


# ----------------------------------------------------------------------
# Model entry
# ----------------------------------------------------------------------

def _setup_model_entry() -> dict | None:
    print("  ── New model entry ──")
    name = _ask("  Display name", "")
    if not name:
        print("  Skipped.")
        return None
    model_id = _ask("  Short ID (used in filenames)",
                    name.lower().replace(" ", "-"))
    description = _ask("  Description (one line, optional)", "")
    adapter = _setup_default_adapter({})

    baseline_values = {}
    if _ask_yn("  Add baseline metrics (MCC / sens / spec) for comparison?", default=False):
        for metric in ["mcc", "sensitivity", "specificity"]:
            v = _ask(f"  baseline {metric}", "").strip()
            if v:
                try:
                    baseline_values[metric] = float(v)
                except ValueError:
                    print(f"  Invalid number, skipping {metric}.")
    baseline_source = ""
    if baseline_values:
        baseline_source = _ask("  baseline source (e.g. 'paper citation, n=...')", "")

    return {
        "name": name,
        "id": model_id,
        "description": description,
        "adapter": adapter,
        "baseline_values": baseline_values,
        "baseline_source": baseline_source,
    }


# ----------------------------------------------------------------------
# Dataset entry
# ----------------------------------------------------------------------

def _setup_dataset_entry() -> dict | None:
    print("  ── New dataset entry ──")
    dataset_path = _ask("  Dataset folder path", "")
    if not dataset_path:
        print("  Skipped.")
        return None

    name = _ask("  Display name", Path(dataset_path).name)
    description = _ask("  Description (optional)", "")
    manifest_path = _ask("  Manifest CSV path",
                         f"{dataset_path.rstrip('/')}/manifest.csv")

    if not Path(manifest_path).expanduser().exists():
        print(f"  ⚠  No manifest at {manifest_path}.")
        if _ask_yn("  Run dataset auto-detect now?", default=True):
            try:
                from tools.dataset_detect import (
                    inspect_dataset, propose_schema, walk_with_schema,
                    write_manifest, save_cached_schema,
                )
                evidence = inspect_dataset(dataset_path)
                schema = propose_schema(evidence)
                print()
                print("  Proposed schema:")
                import json as _json
                print(_json.dumps(schema, indent=4))
                if _ask_yn("  Accept?", default=True):
                    save_cached_schema(dataset_path, schema)
                    rows = walk_with_schema(dataset_path, schema)
                    write_manifest(rows, manifest_path)
                    print(f"  ✓ Wrote {len(rows)} rows to {manifest_path}")
            except Exception as e:
                print(f"  Auto-detect failed: {e}. Skipping.")

    return {
        "name": name,
        "description": description,
        "dataset_path": dataset_path,
        "manifest_path": manifest_path,
    }


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    raw = input(f"{prompt}{suffix}: ").strip()
    return raw if raw else default


def _ask_yn(prompt: str, default: bool = True) -> bool:
    suffix = " [Y/n]" if default else " [y/N]"
    raw = input(f"{prompt}{suffix}: ").strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes"}


def _deepcopy(obj):
    import copy
    return copy.deepcopy(obj)
