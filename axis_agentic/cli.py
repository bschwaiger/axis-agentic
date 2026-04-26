"""axis-agentic CLI dispatcher.

Subcommands:
  init       Interactive setup — write profile to ~/.config/axis-agentic/profile.yaml
  cockpit    Launch the web UI
  cockpit --demo   Launch the demo cockpit (synthetic inference, no model needed)
  eval       Run the pipeline headlessly using the profile
  status     Show what's in the profile right now
  detect     Run dataset auto-detect on a folder
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="axis-agentic",
        description="Agentic evaluation framework for radiology AI models.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="Interactive setup wizard (writes the profile).")

    p_cockpit = sub.add_parser("cockpit", help="Launch the cockpit web UI.")
    p_cockpit.add_argument("--demo", action="store_true",
                            help="Use synthetic inference (no model needed).")
    p_cockpit.add_argument("--host", default="127.0.0.1")
    p_cockpit.add_argument("--port", type=int, default=8322)
    p_cockpit.add_argument("--verbose", action="store_true")

    p_eval = sub.add_parser("eval", help="Run pipeline headlessly using the profile.")
    p_eval.add_argument("--demo", action="store_true",
                        help="Use synthetic inference.")
    p_eval.add_argument("--verbose", action="store_true")

    sub.add_parser("status", help="Show the current profile + engine/adapter selection.")

    p_detect = sub.add_parser("detect", help="Run dataset auto-detect on a folder.")
    p_detect.add_argument("dataset_path")
    p_detect.add_argument("--manifest-out", default=None)
    p_detect.add_argument("--no-cache", action="store_true")
    p_detect.add_argument("--print-evidence", action="store_true")

    args = parser.parse_args(argv)

    _load_dotenv()

    if args.cmd == "init":
        from axis_agentic.init_wizard import run_wizard
        return run_wizard()

    if args.cmd == "cockpit":
        return _run_cockpit(args)

    if args.cmd == "eval":
        return _run_eval(args)

    if args.cmd == "status":
        return _run_status()

    if args.cmd == "detect":
        return _run_detect(args)

    parser.print_help()
    return 1


# ----------------------------------------------------------------------
# Subcommand implementations
# ----------------------------------------------------------------------

def _run_cockpit(args) -> int:
    from axis_agentic.profile import load_profile, apply_engine_env
    profile = load_profile()
    apply_engine_env(profile)

    # Defer import — cockpit/app.py reads the profile lazily through the
    # coordinator module.
    if args.demo:
        from cockpit.app_demo import main as cockpit_main
    else:
        from cockpit.app import main as cockpit_main

    sys.argv = ["cockpit", "--host", args.host, "--port", str(args.port)]
    if args.verbose:
        sys.argv.append("--verbose")
    cockpit_main()
    return 0


def _run_eval(args) -> int:
    from axis_agentic.profile import load_profile, apply_engine_env
    profile = load_profile()
    apply_engine_env(profile)

    if args.demo:
        from tools.evaluator_impl_demo import patch_tool_functions
        patch_tool_functions()

    from orchestrator.coordinator import interactive_setup, run_pipeline
    config = interactive_setup()
    run_pipeline(config, verbose=args.verbose)
    return 0


def _run_status() -> int:
    from axis_agentic.profile import load_profile, profile_exists, PROFILE_PATH
    import yaml as _yaml

    profile = load_profile()
    if profile_exists():
        print(f"  Profile: {PROFILE_PATH}")
    else:
        print(f"  Profile: (none — using built-in defaults)")
        print(f"           Run 'axis-agentic init' to create one at {PROFILE_PATH}")
    print()
    eng = profile.get("engine", {})
    adp = profile.get("adapter", {})
    print(f"  Engine:  {eng.get('kind')}  ({eng.get('model') or eng.get('base_url') or '?'})")
    print(f"  Adapter: {adp.get('kind')}  ({adp.get('url') or adp.get('model_path') or adp.get('model_id') or '?'})")
    print(f"  Models:  {len(profile.get('models', []))}")
    for m in profile.get("models", []):
        ad = m.get("adapter", {})
        print(f"    - {m['name']}  ({ad.get('kind')}: "
              f"{ad.get('url') or ad.get('model_path') or ad.get('model_id') or '?'})")
    print(f"  Datasets: {len(profile.get('datasets', []))}")
    for d in profile.get("datasets", []):
        print(f"    - {d['name']}  ({d['dataset_path']})")

    # Key check
    print()
    if eng.get("kind") == "anthropic":
        ok = "OK" if os.environ.get("ANTHROPIC_API_KEY") else "MISSING"
        print(f"  ANTHROPIC_API_KEY: {ok}")
    elif eng.get("kind") in {"openai", "openai-compat", "openai_compat"}:
        ok = "OK" if os.environ.get("OPENAI_API_KEY") else "(none — fine for Ollama)"
        print(f"  OPENAI_API_KEY: {ok}")
    return 0


def _run_detect(args) -> int:
    from axis_agentic.profile import load_profile, apply_engine_env
    apply_engine_env(load_profile())
    from tools.dataset_detect import main as detect_main
    sys.argv = ["dataset_detect", args.dataset_path]
    if args.manifest_out:
        sys.argv += ["--manifest-out", args.manifest_out]
    if args.no_cache:
        sys.argv.append("--no-cache")
    if args.print_evidence:
        sys.argv.append("--print-evidence")
    detect_main()
    return 0


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _load_dotenv() -> None:
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                k = k.strip()
                if k and not os.environ.get(k):
                    os.environ[k] = v.strip()
