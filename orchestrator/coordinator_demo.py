"""
Coordinator (Demo Mode) — Runs the full AXIS Agentic pipeline with synthetic inference.

Patches run_inference to use fast synthetic predictions (~50ms/study) instead of
calling a real inference adapter. Everything else runs live: agent orchestration,
validation, metrics, analyst, literature search, figures, report.

Usage:
    python -m orchestrator.coordinator_demo
    python -m orchestrator.coordinator_demo --verbose
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Patch inference BEFORE any pipeline imports trigger tool loading
from tools.evaluator_impl_demo import patch_tool_functions  # noqa: E402
patch_tool_functions()

from orchestrator.coordinator import (  # noqa: E402
    interactive_setup, run_pipeline, _load_env, AVAILABLE_MODELS, AVAILABLE_DATASETS,
)


DEMO_BANNER = """
\033[43m\033[30m                                                                 \033[0m
\033[43m\033[30m   DEMO MODE — Local Inference Simulated                         \033[0m
\033[43m\033[30m   Predictions are synthetic (~50ms/study). No inference server  \033[0m
\033[43m\033[30m   needed. All other pipeline steps run live.                    \033[0m
\033[43m\033[30m                                                                 \033[0m
"""


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AXIS Agentic Coordinator (Demo Mode)")
    parser.add_argument("--verbose", action="store_true", help="Show raw JSON output from agents")
    args = parser.parse_args()

    _load_env()
    print(DEMO_BANNER)
    config = interactive_setup()
    run_pipeline(config, verbose=args.verbose)


if __name__ == "__main__":
    main()
