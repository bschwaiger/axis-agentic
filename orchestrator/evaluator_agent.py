"""
Evaluator Agent — Agent 1 in the AXIS Agentic pipeline.

Sends a task to Nemotron Super via NVIDIA NIM API with function calling.
Nemotron decides which tools to call (run_inference → validate_results →
compute_metrics). The agent loop executes tools locally and feeds results
back until Nemotron finishes.

Usage:
    python -m orchestrator.evaluator_agent --task docs/task_eval020.yaml
    python -m orchestrator.evaluator_agent --task docs/task_eval020.yaml --verbose
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.evaluator_impl import execute_tool  # noqa: E402
from orchestrator.formatting import (  # noqa: E402
    format_tool_call, format_tool_result, format_nemotron_text, format_agent_complete,
    _extract_key_args, _summarize_result,
)
from cockpit import events  # noqa: E402

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

NIM_BASE_URL = os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
NIM_MODEL = os.environ.get("NIM_MODEL", "nvidia/nemotron-3-super-120b-a12b")
API_KEY = os.environ.get("NVIDIA_API_KEY", "")

MAX_TURNS = 10

SYSTEM_PROMPT = """\
You are the Evaluator agent in the AXIS Agentic pipeline. Your job is to evaluate a \
medical image classification model (AXIS-MURA-v1) on a dataset of musculoskeletal X-rays.

You have three tools available. Execute them in this order:
1. run_inference — Run the model on the dataset via the local inference server.
2. validate_results — Validate the predictions against the manifest.
3. compute_metrics — Compute classification metrics (MCC, sensitivity, specificity, etc.).

After all three tools complete successfully, summarize the results and finish.

Important:
- Always call run_inference first. It needs dataset_path, manifest_path, and output_path.
- Pass the output_path from run_inference as predictions_path to validate_results and compute_metrics.
- For compute_metrics, also provide an output_path for the metrics JSON file.
- If any tool returns errors, report them clearly.
- When done, respond with a final summary including the key metrics.\
"""

# Module-level verbose flag (set by CLI or Coordinator)
VERBOSE = False


# ------------------------------------------------------------------
# Load tool definitions
# ------------------------------------------------------------------

def load_tools() -> list[dict]:
    tools_path = PROJECT_ROOT / "tools" / "evaluator_tools.json"
    with open(tools_path) as f:
        return json.load(f)


# ------------------------------------------------------------------
# Parse task file
# ------------------------------------------------------------------

def load_task(task_path: str | None = None, task_json: str | None = None) -> dict:
    if task_json:
        return json.loads(task_json)
    if task_path:
        p = Path(task_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        with open(p) as f:
            return yaml.safe_load(f)
    raise ValueError("Provide --task or --task-json")


def task_to_user_message(task: dict) -> str:
    return (
        f"Evaluate the model on the following dataset:\n"
        f"- Dataset path: {task['dataset_path']}\n"
        f"- Manifest path: {task['manifest_path']}\n"
        f"- Output directory: {task['output_dir']}\n"
        f"\n"
        f"Write predictions to {task['output_dir']}/predictions.csv and "
        f"metrics to {task['output_dir']}/metrics.json."
    )


# ------------------------------------------------------------------
# Agent loop
# ------------------------------------------------------------------

def run_agent(task: dict, verbose: bool = False) -> dict:
    """Run the Evaluator agent loop. Returns the final metrics or error."""
    verbose = verbose or VERBOSE
    api_key = os.environ.get("NVIDIA_API_KEY", "") or API_KEY
    tools = load_tools()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task_to_user_message(task)},
    ]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    # Persistent connection — avoids TCP+TLS handshake on every turn
    client = httpx.Client(headers=headers, timeout=120.0)

    final_result = None
    final_metrics = None

    for turn in range(1, MAX_TURNS + 1):
        if verbose:
            print(f"--- Turn {turn} ---")

        events.emit("api_call", provider="NVIDIA NIM", model=NIM_MODEL,
                     endpoint=f"{NIM_BASE_URL}/chat/completions",
                     turn=turn, agent="evaluator")

        payload = {
            "model": NIM_MODEL,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": 1024,
            "temperature": 0.1,
        }

        import time as _time
        t0 = _time.monotonic()
        resp = client.post(
            f"{NIM_BASE_URL}/chat/completions",
            json=payload,
        )
        resp.raise_for_status()
        elapsed_ms = int((_time.monotonic() - t0) * 1000)
        data = resp.json()

        choice = data["choices"][0]
        msg = choice["message"]
        finish_reason = choice["finish_reason"]

        usage = data.get("usage", {})
        events.emit("api_response", provider="NVIDIA NIM", model=NIM_MODEL,
                     turn=turn, finish_reason=finish_reason,
                     elapsed_ms=elapsed_ms,
                     prompt_tokens=usage.get("prompt_tokens"),
                     completion_tokens=usage.get("completion_tokens"),
                     tool_calls=len(msg.get("tool_calls", [])),
                     agent="evaluator")

        messages.append(msg)

        # Handle tool calls
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn_name = tc["function"]["name"]
                fn_args = json.loads(tc["function"]["arguments"])

                # Clean logging
                print(format_tool_call(fn_name, fn_args))
                events.emit("tool_call", fn_name=fn_name, summary=_extract_key_args(fn_name, fn_args), agent="evaluator")

                if verbose:
                    print(f"    Args: {json.dumps(fn_args, indent=2)}")

                result_str = execute_tool(fn_name, fn_args)
                result_data = json.loads(result_str)

                # Clean logging
                summary = _summarize_result(fn_name, result_data)
                print(format_tool_result(fn_name, result_data))
                events.emit("tool_result", fn_name=fn_name, summary=summary, agent="evaluator")

                # Emit milestones for the pipeline sidebar
                if fn_name == "run_inference" and result_data.get("status") == "success":
                    n = result_data.get("predictions_written", "?")
                    errs = result_data.get("error_count", 0)
                    events.emit("milestone", text=f"{n} predictions written, {errs} errors", agent="evaluator")
                elif fn_name == "validate_results":
                    if result_data.get("status") == "pass":
                        events.emit("milestone", text="Results validated", agent="evaluator")
                    else:
                        issues = result_data.get("issues", [])
                        events.emit("milestone", text=f"Validation: {'; '.join(issues[:2])}", agent="evaluator")
                elif fn_name == "compute_metrics" and result_data.get("status") == "success":
                    s = result_data.get("summary", {})
                    events.emit("milestone", text=f"MCC={s.get('mcc','?')}, Sens={s.get('sensitivity','?')}, Spec={s.get('specificity','?')}", agent="evaluator")

                if verbose:
                    print(f"    Raw: {json.dumps(result_data, indent=2)[:500]}")

                # Track metrics result
                if fn_name == "compute_metrics" and result_data.get("status") == "success":
                    final_result = result_data
                    # Load full metrics for the completion box
                    metrics_path = result_data.get("metrics_path", "")
                    if metrics_path:
                        try:
                            p = Path(metrics_path)
                            if p.exists():
                                with open(p) as f:
                                    final_metrics = json.load(f)
                        except Exception:
                            final_metrics = result_data.get("summary")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_str,
                })

        # If the model is done
        if finish_reason == "stop":
            if msg.get("content"):
                print(format_nemotron_text(msg["content"]))
                events.emit("nemotron_text", text=msg["content"], agent="evaluator")
            print(format_agent_complete("Evaluator", final_metrics))
            events.emit("agent_complete", agent="Evaluator", metrics=final_metrics)
            break
    else:
        print(f"  ⚠ Evaluator: hit max turns ({MAX_TURNS})")

    client.close()
    return final_result or {"error": "No metrics computed"}


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AXIS Evaluator Agent")
    parser.add_argument("--task", type=str, help="Path to task YAML file")
    parser.add_argument("--task-json", type=str, help="Task as inline JSON")
    parser.add_argument("--verbose", action="store_true", help="Show raw JSON output")
    args = parser.parse_args()

    if not API_KEY:
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("NVIDIA_API_KEY="):
                    os.environ["NVIDIA_API_KEY"] = line.split("=", 1)[1].strip()
                    break

    if not os.environ.get("NVIDIA_API_KEY") and not API_KEY:
        print("[!] NVIDIA_API_KEY not set. Export it or add to .env")
        sys.exit(1)

    task = load_task(task_path=args.task, task_json=args.task_json)
    result = run_agent(task, verbose=args.verbose)

    if args.verbose:
        print(f"\n{'='*60}")
        print(f"EVALUATOR RESULT:")
        print(json.dumps(result, indent=2))
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
