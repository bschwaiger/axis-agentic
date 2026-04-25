"""
Evaluator Agent — Agent 1 in the AXIS Agentic pipeline.

Drives the inference -> validate -> metrics tool sequence through any
Engine (Anthropic today; OpenAI-compatible in PR2). The agent loop is
provider-agnostic: it hands tool specs to the engine, executes whatever
the model asks for, and feeds results back until the model stops.

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

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.evaluator_impl import execute_tool  # noqa: E402
from orchestrator.engine import Engine, ToolResult, get_default_engine  # noqa: E402
from orchestrator.formatting import (  # noqa: E402
    format_tool_call, format_tool_result, format_engine_text, format_agent_complete,
    _extract_key_args, _summarize_result,
)
from cockpit import events  # noqa: E402

MAX_TURNS = 10

SYSTEM_PROMPT = """\
You are the Evaluator agent in the AXIS Agentic pipeline. Your job is to evaluate a \
medical image classification model on a dataset of musculoskeletal X-rays.

You have three tools available. Execute them in this order:
1. run_inference — Run the model on the dataset via the active inference adapter.
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

VERBOSE = False


def load_tools() -> list[dict]:
    tools_path = PROJECT_ROOT / "tools" / "evaluator_tools.json"
    with open(tools_path) as f:
        return json.load(f)


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


def run_agent(task: dict, verbose: bool = False, engine: Engine | None = None) -> dict:
    """Run the Evaluator agent loop. Returns the final metrics or error."""
    verbose = verbose or VERBOSE
    if engine is None:
        engine = get_default_engine()

    tools = load_tools()
    engine.reset(system=SYSTEM_PROMPT, tools=tools, max_tokens=4096)

    final_result = None
    final_metrics = None

    response = engine.send_user_message(task_to_user_message(task))

    for turn in range(1, MAX_TURNS + 1):
        if verbose:
            print(f"--- Turn {turn} ---")

        events.emit("api_call", provider=engine.provider_name, model=engine.model_name,
                     endpoint="messages.create", turn=turn, agent="evaluator")
        events.emit("api_response", provider=engine.provider_name, model=engine.model_name,
                     turn=turn, finish_reason=response.stop_reason,
                     elapsed_ms=response.elapsed_ms,
                     prompt_tokens=response.usage.get("input_tokens"),
                     completion_tokens=response.usage.get("output_tokens"),
                     tool_calls=len(response.tool_calls),
                     agent="evaluator")

        if response.text and verbose:
            print(f"  {engine.provider_name}: {response.text[:200]}")

        # Execute tool calls
        tool_results: list[ToolResult] = []
        for call in response.tool_calls:
            print(format_tool_call(call.name, call.arguments))
            events.emit("tool_call", fn_name=call.name,
                        summary=_extract_key_args(call.name, call.arguments),
                        agent="evaluator")
            if verbose:
                print(f"    Args: {json.dumps(call.arguments, indent=2)}")

            result_str = execute_tool(call.name, call.arguments)
            result_data = json.loads(result_str)

            print(format_tool_result(call.name, result_data))
            events.emit("tool_result", fn_name=call.name,
                        summary=_summarize_result(call.name, result_data),
                        agent="evaluator")

            # Pipeline-sidebar milestones
            if call.name == "run_inference" and result_data.get("status") == "success":
                n = result_data.get("predictions_written", "?")
                errs = result_data.get("error_count", 0)
                events.emit("milestone", text=f"{n} predictions written, {errs} errors", agent="evaluator")
            elif call.name == "validate_results":
                if result_data.get("status") == "pass":
                    events.emit("milestone", text="Results validated", agent="evaluator")
                else:
                    issues = result_data.get("issues", [])
                    events.emit("milestone", text=f"Validation: {'; '.join(issues[:2])}", agent="evaluator")
            elif call.name == "compute_metrics" and result_data.get("status") == "success":
                s = result_data.get("summary", {})
                events.emit("milestone",
                            text=f"MCC={s.get('mcc','?')}, Sens={s.get('sensitivity','?')}, Spec={s.get('specificity','?')}",
                            agent="evaluator")

            if verbose:
                print(f"    Raw: {json.dumps(result_data, indent=2)[:500]}")

            if call.name == "compute_metrics" and result_data.get("status") == "success":
                final_result = result_data
                metrics_path = result_data.get("metrics_path", "")
                if metrics_path:
                    try:
                        p = Path(metrics_path)
                        if p.exists():
                            with open(p) as f:
                                final_metrics = json.load(f)
                    except Exception:
                        final_metrics = result_data.get("summary")

            tool_results.append(ToolResult(call_id=call.id, content=result_str))

        # Done?
        if not tool_results:
            if response.text:
                print(format_engine_text(engine.provider_name, response.text))
                events.emit("claude_text", text=response.text, agent="evaluator")
            print(format_agent_complete("Evaluator", final_metrics))
            events.emit("agent_complete", agent="Evaluator", metrics=final_metrics)
            break

        response = engine.send_tool_results(tool_results)
    else:
        print(f"  Warning: Evaluator hit max turns ({MAX_TURNS})")

    return final_result or {"error": "No metrics computed"}


def _load_env():
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                key, val = line.split("=", 1)
                key, val = key.strip(), val.strip()
                if key and not os.environ.get(key):
                    os.environ[key] = val


def main():
    parser = argparse.ArgumentParser(description="AXIS Evaluator Agent")
    parser.add_argument("--task", type=str, help="Path to task YAML file")
    parser.add_argument("--task-json", type=str, help="Task as inline JSON")
    parser.add_argument("--verbose", action="store_true", help="Show raw JSON output")
    args = parser.parse_args()

    _load_env()

    task = load_task(task_path=args.task, task_json=args.task_json)
    result = run_agent(task, verbose=args.verbose)

    if args.verbose:
        print(f"\n{'='*60}\nEVALUATOR RESULT:\n{json.dumps(result, indent=2)}\n{'='*60}")


if __name__ == "__main__":
    main()
