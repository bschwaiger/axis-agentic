"""
Analyst Agent — Agent 2 in the AXIS Agentic pipeline.

Receives a comparison matrix (1 model × N datasets) from the Coordinator and runs:
1. compare_baselines — analyze metrics across datasets vs. baselines + cross-dataset deltas
2. Nemotron proposes PubMed search queries based on comparative findings
3. Terminal checkpoint — human reviews/edits proposed queries
4. search_literature — Tavily API scoped to PubMed
5. generate_figures — per-dataset confusion matrices, comparative bar chart, confidence distributions
6. write_report — structured comparative Markdown report with Literature Context

Usage:
    python -m orchestrator.analyst_agent \
        --matrix results/comparative/comparison_matrix.json \
        --task docs/task_demo.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import httpx
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analyst_impl import execute_tool  # noqa: E402
from orchestrator.formatting import (  # noqa: E402
    format_tool_call, format_tool_result, format_nemotron_text,
    format_agent_complete, format_checkpoint, format_literature_results,
    _extract_key_args, _summarize_result,
)
from cockpit import events  # noqa: E402

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

NIM_BASE_URL = os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
NIM_MODEL = os.environ.get("NIM_MODEL", "nvidia/nemotron-3-super-120b-a12b")
API_KEY = os.environ.get("NVIDIA_API_KEY", "")

MAX_TURNS = 15

SYSTEM_PROMPT = """\
You are the Analyst agent in the AXIS Agentic pipeline. You receive a comparison matrix \
containing evaluation metrics for one model across multiple datasets. Your job is to \
perform comparative analysis, contextualize with published literature, and produce a \
comprehensive report.

You have four tools available:

1. compare_baselines — Compare metrics across all datasets against baselines and compute \
cross-dataset deltas. Pass the matrix_path. Call this first.
2. search_literature — Search PubMed for relevant papers. Call this AFTER the human \
approves your proposed search queries.
3. generate_figures — Generate comparative figures (confusion matrices, metrics bar chart, \
confidence distributions). Pass the matrix_path and output_dir.
4. write_report — Write a comparative Markdown evaluation report. Pass matrix_path, \
comparison_path, literature_results, figures_dir, and output_path.

IMPORTANT WORKFLOW:
- After calling compare_baselines, analyze the results: which datasets show different \
performance? Which metrics deviate from baselines? Are there body-part or sample-size effects?
- First, write 2-3 sentences explaining your reasoning: what the model is (e.g. a fine-tuned \
MedGemma vision-language model), what task it performs (e.g. abnormality detection on MSK \
radiographs), and what the comparative findings suggest (e.g. which metrics are flagged, \
what patterns you see across datasets).
- Then propose 2-4 PubMed search queries. STRICT RULES for queries:
  - Maximum 5 words per query
  - Use MeSH-style keywords, not sentences
  - Good: "MedGemma fracture detection", "MURA deep learning benchmark", "musculoskeletal AI sensitivity"
  - Bad: "Effect of training set size on sensitivity and specificity in deep learning models"
- Format your proposed queries as a numbered list:
  1. query one
  2. query two
- After proposing queries, STOP and wait for human approval.
- When approved queries arrive, call search_literature, then generate_figures, then write_report.
- For write_report, pass: matrix_path, comparison_path, literature_results, figures_dir, output_path.\
"""

VERBOSE = False


# ------------------------------------------------------------------
# Load tools
# ------------------------------------------------------------------

def load_tools() -> list[dict]:
    tools_path = PROJECT_ROOT / "tools" / "analyst_tools.json"
    with open(tools_path) as f:
        return json.load(f)


# ------------------------------------------------------------------
# Terminal checkpoint
# ------------------------------------------------------------------

def terminal_checkpoint_queries(proposed_text: str) -> list[str]:
    """Display proposed queries in terminal (or cockpit) and let human edit/approve."""
    queries = _parse_queries(proposed_text)

    # Extract reasoning (non-query lines) to display before queries
    reasoning_lines = []
    for line in proposed_text.strip().split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r'^\s*\d+[.)]\s+', line):
            continue
        if re.match(r'^\s*[-*]\s+', line) and len(stripped) < 100:
            continue
        reasoning_lines.append(stripped)

    reasoning_text = " ".join(reasoning_lines)

    # Emit reasoning to cockpit log
    if reasoning_text:
        events.emit("nemotron_text", text=reasoning_text, agent="analyst")

    # Print reasoning
    if reasoning_lines:
        print()
        print("  💬 Nemotron:")
        for line in reasoning_lines:
            while len(line) > 76:
                print(f"    {line[:76]}")
                line = line[76:]
            print(f"    {line}")

    # Cockpit mode: emit event and wait for HTTP approval
    if events.is_enabled():
        events.emit("checkpoint_queries", queries=queries, reasoning=reasoning_text)
        result = events.wait_for_approval("query_checkpoint", timeout=600.0)
        if result is None or result.get("action") == "skip":
            print("  Literature search skipped.")
            return []
        approved = result.get("queries", queries)
        print(f"  Approved {len(approved)} queries (via cockpit).")
        return approved

    print(format_checkpoint("Review proposed PubMed queries below"))
    print()
    for i, q in enumerate(queries, 1):
        print(f"    {i}. {q}")

    print()
    print("  Options: [Enter] Approve  |  [e] Edit  |  [s] Skip")

    choice = input("\n  Your choice: ").strip().lower()

    if choice == "s":
        print("  Literature search skipped.")
        return []
    elif choice == "e":
        print("  Enter your queries, one per line. Empty line to finish:")
        custom = []
        while True:
            line = input("    > ").strip()
            if not line:
                break
            custom.append(line)
        if custom:
            print(f"  Using {len(custom)} custom queries.")
            return custom
        print("  No queries entered, using original proposals.")

    print(f"  Approved {len(queries)} queries.")
    return queries


def _parse_queries(text: str) -> list[str]:
    """Extract numbered queries from text."""
    lines = text.strip().split("\n")
    queries = []
    for line in lines:
        m = re.match(r'^\s*(?:\d+[.)]\s*|[-*]\s+)(.*)', line)
        if m:
            q = m.group(1).strip()
            if q:
                queries.append(q)
    if not queries:
        queries = [line.strip() for line in lines if line.strip()]
    return queries


# ------------------------------------------------------------------
# Agent loop
# ------------------------------------------------------------------

def run_agent(matrix_path: str, task: dict, verbose: bool = False) -> dict:
    """Run the Analyst agent loop with comparative analysis."""
    verbose = verbose or VERBOSE
    api_key = os.environ.get("NVIDIA_API_KEY", "") or API_KEY
    tools = load_tools()
    output_dir = task.get("output_dir", "results/comparative")
    model_name = task.get("model_name", "unknown")

    # Load matrix to build context message
    matrix = _load_json(matrix_path)
    ds_names = [ev["dataset_name"] for ev in matrix["evaluations"]]
    ds_summary = "\n".join(
        f"  - {ev['dataset_name']}: n={ev['metrics'].get('total_evaluated','?')}, "
        f"MCC={ev['metrics'].get('mcc','?')}, Sens={ev['metrics'].get('sensitivity','?')}, "
        f"Spec={ev['metrics'].get('specificity','?')}"
        for ev in matrix["evaluations"]
    )

    user_msg = (
        f"Perform comparative analysis for {model_name} across {len(ds_names)} datasets.\n\n"
        f"Comparison matrix: {matrix_path}\n"
        f"Output directory (USE THIS EXACT PATH for all output): {output_dir}\n\n"
        f"Dataset summary:\n{ds_summary}\n\n"
        f"Baseline values: {json.dumps(matrix.get('baseline_values', {}))}\n\n"
        f"Start by calling compare_baselines with matrix_path=\"{matrix_path}\", "
        f"then propose literature search queries based on your findings.\n\n"
        f"IMPORTANT: When calling generate_figures, use output_dir=\"{output_dir}\". "
        f"When calling write_report, use figures_dir=\"{output_dir}\" and "
        f"output_path=\"{output_dir}/report.md\"."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    # Persistent connection — avoids TCP+TLS handshake on every turn
    client = httpx.Client(headers=headers, timeout=120.0)

    checkpoint_done = False
    final_result = None

    for turn in range(1, MAX_TURNS + 1):
        if verbose:
            print(f"--- Turn {turn} ---")

        events.emit("api_call", provider="NVIDIA NIM", model=NIM_MODEL,
                     endpoint=f"{NIM_BASE_URL}/chat/completions",
                     turn=turn, agent="analyst")

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
                     agent="analyst")

        messages.append(msg)

        # Handle tool calls
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn_name = tc["function"]["name"]
                fn_args = json.loads(tc["function"]["arguments"])

                print(format_tool_call(fn_name, fn_args))
                events.emit("tool_call", fn_name=fn_name, summary=_extract_key_args(fn_name, fn_args), agent="analyst")
                if verbose:
                    print(f"    Args: {json.dumps(fn_args, indent=2)}")

                result_str = execute_tool(fn_name, fn_args)
                result_data = json.loads(result_str)

                print(format_tool_result(fn_name, result_data))
                events.emit("tool_result", fn_name=fn_name, summary=_summarize_result(fn_name, result_data), agent="analyst")
                if fn_name == "search_literature":
                    lit_display = format_literature_results(result_data)
                    if lit_display:
                        print(lit_display)
                    # Emit papers to cockpit
                    papers = []
                    for search in result_data.get("searches", []):
                        for r in search.get("results", []):
                            papers.append({"title": r.get("title", ""), "meta": r.get("url", "")})
                    if papers:
                        events.emit("literature_results", papers=papers)
                if fn_name == "generate_figures":
                    events.emit("figures_generated", figures=result_data.get("figures", []))
                if verbose:
                    print(f"    Raw: {json.dumps(result_data, indent=2)[:500]}")

                if fn_name == "write_report" and result_data.get("status") == "success":
                    final_result = result_data

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_str,
                })

        # Check for checkpoint
        if finish_reason == "stop" and msg.get("content") and not checkpoint_done:
            content = msg["content"]
            if _parse_queries(content):
                approved_queries = terminal_checkpoint_queries(content)

                if approved_queries:
                    checkpoint_done = True
                    messages.append({
                        "role": "user",
                        "content": (
                            f"The human has approved the following search queries:\n"
                            + "\n".join(f"{i+1}. {q}" for i, q in enumerate(approved_queries))
                            + f"\n\nPlease call search_literature with these queries, "
                            f"then generate_figures with matrix_path=\"{matrix_path}\" "
                            f"and output_dir=\"{output_dir}\", "
                            f"then write_report with matrix_path=\"{matrix_path}\", "
                            f"figures_dir=\"{output_dir}\", and "
                            f"output_path=\"{output_dir}/report.md\"."
                        ),
                    })
                    continue
                else:
                    checkpoint_done = True
                    lit_path = Path(output_dir)
                    if not lit_path.is_absolute():
                        lit_path = PROJECT_ROOT / lit_path
                    lit_path.mkdir(parents=True, exist_ok=True)
                    lit_file = lit_path / "literature_results.json"
                    with open(lit_file, "w") as f:
                        json.dump({"status": "skipped", "searches": []}, f)
                    messages.append({
                        "role": "user",
                        "content": (
                            "The human skipped literature search. "
                            f"Proceed to generate_figures with matrix_path=\"{matrix_path}\" "
                            f"and output_dir=\"{output_dir}\", "
                            f"then write_report with matrix_path=\"{matrix_path}\", "
                            f"literature_results=\"{lit_file}\", "
                            f"figures_dir=\"{output_dir}\", "
                            f"output_path=\"{output_dir}/report.md\"."
                        ),
                    })
                    continue

        # Final stop
        if finish_reason == "stop" and checkpoint_done:
            if msg.get("content"):
                print(format_nemotron_text(msg["content"]))
                events.emit("nemotron_text", text=msg["content"], agent="analyst")
            print(format_agent_complete("Analyst"))
            events.emit("agent_complete", agent="Analyst")
            break
    else:
        print(f"  ⚠ Analyst: hit max turns ({MAX_TURNS})")

    client.close()
    return final_result or {"error": "No report generated"}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _load_json(path: str) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    with open(p) as f:
        return json.load(f)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AXIS Analyst Agent")
    parser.add_argument("--matrix", type=str, required=True, help="Path to comparison matrix JSON")
    parser.add_argument("--task", type=str, required=True, help="Path to task YAML file")
    parser.add_argument("--verbose", action="store_true", help="Show raw JSON output")
    args = parser.parse_args()

    _load_env()

    task_path = Path(args.task)
    if not task_path.is_absolute():
        task_path = PROJECT_ROOT / task_path
    with open(task_path) as f:
        task = yaml.safe_load(f)

    result = run_agent(matrix_path=args.matrix, task=task, verbose=args.verbose)

    if args.verbose:
        print(f"\n{'='*60}")
        print(f"ANALYST RESULT:")
        print(json.dumps(result, indent=2))
        print(f"{'='*60}")


def _load_env():
    global API_KEY
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                key, val = line.split("=", 1)
                key, val = key.strip(), val.strip()
                if key and not os.environ.get(key):
                    os.environ[key] = val
    API_KEY = os.environ.get("NVIDIA_API_KEY", API_KEY)


if __name__ == "__main__":
    main()
