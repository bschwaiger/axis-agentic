"""
Analyst Agent — Agent 2 in the AXIS Agentic pipeline.

Receives a comparison matrix (1 model x N datasets) from the Coordinator and runs:
1. compare_baselines — analyze metrics across datasets vs. baselines + cross-dataset deltas
2. The model proposes PubMed search queries based on comparative findings
3. Terminal/cockpit checkpoint — human reviews/edits proposed queries
4. search_literature — Anthropic web search scoped to PubMed
5. generate_figures — per-dataset confusion matrices, comparative bar chart, confidence distributions
6. write_report — structured comparative Markdown report with Literature Context

Provider-agnostic: drives any Engine (Anthropic today; OpenAI-compatible in PR2).

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

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analyst_impl import execute_tool  # noqa: E402
from orchestrator.engine import Engine, ToolResult, get_default_engine  # noqa: E402
from orchestrator.formatting import (  # noqa: E402
    format_tool_call, format_tool_result, format_engine_text,
    format_agent_complete, format_checkpoint, format_literature_results,
    _extract_key_args, _summarize_result,
)
from cockpit import events  # noqa: E402

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
- Then propose exactly 3 PubMed search queries. STRICT RULES:
  - 3-5 words MAXIMUM per query. Count the words. If more than 5, shorten.
  - Use MeSH-style keyword phrases, NOT natural language sentences.
  - Queries should cover: (1) the model/task domain, (2) the benchmark/dataset, (3) a specific metric or finding.
  - GOOD examples: "MURA deep learning benchmark", "musculoskeletal radiograph AI", "MCC sensitivity fracture detection"
  - BAD examples (DO NOT USE): "Effect of training set size on model performance", "deep learning for detecting abnormalities in musculoskeletal radiographs"
- Format as a numbered list with ONLY the queries, no explanations after each:
  1. query one
  2. query two
  3. query three
- After proposing queries, STOP and wait for human approval. Do not call any tools.
- When approved queries arrive, call search_literature, then generate_figures, then write_report.
- For write_report, pass: matrix_path, comparison_path, literature_results, figures_dir, output_path.\
"""

VERBOSE = False


def load_tools() -> list[dict]:
    tools_path = PROJECT_ROOT / "tools" / "analyst_tools.json"
    with open(tools_path) as f:
        return json.load(f)


# ------------------------------------------------------------------
# Terminal / cockpit checkpoint
# ------------------------------------------------------------------

def terminal_checkpoint_queries(proposed_text: str) -> tuple[list[str], str]:
    """Display proposed queries (terminal or cockpit), let human edit/approve.

    Returns (approved_queries, reasoning_text). reasoning_text is what was
    emitted to the cockpit so the caller can dedupe later emits against it.
    """
    queries = _parse_queries(proposed_text)

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

    if reasoning_text:
        events.emit("claude_text", text=reasoning_text, agent="analyst")

    if reasoning_lines:
        print()
        print("  Reasoning:")
        for line in reasoning_lines:
            while len(line) > 76:
                print(f"    {line[:76]}")
                line = line[76:]
            print(f"    {line}")

    if events.is_enabled():
        events.emit("checkpoint_queries", queries=queries, reasoning=reasoning_text)
        result = events.wait_for_approval("query_checkpoint", timeout=600.0)
        if result is None or result.get("action") == "skip":
            print("  Literature search skipped.")
            return [], reasoning_text
        approved = result.get("queries", queries)
        print(f"  Approved {len(approved)} queries (via cockpit).")
        return approved, reasoning_text

    print(format_checkpoint("Review proposed PubMed queries below"))
    print()
    for i, q in enumerate(queries, 1):
        print(f"    {i}. {q}")

    print()
    print("  Options: [Enter] Approve  |  [e] Edit  |  [s] Skip")

    choice = input("\n  Your choice: ").strip().lower()

    if choice == "s":
        print("  Literature search skipped.")
        return [], reasoning_text
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
            return custom, reasoning_text
        print("  No queries entered, using original proposals.")

    print(f"  Approved {len(queries)} queries.")
    return queries, reasoning_text


def _is_recap(new_text: str, prior_text: str, prefix_len: int = 60) -> bool:
    """True if new_text opens with the same phrase the model already said.

    The model often opens its post-write_report summary with a recap that
    duplicates the reasoning text emitted at the query checkpoint. Catch
    that by checking whether the start of new_text matches prior_text.
    """
    if not prior_text:
        return False
    norm_new = " ".join(new_text.split()).lower()
    norm_prior = " ".join(prior_text.split()).lower()
    n = min(prefix_len, len(norm_prior))
    if n < 20:
        return False
    return norm_new[:n] == norm_prior[:n]


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

def run_agent(matrix_path: str, task: dict, verbose: bool = False, engine: Engine | None = None) -> dict:
    """Run the Analyst agent loop with comparative analysis."""
    verbose = verbose or VERBOSE
    if engine is None:
        engine = get_default_engine()

    output_dir = task.get("output_dir", "results/comparative")
    model_name = task.get("model_name", "unknown")

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

    tools = load_tools()
    engine.reset(system=SYSTEM_PROMPT, tools=tools, max_tokens=4096)

    response = engine.send_user_message(user_msg)
    checkpoint_done = False
    final_result = None
    last_emitted_text = ""

    for turn in range(1, MAX_TURNS + 1):
        if verbose:
            print(f"--- Turn {turn} ---")

        events.emit("api_call", provider=engine.provider_name, model=engine.model_name,
                     endpoint="messages.create", turn=turn, agent="analyst")
        events.emit("api_response", provider=engine.provider_name, model=engine.model_name,
                     turn=turn, finish_reason=response.stop_reason,
                     elapsed_ms=response.elapsed_ms,
                     prompt_tokens=response.usage.get("input_tokens"),
                     completion_tokens=response.usage.get("output_tokens"),
                     tool_calls=len(response.tool_calls),
                     agent="analyst")

        # Execute tool calls (if any)
        tool_results: list[ToolResult] = []
        for call in response.tool_calls:
            print(format_tool_call(call.name, call.arguments))
            events.emit("tool_call", fn_name=call.name,
                        summary=_extract_key_args(call.name, call.arguments),
                        agent="analyst")
            if verbose:
                print(f"    Args: {json.dumps(call.arguments, indent=2)}")

            result_str = execute_tool(call.name, call.arguments)
            result_data = json.loads(result_str)

            print(format_tool_result(call.name, result_data))
            events.emit("tool_result", fn_name=call.name,
                        summary=_summarize_result(call.name, result_data),
                        agent="analyst")

            # Milestones
            if call.name == "compare_baselines":
                flags = result_data.get("flags", [])
                events.emit("milestone", text=f"Baselines compared, {len(flags)} flags", agent="analyst")
            elif call.name == "search_literature":
                total = result_data.get("total_results", 0)
                events.emit("milestone", text=f"{total} papers found", agent="analyst")
            elif call.name == "generate_figures":
                figs = result_data.get("figures", [])
                events.emit("milestone", text=f"{len(figs)} figures generated", agent="analyst")
            elif call.name == "write_report" and result_data.get("status") == "success":
                wc = result_data.get("word_count", "?")
                events.emit("milestone", text=f"Report written ({wc} words)", agent="analyst")

            if call.name == "search_literature":
                lit_display = format_literature_results(result_data)
                if lit_display:
                    print(lit_display)
                papers = []
                for search in result_data.get("searches", []):
                    for r in search.get("results", []):
                        papers.append({"title": r.get("title", ""), "meta": r.get("url", "")})
                if papers:
                    events.emit("literature_results", papers=papers)
            if call.name == "generate_figures":
                events.emit("figures_generated", figures=result_data.get("figures", []))
            if verbose:
                print(f"    Raw: {json.dumps(result_data, indent=2)[:500]}")

            if call.name == "write_report" and result_data.get("status") == "success":
                final_result = result_data

            tool_results.append(ToolResult(call_id=call.id, content=result_str))

        # Checkpoint: model produced text + no tool calls + checkpoint not done -> propose queries
        if not tool_results and not checkpoint_done and response.text and _parse_queries(response.text):
            approved_queries, last_emitted_text = terminal_checkpoint_queries(response.text)
            checkpoint_done = True

            matrix_dir = Path(matrix_path)
            if not matrix_dir.is_absolute():
                matrix_dir = PROJECT_ROOT / matrix_dir
            comparison_path = str(matrix_dir.parent / "comparison.json")

            if approved_queries:
                literature_path = str(matrix_dir.parent / "literature_results.json")
                next_user = (
                    "The human has approved the following search queries:\n"
                    + "\n".join(f"{i+1}. {q}" for i, q in enumerate(approved_queries))
                    + f"\n\nCall these tools in order:\n"
                    f"1. search_literature with queries={json.dumps(approved_queries)}, output_dir=\"{output_dir}\"\n"
                    f"2. generate_figures with matrix_path=\"{matrix_path}\", output_dir=\"{output_dir}\"\n"
                    f"3. write_report with matrix_path=\"{matrix_path}\", "
                    f"comparison_path=\"{comparison_path}\", "
                    f"literature_results=\"{literature_path}\", "
                    f"figures_dir=\"{output_dir}\", "
                    f"output_path=\"{output_dir}/report.md\""
                )
            else:
                lit_dir = Path(output_dir)
                if not lit_dir.is_absolute():
                    lit_dir = PROJECT_ROOT / lit_dir
                lit_dir.mkdir(parents=True, exist_ok=True)
                lit_file = lit_dir / "literature_results.json"
                with open(lit_file, "w") as f:
                    json.dump({"status": "skipped", "searches": []}, f)
                next_user = (
                    "The human skipped literature search.\n\n"
                    f"Call these tools in order:\n"
                    f"1. generate_figures with matrix_path=\"{matrix_path}\", output_dir=\"{output_dir}\"\n"
                    f"2. write_report with matrix_path=\"{matrix_path}\", "
                    f"comparison_path=\"{comparison_path}\", "
                    f"literature_results=\"{lit_file}\", "
                    f"figures_dir=\"{output_dir}\", "
                    f"output_path=\"{output_dir}/report.md\""
                )

            response = engine.send_user_message(next_user)
            continue

        # No tool calls AND checkpoint already done -> we are finished
        if not tool_results and checkpoint_done:
            if response.text and not _is_recap(response.text, last_emitted_text):
                print(format_engine_text(engine.provider_name, response.text))
                events.emit("claude_text", text=response.text, agent="analyst")
            print(format_agent_complete("Analyst"))
            events.emit("agent_complete", agent="Analyst")
            break

        # Continue: send tool results back
        if tool_results:
            response = engine.send_tool_results(tool_results)
        else:
            # No tool calls, no checkpoint trigger -> done (defensive)
            if response.text and not _is_recap(response.text, last_emitted_text):
                print(format_engine_text(engine.provider_name, response.text))
                events.emit("claude_text", text=response.text, agent="analyst")
            print(format_agent_complete("Analyst"))
            events.emit("agent_complete", agent="Analyst")
            break
    else:
        print(f"  Warning: Analyst hit max turns ({MAX_TURNS})")

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


def _load_env():
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                key, val = line.split("=", 1)
                key, val = key.strip(), val.strip()
                if key and not os.environ.get(key):
                    os.environ[key] = val


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
        print(f"\n{'='*60}\nANALYST RESULT:\n{json.dumps(result, indent=2)}\n{'='*60}")


if __name__ == "__main__":
    main()
