"""Shared formatting helpers for demo-friendly terminal output (Anthropic variant).

Only changes from original: "Nemotron" -> "Claude" in display strings.
All logic and structure identical.
"""
from __future__ import annotations

import json


# ------------------------------------------------------------------
# Tool call summaries (one-line, human-readable)
# ------------------------------------------------------------------

def format_tool_call(fn_name: str, fn_args: dict) -> str:
    """Format a tool call as a concise one-liner."""
    key_args = _extract_key_args(fn_name, fn_args)
    return f"  \u2192 Claude: call {fn_name}.py ({key_args})"


def format_tool_result(fn_name: str, result: dict) -> str:
    """Format a tool result as a concise one-liner."""
    summary = _summarize_result(fn_name, result)
    return f"  \u2190 {fn_name}.py: {summary}"


def format_claude_text(content: str) -> str:
    """Format a Claude text response (truncated)."""
    clean = content.replace("\n", " ").strip()
    if len(clean) > 100:
        clean = clean[:97] + "..."
    return f'  \U0001f4ac Claude: "{clean}"'


def format_agent_complete(agent_name: str, metrics: dict | None = None) -> str:
    """Format a boxed agent completion summary."""
    if metrics:
        parts = []
        for key in ["mcc", "sensitivity", "specificity", "f1"]:
            val = metrics.get(key)
            if val is not None:
                label = key.upper() if key == "mcc" else key.capitalize()[:4]
                parts.append(f"{label}={val:.3f}")
        n = metrics.get("total_evaluated", metrics.get("n", "?"))
        inner = f"{agent_name} complete (n={n}): {', '.join(parts)}"
    else:
        inner = f"{agent_name} complete"
    return f"\u250c {'─' * (len(inner) + 2)} \u2510\n\u2502 {inner}   \u2502\n\u2514 {'─' * (len(inner) + 2)} \u2518"


def format_checkpoint(title: str) -> str:
    """Format a checkpoint header."""
    return f"\n\u23f8  CHECKPOINT: {title}"


# ------------------------------------------------------------------
# Literature results formatting (re-use from original)
# ------------------------------------------------------------------

def format_literature_results(result: dict) -> str:
    """Format search_literature results as a citation list with full metadata from PubMed."""
    # Import the full implementation from original formatting module
    from orchestrator.formatting import format_literature_results as _original
    return _original(result)


# ------------------------------------------------------------------
# Key arg extraction per tool
# ------------------------------------------------------------------

def _extract_key_args(fn_name: str, args: dict) -> str:
    if fn_name == "run_inference":
        ds = args.get("dataset_path", "?")
        return f'dataset="{ds}"'
    elif fn_name == "validate_results":
        pp = args.get("predictions_path", "?")
        return f'predictions="{_short_path(pp)}"'
    elif fn_name == "compute_metrics":
        pp = args.get("predictions_path", "?")
        return f'predictions="{_short_path(pp)}"'
    elif fn_name == "compare_baselines":
        mp = args.get("matrix_path") or args.get("metrics_path", "?")
        return f'matrix="{_short_path(mp)}"'
    elif fn_name == "search_literature":
        queries = args.get("queries", [])
        return f"{len(queries)} queries"
    elif fn_name == "generate_figures":
        od = args.get("output_dir", "?")
        return f'output="{_short_path(od)}"'
    elif fn_name == "write_report":
        op = args.get("output_path", "?")
        return f'output="{_short_path(op)}"'
    else:
        return ", ".join(f"{k}={v!r}" for k, v in list(args.items())[:2])


def _summarize_result(fn_name: str, result: dict) -> str:
    if result.get("error"):
        return f"ERROR: {result['error'][:80]}"

    if fn_name == "run_inference":
        n = result.get("predictions_written", "?")
        errs = result.get("error_count", 0)
        return f"{n} predictions written, {errs} errors"
    elif fn_name == "validate_results":
        status = result.get("status", "?")
        nulls = result.get("null_predictions", 0)
        n = result.get("total_predictions", "?")
        if status == "pass":
            return f"PASS \u2014 {n} predictions validated, {nulls} nulls"
        else:
            issues = result.get("issues", [])
            return f"FAIL \u2014 {'; '.join(issues[:2])}"
    elif fn_name == "compute_metrics":
        s = result.get("summary", {})
        n = s.get("n", "?")
        mcc = s.get("mcc", "?")
        sens = s.get("sensitivity", "?")
        spec = s.get("specificity", "?")
        return f"n={n}, MCC={mcc}, Sens={sens}, Spec={spec}"
    elif fn_name == "compare_baselines":
        status = result.get("status", "?")
        flags = result.get("flags", [])
        summary = result.get("summary", "")
        if flags:
            return f"FLAGGED \u2014 {summary}"
        return f"PASS \u2014 {summary}"
    elif fn_name == "search_literature":
        total = result.get("total_results", 0)
        nq = result.get("total_queries", 0)
        return f"{total} papers found across {nq} queries"
    elif fn_name == "generate_figures":
        figs = result.get("figures", [])
        return f"{len(figs)} figures generated"
    elif fn_name == "write_report":
        wc = result.get("word_count", "?")
        return f"report written ({wc} words)"
    else:
        status = result.get("status", "done")
        return str(status)


def _short_path(path: str) -> str:
    """Shorten a path for display."""
    parts = path.replace("\\", "/").split("/")
    if len(parts) > 3:
        return "/".join(parts[-3:])
    return path
