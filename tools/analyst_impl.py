"""Tool implementations for the Analyst agent (multi-dataset comparative analysis).

search_literature uses Anthropic web search (web_search_20250305) scoped to
PubMed. This requires ANTHROPIC_API_KEY even when the orchestrator engine
is something else; future PRs may add a Tavily fallback.
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------
# compare_baselines (multi-dataset)
# ------------------------------------------------------------------

def compare_baselines(matrix_path: str, threshold: float = 0.05) -> dict:
    """Compare each dataset's metrics against baselines and against each other."""
    matrix = _load_json(matrix_path)
    baselines = matrix.get("baseline_values", {})
    evaluations = matrix.get("evaluations", [])

    per_dataset = []
    for ev in evaluations:
        ds_name = ev["dataset_name"]
        metrics = ev["metrics"]
        comparisons = []
        flags = []
        for metric_name, baseline_val in baselines.items():
            current_val = metrics.get(metric_name)
            if current_val is None:
                continue
            delta = current_val - baseline_val
            flagged = abs(delta) > threshold
            direction = "improved" if delta > 0 else "degraded"
            comp = {
                "metric": metric_name,
                "baseline": baseline_val,
                "current": current_val,
                "delta": round(delta, 4),
                "flagged": flagged,
                "direction": direction,
            }
            comparisons.append(comp)
            if flagged:
                flags.append(f"{ds_name}/{metric_name}: {direction} by {abs(delta):.4f}")

        per_dataset.append({
            "dataset": ds_name,
            "comparisons": comparisons,
            "flags": flags,
        })

    cross_dataset = []
    metric_names = ["mcc", "sensitivity", "specificity", "precision", "f1", "accuracy"]
    for i in range(len(evaluations)):
        for j in range(i + 1, len(evaluations)):
            ds_a = evaluations[i]
            ds_b = evaluations[j]
            deltas = {}
            for m in metric_names:
                val_a = ds_a["metrics"].get(m)
                val_b = ds_b["metrics"].get(m)
                if val_a is not None and val_b is not None:
                    deltas[m] = round(val_b - val_a, 4)
            cross_dataset.append({
                "dataset_a": ds_a["dataset_name"],
                "dataset_b": ds_b["dataset_name"],
                "deltas": deltas,
            })

    all_flags = [f for pd in per_dataset for f in pd["flags"]]
    result = {
        "status": "flagged" if all_flags else "pass",
        "per_dataset": per_dataset,
        "cross_dataset": cross_dataset,
        "flags": all_flags,
        "summary": f"{len(all_flags)} flag(s) across {len(evaluations)} datasets",
    }

    out = Path(matrix_path).parent / "comparison.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    result["comparison_path"] = str(out)
    return result


# ------------------------------------------------------------------
# search_literature (Anthropic web search, PubMed-scoped)
# ------------------------------------------------------------------

def search_literature(queries: list[str], max_results_per_query: int = 3, output_dir: str = "") -> dict:
    """Search PubMed via Anthropic web search.

    Each query is scoped to pubmed.ncbi.nlm.nih.gov via allowed_domains. The
    Anthropic SDK handles the actual web traffic; we only parse the structured
    web_search_tool_result blocks back into our citation format.
    """
    import anthropic
    import time as _time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from cockpit import events as cockpit_events

    client = anthropic.Anthropic()
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    def _search_one(qi: int, query: str) -> dict:
        cockpit_events.emit("web_search", query=query, query_index=qi + 1,
                            total_queries=len(queries), domain="pubmed.ncbi.nlm.nih.gov")
        try:
            t0 = _time.monotonic()
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                tools=[{
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": max_results_per_query + 2,
                    "allowed_domains": ["pubmed.ncbi.nlm.nih.gov"],
                }],
                messages=[{
                    "role": "user",
                    "content": (
                        f"Search PubMed for: {query}\n\n"
                        f"Return up to {max_results_per_query} relevant papers from PubMed. "
                        f"For each paper, provide: title, PubMed URL, and a one-sentence summary."
                    ),
                }],
            )
            elapsed_ms = int((_time.monotonic() - t0) * 1000)
            hits = _parse_search_response(response)
            cockpit_events.emit("web_search_result", query=query, hits=len(hits),
                                elapsed_ms=elapsed_ms)
            return {"query": query, "results": hits[:max_results_per_query], "result_count": len(hits[:max_results_per_query])}
        except Exception as e:
            cockpit_events.emit("web_search_result", query=query, hits=0, error=str(e))
            return {"query": query, "results": [], "result_count": 0, "error": str(e)}

    all_results: list[dict | None] = [None] * len(queries)
    with ThreadPoolExecutor(max_workers=min(4, len(queries))) as executor:
        futures = {executor.submit(_search_one, qi, q): qi for qi, q in enumerate(queries)}
        for future in as_completed(futures):
            idx = futures[future]
            all_results[idx] = future.result()

    result = {
        "status": "success",
        "total_queries": len(queries),
        "total_results": sum(r["result_count"] for r in all_results if r),
        "searches": all_results,
    }

    if output_dir:
        out_d = Path(output_dir)
        if not out_d.is_absolute():
            out_d = PROJECT_ROOT / out_d
        out = out_d / "literature_results.json"
    else:
        import glob
        matrices = sorted(glob.glob(str(PROJECT_ROOT / "results" / "*" / "comparison_matrix.json")))
        if matrices:
            out = Path(matrices[-1]).parent / "literature_results.json"
        else:
            out = PROJECT_ROOT / "results" / f"literature_{datetime.now().strftime('%y%m%d%H%M')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    result["literature_path"] = str(out)
    result["output_note"] = f"Literature results written to {out}. Pass this path as literature_results to write_report."
    return result


def _parse_search_response(response) -> list[dict]:
    """Parse an Anthropic web_search_tool_result into structured hits."""
    import re
    hits = []
    seen_urls: set[str] = set()

    for block in response.content:
        if getattr(block, "type", "") == "web_search_tool_result":
            content_list = getattr(block, "content", []) or []
            if not isinstance(content_list, list):
                content_list = []
            for item in content_list:
                if getattr(item, "type", "") == "web_search_result":
                    url = getattr(item, "url", "")
                    title = getattr(item, "title", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        hits.append({
                            "title": title.replace(" - PubMed", "").strip(),
                            "url": url,
                            "content": "",
                        })

    for block in response.content:
        if getattr(block, "type", "") == "text" and getattr(block, "text", ""):
            urls = re.findall(r'https?://pubmed\.ncbi\.nlm\.nih\.gov/\d+/?', block.text)
            for url in urls:
                if url not in seen_urls:
                    seen_urls.add(url)
                    hits.append({"title": "", "url": url, "content": ""})

    return hits


# ------------------------------------------------------------------
# generate_figures (multi-dataset)
# ------------------------------------------------------------------

def generate_figures(matrix_path: str, output_dir: str) -> dict:
    """Generate per-dataset and comparative figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    matrix = _load_json(matrix_path)
    evaluations = matrix["evaluations"]
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    figures = []

    # 1. Per-dataset confusion matrices (side by side)
    n_ds = len(evaluations)
    fig, axes = plt.subplots(1, n_ds, figsize=(6 * n_ds, 5))
    if n_ds == 1:
        axes = [axes]
    for ax, ev in zip(axes, evaluations):
        cm = ev["metrics"].get("confusion_matrix", [[0, 0], [0, 0]])
        cm_arr = np.array(cm)
        ax.imshow(cm_arr, cmap="Blues", interpolation="nearest")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "Abnormal"])
        ax.set_yticklabels(["Normal", "Abnormal"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        n = ev["metrics"].get("total_evaluated", "?")
        ax.set_title(f"{ev['dataset_name']} (n={n})")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center",
                        color="white" if cm_arr[i, j] > cm_arr.max() / 2 else "black",
                        fontsize=18, fontweight="bold")
    fig.suptitle(f"Confusion Matrices — {matrix['model_name']}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    cm_path = out_dir / "confusion_matrices.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    figures.append(str(cm_path))

    # 2. Comparative metrics bar chart
    metric_names = ["mcc", "sensitivity", "specificity", "precision", "f1", "accuracy"]
    x = np.arange(len(metric_names))
    width = 0.8 / n_ds
    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, ev in enumerate(evaluations):
        vals = [ev["metrics"].get(m, 0) for m in metric_names]
        offset = (i - (n_ds - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=ev["dataset_name"],
                       color=colors[i % len(colors)], alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    baselines = matrix.get("baseline_values", {})
    for j, m in enumerate(metric_names):
        if m in baselines:
            ax.hlines(baselines[m], j - 0.45, j + 0.45,
                      colors="black", linestyles="dashed", linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metric_names])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title(f"Comparative Metrics — {matrix['model_name']}", fontweight="bold")
    ax.legend()
    if baselines:
        ax.plot([], [], color="black", linestyle="dashed", label="Baseline")
        ax.legend()
    fig.tight_layout()
    bar_path = out_dir / "comparative_metrics.png"
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)
    figures.append(str(bar_path))

    # 3. Confidence distributions (overlaid)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, ev in enumerate(evaluations):
        preds_path = Path(ev.get("predictions_path", ""))
        if not preds_path.is_absolute():
            preds_path = PROJECT_ROOT / preds_path
        if not preds_path.exists():
            continue
        with open(preds_path) as f:
            preds = list(csv.DictReader(f))
        confs = [float(r["confidence"]) for r in preds
                 if r.get("confidence") and r["confidence"] != ""]
        if confs:
            ax.hist(confs, bins=20, range=(0, 1), alpha=0.5,
                    label=f"{ev['dataset_name']} (n={len(confs)})",
                    color=colors[i % len(colors)], edgecolor="white")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution Comparison", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    conf_path = out_dir / "confidence_comparison.png"
    fig.savefig(conf_path, dpi=150)
    plt.close(fig)
    figures.append(str(conf_path))

    return {
        "status": "success",
        "figures": figures,
        "output_dir": str(out_dir),
    }


# ------------------------------------------------------------------
# write_report (comparative)
# ------------------------------------------------------------------

def write_report(
    matrix_path: str,
    comparison_path: str,
    literature_results: str,
    figures_dir: str,
    output_path: str,
) -> dict:
    """Generate a comparative Markdown evaluation report.

    If literature_results points to a missing file, write an empty
    placeholder so report generation does not crash.
    """
    lit_path = Path(literature_results)
    if not lit_path.is_absolute():
        lit_path = PROJECT_ROOT / lit_path
    if not lit_path.exists():
        matrix_dir = Path(matrix_path)
        if not matrix_dir.is_absolute():
            matrix_dir = PROJECT_ROOT / matrix_dir
        alt_path = matrix_dir.parent / "literature_results.json"
        if alt_path.exists():
            literature_results = str(alt_path)
        else:
            alt_path.parent.mkdir(parents=True, exist_ok=True)
            with open(alt_path, "w") as f:
                json.dump({"status": "not_found", "searches": []}, f)
            literature_results = str(alt_path)

    matrix = _load_json(matrix_path)
    comparison = _load_json(comparison_path)
    literature = _load_json(literature_results)

    model_name = matrix["model_name"]
    evaluations = matrix["evaluations"]
    baselines = matrix.get("baseline_values", {})

    lines = []
    lines.append(f"# Comparative Evaluation Report: {model_name}")
    lines.append("")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ")
    lines.append(f"**Datasets:** {', '.join(ev['dataset_name'] for ev in evaluations)}  ")
    lines.append(f"**Baseline reference:** {json.dumps(baselines)}")
    lines.append("")

    lines.append("## Per-Dataset Metrics")
    lines.append("")
    for ev in evaluations:
        m = ev["metrics"]
        n = m.get("total_evaluated", "?")
        lines.append(f"### {ev['dataset_name']} (n={n})")
        lines.append("")
        lines.append("| Metric | Value | 95% CI |")
        lines.append("|--------|-------|--------|")
        for metric in ["mcc", "sensitivity", "specificity", "precision", "npv", "f1", "accuracy"]:
            val = m.get(metric)
            ci = m.get(f"ci_{metric}_95")
            if val is not None:
                ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "—"
                lines.append(f"| {metric.upper()} | {val:.4f} | {ci_str} |")
        lines.append("")

    lines.append("## Cross-Dataset Comparison")
    lines.append("")
    metric_names = ["mcc", "sensitivity", "specificity", "precision", "f1", "accuracy"]
    header = "| Metric |" + "|".join(f" {ev['dataset_name']} " for ev in evaluations) + "| Delta |"
    sep = "|--------|" + "|".join("-------" for _ in evaluations) + "|-------|"
    lines.append(header)
    lines.append(sep)
    for metric in metric_names:
        vals = [ev["metrics"].get(metric) for ev in evaluations]
        row = f"| {metric.upper()} |"
        for v in vals:
            row += f" {v:.4f} |" if v is not None else " — |"
        if len(vals) >= 2 and all(v is not None for v in vals):
            delta = vals[-1] - vals[0]
            row += f" {delta:+.4f} |"
        else:
            row += " — |"
        lines.append(row)
    lines.append("")

    if comparison.get("per_dataset"):
        lines.append("## Baseline Comparison")
        lines.append("")
        for pd in comparison["per_dataset"]:
            if pd["comparisons"]:
                lines.append(f"### {pd['dataset']}")
                lines.append("")
                lines.append("| Metric | Baseline | Current | Delta | Status |")
                lines.append("|--------|----------|---------|-------|--------|")
                for c in pd["comparisons"]:
                    status = "⚠️ FLAGGED" if c.get("flagged") else "✓"
                    lines.append(f"| {c['metric'].upper()} | {c['baseline']:.4f} | {c['current']:.4f} | {c['delta']:+.4f} | {status} |")
                lines.append("")

    if literature.get("searches"):
        lines.append("## Literature Context")
        lines.append("")
        for search in literature["searches"]:
            lines.append(f"### Query: \"{search['query']}\"")
            lines.append("")
            if search.get("results"):
                for i, r in enumerate(search["results"], 1):
                    lines.append(f"{i}. **{r['title']}**  ")
                    lines.append(f"   {r['url']}  ")
                    if r.get("content"):
                        snippet = r["content"][:200].replace("\n", " ")
                        lines.append(f"   > {snippet}...")
                    lines.append("")
            else:
                lines.append("No results found.")
                lines.append("")

    fig_dir = Path(figures_dir)
    if not fig_dir.is_absolute():
        fig_dir = PROJECT_ROOT / fig_dir
    if fig_dir.exists():
        pngs = sorted(fig_dir.glob("*.png"))
        if pngs:
            lines.append("## Figures")
            lines.append("")
            for png in pngs:
                lines.append(f"![{png.stem}]({png.name})")
                lines.append("")

    lines.append("---")
    lines.append("*Generated by AXIS Agentic evaluation pipeline*")

    report_text = "\n".join(lines)

    out = Path(output_path)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(report_text)

    return {
        "status": "success",
        "report_path": str(out),
        "word_count": len(report_text.split()),
    }


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
# Dispatch
# ------------------------------------------------------------------

TOOL_FUNCTIONS = {
    "compare_baselines": compare_baselines,
    "search_literature": search_literature,
    "generate_figures": generate_figures,
    "write_report": write_report,
}


def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool by name, return JSON string result."""
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        result = fn(**arguments)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})
