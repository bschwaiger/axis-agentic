"""Tool implementations for the Analyst agent (multi-dataset comparative analysis)."""
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

    # Per-dataset baseline comparison
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

    # Cross-dataset comparison (pairwise deltas)
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

    # Write comparison JSON
    out = Path(matrix_path).parent / "comparison.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    result["comparison_path"] = str(out)
    return result


# ------------------------------------------------------------------
# search_literature
# ------------------------------------------------------------------

def search_literature(queries: list[str], max_results_per_query: int = 3) -> dict:
    """Search PubMed via Tavily API."""
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("TAVILY_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break
    if not api_key:
        return {"error": "TAVILY_API_KEY not set"}

    from tavily import TavilyClient
    from cockpit import events as cockpit_events
    import time as _time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    client = TavilyClient(api_key=api_key)

    def _search_one(qi, query):
        cockpit_events.emit("tavily_search", query=query, query_index=qi + 1,
                            total_queries=len(queries), domain="pubmed.ncbi.nlm.nih.gov")
        try:
            t0 = _time.monotonic()
            response = client.search(
                query=query,
                max_results=max_results_per_query,
                include_domains=["pubmed.ncbi.nlm.nih.gov"],
                search_depth="advanced",
            )
            elapsed_ms = int((_time.monotonic() - t0) * 1000)
            hits = []
            for r in response.get("results", []):
                hits.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", "")[:500],
                })
            cockpit_events.emit("tavily_result", query=query, hits=len(hits),
                                elapsed_ms=elapsed_ms)
            return {"query": query, "results": hits, "result_count": len(hits)}
        except Exception as e:
            cockpit_events.emit("tavily_result", query=query, hits=0, error=str(e))
            return {"query": query, "results": [], "result_count": 0, "error": str(e)}

    # Run queries concurrently (typically 2-4 queries)
    all_results = [None] * len(queries)
    with ThreadPoolExecutor(max_workers=min(4, len(queries))) as executor:
        futures = {executor.submit(_search_one, qi, q): qi for qi, q in enumerate(queries)}
        for future in as_completed(futures):
            idx = futures[future]
            all_results[idx] = future.result()

    result = {
        "status": "success",
        "total_queries": len(queries),
        "total_results": sum(r["result_count"] for r in all_results),
        "searches": all_results,
    }

    # Write next to the comparison matrix if one exists, otherwise use timestamped path
    from datetime import datetime as _dt
    fallback = PROJECT_ROOT / "results" / f"literature_{_dt.now().strftime('%y%m%d%H%M')}.json"
    # Try to find the run directory from the most recent comparison_matrix.json
    import glob
    matrices = sorted(glob.glob(str(PROJECT_ROOT / "results" / "*" / "comparison_matrix.json")))
    if matrices:
        out = Path(matrices[-1]).parent / "literature_results.json"
    else:
        out = fallback
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    result["literature_path"] = str(out)
    return result


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
        im = ax.imshow(cm_arr, cmap="Blues", interpolation="nearest")
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
    ds_names = [ev["dataset_name"] for ev in evaluations]
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

    # Add baseline markers
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
    """Generate a comparative Markdown evaluation report."""
    matrix = _load_json(matrix_path)
    comparison = _load_json(comparison_path)
    literature = _load_json(literature_results)

    model_name = matrix["model_name"]
    evaluations = matrix["evaluations"]
    baselines = matrix.get("baseline_values", {})

    lines = []
    lines.append(f"# Comparative Evaluation Report: {model_name}")
    lines.append(f"")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ")
    lines.append(f"**Datasets:** {', '.join(ev['dataset_name'] for ev in evaluations)}  ")
    lines.append(f"**Baseline reference:** {json.dumps(baselines)}")
    lines.append(f"")

    # Per-dataset metrics tables
    lines.append(f"## Per-Dataset Metrics")
    lines.append(f"")
    for ev in evaluations:
        m = ev["metrics"]
        n = m.get("total_evaluated", "?")
        lines.append(f"### {ev['dataset_name']} (n={n})")
        lines.append(f"")
        lines.append(f"| Metric | Value | 95% CI |")
        lines.append(f"|--------|-------|--------|")
        for metric in ["mcc", "sensitivity", "specificity", "precision", "npv", "f1", "accuracy"]:
            val = m.get(metric)
            ci = m.get(f"ci_{metric}_95")
            if val is not None:
                ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "—"
                lines.append(f"| {metric.upper()} | {val:.4f} | {ci_str} |")
        lines.append(f"")

    # Cross-dataset comparison table
    lines.append(f"## Cross-Dataset Comparison")
    lines.append(f"")
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
    lines.append(f"")

    # Baseline comparison
    if comparison.get("per_dataset"):
        lines.append(f"## Baseline Comparison")
        lines.append(f"")
        for pd in comparison["per_dataset"]:
            if pd["comparisons"]:
                lines.append(f"### {pd['dataset']}")
                lines.append(f"")
                lines.append(f"| Metric | Baseline | Current | Delta | Status |")
                lines.append(f"|--------|----------|---------|-------|--------|")
                for c in pd["comparisons"]:
                    status = "⚠️ FLAGGED" if c.get("flagged") else "✓"
                    lines.append(f"| {c['metric'].upper()} | {c['baseline']:.4f} | {c['current']:.4f} | {c['delta']:+.4f} | {status} |")
                lines.append(f"")

    # Literature context
    if literature.get("searches"):
        lines.append(f"## Literature Context")
        lines.append(f"")
        for search in literature["searches"]:
            lines.append(f"### Query: \"{search['query']}\"")
            lines.append(f"")
            if search.get("results"):
                for i, r in enumerate(search["results"], 1):
                    lines.append(f"{i}. **{r['title']}**  ")
                    lines.append(f"   {r['url']}  ")
                    if r.get("content"):
                        snippet = r["content"][:200].replace("\n", " ")
                        lines.append(f"   > {snippet}...")
                    lines.append(f"")
            else:
                lines.append(f"No results found.")
                lines.append(f"")

    # Figures
    fig_dir = Path(figures_dir)
    if not fig_dir.is_absolute():
        fig_dir = PROJECT_ROOT / fig_dir
    if fig_dir.exists():
        pngs = sorted(fig_dir.glob("*.png"))
        if pngs:
            lines.append(f"## Figures")
            lines.append(f"")
            for png in pngs:
                lines.append(f"![{png.stem}]({png.name})")
                lines.append(f"")

    lines.append(f"---")
    lines.append(f"*Generated by AXIS Agentic evaluation pipeline*")

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
