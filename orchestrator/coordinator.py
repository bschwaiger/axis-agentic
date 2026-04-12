"""
Coordinator — Outer loop for the AXIS Agentic pipeline.

Interactive model and dataset selection. Runs Evaluator per dataset, collects
metrics into a comparison matrix, then hands the full matrix to the Analyst
for cross-dataset comparative analysis.

Usage:
    python -m orchestrator.coordinator
    python -m orchestrator.coordinator --verbose
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from orchestrator.evaluator_agent import run_agent as run_evaluator  # noqa: E402
from orchestrator.analyst_agent import run_agent as run_analyst  # noqa: E402
from orchestrator.formatting import format_checkpoint  # noqa: E402
from cockpit import events  # noqa: E402


# ------------------------------------------------------------------
# Available models and datasets
# ------------------------------------------------------------------

AVAILABLE_MODELS = [
    {
        "name": "AXIS-MURA-v1 4-bit",
        "id": "axis-mura-v1-4bit",
        "description": "LoRA fine-tuned MedGemma 1.5 4B, MLX 4-bit quantized",
        "baseline_values": {
            "mcc": 0.637,
            "sensitivity": 0.743,
            "specificity": 0.920,
        },
        "baseline_source": "AXIS-MURA-v1 on full MURA valid set (n=896)",
    },
    {
        "name": "MedGemma 1.5 4B base",
        "id": "medgemma-1.5-4b-base",
        "description": "Unmodified base model (no LoRA), MLX 4-bit quantized",
        "baseline_values": {},
        "baseline_source": "",
    },
]

AVAILABLE_DATASETS = [
    {
        "name": "eval-018",
        "description": "Demo subset (18 studies, ~2 min)",
        "dataset_path": "data/eval-018",
        "manifest_path": "data/eval-018/manifest.csv",
    },
    {
        "name": "eval-020",
        "description": "Debug subset (20 studies, ~2 min)",
        "dataset_path": "data/eval-020",
        "manifest_path": "data/eval-020/manifest.csv",
    },
    {
        "name": "eval-100",
        "description": "Demo subset (100 studies, ~10 min)",
        "dataset_path": "data/eval-100",
        "manifest_path": "data/eval-100/manifest.csv",
    },
    {
        "name": "eval-300",
        "description": "Full evaluation (300 studies, ~30 min)",
        "dataset_path": "data/eval-300",
        "manifest_path": "data/eval-300/manifest.csv",
    },
]


# ------------------------------------------------------------------
# Interactive menu
# ------------------------------------------------------------------

def interactive_setup() -> dict:
    """Interactive model and dataset selection. Returns a config dict."""
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║              AXIS AGENTIC — Evaluation Pipeline              ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # Model selection
    print("  Select model:")
    for i, m in enumerate(AVAILABLE_MODELS, 1):
        print(f"    [{i}] {m['name']}  —  {m['description']}")
    print()
    print("  Comma-separated for multiple models. For a new model, drag & drop")
    print("  or enter path to folder containing model.safetensors.")
    print()

    while True:
        choice = input(f"  Model: ").strip() or "1"
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(AVAILABLE_MODELS):
                model = AVAILABLE_MODELS[idx]
                break
        except ValueError:
            pass
        print(f"  Invalid choice. Enter 1-{len(AVAILABLE_MODELS)}.")

    print(f"  ✓ {model['name']}")
    print()

    # Dataset selection
    print("  Select evaluation datasets:")
    for i, ds in enumerate(AVAILABLE_DATASETS, 1):
        print(f"    [{i}] {ds['name']}  —  {ds['description']}")
    print()
    print("  Comma-separated for multiple datasets. For a new dataset, drag & drop")
    print("  or enter path to folder containing manifest.csv.")
    print()

    while True:
        choice = input(f"  Datasets: ").strip() or "1,3"
        try:
            indices = [int(x.strip()) - 1 for x in choice.split(",")]
            if all(0 <= idx < len(AVAILABLE_DATASETS) for idx in indices) and indices:
                datasets = [AVAILABLE_DATASETS[idx].copy() for idx in indices]
                break
        except ValueError:
            pass
        print(f"  Invalid choice. Enter comma-separated numbers 1-{len(AVAILABLE_DATASETS)}.")

    ds_names = ", ".join(ds["name"] for ds in datasets)
    print(f"  ✓ {ds_names}")
    print()

    return {
        "model_name": model["id"],
        "datasets": datasets,
        "baseline_values": model["baseline_values"],
        "baseline_source": model["baseline_source"],
    }


# ------------------------------------------------------------------
# Comparison matrix checkpoint
# ------------------------------------------------------------------

def matrix_checkpoint(matrix: dict) -> bool:
    """Display full comparison matrix and ask human to approve."""
    print(format_checkpoint("Review comparison matrix"))
    print()

    datasets = [e["dataset_name"] for e in matrix["evaluations"]]
    col_w = max(12, max(len(d) for d in datasets) + 2)

    header = f"  {'Metric':<15s}" + "".join(f"{d:>{col_w}s}" for d in datasets)
    print(header)
    print("  " + "─" * (15 + col_w * len(datasets)))

    for m in ["mcc", "sensitivity", "specificity", "precision", "f1", "accuracy"]:
        row = f"  {m.upper():<15s}"
        for ev in matrix["evaluations"]:
            val = ev["metrics"].get(m)
            row += f"{val:>{col_w}.4f}" if val is not None else f"{'—':>{col_w}s}"
        print(row)

    print()
    for ev in matrix["evaluations"]:
        cm = ev["metrics"].get("confusion_matrix", [[0, 0], [0, 0]])
        n = ev["metrics"].get("total_evaluated", "?")
        print(f"  {ev['dataset_name']} (n={n}): TP={cm[1][1]} FP={cm[0][1]} FN={cm[1][0]} TN={cm[0][0]}")

    # Cockpit mode: emit event and wait for HTTP approval
    if events.is_enabled():
        events.emit("checkpoint_matrix", matrix=matrix)
        result = events.wait_for_approval("matrix_checkpoint", timeout=600.0)
        if result is None or result.get("action") == "abort":
            print("  Pipeline aborted.")
            return False
        print("  Metrics approved. Proceeding to Analyst.\n")
        return True

    print()
    print("  Options: [Enter] Approve  |  [a] Abort")

    choice = input("\n  Your choice: ").strip().lower()
    if choice == "a":
        print("  Pipeline aborted.")
        return False
    print("  Metrics approved. Proceeding to Analyst.\n")
    return True


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------

def run_pipeline(config: dict, verbose: bool = False):
    """Run the full Coordinator pipeline: 1 model × N datasets."""
    model_name = config["model_name"]
    datasets = config["datasets"]
    baseline_values = config.get("baseline_values", {})
    baseline_source = config.get("baseline_source", "")

    # Timestamped run directory
    timestamp = datetime.now().strftime("%y%m%d%H%M")
    run_type = "comparative" if len(datasets) > 1 else "single"
    run_dir = f"results/{timestamp}_{run_type}"

    # Set per-dataset output_dirs under the run directory
    for ds in datasets:
        ds["output_dir"] = f"{run_dir}/{ds['name']}"

    output_dir = run_dir
    n_datasets = len(datasets)
    ds_list = ", ".join(d["name"] for d in datasets)

    print(f"  Model:    {model_name}")
    print(f"  Datasets: {ds_list} ({n_datasets} total)")
    print(f"  Output:   {output_dir}")
    print()

    events.emit("pipeline_start",
                model=model_name,
                datasets=[d["name"] for d in datasets],
                output_dir=output_dir)

    # ----------------------------------------------------------
    # Phase 1: Run Evaluator for each dataset
    # ----------------------------------------------------------
    matrix = {
        "model_name": model_name,
        "baseline_values": baseline_values,
        "baseline_source": baseline_source,
        "evaluations": [],
    }

    for i, ds in enumerate(datasets, 1):
        print(f"▶ Phase 1.{i}: Evaluator — {ds['name']}")
        events.emit("evaluator_start", dataset=ds["name"], eval_index=i - 1)

        eval_task = {
            "model_name": model_name,
            "dataset_path": ds["dataset_path"],
            "manifest_path": ds["manifest_path"],
            "output_dir": ds["output_dir"],
            "baseline_values": baseline_values,
        }

        result = run_evaluator(eval_task, verbose=verbose)

        if result.get("error") and not result.get("metrics_path"):
            print(f"  ⚠ Evaluator failed on {ds['name']}: {result.get('error')}")
            events.emit("evaluator_error", dataset=ds["name"], eval_index=i - 1, error=result.get("error", ""))
            continue

        metrics_path = result.get("metrics_path", "")
        predictions_path = str(Path(ds["output_dir"]) / "predictions.csv")
        if not Path(predictions_path).is_absolute():
            predictions_path = str(PROJECT_ROOT / predictions_path)

        metrics = _load_json(metrics_path) if metrics_path else {}

        matrix["evaluations"].append({
            "dataset_name": ds["name"],
            "dataset_path": ds["dataset_path"],
            "metrics_path": metrics_path,
            "predictions_path": predictions_path,
            "output_dir": ds["output_dir"],
            "metrics": metrics,
        })

        events.emit("evaluator_done", dataset=ds["name"], eval_index=i - 1, metrics=metrics)
        print()

    if not matrix["evaluations"]:
        print("  ✗ No successful evaluations. Aborting.")
        return

    # Write matrix JSON
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = str(out_dir / "comparison_matrix.json")
    with open(matrix_path, "w") as f:
        json.dump(matrix, f, indent=2)

    # ----------------------------------------------------------
    # Phase 2: Comparison matrix checkpoint
    # ----------------------------------------------------------
    if not matrix_checkpoint(matrix):
        return

    # ----------------------------------------------------------
    # Phase 3: Analyst (comparative analysis)
    # ----------------------------------------------------------
    print("▶ Phase 2: Analyst — Comparative Analysis")
    events.emit("analyst_start")

    analyst_config = {
        "model_name": model_name,
        "output_dir": output_dir,
        "baseline_values": baseline_values,
    }

    analyst_result = run_analyst(
        matrix_path=matrix_path,
        task=analyst_config,
        verbose=verbose,
    )

    events.emit("analyst_done", report_path=analyst_result.get("report_path", ""))

    # ----------------------------------------------------------
    # Phase 4: Summary
    # ----------------------------------------------------------
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                     PIPELINE COMPLETE                       ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    _print_results_summary(matrix, out_dir)

    if analyst_result.get("report_path"):
        rp = analyst_result["report_path"]
        try:
            rp = str(Path(rp).relative_to(PROJECT_ROOT))
        except ValueError:
            pass
        print(f"\n  Report: {rp}")

    all_dirs = [out_dir] + [
        PROJECT_ROOT / ev["output_dir"] if not Path(ev["output_dir"]).is_absolute()
        else Path(ev["output_dir"])
        for ev in matrix["evaluations"]
    ]
    total_files = 0
    for d in all_dirs:
        if d.exists():
            total_files += sum(1 for f in d.iterdir() if f.is_file())

    print(f"  Output: {total_files} files in {output_dir}/")
    print()

    events.emit("pipeline_complete", total_files=total_files, output_dir=output_dir)


# ------------------------------------------------------------------
# Results summary
# ------------------------------------------------------------------

def _print_results_summary(matrix: dict, out_dir: Path):
    """Print a human-readable results summary at pipeline end."""
    evaluations = matrix["evaluations"]
    baselines = matrix.get("baseline_values", {})
    baseline_source = matrix.get("baseline_source", "")
    is_comparative = len(evaluations) >= 2

    # Build ASCII lines for cockpit emission
    ascii_lines: list[str] = []

    # Metrics table with delta column for comparative runs
    print()
    datasets = [e["dataset_name"] for e in evaluations]
    col_w = max(12, max(len(d) for d in datasets) + 2)
    delta_w = 10

    header = f"{'Metric':<15s}" + "".join(f"{d:>{col_w}s}" for d in datasets)
    if is_comparative:
        header += f"{'Delta':>{delta_w}s}"
    print(f"  {header}")
    ascii_lines.append(header)
    sep = "─" * (15 + col_w * len(datasets) + (delta_w if is_comparative else 0))
    print(f"  {sep}")
    ascii_lines.append(sep)

    for m in ["mcc", "sensitivity", "specificity", "precision", "f1", "accuracy"]:
        row = f"{m.upper():<15s}"
        vals = []
        for ev in evaluations:
            val = ev["metrics"].get(m)
            vals.append(val)
            row += f"{val:>{col_w}.4f}" if val is not None else f"{'—':>{col_w}s}"
        if is_comparative and all(v is not None for v in vals):
            delta = vals[-1] - vals[0]
            delta_str = f"{delta:+.4f}"
            row += f"{delta_str:>{delta_w}s}"
        elif is_comparative:
            row += f"{'—':>{delta_w}s}"
        print(f"  {row}")
        ascii_lines.append(row)

    # Baseline reference (compact, not the main comparison)
    if baselines and baseline_source:
        print()
        bl_line = f"Ref. baseline ({baseline_source}): {', '.join(f'{k.upper()}={v}' for k, v in baselines.items())}"
        print(f"  {bl_line}")
        ascii_lines.append("")
        ascii_lines.append(bl_line)

    # Side-by-side ASCII confusion matrices for comparative
    print()
    cm_lines: list[str] = []
    if is_comparative and len(evaluations) == 2:
        ev_a, ev_b = evaluations[0], evaluations[1]
        cm_a = ev_a["metrics"].get("confusion_matrix", [[0, 0], [0, 0]])
        cm_b = ev_b["metrics"].get("confusion_matrix", [[0, 0], [0, 0]])
        n_a = ev_a["metrics"].get("total_evaluated", "?")
        n_b = ev_b["metrics"].get("total_evaluated", "?")

        cm_lines.append(f"{ev_a['dataset_name']} (n={n_a}){'':>20s}{ev_b['dataset_name']} (n={n_b})")
        cm_lines.append(f"┌─────────┬─────────┐{'':>8s}┌─────────┬─────────┐")
        cm_lines.append(f"│ TN={cm_a[0][0]:<4d} │ FP={cm_a[0][1]:<4d} │{'':>8s}│ TN={cm_b[0][0]:<4d} │ FP={cm_b[0][1]:<4d} │")
        cm_lines.append(f"├─────────┼─────────┤{'':>8s}├─────────┼─────────┤")
        cm_lines.append(f"│ FN={cm_a[1][0]:<4d} │ TP={cm_a[1][1]:<4d} │{'':>8s}│ FN={cm_b[1][0]:<4d} │ TP={cm_b[1][1]:<4d} │")
        cm_lines.append(f"└─────────┴─────────┘{'':>8s}└─────────┴─────────┘")
    else:
        for ev in evaluations:
            cm = ev["metrics"].get("confusion_matrix", [[0, 0], [0, 0]])
            n = ev["metrics"].get("total_evaluated", "?")
            tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
            cm_lines.append(f"{ev['dataset_name']} (n={n}):")
            cm_lines.append(f"  ┌─────────┬─────────┐")
            cm_lines.append(f"  │ TN={tn:<4d} │ FP={fp:<4d} │")
            cm_lines.append(f"  ├─────────┼─────────┤")
            cm_lines.append(f"  │ FN={fn:<4d} │ TP={tp:<4d} │")
            cm_lines.append(f"  └─────────┴─────────┘")
            cm_lines.append("")
    for line in cm_lines:
        print(f"  {line}")

    # ASCII bar chart — datasets side by side, baseline as reference line
    bar_w = 30
    bar_lines: list[str] = []
    title = "Cross-dataset comparison:" if is_comparative else "Metrics:"
    print(f"\n  {title}\n")
    bar_lines.append(title)
    bar_lines.append("")
    for m in ["mcc", "sensitivity", "specificity"]:
        print(f"  {m.upper()}")
        bar_lines.append(m.upper())
        for ev in evaluations:
            val = ev["metrics"].get(m, 0)
            filled = int(val * bar_w)
            bar = "█" * filled + "░" * (bar_w - filled)
            line = f"  {ev['dataset_name']:<12s} {bar} {val:.3f}"
            print(f"  {line}")
            bar_lines.append(line)
        if baselines and m in baselines:
            bval = baselines[m]
            pos = int(bval * bar_w)
            ref_line = "·" * pos + "│" + "·" * (bar_w - pos - 1)
            line = f"  {'ref.baseline':<12s} {ref_line} {bval:.3f}"
            print(f"  {line}")
            bar_lines.append(line)
        if is_comparative:
            vals = [ev["metrics"].get(m) for ev in evaluations]
            if all(v is not None for v in vals):
                delta = vals[-1] - vals[0]
                sign = "+" if delta >= 0 else ""
                arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
                line = f"  {'delta':<12s} {arrow} {sign}{delta:.3f} ({datasets[0]} → {datasets[-1]})"
                print(f"  {line}")
                bar_lines.append(line)
        print()
        bar_lines.append("")

    # Emit ASCII results to cockpit
    events.emit("ascii_results",
                table="\n".join(ascii_lines),
                confusion_matrices="\n".join(cm_lines),
                bar_chart="\n".join(bar_lines))

    # Literature citations
    lit_path = out_dir / "literature_results.json"
    if lit_path.exists():
        from orchestrator.formatting import format_literature_results
        lit = _load_json(str(lit_path))
        if lit.get("searches"):
            lit_display = format_literature_results(lit)
            if lit_display:
                print(lit_display)

    # Figures
    pngs = sorted(out_dir.glob("*.png"))
    if pngs:
        print("  Figures:")
        for p in pngs:
            name = p.stem.replace("_", " ").title()
            print(f"    📊 {name} → {p.name}")


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
    parser = argparse.ArgumentParser(description="AXIS Agentic Coordinator")
    parser.add_argument("--verbose", action="store_true", help="Show raw JSON output from agents")
    args = parser.parse_args()

    _load_env()
    config = interactive_setup()
    run_pipeline(config, verbose=args.verbose)


if __name__ == "__main__":
    main()
