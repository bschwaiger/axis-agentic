#!/usr/bin/env python3
"""
AXIS — Cross-Run Comparison for Publication
Merges results from multiple evaluation runs (same manifest) and computes:
  - Side-by-side metrics table (overall + per-anatomy)
  - AUC with DeLong pairwise comparison
  - McNemar's test (pairwise)
  - Bootstrap CIs for MCC and Cohen's kappa
  - ROC curves (PNG)
  - Calibration plots (PNG)

Usage:
    python3 compare_runs.py \
        --runs ../results/run_gemma3 ../results/run_medgemma ../results/run_finetuned \
        --labels "Gemma 3" "MedGemma" "AXIS-MURA-v1" \
        --output ../results/comparison

    python3 compare_runs.py \
        --runs ../results/run_gemma3 ../results/run_medgemma \
        --labels "Gemma 3" "MedGemma" \
        --output ../results/comparison \
        --no-plots
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

try:
    from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ============================================================
# LOADING
# ============================================================

def load_run(run_dir: Path) -> tuple[pd.DataFrame, dict, dict | None]:
    """Load per-study CSV, metrics JSON, and per-bodypart JSON from a run directory.

    Returns (per_study_df, metrics_dict, per_bodypart_dict_or_None).
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # Find the per-study CSV (the one without _views or _metrics suffix)
    csvs = sorted(run_dir.glob("*.csv"))
    study_csv = None
    for c in csvs:
        if "_views" not in c.stem and "_per_bodypart" not in c.stem:
            study_csv = c
            break
    if study_csv is None:
        raise FileNotFoundError(f"No per-study CSV found in {run_dir}")

    df = pd.read_csv(study_csv)

    # Load metrics JSON
    metrics_json = sorted(run_dir.glob("*_metrics.json"))
    metrics = {}
    if metrics_json:
        with open(metrics_json[0]) as f:
            metrics = json.load(f)

    # Load per-bodypart JSON
    bp_json = sorted(run_dir.glob("*_per_bodypart.json"))
    bp_metrics = None
    if bp_json:
        with open(bp_json[0]) as f:
            bp_metrics = json.load(f)

    return df, metrics, bp_metrics


def merge_runs(dfs: list[pd.DataFrame], labels: list[str]) -> pd.DataFrame:
    """Merge per-study DataFrames on study_id, suffixing columns with model labels."""
    if not dfs:
        raise ValueError("No DataFrames to merge")

    base = dfs[0][["study_id", "body_part", "ground_truth"]].copy()

    for df, label in zip(dfs, labels):
        safe = label.replace(" ", "_").replace("-", "_").lower()
        subset = df[["study_id", "prediction", "confidence"]].copy()
        subset = subset.rename(columns={
            "prediction": f"pred_{safe}",
            "confidence": f"conf_{safe}",
        })
        base = base.merge(subset, on="study_id", how="outer")

    return base


# ============================================================
# BOOTSTRAP CIs
# ============================================================

def _mcc(tp, fp, tn, fn):
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return 0.0
    return (tp * tn - fp * fn) / denom


def _kappa(tp, fp, tn, fn):
    total = tp + fp + tn + fn
    if total == 0:
        return 0.0
    p_obs = (tp + tn) / total
    p_pos = ((tp + fp) / total) * ((tp + fn) / total)
    p_neg = ((tn + fn) / total) * ((tn + fp) / total)
    p_exp = p_pos + p_neg
    if p_exp >= 1.0:
        return 0.0
    return (p_obs - p_exp) / (1.0 - p_exp)


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray,
                 metric_fn: str = "mcc", n_boot: int = 2000,
                 ci: float = 0.95, seed: int = 42) -> tuple[float, float, float]:
    """Compute bootstrap CI for MCC or kappa.

    Returns (point_estimate, ci_lower, ci_upper).
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    estimates = []

    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        tp = int(np.sum(yt & yp))
        fp = int(np.sum(~yt & yp))
        tn = int(np.sum(~yt & ~yp))
        fn = int(np.sum(yt & ~yp))
        if metric_fn == "mcc":
            estimates.append(_mcc(tp, fp, tn, fn))
        else:
            estimates.append(_kappa(tp, fp, tn, fn))

    estimates = np.array(estimates)
    alpha = (1 - ci) / 2

    # Point estimate on full data
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    if metric_fn == "mcc":
        point = _mcc(tp, fp, tn, fn)
    else:
        point = _kappa(tp, fp, tn, fn)

    return point, float(np.percentile(estimates, 100 * alpha)), float(np.percentile(estimates, 100 * (1 - alpha)))


# ============================================================
# DeLong AUC COMPARISON
# ============================================================

def _compute_midrank(x):
    """Compute midranks for DeLong test."""
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    result = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n - 1 and z[k + 1] == z[k]:
            k += 1
        for m in range(i, k + 1):
            result[j[m]] = (i + k) / 2.0
        i = k + 1
    return result


def _fast_delong(predictions_sorted_transposed, label_1_count):
    """Core DeLong computation. predictions_sorted_transposed shape: (n_models, n_samples)."""
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    aucs = np.zeros(k)
    tx = np.zeros((k, m))
    ty = np.zeros((k, n))

    for r in range(k):
        all_scores = np.concatenate([positive_examples[r], negative_examples[r]])
        ranks = _compute_midrank(all_scores)
        pos_ranks = ranks[:m]
        aucs[r] = (np.sum(pos_ranks) - m * (m + 1) / 2.0) / (m * n)
        tx[r] = pos_ranks - np.arange(1, m + 1)
        neg_ranks = ranks[m:]
        ty[r] = neg_ranks - np.arange(1, n + 1)

    sx = np.cov(tx) if tx.shape[0] > 1 else np.atleast_2d(np.var(tx, axis=1))
    sy = np.cov(ty) if ty.shape[0] > 1 else np.atleast_2d(np.var(ty, axis=1))

    return aucs, sx / m + sy / n


def delong_test(y_true: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray) -> dict:
    """Two-sided DeLong test comparing AUC of two models.

    Returns dict with auc_a, auc_b, z_stat, p_value.
    """
    order = np.argsort(-y_true.astype(int))  # positives first
    y_sorted = y_true[order]
    m = int(np.sum(y_sorted))

    predictions = np.vstack([scores_a[order], scores_b[order]])
    aucs, cov = _fast_delong(predictions, m)

    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        z = 0.0
        p = 1.0
    else:
        z = diff / np.sqrt(var)
        p = 2 * sp_stats.norm.sf(abs(z))

    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]),
            "z_stat": float(z), "p_value": float(p)}


# ============================================================
# McNEMAR'S TEST
# ============================================================

def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """McNemar's test comparing two classifiers on matched samples.

    Uses exact binomial test when discordant pairs < 25, chi-squared otherwise.
    Returns dict with b (A right B wrong), c (A wrong B right), statistic, p_value.
    """
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)

    # b = A correct, B incorrect; c = A incorrect, B correct
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))

    if b + c == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0, "method": "n/a"}

    if b + c < 25:
        # Exact binomial test
        result_bt = sp_stats.binomtest(b, b + c, 0.5)
        p = float(result_bt.pvalue)
        return {"b": b, "c": c, "statistic": float(b), "p_value": p, "method": "exact_binomial"}
    else:
        # Chi-squared with continuity correction
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p = float(sp_stats.chi2.sf(chi2, df=1))
        return {"b": b, "c": c, "statistic": float(chi2), "p_value": p, "method": "chi2_cc"}


# ============================================================
# COCHRAN'S Q TEST (OMNIBUS FOR 3+ CLASSIFIERS)
# ============================================================

def cochrans_q_test(y_true: np.ndarray, pred_dict: dict[str, np.ndarray]) -> dict:
    """Cochran's Q test: omnibus test for whether any classifiers differ.

    This is the matched-classifier analog of chi-squared. Should be run before
    pairwise McNemar's tests when comparing 3+ models. If Q is non-significant,
    pairwise tests are not warranted.

    pred_dict: {model_label: prediction_array}
    Returns dict with Q statistic, df, p_value, and per-model accuracy.
    """
    labels = sorted(pred_dict.keys())
    k = len(labels)
    n = len(y_true)

    if k < 3:
        return {"Q": None, "df": None, "p_value": None,
                "note": "Cochran's Q requires 3+ classifiers"}

    # Build correctness matrix: n_samples x k_classifiers
    correct_matrix = np.zeros((n, k), dtype=int)
    for j, label in enumerate(labels):
        correct_matrix[:, j] = (pred_dict[label] == y_true).astype(int)

    # Row sums (how many classifiers got each sample right)
    row_sums = correct_matrix.sum(axis=1)  # shape (n,)
    # Column sums (how many samples each classifier got right)
    col_sums = correct_matrix.sum(axis=0)  # shape (k,)

    N = row_sums.sum()  # total correct across all classifiers and samples

    # Cochran's Q statistic
    numerator = (k - 1) * (k * np.sum(col_sums ** 2) - N ** 2)
    denominator = k * N - np.sum(row_sums ** 2)

    if denominator == 0:
        return {"Q": 0.0, "df": k - 1, "p_value": 1.0, "models": labels,
                "per_model_accuracy": {l: float(col_sums[i] / n) for i, l in enumerate(labels)}}

    Q = float(numerator / denominator)
    df = k - 1
    p = float(sp_stats.chi2.sf(Q, df))

    return {
        "Q": round(Q, 3),
        "df": df,
        "p_value": round(p, 6),
        "models": labels,
        "per_model_accuracy": {l: round(float(col_sums[i] / n), 4) for i, l in enumerate(labels)},
    }


# ============================================================
# HOLM-BONFERRONI CORRECTION
# ============================================================

def holm_bonferroni(p_values: dict[str, float]) -> dict[str, dict]:
    """Apply Holm-Bonferroni correction to a set of pairwise p-values.

    p_values: {pair_label: raw_p_value}
    Returns: {pair_label: {"raw_p": ..., "adjusted_p": ..., "significant": bool}}
    """
    pairs = sorted(p_values.keys(), key=lambda k: p_values[k])
    m = len(pairs)
    results = {}

    for rank, pair in enumerate(pairs):
        raw_p = p_values[pair]
        adjusted_p = min(raw_p * (m - rank), 1.0)
        # Enforce monotonicity: adjusted p cannot be less than any previously adjusted p
        if rank > 0:
            prev_pair = pairs[rank - 1]
            adjusted_p = max(adjusted_p, results[prev_pair]["adjusted_p"])
        results[pair] = {
            "raw_p": round(raw_p, 6),
            "adjusted_p": round(adjusted_p, 6),
            "significant_005": adjusted_p < 0.05,
        }

    return results


# ============================================================
# CALIBRATION
# ============================================================

def calibration_stats(y_true: np.ndarray, scores: np.ndarray, n_bins: int = 10) -> dict:
    """Compute calibration metrics: Brier score, ECE, bin-level data for reliability diagram."""
    mask = ~np.isnan(scores)
    y_true = y_true[mask]
    scores = scores[mask]

    if len(scores) == 0:
        return {"brier": None, "ece": None, "bins": []}

    brier = float(brier_score_loss(y_true, scores))

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bins = []
    ece = 0.0
    for i in range(n_bins):
        in_bin = (scores > bin_edges[i]) & (scores <= bin_edges[i + 1])
        if i == 0:
            in_bin = (scores >= bin_edges[i]) & (scores <= bin_edges[i + 1])
        count = int(np.sum(in_bin))
        if count == 0:
            bins.append({"bin_center": (bin_edges[i] + bin_edges[i + 1]) / 2,
                         "mean_predicted": None, "fraction_positive": None, "count": 0})
            continue
        mean_pred = float(np.mean(scores[in_bin]))
        frac_pos = float(np.mean(y_true[in_bin]))
        ece += count * abs(frac_pos - mean_pred)
        bins.append({"bin_center": (bin_edges[i] + bin_edges[i + 1]) / 2,
                     "mean_predicted": round(mean_pred, 4),
                     "fraction_positive": round(frac_pos, 4),
                     "count": count})

    ece = ece / len(scores)

    return {"brier": round(brier, 4), "ece": round(ece, 4), "bins": bins}


# ============================================================
# PLOTS
# ============================================================

def plot_roc_curves(y_true: np.ndarray, score_dict: dict[str, np.ndarray],
                    auc_dict: dict[str, float], output_path: Path):
    """Plot overlaid ROC curves for all models."""
    if not HAS_MATPLOTLIB:
        print("[!] matplotlib not available, skipping ROC plot")
        return

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]

    for i, (label, scores) in enumerate(score_dict.items()):
        mask = ~np.isnan(scores)
        if mask.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true[mask], scores[mask])
        auc_val = auc_dict.get(label, 0)
        ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                label=f"{label} (AUC = {auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — AXIS Cross-Model Comparison", fontsize=13)
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] ROC curves saved to: {output_path}")


def plot_calibration(y_true: np.ndarray, score_dict: dict[str, np.ndarray],
                     cal_dict: dict[str, dict], output_path: Path):
    """Plot reliability diagrams for all models."""
    if not HAS_MATPLOTLIB:
        print("[!] matplotlib not available, skipping calibration plot")
        return

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]

    for i, (label, cal) in enumerate(cal_dict.items()):
        bins = cal.get("bins", [])
        xs = [b["mean_predicted"] for b in bins if b["mean_predicted"] is not None]
        ys = [b["fraction_positive"] for b in bins if b["fraction_positive"] is not None]
        if xs:
            brier = cal.get("brier", 0)
            ece = cal.get("ece", 0)
            ax.plot(xs, ys, "o-", color=colors[i % len(colors)], lw=2, markersize=6,
                    label=f"{label} (Brier={brier:.3f}, ECE={ece:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("Mean Predicted Confidence", fontsize=12)
    ax.set_ylabel("Fraction Truly Abnormal", fontsize=12)
    ax.set_title("Calibration — Reliability Diagram", fontsize=13)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] Calibration plot saved to: {output_path}")


# ============================================================
# TABLE GENERATION
# ============================================================

def build_comparison_table(all_metrics: dict[str, dict],
                           bootstrap_results: dict[str, dict]) -> pd.DataFrame:
    """Build the main comparison table: one row per metric, one column per model."""
    metric_keys = [
        ("accuracy", "Accuracy", True),
        ("sensitivity", "Sensitivity", True),
        ("specificity", "Specificity", True),
        ("precision", "Precision (PPV)", True),
        ("npv", "NPV", True),
        ("f1", "F1 Score", True),
        ("balanced_accuracy", "Balanced Accuracy", False),
        ("mcc", "MCC", False),
        ("kappa", "Cohen's kappa", False),
    ]

    rows = []
    for key, display, has_wilson in metric_keys:
        row = {"Metric": display}
        for label, metrics in all_metrics.items():
            val = metrics.get(key, 0)
            ci_key = f"ci_{key}_95"

            if key in ("mcc", "kappa") and label in bootstrap_results:
                bs = bootstrap_results[label].get(key)
                if bs:
                    row[label] = f"{val:.3f} [{bs[1]:.3f}, {bs[2]:.3f}]"
                else:
                    row[label] = f"{val:.3f}"
            elif has_wilson and ci_key in metrics:
                ci = metrics[ci_key]
                row[label] = f"{val:.1%} [{ci[0]:.1%}, {ci[1]:.1%}]"
            elif has_wilson:
                row[label] = f"{val:.1%}"
            else:
                row[label] = f"{val:.3f}"

        rows.append(row)

    # Add confusion matrix row
    for cm_label, cm_keys in [("Confusion Matrix", ["tp", "fp", "fn", "tn"])]:
        row = {"Metric": cm_label}
        for label, metrics in all_metrics.items():
            tp, fp, fn, tn = [metrics.get(k, 0) for k in cm_keys]
            row[label] = f"TP={tp} FP={fp} FN={fn} TN={tn}"
        rows.append(row)

    return pd.DataFrame(rows)


def build_anatomy_table(all_bp: dict[str, dict[str, dict]]) -> pd.DataFrame:
    """Build per-anatomy comparison table.

    all_bp: {model_label: {body_part: metrics_dict}}
    """
    parts = sorted(set(bp for bpd in all_bp.values() for bp in bpd.keys()))
    rows = []

    for part in parts:
        for metric_key, display in [("mcc", "MCC"), ("f1", "F1"), ("sensitivity", "Sens"),
                                     ("specificity", "Spec"), ("accuracy", "Acc")]:
            row = {"Body Part": part, "Metric": display}
            for label, bpd in all_bp.items():
                val = bpd.get(part, {}).get(metric_key, None)
                if val is not None:
                    if metric_key in ("mcc",):
                        row[label] = f"{val:.3f}"
                    else:
                        row[label] = f"{val:.1%}"
                else:
                    row[label] = "—"
            rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="AXIS — Cross-run comparison for publication",
        epilog="Examples:\n"
               "  python3 compare_runs.py --runs ../results/run_a ../results/run_b --labels 'Gemma 3' 'MedGemma'\n"
               "  python3 compare_runs.py --runs ../results/run_a ../results/run_b ../results/run_c --labels 'Gemma 3' 'MedGemma' 'AXIS-MURA-v1' --output ../results/comparison\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories to compare (2-4)")
    parser.add_argument("--labels", nargs="+", required=True, help="Display labels for each run (same order)")
    parser.add_argument("--output", "-o", default=None, help="Output directory (default: ../results/comparison_<timestamp>)")
    parser.add_argument("--n-bootstrap", type=int, default=2000, help="Bootstrap iterations (default: 2000)")
    parser.add_argument("--no-plots", action="store_true", help="Skip ROC and calibration plots")
    parser.add_argument("--latex", action="store_true", help="Also output LaTeX table")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for bootstrap")
    args = parser.parse_args()

    if len(args.runs) != len(args.labels):
        parser.error("Number of --runs must match number of --labels")
    if len(args.runs) < 2:
        parser.error("Need at least 2 runs to compare")

    # Output directory
    if args.output:
        out_dir = Path(args.output)
    else:
        _project_root = Path(__file__).resolve().parent.parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = _project_root / "results" / f"comparison_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[i] Output directory: {out_dir}\n")

    # Load all runs
    dfs = []
    all_metrics = {}
    all_bp_metrics = {}
    for run_path, label in zip(args.runs, args.labels):
        print(f"[↓] Loading: {label} ({run_path})")
        df, metrics, bp = load_run(run_path)
        dfs.append(df)
        all_metrics[label] = metrics
        if bp:
            all_bp_metrics[label] = bp

    # Merge on study_id
    merged = merge_runs(dfs, args.labels)
    n_studies = len(merged)
    print(f"\n[i] Merged {n_studies} studies across {len(args.labels)} models")

    y_true = merged["ground_truth"].values.astype(bool)

    # Prepare prediction and confidence arrays per model
    pred_arrays = {}
    conf_arrays = {}
    for label in args.labels:
        safe = label.replace(" ", "_").replace("-", "_").lower()
        pred_col = f"pred_{safe}"
        conf_col = f"conf_{safe}"
        pred_arrays[label] = merged[pred_col].values.astype(bool)
        # Confidence may have NaN
        conf_arrays[label] = merged[conf_col].values.astype(float)

    # ---- Bootstrap CIs for MCC and kappa ----
    print(f"\n[→] Computing bootstrap CIs ({args.n_bootstrap} iterations)...")
    bootstrap_results = {}
    for label in args.labels:
        mcc_point, mcc_lo, mcc_hi = bootstrap_ci(y_true, pred_arrays[label], "mcc", args.n_bootstrap, seed=args.seed)
        kap_point, kap_lo, kap_hi = bootstrap_ci(y_true, pred_arrays[label], "kappa", args.n_bootstrap, seed=args.seed)
        bootstrap_results[label] = {
            "mcc": (mcc_point, mcc_lo, mcc_hi),
            "kappa": (kap_point, kap_lo, kap_hi),
        }
        print(f"  {label}: MCC = {mcc_point:.3f} [{mcc_lo:.3f}, {mcc_hi:.3f}], "
              f"kappa = {kap_point:.3f} [{kap_lo:.3f}, {kap_hi:.3f}]")

    # ---- AUC ----
    auc_results = {}
    if HAS_SKLEARN:
        print(f"\n[→] Computing AUC...")
        for label in args.labels:
            scores = conf_arrays[label]
            mask = ~np.isnan(scores)
            if mask.sum() > 0:
                try:
                    auc_val = roc_auc_score(y_true[mask], scores[mask])
                    auc_results[label] = float(auc_val)
                    print(f"  {label}: AUC = {auc_val:.3f}")
                except ValueError as e:
                    print(f"  {label}: AUC computation failed ({e})")
    else:
        print("[!] sklearn not available, skipping AUC")

    # ---- DeLong pairwise ----
    delong_results = {}
    if HAS_SKLEARN and len(args.labels) >= 2:
        print(f"\n[→] DeLong pairwise AUC comparisons...")
        for i in range(len(args.labels)):
            for j in range(i + 1, len(args.labels)):
                la, lb = args.labels[i], args.labels[j]
                sa, sb = conf_arrays[la], conf_arrays[lb]
                mask = ~(np.isnan(sa) | np.isnan(sb))
                if mask.sum() > 0:
                    try:
                        result = delong_test(y_true[mask], sa[mask], sb[mask])
                        pair_key = f"{la} vs {lb}"
                        delong_results[pair_key] = result
                        sig = "***" if result["p_value"] < 0.001 else "**" if result["p_value"] < 0.01 else "*" if result["p_value"] < 0.05 else "ns"
                        print(f"  {pair_key}: ΔAUC = {result['auc_a'] - result['auc_b']:.3f}, "
                              f"z = {result['z_stat']:.2f}, p = {result['p_value']:.4f} {sig}")
                    except Exception as e:
                        print(f"  {la} vs {lb}: DeLong failed ({e})")

    # ---- Cochran's Q omnibus test (3+ models) ----
    cochran_result = {}
    if len(args.labels) >= 3:
        print(f"\n[→] Cochran's Q omnibus test ({len(args.labels)} classifiers)...")
        cochran_result = cochrans_q_test(y_true, pred_arrays)
        sig = "***" if cochran_result["p_value"] < 0.001 else "**" if cochran_result["p_value"] < 0.01 else "*" if cochran_result["p_value"] < 0.05 else "ns"
        print(f"  Q = {cochran_result['Q']:.3f}, df = {cochran_result['df']}, "
              f"p = {cochran_result['p_value']:.6f} {sig}")
        if cochran_result["p_value"] >= 0.05:
            print(f"  [!] Cochran's Q non-significant. Pairwise tests may not be warranted.")
        else:
            print(f"  [✓] Significant. Proceeding to pairwise tests with Holm-Bonferroni correction.")

    # ---- McNemar's pairwise ----
    mcnemar_results = {}
    if len(args.labels) >= 2:
        print(f"\n[→] McNemar's pairwise tests...")
        for i in range(len(args.labels)):
            for j in range(i + 1, len(args.labels)):
                la, lb = args.labels[i], args.labels[j]
                result = mcnemar_test(y_true, pred_arrays[la], pred_arrays[lb])
                pair_key = f"{la} vs {lb}"
                mcnemar_results[pair_key] = result
                sig = "***" if result["p_value"] < 0.001 else "**" if result["p_value"] < 0.01 else "*" if result["p_value"] < 0.05 else "ns"
                print(f"  {pair_key}: b={result['b']}, c={result['c']}, "
                      f"p = {result['p_value']:.4f} ({result['method']}) {sig}")

    # ---- Holm-Bonferroni correction (when 3+ models) ----
    mcnemar_corrected = {}
    delong_corrected = {}
    if len(args.labels) >= 3:
        if mcnemar_results:
            print(f"\n[→] Holm-Bonferroni correction (McNemar's, {len(mcnemar_results)} pairs)...")
            raw_ps = {k: v["p_value"] for k, v in mcnemar_results.items()}
            mcnemar_corrected = holm_bonferroni(raw_ps)
            for pair, corr in mcnemar_corrected.items():
                sig = "*" if corr["significant_005"] else "ns"
                print(f"  {pair}: raw p = {corr['raw_p']:.6f}, adjusted p = {corr['adjusted_p']:.6f} {sig}")

        if delong_results:
            print(f"\n[→] Holm-Bonferroni correction (DeLong, {len(delong_results)} pairs)...")
            raw_ps = {k: v["p_value"] for k, v in delong_results.items()}
            delong_corrected = holm_bonferroni(raw_ps)
            for pair, corr in delong_corrected.items():
                sig = "*" if corr["significant_005"] else "ns"
                print(f"  {pair}: raw p = {corr['raw_p']:.6f}, adjusted p = {corr['adjusted_p']:.6f} {sig}")
    elif len(args.labels) == 2:
        print(f"\n[i] 2 models: no multiple comparison correction needed.")

    # ---- Calibration ----
    cal_results = {}
    if HAS_SKLEARN:
        print(f"\n[→] Computing calibration metrics...")
        for label in args.labels:
            scores = conf_arrays[label]
            cal = calibration_stats(y_true, scores)
            cal_results[label] = cal
            if cal["brier"] is not None:
                print(f"  {label}: Brier = {cal['brier']:.4f}, ECE = {cal['ece']:.4f}")

    # ---- Build tables ----
    print(f"\n[→] Building comparison tables...")

    # Main comparison table
    comp_table = build_comparison_table(all_metrics, bootstrap_results)
    comp_csv = out_dir / "comparison_table.csv"
    comp_table.to_csv(comp_csv, index=False)
    print(f"[✓] Comparison table: {comp_csv}")

    # Add AUC row if available
    if auc_results:
        auc_row = {"Metric": "AUC"}
        for label in args.labels:
            auc_row[label] = f"{auc_results.get(label, 0):.3f}"
        auc_df = pd.DataFrame([auc_row])
        # Insert AUC after F1 in the table
        comp_table = pd.concat([comp_table.iloc[:6], auc_df, comp_table.iloc[6:]], ignore_index=True)
        comp_table.to_csv(comp_csv, index=False)

    # Per-anatomy table
    if all_bp_metrics:
        anat_table = build_anatomy_table(all_bp_metrics)
        anat_csv = out_dir / "comparison_per_anatomy.csv"
        anat_table.to_csv(anat_csv, index=False)
        print(f"[✓] Per-anatomy table: {anat_csv}")

    # LaTeX output
    if args.latex:
        latex_path = out_dir / "comparison_table.tex"
        with open(latex_path, "w") as f:
            f.write(comp_table.to_latex(index=False, escape=True))
        print(f"[✓] LaTeX table: {latex_path}")
        if all_bp_metrics:
            anat_latex = out_dir / "comparison_per_anatomy.tex"
            with open(anat_latex, "w") as f:
                f.write(anat_table.to_latex(index=False, escape=True))
            print(f"[✓] Per-anatomy LaTeX: {anat_latex}")

    # ---- Save full statistical results as JSON ----
    stats_output = {
        "timestamp": datetime.now().isoformat(),
        "n_studies": n_studies,
        "models": args.labels,
        "run_directories": args.runs,
        "bootstrap_n": args.n_bootstrap,
        "seed": args.seed,
        "bootstrap_ci": {label: {k: {"point": v[0], "ci_lower": v[1], "ci_upper": v[2]}
                                  for k, v in bsr.items()}
                         for label, bsr in bootstrap_results.items()},
        "auc": auc_results,
        "delong": delong_results,
        "delong_holm_bonferroni": delong_corrected if delong_corrected else None,
        "cochrans_q": cochran_result if cochran_result else None,
        "mcnemar": mcnemar_results,
        "mcnemar_holm_bonferroni": mcnemar_corrected if mcnemar_corrected else None,
        "calibration": {label: {"brier": cal.get("brier"), "ece": cal.get("ece")}
                        for label, cal in cal_results.items()},
    }
    stats_path = out_dir / "statistical_tests.json"
    with open(stats_path, "w") as f:
        json.dump(stats_output, f, indent=2)
    print(f"[✓] Statistical tests: {stats_path}")

    # ---- Merged data ----
    merged_csv = out_dir / "merged_studies.csv"
    merged.to_csv(merged_csv, index=False)
    print(f"[✓] Merged study data: {merged_csv}")

    # ---- Plots ----
    if not args.no_plots:
        if HAS_MATPLOTLIB and HAS_SKLEARN and auc_results:
            plot_roc_curves(y_true, conf_arrays, auc_results, out_dir / "roc_curves.png")
        if HAS_MATPLOTLIB and HAS_SKLEARN and cal_results:
            plot_calibration(y_true, conf_arrays, cal_results, out_dir / "calibration.png")

    # ---- Print summary ----
    print(f"\n{'='*70}")
    print(f"  AXIS CROSS-MODEL COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"  Studies: {n_studies}")
    print(f"  Models:  {', '.join(args.labels)}")
    print(f"")
    print(comp_table.to_string(index=False))
    print(f"{'='*70}")

    if cochran_result and cochran_result.get("Q") is not None:
        sig = "***" if cochran_result["p_value"] < 0.001 else "**" if cochran_result["p_value"] < 0.01 else "*" if cochran_result["p_value"] < 0.05 else "ns"
        print(f"\n  Cochran's Q omnibus: Q = {cochran_result['Q']:.3f}, df = {cochran_result['df']}, p = {cochran_result['p_value']:.6f} {sig}")

    if mcnemar_results:
        corrected = len(mcnemar_corrected) > 0
        label_suffix = " (Holm-Bonferroni adjusted)" if corrected else ""
        print(f"\n  Pairwise McNemar's tests{label_suffix}:")
        for pair, res in mcnemar_results.items():
            if corrected and pair in mcnemar_corrected:
                p_display = mcnemar_corrected[pair]["adjusted_p"]
                sig = "*" if mcnemar_corrected[pair]["significant_005"] else "ns"
                print(f"    {pair}: raw p = {res['p_value']:.4f}, adjusted p = {p_display:.4f} {sig}")
            else:
                sig = "***" if res["p_value"] < 0.001 else "**" if res["p_value"] < 0.01 else "*" if res["p_value"] < 0.05 else "ns"
                print(f"    {pair}: p = {res['p_value']:.4f} {sig}")

    if delong_results:
        corrected = len(delong_corrected) > 0
        label_suffix = " (Holm-Bonferroni adjusted)" if corrected else ""
        print(f"\n  Pairwise DeLong AUC tests{label_suffix}:")
        for pair, res in delong_results.items():
            if corrected and pair in delong_corrected:
                p_display = delong_corrected[pair]["adjusted_p"]
                sig = "*" if delong_corrected[pair]["significant_005"] else "ns"
                print(f"    {pair}: ΔAUC = {res['auc_a'] - res['auc_b']:.3f}, raw p = {res['p_value']:.4f}, adjusted p = {p_display:.4f} {sig}")
            else:
                sig = "***" if res["p_value"] < 0.001 else "**" if res["p_value"] < 0.01 else "*" if res["p_value"] < 0.05 else "ns"
                print(f"    {pair}: ΔAUC = {res['auc_a'] - res['auc_b']:.3f}, p = {res['p_value']:.4f} {sig}")

    print(f"\n[✓] All outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
