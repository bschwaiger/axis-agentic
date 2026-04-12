"""Tool implementations for the Evaluator agent."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import httpx

from cockpit import events as cockpit_events

INFERENCE_URL = "http://localhost:8321/predict"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------
# run_inference
# ------------------------------------------------------------------

def run_inference(dataset_path: str, manifest_path: str, output_path: str) -> dict:
    """Call local inference server for each study in the manifest, write predictions CSV."""
    manifest = _read_manifest(manifest_path)
    dataset_dir = Path(dataset_path)
    if not dataset_dir.is_absolute():
        dataset_dir = PROJECT_ROOT / dataset_dir

    results = []
    errors = []

    from tqdm import tqdm

    # Persistent connection to inference server — avoids TCP setup per study
    inf_client = httpx.Client(timeout=60.0)

    total_studies = len(manifest)
    for i, row in enumerate(tqdm(manifest, desc="    Inference", unit="study",
                                  bar_format="    {l_bar}{bar:30}{r_bar}")):
        study_path = row["study_path"]
        # Emit progress to cockpit every study
        cockpit_events.emit("inference_progress",
                            current=i + 1, total=total_studies,
                            study=study_path)
        label = int(row["label"])

        # Find images in the study directory
        study_dir = dataset_dir / study_path
        if not study_dir.exists():
            errors.append(f"Study dir not found: {study_dir}")
            continue

        images = sorted(
            p for p in study_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if not images:
            errors.append(f"No images in: {study_dir}")
            continue

        # Use first image (MURA convention: one image per study for most)
        image_path = str(images[0])
        # Make path relative to project root for the server
        try:
            rel_path = str(Path(image_path).relative_to(PROJECT_ROOT))
        except ValueError:
            rel_path = image_path

        try:
            import time as _time
            t0 = _time.monotonic()
            resp = inf_client.post(
                INFERENCE_URL,
                json={"image_path": rel_path},
            )
            resp.raise_for_status()
            pred = resp.json()
            elapsed_ms = int((_time.monotonic() - t0) * 1000)
            cockpit_events.emit("inference_call", study=study_path,
                                prediction=pred.get("prediction"),
                                confidence=pred.get("confidence"),
                                elapsed_ms=elapsed_ms,
                                server=INFERENCE_URL)
            results.append({
                "study_path": study_path,
                "ground_truth": label,
                "prediction": pred["prediction"],
                "confidence": pred["confidence"],
                "findings": pred.get("findings", ""),
            })
        except Exception as e:
            errors.append(f"Inference failed for {study_path}: {e}")
            results.append({
                "study_path": study_path,
                "ground_truth": label,
                "prediction": None,
                "confidence": None,
                "findings": f"ERROR: {e}",
            })

    inf_client.close()

    # Write predictions CSV
    out = Path(output_path)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["study_path", "ground_truth", "prediction", "confidence", "findings"])
        writer.writeheader()
        writer.writerows(results)

    return {
        "status": "success",
        "predictions_written": len(results),
        "errors": errors[:10],  # cap error list
        "error_count": len(errors),
        "output_path": str(out),
    }


# ------------------------------------------------------------------
# validate_results
# ------------------------------------------------------------------

def validate_results(predictions_path: str, manifest_path: str) -> dict:
    """Validate predictions against manifest."""
    manifest = _read_manifest(manifest_path)
    preds = _read_predictions(predictions_path)

    issues = []

    # Check count match
    if len(preds) != len(manifest):
        issues.append(f"Count mismatch: {len(preds)} predictions vs {len(manifest)} manifest entries")

    # Check for null predictions
    null_preds = [p for p in preds if p["prediction"] is None or p["prediction"] == ""]
    if null_preds:
        issues.append(f"{len(null_preds)} null/empty predictions")

    # Check prediction distribution
    valid_preds = [int(p["prediction"]) for p in preds if p["prediction"] is not None and p["prediction"] != ""]
    if valid_preds:
        pos = sum(valid_preds)
        neg = len(valid_preds) - pos
        if pos == 0 or neg == 0:
            issues.append(f"All predictions are {'positive' if pos > 0 else 'negative'} — implausible")
        pos_rate = pos / len(valid_preds)
        if pos_rate < 0.05 or pos_rate > 0.95:
            issues.append(f"Extreme class imbalance in predictions: {pos_rate:.1%} positive")

    passed = len(issues) == 0
    return {
        "status": "pass" if passed else "fail",
        "total_predictions": len(preds),
        "total_manifest": len(manifest),
        "null_predictions": len(null_preds),
        "issues": issues,
    }


# ------------------------------------------------------------------
# compute_metrics
# ------------------------------------------------------------------

def compute_metrics(predictions_path: str, manifest_path: str, output_path: str) -> dict:
    """Compute classification metrics and write to JSON."""
    preds = _read_predictions(predictions_path)

    tp = fp = tn = fn = 0
    parse_failures = 0
    confidences = []

    for row in preds:
        gt = int(row["ground_truth"])
        pred_raw = row["prediction"]
        if pred_raw is None or pred_raw == "":
            parse_failures += 1
            continue
        pred = int(pred_raw)
        conf = row.get("confidence")
        if conf is not None and conf != "":
            confidences.append(float(conf))

        if gt == 1 and pred == 1:
            tp += 1
        elif gt == 1 and pred == 0:
            fn += 1
        elif gt == 0 and pred == 1:
            fp += 1
        else:
            tn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    mcc_prod = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc_denom = math.sqrt(mcc_prod) if mcc_prod > 0 else 1
    mcc = (tp * tn - fp * fn) / mcc_denom

    # Wilson CIs
    n_pos, n_neg = tp + fn, tn + fp
    n_pred_pos, n_pred_neg = tp + fp, tn + fn

    metrics = {
        "total_evaluated": total,
        "parse_failures": parse_failures,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "accuracy": round(accuracy, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "precision": round(precision, 4),
        "npv": round(npv, 4),
        "f1": round(f1, 4),
        "mcc": round(mcc, 4),
        "ci_sensitivity_95": [round(x, 4) for x in _wilson_ci(sensitivity, n_pos)],
        "ci_specificity_95": [round(x, 4) for x in _wilson_ci(specificity, n_neg)],
        "ci_precision_95": [round(x, 4) for x in _wilson_ci(precision, n_pred_pos)],
        "ci_npv_95": [round(x, 4) for x in _wilson_ci(npv, n_pred_neg)],
        "ci_accuracy_95": [round(x, 4) for x in _wilson_ci(accuracy, total)],
        "ci_f1_95": [round(x, 4) for x in _wilson_ci(f1, total)],
    }

    out = Path(output_path)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)

    return {
        "status": "success",
        "metrics_path": str(out),
        "summary": {
            "n": total,
            "mcc": metrics["mcc"],
            "sensitivity": metrics["sensitivity"],
            "specificity": metrics["specificity"],
            "f1": metrics["f1"],
        },
    }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))


def _read_manifest(path: str) -> list[dict]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    with open(p) as f:
        return list(csv.DictReader(f))


def _read_predictions(path: str) -> list[dict]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    with open(p) as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------------
# Dispatch — called by the agent loop
# ------------------------------------------------------------------

TOOL_FUNCTIONS = {
    "run_inference": run_inference,
    "validate_results": validate_results,
    "compute_metrics": compute_metrics,
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
