"""Demo tool implementations for the Evaluator agent — synthetic inference.

Replaces run_inference with a fast synthetic prediction generator (~50 ms/image)
while reusing validate_results and compute_metrics from the real implementation.

No inference server, no model, no GPU. Output format matches the real
run_inference exactly so the rest of the pipeline (validation, metrics,
analyst) is exercised end-to-end.
"""
from __future__ import annotations

import csv
import hashlib
import random
import time
from pathlib import Path

from cockpit import events as cockpit_events
from tools.evaluator_impl import _read_manifest  # noqa: F401  (validate_results, compute_metrics imported on demand)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Generic findings text — kept deliberately abstract so the synthetic
# output isn't tied to any specific anatomy or task. The agent layer
# treats findings as opaque strings; this is just so the cockpit feed
# shows something readable.
_FINDINGS_ABNORMAL = [
    "Abnormality detected with moderate confidence.",
    "Findings consistent with positive class.",
    "Suspicious feature flagged for review.",
]
_FINDING_NORMAL = "No abnormalities identified."


def run_inference_demo(dataset_path: str, manifest_path: str, output_path: str) -> dict:
    """Generate synthetic predictions without calling any model.

    Output matches run_inference: ~85% accuracy, bimodal confidence
    (high when correct, low-mid when wrong), seeded RNG for reproducibility.
    """
    manifest = _read_manifest(manifest_path)
    total_images = len(manifest)

    # Seed RNG deterministically from manifest path so identical inputs
    # produce identical demo runs.
    seed = int(hashlib.md5(manifest_path.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    results = []
    errors: list[str] = []

    for i, row in enumerate(manifest):
        image_path = row.get("image_path")
        if not image_path:
            errors.append(f"Row {i}: missing image_path")
            continue
        label = int(row["label"])

        cockpit_events.emit("inference_progress",
                            current=i + 1, total=total_images,
                            image=image_path)

        # Synthetic prediction: ~85% match ground truth, bimodal confidence
        if rng.random() < 0.85:
            prediction = label
            confidence = round(rng.uniform(0.80, 0.99), 2)
        else:
            prediction = 1 - label
            confidence = round(rng.uniform(0.30, 0.65), 2)

        if prediction == 1:
            findings = rng.choice(_FINDINGS_ABNORMAL)
        else:
            findings = _FINDING_NORMAL

        # Simulate brief processing time (~50ms)
        t0 = time.monotonic()
        time.sleep(0.05)
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        cockpit_events.emit("inference_call", image=image_path,
                            prediction=prediction,
                            confidence=confidence,
                            elapsed_ms=elapsed_ms,
                            server="demo://synthetic")

        results.append({
            "image_path": image_path,
            "ground_truth": label,
            "prediction": prediction,
            "confidence": confidence,
            "findings": findings,
        })

    out = Path(output_path)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "ground_truth", "prediction", "confidence", "findings"])
        writer.writeheader()
        writer.writerows(results)

    return {
        "status": "success",
        "predictions_written": len(results),
        "errors": errors,
        "error_count": len(errors),
        "output_path": str(out),
    }


def patch_tool_functions():
    """Patch the evaluator tool dispatch table to use demo inference.

    Call this before running the pipeline. Only run_inference is replaced;
    validate_results and compute_metrics use the real implementations.
    """
    from tools import evaluator_impl
    evaluator_impl.TOOL_FUNCTIONS["run_inference"] = run_inference_demo
