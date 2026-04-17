"""Demo tool implementations for the Evaluator agent — synthetic inference.

Replaces run_inference with a fast synthetic prediction generator (~50ms/study)
while reusing validate_results and compute_metrics from the real implementation.

No inference server needed. Output format is identical to real predictions.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import time
from pathlib import Path

from cockpit import events as cockpit_events
from tools.evaluator_impl import validate_results, compute_metrics, _read_manifest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Body-part findings for synthetic predictions
_FINDINGS_ABNORMAL = {
    "XR_WRIST": [
        "Fracture line visible in distal radius with mild displacement.",
        "Scaphoid fracture suspected with cortical irregularity.",
        "Degenerative changes in radiocarpal joint with osteophyte formation.",
    ],
    "XR_FINGER": [
        "Fracture of proximal phalanx with angular deformity.",
        "Soft tissue swelling and periosteal reaction in middle phalanx.",
        "Erosive changes at DIP joint consistent with inflammatory arthropathy.",
    ],
    "XR_ELBOW": [
        "Posterior fat pad sign with radial head fracture.",
        "Olecranon fracture with mild displacement.",
        "Degenerative changes with loose body in olecranon fossa.",
    ],
    "XR_FOREARM": [
        "Mid-shaft ulnar fracture with cortical disruption.",
        "Both-bone forearm fracture with dorsal angulation.",
        "Periosteal reaction along radial diaphysis.",
    ],
    "XR_HAND": [
        "Multiple degenerative changes with joint space narrowing and osteophytes.",
        "Metacarpal fracture with mild shortening.",
        "Erosive arthropathy affecting MCP joints bilaterally.",
    ],
    "XR_HUMERUS": [
        "Surgical hardware present with plate and screws along humeral shaft.",
        "Proximal humerus fracture involving greater tuberosity.",
        "Pathologic-appearing lucency in mid-diaphysis.",
    ],
    "XR_SHOULDER": [
        "Severe glenohumeral osteoarthritis with joint space narrowing and sclerosis.",
        "Surgical hardware from prior rotator cuff repair.",
        "Hill-Sachs deformity suggesting prior dislocation.",
    ],
}

_FINDINGS_NORMAL = {
    "XR_WRIST": "The wrist X-ray shows normal bone structure with no fractures or dislocations.",
    "XR_FINGER": "The image shows a normal hand X-ray.",
    "XR_ELBOW": "The elbow joint appears normal with no fractures or effusion.",
    "XR_FOREARM": "The forearm radiograph shows normal bone structure.",
    "XR_HAND": "The hand X-ray shows normal bone structure and no obvious fractures.",
    "XR_HUMERUS": "The humerus appears normal on this X-ray.",
    "XR_SHOULDER": "The shoulder joint appears normal with maintained joint space.",
}


def run_inference_demo(dataset_path: str, manifest_path: str, output_path: str) -> dict:
    """Generate synthetic predictions without calling the inference server.

    Produces realistic output matching the real run_inference format:
    - ~85% accuracy (matches AXIS-MURA-v1 performance range)
    - Bimodal confidence: high for correct, low-medium for incorrect
    - Body-part-aware findings text
    - Seeded RNG for reproducibility per manifest
    """
    manifest = _read_manifest(manifest_path)
    total_studies = len(manifest)

    # Seed RNG deterministically from manifest path for reproducible demos
    seed = int(hashlib.md5(manifest_path.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    results = []
    errors = []

    for i, row in enumerate(manifest):
        study_path = row["study_path"]
        label = int(row["label"])

        # Emit progress (same event as real inference)
        cockpit_events.emit("inference_progress",
                            current=i + 1, total=total_studies,
                            study=study_path)

        # Determine body part from path
        body_part = study_path.split("/")[0] if "/" in study_path else "XR_HAND"

        # Synthetic prediction: ~85% match ground truth
        if rng.random() < 0.85:
            prediction = label
            confidence = round(rng.uniform(0.80, 0.99), 2)
        else:
            prediction = 1 - label
            confidence = round(rng.uniform(0.30, 0.65), 2)

        # Generate findings text
        if prediction == 1:
            abnormal_options = _FINDINGS_ABNORMAL.get(body_part, _FINDINGS_ABNORMAL["XR_HAND"])
            findings = rng.choice(abnormal_options)
        else:
            findings = _FINDINGS_NORMAL.get(body_part, _FINDINGS_NORMAL["XR_HAND"])

        # Simulate brief processing time (~50ms)
        t0 = time.monotonic()
        time.sleep(0.05)
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        cockpit_events.emit("inference_call", study=study_path,
                            prediction=prediction,
                            confidence=confidence,
                            elapsed_ms=elapsed_ms,
                            server="demo://synthetic")

        results.append({
            "study_path": study_path,
            "ground_truth": label,
            "prediction": prediction,
            "confidence": confidence,
            "findings": findings,
        })

    # Write predictions CSV (same format as real)
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
