#!/usr/bin/env python3
"""
AXIS — Automated X-Ray Identification for the Skeleton
Batch evaluation module. Works with MURA (PNGs) or custom DICOM directories.
Runs MSK pathology / fracture detection via MLX (default) or Transformers, computes metrics.

Usage (MURA, ships as PNG):
    python3 batch_eval.py --body-part XR_WRIST --max-studies 50
    python3 batch_eval.py --all --per-part 50
    python3 batch_eval.py --all --per-part 50 --prompt-version 4 --seed 42

Usage (ablation with base Gemma 3):
    python3 batch_eval.py --manifest ../results/<run_dir>/manifest.json \
        --model mlx-community/gemma-3-4b-it-4bit --prompt-version 4

Usage (custom DICOMs with CSV labels):
    python3 batch_eval.py --data-dir /path/to/dicoms --labels-csv labels.csv

Body parts in MURA: XR_ELBOW, XR_FINGER, XR_FOREARM, XR_HAND, XR_HUMERUS, XR_SHOULDER, XR_WRIST
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

# Import from our detector
from axis_detector import (
    query_model,
    find_dicoms,
    find_images,
    set_backend,
    _resolve_model,
    DEFAULT_BACKEND,
)

# ============================================================
# CONFIG
# ============================================================

# Project root is one level up from pipeline/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

MURA_DIR = _PROJECT_ROOT / "data"
MURA_DATA_ROOT = MURA_DIR  # default data root for relative path resolution
RESULTS_DIR = _PROJECT_ROOT / "results"
BODY_PARTS = ["XR_ELBOW", "XR_FINGER", "XR_FOREARM", "XR_HAND", "XR_HUMERUS", "XR_SHOULDER", "XR_WRIST"]

# Approximate seconds per image by backend (for runtime estimates)
_SECS_PER_IMAGE = {"mlx": 4, "transformers": 25}


# ============================================================
# SUPPORTED IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".dcm", ".dicom"}


# ============================================================
# MANIFEST PATH RESOLUTION
# ============================================================

def _get_mura_data_root(args_mura_dir: str | None = None) -> Path:
    """Return the MURA data root directory, from --mura-dir or default."""
    if args_mura_dir:
        return Path(args_mura_dir)
    return MURA_DATA_ROOT


def _resolve_manifest_path(raw_path: str, data_root: Path) -> str:
    """Resolve a single image or study path from a manifest entry.

    Resolution modes:
      1. Absolute path that exists on disk → use as-is.
      2. Absolute path that does NOT exist → strip everything up to and
         including 'mura/' and resolve against data_root.
      3. Relative path → resolve against data_root.

    Returns the resolved path string and logs the mode used on first call.
    """
    p = Path(raw_path)

    # Mode 1: absolute and exists
    if p.is_absolute() and p.exists():
        return str(p), "absolute"

    # Mode 3: relative path
    if not p.is_absolute():
        resolved = data_root / raw_path
        return str(resolved), "relative"

    # Mode 2: absolute but missing — try stripping prefix up to 'mura/'
    raw = raw_path
    marker = "/mura/"
    idx = raw.find(marker)
    if idx != -1:
        suffix = raw[idx + len(marker):]  # e.g. valid/XR_WRIST/patient.../image.png
        resolved = data_root / suffix
        return str(resolved), "migrated"

    # Fallback: return as-is (will fail validation)
    return raw_path, "unresolved"


def _resolve_manifest_paths(studies: list[dict], data_root: Path) -> list[dict]:
    """Resolve all paths in a loaded manifest against data_root.

    Modifies studies in place and returns them. Logs which resolution mode
    was used.
    """
    mode_counts: dict[str, int] = {}

    for study in studies:
        # Resolve study path
        resolved_path, mode = _resolve_manifest_path(study["path"], data_root)
        study["path"] = resolved_path
        mode_counts[mode] = mode_counts.get(mode, 0) + 1

        # Resolve each image path
        resolved_images = []
        for img in study["images"]:
            resolved_img, img_mode = _resolve_manifest_path(img, data_root)
            resolved_images.append(resolved_img)
            mode_counts[img_mode] = mode_counts.get(img_mode, 0) + 1
        study["images"] = resolved_images

    # Log resolution summary
    mode_labels = {
        "absolute": "absolute (existing)",
        "relative": "relative (resolved against data root)",
        "migrated": "absolute→migrated (stripped prefix, resolved against data root)",
        "unresolved": "unresolved (path not found)",
    }
    parts = [f"{count} {mode_labels.get(m, m)}" for m, count in sorted(mode_counts.items())]
    print(f"[i] Manifest path resolution: {', '.join(parts)}")

    return studies


def _validate_manifest_paths(studies: list[dict], sample_n: int = 5) -> None:
    """Sample up to sample_n image paths from the manifest and assert they exist.

    Must be called after path resolution and before any inference starts.
    Raises SystemExit with a clear message if any path is missing.
    """
    all_images = [img for s in studies for img in s["images"]]
    if not all_images:
        print("[!] Manifest contains no image paths.")
        sys.exit(1)

    sample = all_images[:sample_n] if len(all_images) <= sample_n else random.sample(all_images, sample_n)

    missing = [p for p in sample if not Path(p).exists()]
    if missing:
        print(f"\n[!] MANIFEST PATH VALIDATION FAILED")
        print(f"    Checked {len(sample)} of {len(all_images)} image paths; "
              f"{len(missing)} do not exist on disk:")
        for p in missing:
            print(f"      ✗ {p}")
        print(f"\n    This usually means the project was relocated and the manifest")
        print(f"    still contains stale absolute paths. For example, if you moved")
        print(f"    from ~/axis/ to ~/Projects/axis/, paths will no longer resolve.")
        print(f"\n    Data root used for resolution: {_get_mura_data_root()}")
        print(f"    Tip: re-generate the manifest or pass --mura-dir to point to")
        print(f"    the correct MURA data root.")
        sys.exit(1)

    print(f"[✓] Manifest path validation passed ({len(sample)} sampled paths exist)")


def _to_relative_path(abs_path: str, data_root: Path) -> str:
    """Convert an absolute path to a path relative to the MURA data root.

    If the path is already relative or cannot be made relative to data_root,
    returns it unchanged.
    """
    try:
        return str(Path(abs_path).relative_to(data_root))
    except ValueError:
        return abs_path


# ============================================================
# SUPPORTED IMAGE EXTENSIONS
# ============================================================

def _is_dicom_by_magic(filepath: Path) -> bool:
    try:
        with open(filepath, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except Exception:
        return False


def _find_images_in_dir(directory: Path) -> list[str]:
    results = []
    for p in sorted(directory.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            results.append(str(p))
        elif p.suffix == "" and _is_dicom_by_magic(p):
            results.append(str(p))
    return results


# ============================================================
# MURA DATA LOADER
# ============================================================

def discover_mura_studies(mura_dir: Path, split: str = "valid", body_part: str = None) -> list[dict]:
    split_dir = mura_dir / split
    if not split_dir.exists():
        alt = list(mura_dir.glob("**/valid"))
        if alt:
            split_dir = alt[0]
        else:
            print(f"[!] Cannot find {split_dir}")
            print(f"    Download MURA from: https://aimi.stanford.edu/datasets/mura-msk-xrays")
            sys.exit(1)

    parts = [body_part] if body_part else BODY_PARTS
    studies = []

    for bp in parts:
        bp_dir = split_dir / bp
        if not bp_dir.exists():
            print(f"[!] Body part directory not found: {bp_dir}")
            continue

        for study_dir in sorted(bp_dir.glob("patient*/study*")):
            label_str = study_dir.name.split("_")[-1]
            is_abnormal = label_str == "positive"
            images = _find_images_in_dir(study_dir)
            if images:
                studies.append({
                    "path": str(study_dir),
                    "label": is_abnormal,
                    "body_part": bp,
                    "images": images,
                    "patient": study_dir.parent.name,
                })

    return studies


# ============================================================
# CUSTOM DICOM/PNG LOADER
# ============================================================

def discover_custom_studies(data_dir: Path, labels_csv: Path = None) -> list[dict]:
    POSITIVE_LABELS = {"1", "positive", "abnormal", "yes", "fracture", "true"}
    NEGATIVE_LABELS = {"0", "negative", "normal", "no", "false"}

    label_map = {}
    if labels_csv and labels_csv.exists():
        with open(labels_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row.get("filename") or row.get("file") or row.get("image") or row.get("path") or ""
                label_raw = row.get("label") or row.get("fracture") or row.get("abnormal") or row.get("ground_truth") or ""
                fname = fname.strip()
                label_raw = label_raw.strip().lower()
                if fname and label_raw in POSITIVE_LABELS:
                    label_map[fname] = True
                elif fname and label_raw in NEGATIVE_LABELS:
                    label_map[fname] = False
        print(f"[i] Loaded {len(label_map)} labels from {labels_csv}")
    elif labels_csv:
        print(f"[!] Labels CSV not found: {labels_csv}")
        print(f"    Proceeding without labels (inference only, no metrics).")

    data_dir = Path(data_dir)
    all_images = []

    flat_images = _find_images_in_dir(data_dir)
    all_images.extend(flat_images)

    for subdir in sorted(data_dir.iterdir()):
        if subdir.is_dir():
            nested = _find_images_in_dir(subdir)
            all_images.extend(nested)
            for subsubdir in sorted(subdir.iterdir()):
                if subsubdir.is_dir():
                    all_images.extend(_find_images_in_dir(subsubdir))

    dicoms = find_dicoms(str(data_dir))
    all_images = list(dict.fromkeys(all_images + dicoms))

    if not all_images:
        print(f"[!] No images found in {data_dir}")
        sys.exit(1)

    studies = []
    for img_path in all_images:
        fname = Path(img_path).name
        label = label_map.get(fname)
        studies.append({
            "path": str(Path(img_path).parent),
            "label": label,
            "body_part": "CUSTOM",
            "images": [img_path],
            "patient": fname,
        })

    return studies


# ============================================================
# METRICS
# ============================================================

def _wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))


def compute_metrics(results: list[dict]) -> dict:
    """Compute full classification metrics from results list."""
    tp = fp = tn = fn = 0
    parse_failures = 0
    no_label = 0

    for r in results:
        gt = r["ground_truth"]
        pred = r.get("prediction")
        if pred is None:
            pred = r.get("fracture")

        if gt is None:
            no_label += 1
            continue
        if pred is None:
            parse_failures += 1
            continue

        if gt and pred:
            tp += 1
        elif gt and not pred:
            fn += 1
        elif not gt and pred:
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
    balanced_accuracy = (sensitivity + specificity) / 2

    # Matthews Correlation Coefficient
    mcc_prod = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc_denom = math.sqrt(mcc_prod) if mcc_prod > 0 else 1
    mcc = (tp * tn - fp * fn) / mcc_denom

    # Cohen's kappa
    p_observed = accuracy
    p_pos = ((tp + fp) / total) * ((tp + fn) / total) if total > 0 else 0
    p_neg = ((tn + fn) / total) * ((tn + fp) / total) if total > 0 else 0
    p_expected = p_pos + p_neg
    kappa = (p_observed - p_expected) / (1 - p_expected) if (1 - p_expected) > 0 else 0

    # 95% Wilson confidence intervals
    n_pos = tp + fn
    n_neg = tn + fp
    n_pred_pos = tp + fp
    n_pred_neg = tn + fn

    ci_sensitivity = _wilson_ci(sensitivity, n_pos)
    ci_specificity = _wilson_ci(specificity, n_neg)
    ci_precision = _wilson_ci(precision, n_pred_pos)
    ci_npv = _wilson_ci(npv, n_pred_neg)
    ci_accuracy = _wilson_ci(accuracy, total)
    ci_f1 = _wilson_ci(f1, total)

    return {
        "total_evaluated": total,
        "parse_failures": parse_failures,
        "no_label": no_label,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": round(accuracy, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "precision": round(precision, 4),
        "npv": round(npv, 4),
        "f1": round(f1, 4),
        "balanced_accuracy": round(balanced_accuracy, 4),
        "mcc": round(mcc, 4),
        "kappa": round(kappa, 4),
        "ci_accuracy_95": [round(ci_accuracy[0], 4), round(ci_accuracy[1], 4)],
        "ci_sensitivity_95": [round(ci_sensitivity[0], 4), round(ci_sensitivity[1], 4)],
        "ci_specificity_95": [round(ci_specificity[0], 4), round(ci_specificity[1], 4)],
        "ci_precision_95": [round(ci_precision[0], 4), round(ci_precision[1], 4)],
        "ci_npv_95": [round(ci_npv[0], 4), round(ci_npv[1], 4)],
        "ci_f1_95": [round(ci_f1[0], 4), round(ci_f1[1], 4)],
    }


def print_metrics(metrics: dict, body_part: str = "ALL"):
    print(f"\n{'='*60}")
    print(f"  AXIS EVALUATION RESULTS — {body_part}")
    print(f"{'='*60}")
    print(f"  Studies evaluated:  {metrics['total_evaluated']}")
    print(f"  Parse failures:     {metrics['parse_failures']}")
    if metrics.get("no_label", 0) > 0:
        print(f"  No ground truth:    {metrics['no_label']}  (inference only)")

    if metrics["total_evaluated"] == 0:
        print(f"\n  [No labeled studies to compute metrics.]")
        print(f"  Provide --labels-csv to enable metrics computation.")
        print(f"{'='*60}\n")
        return

    def _ci_str(key: str) -> str:
        ci = metrics.get(key, [0, 0])
        return f"[{ci[0]:.1%}, {ci[1]:.1%}]"

    print(f"")
    print(f"  Confusion Matrix:")
    print(f"    TP={metrics['tp']}  FP={metrics['fp']}")
    print(f"    FN={metrics['fn']}  TN={metrics['tn']}")
    print(f"")
    print(f"  Accuracy:          {metrics['accuracy']:.1%}  {_ci_str('ci_accuracy_95')}")
    print(f"  Sensitivity:       {metrics['sensitivity']:.1%}  {_ci_str('ci_sensitivity_95')}  (recall)")
    print(f"  Specificity:       {metrics['specificity']:.1%}  {_ci_str('ci_specificity_95')}")
    print(f"  Precision (PPV):   {metrics['precision']:.1%}  {_ci_str('ci_precision_95')}")
    print(f"  NPV:               {metrics['npv']:.1%}  {_ci_str('ci_npv_95')}")
    print(f"  F1 Score:          {metrics['f1']:.1%}  {_ci_str('ci_f1_95')}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.1%}")
    print(f"  MCC:               {metrics['mcc']:.3f}")
    print(f"  Cohen's kappa:     {metrics['kappa']:.3f}")
    print(f"{'='*60}\n")


# ============================================================
# BATCH RUNNER
# ============================================================

def run_batch(studies: list[dict], prompt_version: int = 1, multi_view: str = "majority") -> list[dict]:
    results = []

    for idx, study in enumerate(tqdm(studies, desc="Evaluating", unit="study"), start=1):
        images = study["images"]

        if multi_view == "first":
            images = images[:1]

        view_results = []
        view_details = []
        total_time = 0.0

        for img_path in images:
            try:
                output = query_model(img_path, prompt_version=prompt_version, timeout=300)
            except Exception as e:
                output = {"fracture": None, "abnormal": None, "confidence": None, "findings": f"[ERROR] {e}", "_meta": {}}

            if prompt_version == 4:
                pred = output.get("abnormal")
            else:
                pred = output.get("fracture")
            conf = output.get("confidence")
            t = output.get("_meta", {}).get("inference_time_s", 0) or 0
            total_time += t

            view_results.append(pred)
            view_details.append({
                "image": img_path,
                "prediction": pred,
                "confidence": conf,
                "findings": output.get("findings", ""),
                "category": output.get("category"),
                "location": output.get("location"),
                "raw_response": output.get("_meta", {}).get("raw_response", ""),
            })

        valid_votes = [v for v in view_results if v is not None]
        if not valid_votes:
            final_prediction = None
        elif multi_view == "any":
            final_prediction = any(valid_votes)
        else:
            pos_count = sum(1 for v in valid_votes if v)
            final_prediction = pos_count > len(valid_votes) / 2

        valid_confs = [d["confidence"] for d in view_details if isinstance(d["confidence"], (int, float))]
        avg_confidence = sum(valid_confs) / len(valid_confs) if valid_confs else None

        all_findings = "; ".join(d["findings"] for d in view_details if d["findings"] and not d["findings"].startswith("["))

        study_id = f"{study['body_part']}_{study['patient']}_{Path(study['path']).name}"

        result = {
            "study_id": study_id,
            "study_index": idx,
            "ground_truth": study["label"],
            "body_part": study["body_part"],
            "patient": study["patient"],
            "study_path": study["path"],
            "image": study["images"][0],
            "all_images": "|".join(study["images"]),
            "n_views": len(images),
            "n_valid_votes": len(valid_votes),
            "vote_detail": f"{sum(1 for v in valid_votes if v)}/{len(valid_votes)} positive" if valid_votes else "N/A",
            "prediction": final_prediction,
            "fracture": final_prediction,  # backward compat alias
            "confidence": round(avg_confidence, 3) if avg_confidence is not None else None,
            "findings": all_findings[:500],
            "inference_time_s": round(total_time, 1),
            "view_details": view_details,
        }
        results.append(result)

    return results


def _create_run_dir(run_label: str) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / f"{run_label}_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    return run_dir


def save_results(results: list[dict], metrics: dict, run_dir: Path, filename: str, studies: list[dict] = None):
    # Full per-study CSV
    csv_path = run_dir / f"{filename}.csv"
    fieldnames = [
        "study_index", "study_id", "body_part", "patient", "study_path",
        "image", "all_images", "ground_truth", "n_views", "n_valid_votes",
        "vote_detail", "prediction", "confidence", "findings", "inference_time_s",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"[✓] Per-study results saved to: {csv_path}")

    # Detailed per-view CSV
    views_csv_path = run_dir / f"{filename}_views.csv"
    view_fieldnames = [
        "study_index", "study_id", "body_part", "ground_truth",
        "image", "prediction", "confidence", "findings", "category", "location", "raw_response",
    ]
    with open(views_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=view_fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            for vd in r.get("view_details", []):
                row = {
                    "study_index": r["study_index"],
                    "study_id": r["study_id"],
                    "body_part": r["body_part"],
                    "ground_truth": r["ground_truth"],
                    **vd,
                }
                writer.writerow({k: v for k, v in row.items() if k in view_fieldnames})
    print(f"[✓] Per-view detail saved to: {views_csv_path}")

    # JSON with metrics
    json_path = run_dir / f"{filename}_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[✓] Metrics saved to: {json_path}")

    # Study manifest (paths stored relative to MURA data root for portability)
    if studies:
        manifest_path = run_dir / f"{filename}_manifest.json"
        data_root = _get_mura_data_root()
        manifest = [
            {
                "path": _to_relative_path(s["path"], data_root),
                "body_part": s["body_part"],
                "patient": s["patient"],
                "label": s["label"],
                "images": [_to_relative_path(img, data_root) for img in s["images"]],
            }
            for s in studies
        ]
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[✓] Study manifest saved to: {manifest_path}")
        print(f"    Re-use this exact sample with: --manifest {manifest_path}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="AXIS — Batch evaluate MSK pathology detection on MURA (PNG) or custom DICOMs",
        epilog="Examples:\n"
               "  python3 batch_eval.py --all --per-part 50\n"
               "  python3 batch_eval.py --all --per-part 50 --prompt-version 4 --seed 42\n"
               "  python3 batch_eval.py --manifest ../results/<run_dir>/manifest.json --model mlx-community/gemma-3-4b-it-4bit\n"
               "  python3 batch_eval.py --body-part XR_WRIST --max-studies 50\n"
               "  python3 batch_eval.py --data-dir /path/to/dicoms --labels-csv labels.csv\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode_group = parser.add_argument_group("Data source (pick one)")
    mode_group.add_argument("--body-part", "-b", choices=BODY_PARTS, help="MURA: evaluate single body part")
    mode_group.add_argument("--all", action="store_true", help="MURA: evaluate all body parts")
    mode_group.add_argument("--data-dir", help="Custom: path to directory of DICOMs/PNGs")
    mode_group.add_argument("--manifest", help="Re-run exact study list from a previous manifest JSON")

    custom_group = parser.add_argument_group("Custom data options")
    custom_group.add_argument("--labels-csv", help="CSV with filename,label columns (enables metrics)")

    sampling_group = parser.add_argument_group("Sampling")
    sampling_group.add_argument("--max-studies", "-n", type=int, default=None,
                                help="Max total studies (overrides --per-part)")
    sampling_group.add_argument("--per-part", type=int, default=50,
                                help="Studies per body part when using --all (default 50, balanced pos/neg)")
    sampling_group.add_argument("--exclude", nargs="+", default=[],
                                help="Body parts to exclude (e.g., --exclude XR_SHOULDER)")

    parser.add_argument("--backend", default=DEFAULT_BACKEND, choices=["mlx", "transformers"],
                        help=f"Inference backend (default: {DEFAULT_BACKEND})")
    parser.add_argument("--model", "-m", default=None,
                        help="Model path or HF ID (overrides default for chosen backend)")
    parser.add_argument("--prompt-version", "-p", type=int, default=1, choices=[1, 2, 3, 4],
                        help="Prompt version: 1=fracture binary, 2=fracture detailed, 3=fracture minimal, 4=pathology binary")
    parser.add_argument("--multi-view", default="majority", choices=["first", "majority", "any"],
                        help="Multi-view strategy: first, majority (default), or any")
    parser.add_argument("--split", default="valid", choices=["train", "valid"], help="MURA split to use")
    parser.add_argument("--mura-dir", default=str(MURA_DIR), help="Path to MURA dataset root")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for study sampling")
    parser.add_argument("--run-label", default=None, help="Custom label for the results folder (default: auto-generated)")
    args = parser.parse_args()

    if not args.body_part and not args.all and not args.data_dir and not args.manifest:
        parser.error("Specify --body-part XR_WRIST, --all, --data-dir /path/to/dicoms, or --manifest path/to/manifest.json")

    set_backend(args.backend, args.model)
    model_display = _resolve_model(args.backend, args.model)
    secs_per_image = _SECS_PER_IMAGE.get(args.backend, 25)
    print(f"[i] AXIS | Backend: {args.backend}, Model: {model_display}\n")

    random.seed(args.seed)

    # ---- Discover studies ----
    mura_data_root = _get_mura_data_root(args.mura_dir)

    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            print(f"[!] Manifest not found: {manifest_path}")
            sys.exit(1)
        with open(manifest_path) as f:
            manifest = json.load(f)
        studies = manifest
        has_labels = any(s["label"] is not None for s in studies)
        body_label = "MANIFEST"
        print(f"[i] Loaded manifest: {len(studies)} studies from {manifest_path}")

        # Resolve paths (handles absolute, relative, and migrated paths)
        _resolve_manifest_paths(studies, mura_data_root)

        # Upfront validation: sample image paths and assert they exist
        _validate_manifest_paths(studies)
    elif args.data_dir:
        labels_path = Path(args.labels_csv) if args.labels_csv else None
        studies = discover_custom_studies(Path(args.data_dir), labels_path)
        has_labels = any(s["label"] is not None for s in studies)
        body_label = "CUSTOM"
    else:
        mura_dir = Path(args.mura_dir)
        studies = discover_mura_studies(mura_dir, split=args.split, body_part=args.body_part)
        has_labels = True
        body_label = args.body_part or "ALL"

    if not studies:
        print("[!] No studies found. Check path and file structure.")
        sys.exit(1)

    # ---- Apply exclusions ----
    if args.exclude and not args.manifest:
        before = len(studies)
        studies = [s for s in studies if s["body_part"] not in args.exclude]
        print(f"[i] Excluded {args.exclude}: {before} -> {len(studies)} studies")
        body_label = f"ALL_excl_{'_'.join(args.exclude)}" if body_label == "ALL" else body_label

    # ---- Report totals ----
    labeled = [s for s in studies if s["label"] is not None]
    unlabeled = [s for s in studies if s["label"] is None]
    pos = sum(1 for s in labeled if s["label"])
    neg = sum(1 for s in labeled if not s["label"])
    total_images = sum(len(s["images"]) for s in studies)
    print(f"[i] Found {len(studies)} studies ({pos} abnormal, {neg} normal, {len(unlabeled)} unlabeled)")
    print(f"[i] Total images: {total_images} (multi-view mode: {args.multi_view})")

    # ---- Stratified sampling ----
    if args.manifest:
        pass
    elif args.all and not args.data_dir:
        active_parts = sorted(set(s["body_part"] for s in studies))
        sampled = []
        for bp in active_parts:
            bp_pos = [s for s in studies if s["body_part"] == bp and s["label"] is True]
            bp_neg = [s for s in studies if s["body_part"] == bp and s["label"] is False]
            random.shuffle(bp_pos)
            random.shuffle(bp_neg)
            n_per_class = args.per_part // 2
            n_pos = min(n_per_class, len(bp_pos))
            n_neg = min(n_per_class, len(bp_neg))
            selected = bp_pos[:n_pos] + bp_neg[:n_neg]
            sampled.extend(selected)
            print(f"  {bp:<15} {n_pos} pos + {n_neg} neg = {len(selected)} studies "
                  f"({sum(len(s['images']) for s in selected)} images)")
        random.shuffle(sampled)
        studies = sampled
        total_images = sum(len(s["images"]) for s in studies)
        est_minutes = total_images * secs_per_image / 60
        print(f"[i] Sampled {len(studies)} studies, {total_images} images")
        print(f"[i] Estimated runtime: ~{est_minutes:.0f} minutes ({args.multi_view} mode, {args.backend} backend)")
    elif args.max_studies and len(studies) > args.max_studies:
        positives = [s for s in studies if s["label"] is True]
        negatives = [s for s in studies if s["label"] is False]
        n_per_class = args.max_studies // 2
        random.shuffle(positives)
        random.shuffle(negatives)
        studies = positives[:n_per_class] + negatives[:n_per_class]
        random.shuffle(studies)
        total_images = sum(len(s["images"]) for s in studies)
        print(f"[i] Balanced sample: {len(studies)} studies, {total_images} images")

    # ---- Create run directory ----
    model_short = Path(model_display).name if "/" in model_display else model_display
    model_short = model_short.replace(".", "_")[:30]
    mv_tag = args.multi_view[0]
    if args.run_label:
        run_label = args.run_label
    else:
        run_label = f"axis_{body_label}_pv{args.prompt_version}_{mv_tag}_n{len(studies)}_{model_short}"
    run_dir = _create_run_dir(run_label)
    print(f"[i] Results will be saved to: {run_dir}")

    # ---- Run evaluation ----
    print(f"\n[->] Running AXIS evaluation: prompt v{args.prompt_version}, multi-view={args.multi_view}, backend={args.backend}...")
    t0 = time.time()
    results = run_batch(studies, prompt_version=args.prompt_version, multi_view=args.multi_view)
    elapsed = time.time() - t0

    # ---- Compute and display overall metrics ----
    metrics = compute_metrics(results)
    metrics["total_time_s"] = round(elapsed, 1)
    metrics["avg_time_per_study_s"] = round(elapsed / len(results), 1) if results else 0
    metrics["total_images_processed"] = sum(r.get("n_views", 1) for r in results)
    metrics["prompt_version"] = args.prompt_version
    metrics["multi_view"] = args.multi_view
    metrics["backend"] = args.backend
    metrics["model"] = model_display
    metrics["body_part"] = body_label
    metrics["excluded"] = args.exclude
    metrics["data_source"] = args.manifest or args.data_dir or args.mura_dir
    metrics["seed"] = args.seed
    metrics["manifest"] = args.manifest or None
    metrics["timestamp"] = datetime.now().isoformat()
    metrics["run_dir"] = str(run_dir)

    print_metrics(metrics, body_label)

    # ---- Save ----
    filename = run_dir.name
    save_results(results, metrics, run_dir, filename, studies=studies)

    # ---- Per-body-part breakdown ----
    active_parts = sorted(set(r["body_part"] for r in results))
    if len(active_parts) > 1:
        all_bp_metrics = {}
        for bp in active_parts:
            bp_results = [r for r in results if r["body_part"] == bp]
            if bp_results:
                bp_metrics = compute_metrics(bp_results)
                print_metrics(bp_metrics, bp)
                all_bp_metrics[bp] = bp_metrics

        bp_json_path = run_dir / f"{filename}_per_bodypart.json"
        with open(bp_json_path, "w") as f:
            json.dump(all_bp_metrics, f, indent=2)
        print(f"[✓] Per-body-part metrics saved to: {bp_json_path}")


if __name__ == "__main__":
    main()
