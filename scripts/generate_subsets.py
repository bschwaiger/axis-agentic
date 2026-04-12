#!/usr/bin/env python3
"""Generate balanced MURA subsets for AXIS Agentic hackathon.

Reads the MURA valid set, samples balanced subsets, copies study folders,
and generates manifest CSVs.
"""

import csv
import os
import random
import shutil
import sys
from pathlib import Path

MURA_ROOT = Path(os.environ.get("MURA_DATA_ROOT", "data/mura"))
VALID_CSV = MURA_ROOT / "valid_labeled_studies.csv"
OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "data"

SUBSETS = {
    "eval-020": 20,
    "eval-100": 100,
    "eval-300": 300,
    "train-050": 50,
}

SEED = 42


def load_valid_studies():
    """Load valid set studies grouped by label."""
    normal, abnormal = [], []
    with open(VALID_CSV) as f:
        reader = csv.reader(f)
        for row in reader:
            study_path, label = row[0].rstrip("/"), int(row[1])
            if label == 0:
                normal.append(study_path)
            else:
                abnormal.append(study_path)
    print(f"Valid set: {len(normal)} normal, {len(abnormal)} abnormal")
    return normal, abnormal


def copy_study(study_rel_path: str, dest_dir: Path):
    """Copy a study folder to dest_dir preserving structure."""
    # CSV paths look like: MURA-v1.1/valid/XR_WRIST/patient11185/study1_positive
    # Actual data lives at: mura/valid/XR_WRIST/patient11185/study1_positive
    # Strip the MURA-v1.1/ prefix to get the real filesystem path
    parts = Path(study_rel_path).parts
    fs_rel = Path(*parts[1:])  # skip MURA-v1.1/, keep valid/XR_WRIST/...
    src = MURA_ROOT / fs_rel
    if not src.exists():
        print(f"  WARNING: source not found: {src}")
        return False
    # Preserve body_part/patient/study structure in output
    rel = Path(*parts[2:])  # skip MURA-v1.1/valid/
    dst = dest_dir / rel
    dst.mkdir(parents=True, exist_ok=True)
    for img in sorted(src.iterdir()):
        if img.is_file():
            shutil.copy2(img, dst / img.name)
    return True


def generate_subset(name: str, n: int, normal: list, abnormal: list, rng: random.Random):
    """Sample n/2 normal + n/2 abnormal, copy files, write manifest."""
    half = n // 2
    if half > len(normal) or half > len(abnormal):
        print(f"ERROR: not enough studies for {name} (need {half} per class)")
        sys.exit(1)

    sampled_normal = rng.sample(normal, half)
    sampled_abnormal = rng.sample(abnormal, half)

    subset_dir = OUTPUT_ROOT / name
    manifest_path = subset_dir / "manifest.csv"

    print(f"\n{'='*60}")
    print(f"Generating {name}: {half} normal + {half} abnormal = {n} studies")
    print(f"{'='*60}")

    rows = []
    for study_path in sampled_normal:
        if copy_study(study_path, subset_dir):
            parts = Path(study_path).parts
            rel = str(Path(*parts[2:]))
            rows.append((rel, 0))

    for study_path in sampled_abnormal:
        if copy_study(study_path, subset_dir):
            parts = Path(study_path).parts
            rel = str(Path(*parts[2:]))
            rows.append((rel, 1))

    # Write manifest
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["study_path", "label"])
        for row in sorted(rows):
            writer.writerow(row)

    print(f"  Copied {len(rows)} studies, manifest: {manifest_path}")

    # Remove sampled studies from pools to avoid overlap
    for s in sampled_normal:
        normal.remove(s)
    for s in sampled_abnormal:
        abnormal.remove(s)

    return rows


def validate_subsets():
    """Print summary and check balance."""
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    all_ok = True
    for name in SUBSETS:
        manifest = OUTPUT_ROOT / name / "manifest.csv"
        if not manifest.exists():
            print(f"  {name}: MISSING manifest!")
            all_ok = False
            continue
        with open(manifest) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        n_normal = sum(1 for r in rows if int(r["label"]) == 0)
        n_abnormal = sum(1 for r in rows if int(r["label"]) == 1)
        total = len(rows)
        balance = abs(n_normal - n_abnormal) / total if total > 0 else 1.0
        status = "OK" if balance <= 0.1 else "IMBALANCED"
        if status != "OK":
            all_ok = False
        print(f"  {name}: {total} studies ({n_normal} normal, {n_abnormal} abnormal) — {status}")

    if not all_ok:
        print("\nABORTING: balance check failed!")
        sys.exit(1)
    print("\nAll subsets balanced. Done.")


def main():
    rng = random.Random(SEED)
    normal, abnormal = load_valid_studies()

    # Generate subsets (largest first to ensure we have enough studies)
    for name in ["eval-300", "eval-100", "train-050", "eval-020"]:
        generate_subset(name, SUBSETS[name], normal, abnormal, rng)

    validate_subsets()


if __name__ == "__main__":
    main()
