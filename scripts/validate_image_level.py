#!/usr/bin/env python3
"""Validate image-level views CSVs by re-aggregating to study-level and
comparing against known study-level confusion matrices.

Usage:
    python3 validate_image_level.py --views path/to/views.csv --expected TP=100,FP=10,FN=20,TN=80
    python3 validate_image_level.py --views path/to/views.csv  # no expected = just re-aggregate
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


def parse_bool(val: str) -> bool:
    v = val.strip().lower()
    if v == "true":
        return True
    if v == "false":
        return False
    raise ValueError(f"Cannot parse boolean: {val!r}")


def load_views(csv_path: Path) -> list[dict]:
    """Load views CSV, handling multiline raw_response fields."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def aggregate_majority_vote(rows: list[dict]) -> dict:
    """Group by study_id, majority vote, return confusion matrix."""
    studies = defaultdict(lambda: {"gt": None, "preds": []})
    n_dropped = 0
    for row in rows:
        pred_str = row["prediction"].strip().lower()
        if pred_str not in ("true", "false"):
            n_dropped += 1
            continue
        sid = row["study_id"]
        gt = parse_bool(row["ground_truth"])
        pred = pred_str == "true"
        if studies[sid]["gt"] is None:
            studies[sid]["gt"] = gt
        else:
            assert studies[sid]["gt"] == gt, f"Inconsistent GT for {sid}"
        studies[sid]["preds"].append(pred)
    if n_dropped:
        print(f"  Dropped {n_dropped} rows with missing/invalid predictions")

    tp = fp = fn = tn = 0
    for sid, info in studies.items():
        n_pos = sum(info["preds"])
        n_total = len(info["preds"])
        study_pred = n_pos > n_total / 2
        gt = info["gt"]
        if gt and study_pred:
            tp += 1
        elif not gt and study_pred:
            fp += 1
        elif gt and not study_pred:
            fn += 1
        else:
            tn += 1

    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "n_studies": len(studies)}


def main():
    parser = argparse.ArgumentParser(description="Validate image-level views CSV")
    parser.add_argument("--views", required=True, help="Path to views CSV")
    parser.add_argument("--expected", default=None,
                        help="Expected confusion matrix as TP=x,FP=y,FN=z,TN=w")
    args = parser.parse_args()

    csv_path = Path(args.views)
    if not csv_path.exists():
        print(f"ERROR: CSV not found at {csv_path}")
        sys.exit(1)

    rows = load_views(csv_path)
    print(f"Loaded {len(rows)} image-level rows from {csv_path.name}")

    result = aggregate_majority_vote(rows)
    n = result.pop("n_studies")
    print(f"Studies: {n}")
    print(f"Re-aggregated: TP={result['TP']} FP={result['FP']} FN={result['FN']} TN={result['TN']}")

    if args.expected:
        expected = {}
        for pair in args.expected.split(","):
            k, v = pair.split("=")
            expected[k.strip()] = int(v.strip())

        print(f"Expected:      TP={expected['TP']} FP={expected['FP']} FN={expected['FN']} TN={expected['TN']}")

        if result == expected:
            print("--> MATCH")
        else:
            print("--> MISMATCH")
            for key in ["TP", "FP", "FN", "TN"]:
                delta = result[key] - expected[key]
                if delta != 0:
                    print(f"    {key}: got {result[key]}, expected {expected[key]}, delta={delta:+d}")
            sys.exit(1)


if __name__ == "__main__":
    main()
