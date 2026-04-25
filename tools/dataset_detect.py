"""LLM-assisted dataset structure detection.

Given a dataset directory, gather evidence (file tree, candidate manifests,
sample image metadata), ask the active Engine to propose a schema, walk
the dataset with that schema, and write a manifest.csv. The schema is
cached to <dataset_path>/.axis_schema.json so re-runs skip the LLM call.

Standalone CLI for PR2:

    python -m tools.dataset_detect path/to/dataset

PR3 will surface this as a cockpit checkpoint inside the evaluator flow.
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ----------------------------------------------------------------------
# Evidence gathering — no LLM, pure filesystem inspection
# ----------------------------------------------------------------------

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".dcm", ".tif", ".tiff", ".bmp", ".gif", ".webp"}


def inspect_dataset(dataset_path: str, max_depth: int = 3, sample_files: int = 20) -> dict:
    """Walk the dataset and return structured evidence for an LLM to reason over."""
    root = Path(dataset_path)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset path not found: {root}")

    # Bounded directory listing
    tree: list[str] = []
    image_samples: list[str] = []
    csv_candidates: list[str] = []
    json_candidates: list[str] = []

    for current_root, dirs, files in os.walk(root):
        rel = Path(current_root).relative_to(root)
        depth = len(rel.parts)
        if depth > max_depth:
            dirs[:] = []
            continue
        # Limit branching for huge datasets
        dirs[:] = sorted(dirs)[:50]
        for d in dirs:
            tree.append(f"{rel / d}/")
        for f in sorted(files):
            ext = Path(f).suffix.lower()
            full_rel = str(rel / f) if rel.parts else f
            if ext in IMAGE_EXTS and len(image_samples) < sample_files:
                image_samples.append(full_rel)
            elif ext == ".csv" and len(csv_candidates) < 5:
                csv_candidates.append(full_rel)
            elif ext == ".json" and len(json_candidates) < 5:
                json_candidates.append(full_rel)
        if len(tree) > 200:
            break

    # Read up to 5 lines of each candidate manifest CSV
    csv_previews = {}
    for c in csv_candidates:
        try:
            with open(root / c) as fh:
                lines = []
                for i, line in enumerate(fh):
                    if i >= 5:
                        break
                    lines.append(line.rstrip("\n"))
                csv_previews[c] = lines
        except Exception as e:
            csv_previews[c] = [f"(error reading: {e})"]

    # Distinct folder name patterns at depth 1 + 2 — useful for class-folder detection
    depth1_dirs = sorted({p.parts[0] for p in (Path(t) for t in tree) if len(p.parts) >= 1})
    depth2_dirs = sorted({"/".join(p.parts[:2]) for p in (Path(t) for t in tree) if len(p.parts) >= 2})

    # Image counts per top-level folder (cheap, useful)
    counts: dict[str, int] = {}
    for img in image_samples:
        top = Path(img).parts[0] if Path(img).parts else "."
        counts[top] = counts.get(top, 0) + 1

    return {
        "dataset_path": str(root),
        "tree_sample": tree[:200],
        "image_samples": image_samples,
        "csv_candidates": csv_candidates,
        "csv_previews": csv_previews,
        "json_candidates": json_candidates,
        "depth1_dirs": depth1_dirs[:50],
        "depth2_dirs": depth2_dirs[:50],
        "image_counts_per_top_folder": counts,
        "total_images_seen": len(image_samples),
    }


# ----------------------------------------------------------------------
# Schema definition — what the LLM is asked to propose
# ----------------------------------------------------------------------

SCHEMA_DOC = """\
Propose a JSON schema describing how to extract (study_path, label) pairs.

Required fields:
  source: "manifest_csv" | "folder_name_pattern"

When source = "manifest_csv":
  csv_path:           relative path of the manifest CSV
  path_column:        column name holding study_path (or image_path)
  label_column:       column name holding the label
  label_positive:     value of label_column that means abnormal/positive (label=1)
  label_negative:     (optional) value that means normal/negative (label=0). If
                      omitted, anything != label_positive is 0.

When source = "folder_name_pattern":
  study_glob:         glob (relative to dataset_path) matching study folders
                      (e.g., "XR_*/patient*/study*")
  positive_pattern:   regex; folder matches => label=1 (e.g., "_positive$")
  negative_pattern:   (optional) regex; folder matches => label=0
                      (default: anything not matching positive_pattern is 0)

Return ONLY valid JSON, no commentary.
"""


# ----------------------------------------------------------------------
# LLM-backed schema proposal
# ----------------------------------------------------------------------

def propose_schema(evidence: dict, engine=None) -> dict:
    """Ask the active Engine to propose a schema given the evidence."""
    if engine is None:
        from orchestrator.engine import get_default_engine
        engine = get_default_engine()

    user_msg = (
        f"You are inspecting a radiology image dataset at {evidence['dataset_path']}.\n\n"
        f"EVIDENCE:\n{json.dumps(evidence, indent=2)[:6000]}\n\n"
        f"{SCHEMA_DOC}"
    )

    engine.reset(system="You parse dataset directory structures into manifests.",
                 tools=[], max_tokens=1024)
    response = engine.send_user_message(user_msg)
    text = response.text or ""

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"Engine did not return JSON. Got: {text[:300]}")
    return json.loads(match.group(0))


# ----------------------------------------------------------------------
# Apply schema -> manifest rows
# ----------------------------------------------------------------------

def walk_with_schema(dataset_path: str, schema: dict) -> list[dict]:
    """Apply the schema to the dataset and return manifest rows."""
    root = Path(dataset_path)
    if not root.is_absolute():
        root = PROJECT_ROOT / root

    source = schema.get("source")

    if source == "manifest_csv":
        csv_path = root / schema["csv_path"]
        path_col = schema["path_column"]
        label_col = schema["label_column"]
        label_positive = str(schema["label_positive"])
        label_negative = schema.get("label_negative")
        rows = []
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                study_path = r.get(path_col, "").strip()
                label_val = str(r.get(label_col, "")).strip()
                if not study_path:
                    continue
                if label_val == label_positive:
                    label = 1
                elif label_negative is not None and label_val == str(label_negative):
                    label = 0
                else:
                    label = 0
                rows.append({"study_path": study_path, "label": label})
        return rows

    if source == "folder_name_pattern":
        study_glob = schema["study_glob"]
        pos = re.compile(schema["positive_pattern"])
        neg = re.compile(schema["negative_pattern"]) if schema.get("negative_pattern") else None

        rows = []
        # Walk and match against the glob
        for current_root, dirs, _files in os.walk(root):
            rel = Path(current_root).relative_to(root)
            rel_str = str(rel) if rel != Path(".") else ""
            if rel_str and fnmatch.fnmatch(rel_str, study_glob):
                folder_name = rel.name
                if pos.search(folder_name):
                    label = 1
                elif neg and neg.search(folder_name):
                    label = 0
                else:
                    label = 0
                rows.append({"study_path": rel_str, "label": label})
        rows.sort(key=lambda r: r["study_path"])
        return rows

    raise ValueError(f"Unknown schema source: {source!r}")


# ----------------------------------------------------------------------
# Write manifest CSV + schema cache
# ----------------------------------------------------------------------

def write_manifest(rows: list[dict], output_path: str) -> dict:
    out = Path(output_path)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["study_path", "label"])
        writer.writeheader()
        writer.writerows(rows)

    return {
        "status": "success",
        "manifest_path": str(out),
        "row_count": len(rows),
        "positives": sum(1 for r in rows if r["label"] == 1),
        "negatives": sum(1 for r in rows if r["label"] == 0),
    }


def cached_schema_path(dataset_path: str) -> Path:
    root = Path(dataset_path)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return root / ".axis_schema.json"


def load_cached_schema(dataset_path: str) -> dict | None:
    p = cached_schema_path(dataset_path)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def save_cached_schema(dataset_path: str, schema: dict) -> None:
    p = cached_schema_path(dataset_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(schema, indent=2))


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Detect dataset structure and write a manifest CSV.")
    parser.add_argument("dataset_path", help="Path to the dataset root")
    parser.add_argument("--manifest-out", default=None,
                        help="Output manifest path (default: <dataset>/manifest.csv)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Ignore cached schema and re-detect")
    parser.add_argument("--print-evidence", action="store_true",
                        help="Print the gathered evidence and exit")
    args = parser.parse_args()

    # Load .env
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                if k.strip() and not os.environ.get(k.strip()):
                    os.environ[k.strip()] = v.strip()

    print(f"  Inspecting {args.dataset_path}...")
    evidence = inspect_dataset(args.dataset_path)
    print(f"  Found: {evidence['total_images_seen']} sample images, "
          f"{len(evidence['csv_candidates'])} CSV candidate(s), "
          f"{len(evidence['depth1_dirs'])} top-level folders")

    if args.print_evidence:
        print(json.dumps(evidence, indent=2))
        return

    cached = None if args.no_cache else load_cached_schema(args.dataset_path)
    if cached:
        print(f"  Using cached schema at {cached_schema_path(args.dataset_path)}")
        schema = cached
    else:
        print(f"  Asking engine to propose schema...")
        schema = propose_schema(evidence)
        print(f"  Proposed schema:")
        print(json.dumps(schema, indent=4))
        confirm = input("\n  Accept? [Y/n] ").strip().lower()
        if confirm and confirm != "y":
            print("  Aborted.")
            return
        save_cached_schema(args.dataset_path, schema)
        print(f"  Cached to {cached_schema_path(args.dataset_path)}")

    print(f"  Walking dataset with schema...")
    rows = walk_with_schema(args.dataset_path, schema)

    out = args.manifest_out or str(Path(evidence["dataset_path"]) / "manifest.csv")
    result = write_manifest(rows, out)
    print(f"  Wrote {result['row_count']} rows to {result['manifest_path']}")
    print(f"  Positives: {result['positives']}, Negatives: {result['negatives']}")


if __name__ == "__main__":
    main()
