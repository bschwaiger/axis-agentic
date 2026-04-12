#!/usr/bin/env python3
"""
AXIS — Automated X-Ray Identification for the Skeleton
MedGemma 1.5 4B via MLX (Apple Silicon, default) or HuggingFace Transformers (CUDA/cloud).
Accepts DICOM (.dcm), PNG, or JPEG. DICOMs are auto-windowed for bone.

Backends:
    mlx          — Apple Silicon via mlx-vlm. 4-bit quantized, ~42 tok/s on M4. Default.
    transformers — HuggingFace Transformers. float32 on MPS, float16 on CUDA. For cloud/GPU.

First run downloads model weights from HuggingFace (one-time).
Requires (mlx):          pip install mlx-vlm
Requires (transformers):  pip install torch transformers accelerate pillow pydicom numpy

Usage:
    python3 axis_detector.py --image path/to/xray.dcm
    python3 axis_detector.py --image path/to/xray.dcm --backend transformers
    python3 axis_detector.py --dicom-dir path/to/study_folder/
    python3 axis_detector.py --image xray.dcm --prompt-version 2 --verbose
    python3 axis_detector.py --image xray.dcm --model ~/Projects/axis/models/mlx-medgemma-1.5-4bit
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

# ============================================================
# CONFIG
# ============================================================

# Default models per backend
MODELS = {
    "mlx": os.environ.get("MEDGEMMA_MODEL_PATH", "models/axis-mura-v1-4bit"),
    "mlx_bf16": "mlx-community/medgemma-1.5-4b-it-bf16",
    "transformers": "google/medgemma-1.5-4b-it",
}

# Auto-detect best available backend
def _detect_default_backend() -> str:
    try:
        import mlx_vlm  # noqa: F401
        return "mlx"
    except ImportError:
        return "transformers"

DEFAULT_BACKEND = _detect_default_backend()

PROMPTS = {
    1: """You are an expert musculoskeletal radiologist. Analyze this X-ray image for the presence of fractures.

Respond ONLY with valid JSON in this exact format, no other text:
{
    "fracture": true or false,
    "confidence": 0.0 to 1.0,
    "location": "brief anatomical description of fracture location, or null if no fracture",
    "findings": "one-sentence summary of key findings"
}""",

    2: """You are an expert musculoskeletal radiologist performing a systematic review of this bone X-ray.

Step 1: Identify the body part and projection.
Step 2: Systematically evaluate cortical integrity, alignment, joint spaces, and soft tissues.
Step 3: Determine if a fracture is present.

Respond ONLY with valid JSON in this exact format, no other text:
{
    "fracture": true or false,
    "confidence": 0.0 to 1.0,
    "body_part": "identified body part and projection",
    "location": "fracture location or null",
    "fracture_type": "fracture classification or null",
    "additional_findings": "other abnormalities noted or null",
    "findings": "one-sentence summary"
}""",

    3: """Analyze this X-ray. Is there a fracture? Reply ONLY with JSON: {"fracture": true/false, "confidence": 0.0-1.0, "findings": "one sentence"}""",

    4: """You are an expert musculoskeletal radiologist. Analyze this X-ray image and determine whether it is normal or abnormal.

Consider all possible musculoskeletal pathologies including fractures, post-surgical hardware, degenerative changes, dislocations, soft tissue abnormalities, and any other findings.

Respond ONLY with valid JSON in this exact format, no other text:
{
    "abnormal": true or false,
    "confidence": 0.0 to 1.0,
    "category": "fracture, hardware, degenerative, dislocation, soft_tissue, other, or null if normal",
    "location": "brief anatomical description, or null if normal",
    "findings": "one-sentence summary of key findings"
}""",
}


# ============================================================
# MODEL LOADING — TRANSFORMERS BACKEND (singleton)
# ============================================================

_tf_model = None
_tf_processor = None


def _load_transformers(model_id: str):
    """Load model via HuggingFace Transformers. Cached after first call."""
    global _tf_model, _tf_processor
    if _tf_model is not None:
        return _tf_model, _tf_processor

    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    print(f"[↓] Loading {model_id} (transformers)...")
    print(f"    First run downloads ~8GB. Subsequent runs load from HF cache.")

    if torch.cuda.is_available():
        device_info = f"CUDA ({torch.cuda.get_device_name(0)})"
        dtype = torch.float16
    elif torch.backends.mps.is_available():
        device_info = "MPS (Apple Silicon GPU)"
        dtype = torch.float32  # float16 produces empty output on MPS for Gemma
    else:
        device_info = "CPU (slow)"
        dtype = torch.float32

    print(f"    Device: {device_info}")

    t0 = time.time()

    _tf_processor = AutoProcessor.from_pretrained(model_id)
    _tf_model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=dtype,
        device_map="auto",
    )

    elapsed = time.time() - t0
    print(f"[✓] Model loaded in {elapsed:.1f}s\n")
    return _tf_model, _tf_processor


# ============================================================
# MODEL LOADING — MLX BACKEND (singleton)
# ============================================================

_mlx_model = None
_mlx_processor = None


def _load_mlx(model_path: str):
    """Load model via mlx-vlm. Cached after first call."""
    global _mlx_model, _mlx_processor
    if _mlx_model is not None:
        return _mlx_model, _mlx_processor

    from mlx_vlm import load

    resolved = str(Path(model_path).expanduser())
    # If path is a local directory, use it directly; otherwise treat as HF hub ID
    if Path(resolved).is_dir():
        display = resolved
    else:
        display = model_path
        resolved = model_path  # let mlx-vlm resolve from HF hub

    print(f"[↓] Loading {display} (mlx)...")

    t0 = time.time()
    _mlx_model, _mlx_processor = load(resolved)
    elapsed = time.time() - t0

    print(f"[✓] Model loaded in {elapsed:.1f}s\n")
    return _mlx_model, _mlx_processor


# ============================================================
# DICOM HANDLING
# ============================================================

def _apply_dicom_windowing(pixel_array: np.ndarray, ds) -> np.ndarray:
    """Apply DICOM windowing. Handles RescaleSlope/Intercept, PhotometricInterpretation, VOI LUT."""
    img = pixel_array.astype(np.float64)

    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    img = img * slope + intercept

    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth", None)

    if wc is not None and ww is not None:
        if hasattr(wc, "__iter__") and not isinstance(wc, str):
            wc, ww = float(wc[0]), float(ww[0])
        else:
            wc, ww = float(wc), float(ww)
    else:
        wc = (img.max() + img.min()) / 2
        ww = img.max() - img.min()
        if ww == 0:
            ww = 1

    lower = wc - ww / 2
    upper = wc + ww / 2
    img = np.clip(img, lower, upper)
    img = ((img - lower) / (upper - lower) * 255).astype(np.uint8)

    photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    if photometric == "MONOCHROME1":
        img = 255 - img

    return img


def load_image(image_path: str) -> Image.Image:
    """
    Load image from DICOM, PNG, or JPEG. Auto-detects format.
    Returns PIL Image (RGB, resized to max 1024px).
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Image file is empty: {image_path}")

    is_dicom = path.suffix.lower() in (".dcm", ".dicom")
    if not is_dicom and path.suffix == "":
        try:
            with open(path, "rb") as f:
                f.seek(128)
                is_dicom = f.read(4) == b"DICM"
        except Exception:
            pass

    if is_dicom:
        if not HAS_PYDICOM:
            print("[!] pydicom not installed. Run: pip3 install pydicom")
            sys.exit(1)
        ds = pydicom.dcmread(str(path))
        pixel_array = ds.pixel_array
        if pixel_array.ndim == 3 and pixel_array.shape[0] > 1:
            pixel_array = pixel_array[0]
        img_array = _apply_dicom_windowing(pixel_array, ds)
        pil_img = Image.fromarray(img_array, mode="L")
    else:
        pil_img = Image.open(str(path))

    # MedGemma expects RGB
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    # Resize if very large (some DICOMs are 3000x3000+)
    max_dim = 1024
    if max(pil_img.size) > max_dim:
        pil_img.thumbnail((max_dim, max_dim), Image.LANCZOS)

    return pil_img


def find_dicoms(directory: str) -> list[str]:
    """Recursively find DICOM files in a directory."""
    results = []
    for p in sorted(Path(directory).rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() in (".dcm", ".dicom"):
            results.append(str(p))
        elif p.suffix == "":
            try:
                with open(p, "rb") as f:
                    f.seek(128)
                    if f.read(4) == b"DICM":
                        results.append(str(p))
            except Exception:
                pass
    return results


def find_images(directory: str) -> list[str]:
    """Find all supported image files (DICOM, PNG, JPEG) in a directory."""
    img_extensions = {".png", ".jpg", ".jpeg", ".dcm", ".dicom"}
    results = []
    for p in sorted(Path(directory).rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() in img_extensions:
            results.append(str(p))
        elif p.suffix == "":
            try:
                with open(p, "rb") as f:
                    f.seek(128)
                    if f.read(4) == b"DICM":
                        results.append(str(p))
            except Exception:
                pass
    return results


# ============================================================
# INFERENCE — UNIFIED ENTRY POINT
# ============================================================

# Active backend/model config (set via set_backend or CLI)
_active_backend: str = DEFAULT_BACKEND
_active_model: str | None = None


def set_backend(backend: str, model: str | None = None):
    """Set the active inference backend and model. Called by CLI or batch_eval."""
    global _active_backend, _active_model
    _active_backend = backend
    _active_model = model


def _resolve_model(backend: str, model_override: str | None) -> str:
    """Resolve which model ID/path to use for a given backend."""
    if model_override:
        return model_override
    return MODELS.get(backend, MODELS["transformers"])


def query_model(image_path: str, prompt_version: int = 1, timeout: int = 300) -> dict:
    """
    Send image to the model and parse structured response.
    Uses the active backend (set via set_backend or auto-detected).
    Returns dict with prediction, confidence, findings, plus metadata.
    """
    backend = _active_backend
    model_id = _resolve_model(backend, _active_model)

    if backend == "mlx":
        return _query_mlx(image_path, model_id, prompt_version)
    else:
        return _query_transformers(image_path, model_id, prompt_version)


# Keep legacy alias for backward compatibility during transition
query_medgemma = query_model


def _query_transformers(image_path: str, model_id: str, prompt_version: int) -> dict:
    """Inference via HuggingFace Transformers (original path)."""
    import torch

    model, processor = _load_transformers(model_id)
    pil_image = load_image(image_path)
    prompt_text = PROMPTS[prompt_version]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    t0 = time.time()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )
    elapsed = time.time() - t0

    generated_ids = output_ids[0][input_len:]
    raw_response = processor.decode(generated_ids, skip_special_tokens=True)

    result = _parse_json_response(raw_response, prompt_version)
    result["_meta"] = {
        "image": str(image_path),
        "model": model_id,
        "backend": "transformers",
        "prompt_version": prompt_version,
        "inference_time_s": round(elapsed, 1),
        "raw_response": raw_response,
    }
    return result


def _build_suppress_sampler(processor):
    """Build a sampler that suppresses MedGemma's <unused*> thinking tokens.

    After LoRA merge, Gemma 3's extended thinking mechanism can activate,
    producing <unused94>thought... tokens that hijack generation. This sampler
    wraps the default argmax behavior and zeroes out suppressed token logits
    before sampling.
    """
    import mlx.core as mx

    suppress_ids = []
    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    for i in range(256):
        tok_name = f"<unused{i}>"
        tid = tokenizer.convert_tokens_to_ids(tok_name)
        if tid != tokenizer.unk_token_id:
            suppress_ids.append(tid)

    if not suppress_ids:
        return None

    # Pre-build a boolean mask (True = keep, False = suppress)
    # We don't know vocab size yet, so build on first call
    _mask_cache = {}

    def suppressing_sampler(logits: mx.array) -> mx.array:
        vocab_size = logits.shape[-1]
        if vocab_size not in _mask_cache:
            keep = [True] * vocab_size
            for tid in suppress_ids:
                if tid < vocab_size:
                    keep[tid] = False
            _mask_cache[vocab_size] = mx.array(keep)
        mask = _mask_cache[vocab_size]
        logits = mx.where(mask, logits, mx.array(-1e9))
        return mx.argmax(logits, axis=-1)

    return suppressing_sampler


# Cache the sampler so we only build it once
_suppress_sampler = None
_suppress_sampler_built = False


def _get_suppress_sampler(processor):
    global _suppress_sampler, _suppress_sampler_built
    if not _suppress_sampler_built:
        _suppress_sampler = _build_suppress_sampler(processor)
        if _suppress_sampler:
            print(f"[i] suppress_tokens: sampler active for <unused*> tokens")
        else:
            print(f"[i] suppress_tokens: no <unused*> tokens found in tokenizer")
        _suppress_sampler_built = True
    return _suppress_sampler


def _query_mlx(image_path: str, model_path: str, prompt_version: int) -> dict:
    """Inference via mlx-vlm (Apple Silicon)."""
    import tempfile
    from mlx_vlm import generate

    model, processor = _load_mlx(model_path)
    pil_image = load_image(image_path)
    prompt_text = PROMPTS[prompt_version]

    # Format prompt with chat template so the Gemma 3 processor inserts
    # the <start_of_image> token. Without this, the processor raises
    # "Prompt contained 0 image tokens but received 1 images."
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt_text},
        ]}
    ]
    formatted_prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    # Save processed image to temp PNG. mlx-vlm expects file paths,
    # and our load_image() may have windowed a DICOM, converted to RGB,
    # or resized. Always write temp to be safe.
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    pil_image.save(temp_file.name)

    # Build suppress_tokens sampler for <unused*> thinking tokens
    suppress_sampler = _get_suppress_sampler(processor)

    t0 = time.time()
    try:
        gen_kwargs = dict(
            image=[temp_file.name],
            max_tokens=512,
            temp=0.1,
        )
        if suppress_sampler:
            gen_kwargs["sampler"] = suppress_sampler
        gen_result = generate(
            model,
            processor,
            formatted_prompt,
            **gen_kwargs,
        )
    finally:
        Path(temp_file.name).unlink(missing_ok=True)
    elapsed = time.time() - t0

    # generate() returns a GenerationResult dataclass; extract the text
    raw_response = gen_result.text

    result = _parse_json_response(raw_response, prompt_version)
    result["_meta"] = {
        "image": str(image_path),
        "model": model_path,
        "backend": "mlx",
        "prompt_version": prompt_version,
        "inference_time_s": round(elapsed, 1),
        "generation_tps": round(gen_result.generation_tps, 1),
        "prompt_tps": round(gen_result.prompt_tps, 1),
        "peak_memory_gb": round(gen_result.peak_memory, 2),
        "raw_response": raw_response,
    }
    return result


def _parse_json_response(text: str, prompt_version: int = 1) -> dict:
    """Extract JSON from model response, handling common formatting issues."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()

    parsed = None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                parsed = json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                pass

    if parsed is None:
        return {
            "abnormal": None,
            "fracture": None,
            "confidence": None,
            "findings": f"[PARSE_FAILED] Raw response: {text[:500]}",
        }

    # Normalize: prompt v4 uses "abnormal" key; v1-v3 use "fracture".
    # Always populate both for downstream compatibility.
    if prompt_version == 4:
        parsed.setdefault("fracture", parsed.get("abnormal"))
    else:
        parsed.setdefault("abnormal", parsed.get("fracture"))

    return parsed


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="AXIS — Automated X-Ray Identification for the Skeleton",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", "-i", help="Path to single X-ray (DICOM, PNG, or JPEG)")
    group.add_argument("--dicom-dir", "-d", help="Path to directory of DICOMs/PNGs (processes all)")
    parser.add_argument("--backend", default=DEFAULT_BACKEND, choices=["mlx", "transformers"],
                        help=f"Inference backend (default: {DEFAULT_BACKEND})")
    parser.add_argument("--model", "-m", default=None,
                        help="Model path or HF ID (overrides default for chosen backend)")
    parser.add_argument("--prompt-version", "-p", type=int, default=1, choices=[1, 2, 3, 4],
                        help="Prompt version: 1=fracture binary, 2=fracture detailed, 3=fracture minimal, 4=pathology binary")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show raw model response")
    parser.add_argument("--json", action="store_true", help="Output raw JSON only (for piping)")
    args = parser.parse_args()

    # Set backend
    set_backend(args.backend, args.model)
    model_display = _resolve_model(args.backend, args.model)
    print(f"[i] AXIS | Backend: {args.backend}, Model: {model_display}\n")

    # Collect image paths
    if args.dicom_dir:
        images = find_images(args.dicom_dir)
        if not images:
            print(f"[!] No image files found in: {args.dicom_dir}")
            sys.exit(1)
        print(f"[i] Found {len(images)} image file(s) in {args.dicom_dir}\n")
    else:
        images = [args.image]

    all_results = []
    for img_path in images:
        try:
            result = query_model(img_path, args.prompt_version)
        except Exception as e:
            result = {"error": str(e), "_meta": {"image": img_path}}

        all_results.append(result)

        if args.json:
            continue

        if "error" in result:
            print(f"\n[ERROR] {result['error']}")
            continue

        # Adapt display based on prompt version
        if args.prompt_version == 4:
            abnormal = result.get("abnormal")
            call_str = "ABNORMAL" if abnormal else ("NORMAL" if abnormal is not None else "UNCERTAIN")
            label_name = "PATHOLOGY"
        else:
            abnormal = result.get("fracture")
            call_str = "YES" if abnormal else ("NO" if abnormal is not None else "UNCERTAIN")
            label_name = "FRACTURE"

        confidence = result.get("confidence")
        findings = result.get("findings", "N/A")
        meta = result.get("_meta", {})

        conf_str = f"{confidence:.0%}" if isinstance(confidence, (int, float)) else "N/A"

        label = Path(img_path).name
        print(f"{'='*50}")
        print(f"  FILE:        {label}")
        print(f"  {label_name}:  {' ' * (8 - len(label_name))}{call_str}")
        print(f"  CONFIDENCE:  {conf_str}")
        print(f"  FINDINGS:    {findings}")
        if result.get("location"):
            print(f"  LOCATION:    {result['location']}")
        if result.get("category"):
            print(f"  CATEGORY:    {result['category']}")
        if result.get("fracture_type"):
            print(f"  TYPE:        {result['fracture_type']}")
        if result.get("additional_findings"):
            print(f"  OTHER:       {result['additional_findings']}")
        print(f"  INFERENCE:   {meta.get('inference_time_s', '?')}s")
        print(f"  BACKEND:     {meta.get('backend', '?')}")
        print(f"{'='*50}\n")

        if args.verbose:
            print(f"[RAW RESPONSE]\n{meta.get('raw_response', 'N/A')}\n")

    if args.json:
        output = all_results[0] if len(all_results) == 1 else all_results
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
