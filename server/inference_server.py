#!/usr/bin/env python3
"""
AXIS Agentic — Local Inference Server
FastAPI wrapper around axis_detector for OpenAI function calling agent.

Start:  python server/inference_server.py
Health: curl http://localhost:8321/health
Test:   curl -X POST http://localhost:8321/predict \
            -H "Content-Type: application/json" \
            -d '{"image_path": "data/eval-020/XR_WRIST/patient11185/study1_positive/image1.png"}'
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add scripts/ to path so we can import axis_detector
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from axis_detector import query_model, set_backend, _resolve_model, load_image  # noqa: E402

# ============================================================
# CONFIG
# ============================================================

# Load .env if present (before reading env vars)
_env_path = _PROJECT_ROOT / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _v = _line.split("=", 1)
            _k, _v = _k.strip(), _v.strip()
            if _k and not os.environ.get(_k):
                os.environ[_k] = _v

HOST = "0.0.0.0"
PORT = 8321
BACKEND = "mlx"
MODEL = os.environ.get("MEDGEMMA_MODEL_PATH", "models/axis-mura-v1-4bit")
PROMPT_VERSION = 4  # pathology binary (abnormal/normal)

# ============================================================
# APP
# ============================================================

app = FastAPI(title="AXIS Agentic Inference Server", version="0.1.0")


class PredictRequest(BaseModel):
    image_path: str


class PredictResponse(BaseModel):
    prediction: int  # 0=normal, 1=abnormal
    confidence: float
    findings: str | None = None
    category: str | None = None
    location: str | None = None
    inference_time_s: float | None = None


@app.on_event("startup")
async def startup():
    """Pre-load model on server startup."""
    set_backend(BACKEND, MODEL)
    print(f"[i] AXIS Inference Server starting...")
    print(f"    Backend: {BACKEND}")
    print(f"    Model: {MODEL}")
    print(f"    Prompt version: {PROMPT_VERSION}")
    print(f"    Port: {PORT}")
    # Warm up model by loading it (first inference triggers actual load)
    print(f"[i] Model will be loaded on first inference request.")


@app.get("/health")
async def health():
    return {"status": "ok", "backend": BACKEND, "port": PORT}


@app.get("/model-info")
async def model_info():
    return {
        "model": MODEL,
        "backend": BACKEND,
        "quantization": "4-bit",
        "device": "Apple Silicon (M4) via MLX",
        "prompt_version": PROMPT_VERSION,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    image_path = req.image_path

    # Resolve relative paths against project root
    p = Path(image_path)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {p}")

    try:
        result = query_model(str(p), prompt_version=PROMPT_VERSION)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    # Map boolean prediction to int
    abnormal = result.get("abnormal")
    if abnormal is None:
        abnormal = result.get("fracture")
    prediction = 1 if abnormal else 0

    confidence = result.get("confidence")
    if confidence is None:
        confidence = 0.0

    return PredictResponse(
        prediction=prediction,
        confidence=float(confidence),
        findings=result.get("findings"),
        category=result.get("category"),
        location=result.get("location"),
        inference_time_s=result.get("_meta", {}).get("inference_time_s"),
    )


if __name__ == "__main__":
    # Kill any existing process on the port so restarts always work
    import subprocess, time as _time
    result = subprocess.run(f"lsof -ti:{PORT}", shell=True, capture_output=True, text=True)
    if result.stdout.strip():
        subprocess.run(f"lsof -ti:{PORT} | xargs kill -9", shell=True)
        _time.sleep(1)  # wait for port to be released

    uvicorn.run("inference_server:app", host=HOST, port=PORT, reload=False)
