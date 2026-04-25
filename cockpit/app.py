"""
Cockpit — Lightweight web UI for the AXIS Agentic pipeline.

Serves a single-page dashboard at localhost:8322 with:
- SSE stream of pipeline events
- Approval endpoints for checkpoints
- Pipeline control (start/stop/pause/resume/restart)

Usage:
    python -m cockpit.app                  # start cockpit server
    python -m cockpit.app --verbose        # with verbose agent output
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cockpit import events  # noqa: E402

app = FastAPI(title="AXIS Agentic Cockpit")

COCKPIT_DIR = Path(__file__).resolve().parent


# ------------------------------------------------------------------
# Static
# ------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = COCKPIT_DIR / "index.html"
    return HTMLResponse(html_path.read_text())


# ------------------------------------------------------------------
# SSE stream
# ------------------------------------------------------------------

@app.get("/events")
async def event_stream():
    q = events.subscribe()

    def generate():
        try:
            while True:
                try:
                    event = q.get(timeout=30.0)
                    yield events.format_sse(event)
                except Exception:
                    yield ": keepalive\n\n"
        finally:
            events.unsubscribe(q)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ------------------------------------------------------------------
# Approval endpoints
# ------------------------------------------------------------------

@app.post("/approve/matrix")
async def approve_matrix(request: Request):
    body = await request.json()
    action = body.get("action", "approve")
    events.post_approval("matrix_checkpoint", {"action": action})
    return JSONResponse({"status": "ok"})


@app.post("/approve/queries")
async def approve_queries(request: Request):
    body = await request.json()
    events.post_approval("query_checkpoint", body)
    return JSONResponse({"status": "ok"})


@app.post("/start")
async def start_pipeline(request: Request):
    """Start the pipeline from the UI."""
    body = await request.json()
    model_idx = body.get("model", 0)
    dataset_indices = body.get("datasets", [0])
    verbose = body.get("verbose", False)

    from orchestrator.coordinator import AVAILABLE_MODELS, AVAILABLE_DATASETS, run_pipeline

    if not (0 <= model_idx < len(AVAILABLE_MODELS)):
        return JSONResponse({"error": "Invalid model index"}, status_code=400)
    for idx in dataset_indices:
        if not (0 <= idx < len(AVAILABLE_DATASETS)):
            return JSONResponse({"error": f"Invalid dataset index: {idx}"}, status_code=400)

    model = AVAILABLE_MODELS[model_idx]
    datasets = [AVAILABLE_DATASETS[idx].copy() for idx in dataset_indices]

    config = {
        "model_name": model["id"],
        "model_entry": model,
        "datasets": datasets,
        "baseline_values": model.get("baseline_values", {}),
        "baseline_source": model.get("baseline_source", ""),
    }

    events.reset_controls()

    def _run():
        try:
            run_pipeline(config, verbose=verbose)
        except Exception as e:
            events.emit("pipeline_error", error=str(e))

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    return JSONResponse({"status": "started", "model": model["name"], "datasets": [d["name"] for d in datasets]})


@app.post("/control/stop")
async def stop_pipeline():
    events.request_stop()
    events.emit("pipeline_stopped")
    return JSONResponse({"status": "stop_requested"})


@app.post("/control/pause")
async def pause_pipeline():
    events.request_pause()
    events.emit("pipeline_paused")
    return JSONResponse({"status": "paused"})


@app.post("/control/resume")
async def resume_pipeline():
    events.request_resume()
    events.emit("pipeline_resumed")
    return JSONResponse({"status": "resumed"})


@app.post("/control/restart")
async def restart_pipeline(request: Request):
    events.request_stop()
    import asyncio
    await asyncio.sleep(0.5)
    events.reset_controls()
    return JSONResponse({"status": "ready_to_restart"})


@app.get("/config")
async def get_config():
    from orchestrator.coordinator import AVAILABLE_MODELS, AVAILABLE_DATASETS
    return JSONResponse({
        "models": [{"name": m["name"], "description": m["description"], "id": m["id"]} for m in AVAILABLE_MODELS],
        "datasets": [{"name": d["name"], "description": d["description"]} for d in AVAILABLE_DATASETS],
    })


# ------------------------------------------------------------------
# CLI helpers
# ------------------------------------------------------------------

def _kill_port(port: int):
    """Kill any process listening on the given port so restarts always work."""
    import subprocess
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True, timeout=5,
        )
        pids = result.stdout.strip().split()
        for pid in pids:
            if pid:
                subprocess.run(["kill", "-9", pid], timeout=5)
        if pids:
            import time
            time.sleep(0.5)
            print(f"  Killed previous process on port {port}.")
    except Exception:
        pass


def _load_env():
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                key, val = line.split("=", 1)
                key, val = key.strip(), val.strip()
                if key and not os.environ.get(key):
                    os.environ[key] = val


def main():
    parser = argparse.ArgumentParser(description="AXIS Agentic Cockpit")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8322)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _kill_port(args.port)
    events.enable()
    _load_env()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("[!] ANTHROPIC_API_KEY not set in .env — the default engine will fail at runtime.")
        print("    Add it to .env: ANTHROPIC_API_KEY=sk-ant-...")

    print(f"  Cockpit: http://{args.host}:{args.port}")
    print(f"  Open in browser to start an evaluation run.\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
