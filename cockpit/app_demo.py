"""
Cockpit (Demo Mode) — Web UI with demo banner for synthetic inference.

Serves the same dashboard as the regular cockpit but injects a sticky
"Demo Mode" banner at the top and patches inference to use synthetic data.

Usage:
    python -m cockpit.app_demo
    python -m cockpit.app_demo --run
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Patch inference BEFORE pipeline imports
from tools.evaluator_impl_demo import patch_tool_functions  # noqa: E402
patch_tool_functions()

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

from cockpit.app import app  # noqa: E402
from cockpit import events  # noqa: E402

COCKPIT_DIR = Path(__file__).resolve().parent

DEMO_BANNER_HTML = """\
<div id="demoBanner" style="
  position: fixed; top: 0; left: 0; right: 0; z-index: 9999;
  background: linear-gradient(90deg, #ff9f0a, #ffb340);
  color: #1d1d1f; text-align: center;
  padding: 8px 16px; font-family: -apple-system, BlinkMacSystemFont, sans-serif;
  font-size: 13px; font-weight: 700; letter-spacing: 0.5px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.15);
">
  DEMO MODE &mdash; Local Inference Simulated
  <span style="font-weight:400; margin-left: 12px; opacity: 0.8;">
    Predictions are synthetic. All other pipeline steps run live.
  </span>
</div>
<style>
  body { padding-top: 38px !important; }
  .sidebar { top: 38px !important; height: calc(100vh - 38px) !important; }
</style>
"""


@app.get("/", response_class=HTMLResponse)
async def index_demo():
    """Serve the cockpit HTML with an injected demo banner."""
    html = (COCKPIT_DIR / "index.html").read_text()
    # Inject banner right after <body>
    html = html.replace("<body>", "<body>\n" + DEMO_BANNER_HTML, 1)
    return HTMLResponse(html)


def main():
    parser = argparse.ArgumentParser(description="AXIS Agentic Cockpit (Demo Mode)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8322)
    parser.add_argument("--run", action="store_true", help="Also launch the pipeline interactively")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    events.enable()

    # Load .env
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                key, val = line.split("=", 1)
                key, val = key.strip(), val.strip()
                if key and not os.environ.get(key):
                    os.environ[key] = val

    print()
    print("  \033[43m\033[30m DEMO MODE \033[0m")
    print(f"  Cockpit: http://{args.host}:{args.port}")
    print(f"  Inference is synthetic — no server needed.")
    print(f"  Open in browser to start an evaluation run.\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
