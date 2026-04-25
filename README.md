# AXIS Agentic

An agentic evaluation framework for radiology AI models. Two LLM-driven agents (Evaluator + Analyst) automate the full evaluation loop: run inference, validate predictions, compute metrics, search published literature, and generate a comparative report.

> **Status:** PR1 (spine) refactor in progress. Provider-agnostic Engine + Adapter abstractions are in place; this README will be expanded with full results, model card, and a 60-second demo path in follow-up PRs.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Coordinator                          │
│         1 model × N datasets evaluation loop            │
│                                                         │
│  ┌─────────────┐         ┌─────────────┐               │
│  │  Evaluator   │ ──────▶ │   Analyst   │               │
│  │  (Agent 1)   │ metrics │  (Agent 2)  │               │
│  │              │ matrix  │             │               │
│  │ run_inference│         │ compare_    │               │
│  │ validate_    │         │   baselines │               │
│  │   results    │         │ search_     │               │
│  │ compute_     │         │   literature│               │
│  │   metrics    │         │ generate_   │               │
│  │              │         │   figures   │               │
│  │              │         │ write_report│               │
│  └──────┬───────┘         └──────┬──────┘               │
│         │                        │                      │
│  ┌──────▼───────┐         ┌──────▼──────┐               │
│  │  Inference   │         │  Anthropic  │               │
│  │  Adapter     │         │  Web Search │               │
│  │  (HTTP/MLX/  │         │  (PubMed)   │               │
│  │   transformers│         │             │               │
│  │   /Anthropic)│         │             │               │
│  └──────────────┘         └─────────────┘               │
│                                                         │
│  Engine: Anthropic Claude (default) — pluggable in PR2   │
│  (OpenAI / Nemotron / Ollama via OpenAI-compatible API) │
└─────────────────────────────────────────────────────────┘
```

**Key design — two abstractions:**

- **`Engine`** wraps the LLM provider that drives the agent loop. The agents are written against a uniform `send_user_message` / `send_tool_results` interface; switching from Anthropic to OpenAI to a local Ollama model does not change agent code.
- **`InferenceAdapter`** wraps the model under test. `predict(image_path) -> Prediction` is the only contract; built-in adapters cover an HTTP endpoint (today), with MLX, `transformers`, and Anthropic SDK adapters landing in PR2.

The agents are task-agnostic; the tools are task-specific. Today's tools evaluate a binary MSK classifier on MURA. Swap the tools to evaluate any model on any task — the orchestration layer does not change.

## Setup

### Prerequisites

- Python 3.11+
- An Anthropic API key for the default Engine ([console.anthropic.com](https://console.anthropic.com))
- For full inference: an Apple Silicon Mac running the MedGemma server (or any HTTP endpoint exposing `/predict`)
- For demo mode: nothing else — predictions are synthetic

### Install

```bash
git clone https://github.com/bschwaiger/axis-agentic.git
cd axis-agentic
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Environment

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=sk-ant-...
# Optional — override the default model (defaults to claude-sonnet-4-6)
# ANTHROPIC_MODEL=claude-sonnet-4-6
```

### MURA data

Image data is gitignored (104 MB). Regenerate the included subsets from the full MURA dataset:

```bash
python scripts/generate_subsets.py
```

Or download MURA from [Stanford AIMI](https://aimi.stanford.edu/datasets/mura-msk-xrays) and place studies under `data/`.

### Inference server (full mode only)

```bash
export MEDGEMMA_MODEL_PATH=/path/to/axis-mura-v1-4bit
python server/inference_server.py
curl http://localhost:8321/health
```

## Usage

### Demo mode (no model, no inference server)

The fastest path to seeing the pipeline run end-to-end. `run_inference` is replaced with a synthetic generator (~50 ms / study). The Engine still drives the agents live, including literature search.

```bash
# Cockpit (web UI)
python -m cockpit.app_demo

# Or terminal
python -m orchestrator.coordinator_demo
```

### Full mode (real inference)

```bash
# Terminal 1: inference server
python server/inference_server.py

# Terminal 2: cockpit
python -m cockpit.app

# Or terminal-only
python -m orchestrator.coordinator
```

### Verbose

```bash
python -m orchestrator.coordinator --verbose
```

## Directory layout

```
axis-agentic/
├── orchestrator/       # Pipeline + agents
│   ├── coordinator.py       # Outer loop: 1×N evaluation + checkpoints
│   ├── coordinator_demo.py  # Same, with synthetic inference patched in
│   ├── evaluator_agent.py   # Agent 1: inference → validate → metrics
│   ├── analyst_agent.py     # Agent 2: compare → literature → figures → report
│   ├── formatting.py        # Provider-agnostic terminal output
│   └── engine/              # LLM provider abstraction
│       ├── base.py          #   Engine ABC + ToolCall/ToolResult/TurnResponse
│       └── anthropic.py     #   AnthropicEngine
├── inference/          # Model-under-test abstraction
│   ├── base.py              #   InferenceAdapter ABC + Prediction
│   └── http.py              #   HTTPAdapter (default; localhost:8321)
├── tools/              # Tool definitions (JSON) + implementations (Python)
│   ├── evaluator_tools.json / evaluator_impl.py / evaluator_impl_demo.py
│   └── analyst_tools.json   / analyst_impl.py
├── cockpit/            # Web UI (FastAPI + SSE)
├── server/             # FastAPI inference server (MedGemma via MLX)
├── scripts/            # Standalone pipeline scripts
├── data/               # MURA subsets (manifests tracked, images gitignored)
├── docs/               # Task file templates + architecture docs
└── results/            # Pipeline outputs (gitignored)
```

## License

Apache License 2.0 — see [LICENSE](LICENSE). MURA data is subject to [Stanford AIMI terms](https://aimi.stanford.edu/datasets/mura-msk-xrays); the AXIS-MURA-v1 model weights are not included in this repository (manuscript pending publication).
