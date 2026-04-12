# AXIS Agentic

An agentic evaluation pipeline for radiology AI models. Two LLM-orchestrated agents (Evaluator + Analyst) automate the full evaluation loop: run inference, validate predictions, compute metrics, search published literature, and generate a comparative report — all driven by Nemotron 3 Super 120B via NVIDIA NIM function calling.

Built for the April 2026 hackathon. Dual-track: NVIDIA (Nemotron orchestrator) + Tavily (PubMed literature search).

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
│  │  Inference   │         │   Tavily    │               │
│  │  Server      │         │   (PubMed)  │               │
│  │  :8321       │         │             │               │
│  └──────────────┘         └─────────────┘               │
│                                                         │
│  Orchestrator LLM: Nemotron 3 Super 120B (NVIDIA NIM)   │
│  Function calling: OpenAI-compatible tool_calls format   │
└─────────────────────────────────────────────────────────┘
```

**Key design:** The agents are task-agnostic; the tools are task-specific. Today's tools evaluate a binary MSK classifier on MURA. Swap the tools to evaluate any model on any task — the orchestration layer does not change.

## Setup

### Prerequisites

- Python 3.11+
- Apple Silicon Mac (for MLX inference) or modify the inference server for your backend
- NVIDIA NIM API key ([build.nvidia.com](https://build.nvidia.com))
- Tavily API key ([tavily.com](https://tavily.com))

### Install

```bash
git clone https://github.com/bschwaiger/axis-agentic.git
cd axis-agentic
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```
NVIDIA_API_KEY=nvapi-...
TAVILY_API_KEY=tvly-...
```

### MURA Data

Image data is gitignored (104MB). Regenerate subsets from the full MURA dataset:

```bash
python scripts/generate_subsets.py
```

Or download MURA from [Stanford AIMI](https://aimi.stanford.edu/datasets/mura-msk-xrays) and place studies under `data/`.

### Inference Server

```bash
# Set model path
export MEDGEMMA_MODEL_PATH=/path/to/axis-mura-v1-4bit

# Start server
python server/inference_server.py

# Verify
curl http://localhost:8321/health
```

## Usage

### Demo Run (1 model × 2 datasets)

```bash
# Start inference server in one terminal
python server/inference_server.py

# Run the pipeline in another
python -m orchestrator.coordinator --task docs/task_demo.yaml
```

### Single Dataset

```bash
python -m orchestrator.coordinator --task docs/task_eval020.yaml
```

### Verbose Mode (debug)

```bash
python -m orchestrator.coordinator --task docs/task_demo.yaml --verbose
```

## Sample Output

```
╔══════════════════════════════════════════════════════════════╗
║              AXIS AGENTIC — Evaluation Pipeline             ║
╚══════════════════════════════════════════════════════════════╝
  Model:    axis-mura-v1-4bit
  Datasets: eval-020, eval-100 (2 total)

▶ Phase 1.1: Evaluator — eval-020
  → Nemotron: call run_inference(dataset="data/eval-020")
  ← run_inference: 20 predictions written, 0 errors
  → Nemotron: call validate_results(predictions="eval-020/predictions.csv")
  ← validate_results: PASS — 20 predictions validated, 0 nulls
  → Nemotron: call compute_metrics(predictions="eval-020/predictions.csv")
  ← compute_metrics: n=20, MCC=0.734, Sens=0.7, Spec=1.0
  💬 Nemotron: "All predictions validated successfully..."
┌ ──────────────────────────────────────────────────────────── ┐
│ Evaluator complete (n=20): MCC=0.734, Sens=0.700, Spec=1.000  │
└ ──────────────────────────────────────────────────────────── ┘

▶ Phase 1.2: Evaluator — eval-100
  → Nemotron: call run_inference(dataset="data/eval-100")
  ← run_inference: 100 predictions written, 0 errors
  → Nemotron: call validate_results(predictions="eval-100/predictions.csv")
  ← validate_results: PASS — 100 predictions validated, 0 nulls
  → Nemotron: call compute_metrics(predictions="eval-100/predictions.csv")
  ← compute_metrics: n=100, MCC=0.601, Sens=0.78, Spec=0.82
  💬 Nemotron: "Evaluation complete. 100 studies processed..."
┌ ──────────────────────────────────────────────────────────────── ┐
│ Evaluator complete (n=100): MCC=0.601, Sens=0.780, Spec=0.820   │
└ ──────────────────────────────────────────────────────────────── ┘

⏸  CHECKPOINT: Review comparison matrix

  Metric             eval-020    eval-100
  ─────────────────────────────────────────
  MCC                  0.7338      0.6005
  SENSITIVITY          0.7000      0.7800
  SPECIFICITY          1.0000      0.8200

▶ Phase 2: Analyst — Comparative Analysis
  → Nemotron: call compare_baselines(matrix="comparative/comparison_matrix.json")
  ← compare_baselines: FLAGGED — 4 flag(s) across 2 datasets

⏸  CHECKPOINT: Review proposed PubMed queries below
    1. Effect of training set size on sensitivity in MSK fracture detection
    2. Baseline performance metrics for AI models on the MURA dataset
    3. Impact of small sample size on evaluation metrics in medical imaging

  → Nemotron: call search_literature(3 queries)
  ← search_literature: 9 papers found across 3 queries
  → Nemotron: call generate_figures(output="results/comparative")
  ← generate_figures: 3 figures generated
  → Nemotron: call write_report(output="comparative/report.md")
  ← write_report: report written (954 words)

╔══════════════════════════════════════════════════════════════╗
║                     PIPELINE COMPLETE                       ║
╚══════════════════════════════════════════════════════════════╝
  Report: results/comparative/report.md
  Total: 19 output files
```

## Directory Layout

```
axis-agentic/
├── orchestrator/       # Coordinator + Agent 1 + Agent 2
│   ├── coordinator.py  # Outer loop: 1×N evaluation + checkpoints
│   ├── evaluator_agent.py  # Agent 1: inference → validate → metrics
│   ├── analyst_agent.py    # Agent 2: compare → literature → figures → report
│   └── formatting.py       # Demo-friendly terminal output
├── tools/              # Tool definitions (JSON) + implementations (Python)
│   ├── evaluator_tools.json / evaluator_impl.py
│   └── analyst_tools.json   / analyst_impl.py
├── server/             # FastAPI inference server (MedGemma via MLX)
├── scripts/            # Standalone pipeline scripts (axis_detector, batch_eval)
├── data/               # MURA subsets (manifests tracked, images gitignored)
├── docs/               # Task file templates + architecture docs
└── results/            # Pipeline outputs (gitignored, regenerated per run)
```

## License

Research use only. MURA data subject to [Stanford AIMI terms](https://aimi.stanford.edu/datasets/mura-msk-xrays). Model weights not included.
