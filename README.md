# AXIS Agentic

**Agentic evaluation framework for medical imaging AI.** Two LLM-driven agents (Evaluator + Analyst) automate the full evaluation loop — run inference, validate, compute metrics, search PubMed for context, generate figures, write a comparative report — across whatever model you point them at.

Built around two abstractions: an **Engine** (the LLM driving the agents) and an **InferenceAdapter** (the model under test). Both are swappable. Plug in any vision-language model, a CNN, an HF Hub model, or a frontier API; orchestrate with Claude, GPT-4o, Nemotron, or a local Qwen via Ollama. Same agent loop, no code changes.

```
┌─────────────────────────────────────────────────────────┐
│                    Coordinator                          │
│         1 model × N datasets evaluation loop            │
│                                                         │
│  ┌─────────────┐         ┌─────────────┐               │
│  │  Evaluator  │ ──────▶ │   Analyst   │               │
│  │  (Agent 1)  │ matrix  │  (Agent 2)  │               │
│  │             │         │             │               │
│  │ run_        │         │ compare_    │               │
│  │  inference  │         │  baselines  │               │
│  │ validate_   │         │ search_     │               │
│  │  results    │         │  literature │               │
│  │ compute_    │         │ generate_   │               │
│  │  metrics    │         │  figures    │               │
│  │             │         │ write_      │               │
│  │             │         │  report     │               │
│  └──────┬──────┘         └──────┬──────┘               │
│         │                       │                       │
│   InferenceAdapter         Engine web search           │
│   (HTTP / MLX /            (Anthropic web_search       │
│    transformers /           on PubMed)                 │
│    Anthropic vision)                                    │
└─────────────────────────────────────────────────────────┘
```

A specific trained model and its evaluation results live in a [separate repo](https://github.com/bschwaiger/axis-mura). This one is the framework only — task-agnostic.

## Quick start (60 seconds, no model required)

Demo mode replaces inference with a synthetic generator (~50 ms/image) so you can see the full agentic flow without an inference server, GPU, or model weights.

```bash
git clone https://github.com/bschwaiger/axis-agentic.git
cd axis-agentic
python -m venv venv && source venv/bin/activate
pip install -e .

# One key in .env (Anthropic by default; swap to Ollama in the wizard if you prefer)
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env

axis-agentic cockpit --demo
```

Open <http://127.0.0.1:8322>. Pick **Use defaults**, click **Start**. Watch the Evaluator → Analyst loop run end-to-end: inference, validation, metrics, baseline comparison, PubMed search, figures, report.

For real inference, run `axis-agentic init`, register your model and dataset (browse-button file picker), then `axis-agentic cockpit` — the framework auto-starts Ollama, the bundled inference server, or whatever else is needed.

## Architecture

Two abstractions; both are swappable in the cockpit wizard or via the profile YAML at `~/.config/axis-agentic/profile.yaml`.

### Engine — the LLM driving the agents

| Engine | Providers it covers |
|---|---|
| `AnthropicEngine` | Claude (Sonnet, Opus, …) |
| `OpenAICompatEngine` | OpenAI, **Ollama** (local Qwen / Llama / …), NVIDIA NIM (Nemotron), LM Studio, Together, Groq, anything else with a chat-completions API |

Tool-spec translation between Anthropic-native and OpenAI function-calling format happens inside the engine; the agent code is provider-agnostic.

### InferenceAdapter — the model under test

| Adapter | What it wraps |
|---|---|
| `HTTPAdapter` | Any HTTP endpoint exposing `POST /predict` with `{"image_path": ...}` and returning `{prediction, confidence, findings?}` |
| `MLXAdapter` | In-process Apple Silicon inference via `mlx-vlm` |
| `TransformersAdapter` | HuggingFace Hub ID or local checkpoint via `AutoModelForImageTextToText` |
| `AnthropicAdapter` | Claude vision as the model under test (for benchmarking your model against a frontier VLM) |

Each model entry in the profile registry can declare its own adapter, so a single cockpit session can mix `model-a@MLX`, `model-b@HTTP`, and `claude-opus-4@AnthropicAdapter` in one comparative report.

### Auto-start

The agent layer brings up local services it needs. Pick Ollama in the wizard with the daemon down, click Run, and the framework spawns `ollama serve`, polls until ready, and pulls the configured model if missing (with byte-level progress in the cockpit feed) before starting the agent loop.

Cloud providers (Anthropic, OpenAI, NIM, …) are no-ops — they're someone else's problem. For HTTP adapters pointing at a server you run yourself, the framework fails fast with a clear "start it manually at `<url>`" hint rather than trying to spawn something it doesn't know about.

## Configurable: stats, baselines, datasets

The agents are task-agnostic; the tools are task-specific. Today's tools evaluate a binary classifier image-by-image, but:

- The `Engine` abstraction means swapping the LLM (Anthropic ↔ Ollama-Qwen ↔ GPT-4o) is a config change.
- The `InferenceAdapter` abstraction means swapping the model under test is a config change.
- **Dataset auto-detect** (`axis-agentic detect <path>`): if you have a folder of images with a label CSV in some shape, the agent inspects it (file tree, sample images, candidate manifests) and proposes a schema you confirm before evaluation begins. Cached to `<dataset>/.axis_schema.json`.
- A configurable per-model **baseline** in the profile drives the Analyst's `compare_baselines` flagging.

## Usage

```bash
# Interactive setup (engine, adapter, models, datasets) — writes
# ~/.config/axis-agentic/profile.yaml. Or skip and use defaults.
axis-agentic init

# Browser UI
axis-agentic cockpit          # full mode (real inference)
axis-agentic cockpit --demo   # synthetic inference, no model required

# Headless from the profile
axis-agentic eval

# Inspect current profile + key state
axis-agentic status

# Standalone dataset auto-detect
axis-agentic detect ~/path/to/dataset
```

## Directory layout

```
axis-agentic/
├── axis_agentic/        # CLI + profile
│   ├── cli.py           #   subcommand dispatcher
│   ├── init_wizard.py   #   interactive terminal wizard (cockpit has its own)
│   ├── profile.py       #   load/save ~/.config/axis-agentic/profile.yaml
│   └── services.py      #   auto-start Ollama / inference server
├── orchestrator/        # Agents + coordinator
│   ├── coordinator.py   #   outer loop, 1×N evaluation + checkpoints
│   ├── evaluator_agent.py
│   ├── analyst_agent.py
│   ├── formatting.py    #   provider-agnostic terminal output
│   └── engine/
│       ├── base.py      #     Engine ABC + ToolCall/ToolResult/TurnResponse
│       ├── anthropic.py
│       └── openai_compat.py
├── inference/           # Model-under-test layer
│   ├── base.py          #     InferenceAdapter ABC + Prediction
│   ├── http.py
│   ├── mlx.py
│   ├── transformers.py
│   └── anthropic.py
├── tools/               # Agent tools (JSON specs + Python impls)
│   ├── evaluator_tools.json / evaluator_impl.py / evaluator_impl_demo.py
│   ├── analyst_tools.json   / analyst_impl.py
│   └── dataset_detect.py
├── cockpit/             # FastAPI + SSE web UI
│   ├── app.py / app_demo.py
│   ├── events.py
│   └── index.html       #     welcome → wizard → setup → run
└── data/                # Bundled tiny demo dataset; bring your own elsewhere
```

## License

Apache License 2.0 — see [LICENSE](LICENSE). Bring your own data and models; this repo doesn't bundle either.
