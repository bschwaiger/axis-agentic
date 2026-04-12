# AXIS Agentic — State Compression

## Goal
Agent-orchestrated MSK radiograph classification: inference, validation, metrics, figures, reporting — with human-in-the-loop at defined checkpoints.

## Architecture
OpenAI Chat Completions API with function calling orchestrates four tools (`run_inference`, `validate_results`, `compute_metrics`, `generate_figures`) that call a local FastAPI server running AXIS-MURA-v1 (LoRA fine-tuned MedGemma 1.5 4B, MLX 4-bit quantized) on MacBook Air M4. No framework (no LangGraph, no CrewAI) — bare function calling for maximum debuggability. Data: MURA subsets stored locally with manifest CSVs.

## Current Status
- Folder structure: created
- MURA subsets: generated (eval-020, eval-100, eval-300, train-050; all balanced)
- Pipeline scripts: copied from ~/Projects/axis/, paths updated
- Inference server: FastAPI on localhost:8321 (POST /predict, GET /health, GET /model-info)
- Tool definitions: 4 tools in OpenAI function calling JSON format
- Agent orchestration: not yet built (hackathon day task)

## File Inventory

```
~/Projects/axis-agentic/
├── data/
│   ├── eval-020/          20 studies (10 normal, 10 abnormal) + manifest.csv
│   ├── eval-100/          100 studies (50/50) + manifest.csv
│   ├── eval-300/          300 studies (150/150) + manifest.csv
│   └── train-050/         50 studies (25/25) + manifest.csv
├── scripts/
│   ├── axis_detector.py   Core inference engine (MLX + Transformers backends)
│   ├── batch_eval.py      Batch evaluation runner with metrics
│   ├── compare_runs.py    Cross-run statistical comparison
│   ├── validate_image_level.py  Image→study aggregation validation
│   └── generate_subsets.py      Subset generation utility
├── server/
│   └── inference_server.py    FastAPI server wrapping axis_detector
├── tools/
│   └── tool_definitions.json  OpenAI function calling schemas (4 tools)
├── results/               (empty, populated by agent runs)
├── docs/
│   ├── STATE_COMPRESSION.md   This file
│   └── TASK_FILE_TEMPLATE.md  Template for evaluation run definitions
└── README.md
```

## Known Issues
- `suppress_tokens`: Required for post-merge MedGemma inference to block `<unused*>` thinking tokens. Currently NOT implemented in `axis_detector.py` MLX backend. The `mlx_vlm.generate()` API may or may not support this parameter. Must verify before first inference run.
- Manifest absolute paths: MURA manifests historically stored absolute paths. The agentic workspace uses relative paths in manifest CSVs. If debugging eval failures, check path resolution first.
- Model location: Set `MEDGEMMA_MODEL_PATH` env var to point to the AXIS-MURA-v1 4-bit model directory. Must be available locally before inference.

## Next Steps
1. Verify inference server starts and returns valid predictions
2. Build agent orchestration script (OpenAI function calling loop)
3. Run eval-020 end-to-end as smoke test
4. Demo with eval-100
5. Full eval with eval-300
