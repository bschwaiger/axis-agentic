# AXIS Agentic — Task File Template

A Task File defines a single evaluation run. The agent reads this file and executes the pipeline steps sequentially.

## Template

```yaml
# Task File: [descriptive name]
model_name: "axis-mura-v1-4bit"
dataset_path: "data/eval-020"
manifest_path: "data/eval-020/manifest.csv"

metrics:
  - mcc
  - sensitivity
  - specificity
  - precision
  - npv
  - f1
  - accuracy
  - confusion_matrix

baseline_values:
  mcc: 0.637          # AXIS-MURA-v1 on full MURA valid set (n=896)
  sensitivity: 0.743
  specificity: 0.920

output_dir: "results/eval-020_run1"

checkpoints:
  - pre_flight        # Validate prerequisites before inference
  - post_metrics      # Review metrics before generating figures
  - pre_report        # Approve draft report before finalization

abort_criteria:
  mcc_below: 0.3              # Abort if MCC drops below this threshold
  parse_failure_rate: 0.1     # Abort if >10% of predictions fail to parse
  empty_predictions: true     # Abort if any study returns null prediction
```

## Field Descriptions

| Field | Type | Description |
|---|---|---|
| `model_name` | string | Model identifier for logging and output naming |
| `dataset_path` | string | Relative path to dataset directory containing study folders |
| `manifest_path` | string | Relative path to manifest CSV (study_path, label) |
| `metrics` | list[string] | Metrics to compute after inference |
| `baseline_values` | dict | Reference values from prior runs for comparison |
| `output_dir` | string | Where results, metrics JSON, and figures are saved |
| `checkpoints` | list[string] | Steps where agent pauses for human review |
| `abort_criteria` | dict | Thresholds that trigger automatic abort with diagnostics |

## Checkpoint Behavior

- **pre_flight**: Agent validates model exists, dataset path resolves, venv is active, manifest is readable. Reports and waits for "proceed" before starting inference.
- **post_metrics**: Agent presents metrics and baseline comparison. Human decides whether to generate figures or review raw data first.
- **pre_report**: Agent presents draft Markdown report. Human approves or requests changes.

## Example Task Files

### Debugging (n=20)
```yaml
model_name: "axis-mura-v1-4bit"
dataset_path: "data/eval-020"
manifest_path: "data/eval-020/manifest.csv"
metrics: [mcc, sensitivity, specificity, confusion_matrix]
baseline_values: {}
output_dir: "results/debug_eval020"
checkpoints: []
abort_criteria:
  parse_failure_rate: 0.2
```

### Demo (n=100)
```yaml
model_name: "axis-mura-v1-4bit"
dataset_path: "data/eval-100"
manifest_path: "data/eval-100/manifest.csv"
metrics: [mcc, sensitivity, specificity, precision, f1, confusion_matrix]
baseline_values:
  mcc: 0.637
output_dir: "results/demo_eval100"
checkpoints: [post_metrics, pre_report]
abort_criteria:
  mcc_below: 0.3
  parse_failure_rate: 0.1
```

### Full Evaluation (n=300)
```yaml
model_name: "axis-mura-v1-4bit"
dataset_path: "data/eval-300"
manifest_path: "data/eval-300/manifest.csv"
metrics: [mcc, sensitivity, specificity, precision, npv, f1, accuracy, confusion_matrix]
baseline_values:
  mcc: 0.637
  sensitivity: 0.743
  specificity: 0.920
output_dir: "results/full_eval300"
checkpoints: [pre_flight, post_metrics, pre_report]
abort_criteria:
  mcc_below: 0.3
  parse_failure_rate: 0.05
  empty_predictions: true
```
