# 01 — Industrial Defect Detection

## Objective

Build a reproducible real-time defect detection baseline, then measure the accuracy–latency trade-off across model, resolution, precision, ROI, and deployment backend choices.

## Stage A — Baseline

Start with one small YOLO detector and one public industrial defect dataset.

Recommended first dataset: NEU surface defect dataset (or another equivalent public steel-surface defect dataset if you already have one prepared).

Baseline requirements:

- fixed train / validation / test split
- fixed random seed
- one small YOLO model
- one input resolution
- documented training command / config
- saved metrics and example predictions

## Stage B — Controlled experiments

Change one variable at a time.

| Experiment | Variable | Suggested values |
|---|---|---|
| E01 | input size | 416 / 512 / 640 |
| E02 | inference precision | FP32 / FP16 |
| E03 | model size | nano / small |
| E04 | detection area | full frame / ROI |
| E05 | backend | PyTorch / ONNX / TensorRT |

## Metrics

Record at minimum:

- precision
- recall
- mAP50
- mAP50-95
- average latency (ms)
- P95 latency (ms)
- FPS
- peak VRAM (MB)
- model size (MB)

## Directory layout

```text
01_defect_detection/
├── README.md
├── configs/
├── scripts/
├── src/
├── results/
└── samples/
```

Do not commit large raw datasets, model weights, or generated training runs to Git by default.

## First acceptance criteria

The first baseline is complete when all of the following are true:

1. Training finishes without manual intervention.
2. Validation metrics are saved.
3. A separate inference script can load the trained checkpoint.
4. At least 10 held-out images can be visualized with predictions.
5. One experiment record explains what was trained, with which parameters, and what was learned from the result.

## Next milestone

After the baseline works, build a benchmark script that measures inference latency after warm-up and reports mean, median, P95, FPS, and VRAM usage.
