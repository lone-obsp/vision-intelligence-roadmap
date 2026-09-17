# Vision Intelligence Roadmap

A long-term learning and research repository for building a continuous technical path from visual perception to visual generation, multimodal understanding, evaluation feedback, and photonic AI computing.

## Technical route

```text
Visual perception
    ↓
Visual representation / tokenization
    ↓
Autoregressive visual modeling
    ↓
Multimodal understanding
    ↓
Understanding-conditioned generation
    ↓
Evaluation / feedback
    ↓
Photonic / novel computing acceleration
```

## Current phase

### Phase 1 — Industrial visual perception

Goal: build a reproducible industrial defect detection baseline and study the accuracy–latency trade-off.

Primary questions:

- How do model size and input resolution affect mAP and latency?
- What is the effect of FP32 vs FP16 inference?
- How much can ROI restriction improve real-time performance?
- What changes after export to ONNX / TensorRT?

Metrics to record:

- Precision
- Recall
- mAP50
- mAP50-95
- Average latency
- P95 latency
- FPS
- VRAM usage
- Model size

## Repository structure

```text
01_defect_detection/      # Phase 1: industrial visual perception
02_autoencoder/           # Phase 2: continuous latent representation
03_vqvae/                 # Phase 3: discrete visual tokenizer
04_visual_autoregressive/ # Phase 4: mini autoregressive image model
05_multimodal_reasoning/  # Phase 5: structured visual understanding
06_generation_feedback/   # Phase 6: generation + evaluator loop
07_photonics/             # Phase 7: photonic computing simulations
experiments/              # reproducible experiment records
papers/                   # paper notes
notes/                    # technical notes
```

## Working rule

Every experiment should record:

```text
question → config → code version → metrics → result → conclusion
```

The goal is not to collect model names. The goal is to understand the underlying problems and build progressively stronger prototypes.
