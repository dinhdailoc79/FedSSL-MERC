# Repository guide

This repository contains the complete source for **FedSSL-MERC** in three layers.

## 1. Real-data implementation (PyTorch)

The main package implements the framework on the real corpora (MELD / IEMOCAP /
DailyDialog) with RoBERTa + WavLM + DialogueRNN backbones and a Dirichlet evidential
head.

```
data/              dataset loaders, federated partitioning, preprocessing
models/            encoders, ERC backbones, evidential head, fusion
federated/         client, server, aggregation (EAFA / EAFA-Guard / baselines), privacy
semi_supervised/   ECR, FixMatch, FlexMatch, FSSL baselines, augmentation
configs/           backbone / federated / ssl configuration files
scripts/           training + experiment + figure-generation entry points
results/           executed experiment outputs (JSON) and reliability data
assets/            architecture and result figures
setup.py           installable package metadata
requirements.txt   PyTorch dependencies
```

Typical entry points (see `scripts/`): `train_fed_multimodal.py`,
`train_fed_evidence.py`, `run_noise_robustness.py`, `run_byzantine_robustness.py`,
`run_calibration_analysis.py`, `generate_figures.py`.

## 2. Controlled federated-EDL testbed (NumPy)

A self-contained, dependency-light simulator used for the controlled robustness /
Byzantine study reported separately in the paper. No GPU, no deep-learning framework.

```
testbed/fedsim.py            DualEDL model, aggregators, EAFA-Guard
testbed/fedtrain.py          local training / evaluation utilities
testbed/run_experiments.py   full battery -> testbed/results/*.json
testbed/tests/               quick sanity checks
```

Run: `cd testbed && python run_experiments.py`

## 3. Paper

```
paper/main.tex               manuscript source (comment-free)
paper/figures/               figures used by the manuscript
paper/figure_scripts/        regenerate Fig. 1/2 and the testbed panels
```

Build: `cd paper && for i in 1 2 3; do pdflatex -interaction=nonstopmode main.tex; done`

## Honesty note

Every number is produced by executed code. Controlled-testbed results (layer 2) are
reported separately from the real-corpus results (layer 1); they are not a substitute
for real-data evaluation.
