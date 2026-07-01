# FedSSL-MERC

> **Evidential Federated Learning for Robust Multimodal Emotion Recognition in Conversations**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

FedSSL-MERC couples **Evidential Deep Learning (EDL)** with **Federated Learning (FL)**
for privacy-preserving Emotion Recognition in Conversations (ERC). One Dirichlet
uncertainty estimate is threaded through three mechanisms — **EAFA**
(uncertainty-aware aggregation), **ECR** (threshold-free semi-supervised consistency),
and **ELF** (evidence-level multimodal fusion) — plus **EAFA-Guard**, a server-root
direction filter that restores robustness under update-poisoning attacks.

> **Honesty note.** Every number is produced by executed code. Results from the
> **controlled NumPy testbed** (`testbed/`) are reported *separately* from the
> **real-corpus** runs (MELD / IEMOCAP / DailyDialog); the testbed isolates each
> mechanism cleanly and is not a substitute for real-data evaluation.

---

## Authors

| Author | Role |
|--------|------|
| **Đinh Đại Lộc** | Student author — architecture & implementation |
| **Trần Phi Học** | Student author — training & experiments |
| **Hồ Gia Phú** | Student author — research & analysis |
| **TS. Lê Võ Minh Thư** | Corresponding author / advisor (`thulvm@fe.edu.vn`) |

Department of Artificial Intelligence, FPT University, Ho Chi Minh City, Vietnam.

---

## Key contributions

1. **EDL for dialogue** — Dirichlet evidential head on top of DialogueRNN for
   per-utterance uncertainty in a single forward pass.
2. **EAFA aggregation** — uncertainty-weighted FL aggregation that automatically
   down-weights low-quality clients at O(1) communication overhead.
3. **ECR** — threshold-free semi-supervised consistency: replaces FixMatch's hard
   confidence threshold with a certainty-weighted Dirichlet KL whose gradient vanishes
   when the model is uncertain.
4. **ELF** — evidence-level multimodal fusion that degrades gracefully when audio is
   missing.
5. **EAFA-Guard** — a server-root direction filter + magnitude cap that restores
   robustness under update-poisoning, where uncertainty weighting alone fails.

---

## Repository layout

```
FedSSL-MERC/
├── README.md  REPOSITORY_GUIDE.md  LICENSE
├── Makefile   reproduce.sh                 # one-command reproduction
├── requirements.txt  setup.py  implementation_plan.md
│
├── data/             # dataset loaders, federated partitioning, preprocessing
├── models/           # encoders, erc (DialogueRNN), evidential (EDL head, losses), fusion
├── federated/        # client, server, aggregation (EAFA / EAFA-Guard / baselines), privacy
├── semi_supervised/  # ECR, FixMatch, FlexMatch, FSSL baselines, augmentation
├── configs/          # backbone / federated / ssl configuration files
├── scripts/          # training + experiment + figure-generation entry points
├── results/          # executed real-corpus experiment outputs (JSON)
├── results_*.json    # top-level experiment outputs
├── assets/           # architecture and result figures
│
├── testbed/          # controlled NumPy federated-EDL simulator (no GPU)
│   ├── fedsim.py  fedtrain.py  run_experiments.py
│   ├── results/      # executed JSON outputs
│   └── tests/        # quick sanity checks
│
└── paper/            # manuscript (comment-free LaTeX)
    ├── main.tex
    ├── figures/
    └── figure_scripts/
```

---

## How to run

### 0. Install

```bash
python -m venv .venv && source .venv/bin/activate
make install              # = pip install -r requirements.txt
```

### 1. Reproduce all testbed results with one command

The controlled federated-EDL testbed depends only on NumPy/SciPy/Matplotlib (no GPU,
no PyTorch). It regenerates every Section-7 number and figure:

```bash
make reproduce            # runs the full battery + regenerates figures
# or, equivalently:
./reproduce.sh
```

Outputs land in `testbed/results/*.json` and `paper/figures/`. To run only the
experiments (no figures): `make testbed`. To run the sanity checks: `make test`.

### 2. Real-corpus training (PyTorch)

The real-data pipeline (MELD / IEMOCAP / DailyDialog) uses RoBERTa + WavLM +
DialogueRNN. Datasets are not redistributed here (licensing). Typical entry points:

```bash
# main EDL + EAFA pipeline
python scripts/train_multi_dataset.py --dataset meld --finetuned --seed 42

# CE baseline (centralized)
python scripts/train_multi_dataset.py --dataset meld --loss_type ce --mode centralized

# robustness / Byzantine / calibration studies
python scripts/run_noise_robustness.py
python scripts/run_byzantine_robustness.py
python scripts/run_calibration_analysis.py

# regenerate real-data paper figures
python scripts/generate_figures.py
```

Run `python scripts/<name>.py --help` for per-script options.

### 3. Build the paper

```bash
make paper                # = pdflatex loop in paper/
```

For an actual Elsevier submission, switch the document class in `paper/main.tex` to
`\documentclass[review]{elsarticle}` and use `natbib`.

---

## Selected results

### Cross-dataset (3 seeds, mean Weighted-F1)

| Dataset | EDL Centralized | EDL + EAFA | Δ |
|:--------|:---------------:|:----------:|:--:|
| MELD (7 classes)      | 63.09 | **63.44** | +0.35 |
| IEMOCAP (6 classes)   | 56.33 | **58.46** | +2.13 |
| DailyDialog (6 cls.)  | 87.99 | **88.69** | +0.70 |

### Robustness (controlled testbed, macro-F1)

| Setting | EAFA-Guard | Best classical baseline |
|---------|-----------:|------------------------:|
| Label-flip 20%    | ≈ 89% | Multi-Krum ≈ 91% |
| Sign-flip 20%     | ≈ 89% | Multi-Krum ≈ 42% |
| Adaptive 20%      | ≈ 89% | Krum ≈ 85% |
| Contamination 40% | ≈ 94% | < 56% |

EAFA-Guard is the only aggregator tested that holds up against **all three** attack
types. Plain EAFA is competitive under benign noise but collapses under update
poisoning — reported explicitly as the motivation for the guard.

---

## Components

| Component | Implementation | Purpose |
|:----------|:---------------|:--------|
| Text encoder | RoBERTa-Base (frozen) | utterance features |
| Audio encoder | WavLM (frozen) | speech features |
| Context | DialogueRNN | speaker + global + emotion tracking |
| Head | EDL (Dirichlet) | single-pass uncertainty |
| Aggregation | EAFA / EAFA-Guard | epistemic-guided, robust FL |
| SSL | ECR | certainty-weighted consistency |
| Fusion | ELF | evidence-level multimodal fusion |

---

## Key references

1. Majumder, N. et al. (2019). *DialogueRNN: An Attentive RNN for Emotion Detection in Conversations*. AAAI.
2. Sensoy, M. et al. (2018). *Evidential Deep Learning to Quantify Classification Uncertainty*. NeurIPS.
3. McMahan, B. et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data*. AISTATS.
4. Yin, D. et al. (2018). *Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates*. ICML.
5. Sohn, K. et al. (2020). *FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence*. NeurIPS.

---

## License

Released under the MIT License — see [LICENSE](LICENSE).

**Institution:** FPT University, Ho Chi Minh City, Vietnam.
