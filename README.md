# ThuanPhongNhi — FedSSL-MERC

> **Uncertainty-Aware Federated Learning for Emotion Recognition in Conversations**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 📌 Overview

ThuanPhongNhi is a framework combining **Evidential Deep Learning (EDL)** with **Federated Learning (FL)** for privacy-preserving Emotion Recognition in Conversations (ERC). Our key innovation is **EAFA (Epistemic-Aware Federated Aggregation)**, which uses model uncertainty to intelligently weight client contributions during aggregation.

### Key Contributions

1. **EDL for Dialogue** — First application of Evidential Deep Learning on dialogue-level sequential data (DialogueRNN + Dirichlet head) for per-utterance uncertainty estimation
2. **EAFA Aggregation** — Uncertainty-weighted FL aggregation that auto-downweights noisy clients, outperforming both FedAvg and centralized training
3. **ECR** — Evidential Consistency Regularization for semi-supervised learning, replacing FixMatch's hard 0.95 threshold with certainty-weighted Dirichlet KL divergence
4. **Privacy-Preserving ERC** — Data never leaves client devices; only model weights are shared

### Related Work

This work addresses a complementary challenge to [FedDISC (NeurIPS 2025)](https://neurips.cc), which handles missing modalities in federated MERC via diffusion models. ThuanPhongNhi instead focuses on **client quality heterogeneity** through uncertainty-guided aggregation.

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    SERVER (Central)                        │
│  ┌──────────────────────────────────────────────────┐    │
│  │  EAFA: Epistemic-Aware Federated Aggregation     │    │
│  │  w_k = f(data_size, 1/uncertainty_k)             │    │
│  └──────────────────────────────────────────────────┘    │
│         ▲              ▲              ▲                   │
└─────────┼──────────────┼──────────────┼───────────────────┘
          │              │              │
    ┌─────┴─────┐  ┌─────┴─────┐  ┌─────┴─────┐
    │  Client 1  │  │  Client 2  │  │  Client N  │
    │ RoBERTa    │  │ RoBERTa    │  │ RoBERTa    │
    │ DialogRNN  │  │ DialogRNN  │  │ DialogRNN  │
    │ EDL Head   │  │ EDL Head   │  │ EDL Head   │
    │ → α, u     │  │ → α, u     │  │ → α, u     │
    └────────────┘  └────────────┘  └────────────┘
```

**Pipeline per utterance:**
```
Text → RoBERTa (frozen, 768d) → DialogueRNN (context) → EDL Head → Dirichlet(α)
                                                          ├── belief: b = (α-1)/S
                                                          ├── uncertainty: u = C/S
                                                          └── prediction: argmax(b)
```

## 📊 Results

### Ablation Study (DailyDialog, Micro F1 excl. neutral)

| Config | Loss | Aggregation | WF1 | Uncertainty |
|:-------|:----:|:-----------:|:---:|:----------:|
| CE Centralized | CE | — | 0.882 | ✗ |
| CE FedAvg | CE | FedAvg | 0.876 | ✗ |
| EDL Centralized | EDL | — | 0.880 | ✓ |
| EDL FedAvg | EDL | FedAvg | 0.885 | ✓ |
| **EDL EAFA** | **EDL** | **EAFA** | **0.887** | **✓** |

### Cross-Dataset (3 seeds, mean WF1)

| Dataset | EDL Centralized | EDL EAFA | Δ |
|:--------|:---------------:|:--------:|:-:|
| MELD (7 classes) | 63.09 | **63.44** | +0.35 |
| IEMOCAP (6 classes) | 56.33 | **58.46** | +2.13 |
| DailyDialog (6 classes) | 87.99 | **88.69** | +0.70 |

> EAFA outperforms centralized training on all 3 datasets — uncertainty-guided collaboration captures complementary data patterns.

## 📁 Project Structure

```
FedSSL-MERC/
├── data/
│   ├── datasets/               # Dataset loaders (MELD, IEMOCAP, DailyDialog)
│   ├── raw/                    # Raw dataset CSVs
│   └── features/               # Pre-extracted RoBERTa features (.pt)
├── models/
│   ├── erc/
│   │   └── dialogue_rnn.py     # Base DialogueRNN (CE baseline)
│   └── evidential/
│       ├── evidential_dialogue_rnn.py  # EDL wrapper
│       ├── edl_head.py         # Dirichlet evidence layer
│       └── losses.py           # EDL loss + ECR regularization
├── federated/
│   ├── aggregation/
│   │   └── eafa.py             # EAFA aggregator
│   ├── partitioner.py          # Dirichlet non-IID partitioning
│   ├── client.py               # FL client training
│   └── server.py               # FL server orchestration
├── scripts/
│   ├── train_multi_dataset.py  # Main training script (all configs)
│   ├── run_ablation.py         # Automated 9-run ablation
│   ├── demo_realdata.py        # Inference demo with test data
│   ├── demo_inference.py       # Interactive inference demo
│   ├── finetune_roberta.py     # RoBERTa feature extraction
│   └── extract_text_features_multi.py
├── checkpoints/                # Trained model weights
├── docs/
│   └── literature_survey.md    # 13-paper survey + novelty analysis
└── configs/                    # YAML configurations
```

## 🔬 Supported Components

### Datasets
| Dataset | Modalities | Emotions | Utterances | Source |
|---------|:----------:|:--------:|:----------:|--------|
| MELD | Text (+Audio) | 7 | ~13,708 | Friends TV Series |
| IEMOCAP | Text | 6 | ~7,433 | USC Acted Dialogues |
| DailyDialog | Text | 6 (excl. neutral) | ~59,547 | Open-domain Dialogues |

### Model Components
| Component | Implementation | Purpose |
|:----------|:---------------|:--------|
| Encoder | RoBERTa-Base (125M, frozen) | Text feature extraction |
| Context | DialogueRNN (GRU × 3) | Speaker + global + emotion tracking |
| Head | EDL (Dirichlet) | Uncertainty-aware classification |
| Aggregation | EAFA (β-weighted) | Epistemic-guided FL |
| SSL | ECR | Certainty-weighted consistency |

### Ablation Configurations
| Flag | Options | Description |
|:-----|:--------|:------------|
| `--loss_type` | `edl` / `ce` | Evidential vs CrossEntropy |
| `--aggregation` | `eafa` / `fedavg` | Uncertainty-weighted vs uniform |
| `--mode` | `centralized` / `federated` | Single-site vs multi-client |

## ⚙️ Setup

### Requirements
```bash
pip install -r requirements.txt
```

### Quick Start
```bash
# 1. Train EDL + EAFA (main pipeline)
python scripts/train_multi_dataset.py --dataset meld --finetuned --seed 42

# 2. Train CE baseline (centralized)
python scripts/train_multi_dataset.py --dataset meld --finetuned --loss_type ce --mode centralized

# 3. Run ablation study (9 configs × 3 seeds)
python scripts/run_ablation.py

# 4. Demo inference on test data
python scripts/demo_realdata.py --dataset meld --num 5
```

### Demo Output Example
```
🗣️ "My God! What happened to you?"
   → 😮 surprise     (75.7%)  u:████░░░░░░░░░░░░░░░░ 0.243  [true:surprise] ✓

🗣️ "I don't know. We're talking about whipped fish..."
   → 😊 joy          ( 3.0%)  u:██████████████████░░ 0.944  [true:disgust] ✗
                               ↑ Model knows it doesn't know!
```

## 👥 Team

| Member | Role |
|--------|------|
| **Đinh Đại Lộc** | Lead Developer & Architecture |
| **Trần Phi Học** | Training & Experiments |
| **Hồ Gia Phú** | Research & Survey |

## 📚 Key References

1. Majumder, N. et al. (2019). *DialogueRNN: An Attentive RNN for Emotion Detection in Conversations*. AAAI.
2. Sensoy, M. et al. (2018). *Evidential Deep Learning to Quantify Classification Uncertainty*. NeurIPS.
3. McMahan, B. et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data*. AISTATS.
4. Qiu, X. et al. (2025). *FedDISC: Federated Dialogue-Semantic Diffusion for Emotion Recognition under Incomplete Modalities*. NeurIPS.
5. Sohn, K. et al. (2020). *FixMatch: Simplifying Semi-Supervised Learning*. NeurIPS.

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

**Lab:** AiTA Lab (AI Technology and Application Research Lab)  
**Institution:** FPT University  
**Target:** AAAI 2027 Submission
