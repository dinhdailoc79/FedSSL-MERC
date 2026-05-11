# FedSSL-MERC

> **Federated Semi-Supervised Learning for Multimodal Emotion Recognition in Conversations**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 📌 Overview

This repository implements a novel framework combining **Federated Learning (FL)** and **Semi-Supervised Learning (SSL)** for **Multimodal Emotion Recognition in Conversations (MERC)**. Our approach addresses three critical challenges in conversational emotion recognition:

1. **Privacy Preservation** — Train emotion models across distributed clients without sharing raw conversational data
2. **Label Scarcity** — Leverage large amounts of unlabeled conversational data through semi-supervised techniques
3. **Non-IID Data Distribution** — Handle heterogeneous emotion distributions across different data sources

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                    SERVER (Central)                   │
│  ┌─────────────────────────────────────────────┐     │
│  │  Global Model Aggregation (FedAvg/FedProx)  │     │
│  │  + Pseudo-Label Refinement                   │     │
│  └─────────────────────────────────────────────┘     │
│         ▲              ▲              ▲               │
└─────────┼──────────────┼──────────────┼───────────────┘
          │              │              │
    ┌─────┴─────┐  ┌─────┴─────┐  ┌─────┴─────┐
    │  Client 1  │  │  Client 2  │  │  Client 3  │
    │ Text+Audio │  │ Text+Video │  │ All Modal  │
    │ Local SSL  │  │ Local SSL  │  │ Local SSL  │
    └────────────┘  └────────────┘  └────────────┘
```

## 📁 Project Structure

```
FedSSL-MERC/
├── configs/                     # Experiment configurations
│   ├── federated/               # FL strategy configs
│   ├── ssl/                     # SSL method configs
│   └── backbone/                # Model backbone configs
├── data/                        # Data processing & loaders
│   └── datasets/                # Dataset-specific implementations
├── models/                      # Model architectures
│   ├── encoders/                # Modality-specific encoders
│   ├── fusion/                  # Multimodal fusion modules
│   └── erc/                     # Emotion recognition in conversation
├── federated/                   # Federated Learning core
│   ├── aggregation/             # Aggregation strategies
│   └── privacy/                 # Privacy mechanisms
├── ssl/                         # Semi-Supervised Learning modules
├── scripts/                     # Training & evaluation scripts
├── notebooks/                   # Jupyter/Colab notebooks
├── docs/                        # Documentation & reports
└── results/                     # Experiment results
```

## 🔬 Supported Components

### Datasets
| Dataset | Modalities | Utterances | Source |
|---------|:----------:|:----------:|--------|
| IEMOCAP | T+A+V | ~7,433 | USC Acted Dialogues |
| MELD | T+A+V | ~13,708 | Friends TV Series |

### Federated Strategies
- FedAvg (McMahan et al., 2017)
- FedProx (Li et al., 2020)
- SCAFFOLD (Karimireddy et al., 2020)

### Semi-Supervised Methods
- FixMatch (Sohn et al., 2020)
- FlexMatch (Zhang et al., 2021)
- Pseudo-Labeling with confidence thresholding

### ERC Backbones
- DialogueRNN (Majumder et al., 2019)
- DialogueGCN (Ghosal et al., 2019)
- Hybrid Transformer + GCN

## ⚙️ Setup

### Requirements
```bash
pip install -r requirements.txt
```

### Quick Start
```bash
# 1. Centralized baseline
python scripts/train_centralized.py --config configs/backbone/dialoguernn.yaml

# 2. Federated SSL training
python scripts/train_federated.py \
    --fl_config configs/federated/fedavg.yaml \
    --ssl_config configs/ssl/fixmatch.yaml \
    --backbone_config configs/backbone/dialoguernn.yaml
```

## 👥 Team

| Member | Role | Compute |
|--------|------|---------|
| **Lộc (Dinh Loc)** | Team Lead & Development | T4 GPU |
| **Học** | Training & Experiments | A100 GPU (Colab Ultra) |
| **Phú** | Research & Survey | T4 GPU |

## 📚 Key References

1. Shou, Y., et al. (2026). *A Comprehensive Survey on Multi-modal Conversational Emotion Recognition with Deep Learning*. ACM TOIS.
2. McMahan, B., et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data*. AISTATS.
3. Sohn, K., et al. (2020). *FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence*. NeurIPS.

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

**Lab:** AiTA Lab (AI Technology and Application Research Lab)  
**Institution:** FPT University
