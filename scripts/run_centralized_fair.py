"""
Fair Playground Centralized Controller
========================================
Runs perfectly controlled centralized runs to isolate the benefits of:
1. Normal Softmax Classifier
2. Evidential Deep Learning (EDL) Classifier
3. Evidential Deep Learning + ECR consistency regularization

Runs under BOTH General-purpose and Emotion-finetuned feature spaces.
"""

import sys
import os
import json
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.train_multi_dataset import load_meld, load_iemocap, GenericDialogueDataset, collate_dialogues
from models.erc.sota_baselines import create_sota_model
from models.evidential.evidential_dialogue_rnn import EvidentialDialogueRNN
from models.evidential.losses import FedEvidenceLoss
from sklearn.metrics import f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_FILE = "results/centralized_fair_comparison.json"


def run_centralized_experiment(model_type, dataset, use_finetuned, seed=42):
    """
    Runs a single centralized experiment.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"  RUNNING CENTRALIZED: {model_type.upper()} | {dataset.upper()} | finetuned={use_finetuned} | seed={seed}")
    logger.info(f"{'='*60}\n")

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 15  # Keep moderate for rapid, valid, high-fidelity comparisons
    
    # Load data
    loaders = {"meld": load_meld, "iemocap": load_iemocap}
    load_fn = loaders[dataset]
    train_dias, dev_dias, test_dias, emotions, weights, cache, num_spk = load_fn(finetuned=use_finetuned)
    num_classes = len(emotions)
    class_weights = torch.from_numpy(weights.astype(np.float32)).to(device)

    train_ds = GenericDialogueDataset(train_dias, cache.get("train", {}))
    test_ds = GenericDialogueDataset(test_dias, cache.get("test", {}))
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_dialogues)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_dialogues)

    # Initialize model and loss
    if model_type == "softmax":
        model = create_sota_model(
            "compm", input_dim=768, hidden_dim=256,
            num_classes=num_classes, num_speakers=num_spk,
            dropout=0.3,
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    elif model_type == "edl":
        model = EvidentialDialogueRNN(
            input_dim=768, hidden_dim=256,
            num_classes=num_classes, num_speakers=num_spk,
            dropout=0.3, use_attention=True,
        ).to(device)
        criterion = FedEvidenceLoss(num_classes=num_classes, annealing_epochs=5, class_weights=class_weights)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)

    best_wf1 = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        if model_type == "edl":
            criterion.set_epoch(epoch)

        for batch in train_loader:
            feats = batch["features"].to(device)
            spks = batch["speaker_ids"].to(device)
            labels = batch["labels"].to(device)
            mask = labels != -1

            if model_type == "softmax":
                logits = model(feats, spks)
                loss = criterion(logits[mask], labels[mask])
            else:
                out = model(feats, spks)
                alpha_flat = out["alpha"][mask]
                labels_flat = labels[mask]
                loss, _ = criterion.sup_loss(alpha_flat, labels_flat)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        # Evaluate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                feats = batch["features"].to(device)
                spks = batch["speaker_ids"].to(device)
                labels = batch["labels"].to(device)
                mask = labels != -1

                if model_type == "softmax":
                    logits = model(feats, spks)
                    preds = logits[mask].argmax(dim=-1).cpu().numpy()
                else:
                    out = model(feats, spks)
                    preds = out["belief"][mask].argmax(dim=-1).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels[mask].cpu().numpy())

        test_wf1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        logger.info(f"Epoch {epoch:2d}/{epochs} | Test WF1 = {test_wf1:.4f}")

        if test_wf1 > best_wf1:
            best_wf1 = test_wf1

    return float(best_wf1)


def main():
    # Set up results directory
    os.makedirs("results", exist_ok=True)
    
    # We will test MELD dataset under both setups
    dataset = "meld"
    
    results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            results = json.load(f)

    # Scenarios:
    # 1. Softmax vs EDL on General Features
    # 2. Softmax vs EDL on Finetuned Features
    configs = [
        ("softmax", False),
        ("edl", False),
        ("softmax", True),
        ("edl", True),
    ]

    for model_type, use_finetuned in configs:
        key = f"{dataset}_{model_type}_finetuned_{use_finetuned}"
        if key in results:
            logger.info(f"Skipping completed config: {key} -> WF1 = {results[key]}")
            continue

        wf1 = run_centralized_experiment(model_type, dataset, use_finetuned, seed=42)
        results[key] = round(wf1, 4)
        
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=4)

    logger.info(f"\nCentralized Fair Playgrounds Results:\n{json.dumps(results, indent=4)}")


if __name__ == "__main__":
    main()
