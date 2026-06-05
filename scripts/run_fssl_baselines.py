"""
Advanced FSSL and Uncertainty Aggregation Baselines Runner
===========================================================
Runs:
1. FedEU (Uncertainty-guided aggregation baseline)
2. Mean Teacher (EMA Teacher-Student consistency baseline)
3. FedSwitch (Adaptive confidence-entropy baseline)

Supports both MELD and IEMOCAP across multiple seeds and label ratios.
"""

import sys
import os
import json
import time
import copy
import logging
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.datasets.meld import MELDDataset, MELD_EMOTIONS
from data.federated_partition import FederatedPartitioner
from models.evidential.evidential_dialogue_rnn import EvidentialDialogueRNN
from models.evidential.losses import SupervisedEvidentialLoss
from models.erc.sota_baselines import create_sota_model
from federated.aggregation.fadv_baselines import fedeu_aggregate_state_dicts
from semi_supervised.fssl_baselines import MeanTeacherLoss, FedSwitchLoss
from scripts.train_multi_dataset import GenericDialogueDataset, collate_dialogues, load_meld, load_iemocap
from sklearn.metrics import f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_FILE = "results_sota_baselines.json"


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))


def run_fssl_experiment(method, dataset, label_ratio, seed=42):
    """
    Runs a specific FSSL baseline experiment.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"  RUNNING: {method.upper()} | {dataset.upper()} | label_ratio={label_ratio} | seed={seed}")
    logger.info(f"{'='*60}\n")

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_rounds = 30  # Keep rounds moderate for rapid, representative validation
    local_epochs = 3
    num_clients = 5
    alpha = 0.5
    
    # Load dataset
    loaders = {"meld": load_meld, "iemocap": load_iemocap}
    load_fn = loaders[dataset]
    train_dias, dev_dias, test_dias, emotions, weights, cache, num_spk = load_fn(finetuned=True)
    num_classes = len(emotions)
    class_weights = torch.from_numpy(weights.astype(np.float32)).to(device)

    # Federated Partition
    partitioner = FederatedPartitioner(num_clients=num_clients, strategy="dirichlet", alpha=alpha, seed=seed)
    client_partitions = partitioner.partition(train_dias, label_ratio=label_ratio)
    dialogue_lookup = {d.dialogue_id: d for d in train_dias}

    client_labeled_loaders = []
    client_unlabeled_loaders = []
    client_sizes = []

    for partition in client_partitions:
        # Labeled data
        labeled_dias = [dialogue_lookup[did] for did in partition.labeled_ids if did in dialogue_lookup]
        labeled_ds = GenericDialogueDataset(labeled_dias, cache.get("train", {}))
        l_loader = DataLoader(labeled_ds, batch_size=16, shuffle=True, collate_fn=collate_dialogues)
        client_labeled_loaders.append(l_loader)
        client_sizes.append(len(labeled_dias))

        # Unlabeled data
        unlabeled_dias = [dialogue_lookup[did] for did in partition.unlabeled_ids if did in dialogue_lookup]
        if unlabeled_dias:
            unlabeled_ds = GenericDialogueDataset(unlabeled_dias, cache.get("train", {}))
            u_loader = DataLoader(unlabeled_ds, batch_size=16, shuffle=True, collate_fn=collate_dialogues)
        else:
            u_loader = None
        client_unlabeled_loaders.append(u_loader)

    test_ds = GenericDialogueDataset(test_dias, cache.get("test", {}))
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_dialogues)

    # Initialize model and loss based on method
    if method == "fedeu":
        # FedEU uses evidential deep learning network
        global_model = EvidentialDialogueRNN(
            input_dim=768, hidden_dim=256,
            num_classes=num_classes, num_speakers=num_spk,
            dropout=0.3, use_attention=True,
        ).to(device)
        criterion = SupervisedEvidentialLoss(num_classes=num_classes, class_weights=class_weights)
    else:
        # Mean Teacher and FedSwitch use standard DialogueRNN
        global_model = create_sota_model(
            "compm", input_dim=768, hidden_dim=256,
            num_classes=num_classes, num_speakers=num_spk,
            dropout=0.3,
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Initialize FSSL losses
    if method == "mean_teacher":
        fssl_loss = MeanTeacherLoss(ema_alpha=0.99, lambda_u=1.0, num_classes=num_classes)
    elif method == "fedswitch":
        fssl_loss = FedSwitchLoss(threshold_init=0.95, lambda_u=1.0, num_classes=num_classes)

    best_wf1 = 0.0
    patience = 10
    patience_cnt = 0

    for round_num in range(1, num_rounds + 1):
        client_states = []
        client_uncertainties = []

        for c_idx in range(num_clients):
            l_loader = client_labeled_loaders[c_idx]
            u_loader = client_unlabeled_loaders[c_idx]

            local_model = copy.deepcopy(global_model).to(device)
            local_model.train()
            opt = optim.Adam(local_model.parameters(), lr=1e-3, weight_decay=1e-4)

            # Local training epochs
            for _ in range(local_epochs):
                u_iter = iter(u_loader) if u_loader else None
                for batch_l in l_loader:
                    batch_l = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_l.items()}
                    
                    batch_u = None
                    if u_loader:
                        try:
                            batch_u = next(u_iter)
                        except StopIteration:
                            u_iter = iter(u_loader)
                            batch_u = next(u_iter)
                        batch_u = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_u.items()}

                    if method == "fedeu":
                        feats = batch_l["features"]
                        spks = batch_l["speaker_ids"]
                        labels = batch_l["labels"]
                        mask = labels != -1

                        out = local_model(feats, spks)
                        alpha_flat = out["alpha"][mask]
                        labels_flat = labels[mask]

                        loss, _ = criterion(alpha_flat, labels_flat)
                    elif method == "mean_teacher":
                        loss, _ = fssl_loss(local_model, batch_l, batch_u, criterion)
                    elif method == "fedswitch":
                        loss, _ = fssl_loss(local_model, batch_l, batch_u, criterion)

                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), 5.0)
                    opt.step()

            # Record state
            client_states.append(OrderedDict({k: v.cpu() for k, v in local_model.state_dict().items()}))

            # Record client uncertainty if using FedEU
            if method == "fedeu":
                # Compute average uncertainty
                local_model.eval()
                all_u = []
                with torch.no_grad():
                    for batch_l in l_loader:
                        feats = batch_l["features"].to(device)
                        spks = batch_l["speaker_ids"].to(device)
                        out = local_model(feats, spks)
                        all_u.extend(out["uncertainty"][batch_l["labels"] != -1].cpu().numpy())
                client_uncertainties.append(float(np.mean(all_u)) if all_u else 0.5)

        # SERVER-SIDE AGGREGATION
        global_state = global_model.state_dict()
        if method == "fedeu":
            # FedEU Aggregation (Uncertainty-guided Top-k)
            avg_state = fedeu_aggregate_state_dicts(
                global_state, client_states, client_sizes, client_uncertainties, keep_ratio=0.8
            )
        else:
            # Standard volume-weighted FedAvg
            total_size = sum(client_sizes)
            avg_state = OrderedDict()
            for key in client_states[0]:
                avg_state[key] = sum(
                    client_states[i][key] * (client_sizes[i] / total_size)
                    for i in range(num_clients)
                )

        global_model.load_state_dict(avg_state)
        global_model.to(device)

        # Global Evaluation
        global_model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                feats = batch["features"].to(device)
                spks = batch["speaker_ids"].to(device)
                labels = batch["labels"].to(device)
                mask = labels != -1

                if method == "fedeu":
                    out = global_model(feats, spks)
                    preds = out["belief"][mask].argmax(dim=-1).cpu().numpy()
                else:
                    logits = global_model(feats, spks)
                    preds = logits[mask].argmax(dim=-1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels[mask].cpu().numpy())

        test_wf1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        logger.info(f"Round {round_num:2d}/{num_rounds} | Test WF1 = {test_wf1:.4f}")

        if test_wf1 > best_wf1:
            best_wf1 = test_wf1
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                logger.info(f"Early stopping at round {round_num}")
                break

    return float(best_wf1)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run advanced FL/FSSL baselines")
    parser.add_argument("--methods", type=str, default="fedeu,mean_teacher,fedswitch")
    parser.add_argument("--datasets", type=str, default="meld")
    parser.add_argument("--label_ratios", type=str, default="0.10")
    parser.add_argument("--seeds", type=str, default="42")
    args = parser.parse_args()

    methods = args.methods.split(",")
    datasets = args.datasets.split(",")
    label_ratios = [float(x) for x in args.label_ratios.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    results = load_results()

    for method in methods:
        for ds in datasets:
            for lr in label_ratios:
                for seed in seeds:
                    key = f"{ds}_{method}_lr{lr:.2f}_s{seed}"
                    
                    if key in results and results[key].get("wf1") is not None:
                        logger.info(f"Skipping completed: {key} (WF1 = {results[key]['wf1']})")
                        continue

                    start_time = time.time()
                    try:
                        wf1 = run_fssl_experiment(method, ds, lr, seed)
                        elapsed = time.time() - start_time
                        results[key] = {
                            "wf1": round(wf1, 4),
                            "model": method,
                            "dataset": ds,
                            "label_ratio": lr,
                            "seed": seed,
                            "time": round(elapsed, 1)
                        }
                        save_results(results)
                        logger.info(f">> SUCCESS: {key} -> WF1 = {wf1:.4f} (took {elapsed:.1f}s)")
                    except Exception as e:
                        logger.error(f">> ERROR running {key}: {e}", exc_info=True)


if __name__ == "__main__":
    main()
