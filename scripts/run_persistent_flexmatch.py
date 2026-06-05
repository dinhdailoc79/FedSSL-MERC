"""
Persistent FlexMatch Experiment Runner
========================================
Compares original FlexMatch (shared/reset buffers) against two faithful
federated adaptations:
  1. PersistentFlexMatch: per-client persistent class_counts
  2. ServerAggFlexMatch: server-aggregated global thresholds

Runs on MELD and IEMOCAP with 3 seeds at 10% label ratio (matching
the main SSL comparison in Table 4).

Usage:
    python scripts/run_persistent_flexmatch.py
    python scripts/run_persistent_flexmatch.py --datasets meld --seeds 42
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
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.erc.dialogue_rnn import DialogueRNN
from data.federated_partition import FederatedPartitioner
from scripts.train_multi_dataset import (
    GenericDialogueDataset, collate_dialogues, load_meld, load_iemocap,
)
from semi_supervised.flexmatch import FlexMatchLoss
from semi_supervised.persistent_flexmatch import (
    PersistentFlexMatchLoss,
    ServerAggFlexMatchLoss,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_FILE = "results/persistent_flexmatch_results.json"


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_results(results):
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "item") else str(x))


def run_experiment(method, dataset_name, seed, label_ratio=0.10):
    """Run one FlexMatch variant experiment."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_rounds = 50
    local_epochs = 3
    num_clients = 5
    alpha = 0.5
    lr = 1e-3
    num_classes_map = {"meld": 7, "iemocap": 6}

    # Load data
    if dataset_name == "meld":
        train, dev, test, emotions, wts, cache, num_spk = load_meld(finetuned=True)
    else:
        train, dev, test, emotions, wts, cache, num_spk = load_iemocap(
            finetuned=True, num_classes=6
        )

    num_classes = len(emotions)
    class_weights = torch.from_numpy(wts.astype(np.float32)).to(device)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    # Federated partition with label_ratio
    partitioner = FederatedPartitioner(
        num_clients=num_clients, strategy="dirichlet",
        alpha=alpha, seed=seed,
    )
    client_partitions = partitioner.partition(train, label_ratio=label_ratio)
    dialogue_lookup = {d.dialogue_id: d for d in train}

    client_labeled_loaders = []
    client_unlabeled_loaders = []
    client_sizes = []

    for partition in client_partitions:
        labeled_dias = [dialogue_lookup[did] for did in partition.labeled_ids
                        if did in dialogue_lookup]
        labeled_ds = GenericDialogueDataset(labeled_dias, cache.get("train", {}))
        l_loader = DataLoader(labeled_ds, batch_size=16, shuffle=True,
                              collate_fn=collate_dialogues, num_workers=0)
        client_labeled_loaders.append(l_loader)

        unlabeled_dias = [dialogue_lookup[did] for did in partition.unlabeled_ids
                          if did in dialogue_lookup]
        if unlabeled_dias:
            unlabeled_ds = GenericDialogueDataset(unlabeled_dias, cache.get("train", {}))
            u_loader = DataLoader(unlabeled_ds, batch_size=16, shuffle=True,
                                  collate_fn=collate_dialogues, num_workers=0)
        else:
            u_loader = None
        client_unlabeled_loaders.append(u_loader)
        client_sizes.append(len(labeled_dias) + len(unlabeled_dias))

    test_ds = GenericDialogueDataset(test, cache.get("test", {}))
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False,
                             collate_fn=collate_dialogues, num_workers=0)

    # Initialize model (standard DialogueRNN, same as FixMatch/FlexMatch)
    global_model = DialogueRNN(
        input_dim=768, hidden_dim=256,
        num_classes=num_classes, num_speakers=num_spk,
        dropout=0.3,
    ).to(device)

    # Initialize FlexMatch variant
    if method == "flexmatch_original":
        ssl_loss = FlexMatchLoss(
            threshold=0.95, lambda_u=1.0,
            num_classes=num_classes, threshold_min=0.5,
        )
    elif method == "flexmatch_persistent":
        ssl_loss = PersistentFlexMatchLoss(
            threshold=0.95, lambda_u=1.0,
            num_classes=num_classes, threshold_min=0.5,
            num_clients=num_clients,
        )
    elif method == "flexmatch_serveragg":
        ssl_loss = ServerAggFlexMatchLoss(
            threshold=0.95, lambda_u=1.0,
            num_classes=num_classes, threshold_min=0.5,
            num_clients=num_clients,
        )

    logger.info(f"\n{'='*60}")
    logger.info(f"  {method.upper()} | {dataset_name.upper()} | "
                f"labels={label_ratio:.0%} | seed={seed}")
    logger.info(f"{'='*60}\n")

    best_wf1 = 0.0
    patience_cnt = 0

    for rnd in range(1, num_rounds + 1):
        client_states = []

        for c_idx in range(num_clients):
            l_loader = client_labeled_loaders[c_idx]
            u_loader = client_unlabeled_loaders[c_idx]

            local_model = copy.deepcopy(global_model).to(device)
            local_model.train()
            opt = optim.Adam(local_model.parameters(), lr=lr, weight_decay=1e-4)

            # Set active client for persistent variants
            if method in ("flexmatch_persistent", "flexmatch_serveragg"):
                ssl_loss.set_client(c_idx)

            for _ in range(local_epochs):
                u_iter = iter(u_loader) if u_loader else None
                for batch_l in l_loader:
                    batch_l_dev = {
                        "features": batch_l["features"].to(device),
                        "speaker_ids": batch_l["speaker_ids"].to(device),
                        "labels": batch_l["labels"].to(device),
                    }

                    batch_u = None
                    if u_iter:
                        try:
                            batch_u = next(u_iter)
                        except StopIteration:
                            u_iter = iter(u_loader)
                            batch_u = next(u_iter)
                        batch_u = {
                            k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch_u.items()
                        }

                    loss, stats = ssl_loss(local_model, batch_l_dev, batch_u, ce_loss)

                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), 5.0)
                    opt.step()

            client_states.append(
                OrderedDict({k: v.cpu() for k, v in local_model.state_dict().items()})
            )

        # Server-side aggregation after all clients finish
        if method == "flexmatch_serveragg":
            ssl_loss.server_aggregate()

        # FedAvg aggregation (same as original FlexMatch comparison)
        total = sum(client_sizes)
        weights = [ds / total for ds in client_sizes]
        new_global = OrderedDict()
        for key in client_states[0].keys():
            new_global[key] = sum(
                w * sd[key].float() for w, sd in zip(weights, client_states)
            )

        global_model.load_state_dict(new_global)
        global_model.to(device)

        # Evaluate
        global_model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                feats = batch["features"].to(device)
                spks = batch["speaker_ids"].to(device)
                labels = batch["labels"].to(device)
                mask = labels != -1
                logits = global_model(feats, spks)
                preds = logits[mask].argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels[mask].cpu().numpy())

        wf1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        logger.info(f"  {method} R{rnd:3d}/{num_rounds} | WF1={wf1:.4f}")

        if wf1 > best_wf1:
            best_wf1 = wf1
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= 15:
                logger.info(f"  Early stopping at round {rnd}")
                break

        del client_states
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return best_wf1


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Compare FlexMatch variants in federated setting"
    )
    parser.add_argument("--methods", type=str,
                        default="flexmatch_original,flexmatch_persistent,flexmatch_serveragg")
    parser.add_argument("--datasets", type=str, default="meld,iemocap")
    parser.add_argument("--seeds", type=str, default="42,123,2024")
    parser.add_argument("--label_ratio", type=float, default=0.10)
    args = parser.parse_args()

    methods = args.methods.split(",")
    datasets = args.datasets.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    results = load_results()

    for method in methods:
        for ds in datasets:
            for seed in seeds:
                key = f"{ds}_{method}_lr{args.label_ratio:.2f}_s{seed}"

                if key in results and results[key].get("wf1") is not None:
                    logger.info(f"Skipping: {key} (WF1={results[key]['wf1']})")
                    continue

                start = time.time()
                try:
                    wf1 = run_experiment(method, ds, seed, args.label_ratio)
                    elapsed = time.time() - start
                    results[key] = {
                        "wf1": round(float(wf1), 4),
                        "method": method,
                        "dataset": ds,
                        "seed": seed,
                        "label_ratio": args.label_ratio,
                        "time": round(elapsed, 1),
                    }
                    save_results(results)
                    logger.info(f">> {key} => WF1={wf1:.4f} ({elapsed:.1f}s)")
                except Exception as e:
                    logger.error(f"ERROR {key}: {e}", exc_info=True)
                    results[key] = {"wf1": None, "error": str(e)}
                    save_results(results)

    # Summary
    print(f"\n{'='*70}")
    print(f"  PERSISTENT FLEXMATCH — FINAL SUMMARY (label_ratio={args.label_ratio:.0%})")
    print(f"{'='*70}")
    print(f"  {'Method':<25} {'Dataset':<10} {'Seed':<8} {'WF1 (%)':<10}")
    print(f"  {'-'*55}")
    for key in sorted(results.keys()):
        val = results[key]
        if val.get("wf1") is not None:
            print(f"  {val['method']:<25} {val['dataset']:<10} "
                  f"{val['seed']:<8} {val['wf1']*100:<10.2f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
