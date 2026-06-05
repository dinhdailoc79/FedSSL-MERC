"""
Modern FL Aggregation Baselines Runner
========================================
Runs SCAFFOLD, FedNova, FedAdam, MOON on MELD and IEMOCAP-6
to address reviewer's primary concern about limited FL baselines.

All baselines use the same EvidentialDialogueRNN backbone and
SupervisedEvidentialLoss to ensure fair comparison with EAFA.

Usage:
    python scripts/run_fl_baselines.py
    python scripts/run_fl_baselines.py --methods scaffold,fednova
    python scripts/run_fl_baselines.py --datasets meld,iemocap --seeds 42,123,2024
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

from models.evidential.evidential_dialogue_rnn import EvidentialDialogueRNN
from models.evidential.losses import SupervisedEvidentialLoss
from data.federated_partition import FederatedPartitioner
from scripts.train_multi_dataset import (
    GenericDialogueDataset, collate_dialogues,
    load_meld, load_iemocap, evaluate,
)
from federated.aggregation.fl_baselines import (
    SCAFFOLDAggregator,
    fednova_aggregate,
    FedAdamAggregator,
    MOONContrastiveLoss,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_FILE = "results/fl_baselines_results.json"


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


@torch.no_grad()
def eval_model(model, loader, device):
    """Evaluate model and return WF1."""
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        feats = batch["features"].to(device)
        spks = batch["speaker_ids"].to(device)
        labels = batch["labels"].to(device)
        mask = labels != -1
        out = model(feats, spks)
        preds = out["belief"][mask].argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels[mask].cpu().numpy())
    return f1_score(all_labels, all_preds, average="weighted", zero_division=0)


# ============================================================
# Method-specific federated training loops
# ============================================================

def run_scaffold(global_model, client_loaders, test_loader, client_sizes,
                 num_classes, class_weights, num_speakers, device,
                 num_rounds=50, local_epochs=3, lr=1e-3):
    """Run SCAFFOLD federated training."""
    num_clients = len(client_loaders)
    loss_fn = SupervisedEvidentialLoss(
        num_classes=num_classes, annealing_epochs=30,
        class_weights=class_weights,
    )

    scaffold = SCAFFOLDAggregator(
        global_state_dict=OrderedDict(
            {k: v.cpu() for k, v in global_model.state_dict().items()}
        ),
        num_clients=num_clients,
        lr=lr,
    )

    best_wf1 = 0.0
    patience_cnt = 0

    for rnd in range(1, num_rounds + 1):
        client_states = []
        new_c_clients = []
        local_steps_list = []
        loss_fn.set_epoch(rnd)

        global_state_cpu = OrderedDict(
            {k: v.cpu() for k, v in global_model.state_dict().items()}
        )

        for k, loader in enumerate(client_loaders):
            local_model = copy.deepcopy(global_model).to(device)
            local_model.train()
            opt = optim.Adam(local_model.parameters(), lr=lr, weight_decay=1e-4)

            num_steps = 0
            for _ in range(local_epochs):
                for batch in loader:
                    feats = batch["features"].to(device)
                    spks = batch["speaker_ids"].to(device)
                    labels = batch["labels"].to(device)
                    mask = labels != -1

                    out = local_model(feats, spks)
                    loss, _ = loss_fn(out["alpha"][mask], labels[mask])

                    opt.zero_grad()
                    loss.backward()

                    # Apply SCAFFOLD gradient correction
                    for name, param in local_model.named_parameters():
                        if param.grad is not None and name in scaffold.c_global:
                            param.grad.data = SCAFFOLDAggregator.client_grad_correction(
                                param.grad.data,
                                scaffold.c_global[name].to(device),
                                scaffold.c_clients[k][name].to(device),
                            )

                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), 5.0)
                    opt.step()
                    num_steps += 1

            local_state_cpu = OrderedDict(
                {k_: v.cpu() for k_, v in local_model.state_dict().items()}
            )
            client_states.append(local_state_cpu)
            local_steps_list.append(num_steps)

            # Compute new control variate
            new_ck = scaffold.compute_new_control_variate(
                client_idx=k,
                global_state_dict=global_state_cpu,
                local_state_dict=local_state_cpu,
                num_local_steps=num_steps,
            )
            new_c_clients.append(new_ck)

        # Server aggregation
        new_global = scaffold.aggregate(
            global_state_dict=global_state_cpu,
            client_state_dicts=client_states,
            client_data_sizes=client_sizes,
            new_c_clients=new_c_clients,
            client_indices=list(range(num_clients)),
        )

        global_model.load_state_dict(new_global)
        global_model.to(device)

        wf1 = eval_model(global_model, test_loader, device)
        logger.info(f"  SCAFFOLD Round {rnd:3d}/{num_rounds} | WF1={wf1:.4f}")

        if wf1 > best_wf1:
            best_wf1 = wf1
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= 15:
                logger.info(f"  Early stopping at round {rnd}")
                break

        del client_states, new_c_clients
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return best_wf1


def run_fednova(global_model, client_loaders, test_loader, client_sizes,
                num_classes, class_weights, num_speakers, device,
                num_rounds=50, local_epochs=3, lr=1e-3):
    """Run FedNova federated training."""
    num_clients = len(client_loaders)
    loss_fn = SupervisedEvidentialLoss(
        num_classes=num_classes, annealing_epochs=30,
        class_weights=class_weights,
    )

    best_wf1 = 0.0
    patience_cnt = 0

    for rnd in range(1, num_rounds + 1):
        client_states = []
        client_local_steps = []
        loss_fn.set_epoch(rnd)

        global_state_cpu = OrderedDict(
            {k: v.cpu() for k, v in global_model.state_dict().items()}
        )

        for k, loader in enumerate(client_loaders):
            local_model = copy.deepcopy(global_model).to(device)
            local_model.train()
            opt = optim.Adam(local_model.parameters(), lr=lr, weight_decay=1e-4)

            num_steps = 0
            for _ in range(local_epochs):
                for batch in loader:
                    feats = batch["features"].to(device)
                    spks = batch["speaker_ids"].to(device)
                    labels = batch["labels"].to(device)
                    mask = labels != -1

                    out = local_model(feats, spks)
                    loss, _ = loss_fn(out["alpha"][mask], labels[mask])

                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), 5.0)
                    opt.step()
                    num_steps += 1

            local_state_cpu = OrderedDict(
                {k_: v.cpu() for k_, v in local_model.state_dict().items()}
            )
            client_states.append(local_state_cpu)
            client_local_steps.append(num_steps)

        # FedNova server aggregation
        new_global = fednova_aggregate(
            global_state_dict=global_state_cpu,
            client_state_dicts=client_states,
            client_data_sizes=client_sizes,
            client_local_steps=client_local_steps,
        )

        global_model.load_state_dict(new_global)
        global_model.to(device)

        wf1 = eval_model(global_model, test_loader, device)
        logger.info(f"  FedNova Round {rnd:3d}/{num_rounds} | WF1={wf1:.4f}")

        if wf1 > best_wf1:
            best_wf1 = wf1
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= 15:
                break

        del client_states
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return best_wf1


def run_fedadam(global_model, client_loaders, test_loader, client_sizes,
                num_classes, class_weights, num_speakers, device,
                num_rounds=50, local_epochs=3, lr=1e-3,
                server_lr=1e-2):
    """Run FedAdam (server-side Adam) federated training."""
    num_clients = len(client_loaders)
    loss_fn = SupervisedEvidentialLoss(
        num_classes=num_classes, annealing_epochs=30,
        class_weights=class_weights,
    )

    fed_adam = FedAdamAggregator(lr=server_lr, beta1=0.9, beta2=0.99, eps=1e-3)

    best_wf1 = 0.0
    patience_cnt = 0

    for rnd in range(1, num_rounds + 1):
        client_states = []
        loss_fn.set_epoch(rnd)

        for k, loader in enumerate(client_loaders):
            local_model = copy.deepcopy(global_model).to(device)
            local_model.train()
            opt = optim.Adam(local_model.parameters(), lr=lr, weight_decay=1e-4)

            for _ in range(local_epochs):
                for batch in loader:
                    feats = batch["features"].to(device)
                    spks = batch["speaker_ids"].to(device)
                    labels = batch["labels"].to(device)
                    mask = labels != -1

                    out = local_model(feats, spks)
                    loss, _ = loss_fn(out["alpha"][mask], labels[mask])

                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), 5.0)
                    opt.step()

            client_states.append(
                OrderedDict({k_: v.cpu() for k_, v in local_model.state_dict().items()})
            )

        # FedAdam server aggregation
        global_state_cpu = OrderedDict(
            {k: v.cpu() for k, v in global_model.state_dict().items()}
        )
        new_global = fed_adam.aggregate(
            global_state_dict=global_state_cpu,
            client_state_dicts=client_states,
            client_data_sizes=client_sizes,
        )

        global_model.load_state_dict(new_global)
        global_model.to(device)

        wf1 = eval_model(global_model, test_loader, device)
        logger.info(f"  FedAdam Round {rnd:3d}/{num_rounds} | WF1={wf1:.4f}")

        if wf1 > best_wf1:
            best_wf1 = wf1
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= 15:
                break

        del client_states
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return best_wf1


def run_moon(global_model, client_loaders, test_loader, client_sizes,
             num_classes, class_weights, num_speakers, device,
             num_rounds=50, local_epochs=3, lr=1e-3,
             mu_moon=1.0, temperature=0.5):
    """Run MOON (Model-Contrastive FL) federated training."""
    num_clients = len(client_loaders)
    loss_fn = SupervisedEvidentialLoss(
        num_classes=num_classes, annealing_epochs=30,
        class_weights=class_weights,
    )
    moon_loss = MOONContrastiveLoss(temperature=temperature, mu=mu_moon)

    # Previous-round local models (for contrastive loss)
    prev_local_states = [
        OrderedDict({k: v.cpu() for k, v in global_model.state_dict().items()})
        for _ in range(num_clients)
    ]

    best_wf1 = 0.0
    patience_cnt = 0

    for rnd in range(1, num_rounds + 1):
        client_states = []
        loss_fn.set_epoch(rnd)

        for k, loader in enumerate(client_loaders):
            local_model = copy.deepcopy(global_model).to(device)
            local_model.train()
            opt = optim.Adam(local_model.parameters(), lr=lr, weight_decay=1e-4)

            # Frozen global and previous models for contrastive loss
            global_frozen = copy.deepcopy(global_model).to(device)
            global_frozen.eval()
            for p in global_frozen.parameters():
                p.requires_grad = False

            prev_model = copy.deepcopy(global_model).to(device)
            prev_model.load_state_dict(
                {k_: v.to(device) for k_, v in prev_local_states[k].items()}
            )
            prev_model.eval()
            for p in prev_model.parameters():
                p.requires_grad = False

            for _ in range(local_epochs):
                for batch in loader:
                    feats = batch["features"].to(device)
                    spks = batch["speaker_ids"].to(device)
                    labels = batch["labels"].to(device)
                    mask = labels != -1

                    # Forward pass for task loss
                    out = local_model(feats, spks)
                    task_loss, _ = loss_fn(out["alpha"][mask], labels[mask])

                    # Forward pass for contrastive loss using hidden states
                    z_local = out["hidden"]   # [B, T, D]
                    with torch.no_grad():
                        out_g = global_frozen(feats, spks)
                        z_global = out_g["hidden"]
                        out_p = prev_model(feats, spks)
                        z_prev = out_p["hidden"]

                    con_loss = moon_loss(z_local, z_global, z_prev, mask)

                    total_loss = task_loss + con_loss

                    opt.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), 5.0)
                    opt.step()

            local_state_cpu = OrderedDict(
                {k_: v.cpu() for k_, v in local_model.state_dict().items()}
            )
            client_states.append(local_state_cpu)
            prev_local_states[k] = local_state_cpu  # Store for next round

            del global_frozen, prev_model

        # Standard FedAvg aggregation (MOON only modifies client-side)
        total = sum(client_sizes)
        weights = [ds / total for ds in client_sizes]
        new_global = OrderedDict()
        for key in client_states[0].keys():
            new_global[key] = sum(
                w * sd[key].float() for w, sd in zip(weights, client_states)
            )

        global_model.load_state_dict(new_global)
        global_model.to(device)

        wf1 = eval_model(global_model, test_loader, device)
        logger.info(f"  MOON Round {rnd:3d}/{num_rounds} | WF1={wf1:.4f}")

        if wf1 > best_wf1:
            best_wf1 = wf1
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= 15:
                break

        del client_states
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return best_wf1


# ============================================================
# Main entry point
# ============================================================

def run_experiment(method, dataset_name, seed, device):
    """Run one experiment: method × dataset × seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_rounds = 50
    local_epochs = 3
    num_clients = 5
    alpha = 0.5
    lr = 1e-3

    # Load data
    if dataset_name == "meld":
        train, dev, test, emotions, wts, cache, num_spk = load_meld(finetuned=True)
    elif dataset_name == "iemocap":
        train, dev, test, emotions, wts, cache, num_spk = load_iemocap(
            finetuned=True, num_classes=6
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    num_classes = len(emotions)
    class_weights = torch.from_numpy(wts.astype(np.float32)).to(device)

    # Federated partition (full labels for aggregation comparison)
    partitioner = FederatedPartitioner(
        num_clients=num_clients, strategy="dirichlet",
        alpha=alpha, seed=seed,
    )
    client_partitions = partitioner.partition(train, label_ratio=1.0)
    dialogue_lookup = {d.dialogue_id: d for d in train}

    client_loaders = []
    client_sizes = []
    for partition in client_partitions:
        dias = [dialogue_lookup[did] for did in partition.dialogue_ids
                if did in dialogue_lookup]
        ds = GenericDialogueDataset(dias, cache.get("train", {}))
        loader = DataLoader(ds, batch_size=16, shuffle=True,
                            collate_fn=collate_dialogues, num_workers=0)
        client_loaders.append(loader)
        client_sizes.append(len(dias))

    test_ds = GenericDialogueDataset(test, cache.get("test", {}))
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False,
                             collate_fn=collate_dialogues, num_workers=0)

    # Initialize global model
    global_model = EvidentialDialogueRNN(
        input_dim=768, hidden_dim=256,
        num_classes=num_classes, num_speakers=num_spk,
        dropout=0.3,
    ).to(device)

    # Run method-specific training
    runner = {
        "scaffold": run_scaffold,
        "fednova": run_fednova,
        "fedadam": run_fedadam,
        "moon": run_moon,
    }

    if method not in runner:
        raise ValueError(f"Unknown method: {method}")

    logger.info(f"\n{'='*60}")
    logger.info(f"  {method.upper()} | {dataset_name.upper()} | seed={seed}")
    logger.info(f"{'='*60}\n")

    start = time.time()
    best_wf1 = runner[method](
        global_model=global_model,
        client_loaders=client_loaders,
        test_loader=test_loader,
        client_sizes=client_sizes,
        num_classes=num_classes,
        class_weights=class_weights,
        num_speakers=num_spk,
        device=device,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        lr=lr,
    )
    elapsed = time.time() - start

    logger.info(f"\n>> {method.upper()} | {dataset_name.upper()} | seed={seed} "
                f"=> Best WF1 = {best_wf1:.4f} ({elapsed:.1f}s)\n")

    return best_wf1, elapsed


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Run modern FL aggregation baselines"
    )
    parser.add_argument("--methods", type=str,
                        default="scaffold,fednova,fedadam,moon")
    parser.add_argument("--datasets", type=str, default="meld,iemocap")
    parser.add_argument("--seeds", type=str, default="42,123,2024")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    methods = args.methods.split(",")
    datasets = args.datasets.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    results = load_results()

    for method in methods:
        for ds in datasets:
            for seed in seeds:
                key = f"{ds}_{method}_s{seed}"

                if key in results and results[key].get("wf1") is not None:
                    logger.info(f"Skipping completed: {key} "
                                f"(WF1={results[key]['wf1']})")
                    continue

                try:
                    wf1, elapsed = run_experiment(method, ds, seed, args.device)
                    results[key] = {
                        "wf1": round(float(wf1), 4),
                        "method": method,
                        "dataset": ds,
                        "seed": seed,
                        "time": round(elapsed, 1),
                    }
                    save_results(results)
                except Exception as e:
                    logger.error(f"ERROR {key}: {e}", exc_info=True)
                    results[key] = {"wf1": None, "error": str(e)}
                    save_results(results)

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  FL BASELINES — FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Method':<12} {'Dataset':<12} {'Seed':<8} {'WF1 (%)':<10} {'Time (s)':<10}")
    print(f"  {'-'*58}")
    for key, val in sorted(results.items()):
        if val.get("wf1") is not None:
            print(f"  {val['method']:<12} {val['dataset']:<12} "
                  f"{val['seed']:<8} {val['wf1']*100:<10.2f} {val.get('time', 0):<10.1f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
