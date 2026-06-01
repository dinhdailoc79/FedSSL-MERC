"""
ECR Augmentation Test: Feature Dropout vs Gaussian Noise
=========================================================
Quick test: Does feature dropout improve ECR over Gaussian noise?

Augmentations tested:
  - gaussian_0.01: Current best from tuning (sigma=0.01)
  - gaussian_0.05: Original default (sigma=0.05)
  - dropout_0.10:  Drop 10% of feature dimensions
  - dropout_0.15:  Drop 15%
  - dropout_0.20:  Drop 20%
  - dropout_0.30:  Drop 30%
  - combo:         Dropout 10% + Gaussian 0.01

Datasets: MELD, IEMOCAP, DailyDialog (10% labels)
Seeds: 3 × {42, 123, 456}

Total: 7 augs × 3 datasets × 3 seeds = 63 experiments (~1 hour)

Usage:
    python scripts/run_ecr_augtest.py
"""

import sys, os, json, time, copy
import numpy as np
import torch
import logging
from collections import OrderedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_FILE = "results_ecr_augtest.json"
SEEDS = [42, 123, 456, 789, 2024]

# Best ECR params from tuning
LAMBDA_MAX = 0.3
RAMP_START = 3


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))


def apply_augmentation(feats, aug_type):
    """Apply augmentation to features for strong view."""
    if aug_type == "gaussian_0.01":
        return feats + torch.randn_like(feats) * 0.01
    elif aug_type == "gaussian_0.05":
        return feats + torch.randn_like(feats) * 0.05
    elif aug_type.startswith("dropout_"):
        rate = float(aug_type.split("_")[1])
        mask = (torch.rand_like(feats) > rate).float()
        return feats * mask / (1.0 - rate)  # Scale to maintain magnitude
    elif aug_type == "combo":
        # Dropout 10% + Gaussian 0.01
        mask = (torch.rand_like(feats) > 0.10).float()
        return (feats * mask / 0.9) + torch.randn_like(feats) * 0.01
    else:
        raise ValueError(f"Unknown aug: {aug_type}")


def load_dataset(dataset):
    if dataset == "meld":
        from scripts.train_multi_dataset import load_meld
        return load_meld(finetuned=True)
    elif dataset == "iemocap":
        from scripts.train_multi_dataset import load_iemocap
        return load_iemocap(finetuned=True)
    elif dataset == "dailydialog":
        from scripts.train_multi_dataset import load_dailydialog
        return load_dailydialog(finetuned=True)


def run_experiment(dataset_data, aug_type, label_ratio=0.10, seed=42):
    logging.basicConfig(level=logging.WARNING)

    from torch.utils.data import DataLoader
    from scripts.train_multi_dataset import (
        GenericDialogueDataset, collate_dialogues, evaluate,
    )
    from models.evidential.evidential_dialogue_rnn import EvidentialDialogueRNN
    from models.evidential.losses import SupervisedEvidentialLoss
    from federated.aggregation.eafa import EAFAAggregator
    from data.federated_partition import FederatedPartitioner

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dias, dev_dias, test_dias, emotions, weights, cache, num_spk = dataset_data
    num_classes = len(emotions)
    class_weights = torch.from_numpy(weights.astype(np.float32)).to(device)

    partitioner = FederatedPartitioner(num_clients=5, strategy="dirichlet", alpha=0.5, seed=seed)
    client_partitions = partitioner.partition(train_dias, label_ratio=label_ratio)
    dialogue_lookup = {d.dialogue_id: d for d in train_dias}

    client_labeled_loaders, client_unlabeled_loaders = [], []
    for partition in client_partitions:
        labeled_dias = [dialogue_lookup[did] for did in partition.labeled_ids if did in dialogue_lookup]
        unlabeled_dias = [dialogue_lookup[did] for did in partition.unlabeled_ids if did in dialogue_lookup]

        labeled_ds = GenericDialogueDataset(labeled_dias, cache.get("train", {}))
        client_labeled_loaders.append(
            DataLoader(labeled_ds, batch_size=16, shuffle=True, collate_fn=collate_dialogues, num_workers=0))

        if unlabeled_dias:
            unlabeled_ds = GenericDialogueDataset(unlabeled_dias, cache.get("train", {}))
            client_unlabeled_loaders.append(
                DataLoader(unlabeled_ds, batch_size=16, shuffle=True, collate_fn=collate_dialogues, num_workers=0))
        else:
            client_unlabeled_loaders.append(None)

    dev_ds = GenericDialogueDataset(dev_dias, cache.get("dev", {}))
    dev_loader = DataLoader(dev_ds, batch_size=16, shuffle=False, collate_fn=collate_dialogues, num_workers=0)
    test_ds = GenericDialogueDataset(test_dias, cache.get("test", {}))
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_dialogues, num_workers=0)

    global_model = EvidentialDialogueRNN(
        input_dim=768, hidden_dim=256, num_classes=num_classes,
        num_speakers=num_spk, dropout=0.3,
    ).to(device)
    loss_fn = SupervisedEvidentialLoss(
        num_classes=num_classes, annealing_epochs=30, class_weights=class_weights,
    )
    aggregator = EAFAAggregator(beta=10.0)

    best_dev_wf1, patience_cnt, best_test_wf1 = 0.0, 0, 0.0

    for round_num in range(1, 51):
        client_states, client_sizes, client_us = [], [], []
        loss_fn.set_epoch(round_num)

        lambda_u = 0.0
        if round_num > RAMP_START:
            progress = (round_num - RAMP_START) / 20.0
            lambda_u = LAMBDA_MAX / (1.0 + np.exp(-10 * (progress - 0.5)))

        for k in range(5):
            model = copy.deepcopy(global_model)
            model.train()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            all_u_local = []

            for _ in range(3):
                for batch in client_labeled_loaders[k]:
                    feats = batch["features"].to(device)
                    speakers = batch["speaker_ids"].to(device)
                    labels = batch["labels"].to(device)
                    out = model(feats, speakers)
                    mask = labels != -1
                    sup_loss, _ = loss_fn(out["alpha"][mask], labels[mask])
                    all_u_local.extend(out["uncertainty"][mask].detach().cpu().numpy())

                    ssl_loss = torch.tensor(0.0, device=device)
                    if lambda_u > 0 and client_unlabeled_loaders[k] is not None:
                        try:
                            u_batch = next(iter(client_unlabeled_loaders[k]))
                            u_feats = u_batch["features"].to(device)
                            u_speakers = u_batch["speaker_ids"].to(device)
                            u_labels_raw = u_batch["labels"].to(device)
                            u_mask = u_labels_raw != -1

                            with torch.no_grad():
                                weak_out = model(u_feats, u_speakers)

                            # Apply augmentation
                            aug_feats = apply_augmentation(u_feats, aug_type)
                            strong_out = model(aug_feats, u_speakers)

                            alpha_w = weak_out["alpha"][u_mask]
                            alpha_s = strong_out["alpha"][u_mask]
                            u_w = weak_out["uncertainty"][u_mask]
                            certainty = (1.0 - u_w).clamp(min=0.01)

                            S_w = alpha_w.sum(dim=-1, keepdim=True)
                            S_s = alpha_s.sum(dim=-1, keepdim=True)
                            kl = (torch.lgamma(S_s) - torch.lgamma(S_w)
                                  - (torch.lgamma(alpha_s) - torch.lgamma(alpha_w)).sum(dim=-1, keepdim=True)
                                  + ((alpha_s - alpha_w) * (torch.digamma(alpha_s) - torch.digamma(S_s))).sum(dim=-1, keepdim=True))
                            ssl_loss = (certainty * kl.squeeze(-1)).mean()
                        except StopIteration:
                            pass

                    total_loss = sup_loss + lambda_u * ssl_loss
                    opt.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()

            client_states.append(OrderedDict({key: v.cpu() for key, v in model.state_dict().items()}))
            client_sizes.append(len(client_labeled_loaders[k].dataset) +
                              (len(client_unlabeled_loaders[k].dataset) if client_unlabeled_loaders[k] else 0))
            client_us.append(float(np.mean(all_u_local)) if all_u_local else 1.0)

        aggregated_state, _ = aggregator.aggregate(client_states, client_sizes, client_us, round_num)
        global_model.load_state_dict(aggregated_state)
        global_model.to(device)

        dev_wf1, _, _, _ = evaluate(global_model, dev_loader, device)
        if dev_wf1 > best_dev_wf1:
            best_dev_wf1 = dev_wf1
            patience_cnt = 0
            test_wf1, _, _, _ = evaluate(global_model, test_loader, device)
            best_test_wf1 = test_wf1
        else:
            patience_cnt += 1
        if patience_cnt > 12:
            break
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {"wf1": round(best_test_wf1, 6), "dev_wf1": round(best_dev_wf1, 6)}


def main():
    results = load_results()
    total_start = time.time()

    aug_types = [
        "gaussian_0.01",   # Best from tuning
        "gaussian_0.05",   # Original default
        "dropout_0.10",    # Feature dropout variants
        "dropout_0.15",
        "dropout_0.20",
        "dropout_0.30",
        "combo",           # Dropout + Gaussian
    ]
    datasets = ["meld", "iemocap", "dailydialog"]
    total = len(aug_types) * len(datasets) * len(SEEDS)

    print("=" * 60)
    print("  ECR Augmentation Test")
    print("  %d augmentations x %d datasets x %d seeds = %d experiments" % (
        len(aug_types), len(datasets), len(SEEDS), total))
    print("=" * 60)

    idx = 0
    for ds in datasets:
        print("\n--- %s ---" % ds.upper())
        ds_data = load_dataset(ds)

        for aug in aug_types:
            for seed in SEEDS:
                idx += 1
                key = "%s_%s_seed%d" % (ds, aug, seed)
                if key in results and results[key].get("wf1") is not None:
                    continue

                print("  [%d/%d] %s ..." % (idx, total, key), end=" ", flush=True)
                start = time.time()
                try:
                    r = run_experiment(ds_data, aug, label_ratio=0.10, seed=seed)
                    elapsed = time.time() - start
                    r.update({"time": round(elapsed, 1), "aug": aug, "dataset": ds, "seed": seed})
                    results[key] = r
                    save_results(results)
                    print("WF1=%.4f (%ds)" % (r["wf1"], elapsed))
                except Exception as e:
                    print("ERROR: %s" % e)
                    results[key] = {"wf1": None, "error": str(e)}
                    save_results(results)

    # ===== ANALYSIS =====
    total_time = time.time() - total_start
    print("\n" + "=" * 70)
    print("  RESULTS — %.1f minutes" % (total_time / 60))
    print("=" * 70)

    for ds in datasets:
        print("\n  %s:" % ds.upper())
        for aug in aug_types:
            vals = [results.get("%s_%s_seed%d" % (ds, aug, s), {}).get("wf1", 0) for s in SEEDS]
            vals = [v for v in vals if v]
            if vals:
                m = np.mean(vals)
                s = np.std(vals, ddof=1) if len(vals) > 1 else 0
                tag = " << DEFAULT" if aug == "gaussian_0.05" else (" << TUNED" if aug == "gaussian_0.01" else "")
                print("    %-18s: %.4f +/- %.4f%s" % (aug, m, s, tag))

    # Best per dataset
    print("\n" + "=" * 70)
    print("  BEST AUGMENTATION PER DATASET")
    print("=" * 70)
    for ds in datasets:
        best_aug, best_score = None, 0
        for aug in aug_types:
            vals = [results.get("%s_%s_seed%d" % (ds, aug, s), {}).get("wf1", 0) for s in SEEDS]
            vals = [v for v in vals if v]
            if vals and np.mean(vals) > best_score:
                best_score = np.mean(vals)
                best_aug = aug
        if best_aug:
            is_dropout = "dropout" in best_aug or best_aug == "combo"
            print("  %s: %s -> %.4f %s" % (ds, best_aug, best_score,
                  "(DROPOUT WINS!)" if is_dropout else "(Gaussian still best)"))

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
