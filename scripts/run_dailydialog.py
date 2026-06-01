"""
P2: DailyDialog — 3rd Dataset Experiments
============================================
Run full method comparison on DailyDialog to prove ECR wins on 2/3 datasets.

Methods: FedAvg+Sup, FixMatch-FL, CoMPM-FL, SPCL-FL, EAFA+ECR
Label ratios: 5%, 10%, 50%
Seeds: 5 × {42, 123, 456, 789, 2024}

Total: 5 methods × 3 labels × 5 seeds = 75 experiments

Usage:
    python scripts/run_dailydialog.py
"""

import sys, os, json, time, copy
import numpy as np
import torch
import logging
from collections import OrderedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_FILE = "results_dailydialog.json"
SEEDS = [42, 123, 456, 789, 2024]


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))


def run_experiment(method, label_ratio, seed=42):
    """
    Run one federated experiment on DailyDialog.
    
    Methods:
        - fedavg_sup: FedAvg + Supervised only
        - fixmatch: FedAvg + FixMatch SSL
        - compm: CoMPM-FL baseline
        - spcl: SPCL-FL baseline
        - eafa_ecr: EAFA + ECR (ours)
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    from torch.utils.data import DataLoader

    torch.manual_seed(seed)
    np.random.seed(seed)

    from scripts.train_multi_dataset import (
        load_dailydialog,
        GenericDialogueDataset, collate_dialogues, evaluate,
    )
    from models.evidential.evidential_dialogue_rnn import EvidentialDialogueRNN
    from models.evidential.losses import SupervisedEvidentialLoss
    from federated.aggregation.eafa import EAFAAggregator
    from data.federated_partition import FederatedPartitioner
    from models.erc.sota_baselines import create_sota_model

    # Load DailyDialog
    train_dias, dev_dias, test_dias, emotions, weights, cache, num_spk = load_dailydialog(finetuned=True)
    num_classes = len(emotions)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_weights = torch.from_numpy(weights.astype(np.float32)).to(device)

    logger.info(f"  DailyDialog: {len(train_dias)} train, {len(dev_dias)} dev, {len(test_dias)} test, {num_classes} classes")

    # Partition with semi-supervised split
    partitioner = FederatedPartitioner(
        num_clients=5, strategy="dirichlet", alpha=0.5, seed=seed,
    )
    client_partitions = partitioner.partition(train_dias, label_ratio=label_ratio)
    dialogue_lookup = {d.dialogue_id: d for d in train_dias}

    # Create client data loaders
    client_labeled_loaders = []
    client_unlabeled_loaders = []
    client_full_loaders = []
    for partition in client_partitions:
        labeled_dias = [dialogue_lookup[did] for did in partition.labeled_ids if did in dialogue_lookup]
        unlabeled_dias = [dialogue_lookup[did] for did in partition.unlabeled_ids if did in dialogue_lookup]
        all_dias = [dialogue_lookup[did] for did in partition.dialogue_ids if did in dialogue_lookup]

        labeled_ds = GenericDialogueDataset(labeled_dias, cache.get("train", {}))
        labeled_loader = DataLoader(labeled_ds, batch_size=16, shuffle=True, collate_fn=collate_dialogues, num_workers=0)
        client_labeled_loaders.append(labeled_loader)

        if unlabeled_dias:
            unlabeled_ds = GenericDialogueDataset(unlabeled_dias, cache.get("train", {}))
            unlabeled_loader = DataLoader(unlabeled_ds, batch_size=16, shuffle=True, collate_fn=collate_dialogues, num_workers=0)
        else:
            unlabeled_loader = None
        client_unlabeled_loaders.append(unlabeled_loader)

        full_ds = GenericDialogueDataset(all_dias, cache.get("train", {}))
        full_loader = DataLoader(full_ds, batch_size=16, shuffle=True, collate_fn=collate_dialogues, num_workers=0)
        client_full_loaders.append(full_loader)

    # Dev + Test
    dev_ds = GenericDialogueDataset(dev_dias, cache.get("dev", {}))
    dev_loader = DataLoader(dev_ds, batch_size=16, shuffle=False, collate_fn=collate_dialogues, num_workers=0)
    test_ds = GenericDialogueDataset(test_dias, cache.get("test", {}))
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_dialogues, num_workers=0)

    # Model — use proper architecture per method
    use_eafa = method == "eafa_ecr"
    use_sota_model = method in ["compm", "spcl"]
    beta = 10.0 if use_eafa else 0.0

    if use_sota_model:
        # CoMPM / SPCL: use their own architectures with standard CE loss
        global_model = create_sota_model(
            method, input_dim=768, hidden_dim=256,
            num_classes=num_classes, num_speakers=num_spk, dropout=0.3,
        ).to(device)
        ce_loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
        loss_fn = None  # Not used for SOTA baselines
    else:
        # FedAvg / FixMatch / ECR: use EvidentialDialogueRNN
        global_model = EvidentialDialogueRNN(
            input_dim=768, hidden_dim=256, num_classes=num_classes,
            num_speakers=num_spk, dropout=0.3,
        ).to(device)
        loss_fn = SupervisedEvidentialLoss(
            num_classes=num_classes, annealing_epochs=30, class_weights=class_weights,
        )
        ce_loss_fn = None
    aggregator = EAFAAggregator(beta=beta)

    logger.info(f"\n{'='*60}")
    logger.info(f"  {method.upper()} | DailyDialog | labels={label_ratio:.0%} | seed={seed}")
    logger.info(f"{'='*60}")

    best_dev_wf1, patience_cnt = 0.0, 0
    best_test_wf1, best_test_u = 0.0, 1.0

    # --- Evaluation helper for SOTA models (CE-based, no uncertainty) ---
    def evaluate_sota(model, loader, device):
        """Evaluate a SOTA baseline model that outputs logits (not Dirichlet)."""
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                feats = batch["features"].to(device)
                speakers = batch["speaker_ids"].to(device)
                labels = batch["labels"].to(device)
                logits = model(feats, speakers)  # (B, T, C)
                mask = labels != -1
                preds = logits[mask].argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels[mask].cpu().numpy())
        from sklearn.metrics import f1_score
        wf1 = f1_score(all_labels, all_preds, average='weighted')
        return wf1, 0.5, None, None  # Return dummy uncertainty for compatibility

    for round_num in range(1, 51):
        start = time.time()
        client_states, client_sizes, client_us = [], [], []
        if loss_fn is not None:
            loss_fn.set_epoch(round_num)

        # SSL ramp-up (only for FixMatch / ECR)
        lambda_u = 0.0
        if method in ["fixmatch", "eafa_ecr"] and round_num > 5:
            lambda_u = 1.0 / (1.0 + np.exp(-10 * (round_num / 20.0 - 0.5)))

        # Contrastive loss weight for SPCL (ramp up)
        lambda_cl = 0.0
        if method == "spcl" and round_num > 3:
            lambda_cl = min(0.1, 0.1 * (round_num - 3) / 10.0)

        for k in range(5):
            model = copy.deepcopy(global_model)
            model.train()
            lr = 5e-4 if method == "compm" else 1e-3  # CoMPM uses lower lr (transformer)
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            all_u_local = []

            local_epochs = 4 if method == "compm" else 3  # CoMPM benefits from more local epochs
            for _ in range(local_epochs):
                # Supervised on labeled data
                for batch in client_labeled_loaders[k]:
                    feats = batch["features"].to(device)
                    speakers = batch["speaker_ids"].to(device)
                    labels = batch["labels"].to(device)
                    mask = labels != -1

                    if use_sota_model:
                        # --- CoMPM / SPCL path: logits + CE loss ---
                        logits = model(feats, speakers)  # (B, T, C)
                        logits_flat = logits[mask]
                        labels_flat = labels[mask]
                        sup_loss = ce_loss_fn(logits_flat, labels_flat)

                        # SPCL: add contrastive loss
                        aux_loss = torch.tensor(0.0, device=device)
                        if method == "spcl" and lambda_cl > 0:
                            features_enc = model.get_features(feats, speakers)
                            aux_loss = model.contrastive_loss(features_enc, labels, mask)

                        total_loss = sup_loss + lambda_cl * aux_loss
                    else:
                        # --- FedAvg / FixMatch / ECR path: EDL + uncertainty ---
                        out = model(feats, speakers)
                        sup_loss, _ = loss_fn(out["alpha"][mask], labels[mask])
                        all_u_local.extend(out["uncertainty"][mask].detach().cpu().numpy())

                        # SSL loss on unlabeled data
                        ssl_loss = torch.tensor(0.0, device=device)
                        if lambda_u > 0 and client_unlabeled_loaders[k] is not None:
                            try:
                                u_batch = next(iter(client_unlabeled_loaders[k]))
                                u_feats = u_batch["features"].to(device)
                                u_speakers = u_batch["speaker_ids"].to(device)
                                u_labels_raw = u_batch["labels"].to(device)
                                u_mask = u_labels_raw != -1

                                if method == "eafa_ecr":
                                    # ECR: KL between weak/strong Dirichlet views
                                    with torch.no_grad():
                                        weak_out = model(u_feats, u_speakers)
                                    noisy_feats = u_feats + torch.randn_like(u_feats) * 0.05
                                    strong_out = model(noisy_feats, u_speakers)

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

                                elif method == "fixmatch":
                                    with torch.no_grad():
                                        weak_out = model(u_feats, u_speakers)
                                    probs = weak_out["belief"][u_mask]
                                    max_probs, pseudo_labels = probs.max(dim=-1)
                                    threshold_mask = max_probs > 0.95
                                    if threshold_mask.sum() > 0:
                                        noisy_feats = u_feats + torch.randn_like(u_feats) * 0.05
                                        strong_out = model(noisy_feats, u_speakers)
                                        strong_probs = strong_out["belief"][u_mask]
                                        ssl_loss = torch.nn.functional.cross_entropy(
                                            strong_probs[threshold_mask],
                                            pseudo_labels[threshold_mask],
                                        )
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

        # Aggregate (FedAvg for all non-EAFA methods)
        if use_eafa:
            aggregated_state, agg_stats = aggregator.aggregate(
                client_states, client_sizes, client_us, round_num,
            )
        else:
            total_sz = sum(client_sizes)
            aggregated_state = OrderedDict()
            for key in client_states[0]:
                aggregated_state[key] = sum(
                    client_states[i][key] * (client_sizes[i] / total_sz)
                    for i in range(len(client_states))
                )

        global_model.load_state_dict(aggregated_state)
        global_model.to(device)
        elapsed = time.time() - start

        # Evaluate — use appropriate evaluator
        if use_sota_model:
            dev_wf1, dev_u, _, _ = evaluate_sota(global_model, dev_loader, device)
        else:
            dev_wf1, dev_u, _, _ = evaluate(global_model, dev_loader, device)

        if round_num % 10 == 0 or round_num <= 3:
            logger.info(f"R{round_num:2d}/50 | WF1={dev_wf1:.4f} | {elapsed:.1f}s")

        if dev_wf1 > best_dev_wf1:
            best_dev_wf1 = dev_wf1
            patience_cnt = 0
            if use_sota_model:
                test_wf1, test_u, _, _ = evaluate_sota(global_model, test_loader, device)
            else:
                test_wf1, test_u, _, _ = evaluate(global_model, test_loader, device)
            best_test_wf1 = test_wf1
            best_test_u = test_u
        else:
            patience_cnt += 1

        if patience_cnt > 15:
            break
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    logger.info(f"  RESULT: {method} | DailyDialog | labels={label_ratio} | WF1={best_test_wf1:.4f}")

    return {
        "wf1": round(best_test_wf1, 6),
        "uncertainty": round(best_test_u, 6),
        "dev_wf1": round(best_dev_wf1, 6),
    }


def main():
    results = load_results()
    total_start = time.time()

    methods = ["fedavg_sup", "fixmatch", "compm", "spcl", "eafa_ecr"]
    label_ratios = [0.05, 0.10, 0.50]

    experiments = []
    for method in methods:
        for lr in label_ratios:
            for seed in SEEDS:
                key = f"dd_{method}_l{lr}_s{seed}"
                experiments.append((key, method, lr, seed))

    total = len(experiments)
    done, skipped = 0, 0

    print(f"{'='*60}")
    print(f"  P2: DailyDialog Full Comparison")
    print(f"  Methods: {methods}")
    print(f"  Total: {total} experiments")
    print(f"{'='*60}\n")

    for idx, (key, method, lr, seed) in enumerate(experiments):
        if key in results and results[key].get("wf1") is not None:
            skipped += 1
            continue

        print(f"\n[{idx+1}/{total}] {key}...")
        start = time.time()

        try:
            r = run_experiment(method, lr, seed)
            elapsed = time.time() - start
            r["time"] = round(elapsed, 1)
            r["method"] = method
            r["label_ratio"] = lr
            r["seed"] = seed
            results[key] = r
            save_results(results)
            done += 1
            print(f"  >> WF1={r['wf1']}, time={elapsed:.0f}s")
        except Exception as e:
            import traceback
            print(f"  >> ERROR: {e}")
            traceback.print_exc()
            results[key] = {"wf1": None, "error": str(e)}
            save_results(results)

    # ========== Summary ==========
    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  DAILYDIALOG RESULTS -- {total_time/60:.1f} minutes")
    print(f"  Done: {done}, Skipped: {skipped}")
    print(f"{'='*70}")

    print(f"\n  {'Method':<15} | {'5%':>12} | {'10%':>12} | {'50%':>12}")
    print(f"  {'-'*15}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

    for method in methods:
        row = f"  {method:<15}"
        for lr in label_ratios:
            vals = []
            for seed in SEEDS:
                k = f"dd_{method}_l{lr}_s{seed}"
                v = results.get(k, {}).get("wf1")
                if v is not None:
                    vals.append(v)
            if vals:
                m = np.mean(vals)
                s = np.std(vals, ddof=1) if len(vals) > 1 else 0
                row += f" | {m:.4f}±{s:.4f}"
            else:
                row += f" | {'N/A':>12}"
        print(row)

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
