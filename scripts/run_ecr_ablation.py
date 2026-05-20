"""
ECR Ablation Study
===================
Prove each component of ECR is necessary by removing one at a time.

Variants:
  1. ecr_full        — Full ECR (certainty-weighted KL, strong augmentation)
  2. ecr_no_certainty — Remove (1-u) weighting → uniform weight=1.0
  3. ecr_ce_pseudo    — Replace KL with CE pseudo-label (like FixMatch but with EDL backbone)
  4. ecr_no_augment   — Remove strong augmentation → both views identical

Datasets: MELD 5%, IEMOCAP 5% (hardest conditions where ECR shines)
Seeds: 42, 123, 2024
Total: 2 datasets × 4 variants × 3 seeds = 24 runs (~2-3h)

Usage:
    python scripts/run_ecr_ablation.py
"""

import sys, os, json, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_FILE = "results_ecr_ablation.json"
SEEDS = [42, 123, 2024]


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))


def run_ablation_experiment(dataset, variant, seed=42):
    """
    Run one ECR ablation experiment.
    
    Args:
        dataset: 'meld' or 'iemocap'
        variant: 'ecr_full', 'ecr_no_certainty', 'ecr_ce_pseudo', 'ecr_no_augment'
        seed: random seed
    
    Returns:
        dict with wf1, variant details, etc.
    """
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)
    
    from torch.utils.data import DataLoader
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    from scripts.train_multi_dataset import (
        load_meld, load_iemocap,
        GenericDialogueDataset, collate_dialogues, evaluate,
    )
    from models.evidential.evidential_dialogue_rnn import EvidentialDialogueRNN
    from models.evidential.losses import (
        SupervisedEvidentialLoss, FedEvidenceLoss,
        dirichlet_kl_divergence, EvidentialConsistencyRegularization,
    )
    from semi_supervised.augmentation import StrongAugmentation
    from federated.aggregation.eafa import EAFAAggregator
    from data.federated_partition import FederatedPartitioner
    
    label_ratio = 0.05  # Always 5% — hardest condition
    
    loaders = {"meld": load_meld, "iemocap": load_iemocap}
    
    args = Namespace(
        hidden_dim=256, dropout=0.3, batch_size=16, lr=1e-3,
        annealing_epochs=30, patience=15, num_clients=5,
        alpha=0.5, num_rounds=50, local_epochs=3, beta=1.0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_dir="checkpoints", seed=seed, finetuned=True,
    )
    
    device = args.device
    
    # Load data
    load_fn = loaders[dataset]
    train_dias, dev_dias, test_dias, emotions, weights, cache, num_spk = load_fn(finetuned=True)
    num_classes = len(emotions)
    class_weights = torch.from_numpy(weights.astype(np.float32)).to(device)
    
    # Partition
    partitioner = FederatedPartitioner(
        num_clients=5, strategy="dirichlet", alpha=args.alpha, seed=seed,
    )
    client_partitions = partitioner.partition(train_dias, label_ratio=label_ratio)
    dialogue_lookup = {d.dialogue_id: d for d in train_dias}
    
    # Build per-client loaders
    client_labeled_loaders = []
    client_unlabeled_loaders = []
    client_total_sizes = []
    
    for partition in client_partitions:
        labeled_dias = [dialogue_lookup[did] for did in partition.labeled_ids if did in dialogue_lookup]
        labeled_ds = GenericDialogueDataset(labeled_dias, cache.get("train", {}))
        labeled_loader = DataLoader(
            labeled_ds, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_dialogues, num_workers=0,
        )
        client_labeled_loaders.append(labeled_loader)
        
        unlabeled_dias = [dialogue_lookup[did] for did in partition.unlabeled_ids if did in dialogue_lookup]
        if unlabeled_dias:
            unlabeled_ds = GenericDialogueDataset(unlabeled_dias, cache.get("train", {}))
            unlabeled_loader = DataLoader(
                unlabeled_ds, batch_size=args.batch_size, shuffle=True,
                collate_fn=collate_dialogues, num_workers=0,
            )
            client_unlabeled_loaders.append(unlabeled_loader)
        else:
            client_unlabeled_loaders.append(None)
        
        client_total_sizes.append(len(labeled_dias) + len(unlabeled_dias))
    
    # Test loader
    test_ds = GenericDialogueDataset(test_dias, cache.get("test", {}))
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_dialogues, num_workers=0,
    )
    
    # Model (always EDL)
    global_model = EvidentialDialogueRNN(
        input_dim=768, hidden_dim=args.hidden_dim,
        num_classes=num_classes, num_speakers=num_spk, dropout=args.dropout,
    ).to(device)
    
    # Loss function (always supervised EDL)
    sup_loss_fn = SupervisedEvidentialLoss(
        num_classes=num_classes,
        annealing_epochs=args.annealing_epochs,
        class_weights=class_weights,
    )
    
    # ECR loss (for variants that use it)
    ecr_loss_fn = EvidentialConsistencyRegularization(lambda_u=1.0)
    
    # Strong augmentation (for variants that use it)
    use_augmentation = variant != "ecr_no_augment"
    strong_aug = StrongAugmentation(noise_std=0.05, dropout_p=0.25) if use_augmentation else None
    
    # Certainty weighting
    use_certainty = variant != "ecr_no_certainty"
    
    # CE pseudo-label variant
    use_ce_pseudo = variant == "ecr_ce_pseudo"
    
    # Aggregator (always EAFA)
    aggregator = EAFAAggregator(beta=args.beta)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"  ABLATION: {variant} | {dataset.upper()} | label=5% | seed={seed}")
    logger.info(f"  augmentation={use_augmentation}, certainty={use_certainty}, ce_pseudo={use_ce_pseudo}")
    logger.info(f"{'='*60}\n")
    
    # Training loop
    best_wf1, patience_cnt = 0.0, 0
    lambda_u_max = 1.0
    lambda_u_rampup = 20
    
    for round_num in range(1, args.num_rounds + 1):
        start = time.time()
        client_states, client_sizes, client_us = [], [], []
        
        # Lambda_u ramp-up
        progress = round_num / lambda_u_rampup
        sigmoid = 1.0 / (1.0 + np.exp(-10.0 * (progress - 0.5)))
        lambda_u = lambda_u_max * sigmoid
        
        for c_idx in range(len(client_labeled_loaders)):
            labeled_loader = client_labeled_loaders[c_idx]
            unlabeled_loader = client_unlabeled_loaders[c_idx]
            
            local_model = copy.deepcopy(global_model).to(device)
            local_model.train()
            opt = optim.Adam(local_model.parameters(), lr=args.lr, weight_decay=1e-4)
            all_u_local = []
            
            sup_loss_fn.set_epoch(round_num)
            if strong_aug is not None:
                strong_aug.train()
            
            for _ in range(args.local_epochs):
                unlabeled_iter = iter(unlabeled_loader) if unlabeled_loader else None
                
                for labeled_batch in labeled_loader:
                    feats_l = labeled_batch["features"].to(device)
                    speakers_l = labeled_batch["speaker_ids"].to(device)
                    labels_l = labeled_batch["labels"].to(device)
                    mask_l = labels_l != -1
                    
                    # Supervised EDL loss (same for all variants)
                    out_l = local_model(feats_l, speakers_l)
                    alpha_l = out_l["alpha"][mask_l]
                    labels_flat = labels_l[mask_l]
                    all_u_local.extend(out_l["uncertainty"][mask_l].detach().cpu().numpy())
                    
                    loss_sup, _ = sup_loss_fn(alpha_l, labels_flat)
                    
                    # Unsupervised loss (variant-dependent)
                    loss_unsup = torch.tensor(0.0, device=device)
                    
                    if unlabeled_iter is not None:
                        try:
                            u_batch = next(unlabeled_iter)
                        except StopIteration:
                            unlabeled_iter = iter(unlabeled_loader)
                            u_batch = next(unlabeled_iter)
                        
                        feats_u = u_batch["features"].to(device)
                        speakers_u = u_batch["speaker_ids"].to(device)
                        labels_u = u_batch["labels"].to(device)
                        u_mask = labels_u != -1
                        
                        # Weak view (no augmentation, no gradient)
                        local_model.eval()
                        with torch.no_grad():
                            out_weak = local_model(feats_u, speakers_u)
                        local_model.train()
                        
                        alpha_weak = out_weak["alpha"][u_mask]
                        uncertainty_weak = out_weak["uncertainty"][u_mask]
                        
                        if alpha_weak.numel() == 0:
                            loss_unsup = torch.tensor(0.0, device=device)
                        elif use_ce_pseudo:
                            # === VARIANT: CE pseudo-label (like FixMatch with EDL backbone) ===
                            # Use weak view predictions as pseudo-labels + CE loss
                            probs_weak = alpha_weak / alpha_weak.sum(dim=-1, keepdim=True)
                            pseudo_labels = probs_weak.argmax(dim=-1)
                            
                            # Confidence mask (FixMatch-style threshold)
                            max_probs = probs_weak.max(dim=-1)[0]
                            conf_mask = max_probs > 0.95
                            
                            if conf_mask.sum() > 0:
                                # Strong view
                                if use_augmentation:
                                    feats_strong = strong_aug(feats_u)
                                else:
                                    feats_strong = feats_u
                                out_strong = local_model(feats_strong, speakers_u)
                                alpha_strong = out_strong["alpha"][u_mask]
                                logits_strong = torch.log(alpha_strong / alpha_strong.sum(dim=-1, keepdim=True))
                                
                                ce_loss = F.cross_entropy(
                                    logits_strong[conf_mask],
                                    pseudo_labels[conf_mask],
                                    reduction='mean'
                                )
                                loss_unsup = ce_loss
                            else:
                                loss_unsup = torch.tensor(0.0, device=device)
                        else:
                            # === VARIANTS: ecr_full, ecr_no_certainty, ecr_no_augment ===
                            # Strong view
                            if use_augmentation:
                                feats_strong = strong_aug(feats_u)
                            else:
                                feats_strong = feats_u  # No augmentation variant
                            
                            out_strong = local_model(feats_strong, speakers_u)
                            alpha_strong = out_strong["alpha"][u_mask]
                            
                            if alpha_strong.numel() > 0:
                                # KL divergence
                                kl = dirichlet_kl_divergence(
                                    alpha_strong, alpha_weak.detach()
                                )
                                
                                if use_certainty:
                                    # Certainty-weighted (full ECR)
                                    certainty = (1.0 - uncertainty_weak).clamp(min=0.0)
                                    weighted_kl = certainty.detach() * kl
                                else:
                                    # No certainty weighting (ablation)
                                    weighted_kl = kl
                                
                                loss_unsup = weighted_kl.mean()
                            else:
                                loss_unsup = torch.tensor(0.0, device=device)
                    
                    # Combined loss
                    total_loss = loss_sup + lambda_u * loss_unsup
                    
                    opt.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), 5.0)
                    opt.step()
            
            client_states.append(OrderedDict({k: v.cpu() for k, v in local_model.state_dict().items()}))
            client_sizes.append(client_total_sizes[c_idx])
            client_us.append(float(np.mean(all_u_local)) if all_u_local else 0.0)
        
        # Aggregate
        global_state, agg_stats = aggregator.aggregate(
            client_states, client_sizes, client_us, round_num,
        )
        global_model.load_state_dict(global_state)
        global_model.to(device)
        
        # Evaluate
        test_wf1, test_u, _, _ = evaluate(global_model, test_loader, device)
        elapsed = time.time() - start
        
        logger.info(
            f"R{round_num:2d}/{args.num_rounds} | WF1={test_wf1:.4f} | {elapsed:.1f}s"
        )
        
        if test_wf1 > best_wf1:
            best_wf1 = test_wf1
            patience_cnt = 0
            logger.info(f"  >> New best! WF1={test_wf1:.4f}")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                logger.info(f"  Early stopping at round {round_num}")
                break
        
        del client_states
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"  RESULT: {variant} | {dataset.upper()} | seed={seed} | Best WF1={best_wf1:.4f}")
    logger.info(f"{'='*60}")
    
    return {
        "wf1": round(best_wf1, 4),
        "variant": variant,
        "dataset": dataset,
        "seed": seed,
        "label_ratio": label_ratio,
        "final_round": round_num,
    }


def main():
    results = load_results()
    total_start = time.time()
    
    datasets = ["meld", "iemocap"]
    variants = ["ecr_full", "ecr_no_certainty", "ecr_ce_pseudo", "ecr_no_augment"]
    
    experiments = []
    for dataset in datasets:
        for variant in variants:
            for seed in SEEDS:
                experiments.append((dataset, variant, seed))
    
    total = len(experiments)
    done = 0
    skipped = 0
    
    print(f"{'='*60}")
    print(f"  ECR Ablation Study")
    print(f"  Total experiments: {total}")
    print(f"  Variants: {variants}")
    print(f"  Seeds: {SEEDS}")
    print(f"{'='*60}\n")
    
    for idx, (dataset, variant, seed) in enumerate(experiments):
        key = f"{dataset}_{variant}_s{seed}"
        
        if key in results and results[key].get("wf1") is not None:
            skipped += 1
            print(f"[{idx+1}/{total}] SKIP {key}: WF1={results[key]['wf1']}")
            continue
        
        print(f"\n[{idx+1}/{total}] RUNNING {key}...")
        start = time.time()
        
        try:
            r = run_ablation_experiment(dataset, variant, seed=seed)
            elapsed = time.time() - start
            r["time"] = round(elapsed, 1)
            results[key] = r
            save_results(results)
            done += 1
            print(f"  >> WF1={r['wf1']}, time={elapsed:.0f}s")
        except Exception as e:
            import traceback
            print(f"  >> ERROR: {e}")
            traceback.print_exc()
            results[key] = {"wf1": None, "error": str(e), "seed": seed}
            save_results(results)
    
    # ========== Summary ==========
    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  ECR ABLATION RESULTS -- {total_time/60:.1f} minutes")
    print(f"  Done: {done}, Skipped: {skipped}")
    print(f"{'='*70}")
    
    for dataset in datasets:
        print(f"\n  {dataset.upper()} (5% label):")
        print(f"  {'Variant':<20} | {'Mean WF1':>10} | {'Std':>8} | {'vs Full':>8}")
        print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")
        
        full_vals = []
        for seed in SEEDS:
            fk = f"{dataset}_ecr_full_s{seed}"
            wf1 = results.get(fk, {}).get("wf1")
            if wf1 is not None:
                full_vals.append(wf1)
        full_mean = np.mean(full_vals) if full_vals else 0.0
        
        for variant in variants:
            vals = []
            for seed in SEEDS:
                key = f"{dataset}_{variant}_s{seed}"
                wf1 = results.get(key, {}).get("wf1")
                if wf1 is not None:
                    vals.append(wf1)
            
            if vals:
                mean = np.mean(vals)
                std = np.std(vals)
                delta = mean - full_mean
                marker = "BASE" if variant == "ecr_full" else f"{delta:+.4f}"
                print(f"  {variant:<20} | {mean:10.4f} | {std:8.4f} | {marker:>8}")
            else:
                print(f"  {variant:<20} | {'N/A':>10} | {'N/A':>8} | {'N/A':>8}")
    
    # Statistical significance (full vs each ablation)
    from scipy import stats as scipy_stats
    
    print(f"\n  STATISTICAL TESTS (ECR Full vs Ablations):")
    for dataset in datasets:
        print(f"\n  {dataset.upper()}:")
        full_vals = []
        for seed in SEEDS:
            fk = f"{dataset}_ecr_full_s{seed}"
            wf1 = results.get(fk, {}).get("wf1")
            if wf1 is not None:
                full_vals.append(wf1)
        
        for variant in variants:
            if variant == "ecr_full":
                continue
            vals = []
            for seed in SEEDS:
                key = f"{dataset}_{variant}_s{seed}"
                wf1 = results.get(key, {}).get("wf1")
                if wf1 is not None:
                    vals.append(wf1)
            
            if len(full_vals) >= 3 and len(vals) >= 3:
                n = min(len(full_vals), len(vals))
                t_stat, p_val = scipy_stats.ttest_rel(full_vals[:n], vals[:n])
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                delta = np.mean(full_vals) - np.mean(vals)
                print(f"    Full vs {variant}: delta={delta:+.4f}, p={p_val:.4f} ({sig})")
    
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
