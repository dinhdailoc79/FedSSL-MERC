"""
ECR + EAFA Combined Ablation Study
=====================================
Prove that ECR and EAFA contribute independently and stack together.

Configurations (2x2 matrix):
  1. FedAvg + Supervised  — No ECR, No EAFA (pure baseline)
  2. FedAvg + ECR         — ECR only
  3. EAFA  + Supervised   — EAFA only
  4. EAFA  + ECR          — Full method (both contributions)
  5. FedAvg + FixMatch    — Alternative SSL (control)
  6. EAFA  + FixMatch     — EAFA + alternative SSL (control)

Datasets: MELD, IEMOCAP  
Label ratios: 5%, 10%
Seeds: 42, 123, 456, 789, 2024 (5 seeds)
Noise: 0%, 20% (to demonstrate EAFA benefit)
Total: 2 datasets x 6 configs x 2 noise x 5 seeds = 120 experiments

Usage:
    python scripts/run_combined_ablation.py
"""

import sys, os, json, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict
from argparse import Namespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_FILE = "results_combined_ablation.json"


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2,
                 default=lambda x: float(x) if hasattr(x, 'item') else str(x))


def run_combined(dataset, ssl_method, agg_method, noise_rate, seed=42, label_ratio=0.05):
    """
    Run one combined ablation experiment.
    
    Args:
        dataset: 'meld' or 'iemocap'
        ssl_method: 'supervised', 'fixmatch', 'ecr'
        agg_method: 'fedavg', 'eafa'
        noise_rate: 0.0 or 0.2
        seed, label_ratio: standard params
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
    from models.evidential.losses import SupervisedEvidentialLoss, dirichlet_kl_divergence
    from semi_supervised.augmentation import StrongAugmentation
    from federated.aggregation.eafa import EAFAAggregator
    from data.federated_partition import FederatedPartitioner
    from sklearn.metrics import f1_score
    
    loaders = {"meld": load_meld, "iemocap": load_iemocap}
    
    args = Namespace(
        hidden_dim=256, dropout=0.3, batch_size=16, lr=1e-3,
        annealing_epochs=30, patience=15, num_clients=5,
        alpha=0.5, num_rounds=50, local_epochs=3, beta=1.0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=seed,
    )
    device = args.device
    
    load_fn = loaders[dataset]
    train_dias, dev_dias, test_dias, emotions, weights, cache, num_spk = load_fn(finetuned=True)
    num_classes = len(emotions)
    class_weights = torch.from_numpy(weights.astype(np.float32)).to(device)
    
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
        if unlabeled_dias and ssl_method != 'supervised':
            unlabeled_ds = GenericDialogueDataset(unlabeled_dias, cache.get("train", {}))
            unlabeled_loader = DataLoader(
                unlabeled_ds, batch_size=args.batch_size, shuffle=True,
                collate_fn=collate_dialogues, num_workers=0,
            )
            client_unlabeled_loaders.append(unlabeled_loader)
        else:
            client_unlabeled_loaders.append(None)
        
        client_total_sizes.append(len(labeled_dias) + len(unlabeled_dias))
    
    test_ds = GenericDialogueDataset(test_dias, cache.get("test", {}))
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_dialogues, num_workers=0,
    )
    
    # Model
    global_model = EvidentialDialogueRNN(
        input_dim=768, hidden_dim=args.hidden_dim,
        num_classes=num_classes, num_speakers=num_spk, dropout=args.dropout,
    ).to(device)
    
    sup_loss_fn = SupervisedEvidentialLoss(
        num_classes=num_classes, annealing_epochs=args.annealing_epochs,
        class_weights=class_weights,
    )
    
    # Aggregator
    if agg_method == 'eafa':
        aggregator = EAFAAggregator(beta=args.beta)
    else:
        aggregator = None  # FedAvg
    
    strong_aug = StrongAugmentation(noise_std=0.05, dropout_p=0.25)
    
    config_name = f"{agg_method.upper()}+{ssl_method.upper()}"
    logger.info(f"\n{'='*60}")
    logger.info(f"  {config_name} | {dataset.upper()} | label={label_ratio:.0%} | noise={noise_rate:.0%} | seed={seed}")
    logger.info(f"{'='*60}\n")
    
    best_wf1, patience_cnt = 0.0, 0
    lambda_u_max = 1.0
    lambda_u_rampup = 20
    
    for round_num in range(1, args.num_rounds + 1):
        start = time.time()
        client_states, client_sizes, client_us = [], [], []
        
        # Lambda_u ramp-up
        progress = round_num / lambda_u_rampup
        sigmoid_val = 1.0 / (1.0 + np.exp(-10.0 * (progress - 0.5)))
        lambda_u = lambda_u_max * sigmoid_val
        
        for c_idx in range(len(client_labeled_loaders)):
            labeled_loader = client_labeled_loaders[c_idx]
            unlabeled_loader = client_unlabeled_loaders[c_idx]
            
            local_model = copy.deepcopy(global_model).to(device)
            local_model.train()
            opt = optim.Adam(local_model.parameters(), lr=args.lr, weight_decay=1e-4)
            all_u_local = []
            
            sup_loss_fn.set_epoch(round_num)
            strong_aug.train()
            
            for _ in range(args.local_epochs):
                unlabeled_iter = iter(unlabeled_loader) if unlabeled_loader else None
                
                for labeled_batch in labeled_loader:
                    feats_l = labeled_batch["features"].to(device)
                    speakers_l = labeled_batch["speaker_ids"].to(device)
                    labels_l = labeled_batch["labels"].to(device)
                    mask_l = labels_l != -1
                    
                    # Add label noise if needed
                    if noise_rate > 0 and mask_l.sum() > 0:
                        noise_mask = torch.rand_like(labels_l.float()) < noise_rate
                        noise_labels = torch.randint(0, num_classes, labels_l.shape, device=device)
                        labels_noisy = torch.where(noise_mask & mask_l, noise_labels, labels_l)
                    else:
                        labels_noisy = labels_l
                    
                    # Supervised loss
                    out_l = local_model(feats_l, speakers_l)
                    alpha_l = out_l["alpha"][mask_l]
                    labels_flat = labels_noisy[mask_l]
                    all_u_local.extend(out_l["uncertainty"][mask_l].detach().cpu().numpy())
                    
                    loss_sup, _ = sup_loss_fn(alpha_l, labels_flat)
                    
                    # Unsupervised loss
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
                        
                        # Weak view
                        local_model.eval()
                        with torch.no_grad():
                            out_weak = local_model(feats_u, speakers_u)
                        local_model.train()
                        
                        alpha_weak = out_weak["alpha"][u_mask]
                        uncertainty_weak = out_weak["uncertainty"][u_mask]
                        
                        if alpha_weak.numel() > 0:
                            if ssl_method == 'ecr':
                                # ECR: certainty-weighted KL consistency
                                feats_strong = strong_aug(feats_u)
                                out_strong = local_model(feats_strong, speakers_u)
                                alpha_strong = out_strong["alpha"][u_mask]
                                
                                if alpha_strong.numel() > 0:
                                    kl = dirichlet_kl_divergence(alpha_strong, alpha_weak.detach())
                                    certainty = (1.0 - uncertainty_weak).clamp(min=0.0)
                                    weighted_kl = certainty.detach() * kl
                                    loss_unsup = weighted_kl.mean()
                            
                            elif ssl_method == 'fixmatch':
                                # FixMatch: pseudo-label with threshold
                                probs_weak = alpha_weak / alpha_weak.sum(dim=-1, keepdim=True)
                                pseudo_labels = probs_weak.argmax(dim=-1)
                                max_probs = probs_weak.max(dim=-1)[0]
                                conf_mask = max_probs > 0.95
                                
                                if conf_mask.sum() > 0:
                                    feats_strong = strong_aug(feats_u)
                                    out_strong = local_model(feats_strong, speakers_u)
                                    alpha_strong = out_strong["alpha"][u_mask]
                                    logits_strong = torch.log(alpha_strong / alpha_strong.sum(dim=-1, keepdim=True))
                                    loss_unsup = F.cross_entropy(
                                        logits_strong[conf_mask], pseudo_labels[conf_mask]
                                    )
                    
                    total_loss = loss_sup + lambda_u * loss_unsup
                    
                    opt.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), 5.0)
                    opt.step()
            
            client_states.append(OrderedDict({k: v.cpu() for k, v in local_model.state_dict().items()}))
            client_sizes.append(client_total_sizes[c_idx])
            client_us.append(float(np.mean(all_u_local)) if all_u_local else 0.0)
        
        # Aggregation
        if aggregator is not None:
            global_state, agg_stats = aggregator.aggregate(
                client_states, client_sizes, client_us, round_num,
            )
        else:
            # Standard FedAvg
            total_size = sum(client_sizes)
            global_state = OrderedDict()
            for key in client_states[0]:
                global_state[key] = sum(
                    client_states[i][key] * (client_sizes[i] / total_size)
                    for i in range(len(client_states))
                )
        
        global_model.load_state_dict(global_state)
        global_model.to(device)
        
        # Evaluate
        test_wf1, test_u, _, _ = evaluate(global_model, test_loader, device)
        elapsed = time.time() - start
        
        if round_num % 10 == 0 or round_num <= 3:
            logger.info(f"R{round_num:2d}/{args.num_rounds} | WF1={test_wf1:.4f} | {elapsed:.1f}s")
        
        if test_wf1 > best_wf1:
            best_wf1 = test_wf1
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                logger.info(f"  Early stopping at round {round_num}")
                break
        
        del client_states
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    logger.info(f"  RESULT: {config_name} | {dataset} | noise={noise_rate} | WF1={best_wf1:.4f}")
    
    return {
        "wf1": round(best_wf1, 4),
        "ssl": ssl_method,
        "agg": agg_method,
        "dataset": dataset,
        "noise": noise_rate,
        "label_ratio": label_ratio,
        "seed": seed,
    }


def main():
    results = load_results()
    total_start = time.time()
    
    datasets = ['meld', 'iemocap']
    configs = [
        ('supervised', 'fedavg'),    # Pure baseline
        ('ecr',        'fedavg'),    # ECR only
        ('supervised', 'eafa'),      # EAFA only
        ('ecr',        'eafa'),      # Full (both)
        ('fixmatch',   'fedavg'),    # FixMatch control
        ('fixmatch',   'eafa'),      # EAFA + FixMatch control
    ]
    noise_rates = [0.0, 0.2]
    seeds = [42, 123, 456, 789, 2024]
    label_ratio = 0.05  # Focus on low-label
    
    experiments = []
    for ds in datasets:
        for ssl, agg in configs:
            for noise in noise_rates:
                for seed in seeds:
                    key = f"{ds}_{agg}_{ssl}_n{noise}_s{seed}"
                    experiments.append((key, ds, ssl, agg, noise, seed))
    
    total = len(experiments)
    print(f"{'='*60}")
    print(f"  ECR + EAFA Combined Ablation")
    print(f"  Configs: {len(configs)}")
    print(f"  Datasets: {datasets}")
    print(f"  Noise: {noise_rates}")
    print(f"  Seeds: {seeds}")
    print(f"  Total: {total} experiments")
    print(f"{'='*60}\n")
    
    for idx, (key, ds, ssl, agg, noise, seed) in enumerate(experiments):
        if key in results and results[key].get("wf1") is not None:
            print(f"[{idx+1}/{total}] SKIP {key}: WF1={results[key]['wf1']}")
            continue
        
        print(f"\n[{idx+1}/{total}] RUN {key}...")
        exp_start = time.time()
        
        try:
            r = run_combined(ds, ssl, agg, noise, seed, label_ratio)
            elapsed = time.time() - exp_start
            r["time"] = round(elapsed, 1)
            results[key] = r
            save_results(results)
            print(f"  >> WF1={r['wf1']}, time={elapsed:.0f}s")
        except Exception as e:
            import traceback
            print(f"  >> ERROR: {e}")
            traceback.print_exc()
            results[key] = {"wf1": None, "error": str(e)}
            save_results(results)
    
    # ==================== SUMMARY ====================
    print(f"\n{'='*70}")
    print(f"  COMBINED ABLATION RESULTS")
    print(f"{'='*70}")
    
    for ds in datasets:
        for noise in noise_rates:
            print(f"\n  {ds.upper()} | Noise={noise:.0%} | Label=5%:")
            print(f"  {'Config':<25} | {'Mean WF1':>10} | {'Std':>8} | {'vs Full':>8}")
            print(f"  {'-'*25}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")
            
            full_vals = []
            for seed in seeds:
                fk = f"{ds}_eafa_ecr_n{noise}_s{seed}"
                wf1 = results.get(fk, {}).get("wf1")
                if wf1: full_vals.append(wf1)
            full_mean = np.mean(full_vals) if full_vals else 0
            
            for ssl, agg in configs:
                vals = []
                for seed in seeds:
                    key = f"{ds}_{agg}_{ssl}_n{noise}_s{seed}"
                    wf1 = results.get(key, {}).get("wf1")
                    if wf1: vals.append(wf1)
                
                if vals:
                    m, s = np.mean(vals), np.std(vals, ddof=1) if len(vals) > 1 else 0
                    delta = m - full_mean
                    name = f"{agg.upper()}+{ssl.upper()}"
                    tag = "FULL" if (ssl == 'ecr' and agg == 'eafa') else f"{delta:+.4f}"
                    print(f"  {name:<25} | {m:10.4f} | {s:8.4f} | {tag:>8}")
    
    elapsed = (time.time() - total_start) / 60
    print(f"\n  Total time: {elapsed:.1f} minutes")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
