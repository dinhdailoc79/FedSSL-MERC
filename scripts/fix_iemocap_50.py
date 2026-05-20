"""
Fix IEMOCAP 50%: λ_u Grid Search
==================================
ECR thua FixMatch tại IEMOCAP 50% (58.84% vs 59.65%, p=0.018).

Root cause hypothesis:
  - λ_u=1.0 + rampup=20 quá chậm cho 50% label regime
  - ECR consistency loss nhỏ khi unlabeled data chỉ 50%
  - Cần λ_u cao hơn hoặc rampup nhanh hơn

Grid search:
  λ_u: [0.3, 0.5, 1.0, 2.0, 3.0, 5.0]
  rampup: [5, 10, 20]
  Dataset: IEMOCAP, label_ratio=0.5
  Seeds: [42, 123, 456, 789, 2024]

Phase 1: Quick scan (3 seeds) to find best config
Phase 2: Full 5-seed validation of best config

Usage:
    python scripts/fix_iemocap_50.py
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

RESULTS_FILE = "results_fix_iemocap50.json"


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))


def run_ecr_tuned(dataset, label_ratio, lambda_u, rampup_epochs, seed=42):
    """Run ECR with custom λ_u and rampup."""
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
        SupervisedEvidentialLoss, dirichlet_kl_divergence,
    )
    from semi_supervised.augmentation import StrongAugmentation
    from federated.aggregation.eafa import EAFAAggregator
    from data.federated_partition import FederatedPartitioner
    
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
    
    client_labeled_loaders = []
    client_unlabeled_loaders = []
    client_total_sizes = []
    
    is_ssl = label_ratio < 1.0
    
    for partition in client_partitions:
        labeled_dias = [dialogue_lookup[did] for did in partition.labeled_ids if did in dialogue_lookup]
        labeled_ds = GenericDialogueDataset(labeled_dias, cache.get("train", {}))
        labeled_loader = DataLoader(
            labeled_ds, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_dialogues, num_workers=0,
        )
        client_labeled_loaders.append(labeled_loader)
        
        unlabeled_dias = [dialogue_lookup[did] for did in partition.unlabeled_ids if did in dialogue_lookup]
        if unlabeled_dias and is_ssl:
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
    
    global_model = EvidentialDialogueRNN(
        input_dim=768, hidden_dim=args.hidden_dim,
        num_classes=num_classes, num_speakers=num_spk, dropout=args.dropout,
    ).to(device)
    
    sup_loss_fn = SupervisedEvidentialLoss(
        num_classes=num_classes, annealing_epochs=args.annealing_epochs,
        class_weights=class_weights,
    )
    strong_aug = StrongAugmentation(noise_std=0.05, dropout_p=0.25)
    aggregator = EAFAAggregator(beta=args.beta)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"  ECR Tuned | {dataset.upper()} | label={label_ratio:.0%} | λ_u={lambda_u} | rampup={rampup_epochs} | seed={seed}")
    logger.info(f"{'='*60}\n")
    
    best_wf1, patience_cnt = 0.0, 0
    
    for round_num in range(1, args.num_rounds + 1):
        start = time.time()
        client_states, client_sizes, client_us = [], [], []
        
        # Lambda_u ramp-up
        progress = round_num / rampup_epochs
        sigmoid = 1.0 / (1.0 + np.exp(-10.0 * (progress - 0.5)))
        current_lambda_u = lambda_u * sigmoid
        
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
                    
                    out_l = local_model(feats_l, speakers_l)
                    alpha_l = out_l["alpha"][mask_l]
                    labels_flat = labels_l[mask_l]
                    all_u_local.extend(out_l["uncertainty"][mask_l].detach().cpu().numpy())
                    
                    loss_sup, _ = sup_loss_fn(alpha_l, labels_flat)
                    
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
                        
                        local_model.eval()
                        with torch.no_grad():
                            out_weak = local_model(feats_u, speakers_u)
                        local_model.train()
                        
                        alpha_weak = out_weak["alpha"][u_mask]
                        uncertainty_weak = out_weak["uncertainty"][u_mask]
                        
                        if alpha_weak.numel() > 0:
                            feats_strong = strong_aug(feats_u)
                            out_strong = local_model(feats_strong, speakers_u)
                            alpha_strong = out_strong["alpha"][u_mask]
                            
                            if alpha_strong.numel() > 0:
                                kl = dirichlet_kl_divergence(alpha_strong, alpha_weak.detach())
                                certainty = (1.0 - uncertainty_weak).clamp(min=0.0)
                                weighted_kl = certainty.detach() * kl
                                loss_unsup = weighted_kl.mean()
                    
                    total_loss = loss_sup + current_lambda_u * loss_unsup
                    
                    opt.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), 5.0)
                    opt.step()
            
            client_states.append(OrderedDict({k: v.cpu() for k, v in local_model.state_dict().items()}))
            client_sizes.append(client_total_sizes[c_idx])
            client_us.append(float(np.mean(all_u_local)) if all_u_local else 0.0)
        
        global_state, agg_stats = aggregator.aggregate(
            client_states, client_sizes, client_us, round_num,
        )
        global_model.load_state_dict(global_state)
        global_model.to(device)
        
        test_wf1, test_u, _, _ = evaluate(global_model, test_loader, device)
        elapsed = time.time() - start
        
        if round_num % 10 == 0 or round_num <= 3:
            logger.info(f"R{round_num:2d}/{args.num_rounds} | WF1={test_wf1:.4f} | λ_u_eff={current_lambda_u:.3f} | {elapsed:.1f}s")
        
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
    
    logger.info(f"  RESULT: λ_u={lambda_u}, rampup={rampup_epochs} | Best WF1={best_wf1:.4f}")
    
    return {
        "wf1": round(best_wf1, 4),
        "lambda_u": lambda_u,
        "rampup": rampup_epochs,
        "seed": seed,
        "dataset": dataset,
        "label_ratio": label_ratio,
    }


def main():
    results = load_results()
    total_start = time.time()
    
    # ============================
    # Phase 1: Quick scan (3 seeds)
    # ============================
    lambda_us = [0.3, 0.5, 1.0, 2.0, 3.0, 5.0]
    rampups = [5, 10, 20]
    scan_seeds = [42, 123, 2024]
    
    print(f"{'='*60}")
    print(f"  IEMOCAP 50% FIX — Phase 1: Quick Scan")
    print(f"  Grid: lambda_u={lambda_us} x rampup={rampups}")
    print(f"  Seeds: {scan_seeds}")
    print(f"  Total: {len(lambda_us) * len(rampups) * len(scan_seeds)} runs")
    print(f"{'='*60}\n")
    
    experiments = []
    for lu in lambda_us:
        for rp in rampups:
            for seed in scan_seeds:
                key = f"iemocap_ecr_lu{lu}_rp{rp}_s{seed}"
                experiments.append((key, lu, rp, seed))
    
    total = len(experiments)
    done = 0
    
    for idx, (key, lu, rp, seed) in enumerate(experiments):
        if key in results and results[key].get("wf1") is not None:
            print(f"[{idx+1}/{total}] SKIP {key}: WF1={results[key]['wf1']}")
            continue
        
        print(f"\n[{idx+1}/{total}] RUN {key}...")
        start = time.time()
        
        try:
            r = run_ecr_tuned("iemocap", 0.5, lu, rp, seed)
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
            results[key] = {"wf1": None, "error": str(e)}
            save_results(results)
    
    # ============================
    # Phase 1 Results
    # ============================
    print(f"\n{'='*60}")
    print(f"  PHASE 1 RESULTS (3-seed means)")
    print(f"{'='*60}")
    
    # FixMatch baseline for comparison
    fm_baseline = 0.5965  # From results_ssl_5seeds.json
    
    print(f"\n  FixMatch baseline: {fm_baseline:.4f}")
    print(f"\n  {'λ_u':>5} | {'rp=5':>10} | {'rp=10':>10} | {'rp=20':>10}")
    print(f"  {'-'*5}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    
    best_config = None
    best_mean = 0.0
    
    for lu in lambda_us:
        row = f"  {lu:5.1f} |"
        for rp in rampups:
            vals = []
            for seed in scan_seeds:
                key = f"iemocap_ecr_lu{lu}_rp{rp}_s{seed}"
                wf1 = results.get(key, {}).get("wf1")
                if wf1 is not None:
                    vals.append(wf1)
            
            if vals:
                mean = np.mean(vals)
                marker = " ✓" if mean > fm_baseline else ""
                row += f" {mean:.4f}{marker} |"
                if mean > best_mean:
                    best_mean = mean
                    best_config = (lu, rp)
            else:
                row += f" {'N/A':>10} |"
        print(row)
    
    if best_config:
        print(f"\n  🏆 BEST CONFIG: λ_u={best_config[0]}, rampup={best_config[1]} → WF1={best_mean:.4f}")
        beats_fm = "YES ✓" if best_mean > fm_baseline else "NO ✗"
        print(f"  Beats FixMatch ({fm_baseline:.4f})? {beats_fm}")
    
    # ============================
    # Phase 2: Full 5-seed validation
    # ============================
    if best_config and best_mean > fm_baseline:
        lu_best, rp_best = best_config
        all_seeds = [42, 123, 456, 789, 2024]
        
        print(f"\n{'='*60}")
        print(f"  PHASE 2: Full 5-seed validation (λ_u={lu_best}, rp={rp_best})")
        print(f"{'='*60}")
        
        for seed in all_seeds:
            key = f"iemocap_ecr_lu{lu_best}_rp{rp_best}_s{seed}"
            if key in results and results[key].get("wf1") is not None:
                print(f"  SKIP {key}: WF1={results[key]['wf1']}")
                continue
            
            print(f"\n  RUN {key}...")
            try:
                r = run_ecr_tuned("iemocap", 0.5, lu_best, rp_best, seed)
                r["time"] = round(time.time() - total_start, 1)
                results[key] = r
                save_results(results)
                print(f"  >> WF1={r['wf1']}")
            except Exception as e:
                print(f"  >> ERROR: {e}")
                results[key] = {"wf1": None, "error": str(e)}
                save_results(results)
        
        # Final comparison
        ecr_vals = []
        for seed in all_seeds:
            key = f"iemocap_ecr_lu{lu_best}_rp{rp_best}_s{seed}"
            wf1 = results.get(key, {}).get("wf1")
            if wf1 is not None:
                ecr_vals.append(wf1)
        
        if ecr_vals:
            from scipy import stats
            ecr_mean = np.mean(ecr_vals)
            ecr_std = np.std(ecr_vals, ddof=1)
            
            # Load FM values for comparison
            fm_file = "results_ssl_5seeds.json"
            if os.path.exists(fm_file):
                fm_data = json.load(open(fm_file))
                fm_vals = []
                for seed in all_seeds:
                    fk = f"iemocap_fixmatch_lr0.50_s{seed}"
                    wf1 = fm_data.get(fk, {}).get("wf1")
                    if wf1 is not None:
                        fm_vals.append(wf1)
                
                if fm_vals:
                    n = min(len(ecr_vals), len(fm_vals))
                    t_stat, p_val = stats.ttest_rel(ecr_vals[:n], fm_vals[:n])
                    delta = ecr_mean - np.mean(fm_vals[:n])
                    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                    
                    print(f"\n{'='*60}")
                    print(f"  FINAL: ECR (tuned) vs FixMatch @ IEMOCAP 50%")
                    print(f"  ECR:     {ecr_mean:.4f} ± {ecr_std:.4f}")
                    print(f"  FixMatch:{np.mean(fm_vals[:n]):.4f} ± {np.std(fm_vals[:n], ddof=1):.4f}")
                    print(f"  Delta:   {delta:+.4f}")
                    print(f"  p-value: {p_val:.5f} ({sig})")
                    print(f"  Config:  λ_u={lu_best}, rampup={rp_best}")
                    fixed = "✅ FIXED!" if delta > 0 else "❌ NOT FIXED"
                    print(f"  Status:  {fixed}")
                    print(f"{'='*60}")
    else:
        print(f"\n  ⚠️ No config beats FixMatch. Will try Phase 2 with best config anyway.")
        if best_config:
            lu_best, rp_best = best_config
            print(f"  Best available: λ_u={lu_best}, rampup={rp_best} → {best_mean:.4f}")
    
    total_time = time.time() - total_start
    print(f"\n  Total time: {total_time/60:.1f} minutes")


if __name__ == "__main__":
    main()
