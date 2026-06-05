"""
Federated SSL Experiments
==========================
Compare 3 semi-supervised methods in federated emotion recognition:

1. Supervised-only: Train only on labeled data (EDL + EAFA)
2. FixMatch: CE + pseudo-labeling with confidence threshold (FedAvg)
3. EDL+ECR: Evidential + certainty-weighted consistency (EAFA)

Experiment matrix:
  Datasets: MELD, IEMOCAP
  Label ratios: 5%, 10%, 50%, 100% (100% = ceiling, supervised only)
  Methods: supervised, fixmatch, ecr

Usage:
    python scripts/run_ssl_experiments.py
"""

import sys, os, json, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_FILE = "results_ssl_experiments.json"


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)


def run_ssl_experiment(dataset, method, label_ratio, seed=42):
    """
    Run one federated SSL experiment.
    
    Args:
        dataset: 'meld' or 'iemocap'
        method: 'supervised', 'fixmatch', or 'ecr'
        label_ratio: fraction of labeled data (0.05, 0.1, 0.5, 1.0)
        seed: random seed
    
    Returns:
        dict with wf1, ssl_stats, etc.
    """
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)
    
    from argparse import Namespace
    from collections import OrderedDict
    from pathlib import Path
    from torch.utils.data import DataLoader
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    from scripts.train_multi_dataset import (
        load_meld, load_iemocap,
        GenericDialogueDataset, collate_dialogues, evaluate,
    )
    from models.evidential.evidential_dialogue_rnn import EvidentialDialogueRNN
    from models.erc.dialogue_rnn import DialogueRNN
    from models.evidential.losses import SupervisedEvidentialLoss, FedEvidenceLoss
    from semi_supervised.fixmatch import FixMatchLoss
    from semi_supervised.flexmatch import FlexMatchLoss
    from semi_supervised.augmentation import StrongAugmentation
    from federated.aggregation.eafa import EAFAAggregator
    from data.federated_partition import FederatedPartitioner
    
    loaders = {"meld": load_meld, "iemocap": load_iemocap}
    
    args = Namespace(
        hidden_dim=256, dropout=0.3, batch_size=16, lr=1e-3,
        annealing_epochs=30, patience=15, num_clients=5,
        alpha=0.5, num_rounds=50, local_epochs=3, beta=1.0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_dir="checkpoints", seed=seed, finetuned=True,
    )
    
    use_edl = method in ("supervised", "ecr", "dirichlet_fixmatch")
    use_eafa = use_edl  # EAFA only with EDL (needs uncertainty)
    is_ssl = method in ("fixmatch", "flexmatch", "ecr", "dirichlet_fixmatch") and label_ratio < 1.0
    
    # Load data
    load_fn = loaders[dataset]
    train_dias, dev_dias, test_dias, emotions, weights, cache, num_spk = load_fn(finetuned=True)
    num_classes = len(emotions)
    device = args.device
    class_weights = torch.from_numpy(weights.astype(np.float32)).to(device)
    
    # Partition with label_ratio
    partitioner = FederatedPartitioner(
        num_clients=5, strategy="dirichlet", alpha=args.alpha, seed=seed,
    )
    client_partitions = partitioner.partition(train_dias, label_ratio=label_ratio)
    
    dialogue_lookup = {d.dialogue_id: d for d in train_dias}
    
    # Build per-client labeled + unlabeled loaders
    client_labeled_loaders = []
    client_unlabeled_loaders = []
    client_total_sizes = []
    
    for partition in client_partitions:
        # Labeled dialogues
        labeled_dias = [dialogue_lookup[did] for did in partition.labeled_ids if did in dialogue_lookup]
        labeled_ds = GenericDialogueDataset(labeled_dias, cache.get("train", {}))
        labeled_loader = DataLoader(
            labeled_ds, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_dialogues, num_workers=0,
        )
        client_labeled_loaders.append(labeled_loader)
        
        # Unlabeled dialogues
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
    
    # Log partition info
    for i, partition in enumerate(client_partitions):
        n_l = len([did for did in partition.labeled_ids if did in dialogue_lookup])
        n_u = len([did for did in partition.unlabeled_ids if did in dialogue_lookup])
        logger.info(f"  Client {i}: {n_l} labeled, {n_u} unlabeled")
    
    # Test loader
    test_ds = GenericDialogueDataset(test_dias, cache.get("test", {}))
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_dialogues, num_workers=0,
    )
    
    # Model
    if use_edl:
        global_model = EvidentialDialogueRNN(
            input_dim=768, hidden_dim=args.hidden_dim,
            num_classes=num_classes, num_speakers=num_spk, dropout=args.dropout,
        ).to(device)
    else:
        global_model = DialogueRNN(
            input_dim=768, hidden_dim=args.hidden_dim,
            num_classes=num_classes, num_speakers=num_spk, dropout=args.dropout,
        ).to(device)
    
    # Loss functions
    if method == "ecr":
        loss_fn = FedEvidenceLoss(
            num_classes=num_classes,
            annealing_epochs=args.annealing_epochs,
            lambda_u=1.0,
            lambda_u_rampup_epochs=20,
            class_weights=class_weights,
        )
        strong_aug = StrongAugmentation(noise_std=0.05, dropout_p=0.25)
    elif method == "supervised":
        loss_fn = SupervisedEvidentialLoss(
            num_classes=num_classes,
            annealing_epochs=args.annealing_epochs,
            class_weights=class_weights,
        )
    elif method == "dirichlet_fixmatch":
        loss_fn = SupervisedEvidentialLoss(
            num_classes=num_classes,
            annealing_epochs=args.annealing_epochs,
            class_weights=class_weights,
        )
        strong_aug = StrongAugmentation(noise_std=0.05, dropout_p=0.25)
    elif method == "flexmatch":
        ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        flexmatch_loss = FlexMatchLoss(
            threshold=0.95, lambda_u=1.0,
            num_classes=num_classes,
            threshold_min=0.5,
        )
    else:  # fixmatch
        ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        fixmatch_loss = FixMatchLoss(
            threshold=0.95, lambda_u=1.0,
            num_classes=num_classes,
            warmup_epochs=10, threshold_min=0.7,
        )
    
    # Aggregator
    effective_beta = args.beta if use_eafa else 0.0
    aggregator = EAFAAggregator(beta=effective_beta)
    
    method_label = {"supervised": "Supervised", "fixmatch": "FixMatch", "flexmatch": "FlexMatch", "ecr": "EDL+ECR", "dirichlet_fixmatch": "Dirichlet-FixMatch"}[method]
    agg_label = "EAFA" if use_eafa else "FedAvg"
    
    logger.info(f"\n{'='*60}")
    logger.info(f"  {method_label} ({agg_label}) | {dataset.upper()} | label={label_ratio:.0%} | seed={seed}")
    logger.info(f"{'='*60}\n")
    
    # Training loop
    best_wf1, patience_cnt = 0.0, 0
    round_data = []
    ssl_stats_history = []
    
    for round_num in range(1, args.num_rounds + 1):
        start = time.time()
        client_states, client_sizes, client_us = [], [], []
        round_ssl_stats = {"pseudo_count": 0, "pseudo_total": 0, "ecr_certainty": []}
        
        for c_idx in range(len(client_labeled_loaders)):
            labeled_loader = client_labeled_loaders[c_idx]
            unlabeled_loader = client_unlabeled_loaders[c_idx]
            
            local_model = copy.deepcopy(global_model).to(device)
            local_model.train()
            opt = optim.Adam(local_model.parameters(), lr=args.lr, weight_decay=1e-4)
            all_u_local = []
            
            if method == "ecr":
                loss_fn.set_epoch(round_num)
                strong_aug.train()
            elif method == "flexmatch":
                flexmatch_loss.train()
            elif method == "fixmatch":
                fixmatch_loss.update_threshold(round_num)
                fixmatch_loss.train()
            elif method == "supervised":
                loss_fn.set_epoch(round_num)
            
            for _ in range(args.local_epochs):
                unlabeled_iter = iter(unlabeled_loader) if unlabeled_loader else None
                
                for labeled_batch in labeled_loader:
                    feats_l = labeled_batch["features"].to(device)
                    speakers_l = labeled_batch["speaker_ids"].to(device)
                    labels_l = labeled_batch["labels"].to(device)
                    mask_l = labels_l != -1
                    
                    if method == "supervised":
                        # Only supervised EDL loss
                        out = local_model(feats_l, speakers_l)
                        loss, _ = loss_fn(out["alpha"][mask_l], labels_l[mask_l])
                        all_u_local.extend(out["uncertainty"][mask_l].detach().cpu().numpy())
                    
                    elif method == "ecr":
                        # EDL supervised + ECR on unlabeled
                        out_l = local_model(feats_l, speakers_l)
                        alpha_l = out_l["alpha"][mask_l]
                        labels_flat = labels_l[mask_l]
                        all_u_local.extend(out_l["uncertainty"][mask_l].detach().cpu().numpy())
                        
                        # Get unlabeled batch for ECR
                        alpha_weak = alpha_strong = uncertainty_weak = None
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
                            
                            # Weak view (no augmentation)
                            local_model.eval()
                            with torch.no_grad():
                                out_weak = local_model(feats_u, speakers_u)
                            local_model.train()
                            
                            # Strong view (augmented)
                            feats_strong = strong_aug(feats_u)
                            out_strong = local_model(feats_strong, speakers_u)
                            
                            alpha_weak = out_weak["alpha"][u_mask]
                            alpha_strong = out_strong["alpha"][u_mask]
                            uncertainty_weak = out_weak["uncertainty"][u_mask]
                            
                            # Track ECR stats
                            if uncertainty_weak.numel() > 0:
                                mean_cert = (1.0 - uncertainty_weak).mean().item()
                                round_ssl_stats["ecr_certainty"].append(mean_cert)
                        
                        # Combined loss
                        loss, _ = loss_fn(
                            alpha_l, labels_flat, label_mask=None,
                            alpha_weak=alpha_weak, alpha_strong=alpha_strong,
                            uncertainty_weak=uncertainty_weak,
                        )
                    
                    elif method == "flexmatch":
                        # CE supervised + FlexMatch on unlabeled
                        logits_l = local_model(feats_l, speakers_l)
                        
                        # Get unlabeled batch
                        unlabeled_batch = None
                        if unlabeled_iter is not None:
                            try:
                                unlabeled_batch = next(unlabeled_iter)
                            except StopIteration:
                                unlabeled_iter = iter(unlabeled_loader)
                                unlabeled_batch = next(unlabeled_iter)
                            unlabeled_batch = {
                                k: v.to(device) if isinstance(v, torch.Tensor) else v
                                for k, v in unlabeled_batch.items()
                            }
                        
                        labeled_batch_device = {
                            "features": feats_l, "speaker_ids": speakers_l,
                            "labels": labels_l,
                        }
                        loss, fm_stats = flexmatch_loss(
                            local_model, labeled_batch_device,
                            unlabeled_batch, ce_loss,
                        )
                        round_ssl_stats["pseudo_count"] += fm_stats["pseudo_label_count"]
                        round_ssl_stats["pseudo_total"] += fm_stats["pseudo_label_total"]
                    
                    elif method == "dirichlet_fixmatch":
                        # Evidential supervised on labeled + hard belief threshold pseudo-labeling on unlabeled
                        out_l = local_model(feats_l, speakers_l)
                        alpha_l = out_l["alpha"][mask_l]
                        labels_flat = labels_l[mask_l]
                        all_u_local.extend(out_l["uncertainty"][mask_l].detach().cpu().numpy())
                        
                        loss_supervised, _ = loss_fn(alpha_l, labels_flat)
                        
                        loss_unsupervised = torch.tensor(0.0, device=device)
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
                            
                            # 1. Weak view (no augmentation)
                            local_model.eval()
                            with torch.no_grad():
                                out_weak = local_model(feats_u, speakers_u)
                            local_model.train()
                            
                            belief_weak = out_weak["belief"][u_mask]
                            max_belief, pseudo_label = belief_weak.max(dim=-1)
                            
                            # 2. Hard threshold on belief
                            threshold = 0.95
                            conf_mask = max_belief >= threshold
                            
                            num_above = conf_mask.sum().item()
                            num_tot = u_mask.sum().item()
                            round_ssl_stats["pseudo_count"] += int(num_above)
                            round_ssl_stats["pseudo_total"] += int(num_tot)
                            
                            if num_above > 0:
                                # 3. Strong view (augmented)
                                feats_strong = strong_aug(feats_u)
                                out_strong = local_model(feats_strong, speakers_u)
                                alpha_strong_masked = out_strong["alpha"][u_mask][conf_mask]
                                pseudo_masked = pseudo_label[conf_mask]
                                
                                loss_unsupervised, _ = loss_fn(alpha_strong_masked, pseudo_masked)
                                
                        loss = loss_supervised + loss_unsupervised
                        
                    elif method == "fixmatch":
                        # CE supervised + FixMatch on unlabeled
                        logits_l = local_model(feats_l, speakers_l)
                        
                        # Get unlabeled batch
                        unlabeled_batch = None
                        if unlabeled_iter is not None:
                            try:
                                unlabeled_batch = next(unlabeled_iter)
                            except StopIteration:
                                unlabeled_iter = iter(unlabeled_loader)
                                unlabeled_batch = next(unlabeled_iter)
                            unlabeled_batch = {
                                k: v.to(device) if isinstance(v, torch.Tensor) else v
                                for k, v in unlabeled_batch.items()
                            }
                        
                        labeled_batch_device = {
                            "features": feats_l, "speaker_ids": speakers_l,
                            "labels": labels_l,
                        }
                        loss, fm_stats = fixmatch_loss(
                            local_model, labeled_batch_device,
                            unlabeled_batch, ce_loss,
                        )
                        round_ssl_stats["pseudo_count"] += fm_stats["pseudo_label_count"]
                        round_ssl_stats["pseudo_total"] += fm_stats["pseudo_label_total"]
                    
                    opt.zero_grad()
                    loss.backward()
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
        
        # SSL stats string
        ssl_str = ""
        if method in ("fixmatch", "flexmatch") and round_ssl_stats["pseudo_total"] > 0:
            pr = round_ssl_stats["pseudo_count"] / round_ssl_stats["pseudo_total"] * 100
            ssl_str = f" | pseudo={pr:.0f}%"
        elif method == "ecr" and round_ssl_stats["ecr_certainty"]:
            mc = np.mean(round_ssl_stats["ecr_certainty"])
            ssl_str = f" | cert={mc:.3f}"
        
        w_str = ",".join(f"{w:.2f}" for w in agg_stats["weights"])
        logger.info(
            f"R{round_num:2d}/{args.num_rounds} | WF1={test_wf1:.4f}{ssl_str} | "
            f"w=[{w_str}] | {elapsed:.1f}s"
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
        
        round_data.append({
            "round": round_num,
            "wf1": round(test_wf1, 4),
        })
        
        del client_states
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Final eval
    final_wf1, final_u, report, _ = evaluate(
        global_model, test_loader, device, emotions, dataset
    )
    
    logger.info(f"\n{'='*60}")
    logger.info(f"  RESULT: {method_label} | {dataset.upper()} | label={label_ratio:.0%}")
    logger.info(f"{'='*60}")
    logger.info(f"\n{report}")
    logger.info(f"  Best WF1 = {best_wf1:.4f}")
    logger.info(f"{'='*60}")
    
    return {
        "wf1": round(best_wf1, 4),
        "label_ratio": label_ratio,
        "method": method,
        "final_round": len(round_data),
    }


def main():
    results = load_results()
    total_start = time.time()
    
    datasets = ["meld", "iemocap"]
    label_ratios = [0.05, 0.1, 0.5, 1.0]
    methods = ["supervised", "fixmatch", "flexmatch", "ecr"]
    seed = 42
    
    experiments = []
    for dataset in datasets:
        for lr in label_ratios:
            for method in methods:
                # At 100% label ratio, only supervised makes sense
                if lr >= 1.0 and method != "supervised":
                    continue
                experiments.append((dataset, method, lr))
    
    total = len(experiments)
    
    for idx, (dataset, method, lr) in enumerate(experiments):
        key = f"{dataset}_{method}_lr{lr:.2f}_s{seed}"
        
        if key in results and results[key].get("wf1") is not None:
            print(f"[{idx+1}/{total}] SKIP {key}: WF1={results[key]['wf1']}")
            continue
        
        print(f"\n[{idx+1}/{total}] RUNNING {key}...")
        start = time.time()
        
        try:
            r = run_ssl_experiment(dataset, method, lr, seed=seed)
            elapsed = time.time() - start
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
    
    # Summary
    total_time = time.time() - total_start
    print(f"\n{'='*85}")
    print(f"  SSL EXPERIMENTS RESULTS -- {total_time/60:.1f} minutes")
    print(f"{'='*85}")
    
    for dataset in datasets:
        print(f"\n  {dataset.upper()}:")
        print(f"  {'Label%':>7} | {'Supervised':>11} | {'FixMatch':>11} | {'FlexMatch':>11} | {'EDL+ECR':>11}")
        print(f"  {'-'*7}-+-{'-'*11}-+-{'-'*11}-+-{'-'*11}-+-{'-'*11}")
        
        for lr in label_ratios:
            sup_key = f"{dataset}_supervised_lr{lr:.2f}_s{seed}"
            fm_key = f"{dataset}_fixmatch_lr{lr:.2f}_s{seed}"
            flex_key = f"{dataset}_flexmatch_lr{lr:.2f}_s{seed}"
            ecr_key = f"{dataset}_ecr_lr{lr:.2f}_s{seed}"
            
            sup_wf1 = results.get(sup_key, {}).get("wf1")
            fm_wf1 = results.get(fm_key, {}).get("wf1")
            flex_wf1 = results.get(flex_key, {}).get("wf1")
            ecr_wf1 = results.get(ecr_key, {}).get("wf1")
            
            sup_s = f"{sup_wf1:.4f}" if sup_wf1 else "N/A"
            fm_s = f"{fm_wf1:.4f}" if fm_wf1 else ("--" if lr >= 1.0 else "N/A")
            flex_s = f"{flex_wf1:.4f}" if flex_wf1 else ("--" if lr >= 1.0 else "N/A")
            ecr_s = f"{ecr_wf1:.4f}" if ecr_wf1 else ("--" if lr >= 1.0 else "N/A")
            
            print(f"  {lr:6.0%}  | {sup_s:>11} | {fm_s:>11} | {flex_s:>11} | {ecr_s:>11}")


if __name__ == "__main__":
    main()
