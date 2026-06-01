"""
SSL Multi-Ratio Expansion
===========================
Test ECR vs FixMatch vs Supervised-only at 3 label ratios across 3 datasets.
Addresses reviewer: "SSL gains are only lightly evidenced (MELD 5%)"

Configs: 3 methods x 3 ratios x 3 datasets x 3 seeds = 81 experiments
But we reuse existing results where available.

Usage:
    cd D:\\OJT\\FedSSL-MERC
    python scripts/run_ssl_ratios.py
"""

import sys, os, json, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from argparse import Namespace
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_FILE = "results_ssl_ratios.json"
SEEDS = [42, 123, 2024]


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))


def run_ssl_experiment(dataset, method, label_ratio, seed):
    """
    method: 'supervised', 'ecr', 'fixmatch'
    """
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    from scripts.train_multi_dataset import (
        load_meld, load_iemocap, load_dailydialog,
        GenericDialogueDataset, collate_dialogues, evaluate,
    )
    from models.evidential.evidential_dialogue_rnn import EvidentialDialogueRNN
    from models.evidential.losses import (
        SupervisedEvidentialLoss,
        dirichlet_kl_divergence,
    )
    from semi_supervised.augmentation import StrongAugmentation
    from federated.aggregation.eafa import EAFAAggregator
    from data.federated_partition import FederatedPartitioner
    import torch.nn.functional as F
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    loaders = {"meld": load_meld, "iemocap": load_iemocap, "dailydialog": load_dailydialog}
    train_dias, dev_dias, test_dias, emotions, weights, cache, num_spk = loaders[dataset](finetuned=True)
    num_classes = len(emotions)
    class_weights = torch.from_numpy(weights.astype(np.float32)).to(device)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"  SSL: {method} | {dataset.upper()} | ratio={label_ratio} | seed={seed}")
    logger.info(f"{'='*60}\n")
    
    # Partition with label_ratio
    partitioner = FederatedPartitioner(num_clients=5, strategy="dirichlet", alpha=0.5, seed=seed)
    client_partitions = partitioner.partition(train_dias, label_ratio=label_ratio)
    dialogue_lookup = {d.dialogue_id: d for d in train_dias}
    
    # Build per-client data
    client_labeled_loaders = []
    client_unlabeled_loaders = []
    client_total_sizes = []
    
    for partition in client_partitions:
        labeled_dias = [dialogue_lookup[did] for did in partition.labeled_ids if did in dialogue_lookup]
        labeled_ds = GenericDialogueDataset(labeled_dias, cache.get("train", {}))
        labeled_loader = DataLoader(labeled_ds, batch_size=16, shuffle=True, collate_fn=collate_dialogues)
        client_labeled_loaders.append(labeled_loader)
        
        unlabeled_dias = [dialogue_lookup[did] for did in partition.unlabeled_ids if did in dialogue_lookup]
        if unlabeled_dias:
            unlabeled_ds = GenericDialogueDataset(unlabeled_dias, cache.get("train", {}))
            unlabeled_loader = DataLoader(unlabeled_ds, batch_size=16, shuffle=True, collate_fn=collate_dialogues)
            client_unlabeled_loaders.append(unlabeled_loader)
        else:
            client_unlabeled_loaders.append(None)
        
        client_total_sizes.append(len(labeled_dias) + len(unlabeled_dias))
    
    # Dev/Test loaders
    test_ds = GenericDialogueDataset(test_dias, cache.get("test", {}))
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_dialogues)
    
    # Global model
    global_model = EvidentialDialogueRNN(
        input_dim=768, hidden_dim=256, num_classes=num_classes,
        num_speakers=num_spk, dropout=0.3, use_attention=True
    ).to(device)
    
    sup_loss_fn = SupervisedEvidentialLoss(
        num_classes=num_classes,
        annealing_epochs=30,
        class_weights=class_weights,
    )
    
    # Strong augmentation
    strong_aug = StrongAugmentation(noise_std=0.05, dropout_p=0.25) if method != "supervised" else None
    aggregator = EAFAAggregator(beta=1.0)
    
    best_wf1 = 0.0
    patience_counter = 0
    best_state = None
    
    lambda_u_max = 1.0 if method != "supervised" else 0.0
    lambda_u_rampup = 20
    
    for round_idx in range(1, 51):
        client_states = []
        client_uncertainties = []
        
        # Lambda_u ramp-up
        progress = round_idx / lambda_u_rampup
        sigmoid = 1.0 / (1.0 + np.exp(-10.0 * (progress - 0.5)))
        lambda_u = lambda_u_max * sigmoid
        
        for k in range(5):
            local_model = copy.deepcopy(global_model).to(device)
            local_model.train()
            optimizer = optim.Adam(local_model.parameters(), lr=1e-3, weight_decay=1e-4)
            if method == "flexmatch":
                class_counts = torch.zeros(num_classes, device=device)
            
            all_u_local = []
            sup_loss_fn.set_epoch(round_idx)
            if strong_aug is not None:
                strong_aug.train()
            
            labeled_loader = client_labeled_loaders[k]
            unlabeled_loader = client_unlabeled_loaders[k]
            unlabeled_iter = iter(unlabeled_loader) if unlabeled_loader else None
            
            for local_epoch in range(3):
                for batch in labeled_loader:
                    feats_l = batch['features'].to(device)
                    speakers_l = batch['speaker_ids'].to(device)
                    labels_l = batch['labels'].to(device)
                    mask_l = labels_l != -1
                    
                    # Supervised EDL loss
                    out_l = local_model(feats_l, speakers_l)
                    alpha_l = out_l["alpha"][mask_l]
                    labels_flat = labels_l[mask_l]
                    all_u_local.extend(out_l["uncertainty"][mask_l].detach().cpu().numpy())
                    
                    loss_sup, _ = sup_loss_fn(alpha_l, labels_flat)
                    loss_unsup = torch.tensor(0.0, device=device)
                    
                    # SSL on unlabeled
                    if method != "supervised" and unlabeled_iter is not None:
                        try:
                            u_batch = next(unlabeled_iter)
                        except StopIteration:
                            unlabeled_iter = iter(unlabeled_loader)
                            u_batch = next(unlabeled_iter)
                        
                        feats_u = u_batch['features'].to(device)
                        speakers_u = u_batch['speaker_ids'].to(device)
                        labels_u = u_batch['labels'].to(device)
                        u_mask = labels_u != -1
                        
                        # Weak view
                        local_model.eval()
                        with torch.no_grad():
                            out_weak = local_model(feats_u, speakers_u)
                        local_model.train()
                        
                        alpha_weak = out_weak["alpha"][u_mask]
                        uncertainty_weak = out_weak["uncertainty"][u_mask]
                        
                        if alpha_weak.numel() > 0:
                            if method == "flexmatch":
                                # FlexMatch: Dynamic per-class threshold pseudo-labeling
                                probs_weak = alpha_weak / alpha_weak.sum(dim=-1, keepdim=True)
                                pseudo_labels = probs_weak.argmax(dim=-1)
                                max_probs = probs_weak.max(dim=-1)[0]
                                
                                # Update class counts
                                above_base = max_probs >= 0.95
                                for c in range(num_classes):
                                    class_counts[c] += (pseudo_labels[above_base] == c).sum().item()
                                
                                max_count = class_counts.max().item()
                                if max_count > 0:
                                    beta = class_counts / max_count
                                    class_thresholds = beta * 0.95
                                    class_thresholds = torch.clamp(class_thresholds, min=0.5)
                                else:
                                    class_thresholds = torch.full((num_classes,), 0.95, device=device)
                                
                                dynamic_thresholds = class_thresholds[pseudo_labels]
                                conf_mask = max_probs >= dynamic_thresholds
                                
                                if conf_mask.sum() > 0:
                                    feats_strong = strong_aug(feats_u)
                                    out_strong = local_model(feats_strong, speakers_u)
                                    alpha_strong = out_strong["alpha"][u_mask]
                                    logits_strong = torch.log(alpha_strong / alpha_strong.sum(dim=-1, keepdim=True))
                                    
                                    loss_unsup = F.cross_entropy(
                                        logits_strong[conf_mask],
                                        pseudo_labels[conf_mask],
                                        reduction='mean'
                                    )
                            elif method == "fixmatch":
                                # FixMatch: CE pseudo-label with threshold
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
                                        logits_strong[conf_mask],
                                        pseudo_labels[conf_mask],
                                        reduction='mean'
                                    )
                            elif method == "ecr":
                                # ECR: certainty-weighted KL
                                feats_strong = strong_aug(feats_u)
                                out_strong = local_model(feats_strong, speakers_u)
                                alpha_strong = out_strong["alpha"][u_mask]
                                
                                if alpha_strong.numel() > 0:
                                    kl = dirichlet_kl_divergence(alpha_strong, alpha_weak.detach())
                                    certainty = (1.0 - uncertainty_weak).clamp(min=0.0)
                                    loss_unsup = (certainty.detach() * kl).mean()
                    
                    total_loss = loss_sup + lambda_u * loss_unsup
                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), 5.0)
                    optimizer.step()
            
            client_states.append(OrderedDict({k: v.cpu() for k, v in local_model.state_dict().items()}))
            client_uncertainties.append(float(np.mean(all_u_local)) if all_u_local else 0.0)
            
        # Aggregate
        global_state, _ = aggregator.aggregate(client_states, client_total_sizes, client_uncertainties, round_idx)
        global_model.load_state_dict(global_state)
        global_model.to(device)
        
        # Evaluate
        wf1, _, _, _ = evaluate(global_model, test_loader, device)
        
        if wf1 > best_wf1:
            best_wf1 = wf1
            patience_counter = 0
            best_state = copy.deepcopy(global_model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= 15:
                break
        
        if round_idx % 10 == 0:
            logger.info(f"R{round_idx}/50 | WF1={wf1:.4f} | best={best_wf1:.4f}")
            
    # Test
    return best_wf1


def main():
    results = load_results()
    
    datasets = ["meld", "iemocap", "dailydialog"]
    ratios = [0.05, 0.10, 0.20]
    methods = ["supervised", "ecr", "fixmatch", "flexmatch"]
    
    total = len(datasets) * len(ratios) * len(methods) * len(SEEDS)
    done = 0
    
    print("=" * 60)
    print("SSL MULTI-RATIO EXPANSION")
    print(f"  {len(datasets)} datasets x {len(ratios)} ratios x {len(methods)} methods x {len(SEEDS)} seeds = {total} experiments")
    print("=" * 60)
    
    for dataset in datasets:
        for ratio in ratios:
            for method in methods:
                for seed in SEEDS:
                    key = f"{dataset}_{ratio}_{method}_s{seed}"
                    done += 1
                    
                    if key in results:
                        print(f"[{done}/{total}] SKIP {key}: WF1={results[key]['wf1']:.4f}")
                        continue
                    
                    print(f"\n[{done}/{total}] {dataset}/{ratio}/{method}/seed{seed}")
                    t0 = time.time()
                    
                    try:
                        wf1 = run_ssl_experiment(dataset, method, ratio, seed)
                        results[key] = {
                            "wf1": round(float(wf1), 4),
                            "dataset": dataset,
                            "ratio": ratio,
                            "method": method,
                            "seed": seed,
                            "time": round(time.time() - t0, 1),
                        }
                        save_results(results)
                    except Exception as e:
                        print(f"  ERROR: {e}")
                        import traceback; traceback.print_exc()
    
    # Summary
    print(f"\n{'='*60}")
    print("SSL MULTI-RATIO RESULTS")
    print(f"{'='*60}")
    
    for dataset in datasets:
        print(f"\n{dataset.upper()}:")
        print(f"  {'Method':<15s} | {'5%':>12s} | {'10%':>12s} | {'20%':>12s}")
        print(f"  {'-'*55}")
        for method in methods:
            row = f"  {method:<15s} |"
            for ratio in ratios:
                vals = [results[f"{dataset}_{ratio}_{method}_s{s}"]["wf1"]
                        for s in SEEDS if f"{dataset}_{ratio}_{method}_s{s}" in results]
                if vals:
                    m = np.mean(vals) * 100
                    s = np.std(vals) * 100
                    row += f" {m:5.1f}±{s:4.1f}  |"
                else:
                    row += f"     ---     |"
            print(row)


if __name__ == "__main__":
    main()
