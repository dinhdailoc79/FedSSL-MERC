"""
SOTA Baselines FL Runner
==========================
Run CoMPM and SPCL baselines under federated setting (FedAvg).
Same FL config as ECR experiments for fair comparison.

Experiments:
  - CoMPM-FL + FedAvg on MELD (5%, 10%, 50%)
  - CoMPM-FL + FedAvg on IEMOCAP (5%, 10%, 50%)
  - SPCL-FL + FedAvg on MELD (5%, 10%, 50%)
  - SPCL-FL + FedAvg on IEMOCAP (5%, 10%, 50%)
  Total: 12 configs x 5 seeds = 60 experiments

Usage:
    python scripts/run_sota_baselines.py
"""

import sys, os, json, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_FILE = "results_sota_baselines.json"


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2,
                 default=lambda x: float(x) if hasattr(x, 'item') else str(x))


def run_sota_fl(model_name, dataset, label_ratio, seed=42):
    """Run a SOTA baseline under FedAvg FL setting."""
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)
    
    from torch.utils.data import DataLoader
    from argparse import Namespace
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    from scripts.train_multi_dataset import (
        load_meld, load_iemocap,
        GenericDialogueDataset, collate_dialogues, evaluate,
    )
    from models.erc.sota_baselines import create_sota_model
    from data.federated_partition import FederatedPartitioner
    from sklearn.metrics import f1_score
    
    loaders = {"meld": load_meld, "iemocap": load_iemocap}
    
    args = Namespace(
        hidden_dim=256, dropout=0.3, batch_size=16, lr=1e-3,
        patience=15, num_clients=5, alpha=0.5, num_rounds=50,
        local_epochs=3,
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
    
    client_loaders = []
    client_sizes = []
    
    for partition in client_partitions:
        # For SOTA baselines: only use labeled data (supervised setting)
        labeled_dias = [dialogue_lookup[did] for did in partition.labeled_ids if did in dialogue_lookup]
        labeled_ds = GenericDialogueDataset(labeled_dias, cache.get("train", {}))
        loader = DataLoader(
            labeled_ds, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_dialogues, num_workers=0,
        )
        client_loaders.append(loader)
        client_sizes.append(len(labeled_dias))
    
    test_ds = GenericDialogueDataset(test_dias, cache.get("test", {}))
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_dialogues, num_workers=0,
    )
    
    # Create model
    global_model = create_sota_model(
        model_name,
        input_dim=768, hidden_dim=args.hidden_dim,
        num_classes=num_classes, num_speakers=num_spk,
        dropout=args.dropout,
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"  {model_name.upper()}-FL | {dataset.upper()} | label={label_ratio:.0%} | seed={seed}")
    logger.info(f"  Params: {sum(p.numel() for p in global_model.parameters()):,}")
    logger.info(f"{'='*60}\n")
    
    best_wf1, patience_cnt = 0.0, 0
    
    for round_num in range(1, args.num_rounds + 1):
        start = time.time()
        client_states = []
        
        for c_idx in range(len(client_loaders)):
            loader = client_loaders[c_idx]
            
            local_model = copy.deepcopy(global_model).to(device)
            local_model.train()
            opt = optim.Adam(local_model.parameters(), lr=args.lr, weight_decay=1e-4)
            
            for _ in range(args.local_epochs):
                for batch in loader:
                    feats = batch["features"].to(device)
                    speakers = batch["speaker_ids"].to(device)
                    labels = batch["labels"].to(device)
                    mask = labels != -1
                    
                    logits = local_model(feats, speakers)
                    logits_flat = logits[mask]
                    labels_flat = labels[mask]
                    
                    loss = criterion(logits_flat, labels_flat)
                    
                    # Add SPCL contrastive loss if applicable
                    if model_name == 'spcl' and hasattr(local_model, 'contrastive_loss'):
                        features = local_model.get_features(feats, speakers)
                        cl_loss = local_model.contrastive_loss(features, labels, mask)
                        loss = loss + 0.1 * cl_loss
                    
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), 5.0)
                    opt.step()
            
            client_states.append(
                OrderedDict({k: v.cpu() for k, v in local_model.state_dict().items()})
            )
        
        # FedAvg aggregation (standard)
        total_size = sum(client_sizes)
        avg_state = OrderedDict()
        for key in client_states[0]:
            avg_state[key] = sum(
                client_states[i][key] * (client_sizes[i] / total_size)
                for i in range(len(client_states))
            )
        
        global_model.load_state_dict(avg_state)
        global_model.to(device)
        
        # Evaluate
        global_model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                feats = batch["features"].to(device)
                speakers = batch["speaker_ids"].to(device)
                labels = batch["labels"].to(device)
                mask = labels != -1
                
                logits = global_model(feats, speakers)
                preds = logits[mask].argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels[mask].cpu().numpy())
        
        test_wf1 = f1_score(all_labels, all_preds, average="weighted")
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
    
    logger.info(f"  RESULT: {model_name.upper()}-FL | {dataset} | lr={label_ratio} | WF1={best_wf1:.4f}")
    
    return {
        "wf1": round(best_wf1, 4),
        "model": model_name,
        "dataset": dataset,
        "label_ratio": label_ratio,
        "seed": seed,
    }


def main():
    results = load_results()
    total_start = time.time()
    
    models = ['compm', 'spcl']
    datasets = ['meld', 'iemocap']
    label_ratios = [0.05, 0.10, 0.50]
    seeds = [42, 123, 456, 789, 2024]
    
    experiments = []
    for model in models:
        for ds in datasets:
            for lr in label_ratios:
                for seed in seeds:
                    key = f"{ds}_{model}_lr{lr:.2f}_s{seed}"
                    experiments.append((key, model, ds, lr, seed))
    
    total = len(experiments)
    print(f"{'='*60}")
    print(f"  SOTA Baselines FL Runner")
    print(f"  Models: {models}")
    print(f"  Datasets: {datasets}")
    print(f"  Label ratios: {label_ratios}")
    print(f"  Seeds: {seeds}")
    print(f"  Total: {total} experiments")
    print(f"{'='*60}\n")
    
    for idx, (key, model, ds, lr, seed) in enumerate(experiments):
        if key in results and results[key].get("wf1") is not None:
            print(f"[{idx+1}/{total}] SKIP {key}: WF1={results[key]['wf1']}")
            continue
        
        print(f"\n[{idx+1}/{total}] RUN {key}...")
        start = time.time()
        
        try:
            r = run_sota_fl(model, ds, lr, seed)
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
    
    # ============================
    # Summary
    # ============================
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    
    for model in models:
        print(f"\n  {model.upper()}-FL:")
        print(f"  {'Dataset':<10} {'5%':<12} {'10%':<12} {'50%':<12}")
        print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*12}")
        
        for ds in datasets:
            row = f"  {ds.upper():<10}"
            for lr in label_ratios:
                vals = []
                for seed in seeds:
                    key = f"{ds}_{model}_lr{lr:.2f}_s{seed}"
                    wf1 = results.get(key, {}).get("wf1")
                    if wf1:
                        vals.append(wf1)
                if vals:
                    m, s = np.mean(vals), np.std(vals, ddof=1) if len(vals) > 1 else 0
                    row += f" {m:.4f}+/-{s:.4f}"
                else:
                    row += f" {'N/A':<12}"
            print(row)
    
    elapsed = (time.time() - total_start) / 60
    print(f"\n  Total time: {elapsed:.1f} minutes")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
