"""
Severe Non-IID (alpha=0.1) Validation Sweep
===========================================
Runs federated experiments on MELD under severe Non-IID split (alpha=0.1)
vs standard Non-IID split (alpha=0.5) for both FedAvg and EAFA.

Total configurations:
- 2 alphas [0.1, 0.5]
- 2 aggregations [fedavg, eafa]
- 3 seeds [42, 123, 2024]
Total: 12 experiments.
"""

import sys
import os
import json
import time
import copy
import numpy as np
import torch
import logging
from collections import OrderedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_FILE = "results_severe_non_iid.json"
SEEDS = [42, 123, 2024]

def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_results(results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))

def run_experiment(beta, alpha_dir, seed):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    from torch.utils.data import DataLoader
    from scripts.train_multi_dataset import (
        load_meld, GenericDialogueDataset, collate_dialogues, evaluate,
    )
    from models.evidential.evidential_dialogue_rnn import EvidentialDialogueRNN
    from models.evidential.losses import SupervisedEvidentialLoss
    from federated.aggregation.eafa import EAFAAggregator
    from data.federated_partition import FederatedPartitioner

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Load MELD with finetuned text embeddings
    train_dias, dev_dias, test_dias, emotions, weights, cache, num_spk = load_meld(finetuned=True)
    num_classes = len(emotions)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_weights = torch.from_numpy(weights.astype(np.float32)).to(device)

    # Federated Partitioning
    partitioner = FederatedPartitioner(
        num_clients=5, strategy="dirichlet", alpha=alpha_dir, seed=seed,
    )
    client_partitions = partitioner.partition(train_dias, label_ratio=1.0)
    dialogue_lookup = {d.dialogue_id: d for d in train_dias}

    client_loaders = []
    for idx, partition in enumerate(client_partitions):
        dias = [dialogue_lookup[did] for did in partition.dialogue_ids if did in dialogue_lookup]
        ds = GenericDialogueDataset(dias, cache.get("train", {}))
        loader = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate_dialogues, num_workers=0)
        client_loaders.append(loader)

    # Dev + Test Loaders
    dev_ds = GenericDialogueDataset(dev_dias, cache.get("dev", {}))
    dev_loader = DataLoader(dev_ds, batch_size=16, shuffle=False, collate_fn=collate_dialogues, num_workers=0)
    test_ds = GenericDialogueDataset(test_dias, cache.get("test", {}))
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_dialogues, num_workers=0)

    # Model Initialization
    global_model = EvidentialDialogueRNN(
        input_dim=768, hidden_dim=256, num_classes=num_classes,
        num_speakers=num_spk, dropout=0.3,
    ).to(device)
    loss_fn = SupervisedEvidentialLoss(
        num_classes=num_classes, annealing_epochs=30, class_weights=class_weights,
    )
    aggregator = EAFAAggregator(beta=beta)

    is_eafa = beta > 0
    agg_label = f"EAFA(b={beta})" if is_eafa else "FedAvg"

    logger.info(f"\n{'='*60}")
    logger.info(f"  {agg_label} | MELD | alpha={alpha_dir} | seed={seed}")
    logger.info(f"{'='*60}")

    best_dev_wf1, patience_cnt = 0.0, 0
    best_test_wf1, best_test_u = 0.0, 1.0

    for round_num in range(1, 51):
        start_time = time.time()
        client_states, client_sizes, client_us = [], [], []
        loss_fn.set_epoch(round_num)

        for k, loader in enumerate(client_loaders):
            model = copy.deepcopy(global_model)
            model.train()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            all_u_local = []

            for _ in range(3):
                for batch in loader:
                    feats = batch["features"].to(device)
                    speakers = batch["speaker_ids"].to(device)
                    labels = batch["labels"].to(device)
                    out = model(feats, speakers)
                    mask = labels != -1
                    loss_val, _ = loss_fn(out["alpha"][mask], labels[mask])
                    all_u_local.extend(out["uncertainty"][mask].detach().cpu().numpy())
                    opt.zero_grad()
                    loss_val.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()

            client_states.append(OrderedDict({key: v.cpu() for key, v in model.state_dict().items()}))
            client_sizes.append(len(loader.dataset))
            client_us.append(float(np.mean(all_u_local)) if all_u_local else 1.0)

        # Aggregation
        if is_eafa:
            aggregated_state, _ = aggregator.aggregate(
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
        elapsed = time.time() - start_time

        # Evaluate
        dev_wf1, dev_u, _, _ = evaluate(global_model, dev_loader, device)
        if round_num % 10 == 0 or round_num <= 3:
            logger.info(
                f"R{round_num:2d}/50 | WF1={dev_wf1:.4f} | {elapsed:.1f}s | "
                f"u_mean={np.mean(client_us):.4f}"
            )

        if dev_wf1 > best_dev_wf1:
            best_dev_wf1 = dev_wf1
            patience_cnt = 0
            test_wf1, test_u, _, _ = evaluate(global_model, test_loader, device)
            best_test_wf1 = test_wf1
            best_test_u = test_u
        else:
            patience_cnt += 1

        if patience_cnt > 15:
            break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info(f"  RESULT: {agg_label} | alpha={alpha_dir} | seed={seed} | Best WF1={best_test_wf1:.4f}")

    return {
        "wf1": round(best_test_wf1, 6),
        "uncertainty": round(best_test_u, 6),
        "dev_wf1": round(best_dev_wf1, 6),
    }

def main():
    results = load_results()
    results = {k: v for k, v in results.items() if v.get("wf1") is not None}
    save_results(results)

    total_start = time.time()

    alphas = [0.1, 0.5]
    betas = [(1.0, "eafa"), (0.0, "fedavg")]

    experiments = []
    for alpha_dir in alphas:
        for beta, method in betas:
            for seed in SEEDS:
                key = f"meld_{method}_alpha{alpha_dir}_seed{seed}"
                experiments.append((key, beta, alpha_dir, seed))

    total = len(experiments)
    done, skipped = 0, 0

    print(f"{'='*60}")
    print(f"  SEVERE NON-IID SWEEP (alpha=0.1 vs 0.5)")
    print(f"  Total: {total} experiments")
    print(f"{'='*60}\n")

    for idx, (key, beta, alpha_dir, seed) in enumerate(experiments):
        if key in results and results[key].get("wf1") is not None:
            skipped += 1
            continue

        print(f"\n[{idx+1}/{total}] {key}...")
        start = time.time()

        try:
            r = run_experiment(beta, alpha_dir, seed)
            elapsed = time.time() - start
            r["time"] = round(elapsed, 1)
            r["beta"] = beta
            r["alpha"] = alpha_dir
            r["seed"] = seed
            results[key] = r
            save_results(results)
            done += 1
            print(f"  >> WF1={r['wf1']:.4f}, time={elapsed:.0f}s")
        except Exception as e:
            import traceback
            print(f"  >> ERROR: {e}")
            traceback.print_exc()
            results[key] = {"wf1": None, "error": str(e)}
            save_results(results)

    # Summary
    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  SEVERE NON-IID RESULTS -- Completed in {total_time/60:.1f} minutes")
    print(f"  Done: {done}, Skipped: {skipped}")
    print(f"{'='*70}\n")

    print(f"  {'Alpha':>5} | {'Method':>8} | {'Seeds WF1 Results':>30} | {'Mean±Std':>12}")
    print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*30}-+-{'-'*12}")

    for alpha_dir in alphas:
        for beta, method in betas:
            vals = []
            for seed in SEEDS:
                key = f"meld_{method}_alpha{alpha_dir}_seed{seed}"
                v = results.get(key, {}).get("wf1")
                if v is not None:
                    vals.append(v)
            
            if vals:
                mean_val = np.mean(vals)
                std_val = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
                seeds_str = ", ".join([f"{v:.4f}" for v in vals])
                print(f"  {alpha_dir:5.1f} | {method.upper():8} | {seeds_str:<30} | {mean_val:.4f}±{std_val:.4f}")

    print(f"\n{'='*70}")

if __name__ == "__main__":
    main()
