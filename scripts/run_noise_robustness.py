"""
Label Noise Robustness Experiments
===================================
Tests EAFA vs FedAvg under varying label noise conditions.

Scenario:
  - 5 clients total
  - Clients 0,1,2: CLEAN data (expert annotators)
  - Client 3: noise_rate noise (novice annotator)
  - Client 4: 2×noise_rate noise (crowd-sourced labels)

This asymmetric noise pattern is realistic:
  In real FL, some hospitals have expert annotators while others
  use less trained staff or crowd-sourcing.

EAFA hypothesis:
  Noisy clients → higher EDL uncertainty → EAFA downweights them
  → better global model vs FedAvg which weights by size only.
"""

import sys, os, json, time, copy
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_FILE = "results_noise_robustness.json"


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)


def run_noise_experiment(dataset, aggregation, noise_rate, seed=42):
    """
    Run one federated experiment with label noise injected into clients 3,4.
    
    Returns:
        dict with wf1, per-client uncertainties, aggregation weights
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
        load_meld, load_iemocap, load_dailydialog,
        GenericDialogueDataset, collate_dialogues, evaluate,
    )
    from scripts.label_noise import inject_label_noise
    from models.evidential.evidential_dialogue_rnn import EvidentialDialogueRNN
    from models.evidential.losses import SupervisedEvidentialLoss
    from federated.aggregation.eafa import EAFAAggregator
    from data.federated_partition import FederatedPartitioner
    
    loaders = {"meld": load_meld, "iemocap": load_iemocap, "dailydialog": load_dailydialog}
    
    args = Namespace(
        hidden_dim=256, dropout=0.3, batch_size=16, lr=1e-3,
        annealing_epochs=30, patience=15, num_clients=5,
        alpha=0.5, num_rounds=50, local_epochs=3, beta=1.0,
        loss_type="edl", aggregation=aggregation,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_dir="checkpoints", seed=seed, finetuned=True,
    )
    
    if aggregation == "fedavg":
        args.beta = 0.0
    
    # Load data
    load_fn = loaders[dataset]
    train_dias, dev_dias, test_dias, emotions, weights, cache, num_spk = load_fn(finetuned=True)
    num_classes = len(emotions)
    
    device = args.device
    class_weights = torch.from_numpy(weights.astype(np.float32)).to(device)
    
    # Partition into clients
    partitioner = FederatedPartitioner(
        num_clients=5, strategy="dirichlet", alpha=args.alpha, seed=seed,
    )
    client_partitions = partitioner.partition(train_dias, label_ratio=1.0)
    
    dialogue_lookup = {d.dialogue_id: d for d in train_dias}
    
    # Build client data WITH noise injection
    noise_config = {
        3: noise_rate,        # Client 3: moderate noise
        4: noise_rate * 2,    # Client 4: heavy noise (capped at 0.8)
    }
    
    client_loaders = []
    noise_stats = {}
    
    for client_idx, partition in enumerate(client_partitions):
        dias = [dialogue_lookup[did] for did in partition.dialogue_ids if did in dialogue_lookup]
        
        client_noise = noise_config.get(client_idx, 0.0)
        client_noise = min(client_noise, 0.8)  # Cap at 80%
        
        if client_noise > 0:
            dias, stats = inject_label_noise(
                dias, client_noise, num_classes, seed=seed + client_idx
            )
            noise_stats[f"client_{client_idx}"] = stats
            logger.info(f"  Client {client_idx}: {len(dias)} dialogues, noise={client_noise:.0%}")
        else:
            noise_stats[f"client_{client_idx}"] = {"actual_rate": 0.0}
            logger.info(f"  Client {client_idx}: {len(dias)} dialogues, CLEAN")
        
        ds = GenericDialogueDataset(dias, cache.get("train", {}))
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                           collate_fn=collate_dialogues, num_workers=0)
        client_loaders.append(loader)
    
    # Test loader (ALWAYS clean)
    test_ds = GenericDialogueDataset(test_dias, cache.get("test", {}))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_dialogues, num_workers=0)
    
    # Model
    global_model = EvidentialDialogueRNN(
        input_dim=768, hidden_dim=args.hidden_dim,
        num_classes=num_classes, num_speakers=num_spk, dropout=args.dropout,
    ).to(device)
    loss_fn = SupervisedEvidentialLoss(
        num_classes=num_classes, annealing_epochs=args.annealing_epochs,
        class_weights=class_weights,
    )
    
    effective_beta = args.beta
    aggregator = EAFAAggregator(beta=effective_beta)
    agg_label = "EAFA" if aggregation == "eafa" else "FedAvg"
    
    logger.info(f"\n{'='*60}")
    logger.info(f"  {agg_label} | {dataset.upper()} | noise={noise_rate:.0%} | seed={seed}")
    logger.info(f"{'='*60}\n")
    
    # Training loop
    best_wf1, patience_cnt = 0.0, 0
    round_data = []
    
    for round_num in range(1, args.num_rounds + 1):
        start = time.time()
        client_states, client_sizes, client_us = [], [], []
        
        for loader in client_loaders:
            local_model = copy.deepcopy(global_model).to(device)
            local_model.train()
            loss_fn.set_epoch(round_num)
            opt = torch.optim.Adam(local_model.parameters(), lr=args.lr, weight_decay=1e-4)
            all_u_local = []
            
            for _ in range(args.local_epochs):
                for batch in loader:
                    feats = batch["features"].to(device)
                    speakers = batch["speaker_ids"].to(device)
                    labels = batch["labels"].to(device)
                    out = local_model(feats, speakers)
                    mask = labels != -1
                    loss, _ = loss_fn(out["alpha"][mask], labels[mask])
                    all_u_local.extend(out["uncertainty"][mask].detach().cpu().numpy())
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), 5.0)
                    opt.step()
            
            client_states.append(OrderedDict({k: v.cpu() for k, v in local_model.state_dict().items()}))
            client_sizes.append(len(loader.dataset))
            client_us.append(float(np.mean(all_u_local)) if all_u_local else 0.0)
        
        global_state, agg_stats = aggregator.aggregate(
            client_states, client_sizes, client_us, round_num,
        )
        global_model.load_state_dict(global_state)
        global_model.to(device)
        
        test_wf1, test_u, _, _ = evaluate(global_model, test_loader, device)
        elapsed = time.time() - start
        
        # Track per-round data
        round_data.append({
            "round": round_num,
            "wf1": round(test_wf1, 4),
            "client_uncertainties": [round(u, 4) for u in client_us],
            "weights": [round(w, 4) for w in agg_stats["weights"]],
        })
        
        w_str = ",".join(f"{w:.2f}" for w in agg_stats["weights"])
        u_str = ",".join(f"{u:.3f}" for u in client_us)
        logger.info(
            f"R{round_num:2d}/{args.num_rounds} | WF1={test_wf1:.4f} | "
            f"u=[{u_str}] | w=[{w_str}] | {elapsed:.1f}s"
        )
        
        if test_wf1 > best_wf1:
            best_wf1 = test_wf1
            patience_cnt = 0
            ckpt_path = Path(args.save_dir) / f"best_noise_{agg_label.lower()}_{dataset}_seed{seed}.pt"
            ckpt_path.parent.mkdir(exist_ok=True)
            torch.save({"model_state_dict": global_model.state_dict()}, ckpt_path)
            logger.info(f"  >> New best! WF1={test_wf1:.4f}")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                logger.info(f"  Early stopping at round {round_num}")
                break
        
        del client_states
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Final eval on best checkpoint
    ckpt_path = Path(args.save_dir) / f"best_noise_{agg_label.lower()}_{dataset}_seed{seed}.pt"
    if ckpt_path.exists():
        global_model.load_state_dict(torch.load(ckpt_path, weights_only=False)["model_state_dict"])
    final_wf1, final_u, report, _ = evaluate(global_model, test_loader, device, emotions, dataset)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"  RESULT: {agg_label} | {dataset.upper()} | noise={noise_rate:.0%}")
    logger.info(f"{'='*60}")
    logger.info(f"\n{report}")
    logger.info(f"  Best WF1 = {final_wf1:.4f}")
    logger.info(f"{'='*60}")
    
    return {
        "wf1": round(final_wf1, 4),
        "noise_rate": noise_rate,
        "noise_stats": noise_stats,
        "round_data": round_data[:5] + round_data[-3:],  # Save first 5 + last 3 rounds
    }


def main():
    results = load_results()
    
    total_start = time.time()
    
    # Experiment matrix
    datasets = ["meld", "iemocap"]
    noise_levels = [0.0, 0.1, 0.2, 0.4]
    aggregations = ["eafa", "fedavg"]
    seed = 42
    
    experiments = []
    for dataset in datasets:
        for noise in noise_levels:
            for agg in aggregations:
                experiments.append((dataset, agg, noise))
    
    total = len(experiments)
    
    for idx, (dataset, agg, noise) in enumerate(experiments):
        key = f"{dataset}_{agg}_noise{noise:.1f}_s{seed}"
        
        if key in results and results[key].get("wf1") is not None:
            print(f"[{idx+1}/{total}] SKIP {key}: WF1={results[key]['wf1']}")
            continue
        
        print(f"\n[{idx+1}/{total}] RUNNING {key}...")
        start = time.time()
        
        try:
            r = run_noise_experiment(dataset, agg, noise, seed=seed)
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
    print(f"\n{'='*70}")
    print(f"  NOISE ROBUSTNESS RESULTS — {total_time/60:.1f} minutes")
    print(f"{'='*70}")
    
    for dataset in datasets:
        print(f"\n  {dataset.upper()}:")
        print(f"  {'Noise':>6} | {'EAFA':>8} | {'FedAvg':>8} | {'Δ (EAFA-FedAvg)':>16} | {'Winner':>8}")
        print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*16}-+-{'-'*8}")
        
        for noise in noise_levels:
            eafa_key = f"{dataset}_eafa_noise{noise:.1f}_s{seed}"
            fedavg_key = f"{dataset}_fedavg_noise{noise:.1f}_s{seed}"
            
            eafa_wf1 = results.get(eafa_key, {}).get("wf1", None)
            fedavg_wf1 = results.get(fedavg_key, {}).get("wf1", None)
            
            if eafa_wf1 and fedavg_wf1:
                delta = eafa_wf1 - fedavg_wf1
                winner = "EAFA" if delta > 0 else "FedAvg" if delta < 0 else "Tie"
                print(f"  {noise:5.0%}  | {eafa_wf1:.4f}  | {fedavg_wf1:.4f}  | {delta:+.4f} ({delta*100:+.2f}%) | {winner}")
            else:
                print(f"  {noise:5.0%}  | {'N/A':>8} | {'N/A':>8} |")


if __name__ == "__main__":
    main()
