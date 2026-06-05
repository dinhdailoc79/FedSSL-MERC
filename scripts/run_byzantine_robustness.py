"""
Byzantine Robustness Experiments for FedSSL-MERC
==================================================
Simulates Byzantine attacks (Label Flipping and Sign Flipping Weight Attacks)
on a subset of clients and compares EAFA vs FedAvg robustness.

Settings:
  - 5 clients total.
  - Client 4 is the Byzantine attacker.
  - Attack types:
    1. 'label_flip': Client 4 flips all its labels to a wrong target class.
    2. 'sign_flip': Client 4 multiplies its update by -2.0 to degrade global model.

Usage:
    python scripts/run_byzantine_robustness.py
"""

import sys
import os
import json
import time
import copy
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import DataLoader
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_FILE = "results_byzantine_robustness.json"


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)


def run_byzantine_experiment(dataset, aggregation, attack, seed=42):
    """Run one federated experiment with Byzantine attack on Client 4."""
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
    from models.evidential.losses import SupervisedEvidentialLoss
    from federated.aggregation.eafa import EAFAAggregator
    from data.federated_partition import FederatedPartitioner

    loaders = {"meld": load_meld, "iemocap": load_iemocap, "dailydialog": load_dailydialog}

    args = Namespace(
        hidden_dim=256, dropout=0.3, batch_size=16, lr=1e-3,
        annealing_epochs=30, patience=15, num_clients=5,
        alpha=0.5, num_rounds=30, local_epochs=3, beta=1.0,
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

    client_loaders = []
    for client_idx, partition in enumerate(client_partitions):
        dias = [dialogue_lookup[did] for did in partition.dialogue_ids if did in dialogue_lookup]
        
        # Inject Label Flipping attack for Client 4
        if client_idx == 4 and attack == "label_flip":
            logger.info(f"  Client {client_idx}: Injecting LABEL FLIPPING attack.")
            flipped_dias = []
            for d in dias:
                d_copy = copy.deepcopy(d)
                # Flip all valid labels to (label + 1) % num_classes
                for u in d_copy.utterances:
                    if u.emotion_idx != -1:
                        u.emotion_idx = (u.emotion_idx + 1) % num_classes
                flipped_dias.append(d_copy)
            dias = flipped_dias
        else:
            logger.info(f"  Client {client_idx}: {len(dias)} dialogues (Honest)")

        ds = GenericDialogueDataset(dias, cache.get("train", {}))
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                            collate_fn=collate_dialogues, num_workers=0)
        client_loaders.append(loader)

    # Clean test loader
    test_ds = GenericDialogueDataset(test_dias, cache.get("test", {}))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_dialogues, num_workers=0)

    # Initialize model
    global_model = EvidentialDialogueRNN(
        input_dim=768, hidden_dim=args.hidden_dim,
        num_classes=num_classes, num_speakers=num_spk, dropout=args.dropout,
    ).to(device)
    
    loss_fn = SupervisedEvidentialLoss(
        num_classes=num_classes, annealing_epochs=args.annealing_epochs,
        class_weights=class_weights,
    )
    
    aggregator = EAFAAggregator(beta=args.beta)
    agg_label = "EAFA" if aggregation == "eafa" else "FedAvg"

    logger.info(f"\n{'='*60}")
    logger.info(f"  {agg_label} | {dataset.upper()} | attack={attack} | seed={seed}")
    logger.info(f"{'='*60}\n")

    best_wf1, patience_cnt = 0.0, 0
    round_data = []

    for round_num in range(1, args.num_rounds + 1):
        start = time.time()
        client_states, client_sizes, client_us = [], [], []
        
        global_state_cpu = OrderedDict({k: v.cpu() for k, v in global_model.state_dict().items()})

        for client_idx, loader in enumerate(client_loaders):
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

            client_state_cpu = OrderedDict({k: v.cpu() for k, v in local_model.state_dict().items()})
            
            # Inject Sign Flipping Weight Attack for Client 4
            if client_idx == 4 and attack == "sign_flip":
                logger.info(f"    Applying SIGN FLIPPING Weight Attack to Client 4 update.")
                scale = 2.0
                for name in client_state_cpu.keys():
                    diff = client_state_cpu[name].float() - global_state_cpu[name].float()
                    client_state_cpu[name] = global_state_cpu[name] - scale * diff

            client_states.append(client_state_cpu)
            client_sizes.append(len(loader.dataset))
            client_us.append(float(np.mean(all_u_local)) if all_u_local else 0.0)

        # Aggregate
        global_state, agg_stats = aggregator.aggregate(
            client_states, client_sizes, client_us, round_num,
        )
        global_model.load_state_dict(global_state)
        global_model.to(device)

        test_wf1, test_u, _, _ = evaluate(global_model, test_loader, device)
        elapsed = time.time() - start

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
            ckpt_path = Path(args.save_dir) / f"best_byz_{agg_label.lower()}_{attack}_{dataset}.pt"
            ckpt_path.parent.mkdir(exist_ok=True)
            torch.save({"model_state_dict": global_model.state_dict()}, ckpt_path)
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                logger.info(f"  Early stopping at round {round_num}")
                break

        del client_states
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Load best checkpoint for final evaluation
    ckpt_path = Path(args.save_dir) / f"best_byz_{agg_label.lower()}_{attack}_{dataset}.pt"
    if ckpt_path.exists():
        global_model.load_state_dict(torch.load(ckpt_path, weights_only=False)["model_state_dict"])
    final_wf1, final_u, report, _ = evaluate(global_model, test_loader, device, emotions, dataset)

    logger.info(f"\n{'='*60}")
    logger.info(f"  RESULT: {agg_label} | {dataset.upper()} | attack={attack} | WF1={final_wf1:.4f}")
    logger.info(f"{'='*60}\n")

    return {
        "wf1": round(final_wf1, 4),
        "attack": attack,
        "round_data": round_data[:5] + round_data[-3:],
    }


def main():
    results = load_results()
    total_start = time.time()

    datasets = ["meld"]  # Standardizing on MELD as typical representative
    attacks = ["label_flip", "sign_flip"]
    aggregations = ["eafa", "fedavg"]
    seed = 42

    experiments = []
    for dataset in datasets:
        for attack in attacks:
            for agg in aggregations:
                experiments.append((dataset, agg, attack))

    total = len(experiments)

    for idx, (dataset, agg, attack) in enumerate(experiments):
        key = f"{dataset}_{agg}_{attack}_s{seed}"

        if key in results and results[key].get("wf1") is not None:
            print(f"[{idx+1}/{total}] SKIP {key}: WF1={results[key]['wf1']}")
            continue

        print(f"\n[{idx+1}/{total}] RUNNING {key}...")
        start = time.time()

        try:
            r = run_byzantine_experiment(dataset, agg, attack, seed=seed)
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

    # Print summary
    print(f"\n{'='*70}")
    print(f"  BYZANTINE ROBUSTNESS RESULTS — SUMMARY")
    print(f"{'='*70}")
    for attack in attacks:
        print(f"\n  Attack: {attack.upper()}")
        print(f"  {'Dataset':<10} | {'EAFA WF1':<10} | {'FedAvg WF1':<10} | {'Delta':<10}")
        print(f"  {'-'*50}")
        for dataset in datasets:
            eafa_key = f"{dataset}_eafa_{attack}_s{seed}"
            fedavg_key = f"{dataset}_fedavg_{attack}_s{seed}"
            eafa_wf1 = results.get(eafa_key, {}).get("wf1")
            fedavg_wf1 = results.get(fedavg_key, {}).get("wf1")
            if eafa_wf1 is not None and fedavg_wf1 is not None:
                print(f"  {dataset:<10} | {eafa_wf1:.4f}   | {fedavg_wf1:.4f}     | {eafa_wf1 - fedavg_wf1:+.4f}")
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
