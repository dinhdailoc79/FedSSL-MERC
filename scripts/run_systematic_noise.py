"""
Systematic Label Noise Federated Experiments
=============================================
Runs FedAvg, EAFA, and EAFA-Guard under systematic label noise on MELD and IEMOCAP.

Scenario:
  - 5 clients total
  - Clients 0,1,2: Clean data
  - Client 3: systematic noise_rate
  - Client 4: systematic 2*noise_rate (capped at 0.8)

Seeds: 12 seeds (for statistical significance)
Noise rates: 0.0, 0.2, 0.4
Aggregators: FedAvg, EAFA, EAFA-Guard
Datasets: MELD, IEMOCAP (priority: MELD first)

Performs Wilcoxon signed-rank tests and computes Cohen's d effect size
between (EAFA vs FedAvg) and (EAFA-Guard vs FedAvg). Applies Holm-Bonferroni correction.
"""

import sys, os, json, time, copy, argparse
import numpy as np
import torch
from collections import OrderedDict
from pathlib import Path
from torch.utils.data import DataLoader
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_FILE = "results_systematic_noise.json"


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)


def cohen_d(x, y):
    """Compute Cohen's d effect size between two independent/paired samples."""
    nx, ny = len(x), len(y)
    mean_x, mean_y = np.mean(x), np.mean(y)
    var_x, var_y = np.var(x, ddof=1), np.var(y, ddof=1)
    
    # Pooled standard deviation
    pooled_sd = np.sqrt(((nx - 1) * var_x + (ny - 1) * var_y) / (nx + ny - 2))
    if pooled_sd < 1e-8:
        return 0.0
    return (mean_x - mean_y) / pooled_sd


def holm_bonferroni_correction(p_values):
    """
    Apply Holm-Bonferroni correction to a list of p-values.
    Returns the adjusted p-values.
    """
    m = len(p_values)
    if m == 0:
        return []
    
    # Sort with indices to restore original order later
    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    
    max_p = 0.0
    for rank, (orig_idx, p) in enumerate(indexed_p):
        multiplier = m - rank
        adj_p = min(1.0, p * multiplier)
        # Enforce monotonicity: adjusted p_i >= adjusted p_{i-1}
        max_p = max(max_p, adj_p)
        adjusted[orig_idx] = max_p
        
    return adjusted


def run_systematic_experiment(dataset, aggregation, noise_rate, seed=42):
    """Run one federated experiment under systematic label noise."""
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    torch.manual_seed(seed)
    np.random.seed(seed)

    from scripts.train_multi_dataset import (
        load_meld, load_iemocap, load_dailydialog,
        GenericDialogueDataset, collate_dialogues, evaluate,
    )
    from scripts.systematic_noise import inject_systematic_noise
    from models.evidential.evidential_dialogue_rnn import EvidentialDialogueRNN
    from models.evidential.losses import SupervisedEvidentialLoss
    from federated.aggregation.eafa import EAFAAggregator
    from federated.aggregation.eafa_guard import EAFAGuardAggregator
    from data.federated_partition import FederatedPartitioner

    loaders = {"meld": load_meld, "iemocap": load_iemocap}

    num_clients = 5
    args_ns = argparse.Namespace(
        hidden_dim=256, dropout=0.3, batch_size=16, lr=1e-3,
        annealing_epochs=30, patience=15, num_clients=num_clients,
        alpha=0.5, num_rounds=30, local_epochs=3, beta=4.0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_dir="checkpoints", seed=seed, finetuned=True,
    )
    device = args_ns.device

    # Load data
    load_fn = loaders[dataset]
    train_dias, dev_dias, test_dias, emotions, weights, cache, num_spk = load_fn(finetuned=True)
    num_classes = len(emotions)
    class_weights = torch.from_numpy(weights.astype(np.float32)).to(device)

    # Partition into clients
    partitioner = FederatedPartitioner(
        num_clients=num_clients, strategy="dirichlet", alpha=args_ns.alpha, seed=seed,
    )
    client_partitions = partitioner.partition(train_dias, label_ratio=1.0)
    dialogue_lookup = {d.dialogue_id: d for d in train_dias}

    # Inject systematic label noise to specific clients
    noise_config = {
        3: noise_rate,        # Client 3: moderate noise
        4: noise_rate * 2,    # Client 4: heavy noise
    }

    client_loaders = []
    noise_stats = {}

    for client_idx, partition in enumerate(client_partitions):
        dias = [dialogue_lookup[did] for did in partition.dialogue_ids if did in dialogue_lookup]
        client_noise = noise_config.get(client_idx, 0.0)
        client_noise = min(client_noise, 0.8)  # Cap at 80%

        if client_noise > 0:
            dias, stats = inject_systematic_noise(
                dias, dataset, client_noise, seed=seed + client_idx
            )
            noise_stats[f"client_{client_idx}"] = stats
            logger.info(f"  Client {client_idx}: {len(dias)} dialogues, systematic noise={client_noise:.0%}")
        else:
            noise_stats[f"client_{client_idx}"] = {"actual_rate": 0.0}
            logger.info(f"  Client {client_idx}: {len(dias)} dialogues, CLEAN")

        ds = GenericDialogueDataset(dias, cache.get("train", {}))
        loader = DataLoader(ds, batch_size=args_ns.batch_size, shuffle=True,
                            collate_fn=collate_dialogues, num_workers=0)
        client_loaders.append(loader)

    # Root set for EAFA-Guard: first 5% of dev set
    root_dias = dev_dias[:max(1, len(dev_dias) // 20)]
    root_ds = GenericDialogueDataset(root_dias, cache.get("dev", cache.get("val", {})))
    root_loader = DataLoader(root_ds, batch_size=args_ns.batch_size, shuffle=True,
                             collate_fn=collate_dialogues, num_workers=0)

    # Test loader (clean)
    test_ds = GenericDialogueDataset(test_dias, cache.get("test", {}))
    test_loader = DataLoader(test_ds, batch_size=args_ns.batch_size, shuffle=False,
                             collate_fn=collate_dialogues, num_workers=0)

    # Initialize model
    global_model = EvidentialDialogueRNN(
        input_dim=768, hidden_dim=args_ns.hidden_dim,
        num_classes=num_classes, num_speakers=num_spk, dropout=args_ns.dropout,
    ).to(device)

    loss_fn = SupervisedEvidentialLoss(
        num_classes=num_classes, annealing_epochs=args_ns.annealing_epochs,
        class_weights=class_weights,
    )

    # Setup aggregator
    guard_aggregator = None
    if aggregation == "eafa_guard":
        guard_aggregator = EAFAGuardAggregator(beta=args_ns.beta)
    eafa_aggregator = EAFAAggregator(
        beta=args_ns.beta if aggregation in ("eafa", "eafa_guard") else 0.0
    )

    agg_label = aggregation.upper().replace("_", "-")
    logger.info(f"\n{'='*60}")
    logger.info(f"  {agg_label} | {dataset.upper()} | systematic_noise={noise_rate:.0%} | seed={seed}")
    logger.info(f"{'='*60}\n")

    best_wf1, patience_cnt = 0.0, 0
    round_data = []

    for round_num in range(1, args_ns.num_rounds + 1):
        start = time.time()
        client_states, client_sizes, client_us = [], [], []
        global_state_cpu = OrderedDict(
            {k: v.cpu() for k, v in global_model.state_dict().items()}
        )

        for client_idx, loader in enumerate(client_loaders):
            local_model = copy.deepcopy(global_model).to(device)
            local_model.train()
            loss_fn.set_epoch(round_num)
            opt = torch.optim.Adam(local_model.parameters(), lr=args_ns.lr, weight_decay=1e-4)
            all_u = []

            for _ in range(args_ns.local_epochs):
                for batch in loader:
                    feats = batch["features"].to(device)
                    speakers = batch["speaker_ids"].to(device)
                    labels = batch["labels"].to(device)
                    out = local_model(feats, speakers)
                    mask = labels != -1
                    if mask.sum() == 0:
                        continue
                    loss, _ = loss_fn(out["alpha"][mask], labels[mask])
                    all_u.extend(out["uncertainty"][mask].detach().cpu().numpy())
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), 5.0)
                    opt.step()

            client_states.append(OrderedDict({k: v.cpu() for k, v in local_model.state_dict().items()}))
            client_sizes.append(len(loader.dataset))
            client_us.append(float(np.mean(all_u)) if all_u else 0.0)

            del local_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Aggregate
        if aggregation == "eafa_guard":
            server_delta = guard_aggregator.compute_server_delta(
                global_model, root_loader, loss_fn, device
            )
            global_state, agg_stats = guard_aggregator.aggregate(
                client_states, client_sizes, client_us,
                global_state_cpu, server_delta, round_num,
            )
            agg_weights = agg_stats["weights"]
        else:
            global_state, agg_stats = eafa_aggregator.aggregate(
                client_states, client_sizes, client_us, round_num,
            )
            agg_weights = agg_stats["weights"]

        global_model.load_state_dict(global_state)
        global_model.to(device)

        # Evaluate
        test_wf1, test_u, _, _ = evaluate(global_model, test_loader, device)
        elapsed = time.time() - start

        round_data.append({
            "round": round_num,
            "wf1": round(test_wf1, 4),
            "client_uncertainties": [round(u, 4) for u in client_us],
            "weights": [round(w, 4) for w in agg_weights],
        })

        logger.info(
            f"R{round_num:2d}/{args_ns.num_rounds} | WF1={test_wf1:.4f} | "
            f"u=[{','.join(f'{u:.3f}' for u in client_us)}] | "
            f"w=[{','.join(f'{w:.2f}' for w in agg_weights)}] | {elapsed:.1f}s"
        )

        if test_wf1 > best_wf1:
            best_wf1 = test_wf1
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= args_ns.patience:
                logger.info(f"  Early stopping at round {round_num}")
                break

        del client_states
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Macro-F1 evaluation
    from sklearn.metrics import f1_score
    global_model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            feats = batch["features"].to(device)
            speakers = batch["speaker_ids"].to(device)
            labels = batch["labels"].to(device)
            out = global_model(feats, speakers)
            mask = labels != -1
            preds = out["alpha"][mask].argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels[mask].cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return {
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "noise_rate": noise_rate,
        "aggregation": aggregation,
        "dataset": dataset,
        "seed": seed,
        "noise_stats": noise_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Systematic Label Noise Experiments")
    parser.add_argument("--dataset", type=str, default=None, help="meld or iemocap")
    parser.add_argument("--seeds", type=int, default=12, help="Number of seeds to run")
    parser.add_argument("--quick", action="store_true", help="Quick mode: 1 seed only")
    args = parser.parse_args()

    results = load_results()
    total_start = time.time()

    datasets = [args.dataset] if args.dataset else ["meld", "iemocap"]
    noise_rates = [0.0, 0.2, 0.4]
    aggregations = ["fedavg", "eafa", "eafa_guard"]
    num_seeds = 1 if args.quick else args.seeds

    experiments = []
    for ds in datasets:
        for nr in noise_rates:
            for agg in aggregations:
                for s in range(num_seeds):
                    seed = 42 + s * 111
                    experiments.append((ds, agg, nr, seed))

    total = len(experiments)
    print(f"Total experiments to run: {total}")

    for idx, (ds, agg, nr, seed) in enumerate(experiments):
        key = f"{ds}_{agg}_sysn{int(nr*100)}_s{seed}"

        if key in results and results[key].get("macro_f1") is not None:
            print(f"[{idx+1}/{total}] SKIP {key}: MF1={results[key]['macro_f1']}")
            continue

        print(f"\n[{idx+1}/{total}] RUNNING {key}...")
        exp_start = time.time()

        try:
            r = run_systematic_experiment(ds, agg, nr, seed=seed)
            r["time_seconds"] = round(time.time() - exp_start, 1)
            results[key] = r
            save_results(results)
            print(f"  >> MF1={r['macro_f1']}, WF1={r['weighted_f1']}, "
                  f"time={r['time_seconds']:.0f}s")
        except Exception as e:
            import traceback
            print(f"  >> ERROR: {e}")
            traceback.print_exc()
            results[key] = {"macro_f1": None, "error": str(e)}
            save_results(results)

    # Perform Statistical Significance tests
    print(f"\n{'='*70}")
    print(f"  STATISTICAL SIGNIFICANCE TESTS (Holm-corrected Wilcoxon & Cohen's d)")
    print(f"{'='*70}")

    for ds in datasets:
        print(f"\nDataset: {ds.upper()}")
        for nr in noise_rates:
            print(f"  Noise Rate: {int(nr*100)}%")
            
            # Extract lists of macro-F1 scores over all seeds
            fedavg_scores = []
            eafa_scores = []
            guard_scores = []
            
            for s in range(num_seeds):
                seed = 42 + s * 111
                fedavg_scores.append(results.get(f"{ds}_fedavg_sysn{int(nr*100)}_s{seed}", {}).get("macro_f1"))
                eafa_scores.append(results.get(f"{ds}_eafa_sysn{int(nr*100)}_s{seed}", {}).get("macro_f1"))
                guard_scores.append(results.get(f"{ds}_eafa_guard_sysn{int(nr*100)}_s{seed}", {}).get("macro_f1"))

            # Filter out Nones
            fedavg_scores = [x for x in fedavg_scores if x is not None]
            eafa_scores = [x for x in eafa_scores if x is not None]
            guard_scores = [x for x in guard_scores if x is not None]

            if len(fedavg_scores) < 3:
                print("    Not enough completed runs for statistical tests.")
                continue

            # Print averages
            print(f"    FedAvg MF1:      {np.mean(fedavg_scores):.4f} ± {np.std(fedavg_scores):.4f}")
            print(f"    EAFA MF1:        {np.mean(eafa_scores):.4f} ± {np.std(eafa_scores):.4f}")
            print(f"    EAFA-Guard MF1:  {np.mean(guard_scores):.4f} ± {np.std(guard_scores):.4f}")

            # Pairwise tests vs FedAvg
            p_vals = []
            comparisons = []
            
            # EAFA vs FedAvg
            if len(eafa_scores) == len(fedavg_scores):
                try:
                    stat, p = wilcoxon(eafa_scores, fedavg_scores)
                    p_vals.append(p)
                    comparisons.append(("EAFA vs FedAvg", eafa_scores, fedavg_scores))
                except Exception as e:
                    p_vals.append(1.0)
                    comparisons.append(("EAFA vs FedAvg (failed)", eafa_scores, fedavg_scores))
            
            # EAFA-Guard vs FedAvg
            if len(guard_scores) == len(fedavg_scores):
                try:
                    stat, p = wilcoxon(guard_scores, fedavg_scores)
                    p_vals.append(p)
                    comparisons.append(("EAFA-Guard vs FedAvg", guard_scores, fedavg_scores))
                except Exception as e:
                    p_vals.append(1.0)
                    comparisons.append(("EAFA-Guard vs FedAvg (failed)", guard_scores, fedavg_scores))

            # Apply Holm correction to the 2 p-values
            adj_p_vals = holm_bonferroni_correction(p_values=p_vals)

            for i, (name, x_data, y_data) in enumerate(comparisons):
                raw_p = p_vals[i]
                adj_p = adj_p_vals[i]
                d = cohen_d(x_data, y_data)
                print(f"    {name:<25} | Cohen's d: {d:+.3f} | p-value: {raw_p:.4f} (Holm-corrected: {adj_p:.4f})")

    elapsed_total = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"Total processing time: {elapsed_total/3600:.2f} hours")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
