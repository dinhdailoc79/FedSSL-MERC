"""
EAFA-Guard Real-Data Byzantine Robustness Experiments
=======================================================
Runs EAFA-Guard and baselines under 3 attack types on MELD and IEMOCAP.

Attack types:
  1. label_flip:  Malicious clients flip all labels to (label+1) % C
  2. sign_flip:   Malicious clients negate and scale their parameter update
  3. adaptive:    Malicious clients sign-flip AND report u_k ≈ 0 to spoof EAFA

Contamination levels: 20% (1/5 clients) and 40% (2/5 clients)
Aggregators: FedAvg, EAFA, EAFA-Guard, Krum, Multi-Krum
Seeds: 5 per configuration
Datasets: MELD, IEMOCAP (priority: MELD first)

Usage:
    python scripts/run_eafa_guard_real.py
    python scripts/run_eafa_guard_real.py --dataset meld --attack label_flip --seeds 5
    python scripts/run_eafa_guard_real.py --dataset iemocap --quick  # 1 seed for testing
"""

import sys, os, json, time, copy, argparse
import numpy as np
import torch
from collections import OrderedDict
from pathlib import Path
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_FILE = "results_eafa_guard_real.json"


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)


def run_guard_experiment(dataset, aggregation, attack, contamination, seed=42, use_lf_guard=False):
    """Run one federated experiment with Byzantine attack + specified aggregator."""
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
    from federated.aggregation.eafa_guard import EAFAGuardAggregator
    from federated.aggregation.robust_aggregation import RobustAggregator
    from data.federated_partition import FederatedPartitioner

    loaders = {"meld": load_meld, "iemocap": load_iemocap, "dailydialog": load_dailydialog}

    num_clients = 5
    # Determine malicious client indices based on contamination
    if contamination <= 0.2:
        malicious_ids = {4}  # 1/5 = 20%
    else:
        malicious_ids = {3, 4}  # 2/5 = 40%

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

    # Build client data loaders (with attack injection)
    client_loaders = []
    for client_idx, partition in enumerate(client_partitions):
        dias = [dialogue_lookup[did] for did in partition.dialogue_ids if did in dialogue_lookup]

        # Label-flip attack for malicious clients
        if client_idx in malicious_ids and attack == "label_flip":
            logger.info(f"  Client {client_idx}: LABEL-FLIP attack")
            flipped = []
            for d in dias:
                dc = copy.deepcopy(d)
                for u in dc.utterances:
                    if u.emotion_idx != -1:
                        u.emotion_idx = (u.emotion_idx + 1) % num_classes
                flipped.append(dc)
            dias = flipped
        elif client_idx in malicious_ids:
            logger.info(f"  Client {client_idx}: {attack.upper()} attack (applied at weight level)")
        else:
            logger.info(f"  Client {client_idx}: {len(dias)} dialogues (Honest)")

        ds = GenericDialogueDataset(dias, cache.get("train", {}))
        loader = DataLoader(ds, batch_size=args_ns.batch_size, shuffle=True,
                            collate_fn=collate_dialogues, num_workers=0)
        client_loaders.append(loader)

    # Root set for EAFA-Guard: 25% of dev set (clean validation sample)
    root_dias = dev_dias[:max(1, len(dev_dias) // 4)]
    root_ds = GenericDialogueDataset(root_dias, cache.get("dev", cache.get("val", {})))
    root_loader = DataLoader(root_ds, batch_size=args_ns.batch_size, shuffle=True,
                             collate_fn=collate_dialogues, num_workers=0)

    # Test loader
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
    robust_aggregator = None
    # Check if we should use Label-Flip Guard (either explicit parameter or via naming convention)
    use_lf_guard = use_lf_guard or aggregation.endswith("_lf")
    if aggregation == "eafa_guard" or aggregation == "eafa_guard_lf":
        guard_aggregator = EAFAGuardAggregator(beta=args_ns.beta, use_label_flip_guard=use_lf_guard)
    elif aggregation in ("krum", "multi_krum"):
        f_est = len(malicious_ids)
        strategy = "krum" if aggregation == "krum" else "multi_krum"
        robust_aggregator = RobustAggregator(strategy=strategy, f=f_est)
    # eafa and fedavg use EAFAAggregator with appropriate beta
    eafa_aggregator = EAFAAggregator(
        beta=args_ns.beta if aggregation == "eafa" else 0.0
    )

    agg_label = aggregation.upper().replace("_", "-")
    logger.info(f"\n{'='*60}")
    logger.info(f"  {agg_label} | {dataset.upper()} | attack={attack} | "
                f"contamination={contamination} | seed={seed}")
    logger.info(f"{'='*60}\n")

    best_wf1, patience_cnt = 0.0, 0
    round_data = []

    for round_num in range(1, args_ns.num_rounds + 1):
        start = time.time()
        client_states, client_sizes, client_us = [], [], []
        global_state_cpu = OrderedDict(
            {k: v.cpu() for k, v in global_model.state_dict().items()}
        )

        # Local training
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

            client_state_cpu = OrderedDict(
                {k: v.cpu() for k, v in local_model.state_dict().items()}
            )
            mean_u = float(np.mean(all_u)) if all_u else 0.0

            # Sign-flip / adaptive attack at weight level
            if client_idx in malicious_ids and attack in ("sign_flip", "adaptive"):
                scale = 2.0
                for name in client_state_cpu:
                    diff = client_state_cpu[name].float() - global_state_cpu[name].float()
                    client_state_cpu[name] = global_state_cpu[name] - scale * diff

            # Adaptive attack: also spoof uncertainty to near-zero
            if client_idx in malicious_ids and attack == "adaptive":
                mean_u = 0.01  # Attacker lies about quality

            client_states.append(client_state_cpu)
            client_sizes.append(len(loader.dataset))
            client_us.append(mean_u)

            del local_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Aggregation
        if aggregation in ("eafa_guard", "eafa_guard_lf"):
            server_delta = guard_aggregator.compute_server_delta(
                global_model, root_loader, loss_fn, device
            )
            global_state, agg_stats = guard_aggregator.aggregate(
                client_states, client_sizes, client_us,
                global_state_cpu, server_delta, round_num,
            )
            agg_weights = agg_stats["weights"]
        elif aggregation in ("krum", "multi_krum"):
            global_state = robust_aggregator.aggregate(client_states, client_sizes)
            agg_weights = [1.0 / len(client_states)] * len(client_states)
        else:
            # FedAvg or EAFA
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

    # Final evaluation with macro-F1
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

    logger.info(f"\n{'='*60}")
    logger.info(f"  RESULT: {agg_label} | {dataset.upper()} | {attack} | "
                f"cont={contamination} | seed={seed}")
    logger.info(f"  Macro-F1={macro_f1:.4f}  Weighted-F1={weighted_f1:.4f}")
    logger.info(f"{'='*60}\n")

    return {
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "attack": attack,
        "contamination": contamination,
        "aggregation": aggregation,
        "dataset": dataset,
        "seed": seed,
        "num_rounds_completed": len(round_data),
        "round_data_sample": round_data[:3] + round_data[-3:],
    }


def main():
    parser = argparse.ArgumentParser(description="EAFA-Guard Real-Data Experiments")
    parser.add_argument("--dataset", type=str, default=None, help="meld or iemocap")
    parser.add_argument("--attack", type=str, default=None, help="label_flip, sign_flip, adaptive")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--quick", action="store_true", help="Quick mode: 1 seed only")
    parser.add_argument("--use-lf-guard", action="store_true",
                        help="Enable Label-Flip Detector in EAFA-Guard")
    args = parser.parse_args()

    results = load_results()
    total_start = time.time()

    datasets = [args.dataset] if args.dataset else ["meld", "iemocap"]
    attacks = [args.attack] if args.attack else ["label_flip", "sign_flip", "adaptive"]
    contaminations = [0.2, 0.4]
    use_lf_guard = args.use_lf_guard
    aggregations = ["fedavg", "eafa", "eafa_guard", "krum", "multi_krum"]
    if use_lf_guard:
        aggregations.append("eafa_guard_lf")
    num_seeds = 1 if args.quick else args.seeds

    # Build experiment list
    experiments = []
    for ds in datasets:
        for atk in attacks:
            for cont in contaminations:
                for agg in aggregations:
                    for s in range(num_seeds):
                        seed = 42 + s * 111
                        experiments.append((ds, agg, atk, cont, seed))

    total = len(experiments)
    print(f"Total experiments: {total}")

    for idx, (ds, agg, atk, cont, seed) in enumerate(experiments):
        key = f"{ds}_{agg}_{atk}_c{int(cont*100)}_s{seed}"

        if key in results and results[key].get("macro_f1") is not None:
            print(f"[{idx+1}/{total}] SKIP {key}: MF1={results[key]['macro_f1']}")
            continue

        print(f"\n[{idx+1}/{total}] RUNNING {key}...")
        exp_start = time.time()

        try:
            r = run_guard_experiment(ds, agg, atk, cont, seed=seed, use_lf_guard=use_lf_guard)
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

    # Summary
    elapsed_total = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  EAFA-Guard REAL-DATA RESULTS — SUMMARY")
    print(f"  Total time: {elapsed_total/3600:.1f} hours")
    print(f"{'='*70}")

    for ds in datasets:
        for atk in attacks:
            for cont in contaminations:
                print(f"\n  {ds.upper()} | {atk} | contamination={int(cont*100)}%")
                header = f"  {'Aggregator':<15}"
                for s in range(num_seeds):
                    header += f" | s{42+s*111}"
                header += " | Mean±Std"
                print(header)
                print(f"  {'-'*len(header)}")

                for agg in aggregations:
                    scores = []
                    row = f"  {agg:<15}"
                    for s in range(num_seeds):
                        seed = 42 + s * 111
                        key = f"{ds}_{agg}_{atk}_c{int(cont*100)}_s{seed}"
                        mf1 = results.get(key, {}).get("macro_f1")
                        if mf1 is not None:
                            scores.append(mf1)
                            row += f" | {mf1:.3f}"
                        else:
                            row += f" |   -  "
                    if scores:
                        row += f" | {np.mean(scores):.3f}±{np.std(scores):.3f}"
                    print(row)

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
