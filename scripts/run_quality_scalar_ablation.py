"""
Quality Scalar Ablation Study for EAFA weighting
=================================================
Compares the performance of EAFA under three different quality scalar definitions:
  1. Combined: 0.5 * Brier + 0.5 * Vacuity (Default)
  2. Vacuity-only: 0.0 * Brier + 1.0 * Vacuity
  3. Brier-only: 1.0 * Brier + 0.0 * Vacuity

Runs under noise robustness testing to show how each scalar handles client contamination.
Scenario:
  - 5 clients total
  - Client 3 has 40% systematic label noise
  - Client 4 has 80% systematic label noise
  - Clients 0,1,2: Clean (0% noise)
  - Seeds: 3 seeds per variant

Datasets: MELD, IEMOCAP (priority: MELD first)
"""

import sys, os, json, time, copy, argparse
import numpy as np
import torch
from collections import OrderedDict
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_FILE = "results_quality_scalar_ablation.json"


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)


def compute_client_quality_scalar(model, loader, device, num_classes, variant):
    """
    Computes the client's average quality scalar q_k over its dataset.
    
    Args:
        model: EvidentialDialogueRNN model
        loader: DataLoader for client's training set
        device: Torch device
        num_classes: Number of emotion classes
        variant: 'combined', 'vacuity', or 'brier'
    """
    model.eval()
    total_samples = 0
    total_q = 0.0
    
    with torch.no_grad():
        for batch in loader:
            feats = batch["features"].to(device)
            speakers = batch["speaker_ids"].to(device)
            labels = batch["labels"].to(device)
            
            out = model(feats, speakers)
            mask = labels != -1
            if mask.sum() == 0:
                continue
            
            alpha = out["alpha"][mask]
            y = labels[mask]
            
            # Dirichlet strength and expected probability
            S = alpha.sum(dim=-1, keepdim=True)
            p = alpha / S
            
            # 1. Vacuity: C / S
            vacuity = num_classes / S.squeeze(-1)  # [N]
            
            # 2. Brier: Sum_{c=1}^C (p_c - y_c)^2
            y_onehot = torch.eye(num_classes, device=device)[y]
            brier = ((p - y_onehot) ** 2).sum(dim=-1)  # [N]
            
            if variant == "combined":
                q = 0.5 * brier + 0.5 * vacuity
            elif variant == "vacuity":
                q = vacuity
            elif variant == "brier":
                q = brier
            else:
                raise ValueError(f"Unknown variant: {variant}")
                
            total_q += q.sum().item()
            total_samples += mask.sum().item()
            
    return float(total_q / total_samples) if total_samples > 0 else 1.0


def run_ablation_experiment(dataset, variant, seed=42):
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    torch.manual_seed(seed)
    np.random.seed(seed)

    from scripts.train_multi_dataset import (
        load_meld, load_iemocap, GenericDialogueDataset, collate_dialogues, evaluate
    )
    from scripts.systematic_noise import inject_systematic_noise
    from models.evidential.evidential_dialogue_rnn import EvidentialDialogueRNN
    from models.evidential.losses import SupervisedEvidentialLoss
    from federated.aggregation.eafa import EAFAAggregator
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

    # Inject 40% noise to client 3 and 80% to client 4
    noise_config = {3: 0.4, 4: 0.8}

    client_loaders = []
    for client_idx, partition in enumerate(client_partitions):
        dias = [dialogue_lookup[did] for did in partition.dialogue_ids if did in dialogue_lookup]
        client_noise = noise_config.get(client_idx, 0.0)

        if client_noise > 0:
            dias, stats = inject_systematic_noise(
                dias, dataset, client_noise, seed=seed + client_idx
            )
            logger.info(f"  Client {client_idx}: {len(dias)} dialogues, systematic noise={client_noise:.0%}")
        else:
            logger.info(f"  Client {client_idx}: {len(dias)} dialogues, CLEAN")

        ds = GenericDialogueDataset(dias, cache.get("train", {}))
        loader = DataLoader(ds, batch_size=args_ns.batch_size, shuffle=True,
                            collate_fn=collate_dialogues, num_workers=0)
        client_loaders.append(loader)

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

    aggregator = EAFAAggregator(beta=args_ns.beta)

    logger.info(f"\n{'='*60}")
    logger.info(f"  EAFA-ABLATION | {dataset.upper()} | variant={variant.upper()} | seed={seed}")
    logger.info(f"{'='*60}\n")

    best_wf1, patience_cnt = 0.0, 0
    round_data = []

    for round_num in range(1, args_ns.num_rounds + 1):
        start = time.time()
        client_states, client_sizes, client_qs = [], [], []

        # Local training
        for client_idx, loader in enumerate(client_loaders):
            local_model = copy.deepcopy(global_model).to(device)
            local_model.train()
            loss_fn.set_epoch(round_num)
            opt = torch.optim.Adam(local_model.parameters(), lr=args_ns.lr, weight_decay=1e-4)

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
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), 5.0)
                    opt.step()

            # Compute the custom quality scalar q_k for EAFA weighting
            q_k = compute_client_quality_scalar(
                local_model, loader, device, num_classes, variant
            )

            client_state_cpu = OrderedDict(
                {k: v.cpu() for k, v in local_model.state_dict().items()}
            )
            client_states.append(client_state_cpu)
            client_sizes.append(len(loader.dataset))
            client_qs.append(q_k)

            del local_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Aggregate using EAFA
        global_state, agg_stats = aggregator.aggregate(
            client_states, client_sizes, client_qs, round_num
        )
        global_model.load_state_dict(global_state)
        global_model.to(device)

        # Evaluate
        test_wf1, _, _, _ = evaluate(global_model, test_loader, device)
        elapsed = time.time() - start

        round_data.append({
            "round": round_num,
            "wf1": round(test_wf1, 4),
            "client_qs": [round(q, 4) for q in client_qs],
            "weights": [round(w, 4) for w in agg_stats["weights"]],
        })

        logger.info(
            f"R{round_num:2d}/{args_ns.num_rounds} | WF1={test_wf1:.4f} | "
            f"q=[{','.join(f'{q:.3f}' for q in client_qs)}] | "
            f"w=[{','.join(f'{w:.2f}' for w in agg_stats['weights'])}] | {elapsed:.1f}s"
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
        "variant": variant,
        "dataset": dataset,
        "seed": seed,
    }


def main():
    parser = argparse.ArgumentParser(description="EAFA Quality Scalar Ablation Study")
    parser.add_argument("--dataset", type=str, default=None, help="meld or iemocap")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds")
    parser.add_argument("--quick", action="store_true", help="Quick mode: 1 seed")
    args = parser.parse_args()

    results = load_results()
    total_start = time.time()

    datasets = [args.dataset] if args.dataset else ["meld", "iemocap"]
    variants = ["combined", "vacuity", "brier"]
    num_seeds = 1 if args.quick else args.seeds

    experiments = []
    for ds in datasets:
        for var in variants:
            for s in range(num_seeds):
                seed = 42 + s * 111
                experiments.append((ds, var, seed))

    total = len(experiments)
    print(f"Total ablation experiments: {total}")

    for idx, (ds, var, seed) in enumerate(experiments):
        key = f"{ds}_{var}_s{seed}"

        if key in results and results[key].get("macro_f1") is not None:
            print(f"[{idx+1}/{total}] SKIP {key}: MF1={results[key]['macro_f1']}")
            continue

        print(f"\n[{idx+1}/{total}] RUNNING {key}...")
        exp_start = time.time()

        try:
            r = run_ablation_experiment(ds, var, seed=seed)
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
    print(f"  EAFA QUALITY SCALAR ABLATION SUMMARY")
    print(f"  Total time: {elapsed_total/3600:.2f} hours")
    print(f"{'='*70}")

    for ds in datasets:
        print(f"\nDataset: {ds.upper()}")
        header = f"  {'Quality Scalar Variant':<25}"
        for s in range(num_seeds):
            header += f" | s{42+s*111}"
        header += " | Mean±Std"
        print(header)
        print(f"  {'-'*len(header)}")

        for var in variants:
            scores = []
            row = f"  {var:<25}"
            for s in range(num_seeds):
                seed = 42 + s * 111
                key = f"{ds}_{var}_s{seed}"
                mf1 = results.get(key, {}).get("macro_f1")
                if mf1 is not None:
                    scores.append(mf1)
                    row += f" | {mf1:.3f}"
                else:
                    row += f" |   -  "
            if scores:
                row += f" | {np.mean(scores):.3f}±{np.std(scores):.3f}"
            print(row)

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
