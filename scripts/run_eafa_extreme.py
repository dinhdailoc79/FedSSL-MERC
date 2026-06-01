"""
P1: EAFA Extreme Conditions Test
=================================
Test EAFA vs FedAvg under extreme Non-IID + high noise.
Goal: Prove EAFA is a real contribution (>1% gain).

Conditions:
- Dirichlet alpha: 0.1 (extreme), 0.3 (high)
- Noise: 30%, 50%
- Both datasets, 5 seeds
- Compare EAFA (beta=10) vs FedAvg (beta=0)

Total: 2 alpha × 2 noise × 2 datasets × 2 methods × 5 seeds = 80 experiments

Usage:
    python scripts/run_eafa_extreme.py
"""

import sys, os, json, time, copy
import numpy as np
import torch
import logging
from collections import OrderedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_FILE = "results_eafa_extreme.json"
SEEDS = [42, 123, 456, 789, 2024]


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))


def run_eafa_extreme(dataset, beta, noise_rate, alpha_dir, seed=42):
    """
    Run federated experiment with variable Dirichlet alpha and noise.
    Uses same API as train_multi_dataset.py (model(feats, speakers), etc.)
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    from torch.utils.data import DataLoader
    from sklearn.metrics import f1_score

    torch.manual_seed(seed)
    np.random.seed(seed)

    from scripts.train_multi_dataset import (
        load_meld, load_iemocap,
        GenericDialogueDataset, collate_dialogues, evaluate,
    )
    from scripts.label_noise import inject_label_noise
    from models.evidential.evidential_dialogue_rnn import EvidentialDialogueRNN
    from models.evidential.losses import SupervisedEvidentialLoss
    from federated.aggregation.eafa import EAFAAggregator
    from data.federated_partition import FederatedPartitioner

    loaders_map = {"meld": load_meld, "iemocap": load_iemocap}

    # Load data
    load_fn = loaders_map[dataset]
    train_dias, dev_dias, test_dias, emotions, weights, cache, num_spk = load_fn(finetuned=True)
    num_classes = len(emotions)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_weights = torch.from_numpy(weights.astype(np.float32)).to(device)

    # Key difference: variable Dirichlet alpha
    partitioner = FederatedPartitioner(
        num_clients=5, strategy="dirichlet", alpha=alpha_dir, seed=seed,
    )
    client_partitions = partitioner.partition(train_dias, label_ratio=1.0)
    dialogue_lookup = {d.dialogue_id: d for d in train_dias}

    # More aggressive noise: spread across more clients
    # Clients 2,3,4 get noise (3 out of 5 = 60% corrupted)
    noise_config = {
        2: noise_rate * 0.5,
        3: noise_rate,
        4: min(noise_rate * 1.5, 0.8),
    }

    client_loaders = []
    for idx, partition in enumerate(client_partitions):
        dias = [dialogue_lookup[did] for did in partition.dialogue_ids if did in dialogue_lookup]
        client_noise = noise_config.get(idx, 0.0)
        if client_noise > 0:
            dias, _ = inject_label_noise(dias, client_noise, num_classes, seed=seed + idx)
        ds = GenericDialogueDataset(dias, cache.get("train", {}))
        loader = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate_dialogues, num_workers=0)
        client_loaders.append(loader)

    # Dev + Test
    dev_ds = GenericDialogueDataset(dev_dias, cache.get("dev", {}))
    dev_loader = DataLoader(dev_ds, batch_size=16, shuffle=False, collate_fn=collate_dialogues, num_workers=0)
    test_ds = GenericDialogueDataset(test_dias, cache.get("test", {}))
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_dialogues, num_workers=0)

    # Model
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
    logger.info(f"  {agg_label} | {dataset.upper()} | alpha={alpha_dir} | noise={noise_rate:.0%} | seed={seed}")
    logger.info(f"{'='*60}")

    best_dev_wf1, patience_cnt = 0.0, 0
    best_test_wf1, best_test_u = 0.0, 1.0
    final_weights = []
    final_client_us = []

    for round_num in range(1, 51):
        start = time.time()
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

        # Aggregate
        if is_eafa:
            aggregated_state, agg_stats = aggregator.aggregate(
                client_states, client_sizes, client_us, round_num,
            )
            agg_weights = agg_stats.get("weights", [])
        else:
            total_sz = sum(client_sizes)
            aggregated_state = OrderedDict()
            for key in client_states[0]:
                aggregated_state[key] = sum(
                    client_states[i][key] * (client_sizes[i] / total_sz)
                    for i in range(len(client_states))
                )
            agg_weights = [s / total_sz for s in client_sizes]

        global_model.load_state_dict(aggregated_state)
        global_model.to(device)
        elapsed = time.time() - start

        # Evaluate
        dev_wf1, dev_u, _, _ = evaluate(global_model, dev_loader, device)
        if round_num % 10 == 0 or round_num <= 3:
            u_mean = np.mean(client_us)
            u_std = np.std(client_us)
            logger.info(
                f"R{round_num:2d}/50 | WF1={dev_wf1:.4f} | {elapsed:.1f}s | "
                f"u_mean={u_mean:.4f} u_std={u_std:.4f}"
            )

        if dev_wf1 > best_dev_wf1:
            best_dev_wf1 = dev_wf1
            patience_cnt = 0
            test_wf1, test_u, _, _ = evaluate(global_model, test_loader, device)
            best_test_wf1 = test_wf1
            best_test_u = test_u
            final_weights = agg_weights
            final_client_us = client_us
        else:
            patience_cnt += 1

        if patience_cnt > 15:
            break

        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    logger.info(f"  RESULT: {agg_label} | {dataset} | alpha={alpha_dir} | noise={noise_rate} | WF1={best_test_wf1:.4f}")

    return {
        "wf1": round(best_test_wf1, 6),
        "uncertainty": round(best_test_u, 6),
        "dev_wf1": round(best_dev_wf1, 6),
        "weights": [round(w, 3) for w in final_weights] if final_weights else [],
        "client_uncertainties": [round(u, 4) for u in final_client_us] if final_client_us else [],
        "alpha_dir": alpha_dir,
    }


def main():
    results = load_results()
    # Clear failed results
    results = {k: v for k, v in results.items() if v.get("wf1") is not None}
    save_results(results)

    total_start = time.time()

    datasets = ["meld", "iemocap"]
    alphas = [0.1, 0.3]
    noises = [0.3, 0.5]
    betas = [(10.0, "eafa"), (0.0, "fedavg")]

    experiments = []
    for dataset in datasets:
        for alpha_dir in alphas:
            for noise in noises:
                for beta, method in betas:
                    for seed in SEEDS:
                        key = f"{dataset}_{method}_a{alpha_dir}_n{noise}_s{seed}"
                        experiments.append((key, dataset, beta, noise, alpha_dir, seed))

    total = len(experiments)
    done, skipped = 0, 0

    print(f"{'='*60}")
    print(f"  P1: EAFA Extreme Conditions")
    print(f"  Alphas: {alphas} | Noises: {noises}")
    print(f"  Total: {total} experiments")
    print(f"{'='*60}\n")

    for idx, (key, dataset, beta, noise, alpha_dir, seed) in enumerate(experiments):
        if key in results and results[key].get("wf1") is not None:
            skipped += 1
            continue

        print(f"\n[{idx+1}/{total}] {key}...")
        start = time.time()

        try:
            r = run_eafa_extreme(dataset, beta, noise, alpha_dir, seed)
            elapsed = time.time() - start
            r["time"] = round(elapsed, 1)
            r["beta"] = beta
            r["noise_rate"] = noise
            r["seed"] = seed
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

    # ========== Summary ==========
    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  EAFA EXTREME RESULTS -- {total_time/60:.1f} minutes")
    print(f"  Done: {done}, Skipped: {skipped}")
    print(f"{'='*70}")

    for dataset in datasets:
        for alpha_dir in alphas:
            print(f"\n  {dataset.upper()} | alpha={alpha_dir}:")
            print(f"  {'Noise':>5} | {'EAFA mean±std':>16} | {'FedAvg mean±std':>16} | {'Delta':>8} | {'p-val':>8}")
            print(f"  {'-'*5}-+-{'-'*16}-+-{'-'*16}-+-{'-'*8}-+-{'-'*8}")

            for noise in noises:
                eafa_vals, fedavg_vals = [], []
                for seed in SEEDS:
                    ek = f"{dataset}_eafa_a{alpha_dir}_n{noise}_s{seed}"
                    fk = f"{dataset}_fedavg_a{alpha_dir}_n{noise}_s{seed}"
                    ev = results.get(ek, {}).get("wf1")
                    fv = results.get(fk, {}).get("wf1")
                    if ev is not None: eafa_vals.append(ev)
                    if fv is not None: fedavg_vals.append(fv)

                if len(eafa_vals) >= 3 and len(fedavg_vals) >= 3:
                    from scipy import stats as sp
                    n = min(len(eafa_vals), len(fedavg_vals))
                    em, fm = np.mean(eafa_vals[:n]), np.mean(fedavg_vals[:n])
                    es, fs = np.std(eafa_vals[:n], ddof=1), np.std(fedavg_vals[:n], ddof=1)
                    delta = em - fm
                    _, pv = sp.ttest_rel(eafa_vals[:n], fedavg_vals[:n])
                    print(f"  {int(noise*100):4d}% | {em:.4f}±{es:.4f} | {fm:.4f}±{fs:.4f} | {delta:+.4f} | {pv:.5f}")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
