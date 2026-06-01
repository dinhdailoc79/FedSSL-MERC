"""
ECR Hyperparameter Tuning — Fast 2-Stage Approach
===================================================
Stage 1: Coarse search on MELD only, 1 seed → find top 5 configs (~25 min)
Stage 2: Validate top 5 on all 3 datasets, 3 seeds → confirm (~40 min)

Total: ~1 hour instead of days.

Usage:
    python scripts/run_ecr_tuning.py          # Run both stages
    python scripts/run_ecr_tuning.py --stage2  # Skip to stage 2 (if stage 1 done)
"""

import sys, os, json, time, copy
import numpy as np
import torch
import logging
from collections import OrderedDict
from itertools import product

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_FILE = "results_ecr_tuning.json"


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))


def load_dataset(dataset):
    """Load dataset once, return all needed components."""
    from scripts.train_multi_dataset import (
        GenericDialogueDataset, collate_dialogues, evaluate,
    )
    if dataset == "meld":
        from scripts.train_multi_dataset import load_meld
        return load_meld(finetuned=True)
    elif dataset == "iemocap":
        from scripts.train_multi_dataset import load_iemocap
        return load_iemocap(finetuned=True)
    elif dataset == "dailydialog":
        from scripts.train_multi_dataset import load_dailydialog
        return load_dailydialog(finetuned=True)


def run_ecr_experiment(dataset_data, sigma, lambda_max, ramp_start,
                       label_ratio=0.10, seed=42, num_rounds=50):
    """Run one ECR experiment with specific hyperparameters."""
    logging.basicConfig(level=logging.WARNING)

    from torch.utils.data import DataLoader
    from scripts.train_multi_dataset import (
        GenericDialogueDataset, collate_dialogues, evaluate,
    )
    from models.evidential.evidential_dialogue_rnn import EvidentialDialogueRNN
    from models.evidential.losses import SupervisedEvidentialLoss
    from federated.aggregation.eafa import EAFAAggregator
    from data.federated_partition import FederatedPartitioner

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dias, dev_dias, test_dias, emotions, weights, cache, num_spk = dataset_data
    num_classes = len(emotions)
    class_weights = torch.from_numpy(weights.astype(np.float32)).to(device)

    # Partition
    partitioner = FederatedPartitioner(
        num_clients=5, strategy="dirichlet", alpha=0.5, seed=seed,
    )
    client_partitions = partitioner.partition(train_dias, label_ratio=label_ratio)
    dialogue_lookup = {d.dialogue_id: d for d in train_dias}

    # Create loaders
    client_labeled_loaders, client_unlabeled_loaders = [], []
    for partition in client_partitions:
        labeled_dias = [dialogue_lookup[did] for did in partition.labeled_ids if did in dialogue_lookup]
        unlabeled_dias = [dialogue_lookup[did] for did in partition.unlabeled_ids if did in dialogue_lookup]

        labeled_ds = GenericDialogueDataset(labeled_dias, cache.get("train", {}))
        labeled_loader = DataLoader(labeled_ds, batch_size=16, shuffle=True, collate_fn=collate_dialogues, num_workers=0)
        client_labeled_loaders.append(labeled_loader)

        if unlabeled_dias:
            unlabeled_ds = GenericDialogueDataset(unlabeled_dias, cache.get("train", {}))
            unlabeled_loader = DataLoader(unlabeled_ds, batch_size=16, shuffle=True, collate_fn=collate_dialogues, num_workers=0)
        else:
            unlabeled_loader = None
        client_unlabeled_loaders.append(unlabeled_loader)

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
    aggregator = EAFAAggregator(beta=10.0)

    best_dev_wf1, patience_cnt = 0.0, 0
    best_test_wf1 = 0.0

    for round_num in range(1, num_rounds + 1):
        client_states, client_sizes, client_us = [], [], []
        loss_fn.set_epoch(round_num)

        # ECR ramp-up with tunable parameters
        lambda_u = 0.0
        if round_num > ramp_start:
            progress = (round_num - ramp_start) / 20.0
            lambda_u = lambda_max / (1.0 + np.exp(-10 * (progress - 0.5)))

        for k in range(5):
            model = copy.deepcopy(global_model)
            model.train()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            all_u_local = []

            for _ in range(3):
                for batch in client_labeled_loaders[k]:
                    feats = batch["features"].to(device)
                    speakers = batch["speaker_ids"].to(device)
                    labels = batch["labels"].to(device)
                    out = model(feats, speakers)
                    mask = labels != -1
                    sup_loss, _ = loss_fn(out["alpha"][mask], labels[mask])
                    all_u_local.extend(out["uncertainty"][mask].detach().cpu().numpy())

                    # ECR SSL loss
                    ssl_loss = torch.tensor(0.0, device=device)
                    if lambda_u > 0 and client_unlabeled_loaders[k] is not None:
                        try:
                            u_batch = next(iter(client_unlabeled_loaders[k]))
                            u_feats = u_batch["features"].to(device)
                            u_speakers = u_batch["speaker_ids"].to(device)
                            u_labels_raw = u_batch["labels"].to(device)
                            u_mask = u_labels_raw != -1

                            with torch.no_grad():
                                weak_out = model(u_feats, u_speakers)

                            noisy_feats = u_feats + torch.randn_like(u_feats) * sigma
                            strong_out = model(noisy_feats, u_speakers)

                            alpha_w = weak_out["alpha"][u_mask]
                            alpha_s = strong_out["alpha"][u_mask]
                            u_w = weak_out["uncertainty"][u_mask]
                            certainty = (1.0 - u_w).clamp(min=0.01)

                            S_w = alpha_w.sum(dim=-1, keepdim=True)
                            S_s = alpha_s.sum(dim=-1, keepdim=True)
                            kl = (torch.lgamma(S_s) - torch.lgamma(S_w)
                                  - (torch.lgamma(alpha_s) - torch.lgamma(alpha_w)).sum(dim=-1, keepdim=True)
                                  + ((alpha_s - alpha_w) * (torch.digamma(alpha_s) - torch.digamma(S_s))).sum(dim=-1, keepdim=True))
                            ssl_loss = (certainty * kl.squeeze(-1)).mean()

                        except StopIteration:
                            pass

                    total_loss = sup_loss + lambda_u * ssl_loss
                    opt.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()

            client_states.append(OrderedDict({key: v.cpu() for key, v in model.state_dict().items()}))
            client_sizes.append(len(client_labeled_loaders[k].dataset) +
                              (len(client_unlabeled_loaders[k].dataset) if client_unlabeled_loaders[k] else 0))
            client_us.append(float(np.mean(all_u_local)) if all_u_local else 1.0)

        aggregated_state, _ = aggregator.aggregate(
            client_states, client_sizes, client_us, round_num,
        )
        global_model.load_state_dict(aggregated_state)
        global_model.to(device)

        dev_wf1, _, _, _ = evaluate(global_model, dev_loader, device)
        if dev_wf1 > best_dev_wf1:
            best_dev_wf1 = dev_wf1
            patience_cnt = 0
            test_wf1, _, _, _ = evaluate(global_model, test_loader, device)
            best_test_wf1 = test_wf1
        else:
            patience_cnt += 1

        if patience_cnt > 12:  # Slightly more aggressive early stopping for speed
            break
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {"wf1": round(best_test_wf1, 6), "dev_wf1": round(best_dev_wf1, 6)}


def main():
    results = load_results()
    total_start = time.time()
    skip_to_stage2 = "--stage2" in sys.argv

    # ============================================================
    #  STAGE 1: Coarse search on MELD only, seed=42
    #  48 configs × 1 dataset × 1 seed = 48 experiments (~25 min)
    # ============================================================
    sigmas = [0.01, 0.05, 0.10, 0.20]
    lambda_maxs = [0.3, 0.5, 1.0, 2.0]
    ramp_starts = [3, 5, 10]
    configs = list(product(sigmas, lambda_maxs, ramp_starts))

    if not skip_to_stage2:
        print("=" * 60)
        print("  STAGE 1: Coarse Search on MELD (48 configs × 1 seed)")
        print("=" * 60)

        ds_data = load_dataset("meld")
        done, skipped = 0, 0

        for i, (sigma, lam, ramp) in enumerate(configs):
            key = f"meld_ecr_s{sigma}_l{lam}_r{ramp}_seed42"
            if key in results and results[key].get("wf1") is not None:
                skipped += 1
                continue

            print(f"  [{i+1}/48] sigma={sigma} lambda={lam} ramp={ramp} ...", end=" ", flush=True)
            start = time.time()
            try:
                r = run_ecr_experiment(ds_data, sigma, lam, ramp, label_ratio=0.10, seed=42)
                elapsed = time.time() - start
                r.update({"time": round(elapsed, 1), "sigma": sigma, "lambda_max": lam,
                          "ramp_start": ramp, "dataset": "meld", "seed": 42})
                results[key] = r
                save_results(results)
                done += 1
                print(f"WF1={r['wf1']:.4f} ({elapsed:.0f}s)")
            except Exception as e:
                print(f"ERROR: {e}")
                results[key] = {"wf1": None, "error": str(e)}
                save_results(results)

        print(f"\n  Stage 1 done: {done} new, {skipped} skipped\n")

    # Find top 5 configs from stage 1
    stage1_scores = {}
    for sigma, lam, ramp in configs:
        key = f"meld_ecr_s{sigma}_l{lam}_r{ramp}_seed42"
        v = results.get(key, {}).get("wf1")
        if v is not None:
            stage1_scores[(sigma, lam, ramp)] = v

    if not stage1_scores:
        print("ERROR: No stage 1 results found. Run without --stage2 first.")
        return

    sorted_s1 = sorted(stage1_scores.items(), key=lambda x: -x[1])
    top5 = [cfg for cfg, _ in sorted_s1[:5]]

    print("=" * 60)
    print("  STAGE 1 RESULTS — Top 5 Configs (MELD, seed=42)")
    print("=" * 60)
    current_wf1 = stage1_scores.get((0.05, 1.0, 5), 0)
    print(f"  Current default (s=0.05, l=1.0, r=5): WF1={current_wf1:.4f}\n")
    for i, (cfg, score) in enumerate(sorted_s1[:5]):
        marker = " ★ CURRENT" if cfg == (0.05, 1.0, 5) else ""
        print(f"  #{i+1}: sigma={cfg[0]}, lambda={cfg[1]}, ramp={cfg[2]} → WF1={score:.4f}{marker}")
    print()

    # ============================================================
    #  STAGE 2: Validate top 5 on all 3 datasets × 3 seeds
    #  5 configs × 3 datasets × 3 seeds = 45 experiments (~40 min)
    # ============================================================
    print("=" * 60)
    print(f"  STAGE 2: Validate Top 5 on All Datasets (45 experiments)")
    print("=" * 60)

    datasets = ["meld", "iemocap", "dailydialog"]
    seeds = [42, 123, 456]

    for ds in datasets:
        print(f"\n  --- {ds.upper()} ---")
        ds_data = load_dataset(ds)
        for sigma, lam, ramp in top5:
            for seed in seeds:
                key = f"{ds}_ecr_s{sigma}_l{lam}_r{ramp}_seed{seed}"
                if key in results and results[key].get("wf1") is not None:
                    continue

                print(f"    s={sigma} l={lam} r={ramp} seed={seed} ...", end=" ", flush=True)
                start = time.time()
                try:
                    r = run_ecr_experiment(ds_data, sigma, lam, ramp, label_ratio=0.10, seed=seed)
                    elapsed = time.time() - start
                    r.update({"time": round(elapsed, 1), "sigma": sigma, "lambda_max": lam,
                              "ramp_start": ramp, "dataset": ds, "seed": seed})
                    results[key] = r
                    save_results(results)
                    print(f"WF1={r['wf1']:.4f} ({elapsed:.0f}s)")
                except Exception as e:
                    print(f"ERROR: {e}")
                    results[key] = {"wf1": None, "error": str(e)}
                    save_results(results)

    # ============================================================
    #  FINAL ANALYSIS
    # ============================================================
    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  FINAL ANALYSIS — {total_time/60:.1f} minutes total")
    print(f"{'='*70}")

    for ds in datasets:
        print(f"\n  {ds.upper()} — Top 5 Configs (3-seed average):")
        cfg_means = {}
        for sigma, lam, ramp in top5:
            vals = []
            for seed in seeds:
                key = f"{ds}_ecr_s{sigma}_l{lam}_r{ramp}_seed{seed}"
                v = results.get(key, {}).get("wf1")
                if v is not None:
                    vals.append(v)
            if vals:
                cfg_means[(sigma, lam, ramp)] = (np.mean(vals), np.std(vals, ddof=1) if len(vals) > 1 else 0)

        for cfg, (m, s) in sorted(cfg_means.items(), key=lambda x: -x[1][0]):
            marker = " ★" if cfg == (0.05, 1.0, 5) else ""
            print(f"    s={cfg[0]}, l={cfg[1]}, r={cfg[2]} → {m*100:.2f}±{s*100:.2f}%{marker}")

    # Find single best config that works across all datasets
    print(f"\n{'='*70}")
    print(f"  BEST OVERALL CONFIG (sum of ranks)")
    print(f"{'='*70}")
    cfg_rank_sum = {cfg: 0 for cfg in top5}
    for ds in datasets:
        ds_scores = []
        for cfg in top5:
            vals = [results.get(f"{ds}_ecr_s{cfg[0]}_l{cfg[1]}_r{cfg[2]}_seed{s}", {}).get("wf1", 0) for s in seeds]
            ds_scores.append((cfg, np.mean([v for v in vals if v]) if any(v for v in vals) else 0))
        ds_sorted = sorted(ds_scores, key=lambda x: -x[1])
        for rank, (cfg, _) in enumerate(ds_sorted):
            cfg_rank_sum[cfg] += rank

    best_overall = sorted(cfg_rank_sum.items(), key=lambda x: x[1])
    for cfg, rank_sum in best_overall:
        marker = " ★ CURRENT" if cfg == (0.05, 1.0, 5) else ""
        print(f"  sigma={cfg[0]}, lambda={cfg[1]}, ramp={cfg[2]} → rank_sum={rank_sum}{marker}")

    print(f"\n  RECOMMENDATION: Use sigma={best_overall[0][0][0]}, lambda={best_overall[0][0][1]}, ramp={best_overall[0][0][2]}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
