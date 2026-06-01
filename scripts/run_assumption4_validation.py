"""
Assumption 4 Empirical Validation
===================================
Validates the monotone correspondence between mean client uncertainty ū_k
and client "quality" (local validation error, gradient variance).

This script instruments the EAFA federated training loop to log, per round
per client:
  1. ū_k   — mean epistemic uncertainty from EDL (Dirichlet)
  2. Local validation error (1 - WF1 on global dev set)
  3. Gradient norm variance across local SGD steps

Then computes:
  - Spearman rank correlation between ū_k and val_error
  - Spearman rank correlation between ū_k and grad_var
  - Generates scatter plots for the paper

Usage:
    python scripts/run_assumption4_validation.py
    python scripts/run_assumption4_validation.py --dataset iemocap
"""

import sys
import os
import copy
import time
import json
import logging
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.evidential.evidential_dialogue_rnn import EvidentialDialogueRNN
from models.evidential.losses import SupervisedEvidentialLoss
from federated.aggregation.eafa import EAFAAggregator
from data.federated_partition import FederatedPartitioner

# Reuse data loading utilities from train_multi_dataset
from scripts.train_multi_dataset import (
    GenericDialogueDataset, collate_dialogues,
    load_meld, load_iemocap, evaluate,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_local(model, loader, device):
    """Evaluate a model on a loader, return WF1 and mean uncertainty."""
    model.eval()
    all_preds, all_labels, all_u = [], [], []
    for batch in loader:
        feats = batch["features"].to(device)
        speakers = batch["speaker_ids"].to(device)
        labels = batch["labels"].to(device)
        out = model(feats, speakers)
        mask = labels != -1
        preds = out["belief"][mask].argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels[mask].cpu().numpy())
        all_u.extend(out["uncertainty"][mask].cpu().numpy())
    wf1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    mean_u = np.mean(all_u) if all_u else 1.0
    return wf1, mean_u


def run_instrumented_federated(dataset_name, train_dias, dev_dias, test_dias,
                                emotions, weights, cache, num_speakers,
                                seed, device, num_clients=5, alpha=0.5,
                                num_rounds=50, local_epochs=3, beta=1.0):
    """
    Run EAFA federated training with full per-client instrumentation.
    Returns per-round per-client logs for correlation analysis.
    """
    num_classes = len(emotions)
    class_weights = torch.from_numpy(weights.astype(np.float32)).to(device)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Partition data
    partitioner = FederatedPartitioner(
        num_clients=num_clients, strategy="dirichlet",
        alpha=alpha, seed=seed,
    )
    client_partitions = partitioner.partition(train_dias, label_ratio=1.0)

    dialogue_lookup = {d.dialogue_id: d for d in train_dias}
    client_loaders = []
    for partition in client_partitions:
        dias = [dialogue_lookup[did] for did in partition.dialogue_ids
                if did in dialogue_lookup]
        ds = GenericDialogueDataset(dias, cache.get("train", {}))
        loader = DataLoader(ds, batch_size=16, shuffle=True,
                            collate_fn=collate_dialogues, num_workers=0)
        client_loaders.append(loader)

    # Dev loader (global) — for evaluating each client's local model quality
    dev_ds = GenericDialogueDataset(dev_dias, cache.get("dev", {}))
    dev_loader = DataLoader(dev_ds, batch_size=16, shuffle=False,
                            collate_fn=collate_dialogues, num_workers=0)

    test_ds = GenericDialogueDataset(test_dias, cache.get("test", {}))
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False,
                             collate_fn=collate_dialogues, num_workers=0)

    # Global model
    global_model = EvidentialDialogueRNN(
        input_dim=768, hidden_dim=256,
        num_classes=num_classes, num_speakers=num_speakers,
        dropout=0.3,
    ).to(device)

    loss_fn = SupervisedEvidentialLoss(
        num_classes=num_classes, annealing_epochs=30,
        class_weights=class_weights,
    )

    aggregator = EAFAAggregator(beta=beta)

    # ================================================================
    # Instrumentation storage
    # ================================================================
    round_logs = []  # List of dicts per round

    logger.info(f"\n{'='*60}")
    logger.info(f"  ASSUMPTION 4 VALIDATION — {dataset_name.upper()}")
    logger.info(f"  K={num_clients}, alpha={alpha}, beta={beta}, seed={seed}")
    logger.info(f"{'='*60}\n")

    for round_num in range(1, num_rounds + 1):
        start = time.time()
        client_states, client_sizes, client_us = [], [], []
        client_val_errors = []
        client_grad_vars = []

        loss_fn.set_epoch(round_num)

        for k, loader in enumerate(client_loaders):
            local_model = copy.deepcopy(global_model).to(device)
            local_model.train()
            opt = optim.Adam(local_model.parameters(), lr=1e-3, weight_decay=1e-4)
            all_u_local = []
            grad_norms = []  # Track gradient norms per step

            for _ in range(local_epochs):
                for batch in loader:
                    feats = batch["features"].to(device)
                    speakers = batch["speaker_ids"].to(device)
                    labels = batch["labels"].to(device)
                    out = local_model(feats, speakers)
                    mask = labels != -1
                    loss, _ = loss_fn(out["alpha"][mask], labels[mask])
                    all_u_local.extend(
                        out["uncertainty"][mask].detach().cpu().numpy()
                    )
                    opt.zero_grad()
                    loss.backward()

                    # ── Instrument: gradient norm ──
                    total_norm = 0.0
                    for p in local_model.parameters():
                        if p.grad is not None:
                            total_norm += p.grad.data.norm(2).item() ** 2
                    total_norm = total_norm ** 0.5
                    grad_norms.append(total_norm)

                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), 5.0)
                    opt.step()

            # ── Instrument: local validation error ──
            local_wf1, _ = evaluate_local(local_model, dev_loader, device)
            val_error = 1.0 - local_wf1

            # ── Instrument: gradient norm variance ──
            grad_var = float(np.var(grad_norms)) if grad_norms else 0.0
            grad_mean = float(np.mean(grad_norms)) if grad_norms else 0.0

            # Mean uncertainty
            mean_u = float(np.mean(all_u_local)) if all_u_local else 0.0

            client_states.append(
                OrderedDict({k_: v.cpu()
                             for k_, v in local_model.state_dict().items()})
            )
            client_sizes.append(len(loader.dataset))
            client_us.append(mean_u)
            client_val_errors.append(val_error)
            client_grad_vars.append(grad_var)

        # Aggregate
        global_state, agg_stats = aggregator.aggregate(
            client_states, client_sizes, client_us, round_num,
        )
        global_model.load_state_dict(global_state)
        global_model.to(device)

        test_wf1, test_u, _, _ = evaluate(global_model, test_loader, device)
        elapsed = time.time() - start

        # Log this round
        round_log = {
            "round": round_num,
            "test_wf1": float(test_wf1),
            "test_u": float(test_u),
            "clients": [],
        }
        for k in range(num_clients):
            round_log["clients"].append({
                "client_id": k,
                "u_k": client_us[k],
                "val_error": client_val_errors[k],
                "grad_var": client_grad_vars[k],
                "data_size": client_sizes[k],
                "eafa_weight": agg_stats["weights"][k],
            })

        round_logs.append(round_log)

        logger.info(
            f"Round {round_num:3d}/{num_rounds} | "
            f"Test WF1: {test_wf1:.4f} | "
            f"u_k=[{','.join(f'{u:.3f}' for u in client_us)}] | "
            f"val_err=[{','.join(f'{e:.3f}' for e in client_val_errors)}] | "
            f"{elapsed:.1f}s"
        )

        del client_states
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return round_logs


def analyze_correlation(round_logs, dataset_name, seed, output_dir):
    """
    Compute correlation between ū_k and val_error / grad_var.
    Generate scatter plots.
    """
    # Flatten all (round, client) pairs
    all_u, all_val_err, all_grad_var = [], [], []
    all_weights = []
    all_rounds = []

    for rlog in round_logs:
        for c in rlog["clients"]:
            all_u.append(c["u_k"])
            all_val_err.append(c["val_error"])
            all_grad_var.append(c["grad_var"])
            all_weights.append(c["eafa_weight"])
            all_rounds.append(rlog["round"])

    all_u = np.array(all_u)
    all_val_err = np.array(all_val_err)
    all_grad_var = np.array(all_grad_var)
    all_weights = np.array(all_weights)
    all_rounds = np.array(all_rounds)

    # Spearman correlations
    rho_val, p_val = spearmanr(all_u, all_val_err)
    rho_grad, p_grad = spearmanr(all_u, all_grad_var)
    rho_weight, p_weight = spearmanr(all_u, all_weights)

    # Pearson correlations
    r_val, rp_val = pearsonr(all_u, all_val_err)
    r_grad, rp_grad = pearsonr(all_u, all_grad_var)

    logger.info(f"\n{'='*60}")
    logger.info(f"  CORRELATION ANALYSIS — {dataset_name.upper()} (seed={seed})")
    logger.info(f"{'='*60}")
    logger.info(f"  N = {len(all_u)} (round × client) data points")
    logger.info(f"")
    logger.info(f"  ū_k vs Val Error:    Spearman ρ={rho_val:.4f} (p={p_val:.2e})")
    logger.info(f"                       Pearson  r={r_val:.4f}  (p={rp_val:.2e})")
    logger.info(f"  ū_k vs Grad Var:     Spearman ρ={rho_grad:.4f} (p={p_grad:.2e})")
    logger.info(f"                       Pearson  r={r_grad:.4f}  (p={rp_grad:.2e})")
    logger.info(f"  ū_k vs EAFA Weight:  Spearman ρ={rho_weight:.4f} (p={p_weight:.2e})")
    logger.info(f"{'='*60}\n")

    results = {
        "dataset": dataset_name,
        "seed": seed,
        "n_datapoints": len(all_u),
        "u_vs_val_error": {
            "spearman_rho": round(float(rho_val), 4),
            "spearman_p": float(p_val),
            "pearson_r": round(float(r_val), 4),
            "pearson_p": float(rp_val),
        },
        "u_vs_grad_var": {
            "spearman_rho": round(float(rho_grad), 4),
            "spearman_p": float(p_grad),
            "pearson_r": round(float(r_grad), 4),
            "pearson_p": float(rp_grad),
        },
        "u_vs_eafa_weight": {
            "spearman_rho": round(float(rho_weight), 4),
            "spearman_p": float(p_weight),
        },
    }

    # ── Generate scatter plots ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(
            f"Assumption 4 Validation: {dataset_name.upper()} (seed={seed})",
            fontsize=14, fontweight="bold",
        )

        norm = Normalize(vmin=all_rounds.min(), vmax=all_rounds.max())

        # Plot 1: ū_k vs Validation Error
        ax = axes[0]
        sc = ax.scatter(all_u, all_val_err, c=all_rounds, cmap="viridis",
                        alpha=0.6, s=30, norm=norm)
        ax.set_xlabel("Mean Epistemic Uncertainty (ū_k)", fontsize=11)
        ax.set_ylabel("Validation Error (1 - WF1)", fontsize=11)
        ax.set_title(f"ρ={rho_val:.3f}, p={p_val:.1e}", fontsize=11)

        # Add regression line
        z = np.polyfit(all_u, all_val_err, 1)
        p = np.poly1d(z)
        x_line = np.linspace(all_u.min(), all_u.max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
        ax.grid(True, alpha=0.3)

        # Plot 2: ū_k vs Gradient Variance
        ax = axes[1]
        ax.scatter(all_u, all_grad_var, c=all_rounds, cmap="viridis",
                   alpha=0.6, s=30, norm=norm)
        ax.set_xlabel("Mean Epistemic Uncertainty (ū_k)", fontsize=11)
        ax.set_ylabel("Gradient Norm Variance", fontsize=11)
        ax.set_title(f"ρ={rho_grad:.3f}, p={p_grad:.1e}", fontsize=11)

        z2 = np.polyfit(all_u, all_grad_var, 1)
        p2 = np.poly1d(z2)
        ax.plot(x_line, p2(x_line), "r--", alpha=0.8, linewidth=2)
        ax.grid(True, alpha=0.3)

        # Plot 3: ū_k vs EAFA Weight (should be NEGATIVE — high u → low weight)
        ax = axes[2]
        ax.scatter(all_u, all_weights, c=all_rounds, cmap="viridis",
                   alpha=0.6, s=30, norm=norm)
        ax.set_xlabel("Mean Epistemic Uncertainty (ū_k)", fontsize=11)
        ax.set_ylabel("EAFA Weight (w_k)", fontsize=11)
        ax.set_title(f"ρ={rho_weight:.3f}, p={p_weight:.1e}", fontsize=11)

        z3 = np.polyfit(all_u, all_weights, 1)
        p3 = np.poly1d(z3)
        ax.plot(x_line, p3(x_line), "r--", alpha=0.8, linewidth=2)
        ax.grid(True, alpha=0.3)

        plt.colorbar(sc, ax=axes, label="Communication Round", shrink=0.8)
        plt.tight_layout()

        plot_path = os.path.join(
            output_dir, f"assumption4_{dataset_name}_seed{seed}.png"
        )
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Scatter plot saved: {plot_path}")
        results["plot_path"] = plot_path

    except ImportError:
        logger.warning("matplotlib not available, skipping plots")

    # ── Per-round evolution plot ──
    try:
        fig2, axes2 = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig2.suptitle(
            f"Per-Round Client Metrics: {dataset_name.upper()} (seed={seed})",
            fontsize=14, fontweight="bold",
        )

        rounds = sorted(set(all_rounds))
        num_clients = len(round_logs[0]["clients"])
        colors = plt.cm.tab10(np.linspace(0, 1, num_clients))

        for k in range(num_clients):
            u_series = [round_logs[r]["clients"][k]["u_k"] for r in range(len(rounds))]
            ve_series = [round_logs[r]["clients"][k]["val_error"] for r in range(len(rounds))]

            axes2[0].plot(rounds, u_series, '-o', markersize=3,
                         color=colors[k], label=f"Client {k}", alpha=0.8)
            axes2[1].plot(rounds, ve_series, '-o', markersize=3,
                         color=colors[k], label=f"Client {k}", alpha=0.8)

        axes2[0].set_ylabel("ū_k (Mean Uncertainty)", fontsize=11)
        axes2[0].legend(fontsize=9)
        axes2[0].grid(True, alpha=0.3)

        axes2[1].set_ylabel("Validation Error (1 - WF1)", fontsize=11)
        axes2[1].set_xlabel("Communication Round", fontsize=11)
        axes2[1].legend(fontsize=9)
        axes2[1].grid(True, alpha=0.3)

        plt.tight_layout()
        evo_path = os.path.join(
            output_dir, f"assumption4_evolution_{dataset_name}_seed{seed}.png"
        )
        plt.savefig(evo_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Evolution plot saved: {evo_path}")
        results["evolution_plot_path"] = evo_path

    except Exception as e:
        logger.warning(f"Evolution plot failed: {e}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Assumption 4 Validation: ū_k correlation study"
    )
    parser.add_argument("--dataset", type=str, default="meld",
                        choices=["meld", "iemocap", "both"])
    parser.add_argument("--num_clients", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--num_rounds", type=int, default=50)
    parser.add_argument("--local_epochs", type=int, default=3)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    parser.add_argument("--output_dir", type=str, default="results/assumption4")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    datasets = ["meld", "iemocap"] if args.dataset == "both" else [args.dataset]

    all_results = {}

    for ds_name in datasets:
        logger.info(f"\n{'#'*60}")
        logger.info(f"  Loading {ds_name.upper()}")
        logger.info(f"{'#'*60}")

        if ds_name == "meld":
            train, dev, test, emotions, wts, cache, num_spk = load_meld(
                finetuned=True
            )
        elif ds_name == "iemocap":
            train, dev, test, emotions, wts, cache, num_spk = load_iemocap(
                finetuned=True, num_classes=6
            )
        else:
            continue

        ds_results = []
        for seed in args.seeds:
            logger.info(f"\n--- Seed {seed} ---")
            round_logs = run_instrumented_federated(
                dataset_name=ds_name,
                train_dias=train, dev_dias=dev, test_dias=test,
                emotions=emotions, weights=wts, cache=cache,
                num_speakers=num_spk, seed=seed, device=args.device,
                num_clients=args.num_clients, alpha=args.alpha,
                num_rounds=args.num_rounds, local_epochs=args.local_epochs,
                beta=args.beta,
            )

            result = analyze_correlation(
                round_logs, ds_name, seed, args.output_dir,
            )
            result["round_logs"] = round_logs
            ds_results.append(result)

        all_results[ds_name] = ds_results

        # ── Aggregate across seeds ──
        agg_rho_val = np.mean([r["u_vs_val_error"]["spearman_rho"]
                               for r in ds_results])
        agg_rho_grad = np.mean([r["u_vs_grad_var"]["spearman_rho"]
                                for r in ds_results])
        logger.info(f"\n{'='*60}")
        logger.info(f"  AGGREGATE — {ds_name.upper()} ({len(args.seeds)} seeds)")
        logger.info(f"  Mean Spearman ρ(ū_k, val_error) = {agg_rho_val:.4f}")
        logger.info(f"  Mean Spearman ρ(ū_k, grad_var)  = {agg_rho_grad:.4f}")
        logger.info(f"{'='*60}\n")

    # ── Save results (without large round_logs for the summary file) ──
    summary = {}
    for ds_name, ds_results in all_results.items():
        summary[ds_name] = []
        for r in ds_results:
            s = {k: v for k, v in r.items() if k != "round_logs"}
            summary[ds_name].append(s)

    summary_path = os.path.join(args.output_dir, "assumption4_results.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nResults saved: {summary_path}")

    # Save full logs separately
    full_path = os.path.join(args.output_dir, "assumption4_full_logs.json")
    full_save = {}
    for ds_name, ds_results in all_results.items():
        full_save[ds_name] = []
        for r in ds_results:
            entry = {
                "seed": r["seed"],
                "round_logs": r["round_logs"],
            }
            full_save[ds_name].append(entry)
    with open(full_path, "w") as f:
        json.dump(full_save, f, indent=2)
    logger.info(f"Full logs saved: {full_path}")

    # ── Print final summary table ──
    print(f"\n{'='*60}")
    print(f"  ASSUMPTION 4 VALIDATION — FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Dataset':<12} {'Seed':<8} {'rho(u,val_err)':<16} {'p-value':<12} {'rho(u,grad_var)':<16} {'p-value':<12}")
    print(f"  {'-'*76}")
    for ds_name, ds_results in all_results.items():
        for r in ds_results:
            uv = r["u_vs_val_error"]
            ug = r["u_vs_grad_var"]
            print(
                f"  {ds_name:<12} {r['seed']:<8} "
                f"{uv['spearman_rho']:<16.4f} {uv['spearman_p']:<12.2e} "
                f"{ug['spearman_rho']:<16.4f} {ug['spearman_p']:<12.2e}"
            )
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
