"""
Master Evaluation Suite for Reliable Federated ERC
===================================================
Runs the reliability evaluation script 'run_reliability.py' for:
  1. Standard EDL models (EAFA) vs CE models (FedAvg) across MELD, IEMOCAP, DailyDialog
  2. Severe Non-IID models (alpha=0.1 vs 0.5, EAFA vs FedAvg) on MELD

Generates a comprehensive summary markdown and JSON of results.
"""

import subprocess
import sys
import json
from pathlib import Path
import numpy as np

SEEDS = [42, 123, 2024]

def run_eval(args_list):
    cmd = [sys.executable, "scripts/run_reliability.py"] + args_list
    print(f"\nRunning: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=False, text=True)
    return res.returncode == 0

def extract_key_metrics(json_path):
    """Extract key metrics from JSON result file and average across seeds."""
    if not Path(json_path).exists():
        return None

    with open(json_path, 'r') as f:
        data = json.load(f)

    results_by_seed = data.get("per_seed_results", [])
    if not results_by_seed:
        return None

    # We will average the metrics across the seeds
    # 1. Selective prediction (AURC for different uncertainty scores)
    aurc_metrics = {}
    # 2. Conformal prediction (coverage, avg set size for LAC, APS, RAPS)
    conformal_metrics = {}
    # 3. Federated Conformal prediction (FCP)
    fcp_metrics = {}
    # 4. OOD AUROC
    ood_metrics = {}

    for seed_res in results_by_seed:
        # RQ1
        rq1 = seed_res.get("rq1_selective", {})
        for score_name, metrics in rq1.items():
            if score_name not in aurc_metrics:
                aurc_metrics[score_name] = []
            aurc_metrics[score_name].append(metrics.get("aurc", 0.0))

        # RQ2
        rq2 = seed_res.get("rq2_conformal", {})
        for method, metrics in rq2.items():
            if method not in conformal_metrics:
                conformal_metrics[method] = {"coverage": [], "set_size": []}
            conformal_metrics[method]["coverage"].append(metrics.get("coverage", 0.0))
            conformal_metrics[method]["set_size"].append(metrics.get("avg_set_size", 0.0))

        # RQ3
        rq3 = seed_res.get("rq3_fcp", {})
        for method, metrics in rq3.items():
            if method not in fcp_metrics:
                fcp_metrics[method] = {"coverage": [], "set_size": []}
            fcp_metrics[method]["coverage"].append(metrics.get("coverage", 0.0))
            fcp_metrics[method]["set_size"].append(metrics.get("avg_set_size", 0.0))

        # RQ4
        rq4 = seed_res.get("rq4_ood", {})
        if isinstance(rq4, dict):
            for score_name, metrics in rq4.items():
                if score_name not in ood_metrics:
                    ood_metrics[score_name] = []
                ood_metrics[score_name].append(metrics.get("auroc", 0.0))

    # Compute mean and std
    summary = {
        "dataset": data.get("dataset"),
        "model_type": data.get("model_type", "edl"),
        "conformal_alpha": data.get("conformal_alpha"),
        "aurc": {k: f"{np.mean(v):.4f}±{np.std(v, ddof=1):.4f}" if len(v)>1 else f"{np.mean(v):.4f}" for k, v in aurc_metrics.items()},
        "conformal": {},
        "fcp": {},
        "ood": {k: f"{np.mean(v):.4f}±{np.std(v, ddof=1):.4f}" if len(v)>1 else f"{np.mean(v):.4f}" for k, v in ood_metrics.items()}
    }

    for method, metrics in conformal_metrics.items():
        cov_mean = np.mean(metrics["coverage"])
        cov_std = np.std(metrics["coverage"], ddof=1) if len(metrics["coverage"])>1 else 0.0
        sz_mean = np.mean(metrics["set_size"])
        sz_std = np.std(metrics["set_size"], ddof=1) if len(metrics["set_size"])>1 else 0.0
        summary["conformal"][method] = {
            "coverage": f"{cov_mean:.4f}±{cov_std:.4f}",
            "set_size": f"{sz_mean:.2f}±{sz_std:.2f}"
        }

    for method, metrics in fcp_metrics.items():
        cov_mean = np.mean(metrics["coverage"])
        cov_std = np.std(metrics["coverage"], ddof=1) if len(metrics["coverage"])>1 else 0.0
        sz_mean = np.mean(metrics["set_size"])
        sz_std = np.std(metrics["set_size"], ddof=1) if len(metrics["set_size"])>1 else 0.0
        summary["fcp"][method] = {
            "coverage": f"{cov_mean:.4f}±{cov_std:.4f}",
            "set_size": f"{sz_mean:.2f}±{sz_std:.2f}"
        }

    return summary

def generate_markdown_report(summaries, out_path="results/reliability_summary.md"):
    lines = []
    lines.append("# Reliable Federated ERC - Comprehensive Evaluation Summary\n")
    lines.append("This document summarizes the findings from RQ1-RQ4 on the expanded reliability modules.\n")

    # Table 1: RQ1 Selective Prediction (AURC)
    lines.append("## RQ1: Selective Prediction (AURC Comparison)\n")
    lines.append("Lower AURC indicates better risk-coverage trade-off. We compare different uncertainty confidence signals:\n")
    lines.append("| Dataset | Model | Max Prob (AURC) | Entropy (AURC) | Vacuity 1-u (AURC) |\n")
    lines.append("|---------|-------|-----------------|----------------|--------------------|\n")
    for s in summaries:
        if "non_iid" in s.get("dataset", ""):
            continue
        ds = s["dataset"].upper()
        m = s["model_type"].upper()
        max_prob = s["aurc"].get("max_prob", "N/A")
        entropy = s["aurc"].get("neg_entropy", "N/A")
        vacuity = s["aurc"].get("vacuity_1mu", "N/A")
        lines.append(f"| {ds} | {m} | {max_prob} | {entropy} | {vacuity} |\n")

    lines.append("\n")

    # Table 2: RQ2 Conformal Prediction (Centralized)
    lines.append("## RQ2: Conformal Prediction (Efficiency vs Coverage)\n")
    lines.append("Target coverage: 90% (alpha = 0.1). We report actual coverage and average set size (lower set size at target coverage is better):\n")
    lines.append("| Dataset | Model | Method | Actual Coverage | Avg Set Size |\n")
    lines.append("|---------|-------|--------|-----------------|--------------|\n")
    for s in summaries:
        if "non_iid" in s.get("dataset", ""):
            continue
        ds = s["dataset"].upper()
        m = s["model_type"].upper()
        for method, metrics in s["conformal"].items():
            lines.append(f"| {ds} | {m} | {method} | {metrics['coverage']} | {metrics['set_size']} |\n")

    lines.append("\n")

    # Table 3: RQ3 Federated Conformal Prediction
    lines.append("## RQ3: Federated Conformal Prediction (FCP vs Centralized)\n")
    lines.append("Distributed quantile calibration (FCP) vs Centralized Conformal calibration:\n")
    lines.append("| Dataset | Model | Method | FCP Coverage | FCP Set Size |\n")
    lines.append("|---------|-------|--------|--------------|--------------|\n")
    for s in summaries:
        if "non_iid" in s.get("dataset", ""):
            continue
        ds = s["dataset"].upper()
        m = s["model_type"].upper()
        for method, metrics in s["fcp"].items():
            lines.append(f"| {ds} | {m} | {method} | {metrics['coverage']} | {metrics['set_size']} |\n")

    lines.append("\n")

    # Table 4: RQ4 OOD Detection (IEMOCAP only)
    lines.append("## RQ4: Out-of-Distribution Detection (Speaker Hold-out on IEMOCAP)\n")
    lines.append("Corrected split (ID = train sessions 1-3, OOD = session 5). AUROC for different uncertainty metrics:\n")
    for s in summaries:
        if s["dataset"] == "iemocap":
            m = s["model_type"].upper()
            lines.append(f"### Model: {m}\n")
            for score, val in s["ood"].items():
                lines.append(f"- **{score}**: AUROC = {val}\n")
            lines.append("\n")

    # Save
    with open(out_path, 'w') as f:
        f.writelines(lines)
    print(f"Markdown report generated at: {out_path}")

def main():
    # 1. MELD
    run_eval([
        "--dataset", "meld",
        "--num-classes", "7",
        "--model-type", "edl",
        "--ckpt", "checkpoints/best_eafa_edl_meld_seed{seed}.pt",
        "--out", "results/reliability_meld_edl.json"
    ])
    run_eval([
        "--dataset", "meld",
        "--num-classes", "7",
        "--model-type", "ce",
        "--ckpt", "checkpoints/best_fedavg_ce_meld_seed{seed}.pt",
        "--out", "results/reliability_meld_ce.json"
    ])

    # 2. IEMOCAP
    run_eval([
        "--dataset", "iemocap",
        "--num-classes", "6",
        "--model-type", "edl",
        "--ckpt", "checkpoints/best_eafa_edl_iemocap_seed{seed}.pt",
        "--out", "results/reliability_iemocap_edl.json"
    ])
    run_eval([
        "--dataset", "iemocap",
        "--num-classes", "6",
        "--model-type", "ce",
        "--ckpt", "checkpoints/best_fedavg_ce_iemocap_seed{seed}.pt",
        "--out", "results/reliability_iemocap_ce.json"
    ])

    # 3. DailyDialog
    run_eval([
        "--dataset", "dailydialog",
        "--num-classes", "6",
        "--model-type", "edl",
        "--ckpt", "checkpoints/best_eafa_edl_dailydialog_seed{seed}.pt",
        "--out", "results/reliability_dailydialog_edl.json"
    ])
    run_eval([
        "--dataset", "dailydialog",
        "--num-classes", "6",
        "--model-type", "ce",
        "--ckpt", "checkpoints/best_fedavg_ce_dailydialog_seed{seed}.pt",
        "--out", "results/reliability_dailydialog_ce.json"
    ])

    # 4. Severe Non-IID models
    for alpha_dir in ["0.1", "0.5"]:
        for method in ["eafa", "fedavg"]:
            run_eval([
                "--dataset", "meld",
                "--num-classes", "7",
                "--model-type", "edl",
                "--ckpt", f"checkpoints/best_non_iid_{method}_{alpha_dir}_seed{{seed}}.pt",
                "--out", f"results/reliability_meld_non_iid_{method}_{alpha_dir}.json"
            ])

    # Extract all results and generate summary report
    print("\nExtracting metrics and compiling report...")
    json_files = [
        "results/reliability_meld_edl.json",
        "results/reliability_meld_ce.json",
        "results/reliability_iemocap_edl.json",
        "results/reliability_iemocap_ce.json",
        "results/reliability_dailydialog_edl.json",
        "results/reliability_dailydialog_ce.json"
    ]
    
    summaries = []
    for f in json_files:
        s = extract_key_metrics(f)
        if s:
            summaries.append(s)

    generate_markdown_report(summaries)

if __name__ == "__main__":
    main()
