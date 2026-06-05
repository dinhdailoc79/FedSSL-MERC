"""
Dirichlet FixMatch Experiments
==============================
Runs the Dirichlet FixMatch baseline (hard thresholding on evidential belief
with Evidential DialogueRNN backbone) across different label ratios.

Usage:
    python scripts/run_dirichlet_fixmatch.py --dataset meld --seeds 42
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_FILE = "results_dirichlet_fixmatch.json"


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Dirichlet FixMatch Experiments")
    parser.add_argument("--dataset", type=str, default="meld", choices=["meld", "iemocap"])
    parser.add_argument("--seeds", type=str, default="42,123,456,789,2024")
    parser.add_argument("--label_ratios", type=str, default="0.05,0.10,0.50")
    args = parser.parse_args()

    from scripts.run_ssl_experiments import run_ssl_experiment

    results = load_results()
    
    seeds = [int(s) for s in args.seeds.split(",")]
    ratios = [float(r) for r in args.label_ratios.split(",")]
    
    total_start = time.time()
    
    experiments = []
    for lr in ratios:
        for seed in seeds:
            experiments.append((args.dataset, lr, seed))
            
    total = len(experiments)
    
    print(f"{'='*60}")
    print(f"  Dirichlet FixMatch Baselines")
    print(f"  Dataset: {args.dataset.upper()}")
    print(f"  Seeds:   {seeds}")
    print(f"  Ratios:  {ratios}")
    print(f"{'='*60}\n")
    
    for idx, (dataset, lr, seed) in enumerate(experiments):
        key = f"{dataset}_dirichlet_fixmatch_lr{lr:.2f}_s{seed}"
        
        if key in results and results[key].get("wf1") is not None:
            print(f"[{idx+1}/{total}] SKIP {key}: WF1={results[key]['wf1']}")
            continue
            
        print(f"\n[{idx+1}/{total}] RUNNING {key}...")
        start = time.time()
        
        try:
            r = run_ssl_experiment(dataset, "dirichlet_fixmatch", lr, seed=seed)
            elapsed = time.time() - start
            r["time"] = round(elapsed, 1)
            r["seed"] = seed
            results[key] = r
            save_results(results)
            print(f"  >> WF1={r['wf1']}, time={elapsed:.0f}s")
        except Exception as e:
            import traceback
            print(f"  >> ERROR: {e}")
            traceback.print_exc()
            results[key] = {"wf1": None, "error": str(e), "seed": seed}
            save_results(results)
            
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  Completed Dirichlet FixMatch Experiments in {total_time/60:.1f} minutes")
    print(f"{'='*60}")
    
    # Print summary table
    print(f"\n  Summary for {args.dataset.upper()}:")
    print(f"  {'Ratio':<6} | {'Mean WF1':<10} | {'Individual Seeds'}")
    print(f"  {'-'*50}")
    for lr in ratios:
        vals = []
        for seed in seeds:
            k = f"{args.dataset}_dirichlet_fixmatch_lr{lr:.2f}_s{seed}"
            if k in results and results[k].get("wf1") is not None:
                vals.append(results[k]["wf1"])
        if vals:
            mean_val = sum(vals) / len(vals)
            vals_str = ", ".join(f"{v:.4f}" for v in vals)
            print(f"  {lr:5.0%}  | {mean_val:.4f}   | [{vals_str}]")
        else:
            print(f"  {lr:5.0%}  | N/A        | []")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
