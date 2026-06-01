"""
Client Scalability Test: K=5 vs K=10 vs K=20
================================================
Addresses reviewer: "K=5 is too small, not realistic"

Run EAFA federated training at K=5, K=10, K=20 on MELD and IEMOCAP.
Uses subprocess to call train_multi_dataset.py with --num_clients flag.

Experiments: 3 K values x 2 datasets x 3 seeds = 18 runs
Results saved to results_client_scalability.json

Usage:
    cd D:\\OJT\\FedSSL-MERC
    python scripts/run_client_scalability.py
"""

import subprocess, sys, os, json, time, re
import numpy as np

RESULTS_FILE = "results_client_scalability.json"
SEEDS = [42, 123, 2024]
K_VALUES = [5, 10, 20]
DATASETS = ["meld", "iemocap"]


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)


def run_one(dataset, k, seed):
    """Run one federated EAFA experiment and parse WF1."""
    cmd = [
        sys.executable, "scripts/train_multi_dataset.py",
        "--dataset", dataset,
        "--mode", "federated",
        "--num_clients", str(k),
        "--alpha", "0.5",
        "--num_rounds", "50",
        "--local_epochs", "3",
        "--beta", "10.0",
        "--finetuned",
        "--seed", str(seed),
        "--patience", "15",
    ]
    
    if dataset == "iemocap":
        cmd += ["--iemocap_classes", "6"]
    
    print(f"    CMD: {' '.join(cmd[-8:])}")
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    output = result.stdout + result.stderr
    
    # Parse WF1 from output
    wf1 = None
    for line in output.split('\n'):
        if 'Test WF1' in line:
            match = re.search(r'WF1\s*=\s*([\d.]+)', line)
            if match:
                wf1 = float(match.group(1))
    
    if wf1 is None:
        # Try FINAL SUMMARY
        for line in output.split('\n'):
            if '_eafa' in line and 'WF1' in line:
                match = re.search(r'WF1\s*=\s*([\d.]+)', line)
                if match:
                    wf1 = float(match.group(1))
    
    return wf1


def main():
    results = load_results()
    
    total = len(DATASETS) * len(K_VALUES) * len(SEEDS)
    done = 0
    
    print("=" * 60)
    print("CLIENT SCALABILITY TEST: K=5, K=10, K=20")
    print(f"  {len(DATASETS)} datasets x {len(K_VALUES)} K values x {len(SEEDS)} seeds = {total} experiments")
    print("=" * 60)
    
    for dataset in DATASETS:
        for k in K_VALUES:
            for seed in SEEDS:
                key = f"{dataset}_K{k}_s{seed}"
                done += 1
                
                if key in results:
                    print(f"[{done}/{total}] SKIP {key}: WF1={results[key]['wf1']:.4f}")
                    continue
                
                print(f"\n[{done}/{total}] {dataset.upper()} K={k} seed={seed}")
                t0 = time.time()
                
                try:
                    wf1 = run_one(dataset, k, seed)
                    elapsed = time.time() - t0
                    
                    if wf1 is not None:
                        results[key] = {
                            "wf1": round(wf1, 4),
                            "dataset": dataset,
                            "K": k,
                            "seed": seed,
                            "time": round(elapsed, 1),
                        }
                        save_results(results)
                        print(f"    WF1={wf1:.4f} ({elapsed:.0f}s)")
                    else:
                        print(f"    ERROR: Could not parse WF1")
                except Exception as e:
                    print(f"    ERROR: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("CLIENT SCALABILITY RESULTS (WF1 %)")
    print(f"{'='*60}")
    
    for dataset in DATASETS:
        print(f"\n{dataset.upper()}:")
        print(f"  {'K':>4s} | {'WF1':>14s} | {'Time':>8s}")
        print(f"  {'-'*32}")
        for k in K_VALUES:
            vals = [results[f"{dataset}_K{k}_s{s}"]["wf1"]
                    for s in SEEDS if f"{dataset}_K{k}_s{s}" in results]
            times = [results[f"{dataset}_K{k}_s{s}"]["time"]
                     for s in SEEDS if f"{dataset}_K{k}_s{s}" in results]
            if vals:
                m = np.mean(vals) * 100
                s = np.std(vals) * 100
                t = np.mean(times)
                print(f"  {k:4d} | {m:5.1f}±{s:4.1f}     | {t:6.0f}s")
            else:
                print(f"  {k:4d} |     ---        |    ---")


if __name__ == "__main__":
    main()
