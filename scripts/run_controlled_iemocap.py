"""
Controlled IEMOCAP 4-class Comparison
======================================
All methods use SAME RoBERTa-Base finetuned features.
Isolates EAFA/ECR contribution from feature quality.

Configs: CE Centralized, EDL Centralized, EDL FedAvg, EDL EAFA
Seeds: 42, 123, 2024
Total: 4 configs x 3 seeds = 12 experiments

Usage:
    cd D:\\OJT\\FedSSL-MERC
    python scripts/run_controlled_iemocap.py
"""
import subprocess, json, os, time, sys, re
import numpy as np

SEEDS = [42, 123, 2024]
RESULTS_FILE = "results_controlled_iemocap.json"


def load_results():
    if os.path.exists(RESULTS_FILE):
        return json.load(open(RESULTS_FILE))
    return {}


def save_results(results):
    json.dump(results, open(RESULTS_FILE, "w"), indent=2)


def run_one(key, cmd):
    results = load_results()
    if key in results and results[key].get("wf1") is not None:
        print(f"  SKIP {key}: WF1={results[key]['wf1']:.4f}")
        return results[key]
    
    print(f"  RUN {key}...")
    start = time.time()
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    elapsed = time.time() - start
    
    wf1 = None
    for line in r.stdout.split("\n"):
        m = re.search(r"WF1\s*[:=]\s*([\d.]+)", line)
        if m:
            wf1 = float(m.group(1))
    
    result = {"wf1": wf1, "config": key, "time": round(elapsed, 1)}
    results = load_results()
    results[key] = result
    save_results(results)
    
    status = f"WF1={wf1:.4f}" if wf1 else "FAILED"
    print(f"    {status} ({elapsed:.0f}s)")
    return result


def main():
    print("=" * 60)
    print("CONTROLLED IEMOCAP 4-CLASS COMPARISON")
    print("(All methods use same RoBERTa-Base finetuned features)")
    print("  4 configs x 3 seeds = 12 experiments")
    print("=" * 60)
    
    base_args = [
        sys.executable, "scripts/train_multi_dataset.py",
        "--dataset", "iemocap",
        "--finetuned",
        "--iemocap_classes", "4",
        "--epochs", "50",
        "--patience", "15",
    ]
    
    configs = {
        "centralized_ce": ["--mode", "centralized", "--loss_type", "ce"],
        "centralized_edl": ["--mode", "centralized", "--loss_type", "edl"],
        "fedavg_edl": ["--mode", "federated", "--loss_type", "edl", "--aggregation", "fedavg", "--beta", "0"],
        "eafa_edl": ["--mode", "federated", "--loss_type", "edl", "--aggregation", "eafa", "--beta", "10"],
    }
    
    for config_name, extra_args in configs.items():
        print(f"\n--- {config_name} ---")
        for seed in SEEDS:
            key = f"{config_name}_s{seed}"
            cmd = base_args + extra_args + ["--seed", str(seed)]
            run_one(key, cmd)
    
    # Summary
    print(f"\n{'='*60}")
    print("CONTROLLED IEMOCAP 4-CLASS COMPARISON")
    print("(All methods use same RoBERTa-Base finetuned features)")
    print(f"{'='*60}")
    
    results = load_results()
    for config_name in configs:
        wf1s = []
        for seed in SEEDS:
            key = f"{config_name}_s{seed}"
            if key in results and results[key].get("wf1"):
                wf1s.append(results[key]["wf1"])
        if wf1s:
            m = np.mean(wf1s) * 100
            s = np.std(wf1s) * 100
            print(f"  {config_name:25s}: {m:.2f} +/- {s:.2f}  (n={len(wf1s)})")


if __name__ == "__main__":
    main()
