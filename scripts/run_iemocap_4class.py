"""
IEMOCAP 4-class experiment for fair SOTA comparison.
Merges happy+excited → happy, drops frustrated.
Runs federated EAFA + centralized × 3 seeds.
"""
import subprocess, json, os, time, sys, re
import numpy as np

SEEDS = [42, 123, 2024]
RESULTS_FILE = "results_iemocap_4class.json"

def load_results():
    if os.path.exists(RESULTS_FILE):
        return json.load(open(RESULTS_FILE))
    return {}

def save_results(results):
    json.dump(results, open(RESULTS_FILE, "w"), indent=2)

def run_one(key, cmd):
    results = load_results()
    if key in results and results[key].get("wf1") is not None:
        print(f"  SKIP {key}")
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
    
    result = {"wf1": wf1, "time": round(elapsed, 1)}
    results = load_results()
    results[key] = result
    save_results(results)
    
    status = f"WF1={wf1:.4f}" if wf1 else "FAILED"
    print(f"    {status} ({elapsed:.0f}s)")
    return result

def main():
    print("=" * 60)
    print("IEMOCAP 4-class Experiment (happy+excited merged)")
    print("  3 seeds x 2 modes = 6 experiments")
    print("=" * 60)
    
    base_args = [
        sys.executable, "scripts/train_multi_dataset.py",
        "--dataset", "iemocap",
        "--finetuned",
        "--iemocap_classes", "4",
        "--epochs", "50",
        "--patience", "15",
    ]
    
    # Part 1: Federated EAFA (4-class)
    print("\n--- Part 1: Federated EAFA (4-class) ---")
    for seed in SEEDS:
        key = f"iemocap4_fed_eafa_seed{seed}"
        cmd = base_args + ["--mode", "federated", "--seed", str(seed)]
        run_one(key, cmd)
    
    # Part 2: Centralized EDL (4-class)
    print("\n--- Part 2: Centralized EDL (4-class) ---")
    for seed in SEEDS:
        key = f"iemocap4_cent_edl_seed{seed}"
        cmd = base_args + ["--mode", "centralized", "--seed", str(seed)]
        run_one(key, cmd)
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY — IEMOCAP 4-class")
    print("=" * 60)
    
    results = load_results()
    
    for config in ["fed_eafa", "cent_edl"]:
        label = "Federated EAFA" if "fed" in config else "Centralized EDL"
        wf1s = []
        for seed in SEEDS:
            key = f"iemocap4_{config}_seed{seed}"
            if key in results and results[key].get("wf1"):
                wf1s.append(results[key]["wf1"])
        if wf1s:
            mean = np.mean(wf1s)
            std = np.std(wf1s)
            print(f"  {label:20s}: {mean:.4f} +/- {std:.4f} (n={len(wf1s)})")
    
    print("\n  SOTA Comparison (4-class, centralized, 100% labels):")
    print("    DialogueRNN (GloVe, 5M):       62.57%")
    print("    COSMIC (RoBERTa-L, 355M):      65.28%")
    print("    EmoBERTa (RoBERTa-L, 355M):    68.57%")
    print("\n  Note: Our model uses RoBERTa-Base (125M) in Federated setting")

if __name__ == "__main__":
    main()
