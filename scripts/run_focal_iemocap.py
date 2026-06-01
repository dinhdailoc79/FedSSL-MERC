"""
Focal Loss experiment for IEMOCAP 6-class.
Tests gamma = [0, 0.5, 1.0, 1.5, 2.0] x seeds x [centralized, federated]
Also runs IEMOCAP 4-class for SOTA comparison.
"""
import subprocess, json, os, time, sys

GAMMAS = [0.0, 0.5, 1.0, 1.5, 2.0]
SEEDS = [42, 123, 2024]
RESULTS_FILE = "results_focal_iemocap.json"

def load_results():
    if os.path.exists(RESULTS_FILE):
        return json.load(open(RESULTS_FILE))
    return {}

def save_results(results):
    json.dump(results, open(RESULTS_FILE, "w"), indent=2)

def run_experiment(key, cmd):
    results = load_results()
    if key in results and results[key].get("wf1") is not None:
        print(f"  SKIP {key} (already done)")
        return results[key]
    
    print(f"  RUN {key}...")
    start = time.time()
    r = subprocess.run(cmd, capture_output=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    elapsed = time.time() - start
    
    # Extract WF1 — handle both "WF1: 0.5926" and "WF1 = 0.5926"
    # Note: logging goes to stderr, so we merge stdout+stderr above
    import re
    wf1 = None
    micro = None
    for line in r.stdout.split("\n"):
        if "Test WF1" in line or "FINAL" in line or "WF1" in line:
            m = re.search(r"WF1\s*[:=]\s*([\d.]+)", line)
            if m:
                wf1 = float(m.group(1))
            m2 = re.search(r"Micro\s*(?:F1)?\s*[:=]\s*([\d.]+)", line)
            if m2:
                micro = float(m2.group(1))
    
    if wf1 is None:
        # Try alternative pattern
        for line in r.stdout.split("\n"):
            import re
            m = re.search(r"wf1['\"]?\s*[:=]\s*(0\.\d+)", line, re.IGNORECASE)
            if m:
                wf1 = float(m.group(1))
    
    result = {
        "wf1": wf1,
        "micro": micro,
        "time": round(elapsed, 1),
    }
    
    results = load_results()
    results[key] = result
    save_results(results)
    
    status = f"WF1={wf1:.4f}" if wf1 else "FAILED"
    print(f"    {status} ({elapsed:.0f}s)")
    return result

def main():
    print("=" * 60)
    print("IEMOCAP Focal Loss Experiment")
    print(f"  {len(GAMMAS)} gammas x {len(SEEDS)} seeds x 2 modes = {len(GAMMAS)*len(SEEDS)*2} experiments")
    print("=" * 60)
    
    base_cmd = [
        sys.executable, "scripts/train_multi_dataset.py",
        "--dataset", "iemocap",
        "--finetuned",
        "--epochs", "50",
        "--patience", "15",
    ]
    
    # Part 1: Focal gamma sweep (federated EAFA)
    print("\n--- Part 1: Focal Gamma Sweep (Federated EAFA) ---")
    for gamma in GAMMAS:
        for seed in SEEDS:
            key = f"iemocap_fed_gamma{gamma}_seed{seed}"
            cmd = base_cmd + [
                "--mode", "federated",
                "--focal_gamma", str(gamma),
                "--seed", str(seed),
            ]
            run_experiment(key, cmd)
    
    # Part 2: Focal gamma sweep (centralized)
    print("\n--- Part 2: Focal Gamma Sweep (Centralized) ---")
    for gamma in GAMMAS:
        for seed in SEEDS:
            key = f"iemocap_cent_gamma{gamma}_seed{seed}"
            cmd = base_cmd + [
                "--mode", "centralized",
                "--focal_gamma", str(gamma),
                "--seed", str(seed),
            ]
            run_experiment(key, cmd)
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    results = load_results()
    
    for mode in ["fed", "cent"]:
        mode_name = "Federated EAFA" if mode == "fed" else "Centralized"
        print(f"\n  {mode_name}:")
        for gamma in GAMMAS:
            wf1s = []
            for seed in SEEDS:
                key = f"iemocap_{mode}_gamma{gamma}_seed{seed}"
                if key in results and results[key].get("wf1"):
                    wf1s.append(results[key]["wf1"])
            if wf1s:
                import numpy as np
                mean = np.mean(wf1s)
                std = np.std(wf1s)
                marker = " << BASELINE" if gamma == 0.0 else ""
                print(f"    gamma={gamma:.1f}: {mean:.4f} ± {std:.4f} (n={len(wf1s)}){marker}")
    
    # Find best
    best_key = None
    best_wf1 = 0
    for key, val in results.items():
        if val.get("wf1") and val["wf1"] > best_wf1:
            best_wf1 = val["wf1"]
            best_key = key
    
    if best_key:
        print(f"\n  BEST: {best_key} -> {best_wf1:.4f}")

if __name__ == "__main__":
    main()
