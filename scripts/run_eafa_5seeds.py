"""
EAFA Noise Robustness 5-Seed Upgrade
======================================
Upgrade noise robustness experiments from 3 seeds to 5 seeds.
Reuse existing results from results_beta_sensitivity.json.

Seeds: 42, 123, 456, 789, 2024
Datasets: MELD, IEMOCAP
Noise: 0%, 20%, 40%
Methods: EAFA (β=10), FedAvg (β=0)

Total new: 2 datasets × 3 noise × 2 methods × 2 new seeds = 24 new runs
(existing 3 seeds × 12 = 36 already done → skip)

Usage:
    python scripts/run_eafa_5seeds.py
"""

import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_FILE = "results_eafa_5seeds.json"
SEEDS = [42, 123, 456, 789, 2024]


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))


def main():
    from scripts.run_beta_sensitivity import run_noise_experiment
    
    results = load_results()
    total_start = time.time()
    
    # Copy existing 3-seed results from beta sensitivity file
    old_file = "results_beta_sensitivity.json"
    if os.path.exists(old_file) and not results:
        old = json.load(open(old_file))
        copied = 0
        for ok, ov in old.items():
            # Map old keys to new key format
            # Old: meld_eafa_b10.0_noise0.0_s42 → new: meld_eafa_noise0.0_s42
            # Old: meld_fedavg_noise0.0_s42_v2 → new: meld_fedavg_noise0.0_s42
            if 'eafa_b10.0' in ok:
                new_key = ok.replace('_b10.0', '')
                results[new_key] = ov
                copied += 1
            elif 'fedavg' in ok and '_v2' in ok:
                new_key = ok.replace('_v2', '')
                results[new_key] = ov
                copied += 1
        save_results(results)
        print(f"Copied {copied} existing results from {old_file}")
    
    datasets = ["meld", "iemocap"]
    noise_levels = [0.0, 0.2, 0.4]
    beta_configs = [(10.0, "eafa"), (0.0, "fedavg")]
    
    experiments = []
    for dataset in datasets:
        for noise in noise_levels:
            for beta, method in beta_configs:
                for seed in SEEDS:
                    key = f"{dataset}_{method}_noise{noise}_s{seed}"
                    experiments.append((key, dataset, beta, noise, seed))
    
    total = len(experiments)
    done = 0
    skipped = 0
    
    print(f"{'='*60}")
    print(f"  EAFA 5-Seed Noise Robustness")
    print(f"  Total experiments: {total}")
    print(f"  Seeds: {SEEDS}")
    print(f"{'='*60}\n")
    
    for idx, (key, dataset, beta, noise, seed) in enumerate(experiments):
        if key in results and results[key].get("wf1") is not None:
            skipped += 1
            if skipped <= 5 or skipped % 10 == 0:
                print(f"[{idx+1}/{total}] SKIP {key}: WF1={results[key]['wf1']}")
            continue
        
        print(f"\n[{idx+1}/{total}] RUNNING {key}...")
        start = time.time()
        
        try:
            r = run_noise_experiment(dataset, beta, noise, seed)
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
            results[key] = {"wf1": None, "error": str(e), "seed": seed}
            save_results(results)
    
    # ========== Summary ==========
    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  EAFA 5-SEED NOISE RESULTS -- {total_time/60:.1f} minutes")
    print(f"  Done: {done}, Skipped: {skipped}")
    print(f"{'='*70}")
    
    from scipy import stats as scipy_stats
    
    for dataset in datasets:
        print(f"\n  {dataset.upper()}:")
        print(f"  {'Noise':>5} | {'EAFA mean±std':>16} | {'FedAvg mean±std':>16} | {'Delta':>7} | {'p-value':>8} | {'Sig':>4}")
        print(f"  {'-'*5}-+-{'-'*16}-+-{'-'*16}-+-{'-'*7}-+-{'-'*8}-+-{'-'*4}")
        
        for noise in noise_levels:
            eafa_vals = []
            fedavg_vals = []
            for seed in SEEDS:
                ek = f"{dataset}_eafa_noise{noise}_s{seed}"
                fk = f"{dataset}_fedavg_noise{noise}_s{seed}"
                e_wf1 = results.get(ek, {}).get("wf1")
                f_wf1 = results.get(fk, {}).get("wf1")
                if e_wf1 is not None:
                    eafa_vals.append(e_wf1)
                if f_wf1 is not None:
                    fedavg_vals.append(f_wf1)
            
            if len(eafa_vals) >= 3 and len(fedavg_vals) >= 3:
                n = min(len(eafa_vals), len(fedavg_vals))
                e_mean = np.mean(eafa_vals[:n])
                f_mean = np.mean(fedavg_vals[:n])
                e_std = np.std(eafa_vals[:n], ddof=1)
                f_std = np.std(fedavg_vals[:n], ddof=1)
                delta = e_mean - f_mean
                
                t_stat, p_val = scipy_stats.ttest_rel(eafa_vals[:n], fedavg_vals[:n])
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                
                print(f"  {int(noise*100):4d}% | {e_mean:.4f}±{e_std:.4f} | {f_mean:.4f}±{f_std:.4f} | {delta:+.4f} | {p_val:.5f} | {sig:>4}")
    
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
