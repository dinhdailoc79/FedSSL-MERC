"""
ECR 5-Seed Statistical Validation
===================================
Run all SSL experiments (Supervised, FixMatch, ECR) with 5 seeds
to establish statistical significance for AAAI submission.

Seeds: 42, 123, 456, 789, 2024
Datasets: MELD, IEMOCAP
Label ratios: 5%, 10%, 50%
Methods: supervised, fixmatch, ecr

Total: 2 datasets × 3 ratios × 3 methods × 5 seeds = 90 runs
Estimated time: ~5-6h on RTX 4050

Usage:
    python scripts/run_ssl_5seeds.py
"""

import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_FILE = "results_ssl_5seeds.json"
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
    # Import the experiment runner from existing script
    from scripts.run_ssl_experiments import run_ssl_experiment

    results = load_results()
    total_start = time.time()

    datasets = ["meld", "iemocap"]
    label_ratios = [0.05, 0.1, 0.5]
    methods = ["supervised", "fixmatch", "ecr"]

    # Build experiment queue
    experiments = []
    for dataset in datasets:
        for lr in label_ratios:
            for method in methods:
                for seed in SEEDS:
                    experiments.append((dataset, method, lr, seed))

    total = len(experiments)
    done = 0
    skipped = 0

    print(f"{'='*60}")
    print(f"  ECR 5-Seed Statistical Validation")
    print(f"  Total experiments: {total}")
    print(f"  Seeds: {SEEDS}")
    print(f"{'='*60}\n")

    for idx, (dataset, method, lr, seed) in enumerate(experiments):
        key = f"{dataset}_{method}_lr{lr:.2f}_s{seed}"

        # Skip if already done
        if key in results and results[key].get("wf1") is not None:
            skipped += 1
            if skipped <= 5 or skipped % 10 == 0:
                print(f"[{idx+1}/{total}] SKIP {key}: WF1={results[key]['wf1']}")
            continue

        print(f"\n[{idx+1}/{total}] RUNNING {key}...")
        start = time.time()

        try:
            r = run_ssl_experiment(dataset, method, lr, seed=seed)
            elapsed = time.time() - start
            r["time"] = round(elapsed, 1)
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
    import numpy as np

    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  ECR 5-SEED RESULTS -- {total_time/60:.1f} minutes")
    print(f"  Done: {done}, Skipped: {skipped}")
    print(f"{'='*70}")

    for dataset in datasets:
        print(f"\n  {dataset.upper()}:")
        print(f"  {'Label%':>7} | {'Supervised':>18} | {'FixMatch':>18} | {'EDL+ECR':>18}")
        print(f"  {'-'*7}-+-{'-'*18}-+-{'-'*18}-+-{'-'*18}")

        for lr in label_ratios:
            row = {}
            for method in methods:
                vals = []
                for seed in SEEDS:
                    key = f"{dataset}_{method}_lr{lr:.2f}_s{seed}"
                    wf1 = results.get(key, {}).get("wf1")
                    if wf1 is not None:
                        vals.append(wf1)

                if vals:
                    mean = np.mean(vals)
                    std = np.std(vals)
                    row[method] = f"{mean:.4f}+/-{std:.4f}"
                else:
                    row[method] = "N/A"

            print(f"  {lr:6.0%}  | {row['supervised']:>18} | {row['fixmatch']:>18} | {row['ecr']:>18}")

    # Statistical tests
    print(f"\n{'='*70}")
    print(f"  STATISTICAL TESTS (ECR vs FixMatch)")
    print(f"{'='*70}")

    from scipy import stats as scipy_stats

    for dataset in datasets:
        print(f"\n  {dataset.upper()}:")
        for lr in label_ratios:
            ecr_vals = []
            fm_vals = []
            for seed in SEEDS:
                ecr_key = f"{dataset}_ecr_lr{lr:.2f}_s{seed}"
                fm_key = f"{dataset}_fixmatch_lr{lr:.2f}_s{seed}"
                ecr_wf1 = results.get(ecr_key, {}).get("wf1")
                fm_wf1 = results.get(fm_key, {}).get("wf1")
                if ecr_wf1 is not None:
                    ecr_vals.append(ecr_wf1)
                if fm_wf1 is not None:
                    fm_vals.append(fm_wf1)

            if len(ecr_vals) >= 3 and len(fm_vals) >= 3:
                ecr_mean = np.mean(ecr_vals)
                fm_mean = np.mean(fm_vals)
                delta = ecr_mean - fm_mean

                # Paired t-test
                t_stat, p_value = scipy_stats.ttest_rel(ecr_vals[:min(len(ecr_vals), len(fm_vals))],
                                                         fm_vals[:min(len(ecr_vals), len(fm_vals))])
                # Wilcoxon (non-parametric)
                try:
                    w_stat, w_p = scipy_stats.wilcoxon(ecr_vals[:min(len(ecr_vals), len(fm_vals))],
                                                        fm_vals[:min(len(ecr_vals), len(fm_vals))])
                except ValueError:
                    w_p = 1.0

                sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                print(f"    {lr:5.0%}: ECR={ecr_mean:.4f} FM={fm_mean:.4f} delta={delta:+.4f} | t-test p={p_value:.4f} ({sig}) | Wilcoxon p={w_p:.4f}")
            else:
                print(f"    {lr:5.0%}: Insufficient data for test")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
