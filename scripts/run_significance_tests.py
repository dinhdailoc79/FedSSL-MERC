"""
Statistical Significance Tests for FedSSL-MERC
================================================
Computes paired Wilcoxon signed-rank tests and bootstrap confidence
intervals for all key claims in the paper.

Usage:
    python scripts/run_significance_tests.py
"""

import json
import os
import sys
import numpy as np
from scipy import stats
from itertools import combinations

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def load_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def paired_test(a, b, name_a, name_b, context=""):
    """Run paired Wilcoxon + paired t-test + effect size."""
    a, b = np.array(a), np.array(b)
    n = len(a)
    diff = a - b
    mean_diff = diff.mean()
    
    results = {
        "comparison": f"{name_a} vs {name_b}",
        "context": context,
        "n_pairs": n,
        "mean_a": f"{a.mean()*100:.2f}",
        "mean_b": f"{b.mean()*100:.2f}",
        "mean_diff": f"{mean_diff*100:+.2f}%",
    }
    
    # Paired t-test (works with any n >= 2)
    if n >= 2 and np.std(diff) > 0:
        t_stat, t_pval = stats.ttest_rel(a, b)
        results["t_pval"] = t_pval
    else:
        results["t_pval"] = None
    
    # Wilcoxon signed-rank (needs n >= 5 for meaningful result)
    if n >= 5 and np.any(diff != 0):
        try:
            w_stat, w_pval = stats.wilcoxon(a, b, alternative="two-sided")
            results["wilcoxon_pval"] = w_pval
        except ValueError:
            results["wilcoxon_pval"] = None
    else:
        results["wilcoxon_pval"] = None
    
    # Sign test (works with any n)
    n_pos = np.sum(diff > 0)
    n_neg = np.sum(diff < 0)
    n_ties = np.sum(diff == 0)
    sign_pval = stats.binomtest(n_pos, n_pos + n_neg, 0.5).pvalue if (n_pos + n_neg) > 0 else 1.0
    results["sign_test_pval"] = sign_pval
    results["wins"] = f"{n_pos}W/{n_neg}L/{n_ties}T"
    
    # Cohen's d effect size
    if np.std(diff) > 0:
        results["cohens_d"] = f"{mean_diff / np.std(diff):.3f}"
    else:
        results["cohens_d"] = "N/A"
    
    # Bootstrap 95% CI for mean difference
    rng = np.random.default_rng(42)
    boot_diffs = []
    for _ in range(10000):
        idx = rng.integers(0, n, size=n)
        boot_diffs.append((a[idx] - b[idx]).mean())
    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
    results["bootstrap_ci"] = f"[{ci_lo*100:+.2f}%, {ci_hi*100:+.2f}%]"
    
    return results


def print_result(r):
    """Pretty-print one test result."""
    print(f"\n  {r['comparison']} ({r['context']})")
    print(f"    {r['comparison'].split(' vs ')[0]}: {r['mean_a']}% | "
          f"{r['comparison'].split(' vs ')[1]}: {r['mean_b']}% | "
          f"Delta = {r['mean_diff']}")
    print(f"    Pairs: {r['n_pairs']} | Wins: {r['wins']} | "
          f"Cohen's d: {r['cohens_d']}")
    
    pvals = []
    if r.get("t_pval") is not None:
        pvals.append(f"t-test p={r['t_pval']:.4f}")
    if r.get("wilcoxon_pval") is not None:
        pvals.append(f"Wilcoxon p={r['wilcoxon_pval']:.4f}")
    pvals.append(f"Sign p={r['sign_test_pval']:.4f}")
    print(f"    {' | '.join(pvals)}")
    print(f"    95% Bootstrap CI: {r['bootstrap_ci']}")
    
    # Significance marker
    best_p = min(
        r.get("t_pval") or 1.0,
        r.get("wilcoxon_pval") or 1.0,
        r.get("sign_test_pval", 1.0),
    )
    if best_p < 0.01:
        print(f"    >>> HIGHLY SIGNIFICANT (p < 0.01) <<<")
    elif best_p < 0.05:
        print(f"    >>> SIGNIFICANT (p < 0.05) <<<")
    elif best_p < 0.10:
        print(f"    ~~ Marginally significant (p < 0.10) ~~")
    else:
        print(f"    -- Not significant (p >= 0.10) --")


def main():
    print("=" * 70)
    print("  STATISTICAL SIGNIFICANCE TESTS — FedSSL-MERC")
    print("=" * 70)
    
    all_results = []
    
    # =========================================================
    # 1. EAFA vs FedAvg (5 seeds, clean data)
    # =========================================================
    print("\n" + "=" * 70)
    print("  1. EAFA vs FedAvg (5 seeds, noise=0.0)")
    print("=" * 70)
    
    eafa5 = load_json("results_eafa_5seeds.json")
    seeds_5 = [42, 123, 2024, 456, 789]
    seeds_3 = [42, 123, 2024]
    
    for dataset in ["meld", "iemocap"]:
        eafa_scores = []
        fedavg_scores = []
        for s in seeds_5:
            eafa_key = f"{dataset}_eafa_noise0.0_s{s}"
            favg_key = f"{dataset}_fedavg_noise0.0_s{s}"
            if eafa_key in eafa5 and favg_key in eafa5:
                eafa_scores.append(eafa5[eafa_key]["wf1"])
                fedavg_scores.append(eafa5[favg_key]["wf1"])
        
        if len(eafa_scores) >= 3:
            r = paired_test(eafa_scores, fedavg_scores,
                           "EAFA", "FedAvg", f"{dataset.upper()} noise=0.0")
            print_result(r)
            all_results.append(r)
    
    # =========================================================
    # 2. EAFA vs FedAvg under noise (pooled across noise levels)
    # =========================================================
    print("\n" + "=" * 70)
    print("  2. EAFA vs FedAvg under label noise (pooled)")
    print("=" * 70)
    
    for dataset in ["meld", "iemocap"]:
        for noise in [0.2, 0.4]:
            eafa_scores = []
            fedavg_scores = []
            for s in seeds_5:
                eafa_key = f"{dataset}_eafa_noise{noise}_s{s}"
                favg_key = f"{dataset}_fedavg_noise{noise}_s{s}"
                if eafa_key in eafa5 and favg_key in eafa5:
                    eafa_scores.append(eafa5[eafa_key]["wf1"])
                    fedavg_scores.append(eafa5[favg_key]["wf1"])
            
            if len(eafa_scores) >= 3:
                r = paired_test(eafa_scores, fedavg_scores,
                               "EAFA", "FedAvg",
                               f"{dataset.upper()} noise={noise}")
                print_result(r)
                all_results.append(r)
    
    # =========================================================
    # =========================================================
    # 3. EAFA vs Modern FL Baselines (5 seeds)
    # =========================================================
    print("\n" + "=" * 70)
    print("  3. EAFA vs Modern FL Baselines (5 seeds)")
    print("=" * 70)
    
    fl_baselines = load_json("results/fl_baselines_results.json")
    
    # Get EAFA scores from eafa5 (noise=0.0)
    for dataset in ["meld", "iemocap"]:
        eafa_scores_5 = []
        for s in seeds_5:
            key = f"{dataset}_eafa_noise0.0_s{s}"
            if key in eafa5:
                eafa_scores_5.append(eafa5[key]["wf1"])
        
        for method in ["scaffold", "fednova", "fedadam", "moon"]:
            baseline_scores = []
            for s in seeds_5:
                key = f"{dataset}_{method}_s{s}"
                if key in fl_baselines:
                    baseline_scores.append(fl_baselines[key]["wf1"])
            
            if len(eafa_scores_5) >= 3 and len(baseline_scores) >= 3:
                n = min(len(eafa_scores_5), len(baseline_scores))
                r = paired_test(eafa_scores_5[:n], baseline_scores[:n],
                               "EAFA", method.upper(),
                               f"{dataset.upper()} (5 seeds)")
                print_result(r)
                all_results.append(r)
    
    # =========================================================
    # 4. ECR vs SSL Baselines (5 seeds for MELD/IEMOCAP, 3 for DailyDialog)
    # =========================================================
    print("\n" + "=" * 70)
    print("  4. ECR vs SSL Baselines (Statistical Tests)")
    print("=" * 70)
    
    ssl_5seeds = load_json("results_ssl_5seeds.json")
    ssl_ratios = load_json("results_ssl_ratios.json")
    
    # MELD & IEMOCAP: 5 seeds
    for dataset in ["meld", "iemocap"]:
        for ratio in [0.05, 0.1, 0.5]:
            ecr_scores = []
            for s in seeds_5:
                key = f"{dataset}_ecr_lr{ratio:.2f}_s{s}"
                if key in ssl_5seeds and ssl_5seeds[key].get("wf1"):
                    ecr_scores.append(ssl_5seeds[key]["wf1"])
            
            for method in ["supervised", "fixmatch"]:
                baseline_scores = []
                for s in seeds_5:
                    key = f"{dataset}_{method}_lr{ratio:.2f}_s{s}"
                    if key in ssl_5seeds and ssl_5seeds[key].get("wf1"):
                        baseline_scores.append(ssl_5seeds[key]["wf1"])
                
                if len(ecr_scores) >= 3 and len(baseline_scores) >= 3:
                    n = min(len(ecr_scores), len(baseline_scores))
                    r = paired_test(
                        ecr_scores[:n], baseline_scores[:n],
                        "ECR", method.capitalize(),
                        f"{dataset.upper()} label={ratio:.0%}"
                    )
                    print_result(r)
                    all_results.append(r)
                    
    # DailyDialog: 3 seeds
    dataset = "dailydialog"
    for ratio in [0.05, 0.1, 0.2]:
        ecr_scores = []
        for s in seeds_3:
            key = f"{dataset}_{ratio}_ecr_s{s}"
            if key in ssl_ratios and ssl_ratios[key].get("wf1"):
                ecr_scores.append(ssl_ratios[key]["wf1"])
        
        for method in ["supervised", "fixmatch"]:
            baseline_scores = []
            for s in seeds_3:
                key = f"{dataset}_{ratio}_{method}_s{s}"
                if key in ssl_ratios and ssl_ratios[key].get("wf1"):
                    baseline_scores.append(ssl_ratios[key]["wf1"])
            
            if len(ecr_scores) >= 3 and len(baseline_scores) >= 3:
                n = min(len(ecr_scores), len(baseline_scores))
                r = paired_test(
                    ecr_scores[:n], baseline_scores[:n],
                    "ECR", method.capitalize(),
                    f"{dataset.upper()} label={ratio:.0%}"
                )
                print_result(r)
                all_results.append(r)
    
    # =========================================================
    # 5. Persistent FlexMatch variants (3 seeds)
    # =========================================================
    print("\n" + "=" * 70)
    print("  5. ECR vs Persistent FlexMatch (3 seeds, label=10%)")
    print("=" * 70)
    
    pfm = load_json("results/persistent_flexmatch_results.json")
    # ECR scores from ssl_ratios at 10%
    for dataset in ["meld", "iemocap"]:
        ecr_scores = []
        for s in seeds_3:
            key = f"{dataset}_0.1_ecr_s{s}"
            if key in ssl_ratios and ssl_ratios[key].get("wf1"):
                ecr_scores.append(ssl_ratios[key]["wf1"])
        
        for fm_method in ["flexmatch_persistent", "flexmatch_serveragg"]:
            fm_scores = []
            for s in seeds_3:
                key = f"{dataset}_{fm_method}_lr0.10_s{s}"
                if key in pfm and pfm[key].get("wf1"):
                    fm_scores.append(pfm[key]["wf1"])
            
            if len(ecr_scores) >= 3 and len(fm_scores) >= 3:
                n = min(len(ecr_scores), len(fm_scores))
                r = paired_test(
                    ecr_scores[:n], fm_scores[:n],
                    "ECR", fm_method.replace("_", " ").title(),
                    f"{dataset.upper()} label=10%"
                )
                print_result(r)
                all_results.append(r)
    
    # =========================================================
    # Summary Table
    # =========================================================
    print("\n" + "=" * 70)
    print("  SUMMARY TABLE — For LaTeX Integration")
    print("=" * 70)
    print(f"\n  {'Comparison':<35} {'Context':<25} {'Delta':<8} {'p-value':<10} {'Sig?':<5}")
    print(f"  {'-'*85}")
    
    for r in all_results:
        best_p = min(
            r.get("t_pval") or 1.0,
            r.get("wilcoxon_pval") or 1.0,
            r.get("sign_test_pval", 1.0),
        )
        sig = "***" if best_p < 0.01 else "**" if best_p < 0.05 else "*" if best_p < 0.10 else ""
        p_str = f"{best_p:.4f}" if best_p < 1.0 else "N/A"
        comp = r["comparison"][:35]
        ctx = r["context"][:25]
        print(f"  {comp:<35} {ctx:<25} {r['mean_diff']:<8} {p_str:<10} {sig:<5}")
    
    # Save results
    os.makedirs("results", exist_ok=True)
    save_results = []
    for r in all_results:
        sr = dict(r)
        for k in ["t_pval", "wilcoxon_pval", "sign_test_pval"]:
            if sr.get(k) is not None:
                sr[k] = float(sr[k])
        save_results.append(sr)
    
    with open("results/significance_tests.json", "w") as f:
        json.dump(save_results, f, indent=2)
    
    print(f"\n  Results saved to results/significance_tests.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
