"""
Comprehensive Statistical Analysis
=====================================
Computes all statistical metrics needed for AAAI paper:
- Paired t-test + Wilcoxon signed-rank
- Cohen's d (effect size)
- 95% confidence intervals
- LaTeX-ready tables

Input: results_ssl_5seeds.json, results_beta_sensitivity.json
Output: Printed tables + results_statistical_analysis.json

Usage:
    python scripts/run_statistical_analysis.py
"""

import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def cohens_d(x, y):
    """Compute Cohen's d effect size for paired samples."""
    diff = np.array(x) - np.array(y)
    return np.mean(diff) / max(np.std(diff, ddof=1), 1e-10)


def confidence_interval_95(vals):
    """95% CI using t-distribution."""
    from scipy import stats
    n = len(vals)
    if n < 2:
        return (np.mean(vals), np.mean(vals))
    mean = np.mean(vals)
    se = np.std(vals, ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n-1)
    return (mean - t_crit * se, mean + t_crit * se)


def effect_size_label(d):
    """Interpret Cohen's d."""
    d = abs(d)
    if d < 0.2: return "negligible"
    if d < 0.5: return "small"
    if d < 0.8: return "medium"
    return "large"


def analyze_ssl_results():
    """Analyze ECR vs FixMatch vs Supervised from 5-seed experiments."""
    from scipy import stats
    
    ssl_file = "results_ssl_5seeds.json"
    if not os.path.exists(ssl_file):
        print(f"  {ssl_file} not found!")
        return {}
    
    r = json.load(open(ssl_file))
    
    datasets = ["meld", "iemocap"]
    label_ratios = [0.05, 0.1, 0.5]
    seeds = [42, 123, 456, 789, 2024]
    
    results = {}
    
    print(f"\n{'='*80}")
    print(f"  SSL STATISTICAL ANALYSIS (ECR vs FixMatch, 5 seeds)")
    print(f"{'='*80}")
    
    for dataset in datasets:
        print(f"\n  === {dataset.upper()} ===")
        print(f"  {'LR':>5} | {'ECR mean±std':>16} | {'FM mean±std':>16} | {'Delta':>7} | {'t-test p':>9} | {'Wilcox p':>9} | {'Cohen d':>8} | {'Effect':>10} | {'95% CI':>18}")
        print(f"  {'-'*5}-+-{'-'*16}-+-{'-'*16}-+-{'-'*7}-+-{'-'*9}-+-{'-'*9}-+-{'-'*8}-+-{'-'*10}-+-{'-'*18}")
        
        for lr in label_ratios:
            ecr_vals = []
            fm_vals = []
            sup_vals = []
            
            for seed in seeds:
                ecr_key = f"{dataset}_ecr_lr{lr:.2f}_s{seed}"
                fm_key = f"{dataset}_fixmatch_lr{lr:.2f}_s{seed}"
                sup_key = f"{dataset}_supervised_lr{lr:.2f}_s{seed}"
                
                if ecr_key in r and r[ecr_key].get("wf1") is not None:
                    ecr_vals.append(r[ecr_key]["wf1"])
                if fm_key in r and r[fm_key].get("wf1") is not None:
                    fm_vals.append(r[fm_key]["wf1"])
                if sup_key in r and r[sup_key].get("wf1") is not None:
                    sup_vals.append(r[sup_key]["wf1"])
            
            if len(ecr_vals) >= 3 and len(fm_vals) >= 3:
                n = min(len(ecr_vals), len(fm_vals))
                ecr_a, fm_a = ecr_vals[:n], fm_vals[:n]
                
                ecr_mean = np.mean(ecr_a)
                fm_mean = np.mean(fm_a)
                ecr_std = np.std(ecr_a, ddof=1)
                fm_std = np.std(fm_a, ddof=1)
                delta = ecr_mean - fm_mean
                
                # Paired t-test
                t_stat, p_ttest = stats.ttest_rel(ecr_a, fm_a)
                
                # Wilcoxon signed-rank
                try:
                    w_stat, p_wilcox = stats.wilcoxon(ecr_a, fm_a)
                except ValueError:
                    p_wilcox = 1.0
                
                # Cohen's d
                d = cohens_d(ecr_a, fm_a)
                d_label = effect_size_label(d)
                
                # 95% CI on the difference
                diffs = np.array(ecr_a) - np.array(fm_a)
                ci_lo, ci_hi = confidence_interval_95(diffs)
                
                sig = "***" if p_ttest < 0.001 else "**" if p_ttest < 0.01 else "*" if p_ttest < 0.05 else "ns"
                
                print(f"  {lr:5.0%} | {ecr_mean:.4f}±{ecr_std:.4f} | {fm_mean:.4f}±{fm_std:.4f} | {delta:+.4f} | {p_ttest:.5f}{sig:>2} | {p_wilcox:.5f} | {d:+.4f} | {d_label:>10} | [{ci_lo:+.4f},{ci_hi:+.4f}]")
                
                key = f"{dataset}_lr{lr:.2f}_ecr_vs_fm"
                results[key] = {
                    "ecr_mean": round(ecr_mean, 4), "ecr_std": round(ecr_std, 4),
                    "fm_mean": round(fm_mean, 4), "fm_std": round(fm_std, 4),
                    "delta": round(delta, 4),
                    "p_ttest": round(p_ttest, 5), "p_wilcoxon": round(p_wilcox, 5),
                    "cohens_d": round(d, 4), "effect_size": d_label,
                    "ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
                    "n_seeds": n, "significant": p_ttest < 0.05,
                }
                
                # Also: ECR vs Supervised
                if len(sup_vals) >= 3:
                    sup_a = sup_vals[:n]
                    t2, p2 = stats.ttest_rel(ecr_a, sup_a)
                    d2 = cohens_d(ecr_a, sup_a)
                    key2 = f"{dataset}_lr{lr:.2f}_ecr_vs_sup"
                    results[key2] = {
                        "ecr_mean": round(ecr_mean, 4),
                        "sup_mean": round(np.mean(sup_a), 4),
                        "delta": round(ecr_mean - np.mean(sup_a), 4),
                        "p_ttest": round(p2, 5),
                        "cohens_d": round(d2, 4),
                        "significant": p2 < 0.05,
                    }
    
    return results


def analyze_noise_results():
    """Analyze EAFA vs FedAvg from noise robustness experiments."""
    from scipy import stats
    
    noise_file = "results_beta_sensitivity.json"
    if not os.path.exists(noise_file):
        print(f"  {noise_file} not found!")
        return {}
    
    r = json.load(open(noise_file))
    
    datasets = ["meld", "iemocap"]
    noise_levels = [0.0, 0.2, 0.4]
    seeds = [42, 123, 2024]
    
    results = {}
    
    print(f"\n{'='*80}")
    print(f"  NOISE ROBUSTNESS STATISTICAL ANALYSIS (EAFA vs FedAvg, 3 seeds)")
    print(f"{'='*80}")
    
    for dataset in datasets:
        print(f"\n  === {dataset.upper()} ===")
        print(f"  {'Noise':>5} | {'EAFA mean±std':>16} | {'FedAvg mean±std':>16} | {'Delta':>7} | {'t-test p':>9} | {'Cohen d':>8} | {'Effect':>10}")
        print(f"  {'-'*5}-+-{'-'*16}-+-{'-'*16}-+-{'-'*7}-+-{'-'*9}-+-{'-'*8}-+-{'-'*10}")
        
        for noise in noise_levels:
            eafa_vals = []
            fedavg_vals = []
            
            for seed in seeds:
                ek = f"{dataset}_eafa_b10.0_noise{noise}_s{seed}"
                fk = f"{dataset}_fedavg_noise{noise}_s{seed}_v2"
                
                if ek in r and r[ek].get("wf1") is not None:
                    eafa_vals.append(r[ek]["wf1"])
                if fk in r and r[fk].get("wf1") is not None:
                    fedavg_vals.append(r[fk]["wf1"])
            
            if len(eafa_vals) >= 3 and len(fedavg_vals) >= 3:
                n = min(len(eafa_vals), len(fedavg_vals))
                ea, fa = eafa_vals[:n], fedavg_vals[:n]
                
                ea_mean, fa_mean = np.mean(ea), np.mean(fa)
                ea_std, fa_std = np.std(ea, ddof=1), np.std(fa, ddof=1)
                delta = ea_mean - fa_mean
                
                t_stat, p_val = stats.ttest_rel(ea, fa)
                d = cohens_d(ea, fa)
                d_label = effect_size_label(d)
                
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                
                print(f"  {int(noise*100):4d}% | {ea_mean:.4f}±{ea_std:.4f} | {fa_mean:.4f}±{fa_std:.4f} | {delta:+.4f} | {p_val:.5f}{sig:>2} | {d:+.4f} | {d_label:>10}")
                
                key = f"{dataset}_noise{int(noise*100)}_eafa_vs_fedavg"
                results[key] = {
                    "eafa_mean": round(ea_mean, 4), "fedavg_mean": round(fa_mean, 4),
                    "delta": round(delta, 4), "p_ttest": round(p_val, 5),
                    "cohens_d": round(d, 4), "effect_size": d_label,
                    "significant": p_val < 0.05,
                }
    
    return results


def generate_latex_tables(ssl_results, noise_results):
    """Generate LaTeX-ready tables."""
    
    print(f"\n{'='*80}")
    print(f"  LaTeX TABLE: ECR vs FixMatch (Table 1)")
    print(f"{'='*80}\n")
    
    print(r"\begin{table}[t]")
    print(r"\caption{ECR vs FixMatch comparison across label ratios (WF1, 5 seeds).}")
    print(r"\label{tab:ecr_results}")
    print(r"\centering")
    print(r"\begin{tabular}{llccc}")
    print(r"\toprule")
    print(r"Dataset & Label\% & Supervised & FixMatch & \textbf{ECR (Ours)} \\")
    print(r"\midrule")
    
    ssl_file = "results_ssl_5seeds.json"
    if os.path.exists(ssl_file):
        r = json.load(open(ssl_file))
        seeds = [42, 123, 456, 789, 2024]
        
        for dataset in ["meld", "iemocap"]:
            name = "MELD" if dataset == "meld" else "IEMOCAP"
            for i, lr in enumerate([0.05, 0.1, 0.5]):
                row_label = name if i == 0 else ""
                
                for method in ["supervised", "fixmatch", "ecr"]:
                    vals = []
                    for seed in seeds:
                        key = f"{dataset}_{method}_lr{lr:.2f}_s{seed}"
                        wf1 = r.get(key, {}).get("wf1")
                        if wf1 is not None:
                            vals.append(wf1)
                    
                    if vals:
                        mean = np.mean(vals)
                        std = np.std(vals, ddof=1)
                        if method == "supervised":
                            sup_str = f"{mean:.2%}$\\pm${std:.2%}"
                        elif method == "fixmatch":
                            fm_str = f"{mean:.2%}$\\pm${std:.2%}"
                        else:
                            # Check significance
                            stat_key = f"{dataset}_lr{lr:.2f}_ecr_vs_fm"
                            sig = ssl_results.get(stat_key, {}).get("significant", False)
                            bold = r"\textbf{" if sig else ""
                            bold_end = "}" if sig else ""
                            ecr_str = f"{bold}{mean:.2%}$\\pm${std:.2%}{bold_end}"
                
                # Get p-value
                stat_key = f"{dataset}_lr{lr:.2f}_ecr_vs_fm"
                p = ssl_results.get(stat_key, {}).get("p_ttest", 1.0)
                sig_marker = "$^{***}$" if p < 0.001 else "$^{**}$" if p < 0.01 else "$^{*}$" if p < 0.05 else ""
                
                print(f"{row_label} & {int(lr*100)}\\% & {sup_str} & {fm_str} & {ecr_str}{sig_marker} \\\\")
            
            if dataset == "meld":
                print(r"\midrule")
    
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def main():
    print("=" * 80)
    print("  COMPREHENSIVE STATISTICAL ANALYSIS FOR AAAI PAPER")
    print("=" * 80)
    
    # 1. SSL analysis
    ssl_results = analyze_ssl_results()
    
    # 2. Noise robustness analysis
    noise_results = analyze_noise_results()
    
    # 3. LaTeX tables
    generate_latex_tables(ssl_results, noise_results)
    
    # 4. Save all results
    all_results = {
        "ssl": ssl_results,
        "noise": noise_results,
    }
    
    with open("results_statistical_analysis.json", "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: bool(x) if isinstance(x, np.bool_) else float(x) if hasattr(x, 'item') else str(x))
    
    print(f"\n\nSaved to results_statistical_analysis.json")
    
    # 5. Key findings summary
    print(f"\n{'='*80}")
    print(f"  KEY FINDINGS SUMMARY")
    print(f"{'='*80}")
    
    sig_wins = sum(1 for v in ssl_results.values() if v.get("significant") and v.get("delta", 0) > 0 and "fm" in str(v))
    total_comp = sum(1 for k in ssl_results if "ecr_vs_fm" in k)
    print(f"\n  ECR vs FixMatch: {sig_wins}/{total_comp} significant wins (p<0.05)")
    
    if noise_results:
        noise_wins = sum(1 for v in noise_results.values() if v.get("significant") and v.get("delta", 0) > 0)
        noise_total = len(noise_results)
        print(f"  EAFA vs FedAvg (noise): {noise_wins}/{noise_total} significant wins")
    
    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
