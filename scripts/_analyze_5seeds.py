"""Analyze 5-seed results with full statistics."""
import json, numpy as np
from scipy import stats as scipy_stats

r = json.load(open('results_ssl_5seeds.json'))

datasets = ['meld', 'iemocap']
label_ratios = [0.05, 0.1, 0.5]
methods = ['supervised', 'fixmatch', 'ecr']
SEEDS = [42, 123, 456, 789, 2024]

print('=' * 80)
print('FULL SUMMARY TABLE (Mean +/- Std)')
print('=' * 80)

for dataset in datasets:
    print(f'\n  {dataset.upper()}:')
    header = f"  {'Label%':>7} | {'Supervised':>18} | {'FixMatch':>18} | {'EDL+ECR':>18}"
    print(header)
    print(f"  {'-'*7}-+-{'-'*18}-+-{'-'*18}-+-{'-'*18}")
    for lr in label_ratios:
        row = {}
        for method in methods:
            vals = []
            for seed in SEEDS:
                key = f'{dataset}_{method}_lr{lr:.2f}_s{seed}'
                wf1 = r.get(key, {}).get('wf1')
                if wf1 is not None:
                    vals.append(wf1)
            if vals:
                mean = np.mean(vals)
                std = np.std(vals)
                row[method] = f'{mean:.4f}+/-{std:.4f}'
            else:
                row[method] = 'N/A'
        print(f"  {lr:6.0%}  | {row['supervised']:>18} | {row['fixmatch']:>18} | {row['ecr']:>18}")

print(f'\n{"=" * 80}')
print('  STATISTICAL TESTS')
print('=' * 80)

for dataset in datasets:
    print(f'\n  {dataset.upper()} -- ECR vs FixMatch:')
    for lr in label_ratios:
        ecr_vals = [r.get(f'{dataset}_ecr_lr{lr:.2f}_s{s}', {}).get('wf1') for s in SEEDS]
        fm_vals = [r.get(f'{dataset}_fixmatch_lr{lr:.2f}_s{s}', {}).get('wf1') for s in SEEDS]
        ecr_vals = [v for v in ecr_vals if v is not None]
        fm_vals = [v for v in fm_vals if v is not None]
        n = min(len(ecr_vals), len(fm_vals))
        if n >= 3:
            ecr_mean = np.mean(ecr_vals[:n])
            fm_mean = np.mean(fm_vals[:n])
            delta = ecr_mean - fm_mean
            t_stat, p = scipy_stats.ttest_rel(ecr_vals[:n], fm_vals[:n])
            try:
                w_stat, w_p = scipy_stats.wilcoxon(ecr_vals[:n], fm_vals[:n])
            except:
                w_p = 1.0
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            print(f'    {lr:5.0%}: ECR={ecr_mean:.4f} FM={fm_mean:.4f} delta={delta:+.4f} | t-test p={p:.4f} ({sig}) | Wilcoxon p={w_p:.4f}')

    print(f'\n  {dataset.upper()} -- ECR vs Supervised:')
    for lr in label_ratios:
        ecr_vals = [r.get(f'{dataset}_ecr_lr{lr:.2f}_s{s}', {}).get('wf1') for s in SEEDS]
        sv_vals = [r.get(f'{dataset}_supervised_lr{lr:.2f}_s{s}', {}).get('wf1') for s in SEEDS]
        ecr_vals = [v for v in ecr_vals if v is not None]
        sv_vals = [v for v in sv_vals if v is not None]
        n = min(len(ecr_vals), len(sv_vals))
        if n >= 3:
            ecr_mean = np.mean(ecr_vals[:n])
            sv_mean = np.mean(sv_vals[:n])
            delta = ecr_mean - sv_mean
            t_stat, p = scipy_stats.ttest_rel(ecr_vals[:n], sv_vals[:n])
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            print(f'    {lr:5.0%}: ECR={ecr_mean:.4f} SV={sv_mean:.4f} delta={delta:+.4f} | t-test p={p:.4f} ({sig})')

print(f'\n{"=" * 80}')
