"""
Paper Figures Generator
========================
Generate 6 publication-quality figures for AAAI paper.

Fig 1: Lambda_u sensitivity heatmap (IEMOCAP 50%)
Fig 2: ECR vs FixMatch bar chart (all conditions)
Fig 3: EAFA vs FedAvg noise robustness
Fig 4: ECR Ablation component analysis
Fig 5: Uncertainty evolution during training
Fig 6: EAFA weight distribution under noise

Usage:
    python scripts/generate_figures.py
"""

import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Style setup
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

COLORS = {
    'ecr': '#2196F3',       # Blue
    'fixmatch': '#FF5722',  # Orange-Red
    'supervised': '#4CAF50', # Green
    'eafa': '#2196F3',      # Blue
    'fedavg': '#FF5722',    # Orange-Red
    'full': '#2196F3',
    'no_cert': '#FF9800',
    'ce_pseudo': '#F44336',
    'no_aug': '#9C27B0',
}

OUT_DIR = "paper/figures"
os.makedirs(OUT_DIR, exist_ok=True)


def load_json(path):
    if os.path.exists(path):
        return json.load(open(path))
    return {}


# ============================================================
# Fig 1: Lambda_u sensitivity heatmap
# ============================================================
def fig1_lambda_sensitivity():
    """Heatmap of lambda_u x rampup grid search results."""
    r = load_json("results_fix_iemocap50.json")
    if not r:
        print("  SKIP Fig1: no fix_iemocap50 results")
        return
    
    seeds = [42, 123, 2024]
    lus = [0.3, 0.5, 1.0, 2.0, 3.0, 5.0]
    rps = [5, 10, 20]
    
    data = np.zeros((len(lus), len(rps)))
    for i, lu in enumerate(lus):
        for j, rp in enumerate(rps):
            vals = []
            for s in seeds:
                key = f"iemocap_ecr_lu{lu}_rp{rp}_s{s}"
                wf1 = r.get(key, {}).get("wf1")
                if wf1:
                    vals.append(wf1)
            data[i, j] = np.mean(vals) if vals else 0
    
    fig, ax = plt.subplots(figsize=(5, 4.5))
    
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0.57, vmax=0.60)
    
    ax.set_xticks(range(len(rps)))
    ax.set_xticklabels([f'{rp}' for rp in rps])
    ax.set_yticks(range(len(lus)))
    ax.set_yticklabels([f'{lu}' for lu in lus])
    ax.set_xlabel('Warm-up Rounds')
    ax.set_ylabel(r'$\lambda_u$')
    ax.set_title(r'ECR $\lambda_u$ Sensitivity (IEMOCAP 50%)')
    
    # Add text annotations
    for i in range(len(lus)):
        for j in range(len(rps)):
            val = data[i, j]
            color = 'white' if val < 0.583 else 'black'
            ax.text(j, i, f'{val:.4f}', ha='center', va='center', 
                   fontsize=9, fontweight='bold', color=color)
    
    # Add FixMatch baseline line
    ax.axhline(y=-0.5, color='red', linestyle='--', linewidth=1)
    fig.colorbar(im, ax=ax, label='Weighted F1', shrink=0.8)
    
    # Add baseline annotation
    ax.annotate(f'FixMatch: 0.5965', xy=(2.3, -0.3), fontsize=9, color='red', fontweight='bold')
    
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig1_lambda_sensitivity.pdf")
    fig.savefig(path)
    fig.savefig(path.replace('.pdf', '.png'))
    plt.close(fig)
    print(f"  Fig1 saved: {path}")


# ============================================================
# Fig 2: ECR vs FixMatch vs Supervised (grouped bar)
# ============================================================
def fig2_ecr_comparison():
    """Grouped bar chart: ECR vs FixMatch vs Supervised across all conditions."""
    r = load_json("results_ssl_5seeds.json")
    if not r:
        print("  SKIP Fig2: no ssl_5seeds results")
        return
    
    seeds = [42, 123, 456, 789, 2024]
    datasets = ['meld', 'iemocap']
    label_ratios = [0.05, 0.10, 0.50]
    methods = [
        ('ecr', 'ECR (Ours)'),
        ('fixmatch', 'FixMatch'),
        ('supervised', 'Supervised'),
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)
    
    for ax_idx, dataset in enumerate(datasets):
        ax = axes[ax_idx]
        x = np.arange(len(label_ratios))
        width = 0.25
        
        for m_idx, (method, label) in enumerate(methods):
            means, stds = [], []
            for lr in label_ratios:
                vals = []
                for seed in seeds:
                    key = f"{dataset}_{method}_lr{lr:.2f}_s{seed}"
                    wf1 = r.get(key, {}).get("wf1")
                    if wf1:
                        vals.append(wf1)
                if vals:
                    means.append(np.mean(vals))
                    stds.append(np.std(vals, ddof=1))
                else:
                    means.append(0)
                    stds.append(0)
            
            bars = ax.bar(x + m_idx * width, means, width, 
                         yerr=stds, label=label,
                         color=COLORS.get(method, '#999'),
                         alpha=0.85, capsize=3, edgecolor='white', linewidth=0.5)
            
            # Value labels on top
            for bar, mean in zip(bars, means):
                if mean > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                           f'{mean:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
        
        ax.set_xlabel('Label Ratio')
        ax.set_ylabel('Weighted F1')
        ax.set_title(f'{dataset.upper()}')
        ax.set_xticks(x + width)
        ax.set_xticklabels(['5%', '10%', '50%'])
        ax.legend(loc='lower right', framealpha=0.9)
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)
    
    plt.suptitle('ECR vs Baselines: SSL Performance Comparison', fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig2_ecr_comparison.pdf")
    fig.savefig(path)
    fig.savefig(path.replace('.pdf', '.png'))
    plt.close(fig)
    print(f"  Fig2 saved: {path}")


# ============================================================
# Fig 3: EAFA vs FedAvg noise robustness
# ============================================================
def fig3_noise_robustness():
    """Line plot: EAFA vs FedAvg across noise levels."""
    r = load_json("results_eafa_5seeds.json")
    if not r:
        print("  SKIP Fig3: no eafa_5seeds results")
        return
    
    seeds = [42, 123, 456, 789, 2024]
    datasets = ['meld', 'iemocap']
    noise_levels = [0.0, 0.2, 0.4]
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    
    for ax_idx, dataset in enumerate(datasets):
        ax = axes[ax_idx]
        
        for method, color, marker, label in [
            ('eafa', COLORS['eafa'], 'o', 'EAFA (Ours)'),
            ('fedavg', COLORS['fedavg'], 's', 'FedAvg'),
        ]:
            means, stds = [], []
            for noise in noise_levels:
                vals = []
                for seed in seeds:
                    key = f"{dataset}_{method}_noise{noise}_s{seed}"
                    wf1 = r.get(key, {}).get("wf1")
                    if wf1:
                        vals.append(wf1)
                if vals:
                    means.append(np.mean(vals))
                    stds.append(np.std(vals, ddof=1))
                else:
                    means.append(0)
                    stds.append(0)
            
            noise_pcts = [int(n*100) for n in noise_levels]
            ax.errorbar(noise_pcts, means, yerr=stds, 
                       marker=marker, color=color, label=label,
                       linewidth=2, markersize=8, capsize=4, capthick=1.5)
        
        ax.set_xlabel('Label Noise Rate (%)')
        ax.set_ylabel('Weighted F1')
        ax.set_title(f'{dataset.upper()}')
        ax.set_xticks([0, 20, 40])
        ax.legend(loc='lower left', framealpha=0.9)
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)
    
    plt.suptitle('EAFA Noise Robustness (5 seeds)', fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig3_noise_robustness.pdf")
    fig.savefig(path)
    fig.savefig(path.replace('.pdf', '.png'))
    plt.close(fig)
    print(f"  Fig3 saved: {path}")


# ============================================================
# Fig 4: ECR Ablation bar chart
# ============================================================
def fig4_ablation():
    """Horizontal bar chart: ECR ablation components."""
    r = load_json("results_ecr_ablation.json")
    if not r:
        print("  SKIP Fig4: no ecr_ablation results")
        return
    
    seeds = [42, 123, 2024]
    datasets = ['meld', 'iemocap']
    variants = [
        ('ecr_full', 'ECR Full', COLORS['full']),
        ('ecr_no_certainty', 'w/o Certainty', COLORS['no_cert']),
        ('ecr_ce_pseudo', 'CE Pseudo-Label', COLORS['ce_pseudo']),
        ('ecr_no_augment', 'w/o Augmentation', COLORS['no_aug']),
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    for ax_idx, dataset in enumerate(datasets):
        ax = axes[ax_idx]
        
        names, means, stds, colors = [], [], [], []
        for variant, label, color in variants:
            vals = []
            for seed in seeds:
                key = f"{dataset}_{variant}_s{seed}"
                wf1 = r.get(key, {}).get("wf1")
                if wf1:
                    vals.append(wf1)
            if vals:
                names.append(label)
                means.append(np.mean(vals))
                stds.append(np.std(vals, ddof=1))
                colors.append(color)
        
        y_pos = np.arange(len(names))
        bars = ax.barh(y_pos, means, xerr=stds, color=colors, 
                      alpha=0.85, capsize=3, edgecolor='white', linewidth=0.5,
                      height=0.6)
        
        # Value labels
        for bar, mean in zip(bars, means):
            ax.text(mean + 0.002, bar.get_y() + bar.get_height()/2.,
                   f'{mean:.4f}', ha='left', va='center', fontsize=9, fontweight='bold')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.set_xlabel('Weighted F1')
        ax.set_title(f'{dataset.upper()} (5% label)')
        ax.grid(axis='x', alpha=0.3)
        ax.set_axisbelow(True)
        
        # Highlight full ECR
        ax.axvline(x=means[0], color=COLORS['full'], linestyle='--', alpha=0.5, linewidth=1)
    
    plt.suptitle('ECR Ablation Study', fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig4_ablation.pdf")
    fig.savefig(path)
    fig.savefig(path.replace('.pdf', '.png'))
    plt.close(fig)
    print(f"  Fig4 saved: {path}")


# ============================================================
# Fig 5: Statistical significance heatmap
# ============================================================
def fig5_significance():
    """Heatmap showing p-values and effect sizes."""
    r = load_json("results_ssl_5seeds.json")
    if not r:
        print("  SKIP Fig5: no ssl_5seeds results")
        return
    
    from scipy import stats
    
    seeds = [42, 123, 456, 789, 2024]
    datasets = ['meld', 'iemocap']
    label_ratios = [0.05, 0.10, 0.50]
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    
    for ax_idx, dataset in enumerate(datasets):
        ax = axes[ax_idx]
        
        # Compute deltas and p-values for ECR vs each baseline
        comparisons = ['FixMatch', 'Supervised']
        data_matrix = np.zeros((len(comparisons), len(label_ratios)))
        pval_matrix = np.zeros((len(comparisons), len(label_ratios)))
        
        for c_idx, baseline in enumerate(['fixmatch', 'supervised']):
            for lr_idx, lr in enumerate(label_ratios):
                ecr_vals, base_vals = [], []
                for seed in seeds:
                    ek = f"{dataset}_ecr_lr{lr:.2f}_s{seed}"
                    bk = f"{dataset}_{baseline}_lr{lr:.2f}_s{seed}"
                    e = r.get(ek, {}).get("wf1")
                    b = r.get(bk, {}).get("wf1")
                    if e and b:
                        ecr_vals.append(e)
                        base_vals.append(b)
                
                if len(ecr_vals) >= 3:
                    n = min(len(ecr_vals), len(base_vals))
                    delta = np.mean(ecr_vals[:n]) - np.mean(base_vals[:n])
                    _, p = stats.ttest_rel(ecr_vals[:n], base_vals[:n])
                    data_matrix[c_idx, lr_idx] = delta * 100  # Convert to percentage
                    pval_matrix[c_idx, lr_idx] = p
        
        im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', 
                       vmin=-1.5, vmax=8)
        
        ax.set_xticks(range(len(label_ratios)))
        ax.set_xticklabels(['5%', '10%', '50%'])
        ax.set_yticks(range(len(comparisons)))
        ax.set_yticklabels(comparisons)
        ax.set_xlabel('Label Ratio')
        ax.set_title(f'{dataset.upper()}')
        
        # Annotations with significance stars
        for i in range(len(comparisons)):
            for j in range(len(label_ratios)):
                delta = data_matrix[i, j]
                p = pval_matrix[i, j]
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                color = 'white' if abs(delta) > 3 else 'black'
                ax.text(j, i, f'{delta:+.2f}%\n{sig}', ha='center', va='center',
                       fontsize=8, fontweight='bold', color=color)
        
    fig.colorbar(im, ax=axes, label='WF1 Delta (%)', shrink=0.8)
    plt.suptitle('ECR Improvement vs Baselines (p-value significance)', fontweight='bold', y=1.05)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig5_significance.pdf")
    fig.savefig(path)
    fig.savefig(path.replace('.pdf', '.png'))
    plt.close(fig)
    print(f"  Fig5 saved: {path}")


# ============================================================
# Fig 6: Beta sensitivity (EAFA parameter)
# ============================================================
def fig6_beta_sensitivity():
    """Line plot: beta sensitivity for EAFA."""
    r = load_json("results_beta_sensitivity.json")
    if not r:
        print("  SKIP Fig6: no beta_sensitivity results")
        return
    
    seeds = [42, 123, 2024]
    datasets = ['meld', 'iemocap']
    betas = [0.0, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    
    for ax_idx, dataset in enumerate(datasets):
        ax = axes[ax_idx]
        
        means, stds = [], []
        valid_betas = []
        
        for beta in betas:
            vals = []
            for seed in seeds:
                key = f"{dataset}_eafa_b{beta}_noise0.0_s{seed}"
                if key not in r:
                    key = f"{dataset}_eafa_b{beta:.1f}_noise0.0_s{seed}"
                wf1 = r.get(key, {}).get("wf1")
                if wf1:
                    vals.append(wf1)
            if vals:
                valid_betas.append(beta)
                means.append(np.mean(vals))
                stds.append(np.std(vals, ddof=1))
        
        if not valid_betas:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        ax.errorbar(range(len(valid_betas)), means, yerr=stds,
                   marker='o', color=COLORS['eafa'], linewidth=2, 
                   markersize=8, capsize=4, capthick=1.5)
        
        # Highlight best
        best_idx = np.argmax(means)
        ax.scatter([best_idx], [means[best_idx]], s=200, 
                  facecolors='none', edgecolors='red', linewidth=2, zorder=5)
        ax.annotate(f'Best: {valid_betas[best_idx]}',
                   xy=(best_idx, means[best_idx]),
                   xytext=(best_idx+0.5, means[best_idx]+0.003),
                   fontsize=9, color='red', fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='red'))
        
        # FedAvg baseline (beta=0)
        if 0.0 in valid_betas:
            fedavg_idx = valid_betas.index(0.0)
            ax.axhline(y=means[fedavg_idx], color='gray', linestyle='--', 
                      alpha=0.5, label=f'FedAvg ({means[fedavg_idx]:.4f})')
        
        ax.set_xticks(range(len(valid_betas)))
        ax.set_xticklabels([str(b) for b in valid_betas])
        ax.set_xlabel(r'$\beta$ (EAFA sensitivity)')
        ax.set_ylabel('Weighted F1')
        ax.set_title(f'{dataset.upper()}')
        ax.legend(loc='lower right', framealpha=0.9)
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)
    
    plt.suptitle(r'EAFA $\beta$ Sensitivity Analysis', fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig6_beta_sensitivity.pdf")
    fig.savefig(path)
    fig.savefig(path.replace('.pdf', '.png'))
    plt.close(fig)
    print(f"  Fig6 saved: {path}")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("  AAAI Paper Figure Generator")
    print("=" * 60)
    
    fig1_lambda_sensitivity()
    fig2_ecr_comparison()
    fig3_noise_robustness()
    fig4_ablation()
    fig5_significance()
    fig6_beta_sensitivity()
    
    print(f"\nAll figures saved to {OUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
