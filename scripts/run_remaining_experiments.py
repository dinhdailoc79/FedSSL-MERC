"""
Master Experiments Runner
=========================
Coordinates and runs all remaining experiments sequentially:
1. FL baselines on IEMOCAP for seeds 456 and 789.
2. Multimodal baselines (logit averaging, learnable gating) on MELD.
3. Dirichlet FixMatch baseline on MELD and IEMOCAP for all 5 seeds.
4. Byzantine robustness experiments.
5. Confusion matrices and per-class reports generation.
6. Statistical significance tests calculation.

Usage:
    python scripts/run_remaining_experiments.py
"""

import os
import sys
import subprocess
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def run_command(cmd, desc):
    print("\n" + "="*80)
    print(f"  [START] {desc}")
    print(f"  Command: {cmd}")
    print("="*80 + "\n")
    
    start_time = time.time()
    # Run with print output in real-time
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
            
    rc = process.poll()
    elapsed = time.time() - start_time
    
    print("\n" + "="*80)
    if rc == 0:
        print(f"  [SUCCESS] {desc} (elapsed: {elapsed/60:.1f} minutes)")
    else:
        print(f"  [FAILED] {desc} with exit code {rc} (elapsed: {elapsed/60:.1f} minutes)")
    print("="*80 + "\n")
    return rc == 0

def main():
    print("="*80)
    print("  FedSSL-MERC Master Experiments Pipeline")
    print("  Running all remaining evaluations sequentially to avoid GPU OOM.")
    print("="*80)
    
    # 1. FL Baselines on IEMOCAP (Seeds 456, 789)
    run_command(
        "python scripts/run_fl_baselines.py --methods scaffold,fednova,fedadam,moon --datasets iemocap --seeds 456,789",
        "Modern FL Baselines on IEMOCAP (Seeds 456, 789)"
    )
    
    # 2. Multimodal Baselines on MELD (Logit Avg, Learnable Gating)
    run_command(
        "python scripts/run_multimodal_experiments.py",
        "Multimodal Fusion Baselines (Logit Averaging & Learnable Gating)"
    )
    
    # 3. Dirichlet FixMatch on MELD & IEMOCAP (Seeds 42, 123, 456, 789, 2024)
    run_command(
        "python scripts/run_dirichlet_fixmatch.py --dataset meld --seeds 42,123,456,789,2024",
        "Dirichlet FixMatch Baseline on MELD"
    )
    run_command(
        "python scripts/run_dirichlet_fixmatch.py --dataset iemocap --seeds 42,123,456,789,2024",
        "Dirichlet FixMatch Baseline on IEMOCAP"
    )
    
    # 4. Byzantine Robustness Simulation
    run_command(
        "python scripts/run_byzantine_robustness.py",
        "Byzantine Robustness (Uncertainty Spoofing & Model Poisoning)"
    )
    
    # 5. Confusion Matrices & Per-Class reports
    run_command(
        "python scripts/generate_confusion_matrices.py --dataset meld --rounds 25",
        "Confusion Matrices and Per-Class F1 Tables (MELD)"
    )
    run_command(
        "python scripts/generate_confusion_matrices.py --dataset iemocap --rounds 25",
        "Confusion Matrices and Per-Class F1 Tables (IEMOCAP)"
    )
    
    # 6. Statistical Significance Tests Regeneration
    run_command(
        "python scripts/run_significance_tests.py",
        "Recalculate Statistical Significance (Wilcoxon, paired t-tests)"
    )
    
    # 7. ECR Augmentation Test (Dropout vs Gaussian Noise search)
    run_command(
        "python scripts/run_ecr_augtest.py",
        "ECR Augmentation Test (Dropout vs Gaussian Noise Search)"
    )
    
    print("\n" + "="*80)
    print("  ALL REMAINING EXPERIMENTS COMPLETED!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
