"""
Ablation Study Runner for LucBinh
==================================
Runs all ablation configurations × 3 seeds for DailyDialog.

Configs:
  1. CE  Centralized  (baseline)
  2. CE  FedAvg       (FL baseline)
  3. EDL FedAvg       (isolate EAFA contribution)
  
Already done:
  4. EDL Centralized  ✅
  5. EDL EAFA         ✅
"""

import subprocess
import sys
import time

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

SEEDS = [42, 123, 2024]
DATASET = "dailydialog"
COMMON = ["--dataset", DATASET, "--finetuned", "--epochs", "80", "--patience", "20"]

CONFIGS = [
    {"name": "CE_Centralized",  "args": ["--loss_type", "ce", "--mode", "centralized"]},
    {"name": "CE_FedAvg",       "args": ["--loss_type", "ce", "--aggregation", "fedavg", "--mode", "federated"]},
    {"name": "EDL_FedAvg",      "args": ["--loss_type", "edl", "--aggregation", "fedavg", "--mode", "federated"]},
]

if __name__ == "__main__":
    total = len(CONFIGS) * len(SEEDS)
    done = 0
    
    for cfg in CONFIGS:
        for seed in SEEDS:
            done += 1
            print(f"\n{'#'*60}")
            print(f"  [{done}/{total}] {cfg['name']} | seed={seed}")
            print(f"{'#'*60}\n")
            
            cmd = [
                sys.executable, "scripts/train_multi_dataset.py",
                *COMMON, *cfg["args"],
                "--seed", str(seed),
            ]
            
            start = time.time()
            result = subprocess.run(cmd, cwd=".")
            elapsed = time.time() - start
            
            status = "✅" if result.returncode == 0 else "❌"
            print(f"\n  {status} {cfg['name']} seed={seed} done in {elapsed:.0f}s")
    
    print(f"\n{'='*60}")
    print(f"  All {total} ablation runs complete!")
    print(f"{'='*60}")
