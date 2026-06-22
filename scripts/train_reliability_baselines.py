"""
Utility script to train all baseline models for reliability analysis:
  - CE models (using FedAvg baseline) for MELD, IEMOCAP, DailyDialog
  - EDL models (using EAFA) for MELD, IEMOCAP, DailyDialog
Across seeds 42, 123, 2024.

Checks if checkpoint already exists before running to avoid redundant training.
"""

import os
import subprocess
import sys
from pathlib import Path

SEEDS = [42, 123, 2024]
DATASETS = ["meld", "iemocap", "dailydialog"]
DEVICE = "cuda"

def run_cmd(cmd_list):
    print(f"Running: {' '.join(cmd_list)}")
    res = subprocess.run(cmd_list, capture_output=False, text=True)
    if res.returncode != 0:
        print(f"Command failed with code {res.returncode}")
        return False
    return True

def main():
    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)

    # 1. CE Baselines (FedAvg)
    for ds in DATASETS:
        for seed in SEEDS:
            ckpt_path = save_dir / f"best_fedavg_ce_{ds}_seed{seed}.pt"
            if ckpt_path.exists():
                print(f"CE Checkpoint {ckpt_path} exists, skipping.")
                continue

            print(f"\nTraining CE baseline for {ds.upper()} | Seed {seed}")
            cmd = [
                sys.executable, "scripts/train_multi_dataset.py",
                "--mode", "federated",
                "--loss_type", "ce",
                "--aggregation", "fedavg",
                "--dataset", ds,
                "--finetuned",
                "--seed", str(seed),
                "--device", DEVICE
            ]
            if ds == "iemocap":
                cmd.extend(["--iemocap_classes", "6"])
            run_cmd(cmd)

    # 2. EDL Models (EAFA)
    for ds in DATASETS:
        for seed in SEEDS:
            ckpt_path = save_dir / f"best_eafa_edl_{ds}_seed{seed}.pt"
            if ckpt_path.exists():
                print(f"EDL Checkpoint {ckpt_path} exists, skipping.")
                continue

            # Check if there is a legacy checkpoint (without seed in name) we can rename/copy
            legacy_path = save_dir / f"best_eafa_edl_{ds}.pt"
            if seed == 42 and legacy_path.exists():
                print(f"Copying legacy {legacy_path} to {ckpt_path}")
                import shutil
                shutil.copy(legacy_path, ckpt_path)
                continue

            # In case dataset is dailydialog, and legacy was named best_eafa_dailydialog.pt
            if ds == "dailydialog" and seed == 42:
                legacy_dd = save_dir / "best_eafa_dailydialog.pt"
                if legacy_dd.exists():
                    print(f"Copying legacy {legacy_dd} to {ckpt_path}")
                    import shutil
                    shutil.copy(legacy_dd, ckpt_path)
                    continue

            print(f"\nTraining EDL model for {ds.upper()} | Seed {seed}")
            cmd = [
                sys.executable, "scripts/train_multi_dataset.py",
                "--mode", "federated",
                "--loss_type", "edl",
                "--aggregation", "eafa",
                "--dataset", ds,
                "--finetuned",
                "--seed", str(seed),
                "--device", DEVICE
            ]
            if ds == "iemocap":
                cmd.extend(["--iemocap_classes", "6"])
            run_cmd(cmd)

if __name__ == "__main__":
    main()
