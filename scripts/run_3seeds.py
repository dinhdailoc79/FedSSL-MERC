"""
Run 3 seeds for all datasets with finetuned features.
Collects mean ± std for final results table.
"""
import subprocess
import sys
import re
import json
from collections import defaultdict

SEEDS = [42, 123, 2024]
DATASETS = ["meld", "iemocap", "dailydialog"]

results = defaultdict(lambda: defaultdict(list))

for seed in SEEDS:
    for ds in DATASETS:
        print(f"\n{'='*60}")
        print(f"  Dataset: {ds.upper()} | Seed: {seed}")
        print(f"{'='*60}")

        cmd = [
            sys.executable, "scripts/train_multi_dataset.py",
            "--dataset", ds,
            "--mode", "both",
            "--finetuned",
            "--epochs", "80",
            "--patience", "20",
            "--seed", str(seed),
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        output = proc.stdout + proc.stderr

        # Parse EDL result
        edl_match = re.search(rf"{ds}_edl\s+WF1 = (\d+\.\d+)", output)
        eafa_match = re.search(rf"{ds}_eafa\s+WF1 = (\d+\.\d+)", output)

        if edl_match:
            wf1 = float(edl_match.group(1))
            results[ds]["edl"].append(wf1)
            print(f"  EDL: {wf1:.4f}")
        else:
            print(f"  EDL: FAILED")
            print(output[-500:])

        if eafa_match:
            wf1 = float(eafa_match.group(1))
            results[ds]["eafa"].append(wf1)
            print(f"  EAFA: {wf1:.4f}")
        else:
            print(f"  EAFA: FAILED")

# Summary
print(f"\n{'='*60}")
print(f"  FINAL RESULTS — 3 Seeds (Mean ± Std)")
print(f"{'='*60}")
print(f"{'Dataset':<15} {'Method':<10} {'Seeds':>30} {'Mean±Std':>15}")
print(f"{'-'*70}")

import numpy as np
summary = {}
for ds in DATASETS:
    for method in ["edl", "eafa"]:
        vals = results[ds][method]
        if vals:
            mean = np.mean(vals)
            std = np.std(vals)
            seeds_str = ", ".join(f"{v:.4f}" for v in vals)
            print(f"{ds:<15} {method.upper():<10} {seeds_str:>30} {mean:.4f}±{std:.4f}")
            summary[f"{ds}_{method}"] = {"mean": mean, "std": std, "values": vals}

# Save
with open("results_3seeds.json", "w") as f:
    json.dump({k: {"mean": v["mean"], "std": v["std"], "values": v["values"]} for k, v in summary.items()}, f, indent=2)
print(f"\nSaved: results_3seeds.json")
