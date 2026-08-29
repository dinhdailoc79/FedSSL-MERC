"""
IEMOCAP 4-class experiment with RoBERTa-Large features.
==========================================================
Prerequisite: Fine-tune RoBERTa-Large on Kaggle T4 first:

    !python scripts/finetune_roberta.py \\
        --dataset iemocap \\
        --model_size large \\
        --epochs 5 \\
        --batch_size 8 \\
        --grad_accum 2 \\
        --data_dir data/raw/IEMOCAP/IEMOCAP_full_release \\
        --output_dir outputs

Then download outputs/iemocap_text_roberta_large_finetuned.pt
and place in data/features/ before running this script.

Runs:
    5 seeds x 2 modes (federated EAFA + centralized EDL) = 10 experiments
    Merges happy+excited -> happy (4-class), drops frustrated
    Uses feat_dim=1024 (RoBERTa-Large)

Results saved to: results_iemocap_4class_large.json
"""
import subprocess
import json
import os
import time
import sys
import re
import numpy as np

SEEDS = [42, 123, 2024, 777, 999]
RESULTS_FILE = "results_iemocap_4class_large.json"
FEAT_FILE = "data/features/iemocap_text_roberta_large_finetuned.pt"


def load_results():
    if os.path.exists(RESULTS_FILE):
        return json.load(open(RESULTS_FILE))
    return {}


def save_results(results):
    json.dump(results, open(RESULTS_FILE, "w"), indent=2)


def run_one(key, cmd):
    """Run one experiment, skip if already done."""
    results = load_results()
    if key in results and results[key].get("wf1") is not None:
        print(f"  SKIP {key} (WF1={results[key]['wf1']:.4f})")
        return results[key]

    print(f"  RUN {key}...")
    start = time.time()
    r = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    elapsed = time.time() - start

    wf1 = None
    mf1 = None
    for line in r.stdout.split("\n"):
        m = re.search(r"Test WF1\s*=\s*([\d.]+)", line)
        if m:
            wf1 = float(m.group(1))
        m2 = re.search(r"WF1[:=]\s*([\d.]+)", line)
        if m2 and wf1 is None:
            wf1 = float(m2.group(1))

    result = {"wf1": wf1, "time": round(elapsed, 1), "stdout_tail": r.stdout[-500:]}
    results = load_results()
    results[key] = result
    save_results(results)

    status = f"WF1={wf1:.4f}" if wf1 is not None else "FAILED (check stdout_tail)"
    print(f"    {status} ({elapsed:.0f}s)")
    return result


def check_prerequisites():
    """Check that RoBERTa-Large feature file exists."""
    if not os.path.exists(FEAT_FILE):
        print(f"\n[ERROR] Feature file not found: {FEAT_FILE}")
        print("Please run on Kaggle T4 first:")
        print("  python scripts/finetune_roberta.py \\")
        print("    --dataset iemocap --model_size large \\")
        print("    --epochs 5 --batch_size 8 --grad_accum 2 \\")
        print("    --data_dir data/raw/IEMOCAP/IEMOCAP_full_release \\")
        print("    --output_dir outputs")
        print("Then copy outputs/iemocap_text_roberta_large_finetuned.pt to data/features/")
        return False
    feat_size_mb = os.path.getsize(FEAT_FILE) / 1e6
    print(f"  Feature file found: {FEAT_FILE} ({feat_size_mb:.1f} MB)")
    return True


def main():
    print("=" * 70)
    print("IEMOCAP 4-class with RoBERTa-Large (1024-dim) Features")
    print(f"  5 seeds x 2 modes = {5*2} experiments")
    print(f"  Target: WF1 > 83% (current base: 79.96%)")
    print("=" * 70)

    if not check_prerequisites():
        sys.exit(1)

    # Base arguments — identical to run_iemocap_4class.py but with large flags
    base_args = [
        sys.executable, "scripts/train_multi_dataset.py",
        "--dataset", "iemocap",
        "--finetuned_large",          # use RoBERTa-Large features (1024-dim)
        "--iemocap_classes", "4",     # 4-class: happy+excited merged
        "--epochs", "50",
        "--patience", "15",
        # feat_dim is auto-set to 1024 by --finetuned_large
    ]

    # Part 1: Federated EAFA (4-class, RoBERTa-Large)
    print("\n--- Part 1: Federated EAFA (4-class, RoBERTa-Large) ---")
    for seed in SEEDS:
        key = f"iemocap4_large_fed_eafa_seed{seed}"
        cmd = base_args + ["--mode", "federated", "--seed", str(seed)]
        run_one(key, cmd)

    # Part 2: Centralized EDL (4-class, RoBERTa-Large)
    print("\n--- Part 2: Centralized EDL (4-class, RoBERTa-Large) ---")
    for seed in SEEDS:
        key = f"iemocap4_large_cent_edl_seed{seed}"
        cmd = base_args + ["--mode", "centralized", "--seed", str(seed)]
        run_one(key, cmd)

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY — IEMOCAP 4-class (RoBERTa-Large, 1024-dim)")
    print("=" * 70)

    results = load_results()

    for config, label in [("fed_eafa", "Federated EAFA"), ("cent_edl", "Centralized EDL")]:
        wf1s = []
        times = []
        for seed in SEEDS:
            key = f"iemocap4_large_{config}_seed{seed}"
            if key in results and results[key].get("wf1"):
                wf1s.append(results[key]["wf1"])
                times.append(results[key].get("time", 0))
        if wf1s:
            mean = np.mean(wf1s)
            std = np.std(wf1s)
            print(f"  {label:25s} (RoBERTa-L): {mean:.4f} +/- {std:.4f} (n={len(wf1s)})")
        else:
            print(f"  {label:25s} (RoBERTa-L): No results yet")

    # Compare with RoBERTa-Base results from results_iemocap_4class.json
    base_file = "results_iemocap_4class.json"
    if os.path.exists(base_file):
        base_results = json.load(open(base_file))
        print("\n  --- Comparison: Base vs Large ---")
        for config, label in [("fed_eafa", "Federated EAFA"), ("cent_edl", "Centralized EDL")]:
            base_wf1s = []
            for seed in [42, 123, 2024]:  # original 3 seeds for base
                key = f"iemocap4_{config}_seed{seed}"
                if key in base_results and base_results[key].get("wf1"):
                    base_wf1s.append(base_results[key]["wf1"])

            large_wf1s = []
            for seed in SEEDS:
                key = f"iemocap4_large_{config}_seed{seed}"
                if key in results and results[key].get("wf1"):
                    large_wf1s.append(results[key]["wf1"])

            if base_wf1s and large_wf1s:
                base_mean = np.mean(base_wf1s)
                large_mean = np.mean(large_wf1s)
                delta = large_mean - base_mean
                sign = "+" if delta >= 0 else ""
                print(f"  {label:25s}: Base {base_mean:.4f} -> Large {large_mean:.4f} ({sign}{delta:.4f})")

    print("\n  SOTA Reference (centralized, full labels, different pipelines):")
    print("    DialogueRNN (GloVe, 5M):              62.57%")
    print("    COSMIC      (RoBERTa-L, 355M):        65.28%")
    print("    EmoBERTa    (RoBERTa-L ft, 355M):     68.57%")
    print()
    print("    Ours (RoBERTa-Base ft, Federated):    79.96% (+/- 1.11, 3 seeds)")
    print("    Ours (RoBERTa-Large ft, Federated):   [see above]")
    print()
    print("  Note: Our method uses federated setting; SOTA uses centralized full-label training.")
    print("  Fair comparison: see Table 2 (cross-dataset, identical features).")
    print("=" * 70)

    # Save comparison to JSON for paper update
    comparison = {
        "sota_reference": {
            "DialogueRNN_GloVe": 62.57,
            "COSMIC_RoBERTa_L": 65.28,
            "EmoBERTa_RoBERTa_L": 68.57,
        },
        "ours_base": {
            "encoder": "roberta-base",
            "feat_dim": 768,
            "setting": "federated",
        },
        "ours_large": {
            "encoder": "roberta-large",
            "feat_dim": 1024,
            "setting": "federated",
        },
        "note": "SOTA uses centralized training; comparison is context, not controlled match."
    }
    json.dump(comparison, open("results_sota_comparison.json", "w"), indent=2)
    print(f"\n  Saved comparison metadata to: results_sota_comparison.json")


if __name__ == "__main__":
    main()
