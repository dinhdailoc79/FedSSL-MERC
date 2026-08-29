"""
=======================================================================
Kaggle T4 — RoBERTa-Large Fine-tuning + Feature Extraction
IEMOCAP 4-class for FedSSL-MERC SOTA push
=======================================================================

STEP 1: Upload/clone your repo to Kaggle (or use the FedSSL-MERC zip)
STEP 2: Run this notebook cell by cell

Expected runtime on T4 (16 GB):
  Fine-tuning (5 epochs, batch=8, accum=2): ~3-4 hours
  Feature extraction: ~15-20 minutes
  Total: ~4 hours
"""

# -----------------------------------------------------------------------
# CELL 1: Setup
# -----------------------------------------------------------------------
import os, sys

# If running from Kaggle, clone the repo or upload zip
# !pip install -q transformers==4.40.0 torch scikit-learn tqdm pandas numpy

# Set working directory
REPO_DIR = "/kaggle/working/FedSSL-MERC"   # adjust if different
if os.path.isdir(REPO_DIR):
    os.chdir(REPO_DIR)
    sys.path.insert(0, REPO_DIR)

# Check GPU
import torch
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
print(f"GPU: {gpu}, VRAM: {vram:.1f} GB")
assert vram > 8, f"Need >= 8 GB VRAM for RoBERTa-Large, got {vram:.1f} GB"


# -----------------------------------------------------------------------
# CELL 2: Paths (adjust to your Kaggle dataset structure)
# -----------------------------------------------------------------------
# Mount your IEMOCAP dataset — typically added as a Kaggle dataset:
IEMOCAP_DIR = "/kaggle/input/iemocap-data/IEMOCAP_full_release"  # adjust
OUTPUT_DIR   = "/kaggle/working/outputs"
FEAT_DIR     = "/kaggle/working/data/features"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FEAT_DIR, exist_ok=True)

# Verify IEMOCAP exists
assert os.path.isdir(IEMOCAP_DIR), f"IEMOCAP not found: {IEMOCAP_DIR}"
print(f"IEMOCAP found: {IEMOCAP_DIR}")


# -----------------------------------------------------------------------
# CELL 3: Fine-tune RoBERTa-Large on IEMOCAP
# -----------------------------------------------------------------------
# This is the key step — fine-tunes on emotion classification
# RoBERTa-Large: 355M params, output 1024-dim

cmd_finetune = f"""python scripts/finetune_roberta.py \
    --dataset iemocap \
    --model_size large \
    --data_dir {IEMOCAP_DIR} \
    --epochs 5 \
    --batch_size 8 \
    --grad_accum 2 \
    --lr 1e-5 \
    --warmup_ratio 0.1 \
    --max_length 128 \
    --output_dir {OUTPUT_DIR}
"""
# Effective batch = 8 * 2 = 16. LR=1e-5 slightly lower for large model stability.
print("Running:", cmd_finetune)
# !{cmd_finetune}  # Uncomment in Jupyter/Kaggle


# -----------------------------------------------------------------------
# CELL 4: Verify outputs
# -----------------------------------------------------------------------
import os
ckpt_path  = f"{OUTPUT_DIR}/best_roberta_large_iemocap.pt"
feat_path  = f"{OUTPUT_DIR}/iemocap_text_roberta_large_finetuned.pt"

if os.path.exists(ckpt_path):
    size_ckpt = os.path.getsize(ckpt_path) / 1e6
    print(f"[OK] Checkpoint: {ckpt_path} ({size_ckpt:.0f} MB)")
else:
    print(f"[MISSING] {ckpt_path} — finetune may have failed")

if os.path.exists(feat_path):
    size_feat = os.path.getsize(feat_path) / 1e6
    print(f"[OK] Features:   {feat_path} ({size_feat:.0f} MB)")

    # Inspect feature shape
    import torch
    data = torch.load(feat_path, weights_only=False)
    for split, d in data.items():
        if isinstance(d, dict) and "features" in d:
            print(f"     {split}: features {d['features'].shape}")
            assert d['features'].shape[1] == 1024, f"Expected 1024-dim, got {d['features'].shape[1]}"
    print("[OK] Feature dim = 1024 confirmed")
else:
    print(f"[MISSING] {feat_path} — features not extracted yet")


# -----------------------------------------------------------------------
# CELL 5: Copy features to the right location for training
# -----------------------------------------------------------------------
import shutil

dest_feat = f"{REPO_DIR}/data/features/iemocap_text_roberta_large_finetuned.pt"
os.makedirs(os.path.dirname(dest_feat), exist_ok=True)

if os.path.exists(feat_path):
    shutil.copy(feat_path, dest_feat)
    print(f"[OK] Copied features to: {dest_feat}")
else:
    print("[SKIP] Feature file not available yet")


# -----------------------------------------------------------------------
# CELL 6: Run IEMOCAP 4-class experiment with RoBERTa-Large
# -----------------------------------------------------------------------
# This runs 5 seeds x 2 modes = 10 experiments
# Expected time: ~2-3 hours on T4

cmd_experiment = "python scripts/run_iemocap_4class_large.py"
print("Running:", cmd_experiment)
# !{cmd_experiment}  # Uncomment in Jupyter/Kaggle


# -----------------------------------------------------------------------
# CELL 7: Review results
# -----------------------------------------------------------------------
import json, numpy as np

results_path = f"{REPO_DIR}/results_iemocap_4class_large.json"
if os.path.exists(results_path):
    results = json.load(open(results_path))

    SEEDS = [42, 123, 2024, 777, 999]
    print("\n" + "=" * 60)
    print("FINAL RESULTS — IEMOCAP 4-class, RoBERTa-Large")
    print("=" * 60)

    for config, label in [("fed_eafa", "Federated EAFA"), ("cent_edl", "Centralized EDL")]:
        wf1s = []
        for seed in SEEDS:
            key = f"iemocap4_large_{config}_seed{seed}"
            if key in results and results[key].get("wf1"):
                wf1s.append(results[key]["wf1"])
        if wf1s:
            mean, std = np.mean(wf1s), np.std(wf1s)
            print(f"  {label}: {mean*100:.2f}% +/- {std*100:.2f}% (n={len(wf1s)})")

    print("\n  SOTA comparison:")
    print("    EmoBERTa (RoBERTa-L ft, centralized): 68.57%")
    print("    Ours (RoBERTa-L ft, federated EAFA): [see above]")
else:
    print(f"Results not found at: {results_path}")


# -----------------------------------------------------------------------
# CELL 8: Download output files
# -----------------------------------------------------------------------
# After running, download these files to your local machine:
#   1. outputs/iemocap_text_roberta_large_finetuned.pt
#      -> place in: d:\OJT\FedSSL-MERC\data\features\
#   2. results_iemocap_4class_large.json
#      -> place in: d:\OJT\FedSSL-MERC\
#   3. outputs/best_roberta_large_iemocap.pt (model checkpoint, keep safe)

print("\nFiles to download from /kaggle/working/:")
print("  outputs/iemocap_text_roberta_large_finetuned.pt  (~50-100 MB)")
print("  results_iemocap_4class_large.json")
print("  outputs/best_roberta_large_iemocap.pt")
