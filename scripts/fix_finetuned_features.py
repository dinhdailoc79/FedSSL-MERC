"""
Fix finetuned feature files: replace hash-based IDs with raw CSV IDs.
The finetune_roberta.py used hash(str(id)) which doesn't match the pipeline's raw IDs.
"""
import torch
import pandas as pd
import numpy as np
from pathlib import Path

def fix_meld():
    """Fix MELD finetuned features."""
    path = Path("data/features/meld_text_roberta_finetuned.pt")
    if not path.exists():
        print(f"Not found: {path}")
        return

    cached = torch.load(str(path), weights_only=False)
    fixed = {}

    for split_name, csv_file in [("train", "train_sent_emo.csv"),
                                  ("dev", "dev_sent_emo.csv"),
                                  ("test", "test_sent_emo.csv")]:
        if split_name not in cached:
            print(f"  Skip: {split_name} not in cache")
            continue

        df = pd.read_csv(f"data/raw/MELD/{csv_file}")
        features = cached[split_name]["features"]

        # The features are in CSV row order
        n_features = features.shape[0]
        n_csv = len(df)
        print(f"  {split_name}: {n_features} features, {n_csv} CSV rows")

        if n_features != n_csv:
            print(f"  WARNING: count mismatch!")
            n = min(n_features, n_csv)
        else:
            n = n_features

        fixed[split_name] = {
            "features": features[:n],
            "dialogue_ids": torch.tensor(df["Dialogue_ID"].values[:n], dtype=torch.long),
            "utterance_ids": torch.tensor(df["Utterance_ID"].values[:n], dtype=torch.long),
        }
        print(f"  {split_name}: Fixed IDs (dia range: {df['Dialogue_ID'].min()}-{df['Dialogue_ID'].max()})")

    out_path = Path("data/features/meld_text_roberta_finetuned.pt")
    torch.save(fixed, str(out_path))
    print(f"\nSaved: {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")


def fix_dailydialog():
    """Fix DailyDialog finetuned features."""
    path = Path("data/features/dailydialog_text_roberta_finetuned.pt")
    if not path.exists():
        print(f"Not found: {path}")
        return

    cached = torch.load(str(path), weights_only=False)
    fixed = {}

    for split_name, csv_file in [("train", "train.csv"),
                                  ("dev", "validation.csv"),
                                  ("test", "test.csv")]:
        if split_name not in cached:
            print(f"  Skip: {split_name} not in cache")
            continue

        df = pd.read_csv(f"data/raw/DailyDialog/{csv_file}")
        features = cached[split_name]["features"]

        n_features = features.shape[0]
        n_csv = len(df)
        print(f"  {split_name}: {n_features} features, {n_csv} CSV rows")

        if n_features != n_csv:
            print(f"  WARNING: count mismatch!")
            n = min(n_features, n_csv)
        else:
            n = n_features

        fixed[split_name] = {
            "features": features[:n],
            "dialogue_ids": torch.tensor(df["Dialogue_ID"].values[:n], dtype=torch.long),
            "utterance_ids": torch.tensor(df["Utterance_ID"].values[:n], dtype=torch.long),
        }
        print(f"  {split_name}: Fixed IDs (dia range: {df['Dialogue_ID'].min()}-{df['Dialogue_ID'].max()})")

    out_path = Path("data/features/dailydialog_text_roberta_finetuned.pt")
    torch.save(fixed, str(out_path))
    print(f"\nSaved: {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")


def fix_iemocap():
    """Fix IEMOCAP finetuned features — convert split-based to session-based."""
    path = Path("data/features/iemocap_text_roberta_finetuned.pt")
    if not path.exists():
        print(f"Not found: {path}")
        return

    cached = torch.load(str(path), weights_only=False)
    fixed = {}

    for split_name, csv_file in [("train", "train.csv"),
                                  ("dev", "dev.csv"),
                                  ("test", "test.csv")]:
        if split_name not in cached:
            print(f"  Skip: {split_name} not in cache")
            continue

        df = pd.read_csv(f"kaggle_upload/IEMOCAP/{csv_file}")
        features = cached[split_name]["features"]

        n_features = features.shape[0]
        n_csv = len(df)
        print(f"  {split_name}: {n_features} features, {n_csv} CSV rows")

        n = min(n_features, n_csv)
        fixed[split_name] = {
            "features": features[:n],
            "dialogue_ids": df["Dialogue_ID"].values[:n].tolist(),
            "utterance_ids": df["Utterance_ID"].values[:n].tolist(),
            "dia_id_strs": df["Dialogue_ID"].astype(str).values[:n].tolist(),
            "utt_id_strs": df["Utterance_ID"].astype(str).values[:n].tolist(),
        }
        print(f"  {split_name}: Fixed ({n} utterances)")

    out_path = Path("data/features/iemocap_text_roberta_finetuned.pt")
    torch.save(fixed, str(out_path))
    print(f"\nSaved: {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    print("=== Fixing MELD finetuned features ===")
    fix_meld()
    print("\n=== Fixing DailyDialog finetuned features ===")
    fix_dailydialog()
    print("\n=== Fixing IEMOCAP finetuned features ===")
    fix_iemocap()
    print("\nDone! Now re-run train_multi_dataset.py --finetuned")
