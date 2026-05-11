"""
Download DailyDialog dataset - direct parquet download
"""
import csv
from pathlib import Path
import requests
import pandas as pd

# Parquet files from HuggingFace (no script needed)
PARQUET_URLS = {
    "train": "https://huggingface.co/api/datasets/roskoN/dailydialog/parquet/full/train/0.parquet",
    "validation": "https://huggingface.co/api/datasets/roskoN/dailydialog/parquet/full/validation/0.parquet",
    "test": "https://huggingface.co/api/datasets/roskoN/dailydialog/parquet/full/test/0.parquet",
}

EMO_NAMES = {0: "no_emotion", 1: "anger", 2: "disgust", 3: "fear", 4: "happiness", 5: "sadness", 6: "surprise"}

def main():
    out = Path("data/raw/DailyDialog")
    out.mkdir(parents=True, exist_ok=True)
    print("Downloading DailyDialog (parquet format)...\n")

    for split, url in PARQUET_URLS.items():
        parquet_path = out / f"{split}.parquet"

        # Download parquet
        if not parquet_path.exists():
            print(f"  Downloading {split}.parquet ...")
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            parquet_path.write_bytes(resp.content)
            print(f"    Saved ({len(resp.content) / 1024:.0f} KB)")
        else:
            print(f"  Already exists: {split}.parquet")

        # Read and convert to CSV
        df = pd.read_parquet(parquet_path)
        csv_path = out / f"{split}.csv"
        total_utt = 0

        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Dialogue_ID", "Utterance_ID", "Utterance", "Emotion", "Act"])
            for i, row in df.iterrows():
                dialog = row["utterances"]
                emotions = row["emotions"]
                acts = row["acts"]
                for j, (utt, emo, act) in enumerate(zip(dialog, emotions, acts)):
                    writer.writerow([i, j, utt, emo, act])
                    total_utt += 1

        print(f"  {split}: {len(df)} dialogues, {total_utt} utterances -> {csv_path}")

    print(f"\nEmotion labels: {EMO_NAMES}")
    print("Done!")

if __name__ == "__main__":
    main()
