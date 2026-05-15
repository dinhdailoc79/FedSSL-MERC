"""Export IEMOCAP utterances to CSV for Kaggle fine-tuning."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from data.datasets.iemocap import IEMOCAPDataset, IEMOCAP_EMOTIONS_6

ds = IEMOCAPDataset(data_dir="data/raw/IEMOCAP/IEMOCAP_full_release", num_classes=6)
ds.load()

rows = []
for session in range(1, 6):
    dias = ds.get_dialogues(session=session)
    for d in dias:
        for u in d.utterances:
            rows.append({
                "Session": session,
                "Dialogue_ID": d.dialogue_id,
                "Utterance_ID": u.utterance_id,
                "Speaker": getattr(u, 'speaker', ''),
                "Utterance": u.text if u.text else " ",
                "Emotion": IEMOCAP_EMOTIONS_6[u.emotion_idx] if u.emotion_idx < len(IEMOCAP_EMOTIONS_6) else "unknown",
                "Emotion_ID": u.emotion_idx,
            })

df = pd.DataFrame(rows)
os.makedirs("kaggle_upload/IEMOCAP", exist_ok=True)

# Split: train=S1-3, dev=S4, test=S5
train_df = df[df["Session"].isin([1, 2, 3])]
dev_df = df[df["Session"] == 4]
test_df = df[df["Session"] == 5]

train_df.to_csv("kaggle_upload/IEMOCAP/train.csv", index=False)
dev_df.to_csv("kaggle_upload/IEMOCAP/dev.csv", index=False)
test_df.to_csv("kaggle_upload/IEMOCAP/test.csv", index=False)

print(f"Train: {len(train_df)} | Dev: {len(dev_df)} | Test: {len(test_df)}")
print(f"Emotions: {df['Emotion'].value_counts().to_dict()}")
print(f"Files saved to kaggle_upload/IEMOCAP/")
