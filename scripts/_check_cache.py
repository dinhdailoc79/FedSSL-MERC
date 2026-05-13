"""Quick check: verify feature cache keys match MELD dialogue utterance IDs."""
import sys; sys.path.insert(0, '.')
import torch
from data.datasets.meld import MELDDataset

# Load cache
cached = torch.load("data/features/meld_text_roberta.pt", weights_only=False)
print("Cache keys per split:")
for split in ["train", "dev", "test"]:
    d = cached[split]
    print(f"  {split}: features={d['features'].shape}, utt_ids type={type(d['utterance_ids'])}")
    print(f"    First 5 utt_ids: {d['utterance_ids'][:5].tolist()}")
    print(f"    First 5 dia_ids: {d['dialogue_ids'][:5].tolist()}")

# Load MELD
meld = MELDDataset("data/raw/MELD")
for split in ["train"]:
    dialogues = meld.get_dialogues(split)
    d0 = dialogues[0]
    print(f"\nFirst dialogue ({split}): id={d0.dialogue_id}, {len(d0.utterances)} utts")
    for u in d0.utterances[:5]:
        print(f"  utt_id={u.utterance_id}, dia_id={u.dialogue_id}, speaker={u.speaker}")
        key = f"{u.dialogue_id}_{u.utterance_id}"
        print(f"    cache key: {key}")
