import torch

text = torch.load("data/features/meld_text_roberta.pt", map_location="cpu", weights_only=False)
audio = torch.load("data/features/meld_audio_wavlm.pt", map_location="cpu", weights_only=False)

for split in ["train", "test"]:
    t_ids = text[split]["dialogue_ids"]
    t_uids = text[split]["utterance_ids"]
    a_ids = audio[split]["dialogue_ids"]
    a_uids = audio[split]["utterance_ids"]

    t_keys = set(f"{t_ids[i].item()}_{t_uids[i].item()}" for i in range(len(t_ids)))
    a_keys = set(f"{a_ids[i].item()}_{a_uids[i].item()}" for i in range(len(a_ids)))

    overlap = t_keys & a_keys
    t_only = t_keys - a_keys
    a_only = a_keys - t_keys

    tf = text[split]["features"]
    af = audio[split]["features"]

    print(f"{split}:")
    print(f"  Text:  {len(t_keys)} keys, shape={tf.shape}")
    print(f"  Audio: {len(a_keys)} keys, shape={af.shape}")
    print(f"  Overlap: {len(overlap)}, Text-only: {len(t_only)}, Audio-only: {len(a_only)}")
    print(f"  Sample text:  {sorted(list(t_keys))[:3]}")
    print(f"  Sample audio: {sorted(list(a_keys))[:3]}")
    print()
