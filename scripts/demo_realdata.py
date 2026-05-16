"""
ThuanPhongNhi Demo — Real Test Data Emotion Predictions
===================================================
Uses pre-computed features + trained models → accurate predictions.

Usage:
    python scripts/demo_realdata.py
    python scripts/demo_realdata.py --dataset dailydialog --num 5
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import torch
import numpy as np
from pathlib import Path
from models.evidential.evidential_dialogue_rnn import EvidentialDialogueRNN

CONFIGS = {
    "meld": {
        "emotions": ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"],
        "emoji":    ["😠",     "🤢",      "😨",   "😊",  "😐",      "😢",      "😮"],
        "nc": 7, "ns": 10,
        "edl": "checkpoints/best_edl_meld.pt",
        "eafa": "checkpoints/best_eafa_meld.pt",
        "feat": "data/features/meld_text_roberta_finetuned.pt",
    },
    "dailydialog": {
        "emotions": ["anger", "disgust", "fear", "happiness", "sadness", "surprise"],
        "emoji":    ["😠",     "🤢",      "😨",   "😊",        "😢",      "😮"],
        "nc": 6, "ns": 10,
        "edl": "checkpoints/best_edl_dailydialog.pt",
        "eafa": "checkpoints/best_eafa_dailydialog.pt",
        "feat": "data/features/dailydialog_text_roberta_finetuned.pt",
    },
}


def load_model(nc, ns, ckpt_path, device):
    model = EvidentialDialogueRNN(
        input_dim=768, hidden_dim=256, num_classes=nc,
        num_speakers=ns, dropout=0.3,
    ).to(device)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    return model


def load_data_and_cache(dataset_name):
    """Load test dialogues + feature cache."""
    cfg = CONFIGS[dataset_name]
    
    if dataset_name == "meld":
        from data.datasets.meld import MELDDataset
        ds = MELDDataset(data_dir="data/raw/MELD")
        dialogues = ds.get_dialogues("test")
    elif dataset_name == "dailydialog":
        from data.datasets.dailydialog import DailyDialogDataset
        ds = DailyDialogDataset(data_dir="data/raw/DailyDialog")
        ds.load()
        dialogues = ds.get_dialogues("test")

    # Build feature cache: key = "dialogue_id_utterance_id" → numpy array
    raw = torch.load(cfg["feat"], weights_only=False)
    cache = {}
    split = "test"
    if split in raw:
        feats = raw[split]["features"].numpy()
        dia_ids = raw[split]["dialogue_ids"]
        utt_ids = raw[split]["utterance_ids"]
        for i in range(len(feats)):
            cache[f"{dia_ids[i].item()}_{utt_ids[i].item()}"] = feats[i]

    return dialogues, cache


def u_bar(u, w=20):
    f = min(int(u * w), w)
    c = "\033[92m" if u < 0.2 else "\033[93m" if u < 0.5 else "\033[91m"
    return f"{c}{'█'*f}{'░'*(w-f)}\033[0m {u:.3f}"


def run(dataset_name, num, device):
    cfg = CONFIGS[dataset_name]
    emos = cfg["emotions"]
    emojis = cfg["emoji"]

    print("\033[1;36m")
    print("╔══════════════════════════════════════════════════════════╗")
    print("║       🧠 ThuanPhongNhi — Emotion Recognition Demo            ║")
    print("║    Evidential Deep Learning + Federated Aggregation     ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print("\033[0m")
    print(f"  📦 Dataset: {dataset_name.upper()}")
    print(f"  🏷️  Emotions: {', '.join(emos)}\n")

    # Load
    print("  🔧 Loading data & models...")
    dialogues, cache = load_data_and_cache(dataset_name)
    edl = load_model(cfg["nc"], cfg["ns"], cfg["edl"], device)
    eafa = load_model(cfg["nc"], cfg["ns"], cfg["eafa"], device)
    print(f"  ✅ {len(dialogues)} test dialogues, {len(cache)} cached features\n")

    # Pick dialogues with emotion variety
    scored = []
    for d in dialogues:
        labels = set(u.emotion_idx for u in d.utterances if u.emotion_idx >= 0)
        if len(labels) >= 2 and 3 <= len(d.utterances) <= 12:
            scored.append((len(labels), d))
    scored.sort(key=lambda x: -x[0])
    selected = [d for _, d in scored[:num]] if scored else dialogues[:num]

    total_edl_correct, total_eafa_correct, total_count = 0, 0, 0

    for idx, dia in enumerate(selected):
        # Build tensors
        feat_list, spk_list, texts, true_idxs = [], [], [], []
        for u in dia.utterances:
            key = f"{dia.dialogue_id}_{u.utterance_id}"
            feat_list.append(
                torch.from_numpy(cache[key]) if key in cache
                else torch.zeros(768)
            )
            spk_list.append(getattr(u, 'speaker_id', 0))
            texts.append(u.text if hasattr(u, 'text') and u.text else "")
            true_idxs.append(u.emotion_idx)

        feats = torch.stack(feat_list).unsqueeze(0).to(device)
        spk = torch.tensor(spk_list, dtype=torch.long).unsqueeze(0).to(device)

        print(f"\033[1;35m{'═'*58}\033[0m")
        print(f"\033[1;35m  📖 Dialogue {idx+1}/{len(selected)} (id: {dia.dialogue_id})\033[0m")
        print(f"\033[1;35m{'═'*58}\033[0m")

        for model, label in [(edl, "EDL Centralized"), (eafa, "EAFA Federated")]:
            with torch.no_grad():
                out = model(feats, spk)
            beliefs = out["belief"][0]
            uncertainties = out["uncertainty"][0]
            preds = beliefs.argmax(dim=-1)

            print(f"\n  \033[1;33m🔬 {label}\033[0m")
            print(f"  {'─'*54}")

            correct = 0
            valid = 0
            for i in range(len(texts)):
                pi = preds[i].item()
                ti = true_idxs[i]
                conf = beliefs[i, pi].item() * 100
                u = uncertainties[i].item()

                t_short = texts[i][:45] + "..." if len(texts[i]) > 45 else texts[i]
                true_str = emos[ti] if 0 <= ti < len(emos) else "?"
                match = ""
                if 0 <= ti < len(emos):
                    match = "\033[92m✓\033[0m" if pi == ti else "\033[91m✗\033[0m"
                    if pi == ti:
                        correct += 1
                    valid += 1

                print(f"  🗣️ \"{t_short}\"")
                print(f"     {emojis[pi]} {emos[pi]:12s} ({conf:5.1f}%) "
                      f"u:{u_bar(u)}  [true:{true_str}] {match}")

            if valid > 0:
                acc = correct / valid * 100
                print(f"  📊 {correct}/{valid} = {acc:.0f}%")
                if label == "EDL Centralized":
                    total_edl_correct += correct
                else:
                    total_eafa_correct += correct
                total_count += valid if label == "EDL Centralized" else 0
        print()

    # Summary
    if total_count > 0:
        print(f"\033[1;36m{'═'*58}\033[0m")
        print(f"\033[1;36m  📊 Overall ({len(selected)} dialogues)\033[0m")
        print(f"\033[1;36m{'═'*58}\033[0m")
        print(f"  EDL  Centralized: {total_edl_correct}/{total_count} = {total_edl_correct/total_count*100:.1f}%")
        print(f"  EAFA Federated:   {total_eafa_correct}/{total_count} = {total_eafa_correct/total_count*100:.1f}%")
        print()
        print(f"  🔑 Key: \033[92m✓\033[0m=correct  \033[91m✗\033[0m=wrong")
        print(f"  🔑 Uncertainty: \033[92m█low\033[0m  \033[93m█med\033[0m  \033[91m█high\033[0m")
        print()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="meld", choices=["meld", "dailydialog"])
    p.add_argument("--num", type=int, default=3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    run(args.dataset, args.num, args.device)
