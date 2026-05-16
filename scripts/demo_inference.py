"""
ThuanPhongNhi Demo — Emotion Recognition with Uncertainty
=====================================================
Interactive demo: type a dialogue → see emotion predictions + uncertainty.

Usage:
    python scripts/demo_inference.py
    python scripts/demo_inference.py --dataset meld
    python scripts/demo_inference.py --dataset dailydialog --use_eafa
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
from models.evidential.evidential_dialogue_rnn import EvidentialDialogueRNN

# ─── Constants ───────────────────────────────────────────────
DATASET_CONFIG = {
    "meld": {
        "emotions": ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"],
        "emoji":    ["😠",     "🤢",      "😨",   "😊",  "😐",      "😢",      "😮"],
        "num_classes": 7, "num_speakers": 10,
        "ckpt_edl": "checkpoints/best_edl_meld.pt",
        "ckpt_eafa": "checkpoints/best_eafa_meld.pt",
    },
    "iemocap": {
        "emotions": ["anger", "excited", "frustration", "happiness", "neutral", "sadness"],
        "emoji":    ["😠",     "🤩",      "😤",          "😊",         "😐",      "😢"],
        "num_classes": 6, "num_speakers": 2,
        "ckpt_edl": "checkpoints/best_edl_iemocap.pt",
        "ckpt_eafa": "checkpoints/best_eafa_iemocap.pt",
    },
    "dailydialog": {
        "emotions": ["anger", "disgust", "fear", "happiness", "sadness", "surprise"],
        "emoji":    ["😠",     "🤢",      "😨",   "😊",        "😢",      "😮"],
        "num_classes": 6, "num_speakers": 10,
        "ckpt_edl": "checkpoints/best_edl_dailydialog.pt",
        "ckpt_eafa": "checkpoints/best_eafa_dailydialog.pt",
    },
}

SAMPLE_DIALOGUES = [
    {
        "title": "Office Promotion",
        "utterances": [
            ("A", "I just got promoted to senior engineer!"),
            ("B", "Oh wow, that's incredible! Congratulations!"),
            ("A", "Thanks! But it means I have to relocate to another city."),
            ("B", "Oh no, that's really sad. We'll miss you."),
            ("A", "I'm scared about the change, honestly."),
            ("B", "Don't worry, you'll do amazing there!"),
        ],
    },
    {
        "title": "Restaurant Complaint",
        "utterances": [
            ("A", "Excuse me, I've been waiting for 40 minutes."),
            ("B", "I'm so sorry about that. Let me check on your order."),
            ("A", "This is unacceptable. The food is cold too."),
            ("B", "I sincerely apologize. We'll remake it right away."),
            ("A", "Fine. But I want to speak with the manager."),
        ],
    },
    {
        "title": "Weekend Plans",
        "utterances": [
            ("A", "Hey, want to go hiking this weekend?"),
            ("B", "Sure! That sounds like fun."),
            ("A", "Great, I found this beautiful trail near the mountains."),
            ("B", "Perfect! I can't wait. Should we invite others?"),
            ("A", "Yeah, let's ask the whole group!"),
        ],
    },
]


def load_model(dataset_name, use_eafa, device):
    """Load trained EvidentialDialogueRNN from checkpoint."""
    cfg = DATASET_CONFIG[dataset_name]
    ckpt_path = cfg["ckpt_eafa"] if use_eafa else cfg["ckpt_edl"]

    model = EvidentialDialogueRNN(
        input_dim=768, hidden_dim=256,
        num_classes=cfg["num_classes"],
        num_speakers=cfg["num_speakers"],
        dropout=0.3,
    ).to(device)

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        print(f"  ✅ Loaded: {ckpt_path}")
    else:
        print(f"  ⚠️  Checkpoint not found: {ckpt_path} — using random weights")

    model.eval()
    return model


def extract_features(texts, tokenizer, roberta, device):
    """Extract RoBERTa features for a list of utterance texts."""
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True,
        truncation=True, max_length=128,
    ).to(device)

    with torch.no_grad():
        outputs = roberta(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    return pooled  # (num_utts, 768)


def predict_dialogue(model, features, speaker_ids, device, cfg):
    """Run model on a single dialogue, return predictions."""
    # Shape: (1, seq_len, 768)
    feats = features.unsqueeze(0).to(device)
    spk = speaker_ids.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(feats, spk)

    beliefs = out["belief"][0]       # (seq_len, C)
    uncertainties = out["uncertainty"][0]  # (seq_len,)

    preds = beliefs.argmax(dim=-1)   # (seq_len,)
    confs = beliefs.max(dim=-1).values  # (seq_len,)

    results = []
    for i in range(len(preds)):
        results.append({
            "pred_idx": preds[i].item(),
            "emotion": cfg["emotions"][preds[i].item()],
            "emoji": cfg["emoji"][preds[i].item()],
            "confidence": confs[i].item(),
            "uncertainty": uncertainties[i].item(),
            "belief_dist": beliefs[i].cpu().numpy(),
        })
    return results


def uncertainty_bar(u, width=20):
    """Create a visual uncertainty bar."""
    filled = int(u * width)
    filled = min(filled, width)
    if u < 0.2:
        color = "\033[92m"  # Green
    elif u < 0.5:
        color = "\033[93m"  # Yellow
    else:
        color = "\033[91m"  # Red
    reset = "\033[0m"
    bar = "█" * filled + "░" * (width - filled)
    return f"{color}{bar}{reset} {u:.3f}"


def print_header():
    """Print demo header."""
    print("\033[1;36m")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          🧠 ThuanPhongNhi — Emotion Recognition Demo             ║")
    print("║     Evidential Deep Learning + Federated Aggregation        ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print("\033[0m")


def print_results(utterances, results, model_label):
    """Print formatted results for a dialogue."""
    print(f"\n\033[1;33m  Model: {model_label}\033[0m")
    print(f"  {'─' * 56}")

    for i, (speaker, text) in enumerate(utterances):
        r = results[i]
        conf_pct = r["confidence"] * 100
        print(f"  🗣️ {speaker}: \"{text}\"")
        print(f"     → {r['emoji']} {r['emotion']:12s} ({conf_pct:5.1f}%)  "
              f"uncertainty: {uncertainty_bar(r['uncertainty'])}")
    print()


def run_demo(dataset_name, use_eafa, device):
    """Run interactive demo."""
    cfg = DATASET_CONFIG[dataset_name]
    model_label = f"EAFA Federated ({dataset_name})" if use_eafa else f"EDL Centralized ({dataset_name})"

    print_header()
    print(f"  📦 Dataset config: {dataset_name.upper()}")
    print(f"  🏷️  Emotions: {', '.join(cfg['emotions'])}")
    print(f"  🔧 Loading RoBERTa-base tokenizer & encoder...")

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    roberta = RobertaModel.from_pretrained("roberta-base").to(device)
    roberta.eval()
    print(f"  ✅ RoBERTa loaded on {device}")

    print(f"  🔧 Loading {model_label}...")
    model = load_model(dataset_name, use_eafa, device)

    # ─── Sample dialogues ───
    print(f"\n\033[1;35m{'═' * 60}\033[0m")
    print(f"\033[1;35m  📝 Sample Dialogues\033[0m")
    print(f"\033[1;35m{'═' * 60}\033[0m")

    for sample in SAMPLE_DIALOGUES:
        print(f"\n\033[1;37m  📖 {sample['title']}\033[0m")

        texts = [text for _, text in sample["utterances"]]
        speakers_map = {}
        speaker_ids = []
        for spk, _ in sample["utterances"]:
            if spk not in speakers_map:
                speakers_map[spk] = len(speakers_map)
            speaker_ids.append(speakers_map[spk])

        features = extract_features(texts, tokenizer, roberta, device)
        spk_tensor = torch.tensor(speaker_ids, dtype=torch.long)
        results = predict_dialogue(model, features, spk_tensor, device, cfg)
        print_results(sample["utterances"], results, model_label)

    # ─── Interactive mode ───
    print(f"\n\033[1;35m{'═' * 60}\033[0m")
    print(f"\033[1;35m  💬 Interactive Mode\033[0m")
    print(f"\033[1;35m{'═' * 60}\033[0m")
    print("  Type dialogue utterances (format: 'Speaker: text')")
    print("  Type 'done' to predict, 'quit' to exit, 'clear' to reset\n")

    while True:
        utterances = []
        print("  \033[1;36m--- New dialogue ---\033[0m")

        while True:
            try:
                line = input("  > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  👋 Bye!")
                return

            if line.lower() == "quit":
                print("  👋 Bye!")
                return
            if line.lower() == "clear":
                break
            if line.lower() == "done":
                if not utterances:
                    print("  ⚠️  No utterances entered. Try again.")
                    continue
                break

            # Parse "Speaker: text"
            if ":" in line:
                spk, text = line.split(":", 1)
                utterances.append((spk.strip(), text.strip()))
            else:
                utterances.append(("A", line))

        if not utterances:
            continue

        texts = [text for _, text in utterances]
        speakers_map = {}
        speaker_ids = []
        for spk, _ in utterances:
            if spk not in speakers_map:
                speakers_map[spk] = len(speakers_map)
            speaker_ids.append(speakers_map[spk])

        features = extract_features(texts, tokenizer, roberta, device)
        spk_tensor = torch.tensor(speaker_ids, dtype=torch.long)
        results = predict_dialogue(model, features, spk_tensor, device, cfg)
        print_results(utterances, results, model_label)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ThuanPhongNhi Inference Demo")
    parser.add_argument("--dataset", type=str, default="meld",
                        choices=["meld", "iemocap", "dailydialog"])
    parser.add_argument("--use_eafa", action="store_true",
                        help="Use EAFA federated model instead of EDL centralized")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_demo(args.dataset, args.use_eafa, args.device)
