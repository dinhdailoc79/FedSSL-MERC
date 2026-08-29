"""
ThuanPhongNhi: Fine-tune RoBERTa for Emotion Recognition
=====================================================
Designed to run on Kaggle T4 GPU (16 GB VRAM) OR local RTX 4050 6 GB.

Strategy:
1. Fine-tune RoBERTa-base or RoBERTa-large on per-utterance emotion classification
2. Extract features from fine-tuned model
3. Save as .pt files (same format as existing features)
4. Download and use with existing EDL/EAFA pipeline

Usage:
    # Local RTX 4050 6 GB -- chay duoc, ~4-6 gio
    python scripts/finetune_roberta.py --dataset iemocap --model_size large \
        --epochs 5 --batch_size 2 --grad_accum 8 --gradient_checkpointing \
        --lr 1e-5

    # Kaggle T4 16 GB -- nhanh hon ~2-3x
    python scripts/finetune_roberta.py --dataset iemocap --model_size large \
        --epochs 5 --batch_size 8 --grad_accum 2 --lr 1e-5

    # RoBERTa-Base (nhanh, 768-dim, baseline)
    python scripts/finetune_roberta.py --dataset iemocap --model_size base \
        --epochs 5 --batch_size 16

Output filenames:
    base:  {dataset}_text_roberta_finetuned.pt       (768-dim)
    large: {dataset}_text_roberta_large_finetuned.pt (1024-dim)
"""

import os
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import RobertaTokenizer, RobertaModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, classification_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# -------------------------------------------------------
# Dataset classes
# -------------------------------------------------------
class EmotionUtteranceDataset(Dataset):
    """Simple per-utterance dataset for fine-tuning."""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx], truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# Hidden dim per model size
MODEL_HIDDEN_DIM = {"base": 768, "large": 1024}
MODEL_HF_NAME = {"base": "roberta-base", "large": "roberta-large"}


class RobertaEmotionClassifier(nn.Module):
    """RoBERTa (base or large) + classification head for fine-tuning."""

    def __init__(self, num_classes, model_size="base", dropout=0.3,
                 gradient_checkpointing=False):
        super().__init__()
        hf_name = MODEL_HF_NAME[model_size]
        self.hidden_dim = MODEL_HIDDEN_DIM[model_size]
        self.roberta = RobertaModel.from_pretrained(hf_name)
        if gradient_checkpointing:
            # Saves ~30% VRAM at cost of ~15% slower training
            self.roberta.gradient_checkpointing_enable()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pooling over non-padding tokens
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits, pooled  # logits for training, pooled for feature extraction


# -------------------------------------------------------
# Data loading functions
# -------------------------------------------------------
def load_meld_data(data_dir):
    """Load MELD from CSV files."""
    emotions = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    emo2idx = {e: i for i, e in enumerate(emotions)}

    splits = {}
    for split_name, file_name in [("train", "train_sent_emo.csv"),
                                   ("dev", "dev_sent_emo.csv"),
                                   ("test", "test_sent_emo.csv")]:
        path = Path(data_dir) / file_name
        if not path.exists():
            logger.warning(f"Not found: {path}")
            continue
        df = pd.read_csv(path)
        texts = df["Utterance"].astype(str).tolist()
        labels = [emo2idx.get(e.lower(), -1) for e in df["Emotion"]]
        # Filter invalid
        valid = [(t, l) for t, l in zip(texts, labels) if l >= 0]
        texts, labels = zip(*valid) if valid else ([], [])
        splits[split_name] = {"texts": list(texts), "labels": list(labels),
                              "dialogue_ids": df["Dialogue_ID"].tolist(),
                              "utterance_ids": df["Utterance_ID"].tolist()}

    return splits, emotions


def load_iemocap_data(data_dir):
    """Load IEMOCAP from CSV files (exported by export_iemocap_csv.py)."""
    emotions = ["happy", "sad", "neutral", "angry", "excited", "frustrated"]
    emo2idx = {e: i for i, e in enumerate(emotions)}

    splits = {}
    for split_name, file_name in [("train", "train.csv"),
                                   ("dev", "dev.csv"),
                                   ("test", "test.csv")]:
        path = Path(data_dir) / file_name
        if not path.exists():
            logger.warning(f"Not found: {path}")
            continue
        df = pd.read_csv(path)
        texts = df["Utterance"].astype(str).tolist()
        labels = [emo2idx.get(e.lower(), -1) for e in df["Emotion"]]
        valid = [(t, l, d, u) for t, l, d, u in
                 zip(texts, labels, df["Dialogue_ID"].tolist(), df["Utterance_ID"].tolist())
                 if l >= 0]
        if valid:
            texts, labels, dia_ids, utt_ids = zip(*valid)
        else:
            texts, labels, dia_ids, utt_ids = [], [], [], []
        splits[split_name] = {"texts": list(texts), "labels": list(labels),
                              "dialogue_ids": list(dia_ids), "utterance_ids": list(utt_ids)}

    return splits, emotions


def load_dailydialog_data(data_dir):
    """Load DailyDialog from CSV files."""
    emotions = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]
    emo_id_map = {0: -1, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}

    splits = {}
    for split_name, file_name in [("train", "train.csv"),
                                   ("dev", "validation.csv"),
                                   ("test", "test.csv")]:
        path = Path(data_dir) / file_name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        texts, labels, dia_ids, utt_ids = [], [], [], []
        for _, row in df.iterrows():
            emo_idx = emo_id_map.get(int(row["Emotion"]), -1)
            if emo_idx < 0:
                continue  # Skip no_emotion for fine-tuning
            texts.append(str(row["Utterance"]).strip())
            labels.append(emo_idx)
            dia_ids.append(int(row["Dialogue_ID"]))
            utt_ids.append(int(row["Utterance_ID"]))
        splits[split_name] = {"texts": texts, "labels": labels,
                              "dialogue_ids": dia_ids, "utterance_ids": utt_ids}

    return splits, emotions


# -------------------------------------------------------
# Training
# -------------------------------------------------------
def train_epoch(model, loader, optimizer, scheduler, scaler, device, grad_accum=1):
    """Train one epoch with optional gradient accumulation for large models."""
    model.train()
    total_loss, total_samples = 0, 0
    all_preds, all_labels = [], []
    criterion = nn.CrossEntropyLoss()

    optimizer.zero_grad()
    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        with autocast():
            logits, _ = model(input_ids, attention_mask)
            loss = criterion(logits, labels) / grad_accum  # scale loss

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum * labels.size(0)  # unscale for logging
        total_samples += labels.size(0)
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    wf1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return total_loss / max(total_samples, 1), wf1


@torch.no_grad()
def evaluate(model, loader, device, emotion_names=None):
    model.eval()
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        with autocast():
            logits, _ = model(input_ids, attention_mask)

        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    wf1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    report = classification_report(
        all_labels, all_preds, target_names=emotion_names, digits=4, zero_division=0,
    ) if emotion_names else ""
    return wf1, report


# -------------------------------------------------------
# Feature extraction from fine-tuned model
# -------------------------------------------------------
@torch.no_grad()
def extract_features(model, data_splits, tokenizer, device, batch_size=32):
    """Extract features from fine-tuned RoBERTa for ALL utterances (including no_emotion)."""
    model.eval()
    results = {}

    for split_name, split_data in data_splits.items():
        texts = split_data["texts"]
        all_features = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encoding = tokenizer(
                batch_texts, return_tensors="pt", padding=True,
                truncation=True, max_length=128,
            ).to(device)

            with autocast():
                outputs = model.roberta(**encoding)
                mask = encoding["attention_mask"].unsqueeze(-1).float()
                pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                all_features.append(pooled.cpu().float())

            if (i // batch_size + 1) % 100 == 0:
                logger.info(f"  {split_name}: {i + len(batch_texts)}/{len(texts)}")

        features = torch.cat(all_features, dim=0)
        dia_ids = [str(d) for d in split_data["dialogue_ids"]]
        utt_ids = [str(u) for u in split_data["utterance_ids"]]
        results[split_name] = {
            "features": features,
            "dialogue_ids": dia_ids,
            "utterance_ids": utt_ids,
            "dia_id_strs": dia_ids,
            "utt_id_strs": utt_ids,
        }
        logger.info(f"  {split_name}: {features.shape[0]} features extracted ({features.shape[1]}-dim)")

    return results


def extract_all_utterances(model, tokenizer, dataset_name, data_dir, device, batch_size=32):
    """
    Extract features for ALL utterances (including no_emotion for DailyDialog).
    This ensures compatibility with DialogueRNN which needs full dialogue context.
    """
    model.eval()

    if dataset_name == "meld":
        all_splits = {}
        for split_name, file_name in [("train", "train_sent_emo.csv"),
                                       ("dev", "dev_sent_emo.csv"),
                                       ("test", "test_sent_emo.csv")]:
            path = Path(data_dir) / file_name
            df = pd.read_csv(path)
            all_splits[split_name] = {
                "texts": df["Utterance"].astype(str).tolist(),
                "dialogue_ids": df["Dialogue_ID"].tolist(),
                "utterance_ids": df["Utterance_ID"].tolist(),
            }
        return extract_features(model, all_splits, tokenizer, device, batch_size)

    elif dataset_name == "dailydialog":
        all_splits = {}
        for split_name, file_name in [("train", "train.csv"),
                                       ("dev", "validation.csv"),
                                       ("test", "test.csv")]:
            path = Path(data_dir) / file_name
            df = pd.read_csv(path)
            all_splits[split_name] = {
                "texts": df["Utterance"].astype(str).tolist(),
                "dialogue_ids": df["Dialogue_ID"].tolist(),
                "utterance_ids": df["Utterance_ID"].tolist(),
            }
        return extract_features(model, all_splits, tokenizer, device, batch_size)

    elif dataset_name == "iemocap":
        # Load from CSV files (all sessions)
        all_splits = {}
        for split_name, file_name in [("train", "train.csv"),
                                       ("dev", "dev.csv"),
                                       ("test", "test.csv")]:
            path = Path(data_dir) / file_name
            if not path.exists():
                continue
            df = pd.read_csv(path)
            all_splits[split_name] = {
                "texts": df["Utterance"].astype(str).tolist(),
                "dialogue_ids": df["Dialogue_ID"].tolist(),
                "utterance_ids": df["Utterance_ID"].tolist(),
            }
        return extract_features(model, all_splits, tokenizer, device, batch_size)


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ThuanPhongNhi: Fine-tune RoBERTa (base or large)")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["meld", "iemocap", "dailydialog"])
    parser.add_argument("--model_size", type=str, default="base",
                        choices=["base", "large"],
                        help="RoBERTa size: base (768-dim, 125M) or large (1024-dim, 355M)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to dataset. Auto-detected if not specified.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Per-step batch size. "
                             "RTX 4050 6GB: use 2 with --grad_accum 8 --gradient_checkpointing. "
                             "T4 16GB: use 8 with --grad_accum 2.")
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="Gradient accumulation steps. Effective batch = batch_size * grad_accum")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing to save ~30%% VRAM. "
                             "Recommended for RTX 4050 6GB with RoBERTa-Large.")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Auto-detect data dir
    if args.data_dir is None:
        base_paths = [
            "data/raw",                          # Local
            "/kaggle/input/meld-dataset",         # Kaggle
            "/content/data",                      # Colab
        ]
        data_dirs = {
            "meld": ["MELD", "meld"],
            "iemocap": ["IEMOCAP/IEMOCAP_full_release", "iemocap"],
            "dailydialog": ["DailyDialog", "dailydialog"],
        }
        for base in base_paths:
            for subdir in data_dirs[args.dataset]:
                candidate = Path(base) / subdir
                if candidate.exists():
                    args.data_dir = str(candidate)
                    break
            if args.data_dir:
                break
        if not args.data_dir:
            raise FileNotFoundError(f"Cannot find {args.dataset} data. Use --data_dir")

    logger.info(f"\n{'='*60}")
    logger.info(f"  ThuanPhongNhi: Fine-tune RoBERTa — {args.dataset.upper()}")
    logger.info(f"  Data: {args.data_dir}")
    logger.info(f"  Device: {args.device}")
    logger.info(f"  Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    logger.info(f"{'='*60}\n")

    # 1. Load data
    loaders = {
        "meld": load_meld_data,
        "iemocap": load_iemocap_data,
        "dailydialog": load_dailydialog_data,
    }
    splits, emotions = loaders[args.dataset](args.data_dir)
    num_classes = len(emotions)

    for split_name, data in splits.items():
        logger.info(f"  {split_name}: {len(data['texts'])} utterances")

    # 2. Tokenizer + Model
    hf_name = MODEL_HF_NAME[args.model_size]
    hidden_dim = MODEL_HIDDEN_DIM[args.model_size]
    logger.info(f"Loading {hf_name} ({hidden_dim}-dim features)...")
    tokenizer = RobertaTokenizer.from_pretrained(hf_name)
    model = RobertaEmotionClassifier(
        num_classes=num_classes, model_size=args.model_size,
        gradient_checkpointing=args.gradient_checkpointing,
    ).to(args.device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    eff_batch = args.batch_size * args.grad_accum
    gc_str = " + gradient_checkpointing" if args.gradient_checkpointing else ""
    logger.info(f"Model: {params:,} trainable params | "
                f"batch_size={args.batch_size} x grad_accum={args.grad_accum} "
                f"= effective batch {eff_batch}{gc_str}")

    # 3. DataLoaders
    train_ds = EmotionUtteranceDataset(
        splits["train"]["texts"], splits["train"]["labels"],
        tokenizer, args.max_length,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

    dev_ds = EmotionUtteranceDataset(
        splits["dev"]["texts"], splits["dev"]["labels"],
        tokenizer, args.max_length,
    )
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=2)

    test_ds = EmotionUtteranceDataset(
        splits["test"]["texts"], splits["test"]["labels"],
        tokenizer, args.max_length,
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=2)

    # 4. Optimizer + Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = GradScaler()

    # 5. Train
    logger.info(f"\n{'='*60}")
    logger.info(f"  Training: {args.epochs} epochs, {total_steps} steps (grad_accum={args.grad_accum})")
    logger.info(f"{'='*60}\n")

    best_wf1 = 0.0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Checkpoint name encodes model size for clarity
    ckpt_name = f"best_roberta_{args.model_size}_{args.dataset}.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_wf1 = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            args.device, grad_accum=args.grad_accum,
        )
        dev_wf1, _ = evaluate(model, dev_loader, args.device)

        logger.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} WF1: {train_wf1:.4f} | "
            f"Dev WF1: {dev_wf1:.4f}"
        )

        if dev_wf1 > best_wf1:
            best_wf1 = dev_wf1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "dev_wf1": dev_wf1,
                "dataset": args.dataset,
                "num_classes": num_classes,
                "emotions": emotions,
                "model_size": args.model_size,
                "hidden_dim": hidden_dim,
            }, output_dir / ckpt_name)
            logger.info(f"  >> Saved best model ({args.model_size})! WF1={dev_wf1:.4f}")

    # 6. Test evaluation
    ckpt = torch.load(output_dir / ckpt_name, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_wf1, test_report = evaluate(model, test_loader, args.device, emotions)

    logger.info(f"\n{'='*60}")
    logger.info(f"  Fine-tuned RoBERTa-{args.model_size.upper()} — {args.dataset.upper()}")
    logger.info(f"  Hidden dim: {hidden_dim}")
    logger.info(f"{'='*60}")
    logger.info(f"\n{test_report}")
    logger.info(f"  Test WF1 = {test_wf1:.4f}")
    logger.info(f"{'='*60}")

    # 7. Extract features from fine-tuned model
    logger.info(f"\nExtracting {hidden_dim}-dim features from fine-tuned model...")
    features = extract_all_utterances(
        model, tokenizer, args.dataset, args.data_dir, args.device,
    )

    # Naming convention: base -> _finetuned.pt, large -> _large_finetuned.pt
    if args.model_size == "large":
        feat_fname = f"{args.dataset}_text_roberta_large_finetuned.pt"
    else:
        feat_fname = f"{args.dataset}_text_roberta_finetuned.pt"

    feat_path = output_dir / feat_fname
    torch.save(features, str(feat_path))
    size_mb = feat_path.stat().st_size / 1e6
    logger.info(f"  Saved: {feat_path} ({size_mb:.1f} MB)")

    logger.info(f"\n{'='*60}")
    if args.model_size == "large":
        logger.info(f"  DONE! Download '{feat_path.name}' and place in data/features/")
        logger.info(f"  Then run:")
        logger.info(f"    python scripts/run_iemocap_4class_large.py")
    else:
        logger.info(f"  DONE! Download '{feat_path.name}' and place in data/features/")
        logger.info(f"  Then run: python scripts/train_multi_dataset.py --dataset {args.dataset}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
