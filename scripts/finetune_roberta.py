"""
ThuanPhongNhi: Fine-tune RoBERTa for Emotion Recognition
=====================================================
Designed to run on Kaggle T4 GPU.

Strategy:
1. Fine-tune RoBERTa-base on per-utterance emotion classification
2. Extract features from fine-tuned model
3. Save as .pt files (same format as existing features)
4. Download and use with existing EDL/EAFA pipeline

Usage (Kaggle):
    !python finetune_roberta.py --dataset meld --epochs 5 --batch_size 16
    !python finetune_roberta.py --dataset iemocap --epochs 5
    !python finetune_roberta.py --dataset dailydialog --epochs 3
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


class RobertaEmotionClassifier(nn.Module):
    """RoBERTa + classification head for fine-tuning."""

    def __init__(self, num_classes, dropout=0.3):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pooling
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
def train_epoch(model, loader, optimizer, scheduler, scaler, device):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    all_preds, all_labels = [], []
    criterion = nn.CrossEntropyLoss()

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        with autocast():
            logits, _ = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * labels.size(0)
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
        results[split_name] = {
            "features": features,
            "dialogue_ids": torch.tensor([hash(str(d)) % (2**31) for d in split_data["dialogue_ids"]]),
            "utterance_ids": torch.tensor([hash(str(u)) % (2**31) for u in split_data["utterance_ids"]]),
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
    parser = argparse.ArgumentParser(description="ThuanPhongNhi: Fine-tune RoBERTa")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["meld", "iemocap", "dailydialog"])
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to dataset. Auto-detected if not specified.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
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
    logger.info("Loading RoBERTa-base...")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaEmotionClassifier(num_classes=num_classes).to(args.device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {params:,} trainable params")

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
    logger.info(f"  Training: {args.epochs} epochs, {total_steps} steps")
    logger.info(f"{'='*60}\n")

    best_wf1 = 0.0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_wf1 = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, args.device,
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
            }, output_dir / f"best_roberta_{args.dataset}.pt")
            logger.info(f"  >> Saved best model! WF1={dev_wf1:.4f}")

    # 6. Test evaluation
    ckpt = torch.load(output_dir / f"best_roberta_{args.dataset}.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_wf1, test_report = evaluate(model, test_loader, args.device, emotions)

    logger.info(f"\n{'='*60}")
    logger.info(f"  Fine-tuned RoBERTa — {args.dataset.upper()}")
    logger.info(f"{'='*60}")
    logger.info(f"\n{test_report}")
    logger.info(f"  Test WF1 = {test_wf1:.4f}")
    logger.info(f"{'='*60}")

    # 7. Extract features from fine-tuned model
    logger.info(f"\nExtracting features from fine-tuned model...")
    features = extract_all_utterances(
        model, tokenizer, args.dataset, args.data_dir, args.device,
    )

    feat_path = output_dir / f"{args.dataset}_text_roberta_finetuned.pt"
    torch.save(features, str(feat_path))
    size_mb = feat_path.stat().st_size / 1e6
    logger.info(f"  Saved: {feat_path} ({size_mb:.1f} MB)")

    logger.info(f"\n{'='*60}")
    logger.info(f"  DONE! Download '{feat_path.name}' and place in data/features/")
    logger.info(f"  Then run: python scripts/train_multi_dataset.py --dataset {args.dataset}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
