"""
Centralized Baseline Training
===============================
Train DialogueRNN on MELD dataset (centralized, no FL/SSL).
This establishes the upper-bound baseline for comparison.

Usage:
    python scripts/train_centralized.py
    python scripts/train_centralized.py --epochs 30 --device cuda
    python scripts/train_centralized.py --device cpu --epochs 5  # Quick test
"""

import sys
import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, classification_report

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.datasets.meld import MELDDataset, MELD_EMOTIONS
from models.erc.dialogue_rnn import DialogueRNN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# PyTorch Dataset wrapper
# ============================================================

class DialogueDataset(Dataset):
    """Wraps MELD dialogues into a PyTorch Dataset with pre-extracted features."""

    def __init__(self, dialogues, feature_cache: dict = None, text_dim: int = 768):
        """
        Args:
            dialogues: List of Dialogue objects from MELDDataset
            feature_cache: Dict mapping utterance_id -> feature vector (numpy)
            text_dim: Dimension of text features (768 for RoBERTa)
        """
        self.dialogues = dialogues
        self.text_dim = text_dim
        self.feature_cache = feature_cache or {}

        # Build speaker vocabulary per dialogue
        self.speaker_maps = []
        for d in dialogues:
            speakers = list(set(u.speaker for u in d.utterances))
            smap = {s: i for i, s in enumerate(speakers)}
            self.speaker_maps.append(smap)

        # Log cache hit rate
        total = sum(len(d.utterances) for d in dialogues)
        hits = sum(
            1 for d in dialogues for u in d.utterances
            if f"{u.dialogue_id}_{u.utterance_id}" in self.feature_cache
        )
        if total > 0:
            logger.info(f"  Feature cache hit rate: {hits}/{total} ({hits/total*100:.1f}%)")

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, idx):
        dialogue = self.dialogues[idx]
        smap = self.speaker_maps[idx]

        texts = []
        speaker_ids = []
        labels = []

        for utt in dialogue.utterances:
            # Use cached features with composite key (dialogue_id + utterance_id)
            cache_key = f"{utt.dialogue_id}_{utt.utterance_id}"
            if cache_key in self.feature_cache:
                texts.append(self.feature_cache[cache_key])
            elif utt.text_features is not None:
                texts.append(utt.text_features)
            else:
                texts.append(np.zeros(self.text_dim, dtype=np.float32))

            speaker_ids.append(smap[utt.speaker])
            labels.append(utt.emotion_idx)

        return {
            "features": np.array(texts, dtype=np.float32),
            "speaker_ids": np.array(speaker_ids, dtype=np.int64),
            "labels": np.array(labels, dtype=np.int64),
            "length": len(dialogue.utterances),
        }


def collate_dialogues(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate dialogues into padded batches."""
    max_len = max(item["length"] for item in batch)
    batch_size = len(batch)
    feat_dim = batch[0]["features"].shape[1]

    features = torch.zeros(batch_size, max_len, feat_dim)
    speaker_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    labels = torch.full((batch_size, max_len), -1, dtype=torch.long)  # -1 = padding
    lengths = torch.zeros(batch_size, dtype=torch.long)

    for i, item in enumerate(batch):
        L = item["length"]
        features[i, :L] = torch.from_numpy(item["features"])
        speaker_ids[i, :L] = torch.from_numpy(item["speaker_ids"])
        labels[i, :L] = torch.from_numpy(item["labels"])
        lengths[i] = L

    return {
        "features": features,
        "speaker_ids": speaker_ids,
        "labels": labels,
        "lengths": lengths,
    }


# ============================================================
# Training Loop
# ============================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        features = batch["features"].to(device)
        speaker_ids = batch["speaker_ids"].to(device)
        labels = batch["labels"].to(device)
        lengths = batch["lengths"]

        # Forward
        logits = model(features, speaker_ids)  # (B, T, C)

        # Flatten and mask padding
        mask = labels != -1
        logits_flat = logits[mask]       # (N, C)
        labels_flat = labels[mask]       # (N,)

        loss = criterion(logits_flat, labels_flat)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * labels_flat.size(0)

        preds = logits_flat.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels_flat.cpu().numpy())

    avg_loss = total_loss / len(all_labels)
    wf1 = f1_score(all_labels, all_preds, average="weighted")
    return avg_loss, wf1


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float, str]:
    """Evaluate model on a dataset split."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        features = batch["features"].to(device)
        speaker_ids = batch["speaker_ids"].to(device)
        labels = batch["labels"].to(device)

        logits = model(features, speaker_ids)

        mask = labels != -1
        logits_flat = logits[mask]
        labels_flat = labels[mask]

        loss = criterion(logits_flat, labels_flat)
        total_loss += loss.item() * labels_flat.size(0)

        preds = logits_flat.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels_flat.cpu().numpy())

    avg_loss = total_loss / len(all_labels)
    wf1 = f1_score(all_labels, all_preds, average="weighted")
    report = classification_report(
        all_labels, all_preds,
        target_names=MELD_EMOTIONS,
        digits=4,
        zero_division=0,
    )
    return avg_loss, wf1, report


# ============================================================
# Main
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train DialogueRNN baseline on MELD")
    parser.add_argument("--data_dir", type=str, default="data/raw/MELD")
    parser.add_argument("--feature_cache", type=str, default="data/features/meld_text_roberta.pt",
                        help="Path to pre-extracted feature cache (.pt)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--text_dim", type=int, default=768)  # RoBERTa-base
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    logger.info(f"Device: {args.device}")
    logger.info(f"Config: {vars(args)}")

    # 1. Load data
    logger.info("Loading MELD dataset...")
    meld = MELDDataset(data_dir=args.data_dir)

    train_dialogues = meld.get_dialogues("train")
    dev_dialogues = meld.get_dialogues("dev")
    test_dialogues = meld.get_dialogues("test")

    logger.info(f"Train: {len(train_dialogues)} dialogues")
    logger.info(f"Dev:   {len(dev_dialogues)} dialogues")
    logger.info(f"Test:  {len(test_dialogues)} dialogues")

    # 1b. Load pre-extracted features
    feature_caches = {}
    feature_path = Path(args.feature_cache)
    if feature_path.exists():
        logger.info(f"Loading cached features from {feature_path}...")
        cached = torch.load(feature_path, weights_only=False)
        for split in ["train", "dev", "test"]:
            if split in cached:
                feats = cached[split]["features"].numpy()  # (N, 768)
                utt_ids = cached[split]["utterance_ids"]    # Tensor of int IDs
                # Build lookup: utterance_id -> feature vector
                cache = {}
                # MELD uses integer utterance_ids matching the Dialogue objects
                dia_ids = cached[split]["dialogue_ids"]
                texts = cached[split].get("texts", [])
                for i in range(len(feats)):
                    # Key = (dialogue_id, utterance_id) as string
                    key = f"{dia_ids[i].item()}_{utt_ids[i].item()}"
                    cache[key] = feats[i]
                feature_caches[split] = cache
                logger.info(f"  {split}: {len(cache)} features loaded")
        args.text_dim = feats.shape[1]  # Update dim from actual features
    else:
        logger.warning(f"No feature cache at {feature_path}, using random features!")

    # 2. Create datasets and dataloaders
    train_dataset = DialogueDataset(
        train_dialogues,
        feature_cache=feature_caches.get("train", {}),
        text_dim=args.text_dim,
    )
    dev_dataset = DialogueDataset(
        dev_dialogues,
        feature_cache=feature_caches.get("dev", {}),
        text_dim=args.text_dim,
    )
    test_dataset = DialogueDataset(
        test_dialogues,
        feature_cache=feature_caches.get("test", {}),
        text_dim=args.text_dim,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_dialogues, num_workers=0,
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_dialogues, num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_dialogues, num_workers=0,
    )

    # 3. Model
    class_weights = torch.from_numpy(
        meld.get_emotion_weights("train").astype(np.float32)
    ).to(args.device)

    model = DialogueRNN(
        input_dim=args.text_dim,
        hidden_dim=args.hidden_dim,
        num_classes=len(MELD_EMOTIONS),
        num_speakers=10,
        dropout=args.dropout,
        use_attention=True,
    ).to(args.device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # 4. Training setup
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5,
    )

    # 5. Training loop
    best_wf1 = 0.0
    patience_counter = 0
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"  Starting training: {args.epochs} epochs")
    logger.info(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        # Train
        train_loss, train_wf1 = train_epoch(
            model, train_loader, optimizer, criterion, args.device,
        )

        # Evaluate on dev
        dev_loss, dev_wf1, _ = evaluate(
            model, dev_loader, criterion, args.device,
        )

        elapsed = time.time() - start_time
        scheduler.step(dev_wf1)

        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} WF1: {train_wf1:.4f} | "
            f"Dev Loss: {dev_loss:.4f} WF1: {dev_wf1:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Save best model
        if dev_wf1 > best_wf1:
            best_wf1 = dev_wf1
            patience_counter = 0
            ckpt_path = save_dir / "best_dialoguernn_meld.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "dev_wf1": dev_wf1,
                "config": vars(args),
            }, ckpt_path)
            logger.info(f"  >> New best! WF1={dev_wf1:.4f}, saved to {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    # 6. Final evaluation on test set
    logger.info(f"\n{'='*60}")
    logger.info(f"  Final Evaluation on Test Set")
    logger.info(f"{'='*60}")

    # Load best model
    ckpt = torch.load(save_dir / "best_dialoguernn_meld.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_wf1, test_report = evaluate(
        model, test_loader, criterion, args.device,
    )

    logger.info(f"\nTest Loss: {test_loss:.4f}")
    logger.info(f"Test Weighted F1: {test_wf1:.4f}")
    logger.info(f"\nClassification Report:\n{test_report}")

    logger.info(f"\nBest Dev WF1: {best_wf1:.4f}")
    logger.info(f"Test WF1:     {test_wf1:.4f}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
