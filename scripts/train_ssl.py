"""
Semi-Supervised Training Script (Centralized FixMatch)
=======================================================
Train DialogueRNN with FixMatch on MELD, comparing:
- Fully supervised (label_ratio=1.0)
- Semi-supervised at different label ratios (5%, 10%, 20%)

Usage:
    python scripts/train_ssl.py --label_ratio 0.1 --threshold 0.95
    python scripts/train_ssl.py --label_ratio 0.05 --lambda_u 0.5
    python scripts/train_ssl.py --label_ratio 1.0   # Fully supervised (no SSL)
"""

import sys
import os
import time
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score, classification_report

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.datasets.meld import MELDDataset, MELD_EMOTIONS
from models.erc.dialogue_rnn import DialogueRNN
from scripts.train_centralized import DialogueDataset, collate_dialogues
from semi_supervised.fixmatch import FixMatchLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def split_labeled_unlabeled(dataset, label_ratio, seed=42):
    """
    Split dataset into labeled and unlabeled subsets.

    Args:
        dataset: DialogueDataset
        label_ratio: Fraction of dialogues to keep as labeled (e.g., 0.1 = 10%)
        seed: Random seed for reproducibility

    Returns:
        labeled_dataset, unlabeled_dataset
    """
    n = len(dataset)
    n_labeled = max(1, int(n * label_ratio))

    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)

    labeled_indices = indices[:n_labeled].tolist()
    unlabeled_indices = indices[n_labeled:].tolist()

    labeled_dataset = Subset(dataset, labeled_indices)
    unlabeled_dataset = Subset(dataset, unlabeled_indices) if unlabeled_indices else None

    return labeled_dataset, unlabeled_dataset


@torch.no_grad()
def evaluate(model, loader, criterion, device, emotion_names=None):
    """Evaluate model on a DataLoader."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in loader:
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

    avg_loss = total_loss / max(len(all_labels), 1)
    wf1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    report = classification_report(
        all_labels, all_preds,
        target_names=emotion_names,
        digits=4, zero_division=0,
    ) if emotion_names else ""
    return avg_loss, wf1, report


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Semi-supervised FixMatch training on MELD")
    # Data
    parser.add_argument("--data_dir", type=str, default="data/raw/MELD")
    parser.add_argument("--feature_cache", type=str, default="data/features/meld_text_roberta.pt")
    # SSL settings
    parser.add_argument("--label_ratio", type=float, default=0.1,
                        help="Fraction of labeled data (0.05, 0.1, 0.2, 1.0)")
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="FixMatch confidence threshold")
    parser.add_argument("--lambda_u", type=float, default=1.0,
                        help="Weight for unsupervised loss")
    # Model
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--text_dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.3)
    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    is_ssl = args.label_ratio < 1.0
    mode = f"FixMatch (label_ratio={args.label_ratio})" if is_ssl else "Fully Supervised"

    logger.info(f"\n{'='*60}")
    logger.info(f"  Semi-Supervised Training: {mode}")
    logger.info(f"{'='*60}")
    logger.info(f"Config: {vars(args)}")

    # -------------------------------------------------------
    # 1. Load data + features
    # -------------------------------------------------------
    logger.info("\nLoading MELD dataset...")
    meld = MELDDataset(data_dir=args.data_dir)
    train_dialogues = meld.get_dialogues("train")
    dev_dialogues = meld.get_dialogues("dev")
    test_dialogues = meld.get_dialogues("test")

    # Load cached features
    feature_caches = {}
    feature_path = Path(args.feature_cache)
    if feature_path.exists():
        logger.info(f"Loading cached features from {feature_path}...")
        cached = torch.load(feature_path, weights_only=False)
        for split in ["train", "dev", "test"]:
            if split in cached:
                feats = cached[split]["features"].numpy()
                dia_ids = cached[split]["dialogue_ids"]
                utt_ids = cached[split]["utterance_ids"]
                cache = {}
                for i in range(len(feats)):
                    key = f"{dia_ids[i].item()}_{utt_ids[i].item()}"
                    cache[key] = feats[i]
                feature_caches[split] = cache
        args.text_dim = feats.shape[1]
    else:
        logger.warning(f"No feature cache at {feature_path}!")

    # Create full datasets
    full_train_dataset = DialogueDataset(
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

    # -------------------------------------------------------
    # 2. Split labeled / unlabeled
    # -------------------------------------------------------
    labeled_dataset, unlabeled_dataset = split_labeled_unlabeled(
        full_train_dataset, args.label_ratio, seed=args.seed
    )

    n_labeled = len(labeled_dataset)
    n_unlabeled = len(unlabeled_dataset) if unlabeled_dataset else 0
    n_total = len(full_train_dataset)

    logger.info(f"\nData split:")
    logger.info(f"  Total train dialogues: {n_total}")
    logger.info(f"  Labeled:   {n_labeled} ({n_labeled/n_total*100:.1f}%)")
    logger.info(f"  Unlabeled: {n_unlabeled} ({n_unlabeled/n_total*100:.1f}%)")

    labeled_loader = DataLoader(
        labeled_dataset, batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_dialogues, num_workers=0,
    )
    unlabeled_loader = DataLoader(
        unlabeled_dataset, batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_dialogues, num_workers=0,
    ) if unlabeled_dataset else None
    dev_loader = DataLoader(
        dev_dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_dialogues, num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_dialogues, num_workers=0,
    )

    # -------------------------------------------------------
    # 3. Model
    # -------------------------------------------------------
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

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5,
    )

    # FixMatch loss
    fixmatch = FixMatchLoss(
        threshold=args.threshold,
        lambda_u=args.lambda_u,
        num_classes=len(MELD_EMOTIONS),
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"\nModel parameters: {total_params:,}")

    # -------------------------------------------------------
    # 4. Training loop
    # -------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info(f"  Starting training: {args.epochs} epochs")
    logger.info(f"{'='*60}\n")

    best_dev_wf1 = 0.0
    patience_counter = 0
    unlabeled_iter = None

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        model.train()
        fixmatch.train()

        # Update curriculum threshold
        fixmatch.update_threshold(epoch)

        epoch_stats = {
            "loss_sup": 0, "loss_unsup": 0, "loss_total": 0,
            "pseudo_count": 0, "pseudo_total": 0, "num_batches": 0,
        }

        # Reset unlabeled iterator
        if unlabeled_loader:
            unlabeled_iter = iter(unlabeled_loader)

        for labeled_batch in labeled_loader:
            labeled_batch = {
                k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                for k, v in labeled_batch.items()
            }

            # Get unlabeled batch
            unlabeled_batch = None
            if unlabeled_loader and is_ssl:
                try:
                    unlabeled_batch = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(unlabeled_loader)
                    unlabeled_batch = next(unlabeled_iter)
                unlabeled_batch = {
                    k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                    for k, v in unlabeled_batch.items()
                }

            # FixMatch forward
            loss, stats = fixmatch(model, labeled_batch, unlabeled_batch, criterion)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_stats["loss_sup"] += stats["loss_supervised"]
            epoch_stats["loss_unsup"] += stats["loss_unsupervised"]
            epoch_stats["loss_total"] += stats["loss_total"]
            epoch_stats["pseudo_count"] += stats["pseudo_label_count"]
            epoch_stats["pseudo_total"] += stats["pseudo_label_total"]
            epoch_stats["num_batches"] += 1

        n = max(epoch_stats["num_batches"], 1)
        avg_sup = epoch_stats["loss_sup"] / n
        avg_unsup = epoch_stats["loss_unsup"] / n
        avg_total = epoch_stats["loss_total"] / n
        mask_ratio = epoch_stats["pseudo_count"] / max(epoch_stats["pseudo_total"], 1)

        # Evaluate
        dev_loss, dev_wf1, _ = evaluate(model, dev_loader, criterion, args.device)
        scheduler.step(dev_wf1)

        elapsed = time.time() - start_time

        if is_ssl:
            logger.info(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"Sup: {avg_sup:.4f} Unsup: {avg_unsup:.4f} | "
                f"Pseudo: {epoch_stats['pseudo_count']}/{epoch_stats['pseudo_total']} "
                f"({mask_ratio*100:.0f}%) thr={fixmatch.current_threshold:.2f} | "
                f"Dev WF1: {dev_wf1:.4f} | {elapsed:.1f}s"
            )
        else:
            logger.info(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"Loss: {avg_sup:.4f} | Dev WF1: {dev_wf1:.4f} | {elapsed:.1f}s"
            )

        # Save best
        if dev_wf1 > best_dev_wf1:
            best_dev_wf1 = dev_wf1
            patience_counter = 0
            ckpt_name = f"best_ssl_{int(args.label_ratio*100)}pct.pt"
            ckpt_path = Path(args.save_dir) / ckpt_name
            ckpt_path.parent.mkdir(exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "dev_wf1": dev_wf1,
                "label_ratio": args.label_ratio,
            }, ckpt_path)
            logger.info(f"  >> New best! Dev WF1={dev_wf1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"  Early stopping at epoch {epoch}")
                break

    # -------------------------------------------------------
    # 5. Final evaluation on test set
    # -------------------------------------------------------
    # Load best model
    ckpt_name = f"best_ssl_{int(args.label_ratio*100)}pct.pt"
    ckpt_path = Path(args.save_dir) / ckpt_name
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"\nLoaded best model from epoch {ckpt['epoch']}")

    logger.info(f"\n{'='*60}")
    logger.info(f"  Final Test Evaluation ({mode})")
    logger.info(f"{'='*60}")

    test_loss, test_wf1, test_report = evaluate(
        model, test_loader, criterion, args.device, MELD_EMOTIONS
    )

    logger.info(f"\nTest Loss: {test_loss:.4f}")
    logger.info(f"Test Weighted F1: {test_wf1:.4f}")
    logger.info(f"\n{test_report}")

    logger.info(f"\n{'='*60}")
    logger.info(f"  SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"  Label ratio:     {args.label_ratio*100:.0f}% ({n_labeled}/{n_total} dialogues)")
    logger.info(f"  Best Dev WF1:    {best_dev_wf1:.4f}")
    logger.info(f"  Test WF1:        {test_wf1:.4f}")
    if is_ssl:
        logger.info(f"  FixMatch threshold: {args.threshold}")
        logger.info(f"  Lambda_u:        {args.lambda_u}")
    logger.info(f"{'='*60}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
