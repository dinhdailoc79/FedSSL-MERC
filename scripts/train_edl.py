"""
Evidential Training Script
============================
Train EvidentialDialogueRNN on MELD, comparing:
1. EDL centralized (100% labels) vs softmax baseline
2. EDL + ECR semi-supervised vs FixMatch

Usage:
    # Centralized EDL (compare vs softmax baseline WF1=0.5442)
    python scripts/train_edl.py --label_ratio 1.0

    # Semi-supervised EDL + ECR (compare vs FixMatch)
    python scripts/train_edl.py --label_ratio 0.1 --lambda_u 1.0
    python scripts/train_edl.py --label_ratio 0.2 --lambda_u 1.0
"""

import sys
import os
import time
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score, classification_report

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.datasets.meld import MELDDataset, MELD_EMOTIONS
from scripts.train_centralized import DialogueDataset, collate_dialogues
from models.evidential.evidential_dialogue_rnn import EvidentialDialogueRNN
from models.evidential.losses import FedEvidenceLoss
from semi_supervised.augmentation import StrongAugmentation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def split_labeled_unlabeled(dataset, label_ratio, seed=42):
    """Split dataset into labeled and unlabeled subsets."""
    n = len(dataset)
    n_labeled = max(1, int(n * label_ratio))
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    labeled_indices = indices[:n_labeled].tolist()
    unlabeled_indices = indices[n_labeled:].tolist()
    labeled_ds = Subset(dataset, labeled_indices)
    unlabeled_ds = Subset(dataset, unlabeled_indices) if unlabeled_indices else None
    return labeled_ds, unlabeled_ds


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, emotion_names=None):
    """Evaluate EvidentialDialogueRNN."""
    model.eval()
    all_preds, all_labels, all_uncerts = [], [], []
    total_loss = 0
    total_count = 0

    for batch in loader:
        features = batch["features"].to(device)
        speaker_ids = batch["speaker_ids"].to(device)
        labels = batch["labels"].to(device)

        out = model(features, speaker_ids)
        mask = labels != -1
        alpha_flat = out["alpha"][mask]
        labels_flat = labels[mask]
        uncert_flat = out["uncertainty"][mask]

        loss, _ = loss_fn.sup_loss(alpha_flat, labels_flat)
        total_loss += loss.item() * labels_flat.size(0)
        total_count += labels_flat.size(0)

        preds = out["belief"][mask].argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels_flat.cpu().numpy())
        all_uncerts.extend(uncert_flat.cpu().numpy())

    avg_loss = total_loss / max(total_count, 1)
    wf1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    mean_uncert = np.mean(all_uncerts)
    report = classification_report(
        all_labels, all_preds,
        target_names=emotion_names, digits=4, zero_division=0,
    ) if emotion_names else ""
    return avg_loss, wf1, mean_uncert, report


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evidential Deep Learning training on MELD")
    parser.add_argument("--data_dir", type=str, default="data/raw/MELD")
    parser.add_argument("--feature_cache", type=str, default="data/features/meld_text_roberta.pt")
    # SSL
    parser.add_argument("--label_ratio", type=float, default=1.0)
    parser.add_argument("--lambda_u", type=float, default=1.0)
    # Model
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--text_dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.3)
    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--annealing_epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    is_ssl = args.label_ratio < 1.0
    mode = f"EDL+ECR (label_ratio={args.label_ratio})" if is_ssl else "EDL Centralized"

    logger.info(f"\n{'='*60}")
    logger.info(f"  Evidential Training: {mode}")
    logger.info(f"{'='*60}")
    logger.info(f"Config: {vars(args)}")

    # -------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------
    logger.info("\nLoading MELD dataset...")
    meld = MELDDataset(data_dir=args.data_dir)
    train_dialogues = meld.get_dialogues("train")
    dev_dialogues = meld.get_dialogues("dev")
    test_dialogues = meld.get_dialogues("test")

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

    full_train = DialogueDataset(train_dialogues, feature_caches.get("train", {}), args.text_dim)
    dev_ds = DialogueDataset(dev_dialogues, feature_caches.get("dev", {}), args.text_dim)
    test_ds = DialogueDataset(test_dialogues, feature_caches.get("test", {}), args.text_dim)

    # Split labeled/unlabeled
    labeled_ds, unlabeled_ds = split_labeled_unlabeled(full_train, args.label_ratio, args.seed)
    n_labeled = len(labeled_ds)
    n_unlabeled = len(unlabeled_ds) if unlabeled_ds else 0

    logger.info(f"\nData: {len(full_train)} total, {n_labeled} labeled, {n_unlabeled} unlabeled")

    labeled_loader = DataLoader(labeled_ds, batch_size=args.batch_size, shuffle=True,
                                collate_fn=collate_dialogues, num_workers=0)
    unlabeled_loader = DataLoader(unlabeled_ds, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=collate_dialogues, num_workers=0) if unlabeled_ds else None
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_dialogues, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_dialogues, num_workers=0)

    # -------------------------------------------------------
    # 2. Model + Loss
    # -------------------------------------------------------
    class_weights = torch.from_numpy(
        meld.get_emotion_weights("train").astype(np.float32)
    ).to(args.device)

    model = EvidentialDialogueRNN(
        input_dim=args.text_dim, hidden_dim=args.hidden_dim,
        num_classes=len(MELD_EMOTIONS), num_speakers=10,
        dropout=args.dropout, use_attention=True,
    ).to(args.device)

    loss_fn = FedEvidenceLoss(
        num_classes=len(MELD_EMOTIONS),
        annealing_epochs=args.annealing_epochs,
        lambda_u=args.lambda_u,
        lambda_u_rampup_epochs=20,
        class_weights=class_weights,
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    # Strong augmentation for ECR
    strong_aug = StrongAugmentation(noise_std=0.05, dropout_p=0.25)

    params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {params:,}")

    # -------------------------------------------------------
    # 3. Training loop
    # -------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info(f"  Training: {args.epochs} epochs")
    logger.info(f"{'='*60}\n")

    best_dev_wf1 = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        strong_aug.train()
        loss_fn.set_epoch(epoch)

        stats_acc = {"sup": 0, "ecr": 0, "total": 0, "n_contrib": 0, "n_total_u": 0, "batches": 0}

        unlabeled_iter = iter(unlabeled_loader) if unlabeled_loader else None

        for labeled_batch in labeled_loader:
            # Move to device
            features_l = labeled_batch["features"].to(args.device)
            speakers_l = labeled_batch["speaker_ids"].to(args.device)
            labels_l = labeled_batch["labels"].to(args.device)

            # Forward labeled
            out_l = model(features_l, speakers_l)
            mask_l = labels_l != -1
            alpha_l = out_l["alpha"][mask_l]
            labels_flat = labels_l[mask_l]

            # SSL: ECR on unlabeled
            alpha_weak = alpha_strong = uncertainty_weak = unlabeled_mask = None
            if unlabeled_loader and is_ssl:
                try:
                    unlabeled_batch = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(unlabeled_loader)
                    unlabeled_batch = next(unlabeled_iter)

                features_u = unlabeled_batch["features"].to(args.device)
                speakers_u = unlabeled_batch["speaker_ids"].to(args.device)
                labels_u = unlabeled_batch["labels"].to(args.device)

                # Weak view = original features (no augmentation)
                model.eval()
                with torch.no_grad():
                    out_weak = model(features_u, speakers_u)
                model.train()

                # Strong view = augmented features
                features_strong = strong_aug(features_u)
                out_strong = model(features_strong, speakers_u)

                # Flatten for ECR
                u_mask = labels_u != -1  # padding mask (unlabeled data still has structure labels)
                alpha_weak = out_weak["alpha"][u_mask]
                alpha_strong = out_strong["alpha"][u_mask]
                uncertainty_weak = out_weak["uncertainty"][u_mask]

            # Combined loss
            loss, batch_stats = loss_fn(
                alpha_l, labels_flat, label_mask=None,
                alpha_weak=alpha_weak, alpha_strong=alpha_strong,
                uncertainty_weak=uncertainty_weak,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            stats_acc["sup"] += batch_stats["loss_supervised"]
            stats_acc["ecr"] += batch_stats.get("loss_ecr", 0)
            stats_acc["total"] += batch_stats["loss_total"]
            stats_acc["n_contrib"] += batch_stats.get("ecr_n_contributing", 0)
            stats_acc["n_total_u"] += batch_stats.get("ecr_n_total", 0)
            stats_acc["batches"] += 1

        n = max(stats_acc["batches"], 1)
        avg_sup = stats_acc["sup"] / n
        avg_ecr = stats_acc["ecr"] / n
        avg_total = stats_acc["total"] / n

        # Evaluate
        dev_loss, dev_wf1, dev_uncert, _ = evaluate(model, dev_loader, loss_fn, args.device)
        scheduler.step(dev_wf1)
        elapsed = time.time() - start

        if is_ssl:
            contrib_ratio = stats_acc["n_contrib"] / max(stats_acc["n_total_u"], 1) * 100
            logger.info(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"Sup: {avg_sup:.4f} ECR: {avg_ecr:.4f} | "
                f"ECR contrib: {contrib_ratio:.0f}% | "
                f"Dev WF1: {dev_wf1:.4f} u={dev_uncert:.3f} | {elapsed:.1f}s"
            )
        else:
            logger.info(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"Loss: {avg_sup:.4f} | "
                f"Dev WF1: {dev_wf1:.4f} u={dev_uncert:.3f} | {elapsed:.1f}s"
            )

        if dev_wf1 > best_dev_wf1:
            best_dev_wf1 = dev_wf1
            patience_counter = 0
            tag = f"edl_{int(args.label_ratio*100)}pct"
            ckpt_path = Path(args.save_dir) / f"best_{tag}.pt"
            ckpt_path.parent.mkdir(exist_ok=True)
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "dev_wf1": dev_wf1}, ckpt_path)
            logger.info(f"  >> New best! Dev WF1={dev_wf1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"  Early stopping at epoch {epoch}")
                break

    # -------------------------------------------------------
    # 4. Final test evaluation
    # -------------------------------------------------------
    tag = f"edl_{int(args.label_ratio*100)}pct"
    ckpt_path = Path(args.save_dir) / f"best_{tag}.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"\nLoaded best model from epoch {ckpt['epoch']}")

    logger.info(f"\n{'='*60}")
    logger.info(f"  Final Test Evaluation ({mode})")
    logger.info(f"{'='*60}")

    test_loss, test_wf1, test_uncert, test_report = evaluate(
        model, test_loader, loss_fn, args.device, MELD_EMOTIONS
    )

    logger.info(f"\nTest WF1: {test_wf1:.4f} | Mean uncertainty: {test_uncert:.4f}")
    logger.info(f"\n{test_report}")

    logger.info(f"\n{'='*60}")
    logger.info(f"  COMPARISON")
    logger.info(f"{'='*60}")
    logger.info(f"  Softmax Centralized baseline:  WF1 = 0.5442")
    if is_ssl:
        logger.info(f"  FixMatch SSL {args.label_ratio*100:.0f}%:          WF1 = {'0.2922' if args.label_ratio <= 0.1 else '0.4141'}")
        logger.info(f"  EDL+ECR {args.label_ratio*100:.0f}% (ours):       WF1 = {test_wf1:.4f}")
    else:
        logger.info(f"  EDL Centralized (ours):        WF1 = {test_wf1:.4f}")
    logger.info(f"  Mean epistemic uncertainty:    u = {test_uncert:.4f}")
    logger.info(f"{'='*60}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
