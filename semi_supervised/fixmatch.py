"""
FixMatch for Emotion Recognition in Conversations
====================================================
Sohn et al., "FixMatch: Simplifying Semi-Supervised Learning
with Consistency Regularization and Pseudo-Labeling", NeurIPS 2020.

Core idea:
1. For unlabeled data, apply WEAK augmentation -> get model prediction
2. If prediction confidence > threshold (e.g., 0.95), use it as pseudo-label
3. Apply STRONG augmentation to same data -> train with pseudo-label
4. Combine supervised loss (labeled) + unsupervised loss (pseudo-labeled)

Adapted for dialogue-level ERC with DialogueRNN.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from semi_supervised.augmentation import WeakAugmentation, StrongAugmentation

logger = logging.getLogger(__name__)


class FixMatchLoss(nn.Module):
    """
    FixMatch combined loss = supervised_loss + lambda_u * unsupervised_loss

    For pre-extracted features, weak augmentation = no augmentation (or tiny noise),
    strong augmentation = noise + dropout + cutoff.

    Supports curriculum threshold: starts lower and ramps up to target threshold.

    Args:
        threshold: Target confidence threshold for pseudo-labels
        lambda_u: Weight for unsupervised loss
        temperature: Sharpening temperature for pseudo-label distribution
        num_classes: Number of emotion classes
        warmup_epochs: Number of epochs to ramp threshold from threshold_min to threshold
        threshold_min: Starting threshold (lower = more pseudo-labels early on)
    """

    def __init__(
        self,
        threshold: float = 0.95,
        lambda_u: float = 1.0,
        temperature: float = 0.5,
        num_classes: int = 7,
        warmup_epochs: int = 10,
        threshold_min: float = 0.7,
    ):
        super().__init__()
        self.threshold_target = threshold
        self.threshold_min = threshold_min
        self.lambda_u = lambda_u
        self.temperature = temperature
        self.num_classes = num_classes
        self.warmup_epochs = warmup_epochs

        # Current dynamic threshold
        self.current_threshold = threshold_min

        self.strong_aug = StrongAugmentation(noise_std=0.05, dropout_p=0.25)

    def update_threshold(self, epoch: int):
        """Curriculum: ramp threshold from min to target over warmup_epochs."""
        if epoch >= self.warmup_epochs:
            self.current_threshold = self.threshold_target
        else:
            progress = epoch / self.warmup_epochs
            self.current_threshold = (
                self.threshold_min + progress * (self.threshold_target - self.threshold_min)
            )

    def forward(
        self,
        model: nn.Module,
        labeled_batch: Dict[str, torch.Tensor],
        unlabeled_batch: Optional[Dict[str, torch.Tensor]],
        criterion: nn.Module,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute FixMatch loss.

        Args:
            model: DialogueRNN model
            labeled_batch: Dict with 'features', 'speaker_ids', 'labels', 'lengths'
            unlabeled_batch: Dict with same keys
            criterion: Supervised loss function (CrossEntropyLoss)

        Returns:
            total_loss, stats_dict
        """
        # ============================
        # 1. Supervised loss (labeled data)
        # ============================
        features_l = labeled_batch["features"]
        speakers_l = labeled_batch["speaker_ids"]
        labels_l = labeled_batch["labels"]

        logits_l = model(features_l, speakers_l)

        mask_l = labels_l != -1
        logits_flat_l = logits_l[mask_l]
        labels_flat_l = labels_l[mask_l]

        loss_supervised = criterion(logits_flat_l, labels_flat_l)

        stats = {
            "loss_supervised": loss_supervised.item(),
            "loss_unsupervised": 0.0,
            "loss_total": loss_supervised.item(),
            "pseudo_label_count": 0,
            "pseudo_label_total": 0,
            "mask_ratio": 0.0,
            "threshold": self.current_threshold,
        }

        # If no unlabeled data, return supervised loss only
        if unlabeled_batch is None or unlabeled_batch["features"].size(0) == 0:
            return loss_supervised, stats

        # ============================
        # 2. Unsupervised loss (unlabeled data with pseudo-labels)
        # ============================
        features_u = unlabeled_batch["features"]
        speakers_u = unlabeled_batch["speaker_ids"]
        labels_u = unlabeled_batch["labels"]  # -1 for padding

        # 2a. Generate pseudo-labels from ORIGINAL features (no augmentation)
        # Use eval mode for stable predictions
        model.eval()
        with torch.no_grad():
            logits_clean = model(features_u, speakers_u)
            # Temperature sharpening to produce more confident predictions
            probs = F.softmax(logits_clean / self.temperature, dim=-1)
            max_probs, pseudo_labels = probs.max(dim=-1)
        model.train()

        # 2b. Strong augmentation -> train with pseudo-labels
        features_strong = self.strong_aug(features_u)
        logits_strong = model(features_strong, speakers_u)

        # 2c. Create mask: only use high-confidence pseudo-labels
        padding_mask = labels_u != -1
        confidence_mask = max_probs >= self.current_threshold
        combined_mask = padding_mask & confidence_mask

        num_above_threshold = combined_mask.sum().item()
        num_total = padding_mask.sum().item()

        stats["pseudo_label_count"] = int(num_above_threshold)
        stats["pseudo_label_total"] = int(num_total)
        stats["mask_ratio"] = num_above_threshold / max(num_total, 1)

        if num_above_threshold > 0:
            logits_masked = logits_strong[combined_mask]
            pseudo_masked = pseudo_labels[combined_mask]

            loss_unsupervised = F.cross_entropy(logits_masked, pseudo_masked)
            stats["loss_unsupervised"] = loss_unsupervised.item()
        else:
            loss_unsupervised = torch.tensor(0.0, device=features_u.device)

        # ============================
        # 3. Total loss
        # ============================
        total_loss = loss_supervised + self.lambda_u * loss_unsupervised
        stats["loss_total"] = total_loss.item()

        return total_loss, stats


class FixMatchTrainer:
    """
    High-level trainer for FixMatch semi-supervised learning.

    Manages the training loop with separate labeled/unlabeled data streams.
    """

    def __init__(
        self,
        model: nn.Module,
        labeled_loader,
        unlabeled_loader,
        test_loader,
        criterion: nn.Module,
        optimizer,
        scheduler=None,
        device: str = "cuda",
        threshold: float = 0.95,
        lambda_u: float = 1.0,
        num_classes: int = 7,
    ):
        self.model = model
        self.labeled_loader = labeled_loader
        self.unlabeled_loader = unlabeled_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.fixmatch = FixMatchLoss(
            threshold=threshold,
            lambda_u=lambda_u,
            num_classes=num_classes,
        )

    def train_epoch(self) -> Dict:
        """Train one epoch with FixMatch."""
        self.model.train()
        self.fixmatch.train()

        total_stats = {
            "loss_supervised": 0,
            "loss_unsupervised": 0,
            "loss_total": 0,
            "pseudo_label_count": 0,
            "pseudo_label_total": 0,
            "num_batches": 0,
        }

        # Create infinite iterator for unlabeled data
        # (unlabeled is typically much larger than labeled)
        unlabeled_iter = iter(self.unlabeled_loader)

        for labeled_batch in self.labeled_loader:
            # Move labeled data to device
            labeled_batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in labeled_batch.items()
            }

            # Get unlabeled batch (cycle if exhausted)
            try:
                unlabeled_batch = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(self.unlabeled_loader)
                unlabeled_batch = next(unlabeled_iter)

            unlabeled_batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in unlabeled_batch.items()
            }

            # FixMatch forward
            loss, stats = self.fixmatch(
                self.model, labeled_batch, unlabeled_batch, self.criterion
            )

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            # Accumulate stats
            for key in ["loss_supervised", "loss_unsupervised", "loss_total"]:
                total_stats[key] += stats[key]
            total_stats["pseudo_label_count"] += stats["pseudo_label_count"]
            total_stats["pseudo_label_total"] += stats["pseudo_label_total"]
            total_stats["num_batches"] += 1

        # Average losses
        n = max(total_stats["num_batches"], 1)
        for key in ["loss_supervised", "loss_unsupervised", "loss_total"]:
            total_stats[key] /= n

        total_stats["mask_ratio"] = (
            total_stats["pseudo_label_count"] /
            max(total_stats["pseudo_label_total"], 1)
        )

        return total_stats
