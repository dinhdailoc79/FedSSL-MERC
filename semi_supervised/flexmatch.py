"""
FlexMatch for Emotion Recognition in Conversations
====================================================
Zhang et al., "FlexMatch: Boosting Semi-Supervised Learning with
Curriculum Pseudo-Labeling", NeurIPS 2021.

Core idea:
1. For unlabeled data, instead of using a fixed confidence threshold (FixMatch),
   FlexMatch uses class-specific dynamic thresholds.
2. The dynamic threshold for class c is tau_c = beta_c * tau, where tau is the
   base threshold, and beta_c is estimated by the model's learning status of class c.
3. The learning status beta_c is the normalized sum of pseudo-labels exceeding the
   base threshold for class c relative to the maximum class count.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from semi_supervised.augmentation import WeakAugmentation, StrongAugmentation

logger = logging.getLogger(__name__)


class FlexMatchLoss(nn.Module):
    """
    FlexMatch combined loss = supervised_loss + lambda_u * unsupervised_loss
    with Dynamic Curriculum Pseudo-Labeling.
    """

    def __init__(
        self,
        threshold: float = 0.95,
        lambda_u: float = 1.0,
        temperature: float = 0.5,
        num_classes: int = 7,
        threshold_min: float = 0.5,
    ):
        super().__init__()
        self.threshold_target = threshold
        self.threshold_min = threshold_min
        self.lambda_u = lambda_u
        self.temperature = temperature
        self.num_classes = num_classes

        # Track class-specific learning status
        # Registers count of high-confidence predictions (> base threshold) per class
        self.register_buffer("class_counts", torch.zeros(num_classes))
        
        self.strong_aug = StrongAugmentation(noise_std=0.05, dropout_p=0.25)

    def forward(
        self,
        model: nn.Module,
        labeled_batch: Dict[str, torch.Tensor],
        unlabeled_batch: Optional[Dict[str, torch.Tensor]],
        criterion: nn.Module,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute FlexMatch loss.
        """
        # ============================
        # 1. Supervised loss
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
        }

        # If no unlabeled data, return supervised only
        if unlabeled_batch is None or unlabeled_batch["features"].size(0) == 0:
            return loss_supervised, stats

        # ============================
        # 2. Unsupervised loss (FlexMatch dynamic thresholding)
        # ============================
        features_u = unlabeled_batch["features"]
        speakers_u = unlabeled_batch["speaker_ids"]
        labels_u = unlabeled_batch["labels"]  # -1 for padding

        # 2a. Generate predictions from clean features
        model.eval()
        with torch.no_grad():
            logits_clean = model(features_u, speakers_u)
            probs = F.softmax(logits_clean / self.temperature, dim=-1)
            max_probs, pseudo_labels = probs.max(dim=-1)
            
            # Find which pseudo-labels are above base target threshold
            padding_mask = labels_u != -1
            above_base = (max_probs >= self.threshold_target) & padding_mask
            
            # Update the learning status count (using CPU/GPU buffer operation)
            for c in range(self.num_classes):
                count_c = (pseudo_labels[above_base] == c).sum().item()
                self.class_counts[c] += count_c
        model.train()

        # 2b. Compute dynamic class-specific thresholds
        max_count = self.class_counts.max().item()
        if max_count > 0:
            beta = self.class_counts / max_count
            class_thresholds = beta * self.threshold_target
            # Clamp to min_threshold to avoid noise early on
            class_thresholds = torch.clamp(class_thresholds, min=self.threshold_min)
        else:
            class_thresholds = torch.full(
                (self.num_classes,), self.threshold_target, device=features_u.device
            )

        # Move thresholds to model device if needed
        class_thresholds = class_thresholds.to(features_u.device)

        # 2c. Strong augmentation
        features_strong = self.strong_aug(features_u)
        logits_strong = model(features_strong, speakers_u)

        # 2d. Apply dynamic thresholds
        # Lookup threshold for each argmax class
        dynamic_thresholds = class_thresholds[pseudo_labels]
        confidence_mask = max_probs >= dynamic_thresholds
        combined_mask = padding_mask & confidence_mask

        num_above_threshold = combined_mask.sum().item()
        num_total = padding_mask.sum().item()

        stats["pseudo_label_count"] = int(num_above_threshold)
        stats["pseudo_label_total"] = int(num_total)
        stats["mask_ratio"] = num_above_threshold / max(num_total, 1)

        # Log class thresholds for debugging/analysis
        for c in range(self.num_classes):
            stats[f"thresh_class_{c}"] = class_thresholds[c].item()

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
