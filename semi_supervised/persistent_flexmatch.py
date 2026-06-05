"""
Persistent FlexMatch for Federated Learning
=============================================
Addresses the reviewer's concern that the original FlexMatch comparison
is "implementation-dependent" due to buffer resetting.

Implements two faithful federated adaptations:

1. PersistentFlexMatch: Each client maintains its own persistent
   class_counts buffer across rounds. Buffers are stored in a
   per-client dictionary on the server and restored each round.

2. ServerAggFlexMatch: The server aggregates class_counts from all
   clients and broadcasts global class-wise thresholds each round.
   This is the "federated threshold aggregation" variant suggested
   by the reviewer.

Both variants use the same CE + pseudo-labeling pipeline as the
original FlexMatch, differing only in threshold management.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from semi_supervised.augmentation import StrongAugmentation

logger = logging.getLogger(__name__)


class PersistentFlexMatchLoss(nn.Module):
    """
    FlexMatch with per-client persistent class-status buffers.

    Unlike the original FlexMatch where class_counts are shared and
    implicitly reset, this version:
    - Maintains a separate class_counts vector per client
    - Persists counts across federated rounds
    - Buffers are stored externally (not in model state_dict)
    """

    def __init__(
        self,
        threshold: float = 0.95,
        lambda_u: float = 1.0,
        temperature: float = 0.5,
        num_classes: int = 7,
        threshold_min: float = 0.5,
        num_clients: int = 5,
    ):
        super().__init__()
        self.threshold_target = threshold
        self.threshold_min = threshold_min
        self.lambda_u = lambda_u
        self.temperature = temperature
        self.num_classes = num_classes
        self.num_clients = num_clients

        # Per-client persistent buffers (stored on CPU, not part of model)
        self.client_class_counts: List[torch.Tensor] = [
            torch.zeros(num_classes) for _ in range(num_clients)
        ]

        self.strong_aug = StrongAugmentation(noise_std=0.05, dropout_p=0.25)

    def set_client(self, client_idx: int):
        """Set the active client index for this training step."""
        self.active_client = client_idx

    def get_class_thresholds(self, device: torch.device) -> torch.Tensor:
        """Compute dynamic thresholds from the active client's persistent counts."""
        counts = self.client_class_counts[self.active_client].to(device)
        max_count = counts.max().item()
        if max_count > 0:
            beta = counts / max_count
            thresholds = beta * self.threshold_target
            thresholds = torch.clamp(thresholds, min=self.threshold_min)
        else:
            thresholds = torch.full(
                (self.num_classes,), self.threshold_target, device=device
            )
        return thresholds

    def forward(
        self,
        model: nn.Module,
        labeled_batch: Dict[str, torch.Tensor],
        unlabeled_batch: Optional[Dict[str, torch.Tensor]],
        criterion: nn.Module,
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute Persistent FlexMatch loss."""
        # === Supervised loss ===
        features_l = labeled_batch["features"]
        speakers_l = labeled_batch["speaker_ids"]
        labels_l = labeled_batch["labels"]
        mask_l = labels_l != -1

        logits_l = model(features_l, speakers_l)
        loss_sup = criterion(logits_l[mask_l], labels_l[mask_l])

        stats = {
            "loss_supervised": loss_sup.item(),
            "loss_unsupervised": 0.0,
            "loss_total": loss_sup.item(),
            "pseudo_label_count": 0,
            "pseudo_label_total": 0,
            "mask_ratio": 0.0,
        }

        if unlabeled_batch is None or unlabeled_batch["features"].size(0) == 0:
            return loss_sup, stats

        # === Unsupervised loss ===
        features_u = unlabeled_batch["features"]
        speakers_u = unlabeled_batch["speaker_ids"]
        labels_u = unlabeled_batch["labels"]

        # Weak view predictions
        model.eval()
        with torch.no_grad():
            logits_clean = model(features_u, speakers_u)
            probs = F.softmax(logits_clean / self.temperature, dim=-1)
            max_probs, pseudo_labels = probs.max(dim=-1)

            padding_mask = labels_u != -1
            above_base = (max_probs >= self.threshold_target) & padding_mask

            # Update persistent per-client class counts
            for c in range(self.num_classes):
                count_c = (pseudo_labels[above_base] == c).sum().item()
                self.client_class_counts[self.active_client][c] += count_c
        model.train()

        # Dynamic per-client thresholds
        class_thresholds = self.get_class_thresholds(features_u.device)
        dynamic_thresholds = class_thresholds[pseudo_labels]
        confidence_mask = max_probs >= dynamic_thresholds
        combined_mask = padding_mask & confidence_mask

        num_above = combined_mask.sum().item()
        num_total = padding_mask.sum().item()
        stats["pseudo_label_count"] = int(num_above)
        stats["pseudo_label_total"] = int(num_total)
        stats["mask_ratio"] = num_above / max(num_total, 1)

        # Strong view
        features_strong = self.strong_aug(features_u)
        logits_strong = model(features_strong, speakers_u)

        if num_above > 0:
            loss_unsup = F.cross_entropy(
                logits_strong[combined_mask], pseudo_labels[combined_mask]
            )
            stats["loss_unsupervised"] = loss_unsup.item()
        else:
            loss_unsup = torch.tensor(0.0, device=features_u.device)

        total = loss_sup + self.lambda_u * loss_unsup
        stats["loss_total"] = total.item()
        return total, stats


class ServerAggFlexMatchLoss(nn.Module):
    """
    FlexMatch with server-aggregated class-wise thresholds.

    After each round, clients upload their local class_counts to the
    server. The server averages them and broadcasts global thresholds.
    This is the "federated threshold aggregation" variant.
    """

    def __init__(
        self,
        threshold: float = 0.95,
        lambda_u: float = 1.0,
        temperature: float = 0.5,
        num_classes: int = 7,
        threshold_min: float = 0.5,
        num_clients: int = 5,
    ):
        super().__init__()
        self.threshold_target = threshold
        self.threshold_min = threshold_min
        self.lambda_u = lambda_u
        self.temperature = temperature
        self.num_classes = num_classes
        self.num_clients = num_clients

        # Per-client local counts (reset each round, accumulated during local training)
        self.local_class_counts: List[torch.Tensor] = [
            torch.zeros(num_classes) for _ in range(num_clients)
        ]
        # Server-aggregated global counts (persisted across rounds)
        self.global_class_counts = torch.zeros(num_classes)
        self.active_client = 0

        self.strong_aug = StrongAugmentation(noise_std=0.05, dropout_p=0.25)

    def set_client(self, client_idx: int):
        """Set active client and reset its local counts for this round."""
        self.active_client = client_idx
        self.local_class_counts[client_idx] = torch.zeros(self.num_classes)

    def server_aggregate(self):
        """
        Server aggregates all client class_counts into global thresholds.
        Call this AFTER all clients have finished local training in a round.
        """
        total = torch.zeros(self.num_classes)
        for counts in self.local_class_counts:
            total += counts
        self.global_class_counts += total

    def get_class_thresholds(self, device: torch.device) -> torch.Tensor:
        """Compute thresholds from server-aggregated global counts."""
        counts = self.global_class_counts.to(device)
        max_count = counts.max().item()
        if max_count > 0:
            beta = counts / max_count
            thresholds = beta * self.threshold_target
            thresholds = torch.clamp(thresholds, min=self.threshold_min)
        else:
            thresholds = torch.full(
                (self.num_classes,), self.threshold_target, device=device
            )
        return thresholds

    def forward(
        self,
        model: nn.Module,
        labeled_batch: Dict[str, torch.Tensor],
        unlabeled_batch: Optional[Dict[str, torch.Tensor]],
        criterion: nn.Module,
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute ServerAgg FlexMatch loss."""
        features_l = labeled_batch["features"]
        speakers_l = labeled_batch["speaker_ids"]
        labels_l = labeled_batch["labels"]
        mask_l = labels_l != -1

        logits_l = model(features_l, speakers_l)
        loss_sup = criterion(logits_l[mask_l], labels_l[mask_l])

        stats = {
            "loss_supervised": loss_sup.item(),
            "loss_unsupervised": 0.0,
            "loss_total": loss_sup.item(),
            "pseudo_label_count": 0,
            "pseudo_label_total": 0,
            "mask_ratio": 0.0,
        }

        if unlabeled_batch is None or unlabeled_batch["features"].size(0) == 0:
            return loss_sup, stats

        features_u = unlabeled_batch["features"]
        speakers_u = unlabeled_batch["speaker_ids"]
        labels_u = unlabeled_batch["labels"]

        model.eval()
        with torch.no_grad():
            logits_clean = model(features_u, speakers_u)
            probs = F.softmax(logits_clean / self.temperature, dim=-1)
            max_probs, pseudo_labels = probs.max(dim=-1)

            padding_mask = labels_u != -1
            above_base = (max_probs >= self.threshold_target) & padding_mask

            # Update LOCAL client counts (for later server aggregation)
            for c in range(self.num_classes):
                count_c = (pseudo_labels[above_base] == c).sum().item()
                self.local_class_counts[self.active_client][c] += count_c
        model.train()

        # Use SERVER-AGGREGATED global thresholds
        class_thresholds = self.get_class_thresholds(features_u.device)
        dynamic_thresholds = class_thresholds[pseudo_labels]
        confidence_mask = max_probs >= dynamic_thresholds
        combined_mask = padding_mask & confidence_mask

        num_above = combined_mask.sum().item()
        num_total = padding_mask.sum().item()
        stats["pseudo_label_count"] = int(num_above)
        stats["pseudo_label_total"] = int(num_total)
        stats["mask_ratio"] = num_above / max(num_total, 1)

        features_strong = self.strong_aug(features_u)
        logits_strong = model(features_strong, speakers_u)

        if num_above > 0:
            loss_unsup = F.cross_entropy(
                logits_strong[combined_mask], pseudo_labels[combined_mask]
            )
            stats["loss_unsupervised"] = loss_unsup.item()
        else:
            loss_unsup = torch.tensor(0.0, device=features_u.device)

        total = loss_sup + self.lambda_u * loss_unsup
        stats["loss_total"] = total.item()
        return total, stats
