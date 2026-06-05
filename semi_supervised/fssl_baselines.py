"""
Federated SSL Baselines
=========================
Implements specialized semi-supervised baselines for ERC:
1. Mean Teacher (Teacher-Student EMA consistency)
2. FedSwitch (Dynamic thresholding based on local/global consensus)
"""

import copy
import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from semi_supervised.augmentation import StrongAugmentation

logger = logging.getLogger(__name__)


class MeanTeacherLoss(nn.Module):
    """
    Mean Teacher SSL Baseline (Tarvainen & Valpola, NeurIPS 2017)
    
    Uses an Exponential Moving Average (EMA) teacher model to generate stable
    targets for the student model.
    """

    def __init__(
        self,
        ema_alpha: float = 0.99,
        lambda_u: float = 1.0,
        num_classes: int = 7,
    ):
        super().__init__()
        self.ema_alpha = ema_alpha
        self.lambda_u = lambda_u
        self.num_classes = num_classes
        self.teacher_model = None
        self.strong_aug = StrongAugmentation(noise_std=0.05, dropout_p=0.25)

    def update_teacher(self, student_model: nn.Module):
        """EMA update: theta_teacher = alpha * theta_teacher + (1 - alpha) * theta_student"""
        if self.teacher_model is None:
            self.teacher_model = copy.deepcopy(student_model)
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            return

        with torch.no_grad():
            for t_param, s_param in zip(self.teacher_model.parameters(), student_model.parameters()):
                t_param.data.mul_(self.ema_alpha).add_(s_param.data, alpha=1.0 - self.ema_alpha)

    def forward(
        self,
        student_model: nn.Module,
        labeled_batch: Dict[str, torch.Tensor],
        unlabeled_batch: Optional[Dict[str, torch.Tensor]],
        criterion: nn.Module,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute Mean Teacher loss.
        """
        # Update teacher weights first
        self.update_teacher(student_model)
        device = next(student_model.parameters()).device
        self.teacher_model.to(device)
        self.teacher_model.eval()

        # 1. Supervised loss
        features_l = labeled_batch["features"]
        speakers_l = labeled_batch["speaker_ids"]
        labels_l = labeled_batch["labels"]

        logits_l = student_model(features_l, speakers_l)
        mask_l = labels_l != -1
        loss_supervised = criterion(logits_l[mask_l], labels_l[mask_l])

        stats = {
            "loss_supervised": loss_supervised.item(),
            "loss_unsupervised": 0.0,
            "loss_total": loss_supervised.item(),
        }

        if unlabeled_batch is None or unlabeled_batch["features"].size(0) == 0:
            return loss_supervised, stats

        # 2. Unsupervised consistency loss (MSE between Student strong view and Teacher weak view)
        features_u = unlabeled_batch["features"]
        speakers_u = unlabeled_batch["speaker_ids"]
        labels_u = unlabeled_batch["labels"]
        padding_mask = labels_u != -1

        # Teacher weak view
        with torch.no_grad():
            teacher_logits = self.teacher_model(features_u, speakers_u)
            teacher_probs = F.softmax(teacher_logits, dim=-1)

        # Student strong view
        features_strong = self.strong_aug(features_u)
        student_logits = student_model(features_strong, speakers_u)
        student_probs = F.softmax(student_logits, dim=-1)

        # Consistency loss (mean squared error over active tokens)
        if padding_mask.sum() > 0:
            student_masked = student_probs[padding_mask]
            teacher_masked = teacher_probs[padding_mask]
            loss_unsupervised = F.mse_loss(student_masked, teacher_masked)
            stats["loss_unsupervised"] = loss_unsupervised.item()
        else:
            loss_unsupervised = torch.tensor(0.0, device=features_u.device)

        total_loss = loss_supervised + self.lambda_u * loss_unsupervised
        stats["loss_total"] = total_loss.item()

        return total_loss, stats


class FedSwitchLoss(nn.Module):
    """
    FedSwitch Dynamic Thresholding baseline.
    
    Dynamically switches/adapts confidence thresholds based on local prediction
    distributions or epoch progression, defending against confirmation bias.
    """

    def __init__(
        self,
        threshold_init: float = 0.95,
        lambda_u: float = 1.0,
        num_classes: int = 7,
    ):
        super().__init__()
        self.threshold = threshold_init
        self.lambda_u = lambda_u
        self.num_classes = num_classes
        self.strong_aug = StrongAugmentation(noise_std=0.05, dropout_p=0.25)

    def forward(
        self,
        model: nn.Module,
        labeled_batch: Dict[str, torch.Tensor],
        unlabeled_batch: Optional[Dict[str, torch.Tensor]],
        criterion: nn.Module,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute FedSwitch adaptive threshold loss.
        """
        # 1. Supervised loss
        features_l = labeled_batch["features"]
        speakers_l = labeled_batch["speaker_ids"]
        labels_l = labeled_batch["labels"]

        logits_l = model(features_l, speakers_l)
        mask_l = labels_l != -1
        loss_supervised = criterion(logits_l[mask_l], labels_l[mask_l])

        stats = {
            "loss_supervised": loss_supervised.item(),
            "loss_unsupervised": 0.0,
            "loss_total": loss_supervised.item(),
            "pseudo_label_count": 0,
            "pseudo_label_total": 0,
        }

        if unlabeled_batch is None or unlabeled_batch["features"].size(0) == 0:
            return loss_supervised, stats

        # 2. Unlabeled dynamic thresholding
        features_u = unlabeled_batch["features"]
        speakers_u = unlabeled_batch["speaker_ids"]
        labels_u = unlabeled_batch["labels"]
        padding_mask = labels_u != -1

        model.eval()
        with torch.no_grad():
            logits_u = model(features_u, speakers_u)
            probs_u = F.softmax(logits_u, dim=-1)
            max_probs, pseudo_labels = probs_u.max(dim=-1)
        model.train()

        # Dynamic threshold adjustment based on active uncertainty
        # FedSwitch: lower threshold if prediction entropy is high, or dynamically adjust
        entropy = -torch.sum(probs_u * torch.log(probs_u + 1e-8), dim=-1)
        # Higher entropy (high uncertainty) -> increase threshold to remain strict
        dynamic_thresh = self.threshold * (1.0 - 0.1 * (1.0 - entropy / torch.log(torch.tensor(self.num_classes).float())))
        dynamic_thresh = dynamic_thresh.clamp(min=0.5, max=0.98)

        # Apply dynamic threshold mask
        confidence_mask = max_probs >= dynamic_thresh
        combined_mask = padding_mask & confidence_mask

        features_strong = self.strong_aug(features_u)
        logits_strong = model(features_strong, speakers_u)

        num_above = combined_mask.sum().item()
        num_total = padding_mask.sum().item()

        stats["pseudo_label_count"] = int(num_above)
        stats["pseudo_label_total"] = int(num_total)

        if num_above > 0:
            loss_unsupervised = F.cross_entropy(logits_strong[combined_mask], pseudo_labels[combined_mask])
            stats["loss_unsupervised"] = loss_unsupervised.item()
        else:
            loss_unsupervised = torch.tensor(0.0, device=features_u.device)

        total_loss = loss_supervised + self.lambda_u * loss_unsupervised
        stats["loss_total"] = total_loss.item()

        return total_loss, stats
