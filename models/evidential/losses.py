"""
Evidential Losses for ThuanPhongNhi
====================================
Implements two core loss functions:

1. Supervised Evidential Loss (Type-II Maximum Likelihood)
   - Replaces CrossEntropyLoss for labeled data
   - Includes KL regularization to prevent trivial solutions

2. Evidential Consistency Regularization (ECR)
   - Replaces FixMatch's CE-based pseudo-labeling
   - Certainty-weighted Dirichlet KL divergence
   - Gradient auto-vanishes when uncertain (no poison amplification)

References:
- Sensoy et al. (NeurIPS 2018) for supervised evidential loss
- ThuanPhongNhi proposal for ECR formulation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def dirichlet_kl_divergence(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """
    KL divergence between two Dirichlet distributions:
        KL(Dir(alpha) || Dir(beta))

    Args:
        alpha: (batch, C) Dirichlet params of q
        beta:  (batch, C) Dirichlet params of p

    Returns:
        (batch,) KL divergence values
    """
    alpha0 = alpha.sum(dim=-1)  # (batch,)
    beta0 = beta.sum(dim=-1)

    kl = (
        torch.lgamma(alpha0) - torch.lgamma(beta0)
        - (torch.lgamma(alpha) - torch.lgamma(beta)).sum(dim=-1)
        + ((alpha - beta) * (torch.digamma(alpha) - torch.digamma(alpha0.unsqueeze(-1)))).sum(dim=-1)
    )
    return kl


class SupervisedEvidentialLoss(nn.Module):
    """
    Type-II Maximum Likelihood loss for Evidential Deep Learning.

    L_sup = Σ_c y_c [ψ(S) - ψ(α^(c))]     (negative log-likelihood)
          + λ_KL · KL(Dir(α̃) || Dir(1))     (evidence regularization)

    where:
        α̃ = y + (1-y) ⊙ α   (remove evidence for correct class)
        λ_KL is annealed: min(1, epoch / T_anneal)

    Args:
        num_classes: Number of emotion classes
        annealing_epochs: Number of epochs to anneal KL weight from 0 to 1
        class_weights: Optional class weights for imbalanced data
    """

    def __init__(
        self,
        num_classes: int = 7,
        annealing_epochs: int = 10,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.annealing_epochs = annealing_epochs
        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None
            else torch.ones(num_classes),
        )
        self._current_epoch = 0

    def set_epoch(self, epoch: int):
        """Update current epoch for KL annealing."""
        self._current_epoch = epoch

    def forward(
        self,
        alpha: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute supervised evidential loss.

        Args:
            alpha: (N, C) Dirichlet concentration parameters
            labels: (N,) ground truth class indices
            mask: (N,) optional boolean mask (True = valid)

        Returns:
            loss: scalar loss
            stats: dict with component losses
        """
        # One-hot encode labels
        y = F.one_hot(labels, self.num_classes).float()  # (N, C)

        # Dirichlet strength
        S = alpha.sum(dim=-1, keepdim=True)  # (N, 1)

        # ========================================
        # 1. Negative log-likelihood (Type-II MLE)
        # ========================================
        # L_nll = Σ_c y_c [ψ(S) - ψ(α^(c))]
        nll = (y * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=-1)  # (N,)

        # Apply class weights
        sample_weights = self.class_weights[labels]  # (N,)
        nll = nll * sample_weights

        # ========================================
        # 2. KL regularization (annealed)
        # ========================================
        # Remove evidence for the correct class: α̃ = y + (1-y) ⊙ α
        alpha_tilde = y + (1.0 - y) * alpha

        # KL(Dir(α̃) || Dir(1)) — regularize toward uniform Dirichlet
        ones = torch.ones_like(alpha_tilde)
        kl = dirichlet_kl_divergence(alpha_tilde, ones)  # (N,)

        # Anneal KL weight
        lambda_kl = min(1.0, self._current_epoch / max(self.annealing_epochs, 1))

        # ========================================
        # 3. Combined loss
        # ========================================
        per_sample_loss = nll + lambda_kl * kl

        if mask is not None:
            per_sample_loss = per_sample_loss * mask.float()
            loss = per_sample_loss.sum() / mask.float().sum().clamp(min=1)
        else:
            loss = per_sample_loss.mean()

        stats = {
            "loss_nll": nll.mean().item(),
            "loss_kl": kl.mean().item(),
            "lambda_kl": lambda_kl,
            "loss_total": loss.item(),
        }

        return loss, stats


class EvidentialConsistencyRegularization(nn.Module):
    """
    Evidential Consistency Regularization (ECR) for SSL.

    Replaces FixMatch's CE-based pseudo-labeling with certainty-weighted
    Dirichlet KL divergence:

        L_ssl = (1 - u_f^(w)) · KL(Dir(α^(w)) || Dir(α^(s)))

    where:
        α^(w), α^(s) = Dirichlet params from weak/strong augmented views
        u_f^(w) = epistemic uncertainty from weak view
        (1 - u_f^(w)) = certainty weight

    Key properties:
        - When model is confident (u→0): full consistency loss
        - When model is uncertain (u→1): gradient auto-vanishes
        - No hard threshold needed (unlike FixMatch's 0.95 cutoff)
        - No pseudo-label generation → no confirmation bias

    Args:
        lambda_u: Weight for unsupervised loss
        uncertainty_threshold: Optional soft threshold (samples above this are ignored)
    """

    def __init__(
        self,
        lambda_u: float = 1.0,
        uncertainty_threshold: Optional[float] = None,
    ):
        super().__init__()
        self.lambda_u = lambda_u
        self.uncertainty_threshold = uncertainty_threshold

    def forward(
        self,
        alpha_weak: torch.Tensor,
        alpha_strong: torch.Tensor,
        uncertainty_weak: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute ECR loss.

        Args:
            alpha_weak: (N, C) Dirichlet params from weak/clean view
            alpha_strong: (N, C) Dirichlet params from strong augmented view
            uncertainty_weak: (N,) epistemic uncertainty from weak view
            mask: (N,) boolean mask for valid (non-padding) positions

        Returns:
            loss: scalar ECR loss
            stats: dict with diagnostics
        """
        # Certainty weight: (1 - u) — auto-vanishing gradient
        certainty = 1.0 - uncertainty_weak  # (N,)
        certainty = certainty.clamp(min=0.0)

        # Optional hard threshold on top of soft weighting
        if self.uncertainty_threshold is not None:
            threshold_mask = uncertainty_weak < self.uncertainty_threshold
            certainty = certainty * threshold_mask.float()

        # KL divergence between weak and strong Dirichlet
        # Detach weak view (target) — only strong view receives gradients
        kl = dirichlet_kl_divergence(
            alpha_strong,
            alpha_weak.detach(),
        )  # (N,)

        # Certainty-weighted KL
        weighted_kl = certainty.detach() * kl  # (N,)

        # Apply padding mask
        if mask is not None:
            weighted_kl = weighted_kl * mask.float()
            n_valid = mask.float().sum().clamp(min=1)
        else:
            n_valid = torch.tensor(weighted_kl.size(0), dtype=torch.float32)

        loss = self.lambda_u * weighted_kl.sum() / n_valid

        # Stats
        n_contributing = (certainty > 0.01).sum().item() if mask is None else \
            ((certainty > 0.01) & mask).sum().item()

        stats = {
            "loss_ecr": loss.item(),
            "mean_certainty": certainty.mean().item(),
            "mean_uncertainty": uncertainty_weak.mean().item(),
            "n_contributing": int(n_contributing),
            "n_total": int(n_valid.item()),
            "mean_kl": kl.mean().item(),
        }

        return loss, stats


class FedEvidenceLoss(nn.Module):
    """
    Combined loss for ThuanPhongNhi framework.

    L_client = L_sup + λ_u · L_ecr

    Manages both supervised evidential loss and ECR for SSL.

    Args:
        num_classes: Number of emotion classes
        annealing_epochs: Epochs to anneal KL weight in supervised loss
        lambda_u: Weight for unsupervised ECR loss
        lambda_u_rampup: Sigmoid ramp-up schedule for λ_u
        class_weights: Optional class weights
    """

    def __init__(
        self,
        num_classes: int = 7,
        annealing_epochs: int = 10,
        lambda_u: float = 1.0,
        lambda_u_rampup_epochs: int = 20,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.sup_loss = SupervisedEvidentialLoss(
            num_classes=num_classes,
            annealing_epochs=annealing_epochs,
            class_weights=class_weights,
        )
        self.ecr_loss = EvidentialConsistencyRegularization(
            lambda_u=1.0,  # We handle ramp-up externally
        )
        self.lambda_u_max = lambda_u
        self.lambda_u_rampup_epochs = lambda_u_rampup_epochs
        self._current_epoch = 0

    def set_epoch(self, epoch: int):
        """Update epoch for both annealing and ramp-up."""
        self._current_epoch = epoch
        self.sup_loss.set_epoch(epoch)

    def get_lambda_u(self) -> float:
        """Sigmoid ramp-up: λ_u = λ_max · σ(10·(t/T - 0.5))"""
        if self.lambda_u_rampup_epochs <= 0:
            return self.lambda_u_max
        progress = self._current_epoch / self.lambda_u_rampup_epochs
        sigmoid = 1.0 / (1.0 + torch.exp(torch.tensor(-10.0 * (progress - 0.5))).item())
        return self.lambda_u_max * sigmoid

    def forward(
        self,
        alpha_labeled: torch.Tensor,
        labels: torch.Tensor,
        label_mask: Optional[torch.Tensor] = None,
        alpha_weak: Optional[torch.Tensor] = None,
        alpha_strong: Optional[torch.Tensor] = None,
        uncertainty_weak: Optional[torch.Tensor] = None,
        unlabeled_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute combined ThuanPhongNhi loss.

        Args:
            alpha_labeled: (N_l, C) Dirichlet params for labeled data
            labels: (N_l,) ground truth labels
            label_mask: (N_l,) valid positions for labeled data
            alpha_weak: (N_u, C) Dirichlet params from weak view (unlabeled)
            alpha_strong: (N_u, C) Dirichlet params from strong view (unlabeled)
            uncertainty_weak: (N_u,) uncertainty from weak view
            unlabeled_mask: (N_u,) valid positions for unlabeled data

        Returns:
            total_loss, combined_stats
        """
        # Supervised loss
        sup_loss, sup_stats = self.sup_loss(alpha_labeled, labels, label_mask)

        stats = {
            "loss_supervised": sup_loss.item(),
            "loss_ecr": 0.0,
            "loss_total": sup_loss.item(),
            "lambda_u": self.get_lambda_u(),
            **{f"sup_{k}": v for k, v in sup_stats.items()},
        }

        # ECR loss (if unlabeled data provided)
        if alpha_weak is not None and alpha_strong is not None:
            ecr_loss, ecr_stats = self.ecr_loss(
                alpha_weak, alpha_strong, uncertainty_weak, unlabeled_mask
            )
            lambda_u = self.get_lambda_u()
            total_loss = sup_loss + lambda_u * ecr_loss
            stats["loss_ecr"] = ecr_loss.item()
            stats["loss_total"] = total_loss.item()
            stats.update({f"ecr_{k}": v for k, v in ecr_stats.items()})
        else:
            total_loss = sup_loss

        return total_loss, stats
