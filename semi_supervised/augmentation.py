"""
Feature-Level Augmentation for Semi-Supervised Learning
=========================================================
Since we use pre-extracted RoBERTa features (768-dim vectors),
traditional text augmentation (synonym replacement, etc.) is not applicable.

Instead, we apply augmentation directly on feature vectors:
- Weak: small Gaussian noise + light feature dropout
- Strong: larger noise + heavier dropout + feature cutoff

References:
- UDA (Xie et al., 2020): Unsupervised Data Augmentation
- FixMatch (Sohn et al., 2020): feature perturbation strategy
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class WeakAugmentation(nn.Module):
    """
    Weak augmentation for FixMatch.
    Applies small perturbations that preserve semantic meaning.

    - Gaussian noise with small sigma
    - Light feature dropout (5-10%)
    """

    def __init__(self, noise_std: float = 0.01, dropout_p: float = 0.05):
        super().__init__()
        self.noise_std = noise_std
        self.dropout_p = dropout_p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, feat_dim) or (seq_len, feat_dim)
        Returns:
            Augmented tensor of same shape
        """
        if not self.training:
            return x

        # Gaussian noise
        noise = torch.randn_like(x) * self.noise_std
        x_aug = x + noise

        # Feature dropout (zero out random dimensions)
        mask = torch.bernoulli(
            torch.full_like(x, 1.0 - self.dropout_p)
        )
        x_aug = x_aug * mask

        # Re-scale to maintain expected magnitude
        x_aug = x_aug / (1.0 - self.dropout_p + 1e-8)

        return x_aug


class StrongAugmentation(nn.Module):
    """
    Strong augmentation for FixMatch.
    Applies heavier perturbations while keeping the feature recognizable.

    - Larger Gaussian noise
    - Heavier feature dropout (20-30%)
    - Feature cutoff: zero out contiguous blocks of dimensions
    - Random scaling per dimension
    """

    def __init__(
        self,
        noise_std: float = 0.05,
        dropout_p: float = 0.25,
        cutoff_ratio: float = 0.1,
        scale_range: tuple = (0.8, 1.2),
    ):
        super().__init__()
        self.noise_std = noise_std
        self.dropout_p = dropout_p
        self.cutoff_ratio = cutoff_ratio
        self.scale_range = scale_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, feat_dim) or (seq_len, feat_dim)
        Returns:
            Strongly augmented tensor
        """
        if not self.training:
            return x

        feat_dim = x.shape[-1]

        # 1. Random dimension scaling
        scale = torch.empty(feat_dim, device=x.device).uniform_(
            self.scale_range[0], self.scale_range[1]
        )
        x_aug = x * scale

        # 2. Gaussian noise (larger)
        noise = torch.randn_like(x_aug) * self.noise_std
        x_aug = x_aug + noise

        # 3. Feature dropout (heavier)
        mask = torch.bernoulli(
            torch.full_like(x_aug, 1.0 - self.dropout_p)
        )
        x_aug = x_aug * mask / (1.0 - self.dropout_p + 1e-8)

        # 4. Feature cutoff: zero out a contiguous block
        cutoff_len = int(feat_dim * self.cutoff_ratio)
        if cutoff_len > 0:
            start = torch.randint(0, feat_dim - cutoff_len, (1,)).item()
            x_aug[..., start:start + cutoff_len] = 0.0

        return x_aug


class MixUpAugmentation(nn.Module):
    """
    MixUp augmentation: linearly interpolate between samples.
    Used as an additional regularization technique.

    mixup(x_i, x_j) = lambda * x_i + (1-lambda) * x_j
    """

    def __init__(self, alpha: float = 0.2):
        super().__init__()
        self.alpha = alpha

    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        Args:
            x: features (batch, ...)
            y: labels (batch,) — optional, also mixed if provided
        Returns:
            (mixed_x, mixed_y, lam) or (mixed_x, lam) if y is None
        """
        batch_size = x.size(0)
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1.0

        # Random permutation for mixing partners
        index = torch.randperm(batch_size, device=x.device)
        mixed_x = lam * x + (1 - lam) * x[index]

        if y is not None:
            return mixed_x, y, y[index], lam
        return mixed_x, lam
