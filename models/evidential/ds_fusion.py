"""
Dempster-Shafer Evidence Fusion
=================================
Combines evidence from multiple modalities using Dempster's rule of combination.

Based on:
- Shafer (1976): "A Mathematical Theory of Evidence"
- Han et al. (2022): Trusted Multi-View Classification

Key idea:
- Each modality produces its own Dirichlet evidence via EDL.
- DS fusion combines these evidences at the EVIDENCE level (not softmax).
- This allows uncertainty-aware fusion: a modality with low evidence (high
  uncertainty) contributes less, naturally handling noisy/missing modalities.

Dempster's Rule:
    Given two opinion frames with belief b1, b2 and uncertainty u1, u2:
    Combined belief:  b^(c) = (b1^(c)*b2^(c) + b1^(c)*u2 + b2^(c)*u1) / (1 - conflict)
    Combined uncertainty: u = u1 * u2 / (1 - conflict)
    Conflict: K = Σ_{i≠j} b1^(i) * b2^(j)

Simplified version (evidence addition):
    Combined evidence: e_fused = e_text + e_audio
    This is equivalent to DS combination under Dirichlet assumptions.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional


class DempsterShaferFusion(nn.Module):
    """
    Dempster-Shafer fusion for combining evidence from multiple modalities.

    Supports two fusion modes:
    1. 'evidence_sum': Simply sum evidence (fast, theoretically grounded)
    2. 'dempster': Full Dempster's rule on belief masses (more expressive)

    Args:
        num_classes: Number of classes
        mode: Fusion mode ('evidence_sum' or 'dempster')
    """

    def __init__(
        self,
        num_classes: int,
        mode: str = "evidence_sum",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.mode = mode

    def forward(
        self,
        evidence_list: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Fuse evidence from multiple modalities.

        Args:
            evidence_list: List of evidence tensors, each (batch, [seq], C)
                          Each tensor contains non-negative evidence values.

        Returns:
            Dict with fused EDL outputs:
                'evidence': fused evidence
                'alpha': fused Dirichlet params
                'belief': fused belief masses
                'uncertainty': fused uncertainty
        """
        if self.mode == "evidence_sum":
            return self._evidence_sum_fusion(evidence_list)
        elif self.mode == "dempster":
            return self._dempster_fusion(evidence_list)
        else:
            raise ValueError(f"Unknown fusion mode: {self.mode}")

    def _evidence_sum_fusion(
        self, evidence_list: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Sum evidence from all modalities.

        Under Dirichlet assumptions, summing evidence is equivalent to
        combining independent observations — each modality provides
        additional "counts" of evidence for each class.

        This is the recommended mode:
        - Computationally cheap
        - Theoretically sound (conjugate prior update)
        - Naturally handles missing modalities (zero evidence)
        """
        # Sum all evidence
        fused_evidence = evidence_list[0]
        for e in evidence_list[1:]:
            fused_evidence = fused_evidence + e

        # Compute Dirichlet parameters from fused evidence
        alpha = fused_evidence + 1.0
        strength = alpha.sum(dim=-1)
        belief = fused_evidence / strength.unsqueeze(-1)
        uncertainty = self.num_classes / strength

        return {
            "evidence": fused_evidence,
            "alpha": alpha,
            "strength": strength,
            "belief": belief,
            "uncertainty": uncertainty,
        }

    def _dempster_fusion(
        self, evidence_list: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Full Dempster's rule of combination on belief masses.

        Iteratively combines modalities pairwise.
        """
        # Convert first modality to belief frame
        e = evidence_list[0]
        alpha = e + 1.0
        S = alpha.sum(dim=-1, keepdim=True)
        b = e / S
        u = self.num_classes / S.squeeze(-1)

        # Iteratively combine with remaining modalities
        for e_m in evidence_list[1:]:
            alpha_m = e_m + 1.0
            S_m = alpha_m.sum(dim=-1, keepdim=True)
            b_m = e_m / S_m
            u_m = self.num_classes / S_m.squeeze(-1)

            b, u = self._combine_two(b, u, b_m, u_m)

        # Convert back to evidence
        # From belief and uncertainty: b^(c) = e^(c) / S, u = C / S
        # => S = C / u, e^(c) = b^(c) * S
        S_fused = self.num_classes / u.clamp(min=1e-10)
        fused_evidence = b * S_fused.unsqueeze(-1)
        alpha = fused_evidence + 1.0
        strength = alpha.sum(dim=-1)

        return {
            "evidence": fused_evidence,
            "alpha": alpha,
            "strength": strength,
            "belief": b,
            "uncertainty": u,
        }

    def _combine_two(
        self,
        b1: torch.Tensor,
        u1: torch.Tensor,
        b2: torch.Tensor,
        u2: torch.Tensor,
    ):
        """
        Combine two belief frames using Dempster's rule.

        b1, b2: (batch, [seq], C) belief masses
        u1, u2: (batch, [seq]) uncertainties
        """
        C = self.num_classes

        # Expand uncertainties for broadcasting
        u1_exp = u1.unsqueeze(-1)  # (batch, [seq], 1)
        u2_exp = u2.unsqueeze(-1)

        # Combined belief (unnormalized)
        # b_fused^(c) = b1^(c)*b2^(c) + b1^(c)*u2 + b2^(c)*u1
        bb = b1 * b2  # agreement on same class
        bu = b1 * u2_exp  # b1 with b2's uncertainty
        ub = u1_exp * b2  # b2 with b1's uncertainty

        combined = bb + bu + ub

        # Normalization factor (1 - conflict)
        # conflict K = sum of products where classes disagree
        # K = sum_{i≠j} b1^i * b2^j = (sum b1)(sum b2) - sum(b1*b2)
        K = (b1.sum(dim=-1) * b2.sum(dim=-1) - (b1 * b2).sum(dim=-1))
        norm = (1.0 - K).clamp(min=1e-10).unsqueeze(-1)

        b_fused = combined / norm

        # Combined uncertainty
        u_fused = (u1 * u2) / (1.0 - K).clamp(min=1e-10)

        return b_fused, u_fused
