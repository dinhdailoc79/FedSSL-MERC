"""
Evidential Deep Learning (EDL) Head
=====================================
Replaces softmax output with Dirichlet distribution parameters.

Based on:
- Sensoy et al. (NeurIPS 2018): "Evidential Deep Learning to Quantify
  Classification Uncertainty"

Key idea:
- Instead of predicting class probabilities (softmax), predict EVIDENCE
  for each class using softplus activation.
- Evidence parameterizes a Dirichlet distribution over class probabilities.
- This naturally provides epistemic uncertainty (how much the model "doesn't know").

Mathematical formulation:
    evidence e = softplus(logits) ∈ R^C_≥0
    Dirichlet params α = e + 1
    Dirichlet strength S = Σ α^(c)
    Belief mass b^(c) = e^(c) / S
    Epistemic uncertainty u = C / S ∈ [0, 1]
        u → 0: high evidence (certain)
        u → 1: no evidence (maximum ignorance)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class EvidentialHead(nn.Module):
    """
    Evidential output layer that replaces standard softmax classification head.

    Takes hidden representations and produces:
    - evidence: non-negative evidence for each class
    - alpha: Dirichlet concentration parameters
    - belief: belief mass for each class
    - uncertainty: epistemic uncertainty scalar

    Args:
        input_dim: Dimension of input hidden states
        num_classes: Number of emotion classes
        activation: Activation to ensure non-negative evidence ('softplus' or 'relu' or 'exp')
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        activation: str = "softplus",
    ):
        super().__init__()
        self.num_classes = num_classes

        # Evidence projection layer
        self.evidence_fc = nn.Linear(input_dim, num_classes)

        # Activation for non-negative evidence
        if activation == "softplus":
            self.activation = nn.Softplus()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "exp":
            self.activation = lambda x: torch.exp(torch.clamp(x, max=10))
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            hidden: (batch, seq_len, hidden_dim) or (batch, hidden_dim)

        Returns:
            Dict with keys:
                'evidence': (batch, [seq_len], C) non-negative evidence
                'alpha': (batch, [seq_len], C) Dirichlet parameters (evidence + 1)
                'strength': (batch, [seq_len]) total Dirichlet strength S
                'belief': (batch, [seq_len], C) belief masses
                'uncertainty': (batch, [seq_len]) epistemic uncertainty u ∈ [0,1]
                'logits': (batch, [seq_len], C) raw logits (for compatibility)
        """
        # Raw logits
        logits = self.evidence_fc(hidden)

        # Non-negative evidence via activation
        evidence = self.activation(logits)

        # Dirichlet concentration parameters: α = e + 1
        alpha = evidence + 1.0

        # Dirichlet strength: S = Σ α^(c)
        strength = alpha.sum(dim=-1)  # (batch, [seq_len])

        # Belief mass: b^(c) = e^(c) / S
        belief = evidence / strength.unsqueeze(-1)

        # Epistemic uncertainty: u = C / S
        uncertainty = self.num_classes / strength

        return {
            "evidence": evidence,
            "alpha": alpha,
            "strength": strength,
            "belief": belief,
            "uncertainty": uncertainty,
            "logits": logits,
        }

    def predict(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience method: returns predicted classes and uncertainties.

        Returns:
            predictions: (batch, [seq_len]) predicted class indices
            uncertainties: (batch, [seq_len]) epistemic uncertainty values
        """
        out = self.forward(hidden)
        predictions = out["belief"].argmax(dim=-1)
        return predictions, out["uncertainty"]
