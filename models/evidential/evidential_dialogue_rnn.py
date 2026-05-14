"""
Evidential DialogueRNN
========================
Wraps DialogueRNN with an Evidential Head instead of softmax classifier.

Returns Dirichlet parameters (alpha, belief, uncertainty) instead of logits.
Supports both supervised training and LucBinh SSL pipeline.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from models.erc.dialogue_rnn import DialogueRNN
from models.evidential.edl_head import EvidentialHead


class EvidentialDialogueRNN(nn.Module):
    """
    DialogueRNN with Evidential Deep Learning head.

    Architecture:
        utterance features → DialogueRNN encoder → emotion states → EDL head
        → evidence, alpha, belief, uncertainty

    The EDL head replaces the standard softmax classifier.
    DialogueRNN encoder (GRUs + attention) is reused as-is.

    Args:
        input_dim: Dimension of utterance features (e.g., 768 for RoBERTa)
        hidden_dim: Hidden dimension for DialogueRNN
        num_classes: Number of emotion classes
        num_speakers: Maximum number of speakers
        dropout: Dropout rate
        use_attention: Whether to use attention in DialogueRNN
        edl_activation: Activation for evidence ('softplus', 'relu', 'exp')
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        num_classes: int = 7,
        num_speakers: int = 10,
        dropout: float = 0.3,
        use_attention: bool = True,
        edl_activation: str = "softplus",
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # DialogueRNN encoder (reuse everything except the classifier)
        self.encoder = DialogueRNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_speakers=num_speakers,
            dropout=dropout,
            use_attention=use_attention,
        )

        # Replace the softmax classifier with EDL head
        # Keep the encoder's feature extraction layers (shared dropout + linear)
        self.edl_projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.edl_head = EvidentialHead(
            input_dim=hidden_dim,
            num_classes=num_classes,
            activation=edl_activation,
        )

    def forward(
        self,
        utterances: torch.Tensor,
        speaker_ids: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: DialogueRNN encoder → EDL head.

        Args:
            utterances: (batch, max_seq_len, input_dim)
            speaker_ids: (batch, max_seq_len)
            lengths: (batch,) actual dialogue lengths

        Returns:
            Dict with all EDL outputs:
                'evidence': (batch, seq_len, C)
                'alpha': (batch, seq_len, C)
                'belief': (batch, seq_len, C)
                'uncertainty': (batch, seq_len)
                'logits': (batch, seq_len, C)
        """
        # Get emotion hidden states from encoder
        hidden_states = self.encoder.get_features(utterances, speaker_ids)
        # (batch, max_seq_len, hidden_dim)

        # Project through shared layers
        projected = self.edl_projection(hidden_states)

        # EDL head
        edl_output = self.edl_head(projected)

        return edl_output

    def predict(
        self,
        utterances: torch.Tensor,
        speaker_ids: torch.Tensor,
    ):
        """
        Convenience: returns predicted classes and uncertainties.

        Returns:
            predictions: (batch, seq_len) predicted class indices
            uncertainties: (batch, seq_len) epistemic uncertainty values
        """
        out = self.forward(utterances, speaker_ids)
        predictions = out["belief"].argmax(dim=-1)
        return predictions, out["uncertainty"]

    def get_mean_uncertainty(
        self,
        utterances: torch.Tensor,
        speaker_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """
        Compute mean epistemic uncertainty over valid (non-padding) positions.
        Used by EAFA to get ū_k for each client.

        Args:
            utterances, speaker_ids: model inputs
            labels: (batch, seq_len) with -1 for padding

        Returns:
            Scalar mean uncertainty
        """
        with torch.no_grad():
            out = self.forward(utterances, speaker_ids)
            mask = labels != -1
            uncertainties = out["uncertainty"][mask]
            return uncertainties.mean().item() if uncertainties.numel() > 0 else 1.0
