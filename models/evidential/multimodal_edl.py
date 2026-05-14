"""
Multimodal Evidential DialogueRNN
====================================
Extends EvidentialDialogueRNN with multi-modality support.

Architecture:
    text features  → DialogueRNN encoder → text emotion states  → EDL → text evidence
    audio features → DialogueRNN encoder → audio emotion states → EDL → audio evidence
                                           ↓
                                    DS Fusion (evidence sum)
                                           ↓
                                    fused α, belief, uncertainty

Each modality has its own encoder + EDL head, producing independent evidence.
Dempster-Shafer fusion combines evidences — uncertain modalities contribute less.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List

from models.erc.dialogue_rnn import DialogueRNN
from models.evidential.edl_head import EvidentialHead
from models.evidential.ds_fusion import DempsterShaferFusion


class MultimodalEvidentialDialogueRNN(nn.Module):
    """
    Multimodal EDL model with DS fusion.

    Each modality (text, audio) has:
    - Its own DialogueRNN encoder
    - Its own EDL projection + head

    DS fusion combines evidence from all modalities.

    Args:
        text_dim: Text feature dimension (768 for RoBERTa)
        audio_dim: Audio feature dimension (768 for WavLM)
        hidden_dim: DialogueRNN hidden dimension
        num_classes: Number of emotion classes
        num_speakers: Max speakers
        dropout: Dropout rate
        fusion_mode: 'evidence_sum' or 'dempster'
    """

    def __init__(
        self,
        text_dim: int = 768,
        audio_dim: int = 768,
        hidden_dim: int = 256,
        num_classes: int = 7,
        num_speakers: int = 10,
        dropout: float = 0.3,
        fusion_mode: str = "evidence_sum",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        # Text encoder
        self.text_encoder = DialogueRNN(
            input_dim=text_dim, hidden_dim=hidden_dim,
            num_classes=num_classes, num_speakers=num_speakers,
            dropout=dropout, use_attention=True,
        )
        self.text_projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.text_edl = EvidentialHead(hidden_dim, num_classes)

        # Audio encoder
        self.audio_encoder = DialogueRNN(
            input_dim=audio_dim, hidden_dim=hidden_dim,
            num_classes=num_classes, num_speakers=num_speakers,
            dropout=dropout, use_attention=True,
        )
        self.audio_projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.audio_edl = EvidentialHead(hidden_dim, num_classes)

        # DS Fusion
        self.fusion = DempsterShaferFusion(
            num_classes=num_classes, mode=fusion_mode,
        )

    def forward(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        speaker_ids: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with DS fusion.

        Args:
            text_features: (batch, seq, text_dim)
            audio_features: (batch, seq, audio_dim)
            speaker_ids: (batch, seq)

        Returns:
            Dict with fused EDL outputs + per-modality outputs
        """
        # Text branch
        text_hidden = self.text_encoder.get_features(text_features, speaker_ids)
        text_proj = self.text_projection(text_hidden)
        text_edl = self.text_edl(text_proj)

        # Audio branch
        audio_hidden = self.audio_encoder.get_features(audio_features, speaker_ids)
        audio_proj = self.audio_projection(audio_hidden)
        audio_edl = self.audio_edl(audio_proj)

        # DS Fusion
        fused = self.fusion([text_edl["evidence"], audio_edl["evidence"]])

        # Return fused + per-modality for auxiliary losses
        return {
            # Fused outputs (primary)
            "alpha": fused["alpha"],
            "belief": fused["belief"],
            "uncertainty": fused["uncertainty"],
            "evidence": fused["evidence"],
            # Per-modality (for auxiliary supervision)
            "text_alpha": text_edl["alpha"],
            "text_uncertainty": text_edl["uncertainty"],
            "audio_alpha": audio_edl["alpha"],
            "audio_uncertainty": audio_edl["uncertainty"],
        }

    def forward_text_only(
        self, text_features: torch.Tensor, speaker_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward with text modality only (for comparison)."""
        text_hidden = self.text_encoder.get_features(text_features, speaker_ids)
        text_proj = self.text_projection(text_hidden)
        return self.text_edl(text_proj)
