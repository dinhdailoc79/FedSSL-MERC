"""
DialogueRNN — Baseline ERC Model
==================================
An Attentive RNN for Emotion Detection in Conversations.

Paper: Majumder et al., "DialogueRNN: An Attentive RNN for Emotion
       Detection in Conversations", AAAI 2019.

Architecture:
    - Global GRU: tracks overall conversation context
    - Party GRU: tracks individual speaker states
    - Emotion GRU: produces emotion representations
    - Attention: attends to past utterances for context

This is the primary baseline that ALL Q1 ERC papers compare against.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class DialogueRNN(nn.Module):
    """
    DialogueRNN for Emotion Recognition in Conversations.

    Tracks three interacting states:
    1. Global state (g): Overall conversation flow
    2. Party state (q): Per-speaker emotional state
    3. Emotion state (e): Final emotion representation

    Args:
        input_dim: Dimension of input utterance features
        hidden_dim: Hidden dimension for all GRUs
        num_classes: Number of emotion classes
        num_speakers: Maximum number of speakers (default=2 for dyadic)
        dropout: Dropout rate
        use_attention: Whether to use attention over past utterances
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        num_classes: int = 7,
        num_speakers: int = 10,
        dropout: float = 0.3,
        use_attention: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_speakers = num_speakers
        self.use_attention = use_attention

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Global GRU — tracks overall conversation context
        self.global_gru = nn.GRUCell(hidden_dim, hidden_dim)

        # Party GRU — tracks individual speaker states
        self.party_gru = nn.GRUCell(hidden_dim + hidden_dim, hidden_dim)

        # Emotion GRU — produces emotion representations
        self.emotion_gru = nn.GRUCell(hidden_dim, hidden_dim)

        # Speaker embedding
        self.speaker_embed = nn.Embedding(num_speakers, hidden_dim)

        # Attention over past emotion states
        if use_attention:
            self.attention = ContextAttention(hidden_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        utterances: torch.Tensor,
        speaker_ids: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through DialogueRNN.

        Args:
            utterances: (batch, max_seq_len, input_dim) — utterance features
            speaker_ids: (batch, max_seq_len) — speaker ID per utterance
            lengths: (batch,) — actual dialogue lengths (for masking)

        Returns:
            logits: (batch, max_seq_len, num_classes)
        """
        batch_size, max_len, _ = utterances.shape
        device = utterances.device

        # Project input
        utterances = self.input_proj(utterances)  # (B, T, H)
        utterances = self.dropout(utterances)

        # Initialize hidden states
        global_state = torch.zeros(batch_size, self.hidden_dim, device=device)
        party_states = torch.zeros(
            batch_size, self.num_speakers, self.hidden_dim, device=device
        )
        emotion_state = torch.zeros(batch_size, self.hidden_dim, device=device)

        # Store emotion states for attention
        all_emotion_states = []
        all_logits = []

        for t in range(max_len):
            utt_t = utterances[:, t, :]       # (B, H)
            spk_t = speaker_ids[:, t]          # (B,)

            # 1. Update global state
            global_state = self.global_gru(utt_t, global_state)

            # 2. Get current speaker's party state
            spk_idx = spk_t.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.hidden_dim)
            current_party = party_states.gather(1, spk_idx).squeeze(1)  # (B, H)

            # 3. Update party state with utterance + global context
            party_input = torch.cat([utt_t, global_state], dim=-1)  # (B, 2H)
            new_party = self.party_gru(party_input, current_party)

            # Write back updated party state (out-of-place to avoid autograd error)
            party_states = party_states.clone()
            party_states.scatter_(1, spk_idx, new_party.unsqueeze(1))

            # 4. Compute emotion input (with optional attention)
            if self.use_attention and len(all_emotion_states) > 0:
                past_emotions = torch.stack(all_emotion_states, dim=1)  # (B, t, H)
                context = self.attention(new_party, past_emotions)
                emotion_input = context + new_party
            else:
                emotion_input = new_party

            # 5. Update emotion state
            emotion_state = self.emotion_gru(emotion_input, emotion_state)
            all_emotion_states.append(emotion_state)

            # 6. Classify
            logits_t = self.classifier(emotion_state)  # (B, num_classes)
            all_logits.append(logits_t)

        # Stack: (B, T, num_classes)
        logits = torch.stack(all_logits, dim=1)
        return logits

    def get_features(
        self,
        utterances: torch.Tensor,
        speaker_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Extract emotion features without classification (for FL/SSL)."""
        batch_size, max_len, _ = utterances.shape
        device = utterances.device

        utterances = self.input_proj(utterances)
        global_state = torch.zeros(batch_size, self.hidden_dim, device=device)
        party_states = torch.zeros(
            batch_size, self.num_speakers, self.hidden_dim, device=device
        )
        emotion_state = torch.zeros(batch_size, self.hidden_dim, device=device)
        all_emotion_states = []

        for t in range(max_len):
            utt_t = utterances[:, t, :]
            spk_t = speaker_ids[:, t]

            global_state = self.global_gru(utt_t, global_state)

            spk_idx = spk_t.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.hidden_dim)
            current_party = party_states.gather(1, spk_idx).squeeze(1)

            party_input = torch.cat([utt_t, global_state], dim=-1)
            new_party = self.party_gru(party_input, current_party)
            party_states = party_states.clone()
            party_states.scatter_(1, spk_idx, new_party.unsqueeze(1))

            if self.use_attention and len(all_emotion_states) > 0:
                past_emotions = torch.stack(all_emotion_states, dim=1)
                context = self.attention(new_party, past_emotions)
                emotion_input = context + new_party
            else:
                emotion_input = new_party

            emotion_state = self.emotion_gru(emotion_input, emotion_state)
            all_emotion_states.append(emotion_state)

        return torch.stack(all_emotion_states, dim=1)  # (B, T, H)


class ContextAttention(nn.Module):
    """
    Attention mechanism over past emotion states.
    Computes attention weights based on current party state
    and past emotion states.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query: (batch, hidden_dim) — current party state
            keys: (batch, seq_len, hidden_dim) — past emotion states

        Returns:
            context: (batch, hidden_dim) — weighted sum of past states
        """
        # Score: tanh(W * keys) @ query
        proj = torch.tanh(self.W(keys))  # (B, T, H)
        scores = self.v(proj * query.unsqueeze(1)).squeeze(-1)  # (B, T)
        weights = F.softmax(scores, dim=-1)  # (B, T)

        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)  # (B, H)
        return context
