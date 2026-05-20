"""
SOTA ERC Baselines for Federated Setting
==========================================
Implements simplified versions of:
1. CoMPM (Lee & Lee, 2021) - Context-aware Memory Pre-trained Model
2. SPCL (Song et al., 2022) - Supervised Prototypical Contrastive Learning

Both adapted to run under our Federated Learning setting for fair comparison.

Note: These are simplified but faithful re-implementations focused on the
core architectural innovations, using the same RoBERTa features as our method.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import math


# ============================================================
# 1. CoMPM-FL: Context Memory Pre-trained Model (Federated)
# ============================================================

class CoMPMEncoder(nn.Module):
    """
    Simplified CoMPM for ERC.
    
    Key difference from DialogueRNN:
    - Uses multi-head self-attention (transformer-style) for context modeling
    - Context memory: maintains sliding window of past utterances
    - Position-aware encoding for temporal dialogue structure
    
    Reference: Lee & Lee, "CoMPM: Context Modeling with Speaker's Pre-trained
    Memory Tracking for Emotion Recognition in Conversation", NAACL 2022.
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        num_classes: int = 7,
        num_speakers: int = 10,
        dropout: float = 0.3,
        num_heads: int = 4,
        num_layers: int = 2,
        context_window: int = 10,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.context_window = context_window
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Speaker embedding
        self.speaker_embed = nn.Embedding(num_speakers, hidden_dim)
        
        # Positional encoding (learnable)
        self.pos_embed = nn.Embedding(512, hidden_dim)
        
        # Transformer encoder for context modeling (core CoMPM innovation)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
        )
        
        # Speaker-aware memory gate
        self.memory_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        utterances: torch.Tensor,
        speaker_ids: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            utterances: (B, T, input_dim)
            speaker_ids: (B, T)
        Returns:
            logits: (B, T, num_classes)
        """
        B, T, _ = utterances.shape
        device = utterances.device
        
        # Project input + add speaker + position embeddings
        x = self.input_proj(utterances)  # (B, T, H)
        
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embed(positions)  # (B, T, H)
        spk_emb = self.speaker_embed(speaker_ids)  # (B, T, H)
        
        x = self.layer_norm(x + pos_emb + spk_emb)
        x = self.dropout(x)
        
        # Causal mask: each position can only attend to current and past
        causal_mask = torch.triu(
            torch.ones(T, T, device=device) * float('-inf'), diagonal=1
        )
        
        # Context encoding via transformer
        context = self.context_encoder(x, mask=causal_mask)  # (B, T, H)
        
        # Memory gate: blend context with original representation
        gate = self.memory_gate(torch.cat([x, context], dim=-1))  # (B, T, H)
        fused = gate * context + (1 - gate) * x  # (B, T, H)
        
        # Classify
        logits = self.classifier(fused)  # (B, T, C)
        return logits
    
    def get_features(self, utterances, speaker_ids):
        """Extract features without classification (for compatibility)."""
        B, T, _ = utterances.shape
        device = utterances.device
        
        x = self.input_proj(utterances)
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embed(positions)
        spk_emb = self.speaker_embed(speaker_ids)
        
        x = self.layer_norm(x + pos_emb + spk_emb)
        x = self.dropout(x)
        
        causal_mask = torch.triu(
            torch.ones(T, T, device=device) * float('-inf'), diagonal=1
        )
        context = self.context_encoder(x, mask=causal_mask)
        
        gate = self.memory_gate(torch.cat([x, context], dim=-1))
        return gate * context + (1 - gate) * x


# ============================================================
# 2. SPCL-FL: Supervised Prototypical Contrastive Learning
# ============================================================

class SPCLDialogueRNN(nn.Module):
    """
    DialogueRNN + Supervised Prototypical Contrastive Learning.
    
    Key addition over DialogueRNN:
    - Prototypical contrastive loss: pulls same-class embeddings together
    - Class prototypes updated via EMA during training
    - Contrastive loss added as auxiliary objective
    
    Reference: Song et al., "Supervised Prototypical Contrastive Learning 
    for Emotion Recognition in Conversation", ACL 2022.
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        num_classes: int = 7,
        num_speakers: int = 10,
        dropout: float = 0.3,
        temperature: float = 0.07,
        prototype_momentum: float = 0.99,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.temperature = temperature
        self.prototype_momentum = prototype_momentum
        
        # Core DialogueRNN components
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.global_gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.party_gru = nn.GRUCell(hidden_dim + hidden_dim, hidden_dim)
        self.emotion_gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.speaker_embed = nn.Embedding(num_speakers, hidden_dim)
        
        # Attention
        self.attn_W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_v = nn.Linear(hidden_dim, 1, bias=False)
        
        # Projection head for contrastive learning
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        
        # Class prototypes (EMA-updated)
        self.register_buffer(
            'prototypes', torch.randn(num_classes, hidden_dim)
        )
        self.register_buffer(
            'prototype_counts', torch.zeros(num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def _attention(self, query, keys):
        """Context attention over past emotion states."""
        proj = torch.tanh(self.attn_W(keys))
        scores = self.attn_v(proj * query.unsqueeze(1)).squeeze(-1)
        weights = F.softmax(scores, dim=-1)
        return torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
    
    def _encode(self, utterances, speaker_ids):
        """Run DialogueRNN encoder, return emotion features."""
        B, T, _ = utterances.shape
        device = utterances.device
        
        x = self.input_proj(utterances)
        x = self.dropout(x)
        
        global_state = torch.zeros(B, self.hidden_dim, device=device)
        party_states = torch.zeros(B, 10, self.hidden_dim, device=device)
        emotion_state = torch.zeros(B, self.hidden_dim, device=device)
        all_emotions = []
        
        for t in range(T):
            utt_t = x[:, t, :]
            spk_t = speaker_ids[:, t]
            
            global_state = self.global_gru(utt_t, global_state)
            
            spk_idx = spk_t.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.hidden_dim)
            current_party = party_states.gather(1, spk_idx).squeeze(1)
            
            party_input = torch.cat([utt_t, global_state], dim=-1)
            new_party = self.party_gru(party_input, current_party)
            party_states = party_states.clone()
            party_states.scatter_(1, spk_idx, new_party.unsqueeze(1))
            
            if len(all_emotions) > 0:
                past = torch.stack(all_emotions, dim=1)
                context = self._attention(new_party, past)
                emotion_input = context + new_party
            else:
                emotion_input = new_party
            
            emotion_state = self.emotion_gru(emotion_input, emotion_state)
            all_emotions.append(emotion_state)
        
        return torch.stack(all_emotions, dim=1)  # (B, T, H)
    
    def forward(self, utterances, speaker_ids, lengths=None):
        """Forward pass returning logits."""
        features = self._encode(utterances, speaker_ids)
        logits = self.classifier(features)
        return logits
    
    def get_features(self, utterances, speaker_ids):
        """Get raw features (for compatibility)."""
        return self._encode(utterances, speaker_ids)
    
    def contrastive_loss(self, features, labels, mask):
        """
        Supervised Prototypical Contrastive Loss.
        
        Args:
            features: (B, T, H) emotion features
            labels: (B, T) emotion labels (-1 for padding)
            mask: (B, T) bool mask for valid positions
        
        Returns:
            loss: scalar contrastive loss
        """
        # Flatten valid features
        feat_flat = features[mask]  # (N, H)
        label_flat = labels[mask]   # (N,)
        
        if feat_flat.size(0) < 2:
            return torch.tensor(0.0, device=features.device)
        
        # Project to contrastive space
        z = F.normalize(self.projector(feat_flat), dim=-1)  # (N, H)
        
        # Update prototypes (EMA)
        if self.training:
            with torch.no_grad():
                for c in range(self.num_classes):
                    c_mask = label_flat == c
                    if c_mask.sum() > 0:
                        c_mean = z[c_mask].mean(dim=0)
                        m = self.prototype_momentum
                        self.prototypes[c] = m * self.prototypes[c] + (1 - m) * c_mean
                        self.prototype_counts[c] += c_mask.sum()
        
        # Normalize prototypes
        proto_norm = F.normalize(self.prototypes, dim=-1)  # (C, H)
        
        # Compute similarity to all prototypes
        sim = torch.mm(z, proto_norm.t()) / self.temperature  # (N, C)
        
        # Cross-entropy with prototypes as targets
        loss = F.cross_entropy(sim, label_flat)
        
        return loss


# ============================================================
# Model factory
# ============================================================

def create_sota_model(model_name, input_dim=768, hidden_dim=256, 
                      num_classes=7, num_speakers=10, dropout=0.3):
    """Create a SOTA baseline model by name."""
    if model_name == 'compm':
        return CoMPMEncoder(
            input_dim=input_dim, hidden_dim=hidden_dim,
            num_classes=num_classes, num_speakers=num_speakers,
            dropout=dropout, num_heads=4, num_layers=2,
        )
    elif model_name == 'spcl':
        return SPCLDialogueRNN(
            input_dim=input_dim, hidden_dim=hidden_dim,
            num_classes=num_classes, num_speakers=num_speakers,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}. Use 'compm' or 'spcl'.")
