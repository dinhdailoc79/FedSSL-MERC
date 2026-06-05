"""
Modern Federated Learning Aggregation Baselines
=================================================
Implements:
1. SCAFFOLD  — Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging
               for Federated Learning", ICML 2020.
2. FedNova   — Wang et al., "Tackling the Objective Inconsistency Problem in
               Heterogeneous Federated Optimization", NeurIPS 2020.
3. FedAdam   — Reddi et al., "Adaptive Federated Optimization", ICLR 2021.
               (Server-side Adam on the FedAvg pseudo-gradient)
4. MOON      — Li et al., "Model-Contrastive Federated Learning", CVPR 2021.
               (Client-side contrastive loss; aggregation is standard FedAvg)

All functions operate on OrderedDict state_dicts to match the interface of
eafa_aggregate() and fedeu_aggregate_state_dicts().
"""

import copy
import logging
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# 1. SCAFFOLD
# ============================================================

class SCAFFOLDAggregator:
    """
    SCAFFOLD: Stochastic Controlled Averaging for Federated Learning.

    Maintains a global control variate `c` and per-client control variates
    `c_k`. On each round the server corrects the aggregated pseudo-gradient
    using the difference between the new and old client control variates.

    Interface
    ---------
    * Call ``client_update()`` inside the local training loop to obtain the
      drift-corrected gradient for each SGD step.
    * After local training, call ``compute_new_control_variate()`` to get
      the updated c_k.
    * Finally call ``aggregate()`` on the server to produce the new global
      model and global control variate.
    """

    def __init__(self, global_state_dict: OrderedDict, num_clients: int,
                 lr: float = 1e-3):
        self.num_clients = num_clients
        self.lr = lr

        # Global control variate  c  (initialized to zero)
        self.c_global = OrderedDict()
        for key, val in global_state_dict.items():
            self.c_global[key] = torch.zeros_like(val, dtype=torch.float32)

        # Per-client control variates  c_k
        self.c_clients = [
            OrderedDict({k: torch.zeros_like(v, dtype=torch.float32)
                         for k, v in global_state_dict.items()})
            for _ in range(num_clients)
        ]

    @staticmethod
    def client_grad_correction(
        grad: torch.Tensor,
        c_global_param: torch.Tensor,
        c_client_param: torch.Tensor,
    ) -> torch.Tensor:
        """Return the SCAFFOLD-corrected gradient for one parameter."""
        return grad - c_client_param + c_global_param

    def compute_new_control_variate(
        self,
        client_idx: int,
        global_state_dict: OrderedDict,
        local_state_dict: OrderedDict,
        num_local_steps: int,
    ) -> OrderedDict:
        """
        Option II (recommended): c_k^+ = c_k - c + (θ_global - θ_local) / (K·η)
        """
        new_ck = OrderedDict()
        for key in global_state_dict.keys():
            delta = (global_state_dict[key].float() -
                     local_state_dict[key].float()) / (num_local_steps * self.lr)
            new_ck[key] = (self.c_clients[client_idx][key] -
                           self.c_global[key] + delta)
        return new_ck

    def aggregate(
        self,
        global_state_dict: OrderedDict,
        client_state_dicts: List[OrderedDict],
        client_data_sizes: List[int],
        new_c_clients: List[OrderedDict],
        client_indices: List[int],
    ) -> OrderedDict:
        """
        Server-side SCAFFOLD aggregation.

        1. Weighted average of client state dicts (same as FedAvg).
        2. Update global control variate using delta_c_k.
        """
        # --- 1. Model aggregation (FedAvg) ---
        total = sum(client_data_sizes)
        weights = [ds / total for ds in client_data_sizes]

        aggregated = OrderedDict()
        for key in global_state_dict.keys():
            aggregated[key] = sum(
                w * sd[key].float()
                for w, sd in zip(weights, client_state_dicts)
            )

        # --- 2. Update c_global and per-client c_k ---
        K = self.num_clients
        for key in self.c_global.keys():
            delta_c = torch.zeros_like(self.c_global[key])
            for i, cidx in enumerate(client_indices):
                delta_c += (new_c_clients[i][key] -
                            self.c_clients[cidx][key])
            self.c_global[key] += delta_c / K

        for i, cidx in enumerate(client_indices):
            self.c_clients[cidx] = new_c_clients[i]

        return aggregated


# ============================================================
# 2. FedNova
# ============================================================

def fednova_aggregate(
    global_state_dict: OrderedDict,
    client_state_dicts: List[OrderedDict],
    client_data_sizes: List[int],
    client_local_steps: List[int],
) -> OrderedDict:
    """
    FedNova: Normalized Averaging.

    Instead of weighting by |D_k|/Σ|D_j|, FedNova normalizes the
    pseudo-gradient by the effective number of local SGD steps τ_k:

        d_k = (θ_global - θ_k) / τ_k          (normalized delta)
        d_global = Σ (p_k · d_k)               (p_k = |D_k|/Σ|D_j|)
        θ_global_new = θ_global - τ_eff · d_global

    where τ_eff = Σ p_k · τ_k.

    Args:
        global_state_dict: θ_global before this round
        client_state_dicts: θ_k after local training
        client_data_sizes: |D_k|
        client_local_steps: τ_k (total SGD steps per client this round)

    Returns:
        New aggregated global state dict
    """
    total_data = sum(client_data_sizes)
    p = [ds / total_data for ds in client_data_sizes]  # data-fraction weights

    # Effective number of steps
    tau_eff = sum(pk * tk for pk, tk in zip(p, client_local_steps))

    aggregated = OrderedDict()
    for key in global_state_dict.keys():
        g_param = global_state_dict[key].float()

        # Compute weighted normalized gradient
        d_global = torch.zeros_like(g_param)
        for k in range(len(client_state_dicts)):
            tau_k = client_local_steps[k]
            delta_k = (g_param - client_state_dicts[k][key].float()) / max(tau_k, 1)
            d_global += p[k] * delta_k

        aggregated[key] = g_param - tau_eff * d_global

    return aggregated


# ============================================================
# 3. FedAdam (FedOpt family)
# ============================================================

class FedAdamAggregator:
    """
    FedAdam: Server-side Adam optimizer on the pseudo-gradient.

    The server treats the FedAvg aggregation as producing a pseudo-gradient:
        Δ = θ_global - FedAvg(θ_k)
    and applies Adam to update the global model:
        m_t = β1 · m_{t-1} + (1-β1) · Δ
        v_t = β2 · v_{t-1} + (1-β2) · Δ²
        θ_{t+1} = θ_t - η_s · m_t / (√v_t + ε)
    """

    def __init__(
        self,
        lr: float = 1e-2,
        beta1: float = 0.9,
        beta2: float = 0.99,
        eps: float = 1e-3,
        tau: float = 1e-3,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.tau = tau

        self.m: Optional[OrderedDict] = None  # First moment
        self.v: Optional[OrderedDict] = None  # Second moment
        self.t = 0  # Timestep

    def aggregate(
        self,
        global_state_dict: OrderedDict,
        client_state_dicts: List[OrderedDict],
        client_data_sizes: List[int],
    ) -> OrderedDict:
        """
        1. Compute FedAvg pseudo-gradient Δ = θ_global - FedAvg(θ_k).
        2. Apply Adam update on the server.
        """
        self.t += 1

        total = sum(client_data_sizes)
        weights = [ds / total for ds in client_data_sizes]

        # --- 1. Compute FedAvg result ---
        fedavg_state = OrderedDict()
        for key in global_state_dict.keys():
            fedavg_state[key] = sum(
                w * sd[key].float()
                for w, sd in zip(weights, client_state_dicts)
            )

        # --- 2. Pseudo-gradient: Δ = θ_global - FedAvg ---
        delta = OrderedDict()
        for key in global_state_dict.keys():
            delta[key] = global_state_dict[key].float() - fedavg_state[key]

        # --- 3. Initialize moments if first round ---
        if self.m is None:
            self.m = OrderedDict({k: torch.zeros_like(v, dtype=torch.float32)
                                  for k, v in global_state_dict.items()})
            self.v = OrderedDict({k: torch.zeros_like(v, dtype=torch.float32)
                                  + self.tau
                                  for k, v in global_state_dict.items()})

        # --- 4. Adam update ---
        aggregated = OrderedDict()
        for key in global_state_dict.keys():
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * delta[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (delta[key] ** 2)

            # Bias-corrected moments
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            aggregated[key] = (global_state_dict[key].float()
                               - self.lr * m_hat / (torch.sqrt(v_hat) + self.eps))

        return aggregated


# ============================================================
# 4. MOON (Client-side contrastive loss)
# ============================================================

class MOONContrastiveLoss(nn.Module):
    """
    MOON: Model-Contrastive Federated Learning.

    Adds a contrastive loss to each client's local training objective:
        L_con = -log( exp(sim(z, z_global)/τ) /
                      (exp(sim(z, z_global)/τ) + exp(sim(z, z_prev)/τ)) )

    where z, z_global, z_prev are the feature representations from the
    current local model, the global model, and the previous-round local model.

    In our setting we use the hidden states from DialogueRNN as z.
    Aggregation remains standard FedAvg.

    Args:
        temperature: Temperature τ for contrastive similarity.
        mu: Weight of the contrastive loss relative to task loss.
    """

    def __init__(self, temperature: float = 0.5, mu: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.mu = mu

    def forward(
        self,
        z_local: torch.Tensor,
        z_global: torch.Tensor,
        z_prev: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute MOON contrastive loss.

        Args:
            z_local:  Hidden states from current local model  [B, T, D]
            z_global: Hidden states from frozen global model   [B, T, D]
            z_prev:   Hidden states from previous local model  [B, T, D]
            mask:     Valid utterance mask [B, T]

        Returns:
            Scalar contrastive loss
        """
        # Flatten valid utterances
        z_l = z_local[mask]   # [N, D]
        z_g = z_global[mask]  # [N, D]
        z_p = z_prev[mask]    # [N, D]

        if z_l.size(0) == 0:
            return torch.tensor(0.0, device=z_l.device)

        # Normalize
        z_l = F.normalize(z_l, dim=-1)
        z_g = F.normalize(z_g, dim=-1)
        z_p = F.normalize(z_p, dim=-1)

        # Cosine similarity
        sim_pos = (z_l * z_g).sum(dim=-1) / self.temperature  # [N]
        sim_neg = (z_l * z_p).sum(dim=-1) / self.temperature  # [N]

        # Contrastive loss: -log(exp(pos) / (exp(pos) + exp(neg)))
        logits = torch.stack([sim_pos, sim_neg], dim=-1)  # [N, 2]
        labels = torch.zeros(z_l.size(0), dtype=torch.long,
                             device=z_l.device)  # positive is index 0
        loss = F.cross_entropy(logits, labels)

        return self.mu * loss
