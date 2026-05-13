"""
FedProx — Federated Proximal
==============================
Li et al., "Federated Optimization in Heterogeneous Networks", MLSys 2020.

Adds a proximal regularization term to each client's local objective,
penalizing deviation from the global model. This stabilizes training
under data heterogeneity (Non-IID).

    L_client = L_task + (mu/2) * ||w - w_global||^2
"""

import torch
import torch.nn as nn
from typing import Dict


class FedProxLoss(nn.Module):
    """
    Proximal term for FedProx.

    Adds ||w - w_global||^2 penalty to prevent clients
    from drifting too far from the global model.

    Args:
        mu: Proximal coefficient. Higher = stronger regularization.
            - mu=0: equivalent to FedAvg
            - mu=0.001: light regularization
            - mu=0.01: moderate (recommended for Non-IID)
            - mu=0.1: strong regularization
    """

    def __init__(self, mu: float = 0.01):
        super().__init__()
        self.mu = mu

    def forward(
        self,
        local_model: nn.Module,
        global_params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute proximal term: (mu/2) * sum(||w_local - w_global||^2)

        Args:
            local_model: Client's local model (being trained)
            global_params: Frozen copy of global model parameters

        Returns:
            Proximal loss (scalar)
        """
        prox_loss = torch.tensor(0.0, device=next(local_model.parameters()).device)

        for name, param in local_model.named_parameters():
            if name in global_params:
                diff = param - global_params[name].to(param.device)
                prox_loss += torch.sum(diff ** 2)

        return (self.mu / 2.0) * prox_loss
