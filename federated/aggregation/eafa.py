"""
EAFA: Epistemic-Aware Federated Aggregation
=============================================
Replaces standard FedAvg with uncertainty-weighted aggregation.

w_k = |D_k| · exp(-β · ū_k) / Σ_j |D_j| · exp(-β · ū_j)
θ_global = Σ_k w_k · θ_k

Key properties:
- When β=0: degenerates to standard FedAvg (volume-only)
- When |D_k| uniform: degenerates to pure uncertainty weighting
- Clients with high uncertainty are downweighted → defends against poisoned updates
- |D_k| prior prevents small-but-certain clients from dominating

Privacy: Only two scalars (ū_k, |D_k|) communicated per client per round.
"""

import logging
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict

import torch
import numpy as np

logger = logging.getLogger(__name__)


def eafa_aggregate(
    client_state_dicts: List[OrderedDict],
    client_data_sizes: List[int],
    client_uncertainties: List[float],
    beta: float = 1.0,
) -> OrderedDict:
    """
    Epistemic-Aware Federated Aggregation.

    Args:
        client_state_dicts: List of model state dicts from each client
        client_data_sizes: |D_k| for each client
        client_uncertainties: ū_k (mean epistemic uncertainty) for each client
        beta: Temperature parameter controlling uncertainty sensitivity
              β=0 → standard FedAvg, larger β → more uncertainty-sensitive

    Returns:
        Aggregated global state dict
    """
    num_clients = len(client_state_dicts)
    assert num_clients == len(client_data_sizes) == len(client_uncertainties)

    # Compute EAFA weights: w_k = |D_k| · exp(-β · ū_k)
    raw_weights = []
    for k in range(num_clients):
        w = client_data_sizes[k] * np.exp(-beta * client_uncertainties[k])
        raw_weights.append(w)

    # Normalize
    total = sum(raw_weights)
    weights = [w / total for w in raw_weights]

    # Log weights
    for k in range(num_clients):
        logger.debug(
            f"  Client {k}: |D|={client_data_sizes[k]}, "
            f"u={client_uncertainties[k]:.4f}, "
            f"w={weights[k]:.4f}"
        )

    # Weighted aggregation: θ_global = Σ w_k · θ_k
    global_state = OrderedDict()
    for key in client_state_dicts[0].keys():
        global_state[key] = sum(
            weights[k] * client_state_dicts[k][key].float()
            for k in range(num_clients)
        )

    return global_state


class EAFAAggregator:
    """
    Stateful EAFA aggregator with history tracking.

    Wraps eafa_aggregate with round-by-round logging and
    optional adaptive beta scheduling.
    """

    def __init__(
        self,
        beta: float = 1.0,
        adaptive_beta: bool = False,
        beta_min: float = 0.1,
        beta_max: float = 5.0,
    ):
        self.beta = beta
        self.adaptive_beta = adaptive_beta
        self.beta_min = beta_min
        self.beta_max = beta_max

        # History
        self.round_history = []

    def aggregate(
        self,
        client_state_dicts: List[OrderedDict],
        client_data_sizes: List[int],
        client_uncertainties: List[float],
        round_num: int = 0,
    ) -> Tuple[OrderedDict, Dict]:
        """
        Perform EAFA aggregation for one round.

        Returns:
            global_state_dict, round_stats
        """
        # Adaptive beta: increase when uncertainty variance is high
        if self.adaptive_beta and len(client_uncertainties) > 1:
            u_std = np.std(client_uncertainties)
            # Higher variance → higher beta (more differentiation)
            self.beta = np.clip(
                1.0 + 5.0 * u_std,
                self.beta_min,
                self.beta_max,
            )

        # Aggregate
        global_state = eafa_aggregate(
            client_state_dicts,
            client_data_sizes,
            client_uncertainties,
            beta=self.beta,
        )

        # Compute weights for logging
        raw_weights = [
            ds * np.exp(-self.beta * u)
            for ds, u in zip(client_data_sizes, client_uncertainties)
        ]
        total = sum(raw_weights)
        weights = [w / total for w in raw_weights]

        # Stats
        stats = {
            "beta": self.beta,
            "weights": weights,
            "data_sizes": client_data_sizes,
            "uncertainties": client_uncertainties,
            "mean_uncertainty": np.mean(client_uncertainties),
            "std_uncertainty": np.std(client_uncertainties) if len(client_uncertainties) > 1 else 0,
            "max_weight": max(weights),
            "min_weight": min(weights),
        }

        self.round_history.append(stats)

        logger.info(
            f"  EAFA: beta={self.beta:.2f}, "
            f"u_mean={stats['mean_uncertainty']:.4f}, "
            f"u_std={stats['std_uncertainty']:.4f}, "
            f"w_range=[{stats['min_weight']:.3f}, {stats['max_weight']:.3f}]"
        )

        return global_state, stats
