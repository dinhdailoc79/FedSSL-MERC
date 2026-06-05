"""
Federated Uncertainty-Aware Aggregation Baselines
===================================================
Implements:
1. FedEU (Evidential Uncertainty-guided aggregation / Top-k selection)
"""

import logging
from typing import Dict, List, Tuple
from collections import OrderedDict

import torch
import numpy as np

logger = logging.getLogger(__name__)


def fedeu_aggregate_state_dicts(
    global_state_dict: OrderedDict,
    client_state_dicts: List[OrderedDict],
    client_data_sizes: List[int],
    client_uncertainties: List[float],
    keep_ratio: float = 0.8,
) -> OrderedDict:
    """
    FedEU: Evidential Uncertainty-guided aggregation (Top-k selection style).
    
    Excludes the most uncertain clients (highest epistemic uncertainty) from aggregation,
    then performs volume-weighted FedAvg on the remaining clients.

    Args:
        global_state_dict: Current global model state dict
        client_state_dicts: List of client state dicts
        client_data_sizes: |D_k| for each client
        client_uncertainties: ū_k (mean epistemic uncertainty) for each client
        keep_ratio: Fraction of clients with lowest uncertainty to retain (default: 0.8)

    Returns:
        Aggregated state dict
    """
    num_clients = len(client_state_dicts)
    assert num_clients == len(client_data_sizes) == len(client_uncertainties)
    
    # Number of clients to keep
    k = max(1, int(num_clients * keep_ratio))
    
    # Sort clients by uncertainty ascending (least uncertain first)
    sorted_indices = np.argsort(client_uncertainties)
    selected_indices = sorted_indices[:k]
    
    logger.info(
        f"  FedEU: Selected {len(selected_indices)}/{num_clients} clients with lowest uncertainty. "
        f"Excluded client indices: {sorted_indices[k:].tolist()} with uncertainties "
        f"{[client_uncertainties[i] for i in sorted_indices[k:]]}"
    )
    
    # Weighted FedAvg on selected clients
    selected_data_sizes = [client_data_sizes[i] for i in selected_indices]
    total_size = sum(selected_data_sizes)
    weights = [ds / total_size for ds in selected_data_sizes]
    
    aggregated = OrderedDict()
    for key in global_state_dict.keys():
        aggregated[key] = torch.zeros_like(global_state_dict[key], dtype=torch.float32)
        for idx, w in zip(selected_indices, weights):
            aggregated[key] += w * client_state_dicts[idx][key].float().to(global_state_dict[key].device)
            
    return aggregated
