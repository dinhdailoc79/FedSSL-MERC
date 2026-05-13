"""
FedAvg — Federated Averaging
==============================
McMahan et al., "Communication-Efficient Learning of Deep Networks
from Decentralized Data", AISTATS 2017.

The simplest and most widely-used FL aggregation strategy.
Server averages client model parameters weighted by number of samples.
"""

import copy
from typing import List, Dict, Tuple
from collections import OrderedDict

import torch
import torch.nn as nn


def fedavg_aggregate(
    global_model: nn.Module,
    client_models: List[nn.Module],
    client_weights: List[float],
) -> nn.Module:
    """
    Aggregate client models using FedAvg (weighted parameter averaging).

    Args:
        global_model: The global model to update in-place
        client_models: List of trained client models
        client_weights: Weight for each client (typically num_samples / total_samples)

    Returns:
        Updated global model
    """
    # Normalize weights
    total_weight = sum(client_weights)
    weights = [w / total_weight for w in client_weights]

    # Get global state dict
    global_dict = global_model.state_dict()

    # Weighted average of all parameters
    for key in global_dict.keys():
        global_dict[key] = torch.zeros_like(global_dict[key], dtype=torch.float32)
        for client_model, weight in zip(client_models, weights):
            client_dict = client_model.state_dict()
            global_dict[key] += weight * client_dict[key].float()

    global_model.load_state_dict(global_dict)
    return global_model


def fedavg_aggregate_state_dicts(
    global_state_dict: OrderedDict,
    client_state_dicts: List[OrderedDict],
    client_weights: List[float],
) -> OrderedDict:
    """
    FedAvg on raw state dicts (no model objects needed).

    Args:
        global_state_dict: Current global model state dict
        client_state_dicts: List of client state dicts after local training
        client_weights: Weight per client

    Returns:
        Aggregated state dict
    """
    total_weight = sum(client_weights)
    weights = [w / total_weight for w in client_weights]

    aggregated = OrderedDict()
    for key in global_state_dict.keys():
        aggregated[key] = torch.zeros_like(global_state_dict[key], dtype=torch.float32)
        for sd, w in zip(client_state_dicts, weights):
            aggregated[key] += w * sd[key].float()

    return aggregated
