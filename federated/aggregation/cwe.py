"""
CWE: Class-Wise Epistemic Uncertainty-Aware Federated Aggregation
==================================================================
Implements Class-Wise Epistemic uncertainty aggregation (CWE) for FedSSL-MERC.

Class-wise weights for class 'c' of client 'k':
    w_{k,c} = |D_k| · exp(-β · u_{k,c}) / Σ_j |D_j| · exp(-β · u_{j,c})

For parameters in classifier head and projector associated with class 'c':
    θ_global[c, :] = Σ_k w_{k,c} · θ_k[c, :]

For all other shared parameters (e.g., backbone), we fall back to client-wise
average uncertainty weighting (EAFA):
    w_k = |D_k| · exp(-β · ū_k) / Σ_j |D_j| · exp(-β · ū_j)
    θ_global = Σ_k w_k · θ_k
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from collections import OrderedDict

import torch
import numpy as np

logger = logging.getLogger(__name__)


def cwe_aggregate(
    client_state_dicts: List[OrderedDict],
    client_data_sizes: List[int],
    client_classwise_uncertainties: List[Union[List[float], np.ndarray, torch.Tensor]],
    beta: float = 1.0,
    head_prefix: str = "head.",
    projector_prefix: str = "backbone.projector.",
) -> OrderedDict:
    """
    Class-Wise Epistemic Uncertainty-Aware Federated Aggregation.

    Args:
        client_state_dicts: List of model state dicts from each client.
        client_data_sizes: |D_k| (sample counts) for each client.
        client_classwise_uncertainties: List of class-wise uncertainty vectors (length C) for each client.
        beta: Temperature parameter controlling sensitivity to uncertainty.
        head_prefix: Prefix identifying the classifier head parameters in state dict.
        projector_prefix: Prefix identifying the projector parameters in state dict.

    Returns:
        Aggregated global state dict.
    """
    num_clients = len(client_state_dicts)
    assert num_clients == len(client_data_sizes) == len(client_classwise_uncertainties)

    # Convert uncertainties to numpy arrays for safety
    cwu_list = [np.array(u) for u in client_classwise_uncertainties]
    num_classes = len(cwu_list[0])

    # 1. Compute Class-Wise Weights: shape [num_classes, num_clients]
    class_weights = np.zeros((num_classes, num_clients))
    for c in range(num_classes):
        raw_c_weights = []
        for k in range(num_clients):
            w = client_data_sizes[k] * np.exp(-beta * cwu_list[k][c])
            raw_c_weights.append(w)
        
        total_c = sum(raw_c_weights)
        if total_c > 0:
            class_weights[c, :] = [w / total_c for w in raw_c_weights]
        else:
            # Fallback to uniform weights if total weight is 0
            class_weights[c, :] = [1.0 / num_clients] * num_clients

    # 2. Compute Client-Wise EAFA Weights (fallback for shared backbone parameters)
    # ū_k = average uncertainty across classes for client k
    client_uncertainties = [np.mean(cwu) for cwu in cwu_list]
    raw_eafa_weights = []
    for k in range(num_clients):
        w = client_data_sizes[k] * np.exp(-beta * client_uncertainties[k])
        raw_eafa_weights.append(w)
    
    total_eafa = sum(raw_eafa_weights)
    if total_eafa > 0:
        eafa_weights = [w / total_eafa for w in raw_eafa_weights]
    else:
        eafa_weights = [1.0 / num_clients] * num_clients

    # 3. Perform Aggregation
    global_state = OrderedDict()
    keys = client_state_dicts[0].keys()

    for key in keys:
        param_shape = client_state_dicts[0][key].shape
        is_class_specific = False

        # Check if the parameter belongs to head or projector and has shape matching num_classes
        if (key.startswith(head_prefix) or key.startswith(projector_prefix)) and len(param_shape) > 0:
            if param_shape[0] == num_classes:
                is_class_specific = True

        if is_class_specific:
            # Class-wise aggregation
            logger.debug(f"CWE: Performing class-wise aggregation for {key} with shape {param_shape}")
            aggregated_tensor = torch.zeros_like(client_state_dicts[0][key], dtype=torch.float32)
            
            if len(param_shape) == 1:
                # 1D Bias or vector of shape [num_classes]
                for c in range(num_classes):
                    aggregated_tensor[c] = sum(
                        class_weights[c, k] * client_state_dicts[k][key][c].float()
                        for k in range(num_clients)
                    )
            elif len(param_shape) == 2:
                # 2D Weight matrix of shape [num_classes, input_dim]
                for c in range(num_classes):
                    aggregated_tensor[c, :] = sum(
                        class_weights[c, k] * client_state_dicts[k][key][c, :].float()
                        for k in range(num_clients)
                    )
            else:
                # Higher-dimensional class-specific tensor (fallback to slice aggregation)
                for c in range(num_classes):
                    aggregated_tensor[c, ...] = sum(
                        class_weights[c, k] * client_state_dicts[k][key][c, ...].float()
                        for k in range(num_clients)
                    )
            global_state[key] = aggregated_tensor
        else:
            # Client-wise EAFA aggregation for shared parameters
            global_state[key] = sum(
                eafa_weights[k] * client_state_dicts[k][key].float()
                for k in range(num_clients)
            )

    return global_state


class CWEAggregator:
    """
    Stateful CWE aggregator with history tracking and adaptive beta.
    """

    def __init__(
        self,
        beta: float = 1.0,
        adaptive_beta: bool = False,
        beta_min: float = 0.1,
        beta_max: float = 5.0,
        head_prefix: str = "head.",
        projector_prefix: str = "backbone.projector.",
    ):
        self.beta = beta
        self.adaptive_beta = adaptive_beta
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.head_prefix = head_prefix
        self.projector_prefix = projector_prefix

        # History
        self.round_history = []

    def aggregate(
        self,
        client_state_dicts: List[OrderedDict],
        client_data_sizes: List[int],
        client_classwise_uncertainties: List[Union[List[float], np.ndarray]],
        round_num: int = 0,
    ) -> Tuple[OrderedDict, Dict]:
        """
        Perform CWE aggregation for one round.
        """
        cwu_arrays = [np.array(u) for u in client_classwise_uncertainties]
        num_clients = len(client_state_dicts)

        # Adaptive beta: dynamic scaling based on uncertainty variance
        if self.adaptive_beta and num_clients > 1:
            # Standard deviation across client average uncertainties
            u_means = [np.mean(cwu) for cwu in cwu_arrays]
            u_std = np.std(u_means)
            self.beta = np.clip(
                1.0 + 5.0 * u_std,
                self.beta_min,
                self.beta_max,
            )

        # Run aggregation
        global_state = cwe_aggregate(
            client_state_dicts,
            client_data_sizes,
            client_classwise_uncertainties,
            beta=self.beta,
            head_prefix=self.head_prefix,
            projector_prefix=self.projector_prefix,
        )

        # Log statistics
        u_means = [np.mean(cwu) for cwu in cwu_arrays]
        stats = {
            "beta": self.beta,
            "mean_uncertainty": np.mean(u_means),
            "std_uncertainty": np.std(u_means) if num_clients > 1 else 0,
        }
        self.round_history.append(stats)

        logger.info(
            f"  CWE: beta={self.beta:.2f}, "
            f"u_mean={stats['mean_uncertainty']:.4f}, "
            f"u_std={stats['std_uncertainty']:.4f}"
        )

        return global_state, stats
