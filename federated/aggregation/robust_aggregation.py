import logging
from collections import OrderedDict
from typing import List
import numpy as np
import torch

logger = logging.getLogger(__name__)


def flatten_state_dict(state_dict: OrderedDict) -> torch.Tensor:
    """Flatten all tensors in a state dict into a single 1D tensor."""
    tensors = [val.flatten() for val in state_dict.values()]
    return torch.cat(tensors)


def unflatten_state_dict(flat_tensor: torch.Tensor, template_state_dict: OrderedDict) -> OrderedDict:
    """Unflatten a 1D tensor back into the structure of a template state dict."""
    new_state = OrderedDict()
    offset = 0
    for key, val in template_state_dict.items():
        numel = val.numel()
        new_state[key] = flat_tensor[offset : offset + numel].view_as(val).clone()
        offset += numel
    return new_state


class RobustAggregator:
    """
    Robust Federated Learning Aggregators.
    Implements Median, Trimmed Mean, Krum, and Multi-Krum.
    """

    def __init__(self, strategy: str = "median", f: int = 1, trim_ratio: float = 0.2):
        self.strategy = strategy.lower()
        self.f = f  # Assumed number of Byzantine/compromised clients
        self.trim_ratio = trim_ratio

    def aggregate(self, client_states: List[OrderedDict], client_sizes: List[int]) -> OrderedDict:
        """
        Aggregate client weights using the specified robust aggregation strategy.
        """
        if not client_states:
            raise ValueError("Empty list of client states provided for aggregation.")

        template = client_states[0]
        device = template[next(iter(template))].device

        # Flatten all client state dicts
        flat_states = [flatten_state_dict(sd).to(device) for sd in client_states]
        X = torch.stack(flat_states)  # Shape: [K, D]
        K, D = X.shape

        logger.info(f"RobustAggregator: Aggregating K={K} client updates (D={D}) using strategy='{self.strategy}'")

        if self.strategy == "median":
            # Coordinate-wise Median
            flat_agg = torch.median(X, dim=0).values
        elif self.strategy == "trimmed_mean":
            # Coordinate-wise Trimmed Mean (discard top and bottom f elements)
            # trim_ratio can also be used, but to compare with Krum (where f is known), we use f
            sorted_X, _ = torch.sort(X, dim=0)
            if 2 * self.f >= K:
                logger.warning(f"Number of Byzantine clients f={self.f} is too large for K={K}. Falling back to standard mean.")
                flat_agg = X.mean(dim=0)
            else:
                flat_agg = sorted_X[self.f : K - self.f].mean(dim=0)
        elif self.strategy == "krum":
            # Krum Aggregation (selects the client update closest to its K - f - 2 nearest neighbors)
            if K - self.f - 2 <= 0:
                logger.warning(f"Krum condition K - f - 2 > 0 violated (K={K}, f={self.f}). Falling back to coordinate-wise median.")
                flat_agg = torch.median(X, dim=0).values
            else:
                # Compute pairwise Euclidean distances squared
                dists = torch.cdist(X, X, p=2) ** 2  # Shape: [K, K]
                scores = []
                for i in range(K):
                    sorted_dists, _ = torch.sort(dists[i])
                    # Exclude the distance to itself (first element, which is 0)
                    scores.append(sorted_dists[1 : K - self.f - 1].sum().item())
                best_idx = np.argmin(scores)
                logger.info(f"  Krum selected client index {best_idx} with score {scores[best_idx]:.4f}")
                flat_agg = X[best_idx]
        elif self.strategy == "multi_krum":
            # Multi-Krum Aggregation (averages K - f - 2 client updates with the lowest Krum scores)
            num_select = K - self.f - 2
            if num_select <= 0:
                logger.warning(f"Multi-Krum condition K - f - 2 > 0 violated (K={K}, f={self.f}). Falling back to median.")
                flat_agg = torch.median(X, dim=0).values
            else:
                dists = torch.cdist(X, X, p=2) ** 2
                scores = []
                for i in range(K):
                    sorted_dists, _ = torch.sort(dists[i])
                    scores.append(sorted_dists[1 : K - self.f - 1].sum().item())
                best_indices = np.argsort(scores)[:num_select]
                logger.info(f"  Multi-Krum selected client indices {best_indices.tolist()} to average.")
                flat_agg = X[best_indices].mean(dim=0)
        else:
            raise ValueError(f"Unknown robust aggregation strategy: {self.strategy}")

        return unflatten_state_dict(flat_agg, template)
