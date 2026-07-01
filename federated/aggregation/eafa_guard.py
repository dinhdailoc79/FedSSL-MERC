"""
EAFA-Guard: Direction-Filtered Uncertainty-Aware Federated Aggregation
========================================================================
Extends EAFA with server-side robustness against update-poisoning attacks.

The core EAFA weight exp(-β·u_k) is self-reported and therefore spoofable:
an adversary can submit a poisoned update while declaring low uncertainty.
EAFA-Guard adds two server-side checks that do NOT depend on the self-reported
scalar:

  1. Median-cosine direction filter: compute cosine similarity between each
     client update and a clean server-root reference; drop the lower half.
  2. Magnitude cap: clip surviving updates to the median survivor norm.

After filtering and capping, the evidential weight exp(-β·u_k) still rewards
honest high-quality clients among the survivors.

Reference: Section 3.5 of the manuscript (Eq. 7).
Ported from: testbed/fedsim.py  agg_eafa_guard (lines 317-341).
"""

import logging
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict

import torch
import numpy as np

from federated.aggregation.robust_aggregation import flatten_state_dict, unflatten_state_dict

logger = logging.getLogger(__name__)


def _compute_delta(client_state: OrderedDict, global_state: OrderedDict) -> OrderedDict:
    """Compute parameter update delta = client - global."""
    delta = OrderedDict()
    for key in client_state:
        delta[key] = client_state[key].float() - global_state[key].float()
    return delta


def eafa_guard_aggregate(
    client_state_dicts: List[OrderedDict],
    client_data_sizes: List[int],
    client_uncertainties: List[float],
    global_state_dict: OrderedDict,
    server_delta: Optional[OrderedDict] = None,
    beta: float = 4.0,
) -> Tuple[OrderedDict, Dict]:
    """
    EAFA-Guard aggregation.

    Args:
        client_state_dicts: List of model state dicts from each client
        client_data_sizes: |D_k| for each client
        client_uncertainties: u_k (quality scalar) for each client
        global_state_dict: Current global model state dict (reference point)
        server_delta: Server-root update from one step on clean root set.
                      If None, falls back to standard EAFA (no guard).
        beta: Temperature parameter for EAFA weighting

    Returns:
        (aggregated_state_dict, stats_dict)
    """
    K = len(client_state_dicts)
    ns = np.array(client_data_sizes, dtype=float)
    us = np.array(client_uncertainties, dtype=float)

    # Compute deltas: delta_k = theta_k - theta_global
    deltas = [_compute_delta(sd, global_state_dict) for sd in client_state_dicts]

    # If no server reference, fall back to standard EAFA
    if server_delta is None:
        logger.warning("EAFA-Guard: No server_delta provided, falling back to EAFA.")
        from federated.aggregation.eafa import eafa_aggregate
        agg_state = eafa_aggregate(
            client_state_dicts, client_data_sizes, client_uncertainties, beta=beta,
        )
        w = ns * np.exp(-beta * us)
        w /= w.sum()
        return agg_state, {"weights": w.tolist(), "guard_active": False}

    # Flatten all deltas and server delta
    device = next(iter(global_state_dict.values())).device
    flat_deltas = torch.stack([flatten_state_dict(d).to(device) for d in deltas])  # [K, D]
    flat_server = flatten_state_dict(server_delta).to(device)  # [D]

    # Norms
    norms = torch.norm(flat_deltas, dim=1)  # [K]
    server_norm = torch.norm(flat_server) + 1e-12

    # Cosine similarity to server root
    cosines = (flat_deltas @ flat_server) / (norms + 1e-12) / server_norm  # [K]
    cosines_np = cosines.cpu().numpy()

    # Median-cosine filter: keep upper half by direction alignment
    median_cos = float(np.median(cosines_np))
    keep_mask = cosines_np >= median_cos  # bool array [K]

    # Sharpened trust score: max(0, cos)^2
    trust = np.maximum(0.0, cosines_np) ** 2

    # Evidential quality weight
    ev_weight = np.exp(-beta * us)

    # Combined weight: zero out filtered clients
    w = np.where(keep_mask, trust * ev_weight * ns, 0.0)

    # Magnitude cap: clip to median survivor norm
    norms_np = norms.cpu().numpy()
    if keep_mask.sum() > 0:
        cap = float(np.median(norms_np[keep_mask]))
    else:
        cap = float(norms_np.max())
    scale = np.minimum(1.0, cap / (norms_np + 1e-12))

    # Fallback if all weights are zero
    if w.sum() <= 0:
        logger.warning("EAFA-Guard: All weights zero after filtering, falling back to uniform.")
        w = ns.copy()
    w = w / w.sum()

    # Weighted aggregation of scaled deltas
    scaled_flat = (torch.tensor(w * scale, dtype=torch.float32, device=device).unsqueeze(1)
                   * flat_deltas)
    agg_flat = scaled_flat.sum(dim=0)

    # Reconstruct global state: theta_global + aggregated_delta
    agg_delta = unflatten_state_dict(agg_flat, deltas[0])
    agg_state = OrderedDict()
    for key in global_state_dict:
        agg_state[key] = global_state_dict[key].float() + agg_delta[key].to(device)

    # Stats for logging
    stats = {
        "weights": w.tolist(),
        "cosines": cosines_np.tolist(),
        "median_cosine": median_cos,
        "kept_clients": int(keep_mask.sum()),
        "dropped_clients": int(K - keep_mask.sum()),
        "magnitude_cap": cap,
        "scale_factors": scale.tolist(),
        "guard_active": True,
        "beta": beta,
    }

    logger.info(
        f"  EAFA-Guard: kept {stats['kept_clients']}/{K} clients, "
        f"median_cos={median_cos:.3f}, cap={cap:.1f}, "
        f"w=[{', '.join(f'{wi:.3f}' for wi in w)}]"
    )

    return agg_state, stats


class EAFAGuardAggregator:
    """
    Stateful EAFA-Guard aggregator with history tracking.

    Usage:
        aggregator = EAFAGuardAggregator(beta=4.0)

        # Each round:
        server_delta = aggregator.compute_server_delta(
            global_model, root_loader, loss_fn, device, lr=1e-3
        )
        agg_state, stats = aggregator.aggregate(
            client_state_dicts, client_data_sizes, client_uncertainties,
            global_state_dict, server_delta
        )
    """

    def __init__(self, beta: float = 4.0):
        self.beta = beta
        self.round_history = []

    def compute_server_delta(
        self,
        global_model: torch.nn.Module,
        root_loader: torch.utils.data.DataLoader,
        loss_fn,
        device: str,
        lr: float = 1e-3,
    ) -> OrderedDict:
        """
        Compute server-root reference update by training for 1 epoch
        on a clean server root dataset using the same optimizer (Adam) as clients.

        Args:
            global_model: Current global model (will NOT be modified)
            root_loader: DataLoader for clean root set
            loss_fn: Loss function (SupervisedEvidentialLoss)
            device: Device string
            lr: Learning rate for the server step

        Returns:
            server_delta: OrderedDict of parameter updates
        """
        import copy

        ref_model = copy.deepcopy(global_model).to(device)
        ref_model.train()
        # Use Adam with weight decay to match client optimization dynamics
        optimizer = torch.optim.Adam(ref_model.parameters(), lr=lr, weight_decay=1e-4)

        for batch in root_loader:
            features = batch["features"].to(device)
            speakers = batch["speaker_ids"].to(device)
            labels = batch["labels"].to(device)

            out = ref_model(features, speakers)
            mask = labels != -1
            if mask.sum() == 0:
                continue

            loss, _ = loss_fn(out["alpha"][mask], labels[mask])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ref_model.parameters(), 5.0)
            optimizer.step()

        # server_delta = ref_model - global_model
        global_state = {k: v.cpu() for k, v in global_model.state_dict().items()}
        ref_state = {k: v.cpu() for k, v in ref_model.state_dict().items()}
        server_delta = _compute_delta(OrderedDict(ref_state), OrderedDict(global_state))

        del ref_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return server_delta

    def aggregate(
        self,
        client_state_dicts: List[OrderedDict],
        client_data_sizes: List[int],
        client_uncertainties: List[float],
        global_state_dict: OrderedDict,
        server_delta: Optional[OrderedDict] = None,
        round_num: int = 0,
    ) -> Tuple[OrderedDict, Dict]:
        """Perform EAFA-Guard aggregation for one round."""
        agg_state, stats = eafa_guard_aggregate(
            client_state_dicts,
            client_data_sizes,
            client_uncertainties,
            global_state_dict,
            server_delta=server_delta,
            beta=self.beta,
        )
        stats["round"] = round_num
        self.round_history.append(stats)
        return agg_state, stats
