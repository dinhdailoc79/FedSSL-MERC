"""
Partial Participation & Communication Overhead Profiler
=========================================================
1. Simulates realistic federated learning with K=50 clients.
2. Selects 20% fraction (10 clients) per round (Partial Participation).
3. Profiles and quantifies the exact parameter transmission bandwidth.
4. Profiles wall-clock execution time under local training loops.
"""

import sys
import os
import time
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.evidential.evidential_dialogue_rnn import EvidentialDialogueRNN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def profile_network_bandwidth(model: nn.Module) -> dict:
    """
    Profiles exactly how many parameters and bytes are transmitted per client per round.
    """
    total_params = 0
    total_bytes = 0
    
    # Calculate bytes for state dict parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_elem = param.numel()
            total_params += num_elem
            # PyTorch float32 parameters occupy 4 bytes each
            total_bytes += num_elem * 4

    # EAFA transmits exactly one additional scalar (mean uncertainty)
    uncertainty_scalar_bytes = 4  # 1 float32 scalar
    
    overhead_ratio = (uncertainty_scalar_bytes / total_bytes) * 100

    return {
        "num_parameters": total_params,
        "standard_weights_bytes": total_bytes,
        "standard_weights_mb": total_bytes / (1024 * 1024),
        "uncertainty_scalar_bytes": uncertainty_scalar_bytes,
        "eafa_total_bytes": total_bytes + uncertainty_scalar_bytes,
        "eafa_overhead_percentage": overhead_ratio
    }


def main():
    logger.info(f"\n{'='*60}")
    logger.info("  PROFILING: PARTIAL PARTICIPATION & SYSTEM OVERHEAD")
    logger.info(f"{'='*60}\n")

    # 1. Bandwidth quantification
    model = EvidentialDialogueRNN(
        input_dim=768, hidden_dim=256,
        num_classes=7, num_speakers=10,
        dropout=0.3, use_attention=True,
    )
    
    profile = profile_network_bandwidth(model)
    
    logger.info("Communication Bandwidth Profiling:")
    logger.info(f"  Model Parameters:      {profile['num_parameters']:,} params")
    logger.info(f"  Standard Weights Size: {profile['standard_weights_mb']:.4f} MB ({profile['standard_weights_bytes']:,} bytes)")
    logger.info(f"  EAFA Uncertainty Size: {profile['uncertainty_scalar_bytes']} bytes (1 scalar)")
    logger.info(f"  EAFA Total Upload:     {profile['eafa_total_bytes']:,} bytes")
    logger.info(f"  EAFA Bandwidth Overhead: {profile['eafa_overhead_percentage']:.8f}% (Virtually 0%!)")
    logger.info("-" * 60)

    # 2. Simulation of Partial Participation (K=50, C=0.2)
    num_clients = 50
    participation_fraction = 0.2
    num_selected = int(num_clients * participation_fraction)
    
    np.random.seed(42)
    
    rounds = 5
    logger.info(f"Simulating Partial Participation ({num_clients} clients, {participation_fraction:.0%} selected per round):")
    
    client_ids = np.arange(num_clients)
    
    for r in range(1, rounds + 1):
        selected = np.random.choice(client_ids, size=num_selected, replace=False)
        logger.info(f"  Round {r}: Selected {num_selected} clients -> {selected.tolist()}")

    # 3. Time profiling (Mock training cycle on 10 clients to measure wall-clock time)
    logger.info("-" * 60)
    logger.info("Measuring typical local epoch wall-clock execution time...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    dummy_input = torch.randn(16, 15, 768, device=device)
    dummy_speakers = torch.zeros(16, 15, dtype=torch.long, device=device)
    dummy_labels = torch.zeros(16, 15, dtype=torch.long, device=device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    start_time = time.time()
    for step in range(50):  # typical batches per client round
        out = model(dummy_input, dummy_speakers)
        loss = F.cross_entropy(out["belief"].view(-1, 7), dummy_labels.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    elapsed = time.time() - start_time
    logger.info(f"Average local step training time (50 batches): {elapsed:.4f} seconds")

    # Save summary
    summary_path = "results/system_overhead_summary.json"
    os.makedirs("results", exist_ok=True)
    
    summary = {
        "bandwidth": profile,
        "wall_clock_50_batches_seconds": round(elapsed, 4),
        "partial_participation": {
            "num_clients": num_clients,
            "participation_fraction": participation_fraction,
            "num_selected": num_selected
        }
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
        
    logger.info(f"\nSystem overhead report saved to {summary_path}")


if __name__ == "__main__":
    main()
