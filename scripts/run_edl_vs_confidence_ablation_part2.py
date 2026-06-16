"""
EDL vs Softmax Confidence/Entropy Ablation Study (Part 2)
=========================================================
Runs CE_FedAvg and CE_EAFA_Confidence to parallelize the ablation study.
Saves to results_edl_vs_confidence_ablation_part2.json.
"""

import sys
import os
import json
import time
import numpy as np
import torch
from argparse import Namespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.train_multi_dataset import load_meld, train_federated

RESULTS_FILE = "results_edl_vs_confidence_ablation_part2.json"


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, 'item') else str(x))


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    results = load_results()
    
    seeds = [42, 123, 2024]
    configs = [
        {"name": "CE_FedAvg", "loss_type": "ce", "aggregation": "fedavg", "uncertainty_type": "confidence"},
        {"name": "CE_EAFA_Confidence", "loss_type": "ce", "aggregation": "eafa", "uncertainty_type": "confidence"}
    ]

    # Load MELD dataset
    logger.info("Loading MELD dataset...")
    train, dev, test, emotions, weights, cache, num_spk = load_meld(finetuned=True)

    experiments = []
    for cfg in configs:
        for seed in seeds:
            key = f"{cfg['name']}_s{seed}"
            experiments.append((key, cfg, seed))

    total = len(experiments)
    logger.info(f"Total experiments to run: {total}")

    for idx, (key, cfg, seed) in enumerate(experiments):
        if key in results and results[key].get("wf1") is not None:
            logger.info(f"[{idx+1}/{total}] Skipping {key} (already completed: WF1={results[key]['wf1']:.4f})")
            continue

        logger.info(f"\n[{idx+1}/{total}] Running {key}...")
        
        # Reset seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Set up arguments
        args = Namespace(
            hidden_dim=256,
            dropout=0.3,
            epochs=80,
            batch_size=16,
            lr=1e-3,
            annealing_epochs=30,
            patience=20,
            num_clients=5,
            alpha=0.5,
            num_rounds=50,
            local_epochs=3,
            beta=1.0,
            mu=0.0,
            loss_type=cfg["loss_type"],
            aggregation=cfg["aggregation"],
            uncertainty_type=cfg["uncertainty_type"],
            focal_gamma=0.0,
            device="cuda" if torch.cuda.is_available() else "cpu",
            save_dir="checkpoints",
            seed=seed,
            finetuned=True
        )

        start_time = time.time()
        try:
            wf1, mean_u, _ = train_federated(
                "meld", train, dev, test, emotions, weights, cache, num_spk, args
            )
            elapsed = time.time() - start_time
            
            results[key] = {
                "wf1": round(wf1, 4),
                "mean_u": round(mean_u, 4),
                "config": cfg["name"],
                "seed": seed,
                "time_sec": round(elapsed, 1)
            }
            save_results(results)
            logger.info(f"Finished {key} in {elapsed:.1f}s. WF1={wf1:.4f}")
        except Exception as e:
            logger.exception(f"Error running {key}: {e}")
            results[key] = {"error": str(e)}
            save_results(results)

    logger.info("Part 2 Completed!")


if __name__ == "__main__":
    main()
