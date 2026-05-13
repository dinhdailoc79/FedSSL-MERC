"""
Federated Training Script
============================
Train DialogueRNN in a federated learning setting using FedAvg or FedProx.

Usage:
    python scripts/train_federated.py --strategy fedavg --num_clients 5 --alpha 0.5
    python scripts/train_federated.py --strategy fedprox --mu 0.01 --alpha 0.1
    python scripts/train_federated.py --strategy fedavg --alpha 1.0  # Nearly IID
"""

import sys
import os
import time
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.datasets.meld import MELDDataset, MELD_EMOTIONS
from data.federated_partition import FederatedPartitioner
from models.erc.dialogue_rnn import DialogueRNN
from federated.client import FederatedClient
from federated.server import FederatedServer

# Reuse from centralized script
from scripts.train_centralized import DialogueDataset, collate_dialogues

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Federated training for DialogueRNN on MELD")
    # Data
    parser.add_argument("--data_dir", type=str, default="data/raw/MELD")
    parser.add_argument("--feature_cache", type=str, default="data/features/meld_text_roberta.pt")
    # FL settings
    parser.add_argument("--strategy", type=str, default="fedavg", choices=["fedavg", "fedprox"])
    parser.add_argument("--num_clients", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Dirichlet alpha (lower=more Non-IID)")
    parser.add_argument("--num_rounds", type=int, default=50)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--label_ratio", type=float, default=1.0,
                        help="Fraction of labeled data per client (1.0=fully supervised)")
    parser.add_argument("--mu", type=float, default=0.01, help="FedProx mu")
    # Model
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--text_dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.3)
    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logger.info(f"{'='*60}")
    logger.info(f"  Federated Training — {args.strategy.upper()}")
    logger.info(f"  {args.num_clients} clients, alpha={args.alpha}, "
                f"local_epochs={args.local_epochs}")
    logger.info(f"{'='*60}")
    logger.info(f"Config: {vars(args)}")

    # -------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------
    logger.info("\nLoading MELD dataset...")
    meld = MELDDataset(data_dir=args.data_dir)
    train_dialogues = meld.get_dialogues("train")
    test_dialogues = meld.get_dialogues("test")
    logger.info(f"Train: {len(train_dialogues)} dialogues")
    logger.info(f"Test:  {len(test_dialogues)} dialogues")

    # Load cached features
    feature_cache = {}
    feature_path = Path(args.feature_cache)
    if feature_path.exists():
        logger.info(f"Loading cached features from {feature_path}...")
        cached = torch.load(feature_path, weights_only=False)
        for split in ["train", "test"]:
            if split in cached:
                feats = cached[split]["features"].numpy()
                dia_ids = cached[split]["dialogue_ids"]
                utt_ids = cached[split]["utterance_ids"]
                cache = {}
                for i in range(len(feats)):
                    key = f"{dia_ids[i].item()}_{utt_ids[i].item()}"
                    cache[key] = feats[i]
                feature_cache[split] = cache
                logger.info(f"  {split}: {len(cache)} features loaded")
        args.text_dim = feats.shape[1]
    else:
        logger.warning(f"No feature cache at {feature_path}!")

    # -------------------------------------------------------
    # 2. Partition data across clients
    # -------------------------------------------------------
    logger.info(f"\nPartitioning data: {args.num_clients} clients, "
                f"Dirichlet alpha={args.alpha}")
    partitioner = FederatedPartitioner(
        num_clients=args.num_clients,
        strategy="dirichlet",
        alpha=args.alpha,
        seed=args.seed,
    )
    client_partitions = partitioner.partition(
        train_dialogues,
        label_ratio=args.label_ratio,
    )
    partitioner.print_partition(client_partitions)

    # -------------------------------------------------------
    # 3. Create client DataLoaders
    # -------------------------------------------------------
    # Build dialogue lookup by ID
    dialogue_lookup = {d.dialogue_id: d for d in train_dialogues}

    clients = []
    for partition in client_partitions:
        # Get this client's dialogues
        client_dialogues = [
            dialogue_lookup[did] for did in partition.dialogue_ids
            if did in dialogue_lookup
        ]

        if not client_dialogues:
            logger.warning(f"Client {partition.client_id}: no dialogues, skipping")
            continue

        # Create dataset with features
        client_dataset = DialogueDataset(
            client_dialogues,
            feature_cache=feature_cache.get("train", {}),
            text_dim=args.text_dim,
        )

        client_loader = DataLoader(
            client_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_dialogues,
            num_workers=0,
        )

        client = FederatedClient(
            client_id=partition.client_id,
            dataloader=client_loader,
            device=args.device,
            local_epochs=args.local_epochs,
            lr=args.lr,
            use_fedprox=(args.strategy == "fedprox"),
            mu=args.mu,
        )
        clients.append(client)
        logger.info(
            f"  Client {partition.client_id}: "
            f"{len(client_dialogues)} dialogues, "
            f"{sum(len(d.utterances) for d in client_dialogues)} utterances"
        )

    # -------------------------------------------------------
    # 4. Create test DataLoader
    # -------------------------------------------------------
    test_dataset = DialogueDataset(
        test_dialogues,
        feature_cache=feature_cache.get("test", {}),
        text_dim=args.text_dim,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_dialogues,
        num_workers=0,
    )

    # -------------------------------------------------------
    # 5. Create global model & criterion
    # -------------------------------------------------------
    class_weights = torch.from_numpy(
        meld.get_emotion_weights("train").astype(np.float32)
    ).to(args.device)

    global_model = DialogueRNN(
        input_dim=args.text_dim,
        hidden_dim=args.hidden_dim,
        num_classes=len(MELD_EMOTIONS),
        num_speakers=10,
        dropout=args.dropout,
        use_attention=True,
    ).to(args.device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    total_params = sum(p.numel() for p in global_model.parameters())
    logger.info(f"\nGlobal model parameters: {total_params:,}")

    # -------------------------------------------------------
    # 6. Run FL training
    # -------------------------------------------------------
    server = FederatedServer(
        global_model=global_model,
        clients=clients,
        test_loader=test_loader,
        criterion=criterion,
        device=args.device,
        num_rounds=args.num_rounds,
        fraction_fit=1.0,  # Use all clients each round
        save_dir=args.save_dir,
        emotion_names=MELD_EMOTIONS,
    )

    start_time = time.time()
    history = server.train(patience=args.patience)
    elapsed = time.time() - start_time

    logger.info(f"\nTotal training time: {elapsed / 60:.1f} minutes")
    logger.info(f"Best test WF1: {max(history['test_wf1']):.4f}")

    # -------------------------------------------------------
    # 7. Compare with centralized baseline
    # -------------------------------------------------------
    centralized_ckpt = Path(args.save_dir) / "best_dialoguernn_meld.pt"
    if centralized_ckpt.exists():
        ckpt = torch.load(centralized_ckpt, weights_only=False)
        centralized_wf1 = ckpt.get("dev_wf1", 0)
        logger.info(f"\n{'='*60}")
        logger.info(f"  COMPARISON")
        logger.info(f"{'='*60}")
        logger.info(f"  Centralized baseline (Dev WF1): {centralized_wf1:.4f}")
        logger.info(f"  {args.strategy.upper()} (Test WF1):        {max(history['test_wf1']):.4f}")
        logger.info(f"{'='*60}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
