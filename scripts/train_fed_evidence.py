"""
Fed-Evidence Federated Training Script
=========================================
Train EvidentialDialogueRNN with EAFA aggregation.

Compares: EAFA (ours) vs FedAvg baseline

Usage:
    python scripts/train_fed_evidence.py --num_clients 5 --alpha 0.5
    python scripts/train_fed_evidence.py --num_clients 5 --alpha 0.5 --beta 0.0  # FedAvg mode
"""

import sys
import os
import copy
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.datasets.meld import MELDDataset, MELD_EMOTIONS
from data.federated_partition import FederatedPartitioner
from models.evidential.evidential_dialogue_rnn import EvidentialDialogueRNN
from models.evidential.losses import SupervisedEvidentialLoss
from federated.aggregation.eafa import EAFAAggregator
from scripts.train_centralized import DialogueDataset, collate_dialogues

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def local_train_edl(
    global_model: nn.Module,
    dataloader: DataLoader,
    loss_fn: SupervisedEvidentialLoss,
    device: str,
    local_epochs: int,
    lr: float,
    epoch: int,
) -> Tuple[OrderedDict, Dict]:
    """
    Local EDL training on one client.

    Returns:
        (state_dict, stats) where stats includes mean_uncertainty
    """
    local_model = copy.deepcopy(global_model).to(device)
    local_model.train()
    loss_fn.set_epoch(epoch)

    optimizer = optim.Adam(local_model.parameters(), lr=lr, weight_decay=1e-4)

    total_loss = 0
    total_samples = 0
    all_preds, all_labels = [], []
    all_uncertainties = []

    for ep in range(local_epochs):
        for batch in dataloader:
            features = batch["features"].to(device)
            speakers = batch["speaker_ids"].to(device)
            labels = batch["labels"].to(device)

            out = local_model(features, speakers)
            mask = labels != -1
            alpha_flat = out["alpha"][mask]
            labels_flat = labels[mask]

            loss, _ = loss_fn(alpha_flat, labels_flat)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item() * labels_flat.size(0)
            total_samples += labels_flat.size(0)

            preds = out["belief"][mask].argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels_flat.cpu().numpy())
            all_uncertainties.extend(out["uncertainty"][mask].detach().cpu().numpy())

    avg_loss = total_loss / max(total_samples, 1)
    wf1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    mean_uncertainty = float(np.mean(all_uncertainties)) if all_uncertainties else 1.0

    stats = {
        "num_samples": len(dataloader.dataset),
        "loss": avg_loss,
        "wf1": wf1,
        "mean_uncertainty": mean_uncertainty,
    }

    return local_model.state_dict(), stats


@torch.no_grad()
def evaluate_edl(model, loader, loss_fn, device, emotion_names=None):
    """Evaluate EvidentialDialogueRNN."""
    model.eval()
    model.to(device)
    all_preds, all_labels, all_uncerts = [], [], []

    for batch in loader:
        features = batch["features"].to(device)
        speakers = batch["speaker_ids"].to(device)
        labels = batch["labels"].to(device)

        out = model(features, speakers)
        mask = labels != -1
        preds = out["belief"][mask].argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels[mask].cpu().numpy())
        all_uncerts.extend(out["uncertainty"][mask].cpu().numpy())

    wf1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    mean_u = np.mean(all_uncerts)
    report = classification_report(
        all_labels, all_preds, target_names=emotion_names, digits=4, zero_division=0,
    ) if emotion_names else ""
    return wf1, mean_u, report


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fed-Evidence: EAFA federated training")
    parser.add_argument("--data_dir", type=str, default="data/raw/MELD")
    parser.add_argument("--feature_cache", type=str, default="data/features/meld_text_roberta.pt")
    # FL
    parser.add_argument("--num_clients", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--num_rounds", type=int, default=50)
    parser.add_argument("--local_epochs", type=int, default=3)
    parser.add_argument("--beta", type=float, default=1.0, help="EAFA beta (0=FedAvg)")
    # Model
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--text_dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.3)
    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--annealing_epochs", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    mode = "EAFA" if args.beta > 0 else "FedAvg (EDL)"
    logger.info(f"\n{'='*60}")
    logger.info(f"  Fed-Evidence: {mode} (beta={args.beta})")
    logger.info(f"  {args.num_clients} clients, alpha={args.alpha}")
    logger.info(f"{'='*60}")

    # -------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------
    meld = MELDDataset(data_dir=args.data_dir)
    train_dialogues = meld.get_dialogues("train")
    test_dialogues = meld.get_dialogues("test")

    feature_cache = {}
    feature_path = Path(args.feature_cache)
    if feature_path.exists():
        cached = torch.load(feature_path, weights_only=False)
        for split in ["train", "test"]:
            if split in cached:
                feats = cached[split]["features"].numpy()
                dia_ids = cached[split]["dialogue_ids"]
                utt_ids = cached[split]["utterance_ids"]
                cache = {}
                for i in range(len(feats)):
                    cache[f"{dia_ids[i].item()}_{utt_ids[i].item()}"] = feats[i]
                feature_cache[split] = cache
        args.text_dim = feats.shape[1]

    # -------------------------------------------------------
    # 2. Partition data
    # -------------------------------------------------------
    partitioner = FederatedPartitioner(
        num_clients=args.num_clients, strategy="dirichlet",
        alpha=args.alpha, seed=args.seed,
    )
    client_partitions = partitioner.partition(train_dialogues, label_ratio=1.0)
    partitioner.print_partition(client_partitions)

    dialogue_lookup = {d.dialogue_id: d for d in train_dialogues}

    client_loaders = []
    for partition in client_partitions:
        client_dialogues = [dialogue_lookup[did] for did in partition.dialogue_ids if did in dialogue_lookup]
        ds = DialogueDataset(client_dialogues, feature_cache.get("train", {}), args.text_dim)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                            collate_fn=collate_dialogues, num_workers=0)
        client_loaders.append(loader)
        logger.info(f"  Client {partition.client_id}: {len(client_dialogues)} dialogues")

    test_ds = DialogueDataset(test_dialogues, feature_cache.get("test", {}), args.text_dim)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_dialogues, num_workers=0)

    # -------------------------------------------------------
    # 3. Model + Loss + Aggregator
    # -------------------------------------------------------
    class_weights = torch.from_numpy(
        meld.get_emotion_weights("train").astype(np.float32)
    ).to(args.device)

    global_model = EvidentialDialogueRNN(
        input_dim=args.text_dim, hidden_dim=args.hidden_dim,
        num_classes=len(MELD_EMOTIONS), num_speakers=10,
        dropout=args.dropout, use_attention=True,
    ).to(args.device)

    loss_fn = SupervisedEvidentialLoss(
        num_classes=len(MELD_EMOTIONS),
        annealing_epochs=args.annealing_epochs,
        class_weights=class_weights,
    )

    aggregator = EAFAAggregator(beta=args.beta)

    params = sum(p.numel() for p in global_model.parameters())
    logger.info(f"Model: {params:,} params")

    # -------------------------------------------------------
    # 4. Federated training loop
    # -------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info(f"  Training: {args.num_rounds} rounds")
    logger.info(f"{'='*60}\n")

    best_wf1 = 0.0
    patience_counter = 0

    for round_num in range(1, args.num_rounds + 1):
        start = time.time()

        # Local training
        client_states = []
        client_data_sizes = []
        client_uncertainties = []
        client_wf1s = []

        for i, loader in enumerate(client_loaders):
            state_dict, stats = local_train_edl(
                global_model, loader, loss_fn, args.device,
                args.local_epochs, args.lr, epoch=round_num,
            )
            client_states.append(OrderedDict({k: v.cpu() for k, v in state_dict.items()}))
            client_data_sizes.append(stats["num_samples"])
            client_uncertainties.append(stats["mean_uncertainty"])
            client_wf1s.append(stats["wf1"])

        # EAFA aggregation
        global_state, agg_stats = aggregator.aggregate(
            client_states, client_data_sizes, client_uncertainties, round_num,
        )
        global_model.load_state_dict(global_state)
        global_model.to(args.device)

        # Evaluate
        test_wf1, test_u, _ = evaluate_edl(global_model, test_loader, loss_fn, args.device)
        elapsed = time.time() - start

        avg_client_wf1 = np.mean(client_wf1s)
        weights_str = ",".join([f"{w:.2f}" for w in agg_stats["weights"]])

        logger.info(
            f"Round {round_num:3d}/{args.num_rounds} | "
            f"Client WF1: {avg_client_wf1:.4f} | "
            f"Test WF1: {test_wf1:.4f} u={test_u:.3f} | "
            f"w=[{weights_str}] | {elapsed:.1f}s"
        )

        if test_wf1 > best_wf1:
            best_wf1 = test_wf1
            patience_counter = 0
            ckpt_path = Path(args.save_dir) / f"best_eafa_b{args.beta}.pt"
            ckpt_path.parent.mkdir(exist_ok=True)
            torch.save({"round": round_num, "model_state_dict": global_model.state_dict(),
                        "test_wf1": test_wf1}, ckpt_path)
            logger.info(f"  >> New best! WF1={test_wf1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"  Early stopping at round {round_num}")
                break

        del client_states
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # -------------------------------------------------------
    # 5. Final evaluation
    # -------------------------------------------------------
    ckpt_path = Path(args.save_dir) / f"best_eafa_b{args.beta}.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, weights_only=False)
        global_model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"\nLoaded best model from round {ckpt['round']}")

    test_wf1, test_u, test_report = evaluate_edl(
        global_model, test_loader, loss_fn, args.device, MELD_EMOTIONS
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"  FINAL RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"\n{test_report}")

    logger.info(f"\n{'='*60}")
    logger.info(f"  COMPARISON")
    logger.info(f"{'='*60}")
    logger.info(f"  Softmax FedAvg baseline:     WF1 = 0.5419")
    logger.info(f"  Softmax FedProx baseline:    WF1 = 0.5412")
    logger.info(f"  EDL Centralized:             WF1 = 0.5747")
    logger.info(f"  {mode} (ours):               WF1 = {test_wf1:.4f}")
    logger.info(f"  Mean uncertainty:            u = {test_u:.4f}")
    logger.info(f"{'='*60}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
