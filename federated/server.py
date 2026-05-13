"""
Federated Server — Orchestration
===================================
Coordinates the FL training process:
1. Broadcasts global model to clients
2. Collects trained models from clients
3. Aggregates using FedAvg/FedProx
4. Evaluates global model on test set
"""

import copy
import time
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report

from federated.client import FederatedClient
from federated.aggregation.fedavg import fedavg_aggregate

logger = logging.getLogger(__name__)


class FederatedServer:
    """
    FL Server that orchestrates training across clients.

    Args:
        global_model: The shared global model
        clients: List of FederatedClient objects
        test_loader: DataLoader for global test set
        criterion: Loss function
        device: Device for evaluation
        num_rounds: Number of FL communication rounds
        fraction_fit: Fraction of clients to sample each round (1.0 = all)
        save_dir: Directory to save checkpoints
    """

    def __init__(
        self,
        global_model: nn.Module,
        clients: List[FederatedClient],
        test_loader: DataLoader,
        criterion: nn.Module,
        device: str = "cuda",
        num_rounds: int = 50,
        fraction_fit: float = 1.0,
        save_dir: str = "checkpoints",
        emotion_names: Optional[List[str]] = None,
    ):
        self.global_model = global_model
        self.clients = clients
        self.test_loader = test_loader
        self.criterion = criterion
        self.device = device
        self.num_rounds = num_rounds
        self.fraction_fit = fraction_fit
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.emotion_names = emotion_names

        # Track metrics
        self.history = {
            "round": [],
            "test_loss": [],
            "test_wf1": [],
            "client_losses": [],
            "client_wf1s": [],
        }

    def select_clients(self) -> List[FederatedClient]:
        """Select a subset of clients for this round."""
        num_selected = max(1, int(len(self.clients) * self.fraction_fit))
        if num_selected >= len(self.clients):
            return self.clients
        indices = np.random.choice(len(self.clients), num_selected, replace=False)
        return [self.clients[i] for i in indices]

    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float, str]:
        """Evaluate global model on test set."""
        self.global_model.eval()
        self.global_model.to(self.device)
        total_loss = 0
        all_preds = []
        all_labels = []

        for batch in self.test_loader:
            features = batch["features"].to(self.device)
            speaker_ids = batch["speaker_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            logits = self.global_model(features, speaker_ids)
            mask = labels != -1
            logits_flat = logits[mask]
            labels_flat = labels[mask]

            loss = self.criterion(logits_flat, labels_flat)
            total_loss += loss.item() * labels_flat.size(0)

            preds = logits_flat.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels_flat.cpu().numpy())

        avg_loss = total_loss / max(len(all_labels), 1)
        wf1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        report = classification_report(
            all_labels, all_preds,
            target_names=self.emotion_names,
            digits=4,
            zero_division=0,
        ) if self.emotion_names else ""
        return avg_loss, wf1, report

    def train(self, patience: int = 10) -> Dict:
        """
        Run the full federated training loop.

        Args:
            patience: Early stopping patience (rounds without improvement)

        Returns:
            Training history dict
        """
        best_wf1 = 0.0
        patience_counter = 0

        logger.info(f"\n{'='*60}")
        logger.info(f"  Federated Training: {self.num_rounds} rounds, {len(self.clients)} clients")
        logger.info(f"  Strategy: {'FedProx' if self.clients[0].use_fedprox else 'FedAvg'}")
        logger.info(f"{'='*60}\n")

        for round_num in range(1, self.num_rounds + 1):
            start_time = time.time()

            # 1. Select clients
            selected_clients = self.select_clients()

            # 2. Local training on each client
            client_models = []
            client_weights = []
            client_stats = []

            for client in selected_clients:
                local_model, stats = client.train(self.global_model, self.criterion)
                client_models.append(local_model)
                client_weights.append(stats["num_samples"])
                client_stats.append(stats)

            # 3. Aggregate
            self.global_model = fedavg_aggregate(
                self.global_model, client_models, client_weights
            )

            # 4. Evaluate
            test_loss, test_wf1, _ = self.evaluate()

            elapsed = time.time() - start_time

            # Log
            avg_client_loss = np.mean([s["loss"] for s in client_stats])
            avg_client_wf1 = np.mean([s["wf1"] for s in client_stats])

            self.history["round"].append(round_num)
            self.history["test_loss"].append(test_loss)
            self.history["test_wf1"].append(test_wf1)
            self.history["client_losses"].append(avg_client_loss)
            self.history["client_wf1s"].append(avg_client_wf1)

            logger.info(
                f"Round {round_num:3d}/{self.num_rounds} | "
                f"Client Avg Loss: {avg_client_loss:.4f} WF1: {avg_client_wf1:.4f} | "
                f"Test Loss: {test_loss:.4f} WF1: {test_wf1:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            # Save best model
            if test_wf1 > best_wf1:
                best_wf1 = test_wf1
                patience_counter = 0
                strategy = "fedprox" if self.clients[0].use_fedprox else "fedavg"
                ckpt_path = self.save_dir / f"best_{strategy}_meld.pt"
                torch.save({
                    "round": round_num,
                    "model_state_dict": self.global_model.state_dict(),
                    "test_wf1": test_wf1,
                }, ckpt_path)
                logger.info(f"  >> New best! WF1={test_wf1:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"  Early stopping at round {round_num}")
                    break

            # Free client models from GPU
            del client_models
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Final evaluation
        logger.info(f"\n{'='*60}")
        logger.info(f"  Final Test Evaluation")
        logger.info(f"{'='*60}")
        test_loss, test_wf1, test_report = self.evaluate()
        logger.info(f"Test WF1: {test_wf1:.4f}")
        if test_report:
            logger.info(f"\n{test_report}")
        logger.info(f"Best WF1: {best_wf1:.4f}")

        return self.history
