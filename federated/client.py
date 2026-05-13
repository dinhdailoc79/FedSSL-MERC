"""
Federated Client — Local Training
====================================
Each client trains a copy of the global model on their local data
for a fixed number of local epochs, then sends the updated model
back to the server.
"""

import copy
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from federated.aggregation.fedprox import FedProxLoss

logger = logging.getLogger(__name__)


class FederatedClient:
    """
    A federated learning client that performs local training.

    Each client:
    1. Receives the global model from the server
    2. Trains it on local data for `local_epochs` epochs
    3. Returns the updated model weights

    Args:
        client_id: Unique client identifier
        dataloader: DataLoader for this client's local data
        device: Training device ('cuda' or 'cpu')
        local_epochs: Number of local training epochs per FL round
        lr: Local learning rate
        use_fedprox: Whether to use FedProx proximal term
        mu: FedProx proximal coefficient
    """

    def __init__(
        self,
        client_id: int,
        dataloader: DataLoader,
        device: str = "cuda",
        local_epochs: int = 5,
        lr: float = 1e-3,
        use_fedprox: bool = False,
        mu: float = 0.01,
    ):
        self.client_id = client_id
        self.dataloader = dataloader
        self.device = device
        self.local_epochs = local_epochs
        self.lr = lr
        self.use_fedprox = use_fedprox
        self.num_samples = len(dataloader.dataset)

        if use_fedprox:
            self.prox_loss = FedProxLoss(mu=mu)

    def train(
        self,
        global_model: nn.Module,
        criterion: nn.Module,
    ) -> Tuple[nn.Module, Dict]:
        """
        Perform local training starting from the global model.

        Args:
            global_model: The current global model (will be deep-copied)
            criterion: Loss function (e.g., CrossEntropyLoss)

        Returns:
            Tuple of (trained local model, training stats dict)
        """
        # Deep copy global model for local training
        local_model = copy.deepcopy(global_model).to(self.device)
        local_model.train()

        # Store global params for FedProx
        if self.use_fedprox:
            global_params = {
                name: param.clone().detach()
                for name, param in global_model.named_parameters()
            }

        optimizer = optim.Adam(local_model.parameters(), lr=self.lr, weight_decay=1e-4)

        total_loss = 0
        total_samples = 0
        all_preds = []
        all_labels = []

        for epoch in range(self.local_epochs):
            epoch_loss = 0
            for batch in self.dataloader:
                features = batch["features"].to(self.device)
                speaker_ids = batch["speaker_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits = local_model(features, speaker_ids)

                # Mask padding
                mask = labels != -1
                logits_flat = logits[mask]
                labels_flat = labels[mask]

                # Task loss
                loss = criterion(logits_flat, labels_flat)

                # Add FedProx proximal term
                if self.use_fedprox:
                    loss += self.prox_loss(local_model, global_params)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=5.0)
                optimizer.step()

                epoch_loss += loss.item() * labels_flat.size(0)
                total_samples += labels_flat.size(0)

                preds = logits_flat.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels_flat.cpu().numpy())

            total_loss += epoch_loss

        avg_loss = total_loss / max(total_samples, 1)
        wf1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

        stats = {
            "client_id": self.client_id,
            "num_samples": self.num_samples,
            "loss": avg_loss,
            "wf1": wf1,
            "local_epochs": self.local_epochs,
        }

        return local_model, stats
