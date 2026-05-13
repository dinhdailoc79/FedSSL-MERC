"""
Non-IID Federated Data Partitioning
=====================================
Partition conversational datasets across federated clients
with various non-IID strategies.

In real FL for emotion recognition, different clients (call centers,
hospitals, platforms) naturally have different emotion distributions.
This module simulates such heterogeneity.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ClientData:
    """Data assigned to a single federated client."""
    client_id: int
    dialogue_ids: List[int] = field(default_factory=list)
    labeled_ids: List[int] = field(default_factory=list)    # Semi-supervised: labeled subset
    unlabeled_ids: List[int] = field(default_factory=list)  # Semi-supervised: unlabeled subset
    label_ratio: float = 1.0  # Fraction of data that is labeled

    @property
    def num_dialogues(self) -> int:
        return len(self.dialogue_ids)

    @property
    def num_labeled(self) -> int:
        return len(self.labeled_ids)

    @property
    def num_unlabeled(self) -> int:
        return len(self.unlabeled_ids)


class FederatedPartitioner:
    """
    Partition datasets across federated clients with non-IID strategies.

    Supports:
    1. IID: Uniform random partition
    2. Label-based Non-IID: Dirichlet distribution over emotion labels
    3. Speaker-based Non-IID: Partition by speaker groups
    4. Quantity-based Non-IID: Unequal data sizes

    Each strategy also supports semi-supervised splitting (labeled vs unlabeled).

    Usage:
        >>> partitioner = FederatedPartitioner(num_clients=5, strategy="dirichlet")
        >>> clients = partitioner.partition(dialogues, label_ratio=0.1)
        >>> for c in clients:
        ...     print(f"Client {c.client_id}: {c.num_labeled} labeled, {c.num_unlabeled} unlabeled")
    """

    def __init__(
        self,
        num_clients: int = 5,
        strategy: str = "dirichlet",
        alpha: float = 0.5,
        seed: int = 42,
    ):
        """
        Args:
            num_clients: Number of federated clients
            strategy: Partitioning strategy ("iid", "dirichlet", "speaker", "quantity")
            alpha: Dirichlet concentration parameter (lower = more non-IID)
                   - alpha=100: nearly IID
                   - alpha=1.0: moderate non-IID
                   - alpha=0.1: highly non-IID
            seed: Random seed for reproducibility
        """
        self.num_clients = num_clients
        self.strategy = strategy
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)

        logger.info(
            f"FederatedPartitioner: {num_clients} clients, "
            f"strategy={strategy}, alpha={alpha}"
        )

    def partition(
        self,
        dialogues: List,
        label_ratio: float = 0.1,
    ) -> List[ClientData]:
        """
        Partition dialogues across clients.

        Args:
            dialogues: List of Dialogue objects
            label_ratio: Fraction of each client's data that is labeled (for SSL).
                         0.1 means 10% labeled, 90% unlabeled.

        Returns:
            List of ClientData, one per client
        """
        if self.strategy == "iid":
            clients = self._partition_iid(dialogues)
        elif self.strategy == "dirichlet":
            clients = self._partition_dirichlet(dialogues)
        elif self.strategy == "speaker":
            clients = self._partition_by_speaker(dialogues)
        elif self.strategy == "quantity":
            clients = self._partition_quantity(dialogues)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Apply semi-supervised split
        for client in clients:
            client.label_ratio = label_ratio
            n_total = len(client.dialogue_ids)
            n_labeled = max(1, int(n_total * label_ratio))

            indices = self.rng.permutation(n_total)
            client.labeled_ids = [client.dialogue_ids[i] for i in indices[:n_labeled]]
            client.unlabeled_ids = [client.dialogue_ids[i] for i in indices[n_labeled:]]

        self._log_partition_stats(clients)
        return clients

    # ----------------------------------------------------------
    # Partitioning Strategies
    # ----------------------------------------------------------

    def _partition_iid(self, dialogues: List) -> List[ClientData]:
        """Uniform random partition — baseline (IID)."""
        dialogue_ids = [d.dialogue_id for d in dialogues]
        self.rng.shuffle(dialogue_ids)

        chunks = np.array_split(dialogue_ids, self.num_clients)
        return [
            ClientData(client_id=i, dialogue_ids=list(chunk))
            for i, chunk in enumerate(chunks)
        ]

    def _partition_dirichlet(self, dialogues: List) -> List[ClientData]:
        """
        Dirichlet-based label partition — primary non-IID strategy.

        Uses the dominant emotion of each dialogue to assign to clients
        based on a Dirichlet distribution, creating natural class imbalance.
        """
        # Get dominant emotion for each dialogue
        dialogue_labels = {}
        for d in dialogues:
            labels = d.emotion_labels
            if labels:
                # Most frequent emotion in the dialogue
                dominant = max(set(labels), key=labels.count)
                dialogue_labels[d.dialogue_id] = dominant

        # Group dialogues by dominant emotion
        from collections import defaultdict
        label_to_dialogues = defaultdict(list)
        for dia_id, label in dialogue_labels.items():
            label_to_dialogues[label].append(dia_id)

        # Allocate using Dirichlet distribution
        clients = [ClientData(client_id=i) for i in range(self.num_clients)]

        for label, dia_ids in label_to_dialogues.items():
            # Sample proportions from Dirichlet
            proportions = self.rng.dirichlet([self.alpha] * self.num_clients)
            proportions = (proportions * len(dia_ids)).astype(int)

            # Fix rounding
            diff = len(dia_ids) - proportions.sum()
            proportions[self.rng.integers(self.num_clients)] += diff

            # Shuffle and assign
            self.rng.shuffle(dia_ids)
            start = 0
            for client_idx, count in enumerate(proportions):
                clients[client_idx].dialogue_ids.extend(
                    dia_ids[start:start + count]
                )
                start += count

        return clients

    def _partition_by_speaker(self, dialogues: List) -> List[ClientData]:
        """
        Speaker-based partition — assign dialogues by speaker groups.
        Simulates different platforms/institutions having different users.
        """
        # Collect all unique speakers
        from collections import defaultdict

        speaker_dialogues = defaultdict(set)
        for d in dialogues:
            for speaker in d.speakers:
                speaker_dialogues[speaker].add(d.dialogue_id)

        # Assign speakers to clients round-robin
        speakers = list(speaker_dialogues.keys())
        self.rng.shuffle(speakers)

        clients = [ClientData(client_id=i) for i in range(self.num_clients)]
        for i, speaker in enumerate(speakers):
            client_idx = i % self.num_clients
            clients[client_idx].dialogue_ids.extend(speaker_dialogues[speaker])

        # Remove duplicates (a dialogue can have multiple speakers)
        for client in clients:
            client.dialogue_ids = list(set(client.dialogue_ids))

        return clients

    def _partition_quantity(self, dialogues: List) -> List[ClientData]:
        """
        Quantity-based non-IID — clients have different amounts of data.
        Uses a power-law distribution for realistic imbalance.
        """
        dialogue_ids = [d.dialogue_id for d in dialogues]
        self.rng.shuffle(dialogue_ids)

        # Power-law distribution for data sizes
        proportions = self.rng.dirichlet([0.3] * self.num_clients)
        sizes = (proportions * len(dialogue_ids)).astype(int)
        sizes[-1] = len(dialogue_ids) - sizes[:-1].sum()  # Fix rounding

        clients = []
        start = 0
        for i, size in enumerate(sizes):
            clients.append(ClientData(
                client_id=i,
                dialogue_ids=list(dialogue_ids[start:start + size]),
            ))
            start += size

        return clients

    # ----------------------------------------------------------
    # Logging
    # ----------------------------------------------------------

    def _log_partition_stats(self, clients: List[ClientData]):
        """Log partition statistics."""
        logger.info(f"\nFederated Partition ({self.strategy}):")
        for c in clients:
            logger.info(
                f"  Client {c.client_id}: "
                f"{c.num_dialogues} dialogues "
                f"({c.num_labeled} labeled, {c.num_unlabeled} unlabeled)"
            )

    def print_partition(self, clients: List[ClientData]):
        """Print formatted partition summary."""
        print(f"\n{'='*60}")
        print(f"  Federated Partition - {self.strategy.upper()}")
        print(f"  alpha={self.alpha}, num_clients={self.num_clients}")
        print(f"{'='*60}")

        for c in clients:
            labeled_pct = c.num_labeled / max(c.num_dialogues, 1) * 100
            bar = "#" * (c.num_dialogues // 5)
            print(
                f"  Client {c.client_id}: {c.num_dialogues:>4} dialogues "
                f"[{c.num_labeled:>3}L / {c.num_unlabeled:>3}U] "
                f"({labeled_pct:.0f}% labeled) {bar}"
            )

        total = sum(c.num_dialogues for c in clients)
        print(f"  {'-'*50}")
        print(f"  Total: {total} dialogues across {len(clients)} clients")
        print(f"{'='*60}\n")
