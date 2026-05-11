"""
Quick test: MELD Data Loading + Federated Partition
====================================================
Verifies the data pipeline works end-to-end.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.datasets.meld import MELDDataset
from data.federated_partition import FederatedPartitioner


def main():
    # 1. Load MELD
    print("=" * 60)
    print("  STEP 1: Load MELD Dataset")
    print("=" * 60)
    dataset = MELDDataset(data_dir="data/raw/MELD")
    train_dialogues = dataset.get_dialogues("train")
    print(f"  Loaded {len(train_dialogues)} training dialogues\n")

    # 2. Show class weights (for imbalanced learning)
    weights = dataset.get_emotion_weights("train")
    print("  Class weights (inverse frequency):")
    from data.datasets.meld import MELD_EMOTIONS
    for emo, w in zip(MELD_EMOTIONS, weights):
        print(f"    {emo:<12} {w:.3f}")

    # 3. Federated Partition (Dirichlet Non-IID)
    print(f"\n{'=' * 60}")
    print("  STEP 2: Federated Partition (Dirichlet, alpha=0.5)")
    print("=" * 60)
    partitioner = FederatedPartitioner(
        num_clients=5,
        strategy="dirichlet",
        alpha=0.5,
        seed=42,
    )
    clients = partitioner.partition(
        train_dialogues,
        label_ratio=0.1,   # 10% labeled, 90% unlabeled (SSL)
    )
    partitioner.print_partition(clients)

    # 4. IID baseline comparison
    print(f"{'=' * 60}")
    print("  STEP 3: IID Partition (baseline)")
    print("=" * 60)
    iid_partitioner = FederatedPartitioner(
        num_clients=5,
        strategy="iid",
        seed=42,
    )
    iid_clients = iid_partitioner.partition(train_dialogues, label_ratio=0.1)
    iid_partitioner.print_partition(iid_clients)

    print("All pipeline tests PASSED!")


if __name__ == "__main__":
    main()
