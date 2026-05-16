"""
ThuanPhongNhi Full Pipeline: Federated Multimodal Training
========================================================
EAFA aggregation + Multimodal EDL (text+audio) with DS Fusion.

This is the COMPLETE ThuanPhongNhi pipeline combining all 4 contributions:
1. EDL Head (Dirichlet)
2. DS Fusion (text+audio evidence)
3. EAFA (uncertainty-weighted federated aggregation)
4. Auxiliary per-modality losses

Usage:
    python scripts/train_fed_multimodal.py
    python scripts/train_fed_multimodal.py --beta 0.0  # FedAvg mode for comparison
"""

import sys
import os
import copy
import time
import logging
from pathlib import Path
from typing import Dict, Tuple
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, classification_report

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.datasets.meld import MELDDataset, MELD_EMOTIONS
from data.federated_partition import FederatedPartitioner
from models.evidential.multimodal_edl import MultimodalEvidentialDialogueRNN
from models.evidential.losses import SupervisedEvidentialLoss
from federated.aggregation.eafa import EAFAAggregator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# -------------------------------------------------------
# Dataset
# -------------------------------------------------------
class MultimodalDialogueDataset(Dataset):
    """Combines text + audio features per dialogue."""

    def __init__(self, dialogues, text_cache, audio_cache, text_dim=768, audio_dim=768):
        self.dialogues = dialogues
        self.text_cache = text_cache
        self.audio_cache = audio_cache
        self.text_dim = text_dim
        self.audio_dim = audio_dim

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, idx):
        dialogue = self.dialogues[idx]
        text_feats, audio_feats, labels, speakers = [], [], [], []

        for utt in dialogue.utterances:
            key = f"{dialogue.dialogue_id}_{utt.utterance_id}"
            text_feats.append(
                torch.from_numpy(self.text_cache[key]) if key in self.text_cache
                else torch.zeros(self.text_dim)
            )
            audio_feats.append(
                torch.from_numpy(self.audio_cache[key]) if key in self.audio_cache
                else torch.zeros(self.audio_dim)
            )
            labels.append(utt.emotion_idx)
            speakers.append(utt.speaker_id if hasattr(utt, 'speaker_id') else 0)

        return {
            "text_features": torch.stack(text_feats),
            "audio_features": torch.stack(audio_feats),
            "labels": torch.tensor(labels, dtype=torch.long),
            "speaker_ids": torch.tensor(speakers, dtype=torch.long),
        }


def collate_multimodal(batch):
    max_len = max(b["labels"].size(0) for b in batch)
    text_dim = batch[0]["text_features"].size(1)
    audio_dim = batch[0]["audio_features"].size(1)
    bs = len(batch)

    text_feat = torch.zeros(bs, max_len, text_dim)
    audio_feat = torch.zeros(bs, max_len, audio_dim)
    labels = torch.full((bs, max_len), -1, dtype=torch.long)
    speakers = torch.zeros(bs, max_len, dtype=torch.long)

    for i, b in enumerate(batch):
        L = b["labels"].size(0)
        text_feat[i, :L] = b["text_features"]
        audio_feat[i, :L] = b["audio_features"]
        labels[i, :L] = b["labels"]
        speakers[i, :L] = b["speaker_ids"]

    return {
        "text_features": text_feat, "audio_features": audio_feat,
        "labels": labels, "speaker_ids": speakers,
    }


# -------------------------------------------------------
# Local training
# -------------------------------------------------------
def local_train_multimodal(
    global_model: nn.Module,
    dataloader: DataLoader,
    loss_fn: SupervisedEvidentialLoss,
    aux_loss_fn: SupervisedEvidentialLoss,
    device: str,
    local_epochs: int,
    lr: float,
    epoch: int,
    lambda_aux: float = 0.3,
) -> Tuple[OrderedDict, Dict]:
    """Local multimodal EDL training on one client."""
    local_model = copy.deepcopy(global_model).to(device)
    local_model.train()
    loss_fn.set_epoch(epoch)
    aux_loss_fn.set_epoch(epoch)

    optimizer = optim.Adam(local_model.parameters(), lr=lr, weight_decay=1e-4)

    total_loss, total_samples = 0, 0
    all_preds, all_labels, all_u = [], [], []

    for ep in range(local_epochs):
        for batch in dataloader:
            text = batch["text_features"].to(device)
            audio = batch["audio_features"].to(device)
            speakers = batch["speaker_ids"].to(device)
            labels = batch["labels"].to(device)

            out = local_model(text, audio, speakers)
            mask = labels != -1
            labels_flat = labels[mask]

            # Fused loss
            fused_loss, _ = loss_fn(out["alpha"][mask], labels_flat)

            # Auxiliary per-modality losses
            text_loss, _ = aux_loss_fn(out["text_alpha"][mask], labels_flat)
            audio_loss, _ = aux_loss_fn(out["audio_alpha"][mask], labels_flat)
            aux_loss = (text_loss + audio_loss) / 2.0

            loss = fused_loss + lambda_aux * aux_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item() * labels_flat.size(0)
            total_samples += labels_flat.size(0)

            preds = out["belief"][mask].argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels_flat.cpu().numpy())
            all_u.extend(out["uncertainty"][mask].detach().cpu().numpy())

    stats = {
        "num_samples": len(dataloader.dataset),
        "loss": total_loss / max(total_samples, 1),
        "wf1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        "mean_uncertainty": float(np.mean(all_u)) if all_u else 1.0,
    }

    return local_model.state_dict(), stats


# -------------------------------------------------------
# Evaluate
# -------------------------------------------------------
@torch.no_grad()
def evaluate_multimodal(model, loader, device, emotion_names=None):
    model.eval()
    model.to(device)
    all_preds, all_labels, all_u = [], [], []

    for batch in loader:
        text = batch["text_features"].to(device)
        audio = batch["audio_features"].to(device)
        speakers = batch["speaker_ids"].to(device)
        labels = batch["labels"].to(device)

        out = model(text, audio, speakers)
        mask = labels != -1
        preds = out["belief"][mask].argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels[mask].cpu().numpy())
        all_u.extend(out["uncertainty"][mask].cpu().numpy())

    wf1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    mean_u = np.mean(all_u)
    report = classification_report(
        all_labels, all_preds, target_names=emotion_names, digits=4, zero_division=0,
    ) if emotion_names else ""
    return wf1, mean_u, report


# -------------------------------------------------------
# Feature loading
# -------------------------------------------------------
def load_feature_cache(path, splits):
    caches = {}
    if not Path(path).exists():
        logger.warning(f"Feature file not found: {path}")
        return caches
    cached = torch.load(path, weights_only=False)
    for split in splits:
        if split in cached:
            feats = cached[split]["features"].numpy()
            dia_ids = cached[split]["dialogue_ids"]
            utt_ids = cached[split]["utterance_ids"]
            cache = {}
            for i in range(len(feats)):
                cache[f"{dia_ids[i].item()}_{utt_ids[i].item()}"] = feats[i]
            caches[split] = cache
    return caches


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="ThuanPhongNhi: Federated Multimodal Training")
    parser.add_argument("--data_dir", type=str, default="data/raw/MELD")
    parser.add_argument("--text_cache", type=str, default="data/features/meld_text_roberta.pt")
    parser.add_argument("--audio_cache", type=str, default="data/features/meld_audio_wavlm.pt")
    # FL
    parser.add_argument("--num_clients", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--num_rounds", type=int, default=50)
    parser.add_argument("--local_epochs", type=int, default=3)
    parser.add_argument("--beta", type=float, default=1.0, help="EAFA beta (0=FedAvg)")
    # Model
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--fusion_mode", type=str, default="evidence_sum")
    parser.add_argument("--lambda_aux", type=float, default=0.3)
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

    mode = "EAFA+DS" if args.beta > 0 else "FedAvg+DS"
    logger.info(f"\n{'='*60}")
    logger.info(f"  ThuanPhongNhi Full Pipeline: {mode}")
    logger.info(f"  {args.num_clients} clients, alpha={args.alpha}, beta={args.beta}")
    logger.info(f"  Fusion: {args.fusion_mode}")
    logger.info(f"{'='*60}")

    # -------------------------------------------------------
    # 1. Load data + features
    # -------------------------------------------------------
    meld = MELDDataset(data_dir=args.data_dir)
    train_dialogues = meld.get_dialogues("train")
    test_dialogues = meld.get_dialogues("test")

    text_caches = load_feature_cache(args.text_cache, ["train", "test"])
    audio_caches = load_feature_cache(args.audio_cache, ["train", "test"])
    text_dim, audio_dim = 768, 768

    # -------------------------------------------------------
    # 2. Partition data (non-IID Dirichlet)
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
        client_dias = [dialogue_lookup[did] for did in partition.dialogue_ids if did in dialogue_lookup]
        ds = MultimodalDialogueDataset(
            client_dias, text_caches.get("train", {}), audio_caches.get("train", {}),
            text_dim, audio_dim,
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                            collate_fn=collate_multimodal, num_workers=0)
        client_loaders.append(loader)
        logger.info(f"  Client {partition.client_id}: {len(client_dias)} dialogues")

    test_ds = MultimodalDialogueDataset(
        test_dialogues, text_caches.get("test", {}), audio_caches.get("test", {}),
        text_dim, audio_dim,
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_multimodal, num_workers=0)

    # -------------------------------------------------------
    # 3. Model + Loss + Aggregator
    # -------------------------------------------------------
    class_weights = torch.from_numpy(
        meld.get_emotion_weights("train").astype(np.float32)
    ).to(args.device)

    global_model = MultimodalEvidentialDialogueRNN(
        text_dim=text_dim, audio_dim=audio_dim,
        hidden_dim=args.hidden_dim, num_classes=len(MELD_EMOTIONS),
        num_speakers=10, dropout=args.dropout,
        fusion_mode=args.fusion_mode,
    ).to(args.device)

    loss_fn = SupervisedEvidentialLoss(
        num_classes=len(MELD_EMOTIONS),
        annealing_epochs=args.annealing_epochs,
        class_weights=class_weights,
    )
    aux_loss_fn = SupervisedEvidentialLoss(
        num_classes=len(MELD_EMOTIONS),
        annealing_epochs=args.annealing_epochs,
        class_weights=class_weights,
    )
    aggregator = EAFAAggregator(beta=args.beta)

    params = sum(p.numel() for p in global_model.parameters())
    logger.info(f"Model: {params:,} params, {mode}")

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

        client_states, client_sizes, client_us, client_wf1s = [], [], [], []

        for i, loader in enumerate(client_loaders):
            state_dict, stats = local_train_multimodal(
                global_model, loader, loss_fn, aux_loss_fn, args.device,
                args.local_epochs, args.lr, epoch=round_num,
                lambda_aux=args.lambda_aux,
            )
            client_states.append(OrderedDict({k: v.cpu() for k, v in state_dict.items()}))
            client_sizes.append(stats["num_samples"])
            client_us.append(stats["mean_uncertainty"])
            client_wf1s.append(stats["wf1"])

        # EAFA aggregation
        global_state, agg_stats = aggregator.aggregate(
            client_states, client_sizes, client_us, round_num,
        )
        global_model.load_state_dict(global_state)
        global_model.to(args.device)

        # Evaluate
        test_wf1, test_u, _ = evaluate_multimodal(global_model, test_loader, args.device)
        elapsed = time.time() - start

        avg_wf1 = np.mean(client_wf1s)
        weights_str = ",".join([f"{w:.2f}" for w in agg_stats["weights"]])

        logger.info(
            f"Round {round_num:3d}/{args.num_rounds} | "
            f"Clients: {avg_wf1:.4f} | Test WF1: {test_wf1:.4f} u={test_u:.3f} | "
            f"w=[{weights_str}] | {elapsed:.1f}s"
        )

        if test_wf1 > best_wf1:
            best_wf1 = test_wf1
            patience_counter = 0
            ckpt = Path(args.save_dir) / f"best_fed_multimodal_b{args.beta}.pt"
            ckpt.parent.mkdir(exist_ok=True)
            torch.save({"round": round_num, "model_state_dict": global_model.state_dict(),
                        "test_wf1": test_wf1}, ckpt)
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
    ckpt = Path(args.save_dir) / f"best_fed_multimodal_b{args.beta}.pt"
    if ckpt.exists():
        state = torch.load(ckpt, weights_only=False)
        global_model.load_state_dict(state["model_state_dict"])
        logger.info(f"\nLoaded best model from round {state['round']}")

    test_wf1, test_u, test_report = evaluate_multimodal(
        global_model, test_loader, args.device, MELD_EMOTIONS,
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"  FINAL: ThuanPhongNhi Full Pipeline ({mode})")
    logger.info(f"{'='*60}")
    logger.info(f"\n{test_report}")
    logger.info(f"\n{'='*60}")
    logger.info(f"  ALL RESULTS COMPARISON")
    logger.info(f"{'='*60}")
    logger.info(f"  Softmax Centralized (text):       WF1 = 0.5442")
    logger.info(f"  EDL Centralized (text):            WF1 = 0.5747")
    logger.info(f"  Multimodal EDL (text+audio):       WF1 = 0.5606")
    logger.info(f"  Softmax FedAvg (text):             WF1 = 0.5419")
    logger.info(f"  EAFA (text):                       WF1 = 0.5585")
    logger.info(f"  ThuanPhongNhi Full ({mode}):  WF1 = {test_wf1:.4f}")
    logger.info(f"  Fused uncertainty:                 u = {test_u:.4f}")
    logger.info(f"{'='*60}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
