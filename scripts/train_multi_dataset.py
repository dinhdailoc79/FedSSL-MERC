"""
LucBinh Multi-Dataset EDL Training
=======================================
Run EDL centralized + EAFA federated on MELD, IEMOCAP, DailyDialog.

Usage:
    python scripts/train_multi_dataset.py --dataset meld
    python scripts/train_multi_dataset.py --dataset iemocap
    python scripts/train_multi_dataset.py --dataset dailydialog
    python scripts/train_multi_dataset.py --dataset all
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
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, classification_report

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.evidential.evidential_dialogue_rnn import EvidentialDialogueRNN
from models.evidential.losses import SupervisedEvidentialLoss
from federated.aggregation.eafa import EAFAAggregator
from data.federated_partition import FederatedPartitioner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# -------------------------------------------------------
# Generic Dialogue Dataset (works for all 3 datasets)
# -------------------------------------------------------
class GenericDialogueDataset(Dataset):
    def __init__(self, dialogues, feature_cache, feat_dim=768):
        self.dialogues = dialogues
        self.cache = feature_cache
        self.feat_dim = feat_dim

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, idx):
        d = self.dialogues[idx]
        feats, labels, speakers = [], [], []
        for u in d.utterances:
            key = self._make_key(d, u)
            feats.append(
                torch.from_numpy(self.cache[key]) if key in self.cache
                else torch.zeros(self.feat_dim)
            )
            labels.append(u.emotion_idx)
            speakers.append(getattr(u, 'speaker_id', 0))
        return {
            "features": torch.stack(feats),
            "labels": torch.tensor(labels, dtype=torch.long),
            "speaker_ids": torch.tensor(speakers, dtype=torch.long),
        }

    def _make_key(self, dialogue, utterance):
        return f"{dialogue.dialogue_id}_{utterance.utterance_id}"


def collate_dialogues(batch):
    max_len = max(b["labels"].size(0) for b in batch)
    feat_dim = batch[0]["features"].size(1)
    bs = len(batch)
    feats = torch.zeros(bs, max_len, feat_dim)
    labels = torch.full((bs, max_len), -1, dtype=torch.long)
    speakers = torch.zeros(bs, max_len, dtype=torch.long)
    for i, b in enumerate(batch):
        L = b["labels"].size(0)
        feats[i, :L] = b["features"]
        labels[i, :L] = b["labels"]
        speakers[i, :L] = b["speaker_ids"]
    return {"features": feats, "labels": labels, "speaker_ids": speakers}


# -------------------------------------------------------
# Dataset loaders
# -------------------------------------------------------
def load_meld():
    from data.datasets.meld import MELDDataset, MELD_EMOTIONS
    meld = MELDDataset(data_dir="data/raw/MELD")
    train = meld.get_dialogues("train")
    dev = meld.get_dialogues("dev")
    test = meld.get_dialogues("test")
    weights = meld.get_emotion_weights("train")
    cache = _load_cache("data/features/meld_text_roberta.pt", ["train", "dev", "test"])
    return train, dev, test, MELD_EMOTIONS, weights, cache, 10


def load_iemocap():
    from data.datasets.iemocap import IEMOCAPDataset, IEMOCAP_EMOTIONS_6
    ds = IEMOCAPDataset(data_dir="data/raw/IEMOCAP/IEMOCAP_full_release", num_classes=6)
    ds.load()
    # Standard: test=session5
    train_dias, test_dias = ds.get_session_split(test_session=5)
    # Use session 4 as dev
    dev_dias = [d for d in train_dias if d.session == 4]
    train_dias = [d for d in train_dias if d.session != 4]
    weights = ds.get_emotion_weights()

    # Load features (stored by session)
    cache = _load_iemocap_cache("data/features/iemocap_text_roberta.pt")
    return train_dias, dev_dias, test_dias, IEMOCAP_EMOTIONS_6, weights, cache, 10


def load_dailydialog():
    from data.datasets.dailydialog import DailyDialogDataset, DAILYDIALOG_EMOTIONS
    ds = DailyDialogDataset(data_dir="data/raw/DailyDialog")
    train = ds.get_dialogues("train")
    dev = ds.get_dialogues("dev")
    test = ds.get_dialogues("test")
    weights = ds.get_emotion_weights("train")
    cache = _load_cache("data/features/dailydialog_text_roberta.pt", ["train", "dev", "test"])
    return train, dev, test, DAILYDIALOG_EMOTIONS, weights, cache, 2


def _load_cache(path, splits):
    caches = {}
    if not Path(path).exists():
        return caches
    cached = torch.load(path, weights_only=False)
    for split in splits:
        if split in cached:
            feats = cached[split]["features"].numpy()
            dia_ids = cached[split]["dialogue_ids"]
            utt_ids = cached[split]["utterance_ids"]
            c = {}
            for i in range(len(feats)):
                c[f"{dia_ids[i].item()}_{utt_ids[i].item()}"] = feats[i]
            caches[split] = c
    return caches


def _load_iemocap_cache(path):
    """IEMOCAP features stored by session, need special handling."""
    caches = {"train": {}, "dev": {}, "test": {}}
    if not Path(path).exists():
        return caches
    cached = torch.load(path, weights_only=False)
    for key, data in cached.items():
        session_num = int(key.replace("session", ""))
        feats = data["features"].numpy()
        dia_strs = data["dia_id_strs"]
        utt_strs = data["utt_id_strs"]

        # Map sessions: train=1,2,3, dev=4, test=5
        if session_num <= 3:
            target = "train"
        elif session_num == 4:
            target = "dev"
        else:
            target = "test"

        for i in range(len(feats)):
            caches[target][f"{dia_strs[i]}_{utt_strs[i]}"] = feats[i]

    for split, c in caches.items():
        logger.info(f"  IEMOCAP {split}: {len(c)} features loaded")
    return caches


# -------------------------------------------------------
# Training & Evaluation
# -------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, emotion_names=None):
    model.eval()
    all_preds, all_labels, all_u = [], [], []
    for batch in loader:
        feats = batch["features"].to(device)
        speakers = batch["speaker_ids"].to(device)
        labels = batch["labels"].to(device)
        out = model(feats, speakers)
        mask = labels != -1
        preds = out["belief"][mask].argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels[mask].cpu().numpy())
        all_u.extend(out["uncertainty"][mask].cpu().numpy())

    wf1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    mean_u = np.mean(all_u) if all_u else 1.0
    report = classification_report(
        all_labels, all_preds, target_names=emotion_names, digits=4, zero_division=0,
    ) if emotion_names else ""
    return wf1, mean_u, report


def train_centralized(dataset_name, train_dias, dev_dias, test_dias,
                      emotions, weights, cache, num_speakers, args):
    """Train EDL centralized on one dataset."""
    device = args.device
    num_classes = len(emotions)

    class_weights = torch.from_numpy(weights.astype(np.float32)).to(device)

    train_ds = GenericDialogueDataset(train_dias, cache.get("train", {}))
    dev_ds = GenericDialogueDataset(dev_dias, cache.get("dev", {}))
    test_ds = GenericDialogueDataset(test_dias, cache.get("test", {}))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_dialogues, num_workers=0)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_dialogues, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_dialogues, num_workers=0)

    model = EvidentialDialogueRNN(
        input_dim=768, hidden_dim=args.hidden_dim,
        num_classes=num_classes, num_speakers=num_speakers,
        dropout=args.dropout,
    ).to(device)

    loss_fn = SupervisedEvidentialLoss(
        num_classes=num_classes, annealing_epochs=args.annealing_epochs,
        class_weights=class_weights,
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    logger.info(f"\n{'='*60}")
    logger.info(f"  EDL Centralized — {dataset_name.upper()} ({num_classes} classes)")
    logger.info(f"  Train: {len(train_dias)} dias, Dev: {len(dev_dias)}, Test: {len(test_dias)}")
    logger.info(f"{'='*60}\n")

    best_wf1, patience_cnt = 0.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_fn.set_epoch(epoch)
        total_loss, total_samples = 0, 0
        start = time.time()

        for batch in train_loader:
            feats = batch["features"].to(device)
            speakers = batch["speaker_ids"].to(device)
            labels = batch["labels"].to(device)
            out = model(feats, speakers)
            mask = labels != -1
            loss, _ = loss_fn(out["alpha"][mask], labels[mask])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item() * mask.sum().item()
            total_samples += mask.sum().item()

        dev_wf1, dev_u, _ = evaluate(model, dev_loader, device)
        elapsed = time.time() - start
        logger.info(f"Epoch {epoch:3d}/{args.epochs} | Loss: {total_loss/max(total_samples,1):.4f} | "
                    f"Dev WF1: {dev_wf1:.4f} u={dev_u:.3f} | {elapsed:.1f}s")

        if dev_wf1 > best_wf1:
            best_wf1 = dev_wf1
            patience_cnt = 0
            ckpt = Path(args.save_dir) / f"best_edl_{dataset_name}.pt"
            ckpt.parent.mkdir(exist_ok=True)
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, ckpt)
            logger.info(f"  >> New best! WF1={dev_wf1:.4f}")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                logger.info(f"  Early stopping at epoch {epoch}")
                break

    # Final test
    ckpt = Path(args.save_dir) / f"best_edl_{dataset_name}.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, weights_only=False)["model_state_dict"])
    test_wf1, test_u, test_report = evaluate(model, test_loader, device, emotions)

    logger.info(f"\n{'='*60}")
    logger.info(f"  RESULT: EDL Centralized — {dataset_name.upper()}")
    logger.info(f"{'='*60}")
    logger.info(f"\n{test_report}")
    logger.info(f"  Test WF1 = {test_wf1:.4f}, u = {test_u:.4f}")
    logger.info(f"{'='*60}")

    return test_wf1, test_u


def train_federated(dataset_name, train_dias, dev_dias, test_dias,
                    emotions, weights, cache, num_speakers, args):
    """Train EAFA federated on one dataset."""
    device = args.device
    num_classes = len(emotions)
    class_weights = torch.from_numpy(weights.astype(np.float32)).to(device)

    # Partition
    partitioner = FederatedPartitioner(
        num_clients=args.num_clients, strategy="dirichlet",
        alpha=args.alpha, seed=args.seed,
    )
    client_partitions = partitioner.partition(train_dias, label_ratio=1.0)

    dialogue_lookup = {d.dialogue_id: d for d in train_dias}
    client_loaders = []
    for partition in client_partitions:
        dias = [dialogue_lookup[did] for did in partition.dialogue_ids if did in dialogue_lookup]
        ds = GenericDialogueDataset(dias, cache.get("train", {}))
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                            collate_fn=collate_dialogues, num_workers=0)
        client_loaders.append(loader)

    test_ds = GenericDialogueDataset(test_dias, cache.get("test", {}))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_dialogues, num_workers=0)

    global_model = EvidentialDialogueRNN(
        input_dim=768, hidden_dim=args.hidden_dim,
        num_classes=num_classes, num_speakers=num_speakers, dropout=args.dropout,
    ).to(device)

    loss_fn = SupervisedEvidentialLoss(
        num_classes=num_classes, annealing_epochs=args.annealing_epochs,
        class_weights=class_weights,
    )
    aggregator = EAFAAggregator(beta=args.beta)

    logger.info(f"\n{'='*60}")
    logger.info(f"  EAFA Federated — {dataset_name.upper()} ({num_classes} classes)")
    logger.info(f"  {args.num_clients} clients, alpha={args.alpha}, beta={args.beta}")
    logger.info(f"{'='*60}\n")

    best_wf1, patience_cnt = 0.0, 0
    for round_num in range(1, args.num_rounds + 1):
        start = time.time()
        client_states, client_sizes, client_us = [], [], []

        for loader in client_loaders:
            local_model = copy.deepcopy(global_model).to(device)
            local_model.train()
            loss_fn.set_epoch(round_num)
            opt = optim.Adam(local_model.parameters(), lr=args.lr, weight_decay=1e-4)
            all_u_local = []

            for _ in range(args.local_epochs):
                for batch in loader:
                    feats = batch["features"].to(device)
                    speakers = batch["speaker_ids"].to(device)
                    labels = batch["labels"].to(device)
                    out = local_model(feats, speakers)
                    mask = labels != -1
                    loss, _ = loss_fn(out["alpha"][mask], labels[mask])
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), 5.0)
                    opt.step()
                    all_u_local.extend(out["uncertainty"][mask].detach().cpu().numpy())

            client_states.append(OrderedDict({k: v.cpu() for k, v in local_model.state_dict().items()}))
            client_sizes.append(len(loader.dataset))
            client_us.append(float(np.mean(all_u_local)) if all_u_local else 1.0)

        global_state, agg_stats = aggregator.aggregate(
            client_states, client_sizes, client_us, round_num,
        )
        global_model.load_state_dict(global_state)
        global_model.to(device)

        test_wf1, test_u, _ = evaluate(global_model, test_loader, device)
        elapsed = time.time() - start
        w_str = ",".join(f"{w:.2f}" for w in agg_stats["weights"])
        logger.info(f"Round {round_num:3d}/{args.num_rounds} | Test WF1: {test_wf1:.4f} u={test_u:.3f} | w=[{w_str}] | {elapsed:.1f}s")

        if test_wf1 > best_wf1:
            best_wf1 = test_wf1
            patience_cnt = 0
            ckpt = Path(args.save_dir) / f"best_eafa_{dataset_name}.pt"
            ckpt.parent.mkdir(exist_ok=True)
            torch.save({"round": round_num, "model_state_dict": global_model.state_dict()}, ckpt)
            logger.info(f"  >> New best! WF1={test_wf1:.4f}")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                logger.info(f"  Early stopping at round {round_num}")
                break

        del client_states
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Final
    ckpt = Path(args.save_dir) / f"best_eafa_{dataset_name}.pt"
    if ckpt.exists():
        global_model.load_state_dict(torch.load(ckpt, weights_only=False)["model_state_dict"])
    test_wf1, test_u, test_report = evaluate(global_model, test_loader, device, emotions)

    logger.info(f"\n{'='*60}")
    logger.info(f"  RESULT: EAFA Federated — {dataset_name.upper()}")
    logger.info(f"{'='*60}")
    logger.info(f"\n{test_report}")
    logger.info(f"  Test WF1 = {test_wf1:.4f}, u = {test_u:.4f}")
    logger.info(f"{'='*60}")

    return test_wf1, test_u


def main():
    import argparse
    parser = argparse.ArgumentParser(description="LucBinh Multi-Dataset")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["meld", "iemocap", "dailydialog", "all"])
    parser.add_argument("--mode", type=str, default="both",
                        choices=["centralized", "federated", "both"])
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--annealing_epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=20)
    # FL
    parser.add_argument("--num_clients", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--num_rounds", type=int, default=50)
    parser.add_argument("--local_epochs", type=int, default=3)
    parser.add_argument("--beta", type=float, default=1.0)
    # General
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    datasets_to_run = ["meld", "iemocap", "dailydialog"] if args.dataset == "all" else [args.dataset]
    loaders = {"meld": load_meld, "iemocap": load_iemocap, "dailydialog": load_dailydialog}

    all_results = {}

    for ds_name in datasets_to_run:
        logger.info(f"\n{'#'*60}")
        logger.info(f"  Dataset: {ds_name.upper()}")
        logger.info(f"{'#'*60}")

        train, dev, test, emotions, weights, cache, num_spk = loaders[ds_name]()

        if args.mode in ("centralized", "both"):
            wf1_c, u_c = train_centralized(
                ds_name, train, dev, test, emotions, weights, cache, num_spk, args,
            )
            all_results[f"{ds_name}_edl"] = wf1_c

        if args.mode in ("federated", "both"):
            wf1_f, u_f = train_federated(
                ds_name, train, dev, test, emotions, weights, cache, num_spk, args,
            )
            all_results[f"{ds_name}_eafa"] = wf1_f

    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info(f"  FINAL SUMMARY — LucBinh Multi-Dataset")
    logger.info(f"{'='*60}")
    for key, wf1 in all_results.items():
        logger.info(f"  {key:<30} WF1 = {wf1:.4f}")
    logger.info(f"{'='*60}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
