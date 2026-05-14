"""
Multimodal Evidential Training Script
=========================================
Train MultimodalEvidentialDialogueRNN with text + audio via DS fusion.

Usage:
    python scripts/train_multimodal.py
    python scripts/train_multimodal.py --fusion_mode dempster
"""

import sys
import os
import time
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, classification_report

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.datasets.meld import MELDDataset, MELD_EMOTIONS
from models.evidential.multimodal_edl import MultimodalEvidentialDialogueRNN
from models.evidential.losses import SupervisedEvidentialLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class MultimodalDialogueDataset(Dataset):
    """Dataset combining text + audio features per dialogue."""

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

            # Text features
            if key in self.text_cache:
                text_feats.append(torch.from_numpy(self.text_cache[key]))
            else:
                text_feats.append(torch.zeros(self.text_dim))

            # Audio features
            if key in self.audio_cache:
                audio_feats.append(torch.from_numpy(self.audio_cache[key]))
            else:
                audio_feats.append(torch.zeros(self.audio_dim))

            labels.append(utt.emotion_idx)
            speakers.append(utt.speaker_id if hasattr(utt, 'speaker_id') else 0)

        return {
            "text_features": torch.stack(text_feats),
            "audio_features": torch.stack(audio_feats),
            "labels": torch.tensor(labels, dtype=torch.long),
            "speaker_ids": torch.tensor(speakers, dtype=torch.long),
        }


def collate_multimodal(batch):
    """Pad dialogues to same length."""
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
        "text_features": text_feat,
        "audio_features": audio_feat,
        "labels": labels,
        "speaker_ids": speakers,
    }


def load_feature_cache(path, splits):
    """Load a feature .pt file and return per-split caches."""
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
                key = f"{dia_ids[i].item()}_{utt_ids[i].item()}"
                cache[key] = feats[i]
            caches[split] = cache
            logger.info(f"  {split}: {len(cache)} features loaded from {Path(path).name}")
    return caches


@torch.no_grad()
def evaluate(model, loader, device, emotion_names=None):
    """Evaluate multimodal model."""
    model.eval()
    all_preds, all_labels, all_u = [], [], []
    all_u_text, all_u_audio = [], []

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
        all_u_text.extend(out["text_uncertainty"][mask].cpu().numpy())
        all_u_audio.extend(out["audio_uncertainty"][mask].cpu().numpy())

    wf1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    mean_u = np.mean(all_u)
    mean_u_text = np.mean(all_u_text)
    mean_u_audio = np.mean(all_u_audio)

    report = classification_report(
        all_labels, all_preds, target_names=emotion_names, digits=4, zero_division=0,
    ) if emotion_names else ""

    return wf1, mean_u, mean_u_text, mean_u_audio, report


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Multimodal EDL training")
    parser.add_argument("--data_dir", type=str, default="data/raw/MELD")
    parser.add_argument("--text_cache", type=str, default="data/features/meld_text_roberta.pt")
    parser.add_argument("--audio_cache", type=str, default="data/features/meld_audio_wavlm.pt")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--fusion_mode", type=str, default="evidence_sum",
                        choices=["evidence_sum", "dempster"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--annealing_epochs", type=int, default=30)
    parser.add_argument("--lambda_aux", type=float, default=0.3,
                        help="Weight for per-modality auxiliary EDL loss")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logger.info(f"\n{'='*60}")
    logger.info(f"  Multimodal EDL Training (text + audio)")
    logger.info(f"  Fusion: {args.fusion_mode}")
    logger.info(f"{'='*60}")

    # -------------------------------------------------------
    # 1. Load data + features
    # -------------------------------------------------------
    meld = MELDDataset(data_dir=args.data_dir)
    train_dialogues = meld.get_dialogues("train")
    dev_dialogues = meld.get_dialogues("dev")
    test_dialogues = meld.get_dialogues("test")

    # Load text features
    text_caches = load_feature_cache(args.text_cache, ["train", "dev", "test"])
    text_dim = 768

    # Load audio features
    audio_caches = load_feature_cache(args.audio_cache, ["train", "dev", "test"])
    audio_dim = 768

    # Check coverage
    for split, dialogues in [("train", train_dialogues), ("dev", dev_dialogues), ("test", test_dialogues)]:
        n_utts = sum(len(d.utterances) for d in dialogues)
        t_hit = sum(1 for d in dialogues for u in d.utterances
                    if f"{d.dialogue_id}_{u.utterance_id}" in text_caches.get(split, {}))
        a_hit = sum(1 for d in dialogues for u in d.utterances
                    if f"{d.dialogue_id}_{u.utterance_id}" in audio_caches.get(split, {}))
        logger.info(f"  {split}: {n_utts} utts, text={t_hit} ({100*t_hit/n_utts:.1f}%), "
                    f"audio={a_hit} ({100*a_hit/n_utts:.1f}%)")

    # DataLoaders
    train_ds = MultimodalDialogueDataset(
        train_dialogues, text_caches.get("train", {}), audio_caches.get("train", {}),
        text_dim, audio_dim,
    )
    dev_ds = MultimodalDialogueDataset(
        dev_dialogues, text_caches.get("dev", {}), audio_caches.get("dev", {}),
        text_dim, audio_dim,
    )
    test_ds = MultimodalDialogueDataset(
        test_dialogues, text_caches.get("test", {}), audio_caches.get("test", {}),
        text_dim, audio_dim,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_multimodal, num_workers=0)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_multimodal, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_multimodal, num_workers=0)

    # -------------------------------------------------------
    # 2. Model + Loss
    # -------------------------------------------------------
    class_weights = torch.from_numpy(
        meld.get_emotion_weights("train").astype(np.float32)
    ).to(args.device)

    model = MultimodalEvidentialDialogueRNN(
        text_dim=text_dim, audio_dim=audio_dim,
        hidden_dim=args.hidden_dim, num_classes=len(MELD_EMOTIONS),
        num_speakers=10, dropout=args.dropout,
        fusion_mode=args.fusion_mode,
    ).to(args.device)

    # Main loss (fused) + auxiliary losses (per-modality)
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

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {params:,}")

    # -------------------------------------------------------
    # 3. Training loop
    # -------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info(f"  Training: {args.epochs} epochs")
    logger.info(f"{'='*60}\n")

    best_wf1 = 0.0
    patience_counter = 0
    Path(args.save_dir).mkdir(exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_fn.set_epoch(epoch)
        aux_loss_fn.set_epoch(epoch)

        total_loss = 0
        total_samples = 0
        start = time.time()

        for batch in train_loader:
            text = batch["text_features"].to(args.device)
            audio = batch["audio_features"].to(args.device)
            speakers = batch["speaker_ids"].to(args.device)
            labels = batch["labels"].to(args.device)

            out = model(text, audio, speakers)
            mask = labels != -1
            labels_flat = labels[mask]

            # Main fused loss
            fused_loss, _ = loss_fn(out["alpha"][mask], labels_flat)

            # Auxiliary per-modality losses (helps each encoder learn)
            text_loss, _ = aux_loss_fn(out["text_alpha"][mask], labels_flat)
            audio_loss, _ = aux_loss_fn(out["audio_alpha"][mask], labels_flat)
            aux_loss = (text_loss + audio_loss) / 2.0

            # Total
            loss = fused_loss + args.lambda_aux * aux_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item() * labels_flat.size(0)
            total_samples += labels_flat.size(0)

        avg_loss = total_loss / max(total_samples, 1)

        # Evaluate on dev
        dev_wf1, dev_u, dev_u_t, dev_u_a, _ = evaluate(model, dev_loader, args.device)

        elapsed = time.time() - start
        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | Loss: {avg_loss:.4f} | "
            f"Dev WF1: {dev_wf1:.4f} u={dev_u:.3f} (text={dev_u_t:.3f} audio={dev_u_a:.3f}) | "
            f"{elapsed:.1f}s"
        )

        if dev_wf1 > best_wf1:
            best_wf1 = dev_wf1
            patience_counter = 0
            ckpt_path = Path(args.save_dir) / "best_multimodal_edl.pt"
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "dev_wf1": dev_wf1,
            }, ckpt_path)
            logger.info(f"  >> New best! Dev WF1={dev_wf1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"  Early stopping at epoch {epoch}")
                break

    # -------------------------------------------------------
    # 4. Final test evaluation
    # -------------------------------------------------------
    ckpt_path = Path(args.save_dir) / "best_multimodal_edl.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"\nLoaded best model from epoch {ckpt['epoch']}")

    test_wf1, test_u, test_u_t, test_u_a, test_report = evaluate(
        model, test_loader, args.device, MELD_EMOTIONS
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"  Final Test Evaluation (Multimodal EDL)")
    logger.info(f"{'='*60}")
    logger.info(f"\n{test_report}")
    logger.info(f"\n{'='*60}")
    logger.info(f"  COMPARISON")
    logger.info(f"{'='*60}")
    logger.info(f"  Softmax Centralized (text):      WF1 = 0.5442")
    logger.info(f"  EDL Centralized (text):           WF1 = 0.5747")
    logger.info(f"  Multimodal EDL (text+audio):      WF1 = {test_wf1:.4f}")
    logger.info(f"  Fused uncertainty:    u = {test_u:.4f}")
    logger.info(f"  Text uncertainty:     u = {test_u_t:.4f}")
    logger.info(f"  Audio uncertainty:    u = {test_u_a:.4f}")
    logger.info(f"{'='*60}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
