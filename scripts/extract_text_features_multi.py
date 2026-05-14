"""
Extract RoBERTa text features for IEMOCAP and DailyDialog.

Reuses the same approach as MELD: frozen RoBERTa-base → mean pool → 768-dim per utterance.

Usage:
    python scripts/extract_text_features_multi.py
"""

import sys
import os
import logging
from pathlib import Path

import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_roberta_features(texts, tokenizer, model, device, batch_size=32):
    """Extract RoBERTa features for a list of texts."""
    all_features = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=128,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pool over tokens (excluding padding)
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            all_features.append(pooled.cpu())

        if (i // batch_size + 1) % 50 == 0:
            logger.info(f"  Processed {i + len(batch_texts)}/{len(texts)}")

    return torch.cat(all_features, dim=0)


def extract_iemocap(tokenizer, model, device, batch_size=32):
    """Extract features for IEMOCAP dataset."""
    from data.datasets.iemocap import IEMOCAPDataset

    logger.info("\n" + "=" * 60)
    logger.info("  IEMOCAP Feature Extraction")
    logger.info("=" * 60)

    dataset = IEMOCAPDataset(
        data_dir="data/raw/IEMOCAP/IEMOCAP_full_release", num_classes=6,
    )
    dataset.load()

    # IEMOCAP uses session-based splits (standard: test=session5)
    # Extract features for ALL utterances, store by session
    results = {}
    for session in range(1, 6):
        dialogues = dataset.get_dialogues(session=session)
        texts, dia_ids, utt_ids = [], [], []

        for d in dialogues:
            for u in d.utterances:
                texts.append(u.text if u.text else " ")
                dia_ids.append(hash(d.dialogue_id) % (2**31))
                utt_ids.append(hash(u.utterance_id) % (2**31))

        logger.info(f"  Session {session}: {len(texts)} utterances")
        features = extract_roberta_features(texts, tokenizer, model, device, batch_size)

        results[f"session{session}"] = {
            "features": features,
            "dialogue_ids": torch.tensor(dia_ids),
            "utterance_ids": torch.tensor(utt_ids),
            "texts": texts,
            # Store original IDs for lookup
            "dia_id_strs": [d.dialogue_id for d in dialogues for _ in d.utterances],
            "utt_id_strs": [u.utterance_id for d in dialogues for u in d.utterances],
        }

    output_path = Path("data/features/iemocap_text_roberta.pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, str(output_path))
    size_mb = output_path.stat().st_size / 1e6
    logger.info(f"  Saved: {output_path} ({size_mb:.1f} MB)")
    return results


def extract_dailydialog(tokenizer, model, device, batch_size=32):
    """Extract features for DailyDialog dataset."""
    from data.datasets.dailydialog import DailyDialogDataset

    logger.info("\n" + "=" * 60)
    logger.info("  DailyDialog Feature Extraction")
    logger.info("=" * 60)

    dataset = DailyDialogDataset(
        data_dir="data/raw/DailyDialog", exclude_no_emotion_dialogues=False,
    )

    results = {}
    for split in ["train", "dev", "test"]:
        dialogues = dataset.get_dialogues(split)
        texts, dia_ids, utt_ids = [], [], []

        for d in dialogues:
            for u in d.utterances:
                texts.append(u.text if u.text else " ")
                dia_ids.append(d.dialogue_id)
                utt_ids.append(u.utterance_id)

        logger.info(f"  {split}: {len(texts)} utterances")
        features = extract_roberta_features(texts, tokenizer, model, device, batch_size)

        results[split] = {
            "features": features,
            "dialogue_ids": torch.tensor(dia_ids),
            "utterance_ids": torch.tensor(utt_ids),
        }

    output_path = Path("data/features/dailydialog_text_roberta.pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, str(output_path))
    size_mb = output_path.stat().st_size / 1e6
    logger.info(f"  Saved: {output_path} ({size_mb:.1f} MB)")
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--datasets", nargs="+", default=["iemocap", "dailydialog"])
    args = parser.parse_args()

    logger.info(f"Loading RoBERTa-base on {args.device}...")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaModel.from_pretrained("roberta-base").to(args.device)
    model.eval()
    logger.info("Model loaded!")

    if "iemocap" in args.datasets:
        extract_iemocap(tokenizer, model, args.device, args.batch_size)

    if "dailydialog" in args.datasets:
        extract_dailydialog(tokenizer, model, args.device, args.batch_size)

    logger.info("\nAll done!")


if __name__ == "__main__":
    main()
