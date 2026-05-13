"""
Feature Extraction Script
==========================
Extract text features using RoBERTa-base and save as cached .pt files.

Usage:
    python scripts/extract_features.py --dataset meld
    python scripts/extract_features.py --dataset iemocap
    python scripts/extract_features.py --dataset all

Output:
    data/features/meld_text_roberta.pt
    data/features/iemocap_text_roberta.pt
"""

import sys
import os
import time
import logging
import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from tqdm import tqdm

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_text_features(
    texts: List[str],
    model_name: str = "roberta-base",
    device: str = "cuda",
    batch_size: int = 32,
    max_length: int = 128,
) -> np.ndarray:
    """
    Extract CLS token features from a pre-trained language model.

    Args:
        texts: List of utterance strings
        model_name: HuggingFace model name
        device: 'cuda' or 'cpu'
        batch_size: Batch size for inference (32 fits RTX 4050 6GB)
        max_length: Max token length

    Returns:
        np.ndarray of shape (num_texts, hidden_dim)
    """
    from transformers import AutoTokenizer, AutoModel

    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_features = []

    logger.info(f"Extracting features for {len(texts)} texts (batch_size={batch_size})...")
    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting"):
        batch_texts = texts[i:i + batch_size]

        # Handle empty/None texts
        batch_texts = [t if t and isinstance(t, str) else "" for t in batch_texts]

        encoding = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoding)
            # CLS token = first token
            cls_features = outputs.last_hidden_state[:, 0, :]
            all_features.append(cls_features.cpu().numpy())

    features = np.concatenate(all_features, axis=0)
    logger.info(f"Extracted features shape: {features.shape}")
    return features


def extract_meld(data_dir: str, output_dir: str, device: str, batch_size: int):
    """Extract text features for MELD dataset."""
    from data.datasets.meld import MELDDataset, MELD_EMOTIONS

    logger.info("=" * 60)
    logger.info("  Extracting MELD text features")
    logger.info("=" * 60)

    dataset = MELDDataset(data_dir=data_dir)
    result = {}

    for split in ["train", "dev", "test"]:
        logger.info(f"\n--- {split.upper()} ---")
        dialogues = dataset.get_dialogues(split)

        # Collect all utterances with metadata
        all_texts = []
        all_labels = []
        all_speakers = []
        all_dialogue_ids = []
        all_utterance_ids = []

        for dialogue in dialogues:
            for utt in dialogue.utterances:
                all_texts.append(utt.text)
                all_labels.append(utt.emotion_idx)
                all_speakers.append(utt.speaker)
                all_dialogue_ids.append(utt.dialogue_id)
                all_utterance_ids.append(utt.utterance_id)

        # Extract features
        features = extract_text_features(
            all_texts, device=device, batch_size=batch_size
        )

        result[split] = {
            "features": torch.from_numpy(features),          # (N, 768)
            "labels": torch.tensor(all_labels),               # (N,)
            "speakers": all_speakers,                          # List[str]
            "dialogue_ids": torch.tensor(all_dialogue_ids),   # (N,)
            "utterance_ids": torch.tensor(all_utterance_ids), # (N,)
            "texts": all_texts,                                # List[str]
        }

        logger.info(
            f"  {split}: {len(all_texts)} utterances, "
            f"features={features.shape}, "
            f"labels distribution: {dict(zip(*np.unique(all_labels, return_counts=True)))}"
        )

    # Save
    output_path = Path(output_dir) / "meld_text_roberta.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output_path)
    logger.info(f"\n✅ Saved MELD features to: {output_path}")
    logger.info(f"   File size: {output_path.stat().st_size / 1e6:.1f} MB")

    # Print summary
    logger.info("\n📊 MELD Feature Summary:")
    for split in ["train", "dev", "test"]:
        n = len(result[split]["texts"])
        dim = result[split]["features"].shape[1]
        logger.info(f"   {split:>5}: {n:>6} utterances × {dim}-dim")


def extract_iemocap(data_dir: str, output_dir: str, device: str, batch_size: int, num_classes: int = 4):
    """Extract text features for IEMOCAP dataset."""
    from data.datasets.iemocap import IEMOCAPDataset

    logger.info("=" * 60)
    logger.info(f"  Extracting IEMOCAP text features ({num_classes}-class)")
    logger.info("=" * 60)

    dataset = IEMOCAPDataset(data_dir=data_dir, num_classes=num_classes)
    dataset.load()

    # IEMOCAP uses session-based splits, store per-session
    result = {"num_classes": num_classes, "emotions": dataset.emotions}

    for session_id in range(1, 6):
        logger.info(f"\n--- Session {session_id} ---")
        dialogues = dataset.get_dialogues(session=session_id)

        all_texts = []
        all_labels = []
        all_speakers = []
        all_dialogue_ids = []
        all_utterance_ids = []

        for dialogue in dialogues:
            for utt in dialogue.utterances:
                # IEMOCAP text may need to be loaded from transcription files
                text = utt.text if utt.text else f"[utterance {utt.utterance_id}]"
                all_texts.append(text)
                all_labels.append(utt.emotion_idx)
                all_speakers.append(utt.speaker)
                all_dialogue_ids.append(utt.dialogue_id)
                all_utterance_ids.append(utt.utterance_id)

        if not all_texts:
            logger.warning(f"  Session {session_id}: No utterances found")
            continue

        # Extract features
        features = extract_text_features(
            all_texts, device=device, batch_size=batch_size
        )

        result[f"session{session_id}"] = {
            "features": torch.from_numpy(features),
            "labels": torch.tensor(all_labels),
            "speakers": all_speakers,
            "dialogue_ids": all_dialogue_ids,
            "utterance_ids": all_utterance_ids,
            "texts": all_texts,
        }

        logger.info(
            f"  Session {session_id}: {len(all_texts)} utterances, "
            f"features={features.shape}"
        )

    # Save
    output_path = Path(output_dir) / f"iemocap_text_roberta_{num_classes}class.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output_path)
    logger.info(f"\n✅ Saved IEMOCAP features to: {output_path}")
    logger.info(f"   File size: {output_path.stat().st_size / 1e6:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Extract text features")
    parser.add_argument("--dataset", type=str, default="meld",
                        choices=["meld", "iemocap", "all"])
    parser.add_argument("--meld_dir", type=str, default="data/raw/MELD")
    parser.add_argument("--iemocap_dir", type=str, default="data/raw/IEMOCAP/IEMOCAP_full_release")
    parser.add_argument("--output_dir", type=str, default="data/features")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size (32 fits RTX 4050 6GB)")
    parser.add_argument("--iemocap_classes", type=int, default=4,
                        choices=[4, 6], help="IEMOCAP emotion classes")
    args = parser.parse_args()

    logger.info(f"Device: {args.device}")
    logger.info(f"Output: {args.output_dir}")

    # Check CUDA
    if args.device == "cuda":
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            logger.warning("CUDA not available, falling back to CPU")
            args.device = "cpu"

    start_time = time.time()

    if args.dataset in ["meld", "all"]:
        extract_meld(args.meld_dir, args.output_dir, args.device, args.batch_size)

    if args.dataset in ["iemocap", "all"]:
        extract_iemocap(
            args.iemocap_dir, args.output_dir, args.device,
            args.batch_size, args.iemocap_classes
        )

    elapsed = time.time() - start_time
    logger.info(f"\n⏱️ Total time: {elapsed / 60:.1f} minutes")
    logger.info("Done!")


if __name__ == "__main__":
    main()
