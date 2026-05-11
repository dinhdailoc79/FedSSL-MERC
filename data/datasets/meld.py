"""
MELD Dataset Loader
====================
Multimodal EmotionLines Dataset — Text + Audio + Video
Source: https://github.com/declare-lab/MELD

Dataset Structure:
    MELD.Raw/
    ├── train/                  # Training video clips
    ├── dev/                    # Development video clips
    ├── test/                   # Test video clips
    ├── train_sent_emo.csv      # Train annotations
    ├── dev_sent_emo.csv        # Dev annotations
    └── test_sent_emo.csv       # Test annotations

CSV Columns:
    Sr No., Utterance, Speaker, Emotion, Sentiment,
    Dialogue_ID, Utterance_ID, Season, Episode, StartTime, EndTime

Emotions: Anger, Disgust, Sadness, Joy, Neutral, Surprise, Fear (7 classes)
Sentiment: positive, negative, neutral (3 classes)

Video naming: diaX1_uttX2.mp4 (X1=Dialogue_ID, X2=Utterance_ID)
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ============================================================
# Constants
# ============================================================

MELD_EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
MELD_SENTIMENTS = ["negative", "neutral", "positive"]

EMOTION_TO_IDX = {e: i for i, e in enumerate(MELD_EMOTIONS)}
SENTIMENT_TO_IDX = {s: i for i, s in enumerate(MELD_SENTIMENTS)}

MELD_RAW_URL = "https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz"
MELD_GITHUB_URL = "https://github.com/declare-lab/MELD"

SPLITS = ["train", "dev", "test"]


# ============================================================
# Data Classes
# ============================================================

@dataclass
class Utterance:
    """Single utterance in a conversation."""
    utterance_id: int
    dialogue_id: int
    text: str
    speaker: str
    emotion: str
    sentiment: str
    emotion_idx: int = -1
    sentiment_idx: int = -1
    # Feature placeholders (populated during preprocessing)
    text_features: Optional[np.ndarray] = None
    audio_features: Optional[np.ndarray] = None
    video_features: Optional[np.ndarray] = None
    video_path: Optional[str] = None

    def __post_init__(self):
        self.emotion_idx = EMOTION_TO_IDX.get(self.emotion.lower(), -1)
        self.sentiment_idx = SENTIMENT_TO_IDX.get(self.sentiment.lower(), -1)


@dataclass
class Dialogue:
    """A conversation consisting of multiple utterances."""
    dialogue_id: int
    utterances: List[Utterance] = field(default_factory=list)

    @property
    def num_utterances(self) -> int:
        return len(self.utterances)

    @property
    def speakers(self) -> List[str]:
        return list(set(u.speaker for u in self.utterances))

    @property
    def emotion_labels(self) -> List[int]:
        return [u.emotion_idx for u in self.utterances]

    def add_utterance(self, utterance: Utterance):
        self.utterances.append(utterance)


# ============================================================
# MELD Dataset
# ============================================================

class MELDDataset:
    """
    MELD Dataset handler for Multimodal Emotion Recognition in Conversations.

    Handles loading, preprocessing, and serving of the MELD dataset
    with support for text, audio, and video modalities.

    Usage:
        >>> dataset = MELDDataset(data_dir="./data/raw/MELD")
        >>> train_dialogues = dataset.load_split("train")
        >>> print(f"Loaded {len(train_dialogues)} training dialogues")
        >>> dataset.print_stats("train")
    """

    def __init__(self, data_dir: str, modalities: List[str] = None):
        """
        Args:
            data_dir: Path to MELD dataset root directory
            modalities: List of modalities to load. Options: ["text", "audio", "video"]
                        Defaults to ["text"] (always available)
        """
        self.data_dir = Path(data_dir)
        self.modalities = modalities or ["text"]
        self.dialogues: Dict[str, Dict[int, Dialogue]] = {}  # split -> {dialogue_id -> Dialogue}
        self._raw_data: Dict[str, pd.DataFrame] = {}

        logger.info(f"MELDDataset initialized: data_dir={data_dir}, modalities={self.modalities}")

    # ----------------------------------------------------------
    # Loading
    # ----------------------------------------------------------

    def load_split(self, split: str) -> Dict[int, Dialogue]:
        """
        Load a data split (train/dev/test) and organize into Dialogues.

        Args:
            split: One of "train", "dev", "test"

        Returns:
            Dictionary mapping dialogue_id -> Dialogue
        """
        if split not in SPLITS:
            raise ValueError(f"Invalid split '{split}'. Must be one of {SPLITS}")

        if split in self.dialogues:
            logger.info(f"Split '{split}' already loaded, returning cached version")
            return self.dialogues[split]

        csv_path = self.data_dir / f"{split}_sent_emo.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"CSV not found: {csv_path}\n"
                f"Please download MELD from {MELD_GITHUB_URL}\n"
                f"Or run: python scripts/download_meld.py --output {self.data_dir}"
            )

        logger.info(f"Loading MELD {split} split from {csv_path}")
        df = pd.read_csv(csv_path)
        self._raw_data[split] = df

        # Build dialogues
        dialogues: Dict[int, Dialogue] = {}

        for _, row in df.iterrows():
            dia_id = int(row["Dialogue_ID"])
            utt_id = int(row["Utterance_ID"])

            # Build video path if video modality requested
            video_path = None
            if "video" in self.modalities or "audio" in self.modalities:
                video_file = f"dia{dia_id}_utt{utt_id}.mp4"
                video_path = str(self.data_dir / split / video_file)

            utterance = Utterance(
                utterance_id=utt_id,
                dialogue_id=dia_id,
                text=str(row.get("Utterance", "")),
                speaker=str(row.get("Speaker", "Unknown")),
                emotion=str(row.get("Emotion", "neutral")),
                sentiment=str(row.get("Sentiment", "neutral")),
                video_path=video_path,
            )

            if dia_id not in dialogues:
                dialogues[dia_id] = Dialogue(dialogue_id=dia_id)
            dialogues[dia_id].add_utterance(utterance)

        # Sort utterances within each dialogue by utterance_id
        for dialogue in dialogues.values():
            dialogue.utterances.sort(key=lambda u: u.utterance_id)

        self.dialogues[split] = dialogues
        logger.info(
            f"Loaded {split}: {len(dialogues)} dialogues, "
            f"{sum(d.num_utterances for d in dialogues.values())} utterances"
        )
        return dialogues

    def load_all(self) -> Dict[str, Dict[int, Dialogue]]:
        """Load all splits (train, dev, test)."""
        for split in SPLITS:
            self.load_split(split)
        return self.dialogues

    # ----------------------------------------------------------
    # Statistics
    # ----------------------------------------------------------

    def get_stats(self, split: str) -> Dict[str, Any]:
        """Get detailed statistics for a split."""
        if split not in self.dialogues:
            self.load_split(split)

        dialogues = self.dialogues[split]
        all_utterances = [u for d in dialogues.values() for u in d.utterances]

        # Emotion distribution
        emotion_counts = {}
        for u in all_utterances:
            emotion_counts[u.emotion] = emotion_counts.get(u.emotion, 0) + 1

        # Speaker stats
        all_speakers = set()
        for d in dialogues.values():
            all_speakers.update(d.speakers)

        # Dialogue length distribution
        lengths = [d.num_utterances for d in dialogues.values()]

        return {
            "split": split,
            "num_dialogues": len(dialogues),
            "num_utterances": len(all_utterances),
            "num_unique_speakers": len(all_speakers),
            "avg_dialogue_length": np.mean(lengths),
            "max_dialogue_length": max(lengths),
            "min_dialogue_length": min(lengths),
            "emotion_distribution": emotion_counts,
        }

    def print_stats(self, split: str):
        """Print formatted statistics for a split."""
        stats = self.get_stats(split)

        print(f"\n{'='*50}")
        print(f"  MELD Dataset — {stats['split'].upper()} Split")
        print(f"{'='*50}")
        print(f"  Dialogues:        {stats['num_dialogues']:,}")
        print(f"  Utterances:       {stats['num_utterances']:,}")
        print(f"  Unique Speakers:  {stats['num_unique_speakers']}")
        print(f"  Avg Dialog Len:   {stats['avg_dialogue_length']:.1f}")
        print(f"  Min/Max Len:      {stats['min_dialogue_length']}/{stats['max_dialogue_length']}")
        print(f"\n  Emotion Distribution:")
        print(f"  {'─'*40}")

        total = stats["num_utterances"]
        for emotion in MELD_EMOTIONS:
            count = stats["emotion_distribution"].get(emotion, 0)
            pct = count / total * 100
            bar = "█" * int(pct / 2)
            print(f"    {emotion:<12} {count:>5} ({pct:5.1f}%) {bar}")

        print(f"{'='*50}\n")

    # ----------------------------------------------------------
    # Data Access
    # ----------------------------------------------------------

    def get_dialogues(self, split: str) -> List[Dialogue]:
        """Get list of dialogues for a split."""
        if split not in self.dialogues:
            self.load_split(split)
        return list(self.dialogues[split].values())

    def get_flat_utterances(self, split: str) -> List[Utterance]:
        """Get flat list of all utterances (loses dialogue structure)."""
        dialogues = self.get_dialogues(split)
        return [u for d in dialogues for u in d.utterances]

    def get_emotion_weights(self, split: str) -> np.ndarray:
        """
        Compute inverse frequency class weights for handling class imbalance.
        Useful for weighted loss functions.

        Returns:
            np.ndarray of shape (num_emotions,) with class weights
        """
        utterances = self.get_flat_utterances(split)
        labels = [u.emotion_idx for u in utterances]

        counts = np.zeros(len(MELD_EMOTIONS))
        for label in labels:
            if label >= 0:
                counts[label] += 1

        # Inverse frequency weighting
        total = counts.sum()
        weights = total / (len(MELD_EMOTIONS) * counts + 1e-8)
        return weights / weights.sum() * len(MELD_EMOTIONS)


# ============================================================
# Download Helper
# ============================================================

def download_meld(output_dir: str, include_raw: bool = False):
    """
    Download MELD dataset files.

    Args:
        output_dir: Directory to save the dataset
        include_raw: Whether to download raw video files (large, ~4GB)
    """
    import urllib.request
    import tarfile

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Download CSV annotations from GitHub
    csv_base_url = "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD"
    for split in SPLITS:
        csv_name = f"{split}_sent_emo.csv"
        csv_url = f"{csv_base_url}/{csv_name}"
        csv_path = output_path / csv_name

        if csv_path.exists():
            logger.info(f"CSV already exists: {csv_path}")
            continue

        logger.info(f"Downloading {csv_name}...")
        try:
            urllib.request.urlretrieve(csv_url, csv_path)
            logger.info(f"Saved to {csv_path}")
        except Exception as e:
            logger.error(f"Failed to download {csv_name}: {e}")
            print(f"⚠️  Manual download: {MELD_GITHUB_URL}")

    if include_raw:
        tar_path = output_path / "MELD.Raw.tar.gz"
        if not tar_path.exists():
            logger.info(f"Downloading raw video data (~4GB)...")
            print("📥 Downloading MELD.Raw.tar.gz (this may take a while)...")
            urllib.request.urlretrieve(MELD_RAW_URL, tar_path)

        logger.info("Extracting raw data...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(output_path)
        logger.info("Extraction complete!")

    print(f"✅ MELD dataset ready at: {output_path}")


if __name__ == "__main__":
    # Quick test
    import sys
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "./data/raw/MELD"

    dataset = MELDDataset(data_dir=data_dir)
    for split in SPLITS:
        try:
            dataset.load_split(split)
            dataset.print_stats(split)
        except FileNotFoundError as e:
            print(f"⚠️  {e}")
