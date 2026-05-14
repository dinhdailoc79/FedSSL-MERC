"""
DailyDialog Dataset Loader
============================
Source: Li et al. (2017) "DailyDialog: A Manually Labelled Multi-Turn Dialogue Dataset"

Emotions (original 7-class):
    0: no_emotion, 1: anger, 2: disgust, 3: fear, 4: happiness, 5: sadness, 6: surprise

Standard protocol: Exclude 'no_emotion' utterances from loss/evaluation
but KEEP them in dialogue for context (DialogueRNN needs full sequence).

Data format: CSV with columns [Dialogue_ID, Utterance_ID, Utterance, Emotion, Act]
"""

import os
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 6-class (excluding no_emotion)
DAILYDIALOG_EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]

# Map original CSV emotion IDs to our indices (-1 for no_emotion = masked)
EMOTION_ID_MAP = {
    0: -1,  # no_emotion → mask (keep in dialogue, skip in loss)
    1: 0,   # anger
    2: 1,   # disgust
    3: 2,   # fear
    4: 3,   # happiness
    5: 4,   # sadness
    6: 5,   # surprise
}


@dataclass
class DDUtterance:
    """Single utterance in a DailyDialog conversation."""
    utterance_id: int
    text: str
    emotion_raw: int       # Original CSV emotion (0-6)
    emotion: str           # Standardized name
    emotion_idx: int       # Index in DAILYDIALOG_EMOTIONS (-1 = no_emotion)
    speaker_id: int = 0    # Alternating speakers (0, 1, 0, 1, ...)


@dataclass
class DDDialogue:
    """A dialogue in DailyDialog."""
    dialogue_id: int
    utterances: List[DDUtterance] = field(default_factory=list)

    @property
    def num_utterances(self):
        return len(self.utterances)

    @property
    def emotion_labels(self) -> List[int]:
        return [u.emotion_idx for u in self.utterances]

    @property
    def has_emotion(self) -> bool:
        """True if dialogue has at least 1 non-no_emotion utterance."""
        return any(u.emotion_idx >= 0 for u in self.utterances)


class DailyDialogDataset:
    """
    DailyDialog Dataset handler for ERC.

    Usage:
        >>> dataset = DailyDialogDataset(data_dir="data/raw/DailyDialog")
        >>> train = dataset.get_dialogues("train")
        >>> dataset.print_stats("train")
    """

    def __init__(self, data_dir: str, exclude_no_emotion_dialogues: bool = True):
        """
        Args:
            data_dir: Path to DailyDialog directory (contains train.csv, etc.)
            exclude_no_emotion_dialogues: If True, skip dialogues with ALL no_emotion
        """
        self.data_dir = Path(data_dir)
        self.exclude_no_emotion_dialogues = exclude_no_emotion_dialogues
        self.emotions = DAILYDIALOG_EMOTIONS
        self.num_classes = len(self.emotions)
        self.emotion_to_idx = {e: i for i, e in enumerate(self.emotions)}

        self._splits: Dict[str, List[DDDialogue]] = {}

        logger.info(
            f"DailyDialogDataset initialized: data_dir={data_dir}, "
            f"num_classes={self.num_classes}, exclude_pure_no_emotion={exclude_no_emotion_dialogues}"
        )

    def _load_split(self, split: str) -> List[DDDialogue]:
        """Load a split from CSV."""
        # Map split names
        csv_name = "validation" if split == "dev" else split
        csv_path = self.data_dir / f"{csv_name}.csv"

        if not csv_path.exists():
            logger.warning(f"CSV not found: {csv_path}")
            return []

        logger.info(f"Loading DailyDialog {split} from {csv_path}")
        df = pd.read_csv(csv_path)

        # Emotion name lookup (original IDs)
        emo_names = {0: "no_emotion", 1: "anger", 2: "disgust", 3: "fear",
                     4: "happiness", 5: "sadness", 6: "surprise"}

        dialogues = []
        for dia_id, group in df.groupby("Dialogue_ID"):
            dialogue = DDDialogue(dialogue_id=int(dia_id))

            for _, row in group.iterrows():
                raw_emo = int(row["Emotion"])
                emotion_idx = EMOTION_ID_MAP.get(raw_emo, -1)
                emotion_name = emo_names.get(raw_emo, "no_emotion")

                # Alternating speaker IDs (DailyDialog is dyadic)
                speaker_id = int(row["Utterance_ID"]) % 2

                utt = DDUtterance(
                    utterance_id=int(row["Utterance_ID"]),
                    text=str(row["Utterance"]).strip(),
                    emotion_raw=raw_emo,
                    emotion=emotion_name,
                    emotion_idx=emotion_idx,
                    speaker_id=speaker_id,
                )
                dialogue.utterances.append(utt)

            # Optionally skip dialogues with ALL no_emotion
            if self.exclude_no_emotion_dialogues and not dialogue.has_emotion:
                continue

            dialogues.append(dialogue)

        # Stats
        total_utt = sum(d.num_utterances for d in dialogues)
        emo_utt = sum(1 for d in dialogues for u in d.utterances if u.emotion_idx >= 0)
        logger.info(
            f"Loaded {split}: {len(dialogues)} dialogues, "
            f"{total_utt} utterances ({emo_utt} with emotion, "
            f"{total_utt - emo_utt} no_emotion kept as context)"
        )

        return dialogues

    def get_dialogues(self, split: str) -> List[DDDialogue]:
        """Get dialogues for a split ('train', 'dev', 'test')."""
        if split not in self._splits:
            self._splits[split] = self._load_split(split)
        return self._splits[split]

    def get_emotion_weights(self, split: str) -> np.ndarray:
        """Compute inverse frequency class weights (only counting emotion utterances)."""
        dialogues = self.get_dialogues(split)
        counts = np.zeros(self.num_classes)
        for d in dialogues:
            for u in d.utterances:
                if u.emotion_idx >= 0:
                    counts[u.emotion_idx] += 1

        total = counts.sum()
        weights = total / (self.num_classes * counts + 1e-8)
        return weights / weights.sum() * self.num_classes

    def print_stats(self, split: str):
        """Print formatted statistics."""
        dialogues = self.get_dialogues(split)
        all_utt = [u for d in dialogues for u in d.utterances]
        emo_utt = [u for u in all_utt if u.emotion_idx >= 0]

        print(f"\n{'='*50}")
        print(f"  DailyDialog — {split}")
        print(f"{'='*50}")
        print(f"  Dialogues:      {len(dialogues):,}")
        print(f"  Total utts:     {len(all_utt):,}")
        print(f"  Emotion utts:   {len(emo_utt):,} (used for loss/eval)")
        print(f"  Context utts:   {len(all_utt) - len(emo_utt):,} (no_emotion, context only)")

        print(f"\n  Emotion Distribution (6-class):")
        print(f"  {'-'*40}")
        for i, emo in enumerate(self.emotions):
            count = sum(1 for u in emo_utt if u.emotion_idx == i)
            pct = count / len(emo_utt) * 100 if emo_utt else 0
            bar = "#" * int(pct / 2)
            print(f"    {emo:<12} {count:>5} ({pct:5.1f}%) {bar}")
        print(f"{'='*50}\n")


if __name__ == "__main__":
    dataset = DailyDialogDataset(data_dir="data/raw/DailyDialog")
    for split in ["train", "dev", "test"]:
        dataset.print_stats(split)
    weights = dataset.get_emotion_weights("train")
    print("Class weights:", {e: f"{w:.3f}" for e, w in zip(DAILYDIALOG_EMOTIONS, weights)})
