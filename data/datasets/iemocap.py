"""
IEMOCAP Dataset Loader
=======================
Interactive Emotional Dyadic Motion Capture Database
Source: https://sail.usc.edu/iemocap/

Dataset Structure:
    IEMOCAP_full_release/
    ├── Session1/
    │   ├── dialog/
    │   │   ├── EmoEvaluation/
    │   │   │   └── Categorical/     # Per-utterance emotion labels
    │   │   ├── lab/                  # Timing segmentation
    │   │   │   ├── Ses01_F/         # Female speaker segments
    │   │   │   └── Ses01_M/         # Male speaker segments
    │   │   └── avi/                  # Video (if available)
    │   └── sentences/
    │       ├── wav/                  # Per-utterance audio (.wav)
    │       ├── ForcedAlignment/      # Word-level alignment
    │       ├── MOCAP_hand/           # Motion capture data
    │       └── MOCAP_head/
    ├── Session2/ ... Session5/

Emotion Labels (Categorical):
    Format: "Ses01F_impro06_F000 :Neutral state; ()"
    Emotions: anger, happiness, sadness, neutral, frustration, excitement,
              fear, surprise, disgust, other

For ERC research (standard 4-class or 6-class):
    4-class: happy (happy+excited), sad, angry, neutral
    6-class: happy, sad, angry, neutral, frustrated, excited

Key Info:
    - 5 Sessions × 2 speakers = 10 speakers total
    - ~7,400 utterances (after filtering 'other' and 'xxx')
    - Dyadic conversations (2 speakers per session)
    - Speaker-based CV: Leave-one-session-out
"""

import os
import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================
# Constants
# ============================================================

# Standard 6-class setup (most Q1 papers)
IEMOCAP_EMOTIONS_6 = ["happy", "sad", "angry", "neutral", "frustrated", "excited"]
# Standard 4-class setup (some papers merge happy+excited, drop frustrated)
IEMOCAP_EMOTIONS_4 = ["happy", "sad", "angry", "neutral"]

# Map raw IEMOCAP labels to standardized labels
RAW_LABEL_MAP = {
    # Full names (from Categorical annotations)
    "neutral state": "neutral",
    "neutral": "neutral",
    "happiness": "happy",
    "happy": "happy",
    "sadness": "sad",
    "sad": "sad",
    "anger": "angry",
    "angry": "angry",
    "frustrated": "frustrated",
    "frustration": "frustrated",
    "excited": "excited",
    "excitement": "excited",
    "fear": "fear",
    "surprise": "surprise",
    "disgust": "disgust",
    "other": "other",
    "xxx": "other",
    # Abbreviated names (from EmoEvaluation consensus files)
    "neu": "neutral",
    "hap": "happy",
    "sad": "sad",
    "ang": "angry",
    "fru": "frustrated",
    "exc": "excited",
    "fea": "fear",
    "sur": "surprise",
    "dis": "disgust",
}

# For 4-class: merge excited → happy, drop frustrated
FOUR_CLASS_MAP = {
    "happy": "happy",
    "excited": "happy",
    "sad": "sad",
    "angry": "angry",
    "neutral": "neutral",
}

SESSION_IDS = [1, 2, 3, 4, 5]
SPEAKERS_PER_SESSION = {
    1: ("Ses01_F", "Ses01_M"),
    2: ("Ses02_F", "Ses02_M"),
    3: ("Ses03_F", "Ses03_M"),
    4: ("Ses04_F", "Ses04_M"),
    5: ("Ses05_F", "Ses05_M"),
}


# ============================================================
# Data Classes (reuse from MELD for consistency)
# ============================================================

@dataclass
class IEMOCAPUtterance:
    """Single utterance in an IEMOCAP conversation."""
    utterance_id: str           # e.g., "Ses01F_impro06_F000"
    dialogue_id: str            # e.g., "Ses01F_impro06"
    text: str                   # Transcription
    speaker: str                # Speaker ID (e.g., "Ses01_F")
    emotion_raw: str            # Raw label from annotations
    emotion: str                # Standardized label
    emotion_idx: int = -1       # Numeric index
    session: int = -1           # Session number (1-5)
    start_time: float = 0.0     # Start time in seconds
    end_time: float = 0.0       # End time in seconds
    # Feature placeholders
    text_features: Optional[np.ndarray] = None
    audio_features: Optional[np.ndarray] = None
    video_features: Optional[np.ndarray] = None
    wav_path: Optional[str] = None


@dataclass
class IEMOCAPDialogue:
    """A dialogue (conversation segment) in IEMOCAP."""
    dialogue_id: str
    session: int
    utterances: List[IEMOCAPUtterance] = field(default_factory=list)

    @property
    def num_utterances(self) -> int:
        return len(self.utterances)

    @property
    def speakers(self) -> List[str]:
        return list(set(u.speaker for u in self.utterances))

    @property
    def emotion_labels(self) -> List[int]:
        return [u.emotion_idx for u in self.utterances]

    def add_utterance(self, utt: IEMOCAPUtterance):
        self.utterances.append(utt)


# ============================================================
# IEMOCAP Dataset
# ============================================================

class IEMOCAPDataset:
    """
    IEMOCAP Dataset handler for Emotion Recognition in Conversations.

    Supports:
    - 4-class and 6-class emotion classification
    - Session-based speaker split (standard evaluation protocol)
    - Loading text, audio paths, and emotion labels

    Usage:
        >>> dataset = IEMOCAPDataset(data_dir="data/IEMOCAP/IEMOCAP_full_release")
        >>> dialogues = dataset.get_dialogues()
        >>> print(f"Total: {len(dialogues)} dialogues")
        >>> dataset.print_stats()

        # Session-based split (standard)
        >>> train, test = dataset.get_session_split(test_session=5)
    """

    def __init__(
        self,
        data_dir: str,
        num_classes: int = 4,
        modalities: List[str] = None,
    ):
        """
        Args:
            data_dir: Path to IEMOCAP_full_release directory
            num_classes: 4 or 6 (controls emotion label set)
            modalities: List of modalities ["text", "audio", "video"]
        """
        self.data_dir = Path(data_dir)
        self.num_classes = num_classes
        self.modalities = modalities or ["text"]

        if num_classes == 4:
            self.emotions = IEMOCAP_EMOTIONS_4
        elif num_classes == 6:
            self.emotions = IEMOCAP_EMOTIONS_6
        else:
            raise ValueError(f"num_classes must be 4 or 6, got {num_classes}")

        self.emotion_to_idx = {e: i for i, e in enumerate(self.emotions)}
        self._dialogues: Dict[str, IEMOCAPDialogue] = {}
        self._loaded = False

        logger.info(
            f"IEMOCAPDataset initialized: data_dir={data_dir}, "
            f"num_classes={num_classes}, modalities={self.modalities}"
        )

    # ----------------------------------------------------------
    # Loading
    # ----------------------------------------------------------

    def load(self) -> Dict[str, IEMOCAPDialogue]:
        """
        Load all sessions and parse emotion labels.

        Returns:
            Dictionary mapping dialogue_id -> IEMOCAPDialogue
        """
        if self._loaded:
            return self._dialogues

        logger.info("Loading IEMOCAP dataset...")

        for session_id in SESSION_IDS:
            session_dir = self.data_dir / f"Session{session_id}"
            if not session_dir.exists():
                logger.warning(f"Session{session_id} not found at {session_dir}, skipping")
                continue

            self._load_session(session_id, session_dir)

        self._loaded = True
        total_utt = sum(d.num_utterances for d in self._dialogues.values())
        logger.info(
            f"Loaded IEMOCAP: {len(self._dialogues)} dialogues, "
            f"{total_utt} utterances across {len(SESSION_IDS)} sessions"
        )
        return self._dialogues

    def _load_session(self, session_id: int, session_dir: Path):
        """
        Load all dialogues from a single session.

        Uses the consensus EmoEvaluation files (not Categorical, which has
        per-annotator labels causing duplicates).
        """
        # Use main EmoEvaluation dir (consensus labels)
        emo_dir = session_dir / "dialog" / "EmoEvaluation"

        if not emo_dir.exists():
            logger.warning(f"EmoEvaluation not found in Session{session_id}")
            return

        # Load transcriptions first
        transcriptions = self._load_transcriptions(session_dir)

        # Parse consensus emotion files (Ses*.txt in EmoEvaluation root)
        emo_files = sorted(emo_dir.glob("Ses*.txt"))
        if not emo_files:
            logger.warning(f"No emotion files found in {emo_dir}")
            return

        for emo_file in emo_files:
            self._parse_emotion_file(session_id, emo_file, transcriptions)

        # Sort utterances within each dialogue by utterance_id
        for d in self._dialogues.values():
            if d.session == session_id:
                d.utterances.sort(key=lambda u: u.utterance_id)

    def _load_transcriptions(self, session_dir: Path) -> Dict[str, str]:
        """
        Load transcription text for all utterances in a session.

        Reads from Session*/dialog/transcriptions/*.txt
        Format: "Ses01F_impro01_F000 [006.2901-008.2357]: Excuse me."

        Returns:
            Dictionary mapping utterance_id -> transcription text
        """
        trans_dir = session_dir / "dialog" / "transcriptions"
        transcriptions: Dict[str, str] = {}

        if not trans_dir.exists():
            logger.warning(f"Transcriptions not found at {trans_dir}")
            return transcriptions

        for trans_file in sorted(trans_dir.glob("*.txt")):
            try:
                lines = trans_file.read_text(encoding="utf-8", errors="ignore").strip().split("\n")
            except Exception as e:
                logger.warning(f"Failed to read {trans_file}: {e}")
                continue

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Format: "Ses01F_impro01_F000 [start-end]: text"
                match = re.match(
                    r"(Ses\S+)\s+\[[\d.]+-[\d.]+\]:\s*(.*)",
                    line,
                )
                if match:
                    utt_id = match.group(1)
                    text = match.group(2).strip()
                    transcriptions[utt_id] = text

        logger.info(f"  Loaded {len(transcriptions)} transcriptions from {trans_dir.parent.parent.name}")
        return transcriptions

    def _parse_emotion_file(
        self, session_id: int, emo_file: Path, transcriptions: Dict[str, str]
    ):
        """
        Parse a consensus emotion annotation file.

        Standard format (EmoEvaluation/Ses*.txt):
            [start - end]\tutterance_id\temotion\t[v, a, d]

        Each utterance appears once with the consensus label.
        """
        try:
            lines = emo_file.read_text(encoding="utf-8", errors="ignore").strip().split("\n")
        except Exception as e:
            logger.warning(f"Failed to read {emo_file}: {e}")
            return

        for line in lines:
            line = line.strip()
            if not line or line.startswith("%") or line.startswith("#"):
                continue

            # Parse consensus format: "[start - end]\tutt_id\temotion\t[v, a, d]"
            match = re.match(
                r"\[([\d.]+)\s*-\s*([\d.]+)\]\s+(\S+)\s+(\S+)\s+\[.*\]",
                line,
            )
            if not match:
                continue

            start_time = float(match.group(1))
            end_time = float(match.group(2))
            utt_id = match.group(3)
            emotion_raw = match.group(4).lower()

            # Map to standardized emotion
            emotion = RAW_LABEL_MAP.get(emotion_raw, "other")

            # Apply 4-class mapping if needed
            if self.num_classes == 4:
                emotion = FOUR_CLASS_MAP.get(emotion, None)
                if emotion is None:
                    continue  # Skip non-4-class emotions (frustrated, fear, etc.)

            # Skip 'other' labels
            if emotion == "other" or emotion not in self.emotion_to_idx:
                continue

            # Extract dialogue_id and speaker from utterance_id
            # Format: Ses01F_impro06_F000
            #   dialogue: Ses01F_impro06
            #   speaker: F (Female) or M (Male)
            parts = utt_id.rsplit("_", 1)
            if len(parts) != 2:
                continue

            dialogue_id = parts[0]  # e.g., "Ses01F_impro06"

            # Determine speaker
            speaker_char = parts[1][0]  # F or M
            speaker = f"Ses{session_id:02d}_{speaker_char}"

            # Get transcription text
            text = transcriptions.get(utt_id, "")

            # Build wav path
            wav_path = str(
                self.data_dir / f"Session{session_id}" / "sentences" / "wav"
                / dialogue_id / f"{utt_id}.wav"
            )

            # Create utterance
            utt = IEMOCAPUtterance(
                utterance_id=utt_id,
                dialogue_id=dialogue_id,
                text=text,
                speaker=speaker,
                emotion_raw=emotion_raw,
                emotion=emotion,
                emotion_idx=self.emotion_to_idx[emotion],
                session=session_id,
                start_time=start_time,
                end_time=end_time,
                wav_path=wav_path,
            )

            # Build dialogue
            if dialogue_id not in self._dialogues:
                self._dialogues[dialogue_id] = IEMOCAPDialogue(
                    dialogue_id=dialogue_id,
                    session=session_id,
                )
            self._dialogues[dialogue_id].add_utterance(utt)

    # ----------------------------------------------------------
    # Session-based Splits (Standard Protocol)
    # ----------------------------------------------------------

    def get_session_split(
        self, test_session: int
    ) -> Tuple[List[IEMOCAPDialogue], List[IEMOCAPDialogue]]:
        """
        Leave-one-session-out split (standard IEMOCAP evaluation).

        Args:
            test_session: Session number (1-5) to use as test set

        Returns:
            (train_dialogues, test_dialogues)
        """
        if not self._loaded:
            self.load()

        train, test = [], []
        for d in self._dialogues.values():
            if d.session == test_session:
                test.append(d)
            else:
                train.append(d)

        logger.info(
            f"Session split (test=Session{test_session}): "
            f"train={len(train)} dialogues, test={len(test)} dialogues"
        )
        return train, test

    def get_speaker_ids(self) -> Dict[str, int]:
        """Get mapping of speaker names to indices."""
        if not self._loaded:
            self.load()

        speakers = set()
        for d in self._dialogues.values():
            speakers.update(d.speakers)
        return {s: i for i, s in enumerate(sorted(speakers))}

    # ----------------------------------------------------------
    # Data Access
    # ----------------------------------------------------------

    def get_dialogues(self, session: Optional[int] = None) -> List[IEMOCAPDialogue]:
        """Get all dialogues, optionally filtered by session."""
        if not self._loaded:
            self.load()

        if session is not None:
            return [d for d in self._dialogues.values() if d.session == session]
        return list(self._dialogues.values())

    def get_flat_utterances(self, session: Optional[int] = None) -> List[IEMOCAPUtterance]:
        """Get flat list of all utterances."""
        dialogues = self.get_dialogues(session)
        return [u for d in dialogues for u in d.utterances]

    def get_emotion_weights(self) -> np.ndarray:
        """
        Compute inverse frequency class weights for handling class imbalance.

        Returns:
            np.ndarray of shape (num_classes,) with class weights
        """
        utterances = self.get_flat_utterances()
        labels = [u.emotion_idx for u in utterances if u.emotion_idx >= 0]

        counts = np.zeros(len(self.emotions))
        for label in labels:
            counts[label] += 1

        total = counts.sum()
        weights = total / (len(self.emotions) * counts + 1e-8)
        return weights / weights.sum() * len(self.emotions)

    # ----------------------------------------------------------
    # Statistics
    # ----------------------------------------------------------

    def get_stats(self, session: Optional[int] = None) -> Dict[str, Any]:
        """Get detailed statistics."""
        if not self._loaded:
            self.load()

        dialogues = self.get_dialogues(session)
        all_utt = [u for d in dialogues for u in d.utterances]

        emotion_counts = {}
        for u in all_utt:
            emotion_counts[u.emotion] = emotion_counts.get(u.emotion, 0) + 1

        session_counts = {}
        for d in dialogues:
            session_counts[d.session] = session_counts.get(d.session, 0) + 1

        speakers = set()
        for d in dialogues:
            speakers.update(d.speakers)

        lengths = [d.num_utterances for d in dialogues] if dialogues else [0]

        return {
            "num_dialogues": len(dialogues),
            "num_utterances": len(all_utt),
            "num_speakers": len(speakers),
            "num_sessions": len(session_counts),
            "avg_dialogue_length": np.mean(lengths),
            "emotion_distribution": emotion_counts,
            "session_distribution": session_counts,
        }

    def print_stats(self, session: Optional[int] = None):
        """Print formatted statistics."""
        stats = self.get_stats(session)
        title = f"IEMOCAP ({self.num_classes}-class)"
        if session:
            title += f" — Session {session}"

        print(f"\n{'='*50}")
        print(f"  {title}")
        print(f"{'='*50}")
        print(f"  Dialogues:      {stats['num_dialogues']:,}")
        print(f"  Utterances:     {stats['num_utterances']:,}")
        print(f"  Speakers:       {stats['num_speakers']}")
        print(f"  Avg Dialog Len: {stats['avg_dialogue_length']:.1f}")

        print(f"\n  Emotion Distribution:")
        print(f"  {'-'*40}")
        total = stats["num_utterances"] or 1
        for emo in self.emotions:
            count = stats["emotion_distribution"].get(emo, 0)
            pct = count / total * 100
            bar = "#" * int(pct / 2)
            print(f"    {emo:<12} {count:>5} ({pct:5.1f}%) {bar}")

        if stats["session_distribution"]:
            print(f"\n  Session Distribution:")
            print(f"  {'-'*40}")
            for sid in sorted(stats["session_distribution"].keys()):
                cnt = stats["session_distribution"][sid]
                print(f"    Session {sid}: {cnt} dialogues")

        print(f"{'='*50}\n")


# ============================================================
# Quick Test
# ============================================================

if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/IEMOCAP/IEMOCAP_full_release"

    # Test 4-class
    print("Testing 4-class setup...")
    dataset4 = IEMOCAPDataset(data_dir=data_dir, num_classes=4)
    try:
        dataset4.load()
        dataset4.print_stats()

        # Test session split
        train, test = dataset4.get_session_split(test_session=5)
        print(f"Session 5 split: train={len(train)} / test={len(test)} dialogues")

        # Test weights
        weights = dataset4.get_emotion_weights()
        for emo, w in zip(dataset4.emotions, weights):
            print(f"  {emo:<12} weight: {w:.3f}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 6-class
    print("\nTesting 6-class setup...")
    dataset6 = IEMOCAPDataset(data_dir=data_dir, num_classes=6)
    try:
        dataset6.load()
        dataset6.print_stats()
    except Exception as e:
        print(f"Error: {e}")
