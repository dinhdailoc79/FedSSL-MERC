"""
Multimodal Preprocessing Pipeline
===================================
Feature extraction for text, audio, and video modalities.

Supports both lightweight (CPU/T4-friendly) and heavy (A100) modes:
  - Lightweight: Pre-extracted features loaded from disk
  - Heavy: On-the-fly extraction using pre-trained models
"""

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class MultimodalPreprocessor:
    """
    Handles feature extraction and preprocessing for all modalities.

    For the MELD dataset, features can be extracted from:
    - Text: BERT/RoBERTa embeddings
    - Audio: wav2vec 2.0 / OpenSMILE features
    - Video: ResNet / OpenFace facial features

    Usage:
        >>> preprocessor = MultimodalPreprocessor(
        ...     text_model="roberta-base",
        ...     audio_model="wav2vec2",
        ...     device="cuda"
        ... )
        >>> features = preprocessor.extract_text_features(["Hello!", "How are you?"])
    """

    def __init__(
        self,
        text_model: str = "roberta-base",
        audio_model: str = "wav2vec2",
        video_model: str = "resnet50",
        device: str = "cpu",
        cache_dir: Optional[str] = None,
    ):
        self.text_model_name = text_model
        self.audio_model_name = audio_model
        self.video_model_name = video_model
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Models loaded lazily on first use
        self._text_model = None
        self._text_tokenizer = None
        self._audio_model = None
        self._video_model = None

        logger.info(
            f"MultimodalPreprocessor initialized: "
            f"text={text_model}, audio={audio_model}, video={video_model}, device={device}"
        )

    # ----------------------------------------------------------
    # Text Features
    # ----------------------------------------------------------

    def _load_text_model(self):
        """Lazy-load text encoder (BERT/RoBERTa)."""
        if self._text_model is not None:
            return

        try:
            from transformers import AutoModel, AutoTokenizer
            import torch

            logger.info(f"Loading text model: {self.text_model_name}")
            self._text_tokenizer = AutoTokenizer.from_pretrained(
                self.text_model_name,
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
            )
            self._text_model = AutoModel.from_pretrained(
                self.text_model_name,
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
            ).to(self.device)
            self._text_model.eval()
            logger.info(f"Text model loaded on {self.device}")
        except ImportError:
            raise ImportError("Install transformers: pip install transformers")

    def extract_text_features(
        self,
        texts: List[str],
        max_length: int = 128,
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Extract text features using pre-trained language model.

        Args:
            texts: List of utterance texts
            max_length: Max token length
            batch_size: Processing batch size (reduce for T4, increase for A100)

        Returns:
            np.ndarray of shape (num_texts, hidden_dim)
        """
        import torch

        self._load_text_model()

        all_features = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            encoding = self._text_tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self._text_model(**encoding)
                # Use CLS token representation
                cls_features = outputs.last_hidden_state[:, 0, :]
                all_features.append(cls_features.cpu().numpy())

        return np.concatenate(all_features, axis=0)

    # ----------------------------------------------------------
    # Audio Features
    # ----------------------------------------------------------

    def extract_audio_features_from_video(
        self,
        video_path: str,
    ) -> Optional[np.ndarray]:
        """
        Extract audio track from video and compute audio features.

        Args:
            video_path: Path to .mp4 video file

        Returns:
            np.ndarray of audio features, or None if extraction fails
        """
        if not Path(video_path).exists():
            logger.warning(f"Video file not found: {video_path}")
            return None

        try:
            import librosa
            import subprocess
            import tempfile

            # Extract audio from video using ffmpeg
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            subprocess.run(
                ["ffmpeg", "-i", video_path, "-vn", "-ar", "16000", "-ac", "1", tmp_path, "-y"],
                capture_output=True,
                check=True,
            )

            # Load audio
            waveform, sr = librosa.load(tmp_path, sr=16000)

            # Extract basic features (MFCC + prosody)
            mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=40)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)

            # Pitch and energy
            pitches, magnitudes = librosa.piptrack(y=waveform, sr=sr)
            pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
            energy = np.mean(librosa.feature.rms(y=waveform))

            features = np.concatenate([
                mfcc_mean,
                mfcc_std,
                [pitch_mean, energy],
            ])

            # Cleanup
            Path(tmp_path).unlink(missing_ok=True)

            return features

        except Exception as e:
            logger.warning(f"Audio extraction failed for {video_path}: {e}")
            return None

    # ----------------------------------------------------------
    # Video Features (placeholder for visual/facial features)
    # ----------------------------------------------------------

    def extract_video_features(
        self,
        video_path: str,
        num_frames: int = 8,
    ) -> Optional[np.ndarray]:
        """
        Extract visual features from video frames.

        Args:
            video_path: Path to .mp4 video file
            num_frames: Number of frames to sample

        Returns:
            np.ndarray of visual features, or None if extraction fails
        """
        if not Path(video_path).exists():
            logger.warning(f"Video file not found: {video_path}")
            return None

        try:
            import cv2
            import torch
            from torchvision import transforms, models

            # Read video frames
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames == 0:
                return None

            # Sample frames uniformly
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            frames = []

            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
            cap.release()

            if not frames:
                return None

            # Preprocess frames for ResNet
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            frame_tensors = torch.stack([transform(f) for f in frames])

            # Extract features using ResNet (lazy load)
            if self._video_model is None:
                logger.info("Loading ResNet50 for video feature extraction")
                self._video_model = models.resnet50(pretrained=True)
                # Remove classification head, keep feature extractor
                self._video_model = torch.nn.Sequential(
                    *list(self._video_model.children())[:-1]
                )
                self._video_model.to(self.device)
                self._video_model.eval()

            with torch.no_grad():
                frame_tensors = frame_tensors.to(self.device)
                features = self._video_model(frame_tensors)
                features = features.squeeze(-1).squeeze(-1)
                # Average pool across frames
                pooled = features.mean(dim=0)

            return pooled.cpu().numpy()

        except Exception as e:
            logger.warning(f"Video feature extraction failed for {video_path}: {e}")
            return None

    # ----------------------------------------------------------
    # Batch Processing
    # ----------------------------------------------------------

    def preprocess_dialogue(self, dialogue, modalities: List[str] = None):
        """
        Extract features for all utterances in a dialogue.

        Args:
            dialogue: Dialogue object from MELDDataset
            modalities: Which modalities to extract. Defaults to ["text"]
        """
        modalities = modalities or ["text"]

        if "text" in modalities:
            texts = [u.text for u in dialogue.utterances]
            text_features = self.extract_text_features(texts)
            for i, utt in enumerate(dialogue.utterances):
                utt.text_features = text_features[i]

        if "audio" in modalities:
            for utt in dialogue.utterances:
                if utt.video_path:
                    utt.audio_features = self.extract_audio_features_from_video(
                        utt.video_path
                    )

        if "video" in modalities:
            for utt in dialogue.utterances:
                if utt.video_path:
                    utt.video_features = self.extract_video_features(
                        utt.video_path
                    )
