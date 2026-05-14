"""
WavLM Audio Feature Extraction for MELD
==========================================
Extract WavLM-base features from MELD audio files.

WavLM: https://arxiv.org/abs/2110.13900
- Pre-trained self-supervised speech model
- Output: 768-dim per-frame features
- We pool to get 1 embedding per utterance

Requirements:
    pip install transformers torchaudio soundfile

Usage:
    # Local (if GPU has enough VRAM, ~3GB needed)
    python scripts/extract_audio_features.py

    # On Kaggle (recommended for T4 GPU)
    python scripts/extract_audio_features.py --device cuda
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_audio(audio_path: str, target_sr: int = 16000):
    """Load audio file and resample if needed."""
    waveform, sr = torchaudio.load(audio_path)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)

    return waveform.squeeze(0)  # (num_samples,)


def parse_filename(filename: str):
    """
    Parse MELD filename: dia{X}_utt{Y}.wav → (dialogue_id, utterance_id)
    """
    name = Path(filename).stem  # dia0_utt0
    parts = name.replace("dia", "").replace("utt", "").split("_")
    if len(parts) == 2:
        return int(parts[0]), int(parts[1])
    return None, None


def extract_wavlm_features(
    audio_dir: str,
    output_path: str,
    model_name: str = "microsoft/wavlm-base",
    device: str = "cuda",
    batch_size: int = 8,
    max_length_sec: float = 15.0,
):
    """
    Extract WavLM features for all MELD audio files.

    Args:
        audio_dir: Directory with train/dev/test subdirs containing wav files
        output_path: Path to save features (.pt file)
        model_name: HuggingFace model name
        device: 'cuda' or 'cpu'
        batch_size: Batch size for inference
        max_length_sec: Max audio length (truncate longer)
    """
    from transformers import WavLMModel, AutoFeatureExtractor

    audio_dir = Path(audio_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info(f"Loading {model_name}...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = WavLMModel.from_pretrained(model_name).to(device)
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    logger.info(f"WavLM loaded: {params/1e6:.1f}M params on {device}")

    results = {}
    target_sr = feature_extractor.sampling_rate  # 16000

    for split in ["train", "dev", "test"]:
        split_dir = audio_dir / split
        if not split_dir.exists():
            logger.warning(f"Split dir not found: {split_dir}")
            continue

        wav_files = sorted(split_dir.glob("*.wav"))
        if not wav_files:
            logger.warning(f"No wav files in {split_dir}")
            continue

        logger.info(f"\n{split}: Processing {len(wav_files)} audio files...")

        all_features = []
        all_dia_ids = []
        all_utt_ids = []
        failed = 0

        for i in tqdm(range(0, len(wav_files), batch_size), desc=f"  {split}"):
            batch_files = wav_files[i:i + batch_size]
            batch_waveforms = []
            batch_dia = []
            batch_utt = []

            for wf in batch_files:
                try:
                    waveform = load_audio(str(wf), target_sr)

                    # Truncate
                    max_len = int(max_length_sec * target_sr)
                    if waveform.shape[0] > max_len:
                        waveform = waveform[:max_len]

                    batch_waveforms.append(waveform.numpy())

                    dia_id, utt_id = parse_filename(wf.name)
                    batch_dia.append(dia_id)
                    batch_utt.append(utt_id)
                except Exception as e:
                    logger.warning(f"Failed to load {wf.name}: {e}")
                    failed += 1
                    continue

            if not batch_waveforms:
                continue

            # Process batch
            inputs = feature_extractor(
                batch_waveforms,
                sampling_rate=target_sr,
                return_tensors="pt",
                padding=True,
                max_length=int(max_length_sec * target_sr),
                truncation=True,
            )

            with torch.no_grad():
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                # Mean pool over time dimension → (batch, 768)
                hidden_states = outputs.last_hidden_state
                # Use attention mask to exclude padding
                if "attention_mask" in inputs:
                    mask = inputs["attention_mask"].unsqueeze(-1)
                    pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                else:
                    pooled = hidden_states.mean(dim=1)

            all_features.append(pooled.cpu())
            all_dia_ids.extend(batch_dia)
            all_utt_ids.extend(batch_utt)

        if all_features:
            features = torch.cat(all_features, dim=0)
            results[split] = {
                "features": features,
                "dialogue_ids": torch.tensor(all_dia_ids),
                "utterance_ids": torch.tensor(all_utt_ids),
            }
            logger.info(f"  {split}: {features.shape[0]} utterances, "
                        f"{features.shape[1]}-dim, {failed} failed")
        else:
            logger.warning(f"  {split}: No features extracted!")

    # Save
    torch.save(results, str(output_path))
    size_mb = output_path.stat().st_size / 1e6
    logger.info(f"\nSaved to {output_path} ({size_mb:.1f} MB)")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract WavLM audio features")
    parser.add_argument("--audio_dir", type=str, default="data/raw/MELD/audio")
    parser.add_argument("--output", type=str, default="data/features/meld_audio_wavlm.pt")
    parser.add_argument("--model", type=str, default="microsoft/wavlm-base")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=float, default=15.0)
    args = parser.parse_args()

    logger.info(f"\n{'='*60}")
    logger.info(f"  WavLM Audio Feature Extraction")
    logger.info(f"{'='*60}")

    results = extract_wavlm_features(
        args.audio_dir, args.output,
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        max_length_sec=args.max_length,
    )

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"  Summary")
    logger.info(f"{'='*60}")
    for split, data in results.items():
        logger.info(f"  {split}: {data['features'].shape}")
    logger.info(f"{'='*60}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
