"""
MELD Audio Data Preparation Pipeline
========================================
Step 1: Download MELD raw video files (mp4)
Step 2: Extract audio (wav) from each video using ffmpeg
Step 3: Organize by split (train/dev/test)

Requirements:
    - ffmpeg installed and in PATH
    - ~5GB free disk space (4GB video + 1GB audio)

Usage:
    # Full pipeline
    python scripts/prepare_audio.py

    # Download only
    python scripts/prepare_audio.py --download_only

    # Extract audio only (if videos already downloaded)
    python scripts/prepare_audio.py --extract_only
"""

import os
import sys
import subprocess
import tarfile
import logging
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# MELD Raw data URL (official)
MELD_RAW_URL = "https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz"
MELD_RAW_URL_ALT = "https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz"


def check_ffmpeg():
    """Check if ffmpeg is installed."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            version = result.stdout.split("\n")[0]
            logger.info(f"ffmpeg found: {version}")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    logger.error(
        "ffmpeg not found! Install it:\n"
        "  Windows: winget install ffmpeg\n"
        "  Or download from: https://ffmpeg.org/download.html"
    )
    return False


def download_meld_raw(data_dir: str):
    """Download MELD.Raw.tar.gz."""
    data_dir = Path(data_dir)
    tar_path = data_dir / "MELD.Raw.tar.gz"

    if tar_path.exists():
        size_gb = tar_path.stat().st_size / 1e9
        logger.info(f"MELD.Raw.tar.gz already exists ({size_gb:.2f} GB)")
        return tar_path

    data_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading MELD raw data to {tar_path}...")
    logger.info(f"URL: {MELD_RAW_URL}")
    logger.info("This is ~4GB, may take 10-30 minutes depending on connection...")

    # Try wget first (shows progress)
    try:
        subprocess.run(
            ["wget", "-O", str(tar_path), "--show-progress", MELD_RAW_URL],
            check=True,
        )
        return tar_path
    except (FileNotFoundError, subprocess.CalledProcessError):
        logger.info("wget not available, trying curl...")

    # Try curl
    try:
        subprocess.run(
            ["curl", "-L", "-o", str(tar_path), "--progress-bar", MELD_RAW_URL],
            check=True,
        )
        return tar_path
    except (FileNotFoundError, subprocess.CalledProcessError):
        logger.info("curl not available, trying Python urllib...")

    # Fallback to Python
    import urllib.request

    def _progress(count, block_size, total_size):
        pct = count * block_size * 100 / total_size
        print(f"\r  Downloading: {pct:.1f}%", end="", flush=True)

    urllib.request.urlretrieve(MELD_RAW_URL, str(tar_path), _progress)
    print()
    return tar_path


def extract_tar(tar_path: str, extract_dir: str):
    """Extract MELD.Raw.tar.gz."""
    tar_path = Path(tar_path)
    extract_dir = Path(extract_dir)

    # Check if already extracted
    expected_dir = extract_dir / "MELD.Raw"
    if expected_dir.exists() and any(expected_dir.rglob("*.mp4")):
        n_videos = len(list(expected_dir.rglob("*.mp4")))
        logger.info(f"Already extracted: {n_videos} mp4 files in {expected_dir}")
        return expected_dir

    logger.info(f"Extracting {tar_path}...")
    with tarfile.open(str(tar_path), "r:gz") as tar:
        tar.extractall(path=str(extract_dir))

    n_videos = len(list(expected_dir.rglob("*.mp4")))
    logger.info(f"Extracted {n_videos} video files")
    return expected_dir


def extract_audio_from_video(
    video_path: str,
    audio_path: str,
    sample_rate: int = 16000,
):
    """Extract audio from a single video file using ffmpeg."""
    audio_path = Path(audio_path)
    if audio_path.exists():
        return True

    audio_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-i", str(video_path),
                "-vn",                    # No video
                "-acodec", "pcm_s16le",   # 16-bit PCM
                "-ar", str(sample_rate),  # Sample rate
                "-ac", "1",               # Mono
                "-y",                     # Overwrite
                str(audio_path),
            ],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, Exception) as e:
        logger.warning(f"Failed to extract audio from {video_path}: {e}")
        return False


def extract_all_audio(raw_dir: str, audio_dir: str, sample_rate: int = 16000):
    """Extract audio from all MELD video files."""
    raw_dir = Path(raw_dir)
    audio_dir = Path(audio_dir)

    # MELD.Raw structure:
    #   MELD.Raw/train/train_splits/dia{X}_utt{Y}.mp4
    #   MELD.Raw/dev/dev_splits_complete/dia{X}_utt{Y}.mp4
    #   MELD.Raw/test/output_repeated_splits_test/dia{X}_utt{Y}.mp4
    split_dirs = {
        "train": raw_dir / "train" / "train_splits",
        "dev": raw_dir / "dev" / "dev_splits_complete",
        "test": raw_dir / "test" / "output_repeated_splits_test",
    }

    # Try alternative paths
    for split, sdir in split_dirs.items():
        if not sdir.exists():
            # Search for mp4 files under the split directory
            alt = raw_dir / split
            if alt.exists():
                for subdir in alt.rglob("*.mp4"):
                    split_dirs[split] = subdir.parent
                    break

    total_stats = {"success": 0, "failed": 0, "skipped": 0}

    for split, video_dir in split_dirs.items():
        if not video_dir.exists():
            logger.warning(f"Video directory not found for {split}: {video_dir}")
            logger.info(f"  Looking for mp4 files under {raw_dir / split}...")
            alt_dir = raw_dir / split
            if alt_dir.exists():
                mp4s = list(alt_dir.rglob("*.mp4"))
                if mp4s:
                    video_dir = mp4s[0].parent
                    logger.info(f"  Found {len(mp4s)} mp4 files in {video_dir}")
                else:
                    logger.warning(f"  No mp4 files found under {alt_dir}")
                    continue
            else:
                continue

        video_files = sorted(video_dir.glob("*.mp4"))
        if not video_files:
            logger.warning(f"No mp4 files in {video_dir}")
            continue

        out_dir = audio_dir / split
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n{split}: Extracting audio from {len(video_files)} videos...")

        success, failed, skipped = 0, 0, 0
        for i, vf in enumerate(video_files):
            audio_path = out_dir / vf.with_suffix(".wav").name

            if audio_path.exists():
                skipped += 1
                continue

            ok = extract_audio_from_video(str(vf), str(audio_path), sample_rate)
            if ok:
                success += 1
            else:
                failed += 1

            if (i + 1) % 500 == 0 or (i + 1) == len(video_files):
                logger.info(f"  {split}: {i+1}/{len(video_files)} processed "
                            f"({success} new, {skipped} skipped, {failed} failed)")

        total_stats["success"] += success
        total_stats["failed"] += failed
        total_stats["skipped"] += skipped

        logger.info(f"  {split} done: {success} extracted, {skipped} skipped, {failed} failed")

    return total_stats


def verify_audio(audio_dir: str):
    """Verify extracted audio files and print summary."""
    audio_dir = Path(audio_dir)

    logger.info(f"\n{'='*60}")
    logger.info(f"  Audio Data Verification")
    logger.info(f"{'='*60}")

    for split in ["train", "dev", "test"]:
        split_dir = audio_dir / split
        if not split_dir.exists():
            logger.warning(f"  {split}: NOT FOUND")
            continue

        wav_files = list(split_dir.glob("*.wav"))
        total_size = sum(f.stat().st_size for f in wav_files)

        logger.info(f"  {split}: {len(wav_files)} files, {total_size/1e6:.1f} MB")

        # Sample check: verify a few files
        if wav_files:
            sample = wav_files[0]
            logger.info(f"    Sample: {sample.name} ({sample.stat().st_size/1e3:.1f} KB)")

    logger.info(f"{'='*60}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prepare MELD audio data")
    parser.add_argument("--data_dir", type=str, default="data/raw/MELD")
    parser.add_argument("--audio_dir", type=str, default="data/raw/MELD/audio")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--download_only", action="store_true")
    parser.add_argument("--extract_only", action="store_true")
    args = parser.parse_args()

    logger.info(f"\n{'='*60}")
    logger.info(f"  MELD Audio Data Preparation")
    logger.info(f"{'='*60}")

    # Step 0: Check ffmpeg
    if not args.download_only:
        if not check_ffmpeg():
            logger.error("Please install ffmpeg first!")
            sys.exit(1)

    # Step 1: Download
    if not args.extract_only:
        tar_path = download_meld_raw(args.data_dir)
    else:
        tar_path = Path(args.data_dir) / "MELD.Raw.tar.gz"

    # Step 2: Extract tar
    if not args.extract_only:
        raw_dir = extract_tar(str(tar_path), args.data_dir)
    else:
        raw_dir = Path(args.data_dir) / "MELD.Raw"
        if not raw_dir.exists():
            logger.error(f"Raw directory not found: {raw_dir}")
            sys.exit(1)

    if args.download_only:
        logger.info("Download complete. Run with --extract_only to extract audio.")
        return

    # Step 3: Extract audio
    stats = extract_all_audio(str(raw_dir), args.audio_dir, args.sample_rate)
    logger.info(f"\nAudio extraction: {stats['success']} new, "
                f"{stats['skipped']} skipped, {stats['failed']} failed")

    # Step 4: Verify
    verify_audio(args.audio_dir)

    logger.info("\nDone! Next step: extract WavLM features.")
    logger.info("  python scripts/extract_audio_features.py")


if __name__ == "__main__":
    main()
