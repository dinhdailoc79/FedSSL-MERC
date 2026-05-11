"""
Download MELD Dataset
======================
Downloads the MELD dataset (CSV annotations + optional raw video files).

Usage:
    python scripts/download_meld.py --output data/raw/MELD
    python scripts/download_meld.py --output data/raw/MELD --include-raw
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def download_file(url: str, output_path: str):
    """Download a file using requests (avoids ssl module conflict)."""
    import requests
    print(f"  Downloading: {Path(output_path).name} ...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(resp.content)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Download MELD Dataset")
    parser.add_argument("--output", type=str, default="data/raw/MELD")
    parser.add_argument("--include-raw", action="store_true",
                        help="Also download raw video files (~4GB)")
    parser.add_argument("--verify", action="store_true",
                        help="Verify by loading and printing stats")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading MELD dataset to: {args.output}")
    print(f"Include raw videos: {args.include_raw}\n")

    # Download CSV annotations
    csv_base = "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD"
    for split in ["train", "dev", "test"]:
        csv_name = f"{split}_sent_emo.csv"
        csv_path = output_path / csv_name

        if csv_path.exists():
            print(f"  Already exists: {csv_name}")
            continue

        url = f"{csv_base}/{csv_name}"
        try:
            download_file(url, str(csv_path))
        except Exception as e:
            print(f"  FAILED {csv_name}: {e}")
            print(f"  Manual download: https://github.com/declare-lab/MELD")

    if args.include_raw:
        print("\nRaw video download not implemented in this script.")
        print("Download manually: https://huggingface.co/datasets/declare-lab/MELD")

    print(f"\nMELD dataset ready at: {output_path}")

    if args.verify:
        print("\nVerifying download...\n")
        from data.datasets.meld import MELDDataset
        dataset = MELDDataset(data_dir=args.output)
        dataset.load_all()
        for split in ["train", "dev", "test"]:
            dataset.print_stats(split)


if __name__ == "__main__":
    main()
