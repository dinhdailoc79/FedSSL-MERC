"""
JSON Results Merger for Distributed Experiment Runs
=====================================================
Utility script to merge JSON result files generated from different machines
(e.g., Local MELD runs and Google Colab IEMOCAP runs) into a single consolidated file.

Usage:
    python scripts/merge_results.py --files results_local.json results_colab.json --output results_merged.json
"""

import os, json, argparse


def merge_json_files(file_paths, output_path):
    merged = {}
    
    for path in file_paths:
        if not os.path.exists(path):
            print(f"Warning: File {path} not found, skipping.")
            continue
            
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                
            # If it's a flat dictionary (like results_eafa_guard_real.json or results_systematic_noise.json)
            if isinstance(data, dict):
                for key, val in data.items():
                    # Only merge valid results (skip errored ones if a successful one is present)
                    if key in merged:
                        if merged[key].get("macro_f1") is None and val.get("macro_f1") is not None:
                            merged[key] = val
                            print(f"  Updated/Overwrote key: {key}")
                    else:
                        merged[key] = val
            print(f"Successfully read and merged: {path} ({len(data)} items)")
        except Exception as e:
            print(f"Error reading {path}: {e}")
            
    with open(output_path, 'w') as f:
        json.dump(merged, f, indent=2)
    print(f"\nConsolidated results saved to: {output_path} ({len(merged)} total items)")


def main():
    parser = argparse.ArgumentParser(description="Merge multiple experimental result JSONs.")
    parser.add_argument("--files", nargs="+", required=True, help="List of JSON files to merge")
    parser.add_argument("--output", type=str, required=True, help="Path to save merged results")
    args = parser.parse_args()
    
    merge_json_files(args.files, args.output)


if __name__ == "__main__":
    main()
