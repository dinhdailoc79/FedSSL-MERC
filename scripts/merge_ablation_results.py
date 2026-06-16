"""
Merge Ablation Results
======================
Merges results_edl_vs_confidence_ablation_part2.json into results_edl_vs_confidence_ablation.json,
overwriting the placeholders with the actual results.
"""

import json
import os

def merge():
    f1 = "results_edl_vs_confidence_ablation.json"
    f2 = "results_edl_vs_confidence_ablation_part2.json"
    
    if not os.path.exists(f1):
        print(f"Error: {f1} does not exist.")
        return
        
    if not os.path.exists(f2):
        print(f"Error: {f2} does not exist.")
        return
        
    data1 = json.load(open(f1))
    data2 = json.load(open(f2))
    
    merged_count = 0
    for key, val in data2.items():
        if val.get("wf1") is not None and val.get("wf1") > 0.0:
            data1[key] = val
            merged_count += 1
            
    with open(f1, 'w') as f:
        json.dump(data1, f, indent=2)
        
    print(f"Merged {merged_count} results from {f2} to {f1} successfully!")

if __name__ == "__main__":
    merge()
