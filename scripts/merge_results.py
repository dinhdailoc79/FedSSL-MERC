"""
Merge IEMOCAP and MELD baseline results to resolve race condition.
"""

import json
import os

RESULTS_FILE = "results/fl_baselines_results.json"

# Manually recovered IEMOCAP results from console output
iemocap_results = {
  "iemocap_scaffold_s456": {
    "wf1": 0.5423,
    "method": "scaffold",
    "dataset": "iemocap",
    "seed": 456,
    "time": 64.8
  },
  "iemocap_scaffold_s789": {
    "wf1": 0.4199,
    "method": "scaffold",
    "dataset": "iemocap",
    "seed": 789,
    "time": 71.9
  },
  "iemocap_fednova_s456": {
    "wf1": 0.5971,
    "method": "fednova",
    "dataset": "iemocap",
    "seed": 456,
    "time": 138.6
  },
  "iemocap_fednova_s789": {
    "wf1": 0.6024,
    "method": "fednova",
    "dataset": "iemocap",
    "seed": 789,
    "time": 127.3
  },
  "iemocap_fedadam_s456": {
    "wf1": 0.6049,
    "method": "fedadam",
    "dataset": "iemocap",
    "seed": 456,
    "time": 111.8
  },
  "iemocap_fedadam_s789": {
    "wf1": 0.5761,
    "method": "fedadam",
    "dataset": "iemocap",
    "seed": 789,
    "time": 228.0
  },
  "iemocap_moon_s456": {
    "wf1": 0.5965,
    "method": "moon",
    "dataset": "iemocap",
    "seed": 456,
    "time": 220.6
  },
  "iemocap_moon_s789": {
    "wf1": 0.6012,
    "method": "moon",
    "dataset": "iemocap",
    "seed": 789,
    "time": 278.5
  }
}

def main():
    if not os.path.exists(RESULTS_FILE):
        print(f"Error: {RESULTS_FILE} not found!")
        return
        
    with open(RESULTS_FILE, "r") as f:
        data = json.load(f)
        
    print(f"Loaded {len(data)} existing keys from {RESULTS_FILE}")
    
    added = 0
    for k, v in iemocap_results.items():
        if k not in data or data[k].get("wf1") is None:
            data[k] = v
            added += 1
            print(f"Adding/Updating: {k} -> WF1={v['wf1']}")
            
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2)
        
    print(f"Successfully merged. Total keys in {RESULTS_FILE}: {len(data)} (added {added})")

if __name__ == "__main__":
    main()
