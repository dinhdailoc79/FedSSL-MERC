import json, numpy as np

with open("results_ecr_ablation.json") as f:
    data = json.load(f)

seeds = [42, 123, 2024, 7, 99]
variants = ["ecr_full", "ecr_no_certainty", "ecr_ce_pseudo", "ecr_no_augment"]

for dataset in ["meld", "iemocap"]:
    print(f"\n{'='*50}")
    print(f"{dataset.upper()} (5% labels)")
    print(f"{'='*50}")
    for v in variants:
        vals = [data[f"{dataset}_{v}_s{s}"]["wf1"] for s in seeds if f"{dataset}_{v}_s{s}" in data]
        if vals:
            m = np.mean(vals) * 100
            s = np.std(vals) * 100
            print(f"  {v:25s}: {m:.2f} +/- {s:.2f}  (n={len(vals)})")
