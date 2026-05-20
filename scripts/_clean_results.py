import json

with open("results_significance.json", "r") as f:
    r = json.load(f)

# Remove null results
cleaned = {k: v for k, v in r.items() if v.get("wf1") is not None}
with open("results_significance.json", "w") as f:
    json.dump(cleaned, f, indent=2)

print(f"Kept {len(cleaned)} results, removed {len(r)-len(cleaned)} null entries")
for k, v in cleaned.items():
    wf1 = v["wf1"]
    print(f"  {k}: {wf1}")
