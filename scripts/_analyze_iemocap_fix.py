import json, numpy as np

r = json.load(open('results_fix_iemocap50.json'))
seeds = [42, 123, 2024]
lus = [0.3, 0.5, 1.0, 2.0, 3.0, 5.0]
rps = [5, 10, 20]

FM_BASELINE = 0.5965

print(f"FixMatch baseline: {FM_BASELINE}")
print(f"\n  {'lu':>4}  {'rp=5':>8}  {'rp=10':>8}  {'rp=20':>8}")
print(f"  {'----':>4}  {'--------':>8}  {'--------':>8}  {'--------':>8}")

best_mean = 0
best_cfg = None

for lu in lus:
    row = f"  {lu:4.1f}"
    for rp in rps:
        vals = [r[f"iemocap_ecr_lu{lu}_rp{rp}_s{s}"]["wf1"] for s in seeds]
        mean = np.mean(vals)
        marker = " *" if mean > FM_BASELINE else ""
        row += f"  {mean:.4f}{marker:>2}"
        if mean > best_mean:
            best_mean = mean
            best_cfg = (lu, rp)
    print(row)

print(f"\nBest config: lu={best_cfg[0]}, rp={best_cfg[1]} -> mean={best_mean:.4f}")
beats = "YES" if best_mean > FM_BASELINE else "NO"
print(f"Beats FixMatch ({FM_BASELINE})? {beats} (delta={best_mean - FM_BASELINE:+.4f})")
