"""Controlled test: same clients, different attacks."""
import numpy as np
from fedtrain import build_clients, fed_train, f1_macro

C, DT, DA, K = 6, 16, 16, 20

# Use same seed for clients, only vary attack
rng = np.random.default_rng(42)
clients, test, _ = build_clients(K, C, DT, DA, rng, n_per=200)

attack_clients = set(range(16, 20))

print("="*70)
print("CONTROLLED TEST: Same clients, different attacks")
print("="*70)

results = {}

for attack in ["label-flip", "sign-flip", "adaptive"]:
    print(f"\n--- {attack.upper()} ---")

    # EAFA-Guard (no LF detector)
    r = fed_train(clients, test, DT, DA, C, "EAFA-Guard", rng,
                  rounds=25, beta=8.0, attack=attack,
                  attack_clients=attack_clients, use_lf_guard=False)
    f1_guard = f1_macro(r["model"], test, C)
    print(f"  EAFA-Guard:     {f1_guard:.2f}%")

    # EAFA-Guard-LF (with LF detector)
    r = fed_train(clients, test, DT, DA, C, "EAFA-Guard-LF", rng,
                  rounds=25, beta=8.0, attack=attack,
                  attack_clients=attack_clients, use_lf_guard=True)
    f1_guard_lf = f1_macro(r["model"], test, C)
    print(f"  EAFA-Guard-LF:  {f1_guard_lf:.2f}%")

    results[attack] = {
        "EAFA-Guard": f1_guard,
        "EAFA-Guard-LF": f1_guard_lf,
        "diff": f1_guard_lf - f1_guard
    }

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
for attack, scores in results.items():
    diff = scores["diff"]
    symbol = "+" if diff > 0 else ""
    print(f"  {attack:<12} Guard: {scores['EAFA-Guard']:>6.2f}%  |  Guard-LF: {scores['EAFA-Guard-LF']:>6.2f}%  |  Diff: {symbol}{diff:.2f}%")
print("="*70)

# Check for regressions
regressions = [a for a, s in results.items() if s["diff"] < 0]
if regressions:
    print(f"\n[WARNING] Regressions on: {regressions}")
else:
    print("\n[OK] No regressions on any attack type!")
