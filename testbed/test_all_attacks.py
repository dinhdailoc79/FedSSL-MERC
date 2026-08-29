"""Test all 3 attack types to verify no regression."""
import numpy as np
from fedtrain import build_clients, fed_train, f1_macro

C, DT, DA, K = 6, 16, 16, 20
rng = np.random.default_rng(42)

# 20% attackers = 4 out of 20 clients
attack_clients = set(range(16, 20))

print("="*70)
print("TESTBED: All 3 Attack Types (20% contamination)")
print("="*70)

results = {}

for attack in ["label-flip", "sign-flip", "adaptive"]:
    print(f"\n{'='*70}")
    print(f"ATTACK: {attack.upper()}")
    print(f"{'='*70}")

    # Rebuild clients for each attack
    clients, test, _ = build_clients(K, C, DT, DA, rng, n_per=200)

    # FedAvg
    r = fed_train(clients, test, DT, DA, C, "FedAvg", rng,
                  rounds=30, beta=8.0, attack=attack,
                  attack_clients=attack_clients)
    f1_fedavg = f1_macro(r["model"], test, C)
    print(f"  FedAvg:       {f1_fedavg:.2f}%")

    # EAFA-Guard (no LF detector)
    r = fed_train(clients, test, DT, DA, C, "EAFA-Guard", rng,
                  rounds=30, beta=8.0, attack=attack,
                  attack_clients=attack_clients, use_lf_guard=False)
    f1_guard = f1_macro(r["model"], test, C)
    print(f"  EAFA-Guard:   {f1_guard:.2f}%")

    # EAFA-Guard-LF (with LF detector)
    r = fed_train(clients, test, DT, DA, C, "EAFA-Guard-LF", rng,
                  rounds=30, beta=8.0, attack=attack,
                  attack_clients=attack_clients, use_lf_guard=True)
    f1_guard_lf = f1_macro(r["model"], test, C)
    print(f"  EAFA-Guard-LF: {f1_guard_lf:.2f}%")

    results[attack] = {
        "FedAvg": f1_fedavg,
        "EAFA-Guard": f1_guard,
        "EAFA-Guard-LF": f1_guard_lf,
    }

print("\n" + "="*70)
print("SUMMARY TABLE")
print("="*70)
print(f"{'Attack':<15} {'FedAvg':>10} {'EAFA-Guard':>12} {'EAFA-Guard-LF':>14}")
print("-"*55)
for attack, scores in results.items():
    print(f"{attack:<15} {scores['FedAvg']:>10.2f} {scores['EAFA-Guard']:>12.2f} {scores['EAFA-Guard-LF']:>14.2f}")
print("="*70)

# Check if EAFA-Guard-LF wins on all 3
print("\nVERIFICATION:")
all_wins = True
for attack in results:
    guard = results[attack]['EAFA-Guard']
    guard_lf = results[attack]['EAFA-Guard-LF']
    win = guard_lf >= guard
    status = "[OK]" if win else "[FAIL]"
    print(f"  {attack}: EAFA-Guard-LF ({guard_lf:.2f}) vs EAFA-Guard ({guard:.2f}) {status}")
    if not win:
        all_wins = False

if all_wins:
    print("\n[SUCCESS] EAFA-Guard-LF >= EAFA-Guard on all 3 attacks!")
else:
    print("\n[WARNING] Some regressions detected")
