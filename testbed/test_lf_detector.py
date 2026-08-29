"""Quick test to verify Label-Flip Detector works in testbed."""
import numpy as np
import fedsim
from fedtrain import build_clients, fed_train, f1_macro

C, DT, DA, K = 6, 16, 16, 20
rng = np.random.default_rng(42)

# Build clients with 40% label-flip attackers
attack_clients = set(range(16, 20))  # 20% of 20 = 4 clients
clients, test, _ = build_clients(K, C, DT, DA, rng, n_per=200)

print("="*60)
print("TESTBED: Label-Flip Attack (40% contamination)")
print("="*60)

# Test 1: EAFA-Guard WITHOUT Label-Flip Detector
print("\n[1] EAFA-Guard (no LF detector):")
result1 = fed_train(clients, test, DT, DA, C, "EAFA-Guard", rng,
                    rounds=30, beta=8.0, attack="label-flip",
                    attack_clients=attack_clients, use_lf_guard=False)
f1_1 = f1_macro(result1["model"], test, C)
print(f"   F1 = {f1_1:.2f}%")

# Test 2: EAFA-Guard WITH Label-Flip Detector
print("\n[2] EAFA-Guard-LF (with LF detector):")
result2 = fed_train(clients, test, DT, DA, C, "EAFA-Guard-LF", rng,
                    rounds=30, beta=8.0, attack="label-flip",
                    attack_clients=attack_clients, use_lf_guard=True)
f1_2 = f1_macro(result2["model"], test, C)
print(f"   F1 = {f1_2:.2f}%")

# Test 3: FedAvg baseline
print("\n[3] FedAvg (baseline):")
result3 = fed_train(clients, test, DT, DA, C, "FedAvg", rng,
                    rounds=30, beta=8.0, attack="label-flip",
                    attack_clients=attack_clients)
f1_3 = f1_macro(result3["model"], test, C)
print(f"   F1 = {f1_3:.2f}%")

print("\n" + "="*60)
print("SUMMARY:")
print(f"  FedAvg:           {f1_3:.2f}%")
print(f"  EAFA-Guard:       {f1_1:.2f}%")
print(f"  EAFA-Guard-LF:    {f1_2:.2f}%")
print(f"  Improvement:      {f1_2 - f1_1:+.2f}%")
print("="*60)

if f1_2 > f1_1:
    print("\n[OK] Label-Flip Detector WORKS! EAFA-Guard-LF > EAFA-Guard")
else:
    print("\n[FAIL] No improvement (or regression)")
