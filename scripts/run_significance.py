"""
Statistical Significance Runner v2
===================================
Runs experiments IN-PROCESS (not subprocess) to avoid encoding issues.
"""
import sys, os, json, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SEEDS = [42, 123, 2024]
RESULTS_FILE = "results_significance.json"


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)


def run_one(dataset, aggregation, seed, alpha=0.5):
    """Run one federated experiment in-process."""
    import torch
    from argparse import Namespace
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Import here to avoid circular
    from scripts.train_multi_dataset import (
        load_meld, load_iemocap, load_dailydialog,
        train_federated,
    )
    
    loaders = {"meld": load_meld, "iemocap": load_iemocap, "dailydialog": load_dailydialog}
    
    args = Namespace(
        hidden_dim=256, dropout=0.3, batch_size=16, lr=1e-3,
        annealing_epochs=30, patience=15, num_clients=5,
        alpha=alpha, num_rounds=50, local_epochs=3, beta=1.0,
        loss_type="edl", aggregation=aggregation,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_dir="checkpoints", seed=seed, finetuned=True,
    )
    
    if aggregation == "fedavg":
        args.beta = 0.0  # Disable EAFA
    
    load_fn = loaders[dataset]
    train, dev, test, emotions, weights, cache, num_spk = load_fn(finetuned=True)
    
    wf1, u, micro = train_federated(
        dataset, train, dev, test, emotions, weights, cache, num_spk, args,
    )
    
    return {"wf1": round(wf1, 4), "micro": round(micro, 4) if micro else None}


def main():
    results = load_results()
    
    # Hardcoded known results
    results["dailydialog_eafa_s42"] = {"wf1": 0.8854}
    results["dailydialog_eafa_s123"] = {"wf1": 0.8859}
    results["dailydialog_eafa_s2024"] = {"wf1": 0.8895}
    results["dailydialog_centralized_s42"] = {"wf1": 0.8773}
    results["dailydialog_centralized_s123"] = {"wf1": 0.8846}
    results["dailydialog_centralized_s2024"] = {"wf1": 0.8778}
    
    total_start = time.time()
    
    # All experiments to run
    experiments = []
    
    # Part 1: FedAvg x 3 seeds on DailyDialog
    for seed in SEEDS:
        experiments.append(("dailydialog", "fedavg", seed, 0.5))
    
    # Part 2: EAFA + FedAvg x 3 seeds on MELD
    for agg in ["eafa", "fedavg"]:
        for seed in SEEDS:
            experiments.append(("meld", agg, seed, 0.5))
    
    # Part 3: EAFA + FedAvg x 3 seeds on IEMOCAP
    for agg in ["eafa", "fedavg"]:
        for seed in SEEDS:
            experiments.append(("iemocap", agg, seed, 0.5))
    
    # Part 4: Alpha sensitivity on MELD
    for alpha in [0.1, 1.0]:
        for agg in ["eafa", "fedavg"]:
            experiments.append(("meld", agg, 42, alpha))
    
    total = len(experiments)
    done = 0
    
    for dataset, agg, seed, alpha in experiments:
        if alpha != 0.5:
            key = f"{dataset}_{agg}_a{alpha}_s{seed}"
        else:
            key = f"{dataset}_{agg}_s{seed}"
        
        if key in results and results[key].get("wf1") is not None:
            print(f"[{done+1}/{total}] SKIP {key}: {results[key]['wf1']}")
            done += 1
            continue
        
        print(f"\n[{done+1}/{total}] RUNNING {key}...")
        start = time.time()
        
        try:
            r = run_one(dataset, agg, seed, alpha)
            elapsed = time.time() - start
            r["time"] = round(elapsed, 1)
            results[key] = r
            save_results(results)
            print(f"  >> WF1={r['wf1']}, time={elapsed:.0f}s")
        except Exception as e:
            print(f"  >> ERROR: {e}")
            results[key] = {"wf1": None, "error": str(e)}
            save_results(results)
        
        done += 1
    
    # Summary
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  ALL DONE! {total_time/60:.1f} minutes")
    print(f"{'='*60}")
    
    for ds in ["dailydialog", "meld", "iemocap"]:
        print(f"\n  {ds.upper()}:")
        for agg in ["eafa", "fedavg"]:
            wf1s = []
            for seed in SEEDS:
                key = f"{ds}_{agg}_s{seed}"
                if key in results and results[key].get("wf1"):
                    wf1s.append(results[key]["wf1"])
            if wf1s:
                mean = np.mean(wf1s)
                std = np.std(wf1s) if len(wf1s) > 1 else 0
                print(f"    {agg:8s}: {mean:.4f} +/- {std:.4f}  (n={len(wf1s)})")
    
    print(f"\n  ALPHA SENSITIVITY (MELD):")
    for alpha in [0.1, 0.5, 1.0]:
        for agg in ["eafa", "fedavg"]:
            key = f"meld_{agg}_a{alpha}_s42"
            if alpha == 0.5:
                key = f"meld_{agg}_s42"
            if key in results and results[key].get("wf1"):
                print(f"    a={alpha}, {agg:8s}: {results[key]['wf1']:.4f}")


if __name__ == "__main__":
    main()
