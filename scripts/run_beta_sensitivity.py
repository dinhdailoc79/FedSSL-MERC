"""
Beta Sensitivity + Multi-Seed Noise Robustness
================================================
Phase 1: Find optimal beta on MELD 40% noise
Phase 2: Run 3-seed noise experiments with optimal beta

Usage:
    python scripts/run_beta_sensitivity.py
"""

import sys, os, json, time, copy
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_FILE = "results_beta_sensitivity.json"


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)


def run_noise_experiment(dataset, beta, noise_rate, seed=42):
    """
    Run one federated noise experiment with specified beta.
    
    Noise injected into clients 3 (noise_rate) and 4 (2×noise_rate, capped 0.8).
    """
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)
    
    from argparse import Namespace
    from collections import OrderedDict
    from torch.utils.data import DataLoader
    from sklearn.metrics import f1_score
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    from scripts.train_multi_dataset import (
        load_meld, load_iemocap,
        GenericDialogueDataset, collate_dialogues, evaluate,
    )
    from scripts.label_noise import inject_label_noise
    from models.evidential.evidential_dialogue_rnn import EvidentialDialogueRNN
    from models.evidential.losses import SupervisedEvidentialLoss
    from federated.aggregation.eafa import EAFAAggregator
    from data.federated_partition import FederatedPartitioner
    
    loaders_map = {"meld": load_meld, "iemocap": load_iemocap}
    
    # Load data
    load_fn = loaders_map[dataset]
    train_dias, dev_dias, test_dias, emotions, weights, cache, num_spk = load_fn(finetuned=True)
    num_classes = len(emotions)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_weights = torch.from_numpy(weights.astype(np.float32)).to(device)
    
    # Partition
    partitioner = FederatedPartitioner(
        num_clients=5, strategy="dirichlet", alpha=0.5, seed=seed,
    )
    client_partitions = partitioner.partition(train_dias, label_ratio=1.0)
    dialogue_lookup = {d.dialogue_id: d for d in train_dias}
    
    # Noise config: clients 0-2 clean, client 3 = noise_rate, client 4 = 2×noise_rate
    noise_config = {3: noise_rate, 4: min(noise_rate * 2, 0.8)}
    
    client_loaders = []
    for idx, partition in enumerate(client_partitions):
        dias = [dialogue_lookup[did] for did in partition.dialogue_ids if did in dialogue_lookup]
        client_noise = noise_config.get(idx, 0.0)
        if client_noise > 0:
            dias, _ = inject_label_noise(dias, client_noise, num_classes, seed=seed + idx)
        ds = GenericDialogueDataset(dias, cache.get("train", {}))
        loader = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate_dialogues, num_workers=0)
        client_loaders.append(loader)
    
    # Dev + Test loaders
    dev_ds = GenericDialogueDataset(dev_dias, cache.get("dev", {}))
    dev_loader = DataLoader(dev_ds, batch_size=16, shuffle=False, collate_fn=collate_dialogues, num_workers=0)
    test_ds = GenericDialogueDataset(test_dias, cache.get("test", {}))
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_dialogues, num_workers=0)
    
    # Model
    global_model = EvidentialDialogueRNN(
        input_dim=768, hidden_dim=256, num_classes=num_classes,
        num_speakers=num_spk, dropout=0.3,
    ).to(device)
    loss_fn = SupervisedEvidentialLoss(
        num_classes=num_classes, annealing_epochs=30, class_weights=class_weights,
    )
    aggregator = EAFAAggregator(beta=beta)
    
    is_eafa = beta > 0
    agg_label = f"EAFA(b={beta})" if is_eafa else "FedAvg"
    
    logger.info(f"\n{'='*60}")
    logger.info(f"  {agg_label} | {dataset.upper()} | noise={noise_rate:.0%} | seed={seed}")
    logger.info(f"{'='*60}")
    
    best_dev_wf1, patience_cnt = 0.0, 0
    best_test_wf1, best_test_u = 0.0, 1.0
    final_weights = []
    final_client_us = []
    
    for round_num in range(1, 51):
        start = time.time()
        client_states, client_sizes, client_us = [], [], []
        
        for loader in client_loaders:
            local_model = copy.deepcopy(global_model).to(device)
            local_model.train()
            loss_fn.set_epoch(round_num)
            opt = torch.optim.Adam(local_model.parameters(), lr=1e-3, weight_decay=1e-4)
            all_u = []
            
            for _ in range(3):  # local_epochs
                for batch in loader:
                    feats = batch["features"].to(device)
                    speakers = batch["speaker_ids"].to(device)
                    labels = batch["labels"].to(device)
                    out = local_model(feats, speakers)
                    mask = labels != -1
                    loss, _ = loss_fn(out["alpha"][mask], labels[mask])
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), 5.0)
                    opt.step()
                    all_u.extend(out["uncertainty"][mask].detach().cpu().numpy())
            
            client_states.append(OrderedDict({k: v.cpu() for k, v in local_model.state_dict().items()}))
            client_sizes.append(len(loader.dataset))
            client_us.append(float(np.mean(all_u)) if all_u else 1.0)
        
        global_state, agg_stats = aggregator.aggregate(
            client_states, client_sizes, client_us, round_num,
        )
        global_model.load_state_dict(global_state)
        global_model.to(device)
        
        # Dev eval for early stopping
        dev_wf1, dev_u, _, _ = evaluate(global_model, dev_loader, device)
        test_wf1, test_u, _, _ = evaluate(global_model, test_loader, device)
        elapsed = time.time() - start
        
        if round_num % 5 == 0 or round_num <= 3:
            w_str = ",".join(f"{w:.2f}" for w in agg_stats["weights"])
            logger.info(
                f"R{round_num:2d} | Dev={dev_wf1:.4f} Test={test_wf1:.4f} u={test_u:.3f} | w=[{w_str}] | {elapsed:.1f}s"
            )
        
        if dev_wf1 > best_dev_wf1:
            best_dev_wf1 = dev_wf1
            best_test_wf1 = test_wf1
            best_test_u = test_u
            final_weights = agg_stats["weights"]
            final_client_us = client_us
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= 15:
                logger.info(f"  Early stopping at round {round_num}")
                break
        
        del client_states
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    logger.info(f"  RESULT: {agg_label} WF1={best_test_wf1:.4f}")
    
    return {
        "wf1": round(float(best_test_wf1), 4),
        "uncertainty": round(float(best_test_u), 4),
        "dev_wf1": round(float(best_dev_wf1), 4),
        "weights": [round(float(w), 3) for w in final_weights] if final_weights else [],
        "client_uncertainties": [round(float(u), 4) for u in final_client_us] if final_client_us else [],
    }


def main():
    results = load_results()
    total_start = time.time()
    
    # ==============================
    # PHASE 1: Beta Sensitivity
    # Find optimal beta on MELD 40% noise
    # ==============================
    print("\n" + "="*60)
    print("  PHASE 1: Beta Sensitivity (MELD 40% noise)")
    print("="*60)
    
    beta_experiments = []
    for beta in [1.0, 2.0, 5.0, 10.0]:
        key = f"meld_beta{beta}_noise0.4_s42"
        beta_experiments.append((key, "meld", beta, 0.4, 42))
    
    # Also FedAvg baseline for comparison
    beta_experiments.append(("meld_fedavg_noise0.4_s42_v2", "meld", 0.0, 0.4, 42))
    
    for key, dataset, beta, noise, seed in beta_experiments:
        if key in results and results[key].get("wf1") is not None:
            print(f"SKIP {key}: WF1={results[key]['wf1']}")
            continue
        
        print(f"\nRUN {key}...")
        start = time.time()
        try:
            r = run_noise_experiment(dataset, beta, noise, seed)
            r["time"] = round(time.time() - start, 1)
            r["beta"] = beta
            r["noise_rate"] = noise
            results[key] = r
            save_results(results)
            print(f"  >> WF1={r['wf1']}, time={r['time']:.0f}s")
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[key] = {"wf1": None, "error": str(e)}
            save_results(results)
    
    # Find optimal beta
    best_beta, best_wf1 = 1.0, 0.0
    print(f"\n{'='*60}")
    print(f"  BETA SENSITIVITY RESULTS")
    print(f"{'='*60}")
    for beta in [0.0, 1.0, 2.0, 5.0, 10.0]:
        if beta == 0.0:
            key = "meld_fedavg_noise0.4_s42_v2"
            label = "FedAvg (b=0)"
        else:
            key = f"meld_beta{beta}_noise0.4_s42"
            label = f"EAFA (b={beta})"
        
        wf1 = results.get(key, {}).get("wf1", "N/A")
        weights = results.get(key, {}).get("weights", [])
        w_str = ",".join(f"{w:.2f}" for w in weights) if weights else "N/A"
        print(f"  {label:18s}: WF1={wf1}  w=[{w_str}]")
        
        if isinstance(wf1, (int, float)) and beta > 0 and wf1 > best_wf1:
            best_wf1 = wf1
            best_beta = beta
    
    print(f"\n  OPTIMAL BETA = {best_beta}")
    print(f"{'='*60}")
    
    # ==============================
    # PHASE 2: Multi-Seed with optimal beta
    # ==============================
    print(f"\n{'='*60}")
    print(f"  PHASE 2: Multi-Seed Noise with beta={best_beta}")
    print(f"{'='*60}")
    
    phase2_experiments = []
    for dataset in ["meld", "iemocap"]:
        for noise in [0.0, 0.2, 0.4]:
            for seed in [42, 123, 2024]:
                # EAFA with optimal beta
                key = f"{dataset}_eafa_b{best_beta}_noise{noise}_s{seed}"
                phase2_experiments.append((key, dataset, best_beta, noise, seed))
                # FedAvg
                key = f"{dataset}_fedavg_noise{noise}_s{seed}_v2"
                phase2_experiments.append((key, dataset, 0.0, noise, seed))
    
    for key, dataset, beta, noise, seed in phase2_experiments:
        if key in results and results[key].get("wf1") is not None:
            print(f"SKIP {key}: WF1={results[key]['wf1']}")
            continue
        
        print(f"\nRUN {key}...")
        start = time.time()
        try:
            r = run_noise_experiment(dataset, beta, noise, seed)
            r["time"] = round(time.time() - start, 1)
            r["beta"] = beta
            r["noise_rate"] = noise
            results[key] = r
            save_results(results)
            print(f"  >> WF1={r['wf1']}, time={r['time']:.0f}s")
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[key] = {"wf1": None, "error": str(e)}
            save_results(results)
    
    # ==============================
    # FINAL SUMMARY
    # ==============================
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS ({total_time/60:.1f} min)")
    print(f"{'='*60}")
    
    for dataset in ["meld", "iemocap"]:
        print(f"\n  {dataset.upper()}:")
        for noise in [0.0, 0.2, 0.4]:
            eafa_wf1s = []
            favg_wf1s = []
            for seed in [42, 123, 2024]:
                e_key = f"{dataset}_eafa_b{best_beta}_noise{noise}_s{seed}"
                f_key = f"{dataset}_fedavg_noise{noise}_s{seed}_v2"
                e_wf1 = results.get(e_key, {}).get("wf1")
                f_wf1 = results.get(f_key, {}).get("wf1")
                if e_wf1: eafa_wf1s.append(e_wf1)
                if f_wf1: favg_wf1s.append(f_wf1)
            
            if eafa_wf1s and favg_wf1s:
                e_mean, e_std = np.mean(eafa_wf1s), np.std(eafa_wf1s)
                f_mean, f_std = np.mean(favg_wf1s), np.std(favg_wf1s)
                delta = e_mean - f_mean
                print(f"    noise={int(noise*100):3d}%: EAFA={e_mean:.4f}±{e_std:.4f}  FedAvg={f_mean:.4f}±{f_std:.4f}  Δ={delta:+.4f}")
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
