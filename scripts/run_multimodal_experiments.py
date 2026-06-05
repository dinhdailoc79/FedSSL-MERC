"""
Multimodal Experiments Runner
==============================
Compare text-only vs text+audio with EAFA and FedAvg.

Usage:
    python scripts/run_multimodal_experiments.py
"""

import sys, os, json, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_FILE = "results_multimodal.json"


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))


def run_multimodal_experiment(beta, fusion_mode="evidence_sum", seed=42):
    """
    Run federated multimodal (text+audio) experiment on MELD.
    
    Args:
        beta: EAFA beta (1.0=EAFA, 0.0=FedAvg)
        fusion_mode: Fusion mechanism mode
        seed: random seed
    """
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)
    
    from argparse import Namespace
    from collections import OrderedDict
    from pathlib import Path
    from torch.utils.data import DataLoader
    from sklearn.metrics import f1_score, classification_report
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    from data.datasets.meld import MELDDataset, MELD_EMOTIONS
    from data.federated_partition import FederatedPartitioner
    from models.evidential.multimodal_edl import MultimodalEvidentialDialogueRNN
    from models.evidential.losses import SupervisedEvidentialLoss
    from federated.aggregation.eafa import EAFAAggregator
    from scripts.train_fed_multimodal import (
        MultimodalDialogueDataset, collate_multimodal,
        local_train_multimodal, evaluate_multimodal, load_feature_cache,
    )
    
    args = Namespace(
        hidden_dim=256, dropout=0.3, batch_size=8, lr=1e-3,
        annealing_epochs=30, patience=15, num_clients=5,
        alpha=0.5, num_rounds=50, local_epochs=3,
        beta=beta, lambda_aux=0.3, fusion_mode=fusion_mode,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_dir="checkpoints", seed=seed,
    )
    
    agg_label = f"{'EAFA' if beta > 0 else 'FedAvg'}_{fusion_mode}"
    
    # Load data
    meld = MELDDataset(data_dir="data/raw/MELD")
    train_dias = meld.get_dialogues("train")
    dev_dias = meld.get_dialogues("dev")
    test_dias = meld.get_dialogues("test")
    
    text_caches = load_feature_cache("data/features/meld_text_roberta_finetuned.pt", ["train", "dev", "test"])
    audio_caches = load_feature_cache("data/features/meld_audio_wavlm.pt", ["train", "dev", "test"])
    
    # Partition
    partitioner = FederatedPartitioner(
        num_clients=5, strategy="dirichlet", alpha=args.alpha, seed=seed,
    )
    client_partitions = partitioner.partition(train_dias, label_ratio=1.0)
    dialogue_lookup = {d.dialogue_id: d for d in train_dias}
    
    client_loaders = []
    for partition in client_partitions:
        dias = [dialogue_lookup[did] for did in partition.dialogue_ids if did in dialogue_lookup]
        ds = MultimodalDialogueDataset(dias, text_caches.get("train", {}), audio_caches.get("train", {}), 768, 768)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_multimodal, num_workers=0)
        client_loaders.append(loader)
    
    # Dev loader for early stopping
    dev_ds = MultimodalDialogueDataset(dev_dias, text_caches.get("dev", {}), audio_caches.get("dev", {}), 768, 768)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_multimodal, num_workers=0)
    
    test_ds = MultimodalDialogueDataset(test_dias, text_caches.get("test", {}), audio_caches.get("test", {}), 768, 768)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_multimodal, num_workers=0)
    
    # Model
    class_weights = torch.from_numpy(meld.get_emotion_weights("train").astype(np.float32)).to(args.device)
    
    global_model = MultimodalEvidentialDialogueRNN(
        text_dim=768, audio_dim=768, hidden_dim=args.hidden_dim,
        num_classes=len(MELD_EMOTIONS), num_speakers=10,
        dropout=args.dropout, fusion_mode=args.fusion_mode,
    ).to(args.device)
    
    loss_fn = SupervisedEvidentialLoss(
        num_classes=len(MELD_EMOTIONS), annealing_epochs=args.annealing_epochs,
        class_weights=class_weights,
    )
    aux_loss_fn = SupervisedEvidentialLoss(
        num_classes=len(MELD_EMOTIONS), annealing_epochs=args.annealing_epochs,
        class_weights=class_weights,
    )
    aggregator = EAFAAggregator(beta=args.beta)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"  Multimodal (text+audio) | {agg_label} | seed={seed}")
    logger.info(f"  {sum(p.numel() for p in global_model.parameters()):,} params")
    logger.info(f"{'='*60}\n")
    
    best_wf1, patience_cnt = 0.0, 0
    
    for round_num in range(1, args.num_rounds + 1):
        start = time.time()
        client_states, client_sizes, client_us = [], [], []
        
        for loader in client_loaders:
            state_dict, stats = local_train_multimodal(
                global_model, loader, loss_fn, aux_loss_fn, args.device,
                args.local_epochs, args.lr, epoch=round_num,
                lambda_aux=args.lambda_aux,
            )
            client_states.append(OrderedDict({k: v.cpu() for k, v in state_dict.items()}))
            client_sizes.append(stats["num_samples"])
            client_us.append(stats["mean_uncertainty"])
        
        global_state, agg_stats = aggregator.aggregate(
            client_states, client_sizes, client_us, round_num,
        )
        global_model.load_state_dict(global_state)
        global_model.to(args.device)
        
        # Use dev for early stopping
        dev_wf1, dev_u, _ = evaluate_multimodal(global_model, dev_loader, args.device)
        test_wf1, test_u, _ = evaluate_multimodal(global_model, test_loader, args.device)
        elapsed = time.time() - start
        
        w_str = ",".join(f"{w:.2f}" for w in agg_stats["weights"])
        logger.info(
            f"R{round_num:2d}/{args.num_rounds} | Dev={dev_wf1:.4f} Test={test_wf1:.4f} u={test_u:.3f} | "
            f"w=[{w_str}] | {elapsed:.1f}s"
        )
        
        if dev_wf1 > best_wf1:
            best_wf1 = dev_wf1
            best_test_wf1 = test_wf1
            best_test_u = test_u
            patience_cnt = 0
            logger.info(f"  >> New best! Dev={dev_wf1:.4f} Test={test_wf1:.4f}")
            
            ckpt = Path(args.save_dir) / f"best_multimodal_{agg_label.lower()}.pt"
            ckpt.parent.mkdir(exist_ok=True)
            torch.save({"model_state_dict": global_model.state_dict(), "round": round_num}, ckpt)
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                logger.info(f"  Early stopping at round {round_num}")
                break
        
        del client_states
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Final eval
    ckpt = Path(args.save_dir) / f"best_multimodal_{agg_label.lower()}.pt"
    if ckpt.exists():
        global_model.load_state_dict(torch.load(ckpt, weights_only=False)["model_state_dict"])
    
    final_wf1, final_u, report = evaluate_multimodal(
        global_model, test_loader, args.device, MELD_EMOTIONS
    )
    
    logger.info(f"\n{'='*60}")
    logger.info(f"  RESULT: Multimodal {agg_label} | WF1={final_wf1:.4f}")
    logger.info(f"{'='*60}")
    logger.info(f"\n{report}")
    
    return {
        "wf1": round(final_wf1, 4),
        "uncertainty": round(final_u, 4),
        "best_dev_wf1": round(best_wf1, 4),
    }


def main():
    results = load_results()
    total_start = time.time()
    
    experiments = [
        ("multimodal_eafa", 1.0, "evidence_sum"),
        ("multimodal_fedavg", 0.0, "evidence_sum"),
        ("multimodal_logit_avg", 1.0, "logit_avg"),
        ("multimodal_learnable_gating", 1.0, "learnable_gating"),
    ]
    
    for key, beta, fusion_mode in experiments:
        if key in results and results[key].get("wf1") is not None:
            print(f"SKIP {key}: WF1={results[key]['wf1']}")
            continue
        
        print(f"\nRUNNING {key}...")
        start = time.time()
        
        try:
            r = run_multimodal_experiment(beta=beta, fusion_mode=fusion_mode, seed=42)
            r["time"] = round(time.time() - start, 1)
            results[key] = r
            save_results(results)
            print(f"  >> WF1={r['wf1']}, time={r['time']:.0f}s")
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[key] = {"wf1": None, "error": str(e)}
            save_results(results)
    
    # Summary with text-only baselines
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  MULTIMODAL RESULTS -- {total_time/60:.1f} min")
    print(f"{'='*60}")
    print(f"  Text-only EAFA:             WF1 = 0.6347")
    print(f"  Text-only FedAvg:           WF1 = 0.6345")
    
    for key, _, _ in experiments:
        wf1 = results.get(key, {}).get("wf1")
        if wf1 is not None:
            print(f"  {key:<27}: WF1 = {wf1:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
