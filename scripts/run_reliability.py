"""
Reliability Evaluation Pipeline for FedSSL-MERC
==================================================
Post-hoc evaluation of trained checkpoints for:
  RQ1: Selective Prediction (AURC, Acc@coverage)
  RQ2: Conformal Prediction (LAC, APS, Randomized APS)
  RQ3: Federated Reliability (EAFA vs FedAvg under noise/non-IID)
  RQ4: OOD Detection (IEMOCAP only, appendix)

Supports both:
  - EDL models (EvidentialDialogueRNN)
  - CE models (DialogueRNN)

Usage:
    python scripts/run_reliability.py --use-pipeline \
        --dataset iemocap --num-classes 6 \
        --model-type edl \
        --ckpt checkpoints/best_edl_iemocap.pt \
        --alpha 0.1 --seeds 42 123 2024 \
        --out results/reliability_iemocap.json
"""

import sys, os, json, time, argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.train_multi_dataset import (
    load_meld, load_iemocap, load_dailydialog,
    GenericDialogueDataset, collate_dialogues,
)
from models.evidential.evidential_dialogue_rnn import EvidentialDialogueRNN
from models.erc.dialogue_rnn import DialogueRNN
from models.evidential.selective import evaluate_selective_prediction
from models.evidential.conformal import (
    LACConformal, APSConformal, FederatedConformalPredictor,
)
from models.evidential.ood_detection import prepare_iemocap_id_ood_split, compute_ood_auroc
from torch.utils.data import DataLoader


# ============================================================
# Model inference: extract probs, alpha, predictions, labels
# ============================================================

def extract_model_outputs(
    model, loader, device
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
    """
    Run model inference and collect prediction probabilities, Dirichlet alphas (if EDL), predictions, and labels.

    Returns:
        probs: (N, C) prediction probabilities
        alpha: (N, C) Dirichlet concentration parameters (None if CE model)
        predictions: (N,) predicted class indices
        labels: (N,) true labels
    """
    model.eval()
    all_probs, all_alpha, all_preds, all_labels = [], [], [], []
    is_edl = isinstance(model, EvidentialDialogueRNN)

    with torch.no_grad():
        for batch in loader:
            feats = batch["features"].to(device)
            speakers = batch["speaker_ids"].to(device)
            labels_batch = batch["labels"].to(device)
            out = model(feats, speakers)
            mask = labels_batch != -1

            if is_edl:
                # EDL model
                alpha = out["alpha"][mask].cpu().numpy()
                strength = alpha.sum(axis=-1, keepdims=True)
                probs = alpha / strength
                preds = out["belief"][mask].argmax(dim=-1).cpu().numpy()
                all_alpha.append(alpha)
            else:
                # CE model
                logits = out[mask]
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                preds = logits.argmax(dim=-1).cpu().numpy()

            all_probs.append(probs)
            all_preds.extend(preds)
            all_labels.extend(labels_batch[mask].cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_alpha = np.concatenate(all_alpha, axis=0) if all_alpha else None
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    return all_probs, all_alpha, all_preds, all_labels


# ============================================================
# RQ1: Selective Prediction
# ============================================================

def run_rq1(
    probs: np.ndarray,
    alpha: Optional[np.ndarray],
    predictions: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> Dict:
    """RQ1: Compare selective prediction performance of confidence scores."""
    # Compute confidence scores
    confidence_scores = {
        "max_prob": probs.max(axis=-1),
    }
    # negative entropy
    log_probs = np.log(np.clip(probs, 1e-10, 1.0))
    neg_entropy = -np.sum(probs * log_probs, axis=-1)
    max_entropy = np.log(num_classes)
    confidence_scores["neg_entropy"] = 1.0 - neg_entropy / max_entropy

    if alpha is not None:
        strength = alpha.sum(axis=-1)
        u = num_classes / strength
        confidence_scores["vacuity_1mu"] = 1.0 - u

    results = evaluate_selective_prediction(
        predictions, labels, confidence_scores,
        coverage_levels=[0.5, 0.8, 0.9],
    )

    rq1_output = {}
    for name, result in results.items():
        rq1_output[name] = {
            "aurc": round(result.aurc, 6),
            "eaurc": round(result.eaurc, 6),
            "accuracy_at_coverage": {
                str(k): round(v, 4) for k, v in result.accuracy_at_coverage.items()
            },
            "overall_accuracy": round(result.overall_accuracy, 4),
            "coverages": [float(x) for x in result.coverages],
            "risks": [float(x) for x in result.risks]
        }

    return rq1_output


# ============================================================
# RQ2: Conformal Prediction (LAC, APS, Randomized APS)
# ============================================================

def run_rq2(
    dev_probs: np.ndarray,
    dev_labels: np.ndarray,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    num_classes: int,
    conformal_alpha: float = 0.1,
) -> Dict:
    """RQ2: Conformal prediction with LAC, APS, and Randomized APS."""
    methods = {
        "LAC": LACConformal(alpha=conformal_alpha),
        "APS": APSConformal(alpha=conformal_alpha, randomized=False),
        "APS_randomized": APSConformal(alpha=conformal_alpha, randomized=True),
    }

    rq2_output = {}
    for name, method in methods.items():
        method.calibrate(dev_probs, dev_labels)
        result = method.evaluate(test_probs, test_labels)

        rq2_output[name] = {
            "coverage": round(result.coverage, 4),
            "avg_set_size": round(result.avg_set_size, 4),
            "median_set_size": round(result.median_set_size, 4),
            "coverage_deviation": round(result.coverage_deviation, 4),
            "quantile": round(result.quantile, 6),
            "per_class_coverage": {
                str(k): round(v, 4) for k, v in result.per_class_coverage.items()
            },
        }

    return rq2_output


# ============================================================
# RQ3: Federated Reliability (FCP component)
# ============================================================

def run_rq3_fcp(
    model,
    dev_dialogues: list,
    dev_cache: dict,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    num_classes: int,
    num_clients: int = 5,
    dirichlet_alpha: float = 0.5,
    conformal_alpha: float = 0.1,
    seed: int = 42,
    device: str = "cpu",
) -> Dict:
    """
    RQ3 (FCP component): Real Federated Conformal Prediction.
    Splits dev data into K client partitions, computes distributed quantile.
    """
    from data.federated_partition import FederatedPartitioner

    # Partition dev dialogues into K clients
    partitioner = FederatedPartitioner(
        num_clients=num_clients, strategy="dirichlet",
        alpha=dirichlet_alpha, seed=seed,
    )
    client_partitions = partitioner.partition(dev_dialogues, label_ratio=1.0)
    dialogue_lookup = {d.dialogue_id: d for d in dev_dialogues}

    # Extract per-client dev probs and labels
    client_probs_list = []
    client_labels_list = []

    for partition in client_partitions:
        dias = [dialogue_lookup[did] for did in partition.dialogue_ids
                if did in dialogue_lookup]
        if not dias:
            continue

        ds = GenericDialogueDataset(dias, dev_cache)
        loader = DataLoader(ds, batch_size=16, shuffle=False,
                           collate_fn=collate_dialogues, num_workers=0)
        probs_k, _, _, labels_k = extract_model_outputs(model, loader, device)

        client_probs_list.append(probs_k)
        client_labels_list.append(labels_k)

    # Run FCP with LAC and APS
    fcp_results = {}
    for method_name in ["lac", "aps"]:
        fcp = FederatedConformalPredictor(
            alpha=conformal_alpha, method=method_name, privacy_mode="scores",
        )
        fcp.calibrate_federated(client_probs_list, client_labels_list)
        result = fcp.evaluate(test_probs, test_labels)

        fcp_results[f"FCP_{method_name.upper()}"] = {
            "coverage": round(result.coverage, 4),
            "avg_set_size": round(result.avg_set_size, 4),
            "median_set_size": round(result.median_set_size, 4),
            "coverage_deviation": round(result.coverage_deviation, 4),
            "quantile": round(result.quantile, 6),
            "per_class_coverage": {
                str(k): round(v, 4) for k, v in result.per_class_coverage.items()
            },
        }

    return fcp_results


# ============================================================
# RQ4: OOD Detection (IEMOCAP only)
# ============================================================

def run_rq4(
    model,
    train_dialogues: list,
    test_dialogues: list,
    train_cache: dict,
    test_cache: dict,
    num_classes: int,
    device: str = "cpu",
    seed: int = 42,
) -> Dict:
    """RQ4: OOD detection using corrected ID/OOD split (Advisor A4)."""
    # Corrected split: ID = holdout from train speakers, OOD = test (session 5)
    id_dialogues, ood_dialogues = prepare_iemocap_id_ood_split(
        train_dialogues, test_dialogues, holdout_fraction=0.15, seed=seed,
    )

    # Extract for ID and OOD
    id_ds = GenericDialogueDataset(id_dialogues, train_cache)
    id_loader = DataLoader(id_ds, batch_size=16, shuffle=False,
                          collate_fn=collate_dialogues, num_workers=0)
    id_probs, id_alpha, _, _ = extract_model_outputs(model, id_loader, device)

    ood_ds = GenericDialogueDataset(ood_dialogues, test_cache)
    ood_loader = DataLoader(ood_ds, batch_size=16, shuffle=False,
                           collate_fn=collate_dialogues, num_workers=0)
    ood_probs, ood_alpha, _, _ = extract_model_outputs(model, ood_loader, device)

    # Evaluate OOD detection using appropriate uncertainty metrics
    score_names = ["entropy", "max_prob_inv"]
    if id_alpha is not None and ood_alpha is not None:
        score_names.append("vacuity_u")

    rq4_output = {}
    for name in score_names:
        if name == "entropy":
            id_scores = -np.sum(id_probs * np.log(np.clip(id_probs, 1e-10, 1.0)), axis=-1)
            ood_scores = -np.sum(ood_probs * np.log(np.clip(ood_probs, 1e-10, 1.0)), axis=-1)
        elif name == "max_prob_inv":
            id_scores = 1.0 - id_probs.max(axis=-1)
            ood_scores = 1.0 - ood_probs.max(axis=-1)
        elif name == "vacuity_u":
            id_scores = num_classes / id_alpha.sum(axis=-1)
            ood_scores = num_classes / ood_alpha.sum(axis=-1)

        auroc, fpr95, _, _ = compute_ood_auroc(id_scores, ood_scores)

        rq4_output[name] = {
            "auroc": round(auroc, 4),
            "fpr_at_tpr95": round(fpr95, 4),
            "id_mean_score": round(float(id_scores.mean()), 4),
            "ood_mean_score": round(float(ood_scores.mean()), 4),
        }

    return rq4_output


# ============================================================
# Main Pipeline
# ============================================================

def run_full_pipeline(args):
    """Run all RQs for a single dataset and checkpoint."""
    device = args.device
    dataset = args.dataset
    num_classes = args.num_classes

    # Load dataset
    loaders = {"meld": load_meld, "iemocap": load_iemocap, "dailydialog": load_dailydialog}
    load_fn = loaders[dataset]

    if dataset == "iemocap":
        train_dias, dev_dias, test_dias, emotions, weights, cache, num_spk = load_fn(
            finetuned=True, num_classes=num_classes
        )
    else:
        train_dias, dev_dias, test_dias, emotions, weights, cache, num_spk = load_fn(
            finetuned=True
        )

    # Load model depending on type
    if args.model_type == "ce":
        print(f"[{dataset.upper()}] Loading CE model (DialogueRNN)...")
        model = DialogueRNN(
            input_dim=768, hidden_dim=256,
            num_classes=num_classes, num_speakers=num_spk, dropout=0.3,
        ).to(device)
    else:
        print(f"[{dataset.upper()}] Loading EDL model (EvidentialDialogueRNN)...")
        model = EvidentialDialogueRNN(
            input_dim=768, hidden_dim=256,
            num_classes=num_classes, num_speakers=num_spk, dropout=0.3,
        ).to(device)

    ckpt_path = args.ckpt.replace("{seed}", str(args.seed))
    print(f"Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Create data loaders
    dev_ds = GenericDialogueDataset(dev_dias, cache.get("dev", cache.get("val", {})))
    test_ds = GenericDialogueDataset(test_dias, cache.get("test", {}))

    dev_loader = DataLoader(dev_ds, batch_size=16, shuffle=False,
                           collate_fn=collate_dialogues, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False,
                            collate_fn=collate_dialogues, num_workers=0)

    # Extract model outputs for dev and test
    print(f"[{dataset.upper()}] Extracting model outputs...")
    dev_probs, dev_alpha, dev_preds, dev_labels = extract_model_outputs(model, dev_loader, device)
    test_probs, test_alpha, test_preds, test_labels = extract_model_outputs(model, test_loader, device)

    print(f"  Dev: {len(dev_labels)} utterances, Test: {len(test_labels)} utterances")

    all_results = {
        "dataset": dataset,
        "num_classes": num_classes,
        "checkpoint": ckpt_path,
        "model_type": args.model_type,
        "seed": args.seed,
        "conformal_alpha": args.alpha,
    }

    # RQ1: Selective Prediction
    print(f"[{dataset.upper()}] RQ1: Selective Prediction...")
    all_results["rq1_selective"] = run_rq1(test_probs, test_alpha, test_preds, test_labels, num_classes)

    # RQ2: Conformal Prediction (centralized calibration)
    print(f"[{dataset.upper()}] RQ2: Conformal Prediction...")
    all_results["rq2_conformal"] = run_rq2(
        dev_probs, dev_labels, test_probs, test_labels, num_classes, args.alpha
    )

    # RQ3: Federated Conformal Prediction (distributed quantile)
    print(f"[{dataset.upper()}] RQ3: Federated Conformal Prediction...")
    dev_cache = cache.get("dev", cache.get("val", {}))
    all_results["rq3_fcp"] = run_rq3_fcp(
        model, dev_dias, dev_cache, test_probs, test_labels,
        num_classes, num_clients=5, dirichlet_alpha=0.5,
        conformal_alpha=args.alpha, seed=args.seed, device=device,
    )

    # RQ4: OOD Detection (IEMOCAP only)
    if dataset == "iemocap":
        print(f"[{dataset.upper()}] RQ4: OOD Detection...")
        train_cache = cache.get("train", {})
        test_cache = cache.get("test", {})
        all_results["rq4_ood"] = run_rq4(
            model, train_dias, test_dias, train_cache, test_cache,
            num_classes, device=device, seed=args.seed,
        )
    else:
        all_results["rq4_ood"] = "N/A (IEMOCAP only)"

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Reliability Evaluation Pipeline")
    parser.add_argument("--use-pipeline", action="store_true", help="Enable full pipeline")
    parser.add_argument("--dataset", type=str, required=True, choices=["meld", "iemocap", "dailydialog"])
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--model-type", type=str, default="edl", choices=["edl", "ce"], help="edl or ce model type")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained checkpoint")
    parser.add_argument("--alpha", type=float, default=0.1, help="Conformal miscoverage rate")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    parser.add_argument("--out", type=str, default=None, help="Output JSON path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    all_seed_results = []

    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"  SEED {seed}")
        print(f"{'='*60}")
        args.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        result = run_full_pipeline(args)
        all_seed_results.append(result)

    # Aggregate across seeds
    final_output = {
        "dataset": args.dataset,
        "num_classes": args.num_classes,
        "checkpoint": args.ckpt,
        "model_type": args.model_type,
        "conformal_alpha": args.alpha,
        "seeds": args.seeds,
        "per_seed_results": all_seed_results,
    }

    # Save
    out_path = args.out or f"results/reliability_{args.dataset}_{args.model_type}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(final_output, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  Results saved to: {out_path}")
    print(f"{'='*60}")

    # Print summary
    for seed_result in all_seed_results:
        seed = seed_result["seed"]
        print(f"\n--- Seed {seed} ---")
        if "rq1_selective" in seed_result:
            for score_name, metrics in seed_result["rq1_selective"].items():
                print(f"  RQ1 [{score_name}]: AURC={metrics['aurc']:.6f}")
        if "rq2_conformal" in seed_result:
            for method_name, metrics in seed_result["rq2_conformal"].items():
                print(f"  RQ2 [{method_name}]: coverage={metrics['coverage']:.4f}, "
                      f"avg_set_size={metrics['avg_set_size']:.4f}")
        if "rq3_fcp" in seed_result:
            for method_name, metrics in seed_result["rq3_fcp"].items():
                print(f"  RQ3 [{method_name}]: coverage={metrics['coverage']:.4f}, "
                      f"avg_set_size={metrics['avg_set_size']:.4f}")
        if "rq4_ood" in seed_result and isinstance(seed_result["rq4_ood"], dict):
            for score_name, metrics in seed_result["rq4_ood"].items():
                print(f"  RQ4 [{score_name}]: AUROC={metrics['auroc']:.4f}")


if __name__ == "__main__":
    main()
