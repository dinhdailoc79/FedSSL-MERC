"""
Calibration and Temperature Scaling for Evidential Models
==========================================================
Loads the best evidential models on MELD and IEMOCAP, fits a temperature scaler
on the validation set, calibrates the predictions on the test set,
and saves the side-by-side reliability diagrams.

Usage:
    python scripts/run_calibration_tempscale.py
"""

import sys, os, json, time, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_FILE = "results_calibration.json"


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)


def run_calibration_for_dataset(dataset_name, checkpoint_path, num_classes):
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"\n==================================================")
    logger.info(f"  CALIBRATION: {dataset_name} | Checkpoint: {checkpoint_path}")
    logger.info(f"==================================================")

    # 1. Load data
    from scripts.train_multi_dataset import (
        load_meld, load_iemocap, GenericDialogueDataset, collate_dialogues
    )
    from models.evidential.evidential_dialogue_rnn import EvidentialDialogueRNN
    from models.evidential.calibration import (
        compute_calibration_metrics, TemperatureScaler, plot_reliability_before_after
    )

    if dataset_name.lower() == "meld":
        train_dias, dev_dias, test_dias, emotions, weights, cache, num_spk = load_meld(finetuned=True)
    elif dataset_name.lower() == "iemocap":
        train_dias, dev_dias, test_dias, emotions, weights, cache, num_spk = load_iemocap(finetuned=True, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Build dev loader
    dev_ds = GenericDialogueDataset(dev_dias, cache.get("dev", cache.get("val", {})))
    dev_loader = DataLoader(dev_ds, batch_size=16, shuffle=False, collate_fn=collate_dialogues)

    # Build test loader
    test_ds = GenericDialogueDataset(test_dias, cache.get("test", {}))
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_dialogues)

    # 2. Reconstruct Model
    model = EvidentialDialogueRNN(
        input_dim=768, hidden_dim=256,
        num_classes=num_classes, num_speakers=num_spk, dropout=0.3,
    ).to(device)

    # Load state dict
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return None

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    
    model.eval()

    # 3. Gather validation logits and labels
    val_logits_list, val_labels_list = [], []
    with torch.no_grad():
        for batch in dev_loader:
            feats = batch["features"].to(device)
            speakers = batch["speaker_ids"].to(device)
            labels = batch["labels"].to(device)
            
            out = model(feats, speakers)
            mask = labels != -1
            if mask.sum() == 0:
                continue
            val_logits_list.append(out["logits"][mask])
            val_labels_list.append(labels[mask])

    val_logits = torch.cat(val_logits_list, dim=0)
    val_labels = torch.cat(val_labels_list, dim=0)

    # 4. Gather test logits and labels
    test_logits_list, test_labels_list = [], []
    with torch.no_grad():
        for batch in test_loader:
            feats = batch["features"].to(device)
            speakers = batch["speaker_ids"].to(device)
            labels = batch["labels"].to(device)
            
            out = model(feats, speakers)
            mask = labels != -1
            if mask.sum() == 0:
                continue
            test_logits_list.append(out["logits"][mask])
            test_labels_list.append(labels[mask])

    test_logits = torch.cat(test_logits_list, dim=0)
    test_labels = torch.cat(test_labels_list, dim=0)

    # 5. Compute "Before" Calibration Metrics
    # Compute alpha before scaling (T=1.0)
    with torch.no_grad():
        evidence_b = torch.nn.functional.softplus(test_logits)
        alpha_b = evidence_b + 1.0
    metrics_before = compute_calibration_metrics(alpha_b, test_labels)

    # 6. Fit Temperature Scaler on Dev Set
    scaler = TemperatureScaler().to(device)
    fitted_temp = scaler.fit(val_logits, val_labels)
    logger.info(f"  Fitted Temperature T: {fitted_temp:.4f}")

    # 7. Compute "After" Calibration Metrics
    with torch.no_grad():
        alpha_a = scaler(test_logits)
    metrics_after = compute_calibration_metrics(alpha_a, test_labels)

    logger.info(f"  Before Calibration ECE: {metrics_before['ece']:.4f} | NLL: {metrics_before['nll']:.4f}")
    logger.info(f"  After Calibration ECE:  {metrics_after['ece']:.4f}  | NLL: {metrics_after['nll']:.4f}")

    # 8. Save Reliability Diagram Comparison Plot
    os.makedirs("results", exist_ok=True)
    save_plot_path = f"results/reliability_diagram_{dataset_name.lower()}_before_after.png"
    plot_reliability_before_after(
        metrics_before, metrics_after, save_plot_path, dataset_name=dataset_name
    )
    logger.info(f"  Saved reliability plot to: {save_plot_path}")

    # Convert arrays to lists for JSON serialization
    def clean_metrics(m):
        m_c = dict(m)
        for k in ["bin_accuracies", "bin_confidences", "bin_sizes"]:
            if k in m_c:
                m_c[k] = [float(x) for x in m_c[k]]
        return m_c

    return {
        "dataset": dataset_name,
        "temperature": round(fitted_temp, 4),
        "before": clean_metrics(metrics_before),
        "after": clean_metrics(metrics_after),
        "plot_path": save_plot_path,
    }


def main():
    results = load_results()
    
    # We will calibrate both MELD and IEMOCAP (6 classes)
    configs = [
        ("MELD", "checkpoints/best_eafa_edl_meld.pt", 7),
        ("IEMOCAP", "checkpoints/best_eafa_edl_iemocap.pt", 6),
    ]

    for dataset_name, checkpoint_path, num_classes in configs:
        if not os.path.exists(checkpoint_path):
            print(f"Skipping {dataset_name} as checkpoint {checkpoint_path} does not exist.")
            continue
            
        r = run_calibration_for_dataset(dataset_name, checkpoint_path, num_classes)
        if r is not None:
            results[dataset_name.lower()] = r
            save_results(results)

    print("\n==============================================")
    print("  CALIBRATION & TEMPERATURE SCALING COMPLETED")
    print("==============================================")
    for ds_name, _, _ in configs:
        key = ds_name.lower()
        if key in results:
            data = results[key]
            print(f"{ds_name}:")
            print(f"  Optimal Temperature: T = {data['temperature']:.4f}")
            print(f"  ECE Before: {data['before']['ece']:.4f} --> ECE After: {data['after']['ece']:.4f}")
            print(f"  NLL Before: {data['before']['nll']:.4f} --> NLL After: {data['after']['nll']:.4f}")
    print("==============================================\n")


if __name__ == "__main__":
    main()
