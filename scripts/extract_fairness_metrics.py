"""
Extract Per-Class F1 and Per-Client Fairness Metrics
=====================================================
Evaluates the saved checkpoints for FedAvg vs EAFA
on MELD and IEMOCAP, computing minority class F1 scores
and standard deviation of WF1 across partitioned clients.
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.evidential.evidential_dialogue_rnn import EvidentialDialogueRNN
from data.federated_partition import FederatedPartitioner
from scripts.train_multi_dataset import (
    load_meld, load_iemocap, GenericDialogueDataset, collate_dialogues
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def evaluate_metrics(model, loader, device):
    model.eval()
    model.to(device)
    all_preds, all_labels = [], []
    for batch in loader:
        features = batch["features"].to(device)
        speakers = batch["speaker_ids"].to(device)
        labels = batch["labels"].to(device)
        out = model(features, speakers)
        mask = labels != -1
        preds = out["belief"][mask].argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels[mask].cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds)


def run_evaluation():
    print(f"{'='*60}")
    print(f"  EXTRACTING FAIRNESS & PER-CLASS F1 METRICS")
    print(f"{'='*60}\n")
    
    # ----------------------------------------------------
    # MELD EVALUATION
    # ----------------------------------------------------
    print("Loading MELD (finetuned text embeddings)...")
    train, dev, test, emotions, weights, cache, num_spk = load_meld(finetuned=True)
    
    # Load model
    fedavg_meld = EvidentialDialogueRNN(input_dim=768, hidden_dim=256, num_classes=len(emotions), num_speakers=num_spk).to(DEVICE)
    eafa_meld = EvidentialDialogueRNN(input_dim=768, hidden_dim=256, num_classes=len(emotions), num_speakers=num_spk).to(DEVICE)
    
    ckpt_fedavg = torch.load("checkpoints/best_fedavg_edl_meld.pt", map_location=DEVICE, weights_only=False)
    ckpt_eafa = torch.load("checkpoints/best_eafa_edl_meld.pt", map_location=DEVICE, weights_only=False)
    
    fedavg_meld.load_state_dict(ckpt_fedavg["model_state_dict"])
    eafa_meld.load_state_dict(ckpt_eafa["model_state_dict"])
    
    test_ds = GenericDialogueDataset(test, cache.get("test", {}), 768)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_dialogues)
    
    # Global metrics
    y_true, y_pred_fedavg = evaluate_metrics(fedavg_meld, test_loader, DEVICE)
    _, y_pred_eafa = evaluate_metrics(eafa_meld, test_loader, DEVICE)
    
    print("\nMELD Global Classification Report (FedAvg):")
    print(classification_report(y_true, y_pred_fedavg, target_names=emotions, digits=4, zero_division=0))
    print("\nMELD Global Classification Report (EAFA):")
    print(classification_report(y_true, y_pred_eafa, target_names=emotions, digits=4, zero_division=0))
    
    # Per-client Fairness
    partitioner = FederatedPartitioner(num_clients=5, strategy="dirichlet", alpha=0.5, seed=42)
    client_partitions = partitioner.partition(test, label_ratio=1.0)
    
    test_lookup = {d.dialogue_id: d for d in test}
    client_wf1_fedavg = []
    client_wf1_eafa = []
    
    for c in client_partitions:
        c_dialogues = [test_lookup[did] for did in c.dialogue_ids if did in test_lookup]
        c_ds = GenericDialogueDataset(c_dialogues, cache.get("test", {}), 768)
        c_loader = DataLoader(c_ds, batch_size=16, shuffle=False, collate_fn=collate_dialogues)
        
        yc_true, yc_pred_fedavg = evaluate_metrics(fedavg_meld, c_loader, DEVICE)
        _, yc_pred_eafa = evaluate_metrics(eafa_meld, c_loader, DEVICE)
        
        wf1_fedavg = f1_score(yc_true, yc_pred_fedavg, average="weighted", zero_division=0)
        wf1_eafa = f1_score(yc_true, yc_pred_eafa, average="weighted", zero_division=0)
        
        client_wf1_fedavg.append(wf1_fedavg)
        client_wf1_eafa.append(wf1_eafa)
        
    print(f"\nMELD Per-Client WF1 (FedAvg): {[round(x, 4) for x in client_wf1_fedavg]}")
    print(f"MELD Per-Client WF1 (EAFA):   {[round(x, 4) for x in client_wf1_eafa]}")
    
    # Save statistics
    print(f"\nMELD SUMMARY:")
    print(f"  FedAvg Global WF1: {f1_score(y_true, y_pred_fedavg, average='weighted', zero_division=0):.4f}")
    print(f"  EAFA Global WF1:   {f1_score(y_true, y_pred_eafa, average='weighted', zero_division=0):.4f}")
    print(f"  FedAvg Client Std: {np.std(client_wf1_fedavg):.4f}")
    print(f"  EAFA Client Std:   {np.std(client_wf1_eafa):.4f}")
    
    
    # ----------------------------------------------------
    # IEMOCAP EVALUATION
    # ----------------------------------------------------
    print(f"\n{'-'*60}\nLoading IEMOCAP (finetuned text embeddings)...")
    train, dev, test, emotions_ie, weights_ie, cache_ie, num_spk_ie = load_iemocap(finetuned=True)
    
    fedavg_ie = EvidentialDialogueRNN(input_dim=768, hidden_dim=256, num_classes=len(emotions_ie), num_speakers=num_spk_ie).to(DEVICE)
    eafa_ie = EvidentialDialogueRNN(input_dim=768, hidden_dim=256, num_classes=len(emotions_ie), num_speakers=num_spk_ie).to(DEVICE)
    
    ckpt_fedavg_ie = torch.load("checkpoints/best_fedavg_edl_iemocap.pt", map_location=DEVICE, weights_only=False)
    ckpt_eafa_ie = torch.load("checkpoints/best_eafa_edl_iemocap.pt", map_location=DEVICE, weights_only=False)
    
    fedavg_ie.load_state_dict(ckpt_fedavg_ie["model_state_dict"])
    eafa_ie.load_state_dict(ckpt_eafa_ie["model_state_dict"])
    
    test_ds_ie = GenericDialogueDataset(test, cache_ie.get("test", {}), 768)
    test_loader_ie = DataLoader(test_ds_ie, batch_size=16, shuffle=False, collate_fn=collate_dialogues)
    
    y_true_ie, y_pred_fedavg_ie = evaluate_metrics(fedavg_ie, test_loader_ie, DEVICE)
    _, y_pred_eafa_ie = evaluate_metrics(eafa_ie, test_loader_ie, DEVICE)
    
    print("\nIEMOCAP Global Classification Report (FedAvg):")
    print(classification_report(y_true_ie, y_pred_fedavg_ie, target_names=emotions_ie, digits=4, zero_division=0))
    print("\nIEMOCAP Global Classification Report (EAFA):")
    print(classification_report(y_true_ie, y_pred_eafa_ie, target_names=emotions_ie, digits=4, zero_division=0))
    
    # Per-client Fairness
    partitioner_ie = FederatedPartitioner(num_clients=5, strategy="dirichlet", alpha=0.5, seed=42)
    client_partitions_ie = partitioner_ie.partition(test, label_ratio=1.0)
    
    test_lookup_ie = {d.dialogue_id: d for d in test}
    client_wf1_fedavg_ie = []
    client_wf1_eafa_ie = []
    
    for c in client_partitions_ie:
        c_dialogues = [test_lookup_ie[did] for did in c.dialogue_ids if did in test_lookup_ie]
        c_ds = GenericDialogueDataset(c_dialogues, cache_ie.get("test", {}), 768)
        c_loader = DataLoader(c_ds, batch_size=16, shuffle=False, collate_fn=collate_dialogues)
        
        yc_true, yc_pred_fedavg_ie = evaluate_metrics(fedavg_ie, c_loader, DEVICE)
        _, yc_pred_eafa_ie = evaluate_metrics(eafa_ie, c_loader, DEVICE)
        
        wf1_fedavg = f1_score(yc_true, yc_pred_fedavg_ie, average="weighted", zero_division=0)
        wf1_eafa = f1_score(yc_true, yc_pred_eafa_ie, average="weighted", zero_division=0)
        
        client_wf1_fedavg_ie.append(wf1_fedavg)
        client_wf1_eafa_ie.append(wf1_eafa)
        
    print(f"\nIEMOCAP Per-Client WF1 (FedAvg): {[round(x, 4) for x in client_wf1_fedavg_ie]}")
    print(f"IEMOCAP Per-Client WF1 (EAFA):   {[round(x, 4) for x in client_wf1_eafa_ie]}")
    
    print(f"\nIEMOCAP SUMMARY:")
    print(f"  FedAvg Global WF1: {f1_score(y_true_ie, y_pred_fedavg_ie, average='weighted', zero_division=0):.4f}")
    print(f"  EAFA Global WF1:   {f1_score(y_true_ie, y_pred_eafa_ie, average='weighted', zero_division=0):.4f}")
    print(f"  FedAvg Client Std: {np.std(client_wf1_fedavg_ie):.4f}")
    print(f"  EAFA Client Std:   {np.std(client_wf1_eafa_ie):.4f}")


if __name__ == "__main__":
    run_evaluation()
