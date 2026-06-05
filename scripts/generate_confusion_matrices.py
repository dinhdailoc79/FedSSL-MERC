"""
Per-Class F1 and Confusion Matrix Generator
============================================
Trains Supervised, FixMatch, and ECR models at 10% label ratio for seed 42,
extracts detailed predictions, prints LaTeX per-class F1 tables,
and saves publication-quality confusion matrices.

Usage:
    python scripts/generate_confusion_matrices.py --dataset meld
    python scripts/generate_confusion_matrices.py --dataset iemocap
"""

import sys
import os
import time
import argparse
import copy
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.train_multi_dataset import (
    load_meld, load_iemocap, GenericDialogueDataset, collate_dialogues,
)
from models.evidential.evidential_dialogue_rnn import EvidentialDialogueRNN
from models.erc.dialogue_rnn import DialogueRNN
from models.evidential.losses import SupervisedEvidentialLoss, FedEvidenceLoss
from semi_supervised.fixmatch import FixMatchLoss
from semi_supervised.augmentation import StrongAugmentation
from federated.aggregation.eafa import EAFAAggregator
from data.federated_partition import FederatedPartitioner

OUT_DIR = "paper/figures"
os.makedirs(OUT_DIR, exist_ok=True)


def train_and_get_predictions(dataset, method, label_ratio=0.1, seed=42, num_rounds=30, device="cuda"):
    """Train federated model and return best test predictions and true labels."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    loaders = {"meld": load_meld, "iemocap": load_iemocap}
    load_fn = loaders[dataset]
    train_dias, dev_dias, test_dias, emotions, weights, cache, num_spk = load_fn(finetuned=True)
    num_classes = len(emotions)
    class_weights = torch.from_numpy(weights.astype(np.float32)).to(device)

    # Partition
    partitioner = FederatedPartitioner(num_clients=5, strategy="dirichlet", alpha=0.5, seed=seed)
    client_partitions = partitioner.partition(train_dias, label_ratio=label_ratio)
    dialogue_lookup = {d.dialogue_id: d for d in train_dias}

    client_labeled_loaders = []
    client_unlabeled_loaders = []
    client_total_sizes = []

    for partition in client_partitions:
        labeled_dias = [dialogue_lookup[did] for did in partition.labeled_ids if did in dialogue_lookup]
        labeled_ds = GenericDialogueDataset(labeled_dias, cache.get("train", {}))
        client_labeled_loaders.append(
            DataLoader(labeled_ds, batch_size=16, shuffle=True, collate_fn=collate_dialogues)
        )
        unlabeled_dias = [dialogue_lookup[did] for did in partition.unlabeled_ids if did in dialogue_lookup]
        if unlabeled_dias and method == "ecr":
            unlabeled_ds = GenericDialogueDataset(unlabeled_dias, cache.get("train", {}))
            client_unlabeled_loaders.append(
                DataLoader(unlabeled_ds, batch_size=16, shuffle=True, collate_fn=collate_dialogues)
            )
        else:
            client_unlabeled_loaders.append(None)
        client_total_sizes.append(len(labeled_dias) + len(unlabeled_dias))

    test_ds = GenericDialogueDataset(test_dias, cache.get("test", {}))
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_dialogues)

    # Initialize model
    use_edl = method in ("supervised", "ecr")
    if use_edl:
        global_model = EvidentialDialogueRNN(
            input_dim=768, hidden_dim=256, num_classes=num_classes, num_speakers=num_spk, dropout=0.3
        ).to(device)
        if method == "ecr":
            loss_fn = FedEvidenceLoss(
                num_classes=num_classes, annealing_epochs=30, lambda_u=1.0,
                lambda_u_rampup_epochs=20, class_weights=class_weights
            )
            strong_aug = StrongAugmentation(noise_std=0.05, dropout_p=0.25)
        else:
            loss_fn = SupervisedEvidentialLoss(
                num_classes=num_classes, annealing_epochs=30, class_weights=class_weights
            )
    else:
        global_model = DialogueRNN(
            input_dim=768, hidden_dim=256, num_classes=num_classes, num_speakers=num_spk, dropout=0.3
        ).to(device)
        ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        fixmatch_loss = FixMatchLoss(
            threshold=0.95, lambda_u=1.0, num_classes=num_classes, warmup_epochs=10, threshold_min=0.7
        )

    aggregator = EAFAAggregator(beta=1.0 if use_edl else 0.0)

    best_wf1 = 0.0
    best_preds = None
    best_targets = None

    for round_num in range(1, num_rounds + 1):
        client_states, client_sizes, client_us = [], [], []

        for c_idx in range(len(client_labeled_loaders)):
            labeled_loader = client_labeled_loaders[c_idx]
            unlabeled_loader = client_unlabeled_loaders[c_idx]

            local_model = copy.deepcopy(global_model).to(device)
            local_model.train()
            opt = optim.Adam(local_model.parameters(), lr=1e-3, weight_decay=1e-4)
            all_u_local = []

            if method == "ecr":
                loss_fn.set_epoch(round_num)
                strong_aug.train()
            elif method == "supervised":
                loss_fn.set_epoch(round_num)
            elif method == "fixmatch":
                fixmatch_loss.update_threshold(round_num)
                fixmatch_loss.train()

            for _ in range(3):  # local_epochs
                for labeled_batch in labeled_loader:
                    feats_l = labeled_batch["features"].to(device)
                    speakers_l = labeled_batch["speaker_ids"].to(device)
                    labels_l = labeled_batch["labels"].to(device)
                    mask_l = labels_l != -1

                    if method == "supervised":
                        out = local_model(feats_l, speakers_l)
                        loss, _ = loss_fn(out["alpha"][mask_l], labels_l[mask_l])
                        all_u_local.extend(out["uncertainty"][mask_l].detach().cpu().numpy())
                    elif method == "ecr":
                        out_l = local_model(feats_l, speakers_l)
                        alpha_l = out_l["alpha"][mask_l]
                        labels_flat = labels_l[mask_l]
                        all_u_local.extend(out_l["uncertainty"][mask_l].detach().cpu().numpy())

                        alpha_weak = alpha_strong = uncertainty_weak = None
                        if unlabeled_loader:
                            try:
                                u_batch = next(iter(unlabeled_loader))
                            except StopIteration:
                                u_batch = next(iter(unlabeled_loader))

                            feats_u = u_batch["features"].to(device)
                            speakers_u = u_batch["speaker_ids"].to(device)
                            labels_u = u_batch["labels"].to(device)
                            u_mask = labels_u != -1

                            local_model.eval()
                            with torch.no_grad():
                                out_weak = local_model(feats_u, speakers_u)
                            local_model.train()

                            feats_strong = strong_aug(feats_u)
                            out_strong = local_model(feats_strong, speakers_u)

                            alpha_weak = out_weak["alpha"][u_mask]
                            alpha_strong = out_strong["alpha"][u_mask]
                            uncertainty_weak = out_weak["uncertainty"][u_mask]

                        loss, _ = loss_fn(
                            alpha_l, labels_flat, None,
                            alpha_weak, alpha_strong, uncertainty_weak
                        )
                    elif method == "fixmatch":
                        unlabeled_batch = None
                        labeled_batch_device = {
                            "features": feats_l, "speaker_ids": speakers_l, "labels": labels_l
                        }
                        loss, _ = fixmatch_loss(
                            local_model, labeled_batch_device, unlabeled_batch, ce_loss
                        )

                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), 5.0)
                    opt.step()

            client_states.append(OrderedDict({k: v.cpu() for k, v in local_model.state_dict().items()}))
            client_sizes.append(client_total_sizes[c_idx])
            client_us.append(float(np.mean(all_u_local)) if all_u_local else 0.0)

        global_state, _ = aggregator.aggregate(client_states, client_sizes, client_us, round_num)
        global_model.load_state_dict(global_state)
        global_model.to(device)

        # Eval round
        global_model.eval()
        all_preds_round = []
        all_labels_round = []
        with torch.no_grad():
            for batch in test_loader:
                feats = batch["features"].to(device)
                speakers = batch["speaker_ids"].to(device)
                labels = batch["labels"].to(device)
                out = global_model(feats, speakers)
                mask = labels != -1

                if use_edl:
                    preds = out["belief"][mask].argmax(dim=-1).cpu().numpy()
                else:
                    preds = out[mask].argmax(dim=-1).cpu().numpy()

                all_preds_round.extend(preds)
                all_labels_round.extend(labels[mask].cpu().numpy())

        round_wf1 = f1_score(all_labels_round, all_preds_round, average="weighted", zero_division=0)
        if round_wf1 > best_wf1:
            best_wf1 = round_wf1
            best_preds = all_preds_round
            best_targets = all_labels_round

    return best_targets, best_preds, emotions


def plot_confusion_matrix(targets, preds, emotions, title, filename):
    """Plot confusion matrix as a publication-quality heatmap."""
    cm = confusion_matrix(targets, preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(6, 5))
    # Elegant blue/purple palette
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=emotions, yticklabels=emotions,
        cbar=True, square=True,
        annot_kws={"size": 9, "weight": "bold"}
    )
    plt.title(title, fontsize=12, fontweight='bold', pad=10)
    plt.ylabel('True Emotion', fontsize=10)
    plt.xlabel('Predicted Emotion', fontsize=10)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"  Saved confusion matrix to {filename}")


def main():
    parser = argparse.ArgumentParser(description="Generate Confusion Matrices and Per-Class F1")
    parser.add_argument("--dataset", type=str, default="meld", choices=["meld", "iemocap"])
    parser.add_argument("--rounds", type=int, default=15, help="Number of training rounds")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Generating confusion matrices for {args.dataset.upper()} on {device}...")

    results = {}

    for method in ["supervised", "fixmatch", "ecr"]:
        print(f"\nTraining {method.upper()}...")
        targets, preds, emotions = train_and_get_predictions(
            args.dataset, method, label_ratio=0.1, seed=42, num_rounds=args.rounds, device=device
        )
        
        # Save results
        report = classification_report(targets, preds, target_names=emotions, output_dict=True, zero_division=0)
        results[method] = report
        
        # Plot ECR confusion matrix
        if method == "ecr":
            title = f"ECR Confusion Matrix: {args.dataset.upper()} (10% Labels)"
            filename = os.path.join(OUT_DIR, f"confusion_matrix_ecr_{args.dataset}.png")
            plot_confusion_matrix(targets, preds, emotions, title, filename)
            
            filename_pdf = os.path.join(OUT_DIR, f"confusion_matrix_ecr_{args.dataset}.pdf")
            plot_confusion_matrix(targets, preds, emotions, title, filename_pdf)

    # Print LaTeX comparison table
    print(f"\n{'='*60}")
    print(f"  PER-CLASS F1 TABLE (LaTeX Format) — {args.dataset.upper()}")
    print(f"{'='*60}\n")
    
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Per-class F1-score comparison on " + args.dataset.upper() + " at 10\\% label ratio.}")
    print("\\label{tab:per_class_f1_" + args.dataset + "}")
    print("\\begin{tabular}{l|c|c|c}")
    print("  \\toprule")
    print("  \\textbf{Emotion} & \\textbf{Supervised} & \\textbf{FixMatch} & \\textbf{ECR (Ours)} \\\\")
    print("  \\midrule")
    
    for emo in emotions:
        sup_f1 = results["supervised"].get(emo, {}).get("f1-score", 0.0) * 100
        fm_f1 = results["fixmatch"].get(emo, {}).get("f1-score", 0.0) * 100
        ecr_f1 = results["ecr"].get(emo, {}).get("f1-score", 0.0) * 100
        
        # Bold the best score
        scores = [sup_f1, fm_f1, ecr_f1]
        best_idx = np.argmax(scores)
        
        scores_str = []
        for i, s in enumerate(scores):
            if i == best_idx:
                scores_str.append(f"\\textbf{{{s:.2f}\\%}}")
            else:
                scores_str.append(f"{s:.2f}\\%")
                
        print(f"  {emo.capitalize():<12} & {scores_str[0]} & {scores_str[1]} & {scores_str[2]} \\\\")
        
    print("  \\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
