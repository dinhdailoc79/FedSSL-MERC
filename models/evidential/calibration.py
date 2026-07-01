"""
EDL Calibration Metrics
=========================
Implements Expected Calibration Error (ECE), Negative Log-Likelihood (NLL),
and plotting helper functions for Reliability Diagrams.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from pathlib import Path


def compute_calibration_metrics(
    alpha: torch.Tensor,
    labels: torch.Tensor,
    num_bins: int = 10,
) -> Dict[str, float]:
    """
    Computes Expected Calibration Error (ECE) and Negative Log-Likelihood (NLL)
    for Evidential outputs.

    Args:
        alpha: (N, C) Dirichlet concentration parameters
        labels: (N,) ground truth class labels
        num_bins: Number of confidence bins (default: 10)

    Returns:
        Dict with keys:
            'ece': Expected Calibration Error
            'nll': Negative Log-Likelihood under EDL
            'avg_confidence': Average prediction confidence
            'accuracy': Average prediction accuracy
    """
    # 1. Expected probabilities under Dirichlet: p_c = alpha_c / S
    strength = alpha.sum(dim=-1, keepdim=True)  # (N, 1)
    probs = alpha / strength  # (N, C)
    
    # Confidences and predictions
    confidences, predictions = torch.max(probs, dim=-1)
    
    confidences = confidences.cpu().numpy()
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()
    
    # 2. Expected Calibration Error (ECE)
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    
    bin_accuracies = []
    bin_confidences = []
    bin_sizes = []
    
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Mask for samples falling into this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(predictions[in_bin] == labels[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            
            ece += prop_in_bin * np.abs(accuracy_in_bin - avg_confidence_in_bin)
            
            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(avg_confidence_in_bin)
            bin_sizes.append(prop_in_bin)
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)
            bin_sizes.append(0.0)

    # 3. Negative Log-Likelihood (NLL)
    # Under EDL, NLL is Type-II MLE: NLL = sum_c y_c * (digamma(S) - digamma(alpha_c))
    num_classes = alpha.size(-1)
    y_onehot = torch.eye(num_classes, device=alpha.device)[labels] # (N, C)
    
    # digamma values
    digamma_S = torch.digamma(strength) # (N, 1)
    digamma_alpha = torch.digamma(alpha) # (N, C)
    nll_per_sample = (y_onehot * (digamma_S - digamma_alpha)).sum(dim=-1)
    nll = nll_per_sample.mean().item()

    accuracy = np.mean(predictions == labels)
    avg_confidence = np.mean(confidences)

    return {
        "ece": float(ece),
        "nll": float(nll),
        "accuracy": float(accuracy),
        "avg_confidence": float(avg_confidence),
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_sizes": bin_sizes,
    }


def plot_reliability_diagram(
    metrics: Dict,
    save_path: str,
    dataset_name: str = "MELD",
) -> None:
    """
    Plots a highly polished academic Reliability Diagram.
    """
    bin_accuracies = metrics["bin_accuracies"]
    bin_confidences = metrics["bin_confidences"]
    ece = metrics["ece"]
    
    num_bins = len(bin_accuracies)
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_centers = 0.5 * (bin_boundaries[:-1] + bin_boundaries[1:])

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    
    # Actual calibration bars
    ax.bar(
        bin_centers,
        bin_accuracies,
        width=1.0 / num_bins,
        color="royalblue",
        edgecolor="black",
        alpha=0.7,
        label="Evidential Model (EDL)",
    )
    
    # Under/Over confidence shading
    for center, acc, conf in zip(bin_centers, bin_accuracies, bin_confidences):
        if conf > 0:
            ax.plot([center, center], [acc, conf], color="red", linestyle="-", linewidth=1.5)
            
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("Confidence", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"Reliability Diagram on {dataset_name}\n(ECE: {ece:.4f})", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


class TemperatureScaler(torch.nn.Module):
    """
    Fits a temperature parameter T > 0 on a validation set's evidential logits
    to calibrate the Dirichlet parameters and minimize Expected Calibration Error (ECE).
    """

    def __init__(self):
        super().__init__()
        self.temperature = torch.nn.Parameter(torch.ones(1) * 1.5)  # Start slightly over 1.0

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # Scale raw logits by temperature
        scaled_logits = logits / self.temperature
        # Apply Softplus to get non-negative evidence, then alpha = evidence + 1
        evidence = torch.nn.functional.softplus(scaled_logits)
        alpha = evidence + 1.0
        return alpha

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor, lr: float = 0.01, max_iter: int = 200) -> float:
        """
        Fit the temperature parameter T using L-BFGS to minimize the evidential NLL.
        """
        device = val_logits.device
        val_logits = val_logits.detach().clone()
        val_labels = val_labels.detach().clone()
        num_classes = val_logits.size(-1)
        y_onehot = torch.eye(num_classes, device=device)[val_labels]

        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            alpha = self.forward(val_logits)
            strength = alpha.sum(dim=-1, keepdim=True)
            digamma_S = torch.digamma(strength)
            digamma_alpha = torch.digamma(alpha)
            # Evidential NLL loss (digamma form)
            loss = (y_onehot * (digamma_S - digamma_alpha)).sum(dim=-1).mean()
            loss.backward()
            return loss

        optimizer.step(eval_loss)

        # Enforce positive temperature
        with torch.no_grad():
            self.temperature.clamp_(min=1e-4)

        return float(self.temperature.item())


def plot_reliability_before_after(
    metrics_before: Dict,
    metrics_after: Dict,
    save_path: str,
    dataset_name: str = "MELD",
) -> None:
    """
    Plots a side-by-side reliability diagram comparison before and after calibration.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), dpi=300)

    # Plot 1: Before
    bin_accuracies_b = metrics_before["bin_accuracies"]
    bin_confidences_b = metrics_before["bin_confidences"]
    ece_b = metrics_before["ece"]
    num_bins = len(bin_accuracies_b)
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_centers = 0.5 * (bin_boundaries[:-1] + bin_boundaries[1:])

    ax1.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    ax1.bar(
        bin_centers,
        bin_accuracies_b,
        width=1.0 / num_bins,
        color="salmon",
        edgecolor="black",
        alpha=0.7,
        label="Uncalibrated EDL",
    )
    for center, acc, conf in zip(bin_centers, bin_accuracies_b, bin_confidences_b):
        if conf > 0:
            ax1.plot([center, center], [acc, conf], color="red", linestyle="-", linewidth=1.5)
    
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_xlabel("Confidence", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_title(f"Before Calibration (ECE: {ece_b:.4f})", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Plot 2: After
    bin_accuracies_a = metrics_after["bin_accuracies"]
    bin_confidences_a = metrics_after["bin_confidences"]
    ece_a = metrics_after["ece"]

    ax2.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    ax2.bar(
        bin_centers,
        bin_accuracies_a,
        width=1.0 / num_bins,
        color="royalblue",
        edgecolor="black",
        alpha=0.7,
        label="Calibrated EDL",
    )
    for center, acc, conf in zip(bin_centers, bin_accuracies_a, bin_confidences_a):
        if conf > 0:
            ax2.plot([center, center], [acc, conf], color="red", linestyle="-", linewidth=1.5)

    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.set_xlabel("Confidence", fontsize=11)
    ax2.set_title(f"After Calibration (ECE: {ece_a:.4f})", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(f"Reliability Diagram on {dataset_name} — Before vs After Temperature Scaling", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

