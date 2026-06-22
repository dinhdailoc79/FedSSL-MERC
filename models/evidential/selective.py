"""
Selective Prediction Module for Evidential ERC
=================================================
Implements risk-coverage analysis and AURC computation.

References:
- Geifman & El-Yaniv (2017). "Selective Classification for Deep Neural Networks"
- El-Yaniv & Wiener (2010). "On the Foundations of Noise-free Selective Classification"

Key idea:
    Instead of always making a prediction, the model can ABSTAIN (refuse to answer)
    on samples where it is uncertain. This is critical for safety-critical ERC
    deployments (healthcare, customer care).

    We evaluate using:
    - Risk-Coverage curve: risk (error rate) vs coverage (fraction of samples predicted)
    - AURC: Area Under the Risk-Coverage curve (lower = better)
    - Accuracy at fixed coverage levels (0.5, 0.8, 0.9)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

def _trapz(y, x=None):
    if hasattr(np, 'trapezoid'):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


@dataclass
class SelectiveResult:
    """Result container for selective prediction evaluation."""
    aurc: float                              # Area Under Risk-Coverage curve (lower = better)
    eaurc: float                             # Excess AURC = AURC - AURC* (risk-adjusted)
    accuracy_at_coverage: Dict[float, float] # Accuracy at given coverage levels
    coverages: np.ndarray                    # Coverage values for the curve
    risks: np.ndarray                        # Risk values for the curve
    confidence_name: str                     # Name of the confidence score used
    overall_accuracy: float                  # Accuracy without selection (full coverage)


def compute_risk_coverage_curve(
    predictions: np.ndarray,
    labels: np.ndarray,
    confidences: np.ndarray,
    n_points: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the risk-coverage curve.

    Sort samples by decreasing confidence. At each coverage level,
    risk = error rate on the most confident samples.

    Args:
        predictions: (N,) predicted class indices
        labels: (N,) true labels
        confidences: (N,) confidence scores (higher = more confident)
        n_points: Number of points on the curve

    Returns:
        coverages: (n_points,) coverage levels from 0 to 1
        risks: (n_points,) risk (error rate) at each coverage level
    """
    n = len(predictions)
    correct = (predictions == labels).astype(float)

    # Sort by decreasing confidence
    sorted_idx = np.argsort(-confidences)
    sorted_correct = correct[sorted_idx]

    coverages = np.linspace(1 / n, 1.0, n_points)
    risks = np.zeros(n_points)

    for i, cov in enumerate(coverages):
        n_selected = max(1, int(np.ceil(cov * n)))
        n_selected = min(n_selected, n)
        selected_correct = sorted_correct[:n_selected]
        risks[i] = 1.0 - selected_correct.mean()  # Risk = 1 - accuracy

    return coverages, risks


def compute_aurc(
    predictions: np.ndarray,
    labels: np.ndarray,
    confidences: np.ndarray,
) -> float:
    """
    Compute Area Under the Risk-Coverage curve using trapezoidal integration.

    Lower AURC = better selective prediction (model correctly ranks its confidence).

    Args:
        predictions: (N,) predicted class indices
        labels: (N,) true labels
        confidences: (N,) confidence scores

    Returns:
        AURC value (float, lower is better)
    """
    n = len(predictions)
    correct = (predictions == labels).astype(float)

    # Sort by decreasing confidence
    sorted_idx = np.argsort(-confidences)
    sorted_correct = correct[sorted_idx]

    # Compute cumulative risk at each threshold
    cumulative_correct = np.cumsum(sorted_correct)
    coverages = np.arange(1, n + 1) / n
    risks = 1.0 - cumulative_correct / np.arange(1, n + 1)

    # Trapezoidal integration
    aurc = _trapz(risks, coverages)
    return float(aurc)


def compute_eaurc(
    predictions: np.ndarray,
    labels: np.ndarray,
    confidences: np.ndarray,
) -> float:
    """
    Compute Excess AURC = AURC − AURC* where AURC* is the optimal AURC
    (achieved by a perfect confidence ranking).

    EAURC isolates the quality of the confidence ranking from the model's accuracy.
    """
    accuracy = (predictions == labels).mean()
    aurc = compute_aurc(predictions, labels, confidences)

    # Optimal AURC: all errors are ranked last
    # AURC* = (1 - accuracy) * accuracy / 2  (for binary, approximate for multiclass)
    # More precise: integrate the step function
    n = len(predictions)
    n_correct = int(accuracy * n)
    n_wrong = n - n_correct

    if n_wrong == 0:
        aurc_star = 0.0
    else:
        # Optimal: first n_correct samples are correct, then n_wrong are wrong
        optimal_risks = np.zeros(n)
        optimal_risks[n_correct:] = np.cumsum(np.ones(n_wrong)) / np.arange(
            n_correct + 1, n + 1
        )
        coverages = np.arange(1, n + 1) / n
        aurc_star = float(_trapz(optimal_risks, coverages))

    return float(aurc - aurc_star)


def compute_accuracy_at_coverage(
    predictions: np.ndarray,
    labels: np.ndarray,
    confidences: np.ndarray,
    coverage_levels: List[float] = [0.5, 0.8, 0.9],
) -> Dict[float, float]:
    """
    Compute accuracy when only the top-k% most confident predictions are kept.

    Args:
        predictions: (N,)
        labels: (N,)
        confidences: (N,)
        coverage_levels: List of coverage fractions to evaluate

    Returns:
        Dict mapping coverage_level → accuracy at that level
    """
    n = len(predictions)
    correct = (predictions == labels).astype(float)
    sorted_idx = np.argsort(-confidences)
    sorted_correct = correct[sorted_idx]

    results = {}
    for cov in coverage_levels:
        n_selected = max(1, int(np.ceil(cov * n)))
        n_selected = min(n_selected, n)
        results[cov] = float(sorted_correct[:n_selected].mean())

    return results


def evaluate_selective_prediction(
    predictions: np.ndarray,
    labels: np.ndarray,
    confidence_scores: Dict[str, np.ndarray],
    coverage_levels: List[float] = [0.5, 0.8, 0.9],
    n_curve_points: int = 200,
) -> Dict[str, SelectiveResult]:
    """
    Full selective prediction evaluation comparing multiple confidence scores.

    Args:
        predictions: (N,) predicted class indices (from EDL belief argmax)
        labels: (N,) true labels
        confidence_scores: Dict mapping score_name → (N,) confidence values
            Expected keys: "vacuity_1mu" (1−u), "max_prob", "neg_entropy"
        coverage_levels: Coverage levels for accuracy@coverage
        n_curve_points: Number of points on risk-coverage curve

    Returns:
        Dict mapping score_name → SelectiveResult
    """
    overall_accuracy = float((predictions == labels).mean())
    results = {}

    for name, confidences in confidence_scores.items():
        aurc = compute_aurc(predictions, labels, confidences)
        eaurc = compute_eaurc(predictions, labels, confidences)
        acc_at_cov = compute_accuracy_at_coverage(
            predictions, labels, confidences, coverage_levels
        )
        coverages, risks = compute_risk_coverage_curve(
            predictions, labels, confidences, n_curve_points
        )

        results[name] = SelectiveResult(
            aurc=aurc,
            eaurc=eaurc,
            accuracy_at_coverage=acc_at_cov,
            coverages=coverages,
            risks=risks,
            confidence_name=name,
            overall_accuracy=overall_accuracy,
        )

    return results


def extract_confidence_scores(
    alpha: np.ndarray,
    num_classes: int,
) -> Dict[str, np.ndarray]:
    """
    Extract three types of confidence scores from EDL Dirichlet parameters.

    Args:
        alpha: (N, C) Dirichlet concentration parameters
        num_classes: Number of classes C

    Returns:
        Dict with three confidence scores:
            "vacuity_1mu": 1 − u = 1 − C/S (evidential confidence)
            "max_prob": max_c p̂_c (maximum softmax probability)
            "neg_entropy": −H(p̂) = −Σ p̂_c log p̂_c (negative entropy)
    """
    # Dirichlet mean probabilities: p̂_c = α_c / S
    strength = alpha.sum(axis=-1, keepdims=True)  # (N, 1)
    probs = alpha / strength  # (N, C)

    # 1. Evidential confidence: 1 − u = 1 − C/S
    uncertainty = num_classes / strength.squeeze(-1)  # (N,)
    vacuity_conf = 1.0 - uncertainty

    # 2. Maximum probability
    max_prob = probs.max(axis=-1)

    # 3. Negative entropy: −Σ p_c log p_c (higher = more confident)
    log_probs = np.log(np.clip(probs, 1e-10, 1.0))
    neg_entropy = -np.sum(probs * log_probs, axis=-1)
    # Normalize to [0, 1] range for fair comparison
    max_entropy = np.log(num_classes)
    neg_entropy = 1.0 - neg_entropy / max_entropy  # Flip: higher = more confident

    return {
        "vacuity_1mu": vacuity_conf,
        "max_prob": max_prob,
        "neg_entropy": neg_entropy,
    }
