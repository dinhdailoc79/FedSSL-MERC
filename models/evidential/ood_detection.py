"""
OOD (Out-of-Distribution) Detection Module for Evidential ERC
================================================================
Detects unseen speakers using EDL uncertainty as OOD signal.

Reference:
- Advisor feedback A4: Corrected ID/OOD split for IEMOCAP.

IEMOCAP session structure:
    Session 1: Speaker pair (F1, M1)  ← Train
    Session 2: Speaker pair (F2, M2)  ← Train
    Session 3: Speaker pair (F3, M3)  ← Train
    Session 4: Speaker pair (F4, M4)  ← Dev (standard split)
    Session 5: Speaker pair (F5, M5)  ← Test (standard split)

CORRECTED ID/OOD design (Advisor A4):
    ID  = Held-out utterances from speakers in train sessions (1-3)
    OOD = All utterances from Session 5 (unseen speakers)

    Previously WRONG: Using dev (Session 4) as ID was incorrect because
    Session 4 speakers are also unseen relative to train.

Metrics:
    AUROC: Area Under the ROC curve for binary classification ID vs OOD.
           Higher = better separation. 0.5 = random, 1.0 = perfect.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score, roc_curve


@dataclass
class OODResult:
    """Result container for OOD detection evaluation."""
    auroc: float                           # Area Under ROC curve
    score_name: str                        # Name of the uncertainty score used
    id_mean_score: float                   # Mean uncertainty on ID data
    ood_mean_score: float                  # Mean uncertainty on OOD data
    id_std_score: float                    # Std of uncertainty on ID data
    ood_std_score: float                   # Std of uncertainty on OOD data
    fpr_at_tpr95: float                    # FPR when TPR=95% (lower = better)
    fpr_values: Optional[np.ndarray] = None  # For ROC curve plotting
    tpr_values: Optional[np.ndarray] = None  # For ROC curve plotting


def compute_ood_auroc(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Compute AUROC for OOD detection.

    Convention: Higher uncertainty score → more likely OOD.
    Label: ID = 0 (negative), OOD = 1 (positive).

    Args:
        id_scores: (N_id,) uncertainty scores for in-distribution data
        ood_scores: (N_ood,) uncertainty scores for out-of-distribution data

    Returns:
        auroc: AUROC value
        fpr_at_tpr95: FPR when TPR ≥ 0.95
        fpr_values: FPR values for ROC curve
        tpr_values: TPR values for ROC curve
    """
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])

    auroc = roc_auc_score(labels, scores)
    fpr_values, tpr_values, _ = roc_curve(labels, scores)

    # FPR at TPR=95%
    idx_95 = np.searchsorted(tpr_values, 0.95)
    fpr_at_tpr95 = float(fpr_values[min(idx_95, len(fpr_values) - 1)])

    return auroc, fpr_at_tpr95, fpr_values, tpr_values


def evaluate_ood_detection(
    id_alpha: np.ndarray,
    ood_alpha: np.ndarray,
    num_classes: int,
) -> Dict[str, OODResult]:
    """
    Full OOD detection evaluation comparing multiple uncertainty signals.

    Args:
        id_alpha: (N_id, C) Dirichlet parameters for ID data
        ood_alpha: (N_ood, C) Dirichlet parameters for OOD data
        num_classes: Number of classes C

    Returns:
        Dict mapping score_name → OODResult
    """
    results = {}

    # Extract uncertainty scores for both ID and OOD
    score_extractors = {
        "vacuity_u": lambda alpha: num_classes / alpha.sum(axis=-1),
        "max_prob_inv": lambda alpha: 1.0 - (alpha / alpha.sum(axis=-1, keepdims=True)).max(axis=-1),
        "entropy": lambda alpha: _compute_entropy(alpha),
    }

    for name, extractor in score_extractors.items():
        id_scores = extractor(id_alpha)
        ood_scores = extractor(ood_alpha)

        auroc, fpr95, fpr_vals, tpr_vals = compute_ood_auroc(id_scores, ood_scores)

        results[name] = OODResult(
            auroc=auroc,
            score_name=name,
            id_mean_score=float(id_scores.mean()),
            ood_mean_score=float(ood_scores.mean()),
            id_std_score=float(id_scores.std()),
            ood_std_score=float(ood_scores.std()),
            fpr_at_tpr95=fpr95,
            fpr_values=fpr_vals,
            tpr_values=tpr_vals,
        )

    return results


def prepare_iemocap_id_ood_split(
    train_dialogues: list,
    test_dialogues: list,
    holdout_fraction: float = 0.15,
    seed: int = 42,
) -> Tuple[list, list]:
    """
    Prepare corrected ID/OOD split for IEMOCAP (Advisor A4).

    ID  = Random holdout utterances from train speakers (sessions 1-3).
    OOD = All utterances from test speakers (session 5).

    Args:
        train_dialogues: List of Dialogue objects from sessions 1-3
        test_dialogues: List of Dialogue objects from session 5
        holdout_fraction: Fraction of train dialogues to hold out as ID test
        seed: Random seed

    Returns:
        id_dialogues: Held-out dialogues from train speakers
        ood_dialogues: All dialogues from test (session 5)
    """
    rng = np.random.RandomState(seed)

    n_holdout = max(1, int(len(train_dialogues) * holdout_fraction))
    indices = rng.permutation(len(train_dialogues))
    holdout_indices = indices[:n_holdout]

    id_dialogues = [train_dialogues[i] for i in holdout_indices]
    ood_dialogues = test_dialogues  # Session 5 = unseen speakers

    return id_dialogues, ood_dialogues


def _compute_entropy(alpha: np.ndarray) -> np.ndarray:
    """Compute entropy of Dirichlet mean probabilities."""
    strength = alpha.sum(axis=-1, keepdims=True)
    probs = alpha / strength
    log_probs = np.log(np.clip(probs, 1e-10, 1.0))
    entropy = -np.sum(probs * log_probs, axis=-1)
    return entropy
