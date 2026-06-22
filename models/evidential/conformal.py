"""
Conformal Prediction Module for Evidential ERC
================================================
Implements LAC, APS, Randomized APS, and Federated Conformal Prediction (FCP).

References:
- LAC: Sadinle, Lei & Wasserman (2019). "Least Ambiguous Set-Valued Classifiers"
- APS: Romano, Sesia & Candès (2020). "Classification with Valid Adaptive Prediction Sets"
- FCP: Lu, Kalpathy-Cramer & Jolly (ICML 2023). "Federated Conformal Predictors"
- Angelopoulos & Bates (2021). "A Gentle Introduction to Conformal Prediction"

Exchangeability note (Advisor A2):
    Utterances within a dialogue are temporally dependent and NOT exchangeable.
    The exchangeable unit is the DIALOGUE (not individual utterances).
    Calibration and test sets must be split at the dialogue level.
    We cite Lu et al. 2023 (partial exchangeability) and verify empirical coverage.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ConformalResult:
    """Result container for conformal prediction evaluation."""
    coverage: float                  # Empirical coverage (should ≈ 1−α)
    avg_set_size: float              # Average prediction set size (smaller = better)
    median_set_size: float           # Median prediction set size
    set_sizes: np.ndarray            # Per-sample set sizes
    prediction_sets: List[List[int]] # Per-sample prediction sets
    per_class_coverage: Dict[int, float]  # Coverage broken down by true class
    coverage_deviation: float        # |empirical_coverage − (1−α)|
    quantile: float                  # Calibrated quantile threshold


class LACConformal:
    """
    Least Ambiguous set-valued Classifier (Sadinle et al. 2019).

    Nonconformity score: s(x, y) = 1 − p̂_y
    where p̂_y is the predicted probability for the true class y.

    Produces the SMALLEST prediction sets among threshold-based methods.
    """

    def __init__(self, alpha: float = 0.1):
        """
        Args:
            alpha: Miscoverage rate. Coverage target = 1 − alpha.
        """
        self.alpha = alpha
        self.quantile = None

    def calibrate(self, probs: np.ndarray, labels: np.ndarray) -> float:
        """
        Calibrate the conformal threshold on calibration (dev) data.

        Args:
            probs: (N_cal, C) predicted probabilities for calibration set
            labels: (N_cal,) true labels for calibration set

        Returns:
            Calibrated quantile threshold q̂
        """
        n = len(labels)
        # Nonconformity scores: 1 − p̂_{y_true}
        scores = 1.0 - probs[np.arange(n), labels]

        # Quantile with finite-sample correction: ceil((n+1)(1−α)) / n
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        level = min(level, 1.0)
        self.quantile = float(np.quantile(scores, level))
        return self.quantile

    def predict(self, probs: np.ndarray) -> List[List[int]]:
        """
        Generate prediction sets for test data.

        Args:
            probs: (N_test, C) predicted probabilities

        Returns:
            List of prediction sets (each is a list of class indices)
        """
        assert self.quantile is not None, "Must call calibrate() first"
        prediction_sets = []
        for i in range(len(probs)):
            # Include class c if 1 − p̂_c ≤ q̂  ⟺  p̂_c ≥ 1 − q̂
            pset = [c for c in range(probs.shape[1]) if probs[i, c] >= 1 - self.quantile]
            if len(pset) == 0:
                # Guarantee non-empty: include argmax
                pset = [int(np.argmax(probs[i]))]
            prediction_sets.append(pset)
        return prediction_sets

    def evaluate(self, probs: np.ndarray, labels: np.ndarray) -> ConformalResult:
        """Full evaluation: predict + compute metrics."""
        prediction_sets = self.predict(probs)
        return _compute_conformal_metrics(
            prediction_sets, labels, probs.shape[1], self.alpha, self.quantile
        )


class APSConformal:
    """
    Adaptive Prediction Sets (Romano, Sesia & Candès, 2020).

    Nonconformity score: cumulative probability up to and including the true class
    in the sorted probability order.

    Provides better per-class coverage than LAC but may produce larger sets.
    """

    def __init__(self, alpha: float = 0.1, randomized: bool = False):
        """
        Args:
            alpha: Miscoverage rate.
            randomized: If True, use randomized APS for tighter sets (exact coverage).
                        If False, use standard APS (conservative coverage).
        """
        self.alpha = alpha
        self.randomized = randomized
        self.quantile = None

    def _compute_scores(self, probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Compute APS nonconformity scores."""
        n, C = probs.shape
        scores = np.zeros(n)

        for i in range(n):
            # Sort classes by decreasing probability
            sorted_indices = np.argsort(-probs[i])
            cumsum = 0.0
            for rank, c in enumerate(sorted_indices):
                cumsum += probs[i, c]
                if c == labels[i]:
                    if self.randomized:
                        # Randomized: subtract a random fraction of the last class
                        u = np.random.uniform(0, 1)
                        scores[i] = cumsum - u * probs[i, c]
                    else:
                        scores[i] = cumsum
                    break
        return scores

    def calibrate(self, probs: np.ndarray, labels: np.ndarray) -> float:
        """Calibrate on dev data."""
        n = len(labels)
        scores = self._compute_scores(probs, labels)

        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        level = min(level, 1.0)
        self.quantile = float(np.quantile(scores, level))
        return self.quantile

    def predict(self, probs: np.ndarray) -> List[List[int]]:
        """Generate prediction sets for test data."""
        assert self.quantile is not None, "Must call calibrate() first"
        prediction_sets = []

        for i in range(len(probs)):
            sorted_indices = np.argsort(-probs[i])
            cumsum = 0.0
            pset = []
            for c in sorted_indices:
                cumsum += probs[i, c]
                pset.append(int(c))
                if cumsum >= self.quantile:
                    break
            if len(pset) == 0:
                pset = [int(np.argmax(probs[i]))]
            prediction_sets.append(pset)
        return prediction_sets

    def evaluate(self, probs: np.ndarray, labels: np.ndarray) -> ConformalResult:
        """Full evaluation."""
        prediction_sets = self.predict(probs)
        return _compute_conformal_metrics(
            prediction_sets, labels, probs.shape[1], self.alpha, self.quantile
        )


class FederatedConformalPredictor:
    """
    Federated Conformal Prediction (FCP) with distributed quantile computation.

    Instead of centralizing calibration data, each client computes local
    nonconformity scores and sends a histogram/quantile summary to the server.
    The server aggregates these to compute a global quantile threshold.

    This is TRUE federated conformal prediction (Lu et al., ICML 2023),
    not just "conformal on a federated model."

    Protocol:
    1. Server broadcasts current model to all clients.
    2. Each client k computes nonconformity scores on local calibration data.
    3. Each client sends: (sorted_scores_k, n_k) to server.
       - For privacy, can send histogram bins instead of raw scores.
    4. Server computes weighted quantile: q̂ = Quantile_{1-α}(∪_k scores_k, weights=n_k/N)
    5. Server broadcasts q̂ to all clients.
    6. Each client uses q̂ to construct prediction sets locally.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        method: str = "lac",
        privacy_mode: str = "scores",
    ):
        """
        Args:
            alpha: Miscoverage rate.
            method: "lac" or "aps" for nonconformity score type.
            privacy_mode: "scores" (send raw scores) or "histogram" (send binned counts).
        """
        self.alpha = alpha
        self.method = method
        self.privacy_mode = privacy_mode
        self.global_quantile = None

    def _compute_client_scores(
        self, probs: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """Compute nonconformity scores for one client's calibration data."""
        n = len(labels)
        if self.method == "lac":
            return 1.0 - probs[np.arange(n), labels]
        elif self.method == "aps":
            scores = np.zeros(n)
            for i in range(n):
                sorted_indices = np.argsort(-probs[i])
                cumsum = 0.0
                for c in sorted_indices:
                    cumsum += probs[i, c]
                    if c == labels[i]:
                        scores[i] = cumsum
                        break
            return scores
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def calibrate_federated(
        self,
        client_probs: List[np.ndarray],
        client_labels: List[np.ndarray],
        client_weights: Optional[List[float]] = None,
    ) -> float:
        """
        Distributed quantile calibration across K clients.

        Args:
            client_probs: List of K arrays, each (n_k, C)
            client_labels: List of K arrays, each (n_k,)
            client_weights: Optional aggregation weights (e.g., EAFA weights).
                           If None, weight proportional to n_k.

        Returns:
            Global quantile threshold q̂
        """
        all_scores = []
        all_sizes = []

        for k in range(len(client_probs)):
            scores_k = self._compute_client_scores(client_probs[k], client_labels[k])
            all_scores.append(scores_k)
            all_sizes.append(len(scores_k))

        total_n = sum(all_sizes)

        if client_weights is None:
            # Default: weight by dataset size (FedAvg-style)
            client_weights = [n_k / total_n for n_k in all_sizes]

        if self.privacy_mode == "scores":
            # Concatenate all scores and compute weighted quantile
            # Weight each score by its client's weight / n_k
            weighted_scores = []
            score_weights = []
            for k, scores_k in enumerate(all_scores):
                weighted_scores.extend(scores_k.tolist())
                # Each score from client k gets weight w_k / n_k
                w_per_score = client_weights[k] / len(scores_k)
                score_weights.extend([w_per_score] * len(scores_k))

            weighted_scores = np.array(weighted_scores)
            score_weights = np.array(score_weights)
            score_weights /= score_weights.sum()  # Normalize

            # Weighted quantile
            sorted_idx = np.argsort(weighted_scores)
            sorted_scores = weighted_scores[sorted_idx]
            sorted_weights = score_weights[sorted_idx]
            cumulative_weights = np.cumsum(sorted_weights)

            # Find the quantile level with finite-sample correction
            level = (1 - self.alpha) * (1 + 1 / total_n)
            level = min(level, 1.0)

            quantile_idx = np.searchsorted(cumulative_weights, level)
            quantile_idx = min(quantile_idx, len(sorted_scores) - 1)
            self.global_quantile = float(sorted_scores[quantile_idx])

        elif self.privacy_mode == "histogram":
            # Privacy-preserving: each client sends histogram
            num_bins = 200
            bin_edges = np.linspace(0, 1, num_bins + 1)

            # Aggregate histograms
            global_hist = np.zeros(num_bins)
            for k, scores_k in enumerate(all_scores):
                hist_k, _ = np.histogram(scores_k, bins=bin_edges)
                global_hist += hist_k * client_weights[k]

            global_hist /= global_hist.sum()
            cumulative = np.cumsum(global_hist)
            level = (1 - self.alpha) * (1 + 1 / total_n)
            level = min(level, 1.0)

            bin_idx = np.searchsorted(cumulative, level)
            bin_idx = min(bin_idx, num_bins - 1)
            self.global_quantile = float(bin_edges[bin_idx + 1])

        return self.global_quantile

    def predict(self, probs: np.ndarray) -> List[List[int]]:
        """Generate prediction sets using the global quantile."""
        assert self.global_quantile is not None, "Must call calibrate_federated() first"

        if self.method == "lac":
            prediction_sets = []
            for i in range(len(probs)):
                pset = [c for c in range(probs.shape[1])
                        if probs[i, c] >= 1 - self.global_quantile]
                if not pset:
                    pset = [int(np.argmax(probs[i]))]
                prediction_sets.append(pset)
            return prediction_sets

        elif self.method == "aps":
            prediction_sets = []
            for i in range(len(probs)):
                sorted_indices = np.argsort(-probs[i])
                cumsum = 0.0
                pset = []
                for c in sorted_indices:
                    cumsum += probs[i, c]
                    pset.append(int(c))
                    if cumsum >= self.global_quantile:
                        break
                if not pset:
                    pset = [int(np.argmax(probs[i]))]
                prediction_sets.append(pset)
            return prediction_sets

    def evaluate(self, probs: np.ndarray, labels: np.ndarray) -> ConformalResult:
        """Full evaluation."""
        prediction_sets = self.predict(probs)
        return _compute_conformal_metrics(
            prediction_sets, labels, probs.shape[1], self.alpha, self.global_quantile
        )


# ============================================================
# Helper functions
# ============================================================

def _compute_conformal_metrics(
    prediction_sets: List[List[int]],
    labels: np.ndarray,
    num_classes: int,
    alpha: float,
    quantile: float,
) -> ConformalResult:
    """Compute comprehensive conformal prediction metrics."""
    n = len(labels)
    set_sizes = np.array([len(s) for s in prediction_sets])

    # Coverage: fraction of true labels contained in prediction sets
    covered = sum(1 for i, s in enumerate(prediction_sets) if labels[i] in s)
    coverage = covered / n

    # Per-class coverage
    per_class_coverage = {}
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            class_covered = sum(
                1 for i in range(n) if labels[i] == c and labels[i] in prediction_sets[i]
            )
            per_class_coverage[c] = class_covered / mask.sum()

    return ConformalResult(
        coverage=coverage,
        avg_set_size=float(set_sizes.mean()),
        median_set_size=float(np.median(set_sizes)),
        set_sizes=set_sizes,
        prediction_sets=prediction_sets,
        per_class_coverage=per_class_coverage,
        coverage_deviation=abs(coverage - (1 - alpha)),
        quantile=quantile,
    )
