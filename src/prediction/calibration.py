"""
Probability calibration diagnostics.

Reliability diagrams, multiclass Brier scores, log loss, expected calibration
error (ECE), and calibration intercept/slope. All calibration fitting uses
training-period information or temporally valid out-of-fold predictions only -
never held-out test data.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from prediction.utils import OUTCOME_LABELS, validate_outcome_distribution


def brier_score_multiclass(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Multiclass Brier score: mean sum over classes of (p_ij - y_ij)^2.

    probs: (n, 3) probabilities for H/D/A.
    outcomes: (n,) integer labels 0/1/2.
    """
    probs = np.asarray(probs, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.int64)
    n = probs.shape[0]
    y_onehot = np.zeros_like(probs)
    valid = (outcomes >= 0) & (outcomes < probs.shape[1])
    y_onehot[valid, outcomes[valid]] = 1.0
    return float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))


def log_loss_multiclass(probs: np.ndarray, outcomes: np.ndarray,
                        eps: float = 1e-12) -> float:
    """Multiclass log loss (negative log-likelihood of the true class)."""
    probs = np.asarray(probs, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.int64)
    valid = (outcomes >= 0) & (outcomes < probs.shape[1])
    if not np.any(valid):
        return float("nan")
    p = np.clip(probs[valid, outcomes[valid]], eps, 1.0 - eps)
    return float(-np.mean(np.log(p)))


def expected_calibration_error(probs: np.ndarray, outcomes: np.ndarray,
                               n_bins: int = 10) -> Dict[str, float]:
    """Expected Calibration Error (ECE) for multiclass probabilities.

    Uses the standard binning on the maximum predicted probability (confidence)
    and compares average confidence to accuracy within each bin.

    Returns dict with ece, per-bin calibration, and bin counts.
    """
    probs = np.asarray(probs, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.int64)
    valid = (outcomes >= 0) & (outcomes < probs.shape[1]) & np.all(np.isfinite(probs), axis=1)
    probs = probs[valid]; outcomes = outcomes[valid]
    n = probs.shape[0]
    if n == 0:
        return {"ece": float("nan"), "bins": [], "n": 0}

    confidence = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    correct = (predicted == outcomes).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins = []
    for b in range(n_bins):
        lo = bin_edges[b]; hi = bin_edges[b + 1]
        in_bin = (confidence > lo) & (confidence <= hi) if b < n_bins - 1 else (confidence >= lo) & (confidence <= hi)
        if b == 0:
            in_bin = (confidence >= lo) & (confidence <= hi)
        bin_count = int(in_bin.sum())
        if bin_count == 0:
            bins.append({"bin": b, "lo": lo, "hi": hi, "count": 0,
                         "confidence": None, "accuracy": None, "gap": None})
            continue
        bin_conf = float(confidence[in_bin].mean())
        bin_acc = float(correct[in_bin].mean())
        gap = abs(bin_conf - bin_acc)
        ece += (bin_count / n) * gap
        bins.append({"bin": b, "lo": float(lo), "hi": float(hi), "count": bin_count,
                     "confidence": round(bin_conf, 4), "accuracy": round(bin_acc, 4),
                     "gap": round(gap, 4)})
    return {"ece": round(float(ece), 4), "bins": bins, "n": int(n)}


class ReliabilityDiagramData:
    """Container for reliability-diagram data per class."""

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    def per_class(self, probs: np.ndarray, outcomes: np.ndarray,
                  class_idx: int) -> List[dict]:
        """Reliability data for one class: bin by predicted probability of that class."""
        probs = np.asarray(probs, dtype=np.float64)
        outcomes = np.asarray(outcomes, dtype=np.int64)
        p = probs[:, class_idx]
        y = (outcomes == class_idx).astype(np.float64)
        valid = np.isfinite(p) & (outcomes >= 0)
        p = p[valid]; y = y[valid]
        n = p.shape[0]
        if n == 0:
            return []

        bin_edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        bins = []
        for b in range(self.n_bins):
            lo = bin_edges[b]; hi = bin_edges[b + 1]
            if b < self.n_bins - 1:
                in_bin = (p > lo) & (p <= hi)
            else:
                in_bin = (p >= lo) & (p <= hi)
            count = int(in_bin.sum())
            if count == 0:
                bins.append({"bin": b, "lo": float(lo), "hi": float(hi), "count": 0,
                             "mean_predicted": None, "observed_frequency": None})
                continue
            bins.append({
                "bin": b, "lo": float(lo), "hi": float(hi), "count": count,
                "mean_predicted": round(float(p[in_bin].mean()), 4),
                "observed_frequency": round(float(y[in_bin].mean()), 4),
            })
        return bins


def calibration_intercept_slope(probs: np.ndarray, outcomes: np.ndarray,
                                class_idx: int) -> Tuple[float, float]:
    """Calibration intercept and slope for one class via logistic regression.

    Regresses the binary outcome on the logit of the predicted probability:
        logit(y_hat_prob) -> outcome. Intercept ~0, slope ~1 indicates good
        calibration. Returns (intercept, slope). Requires sklearn.
    """
    from sklearn.linear_model import LogisticRegression
    probs = np.asarray(probs, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.int64)
    p = np.clip(probs[:, class_idx], 1e-6, 1 - 1e-6)
    y = (outcomes == class_idx).astype(np.float64)
    valid = np.isfinite(p) & (outcomes >= 0)
    p = p[valid]; y = y[valid]
    if p.shape[0] < 5 or len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    logit = np.log(p / (1.0 - p)).reshape(-1, 1)
    # Fit logistic regression of y on logit(p); intercept & slope of calibration.
    model = LogisticRegression(solver="lbfgs")
    model.fit(logit, y)
    # Map: P(y=1) = sigmoid(intercept + slope * logit(p))
    intercept = float(model.intercept_[0])
    slope = float(model.coef_[0][0])
    return intercept, slope


def brier_decomposition(probs: np.ndarray, outcomes: np.ndarray,
                        n_bins: int = 10) -> Dict[str, float]:
    """Brier score decomposition into reliability, resolution, uncertainty.

    Brier = reliability - resolution + uncertainty.
    """
    probs = np.asarray(probs, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.int64)
    valid = (outcomes >= 0) & (outcomes < probs.shape[1]) & np.all(np.isfinite(probs), axis=1)
    probs = probs[valid]; outcomes = outcomes[valid]
    n = probs.shape[0]
    if n == 0:
        return {"brier": float("nan"), "reliability": float("nan"),
                "resolution": float("nan"), "uncertainty": float("nan")}

    n_classes = probs.shape[1]
    y_onehot = np.zeros_like(probs)
    y_onehot[np.arange(n), outcomes] = 1.0
    brier = float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))

    # Bin by the predicted probability of the true class? Use max-prob binning
    # for multi-class reliability component approximation.
    confidence = probs.max(axis=1)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    overall_accuracy = float((probs.argmax(axis=1) == outcomes).mean())
    reliability = 0.0; resolution = 0.0
    for b in range(n_bins):
        lo = bin_edges[b]; hi = bin_edges[b + 1]
        if b < n_bins - 1:
            in_bin = (confidence > lo) & (confidence <= hi)
        else:
            in_bin = (confidence >= lo) & (confidence <= hi)
        count = int(in_bin.sum())
        if count == 0:
            continue
        bin_conf = float(confidence[in_bin].mean())
        bin_acc = float((probs[in_bin].argmax(axis=1) == outcomes[in_bin]).mean())
        reliability += (count / n) * (bin_conf - bin_acc) ** 2
        resolution += (count / n) * (bin_acc - overall_accuracy) ** 2
    uncertainty = float(overall_accuracy * (1.0 - overall_accuracy))
    return {
        "brier": round(brier, 5),
        "reliability": round(reliability, 5),
        "resolution": round(resolution, 5),
        "uncertainty": round(uncertainty, 5),
    }
