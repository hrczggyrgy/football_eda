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
from scipy.optimize import minimize_scalar

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
    model = LogisticRegression(C=np.inf, solver="lbfgs")
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


# =============================================================================
# Temperature Scaling Calibration
# =============================================================================

class TemperatureScalingCalibrator:
    """Temperature scaling calibration for multiclass probabilities.
    
    Adjusts logits with a single fitted parameter T:
        calibrated_logits = logits / T
    
    Preserves argmax predictions, naturally maintains multiclass normalization.
    Cannot fix excessive draw argmax by itself, but improves probability calibration.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.temperature: Optional[float] = None
        self.fitted = False
    
    def fit(self, logits: np.ndarray, outcomes: np.ndarray) -> "TemperatureScalingCalibrator":
        """Fit temperature by minimizing NLL on validation set."""
        
        def nll(T):
            T = max(T, 1e-3)
            scaled = logits / T
            # Softmax
            scaled = scaled - scaled.max(axis=1, keepdims=True)
            exp_scaled = np.exp(scaled)
            probs = exp_scaled / exp_scaled.sum(axis=1, keepdims=True)
            # NLL
            n = probs.shape[0]
            y_onehot = np.zeros_like(probs)
            valid = (outcomes >= 0) & (outcomes < 3)
            y_onehot[valid, outcomes[valid]] = 1.0
            log_probs = np.log(np.clip(probs, 1e-12, 1.0))
            return -np.mean(np.sum(y_onehot * log_probs, axis=1)[valid])
        
        result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
        self.temperature = float(result.x)
        self.fitted = True
        return self
    
    def transform(self, logits: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("TemperatureScalingCalibrator must be fitted before transform.")
        T = max(self.temperature, 1e-3)
        scaled = logits / T
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        exp_scaled = np.exp(scaled)
        return exp_scaled / exp_scaled.sum(axis=1, keepdims=True)
    
    def transform_probs(self, probs: np.ndarray) -> np.ndarray:
        """Transform probabilities by converting to logits, scaling, and back."""
        logits = np.log(np.clip(probs, 1e-12, 1 - 1e-12))
        return self.transform(logits)


# =============================================================================
# Isotonic Calibration (Multiclass)
# =============================================================================

class MulticlassIsotonicCalibrator:
    """Isotonic regression calibration for multiclass probabilities.
    
    Fits one isotonic regressor per class (one-vs-rest).
    More flexible than temperature scaling but can overfit on small samples.
    """
    
    def __init__(self, n_bins: int = 100):
        self.n_bins = n_bins
        self.calibrators: List[object] = []
        self.fitted = False
    
    def fit(self, probs: np.ndarray, outcomes: np.ndarray) -> "MulticlassIsotonicCalibrator":
        from sklearn.isotonic import IsotonicRegression
        self.calibrators = []
        for k in range(3):
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(probs[:, k], (outcomes == k).astype(np.float64))
            self.calibrators.append(iso)
        self.fitted = True
        return self
    
    def transform(self, probs: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("MulticlassIsotonicCalibrator must be fitted before transform.")
        cal_probs = np.column_stack([
            self.calibrators[k].predict(probs[:, k]) for k in range(3)
        ])
        cal_probs = cal_probs / cal_probs.sum(axis=1, keepdims=True)
        return cal_probs


# =============================================================================
# Ensemble Calibrator (Temperature + Isotonic)
# =============================================================================

class EnsembleCalibrator:
    """Ensemble of temperature scaling and isotonic calibration.
    
    Uses temperature scaling as primary (more robust), with isotonic
    as fallback for classes with sufficient samples.
    """
    
    def __init__(self, seed: int = 42, min_samples_isotonic: int = 500):
        self.seed = seed
        self.min_samples_isotonic = min_samples_isotonic
        self.temp_calibrator = TemperatureScalingCalibrator(seed)
        self.iso_calibrator = MulticlassIsotonicCalibrator()
        self.use_isotonic = [False, False, False]
        self.fitted = False
    
    def fit(self, probs: np.ndarray, outcomes: np.ndarray) -> "EnsembleCalibrator":
        # Always fit temperature scaling (most robust)
        logits = np.log(np.clip(probs, 1e-12, 1 - 1e-12))
        self.temp_calibrator.fit(logits, outcomes)
        
        # Fit isotonic for classes with enough samples
        n = probs.shape[0]
        for k in range(3):
            class_count = int((outcomes == k).sum())
            if class_count >= self.min_samples_isotonic:
                self.use_isotonic[k] = True
        
        if any(self.use_isotonic):
            self.iso_calibrator.fit(probs, outcomes)
        
        self.fitted = True
        return self
    
    def transform(self, probs: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("EnsembleCalibrator must be fitted before transform.")
        
        # Temperature scaling for all
        logits = np.log(np.clip(probs, 1e-12, 1 - 1e-12))
        temp_probs = self.temp_calibrator.transform(logits)
        
        # Isotonic refinement where available
        if any(self.use_isotonic):
            iso_probs = self.iso_calibrator.transform(probs)
            for k in range(3):
                if self.use_isotonic[k]:
                    temp_probs[:, k] = iso_probs[:, k]
            temp_probs = temp_probs / temp_probs.sum(axis=1, keepdims=True)
        
        return temp_probs


# =============================================================================
# Calibration Factory
# =============================================================================

def get_calibrator(method: str, **kwargs) -> object:
    """Factory function for calibrators.
    
    Args:
        method: One of 'temperature', 'isotonic', 'ensemble', 'none'
    """
    if method == 'temperature':
        return TemperatureScalingCalibrator(**kwargs)
    elif method == 'isotonic':
        return MulticlassIsotonicCalibrator(**kwargs)
    elif method == 'ensemble':
        return EnsembleCalibrator(**kwargs)
    elif method == 'none':
        return None
    else:
        raise ValueError(f"Unknown calibration method: {method}")
