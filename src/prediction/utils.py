"""
Prediction subsystem for the Football Analytics Study.

A scientifically defensible forecasting system that extracts predictive
information from historical football matches, produces calibrated probability
distributions, and demonstrates forecasting performance through rigorous
out-of-sample (chronological) testing.

Guiding principle: scientific validity > predictive reliability >
interpretability > computational efficiency > complexity.

No bookmaker odds are ever used as predictors. No look-ahead leakage is
permitted. Validation uses expanding chronological windows only.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# Default random seed for reproducible results.
DEFAULT_SEED = 42

# Match outcome labels and their canonical integer encoding.
OUTCOME_LABELS = ("H", "D", "A")  # home win, draw, away win
OUTCOME_TO_INDEX = {label: i for i, label in enumerate(OUTCOME_LABELS)}
INDEX_TO_OUTCOME = {i: label for i, label in enumerate(OUTCOME_LABELS)}

# Numeric goal columns.
HOME_GOALS_COL = "FTHG"
AWAY_GOALS_COL = "FTAG"
OUTCOME_COL = "FTR"

# Derived market thresholds (total goals).
OVER_UNDER_THRESHOLDS = (1.5, 2.5, 3.5)

# Scorelines reported explicitly (home goals, away goals).
REPORTED_SCORELINES = [
    (0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2),
    (2, 1), (1, 2), (2, 2), (3, 0), (0, 3),
]

# Maximum goals modelled in the scoreline distribution tail.
MAX_GOALS = 8


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Set NumPy random seed for reproducible results."""
    np.random.seed(seed)


def outcome_to_index(outcome: str) -> Optional[int]:
    """Map a match outcome label ('H'/'D'/'A') to its integer index."""
    return OUTCOME_TO_INDEX.get(outcome)


def index_to_outcome(idx: int) -> Optional[str]:
    """Map an integer index to its match outcome label ('H'/'D'/'A')."""
    return INDEX_TO_OUTCOME.get(idx)


def encode_outcomes(series) -> np.ndarray:
    """Encode an outcome series into integer labels (0=H, 1=D, 2=A).

    Unknown or missing values are encoded as -1.
    """
    import polars as pl

    if isinstance(series, pl.Series):
        values = series.to_list()
    else:
        values = list(series)
    return np.array([OUTCOME_TO_INDEX.get(v, -1) for v in values], dtype=np.int64)


def points_from_outcome(outcome: str) -> int:
    """Return points earned (3/1/0) for a result from a single team's view."""
    return {"W": 3, "D": 1, "L": 0}.get(outcome, 0)


def outcome_from_goals(home_goals, away_goals) -> str:
    """Return the match outcome label from full-time goals."""
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def _jsonify(obj):
    """Recursively convert Polars/NumPy/Python types to JSON-serializable types."""
    import polars as pl

    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, tuple):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, float, str)) or obj is None:
        return obj
    if isinstance(obj, (pl.DataFrame, pl.Series)):
        return _jsonify(obj.to_dicts())
    return str(obj)


def validate_probabilities(probs, name: str = "probabilities", tol: float = 1e-6) -> None:
    """Validate that a 1-D array of probabilities is finite and in [0, 1]."""
    arr = np.asarray(probs, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    if np.any(arr < -tol) or np.any(arr > 1.0 + tol):
        raise ValueError(f"{name} contains values outside [0, 1]")
    return None


def validate_outcome_distribution(probs, name: str = "outcome distribution") -> np.ndarray:
    """Validate and normalize a 1X2 probability vector so it sums to 1.

    Returns a normalized copy. Raises if the input is invalid or cannot be
    normalized (e.g. all zeros).
    """
    arr = np.asarray(probs, dtype=np.float64).copy()
    validate_probabilities(arr, name)
    total = float(arr.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"{name} sums to {total}; cannot normalize")
    if abs(total - 1.0) > 1e-9:
        arr = arr / total
    return arr


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    ex = np.exp(x)
    s = ex.sum()
    if s <= 0 or not np.isfinite(s):
        return np.full_like(x, 1.0 / x.size)
    return ex / s
