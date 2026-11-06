"""
Forecast evaluation metrics for outcomes and goals.

Proper scoring rules only: Brier, log loss, MAE, RMSE, Poisson deviance,
predictive log likelihood. Training performance is never reported as
forecasting performance.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.stats import poisson

from prediction.utils import encode_outcomes


def goal_mae(pred: np.ndarray, actual: np.ndarray) -> float:
    """Mean absolute error for goal predictions."""
    pred = np.asarray(pred, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    valid = np.isfinite(pred) & np.isfinite(actual)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.abs(pred[valid] - actual[actual.shape[0] and valid]))) if False else float(np.mean(np.abs(pred[valid] - actual[valid])))


def goal_rmse(pred: np.ndarray, actual: np.ndarray) -> float:
    """Root mean squared error for goal predictions."""
    pred = np.asarray(pred, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    valid = np.isfinite(pred) & np.isfinite(actual)
    if not np.any(valid):
        return float("nan")
    return float(np.sqrt(np.mean((pred[valid] - actual[valid]) ** 2)))


def poisson_deviance(actual: np.ndarray, lam: np.ndarray) -> float:
    """Poisson deviance: 2 * sum( lam - actual + actual*log(actual/lam) )."""
    actual = np.asarray(actual, dtype=np.float64)
    lam = np.asarray(lam, dtype=np.float64)
    lam = np.clip(lam, 1e-6, None)
    actual = np.clip(actual, 0, None)
    valid = np.isfinite(actual) & np.isfinite(lam)
    a = actual[valid]; l = lam[valid]
    term = np.zeros_like(a)
    mask = a > 0
    term[mask] = a[mask] * np.log(a[mask] / l[mask])
    dev = 2.0 * np.sum(l - a + term)
    return float(dev / a.shape[0])


def predictive_log_likelihood(actual: np.ndarray, lam: np.ndarray) -> float:
    """Average Poisson log-likelihood of observed goals given predicted lambda."""
    actual = np.asarray(actual, dtype=np.float64)
    lam = np.asarray(lam, dtype=np.float64)
    lam = np.clip(lam, 1e-6, None)
    valid = np.isfinite(actual) & np.isfinite(lam)
    if not np.any(valid):
        return float("nan")
    ll = poisson.logpmf(actual[valid], lam[valid])
    return float(np.mean(ll))


def ranked_probability_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Ranked Probability Score (RPS) for ordered outcome classes H < D < A.

    Treats the three outcomes as ordered (home win, draw, away win) and
    compares cumulative predicted vs observed probabilities.
    """
    probs = np.asarray(probs, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.int64)
    n = probs.shape[0]
    # Cumulative predicted probabilities (H, H+D).
    cum_p = np.cumsum(probs, axis=1)[:, :2]
    y_onehot = np.zeros((n, 2))
    valid = (outcomes >= 0) & (outcomes < 3)
    y_onehot[valid, 0] = (outcomes[valid] == 0).astype(float)  # H
    y_onehot[valid, 1] = np.isin(outcomes[valid], [0, 1]).astype(float)  # H or D
    rps = np.mean(np.sum((cum_p - y_onehot) ** 2, axis=1))
    return float(rps)


def outcome_metrics(probs: np.ndarray, outcomes: np.ndarray) -> dict:
    """Aggregate outcome-forecast metrics."""
    from prediction.calibration import (
        brier_score_multiclass, log_loss_multiclass,
        expected_calibration_error,
    )
    probs = np.asarray(probs, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.int64)
    valid = (outcomes >= 0) & (outcomes < 3) & np.all(np.isfinite(probs), axis=1)
    probs = probs[valid]; outcomes = outcomes[valid]
    if probs.shape[0] == 0:
        return {"brier": float("nan"), "log_loss": float("nan"),
                "rps": float("nan"), "ece": float("nan"), "accuracy": float("nan"),
                "n": 0}
    acc = float(np.mean(probs.argmax(axis=1) == outcomes))
    return {
        "brier": round(brier_score_multiclass(probs, outcomes), 5),
        "log_loss": round(log_loss_multiclass(probs, outcomes), 5),
        "rps": round(ranked_probability_score(probs, outcomes), 5),
        "ece": round(expected_calibration_error(probs, outcomes)["ece"], 5),
        "accuracy": round(acc, 5),
        "n": int(probs.shape[0]),
    }


def goal_metrics(home_lambda: np.ndarray, away_lambda: np.ndarray,
                 home_goals: np.ndarray, away_goals: np.ndarray) -> dict:
    """Aggregate goal-forecast metrics."""
    out = {
        "home_mae": round(goal_mae(home_lambda, home_goals), 5),
        "away_mae": round(goal_mae(away_lambda, away_goals), 5),
        "total_mae": round(goal_mae(home_lambda + away_lambda, home_goals + away_goals), 5),
        "home_rmse": round(goal_rmse(home_lambda, home_goals), 5),
        "away_rmse": round(goal_rmse(away_lambda, away_goals), 5),
        "poisson_deviance": round(poisson_deviance(home_goals, home_lambda), 5),
        "home_loglik": round(predictive_log_likelihood(home_goals, home_lambda), 5),
        "away_loglik": round(predictive_log_likelihood(away_goals, away_lambda), 5),
    }
    return out
