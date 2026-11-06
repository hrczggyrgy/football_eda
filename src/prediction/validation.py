"""
Chronological expanding-window validation.

No random train/test splits. Training always precedes testing. Because every
engineered feature is strictly lagged (computed only from matches strictly
before the target fixture), pre-computing the pre-match feature frame once on
the full dataset is mathematically equivalent to a walk-forward simulation:
match t's features never include match t or any later match. After a test
fixture is played, its result naturally enters the feature state of subsequent
fixtures via the lagged rolling/expanding windows.

This is fast (O(n) feature build) and avoids the O(n^2) cost of rebuilding
features per fixture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import polars as pl

from prediction.utils import encode_outcomes
from prediction.calibration import (
    brier_score_multiclass, log_loss_multiclass,
    expected_calibration_error,
)
from prediction.evaluation import outcome_metrics, goal_metrics
from prediction.features import build_pre_match_features, pair_match_features


@dataclass
class FoldResult:
    """Results for a single chronological validation fold."""
    train_seasons: List[str]
    test_season: str
    n_train: int
    n_test: int
    train_cutoff_date: str
    metrics: Dict = field(default_factory=dict)
    predictions: Optional[np.ndarray] = None
    outcomes: Optional[np.ndarray] = None
    home_lambda: Optional[np.ndarray] = None
    away_lambda: Optional[np.ndarray] = None
    home_goals: Optional[np.ndarray] = None
    away_goals: Optional[np.ndarray] = None


@dataclass
class ValidationResult:
    """Aggregated results across all chronological folds."""
    folds: List[FoldResult] = field(default_factory=list)
    seasons: List[str] = field(default_factory=list)
    n_folds: int = 0
    aggregate: Dict = field(default_factory=dict)
    pooled: Dict = field(default_factory=dict)  # concatenated out-of-fold predictions


def chronological_folds(seasons: List[str],
                        min_train_seasons: int = 2) -> List[Tuple[List[str], str]]:
    """Generate expanding-window (train, test) season pairs.

    For seasons [s1..sN], folds are ([s1..s{k}], s{k+1}) for
    k in [min_train_seasons .. N-1]. Requires at least min_train_seasons+1
    seasons to produce a single fold.
    """
    seasons = sorted(seasons)
    folds = []
    for k in range(min_train_seasons, len(seasons)):
        train = seasons[:k]
        test = seasons[k]
        folds.append((train, test))
    return folds


def _fit_model(model, train_tm: pl.DataFrame, train_df: pl.DataFrame):
    """Fit a model, handling different fit signatures."""
    try:
        model.fit(train_tm, train_df)
    except TypeError:
        model.fit(train_df)
    return model


def run_expanding_validation(
    matches: pl.DataFrame,
    team_matches: pl.DataFrame,
    model_factory,
    seasons: Optional[List[str]] = None,
    min_train_seasons: int = 2,
    seed: int = 42,
) -> ValidationResult:
    """Run expanding-window chronological validation for one model.

    Features are pre-computed once (strictly lagged), then split into train/test
    by season. This is equivalent to walk-forward because no feature ever
    looks at the current or a future match.
    """
    if seasons is None:
        seasons = sorted(matches.select(pl.col("Season").unique()).to_series().to_list())

    folds = chronological_folds(seasons, min_train_seasons=min_train_seasons)
    result = ValidationResult(seasons=seasons, n_folds=len(folds))

    # Pre-compute pre-match features once on the FULL dataset. Strictly lagged,
    # so match t's features use only history before t.
    features = build_pre_match_features(team_matches)
    paired = pair_match_features(features)

    # Identify feature/predict columns.
    target_cols = {"Div", "Season", "Date", "HomeTeam", "AwayTeam",
                   "FTHG", "FTAG", "FTR"}

    for train_seasons, test_season in folds:
        train_df = matches.filter(pl.col("Season").is_in(train_seasons))
        train_tm = team_matches.filter(pl.col("Season").is_in(train_seasons))
        test_df = matches.filter(pl.col("Season") == test_season)
        train_cutoff = str(train_df.select(pl.col("Date").max()).to_series()[0])

        train_paired = paired.filter(pl.col("Season").is_in(train_seasons))
        test_paired = paired.filter(pl.col("Season") == test_season)

        fold = FoldResult(
            train_seasons=train_seasons,
            test_season=test_season,
            n_train=int(train_paired.height),
            n_test=int(test_paired.height),
            train_cutoff_date=train_cutoff,
        )

        if train_paired.height == 0 or test_paired.height == 0:
            result.folds.append(fold)
            continue

        model = model_factory()
        model = _fit_model(model, train_tm, train_paired)

        res = model.predict(test_paired)
        probs = res.outcome_probs
        outcomes = encode_outcomes(test_paired.select(pl.col("FTR")).to_series())

        valid = outcomes >= 0
        fold.metrics = outcome_metrics(probs, outcomes)

        # Goal metrics if lambdas available.
        if res.home_lambda is not None and res.away_lambda is not None:
            hg = test_paired.select(pl.col("FTHG")).to_numpy().ravel()
            ag = test_paired.select(pl.col("FTAG")).to_numpy().ravel()
            fold.metrics.update(goal_metrics(res.home_lambda, res.away_lambda, hg, ag))

        fold.predictions = probs
        fold.outcomes = outcomes
        fold.home_lambda = res.home_lambda
        fold.away_lambda = res.away_lambda
        fold.home_goals = hg if res.home_lambda is not None else None
        fold.away_goals = ag if res.away_lambda is not None else None
        result.folds.append(fold)

    # Concatenate out-of-fold predictions across folds for diagnostics/figures.
    all_probs = [f.predictions for f in result.folds if f.predictions is not None]
    all_outcomes = [f.outcomes for f in result.folds if f.outcomes is not None]
    all_hl = [f.home_lambda for f in result.folds if f.home_lambda is not None]
    all_al = [f.away_lambda for f in result.folds if f.away_lambda is not None]
    all_hg = [f.home_goals for f in result.folds if f.home_goals is not None]
    all_ag = [f.away_goals for f in result.folds if f.away_goals is not None]
    pooled = {}
    if all_probs:
        pooled["probs"] = np.vstack(all_probs)
        pooled["outcomes"] = np.concatenate(all_outcomes)
    if all_hl:
        pooled["home_lambda"] = np.concatenate(all_hl)
        pooled["away_lambda"] = np.concatenate(all_al)
        pooled["home_goals"] = np.concatenate(all_hg)
        pooled["away_goals"] = np.concatenate(all_ag)
    result.pooled = pooled

    result.aggregate = _aggregate_folds(result.folds)
    return result


def _aggregate_folds(folds: List[FoldResult]) -> Dict:
    keys = ["brier", "log_loss", "rps", "ece", "accuracy",
            "home_mae", "away_mae", "total_mae", "home_rmse", "away_rmse",
            "poisson_deviance", "home_loglik", "away_loglik"]
    agg = {}
    total_n = sum(f.n_test for f in folds)
    for k in keys:
        pairs = [(f.metrics.get(k, float("nan")), f.n_test) for f in folds]
        finite = [(v, n) for v, n in pairs if np.isfinite(v)]
        if finite:
            weighted = sum(v * n for v, n in finite)
            wn = sum(n for v, n in finite)
            agg[k] = round(float(weighted / wn), 5)
            agg[f"{k}_per_fold"] = [
                round(v, 5) if np.isfinite(v) else None for v, _ in pairs
            ]
        else:
            agg[k] = float("nan")
            agg[f"{k}_per_fold"] = [None] * len(pairs)
    agg["total_test_matches"] = int(total_n)
    agg["n_folds"] = len(folds)
    return agg


def compare_models_validation(matches: pl.DataFrame, team_matches: pl.DataFrame,
                              factories: Dict[str, Callable],
                              seasons: Optional[List[str]] = None,
                              seed: int = 42) -> Dict[str, ValidationResult]:
    return {name: run_expanding_validation(
        matches, team_matches, fn, seasons=seasons, seed=seed
    ) for name, fn in factories.items()}
