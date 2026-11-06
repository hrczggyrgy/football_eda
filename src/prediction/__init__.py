"""
Prediction subsystem public API.

See the individual modules for details:

- :mod:`src.prediction.features`   pre-match feature engineering
- :mod:`src.prediction.ratings`    Elo and team-strength ratings
- :mod:`src.prediction.models`     baseline, Poisson, multinomial, HGB
- :mod:`src.prediction.distributions` scoreline / derived probabilities
- :mod:`src.prediction.calibration` reliability diagnostics
- :mod:`src.prediction.validation` chronological expanding-window validation
- :mod:`src.prediction.evaluation` performance metrics
- :mod:`src.prediction.inference`  fixture prediction and model persistence
- :mod:`src.prediction.reporting`  figures and tables
"""

from __future__ import annotations

from .utils import (
    DEFAULT_SEED,
    MAX_GOALS,
    OUTCOME_LABELS,
    OUTCOME_TO_INDEX,
    REPORTED_SCORELINES,
    INDEX_TO_OUTCOME,
    OVER_UNDER_THRESHOLDS,
    encode_outcomes,
    index_to_outcome,
    outcome_to_index,
    points_from_outcome,
    outcome_from_goals,
    set_seed,
    validate_outcome_distribution,
    validate_probabilities,
    softmax,
)

__all__ = [
    "DEFAULT_SEED",
    "MAX_GOALS",
    "OUTCOME_LABELS",
    "OUTCOME_TO_INDEX",
    "INDEX_TO_OUTCOME",
    "OVER_UNDER_THRESHOLDS",
    "REPORTED_SCORELINES",
    "encode_outcomes",
    "index_to_outcome",
    "outcome_to_index",
    "points_from_outcome",
    "outcome_from_goals",
    "set_seed",
    "validate_outcome_distribution",
    "validate_probabilities",
    "softmax",
]
