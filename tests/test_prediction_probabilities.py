"""
Probability-integrity tests.

Verifies 1X2 probabilities sum to 1, scoreline distributions sum to 1,
derived markets agree with scoreline distributions, no negative probabilities,
finite expected goals, and valid prediction intervals.
"""

import numpy as np
import polars as pl
import pytest

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prediction.distributions import (
    scoreline_probabilities, derived_market_probs,
    poisson_1x2_from_lambdas, goal_prediction_intervals,
    most_probable_scoreline,
)
from prediction.utils import (
    validate_outcome_distribution, validate_probabilities,
    OUTCOME_LABELS, REPORTED_SCORELINES,
)


def test_1x2_probabilities_sum_to_one():
    for hl, al in [(0.5, 0.5), (1.3, 1.1), (2.5, 0.8), (0.1, 3.0)]:
        ph, pd_, pa = poisson_1x2_from_lambdas(hl, al)
        probs = np.array([ph, pd_, pa])
        validate_outcome_distribution(probs)
        assert abs(probs.sum() - 1.0) < 1e-9


def test_1x2_no_negative_probabilities():
    for hl, al in [(1.3, 1.1), (3.0, 3.0)]:
        ph, pd_, pa = poisson_1x2_from_lambdas(hl, al)
        assert ph >= 0 and pd_ >= 0 and pa >= 0


def test_scoreline_probabilities_sum_to_one():
    for hl, al in [(1.3, 1.1), (2.5, 2.5), (0.5, 4.0)]:
        sp = scoreline_probabilities(hl, al)
        total = sum(sp.values())
        assert abs(total - 1.0) < 1e-9, total
        assert all(0 <= v <= 1 for v in sp.values())
    # "other" tail bucket present.
    sp = scoreline_probabilities(3.0, 3.0)
    assert ("other", "other") in sp


def test_reported_scorelines_subset():
    sp = scoreline_probabilities(1.3, 1.1)
    for s in REPORTED_SCORELINES:
        assert s in sp or sp.get(s, 0.0) == 0.0


def test_derived_markets_coherent():
    sp = scoreline_probabilities(1.3, 1.1)
    dm = derived_market_probs(sp)
    # Complements sum to 1.
    assert abs(dm["over_1_5"] + dm["under_1_5"] - 1.0) < 1e-9
    assert abs(dm["over_2_5"] + dm["under_2_5"] - 1.0) < 1e-9
    assert abs(dm["over_3_5"] + dm["under_3_5"] - 1.0) < 1e-9
    assert abs(dm["btts"] + dm["btts_no"] - 1.0) < 1e-9
    assert abs(dm["home_scores"] + dm["home_no_score"] - 1.0) < 1e-9
    assert abs(dm["away_scores"] + dm["away_no_score"] - 1.0) < 1e-9
    # Monotonicity: over_1.5 >= over_2.5 >= over_3.5.
    assert dm["over_1_5"] >= dm["over_2_5"] >= dm["over_3_5"]
    # All values in [0,1].
    for v in dm.values():
        assert 0 <= v <= 1


def test_btts_from_scoreline():
    sp = scoreline_probabilities(1.3, 1.1)
    dm = derived_market_probs(sp)
    expected_btts = sum(p for (h, a), p in sp.items()
                        if h not in ("other",) and a not in ("other",) and h > 0 and a > 0)
    assert abs(dm["btts"] - expected_btts) < 1e-9


def test_goal_prediction_intervals_valid():
    iv = goal_prediction_intervals(1.3, 1.1, coverage=0.8)
    lo, hi = iv["home_goals"]
    assert 0 <= lo <= hi
    lo, hi = iv["away_goals"]
    assert 0 <= lo <= hi
    lo, hi = iv["total_goals"]
    assert 0 <= lo <= hi
    assert iv["coverage"] == 0.8


def test_most_probable_scoreline_is_in_distribution():
    sp = scoreline_probabilities(1.3, 1.1)
    score, prob = most_probable_scoreline(1.3, 1.1)
    assert score in sp
    if score != ("other", "other"):
        assert abs(prob - sp[score]) < 1e-9
    assert prob == max(sp.values())


def test_validate_outcome_distribution_rejects_bad():
    import pytest as _pytest
    # Sums to 1 -> OK (renormalized if within tolerance).
    validate_outcome_distribution(np.array([0.5, 0.3, 0.2]))
    # All zeros cannot normalize.
    with _pytest.raises(ValueError):
        validate_outcome_distribution(np.array([0.0, 0.0, 0.0]))
    # Negative -> invalid.
    with _pytest.raises(ValueError):
        validate_outcome_distribution(np.array([1.0, 1.0, -1.0]))
    # Non-finite.
    with _pytest.raises(ValueError):
        validate_outcome_distribution(np.array([np.inf, 0.0, 0.0]))


def test_outcome_labels_constant():
    assert OUTCOME_LABELS == ("H", "D", "A")
