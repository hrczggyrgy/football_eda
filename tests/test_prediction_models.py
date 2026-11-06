"""
Model tests for the predictive engine.

Verifies fitting, prediction, valid probability outputs, reproducibility, and
graceful handling of missing columns. Uses synthetic data.
"""

from datetime import date

import numpy as np
import polars as pl
import pytest

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.load_data import build_master_dataset, build_team_match_dataset, CORE_COLS
from prediction.models import HistoricalBaseline, PoissonModel, MultinomialModel
from prediction.distributions import (
    scoreline_probabilities, derived_market_probs,
    poisson_1x2_from_lambdas, goal_prediction_intervals,
    most_probable_scoreline,
)
from prediction.utils import validate_outcome_distribution, encode_outcomes


def _small_master():
    rows = [
        {"Div": "E0", "Season": "2122", "Date": date(2021, 9, d),
         "HomeTeam": f"T{i % 3}", "AwayTeam": f"T{(i + 1) % 3}",
         "FTHG": i % 3, "FTAG": (i + 1) % 3,
         "HTHG": i % 2, "HTAG": 0, "HS": 10 + i, "AS": 8, "HST": 4, "AST": 3,
         "HF": 11, "AF": 9, "HC": 5, "AC": 4, "HY": 1, "AY": 1, "HR": 0, "AR": 0}
        for i, d in enumerate(range(1, 13))
    ]
    return pl.DataFrame(rows).with_columns(pl.col("Date").cast(pl.Date)).with_columns(
        pl.when(pl.col("FTHG") > pl.col("FTAG")).then(pl.lit("H"))
        .when(pl.col("FTHG") < pl.col("FTAG")).then(pl.lit("A"))
        .otherwise(pl.lit("D")).alias("FTR")
    ).with_columns(
        pl.when(pl.col("HTHG") > pl.col("HTAG")).then(pl.lit("H"))
        .when(pl.col("HTHG") < pl.col("HTAG")).then(pl.lit("A"))
        .otherwise(pl.lit("D")).alias("HTR")
    )


def test_baseline_fit_predict_valid_probs():
    master = _small_master()
    m = HistoricalBaseline()
    m.fit(None, master)
    res = m.predict(master)
    assert res.fitted
    assert res.outcome_probs.shape[1] == 3
    for row in res.outcome_probs:
        validate_outcome_distribution(row)
    assert np.all(np.isfinite(res.home_lambda))
    assert np.all(res.home_lambda > 0)
    assert np.all(np.isfinite(res.away_lambda))


def test_poisson_fit_predict_valid_probs():
    master = _small_master()
    tm = build_team_match_dataset(master)
    m = PoissonModel()
    m.fit(tm, master)
    res = m.predict(master)
    assert res.fitted
    assert res.outcome_probs.shape[1] == 3
    for row in res.outcome_probs:
        validate_outcome_distribution(row)
    assert np.all(np.isfinite(res.home_lambda))
    assert np.all(np.isfinite(res.away_lambda))


def test_multinomial_fit_predict_valid_probs():
    master = _small_master()
    tm = build_team_match_dataset(master)
    from prediction.features import build_pre_match_features, pair_match_features
    feat = build_pre_match_features(tm)
    paired = pair_match_features(feat)
    m = MultinomialModel()
    m.fit(tm, paired)
    res = m.predict(paired)
    assert res.fitted
    assert res.outcome_probs.shape[1] == 3
    for row in res.outcome_probs:
        validate_outcome_distribution(row)


def test_reproducibility():
    master = _small_master()
    tm = build_team_match_dataset(master)
    m1 = HistoricalBaseline(); m1.fit(None, master)
    m2 = HistoricalBaseline(); m2.fit(None, master)
    assert np.allclose(m1.predict(master).outcome_probs,
                       m2.predict(master).outcome_probs)


def test_poisson_1x2_sums_to_one():
    ph, pd_, pa = poisson_1x2_from_lambdas(1.3, 1.1)
    total = ph + pd_ + pa
    assert abs(total - 1.0) < 1e-9
    assert all(0 <= v <= 1 for v in (ph, pd_, pa))


def test_scoreline_distribution_sums_to_one():
    sp = scoreline_probabilities(1.3, 1.1)
    total = sum(sp.values())
    assert abs(total - 1.0) < 1e-9
    assert all(0 <= v <= 1 for v in sp.values())


def test_derived_markets_agree_with_scoreline():
    sp = scoreline_probabilities(1.3, 1.1)
    dm = derived_market_probs(sp)
    assert 0 <= dm["btts"] <= 1
    assert 0 <= dm["over_2_5"] <= 1
    assert abs(dm["over_2_5"] + dm["under_2_5"] - 1.0) < 1e-9
    assert abs(dm["btts"] + dm["btts_no"] - 1.0) < 1e-9


def test_goal_prediction_intervals_valid():
    iv = goal_prediction_intervals(1.3, 1.1, coverage=0.8)
    assert iv["home_goals"][0] <= iv["home_goals"][1]
    assert iv["away_goals"][0] <= iv["away_goals"][1]
    assert iv["total_goals"][0] <= iv["total_goals"][1]
    assert iv["coverage"] == 0.8


def test_most_probable_scoreline():
    score, prob = most_probable_scoreline(1.3, 1.1)
    assert isinstance(score, tuple) and len(score) == 2
    assert 0 <= prob <= 1


def test_missing_columns_graceful():
    master = _small_master()
    # Baseline only needs Div/FTHG/FTAG/FTR — present.
    m = HistoricalBaseline(); m.fit(None, master)
    res = m.predict(master)
    assert res.fitted


# ---------------------------------------------------------------------------
# F03: Poisson defensive-direction regression tests
# ---------------------------------------------------------------------------
def test_poisson_defence_direction_monotonic():
    # Higher conceded rate (weak defence, factor > 1) must make the opponent
    # more likely to score (higher lambda), not less.
    m = PoissonModel()
    m.league_effects = {"E0": 1.3}
    m.league_home_advantage = {"E0": 0.0}
    m.team_attack = {"E0|Home": 1.0, "E0|Away": 1.0}
    m.team_defense = {"E0|Home": 1.0, "E0|Away": 1.0}

    base_h, base_a = m._lambda("E0", "Home", "Away")

    # Away team concedes more (defence factor up) -> home lambda should rise.
    m.team_defense["E0|Away"] = 2.0
    weak_away_h, _ = m._lambda("E0", "Home", "Away")
    assert weak_away_h > base_h, (weak_away_h, base_h)

    # Reset and test the other direction.
    m.team_defense["E0|Away"] = 1.0
    m.team_defense["E0|Home"] = 2.0
    _, weak_home_a = m._lambda("E0", "Home", "Away")
    assert weak_home_a > base_a, (weak_home_a, base_a)

    # Lower defence factor (strong defence, < 1) reduces opponent scoring.
    m.team_defense["E0|Home"] = 0.5
    _, strong_home_a = m._lambda("E0", "Home", "Away")
    assert strong_home_a < base_a, (strong_home_a, base_a)
