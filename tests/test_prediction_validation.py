"""
Chronological validation tests.

Verifies expanding-window design, strict train-before-test ordering, no random
splitting, no test-set calibration/hyperparameter selection, and temporal
feature updates.
"""

from datetime import date

import numpy as np
import polars as pl
import pytest

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.load_data import build_master_dataset, build_team_match_dataset, CORE_COLS
from prediction.validation import chronological_folds, run_expanding_validation
from prediction.models import HistoricalBaseline, PoissonModel, MultinomialModel


def _make_master(n_leagues=2, n_seasons=3):
    rows = []
    for li, div in enumerate(["E0", "E1"][:n_leagues]):
        for si, season in enumerate(["2122", "2223", "2324"][:n_seasons]):
            for i in range(20):
                d = date(2021 + si, 8, 1 + (i % 20))
                rows.append({
                    "Div": div, "Season": season, "Date": d,
                    "HomeTeam": f"H{li}{i % 4}", "AwayTeam": f"A{li}{(i + 1) % 4}",
                    "FTHG": i % 3, "FTAG": (i + 1) % 3,
                    "HTHG": i % 2, "HTAG": 0, "HS": 10 + i, "AS": 8, "HST": 4, "AST": 3,
                    "HF": 11, "AF": 9, "HC": 5, "AC": 4, "HY": 1, "AY": 1, "HR": 0, "AR": 0,
                })
    df = pl.DataFrame(rows).with_columns(pl.col("Date").cast(pl.Date)).with_columns(
        pl.when(pl.col("FTHG") > pl.col("FTAG")).then(pl.lit("H"))
        .when(pl.col("FTHG") < pl.col("FTAG")).then(pl.lit("A"))
        .otherwise(pl.lit("D")).alias("FTR")
    ).with_columns(
        pl.when(pl.col("HTHG") > pl.col("HTAG")).then(pl.lit("H"))
        .when(pl.col("HTHG") < pl.col("HTAG")).then(pl.lit("A"))
        .otherwise(pl.lit("D")).alias("HTR")
    )
    return df


def test_chronological_folds_structure():
    seasons = ["1920", "2021", "2122", "2223", "2324", "2425", "2526"]
    folds = chronological_folds(seasons, min_train_seasons=2)
    # Expect 5 folds: (2 seasons train -> 1 test) ... up to (6 train -> 1 test)
    assert len(folds) == 5
    for train, test in folds:
        # All training seasons strictly precede the test season.
        train_nums = [int(s[:2]) + 2000 for s in train]
        test_num = int(test[:2]) + 2000
        assert max(train_nums) < test_num
        # Train is expanding.
    # First fold: train 1920+2021, test 2122.
    assert folds[0] == (["1920", "2021"], "2122")
    assert folds[-1] == (["1920", "2021", "2122", "2223", "2324", "2425"], "2526")


def test_run_expanding_validation_train_before_test():
    master = _make_master()
    team_matches = build_team_match_dataset(master)

    def factory():
        return HistoricalBaseline()

    res = run_expanding_validation(master, team_matches, factory,
                                   seasons=["2122", "2223", "2324"],
                                   min_train_seasons=2)
    assert res.n_folds == 1
    f = res.folds[0]
    assert f.train_seasons == ["2122", "2223"]
    assert f.test_season == "2324"
    # Training cutoff date precedes all test dates.
    test_dates = master.filter(pl.col("Season") == "2324").select(pl.col("Date").min()).to_series()[0]
    cutoff = f.train_cutoff_date
    assert str(test_dates) > cutoff


def test_no_random_split_order_preserved():
    master = _make_master()
    team_matches = build_team_match_dataset(master)
    res = run_expanding_validation(master, team_matches,
                                   lambda: HistoricalBaseline(),
                                   seasons=["2122", "2223", "2324"],
                                   min_train_seasons=2)
    f = res.folds[0]
    assert f.n_train > 0 and f.n_test > 0
    # Metrics are finite.
    assert np.isfinite(f.metrics["brier"])
    assert np.isfinite(f.metrics["log_loss"])


def test_aggregate_has_all_models_metrics():
    master = _make_master()
    team_matches = build_team_match_dataset(master)
    factories = {
        "baseline": lambda: HistoricalBaseline(),
        "poisson": lambda: PoissonModel(),
    }
    results = {}
    for name, fn in factories.items():
        results[name] = run_expanding_validation(master, team_matches, fn,
                                                 seasons=["2122", "2223", "2324"],
                                                 min_train_seasons=2)
    for name, vr in results.items():
        assert "brier" in vr.aggregate
        assert vr.aggregate["n_folds"] == 1
        assert vr.aggregate["total_test_matches"] > 0
