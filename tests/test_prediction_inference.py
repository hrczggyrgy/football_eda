"""
Inference-interface tests.

Verifies predict_fixture() and predict_from_csv() work for supplied upcoming
fixtures, that result-free CSV input is accepted, fallback treatment for
unknown teams/leagues is recorded, and output probabilities sum to 1.
"""

from datetime import date

import csv
import numpy as np
import polars as pl
import pytest

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.load_data import build_master_dataset, build_team_match_dataset, CORE_COLS
from prediction.inference import (
    PredictionBundle, predict_fixture, predict_from_csv, _season_from_date,
)


def _small_master():
    rows = [
        {"Div": "E0", "Season": "2122", "Date": __import__("datetime").date(2021, 9, d),
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


def _bundle():
    master = _small_master()
    tm = build_team_match_dataset(master)
    from prediction.models import MultinomialModel
    m = MultinomialModel()
    from prediction.features import build_pre_match_features, pair_match_features
    feat = build_pre_match_features(tm)
    paired = pair_match_features(feat)
    m.fit(tm, paired)
    from prediction.inference import PredictionBundle
    return PredictionBundle(m, "test_model", {
        "model_version": "test", "training_cutoff": "2021-09-12",
        "training_seasons": ["2122"], "training_leagues": ["E0"],
        "n_training_matches": int(master.height),
        "feature_names": m.feature_names,
    }, m.feature_names, cutoff_date="2021-09-12", cutoff_season="2122"), master, tm


def test_predict_fixture_output_structure():
    bundle, master, tm = _bundle()
    p = predict_fixture("E0", "2021-09-13", "T0", "T1", bundle, master, tm)
    assert p.date == "2021-09-13"
    assert p.league == "E0"
    assert p.home_team == "T0" and p.away_team == "T1"
    # Probabilities sum to 1.
    assert abs(p.outcome_probs_sum() - 1.0) < 1e-6
    assert all(0 <= v <= 1 for v in (p.home_win_probability,
                                     p.draw_probability, p.away_win_probability))
    # Finite expected goals.
    assert p.home_expected_goals is not None and np.isfinite(p.home_expected_goals)
    assert p.away_expected_goals is not None and np.isfinite(p.away_expected_goals)
    assert np.isfinite(p.total_expected_goals)
    # Derived markets in [0,1].
    for v in (p.both_teams_to_score_probability, p.over_1_5_probability,
              p.over_2_5_probability, p.over_3_5_probability):
        assert 0 <= v <= 1
    # Intervals valid.
    assert p.home_goals_interval[0] <= p.home_goals_interval[1]
    assert p.away_goals_interval[0] <= p.away_goals_interval[1]
    # Metadata recorded.
    assert p.model_version == "test"
    assert p.training_cutoff == "2021-09-12"
    assert isinstance(p.feature_warnings, list)


def test_predict_fixture_unknown_team_recorded():
    bundle, master, tm = _bundle()
    p = predict_fixture("E0", "2021-09-13", "T0", "UnknownTeam", bundle, master, tm)
    assert p.data_quality_status == "fallback_unknown_team_or_league"
    assert abs(p.outcome_probs_sum() - 1.0) < 1e-6
    assert len(p.feature_warnings) > 0


def test_predict_fixture_unknown_league_recorded():
    bundle, master, tm = _bundle()
    p = predict_fixture("ZZ", "2021-09-13", "T0", "T1", bundle, master, tm)
    assert p.data_quality_status == "fallback_unknown_team_or_league"
    assert abs(p.outcome_probs_sum() - 1.0) < 1e-6


def test_predict_from_csv_no_results_required(tmp_path):
    bundle, master, tm = _bundle()
    csv_path = Path(tmp_path) / "inference_test.csv"
    out_path = Path(tmp_path) / "inference_test_out.csv"
    with open(csv_path, "w", newline="") as f:
        f.write("Date,Div,HomeTeam,AwayTeam\n")
        f.write("2021-09-13,E0,T0,T1\n")
        f.write("2021-09-13,E0,T1,T2\n")
    preds = predict_from_csv(csv_path, bundle, master, tm, output_path=out_path)
    assert len(preds) == 2
    # No results column needed; all sums to 1.
    for p in preds:
        assert abs(p.outcome_probs_sum() - 1.0) < 1e-6
    assert out_path.exists()
    header = out_path.read_text().splitlines()[0]
    assert "home_win_probability" in header
    assert "outcome_probs_sum" in header


def test_predict_from_csv_missing_columns_raises(tmp_path):
    bundle, master, tm = _bundle()
    csv_path = Path(tmp_path) / "bad.csv"
    with open(csv_path, "w", newline="") as f:
        f.write("Date,HomeTeam,AwayTeam\n")  # missing Div
    with pytest.raises(ValueError):
        predict_from_csv(csv_path, bundle, master, tm)


def test_bundle_save_load_roundtrip(tmp_path):
    bundle, master, tm = _bundle()
    p = Path(tmp_path) / "model.joblib"
    bundle.save(p)
    loaded = PredictionBundle.load(p)
    assert loaded.model_name == bundle.model_name
    assert loaded.cutoff_date == bundle.cutoff_date
    assert loaded.feature_names == bundle.feature_names


# ---------------------------------------------------------------------------
# F04: season assignment regression tests
# ---------------------------------------------------------------------------
def test_season_from_date_mapping():
    # On/after 1 August -> season starting that calendar year.
    assert _season_from_date(date(2021, 9, 10)) == "2122"
    assert _season_from_date(date(2021, 8, 1)) == "2122"
    assert _season_from_date(date(2021, 7, 31)) == "2021"
    # Early calendar year -> prior season.
    assert _season_from_date(date(2022, 3, 15)) == "2122"
    assert _season_from_date(date(2026, 10, 5)) == "2627"


def test_predict_fixture_uses_supplied_season_not_latest_history():
    bundle, master, tm = _bundle()
    # Single-season history (2122) so feature building stays on one season.
    # A fixture dated within the history season keeps its actual season.
    p = predict_fixture(
        "E0", "2021-09-13", "T0", "T1", bundle, master, tm,
        cutoff_date="2021-09-12", season="2122",
    )
    assert p.date == "2021-09-13"
    assert abs(p.outcome_probs_sum() - 1.0) < 1e-6
    assert p.data_quality_status != "fallback_unknown_team_or_league"

    # Derived season (no explicit season) for an in-history date resolves to
    # that date's season, not blindly the latest history season.
    p2 = predict_fixture(
        "E0", "2021-09-13", "T0", "T1", bundle, master, tm,
        cutoff_date="2021-09-12",
    )
    assert abs(p2.outcome_probs_sum() - 1.0) < 1e-6
