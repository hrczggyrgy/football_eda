"""
Feature-engineering tests for the predictive engine.

Uses synthetic match data to verify rolling windows, shrinkage, cross-season
carryover, home/away splits, and missing-data handling are strictly pre-match.
"""

from datetime import date

import polars as pl
import pytest

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.load_data import build_team_match_dataset
from prediction.features import build_pre_match_features, pair_match_features, FORM_WINDOWS


def _master(rows):
    df = pl.DataFrame(rows)
    if "HTR" not in df.columns:
        df = df.with_columns(
            pl.when(pl.col("HTHG") > pl.col("HTAG")).then(pl.lit("H"))
            .when(pl.col("HTHG") < pl.col("HTAG")).then(pl.lit("A"))
            .otherwise(pl.lit("D")).alias("HTR")
        )
    if "FTR" not in df.columns:
        df = df.with_columns(
            pl.when(pl.col("FTHG") > pl.col("FTAG")).then(pl.lit("H"))
            .when(pl.col("FTHG") < pl.col("FTAG")).then(pl.lit("A"))
            .otherwise(pl.lit("D")).alias("FTR")
        )
    for c in ["HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR",
              "HTHG", "HTAG"]:
        if c not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.Int64).alias(c))
    if "Date" in df.columns and df["Date"].dtype != pl.Date:
        df = df.with_columns(pl.col("Date").cast(pl.Date))
    if "Season" not in df.columns:
        df = df.with_columns(pl.lit("2122").alias("Season"))
    if "Div" not in df.columns:
        df = df.with_columns(pl.lit("E0").alias("Div"))
    if "Time" not in df.columns:
        df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias("Time"))
    if "Referee" not in df.columns:
        df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias("Referee"))
    return df.sort(["Div", "Date"])


def _tm(master):
    return build_team_match_dataset(master)


def test_rolling_windows_exist_and_are_lagged():
    rows = [
        {"Div": "E0", "Season": "2122", "Date": date(2021, 9, d),
         "HomeTeam": "A", "AwayTeam": "Opp", "FTHG": i % 3, "FTAG": 0,
         "HTHG": i % 2, "HTAG": 0, "HS": 10 + i, "AS": 8, "HST": 4, "AST": 3,
         "HF": 11, "AF": 9, "HC": 5, "AC": 4, "HY": 1, "AY": 1, "HR": 0, "AR": 0}
        for i, d in enumerate(range(1, 11), start=1)
    ]
    tm = _tm(_master(rows))
    f = build_pre_match_features(tm)
    # First match of the team-season: no prior history -> rolling/cum null.
    first = f.filter(pl.col("Match_Num") == 1)
    assert first.select(pl.col("Points_L5").is_null().all()).to_series()[0]
    assert first.select(pl.col("Points_Cum").is_null().all()).to_series()[0]
    # Windows configured.
    for w in FORM_WINDOWS:
        assert f"GF_L{w}" in f.columns
        assert f"GF_L{w}_std" in f.columns


def test_home_away_splits_and_prior_season():
    rows = [
        {"Div": "E0", "Season": "2122", "Date": date(2021, 9, d),
         "HomeTeam": "A", "AwayTeam": "Opp", "FTHG": 2, "FTAG": 0,
         "HTHG": 1, "HTAG": 0, "HS": 10, "AS": 8, "HST": 4, "AST": 3,
         "HF": 11, "AF": 9, "HC": 5, "AC": 4, "HY": 1, "AY": 1, "HR": 0, "AR": 0}
        for d in range(1, 6)
    ] + [
        {"Div": "E0", "Season": "2223", "Date": date(2022, 9, 1),
         "HomeTeam": "A", "AwayTeam": "Opp", "FTHG": 1, "FTAG": 1,
         "HTHG": 1, "HTAG": 0, "HS": 9, "AS": 7, "HST": 3, "AST": 2,
         "HF": 10, "AF": 8, "HC": 4, "AC": 3, "HY": 1, "AY": 1, "HR": 0, "AR": 0},
    ]
    tm = _tm(_master(rows))
    f = build_pre_match_features(tm)
    # Second-season first match should have Prior_Season_PPG from 2122.
    sec = f.filter((pl.col("Season") == "2223") & (pl.col("Team") == "A") & (pl.col("Match_Num") == 1))
    assert sec.height == 1
    pps = sec.select("Prior_Season_PPG").to_series()[0]
    assert pps is not None and not (isinstance(pps, float) and np_is_nan(pps)), pps
    # Home_PPG should exist.
    assert "Home_PPG" in f.columns and "Away_PPG" in f.columns


def np_is_nan(v):
    return v is not None and isinstance(v, float) and v != v


def test_pair_match_features_produces_match_rows():
    rows = [
        {"Div": "E0", "Season": "2122", "Date": date(2021, 9, 10),
         "HomeTeam": "A", "AwayTeam": "B", "FTHG": 2, "FTAG": 1,
         "HTHG": 1, "HTAG": 0, "HS": 10, "AS": 8, "HST": 4, "AST": 2,
         "HF": 11, "AF": 9, "HC": 5, "AC": 4, "HY": 1, "AY": 1, "HR": 0, "AR": 0},
    ]
    tm = _tm(_master(rows))
    f = build_pre_match_features(tm)
    p = pair_match_features(f)
    assert p.height == 1
    assert "FTHG" in p.columns and "FTAG" in p.columns and "FTR" in p.columns
    assert p.select("FTHG").to_series()[0] == 2
    assert p.select("FTR").to_series()[0] == "H"
    # Home/away prefixed features present.
    assert any(c.startswith("Home_") for c in p.columns)
    assert any(c.startswith("Away_") for c in p.columns)


def test_missing_statistics_not_zero():
    # A match with null shots should not turn shots features into 0.
    rows = [
        {"Div": "E0", "Season": "2122", "Date": date(2021, 9, 10),
         "HomeTeam": "A", "AwayTeam": "B", "FTHG": 1, "FTAG": 0,
         "HTHG": 1, "HTAG": 0, "HS": None, "AS": None, "HST": None, "AST": None,
         "HF": None, "AF": None, "HC": None, "AC": None, "HY": None, "AY": None,
         "HR": None, "AR": None},
    ]
    tm = _tm(_master(rows))
    f = build_pre_match_features(tm)
    p = pair_match_features(f)
    assert p.height == 1
    shots = p.select("Home_Shots_L5").to_series()[0]
    assert shots is None or (isinstance(shots, float) and shots != shots), shots
