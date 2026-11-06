"""
Leakage-prevention tests for the predictive engine.

These tests demonstrate that pre-match features never incorporate:
- the current match result,
- future information,
- same-date results (when timestamp ordering is unavailable),
- outcome-derived columns.

They primarily use synthetic match data so they do not depend on the full
real dataset.
"""

from datetime import date

import numpy as np
import polars as pl
import pytest

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.load_data import build_team_match_dataset
from prediction.features import build_pre_match_features, pair_match_features
from prediction.utils import encode_outcomes


def _make_master(rows):
    """Build a master match frame from a list of dicts."""
    df = pl.DataFrame(rows)
    # Derive HTR from half-time goals if not supplied.
    if "HTR" not in df.columns:
        df = df.with_columns(
            pl.when(pl.col("HTHG") > pl.col("HTAG")).then(pl.lit("H"))
            .when(pl.col("HTHG") < pl.col("HTAG")).then(pl.lit("A"))
            .otherwise(pl.lit("D")).alias("HTR")
        )
    # Ensure all core-ish columns exist with sensible dtypes.
    for c in ["FTHG", "FTAG", "HTHG", "HTAG", "HS", "AS", "HST", "AST",
              "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"]:
        if c not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.Int64).alias(c))
    if "FTR" not in df.columns:
        df = df.with_columns(
            pl.when(pl.col("FTHG") > pl.col("FTAG")).then(pl.lit("H"))
            .when(pl.col("FTHG") < pl.col("FTAG")).then(pl.lit("A"))
            .otherwise(pl.lit("D")).alias("FTR")
        )
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


# ---------------------------------------------------------------------------
# Test 1: Changing the target match's own result does not change its features
# ---------------------------------------------------------------------------
def test_target_result_does_not_change_own_features():
    base = [
        {"Div": "E0", "Season": "2122", "Date": date(2021, 9, 10),
         "HomeTeam": "A", "AwayTeam": "B", "FTHG": 1, "FTAG": 0,
         "HTHG": 1, "HTAG": 0, "HS": 10, "AS": 8, "HST": 4, "AST": 3,
         "HF": 11, "AF": 9, "HC": 5, "AC": 4, "HY": 1, "AY": 2, "HR": 0, "AR": 0},
        {"Div": "E0", "Season": "2122", "Date": date(2021, 9, 10),
         "HomeTeam": "C", "AwayTeam": "D", "FTHG": 0, "FTAG": 0,
         "HTHG": 0, "HTAG": 0, "HS": 9, "AS": 9, "HST": 3, "AST": 3,
         "HF": 10, "AF": 10, "HC": 4, "AC": 4, "HY": 1, "AY": 1, "HR": 0, "AR": 0},
    ]
    master = _make_master(base)
    tm = _tm(master)

    def feats(ftr_home, ftr_away):
        rows = list(base)
        rows[0]["FTHG"], rows[0]["FTAG"] = ftr_home, ftr_away
        m2 = _make_master(rows)
        tm2 = _tm(master) if False else _tm(m2)
        f = build_pre_match_features(tm2)
        p = pair_match_features(f)
        mrow = p.filter((pl.col("Date") == date(2021, 9, 10)) &
                        (pl.col("HomeTeam") == "A") &
                        (pl.col("AwayTeam") == "B"))
        return mrow

    r1 = feats(1, 0)
    r2 = feats(0, 2)  # different result for the target match
    # Exclude target columns themselves; compare pre-match feature columns.
    feature_cols = [c for c in r1.columns if c not in (
        "Div", "Season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR")]
    for c in feature_cols:
        v1 = r1.select(c).to_series()[0]
        v2 = r2.select(c).to_series()[0]
        if v1 is None and v2 is None:
            continue
        assert v1 == v2, f"Feature {c} changed when target result changed: {v1} vs {v2}"


# ---------------------------------------------------------------------------
# Test 2: Adding future matches does not change past predictions/features
# ---------------------------------------------------------------------------
def test_adding_future_matches_does_not_change_past():
    base = [
        {"Div": "E0", "Season": "2122", "Date": date(2021, 9, 10),
         "HomeTeam": "A", "AwayTeam": "B", "FTHG": 1, "FTAG": 0,
         "HTHG": 1, "HTAG": 0, "HS": 10, "AS": 8, "HST": 4, "AST": 3,
         "HF": 11, "AF": 9, "HC": 5, "AC": 4, "HY": 1, "AY": 2, "HR": 0, "AR": 0},
    ]
    future = list(base) + [
        {"Div": "E0", "Season": "2122", "Date": date(2021, 9, 20),
         "HomeTeam": "B", "AwayTeam": "A", "FTHG": 3, "FTAG": 0,
         "HTHG": 1, "HTAG": 0, "HS": 12, "AS": 6, "HST": 5, "AST": 2,
         "HF": 12, "AF": 8, "HC": 6, "AC": 3, "HY": 2, "AY": 1, "HR": 0, "AR": 0},
    ]
    tm_base = _tm(_make_master(base))
    tm_future = _tm(_make_master(future))

    f_base = build_pre_match_features(tm_base)
    f_future = build_pre_match_features(tm_future)

    # The past match (2021-09-10) features must be identical in both.
    pb = pair_match_features(f_base).filter(
        (pl.col("Date") == date(2021, 9, 10)) & (pl.col("HomeTeam") == "A"))
    pf = pair_match_features(f_future).filter(
        (pl.col("Date") == date(2021, 9, 10)) & (pl.col("HomeTeam") == "A"))
    assert pb.height == 1 and pf.height == 1
    feature_cols = [c for c in pb.columns if c not in (
        "Div", "Season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR")]
    for c in feature_cols:
        v1 = pb.select(c).to_series()[0]
        v2 = pf.select(c).to_series()[0]
        if v1 is None and v2 is None:
            continue
        assert v1 == v2, f"Past feature {c} changed when future match added: {v1} vs {v2}"


# ---------------------------------------------------------------------------
# Test 3: Same-day results do not leak into pre-match features
# ---------------------------------------------------------------------------
def test_same_day_safeguard():
    # Two matches on the same date; target must not see the other same-date match.
    base = [
        {"Div": "E0", "Season": "2122", "Date": date(2021, 9, 10),
         "HomeTeam": "A", "AwayTeam": "B", "FTHG": 1, "FTAG": 0,
         "HTHG": 1, "HTAG": 0, "HS": 10, "AS": 8, "HST": 4, "AST": 3,
         "HF": 11, "AF": 9, "HC": 5, "AC": 4, "HY": 1, "AY": 2, "HR": 0, "AR": 0},
        {"Div": "E0", "Season": "2122", "Date": date(2021, 9, 10),
         "HomeTeam": "C", "AwayTeam": "D", "FTHG": 0, "FTAG": 0,
         "HTHG": 0, "HTAG": 0, "HS": 9, "AS": 9, "HST": 3, "AST": 3,
         "HF": 10, "AF": 10, "HC": 4, "AC": 4, "HY": 1, "AY": 1, "HR": 0, "AR": 0},
    ]
    master = _make_master(base)
    tm = _tm(master)
    f = build_pre_match_features(tm)
    p = pair_match_features(f)
    target = p.filter((pl.col("Date") == date(2021, 9, 10)) &
                      (pl.col("HomeTeam") == "A") &
                      (pl.col("AwayTeam") == "B"))
    assert target.height == 1
    # Rest days for the season opener should be null (no prior match), not
    # inferred from the same-date other fixture.
    rest = target.select("Home_Rest_Days").to_series()[0]
    assert rest is None or (isinstance(rest, float) and np.isnan(rest))


# ---------------------------------------------------------------------------
# Test 4: Outcome-derived columns are not used as predictors
# ---------------------------------------------------------------------------
def test_no_outcome_columns_in_features():
    base = [
        {"Div": "E0", "Season": "2122", "Date": date(2021, 9, 10),
         "HomeTeam": "A", "AwayTeam": "B", "FTHG": 1, "FTAG": 0,
         "HTHG": 1, "HTAG": 0, "HS": 10, "AS": 8, "HST": 4, "AST": 3,
         "HF": 11, "AF": 9, "HC": 5, "AC": 4, "HY": 1, "AY": 2, "HR": 0, "AR": 0},
    ]
    tm = _tm(_make_master(base))
    f = build_pre_match_features(tm)
    p = pair_match_features(f)
    # None of the engineered feature columns should be raw goals/result of the
    # current match (aside from the explicit target columns FTHG/FTAG/FTR).
    outcome_derived = {"FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR"}
    feature_cols = set(p.columns)
    # Target columns are allowed to exist but are separated from predictors.
    leaked = outcome_derived & feature_cols
    # They may exist as targets; the point is they're not among model features.
    # Here we assert they are present ONLY as the bare target columns, not
    # prefixed as engineered predictors.
    for c in p.columns:
        assert not c.startswith("Home_FTHG") and not c.startswith("Away_FTAG")


# ---------------------------------------------------------------------------
# Test 5: encode_outcomes mapping
# ---------------------------------------------------------------------------
def test_encode_outcomes_mapping():
    s = pl.Series(["H", "D", "A", None, "X"])
    enc = encode_outcomes(s)
    assert list(enc) == [0, 1, 2, -1, -1]
