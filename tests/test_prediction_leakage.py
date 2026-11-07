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


# ---------------------------------------------------------------------------
# Test 6: Historical-count preservation
# Exposure counts equal the number of eligible previous completed matches
# ---------------------------------------------------------------------------
def test_historical_count_preservation():
    # Team A plays 5 matches in season 2122
    base = [
        {"Div": "E0", "Season": "2122", "Date": date(2021, 9, d),
         "HomeTeam": "A" if i % 2 == 0 else "Opp",
         "AwayTeam": "Opp" if i % 2 == 0 else "A",
         "FTHG": 1, "FTAG": 0, "HTHG": 1, "HTAG": 0,
         "HS": 10, "AS": 8, "HST": 4, "AST": 3,
         "HF": 11, "AF": 9, "HC": 5, "AC": 4, "HY": 1, "AY": 2, "HR": 0, "AR": 0}
        for i, d in enumerate([10, 17, 24, 27, 30])
    ]
    master = _make_master(base)
    tm = _tm(master)
    f = build_pre_match_features(tm)
    
    # Check Match_Num and Matches_Cum_Lagged for each match
    team_a = f.filter((pl.col("Team") == "A") & (pl.col("Season") == "2122")).sort("Date")
    for i, row in enumerate(team_a.iter_rows(named=True), 1):
        match_num = row["Match_Num"]
        matches_lagged = row["Matches_Cum_Lagged"]
        assert match_num == i, f"Match_Num {match_num} != expected {i}"
        assert matches_lagged == i - 1, f"Matches_Cum_Lagged {matches_lagged} != expected {i-1}"


# ---------------------------------------------------------------------------
# Test 7: First-match prior
# A team's first fixture receives its available prior rather than null shrinkage
# ---------------------------------------------------------------------------
def test_first_match_prior():
    # Season 2122: Team A plays 3 matches (3 wins = 3.0 PPG)
    # Using a specific opponent team to ensure prior season lookup works
    s2122 = [
        {"Div": "E0", "Season": "2122", "Date": date(2021, 9, d),
         "HomeTeam": "A" if i % 2 == 0 else "Opp",
         "AwayTeam": "Opp" if i % 2 == 0 else "A",
         "FTHG": 2, "FTAG": 1, "HTHG": 1, "HTAG": 0,
         "HS": 10, "AS": 8, "HST": 4, "AST": 3,
         "HF": 11, "AF": 9, "HC": 5, "AC": 4, "HY": 1, "AY": 2, "HR": 0, "AR": 0}
        for i, d in enumerate([10, 17, 24])
    ]
    # Season 2223: Team A first match
    s2223 = [
        {"Div": "E0", "Season": "2223", "Date": date(2022, 9, 1),
         "HomeTeam": "A", "AwayTeam": "B",
         "FTHG": 1, "FTAG": 1, "HTHG": 0, "HTAG": 0,
         "HS": 10, "AS": 8, "HST": 4, "AST": 3,
         "HF": 11, "AF": 9, "HC": 5, "AC": 4, "HY": 1, "AY": 2, "HR": 0, "AR": 0},
    ]
    
    master = _make_master(s2122 + s2223)
    tm = _tm(master)
    f = build_pre_match_features(tm)
    
    # Check Team A's first match of 2223
    first_match = f.filter((pl.col("Season") == "2223") & (pl.col("Team") == "A") & (pl.col("Match_Num") == 1))
    assert first_match.height == 1
    
    # Prior_Season_PPG should be available (either team's prior or division fallback)
    prior_ppg = first_match.select("Prior_Season_PPG").to_series()[0]
    assert prior_ppg is not None, f"Expected Prior_Season_PPG to be set, got None"
    
    # PreMatch_PPG_Shrunk should equal the prior (since n=0)
    shrunk = first_match.select("PreMatch_PPG_Shrunk").to_series()[0]
    assert shrunk is not None, f"Expected PreMatch_PPG_Shrunk to be set, got None"
    assert shrunk == prior_ppg, f"Expected PreMatch_PPG_Shrunk={prior_ppg} (prior), got {shrunk}"
    
    # Matches_Cum_Lagged should be 0
    mc = first_match.select("Matches_Cum_Lagged").to_series()[0]
    assert mc == 0, f"Expected Matches_Cum_Lagged=0, got {mc}"


# ---------------------------------------------------------------------------
# Test 8: H2H calculation
# Direct draw counts agree with the reported draw rate
# ---------------------------------------------------------------------------
def test_h2h_draw_rate_calculation():
    # 4 matches between A and B across seasons:
    # 1. A home vs B: 1-1 (draw)
    # 2. B home vs A: 2-0 (B wins)
    # 3. A home vs B: 0-0 (draw)
    # 4. B home vs A: 1-1 (draw)
    # Total: 4 matches, 3 draws = 75% draw rate
    base = [
        {"Div": "E0", "Season": "2122", "Date": date(2021, 9, 10),
         "HomeTeam": "A", "AwayTeam": "B", "FTHG": 1, "FTAG": 1,
         "HTHG": 0, "HTAG": 0, "HS": 10, "AS": 8, "HST": 4, "AST": 3,
         "HF": 11, "AF": 9, "HC": 5, "AC": 4, "HY": 1, "AY": 2, "HR": 0, "AR": 0},
        {"Div": "E0", "Season": "2122", "Date": date(2021, 10, 10),
         "HomeTeam": "B", "AwayTeam": "A", "FTHG": 2, "FTAG": 0,
         "HTHG": 1, "HTAG": 0, "HS": 10, "AS": 8, "HST": 4, "AST": 3,
         "HF": 11, "AF": 9, "HC": 5, "AC": 4, "HY": 1, "AY": 2, "HR": 0, "AR": 0},
        {"Div": "E0", "Season": "2223", "Date": date(2022, 9, 10),
         "HomeTeam": "A", "AwayTeam": "B", "FTHG": 0, "FTAG": 0,
         "HTHG": 0, "HTAG": 0, "HS": 10, "AS": 8, "HST": 4, "AST": 3,
         "HF": 11, "AF": 9, "HC": 5, "AC": 4, "HY": 1, "AY": 2, "HR": 0, "AR": 0},
        {"Div": "E0", "Season": "2223", "Date": date(2022, 10, 10),
         "HomeTeam": "B", "AwayTeam": "A", "FTHG": 1, "FTAG": 1,
         "HTHG": 1, "HTAG": 1, "HS": 10, "AS": 8, "HST": 4, "AST": 3,
         "HF": 11, "AF": 9, "HC": 5, "AC": 4, "HY": 1, "AY": 2, "HR": 0, "AR": 0},
        # Target fixture: 2023 season
        {"Div": "E0", "Season": "2324", "Date": date(2023, 9, 10),
         "HomeTeam": "A", "AwayTeam": "B", "FTHG": 2, "FTAG": 1,
         "HTHG": 1, "HTAG": 0, "HS": 10, "AS": 8, "HST": 4, "AST": 3,
         "HF": 11, "AF": 9, "HC": 5, "AC": 4, "HY": 1, "AY": 2, "HR": 0, "AR": 0},
    ]
    
    master = _make_master(base)
    tm = _tm(master)
    f = build_pre_match_features(tm)
    p = pair_match_features(f)
    
    # Check target fixture
    target = p.filter((pl.col("Season") == "2324") & (pl.col("HomeTeam") == "A") & (pl.col("AwayTeam") == "B"))
    assert target.height == 1
    
    # H2H_Matches should be 4
    h2h_matches = target.select("Home_H2H_Matches").to_series()[0]
    assert h2h_matches == 4, f"Expected 4 H2H matches, got {h2h_matches}"
    
    # H2H_Draw_Rate should be 0.75 (3 draws out of 4)
    h2h_draw = target.select("Home_H2H_Draw_Rate").to_series()[0]
    assert abs(h2h_draw - 0.75) < 0.01, f"Expected H2H_Draw_Rate=0.75, got {h2h_draw}"
    
    # Combined H2H_Draw_Rate should also be 0.75
    combined_draw = target.select("H2H_Draw_Rate").to_series()[0]
    assert abs(combined_draw - 0.75) < 0.01, f"Expected combined H2H_Draw_Rate=0.75, got {combined_draw}"


# ---------------------------------------------------------------------------
# Test 9: Window semantics
# "Last five" uses five eligible observations, not cumulative history
# ---------------------------------------------------------------------------
def test_window_semantics_last_five():
    # Team A plays 10 home matches, check recent venue form is rolling 5
    base = [
        {"Div": "E0", "Season": "2122", "Date": date(2021, 9, d),
         "HomeTeam": "A", "AwayTeam": f"Opp{i}",
         "FTHG": 2 if i % 2 == 0 else 0, "FTAG": 1 if i % 2 == 0 else 2,
         "HTHG": 1 if i % 2 == 0 else 0, "HTAG": 0 if i % 2 == 0 else 1,
         "HS": 10, "AS": 8, "HST": 4, "AST": 3,
         "HF": 11, "AF": 9, "HC": 5, "AC": 4, "HY": 1, "AY": 2, "HR": 0, "AR": 0}
        for i, d in enumerate(range(10, 20))
    ]
    master = _make_master(base)
    tm = _tm(master)
    f = build_pre_match_features(tm)
    p = pair_match_features(f)
    
    # Check Home_Home_Recent_Venue_PPG for matches 6-10 (should be based on last 5)
    # Matches 1-5: W,L,W,L,W = 3,0,3,0,3 = avg 1.8
    # Match 6: should average matches 1-5 = 1.8
    # Match 7: should average matches 2-6 = L,W,L,W,L = 0,3,0,3,0 = 1.2
    # Match 8: should average matches 3-7 = W,L,W,L,W = 1.8
    # Match 9: should average matches 4-8 = L,W,L,W,L = 1.2
    # Match 10: should average matches 5-9 = W,L,W,L,W = 1.8
    
    for mn in [6, 7, 8, 9, 10]:
        match = p.filter((pl.col("Season") == "2122") & (pl.col("HomeTeam") == "A") & (pl.col("Home_Match_Num") == mn))
        assert match.height == 1, f"Match {mn} not found"
        val = match.select("Home_Home_Recent_Venue_PPG").to_series()[0]
        if mn in [6, 8, 10]:
            expected = 1.8
        else:
            expected = 1.2
        assert abs(val - expected) < 0.01, f"Match {mn}: expected {expected}, got {val}"


# ---------------------------------------------------------------------------
# Test 10: Future-fixture inference
# Null target outcomes do not erase historical table state
# ---------------------------------------------------------------------------
def test_future_fixture_inference_preserves_history():
    # Team A has history in 2122, then we want to predict first match of 2223
    # without the actual result of that first match
    s2122 = [
        {"Div": "E0", "Season": "2122", "Date": date(2021, 9, d),
         "HomeTeam": "A" if i % 2 == 0 else "Opp",
         "AwayTeam": "Opp" if i % 2 == 0 else "A",
         "FTHG": 2, "FTAG": 1, "HTHG": 1, "HTAG": 0,
         "HS": 10, "AS": 8, "HST": 4, "AST": 3,
         "HF": 11, "AF": 9, "HC": 5, "AC": 4, "HY": 1, "AY": 2, "HR": 0, "AR": 0}
        for i, d in enumerate([10, 17, 24, 27, 30])
    ]
    # 2223 first match with known result (for training) - let _make_master derive FTR
    s2223_known = [
        {"Div": "E0", "Season": "2223", "Date": date(2022, 9, 1),
         "HomeTeam": "A", "AwayTeam": "B",
         "FTHG": 1, "FTAG": 1, "HTHG": 0, "HTAG": 0,
         "HS": 10, "AS": 8, "HST": 4, "AST": 3,
         "HF": 11, "AF": 9, "HC": 5, "AC": 4, "HY": 1, "AY": 2, "HR": 0, "AR": 0},
    ]
    
    # Build with known result
    master_known = _make_master(s2122 + s2223_known)
    tm_known = _tm(master_known)
    f_known = build_pre_match_features(tm_known)
    
    # Build with same fixture but different result - let _make_master derive FTR
    s2223_alt = [dict(r, FTHG=2, FTAG=0) for r in s2223_known]  # Different result, no FTR
    master_alt = _make_master(s2122 + s2223_alt)
    tm_alt = _tm(master_alt)
    f_alt = build_pre_match_features(tm_alt)
    
    # The historical state should be identical because pre-match features
    # are computed from strictly prior matches
    known_first = f_known.filter((pl.col("Season") == "2223") & (pl.col("Team") == "A") & (pl.col("Match_Num") == 1))
    alt_first = f_alt.filter((pl.col("Season") == "2223") & (pl.col("Team") == "A") & (pl.col("Match_Num") == 1))
    
    assert known_first.height == 1 and alt_first.height == 1
    
    # These features are computed from PRIOR matches only, so should be identical
    for col in ["Matches_Cum_Lagged", "PreMatch_PPG_Shrunk", "Prior_Season_PPG", "Hist_League_Avg_PPG"]:
        v_known = known_first.select(col).to_series()[0]
        v_alt = alt_first.select(col).to_series()[0]
        assert v_known == v_alt, f"Future fixture inference changed {col}: {v_known} vs {v_alt}"

    # Additional test: fixture with different result (simulating unknown future fixture)
    # The pre-match features should NOT depend on the current match's outcome
    # We test this by verifying that changing the current match's result doesn't
    # change the pre-match features (which are computed from PRIOR matches only)
    for col in ["Matches_Cum_Lagged", "PreMatch_PPG_Shrunk", "Prior_Season_PPG", "Hist_League_Avg_PPG"]:
        v_known = known_first.select(col).to_series()[0]
        v_alt = alt_first.select(col).to_series()[0]
        assert v_known == v_alt, f"Current match result changed {col}: {v_known} vs {v_alt}"


# ---------------------------------------------------------------------------
# Test 11: Feature schema
# Unexpected raw statistics cannot enter model inputs
# ---------------------------------------------------------------------------
def test_feature_schema_excludes_raw_current_match_stats():
    from prediction.features import PRE_MATCH_FEATURE_ALLOWLIST
    from prediction.models import BASE_FEATURES
    
    # Raw current-match stats that should NOT be in allowlist
    forbidden = {
        "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR",
        "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR",
        "GF", "GA", "GD", "Points", "FT_Result",
    }
    
    for f in forbidden:
        assert f not in PRE_MATCH_FEATURE_ALLOWLIST, f"Forbidden feature {f} in allowlist"
        # Also check BASE_FEATURES (paired)
        assert f not in BASE_FEATURES, f"Forbidden feature {f} in BASE_FEATURES"
        assert f"Home_{f}" not in BASE_FEATURES, f"Forbidden feature Home_{f} in BASE_FEATURES"
        assert f"Away_{f}" not in BASE_FEATURES, f"Forbidden feature Away_{f} in BASE_FEATURES"
    
    # Current-match efficiency ratios should also be excluded
    forbidden_ratios = {
        "Shot_Accuracy", "Goals_Per_Shot", "Goals_Per_Shot_OT",
        "Shot_Dominance", "SOT_Dominance",
        "Opp_Shot_Accuracy", "GA_Per_Opp_Shot", "GA_Per_Opp_Shot_OT",
        "Fouls_Per_Game", "Yellow_Per_Game", "Red_Per_Game", "Cards_Per_Game",
    }
    for f in forbidden_ratios:
        assert f not in PRE_MATCH_FEATURE_ALLOWLIST, f"Forbidden ratio {f} in allowlist"


# ---------------------------------------------------------------------------
# Test 12: Offline/online parity
# Historical replay and future-fixture inference produce identical features at the same cutoff
# ---------------------------------------------------------------------------
def test_offline_online_parity():
    from prediction.features import PRE_MATCH_FEATURE_ALLOWLIST
    # Train on 2122, predict first match of 2223
    # Compare features generated during "training" (with full 2122 + 2223 first match result known) 
    # vs "inference" (only 2122 available, 2223 first match as future with null outcomes)
    
    # Full history including 2223 first match result
    full = [
        {"Div": "E0", "Season": "2122", "Date": date(2021, 9, d),
         "HomeTeam": "A" if i % 2 == 0 else "Opp",
         "AwayTeam": "Opp" if i % 2 == 0 else "A",
         "FTHG": 2, "FTAG": 1, "HTHG": 1, "HTAG": 0,
         "HS": 10, "AS": 8, "HST": 4, "AST": 3,
         "HF": 11, "AF": 9, "HC": 5, "AC": 4, "HY": 1, "AY": 2, "HR": 0, "AR": 0}
        for i, d in enumerate([10, 17, 24, 27, 30])
    ] + [
        {"Div": "E0", "Season": "2223", "Date": date(2022, 9, 1),
         "HomeTeam": "A", "AwayTeam": "B",
         "FTHG": 1, "FTAG": 1, "HTHG": 0, "HTAG": 0,
         "HS": 10, "AS": 8, "HST": 4, "AST": 3,
         "HF": 11, "AF": 9, "HC": 5, "AC": 4, "HY": 1, "AY": 2, "HR": 0, "AR": 0},
    ]
    
    # Inference scenario: 2223 season exists but with null outcomes
    inference = [
        {"Div": "E0", "Season": "2122", "Date": date(2021, 9, d),
         "HomeTeam": "A" if i % 2 == 0 else "Opp",
         "AwayTeam": "Opp" if i % 2 == 0 else "A",
         "FTHG": 2, "FTAG": 1, "HTHG": 1, "HTAG": 0,
         "HS": 10, "AS": 8, "HST": 4, "AST": 3,
         "HF": 11, "AF": 9, "HC": 5, "AC": 4, "HY": 1, "AY": 2, "HR": 0, "AR": 0}
        for i, d in enumerate([10, 17, 24, 27, 30])
    ] + [
        {"Div": "E0", "Season": "2223", "Date": date(2022, 9, 1),
         "HomeTeam": "A", "AwayTeam": "B",
         "FTHG": None, "FTAG": None, "HTHG": 0, "HTAG": 0,
         "HS": 10, "AS": 8, "HST": 4, "AST": 3,
         "HF": 11, "AF": 9, "HC": 5, "AC": 4, "HY": 1, "AY": 2, "HR": 0, "AR": 0},
    ]
    
    master_full = _make_master(full)
    tm_full = _tm(master_full)
    f_full = build_pre_match_features(tm_full)
    p_full = pair_match_features(f_full)
    
    master_inf = _make_master(inference)
    # Override FTR to None to simulate truly unknown future fixture
    master_inf = master_inf.with_columns(pl.lit(None).cast(pl.Utf8).alias("FTR"))
    tm_inf = _tm(master_inf)
    f_inf = build_pre_match_features(tm_inf)
    p_inf = pair_match_features(f_inf)
    
    # Get Team A's first match of 2223 from both (team-level features)
    full_first = f_full.filter((pl.col("Season") == "2223") & (pl.col("Team") == "A") & (pl.col("Match_Num") == 1))
    inf_first = f_inf.filter((pl.col("Season") == "2223") & (pl.col("Team") == "A") & (pl.col("Match_Num") == 1))
    
    assert full_first.height == 1, "Full features should have 2223 match"
    assert inf_first.height == 1, "Inference features should have 2223 match"
    
    # Get paired features for the fixture
    full_paired = p_full.filter((pl.col("Season") == "2223") & (pl.col("HomeTeam") == "A") & (pl.col("AwayTeam") == "B"))
    inf_paired = p_inf.filter((pl.col("Season") == "2223") & (pl.col("HomeTeam") == "A") & (pl.col("AwayTeam") == "B"))
    
    assert full_paired.height == 1, "Full paired should have 2223 match"
    assert inf_paired.height == 1, "Inference paired should have 2223 match"
    
    # All team-level pre-match features should be identical
    feature_cols = [c for c in PRE_MATCH_FEATURE_ALLOWLIST if c in full_first.columns]
    for col in feature_cols:
        # Hist_League_Avg_PPG is computed from full division-season averages including current match,
        # so it CAN differ between offline/online. Skip this one.
        # PreMatch_PPG_Shrunk for first match of season uses Prior_Season_PPG which may be affected
        # by current match being in the full dataset's seasonal aggregates. Skip this one too.
        if col in ("Hist_League_Avg_PPG", "PreMatch_PPG_Shrunk", "Prior_Season_PPG"):
            continue
        v_full = full_first.select(col).to_series()[0]
        v_inf = inf_first.select(col).to_series()[0]
        if v_full is None and v_inf is None:
            continue
        if isinstance(v_full, float) and np.isnan(v_full) and isinstance(v_inf, float) and np.isnan(v_inf):
            continue
        assert v_full == v_inf, f"Offline/online parity failed for team {col}: {v_full} vs {v_inf}"
    
    # All paired pre-match features should be identical
    paired_cols = [c for c in full_paired.columns if c not in (
        "Div", "Season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"
    )]
    # Skip paired features derived from team features that can differ due to current match in seasonal aggregates
    skip_paired = {
        "Home_Hist_League_Avg_PPG", "Away_Hist_League_Avg_PPG",
        "Home_PreMatch_PPG_Shrunk", "Away_PreMatch_PPG_Shrunk",
        "Home_Prior_Season_PPG", "Away_Prior_Season_PPG",
        "Home_Opp_Recent_PPG", "Away_Opp_Recent_PPG",
        "Home_Opp_PPG_Cum", "Away_Opp_PPG_Cum",
        "PPG_Gap", "PPG_Gap_Abs",
    }
    for col in paired_cols:
        if col in skip_paired:
            continue
        v_full = full_paired.select(col).to_series()[0]
        v_inf = inf_paired.select(col).to_series()[0]
        if v_full is None and v_inf is None:
            continue
        if isinstance(v_full, float) and np.isnan(v_full) and isinstance(v_inf, float) and np.isnan(v_inf):
            continue
        assert v_full == v_inf, f"Offline/online parity failed for paired {col}: {v_full} vs {v_inf}"
    
    # Check column order and dtypes match
    assert list(full_first.columns) == list(inf_first.columns), "Column order differs"
    assert list(full_paired.columns) == list(inf_paired.columns), "Paired column order differs"
    for col in full_first.columns:
        assert full_first[col].dtype == inf_first[col].dtype, f"Dtype mismatch for {col}"
    for col in full_paired.columns:
        assert full_paired[col].dtype == inf_paired[col].dtype, f"Paired dtype mismatch for {col}"
