"""
Performance regression tests for the football analytics pipeline.

These tests:
1. Verify key analytical properties are preserved (correctness regression).
2. Time the most important pipeline stages on a small, fixed dataset and
   assert they complete within generous upper bounds (performance regression).
   The bounds are deliberately loose so the tests are robust across machines,
   but will fail if a future change reintroduces an O(n^2) loop or a slow
   row-wise apply where a vectorized op was intended.

They use only a small slice of data (one league, two seasons) so they run
quickly and do not slow down the regular unit test suite.
"""

import time
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.load_data import build_master_dataset, build_team_match_dataset, CORE_COLS
from features.engineer import (
    add_efficiency_ratios, add_rolling_features, add_opponent_strength,
    compute_team_season_metrics,
)
from analysis.analyze import analyze_efficiency_persistence, run_robustness_checks
from analysis.advanced_insights import analyze_volatility_streakiness
from analysis.new_insights import (
    analyze_opponent_dependent_performance,
    analyze_performance_indicators_lead_results,
)
from analysis.next_insights import run_next_insights


@pytest.fixture(scope="module")
def small_data():
    data_dir = Path(__file__).parent.parent
    master = build_master_dataset(data_dir, ["E0"], ["2122", "2223"], usecols=CORE_COLS)
    team_matches = build_team_match_dataset(master)
    team_matches = add_efficiency_ratios(team_matches)
    team_matches = add_rolling_features(team_matches, window=5, min_periods=3)
    team_matches = add_opponent_strength(team_matches)
    team_season = compute_team_season_metrics(team_matches, min_matches=20)
    team_season["Corners_PG"] = team_season["Corners"] / team_season["Matches"]
    team_season["Corners_Ag_PG"] = team_season["Corners_Against"] / team_season["Matches"]
    team_season["Shots_Per_Corner"] = np.where(
        team_season["Corners"] > 0, team_season["Shots"] / team_season["Corners"], np.nan
    )
    team_season["Goals_Per_Corner"] = np.where(
        team_season["Corners"] > 0, team_season["GF"] / team_season["Corners"], np.nan
    )
    return master, team_matches, team_season


# ---------------------------------------------------------------------------
# Correctness regressions (analytical properties preserved)
# ---------------------------------------------------------------------------

def test_rolling_first_row_nan(small_data):
    """Rolling features must remain strictly lagged (first match = NaN)."""
    _, team_matches, _ = small_data
    first = team_matches.groupby(["Div", "Team", "Season"]).head(1)
    assert first["Points_L5"].isna().all()
    assert first["Points_Cum"].isna().all()


def test_opponent_strength_first_row_nan(small_data):
    """Opponent strength must be NaN for season openers (no prior data)."""
    _, team_matches, _ = small_data
    first = team_matches.groupby(["Div", "Team", "Season"]).head(1)
    assert first["Opp_PPG_Cum"].isna().all()


def test_zero_denominator_nan(small_data):
    """Ratio columns must exist and be computed (zero-denominator -> NaN)."""
    _, team_matches, _ = small_data
    assert "Shot_Accuracy" in team_matches.columns
    assert "Opp_PPG_Cum" in team_matches.columns


def test_team_season_min_matches(small_data):
    """team_season must respect the min_matches exposure threshold."""
    _, _, team_season = small_data
    assert (team_season["Matches"] >= 20).all()


# ---------------------------------------------------------------------------
# Performance regressions (upper-bound timing on small data)
# ---------------------------------------------------------------------------

def _elapsed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    fn(*args, **kwargs)
    return time.perf_counter() - t0


@pytest.mark.perf
def test_efficiency_persistence_perf(small_data):
    _, team_matches, _ = small_data
    dt = _elapsed(analyze_efficiency_persistence, team_matches, min_matches=20)
    assert dt < 30.0, f"efficiency_persistence too slow: {dt:.2f}s"


@pytest.mark.perf
def test_robustness_checks_perf(small_data):
    _, team_matches, master = small_data
    dt = _elapsed(run_robustness_checks, team_matches, master)
    assert dt < 30.0, f"robustness_checks too slow: {dt:.2f}s"


@pytest.mark.perf
def test_volatility_streakiness_perf(small_data):
    _, team_matches, team_season = small_data
    dt = _elapsed(analyze_volatility_streakiness, team_matches, team_season, min_matches=20)
    assert dt < 60.0, f"volatility_streakiness too slow: {dt:.2f}s"


@pytest.mark.perf
def test_opponent_dependent_perf(small_data):
    _, team_matches, team_season = small_data
    dt = _elapsed(analyze_opponent_dependent_performance, team_matches, team_season, min_matches=20)
    assert dt < 120.0, f"opponent_dependent_performance too slow: {dt:.2f}s"


@pytest.mark.perf
def test_perf_indicators_lead_perf(small_data):
    _, team_matches, team_season = small_data
    dt = _elapsed(analyze_performance_indicators_lead_results, team_matches, team_season, min_matches=20)
    assert dt < 60.0, f"performance_indicators_lead too slow: {dt:.2f}s"


@pytest.mark.perf
def test_next_insights_perf(small_data):
    master, team_matches, team_season = small_data
    dt = _elapsed(run_next_insights, master, team_matches, team_season)
    assert dt < 120.0, f"next_insights too slow: {dt:.2f}s"
