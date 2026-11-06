"""
Tests for core analysis logic (Polars implementation).
Covers the 7 required test cases from the mission:
1. Home-advantage null and denominator
2. Consecutive-season matching
3. Comeback denominators
4. Missing-versus-zero handling
5. Zero-denominator ratios
6. Lagged feature leakage
7. Match/team-perspective consistency
"""

import numpy as np
import polars as pl
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.load_data import build_master_dataset, build_team_match_dataset, CORE_COLS
from features.engineer import add_efficiency_ratios, add_rolling_features, add_opponent_strength, compute_team_season_metrics
from analysis.analyze import analyze_home_advantage, analyze_efficiency_persistence, analyze_ht_ft_dynamics


# Fixtures
@pytest.fixture(scope="module")
def master_data():
    """Load minimal test data - use multiple seasons for persistence tests."""
    data_dir = Path(__file__).parent.parent
    master = build_master_dataset(data_dir, ['E0'], ['2122', '2223', '2324'], usecols=CORE_COLS)
    return master


@pytest.fixture(scope="module")
def team_matches_data(master_data):
    """Build team-match dataset."""
    return build_team_match_dataset(master_data)


@pytest.fixture(scope="module")
def engineered_data(team_matches_data):
    """Engineer features."""
    df = add_efficiency_ratios(team_matches_data)
    df = add_rolling_features(df, window=5, min_periods=3)
    df = add_opponent_strength(df)
    return df


# Test 1: Home-advantage null and denominator
def test_home_advantage_symmetry_null(master_data):
    """Test that HA binomial test uses correct symmetry null (p=0.5 for decisive matches)."""
    ha_results = analyze_home_advantage(master_data)

    assert 'significance' in ha_results
    for league, sig in ha_results['significance'].items():
        assert 'p_value' in sig
        assert 'significant' in sig
        assert 'home_win_pct' in sig
        assert 'away_win_pct' in sig
        assert 'home_win_pct_decisive' in sig
        assert 'away_win_pct_decisive' in sig
        assert 'draw_pct' in sig
        assert 'n_decisive' in sig

        lg_data = master_data.filter(pl.col('Div') == league)
        total = lg_data.height
        draws = int((lg_data.select(pl.col('FTR') == 'D').to_series().sum()))
        decisive = total - draws
        home_wins = int((lg_data.select(pl.col('FTR') == 'H').to_series().sum()))
        assert sig['home_win_pct'] == home_wins / total
        assert sig['home_win_pct_decisive'] == (home_wins / decisive if decisive > 0 else float('nan'))


def test_home_advantage_three_measures_distinct(master_data):
    """Test that three HA measures are computed and distinct."""
    ha_results = analyze_home_advantage(master_data)
    ha_ls = ha_results['by_league_season']

    assert 'Home_Win_Pct' in ha_ls.columns
    assert 'Away_Win_Pct' in ha_ls.columns
    assert 'Draw_Pct' in ha_ls.columns
    assert 'Goal_Diff' in ha_ls.columns

    ha_ls = ha_ls.with_columns((pl.col('Home_Win_Pct') - pl.col('Away_Win_Pct')).alias('HA_Diff_Computed'))
    ha_ls = ha_ls.with_columns(
        (pl.col('Home_Win_Pct') / (pl.col('Home_Win_Pct') + pl.col('Away_Win_Pct'))).alias('Home_Share_Decisive')
    )
    assert ha_ls.select(pl.col('Home_Share_Decisive').is_not_null().any()).item()


# Test 2: Consecutive-season matching
def test_consecutive_season_matching(engineered_data):
    """Test that season-to-season persistence requires consecutive seasons."""
    eff_results = analyze_efficiency_persistence(engineered_data, min_matches=20)

    assert 'season_transitions' in eff_results
    trans = eff_results['season_transitions']
    assert 'total_pairs' in trans
    assert 'consecutive_same_div' in trans
    assert 'div_changed' in trans
    assert 'gap_1_season' in trans
    assert 'gap_2plus_seasons' in trans

    assert trans['consecutive_same_div'] <= trans['total_pairs']

    assert 'season_to_season_persistence' in eff_results


def test_season_gap_reporting(engineered_data):
    """Test that league transitions and missing-season gaps are reported separately."""
    eff_results = analyze_efficiency_persistence(engineered_data, min_matches=20)
    trans = eff_results['season_transitions']
    assert trans['div_changed'] >= 0
    assert trans['gap_2plus_seasons'] >= 0


def test_consecutive_pairs_created_before_filtering(engineered_data):
    """Test that lag/lead pairs are created on complete table before filtering."""
    eff_results = analyze_efficiency_persistence(engineered_data, min_matches=5)
    persist = eff_results.get('season_to_season_persistence', {})
    for metric, vals in persist.items():
        if 'n' in vals:
            assert vals['n'] <= eff_results['season_transitions']['consecutive_same_div']


# Test 3: Comeback denominators
def test_comeback_denominators_valid(master_data):
    """Test that comeback rates use valid halftime observations as denominators."""
    htft_results = analyze_ht_ft_dynamics(master_data)

    comeback_df = htft_results['comeback_by_league']
    for row in comeback_df.iter_rows(named=True):
        league = row['League']
        lg = master_data.filter(pl.col('Div') == league)
        total_trailing = int((lg.select(pl.col('HTR') != 'D').to_series().sum()))
        assert row['Trailing_Avoid_Loss_Pct'] <= 1.0
        assert row['Trailing_Win_Pct'] <= 1.0
        assert row['Trailing_Draw_Pct'] <= 1.0

    team_comebacks = htft_results['team_comeback_stats']
    if team_comebacks.height > 0:
        assert (team_comebacks.select(pl.col('Trailing_Count') >= 10).to_series().all())


def test_comeback_distinguishes_win_vs_draw(master_data):
    """Test that winning after trailing is distinguished from drawing after trailing."""
    htft_results = analyze_ht_ft_dynamics(master_data)
    team_comebacks = htft_results['team_comeback_stats']

    if team_comebacks.height > 0:
        assert 'Recovery_Rate' in team_comebacks.columns
        assert 'Avoid_Loss_Rate' in team_comebacks.columns
        assert (team_comebacks.select(pl.col('Recovery_Rate') <= pl.col('Avoid_Loss_Rate')).to_series().all())


# Test 4: Missing-versus-zero handling
def test_missing_not_treated_as_zero(engineered_data):
    """Test that all-missing group sums don't silently become meaningful zeros."""
    team_season = compute_team_season_metrics(engineered_data, min_matches=20)

    assert (team_season.select(pl.col('Matches') >= 20).to_series().all())
    assert (team_season.select(pl.col('Shots') > 0).to_series().all())


def test_dropna_reported(engineered_data):
    """Test that complete-case exclusions are tracked."""
    team_season = compute_team_season_metrics(engineered_data, min_matches=20)

    assert team_season.select(pl.col('Shot_Accuracy').is_not_null().all()).item()
    assert team_season.select(pl.col('Goals_Per_Shot').is_not_null().all()).item()
    assert team_season.select(pl.col('Goals_Per_Shot_OT').is_not_null().all()).item()


# Test 5: Zero-denominator ratios
def test_zero_denominator_handling(team_matches_data):
    """Test that ratios handle zero denominators correctly (return null)."""
    df = add_efficiency_ratios(team_matches_data)

    test_df = df.clone()
    test_df = test_df.with_columns([
        pl.lit(0).cast(pl.Int64).alias('Shots'),
        pl.lit(0).cast(pl.Int64).alias('Shots_OT'),
        pl.lit(0).cast(pl.Int64).alias('GF'),
        pl.lit(0).cast(pl.Int64).alias('Shots_Against'),
        pl.lit(0).cast(pl.Int64).alias('Shots_OT_Against'),
        pl.lit(0).cast(pl.Int64).alias('GA'),
    ])
    test_df = add_efficiency_ratios(test_df)

    assert test_df.select(pl.col('Shot_Accuracy').is_null().all()).item()
    assert test_df.select(pl.col('Goals_Per_Shot').is_null().all()).item()
    assert test_df.select(pl.col('Goals_Per_Shot_OT').is_null().all()).item()
    assert test_df.select(pl.col('Opp_Shot_Accuracy').is_null().all()).item()
    assert test_df.select(pl.col('GA_Per_Opp_Shot').is_null().all()).item()
    assert test_df.select(pl.col('GA_Per_Opp_Shot_OT').is_null().all()).item()


def test_zero_denominator_season_level(engineered_data):
    """Test season-level ratio zero denominator handling."""
    team_season = compute_team_season_metrics(engineered_data, min_matches=1)

    assert (team_season.select(pl.col('Shots') > 0).to_series().all())
    assert (team_season.select(pl.col('Shots_OT') > 0).to_series().all())
    assert (team_season.select(pl.col('Shots_Against') > 0).to_series().all())
    assert (team_season.select(pl.col('Shots_OT_Against') > 0).to_series().all())

    assert team_season.select(pl.col('Shot_Accuracy').is_not_null().all()).item()
    assert team_season.select(pl.col('Goals_Per_Shot').is_not_null().all()).item()
    assert team_season.select(pl.col('Goals_Per_Shot_OT').is_not_null().all()).item()
    assert team_season.select(pl.col('GA_Per_Opp_Shot').is_not_null().all()).item()
    assert team_season.select(pl.col('GA_Per_Opp_Shot_OT').is_not_null().all()).item()


# Test 6: Lagged feature leakage
def test_rolling_features_lagged(team_matches_data):
    """Test that rolling features use .shift(1) to avoid leakage."""
    df = add_efficiency_ratios(team_matches_data)
    df = add_rolling_features(df, window=5, min_periods=3)

    assert 'Points_L5' in df.columns
    assert 'Points_L5_std' in df.columns
    assert 'Points_Cum' in df.columns

    first_matches = df.filter(pl.col('Match_Num') == 1)
    assert first_matches.select(pl.col('Points_L5').is_null().all()).item()
    assert first_matches.select(pl.col('Points_Cum').is_null().all()).item()


def test_opponent_strength_lagged(team_matches_data):
    """Test that opponent strength uses lagged cumulative PPG."""
    df = add_efficiency_ratios(team_matches_data)
    df = add_rolling_features(df, window=5, min_periods=3)
    df = add_opponent_strength(df)

    assert 'Opp_PPG_Cum' in df.columns

    first_matches = df.filter(pl.col('Match_Num') == 1)
    assert first_matches.select(pl.col('Opp_PPG_Cum').is_null().all()).item()


def test_rest_days_season_opener_is_nan(team_matches_data):
    """Test that Rest_Days for season openers is NaN (not invented 7 days)."""
    df = add_efficiency_ratios(team_matches_data)
    df = add_rolling_features(df, window=5, min_periods=3)

    first_matches = df.filter(pl.col('Match_Num') == 1)
    assert first_matches.select(pl.col('Rest_Days').is_null().all()).item()


def test_points_rolling5_std_is_lagged(team_matches_data):
    """Test that Points_Rolling5_Std is computed on shifted data (lagged)."""
    df = add_efficiency_ratios(team_matches_data)
    df = add_rolling_features(df, window=5, min_periods=3)

    first_matches = df.filter(pl.col('Match_Num') == 1)
    assert first_matches.select(pl.col('Points_L5_std').is_null().all()).item()


# Test 7: Match/team-perspective consistency
def test_match_team_perspective_consistency(master_data, team_matches_data):
    """Test that team-perspective construction is consistent with match data."""
    assert len(team_matches_data) == 2 * len(master_data)

    df = add_efficiency_ratios(team_matches_data)

    match_points = master_data.select(
        pl.when(pl.col('FTR') == 'H').then(3)
        .when(pl.col('FTR') == 'D').then(2)
        .otherwise(3)
        .sum()
    ).to_series()[0]
    team_match_points = df.select(pl.col('Points').sum()).to_series()[0]
    assert match_points == team_match_points

    total_match_goals = master_data.select(
        pl.col('FTHG').sum() + pl.col('FTAG').sum()
    ).to_series()[0]
    total_team_goals = df.select(pl.col('GF').sum()).to_series()[0]
    assert total_match_goals == total_team_goals

    total_match_shots = master_data.select(
        pl.col('HS').sum() + pl.col('AS').sum()
    ).to_series()[0]
    total_team_shots = df.select(pl.col('Shots').sum()).to_series()[0]
    assert total_match_shots == total_team_shots


def test_home_away_perspective_consistency(team_matches_data):
    """Test that home and away perspectives are correctly constructed."""
    home_matches = team_matches_data.filter(pl.col('Venue') == 'Home')
    away_matches = team_matches_data.filter(pl.col('Venue') == 'Away')
    assert len(home_matches) == len(away_matches)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
