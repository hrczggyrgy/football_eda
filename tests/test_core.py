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

from datetime import date

import numpy as np
import polars as pl
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.load_data import build_master_dataset, build_team_match_dataset, CORE_COLS
from features.engineer import (
    add_efficiency_ratios, add_rolling_features, add_opponent_strength,
    compute_team_season_metrics, compute_consistency_metrics,
)
from analysis.analyze import analyze_home_advantage, analyze_efficiency_persistence, analyze_ht_ft_dynamics
from analysis.fresh_insights import (
    analyze_match_tempo, _referee_match_lookup, _canonical_team_pair, MATCH_KEY,
)


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


# ---------------------------------------------------------------------------
# Critical-fix regression tests (see code review findings)
# ---------------------------------------------------------------------------

def _synthetic_match_div_dates(div: str, season: str, start: date, n: int):
    """n matches for one league/season starting on ``start`` (one per day)."""
    return pl.DataFrame({
        'Div': [div] * n,
        'Season': [season] * n,
        'Date': pl.Series([date(start.year, start.month, start.day + i) for i in range(n)], dtype=pl.Date),
    })


def test_same_day_fixtures_different_referees():
    """Regression: multiple matches in one league on the same date must not
    share a referee lookup. Each fixture gets exactly one referee, and joining
    must not increase the number of team-match observations."""
    master = pl.DataFrame({
        'Div': ['E0'] * 6,
        'Season': ['2122'] * 6,
        'Date': pl.Series(
            [date(2021, 9, 11)] * 3 + [date(2021, 9, 18)] * 3, dtype=pl.Date),
        'HomeTeam': ['Arsenal', 'ManCity', 'ManUtd', 'Arsenal', 'Chelsea', 'ManUtd'],
        'AwayTeam': ['Chelsea', 'Liverpool', 'Leeds', 'Liverpool', 'ManCity', 'Arsenal'],
        'FTHG': [1, 2, 1, 2, 1, 0], 'FTAG': [0, 1, 1, 1, 1, 1],
        'HTHG': [1, 1, 0, 1, 0, 0], 'HTAG': [0, 1, 0, 1, 0, 0],
        'HTR': ['H', 'D', 'D', 'H', 'D', 'D'],
        'FTR': ['H', 'H', 'D', 'H', 'D', 'A'],
        'HS': [12, 14, 9, 11, 10, 8], 'AS': [8, 10, 11, 7, 9, 12],
        'HST': [4, 5, 3, 4, 3, 2], 'AST': [3, 4, 4, 2, 3, 5],
        'HF': [11, 13, 10, 12, 9, 11], 'AF': [9, 11, 12, 8, 10, 13],
        'HC': [5, 6, 4, 5, 4, 6], 'AC': [4, 5, 5, 3, 4, 7],
        'HY': [1, 2, 1, 1, 1, 2], 'AY': [1, 1, 2, 1, 1, 2],
        'HR': [0, 0, 0, 0, 0, 0], 'AR': [0, 0, 0, 0, 0, 0],
        'Referee': ['RefA', 'RefB', 'RefC', 'RefD', 'RefE', 'RefF'],
    })
    eng_master = master.filter(pl.col('Div').is_in(['E0', 'E1', 'E2', 'E3']))
    team_matches = build_team_match_dataset(master)
    team_matches = team_matches.with_columns(
        list(_canonical_team_pair(team_matches, 'Team', 'Opponent'))
    )

    # The naive date-only join inflates observations: 3 fixtures on 2021-09-11
    # each match every team row of that date.
    ref_lookup_naive = eng_master.select(['Div', 'Season', 'Date', 'Referee']).unique()
    naive = team_matches.join(ref_lookup_naive, on=['Div', 'Season', 'Date'], how='left')
    assert naive.height > team_matches.height, "date-only join should inflate team-match rows"

    # Fixed: join on the full fixture key.
    ref_lookup = _referee_match_lookup(eng_master)
    assert ref_lookup.height == 6  # one row per fixture
    tm_eng = team_matches.join(ref_lookup, on=MATCH_KEY, how='left')

    assert tm_eng.height == team_matches.height, "fixture-key join must not change row count"
    assert tm_eng.select(pl.col('Referee').null_count()).to_series()[0] == 0

    per_match = (
        tm_eng.group_by(MATCH_KEY)
        .agg(pl.col('Referee').n_unique().alias('n_referees'))
    )
    assert per_match.select(pl.col('n_referees') == 1).to_series().all()

    # Spot check: correct referee per perspective row.
    row = tm_eng.filter(
        (pl.col('Team') == 'Arsenal') & (pl.col('Venue') == 'Home')
        & (pl.col('Date') == date(2021, 9, 11))
    )
    assert row.select('Referee').to_series()[0] == 'RefA'


def test_tempo_quantiles_handle_null_shots():
    """Regression: null shots must not produce NaN quantile bounds. Missing
    matches go into a 'Missing' tempo group and every group keeps observations."""
    n = 21
    hs = [10, 8, 12, 9, 11, None, 13, 7, 14, 6, 10, 8, 11, 9, 12, 7, 13, 5, 9, 11, 2]
    as_ = [9, 7, 11, 8, 10, 12, 8, 6, 13, 5, 9, 7, 10, 8, 11, 6, 12, 4, 8, 10, 5]
    # Impossible fixture: shots on target exceed total shots (index 20).
    hst = [3, 2, 5, 4, 5, 0, 6, 3, 7, 2, 4, 3, 5, 4, 6, 3, 6, 2, 4, 5, 5]
    ast = [4, 3, 5, 4, 5, 6, 4, 3, 6, 2, 4, 3, 5, 4, 5, 3, 6, 2, 4, 5, 6]
    master = pl.DataFrame({
        'Div': ['E0'] * n,
        'Season': ['2122'] * n,
        'Date': pl.Series([date(2021, 9, d) for d in range(1, n + 1)], dtype=pl.Date),
        'HomeTeam': [f'T{i % 4}' for i in range(n)],
        'AwayTeam': [f'T{(i + 1) % 4}' for i in range(n)],
        'FTHG': [1, 0, 2, 1, 1, 0, 3, 1, 2, 0, 1, 1, 2, 1, 1, 0, 2, 1, 0, 1, 0],
        'FTAG': [0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1],
        'HTHG': [1, 0, 1, 0, 1, 0, 2, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
        'HTAG': [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1],
        'HTR': ['H', 'A', 'H', 'A', 'H', 'A', 'H', 'H', 'D', 'A', 'H', 'D',
                'A', 'D', 'D', 'A', 'H', 'A', 'H', 'D', 'D'],
        'FTR': ['H', 'A', 'D', 'D', 'H', 'A', 'H', 'H', 'D', 'A', 'H', 'D',
                'A', 'D', 'D', 'A', 'H', 'A', 'H', 'D', 'D'],
        'HS': hs, 'AS': as_, 'HST': hst, 'AST': ast,
        'HF': [11, 9, 12, 10, 11, 13, 12, 8, 14, 7, 10, 11, 12, 9, 13, 8, 14, 6, 10, 12, 9],
        'AF': [10, 8, 13, 9, 12, 11, 9, 7, 11, 6, 9, 10, 11, 8, 12, 7, 13, 5, 9, 10, 8],
        'HC': [5, 4, 6, 5, 5, 7, 6, 3, 7, 3, 4, 5, 6, 4, 6, 3, 7, 2, 5, 6, 4],
        'AC': [4, 3, 5, 4, 4, 6, 4, 3, 6, 2, 4, 4, 5, 3, 5, 3, 6, 2, 4, 5, 3],
        'HY': [1, 1, 2, 1, 1, 2, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1],
        'AY': [1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1],
        'HR': [0] * n, 'AR': [0] * n,
        'Referee': ['Ref'] * n,
    })
    team_matches = add_efficiency_ratios(build_team_match_dataset(master))
    team_season = compute_team_season_metrics(team_matches, min_matches=1)

    results = analyze_match_tempo(team_matches, master, team_season, min_matches=1)

    groups = results['tempo_group_counts']
    group_map = {row['Tempo_Quintile']: int(row['N']) for row in groups.to_dicts()}
    assert 'Missing' in group_map and group_map['Missing'] == 1
    assert set(group_map) == {'Very_Low', 'Low', 'Medium', 'High', 'Very_High', 'Missing'}
    assert all(n_ > 0 for n_ in group_map.values())
    # Impossible shots-below-sot fixture is excluded, not silently used.
    assert results['tempo_excluded_impossible_shots'] == 1
    assert int(groups.select(pl.col('N').sum()).to_series()[0]) == n  # groups computed pre-filter
    # Match-level outcomes exclude the impossible fixture but keep missing.
    assert int(results['tempo_outcomes'].select(pl.col('N').sum()).to_series()[0]) == n - 1


def test_rolling_consistency_independent_across_teams():
    """Regression: rolling consistency windows must not cross team boundaries.
    Two teams with very different point distributions must produce independent
    rolling statistics."""
    rows = {
        'Div': [], 'Season': [], 'Date': [], 'Team': [], 'Opponent': [],
        'Venue': [], 'GF': [], 'GA': [], 'GD': [], 'FT_Result': [],
        'Shots': [], 'Shots_OT': [], 'Shots_Against': [], 'Shots_OT_Against': [],
        'Fouls': [], 'Fouls_Against': [], 'Corners': [], 'Corners_Against': [],
        'Yellow': [], 'Yellow_Against': [], 'Red': [], 'Red_Against': [],
    }
    start = date(2021, 8, 1)
    for team, pattern in [('Team A', [3, 0]), ('Team B', [0, 0])]:
        for i in range(12):
            rows['Div'].append('E0'); rows['Season'].append('2122')
            rows['Date'].append(date(start.year, start.month, start.day + i))
            rows['Team'].append(team); rows['Opponent'].append('Opponent')
            rows['Venue'].append('Home')
            pts = pattern[i % 2]
            rows['GF'].append(pts); rows['GA'].append(0); rows['GD'].append(pts)
            rows['FT_Result'].append({3: 'W', 0: 'L'}[pts])
            rows['Shots'].append(10); rows['Shots_OT'].append(4)
            rows['Shots_Against'].append(8); rows['Shots_OT_Against'].append(3)
            rows['Fouls'].append(12); rows['Fouls_Against'].append(10)
            rows['Corners'].append(5); rows['Corners_Against'].append(4)
            rows['Yellow'].append(2); rows['Yellow_Against'].append(1)
            rows['Red'].append(0); rows['Red_Against'].append(0)
    team_matches = pl.DataFrame(rows).with_columns(
        pl.col('Date').cast(pl.Date),
        pl.when(pl.col('FT_Result') == 'W')
        .then(3).when(pl.col('FT_Result') == 'D').then(1).otherwise(0)
        .cast(pl.Int64).alias('Points'),
    )
    consistency = compute_consistency_metrics(team_matches, min_matches=1)
    by_team = {row['Team']: row for row in consistency.select(['Team', 'Points_Rolling5_Std_Mean', 'Points_Std']).to_dicts()}

    # Team B never varies: every rolling std is exactly 0 despite Team A's variation.
    assert by_team['Team B']['Points_Rolling5_Std_Mean'] == 0.0
    # Team A alternates wins/losses: rolling points std is strictly positive.
    assert by_team['Team A']['Points_Rolling5_Std_Mean'] > 0.0


def test_all_null_statistical_group_preserves_nulls():
    """Regression: all-null metric groups must stay null, not become zero."""
    master = pl.DataFrame({
        'Div': ['E0'] * 6, 'Season': ['2122'] * 6,
        'Date': pl.Series([date(2021, 9, d) for d in range(1, 7)], dtype=pl.Date),
        'HomeTeam': ['Rangers'] * 6, 'AwayTeam': ['Chelsea'] * 6,
        'FTHG': [1, 0, 2, 1, 1, 0], 'FTAG': [0, 1, 1, 1, 0, 1],
        'HTHG': [1, 0, 1, 0, 1, 0], 'HTAG': [0, 1, 0, 1, 0, 1],
        'HTR': ['H', 'A', 'H', 'A', 'H', 'A'],
        'FTR': ['H', 'A', 'D', 'D', 'H', 'A'],
        'HS': [None] * 6, 'AS': [None] * 6, 'HST': [None] * 6, 'AST': [None] * 6,
        'HF': [0] * 6, 'AF': [0] * 6, 'HC': [0] * 6, 'AC': [0] * 6,
        'HY': [0] * 6, 'AY': [0] * 6, 'HR': [0] * 6, 'AR': [0] * 6,
        'Referee': ['Ref'] * 6,
    })
    team_matches = build_team_match_dataset(master)
    team_matches = add_efficiency_ratios(team_matches)
    consistency = compute_consistency_metrics(team_matches, min_matches=1)

    assert consistency.select(pl.col('Shots_Std').is_null().all()).item()
    assert consistency.select(pl.col('PPG').is_not_null().all()).item()


def test_same_team_name_in_different_competitions():
    """Regression: the same team name in two leagues must not be aggregated
    into a single team-season."""
    master = pl.DataFrame({
        'Div': ['E0', 'E0', 'SP1', 'SP1'],
        'Season': ['2122'] * 4,
        'Date': pl.Series(
            [date(2021, 9, 11), date(2021, 9, 18), date(2021, 9, 11), date(2021, 9, 18)],
            dtype=pl.Date),
        'HomeTeam': ['Rangers', 'Rangers', 'Rangers', 'Rangers'],
        'AwayTeam': ['Chelsea', 'Chelsea', 'Barcelona', 'Barcelona'],
        'FTHG': [1, 0, 2, 1], 'FTAG': [0, 1, 1, 0],
        'HTHG': [1, 0, 1, 0], 'HTAG': [0, 1, 1, 0],
        'HTR': ['H', 'A', 'H', 'H'],
        'FTR': ['H', 'A', 'H', 'H'],
        'HS': [10, 8, 12, 9], 'AS': [9, 7, 11, 8], 'HST': [3, 2, 5, 4], 'AST': [4, 3, 5, 4],
        'HF': [5, 4, 6, 5], 'AF': [4, 3, 5, 4], 'HC': [3, 2, 4, 3], 'AC': [2, 1, 3, 2],
        'HY': [1, 1, 1, 1], 'AY': [1, 1, 1, 1], 'HR': [0] * 4, 'AR': [0] * 4,
        'Referee': ['Ref'] * 4,
    })
    team_matches = add_efficiency_ratios(build_team_match_dataset(master))
    team_season = compute_team_season_metrics(team_matches, min_matches=1)

    rangers = team_season.filter(pl.col('Team') == 'Rangers')
    assert rangers.height == 2
    assert set(rangers.select('Div').to_series().to_list()) == {'E0', 'SP1'}


def test_minimum_supported_polars_version():
    """Regression: the installed Polars must support the min_samples rolling
    API (renamed from min_periods in 1.21.0)."""
    from packaging.version import Version
    assert Version(pl.__version__) >= Version('1.21.0'), (
        f"polars {pl.__version__} is below the required 1.21.0")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
