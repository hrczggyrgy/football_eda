"""
Feature engineering module.
Creates efficiency ratios, rolling form metrics, temporal features,
and team-season aggregate metrics from team-perspective match data.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


# Columns to compute rolling features for
ROLLING_COLS = [
    'GF', 'GA', 'GD', 'Points', 'Shots', 'Shots_OT', 
    'Shots_Against', 'Shots_OT_Against',
    'Shot_Accuracy', 'Goals_Per_Shot', 'Goals_Per_Shot_OT',
    'Opp_Shot_Accuracy', 'GA_Per_Opp_Shot', 'GA_Per_Opp_Shot_OT',
    'Corners', 'Corners_Against', 'Yellow', 'Red', 'Fouls'
]


def add_efficiency_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Add match-level efficiency ratios with zero-denominator handling."""
    df = df.copy()
    
    # Attacking efficiency
    df['Shot_Accuracy'] = np.where(df['Shots'] > 0, df['Shots_OT'] / df['Shots'], np.nan)
    df['Goals_Per_Shot'] = np.where(df['Shots'] > 0, df['GF'] / df['Shots'], np.nan)
    df['Goals_Per_Shot_OT'] = np.where(df['Shots_OT'] > 0, df['GF'] / df['Shots_OT'], np.nan)
    df['Shot_Dominance'] = df['Shots'] - df['Shots_Against']
    df['SOT_Dominance'] = df['Shots_OT'] - df['Shots_OT_Against']
    
    # Defensive efficiency
    df['Opp_Shot_Accuracy'] = np.where(df['Shots_Against'] > 0, 
                                        df['Shots_OT_Against'] / df['Shots_Against'], np.nan)
    df['GA_Per_Opp_Shot'] = np.where(df['Shots_Against'] > 0, 
                                      df['GA'] / df['Shots_Against'], np.nan)
    df['GA_Per_Opp_Shot_OT'] = np.where(df['Shots_OT_Against'] > 0, 
                                         df['GA'] / df['Shots_OT_Against'], np.nan)
    
    # Discipline
    df['Fouls_Per_Game'] = df['Fouls']
    df['Yellow_Per_Game'] = df['Yellow']
    df['Red_Per_Game'] = df['Red']
    df['Cards_Per_Game'] = df['Yellow'] + df['Red']
    
    # Points
    result_map = {'W': 3, 'D': 1, 'L': 0}
    df['Points'] = df['FT_Result'].map(result_map)
    
    return df


def add_rolling_features(df: pd.DataFrame, window: int = 5, min_periods: int = 3) -> pd.DataFrame:
    """Add lagged rolling and expanding features per team-season.
    
    FIXED: 
    - Rest_Days for season openers is NaN (not invented 7 days)
    - Points_Rolling5_Std is properly lagged (shifted before rolling)
    - All rolling features use shift(1) to avoid leakage
    """
    df = df.sort_values(['Div', 'Team', 'Season', 'Date']).copy()
    
    # Match number in season
    df['Match_Num'] = df.groupby(['Div', 'Team', 'Season']).cumcount() + 1
    
    # Rest days - NaN for season openers (unknown, not invented)
    df['Rest_Days'] = df.groupby(['Div', 'Team', 'Season'])['Date'].diff().dt.days
    # Do NOT fillna(7) - season opener rest is genuinely unknown
    
    # Group key for vectorized rolling
    df['_group_key'] = df.groupby(['Div', 'Team', 'Season']).ngroup()
    
    # Shift every rolling column by 1 in a single groupby pass to avoid
    # lookahead leakage. Doing this once (rather than once per column)
    # avoids repeating the groupby setup for each of ~19 columns.
    shift_cols = [c for c in ROLLING_COLS if c in df.columns]
    shifted = df.groupby(['Div', 'Team', 'Season'])[shift_cols].shift(1)
    
    for col in shift_cols:
        s = shifted[col]
        
        # Rolling mean/std on SHIFTED data (no leakage)
        rolled = s.groupby(df['_group_key']).rolling(window=window, min_periods=min_periods)
        df[f'{col}_L{window}'] = rolled.mean().reset_index(level=0, drop=True)
        df[f'{col}_L{window}_std'] = rolled.std().reset_index(level=0, drop=True)
        
        # Cumulative expanding mean (lagged)
        cum = s.groupby(df['_group_key']).expanding(min_periods=1).mean()
        df[f'{col}_Cum'] = cum.reset_index(level=0, drop=True)
    
    df = df.drop(columns=['_group_key'])
    return df


def add_opponent_strength(df: pd.DataFrame) -> pd.DataFrame:
    """Add opponent's season-to-date PPG before this match (lagged).
    
    The opponent's pre-match cumulative PPG is looked up via a vectorized
    merge on (Div, Season, Date, Opponent) instead of a slow row-wise
    DataFrame.apply(axis=1) dictionary lookup. Results are identical.
    """
    df = df.sort_values(['Div', 'Team', 'Season', 'Date']).copy()
    
    # Pre-match (lagged) cumulative PPG for each team in each match.
    # cumsum - current points = points earned before this match;
    # cumcount = number of prior matches.
    grp = ['Div', 'Team', 'Season']
    points_cum = df.groupby(grp)['Points'].cumsum()
    matches_cum = df.groupby(grp).cumcount()
    pre_match_ppg = np.where(
        matches_cum > 0,
        (points_cum - df['Points'].values) / matches_cum,
        np.nan,
    )
    
    # Build a lookup keyed by (Div, Season, Date, Team) and merge so each
    # match row picks up its opponent's pre-match PPG.
    lookup = pd.DataFrame({
        'Div': df['Div'].values,
        'Season': df['Season'].values,
        'Date': df['Date'].values,
        'Opponent': df['Team'].values,
        'Opp_PPG_Cum': pre_match_ppg,
    })
    df = df.merge(lookup, on=['Div', 'Season', 'Date', 'Opponent'], how='left')
    
    return df


def compute_team_season_metrics(team_matches: pd.DataFrame, min_matches: int = 20) -> pd.DataFrame:
    """Aggregate team-match data to team-season level with derived metrics.
    
    FIXED: Uses metric-specific valid-match exposure for per-game rates.
    Missing values in source data produce NaN (not 0) in sums via skipna=False.
    """
    # Track which metrics have valid data per match
    metric_cols = ['GF', 'GA', 'GD', 'Points', 'Shots', 'Shots_OT', 
                   'Shots_Against', 'Shots_OT_Against',
                   'Yellow', 'Red', 'Fouls', 'Corners', 'Corners_Against']
    
    # For each metric, count valid (non-NaN) matches. DataFrameGroupBy.count()
    # already excludes NA values, so this is equivalent to notna().sum() but
    # fully vectorized (avoids a Python row-wise apply per group).
    valid_counts = team_matches.groupby(['Div', 'Team', 'Season'])[metric_cols].count().reset_index()
    valid_counts.columns = ['Div', 'Team', 'Season'] + [f'{c}_Valid' for c in metric_cols]
    
    # Sums with skipna=False to preserve NaN for all-missing groups
    sums = team_matches.groupby(['Div', 'Team', 'Season'])[metric_cols].sum(skipna=False).reset_index()
    
    # Match count (total rows, regardless of missingness)
    match_counts = team_matches.groupby(['Div', 'Team', 'Season']).size().reset_index(name='Matches')
    
    # Merge
    team_season = match_counts.merge(sums, on=['Div', 'Team', 'Season'])
    team_season = team_season.merge(valid_counts, on=['Div', 'Team', 'Season'])
    
    team_season = team_season[team_season['Matches'] >= min_matches].copy()
    
    # Per-game rates using METRIC-SPECIFIC valid match exposure
    team_season['PPG'] = team_season['Points'] / team_season['Matches']
    team_season['GF_PG'] = team_season['GF'] / team_season['GF_Valid'].replace(0, np.nan)
    team_season['GA_PG'] = team_season['GA'] / team_season['GA_Valid'].replace(0, np.nan)
    team_season['GD_PG'] = team_season['GD'] / team_season['Matches']
    team_season['Shots_PG'] = team_season['Shots'] / team_season['Shots_Valid'].replace(0, np.nan)
    team_season['Shots_OT_PG'] = team_season['Shots_OT'] / team_season['Shots_OT_Valid'].replace(0, np.nan)
    team_season['Shots_Against_PG'] = team_season['Shots_Against'] / team_season['Shots_Against_Valid'].replace(0, np.nan)
    
    # Efficiency ratios (only where denominators > 0)
    team_season['Shot_Accuracy'] = np.where(
        team_season['Shots_Valid'] > 0, 
        team_season['Shots_OT'] / team_season['Shots'], 
        np.nan
    )
    team_season['Opp_Shot_Accuracy'] = np.where(
        team_season['Shots_Against_Valid'] > 0, 
        team_season['Shots_OT_Against'] / team_season['Shots_Against'], 
        np.nan
    )
    team_season['Goals_Per_Shot'] = np.where(
        team_season['Shots_Valid'] > 0, 
        team_season['GF'] / team_season['Shots'], 
        np.nan
    )
    team_season['Goals_Per_Shot_OT'] = np.where(
        team_season['Shots_OT_Valid'] > 0, 
        team_season['GF'] / team_season['Shots_OT'], 
        np.nan
    )
    team_season['GA_Per_Opp_Shot'] = np.where(
        team_season['Shots_Against_Valid'] > 0, 
        team_season['GA'] / team_season['Shots_Against'], 
        np.nan
    )
    team_season['GA_Per_Opp_Shot_OT'] = np.where(
        team_season['Shots_OT_Against_Valid'] > 0, 
        team_season['GA'] / team_season['Shots_OT_Against'], 
        np.nan
    )
    
    # Discipline per game using metric-specific exposure
    team_season['Yellow_PG'] = team_season['Yellow'] / team_season['Yellow_Valid'].replace(0, np.nan)
    team_season['Red_PG'] = team_season['Red'] / team_season['Red_Valid'].replace(0, np.nan)
    team_season['Fouls_PG'] = team_season['Fouls'] / team_season['Fouls_Valid'].replace(0, np.nan)
    team_season['Cards_PG'] = team_season['Yellow_PG'] + team_season['Red_PG']
    
    return team_season


def compute_consistency_metrics(team_matches: pd.DataFrame, min_matches: int = 10) -> pd.DataFrame:
    """Compute within-season consistency metrics (CV of match-level stats).
    
    FIXED: Points_Rolling5_Std is now properly LAGGED (shifted before rolling).
    """
    team_matches = team_matches.sort_values(['Div', 'Team', 'Season', 'Date'])
    
    # Rolling 5-game points std (LAGGED - shifted by 1 before rolling)
    team_matches['_group_key'] = team_matches.groupby(['Div', 'Team', 'Season']).ngroup()
    team_matches['Points_Rolling5_Std'] = team_matches.groupby(['Div', 'Team', 'Season'])['Points'].apply(
        lambda x: x.shift(1).rolling(5, min_periods=3).std()
    ).reset_index(level=[0,1,2], drop=True)
    team_matches = team_matches.drop(columns=['_group_key'])
    
    # Season-level consistency
    consistency = team_matches.groupby(['Div', 'Team', 'Season']).agg(
        Points_Std=('Points', 'std'),
        Points_Rolling5_Std_Mean=('Points_Rolling5_Std', 'mean'),
        GF_Std=('GF', 'std'),
        GA_Std=('GA', 'std'),
        Shots_Std=('Shots', 'std'),
        GD_Std=('GD', 'std'),
        PPG=('Points', 'mean'),
        GF_PG=('GF', 'mean'),
        GA_PG=('GA', 'mean'),
        Matches=('Points', 'count')
    ).reset_index()
    
    consistency = consistency[consistency['Matches'] >= min_matches].copy()
    
    # Coefficient of variation
    consistency['Points_CV'] = consistency['Points_Std'] / consistency['PPG']
    consistency['GF_CV'] = consistency['GF_Std'] / consistency['GF_PG']
    consistency['GA_CV'] = consistency['GA_Std'] / consistency['GA_PG']
    
    return consistency


def add_season_order(team_season: pd.DataFrame) -> pd.DataFrame:
    """Add numeric season order for consecutive season analysis."""
    season_order = {
        '1920': 0, '2021': 1, '2122': 2, '2223': 3,
        '2324': 4, '2425': 5, '2526': 6, '2627': 7
    }
    team_season = team_season.copy()
    team_season['Season_Num'] = team_season['Season'].map(season_order)
    team_season = team_season.sort_values(['Div', 'Team', 'Season_Num'])
    return team_season


def add_next_season_metrics(team_season: pd.DataFrame, 
                            metrics: List[str] = ['GPS', 'GPSOT', 'GAOS', 'PPG']) -> pd.DataFrame:
    """Add next-season metrics for persistence analysis."""
    team_season = team_season.copy()
    
    # Ensure required base metrics exist
    if 'GPS' not in team_season.columns:
        team_season['GPS'] = team_season['GF'] / team_season['Shots']
    if 'GPSOT' not in team_season.columns:
        team_season['GPSOT'] = team_season['GF'] / team_season['Shots_OT']
    if 'GAOS' not in team_season.columns:
        team_season['GAOS'] = team_season['GA'] / team_season['Shots_Against']
    
    for metric in metrics:
        if metric in team_season.columns:
            team_season[f'Next_{metric}'] = team_season.groupby(['Div', 'Team'])[metric].shift(-1)
    
    return team_season


def cluster_team_profiles(team_season: pd.DataFrame, 
                          features: List[str],
                          n_clusters: int = 5,
                          random_state: int = 42) -> pd.DataFrame:
    """K-means clustering of team-season profiles."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    team_season = team_season.copy()
    profile_data = team_season[features].dropna()
    
    if len(profile_data) < n_clusters:
        team_season['Cluster'] = np.nan
        return team_season
    
    scaler = StandardScaler()
    profile_scaled = scaler.fit_transform(profile_data)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    team_season.loc[profile_data.index, 'Cluster'] = kmeans.fit_predict(profile_scaled)
    
    return team_season