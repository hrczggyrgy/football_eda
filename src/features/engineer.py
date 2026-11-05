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
    """Add lagged rolling and expanding features per team-season."""
    df = df.sort_values(['Div', 'Team', 'Season', 'Date']).copy()
    
    # Match number in season
    df['Match_Num'] = df.groupby(['Div', 'Team', 'Season']).cumcount() + 1
    
    # Rest days
    df['Rest_Days'] = df.groupby(['Div', 'Team', 'Season'])['Date'].diff().dt.days
    df['Rest_Days'] = df['Rest_Days'].fillna(7)
    
    # Group key for vectorized rolling
    df['_group_key'] = df.groupby(['Div', 'Team', 'Season']).ngroup()
    
    for col in ROLLING_COLS:
        if col not in df.columns:
            continue
        
        # Shift by 1 to avoid leakage
        shifted = df.groupby(['Div', 'Team', 'Season'])[col].shift(1)
        
        # Rolling mean/std
        rolled = shifted.groupby(df['_group_key']).rolling(window=window, min_periods=min_periods)
        df[f'{col}_L{window}'] = rolled.mean().reset_index(level=0, drop=True)
        df[f'{col}_L{window}_std'] = rolled.std().reset_index(level=0, drop=True)
        
        # Cumulative expanding mean (lagged)
        cum = shifted.groupby(df['_group_key']).expanding(min_periods=1).mean()
        df[f'{col}_Cum'] = cum.reset_index(level=0, drop=True)
    
    df = df.drop(columns=['_group_key'])
    return df


def add_opponent_strength(df: pd.DataFrame) -> pd.DataFrame:
    """Add opponent's season-to-date PPG before this match (lagged)."""
    df = df.sort_values(['Div', 'Team', 'Season', 'Date']).copy()
    
    # Compute each team's cumulative PPG before each match
    df['_group_key'] = df.groupby(['Div', 'Team', 'Season']).ngroup()
    
    # Points cumulative mean (lagged)
    points_cum = df.groupby(['Div', 'Team', 'Season'])['Points'].apply(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    ).reset_index(level=[0,1,2], drop=True)
    df['Points_Cum'] = points_cum
    
    # For each match, get opponent's Points_Cum
    # Create lookup: (Div, Season, Date, Opponent) -> Points_Cum
    opp_lookup = df.set_index(['Div', 'Season', 'Date', 'Team'])['Points_Cum']
    
    df['Opp_PPG_Cum'] = df.apply(
        lambda row: opp_lookup.get((row['Div'], row['Season'], row['Date'], row['Opponent']), np.nan),
        axis=1
    )
    
    df = df.drop(columns=['_group_key', 'Points_Cum'])
    return df


def compute_team_season_metrics(team_matches: pd.DataFrame, min_matches: int = 20) -> pd.DataFrame:
    """Aggregate team-match data to team-season level with derived metrics."""
    agg_dict = {
        'GF': 'sum', 'GA': 'sum', 'GD': 'sum',
        'Points': 'sum',
        'Shots': 'sum', 'Shots_OT': 'sum',
        'Shots_Against': 'sum', 'Shots_OT_Against': 'sum',
        'Yellow': 'sum', 'Red': 'sum', 'Fouls': 'sum',
        'Corners': 'sum', 'Corners_Against': 'sum',
        'Match_Num': 'max'  # total matches
    }
    
    team_season = team_matches.groupby(['Div', 'Team', 'Season']).agg(**{
        k: (k, v) for k, v in agg_dict.items()
    }).reset_index()
    
    team_season = team_season.rename(columns={'Match_Num': 'Matches'})
    team_season = team_season[team_season['Matches'] >= min_matches].copy()
    
    # Per-game rates
    team_season['PPG'] = team_season['Points'] / team_season['Matches']
    team_season['GF_PG'] = team_season['GF'] / team_season['Matches']
    team_season['GA_PG'] = team_season['GA'] / team_season['Matches']
    team_season['GD_PG'] = team_season['GD'] / team_season['Matches']
    team_season['Shots_PG'] = team_season['Shots'] / team_season['Matches']
    team_season['Shots_OT_PG'] = team_season['Shots_OT'] / team_season['Matches']
    team_season['Shots_Against_PG'] = team_season['Shots_Against'] / team_season['Matches']
    
    # Efficiency ratios
    team_season['Shot_Accuracy'] = team_season['Shots_OT'] / team_season['Shots']
    team_season['Opp_Shot_Accuracy'] = team_season['Shots_OT_Against'] / team_season['Shots_Against']
    team_season['Goals_Per_Shot'] = team_season['GF'] / team_season['Shots']
    team_season['Goals_Per_Shot_OT'] = team_season['GF'] / team_season['Shots_OT']
    team_season['GA_Per_Opp_Shot'] = team_season['GA'] / team_season['Shots_Against']
    team_season['GA_Per_Opp_Shot_OT'] = team_season['GA'] / team_season['Shots_OT_Against']
    
    # Discipline per game
    team_season['Yellow_PG'] = team_season['Yellow'] / team_season['Matches']
    team_season['Red_PG'] = team_season['Red'] / team_season['Matches']
    team_season['Fouls_PG'] = team_season['Fouls'] / team_season['Matches']
    team_season['Cards_PG'] = team_season['Yellow_PG'] + team_season['Red_PG']
    
    return team_season


def compute_consistency_metrics(team_matches: pd.DataFrame, min_matches: int = 10) -> pd.DataFrame:
    """Compute within-season consistency metrics (CV of match-level stats)."""
    team_matches = team_matches.sort_values(['Div', 'Team', 'Season', 'Date'])
    
    # Rolling 5-game points std (lagged)
    team_matches['_group_key'] = team_matches.groupby(['Div', 'Team', 'Season']).ngroup()
    team_matches['Points_Rolling5_Std'] = team_matches.groupby(['Div', 'Team', 'Season'])['Points'].transform(
        lambda x: x.rolling(5, min_periods=3).std()
    )
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