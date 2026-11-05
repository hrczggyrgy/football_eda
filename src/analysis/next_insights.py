# Next Insights Analysis Module
# Implements 5 new research questions beyond existing analyses

import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
warnings.filterwarnings('ignore', category=UserWarning, message='X does not have valid feature names')

SEASON_ORDER = {
    '1920': 0, '2021': 1, '2122': 2, '2223': 3,
    '2324': 4, '2425': 5, '2526': 6, '2627': 7
}


def analyze_competitive_balance(team_season: pd.DataFrame) -> Dict:
    """
    Question 1: How has competitive balance evolved across leagues and seasons?
    
    - Herfindahl-Hirschman Index (HHI) of points distribution
    - Gini coefficient of PPG
    - Top-4 vs Bottom-4 PPG gap
    - Top-4 share of total points
    - League-season trends with uncertainty
    """
    results = {}
    
    # Compute balance metrics per league-season
    balance_data = []
    for (div, season), group in team_season.groupby(['Div', 'Season']):
        ppg = group['PPG'].values
        n = len(group)
        if n < 4:
            continue
            
        # Herfindahl-Hirschman Index (normalized)
        shares = ppg / ppg.sum()
        hhi = (shares**2).sum()
        hhi_norm = (hhi - 1/n) / (1 - 1/n) if n > 1 else 0
        
        # Gini coefficient
        sorted_ppg = np.sort(ppg)
        cumsum = np.cumsum(sorted_ppg)
        gini = (2 * np.sum((np.arange(1, n+1) * sorted_ppg))) / (n * cumsum[-1]) - (n+1)/n
        
        # Top 4 vs Bottom 4 gap
        top4 = np.mean(np.sort(ppg)[-4:])
        bot4 = np.mean(np.sort(ppg)[:4])
        gap = top4 - bot4
        
        # Points concentration: top 4 share
        top4_share = np.sum(np.sort(ppg)[-4:]) / np.sum(ppg)
        
        balance_data.append({
            'Div': div, 'Season': season, 'N': n,
            'HHI': hhi, 'HHI_Norm': hhi_norm, 'Gini': gini,
            'Top4_PPG': top4, 'Bot4_PPG': bot4, 'Gap': gap,
            'Top4_Share': top4_share,
            'Mean_PPG': ppg.mean(), 'Std_PPG': ppg.std(),
        })
    
    balance_df = pd.DataFrame(balance_data)
    results['league_season_balance'] = balance_df
    
    # League averages
    league_avg = balance_df.groupby('Div').agg({
        'HHI_Norm': 'mean', 'Gini': 'mean', 'Gap': 'mean', 
        'Top4_Share': 'mean', 'Mean_PPG': 'mean', 'Std_PPG': 'mean'
    }).round(4).reset_index()
    results['league_averages'] = league_avg
    
    # Trends per league (linear regression across seasons)
    league_trends = {}
    for div in balance_df['Div'].unique():
        lg = balance_df[balance_df['Div'] == div].sort_values('Season')
        if len(lg) >= 3:
            season_nums = np.arange(len(lg))
            for metric in ['HHI_Norm', 'Gini', 'Gap', 'Top4_Share']:
                v = lg[metric].notna()
                if v.sum() >= 3:
                    slope, intercept, r, p, se = stats.linregress(season_nums[v], lg.loc[v, metric])
                    league_trends[f'{div}_{metric}'] = {
                        'slope_per_season': slope,
                        'intercept': intercept,
                        'r': r, 'p': p, 'se': se,
                        'n_seasons': int(v.sum()),
                        'significant': p < 0.05
                    }
    results['league_trends'] = league_trends
    
    # Overall trend (match-weighted)
    overall_trend = {}
    for metric in ['HHI_Norm', 'Gini', 'Gap', 'Top4_Share']:
        v = balance_df[metric].notna()
        if v.sum() >= 3:
            slope, intercept, r, p, se = stats.linregress(np.arange(v.sum()), balance_df.loc[v, metric])
            overall_trend[metric] = {
                'slope_per_season': slope, 'p': p, 'r': r, 'n': int(v.sum())
            }
    results['overall_trend'] = overall_trend
    
    # Bootstrap confidence intervals for trend slopes
    bootstrap_trends = {}
    n_boot = 1000
    for metric in ['HHI_Norm', 'Gini', 'Gap', 'Top4_Share']:
        v = balance_df[metric].notna()
        if v.sum() >= 3:
            df_valid = balance_df.loc[v].copy()
            slopes = []
            for _ in range(n_boot):
                sample = df_valid.sample(n=len(df_valid), replace=True)
                slope, _, _, _, _ = stats.linregress(np.arange(len(sample)), sample[metric])
                slopes.append(slope)
            bootstrap_trends[metric] = {
                'slope_mean': np.mean(slopes),
                'slope_std': np.std(slopes),
                'ci_95': (np.percentile(slopes, 2.5), np.percentile(slopes, 97.5)),
            }
    results['bootstrap_trends'] = bootstrap_trends
    
    return results


def analyze_league_style_fingerprints(team_season: pd.DataFrame, team_matches: pd.DataFrame) -> Dict:
    """
    Question 2: Do leagues have distinct, persistent statistical fingerprints?
    
    - Aggregate style metrics per league-season (shots, corners, fouls, accuracy ratios)
    - League-season trajectories in style space
    - Clustering of league-seasons
    - Convergence/divergence analysis
    """
    results = {}
    
    # League-season style metrics
    style_metrics = team_season.groupby(['Div', 'Season']).agg(
        Shots_PG=('Shots_PG', 'mean'),
        Shots_OT_PG=('Shots_OT_PG', 'mean'),
        Shot_Accuracy=('Shot_Accuracy', 'mean'),
        Goals_Per_Shot=('Goals_Per_Shot', 'mean'),
        Goals_Per_Shot_OT=('Goals_Per_Shot_OT', 'mean'),
        Corners_PG=('Corners_PG', 'mean'),
        Corners_Ag_PG=('Corners_Ag_PG', 'mean'),
        Goals_Per_Corner=('Goals_Per_Corner', 'mean'),
        Fouls_PG=('Fouls_PG', 'mean'),
        Yellow_PG=('Yellow', lambda x: (x/team_season.loc[x.index, 'Matches']).mean()),
        Red_PG=('Red', lambda x: (x/team_season.loc[x.index, 'Matches']).mean()),
        GF_PG=('GF_PG', 'mean'),
        GA_PG=('GA_PG', 'mean'),
    ).reset_index()
    
    # Derived ratios
    style_metrics['Shots_Per_Goal'] = style_metrics['Shots_PG'] / style_metrics['GF_PG']
    style_metrics['Corners_Per_Goal'] = style_metrics['Corners_PG'] / style_metrics['GF_PG']
    style_metrics['Fouls_Per_Goal'] = style_metrics['Fouls_PG'] / style_metrics['GF_PG']
    style_metrics['Shots_Per_Corner'] = style_metrics['Shots_PG'] / style_metrics['Corners_PG']
    style_metrics['Corners_Per_Foul'] = style_metrics['Corners_PG'] / style_metrics['Fouls_PG']
    
    results['league_season_style'] = style_metrics
    
    # League averages across seasons
    league_avg = style_metrics.groupby('Div').agg({
        'Shots_PG': 'mean', 'Shot_Accuracy': 'mean', 'Goals_Per_Shot': 'mean',
        'Corners_PG': 'mean', 'Goals_Per_Corner': 'mean', 'Fouls_PG': 'mean',
        'Shots_Per_Goal': 'mean', 'Corners_Per_Goal': 'mean', 'Fouls_Per_Goal': 'mean',
        'Shots_Per_Corner': 'mean', 'Corners_Per_Foul': 'mean',
    }).round(3).reset_index()
    results['league_averages'] = league_avg
    
    # Trends over seasons
    style_trends = {}
    for div in style_metrics['Div'].unique():
        lg = style_metrics[style_metrics['Div'] == div].sort_values('Season')
        if len(lg) >= 3:
            season_nums = np.arange(len(lg))
            for metric in ['Shots_PG', 'Shot_Accuracy', 'Goals_Per_Shot', 'Corners_PG', 
                          'Goals_Per_Corner', 'Fouls_PG', 'Shots_Per_Goal', 'Fouls_Per_Goal']:
                v = lg[metric].notna()
                if v.sum() >= 3:
                    slope, intercept, r, p, se = stats.linregress(season_nums[v], lg.loc[v, metric])
                    if p < 0.1:
                        style_trends[f'{div}_{metric}'] = {
                            'slope_per_season': slope, 'p': p, 'r': r,
                            'significant': p < 0.05
                        }
    results['style_trends'] = style_trends
    
    # Style distance matrix (league-season pairwise distances)
    style_features = ['Shots_PG', 'Shot_Accuracy', 'Goals_Per_Shot', 'Corners_PG', 
                     'Goals_Per_Corner', 'Fouls_PG', 'Shots_Per_Goal', 'Fouls_Per_Goal']
    
    # Z-score within each feature across all league-seasons
    style_z = style_metrics.copy()
    for feat in style_features:
        mean = style_z[feat].mean()
        std = style_z[feat].std()
        style_z[f'{feat}_z'] = (style_z[feat] - mean) / std if std > 0 else 0
    
    # Pairwise Euclidean distances between league-season centroids
    league_season_centroids = style_z.groupby(['Div', 'Season'])[[f'{f}_z' for f in style_features]].mean()
    
    from scipy.spatial.distance import pdist, squareform
    distances = pdist(league_season_centroids.values, metric='euclidean')
    dist_matrix = squareform(distances)
    dist_df = pd.DataFrame(dist_matrix, index=league_season_centroids.index, columns=league_season_centroids.index)
    results['style_distance_matrix'] = dist_df
    
    # K-means clustering of league-seasons
    from sklearn.cluster import KMeans
    n_clusters = min(4, len(league_season_centroids))
    if len(league_season_centroids) >= n_clusters:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(league_season_centroids)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        style_z['Style_Cluster'] = kmeans.fit_predict(X_scaled)
        results['style_clusters'] = style_z[['Div', 'Season', 'Style_Cluster']].copy()
    
    # Convergence test: are league-seasons becoming more similar over time?
    # Average pairwise distance within each season
    convergence = []
    for season in sorted(style_metrics['Season'].unique()):
        season_data = league_season_centroids.loc[league_season_centroids.index.get_level_values('Season') == season]
        if len(season_data) >= 2:
            from scipy.spatial.distance import pdist
            d = pdist(season_data.values, metric='euclidean')
            convergence.append({'Season': season, 'Avg_Distance': d.mean(), 'N_Leagues': len(season_data)})
    results['convergence'] = pd.DataFrame(convergence)
    
    return results


def analyze_corner_efficiency(team_season: pd.DataFrame, team_matches: pd.DataFrame) -> Dict:
    """
    Question 3: Corner efficiency and set-piece profiles.
    
    - Corner generation (Corners_PG), conversion (Goals_Per_Corner), shot creation (Shots_Per_Corner)
    - Persistence (half-season, season-to-season)
    - Predictive value beyond shots
    - Set-piece specialist identification
    """
    results = {}
    
    # Filter valid corner data
    valid = team_season[team_season['Corners'] > 0].copy()
    
    # Basic correlations with PPG
    corner_corrs = {}
    for metric in ['Corners_PG', 'Goals_Per_Corner', 'Shots_Per_Corner']:
        v = valid[metric].notna() & valid['PPG'].notna()
        if v.sum() > 10:
            r, p = stats.pearsonr(valid.loc[v, metric], valid.loc[v, 'PPG'])
            corner_corrs[metric] = {'r': r, 'p': p, 'n': int(v.sum())}
    results['ppg_correlations'] = corner_corrs
    
    # Incremental value over shots
    # Model 1: PPG ~ Shots_PG
    # Model 2: PPG ~ Shots_PG + Corners_PG
    # Model 3: PPG ~ Shots_PG + Goals_Per_Corner
    valid_reg = valid[['PPG', 'Shots_PG', 'Corners_PG', 'Goals_Per_Corner', 'Shots_Per_Corner']].dropna()
    if len(valid_reg) > 50:
        X1 = valid_reg[['Shots_PG']].values
        y = valid_reg['PPG'].values
        r2_1 = LinearRegression().fit(X1, y).score(X1, y)
        
        X2 = valid_reg[['Shots_PG', 'Corners_PG']].values
        r2_2 = LinearRegression().fit(X2, y).score(X2, y)
        
        X3 = valid_reg[['Shots_PG', 'Goals_Per_Corner']].values
        r2_3 = LinearRegression().fit(X3, y).score(X3, y)
        
        X4 = valid_reg[['Shots_PG', 'Shots_Per_Corner']].values
        r2_4 = LinearRegression().fit(X4, y).score(X4, y)
        
        results['incremental_r2'] = {
            'shots_only': r2_1,
            'shots_plus_corners_pg': r2_2,
            'shots_plus_gpc': r2_3,
            'shots_plus_spc': r2_4,
            'inc_corners_pg': r2_2 - r2_1,
            'inc_gpc': r2_3 - r2_1,
            'inc_spc': r2_4 - r2_1,
        }
    
    # Half-season persistence
    def split_half_corners(group):
        group = group.sort_values('Date')
        n = len(group)
        mid = n // 2
        first = group.iloc[:mid]
        second = group.iloc[mid:]
        
        def calc(df):
            corners = df['Corners'].sum()
            shots = df['Shots'].sum()
            gf = df['GF'].sum()
            return pd.Series({
                'Corners_PG': corners / len(df) if len(df) > 0 else np.nan,
                'Shots_Per_Corner': shots / corners if corners > 0 else np.nan,
                'Goals_Per_Corner': gf / corners if corners > 0 else np.nan,
            })
        
        first_eff = calc(first)
        second_eff = calc(second)
        first_eff.index = [c + '_First' for c in first_eff.index]
        second_eff.index = [c + '_Second' for c in second_eff.index]
        return pd.concat([first_eff, second_eff])
    
    half_corners = team_matches[
        team_matches.groupby(['Div', 'Team', 'Season']).transform('size') >= 20
    ].groupby(['Div', 'Team', 'Season']).apply(split_half_corners).reset_index()
    
    half_persistence = {}
    for metric in ['Corners_PG', 'Goals_Per_Corner', 'Shots_Per_Corner']:
        first_col = f'{metric}_First'
        second_col = f'{metric}_Second'
        v = half_corners[first_col].notna() & half_corners[second_col].notna()
        if v.sum() > 10:
            r, p = stats.pearsonr(half_corners.loc[v, first_col], half_corners.loc[v, second_col])
            half_persistence[metric] = {'r': r, 'p': p, 'n': int(v.sum())}
    results['half_season_persistence'] = half_persistence
    
    # Season-to-season persistence (consecutive seasons, same division)
    team_season_sorted = team_season.sort_values(['Div', 'Team', 'Season']).copy()
    team_season_sorted['Season_Num'] = team_season_sorted['Season'].map(SEASON_ORDER)
    
    for metric in ['Corners_PG', 'Goals_Per_Corner', 'Shots_Per_Corner']:
        team_season_sorted[f'Next_{metric}'] = team_season_sorted.groupby(['Div', 'Team'])[metric].shift(-1)
    team_season_sorted['Next_Season_Num'] = team_season_sorted.groupby(['Div', 'Team'])['Season_Num'].shift(-1)
    team_season_sorted['Next_Div'] = team_season_sorted.groupby(['Div', 'Team'])['Div'].shift(-1)
    
    valid_pairs = (
        team_season_sorted['Next_Season_Num'].notna() & 
        (team_season_sorted['Next_Season_Num'] - team_season_sorted['Season_Num'] == 1) &
        (team_season_sorted['Next_Div'] == team_season_sorted['Div'])
    )
    consecutive = team_season_sorted[valid_pairs].copy()
    
    season_persistence = {}
    for metric in ['Corners_PG', 'Goals_Per_Corner', 'Shots_Per_Corner']:
        v = consecutive[metric].notna() & consecutive[f'Next_{metric}'].notna()
        if v.sum() > 10:
            r, p = stats.pearsonr(consecutive.loc[v, metric], consecutive.loc[v, f'Next_{metric}'])
            season_persistence[metric] = {'r': r, 'p': p, 'n': int(v.sum())}
    results['season_to_season_persistence'] = season_persistence
    
    # Top/bottom specialists (min 50 corners per season)
    min_corners = 50
    specialists = valid[valid['Corners'] >= min_corners].copy()
    
    results['top_gpc'] = specialists.nlargest(15, 'Goals_Per_Corner')[
        ['Div', 'Team', 'Season', 'Corners_PG', 'Goals_Per_Corner', 'Shots_Per_Corner', 'PPG']
    ].to_dict('records')
    results['bottom_gpc'] = specialists.nsmallest(15, 'Goals_Per_Corner')[
        ['Div', 'Team', 'Season', 'Corners_PG', 'Goals_Per_Corner', 'Shots_Per_Corner', 'PPG']
    ].to_dict('records')
    
    # Set-piece profile clustering
    profile_features = ['Corners_PG', 'Goals_Per_Corner', 'Shots_Per_Corner']
    profile_data = specialists[profile_features].dropna()
    if len(profile_data) >= 5:
        scaler = StandardScaler()
        profile_scaled = scaler.fit_transform(profile_data)
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        specialists.loc[profile_data.index, 'SP_Cluster'] = kmeans.fit_predict(profile_scaled)
        
        cluster_summary = specialists.groupby('SP_Cluster').agg(
            N=('Team', 'count'),
            Corners_PG=('Corners_PG', 'mean'),
            Goals_Per_Corner=('Goals_Per_Corner', 'mean'),
            Shots_Per_Corner=('Shots_Per_Corner', 'mean'),
            PPG=('PPG', 'mean'),
        ).round(3)
        results['sp_clusters'] = cluster_summary.to_dict()
    
    return results


def analyze_shooting_profiles(team_season: pd.DataFrame, team_matches: pd.DataFrame) -> Dict:
    """
    Question 4: Selective vs Volume shooting profiles.
    
    - Quadrant analysis: Shots_PG x Shot_Accuracy (league-season adjusted)
    - Profile persistence and transitions
    - PPG by profile
    - Stability of profiles across seasons
    """
    results = {}
    
    # League-season adjusted z-scores
    team_season = team_season.copy()
    team_season['LeagueSeason'] = team_season['Div'] + '_' + team_season['Season']
    
    for metric in ['Shots_PG', 'Shot_Accuracy']:
        ls_mean = team_season.groupby('LeagueSeason')[metric].transform('mean')
        ls_std = team_season.groupby('LeagueSeason')[metric].transform('std')
        team_season[f'{metric}_z'] = (team_season[metric] - ls_mean) / ls_std.replace(0, np.nan)
    
    # Quadrant profiles in z-space
    team_season['Vol_Profile'] = np.where(team_season['Shots_PG_z'] >= 0, 'High_Vol', 'Low_Vol')
    team_season['Acc_Profile'] = np.where(team_season['Shot_Accuracy_z'] >= 0, 'High_Acc', 'Low_Acc')
    team_season['Shooting_Profile_z'] = team_season['Vol_Profile'] + '_' + team_season['Acc_Profile']
    
    # Also raw profiles for comparison
    shots_median = team_season['Shots_PG'].median()
    acc_median = team_season['Shot_Accuracy'].median()
    team_season['Vol_Profile_Raw'] = np.where(team_season['Shots_PG'] >= shots_median, 'High_Vol', 'Low_Vol')
    team_season['Acc_Profile_Raw'] = np.where(team_season['Shot_Accuracy'] >= acc_median, 'High_Acc', 'Low_Acc')
    team_season['Shooting_Profile_Raw'] = team_season['Vol_Profile_Raw'] + '_' + team_season['Acc_Profile_Raw']
    
    # Profile PPG (z-score adjusted)
    profile_ppg_z = team_season.groupby('Shooting_Profile_z').agg(
        N=('Team', 'count'),
        PPG=('PPG', 'mean'),
        GF_PG=('GF_PG', 'mean'),
        Shots_PG=('Shots_PG', 'mean'),
        Shot_Accuracy=('Shot_Accuracy', 'mean'),
        Goals_Per_Shot=('Goals_Per_Shot', 'mean'),
        Goals_Per_Shot_OT=('Goals_Per_Shot_OT', 'mean'),
    ).round(3).reset_index()
    results['profile_ppg_z'] = profile_ppg_z
    
    # Profile PPG (raw)
    profile_ppg_raw = team_season.groupby('Shooting_Profile_Raw').agg(
        N=('Team', 'count'),
        PPG=('PPG', 'mean'),
        GF_PG=('GF_PG', 'mean'),
        Shots_PG=('Shots_PG', 'mean'),
        Shot_Accuracy=('Shot_Accuracy', 'mean'),
        Goals_Per_Shot=('Goals_Per_Shot', 'mean'),
        Goals_Per_Shot_OT=('Goals_Per_Shot_OT', 'mean'),
    ).round(3).reset_index()
    results['profile_ppg_raw'] = profile_ppg_raw
    
    # Transition matrix (consecutive seasons, same division)
    team_season_sorted = team_season.sort_values(['Div', 'Team', 'Season']).copy()
    team_season_sorted['Season_Num'] = team_season_sorted['Season'].map(SEASON_ORDER)
    team_season_sorted['Prev_Profile_z'] = team_season_sorted.groupby(['Div', 'Team'])['Shooting_Profile_z'].shift(1)
    team_season_sorted['Prev_Profile_Raw'] = team_season_sorted.groupby(['Div', 'Team'])['Shooting_Profile_Raw'].shift(1)
    team_season_sorted['Consecutive'] = team_season_sorted.groupby(['Div', 'Team'])['Season_Num'].diff() == 1
    team_season_sorted['Same_Div'] = team_season_sorted.groupby(['Div', 'Team'])['Div'].shift(1) == team_season_sorted['Div']
    team_season_sorted['Valid_Transition'] = team_season_sorted['Consecutive'] & team_season_sorted['Same_Div']
    
    transitions_z = team_season_sorted[team_season_sorted['Valid_Transition']].copy()
    trans_matrix_z = pd.crosstab(
        transitions_z['Prev_Profile_z'], transitions_z['Shooting_Profile_z'],
        normalize='index'
    ).round(3)
    results['transition_matrix_z'] = trans_matrix_z
    
    transitions_raw = team_season_sorted[team_season_sorted['Valid_Transition']].copy()
    trans_matrix_raw = pd.crosstab(
        transitions_raw['Prev_Profile_Raw'], transitions_raw['Shooting_Profile_Raw'],
        normalize='index'
    ).round(3)
    results['transition_matrix_raw'] = trans_matrix_raw
    
    # Profile stability
    stability_z = (transitions_z['Shooting_Profile_z'] == transitions_z['Prev_Profile_z']).mean()
    stability_raw = (transitions_raw['Shooting_Profile_Raw'] == transitions_raw['Prev_Profile_Raw']).mean()
    results['profile_stability'] = {
        'z_score_adjusted': stability_z,
        'raw': stability_raw,
        'n_transitions': len(transitions_z)
    }
    
    # Half-season persistence of profile metrics
    def split_half_shooting(group):
        group = group.sort_values('Date')
        n = len(group)
        mid = n // 2
        first = group.iloc[:mid]
        second = group.iloc[mid:]
        
        def calc(df):
            shots = df['Shots'].sum()
            shots_ot = df['Shots_OT'].sum()
            gf = df['GF'].sum()
            return pd.Series({
                'Shots_PG': shots / len(df) if len(df) > 0 else np.nan,
                'Shot_Accuracy': shots_ot / shots if shots > 0 else np.nan,
                'Goals_Per_Shot': gf / shots if shots > 0 else np.nan,
                'Goals_Per_Shot_OT': gf / shots_ot if shots_ot > 0 else np.nan,
            })
        
        first_eff = calc(first)
        second_eff = calc(second)
        first_eff.index = [c + '_First' for c in first_eff.index]
        second_eff.index = [c + '_Second' for c in second_eff.index]
        return pd.concat([first_eff, second_eff])
    
    half_shooting = team_matches[
        team_matches.groupby(['Div', 'Team', 'Season']).transform('size') >= 20
    ].groupby(['Div', 'Team', 'Season']).apply(split_half_shooting).reset_index()
    
    half_persistence = {}
    for metric in ['Shots_PG', 'Shot_Accuracy', 'Goals_Per_Shot', 'Goals_Per_Shot_OT']:
        first_col = f'{metric}_First'
        second_col = f'{metric}_Second'
        v = half_shooting[first_col].notna() & half_shooting[second_col].notna()
        if v.sum() > 10:
            r, p = stats.pearsonr(half_shooting.loc[v, first_col], half_shooting.loc[v, second_col])
            half_persistence[metric] = {'r': r, 'p': p, 'n': int(v.sum())}
    results['half_season_persistence'] = half_persistence
    
    # Profile-specific persistence: do High_Vol_High_Acc teams stay High_Vol_High_Acc?
    profile_persistence = {}
    for profile in team_season['Shooting_Profile_z'].unique():
        profile_teams = team_season_sorted[team_season_sorted['Shooting_Profile_z'] == profile]
        if len(profile_teams) > 20:
            v = profile_teams['Valid_Transition']
            if v.sum() > 10:
                same = (profile_teams.loc[v, 'Shooting_Profile_z'] == profile_teams.loc[v, 'Prev_Profile_z']).mean()
                profile_persistence[profile] = {'stability': same, 'n': int(v.sum())}
    results['profile_specific_persistence'] = profile_persistence
    
    return results


def analyze_clutch_performance(master: pd.DataFrame, team_matches: pd.DataFrame) -> Dict:
    """
    Question 5: Points from winning/losing positions - "Clutch" performance.
    
    - Points dropped when leading at HT, points gained when trailing at HT
    - Team-season clutch metrics
    - Persistence (early vs late season)
    - Relationship with overall PPG
    - Comparison with close-match dependence (new insight 3)
    """
    results = {}
    
    # Team-perspective HT/FT data
    ht_data = pd.concat([
        master.assign(
            Team=master['HomeTeam'], Venue='Home',
            HT_GD=master['HTHG'] - master['HTAG'], FT_GD=master['FTHG'] - master['FTAG'],
            HT_Result=master['HTR'].map({'H': 'W', 'D': 'D', 'A': 'L'}),
            FT_Result=master['FTR'].map({'H': 'W', 'D': 'D', 'A': 'L'})
        ),
        master.assign(
            Team=master['AwayTeam'], Venue='Away',
            HT_GD=master['HTAG'] - master['HTHG'], FT_GD=master['FTAG'] - master['FTHG'],
            HT_Result=master['HTR'].map({'A': 'W', 'D': 'D', 'H': 'L'}),
            FT_Result=master['FTR'].map({'A': 'W', 'D': 'D', 'H': 'L'})
        )
    ], ignore_index=True)
    
    result_map = {'W': 3, 'D': 1, 'L': 0}
    ht_data['HT_Points'] = ht_data['HT_Result'].map(result_map)
    ht_data['FT_Points'] = ht_data['FT_Result'].map(result_map)
    
    # Overall rates
    leading = ht_data[ht_data['HT_Result'] == 'W']
    drawing = ht_data[ht_data['HT_Result'] == 'D']
    trailing = ht_data[ht_data['HT_Result'] == 'L']
    
    results['overall_rates'] = {
        'leading': {
            'n': len(leading),
            'win_pct': (leading['FT_Result'] == 'W').mean(),
            'draw_pct': (leading['FT_Result'] == 'D').mean(),
            'loss_pct': (leading['FT_Result'] == 'L').mean(),
            'avg_ft_points': leading['FT_Points'].mean(),
            'avg_points_dropped': (3 - leading['FT_Points']).mean(),
        },
        'drawing': {
            'n': len(drawing),
            'win_pct': (drawing['FT_Result'] == 'W').mean(),
            'draw_pct': (drawing['FT_Result'] == 'D').mean(),
            'loss_pct': (drawing['FT_Result'] == 'L').mean(),
            'avg_ft_points': drawing['FT_Points'].mean(),
        },
        'trailing': {
            'n': len(trailing),
            'win_pct': (trailing['FT_Result'] == 'W').mean(),
            'draw_pct': (trailing['FT_Result'] == 'D').mean(),
            'loss_pct': (trailing['FT_Result'] == 'L').mean(),
            'avg_ft_points': trailing['FT_Points'].mean(),
            'avg_points_gained': trailing['FT_Points'].mean(),
        }
    }
    
    # Team-season aggregates
    team_ht = ht_data.groupby(['Div', 'Team', 'Season', 'HT_Result']).agg(
        Matches=('Team', 'count'),
        HT_Points=('HT_Points', 'sum'),
        FT_Points=('FT_Points', 'sum'),
    ).reset_index()
    
    team_ht_pivot = team_ht.pivot_table(
        index=['Div', 'Team', 'Season'], columns='HT_Result',
        values=['Matches', 'HT_Points', 'FT_Points'],
        fill_value=0
    ).reset_index()
    team_ht_pivot.columns = ['_'.join(col).strip('_') for col in team_ht_pivot.columns.values]
    
    for col in ['Matches_W', 'Matches_D', 'Matches_L', 'FT_Points_W', 'FT_Points_D', 'FT_Points_L']:
        if col not in team_ht_pivot.columns:
            team_ht_pivot[col] = 0
    
    team_ht_pivot['Total_Matches'] = team_ht_pivot[['Matches_W', 'Matches_D', 'Matches_L']].sum(axis=1)
    team_ht_pivot['Total_FT_Points'] = team_ht_pivot[['FT_Points_W', 'FT_Points_D', 'FT_Points_L']].sum(axis=1)
    team_ht_pivot['PPG'] = team_ht_pivot['Total_FT_Points'] / team_ht_pivot['Total_Matches']
    
    # Points dropped when leading (per match)
    team_ht_pivot['Points_Dropped_When_Leading'] = (3 * team_ht_pivot['Matches_W'] - team_ht_pivot['FT_Points_W']) / team_ht_pivot['Total_Matches']
    # Points gained when trailing (per match)
    team_ht_pivot['Points_Gained_When_Trailing'] = team_ht_pivot['FT_Points_L'] / team_ht_pivot['Total_Matches']
    # Points from drawing at HT
    team_ht_pivot['Points_From_Drawing'] = team_ht_pivot['FT_Points_D'] / team_ht_pivot['Total_Matches']
    # Clutch score
    team_ht_pivot['Clutch_Score'] = team_ht_pivot['Points_Gained_When_Trailing'] - team_ht_pivot['Points_Dropped_When_Leading']
    
    valid_seasons = team_ht_pivot[team_ht_pivot['Total_Matches'] >= 20].copy()
    results['team_season_clutch'] = valid_seasons
    
    # Correlations with PPG
    clutch_corrs = {}
    for metric in ['Points_Dropped_When_Leading', 'Points_Gained_When_Trailing', 'Clutch_Score', 'Points_From_Drawing']:
        v = valid_seasons[metric].notna() & valid_seasons['PPG'].notna()
        if v.sum() > 10:
            r, p = stats.pearsonr(valid_seasons.loc[v, metric], valid_seasons.loc[v, 'PPG'])
            clutch_corrs[metric] = {'r': r, 'p': p, 'n': int(v.sum())}
    results['ppg_correlations'] = clutch_corrs
    
    # Team-level (3+ seasons)
    team_clutch = valid_seasons.groupby('Team').agg(
        Seasons=('Season', 'count'),
        Avg_PPG=('PPG', 'mean'),
        Avg_Points_Dropped_When_Leading=('Points_Dropped_When_Leading', 'mean'),
        Avg_Points_Gained_When_Trailing=('Points_Gained_When_Trailing', 'mean'),
        Avg_Clutch_Score=('Clutch_Score', 'mean'),
    ).reset_index()
    team_clutch = team_clutch[team_clutch['Seasons'] >= 3].sort_values('Avg_Clutch_Score', ascending=False)
    results['team_clutch'] = team_clutch
    
    # Early vs Late season persistence
    ht_data['Match_Num'] = ht_data.groupby(['Div', 'Team', 'Season']).cumcount() + 1
    ht_data['Season_Half'] = np.where(
        ht_data['Match_Num'] <= ht_data.groupby(['Div', 'Team', 'Season'])['Match_Num'].transform('max') / 2,
        'Early', 'Late'
    )
    
    half_clutch = ht_data.groupby(['Div', 'Team', 'Season', 'Season_Half', 'HT_Result']).agg(
        Matches=('Team', 'count'),
        FT_Points=('FT_Points', 'sum'),
    ).reset_index()
    
    half_pivot = half_clutch.pivot_table(
        index=['Div', 'Team', 'Season', 'Season_Half'], columns='HT_Result',
        values=['Matches', 'FT_Points'], fill_value=0
    ).reset_index()
    half_pivot.columns = ['_'.join(col).strip('_') for col in half_pivot.columns.values]
    
    for col in ['Matches_W', 'Matches_D', 'Matches_L', 'FT_Points_W', 'FT_Points_D', 'FT_Points_L']:
        if col not in half_pivot.columns:
            half_pivot[col] = 0
    
    half_pivot['Total_Matches'] = half_pivot[['Matches_W', 'Matches_D', 'Matches_L']].sum(axis=1)
    half_pivot['Points_Dropped_When_Leading'] = (3 * half_pivot['Matches_W'] - half_pivot['FT_Points_W']) / half_pivot['Total_Matches']
    half_pivot['Points_Gained_When_Trailing'] = half_pivot['FT_Points_L'] / half_pivot['Total_Matches']
    
    early = half_pivot[half_pivot['Season_Half'] == 'Early'][['Div', 'Team', 'Season', 'Points_Dropped_When_Leading', 'Points_Gained_When_Trailing']].rename(
        columns={'Points_Dropped_When_Leading': 'Drop_Early', 'Points_Gained_When_Trailing': 'Gain_Early'})
    late = half_pivot[half_pivot['Season_Half'] == 'Late'][['Div', 'Team', 'Season', 'Points_Dropped_When_Leading', 'Points_Gained_When_Trailing']].rename(
        columns={'Points_Dropped_When_Leading': 'Drop_Late', 'Points_Gained_When_Trailing': 'Gain_Late'})
    
    early_late = early.merge(late, on=['Div', 'Team', 'Season'])
    early_late = early_late[early_late['Drop_Early'].notna() & early_late['Gain_Early'].notna() & 
                            early_late['Drop_Late'].notna() & early_late['Gain_Late'].notna()]
    
    persistence = {}
    if len(early_late) > 20:
        r_drop, p_drop = stats.pearsonr(early_late['Drop_Early'], early_late['Drop_Late'])
        r_gain, p_gain = stats.pearsonr(early_late['Gain_Early'], early_late['Gain_Late'])
        persistence = {
            'points_dropped_when_leading': {'r': r_drop, 'p': p_drop, 'n': len(early_late)},
            'points_gained_when_trailing': {'r': r_gain, 'p': p_gain, 'n': len(early_late)},
        }
    results['early_late_persistence'] = persistence
    results['early_late_data'] = early_late
    
    # By venue
    venue_clutch = ht_data.groupby(['Venue', 'HT_Result']).agg(
        N=('Team', 'count'),
        Avg_FT_Points=('FT_Points', 'mean'),
    ).reset_index()
    results['by_venue'] = venue_clutch
    
    # By deficit size (1 goal vs 2+ goals)
    trailing_detail = trailing.copy()
    trailing_detail['Deficit_Size'] = -trailing_detail['HT_GD']
    trailing_detail['Deficit_Cat'] = np.where(trailing_detail['Deficit_Size'] == 1, 'One_Goal', 'Two_Plus')
    
    deficit_recovery = trailing_detail.groupby(['Deficit_Cat', 'Venue']).agg(
        N=('Team', 'count'),
        Win_Rate=('FT_Result', lambda x: (x == 'W').mean()),
        Draw_Rate=('FT_Result', lambda x: (x == 'D').mean()),
        Avg_Points=('FT_Points', 'mean'),
    ).reset_index()
    results['by_deficit_size'] = deficit_recovery
    
    return results


# ============================================================
# MAIN ORCHESTRATOR
# ============================================================

def run_next_insights(master: pd.DataFrame, team_matches: pd.DataFrame, team_season: pd.DataFrame) -> Dict:
    """Run all 5 next insight analyses."""
    print("  [1/5] Competitive Balance Evolution...")
    balance = analyze_competitive_balance(team_season)
    
    print("  [2/5] League Style Fingerprints...")
    style = analyze_league_style_fingerprints(team_season, team_matches)
    
    print("  [3/5] Corner Efficiency & Set-Piece Profiles...")
    corners = analyze_corner_efficiency(team_season, team_matches)
    
    print("  [4/5] Selective vs Volume Shooting Profiles...")
    shooting = analyze_shooting_profiles(team_season, team_matches)
    
    print("  [5/5] Clutch Performance (Points from Win/Loss Positions)...")
    clutch = analyze_clutch_performance(master, team_matches)
    
    return {
        'competitive_balance': balance,
        'league_style': style,
        'corner_efficiency': corners,
        'shooting_profiles': shooting,
        'clutch_performance': clutch,
    }