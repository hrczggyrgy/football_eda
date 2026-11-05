"""
Advanced insights module.
Implements priority analyses A-F for the visual research portfolio:
A. Finishing persistence (early vs late season)
B. Team profile changes (shot generation/suppression trajectories)
C. Halftime deficit recovery (conditional on deficit size, venue, strength)
D. Volatility and streakiness beyond simple dispersion
E. Pooled vs within-league relationships
F. Home advantage changes across seasons
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import binomtest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import warnings

# Suppress only specific known warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
warnings.filterwarnings('ignore', category=UserWarning, message='X does not have valid feature names')


def analyze_finishing_persistence(team_matches: pd.DataFrame, min_matches: int = 20) -> Dict:
    """
    Priority A: Does apparent finishing strength persist?
    
    Split team-seasons into non-overlapping early and later periods.
    Compare shot volume, shot accuracy, goals per shot, goals per shot on target.
    Show raw and league-season-adjusted relationships.
    Use exposure thresholds and appropriate shrinkage.
    """
    results = {}
    
    # Filter to team-seasons with sufficient matches
    season_sizes = team_matches.groupby(['Div', 'Team', 'Season']).size().reset_index(name='Matches')
    valid_seasons = season_sizes[season_sizes['Matches'] >= min_matches]
    team_matches_valid = team_matches.merge(valid_seasons[['Div', 'Team', 'Season']], on=['Div', 'Team', 'Season'])
    
    # Split each team-season into early and late periods (by match order)
    def split_period_efficiency(group):
        group = group.sort_values('Date')
        n = len(group)
        mid = n // 2
        early = group.iloc[:mid]
        late = group.iloc[mid:]
        
        def calc_eff(df):
            shots = df['Shots'].sum()
            shots_ot = df['Shots_OT'].sum()
            gf = df['GF'].sum()
            ga = df['GA'].sum()
            shots_a = df['Shots_Against'].sum()
            shots_ot_a = df['Shots_OT_Against'].sum()
            return pd.Series({
                'Shots': shots,
                'Shots_OT': shots_ot,
                'GF': gf,
                'GA': ga,
                'Shots_Against': shots_a,
                'Shots_OT_Against': shots_ot_a,
                'Matches': len(df),
                'Shot_Accuracy': shots_ot / shots if shots > 0 else np.nan,
                'Goals_Per_Shot': gf / shots if shots > 0 else np.nan,
                'Goals_Per_Shot_OT': gf / shots_ot if shots_ot > 0 else np.nan,
                'GA_Per_Opp_Shot': ga / shots_a if shots_a > 0 else np.nan,
                'GA_Per_Opp_Shot_OT': ga / shots_ot_a if shots_ot_a > 0 else np.nan,
            })
        
        early_eff = calc_eff(early)
        late_eff = calc_eff(late)
        early_eff.index = [c + '_Early' for c in early_eff.index]
        late_eff.index = [c + '_Late' for c in late_eff.index]
        return pd.concat([early_eff, late_eff])
    
    period_eff = team_matches_valid.groupby(['Div', 'Team', 'Season']).apply(split_period_efficiency).reset_index()
    results['period_efficiency'] = period_eff
    
    # Exposure thresholds: minimum shots for finishing metrics
    min_shots = 30  # minimum total shots in period for GPS
    min_shots_ot = 10  # minimum shots on target for GPSOT
    
    # Raw persistence correlations
    raw_persistence = {}
    for metric in ['Shots_Early', 'Shots_OT_Early', 'Shot_Accuracy_Early', 'Goals_Per_Shot_Early', 'Goals_Per_Shot_OT_Early',
                   'GA_Per_Opp_Shot_Early', 'GA_Per_Opp_Shot_OT_Early']:
        late_metric = metric.replace('_Early', '_Late')
        if metric in period_eff.columns and late_metric in period_eff.columns:
            # Apply exposure filter
            if 'Goals_Per_Shot' in metric or 'Goals_Per_Shot_OT' in metric or 'GA_Per_Opp_Shot' in metric:
                exposure_col = 'Shots_Early' if 'Goals_Per_Shot' in metric and 'Opp' not in metric else \
                              'Shots_OT_Early' if 'Goals_Per_Shot_OT' in metric else \
                              'Shots_Against_Early'
                valid = (period_eff[metric].notna() & period_eff[late_metric].notna() & 
                         (period_eff[exposure_col] >= (min_shots if 'Shot_OT' not in exposure_col else min_shots_ot)))
            else:
                valid = period_eff[metric].notna() & period_eff[late_metric].notna()
            
            if valid.sum() > 20:
                r, p = stats.pearsonr(period_eff.loc[valid, metric], period_eff.loc[valid, late_metric])
                raw_persistence[metric] = {'r': r, 'p': p, 'n': int(valid.sum())}
    
    results['raw_persistence'] = raw_persistence
    
    # League-season adjusted (residualized) persistence
    # For each metric, regress on league-season dummies and use residuals
    adjusted_persistence = {}
    period_eff['LeagueSeason'] = period_eff['Div'] + '_' + period_eff['Season']
    
    for metric in ['Shots_Early', 'Shots_OT_Early', 'Shot_Accuracy_Early', 'Goals_Per_Shot_Early', 'Goals_Per_Shot_OT_Early',
                   'GA_Per_Opp_Shot_Early', 'GA_Per_Opp_Shot_OT_Early']:
        late_metric = metric.replace('_Early', '_Late')
        if metric in period_eff.columns and late_metric in period_eff.columns:
            # Regress early metric on league-season
            valid_early = period_eff[metric].notna()
            if valid_early.sum() > 20:
                # Simple league-season fixed effects via groupby transform
                ls_mean_early = period_eff.loc[valid_early].groupby('LeagueSeason')[metric].transform('mean')
                period_eff.loc[valid_early, f'{metric}_Adj'] = period_eff.loc[valid_early, metric] - ls_mean_early
                
                valid_late = period_eff[late_metric].notna()
                ls_mean_late = period_eff.loc[valid_late].groupby('LeagueSeason')[late_metric].transform('mean')
                period_eff.loc[valid_late, f'{late_metric}_Adj'] = period_eff.loc[valid_late, late_metric] - ls_mean_late
                
                adj_early = f'{metric}_Adj'
                adj_late = f'{late_metric}_Adj'
                valid = period_eff[adj_early].notna() & period_eff[adj_late].notna()
                if valid.sum() > 20:
                    r, p = stats.pearsonr(period_eff.loc[valid, adj_early], period_eff.loc[valid, adj_late])
                    adjusted_persistence[metric] = {'r': r, 'p': p, 'n': int(valid.sum())}
    
    results['adjusted_persistence'] = adjusted_persistence
    
    # Quantile group trajectories
    trajectory_data = {}
    for metric in ['Goals_Per_Shot_Early', 'Goals_Per_Shot_OT_Early', 'GA_Per_Opp_Shot_Early']:
        late_metric = metric.replace('_Early', '_Late')
        if metric in period_eff.columns and late_metric in period_eff.columns:
            valid = period_eff[metric].notna() & period_eff[late_metric].notna()
            if valid.sum() > 50:
                df_valid = period_eff.loc[valid].copy()
                # Quintile bins
                df_valid['Quantile'] = pd.qcut(df_valid[metric], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
                traj = df_valid.groupby('Quantile')[late_metric].agg(['mean', 'std', 'count']).reset_index()
                trajectory_data[metric] = traj.to_dict('records')
    
    results['quantile_trajectories'] = trajectory_data
    
    # Sensitivity to split point (1/3 vs 1/2 vs 2/3)
    split_sensitivity = {}
    for split_frac in [1/3, 1/2, 2/3]:
        def split_custom(group):
            group = group.sort_values('Date')
            n = len(group)
            split_idx = int(n * split_frac)
            early = group.iloc[:split_idx]
            late = group.iloc[split_idx:]
            
            def calc_eff(df):
                shots = df['Shots'].sum()
                shots_ot = df['Shots_OT'].sum()
                gf = df['GF'].sum()
                return pd.Series({
                    'GPS': gf / shots if shots > 0 else np.nan,
                    'GPSOT': gf / shots_ot if shots_ot > 0 else np.nan,
                })
            
            early_eff = calc_eff(early)
            late_eff = calc_eff(late)
            early_eff.index = [c + '_Early' for c in early_eff.index]
            late_eff.index = [c + '_Late' for c in late_eff.index]
            return pd.concat([early_eff, late_eff])
        
        custom_eff = team_matches_valid.groupby(['Div', 'Team', 'Season']).apply(split_custom).reset_index()
        for metric in ['GPS_Early', 'GPSOT_Early']:
            late_metric = metric.replace('_Early', '_Late')
            valid = custom_eff[metric].notna() & custom_eff[late_metric].notna()
            if valid.sum() > 20:
                r, p = stats.pearsonr(custom_eff.loc[valid, metric], custom_eff.loc[valid, late_metric])
                split_sensitivity[f'{metric}_split{int(split_frac*100)}'] = {'r': r, 'p': p, 'n': int(valid.sum())}
    
    results['split_sensitivity'] = split_sensitivity
    
    return results


def analyze_team_profile_changes(team_season: pd.DataFrame, min_seasons: int = 2) -> Dict:
    """
    Priority B: Which teams changed their performance profile?
    
    Engineer interpretable shot-generation, shot-suppression, accuracy, conversion measures.
    Normalize against league-season context.
    Compare consecutive seasons.
    Separate within-division changes from promotion/relegation transitions.
    Uses CHRONOLOGICAL team sorting with adjacent-season requirement.
    Infers promotion/relegation only where observed league hierarchy supports it.
    """
    from .new_insights import build_promotion_relegation_transitions
    
    results = {}
    
    # Ensure required metrics exist
    if 'Shots_PG' not in team_season.columns:
        team_season = team_season.copy()
        team_season['Shots_PG'] = team_season['Shots'] / team_season['Matches']
    if 'Shots_Against_PG' not in team_season.columns:
        team_season = team_season.copy()
        team_season['Shots_Against_PG'] = team_season['Shots_Against'] / team_season['Matches']
    
    # League-season normalization (z-score within league-season)
    metrics_to_normalize = ['Shots_PG', 'Shots_Against_PG', 'Shot_Accuracy', 'Goals_Per_Shot_OT', 'GA_Per_Opp_Shot_OT']
    
    team_season = team_season.copy()
    team_season['LeagueSeason'] = team_season['Div'] + '_' + team_season['Season']
    
    for metric in metrics_to_normalize:
        if metric in team_season.columns:
            ls_mean = team_season.groupby('LeagueSeason')[metric].transform('mean')
            ls_std = team_season.groupby('LeagueSeason')[metric].transform('std')
            team_season[f'{metric}_z'] = (team_season[metric] - ls_mean) / ls_std.replace(0, np.nan)
    
    # Key profile dimensions (z-scored)
    # Attacking: shot generation (Shots_PG_z) and finishing (GPSOT_z)
    # Defensive: shot suppression (Shots_Against_PG_z, inverted so higher = better) and resilience (GAOSOT_z, inverted)
    if 'Shots_Against_PG_z' in team_season.columns:
        team_season['Shot_Suppression_z'] = -team_season['Shots_Against_PG_z']  # Higher = better suppression
    if 'GA_Per_Opp_Shot_OT_z' in team_season.columns:
        team_season['Def_Resilience_z'] = -team_season['GA_Per_Opp_Shot_OT_z']  # Higher = better resilience
    
    # Build promotion/relegation transitions with proper chronological sorting
    ts_transitions = build_promotion_relegation_transitions(team_season)
    
    # Consecutive season changes (same division)
    consecutive = ts_transitions[ts_transitions['Consecutive'] & ~ts_transitions['Div_Changed']].copy()
    
    profile_dims = ['Shots_PG_z', 'Goals_Per_Shot_OT_z', 'Shot_Suppression_z', 'Def_Resilience_z']
    available_dims = [d for d in profile_dims if d in consecutive.columns]
    
    # Compute changes for consecutive seasons
    for dim in available_dims:
        consecutive[f'Delta_{dim}'] = consecutive.groupby(['Div', 'Team'])[dim].diff()
    
    # Within-division changes
    within_div_changes = consecutive[available_dims + [f'Delta_{d}' for d in available_dims] + ['Div', 'Team', 'Season']].copy()
    results['within_division_changes'] = within_div_changes
    
    # Promotion/relegation transitions (from proper chronological matching)
    promo_releg = ts_transitions[ts_transitions['Consecutive'] & ts_transitions['Div_Changed']].copy()
    div_change_dims = [f'Delta_{d}' for d in available_dims if f'Delta_{d}' in promo_releg.columns]
    results['promotion_relegation_changes'] = promo_releg[['Div', 'Prev_Div', 'Team', 'Season', 'Transition_Type'] + div_change_dims].copy()
    
    # Summary statistics
    change_summary = {}
    for dim in available_dims:
        delta_col = f'Delta_{dim}'
        if delta_col in consecutive.columns:
            valid = consecutive[delta_col].notna()
            change_summary[dim] = {
                'mean': consecutive.loc[valid, delta_col].mean(),
                'std': consecutive.loc[valid, delta_col].std(),
                'n': int(valid.sum()),
                'top_improvers': consecutive.loc[valid].nlargest(10, delta_col)[['Div', 'Team', 'Season', delta_col]].to_dict('records'),
                'top_decliners': consecutive.loc[valid].nsmallest(10, delta_col)[['Div', 'Team', 'Season', delta_col]].to_dict('records'),
            }
    
    results['change_summary'] = change_summary
    
    # Statistical significance: do changes exceed sampling variation?
    sampling_var = {}
    for dim in available_dims:
        team_var = ts_transitions.groupby('Team')[dim].var()
        sampling_var[dim] = {
            'median_within_team_var': team_var.median(),
            'mean_change_var': consecutive[f'Delta_{dim}'].var() if f'Delta_{dim}' in consecutive.columns else np.nan,
        }
    
    results['sampling_variation'] = sampling_var
    
    return results


def analyze_halftime_deficit_recovery(master: pd.DataFrame, team_season: pd.DataFrame, 
                                        team_matches: pd.DataFrame) -> Dict:
    """
    Priority C: How much does a halftime deficit really matter?
    
    FIXED: Uses GENUINELY LAGGED pre-match strength estimate.
    - Cumulative PPG before this match (shifted)
    - Early-season initialization uses PRIOR SEASON average, not current season
    - Strength terciles defined using EXPANDING HISTORICAL thresholds (not current season)
    - First-match initialization explicitly handled (NaN, not invented)
    """
    results = {}
    
    # Create team-perspective halftime data
    ht_data = pd.concat([
        master.assign(
            Team=master['HomeTeam'],
            Venue='Home',
            HT_GD=master['HTHG'] - master['HTAG'],
            FT_GD=master['FTHG'] - master['FTAG'],
            HT_Goals=master['HTHG'],
            FT_Goals=master['FTHG'],
            Opp_HT_Goals=master['HTAG'],
            Opp_FT_Goals=master['FTAG']
        ),
        master.assign(
            Team=master['AwayTeam'],
            Venue='Away',
            HT_GD=master['HTAG'] - master['HTHG'],
            FT_GD=master['FTAG'] - master['FTHG'],
            HT_Goals=master['HTAG'],
            FT_Goals=master['FTAG'],
            Opp_HT_Goals=master['HTHG'],
            Opp_FT_Goals=master['FTHG']
        )
    ], ignore_index=True)
    
    # Add GENUINELY LAGGED pre-match strength using the FIXED function from new_insights
    from .new_insights import add_lagged_team_strength
    tm = add_lagged_team_strength(team_matches)
    
    # Merge strength into ht_data
    strength_lookup = tm[['Div', 'Team', 'Season', 'Date', 'PreMatch_PPG_Shrunk']].copy()
    ht_data = ht_data.merge(strength_lookup, on=['Div', 'Team', 'Season', 'Date'], how='left')
    
    # FIXED: Strength categories using EXPANDING HISTORICAL thresholds
    # For each league-season, use historical (prior seasons) distribution to define terciles
    # This avoids using future data to define current categories
    
    # Build historical league-season PPG distribution
    season_order = {'1920':0, '2021':1, '2122':2, '2223':3, '2324':4, '2425':5, '2526':6}
    
    # Compute historical tercile thresholds for each league-season
    hist_thresholds = {}
    for div in ht_data['Div'].unique():
        div_data = tm[tm['Div'] == div].copy()
        div_data['Season_Num'] = div_data['Season'].map(season_order)
        div_data = div_data.sort_values(['Season_Num', 'Date'])
        
        for season_code in div_data['Season'].unique():
            season_num = season_order.get(season_code, np.nan)
            if pd.isna(season_num):
                continue
            
            # Historical data: all PRIOR seasons in this division
            hist_data = div_data[div_data['Season_Num'] < season_num]
            if len(hist_data) > 50 and hist_data['PreMatch_PPG_Shrunk'].notna().sum() > 30:
                hist_ppg = hist_data['PreMatch_PPG_Shrunk'].dropna()
                if len(hist_ppg) > 30:
                    q33, q66 = hist_ppg.quantile([1/3, 2/3])
                    hist_thresholds[(div, season_code)] = (q33, q66)
    
    # Apply historical thresholds
    def assign_strength_tercile(row):
        key = (row['Div'], row['Season'])
        if key in hist_thresholds and pd.notna(row['PreMatch_PPG_Shrunk']):
            q33, q66 = hist_thresholds[key]
            if row['PreMatch_PPG_Shrunk'] <= q33:
                return 'Weak'
            elif row['PreMatch_PPG_Shrunk'] <= q66:
                return 'Average'
            else:
                return 'Strong'
        return np.nan
    
    ht_data['Strength_Tercile'] = ht_data.apply(assign_strength_tercile, axis=1)
    
    # Categorize HT goal difference
    ht_data['HT_GD_Cat'] = pd.cut(ht_data['HT_GD'], 
                                   bins=[-np.inf, -3, -2, -1, 0, 1, 2, 3, np.inf],
                                   labels=['-3 or worse', '-2', '-1', '0', '+1', '+2', '+3', '+3 or better'])
    
    # Outcome categories
    ht_data['FT_Result'] = np.where(ht_data['FT_GD'] > 0, 'Win',
                           np.where(ht_data['FT_GD'] < 0, 'Loss', 'Draw'))
    ht_data['Avoided_Defeat'] = (ht_data['FT_GD'] >= 0).astype(int)
    ht_data['Won'] = (ht_data['FT_GD'] > 0).astype(int)
    
    results['ht_data'] = ht_data
    
    # Recovery rates by HT goal difference, venue
    recovery_by_gd_venue = ht_data.groupby(['HT_GD_Cat', 'Venue']).agg(
        N=('Team', 'count'),
        Win_Rate=('Won', 'mean'),
        Avoid_Defeat_Rate=('Avoided_Defeat', 'mean'),
        Avg_FT_GD=('FT_GD', 'mean'),
    ).reset_index()
    results['recovery_by_gd_venue'] = recovery_by_gd_venue
    
    # Recovery rates by HT goal difference, venue, strength
    if 'Strength_Tercile' in ht_data.columns:
        recovery_by_gd_venue_strength = ht_data.groupby(['HT_GD_Cat', 'Venue', 'Strength_Tercile']).agg(
            N=('Team', 'count'),
            Win_Rate=('Won', 'mean'),
            Avoid_Defeat_Rate=('Avoided_Defeat', 'mean'),
        ).reset_index()
        results['recovery_by_gd_venue_strength'] = recovery_by_gd_venue_strength
    
    # Focus on trailing teams (HT_GD < 0)
    trailing = ht_data[ht_data['HT_GD'] < 0].copy()
    trailing['Deficit_Size'] = -trailing['HT_GD']
    
    trailing_recovery = trailing.groupby(['Deficit_Size', 'Venue']).agg(
        N=('Team', 'count'),
        Win_Rate=('Won', 'mean'),
        Avoid_Defeat_Rate=('Avoided_Defeat', 'mean'),
    ).reset_index()
    results['trailing_recovery'] = trailing_recovery
    
    if 'Strength_Tercile' in trailing.columns:
        trailing_recovery_strength = trailing.groupby(['Deficit_Size', 'Venue', 'Strength_Tercile']).agg(
            N=('Team', 'count'),
            Win_Rate=('Won', 'mean'),
            Avoid_Defeat_Rate=('Avoided_Defeat', 'mean'),
        ).reset_index()
        results['trailing_recovery_strength'] = trailing_recovery_strength
    
    # League-level recovery rates (for teams trailing at HT)
    trailing_by_league = trailing.groupby(['Div', 'Venue']).agg(
        N=('Team', 'count'),
        Win_Rate=('Won', 'mean'),
        Avoid_Defeat_Rate=('Avoided_Defeat', 'mean'),
    ).reset_index()
    results['trailing_by_league'] = trailing_by_league
    
    # Overall stats
    results['overall'] = {
        'total_matches': len(ht_data),
        'trailing_at_ht': len(trailing),
        'trailing_win_rate': trailing['Won'].mean(),
        'trailing_avoid_defeat_rate': trailing['Avoided_Defeat'].mean(),
        'leading_at_ht': len(ht_data[ht_data['HT_GD'] > 0]),
        'leading_win_rate': ht_data[ht_data['HT_GD'] > 0]['Won'].mean(),
        'drawing_at_ht': len(ht_data[ht_data['HT_GD'] == 0]),
        'drawing_win_rate': ht_data[ht_data['HT_GD'] == 0]['Won'].mean(),
        'strength_tercile_coverage': ht_data['Strength_Tercile'].notna().mean(),
    }
    
    return results


def analyze_volatility_streakiness(team_matches: pd.DataFrame, team_season: pd.DataFrame, min_matches: int = 20) -> Dict:
    """
    Priority D: Are some teams unusually volatile or streaky?
    
    Separate performance level, dispersion, and temporal dependence.
    Estimate match expectations using appropriate models.
    Evaluate residual dispersion with simulation-based benchmark.
    For streakiness, define order-sensitive statistic and suitable null.
    """
    results = {}
    
    # Filter to valid team-seasons
    valid_seasons = team_season[team_season['Matches'] >= min_matches].copy()
    team_matches_valid = team_matches.merge(valid_seasons[['Div', 'Team', 'Season']], on=['Div', 'Team', 'Season'])
    team_matches_valid = team_matches_valid.sort_values(['Div', 'Team', 'Season', 'Date'])
    
    # 1. Performance level: season PPG
    # 2. Dispersion: Points_CV (already computed)
    # 3. Temporal dependence: autocorrelation, runs test
    
    # Match-level expected points model (simple: league-season average + home advantage)
    match_expectations = team_matches_valid.groupby(['Div', 'Season', 'Venue'])['Points'].mean().reset_index()
    match_expectations.columns = ['Div', 'Season', 'Venue', 'Exp_Points_Baseline']
    team_matches_valid = team_matches_valid.merge(match_expectations, on=['Div', 'Season', 'Venue'], how='left')
    
    # Better model: include opponent strength (Opp_PPG_Cum)
    if 'Opp_PPG_Cum' in team_matches_valid.columns:
        # Simple linear model for expected points
        valid = team_matches_valid['Opp_PPG_Cum'].notna() & team_matches_valid['Points'].notna()
        if valid.sum() > 100:
            X = team_matches_valid.loc[valid, ['Opp_PPG_Cum']].values
            y = team_matches_valid.loc[valid, 'Points'].values
            reg = LinearRegression().fit(X, y)
            team_matches_valid.loc[valid, 'Exp_Points_Model'] = reg.predict(X)
            team_matches_valid['Residual'] = team_matches_valid['Points'] - team_matches_valid['Exp_Points_Model']
        else:
            team_matches_valid['Exp_Points_Model'] = team_matches_valid['Exp_Points_Baseline']
            team_matches_valid['Residual'] = team_matches_valid['Points'] - team_matches_valid['Exp_Points_Model']
    else:
        team_matches_valid['Exp_Points_Model'] = team_matches_valid['Exp_Points_Baseline']
        team_matches_valid['Residual'] = team_matches_valid['Points'] - team_matches_valid['Exp_Points_Model']
    
    # Season-level residual dispersion
    residual_disp = team_matches_valid.groupby(['Div', 'Team', 'Season']).agg(
        Residual_Std=('Residual', 'std'),
        Residual_Mean=('Residual', 'mean'),
        Points_Std=('Points', 'std'),
        Points_Mean=('Points', 'mean'),
        Matches=('Points', 'count'),
    ).reset_index()
    residual_disp = residual_disp[residual_disp['Matches'] >= min_matches].copy()
    residual_disp['Residual_CV'] = residual_disp['Residual_Std'] / residual_disp['Residual_Mean'].abs().replace(0, np.nan)
    
    results['residual_dispersion'] = residual_disp
    
    # Simulation-based benchmark for residual dispersion
    # Simulate matches from estimated model using calibrated outcome probabilities
    # Generate match outcomes ONCE per match, then derive both team perspectives
    n_sims = 100  # Increased for Monte Carlo precision
    sim_dispersions = []
    
    for _, season_row in residual_disp.iterrows():
        div, team, season = season_row['Div'], season_row['Team'], season_row['Season']
        matches = team_matches_valid[(team_matches_valid['Div']==div) & 
                                     (team_matches_valid['Team']==team) & 
                                     (team_matches_valid['Season']==season)].copy()
        if len(matches) < min_matches:
            continue
        
        # Get expected points and calibrate outcome probabilities per match
        exp_points = matches['Exp_Points_Model'].values
        exp_points = exp_points[~np.isnan(exp_points)]
        if len(exp_points) < min_matches:
            continue
        
        # For each match, estimate Win/Draw/Loss probabilities from expected points
        # Using ordered probit approximation: p3 = P(Win), p1 = P(Draw), p0 = P(Loss)
        # 3*p3 + 1*p1 = exp_points, p3+p1+p0=1
        # Calibrate using league-season average draw rate
        lg_season_draw_rate = matches.groupby(['Div', 'Season'])['FT_Result'].apply(
            lambda x: (x == 'D').mean()
        )
        # Default draw rate if not available
        default_draw_rate = 0.25
        
        n_matches = len(exp_points)
        sim_stds = []
        
        for _ in range(n_sims):
            sim_points = []
            for i, ep in enumerate(exp_points):
                # Use match-specific draw rate if available, else default
                if len(lg_season_draw_rate) > 0:
                    p1 = lg_season_draw_rate.iloc[0] if hasattr(lg_season_draw_rate, 'iloc') else default_draw_rate
                else:
                    p1 = default_draw_rate
                
                # Solve for p3, p0 given expected points and draw probability
                # 3*p3 + p1 = ep  =>  p3 = (ep - p1) / 3
                p3 = max(0, min(1, (ep - p1) / 3))
                p0 = 1 - p3 - p1
                
                # Ensure valid probabilities
                if p0 < 0:
                    # Edge case: very low expected points
                    p0 = 0
                    p3 = ep / 3
                    p1 = 1 - p3
                elif p3 < 0:
                    p3 = 0
                    p0 = 1 - p1
                
                if p3 < 0 or p1 < 0 or p0 < 0 or abs(p3 + p1 + p0 - 1) > 1e-10:
                    # Fallback: use empirical distribution
                    outcome = np.random.choice([3, 1, 0], p=[1/3, 1/3, 1/3])
                else:
                    outcome = np.random.choice([3, 1, 0], p=[p3, p1, p0])
                sim_points.append(outcome)
            
            if len(sim_points) == n_matches:
                sim_stds.append(np.std(sim_points))
        
        if sim_stds:
            sim_mean = np.mean(sim_stds)
            sim_std = np.std(sim_stds)
            z_score = (season_row['Points_Std'] - sim_mean) / sim_std if sim_std > 0 else np.nan
            sim_dispersions.append({
                'Div': div, 'Team': team, 'Season': season,
                'Observed_Std': season_row['Points_Std'],
                'Simulated_Std_Mean': sim_mean,
                'Simulated_Std_Std': sim_std,
                'Z_Score': z_score,
                'N_Sims': len(sim_stds),
            })
    
    results['simulation_benchmark'] = pd.DataFrame(sim_dispersions)
    
    # Document exchangeability assumptions
    results['simulation_assumptions'] = {
        'exchangeability': 'Matches within a team-season are exchangeable conditional on expected points',
        'outcome_model': 'Multinomial (Win/Draw/Loss) with probabilities calibrated to expected points and league-season draw rate',
        'n_sims': n_sims,
        'monte_carlo_precision': f'SE of Z-score ~ 1/sqrt({n_sims}) ≈ 0.1',
    }
    
    # Streakiness: runs test and autocorrelation on residuals
    streakiness = []
    for _, season_row in residual_disp.iterrows():
        div, team, season = season_row['Div'], season_row['Team'], season_row['Season']
        matches = team_matches_valid[(team_matches_valid['Div']==div) & 
                                     (team_matches_valid['Team']==team) & 
                                     (team_matches_valid['Season']==season)].copy()
        if len(matches) < min_matches:
            continue
        
        residuals = matches['Residual'].dropna().values
        if len(residuals) < 10:
            continue
        
        # Runs test (above/below median)
        median_res = np.median(residuals)
        signs = (residuals > median_res).astype(int)
        runs = 1 + np.sum(signs[1:] != signs[:-1])
        n_pos = np.sum(signs)
        n_neg = len(signs) - n_pos
        expected_runs = 1 + 2 * n_pos * n_neg / (n_pos + n_neg)
        runs_var = (2 * n_pos * n_neg * (2 * n_pos * n_neg - n_pos - n_neg)) / \
                   ((n_pos + n_neg)**2 * (n_pos + n_neg - 1))
        runs_z = (runs - expected_runs) / np.sqrt(runs_var) if runs_var > 0 else 0
        
        # Lag-1 autocorrelation
        acf1 = np.corrcoef(residuals[:-1], residuals[1:])[0,1] if len(residuals) > 1 else 0
        
        streakiness.append({
            'Div': div, 'Team': team, 'Season': season,
            'Runs': runs,
            'Expected_Runs': expected_runs,
            'Runs_Z': runs_z,
            'ACF1': acf1,
            'Residual_Std': np.std(residuals),
        })
    
    results['streakiness'] = pd.DataFrame(streakiness)
    
    # Permutation test for streakiness (within-season shuffle)
    perm_results = []
    n_perm = 20
    for _, season_row in residual_disp.iterrows():
        div, team, season = season_row['Div'], season_row['Team'], season_row['Season']
        matches = team_matches_valid[(team_matches_valid['Div']==div) & 
                                     (team_matches_valid['Team']==team) & 
                                     (team_matches_valid['Season']==season)].copy()
        if len(matches) < min_matches:
            continue
        
        residuals = matches['Residual'].dropna().values
        if len(residuals) < 10:
            continue
        
        # Observed runs Z
        median_res = np.median(residuals)
        signs = (residuals > median_res).astype(int)
        obs_runs = 1 + np.sum(signs[1:] != signs[:-1])
        n_pos = np.sum(signs)
        n_neg = len(signs) - n_pos
        expected_runs = 1 + 2 * n_pos * n_neg / (n_pos + n_neg)
        runs_var = (2 * n_pos * n_neg * (2 * n_pos * n_neg - n_pos - n_neg)) / \
                   ((n_pos + n_neg)**2 * (n_pos + n_neg - 1))
        obs_runs_z = (obs_runs - expected_runs) / np.sqrt(runs_var) if runs_var > 0 else 0
        
        # Permutation distribution
        perm_zs = []
        for _ in range(n_perm):
            perm_res = np.random.permutation(residuals)
            perm_signs = (perm_res > np.median(perm_res)).astype(int)
            perm_runs = 1 + np.sum(perm_signs[1:] != perm_signs[:-1])
            perm_z = (perm_runs - expected_runs) / np.sqrt(runs_var) if runs_var > 0 else 0
            perm_zs.append(perm_z)
        
        p_value = (np.sum(np.abs(perm_zs) >= np.abs(obs_runs_z)) + 1) / (n_perm + 1)
        
        perm_results.append({
            'Div': div, 'Team': team, 'Season': season,
            'Observed_Runs_Z': obs_runs_z,
            'Perm_P_Value': p_value,
            'Streaky': p_value < 0.05 and obs_runs_z < 0,  # Negative Z = fewer runs = streakier
        })
    
    results['permutation_streakiness'] = pd.DataFrame(perm_results)
    
    # Summary: most/least streaky teams (3+ seasons)
    if len(results['permutation_streakiness']) > 0:
        team_streak = results['permutation_streakiness'].groupby('Team').agg(
            Seasons=('Season', 'count'),
            Avg_Runs_Z=('Observed_Runs_Z', 'mean'),
            Streaky_Seasons=('Streaky', 'sum'),
        ).reset_index()
        team_streak = team_streak[team_streak['Seasons'] >= 3]
        results['team_streakiness_summary'] = team_streak.sort_values('Avg_Runs_Z')
    
    return results


def analyze_pooled_vs_within_league(team_season: pd.DataFrame, min_matches: int = 20) -> Dict:
    """
    Priority E: Do pooled relationships survive within leagues?
    
    Compare pooled, within-league, and league-season-adjusted relationships.
    Examine shot volume, accuracy, conversion, discipline, and PPG.
    Account for team repetition and paired match structure.
    """
    results = {}
    
    valid = team_season[team_season['Matches'] >= min_matches].copy()
    
    # Key relationships to test
    relationships = [
        ('Shots_PG', 'PPG', 'Shot Volume -> Points'),
        ('Shot_Accuracy', 'PPG', 'Shot Accuracy -> Points'),
        ('Goals_Per_Shot_OT', 'PPG', 'Finishing -> Points'),
        ('GA_Per_Opp_Shot', 'PPG', 'Defensive Efficiency -> Points'),
        ('Yellow_PG', 'PPG', 'Discipline -> Points'),
        ('Fouls_PG', 'PPG', 'Fouls -> Points'),
        ('Shots_PG', 'Goals_Per_Shot_OT', 'Volume -> Finishing'),
    ]
    
    # Pooled correlations
    pooled_corrs = {}
    for x, y, label in relationships:
        if x in valid.columns and y in valid.columns:
            v = valid[x].notna() & valid[y].notna()
            if v.sum() > 20:
                r, p = stats.pearsonr(valid.loc[v, x], valid.loc[v, y])
                pooled_corrs[label] = {'r': r, 'p': p, 'n': int(v.sum())}
    results['pooled'] = pooled_corrs
    
    # Within-league correlations (fixed effects)
    within_league_corrs = {}
    for league in valid['Div'].unique():
        lg_data = valid[valid['Div'] == league]
        league_corrs = {}
        for x, y, label in relationships:
            if x in lg_data.columns and y in lg_data.columns:
                v = lg_data[x].notna() & lg_data[y].notna()
                if v.sum() > 10:
                    r, p = stats.pearsonr(lg_data.loc[v, x], lg_data.loc[v, y])
                    league_corrs[label] = {'r': r, 'p': p, 'n': int(v.sum())}
        within_league_corrs[league] = league_corrs
    results['within_league'] = within_league_corrs
    
    # League-season adjusted (residualized) correlations
    valid = valid.copy()
    valid['LeagueSeason'] = valid['Div'] + '_' + valid['Season']
    
    adjusted_corrs = {}
    for x, y, label in relationships:
        if x in valid.columns and y in valid.columns:
            # Residualize both on league-season
            v = valid[x].notna() & valid[y].notna()
            if v.sum() > 20:
                ls_x = valid.loc[v].groupby('LeagueSeason')[x].transform('mean')
                ls_y = valid.loc[v].groupby('LeagueSeason')[y].transform('mean')
                x_resid = valid.loc[v, x] - ls_x
                y_resid = valid.loc[v, y] - ls_y
                v2 = x_resid.notna() & y_resid.notna()
                if v2.sum() > 20:
                    r, p = stats.pearsonr(x_resid[v2], y_resid[v2])
                    adjusted_corrs[label] = {'r': r, 'p': p, 'n': int(v2.sum())}
    results['league_season_adjusted'] = adjusted_corrs
    
    # Regression coefficients with league-season fixed effects
    reg_results = {}
    for x, y, label in [('Shots_PG', 'PPG', 'Volume->Points'), ('Goals_Per_Shot_OT', 'PPG', 'Finishing->Points')]:
        if x in valid.columns and y in valid.columns:
            v = valid[x].notna() & valid[y].notna() & valid['LeagueSeason'].notna()
            if v.sum() > 50:
                df_reg = valid.loc[v, [x, y, 'LeagueSeason']].copy()
                # OLS with league-season dummies
                dummies = pd.get_dummies(df_reg['LeagueSeason'], drop_first=True)
                X = pd.concat([df_reg[[x]], dummies], axis=1)
                y_reg = df_reg[y]
                reg = LinearRegression(fit_intercept=True).fit(X, y_reg)
                reg_results[label] = {
                    'coef': reg.coef_[0],
                    'intercept': reg.intercept_,
                    'r2': reg.score(X, y_reg),
                    'n': len(df_reg),
                    'n_fixed_effects': X.shape[1] - 1,
                }
    results['fixed_effects_regression'] = reg_results
    
    # Meta-analysis: test for heterogeneity across leagues
    heterogeneity = {}
    for x, y, label in relationships:
        if x in valid.columns and y in valid.columns:
            league_rs = []
            league_ns = []
            for league in valid['Div'].unique():
                lg_data = valid[valid['Div'] == league]
                v = lg_data[x].notna() & lg_data[y].notna()
                if v.sum() > 10:
                    r, _ = stats.pearsonr(lg_data.loc[v, x], lg_data.loc[v, y])
                    league_rs.append(r)
                    league_ns.append(v.sum())
            if len(league_rs) >= 3:
                # Fisher's z transformation for meta-analysis
                z_scores = [np.arctanh(r) for r in league_rs]
                weights = [n - 3 for n in league_ns]
                weighted_mean_z = np.average(z_scores, weights=weights)
                # Q statistic for heterogeneity
                Q = np.sum([w * (z - weighted_mean_z)**2 for w, z in zip(weights, z_scores)])
                df = len(z_scores) - 1
                p_het = 1 - stats.chi2.cdf(Q, df) if df > 0 else 1.0
                heterogeneity[label] = {
                    'league_correlations': dict(zip(valid['Div'].unique(), league_rs)),
                    'mean_r': np.tanh(weighted_mean_z),
                    'Q': Q,
                    'df': df,
                    'p_heterogeneity': p_het,
                    'I2': max(0, (Q - df) / Q * 100) if Q > 0 else 0,
                }
    results['heterogeneity'] = heterogeneity
    
    return results


def analyze_home_advantage_changes(master: pd.DataFrame, team_season: pd.DataFrame) -> Dict:
    """
    Priority F: How did home advantage change across seasons?
    
    Show home-minus-away outcomes and goal difference.
    Use league-specific trends and uncertainty.
    Compare balanced team samples as sensitivity check.
    Treat pandemic-era season labels as descriptive.
    """
    results = {}
    
    # League-season home advantage metrics
    ha_ls = master.groupby(['Div', 'Season']).apply(
        lambda x: pd.Series({
            'Home_Win_Pct': (x['FTR'] == 'H').mean(),
            'Draw_Pct': (x['FTR'] == 'D').mean(),
            'Away_Win_Pct': (x['FTR'] == 'A').mean(),
            'Home_Goals': x['FTHG'].mean(),
            'Away_Goals': x['FTAG'].mean(),
            'Goal_Diff': x['FTHG'].mean() - x['FTAG'].mean(),
            'Home_Win_Share_Decisive': (x['FTR'] == 'H').sum() / ((x['FTR'] == 'H').sum() + (x['FTR'] == 'A').sum()) if ((x['FTR'] == 'H').sum() + (x['FTR'] == 'A').sum()) > 0 else np.nan,
            'Matches': len(x)
        })
    ).reset_index()
    results['league_season'] = ha_ls
    
    # Three distinct HA measures
    # 1. Raw home win rate
    # 2. Home minus away win rate
    # 3. Home share of decisive matches
    ha_ls['HA_Win_Diff'] = ha_ls['Home_Win_Pct'] - ha_ls['Away_Win_Pct']
    
    # League trends with uncertainty (linear trend per league)
    league_trends = {}
    for league in ha_ls['Div'].unique():
        lg = ha_ls[ha_ls['Div'] == league].sort_values('Season')
        if len(lg) >= 3:
            season_nums = np.arange(len(lg))
            for metric in ['Home_Win_Pct', 'HA_Win_Diff', 'Home_Win_Share_Decisive', 'Goal_Diff']:
                if lg[metric].notna().sum() >= 3:
                    v = lg[metric].notna()
                    slope, intercept, r, p, se = stats.linregress(season_nums[v], lg.loc[v, metric])
                    league_trends[f'{league}_{metric}'] = {
                        'slope_per_season': slope,
                        'p_value': p,
                        'r_squared': r**2,
                        'n_seasons': int(v.sum()),
                    }
    results['league_trends'] = league_trends
    
    # Overall trend (match-weighted)
    overall_trend = {}
    for metric in ['Home_Win_Pct', 'HA_Win_Diff', 'Home_Win_Share_Decisive', 'Goal_Diff']:
        v = ha_ls[metric].notna()
        if v.sum() >= 3:
            slope, intercept, r, p, se = stats.linregress(np.arange(v.sum()), ha_ls.loc[v, metric])
            overall_trend[metric] = {
                'slope_per_season': slope,
                'p_value': p,
                'r_squared': r**2,
                'n_league_seasons': int(v.sum()),
            }
    results['overall_trend'] = overall_trend
    
    # Balanced team sample: teams present in all 7 seasons in same division
    balanced_teams = team_season.groupby(['Div', 'Team'])['Season'].nunique().reset_index(name='n_seasons')
    balanced_teams = balanced_teams[balanced_teams['n_seasons'] == 7]
    
    if len(balanced_teams) > 0:
        balanced_ha = master.merge(balanced_teams[['Div', 'Team']], 
                                    left_on=['Div', 'HomeTeam'], right_on=['Div', 'Team'], how='inner')
        balanced_ha = balanced_ha.groupby(['Div', 'Season']).apply(
            lambda x: pd.Series({
                'Home_Win_Pct': (x['FTR'] == 'H').mean(),
                'Away_Win_Pct': (x['FTR'] == 'A').mean(),
                'Goal_Diff': x['FTHG'].mean() - x['FTAG'].mean(),
                'Matches': len(x)
            })
        ).reset_index()
        balanced_ha['HA_Win_Diff'] = balanced_ha['Home_Win_Pct'] - balanced_ha['Away_Win_Pct']
    else:
        # Empty dataframe with expected columns
        balanced_ha = pd.DataFrame(columns=['Div', 'Season', 'Home_Win_Pct', 'Away_Win_Pct', 'Goal_Diff', 'Matches', 'HA_Win_Diff'])
    results['balanced_sample'] = balanced_ha
    
    # Pre/post COVID comparison (descriptive only)
    pre_covid = ha_ls[ha_ls['Season'].isin(['1920'])]  # Only 1920 is truly pre-COVID
    covid = ha_ls[ha_ls['Season'] == '2021']
    post_covid = ha_ls[ha_ls['Season'].isin(['2122', '2223', '2324', '2425', '2526'])]
    
    covid_comparison = {}
    for metric in ['Home_Win_Pct', 'HA_Win_Diff', 'Home_Win_Share_Decisive', 'Goal_Diff']:
        pre_val = pre_covid[metric].mean()
        covid_val = covid[metric].mean()
        post_val = post_covid[metric].mean()
        covid_comparison[metric] = {
            'pre_covid': pre_val,
            'covid_season': covid_val,
            'post_covid': post_val,
            'covid_drop': covid_val - pre_val,
            'recovery': post_val - covid_val,
        }
    results['covid_comparison'] = covid_comparison
    
    # Team-level HA stability (teams with 5+ seasons)
    team_ha = master.groupby(['Div', 'HomeTeam', 'Season']).agg(
        Home_Win_Pct=('FTR', lambda x: (x == 'H').mean()),
        Home_Matches=('FTR', 'count'),
    ).reset_index()
    team_ha = team_ha[team_ha['Home_Matches'] >= 15].rename(columns={'HomeTeam': 'Team'})
    
    away_ha = master.groupby(['Div', 'AwayTeam', 'Season']).agg(
        Away_Win_Pct=('FTR', lambda x: (x == 'A').mean()),
        Away_Matches=('FTR', 'count'),
    ).reset_index()
    away_ha = away_ha[away_ha['Away_Matches'] >= 15].rename(columns={'AwayTeam': 'Team'})
    
    team_ha_full = team_ha.merge(away_ha[['Div', 'Team', 'Season', 'Away_Win_Pct']], on=['Div', 'Team', 'Season'])
    team_ha_full['HA_Diff'] = team_ha_full['Home_Win_Pct'] - team_ha_full['Away_Win_Pct']
    
    # Team HA trend
    team_ha_trends = {}
    for (div, team), group in team_ha_full.groupby(['Div', 'Team']):
        if len(group) >= 4:
            v = group['HA_Diff'].notna()
            if v.sum() >= 4:
                slope, intercept, r, p, se = stats.linregress(np.arange(v.sum()), group.loc[v, 'HA_Diff'])
                team_ha_trends[f'{div}_{team}'] = {
                    'slope': slope,
                    'p': p,
                    'n_seasons': int(v.sum()),
                }
    results['team_ha_trends'] = team_ha_trends
    
    return results