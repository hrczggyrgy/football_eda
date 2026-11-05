"""
Statistical analysis module.
Contains functions for home advantage, efficiency persistence, shot decomposition,
half-time/full-time dynamics, consistency analysis, and disciplinary patterns.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import binomtest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional


def analyze_home_advantage(master: pd.DataFrame) -> Dict:
    """Analyze home advantage by league, season, and team."""
    results = {}
    
    # By league and season
    ha_ls = master.groupby(['Div', 'Season']).apply(
        lambda x: pd.Series({
            'Home_Win_Pct': (x['FTR'] == 'H').mean(),
            'Draw_Pct': (x['FTR'] == 'D').mean(),
            'Away_Win_Pct': (x['FTR'] == 'A').mean(),
            'Home_Goals': x['FTHG'].mean(),
            'Away_Goals': x['FTAG'].mean(),
            'Goal_Diff': x['FTHG'].mean() - x['FTAG'].mean(),
            'Matches': len(x)
        })
    ).reset_index()
    results['by_league_season'] = ha_ls
    
    # By league (averaged)
    ha_league = ha_ls.groupby('Div').agg({
        'Home_Win_Pct': 'mean',
        'Draw_Pct': 'mean',
        'Away_Win_Pct': 'mean',
        'Goal_Diff': 'mean'
    }).round(3)
    results['by_league'] = ha_league
    
    # By season (averaged)
    ha_season = ha_ls.groupby('Season').agg({
        'Home_Win_Pct': 'mean',
        'Draw_Pct': 'mean',
        'Away_Win_Pct': 'mean',
        'Goal_Diff': 'mean'
    }).round(3)
    results['by_season'] = ha_season
    
    # Statistical significance per league
    sig_results = {}
    for league in master['Div'].unique():
        lg_data = master[master['Div'] == league]
        home_wins = (lg_data['FTR'] == 'H').sum()
        away_wins = (lg_data['FTR'] == 'A').sum()
        draws = (lg_data['FTR'] == 'D').sum()
        total = len(lg_data)
        draw_rate = draws / total
        expected_home_win_rate = (1 - draw_rate) / 2
        result = binomtest(home_wins, n=total - draws, p=expected_home_win_rate, alternative='greater')
        sig_results[league] = {
            'home_win_pct': home_wins / total,
            'away_win_pct': away_wins / total,
            'p_value': result.pvalue,
            'significant': result.pvalue < 0.05
        }
    results['significance'] = sig_results
    
    # COVID season comparison
    covid = '2021'
    ha_covid = master[master['Season'] == covid].groupby('Div').apply(
        lambda x: pd.Series({
            'Home_Win_Pct': (x['FTR'] == 'H').mean(),
            'Away_Win_Pct': (x['FTR'] == 'A').mean(),
            'Goal_Diff': x['FTHG'].mean() - x['FTAG'].mean()
        })
    )
    ha_other = master[master['Season'] != covid].groupby('Div').apply(
        lambda x: pd.Series({
            'Home_Win_Pct': (x['FTR'] == 'H').mean(),
            'Away_Win_Pct': (x['FTR'] == 'A').mean(),
            'Goal_Diff': x['FTHG'].mean() - x['FTAG'].mean()
        })
    )
    # Combine properly - both have MultiIndex (Div, metric)
    results['covid_vs_normal'] = pd.concat({
        'covid': ha_covid,
        'normal': ha_other
    }, axis=1)
    
    return results


def analyze_efficiency_persistence(team_matches: pd.DataFrame, min_matches: int = 20) -> Dict:
    """Analyze persistence of efficiency metrics across half-seasons and seasons."""
    results = {}
    
    # Team-season aggregates
    team_season = team_matches.groupby(['Div', 'Team', 'Season']).agg(
        Matches=('GF', 'count'),
        GF=('GF', 'sum'), GA=('GA', 'sum'),
        Shots=('Shots', 'sum'), Shots_OT=('Shots_OT', 'sum'),
        Shots_Against=('Shots_Against', 'sum'), Shots_OT_Against=('Shots_OT_Against', 'sum'),
        Points=('Points', 'sum'),
    ).reset_index()
    team_season = team_season[team_season['Matches'] >= min_matches].copy()
    team_season['GPS'] = team_season['GF'] / team_season['Shots']
    team_season['GPSOT'] = team_season['GF'] / team_season['Shots_OT']
    team_season['GAOS'] = team_season['GA'] / team_season['Shots_Against']
    team_season['PPG'] = team_season['Points'] / team_season['Matches']
    results['team_season'] = team_season
    
    # Correlation with PPG
    corr_results = {}
    for metric in ['GPS', 'GPSOT', 'GAOS', 'Shot_Accuracy']:
        if metric not in team_season.columns:
            if metric == 'Shot_Accuracy':
                team_season['Shot_Accuracy'] = team_season['Shots_OT'] / team_season['Shots']
        valid = team_season[metric].notna()
        if valid.sum() > 10:
            r, p = stats.pearsonr(team_season.loc[valid, metric], team_season.loc[valid, 'PPG'])
            corr_results[metric] = {'r': r, 'p': p, 'n': int(valid.sum())}
    results['ppg_correlations'] = corr_results
    
    # Half-season persistence
    def split_half_efficiency(group):
        group = group.sort_values('Date')
        n = len(group)
        mid = n // 2
        first = group.iloc[:mid]
        second = group.iloc[mid:]
        
        def calc_eff(df):
            shots = df['Shots'].sum()
            shots_ot = df['Shots_OT'].sum()
            gf = df['GF'].sum()
            ga = df['GA'].sum()
            shots_a = df['Shots_Against'].sum()
            shots_ot_a = df['Shots_OT_Against'].sum()
            return pd.Series({
                'GPS': gf / shots if shots > 0 else np.nan,
                'GPSOT': gf / shots_ot if shots_ot > 0 else np.nan,
                'GAOS': ga / shots_a if shots_a > 0 else np.nan,
                'GAOSOT': ga / shots_ot_a if shots_ot_a > 0 else np.nan,
            })
        
        first_eff = calc_eff(first)
        second_eff = calc_eff(second)
        first_eff.index = [c + '_First' for c in first_eff.index]
        second_eff.index = [c + '_Second' for c in second_eff.index]
        return pd.concat([first_eff, second_eff])
    
    half_eff = team_matches[
        team_matches.groupby(['Div', 'Team', 'Season']).transform('size') >= min_matches
    ].groupby(['Div', 'Team', 'Season']).apply(split_half_efficiency).reset_index()
    results['half_season'] = half_eff
    
    half_persistence = {}
    for metric in ['GPS', 'GPSOT', 'GAOS', 'GAOSOT']:
        first_col = f'{metric}_First'
        second_col = f'{metric}_Second'
        valid = half_eff[first_col].notna() & half_eff[second_col].notna()
        if valid.sum() > 10:
            r, p = stats.pearsonr(half_eff.loc[valid, first_col], half_eff.loc[valid, second_col])
            half_persistence[metric] = {'r': r, 'p': p, 'n': int(valid.sum())}
    results['half_season_persistence'] = half_persistence
    
    # Season-to-season persistence (requiring genuinely consecutive seasons in same division)
    season_order = {'1920':0, '2021':1, '2122':2, '2223':3, '2324':4, '2425':5, '2526':6}
    team_season['Season_Num'] = team_season['Season'].map(season_order)
    team_season = team_season.sort_values(['Div', 'Team', 'Season_Num'])
    
    # Identify consecutive seasons: same Div, same Team, Season_Num diff == 1
    team_season['Prev_Season_Num'] = team_season.groupby(['Div', 'Team'])['Season_Num'].shift(1)
    team_season['Consecutive'] = (team_season['Season_Num'] - team_season['Prev_Season_Num'] == 1)
    
    # Track transitions
    team_season['Prev_Div'] = team_season.groupby(['Team'])['Div'].shift(1)
    team_season['Div_Changed'] = (team_season['Div'] != team_season['Prev_Div'])
    
    # Gap seasons
    team_season['Season_Gap'] = team_season['Season_Num'] - team_season['Prev_Season_Num']
    
    # Report transition summary
    transition_summary = {
        'total_pairs': int(team_season['Prev_Season_Num'].notna().sum()),
        'consecutive_same_div': int(team_season['Consecutive'].sum()),
        'div_changed': int(team_season['Div_Changed'].sum()),
        'gap_1_season': int((team_season['Season_Gap'] == 1).sum()),
        'gap_2plus_seasons': int((team_season['Season_Gap'] > 1).sum()),
    }
    results['season_transitions'] = transition_summary
    
    # Only use consecutive same-division seasons for persistence
    consecutive = team_season[team_season['Consecutive']].copy()
    
    for metric in ['GPS', 'GPSOT', 'GAOS']:
        consecutive[f'Next_{metric}'] = consecutive.groupby(['Div', 'Team'])[metric].shift(-1)
    
    consecutive['Next_PPG'] = consecutive.groupby(['Div', 'Team'])['PPG'].shift(-1)
    
    season_persistence = {}
    for metric in ['GPS', 'GPSOT', 'GAOS']:
        v = consecutive[metric].notna() & consecutive[f'Next_{metric}'].notna()
        if v.sum() > 10:
            r, p = stats.pearsonr(consecutive.loc[v, metric], consecutive.loc[v, f'Next_{metric}'])
            season_persistence[metric] = {'r': r, 'p': p, 'n': int(v.sum())}
    
    for metric in ['GPS', 'GPSOT', 'GAOS', 'PPG']:
        v = consecutive[metric].notna() & consecutive['Next_PPG'].notna()
        if v.sum() > 10:
            r, p = stats.pearsonr(consecutive.loc[v, metric], consecutive.loc[v, 'Next_PPG'])
            season_persistence[f'{metric}_to_NextPPG'] = {'r': r, 'p': p, 'n': int(v.sum())}
    
    results['season_to_season_persistence'] = season_persistence
    
    return results


def analyze_shot_decomposition(team_season: pd.DataFrame) -> Dict:
    """Decompose goals into volume, accuracy, finishing components."""
    results = {}
    
    # Multiple regression
    X = team_season[['Shots_PG', 'Shot_Accuracy', 'Goals_Per_Shot_OT']].dropna()
    y = team_season.loc[X.index, 'PPG']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    reg = LinearRegression().fit(X_scaled, y)
    
    results['multiple_regression'] = {
        'r2': reg.score(X_scaled, y),
        'coefficients': dict(zip(X.columns, reg.coef_)),
        'intercept': reg.intercept_
    }
    
    # Log-linear decomposition
    team_season['Log_GF'] = np.log(team_season['GF'] + 1)
    team_season['Log_Shots'] = np.log(team_season['Shots'] + 1)
    team_season['Log_Acc'] = np.log(team_season['Shot_Accuracy'].clip(0.01))
    team_season['Log_GPSOT'] = np.log(team_season['Goals_Per_Shot_OT'].clip(0.01))
    
    X_log = team_season[['Log_Shots', 'Log_Acc', 'Log_GPSOT']].dropna()
    y_log = team_season.loc[X_log.index, 'Log_GF']
    reg_log = LinearRegression().fit(X_log, y_log)
    
    results['log_linear'] = {
        'r2': reg_log.score(X_log, y_log),
        'coefficients': dict(zip(X_log.columns, reg_log.coef_)),
        'intercept': reg_log.intercept_
    }
    
    # Correlation matrix
    corr_vars = ['Shots_PG', 'Shot_Accuracy', 'Goals_Per_Shot_OT', 'PPG']
    results['correlation_matrix'] = team_season[corr_vars].corr().round(3).to_dict()
    
    # Attack-Defense profiles
    team_season['Attack_Profile'] = pd.qcut(team_season['Shots_PG'], 3, labels=['Low', 'Medium', 'High'])
    team_season['Finishing_Profile'] = pd.qcut(team_season['Goals_Per_Shot_OT'], 3, labels=['Poor', 'Average', 'Clinical'])
    pivot_attack = team_season.pivot_table(values='PPG', index='Attack_Profile', columns='Finishing_Profile', aggfunc='mean')
    results['attack_profile_ppg'] = pivot_attack.round(2).to_dict()
    
    team_season['Opp_Shots_PG'] = team_season['Shots_Against'] / team_season['Matches']
    team_season['Defensive_Resilience'] = team_season['GA'] / team_season['Shots_Against']
    team_season['Def_Volume'] = pd.qcut(team_season['Opp_Shots_PG'], 3, labels=['Low', 'Med', 'High'])
    team_season['Def_Resilience'] = pd.qcut(team_season['Defensive_Resilience'], 3, labels=['Resilient', 'Average', 'Vulnerable'])
    pivot_def = team_season.pivot_table(values='PPG', index='Def_Volume', columns='Def_Resilience', aggfunc='mean')
    results['defense_profile_ppg'] = pivot_def.round(2).to_dict()
    
    return results


def analyze_ht_ft_dynamics(master: pd.DataFrame) -> Dict:
    """Analyze half-time to full-time transitions and second-half patterns."""
    results = {}
    
    # Transition matrix (normalized)
    transitions = pd.crosstab(master['HTR'], master['FTR'], normalize='index').round(3)
    results['transitions_norm'] = transitions
    
    # Transition counts
    trans_counts = pd.crosstab(master['HTR'], master['FTR'])
    results['transitions_counts'] = trans_counts
    
    # Comeback analysis
    ht_trailing_ft_not_lose = master[
        ((master['HTR'] == 'H') & (master['FTR'] == 'A')) |
        ((master['HTR'] == 'A') & (master['FTR'] == 'H')) |
        ((master['HTR'] == 'H') & (master['FTR'] == 'D')) |
        ((master['HTR'] == 'A') & (master['FTR'] == 'D'))
    ]
    results['comeback_matches'] = len(ht_trailing_ft_not_lose)
    results['comeback_rate'] = len(ht_trailing_ft_not_lose) / len(master)
    
    # Failed to hold lead
    ht_leading_ft_not_win = master[
        ((master['HTR'] == 'H') & (master['FTR'] != 'H')) |
        ((master['HTR'] == 'A') & (master['FTR'] != 'A'))
    ]
    results['failed_hold_lead_matches'] = len(ht_leading_ft_not_win)
    results['failed_hold_lead_rate'] = len(ht_leading_ft_not_win) / len(master)
    
    # Second-half goals
    master = master.copy()
    master['SH_Goals_Home'] = master['FTHG'] - master['HTHG']
    master['SH_Goals_Away'] = master['FTAG'] - master['HTAG']
    master['SH_Total_Goals'] = master['SH_Goals_Home'] + master['SH_Goals_Away']
    master['FH_Total_Goals'] = master['HTHG'] + master['HTAG']
    
    results['avg_fh_goals'] = master['FH_Total_Goals'].mean()
    results['avg_sh_goals'] = master['SH_Total_Goals'].mean()
    results['sh_goal_share'] = master['SH_Total_Goals'].sum() / (master['FH_Total_Goals'].sum() + master['SH_Total_Goals'].sum())
    
    # By league
    sh_by_league = master.groupby('Div').apply(
        lambda x: pd.Series({
            'FH_Goals': x['FH_Total_Goals'].sum(),
            'SH_Goals': x['SH_Total_Goals'].sum(),
            'SH_Pct': x['SH_Total_Goals'].sum() / (x['FH_Total_Goals'].sum() + x['SH_Total_Goals'].sum())
        })
    )
    results['sh_by_league'] = sh_by_league
    
    # SH goals by HT result
    sh_by_ht = master.groupby('HTR')['SH_Total_Goals'].mean()
    results['sh_by_ht_result'] = sh_by_ht
    
    # Comeback rates by league
    comeback_data = []
    for league in master['Div'].unique():
        lg = master[master['Div'] == league]
        trailing = ((lg['HTR'] == 'H') & (lg['FTR'] == 'A')) | ((lg['HTR'] == 'A') & (lg['FTR'] == 'H'))
        drawing = ((lg['HTR'] == 'H') & (lg['FTR'] == 'D')) | ((lg['HTR'] == 'A') & (lg['FTR'] == 'D'))
        total_trailing = (lg['HTR'] != 'D').sum()
        if total_trailing > 0:
            comeback_data.append({
                'League': league,
                'Trailing_Win_Pct': trailing.sum() / total_trailing,
                'Trailing_Draw_Pct': drawing.sum() / total_trailing,
                'Trailing_Avoid_Loss_Pct': (trailing.sum() + drawing.sum()) / total_trailing
            })
    results['comeback_by_league'] = pd.DataFrame(comeback_data)
    
    # Team-level comeback ability
    team_comebacks = pd.concat([
        master.assign(Team=master['HomeTeam'], HT_GD=master['HTHG']-master['HTAG'], FT_GD=master['FTHG']-master['FTAG']),
        master.assign(Team=master['AwayTeam'], HT_GD=master['HTAG']-master['HTHG'], FT_GD=master['FTAG']-master['FTHG'])
    ])
    trailing = team_comebacks[team_comebacks['HT_GD'] < 0].copy()
    trailing['Recovered'] = (trailing['FT_GD'] > 0).astype(int)
    trailing['Avoided_Loss'] = (trailing['FT_GD'] >= 0).astype(int)
    
    team_comeback_stats = trailing.groupby('Team').agg(
        Trailing_Count=('Recovered', 'count'),
        Recoveries=('Recovered', 'sum'),
        Avoided_Losses=('Avoided_Loss', 'sum')
    ).reset_index()
    team_comeback_stats = team_comeback_stats[team_comeback_stats['Trailing_Count'] >= 10]
    team_comeback_stats['Recovery_Rate'] = team_comeback_stats['Recoveries'] / team_comeback_stats['Trailing_Count']
    team_comeback_stats['Avoid_Loss_Rate'] = team_comeback_stats['Avoided_Losses'] / team_comeback_stats['Trailing_Count']
    results['team_comeback_stats'] = team_comeback_stats.sort_values('Recovery_Rate', ascending=False)
    
    return results


def analyze_consistency(team_matches: pd.DataFrame, min_matches: int = 20) -> Dict:
    """Analyze team performance consistency."""
    results = {}
    
    # Season-level consistency
    consistency = team_matches.groupby(['Div', 'Team', 'Season']).agg(
        Points_Std=('Points', 'std'),
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
    consistency['Points_CV'] = consistency['Points_Std'] / consistency['PPG']
    consistency['GF_CV'] = consistency['GF_Std'] / consistency['GF_PG']
    consistency['GA_CV'] = consistency['GA_Std'] / consistency['GA_PG']
    results['season_consistency'] = consistency
    
    # Team-level (3+ seasons)
    team_consistency = consistency.groupby('Team').agg(
        Seasons=('Season', 'count'),
        Avg_PPG=('PPG', 'mean'),
        Avg_Points_CV=('Points_CV', 'mean'),
        Avg_GF_CV=('GF_CV', 'mean'),
        Avg_GA_CV=('GA_CV', 'mean')
    ).reset_index()
    team_consistency = team_consistency[team_consistency['Seasons'] >= 3]
    results['team_consistency'] = team_consistency
    
    # Most/least consistent
    results['most_consistent'] = team_consistency.nsmallest(10, 'Avg_Points_CV')
    results['least_consistent'] = team_consistency.nlargest(10, 'Avg_Points_CV')
    
    # Summary stats
    results['median_points_cv'] = consistency['Points_CV'].median()
    results['median_gf_cv'] = consistency['GF_CV'].median()
    results['median_ga_cv'] = consistency['GA_CV'].median()
    
    return results


def analyze_discipline(team_matches: pd.DataFrame, master: pd.DataFrame, min_matches: int = 20) -> Dict:
    """Analyze disciplinary patterns and their association with results."""
    results = {}
    
    # Team-season discipline
    team_season_disc = team_matches.groupby(['Div', 'Team', 'Season']).agg(
        Matches=('GF', 'count'),
        Yellow=('Yellow', 'sum'),
        Red=('Red', 'sum'),
        Fouls=('Fouls', 'sum'),
        Fouls_Against=('Fouls_Against', 'sum'),
        Points=('Points', 'sum'),
    ).reset_index()
    team_season_disc = team_season_disc[team_season_disc['Matches'] >= min_matches].copy()
    team_season_disc['PPG'] = team_season_disc['Points'] / team_season_disc['Matches']
    team_season_disc['Yellow_PG'] = team_season_disc['Yellow'] / team_season_disc['Matches']
    team_season_disc['Red_PG'] = team_season_disc['Red'] / team_season_disc['Matches']
    team_season_disc['Fouls_PG'] = team_season_disc['Fouls'] / team_season_disc['Matches']
    team_season_disc['Cards_PG'] = team_season_disc['Yellow_PG'] + team_season_disc['Red_PG']
    
    # Correlation with PPG
    disc_corrs = {}
    for metric in ['Yellow_PG', 'Red_PG', 'Fouls_PG', 'Cards_PG']:
        r, p = stats.pearsonr(team_season_disc[metric], team_season_disc['PPG'])
        disc_corrs[metric] = {'r': r, 'p': p}
    results['ppg_correlations'] = disc_corrs
    
    # By venue
    venue_disc = team_matches.groupby('Venue').agg(
        Yellow_PG=('Yellow', 'mean'),
        Red_PG=('Red', 'mean'),
        Fouls_PG=('Fouls', 'mean'),
        Matches=('Yellow', 'count')
    ).reset_index()
    results['by_venue'] = venue_disc
    
    # Red card impact
    home_red = master[master['HR'] > 0]
    away_red = master[master['AR'] > 0]
    
    baseline_home_win = (master['FTR'] == 'H').mean()
    baseline_away_win = (master['FTR'] == 'A').mean()
    baseline_draw = (master['FTR'] == 'D').mean()
    
    results['red_card_impact'] = {
        'baseline': {'home_win': baseline_home_win, 'draw': baseline_draw, 'away_win': baseline_away_win},
        'home_red': {
            'home_win': (home_red['FTR'] == 'H').mean(),
            'draw': (home_red['FTR'] == 'D').mean(),
            'away_win': (home_red['FTR'] == 'A').mean(),
            'n': len(home_red)
        },
        'away_red': {
            'home_win': (away_red['FTR'] == 'H').mean(),
            'draw': (away_red['FTR'] == 'D').mean(),
            'away_win': (away_red['FTR'] == 'A').mean(),
            'n': len(away_red)
        }
    }
    
    # Red card by HT result
    results['home_red_by_ht'] = home_red.groupby('HTR')['FTR'].value_counts(normalize=True).unstack(fill_value=0).round(3).to_dict()
    results['away_red_by_ht'] = away_red.groupby('HTR')['FTR'].value_counts(normalize=True).unstack(fill_value=0).round(3).to_dict()
    
    # Match-level: cards/fouls vs result
    team_matches = team_matches.copy()
    team_matches['Card_Count'] = team_matches['Yellow'] + team_matches['Red']
    result_numeric = team_matches['FT_Result'].map({'W': 2, 'D': 1, 'L': 0})
    valid = result_numeric.notna() & team_matches['Card_Count'].notna() & team_matches['Fouls'].notna() & team_matches['Fouls_Against'].notna()
    
    r_cards, p_cards = stats.pearsonr(team_matches.loc[valid, 'Card_Count'], result_numeric.loc[valid])
    r_fouls, p_fouls = stats.pearsonr(team_matches.loc[valid, 'Fouls'], result_numeric.loc[valid])
    r_fouls_against, p_fouls_against = stats.pearsonr(team_matches.loc[valid, 'Fouls_Against'], result_numeric.loc[valid])
    
    results['match_level_correlations'] = {
        'cards_vs_result': {'r': r_cards, 'p': p_cards},
        'fouls_vs_result': {'r': r_fouls, 'p': p_fouls},
        'fouls_drawn_vs_result': {'r': r_fouls_against, 'p': p_fouls_against}
    }
    
    # Foul differential
    team_season_disc2 = team_matches.groupby(['Div', 'Team', 'Season']).agg(
        Matches=('GF', 'count'),
        Fouls=('Fouls', 'sum'),
        Fouls_Against=('Fouls_Against', 'sum'),
        Points=('Points', 'sum'),
    ).reset_index()
    team_season_disc2 = team_season_disc2[team_season_disc2['Matches'] >= min_matches].copy()
    team_season_disc2['PPG'] = team_season_disc2['Points'] / team_season_disc2['Matches']
    team_season_disc2['Foul_Diff_PG'] = (team_season_disc2['Fouls'] - team_season_disc2['Fouls_Against']) / team_season_disc2['Matches']
    
    r_fd, p_fd = stats.pearsonr(team_season_disc2['Foul_Diff_PG'], team_season_disc2['PPG'])
    results['foul_differential_correlation'] = {'r': r_fd, 'p': p_fd}
    
    # Return the team-season discipline dataframes for plotting
    results['team_season_disc'] = team_season_disc
    results['team_season_disc2'] = team_season_disc2
    
    return results


def run_robustness_checks(team_matches: pd.DataFrame, master: pd.DataFrame) -> Dict:
    """Run robustness and sensitivity checks."""
    results = {}
    
    # Sample size sensitivity
    team_season = team_matches.groupby(['Div', 'Team', 'Season']).agg(
        Matches=('GF', 'count'),
        GF=('GF', 'sum'), GA=('GA', 'sum'),
        Shots=('Shots', 'sum'), Shots_OT=('Shots_OT', 'sum'),
        Shots_Against=('Shots_Against', 'sum'),
        Points=('Points', 'sum'),
    ).reset_index()
    team_season['GPS'] = team_season['GF'] / team_season['Shots']
    team_season['PPG'] = team_season['Points'] / team_season['Matches']
    
    sample_sensitivity = {}
    for min_m in [10, 20, 30]:
        ts = team_season[team_season['Matches'] >= min_m]
        if len(ts) > 10:
            r, p = stats.pearsonr(ts['GPS'], ts['PPG'])
            sample_sensitivity[f'min_{min_m}'] = {'n': len(ts), 'r': r, 'p': p}
    results['sample_size_sensitivity'] = sample_sensitivity
    
    # League tier comparison
    top_tier = ['E0', 'SP1', 'D1', 'I1', 'F1']
    lower_tier = ['E1', 'E2', 'E3']
    
    def half_persistence(df):
        df = df.sort_values('Date')
        n = len(df)
        mid = n // 2
        first = df.iloc[:mid]
        second = df.iloc[mid:]
        def eff(g):
            return pd.Series({
                'GPS': g['GF'].sum() / g['Shots'].sum() if g['Shots'].sum() > 0 else np.nan,
                'GPSOT': g['GF'].sum() / g['Shots_OT'].sum() if g['Shots_OT'].sum() > 0 else np.nan,
                'GAOS': g['GA'].sum() / g['Shots_Against'].sum() if g['Shots_Against'].sum() > 0 else np.nan,
            })
        f = eff(first)
        s = eff(second)
        f.index = [c + '_F' for c in f.index]
        s.index = [c + '_S' for c in s.index]
        return pd.concat([f, s])
    
    tier_persistence = {}
    for tier_name, tiers in [('Top_5', top_tier), ('Lower_Eng', lower_tier)]:
        tier_data = team_matches[team_matches['Div'].isin(tiers)]
        if len(tier_data) == 0:
            continue
        half_eff = tier_data.groupby(['Div', 'Team', 'Season']).apply(half_persistence).reset_index()
        for metric in ['GPS', 'GPSOT', 'GAOS']:
            f_col = f'{metric}_F'
            s_col = f'{metric}_S'
            if f_col in half_eff.columns and s_col in half_eff.columns:
                v = half_eff[f_col].notna() & half_eff[s_col].notna()
                if v.sum() > 10:
                    r, p = stats.pearsonr(half_eff.loc[v, f_col], half_eff.loc[v, s_col])
                    tier_persistence[f'{tier_name}_{metric}'] = {'r': r, 'p': p, 'n': int(v.sum())}
    results['tier_persistence'] = tier_persistence
    
    return results