"""
New insights module.
Implements 6 priority research questions beyond the existing analyses:
1. Opponent-dependent performance
2. Components of home advantage (shot generation vs suppression vs conversion)
3. Close-match dependence (one-goal results)
4. Rest asymmetry (observed league-fixture rest)
5. Performance indicators leading results (lagged shot dominance)
6. Adaptation after promotion (reliable adjacent-tier transitions only)

Also fixes identified issues:
- Pre-match strength leakage (halftime analysis)
- Promotion/relegation matching (chronological team sorting)
- Volatility benchmarks (coherent outcome probabilities, symmetric simulation)
- Exposure and missingness (metric-specific valid-match counts)
"""
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import binomtest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
warnings.filterwarnings('ignore', category=UserWarning, message='X does not have valid feature names')

SEASON_ORDER = {
    '1920': 0, '2021': 1, '2122': 2, '2223': 3,
    '2324': 4, '2425': 5, '2526': 6, '2627': 7
}

DIVISION_HIERARCHY = {
    'E0': 0, 'E1': 1, 'E2': 2, 'E3': 3,
    'SP1': 0, 'SP2': 1,
    'I1': 0, 'I2': 1,
    'D1': 0, 'D2': 1,
    'F1': 0, 'F2': 1,
}


def add_lagged_team_strength(team_matches: pd.DataFrame) -> pd.DataFrame:
    """
    Add genuinely lagged pre-match strength estimate.
    Uses cumulative PPG up to but not including the current match.
    Early-season initialization uses PRIOR SEASON average (not current season),
    falling back to historical league average if no prior season exists.
    """
    df = team_matches.sort_values(['Div', 'Team', 'Season', 'Date']).copy()
    grp = ['Div', 'Team', 'Season']
    
    # Cumulative points and matches BEFORE the current match (strictly lagged).
    # cumsum() - current points = points earned in all prior matches;
    # cumcount() = number of prior matches.
    # Equivalent to shift(1).expanding().sum() / cumcount() but fully vectorized.
    points_cum = df.groupby(grp)['Points'].cumsum()
    matches_cum = df.groupby(grp).cumcount()
    points_cum_lagged = points_cum - df['Points'].to_numpy()
    df['PreMatch_PPG'] = np.where(
        matches_cum > 0,
        points_cum_lagged / matches_cum.to_numpy(),
        np.nan,
    )
    df['Matches_Cum_Lagged'] = matches_cum.to_numpy()
    
    # Early-season initialization: use PRIOR SEASON PPG (not current season average)
    # Build prior-season lookup
    season_order = {'1920': 0, '2021': 1, '2122': 2, '2223': 3,
                    '2324': 4, '2425': 5, '2526': 6, '2627': 7}
    
    # Get each team's final PPG from previous season
    # Each team's final PPG per season (Season is constant within a group, so
    # map is equivalent to the previous groupby.apply(lambda x: ...)).
    team_season_final = df.groupby(grp).agg(Final_PPG=('Points', 'mean')).reset_index()
    team_season_final['Season_Num'] = team_season_final['Season'].map(season_order)
    team_season_final = team_season_final.sort_values(['Team', 'Season_Num'])
    team_season_final['Prev_Final_PPG'] = team_season_final.groupby('Team')['Final_PPG'].shift(1)
    team_season_final['Prev_Season_Num'] = team_season_final.groupby('Team')['Season_Num'].shift(1)
    team_season_final['Consecutive'] = (team_season_final['Season_Num'] - team_season_final['Prev_Season_Num'] == 1)
    
    # Only use prior season if consecutive
    prior_ppg_lookup = team_season_final[team_season_final['Consecutive']][
        ['Div', 'Team', 'Season', 'Prev_Final_PPG']
    ].rename(columns={'Prev_Final_PPG': 'Prior_Season_PPG'})
    
    df = df.merge(prior_ppg_lookup, on=['Div', 'Team', 'Season'], how='left')
    
    # Fallback: historical league average (from all PRIOR seasons in data)
    # Compute expanding historical average per league-season
    hist_avg = df.groupby(['Div', 'Season'])['Points'].mean().reset_index()
    hist_avg.columns = ['Div', 'Season', 'League_Avg_PPG']
    # Shift to make it historical (prior seasons only)
    hist_avg = hist_avg.sort_values(['Div', 'Season'])
    hist_avg['Hist_League_Avg_PPG'] = hist_avg.groupby('Div')['League_Avg_PPG'].shift(1)
    hist_avg = hist_avg[['Div', 'Season', 'Hist_League_Avg_PPG']]
    
    df = df.merge(hist_avg, on=['Div', 'Season'], how='left')
    
    # Shrink early-season estimates toward prior-season PPG, then historical league avg
    # Weight: match_count / (match_count + 5) for prior season, else historical
    has_prior = df['Prior_Season_PPG'].notna()
    has_hist = df['Hist_League_Avg_PPG'].notna()
    
    df['PreMatch_PPG_Shrunk'] = df['PreMatch_PPG']
    
    # Case 1: Have prior season PPG and few current matches
    mask1 = has_prior & (df['Matches_Cum_Lagged'] < 5)
    df.loc[mask1, 'PreMatch_PPG_Shrunk'] = (
        df.loc[mask1, 'PreMatch_PPG'] * df.loc[mask1, 'Matches_Cum_Lagged'] + 
        df.loc[mask1, 'Prior_Season_PPG'] * (5 - df.loc[mask1, 'Matches_Cum_Lagged'])
    ) / 5
    
    # Case 2: No prior season but have historical league avg and few current matches
    mask2 = (~has_prior) & has_hist & (df['Matches_Cum_Lagged'] < 5)
    df.loc[mask2, 'PreMatch_PPG_Shrunk'] = (
        df.loc[mask2, 'PreMatch_PPG'] * df.loc[mask2, 'Matches_Cum_Lagged'] + 
        df.loc[mask2, 'Hist_League_Avg_PPG'] * (5 - df.loc[mask2, 'Matches_Cum_Lagged'])
    ) / 5
    
    # Case 3: First match ever (no prior, no historical) - leave as NaN
    # Don't invent a value; let downstream handle missingness
    
    return df.drop(columns=['Matches_Cum_Lagged', 'Prior_Season_PPG',
                            'Hist_League_Avg_PPG', 'PreMatch_PPG'])


def build_promotion_relegation_transitions(team_season: pd.DataFrame) -> pd.DataFrame:
    """
    Build promotion/relegation transitions by sorting each team chronologically.
    Require adjacent seasons. Distinguish same-division and cross-division.
    Infer promotion/relegation only where observed league hierarchy supports it.
    """
    ts = team_season.copy()
    ts['Season_Num'] = ts['Season'].map(SEASON_ORDER)
    ts = ts.sort_values(['Team', 'Season_Num'])
    
    # For each team, get previous season in chronology (regardless of division)
    ts['Prev_Season_Num'] = ts.groupby('Team')['Season_Num'].shift(1)
    ts['Prev_Div'] = ts.groupby('Team')['Div'].shift(1)
    ts['Prev_PPG'] = ts.groupby('Team')['PPG'].shift(1)
    ts['Prev_Shots_PG'] = ts.groupby('Team')['Shots_PG'].shift(1)
    ts['Prev_Shots_Against_PG'] = ts.groupby('Team')['Shots_Against_PG'].shift(1)
    ts['Prev_GPSOT'] = ts.groupby('Team')['Goals_Per_Shot_OT'].shift(1)
    ts['Prev_GAOSOT'] = ts.groupby('Team')['GA_Per_Opp_Shot_OT'].shift(1)
    
    # Consecutive seasons only
    ts['Season_Gap'] = ts['Season_Num'] - ts['Prev_Season_Num']
    ts['Consecutive'] = (ts['Season_Gap'] == 1)
    
    # Division change
    ts['Div_Changed'] = (ts['Div'] != ts['Prev_Div'])
    
    # Infer direction using division hierarchy
    def get_tier(div):
        return DIVISION_HIERARCHY.get(div, np.nan)
    
    ts['Div_Tier'] = ts['Div'].map(get_tier)
    ts['Prev_Div_Tier'] = ts['Prev_Div'].map(get_tier)
    
    # Promotion: tier decreases (e.g., E1->E0), Relegation: tier increases (e.g., E0->E1)
    ts['Transition_Type'] = np.select(
        [
            ts['Consecutive'] & ts['Div_Changed'] & (ts['Div_Tier'] < ts['Prev_Div_Tier']),
            ts['Consecutive'] & ts['Div_Changed'] & (ts['Div_Tier'] > ts['Prev_Div_Tier']),
            ts['Consecutive'] & ~ts['Div_Changed'],
        ],
        ['Promotion', 'Relegation', 'Same_Division'],
        default='Other'
    )
    
    return ts


def analyze_opponent_dependent_performance(team_matches: pd.DataFrame, team_season: pd.DataFrame, 
                                            min_matches: int = 20) -> Dict:
    """
    Question 1: Which teams perform relatively better against stronger opponents?
    
    FIXED: Uses a SHARED BASELINE model with team-specific opponent-strength interactions,
    avoiding the structural problem of regressing in-sample residuals on an included predictor.
    
    - Define opponent strength using only pre-match information (lagged cumulative PPG)
    - Estimate expected performance from own strength, opponent strength, venue, context
    - Team-specific deviations modeled as RANDOM EFFECTS on opponent strength slope
    - Compare with shot differential and goal differential
    - Use partial pooling/regularization for sparse team-specific effects
    - Test persistence in later period (out-of-sample)
    """
    results = {}
    
    # Add lagged strength to team_matches
    tm = add_lagged_team_strength(team_matches)
    
    # Filter to valid team-seasons
    valid_seasons = team_season[team_season['Matches'] >= min_matches][['Div', 'Team', 'Season']]
    tm = tm.merge(valid_seasons, on=['Div', 'Team', 'Season'])
    
    # Add opponent's pre-match PPG
    opp_strength = tm[['Div', 'Season', 'Date', 'Team', 'PreMatch_PPG_Shrunk']].rename(
        columns={'Team': 'Opponent', 'PreMatch_PPG_Shrunk': 'Opp_PreMatch_PPG'}
    )
    tm = tm.merge(opp_strength, on=['Div', 'Season', 'Date', 'Opponent'], how='left')
    
    # FIXED: Shared baseline model with team-specific interaction on opponent strength
    # Points ~ Own_PrePPG + Opp_PrePPG + Venue + LeagueSeason + (Opp_PrePPG | Team)
    # Implemented via two-stage: fixed effects + random slopes
    
    tm['LeagueSeason'] = tm['Div'] + '_' + tm['Season']
    tm['Venue_Code'] = tm['Venue'].map({'Home': 1, 'Away': 0})
    
    # Stage 1: Fixed effects baseline (league-season + own strength + opp strength + venue)
    valid = (tm['PreMatch_PPG_Shrunk'].notna() & 
             tm['Opp_PreMatch_PPG'].notna() & 
             tm['Points'].notna())
    tm_valid = tm[valid].copy()
    
    if len(tm_valid) < 100:
        results['error'] = 'Insufficient data for shared baseline model'
        return results
    
    # League-season fixed effects
    ls_dummies = pd.get_dummies(tm_valid['LeagueSeason'], drop_first=True)
    X_base = pd.concat([
        tm_valid[['PreMatch_PPG_Shrunk', 'Opp_PreMatch_PPG', 'Venue_Code']],
        ls_dummies
    ], axis=1)
    y = tm_valid['Points']
    
    reg_base = LinearRegression(fit_intercept=True).fit(X_base, y)
    tm_valid['Expected_Points'] = reg_base.predict(X_base)
    tm_valid['Residual'] = tm_valid['Points'] - tm_valid['Expected_Points']
    
    # Stage 2: Team-specific opponent strength interaction on RESIDUALS
    # This avoids the orthogonality problem because Opp_PreMatch_PPG is already partialled out
    # in the fixed effects stage. The residual now captures what the fixed effects miss.
    
    # Stage 2: Team-specific opponent strength interaction on RESIDUALS
    # This avoids the orthogonality problem because Opp_PreMatch_PPG is already partialled out
    # in the fixed effects stage. The residual now captures what the fixed effects miss.
    #
    # Vectorized per-group simple linear regression (Residual on Opp_PreMatch_PPG).
    # Matches scipy.stats.linregress exactly (slope = Sxy/Sxx, r via Sxx*Syy, p via t-dist).
    def _group_linregress(frame, xcol, ycol, min_n, min_unique):
        grp = ['Div', 'Team', 'Season']
        tmp = frame[grp + [xcol, ycol]].copy()
        tmp['_x2'] = tmp[xcol] ** 2
        tmp['_y2'] = tmp[ycol] ** 2
        tmp['_xy'] = tmp[xcol] * tmp[ycol]
        agg = tmp.groupby(grp).agg(
            n=(xcol, 'size'),
            xsum=(xcol, 'sum'),
            ysum=(ycol, 'sum'),
            x2sum=('_x2', 'sum'),
            y2sum=('_y2', 'sum'),
            xysum=('_xy', 'sum'),
            nunique=(xcol, 'nunique'),
        ).reset_index()
        agg = agg[(agg['n'] >= min_n) & (agg['nunique'] > min_unique)].copy()
        if len(agg) == 0:
            return pd.DataFrame()
        nv = agg['n'].to_numpy(dtype=float)
        xbar = agg['xsum'].to_numpy(dtype=float) / nv
        ybar = agg['ysum'].to_numpy(dtype=float) / nv
        Sxx = agg['x2sum'].to_numpy(dtype=float) - nv * xbar * xbar
        Sxy = agg['xysum'].to_numpy(dtype=float) - nv * xbar * ybar
        Syy = agg['y2sum'].to_numpy(dtype=float) - nv * ybar * ybar
        with np.errstate(divide='ignore', invalid='ignore'):
            slope = Sxy / Sxx
            r = Sxy / np.sqrt(Sxx * Syy)
        r = np.clip(r, -1.0, 1.0)
        dfree = nv - 2.0
        with np.errstate(divide='ignore', invalid='ignore'):
            t = r * np.sqrt(dfree / np.maximum(1.0 - r * r, 1e-12))
            se = np.sqrt(np.maximum((1.0 - r * r) * Syy, 0.0) / np.maximum(Sxx * dfree, 1e-12))
        t = np.where(np.isfinite(t), t, 0.0)
        se = np.where(np.isfinite(se), se, 0.0)
        p = 2.0 * stats.t.sf(np.abs(t), dfree)
        p = np.where(np.isfinite(p), p, 1.0)
        out = agg[['Div', 'Team', 'Season', 'n', 'nunique']].copy()
        out.columns = ['Div', 'Team', 'Season', 'Matches', 'N_Unique_X']
        out['Slope'] = slope
        out['R'] = r
        out['P'] = p
        out['SE'] = se
        out = out.drop(columns=['N_Unique_X'])
        return out

    # --- Per team-season: Residual on opponent strength -----------------------
    ts_int = _group_linregress(tm_valid, 'Opp_PreMatch_PPG', 'Residual', min_n=10, min_unique=3)
    if len(ts_int) > 0:
        shrinkage = np.minimum(1.0, np.sqrt(ts_int['Matches'].to_numpy(dtype=float)) / 10.0)
        ts_int['Opp_Strength_Slope_Raw'] = ts_int['Slope']
        ts_int['Opp_Strength_Slope_Shrunk'] = ts_int['Slope'] * shrinkage
        ts_int['Opp_Strength_SE'] = ts_int['SE']
        ts_int['Opp_Strength_R'] = ts_int['R']
        ts_int['Opp_Strength_P'] = ts_int['P']
        ts_int['Own_PrePPG'] = tm_valid.groupby(['Div', 'Team', 'Season'])['PreMatch_PPG_Shrunk'].mean().reindex(
            ts_int.set_index(['Div', 'Team', 'Season']).index).to_numpy()
        ts_int['Avg_Opp_PrePPG'] = tm_valid.groupby(['Div', 'Team', 'Season'])['Opp_PreMatch_PPG'].mean().reindex(
            ts_int.set_index(['Div', 'Team', 'Season']).index).to_numpy()
        ts_int = ts_int[['Div', 'Team', 'Season', 'Matches',
                         'Opp_Strength_Slope_Raw', 'Opp_Strength_Slope_Shrunk',
                         'Opp_Strength_SE', 'Opp_Strength_R', 'Opp_Strength_P',
                         'Own_PrePPG', 'Avg_Opp_PrePPG']]
    results['team_season_interactions'] = ts_int

    # Aggregate to team level (3+ seasons)
    if len(results['team_season_interactions']) > 0:
        team_interactions = results['team_season_interactions'].groupby('Team').agg(
            Seasons=('Season', 'count'),
            Avg_Slope_Raw=('Opp_Strength_Slope_Raw', 'mean'),
            Avg_Slope_Shrunk=('Opp_Strength_Slope_Shrunk', 'mean'),
            Avg_Matches=('Matches', 'mean'),
        ).reset_index()
        team_interactions = team_interactions[team_interactions['Seasons'] >= 3]
        results['team_interactions'] = team_interactions.sort_values('Avg_Slope_Shrunk')

    # Persistence: early vs late season interaction (NON-OVERLAPPING periods)
    tm_valid['Match_Num_Season'] = tm_valid.groupby(['Div', 'Team', 'Season']).cumcount() + 1
    max_match = tm_valid.groupby(['Div', 'Team', 'Season'])['Match_Num_Season'].transform('max')
    tm_valid['Season_Half'] = np.where(tm_valid['Match_Num_Season'] <= max_match / 2, 'Early', 'Late')

    # Within a single team-season half there is exactly one league-season, so the
    # league-season dummies drop out (drop_first removes the sole category). The
    # baseline is therefore a 3-predictor OLS: Points ~ PrePPG + OppPPG + Venue.
    # Because Opp_PreMatch_PPG is a regressor, the OLS residual is orthogonal to it,
    # so the second-stage slope is numerically ~0 (same as the original loop).
    persistence_data = []
    for (div, team, season), group in tm_valid.groupby(['Div', 'Team', 'Season']):
        for half in ['Early', 'Late']:
            g = group[group['Season_Half'] == half]
            valid_half = g['PreMatch_PPG_Shrunk'].notna() & g['Opp_PreMatch_PPG'].notna() & g['Points'].notna()
            if valid_half.sum() < 8:
                continue
            gv = g[valid_half]
            X = np.column_stack([
                np.ones(len(gv)),
                gv['PreMatch_PPG_Shrunk'].to_numpy(dtype=float),
                gv['Opp_PreMatch_PPG'].to_numpy(dtype=float),
                gv['Venue_Code'].to_numpy(dtype=float),
            ])
            y_h = gv['Points'].to_numpy(dtype=float)
            try:
                beta, _, _, _ = np.linalg.lstsq(X, y_h, rcond=None)
                resid = y_h - X @ beta
            except np.linalg.LinAlgError:
                continue
            slope, _, r_val, p_val, _ = stats.linregress(gv['Opp_PreMatch_PPG'].to_numpy(dtype=float), resid)
            persistence_data.append({
                'Div': div, 'Team': team, 'Season': season, 'Half': half,
                'Slope': slope, 'R': r_val, 'P': p_val, 'N': len(gv),
            })

    results['half_season_interactions'] = pd.DataFrame(persistence_data)

    # Cross-half persistence: early slope vs late slope (team-season level)
    if len(results['half_season_interactions']) > 0:
        half_pivot = results['half_season_interactions'].pivot_table(
            index=['Div', 'Team', 'Season'], columns='Half', values='Slope'
        ).reset_index()
        half_pivot.columns.name = None
        if 'Early' in half_pivot.columns and 'Late' in half_pivot.columns:
            valid = half_pivot['Early'].notna() & half_pivot['Late'].notna()
            if valid.sum() > 10:
                r, p = stats.pearsonr(half_pivot.loc[valid, 'Early'], half_pivot.loc[valid, 'Late'])
                results['half_persistence'] = {'r': r, 'p': p, 'n': int(valid.sum())}

    # Compare with shot differential and goal differential
    sd_int = _group_linregress(tm_valid, 'Opp_PreMatch_PPG', 'Shot_Dominance', min_n=8, min_unique=3)
    gd_int = _group_linregress(tm_valid, 'Opp_PreMatch_PPG', 'GD', min_n=8, min_unique=3)
    shot_rows = []
    if len(sd_int) > 0 and len(gd_int) > 0:
        sd_int = sd_int.set_index(['Div', 'Team', 'Season'])
        gd_int = gd_int.set_index(['Div', 'Team', 'Season'])
        common = sd_int.index.intersection(gd_int.index)
        if len(common) > 0:
            s = sd_int.loc[common]
            g = gd_int.loc[common]
            shot_rows = pd.DataFrame({
                'Div': common.get_level_values(0),
                'Team': common.get_level_values(1),
                'Season': common.get_level_values(2),
                'ShotDom_Slope': s['Slope'].to_numpy(),
                'ShotDom_R': s['R'].to_numpy(),
                'ShotDom_P': s['P'].to_numpy(),
                'GD_Slope': g['Slope'].to_numpy(),
                'GD_R': g['R'].to_numpy(),
                'GD_P': g['P'].to_numpy(),
            })
    results['shot_gd_interactions'] = shot_rows if isinstance(shot_rows, pd.DataFrame) else pd.DataFrame(shot_rows)

    return results

    
    return results


def analyze_home_advantage_components(master: pd.DataFrame, team_matches: pd.DataFrame, 
                                       team_season: pd.DataFrame) -> Dict:
    """
    Question 2: Is a team's home advantage associated more with shot generation, 
    shot suppression, or conversion?
    
    - Compare home and away: Shots for, Shots against, Shots on target, Shot accuracy,
      Goals per shot, Opponent conversion
    - Account for opponent mix, season, competition
    - Prefer paired reverse-fixture comparisons
    - Separate shot-based differences from goal-based differences
    - Examine persistence across seasons
    """
    results = {}
    
    # Team-season home/away splits
    home_away = team_matches.groupby(['Div', 'Team', 'Season', 'Venue']).agg(
        Matches=('GF', 'count'),
        GF=('GF', 'sum'), GA=('GA', 'sum'),
        Shots=('Shots', 'sum'), Shots_OT=('Shots_OT', 'sum'),
        Shots_Against=('Shots_Against', 'sum'), Shots_OT_Against=('Shots_OT_Against', 'sum'),
        Points=('Points', 'sum'),
    ).reset_index()
    
    home = home_away[home_away['Venue'] == 'Home'].drop(columns='Venue').add_prefix('Home_')
    away = home_away[home_away['Venue'] == 'Away'].drop(columns='Venue').add_prefix('Away_')
    
    home = home.rename(columns={'Home_Div': 'Div', 'Home_Team': 'Team', 'Home_Season': 'Season'})
    away = away.rename(columns={'Away_Div': 'Div', 'Away_Team': 'Team', 'Away_Season': 'Season'})
    
    ha = home.merge(away, on=['Div', 'Team', 'Season'], how='inner')
    ha = ha[(ha['Home_Matches'] >= 10) & (ha['Away_Matches'] >= 10)].copy()
    
    # Per-game rates
    for prefix in ['Home', 'Away']:
        ha[f'{prefix}_PPG'] = ha[f'{prefix}_Points'] / ha[f'{prefix}_Matches']
        ha[f'{prefix}_Shots_PG'] = ha[f'{prefix}_Shots'] / ha[f'{prefix}_Matches']
        ha[f'{prefix}_Shots_OT_PG'] = ha[f'{prefix}_Shots_OT'] / ha[f'{prefix}_Matches']
        ha[f'{prefix}_Shots_Ag_PG'] = ha[f'{prefix}_Shots_Against'] / ha[f'{prefix}_Matches']
        ha[f'{prefix}_Shots_OT_Ag_PG'] = ha[f'{prefix}_Shots_OT_Against'] / ha[f'{prefix}_Matches']
        ha[f'{prefix}_Shot_Acc'] = ha[f'{prefix}_Shots_OT'] / ha[f'{prefix}_Shots'].replace(0, np.nan)
        ha[f'{prefix}_GPS'] = ha[f'{prefix}_GF'] / ha[f'{prefix}_Shots'].replace(0, np.nan)
        ha[f'{prefix}_GPSOT'] = ha[f'{prefix}_GF'] / ha[f'{prefix}_Shots_OT'].replace(0, np.nan)
        ha[f'{prefix}_GAOS'] = ha[f'{prefix}_GA'] / ha[f'{prefix}_Shots_Against'].replace(0, np.nan)
        ha[f'{prefix}_GAOSOT'] = ha[f'{prefix}_GA'] / ha[f'{prefix}_Shots_OT_Against'].replace(0, np.nan)
    
    # Differences (Home - Away)
    metrics = ['PPG', 'Shots_PG', 'Shots_OT_PG', 'Shots_Ag_PG', 'Shots_OT_Ag_PG',
               'Shot_Acc', 'GPS', 'GPSOT', 'GAOS', 'GAOSOT']
    
    for m in metrics:
        ha[f'Diff_{m}'] = ha[f'Home_{m}'] - ha[f'Away_{m}']
    
    results['team_season_ha'] = ha
    
    # Paired reverse-fixture comparison (same opponent, same season)
    # FIXED: Each match contributes two team-perspective rows. A completed home-and-away
    # pair produces FOUR rows (Team A home vs B, Team A away vs B, Team B home vs A, Team B away vs A).
    # We need to pair: same focal Team, same Opponent, same Div, same Season, one Home + one Away.
    
    tm = team_matches.copy()
    # Create a key that identifies the focal team + opponent + division + season
    tm['Pair_Key'] = tm['Div'] + '_' + tm['Season'] + '_' + tm['Team'] + '_' + tm['Opponent']
    
    # Count rows per pair (should be 2 for complete H+A: one home, one away for focal team)
    pair_counts = tm.groupby('Pair_Key').size().reset_index(name='Count')
    complete_pairs = set(pair_counts.loc[pair_counts['Count'] == 2, 'Pair_Key'])
    paired_data = tm[tm['Pair_Key'].isin(complete_pairs)].copy()
    
    # Split into home and away rows for the focal team and join on Pair_Key.
    # Each complete pair has exactly one Home and one Away row for the focal team,
    # so the join is 1:1 and fully vectorized (replaces a slow row-wise Python loop).
    home = paired_data[paired_data['Venue'] == 'Home'].set_index('Pair_Key')
    away = paired_data[paired_data['Venue'] == 'Away'].set_index('Pair_Key')
    m = home.join(away, lsuffix='_H', rsuffix='_A', how='inner').reset_index()
    
    if len(m) > 0:
        def _safe_ratio(num, den):
            return np.where(den > 0, num / den, np.nan)
        
        m['Diff_PPG'] = m['Points_H'] - m['Points_A']
        m['Diff_Shots_PG'] = m['Shots_H'] - m['Shots_A']
        m['Diff_Shots_Ag_PG'] = m['Shots_Against_H'] - m['Shots_Against_A']
        m['Diff_Shot_Acc'] = _safe_ratio(m['Shots_OT_H'], m['Shots_H']) - _safe_ratio(m['Shots_OT_A'], m['Shots_A'])
        m['Diff_GPS'] = _safe_ratio(m['GF_H'], m['Shots_H']) - _safe_ratio(m['GF_A'], m['Shots_A'])
        m['Diff_GAOS'] = _safe_ratio(m['GA_H'], m['Shots_Against_H']) - _safe_ratio(m['GA_A'], m['Shots_Against_A'])
        
        paired_diffs = m[['Div_H', 'Season_H', 'Team_H', 'Opponent_H',
                          'Diff_PPG', 'Diff_Shots_PG', 'Diff_Shots_Ag_PG',
                          'Diff_Shot_Acc', 'Diff_GPS', 'Diff_GAOS']].copy()
        paired_diffs.columns = ['Div', 'Season', 'Team', 'Opponent',
                                'Diff_PPG', 'Diff_Shots_PG', 'Diff_Shots_Ag_PG',
                                'Diff_Shot_Acc', 'Diff_GPS', 'Diff_GAOS']
    else:
        paired_diffs = pd.DataFrame(columns=['Div', 'Season', 'Team', 'Opponent',
                                             'Diff_PPG', 'Diff_Shots_PG', 'Diff_Shots_Ag_PG',
                                             'Diff_Shot_Acc', 'Diff_GPS', 'Diff_GAOS'])
    
    results['paired_fixture_diffs'] = paired_diffs

    
    # Overall component contributions to HA
    # HA in PPG = f(HA in shots, HA in accuracy, HA in finishing, HA in defense)
    # FIXED: Drop rows with missing components instead of filling with 0
    valid = ha['Diff_PPG'].notna()
    results['ha_decomposition'] = {}  # Always include key
    if valid.sum() > 30:
        X = ha.loc[valid, ['Diff_Shots_PG', 'Diff_Shots_Ag_PG', 'Diff_Shot_Acc', 'Diff_GPS', 'Diff_GAOS']]
        y = ha.loc[valid, 'Diff_PPG']
        # Drop rows with any missing component
        complete = X.notna().all(axis=1) & y.notna()
        if complete.sum() > 30:
            X_clean = X.loc[complete]
            y_clean = y.loc[complete]
            reg = LinearRegression().fit(X_clean, y_clean)
            results['ha_decomposition'] = {
                'coefficients': dict(zip(X_clean.columns, reg.coef_)),
                'intercept': reg.intercept_,
                'r2': reg.score(X_clean, y_clean),
                'n': len(y_clean)
            }
        else:
            results['ha_decomposition'] = {'note': 'Insufficient complete cases after dropping missing components'}
    else:
        results['ha_decomposition'] = {'note': 'Insufficient valid Diff_PPG cases'}
    
    # Persistence across seasons
    ha['Season_Num'] = ha['Season'].map(SEASON_ORDER)
    ha = ha.sort_values(['Div', 'Team', 'Season_Num'])
    
    persistence = []
    for (div, team), group in ha.groupby(['Div', 'Team']):
        if len(group) >= 2:
            for m in ['Diff_PPG', 'Diff_Shots_PG', 'Diff_Shot_Acc', 'Diff_GPS', 'Diff_GAOS']:
                valid = group[m].notna()
                if valid.sum() >= 2:
                    r, p = stats.pearsonr(group.loc[valid, 'Season_Num'], group.loc[valid, m])
                    persistence.append({'Div': div, 'Team': team, 'Metric': m, 'Trend_R': r, 'Trend_P': p, 'Seasons': int(valid.sum())})
    
    results['ha_component_persistence'] = pd.DataFrame(persistence)
    
    # Aggregate by league
    league_ha = ha.groupby('Div').agg(
        Teams=('Team', 'nunique'),
        Avg_Diff_PPG=('Diff_PPG', 'mean'),
        Avg_Diff_Shots_PG=('Diff_Shots_PG', 'mean'),
        Avg_Diff_Shots_Ag_PG=('Diff_Shots_Ag_PG', 'mean'),
        Avg_Diff_Shot_Acc=('Diff_Shot_Acc', 'mean'),
        Avg_Diff_GPS=('Diff_GPS', 'mean'),
        Avg_Diff_GAOS=('Diff_GAOS', 'mean'),
    ).reset_index()
    results['league_ha_components'] = league_ha
    
    return results


def analyze_close_match_dependence(team_matches: pd.DataFrame, team_season: pd.DataFrame,
                                    min_matches: int = 20) -> Dict:
    """
    Question 3: Which teams rely unusually heavily on one-goal results?
    
    - Describe one-goal wins, one-goal losses, draws, wider-margin outcomes
    - Quantify how much observed points come from narrow victories
    - Compare early and later non-overlapping periods
    - Adjust for underlying team strength
    - Compare persistence with shot differential persistence
    """
    results = {}
    
    # Classify match outcomes by margin
    tm = team_matches.copy()
    tm['Margin'] = tm['GD'].abs()
    tm['Result_Type'] = np.select(
        [
            (tm['FT_Result'] == 'W') & (tm['Margin'] == 1),
            (tm['FT_Result'] == 'W') & (tm['Margin'] >= 2),
            (tm['FT_Result'] == 'L') & (tm['Margin'] == 1),
            (tm['FT_Result'] == 'L') & (tm['Margin'] >= 2),
            (tm['FT_Result'] == 'D'),
        ],
        ['One_Goal_Win', 'Multi_Goal_Win', 'One_Goal_Loss', 'Multi_Goal_Loss', 'Draw'],
        default='Other'
    )
    
    # Team-season outcome composition
    outcome_comp = tm.groupby(['Div', 'Team', 'Season', 'Result_Type']).agg(
        Count=('Points', 'count'),
        Points=('Points', 'sum'),
    ).reset_index()
    
    outcome_pivot = outcome_comp.pivot_table(
        index=['Div', 'Team', 'Season'], columns='Result_Type', 
        values=['Count', 'Points'], fill_value=0
    ).reset_index()
    outcome_pivot.columns = ['_'.join(col).strip('_') for col in outcome_pivot.columns.values]
    
    # Ensure all columns exist
    for col in ['Count_One_Goal_Win', 'Count_Multi_Goal_Win', 'Count_One_Goal_Loss', 'Count_Multi_Goal_Loss', 'Count_Draw',
                'Points_One_Goal_Win', 'Points_Multi_Goal_Win', 'Points_One_Goal_Loss', 'Points_Multi_Goal_Loss', 'Points_Draw']:
        if col not in outcome_pivot.columns:
            outcome_pivot[col] = 0
    
    outcome_pivot['Total_Matches'] = outcome_pivot[[c for c in outcome_pivot.columns if c.startswith('Count_')]].sum(axis=1)
    outcome_pivot['Total_Points'] = outcome_pivot[[c for c in outcome_pivot.columns if c.startswith('Points_')]].sum(axis=1)
    outcome_pivot['PPG'] = outcome_pivot['Total_Points'] / outcome_pivot['Total_Matches']
    outcome_pivot['One_Goal_Win_Pct'] = outcome_pivot['Count_One_Goal_Win'] / outcome_pivot['Total_Matches']
    outcome_pivot['Points_From_One_Goal_Wins'] = outcome_pivot['Points_One_Goal_Win'] / outcome_pivot['Total_Points']
    
    results['season_outcome_composition'] = outcome_pivot
    
    # Close-match dependence: proportion of points from one-goal wins
    valid_seasons = outcome_pivot[outcome_pivot['Total_Matches'] >= min_matches].copy()
    
    # Adjust for team strength (regress on PPG)
    valid = valid_seasons['PPG'].notna() & valid_seasons['Points_From_One_Goal_Wins'].notna()
    if valid.sum() > 50:
        slope, intercept, r, p, se = stats.linregress(valid_seasons.loc[valid, 'PPG'], valid_seasons.loc[valid, 'Points_From_One_Goal_Wins'])
        valid_seasons['Adj_Points_From_One_Goal_Wins'] = valid_seasons['Points_From_One_Goal_Wins'] - (slope * valid_seasons['PPG'] + intercept)
    else:
        valid_seasons['Adj_Points_From_One_Goal_Wins'] = valid_seasons['Points_From_One_Goal_Wins']
    
    results['close_match_dependence'] = valid_seasons
    
    # Early vs late season comparison (non-overlapping)
    tm['Match_Num'] = tm.groupby(['Div', 'Team', 'Season']).cumcount() + 1
    tm['Season_Half'] = np.where(
        tm['Match_Num'] <= tm.groupby(['Div', 'Team', 'Season'])['Match_Num'].transform('max') / 2,
        'Early', 'Late'
    )
    
    half_comp = tm.groupby(['Div', 'Team', 'Season', 'Season_Half', 'Result_Type']).agg(
        Count=('Points', 'count'),
        Points=('Points', 'sum'),
    ).reset_index()
    
    half_pivot = half_comp.pivot_table(
        index=['Div', 'Team', 'Season', 'Season_Half'], columns='Result_Type',
        values=['Count', 'Points'], fill_value=0
    ).reset_index()
    half_pivot.columns = ['_'.join(col).strip('_') for col in half_pivot.columns.values]
    
    for col in ['Count_One_Goal_Win', 'Count_Multi_Goal_Win', 'Count_One_Goal_Loss', 'Count_Multi_Goal_Loss', 'Count_Draw',
                'Points_One_Goal_Win', 'Points_Multi_Goal_Win', 'Points_One_Goal_Loss', 'Points_Multi_Goal_Loss', 'Points_Draw']:
        if col not in half_pivot.columns:
            half_pivot[col] = 0
    
    half_pivot['Total_Matches'] = half_pivot[[c for c in half_pivot.columns if c.startswith('Count_')]].sum(axis=1)
    half_pivot['Total_Points'] = half_pivot[[c for c in half_pivot.columns if c.startswith('Points_')]].sum(axis=1)
    half_pivot['PPG'] = half_pivot['Total_Points'] / half_pivot['Total_Matches']
    half_pivot['One_Goal_Win_Pct'] = half_pivot['Count_One_Goal_Win'] / half_pivot['Total_Matches']
    half_pivot['Points_From_One_Goal_Wins'] = half_pivot['Points_One_Goal_Win'] / half_pivot['Total_Points'].replace(0, np.nan)
    
    # Early vs late comparison
    early = half_pivot[half_pivot['Season_Half'] == 'Early'][['Div', 'Team', 'Season', 'PPG', 'One_Goal_Win_Pct', 'Points_From_One_Goal_Wins']].rename(
        columns={'PPG': 'PPG_Early', 'One_Goal_Win_Pct': 'OGW_Pct_Early', 'Points_From_One_Goal_Wins': 'Points_From_OGW_Early'}
    )
    late = half_pivot[half_pivot['Season_Half'] == 'Late'][['Div', 'Team', 'Season', 'PPG', 'One_Goal_Win_Pct', 'Points_From_One_Goal_Wins']].rename(
        columns={'PPG': 'PPG_Late', 'One_Goal_Win_Pct': 'OGW_Pct_Late', 'Points_From_One_Goal_Wins': 'Points_From_OGW_Late'}
    )
    
    early_late = early.merge(late, on=['Div', 'Team', 'Season'])
    early_late = early_late[(early_late['PPG_Early'].notna()) & (early_late['PPG_Late'].notna())]
    
    # Persistence of close-match dependence
    if len(early_late) > 20:
        r_pct, p_pct = stats.pearsonr(early_late['OGW_Pct_Early'], early_late['OGW_Pct_Late'])
        r_pts, p_pts = stats.pearsonr(early_late['Points_From_OGW_Early'].dropna(), early_late['Points_From_OGW_Late'].dropna())
        results['early_late_persistence'] = {
            'ogw_pct': {'r': r_pct, 'p': p_pct, 'n': len(early_late)},
            'points_from_ogw': {'r': r_pts, 'p': p_pts, 'n': early_late['Points_From_OGW_Early'].notna().sum()}
        }
    
    results['early_late_comparison'] = early_late
    
    # Compare with shot differential persistence
    shot_dom_persist = []
    for (div, team, season), group in tm.groupby(['Div', 'Team', 'Season']):
        early_g = group[group['Season_Half'] == 'Early']
        late_g = group[group['Season_Half'] == 'Late']
        if len(early_g) >= 5 and len(late_g) >= 5:
            shot_dom_persist.append({
                'Div': div, 'Team': team, 'Season': season,
                'ShotDom_Early': early_g['Shot_Dominance'].mean(),
                'ShotDom_Late': late_g['Shot_Dominance'].mean(),
            })
    
    shot_dom_df = pd.DataFrame(shot_dom_persist)
    if len(shot_dom_df) > 20:
        r_sd, p_sd = stats.pearsonr(shot_dom_df['ShotDom_Early'], shot_dom_df['ShotDom_Late'])
        results['shot_dominance_persistence'] = {'r': r_sd, 'p': p_sd, 'n': len(shot_dom_df)}
    
    results['shot_dominance_early_late'] = shot_dom_df
    
    # Team-level summary (3+ seasons)
    team_summary = valid_seasons.groupby('Team').agg(
        Seasons=('Season', 'count'),
        Avg_PPG=('PPG', 'mean'),
        Avg_One_Goal_Win_Pct=('One_Goal_Win_Pct', 'mean'),
        Avg_Points_From_One_Goal_Wins=('Points_From_One_Goal_Wins', 'mean'),
        Avg_Adj_Points_From_One_Goal_Wins=('Adj_Points_From_One_Goal_Wins', 'mean'),
    ).reset_index()
    team_summary = team_summary[team_summary['Seasons'] >= 3]
    results['team_close_match_summary'] = team_summary.sort_values('Avg_Adj_Points_From_One_Goal_Wins', ascending=False)
    
    return results


def analyze_rest_asymmetry(team_matches: pd.DataFrame, master: pd.DataFrame,
                            min_matches: int = 20) -> Dict:
    """
    Question 4: Is being less rested than the opponent associated with worse performance?
    
    - Compute days since previous observed fixture for both teams
    - Define own rest, opponent rest, rest difference
    - Handle season openers as unknown (not invented rest)
    - Separate unusually long gaps from ordinary rest
    - Examine interactions with venue and pre-match strength
    - Report density across rest combinations
    """
    results = {}
    
    # Add lagged team strength
    tm = add_lagged_team_strength(team_matches)
    
    # Add opponent rest
    tm = tm.sort_values(['Div', 'Team', 'Season', 'Date'])
    
    # Create match-level rest data from master
    master_rest = master[['Div', 'Season', 'Date', 'HomeTeam', 'AwayTeam']].copy()
    master_rest = master_rest.sort_values(['Div', 'Season', 'Date'])
    
    # For each team, compute rest days before each match
    master_rest['Home_Rest'] = master_rest.groupby(['Div', 'Season', 'HomeTeam'])['Date'].diff().dt.days
    master_rest['Away_Rest'] = master_rest.groupby(['Div', 'Season', 'AwayTeam'])['Date'].diff().dt.days
    
    # Season openers: NaN (unknown), not 7
    # Merge back to team_matches
    home_rest = master_rest[['Div', 'Season', 'Date', 'HomeTeam', 'Home_Rest']].rename(
        columns={'HomeTeam': 'Team', 'Home_Rest': 'Rest_Days_Actual'}
    )
    away_rest = master_rest[['Div', 'Season', 'Date', 'AwayTeam', 'Away_Rest']].rename(
        columns={'AwayTeam': 'Team', 'Away_Rest': 'Rest_Days_Actual'}
    )
    
    rest_data = pd.concat([home_rest, away_rest], ignore_index=True)
    tm = tm.merge(rest_data, on=['Div', 'Season', 'Date', 'Team'], how='left')
    
    # Add opponent rest
    opp_rest = tm[['Div', 'Season', 'Date', 'Opponent', 'Rest_Days_Actual']].rename(
        columns={'Opponent': 'Team', 'Rest_Days_Actual': 'Opp_Rest_Days'}
    )
    tm = tm.merge(opp_rest, on=['Div', 'Season', 'Date', 'Team'], how='left')
    
    # Rest difference (positive = more rest than opponent)
    tm['Rest_Diff'] = tm['Rest_Days_Actual'] - tm['Opp_Rest_Days']
    
    # Categorize rest
    tm['Own_Rest_Cat'] = pd.cut(tm['Rest_Days_Actual'], 
                                 bins=[-np.inf, 2, 3, 4, 5, 6, 7, 10, 14, np.inf],
                                 labels=['≤2', '3', '4', '5', '6', '7', '8-10', '11-14', '15+'])
    tm['Opp_Rest_Cat'] = pd.cut(tm['Opp_Rest_Days'],
                                 bins=[-np.inf, 2, 3, 4, 5, 6, 7, 10, 14, np.inf],
                                 labels=['≤2', '3', '4', '5', '6', '7', '8-10', '11-14', '15+'])
    tm['Rest_Diff_Cat'] = pd.cut(tm['Rest_Diff'],
                                  bins=[-np.inf, -4, -3, -2, -1, 0, 1, 2, 3, 4, np.inf],
                                  labels=['≤-4', '-3', '-2', '-1', '0', '+1', '+2', '+3', '+4', '≥+5'])
    
    results['match_rest_data'] = tm
    
    # Density of observations
    density = tm.groupby(['Own_Rest_Cat', 'Opp_Rest_Cat']).size().reset_index(name='Count')
    results['rest_density'] = density
    
    # Outcome by rest difference, controlling for strength and venue
    valid = tm['Rest_Days_Actual'].notna() & tm['Opp_Rest_Days'].notna() & tm['Points'].notna()
    tm_valid = tm[valid].copy()
    
    # Add pre-match strength
    tm_valid['Strength_Tercile'] = tm_valid.groupby(['Div', 'Season'])['PreMatch_PPG_Shrunk'].transform(
        lambda x: pd.qcut(x, 3, labels=['Weak', 'Average', 'Strong'], duplicates='drop')
    )
    
    # Adjusted response surface: Points ~ Rest_Diff + Venue + Strength + Div + Season
    # Use league-season fixed effects
    tm_valid['LeagueSeason'] = tm_valid['Div'] + '_' + tm_valid['Season']
    tm_valid['Venue_Code'] = tm_valid['Venue'].map({'Home': 1, 'Away': 0})
    
    # Linear model for points
    ls_dummies = pd.get_dummies(tm_valid['LeagueSeason'], drop_first=True)
    X = pd.concat([
        tm_valid[['Rest_Diff', 'Venue_Code', 'PreMatch_PPG_Shrunk']],
        ls_dummies
    ], axis=1)
    y = tm_valid['Points']
    reg = LinearRegression(fit_intercept=True).fit(X, y)
    tm_valid['Expected_Points'] = reg.predict(X)
    tm_valid['Residual'] = tm_valid['Points'] - tm_valid['Expected_Points']
    
    results['rest_points_model'] = {
        'rest_diff_coef': reg.coef_[0],
        'venue_coef': reg.coef_[1],
        'strength_coef': reg.coef_[2],
        'r2': reg.score(X, y),
        'n': len(y)
    }
    
    # Marginal effect of rest difference by venue
    rest_effects = []
    for venue in ['Home', 'Away']:
        for strength in ['Weak', 'Average', 'Strong']:
            subset = tm_valid[(tm_valid['Venue'] == venue) & (tm_valid['Strength_Tercile'] == strength)]
            if len(subset) < 20:
                continue
            v = subset['Rest_Diff'].notna() & subset['Residual'].notna()
            if v.sum() < 20:
                continue
            slope, intercept, r, p, se = stats.linregress(subset.loc[v, 'Rest_Diff'], subset.loc[v, 'Residual'])
            rest_effects.append({
                'Venue': venue, 'Strength': strength,
                'Slope': slope, 'R': r, 'P': p, 'N': int(v.sum())
            })
    
    results['rest_effects_by_venue_strength'] = pd.DataFrame(rest_effects)
    
    # Shot differential by rest difference
    shot_effects = []
    for venue in ['Home', 'Away']:
        for strength in ['Weak', 'Average', 'Strong']:
            subset = tm_valid[(tm_valid['Venue'] == venue) & (tm_valid['Strength_Tercile'] == strength)]
            if len(subset) < 20:
                continue
            v = subset['Rest_Diff'].notna() & subset['Shot_Dominance'].notna()
            if v.sum() < 20:
                continue
            slope, intercept, r, p, se = stats.linregress(subset.loc[v, 'Rest_Diff'], subset.loc[v, 'Shot_Dominance'])
            shot_effects.append({
                'Venue': venue, 'Strength': strength,
                'Slope': slope, 'R': r, 'P': p, 'N': int(v.sum())
            })
    
    results['rest_shot_effects'] = pd.DataFrame(shot_effects)
    
    # Supported region heatmap data (rest diff x venue x strength -> adjusted points)
    heatmap_data = []
    for rest_diff in range(-4, 5):
        for venue in ['Home', 'Away']:
            for strength in ['Weak', 'Average', 'Strong']:
                subset = tm_valid[(tm_valid['Rest_Diff'] == rest_diff) & 
                                  (tm_valid['Venue'] == venue) & 
                                  (tm_valid['Strength_Tercile'] == strength)]
                if len(subset) >= 10:
                    heatmap_data.append({
                        'Rest_Diff': rest_diff, 'Venue': venue, 'Strength': strength,
                        'Mean_Points': subset['Points'].mean(),
                        'Mean_Residual': subset['Residual'].mean(),
                        'Mean_Shot_Dom': subset['Shot_Dominance'].mean(),
                        'N': len(subset)
                    })
    
    results['rest_heatmap'] = pd.DataFrame(heatmap_data)
    
    # Exclude disrupted periods (COVID season)
    tm_no_covid = tm[tm['Season'] != '2021'].copy()
    results['match_rest_data_no_covid'] = tm_no_covid
    
    return results


def analyze_performance_indicators_lead_results(team_matches: pd.DataFrame, team_season: pd.DataFrame,
                                                 min_matches: int = 20) -> Dict:
    """
    Question 5: Does sustained improvement in shot dominance precede improved results?
    
    - Use lagged rolling shot differential and shots-on-target differential
    - Compare incremental information with lagged points and goal difference
    - Evaluate future, non-overlapping performance windows
    - Avoid overlap between indicator window and outcome window
    - Use chronological validation
    - Test survival after detrending and competition/season adjustment
    """
    results = {}
    
    tm = team_matches.sort_values(['Div', 'Team', 'Season', 'Date']).copy()
    
    # Add lagged team strength
    tm = add_lagged_team_strength(tm)
    
    # Rolling shot metrics (already in data: Shot_Dominance_L5, SOT_Dominance_L5, etc.)
    # But we need to ensure they're lagged by the outcome window
    
    # Define indicator windows and outcome windows
    # Indicator: matches 1-10 -> Outcome: matches 11-20 (non-overlapping)
    # Indicator: matches 1-15 -> Outcome: matches 16-30
    
    # Vectorized window metrics.
    # For each team-season (sorted by Date), the mean of the first K matches is
    # cumsum[K-1]/K, and the mean of matches [a, b) is (cumsum[b-1]-cumsum[a-1])/(b-a).
    # Computed with groupby masks instead of a Python groupby.apply loop.
    ind_cols = ['Shot_Dominance', 'SOT_Dominance', 'Points', 'GD', 'GF', 'GA']
    out_cols = ['Points', 'GD', 'GF', 'GA', 'Shot_Dominance', 'SOT_Dominance']
    ind_rename = {'Shot_Dominance': 'Ind_Shot_Dom', 'SOT_Dominance': 'Ind_SOT_Dom',
                  'Points': 'Ind_PPG', 'GD': 'Ind_GD_PG', 'GF': 'Ind_GF_PG', 'GA': 'Ind_GA_PG'}
    out_rename = {'Points': 'Out_PPG', 'GD': 'Out_GD_PG', 'GF': 'Out_GF_PG', 'GA': 'Out_GA_PG',
                  'Shot_Dominance': 'Out_Shot_Dom', 'SOT_Dominance': 'Out_SOT_Dom'}

    def _window_df(ind_end, out_end, half=False):
        """Indicator = first ind_end matches; Outcome = matches [ind_end, out_end).
        If half=True, split at n//2 per team-season."""
        gk = ['Div', 'Team', 'Season']
        g = tm.groupby(gk)
        cc = g.cumcount().to_numpy()
        n = g.transform('size').to_numpy()
        if half:
            valid = n >= 10
        else:
            valid = n >= out_end
        tmv = tm[valid].copy()
        ccv = cc[valid]
        nv = n[valid]
        if half:
            mid = nv // 2
            imask = ccv < mid
            omask = ccv >= mid
        else:
            imask = ccv < ind_end
            omask = (ccv >= ind_end) & (ccv < out_end)
        ind_sub = tmv[imask]
        out_sub = tmv[omask]
        ind_means = ind_sub.groupby(gk)[ind_cols].mean().rename(columns=ind_rename)
        out_means = out_sub.groupby(gk)[out_cols].mean().rename(columns=out_rename)
        n_ind = ind_sub.groupby(gk).size().rename('N_Indicator')
        n_out = out_sub.groupby(gk).size().rename('N_Outcome')
        df = ind_means.join(out_means, how='outer').join(n_ind).join(n_out)
        df = df[(df['N_Indicator'] >= 5) & (df['N_Outcome'] >= 5)]
        return df.reset_index()

    # Window 1: First 10 matches -> Next 10 matches (11-20)
    window1 = _window_df(10, 20)
    results['window_10_10'] = window1

    # Window 2: First 15 matches -> Next 15 matches (16-30)
    window2 = _window_df(15, 30)
    results['window_15_15'] = window2

    # Window 3: First half -> Second half (non-overlapping by match order)
    window3 = _window_df(0, 0, half=True)
    results['window_half_half'] = window3
    
    # Lead-lag correlations for each window
    lead_lag_results = {}
    for window_name, window_df in [('10_10', window1), ('15_15', window2), ('half_half', window3)]:
        if len(window_df) < 20:
            continue
        
        indicators = ['Ind_Shot_Dom', 'Ind_SOT_Dom', 'Ind_PPG', 'Ind_GD_PG', 'Ind_GF_PG', 'Ind_GA_PG']
        outcomes = ['Out_PPG', 'Out_GD_PG', 'Out_GF_PG', 'Out_GA_PG', 'Out_Shot_Dom', 'Out_SOT_Dom']
        
        window_corrs = {}
        for ind in indicators:
            for out in outcomes:
                valid = window_df[ind].notna() & window_df[out].notna()
                if valid.sum() > 20:
                    r, p = stats.pearsonr(window_df.loc[valid, ind], window_df.loc[valid, out])
                    window_corrs[f'{ind}_vs_{out}'] = {'r': r, 'p': p, 'n': int(valid.sum())}
        
        lead_lag_results[window_name] = window_corrs
    
    results['lead_lag_correlations'] = lead_lag_results
    
    # Incremental prediction: does shot dominance add to lagged PPG?
    for window_name, window_df in [('10_10', window1), ('15_15', window2), ('half_half', window3)]:
        if len(window_df) < 30:
            continue
            
        valid = (window_df['Ind_Shot_Dom'].notna() & window_df['Ind_PPG'].notna() & 
                 window_df['Out_PPG'].notna())
        if valid.sum() < 30:
            continue
        
        dfv = window_df[valid].copy()
        
        # Model 1: Out_PPG ~ Ind_PPG
        X1 = dfv[['Ind_PPG']].values
        y = dfv['Out_PPG'].values
        reg1 = LinearRegression().fit(X1, y)
        r2_1 = reg1.score(X1, y)
        
        # Model 2: Out_PPG ~ Ind_PPG + Ind_Shot_Dom
        X2 = dfv[['Ind_PPG', 'Ind_Shot_Dom']].values
        reg2 = LinearRegression().fit(X2, y)
        r2_2 = reg2.score(X2, y)
        
        # Model 3: Out_PPG ~ Ind_PPG + Ind_SOT_Dom
        X3 = dfv[['Ind_PPG', 'Ind_SOT_Dom']].values
        reg3 = LinearRegression().fit(X3, y)
        r2_3 = reg3.score(X3, y)
        
        results[f'incremental_{window_name}'] = {
            'ppg_only_r2': r2_1,
            'ppg_plus_shotdom_r2': r2_2,
            'ppg_plus_sotdom_r2': r2_3,
            'incremental_shotdom': r2_2 - r2_1,
            'incremental_sotdom': r2_3 - r2_1,
            'n': len(dfv)
        }
    
    # League-season adjusted (residualized)
    tm['LeagueSeason'] = tm['Div'] + '_' + tm['Season']
    
    for window_name, window_df in [('10_10', window1), ('15_15', window2), ('half_half', window3)]:
        if len(window_df) < 30:
            continue
        
        # Merge league-season
        window_df = window_df.copy()
        window_df['LeagueSeason'] = window_df['Div'] + '_' + window_df['Season']
        
        for ind in ['Ind_Shot_Dom', 'Ind_SOT_Dom', 'Ind_PPG', 'Ind_GD_PG']:
            for out in ['Out_PPG', 'Out_GD_PG']:
                valid = window_df[ind].notna() & window_df[out].notna() & window_df['LeagueSeason'].notna()
                if valid.sum() < 30:
                    continue
                dfv = window_df[valid].copy()
                
                # Residualize both on league-season
                ls_ind = dfv.groupby('LeagueSeason')[ind].transform('mean')
                ls_out = dfv.groupby('LeagueSeason')[out].transform('mean')
                ind_resid = dfv[ind] - ls_ind
                out_resid = dfv[out] - ls_out
                
                v2 = ind_resid.notna() & out_resid.notna()
                if v2.sum() > 20:
                    r, p = stats.pearsonr(ind_resid[v2], out_resid[v2])
                    if f'adjusted_{window_name}' not in results:
                        results[f'adjusted_{window_name}'] = {}
                    results[f'adjusted_{window_name}'][f'{ind}_vs_{out}'] = {'r': r, 'p': p, 'n': int(v2.sum())}
    
    # Sensitivity to horizon: test multiple outcome windows
    horizons = [(5, 10), (10, 10), (10, 15), (15, 15), (15, 20)]
    horizon_results = {}
    
    for ind_end, out_end in horizons:
        w = _window_df(ind_end, out_end)
        if len(w) > 0:
            w = w.dropna()
            if len(w) > 20:
                r, p = stats.pearsonr(w['Ind_Shot_Dom'], w['Out_PPG'])
                horizon_results[f'ind{ind_end}_out{out_end}'] = {'r': r, 'p': p, 'n': len(w)}
    
    results['horizon_sensitivity'] = horizon_results

    return results


def analyze_adaptation_after_promotion(team_matches: pd.DataFrame, team_season: pd.DataFrame,
                                        master: pd.DataFrame) -> Dict:
    """
    Question 7: Do promoted teams improve their relative performance during their first observed season?
    
    - Restrict promotion identification to reliable adjacent-tier transitions
    - Clearly separate newly promoted teams from teams merely entering dataset coverage
    - Align first-season trajectories by match number
    - Adjust for opponent strength and venue
    - Compare shot creation, shot suppression, and results
    - Examine whether baseline strength explains different trajectories
    """
    results = {}
    
    # Build reliable promotion/relegation transitions
    ts = build_promotion_relegation_transitions(team_season)
    
    # Identify promoted teams: consecutive season, tier decreased (e.g., E1->E0)
    promoted = ts[(ts['Transition_Type'] == 'Promotion') & ts['Consecutive']].copy()
    relegated = ts[(ts['Transition_Type'] == 'Relegation') & ts['Consecutive']].copy()
    
    results['promoted_teams'] = promoted[['Team', 'Prev_Div', 'Div', 'Prev_Season_Num', 'Season_Num', 'Prev_PPG', 'PPG']]
    results['relegated_teams'] = relegated[['Team', 'Prev_Div', 'Div', 'Prev_Season_Num', 'Season_Num', 'Prev_PPG', 'PPG']]
    
    # Get match data for promoted teams in their first season in new division
    promo_matches = []
    for _, row in promoted.iterrows():
        team = row['Team']
        div = row['Div']
        season = row['Season_Num']
        season_code = [k for k, v in SEASON_ORDER.items() if v == season][0]
        
        matches = team_matches[
            (team_matches['Team'] == team) & 
            (team_matches['Div'] == div) & 
            (team_matches['Season'] == season_code)
        ].copy()
        
        if len(matches) >= 15:
            matches = matches.sort_values('Date')
            matches['Match_Num'] = range(1, len(matches) + 1)
            matches['Promoted_From'] = row['Prev_Div']
            matches['Prev_PPG'] = row['Prev_PPG']
            promo_matches.append(matches)
    
    if promo_matches:
        all_promo = pd.concat(promo_matches, ignore_index=True)
        results['promoted_match_data'] = all_promo
        
        # Align by match number and compute trajectory
        trajectory = all_promo.groupby('Match_Num').agg(
            N=('GF', 'count'),
            PPG=('Points', 'mean'),
            GF_PG=('GF', 'mean'),
            GA_PG=('GA', 'mean'),
            Shots_PG=('Shots', 'mean'),
            Shots_Ag_PG=('Shots_Against', 'mean'),
            Shot_Dom=('Shot_Dominance', 'mean'),
            SOT_Dom=('SOT_Dominance', 'mean'),
            GPS=('Goals_Per_Shot', 'mean'),
            GAOS=('GA_Per_Opp_Shot', 'mean'),
        ).reset_index()
        
        # Rolling 5-match for smoothing
        for col in ['PPG', 'GF_PG', 'GA_PG', 'Shots_PG', 'Shots_Ag_PG', 'Shot_Dom', 'SOT_Dom', 'GPS', 'GAOS']:
            trajectory[f'{col}_Roll5'] = trajectory[col].rolling(5, min_periods=1, center=True).mean()
        
        results['promoted_trajectory'] = trajectory
        
        # Compare with non-promoted teams in same division-season
        comparison_trajectories = {}
        for div in all_promo['Div'].unique():
            for season in all_promo['Season'].unique():
                # Other teams in same division-season
                other_teams = team_matches[
                    (team_matches['Div'] == div) & 
                    (team_matches['Season'] == season) & 
                    (~team_matches['Team'].isin(all_promo['Team'].unique()))
                ].copy()
                
                if len(other_teams) > 100:
                    other_traj = other_teams.groupby('Match_Num').agg(
                        PPG=('Points', 'mean'),
                        GF_PG=('GF', 'mean'),
                        GA_PG=('GA', 'mean'),
                        Shots_PG=('Shots', 'mean'),
                        Shots_Ag_PG=('Shots_Against', 'mean'),
                        Shot_Dom=('Shot_Dominance', 'mean'),
                        SOT_Dom=('SOT_Dominance', 'mean'),
                    ).reset_index()
                    comparison_trajectories[f'{div}_{season}'] = other_traj
        
        results['comparison_trajectories'] = comparison_trajectories
        
        # Individual team adaptation curves
        team_curves = []
        for team in all_promo['Team'].unique():
            tdata = all_promo[all_promo['Team'] == team].sort_values('Match_Num')
            if len(tdata) >= 15:
                # Compute cumulative PPG
                tdata = tdata.copy()
                tdata['Cum_Points'] = tdata['Points'].cumsum()
                tdata['Cum_PPG'] = tdata['Cum_Points'] / tdata['Match_Num']
                # Fit trend on cumulative PPG
                v = tdata['Cum_PPG'].notna()
                if v.sum() >= 10:
                    slope, intercept, r, p, se = stats.linregress(tdata.loc[v, 'Match_Num'], tdata.loc[v, 'Cum_PPG'])
                    # Get Prev_PPG from transition_table
                    trans_table = results.get('promotion_transition_table', pd.DataFrame())
                    if len(trans_table) > 0:
                        prev_ppg = trans_table[trans_table['Team'] == team]['Prev_PPG'].values
                        prev_ppg_val = prev_ppg[0] if len(prev_ppg) > 0 else np.nan
                    else:
                        prev_ppg_val = np.nan
                    team_curves.append({
                        'Team': team, 'Div': tdata['Div'].iloc[0], 'Season': tdata['Season'].iloc[0],
                        'Prev_PPG': prev_ppg_val,
                        'PPG_Slope': slope, 'PPG_R': r, 'PPG_P': p, 'Matches': int(v.sum()),
                        'Start_PPG': tdata.head(5)['Points'].mean(),
                        'End_PPG': tdata.tail(5)['Points'].mean(),
                    })
        
        results['promoted_team_curves'] = pd.DataFrame(team_curves)
        
        # Check if baseline strength (Prev_PPG) explains trajectory
        if len(results['promoted_team_curves']) > 5:
            curves = results['promoted_team_curves']
            r, p = stats.pearsonr(curves['Prev_PPG'], curves['PPG_Slope'])
            results['baseline_strength_effect'] = {'r': r, 'p': p, 'n': len(curves)}
    
    # Relegated teams for comparison
    releg_matches = []
    for _, row in relegated.iterrows():
        team = row['Team']
        div = row['Div']
        season = row['Season_Num']
        season_code = [k for k, v in SEASON_ORDER.items() if v == season][0]
        
        matches = team_matches[
            (team_matches['Team'] == team) & 
            (team_matches['Div'] == div) & 
            (team_matches['Season'] == season_code)
        ].copy()
        
        if len(matches) >= 15:
            matches = matches.sort_values('Date')
            matches['Match_Num'] = range(1, len(matches) + 1)
            matches['Relegated_From'] = row['Prev_Div']
            matches['Prev_PPG'] = row['Prev_PPG']
            releg_matches.append(matches)
    
    if releg_matches:
        all_releg = pd.concat(releg_matches, ignore_index=True)
        results['relegated_match_data'] = all_releg
        
        releg_traj = all_releg.groupby('Match_Num').agg(
            N=('GF', 'count'),
            PPG=('Points', 'mean'),
            GF_PG=('GF', 'mean'),
            GA_PG=('GA', 'mean'),
            Shots_PG=('Shots', 'mean'),
            Shots_Ag_PG=('Shots_Against', 'mean'),
        ).reset_index()
        results['relegated_trajectory'] = releg_traj
    
    # Transition table
    transition_table = promoted[['Team', 'Prev_Div', 'Div', 'Prev_Season_Num', 'Season_Num', 
                                 'Prev_PPG', 'PPG', 'Prev_Shots_PG', 'Shots_PG',
                                 'Prev_Shots_Against_PG', 'Shots_Against_PG']].copy()
    transition_table['PPG_Change'] = transition_table['PPG'] - transition_table['Prev_PPG']
    transition_table['Shots_PG_Change'] = transition_table['Shots_PG'] - transition_table['Prev_Shots_PG']
    transition_table['Shots_Ag_PG_Change'] = transition_table['Shots_Against_PG'] - transition_table['Prev_Shots_Against_PG']
    results['promotion_transition_table'] = transition_table
    
    return results


# Helper function for chronological team-season ordering
def ensure_chronological_order(team_season: pd.DataFrame) -> pd.DataFrame:
    """Ensure team-season data is sorted chronologically per team."""
    ts = team_season.copy()
    ts['Season_Num'] = ts['Season'].map(SEASON_ORDER)
    ts = ts.sort_values(['Team', 'Season_Num'])
    return ts


# Metric-specific exposure tracking
def compute_metric_exposure(team_matches: pd.DataFrame, metric: str, min_periods: int = 1) -> pd.DataFrame:
    """
    Compute valid match counts for a specific metric per team-season.
    Preserves all-missing as NaN.
    """
    tm = team_matches.sort_values(['Div', 'Team', 'Season', 'Date'])
    valid = tm[metric].notna()
    exposure = tm[valid].groupby(['Div', 'Team', 'Season']).size().reset_index(name=f'{metric}_Valid_Matches')
    return exposure