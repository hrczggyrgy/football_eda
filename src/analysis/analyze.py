"""
Statistical analysis module.
Core statistics: home advantage, efficiency persistence, shot decomposition,
half-time/full-time dynamics, consistency, and disciplinary patterns.

Migrated to Polars: Polars is the primary analytical data structure. NumPy
and SciPy/scikit-learn are used only where they provide the appropriate
statistical functionality (pearsonr, linregress, binomtest, LinearRegression).
No pandas.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from scipy import stats
from scipy.stats import binomtest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

TEAM_SEASON_GROUP = ['Div', 'Team', 'Season']
SEASON_ORDER = {
    '1920': 0, '2021': 1, '2122': 2, '2223': 3,
    '2324': 4, '2425': 5, '2526': 6, '2627': 7,
}


def _pearsonr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, int]:
    """Pearson correlation returning (r, p, n) on valid pairs."""
    m = (~np.isnan(x)) & (~np.isnan(y))
    n = int(m.sum())
    if n < 2:
        return float('nan'), float('nan'), n
    r, p = stats.pearsonr(x[m], y[m])
    return float(r), float(p), n


def _df_to_np(df: pl.DataFrame, col: str) -> np.ndarray:
    return df.select(pl.col(col).cast(pl.Float64)).to_series().to_numpy()


def analyze_home_advantage(master: pl.DataFrame) -> Dict:
    """Analyze home advantage by league, season, and team."""
    results: Dict = {}

    # By league and season
    ha_ls = master.group_by(['Div', 'Season']).agg([
        ((pl.col('FTR') == 'H').mean()).alias('Home_Win_Pct'),
        ((pl.col('FTR') == 'D').mean()).alias('Draw_Pct'),
        ((pl.col('FTR') == 'A').mean()).alias('Away_Win_Pct'),
        pl.col('FTHG').mean().alias('Home_Goals'),
        pl.col('FTAG').mean().alias('Away_Goals'),
        (pl.col('FTHG').mean() - pl.col('FTAG').mean()).alias('Goal_Diff'),
        pl.len().alias('Matches'),
    ]).sort(['Div', 'Season'])
    results['by_league_season'] = ha_ls

    # By league (averaged across league-seasons)
    ha_league = ha_ls.group_by('Div').agg([
        pl.col('Home_Win_Pct').mean().round(3),
        pl.col('Draw_Pct').mean().round(3),
        pl.col('Away_Win_Pct').mean().round(3),
        pl.col('Goal_Diff').mean().round(3),
    ]).sort('Div')
    results['by_league'] = ha_league

    # By season (averaged across leagues)
    ha_season = ha_ls.group_by('Season').agg([
        pl.col('Home_Win_Pct').mean().round(3),
        pl.col('Draw_Pct').mean().round(3),
        pl.col('Away_Win_Pct').mean().round(3),
        pl.col('Goal_Diff').mean().round(3),
    ]).sort('Season')
    results['by_season'] = ha_season

    # Statistical significance per league (symmetric null: p=0.5 decisive)
    sig_results: Dict = {}
    for league in master.select(pl.col('Div').unique()).to_series().to_list():
        lg_data = master.filter(pl.col('Div') == league)
        home_wins = int((lg_data.select(pl.col('FTR') == 'H').to_series().sum()))
        away_wins = int((lg_data.select(pl.col('FTR') == 'A').to_series().sum()))
        draws = int((lg_data.select(pl.col('FTR') == 'D').to_series().sum()))
        total = lg_data.height
        decisive = total - draws
        res = binomtest(home_wins, n=decisive, p=0.5, alternative='greater')
        sig_results[league] = {
            'home_win_pct': home_wins / total,
            'away_win_pct': away_wins / total,
            'home_win_pct_decisive': (home_wins / decisive) if decisive > 0 else float('nan'),
            'away_win_pct_decisive': (away_wins / decisive) if decisive > 0 else float('nan'),
            'draw_pct': draws / total,
            'p_value': float(res.pvalue),
            'significant': bool(res.pvalue < 0.05),
            'n_decisive': int(decisive),
        }
    results['significance'] = sig_results

    # COVID season comparison
    covid = '2021'
    ha_covid = master.filter(pl.col('Season') == covid).group_by('Div').agg([
        ((pl.col('FTR') == 'H').mean()).alias('Home_Win_Pct'),
        ((pl.col('FTR') == 'A').mean()).alias('Away_Win_Pct'),
        (pl.col('FTHG').mean() - pl.col('FTAG').mean()).alias('Goal_Diff'),
    ]).sort('Div')
    ha_other = master.filter(pl.col('Season') != covid).group_by('Div').agg([
        ((pl.col('FTR') == 'H').mean()).alias('Home_Win_Pct'),
        ((pl.col('FTR') == 'A').mean()).alias('Away_Win_Pct'),
        (pl.col('FTHG').mean() - pl.col('FTAG').mean()).alias('Goal_Diff'),
    ]).sort('Div')
    # Rename columns to avoid clash on concat
    ha_covid = ha_covid.rename({c: f'{c}_covid' for c in ha_covid.columns if c != 'Div'})
    ha_other = ha_other.rename({c: f'{c}_normal' for c in ha_other.columns if c != 'Div'})
    results['covid_vs_normal'] = ha_covid.join(ha_other, on='Div', how='full')

    return results


def analyze_efficiency_persistence(team_matches: pl.DataFrame,
                                   min_matches: int = 20) -> Dict:
    """Analyze persistence of efficiency metrics across half-seasons and seasons."""
    results: Dict = {}

    # Team-season aggregates
    team_season = team_matches.group_by(TEAM_SEASON_GROUP).agg([
        pl.len().alias('Matches'),
        pl.col('GF').sum().alias('GF'),
        pl.col('GA').sum().alias('GA'),
        pl.col('Shots').sum().alias('Shots'),
        pl.col('Shots_OT').sum().alias('Shots_OT'),
        pl.col('Shots_Against').sum().alias('Shots_Against'),
        pl.col('Shots_OT_Against').sum().alias('Shots_OT_Against'),
        pl.col('Points').sum().alias('Points'),
    ])
    team_season = team_season.filter(pl.col('Matches') >= min_matches)

    def _safe_ratio(num: str, den: str, name: str) -> pl.Expr:
        return pl.when(pl.col(den) > 0).then(
            pl.col(num) / pl.col(den)).otherwise(None).alias(name)

    team_season = team_season.with_columns([
        _safe_ratio('GF', 'Shots', 'GPS'),
        _safe_ratio('GF', 'Shots_OT', 'GPSOT'),
        _safe_ratio('GA', 'Shots_Against', 'GAOS'),
        (pl.col('Points') / pl.col('Matches')).alias('PPG'),
    ])
    results['team_season'] = team_season

    # Correlation with PPG
    corr_results: Dict = {}
    for metric in ['GPS', 'GPSOT', 'GAOS', 'Shot_Accuracy']:
        if metric not in team_season.columns:
            if metric == 'Shot_Accuracy' and {'Shots_OT', 'Shots'} <= set(team_season.columns):
                team_season = team_season.with_columns(
                    _safe_ratio('Shots_OT', 'Shots', 'Shot_Accuracy'))
        valid = team_season.select(pl.col(metric).is_not_null()).to_series()
        if int(valid.sum()) > 10:
            x = _df_to_np(team_season.filter(valid), metric)
            y = _df_to_np(team_season.filter(valid), 'PPG')
            r, p, n = _pearsonr(x, y)
            corr_results[metric] = {'r': r, 'p': p, 'n': n}
    results['ppg_correlations'] = corr_results

    # Half-season persistence (vectorized cumulative sums, no groupby.apply loop).
    tm_sub = team_matches.with_columns(
        pl.len().over(TEAM_SEASON_GROUP).alias('season_size')
    ).filter(pl.col('season_size') >= min_matches)
    tm_sub = tm_sub.sort(['Div', 'Team', 'Season', 'Date'])
    grp = TEAM_SEASON_GROUP
    cc = tm_sub.select(pl.cum_count('Date').over(grp)).to_series().to_numpy()
    n = tm_sub.select(pl.len().over(grp)).to_series().to_numpy()
    mid = n // 2
    first_mask = cc < mid
    first = tm_sub.filter(pl.Series(first_mask))
    second = tm_sub.filter(pl.Series(~first_mask))

    sum_cols = ['Shots', 'Shots_OT', 'GF', 'GA', 'Shots_Against', 'Shots_OT_Against']
    first_sums = first.group_by(grp).agg([pl.col(c).sum().alias(f'{c}_First') for c in sum_cols])
    second_sums = second.group_by(grp).agg([pl.col(c).sum().alias(f'{c}_Second') for c in sum_cols])
    first_matches = first.group_by(grp).agg(pl.len().alias('Matches_First'))
    second_matches = second.group_by(grp).agg(pl.len().alias('Matches_Second'))

    half_eff = first_sums.join(second_sums, on=grp, how='full', coalesce=True).join(
        first_matches, on=grp, how='full', coalesce=True).join(second_matches, on=grp, how='full', coalesce=True)

    def _ratio0_expr(num: str, den: str, name: str) -> pl.Expr:
        return pl.when(pl.col(den) > 0).then(pl.col(num) / pl.col(den)).otherwise(None).alias(name)

    ratio_cols = [
        ('GF_First', 'Shots_First', 'GPS_First'),
        ('GF_First', 'Shots_OT_First', 'GPSOT_First'),
        ('GA_First', 'Shots_Against_First', 'GAOS_First'),
        ('GA_First', 'Shots_OT_Against_First', 'GAOSOT_First'),
        ('GF_Second', 'Shots_Second', 'GPS_Second'),
        ('GF_Second', 'Shots_OT_Second', 'GPSOT_Second'),
        ('GA_Second', 'Shots_Against_Second', 'GAOS_Second'),
        ('GA_Second', 'Shots_OT_Against_Second', 'GAOSOT_Second'),
    ]
    half_eff = half_eff.with_columns([_ratio0_expr(n_, d, name) for n_, d, name in ratio_cols])
    results['half_season'] = half_eff

    half_persistence: Dict = {}
    for metric in ['GPS', 'GPSOT', 'GAOS', 'GAOSOT']:
        f_col = f'{metric}_First'
        s_col = f'{metric}_Second'
        valid = half_eff.select(pl.col(f_col).is_not_null() & pl.col(s_col).is_not_null()).to_series()
        if int(valid.sum()) > 10:
            x = _df_to_np(half_eff.filter(valid), f_col)
            y = _df_to_np(half_eff.filter(valid), s_col)
            r, p, n = _pearsonr(x, y)
            half_persistence[metric] = {'r': r, 'p': p, 'n': n}
    results['half_season_persistence'] = half_persistence

    # Season-to-season persistence (consecutive seasons, same division)
    season_order = {'1920': 0, '2021': 1, '2122': 2, '2223': 3,
                    '2324': 4, '2425': 5, '2526': 6}
    mapping = pl.DataFrame({'Season': list(season_order.keys()),
                            'Season_Num': list(season_order.values())})
    team_season = team_season.join(mapping, on='Season', how='left').sort(
        ['Div', 'Team', 'Season_Num'])

    for metric in ['GPS', 'GPSOT', 'GAOS', 'PPG']:
        team_season = team_season.with_columns(
            pl.col(metric).shift(-1).over(['Div', 'Team']).alias(f'Next_{metric}')
        )
    team_season = team_season.with_columns([
        pl.col('Season_Num').shift(-1).over(['Div', 'Team']).alias('Next_Season_Num'),
        pl.col('Div').shift(-1).over(['Div', 'Team']).alias('Next_Div'),
    ])

    valid_pairs = (
        pl.col('Next_Season_Num').is_not_null() &
        (pl.col('Next_Season_Num') - pl.col('Season_Num') == 1) &
        (pl.col('Next_Div') == pl.col('Div'))
    )
    consecutive = team_season.filter(valid_pairs)

    # Transition summary on full table
    team_season = team_season.with_columns([
        pl.col('Season_Num').shift(1).over(['Div', 'Team']).alias('Prev_Season_Num'),
        pl.col('Div').shift(1).over(['Div', 'Team']).alias('Prev_Div'),
    ])
    transition_summary = {
        'total_pairs': int(team_season.select(pl.col('Prev_Season_Num').is_not_null().sum()).to_series()[0]),
        'consecutive_same_div': int(consecutive.height),
        'div_changed': int(team_season.select((pl.col('Div') != pl.col('Prev_Div')).sum()).to_series()[0]),
        'gap_1_season': int(team_season.select((pl.col('Season_Num') - pl.col('Prev_Season_Num') == 1).sum()).to_series()[0]),
        'gap_2plus_seasons': int(team_season.select((pl.col('Season_Num') - pl.col('Prev_Season_Num') > 1).sum()).to_series()[0]),
    }
    results['season_transitions'] = transition_summary

    season_persistence: Dict = {}
    for metric in ['GPS', 'GPSOT', 'GAOS']:
        v = consecutive.select(pl.col(metric).is_not_null() & pl.col(f'Next_{metric}').is_not_null()).to_series()
        if int(v.sum()) > 10:
            x = _df_to_np(consecutive.filter(v), metric)
            y = _df_to_np(consecutive.filter(v), f'Next_{metric}')
            r, p, n = _pearsonr(x, y)
            season_persistence[metric] = {'r': r, 'p': p, 'n': n}

    for metric in ['GPS', 'GPSOT', 'GAOS', 'PPG']:
        v = consecutive.select(pl.col(metric).is_not_null() & pl.col('Next_PPG').is_not_null()).to_series()
        if int(v.sum()) > 10:
            x = _df_to_np(consecutive.filter(v), metric)
            y = _df_to_np(consecutive.filter(v), 'Next_PPG')
            r, p, n = _pearsonr(x, y)
            season_persistence[f'{metric}_to_NextPPG'] = {'r': r, 'p': p, 'n': n}

    results['season_to_season_persistence'] = season_persistence

    return results


def analyze_shot_decomposition(team_season: pl.DataFrame) -> Dict:
    """Decompose goals into volume, accuracy, finishing components."""
    results: Dict = {}

    # Multiple regression: PPG ~ Shots_PG + Shot_Accuracy + Goals_Per_Shot_OT
    x_cols = ['Shots_PG', 'Shot_Accuracy', 'Goals_Per_Shot_OT']
    if all(c in team_season.columns for c in x_cols) and 'PPG' in team_season.columns:
        Xdf = team_season.select(x_cols + ['PPG']).drop_nulls()
        if Xdf.height > 10:
            X = Xdf.select(x_cols).to_numpy()
            y = Xdf.select('PPG').to_numpy().ravel()
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            reg = LinearRegression().fit(X_scaled, y)
            results['multiple_regression'] = {
                'r2': float(reg.score(X_scaled, y)),
                'coefficients': {c: float(v) for c, v in zip(x_cols, reg.coef_)},
                'intercept': float(reg.intercept_),
            }

    # Log-linear decomposition
    ts = team_season.with_columns([
        (pl.col('GF') + 1).log().alias('Log_GF'),
        (pl.col('Shots') + 1).log().alias('Log_Shots'),
        pl.col('Shot_Accuracy').clip(0.01).log().alias('Log_Acc'),
        pl.col('Goals_Per_Shot_OT').clip(0.01).log().alias('Log_GPSOT'),
    ])
    xl_cols = ['Log_Shots', 'Log_Acc', 'Log_GPSOT']
    if all(c in ts.columns for c in xl_cols) and 'Log_GF' in ts.columns:
        Xldf = ts.select(xl_cols + ['Log_GF']).drop_nulls()
        if Xldf.height > 10:
            X = Xldf.select(xl_cols).to_numpy()
            y = Xldf.select('Log_GF').to_numpy().ravel()
            reg_log = LinearRegression().fit(X, y)
            results['log_linear'] = {
                'r2': float(reg_log.score(X, y)),
                'coefficients': {c: float(v) for c, v in zip(xl_cols, reg_log.coef_)},
                'intercept': float(reg_log.intercept_),
            }

    # Correlation matrix
    corr_vars = ['Shots_PG', 'Shot_Accuracy', 'Goals_Per_Shot_OT', 'PPG']
    if all(c in team_season.columns for c in corr_vars):
        cm = team_season.select(corr_vars).to_numpy()
        corr_mat = np.corrcoef(cm, rowvar=False)
        results['correlation_matrix'] = {
            (i, j): float(round(corr_mat[i, j], 3))
            for i in range(len(corr_vars)) for j in range(len(corr_vars))
        }

    # Attack-Defense profiles (quartile-style terciles via qcut equivalent)
    ts2 = team_season.clone()
    def _qcut3(expr: pl.Expr, labels: List[str]) -> pl.Expr:
        q1 = expr.quantile(1/3)
        q2 = expr.quantile(2/3)
        return (pl.when(expr <= q1).then(pl.lit(labels[0]))
                .when(expr <= q2).then(pl.lit(labels[1]))
                .otherwise(pl.lit(labels[2])))

    ts2 = ts2.with_columns([
        _qcut3(pl.col('Shots_PG'), ['Low', 'Medium', 'High']).alias('Attack_Profile'),
        _qcut3(pl.col('Goals_Per_Shot_OT'), ['Poor', 'Average', 'Clinical']).alias('Finishing_Profile'),
    ])
    pivot_attack = ts2.group_by(['Attack_Profile', 'Finishing_Profile']).agg(
        pl.col('PPG').mean().round(2)
    ).sort(['Attack_Profile', 'Finishing_Profile'])
    pivot_attack = pivot_attack.pivot(on='Finishing_Profile', index='Attack_Profile', values='PPG')
    results['attack_profile_ppg'] = pivot_attack

    ts2 = ts2.with_columns([
        (pl.col('Shots_Against') / pl.col('Matches')).alias('Opp_Shots_PG'),
        (pl.col('GA') / pl.col('Shots_Against')).alias('Defensive_Resilience'),
    ])
    ts2 = ts2.with_columns([
        _qcut3(pl.col('Opp_Shots_PG'), ['Low', 'Med', 'High']).alias('Def_Volume'),
        _qcut3(pl.col('Defensive_Resilience'), ['Resilient', 'Average', 'Vulnerable']).alias('Def_Resilience'),
    ])
    pivot_def = ts2.group_by(['Def_Volume', 'Def_Resilience']).agg(
        pl.col('PPG').mean().round(2)
    ).sort(['Def_Volume', 'Def_Resilience'])
    pivot_def = pivot_def.pivot(on='Def_Resilience', index='Def_Volume', values='PPG')
    results['defense_profile_ppg'] = pivot_def

    return results


def analyze_ht_ft_dynamics(master: pl.DataFrame) -> Dict:
    """Analyze half-time to full-time transitions and second-half patterns."""
    results: Dict = {}

    # Transition matrix (normalized)
    ct = master.group_by(['HTR', 'FTR']).agg(pl.len().alias('n')).sort(['HTR', 'FTR'])
    row_totals = ct.group_by('HTR').agg(pl.col('n').sum().alias('row_total'))
    ct = ct.join(row_totals, on='HTR', how='left').with_columns(
        (pl.col('n') / pl.col('row_total')).round(3).alias('prob')
    )
    transitions_norm = ct.pivot(on='FTR', index='HTR', values='prob')
    results['transitions_norm'] = transitions_norm

    trans_counts = master.group_by(['HTR', 'FTR']).agg(pl.len().alias('n')).sort(['HTR', 'FTR'])
    trans_counts = trans_counts.pivot(on='FTR', index='HTR', values='n')
    results['transitions_counts'] = trans_counts

    # Comeback analysis
    ht_trailing_ft_not_lose = master.filter(
        ((pl.col('HTR') == 'H') & (pl.col('FTR').is_in(['A', 'D']))) |
        ((pl.col('HTR') == 'A') & (pl.col('FTR').is_in(['H', 'D'])))
    )
    results['comeback_matches'] = int(ht_trailing_ft_not_lose.height)
    results['comeback_rate'] = float(ht_trailing_ft_not_lose.height / master.height)

    # Failed to hold lead
    ht_leading_ft_not_win = master.filter(
        ((pl.col('HTR') == 'H') & (pl.col('FTR') != 'H')) |
        ((pl.col('HTR') == 'A') & (pl.col('FTR') != 'A'))
    )
    results['failed_hold_lead_matches'] = int(ht_leading_ft_not_win.height)
    results['failed_hold_lead_rate'] = float(ht_leading_ft_not_win.height / master.height)

    # Second-half goals
    master = master.with_columns([
        (pl.col('FTHG') - pl.col('HTHG')).alias('SH_Goals_Home'),
        (pl.col('FTAG') - pl.col('HTAG')).alias('SH_Goals_Away'),
    ])
    master = master.with_columns([
        (pl.col('SH_Goals_Home') + pl.col('SH_Goals_Away')).alias('SH_Total_Goals'),
        (pl.col('HTHG') + pl.col('HTAG')).alias('FH_Total_Goals'),
    ])

    results['avg_fh_goals'] = float(master.select(pl.col('FH_Total_Goals').mean()).to_series()[0])
    results['avg_sh_goals'] = float(master.select(pl.col('SH_Total_Goals').mean()).to_series()[0])
    results['sh_goal_share'] = float(
        master.select(pl.col('SH_Total_Goals').sum()).to_series()[0] /
        (master.select(pl.col('FH_Total_Goals').sum()).to_series()[0] +
         master.select(pl.col('SH_Total_Goals').sum()).to_series()[0])
    )

    # By league
    sh_by_league = master.group_by('Div').agg([
        pl.col('FH_Total_Goals').sum().alias('FH_Goals'),
        pl.col('SH_Total_Goals').sum().alias('SH_Goals'),
    ]).with_columns(
        (pl.col('SH_Goals') / (pl.col('FH_Goals') + pl.col('SH_Goals'))).round(4).alias('SH_Pct')
    ).sort('Div')
    results['sh_by_league'] = sh_by_league

    # SH goals by HT result
    sh_by_ht = master.group_by('HTR').agg(pl.col('SH_Total_Goals').mean().round(4)).sort('HTR')
    results['sh_by_ht_result'] = sh_by_ht

    # Comeback rates by league
    comeback_data = []
    for league in master.select(pl.col('Div').unique()).to_series().to_list():
        lg = master.filter(pl.col('Div') == league)
        trailing_series = ((lg.select(pl.col('HTR') == 'H').to_series()) &
                           (lg.select(pl.col('FTR') == 'A').to_series())) | \
                          ((lg.select(pl.col('HTR') == 'A').to_series()) &
                           (lg.select(pl.col('FTR') == 'H').to_series()))
        drawing_series = ((lg.select(pl.col('HTR') == 'H').to_series()) &
                          (lg.select(pl.col('FTR') == 'D').to_series())) | \
                         ((lg.select(pl.col('HTR') == 'A').to_series()) &
                          (lg.select(pl.col('FTR') == 'D').to_series()))
        total_trailing = int((lg.select(pl.col('HTR') != 'D').to_series().sum()))
        if total_trailing > 0:
            comeback_data.append({
                'League': league,
                'Trailing_Win_Pct': float(trailing_series.sum() / total_trailing),
                'Trailing_Draw_Pct': float(drawing_series.sum() / total_trailing),
                'Trailing_Avoid_Loss_Pct': float((trailing_series.sum() + drawing_series.sum()) / total_trailing),
            })
    results['comeback_by_league'] = pl.DataFrame(comeback_data)

    # Team-level comeback ability
    home_side = master.with_columns([
        pl.col('HomeTeam').alias('Team'),
        (pl.col('HTHG') - pl.col('HTAG')).alias('HT_GD'),
        (pl.col('FTHG') - pl.col('FTAG')).alias('FT_GD'),
    ])
    away_side = master.with_columns([
        pl.col('AwayTeam').alias('Team'),
        (pl.col('HTAG') - pl.col('HTHG')).alias('HT_GD'),
        (pl.col('FTAG') - pl.col('FTHG')).alias('FT_GD'),
    ])
    team_comebacks = pl.concat([home_side.select(['Div', 'Season', 'Team', 'HT_GD', 'FT_GD']),
                                away_side.select(['Div', 'Season', 'Team', 'HT_GD', 'FT_GD'])])
    trailing = team_comebacks.filter(pl.col('HT_GD') < 0).with_columns([
        (pl.col('FT_GD') > 0).cast(pl.Int8).alias('Recovered'),
        (pl.col('FT_GD') >= 0).cast(pl.Int8).alias('Avoided_Loss'),
    ])
    team_comeback_stats = trailing.group_by('Team').agg([
        pl.len().alias('Trailing_Count'),
        pl.col('Recovered').sum().alias('Recoveries'),
        pl.col('Avoided_Loss').sum().alias('Avoided_Losses'),
    ]).filter(pl.col('Trailing_Count') >= 10).with_columns([
        (pl.col('Recoveries') / pl.col('Trailing_Count')).round(4).alias('Recovery_Rate'),
        (pl.col('Avoided_Losses') / pl.col('Trailing_Count')).round(4).alias('Avoid_Loss_Rate'),
    ]).sort('Recovery_Rate', descending=True)
    results['team_comeback_stats'] = team_comeback_stats

    return results


def analyze_consistency(team_matches: pl.DataFrame, min_matches: int = 20) -> Dict:
    """Analyze team performance consistency."""
    results: Dict = {}

    consistency = team_matches.group_by(TEAM_SEASON_GROUP).agg([
        pl.col('Points').std().alias('Points_Std'),
        pl.col('GF').std().alias('GF_Std'),
        pl.col('GA').std().alias('GA_Std'),
        pl.col('Shots').std().alias('Shots_Std'),
        pl.col('GD').std().alias('GD_Std'),
        pl.col('Points').mean().alias('PPG'),
        pl.col('GF').mean().alias('GF_PG'),
        pl.col('GA').mean().alias('GA_PG'),
        pl.len().alias('Matches'),
    ]).filter(pl.col('Matches') >= min_matches).with_columns([
        (pl.col('Points_Std') / pl.col('PPG')).round(4).alias('Points_CV'),
        (pl.col('GF_Std') / pl.col('GF_PG')).round(4).alias('GF_CV'),
        (pl.col('GA_Std') / pl.col('GA_PG')).round(4).alias('GA_CV'),
    ])
    results['season_consistency'] = consistency

    # Team-level (3+ seasons)
    team_consistency = consistency.group_by('Team').agg([
        pl.len().alias('Seasons'),
        pl.col('PPG').mean().round(4).alias('Avg_PPG'),
        pl.col('Points_CV').mean().round(4).alias('Avg_Points_CV'),
        pl.col('GF_CV').mean().round(4).alias('Avg_GF_CV'),
        pl.col('GA_CV').mean().round(4).alias('Avg_GA_CV'),
    ]).filter(pl.col('Seasons') >= 3)
    results['team_consistency'] = team_consistency

    results['most_consistent'] = team_consistency.sort('Avg_Points_CV').head(10)
    results['least_consistent'] = team_consistency.sort('Avg_Points_CV', descending=True).head(10)

    results['median_points_cv'] = float(consistency.select(pl.col('Points_CV').median()).to_series()[0])
    results['median_gf_cv'] = float(consistency.select(pl.col('GF_CV').median()).to_series()[0])
    results['median_ga_cv'] = float(consistency.select(pl.col('GA_CV').median()).to_series()[0])

    return results


def analyze_discipline(team_matches: pl.DataFrame, master: pl.DataFrame,
                       min_matches: int = 20) -> Dict:
    """Analyze disciplinary patterns and their association with results."""
    results: Dict = {}

    # Team-season discipline
    team_season_disc = team_matches.group_by(TEAM_SEASON_GROUP).agg([
        pl.len().alias('Matches'),
        pl.col('Yellow').sum().alias('Yellow'),
        pl.col('Red').sum().alias('Red'),
        pl.col('Fouls').sum().alias('Fouls'),
        pl.col('Fouls_Against').sum().alias('Fouls_Against'),
        pl.col('Points').sum().alias('Points'),
    ]).filter(pl.col('Matches') >= min_matches).with_columns([
        (pl.col('Points') / pl.col('Matches')).round(4).alias('PPG'),
        (pl.col('Yellow') / pl.col('Matches')).round(4).alias('Yellow_PG'),
        (pl.col('Red') / pl.col('Matches')).round(4).alias('Red_PG'),
        (pl.col('Fouls') / pl.col('Matches')).round(4).alias('Fouls_PG'),
    ]).with_columns(
        (pl.col('Yellow_PG') + pl.col('Red_PG')).round(4).alias('Cards_PG')
    )

    # Correlation with PPG
    disc_corrs: Dict = {}
    for metric in ['Yellow_PG', 'Red_PG', 'Fouls_PG', 'Cards_PG']:
        v = team_season_disc.select(pl.col(metric).is_not_null() & pl.col('PPG').is_not_null()).to_series()
        if int(v.sum()) > 10:
            x = _df_to_np(team_season_disc.filter(v), metric)
            y = _df_to_np(team_season_disc.filter(v), 'PPG')
            r, p, n = _pearsonr(x, y)
            disc_corrs[metric] = {'r': r, 'p': p}
    results['ppg_correlations'] = disc_corrs

    # By venue
    venue_disc = team_matches.group_by('Venue').agg([
        pl.col('Yellow').mean().round(4).alias('Yellow_PG'),
        pl.col('Red').mean().round(4).alias('Red_PG'),
        pl.col('Fouls').mean().round(4).alias('Fouls_PG'),
        pl.len().alias('Matches'),
    ]).sort('Venue')
    results['by_venue'] = venue_disc

    # Red card impact
    home_red = master.filter(pl.col('HR') > 0)
    away_red = master.filter(pl.col('AR') > 0)
    baseline_home_win = float((master.select(pl.col('FTR') == 'H').to_series().mean()))
    baseline_away_win = float((master.select(pl.col('FTR') == 'A').to_series().mean()))
    baseline_draw = float((master.select(pl.col('FTR') == 'D').to_series().mean()))

    def _mean_bool(df: pl.DataFrame, cond: pl.Expr) -> float:
        if df.height == 0:
            return float('nan')
        return float(df.select(cond).to_series().mean())

    results['red_card_impact'] = {
        'baseline': {'home_win': baseline_home_win, 'draw': baseline_draw, 'away_win': baseline_away_win},
        'home_red': {
            'home_win': _mean_bool(home_red, pl.col('FTR') == 'H'),
            'draw': _mean_bool(home_red, pl.col('FTR') == 'D'),
            'away_win': _mean_bool(home_red, pl.col('FTR') == 'A'),
            'n': home_red.height,
        },
        'away_red': {
            'home_win': _mean_bool(away_red, pl.col('FTR') == 'H'),
            'draw': _mean_bool(away_red, pl.col('FTR') == 'D'),
            'away_win': _mean_bool(away_red, pl.col('FTR') == 'A'),
            'n': away_red.height,
        },
    }

    # Red card by HT result
    def _red_by_ht(df: pl.DataFrame):
        if df.height == 0:
            return {}
        g = df.group_by(['HTR', 'FTR']).agg(pl.len().alias('n'))
        totals = g.group_by('HTR').agg(pl.col('n').sum().alias('tot'))
        g = g.join(totals, on='HTR').with_columns((pl.col('n') / pl.col('tot')).round(3))
        return g.pivot(on='FTR', index='HTR', values='n').to_dicts()
    results['home_red_by_ht'] = _red_by_ht(home_red)
    results['away_red_by_ht'] = _red_by_ht(away_red)

    # Match-level: cards/fouls vs result
    tm = team_matches.with_columns(
        (pl.col('Yellow') + pl.col('Red')).alias('Card_Count')
    )
    result_numeric = pl.when(pl.col('FT_Result') == 'W').then(2).when(
        pl.col('FT_Result') == 'D').then(1).otherwise(0).cast(pl.Float64).alias('result_numeric')
    tm = tm.with_columns(result_numeric)
    valid = tm.select(pl.col('result_numeric').is_not_null() & pl.col('Card_Count').is_not_null() &
                      pl.col('Fouls').is_not_null() & pl.col('Fouls_Against').is_not_null()).to_series()
    if int(valid.sum()) > 2:
        tmv = tm.filter(valid)
        y = _df_to_np(tmv, 'result_numeric')
        x_cards = _df_to_np(tmv, 'Card_Count')
        r_cards, p_cards, _ = _pearsonr(x_cards, y)
        x_fouls = _df_to_np(tmv, 'Fouls')
        r_fouls, p_fouls, _ = _pearsonr(x_fouls, y)
        x_fa = _df_to_np(tmv, 'Fouls_Against')
        r_fouls_against, p_fouls_against, _ = _pearsonr(x_fa, y)
    else:
        r_cards = p_cards = r_fouls = p_fouls = r_fouls_against = p_fouls_against = float('nan')
    results['match_level_correlations'] = {
        'cards_vs_result': {'r': r_cards, 'p': p_cards},
        'fouls_vs_result': {'r': r_fouls, 'p': p_fouls},
        'fouls_drawn_vs_result': {'r': r_fouls_against, 'p': p_fouls_against},
    }

    # Foul differential
    team_season_disc2 = team_matches.group_by(TEAM_SEASON_GROUP).agg([
        pl.len().alias('Matches'),
        pl.col('Fouls').sum().alias('Fouls'),
        pl.col('Fouls_Against').sum().alias('Fouls_Against'),
        pl.col('Points').sum().alias('Points'),
    ]).filter(pl.col('Matches') >= min_matches).with_columns([
        (pl.col('Points') / pl.col('Matches')).round(4).alias('PPG'),
        ((pl.col('Fouls') - pl.col('Fouls_Against')) / pl.col('Matches')).round(4).alias('Foul_Diff_PG'),
    ])

    v = team_season_disc2.select(pl.col('Foul_Diff_PG').is_not_null() & pl.col('PPG').is_not_null()).to_series()
    if int(v.sum()) > 10:
        x = _df_to_np(team_season_disc2.filter(v), 'Foul_Diff_PG')
        y = _df_to_np(team_season_disc2.filter(v), 'PPG')
        r_fd, p_fd, _ = _pearsonr(x, y)
    else:
        r_fd = p_fd = float('nan')
    results['foul_differential_correlation'] = {'r': r_fd, 'p': p_fd}

    results['team_season_disc'] = team_season_disc
    results['team_season_disc2'] = team_season_disc2

    return results


def run_robustness_checks(team_matches: pl.DataFrame, master: pl.DataFrame) -> Dict:
    """Run robustness and sensitivity checks."""
    results: Dict = {}

    # Sample size sensitivity
    team_season = team_matches.group_by(TEAM_SEASON_GROUP).agg([
        pl.len().alias('Matches'),
        pl.col('GF').sum().alias('GF'),
        pl.col('GA').sum().alias('GA'),
        pl.col('Shots').sum().alias('Shots'),
        pl.col('Shots_OT').sum().alias('Shots_OT'),
        pl.col('Shots_Against').sum().alias('Shots_Against'),
        pl.col('Points').sum().alias('Points'),
    ]).with_columns([
        (pl.col('GF') / pl.col('Shots')).alias('GPS'),
        (pl.col('Points') / pl.col('Matches')).alias('PPG'),
    ])

    sample_sensitivity: Dict = {}
    for min_m in [10, 20, 30]:
        ts = team_season.filter(pl.col('Matches') >= min_m)
        if ts.height > 10:
            v = ts.select(pl.col('GPS').is_not_null() & pl.col('PPG').is_not_null()).to_series()
            if int(v.sum()) > 2:
                x = _df_to_np(ts.filter(v), 'GPS')
                y = _df_to_np(ts.filter(v), 'PPG')
                r, p, n = _pearsonr(x, y)
                sample_sensitivity[f'min_{min_m}'] = {'n': int(v.sum()), 'r': r, 'p': p}
    results['sample_size_sensitivity'] = sample_sensitivity

    # League tier comparison
    top_tier = ['E0', 'SP1', 'D1', 'I1', 'F1']
    lower_tier = ['E1', 'E2', 'E3']

    def _tier_half_eff(tier_data: pl.DataFrame) -> pl.DataFrame:
        tm_sub = tier_data.sort(['Div', 'Team', 'Season', 'Date'])
        grp = TEAM_SEASON_GROUP
        cc = tm_sub.select(pl.cum_count('Date').over(grp)).to_series().to_numpy()
        n = tm_sub.select(pl.len().over(grp)).to_series().to_numpy()
        mid = n // 2
        first_mask = cc < mid
        first = tm_sub.filter(pl.Series(first_mask))
        second = tm_sub.filter(pl.Series(~first_mask))
        sum_cols = ['GF', 'GA', 'Shots', 'Shots_OT', 'Shots_Against', 'Shots_OT_Against']
        first_sums = first.group_by(grp).agg([pl.col(c).sum().alias(f'{c}_F') for c in sum_cols])
        second_sums = second.group_by(grp).agg([pl.col(c).sum().alias(f'{c}_S') for c in sum_cols])
        half_eff = first_sums.join(second_sums, on=grp, how='full', coalesce=True)

        def _ratio0(num: str, den: str, name: str) -> pl.Expr:
            return pl.when(pl.col(den) > 0).then(pl.col(num) / pl.col(den)).otherwise(None).alias(name)

        half_eff = half_eff.with_columns([
            _ratio0('GF_F', 'Shots_F', 'GPS_F'),
            _ratio0('GF_F', 'Shots_OT_F', 'GPSOT_F'),
            _ratio0('GA_F', 'Shots_Against_F', 'GAOS_F'),
            _ratio0('GF_S', 'Shots_S', 'GPS_S'),
            _ratio0('GF_S', 'Shots_OT_S', 'GPSOT_S'),
            _ratio0('GA_S', 'Shots_Against_S', 'GAOS_S'),
        ])
        return half_eff

    tier_persistence: Dict = {}
    for tier_name, tiers in [('Top_5', top_tier), ('Lower_Eng', lower_tier)]:
        tier_data = team_matches.filter(pl.col('Div').is_in(tiers))
        if tier_data.height == 0:
            continue
        half_eff = _tier_half_eff(tier_data)
        for metric in ['GPS', 'GPSOT', 'GAOS']:
            f_col = f'{metric}_F'; s_col = f'{metric}_S'
            if f_col in half_eff.columns and s_col in half_eff.columns:
                v = half_eff.select(pl.col(f_col).is_not_null() & pl.col(s_col).is_not_null()).to_series()
                if int(v.sum()) > 10:
                    x = _df_to_np(half_eff.filter(v), f_col)
                    y = _df_to_np(half_eff.filter(v), s_col)
                    r, p, n = _pearsonr(x, y)
                    tier_persistence[f'{tier_name}_{metric}'] = {'r': r, 'p': p, 'n': n}
    results['tier_persistence'] = tier_persistence

    return results
