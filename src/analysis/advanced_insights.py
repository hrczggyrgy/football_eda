"""
Advanced insights module.
Implements priority analyses A-F for the visual research portfolio:
A. Finishing persistence (early vs late season)
B. Team profile changes (shot generation/suppression trajectories)
C. Halftime deficit recovery (conditional on deficit size, venue, strength)
D. Volatility and streakiness beyond simple dispersion
E. Pooled vs within-league relationships
F. Home advantage changes across seasons

Migrated to Polars: Polars is the primary analytical data structure. NumPy and
SciPy/scikit-learn used only where they provide appropriate statistical
functionality. No pandas.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import polars as pl
from scipy import stats
from scipy.stats import binomtest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from features.engineer import add_lagged_team_strength, SEASON_ORDER

TEAM_SEASON_GROUP = ['Div', 'Team', 'Season']


def _pearsonr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, int]:
    m = (~np.isnan(x)) & (~np.isnan(y))
    n = int(m.sum())
    if n < 2:
        return float('nan'), float('nan'), n
    r, p = stats.pearsonr(x[m], y[m])
    return float(r), float(p), n


def _np(df: pl.DataFrame, col: str) -> np.ndarray:
    return df.select(pl.col(col).cast(pl.Float64)).to_series().to_numpy()


def _ratio0_expr(num: str, den: str, name: str) -> pl.Expr:
    return pl.when(pl.col(den) > 0).then(pl.col(num) / pl.col(den)).otherwise(None).alias(name)


def _qcut3(expr: pl.Expr, labels: List[str]) -> pl.Expr:
    q1 = expr.quantile(1 / 3)
    q2 = expr.quantile(2 / 3)
    return (pl.when(expr <= q1).then(pl.lit(labels[0]))
            .when(expr <= q2).then(pl.lit(labels[1]))
            .otherwise(pl.lit(labels[2])))


def analyze_finishing_persistence(team_matches: pl.DataFrame, min_matches: int = 20) -> Dict:
    """
    Priority A: Does apparent finishing strength persist?
    """
    results: Dict = {}

    season_sizes = team_matches.group_by(TEAM_SEASON_GROUP).agg(pl.len().alias('Matches'))
    valid_seasons = season_sizes.filter(pl.col('Matches') >= min_matches)
    team_matches_valid = team_matches.join(valid_seasons.select(['Div', 'Team', 'Season']),
                                           on=['Div', 'Team', 'Season'], how='inner')

    tm_sorted = team_matches_valid.sort(['Div', 'Team', 'Season', 'Date'])
    grp = TEAM_SEASON_GROUP
    sum_cols = ['Shots', 'Shots_OT', 'GF', 'GA', 'Shots_Against', 'Shots_OT_Against']
    n = tm_sorted.select(pl.len().over(grp)).to_series().to_numpy()
    cc = tm_sorted.select(pl.cum_count('Date').over(grp)).to_series().to_numpy()
    early_mask = cc < (n // 2)
    early = tm_sorted.filter(pl.Series(early_mask))
    late = tm_sorted.filter(pl.Series(~early_mask))

    early_sums = early.group_by(grp).agg([pl.col(c).sum().alias(f'{c}_Early') for c in sum_cols])
    late_sums = late.group_by(grp).agg([pl.col(c).sum().alias(f'{c}_Late') for c in sum_cols])
    early_matches = early.group_by(grp).agg(pl.len().alias('Matches_Early'))
    late_matches = late.group_by(grp).agg(pl.len().alias('Matches_Late'))

    period_eff = early_sums.join(late_sums, on=grp, how='full', coalesce=True).join(
        early_matches, on=grp, how='full', coalesce=True).join(late_matches, on=grp, how='full', coalesce=True)

    ratio_pairs = [
        ('Shots_OT_Early', 'Shots_Early', 'Shot_Accuracy_Early'),
        ('GF_Early', 'Shots_Early', 'Goals_Per_Shot_Early'),
        ('GF_Early', 'Shots_OT_Early', 'Goals_Per_Shot_OT_Early'),
        ('GA_Early', 'Shots_Against_Early', 'GA_Per_Opp_Shot_Early'),
        ('GA_Early', 'Shots_OT_Against_Early', 'GA_Per_Opp_Shot_OT_Early'),
        ('Shots_OT_Late', 'Shots_Late', 'Shot_Accuracy_Late'),
        ('GF_Late', 'Shots_Late', 'Goals_Per_Shot_Late'),
        ('GF_Late', 'Shots_OT_Late', 'Goals_Per_Shot_OT_Late'),
        ('GA_Late', 'Shots_Against_Late', 'GA_Per_Opp_Shot_Late'),
        ('GA_Late', 'Shots_OT_Against_Late', 'GA_Per_Opp_Shot_OT_Late'),
    ]
    for num, den, name in ratio_pairs:
        if num in period_eff.columns and den in period_eff.columns:
            period_eff = period_eff.with_columns(_ratio0_expr(num, den, name))

    results['period_efficiency'] = period_eff

    min_shots = 30
    min_shots_ot = 10

    raw_persistence: Dict = {}
    pair_list = [
        'Shots_Early', 'Shots_OT_Early', 'Shot_Accuracy_Early', 'Goals_Per_Shot_Early',
        'Goals_Per_Shot_OT_Early', 'GA_Per_Opp_Shot_Early', 'GA_Per_Opp_Shot_OT_Early',
    ]
    for metric in pair_list:
        late_metric = metric.replace('_Early', '_Late')
        if metric not in period_eff.columns or late_metric not in period_eff.columns:
            continue
        if 'Goals_Per_Shot' in metric or 'GA_Per_Opp_Shot' in metric:
            exposure_col = 'Shots_Early' if ('Goals_Per_Shot' in metric and 'Opp' not in metric) else (
                'Shots_OT_Early' if 'Goals_Per_Shot_OT' in metric else 'Shots_Against_Early')
            thr = min_shots if 'Shot_OT' not in exposure_col else min_shots_ot
            valid = (period_eff.select(
                pl.col(metric).is_not_null() & pl.col(late_metric).is_not_null() &
                (pl.col(exposure_col) >= thr)).to_series())
        else:
            valid = period_eff.select(pl.col(metric).is_not_null() & pl.col(late_metric).is_not_null()).to_series()
        if int(valid.sum()) > 20:
            x = _np(period_eff.filter(valid), metric)
            y = _np(period_eff.filter(valid), late_metric)
            r, p, n = _pearsonr(x, y)
            raw_persistence[metric] = {'r': r, 'p': p, 'n': n}
    results['raw_persistence'] = raw_persistence

    # League-season adjusted (residualized) persistence
    period_eff = period_eff.with_columns(
        (pl.col('Div') + pl.lit('_') + pl.col('Season')).alias('LeagueSeason')
    )
    adjusted_persistence: Dict = {}
    for metric in pair_list:
        late_metric = metric.replace('_Early', '_Late')
        if metric not in period_eff.columns or late_metric not in period_eff.columns:
            continue
        valid_early = period_eff.select(pl.col(metric).is_not_null()).to_series()
        if int(valid_early.sum()) > 20:
            ls_mean_early = period_eff.filter(valid_early).group_by('LeagueSeason').agg(
                pl.col(metric).mean().alias('ls_mean_early'))
            period_eff = period_eff.join(ls_mean_early, on='LeagueSeason', how='left')
            period_eff = period_eff.with_columns(
                (pl.col(metric) - pl.col('ls_mean_early')).alias(f'{metric}_Adj')
            )
        valid_late = period_eff.select(pl.col(late_metric).is_not_null()).to_series()
        if int(valid_late.sum()) > 20:
            ls_mean_late = period_eff.filter(valid_late).group_by('LeagueSeason').agg(
                pl.col(late_metric).mean().alias('ls_mean_late'))
            period_eff = period_eff.join(ls_mean_late, on='LeagueSeason', how='left')
            period_eff = period_eff.with_columns(
                (pl.col(late_metric) - pl.col('ls_mean_late')).alias(f'{late_metric}_Adj')
            )
        adj_early = f'{metric}_Adj'
        adj_late = f'{late_metric}_Adj'
        if adj_early in period_eff.columns and adj_late in period_eff.columns:
            valid = period_eff.select(pl.col(adj_early).is_not_null() & pl.col(adj_late).is_not_null()).to_series()
            if int(valid.sum()) > 20:
                x = _np(period_eff.filter(valid), adj_early)
                y = _np(period_eff.filter(valid), adj_late)
                r, p, n = _pearsonr(x, y)
                adjusted_persistence[metric] = {'r': r, 'p': p, 'n': n}
        # Remove helper join columns so they don't collide on the next metric iteration
        drop_cols = [c for c in ['ls_mean_early', 'ls_mean_late'] if c in period_eff.columns]
        if drop_cols:
            period_eff = period_eff.drop(drop_cols)
    results['adjusted_persistence'] = adjusted_persistence

    # Quantile group trajectories
    trajectory_data: Dict = {}
    for metric in ['Goals_Per_Shot_Early', 'Goals_Per_Shot_OT_Early', 'GA_Per_Opp_Shot_Early']:
        late_metric = metric.replace('_Early', '_Late')
        if metric not in period_eff.columns or late_metric not in period_eff.columns:
            continue
        valid = period_eff.select(pl.col(metric).is_not_null() & pl.col(late_metric).is_not_null()).to_series()
        if int(valid.sum()) > 50:
            df_valid = period_eff.filter(valid).clone()
            # quintile bins via qcut equivalent
            qv = df_valid.select(pl.col(metric)).to_series()
            bounds = [float(qv.quantile(k / 5)) for k in range(1, 5)]
            cond = pl.lit(True)
            for k, b in enumerate(bounds, start=1):
                cond = pl.when(pl.col(metric) <= b).then(pl.lit(f'Q{k}')).otherwise(cond)
            df_valid = df_valid.with_columns(cond.alias('Quantile'))
            traj = df_valid.group_by('Quantile').agg([
                pl.col(late_metric).mean().alias('mean'),
                pl.col(late_metric).std().alias('std'),
                pl.len().alias('count'),
            ]).sort('Quantile')
            trajectory_data[metric] = traj.to_dicts()
    results['quantile_trajectories'] = trajectory_data

    # Sensitivity to split point (1/3, 1/2, 2/3)
    split_sensitivity: Dict = {}
    for split_frac in [1 / 3, 1 / 2, 2 / 3]:
        tm_v = team_matches_valid.sort(['Div', 'Team', 'Season', 'Date'])
        n_arr = tm_v.select(pl.len().over(TEAM_SEASON_GROUP)).to_series().to_numpy()
        cc_arr = tm_v.select(pl.cum_count('Date').over(TEAM_SEASON_GROUP)).to_series().to_numpy()
        split_mask = cc_arr < (n_arr * split_frac)
        early_s = tm_v.filter(pl.Series(split_mask))
        late_s = tm_v.filter(pl.Series(~split_mask))

        def calc(group_df, shots_col='Shots', shots_ot_col='Shots_OT', gf_col='GF'):
            shots = float(group_df.select(pl.col(shots_col).sum()).to_series()[0])
            shots_ot = float(group_df.select(pl.col(shots_ot_col).sum()).to_series()[0])
            gf = float(group_df.select(pl.col(gf_col).sum()).to_series()[0])
            return {'GPS': gf / shots if shots > 0 else float('nan'),
                    'GPSOT': gf / shots_ot if shots_ot > 0 else float('nan')}

        groups = []
        gb = early_s.group_by(TEAM_SEASON_GROUP)
        for key in gb:
            (div, team, season), gdf = key
            e = calc(early_s.filter((pl.col('Div') == div) & (pl.col('Team') == team) & (pl.col('Season') == season)))
            l = calc(late_s.filter((pl.col('Div') == div) & (pl.col('Team') == team) & (pl.col('Season') == season)))
            groups.append({'Div': div, 'Team': team, 'Season': season,
                           'GPS_Early': e['GPS'], 'GPSOT_Early': e['GPSOT'],
                           'GPS_Late': l['GPS'], 'GPSOT_Late': l['GPSOT']})
        custom_eff = pl.DataFrame(groups) if groups else pl.DataFrame()
        for metric in ['GPS_Early', 'GPSOT_Early']:
            late_metric = metric.replace('_Early', '_Late')
            if custom_eff.height > 0:
                valid = custom_eff.select(pl.col(metric).is_not_null() & pl.col(late_metric).is_not_null()).to_series()
                if int(valid.sum()) > 20:
                    x = _np(custom_eff.filter(valid), metric)
                    y = _np(custom_eff.filter(valid), late_metric)
                    r, p, n = _pearsonr(x, y)
                    split_sensitivity[f'{metric}_split{int(split_frac * 100)}'] = {'r': r, 'p': p, 'n': n}
    results['split_sensitivity'] = split_sensitivity

    return results


def analyze_team_profile_changes(team_season: pl.DataFrame, min_seasons: int = 2) -> Dict:
    """
    Priority B: Which teams changed their performance profile?
    """
    from analysis.new_insights import build_promotion_relegation_transitions

    results: Dict = {}

    ts = team_season.clone()
    if 'Shots_PG' not in ts.columns and {'Shots', 'Matches'} <= set(ts.columns):
        ts = ts.with_columns((pl.col('Shots') / pl.col('Matches')).alias('Shots_PG'))
    if 'Shots_Against_PG' not in ts.columns and {'Shots_Against', 'Matches'} <= set(ts.columns):
        ts = ts.with_columns((pl.col('Shots_Against') / pl.col('Matches')).alias('Shots_Against_PG'))

    ts = ts.with_columns((pl.col('Div') + pl.lit('_') + pl.col('Season')).alias('LeagueSeason'))

    metrics_to_normalize = ['Shots_PG', 'Shots_Against_PG', 'Shot_Accuracy',
                            'Goals_Per_Shot_OT', 'GA_Per_Opp_Shot_OT']
    for metric in metrics_to_normalize:
        if metric in ts.columns:
            ls_agg = ts.group_by('LeagueSeason').agg([
                pl.col(metric).mean().alias('ls_mean'),
                pl.col(metric).std().alias('ls_std'),
            ])
            ts = ts.join(ls_agg, on='LeagueSeason', how='left')
            ts = ts.with_columns(((pl.col(metric) - pl.col('ls_mean')) / pl.col('ls_std').replace(0, None)).alias(f'{metric}_z'))
            ts = ts.drop(['ls_mean', 'ls_std'])

    if 'Shots_Against_PG_z' in ts.columns:
        ts = ts.with_columns((-pl.col('Shots_Against_PG_z')).alias('Shot_Suppression_z'))
    if 'GA_Per_Opp_Shot_OT_z' in ts.columns:
        ts = ts.with_columns((-pl.col('GA_Per_Opp_Shot_OT_z')).alias('Def_Resilience_z'))

    ts_transitions = build_promotion_relegation_transitions(ts)

    consecutive = ts_transitions.filter(pl.col('Consecutive') & ~pl.col('Div_Changed'))

    profile_dims = ['Shots_PG_z', 'Goals_Per_Shot_OT_z', 'Shot_Suppression_z', 'Def_Resilience_z']
    available_dims = [d for d in profile_dims if d in consecutive.columns]

    for dim in available_dims:
        consecutive = consecutive.with_columns(
            pl.col(dim).diff().over(['Div', 'Team']).alias(f'Delta_{dim}')
        )

    within_div_changes = consecutive.select(
        available_dims + [f'Delta_{d}' for d in available_dims] + ['Div', 'Team', 'Season']
    )
    results['within_division_changes'] = within_div_changes

    promo_releg = ts_transitions.filter(pl.col('Consecutive') & pl.col('Div_Changed'))
    div_change_dims = [f'Delta_{d}' for d in available_dims if f'Delta_{d}' in promo_releg.columns]
    results['promotion_relegation_changes'] = promo_releg.select(
        ['Div', 'Prev_Div', 'Team', 'Season', 'Transition_Type'] + div_change_dims
    )

    change_summary: Dict = {}
    for dim in available_dims:
        delta_col = f'Delta_{dim}'
        if delta_col in consecutive.columns:
            valid = consecutive.select(pl.col(delta_col).is_not_null()).to_series()
            sub = consecutive.filter(valid)
            m = sub.select(pl.col(delta_col).mean()).to_series()[0]
            s = sub.select(pl.col(delta_col).std()).to_series()[0]
            change_summary[dim] = {
                'mean': (float(m) if m is not None else None),
                'std': (float(s) if s is not None else None),
                'n': int(sub.height),
                'top_improvers': sub.sort(delta_col, descending=True).head(10).to_dicts(),
                'top_decliners': sub.sort(delta_col).head(10).to_dicts(),
            }
    results['change_summary'] = change_summary

    sampling_var: Dict = {}
    for dim in available_dims:
        team_var = ts_transitions.group_by('Team').agg(pl.col(dim).var().alias('tv'))
        med_var = team_var.select(pl.col('tv').median()).to_series()[0]
        med_var = float(med_var) if med_var is not None else None
        delta_col = f'Delta_{dim}'
        mcv = consecutive.select(pl.col(delta_col).var()).to_series()[0] if delta_col in consecutive.columns else None
        mean_change_var = float(mcv) if mcv is not None else None
        sampling_var[dim] = {'median_within_team_var': med_var, 'mean_change_var': mean_change_var}
    results['sampling_variation'] = sampling_var

    return results


def analyze_halftime_deficit_recovery(master: pl.DataFrame, team_season: pl.DataFrame,
                                      team_matches: pl.DataFrame) -> Dict:
    """
    Priority C: How much does a halftime deficit really matter?
    Uses genuinely lagged pre-match strength.
    """
    results: Dict = {}

    ht_data = pl.concat([
        master.with_columns([
            pl.col('HomeTeam').alias('Team'), pl.lit('Home').alias('Venue'),
            (pl.col('HTHG') - pl.col('HTAG')).alias('HT_GD'),
            (pl.col('FTHG') - pl.col('FTAG')).alias('FT_GD'),
            pl.col('HTHG').alias('HT_Goals'), pl.col('FTHG').alias('FT_Goals'),
            pl.col('HTAG').alias('Opp_HT_Goals'), pl.col('FTAG').alias('Opp_FT_Goals'),
        ]),
        master.with_columns([
            pl.col('AwayTeam').alias('Team'), pl.lit('Away').alias('Venue'),
            (pl.col('HTAG') - pl.col('HTHG')).alias('HT_GD'),
            (pl.col('FTAG') - pl.col('FTHG')).alias('FT_GD'),
            pl.col('HTAG').alias('HT_Goals'), pl.col('FTAG').alias('FT_Goals'),
            pl.col('HTHG').alias('Opp_HT_Goals'), pl.col('FTHG').alias('Opp_FT_Goals'),
        ]),
    ], how='diagonal')

    tm = add_lagged_team_strength(team_matches)
    strength_lookup = tm.select(['Div', 'Team', 'Season', 'Date', 'PreMatch_PPG_Shrunk'])
    ht_data = ht_data.join(strength_lookup, on=['Div', 'Team', 'Season', 'Date'], how='left')

    season_order = {'1920': 0, '2021': 1, '2122': 2, '2223': 3,
                    '2324': 4, '2425': 5, '2526': 6, '2627': 7}
    smap = pl.DataFrame({'Season': list(season_order.keys()),
                         'Season_Num': list(season_order.values())})

    hist_thresholds = {}
    for div in ht_data.select(pl.col('Div').unique()).to_series().to_list():
        div_data = tm.filter(pl.col('Div') == div).join(smap, on='Season', how='left')
        div_data = div_data.sort(['Season_Num', 'Date'])
        for season_code in div_data.select(pl.col('Season').unique()).to_series().to_list():
            sn = season_order.get(season_code)
            if sn is None:
                continue
            hist_data = div_data.filter(pl.col('Season_Num') < sn)
            if hist_data.height > 50 and int(hist_data.select(pl.col('PreMatch_PPG_Shrunk').is_not_null().sum()).to_series()[0]) > 30:
                hist_ppg = hist_data.select(pl.col('PreMatch_PPG_Shrunk').drop_nulls()).to_series()
                q33 = float(hist_ppg.quantile(1 / 3))
                q66 = float(hist_ppg.quantile(2 / 3))
                hist_thresholds[(div, season_code)] = (q33, q66)

    ht_data = ht_data.join(smap, on='Season', how='left')

    def assign_strength(row_div, row_season, ppg):
        key = (row_div, row_season)
        if key in hist_thresholds and ppg is not None and not (isinstance(ppg, float) and np.isnan(ppg)):
            q33, q66 = hist_thresholds[key]
            if ppg <= q33:
                return 'Weak'
            elif ppg <= q66:
                return 'Average'
            else:
                return 'Strong'
        return None

    strength_col = []
    for r in ht_data.iter_rows(named=True):
        strength_col.append(assign_strength(r['Div'], r['Season'], r['PreMatch_PPG_Shrunk']))
    ht_data = ht_data.with_columns(pl.Series('Strength_Tercile', strength_col))

    ht_data = ht_data.with_columns([
        pl.when(pl.col('HT_GD') <= -3).then(pl.lit('-3 or worse'))
        .when(pl.col('HT_GD') == -2).then(pl.lit('-2'))
        .when(pl.col('HT_GD') == -1).then(pl.lit('-1'))
        .when(pl.col('HT_GD') == 0).then(pl.lit('0'))
        .when(pl.col('HT_GD') == 1).then(pl.lit('+1'))
        .when(pl.col('HT_GD') == 2).then(pl.lit('+2'))
        .when(pl.col('HT_GD') == 3).then(pl.lit('+3'))
        .otherwise(pl.lit('+3 or better')).alias('HT_GD_Cat'),
        pl.when(pl.col('FT_GD') > 0).then(pl.lit('Win'))
        .when(pl.col('FT_GD') < 0).then(pl.lit('Loss')).otherwise(pl.lit('Draw')).alias('FT_Result'),
        (pl.col('FT_GD') >= 0).cast(pl.Int8).alias('Avoided_Defeat'),
        (pl.col('FT_GD') > 0).cast(pl.Int8).alias('Won'),
    ])
    results['ht_data'] = ht_data

    recovery_by_gd_venue = ht_data.group_by(['HT_GD_Cat', 'Venue']).agg([
        pl.len().alias('N'),
        pl.col('Won').mean().alias('Win_Rate'),
        pl.col('Avoided_Defeat').mean().alias('Avoid_Defeat_Rate'),
        pl.col('FT_GD').mean().alias('Avg_FT_GD'),
    ]).sort(['HT_GD_Cat', 'Venue'])
    results['recovery_by_gd_venue'] = recovery_by_gd_venue

    if 'Strength_Tercile' in ht_data.columns:
        recovery_by_gd_venue_strength = ht_data.group_by(['HT_GD_Cat', 'Venue', 'Strength_Tercile']).agg([
            pl.len().alias('N'),
            pl.col('Won').mean().alias('Win_Rate'),
            pl.col('Avoided_Defeat').mean().alias('Avoid_Defeat_Rate'),
        ]).sort(['HT_GD_Cat', 'Venue', 'Strength_Tercile'])
        results['recovery_by_gd_venue_strength'] = recovery_by_gd_venue_strength

    trailing = ht_data.filter(pl.col('HT_GD') < 0).with_columns(
        (-pl.col('HT_GD')).alias('Deficit_Size')
    )
    trailing_recovery = trailing.group_by(['Deficit_Size', 'Venue']).agg([
        pl.len().alias('N'),
        pl.col('Won').mean().alias('Win_Rate'),
        pl.col('Avoided_Defeat').mean().alias('Avoid_Defeat_Rate'),
    ]).sort(['Deficit_Size', 'Venue'])
    results['trailing_recovery'] = trailing_recovery

    if 'Strength_Tercile' in trailing.columns:
        trailing_recovery_strength = trailing.group_by(['Deficit_Size', 'Venue', 'Strength_Tercile']).agg([
            pl.len().alias('N'),
            pl.col('Won').mean().alias('Win_Rate'),
            pl.col('Avoided_Defeat').mean().alias('Avoid_Defeat_Rate'),
        ]).sort(['Deficit_Size', 'Venue', 'Strength_Tercile'])
        results['trailing_recovery_strength'] = trailing_recovery_strength

    trailing_by_league = trailing.group_by(['Div', 'Venue']).agg([
        pl.len().alias('N'),
        pl.col('Won').mean().alias('Win_Rate'),
        pl.col('Avoided_Defeat').mean().alias('Avoid_Defeat_Rate'),
    ]).sort(['Div', 'Venue'])
    results['trailing_by_league'] = trailing_by_league

    trailing_win_rate = float(trailing.select(pl.col('Won').mean()).to_series()[0]) if trailing.height else float('nan')
    trailing_avoid = float(trailing.select(pl.col('Avoided_Defeat').mean()).to_series()[0]) if trailing.height else float('nan')
    leading = ht_data.filter(pl.col('HT_GD') > 0)
    drawing = ht_data.filter(pl.col('HT_GD') == 0)
    results['overall'] = {
        'total_matches': ht_data.height,
        'trailing_at_ht': trailing.height,
        'trailing_win_rate': trailing_win_rate,
        'trailing_avoid_defeat_rate': trailing_avoid,
        'leading_at_ht': leading.height,
        'leading_win_rate': float(leading.select(pl.col('Won').mean()).to_series()[0]) if leading.height else float('nan'),
        'drawing_at_ht': drawing.height,
        'drawing_win_rate': float(drawing.select(pl.col('Won').mean()).to_series()[0]) if drawing.height else float('nan'),
        'strength_tercile_coverage': float(ht_data.select(pl.col('Strength_Tercile').is_not_null().mean()).to_series()[0]),
    }

    return results


def analyze_volatility_streakiness(team_matches: pl.DataFrame, team_season: pl.DataFrame, min_matches: int = 20) -> Dict:
    """
    Priority D: Are some teams unusually volatile or streaky?
    """
    results: Dict = {}

    valid_seasons = team_season.filter(pl.col('Matches') >= min_matches).select(['Div', 'Team', 'Season'])
    team_matches_valid = team_matches.join(valid_seasons, on=['Div', 'Team', 'Season'], how='inner')
    team_matches_valid = team_matches_valid.sort(['Div', 'Team', 'Season', 'Date'])

    match_expectations = team_matches_valid.group_by(['Div', 'Season', 'Venue']).agg(
        pl.col('Points').mean().alias('Exp_Points_Baseline')
    )
    team_matches_valid = team_matches_valid.join(match_expectations, on=['Div', 'Season', 'Venue'], how='left')

    team_matches_valid = team_matches_valid.with_columns(
        pl.col('Exp_Points_Baseline').alias('Exp_Points_Model')
    )
    if 'Opp_PPG_Cum' in team_matches_valid.columns:
        valid = team_matches_valid.select(
            pl.col('Opp_PPG_Cum').is_not_null() & pl.col('Points').is_not_null()
        ).to_series()
        if int(valid.sum()) > 100:
            tmv = team_matches_valid.filter(valid)
            X = tmv.select('Opp_PPG_Cum').to_numpy().reshape(-1, 1)
            y = tmv.select('Points').to_numpy().ravel()
            reg = LinearRegression().fit(X, y)
            opp_vals = team_matches_valid.select('Opp_PPG_Cum').to_numpy().ravel()
            pred = np.full(opp_vals.shape, np.nan)
            m = ~np.isnan(opp_vals)
            pred[m] = reg.predict(opp_vals[m].reshape(-1, 1))
            team_matches_valid = team_matches_valid.with_columns(
                pl.Series('Exp_Points_Model_reg', pred)
            )
            team_matches_valid = team_matches_valid.with_columns(
                pl.coalesce(pl.col('Exp_Points_Model_reg'), pl.col('Exp_Points_Model')).alias('Exp_Points_Model')
            ).drop('Exp_Points_Model_reg')
    team_matches_valid = team_matches_valid.with_columns(
        (pl.col('Points') - pl.col('Exp_Points_Model')).alias('Residual')
    )

    residual_disp = team_matches_valid.group_by(TEAM_SEASON_GROUP).agg([
        pl.col('Residual').std().alias('Residual_Std'),
        pl.col('Residual').mean().alias('Residual_Mean'),
        pl.col('Points').std().alias('Points_Std'),
        pl.col('Points').mean().alias('Points_Mean'),
        pl.len().alias('Matches'),
    ]).filter(pl.col('Matches') >= min_matches).with_columns(
        (pl.col('Residual_Std') / pl.col('Residual_Mean').abs().replace(0, None)).alias('Residual_CV')
    )
    results['residual_dispersion'] = residual_disp

    resid_lookup = {}
    exp_lookup = {}
    gb = team_matches_valid.group_by(TEAM_SEASON_GROUP)
    for key in gb:
        (d, t, s), g = key
        resid_lookup[(d, t, s)] = g.select(pl.col('Residual').drop_nulls().cast(pl.Float64)).to_series().to_numpy()
        exp_lookup[(d, t, s)] = g.select(pl.col('Exp_Points_Model').drop_nulls().cast(pl.Float64)).to_series().to_numpy()

    draw_rate = team_matches_valid.with_columns(
        (pl.col('FT_Result') == 'D').cast(pl.Float64).alias('_is_d')
    ).group_by(['Div', 'Season']).agg(pl.col('_is_d').mean())
    default_draw_rate = 0.25

    n_sims = 100
    sim_dispersions = []
    pts_vals = np.array([0.0, 1.0, 3.0])
    np.random.seed(42)

    for row in residual_disp.iter_rows(named=True):
        key = (row['Div'], row['Team'], row['Season'])
        exp_points = exp_lookup.get(key)
        if exp_points is None or len(exp_points) < min_matches:
            continue
        n_matches = len(exp_points)
        dr = draw_rate.filter((pl.col('Div') == row['Div']) & (pl.col('Season') == row['Season']))
        p1 = float(dr.select('_is_d').to_series()[0]) if dr.height else default_draw_rate
        if not np.isfinite(p1):
            p1 = default_draw_rate
        p3 = np.clip((exp_points - p1) / 3.0, 0.0, 1.0)
        p1_arr = np.full(n_matches, p1, dtype=float)
        p0 = 1.0 - p3 - p1_arr
        m0 = p0 < 0
        if m0.any():
            p3[m0] = np.clip(exp_points[m0] / 3.0, 0.0, 1.0)
            p1_arr[m0] = 1.0 - p3[m0]
            p0 = 1.0 - p3 - p1_arr
        ok = (np.isfinite(p3) & np.isfinite(p0) & (p3 >= 0) & (p0 >= 0)
              & (np.abs(p3 + p1_arr + p0 - 1.0) <= 1e-10))
        if not ok.all():
            p3 = np.where(ok, p3, 1.0 / 3.0)
            p0 = np.where(ok, p0, 1.0 / 3.0)
            p1_arr = np.where(ok, p1_arr, 1.0 / 3.0)
        u = np.random.random((n_sims, n_matches))
        cp0 = p0[None, :]
        cp1 = (p0 + p1_arr)[None, :]
        draw_idx = (u >= cp0).astype(np.int8) + (u >= cp1).astype(np.int8)
        sim_points = np.take(pts_vals, draw_idx)
        sim_stds = sim_points.std(axis=1)
        sim_mean = float(sim_stds.mean())
        sim_std = float(sim_stds.std(ddof=0))
        z_score = (row['Points_Std'] - sim_mean) / sim_std if sim_std > 0 else float('nan')
        sim_dispersions.append({
            'Div': row['Div'], 'Team': row['Team'], 'Season': row['Season'],
            'Observed_Std': float(row['Points_Std']),
            'Simulated_Std_Mean': sim_mean,
            'Simulated_Std_Std': sim_std,
            'Z_Score': z_score,
            'N_Sims': len(sim_stds),
        })
    results['simulation_benchmark'] = pl.DataFrame(sim_dispersions) if sim_dispersions else pl.DataFrame()

    results['simulation_assumptions'] = {
        'exchangeability': 'Matches within a team-season are exchangeable conditional on expected points',
        'outcome_model': 'Multinomial (Win/Draw/Loss) with probabilities calibrated to expected points and league-season draw rate',
        'n_sims': n_sims,
        'monte_carlo_precision': f'SE of Z-score ~ 1/sqrt({n_sims}) ≈ 0.1',
    }

    streakiness = []
    for key, residuals in resid_lookup.items():
        if len(residuals) < 10:
            continue
        div, team, season = key
        median_res = float(np.median(residuals))
        signs = (residuals > median_res).astype(int)
        runs = 1 + int(np.sum(signs[1:] != signs[:-1]))
        n_pos = int(np.sum(signs))
        n_neg = len(signs) - n_pos
        expected_runs = 1 + 2 * n_pos * n_neg / (n_pos + n_neg)
        runs_var = ((2 * n_pos * n_neg * (2 * n_pos * n_neg - n_pos - n_neg))
                    / ((n_pos + n_neg) ** 2 * (n_pos + n_neg - 1))) if (n_pos + n_neg) > 1 else 0.0
        runs_z = (runs - expected_runs) / np.sqrt(runs_var) if runs_var > 0 else 0.0
        acf1 = float(np.corrcoef(residuals[:-1], residuals[1:])[0, 1]) if len(residuals) > 1 else 0.0
        streakiness.append({
            'Div': div, 'Team': team, 'Season': season,
            'Runs': runs, 'Expected_Runs': expected_runs, 'Runs_Z': runs_z,
            'ACF1': acf1, 'Residual_Std': float(np.std(residuals)),
        })
    results['streakiness'] = pl.DataFrame(streakiness) if streakiness else pl.DataFrame()

    perm_results = []
    n_perm = 20
    for key, residuals in resid_lookup.items():
        if len(residuals) < 10:
            continue
        div, team, season = key
        median_res = float(np.median(residuals))
        signs = (residuals > median_res).astype(int)
        obs_runs = 1 + int(np.sum(signs[1:] != signs[:-1]))
        n_pos = int(np.sum(signs))
        n_neg = len(signs) - n_pos
        expected_runs = 1 + 2 * n_pos * n_neg / (n_pos + n_neg)
        runs_var = ((2 * n_pos * n_neg * (2 * n_pos * n_neg - n_pos - n_neg))
                    / ((n_pos + n_neg) ** 2 * (n_pos + n_neg - 1))) if (n_pos + n_neg) > 1 else 0.0
        obs_runs_z = (obs_runs - expected_runs) / np.sqrt(runs_var) if runs_var > 0 else 0.0
        perm_zs = []
        for _ in range(n_perm):
            perm_res = np.random.permutation(residuals)
            perm_signs = (perm_res > median_res).astype(int)
            perm_runs = 1 + int(np.sum(perm_signs[1:] != perm_signs[:-1]))
            perm_z = (perm_runs - expected_runs) / np.sqrt(runs_var) if runs_var > 0 else 0.0
            perm_zs.append(perm_z)
        p_value = (float(np.sum(np.abs(perm_zs) >= np.abs(obs_runs_z))) + 1) / (n_perm + 1)
        perm_results.append({
            'Div': div, 'Team': team, 'Season': season,
            'Observed_Runs_Z': float(obs_runs_z), 'Perm_P_Value': float(p_value),
            'Streaky': bool(p_value < 0.05 and obs_runs_z < 0),
        })
    results['permutation_streakiness'] = pl.DataFrame(perm_results) if perm_results else pl.DataFrame()

    if results['permutation_streakiness'].height > 0:
        team_streak = results['permutation_streakiness'].group_by('Team').agg([
            pl.len().alias('Seasons'),
            pl.col('Observed_Runs_Z').mean().alias('Avg_Runs_Z'),
            pl.col('Streaky').sum().alias('Streaky_Seasons'),
        ]).filter(pl.col('Seasons') >= 3).sort('Avg_Runs_Z')
        results['team_streakiness_summary'] = team_streak

    return results


def analyze_pooled_vs_within_league(team_season: pl.DataFrame, min_matches: int = 20) -> Dict:
    """
    Priority E: Do pooled relationships survive within leagues?
    """
    results: Dict = {}

    valid = team_season.filter(pl.col('Matches') >= min_matches).clone()

    relationships = [
        ('Shots_PG', 'PPG', 'Shot Volume -> Points'),
        ('Shot_Accuracy', 'PPG', 'Shot Accuracy -> Points'),
        ('Goals_Per_Shot_OT', 'PPG', 'Finishing -> Points'),
        ('GA_Per_Opp_Shot', 'PPG', 'Defensive Efficiency -> Points'),
        ('Yellow_PG', 'PPG', 'Discipline -> Points'),
        ('Fouls_PG', 'PPG', 'Fouls -> Points'),
        ('Shots_PG', 'Goals_Per_Shot_OT', 'Volume -> Finishing'),
    ]

    pooled_corrs: Dict = {}
    for x, y, label in relationships:
        if x in valid.columns and y in valid.columns:
            v = valid.select(pl.col(x).is_not_null() & pl.col(y).is_not_null()).to_series()
            if int(v.sum()) > 20:
                r, p, n = _pearsonr(_np(valid.filter(v), x), _np(valid.filter(v), y))
                pooled_corrs[label] = {'r': r, 'p': p, 'n': n}
    results['pooled'] = pooled_corrs

    within_league_corrs: Dict = {}
    for league in valid.select(pl.col('Div').unique()).to_series().to_list():
        lg_data = valid.filter(pl.col('Div') == league)
        league_corrs = {}
        for x, y, label in relationships:
            if x in lg_data.columns and y in lg_data.columns:
                v = lg_data.select(pl.col(x).is_not_null() & pl.col(y).is_not_null()).to_series()
                if int(v.sum()) > 10:
                    r, p, n = _pearsonr(_np(lg_data.filter(v), x), _np(lg_data.filter(v), y))
                    league_corrs[label] = {'r': r, 'p': p, 'n': n}
        within_league_corrs[league] = league_corrs
    results['within_league'] = within_league_corrs

    valid = valid.with_columns((pl.col('Div') + pl.lit('_') + pl.col('Season')).alias('LeagueSeason'))

    adjusted_corrs: Dict = {}
    for x, y, label in relationships:
        if x in valid.columns and y in valid.columns:
            v = valid.select(pl.col(x).is_not_null() & pl.col(y).is_not_null()).to_series()
            if int(v.sum()) > 20:
                vdf = valid.filter(v)
                ls_x = vdf.group_by('LeagueSeason').agg(pl.col(x).mean().alias('ls_x'))
                ls_y = vdf.group_by('LeagueSeason').agg(pl.col(y).mean().alias('ls_y'))
                vdf = vdf.join(ls_x, on='LeagueSeason', how='left').join(ls_y, on='LeagueSeason', how='left')
                vdf = vdf.with_columns([
                    (pl.col(x) - pl.col('ls_x')).alias('x_resid'),
                    (pl.col(y) - pl.col('ls_y')).alias('y_resid'),
                ])
                v2 = vdf.select(pl.col('x_resid').is_not_null() & pl.col('y_resid').is_not_null()).to_series()
                if int(v2.sum()) > 20:
                    r, p, n = _pearsonr(_np(vdf.filter(v2), 'x_resid'), _np(vdf.filter(v2), 'y_resid'))
                    adjusted_corrs[label] = {'r': r, 'p': p, 'n': n}
    results['league_season_adjusted'] = adjusted_corrs

    reg_results: Dict = {}
    for x, y, label in [('Shots_PG', 'PPG', 'Volume->Points'), ('Goals_Per_Shot_OT', 'PPG', 'Finishing->Points')]:
        if x in valid.columns and y in valid.columns:
            v = valid.select(pl.col(x).is_not_null() & pl.col(y).is_not_null() & pl.col('LeagueSeason').is_not_null()).to_series()
            if int(v.sum()) > 50:
                df_reg = valid.filter(v).select([x, y, 'LeagueSeason']).drop_nulls()
                dummies = df_reg.select('LeagueSeason').to_dummies(columns='LeagueSeason', drop_first=True)
                X = pl.concat([df_reg.select(x), dummies], how='horizontal').to_numpy()
                y_reg = df_reg.select(y).to_numpy().ravel()
                reg = LinearRegression(fit_intercept=True).fit(X, y_reg)
                reg_results[label] = {
                    'coef': float(reg.coef_[0]),
                    'intercept': float(reg.intercept_),
                    'r2': float(reg.score(X, y_reg)),
                    'n': int(df_reg.height),
                    'n_fixed_effects': int(X.shape[1] - 1),
                }
    results['fixed_effects_regression'] = reg_results

    heterogeneity: Dict = {}
    for x, y, label in relationships:
        if x in valid.columns and y in valid.columns:
            league_rs = []
            league_ns = []
            for league in valid.select(pl.col('Div').unique()).to_series().to_list():
                lg_data = valid.filter(pl.col('Div') == league)
                v = lg_data.select(pl.col(x).is_not_null() & pl.col(y).is_not_null()).to_series()
                if int(v.sum()) > 10:
                    r, _, _ = _pearsonr(_np(lg_data.filter(v), x), _np(lg_data.filter(v), y))
                    league_rs.append(r)
                    league_ns.append(int(v.sum()))
            if len(league_rs) >= 3:
                z_scores = [float(np.arctanh(r)) for r in league_rs]
                weights = [n - 3 for n in league_ns]
                wsum = float(sum(weights))
                weighted_mean_z = sum(w * z for w, z in zip(weights, z_scores)) / wsum if wsum > 0 else 0.0
                Q = float(sum(w * (z - weighted_mean_z) ** 2 for w, z in zip(weights, z_scores)))
                df = len(z_scores) - 1
                p_het = float(1 - stats.chi2.cdf(Q, df)) if df > 0 else 1.0
                heterogeneity[label] = {
                    'league_correlations': dict(zip(valid.select(pl.col('Div').unique()).to_series().to_list(), league_rs)),
                    'mean_r': float(np.tanh(weighted_mean_z)),
                    'Q': Q,
                    'df': int(df),
                    'p_heterogeneity': p_het,
                    'I2': float(max(0, (Q - df) / Q * 100)) if Q > 0 else 0.0,
                }
    results['heterogeneity'] = heterogeneity

    return results


def analyze_home_advantage_changes(master: pl.DataFrame, team_season: pl.DataFrame) -> Dict:
    """
    Priority F: How did home advantage change across seasons?
    """
    results: Dict = {}

    ha_ls = master.group_by(['Div', 'Season']).agg([
        ((pl.col('FTR') == 'H').mean()).alias('Home_Win_Pct'),
        ((pl.col('FTR') == 'D').mean()).alias('Draw_Pct'),
        ((pl.col('FTR') == 'A').mean()).alias('Away_Win_Pct'),
        pl.col('FTHG').mean().alias('Home_Goals'),
        pl.col('FTAG').mean().alias('Away_Goals'),
        (pl.col('FTHG').mean() - pl.col('FTAG').mean()).alias('Goal_Diff'),
        pl.len().alias('Matches'),
    ]).sort(['Div', 'Season'])
    # Home share of decisive matches (computed from raw match counts)
    decisive = master.group_by(['Div', 'Season']).agg([
        ((pl.col('FTR') == 'H').sum()).alias('HW'),
        ((pl.col('FTR') == 'A').sum()).alias('AW'),
    ]).with_columns(
        pl.when((pl.col('HW') + pl.col('AW')) > 0)
        .then(pl.col('HW') / (pl.col('HW') + pl.col('AW')))
        .otherwise(None).alias('Home_Win_Share_Decisive')
    ).select(['Div', 'Season', 'Home_Win_Share_Decisive'])
    ha_ls = ha_ls.join(decisive, on=['Div', 'Season'], how='left')
    ha_ls = ha_ls.with_columns((pl.col('Home_Win_Pct') - pl.col('Away_Win_Pct')).alias('HA_Win_Diff'))
    results['league_season'] = ha_ls

    ha_ls = ha_ls.with_columns((pl.col('Home_Win_Pct') - pl.col('Away_Win_Pct')).alias('HA_Win_Diff'))

    league_trends: Dict = {}
    for league in ha_ls.select(pl.col('Div').unique()).to_series().to_list():
        lg = ha_ls.filter(pl.col('Div') == league).sort('Season')
        if lg.height >= 3:
            season_nums = np.arange(lg.height)
            for metric in ['Home_Win_Pct', 'HA_Win_Diff', 'Home_Win_Share_Decisive', 'Goal_Diff']:
                if metric in lg.columns and lg.select(pl.col(metric).is_not_null().sum()).to_series()[0] >= 3:
                    v = lg.select(pl.col(metric).is_not_null()).to_series()
                    sn_v = season_nums[v.to_numpy()]
                    y_v = _np(lg.filter(v), metric)
                    slope, intercept, r, p, se = stats.linregress(sn_v, y_v)
                    league_trends[f'{league}_{metric}'] = {
                        'slope_per_season': float(slope), 'p_value': float(p),
                        'r_squared': float(r ** 2), 'n_seasons': int(v.sum()),
                    }
    results['league_trends'] = league_trends

    overall_trend: Dict = {}
    for metric in ['Home_Win_Pct', 'HA_Win_Diff', 'Home_Win_Share_Decisive', 'Goal_Diff']:
        if metric in ha_ls.columns:
            v = ha_ls.select(pl.col(metric).is_not_null()).to_series()
            if int(v.sum()) >= 3:
                slope, intercept, r, p, se = stats.linregress(np.arange(int(v.sum())), _np(ha_ls.filter(v), metric))
                overall_trend[metric] = {
                    'slope_per_season': float(slope), 'p_value': float(p),
                    'r_squared': float(r ** 2), 'n_league_seasons': int(v.sum()),
                }
    results['overall_trend'] = overall_trend

    balanced_teams = team_season.group_by(['Div', 'Team']).agg(pl.col('Season').n_unique().alias('n_seasons'))
    balanced_teams = balanced_teams.filter(pl.col('n_seasons') == 7).select(['Div', 'Team'])

    if balanced_teams.height > 0:
        balanced_ha = master.join(balanced_teams, left_on=['Div', 'HomeTeam'], right_on=['Div', 'Team'], how='inner')
        balanced_ha = balanced_ha.group_by(['Div', 'Season']).agg([
            ((pl.col('FTR') == 'H').mean()).alias('Home_Win_Pct'),
            ((pl.col('FTR') == 'A').mean()).alias('Away_Win_Pct'),
            (pl.col('FTHG').mean() - pl.col('FTAG').mean()).alias('Goal_Diff'),
            pl.len().alias('Matches'),
        ]).with_columns((pl.col('Home_Win_Pct') - pl.col('Away_Win_Pct')).alias('HA_Win_Diff')).sort(['Div', 'Season'])
    else:
        balanced_ha = pl.DataFrame({'Div': [], 'Season': [], 'Home_Win_Pct': [], 'Away_Win_Pct': [], 'Goal_Diff': [], 'Matches': [], 'HA_Win_Diff': []})
    results['balanced_sample'] = balanced_ha

    pre_covid = ha_ls.filter(pl.col('Season').is_in(['1920']))
    covid = ha_ls.filter(pl.col('Season') == '2021')
    post_covid = ha_ls.filter(pl.col('Season').is_in(['2122', '2223', '2324', '2425', '2526']))
    covid_comparison: Dict = {}
    for metric in ['Home_Win_Pct', 'HA_Win_Diff', 'Home_Win_Share_Decisive', 'Goal_Diff']:
        pre_val = float(pre_covid.select(pl.col(metric).mean()).to_series()[0]) if pre_covid.height else float('nan')
        covid_val = float(covid.select(pl.col(metric).mean()).to_series()[0]) if covid.height else float('nan')
        post_val = float(post_covid.select(pl.col(metric).mean()).to_series()[0]) if post_covid.height else float('nan')
        covid_comparison[metric] = {
            'pre_covid': pre_val, 'covid_season': covid_val, 'post_covid': post_val,
            'covid_drop': covid_val - pre_val, 'recovery': post_val - covid_val,
        }
    results['covid_comparison'] = covid_comparison

    team_ha = master.group_by(['Div', 'HomeTeam', 'Season']).agg([
        ((pl.col('FTR') == 'H').mean()).alias('Home_Win_Pct'),
        pl.len().alias('Home_Matches'),
    ]).filter(pl.col('Home_Matches') >= 15).rename({'HomeTeam': 'Team'})
    away_ha = master.group_by(['Div', 'AwayTeam', 'Season']).agg([
        ((pl.col('FTR') == 'A').mean()).alias('Away_Win_Pct'),
        pl.len().alias('Away_Matches'),
    ]).filter(pl.col('Away_Matches') >= 15).rename({'AwayTeam': 'Team'})
    team_ha_full = team_ha.join(away_ha.select(['Div', 'Team', 'Season', 'Away_Win_Pct']), on=['Div', 'Team', 'Season'], how='inner')
    team_ha_full = team_ha_full.with_columns((pl.col('Home_Win_Pct') - pl.col('Away_Win_Pct')).alias('HA_Diff'))

    team_ha_trends: Dict = {}
    gb = team_ha_full.group_by(['Div', 'Team'])
    for key in gb:
        (div, team), g = key
        if g.height >= 4:
            v = g.select(pl.col('HA_Diff').is_not_null()).to_series()
            if int(v.sum()) >= 4:
                slope, intercept, r, p, se = stats.linregress(np.arange(int(v.sum())), _np(g.filter(v), 'HA_Diff'))
                team_ha_trends[f'{div}_{team}'] = {
                    'slope': float(slope), 'p': float(p), 'n_seasons': int(v.sum()),
                }
    results['team_ha_trends'] = team_ha_trends

    return results
