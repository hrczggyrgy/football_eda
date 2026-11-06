"""
Next Insights Analysis Module
Implements 5 new research questions beyond existing analyses.

Migrated to Polars: Polars is the primary analytical data structure. NumPy and
SciPy/scikit-learn used where they provide appropriate statistical
functionality. No pandas.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import polars as pl
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from features.engineer import SEASON_ORDER

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


def analyze_competitive_balance(team_season: pl.DataFrame) -> Dict:
    """
    Question 1: How has competitive balance evolved across leagues and seasons?
    """
    results: Dict = {}

    balance_data = []
    gb = team_season.group_by(['Div', 'Season'])
    for key in gb:
        (div, season), g = key
        ppg = g.select(pl.col('PPG')).to_series().to_numpy()
        n = len(ppg)
        if n < 4:
            continue
        shares = ppg / ppg.sum()
        hhi = float((shares ** 2).sum())
        hhi_norm = (hhi - 1 / n) / (1 - 1 / n) if n > 1 else 0.0
        sorted_ppg = np.sort(ppg)
        cumsum = np.cumsum(sorted_ppg)
        gini = float((2 * np.sum((np.arange(1, n + 1) * sorted_ppg))) / (n * cumsum[-1]) - (n + 1) / n)
        top4 = float(np.mean(np.sort(ppg)[-4:]))
        bot4 = float(np.mean(np.sort(ppg)[:4]))
        gap = top4 - bot4
        top4_share = float(np.sum(np.sort(ppg)[-4:]) / np.sum(ppg))
        balance_data.append({
            'Div': div, 'Season': season, 'N': n,
            'HHI': hhi, 'HHI_Norm': hhi_norm, 'Gini': gini,
            'Top4_PPG': top4, 'Bot4_PPG': bot4, 'Gap': gap,
            'Top4_Share': top4_share,
            'Mean_PPG': float(ppg.mean()), 'Std_PPG': float(ppg.std()),
        })
    balance_df = pl.DataFrame(balance_data) if balance_data else pl.DataFrame()
    results['league_season_balance'] = balance_df

    league_avg = balance_df.group_by('Div').agg([
        pl.col('HHI_Norm').mean().round(4),
        pl.col('Gini').mean().round(4),
        pl.col('Gap').mean().round(4),
        pl.col('Top4_Share').mean().round(4),
        pl.col('Mean_PPG').mean().round(4),
        pl.col('Std_PPG').mean().round(4),
    ]).sort('Div')
    results['league_averages'] = league_avg

    league_trends: Dict = {}
    for div in balance_df.select(pl.col('Div').unique()).to_series().to_list():
        lg = balance_df.filter(pl.col('Div') == div).sort('Season')
        if lg.height >= 3:
            season_nums = np.arange(lg.height)
            for metric in ['HHI_Norm', 'Gini', 'Gap', 'Top4_Share']:
                if metric in lg.columns and int(lg.select(pl.col(metric).is_not_null().sum()).to_series()[0]) >= 3:
                    v = lg.select(pl.col(metric).is_not_null()).to_series()
                    sn_v = season_nums[v.to_numpy()]
                    y_v = _np(lg.filter(v), metric)
                    slope, intercept, r, p, se = stats.linregress(sn_v, y_v)
                    league_trends[f'{div}_{metric}'] = {
                        'slope_per_season': float(slope),
                        'intercept': float(intercept),
                        'r': float(r), 'p': float(p), 'se': float(se),
                        'n_seasons': int(v.sum()),
                        'significant': bool(p < 0.05),
                    }
    results['league_trends'] = league_trends

    overall_trend: Dict = {}
    for metric in ['HHI_Norm', 'Gini', 'Gap', 'Top4_Share']:
        if metric in balance_df.columns:
            v = balance_df.select(pl.col(metric).is_not_null()).to_series()
            if int(v.sum()) >= 3:
                slope, intercept, r, p, se = stats.linregress(np.arange(int(v.sum())), _np(balance_df.filter(v), metric))
                overall_trend[metric] = {
                    'slope_per_season': float(slope), 'p': float(p), 'r': float(r), 'n': int(v.sum()),
                }
    results['overall_trend'] = overall_trend

    # Bootstrap confidence intervals for trend slopes.
    n_boot = 1000
    np.random.seed(42)
    bootstrap_trends: Dict = {}
    for metric in ['HHI_Norm', 'Gini', 'Gap', 'Top4_Share']:
        if metric in balance_df.columns:
            v = balance_df.select(pl.col(metric).is_not_null()).to_series()
            if int(v.sum()) >= 3:
                df_valid = balance_df.filter(v)
                y = df_valid.select(pl.col(metric).cast(pl.Float64)).to_series().to_numpy()
                n = len(y)
                x = np.arange(n, dtype=float)
                xc = x - x.mean()
                denom = float((xc * xc).sum())
                idx = np.random.randint(0, n, size=(n_boot, n))
                y_boot = y[idx]
                slopes = (xc[None, :] * y_boot).sum(axis=1) / denom
                bootstrap_trends[metric] = {
                    'slope_mean': float(np.mean(slopes)),
                    'slope_std': float(np.std(slopes)),
                    'ci_95': (float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))),
                }
    results['bootstrap_trends'] = bootstrap_trends

    return results


def analyze_league_style_fingerprints(team_season: pl.DataFrame, team_matches: pl.DataFrame) -> Dict:
    """
    Question 2: Do leagues have distinct, persistent statistical fingerprints?
    """
    results: Dict = {}

    style_metrics = team_season.group_by(['Div', 'Season']).agg([
        pl.col('Shots_PG').mean().alias('Shots_PG'),
        pl.col('Shots_OT_PG').mean().alias('Shots_OT_PG'),
        pl.col('Shot_Accuracy').mean().alias('Shot_Accuracy'),
        pl.col('Goals_Per_Shot').mean().alias('Goals_Per_Shot'),
        pl.col('Goals_Per_Shot_OT').mean().alias('Goals_Per_Shot_OT'),
        pl.col('Corners_PG').mean().alias('Corners_PG'),
        pl.col('Corners_Ag_PG').mean().alias('Corners_Ag_PG'),
        pl.col('Goals_Per_Corner').mean().alias('Goals_Per_Corner'),
        pl.col('Fouls_PG').mean().alias('Fouls_PG'),
        pl.col('Yellow').mean().alias('Yellow_PG'),
        pl.col('Red').mean().alias('Red_PG'),
        pl.col('GF_PG').mean().alias('GF_PG'),
        pl.col('GA_PG').mean().alias('GA_PG'),
    ]).sort(['Div', 'Season'])

    style_metrics = style_metrics.with_columns([
        (pl.col('Shots_PG') / pl.col('GF_PG')).alias('Shots_Per_Goal'),
        (pl.col('Corners_PG') / pl.col('GF_PG')).alias('Corners_Per_Goal'),
        (pl.col('Fouls_PG') / pl.col('GF_PG')).alias('Fouls_Per_Goal'),
        (pl.col('Shots_PG') / pl.col('Corners_PG')).alias('Shots_Per_Corner'),
        (pl.col('Corners_PG') / pl.col('Fouls_PG')).alias('Corners_Per_Foul'),
    ])

    results['league_season_style'] = style_metrics

    league_avg = style_metrics.group_by('Div').agg([
        pl.col('Shots_PG').mean().round(3),
        pl.col('Shot_Accuracy').mean().round(3),
        pl.col('Goals_Per_Shot').mean().round(3),
        pl.col('Corners_PG').mean().round(3),
        pl.col('Goals_Per_Corner').mean().round(3),
        pl.col('Fouls_PG').mean().round(3),
        pl.col('Shots_Per_Goal').mean().round(3),
        pl.col('Corners_Per_Goal').mean().round(3),
        pl.col('Fouls_Per_Goal').mean().round(3),
        pl.col('Shots_Per_Corner').mean().round(3),
        pl.col('Corners_Per_Foul').mean().round(3),
    ]).sort('Div')
    results['league_averages'] = league_avg

    style_trends: Dict = {}
    for div in style_metrics.select(pl.col('Div').unique()).to_series().to_list():
        lg = style_metrics.filter(pl.col('Div') == div).sort('Season')
        if lg.height >= 3:
            season_nums = np.arange(lg.height)
            for metric in ['Shots_PG', 'Shot_Accuracy', 'Goals_Per_Shot', 'Corners_PG',
                           'Goals_Per_Corner', 'Fouls_PG', 'Shots_Per_Goal', 'Fouls_Per_Goal']:
                if metric in lg.columns and int(lg.select(pl.col(metric).is_not_null().sum()).to_series()[0]) >= 3:
                    v = lg.select(pl.col(metric).is_not_null()).to_series()
                    sn_v = season_nums[v.to_numpy()]
                    y_v = _np(lg.filter(v), metric)
                    slope, intercept, r, p, se = stats.linregress(sn_v, y_v)
                    if p < 0.1:
                        style_trends[f'{div}_{metric}'] = {
                            'slope_per_season': float(slope), 'p': float(p), 'r': float(r),
                            'significant': bool(p < 0.05),
                        }
    results['style_trends'] = style_trends

    style_features = ['Shots_PG', 'Shot_Accuracy', 'Goals_Per_Shot', 'Corners_PG',
                      'Goals_Per_Corner', 'Fouls_PG', 'Shots_Per_Goal', 'Fouls_Per_Goal']

    style_z = style_metrics.clone()
    for feat in style_features:
        if feat in style_z.columns:
            mean = float(style_z.select(pl.col(feat).mean()).to_series()[0])
            std = float(style_z.select(pl.col(feat).std()).to_series()[0])
            style_z = style_z.with_columns(
                ((pl.col(feat) - mean) / (std if std > 0 else 1.0)).alias(f'{feat}_z')
            )

    z_cols = [f'{f}_z' for f in style_features if f'{f}_z' in style_z.columns]
    league_season_centroids = style_z.group_by(['Div', 'Season']).agg(
        [pl.col(c).mean().alias(c) for c in z_cols]
    ).sort(['Div', 'Season'])

    centroid_mat = league_season_centroids.select(z_cols).to_numpy()
    distances = pdist(centroid_mat, metric='euclidean')
    dist_matrix_np = squareform(distances)
    idx = [tuple(r) for r in league_season_centroids.select(['Div', 'Season']).iter_rows()]
    dist_df = pl.DataFrame(dist_matrix_np).with_columns(pl.Series('idx', idx))
    results['style_distance_matrix'] = dist_df

    # Also provide a square labelled pandas DataFrame for the plotting boundary.
    dist_df_sq = pd.DataFrame(dist_matrix_np, index=idx, columns=idx)
    results['style_distance_matrix_sq'] = dist_df_sq

    n_clusters = min(4, league_season_centroids.height)
    if league_season_centroids.height >= n_clusters:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(centroid_mat)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        style_z = style_z.with_columns(pl.Series('Style_Cluster', labels.astype(float)))
        results['style_clusters'] = style_z.select(['Div', 'Season', 'Style_Cluster']).clone()

    convergence = []
    for season in sorted(style_metrics.select(pl.col('Season').unique()).to_series().to_list()):
        season_data = league_season_centroids.filter(pl.col('Season') == season)
        if season_data.height >= 2:
            sd = season_data.select(z_cols).to_numpy()
            d = pdist(sd, metric='euclidean')
            convergence.append({'Season': season, 'Avg_Distance': float(d.mean()), 'N_Leagues': season_data.height})
    results['convergence'] = pl.DataFrame(convergence) if convergence else pl.DataFrame()

    return results


def analyze_corner_efficiency(team_season: pl.DataFrame, team_matches: pl.DataFrame) -> Dict:
    """
    Question 3: Corner efficiency and set-piece profiles.
    """
    results: Dict = {}

    valid = team_season.filter(pl.col('Corners') > 0).clone()

    corner_corrs: Dict = {}
    for metric in ['Corners_PG', 'Goals_Per_Corner', 'Shots_Per_Corner']:
        v = valid.select(pl.col(metric).is_not_null() & pl.col('PPG').is_not_null()).to_series()
        if int(v.sum()) > 10:
            r, p, n = _pearsonr(_np(valid.filter(v), metric), _np(valid.filter(v), 'PPG'))
            corner_corrs[metric] = {'r': r, 'p': p, 'n': n}
    results['ppg_correlations'] = corner_corrs

    valid_reg = valid.select(['PPG', 'Shots_PG', 'Corners_PG', 'Goals_Per_Corner', 'Shots_Per_Corner']).drop_nulls()
    if valid_reg.height > 50:
        y = valid_reg.select('PPG').to_numpy().ravel()
        X1 = valid_reg.select('Shots_PG').to_numpy()
        r2_1 = LinearRegression().fit(X1, y).score(X1, y)
        X2 = valid_reg.select(['Shots_PG', 'Corners_PG']).to_numpy()
        r2_2 = LinearRegression().fit(X2, y).score(X2, y)
        X3 = valid_reg.select(['Shots_PG', 'Goals_Per_Corner']).to_numpy()
        r2_3 = LinearRegression().fit(X3, y).score(X3, y)
        X4 = valid_reg.select(['Shots_PG', 'Shots_Per_Corner']).to_numpy()
        r2_4 = LinearRegression().fit(X4, y).score(X4, y)
        results['incremental_r2'] = {
            'shots_only': float(r2_1), 'shots_plus_corners_pg': float(r2_2),
            'shots_plus_gpc': float(r2_3), 'shots_plus_spc': float(r2_4),
            'inc_corners_pg': float(r2_2 - r2_1), 'inc_gpc': float(r2_3 - r2_1),
            'inc_spc': float(r2_4 - r2_1),
        }

    gk = TEAM_SEASON_GROUP
    tm_c = team_matches.with_columns(pl.len().over(gk).alias('season_size')).filter(pl.col('season_size') >= 20)
    tm_c = tm_c.sort(['Div', 'Team', 'Season', 'Date'])
    cc = tm_c.select(pl.cum_count('Date').over(gk)).to_series().to_numpy()
    n = tm_c.select(pl.len().over(gk)).to_series().to_numpy()
    mid = n // 2
    first_mask = cc < mid
    first = tm_c.filter(pl.Series(first_mask))
    second = tm_c.filter(pl.Series(~first_mask))

    first_corners = first.group_by(gk).agg(pl.col('Corners').sum().alias('Corners_First'))
    second_corners = second.group_by(gk).agg(pl.col('Corners').sum().alias('Corners_Second'))
    first_shots = first.group_by(gk).agg(pl.col('Shots').sum().alias('Shots_First'))
    second_shots = second.group_by(gk).agg(pl.col('Shots').sum().alias('Shots_Second'))
    first_gf = first.group_by(gk).agg(pl.col('GF').sum().alias('GF_First'))
    second_gf = second.group_by(gk).agg(pl.col('GF').sum().alias('GF_Second'))
    first_n = first.group_by(gk).agg(pl.len().alias('N_First'))
    second_n = second.group_by(gk).agg(pl.len().alias('N_Second'))

    half_corners = first_corners.join(second_corners, on=gk, how='full', coalesce=True).join(
        first_shots, on=gk, how='full', coalesce=True).join(second_shots, on=gk, how='full', coalesce=True).join(
        first_gf, on=gk, how='full', coalesce=True).join(second_gf, on=gk, how='full', coalesce=True).join(
        first_n, on=gk, how='full', coalesce=True).join(second_n, on=gk, how='full', coalesce=True)

    half_corners = half_corners.with_columns([
        _ratio0_expr('Corners_First', 'N_First', 'Corners_PG_First'),
        _ratio0_expr('Corners_Second', 'N_Second', 'Corners_PG_Second'),
        _ratio0_expr('Shots_First', 'Corners_First', 'Shots_Per_Corner_First'),
        _ratio0_expr('Shots_Second', 'Corners_Second', 'Shots_Per_Corner_Second'),
        _ratio0_expr('GF_First', 'Corners_First', 'Goals_Per_Corner_First'),
        _ratio0_expr('GF_Second', 'Corners_Second', 'Goals_Per_Corner_Second'),
    ])

    half_persistence: Dict = {}
    for metric in ['Corners_PG', 'Goals_Per_Corner', 'Shots_Per_Corner']:
        first_col = f'{metric}_First'
        second_col = f'{metric}_Second'
        if first_col in half_corners.columns and second_col in half_corners.columns:
            v = half_corners.select(pl.col(first_col).is_not_null() & pl.col(second_col).is_not_null()).to_series()
            if int(v.sum()) > 10:
                r, p, n = _pearsonr(_np(half_corners.filter(v), first_col), _np(half_corners.filter(v), second_col))
                half_persistence[metric] = {'r': r, 'p': p, 'n': n}
    results['half_season_persistence'] = half_persistence

    team_season_sorted = team_season.sort(['Div', 'Team', 'Season']).clone()
    smap = pl.DataFrame({'Season': list(SEASON_ORDER.keys()),
                         'Season_Num': list(SEASON_ORDER.values())})
    team_season_sorted = team_season_sorted.join(smap, on='Season', how='left')
    for metric in ['Corners_PG', 'Goals_Per_Corner', 'Shots_Per_Corner']:
        team_season_sorted = team_season_sorted.with_columns(
            pl.col(metric).shift(-1).over(['Div', 'Team']).alias(f'Next_{metric}')
        )
    team_season_sorted = team_season_sorted.with_columns([
        pl.col('Season_Num').shift(-1).over(['Div', 'Team']).alias('Next_Season_Num'),
        pl.col('Div').shift(-1).over(['Div', 'Team']).alias('Next_Div'),
    ])
    valid_pairs = (
        pl.col('Next_Season_Num').is_not_null() &
        (pl.col('Next_Season_Num') - pl.col('Season_Num') == 1) &
        (pl.col('Next_Div') == pl.col('Div'))
    )
    consecutive = team_season_sorted.filter(valid_pairs)

    season_persistence: Dict = {}
    for metric in ['Corners_PG', 'Goals_Per_Corner', 'Shots_Per_Corner']:
        v = consecutive.select(pl.col(metric).is_not_null() & pl.col(f'Next_{metric}').is_not_null()).to_series()
        if int(v.sum()) > 10:
            r, p, n = _pearsonr(_np(consecutive.filter(v), metric), _np(consecutive.filter(v), f'Next_{metric}'))
            season_persistence[metric] = {'r': r, 'p': p, 'n': n}
    results['season_to_season_persistence'] = season_persistence

    min_corners = 50
    specialists = valid.filter(pl.col('Corners') >= min_corners).clone()
    results['top_gpc'] = specialists.sort('Goals_Per_Corner', descending=True).head(15).to_dicts()
    results['bottom_gpc'] = specialists.sort('Goals_Per_Corner').head(15).to_dicts()

    profile_features = ['Corners_PG', 'Goals_Per_Corner', 'Shots_Per_Corner']
    profile_data = specialists.select(profile_features).drop_nulls()
    if profile_data.height >= 5:
        scaler = StandardScaler()
        profile_scaled = scaler.fit_transform(profile_data.to_numpy())
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        labels = kmeans.fit_predict(profile_scaled)
        specialists = specialists.with_row_index('__idx')
        lbl = pl.DataFrame({'__idx': list(range(profile_data.height)), 'SP_Cluster': labels.astype(float)})
        specialists = specialists.join(lbl, on='__idx', how='left').drop('__idx')
        cluster_summary = specialists.group_by('SP_Cluster').agg([
            pl.len().alias('N'),
            pl.col('Corners_PG').mean().round(3),
            pl.col('Goals_Per_Corner').mean().round(3),
            pl.col('Shots_Per_Corner').mean().round(3),
            pl.col('PPG').mean().round(3),
        ])
        results['sp_clusters'] = cluster_summary.to_dicts()

    return results


def analyze_shooting_profiles(team_season: pl.DataFrame, team_matches: pl.DataFrame) -> Dict:
    """
    Question 4: Selective vs Volume shooting profiles.
    """
    results: Dict = {}

    ts = team_season.clone()
    ts = ts.with_columns((pl.col('Div') + pl.lit('_') + pl.col('Season')).alias('LeagueSeason'))

    for metric in ['Shots_PG', 'Shot_Accuracy']:
        if metric in ts.columns:
            ls_agg = ts.group_by('LeagueSeason').agg([
                pl.col(metric).mean().alias('ls_mean'),
                pl.col(metric).std().alias('ls_std'),
            ])
            ts = ts.join(ls_agg, on='LeagueSeason', how='left')
            ts = ts.with_columns(((pl.col(metric) - pl.col('ls_mean')) / pl.col('ls_std').replace(0, None)).alias(f'{metric}_z'))
            ts = ts.drop(['ls_mean', 'ls_std'])

    if 'Shots_PG_z' in ts.columns and 'Shot_Accuracy_z' in ts.columns:
        ts = ts.with_columns([
            pl.when(pl.col('Shots_PG_z') >= 0).then(pl.lit('High_Vol')).otherwise(pl.lit('Low_Vol')).alias('Vol_Profile'),
            pl.when(pl.col('Shot_Accuracy_z') >= 0).then(pl.lit('High_Acc')).otherwise(pl.lit('Low_Acc')).alias('Acc_Profile'),
        ])
        ts = ts.with_columns((pl.col('Vol_Profile') + pl.lit('_') + pl.col('Acc_Profile')).alias('Shooting_Profile_z'))

    shots_median = float(ts.select(pl.col('Shots_PG').median()).to_series()[0]) if 'Shots_PG' in ts.columns else 0.0
    acc_median = float(ts.select(pl.col('Shot_Accuracy').median()).to_series()[0]) if 'Shot_Accuracy' in ts.columns else 0.0
    ts = ts.with_columns([
        pl.when(pl.col('Shots_PG') >= shots_median).then(pl.lit('High_Vol')).otherwise(pl.lit('Low_Vol')).alias('Vol_Profile_Raw'),
        pl.when(pl.col('Shot_Accuracy') >= acc_median).then(pl.lit('High_Acc')).otherwise(pl.lit('Low_Acc')).alias('Acc_Profile_Raw'),
    ])
    ts = ts.with_columns((pl.col('Vol_Profile_Raw') + pl.lit('_') + pl.col('Acc_Profile_Raw')).alias('Shooting_Profile_Raw'))

    if 'Shooting_Profile_z' in ts.columns:
        profile_ppg_z = ts.group_by('Shooting_Profile_z').agg([
            pl.len().alias('N'),
            pl.col('PPG').mean().round(3),
            pl.col('GF_PG').mean().round(3),
            pl.col('Shots_PG').mean().round(3),
            pl.col('Shot_Accuracy').mean().round(3),
            pl.col('Goals_Per_Shot').mean().round(3),
            pl.col('Goals_Per_Shot_OT').mean().round(3),
        ]).sort('Shooting_Profile_z')
        results['profile_ppg_z'] = profile_ppg_z

    if 'Shooting_Profile_Raw' in ts.columns:
        profile_ppg_raw = ts.group_by('Shooting_Profile_Raw').agg([
            pl.len().alias('N'),
            pl.col('PPG').mean().round(3),
            pl.col('GF_PG').mean().round(3),
            pl.col('Shots_PG').mean().round(3),
            pl.col('Shot_Accuracy').mean().round(3),
            pl.col('Goals_Per_Shot').mean().round(3),
            pl.col('Goals_Per_Shot_OT').mean().round(3),
        ]).sort('Shooting_Profile_Raw')
        results['profile_ppg_raw'] = profile_ppg_raw

    ts_sorted = ts.sort(['Div', 'Team', 'Season']).clone()
    smap = pl.DataFrame({'Season': list(SEASON_ORDER.keys()),
                         'Season_Num': list(SEASON_ORDER.values())})
    ts_sorted = ts_sorted.join(smap, on='Season', how='left')
    if 'Shooting_Profile_z' in ts_sorted.columns:
        ts_sorted = ts_sorted.with_columns([
            pl.col('Shooting_Profile_z').shift(1).over(['Div', 'Team']).alias('Prev_Profile_z'),
            pl.col('Season_Num').diff().over(['Div', 'Team']).alias('season_diff'),
            pl.col('Div').shift(1).over(['Div', 'Team']).alias('Prev_Div'),
        ]).with_columns(
            (pl.col('season_diff') == 1).alias('Consecutive')
        ).with_columns(
            (pl.col('Consecutive') & (pl.col('Div') == pl.col('Prev_Div'))).alias('Valid_Transition')
        )
    if 'Shooting_Profile_Raw' in ts_sorted.columns:
        ts_sorted = ts_sorted.with_columns(
            pl.col('Shooting_Profile_Raw').shift(1).over(['Div', 'Team']).alias('Prev_Profile_Raw')
        )

    transitions_z = ts_sorted.filter(pl.col('Valid_Transition')) if 'Valid_Transition' in ts_sorted.columns else ts_sorted
    if 'Shooting_Profile_z' in ts_sorted.columns and 'Prev_Profile_z' in ts_sorted.columns:
        tmz = transitions_z.group_by(['Prev_Profile_z', 'Shooting_Profile_z']).agg(pl.len().alias('n'))
        # row-normalise to a crosstab-like matrix
        row_tot = tmz.group_by('Prev_Profile_z').agg(pl.col('n').sum().alias('row_tot'))
        tmz = tmz.join(row_tot, on='Prev_Profile_z', how='left').with_columns((pl.col('n') / pl.col('row_tot')).round(3))
        trans_matrix_z = tmz.pivot(on='Shooting_Profile_z', index='Prev_Profile_z', values='n')
        results['transition_matrix_z'] = trans_matrix_z
        stability_z = float(transitions_z.select(
            (pl.col('Shooting_Profile_z') == pl.col('Prev_Profile_z')).mean()
        ).to_series()[0]) if transitions_z.height else float('nan')
    else:
        stability_z = float('nan')

    if 'Shooting_Profile_Raw' in ts_sorted.columns and 'Prev_Profile_Raw' in ts_sorted.columns:
        transitions_raw = ts_sorted.filter(pl.col('Valid_Transition')) if 'Valid_Transition' in ts_sorted.columns else ts_sorted
        tmr = transitions_raw.group_by(['Prev_Profile_Raw', 'Shooting_Profile_Raw']).agg(pl.len().alias('n'))
        row_tot_r = tmr.group_by('Prev_Profile_Raw').agg(pl.col('n').sum().alias('row_tot'))
        tmr = tmr.join(row_tot_r, on='Prev_Profile_Raw', how='left').with_columns((pl.col('n') / pl.col('row_tot')).round(3))
        trans_matrix_raw = tmr.pivot(on='Shooting_Profile_Raw', index='Prev_Profile_Raw', values='n')
        results['transition_matrix_raw'] = trans_matrix_raw
        stability_raw = float(transitions_raw.select(
            (pl.col('Shooting_Profile_Raw') == pl.col('Prev_Profile_Raw')).mean()
        ).to_series()[0]) if transitions_raw.height else float('nan')
    else:
        stability_raw = float('nan')

    results['profile_stability'] = {
        'z_score_adjusted': stability_z,
        'raw': stability_raw,
        'n_transitions': transitions_z.height if 'Valid_Transition' in ts_sorted.columns else 0,
    }

    half_persistence: Dict = {}
    gk = TEAM_SEASON_GROUP
    tm_s = team_matches.with_columns(pl.len().over(gk).alias('season_size')).filter(pl.col('season_size') >= 20)
    tm_s = tm_s.sort(['Div', 'Team', 'Season', 'Date'])
    cc = tm_s.select(pl.cum_count('Date').over(gk)).to_series().to_numpy()
    n = tm_s.select(pl.len().over(gk)).to_series().to_numpy()
    mid = n // 2
    first_mask = cc < mid
    first = tm_s.filter(pl.Series(first_mask))
    second = tm_s.filter(pl.Series(~first_mask))

    first_shots = first.group_by(gk).agg(pl.col('Shots').sum().alias('Shots_First'))
    second_shots = second.group_by(gk).agg(pl.col('Shots').sum().alias('Shots_Second'))
    first_shots_ot = first.group_by(gk).agg(pl.col('Shots_OT').sum().alias('Shots_OT_First'))
    second_shots_ot = second.group_by(gk).agg(pl.col('Shots_OT').sum().alias('Shots_OT_Second'))
    first_gf = first.group_by(gk).agg(pl.col('GF').sum().alias('GF_First'))
    second_gf = second.group_by(gk).agg(pl.col('GF').sum().alias('GF_Second'))
    first_n = first.group_by(gk).agg(pl.len().alias('N_First'))
    second_n = second.group_by(gk).agg(pl.len().alias('N_Second'))

    half_shooting = first_shots.join(second_shots, on=gk, how='full', coalesce=True).join(
        first_shots_ot, on=gk, how='full', coalesce=True).join(second_shots_ot, on=gk, how='full', coalesce=True).join(
        first_gf, on=gk, how='full', coalesce=True).join(second_gf, on=gk, how='full', coalesce=True).join(
        first_n, on=gk, how='full', coalesce=True).join(second_n, on=gk, how='full', coalesce=True)

    half_shooting = half_shooting.with_columns([
        _ratio0_expr('Shots_First', 'N_First', 'Shots_PG_First'),
        _ratio0_expr('Shots_Second', 'N_Second', 'Shots_PG_Second'),
        _ratio0_expr('Shots_OT_First', 'Shots_First', 'Shot_Accuracy_First'),
        _ratio0_expr('Shots_OT_Second', 'Shots_Second', 'Shot_Accuracy_Second'),
        _ratio0_expr('GF_First', 'Shots_First', 'Goals_Per_Shot_First'),
        _ratio0_expr('GF_Second', 'Shots_Second', 'Goals_Per_Shot_Second'),
        _ratio0_expr('GF_First', 'Shots_OT_First', 'Goals_Per_Shot_OT_First'),
        _ratio0_expr('GF_Second', 'Shots_OT_Second', 'Goals_Per_Shot_OT_Second'),
    ])

    for metric in ['Shots_PG', 'Shot_Accuracy', 'Goals_Per_Shot', 'Goals_Per_Shot_OT']:
        first_col = f'{metric}_First'
        second_col = f'{metric}_Second'
        if first_col in half_shooting.columns and second_col in half_shooting.columns:
            v = half_shooting.select(pl.col(first_col).is_not_null() & pl.col(second_col).is_not_null()).to_series()
            if int(v.sum()) > 10:
                r, p, n = _pearsonr(_np(half_shooting.filter(v), first_col), _np(half_shooting.filter(v), second_col))
                half_persistence[metric] = {'r': r, 'p': p, 'n': n}
    results['half_season_persistence'] = half_persistence

    profile_persistence: Dict = {}
    if 'Shooting_Profile_z' in ts_sorted.columns and ts_sorted.height > 20:
        for profile in ts_sorted.select(pl.col('Shooting_Profile_z').unique()).to_series().to_list():
            profile_teams = ts_sorted.filter(pl.col('Shooting_Profile_z') == profile)
            vt = profile_teams.filter(pl.col('Valid_Transition')) if 'Valid_Transition' in profile_teams.columns else profile_teams
            if vt.height > 10:
                same = float(vt.select((pl.col('Shooting_Profile_z') == pl.col('Prev_Profile_z')).mean()).to_series()[0])
                profile_persistence[profile] = {'stability': same, 'n': int(vt.height)}
    results['profile_specific_persistence'] = profile_persistence

    return results


def analyze_clutch_performance(master: pl.DataFrame, team_matches: pl.DataFrame) -> Dict:
    """
    Question 5: Points from winning/losing positions - "Clutch" performance.
    """
    results: Dict = {}

    home_side = master.with_columns([
        pl.col('HomeTeam').alias('Team'), pl.lit('Home').alias('Venue'),
        (pl.col('HTHG') - pl.col('HTAG')).alias('HT_GD'),
        (pl.col('FTHG') - pl.col('FTAG')).alias('FT_GD'),
        pl.col('HTR').replace_strict({'H': 'W', 'D': 'D', 'A': 'L'}).alias('HT_Result'),
        pl.col('FTR').replace_strict({'H': 'W', 'D': 'D', 'A': 'L'}).alias('FT_Result'),
    ])
    away_side = master.with_columns([
        pl.col('AwayTeam').alias('Team'), pl.lit('Away').alias('Venue'),
        (pl.col('HTAG') - pl.col('HTHG')).alias('HT_GD'),
        (pl.col('FTAG') - pl.col('FTHG')).alias('FT_GD'),
        pl.col('HTR').replace_strict({'A': 'W', 'D': 'D', 'H': 'L'}).alias('HT_Result'),
        pl.col('FTR').replace_strict({'A': 'W', 'D': 'D', 'H': 'L'}).alias('FT_Result'),
    ])
    ht_data = pl.concat([home_side.select(['Div', 'Season', 'Date', 'Team', 'Venue', 'HT_GD', 'FT_GD', 'HT_Result', 'FT_Result']),
                         away_side.select(['Div', 'Season', 'Date', 'Team', 'Venue', 'HT_GD', 'FT_GD', 'HT_Result', 'FT_Result'])],
                        how='diagonal')

    result_map = {'W': 3, 'D': 1, 'L': 0}
    ht_data = ht_data.with_columns([
        pl.col('HT_Result').replace_strict(result_map, default=None).cast(pl.Int64).alias('HT_Points'),
        pl.col('FT_Result').replace_strict(result_map, default=None).cast(pl.Int64).alias('FT_Points'),
    ])

    leading = ht_data.filter(pl.col('HT_Result') == 'W')
    drawing = ht_data.filter(pl.col('HT_Result') == 'D')
    trailing = ht_data.filter(pl.col('HT_Result') == 'L')

    def _rate(df: pl.DataFrame) -> Dict:
        if df.height == 0:
            return {'n': 0, 'win_pct': float('nan'), 'draw_pct': float('nan'),
                    'loss_pct': float('nan'), 'avg_ft_points': float('nan')}
        return {
            'n': df.height,
            'win_pct': float((df.select(pl.col('FT_Result') == 'W').to_series().mean())),
            'draw_pct': float((df.select(pl.col('FT_Result') == 'D').to_series().mean())),
            'loss_pct': float((df.select(pl.col('FT_Result') == 'L').to_series().mean())),
            'avg_ft_points': float(df.select(pl.col('FT_Points').mean()).to_series()[0]),
        }

    results['overall_rates'] = {
        'leading': _rate(leading) | {'avg_points_dropped': float((3 - leading.select(pl.col('FT_Points')).to_series().mean())) if leading.height else float('nan')},
        'drawing': _rate(drawing),
        'trailing': _rate(trailing) | {'avg_points_gained': float(trailing.select(pl.col('FT_Points')).to_series().mean()) if trailing.height else float('nan')},
    }

    team_ht = ht_data.group_by(['Div', 'Team', 'Season', 'HT_Result']).agg([
        pl.len().alias('Matches'),
        pl.col('HT_Points').sum().alias('HT_Points'),
        pl.col('FT_Points').sum().alias('FT_Points'),
    ])
    team_ht_pivot = team_ht.pivot(on='HT_Result', index=['Div', 'Team', 'Season'],
                                  values=['Matches', 'HT_Points', 'FT_Points'],
                                  aggregate_function='sum')

    for col in ['Matches_W', 'Matches_D', 'Matches_L', 'FT_Points_W', 'FT_Points_D', 'FT_Points_L']:
        if col not in team_ht_pivot.columns:
            team_ht_pivot = team_ht_pivot.with_columns(pl.lit(0).alias(col))

    team_ht_pivot = team_ht_pivot.with_columns([
        pl.sum_horizontal(['Matches_W', 'Matches_D', 'Matches_L']).alias('Total_Matches'),
        pl.sum_horizontal(['FT_Points_W', 'FT_Points_D', 'FT_Points_L']).alias('Total_FT_Points'),
    ]).with_columns([
        (pl.col('Total_FT_Points') / pl.col('Total_Matches')).alias('PPG'),
        ((3 * pl.col('Matches_W') - pl.col('FT_Points_W')) / pl.col('Total_Matches')).alias('Points_Dropped_When_Leading'),
        (pl.col('FT_Points_L') / pl.col('Total_Matches')).alias('Points_Gained_When_Trailing'),
        (pl.col('FT_Points_D') / pl.col('Total_Matches')).alias('Points_From_Drawing'),
    ]).with_columns(
        (pl.col('Points_Gained_When_Trailing') - pl.col('Points_Dropped_When_Leading')).alias('Clutch_Score')
    )

    valid_seasons = team_ht_pivot.filter(pl.col('Total_Matches') >= 20)
    results['team_season_clutch'] = valid_seasons

    clutch_corrs: Dict = {}
    for metric in ['Points_Dropped_When_Leading', 'Points_Gained_When_Trailing', 'Clutch_Score', 'Points_From_Drawing']:
        v = valid_seasons.select(pl.col(metric).is_not_null() & pl.col('PPG').is_not_null()).to_series()
        if int(v.sum()) > 10:
            r, p, n = _pearsonr(_np(valid_seasons.filter(v), metric), _np(valid_seasons.filter(v), 'PPG'))
            clutch_corrs[metric] = {'r': r, 'p': p, 'n': n}
    results['ppg_correlations'] = clutch_corrs

    team_clutch = valid_seasons.group_by('Team').agg([
        pl.len().alias('Seasons'),
        pl.col('PPG').mean().alias('Avg_PPG'),
        pl.col('Points_Dropped_When_Leading').mean().alias('Avg_Points_Dropped_When_Leading'),
        pl.col('Points_Gained_When_Trailing').mean().alias('Avg_Points_Gained_When_Trailing'),
        pl.col('Clutch_Score').mean().alias('Avg_Clutch_Score'),
    ]).filter(pl.col('Seasons') >= 3).sort('Avg_Clutch_Score', descending=True)
    results['team_clutch'] = team_clutch

    ht_data = ht_data.with_columns([
        (pl.col('Date').cum_count().over(TEAM_SEASON_GROUP) + 1).alias('Match_Num')
    ])
    max_match = ht_data.select(pl.col('Match_Num').max().over(TEAM_SEASON_GROUP)).to_series()
    ht_data = ht_data.with_columns(pl.Series('max_match', max_match))
    ht_data = ht_data.with_columns(
        pl.when(pl.col('Match_Num') <= pl.col('max_match') / 2).then(pl.lit('Early')).otherwise(pl.lit('Late')).alias('Season_Half')
    )

    half_clutch = ht_data.group_by(['Div', 'Team', 'Season', 'Season_Half', 'HT_Result']).agg([
        pl.len().alias('Matches'),
        pl.col('FT_Points').sum().alias('FT_Points'),
    ])
    half_pivot = half_clutch.pivot(on='HT_Result', index=['Div', 'Team', 'Season', 'Season_Half'],
                                   values=['Matches', 'FT_Points'], aggregate_function='sum')
    for col in ['Matches_W', 'Matches_D', 'Matches_L', 'FT_Points_W', 'FT_Points_D', 'FT_Points_L']:
        if col not in half_pivot.columns:
            half_pivot = half_pivot.with_columns(pl.lit(0).alias(col))
    half_pivot = half_pivot.with_columns([
        pl.sum_horizontal(['Matches_W', 'Matches_D', 'Matches_L']).alias('Total_Matches'),
    ]).with_columns([
        ((3 * pl.col('Matches_W') - pl.col('FT_Points_W')) / pl.col('Total_Matches')).alias('Points_Dropped_When_Leading'),
        (pl.col('FT_Points_L') / pl.col('Total_Matches')).alias('Points_Gained_When_Trailing'),
    ])

    early = half_pivot.filter(pl.col('Season_Half') == 'Early').select([
        'Div', 'Team', 'Season',
        pl.col('Points_Dropped_When_Leading').alias('Drop_Early'),
        pl.col('Points_Gained_When_Trailing').alias('Gain_Early'),
    ])
    late = half_pivot.filter(pl.col('Season_Half') == 'Late').select([
        'Div', 'Team', 'Season',
        pl.col('Points_Dropped_When_Leading').alias('Drop_Late'),
        pl.col('Points_Gained_When_Trailing').alias('Gain_Late'),
    ])
    early_late = early.join(late, on=['Div', 'Team', 'Season'], how='inner')
    early_late = early_late.filter(pl.col('Drop_Early').is_not_null() & pl.col('Gain_Early').is_not_null() &
                                 pl.col('Drop_Late').is_not_null() & pl.col('Gain_Late').is_not_null())

    persistence: Dict = {}
    if early_late.height > 20:
        r_drop, p_drop, _ = _pearsonr(_np(early_late, 'Drop_Early'), _np(early_late, 'Drop_Late'))
        r_gain, p_gain, _ = _pearsonr(_np(early_late, 'Gain_Early'), _np(early_late, 'Gain_Late'))
        persistence = {
            'points_dropped_when_leading': {'r': float(r_drop), 'p': float(p_drop), 'n': int(early_late.height)},
            'points_gained_when_trailing': {'r': float(r_gain), 'p': float(p_gain), 'n': int(early_late.height)},
        }
    results['early_late_persistence'] = persistence
    results['early_late_data'] = early_late

    venue_clutch = ht_data.group_by(['Venue', 'HT_Result']).agg([
        pl.len().alias('N'),
        pl.col('FT_Points').mean().alias('Avg_FT_Points'),
    ]).sort(['Venue', 'HT_Result'])
    results['by_venue'] = venue_clutch

    trailing_detail = trailing.with_columns([
        (-pl.col('HT_GD')).alias('Deficit_Size'),
        pl.when(-pl.col('HT_GD') == 1).then(pl.lit('One_Goal')).otherwise(pl.lit('Two_Plus')).alias('Deficit_Cat'),
    ])
    deficit_recovery = trailing_detail.group_by(['Deficit_Cat', 'Venue']).agg([
        pl.len().alias('N'),
        ((pl.col('FT_Result') == 'W').mean()).alias('Win_Rate'),
        ((pl.col('FT_Result') == 'D').mean()).alias('Draw_Rate'),
        pl.col('FT_Points').mean().alias('Avg_Points'),
    ]).sort(['Deficit_Cat', 'Venue'])
    results['by_deficit_size'] = deficit_recovery

    return results


def run_next_insights(master: pl.DataFrame, team_matches: pl.DataFrame, team_season: pl.DataFrame) -> Dict:
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
