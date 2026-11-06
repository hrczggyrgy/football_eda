"""
New insights module.
Implements 6 priority research questions beyond the existing analyses:
1. Opponent-dependent performance
2. Components of home advantage (shot generation vs suppression vs conversion)
3. Close-match dependence (one-goal results)
4. Rest asymmetry (observed league-fixture rest)
5. Performance indicators leading results (lagged shot dominance)
6. Adaptation after promotion (reliable adjacent-tier transitions only)

Migrated to Polars: Polars is the primary analytical data structure. NumPy and
SciPy/scikit-learn are used only where they provide the appropriate statistical
functionality. No pandas.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from scipy import stats
from scipy.stats import binomtest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

from features.engineer import add_lagged_team_strength, build_promotion_relegation_transitions, SEASON_ORDER

TEAM_SEASON_GROUP = ['Div', 'Team', 'Season']

DIVISION_HIERARCHY = {
    'E0': 0, 'E1': 1, 'E2': 2, 'E3': 3,
    'SP1': 0, 'SP2': 1,
    'I1': 0, 'I2': 1,
    'D1': 0, 'D2': 1,
    'F1': 0, 'F2': 1,
}


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


def _safe_ratio(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    return pl.when(den > 0).then(num / den).otherwise(None)


def analyze_opponent_dependent_performance(team_matches: pl.DataFrame,
                                           team_season: pl.DataFrame,
                                           min_matches: int = 20) -> Dict:
    """
    Question 1: Which teams perform relatively better against stronger opponents?

    Uses a SHARED BASELINE model with team-specific opponent-strength interactions.
    Opponent strength defined using only pre-match information (lagged cumulative PPG).
    """
    results: Dict = {}

    tm = add_lagged_team_strength(team_matches)

    valid_seasons = team_season.filter(pl.col('Matches') >= min_matches).select(
        ['Div', 'Team', 'Season']
    )
    tm = tm.join(valid_seasons, on=['Div', 'Team', 'Season'], how='inner')

    # Opponent's pre-match PPG lookup
    opp_strength = tm.select([
        pl.col('Div'), pl.col('Season'), pl.col('Date'),
        pl.col('Team').alias('Opponent'),
        pl.col('PreMatch_PPG_Shrunk').alias('Opp_PreMatch_PPG'),
    ])
    tm = tm.join(opp_strength, on=['Div', 'Season', 'Date', 'Opponent'], how='left')

    tm = tm.with_columns([
        (pl.col('Div') + pl.lit('_') + pl.col('Season')).alias('LeagueSeason'),
        pl.when(pl.col('Venue') == 'Home').then(1).otherwise(0).cast(pl.Int8).alias('Venue_Code'),
    ])

    # Stage 1: Fixed effects baseline
    valid = tm.select(
        pl.col('PreMatch_PPG_Shrunk').is_not_null() &
        pl.col('Opp_PreMatch_PPG').is_not_null() &
        pl.col('Points').is_not_null()
    ).to_series()
    tm_valid = tm.filter(valid)

    if tm_valid.height < 100:
        results['error'] = 'Insufficient data for shared baseline model'
        return results

    dummies_df = tm_valid.select('LeagueSeason').to_dummies(columns='LeagueSeason', drop_first=True)
    X_base = pl.concat([
        tm_valid.select(['PreMatch_PPG_Shrunk', 'Opp_PreMatch_PPG', 'Venue_Code']),
        dummies_df,
    ], how='horizontal')
    y = tm_valid.select('PPG' if 'PPG' in tm_valid.columns else 'Points').to_series()

    # Points column: team_matches has Points already (from add_efficiency_ratios)
    y = tm_valid.select('Points').to_series().cast(pl.Float64).to_numpy()
    X = X_base.to_numpy()
    reg_base = LinearRegression(fit_intercept=True).fit(X, y)
    expected = reg_base.predict(X)
    tm_valid = tm_valid.with_columns(pl.Series('Expected_Points', expected))
    tm_valid = tm_valid.with_columns((pl.col('Points') - pl.col('Expected_Points')).alias('Residual'))

    # Stage 2: per team-season residual on opponent strength
    def _group_linregress(frame: pl.DataFrame, xcol: str, ycol: str,
                          min_n: int, min_unique: int) -> pl.DataFrame:
        tmp = frame.select(['Div', 'Team', 'Season', xcol, ycol]).with_columns([
            (pl.col(xcol) ** 2).alias('_x2'),
            (pl.col(ycol) ** 2).alias('_y2'),
            (pl.col(xcol) * pl.col(ycol)).alias('_xy'),
        ])
        agg = tmp.group_by(['Div', 'Team', 'Season']).agg([
            pl.len().alias('n'),
            pl.col(xcol).sum().alias('xsum'),
            pl.col(ycol).sum().alias('ysum'),
            pl.col('_x2').sum().alias('x2sum'),
            pl.col('_y2').sum().alias('y2sum'),
            pl.col('_xy').sum().alias('xysum'),
            pl.col(xcol).n_unique().alias('nunique'),
        ])
        agg = agg.filter((pl.col('n') >= min_n) & (pl.col('nunique') > min_unique))
        if agg.height == 0:
            return pl.DataFrame()
        nv = agg.select('n').to_series().to_numpy().astype(float)
        xbar = agg.select('xsum').to_series().to_numpy().astype(float) / nv
        ybar = agg.select('ysum').to_series().to_numpy().astype(float) / nv
        Sxx = agg.select('x2sum').to_series().to_numpy().astype(float) - nv * xbar * xbar
        Sxy = agg.select('xysum').to_series().to_numpy().astype(float) - nv * xbar * ybar
        Syy = agg.select('y2sum').to_series().to_numpy().astype(float) - nv * ybar * ybar
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
        out = pl.DataFrame({
            'Div': agg.select('Div').to_series(),
            'Team': agg.select('Team').to_series(),
            'Season': agg.select('Season').to_series(),
            'Matches': agg.select('n').to_series().cast(pl.Int64),
            'Slope': slope,
            'R': r,
            'P': p,
            'SE': se,
        })
        return out

    ts_int = _group_linregress(tm_valid, 'Opp_PreMatch_PPG', 'Residual', min_n=10, min_unique=3)
    if ts_int.height > 0:
        shrinkage = np.minimum(1.0, np.sqrt(ts_int.select('Matches').to_series().to_numpy().astype(float)) / 10.0)
        ts_int = ts_int.with_columns([
            pl.Series('Opp_Strength_Slope_Raw', ts_int.select('Slope').to_series()),
            pl.Series('Opp_Strength_Slope_Shrunk', ts_int.select('Slope').to_series() * shrinkage),
            pl.Series('Opp_Strength_SE', ts_int.select('SE').to_series()),
            pl.Series('Opp_Strength_R', ts_int.select('R').to_series()),
            pl.Series('Opp_Strength_P', ts_int.select('P').to_series()),
        ])
        # Own/Avg opponent prePPG per team-season
        own_ppg = tm_valid.group_by(['Div', 'Team', 'Season']).agg(
            pl.col('PreMatch_PPG_Shrunk').mean().alias('Own_PrePPG')
        )
        avg_opp = tm_valid.group_by(['Div', 'Team', 'Season']).agg(
            pl.col('Opp_PreMatch_PPG').mean().alias('Avg_Opp_PrePPG')
        )
        ts_int = ts_int.join(own_ppg, on=['Div', 'Team', 'Season'], how='left').join(
            avg_opp, on=['Div', 'Team', 'Season'], how='left'
        )
        ts_int = ts_int.select([
            'Div', 'Team', 'Season', 'Matches',
            'Opp_Strength_Slope_Raw', 'Opp_Strength_Slope_Shrunk',
            'Opp_Strength_SE', 'Opp_Strength_R', 'Opp_Strength_P',
            'Own_PrePPG', 'Avg_Opp_PrePPG',
        ])
    results['team_season_interactions'] = ts_int

    # Aggregate to team level (3+ seasons)
    if ts_int.height > 0:
        team_interactions = ts_int.group_by('Team').agg([
            pl.len().alias('Seasons'),
            pl.col('Opp_Strength_Slope_Raw').mean().alias('Avg_Slope_Raw'),
            pl.col('Opp_Strength_Slope_Shrunk').mean().alias('Avg_Slope_Shrunk'),
            pl.col('Matches').mean().alias('Avg_Matches'),
        ]).filter(pl.col('Seasons') >= 3).sort('Avg_Slope_Shrunk')
        results['team_interactions'] = team_interactions

    # Persistence: early vs late season interaction (non-overlapping)
    tm_valid = tm_valid.with_columns([
        (pl.col('Date').cum_count().over(['Div', 'Team', 'Season']) + 1).alias('Match_Num_Season')
    ])
    max_match = tm_valid.select(
        pl.col('Match_Num_Season').max().over(['Div', 'Team', 'Season'])
    ).to_series()
    tm_valid = tm_valid.with_columns(pl.Series('max_match', max_match))
    tm_valid = tm_valid.with_columns(
        pl.when(pl.col('Match_Num_Season') <= pl.col('max_match') / 2)
        .then(pl.lit('Early')).otherwise(pl.lit('Late')).alias('Season_Half')
    )

    persistence_data = []
    gb = tm_valid.group_by(['Div', 'Team', 'Season'])
    for key in gb:
        (div, team, season), g = key
        for half in ['Early', 'Late']:
            gh = g.filter(pl.col('Season_Half') == half)
            valid_half = gh.select(
                pl.col('PreMatch_PPG_Shrunk').is_not_null() &
                pl.col('Opp_PreMatch_PPG').is_not_null() &
                pl.col('Points').is_not_null()
            ).to_series()
            if int(valid_half.sum()) < 8:
                continue
            gv = gh.filter(valid_half)
            X = np.column_stack([
                np.ones(gv.height),
                _np(gv, 'PreMatch_PPG_Shrunk'),
                _np(gv, 'Opp_PreMatch_PPG'),
                _np(gv, 'Venue_Code'),
            ])
            y_h = _np(gv, 'Points')
            try:
                beta, _, _, _ = np.linalg.lstsq(X, y_h, rcond=None)
                resid = y_h - X @ beta
            except np.linalg.LinAlgError:
                continue
            slope, r_val, p_val, _, _ = stats.linregress(_np(gv, 'Opp_PreMatch_PPG'), resid)
            persistence_data.append({
                'Div': div, 'Team': team, 'Season': season, 'Half': half,
                'Slope': float(slope), 'R': float(r_val), 'P': float(p_val), 'N': int(gv.height),
            })

    results['half_season_interactions'] = pl.DataFrame(persistence_data) if persistence_data else pl.DataFrame()

    # Cross-half persistence
    hsi = results['half_season_interactions']
    if hsi.height > 0:
        half_pivot = hsi.pivot(on='Half', index=['Div', 'Team', 'Season'], values='Slope')
        if 'Early' in half_pivot.columns and 'Late' in half_pivot.columns:
            v = half_pivot.select(pl.col('Early').is_not_null() & pl.col('Late').is_not_null()).to_series()
            if int(v.sum()) > 10:
                x = _np(half_pivot.filter(v), 'Early')
                y = _np(half_pivot.filter(v), 'Late')
                r, p, n = _pearsonr(x, y)
                results['half_persistence'] = {'r': r, 'p': p, 'n': n}

    # Compare with shot differential and goal differential
    sd_int = _group_linregress(tm_valid, 'Opp_PreMatch_PPG', 'Shot_Dominance', min_n=8, min_unique=3)
    gd_int = _group_linregress(tm_valid, 'Opp_PreMatch_PPG', 'GD', min_n=8, min_unique=3)
    shot_rows = []
    if sd_int.height > 0 and gd_int.height > 0:
        sd_int = sd_int.with_columns([
            pl.col('Slope').alias('sd_slope'), pl.col('R').alias('sd_r'), pl.col('P').alias('sd_p')
        ])
        gd_int = gd_int.with_columns([
            pl.col('Slope').alias('gd_slope'), pl.col('R').alias('gd_r'), pl.col('P').alias('gd_p')
        ])
        joined = sd_int.join(gd_int, on=['Div', 'Team', 'Season'], how='inner', suffix='_gd')
        shot_rows = joined.select([
            'Div', 'Team', 'Season',
            pl.col('sd_slope').alias('ShotDom_Slope'),
            pl.col('sd_r').alias('ShotDom_R'),
            pl.col('sd_p').alias('ShotDom_P'),
            pl.col('gd_slope').alias('GD_Slope'),
            pl.col('gd_r').alias('GD_R'),
            pl.col('gd_p').alias('GD_P'),
        ])
    results['shot_gd_interactions'] = shot_rows if isinstance(shot_rows, pl.DataFrame) else (
        pl.DataFrame(shot_rows) if shot_rows else pl.DataFrame()
    )

    return results


def analyze_home_advantage_components(master: pl.DataFrame, team_matches: pl.DataFrame,
                                      team_season: pl.DataFrame) -> Dict:
    """
    Question 2: Is a team's home advantage associated more with shot generation,
    shot suppression, or conversion?
    """
    results: Dict = {}

    home_away = team_matches.group_by(['Div', 'Team', 'Season', 'Venue']).agg([
        pl.len().alias('Matches'),
        pl.col('GF').sum().alias('GF'), pl.col('GA').sum().alias('GA'),
        pl.col('Shots').sum().alias('Shots'), pl.col('Shots_OT').sum().alias('Shots_OT'),
        pl.col('Shots_Against').sum().alias('Shots_Against'),
        pl.col('Shots_OT_Against').sum().alias('Shots_OT_Against'),
        pl.col('Points').sum().alias('Points'),
    ])
    home = home_away.filter(pl.col('Venue') == 'Home').drop('Venue').rename({
        'Matches': 'Home_Matches', 'GF': 'Home_GF', 'GA': 'Home_GA',
        'Shots': 'Home_Shots', 'Shots_OT': 'Home_Shots_OT',
        'Shots_Against': 'Home_Shots_Against', 'Shots_OT_Against': 'Home_Shots_OT_Against',
        'Points': 'Home_Points',
    })
    away = home_away.filter(pl.col('Venue') == 'Away').drop('Venue').rename({
        'Matches': 'Away_Matches', 'GF': 'Away_GF', 'GA': 'Away_GA',
        'Shots': 'Away_Shots', 'Shots_OT': 'Away_Shots_OT',
        'Shots_Against': 'Away_Shots_Against', 'Shots_OT_Against': 'Away_Shots_OT_Against',
        'Points': 'Away_Points',
    })
    ha = home.join(away, on=['Div', 'Team', 'Season'], how='inner')
    ha = ha.filter((pl.col('Home_Matches') >= 10) & (pl.col('Away_Matches') >= 10))

    for prefix, mcol, amcol in [('Home', 'Home_', ''), ('Away', 'Away_', '')]:
        pass
    ha = ha.with_columns([
        (pl.col('Home_Points') / pl.col('Home_Matches')).alias('Home_PPG'),
        (pl.col('Away_Points') / pl.col('Away_Matches')).alias('Away_PPG'),
        (pl.col('Home_Shots') / pl.col('Home_Matches')).alias('Home_Shots_PG'),
        (pl.col('Away_Shots') / pl.col('Away_Matches')).alias('Away_Shots_PG'),
        (pl.col('Home_Shots_OT') / pl.col('Home_Matches')).alias('Home_Shots_OT_PG'),
        (pl.col('Away_Shots_OT') / pl.col('Away_Matches')).alias('Away_Shots_OT_PG'),
        (pl.col('Home_Shots_Against') / pl.col('Home_Matches')).alias('Home_Shots_Ag_PG'),
        (pl.col('Away_Shots_Against') / pl.col('Away_Matches')).alias('Away_Shots_Ag_PG'),
        (pl.col('Home_Shots_OT_Against') / pl.col('Home_Matches')).alias('Home_Shots_OT_Ag_PG'),
        (pl.col('Away_Shots_OT_Against') / pl.col('Away_Matches')).alias('Away_Shots_OT_Ag_PG'),
        _safe_ratio(pl.col('Home_Shots_OT'), pl.col('Home_Shots')).alias('Home_Shot_Acc'),
        _safe_ratio(pl.col('Away_Shots_OT'), pl.col('Away_Shots')).alias('Away_Shot_Acc'),
        _safe_ratio(pl.col('Home_GF'), pl.col('Home_Shots')).alias('Home_GPS'),
        _safe_ratio(pl.col('Away_GF'), pl.col('Away_Shots')).alias('Away_GPS'),
        _safe_ratio(pl.col('Home_GF'), pl.col('Home_Shots_OT')).alias('Home_GPSOT'),
        _safe_ratio(pl.col('Away_GF'), pl.col('Away_Shots_OT')).alias('Away_GPSOT'),
        _safe_ratio(pl.col('Home_GA'), pl.col('Home_Shots_Against')).alias('Home_GAOS'),
        _safe_ratio(pl.col('Away_GA'), pl.col('Away_Shots_Against')).alias('Away_GAOS'),
        _safe_ratio(pl.col('Home_GA'), pl.col('Home_Shots_OT_Against')).alias('Home_GAOSOT'),
        _safe_ratio(pl.col('Away_GA'), pl.col('Away_Shots_OT_Against')).alias('Away_GAOSOT'),
    ])

    metrics = ['PPG', 'Shots_PG', 'Shots_OT_PG', 'Shots_Ag_PG', 'Shots_OT_Ag_PG',
               'Shot_Acc', 'GPS', 'GPSOT', 'GAOS', 'GAOSOT']
    for m in metrics:
        ha = ha.with_columns((pl.col(f'Home_{m}') - pl.col(f'Away_{m}')).alias(f'Diff_{m}'))

    results['team_season_ha'] = ha

    # Paired reverse-fixture comparison
    tm = team_matches.with_columns(
        (pl.col('Div') + pl.lit('_') + pl.col('Season') + pl.lit('_') +
         pl.col('Team') + pl.lit('_') + pl.col('Opponent')).alias('Pair_Key')
    )
    pair_counts = tm.group_by('Pair_Key').agg(pl.len().alias('Count'))
    complete_pairs = set(pair_counts.filter(pl.col('Count') == 2).select('Pair_Key').to_series().to_list())
    paired_data = tm.filter(pl.col('Pair_Key').is_in(list(complete_pairs)))

    home = paired_data.filter(pl.col('Venue') == 'Home')
    away = paired_data.filter(pl.col('Venue') == 'Away')
    # join on Pair_Key
    if home.height > 0 and away.height > 0:
        m = home.join(away, on='Pair_Key', how='inner', suffix='_A').with_columns([
            pl.col('Div').alias('Div_H'),
        ])
        # rebuild clean column set
        m = m.select([
            pl.col('Div_H').alias('Div'),
            pl.col('Season').alias('Season'),
            pl.col('Team').alias('Team'),
            pl.col('Opponent').alias('Opponent'),
            pl.col('Points').alias('Points_H'),
            pl.col('Points_A').alias('Points_A'),
            pl.col('Shots').alias('Shots_H'),
            pl.col('Shots_A').alias('Shots_A'),
            pl.col('Shots_Against').alias('Shots_Against_H'),
            pl.col('Shots_Against_A').alias('Shots_Against_A'),
            pl.col('Shots_OT').alias('Shots_OT_H'),
            pl.col('Shots_OT_A').alias('Shots_OT_A'),
            pl.col('GF').alias('GF_H'),
            pl.col('GF_A').alias('GF_A'),
            pl.col('GA').alias('GA_H'),
            pl.col('GA_A').alias('GA_A'),
        ])
        m = m.with_columns([
            (pl.col('Points_H') - pl.col('Points_A')).alias('Diff_PPG'),
            (pl.col('Shots_H') - pl.col('Shots_A')).alias('Diff_Shots_PG'),
            (pl.col('Shots_Against_H') - pl.col('Shots_Against_A')).alias('Diff_Shots_Ag_PG'),
            (_safe_ratio(pl.col('Shots_OT_H'), pl.col('Shots_H')) -
             _safe_ratio(pl.col('Shots_OT_A'), pl.col('Shots_A'))).alias('Diff_Shot_Acc'),
            (_safe_ratio(pl.col('GF_H'), pl.col('Shots_H')) -
             _safe_ratio(pl.col('GF_A'), pl.col('Shots_A'))).alias('Diff_GPS'),
            (_safe_ratio(pl.col('GA_H'), pl.col('Shots_Against_H')) -
             _safe_ratio(pl.col('GA_A'), pl.col('Shots_Against_A'))).alias('Diff_GAOS'),
        ])
        paired_diffs = m.select(['Div', 'Season', 'Team', 'Opponent',
                                'Diff_PPG', 'Diff_Shots_PG', 'Diff_Shots_Ag_PG',
                                'Diff_Shot_Acc', 'Diff_GPS', 'Diff_GAOS'])
    else:
        paired_diffs = pl.DataFrame({
            'Div': [], 'Season': [], 'Team': [], 'Opponent': [],
            'Diff_PPG': [], 'Diff_Shots_PG': [], 'Diff_Shots_Ag_PG': [],
            'Diff_Shot_Acc': [], 'Diff_GPS': [], 'Diff_GAOS': [],
        })
    results['paired_fixture_diffs'] = paired_diffs

    # Overall component contributions to HA
    valid = ha.select(pl.col('Diff_PPG').is_not_null()).to_series()
    results['ha_decomposition'] = {}
    if int(valid.sum()) > 30:
        hav = ha.filter(valid)
        X = hav.select(['Diff_Shots_PG', 'Diff_Shots_Ag_PG', 'Diff_Shot_Acc', 'Diff_GPS', 'Diff_GAOS']).drop_nulls()
        y = _np(hav.filter(pl.col('Diff_PPG').is_not_null() & pl.col('Diff_Shots_PG').is_not_null() &
                       pl.col('Diff_Shots_Ag_PG').is_not_null() & pl.col('Diff_Shot_Acc').is_not_null() &
                       pl.col('Diff_GPS').is_not_null() & pl.col('Diff_GAOS').is_not_null()), 'Diff_PPG')
        Xv = X.to_numpy()
        if Xv.shape[0] > 30 and len(y) > 30:
            reg = LinearRegression().fit(Xv, y)
            results['ha_decomposition'] = {
                'coefficients': {c: float(v) for c, v in zip(X.columns, reg.coef_)},
                'intercept': float(reg.intercept_),
                'r2': float(reg.score(Xv, y)),
                'n': len(y),
            }
        else:
            results['ha_decomposition'] = {'note': 'Insufficient complete cases after dropping missing components'}
    else:
        results['ha_decomposition'] = {'note': 'Insufficient valid Diff_PPG cases'}

    # Persistence across seasons
    season_map = pl.DataFrame({'Season': list(SEASON_ORDER.keys()),
                               'Season_Num': list(SEASON_ORDER.values())})
    ha = ha.join(season_map, on='Season', how='left').sort(['Div', 'Team', 'Season_Num'])

    persistence = []
    gb = ha.group_by(['Div', 'Team'])
    for key in gb:
        (div, team), g = key
        if g.height >= 2:
            for m in ['Diff_PPG', 'Diff_Shots_PG', 'Diff_Shot_Acc', 'Diff_GPS', 'Diff_GAOS']:
                v = g.select(pl.col(m).is_not_null()).to_series()
                if int(v.sum()) >= 2:
                    x = _np(g.filter(v), 'Season_Num')
                    y = _np(g.filter(v), m)
                    r, p, n = _pearsonr(x, y)
                    persistence.append({'Div': div, 'Team': team, 'Metric': m,
                                        'Trend_R': float(r), 'Trend_P': float(p), 'Seasons': int(n)})
    results['ha_component_persistence'] = pl.DataFrame(persistence) if persistence else pl.DataFrame()

    league_ha = ha.group_by('Div').agg([
        pl.col('Team').n_unique().alias('Teams'),
        pl.col('Diff_PPG').mean().alias('Avg_Diff_PPG'),
        pl.col('Diff_Shots_PG').mean().alias('Avg_Diff_Shots_PG'),
        pl.col('Diff_Shots_Ag_PG').mean().alias('Avg_Diff_Shots_Ag_PG'),
        pl.col('Diff_Shot_Acc').mean().alias('Avg_Diff_Shot_Acc'),
        pl.col('Diff_GPS').mean().alias('Avg_Diff_GPS'),
        pl.col('Diff_GAOS').mean().alias('Avg_Diff_GAOS'),
    ]).sort('Div')
    results['league_ha_components'] = league_ha

    return results


def analyze_close_match_dependence(team_matches: pl.DataFrame, team_season: pl.DataFrame,
                                   min_matches: int = 20) -> Dict:
    """
    Question 3: Which teams rely unusually heavily on one-goal results?
    """
    results: Dict = {}

    tm = team_matches.with_columns([
        pl.col('GD').abs().alias('Margin'),
        pl.when((pl.col('FT_Result') == 'W') & (pl.col('GD').abs() == 1)).then(pl.lit('One_Goal_Win'))
        .when((pl.col('FT_Result') == 'W') & (pl.col('GD').abs() >= 2)).then(pl.lit('Multi_Goal_Win'))
        .when((pl.col('FT_Result') == 'L') & (pl.col('GD').abs() == 1)).then(pl.lit('One_Goal_Loss'))
        .when((pl.col('FT_Result') == 'L') & (pl.col('GD').abs() >= 2)).then(pl.lit('Multi_Goal_Loss'))
        .when(pl.col('FT_Result') == 'D').then(pl.lit('Draw'))
        .otherwise(pl.lit('Other')).alias('Result_Type')
    ])

    outcome_comp = tm.group_by(['Div', 'Team', 'Season', 'Result_Type']).agg([
        pl.len().alias('Count'),
        pl.col('Points').sum().alias('Points'),
    ])
    outcome_pivot = outcome_comp.pivot(on='Result_Type', index=['Div', 'Team', 'Season'],
                                       values=['Count', 'Points'], aggregate_function='sum')
    # Fill nulls with 0
    count_cols = [c for c in outcome_pivot.columns if c.startswith('Count_')]
    pts_cols = [c for c in outcome_pivot.columns if c.startswith('Points_')]
    fill_exprs = [pl.col(c).fill_null(0) for c in count_cols + pts_cols]
    if fill_exprs:
        outcome_pivot = outcome_pivot.with_columns(fill_exprs)

    outcome_pivot = outcome_pivot.with_columns([
        pl.sum_horizontal(count_cols).alias('Total_Matches'),
        pl.sum_horizontal(pts_cols).alias('Total_Points'),
    ]).with_columns([
        (pl.col('Total_Points') / pl.col('Total_Matches')).alias('PPG'),
        _safe_ratio(pl.col('Count_One_Goal_Win'), pl.col('Total_Matches')).alias('One_Goal_Win_Pct'),
        _safe_ratio(pl.col('Points_One_Goal_Win'), pl.col('Total_Points')).alias('Points_From_One_Goal_Wins'),
    ])

    results['season_outcome_composition'] = outcome_pivot

    valid_seasons = outcome_pivot.filter(pl.col('Total_Matches') >= min_matches)
    valid = valid_seasons.select(pl.col('PPG').is_not_null() & pl.col('Points_From_One_Goal_Wins').is_not_null()).to_series()
    if int(valid.sum()) > 50:
        vs = valid_seasons.filter(valid)
        slope, intercept, r, p, se = stats.linregress(_np(vs, 'PPG'), _np(vs, 'Points_From_One_Goal_Wins'))
        valid_seasons = vs.with_columns(
            (pl.col('Points_From_One_Goal_Wins') - (pl.lit(slope) * pl.col('PPG') + pl.lit(intercept)))
            .alias('Adj_Points_From_One_Goal_Wins')
        )
    else:
        valid_seasons = valid_seasons.with_columns(pl.col('Points_From_One_Goal_Wins').alias('Adj_Points_From_One_Goal_Wins'))
    results['close_match_dependence'] = valid_seasons

    # Early vs late season
    tm = tm.with_columns([
        (pl.col('Date').cum_count().over(TEAM_SEASON_GROUP) + 1).alias('Match_Num')
    ])
    max_match = tm.select(pl.col('Match_Num').max().over(TEAM_SEASON_GROUP)).to_series()
    tm = tm.with_columns(pl.Series('max_match', max_match))
    tm = tm.with_columns(
        pl.when(pl.col('Match_Num') <= pl.col('max_match') / 2).then(pl.lit('Early')).otherwise(pl.lit('Late')).alias('Season_Half')
    )

    half_comp = tm.group_by(['Div', 'Team', 'Season', 'Season_Half', 'Result_Type']).agg([
        pl.len().alias('Count'), pl.col('Points').sum().alias('Points'),
    ])
    half_pivot = half_comp.pivot(on='Result_Type', index=['Div', 'Team', 'Season', 'Season_Half'],
                                 values=['Count', 'Points'], aggregate_function='sum')
    hc_cols = [c for c in half_pivot.columns if c.startswith('Count_')]
    hp_cols = [c for c in half_pivot.columns if c.startswith('Points_')]
    if hc_cols + hp_cols:
        half_pivot = half_pivot.with_columns([pl.col(c).fill_null(0) for c in hc_cols + hp_cols])
    half_pivot = half_pivot.with_columns([
        pl.sum_horizontal(hc_cols).alias('Total_Matches'),
        pl.sum_horizontal(hp_cols).alias('Total_Points'),
    ]).with_columns([
        (pl.col('Total_Points') / pl.col('Total_Matches')).alias('PPG'),
        _safe_ratio(pl.col('Count_One_Goal_Win'), pl.col('Total_Matches')).alias('One_Goal_Win_Pct'),
        _safe_ratio(pl.col('Points_One_Goal_Win'), pl.col('Total_Points')).alias('Points_From_One_Goal_Wins'),
    ])

    early = half_pivot.filter(pl.col('Season_Half') == 'Early').select([
        'Div', 'Team', 'Season',
        pl.col('PPG').alias('PPG_Early'),
        pl.col('One_Goal_Win_Pct').alias('OGW_Pct_Early'),
        pl.col('Points_From_One_Goal_Wins').alias('Points_From_OGW_Early'),
    ])
    late = half_pivot.filter(pl.col('Season_Half') == 'Late').select([
        'Div', 'Team', 'Season',
        pl.col('PPG').alias('PPG_Late'),
        pl.col('One_Goal_Win_Pct').alias('OGW_Pct_Late'),
        pl.col('Points_From_One_Goal_Wins').alias('Points_From_OGW_Late'),
    ])
    early_late = early.join(late, on=['Div', 'Team', 'Season'], how='inner')
    early_late = early_late.filter(pl.col('PPG_Early').is_not_null() & pl.col('PPG_Late').is_not_null())

    if early_late.height > 20:
        v = early_late.select(pl.col('OGW_Pct_Early').is_not_null() & pl.col('OGW_Pct_Late').is_not_null()).to_series()
        r_pct, p_pct, _ = _pearsonr(_np(early_late.filter(v), 'OGW_Pct_Early'), _np(early_late.filter(v), 'OGW_Pct_Late'))
        v2 = early_late.select(pl.col('Points_From_OGW_Early').is_not_null() & pl.col('Points_From_OGW_Late').is_not_null()).to_series()
        r_pts, p_pts, _ = _pearsonr(_np(early_late.filter(v2), 'Points_From_OGW_Early'), _np(early_late.filter(v2), 'Points_From_OGW_Late'))
        results['early_late_persistence'] = {
            'ogw_pct': {'r': float(r_pct), 'p': float(p_pct), 'n': int(v.sum())},
            'points_from_ogw': {'r': float(r_pts), 'p': float(p_pts), 'n': int(v2.sum())},
        }
    results['early_late_comparison'] = early_late

    # Shot dominance persistence
    shot_dom_persist = []
    gb = tm.group_by(['Div', 'Team', 'Season'])
    for key in gb:
        (div, team, season), g = key
        early_g = g.filter(pl.col('Season_Half') == 'Early')
        late_g = g.filter(pl.col('Season_Half') == 'Late')
        if early_g.height >= 5 and late_g.height >= 5:
            shot_dom_persist.append({
                'Div': div, 'Team': team, 'Season': season,
                'ShotDom_Early': float(early_g.select(pl.col('Shot_Dominance').mean()).to_series()[0]),
                'ShotDom_Late': float(late_g.select(pl.col('Shot_Dominance').mean()).to_series()[0]),
            })
    shot_dom_df = pl.DataFrame(shot_dom_persist) if shot_dom_persist else pl.DataFrame()
    if shot_dom_df.height > 20:
        v = shot_dom_df.select(pl.col('ShotDom_Early').is_not_null() & pl.col('ShotDom_Late').is_not_null()).to_series()
        r_sd, p_sd, _ = _pearsonr(_np(shot_dom_df.filter(v), 'ShotDom_Early'), _np(shot_dom_df.filter(v), 'ShotDom_Late'))
        results['shot_dominance_persistence'] = {'r': float(r_sd), 'p': float(p_sd), 'n': int(v.sum())}
    results['shot_dominance_early_late'] = shot_dom_df

    team_summary = valid_seasons.group_by('Team').agg([
        pl.len().alias('Seasons'),
        pl.col('PPG').mean().alias('Avg_PPG'),
        pl.col('One_Goal_Win_Pct').mean().alias('Avg_One_Goal_Win_Pct'),
        pl.col('Points_From_One_Goal_Wins').mean().alias('Avg_Points_From_One_Goal_Wins'),
        pl.col('Adj_Points_From_One_Goal_Wins').mean().alias('Avg_Adj_Points_From_One_Goal_Wins'),
    ]).filter(pl.col('Seasons') >= 3).sort('Avg_Adj_Points_From_One_Goal_Wins', descending=True)
    results['team_close_match_summary'] = team_summary

    return results


def analyze_rest_asymmetry(team_matches: pl.DataFrame, master: pl.DataFrame,
                           min_matches: int = 20) -> Dict:
    """
    Question 4: Is being less rested than the opponent associated with worse performance?
    """
    results: Dict = {}

    tm = add_lagged_team_strength(team_matches).sort(['Div', 'Team', 'Season', 'Date'])

    # Rest days per team per match from master
    master_rest = master.select(['Div', 'Season', 'Date', 'HomeTeam', 'AwayTeam']).sort(
        ['Div', 'Season', 'Date']
    )
    master_rest = master_rest.with_columns([
        pl.col('Date').diff().over(['Div', 'Season', 'HomeTeam']).dt.total_days().alias('Home_Rest'),
        pl.col('Date').diff().over(['Div', 'Season', 'AwayTeam']).dt.total_days().alias('Away_Rest'),
    ])
    home_rest = master_rest.select(['Div', 'Season', 'Date', 'HomeTeam', 'Home_Rest']).rename(
        {'HomeTeam': 'Team', 'Home_Rest': 'Rest_Days_Actual'}
    )
    away_rest = master_rest.select(['Div', 'Season', 'Date', 'AwayTeam', 'Away_Rest']).rename(
        {'AwayTeam': 'Team', 'Away_Rest': 'Rest_Days_Actual'}
    )
    rest_data = pl.concat([home_rest, away_rest], how='diagonal')
    tm = tm.join(rest_data, on=['Div', 'Season', 'Date', 'Team'], how='left')

    # Opponent rest
    opp_rest = tm.select(['Div', 'Season', 'Date', 'Opponent', 'Rest_Days_Actual']).rename(
        {'Opponent': 'Team', 'Rest_Days_Actual': 'Opp_Rest_Days'}
    )
    tm = tm.join(opp_rest, on=['Div', 'Season', 'Date', 'Team'], how='left')
    tm = tm.with_columns((pl.col('Rest_Days_Actual') - pl.col('Opp_Rest_Days')).alias('Rest_Diff'))

    results['match_rest_data'] = tm

    def _rest_days_cat(e: pl.Expr) -> pl.Expr:
        return (pl.when(e <= 2).then(pl.lit('\u22642'))
                .when(e <= 3).then(pl.lit('3'))
                .when(e <= 4).then(pl.lit('4'))
                .when(e <= 5).then(pl.lit('5'))
                .when(e <= 6).then(pl.lit('6'))
                .when(e <= 7).then(pl.lit('7'))
                .when(e <= 10).then(pl.lit('8-10'))
                .when(e <= 14).then(pl.lit('11-14'))
                .otherwise(pl.lit('15+')))

    density = tm.group_by([_rest_days_cat(pl.col('Rest_Days_Actual')).alias('Own_Rest_Cat'),
                           _rest_days_cat(pl.col('Opp_Rest_Days')).alias('Opp_Rest_Cat')]).agg(
        pl.len().alias('Count')
    ).sort(['Own_Rest_Cat', 'Opp_Rest_Cat'])
    results['rest_density'] = density

    valid = tm.select(pl.col('Rest_Days_Actual').is_not_null() & pl.col('Opp_Rest_Days').is_not_null() & pl.col('Points').is_not_null()).to_series()
    tm_valid = tm.filter(valid)

    tm_valid = tm_valid.with_columns(
        (pl.col('Div') + pl.lit('_') + pl.col('Season')).alias('LeagueSeason'),
        pl.when(pl.col('Venue') == 'Home').then(1).otherwise(0).cast(pl.Int8).alias('Venue_Code'),
    )
    ls_dummies = tm_valid.select('LeagueSeason').to_dummies(columns='LeagueSeason', drop_first=True)
    X = pl.concat([
        tm_valid.select(['Rest_Diff', 'Venue_Code', 'PreMatch_PPG_Shrunk']),
        ls_dummies,
    ], how='horizontal').to_numpy()
    y = _np(tm_valid, 'Points')
    reg = LinearRegression(fit_intercept=True).fit(X, y)
    tm_valid = tm_valid.with_columns(pl.Series('Expected_Points', reg.predict(X)))
    tm_valid = tm_valid.with_columns((pl.col('Points') - pl.col('Expected_Points')).alias('Residual'))

    results['rest_points_model'] = {
        'rest_diff_coef': float(reg.coef_[0]),
        'venue_coef': float(reg.coef_[1]),
        'strength_coef': float(reg.coef_[2]),
        'r2': float(reg.score(X, y)),
        'n': int(y.shape[0]),
    }

    rest_effects = []
    for venue in ['Home', 'Away']:
        for strength in ['Weak', 'Average', 'Strong']:
            vc = 1 if venue == 'Home' else 0
            subset = tm_valid.filter((pl.col('Venue_Code') == vc))
            q33 = pl.col('PreMatch_PPG_Shrunk').quantile(1 / 3).over(['Div', 'Season'])
            q66 = pl.col('PreMatch_PPG_Shrunk').quantile(2 / 3).over(['Div', 'Season'])
            subset = subset.with_columns(
                pl.when(pl.col('PreMatch_PPG_Shrunk') <= q33).then(pl.lit('Weak'))
                .when(pl.col('PreMatch_PPG_Shrunk') <= q66).then(pl.lit('Average'))
                .otherwise(pl.lit('Strong')).alias('Strength_Tercile')
            )
            subset = subset.filter(pl.col('Strength_Tercile') == strength)
            v = subset.select(pl.col('Rest_Diff').is_not_null() & pl.col('Residual').is_not_null()).to_series()
            if int(v.sum()) < 20:
                continue
            subv = subset.filter(v)
            slope, intercept, r, p, se = stats.linregress(_np(subv, 'Rest_Diff'), _np(subv, 'Residual'))
            rest_effects.append({'Venue': venue, 'Strength': strength,
                                 'Slope': float(slope), 'R': float(r), 'P': float(p), 'N': int(v.sum())})
    results['rest_effects_by_venue_strength'] = pl.DataFrame(rest_effects) if rest_effects else pl.DataFrame()

    shot_effects = []
    for venue in ['Home', 'Away']:
        for strength in ['Weak', 'Average', 'Strong']:
            vc = 1 if venue == 'Home' else 0
            subset = tm_valid.filter(pl.col('Venue_Code') == vc)
            q33 = pl.col('PreMatch_PPG_Shrunk').quantile(1 / 3).over(['Div', 'Season'])
            q66 = pl.col('PreMatch_PPG_Shrunk').quantile(2 / 3).over(['Div', 'Season'])
            subset = subset.with_columns(
                pl.when(pl.col('PreMatch_PPG_Shrunk') <= q33).then(pl.lit('Weak'))
                .when(pl.col('PreMatch_PPG_Shrunk') <= q66).then(pl.lit('Average'))
                .otherwise(pl.lit('Strong')).alias('Strength_Tercile')
            )
            subset = subset.filter(pl.col('Strength_Tercile') == strength)
            v = subset.select(pl.col('Rest_Diff').is_not_null() & pl.col('Shot_Dominance').is_not_null()).to_series()
            if int(v.sum()) < 20:
                continue
            subv = subset.filter(v)
            slope, intercept, r, p, se = stats.linregress(_np(subv, 'Rest_Diff'), _np(subv, 'Shot_Dominance'))
            shot_effects.append({'Venue': venue, 'Strength': strength,
                                 'Slope': float(slope), 'R': float(r), 'P': float(p), 'N': int(v.sum())})
    results['rest_shot_effects'] = pl.DataFrame(shot_effects) if shot_effects else pl.DataFrame()

    heatmap_data = []
    for rd in range(-4, 5):
        for venue in ['Home', 'Away']:
            for strength in ['Weak', 'Average', 'Strong']:
                vc = 1 if venue == 'Home' else 0
                subset = tm_valid.filter((pl.col('Rest_Diff') == rd) & (pl.col('Venue_Code') == vc))
                subset = subset.with_columns(
                    pl.when(pl.col('PreMatch_PPG_Shrunk') <= pl.col('PreMatch_PPG_Shrunk').quantile(1 / 3).over(['Div', 'Season']))
                    .then(pl.lit('Weak'))
                    .when(pl.col('PreMatch_PPG_Shrunk') <= pl.col('PreMatch_PPG_Shrunk').quantile(2 / 3).over(['Div', 'Season']))
                    .then(pl.lit('Average'))
                    .otherwise(pl.lit('Strong')).alias('Strength_Tercile')
                )
                subset = subset.filter(pl.col('Strength_Tercile') == strength)
                if subset.height >= 10:
                    heatmap_data.append({
                        'Rest_Diff': rd, 'Venue': venue, 'Strength': strength,
                        'Mean_Points': float(subset.select(pl.col('Points').mean()).to_series()[0]),
                        'Mean_Residual': float(subset.select(pl.col('Residual').mean()).to_series()[0]),
                        'Mean_Shot_Dom': float(subset.select(pl.col('Shot_Dominance').mean()).to_series()[0]),
                        'N': subset.height,
                    })
    results['rest_heatmap'] = pl.DataFrame(heatmap_data) if heatmap_data else pl.DataFrame()

    tm_no_covid = tm.filter(pl.col('Season') != '2021')
    results['match_rest_data_no_covid'] = tm_no_covid

    return results


def analyze_performance_indicators_lead_results(team_matches: pl.DataFrame, team_season: pl.DataFrame,
                                                min_matches: int = 20) -> Dict:
    """
    Question 5: Does sustained improvement in shot dominance precede improved results?
    """
    results: Dict = {}

    tm = add_lagged_team_strength(team_matches.sort(['Div', 'Team', 'Season', 'Date']))

    ind_cols = ['Shot_Dominance', 'SOT_Dominance', 'Points', 'GD', 'GF', 'GA']
    out_cols = ['Points', 'GD', 'GF', 'GA', 'Shot_Dominance', 'SOT_Dominance']
    ind_rename = {'Shot_Dominance': 'Ind_Shot_Dom', 'SOT_Dominance': 'Ind_SOT_Dom',
                  'Points': 'Ind_PPG', 'GD': 'Ind_GD_PG', 'GF': 'Ind_GF_PG', 'GA': 'Ind_GA_PG'}
    out_rename = {'Points': 'Out_PPG', 'GD': 'Out_GD_PG', 'GF': 'Out_GF_PG', 'GA': 'Out_GA_PG',
                  'Shot_Dominance': 'Out_Shot_Dom', 'SOT_Dominance': 'Out_SOT_Dom'}

    def _window_df(ind_end, out_end, half=False):
        gk = TEAM_SEASON_GROUP
        g = tm
        cc = g.select(pl.cum_count('Date').over(gk)).to_series().to_numpy()
        n = g.select(pl.len().over(gk)).to_series().to_numpy()
        if half:
            valid = n >= 10
        else:
            valid = n >= out_end
        tmv = tm.filter(pl.Series(valid))
        ccv = cc[valid]
        nv = n[valid]
        if half:
            mid = nv // 2
            imask = ccv < mid
            omask = ccv >= mid
        else:
            imask = ccv < ind_end
            omask = (ccv >= ind_end) & (ccv < out_end)
        ind_sub = tmv.filter(pl.Series(imask))
        out_sub = tmv.filter(pl.Series(omask))
        ind_means = ind_sub.group_by(gk).agg([pl.col(c).mean().alias(ind_rename[c]) for c in ind_cols])
        out_means = out_sub.group_by(gk).agg([pl.col(c).mean().alias(out_rename[c]) for c in out_cols])
        n_ind = ind_sub.group_by(gk).agg(pl.len().alias('N_Indicator'))
        n_out = out_sub.group_by(gk).agg(pl.len().alias('N_Outcome'))
        df = ind_means.join(out_means, on=gk, how='full', coalesce=True).join(n_ind, on=gk, how='full', coalesce=True).join(n_out, on=gk, how='full', coalesce=True)
        df = df.filter((pl.col('N_Indicator') >= 5) & (pl.col('N_Outcome') >= 5))
        return df

    window1 = _window_df(10, 20)
    results['window_10_10'] = window1
    window2 = _window_df(15, 30)
    results['window_15_15'] = window2
    window3 = _window_df(0, 0, half=True)
    results['window_half_half'] = window3

    lead_lag_results = {}
    for window_name, wdf in [('10_10', window1), ('15_15', window2), ('half_half', window3)]:
        if wdf.height < 20:
            continue
        indicators = ['Ind_Shot_Dom', 'Ind_SOT_Dom', 'Ind_PPG', 'Ind_GD_PG', 'Ind_GF_PG', 'Ind_GA_PG']
        outcomes = ['Out_PPG', 'Out_GD_PG', 'Out_GF_PG', 'Out_GA_PG', 'Out_Shot_Dom', 'Out_SOT_Dom']
        window_corrs = {}
        for ind in indicators:
            for out in outcomes:
                if ind not in wdf.columns or out not in wdf.columns:
                    continue
                v = wdf.select(pl.col(ind).is_not_null() & pl.col(out).is_not_null()).to_series()
                if int(v.sum()) > 20:
                    r, p, n = _pearsonr(_np(wdf.filter(v), ind), _np(wdf.filter(v), out))
                    window_corrs[f'{ind}_vs_{out}'] = {'r': r, 'p': p, 'n': n}
        lead_lag_results[window_name] = window_corrs
    results['lead_lag_correlations'] = lead_lag_results

    for window_name, wdf in [('10_10', window1), ('15_15', window2), ('half_half', window3)]:
        if wdf.height < 30:
            continue
        v = wdf.select(pl.col('Ind_Shot_Dom').is_not_null() & pl.col('Ind_PPG').is_not_null() & pl.col('Out_PPG').is_not_null()).to_series()
        if int(v.sum()) < 30:
            continue
        dfv = wdf.filter(v)
        X1 = dfv.select('Ind_PPG').to_numpy(); y = dfv.select('Out_PPG').to_numpy().ravel()
        r2_1 = LinearRegression().fit(X1, y).score(X1, y)
        X2 = dfv.select(['Ind_PPG', 'Ind_Shot_Dom']).to_numpy()
        r2_2 = LinearRegression().fit(X2, y).score(X2, y)
        X3 = dfv.select(['Ind_PPG', 'Ind_SOT_Dom']).to_numpy()
        r2_3 = LinearRegression().fit(X3, y).score(X3, y)
        results[f'incremental_{window_name}'] = {
            'ppg_only_r2': float(r2_1), 'ppg_plus_shotdom_r2': float(r2_2),
            'ppg_plus_sotdom_r2': float(r2_3), 'incremental_shotdom': float(r2_2 - r2_1),
            'incremental_sotdom': float(r2_3 - r2_1), 'n': int(dfv.height),
        }

    tm = tm.with_columns((pl.col('Div') + pl.lit('_') + pl.col('Season')).alias('LeagueSeason'))
    for window_name, wdf in [('10_10', window1), ('15_15', window2), ('half_half', window3)]:
        if wdf.height < 30:
            continue
        wdf2 = wdf.with_columns((pl.col('Div') + pl.lit('_') + pl.col('Season')).alias('LeagueSeason'))
        for ind in ['Ind_Shot_Dom', 'Ind_SOT_Dom', 'Ind_PPG', 'Ind_GD_PG']:
            for out in ['Out_PPG', 'Out_GD_PG']:
                if ind not in wdf2.columns or out not in wdf2.columns:
                    continue
                v = wdf2.select(pl.col(ind).is_not_null() & pl.col(out).is_not_null() & pl.col('LeagueSeason').is_not_null()).to_series()
                if int(v.sum()) < 30:
                    continue
                dfv = wdf2.filter(v)
                ls_ind = dfv.group_by('LeagueSeason').agg(pl.col(ind).mean().alias('ls_ind'))
                ls_out = dfv.group_by('LeagueSeason').agg(pl.col(out).mean().alias('ls_out'))
                dfv = dfv.join(ls_ind, on='LeagueSeason', how='left').join(ls_out, on='LeagueSeason', how='left')
                dfv = dfv.with_columns([
                    (pl.col(ind) - pl.col('ls_ind')).alias('ind_resid'),
                    (pl.col(out) - pl.col('ls_out')).alias('out_resid'),
                ])
                v2 = dfv.select(pl.col('ind_resid').is_not_null() & pl.col('out_resid').is_not_null()).to_series()
                if int(v2.sum()) > 20:
                    key = f'adjusted_{window_name}'
                    if key not in results:
                        results[key] = {}
                    rr, pp, nn = _pearsonr(_np(dfv.filter(v2), 'ind_resid'), _np(dfv.filter(v2), 'out_resid'))
                    results[key][f'{ind}_vs_{out}'] = {'r': rr, 'p': pp, 'n': int(nn)}

    horizons = [(5, 10), (10, 10), (10, 15), (15, 15), (15, 20)]
    horizon_results = {}
    for ind_end, out_end in horizons:
        w = _window_df(ind_end, out_end)
        if w.height > 0:
            wd = w.drop_nulls()
            if wd.height > 20:
                v = wd.select(pl.col('Ind_Shot_Dom').is_not_null() & pl.col('Out_PPG').is_not_null()).to_series()
                if int(v.sum()) > 20:
                    r, p, n = _pearsonr(_np(wd.filter(v), 'Ind_Shot_Dom'), _np(wd.filter(v), 'Out_PPG'))
                    horizon_results[f'ind{ind_end}_out{out_end}'] = {'r': r, 'p': p, 'n': int(n)}
    results['horizon_sensitivity'] = horizon_results

    return results


def analyze_adaptation_after_promotion(team_matches: pl.DataFrame, team_season: pl.DataFrame,
                                       master: pl.DataFrame) -> Dict:
    """
    Question 7: Do promoted teams improve their relative performance during their first observed season?
    """
    results: Dict = {}

    ts = build_promotion_relegation_transitions(team_season)
    promoted = ts.filter((pl.col('Transition_Type') == 'Promotion') & pl.col('Consecutive'))
    relegated = ts.filter((pl.col('Transition_Type') == 'Relegation') & pl.col('Consecutive'))

    results['promoted_teams'] = promoted.select(['Team', 'Prev_Div', 'Div', 'Prev_Season_Num', 'Season_Num', 'Prev_PPG', 'PPG'])
    results['relegated_teams'] = relegated.select(['Team', 'Prev_Div', 'Div', 'Prev_Season_Num', 'Season_Num', 'Prev_PPG', 'PPG'])

    season_codes = {v: k for k, v in SEASON_ORDER.items()}

    promo_matches = []
    for row in promoted.iter_rows(named=True):
        team = row['Team']; div = row['Div']; season_num = row['Season_Num']
        season_code = season_codes.get(int(season_num), None)
        if season_code is None:
            continue
        matches = team_matches.filter((pl.col('Team') == team) & (pl.col('Div') == div) & (pl.col('Season') == season_code))
        if matches.height >= 15:
            matches = matches.sort('Date').with_columns([
                (pl.col('Date').cum_count() + 1).alias('Match_Num'),
                pl.lit(row['Prev_Div']).alias('Promoted_From'),
                pl.lit(row['Prev_PPG']).alias('Prev_PPG'),
            ])
            promo_matches.append(matches)

    if promo_matches:
        all_promo = pl.concat(promo_matches, how='diagonal')
        results['promoted_match_data'] = all_promo

        trajectory = all_promo.group_by('Match_Num').agg([
            pl.len().alias('N'),
            pl.col('Points').mean().alias('PPG'),
            pl.col('GF').mean().alias('GF_PG'),
            pl.col('GA').mean().alias('GA_PG'),
            pl.col('Shots').mean().alias('Shots_PG'),
            pl.col('Shots_Against').mean().alias('Shots_Ag_PG'),
            pl.col('Shot_Dominance').mean().alias('Shot_Dom'),
            pl.col('SOT_Dominance').mean().alias('SOT_Dom'),
            pl.col('Goals_Per_Shot').mean().alias('GPS'),
            pl.col('GA_Per_Opp_Shot').mean().alias('GAOS'),
        ]).sort('Match_Num')
        for col in ['PPG', 'GF_PG', 'GA_PG', 'Shots_PG', 'Shots_Ag_PG', 'Shot_Dom', 'SOT_Dom', 'GPS', 'GAOS']:
            trajectory = trajectory.with_columns(
                pl.col(col).rolling_mean(5, min_samples=1).alias(f'{col}_Roll5')
            )
        results['promoted_trajectory'] = trajectory

        comparison_trajectories = {}
        for div in all_promo.select('Div').unique().to_series().to_list():
            for season in all_promo.select('Season').unique().to_series().to_list():
                other_teams = all_promo.select('Team').unique().to_series().to_list()
                ot = team_matches.filter((pl.col('Div') == div) & (pl.col('Season') == season) & (~pl.col('Team').is_in(other_teams)))
                if ot.height > 100:
                    other_traj = ot.group_by('Match_Num').agg([
                        pl.col('Points').mean().alias('PPG'),
                        pl.col('GF').mean().alias('GF_PG'),
                        pl.col('GA').mean().alias('GA_PG'),
                        pl.col('Shots').mean().alias('Shots_PG'),
                        pl.col('Shots_Against').mean().alias('Shots_Ag_PG'),
                        pl.col('Shot_Dominance').mean().alias('Shot_Dom'),
                        pl.col('SOT_Dominance').mean().alias('SOT_Dom'),
                    ]).sort('Match_Num')
                    comparison_trajectories[f'{div}_{season}'] = other_traj
        results['comparison_trajectories'] = comparison_trajectories

        team_curves = []
        for team in all_promo.select('Team').unique().to_series().to_list():
            tdata = all_promo.filter(pl.col('Team') == team).sort('Match_Num')
            if tdata.height >= 15:
                tdata = tdata.with_columns([
                    pl.col('Points').cum_sum().alias('Cum_Points'),
                ]).with_columns((pl.col('Cum_Points') / pl.col('Match_Num')).alias('Cum_PPG'))
                v = tdata.select(pl.col('Cum_PPG').is_not_null()).to_series()
                if int(v.sum()) >= 10:
                    tdv = tdata.filter(v)
                    slope, intercept, r, p, se = stats.linregress(_np(tdv, 'Match_Num'), _np(tdv, 'Cum_PPG'))
                    prev_ppg_val = float('nan')
                    ttable = results.get('promotion_transition_table')
                    if ttable is not None and ttable.height > 0:
                        pp = ttable.filter(pl.col('Team') == team).select('Prev_PPG')
                        if pp.height > 0:
                            prev_ppg_val = float(pp.to_series()[0])
                    team_curves.append({
                        'Team': team, 'Div': tdata.select('Div').to_series()[0], 'Season': tdata.select('Season').to_series()[0],
                        'Prev_PPG': prev_ppg_val, 'PPG_Slope': float(slope), 'PPG_R': float(r),
                        'PPG_P': float(p), 'Matches': int(v.sum()),
                        'Start_PPG': float(tdata.head(5).select(pl.col('Points').mean()).to_series()[0]),
                        'End_PPG': float(tdata.tail(5).select(pl.col('Points').mean()).to_series()[0]),
                    })
        results['promoted_team_curves'] = pl.DataFrame(team_curves) if team_curves else pl.DataFrame()

        if results['promoted_team_curves'].height > 5:
            curves = results['promoted_team_curves']
            v = curves.select(pl.col('Prev_PPG').is_not_null() & pl.col('PPG_Slope').is_not_null()).to_series()
            if int(v.sum()) > 2:
                r, p, n = _pearsonr(_np(curves.filter(v), 'Prev_PPG'), _np(curves.filter(v), 'PPG_Slope'))
                results['baseline_strength_effect'] = {'r': float(r), 'p': float(p), 'n': int(n)}

    releg_matches = []
    for row in relegated.iter_rows(named=True):
        team = row['Team']; div = row['Div']; season_num = row['Season_Num']
        season_code = season_codes.get(int(season_num), None)
        if season_code is None:
            continue
        matches = team_matches.filter((pl.col('Team') == team) & (pl.col('Div') == div) & (pl.col('Season') == season_code))
        if matches.height >= 15:
            matches = matches.sort('Date').with_columns([
                (pl.col('Date').cum_count() + 1).alias('Match_Num'),
                pl.lit(row['Prev_Div']).alias('Relegated_From'),
                pl.lit(row['Prev_PPG']).alias('Prev_PPG'),
            ])
            releg_matches.append(matches)

    if releg_matches:
        all_releg = pl.concat(releg_matches, how='diagonal')
        results['relegated_match_data'] = all_releg
        releg_traj = all_releg.group_by('Match_Num').agg([
            pl.len().alias('N'),
            pl.col('Points').mean().alias('PPG'),
            pl.col('GF').mean().alias('GF_PG'),
            pl.col('GA').mean().alias('GA_PG'),
            pl.col('Shots').mean().alias('Shots_PG'),
            pl.col('Shots_Against').mean().alias('Shots_Ag_PG'),
        ]).sort('Match_Num')
        results['relegated_trajectory'] = releg_traj

    transition_table = promoted.select([
        'Team', 'Prev_Div', 'Div', 'Prev_Season_Num', 'Season_Num',
        'Prev_PPG', 'PPG', 'Prev_Shots_PG', 'Shots_PG',
        'Prev_Shots_Against_PG', 'Shots_Against_PG',
    ]).with_columns([
        (pl.col('PPG') - pl.col('Prev_PPG')).alias('PPG_Change'),
        (pl.col('Shots_PG') - pl.col('Prev_Shots_PG')).alias('Shots_PG_Change'),
        (pl.col('Shots_Against_PG') - pl.col('Prev_Shots_Against_PG')).alias('Shots_Ag_PG_Change'),
    ])
    results['promotion_transition_table'] = transition_table

    return results


def compute_metric_exposure(team_matches: pl.DataFrame, metric: str, min_periods: int = 1) -> pl.DataFrame:
    tm = team_matches.sort(['Div', 'Team', 'Season', 'Date'])
    valid = tm.select(pl.col(metric).is_not_null()).to_series()
    exposure = tm.filter(pl.Series(valid)).group_by(['Div', 'Team', 'Season']).agg(
        pl.len().alias(f'{metric}_Valid_Matches')
    )
    return exposure
