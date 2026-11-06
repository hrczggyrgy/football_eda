"""
Fresh Insights Analysis Module
Implements 5 new research questions beyond existing analyses:
1. Draw Specialists and Match-Outcome Structure
2. Match Tempo and Statistical Structure
3. Game State Effects on Match Statistics
4. Referee Characteristics and Potential Biases (English leagues only)
5. Expected Points vs Actual Points (Pythagorean Residuals)

Migrated to Polars: Polars is the primary analytical data structure. NumPy and
SciPy/scikit-learn used where they provide appropriate statistical
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

from features.engineer import SEASON_ORDER

TEAM_SEASON_GROUP = ['Div', 'Team', 'Season']
ENGLISH_LEAGUES = ['E0', 'E1', 'E2', 'E3']


def _pearsonr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, int]:
    m = (~np.isnan(x)) & (~np.isnan(y))
    n = int(m.sum())
    if n < 2:
        return float('nan'), float('nan'), n
    r, p = stats.pearsonr(x[m], y[m])
    return float(r), float(p), n


def _np(df: pl.DataFrame, col: str) -> np.ndarray:
    return df.select(pl.col(col).cast(pl.Float64)).to_series().to_numpy()


def _safe_ratio(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    return pl.when(den > 0).then(num / den).otherwise(None)


def analyze_draw_specialists(team_matches: pl.DataFrame, team_season: pl.DataFrame,
                             min_matches: int = 20) -> Dict:
    """
    Question 1: Draw Specialists and Match-Outcome Structure.
    """
    results: Dict = {}

    draw_data = team_matches.group_by(TEAM_SEASON_GROUP).agg([
        pl.len().alias('Matches'),
        ((pl.col('FT_Result') == 'D').sum()).alias('Draws'),
        ((pl.col('FT_Result') == 'W').sum()).alias('Wins'),
        ((pl.col('FT_Result') == 'L').sum()).alias('Losses'),
    ]).filter(pl.col('Matches') >= min_matches).with_columns([
        (pl.col('Draws') / pl.col('Matches')).alias('Draw_Pct'),
        (pl.col('Wins') / pl.col('Matches')).alias('Win_Pct'),
        (pl.col('Losses') / pl.col('Matches')).alias('Loss_Pct'),
    ])

    strength_cols = ['PPG', 'GF_PG', 'GA_PG', 'GD_PG', 'Shots_PG', 'Shot_Accuracy', 'Goals_Per_Shot_OT']
    available_strength = [c for c in strength_cols if c in team_season.columns]
    draw_data = draw_data.join(team_season.select(['Div', 'Team', 'Season'] + available_strength),
                               on=['Div', 'Team', 'Season'], how='left')

    # Per-division quadratic fit of expected draw rate on PPG; accumulate per-division
    # predictions into a list and join once (avoids duplicate-column issues across divisions).
    expected_parts: List[pl.DataFrame] = []
    for div in draw_data.select(pl.col('Div').unique()).to_series().to_list():
        lg = draw_data.filter(pl.col('Div') == div)
        valid = lg.select(pl.col('PPG').is_not_null() & pl.col('Draw_Pct').is_not_null()).to_series()
        if int(valid.sum()) < 20:
            continue
        lg_valid = lg.filter(valid)
        X = lg_valid.select('PPG').to_series().to_numpy()
        y = lg_valid.select('Draw_Pct').to_series().to_numpy()
        z = np.polyfit(X, y, 2)
        p = np.poly1d(z)
        all_lg = draw_data.filter(pl.col('Div') == div)
        pred = p(all_lg.select('PPG').fill_null(0).to_series().to_numpy())
        pred_clipped = np.clip(pred, 0, 1)
        expected_parts.append(
            all_lg.select(['Div', 'Team', 'Season']).with_columns(
                pl.Series('Expected_Draw_Pct', pred_clipped)
            )
        )

    if expected_parts:
        expected_df = pl.concat(expected_parts, how='diagonal')
        draw_data = draw_data.join(expected_df, on=['Div', 'Team', 'Season'], how='left')
    else:
        draw_data = draw_data.with_columns(
            pl.lit(float('nan')).cast(pl.Float64).alias('Expected_Draw_Pct')
        )

    draw_data = draw_data.with_columns(
        (pl.col('Draw_Pct') - pl.col('Expected_Draw_Pct')).alias('Draw_Residual')
    )

    results['team_season_draws'] = draw_data

    draw_corrs: Dict = {}
    for metric in ['PPG', 'GF_PG', 'GA_PG', 'GD_PG', 'Shots_PG', 'Shot_Accuracy', 'Goals_Per_Shot_OT',
                   'Win_Pct', 'Loss_Pct']:
        if metric in draw_data.columns:
            v = draw_data.select(pl.col(metric).is_not_null() & pl.col('Draw_Residual').is_not_null()).to_series()
            if int(v.sum()) > 20:
                r, p, n = _pearsonr(_np(draw_data.filter(v), metric), _np(draw_data.filter(v), 'Draw_Residual'))
                draw_corrs[metric] = {'r': r, 'p': p, 'n': n}
    results['draw_residual_correlations'] = draw_corrs

    league_draws = draw_data.group_by('Div').agg([
        pl.col('Draw_Pct').mean().alias('Mean_Draw_Pct'),
        pl.col('Expected_Draw_Pct').mean().alias('Mean_Expected_Draw'),
        pl.col('Draw_Residual').std().alias('Std_Draw_Residual'),
    ]).sort('Div')
    results['league_draw_rates'] = league_draws

    draw_data = draw_data.with_columns(
        pl.col('Season').replace(SEASON_ORDER).cast(pl.Int64).alias('Season_Num')
    ).sort(['Div', 'Team', 'Season_Num'])
    draw_data = draw_data.with_columns([
        pl.col('Draw_Residual').shift(-1).over(['Div', 'Team']).alias('Next_Draw_Residual'),
        pl.col('Draw_Pct').shift(-1).over(['Div', 'Team']).alias('Next_Draw_Pct'),
        pl.col('Season_Num').shift(-1).over(['Div', 'Team']).alias('Next_Season_Num'),
        pl.col('Div').shift(-1).over(['Div', 'Team']).alias('Next_Div'),
        pl.col('PPG').shift(-1).over(['Div', 'Team']).alias('Next_PPG'),
    ])
    valid_pairs = (
        pl.col('Next_Season_Num').is_not_null() &
        (pl.col('Next_Season_Num') - pl.col('Season_Num') == 1) &
        (pl.col('Next_Div') == pl.col('Div'))
    )
    consecutive = draw_data.filter(valid_pairs)

    persistence: Dict = {}
    for metric in ['Draw_Residual', 'Draw_Pct', 'Win_Pct', 'Loss_Pct']:
        nxt = f'Next_{metric}'
        if metric in consecutive.columns and nxt in consecutive.columns:
            v = consecutive.select(pl.col(metric).is_not_null() & pl.col(nxt).is_not_null()).to_series()
            if int(v.sum()) > 10:
                r, p, n = _pearsonr(_np(consecutive.filter(v), metric), _np(consecutive.filter(v), nxt))
                persistence[metric] = {'r': r, 'p': p, 'n': n}
    results['season_to_season_persistence'] = persistence

    tm = team_matches.with_columns([
        (pl.col('Date').cum_count().over(TEAM_SEASON_GROUP) + 1).alias('Match_Num')
    ])
    max_match = tm.select(pl.col('Match_Num').max().over(TEAM_SEASON_GROUP)).to_series()
    tm = tm.with_columns(pl.Series('max_match', max_match))
    tm = tm.with_columns(
        pl.when(pl.col('Match_Num') <= pl.col('max_match') / 2).then(pl.lit('First')).otherwise(pl.lit('Second')).alias('Half')
    )
    half_draws = tm.group_by(['Div', 'Team', 'Season', 'Half']).agg([
        pl.len().alias('Matches'),
        ((pl.col('FT_Result') == 'D').sum()).alias('Draws'),
    ]).with_columns((pl.col('Draws') / pl.col('Matches')).alias('Draw_Pct'))
    half_pivot = half_draws.pivot(on='Half', index=['Div', 'Team', 'Season'], values='Draw_Pct', aggregate_function='mean')
    if 'First' in half_pivot.columns and 'Second' in half_pivot.columns:
        v = half_pivot.select(pl.col('First').is_not_null() & pl.col('Second').is_not_null()).to_series()
        if int(v.sum()) > 10:
            r, p, n = _pearsonr(_np(half_pivot.filter(v), 'First'), _np(half_pivot.filter(v), 'Second'))
            results['half_season_persistence'] = {'r': r, 'p': p, 'n': n}

    valid_res = draw_data.select(pl.col('Draw_Residual').is_not_null()).to_series()
    if int(valid_res.sum()) > 50:
        dr = draw_data.filter(valid_res)
        high_draw = dr.sort('Draw_Residual', descending=True).head(20)
        low_draw = dr.sort('Draw_Residual').head(20)
        profile_cols = ['PPG', 'GF_PG', 'GA_PG', 'GD_PG', 'Shots_PG', 'Shot_Accuracy',
                        'Goals_Per_Shot_OT', 'Win_Pct', 'Loss_Pct']
        available = [c for c in profile_cols if c in draw_data.columns]
        results['high_draw_profile'] = high_draw.select(['Div', 'Team', 'Season', 'Draw_Residual'] + available).to_dicts()
        results['low_draw_profile'] = low_draw.select(['Div', 'Team', 'Season', 'Draw_Residual'] + available).to_dicts()
        profile_diff: Dict = {}
        for col in available:
            h = high_draw.select(pl.col(col).drop_nulls()).to_series().to_numpy()
            l = low_draw.select(pl.col(col).drop_nulls()).to_series().to_numpy()
            if len(h) > 5 and len(l) > 5:
                t_stat, p_val = stats.ttest_ind(h, l, equal_var=False)
                profile_diff[col] = {'high_mean': float(h.mean()), 'low_mean': float(l.mean()),
                                     't': float(t_stat), 'p': float(p_val)}
        results['profile_differences'] = profile_diff

    return results


def analyze_match_tempo(team_matches: pl.DataFrame, master: pl.DataFrame,
                        team_season: pl.DataFrame, min_matches: int = 20) -> Dict:
    """
    Question 2: Match Tempo and Statistical Structure.
    """
    results: Dict = {}

    master = master.with_columns([
        (pl.col('HS') + pl.col('AS')).alias('Total_Shots'),
        (pl.col('FTHG') + pl.col('FTAG')).alias('Total_Goals'),
        (pl.col('HST') + pl.col('AST')).alias('Total_SOT'),
        (pl.col('FTHG') - pl.col('FTAG')).alias('Goal_Diff'),
    ])

    # Tempo quintiles
    q = master.select('Total_Shots').to_series().to_numpy()
    bounds = [float(np.quantile(q, k / 5)) for k in range(1, 5)]
    master = master.with_columns(
        pl.when(pl.col('Total_Shots') <= bounds[0]).then(pl.lit('Very_Low'))
        .when(pl.col('Total_Shots') <= bounds[1]).then(pl.lit('Low'))
        .when(pl.col('Total_Shots') <= bounds[2]).then(pl.lit('Medium'))
        .when(pl.col('Total_Shots') <= bounds[3]).then(pl.lit('High'))
        .otherwise(pl.lit('Very_High')).alias('Tempo_Quintile')
    )

    tempo_outcomes = master.group_by('Tempo_Quintile').agg([
        pl.len().alias('N'),
        ((pl.col('FTR') == 'H').mean()).alias('Home_Win'),
        ((pl.col('FTR') == 'D').mean()).alias('Draw'),
        ((pl.col('FTR') == 'A').mean()).alias('Away_Win'),
        pl.col('Total_Goals').mean().alias('Avg_Goals'),
        pl.col('Total_Shots').mean().alias('Avg_Shots'),
        pl.col('Total_SOT').mean().alias('Avg_SOT'),
        pl.col('Total_Goals').var().alias('Goal_Variance'),
    ]).sort('Tempo_Quintile')
    results['tempo_outcomes'] = tempo_outcomes

    # Entropy of result distribution by tempo
    from scipy.stats import entropy
    tempo_entropy_rows = []
    gb = master.group_by('Tempo_Quintile')
    for key in gb:
        (tq,), g = key
        vc = g.select(pl.col('FTR')).to_series().value_counts()
        counts = vc.select('count').to_series().to_numpy()
        probs = counts / counts.sum()
        tempo_entropy_rows.append({'Tempo_Quintile': tq, 'Result_Entropy': float(entropy(probs))})
    results['tempo_entropy'] = pl.DataFrame(tempo_entropy_rows).sort('Tempo_Quintile')

    tempo_gd_var = master.group_by('Tempo_Quintile').agg(
        pl.col('Goal_Diff').var().alias('GD_Variance')
    ).sort('Tempo_Quintile')
    results['tempo_gd_variance'] = tempo_gd_var

    tm = team_matches.with_columns((pl.col('Shots') + pl.col('Shots_Against')).alias('Match_Tempo'))
    team_tempo = tm.group_by(TEAM_SEASON_GROUP).agg([
        pl.len().alias('Matches'),
        pl.col('Match_Tempo').mean().alias('Avg_Tempo'),
        pl.col('Match_Tempo').std().alias('Tempo_Std'),
        pl.col('Points').mean().alias('PPG'),
    ]).filter(pl.col('Matches') >= min_matches)
    results['team_season_tempo'] = team_tempo

    valid = team_tempo.select(pl.col('Avg_Tempo').is_not_null() & pl.col('PPG').is_not_null()).to_series()
    if int(valid.sum()) > 20:
        r, p, n = _pearsonr(_np(team_tempo.filter(valid), 'Avg_Tempo'), _np(team_tempo.filter(valid), 'PPG'))
        results['tempo_ppg_correlation'] = {'r': r, 'p': p, 'n': n}

    smap = pl.DataFrame({'Season': list(SEASON_ORDER.keys()),
                         'Season_Num': list(SEASON_ORDER.values())})
    team_tempo = team_tempo.join(smap, on='Season', how='left').sort(['Div', 'Team', 'Season_Num'])
    team_tempo = team_tempo.with_columns([
        pl.col('Avg_Tempo').shift(-1).over(['Div', 'Team']).alias('Next_Avg_Tempo'),
        pl.col('Season_Num').shift(-1).over(['Div', 'Team']).alias('Next_Season_Num'),
        pl.col('Div').shift(-1).over(['Div', 'Team']).alias('Next_Div'),
    ])
    valid_pairs = (
        pl.col('Next_Season_Num').is_not_null() &
        (pl.col('Next_Season_Num') - pl.col('Season_Num') == 1) &
        (pl.col('Next_Div') == pl.col('Div'))
    )
    consecutive = team_tempo.filter(valid_pairs)
    v = consecutive.select(pl.col('Avg_Tempo').is_not_null() & pl.col('Next_Avg_Tempo').is_not_null()).to_series()
    if int(v.sum()) > 10:
        r, p, n = _pearsonr(_np(consecutive.filter(v), 'Avg_Tempo'), _np(consecutive.filter(v), 'Next_Avg_Tempo'))
        results['tempo_persistence'] = {'r': r, 'p': p, 'n': n}

    tm = tm.with_columns([
        (pl.col('Date').cum_count().over(TEAM_SEASON_GROUP) + 1).alias('Match_Num')
    ])
    max_match = tm.select(pl.col('Match_Num').max().over(TEAM_SEASON_GROUP)).to_series()
    tm = tm.with_columns(pl.Series('max_match', max_match))
    tm = tm.with_columns(
        pl.when(pl.col('Match_Num') <= pl.col('max_match') / 2).then(pl.lit('First')).otherwise(pl.lit('Second')).alias('Half')
    )
    half_tempo = tm.group_by(['Div', 'Team', 'Season', 'Half']).agg(
        pl.col('Match_Tempo').mean().alias('Match_Tempo')
    )
    half_pivot = half_tempo.pivot(on='Half', index=['Div', 'Team', 'Season'], values='Match_Tempo', aggregate_function='mean')
    if 'First' in half_pivot.columns and 'Second' in half_pivot.columns:
        v = half_pivot.select(pl.col('First').is_not_null() & pl.col('Second').is_not_null()).to_series()
        if int(v.sum()) > 10:
            r, p, n = _pearsonr(_np(half_pivot.filter(v), 'First'), _np(half_pivot.filter(v), 'Second'))
            results['half_season_tempo_persistence'] = {'r': r, 'p': p, 'n': n}

    valid = team_tempo.select(pl.col('Avg_Tempo').is_not_null() & pl.col('Tempo_Std').is_not_null()).to_series()
    if int(valid.sum()) > 20:
        r, p, n = _pearsonr(_np(team_tempo.filter(valid), 'Avg_Tempo'), _np(team_tempo.filter(valid), 'Tempo_Std'))
        results['tempo_mean_std_correlation'] = {'r': r, 'p': p, 'n': n}

    league_tempo = master.group_by('Div').agg([
        pl.col('Total_Shots').mean().alias('Avg_Tempo'),
        pl.col('Total_Shots').std().alias('Tempo_Std'),
        pl.col('Total_Goals').mean().alias('Avg_Goals'),
    ]).sort('Div')
    results['league_tempo'] = league_tempo

    valid = team_tempo.select(pl.col('Avg_Tempo').is_not_null()).to_series()
    if int(valid.sum()) > 50:
        tt = team_tempo.filter(valid)
        high_tempo = tt.sort('Avg_Tempo', descending=True).head(20)
        low_tempo = tt.sort('Avg_Tempo').head(20)
        style_cols = ['Div', 'Team', 'Season', 'Shots_PG', 'Shots_Against_PG', 'Shot_Accuracy',
                      'Goals_Per_Shot_OT', 'GA_Per_Opp_Shot_OT', 'Corners_PG', 'Fouls_PG']
        available = [c for c in style_cols if c in team_season.columns]
        style_only = [c for c in available if c not in ('Div', 'Team', 'Season')]
        high_tempo = high_tempo.join(team_season.select(['Div', 'Team', 'Season'] + style_only),
                                     on=['Div', 'Team', 'Season'], how='left')
        low_tempo = low_tempo.join(team_season.select(['Div', 'Team', 'Season'] + style_only),
                                   on=['Div', 'Team', 'Season'], how='left')
        results['high_tempo_teams'] = high_tempo.to_dicts()
        results['low_tempo_teams'] = low_tempo.to_dicts()

    return results


def analyze_game_state_effects(team_matches: pl.DataFrame, master: pl.DataFrame,
                               team_season: pl.DataFrame, min_matches: int = 20) -> Dict:
    """
    Question 3: Game State Effects on Match Statistics.
    """
    results: Dict = {}

    home_side = master.with_columns([
        pl.col('HomeTeam').alias('Team'), pl.lit('Home').alias('Venue'),
        pl.col('HTR').replace_strict({'H': 'W', 'D': 'D', 'A': 'L'}).alias('HT_Result'),
        (pl.col('HTHG') - pl.col('HTAG')).alias('HT_GD'),
        pl.col('FTHG').alias('GF'), pl.col('FTAG').alias('GA'),
        pl.col('HS').alias('Shots'), pl.col('HST').alias('Shots_OT'),
        pl.col('AS').alias('Shots_Ag'), pl.col('AST').alias('Shots_OT_Ag'),
        pl.col('HF').alias('Fouls'), pl.col('AF').alias('Fouls_Ag'),
        pl.col('HC').alias('Corners'), pl.col('AC').alias('Corners_Ag'),
        pl.col('HY').alias('Yellow'), pl.col('AY').alias('Yellow_Ag'),
    ])
    away_side = master.with_columns([
        pl.col('AwayTeam').alias('Team'), pl.lit('Away').alias('Venue'),
        pl.col('HTR').replace_strict({'A': 'W', 'D': 'D', 'H': 'L'}).alias('HT_Result'),
        (pl.col('HTAG') - pl.col('HTHG')).alias('HT_GD'),
        pl.col('FTAG').alias('GF'), pl.col('FTHG').alias('GA'),
        pl.col('AS').alias('Shots'), pl.col('AST').alias('Shots_OT'),
        pl.col('HS').alias('Shots_Ag'), pl.col('HST').alias('Shots_OT_Ag'),
        pl.col('AF').alias('Fouls'), pl.col('HF').alias('Fouls_Ag'),
        pl.col('AC').alias('Corners'), pl.col('HC').alias('Corners_Ag'),
        pl.col('AY').alias('Yellow'), pl.col('HY').alias('Yellow_Ag'),
    ])
    ht_data = pl.concat([home_side, away_side], how='diagonal')

    strength = team_season.select(['Div', 'Team', 'Season', 'PPG', 'Shots_PG', 'Shot_Accuracy'])
    ht_data = ht_data.join(strength, on=['Div', 'Team', 'Season'], how='left')

    state_stats_rows = []
    gb = ht_data.group_by(['HT_Result', 'Venue'])
    for key in gb:
        (ht_res, venue), g = key
        shots = g.select(pl.col('Shots')).to_series().to_numpy().astype(float)
        shots_ot = g.select(pl.col('Shots_OT')).to_series().to_numpy()
        gf = g.select(pl.col('GF')).to_series().to_numpy()
        ga = g.select(pl.col('GA')).to_series().to_numpy()
        sa = shots.copy()
        sa[sa == 0] = np.nan
        shot_acc = float(np.nanmean(shots_ot / sa))
        ppg = float(np.mean((3 * (gf > ga) + 1 * (gf == ga))))
        state_stats_rows.append({
            'HT_Result': ht_res, 'Venue': venue, 'N': g.height,
            'Shots': float(np.mean(shots)), 'Shots_OT': float(np.mean(shots_ot)),
            'Shot_Accuracy': float(shot_acc), 'Fouls': float(np.mean(g.select(pl.col('Fouls')).to_series().to_numpy())),
            'Corners': float(np.mean(g.select(pl.col('Corners')).to_series().to_numpy())),
            'Yellow': float(np.mean(g.select(pl.col('Yellow')).to_series().to_numpy())),
            'GF': float(np.mean(gf)), 'GA': float(np.mean(ga)), 'PPG': ppg,
        })
    results['state_stats'] = pl.DataFrame(state_stats_rows) if state_stats_rows else pl.DataFrame()

    trailing = ht_data.filter(pl.col('HT_Result') == 'L').with_columns(
        (-pl.col('HT_GD')).alias('Deficit')
    )
    deficit_stats_rows = []
    gb = trailing.group_by(['Deficit', 'Venue'])
    for key in gb:
        (deficit, venue), g = key
        gf = g.select(pl.col('GF')).to_series().to_numpy()
        ga = g.select(pl.col('GA')).to_series().to_numpy()
        deficit_stats_rows.append({
            'Deficit': int(deficit), 'Venue': venue, 'N': g.height,
            'Shots': float(np.mean(g.select(pl.col('Shots')).to_series().to_numpy())),
            'Shots_OT': float(np.mean(g.select(pl.col('Shots_OT')).to_series().to_numpy())),
            'Fouls': float(np.mean(g.select(pl.col('Fouls')).to_series().to_numpy())),
            'Corners': float(np.mean(g.select(pl.col('Corners')).to_series().to_numpy())),
            'Yellow': float(np.mean(g.select(pl.col('Yellow')).to_series().to_numpy())),
            'Comeback_Rate': float(np.mean(gf > ga)),
            'Draw_Rate': float(np.mean(gf == ga)),
        })
    results['deficit_stats'] = pl.DataFrame(deficit_stats_rows) if deficit_stats_rows else pl.DataFrame()

    team_state = ht_data.group_by(['Div', 'Team', 'Season', 'HT_Result', 'Venue']).agg([
        pl.len().alias('Matches'),
        pl.col('Shots').mean().alias('Shots'),
        pl.col('Shots_OT').mean().alias('Shots_OT'),
        pl.col('Fouls').mean().alias('Fouls'),
        pl.col('Corners').mean().alias('Corners'),
        pl.col('Yellow').mean().alias('Yellow'),
    ])
    team_state_pivot = team_state.pivot(on='HT_Result', index=['Div', 'Team', 'Season', 'Venue'],
                                        values=['Shots', 'Shots_OT', 'Fouls', 'Corners', 'Yellow', 'Matches'],
                                        aggregate_function='mean')

    for metric in ['Shots', 'Shots_OT', 'Fouls', 'Corners', 'Yellow']:
        lead_col = f'{metric}_W'
        trail_col = f'{metric}_L'
        if lead_col in team_state_pivot.columns and trail_col in team_state_pivot.columns:
            team_state_pivot = team_state_pivot.with_columns(
                (pl.col(trail_col) - pl.col(lead_col)).alias(f'Score_Effect_{metric}')
            )
    results['team_season_state'] = team_state_pivot

    league_effects_cols = []
    for m in ['Shots', 'Shots_OT', 'Fouls', 'Corners', 'Yellow']:
        col = f'Score_Effect_{m}'
        if col in team_state_pivot.columns:
            league_effects_cols.append(pl.col(col).mean().round(4).alias(f'Score_Effect_{m}_mean'))
    if league_effects_cols:
        league_effects = team_state_pivot.group_by('Div').agg(league_effects_cols).sort('Div')
    else:
        league_effects = pl.DataFrame()
    results['league_score_effects'] = league_effects

    if 'PPG' in team_state_pivot.columns:
        team_state_pivot = team_state_pivot.with_columns(
            pl.when(pl.col('PPG') <= pl.col('PPG').quantile(1/3).over(['Div', 'Season'])).then(pl.lit('Weak'))
            .when(pl.col('PPG') <= pl.col('PPG').quantile(2/3).over(['Div', 'Season'])).then(pl.lit('Average'))
            .otherwise(pl.lit('Strong')).alias('Strength_Tercile')
        )
        strength_effects_cols = []
        for m in ['Shots', 'Shots_OT', 'Fouls', 'Corners', 'Yellow']:
            col = f'Score_Effect_{m}'
            if col in team_state_pivot.columns:
                strength_effects_cols.append(pl.col(col).mean().round(4).alias(f'Score_Effect_{m}_mean'))
        if strength_effects_cols:
            strength_effects = team_state_pivot.group_by('Strength_Tercile').agg(strength_effects_cols).sort('Strength_Tercile')
        else:
            strength_effects = pl.DataFrame()
        results['strength_score_effects'] = strength_effects

    smap = pl.DataFrame({'Season': list(SEASON_ORDER.keys()),
                         'Season_Num': list(SEASON_ORDER.values())})
    team_state_pivot = team_state_pivot.join(smap, on='Season', how='left').sort(['Div', 'Team', 'Season_Num'])
    for metric in ['Shots', 'Shots_OT', 'Fouls', 'Corners', 'Yellow']:
        col = f'Score_Effect_{metric}'
        if col in team_state_pivot.columns:
            team_state_pivot = team_state_pivot.with_columns(
                pl.col(col).shift(-1).over(['Div', 'Team']).alias(f'Next_{col}')
            )
    team_state_pivot = team_state_pivot.with_columns([
        pl.col('Season_Num').shift(-1).over(['Div', 'Team']).alias('Next_Season_Num'),
        pl.col('Div').shift(-1).over(['Div', 'Team']).alias('Next_Div'),
    ])
    valid_pairs = (
        pl.col('Next_Season_Num').is_not_null() &
        (pl.col('Next_Season_Num') - pl.col('Season_Num') == 1) &
        (pl.col('Next_Div') == pl.col('Div'))
    )
    consecutive = team_state_pivot.filter(valid_pairs)

    persistence: Dict = {}
    for metric in ['Shots', 'Shots_OT', 'Fouls', 'Corners', 'Yellow']:
        col = f'Score_Effect_{metric}'
        nxt = f'Next_{col}'
        if col in consecutive.columns and nxt in consecutive.columns:
            v = consecutive.select(pl.col(col).is_not_null() & pl.col(nxt).is_not_null()).to_series()
            if int(v.sum()) > 10:
                r, p, n = _pearsonr(_np(consecutive.filter(v), col), _np(consecutive.filter(v), nxt))
                persistence[metric] = {'r': r, 'p': p, 'n': n}
    results['score_effect_persistence'] = persistence

    return results


def analyze_referee_characteristics(master: pl.DataFrame, team_matches: pl.DataFrame,
                                    min_matches: int = 30) -> Dict:
    """
    Question 4: Referee Characteristics and Potential Biases (English leagues only).
    """
    results: Dict = {}

    eng_master = master.filter(pl.col('Div').is_in(ENGLISH_LEAGUES)).clone()
    if eng_master.height == 0:
        results['error'] = 'No English league data available'
        return results

    eng_master = eng_master.with_columns([
        (pl.col('HY') + pl.col('AY')).alias('Total_Yellow'),
        (pl.col('HR') + pl.col('AR')).alias('Total_Red'),
        (pl.col('HF') + pl.col('AF')).alias('Total_Fouls'),
    ])

    ref_season = eng_master.group_by(['Div', 'Season', 'Referee']).agg([
        pl.len().alias('Matches'),
        ((pl.col('FTR') == 'H').mean()).alias('Home_Win_Pct'),
        ((pl.col('FTR') == 'A').mean()).alias('Away_Win_Pct'),
        ((pl.col('FTR') == 'D').mean()).alias('Draw_Pct'),
        pl.col('FTHG').mean().alias('Home_Goals'),
        pl.col('FTAG').mean().alias('Away_Goals'),
        pl.col('Total_Yellow').sum().alias('Total_Yellow'),
        pl.col('Total_Red').sum().alias('Total_Red'),
        pl.col('Total_Fouls').sum().alias('Total_Fouls'),
        pl.col('HY').mean().alias('Home_Yellow'),
        pl.col('AY').mean().alias('Away_Yellow'),
        pl.col('HR').mean().alias('Home_Red'),
        pl.col('AR').mean().alias('Away_Red'),
        pl.col('HF').mean().alias('Home_Fouls'),
        pl.col('AF').mean().alias('Away_Fouls'),
    ]).filter(pl.col('Matches') >= min_matches).with_columns([
        (pl.col('Total_Yellow') / pl.col('Matches')).alias('Yellow_PG'),
        (pl.col('Total_Red') / pl.col('Matches')).alias('Red_PG'),
        (pl.col('Total_Fouls') / pl.col('Matches')).alias('Fouls_PG'),
        (pl.col('Home_Yellow') - pl.col('Away_Yellow')).alias('Yellow_Home_Bias'),
        (pl.col('Home_Red') - pl.col('Away_Red')).alias('Red_Home_Bias'),
        (pl.col('Home_Fouls') - pl.col('Away_Fouls')).alias('Fouls_Home_Bias'),
        (pl.col('Home_Win_Pct') - pl.col('Away_Win_Pct')).alias('HA_Win_Diff'),
    ])
    results['referee_season'] = ref_season

    ref_career = ref_season.group_by('Referee').agg([
        pl.col('Season').n_unique().alias('Seasons'),
        pl.col('Matches').sum().alias('Total_Matches'),
        pl.col('Yellow_PG').mean().alias('Avg_Yellow_PG'),
        pl.col('Red_PG').mean().alias('Avg_Red_PG'),
        pl.col('Fouls_PG').mean().alias('Avg_Fouls_PG'),
        pl.col('HA_Win_Diff').mean().alias('Avg_HA_Win_Diff'),
        pl.col('Yellow_Home_Bias').mean().alias('Avg_Yellow_Home_Bias'),
        pl.col('Red_Home_Bias').mean().alias('Avg_Red_Home_Bias'),
        pl.col('Fouls_Home_Bias').mean().alias('Avg_Fouls_Home_Bias'),
        pl.col('Yellow_PG').std().alias('Std_Yellow_PG'),
        pl.col('Red_PG').std().alias('Std_Red_PG'),
        pl.col('HA_Win_Diff').std().alias('Std_HA_Win_Diff'),
    ]).filter(pl.col('Total_Matches') >= 30)
    results['referee_career'] = ref_career

    league_avg = eng_master.group_by('Div').agg([
        ((pl.col('FTR') == 'H').mean()).alias('Home_Win_Pct'),
        (pl.col('Total_Yellow').mean()).alias('Yellow_PG'),
        (pl.col('Total_Red').mean()).alias('Red_PG'),
        (pl.col('Total_Fouls').mean()).alias('Fouls_PG'),
    ]).sort('Div')
    results['league_referee_avg'] = league_avg

    ref_season = ref_season.join(league_avg, on='Div', how='left', suffix='_League')
    ref_season = ref_season.with_columns([
        (pl.col('Yellow_PG') - pl.col('Yellow_PG_League')).alias('Yellow_PG_Dev'),
        (pl.col('Red_PG') - pl.col('Red_PG_League')).alias('Red_PG_Dev'),
        (pl.col('Fouls_PG') - pl.col('Fouls_PG_League')).alias('Fouls_PG_Dev'),
    ])

    smap = pl.DataFrame({'Season': list(SEASON_ORDER.keys()),
                         'Season_Num': list(SEASON_ORDER.values())})
    ref_season = ref_season.join(smap, on='Season', how='left').sort(['Referee', 'Season_Num'])
    for metric in ['Yellow_PG', 'Red_PG', 'Fouls_PG', 'HA_Win_Diff', 'Yellow_Home_Bias', 'Red_Home_Bias']:
        ref_season = ref_season.with_columns(
            pl.col(metric).shift(-1).over('Referee').alias(f'Next_{metric}')
        )
    ref_season = ref_season.with_columns(
        pl.col('Season_Num').shift(-1).over('Referee').alias('Next_Season_Num')
    )
    consecutive = ref_season.filter(pl.col('Next_Season_Num').is_not_null() &
                                    (pl.col('Next_Season_Num') - pl.col('Season_Num') == 1))

    persistence: Dict = {}
    for metric in ['Yellow_PG', 'Red_PG', 'Fouls_PG', 'HA_Win_Diff', 'Yellow_Home_Bias', 'Red_Home_Bias']:
        nxt = f'Next_{metric}'
        if metric in consecutive.columns and nxt in consecutive.columns:
            v = consecutive.select(pl.col(metric).is_not_null() & pl.col(nxt).is_not_null()).to_series()
            if int(v.sum()) > 10:
                r, p, n = _pearsonr(_np(consecutive.filter(v), metric), _np(consecutive.filter(v), nxt))
                persistence[metric] = {'r': r, 'p': p, 'n': n}
    results['referee_persistence'] = persistence

    tm_eng = team_matches.filter(pl.col('Div').is_in(ENGLISH_LEAGUES)).clone()
    tm_eng = tm_eng.with_columns([
        (pl.col('Div') + pl.lit('_') + pl.col('Season')).alias('LeagueSeason'),
        pl.when(pl.col('Venue') == 'Home').then(1).otherwise(0).cast(pl.Int8).alias('Venue_Code'),
    ])
    ref_lookup = eng_master.select(['Div', 'Season', 'Date', 'Referee']).unique()
    tm_eng = tm_eng.join(ref_lookup, on=['Div', 'Season', 'Date'], how='left')

    valid = tm_eng.select(pl.col('Referee').is_not_null() & pl.col('Yellow').is_not_null() &
                          pl.col('Fouls').is_not_null() & pl.col('Points').is_not_null() &
                          pl.col('Venue_Code').is_not_null()).to_series()
    tm_valid = tm_eng.filter(valid)

    if tm_valid.height > 100:
        team_ppg = tm_valid.group_by(['Div', 'Team', 'Season']).agg(
            pl.col('Points').mean().alias('Team_PPG')
        )
        tm_valid = tm_valid.join(team_ppg, on=['Div', 'Team', 'Season'], how='left')
        opp_ppg = team_ppg.rename({'Team': 'Opponent', 'Team_PPG': 'Opp_PPG'})
        tm_valid = tm_valid.join(opp_ppg, on=['Div', 'Season', 'Opponent'], how='left')

        ref_counts = tm_valid.group_by('Referee').agg(pl.len().alias('rc'))
        common_refs = ref_counts.filter(pl.col('rc') >= 20).select('Referee').to_series().to_list()
        tm_valid = tm_valid.with_columns(
            pl.when(pl.col('Referee').is_in(common_refs)).then(pl.col('Referee')).otherwise(pl.lit('Other')).alias('Referee_Grouped')
        )
        ref_dummies = tm_valid.select('Referee_Grouped').to_dummies(columns='Referee_Grouped', drop_first=True)
        X = pl.concat([tm_valid.select(['Venue_Code', 'Team_PPG', 'Opp_PPG']), ref_dummies], how='horizontal').to_numpy()

        def _ols_coefs(X_df_arr, y_series_arr, mask_arr):
            X_sub = X_df_arr[mask_arr]
            y_sub = y_series_arr[mask_arr]
            Xc = np.column_stack([np.ones(X_sub.shape[0]), X_sub])
            XtX = Xc.T @ Xc
            Xty = Xc.T @ y_sub
            beta = np.linalg.solve(XtX, Xty)
            return beta[1:]

        y_yellow = tm_valid.select('Yellow').to_numpy().ravel()
        v_y = y_yellow != y_yellow  # placeholder
        v_y = (~np.isnan(y_yellow)) & (~np.isnan(X).any(axis=1))
        if int(v_y.sum()) > 100:
            coefs_y = _ols_coefs(X, y_yellow, v_y)
            ref_cols = ref_dummies.columns
            ref_coefs_y = {col: float(coefs_y[i]) for i, col in enumerate(ref_cols) if col.startswith('Referee_Grouped_')}
            results['referee_yellow_effects'] = ref_coefs_y

        y_fouls = tm_valid.select('Fouls').to_numpy().ravel()
        v_f = (~np.isnan(y_fouls)) & (~np.isnan(X).any(axis=1))
        if int(v_f.sum()) > 100:
            coefs_f = _ols_coefs(X, y_fouls, v_f)
            ref_coefs_f = {col: float(coefs_f[i]) for i, col in enumerate(ref_cols) if col.startswith('Referee_Grouped_')}
            results['referee_fouls_effects'] = ref_coefs_f

    for metric in ['Avg_Yellow_PG', 'Avg_Red_PG', 'Avg_Fouls_PG', 'Avg_HA_Win_Diff',
                   'Avg_Yellow_Home_Bias', 'Avg_Red_Home_Bias']:
        if metric in ref_career.columns:
            top = ref_career.sort(metric, descending=True).head(5).select(['Referee', metric, 'Total_Matches', 'Seasons']).to_dicts()
            bottom = ref_career.sort(metric).head(5).select(['Referee', metric, 'Total_Matches', 'Seasons']).to_dicts()
            results[f'top_{metric}'] = top
            results[f'bottom_{metric}'] = bottom

    return results


def analyze_pythagorean_residuals(team_season: pl.DataFrame, team_matches: pl.DataFrame,
                                  min_matches: int = 20) -> Dict:
    """
    Question 5: Expected Points vs Actual Points (Pythagorean Residuals).
    """
    results: Dict = {}

    ts = team_season.filter(pl.col('Matches') >= min_matches).clone()

    best_exponents: Dict = {}
    for div in ts.select(pl.col('Div').unique()).to_series().to_list():
        lg = ts.filter(pl.col('Div') == div)
        valid = lg.select(pl.col('GF').is_not_null() & pl.col('GA').is_not_null() & pl.col('PPG').is_not_null() &
                          (pl.col('GF') > 0) & (pl.col('GA') > 0)).to_series()
        if int(valid.sum()) < 20:
            best_exponents[div] = 1.35
            continue
        lg_valid = lg.filter(valid)
        best_exp = 1.35
        best_r2 = -1.0
        for exp in np.arange(1.0, 2.0, 0.05):
            gf = lg_valid.select(pl.col('GF')).to_series().to_numpy()
            ga = lg_valid.select(pl.col('GA')).to_series().to_numpy()
            ppg = lg_valid.select(pl.col('PPG')).to_series().to_numpy()
            pred_ppg = 3 * (gf ** exp) / (gf ** exp + ga ** exp)
            ss_res = float(np.sum((ppg - pred_ppg) ** 2))
            ss_tot = float(np.sum((ppg - ppg.mean()) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            if r2 > best_r2:
                best_r2 = r2
                best_exp = float(exp)
        best_exponents[div] = best_exp
    results['pythagorean_exponents'] = best_exponents

    exp_map = pl.DataFrame({'Div': list(best_exponents.keys()),
                            'Pythag_Exp': list(best_exponents.values())})
    ts = ts.join(exp_map, on='Div', how='left').with_columns(
        pl.col('Pythag_Exp').fill_null(1.35)
    )
    ts = ts.with_columns([
        (3 * (pl.col('GF') ** pl.col('Pythag_Exp')) /
         (pl.col('GF') ** pl.col('Pythag_Exp') + pl.col('GA') ** pl.col('Pythag_Exp'))).alias('Pythag_PPG'),
    ]).with_columns(
        (pl.col('PPG') - pl.col('Pythag_PPG')).alias('Pythag_Residual')
    )
    ts = ts.with_columns([
        (3 * (pl.col('GF') ** 1.35) / (pl.col('GF') ** 1.35 + pl.col('GA') ** 1.35)).alias('Pythag_PPG_Global'),
    ]).with_columns(
        (pl.col('PPG') - pl.col('Pythag_PPG_Global')).alias('Pythag_Residual_Global')
    )
    results['team_season_pythag'] = ts

    pythag_corrs: Dict = {}
    corr_metrics = ['PPG', 'GF_PG', 'GA_PG', 'GD_PG', 'Shots_PG', 'Shot_Accuracy',
                    'Goals_Per_Shot_OT', 'GA_Per_Opp_Shot_OT']
    for metric in corr_metrics:
        if metric in ts.columns:
            v = ts.select(pl.col(metric).is_not_null() & pl.col('Pythag_Residual').is_not_null()).to_series()
            if int(v.sum()) > 20:
                r, p, n = _pearsonr(_np(ts.filter(v), metric), _np(ts.filter(v), 'Pythag_Residual'))
                pythag_corrs[metric] = {'r': r, 'p': p, 'n': n}
    results['pythag_residual_correlations'] = pythag_corrs

    if 'Shot_Dominance_PG' not in ts.columns and {'Shots_PG', 'Shots_Against_PG'} <= set(ts.columns):
        ts = ts.with_columns((pl.col('Shots_PG') - pl.col('Shots_Against_PG')).alias('Shot_Dominance_PG'))
    if 'GD_PG' not in ts.columns and {'GD', 'Matches'} <= set(ts.columns):
        ts = ts.with_columns((pl.col('GD') / pl.col('Matches')).alias('GD_PG'))

    league_pythag = ts.group_by('Div').agg([
        pl.col('Pythag_Residual').mean().alias('Mean_Pythag_Residual'),
        pl.col('Pythag_Residual').std().alias('Std_Pythag_Residual'),
        (pl.col('Pythag_Residual') ** 2).mean().sqrt().alias('RMSE'),
    ]).sort('Div')
    results['league_pythag'] = league_pythag

    ts = ts.with_columns(pl.col('Season').replace(SEASON_ORDER).cast(pl.Int64).alias('Season_Num')).sort(['Div', 'Team', 'Season_Num'])
    ts = ts.with_columns([
        pl.col('Pythag_Residual').shift(-1).over(['Div', 'Team']).alias('Next_Pythag_Residual'),
        pl.col('Pythag_Residual_Global').shift(-1).over(['Div', 'Team']).alias('Next_Pythag_Residual_Global'),
        pl.col('PPG').shift(-1).over(['Div', 'Team']).alias('Next_PPG'),
        pl.col('Shot_Dominance_PG').shift(-1).over(['Div', 'Team']).alias('Next_Shot_Dom'),
        pl.col('Season_Num').shift(-1).over(['Div', 'Team']).alias('Next_Season_Num'),
        pl.col('Div').shift(-1).over(['Div', 'Team']).alias('Next_Div'),
    ])
    valid_pairs = (
        pl.col('Next_Season_Num').is_not_null() &
        (pl.col('Next_Season_Num') - pl.col('Season_Num') == 1) &
        (pl.col('Next_Div') == pl.col('Div'))
    )
    consecutive = ts.filter(valid_pairs)

    persistence: Dict = {}
    for metric in ['Pythag_Residual', 'Pythag_Residual_Global', 'PPG', 'Shot_Dominance_PG']:
        nxt = f'Next_{metric}'
        if metric in consecutive.columns and nxt in consecutive.columns:
            v = consecutive.select(pl.col(metric).is_not_null() & pl.col(nxt).is_not_null()).to_series()
            if int(v.sum()) > 10:
                r, p, n = _pearsonr(_np(consecutive.filter(v), metric), _np(consecutive.filter(v), nxt))
                persistence[metric] = {'r': r, 'p': p, 'n': n}
    results['season_to_season_persistence'] = persistence

    tm = team_matches.with_columns([
        (pl.col('Date').cum_count().over(TEAM_SEASON_GROUP) + 1).alias('Match_Num')
    ])
    max_match = tm.select(pl.col('Match_Num').max().over(TEAM_SEASON_GROUP)).to_series()
    tm = tm.with_columns(pl.Series('max_match', max_match))
    tm = tm.with_columns(
        pl.when(pl.col('Match_Num') <= pl.col('max_match') / 2).then(pl.lit('First')).otherwise(pl.lit('Second')).alias('Half')
    )
    half_pythag = tm.group_by(['Div', 'Team', 'Season', 'Half']).agg([
        pl.col('GF').sum().alias('GF'),
        pl.col('GA').sum().alias('GA'),
        pl.col('Points').sum().alias('Points'),
        pl.len().alias('Matches'),
    ])
    half_pythag = half_pythag.join(exp_map, on='Div', how='left').with_columns(
        pl.col('Pythag_Exp').fill_null(1.35)
    )
    half_pythag = half_pythag.with_columns([
        (3 * (pl.col('GF') ** pl.col('Pythag_Exp')) /
         (pl.col('GF') ** pl.col('Pythag_Exp') + pl.col('GA') ** pl.col('Pythag_Exp'))).alias('Pythag_PPG'),
        (pl.col('Points') / pl.col('Matches')).alias('Actual_PPG'),
    ]).with_columns(
        (pl.col('Actual_PPG') - pl.col('Pythag_PPG')).alias('Residual')
    )
    half_pivot = half_pythag.pivot(on='Half', index=['Div', 'Team', 'Season'], values='Residual', aggregate_function='mean')
    if 'First' in half_pivot.columns and 'Second' in half_pivot.columns:
        v = half_pivot.select(pl.col('First').is_not_null() & pl.col('Second').is_not_null()).to_series()
        if int(v.sum()) > 10:
            r, p, n = _pearsonr(_np(half_pivot.filter(v), 'First'), _np(half_pivot.filter(v), 'Second'))
            results['half_season_persistence'] = {'r': r, 'p': p, 'n': n}

    team_pythag = ts.group_by('Team').agg([
        pl.len().alias('Seasons'),
        pl.col('Pythag_Residual').mean().alias('Avg_Pythag_Residual'),
        pl.col('Pythag_Residual').std().alias('Std_Pythag_Residual'),
        pl.col('PPG').mean().alias('Avg_PPG'),
    ]).filter(pl.col('Seasons') >= 3).with_columns(
        (pl.col('Avg_Pythag_Residual') / pl.col('Std_Pythag_Residual').replace(0, None)).alias('Consistency')
    ).sort('Avg_Pythag_Residual', descending=True)
    results['team_pythag_career'] = team_pythag
    results['top_overperformers'] = team_pythag.head(15).to_dicts()
    results['top_underperformers'] = team_pythag.sort('Avg_Pythag_Residual').head(15).to_dicts()

    valid = ts.select(pl.col('Shot_Dominance_PG').is_not_null() & pl.col('GD_PG').is_not_null() & pl.col('Pythag_Residual').is_not_null()).to_series()
    if int(valid.sum()) > 50:
        ts_v = ts.filter(valid)
        y = ts_v.select('Pythag_Residual').to_numpy().ravel()
        X1 = ts_v.select('GD_PG').to_numpy()
        r2_1 = LinearRegression().fit(X1, y).score(X1, y)
        X2 = ts_v.select(['GD_PG', 'Shot_Dominance_PG']).to_numpy()
        r2_2 = LinearRegression().fit(X2, y).score(X2, y)
        X3 = ts_v.select('Shot_Dominance_PG').to_numpy()
        r2_3 = LinearRegression().fit(X3, y).score(X3, y)
        results['incremental_r2'] = {
            'gd_only': float(r2_1), 'gd_plus_shotdom': float(r2_2),
            'shotdom_only': float(r2_3), 'inc_shotdom_over_gd': float(r2_2 - r2_1),
            'inc_gd_over_shotdom': float(r2_2 - r2_3),
        }

    ts = ts.with_columns((pl.col('Div') + pl.lit('_') + pl.col('Season')).alias('LeagueSeason'))
    ls_mean = ts.group_by('LeagueSeason').agg(pl.col('Pythag_Residual').mean().alias('ls_mean'))
    ls_std = ts.group_by('LeagueSeason').agg(pl.col('Pythag_Residual').std().alias('ls_std'))
    ts = ts.join(ls_mean, on='LeagueSeason', how='left').join(ls_std, on='LeagueSeason', how='left')
    ts = ts.with_columns(((pl.col('Pythag_Residual') - pl.col('ls_mean')) / pl.col('ls_std').replace(0, None)).alias('Pythag_Residual_z'))
    ts = ts.with_columns(pl.col('Pythag_Residual_z').shift(-1).over(['Div', 'Team']).alias('Next_Pythag_Residual_z'))
    consecutive = ts.filter(valid_pairs)
    v = consecutive.select(pl.col('Pythag_Residual_z').is_not_null() & pl.col('Next_Pythag_Residual_z').is_not_null()).to_series()
    if int(v.sum()) > 10:
        r, p, n = _pearsonr(_np(consecutive.filter(v), 'Pythag_Residual_z'), _np(consecutive.filter(v), 'Next_Pythag_Residual_z'))
        results['z_scored_persistence'] = {'r': r, 'p': p, 'n': n}

    return results


def run_fresh_insights(master: pl.DataFrame, team_matches: pl.DataFrame, team_season: pl.DataFrame) -> Dict:
    """Run all 5 fresh insight analyses."""
    print("  [1/5] Draw Specialists and Match-Outcome Structure...")
    draws = analyze_draw_specialists(team_matches, team_season)

    print("  [2/5] Match Tempo and Statistical Structure...")
    tempo = analyze_match_tempo(team_matches, master, team_season)

    print("  [3/5] Game State Effects on Match Statistics...")
    game_state = analyze_game_state_effects(team_matches, master, team_season)

    print("  [4/5] Referee Characteristics and Potential Biases...")
    referee = analyze_referee_characteristics(master, team_matches)

    print("  [5/5] Expected Points vs Actual Points (Pythagorean Residuals)...")
    pythag = analyze_pythagorean_residuals(team_season, team_matches)

    return {
        'draw_specialists': draws,
        'match_tempo': tempo,
        'game_state_effects': game_state,
        'referee_characteristics': referee,
        'pythagorean_residuals': pythag,
    }
