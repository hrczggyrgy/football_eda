"""
Feature engineering module.
Creates efficiency ratios, rolling form metrics, temporal features,
and team-season aggregate metrics from team-perspective match data.

Migrated to Polars: all calculations use vectorized Polars expressions and
window operations (`.over()`, `shift`, `rolling_*`, `cum_sum`). No row-wise
Python loops and no pandas.
"""

from __future__ import annotations

from typing import List, Optional

import polars as pl

# Columns to compute rolling features for
ROLLING_COLS = [
    'GF', 'GA', 'GD', 'Points', 'Shots', 'Shots_OT',
    'Shots_Against', 'Shots_OT_Against',
    'Shot_Accuracy', 'Goals_Per_Shot', 'Goals_Per_Shot_OT',
    'Opp_Shot_Accuracy', 'GA_Per_Opp_Shot', 'GA_Per_Opp_Shot_OT',
    'Corners', 'Corners_Against', 'Yellow', 'Red', 'Fouls',
]

# Grouping key for team-season calculations
TEAM_SEASON_GROUP = ['Div', 'Team', 'Season']

# Season code -> numeric order (chronological)
SEASON_ORDER = {
    '1920': 0, '2021': 1, '2122': 2, '2223': 3,
    '2324': 4, '2425': 5, '2526': 6, '2627': 7,
}


def add_efficiency_ratios(df: pl.DataFrame) -> pl.DataFrame:
    """Add match-level efficiency ratios with zero-denominator handling.

    Ratios with a zero denominator are null (not zero), preserving the
    missing-vs-zero distinction.
    """
    # Points from full-time result
    result_map = pl.when(pl.col('FT_Result') == 'W').then(3).when(
        pl.col('FT_Result') == 'D'
    ).then(1).otherwise(0).cast(pl.Int64)

    out = df.with_columns([
        result_map.alias('Points'),
        # Attacking efficiency
        pl.when(pl.col('Shots') > 0)
        .then(pl.col('Shots_OT') / pl.col('Shots'))
        .otherwise(None)
        .alias('Shot_Accuracy'),
        pl.when(pl.col('Shots') > 0)
        .then(pl.col('GF') / pl.col('Shots'))
        .otherwise(None)
        .alias('Goals_Per_Shot'),
        pl.when(pl.col('Shots_OT') > 0)
        .then(pl.col('GF') / pl.col('Shots_OT'))
        .otherwise(None)
        .alias('Goals_Per_Shot_OT'),
        (pl.col('Shots') - pl.col('Shots_Against')).alias('Shot_Dominance'),
        (pl.col('Shots_OT') - pl.col('Shots_OT_Against')).alias('SOT_Dominance'),
        # Defensive efficiency
        pl.when(pl.col('Shots_Against') > 0)
        .then(pl.col('Shots_OT_Against') / pl.col('Shots_Against'))
        .otherwise(None)
        .alias('Opp_Shot_Accuracy'),
        pl.when(pl.col('Shots_Against') > 0)
        .then(pl.col('GA') / pl.col('Shots_Against'))
        .otherwise(None)
        .alias('GA_Per_Opp_Shot'),
        pl.when(pl.col('Shots_OT_Against') > 0)
        .then(pl.col('GA') / pl.col('Shots_OT_Against'))
        .otherwise(None)
        .alias('GA_Per_Opp_Shot_OT'),
        # Discipline (per-match counts; per-game derived at season level)
        pl.col('Fouls').alias('Fouls_Per_Game'),
        pl.col('Yellow').alias('Yellow_Per_Game'),
        pl.col('Red').alias('Red_Per_Game'),
        (pl.col('Yellow') + pl.col('Red')).alias('Cards_Per_Game'),
    ])
    return out


def add_rolling_features(df: pl.DataFrame,
                         window: int = 5,
                         min_periods: int = 3) -> pl.DataFrame:
    """Add lagged rolling and expanding features per team-season.

    All historical rolling calculations remain strictly lagged: the current
    match never contributes to its own pre-match rolling features. This is
    achieved by shifting each column by one match *within* the team-season
    before applying the window (equivalent to the pandas ``.shift(1)`` logic).

    Rest days for season openers is null (genuinely unknown, not invented).
    """
    out = df.sort(['Div', 'Team', 'Season', 'Date'])

    # Match number within team-season. Polars cum_count() is 1-based, which
    # directly equals pandas groupby.cumcount() + 1.
    out = out.with_columns(
        pl.col('Date').cum_count().over(TEAM_SEASON_GROUP).alias('Match_Num')
    )

    # Rest days: days since previous match within team-season; null for opener.
    out = out.with_columns(
        pl.col('Date').diff().over(TEAM_SEASON_GROUP).dt.total_days().alias('Rest_Days')
    )

    shift_cols = [c for c in ROLLING_COLS if c in out.columns]
    if not shift_cols:
        return out

    # Shift every rolling column by one match within the team-season in a
    # single pass. This is the lookahead-leakage guard.
    shifted_exprs = [pl.col(c).shift(1).over(TEAM_SEASON_GROUP).alias(f'{c}_lag1')
                     for c in shift_cols]
    out = out.with_columns(shifted_exprs)

    for col in shift_cols:
        lagged = pl.col(f'{col}_lag1')

        # Rolling mean/std on SHIFTED data (no leakage). ``.over()`` must be
        # outermost so the window is computed per team-season (calling
        # ``rolling_mean().over(...)`` partitions correctly; the reverse order
        # rolls globally across groups).
        out = out.with_columns([
            lagged.rolling_mean(window, min_samples=min_periods)
                .over(TEAM_SEASON_GROUP).alias(f'{col}_L{window}'),
            lagged.rolling_std(window, min_samples=min_periods)
                .over(TEAM_SEASON_GROUP).alias(f'{col}_L{window}_std'),
        ])

        # Cumulative expanding mean on SHIFTED data (lagged). Computed as the
        # cumulative sum of valid (non-null) lagged values divided by the
        # running count of valid values. First match -> null (no prior data).
        valid_count = lagged.is_not_null().cast(pl.Int64).cum_sum().over(TEAM_SEASON_GROUP)
        out = out.with_columns(
            (lagged.fill_null(0).cum_sum().over(TEAM_SEASON_GROUP)
             / valid_count.replace(0, None)).alias(f'{col}_Cum')
        )

    # Drop the temporary lagged helper columns.
    out = out.drop([f'{c}_lag1' for c in shift_cols])
    return out


def add_opponent_strength(df: pl.DataFrame) -> pl.DataFrame:
    """Add opponent's season-to-date PPG before this match (lagged).

    The opponent's pre-match cumulative PPG is looked up via a Polars join on
    (Div, Season, Date, Opponent). No row-wise Python operations.
    """
    out = df.sort(['Div', 'Team', 'Season', 'Date'])
    grp = TEAM_SEASON_GROUP

    # Pre-match (lagged) cumulative PPG for each team in each match.
    # cum_sum - current points = points earned in all prior matches;
    # cum_count - 1 = number of prior matches (polars cum_count is 1-based).
    points = pl.col('Points')
    prior_points = (points.cum_sum().over(grp) - points).alias('prior_points')
    prior_matches = (pl.col('Points').cum_count().over(grp) - 1).alias('prior_matches')

    out = out.with_columns([prior_points, prior_matches])

    pre_match_ppg = pl.when(pl.col('prior_matches') > 0).then(
        pl.col('prior_points') / pl.col('prior_matches')
    ).otherwise(None).cast(pl.Float64).alias('PreMatch_PPG')
    out = out.with_columns(pre_match_ppg)

    # Build lookup keyed by (Div, Season, Date, Team) -> that team's pre-match PPG.
    lookup = out.select([
        pl.col('Div'), pl.col('Season'), pl.col('Date'),
        pl.col('Team').alias('Opponent'),
        pl.col('PreMatch_PPG').alias('Opp_PPG_Cum'),
    ])

    # maintain_order='left' keeps the left frame (chronologically sorted) row
    # order so previously computed per-row columns stay aligned to their match.
    out = out.join(lookup, on=['Div', 'Season', 'Date', 'Opponent'], how='left',
                   maintain_order='left')

    out = out.drop(['prior_points', 'prior_matches', 'PreMatch_PPG'])
    return out


def compute_team_season_metrics(team_matches: pl.DataFrame,
                                min_matches: int = 20) -> pl.DataFrame:
    """Aggregate team-match data to team-season level with derived metrics.

    Preserves metric-specific valid-match counts, missing-value semantics
    (all-missing sums stay null), minimum match thresholds, and per-game
    calculations.
    """
    metric_cols = ['GF', 'GA', 'GD', 'Points', 'Shots', 'Shots_OT',
                   'Shots_Against', 'Shots_OT_Against',
                   'Yellow', 'Red', 'Fouls', 'Corners', 'Corners_Against']
    metric_cols = [c for c in metric_cols if c in team_matches.columns]

    # Total matches per team-season (all rows, regardless of missingness).
    match_counts = team_matches.group_by(TEAM_SEASON_GROUP).agg(
        pl.len().alias('Matches')
    )

    # Valid (non-null) match counts per metric.
    valid_exprs = [pl.col(c).count().alias(f'{c}_Valid') for c in metric_cols]
    valid_counts = team_matches.group_by(TEAM_SEASON_GROUP).agg(valid_exprs)

    # Sums per metric. Polars sum returns null for all-null groups, preserving
    # the missing-vs-zero distinction (equivalent to pandas skipna=False).
    sum_exprs = [pl.col(c).sum().alias(c) for c in metric_cols]
    sums = team_matches.group_by(TEAM_SEASON_GROUP).agg(sum_exprs)

    team_season = match_counts.join(sums, on=TEAM_SEASON_GROUP, how='inner').join(
        valid_counts, on=TEAM_SEASON_GROUP, how='inner'
    )

    team_season = team_season.filter(pl.col('Matches') >= min_matches)

    # Per-game rates using metric-specific valid-match exposure.
    def pg(num: str, valid: str) -> pl.Expr:
        return (pl.col(num) / pl.col(valid).replace(0, None)).alias(f'{num.split("_")[0]}_PG') \
            if num != 'Points' else (pl.col(num) / pl.col('Matches')).alias('PPG')

    rate_exprs = []
    rate_exprs.append((pl.col('Points') / pl.col('Matches')).alias('PPG'))
    if 'GD' in team_season.columns:
        rate_exprs.append((pl.col('GD') / pl.col('Matches')).alias('GD_PG'))
    for num, valid in [
        ('GF', 'GF_Valid'), ('GA', 'GA_Valid'), ('Shots', 'Shots_Valid'),
        ('Shots_OT', 'Shots_OT_Valid'), ('Shots_Against', 'Shots_Against_Valid'),
    ]:
        if num in team_season.columns and valid in team_season.columns:
            base = num.replace('Shots_OT', 'Shots_OT').replace('Shots_Against', 'Shots_Against')
            rate_exprs.append(
                (pl.col(num) / pl.col(valid).replace(0, None)).alias(f'{base}_PG')
            )
    # Yellow/Red/Fouls per game
    for num, valid in [('Yellow', 'Yellow_Valid'), ('Red', 'Red_Valid'), ('Fouls', 'Fouls_Valid')]:
        if num in team_season.columns and valid in team_season.columns:
            rate_exprs.append(
                (pl.col(num) / pl.col(valid).replace(0, None)).alias(f'{num}_PG')
            )

    if rate_exprs:
        team_season = team_season.with_columns(rate_exprs)

    # Efficiency ratios (only where denominators > 0).
    def ratio(num: str, den: str, name: str, cond_col: str) -> Optional[pl.Expr]:
        if num in team_season.columns and den in team_season.columns and cond_col in team_season.columns:
            return pl.when(pl.col(cond_col) > 0).then(pl.col(num) / pl.col(den)).otherwise(None).alias(name)
        return None

    ratio_exprs = [
        ratio('Shots_OT', 'Shots', 'Shot_Accuracy', 'Shots_Valid'),
        ratio('GF', 'Shots', 'Goals_Per_Shot', 'Shots_Valid'),
        ratio('GF', 'Shots_OT', 'Goals_Per_Shot_OT', 'Shots_OT_Valid'),
        ratio('Shots_OT_Against', 'Shots_Against', 'Opp_Shot_Accuracy', 'Shots_Against_Valid'),
        ratio('GA', 'Shots_Against', 'GA_Per_Opp_Shot', 'Shots_Against_Valid'),
        ratio('GA', 'Shots_OT_Against', 'GA_Per_Opp_Shot_OT', 'Shots_OT_Against_Valid'),
    ]
    ratio_exprs = [e for e in ratio_exprs if e is not None]
    if ratio_exprs:
        team_season = team_season.with_columns(ratio_exprs)

    # Cards per game
    if {'Yellow_PG', 'Red_PG'} <= set(team_season.columns):
        team_season = team_season.with_columns(
            (pl.col('Yellow_PG') + pl.col('Red_PG')).alias('Cards_PG')
        )

    return team_season


def compute_consistency_metrics(team_matches: pl.DataFrame,
                                min_matches: int = 10) -> pl.DataFrame:
    """Compute within-season consistency metrics (CV of match-level stats).

    Rolling 5-game points std is computed on shifted (lagged) points to avoid
    leakage.
    """
    out = team_matches.sort(['Div', 'Team', 'Season', 'Date'])

    # Rolling 5-game points std (lagged by one match).
    out = out.with_columns(
        pl.col('Points').shift(1).over(TEAM_SEASON_GROUP).alias('points_lag1')
    )
    out = out.with_columns(
        pl.col('points_lag1').over(TEAM_SEASON_GROUP).rolling_std(5, min_samples=3)
        .alias('Points_Rolling5_Std')
    )

    consistency = out.group_by(TEAM_SEASON_GROUP).agg([
        pl.col('Points').std().alias('Points_Std'),
        pl.col('Points_Rolling5_Std').mean().alias('Points_Rolling5_Std_Mean'),
        pl.col('GF').std().alias('GF_Std'),
        pl.col('GA').std().alias('GA_Std'),
        pl.col('Shots').std().alias('Shots_Std'),
        pl.col('GD').std().alias('GD_Std'),
        pl.col('Points').mean().alias('PPG'),
        pl.col('GF').mean().alias('GF_PG'),
        pl.col('GA').mean().alias('GA_PG'),
        pl.len().alias('Matches'),
    ])

    consistency = consistency.filter(pl.col('Matches') >= min_matches)

    # Coefficient of variation
    consistency = consistency.with_columns([
        (pl.col('Points_Std') / pl.col('PPG')).alias('Points_CV'),
        (pl.col('GF_Std') / pl.col('GF_PG')).alias('GF_CV'),
        (pl.col('GA_Std') / pl.col('GA_PG')).alias('GA_CV'),
    ])

    return consistency


def add_season_order(team_season: pl.DataFrame) -> pl.DataFrame:
    """Add numeric season order for consecutive season analysis."""
    mapping = pl.DataFrame({
        'Season': list(SEASON_ORDER.keys()),
        'Season_Num': list(SEASON_ORDER.values()),
    })
    out = team_season.join(mapping, on='Season', how='left')
    out = out.sort(['Div', 'Team', 'Season_Num'])
    return out


def add_next_season_metrics(team_season: pl.DataFrame,
                            metrics: List[str] = ['GPS', 'GPSOT', 'GAOS', 'PPG']) -> pl.DataFrame:
    """Add next-season metrics for persistence analysis."""
    out = team_season.sort(['Div', 'Team', 'Season_Num'])

    # Ensure required base metrics exist
    if 'GPS' not in out.columns and {'GF', 'Shots'} <= set(out.columns):
        out = out.with_columns((pl.col('GF') / pl.col('Shots')).alias('GPS'))
    if 'GPSOT' not in out.columns and {'GF', 'Shots_OT'} <= set(out.columns):
        out = out.with_columns((pl.col('GF') / pl.col('Shots_OT')).alias('GPSOT'))
    if 'GAOS' not in out.columns and {'GA', 'Shots_Against'} <= set(out.columns):
        out = out.with_columns((pl.col('GA') / pl.col('Shots_Against')).alias('GAOS'))

    for metric in metrics:
        if metric in out.columns:
            out = out.with_columns(
                pl.col(metric).shift(-1).over(['Div', 'Team']).alias(f'Next_{metric}')
            )

    return out


def cluster_team_profiles(team_season: pl.DataFrame,
                          features: List[str],
                          n_clusters: int = 5,
                          random_state: int = 42) -> pl.DataFrame:
    """K-means clustering of team-season profiles.

    Uses scikit-learn on NumPy arrays extracted from Polars; cluster labels
    are written back as a Polars column aligned on row position.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    out = team_season
    available = [f for f in features if f in out.columns]
    if not available:
        out = out.with_columns(pl.lit(None).cast(pl.Float64).alias('Cluster'))
        return out

    # Row mask of complete cases across the feature columns.
    complete = pl.lit(True)
    for f in available:
        complete = complete & pl.col(f).is_not_null()
    profile_data = out.filter(complete)

    if profile_data.height < n_clusters:
        out = out.with_columns(pl.lit(None).cast(pl.Float64).alias('Cluster'))
        return out

    scaler = StandardScaler()
    profile_scaled = scaler.fit_transform(profile_data.select(available).to_numpy())

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(profile_scaled)

    # Write labels back aligned to the original row order via a row index join.
    labels_aligned = pl.DataFrame({
        '__idx': list(range(profile_data.height)),
        'Cluster': labels.astype(float),
    })
    profile_indexed = profile_data.with_row_index('__idx').select(['__idx']).join(
        labels_aligned, on='__idx', how='left'
    )
    out = out.with_row_index('__idx').join(
        profile_indexed.select(['__idx', 'Cluster']), on='__idx', how='left'
    ).drop('__idx')

    return out


# ---------------------------------------------------------------------------
# Lagged team strength (pre-match cumulative PPG with shrinkage)
# ---------------------------------------------------------------------------


def add_lagged_team_strength(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add genuinely lagged pre-match strength estimate.
    Uses cumulative PPG up to but not including the current match.
    Early-season initialization uses PRIOR SEASON average (not current season),
    falling back to historical league average if no prior season exists.
    """
    out = df.sort(['Div', 'Team', 'Season', 'Date'])
    grp = TEAM_SEASON_GROUP

    points = pl.col('Points')
    points_cum = points.cum_sum().over(grp)
    matches_cum = points.cum_count().over(grp)

    points_cum_lagged = points_cum - points
    matches_cum_lagged = matches_cum - 1  # number of prior matches

    out = out.with_columns([
        points_cum_lagged.alias('points_cum_lagged'),
        matches_cum_lagged.alias('matches_cum_lagged'),
    ])

    out = out.with_columns(
        pl.when(pl.col('matches_cum_lagged') > 0)
        .then(pl.col('points_cum_lagged') / pl.col('matches_cum_lagged'))
        .otherwise(None)
        .cast(pl.Float64)
        .alias('PreMatch_PPG')
    )
    out = out.with_columns(pl.col('matches_cum_lagged').alias('Matches_Cum_Lagged'))

    # Prior-season final PPG lookup (consecutive seasons only)
    season_num_expr = pl.col('Season').replace(SEASON_ORDER).cast(pl.Int64)
    team_season_final = out.group_by(grp).agg(pl.col('Points').mean().alias('Final_PPG'))
    team_season_final = team_season_final.with_columns(
        pl.col('Season').replace(SEASON_ORDER).cast(pl.Int64).alias('Season_Num')
    ).sort(['Team', 'Season_Num'])

    team_season_final = team_season_final.with_columns([
        pl.col('Final_PPG').shift(1).over('Team').alias('Prev_Final_PPG'),
        pl.col('Season_Num').shift(1).over('Team').alias('Prev_Season_Num'),
    ]).with_columns(
        (pl.col('Season_Num') - pl.col('Prev_Season_Num') == 1).alias('Consecutive')
    )

    prior_ppg_lookup = team_season_final.filter(pl.col('Consecutive')).select([
        pl.col('Div'), pl.col('Team'), pl.col('Season'),
        pl.col('Prev_Final_PPG').alias('Prior_Season_PPG'),
    ])

    out = out.join(prior_ppg_lookup, on=['Div', 'Team', 'Season'], how='left',
                   maintain_order='left')

    # Historical league average per league-season (shifted to prior seasons only)
    hist_avg = out.group_by(['Div', 'Season']).agg(pl.col('Points').mean().alias('League_Avg_PPG'))
    hist_avg = hist_avg.sort(['Div', 'Season']).with_columns(
        pl.col('League_Avg_PPG').shift(1).over('Div').alias('Hist_League_Avg_PPG')
    ).select(['Div', 'Season', 'Hist_League_Avg_PPG'])

    out = out.join(hist_avg, on=['Div', 'Season'], how='left', maintain_order='left')

    # Shrink early-season estimates toward prior-season PPG, then historical league avg
    has_prior = pl.col('Prior_Season_PPG').is_not_null()
    has_hist = pl.col('Hist_League_Avg_PPG').is_not_null()
    mc = pl.col('Matches_Cum_Lagged')

    out = out.with_columns(pl.col('PreMatch_PPG').alias('PreMatch_PPG_Shrunk'))

    # Case 1: prior season PPG + few current matches
    mask1 = has_prior & (mc < 5)
    out = out.with_columns(
        pl.when(mask1).then(
            (pl.col('PreMatch_PPG') * mc + pl.col('Prior_Season_PPG') * (5 - mc)) / 5
        ).otherwise(pl.col('PreMatch_PPG_Shrunk')).alias('PreMatch_PPG_Shrunk')
    )

    # Case 2: historical league avg + few current matches
    mask2 = (~has_prior) & has_hist & (mc < 5)
    out = out.with_columns(
        pl.when(mask2).then(
            (pl.col('PreMatch_PPG') * mc + pl.col('Hist_League_Avg_PPG') * (5 - mc)) / 5
        ).otherwise(pl.col('PreMatch_PPG_Shrunk')).alias('PreMatch_PPG_Shrunk')
    )

    # Case 3: first match ever - leave as NaN

    return out.drop([
        'points_cum_lagged', 'matches_cum_lagged', 'Matches_Cum_Lagged',
        'Prior_Season_PPG', 'Hist_League_Avg_PPG', 'PreMatch_PPG',
    ])


def build_promotion_relegation_transitions(team_season: pl.DataFrame) -> pl.DataFrame:
    """
    Build promotion/relegation transitions by sorting each team chronologically.
    Require adjacent seasons. Distinguish same-division and cross-division.
    Infer promotion/relegation only where observed league hierarchy supports it.
    """
    DIVISION_HIERARCHY = {
        'E0': 0, 'E1': 1, 'E2': 2, 'E3': 3,
        'SP1': 0, 'SP2': 1,
        'I1': 0, 'I2': 1,
        'D1': 0, 'D2': 1,
        'F1': 0, 'F2': 1,
    }

    ts = team_season.clone()
    ts = ts.with_columns(
        pl.col('Season').replace(SEASON_ORDER).cast(pl.Int64).alias('Season_Num')
    ).sort(['Team', 'Season_Num'])

    ts = ts.with_columns([
        pl.col('Season_Num').shift(1).over('Team').alias('Prev_Season_Num'),
        pl.col('Div').shift(1).over('Team').alias('Prev_Div'),
        pl.col('PPG').shift(1).over('Team').alias('Prev_PPG'),
        pl.col('Shots_PG').shift(1).over('Team').alias('Prev_Shots_PG') if 'Shots_PG' in ts.columns else pl.lit(None).alias('Prev_Shots_PG'),
        pl.col('Shots_Against_PG').shift(1).over('Team').alias('Prev_Shots_Against_PG') if 'Shots_Against_PG' in ts.columns else pl.lit(None).alias('Prev_Shots_Against_PG'),
        pl.col('Goals_Per_Shot_OT').shift(1).over('Team').alias('Prev_GPSOT') if 'Goals_Per_Shot_OT' in ts.columns else pl.lit(None).alias('Prev_GPSOT'),
        pl.col('GA_Per_Opp_Shot_OT').shift(1).over('Team').alias('Prev_GAOSOT') if 'GA_Per_Opp_Shot_OT' in ts.columns else pl.lit(None).alias('Prev_GAOSOT'),
    ])

    ts = ts.with_columns([
        (pl.col('Season_Num') - pl.col('Prev_Season_Num')).alias('Season_Gap'),
    ]).with_columns([
        (pl.col('Season_Gap') == 1).alias('Consecutive'),
        (pl.col('Div') != pl.col('Prev_Div')).alias('Div_Changed'),
        pl.col('Div').replace(DIVISION_HIERARCHY).cast(pl.Int64).alias('Div_Tier'),
        pl.col('Prev_Div').replace(DIVISION_HIERARCHY).cast(pl.Int64).alias('Prev_Div_Tier'),
    ])

    ts = ts.with_columns(
        pl.when(pl.col('Consecutive') & pl.col('Div_Changed') & (pl.col('Div_Tier') < pl.col('Prev_Div_Tier')))
        .then(pl.lit('Promotion'))
        .when(pl.col('Consecutive') & pl.col('Div_Changed') & (pl.col('Div_Tier') > pl.col('Prev_Div_Tier')))
        .then(pl.lit('Relegation'))
        .when(pl.col('Consecutive') & ~pl.col('Div_Changed'))
        .then(pl.lit('Same_Division'))
        .otherwise(pl.lit('Other'))
        .alias('Transition_Type')
    )

    return ts
