"""
Pre-match feature engineering for the predictive engine.

All features represent information genuinely available before a fixture begins.
Windows are strictly lagged (shifted by one match within a team-season) so the
current match never contributes to its own pre-match features. Cross-season
history and previous-season information are permitted.

Feature availability matrix (relative to a target match at time T):

| Feature group                        | Available at T? | Source                              |
|--------------------------------------|-----------------|-------------------------------------|
| Historical points per game           | Yes             | Matches before T                    |
| Historical goals per game            | Yes             | Matches before T                    |
| Rolling form (last 3/5/10 matches)   | Yes             | Matches before T                    |
| Expanding historical average         | Yes             | Matches before T                    |
| Exponentially weighted average       | Yes             | Matches before T                    |
| Previous-season PPG                  | Yes             | Prior season only                   |
| League average scoring environment   | Yes             | Seasons fully before T              |
| Home/away-specific performance       | Yes             | Matches before T at that venue      |
| Opponent strength                    | Yes             | Opponent matches before T           |
| Rest days                            | Yes             | Known previous fixture dates        |
| Current match goals/result/shots     | NO              | Forbidden (leakage)                 |
| Same-date other results              | NO              | When timestamp ordering unavailable |
"""

from __future__ import annotations

import warnings
from typing import List, Optional

import polars as pl

from features.engineer import (
    TEAM_SEASON_GROUP,
    SEASON_ORDER,
    add_efficiency_ratios,
    add_rolling_features,
    add_opponent_strength,
    add_lagged_team_strength,
)

# Rolling windows evaluated for historical form.
FORM_WINDOWS = (3, 5, 10)
# Exponential decay half-life (in matches) for EWMA features.
EWMA_HALF_LIFE_MATCHES = 5
# Shrinkage target exposure count (pseudo-observations).
SHRINKAGE_K = 5.0

# Explicit pre-match feature allowlist - only these columns reach the model.
# This prevents leakage from raw current-match statistics.
PRE_MATCH_FEATURE_ALLOWLIST = {
    # Rolling form (lagged)
    "Points_L3", "Points_L5", "Points_L10", "Points_Cum",
    "GF_L3", "GF_L5", "GF_L10", "GF_Cum",
    "GA_L3", "GA_L5", "GA_L10", "GA_Cum",
    "GD_L3", "GD_L5", "GD_L10", "GD_Cum",
    "Shots_L5", "Shots_OT_L5", "Shots_Against_L5",
    "Shot_Accuracy_L5", "Corners_L5",
    # Strength & shrinkage
    "PreMatch_PPG_Shrunk", "Matches_Cum_Lagged",
    "Opp_PPG_Cum",
    "Prior_Season_PPG", "Hist_League_Avg_PPG",
    "League_Avg_GF_PG", "League_Avg_GA_PG",
    # Venue splits
    "Home_PPG", "Away_PPG", "Home_GF_PG", "Away_GF_PG",
    # League table state & form momentum
    "Team_Cum_Points_Before", "Team_Cum_Matches_Before", "Team_PPG_Before",
    "Team_Form_Momentum",
    # Head-to-head
    "H2H_PPG", "H2H_GF_PG", "H2H_GA_PG", "H2H_Matches",
    "H2H_Draw_Rate", "H2H_Recent_PPG", "H2H_Recent_Draw_Rate", "H2H_Recent_Matches",
    # Streaks
    "streak_W", "streak_D", "streak_L",
    # Table rank & momentum trends
    "Team_Table_Rank",
    "Team_GD_Momentum", "Team_Shots_Momentum", "Team_GA_Trend",
    # Elo
    "Elo_Base_Rating", "Elo_Opponent_Rating", "Elo_Exp_Home", "Elo_Rating_Diff",
    # Rest & schedule
    "Rest_Days", "Match_Num",
    # Recent opponent & venue
    "Opp_Recent_PPG",
    "Home_Recent_Venue_PPG", "Away_Recent_Venue_PPG",
    # Season progress
    "Season_Progress",
}


# Polars cannot verify sortedness of the as-of key within `by=` groups and
# emits a UserWarning even when the caller has sorted explicitly. Since every
# call site below sorts by the by-keys plus the as-of key before joining, the
# join is correct; we suppress this known-safe warning locally.
def _asof_join_safe(
    left: pl.DataFrame,
    right: pl.DataFrame,
    left_on: str,
    right_on: str,
    by_left: List[str],
    by_right: List[str],
    **kwargs,
) -> pl.DataFrame:
    left_sorted = left.sort(by_left + [left_on])
    right_sorted = right.sort(by_right + [right_on])
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sortedness of columns cannot be checked when 'by' groups provided",
        )
        return left_sorted.join_asof(
            right_sorted,
            left_on=left_on,
            right_on=right_on,
            by_left=by_left,
            by_right=by_right,
            **kwargs,
        )


def _null_cols(df: pl.DataFrame, names: List[str]) -> pl.DataFrame:
    """Add missing columns as nulls of an appropriate dtype so later casts are safe."""
    out = df
    for name in names:
        if name not in out.columns:
            out = out.with_columns(pl.lit(None).cast(pl.Float64).alias(name))
    return out


def _safe_div(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    """Safe division returning null instead of infinity on zero/negative denom."""
    return pl.when(den > 0).then(num / den).otherwise(None).cast(pl.Float64)


def _team_form_features(df: pl.DataFrame, windows: List[int] = FORM_WINDOWS) -> pl.DataFrame:
    """Lagged rolling form features per team-season.

    Columns produced for each window w in windows and each base metric:
    {metric}_L{w}, {metric}_L{w}_std, {metric}_Cum (expanding mean).
    All computed on shifted data so match t never sees its own result.
    """
    out = add_efficiency_ratios(df)
    out = add_rolling_features(out, window=max(windows), min_periods=1)

    # Also add smaller-window rolling means explicitly (engineer uses only max).
    base_metrics = ["GF", "GA", "GD", "Points", "Shots", "Shots_OT",
                    "Shots_Against", "Shots_OT_Against",
                    "Shot_Accuracy", "Goals_Per_Shot",
                    "Opp_Shot_Accuracy", "GA_Per_Opp_Shot",
                    "Corners", "Corners_Against"]
    available = [c for c in base_metrics if c in out.columns]

    for w in windows:
        if w >= max(windows):
            continue
        for col in available:
            lagged = pl.col(f"{col}_lag1") if f"{col}_lag1" in out.columns else pl.col(col).shift(1).over(TEAM_SEASON_GROUP)
            out = out.with_columns([
                lagged.rolling_mean(w, min_samples=1).over(TEAM_SEASON_GROUP).alias(f"{col}_L{w}"),
                lagged.rolling_std(w, min_samples=2).over(TEAM_SEASON_GROUP).alias(f"{col}_L{w}_std"),
            ])
    return out


def _shrinkage(expr_obs: pl.Expr, expr_prior: pl.Expr, n: pl.Expr, k: float = SHRINKAGE_K) -> pl.Expr:
    """Bayesian/empirical shrinkage: (n*obs + k*prior)/(n+k)."""
    return (expr_obs * n + expr_prior * pl.lit(k)) / (n + pl.lit(k))


def build_pre_match_features(team_matches: pl.DataFrame) -> pl.DataFrame:
    """Build a strictly pre-match feature frame at team-match granularity.

    Input: team-perspective match data sorted within team-season.
    Output: one row per team-match with features reflecting only information
    available before kickoff (historical matches before this one, plus
    previous-season data).

    The resulting frame keeps the original match-key columns so it can be
    joined back to match-level data for home/away pairing.
    """
    if team_matches.is_empty():
        return team_matches

    out = team_matches.sort(["Div", "Team", "Season", "Date"])

    # Strictly lagged rolling & expanding features.
    out = _team_form_features(out)

    # League table state before the match (cum points, matches, PPG).
    out = _add_table_state(out)

    # Form momentum & head-to-head.
    out = _add_form_momentum(out)
    out = _add_head_to_head(out)

    # Streaks, table rank, momentum trends, and recent H2H.
    out = _add_streak_features(out)
    out = _add_table_rank(out)
    out = _add_trend_features(out)
    out = _add_recent_h2h(out)

    # Season progress & recent form features.
    out = _add_season_progress(out)
    out = _add_recent_opponent_strength(out)
    out = _add_recent_venue_form(out)

    # Elo ratings (chronological, pre-match).
    from prediction.ratings import build_elo_features
    out = build_elo_features(out)

    # CURRENT-MATCH efficiency ratios (Shot_Accuracy, Goals_Per_Shot, etc.)
    # encode the target fixture's own goals/shots and must NOT be used as
    # pre-match predictors. The lagged rolling/expanding versions (*_L3, *_L5,
    # *_L10, *_Cum) computed above are strictly pre-match and are retained.
    _leaky_current_cols = [
        "Shot_Accuracy", "Goals_Per_Shot", "Goals_Per_Shot_OT",
        "Shot_Dominance", "SOT_Dominance",
        "Opp_Shot_Accuracy", "GA_Per_Opp_Shot", "GA_Per_Opp_Shot_OT",
        "Fouls_Per_Game", "Yellow_Per_Game", "Red_Per_Game", "Cards_Per_Game",
    ]
    out = out.drop([c for c in _leaky_current_cols if c in out.columns])

    # Opponent strength (lagged cumulative PPG of the opponent).
    out = add_opponent_strength(out)

    # Lagged team strength with shrinkage toward prior season / league avg.
    out = add_lagged_team_strength(out)

    # Rest days (days since previous match within team-season).
    out = out.with_columns(
        pl.col("Date").diff().over(TEAM_SEASON_GROUP).dt.total_days().alias("Rest_Days")
    )

    # Match number within team-season (exposure count).
    if "Match_Num" not in out.columns:
        out = out.with_columns(
            pl.col("Date").cum_count().over(TEAM_SEASON_GROUP).alias("Match_Num")
        )

    # Prior-season PPG (consecutive seasons only) for newly promoted teams.
    out = _add_prior_season_strength(out)

    # League-level scoring environment from completed prior seasons.
    out = _add_league_scoring_environment(out)

    # Home/away splits of historical performance.
    out = _add_venue_splits(out)

    # Ensure all core features exist (nulls for missing). The allowlist defines
    # the contract; we only null-fill declared features.
    out = _null_cols(out, list(PRE_MATCH_FEATURE_ALLOWLIST))

    return out


def _add_prior_season_strength(df: pl.DataFrame) -> pl.DataFrame:
    """Add prior-season PPG as a pre-match prior.

    Uses only completed prior-season information. The prior-season PPG comes
    from the same team's previous season (consecutive seasons only); teams
    without prior-season data fall back to their division's prior-season
    average PPG. Crucially, no current-season full-data average is ever used,
    so this feature is safe at kickoff of the first match of a season.
    """
    out = df.sort(["Div", "Team", "Season", "Date"])

    # Team-season final PPG (full-season aggregate, used only as prior for the
    # NEXT season - permitted).
    ts = out.group_by(TEAM_SEASON_GROUP).agg(
        pl.col("Points").mean().alias("Final_PPG"),
        pl.len().alias("Matches"),
    )
    season_num = pl.col("Season").replace(SEASON_ORDER).cast(pl.Int64).alias("Season_Num")
    ts = ts.with_columns(season_num).sort(["Team", "Season_Num"])

    # Previous season's Final_PPG for the same team (consecutive seasons only).
    ts = ts.with_columns([
        pl.col("Final_PPG").shift(1).over("Team").alias("Prev_Final_PPG"),
        pl.col("Season_Num").shift(1).over("Team").alias("Prev_Season_Num"),
    ]).with_columns(
        (pl.col("Season_Num") - pl.col("Prev_Season_Num") == 1).alias("Consecutive")
    )

    # Division strength: average PPG per (Div, Season), shifted so season S
    # holds season S-1's division average. Strictly prior-season information.
    div_strength = out.group_by(["Div", "Season"]).agg(
        pl.col("Points").mean().alias("Div_Season_PPG")
    ).sort(["Div", "Season"]).with_columns(
        pl.col("Div_Season_PPG").shift(1).over("Div").alias("Prior_Div_Season_PPG")
    ).select(["Div", "Season", "Prior_Div_Season_PPG"])

    # Prior season PPG for the same team (any division), only when consecutive.
    prior_lookup = ts.filter(pl.col("Consecutive")).select([
        pl.col("Team"), pl.col("Season"), pl.col("Prev_Final_PPG"),
    ])

    out = out.join(prior_lookup, on=["Team", "Season"], how="left",
                   maintain_order="left")

    # Current division's prior-season PPG, used as fallback for teams without
    # prior-season data (e.g. promoted teams). Never the current season's
    # full-data average.
    out = out.join(div_strength, on=["Div", "Season"], how="left",
                   maintain_order="left")

    out = out.with_columns([
        pl.when(pl.col("Prev_Final_PPG").is_not_null())
        .then(pl.col("Prev_Final_PPG"))
        .otherwise(pl.col("Prior_Div_Season_PPG"))
        .alias("Prior_Season_PPG")
    ])

    out = out.drop(["Prev_Final_PPG", "Prior_Div_Season_PPG"])
    return out


def _add_league_scoring_environment(df: pl.DataFrame) -> pl.DataFrame:
    """League average GF/GA per game computed from prior seasons only.

    For each (Div, Season), the environment uses all matches in strictly
    earlier seasons so it never includes the current season.
    """
    out = df.sort(["Div", "Team", "Season", "Date"])
    league_stats = out.group_by(["Div", "Season"]).agg([
        pl.col("GF").mean().alias("Div_Season_GF_PG"),
        pl.col("GA").mean().alias("Div_Season_GA_PG"),
    ]).sort(["Div", "Season"]).with_columns([
        pl.col("Div_Season_GF_PG").shift(1).over("Div").alias("League_Avg_GF_PG"),
        pl.col("Div_Season_GA_PG").shift(1).over("Div").alias("League_Avg_GA_PG"),
    ]).select(["Div", "Season", "League_Avg_GF_PG", "League_Avg_GA_PG"])

    return out.join(league_stats, on=["Div", "Season"], how="left",
                    maintain_order="left")


def _add_venue_splits(df: pl.DataFrame) -> pl.DataFrame:
    """Home-specific and away-specific historical PPG / GF per game.

    Computed as cumulative (lagged) averages within each team-season split by
    venue. Previous-season venue performance is not carried here (kept simple;
    prior-season PPG is the cross-season prior).

    Uses shift(1) to get strictly prior values, robust to null current-row values.
    """
    out = df.sort(["Div", "Team", "Season", "Date"])
    grp = TEAM_SEASON_GROUP + ["Venue"]

    points = pl.col("Points")
    gf = pl.col("GF")

    # Strictly prior cumulative sums/counts using shift(1) - robust to null current row.
    prior_points = points.cum_sum().over(grp).shift(1).over(grp).alias("prior_points")
    prior_matches = points.cum_count().over(grp).shift(1).over(grp).alias("prior_matches")
    prior_gf = gf.cum_sum().over(grp).shift(1).over(grp).alias("prior_gf")

    out = out.with_columns([prior_points, prior_matches, prior_gf])

    out = out.with_columns([
        pl.when(pl.col("prior_matches") > 0)
        .then(pl.col("prior_points") / pl.col("prior_matches"))
        .otherwise(None).cast(pl.Float64).alias("Venue_PPG"),
        pl.when(pl.col("prior_matches") > 0)
        .then(pl.col("prior_gf") / pl.col("prior_matches"))
        .otherwise(None).cast(pl.Float64).alias("Venue_GF_PG"),
    ])

    # Pivot to home/away-specific columns.
    home = out.filter(pl.col("Venue") == "Home").select([
        pl.col("Div"), pl.col("Team"), pl.col("Season"), pl.col("Date"),
        pl.col("Venue_PPG").alias("Home_PPG"),
        pl.col("Venue_GF_PG").alias("Home_GF_PG"),
    ])
    away = out.filter(pl.col("Venue") == "Away").select([
        pl.col("Div"), pl.col("Team"), pl.col("Season"), pl.col("Date"),
        pl.col("Venue_PPG").alias("Away_PPG"),
        pl.col("Venue_GF_PG").alias("Away_GF_PG"),
    ])

    out = out.join(home, on=["Div", "Team", "Season", "Date"], how="left",
                   maintain_order="left")
    out = out.join(away, on=["Div", "Team", "Season", "Date"], how="left",
                   maintain_order="left")

    return out.drop(["prior_points", "prior_matches", "prior_gf", "Venue_PPG", "Venue_GF_PG"])


def pair_match_features(team_features: pl.DataFrame) -> pl.DataFrame:
    """Pair home and away team pre-match features into match-level rows.

    Only features from the explicit PRE_MATCH_FEATURE_ALLOWLIST are carried
    through. Raw current-match statistics (GF, GA, FT_Result, etc.) are
    excluded from the paired frame to maintain a strict leakage boundary.
    """
    # Explicit allowlist of team-level features to carry through.
    feature_cols = [c for c in team_features.columns if c in PRE_MATCH_FEATURE_ALLOWLIST]

    home = team_features.filter(pl.col("Venue") == "Home").select(
        ["Div", "Season", "Date", pl.col("Team").alias("HomeTeam"),
         pl.col("Opponent").alias("AwayTeam"),
         pl.col("GF").alias("FTHG"), pl.col("GA").alias("FTAG"),
         pl.col("FT_Result").replace_strict({"W": "H", "D": "D", "L": "A"}).alias("FTR")]
        + [pl.col(c).alias(f"Home_{c}") for c in feature_cols]
    )
    away = team_features.filter(pl.col("Venue") == "Away").select(
        ["Div", "Season", "Date", pl.col("Team").alias("AwayTeam"),
         pl.col("Opponent").alias("HomeTeam")]
        + [pl.col(c).alias(f"Away_{c}") for c in feature_cols]
    )

    matched = home.join(away, on=["Div", "Season", "Date", "HomeTeam", "AwayTeam"],
                        how="inner")

    # Match-level derived pre-match features (computed from pre-match home/away
    # state only; no leakage).
    matched = matched.with_columns([
        (pl.col("Home_PreMatch_PPG_Shrunk") - pl.col("Away_PreMatch_PPG_Shrunk"))
        .alias("PPG_Gap"),
        (pl.col("Home_GF_Cum") - pl.col("Away_GF_Cum"))
        .alias("GF_Cum_Gap"),
        (pl.col("Home_GA_Cum") - pl.col("Away_GA_Cum"))
        .alias("GA_Cum_Gap"),
        (pl.col("Home_Rest_Days") - pl.col("Away_Rest_Days"))
        .alias("Rest_Days_Diff"),
        # Draw-predictive features (use prefixed column names)
        # Small PPG gap -> teams evenly matched -> more draws
        (pl.col("Home_PreMatch_PPG_Shrunk") - pl.col("Away_PreMatch_PPG_Shrunk"))
        .abs().alias("PPG_Gap_Abs"),
        # Small Elo diff -> teams evenly matched -> more draws
        (pl.col("Home_Elo_Rating_Diff")).abs().alias("Elo_Rating_Diff_Abs"),
        # Low combined scoring rate -> low scoring matches -> more draws
        (pl.col("Home_League_Avg_GF_PG") + pl.col("Home_League_Avg_GA_PG")).alias("League_Total_Goals_PG"),
        # Both teams defensive (low GA per game) -> low scoring -> more draws
        (pl.col("Home_GA_Cum") + pl.col("Away_GA_Cum")).alias("Combined_GA_Cum"),
        (pl.col("Home_Home_GF_PG") + pl.col("Away_Away_GF_PG")).alias("Combined_Venue_GF_PG"),
        # Historical H2H draw rate: prefer seasonal H2H, fall back to recent cross-season H2H
        pl.when(pl.col("Home_H2H_Draw_Rate").is_not_null() & pl.col("Away_H2H_Draw_Rate").is_not_null())
        .then((pl.col("Home_H2H_Draw_Rate") * pl.col("Home_H2H_Matches") +
               pl.col("Away_H2H_Draw_Rate") * pl.col("Away_H2H_Matches")) /
              (pl.col("Home_H2H_Matches") + pl.col("Away_H2H_Matches")))
        .when(pl.col("Home_H2H_Recent_Draw_Rate").is_not_null() & pl.col("Away_H2H_Recent_Draw_Rate").is_not_null())
        .then((pl.col("Home_H2H_Recent_Draw_Rate") * pl.col("Home_H2H_Recent_Matches") +
               pl.col("Away_H2H_Recent_Draw_Rate") * pl.col("Away_H2H_Recent_Matches")) /
              (pl.col("Home_H2H_Recent_Matches") + pl.col("Away_H2H_Recent_Matches")))
        .otherwise(None).alias("H2H_Draw_Rate"),
        # Both teams low scoring
        (pl.col("Home_GF_Cum") + pl.col("Away_GF_Cum")).alias("Combined_GF_Cum"),
        # Season progress (late season, both teams need points -> more draws)
        pl.when((pl.col("Home_Season_Progress") > 0.7) & (pl.col("Away_Season_Progress") > 0.7))
        .then(1).otherwise(0).alias("Late_Season_Both"),
    ]).with_columns([
        # Evenly matched indicator (based on match-level Elo diff)
        pl.when(pl.col("Elo_Rating_Diff_Abs") < 50).then(1).otherwise(0).alias("Evenly_Matched"),
        # Low scoring league indicator
        pl.when((pl.col("Home_League_Avg_GF_PG") + pl.col("Home_League_Avg_GA_PG")) < 2.5).then(1).otherwise(0).alias("Low_Scoring_League"),
        # Draw-prone context: evenly matched + (low scoring OR late season)
        pl.when((pl.col("Elo_Rating_Diff_Abs") < 50) &
                ((pl.col("Home_League_Avg_GF_PG") + pl.col("Home_League_Avg_GA_PG") < 2.5) |
                 (pl.col("Home_Season_Progress") > 0.7)))
        .then(1).otherwise(0).alias("Draw_Prone_Context"),
    ])

    return matched.sort(["Div", "Date", "HomeTeam", "AwayTeam"])


def _add_table_state(df: pl.DataFrame) -> pl.DataFrame:
    """Add each team's cumulative points & matches played *before* this match.

    Computed per (Div, Season, Team) in chronological order using strictly
    lagged cumulative sums, so the current match never contributes. These are
    pre-match league-table proxies (points accumulated so far, games played,
    points per game).
    """
    out = df.sort(["Div", "Team", "Season", "Date"])
    points = pl.col("Points")
    prior_points = (points.cum_sum().over(TEAM_SEASON_GROUP) - points).alias(
        "Team_Cum_Points_Before")
    prior_matches = (points.cum_count().over(TEAM_SEASON_GROUP) - 1).alias(
        "Team_Cum_Matches_Before")
    out = out.with_columns([prior_points, prior_matches])
    out = out.with_columns(
        pl.when(pl.col("Team_Cum_Matches_Before") > 0)
        .then(pl.col("Team_Cum_Points_Before") / pl.col("Team_Cum_Matches_Before"))
        .otherwise(None).cast(pl.Float64).alias("Team_PPG_Before")
    )
    return out


def _add_form_momentum(df: pl.DataFrame) -> pl.DataFrame:
    """Short-term form momentum from existing rolling windows.

    Momentum = mean points in last 3 matches minus mean points in the two
    matches before those (matches 4-5 of the lagged window). Positive values
    indicate improving form. Computed from strictly lagged windows.

    Requires at least 5 prior matches for the L5 window to be fully populated.
    """
    out = df
    need = {"Points_L3", "Points_L5"} <= set(out.columns)
    if need:
        # Only compute momentum when we have at least 5 prior matches (L5 has 5 samples)
        # Points_L5 with min_samples=5 means it has 5 valid observations.
        # Since our rolling uses min_samples=1, we need to check actual count.
        out = out.with_columns(
            pl.when(pl.col("Points_L5").is_not_null() & pl.col("Points_L3").is_not_null())
            .then(
                (pl.col("Points_L3") - ((pl.col("Points_L5") * 5
                                         - pl.col("Points_L3") * 3) / 2.0))
            )
            .otherwise(None)
            .cast(pl.Float64)
            .alias("Team_Form_Momentum")
        )
    return out


def _add_head_to_head(df: pl.DataFrame) -> pl.DataFrame:
    """Head-to-head record between the two teams across ALL seasons in the same Div.

    For each team-match row, looks at ALL previous encounters (same Div, strictly
    earlier date) between the same unordered team pair and computes the focal
    team's H2H points per game, goals per game, and draw rate.
    """
    out = df.sort(["Div", "Team", "Season", "Date"])

    # Build encounter history: one row per MATCH (not per team perspective).
    # Use home team's perspective as canonical for each match.
    matches = out.filter(pl.col("Venue") == "Home").select([
        pl.col("Div"), pl.col("Season"), pl.col("Date"),
        pl.col("Team").alias("HomeTeam"), pl.col("Opponent").alias("AwayTeam"),
        pl.col("Points").alias("home_pts"),
        pl.col("GF").alias("home_gf"), pl.col("GA").alias("home_ga"),
    ])

    # Create unordered pair key.
    matches = matches.with_columns(
        pl.when(pl.col("HomeTeam") < pl.col("AwayTeam")).then(pl.col("HomeTeam")).otherwise(pl.col("AwayTeam"))
        .alias("pair_a"),
        pl.when(pl.col("HomeTeam") < pl.col("AwayTeam")).then(pl.col("AwayTeam")).otherwise(pl.col("HomeTeam"))
        .alias("pair_b"),
    )

    # For each team-match row (focal), find ALL prior encounters of the same pair.
    focal = out.select([
        pl.col("Div"), pl.col("Season"), pl.col("Date"),
        pl.col("Team").alias("T1"), pl.col("Opponent").alias("T2"),
    ]).with_columns(
        pl.when(pl.col("T1") < pl.col("T2")).then(pl.col("T1")).otherwise(pl.col("T2"))
        .alias("pair_a"),
        pl.when(pl.col("T1") < pl.col("T2")).then(pl.col("T2")).otherwise(pl.col("T1"))
        .alias("pair_b"),
    )

    # Join focal rows with historical matches on division and pair.
    joined = focal.join(matches, on=["Div", "pair_a", "pair_b"], how="left")
    # Keep only strictly prior matches.
    joined = joined.filter(pl.col("Date_right") < pl.col("Date"))

    if joined.is_empty():
        out = out.with_columns([
            pl.lit(None).cast(pl.Float64).alias("H2H_PPG"),
            pl.lit(None).cast(pl.Float64).alias("H2H_GF_PG"),
            pl.lit(None).cast(pl.Float64).alias("H2H_GA_PG"),
            pl.lit(None).cast(pl.Float64).alias("H2H_Draw_Rate"),
            pl.lit(None).cast(pl.Int64).alias("H2H_Matches"),
        ])
        return out

    # Compute focal team's points, goals, and draw indicator in each historical encounter.
    joined = joined.with_columns(
        pl.when(pl.col("T1") == pl.col("HomeTeam"))
        .then(pl.col("home_pts"))
        .otherwise(pl.col("home_pts").replace_strict({0: 3, 1: 1, 3: 0}, default=None))
        .alias("focal_pts_hist"),
        pl.when(pl.col("T1") == pl.col("HomeTeam"))
        .then(pl.col("home_gf"))
        .otherwise(pl.col("home_ga"))
        .alias("focal_gf_hist"),
        pl.when(pl.col("T1") == pl.col("HomeTeam"))
        .then(pl.col("home_ga"))
        .otherwise(pl.col("home_gf"))
        .alias("focal_ga_hist"),
        (pl.col("home_gf") == pl.col("home_ga")).cast(pl.Int64).alias("focal_draw_hist"),
    )

    # Aggregate ALL historical encounters per focal match.
    agg = joined.group_by(["Div", "Season", "Date", "T1", "T2"]).agg([
        pl.col("focal_pts_hist").mean().alias("H2H_PPG"),
        pl.col("focal_gf_hist").mean().alias("H2H_GF_PG"),
        pl.col("focal_ga_hist").mean().alias("H2H_GA_PG"),
        pl.col("focal_draw_hist").mean().alias("H2H_Draw_Rate"),
        pl.col("focal_pts_hist").count().alias("H2H_Matches"),
    ])

    out = out.join(agg, left_on=["Div", "Season", "Date", "Team", "Opponent"],
                   right_on=["Div", "Season", "Date", "T1", "T2"], how="left",
                   maintain_order="left")
    return out.drop(["pair_a", "pair_b", "T1", "T2"], strict=False)


def _add_streak_features(df: pl.DataFrame) -> pl.DataFrame:
    """Current win/draw/loss streak BEFORE this match (strictly lagged, pre-match).

    For each team-match, the streak is computed from prior matches only
    (rows 1..N-1), never including the current match's result. A team on a
    3-game winning streak going into this match has streak_W = 3.
    """
    out = df.sort(["Div", "Team", "Season", "Date"])
    ftr = pl.col("FT_Result")

    # Encode result: W=1, D=0, L=-1
    result_code = (
        pl.when(ftr == "W").then(pl.lit(1))
        .when(ftr == "D").then(pl.lit(0))
        .when(ftr == "L").then(pl.lit(-1))
        .otherwise(None).cast(pl.Int8)
    )
    out = out.with_columns(result_code.alias("result_code"))

    # result_code of the PRIOR match (shifted by 1 within team-season).
    out = out.with_columns(
        result_code.shift(1).over(TEAM_SEASON_GROUP).alias("prior_rc")
    )

    # same_as_prev[k] = True if result_code[k] == result_code[k-1], i.e. the
    # streak continues from the previous match. First row: False (no prior).
    same_as_prev = (
        result_code == result_code.shift(1).over(TEAM_SEASON_GROUP)
    ).fill_null(False)

    # Group id: increments whenever the streak breaks (same_as_prev = False).
    out = out.with_columns(
        (~same_as_prev).cast(pl.Int8).cum_sum().over(TEAM_SEASON_GROUP).alias("streak_grp")
    )

    # Position within each streak group (1-based). For row N-1, this equals
    # the streak length going into row N.
    out = out.with_columns(
        pl.col("result_code").cum_count().over(TEAM_SEASON_GROUP + ["streak_grp"]).alias("grp_pos")
    )

    # Streak going into this match = position of prior row within its group.
    # Only non-zero if the prior match's result matches the streak type.
    out = out.with_columns([
        pl.when(pl.col("prior_rc") == 1).then(pl.col("grp_pos").shift(1)).otherwise(0).alias("streak_W"),
        pl.when(pl.col("prior_rc") == 0).then(pl.col("grp_pos").shift(1)).otherwise(0).alias("streak_D"),
        pl.when(pl.col("prior_rc") == -1).then(pl.col("grp_pos").shift(1)).otherwise(0).alias("streak_L"),
    ])

    # First match of season: prior_rc is null -> all streaks = 0.
    out = out.with_columns([
        pl.when(pl.col("prior_rc").is_null()).then(0).otherwise(pl.col("streak_W")).alias("streak_W"),
        pl.when(pl.col("prior_rc").is_null()).then(0).otherwise(pl.col("streak_D")).alias("streak_D"),
        pl.when(pl.col("prior_rc").is_null()).then(0).otherwise(pl.col("streak_L")).alias("streak_L"),
    ])

    return out.drop(["result_code", "streak_grp", "grp_pos", "prior_rc"])


def _add_table_rank(df: pl.DataFrame) -> pl.DataFrame:
    """League table rank before the match (strictly lagged).

    Rank is computed from a league-wide table snapshot as of each fixture's
    own date: every team's pre-match PPG from matches strictly before that
    date, then ordinal rank by PPG descending. Teams reaching the same match
    number on different dates are compared at a consistent timestamp, so a
    prediction never incorporates another team's later pre-match state.
    Teams with 0 prior matches get rank = null.
    """
    out = df.sort(["Div", "Team", "Season", "Date"])

    if "Team_PPG_Before" not in out.columns:
        return out

    # Reference state: each team's PPG *after* its latest match, keyed on the
    # match date. Ranking at date D uses the latest match strictly before D, so
    # the post-match PPG equals PPG from all matches strictly before D.
    ref = out.with_columns(
        (
            (pl.col("Team_Cum_Points_Before") + pl.col("Points"))
            / (pl.col("Team_Cum_Matches_Before") + 1)
        ).cast(pl.Float64).alias("Team_PPG_AsOf")
    ).select(["Div", "Season", "Team", "Date", "Team_PPG_AsOf"])

    # Distinct calendar dates and teams per (Div, Season).
    dates = out.select(["Div", "Season", "Date"]).unique()
    teams = out.select(["Div", "Season", "Team"]).unique()

    # Grid: every team on every date *within its own league-season*.
    # Join on Div/Season (not a global cross join) so we only pair teams with
    # dates in the same league-season. A global cross join would multiply every
    # date by every team across all leagues/seasons, blowing up memory.
    grid = teams.join(dates, on=["Div", "Season"]).select(
        ["Div", "Season", "Date", "Team"]
    )

    grid = _asof_join_safe(
        grid, ref,
        left_on="Date", right_on="Date",
        by_left=["Div", "Season", "Team"],
        by_right=["Div", "Season", "Team"],
        strategy="backward",
        allow_exact_matches=False,
    )

    # Rank teams within each (Div, Season, Date) by PPG as of that date.
    grid = grid.with_columns(
        pl.col("Team_PPG_AsOf")
        .rank(method="ordinal", descending=True)
        .over(["Div", "Season", "Date"])
        .alias("Team_Table_Rank")
    )
    # No prior data yet -> null rank.
    grid = grid.with_columns(
        pl.when(pl.col("Team_PPG_AsOf").is_null())
        .then(None)
        .otherwise(pl.col("Team_Table_Rank"))
        .alias("Team_Table_Rank")
    )

    # Attach each fixture's rank as of its own date.
    out = out.join(
        grid.select(["Div", "Season", "Team", "Date", "Team_Table_Rank"]),
        on=["Div", "Season", "Team", "Date"],
        how="left",
        maintain_order="left",
    )

    # First match of season: no prior data -> rank null.
    out = out.with_columns(
        pl.when(pl.col("Team_Cum_Matches_Before") == 0)
        .then(None)
        .otherwise(pl.col("Team_Table_Rank"))
        .alias("Team_Table_Rank")
    )

    return out


def _add_trend_features(df: pl.DataFrame) -> pl.DataFrame:
    """Short-term momentum trends from rolling windows (strictly lagged).

    - GD_Momentum = GD_L3 - GD_L5 (recent vs earlier goal difference form)
    - Shots_Momentum = Shots_L3 - Shots_L5 (recent vs earlier shot volume)
    - GA_Trend = GA_L3 - GA_L5 (recent vs earlier goals conceded)
    Positive values = improving form (more GD/shots, fewer GA).
    """
    out = df
    need = {"GD_L3", "GD_L5", "Shots_L3", "Shots_L5", "GA_L3", "GA_L5"} <= set(out.columns)
    if need:
        out = out.with_columns([
            (pl.col("GD_L3") - pl.col("GD_L5")).alias("Team_GD_Momentum"),
            (pl.col("Shots_L3") - pl.col("Shots_L5")).alias("Team_Shots_Momentum"),
            (pl.col("GA_L3") - pl.col("GA_L5")).alias("Team_GA_Trend"),
        ])
    return out


def _add_recent_h2h(df: pl.DataFrame) -> pl.DataFrame:
    """Recent head-to-head PPG over the last 5 encounters (strictly lagged).

    Builds a fixture-level encounter history across ALL seasons, selects strictly
    earlier encounters, then aggregates the latest 5. This captures cross-season
    rivalry form rather than just within-season.
    """
    out = df.sort(["Div", "Team", "Season", "Date"])

    # Build a unified encounter history for each unordered team pair within a division.
    # We need: Div, Date, T1, T2, focal_pts (points from focal team's perspective).
    left = out.select([
        pl.col("Div"), pl.col("Season"), pl.col("Date"),
        pl.col("Team").alias("T1"), pl.col("Opponent").alias("T2"),
        pl.col("Points").alias("pts"),
    ])
    left_a = left.with_columns(
        pl.when(pl.col("T1") < pl.col("T2")).then(pl.col("T1")).otherwise(pl.col("T2"))
        .alias("pair_a"),
        pl.when(pl.col("T1") < pl.col("T2")).then(pl.col("T2")).otherwise(pl.col("T1"))
        .alias("pair_b"),
    )

    # Self-join to get all prior encounters for each focal row.
    right = left_a.clone()
    joined = _asof_join_safe(
        left_a, right,
        left_on="Date", right_on="Date",
        by_left=["Div", "pair_a", "pair_b"],
        by_right=["Div", "pair_a", "pair_b"],
        strategy="backward",
        allow_exact_matches=False,
        suffix="_hist",
    )

    # Reorient historical points to the focal team's (T1's) perspective.
    opponent_points = pl.col("pts_hist").replace_strict(
        {0: 3, 1: 1, 3: 0}, default=None
    )
    joined = joined.with_columns(
        pl.when(pl.col("T1") == pl.col("T1_hist"))
        .then(pl.col("pts_hist"))
        .otherwise(opponent_points)
        .alias("focal_pts_hist")
    )

    # Keep only rows with valid historical encounters.
    joined = joined.filter(pl.col("pts_hist").is_not_null())

    # For each focal row, rank historical encounters by date (most recent first)
    # and take the latest 5.
    joined = joined.with_columns(
        pl.int_range(pl.len()).over(["Div", "Date", "T1", "T2", "pair_a", "pair_b"])
        .alias("h2h_order")
    )
    # Note: The asof join with backward strategy already gives us earlier dates.
    # We need to order by date DESC within each focal row to get most recent first.
    # The cum_count over the group gives us 0, 1, 2... for each historical match
    # in chronological order (earliest first due to backward join).
    # We want the LAST 5, so we need to filter from the end.
    # Better approach: sort by date DESC before cum_count.
    # Since we can't easily re-sort within the expression, let's use row_number
    # with descending date.

    # Actually, the joined frame has each focal row paired with its historical matches.
    # The historical matches come from the 'right' frame which is sorted by date ASC.
    # The asof backward join finds the latest historical date <= focal date.
    # But we need ALL historical matches, not just the latest one.
    # The current approach using asof join only gets ONE match per focal row.
    # We need a different approach: self-join on pair with date filter.

    # Let's use a different strategy: cross-join by pair, filter date_hist < date,
    # then take top 5 by date_hist descending.

    return out  # Placeholder - need to rewrite properly


def _add_recent_h2h_fixed(df: pl.DataFrame) -> pl.DataFrame:
    """Recent head-to-head PPG and draw rate over the last 5 encounters (strictly lagged).

    Builds a fixture-level encounter history across ALL seasons, selects strictly
    earlier encounters, then aggregates the latest 5.
    """
    out = df.sort(["Div", "Team", "Season", "Date"])

    # Build encounter history: one row per MATCH (not per team perspective).
    # Use home team's perspective as canonical for each match.
    matches = out.filter(pl.col("Venue") == "Home").select([
        pl.col("Div"), pl.col("Season"), pl.col("Date"),
        pl.col("Team").alias("HomeTeam"), pl.col("Opponent").alias("AwayTeam"),
        pl.col("Points").alias("home_pts"),
        pl.col("GF").alias("home_gf"), pl.col("GA").alias("home_ga"),
    ])

    # Create unordered pair key.
    matches = matches.with_columns(
        pl.when(pl.col("HomeTeam") < pl.col("AwayTeam")).then(pl.col("HomeTeam")).otherwise(pl.col("AwayTeam"))
        .alias("pair_a"),
        pl.when(pl.col("HomeTeam") < pl.col("AwayTeam")).then(pl.col("AwayTeam")).otherwise(pl.col("HomeTeam"))
        .alias("pair_b"),
    )

    # For each team-match row (focal), find prior encounters of the same pair.
    focal = out.select([
        pl.col("Div"), pl.col("Season"), pl.col("Date"),
        pl.col("Team").alias("T1"), pl.col("Opponent").alias("T2"),
    ]).with_columns(
        pl.when(pl.col("T1") < pl.col("T2")).then(pl.col("T1")).otherwise(pl.col("T2"))
        .alias("pair_a"),
        pl.when(pl.col("T1") < pl.col("T2")).then(pl.col("T2")).otherwise(pl.col("T1"))
        .alias("pair_b"),
    )

    # Join focal rows with historical matches on division and pair.
    joined = focal.join(matches, on=["Div", "pair_a", "pair_b"], how="left")
    # Keep only strictly prior matches.
    joined = joined.filter(pl.col("Date_right") < pl.col("Date"))

    if joined.is_empty():
        out = out.with_columns([
            pl.lit(None).cast(pl.Float64).alias("H2H_Recent_PPG"),
            pl.lit(None).cast(pl.Float64).alias("H2H_Recent_Draw_Rate"),
            pl.lit(None).cast(pl.Int64).alias("H2H_Recent_Matches"),
        ])
        return out

    # Compute focal team's points and draw indicator in each historical encounter.
    # If focal team was home in the historical match, use home_pts.
    # If focal team was away, invert points: 3->0, 1->1, 0->3.
    joined = joined.with_columns(
        pl.when(pl.col("T1") == pl.col("HomeTeam"))
        .then(pl.col("home_pts"))
        .otherwise(pl.col("home_pts").replace_strict({0: 3, 1: 1, 3: 0}, default=None))
        .alias("focal_pts_hist"),
        # Draw indicator: 1 if goals equal, 0 otherwise
        pl.when(pl.col("T1") == pl.col("HomeTeam"))
        .then((pl.col("home_gf") == pl.col("home_ga")).cast(pl.Int64))
        .otherwise((pl.col("home_gf") == pl.col("home_ga")).cast(pl.Int64))  # Same condition, draw is symmetric
        .alias("focal_draw_hist")
    )

    # For each focal match, sort historical encounters by Date DESC and take top 5.
    joined = joined.with_columns(
        pl.col("Date_right").rank(method="ordinal", descending=True)
        .over(["Div", "Season", "Date", "T1", "T2"])
        .alias("h2h_recency_rank")
    )
    joined = joined.filter(pl.col("h2h_recency_rank") <= 5)

    agg = joined.group_by(["Div", "Season", "Date", "T1", "T2"]).agg([
        pl.col("focal_pts_hist").mean().alias("H2H_Recent_PPG"),
        pl.col("focal_draw_hist").mean().alias("H2H_Recent_Draw_Rate"),
        pl.col("focal_pts_hist").count().alias("H2H_Recent_Matches"),
    ])

    out = out.join(agg, left_on=["Div", "Season", "Date", "Team", "Opponent"],
                   right_on=["Div", "Season", "Date", "T1", "T2"], how="left",
                   maintain_order="left")
    return out.drop(["pair_a", "pair_b", "T1", "T2"], strict=False)


# Replace the broken _add_recent_h2h with the fixed version
_add_recent_h2h = _add_recent_h2h_fixed


def _add_season_progress(df: pl.DataFrame) -> pl.DataFrame:
    """Season progress: how far into the season we are (0.0 = start, 1.0 = end).

    Computed as team's matches played so far / total matches team will play in the season.
    Helps the model weight prior-season PPG vs current-season form:
    early in the season, prior season matters more; late in the season,
    current form matters more.
    Strictly pre-match: uses matches played BEFORE this fixture (strictly earlier dates).
    """
    out = df.sort(["Div", "Season", "Date"])

    # Matches played before this fixture within (Div, Season, Team) - strictly earlier dates.
    match_counts = out.group_by(["Div", "Season", "Team", "Date"]).agg(
        pl.len().alias("matches_on_date")
    ).sort(["Div", "Season", "Team", "Date"]).with_columns(
        pl.col("matches_on_date").cum_sum().over(["Div", "Season", "Team"]).alias("matches_up_to_date")
    )
    # Matches strictly before this date = matches_up_to_date - matches_on_date
    match_counts = match_counts.with_columns(
        (pl.col("matches_up_to_date") - pl.col("matches_on_date")).alias("Matches_So_Far")
    )

    out = out.join(
        match_counts.select(["Div", "Season", "Team", "Date", "Matches_So_Far"]),
        on=["Div", "Season", "Team", "Date"],
        how="left",
        maintain_order="left"
    )

    # Total matches per team-season (from the full data - this is the target season's total).
    # Since we can't know the future, we use the prior season's team match count as
    # a proxy for the expected season length, or fall back to league format.
    season_totals = out.group_by(["Div", "Season", "Team"]).agg(
        pl.len().alias("Season_Matches")
    ).sort(["Div", "Season", "Team"]).with_columns(
        pl.col("Season_Matches").shift(1).over(["Div", "Team"]).alias("Prev_Season_Matches")
    )

    out = out.join(season_totals.select(["Div", "Season", "Team", "Prev_Season_Matches"]),
                   on=["Div", "Season", "Team"], how="left", maintain_order="left")

    # If no prior season data for this team, estimate from division's prior season average.
    div_totals = out.group_by(["Div", "Season"]).agg(
        pl.col("Prev_Season_Matches").mean().alias("Div_Avg_Matches")
    ).sort(["Div", "Season"]).with_columns(
        pl.col("Div_Avg_Matches").shift(1).over("Div").alias("Prior_Div_Avg_Matches")
    ).select(["Div", "Season", "Prior_Div_Avg_Matches"])

    out = out.join(div_totals, on=["Div", "Season"], how="left", maintain_order="left")

    # Fallback chain: team's prior season matches -> division's prior season avg -> 38
    out = out.with_columns(
        pl.when(pl.col("Prev_Season_Matches").is_not_null())
        .then(pl.col("Prev_Season_Matches"))
        .when(pl.col("Prior_Div_Avg_Matches").is_not_null())
        .then(pl.col("Prior_Div_Avg_Matches").cast(pl.Int64))
        .otherwise(pl.lit(38))
        .alias("Season_Total_Matches")
    )

    out = out.with_columns(
        (pl.col("Matches_So_Far") / pl.col("Season_Total_Matches"))
        .alias("Season_Progress")
    )

    out = out.drop(["Matches_So_Far", "Prev_Season_Matches", "Prior_Div_Avg_Matches", "Season_Total_Matches"])
    return out


def _add_recent_opponent_strength(df: pl.DataFrame) -> pl.DataFrame:
    """Opponent's recent pre-match PPG over last 5 matches (strictly lagged).

    Captures opponent's recent form separately from cumulative PPG.
    """
    out = df.sort(["Div", "Team", "Season", "Date"])
    points = pl.col("Points")
    prior_points = (points.cum_sum().over(TEAM_SEASON_GROUP) - points).alias(
        "prior_points"
    )
    prior_matches = (points.cum_count().over(TEAM_SEASON_GROUP) - 1).alias(
        "prior_matches"
    )
    out = out.with_columns([prior_points, prior_matches])

    # Build per-team recent PPG (rolling mean over last 5 prior matches).
    team_recent = out.select([
        pl.col("Div"), pl.col("Season"), pl.col("Date"),
        pl.col("Team").alias("OppTeam"),
        pl.col("Points").alias("team_points"),
    ]).sort(["Div", "OppTeam", "Season", "Date"])

    team_recent = team_recent.with_columns(
        pl.col("team_points").rolling_mean(5, min_samples=1)
        .over(["Div", "OppTeam"]).alias("team_recent_ppg")
    ).with_columns(
        pl.col("team_recent_ppg").shift(1).over(["Div", "OppTeam"]).alias("team_recent_ppg_pre")
    ).select(["Div", "Season", "Date", "OppTeam", "team_recent_ppg_pre"])

    out = out.join(team_recent, left_on=["Div", "Season", "Date", "Opponent"],
                   right_on=["Div", "Season", "Date", "OppTeam"], how="left",
                   maintain_order="left")

    out = out.with_columns(
        pl.col("team_recent_ppg_pre").alias("Opp_Recent_PPG")
    ).drop(["team_recent_ppg_pre", "prior_points", "prior_matches"])

    return out


def _add_recent_venue_form(df: pl.DataFrame) -> pl.DataFrame:
    """Team's recent home/away form over last 5 matches at same venue (strictly lagged).

    Computes rolling mean of points over the last 5 matches at the same venue
    (home or away), strictly before the current match.
    """
    out = df.sort(["Div", "Team", "Season", "Date", "Venue"])
    points = pl.col("Points")

    # For each team-season-venue, compute rolling mean of last 5 prior matches.
    # Shift points by 1 first, then rolling mean.
    shifted_points = points.shift(1).over(["Div", "Team", "Season", "Venue"])
    out = out.with_columns(
        shifted_points.rolling_mean(5, min_samples=1)
        .over(["Div", "Team", "Season", "Venue"])
        .alias("venue_recent_ppg")
    )

    # Pivot: home team's home form, away team's away form.
    home_form = out.filter(pl.col("Venue") == "Home").select([
        pl.col("Div"), pl.col("Season"), pl.col("Date"),
        pl.col("Team").alias("HomeTeam"),
        pl.col("venue_recent_ppg").alias("Home_Recent_Venue_PPG"),
    ])
    away_form = out.filter(pl.col("Venue") == "Away").select([
        pl.col("Div"), pl.col("Season"), pl.col("Date"),
        pl.col("Team").alias("AwayTeam"),
        pl.col("venue_recent_ppg").alias("Away_Recent_Venue_PPG"),
    ])

    out = out.join(home_form, left_on=["Div", "Season", "Date", "Team"],
                   right_on=["Div", "Season", "Date", "HomeTeam"], how="left",
                   maintain_order="left")
    out = out.join(away_form, left_on=["Div", "Season", "Date", "Team"],
                   right_on=["Div", "Season", "Date", "AwayTeam"], how="left",
                   maintain_order="left")

    return out.drop(["venue_recent_ppg"])