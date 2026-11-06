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

    # Core feature list (documented availability).
    core_features = [
        "Points_L3", "Points_L5", "Points_L10", "Points_Cum",
        "GF_L3", "GF_L5", "GF_L10", "GF_Cum",
        "GA_L3", "GA_L5", "GA_L10", "GA_Cum",
        "GD_L3", "GD_L5", "GD_L10", "GD_Cum",
        "Shots_L5", "Shots_OT_L5", "Shots_Against_L5",
        "Shot_Accuracy_L5", "Corners_L5",
        "PreMatch_PPG_Shrunk", "Matches_Cum_Lagged",
        "Rest_Days", "Match_Num",
        "Opp_PPG_Cum",
        "Prior_Season_PPG", "Hist_League_Avg_PPG",
        "League_Avg_GF_PG", "League_Avg_GA_PG",
        "Home_PPG", "Away_PPG", "Home_GF_PG", "Away_GF_PG",
        # League table state & form momentum (team-level, prefixed after pairing).
        "Team_Cum_Points_Before", "Team_Cum_Matches_Before", "Team_PPG_Before",
        "Team_Form_Momentum",
        # Head-to-head.
        "H2H_PPG", "H2H_GF_PG", "H2H_GA_PG", "H2H_Matches",
        "H2H_Recent_PPG", "H2H_Recent_Matches",
        # Streaks.
        "streak_W", "streak_D", "streak_L",
        # Table rank & momentum trends.
        "Team_Table_Rank",
        "Team_GD_Momentum", "Team_Shots_Momentum", "Team_GA_Trend",
        # Elo.
        "Elo_Base_Rating", "Elo_Opponent_Rating", "Elo_Exp_Home", "Elo_Rating_Diff",
    ]
    out = _null_cols(out, core_features)

    return out


def _add_prior_season_strength(df: pl.DataFrame) -> pl.DataFrame:
    """Add prior-season PPG (consecutive seasons, same division) as a prior."""
    out = df.sort(["Div", "Team", "Season", "Date"])
    season_num = pl.col("Season").replace(SEASON_ORDER).cast(pl.Int64).alias("Season_Num")

    # Team-season final PPG (mean over matches; full-season aggregate, used
    # only as prior for the NEXT season - permitted).
    ts = out.group_by(TEAM_SEASON_GROUP).agg(
        pl.col("Points").mean().alias("Final_PPG")
    )
    ts = ts.with_columns(season_num.alias("Season_Num")).sort(["Team", "Season_Num"])
    ts = ts.with_columns([
        pl.col("Final_PPG").shift(1).over("Team").alias("Prev_Final_PPG"),
        pl.col("Season_Num").shift(1).over("Team").alias("Prev_Season_Num"),
    ]).with_columns(
        (pl.col("Season_Num") - pl.col("Prev_Season_Num") == 1).alias("Consecutive")
    )

    prior_lookup = ts.filter(pl.col("Consecutive")).select([
        pl.col("Div"), pl.col("Team"), pl.col("Season"),
        pl.col("Prev_Final_PPG").alias("Prior_Season_PPG"),
    ])

    out = out.join(prior_lookup, on=["Div", "Team", "Season"], how="left",
                   maintain_order="left")

    # Historical league average PPG from prior seasons only.
    hist_avg = out.group_by(["Div", "Season"]).agg(
        pl.col("Points").mean().alias("League_Avg_PPG")
    ).sort(["Div", "Season"]).with_columns(
        pl.col("League_Avg_PPG").shift(1).over("Div").alias("Hist_League_Avg_PPG")
    ).select(["Div", "Season", "Hist_League_Avg_PPG"])

    out = out.join(hist_avg, on=["Div", "Season"], how="left", maintain_order="left")
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
    """
    out = df.sort(["Div", "Team", "Season", "Date"])
    grp = TEAM_SEASON_GROUP + ["Venue"]

    points = pl.col("Points")
    gf = pl.col("GF")

    prior_points = (points.cum_sum().over(grp) - points).alias("prior_points")
    prior_matches = (points.cum_count().over(grp) - 1).alias("prior_matches")
    prior_gf = (gf.cum_sum().over(grp) - gf).alias("prior_gf")

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
    """Pair home and away team pre-match features into match-level rows."""
    id_cols = {"Div", "Season", "Date", "Team", "Opponent", "Venue",
               "GF", "GA", "GD", "FT_Result", "Points"}
    home_cols = [c for c in team_features.columns if c not in id_cols]

    home = team_features.filter(pl.col("Venue") == "Home").select(
        ["Div", "Season", "Date", pl.col("Team").alias("HomeTeam"),
         pl.col("Opponent").alias("AwayTeam"),
         pl.col("GF").alias("FTHG"), pl.col("GA").alias("FTAG"),
         pl.col("FT_Result").replace_strict({"W": "H", "D": "D", "L": "A"}).alias("FTR")]
        + [pl.col(c).alias(f"Home_{c}") for c in home_cols]
    )
    away = team_features.filter(pl.col("Venue") == "Away").select(
        ["Div", "Season", "Date", pl.col("Team").alias("AwayTeam"),
         pl.col("Opponent").alias("HomeTeam")]
        + [pl.col(c).alias(f"Away_{c}") for c in home_cols]
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
    """
    out = df
    need = {"Points_L3", "Points_L5"} <= set(out.columns)
    if need:
        out = out.with_columns(
            (pl.col("Points_L3") - ((pl.col("Points_L5") * 5
                                     - pl.col("Points_L3") * 3) / 2.0))
            .alias("Team_Form_Momentum")
        )
    return out


def _add_head_to_head(df: pl.DataFrame) -> pl.DataFrame:
    """Head-to-head record between the two teams within the same Div/Season.

    For each team-match row, looks at previous encounters (same Div/Season,
    strictly earlier date) between the same unordered team pair and computes
    the focal team's H2H points per game and goals per game. Because league
    fixtures typically meet at most twice per season, this is a small,
    high-signal, strictly pre-match feature.
    """
    out = df.sort(["Div", "Team", "Season", "Date"])
    # Self-join: pair each row with earlier encounters of the same unordered pair.
    left = out.select([
        pl.col("Div"), pl.col("Season"), pl.col("Date"),
        pl.col("Team").alias("T1"), pl.col("Opponent").alias("T2"),
        pl.col("GF").alias("gf"), pl.col("GA").alias("ga"),
        pl.col("Points").alias("pts"),
    ])
    # Build both orientations so the unordered pair matches regardless of which
    # side is home/away in each historical encounter.
    left_a = left.with_columns(
        pl.when(pl.col("T1") < pl.col("T2")).then(pl.col("T1")).otherwise(pl.col("T2"))
        .alias("pair_a"),
        pl.when(pl.col("T1") < pl.col("T2")).then(pl.col("T2")).otherwise(pl.col("T1"))
        .alias("pair_b"),
    )
    right = left_a.clone()
    # As-of join on date within pair: for each encounter, find earlier encounters
    # of the same pair in the same Div/Season.
    joined = left_a.join_asof(
        right,
        left_on="Date", right_on="Date",
        by_left=["Div", "Season", "pair_a", "pair_b"],
        by_right=["Div", "Season", "pair_a", "pair_b"],
        strategy="backward",
        allow_exact_matches=False,
        suffix="_hist",
    )
    # Aggregate historical encounters per (Div, Season, Date, T1, T2).
    agg = joined.group_by(["Div", "Season", "Date", "T1", "T2"]).agg([
        pl.col("pts_hist").mean().alias("H2H_PPG"),
        pl.col("gf_hist").mean().alias("H2H_GF_PG"),
        pl.col("ga_hist").mean().alias("H2H_GA_PG"),
        pl.len().alias("H2H_Matches"),
    ])
    # Map back: the focal row's team is T1.
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

    Rank is computed within (Div, Season, Match_Num) so teams are compared
    at the same stage of the season. Based on pre-match PPG only — no future
    information from other teams' later matches enters the rank.
    Teams with 0 prior matches get rank = null.
    """
    out = df

    if "Team_PPG_Before" not in out.columns or "Match_Num" not in out.columns:
        return out

    # Rank within (Div, Season, Match_Num) by pre-match PPG.
    # Teams at the same match number are compared against each other.
    out = out.with_columns(
        pl.col("Team_PPG_Before")
        .rank(method="ordinal", descending=True)
        .over(["Div", "Season", "Match_Num"])
        .alias("Team_Table_Rank")
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

    Similar to _add_head_to_head but limited to the most recent 5 prior
    encounters between the same unordered team pair in the same Div/Season.
    Captures short-term rivalry form rather than all-time H2H.
    """
    out = df.sort(["Div", "Team", "Season", "Date"])
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
    right = left_a.clone()
    joined = left_a.join_asof(
        right,
        left_on="Date", right_on="Date",
        by_left=["Div", "Season", "pair_a", "pair_b"],
        by_right=["Div", "Season", "pair_a", "pair_b"],
        strategy="backward",
        allow_exact_matches=False,
        suffix="_hist",
    )
    # Limit to last 5 historical encounters per pair per date.
    joined = joined.with_columns(
        pl.col("pts_hist").cum_count().over(["Div", "Season", "Date", "T1", "T2", "pair_a", "pair_b"]).alias("h2h_order")
    )
    joined = joined.filter(pl.col("h2h_order") <= 5)

    agg = joined.group_by(["Div", "Season", "Date", "T1", "T2"]).agg([
        pl.col("pts_hist").mean().alias("H2H_Recent_PPG"),
        pl.len().alias("H2H_Recent_Matches"),
    ])
    out = out.join(agg, left_on=["Div", "Season", "Date", "Team", "Opponent"],
                   right_on=["Div", "Season", "Date", "T1", "T2"], how="left",
                   maintain_order="left")
    return out.drop(["pair_a", "pair_b", "T1", "T2"], strict=False)
