"""
Team-strength rating systems for the predictive engine.

Includes:
- Historical points-per-game based ratings (reused from features).
- An Elo-style rating system as a candidate enhancement.

Elo design constraints (from the mission):
- New teams initialize safely to a league-average prior.
- Ratings update chronologically, match by match.
- Home advantage is incorporated.
- The current match result never enters its own prediction (prediction uses
  pre-match ratings; update happens after).
- Promotion/relegation handled via division-specific rating pools with
  regression toward the mean on division change.
- Teams with limited observations receive shrinkage toward the prior.
- Parameters tuned on training data only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl

from prediction.utils import OUTCOME_LABELS, set_seed


@dataclass
class EloConfig:
    """Configuration for the Elo rating system.

    All parameters are fit on training data only (chronological validation).
    """
    k_factor: float = 20.0          # Max rating points exchanged per match.
    home_advantage: float = 60.0    # Rating points added to the home side.
    initial_rating: float = 1500.0  # Default starting rating for new teams.
    regression_factor: float = 0.25 # Fraction toward mean on promotion/relegation.
    min_matches_before_reliable: int = 6  # Matches before a rating is trusted.
    goal_diff_weight: bool = False  # Whether to weight by goal difference (kept False for stability).


class EloRatings:
    """Chronological Elo rating tracker.

    The tracker is stateful and must be updated in strict date order. The
    ``rate`` method returns pre-match ratings (before incorporating the current
    match result); ``update`` incorporates a played result.
    """

    def __init__(self, config: Optional[EloConfig] = None, seed: int = 42):
        self.config = config or EloConfig()
        self.ratings: Dict[str, float] = {}          # team_key -> rating
        self.match_counts: Dict[str, int] = {}        # team_key -> matches played
        self.history: List[dict] = []                 # audit trail of updates
        set_seed(seed)
        self._div_means: Dict[str, float] = {}        # division -> mean rating

    # ------------------------------------------------------------------ init
    def _team_key(self, div: str, team: str) -> str:
        return f"{div}|{team}"

    def initialize_team(self, div: str, team: str, rating: Optional[float] = None) -> float:
        """Initialize a team's rating, defaulting to the division mean or prior."""
        key = self._team_key(div, team)
        if key in self.ratings:
            return self.ratings[key]
        if rating is None:
            rating = self._div_means.get(div, self.config.initial_rating)
        self.ratings[key] = float(rating)
        self.match_counts[key] = 0
        return float(rating)

    def _ensure(self, div: str, team: str) -> None:
        self.initialize_team(div, team)

    # ----------------------------------------------------------------- query
    def rate(self, div: str, home_team: str, away_team: str) -> Tuple[float, float]:
        """Return pre-match (home_rating, away_rating) with home advantage applied."""
        self._ensure(div, home_team)
        self._ensure(div, away_team)
        home = self.ratings[self._team_key(div, home_team)] + self.config.home_advantage
        away = self.ratings[self._team_key(div, away_team)]
        return home, away

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Expected score for team A against team B (logistic curve)."""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    # ----------------------------------------------------------------- update
    def update(self, div: str, home_team: str, away_team: str,
               home_goals: int, away_goals: int) -> None:
        """Incorporate a played match result into the ratings (chronological)."""
        self._ensure(div, home_team)
        self._ensure(div, away_team)
        hk = self._team_key(div, home_team)
        ak = self._team_key(div, away_team)

        r_home = self.ratings[hk] + self.config.home_advantage
        r_away = self.ratings[ak]

        e_home = self.expected_score(r_home, r_away)
        e_away = 1.0 - e_home

        # Actual score: win=1, draw=0.5, loss=0.
        if home_goals > away_goals:
            s_home, s_away = 1.0, 0.0
        elif home_goals < away_goals:
            s_home, s_away = 0.0, 1.0
        else:
            s_home, s_away = 0.5, 0.5

        k = self.config.k_factor
        # Reduce K for teams with few observations (more stable early on).
        for key, s, e, old in ((hk, s_home, e_home, self.ratings[hk]),
                               (ak, s_away, e_away, self.ratings[ak])):
            n = self.match_counts.get(key, 0)
            k_eff = k * min(1.0, np.sqrt(max(n, 1) / self.config.min_matches_before_reliable))
            new = old + k_eff * (s - e)
            self.ratings[key] = float(new)
            self.match_counts[key] = self.match_counts.get(key, 0) + 1
            self.history.append({
                "div": div, "team": key.split("|", 1)[1],
                "old_rating": old, "new_rating": new,
                "expected": e, "actual": s, "k_eff": k_eff,
            })

        # Recompute division means occasionally.
        self._recompute_div_means(div)

    def _recompute_div_means(self, div: str) -> None:
        ratings = [v for k, v in self.ratings.items() if k.startswith(f"{div}|")]
        self._div_means[div] = float(np.mean(ratings)) if ratings else self.config.initial_rating

    def promote_relegate(self, div_from: str, div_to: str, team: str) -> None:
        """Handle a team changing division: regress rating toward the new division mean."""
        key_from = self._team_key(div_from, team)
        key_to = self._team_key(div_to, team)
        old = self.ratings.get(key_from, self.config.initial_rating)
        target_mean = self._div_means.get(div_to, self.config.initial_rating)
        regressed = old + self.config.regression_factor * (target_mean - old)
        self.ratings[key_to] = float(regressed)
        self.match_counts[key_to] = self.match_counts.get(key_from, 0)
        self.history.append({
            "div": div_to, "team": team,
            "old_rating": old, "new_rating": regressed,
            "event": "division_change",
        })


def elo_match_probabilities(home_rating: float, away_rating: float) -> Tuple[float, float, float]:
    """Convert Elo ratings to 1X2 probabilities.

    Uses the logistic expected score for the home side and a draw model based on
    the closeness of the two teams (standard approach: draw probability
    increases as ratings converge).
    """
    e_home = EloRatings.expected_score(home_rating, away_rating)
    e_away = 1.0 - e_home
    # Draw probability: higher when teams are evenly matched.
    diff = abs(home_rating - away_rating)
    p_draw = 0.30 * np.exp(-(diff / 150.0) ** 2)
    p_draw = float(np.clip(p_draw, 0.05, 0.35))
    remaining = 1.0 - p_draw
    p_home = remaining * e_home
    p_away = remaining * e_away
    return p_home, p_draw, p_away


def build_elo_features(team_matches: pl.DataFrame,
                       config: Optional[EloConfig] = None) -> pl.DataFrame:
    """Compute pre-match Elo-based features for every team-match.

    Each match is processed exactly once (using the home-perspective row as the
    canonical record) so that neither team's pre-match rating is contaminated
    by the other's update of the *same* match. Ratings are updated strictly
    chronologically within each division: the pre-match state is recorded,
    then the match result updates the ratings. New teams initialize to the
    division mean; the current match result never enters its own prediction.

    CRITICAL: Pre-match ratings for a fixture on date D use ONLY matches with
    strictly earlier dates. Same-date matches are NOT included (no timestamp
    ordering available). This ensures training-inference parity.

    Output columns (base ratings, home advantage NOT added to the stored value):
    - Elo_Base_Rating: the team's own pre-match base rating.
    - Elo_Opponent_Rating: the opponent's pre-match base rating.
    - Elo_Exp_Home: expected home score pre-match (home advantage applied).
    - Elo_Rating_Diff: home base - away base, pre-match.
    """
    if team_matches.is_empty():
        return team_matches

    config = config or EloConfig()
    elo = EloRatings(config=config)

    # Canonical match record: one row per match (home perspective).
    matches = team_matches.filter(pl.col("Venue") == "Home").select([
        pl.col("Div"), pl.col("Season"), pl.col("Date"),
        pl.col("Team").alias("HomeTeam"),
        pl.col("Opponent").alias("AwayTeam"),
        pl.col("GF").alias("HomeGoals"),
        pl.col("GA").alias("AwayGoals"),
    ]).sort(["Div", "Date", "HomeTeam", "AwayTeam"])

    if matches.is_empty():
        return team_matches

    # Group matches by division and date. For each date, we need the Elo state
    # as of the START of that date (before any matches on that date).
    # We'll process dates in order, computing pre-match ratings for all matches
    # on that date using the Elo state from the end of the previous date.
    div_dates = matches.group_by(["Div", "Date"]).agg(
        pl.len().alias("n_matches")
    ).sort(["Div", "Date"])

    # We'll build a lookup of pre-match Elo for each match
    elo_lookup_rows = []

    # Track the START rating of each team's last match
    # This will be used to compute momentum for their next match
    last_match_start_rating: Dict[str, float] = {}

    # Process each division separately
    for div in matches.select(pl.col("Div").unique()).to_series().to_list():
        div_matches = matches.filter(pl.col("Div") == div).sort(["Date", "HomeTeam", "AwayTeam"])
        div_dates_list = div_matches.select(pl.col("Date").unique()).to_series().to_list()

        # Get all matches for this division as dicts
        div_rows = div_matches.to_dicts()

        # Initialize last_match_start_rating with starting ratings for this season
        teams_in_div = set()
        for row in div_rows:
            teams_in_div.add(row["HomeTeam"])
            teams_in_div.add(row["AwayTeam"])
        
        for team in teams_in_div:
            key = f"{div}|{team}"
            if key not in last_match_start_rating:
                if key not in elo.ratings:
                    elo.initialize_team(div, team)
                last_match_start_rating[key] = elo.ratings.get(key, config.initial_rating)

        # Get all matches for this division as dicts
        div_rows = div_matches.to_dicts()

        for date_val in div_dates_list:
            # Matches on this date
            date_matches = [r for r in div_rows if r["Date"] == date_val]

            # Initialize any new teams for this division
            for row in date_matches:
                for t in (row["HomeTeam"], row["AwayTeam"]):
                    key = f"{div}|{t}"
                    if key not in elo.ratings:
                        elo.initialize_team(div, t)
                    if key not in last_match_start_rating:
                        last_match_start_rating[key] = elo.ratings.get(key, config.initial_rating)

            # Record pre-match ratings for all matches on this date
            for row in date_matches:
                ht, at = row["HomeTeam"], row["AwayTeam"]
                h_r, a_r = elo.rate(div, ht, at)  # pre-match; home advantage added inside
                
                # Track rating changes for momentum feature
                # Use BASE ratings (without home advantage) for momentum calculation
                # Compare current match's START rating with the team's PREVIOUS match's START rating
                h_key = f"{div}|{ht}"
                a_key = f"{div}|{at}"
                h_base_curr = h_r - config.home_advantage  # current match's start base rating
                a_base_curr = a_r  # current base rating (away already has no HA)
                
                # Get last match's START rating (or initial if first match)
                h_prev_start = last_match_start_rating.get(h_key, config.initial_rating)
                a_prev_start = last_match_start_rating.get(a_key, config.initial_rating)
                
                h_change = h_base_curr - h_prev_start
                a_change = a_base_curr - a_prev_start
                
                elo_lookup_rows.append({
                    "Div": div,
                    "Season": row["Season"],
                    "Date": date_val,
                    "HomeTeam": ht,
                    "AwayTeam": at,
                    "Elo_Home_Base": h_base_curr,
                    "Elo_Away_Base": a_base_curr,
                    "Elo_Exp_Home": elo.expected_score(h_r, a_r),
                    "Elo_Rating_Diff": h_r - a_r,
                    "Elo_Home_Change": h_change,
                    "Elo_Away_Change": a_change,
                })

            # After recording pre-match ratings for all matches on this date,
            # update Elo with the results of these matches
            for row in date_matches:
                hg, ag = row["HomeGoals"], row["AwayGoals"]
                if hg is not None and ag is not None:
                    ht, at = row["HomeTeam"], row["AwayTeam"]
                    h_key = f"{div}|{ht}"
                    a_key = f"{div}|{at}"
                    
                    # Before updating, save the current match's START rating as the "last match start rating"
                    # for the next match this team plays
                    h_base_curr = elo.ratings.get(h_key, config.initial_rating)
                    a_base_curr = elo.ratings.get(a_key, config.initial_rating)
                    
                    elo.update(div, row["HomeTeam"], row["AwayTeam"], int(hg), int(ag))
                    
                    # After update, save the current match's START rating (before update)
                    # as the "last match start rating" for the next match
                    last_match_start_rating[h_key] = h_base_curr
                    last_match_start_rating[a_key] = a_base_curr

    elo_lookup = pl.DataFrame(elo_lookup_rows)

    out = team_matches
    # Home-perspective rows: own rating = home base, opponent = away base.
    home_elo = elo_lookup.select([
        pl.col("Div"), pl.col("Season"), pl.col("Date"),
        pl.col("HomeTeam").alias("Team"),
        pl.col("AwayTeam").alias("Opponent"),
        pl.col("Elo_Home_Base").alias("Elo_Base_Rating"),
        pl.col("Elo_Away_Base").alias("Elo_Opponent_Rating"),
        pl.col("Elo_Exp_Home"), pl.col("Elo_Rating_Diff"),
        pl.col("Elo_Home_Change").alias("Elo_Momentum"),
    ])
    # Away-perspective rows: own rating = away base, opponent = home base.
    away_elo = elo_lookup.select([
        pl.col("Div"), pl.col("Season"), pl.col("Date"),
        pl.col("AwayTeam").alias("Team"),
        pl.col("HomeTeam").alias("Opponent"),
        pl.col("Elo_Away_Base").alias("Elo_Base_Rating"),
        pl.col("Elo_Home_Base").alias("Elo_Opponent_Rating"),
        pl.col("Elo_Exp_Home"), pl.col("Elo_Rating_Diff"),
        pl.col("Elo_Away_Change").alias("Elo_Momentum"),
    ])
    elo_rows = pl.concat([home_elo, away_elo], how="diagonal")

    out = out.join(elo_rows, on=["Div", "Season", "Date", "Team", "Opponent"],
                   how="left", maintain_order="left")
    return out
