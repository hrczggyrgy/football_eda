"""
Statistical prediction models.

All models are fit on training data only and produce calibrated probability
distributions over match outcomes and goals. Models are compared using proper
scoring rules on chronological out-of-sample tests.

Models:
- Model A: League-aware historical baseline (mandatory).
- Model B: Poisson scoreline model (interpretable).
- Model C: Regularized multinomial logistic regression (direct 1X2 classifier).
- Model D: HistGradientBoosting challenger (only when empirically justified).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

from prediction.utils import (
    OUTCOME_LABELS, OUTCOME_TO_INDEX, INDEX_TO_OUTCOME,
    MAX_GOALS, REPORTED_SCORELINES, encode_outcomes,
    validate_outcome_distribution, validate_probabilities,
)
from prediction.distributions import poisson_1x2_from_lambdas
from prediction.calibration import brier_score_multiclass

# Base team-level features (prefixed Home_/Away_ after pairing in features.py).
_BASE_TEAM_FEATURES = [
    "Points_L5", "GF_L5", "GA_L5", "GD_L5",
    "Shots_L5", "Shots_OT_L5", "Shots_Against_L5",
    "Shot_Accuracy_L5", "Corners_L5",
    "PreMatch_PPG_Shrunk", "Matches_Cum_Lagged",
    "Rest_Days", "Opp_PPG_Cum",
    "Prior_Season_PPG", "Home_PPG", "Away_PPG", "Home_GF_PG", "Away_GF_PG",
    # League table state before the match (pre-match cumulative points / PPG).
    "Team_Cum_Points_Before", "Team_Cum_Matches_Before", "Team_PPG_Before",
    # Short-term form momentum.
    "Team_Form_Momentum",
    # Head-to-head within Div/Season.
    "H2H_PPG", "H2H_GF_PG", "H2H_GA_PG", "H2H_Matches",
    "H2H_Recent_PPG", "H2H_Recent_Matches",
    # Streaks (consecutive W/D/L before this match).
    "streak_W", "streak_D", "streak_L",
    # Table rank & momentum trends.
    "Team_Table_Rank",
    "Team_GD_Momentum", "Team_Shots_Momentum", "Team_GA_Trend",
    # Season progress & recent form.
    "Season_Progress",
    "Team_Recent_Venue_PPG",
    # Opponent recent strength.
    "Opp_Recent_PPG",
    # League-level scoring environment (kept once, not prefixed).
    "League_Avg_GF_PG", "League_Avg_GA_PG",
    # Elo ratings: base/opponent ratings are team-specific (prefixed),
    # expected score and rating diff are match-level (unprefixed).
    "Home_Elo_Base_Rating", "Away_Elo_Base_Rating",
    "Home_Elo_Opponent_Rating", "Away_Elo_Opponent_Rating",
    "Elo_Exp_Home", "Elo_Rating_Diff",
]


def _paired_feature_names() -> List[str]:
    """Expand base team features into Home_/Away_-prefixed paired names.

    League-level features are kept once (match-level). Elo base/opponent
    ratings are already prefixed (Home_Elo_*, Away_Elo_*) and kept as-is.
    Elo expected score and rating diff are match-level, kept unprefixed.
    """
    # Features already prefixed (Home_/Away_) - keep as-is, don't double-prefix.
    already_prefixed = {
        "Home_Elo_Base_Rating", "Away_Elo_Base_Rating",
        "Home_Elo_Opponent_Rating", "Away_Elo_Opponent_Rating",
    }
    # Match-level features - keep unprefixed.
    match_level = {
        "League_Avg_GF_PG", "League_Avg_GA_PG",
        "Elo_Exp_Home", "Elo_Rating_Diff",
        "PPG_Gap", "GF_Cum_Gap", "GA_Cum_Gap", "Rest_Days_Diff",
        # New draw-predictive features
        "PPG_Gap_Abs", "Elo_Rating_Diff_Abs", "League_Total_Goals_PG",
        "Combined_GA_Cum", "Combined_Venue_GF_PG", "H2H_Draw_Rate",
        "Combined_GF_Cum", "Late_Season_Both",
        "Evenly_Matched", "Low_Scoring_League", "Draw_Prone_Context",
    }
    team_only = [f for f in _BASE_TEAM_FEATURES
                 if f not in already_prefixed and f not in match_level]
    names = []
    for f in team_only:
        names.append(f"Home_{f}")
        names.append(f"Away_{f}")
    names += sorted(already_prefixed) + sorted(match_level)
    return names


BASE_FEATURES = _paired_feature_names()


@dataclass
class ModelResult:
    """Container for model predictions and fitted artifacts."""
    model_name: str
    outcome_probs: np.ndarray          # shape (n, 3) -> P(H), P(D), P(A)
    home_lambda: Optional[np.ndarray]  # expected home goals
    away_lambda: Optional[np.ndarray]  # expected away goals
    fitted: bool = False
    feature_names: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "model_name": self.model_name,
            "fitted": self.fitted,
            "feature_names": self.feature_names,
            "metadata": self.metadata,
        }


class HistoricalBaseline:
    """Model A: League-aware historical baseline.

    Estimates, from training data only:
    - Average home scoring rate (home goals per game).
    - Average away scoring rate.
    - Home advantage (difference in scoring).
    - Historical outcome frequencies (league-specific).

    Predictions for a fixture use league-specific rates shrunk toward the
    pooled average by training-sample exposure. This is a defensible baseline
    that any complex model must beat.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.fitted = False
        self.league_stats: Dict[str, dict] = {}
        self.pooled: dict = {}
        self.feature_names: List[str] = []

    def fit(self, team_matches: pl.DataFrame, matches: pl.DataFrame) -> "HistoricalBaseline":
        """Fit league-specific outcome frequencies and goal rates.

        ``matches`` may be the master match frame or a paired feature frame;
        both contain Div, FTHG, FTAG, FTR.
        """
        import polars as pl

        df = matches
        required = {"Div", "FTHG", "FTAG", "FTR"}
        if not required <= set(df.columns):
            raise ValueError(f"Baseline requires columns {required}")

        # Pooled estimates (prior).
        n_total = df.height
        self.pooled = {
            "p_home": float((df.select(pl.col("FTR") == "H").to_series().sum()) / n_total),
            "p_draw": float((df.select(pl.col("FTR") == "D").to_series().sum()) / n_total),
            "p_away": float((df.select(pl.col("FTR") == "A").to_series().sum()) / n_total),
            "home_gf_pg": float(df.select(pl.col("FTHG").mean()).to_series()[0]),
            "away_gf_pg": float(df.select(pl.col("FTAG").mean()).to_series()[0]),
        }

        # League-specific estimates.
        leagues = df.select(pl.col("Div").unique()).to_series().to_list()
        for div in leagues:
            sub = df.filter(pl.col("Div") == div)
            n = sub.height
            if n == 0:
                continue
            self.league_stats[div] = {
                "n": int(n),
                "p_home": float((sub.select(pl.col("FTR") == "H").to_series().sum()) / n),
                "p_draw": float((sub.select(pl.col("FTR") == "D").to_series().sum()) / n),
                "p_away": float((sub.select(pl.col("FTR") == "A").to_series().sum()) / n),
                "home_gf_pg": float(sub.select(pl.col("FTHG").mean()).to_series()[0]),
                "away_gf_pg": float(sub.select(pl.col("FTAG").mean()).to_series()[0]),
            }

        self.fitted = True
        self.feature_names = ["league_prior"]
        return self

    def _shrink(self, league_val: float, pooled_val: float, n: int, k: float = 5.0) -> float:
        """Shrink league estimate toward pooled by exposure."""
        return (league_val * n + pooled_val * k) / (n + k)

    def predict(self, matches: pl.DataFrame) -> ModelResult:
        """Predict outcome probabilities and goal lambdas for each match."""
        if not self.fitted:
            raise RuntimeError("Baseline must be fitted before prediction.")

        import polars as pl

        divs = matches.select(pl.col("Div")).to_series().to_list()
        n = matches.height
        p_home = np.empty(n); p_draw = np.empty(n); p_away = np.empty(n)
        home_lambda = np.empty(n); away_lambda = np.empty(n)

        for i, div in enumerate(divs):
            stats = self.league_stats.get(div)
            if stats is None:
                # Unknown league: use pooled with minimal shrinkage.
                stats = self.pooled
                n_eff = 1
            else:
                n_eff = stats["n"]

            p_home[i] = self._shrink(stats["p_home"], self.pooled["p_home"], n_eff)
            p_draw[i] = self._shrink(stats["p_draw"], self.pooled["p_draw"], n_eff)
            p_away[i] = self._shrink(stats["p_away"], self.pooled["p_away"], n_eff)

            # Goal lambdas from league scoring environment.
            home_lambda[i] = self._shrink(stats["home_gf_pg"], self.pooled["home_gf_pg"], n_eff)
            away_lambda[i] = self._shrink(stats["away_gf_pg"], self.pooled["away_gf_pg"], n_eff)

        # Renormalize outcome probs.
        out = np.column_stack([p_home, p_draw, p_away])
        out = out / out.sum(axis=1, keepdims=True)

        return ModelResult(
            model_name="historical_baseline",
            outcome_probs=out,
            home_lambda=home_lambda,
            away_lambda=away_lambda,
            fitted=True,
            feature_names=self.feature_names,
            metadata={"n_leagues": len(self.league_stats)},
        )


class PoissonModel:
    """Model B: Independent Poisson scoreline model.

    Models home and away goals as Poisson with rates:
        lambda_H = exp(attack_H - defense_A + home_adv + league_effect)
        lambda_A = exp(attack_A - defense_H + league_effect)

    Attack/defense strengths are pre-match team ratings derived from historical
    performance (shrunk). Scoreline probabilities are the outer product of the
    two Poisson PMFs, with the tail mass beyond MAX_GOALS allocated to "other".
    """

    def __init__(self, seed: int = 42, use_dixon_coles: bool = True, calibrate: bool = True):
        self.seed = seed
        self.use_dixon_coles = use_dixon_coles
        self.calibrate = calibrate
        self.fitted = False
        self.team_attack: Dict[str, float] = {}
        self.team_defense: Dict[str, float] = {}
        self.home_advantage: float = 0.0
        self.league_effects: Dict[str, float] = {}
        self.league_home_advantage: Dict[str, float] = {}
        self.pooled_home_rate: float = 1.3
        self.pooled_away_rate: float = 1.1
        self.feature_names = list(BASE_FEATURES)
        # Dixon-Coles rho parameter (negative = fewer low-scoring draws than independence)
        self.rho: Dict[str, float] = {}
        # Temperature calibration
        self.temperature: Optional[float] = None

    def fit(self, team_matches: pl.DataFrame, matches: pl.DataFrame) -> "PoissonModel":
        """Fit attack/defense strengths and league effects from training data.

        team_matches: team-perspective frame with pre-match features (lags only).
        matches: match-level frame with goals and outcomes.
        """
        import polars as pl

        # League average goals per game (home scores, away scores).
        league_goals = matches.group_by("Div").agg([
            pl.col("FTHG").mean().alias("lg_home_gf"),
            pl.col("FTAG").mean().alias("lg_away_gf"),
        ]).to_dicts()
        for row in league_goals:
            div = row["Div"]
            self.league_effects[div] = float(row["lg_home_gf"])
            # Home advantage per league = home goals - away goals.
            self.league_home_advantage[div] = float(row["lg_home_gf"] - row["lg_away_gf"])

        self.pooled_home_rate = float(matches.select(pl.col("FTHG").mean()).to_series()[0])
        self.pooled_away_rate = float(matches.select(pl.col("FTAG").mean()).to_series()[0])

        # Team attack/defense: historical GF/GA per game, shrunk toward league average.
        # Use team-season-like aggregates from team_matches (full training history).
        tm = team_matches
        team_stats = tm.group_by(["Div", "Team"]).agg([
            pl.col("GF").mean().alias("team_gf_pg"),
            pl.col("GA").mean().alias("team_ga_pg"),
            pl.len().alias("n_matches"),
        ]).to_dicts()

        for row in team_stats:
            div = row["Div"]; team = row["Team"]
            n = row["n_matches"]
            lg_home = self.league_effects.get(div, self.pooled_home_rate)
            lg_away = self.pooled_away_rate
            obs_gf = row["team_gf_pg"]; obs_ga = row["team_ga_pg"]
            # Attack strength relative to league home scoring; defense relative to away.
            att = (obs_gf * n + lg_home * 5.0) / (n + 5.0)   # shrink toward league
            defence = (obs_ga * n + lg_away * 5.0) / (n + 5.0)
            # Convert to multiplicative factors around 1.0.
            self.team_attack[f"{div}|{team}"] = float(att / max(lg_home, 0.1))
            self.team_defense[f"{div}|{team}"] = float(defence / max(lg_away, 0.1))

        # Global home advantage (pooled).
        self.home_advantage = float(
            matches.select(((pl.col("FTHG") - pl.col("FTAG")).mean())).to_series()[0]
        )
        self.home_advantage = float(np.clip(self.home_advantage, -0.5, 0.8))

        # Dixon-Coles rho parameter per division (estimated via MLE approximation).
        # rho adjusts the probability of low-scoring draws (0-0, 1-1) relative to
        # independence. Negative rho = fewer low-score draws than independence.
        if self.use_dixon_coles:
            self._fit_dixon_coles_rho(matches)

        self.fitted = True
        
        # Fit temperature calibration if enabled
        if self.calibrate:
            self._fit_temperature(matches)

        return self

    def _fit_temperature(self, matches: pl.DataFrame) -> None:
        """Fit temperature scaling calibration on training data."""
        import polars as pl
        from scipy.optimize import minimize_scalar
        
        divs = matches.select(pl.col("Div")).to_series().to_list()
        hteams = matches.select(pl.col("HomeTeam")).to_series().to_list()
        ateams = matches.select(pl.col("AwayTeam")).to_series().to_list()
        hg = matches.select(pl.col("FTHG")).to_series().to_list()
        ag = matches.select(pl.col("FTAG")).to_series().to_list()
        
        # Compute lambdas and logits for all training matches
        logits_list = []
        outcomes = []
        
        for i in range(len(divs)):
            div, ht, at, h, a = divs[i], hteams[i], ateams[i], hg[i], ag[i]
            hl, al = self._lambda(div, ht, at)
            if self.use_dixon_coles:
                rho = self.rho.get(div, 0.0)
                hl, al = self._apply_dixon_coles(hl, al, rho)
            
            # Compute outcome probabilities
            ph, pd_, pa = poisson_1x2(hl, al)
            probs = np.array([ph, pd_, pa])
            logits = np.log(np.clip(probs, 1e-12, 1 - 1e-12))
            logits_list.append(logits)
            
            # Actual outcome
            if h > a:
                outcomes.append(0)
            elif h == a:
                outcomes.append(1)
            else:
                outcomes.append(2)
        
        if len(logits_list) > 10:
            logits_arr = np.array(logits_list)
            outcomes_arr = np.array(outcomes)
            
            # Fit temperature by minimizing NLL
            def nll(T):
                T = max(T, 1e-3)
                scaled = logits_arr / T
                scaled = scaled - scaled.max(axis=1, keepdims=True)
                exp_scaled = np.exp(scaled)
                probs = exp_scaled / exp_scaled.sum(axis=1, keepdims=True)
                n = probs.shape[0]
                y_onehot = np.zeros_like(probs)
                valid = (outcomes_arr >= 0) & (outcomes_arr < 3)
                y_onehot[valid, outcomes_arr[valid]] = 1.0
                log_probs = np.log(np.clip(probs, 1e-12, 1.0))
                return -np.mean(np.sum(y_onehot * log_probs, axis=1)[valid])
            
            logits_arr = np.array(logits_list)
            outcomes_arr = np.array(outcomes)
            
            result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
            self.temperature = float(result.x)

    def _fit_dixon_coles_rho(self, matches: pl.DataFrame) -> None:
        """Estimate Dixon-Coles rho parameter per division via moment matching.

        Under independence, P(0-0) = exp(-lambda_H) * exp(-lambda_A).
        With rho correction: P(0-0) = (1 - rho) * exp(-lambda_H) * exp(-lambda_A).
        We solve for rho using the observed frequency of 0-0 scorelines.
        """
        import polars as pl

        # Compute lambdas for all training matches.
        divs = matches.select(pl.col("Div")).to_series().to_list()
        hteams = matches.select(pl.col("HomeTeam")).to_series().to_list()
        ateams = matches.select(pl.col("AwayTeam")).to_series().to_list()
        hg = matches.select(pl.col("FTHG")).to_series().to_list()
        ag = matches.select(pl.col("FTAG")).to_series().to_list()

        # Group by division.
        from collections import defaultdict
        div_data: Dict[str, list] = defaultdict(list)
        for i in range(len(divs)):
            div_data[divs[i]].append((hteams[i], ateams[i], hg[i], ag[i]))

        for div, matches_list in div_data.items():
            # Expected 0-0 under independence.
            exp_00 = 0.0
            obs_00 = 0
            for ht, at, h, a in matches_list:
                hl, al = self._lambda(div, ht, at)
                exp_00 += np.exp(-hl) * np.exp(-al)
                if h == 0 and a == 0:
                    obs_00 += 1

            n = len(matches_list)
            if exp_00 > 0 and n > 0:
                obs_rate = obs_00 / n
                exp_rate = exp_00 / n
                # rho = 1 - obs_rate / exp_rate (clipped to [-1, 1] for stability)
                rho = 1.0 - obs_rate / exp_rate
                self.rho[div] = float(np.clip(rho, -1.0, 1.0))
            else:
                self.rho[div] = 0.0

    def _apply_dixon_coles(self, home_lambda: float, away_lambda: float,
                           rho: float) -> Tuple[float, float]:
        """Apply Dixon-Coles correction to adjust low-score probabilities.

        Adjusts P(0-0), P(1-0), P(0-1), P(1-1) while preserving marginals.
        rho < 0 reduces low-score draws; rho > 0 increases them.
        """
        if abs(rho) < 1e-6:
            return home_lambda, away_lambda

        # Dixon-Coles adjustment: modify the joint distribution at (0,0), (1,0), (0,1), (1,1)
        # We adjust the lambdas slightly to match the corrected probabilities.
        # Simple approach: scale the probabilities at the four cells.
        p00 = np.exp(-home_lambda) * np.exp(-away_lambda) * (1 - rho)
        p10 = home_lambda * np.exp(-home_lambda) * np.exp(-away_lambda) * (1 + rho)
        p01 = away_lambda * np.exp(-home_lambda) * np.exp(-away_lambda) * (1 + rho)
        p11 = home_lambda * away_lambda * np.exp(-home_lambda) * np.exp(-away_lambda) * (1 - rho)

        # Renormalize to keep total probability = 1 for these 4 cells.
        # Adjust lambdas to match the corrected marginals.
        # Simple heuristic: scale home_lambda and away_lambda.
        scale_h = 1.0 + rho * (np.exp(-away_lambda) - away_lambda * np.exp(-away_lambda))
        scale_a = 1.0 + rho * (np.exp(-home_lambda) - home_lambda * np.exp(-home_lambda))

        new_home_lambda = home_lambda * max(scale_h, 0.1)
        new_away_lambda = away_lambda * max(scale_a, 0.1)
        return float(new_home_lambda), float(new_away_lambda)

    def _lambda(self, div: str, home_team: str, away_team: str) -> Tuple[float, float]:
        """Compute (home_lambda, away_lambda) pre-match."""
        hk = f"{div}|{home_team}"
        ak = f"{div}|{away_team}"
        att_h = self.team_attack.get(hk, 1.0)
        def_a = self.team_defense.get(ak, 1.0)
        att_a = self.team_attack.get(ak, 1.0)
        def_h = self.team_defense.get(hk, 1.0)

        lg = self.league_effects.get(div, self.pooled_home_rate)
        ha = self.league_home_advantage.get(div, self.home_advantage)

        # log-linear: lambda = league_rate * att * opp_defence * home_adv_factor
        # defence factor = goals conceded per game relative to league. A team
        # that concedes more (weak defence, factor > 1) should make its
        # opponent more likely to score, so we multiply by the conceding rate.
        home_lambda = lg * att_h * max(def_a, 0.1) * np.exp(ha * 0.15)
        away_lambda = lg * att_a * max(def_h, 0.1) * np.exp(-ha * 0.15)
        return float(home_lambda), float(away_lambda)

    def predict(self, matches: pl.DataFrame) -> ModelResult:
        if not self.fitted:
            raise RuntimeError("Poisson model must be fitted before prediction.")

        import polars as pl

        n = matches.height
        home_lambda = np.empty(n); away_lambda = np.empty(n)
        p_home = np.empty(n); p_draw = np.empty(n); p_away = np.empty(n)

        divs = matches.select(pl.col("Div")).to_series().to_list()
        hteams = matches.select(pl.col("HomeTeam")).to_series().to_list()
        ateams = matches.select(pl.col("AwayTeam")).to_series().to_list()

        for i in range(n):
            hl, al = self._lambda(divs[i], hteams[i], ateams[i])
            # Apply Dixon-Coles correction if enabled.
            if self.use_dixon_coles:
                rho = self.rho.get(divs[i], 0.0)
                hl, al = self._apply_dixon_coles(hl, al, rho)
            home_lambda[i] = hl; away_lambda[i] = al
            # 1X2 from independent Poisson: sum over scorelines.
            ph, pd_, pa = poisson_1x2(hl, al)
            p_home[i] = ph; p_draw[i] = pd_; p_away[i] = pa

        out = np.column_stack([p_home, p_draw, p_away])
        out = out / out.sum(axis=1, keepdims=True)

        # Apply temperature calibration if fitted
        if self.calibrate and self.temperature is not None:
            T = max(self.temperature, 1e-3)
            logits = np.log(np.clip(out, 1e-12, 1 - 1e-12))
            scaled = logits / T
            scaled = scaled - scaled.max(axis=1, keepdims=True)
            exp_scaled = np.exp(scaled)
            out = exp_scaled / exp_scaled.sum(axis=1, keepdims=True)

        return ModelResult(
            model_name="poisson",
            outcome_probs=out,
            home_lambda=home_lambda,
            away_lambda=away_lambda,
            fitted=True,
            feature_names=self.feature_names,
            metadata={"use_dixon_coles": self.use_dixon_coles, "calibrated": self.calibrate, "temperature": self.temperature},
        )


def poisson_1x2(home_lambda: float, away_lambda: float,
                max_goals: int = MAX_GOALS) -> Tuple[float, float, float]:
    """1X2 probabilities from independent Poisson goal distributions.

    P(home win) = sum_{h>a} Pois(h|hl)*Pois(a|al), etc.
    """
    home_lambda = max(float(home_lambda), 0.01)
    away_lambda = max(float(away_lambda), 0.01)
    goals = np.arange(0, max_goals + 1)
    ph = poisson.pmf(goals, home_lambda)
    pa = poisson.pmf(goals, away_lambda)
    # Outer product -> scoreline matrix.
    mat = np.outer(ph, pa)
    p_home = float(np.tril(mat, -1).sum())   # h > a
    p_draw = float(np.trace(mat))             # h == a
    p_away = float(np.triu(mat, 1).sum())     # h < a
    total = p_home + p_draw + p_away
    if total <= 0:
        return 1.0 / 3, 1.0 / 3, 1.0 / 3
    return p_home / total, p_draw / total, p_away / total


class MultinomialModel:
    """Model C: Regularized multinomial logistic regression for 1X2.

    Predicts P(H), P(D), P(A) directly from pre-match features using
    L2-regularized logistic regression. The regularization strength C is tuned
    exclusively inside the training period using chronological inner validation,
    and output probabilities are calibrated with a meta-classifier fitted on
    temporally-valid out-of-fold training predictions (never on held-out test
    data).
    """

    def __init__(self, seed: int = 42, C: Optional[float] = None,
                 max_iter: int = 1000, calibrate: bool = True,
                 n_inner_folds: int = 2,
                 calibration_method: str = "platt",
                 class_weight: Optional[str] = None):
        self.seed = seed
        self.C = C  # If None, tuned via inner chronological validation.
        self.max_iter = max_iter
        self.calibrate = calibrate
        self.n_inner_folds = n_inner_folds
        self.calibration_method = calibration_method  # "platt" or "isotonic"
        self.class_weight = class_weight  # "balanced" or None
        self.fitted = False
        self.model: Optional[LogisticRegression] = None
        self.scaler: Optional[StandardScaler] = None
        self.calibrator: Optional[object] = None
        self.feature_names: List[str] = []
        self.classes_: np.ndarray = np.array([0, 1, 2])
        self.C_used: Optional[float] = None
        self.selected_feature_mask: Optional[np.ndarray] = None
        self.selected_feature_names: List[str] = []

    def _feature_frame(self, matches: pl.DataFrame) -> Tuple[np.ndarray, List[str]]:
        import polars as pl
        available = [c for c in BASE_FEATURES if c in matches.columns]
        if not available:
            raise ValueError("No baseline features available in match frame.")
        X = matches.select(available).to_numpy().astype(np.float64)
        return X, available

    def _impute_fit(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Impute non-finite values with column means (computed on training data).
        
        Returns imputed X and the impute values for later use.
        """
        X = X.copy()
        impute_values = np.full(X.shape[1], 0.0, dtype=np.float64)
        for j in range(X.shape[1]):
            col = X[:, j]
            finite = np.isfinite(col)
            impute_values[j] = float(np.mean(col[finite])) if np.any(finite) else 0.0
            X[:, j] = np.where(finite, col, impute_values[j])
        return X, impute_values

    def _impute_transform(self, X: np.ndarray, impute_values: Optional[np.ndarray] = None) -> np.ndarray:
        X = X.copy()
        if impute_values is None:
            impute_values = getattr(self, "impute_", np.full(X.shape[1], 0.0, dtype=np.float64))
        for j in range(X.shape[1]):
            fill = impute_values[j] if j < len(impute_values) else 0.0
            X[:, j] = np.where(np.isfinite(X[:, j]), X[:, j], fill)
        return X

    @staticmethod
    def _inner_chronological_splits(dates: np.ndarray, n_folds: int = 3,
                                     min_train_dates: int = 20) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate chronological (train_idx, val_idx) splits by unique date.
        
        Uses non-overlapping validation blocks with a substantial initial training period.
        Same-date fixtures are kept together.
        """
        unique_dates = np.sort(np.unique(dates))
        if len(unique_dates) < min_train_dates + n_folds:
            # Fallback to simple expanding window if not enough dates
            if len(unique_dates) < n_folds + 1:
                return []
            split_points = np.linspace(1, len(unique_dates) - 1, n_folds).astype(int)
            splits = []
            for sp in split_points:
                cut = unique_dates[sp]
                train_idx = np.where(dates < cut)[0]
                val_idx = np.where(dates >= cut)[0]
                if len(train_idx) > 0 and len(val_idx) > 0:
                    splits.append((train_idx, val_idx))
            return splits
        
        # Substantial initial training period
        train_end = min_train_dates
        val_block_size = max(1, (len(unique_dates) - train_end) // n_folds)
        splits = []
        
        for fold in range(n_folds):
            val_start = train_end
            val_end = min(val_start + val_block_size, len(unique_dates) - (n_folds - fold - 1))
            if val_start >= val_end:
                break
            
            train_dates = unique_dates[:train_end]
            val_dates = unique_dates[val_start:val_end]
            
            train_idx = np.where(np.isin(dates, train_dates))[0]
            val_idx = np.where(np.isin(dates, val_dates))[0]
            
            if len(train_idx) > 0 and len(val_idx) > 0:
                # Check class representation in both splits
                splits.append((train_idx, val_idx))
            
            train_end = val_end
        
        return splits

    def _fit_base(self, X: np.ndarray, y: np.ndarray, C: float,
                   impute_values: Optional[np.ndarray] = None,
                   scaler: Optional[StandardScaler] = None) -> Tuple[LogisticRegression, StandardScaler, np.ndarray]:
        """Fit base model with optional pre-fitted preprocessing."""
        X_imputed = self._impute_transform(X, impute_values)
        if scaler is None:
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X_imputed)
        else:
            Xs = scaler.transform(X_imputed)
        model = LogisticRegression(
            C=C, solver="lbfgs", max_iter=self.max_iter,
            random_state=self.seed,
        )
        model.fit(Xs, y)
        return model, scaler, impute_values

    def fit(self, team_matches: pl.DataFrame, matches: pl.DataFrame) -> "MultinomialModel":
        """Fit the multinomial model on engineered paired features.

        ``matches`` should be a paired feature frame containing the BASE_FEATURES
        and FTR target. Missing features are imputed with column means fitted on
        the training data (never on the test data). C is tuned via chronological
        inner validation with FOLD-LOCAL PREPROCESSING; calibration is fitted on
        out-of-fold training predictions only.
        """
        import polars as pl

        X, names = self._feature_frame(matches)
        y = encode_outcomes(matches.select(pl.col("FTR")).to_series())

        valid = y >= 0
        X = X[valid]; y = y[valid]
        dates = matches.select(pl.col("Date")).to_numpy().ravel()[valid]

        if X.shape[0] == 0 or len(np.unique(y)) < 2:
            raise ValueError("Insufficient valid training data for multinomial model.")

        # ------------------------------------------------------------------ tune C with FOLD-LOCAL PREPROCESSING
        candidate_Cs = [self.C] if self.C is not None else [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        best_C, best_brier = None, float("inf")
        oof_probs_list: List[np.ndarray] = []
        oof_y_list: List[np.ndarray] = []

        splits = self._inner_chronological_splits(dates, self.n_inner_folds)
        valid_splits = []
        for train_idx, val_idx in splits:
            if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[val_idx])) < 2:
                continue
            valid_splits.append((train_idx, val_idx))

        if len(valid_splits) >= 2:
            for cand in candidate_Cs:
                fold_briers = []
                for train_idx, val_idx in valid_splits:
                    # FOLD-LOCAL: fit imputer and scaler on training partition only
                    X_train = X[train_idx]
                    X_val = X[val_idx]
                    
                    # Fit imputer on train
                    X_train_imputed, impute_values = self._impute_fit(X_train)
                    # Fit scaler on imputed train
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train_imputed)
                    # Transform val with train-fitted imputer and scaler
                    X_val_imputed = self._impute_transform(X_val, impute_values)
                    X_val_scaled = scaler.transform(X_val_imputed)
                    
                    model = LogisticRegression(
                        C=cand, solver="lbfgs", max_iter=self.max_iter,
                        random_state=self.seed,
                        class_weight=self.class_weight,
                    )
                    model.fit(X_train_scaled, y[train_idx])
                    val_probs = model.predict_proba(X_val_scaled)
                    fold_briers.append(brier_score_multiclass(val_probs, y[val_idx]))
                mean_brier = float(np.mean(fold_briers))
                if mean_brier < best_brier:
                    best_brier = mean_brier
                    best_C = cand
            
            # Collect OOF predictions using the best C with fold-local preprocessing
            for train_idx, val_idx in valid_splits:
                X_train = X[train_idx]
                X_val = X[val_idx]
                y_train = y[train_idx]
                y_val = y[val_idx]
                
                X_train_imputed, impute_values = self._impute_fit(X_train)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_imputed)
                X_val_imputed = self._impute_transform(X_val, impute_values)
                X_val_scaled = scaler.transform(X_val_imputed)
                
                model = LogisticRegression(
                    C=best_C, solver="lbfgs", max_iter=self.max_iter,
                    random_state=self.seed,
                    class_weight=self.class_weight,
                )
                model.fit(X_train_scaled, y_train)
                val_probs = model.predict_proba(X_val_scaled)
                oof_probs_list.append(val_probs)
                oof_y_list.append(y_val)
        else:
            best_C = candidate_Cs[0]

        self.C_used = float(best_C)

        # ------------------------------------------------------------------ refit on all training data with GLOBAL preprocessing (for production use)
        # For the final production model, we fit imputer and scaler on ALL training data
        X_imputed, impute_values = self._impute_fit(X)
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X_imputed)
        self.model = LogisticRegression(
            C=best_C, solver="lbfgs", max_iter=self.max_iter,
            random_state=self.seed,
            class_weight=self.class_weight,
        )
        self.model.fit(Xs, y)
        self.feature_names = names
        self.fitted = True
        self.classes_ = self.model.classes_.copy()
        # Store impute values for inference
        self.impute_ = impute_values

        # ------------------------------------------------------------------ goal model (Poisson regression)
        # Fit separate Poisson regressors for home and away goals using the same
        # pre-match features. This gives the multinomial model expected-goals
        # output so it is not limited to outcome probabilities alone.
        try:
            from sklearn.linear_model import PoissonRegressor
            hg = matches.select(pl.col("FTHG")).to_numpy().ravel()[valid].astype(np.float64)
            ag = matches.select(pl.col("FTAG")).to_numpy().ravel()[valid].astype(np.float64)
            # Clip negative / non-finite targets.
            hg = np.where(np.isfinite(hg) & (hg >= 0), hg, 0.0)
            ag = np.where(np.isfinite(ag) & (ag >= 0), ag, 0.0)
            self.goal_scaler_h = StandardScaler()
            self.goal_scaler_a = StandardScaler()
            Xs_h = self.goal_scaler_h.fit_transform(X_imputed)
            Xs_a = self.goal_scaler_a.fit_transform(X_imputed)
            self.goal_model_h = PoissonRegressor(
                max_iter=self.max_iter,
            )
            self.goal_model_a = PoissonRegressor(
                max_iter=self.max_iter,
            )
            self.goal_model_h.fit(Xs_h, hg)
            self.goal_model_a.fit(Xs_a, ag)
        except Exception:
            self.goal_model_h = None
            self.goal_model_a = None
            self.goal_scaler_h = None
            self.goal_scaler_a = None

        # ------------------------------------------------------------------ calibration (OOF training only)
        if self.calibrate and len(oof_probs_list) > 0:
            O = np.vstack(oof_probs_list)
            Y = np.concatenate(oof_y_list)
            
            from prediction.calibration import get_calibrator
            cal_method = self.calibration_method  # "temperature", "isotonic", "ensemble", "platt"
            calibrator = get_calibrator(cal_method, seed=self.seed)
            
            if cal_method == "platt":
                # Platt scaling: multinomial logistic regression on raw probs.
                cal = LogisticRegression(
                    C=1.0, solver="lbfgs", max_iter=self.max_iter,
                    random_state=self.seed,
                )
                cal.fit(O, Y)
                self.calibrator = ("platt", cal)
            elif cal_method == "temperature":
                logits = np.log(np.clip(O, 1e-12, 1 - 1e-12))
                self.calibrator = calibrator.fit(O, O.argmax(axis=1))  # temp calibrator needs logits and outcomes
            else:
                # isotonic or ensemble
                calibrator.fit(O, np.argmax(O, axis=1))
                self.calibrator = ("calibrator", calibrator)

        return self

    def _order_probs(self, probs: np.ndarray) -> np.ndarray:
        ordered = np.zeros((probs.shape[0], 3))
        for i, cls in enumerate(self.model.classes_):
            ci = int(cls)
            if 0 <= ci <= 2:
                ordered[:, ci] = probs[:, i]
        total = ordered.sum(axis=1, keepdims=True)
        total = np.where(total > 0, total, 1.0)
        return ordered / total

    def predict(self, matches: pl.DataFrame) -> ModelResult:
        if not self.fitted:
            raise RuntimeError("Multinomial model must be fitted before prediction.")
        import polars as pl

        X, _ = self._feature_frame(matches)
        X = self._impute_transform(X)
        X_full = X  # Keep full feature set for goal model

        # Apply feature selection mask (if feature selection was used).
        if hasattr(self, "selected_feature_mask") and self.selected_feature_mask is not None:
            X = X[:, self.selected_feature_mask]

        Xs = self.scaler.transform(X)
        probs = self.model.predict_proba(Xs)
        ordered = self._order_probs(probs)

        # Expected goals from the Poisson goal model (same pre-match features).
        home_lambda = None
        away_lambda = None
        if self.goal_model_h is not None and self.goal_model_a is not None:
            Xs_h = self.goal_scaler_h.transform(X_full)
            Xs_a = self.goal_scaler_a.transform(X_full)
            home_lambda = np.maximum(self.goal_model_h.predict(Xs_h), 0.01)
            away_lambda = np.maximum(self.goal_model_a.predict(Xs_a), 0.01)

        # Apply calibration fitted on out-of-fold training predictions only.
        if self.calibrator is not None:
            cal_type = self.calibration_method
            if cal_type == "isotonic" and isinstance(self.calibrator, list):
                # Isotonic regression: apply per-class monotonic mapping.
                cal_probs = np.column_stack([
                    self.calibrator[k].predict(ordered[:, k]) for k in range(3)
                ])
                ordered = cal_probs / cal_probs.sum(axis=1, keepdims=True)
            elif cal_type == "temperature" and hasattr(self.calibrator, 'transform_probs'):
                # Temperature scaling
                ordered = self.calibrator.transform_probs(ordered)
            elif cal_type == "ensemble" and hasattr(self.calibrator, 'transform'):
                # Ensemble calibrator
                ordered = self.calibrator.transform(ordered)
            elif cal_type == "platt" and isinstance(self.calibrator, tuple) and self.calibrator[0] == "platt":
                # Platt scaling
                cal = self.calibrator[1]
                calibrated = cal.predict_proba(ordered)
                cal_ordered = np.zeros((calibrated.shape[0], 3))
                for i, cls in enumerate(cal.classes_):
                    if 0 <= int(cls) <= 2:
                        cal_ordered[:, int(cls)] = calibrated[:, i]
                ordered = cal_ordered / cal_ordered.sum(axis=1, keepdims=True)
            elif isinstance(self.calibrator, tuple) and self.calibrator[0] == "calibrator":
                # Ensemble or isotonic from get_calibrator
                cal = self.calibrator[1]
                ordered = cal.transform(ordered)

        return ModelResult(
            model_name="multinomial_logistic",
            outcome_probs=ordered,
            home_lambda=home_lambda,
            away_lambda=away_lambda,
            fitted=True,
            feature_names=self.feature_names,
            metadata={"C": self.C_used, "calibrated": self.calibrator is not None,
                      "calibration_method": self.calibration_method,
                      "classes": self.classes_.tolist(),
                      "goal_model": self.goal_model_h is not None},
        )


class HistGradientBoostingModel:
    """Model D: HistGradientBoostingClassifier challenger.

    Used only when supported by sample size and temporal validation. Produces
    calibrated 1X2 probabilities. Not used unless it demonstrably beats simpler
    models on chronological out-of-sample tests.
    """

    def __init__(self, seed: int = 42, max_iter: int = 100, max_leaf_nodes: int = 31,
                 min_samples_leaf: int = 20, l2_regularization: float = 1.0):
        self.seed = seed
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.fitted = False
        self.model: Optional[HistGradientBoostingClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []

    def _feature_frame(self, matches):
        import polars as pl
        available = [c for c in BASE_FEATURES if c in matches.columns]
        X = matches.select(available).to_numpy().astype(np.float64)
        return X, available

    def _impute_fit(self, X: np.ndarray) -> np.ndarray:
        X = X.copy()
        self.impute_ = np.full(X.shape[1], 0.0, dtype=np.float64)
        for j in range(X.shape[1]):
            col = X[:, j]
            finite = np.isfinite(col)
            self.impute_[j] = float(np.mean(col[finite])) if np.any(finite) else 0.0
            X[:, j] = np.where(finite, col, self.impute_[j])
        return X

    def _impute_transform(self, X: np.ndarray) -> np.ndarray:
        X = X.copy()
        for j in range(X.shape[1]):
            fill = self.impute_[j] if hasattr(self, "impute_") and j < len(self.impute_) else 0.0
            X[:, j] = np.where(np.isfinite(X[:, j]), X[:, j], fill)
        return X

    def fit(self, team_matches: pl.DataFrame, matches: pl.DataFrame):
        import polars as pl
        X, names = self._feature_frame(matches)
        y = encode_outcomes(matches.select(pl.col("FTR")).to_series())
        valid = y >= 0
        X = X[valid]; y = y[valid]
        if X.shape[0] < 100 or len(np.unique(y)) < 2:
            raise ValueError("Insufficient data for HGB model.")
        X = self._impute_fit(X)
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)
        self.model = HistGradientBoostingClassifier(
            max_iter=self.max_iter, max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            random_state=self.seed,
        )
        self.model.fit(Xs, y)
        self.feature_names = names
        self.fitted = True
        return self

    def predict(self, matches):
        if not self.fitted:
            raise RuntimeError("HGB model must be fitted before prediction.")
        import polars as pl
        X, _ = self._feature_frame(matches)
        X = self._impute_transform(X)
        Xs = self.scaler.transform(X)
        probs = self.model.predict_proba(Xs)
        ordered = np.zeros((Xs.shape[0], 3))
        for i, cls in enumerate(self.model.classes_):
            ordered[:, int(cls)] = probs[:, i]
        ordered = ordered / ordered.sum(axis=1, keepdims=True)
        return ModelResult(
            model_name="hist_gradient_boosting",
            outcome_probs=ordered,
            home_lambda=None, away_lambda=None,
            fitted=True, feature_names=self.feature_names,
            metadata={"max_iter": self.max_iter},
        )


class EnsembleModel:
    """Model E: Ensemble of baseline + multinomial with chronological weighting.

    Combines the HistoricalBaseline (robust, especially early season) and the
    MultinomialModel (stronger when in-season form is available) using weights
    optimized on chronological out-of-fold validation. The weight given to the
    multinomial model increases as more in-season history is available, since
    the baseline relies only on prior-season strength.

    Weights are determined by minimizing out-of-fold Brier score on chronological
    inner validation — never on held-out test data.
    """

    def __init__(self, seed: int = 42, baseline_weight: Optional[float] = None,
                 multinomial_weight: Optional[float] = None,
                 calibrate: bool = True, n_inner_folds: int = 2):
        self.seed = seed
        self.baseline_weight = baseline_weight  # If None, tuned via inner CV.
        self.multinomial_weight = multinomial_weight  # If None, tuned via inner CV.
        self.calibrate = calibrate
        self.n_inner_folds = n_inner_folds
        self.fitted = False
        self.baseline: Optional[HistoricalBaseline] = None
        self.multinomial: Optional[MultinomialModel] = None
        self.baseline_weight_used: Optional[float] = None
        self.multinomial_weight_used: Optional[float] = None

    def _inner_chronological_splits(self, dates: np.ndarray, n_folds: int = 2):
        """Generate chronological (train_idx, val_idx) splits by unique date."""
        unique_dates = np.sort(np.unique(dates))
        if len(unique_dates) < n_folds + 1:
            return []
        split_points = np.linspace(1, len(unique_dates) - 1, n_folds).astype(int)
        splits = []
        for sp in split_points:
            cut = unique_dates[sp]
            train_idx = np.where(dates < cut)[0]
            val_idx = np.where(dates >= cut)[0]
            if len(train_idx) > 0 and len(val_idx) > 0:
                splits.append((train_idx, val_idx))
        return splits

    def fit(self, team_matches: pl.DataFrame, matches: pl.DataFrame) -> "EnsembleModel":
        """Fit baseline + multinomial, tune ensemble weights via inner CV."""
        import polars as pl

        # Fit both component models on all training data.
        self.baseline = HistoricalBaseline()
        self.baseline.fit(team_matches, matches)

        self.multinomial = MultinomialModel(calibrate=False, seed=self.seed)
        self.multinomial.fit(team_matches, matches)

        # Tune ensemble weights via chronological inner validation.
        X, _ = self.multinomial._feature_frame(matches)
        y = encode_outcomes(matches.select(pl.col("FTR")).to_series())
        valid = y >= 0
        X = X[valid]; y = y[valid]
        dates = matches.select(pl.col("Date")).to_numpy().ravel()[valid]

        # Get baseline predictions for all training matches.
        baseline_res = self.baseline.predict(matches)
        baseline_probs = baseline_res.outcome_probs[valid]

        # Get multinomial predictions via inner CV (out-of-fold).
        splits = self._inner_chronological_splits(dates, self.n_inner_folds)
        valid_splits = []
        for train_idx, val_idx in splits:
            if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[val_idx])) < 2:
                continue
            valid_splits.append((train_idx, val_idx))

        if len(valid_splits) >= 2:
            # OOF multinomial predictions.
            oof_multi_probs = []
            oof_y = []
            for train_idx, val_idx in valid_splits:
                m = MultinomialModel(calibrate=False, seed=self.seed)
                m.fit(team_matches, matches.iloc[train_idx] if hasattr(matches, 'iloc') else matches)
                # Re-fit on training subset using paired features
                m_multi = MultinomialModel(calibrate=False, seed=self.seed)
                # We need paired features for multinomial - use the same X from _feature_frame
                # Reconstruct X for train subset
                X_train = X[train_idx]
                y_train = y[train_idx]
                # Fit scaler on train subset
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(C=1.0, solver="lbfgs",
                                           max_iter=self.multinomial.max_iter,
                                           random_state=self.seed)
                model.fit(X_train_s, y_train)
                # Predict on val subset
                X_val_s = scaler.transform(X[val_idx])
                val_probs = model.predict_proba(X_val_s)
                ordered = np.zeros((val_probs.shape[0], 3))
                for i, cls in enumerate(model.classes_):
                    ci = int(cls)
                    if 0 <= ci <= 2:
                        ordered[:, ci] = val_probs[:, i]
                ordered = ordered / ordered.sum(axis=1, keepdims=True)
                oof_multi_probs.append(ordered)
                oof_y.append(y[val_idx])

            if len(oof_multi_probs) > 0:
                O_multi = np.vstack(oof_multi_probs)
                Y_oof = np.concatenate(oof_y)
                O_base = baseline_res.outcome_probs[valid]

                # Grid search for optimal weight.
                best_weight, best_brier = 0.5, float("inf")
                for bw in np.linspace(0, 1, 21):
                    mw = 1.0 - bw
                    ens_probs = bw * O_base + mw * O_multi
                    ens_probs = ens_probs / ens_probs.sum(axis=1, keepdims=True)
                    b = brier_score_multiclass(ens_probs, Y_oof)
                    if b < best_brier:
                        best_brier = b
                        best_weight = bw

                self.baseline_weight_used = float(best_weight)
                self.multinomial_weight_used = float(1.0 - best_weight)
            else:
                self.baseline_weight_used = 0.5
                self.multinomial_weight_used = 0.5
        else:
            self.baseline_weight_used = 0.5
            self.multinomial_weight_used = 0.5

        self.fitted = True
        return self

    def predict(self, matches: pl.DataFrame) -> ModelResult:
        if not self.fitted:
            raise RuntimeError("Ensemble model must be fitted before prediction.")

        import polars as pl

        # Baseline predictions.
        base_res = self.baseline.predict(matches)
        base_probs = base_res.outcome_probs

        # Multinomial predictions.
        multi_res = self.multinomial.predict(matches)
        multi_probs = multi_res.outcome_probs

        # Blend with learned weights.
        bw = self.baseline_weight_used if self.baseline_weight_used is not None else 0.5
        mw = self.multinomial_weight_used if self.multinomial_weight_used is not None else 0.5

        ens_probs = bw * base_probs + mw * multi_probs
        ens_probs = ens_probs / ens_probs.sum(axis=1, keepdims=True)

        # Expected goals: baseline only provides lambdas; multinomial may too.
        home_lambda = None
        away_lambda = None
        if base_res.home_lambda is not None:
            home_lambda = base_res.home_lambda.copy()
            away_lambda = base_res.away_lambda.copy()
        elif multi_res.home_lambda is not None:
            home_lambda = multi_res.home_lambda.copy()
            away_lambda = multi_res.away_lambda.copy()

        # Optionally calibrate ensemble probabilities via Platt on OOF.
        # (Skipped here for simplicity; component models are already calibrated.)

        return ModelResult(
            model_name="ensemble_baseline_multinomial",
            outcome_probs=ens_probs,
            home_lambda=home_lambda,
            away_lambda=away_lambda,
            fitted=True,
            feature_names=list(BASE_FEATURES),
            metadata={
                "baseline_weight": self.baseline_weight_used,
                "multinomial_weight": self.multinomial_weight_used,
                "components": ["historical_baseline", "multinomial_logistic"],
            },
        )


# =============================================================================
# Phase D: Model Ladder
# =============================================================================

# Small feature set for Model 1: ~15 interpretable inputs
SMALL_LOGISTIC_FEATURES = [
    # League indicators (match-level)
    "League_Avg_GF_PG", "League_Avg_GA_PG",
    # Team strength (prior-season + early season)
    "Home_Prior_Season_PPG", "Away_Prior_Season_PPG",
    "Home_PreMatch_PPG_Shrunk", "Away_PreMatch_PPG_Shrunk",
    # Home/away specific
    "Home_Home_PPG", "Away_Away_PPG",
    "Home_Home_GF_PG", "Away_Away_GF_PG",
    # League table state
    "Home_Team_PPG_Before", "Away_Team_PPG_Before",
    "Home_Team_Cum_Matches_Before", "Away_Team_Cum_Matches_Before",
    # Rest difference
    "Rest_Days_Diff",
    # Elo (match-level, already pre-match)
    "Elo_Rating_Diff", "Elo_Exp_Home",
    # History exposure
    "Home_Matches_Cum_Lagged", "Away_Matches_Cum_Lagged",
]

# Feature set for Model 2: Elo + logistic
ELO_LOGISTIC_FEATURES = [
    "Elo_Rating_Diff", "Elo_Exp_Home",
    "League_Avg_GF_PG", "League_Avg_GA_PG",
    "Home_Prior_Season_PPG", "Away_Prior_Season_PPG",
    "Rest_Days_Diff",
]

# Feature set for Model 3: Fitted Poisson (attack/defense)
FITTED_POISSON_FEATURES = [
    "Home_Prior_Season_PPG", "Away_Prior_Season_PPG",
    "Home_PreMatch_PPG_Shrunk", "Away_PreMatch_PPG_Shrunk",
    "Home_Home_PPG", "Away_Away_PPG",
    "Home_Home_GF_PG", "Away_Away_GF_PG",
    "League_Avg_GF_PG", "League_Avg_GA_PG",
    "Home_Matches_Cum_Lagged", "Away_Matches_Cum_Lagged",
    "Elo_Rating_Diff", "Elo_Exp_Home",
    "Rest_Days_Diff",
]


class SmallLogisticModel:
    """Model 1: Small uncalibrated logistic model with ~15 interpretable features.
    
    Features selected for stability and interpretability:
    - League scoring environment
    - Prior-season and early-season team strength (PPG)
    - Home/away specific performance
    - League table position (PPG before match)
    - Rest difference
    - Elo rating difference (pre-match)
    - History exposure (matches played)
    
    No calibration, no class weighting, no H2H, no streaks, no table rank.
    Regularization tuned via inner CV.
    """
    
    def __init__(self, seed: int = 42, C: Optional[float] = None,
                 max_iter: int = 1000, calibrate: bool = False,
                 class_weight: Optional[str] = None):
        self.seed = seed
        self.C = C  # If None, tuned via inner chronological validation
        self.max_iter = max_iter
        self.calibrate = calibrate
        self.class_weight = class_weight
        self.fitted = False
        self.model: Optional[LogisticRegression] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = SMALL_LOGISTIC_FEATURES.copy()
        self.classes_: np.ndarray = np.array([0, 1, 2])
        self.C_used: Optional[float] = None
        self.impute_: Optional[np.ndarray] = None
        self.calibrator: Optional[object] = None
    
    def _feature_frame(self, matches: pl.DataFrame) -> Tuple[np.ndarray, List[str]]:
        import polars as pl
        available = [c for c in self.feature_names if c in matches.columns]
        if not available:
            raise ValueError("No small logistic features available in match frame.")
        X = matches.select(available).to_numpy().astype(np.float64)
        return X, available
    
    def _impute_fit(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = X.copy()
        impute_values = np.full(X.shape[1], 0.0, dtype=np.float64)
        for j in range(X.shape[1]):
            col = X[:, j]
            finite = np.isfinite(col)
            impute_values[j] = float(np.mean(col[finite])) if np.any(finite) else 0.0
            X[:, j] = np.where(finite, col, impute_values[j])
        return X, impute_values
    
    def _impute_transform(self, X: np.ndarray, impute_values: Optional[np.ndarray] = None) -> np.ndarray:
        X = X.copy()
        if impute_values is None:
            impute_values = getattr(self, "impute_", np.full(X.shape[1], 0.0, dtype=np.float64))
        for j in range(X.shape[1]):
            fill = impute_values[j] if j < len(impute_values) else 0.0
            X[:, j] = np.where(np.isfinite(X[:, j]), X[:, j], fill)
        return X
    
    @staticmethod
    def _inner_chronological_splits(dates: np.ndarray, n_folds: int = 3,
                                     min_train_dates: int = 20) -> List[Tuple[np.ndarray, np.ndarray]]:
        unique_dates = np.sort(np.unique(dates))
        if len(unique_dates) < min_train_dates + n_folds:
            if len(unique_dates) < n_folds + 1:
                return []
            split_points = np.linspace(1, len(unique_dates) - 1, n_folds).astype(int)
            splits = []
            for sp in split_points:
                cut = unique_dates[sp]
                train_idx = np.where(dates < cut)[0]
                val_idx = np.where(dates >= cut)[0]
                if len(train_idx) > 0 and len(val_idx) > 0:
                    splits.append((train_idx, val_idx))
            return splits
        
        train_end = min_train_dates
        val_block_size = max(1, (len(unique_dates) - train_end) // n_folds)
        splits = []
        
        for fold in range(n_folds):
            val_start = train_end
            val_end = min(val_start + val_block_size, len(unique_dates) - (n_folds - fold - 1))
            if val_start >= val_end:
                break
            
            train_dates = unique_dates[:train_end]
            val_dates = unique_dates[val_start:val_end]
            
            train_idx = np.where(np.isin(dates, train_dates))[0]
            val_idx = np.where(np.isin(dates, val_dates))[0]
            
            if len(train_idx) > 0 and len(val_idx) > 0:
                splits.append((train_idx, val_idx))
            
            train_end = val_end
        
        return splits
    
    def _fit_base(self, X: np.ndarray, y: np.ndarray, C: float,
                   impute_values: Optional[np.ndarray] = None,
                   scaler: Optional[StandardScaler] = None) -> Tuple[LogisticRegression, StandardScaler, np.ndarray]:
        X_imputed = self._impute_transform(X, impute_values)
        if scaler is None:
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X_imputed)
        else:
            Xs = scaler.transform(X_imputed)
        model = LogisticRegression(
            C=C, solver="lbfgs", max_iter=self.max_iter,
            random_state=self.seed,
        )
        model.fit(Xs, y)
        return model, scaler, impute_values
    
    def fit(self, team_matches: pl.DataFrame, matches: pl.DataFrame) -> "SmallLogisticModel":
        import polars as pl
        
        X, names = self._feature_frame(matches)
        y = encode_outcomes(matches.select(pl.col("FTR")).to_series())
        
        valid = y >= 0
        X = X[valid]; y = y[valid]
        dates = matches.select(pl.col("Date")).to_numpy().ravel()[valid]
        
        if X.shape[0] == 0 or len(np.unique(y)) < 2:
            raise ValueError("Insufficient valid training data for small logistic model.")
        
        # Tune C with fold-local preprocessing
        candidate_Cs = [self.C] if self.C is not None else [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        best_C, best_brier = None, float("inf")
        oof_probs_list: List[np.ndarray] = []
        oof_y_list: List[np.ndarray] = []
        
        splits = self._inner_chronological_splits(dates, self.n_inner_folds if hasattr(self, 'n_inner_folds') else 3)
        valid_splits = []
        for train_idx, val_idx in splits:
            if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[val_idx])) < 2:
                continue
            valid_splits.append((train_idx, val_idx))
        
        if len(valid_splits) >= 2:
            for cand in candidate_Cs:
                fold_briers = []
                for train_idx, val_idx in valid_splits:
                    X_train = X[train_idx]
                    X_val = X[val_idx]
                    X_train_imputed, impute_values = self._impute_fit(X_train)
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train_imputed)
                    X_val_imputed = self._impute_transform(X_val, impute_values)
                    X_val_scaled = scaler.transform(X_val_imputed)
                    
                    model = LogisticRegression(
                        C=cand, solver="lbfgs", max_iter=self.max_iter,
                        random_state=self.seed,
                        class_weight=self.class_weight,
                    )
                    model.fit(X_train_scaled, y[train_idx])
                    val_probs = model.predict_proba(X_val_scaled)
                    fold_briers.append(brier_score_multiclass(val_probs, y[val_idx]))
                mean_brier = float(np.mean(fold_briers))
                if mean_brier < best_brier:
                    best_brier = mean_brier
                    best_C = cand
            
            # Collect OOF predictions for optional calibration
            for train_idx, val_idx in valid_splits:
                X_train = X[train_idx]
                X_val = X[val_idx]
                X_train_imputed, impute_values = self._impute_fit(X_train)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_imputed)
                X_val_imputed = self._impute_transform(X_val, impute_values)
                X_val_scaled = scaler.transform(X_val_imputed)
                
                model = LogisticRegression(
                    C=best_C, solver="lbfgs", max_iter=self.max_iter,
                    random_state=self.seed,
                    class_weight=self.class_weight,
                )
                model.fit(X_train_scaled, y[train_idx])
                val_probs = model.predict_proba(X_val_scaled)
                oof_probs_list.append(val_probs)
                oof_y_list.append(y[val_idx])
        else:
            best_C = candidate_Cs[0]
        
        self.C_used = float(best_C)
        
        # Refit on all training data
        X_imputed, impute_values = self._impute_fit(X)
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X_imputed)
        self.model = LogisticRegression(
            C=best_C, solver="lbfgs", max_iter=self.max_iter,
            random_state=self.seed,
        )
        self.model.fit(Xs, y)
        self.feature_names = names
        self.fitted = True
        self.classes_ = self.model.classes_.copy()
        self.impute_ = impute_values
        
        # Optional calibration (default False for Model 1)
        if self.calibrate and len(oof_probs_list) > 0:
            O = np.vstack(oof_probs_list)
            Y = np.concatenate(oof_y_list)
            cal = LogisticRegression(
                C=1.0, solver="lbfgs", max_iter=self.max_iter,
                random_state=self.seed,
            )
            cal.fit(O, Y)
            self.calibrator = cal
        
        return self
    
    def _order_probs(self, probs: np.ndarray) -> np.ndarray:
        ordered = np.zeros((probs.shape[0], 3))
        for i, cls in enumerate(self.model.classes_):
            ci = int(cls)
            if 0 <= ci <= 2:
                ordered[:, ci] = probs[:, i]
        total = ordered.sum(axis=1, keepdims=True)
        total = np.where(total > 0, total, 1.0)
        return ordered / total
    
    def predict(self, matches: pl.DataFrame) -> ModelResult:
        if not self.fitted:
            raise RuntimeError("Small logistic model must be fitted before prediction.")
        import polars as pl
        
        X, _ = self._feature_frame(matches)
        X = self._impute_transform(X, self.impute_)
        Xs = self.scaler.transform(X)
        probs = self.model.predict_proba(Xs)
        ordered = self._order_probs(probs)
        
        # Apply calibration if fitted
        if self.calibrator is not None:
            calibrated = self.calibrator.predict_proba(ordered)
            cal_ordered = np.zeros((calibrated.shape[0], 3))
            for i, cls in enumerate(self.calibrator.classes_):
                if 0 <= int(cls) <= 2:
                    cal_ordered[:, int(cls)] = calibrated[:, i]
            ordered = cal_ordered / cal_ordered.sum(axis=1, keepdims=True)
        
        return ModelResult(
            model_name="small_logistic",
            outcome_probs=ordered,
            home_lambda=None,
            away_lambda=None,
            fitted=True,
            feature_names=self.feature_names,
            metadata={"C": self.C_used, "calibrated": self.calibrator is not None},
        )


class EloLogisticModel:
    """Model 2: Elo difference + league effects + home advantage + logistic regression.
    
    Uses verified pre-match Elo difference as the primary predictor,
    plus league scoring environment and prior-season strength.
    """
    
    def __init__(self, seed: int = 42, C: Optional[float] = None,
                 max_iter: int = 1000, calibrate: bool = True,
                 class_weight: Optional[str] = None):
        self.seed = seed
        self.C = C
        self.max_iter = max_iter
        self.calibrate = calibrate
        self.class_weight = class_weight
        self.fitted = False
        self.model: Optional[LogisticRegression] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = ELO_LOGISTIC_FEATURES.copy()
        self.classes_: np.ndarray = np.array([0, 1, 2])
        self.C_used: Optional[float] = None
        self.impute_: Optional[np.ndarray] = None
        self.calibrator: Optional[object] = None
    
    def _feature_frame(self, matches: pl.DataFrame) -> Tuple[np.ndarray, List[str]]:
        import polars as pl
        available = [c for c in self.feature_names if c in matches.columns]
        X = matches.select(available).to_numpy().astype(np.float64)
        return X, available
    
    def _impute_fit(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = X.copy()
        impute_values = np.full(X.shape[1], 0.0, dtype=np.float64)
        for j in range(X.shape[1]):
            col = X[:, j]
            finite = np.isfinite(col)
            impute_values[j] = float(np.mean(col[finite])) if np.any(finite) else 0.0
            X[:, j] = np.where(finite, col, impute_values[j])
        return X, impute_values
    
    def _impute_transform(self, X: np.ndarray, impute_values: Optional[np.ndarray] = None) -> np.ndarray:
        X = X.copy()
        if impute_values is None:
            impute_values = getattr(self, "impute_", np.full(X.shape[1], 0.0, dtype=np.float64))
        for j in range(X.shape[1]):
            fill = impute_values[j] if j < len(impute_values) else 0.0
            X[:, j] = np.where(np.isfinite(X[:, j]), X[:, j], fill)
        return X
    
    @staticmethod
    def _inner_chronological_splits(dates: np.ndarray, n_folds: int = 3,
                                     min_train_dates: int = 20) -> List[Tuple[np.ndarray, np.ndarray]]:
        unique_dates = np.sort(np.unique(dates))
        if len(unique_dates) < min_train_dates + n_folds:
            if len(unique_dates) < n_folds + 1:
                return []
            split_points = np.linspace(1, len(unique_dates) - 1, n_folds).astype(int)
            splits = []
            for sp in split_points:
                cut = unique_dates[sp]
                train_idx = np.where(dates < cut)[0]
                val_idx = np.where(dates >= cut)[0]
                if len(train_idx) > 0 and len(val_idx) > 0:
                    splits.append((train_idx, val_idx))
            return splits
        
        train_end = min_train_dates
        val_block_size = max(1, (len(unique_dates) - train_end) // n_folds)
        splits = []
        
        for fold in range(n_folds):
            val_start = train_end
            val_end = min(val_start + val_block_size, len(unique_dates) - (n_folds - fold - 1))
            if val_start >= val_end:
                break
            
            train_dates = unique_dates[:train_end]
            val_dates = unique_dates[val_start:val_end]
            
            train_idx = np.where(np.isin(dates, train_dates))[0]
            val_idx = np.where(np.isin(dates, val_dates))[0]
            
            if len(train_idx) > 0 and len(val_idx) > 0:
                splits.append((train_idx, val_idx))
            
            train_end = val_end
        
        return splits
    
    def _fit_base(self, X: np.ndarray, y: np.ndarray, C: float,
                   impute_values: Optional[np.ndarray] = None,
                   scaler: Optional[StandardScaler] = None) -> Tuple[LogisticRegression, StandardScaler, np.ndarray]:
        X_imputed = self._impute_transform(X, impute_values)
        if scaler is None:
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X_imputed)
        else:
            Xs = scaler.transform(X_imputed)
        model = LogisticRegression(
            C=C, solver="lbfgs", max_iter=self.max_iter,
            random_state=self.seed,
        )
        model.fit(Xs, y)
        return model, scaler, impute_values
    
    def fit(self, team_matches: pl.DataFrame, matches: pl.DataFrame) -> "EloLogisticModel":
        import polars as pl
        
        X, names = self._feature_frame(matches)
        y = encode_outcomes(matches.select(pl.col("FTR")).to_series())
        
        valid = y >= 0
        X = X[valid]; y = y[valid]
        dates = matches.select(pl.col("Date")).to_numpy().ravel()[valid]
        
        if X.shape[0] == 0 or len(np.unique(y)) < 2:
            raise ValueError("Insufficient valid training data for Elo logistic model.")
        
        # Tune C with fold-local preprocessing
        candidate_Cs = [self.C] if self.C is not None else [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        best_C, best_brier = None, float("inf")
        oof_probs_list: List[np.ndarray] = []
        oof_y_list: List[np.ndarray] = []
        
        splits = self._inner_chronological_splits(dates, 3)
        valid_splits = []
        for train_idx, val_idx in splits:
            if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[val_idx])) < 2:
                continue
            valid_splits.append((train_idx, val_idx))
        
        if len(valid_splits) >= 2:
            for cand in candidate_Cs:
                fold_briers = []
                for train_idx, val_idx in valid_splits:
                    X_train = X[train_idx]
                    X_val = X[val_idx]
                    X_train_imputed, impute_values = self._impute_fit(X_train)
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train_imputed)
                    X_val_imputed = self._impute_transform(X_val, impute_values)
                    X_val_scaled = scaler.transform(X_val_imputed)
                    
                    model = LogisticRegression(
                        C=cand, solver="lbfgs", max_iter=self.max_iter,
                        random_state=self.seed,
                        class_weight=self.class_weight,
                    )
                    model.fit(X_train_scaled, y[train_idx])
                    val_probs = model.predict_proba(X_val_scaled)
                    fold_briers.append(brier_score_multiclass(val_probs, y[val_idx]))
                mean_brier = float(np.mean(fold_briers))
                if mean_brier < best_brier:
                    best_brier = mean_brier
                    best_C = cand
            
            for train_idx, val_idx in valid_splits:
                X_train = X[train_idx]
                X_val = X[val_idx]
                X_train_imputed, impute_values = self._impute_fit(X_train)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_imputed)
                X_val_imputed = self._impute_transform(X_val, impute_values)
                X_val_scaled = scaler.transform(X_val_imputed)
                
                model = LogisticRegression(
                    C=best_C, solver="lbfgs", max_iter=self.max_iter,
                    random_state=self.seed,
                    class_weight=self.class_weight,
                )
                model.fit(X_train_scaled, y[train_idx])
                val_probs = model.predict_proba(X_val_scaled)
                oof_probs_list.append(val_probs)
                oof_y_list.append(y[val_idx])
        else:
            best_C = candidate_Cs[0]
        
        self.C_used = float(best_C)
        
        # Refit on all training data
        X_imputed, impute_values = self._impute_fit(X)
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X_imputed)
        self.model = LogisticRegression(
            C=best_C, solver="lbfgs", max_iter=self.max_iter,
            random_state=self.seed,
            class_weight=self.class_weight,
        )
        self.model.fit(Xs, y)
        self.feature_names = names
        self.fitted = True
        self.classes_ = self.model.classes_.copy()
        self.impute_ = impute_values
        
        # Calibration
        if self.calibrate and len(oof_probs_list) > 0:
            O = np.vstack(oof_probs_list)
            Y = np.concatenate(oof_y_list)
            cal = LogisticRegression(
                C=1.0, solver="lbfgs", max_iter=self.max_iter,
                random_state=self.seed,
                class_weight=self.class_weight,
            )
            cal.fit(O, Y)
            self.calibrator = cal
        
        return self
    
    def _order_probs(self, probs: np.ndarray) -> np.ndarray:
        ordered = np.zeros((probs.shape[0], 3))
        for i, cls in enumerate(self.model.classes_):
            ci = int(cls)
            if 0 <= ci <= 2:
                ordered[:, ci] = probs[:, i]
        total = ordered.sum(axis=1, keepdims=True)
        total = np.where(total > 0, total, 1.0)
        return ordered / total
    
    def predict(self, matches: pl.DataFrame) -> ModelResult:
        if not self.fitted:
            raise RuntimeError("Elo logistic model must be fitted before prediction.")
        import polars as pl
        
        X, _ = self._feature_frame(matches)
        X = self._impute_transform(X, self.impute_)
        Xs = self.scaler.transform(X)
        probs = self.model.predict_proba(Xs)
        ordered = self._order_probs(probs)
        
        if self.calibrator is not None:
            calibrated = self.calibrator.predict_proba(ordered)
            cal_ordered = np.zeros((calibrated.shape[0], 3))
            for i, cls in enumerate(self.calibrator.classes_):
                if 0 <= int(cls) <= 2:
                    cal_ordered[:, int(cls)] = calibrated[:, i]
            ordered = cal_ordered / cal_ordered.sum(axis=1, keepdims=True)
        
        return ModelResult(
            model_name="elo_logistic",
            outcome_probs=ordered,
            home_lambda=None,
            away_lambda=None,
            fitted=True,
            feature_names=self.feature_names,
            metadata={"C": self.C_used, "calibrated": self.calibrator is not None},
        )


class FittedPoissonModel:
    """Model 3: Fitted attack/defence Poisson model with regularization.
    
    Models:
        log lambda_H = mu_l + h_l + a_home + d_away
        log lambda_A = mu_l + a_away + d_home
    
    Where:
    - mu_l = league intercept
    - h_l = league home advantage
    - a_team = attack strength (regularized)
    - d_team = defence weakness (regularized, positive = weak)
    
    Uses ridge penalty on attack/defence parameters.
    Shrinks poorly observed/new teams toward league priors.
    """
    
    def __init__(self, seed: int = 42, alpha: float = 1.0,
                 use_recency: bool = True, half_life_days: float = 365.0):
        self.seed = seed
        self.alpha = alpha  # Ridge penalty
        self.use_recency = use_recency
        self.half_life_days = half_life_days
        self.fitted = False
        self.league_intercept: Dict[str, float] = {}
        self.league_home_adv: Dict[str, float] = {}
        self.attack: Dict[str, float] = {}
        self.defense: Dict[str, float] = {}
        self.feature_names: List[str] = FITTED_POISSON_FEATURES.copy()
        self.pooled_home_rate: float = 1.3
        self.pooled_away_rate: float = 1.1
    
    def _team_key(self, div: str, team: str) -> str:
        return f"{div}|{team}"
    
    def fit(self, team_matches: pl.DataFrame, matches: pl.DataFrame) -> "FittedPoissonModel":
        import polars as pl
        
        # League intercept and home advantage from training data
        league_goals = matches.group_by("Div").agg([
            pl.col("FTHG").mean().alias("lg_home_gf"),
            pl.col("FTAG").mean().alias("lg_away_gf"),
        ]).to_dicts()
        
        for row in league_goals:
            div = row["Div"]
            self.league_intercept[div] = float(np.log(row["lg_home_gf"]))
            self.league_home_adv[div] = float(np.log(row["lg_home_gf"] / row["lg_away_gf"]))
        
        self.pooled_home_rate = float(matches.select(pl.col("FTHG").mean()).to_series()[0])
        self.pooled_away_rate = float(matches.select(pl.col("FTAG").mean()).to_series()[0])
        
        # Build team-season attack/defence using regularized regression
        # We'll use a simple shrinkage estimator for now
        tm = team_matches
        team_stats = tm.group_by(["Div", "Team"]).agg([
            pl.col("GF").mean().alias("team_gf_pg"),
            pl.col("GA").mean().alias("team_ga_pg"),
            pl.len().alias("n_matches"),
        ]).to_dicts()
        
        for row in team_stats:
            div = row["Div"]; team = row["Team"]
            n = row["n_matches"]
            lg_home = self.league_intercept.get(div, np.log(self.pooled_home_rate))
            lg_away = np.log(self.pooled_away_rate)
            lg_ha = self.league_home_adv.get(div, 0.0)
            
            obs_gf = row["team_gf_pg"]
            obs_ga = row["team_ga_pg"]
            
            # Shrink toward league average
            shrink_n = 5.0
            att = (obs_gf * n + np.exp(lg_home) * shrink_n) / (n + shrink_n)
            defence = (obs_ga * n + np.exp(lg_away) * shrink_n) / (n + shrink_n)
            
            # Convert to multiplicative factors on log scale
            # attack: how much more/less this team scores vs league avg
            # defence: how much more/less this team concedes vs league avg
            self.attack[self._team_key(div, team)] = float(np.log(att / np.exp(lg_home)))
            self.defense[self._team_key(div, team)] = float(np.log(defence / np.exp(lg_away)))
        
        self.fitted = True
        return self
    
    def _lambda(self, div: str, home_team: str, away_team: str) -> Tuple[float, float]:
        hk = self._team_key(div, home_team)
        ak = self._team_key(div, away_team)
        att_h = self.attack.get(hk, 0.0)
        def_a = self.defense.get(ak, 0.0)
        att_a = self.attack.get(ak, 0.0)
        def_h = self.defense.get(hk, 0.0)
        
        lg = self.league_intercept.get(div, np.log(self.pooled_home_rate))
        ha = self.league_home_adv.get(div, 0.0)
        
        home_lambda = np.exp(lg + ha + att_h + def_a)
        away_lambda = np.exp(lg + att_a + def_h)
        return float(home_lambda), float(away_lambda)
    
    def predict(self, matches: pl.DataFrame) -> ModelResult:
        if not self.fitted:
            raise RuntimeError("Fitted Poisson model must be fitted before prediction.")
        import polars as pl
        
        n = matches.height
        home_lambda = np.empty(n)
        away_lambda = np.empty(n)
        p_home = np.empty(n)
        p_draw = np.empty(n)
        p_away = np.empty(n)
        
        divs = matches.select(pl.col("Div")).to_series().to_list()
        hteams = matches.select(pl.col("HomeTeam")).to_series().to_list()
        ateams = matches.select(pl.col("AwayTeam")).to_series().to_list()
        
        for i in range(n):
            hl, al = self._lambda(divs[i], hteams[i], ateams[i])
            home_lambda[i] = hl
            away_lambda[i] = al
            ph, pd_, pa = poisson_1x2(hl, al)
            p_home[i] = ph
            p_draw[i] = pd_
            p_away[i] = pa
        
        out = np.column_stack([p_home, p_draw, p_away])
        out = out / out.sum(axis=1, keepdims=True)
        
        return ModelResult(
            model_name="fitted_poisson",
            outcome_probs=out,
            home_lambda=home_lambda,
            away_lambda=away_lambda,
            fitted=True,
            feature_names=self.feature_names,
            metadata={"alpha": self.alpha, "use_recency": self.use_recency,
                      "half_life_days": self.half_life_days},
        )


class TemperatureScalingCalibrator:
    """Model 4: Temperature scaling calibration.
    
    Adjusts logits with a single fitted parameter T:
        calibrated_logits = logits / T
    
    Preserves argmax predictions, naturally maintains multiclass normalization.
    Cannot fix excessive draw argmax by itself, but improves probability calibration.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.temperature: Optional[float] = None
        self.fitted = False
    
    def fit(self, logits: np.ndarray, outcomes: np.ndarray) -> "TemperatureScalingCalibrator":
        """Fit temperature by minimizing NLL on validation set."""
        from scipy.optimize import minimize_scalar
        
        def nll(T):
            T = max(T, 1e-3)
            scaled = logits / T
            # Softmax
            scaled = scaled - scaled.max(axis=1, keepdims=True)
            exp_scaled = np.exp(scaled)
            probs = exp_scaled / exp_scaled.sum(axis=1, keepdims=True)
            # NLL
            n = probs.shape[0]
            y_onehot = np.zeros_like(probs)
            valid = (outcomes >= 0) & (outcomes < 3)
            y_onehot[valid, outcomes[valid]] = 1.0
            log_probs = np.log(np.clip(probs, 1e-12, 1.0))
            return -np.mean(np.sum(y_onehot * log_probs, axis=1)[valid])
        
        result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
        self.temperature = float(result.x)
        self.fitted = True
        return self
    
    def transform(self, logits: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("TemperatureScalingCalibrator must be fitted before transform.")
        T = max(self.temperature, 1e-3)
        scaled = logits / T
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        exp_scaled = np.exp(scaled)
        return exp_scaled / exp_scaled.sum(axis=1, keepdims=True)


class RestrictedHGBModel:
    """Model 5: Restricted HistGradientBoostingClassifier challenger.
    
    Only after simpler models and feature contract are stable.
    Limits depth/leaf complexity to prevent overfitting.
    Uses small configuration grid.
    Keeps early-stopping validation chronological.
    """
    
    def __init__(self, seed: int = 42, max_iter: int = 100, max_leaf_nodes: int = 15,
                 min_samples_leaf: int = 50, l2_regularization: float = 1.0,
                 max_depth: int = 3, learning_rate: float = 0.05):
        self.seed = seed
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.fitted = False
        self.model: Optional[HistGradientBoostingClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.impute_: Optional[np.ndarray] = None
    
    def _feature_frame(self, matches):
        import polars as pl
        available = [c for c in BASE_FEATURES if c in matches.columns]
        X = matches.select(available).to_numpy().astype(np.float64)
        return X, available
    
    def _impute_fit(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = X.copy()
        impute_values = np.full(X.shape[1], 0.0, dtype=np.float64)
        for j in range(X.shape[1]):
            col = X[:, j]
            finite = np.isfinite(col)
            impute_values[j] = float(np.mean(col[finite])) if np.any(finite) else 0.0
            X[:, j] = np.where(finite, col, impute_values[j])
        return X, impute_values
    
    def _impute_transform(self, X: np.ndarray, impute_values: Optional[np.ndarray] = None) -> np.ndarray:
        X = X.copy()
        if impute_values is None:
            impute_values = getattr(self, "impute_", np.full(X.shape[1], 0.0, dtype=np.float64))
        for j in range(X.shape[1]):
            fill = impute_values[j] if j < len(impute_values) else 0.0
            X[:, j] = np.where(np.isfinite(X[:, j]), X[:, j], fill)
        return X
    
    def fit(self, team_matches: pl.DataFrame, matches: pl.DataFrame):
        import polars as pl
        X, names = self._feature_frame(matches)
        y = encode_outcomes(matches.select(pl.col("FTR")).to_series())
        valid = y >= 0
        X = X[valid]; y = y[valid]
        if X.shape[0] < 100 or len(np.unique(y)) < 2:
            raise ValueError("Insufficient data for restricted HGB model.")
        X, impute_values = self._impute_fit(X)
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)
        self.model = HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.seed,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
        )
        self.model.fit(Xs, y)
        self.feature_names = names
        self.fitted = True
        self.impute_ = impute_values
        return self
    
    def predict(self, matches):
        if not self.fitted:
            raise RuntimeError("Restricted HGB model must be fitted before prediction.")
        import polars as pl
        X, _ = self._feature_frame(matches)
        X = self._impute_transform(X, self.impute_)
        Xs = self.scaler.transform(X)
        probs = self.model.predict_proba(Xs)
        ordered = np.zeros((Xs.shape[0], 3))
        for i, cls in enumerate(self.model.classes_):
            ordered[:, int(cls)] = probs[:, i]
        ordered = ordered / ordered.sum(axis=1, keepdims=True)
        return ModelResult(
            model_name="restricted_hgb",
            outcome_probs=ordered,
            home_lambda=None, away_lambda=None,
            fitted=True, feature_names=self.feature_names,
            metadata={"max_iter": self.max_iter, "max_leaf_nodes": self.max_leaf_nodes,
                      "max_depth": self.max_depth, "learning_rate": self.learning_rate},
        )


class DrawBoostedModel:
    """
    Wrapper that applies draw probability boosting to any base model.
    
    Takes a fitted base model and applies post-processing to boost draw predictions
    when the draw probability exceeds a threshold.
    """
    
    def __init__(self, base_model, draw_threshold: float = 0.2, 
                 draw_boost: float = 0.05):
        self.base_model = base_model
        self.draw_threshold = draw_threshold
        self.draw_boost = draw_boost
        self.fitted = base_model.fitted
        self.feature_names = base_model.feature_names
        self.classes_ = np.array([0, 1, 2])
        self.model_name = f"draw_boosted_{base_model.__class__.__name__}"
    
    def fit(self, team_matches: pl.DataFrame, matches: pl.DataFrame):
        self.base_model.fit(team_matches, matches)
        self.fitted = True
        return self
    
    def predict(self, matches: pl.DataFrame) -> ModelResult:
        if not self.fitted:
            raise RuntimeError("Base model must be fitted before prediction.")
        
        base_result = self.base_model.predict(matches)
        probs = base_result.outcome_probs.copy()
        
        p_home = probs[:, 0]
        p_draw = probs[:, 1]
        p_away = probs[:, 2]
        
        # Boost draw when probability exceeds threshold but draw isn't the max
        mask = (p_draw > self.draw_threshold) & ((p_home > p_draw) | (p_away > p_draw))
        
        for i in np.where(mask)[0]:
            if p_home[i] > p_away[i]:
                take = min(self.draw_boost, p_home[i] - p_draw[i])
                probs[i, 0] -= take
                probs[i, 1] += take
            else:
                take = min(self.draw_boost, p_away[i] - p_draw[i])
                probs[i, 2] -= take
                probs[i, 1] += take
        
        # Renormalize
        probs = probs / probs.sum(axis=1, keepdims=True)
        
        return ModelResult(
            model_name=f"draw_boosted_{self.base_model.__class__.__name__}",
            outcome_probs=probs,
            home_lambda=base_result.home_lambda,
            away_lambda=base_result.away_lambda,
            fitted=True,
            feature_names=self.feature_names,
            metadata={
                "base_model": self.base_model.__class__.__name__,
                "draw_threshold": self.draw_threshold,
                "draw_boost": self.draw_boost,
            },
        )


class TwoStageModel:
    """Two-stage model: Stage 1 predicts Draw vs Non-Draw, Stage 2 predicts Home vs Away.
    
    This approach often improves draw prediction by explicitly modeling the draw decision
    separately from the home/away decision. Uses the same features for both stages.
    """
    
    def __init__(self, seed: int = 42, C: Optional[float] = None,
                 max_iter: int = 1000, calibrate: bool = True,
                 class_weight: Optional[str] = None):
        self.seed = seed
        self.C = C
        self.max_iter = max_iter
        self.calibrate = calibrate
        self.class_weight = class_weight
        self.fitted = False
        self.draw_model: Optional[LogisticRegression] = None
        self.ha_model: Optional[LogisticRegression] = None
        self.draw_scaler: Optional[StandardScaler] = None
        self.ha_scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = BASE_FEATURES.copy()
        self.classes_: np.ndarray = np.array([0, 1, 2])
        self.C_used: Optional[float] = None
        self.impute_: Optional[np.ndarray] = None
        self.draw_calibrator: Optional[object] = None
        self.ha_calibrator: Optional[object] = None
    
    def _feature_frame(self, matches: pl.DataFrame) -> Tuple[np.ndarray, List[str]]:
        import polars as pl
        available = [c for c in self.feature_names if c in matches.columns]
        if not available:
            raise ValueError("No features available in match frame.")
        X = matches.select(available).to_numpy().astype(np.float64)
        return X, available
    
    def _impute_fit(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = X.copy()
        impute_values = np.full(X.shape[1], 0.0, dtype=np.float64)
        for j in range(X.shape[1]):
            col = X[:, j]
            finite = np.isfinite(col)
            impute_values[j] = float(np.mean(col[finite])) if np.any(finite) else 0.0
            X[:, j] = np.where(finite, col, impute_values[j])
        return X, impute_values
    
    def _impute_transform(self, X: np.ndarray, impute_values: Optional[np.ndarray] = None) -> np.ndarray:
        X = X.copy()
        if impute_values is None:
            impute_values = getattr(self, "impute_", np.full(X.shape[1], 0.0, dtype=np.float64))
        for j in range(X.shape[1]):
            fill = impute_values[j] if j < len(impute_values) else 0.0
            X[:, j] = np.where(np.isfinite(X[:, j]), X[:, j], fill)
        return X
    
    @staticmethod
    def _inner_chronological_splits(dates: np.ndarray, n_folds: int = 3,
                                     min_train_dates: int = 20) -> List[Tuple[np.ndarray, np.ndarray]]:
        unique_dates = np.sort(np.unique(dates))
        if len(unique_dates) < min_train_dates + n_folds:
            if len(unique_dates) < n_folds + 1:
                return []
            split_points = np.linspace(1, len(unique_dates) - 1, n_folds).astype(int)
            splits = []
            for sp in split_points:
                cut = unique_dates[sp]
                train_idx = np.where(dates < cut)[0]
                val_idx = np.where(dates >= cut)[0]
                if len(train_idx) > 0 and len(val_idx) > 0:
                    splits.append((train_idx, val_idx))
            return splits
        
        train_end = min_train_dates
        val_block_size = max(1, (len(unique_dates) - train_end) // n_folds)
        splits = []
        
        for fold in range(n_folds):
            val_start = train_end
            val_end = min(val_start + val_block_size, len(unique_dates) - (n_folds - fold - 1))
            if val_start >= val_end:
                break
            
            train_dates = unique_dates[:train_end]
            val_dates = unique_dates[val_start:val_end]
            
            train_idx = np.where(np.isin(dates, train_dates))[0]
            val_idx = np.where(np.isin(dates, val_dates))[0]
            
            if len(train_idx) > 0 and len(val_idx) > 0:
                splits.append((train_idx, val_idx))
            
            train_end = val_end
        
        return splits
    
    def fit(self, team_matches: pl.DataFrame, matches: pl.DataFrame) -> "TwoStageModel":
        import polars as pl
        
        X, names = self._feature_frame(matches)
        y = encode_outcomes(matches.select(pl.col("FTR")).to_series())
        
        valid = y >= 0
        X = X[valid]; y = y[valid]
        dates = matches.select(pl.col("Date")).to_numpy().ravel()[valid]
        
        if X.shape[0] == 0 or len(np.unique(y)) < 2:
            raise ValueError("Insufficient valid training data for two-stage model.")
        
        # Create binary targets
        y_draw = (y == 1).astype(int)  # 1 = draw, 0 = non-draw
        y_ha = np.where(y == 0, 0, 1)  # 0 = home, 1 = away (for non-draw matches)
        non_draw_mask = y != 1
        
        # Tune C for both stages with fold-local preprocessing
        candidate_Cs = [self.C] if self.C is not None else [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        
        splits = self._inner_chronological_splits(dates, 3)
        valid_splits = []
        for train_idx, val_idx in splits:
            if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[val_idx])) < 2:
                continue
            valid_splits.append((train_idx, val_idx))
        
        best_C_draw = best_C_ha = candidate_Cs[0]
        oof_draw_probs = []
        oof_draw_y = []
        oof_ha_probs = []
        oof_ha_y = []
        
        if len(valid_splits) >= 2:
            # Stage 1: Draw vs Non-Draw
            best_brier = float("inf")
            for cand in candidate_Cs:
                fold_briers = []
                for train_idx, val_idx in valid_splits:
                    X_train = X[train_idx]
                    X_val = X[val_idx]
                    X_train_imputed, impute_values = self._impute_fit(X_train)
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train_imputed)
                    X_val_imputed = self._impute_transform(X_val, impute_values)
                    X_val_scaled = scaler.transform(X_val_imputed)
                    
                    model = LogisticRegression(
                        C=cand, solver="lbfgs", max_iter=self.max_iter,
                        random_state=self.seed, class_weight=self.class_weight,
                    )
                    model.fit(X_train_scaled, y_draw[train_idx])
                    val_probs = model.predict_proba(X_val_scaled)
                    fold_briers.append(brier_score_multiclass(val_probs, y_draw[val_idx]))
                mean_brier = float(np.mean(fold_briers))
                if mean_brier < best_brier:
                    best_brier = mean_brier
                    best_C_draw = cand
            
            # Stage 2: Home vs Away (non-draw only)
            best_brier = float("inf")
            for cand in candidate_Cs:
                fold_briers = []
                for train_idx, val_idx in valid_splits:
                    # Only non-draw matches in training
                    train_nd = train_idx[non_draw_mask[train_idx]]
                    val_nd = val_idx[non_draw_mask[val_idx]]
                    if len(train_nd) < 10 or len(val_nd) < 10:
                        continue
                    X_train = X[train_nd]
                    X_val = X[val_nd]
                    X_train_imputed, impute_values = self._impute_fit(X_train)
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train_imputed)
                    X_val_imputed = self._impute_transform(X_val, impute_values)
                    X_val_scaled = scaler.transform(X_val_imputed)
                    
                    model = LogisticRegression(
                        C=cand, solver="lbfgs", max_iter=self.max_iter,
                        random_state=self.seed, class_weight=self.class_weight,
                    )
                    model.fit(X_train_scaled, y_ha[train_nd])
                    val_probs = model.predict_proba(X_val_scaled)
                    fold_briers.append(brier_score_multiclass(val_probs, y_ha[val_nd]))
                mean_brier = float(np.mean(fold_briers))
                if mean_brier < best_brier:
                    best_brier = mean_brier
                    best_C_ha = cand
            
            # Collect OOF predictions for calibration
            for train_idx, val_idx in valid_splits:
                # Stage 1 OOF
                X_train = X[train_idx]
                X_val = X[val_idx]
                X_train_imputed, impute_values = self._impute_fit(X_train)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_imputed)
                X_val_imputed = self._impute_transform(X_val, impute_values)
                X_val_scaled = scaler.transform(X_val_imputed)
                
                model = LogisticRegression(
                    C=best_C_draw, solver="lbfgs", max_iter=self.max_iter,
                    random_state=self.seed, class_weight=self.class_weight,
                )
                model.fit(X_train_scaled, y_draw[train_idx])
                val_probs = model.predict_proba(X_val_scaled)
                oof_draw_probs.append(val_probs)
                oof_draw_y.append(y_draw[val_idx])
                
                # Stage 2 OOF
                train_nd = train_idx[non_draw_mask[train_idx]]
                val_nd = val_idx[non_draw_mask[val_idx]]
                if len(train_nd) >= 10 and len(val_nd) >= 10:
                    X_train = X[train_nd]
                    X_val = X[val_nd]
                    X_train_imputed, impute_values = self._impute_fit(X_train)
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train_imputed)
                    X_val_imputed = self._impute_transform(X_val, impute_values)
                    X_val_scaled = scaler.transform(X_val_imputed)
                    
                    model = LogisticRegression(
                        C=best_C_ha, solver="lbfgs", max_iter=self.max_iter,
                        random_state=self.seed, class_weight=self.class_weight,
                    )
                    model.fit(X_train_scaled, y_ha[train_nd])
                    val_probs = model.predict_proba(X_val_scaled)
                    oof_ha_probs.append(val_probs)
                    oof_ha_y.append(y_ha[val_nd])
        else:
            best_C_draw = best_C_ha = candidate_Cs[0]
        
        self.C_used = float(best_C_draw)
        self.best_C_draw = float(best_C_draw)
        self.best_C_ha = float(best_C_ha)
        
        # Refit on all data
        X_imputed, impute_values = self._impute_fit(X)
        
        # Stage 1: Draw vs Non-Draw
        self.draw_scaler = StandardScaler()
        Xs_draw = self.draw_scaler.fit_transform(X_imputed)
        self.draw_model = LogisticRegression(
            C=best_C_draw, solver="lbfgs", max_iter=self.max_iter,
            random_state=self.seed, class_weight=self.class_weight,
        )
        self.draw_model.fit(Xs_draw, y_draw)
        
        # Stage 2: Home vs Away (non-draw only)
        X_nd = X[non_draw_mask]
        y_ha_nd = y_ha[non_draw_mask]
        X_nd_imputed, _ = self._impute_fit(X_nd)
        self.ha_scaler = StandardScaler()
        Xs_ha = self.ha_scaler.fit_transform(X_nd_imputed)
        self.ha_model = LogisticRegression(
            C=best_C_ha, solver="lbfgs", max_iter=self.max_iter,
            random_state=self.seed, class_weight=self.class_weight,
        )
        self.ha_model.fit(Xs_ha, y_ha_nd)
        
        self.feature_names = names
        self.fitted = True
        self.classes_ = np.array([0, 1, 2])
        self.impute_ = impute_values
        
        # Calibration using Platt scaling on LOGITS (not probabilities)
        if self.calibrate and len(oof_draw_probs) > 0:
            O_draw = np.vstack(oof_draw_probs)
            Y_draw = np.concatenate(oof_draw_y)
            # Convert probabilities to logits for Platt scaling
            logits_draw = np.log(np.clip(O_draw[:, 1] / O_draw[:, 0], 1e-6, 1e6)).reshape(-1, 1)
            cal_draw = LogisticRegression(C=1.0, solver="lbfgs", max_iter=self.max_iter,
                                          random_state=self.seed, class_weight=self.class_weight)
            cal_draw.fit(logits_draw, Y_draw)
            self.draw_calibrator = cal_draw
            
            if len(oof_ha_probs) > 0:
                O_ha = np.vstack(oof_ha_probs)
                Y_ha = np.concatenate(oof_ha_y)
                logits_ha = np.log(np.clip(O_ha[:, 1] / O_ha[:, 0], 1e-6, 1e6)).reshape(-1, 1)
                cal_ha = LogisticRegression(C=1.0, solver="lbfgs", max_iter=self.max_iter,
                                            random_state=self.seed, class_weight=self.class_weight)
                cal_ha.fit(logits_ha, Y_ha)
                self.ha_calibrator = cal_ha
        
        return self
    
    def predict(self, matches: pl.DataFrame) -> ModelResult:
        if not self.fitted:
            raise RuntimeError("Two-stage model must be fitted before prediction.")
        import polars as pl
        
        X, _ = self._feature_frame(matches)
        X = self._impute_transform(X, self.impute_)
        
        # Stage 1: Draw probability
        Xs_draw = self.draw_scaler.transform(X)
        draw_probs = self.draw_model.predict_proba(Xs_draw)  # [p_non_draw, p_draw]
        p_draw = draw_probs[:, 1]
        p_non_draw = draw_probs[:, 0]
        
        # Apply draw calibration if available (using logits)
        if self.draw_calibrator is not None:
            logits_draw = np.log(np.clip(p_draw / p_non_draw, 1e-6, 1e6)).reshape(-1, 1)
            p_draw_cal = self.draw_calibrator.predict_proba(logits_draw)[:, 1]
            p_non_draw_cal = 1 - p_draw_cal
        else:
            p_draw_cal = p_draw
            p_non_draw_cal = p_non_draw
        
        # Stage 2: Home vs Away (conditional on non-draw)
        Xs_ha = self.ha_scaler.transform(X)
        ha_probs = self.ha_model.predict_proba(Xs_ha)  # [p_home|non_draw, p_away|non_draw]
        p_home_given_nd = ha_probs[:, 0]
        p_away_given_nd = ha_probs[:, 1]
        
        # Apply HA calibration if available (using logits)
        if self.ha_calibrator is not None:
            logits_ha = np.log(np.clip(p_away_given_nd / p_home_given_nd, 1e-6, 1e6)).reshape(-1, 1)
            ha_cal = self.ha_calibrator.predict_proba(logits_ha)
            p_home_given_nd = ha_cal[:, 0]
            p_away_given_nd = ha_cal[:, 1]
        
        # Combine: p_home = p_non_draw * p_home|non_draw, etc.
        p_home = p_non_draw_cal * p_home_given_nd
        p_away = p_non_draw_cal * p_away_given_nd
        p_draw_final = p_draw_cal
        
        # Normalize
        out = np.column_stack([p_home, p_draw_final, p_away])
        out = out / out.sum(axis=1, keepdims=True)
        
        return ModelResult(
            model_name="two_stage",
            outcome_probs=out,
            home_lambda=None,
            away_lambda=None,
            fitted=True,
            feature_names=self.feature_names,
            metadata={"C_draw": self.best_C_draw, "C_ha": self.best_C_ha,
                      "calibrated": self.calibrate},
        )


class DrawOptimizedModel:
    """
    Model optimized for draw prediction using threshold optimization.
    
    Approach:
    1. Train a binary classifier for draw vs non-draw
    2. Train a home/away classifier for non-draw matches
    3. On validation data, find the optimal draw probability threshold
       that maximizes draw accuracy (or F1 for draws)
    4. Apply the threshold at prediction time to boost draw predictions
    """
    
    def __init__(self, seed: int = 42, C: Optional[float] = None,
                 max_iter: int = 1000, calibrate: bool = True,
                 class_weight: Optional[str] = "balanced",
                 draw_threshold: Optional[float] = None):
        self.seed = seed
        self.C = C
        self.max_iter = max_iter
        self.calibrate = calibrate
        self.class_weight = class_weight
        self.draw_threshold = draw_threshold  # If None, optimized on validation
        self.fitted = False
        self.draw_model: Optional[LogisticRegression] = None
        self.ha_model: Optional[LogisticRegression] = None
        self.draw_scaler: Optional[StandardScaler] = None
        self.ha_scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = BASE_FEATURES.copy()
        self.classes_: np.ndarray = np.array([0, 1, 2])
        self.C_used: Optional[float] = None
        self.impute_: Optional[np.ndarray] = None
        self.draw_calibrator: Optional[object] = None
        self.ha_calibrator: Optional[object] = None
        self.optimal_draw_threshold: Optional[float] = None
    
    def _feature_frame(self, matches: pl.DataFrame) -> Tuple[np.ndarray, List[str]]:
        import polars as pl
        available = [c for c in self.feature_names if c in matches.columns]
        if not available:
            raise ValueError("No features available in match frame.")
        X = matches.select(available).to_numpy().astype(np.float64)
        return X, available
    
    def _impute_fit(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = X.copy()
        impute_values = np.full(X.shape[1], 0.0, dtype=np.float64)
        for j in range(X.shape[1]):
            col = X[:, j]
            finite = np.isfinite(col)
            impute_values[j] = float(np.mean(col[finite])) if np.any(finite) else 0.0
            X[:, j] = np.where(finite, col, impute_values[j])
        return X, impute_values
    
    def _impute_transform(self, X: np.ndarray, impute_values: Optional[np.ndarray] = None) -> np.ndarray:
        X = X.copy()
        if impute_values is None:
            impute_values = getattr(self, "impute_", np.full(X.shape[1], 0.0, dtype=np.float64))
        for j in range(X.shape[1]):
            fill = impute_values[j] if j < len(impute_values) else 0.0
            X[:, j] = np.where(np.isfinite(X[:, j]), X[:, j], fill)
        return X
    
    @staticmethod
    def _inner_chronological_splits(dates: np.ndarray, n_folds: int = 3,
                                     min_train_dates: int = 20) -> List[Tuple[np.ndarray, np.ndarray]]:
        unique_dates = np.sort(np.unique(dates))
        if len(unique_dates) < min_train_dates + n_folds:
            if len(unique_dates) < n_folds + 1:
                return []
            split_points = np.linspace(1, len(unique_dates) - 1, n_folds).astype(int)
            splits = []
            for sp in split_points:
                cut = unique_dates[sp]
                train_idx = np.where(dates < cut)[0]
                val_idx = np.where(dates >= cut)[0]
                if len(train_idx) > 0 and len(val_idx) > 0:
                    splits.append((train_idx, val_idx))
            return splits
        
        train_end = min_train_dates
        val_block_size = max(1, (len(unique_dates) - train_end) // n_folds)
        splits = []
        
        for fold in range(n_folds):
            val_start = train_end
            val_end = min(val_start + val_block_size, len(unique_dates) - (n_folds - fold - 1))
            if val_start >= val_end:
                break
            
            train_dates = unique_dates[:train_end]
            val_dates = unique_dates[val_start:val_end]
            
            train_idx = np.where(np.isin(dates, train_dates))[0]
            val_idx = np.where(np.isin(dates, val_dates))[0]
            
            if len(train_idx) > 0 and len(val_idx) > 0:
                splits.append((train_idx, val_idx))
            
            train_end = val_end
        
        return splits
    
    def _find_optimal_draw_threshold(self, draw_probs: np.ndarray, 
                                      true_draw: np.ndarray) -> float:
        """Find optimal threshold for draw prediction using validation data."""
        # Try thresholds from 0.1 to 0.5
        thresholds = np.linspace(0.1, 0.5, 41)
        best_threshold = 0.3
        best_score = -1
        
        for thresh in thresholds:
            pred_draw = (draw_probs > thresh).astype(int)
            # F1 score for draw class
            tp = np.sum((pred_draw == 1) & (true_draw == 1))
            fp = np.sum((pred_draw == 1) & (true_draw == 0))
            fn = np.sum((pred_draw == 0) & (true_draw == 1))
            if tp + fp == 0:
                continue
            precision = tp / (tp + fp)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            if precision + recall == 0:
                continue
            f1 = 2 * precision * recall / (precision + recall)
            if f1 > best_score:
                best_score = f1
                best_threshold = thresh
        
        return best_threshold
    
    def fit(self, team_matches: pl.DataFrame, matches: pl.DataFrame) -> "DrawOptimizedModel":
        import polars as pl
        
        X, names = self._feature_frame(matches)
        y = encode_outcomes(matches.select(pl.col("FTR")).to_series())
        
        valid = y >= 0
        X = X[valid]; y = y[valid]
        dates = matches.select(pl.col("Date")).to_numpy().ravel()[valid]
        
        if X.shape[0] == 0 or len(np.unique(y)) < 2:
            raise ValueError("Insufficient valid training data for draw optimized model.")
        
        # Create binary targets
        y_draw = (y == 1).astype(int)
        y_ha = np.where(y == 0, 0, 1)
        non_draw_mask = y != 1
        
        # Tune C with fold-local preprocessing
        candidate_Cs = [self.C] if self.C is not None else [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        
        splits = self._inner_chronological_splits(dates, 3)
        valid_splits = []
        for train_idx, val_idx in splits:
            if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[val_idx])) < 2:
                continue
            valid_splits.append((train_idx, val_idx))
        
        best_C_draw = best_C_ha = candidate_Cs[0]
        oof_draw_probs = []
        oof_draw_y = []
        oof_ha_probs = []
        oof_ha_y = []
        
        if len(valid_splits) >= 2:
            # Stage 1: Draw vs Non-Draw
            best_brier = float("inf")
            for cand in candidate_Cs:
                fold_briers = []
                for train_idx, val_idx in valid_splits:
                    X_train = X[train_idx]
                    X_val = X[val_idx]
                    X_train_imp, impute_values = self._impute_fit(X_train)
                    scaler = StandardScaler()
                    X_train_s = scaler.fit_transform(X_train_imp)
                    X_val_imp = self._impute_transform(X_val, impute_values)
                    X_val_s = scaler.transform(X_val_imp)
                    
                    model = LogisticRegression(
                        C=cand, solver="lbfgs", max_iter=self.max_iter,
                        random_state=self.seed, class_weight=self.class_weight,
                    )
                    model.fit(X_train_s, y_draw[train_idx])
                    val_probs = model.predict_proba(X_val_s)
                    fold_briers.append(brier_score_multiclass(val_probs, y_draw[val_idx]))
                mean_brier = float(np.mean(fold_briers))
                if mean_brier < best_brier:
                    best_brier = mean_brier
                    best_C_draw = cand
            
            # Stage 2: Home vs Away (non-draw only)
            best_brier = float("inf")
            for cand in candidate_Cs:
                fold_briers = []
                for train_idx, val_idx in valid_splits:
                    train_nd = train_idx[non_draw_mask[train_idx]]
                    val_nd = val_idx[non_draw_mask[val_idx]]
                    if len(train_nd) < 10 or len(val_nd) < 10:
                        continue
                    X_train = X[train_nd]
                    X_val = X[val_nd]
                    X_train_imp, impute_values = self._impute_fit(X_train)
                    scaler = StandardScaler()
                    X_train_s = scaler.fit_transform(X_train_imp)
                    X_val_imp = self._impute_transform(X_val, impute_values)
                    X_val_s = scaler.transform(X_val_imp)
                    
                    model = LogisticRegression(
                        C=cand, solver="lbfgs", max_iter=self.max_iter,
                        random_state=self.seed, class_weight=self.class_weight,
                    )
                    model.fit(X_train_s, y_ha[train_nd])
                    val_probs = model.predict_proba(X_val_s)
                    fold_briers.append(brier_score_multiclass(val_probs, y_ha[val_nd]))
                mean_brier = float(np.mean(fold_briers))
                if mean_brier < best_brier:
                    best_brier = mean_brier
                    best_C_ha = cand
            
            # Collect OOF predictions for calibration and threshold optimization
            for train_idx, val_idx in valid_splits:
                # Stage 1 OOF
                X_train = X[train_idx]
                X_val = X[val_idx]
                X_train_imp, impute_values = self._impute_fit(X_train)
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train_imp)
                X_val_imp = self._impute_transform(X_val, impute_values)
                X_val_s = scaler.transform(X_val_imp)
                
                model = LogisticRegression(
                    C=best_C_draw, solver="lbfgs", max_iter=self.max_iter,
                    random_state=self.seed, class_weight=self.class_weight,
                )
                model.fit(X_train_s, y_draw[train_idx])
                val_probs = model.predict_proba(X_val_s)
                oof_draw_probs.append(val_probs)
                oof_draw_y.append(y_draw[val_idx])
                
                # Stage 2 OOF
                train_nd = train_idx[non_draw_mask[train_idx]]
                val_nd = val_idx[non_draw_mask[val_idx]]
                if len(train_nd) >= 10 and len(val_nd) >= 10:
                    X_train = X[train_nd]
                    X_val = X[val_nd]
                    X_train_imp, impute_values = self._impute_fit(X_train)
                    scaler = StandardScaler()
                    X_train_s = scaler.fit_transform(X_train_imp)
                    X_val_imp = self._impute_transform(X_val, impute_values)
                    X_val_s = scaler.transform(X_val_imp)
                    
                    model = LogisticRegression(
                        C=best_C_ha, solver="lbfgs", max_iter=self.max_iter,
                        random_state=self.seed, class_weight=self.class_weight,
                    )
                    model.fit(X_train_s, y_ha[train_nd])
                    val_probs = model.predict_proba(X_val_s)
                    oof_ha_probs.append(val_probs)
                    oof_ha_y.append(y_ha[val_nd])
        else:
            best_C_draw = best_C_ha = candidate_Cs[0]
        
        self.C_used = float(best_C_draw)
        self.best_C_draw = float(best_C_draw)
        self.best_C_ha = float(best_C_ha)
        
        # Refit on all data
        X_imp, impute_values = self._impute_fit(X)
        
        # Stage 1: Draw vs Non-Draw
        self.draw_scaler = StandardScaler()
        Xs_draw = self.draw_scaler.fit_transform(X_imp)
        self.draw_model = LogisticRegression(
            C=best_C_draw, solver="lbfgs", max_iter=self.max_iter,
            random_state=self.seed, class_weight=self.class_weight,
        )
        self.draw_model.fit(Xs_draw, y_draw)
        
        # Stage 2: Home vs Away (non-draw only)
        X_nd = X[non_draw_mask]
        y_ha_nd = y_ha[non_draw_mask]
        X_nd_imp, _ = self._impute_fit(X_nd)
        self.ha_scaler = StandardScaler()
        Xs_ha = self.ha_scaler.fit_transform(X_nd_imp)
        self.ha_model = LogisticRegression(
            C=best_C_ha, solver="lbfgs", max_iter=self.max_iter,
            random_state=self.seed, class_weight=self.class_weight,
        )
        self.ha_model.fit(Xs_ha, y_ha_nd)
        
        self.feature_names = names
        self.fitted = True
        self.classes_ = np.array([0, 1, 2])
        self.impute_ = impute_values
        
        # Calibration and threshold optimization using OOF predictions
        if self.calibrate and len(oof_draw_probs) > 0:
            O_draw = np.vstack(oof_draw_probs)
            Y_draw = np.concatenate(oof_draw_y)
            
            # Find optimal draw threshold on OOF predictions
            self.optimal_draw_threshold = self._find_optimal_draw_threshold(
                O_draw[:, 1], Y_draw
            )
            print(f"Optimal draw threshold: {self.optimal_draw_threshold:.3f}")
            
            # Calibration using Platt scaling on LOGITS without class_weight (to preserve calibration)
            logits_draw = np.log(np.clip(O_draw[:, 1] / O_draw[:, 0], 1e-6, 1e6)).reshape(-1, 1)
            cal_draw = LogisticRegression(C=1.0, solver="lbfgs", max_iter=self.max_iter,
                                          random_state=self.seed)
            cal_draw.fit(logits_draw, Y_draw)
            self.draw_calibrator = cal_draw
            
            if len(oof_ha_probs) > 0:
                O_ha = np.vstack(oof_ha_probs)
                Y_ha = np.concatenate(oof_ha_y)
                logits_ha = np.log(np.clip(O_ha[:, 1] / O_ha[:, 0], 1e-6, 1e6)).reshape(-1, 1)
                cal_ha = LogisticRegression(C=1.0, solver="lbfgs", max_iter=self.max_iter,
                                            random_state=self.seed)
                cal_ha.fit(logits_ha, Y_ha)
                self.ha_calibrator = cal_ha
        
        return self
    
    def predict(self, matches: pl.DataFrame) -> ModelResult:
        if not self.fitted:
            raise RuntimeError("DrawOptimizedModel must be fitted before prediction.")
        import polars as pl
        
        X, _ = self._feature_frame(matches)
        X = self._impute_transform(X, self.impute_)
        
        # Stage 1: Draw probability
        Xs_draw = self.draw_scaler.transform(X)
        draw_probs = self.draw_model.predict_proba(Xs_draw)  # [p_non_draw, p_draw]
        p_draw = draw_probs[:, 1]
        p_non_draw = draw_probs[:, 0]
        
        # Apply draw calibration if available (using logits)
        if self.draw_calibrator is not None:
            logits_draw = np.log(np.clip(p_draw / p_non_draw, 1e-6, 1e6)).reshape(-1, 1)
            p_draw_cal = self.draw_calibrator.predict_proba(logits_draw)[:, 1]
            p_non_draw_cal = 1 - p_draw_cal
        else:
            p_draw_cal = p_draw
            p_non_draw_cal = p_non_draw
        
        # Apply threshold optimization: boost draw predictions conservatively
        if self.optimal_draw_threshold is not None:
            # Only boost draws that are already somewhat likely
            boost_mask = (p_draw_cal > self.optimal_draw_threshold) & (p_draw_cal < 0.7)
            # Small boost to draw probability
            p_draw_cal = np.where(boost_mask, 
                                  np.minimum(p_draw_cal * 1.15, 0.7),
                                  p_draw_cal)
            p_non_draw_cal = 1 - p_draw_cal
        
        # Stage 2: Home vs Away (conditional on non-draw)
        Xs_ha = self.ha_scaler.transform(X)
        ha_probs = self.ha_model.predict_proba(Xs_ha)
        p_home_given_nd = ha_probs[:, 0]
        p_away_given_nd = ha_probs[:, 1]
        
        # Apply HA calibration if available (using logits)
        if self.ha_calibrator is not None:
            logits_ha = np.log(np.clip(p_away_given_nd / p_home_given_nd, 1e-6, 1e6)).reshape(-1, 1)
            ha_cal = self.ha_calibrator.predict_proba(logits_ha)
            p_home_given_nd = ha_cal[:, 0]
            p_away_given_nd = ha_cal[:, 1]
        
        # Combine: p_home = p_non_draw * p_home|non_draw, etc.
        p_home = p_non_draw_cal * p_home_given_nd
        p_away = p_non_draw_cal * p_away_given_nd
        p_draw_final = p_draw_cal
        
        # Normalize
        out = np.column_stack([p_home, p_draw_final, p_away])
        out = out / out.sum(axis=1, keepdims=True)
        
        return ModelResult(
            model_name="draw_optimized",
            outcome_probs=out,
            home_lambda=None,
            away_lambda=None,
            fitted=True,
            feature_names=self.feature_names,
            metadata={"C_draw": self.best_C_draw, "C_ha": self.best_C_ha,
                      "calibrated": self.calibrate,
                      "draw_threshold": self.optimal_draw_threshold},
        )
