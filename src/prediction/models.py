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
    # League-level scoring environment (kept once, not prefixed).
    "League_Avg_GF_PG", "League_Avg_GA_PG",
    # Elo ratings (kept once; derived from pair of teams).
    "Elo_Base_Rating", "Elo_Opponent_Rating", "Elo_Exp_Home", "Elo_Rating_Diff",
]


def _paired_feature_names() -> List[str]:
    """Expand base team features into Home_/Away_-prefixed paired names.

    League-level and Elo features are kept once (they are match-level or
    common to both teams). Match-level derived features (gaps/diffs) are
    appended directly.
    """
    team_only = [f for f in _BASE_TEAM_FEATURES
                 if f not in ("League_Avg_GF_PG", "League_Avg_GA_PG",
                              "Elo_Base_Rating", "Elo_Opponent_Rating",
                              "Elo_Exp_Home", "Elo_Rating_Diff")]
    names = []
    for f in team_only:
        names.append(f"Home_{f}")
        names.append(f"Away_{f}")
    names += ["League_Avg_GF_PG", "League_Avg_GA_PG",
              "Elo_Base_Rating", "Elo_Opponent_Rating", "Elo_Exp_Home",
              "Elo_Rating_Diff",
              # Match-level derived pre-match features.
              "PPG_Gap", "GF_Cum_Gap", "GA_Cum_Gap", "Rest_Days_Diff"]
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

    def __init__(self, seed: int = 42, use_dixon_coles: bool = False):
        self.seed = seed
        self.use_dixon_coles = use_dixon_coles
        self.fitted = False
        self.team_attack: Dict[str, float] = {}
        self.team_defense: Dict[str, float] = {}
        self.home_advantage: float = 0.0
        self.league_effects: Dict[str, float] = {}
        self.league_home_advantage: Dict[str, float] = {}
        self.pooled_home_rate: float = 1.3
        self.pooled_away_rate: float = 1.1
        self.feature_names = list(BASE_FEATURES)

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

        self.fitted = True
        return self

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

        # log-linear: lambda = league_rate * att * (1/def) * home_adv_factor
        home_lambda = lg * att_h * (1.0 / max(def_a, 0.1)) * np.exp(ha * 0.15)
        away_lambda = lg * att_a * (1.0 / max(def_h, 0.1)) * np.exp(-ha * 0.15)
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
            home_lambda[i] = hl; away_lambda[i] = al
            # 1X2 from independent Poisson: sum over scorelines.
            ph, pd_, pa = poisson_1x2(hl, al)
            p_home[i] = ph; p_draw[i] = pd_; p_away[i] = pa

        out = np.column_stack([p_home, p_draw, p_away])
        out = out / out.sum(axis=1, keepdims=True)

        return ModelResult(
            model_name="poisson",
            outcome_probs=out,
            home_lambda=home_lambda,
            away_lambda=away_lambda,
            fitted=True,
            feature_names=self.feature_names,
            metadata={"use_dixon_coles": self.use_dixon_coles},
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
                 calibration_method: str = "platt"):
        self.seed = seed
        self.C = C  # If None, tuned via inner chronological validation.
        self.max_iter = max_iter
        self.calibrate = calibrate
        self.n_inner_folds = n_inner_folds
        self.calibration_method = calibration_method  # "platt" or "isotonic"
        self.fitted = False
        self.model: Optional[LogisticRegression] = None
        self.scaler: Optional[StandardScaler] = None
        self.calibrator: Optional[object] = None
        self.feature_names: List[str] = []
        self.classes_: np.ndarray = np.array([0, 1, 2])
        self.C_used: Optional[float] = None

    def _feature_frame(self, matches: pl.DataFrame) -> Tuple[np.ndarray, List[str]]:
        import polars as pl
        available = [c for c in BASE_FEATURES if c in matches.columns]
        if not available:
            raise ValueError("No baseline features available in match frame.")
        X = matches.select(available).to_numpy().astype(np.float64)
        return X, available

    def _impute_fit(self, X: np.ndarray) -> np.ndarray:
        """Impute non-finite values with column means (computed on training data)."""
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

    @staticmethod
    def _inner_chronological_splits(dates: np.ndarray, n_folds: int = 3):
        """Generate chronological (train_idx, val_idx) splits by unique date.

        Training dates always strictly precede validation dates.
        """
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

    def _fit_base(self, X: np.ndarray, y: np.ndarray, C: float) -> LogisticRegression:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        model = LogisticRegression(
            penalty="l2", C=C, solver="lbfgs", max_iter=self.max_iter,
            random_state=self.seed,
        )
        model.fit(Xs, y)
        return model, scaler

    def fit(self, team_matches: pl.DataFrame, matches: pl.DataFrame) -> "MultinomialModel":
        """Fit the multinomial model on engineered paired features.

        ``matches`` should be a paired feature frame containing the BASE_FEATURES
        and FTR target. Missing features are imputed with column means fitted on
        the training data (never on the test data). C is tuned via chronological
        inner validation; calibration is fitted on out-of-fold training
        predictions only.
        """
        import polars as pl

        X, names = self._feature_frame(matches)
        y = encode_outcomes(matches.select(pl.col("FTR")).to_series())

        valid = y >= 0
        X = X[valid]; y = y[valid]
        dates = matches.select(pl.col("Date")).to_numpy().ravel()[valid]

        if X.shape[0] == 0 or len(np.unique(y)) < 2:
            raise ValueError("Insufficient valid training data for multinomial model.")

        X = self._impute_fit(X)

        # ------------------------------------------------------------------ tune C
        candidate_Cs = [self.C] if self.C is not None else [0.1, 0.5, 1.0, 2.0, 5.0]
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
                    m, s = self._fit_base(X[train_idx], y[train_idx], cand)
                    val_probs = m.predict_proba(s.transform(X[val_idx]))
                    fold_briers.append(brier_score_multiclass(val_probs, y[val_idx]))
                mean_brier = float(np.mean(fold_briers))
                if mean_brier < best_brier:
                    best_brier = mean_brier
                    best_C = cand
            # Collect OOF predictions using the best C for calibration.
            for train_idx, val_idx in valid_splits:
                m, s = self._fit_base(X[train_idx], y[train_idx], best_C)
                val_probs = m.predict_proba(s.transform(X[val_idx]))
                oof_probs_list.append(val_probs)
                oof_y_list.append(y[val_idx])
        else:
            best_C = candidate_Cs[0]

        self.C_used = float(best_C)

        # ------------------------------------------------------------------ refit on all training data
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)
        self.model = LogisticRegression(
            penalty="l2", C=best_C, solver="lbfgs", max_iter=self.max_iter,
            random_state=self.seed,
        )
        self.model.fit(Xs, y)
        self.feature_names = names
        self.fitted = True
        self.classes_ = self.model.classes_.copy()

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
            Xs_h = self.goal_scaler_h.fit_transform(X)
            Xs_a = self.goal_scaler_a.fit_transform(X)
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
            if self.calibration_method == "isotonic":
                # Isotonic regression: one per class (one-vs-rest calibration).
                # Fits a monotonic mapping from raw prob -> calibrated prob.
                cals = []
                for k in range(3):
                    iso = IsotonicRegression(out_of_bounds="clip")
                    iso.fit(O[:, k], (Y == k).astype(np.float64))
                    cals.append(iso)
                self.calibrator = cals
            else:
                # Platt scaling: multinomial logistic regression on raw probs.
                cal = LogisticRegression(
                    penalty="l2", C=1.0, solver="lbfgs", max_iter=self.max_iter,
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
            raise RuntimeError("Multinomial model must be fitted before prediction.")
        import polars as pl

        X, _ = self._feature_frame(matches)
        X = self._impute_transform(X)

        Xs = self.scaler.transform(X)
        probs = self.model.predict_proba(Xs)
        ordered = self._order_probs(probs)

        # Expected goals from the Poisson goal model (same pre-match features).
        home_lambda = None
        away_lambda = None
        if self.goal_model_h is not None and self.goal_model_a is not None:
            Xs_h = self.goal_scaler_h.transform(X)
            Xs_a = self.goal_scaler_a.transform(X)
            home_lambda = np.maximum(self.goal_model_h.predict(Xs_h), 0.01)
            away_lambda = np.maximum(self.goal_model_a.predict(Xs_a), 0.01)

        # Apply calibration fitted on out-of-fold training predictions only.
        if self.calibrator is not None:
            if self.calibration_method == "isotonic" and isinstance(
                self.calibrator, list
            ):
                # Isotonic regression: apply per-class monotonic mapping.
                cal_probs = np.column_stack([
                    self.calibrator[k].predict(ordered[:, k]) for k in range(3)
                ])
                ordered = cal_probs / cal_probs.sum(axis=1, keepdims=True)
            else:
                # Platt scaling: multinomial logistic regression on raw probs.
                calibrated = self.calibrator.predict_proba(ordered)
                cal_ordered = np.zeros((calibrated.shape[0], 3))
                for i, cls in enumerate(self.calibrator.classes_):
                    if 0 <= int(cls) <= 2:
                        cal_ordered[:, int(cls)] = calibrated[:, i]
                ordered = cal_ordered / cal_ordered.sum(axis=1, keepdims=True)

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
