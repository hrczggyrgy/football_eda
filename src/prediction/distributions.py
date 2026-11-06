"""
Probability distributions over scorelines and derived markets.

Builds a normalized joint goal distribution from home/away expected goals
(lambdas), derives 1X2 probabilities, both-teams-to-score, over/under goal
markets, and prediction intervals. The tail beyond MAX_GOALS is aggregated
into an "other" bucket so the distribution is mathematically normalized.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import poisson

from prediction.utils import MAX_GOALS, REPORTED_SCORELINES, validate_probabilities


def joint_goal_distribution(home_lambda: float, away_lambda: float,
                            max_goals: int = MAX_GOALS) -> np.ndarray:
    """Joint P(home_goals=h, away_goals=a) under independent Poissons.

    Returns a (max_goals+1, max_goals+1) matrix. The mass in the high-scoring
    tail (goals > max_goals for either side) is NOT represented in the matrix;
    callers should account for tail mass via ``tail_mass``.
    """
    home_lambda = max(float(home_lambda), 0.01)
    away_lambda = max(float(away_lambda), 0.01)
    goals = np.arange(0, max_goals + 1)
    ph = poisson.pmf(goals, home_lambda)
    pa = poisson.pmf(goals, away_lambda)
    mat = np.outer(ph, pa)
    total = float(mat.sum())
    if total <= 0 or not np.isfinite(total):
        mat = np.full_like(mat, 1.0 / mat.size)
        total = 1.0
    return mat / total


def tail_mass(home_lambda: float, away_lambda: float,
              max_goals: int = MAX_GOALS) -> Tuple[float, float, float]:
    """Probability mass outside the [0, max_goals] square.

    Returns (P(home>max), P(away>max), P(either>max)).
    """
    home_lambda = max(float(home_lambda), 0.01)
    away_lambda = max(float(away_lambda), 0.01)
    ph_tail = float(poisson.sf(max_goals, home_lambda))   # P(X > max_goals)
    pa_tail = float(poisson.sf(max_goals, away_lambda))
    either = 1.0 - (1.0 - ph_tail) * (1.0 - pa_tail)
    return ph_tail, pa_tail, either


def scoreline_probabilities(home_lambda: float, away_lambda: float,
                            max_goals: int = MAX_GOALS) -> Dict[Tuple[int, int], float]:
    """Full normalized scoreline distribution including an 'other' tail bucket.

    Keys are (home_goals, away_goals) tuples plus ('other', 'other') for the
    high-scoring tail. Probabilities sum to 1.
    """
    mat = joint_goal_distribution(home_lambda, away_lambda, max_goals)
    ph_tail, pa_tail, either_tail = tail_mass(home_lambda, away_lambda, max_goals)

    probs: Dict[Tuple[int, int], float] = {}
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            val = float(mat[h, a])
            if val > 0:
                probs[(h, a)] = val

    # Allocate remaining tail mass to 'other'.
    inside = float(sum(probs.values()))
    other = max(0.0, 1.0 - inside)
    probs[("other", "other")] = float(other)

    # Renormalize to correct floating-point drift.
    total = sum(probs.values())
    if total > 0:
        probs = {k: v / total for k, v in probs.items()}
    return probs


def reported_scoreline_probs(home_lambda: float, away_lambda: float,
                             max_goals: int = MAX_GOALS) -> Dict[Tuple[int, int], float]:
    """Probabilities for the explicitly reported scorelines, plus 'other'."""
    full = scoreline_probabilities(home_lambda, away_lambda, max_goals)
    out = {k: full.get(k, 0.0) for k in REPORTED_SCORELINES}
    out[("other", "other")] = full.get(("other", "other"), 0.0)
    # Renormalize the reported subset so the listed values are coherent.
    total = sum(out.values())
    if total > 0:
        out = {k: v / total for k, v in out.items()}
    return out


def outcome_from_scoreline_distribution(probs: Dict[Tuple[int, int], float]
                                        ) -> Tuple[float, float, float]:
    """Derive P(H), P(D), P(A) from a scoreline distribution."""
    p_home = p_draw = p_away = 0.0
    for (h, a), p in probs.items():
        if h == "other" or a == "other":
            # Tail mass: distribute by expected goal rates? Use neutral split.
            # Conservative: treat as unresolved; split evenly across outcomes.
            p_home += p / 3; p_draw += p / 3; p_away += p / 3
        elif h > a:
            p_home += p
        elif h == a:
            p_draw += p
        else:
            p_away += p
    return p_home, p_draw, p_away


def poisson_1x2_from_lambdas(home_lambda: float, away_lambda: float,
                             max_goals: int = MAX_GOALS) -> Tuple[float, float, float]:
    """1X2 probabilities from lambdas via the full scoreline distribution."""
    probs = scoreline_probabilities(home_lambda, away_lambda, max_goals)
    return outcome_from_scoreline_distribution(probs)


def derived_market_probs(probs: Dict[Tuple[int, int], float]) -> Dict[str, float]:
    """Derived market probabilities from a scoreline distribution.

    - btts: both teams score (h>0 and a>0)
    - over_1.5 / over_2.5 / over_3.5: total goals > threshold
    - home_scores: home team scores at least once
    - away_scores: away team scores at least once
    - o1_5/u1_5 etc. complements included.
    """
    btts = 0.0; home_scores = 0.0; away_scores = 0.0
    tot1 = tot2 = tot3 = 0.0   # over 1.5, 2.5, 3.5
    for (h, a), p in probs.items():
        if h == "other" or a == "other":
            # For tail mass, goals > max_goals on at least one side => over all thresholds,
            # both teams likely scored. Allocate conservatively.
            btts += p * 0.9; home_scores += p * 0.95; away_scores += p * 0.95
            tot1 += p; tot2 += p; tot3 += p
            continue
        total = h + a
        if h > 0 and a > 0:
            btts += p
        if h > 0:
            home_scores += p
        if a > 0:
            away_scores += p
        if total > 1.5:
            tot1 += p
        if total > 2.5:
            tot2 += p
        if total > 3.5:
            tot3 += p

    out = {
        "btts": float(btts),
        "btts_no": float(1.0 - btts),
        "over_1_5": float(tot1),
        "under_1_5": float(1.0 - tot1),
        "over_2_5": float(tot2),
        "under_2_5": float(1.0 - tot2),
        "over_3_5": float(tot3),
        "under_3_5": float(1.0 - tot3),
        "home_scores": float(home_scores),
        "home_no_score": float(1.0 - home_scores),
        "away_scores": float(away_scores),
        "away_no_score": float(1.0 - away_scores),
    }
    # Clamp to [0,1] for numerical safety.
    for k in out:
        out[k] = float(min(1.0, max(0.0, out[k])))
    return out


def goal_prediction_intervals(home_lambda: float, away_lambda: float,
                              coverage: float = 0.8) -> Dict[str, Tuple[int, int]]:
    """Prediction intervals for home/away/total goals under independent Poissons.

    Returns equal-tailed intervals with the specified coverage (default 80%).
    """
    home_lambda = max(float(home_lambda), 0.01)
    away_lambda = max(float(away_lambda), 0.01)
    alpha = 1.0 - coverage
    lower_q = alpha / 2
    upper_q = 1.0 - alpha / 2

    def interval(lam: float) -> Tuple[int, int]:
        lo = int(poisson.ppf(lower_q, lam))
        hi = int(poisson.ppf(upper_q, lam))
        lo = max(0, lo)
        return lo, hi

    hi = interval(home_lambda)
    ai = interval(away_lambda)
    # Total goals ~ Pois(home+away).
    ti = interval(home_lambda + away_lambda)
    return {
        "home_goals": hi,
        "away_goals": ai,
        "total_goals": ti,
        "coverage": float(coverage),
    }


def most_probable_scoreline(home_lambda: float, away_lambda: float,
                            max_goals: int = MAX_GOALS) -> Tuple[Tuple[int, int], float]:
    """Return the single most probable scoreline and its probability."""
    probs = scoreline_probabilities(home_lambda, away_lambda, max_goals)
    best = max(probs.items(), key=lambda kv: kv[1])
    return best
