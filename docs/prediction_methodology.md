# Prediction Methodology

**Football Analytics Study — Predictive Intelligence Layer**

This document describes the methodology of the predictive engine implemented in
`src/prediction/` and orchestrated by `run_predictions.py`. It extends the
existing exploratory data analysis (EDA) without modifying it.

## Guiding principle

Scientific validity > predictive reliability > interpretability >
computational efficiency > complexity.

No bookmaker odds are ever used as predictors. No look-ahead leakage is
permitted. Validation uses expanding chronological windows only. A model may
fail to beat the baseline; that is an acceptable scientific result.

## Scope

- Population: 7 seasons (2019/20–2025/26), 8 leagues (E0, E1, E2, E3, SP1,
  D1, I1, F1), 23,787 matches, 215 teams (football-data.co.uk).
- Betting/odds data: excluded throughout.
- Targets per fixture: 1X2 probabilities, expected goals (home/away/total),
  scoreline distribution, derived markets (BTTS, Over/Under 1.5/2.5/3.5),
  prediction intervals, and calibration diagnostics.

## 1. Feature engineering (strictly pre-match)

All features represent information available before kickoff. The feature
pipeline lives in `src/prediction/features.py` and reuses the EDA's lagged
rolling primitives (`src/features/engineer.py`).

### 1.1 Temporal windows

For each team, the following are computed on **shifted** (lagged) data so that
match t never sees its own result:

- Last 3, 5, 10 matches (rolling mean/std)
- Expanding (cumulative) historical average
- Points per game, goals for/against per game, goal difference per game,
  shots, shots on target, shots against, shot accuracy, corners

### 1.2 Team strength (shrinkage)

Pre-match points-per-game is shrunk toward prior-season PPG and the historical
league average using Bayesian/empirical shrinkage:

\[
\hat{\theta} = \frac{n\theta_{observed} + k\theta_{prior}}{n + k}
\]

where \(n\) is the exposure (matches played so far), \(\theta_{prior}\) is the
prior-season PPG (or historical league average when no prior season exists),
and \(k = 5\) is the shrinkage strength (pseudo-observations). Teams with few
matches are pulled toward the prior; teams with long histories approach their
observed value.

### 1.3 Opponent strength

The opponent's pre-match cumulative PPG is looked up from matches played
before the target fixture (never including the target). Features are
opponent-aware: a team's defensive record is interpreted relative to the
strength of opponents faced.

### 1.4 Venue splits

Home-specific and away-specific historical PPG and goals-for per game are
tracked separately.

### 1.5 League effects

League-average home and away goals per game are computed from **completed prior
seasons only**, so the current season never contributes to its own scoring
environment.

### 1.6 Promotion / relegation

Prior-season PPG is carried across consecutive seasons as a prior, but is not
treated as equivalent in the new division. Newly promoted / unseen teams fall
back to the shrinkage prior (prior-season PPG or historical league average).

### 1.7 Missing data

Missing statistics are reported per-feature, never silently converted to zero.
Ratios with zero/missing denominators are null. A core-feature model works
without optional match statistics (shots, corners, etc.); an enhanced model can
use them when sufficiently available.

## 2. Elo ratings (candidate enhancement)

`src/prediction/ratings.py` implements a chronological Elo system:
new teams initialize to the division mean, ratings update match-by-match in
date order, home advantage is added to the home side's rating for prediction
only (and removed before storage), the current match result never enters its
own prediction, and K is reduced for teams with few observations. Division
changes regress the rating toward the new division mean.

## 3. Models

### Model A — Historical baseline (mandatory)

League-specific outcome frequencies and home/away goal rates, shrunk toward the
pooled average by training-sample exposure. Produces outcome probabilities and
expected goals. Mandatory baseline that every complex model must beat.

### Model B — Poisson scoreline model

Home and away goals modeled as independent Poisson with log-linear rates built
from pre-match attack strength, defensive strength, league scoring
environment, and home advantage. Scoreline probabilities are the normalized
outer product of the two Poisson PMFs; the high-scoring tail beyond 8 goals is
aggregated into an "other" bucket so the distribution sums to 1. 1X2
probabilities are derived by summing the scoreline matrix.

Dixon–Coles low-score correction is implemented as an optional, disabled
extension (not used unless numerically stable and demonstrably better
chronologically).

### Model C — Multinomial logistic regression

L2-regularized multinomial logistic regression (sklearn `LogisticRegression`)
predicting P(H), P(D), P(A) directly from pre-match features. Features are
standardized with a training-fold scaler; missing features imputed with
training column means.

### Model D — HistGradientBoosting challenger

Available behind `--skip-hgb` (disabled by default). Evaluated only when
supported by sample size and temporal validation; not used unless it
demonstrably beats simpler models chronologically.

## 4. Validation (chronological, no random splits)

Expanding-window season-based rolling-origin evaluation (5 folds for the
7-season data):

| Fold | Training | Testing |
|------|----------|---------|
| 1 | 2019/20–2020/21 | 2021/22 |
| 2 | 2019/20–2021/22 | 2022/23 |
| 3 | 2019/20–2022/23 | 2023/24 |
| 4 | 2019/20–2023/24 | 2024/25 |
| 5 | 2019/20–2024/25 | 2025/26 |

At each fold the model is fit on training history only, frozen, and used to
predict every test fixture pre-match. Features are pre-computed once on the full
dataset using strictly lagged windows; this is mathematically equivalent to a
walk-forward simulation because no feature ever incorporates the current or a
future match.

Hyperparameters and calibration are never tuned on the held-out test seasons.

## 5. Probability distributions and derived markets

From home/away expected goals (\(\lambda_H, \lambda_A\)):

- Joint scoreline distribution: independent Poisson PMFs, outer product,
  normalized, tail aggregated into "other".
- 1X2: sum of scoreline matrix (home win: h>a, draw: h=a, away win: h<a).
- BTTS, Over/Under 1.5/2.5/3.5, home/away team scores: summed from the
  scoreline distribution.
- Prediction intervals: equal-tailed Poisson intervals (default 80% coverage).
- Most probable scoreline: argmax of the distribution.

## 6. Calibration

- Multiclass Brier score and log loss.
- Expected Calibration Error (ECE) with 10 equal-width bins on confidence.
- Reliability diagrams (mean predicted vs observed frequency per class).
- Calibration intercept/slope via logistic regression on the logit.
- Brier decomposition (reliability – resolution + uncertainty).

Calibration diagnostics are computed on out-of-fold chronological predictions,
never on held-out test data used for model selection.

## 7. Evaluation

1X2: multiclass log loss, multiclass Brier, ranked probability score, ECE,
accuracy (never the sole measure), per-class precision/recall.

Goals: home/away/total MAE, RMSE, Poisson deviance, predictive log-likelihood.

Performance is reported by league, season, fold, outcome class, and
data-availability subgroup, with sample sizes always stated.

## 8. Inference interface

`predict_fixture(division, date, home_team, away_team, bundle, history_matches,
history_team_matches)` returns a structured prediction with outcome
probabilities, expected goals, scoreline, derived markets, intervals, model
version, training cutoff, and data-quality status.

`predict_from_csv(csv_path, ...)` batch-predicts fixtures from a
`Date,Div,HomeTeam,AwayTeam` CSV (no results required).

Unknown teams or unsupported leagues receive a documented neutral prior rather
than a silent forecast; this is recorded in `data_quality_status` and
`feature_warnings`.

## 9. Persistence and reproducibility

Trained model bundles are saved with joblib plus sidecar metadata JSON
(`training_cutoff`, `training_seasons`, `feature_names`, `random_seed`,
software versions, validation results, calibration method, data-quality
diagnostics). A manifest records stage timings and the source-data fingerprint.
Deterministic random seeds are used throughout.

## 10. Leakage safeguards (demonstrated by tests)

`tests/test_prediction_leakage.py` verifies:

1. Changing a target match's result does not change its own pre-match features.
2. Adding future matches does not change past predictions/features.
3. Same-date results do not leak into pre-match features (strictly-before date
   filter when timestamp ordering is unavailable).
4. Training and testing periods are strictly chronological.
5. No outcome-derived (current-goal) columns enter model predictors — current
   match efficiency ratios are dropped; only lagged rolling versions are used.

A feature availability matrix is documented in `src/prediction/features.py`.
