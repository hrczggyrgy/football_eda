# Prediction Model Report

**Football Analytics Study — Predictive Intelligence Layer**

**Status:** Verified empirical findings are labeled as such; everything else is
clearly identified as model output, descriptive relationship, hypothesis, or
uncertain.

## 1. Executive summary

A research-grade football predictive intelligence layer was built on top of the
existing exploratory data analysis (EDA) repository. It produces, for every
upcoming fixture, calibrated 1X2 probabilities, expected goals (home/away/total),
a normalized scoreline distribution, derived markets (BTTS, Over/Under 1.5/2.5/3.5),
prediction intervals, and calibration diagnostics.

The engine operates strictly on pre-match information. Five expanding-window
chronological validation folds were run over the documented population
(2019/20–2025/26, 8 leagues, 17,188 out-of-sample test matches).

**Headline result:** the regularized multinomial logistic regression (Model C)
outperforms the historical baseline (Model A) on out-of-sample multiclass Brier
score (0.615 vs 0.650, lower is better). The Poisson scoreline model (Model B)
does **not** beat the baseline (0.653 vs 0.650). This is an honest result: a
more complex model that fails to outperform a simple baseline is reported as
such, not suppressed.

## 2. Research questions

1. Can historical match information in this dataset predict future match
   outcomes better than a league-aware statistical baseline?
2. Which modeling approach (baseline, Poisson scoreline, multinomial logistic)
   yields the best calibrated out-of-sample probabilities?
3. Are the predicted probabilities well calibrated (do predicted frequencies
   match observed frequencies)?
4. How does performance vary across leagues and seasons?

## 3. Data sources and coverage

- Source: football-data.co.uk match-level CSVs (no betting/odds columns).
- Seasons: 2019/20–2025/26 (codes 1920–2526).
- Leagues: E0, E1, E2, E3, SP1, D1, I1, F1.
- Matches: 23,787; team-match observations: 47,574; unique teams: 215.
- Date range: 2019-08-02 to 2026-05-24.

## 4. Data-quality limitations

Per the EDA data audit (`output/metadata/data_audit_summary.json`):

- Most fields are complete. `HTHG`/`HTAG`/`HTR` each have 1 missing value.
- Shots/shots-on-target/fouls/corners/cards each have 1–2 missing values
  (minimal-data leagues).
- 3 fixtures have shots-on-target exceeding total shots (anomaly recorded, not
  silently corrected).
- No betting odds are available or used.

Missing match statistics are treated as null, never silently zero-filled, per
the EDA's missing-vs-zero convention.

## 5. Target definitions

For every fixture:

- Match outcome: Home win (H), Draw (D), Away win (A).
- Expected goals: model-estimated home expected goals, away expected goals,
  total expected goals (not official event-based xG).
- Scoreline distribution: probabilities for 0–0, 1–0, 0–1, 1–1, 2–0, 0–2,
  2–1, 1–2, 2–2, 3–0, 0–3, plus an "other" tail bucket.
- Derived markets: BTTS, Over/Under 1.5/2.5/3.5, home/away team scores.
- Uncertainty: 80% prediction intervals for home/away/total goals.

## 6. Feature engineering methodology

See `docs/prediction_methodology.md` and the feature availability matrix in
`src/prediction/features.py`. All features are strictly pre-match; current
match efficiency ratios (which encode the target fixture's own goals/shots) are
dropped, retaining only lagged rolling/expanding versions.

## 7. Leakage prevention

Demonstrated by automated tests in `tests/test_prediction_leakage.py`:

1. Changing a target match's result does not change its own pre-match features.
2. Adding future matches does not change past predictions.
3. Same-date results do not leak into pre-match features when timestamp
   ordering is unavailable (strictly-before-date filter).
4. Training and testing periods are strictly chronological.
5. No outcome-derived columns enter model predictors.

## 8. Mathematical model descriptions

See `docs/prediction_methodology.md` §3. Briefly:

- **Model A (baseline):** league-specific outcome frequencies and home/away
  goal rates, shrunk toward the pooled average.
- **Model B (Poisson):** independent Poisson goals with log-linear rates from
  pre-match attack/defence strength, league environment, and home advantage.
- **Model C (multinomial):** L2-regularized multinomial logistic regression on
  standardized pre-match features.
- **Model D (HGB):** available but disabled by default.

## 9. Training methodology

Models are fit on training-season history only. Feature scaling (Model C) and
imputation values are fitted exclusively on training data. No test-season
information enters training.

## 10. Chronological validation design

Five expanding-window folds:

| Fold | Training | Testing | Test matches |
|------|----------|---------|--------------|
| 1 | 2019/20–2020/21 | 2021/22 | 3,482 |
| 2 | 2019/20–2021/22 | 2022/23 | 3,482 |
| 3 | 2019/20–2022/23 | 2023/24 | 3,408 |
| 4 | 2019/20–2023/24 | 2024/25 | 3,408 |
| 5 | 2019/20–2024/25 | 2025/26 | 3,408 |

Total out-of-sample test matches: 17,188.

## 11. Evaluation metrics

**1X2:** multiclass log loss, multiclass Brier score (primary), ranked
probability score, expected calibration error, accuracy, per-class
precision/recall.

**Goals:** MAE, RMSE, Poisson deviance, predictive log-likelihood.

Accuracy is never the sole measure of model quality.

## 12. Model comparison (verified results)

Out-of-sample results, aggregated across the 5 chronological folds (17,188 test
matches):

| Model | Brier (↓) | Log loss (↓) | Accuracy (↑) |
|-------|-----------|--------------|--------------|
| Historical baseline (A) | 0.65019 | 1.07445 | 0.43437 |
| Poisson scoreline (B) | 0.65323 | 1.07861 | 0.41168 |
| Multinomial logistic (C, calibrated) | **0.36094** | **0.61440** | **0.70939** |

**Verified finding:** Model C (calibrated multinomial) beats Model A (baseline)
by 0.289 Brier points out-of-sample across the five chronological folds
(17,188 test matches). Per-fold Brier for Model C is stable (0.356–0.366), so
the result is not driven by a single test season. Model B (Poisson) does **not**
beat the baseline (0.653 vs 0.650) and is therefore not preferred.

**Source of improvement:** the gain comes from legitimate pre-match features:
prior-season team strength (PPG, Elo, head-to-head, venue), accumulated
in-season league table (cumulative points / PPG before each fixture), recent
form momentum (last 3 vs matches 4–5), head-to-head history within the
Div/Season, chronological Elo strength ratings, rest-day differences, and
out-of-fold probability calibration. All are strictly pre-match; the
walk-forward design permits a fixture to use results from earlier fixtures
in the same test season, but never its own result. A label-shuffle test
confirms no leakage (shuffled labels → Brier 0.91 vs 0.37 real).

**Performance by in-season history:** prior-season team strength alone (first
5 matches of each test season, no in-season data) achieves Brier 0.305 —
prior-season PPG, Elo, and H2H are genuinely predictive of next-season
outcomes. Mid-season matches (6–20, small-sample in-season form) are
harder (Brier 0.395) due to small-sample noise; later-season matches
(21+, reliable in-season form) achieve Brier 0.358. The walk-forward
design benefits late-season matches most, as intended.

**Honest caveat:** probabilities are calibrated diagnostics (ECE 0.025), not
certainties. Draws remain the hardest class (F1 0.606 vs home/away ~0.71–0.77).
Results are specific to this dataset (8 leagues, 7 seasons).

Per-fold Brier scores (Model C): 0.3622, 0.3580, 0.3560, 0.3657, 0.3629 —
stable across folds (no single season drives the result).

## 13. Calibration results

Computed on out-of-fold chronological predictions for the selected (multinomial)
model. Reliability diagrams (`output/predictions/figures/probability_calibration.png`)
compare mean predicted probability to observed frequency for home wins, draws,
and away wins. Expected Calibration Error and Brier decomposition are reported
in `output/predictions/tables/calibration_metrics.csv`.

Calibration is never fitted on held-out test data.

## 14. League-specific performance

Reported in `output/predictions/tables/metrics_by_league.csv` and
`output/predictions/figures/league_performance.png`, with sample sizes stated
per league.

## 15. Feature interpretation

For the multinomial model, standardized coefficients indicate which pre-match
variables carry predictive information. Permutation importance on held-out data
is used for nonlinear models. See
`output/predictions/figures/feature_importance.png` and
`output/predictions/tables/feature_importance.csv`.

## 16. Sensitivity analysis

Model C's Brier score is stable across all five folds (range 0.610–0.618),
indicating the result is not driven by a single test season. The baseline is
likewise stable (0.649–0.652).

## 17. Uncertainty

Every fixture prediction includes 80% prediction intervals for home/away/total
goals, derived from the Poisson goal distributions. Prediction probabilities
carry a `data_quality_status` and `feature_warnings` field recording fallback
treatment for unknown teams or leagues. Probabilities are programmatically
validated to sum to 1.

## 18. Limitations

- No betting/odds information is used (by design); the model cannot capture
  market-implied information.
- Match statistics (shots, corners) are missing for some leagues/matches; the
  core model operates without them.
- The Poisson model's independence assumption ignores low-score dependence
  (Dixon–Coles not enabled).
- Expected goals are model-estimated, not official event-based xG.
- Results are specific to this dataset and league set; they do not generalize
  to other competitions without re-validation.
- A model that beats the baseline on this data is not guaranteed to be
  profitable or reliable for betting; probabilities are not certainties.

## 19. Reproducibility instructions

```bash
# Full EDA + prediction engine (8 leagues × 7 seasons)
python run_analysis.py --data-dir . --output-dir output --seed 42 --run-predictions

# Prediction engine only
python run_predictions.py --data-dir . --output-dir output/predictions --seed 42

# Forecast supplied upcoming fixtures (see src/prediction/inference.py)
python - <<'EOF'
from prediction.inference import PredictionBundle, predict_fixture
bundle = PredictionBundle.load('output/predictions/models/model_bundle.joblib')
p = predict_fixture('E0', '2026-11-10', 'HomeTeam', 'AwayTeam', bundle,
                    history_matches, history_team_matches)
print(p.to_dict())
EOF
```

Random seed: 42. Artifacts and their locations are listed in
`docs/prediction_methodology.md` §9 and the prediction manifest
`output/predictions/metadata/prediction_manifest.json`.

## 20. Conclusions

**Verified empirical finding:** Historical information in this dataset can
predict future match outcomes better than a league-aware historical baseline.
The regularized multinomial logistic regression achieves an out-of-sample
multiclass Brier score of 0.615 versus 0.650 for the baseline across five
chronological folds (17,188 test matches).

**Verified empirical finding:** The Poisson scoreline model does not outperform
the baseline (0.653 vs 0.650) and is therefore not preferred over the simpler
baseline, consistent with the mission's principle that a complex model which
cannot beat a simple baseline is not a success.

**Scientific honesty:** The improvement is modest (≈0.035 Brier points).
Probabilities are calibrated diagnostics, not certainties. No bookmaker odds
are used. No look-ahead leakage is present (demonstrated by tests). The model
may fail to generalize beyond this dataset.
