# Phase C: Corrected Validation & Calibration Protocol — Complete

## Summary

Phase C has been successfully completed. The validation and calibration protocol has been repaired with proper temporal separation, fold-local preprocessing, and keyed prediction outputs.

## Key Changes Made

### 1. New Validation Module (`src/prediction/validation_v2.py`)
- **Fold-local preprocessing**: Imputation and scaling fitted inside each training partition only
- **Non-overlapping calendar validation blocks**: Substantial initial training period (min 20 dates), then sequential validation blocks
- **Same-date fixtures kept together**: All matches on a date go to the same split
- **Keyed predictions**: Full fixture identifiers (Div, Season, Date, HomeTeam, AwayTeam) plus fold/model/refit IDs, cutoffs, probabilities, actual outcome, lambdas, fallback status
- **League metrics from own rows**: P5 chart now uses league-specific Brier scores
- **Proper temporal separation**: Training dates always strictly precede validation dates

### 2. Fixed MultinomialModel (`src/prediction/models.py`)

#### Fold-Local Preprocessing in Inner CV
**Before**: Single imputation/scaling on full training data before inner splits (leakage)
```python
X = self._impute_fit(X)  # LEAKAGE: fits on ALL training data
# Then splits use same impute_values for all folds
```

**After**: Imputation and scaling fitted independently inside each training partition
```python
for train_idx, val_idx in valid_splits:
    X_train = X[train_idx]; X_val = X[val_idx]
    X_train_imputed, impute_values = self._impute_fit(X_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_val_imputed = self._impute_transform(X_val, impute_values)
    X_val_scaled = scaler.transform(X_val_imputed)
    # ... fit and evaluate
```

#### Corrected Inner Chronological Splits
**Before**: Overlapping validation tails, `np.linspace` split points
```python
split_points = np.linspace(1, len(unique_dates) - 1, n_folds).astype(int)
# Creates overlapping validation sets
```

**After**: Non-overlapping calendar blocks with substantial initial training
```python
train_end = min_train_dates  # e.g., 20 dates
for fold in range(n_folds):
    val_start = train_end
    val_end = min(val_start + val_block_size, ...)
    train_dates = unique_dates[:train_end]
    val_dates = unique_dates[val_start:val_end]
    train_end = val_end  # Expand training window
```

#### Regularization Search Enabled
**Before**: `factory_multinomial()` hardcoded `C=1.0`
**After**: `C=None` triggers search over `[0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]`

### 3. Updated Pipeline (`run_predictions.py`)
- Uses `validation_v2.run_chronological_validation` instead of old validation
- Saves keyed predictions to `keyed_predictions.parquet` and `.csv`
- P5 league chart uses league-specific Brier scores from keyed predictions
- Metadata `calibration_method` accurately reflects actual method used (`"platt (OOF)"`)

## Validation Results Comparison

### Before (Leaky Validation)
| Model | Brier (Aggregate) | Fold Briers | Issue |
|-------|-------------------|-------------|-------|
| Historical Baseline | 0.65023 | [0.651, 0.649, 0.650, 0.652, 0.649, **0.651**] | Stable but leaky |
| Poisson | 0.65316 | [0.659, 0.659, 0.649, 0.655, 0.643, 0.651] | Stable but leaky |
| Multinomial | **0.38539** | [0.365, 0.358, 0.353, 0.365, 0.361, **0.984**] | **Wild outlier in last fold** |

### After (Corrected Validation)
| Model | Brier (Aggregate) | Fold Briers | Improvement |
|-------|-------------------|-------------|-------------|
| Historical Baseline | 0.65006 | [0.649, 0.650, 0.652, 0.649, 0.651] | Consistent, no leakage |
| Poisson | 0.63163 | [0.640, 0.625, 0.630, 0.631, 0.630] | Consistent, beats baseline |
| Multinomial | **0.61128** | [0.612, 0.608, 0.612, 0.612, 0.618] | **Consistent across all folds** |

**Key improvement**: Multinomial model no longer has the wild 0.984 Brier in the last fold. The performance is now stable and credible across all 5 folds.

## League-Specific Brier Scores (P5 Chart)
| League | N | Brier |
|--------|---|-------|
| E0 (Premier League) | 1590 | 0.58769 |
| SP1 (La Liga) | 1608 | 0.58771 |
| D1 (Bundesliga) | 1278 | 0.59315 |
| I1 (Serie A) | 1590 | 0.59753 |
| F1 (Ligue 1) | 1361 | 0.60069 |
| E2 (League One) | 2322 | 0.62090 |
| E1 (Championship) | 2339 | 0.63269 |
| E3 (League Two) | 2334 | 0.63803 |

## Keyed Predictions (Auditability)
- **14,422 rows** saved to `output/predictions/tables/keyed_predictions.parquet`
- Each row contains: div, season, date, home_team, away_team, fold_id, model_id, refit_id, training_cutoff, feature_cutoff, p_home, p_draw, p_away, actual_outcome, home_lambda, away_lambda, fallback_status
- Enables per-league, per-season, per-team-history segment analysis

## Metadata Corrections
| Field | Before | After |
|-------|--------|-------|
| `calibration_method` | `"none (reliability diagnostics only)"` | `"platt (OOF)"` |
| `C` (multinomial) | Fixed at 1.0 | Tuned via inner CV (selected from grid) |

## Phase C Acceptance Criteria Met

✅ Inner validation uses fold-local preprocessing (imputation/scaling fitted inside each training partition)  
✅ Non-overlapping calendar validation blocks with substantial initial training period  
✅ Same-date fixtures kept together in splits  
✅ Strictly earlier training dates — no leakage  
✅ Keyed predictions with full fixture identifiers  
✅ League metrics computed from their own rows  
✅ Scored-row accounting for decomposable losses  
✅ Calibration fitted on OOF training predictions only  
✅ Metadata accurately describes the fitted forecasting procedure  
✅ Regularization strength actually searched (C=None)  
✅ No model selection using the 17 diagnostic fixtures  

## Artifacts

- **New validation module**: `src/prediction/validation_v2.py`
- **Updated model**: `src/prediction/models.py` (MultinomialModel fixes)
- **Updated pipeline**: `run_predictions.py` (uses v2 validation, saves keyed predictions)
- **Keyed predictions**: `output/predictions/tables/keyed_predictions.parquet` (14,422 rows)
- **League metrics**: `output/predictions/figures/metrics_by_league.csv`
- **Fold metrics**: `output/predictions/tables/backtest_fold_metrics.csv`
- **Model bundle**: `output/predictions/models/model_bundle.joblib` with corrected metadata

## Next Steps (Phase D)

Phase C establishes a trustworthy validation protocol. The next phase should:

1. **Establish model ladder** (Phase D)
   - Model 0: Historical baseline (mandatory reference)
   - Model 1: Small uncalibrated logistic (~10-20 interpretable inputs)
   - Model 2: Elo + logistic regression
   - Model 3: Fitted attack/defence Poisson model
   - Model 4: Calibration challenger (temperature scaling vs Platt vs none)
   - Model 5: Restricted HGB challenger (only after simpler models stable)

2. **Run ranked experiment backlog** with predeclared acceptance rules:
   - Candidate must reduce fixture-weighted development Brier
   - Candidate must improve majority of evaluation seasons
   - Mean log loss must not worsen
   - Report paired calendar-block uncertainty intervals
   - If inconclusive, retain simpler model

3. **Prospective evaluation gate**: Freeze code/config/artifacts, issue predictions before kickoff, never overwrite issued predictions.