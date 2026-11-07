# Phase B: Feature Parity & Replay/Inference Parity Tests — Complete

## Summary

Phase B has been successfully completed. Training–inference feature parity has been established for the football predictive engine. All parity tests pass with zero critical mismatches.

## Key Changes Made

### 1. Feature Schema Definition (`src/prediction/feature_schema.py`)
- Created a versioned feature contract (`FEATURE_SCHEMA_VERSION = "1.0.0"`)
- Defines 84 features with specifications: name, dtype, meaning, availability, grouping, missing semantics, fallback, units, requirement level
- Single source of truth for feature compatibility between historical replay and prospective inference
- Loadable from persisted model bundle metadata

### 2. Parity Test Harness (`src/prediction/parity_test.py`)
- Compares two independent construction routes for each fixture:
  - **Historical route**: Pre-computed features from full historical pipeline
  - **Inference route**: Rebuilt from history strictly before fixture date + synthetic target row
- Tests 8 critical edge cases:
  1. Ordinary mid-season fixture
  2. New-season opener (only prior-season history)
  3. Promoted team
  4. Postponed fixture / unequal games played
  5. Missing match statistics (shots/corners)
  6. Historical H2H draw
  7. Same-date fixture reordering
  8. Future-result mutation (must not affect earlier predictors)
- Reports per-feature mismatches with severity levels (critical/warning/info)
- Verifies prediction parity using a small logistic regression model
- Tolerance: `rtol=1e-8, atol=1e-10` for floating-point comparisons

### 3. Fixes to Feature Engineering (`src/prediction/features.py`)

#### Venue Splits (`_add_venue_splits`)
**Problem**: Used `cum_sum - current_value` which fails when current row has null Points/GF (inference synthetic row).
**Fix**: Changed to `cum_sum().shift(1)` — strictly prior cumulative values, robust to null current row.
```python
# Before (broken for inference):
prior_points = (points.cum_sum().over(grp) - points).alias("prior_points")

# After (robust):
prior_points = points.cum_sum().over(grp).shift(1).over(grp).alias("prior_points")
```

#### Season Progress (`_add_season_progress`)
**Problem**: Used `cum_count().over(["Div", "Season"]) - 1` which includes same-date matches. Historical route computed on full dataset (includes same-date matches); inference route computed on history-only (no same-date matches).
**Fix**: Count strictly prior matches by date using a join on division-season-date match counts.
```python
# Before (same-date sensitive):
matches_so_far = pl.col("Date").cum_count().over(["Div", "Season"]) - 1

# After (strictly prior dates):
match_counts = out.group_by(["Div", "Season", "Date"]).agg(pl.len().alias("matches_on_date"))
# ... cum_sum by date, then matches_so_far = matches_up_to_date - matches_on_date
```

### 4. Fix to Elo Ratings (`src/prediction/ratings.py`)

#### `build_elo_features`
**Problem**: Processed matches sequentially within same date (alphabetical by HomeTeam). Pre-match ratings for a fixture included updates from same-date matches ordered before it. Inference route had no same-date matches.
**Fix**: Process matches **by date batch** — all matches on date D use Elo state from end of date D-1; updates applied after all pre-match ratings for date D are recorded.
```python
# Before: sequential loop over matches
for row in rows:
    h_r, a_r = elo.rate(div, ht, at)  # includes same-date prior updates
    # ... record ...
    elo.update(div, ht, at, hg, ag)   # updates affect later same-date matches

# After: date-batched processing
for date_val in div_dates_list:
    # 1. Initialize new teams
    # 2. Record pre-match ratings for ALL matches on this date
    for row in date_matches:
        h_r, a_r = elo.rate(div, ht, at)  # state from PREVIOUS date
    # 3. Update with ALL results from this date
    for row in date_matches:
        elo.update(div, ht, at, hg, ag)
```

## Test Results

```
============================================================
PARITY TEST SUMMARY
============================================================
Total fixtures tested: 18
Passed: 18
Failed: 0
Total mismatches: 0
Critical mismatches: 0
Overall: PASS
```

### Test Coverage
- **12 feature parity tests** across 3 divisions (I1, E2, E3) and multiple seasons
- **3 future mutation tests** (adding fake future results must not change past predictors)
- **3 prediction parity tests** (same model + same fitted preprocessing → equivalent probabilities)

### Edge Cases Verified
| Case | Description | Status |
|------|-------------|--------|
| Mid-season | Ordinary fixture with full history | ✅ PASS |
| Season opener | First match, only prior-season history | ✅ PASS |
| Promoted team | Team new to division | ✅ PASS |
| Same-date | Multiple fixtures on same date | ✅ PASS |
| Missing stats | Early seasons with no shots/corners | ✅ PASS |
| H2H draw | Teams with drawn head-to-head history | ✅ PASS |
| Future mutation | Adding future results doesn't affect past | ✅ PASS |
| Prediction parity | Same model → same probabilities | ✅ PASS |

## Acceptance Criteria Met

✅ Required predictor names and order match exactly  
✅ Missingness matches exactly  
✅ Categorical/integer values match exactly  
✅ Floating-point values match within declared tolerance (`rtol=1e-8, atol=1e-10`)  
✅ Same-date reordering and future-result mutation leave earlier predictors unchanged  
✅ Using one fixed, earlier-trained model and its fitted preprocessing, both routes produce equivalent transformed vectors and probabilities  
✅ No legitimate missing values are silently converted into fabricated observations  

## Artifacts

- **Parity test harness**: `src/prediction/parity_test.py`
- **Feature schema**: `src/prediction/feature_schema.py`
- **Test report**: `output/parity_test/parity_report.json` (18 fixtures, 0 mismatches)
- **Preserved original artifacts**: `preserved_artifacts_20261106_211504/` (SHA256 verified)

## Next Steps (Phase C)

Phase B establishes input correctness. The next phase should address:

1. **Repair inner validation splits** (`src/prediction/models.py:529-545`)
   - Fold-local preprocessing (imputation/scaling inside each training partition)
   - Non-overlapping calendar validation blocks
   - Strictly earlier training dates

2. **Separate tuning, calibration, and scoring** (`src/prediction/models.py:580-676`)
   - Earlier training period → choose base-model settings via inner forward folds
   - Later calibration period → fit calibrator on predictions from earlier-fitted base model
   - Subsequent evaluation period → score frozen base-model/calibrator pair

3. **Keyed prediction outputs** (`src/prediction/validation.py:34-47`)
   - Extend `FoldResult` with fixture identifiers (Div, Season, Date, HomeTeam, AwayTeam)
   - League metrics from their own rows
   - Scored-row accounting for decomposable losses

4. **Fix metadata inconsistencies** (`run_predictions.py:164, 230, 278, 355`)
   - `factory_multinomial()` should use `C=None` to enable regularization search
   - Best model selection should use final evaluation period only
   - P5 league chart should use league-specific Brier scores
   - Metadata `"calibration_method"` should match actual calibration usage

## Notes

The fixes to `features.py` and `ratings.py` change the computed feature values for ALL historical fixtures. The existing model bundle (trained on pre-fix features) is now incompatible with the corrected feature pipeline. A full model rebuild using the corrected features is required before Phase A diagnostics or prospective evaluation.

The parity test harness can be reused to verify that any future feature engineering changes maintain training-inference parity.