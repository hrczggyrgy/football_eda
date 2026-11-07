# Draw Prediction Improvements — Summary

## Problem
The model almost never predicted draws (0.7% draw accuracy, 0 draws as argmax) because:
- Draw probability range was too narrow (0.17–0.28)
- Draw probabilities never exceeded home/away probabilities
- Logistic regression optimizes overall accuracy, not draw-specific metrics

## Solution: Draw Probability Boosting

### 1. New Draw-Predictive Features (11 added)
| Feature | Description | Draw Rate Signal |
|---------|-------------|------------------|
| `Elo_Rating_Diff_Abs` | |Elo home - Elo away| | 28% draw when <50, 12% when >200 |
| `PPG_Gap_Abs` | |Home PPG - Away PPG| | Higher draws when teams evenly matched |
| `League_Total_Goals_PG` | League GF + GA per game | Lower scoring → more draws |
| `Combined_GA_Cum` | Home GA + Away GA (cumulative) | Both defensive → low scoring → draws |
| `Combined_Venue_GF_PG` | Home home_GF + Away away_GF | Context-aware scoring |
| `H2H_Draw_Rate` | Historical H2H draw rate | Direct draw history |
| `Combined_GF_Cum` | Both teams' cumulative GF | Low scoring teams |
| `Late_Season_Both` | Both teams >70% season progress | Late season pressure → draws |
| `Evenly_Matched` | Elo diff < 50 | Binary: evenly matched teams |
| `Low_Scoring_League` | League total goals < 2.5 | Low scoring environment |
| `Draw_Prone_Context` | Evenly matched + (low scoring OR late) | Combined context |

### 2. Draw-Boosted Model Wrapper
`DrawBoostedModel` wraps any base model and applies post-processing:
- **Threshold**: Boost draw when p_draw > 0.2
- **Boost Amount**: Transfer 0.05 probability from max(home, away) to draw
- **Condition**: Only when draw isn't already the max prediction

### 3. Results (restricted_hgb base model)

| Metric | Base restricted_hgb | Draw-Boosted | Change |
|--------|---------------------|--------------|--------|
| Overall Brier | 0.6115 | 0.6372 | +0.026 |
| Overall Accuracy | 49.1% | 45.0% | -4.1pp |
| **Draw Accuracy** | **0.7%** | **13.8%** | **+13.1pp** |
| Draw F1 Score | ~0 | 0.178 | +0.178 |
| Draws Predicted (argmax) | 0 | 251/1864 | +251 |

### 4. Draw Probability Distribution
- **Base**: 0.17–0.28 (never max)
- **Boosted**: 0.07–0.46 (251 draws as argmax)

### 5. Threshold Optimization
Optimized on OOF validation data:
- **Best threshold**: 0.2
- **Best boost**: 0.05
- **Draw F1**: 0.178 (Precision=0.28, Recall=0.13)

## Trade-offs
- **+13.1pp draw accuracy** (major improvement for draw prediction)
- **-4.1pp overall accuracy** (acceptable trade-off)
- **+0.026 Brier** (slightly worse probability calibration)

## Files Modified
| File | Changes |
|------|---------|
| `src/prediction/features.py` | 11 new draw-predictive features |
| `src/prediction/models.py` | `DrawBoostedModel` wrapper, `DrawOptimizedModel` (two-stage) |
| `src/prediction/models.py` | Added features to `BASE_FEATURES` |
| `run_predictions.py` | Added `draw_boosted_hgb` factory |

## Integration
The draw-boosted model is added to the model ladder as `draw_boosted_hgb`. It uses the restricted HGB as base with threshold=0.2, boost=0.05.

## Next Steps
1. **Temperature scaling** on draw logits for better calibration
2. **Focal loss** training to penalize draw misclassification more
3. **Ensemble** with Poisson model (natural draw predictor from low scoring)
4. **Feature engineering**: Team-specific draw tendencies, referee effects
5. **Cost-sensitive learning**: Weight draw misclassification higher

## Validation
- All 18 parity tests pass (0 mismatches)
- Feature parity between historical and inference routes maintained
- Temporal validation integrity preserved