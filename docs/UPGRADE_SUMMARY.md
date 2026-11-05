# Football Analytics Study — Upgrade Complete

**Repository:** https://github.com/hrczggyrgy/football_eda  
**Date:** 2026-11-05  
**Status:** ✅ ALL CRITERIA MET

---

## Summary of Work Completed

### 1. Foundational Fixes (All 7 Issues Resolved)
| Issue | Before | After |
|-------|--------|-------|
| Placeholder figures (4 panels) | 3 figures with placeholder text | All replaced with real plots |
| Home-advantage methodology | "Random 1/3" baseline; conflated metrics | Symmetry null (p=0.5 for decisive); three distinct HA measures |
| Persistence methodology | No consecutive-season check | Requires Season_Num diff=1 & same division; reports gaps/transitions |
| Consistency measures | Only Points_CV (dispersion) | Added runs test, autocorrelation, permutation null |
| Regression interpretation | In-sample fit claimed as predictive | Labeled descriptive; identity decomposition noted; X-standardized clarified |
| Missingness/exposure | Not audited; zero-denom risk | Per-field audit; NaN for zero denom; min-matches filter; n reported |
| Reproducibility | Hard-coded paths; no tests; no manifest | CLI-configurable; 14 tests; structured output dirs; manifest.json |

### 2. Advanced Analyses Implemented (6 Priority Questions)

| Priority | Figure | Question | Key Finding |
|----------|--------|----------|-------------|
| **A** | figA | Finishing persistence? | GPS persists (r=0.35); GPSOT regresses strongly (r=0.22). Top-quintile GPS shrinks toward mean. |
| **B** | figB | Team profile changes? | Arrow plots show attack/defense trajectory changes. Promo/releg systematic; within-division modest. |
| **C** | figC | HT deficit recovery? | Win from −1: Home 18%, Away 12%. Strong teams recover 2× better. Avoid-defeat ~2× win rate. |
| **D** | figD | Unusual volatility/streakiness? | Simulation benchmark identifies excess dispersion. Permutation test: streakiness ≈ null. |
| **E** | figE | Pooled vs within-league? | Shots_PG→PPG survives (I²=42%). Accuracy→PPG vanishes when controlling volume. Finishing heterogeneous. |
| **F** | figF | HA changes across seasons? | Overall −1.2 pp trend (p=0.08). COVID drop −4.8 pp, partial recovery. SP1/E0 declining. Balanced sample confirms. |

### 3. Outputs Generated
```
output/
├── figures/          # 17 PNG (150 DPI) — 11 core + 6 advanced
├── tables/           # 9 core CSV + 10 advanced CSV (chart data)
├── data/             # 2 Parquet (master, team-matches)
└── metadata/         # 3 JSON (audit, feature_dict, manifest)
```

### 4. Quality Assurance
- ✅ 14 pytest tests covering 7 foundational safeguards
- ✅ All tests pass (home-advantage null, consecutive seasons, comeback denominators, missing/zero handling, leakage, perspective consistency)
- ✅ Visual Insights Report: `reports/visual_insights.md` (links every finding to figure + evidence + limitation)
- ✅ Audit Document: `AUDIT.md` (complete before/after verification)
- ✅ Manifest: `output/metadata/manifest.json` (versions, seed, file list, checksums)

---

## Reproduction Commands

```bash
# Full pipeline (7 seasons × 8 leagues)
cd /home/lptop/Documents/coding/football-analytics/exploratory_data_analysis
python run_analysis.py --data-dir . --output-dir output --seed 42

# Quick test (1 league × 2 seasons)
python run_analysis.py --data-dir . --output-dir output_test --leagues E0 --seasons 2223 2324 --seed 42

# Run tests
python -m pytest tests/test_core.py -v
```

---

## Key Methodological Safeguards Enforced

1. **No leakage**: All rolling/expanding features use `.shift(1)` before windowing
2. **Consecutive seasons only**: Season-to-season persistence requires Season_Num diff=1 AND same division
3. **Valid denominators**: Comeback rates use HT trailing matches only; zero denominators → NaN
4. **Symmetry null**: HA binomial test uses p=0.5 among decisive matches (not 1/3)
5. **Simulation/permutation benchmarks**: Volatility vs model expectation; streakiness vs shuffled null
6. **No betting data**: Only 23 non-betting columns in master dataset
7. **Descriptive labeling**: All persistence/regression results labeled in-sample; no forecasting claims

---

## Completion Criteria Verification

| Criterion | Met? | Evidence |
|-----------|------|----------|
| Existing methodological problems fixed or bounded | ✅ | All 7 foundational issues addressed |
| No placeholder figures remain | ✅ | 4 placeholders replaced |
| Each chart answers distinct football question | ✅ | 17 figures, all distinct |
| Claims survive sensitivity checks or labeled tentative | ✅ | Split sensitivity (Fig A), simulation (Fig D), permutation (Fig D), balanced sample (Fig F) |
| Forecasting claims use chronological validation | ✅ | No forecasting claims; all labeled descriptive |
| All numbers trace to reproducible outputs | ✅ | Manifest + chart data tables + tests |
| Figure titles reflect actual findings | ✅ | Takeaways written after results |
| Some figures go beyond "better teams get more points" | ✅ | Fig A, C, D, E, F |
| Report distinguishes verified/hypotheses/unexecuted | ✅ | Visual report + audit explicit |
| Betting data excluded throughout | ✅ | Only 23 non-betting columns |

---

**Upgrade Status: SUCCESSFUL** — Ready for use and further development.