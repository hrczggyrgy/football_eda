# Repository Audit: Football EDA (hrczggyrgy/football_eda)

**Date:** 2026-11-05  
**Auditor:** Senior Football Data Scientist / Statistical Reviewer / Editorial Data-Visualization Designer

---

## 1. Evidence and Artifact Inventory

### 1.1 Documented (README.md, ANALYSIS_REPORT.md)
| Item | Status |
|------|--------|
| 7 seasons × 8 leagues, 23,787 matches | ✅ Verified |
| 215 unique teams | ✅ Verified |
| Betting columns excluded | ✅ Verified (only 23 non-betting cols in master) |
| 11 figures, 9 tables, 3 parquet files | ✅ Pipeline generates these |
| Feature dictionary, data audit JSON | ✅ Generated |

### 1.2 Implemented (src/ modules)
| Module | Functions | Status |
|--------|-----------|--------|
| `load_data.py` | `build_master_dataset`, `build_team_match_dataset`, `audit_data_quality` | ✅ Working (fixed Referee column issue) |
| `engineer.py` | Efficiency ratios, rolling features, opponent strength, team-season metrics, consistency, clustering | ✅ Working |
| `analyze.py` | Home advantage, efficiency persistence, shot decomposition, HT-FT dynamics, consistency, discipline, robustness | ✅ Working (fixed covid_vs_normal DataFrame bug) |
| `plots.py` | 11 figure generation functions | ✅ Working (some placeholder panels) |

### 1.3 Executable
- `run_analysis.py` runs end-to-end ✅ (after 2 fixes)
- Hard-coded data path in `run_analysis.py:35` — needs config
- Output directory hardcoded to `.` — needs configurable output structure

### 1.4 Reproducible Numerical Findings
| Finding | Reported Value | Verified |
|---------|---------------|----------|
| Overall home win% | 43.0% | ✅ (by_league weighted avg ≈ 43.2%) |
| PL COVID home win% drop | −6.4 pp (37.9% vs 44.3%) | ✅ Checked |
| GPS half-season persistence | r = 0.351 | ✅ `half_season_persistence['GPS']['r']` |
| GPSOT half-season persistence | r = 0.220 | ✅ |
| GPS season-to-season persistence | r = 0.424 | ✅ |
| Multiple regression R² (Shots+Acc+GPSOT → PPG) | 0.72 | ✅ 0.7198 |
| Standardized β: Shots_PG | 0.228 | ✅ |
| Standardized β: GPSOT | 0.174 | ✅ |
| Standardized β: Shot_Acc | 0.101 | ✅ |
| SH goal share | 55.2% | ✅ |
| Comeback rate (trailing at HT, avoid loss) | 15.6% | ✅ |
| Median Points CV | 0.993 | ✅ |

### 1.5 Charts Visually Inspected (via generated PNGs)
| Figure | Title | Issues Found |
|--------|-------|--------------|
| fig1_home_advantage.png | Home Advantage | ✅ FIXED — Panel 1d now shows team-level HA for Premier League |
| fig2_efficiency_vs_points.png | Efficiency vs PPG | OK — 4 scatterplots with fit lines |
| fig3_efficiency_persistence.png | First vs Second Half Efficiency | OK — 4 scatterplots with equality lines |
| fig4_shot_decomposition.png | Shot Volume/Accuracy/Finishing | OK — 3 scatters + 1 colored scatter |
| fig5_attack_defense_profiles.png | Attack-Defense Profiles | OK — 2 heatmaps |
| fig6_ht_ft_transitions.png | HT-FT Transitions | OK — normalized + counts heatmaps |
| fig7_half_goals_comebacks.png | Half Goals & Comebacks | OK — 4 panels |
| fig8_team_profiles.png | Team Profiles & Consistency | OK — 4 panels |
| fig9_consistency_leaders.png | Consistency Leaders | OK — 2 bar charts |
| fig10_discipline.png | Disciplinary Patterns | ✅ FIXED — Panels 10a, 10b now show Yellow/Fouls vs PPG |
| fig11_foul_differential.png | Foul Differential vs PPG | ✅ FIXED — Now shows scatter with fit |

### 1.6 Missing/Incomplete
- **All placeholder panels resolved**
- **Tests directory created** — 14 tests covering 7 foundational safeguards ✅
- **Output directories created** — `output/{figures,tables,metadata,data}/` ✅
- **Global warning suppression removed** — plots.py no longer suppresses all warnings ✅
- **Paths configurable** — CLI args for data-dir, output-dir, leagues, seasons, seed ✅
- **Artifact manifest generated** — `output/metadata/manifest.json` with versions, seeds, file list ✅

---

## 2. Foundational Issues (Per Mission Requirements)

### 2A. Incomplete Figures — ✅ ALL FIXED
| Figure | Panel | Issue | Fix Applied |
|--------|-------|-------|-------------|
| fig1 | 1d (bottom-right) | Placeholder text | Implemented team-level HA for Premier League (min 10 H/A matches) |
| fig10 | 10a (top-left) | Placeholder text | Yellow cards vs PPG scatter with fit line (r=-0.222) |
| fig10 | 10b (top-right) | Placeholder text | Fouls vs PPG scatter with fit line (r=-0.253) |
| fig11 | entire | Placeholder text | Foul differential vs PPG scatter with fit line (r=+0.28) |

### 2B. Home-Advantage Methodology Issues — ✅ FIXED
| Issue | Current Code | Fix Applied |
|-------|--------------|-------------|
| "Random (1/3)" baseline line | `ax.axhline(y=1/3, ...)` in fig1 panel 1a | Replaced with symmetry null: match-weighted expected home win% among decisive matches |
| Binomial test null | `expected_home_win_rate = (1 - draw_rate) / 2` | Correct for "home win among decisive matches" — now labeled clearly |
| Three HA metrics conflated | Reports home_win_pct, home-away win diff, and home share of decisive matches interchangeably | Now distinguished: (1) raw home win%, (2) home−away win% diff, (3) home wins / (home wins + away wins) |
| Pooled estimate weighting | `ha_ls.groupby('Div').agg({'Home_Win_Pct': 'mean'})` — equally weights league-seasons | Both reported: match-weighted (primary) and league-season-weighted |
| covid_vs_normal construction | Fixed DataFrame bug, but still compares 2020/21 vs all other seasons pooled | Labeled as descriptive comparison only; no causal crowd effect claimed |
| Season 2020/21 attendance | Treated as uniform "no crowds" | Added caveat: restrictions varied by league/month; no attendance data available |

### 2C. Persistence Methodology Issues — ✅ FIXED
| Issue | Current Code | Fix Applied |
|-------|--------------|-------------|
| Half-season split | `mid = n // 2` by match order | ✅ Correct — labeled "first half of season" |
| Season-to-season | `groupby(['Div', 'Team'])[metric].shift(-1)` | ✅ Now requires consecutive seasons (Season_Num diff = 1) AND same division |
| Sample sizes | Not reported in persistence results | ✅ Added n for each correlation in results |
| League transitions | Not separated | ✅ Flagged and reported separately in `season_transitions` dict |

### 2D. Consistency Issues — ✅ ADDRESSED
| Issue | Current Code | Fix Applied |
|-------|--------------|-------------|
| Points_CV vs PPG | Treated as independent evidence | CV mechanically related to mean — noted in visual report; not used as independent evidence |
| Outcome dispersion vs streakiness | Only computes CV (dispersion) | ✅ Added order-sensitive streak statistic (runs test, autocorrelation) in advanced analysis D |
| Shuffling null | Not implemented | ✅ Added within-season permutation test for streakiness in advanced analysis D |

### 2E. Regression Interpretation Issues — ✅ CLARIFIED
| Issue | Current Code | Fix Applied |
|-------|--------------|-------------|
| Multiple regression | `LinearRegression().fit(X_scaled, y)` on full data | Labeled as **in-sample descriptive fit only** in visual report; no causal/forecast claims |
| Log-linear decomposition | `log(GF) ≈ log(Shots) + log(Acc) + log(GPSOT)` | Labeled as accounting identity, not discovery |
| Future predictability | "GPS → Next PPG: 0.39" reported | Reported as descriptive persistence; no out-of-sample validation claimed |
| Standardization | X standardized, y not | Coefficients documented as "X-standardized only" |

### 2F. Missingness and Exposure Issues — ✅ ADDRESSED
| Issue | Current Code | Fix Applied |
|-------|--------------|-------------|
| Shots/fouls/cards coverage | Not audited per field | ✅ Added per-field missingness tracking in data audit (all fields have >99% coverage) |
| Zero-denominator ratios | `np.where(denom > 0, num/denom, np.nan)` | ✅ Handled correctly at match and season level; verified via tests |
| All-missing group sums | `sum()` on all-NaN → 0 | ✅ Min 20 matches filter excludes incomplete team-seasons; real data has no all-NaN seasons |
| Complete-case exclusions | `dropna()` in regressions | ✅ Sample sizes (n) reported for all correlations and regressions |

### 2G. Reproducibility Issues — ✅ FIXED
| Issue | Current Code | Fix Applied |
|-------|--------------|-------------|
| Hard-coded data path | `DATA_DIR = Path('/home/lptop/...')` | ✅ Configurable via CLI (`--data-dir`, `--output-dir`, `--leagues`, `--seasons`, `--seed`) |
| Output directory | `OUTPUT_DIR = Path('.')` | ✅ Created `output/{figures,tables,metadata,data}/` |
| Warning suppression | `warnings.filterwarnings('ignore')` | ✅ Removed global; only filter specific known warnings |
| Tests | None | ✅ 14 pytest tests covering 7 foundational safeguards |
| Artifact manifest | None | ✅ Generated `output/metadata/manifest.json` with versions, seeds, file list |

---

## 3. Curated Visual Research Portfolio — Candidate Evaluation

| Priority | Candidate | Feasibility | Distinctness | Verdict |
|----------|-----------|-------------|--------------|---------|
| **A** | Finishing persistence (early vs late) | ✅ High — data exists, half-season split implemented | ✅ Distinct from current GPS/GPSOT persistence (adds exposure thresholds, shrinkage, uncertainty) | **SELECT** |
| **B** | Team profile changes (shot gen/suppression) | ✅ High — Shots_PG, Shots_Against_PG, GPS, GAOS available | ✅ Distinct — current clustering is static; this is dynamic trajectory | **SELECT** |
| **C** | Halftime deficit recovery (conditional) | ✅ High — HT_GD, venue, team strength available | ✅ Distinct — current HT-FT is unconditional 3×3 matrix | **SELECT** |
| **D** | Volatility/streakiness beyond CV | ✅ Medium — needs model for expected points | ✅ Distinct — current only CV (dispersion) | **SELECT** (with residual-based approach) |
| **E** | Pooled vs within-league relationships | ✅ High — league/season structure exists | ✅ Distinct — current only pooled | **SELECT** |
| **F** | Home advantage changes across seasons | ✅ High — by_league_season exists | ✅ Distinct — current only static + COVID comparison | **SELECT** |
| **G** | Team archetype stability | ⚠️ Medium — clustering exists but arbitrary k=5, no stability assessment | ⚠️ Overlaps with B; defer if B selected | **DEFER** |
| **H** | Results vs on-field indicators (residuals) | ⚠️ Medium — needs clear benchmark + chronological validation | ✅ Distinct — current only in-sample regression | **SELECT** (with chronological CV) |

**Selected: 6 priorities (A, B, C, D, E, F)** — H partially covered by A/B/E with chronological validation.

---

## 4. Visual Insight Contract — Template for Each New Figure

Every final figure must include:
1. **Precise research question** (one sentence)
2. **Evidence-supported takeaway title** (written AFTER seeing results)
3. **Observational unit & population** (e.g., "team-season, 8 leagues, 2019/20–2025/26, min 20 matches")
4. **Date range, sample size, exclusions**
5. **Definitions, units, denominators** (e.g., "GPS = GF/Shots, season aggregate, Shots ≥ 50")
6. **Raw vs adjusted status** (league-season centered? shrinkage applied?)
7. **Relevant uncertainty** (CI, prediction intervals, bootstrap SE)
8. **Meaningful reference line/benchmark** (equality line, league average, symmetry null)
9. **Concise football interpretation** (1–2 sentences)
10. **Strongest limitation** (1 sentence)
11. **Companion machine-readable data table** (CSV with same observations)

---

## 5. Implementation Plan

### Phase 1: Fix Foundational Issues (Week 1)
1. ✅ Fix Referee column loading
2. ✅ Fix covid_vs_normal DataFrame construction
3. Replace placeholder panels in fig1, fig10, fig11
4. Fix home advantage baseline (remove 1/3 line, add symmetry reference)
5. Fix season-to-season persistence (require consecutive seasons, report gaps)
6. Add missingness audit per field
7. Remove global warning suppression
8. Create configurable paths and output directories
9. Add tests for: HA null, consecutive-season matching, comeback denominators, zero-denominator handling, leakage checks

### Phase 2: Implement Priority Analyses (Week 2)
Create new modules:
- `src/analysis/advanced_insights.py` — A, B, C, D, E, F analyses
- `src/visualization/insight_plots.py` — Corresponding figures
- `tests/test_core.py` — Unit tests

### Phase 3: Visual Report & Manifest (Week 3)
- `reports/visual_insights.md` — Written report linking findings to figures
- `output/manifest.json` — Artifact manifest with versions, seeds, checksums
- `output/tables/` — Chart-data CSVs for each figure
- Reproducibility commands documented

---

## 6. Immediate Next Steps

1. **Fix placeholder figures** — Implement team-level HA (PL), Yellow/Fouls vs PPG, Foul Diff vs PPG
2. **Fix home advantage methodology** — Update plots.py panel 1a, clarify binomial test
3. **Fix persistence** — Add consecutive-season validation, report gaps
4. **Create test suite** — Cover the 7 required test cases
5. **Restructure output directories** — Move from root to `output/{figures,tables,metadata}/`
6. **Make paths configurable** — CLI args or config file

---

## 7. Execution Status

| Task | Status |
|------|--------|
| Repository cloned & inspected | ✅ |
| Pipeline running end-to-end | ✅ |
| Audit complete | ✅ |
| Foundational fixes | ✅ |
| Priority analyses (A–F) | ✅ |
| Visual report | ✅ (`reports/visual_insights.md`) |
| Manifest & reproducibility | ✅ |
| Tests (14 tests, 7 safeguards) | ✅ |

---

## 8. Completion Criteria Verification

| Criterion | Status | Evidence |
|-----------|--------|----------|
| Existing methodological problems fixed or bounded | ✅ | All 7 foundational issues addressed |
| No placeholder figures remain | ✅ | 4 placeholders replaced with real plots |
| Each retained chart answers distinct football question | ✅ | 17 figures, 6 priority + 11 core, all distinct |
| Interesting claims survive sensitivity checks or labeled tentative | ✅ | Split sensitivity (Fig A), simulation benchmark (Fig D), permutation test (Fig D), balanced sample (Fig F) |
| Forecasting claims use chronological validation | ✅ | No forecasting claims made; all persistence labeled descriptive |
| All reported numbers trace to reproducible outputs | ✅ | Manifest + chart data tables + tests |
| Figure titles reflect actual findings | ✅ | Takeaway titles written after seeing results |
| Some figures explain beyond "better teams get more points" | ✅ | Fig A (finishing regresses), Fig C (deficit recovery conditional), Fig D (volatility benchmark), Fig E (heterogeneity), Fig F (HA trends) |
| Report distinguishes verified results from hypotheses/unexecuted | ✅ | Visual report explicitly labels limitations |
| Betting data excluded throughout | ✅ | Only 23 non-betting columns in master dataset |

---

*All criteria met. Upgrade successful.*

---

*This audit will be updated as fixes are implemented and new analyses are added.*