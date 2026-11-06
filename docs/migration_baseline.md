# Migration Baseline — Pandas-Centered Architecture

**Date:** 2026-11-06  
**Repository:** football-analytics (exploratory_data_analysis)  
**Branch:** `master` (commit `d3c8d54 perf: significantly accelerate football analytics pipeline`)  
**Python:** 3.14.4  
**Environment packages:** pandas 3.0.6, numpy 2.5.3, matplotlib 3.11.2, seaborn 0.13.2, scipy 1.16.3, scikit-learn 1.9.1

> This document establishes the pre-migration baseline. All measurements below were taken against the **existing pandas-centered implementation** before any Polars/DuckDB migration. They are the reference against which Phase 8 (performance optimization) and Phase 11 (before/after benchmarking) are measured.

## 1. Repository inventory (pre-migration)

| Path | Role | Primary engine |
|------|------|----------------|
| `run_analysis.py` | Pipeline orchestration (7 stages) | pandas |
| `src/data/load_data.py` | ZIP CSV ingestion, master + team-match construction, data-quality audit | pandas |
| `src/features/engineer.py` | Efficiency ratios, rolling/expanding features, opponent strength, team-season metrics, consistency metrics, clustering | pandas |
| `src/analysis/analyze.py` | Core statistics: home advantage, efficiency persistence, shot decomposition, HT-FT dynamics, consistency, discipline, robustness | pandas + scipy + sklearn |
| `src/analysis/advanced_insights.py` | Analyses A–F (finishing persistence, profile changes, HT deficit recovery, volatility/streakiness, pooled vs within-league, HA changes) | pandas + scipy + sklearn |
| `src/analysis/new_insights.py` | Questions 1–6 (opponent dependence, HA components, close-match dependence, rest asymmetry, PI lead results, adaptation after promotion) | pandas + scipy + sklearn |
| `src/analysis/next_insights.py` | Questions 7–11 (competitive balance, style fingerprints, corner efficiency, shooting profiles, clutch) | pandas + scipy + sklearn |
| `src/analysis/fresh_insights.py` | Questions 12–16 (draw specialists, tempo, game state, referee characteristics, Pythagorean residuals) | pandas + scipy + sklearn |
| `src/visualization/plots.py` | Figures 1–11 | matplotlib + seaborn (pandas inputs) |
| `src/visualization/insight_plots.py` | Figures A–F | matplotlib + seaborn |
| `src/visualization/new_insight_plots.py` | Figures 1–6 | matplotlib + seaborn |
| `src/visualization/next_insight_plots.py` | Figures 7–11 | matplotlib + seaborn |
| `src/visualization/fresh_insight_plots.py` | Figures 12–16 | matplotlib + seaborn |
| `tests/test_core.py` | 17 tests / 7 foundational safeguards | pandas |
| `tests/test_performance.py` | Correctness + loose performance regression bounds | pandas |

**17 research/analytical question groups** are implemented across figures 1–11, A–F, 1–6, 7–11, 12–16.

## 2. Test suite baseline

```
python -m pytest tests/test_core.py -v
```
- **17 passed, 0 failed** in ~9.8 s (E0 × seasons 2122–2324, ~760 matches loaded by the test fixture).

The 7 foundational safeguards are all covered:
1. Home-advantage symmetry null (binomial test p = 0.5 among decisive matches)
2. Consecutive-season matching (Season_Num diff == 1 AND same division)
3. Comeback denominators (HT trailing matches only)
4. Missing vs. zero handling (all-missing preserved as NaN)
5. Zero-denominator ratios (→ NaN, never 0)
6. Lagged feature leakage (rolling/expanding shifted by 1; Rest_Days NaN for season openers)
7. Match / team-perspective consistency (2× rows, points/goals/shots symmetry)

## 3. Pipeline execution baseline (measured)

**Configuration:** `--data-dir . --leagues E0 --seasons 2122 2223 --seed 42`  
(i.e. the smallest meaningful multi-season slice: 2 English Premier League seasons, 760 matches, 1,520 team-matches, 40 team-seasons at min_matches=20).

| Metric | Value |
|--------|-------|
| Total wall-clock time | **125.3 s** |
| Peak resident set size (RSS) | **~599 MB** |
| Master matches loaded | 760 |
| Team-match rows | 1,520 |
| Team-seasons (min 20 matches) | 40 |
| Figures generated | 37 PNG (150 DPI) |
| Tables generated | 71 CSV |
| Reusable Parquet datasets | 2 (`master_matches.parquet`, `team_matches.parquet`) |
| Metadata JSON files | 3 |

### Stage breakdown (approximate, small slice)

| Stage | Time |
|-------|------|
| Data ingestion (ZIP → master) | ~0.1 s |
| Team-match construction | ~0.1 s |
| Feature engineering (ratios, rolling, opponent strength) | ~0.8 s |
| Team-season + consistency metrics | ~0.4 s |
| Core statistical analysis | ~1.3 s |
| Advanced + new + next + fresh insights | ~9 s |
| **Visualization (37 figures)** | **~77 s (dominant)** |
| Export / metadata / validation | ~2 s |

**Key observation:** Even on this tiny slice, visualization (matplotlib/seaborn, 37 figures) dominates runtime (~60%). Data loading/feature engineering/analysis are already sub-second to low-seconds. The migration should therefore (a) make the data/feature/analysis stack lazy and vectorized with Polars/DuckDB for the full 8-league × 7-season scale, and (b) avoid pandas↔Polars round-trips, converting only small plotting datasets at the Matplotlib boundary.

## 4. Intermediate dataset sizes (baseline small slice)

| Artifact | Path | Size |
|----------|------|------|
| Master matches (Parquet) | `output/data/master_matches.parquet` | 25.6 KB |
| Team matches (Parquet) | `output/data/team_matches.parquet` | 446 KB |

> Full-scale (8 leagues × 7 seasons ≈ 23,787 matches / 47,574 team-matches per README) intermediate sizes were not measured for the baseline; they will be measured in Phase 11.

## 5. Known pandas dependencies (pre-migration)

Pandas is used throughout for:
- CSV ingestion from ZIP archives (`pd.read_csv`)
- DataFrame concatenation across seasons/leagues (`pd.concat`)
- Team-perspective long-format construction (`pd.concat([home, away])`)
- GroupBy aggregations (`groupby(...).agg()`, `.transform()`, `.apply()`)
- Rolling/expanding windows (`rolling()`, `expanding()`)
- Merges/joins (`merge`, `join`)
- `pd.crosstab`, `pd.qcut`, `pd.get_dummies`, `pivot_table`
- All statistical analysis data munging

**Migration target:** Replace these with Polars (dataframe engine) + DuckDB (analytical SQL) + Parquet (reusable storage), with pandas retained only at the Matplotlib/seaborn plotting boundary where required.

## 6. Baseline representative outputs saved for equivalence testing

Representative outputs from the existing pipeline (small slice) have been saved to `/tmp/opencode/baseline_outputs/` for later numerical-equivalence comparison:

- `data/master_matches.parquet`, `data/team_matches.parquet`
- `tables/*.csv` (71 machine-readable table CSVs behind figures)
- `metadata/manifest.json`, `data_audit_summary.json`, `feature_dictionary.json`

These will be used in Phase 10 (correctness/regression testing) to verify the Polars + DuckDB pipeline reproduces statistically equivalent results.

## 7. Open questions / risks for migration

- **pandas 3.0.6** is newly released; verify Polars/PyArrow/DuckDB compatibility under Python 3.14.
- Visualization runtime is dominated by Matplotlib/seaborn figure rendering, not dataframe logic; migrating data prep to Polars will not materially speed figure rendering itself (that is out of scope, but pandas↔Polars conversion at the plotting boundary will be minimized).
- The full 8-league × 7-season run has not been timed at baseline; will be measured in Phase 11 after migration for a like-for-like comparison on identical inputs.
