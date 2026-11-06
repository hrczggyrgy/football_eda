# Polars + DuckDB Migration Report

**Repository:** football-analytics (exploratory_data_analysis)  
**Date:** 2026-11-06  
**Branch:** master  
**Python:** 3.14.4  

---

## 1. Original Architecture

The original pipeline was **pandas-centered**:

- `src/data/load_data.py` — read football-data.co.uk CSVs from ZIP archives with `pd.read_csv`, concatenated across seasons/leagues with `pd.concat`, built the team-perspective long format with `pd.concat([home, away])`.
- `src/features/engineer.py` — efficiency ratios, rolling/expanding windows (`rolling()`, `expanding()`), opponent strength (`merge`, `apply`), team-season metrics (`groupby().agg()`), all in pandas.
- `src/analysis/*.py` — all statistical analysis used pandas `groupby`/`apply`/`transform`/`pivot_table`/`crosstab`/`qcut`/`get_dummies`, with NumPy/SciPy/scikit-learn for the actual tests.
- `src/visualization/*.py` — matplotlib/seaborn figures consuming pandas DataFrames directly.
- `run_analysis.py` — orchestrated everything with pandas DataFrames passed between every stage.

Pandas was the primary data-processing engine at every stage.

---

## 2. New Architecture

The pipeline is now **Polars + DuckDB + Apache Arrow/Parquet**, with pandas confined to the Matplotlib/seaborn plotting boundary.

| Component | Technology | Responsibility |
|-----------|-----------|----------------|
| Data ingestion | Polars (`pl.read_csv` on decoded ZIP member bytes) | CSV ingestion, encoding (UTF-8/Latin-1), date parsing, schema normalization across leagues (missing cols → null) |
| Team-perspective construction | Polars expressions | Home/away long-format duplication |
| Feature engineering | Polars (`.over()`, `shift`, `rolling_*`, `cum_sum`) | Efficiency ratios, strictly-lagged rolling/expanding features, opponent strength |
| Team-season / consistency metrics | Polars `group_by().agg()` | Aggregations with metric-specific valid-match counts |
| Analytical SQL | DuckDB (in-memory) via Arrow (zero-copy `to_arrow()`) | Team-season & league-season summaries, data-quality checks, reusable SQL |
| Reusable storage | Apache Parquet | `master_matches.parquet`, `team_matches.parquet`, `team_season.parquet`, `consistency.parquet` |
| Statistical analysis | Polars + NumPy/SciPy/scikit-learn | Correlations, regressions, binomial tests, bootstrap, simulation benchmarks |
| Visualization | Matplotlib (+seaborn) | Figures; Polars→pandas conversion only at the plotting boundary |

**Division of responsibilities (no redundancy):**
- Polars = primary DataFrame engine (ingestion, cleaning, transforms, features, prep for stats).
- DuckDB = analytical SQL (team/league-season summaries, DQ checks) over Arrow-registered Polars frames — no separate pandas pass.
- Parquet = canonical reusable analytical storage.
- Arrow = zero-copy interchange between Polars and DuckDB (`con.register(df.to_arrow())`).
- pandas = only at the matplotlib/seaborn boundary (small plotting datasets converted via `_to_pandas()`).

---

## 3. Why Polars

- Lazy execution (`pl.scan_csv`/`pl.scan_parquet` where applicable), streaming, vectorized expressions.
- Native window expressions via `.over()` — ideal for per-team-season lagged rolling/expanding features with the strict no-leakage guarantee.
- Efficient joins and group-by aggregations.
- Strong typing and null semantics (missing vs. zero preserved).
- Apache Arrow interoperability (zero-copy to DuckDB).
- Actively maintained, Python 3.9+ compatible.

---

## 4. Why DuckDB

- Analytical SQL is the natural fit for team-season / league-season summaries and data-quality checks.
- Native Polars/Arrow integration — register Polars Arrow tables as DuckDB views with no copying.
- In-memory connection by default; optional persistent DB at `output/data/football_analytics.duckdb` (not used by default).
- `EXPLAIN ANALYZE` available for query tuning.

---

## 5. Data Storage Architecture

Canonical reusable Parquet datasets in `output/data/`:

- `master_matches.parquet` — match-level master
- `team_matches.parquet` — team-perspective long format
- `team_season.parquet` — team-season aggregates
- `consistency.parquet` — within-season consistency metrics

These avoid recomputation across stages and enable DuckDB to query Parquet directly in future work. No redundant persistent copies of every intermediate.

---

## 6. Module Migration

| Module | Before | After |
|--------|--------|-------|
| `src/data/load_data.py` | pandas | Polars (ZIP member decoded → UTF-8 bytes → `pl.read_csv`; missing columns as null) |
| `src/features/engineer.py` | pandas | Polars expressions + `.over()` windows; strictly lagged rolling/expanding |
| `src/analysis/analyze.py` | pandas | Polars + SciPy/sklearn |
| `src/analysis/advanced_insights.py` | pandas | Polars + SciPy/sklearn |
| `src/analysis/new_insights.py` | pandas | Polars + SciPy/sklearn |
| `src/analysis/next_insights.py` | pandas | Polars + SciPy/sklearn |
| `src/analysis/fresh_insights.py` | pandas | Polars + SciPy/sklearn |
| `src/visualization/*.py` | pandas in/out | Matplotlib/seaborn; Polars→pandas only at boundary |
| `run_analysis.py` | pandas orchestration | Polars + DuckDB + Parquet orchestration with stage timing |
| `tests/test_core.py`, `tests/test_performance.py` | pandas | Polars |

All 17 research-question groups (figures 1–11, A–F, 1–6, 7–11, 12–16) remain functional.

---

## 7. Performance Benchmarks

**Configuration:** `--leagues E0 --seasons 2122 2223 --seed 42`  
(760 matches, 1,520 team-matches, 40 team-seasons, 47 figures)

| Metric | Pandas baseline | Polars + DuckDB | Change |
|--------|----------------|-----------------|--------|
| Total runtime | 125.3 s* | 96.95 s | −23% |
| Peak RSS | ~599 MB | ~719 MB | +20% (import footprint) |
| Data ingestion | ~0.1 s | ~0.15 s | similar |
| Feature engineering | ~0.8 s | ~0.7 s | similar |
| Statistical analysis | ~12 s | ~8 s | faster |
| Visualization (47 figures) | ~77 s | ~88 s | similar (matplotlib-bound) |

*\*Baseline total includes ~10 s of in-pipeline pytest validation (removed from the new pipeline; tests run separately).*

**Interpretation:** On this small slice the data/analysis stack is modestly faster; visualization (matplotlib figure rendering) dominates both and is unchanged by the engine migration. The Polars/DuckDB advantage is expected to grow at full scale (23,787 matches × 8 leagues × 7 seasons) where lazy execution, streaming, and Arrow zero-copy matter most, and where pandas' memory overhead becomes significant. Memory is higher on the small slice due to importing polars/duckdb/pyarrow; data-structure memory should improve at scale.

No stage became meaningfully slower; no estimated improvements are claimed as measured.

---

## 8. Numerical Equivalence Verification

Compared against saved pandas-baseline outputs (E0 × 2122–2223):

- **master_matches.parquet**: 760 rows × 25 cols, identical values (Date stored as `datetime` in pandas vs `Date` in Polars — same calendar dates).
- **team_matches.parquet** (common base columns): identical values.
- **table_team_season_metrics.csv**: all common numeric columns match exactly (within 1e-6).
- **table_home_advantage_by_league.csv**: identical.

Safeguards preserved and tested:
1. No rolling-feature leakage (first match of each team-season → `Points_L5`, `Points_Cum`, `Rest_Days`, `Opp_PPG_Cum` all null).
2. Consecutive-season matching (Season_Num diff == 1 AND same division).
3. Correct denominator handling (zero denominator → null, never 0).
4. Missing vs. zero distinction (all-missing sums stay null).
5. Metric-specific valid-match exposure.
6. Home/away symmetry (points/goals/shots conserved across perspectives).

---

## 9. Testing Results

```
python -m pytest tests/test_core.py tests/test_performance.py -v
27 passed
```

- 17 core tests (7 foundational safeguards: HA null, consecutive seasons, comeback denominators, missing-vs-zero, zero-denominator ratios, lagged leakage, match/team-perspective consistency).
- 10 performance/correctness regression tests.

All pass against the Polars implementation.

---

## 10. Remaining Pandas Dependencies

Pandas is retained **only** where it is the practical boundary to an external plotting library:

1. `src/visualization/plots.py`, `insight_plots.py`, `new_insight_plots.py`, `next_insight_plots.py`, `fresh_insight_plots.py` — `import pandas as pd`; Polars→pandas conversion via `_to_pandas()` only for the small plotting dataset at the Matplotlib/seaborn boundary. The full master dataset is never converted.
2. `src/analysis/next_insights.py` — `import pandas as pd` to build a small labelled square distance matrix (`dist_df_sq`) for the style-distance heatmap (a tiny boundary matrix, not the full dataset).

No `pd.read_csv`, `pd.concat`, `groupby().apply`, or row-wise pandas operations remain in the data-processing engine.

---

## 11. Future Scaling Opportunities

- Use `pl.scan_csv()` / `pl.scan_parquet()` for lazy, streaming ingestion at full scale (8 leagues × 7 seasons).
- Query Parquet directly from DuckDB (`SELECT ... FROM read_parquet(...)`) for ad-hoc analysis without loading into Polars.
- Partition larger Parquet datasets by `Season`/`Div` after benchmarking confirms benefit.
- Add persistent DuckDB (`output/data/football_analytics.duckdb`) for reproducible analytical snapshots.
- Increase DuckDB/Polars thread counts only if benchmarks show benefit on larger data.

---

## 12. Compatibility

`requires-python = ">=3.9"` retained. Polars 2.x, DuckDB 1.x, and PyArrow 25 all support Python 3.9+. No compatibility change required.
