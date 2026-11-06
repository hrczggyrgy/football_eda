# Football Analytics Study

A rigorous exploratory data analysis of 7 seasons (2019/20–2025/26) across 8 European football leagues, examining team performance, match dynamics, and competition structure using football-data.co.uk datasets.

**Engine:** Polars (primary DataFrame engine) + DuckDB (analytical SQL) + Apache Arrow/Parquet (reusable storage), with NumPy/SciPy/scikit-learn for statistics and Matplotlib for plotting. Pandas is retained only as a compatibility layer at the Matplotlib/seaborn plotting boundary.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full analysis pipeline (7 seasons × 8 leagues)
python run_analysis.py --data-dir . --output-dir output --seed 42

# Quick test (1 league × 2 seasons)
python run_analysis.py --data-dir . --output-dir output_test --leagues E0 --seasons 2122 2223 --seed 42

# Skip insight suites for a faster run
python run_analysis.py --skip-new --skip-next --skip-fresh

# Run tests
python -m pytest tests/ -v
```

## New Architecture

```
Raw ZIP CSVs
    │
    ▼
Polars ingestion  (UTF-8/Latin-1, Date parsing, missing cols → null)
    │
    ▼
master_matches.parquet  ◄──►  DuckDB (in-memory, Arrow zero-copy)
team_matches.parquet        analytical SQL: team/league-season
team_season.parquet         summaries & data-quality checks
consistency.parquet
    │
    ▼
Polars feature engineering  (efficiency ratios, strictly-lagged
rolling/expanding windows, opponent strength, team-season metrics,
clustering)
    │
    ▼
Statistical analysis  (Polars + NumPy/SciPy/scikit-learn)
    │
    ▼
Visualization  (Matplotlib; Polars→pandas only at the plotting boundary)
    │
    ▼
output/  (figures PNG, tables CSV, data Parquet, metadata JSON)
```

## Repository Structure

```
.
├── run_analysis.py              # Main pipeline entry point (CLI-configurable)
├── requirements.txt             # Python dependencies
├── pyproject.toml               # Modern packaging config
├── LICENSE                      # MIT License
├── README.md                    # This file
├── .gitignore
├── docs/                        # Documentation
│   ├── migration_baseline.md    # Pre-migration baseline & measurements
│   └── polars_duckdb_migration_report.md  # Full migration report
├── src/
│   ├── data/
│   │   └── load_data.py         # Polars ingestion & normalization
│   ├── features/
│   │   └── engineer.py          # Polars feature engineering
│   ├── analysis/                # Polars + SciPy/sklearn statistics
│   │   ├── analyze.py
│   │   ├── advanced_insights.py
│   │   ├── new_insights.py
│   │   ├── next_insights.py
│   │   └── fresh_insights.py
│   └── visualization/           # Matplotlib; Polars→pandas at boundary only
│       ├── plots.py
│   │   ├── insight_plots.py
│   │   ├── new_insight_plots.py
│   │   ├── next_insight_plots.py
│   │   └── fresh_insight_plots.py
├── data/                        # Raw data (football-data.co.uk zips)
│   ├── 1920/data.zip
│   └── ...
├── output/                      # Generated artifacts (after running)
│   ├── figures/                 # publication-quality PNGs
│   ├── tables/                  # machine-readable CSVs
│   ├── data/                    # Parquet: master_matches, team_matches,
│   │                              team_season, consistency
│   └── metadata/                # manifest, audit, feature dictionary
└── tests/
    ├── test_core.py             # 17 tests / 7 foundational safeguards
    └── test_performance.py      # correctness + performance regressions
```

## Data Sources

- **football-data.co.uk** match-level CSVs for 8 leagues × 7 seasons
- Leagues: Premier League (E0), Championship (E1), League One (E2), League Two (E3), La Liga (SP1), Bundesliga (D1), Serie A (I1), Ligue 1 (F1)
- **Excluded**: All betting/odds columns (80–100+ per file)
- **Total**: 23,787 matches, 47,574 team-matches, 215 unique teams

## Analyses

### Core Analyses (Figures 1–11)
1. Home advantage across leagues, seasons, COVID disruption
2. Efficiency metrics vs Points per Game
3. Efficiency persistence (first vs second half of season)
4. Shot volume, accuracy, finishing decomposition
5. Attack-Defense profiles (K-means clustering)
6. Half-time to Full-time transitions
7. Second-half goals & comebacks
8. Team profiles & consistency
9. Consistency leaders
10. Disciplinary patterns
11. Foul differential

### Advanced Analyses (Figures A–F)
A. Finishing persistence (early vs late season)
B. Team profile changes (consecutive season trajectories)
C. Halftime deficit recovery (conditional on deficit, venue, strength)
D. Volatility & streakiness (residual dispersion + runs test)
E. Pooled vs within-league relationships
F. Home advantage changes across seasons

### New Research Questions (Figures 1–6)
1. Opponent-dependent performance
2. Components of home advantage
3. Close-match dependence
4. Rest asymmetry
5. Performance indicators lead results
6. Adaptation after promotion

### Next Insights (Figures 7–11)
7. Competitive balance evolution
8. League style fingerprints
9. Corner efficiency & set-piece profiles
10. Selective vs volume shooting profiles
11. Clutch performance

### Fresh Insights (Figures 12–16)
12. Draw specialists & match-outcome structure
13. Match tempo & statistical structure
14. Game state effects on match statistics
15. Referee characteristics & potential biases (English leagues)
16. Expected vs actual points (Pythagorean residuals)

## How Parquet Storage Works

Reusable processed datasets are written to `output/data/` as Parquet:

- `master_matches.parquet` — match-level master
- `team_matches.parquet` — team-perspective long format
- `team_season.parquet` — team-season aggregates
- `consistency.parquet` — within-season consistency metrics

These are created once and reused across stages, avoiding recomputation. DuckDB can query them directly:

```python
import duckdb
con = duckdb.connect()
con.execute("""
    SELECT Div, Season, AVG(PPG) AS avg_ppg
    FROM read_parquet('output/data/team_season.parquet')
    GROUP BY Div, Season
""").show()
```

## How to Query Using DuckDB

The pipeline registers Polars DataFrames as DuckDB Arrow views (zero-copy) and runs reusable SQL for team-season and league-season summaries plus data-quality checks. An in-memory connection is used by default.

## How to Reproduce Benchmarks

Stage timings are recorded in `output/metadata/manifest.json` (`stage_times_seconds`, `total_runtime_seconds`).

```bash
# Baseline-sized slice (1 league × 2 seasons)
python run_analysis.py --data-dir . --output-dir bench_slice --leagues E0 --seasons 2122 2223 --seed 42

# Full scale (8 leagues × 7 seasons) — longer run
python run_analysis.py --data-dir . --output-dir bench_full --seed 42
```

Compare `stage_times_seconds` between runs; the manifest also records software versions (polars, duckdb, numpy).

## Key Methodological Safeguards

- **No leakage**: all rolling/expanding features use `.shift(1)` before windowing
- **Consecutive seasons only**: promotion/relegation requires Season_Num diff = 1 AND same division
- **Valid denominators**: zero denominator → null, never 0
- **Symmetry null**: home-advantage binomial test uses p = 0.5 among decisive matches
- **Missing vs. zero**: all-missing sums preserved as null
- **Metric-specific valid-match exposure** for per-game rates

## Requirements

- Python 3.9+
- polars ≥ 1.5, duckdb ≥ 0.8, pyarrow ≥ 8
- numpy ≥ 1.24, matplotlib ≥ 3.7, seaborn ≥ 0.12, scikit-learn ≥ 1.3, scipy ≥ 1.10
- pandas ≥ 2.0 (plotting boundary only)

Install via: `pip install -r requirements.txt` or `pip install -e .[dev]` for development.

## License

MIT License — see [LICENSE](LICENSE).
