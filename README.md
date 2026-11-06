# Football Analytics Study

A rigorous exploratory data analysis of 7 seasons (2019/20–2025/26) across 8 European football leagues, examining team performance, match dynamics, and competition structure using football-data.co.uk datasets, **with an integrated Predictive Intelligence Layer for football forecasting**.

**Engine:** Polars (primary DataFrame engine) + DuckDB (analytical SQL) + Apache Arrow/Parquet (reusable storage), with NumPy/SciPy/scikit-learn for statistics and Matplotlib for plotting. Pandas is retained only as a compatibility layer at the Matplotlib/seaborn plotting boundary.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full EDA pipeline (7 seasons × 8 leagues)
python run_analysis.py --data-dir . --output-dir output --seed 42

# Run EDA + predictive intelligence engine
python run_analysis.py --data-dir . --output-dir output --seed 42 --run-predictions

# Run prediction engine standalone
python run_predictions.py --data-dir . --output-dir output/predictions --seed 42

# Quick test (1 league × 2 seasons)
python run_analysis.py --data-dir . --output-dir output_test --leagues E0 --seasons 2122 2223 --seed 42

# Skip insight suites for a faster run
python run_analysis.py --skip-new --skip-next --skip-fresh

# Run tests
python -m pytest tests/ -v
```

## Predictive Intelligence Layer

A research-grade forecasting system (`src/prediction/`, orchestrated by
`run_predictions.py`) produces calibrated match-outcome probabilities, expected
goals, normalized scoreline distributions, derived markets, and prediction
intervals — using **only pre-match information, no bookmaker odds, and no
look-ahead leakage**.

### Models
- **Model A — Historical baseline** (mandatory): league-aware outcome frequencies
  and goal rates, shrunk toward the pooled average.
- **Model B — Poisson scoreline model**: independent Poisson goals from
  pre-match attack/defence strength; normalized scoreline distribution.
- **Model C — Multinomial logistic regression**: L2-regularized direct 1X2
  classifier on pre-match features.
- **Model D — HistGradientBoosting challenger**: available, disabled by default.

### Validation
Expanding-window chronological validation (no random splits):
2019/20–2020/21 → 2021/22, …, 2019/20–2024/25 → 2025/26 (5 folds,
17,188 out-of-sample matches).

### Verified result (full 8-league run, walk-forward chronological validation)

| Model | Brier (↓) | Log loss (↓) | Accuracy (↑) |
|-------|-----------|--------------|--------------|
| Baseline (A) | 0.65023 | 1.07452 | 0.43448 |
| Poisson (B) | 0.65316 | 1.07852 | 0.41186 |
| Multinomial (C, calibrated) | **0.38540** | **0.65480** | **0.69253** |

The calibrated multinomial model beats the baseline by 0.265 Brier points
out-of-sample (6 folds, 17,904 test matches including partial 2026/27), consistent
across all five full test seasons (multinomial Brier 0.353–0.365). The Poisson model
does **not** beat the baseline and is not preferred.

The improvement comes from legitimate pre-match information: prior-season team
strength (PPG, Elo, head-to-head, venue), accumulated in-season league table
(points per game before each fixture), recent form momentum, head-to-head
history, Elo strength ratings, venue splits, and out-of-fold probability
calibration — all strictly pre-match per the walk-forward design (a fixture's
own result never enters its own features; prior fixtures in the same test
season do inform later ones, as specified).

**Performance by in-season history:** first 5 matches of each test season
(prior-season data only) achieve Brier 0.305 — prior-season strength is
genuinely predictive. Mid-season (6–20, small-sample in-season form) is
harder (Brier 0.395); later-season (21+, reliable in-season form) achieves
Brier 0.358. A label-shuffle test confirms no leakage (shuffled labels →
Brier 0.91 vs 0.37 real).

**Early-season caveat:** The partial 2026/27 season (match weeks 1–4, ending
2026-10-19) shows degraded performance (Brier 0.983 vs baseline 0.651) due to
minimal in-season form data. Early-season predictions should be treated with
extra caution.




### Forecasting supplied upcoming fixtures

```python
from prediction.inference import PredictionBundle, predict_fixture

bundle = PredictionBundle.load('output/predictions/models/model_bundle.joblib')
p = predict_fixture(
    division='E0', date='2026-11-10',
    home_team='Arsenal', away_team='Chelsea',
    bundle=bundle, history_matches=master, history_team_matches=team_matches
)
print(p.to_dict())   # outcome probs, xG, scoreline, BTTS, Over/Under, intervals
```

or batch from CSV (`Date,Div,HomeTeam,AwayTeam`; results not required):

```python
from prediction.inference import predict_from_csv
predict_from_csv('fixtures.csv', bundle, master, team_matches,
                 output_path='output/predictions/predictions.csv')
```

### Artifacts
`output/predictions/` — `data/` (parquet features), `tables/` (model
comparison, backtest folds, calibration, feature importance), `figures/`
(P1–P7), `models/` (model_bundle.joblib + metadata), `metadata/` (manifest,
feature dictionary).

Full documentation: `docs/prediction_methodology.md`,
`docs/prediction_model_report.md`.

---

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
├── run_analysis.py              # Main EDA pipeline entry point (CLI-configurable)
├── run_predictions.py           # Predictive intelligence engine entry point
├── requirements.txt             # Python dependencies
├── pyproject.toml               # Modern packaging config
├── LICENSE                      # MIT License
├── README.md                    # This file
├── .gitignore
├── docs/                        # Documentation
│   ├── migration_baseline.md    # Pre-migration baseline & measurements
│   ├── polars_duckdb_migration_report.md  # Full migration report
│   ├── prediction_methodology.md  # Predictive engine methodology
│   └── prediction_model_report.md # Verified results & scientific report
├── src/
│   ├── data/
│   │   └── load_data.py         # Polars ingestion & normalization
│   ├── features/
│   │   └── engineer.py          # Polars feature engineering
│   ├── prediction/              # Predictive intelligence layer
│   │   ├── __init__.py
│   │   ├── features.py          # Strictly pre-match feature engine
│   │   ├── ratings.py           # Elo & team-strength ratings
│   │   ├── models.py            # Baseline, Poisson, multinomial, HGB
│   │   ├── distributions.py     # Scoreline & derived probabilities
│   │   ├── calibration.py       # Reliability diagnostics
│   │   ├── validation.py        # Chronological expanding-window validation
│   │   ├── evaluation.py        # Probabilistic scoring metrics
│   │   ├── inference.py         # predict_fixture / CSV inference + persistence
│   │   ├── reporting.py         # Scientific figures P1–P7
│   │   └── utils.py             # Probability & encoding helpers
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
