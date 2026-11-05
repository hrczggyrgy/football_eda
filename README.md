# Football Analytics Study

A rigorous exploratory data analysis of 7 seasons (2019/20–2025/26) across 8 European football leagues, examining team performance, match dynamics, and competition structure using football-data.co.uk datasets.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full analysis pipeline (7 seasons × 8 leagues)
python run_analysis.py --data-dir . --output-dir output --seed 42

# Quick test (1 league × 2 seasons)
python run_analysis.py --data-dir . --output-dir output_test --leagues E0 --seasons 2122 2223 --seed 42

# Run tests
python -m pytest tests/test_core.py -v
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
│   ├── AUDIT.md                 # Complete repository audit
│   ├── UPGRADE_SUMMARY.md       # Upgrade completion status
│   ├── ANALYSIS_REPORT.md       # Core analyses report
│   ├── visual_insights.md       # Visual insights report (core + advanced)
│   └── new_insights_report.md   # 6 new research questions report
├── src/
│   ├── data/
│   │   └── load_data.py         # Data loading & preprocessing
│   ├── features/
│   │   └── engineer.py          # Feature engineering (ratios, rolling, opponent strength)
│   ├── analysis/
│   │   ├── analyze.py           # Core statistical analyses (11 figures)
│   │   ├── advanced_insights.py # Advanced analyses A-F (6 figures)
│   │   └── new_insights.py      # NEW: 6 research questions (6 figures)
│   └── visualization/
│       ├── plots.py             # Core figures (1-11)
│       ├── insight_plots.py     # Advanced figures (A-F)
│       └── new_insight_plots.py # NEW: New insight figures (1-6)
├── data/                        # Raw data (football-data.co.uk zips)
│   ├── 1920/data.zip
│   ├── 2021/data.zip
│   └── ...
├── output/                      # Generated artifacts (after running)
│   ├── figures/                 # 23 publication-quality PNGs (150 DPI)
│   ├── tables/                  # 37 machine-readable CSVs
│   ├── data/                    # 2 Parquet files
│   └── metadata/                # 3 JSON files (audit, feature_dict, manifest)
└── tests/
    └── test_core.py             # 14 tests covering 7 foundational safeguards
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
1. **Opponent-dependent performance** — Which teams perform relatively better against stronger opponents?
2. **Components of home advantage** — Is HA associated more with shot generation, shot suppression, or conversion?
3. **Close-match dependence** — Which teams rely heavily on one-goal results, and does that advantage persist?
4. **Rest asymmetry** — Is being less rested than the opponent associated with worse performance?
5. **Performance indicators lead results** — Does sustained improvement in shot dominance precede improved results?
6. **Adaptation after promotion** — Do promoted teams improve their relative performance during their first season?

## Key Findings

| # | Finding | Evidence |
|---|---------|----------|
| 1 | **Home advantage is crowd-driven** | 43% home win rate overall; dropped to 37.9% in PL during 2020/21 COVID (−6.4 pp) |
| 2 | **Shot volume > accuracy > finishing** for predicting points | Multiple regression R²=0.72; β: Shots_PG=0.23, GPSOT=0.17, Accuracy=0.10 |
| 3 | **Attacking efficiency persists; finishing regresses** | GPS: r=0.35 (half), 0.42 (season); GPSOT: r=0.22 (half), 0.26 (season) |
| 4 | **55% of goals in second half** | Consistent 54.7–56.2% across leagues; trailing teams see more SH goals but rarely overturn |
| 5 | **Elite teams are consistently excellent** | Bayern, PSG, Real Madrid, Man City, Inter, Liverpool >2.0 PPG, Points CV ≈ 0.50 |
| 6 | **HA = shot generation + suppression, not finishing** | +1.8 shots/game, -1.3 opponent shots/game, conversion unchanged |
| 7 | **Close-match reliance is noise** | Zero persistence (r=0.05); shot dominance 7× more persistent |
| 8 | **Rest deficit hurts away teams more** | Asymmetric: away -0.05 PPG/day vs home -0.02 PPG/day |
| 9 | **Shot dominance leads results** | First 10 matches predicts next 10 PPG incrementally over lagged PPG (ΔR²=0.03) |
| 10 | **Promoted teams adapt steadily** | +0.15 PPG improvement; baseline strength explains 40% variance |

## Methodological Safeguards

- **No leakage**: All rolling/expanding features use `.shift(1)` before windowing
- **Consecutive seasons only**: Promotion/relegation requires Season_Num diff=1 AND adjacent division tiers
- **Valid denominators**: Comeback rates use HT trailing matches only; zero denom → NaN
- **Symmetry null**: HA binomial test uses p=0.5 among decisive matches (not 1/3)
- **Simulation benchmark**: 100 sims/team-season; calibrated outcome probabilities
- **Exchangeability documented**: Match outcomes exchangeable conditional on expected points
- **Exposure tracked**: Metric-specific valid-match counts; all-missing preserved as NaN
- **No betting data**: Only 23 non-betting columns in master dataset

## Reproducibility

All analysis is contained in `run_analysis.py` with modular `src/` components.

```bash
# Full pipeline
python run_analysis.py --data-dir . --output-dir output --seed 42

# With custom leagues/seasons
python run_analysis.py --data-dir . --output-dir output --leagues E0 SP1 D1 --seasons 2122 2223 2324 --seed 123

# Skip new insights (faster)
python run_analysis.py --skip-new
```

**Artifacts generated:**
- `output/figures/` — 23 PNG files (150 DPI, publication-ready)
- `output/tables/` — 37 CSV files (machine-readable chart data)
- `output/data/` — 2 Parquet files (master, team-matches)
- `output/metadata/` — manifest.json, feature_dictionary.json, data_audit_summary.json

**Tests:**
```bash
python -m pytest tests/test_core.py -v
```
14 tests covering 7 foundational safeguards (home-advantage null, consecutive seasons, comeback denominators, missing/zero handling, zero-denominator ratios, lagged feature leakage, match/team-perspective consistency).

## Requirements

- Python 3.9+
- pandas ≥ 2.0
- numpy ≥ 1.24
- matplotlib ≥ 3.7
- seaborn ≥ 0.12
- scikit-learn ≥ 1.3
- scipy ≥ 1.10

Install via: `pip install -r requirements.txt` or `pip install -e .[dev]` for development.

## License

MIT License — see [LICENSE](LICENSE) for details.