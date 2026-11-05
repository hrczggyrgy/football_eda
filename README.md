# Football Analytics Study

A rigorous exploratory data analysis of 7 seasons (2019/20–2025/26) across 8 European football leagues, examining team performance, match dynamics, and competition structure using football-data.co.uk datasets.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full analysis pipeline
python run_analysis.py
```

## Repository Structure

```
.
├── run_analysis.py              # Main pipeline entry point
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── src/
│   ├── data/
│   │   └── load_data.py         # Data loading & preprocessing
│   ├── features/
│   │   └── engineer.py          # Feature engineering
│   ├── analysis/
│   │   └── analyze.py           # Statistical analysis
│   └── visualization/
│       └── plots.py             # Publication-quality figures
├── data/                        # Raw data (football-data.co.uk zips)
│   ├── 1920/data.zip
│   ├── 2021/data.zip
│   └── ...
└── output/                      # Generated artifacts (after running)
    ├── fig1_home_advantage.png
    ├── fig2_efficiency_vs_points.png
    ├── ...
    ├── table_*.csv
    ├── master_matches.parquet
    ├── team_matches.parquet
    ├── feature_dictionary.json
    └── data_audit_summary.json
```

## Data Sources

- **football-data.co.uk** match-level CSVs for 8 leagues × 7 seasons
- Leagues: Premier League (E0), Championship (E1), League One (E2), League Two (E3), La Liga (SP1), Bundesliga (D1), Serie A (I1), Ligue 1 (F1)
- **Excluded**: All betting/odds columns (80–100+ per file)
- **Total**: 23,787 matches, 47,574 team-matches, 215 unique teams

## Key Findings

1. **Home advantage is crowd-driven**: 43% home win rate overall; dropped to **37.9% in Premier League during 2020/21 COVID season** (−6.4 pp)

2. **Shot volume > accuracy > finishing** for predicting points: Multiple regression R²=0.72; standardized β: Shots_PG=0.23, GPSOT=0.17, Accuracy=0.10

3. **Attacking efficiency persists; finishing regresses**: Goals/Shot correlates 0.35 across half-seasons, 0.42 season-to-season. Goals/Shot on Target only 0.22 / 0.26

4. **55% of goals come in second half** (every league 54.7–56.2%). Trailing teams see more SH goals but rarely overturn deficits: only **7.4% win, 15.6% avoid defeat** from HT trailing

5. **Elite teams are consistently excellent**: Bayern, PSG, Real Madrid, Man City, Inter, Liverpool all >2.0 PPG with Points CV ≈ 0.50

## Outputs

| Type | Files |
|------|-------|
| **Figures** | 11 publication-quality PNGs (150 DPI) |
| **Tables** | 9 machine-readable CSVs |
| **Data** | 3 Parquet files (master, team-match, features) |
| **Metadata** | Feature dictionary, data audit summary |

## Requirements

- Python 3.9+
- pandas ≥ 2.0
- numpy ≥ 1.24
- matplotlib ≥ 3.7
- seaborn ≥ 0.12
- scikit-learn ≥ 1.3
- scipy ≥ 1.10

## Methodology

- **Observational unit**: Match (23,787) → Team-match (47,574) for perspective-specific analysis
- **No leakage**: All rolling/expanding features use `.shift(1)` before windowing
- **No betting data**: All bookmaker odds and market-derived fields excluded
- **COVID season handled**: 2020/21 flagged as structural break for home advantage
- **Statistical rigor**: Effect sizes + uncertainty prioritized over p-values; Pearson correlations validated with Spearman

## Reproducibility

All analysis is contained in `run_analysis.py` with modular `src/` components. Run end-to-end with:

```bash
python run_analysis.py
```

Outputs saved to current directory.