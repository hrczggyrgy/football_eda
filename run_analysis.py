#!/usr/bin/env python3
"""
Main analysis pipeline for Football Analytics Study (Polars + DuckDB).

Orchestrates data loading, schema normalization, validation, Polars feature
engineering, DuckDB analytical SQL, statistical analysis, visualization, and
artifact generation.

Pandas is used only at the Matplotlib/seaborn plotting boundary.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import polars as pl
import duckdb

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data.load_data import (
    build_master_dataset, build_team_match_dataset, audit_data_quality,
    CORE_COLS, FULL_DATA_LEAGUES, PARTIAL_DATA_LEAGUES
)
from features.engineer import (
    add_efficiency_ratios, add_rolling_features, add_opponent_strength,
    compute_team_season_metrics, compute_consistency_metrics,
    add_season_order, add_next_season_metrics, cluster_team_profiles,
    SEASON_ORDER as ENG_SEASON_ORDER,
)
from analysis.analyze import (
    analyze_home_advantage, analyze_efficiency_persistence,
    analyze_shot_decomposition, analyze_ht_ft_dynamics,
    analyze_consistency, analyze_discipline, run_robustness_checks
)
from analysis.advanced_insights import (
    analyze_finishing_persistence, analyze_team_profile_changes,
    analyze_halftime_deficit_recovery, analyze_volatility_streakiness,
    analyze_pooled_vs_within_league, analyze_home_advantage_changes
)
from analysis.new_insights import (
    analyze_opponent_dependent_performance, analyze_home_advantage_components,
    analyze_close_match_dependence, analyze_rest_asymmetry,
    analyze_performance_indicators_lead_results, analyze_adaptation_after_promotion
)
from analysis.next_insights import run_next_insights
from analysis.fresh_insights import run_fresh_insights
from visualization.plots import create_all_figures
from visualization.insight_plots import create_advanced_figures
from visualization.new_insight_plots import create_all_new_figures
from visualization.next_insight_plots import create_all_next_figures
from visualization.fresh_insight_plots import create_all_fresh_figures

# DuckDB persistent database (optional, alongside in-memory)
DUCKDB_PATH = Path('output') / 'data' / 'football_analytics.duckdb'


def parse_args():
    parser = argparse.ArgumentParser(description='Football Analytics Study Pipeline')
    parser.add_argument('--data-dir', type=str, default='.',
                        help='Root directory containing season subdirectories with data.zip')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory for results')
    parser.add_argument('--leagues', type=str, nargs='+',
                        default=['E0', 'E1', 'E2', 'E3', 'SP1', 'D1', 'I1', 'F1'],
                        help='League codes to analyze')
    parser.add_argument('--seasons', type=str, nargs='+',
                        default=['1920', '2021', '2122', '2223', '2324', '2425', '2526'],
                        help='Season codes to analyze (e.g., 1920 for 2019/20)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--skip-new', action='store_true', help='Skip new insight analyses (questions 1-6)')
    parser.add_argument('--skip-next', action='store_true', help='Skip next insight analyses (questions 7-11)')
    parser.add_argument('--skip-fresh', action='store_true', help='Skip fresh insight analyses (questions 12-16)')
    return parser.parse_args()


def setup_output_dirs(output_dir: Path) -> Dict[str, Path]:
    dirs = {
        'root': output_dir,
        'figures': output_dir / 'figures',
        'tables': output_dir / 'tables',
        'metadata': output_dir / 'metadata',
        'data': output_dir / 'data',
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _jsonify(obj):
    """Recursively convert Polars/NumPy types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, float, str)) or obj is None:
        return obj
    return str(obj)


def main():
    args = parse_args()

    # Set random seeds for reproducibility
    np.random.seed(args.seed)

    DATA_DIR = Path(args.data_dir)
    OUTPUT_DIRS = setup_output_dirs(Path(args.output_dir))
    TARGET_LEAGUES = args.leagues
    SEASONS = args.seasons

    stage_times: Dict[str, float] = {}

    print("=" * 60)
    print("FOOTBALL ANALYTICS STUDY - PIPELINE (Polars + DuckDB)")
    print("=" * 60)
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIRS['root']}")
    print(f"Leagues: {TARGET_LEAGUES}")
    print(f"Seasons: {SEASONS}")
    print(f"Random seed: {args.seed}")

    # -----------------------------------------------------------------
    # 1. DATA INGESTION (Polars)
    # -----------------------------------------------------------------
    print("\n[1/11] Ingesting data (Polars)...")
    t0 = time.perf_counter()
    master = build_master_dataset(DATA_DIR, TARGET_LEAGUES, SEASONS, usecols=CORE_COLS)
    t_load = time.perf_counter() - t0
    stage_times['data_ingestion'] = t_load
    print(f"  Master matches: {master.height}")
    print(f"  Leagues: {master.select(pl.col('Div').unique()).to_series().to_list()}")
    print(f"  Seasons: {sorted(master.select(pl.col('Season').unique()).to_series().to_list())}")
    print(f"  Date range: {master.select(pl.col('Date').min()).to_series()[0]} to {master.select(pl.col('Date').max()).to_series()[0]}")

    t0 = time.perf_counter()
    team_matches = build_team_match_dataset(master)
    stage_times['team_match_construction'] = time.perf_counter() - t0
    print(f"  Team-match rows: {team_matches.height}")
    print(f"  Unique teams: {team_matches.select(pl.col('Team').n_unique()).to_series()[0]}")

    # Data quality audit
    t0 = time.perf_counter()
    audit = audit_data_quality(master, team_matches)
    stage_times['data_validation'] = time.perf_counter() - t0
    n_issues = sum(1 for v in audit.values() if v)
    print(f"  Data quality issues: {n_issues} categories checked")

    with open(OUTPUT_DIRS['metadata'] / 'data_audit_summary.json', 'w') as f:
        json.dump(_jsonify(audit), f, indent=2)

    # -----------------------------------------------------------------
    # 2. FEATURE ENGINEERING (Polars)
    # -----------------------------------------------------------------
    print("\n[2/11] Engineering features (Polars)...")
    t0 = time.perf_counter()
    team_matches = add_efficiency_ratios(team_matches)
    team_matches = add_rolling_features(team_matches, window=5, min_periods=3)
    team_matches = add_opponent_strength(team_matches)
    stage_times['efficiency_ratios'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    team_season = compute_team_season_metrics(team_matches, min_matches=20)
    stage_times['team_season_metrics'] = time.perf_counter() - t0
    print(f"  Team-seasons (20+ matches): {team_season.height}")

    # Add corner metrics for next insights
    team_season = team_season.with_columns([
        (pl.col('Corners') / pl.col('Matches')).alias('Corners_PG'),
        (pl.col('Corners_Against') / pl.col('Matches')).alias('Corners_Ag_PG'),
        pl.when(pl.col('Corners') > 0).then(pl.col('Shots') / pl.col('Corners')).otherwise(None).alias('Shots_Per_Corner'),
        pl.when(pl.col('Corners') > 0).then(pl.col('GF') / pl.col('Corners')).otherwise(None).alias('Goals_Per_Corner'),
    ])

    t0 = time.perf_counter()
    consistency = compute_consistency_metrics(team_matches, min_matches=10)
    stage_times['consistency_metrics'] = time.perf_counter() - t0
    print(f"  Consistency records: {consistency.height}")

    # -----------------------------------------------------------------
    # 3. ADVANCED FEATURES (Polars)
    # -----------------------------------------------------------------
    print("\n[3/11] Computing advanced features (Polars)...")
    t0 = time.perf_counter()
    team_season = add_season_order(team_season)
    team_season = add_next_season_metrics(team_season)

    profile_features = ['GF_PG', 'GA_PG', 'Shots_PG', 'Shots_Against_PG',
                        'Shot_Accuracy', 'Opp_Shot_Accuracy',
                        'Goals_Per_Shot', 'GA_Per_Opp_Shot']
    team_season = cluster_team_profiles(team_season, profile_features, n_clusters=5)
    stage_times['advanced_features'] = time.perf_counter() - t0
    print(f"  Clusters assigned: {team_season.select(pl.col('Cluster').is_not_null().sum()).to_series()[0]} team-seasons")

    # -----------------------------------------------------------------
    # 4. DUCKDB ANALYTICAL SQL (team-season & league-season summaries)
    # -----------------------------------------------------------------
    print("\n[4/11] DuckDB analytical SQL...")
    t0 = time.perf_counter()

    con = duckdb.connect(database=':memory:')
    try:
        # Register Polars DataFrames as DuckDB views via Arrow (zero-copy)
        con.register('master_matches', master.to_arrow())
        con.register('team_matches', team_matches.to_arrow())
        con.register('team_season', team_season.to_arrow())

        # Reusable analytical SQL: team-season summary
        team_season_sql = con.execute("""
            SELECT Div, Team, Season,
                   Matches, PPG, GF_PG, GA_PG,
                   Shots_PG, Shots_OT_PG, Shot_Accuracy,
                   Goals_Per_Shot, GA_Per_Opp_Shot,
                   Yellow_PG, Red_PG, Fouls_PG, Cards_PG
            FROM team_season
            ORDER BY Div, Team, Season
        """).pl()

        # Reusable analytical SQL: league-season summary
        league_season_sql = con.execute("""
            SELECT Div, Season,
                   COUNT(*) AS n_teams,
                   AVG(PPG) AS avg_ppg,
                   AVG(GF_PG) AS avg_gf_pg,
                   AVG(GA_PG) AS avg_ga_pg,
                   AVG(Shots_PG) AS avg_shots_pg
            FROM team_season
            GROUP BY Div, Season
            ORDER BY Div, Season
        """).pl()

        # Data quality check via SQL
        dq_checks = con.execute("""
            SELECT
                COUNT(*) AS total_rows,
                COUNT(Div) AS non_null_div,
                COUNT(Team) AS non_null_team,
                SUM(CASE WHEN Matches < 1 THEN 1 ELSE 0 END) AS invalid_matches
            FROM team_season
        """).fetchall()
        print(f"  DuckDB team_season rows: {team_season_sql.height}")
        print(f"  DuckDB league_season rows: {league_season_sql.height}")
        print(f"  DuckDB data-quality check: {dq_checks[0]}")
    finally:
        con.close()

    stage_times['duckdb_sql'] = time.perf_counter() - t0

    # -----------------------------------------------------------------
    # 5. STATISTICAL ANALYSIS (Polars + NumPy/SciPy/sklearn)
    # -----------------------------------------------------------------
    print("\n[5/11] Running core statistical analysis...")
    t0 = time.perf_counter()
    ha_results = analyze_home_advantage(master)
    stage_times['home_advantage'] = time.perf_counter() - t0
    print(f"  Home advantage: {ha_results['by_league'].height} leagues analyzed")

    t0 = time.perf_counter()
    eff_results = analyze_efficiency_persistence(team_matches, min_matches=20)
    stage_times['efficiency_persistence'] = time.perf_counter() - t0
    print(f"  Efficiency persistence: half-season & season-to-season")

    t0 = time.perf_counter()
    shot_results = analyze_shot_decomposition(team_season)
    stage_times['shot_decomposition'] = time.perf_counter() - t0
    print(f"  Shot decomposition: R²={shot_results['multiple_regression']['r2']:.4f}")

    t0 = time.perf_counter()
    htft_results = analyze_ht_ft_dynamics(master)
    stage_times['ht_ft_dynamics'] = time.perf_counter() - t0
    print(f"  HT-FT dynamics: {htft_results['comeback_rate']:.1%} comeback rate")

    t0 = time.perf_counter()
    consistency_results = analyze_consistency(team_matches)
    stage_times['consistency'] = time.perf_counter() - t0
    print(f"  Consistency: median Points CV={consistency_results['median_points_cv']:.3f}")

    t0 = time.perf_counter()
    disc_results = analyze_discipline(team_matches, master)
    stage_times['discipline'] = time.perf_counter() - t0
    print(f"  Discipline: Yellow_PG r={disc_results['ppg_correlations']['Yellow_PG']['r']:.3f}")

    t0 = time.perf_counter()
    robustness_results = run_robustness_checks(team_matches, master)
    stage_times['robustness'] = time.perf_counter() - t0
    print(f"  Robustness checks completed")

    team_season_disc = disc_results['team_season_disc']
    team_season_disc2 = disc_results['team_season_disc2']

    # 4b. Advanced statistical analyses (A-F)
    print("\n[5b/11] Running advanced analyses (A-F)...")
    t0 = time.perf_counter()
    finishing_results = analyze_finishing_persistence(team_matches, min_matches=20)
    profile_changes = analyze_team_profile_changes(team_season)
    ht_recovery = analyze_halftime_deficit_recovery(master, team_season, team_matches)
    volatility = analyze_volatility_streakiness(team_matches, team_season, min_matches=20)
    pooled_vs_within = analyze_pooled_vs_within_league(team_season)
    ha_changes = analyze_home_advantage_changes(master, team_season)
    stage_times['advanced_insights'] = time.perf_counter() - t0
    print(f"  Advanced analyses A-F completed")

    # 4c. New insight analyses (questions 1-6)
    if not args.skip_new:
        print("\n[5c/11] Running new insight analyses...")
        t0 = time.perf_counter()
        opp_dep = analyze_opponent_dependent_performance(team_matches, team_season, min_matches=20)
        ha_comp = analyze_home_advantage_components(master, team_matches, team_season)
        close_match = analyze_close_match_dependence(team_matches, team_season, min_matches=20)
        rest_asym = analyze_rest_asymmetry(team_matches, master, min_matches=20)
        perf_lead = analyze_performance_indicators_lead_results(team_matches, team_season, min_matches=20)
        promo_adapt = analyze_adaptation_after_promotion(team_matches, team_season, master)
        stage_times['new_insights'] = time.perf_counter() - t0
        print(f"  New insight analyses 1-6 completed")
    else:
        print("\n[5c/11] Skipping new insight analyses (--skip-new)")
        opp_dep = ha_comp = close_match = rest_asym = perf_lead = promo_adapt = {}

    # 4d. Next insight analyses (questions 7-11)
    if not args.skip_next:
        print("\n[5d/11] Running next insight analyses...")
        t0 = time.perf_counter()
        next_results = run_next_insights(master, team_matches, team_season)
        stage_times['next_insights'] = time.perf_counter() - t0
        print(f"  Next insights completed: {list(next_results.keys())}")
    else:
        print("\n[5d/11] Skipping next insight analyses (--skip-next)")
        next_results = {}

    # 4e. Fresh insight analyses (questions 12-16)
    if not args.skip_fresh:
        print("\n[5e/11] Running fresh insight analyses...")
        t0 = time.perf_counter()
        fresh_results = run_fresh_insights(master, team_matches, team_season)
        stage_times['fresh_insights'] = time.perf_counter() - t0
        print(f"  Fresh insights completed: {list(fresh_results.keys())}")
    else:
        print("\n[5e/11] Skipping fresh insight analyses (--skip-fresh)")
        fresh_results = {}

    # -----------------------------------------------------------------
    # 6. VISUALIZATION (Matplotlib; Polars->pandas at boundary)
    # -----------------------------------------------------------------
    print("\n[6/11] Creating visualizations...")
    t0 = time.perf_counter()
    half_eff = eff_results['half_season']
    fig_files = create_all_figures(
        ha_results, team_season, shot_results, half_eff, htft_results, master,
        consistency_results['season_consistency'], consistency_results['team_consistency'],
        disc_results, team_season_disc, team_season_disc2,
        output_dir=str(OUTPUT_DIRS['figures'])
    )
    print(f"  Created {len(fig_files)} core figures")

    adv_fig_files = create_advanced_figures(
        finishing_results, profile_changes, ht_recovery,
        volatility, pooled_vs_within, ha_changes,
        output_dir=str(OUTPUT_DIRS['figures'])
    )
    print(f"  Created {len(adv_fig_files)} advanced figures (A-F)")
    fig_files.extend(adv_fig_files)

    if not args.skip_new:
        new_fig_files = create_all_new_figures(
            opp_dep, ha_comp, close_match,
            rest_asym, perf_lead, promo_adapt,
            figures_dir=str(OUTPUT_DIRS['figures']),
            tables_dir=str(OUTPUT_DIRS['tables'])
        )
        print(f"  Created {len(new_fig_files)} new insight figures (1-6)")
        fig_files.extend(new_fig_files)

    if not args.skip_next:
        next_fig_files = create_all_next_figures(
            next_results,
            figures_dir=str(OUTPUT_DIRS['figures']),
            tables_dir=str(OUTPUT_DIRS['tables'])
        )
        print(f"  Created {len(next_fig_files)} next insight figures (7-11)")
        fig_files.extend(next_fig_files)

    if not args.skip_fresh:
        fresh_fig_files = create_all_fresh_figures(
            fresh_results,
            figures_dir=str(OUTPUT_DIRS['figures']),
            tables_dir=str(OUTPUT_DIRS['tables'])
        )
        print(f"  Created {len(fresh_fig_files)} fresh insight figures (12-16)")
        fig_files.extend(fresh_fig_files)

    stage_times['visualization'] = time.perf_counter() - t0

    # -----------------------------------------------------------------
    # 7. ARTIFACT GENERATION (Parquet + CSV)
    # -----------------------------------------------------------------
    print("\n[7/11] Exporting results...")
    t0 = time.perf_counter()

    # Machine-readable tables (Polars write_csv)
    ha_results['by_league'].write_csv(OUTPUT_DIRS['tables'] / 'table_home_advantage_by_league.csv')
    team_season.write_csv(OUTPUT_DIRS['tables'] / 'table_team_season_metrics.csv')
    htft_results['transitions_norm'].write_csv(OUTPUT_DIRS['tables'] / 'table_htft_transitions.csv')
    htft_results['transitions_counts'].write_csv(OUTPUT_DIRS['tables'] / 'table_htft_transitions_counts.csv')
    htft_results['team_comeback_stats'].write_csv(OUTPUT_DIRS['tables'] / 'table_team_comeback_ability.csv')
    consistency_results['season_consistency'].write_csv(OUTPUT_DIRS['tables'] / 'table_team_season_consistency.csv')
    consistency_results['team_consistency'].write_csv(OUTPUT_DIRS['tables'] / 'table_team_consistency.csv')
    team_season.select(['Div', 'Team', 'Season', 'Cluster']).write_csv(
        OUTPUT_DIRS['tables'] / 'table_team_season_clusters.csv')

    # Discipline by league (Polars)
    disc_by_league = team_season.group_by('Div').agg([
        pl.col('Yellow_PG').mean().alias('Yellow_PG'),
        pl.col('Red_PG').mean().alias('Red_PG'),
        pl.col('Fouls_PG').mean().alias('Fouls_PG'),
        pl.col('PPG').mean().alias('PPG'),
    ]).sort('Div')
    disc_by_league.write_csv(OUTPUT_DIRS['tables'] / 'table_discipline_by_league.csv')

    # Feature dictionary
    feature_dict = {
        "core_match_fields": {
            "Div": "League division code",
            "Date": "Match date (DD/MM/YYYY)",
            "HomeTeam": "Home team name",
            "AwayTeam": "Away team name",
            "FTHG": "Full-time home goals",
            "FTAG": "Full-time away goals",
            "FTR": "Full-time result (H/D/A)",
            "HTHG": "Half-time home goals",
            "HTAG": "Half-time away goals",
            "HTR": "Half-time result (H/D/A)",
            "Referee": "Referee name (missing for non-English/Scottish leagues)",
            "HS": "Home shots", "AS": "Away shots",
            "HST": "Home shots on target", "AST": "Away shots on target",
            "HF": "Home fouls committed", "AF": "Away fouls committed",
            "HC": "Home corners", "AC": "Away corners",
            "HY": "Home yellow cards", "AY": "Away yellow cards",
            "HR": "Home red cards", "AR": "Away red cards"
        },
        "engineered_team_perspective": {
            "Team": "Team name (perspective team)",
            "Opponent": "Opponent team name",
            "Venue": "Home or Away",
            "GF": "Goals scored by team", "GA": "Goals conceded by team",
            "GD": "Goal difference (GF - GA)",
            "FT_Result": "Match result from team perspective (W/D/L)",
            "HT_GF": "Half-time goals scored", "HT_GA": "Half-time goals conceded",
            "HT_GD": "Half-time goal difference",
            "HT_Result": "Half-time result from team perspective (W/D/L)",
            "Shots": "Team shots", "Shots_OT": "Team shots on target",
            "Shots_Against": "Opponent shots", "Shots_OT_Against": "Opponent shots on target",
            "Fouls": "Team fouls committed", "Fouls_Against": "Opponent fouls committed (fouls drawn)",
            "Corners": "Team corners", "Corners_Against": "Opponent corners",
            "Yellow": "Team yellow cards", "Yellow_Against": "Opponent yellow cards",
            "Red": "Team red cards", "Red_Against": "Opponent red cards",
            "Points": "Points earned (W=3, D=1, L=0)"
        },
    }
    with open(OUTPUT_DIRS['metadata'] / 'feature_dictionary.json', 'w') as f:
        json.dump(feature_dict, f, indent=2)

    # Save processed data as Parquet (canonical reusable storage)
    master.write_parquet(OUTPUT_DIRS['data'] / 'master_matches.parquet')
    team_matches.write_parquet(OUTPUT_DIRS['data'] / 'team_matches.parquet')
    team_season.write_parquet(OUTPUT_DIRS['data'] / 'team_season.parquet')
    consistency.write_parquet(OUTPUT_DIRS['data'] / 'consistency.parquet')

    # Manifest
    manifest = {
        'timestamp': datetime.now().isoformat(),
        'seed': args.seed,
        'data_dir': str(DATA_DIR),
        'output_dir': str(OUTPUT_DIRS['root']),
        'leagues': TARGET_LEAGUES,
        'seasons': SEASONS,
        'n_matches': master.height,
        'n_team_matches': team_matches.height,
        'n_team_seasons': team_season.height,
        'figures': [Path(f).name for f in fig_files],
        'tables': [
            'table_home_advantage_by_league.csv',
            'table_team_season_metrics.csv',
            'table_htft_transitions.csv',
            'table_htft_transitions_counts.csv',
            'table_team_comeback_ability.csv',
            'table_team_season_consistency.csv',
            'table_team_consistency.csv',
            'table_team_season_clusters.csv',
            'table_discipline_by_league.csv',
        ],
        'metadata': ['data_audit_summary.json', 'feature_dictionary.json', 'manifest.json'],
        'data_files': ['master_matches.parquet', 'team_matches.parquet',
                       'team_season.parquet', 'consistency.parquet'],
        'software_versions': {
            'polars': pl.__version__,
            'duckdb': duckdb.__version__,
            'numpy': np.__version__,
        },
        'stage_times_seconds': {k: round(v, 4) for k, v in stage_times.items()},
        'total_runtime_seconds': round(sum(stage_times.values()), 4),
    }
    with open(OUTPUT_DIRS['metadata'] / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    stage_times['artifact_generation'] = time.perf_counter() - t0

    # -----------------------------------------------------------------
    # 8. VALIDATION & SUMMARY
    # -----------------------------------------------------------------
    print("\n[8/11] Validation summary...")
    print("  Run tests separately: python -m pytest tests/ -v")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIRS['root']}")
    print(f"Figures: {len(fig_files)} PNG files in {OUTPUT_DIRS['figures']}")
    print(f"Tables: CSV files in {OUTPUT_DIRS['tables']}")
    print(f"Data: Parquet files in {OUTPUT_DIRS['data']}")
    print(f"Metadata: JSON files in {OUTPUT_DIRS['metadata']}")
    print("\nStage timings:")
    for k, v in stage_times.items():
        print(f"  {k}: {v:.2f}s")
    print(f"  TOTAL: {sum(stage_times.values()):.2f}s")


if __name__ == '__main__':
    main()
