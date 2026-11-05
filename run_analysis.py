#!/usr/bin/env python3
"""
Main analysis pipeline for Football Analytics Study.
Orchestrates data loading, feature engineering, statistical analysis, and visualization.
"""

import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data.load_data import (
    build_master_dataset, build_team_match_dataset, audit_data_quality,
    CORE_COLS, FULL_DATA_LEAGUES, PARTIAL_DATA_LEAGUES
)
from features.engineer import (
    add_efficiency_ratios, add_rolling_features, add_opponent_strength,
    compute_team_season_metrics, compute_consistency_metrics,
    add_season_order, add_next_season_metrics, cluster_team_profiles
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
from visualization.plots import create_all_figures
from visualization.insight_plots import create_advanced_figures


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
    return parser.parse_args()


def setup_output_dirs(output_dir: Path) -> Dict[str, Path]:
    """Create output subdirectories and return their paths."""
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


def main():
    args = parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    
    # Configuration
    DATA_DIR = Path(args.data_dir)
    OUTPUT_DIRS = setup_output_dirs(Path(args.output_dir))
    
    TARGET_LEAGUES = args.leagues
    SEASONS = args.seasons
    
    print("="*60)
    print("FOOTBALL ANALYTICS STUDY - PIPELINE")
    print("="*60)
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIRS['root']}")
    print(f"Leagues: {TARGET_LEAGUES}")
    print(f"Seasons: {SEASONS}")
    print(f"Random seed: {args.seed}")
    
    # -----------------------------------------------------------------
    # 1. DATA LOADING
    # -----------------------------------------------------------------
    print("\n[1/6] Loading data...")
    master = build_master_dataset(DATA_DIR, TARGET_LEAGUES, SEASONS, usecols=CORE_COLS)
    print(f"  Master matches: {len(master)}")
    print(f"  Leagues: {master['Div'].unique()}")
    print(f"  Seasons: {sorted(master['Season'].unique())}")
    print(f"  Date range: {master['Date'].min()} to {master['Date'].max()}")
    
    team_matches = build_team_match_dataset(master)
    print(f"  Team-match rows: {len(team_matches)}")
    print(f"  Unique teams: {team_matches['Team'].nunique()}")
    
    # Data quality audit
    audit = audit_data_quality(master, team_matches)
    print(f"  Data quality issues: {sum(1 for v in audit.values() if v)} categories checked")
    
    # Save audit
    with open(OUTPUT_DIRS['metadata'] / 'data_audit_summary.json', 'w') as f:
        def convert(obj):
            if isinstance(obj, (np.integer, np.int64)): return int(obj)
            if isinstance(obj, (np.floating, np.float64)): return float(obj)
            if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list): return [convert(v) for v in obj]
            return obj
        json.dump(convert(audit), f, indent=2)
    
    # -----------------------------------------------------------------
    # 2. FEATURE ENGINEERING
    # -----------------------------------------------------------------
    print("\n[2/6] Engineering features...")
    team_matches = add_efficiency_ratios(team_matches)
    team_matches = add_rolling_features(team_matches, window=5, min_periods=3)
    team_matches = add_opponent_strength(team_matches)
    
    team_season = compute_team_season_metrics(team_matches, min_matches=20)
    print(f"  Team-seasons (20+ matches): {len(team_season)}")
    
    consistency = compute_consistency_metrics(team_matches, min_matches=10)
    print(f"  Consistency records: {len(consistency)}")
    
    # -----------------------------------------------------------------
    # 3. ADVANCED FEATURES
    # -----------------------------------------------------------------
    print("\n[3/6] Computing advanced features...")
    team_season = add_season_order(team_season)
    team_season = add_next_season_metrics(team_season)
    
    # Clustering
    profile_features = ['GF_PG', 'GA_PG', 'Shots_PG', 'Shots_Against_PG',
                        'Shot_Accuracy', 'Opp_Shot_Accuracy',
                        'Goals_Per_Shot', 'GA_Per_Opp_Shot']
    team_season = cluster_team_profiles(team_season, profile_features, n_clusters=5)
    print(f"  Clusters assigned: {team_season['Cluster'].notna().sum()} team-seasons")
    
    # -----------------------------------------------------------------
    # 4. STATISTICAL ANALYSIS
    # -----------------------------------------------------------------
    print("\n[4/6] Running statistical analysis...")
    
    ha_results = analyze_home_advantage(master)
    print(f"  Home advantage: {len(ha_results['by_league'])} leagues analyzed")
    
    eff_results = analyze_efficiency_persistence(team_matches, min_matches=20)
    print(f"  Efficiency persistence: half-season & season-to-season")
    
    shot_results = analyze_shot_decomposition(team_season)
    print(f"  Shot decomposition: R²={shot_results['multiple_regression']['r2']:.4f}")
    
    htft_results = analyze_ht_ft_dynamics(master)
    print(f"  HT-FT dynamics: {htft_results['comeback_rate']:.1%} comeback rate")
    
    consistency_results = analyze_consistency(team_matches)
    print(f"  Consistency: median Points CV={consistency_results['median_points_cv']:.3f}")
    
    disc_results = analyze_discipline(team_matches, master)
    print(f"  Discipline: Yellow_PG r={disc_results['ppg_correlations']['Yellow_PG']['r']:.3f}")
    
    robustness_results = run_robustness_checks(team_matches, master)
    print(f"  Robustness checks completed")
    
    # Extract discipline dataframes for plotting
    team_season_disc = disc_results['team_season_disc']
    team_season_disc2 = disc_results['team_season_disc2']
    
    # -----------------------------------------------------------------
    # 4b. ADVANCED STATISTICAL ANALYSES (Priority A-F)
    # -----------------------------------------------------------------
    print("\n[4b/6] Running advanced analyses...")
    
    # Priority A: Finishing persistence
    finishing_results = analyze_finishing_persistence(team_matches, min_matches=20)
    print(f"  Finishing persistence: early vs late season")
    
    # Priority B: Team profile changes
    profile_changes = analyze_team_profile_changes(team_season)
    print(f"  Team profile changes: consecutive season trajectories")
    
    # Priority C: Halftime deficit recovery
    ht_recovery = analyze_halftime_deficit_recovery(master, team_season)
    print(f"  HT deficit recovery: conditional on deficit, venue, strength")
    
    # Priority D: Volatility and streakiness
    volatility = analyze_volatility_streakiness(team_matches, team_season, min_matches=20)
    print(f"  Volatility & streakiness: residual dispersion + runs test")
    
    # Priority E: Pooled vs within-league
    pooled_vs_within = analyze_pooled_vs_within_league(team_season)
    print(f"  Pooled vs within-league: correlation heterogeneity")
    
    # Priority F: Home advantage changes
    ha_changes = analyze_home_advantage_changes(master, team_season)
    print(f"  HA changes: league trends, balanced sample, COVID comparison")
    
    # -----------------------------------------------------------------
    # 5. VISUALIZATION
    # -----------------------------------------------------------------
    print("\n[5/6] Creating visualizations...")
    half_eff = eff_results['half_season']
    fig_files = create_all_figures(
        ha_results, team_season, shot_results, half_eff, htft_results, master,
        consistency_results['season_consistency'], consistency_results['team_consistency'],
        disc_results, team_season_disc, team_season_disc2,
        output_dir=str(OUTPUT_DIRS['figures'])
    )
    print(f"  Created {len(fig_files)} core figures")
    
    # Advanced figures
    adv_fig_files = create_advanced_figures(
        finishing_results, profile_changes, ht_recovery,
        volatility, pooled_vs_within, ha_changes,
        output_dir=str(OUTPUT_DIRS['figures'])
    )
    print(f"  Created {len(adv_fig_files)} advanced figures")
    fig_files.extend(adv_fig_files)
    
    # -----------------------------------------------------------------
    # 6. EXPORT RESULTS
    # -----------------------------------------------------------------
    print("\n[6/6] Exporting results...")
    
    # Machine-readable tables
    ha_results['by_league'].reset_index().to_csv(OUTPUT_DIRS['tables'] / 'table_home_advantage_by_league.csv', index=False)
    team_season.to_csv(OUTPUT_DIRS['tables'] / 'table_team_season_metrics.csv', index=False)
    htft_results['transitions_norm'].reset_index().to_csv(OUTPUT_DIRS['tables'] / 'table_htft_transitions.csv', index=False)
    htft_results['transitions_counts'].reset_index().to_csv(OUTPUT_DIRS['tables'] / 'table_htft_transitions_counts.csv', index=False)
    htft_results['team_comeback_stats'].to_csv(OUTPUT_DIRS['tables'] / 'table_team_comeback_ability.csv', index=False)
    consistency_results['season_consistency'].to_csv(OUTPUT_DIRS['tables'] / 'table_team_season_consistency.csv', index=False)
    consistency_results['team_consistency'].to_csv(OUTPUT_DIRS['tables'] / 'table_team_consistency.csv', index=False)
    team_season[['Div', 'Team', 'Season', 'Cluster']].to_csv(OUTPUT_DIRS['tables'] / 'table_team_season_clusters.csv', index=False)
    
    # Discipline by league
    disc_by_league = team_season.groupby('Div').agg(
        Yellow_PG=('Yellow_PG', 'mean'),
        Red_PG=('Red_PG', 'mean'),
        Fouls_PG=('Fouls_PG', 'mean'),
        PPG=('PPG', 'mean')
    ).reset_index()
    disc_by_league.to_csv(OUTPUT_DIRS['tables'] / 'table_discipline_by_league.csv', index=False)
    
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
        "efficiency_ratios": {
            "Shot_Accuracy": "Shots_OT / Shots (team)",
            "Goals_Per_Shot": "GF / Shots (team)",
            "Goals_Per_Shot_OT": "GF / Shots_OT (team)",
            "Shot_Dominance": "Shots - Shots_Against",
            "SOT_Dominance": "Shots_OT - Shots_OT_Against",
            "Opp_Shot_Accuracy": "Shots_OT_Against / Shots_Against",
            "GA_Per_Opp_Shot": "GA / Shots_Against",
            "GA_Per_Opp_Shot_OT": "GA / Shots_OT_Against"
        },
        "rolling_features": {
            "X_L5": "5-match rolling mean of X (lagged by 1 match)",
            "X_L5_std": "5-match rolling std of X (lagged by 1 match)",
            "X_Cum": "Cumulative expanding mean of X (lagged by 1 match)"
        },
        "temporal_features": {
            "Match_Num": "Match number within team-season",
            "Rest_Days": "Days since previous match (7 for season opener)"
        },
        "season_aggregate": {
            "PPG": "Points per game",
            "GF_PG": "Goals for per game",
            "GA_PG": "Goals against per game",
            "GD_PG": "Goal difference per game",
            "Shots_PG": "Shots per game",
            "Shots_OT_PG": "Shots on target per game",
            "Shot_Accuracy": "Season shot accuracy",
            "Goals_Per_Shot": "Season goals per shot",
            "Goals_Per_Shot_OT": "Season goals per shot on target",
            "GA_Per_Opp_Shot": "Season goals against per opponent shot",
            "Yellow_PG": "Yellow cards per game",
            "Red_PG": "Red cards per game",
            "Fouls_PG": "Fouls per game",
            "Points_CV": "Coefficient of variation of match points",
            "GF_CV": "Coefficient of variation of match goals for",
            "GA_CV": "Coefficient of variation of match goals against"
        }
    }
    
    with open(OUTPUT_DIRS['metadata'] / 'feature_dictionary.json', 'w') as f:
        json.dump(feature_dict, f, indent=2)
    
    # Save processed data
    master.to_parquet(OUTPUT_DIRS['data'] / 'master_matches.parquet', index=False)
    team_matches.to_parquet(OUTPUT_DIRS['data'] / 'team_matches.parquet', index=False)
    
    # Generate manifest
    manifest = {
        'timestamp': datetime.now().isoformat(),
        'seed': args.seed,
        'data_dir': str(DATA_DIR),
        'output_dir': str(OUTPUT_DIRS['root']),
        'leagues': TARGET_LEAGUES,
        'seasons': SEASONS,
        'n_matches': len(master),
        'n_team_matches': len(team_matches),
        'n_team_seasons': len(team_season),
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
            # Advanced analysis tables
            'tableA_finishing_persistence_data.csv',
            'tableB_within_div_changes.csv',
            'tableB_promo_releg_changes.csv',
            'tableC_trailing_recovery.csv',
            'tableC_trailing_recovery_strength.csv',
            'tableD_simulation_benchmark.csv',
            'tableD_permutation_streakiness.csv',
            'tableE_correlation_comparison.csv',
            'tableF_league_season_ha.csv',
            'tableF_balanced_sample_ha.csv',
        ],
        'metadata': ['data_audit_summary.json', 'feature_dictionary.json', 'manifest.json'],
        'data_files': ['master_matches.parquet', 'team_matches.parquet'],
        'software_versions': {
            'pandas': pd.__version__,
            'numpy': np.__version__,
        }
    }
    with open(OUTPUT_DIRS['metadata'] / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Output directory: {OUTPUT_DIRS['root']}")
    print(f"Figures: {len(fig_files)} PNG files in {OUTPUT_DIRS['figures']}")
    print(f"Tables: 9 CSV files in {OUTPUT_DIRS['tables']}")
    print(f"Data: 2 Parquet files in {OUTPUT_DIRS['data']}")
    print(f"Metadata: 3 JSON files in {OUTPUT_DIRS['metadata']}")


if __name__ == '__main__':
    main()