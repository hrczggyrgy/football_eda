"""
New insight plots module.
Creates publication-quality figures for the 6 new research questions.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Optional

COLORS = {'Home': '#2E86AB', 'Away': '#A23B72', 'Draw': '#F18F01'}
PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

def save_figure(fig, filepath: str, dpi: int = 150):
    """Save figure with consistent settings."""
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_opponent_dependent_performance(results: Dict, figures_dir: str = '.', tables_dir: str = '.') -> List[str]:
    """
    Figure 1: Opponent-dependent performance.
    ...
    """
    files = []
    
    team_interactions = results['team_interactions']
    half_interactions = results['half_season_interactions']
    shot_gd = results['shot_gd_interactions']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # ... (rest of plotting code unchanged)
    
    plt.tight_layout()
    filepath = f'{figures_dir}/fig1_opponent_dependent_performance.png'
    save_figure(fig, filepath)
    files.append(filepath)
    
    # Save chart data
    team_interactions.to_csv(f'{tables_dir}/table1_opponent_dependent_performance.csv', index=False)
    
    return files


def plot_home_advantage_components(results: Dict, figures_dir: str = '.', tables_dir: str = '.') -> List[str]:
    """
    Figure 2: Components of home advantage.
    ...
    """
    files = []
    
    ha = results['team_season_ha']
    paired = results['paired_fixture_diffs']
    ha_decomp = results['ha_decomposition']
    persistence = results['ha_component_persistence']
    league = results['league_ha_components']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # ... (rest of plotting code unchanged)
    
    plt.tight_layout()
    filepath = f'{figures_dir}/fig2_home_advantage_components.png'
    save_figure(fig, filepath)
    files.append(filepath)
    
    # Save chart data
    ha.to_csv(f'{tables_dir}/table2a_team_season_ha_components.csv', index=False)
    paired.to_csv(f'{tables_dir}/table2b_paired_fixture_diffs.csv', index=False)
    league.to_csv(f'{tables_dir}/table2c_league_ha_components.csv', index=False)
    
    return files


def plot_close_match_dependence(results: Dict, figures_dir: str = '.', tables_dir: str = '.') -> List[str]:
    """
    Figure 3: Close-match dependence.
    ...
    """
    files = []
    
    comp = results['season_outcome_composition']
    early_late = results['early_late_comparison']
    team_summary = results['team_close_match_summary']
    shot_dom = results['shot_dominance_early_late']
    persistence = results.get('early_late_persistence', {})
    shot_persist = results.get('shot_dominance_persistence', {})
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # ... (rest of plotting code unchanged)
    
    plt.tight_layout()
    filepath = f'{figures_dir}/fig3_close_match_dependence.png'
    save_figure(fig, filepath)
    files.append(filepath)
    
    # Save chart data
    comp.to_csv(f'{tables_dir}/table3a_season_outcome_composition.csv', index=False)
    early_late.to_csv(f'{tables_dir}/table3b_early_late_comparison.csv', index=False)
    team_summary.to_csv(f'{tables_dir}/table3c_team_close_match_summary.csv', index=False)
    shot_dom.to_csv(f'{tables_dir}/table3d_shot_dominance_early_late.csv', index=False)
    
    return files


def plot_rest_asymmetry(results: Dict, figures_dir: str = '.', tables_dir: str = '.') -> List[str]:
    """
    Figure 4: Rest asymmetry.
    ...
    """
    files = []
    
    heatmap = results['rest_heatmap']
    rest_effects = results['rest_effects_by_venue_strength']
    shot_effects = results['rest_shot_effects']
    density = results['rest_density']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # ... (rest of plotting code unchanged)
    
    plt.tight_layout()
    filepath = f'{figures_dir}/fig4_rest_asymmetry.png'
    save_figure(fig, filepath)
    files.append(filepath)
    
    # Save chart data
    heatmap.to_csv(f'{tables_dir}/table4a_rest_heatmap.csv', index=False)
    rest_effects.to_csv(f'{tables_dir}/table4b_rest_effects.csv', index=False)
    shot_effects.to_csv(f'{tables_dir}/table4c_shot_effects.csv', index=False)
    density.to_csv(f'{tables_dir}/table4d_rest_density.csv', index=False)
    
    return files


def plot_performance_indicators_lead(results: Dict, figures_dir: str = '.', tables_dir: str = '.') -> List[str]:
    """
    Figure 5: Performance indicators leading results.
    ...
    """
    files = []
    
    windows = ['window_10_10', 'window_15_15', 'window_half_half']
    lead_lag = results['lead_lag_correlations']
    incremental = {k: v for k, v in results.items() if k.startswith('incremental_')}
    horizon = results.get('horizon_sensitivity', {})
    adjusted = {k: v for k, v in results.items() if k.startswith('adjusted_')}
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # ... (rest of plotting code unchanged)
    
    plt.tight_layout()
    filepath = f'{figures_dir}/fig5_performance_indicators_lead.png'
    save_figure(fig, filepath)
    files.append(filepath)
    
    # Save chart data
    for w_name in windows:
        if w_name in results:
            results[w_name].to_csv(f'{tables_dir}/table5_{w_name}.csv', index=False)
    
    return files


def plot_adaptation_after_promotion(results: Dict, figures_dir: str = '.', tables_dir: str = '.') -> List[str]:
    """
    Figure 6: Adaptation after promotion.
    ...
    """
    files = []
    
    promo_traj = results.get('promoted_trajectory', pd.DataFrame())
    promo_curves = results.get('promoted_team_curves', pd.DataFrame())
    promo_table = results.get('promotion_transition_table', pd.DataFrame())
    releg_traj = results.get('relegated_trajectory', pd.DataFrame())
    comparison = results.get('comparison_trajectories', {})
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # ... (rest of plotting code unchanged)
    
    plt.tight_layout()
    filepath = f'{figures_dir}/fig6_adaptation_after_promotion.png'
    save_figure(fig, filepath)
    files.append(filepath)
    
    # Save chart data
    promo_traj.to_csv(f'{tables_dir}/table6a_promoted_trajectory.csv', index=False)
    if len(promo_curves) > 0:
        promo_curves.to_csv(f'{tables_dir}/table6b_promoted_team_curves.csv', index=False)
    promo_table.to_csv(f'{tables_dir}/table6c_promotion_transition_table.csv', index=False)
    if len(releg_traj) > 0:
        releg_traj.to_csv(f'{tables_dir}/table6d_relegated_trajectory.csv', index=False)
    
    return files


def create_all_new_figures(results_1: Dict, results_2: Dict, results_3: Dict,
                           results_4: Dict, results_5: Dict, results_6: Dict,
                           figures_dir: str = '.', tables_dir: str = None) -> List[str]:
    """Create all new insight figures."""
    if tables_dir is None:
        # Default to tables subdirectory alongside figures
        from pathlib import Path
        tables_dir = str(Path(figures_dir).parent / 'tables')
        Path(tables_dir).mkdir(parents=True, exist_ok=True)
    
    all_files = []
    
    all_files.extend(plot_opponent_dependent_performance(results_1, figures_dir, tables_dir))
    all_files.extend(plot_home_advantage_components(results_2, figures_dir, tables_dir))
    all_files.extend(plot_close_match_dependence(results_3, figures_dir, tables_dir))
    all_files.extend(plot_rest_asymmetry(results_4, figures_dir, tables_dir))
    all_files.extend(plot_performance_indicators_lead(results_5, figures_dir, tables_dir))
    all_files.extend(plot_adaptation_after_promotion(results_6, figures_dir, tables_dir))
    
    return all_files