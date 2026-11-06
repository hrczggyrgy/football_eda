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
from typing import Dict, List, Optional, Union

import polars as pl

COLORS = {'Home': '#2E86AB', 'Away': '#A23B72', 'Draw': '#F18F01'}
PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']


def _to_pandas(obj):
    if isinstance(obj, pl.DataFrame):
        return obj.to_pandas()
    if isinstance(obj, pl.Series):
        return obj.to_pandas()
    if isinstance(obj, dict):
        return {k: _to_pandas(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_pandas(v) for v in obj)
    return obj


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
    
    team_interactions = _to_pandas(team_interactions)
    half_interactions = _to_pandas(half_interactions)
    shot_gd = _to_pandas(shot_gd)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1a: Shrunk opponent-strength slope by team (sorted)
    ax = axes[0, 0]
    if team_interactions.shape[0] > 0:
        ti = team_interactions.sort_values('Avg_Slope_Shrunk')
        colors = ['green' if v > 0 else 'red' for v in ti['Avg_Slope_Shrunk']]
        ax.barh(range(len(ti)), ti['Avg_Slope_Shrunk'], color=colors, alpha=0.7)
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.set_yticks(range(len(ti)))
        ax.set_yticklabels(ti['Team'], fontsize=7)
        ax.set_xlabel('Avg Opponent-Strength Slope (shrunk)')
        ax.set_title('A. Opponent-Strength Interaction by Team')
    
    # 1b: Early vs late season slope persistence
    ax = axes[0, 1]
    if half_interactions.shape[0] > 0:
        hi = half_interactions.dropna(subset=['Slope'])
        early = hi[hi['Half'] == 'Early']
        late = hi[hi['Half'] == 'Late']
        merged = early.merge(late, on=['Div', 'Team', 'Season'], suffixes=('_Early', '_Late'))
        if len(merged) > 0:
            ax.scatter(merged['Slope_Early'], merged['Slope_Late'], alpha=0.6, s=40)
            mn = min(merged['Slope_Early'].min(), merged['Slope_Late'].min())
            mx = max(merged['Slope_Early'].max(), merged['Slope_Late'].max())
            ax.plot([mn, mx], [mn, mx], 'k--', alpha=0.5, label='Equality')
            ax.set_xlabel('Early-Season Slope')
            ax.set_ylabel('Late-Season Slope')
            ax.set_title('B. Early vs Late Opponent-Strength Slope')
            ax.legend(fontsize=8)
    
    # 1c: Shot-dominance vs GD interaction slopes
    ax = axes[1, 0]
    if shot_gd.shape[0] > 0:
        ax.scatter(shot_gd['ShotDom_Slope'], shot_gd['GD_Slope'], alpha=0.6, s=40, color=PALETTE[2])
        ax.axhline(y=0, color='black', linewidth=0.8, alpha=0.5)
        ax.axvline(x=0, color='black', linewidth=0.8, alpha=0.5)
        ax.set_xlabel('Shot Dominance Slope')
        ax.set_ylabel('Goal Difference Slope')
        ax.set_title('C. Shot-Dominance vs GD Interaction Slopes')
    
    # 1d: Average shrunk slope distribution
    ax = axes[1, 1]
    if team_interactions.shape[0] > 0:
        ax.hist(team_interactions['Avg_Slope_Shrunk'], bins=10, color=PALETTE[3], alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.set_xlabel('Avg Opponent-Strength Slope (shrunk)')
        ax.set_ylabel('Count')
        ax.set_title('D. Distribution of Opponent-Strength Slopes')
    
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
    
    ha = _to_pandas(ha)
    paired = _to_pandas(paired)
    league = _to_pandas(league)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 2a: Home vs Away PPG scatter
    ax = axes[0, 0]
    valid = ha['Home_PPG'].notna() & ha['Away_PPG'].notna()
    if valid.sum() > 5:
        ax.scatter(ha.loc[valid, 'Away_PPG'], ha.loc[valid, 'Home_PPG'], alpha=0.6, s=40)
        mn = min(ha.loc[valid, 'Away_PPG'].min(), ha.loc[valid, 'Home_PPG'].min())
        mx = max(ha.loc[valid, 'Away_PPG'].max(), ha.loc[valid, 'Home_PPG'].max())
        ax.plot([mn, mx], [mn, mx], 'k--', alpha=0.5, label='Equality')
        ax.set_xlabel('Away PPG')
        ax.set_ylabel('Home PPG')
        ax.set_title('A. Home vs Away PPG by Team-Season')
        ax.legend(fontsize=8)
    
    # 2b: Difference in PPG (Home - Away)
    ax = axes[0, 1]
    valid = ha['Diff_PPG'].notna()
    if valid.sum() > 5:
        ax.bar(range(valid.sum()), ha.loc[valid, 'Diff_PPG'],
               color=['green' if v > 0 else 'red' for v in ha.loc[valid, 'Diff_PPG']], alpha=0.7)
        ax.axhline(y=0, color='black', linewidth=0.8)
        ax.set_xlabel('Team-Season (sorted)')
        ax.set_ylabel('Home PPG − Away PPG')
        ax.set_title('B. Home Advantage (PPG Diff) by Team-Season')
    
    # 2c: Difference in Shots per Game
    ax = axes[1, 0]
    valid = ha['Diff_Shots_PG'].notna()
    if valid.sum() > 5:
        ax.bar(range(valid.sum()), ha.loc[valid, 'Diff_Shots_PG'], color=PALETTE[1], alpha=0.7)
        ax.axhline(y=0, color='black', linewidth=0.8)
        ax.set_xlabel('Team-Season (sorted)')
        ax.set_ylabel('Home Shots/PG − Away Shots/PG')
        ax.set_title('C. Shot-Generation Component of HA')
    
    # 2d: League average components
    ax = axes[1, 1]
    if league.shape[0] > 0:
        x = range(len(league))
        width = 0.35
        ax.bar([i - width/2 for i in x], league['Avg_Diff_PPG'], width, label='Δ PPG', alpha=0.7)
        ax.bar([i + width/2 for i in x], league['Avg_Diff_Shots_PG'], width, label='Δ Shots/PG', alpha=0.7)
        ax.axhline(y=0, color='black', linewidth=0.8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(league['Div'], fontsize=8)
        ax.set_ylabel('League Average Difference')
        ax.set_title('D. HA Components by League')
        ax.legend(fontsize=8)
    
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
    
    comp = _to_pandas(comp)
    early_late = _to_pandas(early_late)
    team_summary = _to_pandas(team_summary)
    shot_dom = _to_pandas(shot_dom)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 3a: One-Goal Win % vs PPG
    ax = axes[0, 0]
    valid = team_summary['Avg_One_Goal_Win_Pct'].notna() & team_summary['Avg_PPG'].notna()
    if valid.sum() > 5:
        ax.scatter(team_summary.loc[valid, 'Avg_PPG'], team_summary.loc[valid, 'Avg_One_Goal_Win_Pct'], alpha=0.6, s=40)
        ax.set_xlabel('Avg PPG')
        ax.set_ylabel('Avg One-Goal Win %')
        ax.set_title('A. One-Goal Win Reliance vs Team Strength')
    
    # 3b: Points from one-goal wins early vs late
    ax = axes[0, 1]
    valid = early_late['Points_From_OGW_Early'].notna() & early_late['Points_From_OGW_Late'].notna()
    if valid.sum() > 5:
        ax.scatter(early_late.loc[valid, 'Points_From_OGW_Early'], early_late.loc[valid, 'Points_From_OGW_Late'], alpha=0.6, s=40)
        mn = min(early_late.loc[valid, 'Points_From_OGW_Early'].min(), early_late.loc[valid, 'Points_From_OGW_Late'].min())
        mx = max(early_late.loc[valid, 'Points_From_OGW_Early'].max(), early_late.loc[valid, 'Points_From_OGW_Late'].max())
        ax.plot([mn, mx], [mn, mx], 'k--', alpha=0.5, label='Equality')
        ax.set_xlabel('Early-Season Points from 1-Goal Wins')
        ax.set_ylabel('Late-Season Points from 1-Goal Wins')
        ax.set_title('B. Early vs Late One-Goal Win Points')
        ax.legend(fontsize=8)
    
    # 3c: Top teams by adjusted one-goal win reliance
    ax = axes[1, 0]
    if team_summary.shape[0] > 0:
        top = team_summary.sort_values('Avg_Adj_Points_From_One_Goal_Wins', ascending=False).head(10)
        ax.barh(range(len(top)), top['Avg_Adj_Points_From_One_Goal_Wins'], color=PALETTE[2], alpha=0.7)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top['Team'], fontsize=7)
        ax.set_xlabel('Avg Adjusted Points from 1-Goal Wins')
        ax.set_title('C. Top Teams by 1-Goal Win Reliance')
        ax.invert_yaxis()
    
    # 3d: One-goal win % distribution
    ax = axes[1, 1]
    valid = team_summary['Avg_One_Goal_Win_Pct'].notna()
    if valid.sum() > 5:
        ax.hist(team_summary.loc[valid, 'Avg_One_Goal_Win_Pct'], bins=10, color=PALETTE[3], alpha=0.7, edgecolor='black')
        ax.set_xlabel('Avg One-Goal Win %')
        ax.set_ylabel('Count')
        ax.set_title('D. Distribution of One-Goal Win Reliance')
    
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
    
    heatmap = _to_pandas(heatmap)
    rest_effects = _to_pandas(rest_effects)
    shot_effects = _to_pandas(shot_effects)
    density = _to_pandas(density)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 4a: Rest difference effect on points residual
    ax = axes[0, 0]
    if rest_effects.shape[0] > 0:
        ax.scatter(rest_effects['Venue'].astype(str) + '/' + rest_effects['Strength'], rest_effects['Slope'], alpha=0.7, s=40)
        ax.axhline(y=0, color='black', linewidth=0.8)
        ax.set_xlabel('Venue / Strength')
        ax.set_ylabel('Slope of Points Residual on Rest Diff')
        ax.set_title('A. Rest Effect by Venue & Strength')
        ax.tick_params(axis='x', rotation=45)
    
    # 4b: Rest heatmap (rest diff x venue -> mean residual)
    ax = axes[0, 1]
    if heatmap.shape[0] > 0:
        hm = heatmap.pivot_table(index='Rest_Diff', columns='Venue', values='Mean_Residual')
        if not hm.empty:
            sns.heatmap(hm, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax, cbar_kws={'label': 'Mean Residual'})
            ax.set_xlabel('Venue')
            ax.set_ylabel('Rest Diff (Own − Opponent)')
            ax.set_title('B. Mean Points Residual Heatmap')
    
    # 4c: Rest density
    ax = axes[1, 0]
    if density.shape[0] > 0:
        ax.bar(range(len(density)), density['Count'], color=PALETTE[1], alpha=0.7)
        ax.set_xticks(range(len(density)))
        ax.set_xticklabels([f"{r['Own_Rest_Cat']}/{r['Opp_Rest_Cat']}" for _, r in density.iterrows()], rotation=90, fontsize=6)
        ax.set_ylabel('Match Count')
        ax.set_title('C. Rest Combination Density')
    
    # 4d: Shot dominance vs rest diff
    ax = axes[1, 1]
    if heatmap.shape[0] > 0:
        ax.scatter(heatmap['Rest_Diff'], heatmap['Mean_Shot_Dom'], alpha=0.7, s=40, color=PALETTE[2])
        ax.set_xlabel('Rest Diff')
        ax.set_ylabel('Mean Shot Dominance')
        ax.set_title('D. Shot Dominance by Rest Diff')
    
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
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 5a: Incremental R² of shot dominance over PPG
    ax = axes[0, 0]
    if incremental:
        inc_labels = [k.replace('incremental_', '') for k in incremental.keys()]
        inc_vals = [v.get('incremental_shotdom', np.nan) for v in incremental.values()]
        ax.bar(range(len(inc_labels)), inc_vals, color=PALETTE[0], alpha=0.7)
        ax.set_xticks(range(len(inc_labels)))
        ax.set_xticklabels(inc_labels, rotation=15)
        ax.set_ylabel('Incremental R² (Shot Dominance over PPG)')
        ax.set_title('A. Incremental Predictive R² by Window')
    
    # 5b: Lead indicator vs outcome correlation
    ax = axes[0, 1]
    if lead_lag:
        # collect shot dominance -> outcome PPG correlations
        rows = []
        for w, corrs in lead_lag.items():
            for k, v in corrs.items():
                if k.startswith('Ind_Shot_Dom_vs_Out_PPG'):
                    rows.append({'window': w, 'r': v['r']})
        if rows:
            import pandas as _pd
            df = _pd.DataFrame(rows)
            ax.bar(range(len(df)), df['r'], color=PALETTE[1], alpha=0.7)
            ax.set_xticks(range(len(df)))
            ax.set_xticklabels(df['window'], rotation=15)
            ax.set_ylabel('Correlation r')
            ax.set_title('B. Shot Dominance → Future PPG Correlation')
    
    # 5c: Horizon sensitivity
    ax = axes[1, 0]
    if horizon:
        h_labels = list(horizon.keys())
        h_vals = [v['r'] for v in horizon.values()]
        ax.bar(range(len(h_labels)), h_vals, color=PALETTE[2], alpha=0.7)
        ax.set_xticks(range(len(h_labels)))
        ax.set_xticklabels(h_labels, rotation=15)
        ax.set_ylabel('Correlation r')
        ax.set_title('C. Shot Dominance → Future PPG by Horizon')
    
    # 5d: Incremental R² bar for shot dominance vs SOT dominance
    ax = axes[1, 1]
    if incremental:
        labels = []
        vals = []
        for k, v in incremental.items():
            labels.append(k.replace('incremental_', ''))
            vals.append(v.get('incremental_shotdom', np.nan))
        if vals:
            x = range(len(labels))
            ax.bar([i - 0.2 for i in x], vals, width=0.4, label='Shot Dominance', color=PALETTE[0], alpha=0.7)
            sot_vals = [v.get('incremental_sotdom', np.nan) for v in incremental.values()]
            ax.bar([i + 0.2 for i in x], sot_vals, width=0.4, label='SOT Dominance', color=PALETTE[1], alpha=0.7)
            ax.set_xticks(list(x))
            ax.set_xticklabels(labels, rotation=15)
            ax.set_ylabel('Incremental R²')
            ax.set_title('D. Incremental R²: Shot vs SOT Dominance')
            ax.legend(fontsize=8)
    
    plt.tight_layout()
    filepath = f'{figures_dir}/fig5_performance_indicators_lead.png'
    save_figure(fig, filepath)
    files.append(filepath)
    
    # Save chart data
    for w_name in windows:
        if w_name in results:
            _to_pandas(results[w_name]).to_csv(f'{tables_dir}/table5_{w_name}.csv', index=False)
    
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
    
    promo_traj = _to_pandas(promo_traj)
    promo_curves = _to_pandas(promo_curves)
    promo_table = _to_pandas(promo_table)
    releg_traj = _to_pandas(releg_traj)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 6a: Promoted team PPG trajectory by match number
    ax = axes[0, 0]
    if promo_traj.shape[0] > 0:
        ax.plot(promo_traj['Match_Num'], promo_traj['PPG'], 'o-', color=PALETTE[0], label='Promoted', alpha=0.8)
        # comparison average if available
        comp_rows = []
        for key, df in comparison.items():
            df = _to_pandas(df)
            comp_rows.append(df[['Match_Num', 'PPG']].rename(columns={'PPG': 'comp_PPG'}))
        if comp_rows:
            import pandas as _pd
            comp_all = _pd.concat(comp_rows).groupby('Match_Num')['comp_PPG'].mean().reset_index()
            ax.plot(comp_all['Match_Num'], comp_all['comp_PPG'], 's--', color='gray', label='Other Teams', alpha=0.7)
        ax.set_xlabel('Match Number')
        ax.set_ylabel('PPG')
        ax.set_title('A. Promoted Team PPG Trajectory')
        ax.legend(fontsize=8)
    
    # 6b: Promoted vs relegated trajectory
    ax = axes[0, 1]
    if promo_traj.shape[0] > 0 and releg_traj.shape[0] > 0:
        ax.plot(promo_traj['Match_Num'], promo_traj['PPG'], 'o-', color=PALETTE[0], label='Promoted', alpha=0.8)
        ax.plot(releg_traj['Match_Num'], releg_traj['PPG'], 's-', color=PALETTE[1], label='Relegated', alpha=0.8)
        ax.set_xlabel('Match Number')
        ax.set_ylabel('PPG')
        ax.set_title('B. Promoted vs Relegated Trajectory')
        ax.legend(fontsize=8)
    
    # 6c: PPG change (Promoted - Prev)
    ax = axes[1, 0]
    if promo_table.shape[0] > 0:
        ax.bar(range(len(promo_table)), promo_table['PPG_Change'], color=['green' if v > 0 else 'red' for v in promo_table['PPG_Change']], alpha=0.7)
        ax.axhline(y=0, color='black', linewidth=0.8)
        ax.set_xticks(range(len(promo_table)))
        ax.set_xticklabels(promo_table['Team'], rotation=90, fontsize=7)
        ax.set_ylabel('PPG Change (New − Prev Division)')
        ax.set_title('C. PPG Change on Promotion')
    
    # 6d: Prev PPG vs PPG slope
    ax = axes[1, 1]
    if promo_curves.shape[0] > 0:
        valid = promo_curves['Prev_PPG'].notna() & promo_curves['PPG_Slope'].notna()
        if valid.sum() > 2:
            ax.scatter(promo_curves.loc[valid, 'Prev_PPG'], promo_curves.loc[valid, 'PPG_Slope'], alpha=0.7, s=40)
            ax.set_xlabel('Prev Season PPG')
            ax.set_ylabel('PPG Slope (by Match)')
            ax.set_title('D. Baseline Strength vs Adaptation Slope')
    
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
    """Create all new insight figures.

    Polars inputs are converted to pandas at the plotting boundary.
    """
    results_1 = _to_pandas(results_1)
    results_2 = _to_pandas(results_2)
    results_3 = _to_pandas(results_3)
    results_4 = _to_pandas(results_4)
    results_5 = _to_pandas(results_5)
    results_6 = _to_pandas(results_6)

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