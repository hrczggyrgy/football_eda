# Next Insights Visualization Module
# Publication-quality figures for the 5 new research questions

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
LEAGUE_COLORS = {'E0': '#1f77b4', 'E1': '#ff7f0e', 'E2': '#2ca02c', 'E3': '#d62728',
                 'SP1': '#9467bd', 'D1': '#8c564b', 'I1': '#e377c2', 'F1': '#7f7f7f'}

def save_figure(fig, filepath: str, dpi: int = 150):
    """Save figure with consistent settings."""
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_competitive_balance(results: Dict, figures_dir: str = '.', tables_dir: str = '.') -> List[str]:
    """
    Figure 1: Competitive Balance Evolution.
    
    Panel A: HHI_Norm trends by league (small multiples)
    Panel B: Gini coefficient trends by league
    Panel C: Top4-Bot4 Gap trends by league
    Panel D: Top4 Share trends by league
    """
    files = []
    
    balance_df = results['league_season_balance']
    league_trends = results['league_trends']
    bootstrap = results.get('bootstrap_trends', {})
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['HHI_Norm', 'Gini', 'Gap', 'Top4_Share']
    titles = ['A. Normalized HHI (Concentration)', 'B. Gini Coefficient (Inequality)', 
              'C. Top-4 vs Bottom-4 PPG Gap', 'D. Top-4 Share of Total Points']
    ylabels = ['HHI (Normalized)', 'Gini', 'PPG Gap', 'Top-4 Points Share']
    
    for ax, metric, title, ylabel in zip(axes.flat, metrics, titles, ylabels):
        for div in sorted(balance_df['Div'].unique()):
            lg = balance_df[balance_df['Div'] == div].sort_values('Season')
            ax.plot(lg['Season'], lg[metric], 'o-', color=LEAGUE_COLORS.get(div, 'gray'), 
                   label=div, alpha=0.8, linewidth=1.5, markersize=5)
        
        # Add overall trend line
        overall = results['overall_trend'].get(metric, {})
        if overall:
            slope = overall['slope_per_season']
            intercept = np.mean(balance_df[metric].dropna()) - slope * np.mean(np.arange(len(balance_df['Season'].unique())))
            x_trend = np.arange(len(balance_df['Season'].unique()))
            ax.plot(sorted(balance_df['Season'].unique()), intercept + slope * x_trend, 
                   'k--', alpha=0.5, linewidth=1, label=f'Overall (slope={slope:.4f})')
        
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Season')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(fontsize=7, ncol=2, loc='upper left')
        
        # Annotate significant league trends
        for div in balance_df['Div'].unique():
            key = f'{div}_{metric}'
            if key in league_trends and league_trends[key]['significant']:
                lg = balance_df[balance_df['Div'] == div].sort_values('Season')
                last_val = lg[metric].iloc[-1]
                ax.annotate('*', xy=(lg['Season'].iloc[-1], last_val), 
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=10, fontweight='bold', color=LEAGUE_COLORS.get(div, 'gray'))
    
    plt.tight_layout()
    filepath = f'{figures_dir}/fig1_competitive_balance.png'
    save_figure(fig, filepath)
    files.append(filepath)
    
    # Save chart data
    balance_df.to_csv(f'{tables_dir}/table1_competitive_balance.csv', index=False)
    
    return files


def plot_league_style_fingerprints(results: Dict, figures_dir: str = '.', tables_dir: str = '.') -> List[str]:
    """
    Figure 2: League Style Fingerprints.
    
    Panel A: Shots_PG vs Shot_Accuracy by league (trajectories over seasons)
    Panel B: Corners_PG vs Goals_Per_Corner by league
    Panel C: Fouls_PG vs Shots_Per_Goal by league
    Panel D: Style distance heatmap (league-season similarity)
    """
    files = []
    
    style_df = results['league_season_style']
    dist_matrix = results.get('style_distance_matrix', pd.DataFrame())
    convergence = results.get('convergence', pd.DataFrame())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel A: Shots vs Accuracy trajectories
    ax = axes[0, 0]
    for div in sorted(style_df['Div'].unique()):
        lg = style_df[style_df['Div'] == div].sort_values('Season')
        ax.plot(lg['Shots_PG'], lg['Shot_Accuracy'], 'o-', color=LEAGUE_COLORS.get(div, 'gray'),
               label=div, alpha=0.8, linewidth=1.5, markersize=5)
        # Annotate start and end
        ax.annotate(lg['Season'].iloc[0], (lg['Shots_PG'].iloc[0], lg['Shot_Accuracy'].iloc[0]),
                   fontsize=6, alpha=0.7)
        ax.annotate(lg['Season'].iloc[-1], (lg['Shots_PG'].iloc[-1], lg['Shot_Accuracy'].iloc[-1]),
                   fontsize=6, alpha=0.7, fontweight='bold')
    ax.set_xlabel('Shots per Game')
    ax.set_ylabel('Shot Accuracy')
    ax.set_title('A. Shots vs Accuracy Trajectories by League', fontsize=11, fontweight='bold')
    ax.legend(fontsize=7, ncol=2)
    
    # Panel B: Corners vs Goals_Per_Corner
    ax = axes[0, 1]
    for div in sorted(style_df['Div'].unique()):
        lg = style_df[style_df['Div'] == div].sort_values('Season')
        ax.plot(lg['Corners_PG'], lg['Goals_Per_Corner'], 's-', color=LEAGUE_COLORS.get(div, 'gray'),
               label=div, alpha=0.8, linewidth=1.5, markersize=5)
    ax.set_xlabel('Corners per Game')
    ax.set_ylabel('Goals per Corner')
    ax.set_title('B. Corners vs Conversion Trajectories', fontsize=11, fontweight='bold')
    ax.legend(fontsize=7, ncol=2)
    
    # Panel C: Fouls vs Shots_Per_Goal
    ax = axes[1, 0]
    for div in sorted(style_df['Div'].unique()):
        lg = style_df[style_df['Div'] == div].sort_values('Season')
        ax.plot(lg['Fouls_PG'], lg['Shots_Per_Goal'], '^-', color=LEAGUE_COLORS.get(div, 'gray'),
               label=div, alpha=0.8, linewidth=1.5, markersize=5)
    ax.set_xlabel('Fouls per Game')
    ax.set_ylabel('Shots per Goal')
    ax.set_title('C. Fouls vs Shot Efficiency Trajectories', fontsize=11, fontweight='bold')
    ax.legend(fontsize=7, ncol=2)
    
    # Panel D: Style distance heatmap
    ax = axes[1, 1]
    if len(dist_matrix) > 0:
        # Order by league then season
        ordered_idx = sorted(dist_matrix.index, key=lambda x: (x[0], x[1]))
        dist_ordered = dist_matrix.loc[ordered_idx, ordered_idx]
        
        im = ax.imshow(dist_ordered.values, aspect='auto', cmap='RdYlBu_r')
        ax.set_xticks(range(len(ordered_idx)))
        ax.set_xticklabels([f'{i[0]}{i[1][-2:]}' for i in ordered_idx], rotation=90, fontsize=6)
        ax.set_yticks(range(len(ordered_idx)))
        ax.set_yticklabels([f'{i[0]}{i[1][-2:]}' for i in ordered_idx], fontsize=6)
        plt.colorbar(im, ax=ax, label='Euclidean Distance (z-scored style)')
        ax.set_title('D. League-Season Style Distance Matrix', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    filepath = f'{figures_dir}/fig2_league_style_fingerprints.png'
    save_figure(fig, filepath)
    files.append(filepath)
    
    # Convergence plot
    if len(convergence) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(convergence['Season'], convergence['Avg_Distance'], 'o-', color='black', linewidth=2)
        ax.set_xlabel('Season')
        ax.set_ylabel('Average Pairwise Style Distance')
        ax.set_title('League Style Convergence/Divergence Over Time', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        filepath = f'{figures_dir}/fig2b_style_convergence.png'
        save_figure(fig, filepath)
        files.append(filepath)
    
    # Save chart data
    style_df.to_csv(f'{tables_dir}/table2_league_style_fingerprints.csv', index=False)
    if len(dist_matrix) > 0:
        dist_matrix.to_csv(f'{tables_dir}/table2b_style_distance_matrix.csv')
    if len(convergence) > 0:
        convergence.to_csv(f'{tables_dir}/table2c_style_convergence.csv', index=False)
    
    return files


def plot_corner_efficiency(results: Dict, figures_dir: str = '.', tables_dir: str = '.') -> List[str]:
    """
    Figure 3: Corner Efficiency & Set-Piece Profiles.
    
    Panel A: Corners_PG vs PPG with fit line
    Panel B: Goals_Per_Corner vs PPG with fit line
    Panel C: Half-season persistence (Goals_Per_Corner)
    Panel D: Set-piece cluster profiles
    """
    files = []
    
    # We need the raw data for scatter plots - but we only have summary stats
    # The scatter plots need team-season data. We'll create representative plots from summary.
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel A: Correlation summary bar chart
    ax = axes[0, 0]
    corrs = results['ppg_correlations']
    metrics = list(corrs.keys())
    r_vals = [corrs[m]['r'] for m in metrics]
    p_vals = [corrs[m]['p'] for m in metrics]
    colors_bar = ['green' if p < 0.05 else 'gray' for p in p_vals]
    ax.barh(range(len(metrics)), r_vals, color=colors_bar, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([m.replace('_', ' ') for m in metrics])
    ax.set_xlabel('Correlation with PPG (r)')
    ax.set_title('A. Corner Metrics vs PPG Correlation', fontsize=11, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.5)
    for i, (r, p) in enumerate(zip(r_vals, p_vals)):
        ax.text(r + 0.02 if r >= 0 else r - 0.02, i, f'r={r:.3f} (p={p:.4f})', 
               va='center', ha='left' if r >= 0 else 'right', fontsize=9)
    
    # Panel B: Half-season persistence
    ax = axes[0, 1]
    persist = results['half_season_persistence']
    metrics_p = list(persist.keys())
    r_persist = [persist[m]['r'] for m in metrics_p]
    colors_p = [PALETTE[i % len(PALETTE)] for i in range(len(metrics_p))]
    ax.barh(range(len(metrics_p)), r_persist, color=colors_p, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(metrics_p)))
    ax.set_yticklabels([m.replace('_', ' ') for m in metrics_p])
    ax.set_xlabel('Half-Season Persistence (r)')
    ax.set_title('B. Half-Season Persistence of Corner Metrics', fontsize=11, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.5)
    for i, r in enumerate(r_persist):
        ax.text(r + 0.02, i, f'r={r:.3f}', va='center', fontsize=9)
    
    # Panel C: Season-to-season persistence
    ax = axes[1, 0]
    persist_s = results['season_to_season_persistence']
    metrics_s = list(persist_s.keys())
    r_s = [persist_s[m]['r'] for m in metrics_s]
    colors_s = [PALETTE[i % len(PALETTE)] for i in range(len(metrics_s))]
    ax.barh(range(len(metrics_s)), r_s, color=colors_s, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(metrics_s)))
    ax.set_yticklabels([m.replace('_', ' ') for m in metrics_s])
    ax.set_xlabel('Season-to-Season Persistence (r)')
    ax.set_title('C. Season-to-Season Persistence (Consecutive, Same Division)', fontsize=11, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.5)
    for i, r in enumerate(r_s):
        ax.text(r + 0.02, i, f'r={r:.3f}', va='center', fontsize=9)
    
    # Panel D: Top/Bottom specialists table visualization
    ax = axes[1, 1]
    ax.axis('off')
    
    top_gpc = results.get('top_gpc', [])
    bottom_gpc = results.get('bottom_gpc', [])
    
    if top_gpc:
        top_df = pd.DataFrame(top_gpc[:8])
        bottom_df = pd.DataFrame(bottom_gpc[:8])
        
        table_data = []
        table_data.append(['Top Goals/Corner Specialists', '', '', '', ''])
        for _, row in top_df.iterrows():
            table_data.append([f"{row['Team']} ({row['Div']} {row['Season']})", 
                              f"{row['Corners_PG']:.1f}", f"{row['Goals_Per_Corner']:.3f}", 
                              f"{row['Shots_Per_Corner']:.1f}", f"{row['PPG']:.2f}"])
        table_data.append(['', '', '', '', ''])
        table_data.append(['Bottom Goals/Corner', '', '', '', ''])
        for _, row in bottom_df.iterrows():
            table_data.append([f"{row['Team']} ({row['Div']} {row['Season']})", 
                              f"{row['Corners_PG']:.1f}", f"{row['Goals_Per_Corner']:.3f}", 
                              f"{row['Shots_Per_Corner']:.1f}", f"{row['PPG']:.2f}"])
        
        table = ax.table(cellText=table_data, colLabels=['Team (Div Season)', 'Corners/PG', 'Goals/Corner', 'Shots/Corner', 'PPG'],
                        loc='center', cellLoc='left', colWidths=[0.35, 0.15, 0.15, 0.15, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        ax.set_title('D. Set-Piece Specialists (Min 50 Corners/Season)', fontsize=11, fontweight='bold', pad=20)
    
    plt.tight_layout()
    filepath = f'{figures_dir}/fig3_corner_efficiency.png'
    save_figure(fig, filepath)
    files.append(filepath)
    
    # Incremental R² plot
    if 'incremental_r2' in results:
        fig, ax = plt.subplots(figsize=(8, 5))
        inc = results['incremental_r2']
        models = ['Shots Only', '+ Corners/PG', '+ Goals/Corner', '+ Shots/Corner']
        r2_vals = [inc['shots_only'], inc['shots_plus_corners_pg'], 
                  inc['shots_plus_gpc'], inc['shots_plus_spc']]
        colors_inc = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        bars = ax.bar(models, r2_vals, color=colors_inc, alpha=0.7, edgecolor='black')
        ax.set_ylabel('R² (PPG Prediction)')
        ax.set_title('Incremental Predictive Value of Corner Metrics Beyond Shots', fontsize=12, fontweight='bold')
        for bar, r2 in zip(bars, r2_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                   f'{r2:.4f}', ha='center', va='bottom', fontweight='bold')
        # Annotate increments
        for i in range(1, len(r2_vals)):
            inc_val = r2_vals[i] - r2_vals[0]
            ax.annotate(f'ΔR²={inc_val:.4f}', xy=(i, r2_vals[i]), 
                       xytext=(0, 10), textcoords='offset points', ha='center', fontsize=9)
        plt.tight_layout()
        filepath = f'{figures_dir}/fig3b_corner_incremental_r2.png'
        save_figure(fig, filepath)
        files.append(filepath)
    
    # Save chart data
    pd.DataFrame([
        {'metric': m, 'r': corrs[m]['r'], 'p': corrs[m]['p'], 'n': corrs[m]['n']}
        for m in corrs
    ]).to_csv(f'{tables_dir}/table3a_corner_ppg_correlations.csv', index=False)
    
    pd.DataFrame([
        {'metric': m, 'half_r': persist[m]['r'], 'half_p': persist[m]['p'], 'half_n': persist[m]['n'],
         'season_r': persist_s.get(m, {}).get('r', np.nan), 
         'season_p': persist_s.get(m, {}).get('p', np.nan),
         'season_n': persist_s.get(m, {}).get('n', np.nan)}
        for m in metrics_p
    ]).to_csv(f'{tables_dir}/table3b_corner_persistence.csv', index=False)
    
    if 'incremental_r2' in results:
        pd.DataFrame([results['incremental_r2']]).to_csv(f'{tables_dir}/table3c_corner_incremental_r2.csv', index=False)
    
    return files


def plot_shooting_profiles(results: Dict, figures_dir: str = '.', tables_dir: str = '.') -> List[str]:
    """
    Figure 4: Selective vs Volume Shooting Profiles.
    
    Panel A: Quadrant scatter (Shots_PG_z vs Shot_Accuracy_z) colored by profile
    Panel B: PPG by profile (z-score adjusted)
    Panel C: Transition matrix heatmap
    Panel D: Profile-specific persistence
    """
    files = []
    
    profile_ppg_z = pd.DataFrame(results['profile_ppg_z'])
    trans_matrix_z = results['transition_matrix_z']
    stability = results['profile_stability']
    half_persist = results['half_season_persistence']
    profile_persist = results.get('profile_specific_persistence', {})
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel A: Quadrant visualization
    ax = axes[0, 0]
    profiles = ['High_Vol_High_Acc', 'High_Vol_Low_Acc', 'Low_Vol_High_Acc', 'Low_Vol_Low_Acc']
    profile_colors = {'High_Vol_High_Acc': '#2ca02c', 'High_Vol_Low_Acc': '#1f77b4', 
                     'Low_Vol_High_Acc': '#ff7f0e', 'Low_Vol_Low_Acc': '#d62728'}
    
    for profile in profiles:
        subset = profile_ppg_z[profile_ppg_z['Shooting_Profile_z'] == profile]
        if len(subset) > 0:
            # We don't have the raw z-scores here, so create representative positions
            vol_center = 1 if 'High_Vol' in profile else -1
            acc_center = 1 if 'High_Acc' in profile else -1
            ax.scatter(vol_center, acc_center, s=subset['N'].values[0]*2, 
                      color=profile_colors[profile], alpha=0.6, label=profile.replace('_', ' '), edgecolor='black')
            ax.annotate(f"N={subset['N'].values[0]}\nPPG={subset['PPG'].values[0]:.2f}", 
                       (vol_center, acc_center), ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Shot Volume (z-score, League-Season Adjusted)')
    ax.set_ylabel('Shot Accuracy (z-score, League-Season Adjusted)')
    ax.set_title('A. Shooting Profile Quadrants (z-score space)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    
    # Panel B: PPG by profile
    ax = axes[0, 1]
    x_pos = np.arange(len(profile_ppg_z))
    colors_b = [profile_colors.get(p, 'gray') for p in profile_ppg_z['Shooting_Profile_z']]
    bars = ax.bar(x_pos, profile_ppg_z['PPG'], color=colors_b, alpha=0.7, edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([p.replace('_', ' ') for p in profile_ppg_z['Shooting_Profile_z']], rotation=15, ha='right')
    ax.set_ylabel('Points per Game')
    ax.set_title('B. PPG by Shooting Profile (League-Season Adjusted)', fontsize=11, fontweight='bold')
    ax.axhline(y=1.33, color='gray', linestyle=':', alpha=0.7, label='Global Avg')
    for bar, ppg, n in zip(bars, profile_ppg_z['PPG'], profile_ppg_z['N']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
               f'{ppg:.2f}\n(n={n})', ha='center', va='bottom', fontsize=8)
    ax.legend(fontsize=8)
    
    # Panel C: Transition matrix heatmap
    ax = axes[1, 0]
    if len(trans_matrix_z) > 0:
        im = ax.imshow(trans_matrix_z.values, aspect='auto', cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(len(trans_matrix_z.columns)))
        ax.set_xticklabels([c.replace('_', ' ') for c in trans_matrix_z.columns], rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(trans_matrix_z.index)))
        ax.set_yticklabels([i.replace('_', ' ') for i in trans_matrix_z.index], fontsize=8)
        
        # Annotate
        for i in range(trans_matrix_z.shape[0]):
            for j in range(trans_matrix_z.shape[1]):
                val = trans_matrix_z.iloc[i, j]
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                       color='white' if val > 0.5 else 'black', fontsize=9)
        
        plt.colorbar(im, ax=ax, label='Transition Probability')
        ax.set_xlabel('Next Season Profile')
        ax.set_ylabel('Previous Season Profile')
        ax.set_title(f'C. Profile Transition Matrix (Stability={stability["z_score_adjusted"]:.1%})', fontsize=11, fontweight='bold')
    
    # Panel D: Half-season persistence + profile-specific persistence
    ax = axes[1, 1]
    metrics_h = list(half_persist.keys())
    r_h = [half_persist[m]['r'] for m in metrics_h]
    labels_h = [m.replace('_', ' ') for m in metrics_h]
    
    x_pos = np.arange(len(metrics_h) + len(profile_persist))
    all_r = r_h + [profile_persist[p]['stability'] for p in profile_persist]
    all_labels = labels_h + [f'{p.replace("_", " ")} Stability' for p in profile_persist]
    colors_d = ['#1f77b4']*len(metrics_h) + ['#2ca02c']*len(profile_persist)
    
    bars = ax.bar(x_pos, all_r, color=colors_d, alpha=0.7, edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Persistence / Stability (r or %)')
    ax.set_title('D. Metric Persistence & Profile Stability', fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(1.0, max(all_r)*1.2))
    for bar, val in zip(bars, all_r):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
               f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    filepath = f'{figures_dir}/fig4_shooting_profiles.png'
    save_figure(fig, filepath)
    files.append(filepath)
    
    # Save chart data
    profile_ppg_z.to_csv(f'{tables_dir}/table4a_shooting_profile_ppg.csv', index=False)
    trans_matrix_z.to_csv(f'{tables_dir}/table4b_shooting_transition_matrix.csv')
    pd.DataFrame([{'profile_stability_z': stability['z_score_adjusted'], 
                   'profile_stability_raw': stability['raw'],
                   'n_transitions': stability['n_transitions']}]).to_csv(
        f'{tables_dir}/table4c_shooting_profile_stability.csv', index=False)
    pd.DataFrame([
        {'metric': m, 'r': half_persist[m]['r'], 'p': half_persist[m]['p'], 'n': half_persist[m]['n']}
        for m in half_persist
    ]).to_csv(f'{tables_dir}/table4d_shooting_half_persistence.csv', index=False)
    
    return files


def plot_clutch_performance(results: Dict, figures_dir: str = '.', tables_dir: str = '.') -> List[str]:
    """
    Figure 5: Clutch Performance (Points from Winning/Losing Positions).
    
    Panel A: Points Dropped When Leading vs Points Gained When Trailing (team-season)
    Panel B: Clutch Score distribution + top/bottom teams
    Panel C: Early vs Late season persistence
    Panel D: By venue and deficit size
    """
    files = []
    
    team_season_clutch = results['team_season_clutch']
    early_late = results.get('early_late_data', pd.DataFrame())
    by_venue = results.get('by_venue', pd.DataFrame())
    by_deficit = results.get('by_deficit_size', pd.DataFrame())
    team_clutch = results.get('team_clutch', pd.DataFrame())
    overall = results['overall_rates']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel A: Scatter of Drop vs Gain
    ax = axes[0, 0]
    valid = team_season_clutch['Points_Dropped_When_Leading'].notna() & team_season_clutch['Points_Gained_When_Trailing'].notna()
    scatter = ax.scatter(team_season_clutch.loc[valid, 'Points_Dropped_When_Leading'], 
                        team_season_clutch.loc[valid, 'Points_Gained_When_Trailing'],
                        c=team_season_clutch.loc[valid, 'PPG'], cmap='RdYlGn', alpha=0.6, s=20)
    ax.axhline(y=team_season_clutch['Points_Gained_When_Trailing'].mean(), color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=team_season_clutch['Points_Dropped_When_Leading'].mean(), color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Points Dropped When Leading (per match)')
    ax.set_ylabel('Points Gained When Trailing (per match)')
    ax.set_title('A. Clutch Performance: Holding Leads vs Coming Back', fontsize=11, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Season PPG')
    
    # Quadrant labels
    ax.text(0.95, 0.95, 'Good at both', transform=ax.transAxes, ha='right', va='top', fontsize=9, alpha=0.7)
    ax.text(0.05, 0.95, 'Hold leads,\ndon\'t comeback', transform=ax.transAxes, ha='left', va='top', fontsize=9, alpha=0.7)
    ax.text(0.95, 0.05, 'Comeback,\ndrop leads', transform=ax.transAxes, ha='right', va='bottom', fontsize=9, alpha=0.7)
    ax.text(0.05, 0.05, 'Poor at both', transform=ax.transAxes, ha='left', va='bottom', fontsize=9, alpha=0.7)
    
    # Panel B: Clutch Score distribution + top/bottom
    ax = axes[0, 1]
    ax.hist(team_season_clutch['Clutch_Score'].dropna(), bins=30, alpha=0.7, color='gray', edgecolor='black')
    ax.axvline(x=0, color='black', linewidth=1)
    ax.axvline(x=team_season_clutch['Clutch_Score'].mean(), color='red', linestyle='--', 
               label=f'Mean = {team_season_clutch["Clutch_Score"].mean():.3f}')
    ax.set_xlabel('Clutch Score (Gain When Trailing - Drop When Leading)')
    ax.set_ylabel('Frequency')
    ax.set_title('B. Clutch Score Distribution', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    
    # Top/bottom teams annotation
    if len(team_clutch) > 0:
        top3 = team_clutch.head(3)
        bot3 = team_clutch.tail(3)
        for _, row in top3.iterrows():
            ax.annotate(f"{row['Team']} ({row['Avg_Clutch_Score']:.3f})", 
                       (row['Avg_Clutch_Score'], ax.get_ylim()[1]*0.9), 
                       ha='center', fontsize=7, color='green', fontweight='bold')
        for _, row in bot3.iterrows():
            ax.annotate(f"{row['Team']} ({row['Avg_Clutch_Score']:.3f})", 
                       (row['Avg_Clutch_Score'], ax.get_ylim()[1]*0.1), 
                       ha='center', fontsize=7, color='red', fontweight='bold')
    
    # Panel C: Early vs Late persistence
    ax = axes[1, 0]
    if len(early_late) > 0:
        ax.scatter(early_late['Drop_Early'], early_late['Drop_Late'], alpha=0.5, s=20, color='#1f77b4', label='Dropped When Leading')
        ax.scatter(early_late['Gain_Early'], early_late['Gain_Late'], alpha=0.5, s=20, color='#ff7f0e', label='Gained When Trailing')
        mn = min(early_late[['Drop_Early', 'Drop_Late', 'Gain_Early', 'Gain_Late']].min().min(), 0)
        mx = max(early_late[['Drop_Early', 'Drop_Late', 'Gain_Early', 'Gain_Late']].max().max(), 0.5)
        ax.plot([mn, mx], [mn, mx], 'k--', alpha=0.5, label='Equality')
        ax.set_xlabel('Early Season')
        ax.set_ylabel('Late Season')
        ax.set_title('C. Early vs Late Season Persistence', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        
        # Annotate correlations
        persist = results.get('early_late_persistence', {})
        if persist:
            r_drop = persist.get('points_dropped_when_leading', {}).get('r', 0)
            r_gain = persist.get('points_gained_when_trailing', {}).get('r', 0)
            ax.text(0.05, 0.95, f'Drop r={r_drop:.3f}\nGain r={r_gain:.3f}', 
                   transform=ax.transAxes, va='top', fontsize=9, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Panel D: By venue and deficit size
    ax = axes[1, 1]
    
    # Venue comparison
    if len(by_venue) > 0:
        venue_pivot = by_venue.pivot_table(index='Venue', columns='HT_Result', values='Avg_FT_Points')
        venue_pivot = venue_pivot.reindex(columns=['W', 'D', 'L'])
        x = np.arange(len(venue_pivot.index))
        width = 0.25
        for i, col in enumerate(['W', 'D', 'L']):
            if col in venue_pivot.columns:
                ax.bar(x + (i-1)*width, venue_pivot[col], width, label=f'HT {col}', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(venue_pivot.index)
        ax.set_ylabel('Avg FT Points')
        ax.set_title('D. HT-FT Points by Venue', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    filepath = f'{figures_dir}/fig5_clutch_performance.png'
    save_figure(fig, filepath)
    files.append(filepath)
    
    # Deficit size recovery plot
    if len(by_deficit) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        deficit_pivot = by_deficit.pivot_table(index='Deficit_Cat', columns='Venue', values='Win_Rate')
        x = np.arange(len(deficit_pivot.index))
        width = 0.35
        for i, venue in enumerate(['Home', 'Away']):
            if venue in deficit_pivot.columns:
                ax.bar(x + (i-0.5)*width, deficit_pivot[venue], width, label=venue, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(deficit_pivot.index)
        ax.set_ylabel('Win Rate from Trailing at HT')
        ax.set_title('Recovery from Halftime Deficit by Size and Venue', fontsize=12, fontweight='bold')
        ax.legend()
        plt.tight_layout()
        filepath = f'{figures_dir}/fig5b_clutch_by_deficit.png'
        save_figure(fig, filepath)
        files.append(filepath)
    
    # Save chart data
    team_season_clutch.to_csv(f'{tables_dir}/table5a_team_season_clutch.csv', index=False)
    if len(early_late) > 0:
        early_late.to_csv(f'{tables_dir}/table5b_clutch_early_late.csv', index=False)
    if len(team_clutch) > 0:
        team_clutch.to_csv(f'{tables_dir}/table5c_team_clutch.csv', index=False)
    by_venue.to_csv(f'{tables_dir}/table5d_clutch_by_venue.csv', index=False)
    by_deficit.to_csv(f'{tables_dir}/table5e_clutch_by_deficit.csv', index=False)
    
    return files


def create_all_next_figures(results: Dict, figures_dir: str = '.', tables_dir: str = '.') -> List[str]:
    """Create all 5 next insight figures."""
    all_files = []
    
    all_files.extend(plot_competitive_balance(results['competitive_balance'], figures_dir, tables_dir))
    all_files.extend(plot_league_style_fingerprints(results['league_style'], figures_dir, tables_dir))
    all_files.extend(plot_corner_efficiency(results['corner_efficiency'], figures_dir, tables_dir))
    all_files.extend(plot_shooting_profiles(results['shooting_profiles'], figures_dir, tables_dir))
    all_files.extend(plot_clutch_performance(results['clutch_performance'], figures_dir, tables_dir))
    
    return all_files