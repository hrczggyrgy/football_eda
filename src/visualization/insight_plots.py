"""
Advanced insight plots module.
Creates publication-quality figures for the priority analyses A-F.
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


def plot_finishing_persistence(results: Dict, output_dir: str = '.') -> List[str]:
    """
    Figure A: Finishing persistence — early vs late season.
    
    Linked two-panel figure:
    1. Early vs Later performance with equality and fitted lines
    2. Quantile-group trajectories showing subsequent performance
    """
    files = []
    
    period_eff = results['period_efficiency']
    raw_persist = results['raw_persistence']
    adj_persist = results['adjusted_persistence']
    trajectories = results['quantile_trajectories']
    split_sens = results['split_sensitivity']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Panel 1a: GPS Early vs Late
    ax = axes[0, 0]
    metric = 'Goals_Per_Shot_Early'
    late_metric = 'Goals_Per_Shot_Late'
    valid = period_eff[metric].notna() & period_eff[late_metric].notna()
    if valid.sum() > 10:
        ax.scatter(period_eff.loc[valid, metric], period_eff.loc[valid, late_metric], alpha=0.4, s=15, color=PALETTE[0])
        mn = min(period_eff.loc[valid, metric].min(), period_eff.loc[valid, late_metric].min())
        mx = max(period_eff.loc[valid, metric].max(), period_eff.loc[valid, late_metric].max())
        ax.plot([mn, mx], [mn, mx], 'k--', alpha=0.5, label='Equality')
        z = np.polyfit(period_eff.loc[valid, metric], period_eff.loc[valid, late_metric], 1)
        p = np.poly1d(z)
        x_range = np.linspace(mn, mx, 100)
        ax.plot(x_range, p(x_range), 'r-', alpha=0.8, label=f'Fit (r={raw_persist.get(metric, {}).get("r", 0):.3f})')
        ax.set_xlabel('Goals per Shot (Early Season)')
        ax.set_ylabel('Goals per Shot (Late Season)')
        ax.set_title('A. Finishing (GPS) Persistence')
        ax.legend(fontsize=8)
    
    # Panel 1b: GPSOT Early vs Late
    ax = axes[0, 1]
    metric = 'Goals_Per_Shot_OT_Early'
    late_metric = 'Goals_Per_Shot_OT_Late'
    valid = period_eff[metric].notna() & period_eff[late_metric].notna()
    if valid.sum() > 10:
        ax.scatter(period_eff.loc[valid, metric], period_eff.loc[valid, late_metric], alpha=0.4, s=15, color=PALETTE[1])
        mn = min(period_eff.loc[valid, metric].min(), period_eff.loc[valid, late_metric].min())
        mx = max(period_eff.loc[valid, metric].max(), period_eff.loc[valid, late_metric].max())
        ax.plot([mn, mx], [mn, mx], 'k--', alpha=0.5, label='Equality')
        z = np.polyfit(period_eff.loc[valid, metric], period_eff.loc[valid, late_metric], 1)
        p = np.poly1d(z)
        x_range = np.linspace(mn, mx, 100)
        ax.plot(x_range, p(x_range), 'r-', alpha=0.8, label=f'Fit (r={raw_persist.get(metric, {}).get("r", 0):.3f})')
        ax.set_xlabel('Goals per Shot on Target (Early)')
        ax.set_ylabel('Goals per Shot on Target (Late)')
        ax.set_title('B. Finishing (GPSOT) Persistence')
        ax.legend(fontsize=8)
    
    # Panel 1c: Shot Accuracy Early vs Late
    ax = axes[0, 2]
    metric = 'Shot_Accuracy_Early'
    late_metric = 'Shot_Accuracy_Late'
    valid = period_eff[metric].notna() & period_eff[late_metric].notna()
    if valid.sum() > 10:
        ax.scatter(period_eff.loc[valid, metric], period_eff.loc[valid, late_metric], alpha=0.4, s=15, color=PALETTE[2])
        mn = min(period_eff.loc[valid, metric].min(), period_eff.loc[valid, late_metric].min())
        mx = max(period_eff.loc[valid, metric].max(), period_eff.loc[valid, late_metric].max())
        ax.plot([mn, mx], [mn, mx], 'k--', alpha=0.5, label='Equality')
        z = np.polyfit(period_eff.loc[valid, metric], period_eff.loc[valid, late_metric], 1)
        p = np.poly1d(z)
        x_range = np.linspace(mn, mx, 100)
        ax.plot(x_range, p(x_range), 'r-', alpha=0.8, label=f'Fit (r={raw_persist.get(metric, {}).get("r", 0):.3f})')
        ax.set_xlabel('Shot Accuracy (Early)')
        ax.set_ylabel('Shot Accuracy (Late)')
        ax.set_title('C. Shot Accuracy Persistence')
        ax.legend(fontsize=8)
    
    # Panel 2a: Quantile trajectories for GPS
    ax = axes[1, 0]
    metric = 'Goals_Per_Shot_Early'
    if metric in trajectories:
        traj_data = pd.DataFrame(trajectories[metric])
        q_order = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
        traj_data = traj_data.set_index('Quantile').loc[q_order].reset_index()
        x_pos = np.arange(len(traj_data))
        ax.errorbar(x_pos, traj_data['mean'], yerr=traj_data['std'], 
                    fmt='o-', capsize=4, capthick=1, color=PALETTE[0], label='Late GPS')
        ax.axhline(y=period_eff['Goals_Per_Shot_Late'].mean(), color='gray', linestyle=':', alpha=0.7, label='Overall Mean')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(traj_data['Quantile'])
        ax.set_xlabel('Early Season GPS Quintile')
        ax.set_ylabel('Late Season GPS (Mean ± SD)')
        ax.set_title('D. GPS: Early Quintile → Late Performance')
        ax.legend(fontsize=8)
    
    # Panel 2b: Quantile trajectories for GPSOT
    ax = axes[1, 1]
    metric = 'Goals_Per_Shot_OT_Early'
    if metric in trajectories:
        traj_data = pd.DataFrame(trajectories[metric])
        q_order = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
        traj_data = traj_data.set_index('Quantile').loc[q_order].reset_index()
        x_pos = np.arange(len(traj_data))
        ax.errorbar(x_pos, traj_data['mean'], yerr=traj_data['std'], 
                    fmt='o-', capsize=4, capthick=1, color=PALETTE[1], label='Late GPSOT')
        ax.axhline(y=period_eff['Goals_Per_Shot_OT_Late'].mean(), color='gray', linestyle=':', alpha=0.7, label='Overall Mean')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(traj_data['Quantile'])
        ax.set_xlabel('Early Season GPSOT Quintile')
        ax.set_ylabel('Late Season GPSOT (Mean ± SD)')
        ax.set_title('E. GPSOT: Early Quintile → Late Performance')
        ax.legend(fontsize=8)
    
    # Panel 2c: Split sensitivity
    ax = axes[1, 2]
    split_labels = ['1/3', '1/2', '2/3']
    gps_splits = [split_sens.get(f'GPS_Early_split{int(s*100)}', {}).get('r', np.nan) for s in [1/3, 1/2, 2/3]]
    gpsot_splits = [split_sens.get(f'GPSOT_Early_split{int(s*100)}', {}).get('r', np.nan) for s in [1/3, 1/2, 2/3]]
    ax.plot(split_labels, gps_splits, 'o-', color=PALETTE[0], label='GPS')
    ax.plot(split_labels, gpsot_splits, 's-', color=PALETTE[1], label='GPSOT')
    ax.set_xlabel('Early Season Split Point')
    ax.set_ylabel('Early-Late Correlation (r)')
    ax.set_title('F. Sensitivity to Split Point')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 0.5)
    
    plt.tight_layout()
    filepath = f'{output_dir}/figA_finishing_persistence.png'
    save_figure(fig, filepath)
    files.append(filepath)
    
    # Save chart data
    chart_data = period_eff[['Div', 'Team', 'Season', 'Goals_Per_Shot_Early', 'Goals_Per_Shot_Late',
                              'Goals_Per_Shot_OT_Early', 'Goals_Per_Shot_OT_Late',
                              'Shot_Accuracy_Early', 'Shot_Accuracy_Late']].copy()
    chart_data.to_csv(f'{output_dir}/tableA_finishing_persistence_data.csv', index=False)
    
    return files


def plot_team_profile_changes(results: Dict, output_dir: str = '.') -> List[str]:
    """
    Figure B: Team profile changes — arrow plot of consecutive season changes.
    
    X: relative attacking shot generation (Shots_PG_z)
    Y: relative opponent-shot suppression (Shot_Suppression_z)
    Arrows: consecutive-season changes
    Highlight substantial changes.
    """
    files = []
    
    within_div = results['within_division_changes']
    change_summary = results['change_summary']
    
    # Get the latest season for each team as arrow start point
    # and the change vector as arrow direction
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Panel 1: Attack profile changes (Shots_PG_z vs GPSOT_z)
    ax = axes[0]
    if 'Shots_PG_z' in within_div.columns and 'Goals_Per_Shot_OT_z' in within_div.columns:
        # Plot all team-seasons as points (light)
        valid = within_div['Shots_PG_z'].notna() & within_div['Goals_Per_Shot_OT_z'].notna()
        ax.scatter(within_div.loc[valid, 'Shots_PG_z'], within_div.loc[valid, 'Goals_Per_Shot_OT_z'], 
                   alpha=0.2, s=10, color='gray')
        
        # Plot arrows for changes
        arrow_data = within_div[
            within_div['Delta_Shots_PG_z'].notna() & within_div['Delta_Goals_Per_Shot_OT_z'].notna()
        ].copy()
        
        # Select top changes for labeling
        arrow_data['Change_Magnitude'] = np.sqrt(arrow_data['Delta_Shots_PG_z']**2 + arrow_data['Delta_Goals_Per_Shot_OT_z']**2)
        top_changes = arrow_data.nlargest(15, 'Change_Magnitude')
        
        for _, row in arrow_data.iterrows():
            color = PALETTE[2] if row['Delta_Shots_PG_z'] > 0 else PALETTE[3]
            ax.arrow(row['Shots_PG_z'] - row['Delta_Shots_PG_z'], 
                     row['Goals_Per_Shot_OT_z'] - row['Delta_Goals_Per_Shot_OT_z'],
                     row['Delta_Shots_PG_z'], row['Delta_Goals_Per_Shot_OT_z'],
                     head_width=0.08, head_length=0.1, fc=color, ec=color, alpha=0.4, length_includes_head=True)
        
        # Label top changes
        for _, row in top_changes.iterrows():
            ax.annotate(f"{row['Team']} {row['Season']}", 
                       (row['Shots_PG_z'], row['Goals_Per_Shot_OT_z']),
                       fontsize=7, alpha=0.8, ha='center')
        
        ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Shot Generation (z-score vs League-Season)')
        ax.set_ylabel('Finishing (GPSOT z-score vs League-Season)')
        ax.set_title('A. Attack Profile Changes (Consecutive Seasons)')
        ax.set_aspect('equal')
    
    # Panel 2: Defense profile changes (Shot_Suppression_z vs Def_Resilience_z)
    ax = axes[1]
    if 'Shot_Suppression_z' in within_div.columns and 'Def_Resilience_z' in within_div.columns:
        valid = within_div['Shot_Suppression_z'].notna() & within_div['Def_Resilience_z'].notna()
        ax.scatter(within_div.loc[valid, 'Shot_Suppression_z'], within_div.loc[valid, 'Def_Resilience_z'], 
                   alpha=0.2, s=10, color='gray')
        
        arrow_data = within_div[
            within_div['Delta_Shot_Suppression_z'].notna() & within_div['Delta_Def_Resilience_z'].notna()
        ].copy()
        
        arrow_data['Change_Magnitude'] = np.sqrt(arrow_data['Delta_Shot_Suppression_z']**2 + arrow_data['Delta_Def_Resilience_z']**2)
        top_changes = arrow_data.nlargest(15, 'Change_Magnitude')
        
        for _, row in arrow_data.iterrows():
            color = PALETTE[2] if row['Delta_Shot_Suppression_z'] > 0 else PALETTE[3]
            ax.arrow(row['Shot_Suppression_z'] - row['Delta_Shot_Suppression_z'], 
                     row['Def_Resilience_z'] - row['Delta_Def_Resilience_z'],
                     row['Delta_Shot_Suppression_z'], row['Delta_Def_Resilience_z'],
                     head_width=0.08, head_length=0.1, fc=color, ec=color, alpha=0.4, length_includes_head=True)
        
        for _, row in top_changes.iterrows():
            ax.annotate(f"{row['Team']} {row['Season']}", 
                       (row['Shot_Suppression_z'], row['Def_Resilience_z']),
                       fontsize=7, alpha=0.8, ha='center')
        
        ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Shot Suppression (z-score, higher = fewer opp shots)')
        ax.set_ylabel('Defensive Resilience (z-score, higher = fewer goals/opp SoT)')
        ax.set_title('B. Defense Profile Changes (Consecutive Seasons)')
        ax.set_aspect('equal')
    
    plt.tight_layout()
    filepath = f'{output_dir}/figB_team_profile_changes.png'
    save_figure(fig, filepath)
    files.append(filepath)
    
    # Promotion/relegation changes
    if 'promotion_relegation_changes' in results and len(results['promotion_relegation_changes']) > 0:
        pr_changes = results['promotion_relegation_changes']
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if 'Delta_Shots_PG_z' in pr_changes.columns and 'Delta_Goals_Per_Shot_OT_z' in pr_changes.columns:
            valid = pr_changes['Delta_Shots_PG_z'].notna() & pr_changes['Delta_Goals_Per_Shot_OT_z'].notna()
            pr_valid = pr_changes.loc[valid].copy()
            
            for _, row in pr_valid.iterrows():
                color = PALETTE[0] if 'Div' in row and row['Div'] in ['E0', 'SP1', 'D1', 'I1', 'F1'] else PALETTE[1]
                marker = '^' if row['Prev_Div'] in ['E1', 'E2', 'E3', 'SP2', 'D2', 'I2', 'F2'] else 'v'  # promotion vs relegation
                ax.scatter(row['Delta_Shots_PG_z'], row['Delta_Goals_Per_Shot_OT_z'], 
                          color=color, marker=marker, s=80, alpha=0.8, edgecolor='black', linewidth=0.5)
                ax.annotate(f"{row['Team']} {row['Prev_Div']}→{row['Div']}", 
                           (row['Delta_Shots_PG_z'], row['Delta_Goals_Per_Shot_OT_z']),
                           fontsize=8, alpha=0.9, ha='center')
            
            ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
            ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
            ax.set_xlabel('Change in Shot Generation (z-score)')
            ax.set_ylabel('Change in Finishing (z-score)')
            ax.set_title('Promotion/Relegation Profile Changes')
        
        plt.tight_layout()
        filepath = f'{output_dir}/figB_promo_relegation_changes.png'
        save_figure(fig, filepath)
        files.append(filepath)
    
    # Save chart data
    within_div.to_csv(f'{output_dir}/tableB_within_div_changes.csv', index=False)
    if 'promotion_relegation_changes' in results:
        results['promotion_relegation_changes'].to_csv(f'{output_dir}/tableB_promo_releg_changes.csv', index=False)
    
    return files


def plot_halftime_deficit_recovery(results: Dict, output_dir: str = '.') -> List[str]:
    """
    Figure C: Halftime deficit recovery — conditional probability heatmaps.
    
    Deficit size, venue, strength category.
    One clear conditional visual.
    """
    files = []
    
    trailing_recovery = results['trailing_recovery']
    trailing_recovery_strength = results.get('trailing_recovery_strength', pd.DataFrame())
    overall = results['overall']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel 1: Win rate by deficit size and venue
    ax = axes[0, 0]
    if len(trailing_recovery) > 0:
        pivot_win = trailing_recovery.pivot_table(values='Win_Rate', index='Deficit_Size', columns='Venue', aggfunc='mean')
        pivot_n = trailing_recovery.pivot_table(values='N', index='Deficit_Size', columns='Venue', aggfunc='sum')
        
        im = ax.imshow(pivot_win.values, aspect='auto', cmap='RdYlGn', vmin=0, vmax=0.5)
        ax.set_xticks(range(len(pivot_win.columns)))
        ax.set_xticklabels(pivot_win.columns)
        ax.set_yticks(range(len(pivot_win.index)))
        ax.set_yticklabels([f'-{int(x)}' for x in pivot_win.index])
        
        # Annotate with rates and counts
        for i in range(pivot_win.shape[0]):
            for j in range(pivot_win.shape[1]):
                rate = pivot_win.iloc[i, j]
                n = pivot_n.iloc[i, j]
                if not np.isnan(rate):
                    ax.text(j, i, f'{rate:.1%}\n(n={int(n)})', ha='center', va='center', fontsize=9,
                           color='white' if rate < 0.25 else 'black')
        
        plt.colorbar(im, ax=ax, label='Win Rate')
        ax.set_xlabel('Venue')
        ax.set_ylabel('Halftime Deficit (Goals)')
        ax.set_title('A. Win Rate from Halftime Deficit by Venue')
    
    # Panel 2: Avoid defeat rate by deficit size and venue
    ax = axes[0, 1]
    if len(trailing_recovery) > 0:
        pivot_avoid = trailing_recovery.pivot_table(values='Avoid_Defeat_Rate', index='Deficit_Size', columns='Venue', aggfunc='mean')
        
        im = ax.imshow(pivot_avoid.values, aspect='auto', cmap='RdYlGn', vmin=0, vmax=0.6)
        ax.set_xticks(range(len(pivot_avoid.columns)))
        ax.set_xticklabels(pivot_avoid.columns)
        ax.set_yticks(range(len(pivot_avoid.index)))
        ax.set_yticklabels([f'-{int(x)}' for x in pivot_avoid.index])
        
        for i in range(pivot_avoid.shape[0]):
            for j in range(pivot_avoid.shape[1]):
                rate = pivot_avoid.iloc[i, j]
                n = pivot_n.iloc[i, j]
                if not np.isnan(rate):
                    ax.text(j, i, f'{rate:.1%}\n(n={int(n)})', ha='center', va='center', fontsize=9,
                           color='white' if rate < 0.3 else 'black')
        
        plt.colorbar(im, ax=ax, label='Avoid Defeat Rate')
        ax.set_xlabel('Venue')
        ax.set_ylabel('Halftime Deficit (Goals)')
        ax.set_title('B. Avoid Defeat Rate from Halftime Deficit by Venue')
    
    # Panel 3: Recovery by strength category (home teams trailing)
    ax = axes[1, 0]
    if len(trailing_recovery_strength) > 0:
        home_trailing = trailing_recovery_strength[trailing_recovery_strength['Venue'] == 'Home']
        if len(home_trailing) > 0:
            pivot_strength = home_trailing.pivot_table(values='Avoid_Defeat_Rate', index='Deficit_Size', columns='Strength_Tercile', aggfunc='mean')
            pivot_n_s = home_trailing.pivot_table(values='N', index='Deficit_Size', columns='Strength_Tercile', aggfunc='sum')
            
            strength_order = ['Weak', 'Average', 'Strong']
            pivot_strength = pivot_strength.reindex(columns=[c for c in strength_order if c in pivot_strength.columns])
            pivot_n_s = pivot_n_s.reindex(columns=[c for c in strength_order if c in pivot_n_s.columns])
            
            im = ax.imshow(pivot_strength.values, aspect='auto', cmap='RdYlGn', vmin=0, vmax=0.6)
            ax.set_xticks(range(len(pivot_strength.columns)))
            ax.set_xticklabels(pivot_strength.columns)
            ax.set_yticks(range(len(pivot_strength.index)))
            ax.set_yticklabels([f'-{int(x)}' for x in pivot_strength.index])
            
            for i in range(pivot_strength.shape[0]):
                for j in range(pivot_strength.shape[1]):
                    rate = pivot_strength.iloc[i, j]
                    n = pivot_n_s.iloc[i, j]
                    if not np.isnan(rate):
                        ax.text(j, i, f'{rate:.1%}\n(n={int(n)})', ha='center', va='center', fontsize=9,
                               color='white' if rate < 0.3 else 'black')
            
            plt.colorbar(im, ax=ax, label='Avoid Defeat Rate')
            ax.set_xlabel('Pre-Match Strength Tercile')
            ax.set_ylabel('Halftime Deficit (Goals)')
            ax.set_title('C. Home Teams: Avoid Defeat by Strength & Deficit')
    
    # Panel 4: Recovery by strength category (away teams trailing)
    ax = axes[1, 1]
    if len(trailing_recovery_strength) > 0:
        away_trailing = trailing_recovery_strength[trailing_recovery_strength['Venue'] == 'Away']
        if len(away_trailing) > 0:
            pivot_strength = away_trailing.pivot_table(values='Avoid_Defeat_Rate', index='Deficit_Size', columns='Strength_Tercile', aggfunc='mean')
            pivot_n_s = away_trailing.pivot_table(values='N', index='Deficit_Size', columns='Strength_Tercile', aggfunc='sum')
            
            strength_order = ['Weak', 'Average', 'Strong']
            pivot_strength = pivot_strength.reindex(columns=[c for c in strength_order if c in pivot_strength.columns])
            pivot_n_s = pivot_n_s.reindex(columns=[c for c in strength_order if c in pivot_n_s.columns])
            
            im = ax.imshow(pivot_strength.values, aspect='auto', cmap='RdYlGn', vmin=0, vmax=0.6)
            ax.set_xticks(range(len(pivot_strength.columns)))
            ax.set_xticklabels(pivot_strength.columns)
            ax.set_yticks(range(len(pivot_strength.index)))
            ax.set_yticklabels([f'-{int(x)}' for x in pivot_strength.index])
            
            for i in range(pivot_strength.shape[0]):
                for j in range(pivot_strength.shape[1]):
                    rate = pivot_strength.iloc[i, j]
                    n = pivot_n_s.iloc[i, j]
                    if not np.isnan(rate):
                        ax.text(j, i, f'{rate:.1%}\n(n={int(n)})', ha='center', va='center', fontsize=9,
                               color='white' if rate < 0.3 else 'black')
            
            plt.colorbar(im, ax=ax, label='Avoid Defeat Rate')
            ax.set_xlabel('Pre-Match Strength Tercile')
            ax.set_ylabel('Halftime Deficit (Goals)')
            ax.set_title('D. Away Teams: Avoid Defeat by Strength & Deficit')
    
    plt.tight_layout()
    filepath = f'{output_dir}/figC_halftime_deficit_recovery.png'
    save_figure(fig, filepath)
    files.append(filepath)
    
    # Save chart data
    results['trailing_recovery'].to_csv(f'{output_dir}/tableC_trailing_recovery.csv', index=False)
    if 'trailing_recovery_strength' in results:
        results['trailing_recovery_strength'].to_csv(f'{output_dir}/tableC_trailing_recovery_strength.csv', index=False)
    
    return files


def plot_volatility_streakiness(results: Dict, output_dir: str = '.') -> List[str]:
    """
    Figure D: Volatility and streakiness — two panels.
    
    1. Expected vs observed dispersion (simulation benchmark)
    2. Excess streakiness (runs test Z vs permutation null)
    """
    files = []
    
    residual_disp = results['residual_dispersion']
    sim_bench = results['simulation_benchmark']
    streakiness = results['streakiness']
    perm_streak = results['permutation_streakiness']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel 1: Observed vs simulated residual dispersion
    ax = axes[0]
    if len(sim_bench) > 0:
        # Merge observed and simulated
        merged = residual_disp[['Div', 'Team', 'Season', 'Points_Std', 'Matches']].merge(
            sim_bench[['Div', 'Team', 'Season', 'Simulated_Std_Mean', 'Simulated_Std_Std', 'Z_Score']],
            on=['Div', 'Team', 'Season']
        )
        valid = merged['Z_Score'].notna()
        if valid.sum() > 10:
            ax.scatter(merged.loc[valid, 'Simulated_Std_Mean'], merged.loc[valid, 'Points_Std'], 
                       alpha=0.5, s=20, c=merged.loc[valid, 'Z_Score'], cmap='RdBu_r', vmin=-3, vmax=3)
            mx = max(merged.loc[valid, 'Simulated_Std_Mean'].max(), merged.loc[valid, 'Points_Std'].max())
            ax.plot([0, mx], [0, mx], 'k--', alpha=0.5, label='Equality')
            ax.axhline(y=merged.loc[valid, 'Points_Std'].mean(), color='gray', linestyle=':', alpha=0.5)
            ax.axvline(x=merged.loc[valid, 'Simulated_Std_Mean'].mean(), color='gray', linestyle=':', alpha=0.5)
            
            # Highlight extreme teams
            high_z = merged.loc[valid].nlargest(10, 'Z_Score')
            low_z = merged.loc[valid].nsmallest(10, 'Z_Score')
            ax.scatter(high_z['Simulated_Std_Mean'], high_z['Points_Std'], color='red', s=50, label='High Volatility (Z>0)', zorder=5)
            ax.scatter(low_z['Simulated_Std_Mean'], low_z['Points_Std'], color='blue', s=50, label='Low Volatility (Z<0)', zorder=5)
            
            ax.set_xlabel('Simulated Residual Std (Model-Based Expectation)')
            ax.set_ylabel('Observed Points Std')
            ax.set_title('A. Observed vs Expected Dispersion\n(Color = Z-score vs Simulation)')
            ax.legend(fontsize=8)
            plt.colorbar(ax.collections[0], ax=ax, label='Z-Score')
    
    # Panel 2: Streakiness (runs test) vs permutation null
    ax = axes[1]
    if len(perm_streak) > 0:
        valid = perm_streak['Observed_Runs_Z'].notna() & perm_streak['Perm_P_Value'].notna()
        if valid.sum() > 10:
            colors = ['red' if (p < 0.05 and z < 0) else 'blue' if (p < 0.05 and z > 0) else 'gray' 
                     for z, p in zip(perm_streak.loc[valid, 'Observed_Runs_Z'], perm_streak.loc[valid, 'Perm_P_Value'])]
            ax.scatter(perm_streak.loc[valid, 'Observed_Runs_Z'], -np.log10(perm_streak.loc[valid, 'Perm_P_Value']), 
                       alpha=0.6, s=30, c=colors)
            ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
            ax.axhline(y=-np.log10(0.05), color='red', linestyle=':', alpha=0.5, label='p=0.05')
            
            # Label significant streaky/consistent teams
            sig_streaky = perm_streak.loc[valid][(perm_streak.loc[valid, 'Perm_P_Value'] < 0.05) & (perm_streak.loc[valid, 'Observed_Runs_Z'] < -1.5)]
            sig_consistent = perm_streak.loc[valid][(perm_streak.loc[valid, 'Perm_P_Value'] < 0.05) & (perm_streak.loc[valid, 'Observed_Runs_Z'] > 1.5)]
            
            for _, row in sig_streaky.head(5).iterrows():
                ax.annotate(f"{row['Team']} {row['Season']}", 
                           (row['Observed_Runs_Z'], -np.log10(row['Perm_P_Value'])),
                           fontsize=7, color='red', ha='center')
            for _, row in sig_consistent.head(5).iterrows():
                ax.annotate(f"{row['Team']} {row['Season']}", 
                           (row['Observed_Runs_Z'], -np.log10(row['Perm_P_Value'])),
                           fontsize=7, color='blue', ha='center')
            
            ax.set_xlabel('Runs Test Z-Score (Negative = Streakier)')
            ax.set_ylabel('-log10(Permutation p-value)')
            ax.set_title('B. Excess Streakiness vs Permutation Null\n(Red = Streaky, Blue = Anti-Streaky, Gray = Null)')
            ax.legend(fontsize=8)
    
    plt.tight_layout()
    filepath = f'{output_dir}/figD_volatility_streakiness.png'
    save_figure(fig, filepath)
    files.append(filepath)
    
    # Save chart data
    if len(sim_bench) > 0:
        sim_bench.to_csv(f'{output_dir}/tableD_simulation_benchmark.csv', index=False)
    perm_streak.to_csv(f'{output_dir}/tableD_permutation_streakiness.csv', index=False)
    
    return files


def plot_pooled_vs_within_league(results: Dict, output_dir: str = '.') -> List[str]:
    """
    Figure E: Pooled vs within-league relationships — coefficient forest plot.
    
    Forest plot with confidence intervals + one illustrative faceted scatterplot.
    """
    files = []
    
    pooled = results['pooled']
    within_league = results['within_league']
    adjusted = results['league_season_adjusted']
    heterogeneity = results['heterogeneity']
    fe_reg = results['fixed_effects_regression']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel 1: Forest plot of correlations
    ax = axes[0, 0]
    
    metrics = list(pooled.keys())
    y_pos = np.arange(len(metrics))
    
    # Pooled estimates
    pooled_rs = [pooled[m]['r'] for m in metrics]
    pooled_ns = [pooled[m]['n'] for m in metrics]
    
    # Within-league range
    within_ranges = []
    for m in metrics:
        league_rs = [within_league[lg].get(m, {}).get('r', np.nan) for lg in within_league.keys()]
        league_rs = [r for r in league_rs if not np.isnan(r)]
        if league_rs:
            within_ranges.append((min(league_rs), max(league_rs)))
        else:
            within_ranges.append((np.nan, np.nan))
    
    # Adjusted estimates
    adjusted_rs = [adjusted.get(m, {}).get('r', np.nan) for m in metrics]
    
    ax.scatter(pooled_rs, y_pos, color=PALETTE[0], s=60, label='Pooled', zorder=5)
    ax.scatter(adjusted_rs, y_pos + 0.2, color=PALETTE[1], s=60, marker='s', label='League-Season Adjusted', zorder=5)
    
    # Within-league ranges
    for i, (lo, hi) in enumerate(within_ranges):
        if not np.isnan(lo):
            ax.plot([lo, hi], [i, i], color=PALETTE[2], linewidth=3, alpha=0.5, zorder=3)
    
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_yticks(y_pos + 0.1)
    ax.set_yticklabels(metrics, fontsize=9)
    ax.set_xlabel('Correlation (r)')
    ax.set_title('A. Pooled vs Within-League vs Adjusted Correlations')
    ax.legend(fontsize=8)
    ax.set_xlim(-0.8, 0.8)
    
    # Panel 2: Heterogeneity I² statistics
    ax = axes[0, 1]
    if heterogeneity:
        het_labels = list(heterogeneity.keys())
        I2_values = [heterogeneity[m]['I2'] for m in het_labels]
        p_het = [heterogeneity[m]['p_heterogeneity'] for m in het_labels]
        
        colors = ['red' if p < 0.05 else 'gray' for p in p_het]
        bars = ax.barh(range(len(het_labels)), I2_values, color=colors, alpha=0.7)
        ax.set_yticks(range(len(het_labels)))
        ax.set_yticklabels(het_labels, fontsize=9)
        ax.set_xlabel('I² Heterogeneity (%)')
        ax.set_title('B. Cross-League Heterogeneity (I²)\n(Red = Significant Heterogeneity, p<0.05)')
        ax.set_xlim(0, 100)
        
        # Annotate
        for i, (I2, p) in enumerate(zip(I2_values, p_het)):
            ax.text(I2 + 2, i, f'I²={I2:.0f}%, p={p:.3f}', va='center', fontsize=8)
    
    # Panel 3: Fixed effects regression coefficients
    ax = axes[1, 0]
    if fe_reg:
        reg_labels = list(fe_reg.keys())
        coefs = [fe_reg[m]['coef'] for m in reg_labels]
        # Standard errors not directly available; approximate from r2 and n
        # This is a simplified visualization
        y_pos = np.arange(len(reg_labels))
        ax.barh(y_pos, coefs, color=PALETTE[3], alpha=0.7)
        ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(reg_labels, fontsize=9)
        ax.set_xlabel('Coefficient (League-Season FE)')
        ax.set_title('C. Fixed Effects Regression Coefficients')
    
    # Panel 4: Illustrative faceted scatterplot (Shots_PG vs PPG by league)
    ax = axes[1, 1]
    # This would need the original team_season data; for now, show league correlations
    if within_league:
        leagues = list(within_league.keys())
        metric = 'Shot Volume -> Points'
        if metric in within_league[leagues[0]]:
            league_rs = [within_league[lg].get(metric, {}).get('r', np.nan) for lg in leagues]
            league_ns = [within_league[lg].get(metric, {}).get('n', 0) for lg in leagues]
            
            y_pos = np.arange(len(leagues))
            colors = [PALETTE[i % len(PALETTE)] for i in range(len(leagues))]
            bars = ax.barh(y_pos, league_rs, color=colors, alpha=0.7)
            ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([f'{lg} (n={n})' for lg, n in zip(leagues, league_ns)], fontsize=9)
            ax.set_xlabel('Within-League Correlation (r)')
            ax.set_title('D. Shot Volume -> Points by League')
            ax.set_xlim(-0.2, 0.8)
    
    plt.tight_layout()
    filepath = f'{output_dir}/figE_pooled_vs_within_league.png'
    save_figure(fig, filepath)
    files.append(filepath)
    
    # Save chart data
    pd.DataFrame([
        {'metric': m, 'pooled_r': pooled[m]['r'], 'pooled_n': pooled[m]['n'],
         'adjusted_r': adjusted.get(m, {}).get('r', np.nan),
         'I2': heterogeneity.get(m, {}).get('I2', np.nan),
         'p_heterogeneity': heterogeneity.get(m, {}).get('p_heterogeneity', np.nan)}
        for m in metrics
    ]).to_csv(f'{output_dir}/tableE_correlation_comparison.csv', index=False)
    
    return files


def plot_home_advantage_changes(results: Dict, output_dir: str = '.') -> List[str]:
    """
    Figure F: Home advantage changes across seasons.
    
    League small multiples + change-estimate forest plot.
    Replace equal-thirds benchmark with symmetry reference.
    """
    files = []
    
    ha_ls = results['league_season']
    league_trends = results['league_trends']
    balanced_ha = results['balanced_sample']
    covid_comp = results['covid_comparison']
    team_ha_trends = results['team_ha_trends']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel 1: League small multiples - Home Win% over seasons
    ax = axes[0, 0]
    leagues = sorted(ha_ls['Div'].unique())
    for i, league in enumerate(leagues):
        lg_data = ha_ls[ha_ls['Div'] == league].sort_values('Season')
        ax.plot(lg_data['Season'], lg_data['Home_Win_Pct'], 'o-', 
                color=PALETTE[i % len(PALETTE)], label=league, alpha=0.8)
    # Symmetry reference
    ha_ls['Symmetry_Ref'] = (1 - ha_ls['Draw_Pct']) * 0.5
    sym_ref = ha_ls.groupby('Season')['Symmetry_Ref'].mean().reset_index()
    ax.plot(sym_ref['Season'], sym_ref['Symmetry_Ref'], 'k--', alpha=0.5, label='Symmetry Null', linewidth=2)
    
    ax.set_ylabel('Home Win %')
    ax.set_xlabel('Season')
    ax.set_title('A. Home Win% by League & Season (with Symmetry Null)')
    ax.legend(fontsize=7, ncol=2)
    ax.tick_params(axis='x', rotation=45)
    
    # Panel 2: League small multiples - HA Win Diff over seasons
    ax = axes[0, 1]
    for i, league in enumerate(leagues):
        lg_data = ha_ls[ha_ls['Div'] == league].sort_values('Season')
        ax.plot(lg_data['Season'], lg_data['HA_Win_Diff'], 's-', 
                color=PALETTE[i % len(PALETTE)], label=league, alpha=0.8)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylabel('Home Win% − Away Win%')
    ax.set_xlabel('Season')
    ax.set_title('B. Home Advantage (Win% Diff) by League & Season')
    ax.legend(fontsize=7, ncol=2)
    ax.tick_params(axis='x', rotation=45)
    
    # Panel 3: Forest plot of league trends
    ax = axes[1, 0]
    trend_metrics = ['Home_Win_Pct', 'HA_Win_Diff', 'Home_Win_Share_Decisive', 'Goal_Diff']
    trend_labels = []
    slopes = []
    p_values = []
    colors_trend = []
    
    for league in leagues:
        for metric in trend_metrics:
            key = f'{league}_{metric}'
            if key in league_trends:
                trend_labels.append(f'{league}: {metric}')
                slopes.append(league_trends[key]['slope_per_season'])
                p_values.append(league_trends[key]['p_value'])
                colors_trend.append('red' if league_trends[key]['p_value'] < 0.05 else 'gray')
    
    if trend_labels:
        y_pos = np.arange(len(trend_labels))
        ax.barh(y_pos, slopes, color=colors_trend, alpha=0.7)
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(trend_labels, fontsize=7)
        ax.set_xlabel('Trend Slope per Season')
        ax.set_title('C. League-Specific HA Trends (Red = p<0.05)')
    
    # Panel 4: COVID comparison + balanced sample
    ax = axes[1, 1]
    # Pre/COVID/Post comparison
    metrics = ['Home_Win_Pct', 'HA_Win_Diff', 'Home_Win_Share_Decisive', 'Goal_Diff']
    periods = ['pre_covid', 'covid_season', 'post_covid']
    x_pos = np.arange(len(metrics))
    width = 0.25
    
    for i, period in enumerate(periods):
        vals = [covid_comp[m][period] for m in metrics]
        ax.bar(x_pos + (i-1)*width, vals, width, label=period.replace('_', ' ').title(), alpha=0.8)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Home Win%', 'HA Win Diff', 'Decisive Share', 'Goal Diff'], fontsize=8)
    ax.set_ylabel('Value')
    ax.set_title('D. Pre-COVID / COVID / Post-COVID Comparison')
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    filepath = f'{output_dir}/figF_home_advantage_changes.png'
    save_figure(fig, filepath)
    files.append(filepath)
    
    # Save chart data
    ha_ls.to_csv(f'{output_dir}/tableF_league_season_ha.csv', index=False)
    balanced_ha.to_csv(f'{output_dir}/tableF_balanced_sample_ha.csv', index=False)
    
    return files


def create_advanced_figures(results_A: Dict, results_B: Dict, results_C: Dict,
                           results_D: Dict, results_E: Dict, results_F: Dict,
                           output_dir: str = '.') -> List[str]:
    """Create all advanced insight figures."""
    all_files = []
    
    all_files.extend(plot_finishing_persistence(results_A, output_dir))
    all_files.extend(plot_team_profile_changes(results_B, output_dir))
    all_files.extend(plot_halftime_deficit_recovery(results_C, output_dir))
    all_files.extend(plot_volatility_streakiness(results_D, output_dir))
    all_files.extend(plot_pooled_vs_within_league(results_E, output_dir))
    all_files.extend(plot_home_advantage_changes(results_F, output_dir))
    
    return all_files