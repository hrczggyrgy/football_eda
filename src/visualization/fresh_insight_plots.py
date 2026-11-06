"""
Fresh Insights Visualization Module
Publication-quality figures for the 5 new research questions.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import entropy
from typing import Dict, List, Optional, Union

import polars as pl

COLORS = {'Home': '#2E86AB', 'Away': '#A23B72', 'Draw': '#F18F01', 'Leading': '#2ca02c', 'Drawing': '#1f77b4', 'Trailing': '#d62728'}
PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
LEAGUE_COLORS = {'E0': '#1f77b4', 'E1': '#ff7f0e', 'E2': '#2ca02c', 'E3': '#d62728',
                 'SP1': '#9467bd', 'D1': '#8c564b', 'I1': '#e377c2', 'F1': '#7f7f7f'}


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


def plot_draw_specialists(results: Dict, figures_dir: str = '.', tables_dir: str = '.') -> List[str]:
    """
    Figure 1: Draw Specialists and Match-Outcome Structure.
    
    Panel A: Observed vs Expected Draw Rate (scatter, colored by PPG)
    Panel B: Draw Residual Distribution + Top/Bottom Teams
    Panel C: Season-to-Season Persistence of Draw Residual
    Panel D: Profile Comparison (High vs Low Draw Residual Teams)
    """
    files = []
    
    draw_data = results['team_season_draws']
    persistence = results.get('season_to_season_persistence', {})
    profile_diff = results.get('profile_differences', {})
    high_draw = pd.DataFrame(results.get('high_draw_profile', []))
    low_draw = pd.DataFrame(results.get('low_draw_profile', []))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel A: Observed vs Expected Draw Rate
    ax = axes[0, 0]
    valid = draw_data['Draw_Pct'].notna() & draw_data['Expected_Draw_Pct'].notna() & draw_data['PPG'].notna()
    if valid.sum() > 0:
        scatter = ax.scatter(draw_data.loc[valid, 'Expected_Draw_Pct'], 
                            draw_data.loc[valid, 'Draw_Pct'],
                            c=draw_data.loc[valid, 'PPG'], cmap='RdYlGn', alpha=0.6, s=20)
        ax.plot([0, 0.5], [0, 0.5], 'k--', alpha=0.5, label='Expected = Observed')
        ax.set_xlabel('Expected Draw Rate (from PPG)')
        ax.set_ylabel('Observed Draw Rate')
        ax.set_title('A. Observed vs Expected Draw Rate\n(colored by PPG)', fontsize=11, fontweight='bold')
        plt.colorbar(scatter, ax=ax, label='Season PPG')
        ax.legend(fontsize=8)
    
    # Panel B: Draw Residual Distribution
    ax = axes[0, 1]
    residuals = draw_data['Draw_Residual'].dropna()
    ax.hist(residuals, bins=40, alpha=0.7, color='gray', edgecolor='black', density=True)
    ax.axvline(x=0, color='black', linewidth=1)
    ax.axvline(x=residuals.mean(), color='red', linestyle='--', 
               label=f'Mean = {residuals.mean():.4f}')
    ax.set_xlabel('Draw Residual (Observed - Expected)')
    ax.set_ylabel('Density')
    ax.set_title('B. Draw Residual Distribution', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    
    # Annotate top/bottom teams
    if len(high_draw) > 0 and len(low_draw) > 0:
        for _, row in high_draw.head(3).iterrows():
            ax.annotate(f"{row['Team']} ({row['Draw_Residual']:.3f})", 
                       (row['Draw_Residual'], ax.get_ylim()[1]*0.9),
                       ha='center', fontsize=7, color='green', fontweight='bold', rotation=90)
        for _, row in low_draw.head(3).iterrows():
            ax.annotate(f"{row['Team']} ({row['Draw_Residual']:.3f})", 
                       (row['Draw_Residual'], ax.get_ylim()[1]*0.1),
                       ha='center', fontsize=7, color='red', fontweight='bold', rotation=90)
    
    # Panel C: Persistence
    ax = axes[1, 0]
    if 'Draw_Residual' in persistence:
        pers = persistence['Draw_Residual']
        consecutive = results['team_season_draws']  # This has the consecutive data
        # Need to reconstruct from results
        # For now, show persistence as bar
        metrics = ['Draw_Residual', 'Draw_Pct', 'Win_Pct', 'Loss_Pct']
        r_vals = []
        labels = []
        for m in metrics:
            if m in persistence:
                r_vals.append(persistence[m]['r'])
                labels.append(m.replace('_', ' '))
        if r_vals:
            colors_bar = ['green' if r > 0 else 'red' for r in r_vals]
            ax.barh(range(len(r_vals)), r_vals, color=colors_bar, alpha=0.7, edgecolor='black')
            ax.set_yticks(range(len(r_vals)))
            ax.set_yticklabels(labels)
            ax.set_xlabel('Season-to-Season Persistence (r)')
            ax.set_title('C. Persistence of Draw/Win/Loss Rates', fontsize=11, fontweight='bold')
            ax.axvline(x=0, color='black', linewidth=0.5)
            for i, r in enumerate(r_vals):
                ax.text(r + 0.02 if r >= 0 else r - 0.02, i, f'r={r:.3f}', 
                       va='center', ha='left' if r >= 0 else 'right', fontsize=9)
    
    # Panel D: Profile Comparison
    ax = axes[1, 1]
    if profile_diff:
        metrics = list(profile_diff.keys())
        high_means = [profile_diff[m]['high_mean'] for m in metrics]
        low_means = [profile_diff[m]['low_mean'] for m in metrics]
        p_vals = [profile_diff[m]['p'] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        bars1 = ax.bar(x - width/2, high_means, width, label='High Draw Residual', 
                      color='green', alpha=0.7, edgecolor='black')
        bars2 = ax.bar(x + width/2, low_means, width, label='Low Draw Residual', 
                      color='red', alpha=0.7, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ') for m in metrics], rotation=45, ha='right')
        ax.set_ylabel('Mean Value')
        ax.set_title('D. Profile: High vs Low Draw Residual Teams', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        
        # Significance stars
        for i, p in enumerate(p_vals):
            if p < 0.05:
                ax.text(i, max(high_means[i], low_means[i]) * 1.02, '*', ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    filepath = f'{figures_dir}/fig1_draw_specialists.png'
    save_figure(fig, filepath)
    files.append(filepath)
    
    # Save chart data
    draw_data.to_csv(f'{tables_dir}/table1_draw_specialists.csv', index=False)
    if 'league_draw_rates' in results:
        results['league_draw_rates'].to_csv(f'{tables_dir}/table1_league_draw_rates.csv', index=False)
    if 'team_pythag_career' in results:
        results['team_pythag_career'].to_csv(f'{tables_dir}/table1_team_draw_career.csv', index=False)
    
    return files


def plot_match_tempo(results: Dict, figures_dir: str = '.', tables_dir: str = '.') -> List[str]:
    """
    Figure 2: Match Tempo and Statistical Structure.
    
    Panel A: Outcome Distribution by Tempo Quintile
    Panel B: Goal Variance and Result Entropy by Tempo
    Panel C: Team-Season Tempo Profile (Avg Tempo vs PPG)
    Panel D: Tempo Persistence and High/Low Tempo Team Profiles
    """
    files = []
    
    tempo_outcomes = results['tempo_outcomes']
    tempo_entropy = results['tempo_entropy']
    team_tempo = results['team_season_tempo']
    league_tempo = results['league_tempo']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel A: Outcome Distribution by Tempo
    ax = axes[0, 0]
    x = np.arange(len(tempo_outcomes))
    width = 0.25
    ax.bar(x - width, tempo_outcomes['Home_Win'], width, label='Home Win', color=COLORS['Home'], alpha=0.8)
    ax.bar(x, tempo_outcomes['Draw'], width, label='Draw', color=COLORS['Draw'], alpha=0.8)
    ax.bar(x + width, tempo_outcomes['Away_Win'], width, label='Away Win', color=COLORS['Away'], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tempo_outcomes['Tempo_Quintile'])
    ax.set_ylabel('Proportion')
    ax.set_title('A. Match Outcome by Tempo Quintile', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    
    # Panel B: Goal Variance and Entropy by Tempo
    ax = axes[0, 1]
    # Merge entropy and variance
    tempo_merged = tempo_outcomes.merge(tempo_entropy, on='Tempo_Quintile')
    if 'GD_Variance' in results:
        tempo_merged = tempo_merged.merge(results['GD_Variance'], on='Tempo_Quintile')
    
    ax2 = ax.twinx()
    line1 = ax.plot(tempo_merged['Tempo_Quintile'], tempo_merged['Result_Entropy'], 
                   'o-', color='#1f77b4', linewidth=2, label='Result Entropy')
    if 'GD_Variance' in tempo_merged.columns:
        line2 = ax2.plot(tempo_merged['Tempo_Quintile'], tempo_merged['GD_Variance'], 
                        's-', color='#d62728', linewidth=2, label='Goal Diff Variance')
    else:
        line2 = ax2.plot(tempo_merged['Tempo_Quintile'], tempo_merged['Goal_Variance'], 
                        's-', color='#d62728', linewidth=2, label='Goal Variance')
    
    ax.set_ylabel('Result Entropy', color='#1f77b4')
    ax2.set_ylabel('Goal Variance', color='#d62728')
    ax.set_title('B. Unpredictability by Match Tempo', fontsize=11, fontweight='bold')
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left', fontsize=8)
    
    # Panel C: Team-Season Tempo vs PPG
    ax = axes[1, 0]
    valid = team_tempo['Avg_Tempo'].notna() & team_tempo['PPG'].notna()
    if valid.sum() > 0:
        scatter = ax.scatter(team_tempo.loc[valid, 'Avg_Tempo'], 
                            team_tempo.loc[valid, 'PPG'],
                            c=team_tempo.loc[valid, 'Matches'], cmap='viridis', alpha=0.6, s=20)
        z = np.polyfit(team_tempo.loc[valid, 'Avg_Tempo'], team_tempo.loc[valid, 'PPG'], 1)
        p = np.poly1d(z)
        x_range = np.linspace(team_tempo.loc[valid, 'Avg_Tempo'].min(), 
                             team_tempo.loc[valid, 'Avg_Tempo'].max(), 100)
        ax.plot(x_range, p(x_range), 'r--', alpha=0.8)
        
        r = results.get('tempo_ppg_correlation', {}).get('r', 0)
        ax.set_xlabel('Average Match Tempo (Total Shots)')
        ax.set_ylabel('Points per Game')
        ax.set_title(f'C. Team Tempo vs PPG (r={r:.3f})', fontsize=11, fontweight='bold')
        plt.colorbar(scatter, ax=ax, label='Matches')
    
    # Panel D: Tempo Persistence + High/Low Profiles
    ax = axes[1, 1]
    
    # Persistence bars
    pers_metrics = []
    pers_values = []
    pers_labels = []
    
    if 'tempo_persistence' in results:
        pers_metrics.append('Season-to-Season')
        pers_values.append(results['tempo_persistence']['r'])
        pers_labels.append(f"Season (r={results['tempo_persistence']['r']:.3f})")
    
    if 'half_season_tempo_persistence' in results:
        pers_metrics.append('Half-Season')
        pers_values.append(results['half_season_tempo_persistence']['r'])
        pers_labels.append(f"Half (r={results['half_season_tempo_persistence']['r']:.3f})")
    
    if pers_metrics:
        colors_p = [PALETTE[i] for i in range(len(pers_metrics))]
        ax.barh(range(len(pers_metrics)), pers_values, color=colors_p, alpha=0.7, edgecolor='black')
        ax.set_yticks(range(len(pers_metrics)))
        ax.set_yticklabels(pers_labels)
        ax.set_xlabel('Persistence (r)')
        ax.set_title('D. Tempo Persistence', fontsize=11, fontweight='bold')
        ax.axvline(x=0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    filepath = f'{figures_dir}/fig2_match_tempo.png'
    save_figure(fig, filepath)
    files.append(filepath)
    
    # League tempo comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(league_tempo))
    ax.bar(x, league_tempo['Avg_Tempo'], yerr=league_tempo['Tempo_Std'], 
           color=[LEAGUE_COLORS.get(d, 'gray') for d in league_tempo['Div']], 
           alpha=0.7, edgecolor='black', capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(league_tempo['Div'])
    ax.set_ylabel('Average Total Shots per Match')
    ax.set_title('Average Match Tempo by League', fontsize=12, fontweight='bold')
    plt.tight_layout()
    filepath = f'{figures_dir}/fig2b_league_tempo.png'
    save_figure(fig, filepath)
    files.append(filepath)
    
    # Save chart data
    tempo_outcomes.to_csv(f'{tables_dir}/table2_tempo_outcomes.csv', index=False)
    tempo_entropy.to_csv(f'{tables_dir}/table2_tempo_entropy.csv', index=False)
    team_tempo.to_csv(f'{tables_dir}/table2_team_season_tempo.csv', index=False)
    league_tempo.to_csv(f'{tables_dir}/table2_league_tempo.csv', index=False)
    
    return files


def plot_game_state_effects(results: Dict, figures_dir: str = '.', tables_dir: str = '.') -> List[str]:
    """
    Figure 3: Game State Effects on Match Statistics.
    
    Panel A: Score Effects by Metric (Trailing - Leading) by Venue
    Panel B: Deficit Size Gradient (Shots, Fouls, Corners)
    Panel C: League Differences in Score Effects
    Panel D: Score Effect Persistence (Season-to-Season)
    """
    files = []
    
    state_stats = results['state_stats']
    deficit_stats = results['deficit_stats']
    league_effects = results['league_score_effects']
    persistence = results.get('score_effect_persistence', {})
    strength_effects = results.get('strength_score_effects', pd.DataFrame())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel A: Score Effects by Venue
    ax = axes[0, 0]
    metrics = ['Shots', 'Shots_OT', 'Fouls', 'Corners', 'Yellow']
    venues = ['Home', 'Away']
    
    x = np.arange(len(metrics))
    width = 0.35
    
    for i, venue in enumerate(venues):
        venue_data = state_stats[state_stats['Venue'] == venue]
        # Exclude matches with a missing halftime result (HT_Result is null);
        # a venue may carry more than 3 rows across leagues/seasons, so do not
        # gate on an exact row count -- only require both leading and trailing
        # states to be present.
        leading = venue_data[venue_data['HT_Result'] == 'W']
        trailing = venue_data[venue_data['HT_Result'] == 'L']
        if len(leading) > 0 and len(trailing) > 0:
            effects = []
            for m in metrics:
                lead_val = leading[m].values[0] if m in leading.columns else 0
                trail_val = trailing[m].values[0] if m in trailing.columns else 0
                effects.append(trail_val - lead_val)
            ax.bar(x + (i - 0.5) * width, effects, width, label=venue, alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ') for m in metrics])
    ax.set_ylabel('Score Effect (Trailing - Leading)')
    ax.set_title('A. Score Effects by Venue', fontsize=11, fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.legend(fontsize=8)
    
    # Panel B: Deficit Size Gradient
    ax = axes[0, 1]
    if len(deficit_stats) > 0:
        for venue in ['Home', 'Away']:
            vd = deficit_stats[deficit_stats['Venue'] == venue].sort_values('Deficit')
            if len(vd) > 0:
                ax.plot(vd['Deficit'], vd['Shots'], 'o-', label=f'{venue} Shots', alpha=0.8)
                ax.plot(vd['Deficit'], vd['Fouls'], 's--', label=f'{venue} Fouls', alpha=0.6)
                ax.plot(vd['Deficit'], vd['Corners'], '^--', label=f'{venue} Corners', alpha=0.6)
        
        ax.set_xlabel('Halftime Deficit (Goals)')
        ax.set_ylabel('Average per Match')
        ax.set_title('B. Statistics by Deficit Size', fontsize=11, fontweight='bold')
        ax.legend(fontsize=7, ncol=2)
    
    # Panel C: League Differences in Score Effects
    ax = axes[1, 0]
    if len(league_effects) > 0:
        metric = 'Score_Effect_Shots_mean'
        if metric in league_effects.columns:
            league_effects_sorted = league_effects.sort_values(metric)
            colors_l = [LEAGUE_COLORS.get(d, 'gray') for d in league_effects_sorted['Div']]
            ax.barh(range(len(league_effects_sorted)), league_effects_sorted[metric], 
                   color=colors_l, alpha=0.7, edgecolor='black')
            ax.set_yticks(range(len(league_effects_sorted)))
            ax.set_yticklabels(league_effects_sorted['Div'])
            ax.set_xlabel('Score Effect: Trailing Shots - Leading Shots')
            ax.set_title('C. League Differences in Shot Score Effect', fontsize=11, fontweight='bold')
            ax.axvline(x=0, color='black', linewidth=0.5)
    
    # Panel D: Persistence + Strength Interaction
    ax = axes[1, 1]
    
    # Persistence bars
    if persistence:
        metrics_p = list(persistence.keys())
        r_vals = [persistence[m]['r'] for m in metrics_p]
        colors_p = [PALETTE[i] for i in range(len(metrics_p))]
        ax.barh(range(len(metrics_p)), r_vals, color=colors_p, alpha=0.7, edgecolor='black')
        ax.set_yticks(range(len(metrics_p)))
        ax.set_yticklabels([f"{m} (r={r:.3f})" for m, r in zip(metrics_p, r_vals)])
        ax.set_xlabel('Season-to-Season Persistence (r)')
        ax.set_title('D. Score Effect Persistence', fontsize=11, fontweight='bold')
        ax.axvline(x=0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    filepath = f'{figures_dir}/fig3_game_state_effects.png'
    save_figure(fig, filepath)
    files.append(filepath)
    
    # Strength interaction plot
    if len(strength_effects) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics_s = ['Shots', 'Shots_OT', 'Fouls', 'Corners', 'Yellow']
        x = np.arange(len(metrics_s))
        width = 0.25
        for i, strength in enumerate(['Weak', 'Average', 'Strong']):
            se = strength_effects[strength_effects['Strength_Tercile'] == strength]
            if len(se) > 0:
                vals = [se[f'Score_Effect_{m}_mean'].values[0] if f'Score_Effect_{m}_mean' in se.columns else 0 
                       for m in metrics_s]
                ax.bar(x + (i - 1) * width, vals, width, label=strength, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ') for m in metrics_s])
        ax.set_ylabel('Score Effect (Trailing - Leading)')
        ax.set_title('Score Effects by Pre-Match Team Strength', fontsize=12, fontweight='bold')
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.legend()
        plt.tight_layout()
        filepath = f'{figures_dir}/fig3b_strength_score_effects.png'
        save_figure(fig, filepath)
        files.append(filepath)
    
    # Save chart data
    state_stats.to_csv(f'{tables_dir}/table3_state_stats.csv', index=False)
    deficit_stats.to_csv(f'{tables_dir}/table3_deficit_stats.csv', index=False)
    league_effects.to_csv(f'{tables_dir}/table3_league_score_effects.csv', index=False)
    if len(strength_effects) > 0:
        strength_effects.to_csv(f'{tables_dir}/table3_strength_score_effects.csv', index=False)
    
    return files


def plot_referee_characteristics(results: Dict, figures_dir: str = '.', tables_dir: str = '.') -> List[str]:
    """
    Figure 4: Referee Characteristics and Potential Biases (English Leagues).
    
    Panel A: Referee Career Averages (Yellow PG, Red PG, Fouls PG, HA Win Diff)
    Panel B: Referee Home Bias (Yellow, Red, Fouls)
    Panel C: Referee Persistence (Season-to-Season)
    Panel D: Extreme Referees (Top/Bottom 5 by Key Metrics)
    """
    files = []
    
    ref_career = results.get('referee_career', pd.DataFrame())
    ref_season = results.get('referee_season', pd.DataFrame())
    persistence = results.get('referee_persistence', {})
    
    if len(ref_career) == 0:
        return files
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel A: Career averages distribution
    ax = axes[0, 0]
    metrics_a = ['Avg_Yellow_PG', 'Avg_Red_PG', 'Avg_Fouls_PG', 'Avg_HA_Win_Diff']
    labels_a = ['Yellow PG', 'Red PG', 'Fouls PG', 'HA Win Diff']
    
    for i, (metric, label) in enumerate(zip(metrics_a, labels_a)):
        if metric in ref_career.columns:
            ax.hist(ref_career[metric].dropna(), bins=20, alpha=0.5, label=label, density=True)
    
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title('A. Referee Career Metric Distributions', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    
    # Panel B: Home Bias
    ax = axes[0, 1]
    bias_metrics = ['Avg_Yellow_Home_Bias', 'Avg_Red_Home_Bias', 'Avg_Fouls_Home_Bias']
    bias_labels = ['Yellow Bias', 'Red Bias', 'Fouls Bias']
    
    for i, (metric, label) in enumerate(zip(bias_metrics, bias_labels)):
        if metric in ref_career.columns:
            data = ref_career[metric].dropna()
            ax.hist(data, bins=20, alpha=0.5, label=label, density=True)
            ax.axvline(x=data.mean(), color=PALETTE[i], linestyle='--', alpha=0.8)
    
    ax.set_xlabel('Home - Away Difference')
    ax.set_ylabel('Density')
    ax.set_title('B. Referee Home Bias Distributions', fontsize=11, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.legend(fontsize=8)
    
    # Panel C: Persistence
    ax = axes[1, 0]
    if persistence:
        metrics_p = list(persistence.keys())
        r_vals = [persistence[m]['r'] for m in metrics_p]
        p_vals = [persistence[m]['p'] for m in metrics_p]
        colors_p = ['green' if p < 0.05 else 'gray' for p in p_vals]
        
        ax.barh(range(len(metrics_p)), r_vals, color=colors_p, alpha=0.7, edgecolor='black')
        ax.set_yticks(range(len(metrics_p)))
        ax.set_yticklabels([m.replace('_', ' ') for m in metrics_p])
        ax.set_xlabel('Season-to-Season Persistence (r)')
        ax.set_title('C. Referee Metric Persistence', fontsize=11, fontweight='bold')
        ax.axvline(x=0, color='black', linewidth=0.5)
        for i, (r, p) in enumerate(zip(r_vals, p_vals)):
            ax.text(r + 0.02 if r >= 0 else r - 0.02, i, f'r={r:.3f} (p={p:.3f})', 
                   va='center', ha='left' if r >= 0 else 'right', fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No persistence data\n(insufficient consecutive-season referee overlaps)',
                transform=ax.transAxes, ha='center', va='center', fontsize=9, color='gray')
        ax.set_title('C. Referee Metric Persistence', fontsize=11, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Panel D: Extreme Referees Table
    ax = axes[1, 1]
    ax.axis('off')
    
    table_data = []
    table_data.append(['Metric', 'Top 3 (High)', '', '', 'Bottom 3 (Low)', '', ''])
    
    for metric in ['Avg_Yellow_PG', 'Avg_Red_PG', 'Avg_Fouls_PG', 'Avg_HA_Win_Diff']:
        top_key = f'top_{metric}'
        bottom_key = f'bottom_{metric}'
        if top_key in results and bottom_key in results:
            top = results[top_key]
            bottom = results[bottom_key]
            top_str = ', '.join([f"{r['Referee']} ({r[metric]:.2f})" for r in top[:3]])
            bottom_str = ', '.join([f"{r['Referee']} ({r[metric]:.2f})" for r in bottom[:3]])
            table_data.append([metric.replace('Avg_', '').replace('_PG', '/PG'), top_str, '', '', bottom_str, '', ''])
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='left',
                    colWidths=[0.22, 0.28, 0.05, 0.05, 0.28, 0.06, 0.06])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    ax.set_title('D. Extreme Referees (Career Averages)', fontsize=11, fontweight='bold', pad=20)
    
    plt.tight_layout()
    filepath = f'{figures_dir}/fig4_referee_characteristics.png'
    save_figure(fig, filepath)
    files.append(filepath)
    
    # Referee career scatter: Yellow vs HA Bias
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(ref_career['Avg_Yellow_PG'], ref_career['Avg_HA_Win_Diff'],
                        c=ref_career['Total_Matches'], cmap='viridis', alpha=0.7, s=50, edgecolor='white')
    ax.set_xlabel('Average Yellow Cards per Game')
    ax.set_ylabel('Home Win % - Away Win %')
    ax.set_title('Referee Card Tendency vs Home Advantage', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Total Matches')
    
    # Label extremes
    for _, row in ref_career.nlargest(3, 'Avg_Yellow_PG').iterrows():
        ax.annotate(row['Referee'], (row['Avg_Yellow_PG'], row['Avg_HA_Win_Diff']), fontsize=8)
    for _, row in ref_career.nsmallest(3, 'Avg_Yellow_PG').iterrows():
        ax.annotate(row['Referee'], (row['Avg_Yellow_PG'], row['Avg_HA_Win_Diff']), fontsize=8)
    
    plt.tight_layout()
    filepath = f'{figures_dir}/fig4b_referee_yellow_vs_ha.png'
    save_figure(fig, filepath)
    files.append(filepath)
    
    # Save chart data
    ref_career.to_csv(f'{tables_dir}/table4_referee_career.csv', index=False)
    ref_season.to_csv(f'{tables_dir}/table4_referee_season.csv', index=False)
    
    return files


def plot_pythagorean_residuals(results: Dict, figures_dir: str = '.', tables_dir: str = '.') -> List[str]:
    """
    Figure 5: Expected Points vs Actual Points (Pythagorean Residuals).
    
    Panel A: Pythagorean Residual vs PPG (colored by Shot Dominance)
    Panel B: Residual Distribution + Top/Bottom Overperformers
    Panel C: Season-to-Season Persistence (Pythag Residual vs Shot Dominance)
    Panel D: Incremental Predictive Value (GD vs Shot Dominance)
    """
    files = []
    
    ts = results['team_season_pythag']
    persistence = results.get('season_to_season_persistence', {})
    incremental = results.get('incremental_r2', {})
    team_career = results.get('team_pythag_career', pd.DataFrame())
    top_over = results.get('top_overperformers', [])
    top_under = results.get('top_underperformers', [])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    if 'Shot_Dominance_PG' not in ts.columns and {'Shots_PG', 'Shots_Against_PG'} <= set(ts.columns):
        ts = ts.assign(Shot_Dominance_PG=ts['Shots_PG'] - ts['Shots_Against_PG'])

    # Panel A: Residual vs PPG
    ax = axes[0, 0]
    valid = ts['Pythag_Residual'].notna() & ts['PPG'].notna() & ts['Shot_Dominance_PG'].notna()
    if valid.sum() > 0:
        scatter = ax.scatter(ts.loc[valid, 'PPG'], ts.loc[valid, 'Pythag_Residual'],
                            c=ts.loc[valid, 'Shot_Dominance_PG'], cmap='RdBu_r', alpha=0.6, s=20)
        ax.axhline(y=0, color='black', linewidth=0.8)
        ax.set_xlabel('Actual Points per Game')
        ax.set_ylabel('Pythagorean Residual (Actual - Expected)')
        ax.set_title('A. Pythagorean Residual vs PPG\n(color = Shot Dominance)', fontsize=11, fontweight='bold')
        plt.colorbar(scatter, ax=ax, label='Shot Dominance (Shots - Shots Against)')
    
    # Panel B: Residual Distribution
    ax = axes[0, 1]
    residuals = ts['Pythag_Residual'].dropna()
    ax.hist(residuals, bins=40, alpha=0.7, color='gray', edgecolor='black', density=True)
    ax.axvline(x=0, color='black', linewidth=1)
    ax.axvline(x=residuals.mean(), color='red', linestyle='--', 
               label=f'Mean = {residuals.mean():.4f}')
    ax.set_xlabel('Pythagorean Residual')
    ax.set_ylabel('Density')
    ax.set_title('B. Residual Distribution + Over/Underperformers', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    
    # Annotate top over/underperformers
    for i, team in enumerate(top_over[:3]):
        ax.annotate(f"{team['Team']} ({team['Avg_Pythag_Residual']:.3f})", 
                   (team['Avg_Pythag_Residual'], ax.get_ylim()[1] * (0.9 - i*0.05)),
                   ha='center', fontsize=8, color='green', fontweight='bold', rotation=90)
    for i, team in enumerate(top_under[:3]):
        ax.annotate(f"{team['Team']} ({team['Avg_Pythag_Residual']:.3f})", 
                   (team['Avg_Pythag_Residual'], ax.get_ylim()[1] * (0.1 + i*0.05)),
                   ha='center', fontsize=8, color='red', fontweight='bold', rotation=90)
    
    # Panel C: Persistence Comparison
    ax = axes[1, 0]
    pers_metrics = []
    pers_values = []
    pers_labels = []
    
    for metric in ['Pythag_Residual', 'PPG', 'Shot_Dominance_PG']:
        if metric in persistence:
            pers_metrics.append(metric)
            pers_values.append(persistence[metric]['r'])
            pers_labels.append(f"{metric.replace('_', ' ')} (r={persistence[metric]['r']:.3f})")
    
    if pers_metrics:
        colors_p = ['#1f77b4', '#2ca02c', '#ff7f0e']
        ax.barh(range(len(pers_metrics)), pers_values, color=colors_p, alpha=0.7, edgecolor='black')
        ax.set_yticks(range(len(pers_metrics)))
        ax.set_yticklabels(pers_labels)
        ax.set_xlabel('Season-to-Season Persistence (r)')
        ax.set_title('C. Persistence: Residual vs PPG vs Shot Dominance', fontsize=11, fontweight='bold')
        ax.axvline(x=0, color='black', linewidth=0.5)
    
    # Panel D: Incremental R²
    ax = axes[1, 1]
    if incremental:
        models = ['GD Only', 'GD + Shot Dom', 'Shot Dom Only']
        r2_vals = [incremental.get('gd_only', 0), 
                   incremental.get('gd_plus_shotdom', 0),
                   incremental.get('shotdom_only', 0)]
        colors_d = ['#1f77b4', '#2ca02c', '#ff7f0e']
        bars = ax.bar(models, r2_vals, color=colors_d, alpha=0.7, edgecolor='black')
        ax.set_ylabel('R² (Predicting Pythagorean Residual)')
        ax.set_title('D. Incremental Predictive Value', fontsize=11, fontweight='bold')
        for bar, r2 in zip(bars, r2_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                   f'{r2:.4f}', ha='center', va='bottom', fontweight='bold')
        
        inc = incremental.get('inc_shotdom_over_gd', 0)
        ax.annotate(f'ΔR² = {inc:.4f}', xy=(1, r2_vals[1]), 
                   xytext=(0, 15), textcoords='offset points', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    filepath = f'{figures_dir}/fig5_pythagorean_residuals.png'
    save_figure(fig, filepath)
    files.append(filepath)
    
    # League Pythagorean performance
    fig, ax = plt.subplots(figsize=(10, 6))
    league_pythag = results.get('league_pythag', pd.DataFrame())
    if len(league_pythag) > 0:
        league_pythag = league_pythag.sort_values('Mean_Pythag_Residual')
        colors_l = [LEAGUE_COLORS.get(d, 'gray') for d in league_pythag['Div']]
        ax.barh(range(len(league_pythag)), league_pythag['Mean_Pythag_Residual'],
               color=colors_l, alpha=0.7, edgecolor='black', xerr=league_pythag['Std_Pythag_Residual'])
        ax.set_yticks(range(len(league_pythag)))
        ax.set_yticklabels(league_pythag['Div'])
        ax.set_xlabel('Mean Pythagorean Residual (PPG)')
        ax.set_title('League Pythagorean Residual (Positive = Overperformance)', fontsize=12, fontweight='bold')
        ax.axvline(x=0, color='black', linewidth=0.5)
        plt.tight_layout()
        filepath = f'{figures_dir}/fig5b_league_pythag.png'
        save_figure(fig, filepath)
        files.append(filepath)
    
    # Save chart data
    ts.to_csv(f'{tables_dir}/table5_team_season_pythag.csv', index=False)
    if len(team_career) > 0:
        team_career.to_csv(f'{tables_dir}/table5_team_pythag_career.csv', index=False)
    league_pythag.to_csv(f'{tables_dir}/table5_league_pythag.csv', index=False)
    if incremental:
        pd.DataFrame([incremental]).to_csv(f'{tables_dir}/table5_incremental_r2.csv', index=False)
    
    return files


def create_all_fresh_figures(results: Dict, figures_dir: str = '.', tables_dir: str = '.') -> List[str]:
    """Create all 5 fresh insight figures.

    Polars inputs are converted to pandas at the plotting boundary.
    """
    results = _to_pandas(results)
    all_files = []
    
    all_files.extend(plot_draw_specialists(results['draw_specialists'], figures_dir, tables_dir))
    all_files.extend(plot_match_tempo(results['match_tempo'], figures_dir, tables_dir))
    all_files.extend(plot_game_state_effects(results['game_state_effects'], figures_dir, tables_dir))
    all_files.extend(plot_referee_characteristics(results['referee_characteristics'], figures_dir, tables_dir))
    all_files.extend(plot_pythagorean_residuals(results['pythagorean_residuals'], figures_dir, tables_dir))
    
    return all_files