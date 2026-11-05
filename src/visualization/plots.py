"""
Visualization module.
Creates publication-quality figures for the football analytics study.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Optional

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk", font_scale=1.1)

COLORS = {'Home': '#2E86AB', 'Away': '#A23B72', 'Draw': '#F18F01'}
CLUSTER_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']


def save_figure(fig, filepath: str, dpi: int = 150):
    """Save figure with consistent settings."""
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_home_advantage(ha_results: Dict, master: pd.DataFrame, output_dir: str = '.') -> List[str]:
    """Create Figure 1: Home advantage analysis."""
    files = []
    
    ha_ls = ha_results['by_league_season']
    ha_season = ha_results['by_season'].reset_index()
    ha_league = ha_results['by_league']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1a: Home win % by league — reference: symmetry null (home win share of decisive matches = 0.5)
    ax = axes[0, 0]
    sns.boxplot(data=ha_ls, x='Div', y='Home_Win_Pct', ax=ax, 
                order=sorted(ha_ls['Div'].unique()))
    # Symmetry reference: among decisive matches (non-draws), home win probability = 0.5
    # This line shows expected home win% if home/away symmetric given observed draw rate
    # For each league-season, expected = (1 - draw_rate) * 0.5
    # We plot the pooled average as a reference
    match_weighted_ref = ha_ls.apply(lambda r: (1 - (r['Draw_Pct'] if 'Draw_Pct' in r else 0.25)) * 0.5, axis=1).mean()
    ax.axhline(y=match_weighted_ref, color='red', linestyle='--', alpha=0.7, 
               label=f'Symmetry null (match-wtd) = {match_weighted_ref:.3f}')
    ax.set_title('Home Win Percentage by League-Season')
    ax.set_ylabel('Home Win %')
    ax.set_xlabel('League')
    ax.legend(fontsize=9)
    
    # 1b: Goal difference by league
    ax = axes[0, 1]
    sns.boxplot(data=ha_ls, x='Div', y='Goal_Diff', ax=ax,
                order=sorted(ha_ls['Div'].unique()))
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='No goal advantage')
    ax.set_title('Average Home Goal Advantage by League-Season')
    ax.set_ylabel('Home Goals - Away Goals per Match')
    ax.set_xlabel('League')
    ax.legend(fontsize=9)
    
    # 1c: Trend over seasons (match-weighted averages)
    ax = axes[1, 0]
    # Compute match-weighted seasonal averages
    season_avg = ha_ls.groupby('Season').apply(
        lambda g: pd.Series({
            'Home_Win_Pct': np.average(g['Home_Win_Pct'], weights=g['Matches']),
            'Away_Win_Pct': np.average(g['Away_Win_Pct'], weights=g['Matches']),
            'Draw_Pct': np.average(g['Draw_Pct'], weights=g['Matches']),
        })
    ).reset_index()
    ax.plot(season_avg['Season'], season_avg['Home_Win_Pct'], 'o-', label='Home Win %', color=COLORS['Home'])
    ax.plot(season_avg['Season'], season_avg['Away_Win_Pct'], 's-', label='Away Win %', color=COLORS['Away'])
    ax.plot(season_avg['Season'], season_avg['Draw_Pct'], '^-', label='Draw %', color=COLORS['Draw'])
    # Symmetry reference line (match-weighted per season)
    season_avg['symmetry_ref'] = (1 - season_avg['Draw_Pct']) * 0.5
    ax.plot(season_avg['Season'], season_avg['symmetry_ref'], '--', color='gray', alpha=0.7, 
            label='Symmetry null (decisive matches)')
    ax.set_title('Home/Away/Draw Trends Across Seasons (Match-Weighted)')
    ax.set_ylabel('Percentage')
    ax.set_xlabel('Season')
    ax.legend(fontsize=9)
    ax.tick_params(axis='x', rotation=45)
    
    # 1d: Team-level home advantage in Premier League (min 10 home matches)
    ax = axes[1, 1]
    pl_matches = master[(master['Div'] == 'E0')].copy()
    # Home perspective
    home_stats = pl_matches.groupby('HomeTeam').agg(
        Home_Matches=('FTR', 'count'),
        Home_Wins=('FTR', lambda x: (x == 'H').sum()),
        Home_Draws=('FTR', lambda x: (x == 'D').sum()),
        Home_Losses=('FTR', lambda x: (x == 'A').sum()),
        Home_Goals=('FTHG', 'mean'),
        Away_Goals=('FTAG', 'mean'),
    ).reset_index()
    home_stats = home_stats[home_stats['Home_Matches'] >= 10].copy()
    home_stats['Home_Win_Pct'] = home_stats['Home_Wins'] / home_stats['Home_Matches']
    home_stats['Home_Goal_Diff'] = home_stats['Home_Goals'] - home_stats['Away_Goals']
    
    # Away perspective for same teams
    away_stats = pl_matches.groupby('AwayTeam').agg(
        Away_Matches=('FTR', 'count'),
        Away_Wins=('FTR', lambda x: (x == 'A').sum()),
        Away_Draws=('FTR', lambda x: (x == 'D').sum()),
        Away_Losses=('FTR', lambda x: (x == 'H').sum()),
    ).reset_index()
    away_stats = away_stats.rename(columns={'AwayTeam': 'HomeTeam'})
    away_stats = away_stats[away_stats['Away_Matches'] >= 10]
    away_stats['Away_Win_Pct'] = away_stats['Away_Wins'] / away_stats['Away_Matches']
    
    # Merge
    team_ha = home_stats.merge(away_stats[['HomeTeam', 'Away_Matches', 'Away_Win_Pct']], on='HomeTeam', how='inner')
    team_ha = team_ha[team_ha['Away_Matches'] >= 10].copy()
    team_ha['HA_Win_Diff'] = team_ha['Home_Win_Pct'] - team_ha['Away_Win_Pct']
    team_ha = team_ha.sort_values('HA_Win_Diff', ascending=True)
    
    # Plot horizontal bar chart
    y_pos = np.arange(len(team_ha))
    colors = ['#2E86AB' if x > 0 else '#A23B72' for x in team_ha['HA_Win_Diff']]
    ax.barh(y_pos, team_ha['HA_Win_Diff'], color=colors, alpha=0.7, edgecolor='white', linewidth=0.5)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(team_ha['HomeTeam'], fontsize=7)
    ax.set_xlabel('Home Win% − Away Win% (percentage points)')
    ax.set_title('Premier League: Team Home Advantage (Win% Diff, min 10 H/A matches)')
    # Add reference line for league average
    league_avg_diff = team_ha['HA_Win_Diff'].mean()
    ax.axvline(x=league_avg_diff, color='gray', linestyle=':', alpha=0.7, 
               label=f'League avg = {league_avg_diff:.3f}')
    ax.legend(fontsize=8, loc='lower right')
    
    plt.tight_layout()
    filepath = f'{output_dir}/fig1_home_advantage.png'
    save_figure(fig, filepath)
    files.append(filepath)
    
    return files


def plot_efficiency_vs_points(team_season: pd.DataFrame, output_dir: str = '.') -> List[str]:
    """Create Figure 2: Efficiency metrics vs Points per Game."""
    files = []
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = [
        ('Goals_Per_Shot', 'Goals per Shot'),
        ('Goals_Per_Shot_OT', 'Goals per Shot on Target'),
        ('GA_Per_Opp_Shot', 'Goals Against per Opp Shot'),
        ('GA_Per_Opp_Shot_OT', 'Goals Against per Opp Shot on Target')
    ]
    
    for ax, (metric, label) in zip(axes.flat, metrics):
        if metric not in team_season.columns:
            continue
        valid = team_season[metric].notna()
        if valid.sum() < 10:
            continue
        
        ax.scatter(team_season.loc[valid, metric], team_season.loc[valid, 'PPG'], alpha=0.5, s=20)
        z = np.polyfit(team_season.loc[valid, metric], team_season.loc[valid, 'PPG'], 1)
        p = np.poly1d(z)
        x_range = np.linspace(team_season.loc[valid, metric].min(), team_season.loc[valid, metric].max(), 100)
        ax.plot(x_range, p(x_range), 'r--', alpha=0.8)
        r, _ = stats.pearsonr(team_season.loc[valid, metric], team_season.loc[valid, 'PPG'])
        ax.set_title(f'{label} vs Points per Game (r={r:.3f})')
        ax.set_xlabel(label)
        ax.set_ylabel('Points per Game')
    
    plt.tight_layout()
    filepath = f'{output_dir}/fig2_efficiency_vs_points.png'
    save_figure(fig, filepath)
    files.append(filepath)
    
    return files


def plot_efficiency_persistence(half_eff: pd.DataFrame, output_dir: str = '.') -> List[str]:
    """Create Figure 3: First vs Second half efficiency persistence."""
    files = []
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for ax, metric in zip(axes.flat, ['GPS', 'GPSOT', 'GAOS', 'GAOSOT']):
        first_col = f'{metric}_First'
        second_col = f'{metric}_Second'
        valid = half_eff[first_col].notna() & half_eff[second_col].notna()
        if valid.sum() < 10:
            continue
        
        ax.scatter(half_eff.loc[valid, first_col], half_eff.loc[valid, second_col], alpha=0.5, s=20)
        mn = min(half_eff.loc[valid, first_col].min(), half_eff.loc[valid, second_col].min())
        mx = max(half_eff.loc[valid, first_col].max(), half_eff.loc[valid, second_col].max())
        ax.plot([mn, mx], [mn, mx], 'r--', alpha=0.5)
        r, _ = stats.pearsonr(half_eff.loc[valid, first_col], half_eff.loc[valid, second_col])
        ax.set_title(f'{metric}: First vs Second Half (r={r:.3f})')
        ax.set_xlabel(f'First Half {metric}')
        ax.set_ylabel(f'Second Half {metric}')
    
    plt.tight_layout()
    filepath = f'{output_dir}/fig3_efficiency_persistence.png'
    save_figure(fig, filepath)
    files.append(filepath)
    
    return files


def plot_shot_decomposition(team_season: pd.DataFrame, output_dir: str = '.') -> List[str]:
    """Create Figure 4: Shot volume, accuracy, finishing decomposition."""
    files = []
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 4a: Shots per game vs PPG
    ax = axes[0, 0]
    ax.scatter(team_season['Shots_PG'], team_season['PPG'], alpha=0.5, s=20)
    z = np.polyfit(team_season['Shots_PG'], team_season['PPG'], 1)
    p = np.poly1d(z)
    ax.plot(np.linspace(5, 20, 100), p(np.linspace(5, 20, 100)), 'r--', alpha=0.8)
    r, _ = stats.pearsonr(team_season['Shots_PG'], team_season['PPG'])
    ax.set_title(f'Shots per Game vs PPG (r={r:.3f})')
    ax.set_xlabel('Shots per Game')
    ax.set_ylabel('Points per Game')
    
    # 4b: Shot Accuracy vs PPG
    ax = axes[0, 1]
    ax.scatter(team_season['Shot_Accuracy'], team_season['PPG'], alpha=0.5, s=20)
    z = np.polyfit(team_season['Shot_Accuracy'], team_season['PPG'], 1)
    p = np.poly1d(z)
    ax.plot(np.linspace(0.2, 0.5, 100), p(np.linspace(0.2, 0.5, 100)), 'r--', alpha=0.8)
    r, _ = stats.pearsonr(team_season['Shot_Accuracy'], team_season['PPG'])
    ax.set_title(f'Shot Accuracy vs PPG (r={r:.3f})')
    ax.set_xlabel('Shot Accuracy (SoT/Shots)')
    ax.set_ylabel('Points per Game')
    
    # 4c: Goals per Shot on Target vs PPG
    ax = axes[1, 0]
    ax.scatter(team_season['Goals_Per_Shot_OT'], team_season['PPG'], alpha=0.5, s=20)
    z = np.polyfit(team_season['Goals_Per_Shot_OT'], team_season['PPG'], 1)
    p = np.poly1d(z)
    ax.plot(np.linspace(0.1, 0.6, 100), p(np.linspace(0.1, 0.6, 100)), 'r--', alpha=0.8)
    r, _ = stats.pearsonr(team_season['Goals_Per_Shot_OT'], team_season['PPG'])
    ax.set_title(f'Goals per Shot on Target vs PPG (r={r:.3f})')
    ax.set_xlabel('Goals per Shot on Target')
    ax.set_ylabel('Points per Game')
    
    # 4d: 3D-like: Shots vs Accuracy colored by Finishing
    ax = axes[1, 1]
    sc = ax.scatter(team_season['Shots_PG'], team_season['Shot_Accuracy'], 
                    c=team_season['Goals_Per_Shot_OT'], cmap='RdYlGn', alpha=0.6, s=30)
    plt.colorbar(sc, ax=ax, label='Goals per Shot on Target')
    ax.set_title('Shot Volume vs Accuracy (color = Finishing)')
    ax.set_xlabel('Shots per Game')
    ax.set_ylabel('Shot Accuracy')
    
    plt.tight_layout()
    filepath = f'{output_dir}/fig4_shot_decomposition.png'
    save_figure(fig, filepath)
    files.append(filepath)
    
    return files


def plot_attack_defense_profiles(team_season: pd.DataFrame, shot_results: Dict, output_dir: str = '.') -> List[str]:
    """Create Figure 5: Attack-Defense profiles heatmaps."""
    files = []
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 5a: Attack profile
    if 'attack_profile_ppg' in shot_results:
        pivot = pd.DataFrame(shot_results['attack_profile_ppg'])
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[0], center=1.3)
        axes[0].set_title('PPG by Attack Volume and Finishing Quality')
        axes[0].set_xlabel('Finishing Quality')
        axes[0].set_ylabel('Shot Volume')
    
    # 5b: Defense profile
    if 'defense_profile_ppg' in shot_results:
        pivot2 = pd.DataFrame(shot_results['defense_profile_ppg'])
        sns.heatmap(pivot2, annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[1], center=1.3)
        axes[1].set_title('PPG by Opp Shot Volume and Defensive Resilience')
        axes[1].set_xlabel('Defensive Resilience (low GA/shot = good)')
        axes[1].set_ylabel('Opponent Shot Volume Faced')
    
    plt.tight_layout()
    filepath = f'{output_dir}/fig5_attack_defense_profiles.png'
    save_figure(fig, filepath)
    files.append(filepath)
    
    return files


def plot_ht_ft_transitions(htft_results: Dict, output_dir: str = '.') -> List[str]:
    """Create Figure 6: Half-time to Full-time transitions."""
    files = []
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 6a: Normalized transitions
    transitions = pd.DataFrame(htft_results['transitions_norm'])
    sns.heatmap(transitions, annot=True, fmt='.3f', cmap='Blues', ax=axes[0], 
                cbar_kws={'label': 'Probability'})
    axes[0].set_title('Half-Time to Full-Time Result Transitions')
    axes[0].set_xlabel('Full-Time Result')
    axes[0].set_ylabel('Half-Time Result')
    
    # 6b: Counts
    trans_counts = pd.DataFrame(htft_results['transitions_counts'])
    sns.heatmap(trans_counts, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                cbar_kws={'label': 'Count'})
    axes[1].set_title('Half-Time to Full-Time Result Transitions (Counts)')
    axes[1].set_xlabel('Full-Time Result')
    axes[1].set_ylabel('Half-Time Result')
    
    plt.tight_layout()
    filepath = f'{output_dir}/fig6_ht_ft_transitions.png'
    save_figure(fig, filepath)
    files.append(filepath)
    
    return files


def plot_half_goals_comebacks(htft_results: Dict, master: pd.DataFrame, output_dir: str = '.') -> List[str]:
    """Create Figure 7: Half goals and comebacks."""
    files = []
    
    # Add SH goals to master if not present
    master = master.copy()
    master['SH_Goals_Home'] = master['FTHG'] - master['HTHG']
    master['SH_Goals_Away'] = master['FTAG'] - master['HTAG']
    master['SH_Total_Goals'] = master['SH_Goals_Home'] + master['SH_Goals_Away']
    master['FH_Total_Goals'] = master['HTHG'] + master['HTAG']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 7a: Distribution
    ax = axes[0, 0]
    ax.hist([master['FH_Total_Goals'], master['SH_Total_Goals']], bins=range(0, 10), 
            label=['First Half', 'Second Half'], alpha=0.7, density=True)
    ax.set_title('Distribution of Goals: First Half vs Second Half')
    ax.set_xlabel('Total Goals in Half')
    ax.set_ylabel('Density')
    ax.legend()
    
    # 7b: SH goals by HT result
    ax = axes[0, 1]
    ht_result_goals = master.groupby('HTR')['SH_Total_Goals'].mean()
    ht_result_goals.plot(kind='bar', ax=ax, color=[COLORS['Home'], COLORS['Draw'], COLORS['Away']])
    ax.set_title('Average Second-Half Goals by Half-Time Result')
    ax.set_xlabel('Half-Time Result')
    ax.set_ylabel('Second-Half Goals')
    ax.tick_params(axis='x', rotation=0)
    
    # 7c: SH goals by league
    ax = axes[1, 0]
    league_sh = master.groupby('Div')['SH_Total_Goals'].mean().sort_values()
    league_sh.plot(kind='barh', ax=ax, color=COLORS['Home'])
    ax.set_title('Average Second-Half Goals by League')
    ax.set_xlabel('Second-Half Goals')
    
    # 7d: Comeback rates by league
    ax = axes[1, 1]
    comeback_df = htft_results['comeback_by_league'].sort_values('Trailing_Avoid_Loss_Pct')
    ax.barh(comeback_df['League'], comeback_df['Trailing_Avoid_Loss_Pct'], color=COLORS['Away'], alpha=0.7)
    ax.set_title('Comeback Rate: Trailing at HT, Avoiding Defeat at FT')
    ax.set_xlabel('% Avoiding Defeat')
    
    plt.tight_layout()
    filepath = f'{output_dir}/fig7_half_goals_comebacks.png'
    save_figure(fig, filepath)
    files.append(filepath)
    
    return files


def plot_team_profiles(team_season: pd.DataFrame, consistency: pd.DataFrame, output_dir: str = '.') -> List[str]:
    """Create Figure 8: Team profiles and consistency."""
    files = []
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 8a: Attack-Defense map colored by cluster
    ax = axes[0, 0]
    if 'Cluster' in team_season.columns:
        scatter = ax.scatter(team_season['GF_PG'], team_season['GA_PG'], 
                             c=team_season['Cluster'], cmap='tab10', alpha=0.6, s=30)
        ax.axvline(x=team_season['GF_PG'].median(), color='gray', linestyle=':', alpha=0.5)
        ax.axhline(y=team_season['GA_PG'].median(), color='gray', linestyle=':', alpha=0.5)
        ax.text(0.95, 0.05, 'Strong Attack\nStrong Defense', transform=ax.transAxes, ha='right', va='bottom', fontsize=9, alpha=0.7)
        ax.text(0.05, 0.05, 'Weak Attack\nStrong Defense', transform=ax.transAxes, ha='left', va='bottom', fontsize=9, alpha=0.7)
        ax.text(0.95, 0.95, 'Strong Attack\nWeak Defense', transform=ax.transAxes, ha='right', va='top', fontsize=9, alpha=0.7)
        ax.text(0.05, 0.95, 'Weak Attack\nWeak Defense', transform=ax.transAxes, ha='left', va='top', fontsize=9, alpha=0.7)
        plt.colorbar(scatter, ax=ax, label='Cluster')
    ax.set_xlabel('Goals For per Game')
    ax.set_ylabel('Goals Against per Game')
    ax.set_title('Team-Season Attack-Defense Profile (5 Clusters)')
    
    # 8b: Shots vs Finishing by cluster
    ax = axes[0, 1]
    if 'Cluster' in team_season.columns:
        for c in sorted(team_season['Cluster'].dropna().unique()):
            subset = team_season[team_season['Cluster'] == c]
            ax.scatter(subset['Shots_PG'], subset['Goals_Per_Shot'], alpha=0.5, s=20, label=f'Cluster {int(c)}')
    ax.set_xlabel('Shots per Game')
    ax.set_ylabel('Goals per Shot')
    ax.set_title('Shot Volume vs Finishing by Cluster')
    ax.legend()
    
    # 8c: Consistency vs Performance
    ax = axes[1, 0]
    ax.scatter(consistency['PPG'], consistency['Points_CV'], alpha=0.5, s=20)
    ax.set_xlabel('Points per Game')
    ax.set_ylabel('Points Coefficient of Variation')
    ax.set_title('Performance vs Consistency')
    high_perf_consistent = consistency[(consistency['PPG'] > 1.8) & (consistency['Points_CV'] < 0.6)]
    low_perf_inconsistent = consistency[(consistency['PPG'] < 1.0) & (consistency['Points_CV'] > 0.8)]
    ax.scatter(high_perf_consistent['PPG'], high_perf_consistent['Points_CV'], color='green', s=50, label='High Perf + Consistent', zorder=5)
    ax.scatter(low_perf_inconsistent['PPG'], low_perf_inconsistent['Points_CV'], color='red', s=50, label='Low Perf + Inconsistent', zorder=5)
    ax.legend()
    
    # 8d: Cluster PPG distribution
    ax = axes[1, 1]
    if 'Cluster' in team_season.columns:
        cluster_order = team_season.groupby('Cluster')['PPG'].mean().sort_values().index
        sns.boxplot(data=team_season, x='Cluster', y='PPG', order=cluster_order, ax=ax)
    ax.set_title('Points per Game by Cluster')
    ax.set_xlabel('Cluster (ordered by mean PPG)')
    ax.set_ylabel('Points per Game')
    
    plt.tight_layout()
    filepath = f'{output_dir}/fig8_team_profiles.png'
    save_figure(fig, filepath)
    files.append(filepath)
    
    return files


def plot_consistency_leaders(team_consistency: pd.DataFrame, output_dir: str = '.') -> List[str]:
    """Create Figure 9: Consistency leaders."""
    files = []
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 9a: Most consistent high-performers
    top_teams = team_consistency[team_consistency['Avg_PPG'] > 1.5].nsmallest(15, 'Avg_Points_CV')
    ax = axes[0]
    colors = ['green' if x < team_consistency['Avg_Points_CV'].median() else 'orange' for x in top_teams['Avg_Points_CV']]
    ax.barh(range(len(top_teams)), top_teams['Avg_Points_CV'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(top_teams)))
    ax.set_yticklabels([f"{row['Team']} ({row['Avg_PPG']:.2f} PPG)" for _, row in top_teams.iterrows()])
    ax.set_xlabel('Points Coefficient of Variation (lower = more consistent)')
    ax.set_title('Most Consistent High-Performing Teams (3+ seasons)')
    
    # 9b: Least consistent
    bottom_teams = team_consistency.nlargest(15, 'Avg_Points_CV')
    ax = axes[1]
    colors = ['red' if x > team_consistency['Avg_Points_CV'].median() else 'orange' for x in bottom_teams['Avg_Points_CV']]
    ax.barh(range(len(bottom_teams)), bottom_teams['Avg_Points_CV'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(bottom_teams)))
    ax.set_yticklabels([f"{row['Team']} ({row['Avg_PPG']:.2f} PPG)" for _, row in bottom_teams.iterrows()])
    ax.set_xlabel('Points Coefficient of Variation')
    ax.set_title('Least Consistent Teams (3+ seasons)')
    
    plt.tight_layout()
    filepath = f'{output_dir}/fig9_consistency_leaders.png'
    save_figure(fig, filepath)
    files.append(filepath)
    
    return files


def plot_discipline(disc_results: Dict, team_season_disc: pd.DataFrame, output_dir: str = '.') -> List[str]:
    """Create Figure 10: Disciplinary patterns."""
    files = []
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 10a: Yellow cards vs PPG
    ax = axes[0, 0]
    valid = team_season_disc['Yellow_PG'].notna() & team_season_disc['PPG'].notna()
    if valid.sum() > 10:
        ax.scatter(team_season_disc.loc[valid, 'Yellow_PG'], team_season_disc.loc[valid, 'PPG'], alpha=0.5, s=20)
        z = np.polyfit(team_season_disc.loc[valid, 'Yellow_PG'], team_season_disc.loc[valid, 'PPG'], 1)
        p = np.poly1d(z)
        x_range = np.linspace(team_season_disc.loc[valid, 'Yellow_PG'].min(), team_season_disc.loc[valid, 'Yellow_PG'].max(), 100)
        ax.plot(x_range, p(x_range), 'r--', alpha=0.8)
        r = disc_results['ppg_correlations']['Yellow_PG']['r']
        ax.set_title(f'Yellow Cards per Game vs PPG (r={r:.3f})')
    else:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Yellow Cards per Game vs PPG')
    ax.set_xlabel('Yellow Cards per Game')
    ax.set_ylabel('Points per Game')
    
    # 10b: Fouls vs PPG
    ax = axes[0, 1]
    valid = team_season_disc['Fouls_PG'].notna() & team_season_disc['PPG'].notna()
    if valid.sum() > 10:
        ax.scatter(team_season_disc.loc[valid, 'Fouls_PG'], team_season_disc.loc[valid, 'PPG'], alpha=0.5, s=20)
        z = np.polyfit(team_season_disc.loc[valid, 'Fouls_PG'], team_season_disc.loc[valid, 'PPG'], 1)
        p = np.poly1d(z)
        x_range = np.linspace(team_season_disc.loc[valid, 'Fouls_PG'].min(), team_season_disc.loc[valid, 'Fouls_PG'].max(), 100)
        ax.plot(x_range, p(x_range), 'r--', alpha=0.8)
        r = disc_results['ppg_correlations']['Fouls_PG']['r']
        ax.set_title(f'Fouls per Game vs PPG (r={r:.3f})')
    else:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Fouls per Game vs PPG')
    ax.set_xlabel('Fouls per Game')
    ax.set_ylabel('Points per Game')
    
    # 10c: Discipline by venue
    ax = axes[1, 0]
    venue_disc = pd.DataFrame(disc_results['by_venue'])
    x = np.arange(len(venue_disc))
    width = 0.25
    ax.bar(x - width, venue_disc['Yellow_PG'], width, label='Yellows', alpha=0.8)
    ax.bar(x, venue_disc['Red_PG'], width, label='Reds', alpha=0.8)
    ax.bar(x + width, venue_disc['Fouls_PG'], width, label='Fouls', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(venue_disc['Venue'])
    ax.set_ylabel('Per Game')
    ax.set_title('Discipline by Venue')
    ax.legend()
    
    # 10d: Red card impact
    ax = axes[1, 1]
    red_impact = pd.DataFrame({
        'Situation': ['Baseline', 'Home Red', 'Away Red'],
        'Home Win': [disc_results['red_card_impact']['baseline']['home_win'],
                     disc_results['red_card_impact']['home_red']['home_win'],
                     disc_results['red_card_impact']['away_red']['home_win']],
        'Draw': [disc_results['red_card_impact']['baseline']['draw'],
                 disc_results['red_card_impact']['home_red']['draw'],
                 disc_results['red_card_impact']['away_red']['draw']],
        'Away Win': [disc_results['red_card_impact']['baseline']['away_win'],
                      disc_results['red_card_impact']['home_red']['away_win'],
                      disc_results['red_card_impact']['away_red']['away_win']]
    })
    red_impact.set_index('Situation').plot(kind='bar', stacked=True, ax=ax, 
                                            color=[COLORS['Home'], COLORS['Draw'], COLORS['Away']])
    ax.set_title('Match Outcome Distribution: Red Card Impact')
    ax.set_ylabel('Proportion')
    ax.legend(title='FT Result')
    ax.tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    filepath = f'{output_dir}/fig10_discipline.png'
    save_figure(fig, filepath)
    files.append(filepath)
    
    return files


def plot_foul_differential(disc_results: Dict, team_season_disc2: pd.DataFrame, output_dir: str = '.') -> List[str]:
    """Create Figure 11: Foul differential vs PPG."""
    files = []
    
    fig, ax = plt.subplots(figsize=(10, 6))
    valid = team_season_disc2['Foul_Diff_PG'].notna() & team_season_disc2['PPG'].notna()
    if valid.sum() > 10:
        ax.scatter(team_season_disc2.loc[valid, 'Foul_Diff_PG'], team_season_disc2.loc[valid, 'PPG'], alpha=0.5, s=20)
        z = np.polyfit(team_season_disc2.loc[valid, 'Foul_Diff_PG'], team_season_disc2.loc[valid, 'PPG'], 1)
        p = np.poly1d(z)
        x_range = np.linspace(team_season_disc2.loc[valid, 'Foul_Diff_PG'].min(), team_season_disc2.loc[valid, 'Foul_Diff_PG'].max(), 100)
        ax.plot(x_range, p(x_range), 'r--', alpha=0.8)
        r = disc_results['foul_differential_correlation']['r']
        ax.set_title(f'Foul Differential (Committed \u2212 Drawn) per Game vs PPG (r={r:.3f})')
    else:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Foul Differential vs PPG')
    ax.set_xlabel('Foul Differential per Game (positive = commit more than draw)')
    ax.set_ylabel('Points per Game')
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5, label='Neutral (commit = draw)')
    ax.legend()
    
    plt.tight_layout()
    filepath = f'{output_dir}/fig11_foul_differential.png'
    save_figure(fig, filepath)
    files.append(filepath)
    
    return files


def create_all_figures(ha_results: Dict, team_season: pd.DataFrame, shot_results: Dict,
                       half_eff: pd.DataFrame, htft_results: Dict, master: pd.DataFrame,
                       consistency: pd.DataFrame, team_consistency: pd.DataFrame,
                       disc_results: Dict, team_season_disc: pd.DataFrame,
                       team_season_disc2: pd.DataFrame, output_dir: str = '.') -> List[str]:
    """Create all figures and return list of filepaths."""
    all_files = []
    
    all_files.extend(plot_home_advantage(ha_results, master, output_dir))
    all_files.extend(plot_efficiency_vs_points(team_season, output_dir))
    all_files.extend(plot_efficiency_persistence(half_eff, output_dir))
    all_files.extend(plot_shot_decomposition(team_season, output_dir))
    all_files.extend(plot_attack_defense_profiles(team_season, shot_results, output_dir))
    all_files.extend(plot_ht_ft_transitions(htft_results, output_dir))
    all_files.extend(plot_half_goals_comebacks(htft_results, master, output_dir))
    all_files.extend(plot_team_profiles(team_season, consistency, output_dir))
    all_files.extend(plot_consistency_leaders(team_consistency, output_dir))
    all_files.extend(plot_discipline(disc_results, team_season_disc, output_dir))
    all_files.extend(plot_foul_differential(disc_results, team_season_disc2, output_dir))
    
    return all_files