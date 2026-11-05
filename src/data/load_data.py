"""
Data loading and preprocessing module.
Loads football-data.co.uk CSVs from zip archives, handles encoding issues,
and builds the master match and team-perspective datasets.
"""

import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional


# League code to name mapping
LEAGUE_NAMES = {
    'E0': 'England Premier League',
    'E1': 'England Championship',
    'E2': 'England League One',
    'E3': 'England League Two',
    'EC': 'England Conference',
    'SP1': 'Spain La Liga',
    'SP2': 'Spain Segunda Division',
    'I1': 'Italy Serie A',
    'I2': 'Italy Serie B',
    'D1': 'Germany Bundesliga',
    'D2': 'Germany Bundesliga 2',
    'F1': 'France Ligue 1',
    'F2': 'France Ligue 2',
    'N1': 'Netherlands Eredivisie',
    'B1': 'Belgium Pro League',
    'P1': 'Portugal Primeira Liga',
    'T1': 'Turkey Super Lig',
    'G1': 'Greece Super League',
    'SC0': 'Scotland Premiership',
    'SC1': 'Scotland Championship',
    'SC2': 'Scotland League One',
    'SC3': 'Scotland League Two',
}

# Core non-betting columns available across all leagues
CORE_COLS = [
    'Div', 'Date', 'Time', 'HomeTeam', 'AwayTeam',
    'FTHG', 'FTAG', 'FTR',
    'HTHG', 'HTAG', 'HTR',
    'Referee',
    'HS', 'AS', 'HST', 'AST',
    'HF', 'AF', 'HC', 'AC',
    'HY', 'AY', 'HR', 'AR'
]

# Leagues with full data (including Referee)
FULL_DATA_LEAGUES = ['E0', 'E1', 'E2', 'E3', 'SC0', 'SC1', 'SC2', 'SC3']

# Leagues with partial data (no Referee)
PARTIAL_DATA_LEAGUES = ['B1', 'D1', 'D2', 'F1', 'F2', 'G1', 'I1', 'I2', 'N1', 'P1', 'SP1', 'SP2', 'T1']

# Leagues with minimal data (no shots/fouls/corners)
MINIMAL_DATA_LEAGUES = ['EC']


def read_csv_from_zip(zip_path: Path, csv_name: str, usecols: Optional[List[str]] = None) -> pd.DataFrame:
    """Read a CSV from a zip archive, handling UTF-8 and Latin-1 encodings."""
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open(csv_name) as f:
            try:
                df = pd.read_csv(f, encoding='utf-8', usecols=usecols)
            except UnicodeDecodeError:
                f.seek(0)
                df = pd.read_csv(f, encoding='latin-1', usecols=usecols)
    return df


def get_available_leagues(zip_path: Path) -> List[str]:
    """Get list of league codes available in a season zip file."""
    with zipfile.ZipFile(zip_path, 'r') as z:
        leagues = [f.replace('.csv', '') for f in z.namelist() if f.endswith('.csv')]
    return sorted(leagues)


def load_season(zip_path: Path, leagues: Optional[List[str]] = None, 
                usecols: Optional[List[str]] = None) -> pd.DataFrame:
    """Load all specified leagues from a season zip file."""
    if leagues is None:
        leagues = get_available_leagues(zip_path)
    
    dfs = []
    for league in leagues:
        csv_name = f'{league}.csv'
        if csv_name in zipfile.ZipFile(zip_path, 'r').namelist():
            # Use appropriate columns based on league data availability
            if league in FULL_DATA_LEAGUES:
                league_usecols = usecols
            elif league in PARTIAL_DATA_LEAGUES:
                league_usecols = [c for c in usecols if c != 'Referee']
            else:
                league_usecols = [c for c in usecols if c not in ['Referee', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']]
            df = read_csv_from_zip(zip_path, csv_name, usecols=league_usecols)
            df['Season'] = zip_path.parent.name
            dfs.append(df)
    
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def build_master_dataset(data_dir: Path, 
                         target_leagues: List[str],
                         seasons: List[str],
                         usecols: Optional[List[str]] = None) -> pd.DataFrame:
    """Build master match dataset across multiple seasons and leagues."""
    if usecols is None:
        usecols = CORE_COLS
    
    all_dfs = []
    for season in seasons:
        zip_path = data_dir / season / 'data.zip'
        if zip_path.exists():
            df = load_season(zip_path, leagues=target_leagues, usecols=usecols)
            if not df.empty:
                all_dfs.append(df)
    
    if not all_dfs:
        raise ValueError("No data loaded")
    
    master = pd.concat(all_dfs, ignore_index=True)
    master['Date'] = pd.to_datetime(master['Date'], format='%d/%m/%Y', dayfirst=True)
    master = master.sort_values(['Div', 'Date']).reset_index(drop=True)
    return master


def build_team_match_dataset(master: pd.DataFrame) -> pd.DataFrame:
    """Convert match-level data to team-perspective (long) format."""
    # Home team perspective
    home = master.copy()
    home['Team'] = home['HomeTeam']
    home['Opponent'] = home['AwayTeam']
    home['Venue'] = 'Home'
    home['GF'] = home['FTHG']
    home['GA'] = home['FTAG']
    home['GD'] = home['FTHG'] - home['FTAG']
    home['HT_GF'] = home['HTHG']
    home['HT_GA'] = home['HTAG']
    home['HT_GD'] = home['HTHG'] - home['HTAG']
    home['HT_Result'] = home['HTR'].map({'H': 'W', 'D': 'D', 'A': 'L'})
    home['FT_Result'] = home['FTR'].map({'H': 'W', 'D': 'D', 'A': 'L'})
    home['Shots'] = home['HS']
    home['Shots_OT'] = home['HST']
    home['Shots_Against'] = home['AS']
    home['Shots_OT_Against'] = home['AST']
    home['Fouls'] = home['HF']
    home['Fouls_Against'] = home['AF']
    home['Corners'] = home['HC']
    home['Corners_Against'] = home['AC']
    home['Yellow'] = home['HY']
    home['Yellow_Against'] = home['AY']
    home['Red'] = home['HR']
    home['Red_Against'] = home['AR']

    # Away team perspective
    away = master.copy()
    away['Team'] = away['AwayTeam']
    away['Opponent'] = away['HomeTeam']
    away['Venue'] = 'Away'
    away['GF'] = away['FTAG']
    away['GA'] = away['FTHG']
    away['GD'] = away['FTAG'] - away['FTHG']
    away['HT_GF'] = away['HTAG']
    away['HT_GA'] = away['HTHG']
    away['HT_GD'] = away['HTAG'] - away['HTHG']
    away['HT_Result'] = away['HTR'].map({'A': 'W', 'D': 'D', 'H': 'L'})
    away['FT_Result'] = away['FTR'].map({'A': 'W', 'D': 'D', 'H': 'L'})
    away['Shots'] = away['AS']
    away['Shots_OT'] = away['AST']
    away['Shots_Against'] = away['HS']
    away['Shots_OT_Against'] = away['HST']
    away['Fouls'] = away['AF']
    away['Fouls_Against'] = away['HF']
    away['Corners'] = away['AC']
    away['Corners_Against'] = away['HC']
    away['Yellow'] = away['AY']
    away['Yellow_Against'] = away['HY']
    away['Red'] = away['AR']
    away['Red_Against'] = away['HR']

    cols = ['Div', 'Date', 'Season', 'Team', 'Opponent', 'Venue',
            'GF', 'GA', 'GD', 'FT_Result',
            'HT_GF', 'HT_GA', 'HT_GD', 'HT_Result',
            'Shots', 'Shots_OT', 'Shots_Against', 'Shots_OT_Against',
            'Fouls', 'Fouls_Against', 'Corners', 'Corners_Against',
            'Yellow', 'Yellow_Against', 'Red', 'Red_Against']

    team_matches = pd.concat([home[cols], away[cols]], ignore_index=True)
    team_matches = team_matches.sort_values(['Div', 'Team', 'Date']).reset_index(drop=True)
    return team_matches


def audit_data_quality(master: pd.DataFrame, team_matches: pd.DataFrame) -> Dict:
    """Run data quality checks and return summary."""
    issues = {}
    
    # Missing values
    issues['missing_core'] = master[CORE_COLS].isnull().sum().to_dict()
    issues['missing_team'] = team_matches.isnull().sum().to_dict()
    
    # Duplicates
    issues['match_duplicates'] = master.duplicated(subset=['Date', 'HomeTeam', 'AwayTeam']).sum()
    issues['team_match_duplicates'] = team_matches.duplicated(subset=['Date', 'Team', 'Opponent', 'Venue']).sum()
    
    # Impossible values
    issues['negative_goals'] = {
        'FTHG': int((master['FTHG'] < 0).sum()),
        'FTAG': int((master['FTAG'] < 0).sum())
    }
    issues['shots_lt_sot'] = {
        'home': int((master['HS'] < master['HST']).sum()),
        'away': int((master['AS'] < master['AST']).sum())
    }
    
    # Result consistency
    ftr_inconsistent = (
        ((master['FTHG'] > master['FTAG']) & (master['FTR'] != 'H')).sum() +
        ((master['FTHG'] < master['FTAG']) & (master['FTR'] != 'A')).sum() +
        ((master['FTHG'] == master['FTAG']) & (master['FTR'] != 'D')).sum()
    )
    htr_inconsistent = (
        ((master['HTHG'] > master['HTAG']) & (master['HTR'] != 'H')).sum() +
        ((master['HTHG'] < master['HTAG']) & (master['HTR'] != 'A')).sum() +
        ((master['HTHG'] == master['HTAG']) & (master['HTR'] != 'D')).sum()
    )
    issues['ftr_inconsistent'] = int(ftr_inconsistent)
    issues['htr_inconsistent'] = int(htr_inconsistent)
    
    return issues