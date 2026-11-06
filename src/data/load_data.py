"""
Data loading and preprocessing module.
Loads football-data.co.uk CSVs from zip archives, handles encoding issues,
and builds the master match and team-perspective datasets.

Migrated to Polars: all ingestion, schema normalization and validation now
operate on Polars LazyFrames / DataFrames. Pandas is not used anywhere in
this module.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import polars as pl

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
    'HY', 'AY', 'HR', 'AR',
]

# Numeric core columns that must be present (as null when missing)
_NUMERIC_CORE_COLS = [
    'FTHG', 'FTAG', 'HTHG', 'HTAG',
    'HS', 'AS', 'HST', 'AST',
    'HF', 'AF', 'HC', 'AC',
    'HY', 'AY', 'HR', 'AR',
]

# Leagues with full data (including Referee)
FULL_DATA_LEAGUES = ['E0', 'E1', 'E2', 'E3', 'SC0', 'SC1', 'SC2', 'SC3']

# Leagues with partial data (no Referee)
PARTIAL_DATA_LEAGUES = ['B1', 'D1', 'D2', 'F1', 'F2', 'G1', 'I1', 'I2', 'N1', 'P1', 'SP1', 'SP2', 'T1']

# Leagues with minimal data (no shots/fouls/corners)
MINIMAL_DATA_LEAGUES = ['EC']


def _decode_csv_bytes(raw: bytes) -> bytes:
    """Decode raw CSV bytes, trying UTF-8 first and falling back to Latin-1.

    Returns UTF-8 encoded bytes suitable for Polars CSV parsing.
    """
    try:
        raw.decode('utf-8')
        return raw
    except UnicodeDecodeError:
        return raw.decode('latin-1').encode('utf-8')


def read_csv_from_zip(zip_path: Path, csv_name: str,
                      usecols: Optional[List[str]] = None,
                      zf: Optional[zipfile.ZipFile] = None) -> pl.DataFrame:
    """Read a CSV from a zip archive, handling UTF-8 and Latin-1 encodings.

    The ZIP member is read into memory once and decoded before being handed to
    Polars. Polars cannot transparently stream a single member of a ZIP
    archive, so we materialize the member bytes (small CSV files) and parse via
    ``pl.read_csv`` on a byte stream. An already-open archive handle can be
    passed in ``zf`` so the archive is not re-opened for every member.

    Only the columns requested in ``usecols`` that are present in the file are
    parsed: the header is inspected first and the rest (e.g. betting columns)
    is never materialized by the parser. Missing requested columns are added as
    nulls rather than raising, so concatenation across leagues keeps a stable
    schema.
    """
    own_handle = zf is None
    if own_handle:
        zf = zipfile.ZipFile(zip_path, 'r')
    try:
        if csv_name not in zf.namelist():
            return pl.DataFrame()
        raw = zf.read(csv_name)
    finally:
        if own_handle:
            zf.close()

    data = _decode_csv_bytes(raw)
    # football-data.co.uk files may start with a UTF-8 BOM; strip it so header
    # matching is reliable.
    if data.startswith(b'\xef\xbb\xbf'):
        data = data[3:]

    if usecols is None:
        return pl.read_csv(io.BytesIO(data), try_parse_dates=False, infer_schema_length=1000)

    # Inspect the header only and ask Polars to parse just the requested
    # columns that exist in the file; betting columns are never parsed.
    header_line = data.split(b'\n', 1)[0].decode('utf-8', errors='replace')
    present = [name.strip() for name in header_line.split(',')]
    select_cols = [col for col in usecols if col in present]

    if not select_cols:
        # Nothing to parse: return every requested column as nulls.
        return pl.DataFrame({
            col: pl.Series([], dtype=pl.Int64 if col in _NUMERIC_CORE_COLS else pl.Utf8)
            for col in usecols
        })

    df = pl.read_csv(
        io.BytesIO(data),
        try_parse_dates=False,
        infer_schema_length=1000,
        columns=select_cols,
    )

    select_exprs = [pl.col(col) for col in usecols if col in df.columns]
    for col in usecols:
        if col not in df.columns:
            # Column not available for this league/competition: represent as
            # null. Numeric core columns get Int64, everything else Utf8, so
            # that concatenation across leagues keeps a stable schema.
            if col in _NUMERIC_CORE_COLS:
                select_exprs.append(pl.lit(None).cast(pl.Int64).alias(col))
            else:
                select_exprs.append(pl.lit(None).cast(pl.Utf8).alias(col))
    return df.select(select_exprs)


def get_available_leagues(zip_path: Path) -> List[str]:
    """Get list of league codes available in a season zip file."""
    with zipfile.ZipFile(zip_path, 'r') as z:
        leagues = [f.replace('.csv', '') for f in z.namelist() if f.endswith('.csv')]
    return sorted(leagues)


def _usecols_for_league(league: str, usecols: Optional[List[str]]) -> Optional[List[str]]:
    """Return the column subset appropriate for a league's data availability."""
    if usecols is None:
        usecols = list(CORE_COLS)
    if league in FULL_DATA_LEAGUES:
        return list(usecols)
    if league in PARTIAL_DATA_LEAGUES:
        return [c for c in usecols if c != 'Referee']
    # Minimal-data leagues: drop Referee and all match statistics.
    return [c for c in usecols if c not in [
        'Referee', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC',
        'HY', 'AY', 'HR', 'AR',
    ]]


def load_season(zip_path: Path,
                leagues: Optional[List[str]] = None,
                usecols: Optional[List[str]] = None) -> pl.DataFrame:
    """Load all specified leagues from a season zip file.

    The ZIP archive is opened once and each league member is read from the
    already-opened handle (the previous implementation re-opened the archive
    per league).
    """
    if leagues is None:
        leagues = get_available_leagues(zip_path)

    frames: List[pl.DataFrame] = []
    with zipfile.ZipFile(zip_path, 'r') as z:
        namelist = set(z.namelist())
        for league in leagues:
            csv_name = f'{league}.csv'
            if csv_name not in namelist:
                continue
            league_usecols = _usecols_for_league(league, usecols)
            # Read through the already-open archive instead of re-opening it
            # for every league member.
            df = read_csv_from_zip(zip_path, csv_name, usecols=league_usecols, zf=z)
            if df.is_empty():
                continue
            df = df.with_columns(pl.lit(zip_path.parent.name).alias('Season'))
            frames.append(df)

    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how='diagonal')


def build_master_dataset(data_dir: Path,
                         target_leagues: List[str],
                         seasons: List[str],
                         usecols: Optional[List[str]] = None) -> pl.DataFrame:
    """Build master match dataset across multiple seasons and leagues.

    Returns an eager Polars DataFrame sorted by ``['Div', 'Date']`` with
    ``Date`` parsed to ``pl.Date``. Missing columns across competitions are
    represented as nulls (never silently zero-filled).
    """
    if usecols is None:
        usecols = list(CORE_COLS)

    frames: List[pl.DataFrame] = []
    for season in seasons:
        zip_path = Path(data_dir) / season / 'data.zip'
        if zip_path.exists():
            df = load_season(zip_path, leagues=target_leagues, usecols=usecols)
            if not df.is_empty():
                frames.append(df)

    if not frames:
        raise ValueError("No data loaded")

    master = pl.concat(frames, how='diagonal')

    # Parse DD/MM/YYYY dates deterministically; unparseable values -> null.
    master = master.with_columns(
        pl.col('Date').str.to_date('%d/%m/%Y', strict=False)
    )

    master = master.sort(['Div', 'Date'])
    return master


def build_team_match_dataset(master: pl.DataFrame) -> pl.DataFrame:
    """Convert match-level data to team-perspective (long) format.

    Each match produces two rows: one from the home team's perspective and
    one from the away team's perspective. Columns missing from the master
    (e.g. statistics for minimal-data leagues) are carried as nulls.
    """
    home = master.with_columns([
        pl.col('HomeTeam').alias('Team'),
        pl.col('AwayTeam').alias('Opponent'),
        pl.lit('Home').alias('Venue'),
        pl.col('FTHG').alias('GF'),
        pl.col('FTAG').alias('GA'),
        (pl.col('FTHG') - pl.col('FTAG')).alias('GD'),
        pl.col('HTHG').alias('HT_GF'),
        pl.col('HTAG').alias('HT_GA'),
        (pl.col('HTHG') - pl.col('HTAG')).alias('HT_GD'),
        pl.col('HTR').replace_strict({'H': 'W', 'D': 'D', 'A': 'L'}).alias('HT_Result'),
        pl.col('FTR').replace_strict({'H': 'W', 'D': 'D', 'A': 'L'}).alias('FT_Result'),
        pl.col('HS').alias('Shots'),
        pl.col('HST').alias('Shots_OT'),
        pl.col('AS').alias('Shots_Against'),
        pl.col('AST').alias('Shots_OT_Against'),
        pl.col('HF').alias('Fouls'),
        pl.col('AF').alias('Fouls_Against'),
        pl.col('HC').alias('Corners'),
        pl.col('AC').alias('Corners_Against'),
        pl.col('HY').alias('Yellow'),
        pl.col('AY').alias('Yellow_Against'),
        pl.col('HR').alias('Red'),
        pl.col('AR').alias('Red_Against'),
    ])

    away = master.with_columns([
        pl.col('AwayTeam').alias('Team'),
        pl.col('HomeTeam').alias('Opponent'),
        pl.lit('Away').alias('Venue'),
        pl.col('FTAG').alias('GF'),
        pl.col('FTHG').alias('GA'),
        (pl.col('FTAG') - pl.col('FTHG')).alias('GD'),
        pl.col('HTAG').alias('HT_GF'),
        pl.col('HTHG').alias('HT_GA'),
        (pl.col('HTAG') - pl.col('HTHG')).alias('HT_GD'),
        pl.col('HTR').replace_strict({'A': 'W', 'D': 'D', 'H': 'L'}).alias('HT_Result'),
        pl.col('FTR').replace_strict({'A': 'W', 'D': 'D', 'H': 'L'}).alias('FT_Result'),
        pl.col('AS').alias('Shots'),
        pl.col('AST').alias('Shots_OT'),
        pl.col('HS').alias('Shots_Against'),
        pl.col('HST').alias('Shots_OT_Against'),
        pl.col('AF').alias('Fouls'),
        pl.col('HF').alias('Fouls_Against'),
        pl.col('AC').alias('Corners'),
        pl.col('HC').alias('Corners_Against'),
        pl.col('AY').alias('Yellow'),
        pl.col('HY').alias('Yellow_Against'),
        pl.col('AR').alias('Red'),
        pl.col('HR').alias('Red_Against'),
    ])

    cols = ['Div', 'Date', 'Season', 'Team', 'Opponent', 'Venue',
            'GF', 'GA', 'GD', 'FT_Result',
            'HT_GF', 'HT_GA', 'HT_GD', 'HT_Result',
            'Shots', 'Shots_OT', 'Shots_Against', 'Shots_OT_Against',
            'Fouls', 'Fouls_Against', 'Corners', 'Corners_Against',
            'Yellow', 'Yellow_Against', 'Red', 'Red_Against']

    team_matches = pl.concat([home.select(cols), away.select(cols)], how='diagonal')
    team_matches = team_matches.sort(['Div', 'Team', 'Date'])
    return team_matches


def audit_data_quality(master: pl.DataFrame, team_matches: pl.DataFrame) -> Dict:
    """Run data quality checks and return summary (all values JSON-serializable)."""
    issues: Dict = {}

    # Missing values
    issues['missing_core'] = {
        col: int(master.select(pl.col(col).null_count()).to_series()[0])
        for col in CORE_COLS if col in master.columns
    }
    issues['missing_team'] = {
        col: int(team_matches.select(pl.col(col).null_count()).to_series()[0])
        for col in team_matches.columns
    }

    # Duplicates: count rows involved in duplicate (Date, HomeTeam, AwayTeam) keys
    issues['match_duplicates'] = int(
        master.group_by(['Date', 'HomeTeam', 'AwayTeam']).len()
              .filter(pl.col('len') > 1).select(pl.col('len').sum() - pl.col('len').count())
              .to_series()[0]
    ) if master.height > 0 else 0

    issues['team_match_duplicates'] = int(
        team_matches.group_by(['Date', 'Team', 'Opponent', 'Venue']).len()
              .filter(pl.col('len') > 1).select(pl.col('len').sum() - pl.col('len').count())
              .to_series()[0]
    ) if team_matches.height > 0 else 0

    # Impossible values
    issues['negative_goals'] = {
        'FTHG': int(master.filter(pl.col('FTHG') < 0).height) if 'FTHG' in master.columns else 0,
        'FTAG': int(master.filter(pl.col('FTAG') < 0).height) if 'FTAG' in master.columns else 0,
    }

    def _shot_anomaly_fixtures(cond_expr: pl.Expr) -> List[Dict]:
        """Return fixture identifiers for shots-below-shots-on-target rows.

        Shots on target cannot exceed total shots under the standard
        definitions; affected fixtures are recorded (not silently modified) so
        downstream analyses can apply a documented exclusion policy.
        """
        key_cols = ['Div', 'Season', 'Date', 'HomeTeam', 'AwayTeam']
        if not set(key_cols) <= set(master.columns) or not {'HS', 'HST', 'AS', 'AST'} <= set(master.columns):
            return []
        return (
            master.filter(cond_expr)
            .select([
                pl.col('Div'),
                pl.col('Season'),
                pl.col('Date').cast(pl.Utf8).alias('Date'),
                pl.col('HomeTeam'),
                pl.col('AwayTeam'),
                pl.col('HS'), pl.col('HST'), pl.col('AS'), pl.col('AST'),
            ])
            .to_dicts()
        )

    issues['shots_lt_sot'] = {
        'home': int(master.filter(pl.col('HS') < pl.col('HST')).height)
                if {'HS', 'HST'} <= set(master.columns) else 0,
        'away': int(master.filter(pl.col('AS') < pl.col('AST')).height)
                if {'AS', 'AST'} <= set(master.columns) else 0,
        'home_fixtures': _shot_anomaly_fixtures(pl.col('HS') < pl.col('HST')),
        'away_fixtures': _shot_anomaly_fixtures(pl.col('AS') < pl.col('AST')),
    }

    # Result consistency: FTR must match goal comparison
    if {'FTHG', 'FTAG', 'FTR'} <= set(master.columns):
        ftr_inconsistent = (
            ((pl.col('FTHG') > pl.col('FTAG')) & (pl.col('FTR') != 'H')) |
            ((pl.col('FTHG') < pl.col('FTAG')) & (pl.col('FTR') != 'A')) |
            ((pl.col('FTHG') == pl.col('FTAG')) & (pl.col('FTR') != 'D'))
        )
        issues['ftr_inconsistent'] = int(master.filter(ftr_inconsistent).height)
    else:
        issues['ftr_inconsistent'] = 0

    if {'HTHG', 'HTAG', 'HTR'} <= set(master.columns):
        htr_inconsistent = (
            ((pl.col('HTHG') > pl.col('HTAG')) & (pl.col('HTR') != 'H')) |
            ((pl.col('HTHG') < pl.col('HTAG')) & (pl.col('HTR') != 'A')) |
            ((pl.col('HTHG') == pl.col('HTAG')) & (pl.col('HTR') != 'D'))
        )
        issues['htr_inconsistent'] = int(master.filter(htr_inconsistent).height)
    else:
        issues['htr_inconsistent'] = 0

    return issues
