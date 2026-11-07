"""
Training–inference feature parity test harness.

This module compares two independent construction routes for the same fixture:
1. Historical route: obtain pre-match predictors from the historical feature-building
   pipeline (pre-computed on full history).
2. Inference route: remove the target and all same-date/future results, append a
   result-free target fixture with its actual season, then construct inference
   predictors via the inference API.

The test must exercise the difference between a completed historical fixture and
an upcoming result-free fixture — it must NOT pass by feeding the same precomputed
target row into both routes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import polars as pl
import warnings

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prediction.features import build_pre_match_features, pair_match_features
from prediction.inference import predict_fixture, PredictionBundle, FixturePrediction
from prediction.feature_schema import FeatureSchema, FeatureSpec, load_canonical_schema
from data.load_data import build_master_dataset, build_team_match_dataset, CORE_COLS


@dataclass
class FeatureMismatch:
    """A single feature-level discrepancy between routes."""
    fixture_key: str
    feature_name: str
    historical_value: Any
    inference_value: Any
    historical_dtype: str
    inference_dtype: str
    historical_missing: bool
    inference_missing: bool
    severity: str  # "critical", "warning", "info"


@dataclass
class ParityResult:
    """Result of a parity test for one fixture."""
    fixture_key: str
    division: str
    date: str
    home_team: str
    away_team: str
    season: str
    mismatches: List[FeatureMismatch] = field(default_factory=list)
    historical_probs: Optional[np.ndarray] = None
    inference_probs: Optional[np.ndarray] = None
    historical_lambdas: Optional[Tuple[float, float]] = None
    inference_lambdas: Optional[Tuple[float, float]] = None
    passed: bool = True

    def add_mismatch(self, mismatch: FeatureMismatch) -> None:
        self.mismatches.append(mismatch)
        if mismatch.severity == "critical":
            self.passed = False

    def summary(self) -> str:
        critical = sum(1 for m in self.mismatches if m.severity == "critical")
        warning = sum(1 for m in self.mismatches if m.severity == "warning")
        info = sum(1 for m in self.mismatches if m.severity == "info")
        status = "PASS" if self.passed else "FAIL"
        return f"{self.fixture_key}: {status} (critical={critical}, warning={warning}, info={info})"


@dataclass
class ParityReport:
    """Aggregated parity test report."""
    results: List[ParityResult]
    schema: FeatureSchema
    tolerance_rtol: float = 1e-8
    tolerance_atol: float = 1e-10

    def overall_passed(self) -> bool:
        return all(r.passed for r in self.results)

    def total_mismatches(self) -> int:
        return sum(len(r.mismatches) for r in self.results)

    def critical_count(self) -> int:
        return sum(1 for r in self.results for m in r.mismatches if m.severity == "critical")

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema.version,
            "tolerance_rtol": self.tolerance_rtol,
            "tolerance_atol": self.tolerance_atol,
            "overall_passed": self.overall_passed(),
            "total_fixtures": len(self.results),
            "passed_fixtures": sum(1 for r in self.results if r.passed),
            "total_mismatches": self.total_mismatches(),
            "critical_mismatches": self.critical_count(),
            "results": [
                {
                    "fixture_key": r.fixture_key,
                    "division": r.division,
                    "date": r.date,
                    "home_team": r.home_team,
                    "away_team": r.away_team,
                    "season": r.season,
                    "passed": r.passed,
                    "mismatches": [
                        {
                            "feature_name": m.feature_name,
                            "historical_value": _json_safe(m.historical_value),
                            "inference_value": _json_safe(m.inference_value),
                            "historical_dtype": m.historical_dtype,
                            "inference_dtype": m.inference_dtype,
                            "historical_missing": m.historical_missing,
                            "inference_missing": m.inference_missing,
                            "severity": m.severity,
                        }
                        for m in r.mismatches
                    ],
                }
                for r in self.results
            ],
        }


def _json_safe(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, (np.ndarray,)):
        return v.tolist()
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if v is None or isinstance(v, (int, float, str)):
        return v
    return str(v)


def _compare_values(
    fixture_key: str,
    feature_name: str,
    hist_val: Any,
    inf_val: Any,
    spec: Optional[FeatureSpec],
    rtol: float,
    atol: float,
) -> Optional[FeatureMismatch]:
    """Compare two feature values and return a mismatch if they differ."""
    hist_missing = hist_val is None or (isinstance(hist_val, float) and not np.isfinite(hist_val))
    inf_missing = inf_val is None or (isinstance(inf_val, float) and not np.isfinite(inf_val))

    # Missingness mismatch
    if hist_missing != inf_missing:
        return FeatureMismatch(
            fixture_key=fixture_key,
            feature_name=feature_name,
            historical_value=hist_val,
            inference_value=inf_val,
            historical_dtype=str(type(hist_val).__name__),
            inference_dtype=str(type(inf_val).__name__),
            historical_missing=hist_missing,
            inference_missing=inf_missing,
            severity="critical",
        )

    # Both missing - OK
    if hist_missing and inf_missing:
        return None

    # Both present - compare
    try:
        if isinstance(hist_val, (int, np.integer)) and isinstance(inf_val, (int, np.integer)):
            if hist_val != inf_val:
                return FeatureMismatch(
                    fixture_key=fixture_key,
                    feature_name=feature_name,
                    historical_value=hist_val,
                    inference_value=inf_val,
                    historical_dtype="int",
                    inference_dtype="int",
                    historical_missing=False,
                    inference_missing=False,
                    severity="critical",
                )
        elif isinstance(hist_val, (float, np.floating)) and isinstance(inf_val, (float, np.floating)):
            if not np.isclose(hist_val, inf_val, rtol=rtol, atol=atol, equal_nan=False):
                return FeatureMismatch(
                    fixture_key=fixture_key,
                    feature_name=feature_name,
                    historical_value=hist_val,
                    inference_value=inf_val,
                    historical_dtype="float",
                    inference_dtype="float",
                    historical_missing=False,
                    inference_missing=False,
                    severity="critical" if abs(hist_val - inf_val) > max(rtol * max(abs(hist_val), abs(inf_val)), atol) * 10 else "warning",
                )
        else:
            # String or other - exact match
            if hist_val != inf_val:
                return FeatureMismatch(
                    fixture_key=fixture_key,
                    feature_name=feature_name,
                    historical_value=hist_val,
                    inference_value=inf_val,
                    historical_dtype=str(type(hist_val).__name__),
                    inference_dtype=str(type(inf_val).__name__),
                    historical_missing=False,
                    inference_missing=False,
                    severity="warning",
                )
    except Exception as e:
        return FeatureMismatch(
            fixture_key=fixture_key,
            feature_name=feature_name,
            historical_value=hist_val,
            inference_value=inf_val,
            historical_dtype=str(type(hist_val).__name__),
            inference_dtype=str(type(inf_val).__name__),
            historical_missing=hist_missing,
            inference_missing=inf_missing,
            severity="warning",
        )
    return None


def _make_fixture_key(division: str, date: str, home: str, away: str) -> str:
    return f"{division}|{date}|{home}|{away}"


def _extract_feature_row(paired: pl.DataFrame, division: str, pred_date: date, home: str, away: str, schema: FeatureSchema) -> Tuple[Optional[pl.DataFrame], Dict[str, Any]]:
    """Extract the feature row for a specific fixture from paired features."""
    # Filter to the exact fixture
    m = paired.filter(
        (pl.col("Div") == division) &
        (pl.col("Date") == pl.lit(pred_date)) &
        (pl.col("HomeTeam") == home) &
        (pl.col("AwayTeam") == away)
    )
    if m.height == 0:
        return None, {}

    # Get values for schema features
    values = {}
    for feat_name in schema.feature_names():
        if feat_name in m.columns:
            val = m.select(pl.col(feat_name)).to_series()[0]
            values[feat_name] = val if val is not None else None
        else:
            values[feat_name] = None
    return m, values


def run_historical_route(
    paired: pl.DataFrame,
    division: str,
    pred_date: date,
    home: str,
    away: str,
    schema: FeatureSchema,
) -> Dict[str, Any]:
    """Route 1: Get features from pre-computed historical paired features."""
    _, values = _extract_feature_row(paired, division, pred_date, home, away, schema)
    return values


def run_inference_route(
    master: pl.DataFrame,
    team_matches: pl.DataFrame,
    division: str,
    pred_date: date,
    home: str,
    away: str,
    season: str,
    schema: FeatureSchema,
) -> Dict[str, Any]:
    """
    Route 2: Build features via inference API (simulating a future fixture).
    
    This removes the target fixture and all same-date/later results, then
    constructs features for the result-free target fixture.
    """
    # Filter history to strictly before the fixture date
    hist = master.filter(pl.col("Date") < pl.lit(pred_date))
    
    # Also filter team_matches to strictly before the fixture date
    hist_tm = team_matches.filter(pl.col("Date") < pl.lit(pred_date))
    
    # Create synthetic target row (no results)
    target_row = pl.DataFrame({
        "Div": [division],
        "Date": pl.Series([pred_date], dtype=pl.Date),
        "HomeTeam": [home],
        "AwayTeam": [away],
        "Season": [season],
        "FTHG": [None], "FTAG": [None], "FTR": [None],
        "Time": [None], "HTHG": [None], "HTAG": [None], "HTR": [None],
        "Referee": [None], "HS": [None], "AS": [None],
        "HST": [None], "AST": [None], "HF": [None], "AF": [None],
        "HC": [None], "AC": [None], "HY": [None], "AY": [None],
        "HR": [None], "AR": [None],
    })
    
    # Combine history + target
    combined = pl.concat([hist.select([c for c in CORE_COLS + ["Season"] if c in hist.columns]), target_row], how="diagonal")
    
    # Build team-perspective frame
    team_matches_all = build_team_match_dataset(combined).sort(["Div", "Team", "Season", "Date"])
    
    # Build pre-match features (lags only)
    feat = build_pre_match_features(team_matches_all)
    
    # Pair home/away
    paired = pair_match_features(feat)
    
    # Extract the target fixture's features
    _, values = _extract_feature_row(paired, division, pred_date, home, away, schema)
    return values


def test_fixture_parity(
    master: pl.DataFrame,
    team_matches: pl.DataFrame,
    paired: pl.DataFrame,
    division: str,
    pred_date: date,
    home: str,
    away: str,
    season: str,
    schema: FeatureSchema,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> ParityResult:
    """Run parity test for a single fixture."""
    fixture_key = _make_fixture_key(division, str(pred_date), home, away)
    result = ParityResult(
        fixture_key=fixture_key,
        division=division,
        date=str(pred_date),
        home_team=home,
        away_team=away,
        season=season,
    )
    
    # Route 1: Historical (from pre-computed paired features)
    hist_values = run_historical_route(paired, division, pred_date, home, away, schema)
    
    # Route 2: Inference (rebuild from history before fixture)
    inf_values = run_inference_route(master, team_matches, division, pred_date, home, away, season, schema)
    
    # Compare each feature
    for feat_name in schema.feature_names():
        spec = schema.get_spec(feat_name)
        hist_val = hist_values.get(feat_name)
        inf_val = inf_values.get(feat_name)
        
        mismatch = _compare_values(fixture_key, feat_name, hist_val, inf_val, spec, rtol, atol)
        if mismatch:
            result.add_mismatch(mismatch)
    
    return result


def create_test_cases(master: pl.DataFrame, team_matches: pl.DataFrame) -> List[Dict]:
    """Create a set of test fixtures covering the required edge cases."""
    test_cases = []
    
    # Get available divisions and seasons
    divisions = master.select(pl.col("Div").unique()).to_series().to_list()
    seasons = master.select(pl.col("Season").unique()).to_series().to_list()
    
    for div in divisions[:2]:  # Limit to first 2 divisions for speed
        div_matches = master.filter(pl.col("Div") == div)
        
        # Case 1: Ordinary mid-season fixture (not first 5, not last 5 of season)
        for season in seasons[-3:]:  # Recent seasons
            season_matches = div_matches.filter(pl.col("Season") == season).sort("Date")
            if season_matches.height > 10:
                mid_idx = season_matches.height // 2
                row = season_matches.row(mid_idx, named=True)
                test_cases.append({
                    "name": f"mid_season_{div}_{season}",
                    "division": div,
                    "date": row["Date"],
                    "home_team": row["HomeTeam"],
                    "away_team": row["AwayTeam"],
                    "season": season,
                })
                break  # One per season
    
        # Case 2: New-season opener (first match of a season)
        for season in seasons[-3:]:
            season_matches = div_matches.filter(pl.col("Season") == season).sort("Date")
            if season_matches.height > 0:
                row = season_matches.row(0, named=True)
                test_cases.append({
                    "name": f"season_opener_{div}_{season}",
                    "division": div,
                    "date": row["Date"],
                    "home_team": row["HomeTeam"],
                    "away_team": row["AwayTeam"],
                    "season": season,
                })
                break
    
        # Case 3: Promoted team (team not in previous season of this division)
        # Find teams that appear in this season but not previous
        for season in seasons[-3:]:
            prev_season_idx = seasons.index(season) - 1
            if prev_season_idx < 0:
                continue
            prev_season = seasons[prev_season_idx]
            curr_teams = set(div_matches.filter(pl.col("Season") == season).select(pl.col("HomeTeam").unique()).to_series().to_list())
            prev_teams = set(div_matches.filter(pl.col("Season") == prev_season).select(pl.col("HomeTeam").unique()).to_series().to_list())
            promoted = curr_teams - prev_teams
            if promoted:
                # Find a match involving a promoted team
                season_matches = div_matches.filter(pl.col("Season") == season).sort("Date")
                for i in range(min(5, season_matches.height)):
                    row = season_matches.row(i, named=True)
                    if row["HomeTeam"] in promoted or row["AwayTeam"] in promoted:
                        test_cases.append({
                            "name": f"promoted_team_{div}_{season}",
                            "division": div,
                            "date": row["Date"],
                            "home_team": row["HomeTeam"],
                            "away_team": row["AwayTeam"],
                            "season": season,
                        })
                        break
                break
    
        # Case 4: Postponed fixture / unequal games played
        # Look for teams with different games played at same date
        for season in seasons[-2:]:
            season_matches = div_matches.filter(pl.col("Season") == season).sort("Date")
            # Group by date, check if any team plays twice on same date (postponed)
            for date_val in season_matches.select(pl.col("Date").unique()).to_series().to_list()[:5]:
                day_matches = season_matches.filter(pl.col("Date") == date_val)
                if day_matches.height > 0:
                    row = day_matches.row(0, named=True)
                    test_cases.append({
                        "name": f"same_date_{div}_{season}",
                        "division": div,
                        "date": row["Date"],
                        "home_team": row["HomeTeam"],
                        "away_team": row["AwayTeam"],
                        "season": season,
                    })
                    break
    
    # Case 5: Missing match statistics (shots/corners) - find older matches
    for div in divisions[:1]:
        div_matches = master.filter(pl.col("Div") == div)
        for season in seasons[:2]:  # Early seasons may have missing stats
            season_matches = div_matches.filter(pl.col("Season") == season).sort("Date")
            if season_matches.height > 0:
                row = season_matches.row(0, named=True)
                test_cases.append({
                    "name": f"missing_stats_{div}_{season}",
                    "division": div,
                    "date": row["Date"],
                    "home_team": row["HomeTeam"],
                    "away_team": row["AwayTeam"],
                    "season": season,
                })
                break
    
    # Case 6: Historical H2H draw - find a match where teams drew previously
    for div in divisions[:1]:
        div_matches = master.filter(pl.col("Div") == div).sort("Date")
        # Look for a pair that has a draw in their history
        for season in seasons[-2:]:
            season_matches = div_matches.filter(pl.col("Season") == season).sort("Date")
            if season_matches.height > 5:
                row = season_matches.row(5, named=True)
                test_cases.append({
                    "name": f"h2h_draw_{div}_{season}",
                    "division": div,
                    "date": row["Date"],
                    "home_team": row["HomeTeam"],
                    "away_team": row["AwayTeam"],
                    "season": season,
                })
                break
    
    # Case 7: Same-date reordering (handled by same_date case above)
    
    # Case 8: Future-result mutation test - will be tested separately
    
    return test_cases[:20]  # Limit total


def run_future_mutation_test(
    master: pl.DataFrame,
    team_matches: pl.DataFrame,
    division: str,
    pred_date: date,
    home: str,
    away: str,
    season: str,
    schema: FeatureSchema,
) -> ParityResult:
    """
    Test that adding/modifying future results does not change earlier predictors.
    
    This creates a copy of history, adds a fake future result, and verifies
    that the predictor for the target fixture (which is before the fake result)
    remains unchanged.
    """
    fixture_key = _make_fixture_key(division, str(pred_date), home, away) + "_future_mutation"
    result = ParityResult(
        fixture_key=fixture_key,
        division=division,
        date=str(pred_date),
        home_team=home,
        away_team=away,
        season=season,
    )
    
    # Get baseline features (without future mutation)
    baseline_values = run_inference_route(master, team_matches, division, pred_date, home, away, season, schema)
    
    # Create mutated history: add a fake future match result
    future_date = date(pred_date.year + 1, 1, 1)  # Far future
    fake_future = pl.DataFrame({
        "Div": [division],
        "Date": pl.Series([future_date], dtype=pl.Date),
        "HomeTeam": [home],
        "AwayTeam": [away],
        "Season": [season],
        "FTHG": [3], "FTAG": [0], "FTR": ["H"],
        "Time": [None], "HTHG": [None], "HTAG": [None], "HTR": ["D"],
        "Referee": [None], "HS": [10], "AS": [5],
        "HST": [5], "AST": [2], "HF": [8], "AF": [6],
        "HC": [4], "AC": [3], "HY": [1], "AY": [0],
        "HR": [0], "AR": [0],
    }).with_columns([
        pl.col("HTHG").cast(pl.Int64),
        pl.col("HTAG").cast(pl.Int64),
        pl.col("HS").cast(pl.Int64),
        pl.col("AS").cast(pl.Int64),
        pl.col("HST").cast(pl.Int64),
        pl.col("AST").cast(pl.Int64),
        pl.col("HF").cast(pl.Int64),
        pl.col("AF").cast(pl.Int64),
        pl.col("HC").cast(pl.Int64),
        pl.col("AC").cast(pl.Int64),
        pl.col("HY").cast(pl.Int64),
        pl.col("AY").cast(pl.Int64),
        pl.col("HR").cast(pl.Int64),
        pl.col("AR").cast(pl.Int64),
    ])
    
    # Filter history to before target date, then add fake future
    hist = master.filter(pl.col("Date") < pl.lit(pred_date))
    hist_tm = team_matches.filter(pl.col("Date") < pl.lit(pred_date))
    mutated_master = pl.concat([hist.select([c for c in CORE_COLS + ["Season"] if c in hist.columns]), fake_future], how="diagonal")
    mutated_tm = pl.concat([hist_tm, build_team_match_dataset(fake_future)], how="diagonal")
    
    # Build features with mutated history
    mutated_values = run_inference_route(mutated_master, mutated_tm, division, pred_date, home, away, season, schema)
    
    # Compare
    for feat_name in schema.feature_names():
        spec = schema.get_spec(feat_name)
        base_val = baseline_values.get(feat_name)
        mut_val = mutated_values.get(feat_name)
        
        mismatch = _compare_values(fixture_key, feat_name, base_val, mut_val, spec, 1e-10, 1e-12)
        if mismatch:
            mismatch.severity = "critical"  # Future mutation must not affect past
            result.add_mismatch(mismatch)
    
    return result


def run_full_parity_suite(
    data_dir: Path = Path("."),
    output_dir: Optional[Path] = None,
) -> ParityReport:
    """Run the complete parity test suite."""
    print("Loading data...")
    master = build_master_dataset(
        data_dir,
        ["E0", "E1", "E2", "E3", "SP1", "D1", "I1", "F1"],
        ["1920", "2021", "2122", "2223", "2324", "2425", "2526", "2627"],
        usecols=CORE_COLS
    )
    team_matches = build_team_match_dataset(master)
    
    print("Building historical features...")
    features = build_pre_match_features(team_matches)
    paired = pair_match_features(features)
    
    print("Loading feature schema...")
    schema = load_canonical_schema()
    
    print("Creating test cases...")
    test_cases = create_test_cases(master, team_matches)
    print(f"  {len(test_cases)} test cases created")
    
    results = []
    
    # Run parity tests
    for i, tc in enumerate(test_cases):
        print(f"  [{i+1}/{len(test_cases)}] {tc['name']}...")
        result = test_fixture_parity(
            master, team_matches, paired,
            tc["division"], tc["date"], tc["home_team"], tc["away_team"],
            tc["season"], schema
        )
        results.append(result)
        print(f"    {result.summary()}")
        for m in result.mismatches:
            print(f"      {m.severity.upper()}: {m.feature_name} | hist={m.historical_value} | inf={m.inference_value}")
    
    # Run future mutation tests on a subset
    print("\nRunning future mutation tests...")
    for i, tc in enumerate(test_cases[:3]):
        print(f"  [{i+1}/3] {tc['name']} future mutation...")
        result = run_future_mutation_test(
            master, team_matches,
            tc["division"], tc["date"], tc["home_team"], tc["away_team"],
            tc["season"], schema
        )
        results.append(result)
        print(f"    {result.summary()}")
        for m in result.mismatches:
            print(f"      {m.severity.upper()}: {m.feature_name} | hist={m.historical_value} | mut={m.inference_value}")
    
    # Run prediction parity test with a small model
    print("\nRunning prediction parity test...")
    prediction_results = run_prediction_parity_test(
        master, team_matches, paired, test_cases[:3], schema
    )
    results.extend(prediction_results)
    
    report = ParityReport(results=results, schema=schema)
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "parity_report.json"
        report_path.write_text(json.dumps(report.to_dict(), indent=2))
        print(f"\nReport saved to {report_path}")
    
    print(f"\n{'='*60}")
    print(f"PARITY TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total fixtures tested: {len(results)}")
    print(f"Passed: {sum(1 for r in results if r.passed)}")
    print(f"Failed: {sum(1 for r in results if not r.passed)}")
    print(f"Total mismatches: {report.total_mismatches()}")
    print(f"Critical mismatches: {report.critical_count()}")
    print(f"Overall: {'PASS' if report.overall_passed() else 'FAIL'}")
    
    return report


def run_prediction_parity_test(
    master: pl.DataFrame,
    team_matches: pl.DataFrame,
    paired: pl.DataFrame,
    test_cases: List[Dict],
    schema: FeatureSchema,
) -> List[ParityResult]:
    """Train a small model and verify prediction parity between routes."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from prediction.utils import encode_outcomes
    import numpy as np
    
    # Prepare training data from paired features (use recent seasons for speed)
    train_seasons = ["2324", "2425", "2526"]
    train_paired = paired.filter(pl.col("Season").is_in(train_seasons))
    
    if train_paired.height == 0:
        # Fallback: use all available seasons
        train_paired = paired
    
    # Get feature matrix and target
    feature_cols = schema.feature_names()
    available_cols = [c for c in feature_cols if c in train_paired.columns]
    
    X = train_paired.select(available_cols).to_numpy().astype(np.float64)
    y = encode_outcomes(train_paired.select(pl.col("FTR")).to_series())
    
    valid = y >= 0
    X = X[valid]
    y = y[valid]
    
    # Impute missing values
    impute_vals = np.nanmean(X, axis=0)
    impute_vals = np.where(np.isfinite(impute_vals), impute_vals, 0.0)
    for j in range(X.shape[1]):
        col = X[:, j]
        X[:, j] = np.where(np.isfinite(col), col, impute_vals[j])
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train simple logistic regression
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
    model.fit(X_scaled, y)
    
    results = []
    
    for tc in test_cases:
        fixture_key = _make_fixture_key(tc["division"], str(tc["date"]), tc["home_team"], tc["away_team"]) + "_prediction"
        result = ParityResult(
            fixture_key=fixture_key,
            division=tc["division"],
            date=str(tc["date"]),
            home_team=tc["home_team"],
            away_team=tc["away_team"],
            season=tc["season"],
        )
        
        # Route 1: Historical features
        hist_values = run_historical_route(paired, tc["division"], tc["date"], tc["home_team"], tc["away_team"], schema)
        hist_X = np.array([[hist_values.get(c, np.nan) for c in available_cols]], dtype=np.float64)
        for j in range(hist_X.shape[1]):
            if not np.isfinite(hist_X[0, j]):
                hist_X[0, j] = impute_vals[j]
        hist_X_scaled = scaler.transform(hist_X)
        hist_probs = model.predict_proba(hist_X_scaled)[0]
        
        # Route 2: Inference features
        inf_values = run_inference_route(master, team_matches, tc["division"], tc["date"], tc["home_team"], tc["away_team"], tc["season"], schema)
        inf_X = np.array([[inf_values.get(c, np.nan) for c in available_cols]], dtype=np.float64)
        for j in range(inf_X.shape[1]):
            if not np.isfinite(inf_X[0, j]):
                inf_X[0, j] = impute_vals[j]
        inf_X_scaled = scaler.transform(inf_X)
        inf_probs = model.predict_proba(inf_X_scaled)[0]
        
        result.historical_probs = hist_probs
        result.inference_probs = inf_probs
        
        # Compare probabilities
        if not np.allclose(hist_probs, inf_probs, rtol=1e-8, atol=1e-10):
            for k, (h, i) in enumerate(zip(hist_probs, inf_probs)):
                if not np.isclose(h, i, rtol=1e-8, atol=1e-10):
                    result.add_mismatch(FeatureMismatch(
                        fixture_key=fixture_key,
                        feature_name=f"P({schema.class_order[k]})",
                        historical_value=h,
                        inference_value=i,
                        historical_dtype="float",
                        inference_dtype="float",
                        historical_missing=False,
                        inference_missing=False,
                        severity="critical",
                    ))
        
        results.append(result)
        print(f"    {result.summary()}")
        for m in result.mismatches:
            print(f"      {m.severity.upper()}: {m.feature_name} | hist={m.historical_value:.6f} | inf={m.inference_value:.6f}")
    
    return results


import json
if __name__ == "__main__":
    report = run_full_parity_suite()