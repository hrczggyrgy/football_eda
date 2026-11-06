"""
Prediction inference interface.

Provides:
- predict_fixture(division, date, home_team, away_team, ...) -> structured prediction.
- CSV-based batch inference for upcoming fixtures.
- Model persistence (joblib bundle + metadata JSON).
- Reproducible inference using a fitted model and a historical cutoff.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import joblib
import numpy as np
import polars as pl

from prediction.utils import (
    DEFAULT_SEED, MAX_GOALS, OUTCOME_LABELS, REPORTED_SCORELINES,
    OVER_UNDER_THRESHOLDS, set_seed, validate_outcome_distribution,
)
from prediction.features import build_pre_match_features, pair_match_features
from prediction.distributions import (
    scoreline_probabilities, derived_market_probs,
    goal_prediction_intervals, most_probable_scoreline,
    poisson_1x2_from_lambdas,
)
from prediction.calibration import brier_score_multiclass
from data.load_data import CORE_COLS


@dataclass
class FixturePrediction:
    """Structured prediction for a single fixture."""
    date: str
    league: str
    home_team: str
    away_team: str

    home_win_probability: float
    draw_probability: float
    away_win_probability: float

    home_expected_goals: Optional[float]
    away_expected_goals: Optional[float]
    total_expected_goals: Optional[float]

    most_probable_scoreline: Tuple[int, int]
    most_probable_scoreline_probability: float

    both_teams_to_score_probability: float
    over_1_5_probability: float
    over_2_5_probability: float
    over_3_5_probability: float

    home_goals_interval: Tuple[int, int]
    away_goals_interval: Tuple[int, int]
    total_goals_interval: Tuple[int, int]
    interval_coverage: float

    model_version: str
    training_cutoff: Optional[str]
    data_quality_status: str
    feature_warnings: List[str] = field(default_factory=list)

    def outcome_probs_sum(self) -> float:
        return self.home_win_probability + self.draw_probability + self.away_win_probability

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "league": self.league,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "home_win_probability": round(self.home_win_probability, 4),
            "draw_probability": round(self.draw_probability, 4),
            "away_win_probability": round(self.away_win_probability, 4),
            "outcome_probs_sum": round(self.outcome_probs_sum(), 6),
            "home_expected_goals": (round(self.home_expected_goals, 4)
                                    if self.home_expected_goals is not None else None),
            "away_expected_goals": (round(self.away_expected_goals, 4)
                                    if self.away_expected_goals is not None else None),
            "total_expected_goals": (round(self.total_expected_goals, 4)
                                     if self.total_expected_goals is not None else None),
            "most_probable_scoreline": list(self.most_probable_scoreline),
            "most_probable_scoreline_probability": round(self.most_probable_scoreline_probability, 4),
            "both_teams_to_score_probability": round(self.both_teams_to_score_probability, 4),
            "over_1_5_probability": round(self.over_1_5_probability, 4),
            "over_2_5_probability": round(self.over_2_5_probability, 4),
            "over_3_5_probability": round(self.over_3_5_probability, 4),
            "home_goals_interval": list(self.home_goals_interval),
            "away_goals_interval": list(self.away_goals_interval),
            "total_goals_interval": list(self.total_goals_interval),
            "interval_coverage": self.interval_coverage,
            "model_version": self.model_version,
            "training_cutoff": self.training_cutoff,
            "data_quality_status": self.data_quality_status,
            "feature_warnings": self.feature_warnings,
        }


class PredictionBundle:
    """A fitted prediction model plus its metadata and historical state."""

    def __init__(self, model, model_name: str, metadata: dict,
                 feature_names: List[str], cutoff_date: Optional[str] = None,
                 cutoff_season: Optional[str] = None):
        self.model = model
        self.model_name = model_name
        self.metadata = metadata
        self.feature_names = feature_names
        self.cutoff_date = cutoff_date
        self.cutoff_season = cutoff_season
        self.model_version = metadata.get("model_version", "1.0.0")

    @classmethod
    def load(cls, path: Path) -> "PredictionBundle":
        """Load a saved model bundle (joblib) and sidecar metadata JSON."""
        path = Path(path)
        bundle = joblib.load(path)
        meta_path = path.with_suffix(".metadata.json")
        metadata = {}
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text())
        return cls(
            model=bundle["model"],
            model_name=bundle.get("model_name", "unknown"),
            metadata=metadata,
            feature_names=bundle.get("feature_names", []),
            cutoff_date=metadata.get("training_cutoff"),
            cutoff_season=metadata.get("cutoff_season"),
        )

    def save(self, path: Path) -> None:
        """Save the model bundle (joblib) and sidecar metadata JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "model": self.model,
            "model_name": self.model_name,
            "feature_names": self.feature_names,
        }
        joblib.dump(bundle, path)
        meta_path = path.with_suffix(".metadata.json")
        meta_path.write_text(json.dumps(self.metadata, indent=2))


def _empty_prediction(date: str, league: str, home: str, away: str,
                      model_version: str, cutoff: Optional[str],
                      extra_warning: Optional[str] = None) -> FixturePrediction:
    """Return a fallback prediction with neutral probabilities and recorded warnings."""
    warnings = ["Unknown team or league; neutral prior used"]
    if extra_warning:
        warnings.append(extra_warning)
    return FixturePrediction(
        date=date, league=league, home_team=home, away_team=away,
        home_win_probability=1.0 / 3, draw_probability=1.0 / 3,
        away_win_probability=1.0 / 3,
        home_expected_goals=None, away_expected_goals=None,
        total_expected_goals=None,
        most_probable_scoreline=(1, 1), most_probable_scoreline_probability=1.0 / 3,
        both_teams_to_score_probability=1.0 / 3,
        over_1_5_probability=1.0 / 3, over_2_5_probability=1.0 / 3,
        over_3_5_probability=1.0 / 3,
        home_goals_interval=(0, 3), away_goals_interval=(0, 3),
        total_goals_interval=(0, 6), interval_coverage=0.8,
        model_version=model_version, training_cutoff=cutoff,
        data_quality_status="fallback_unknown_team_or_league",
        feature_warnings=warnings,
    )


def _season_from_date(pred_date: date) -> str:
    """Derive a season code ``YYZZ`` from a fixture date.

    A football season runs roughly August of year YY to May of year ZZ. A
    fixture on/after 1 August belongs to the season starting that calendar
    year; earlier in the year it belongs to the season started the previous
    calendar year (e.g. 2021-09-10 -> "2122", 2022-03-15 -> "2122").
    """
    start_year = pred_date.year if pred_date.month >= 8 else pred_date.year - 1
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def _season_for_fixture(hist: pl.DataFrame, pred_date: date, season: Optional[str]) -> str:
    """Resolve the fixture's season.

    Prefers the caller-supplied season (actual season from the fixture source).
    Otherwise derives it from the fixture date. The derived season is preserved
    even when it has no historical rows yet — a brand-new season's opener must
    keep its own season so prior-season strength enters through prior features,
    not by relabeling the fixture onto an earlier season.
    """
    if season is not None:
        return season
    return _season_from_date(pred_date)


def predict_fixture(
    division: str,
    date,
    home_team: str,
    away_team: str,
    bundle: PredictionBundle,
    history_matches: pl.DataFrame,
    history_team_matches: pl.DataFrame,
    cutoff_date=None,
    season: Optional[str] = None,
) -> FixturePrediction:
    """Produce a structured prediction for a single upcoming fixture.

    Uses only information available before the fixture (pre-match features built
    from history up to but not including the fixture date). Unknown teams or
    unsupported leagues receive a documented fallback prior rather than a silent
    forecast.

    ``season`` is the fixture's actual season (e.g. "2627"). It should be
    supplied by the fixture source. When omitted it is derived from the fixture
    date; never inferred as the latest season in history, which would misassign
    season openers and cross-season fixtures.
    """
    set_seed(DEFAULT_SEED)
    date_str = str(date) if not isinstance(date, str) else date
    if cutoff_date is None:
        cutoff_date = bundle.cutoff_date
    cutoff_str = str(cutoff_date) if cutoff_date is not None else None

    # Enforce the model-training cutoff: never provide an out-of-sample
    # prediction for a fixture at or before the bundle's training cutoff.
    bundle_cutoff = None
    if cutoff_str:
        try:
            bundle_cutoff = datetime.strptime(cutoff_str[:10], "%Y-%m-%d").date()
        except ValueError:
            bundle_cutoff = None

    warnings: List[str] = []

    # Unknown league check.
    known_leagues = set()
    if "Div" in history_matches.columns:
        known_leagues = set(history_matches.select(pl.col("Div").unique()).to_series().to_list())
    if division not in known_leagues:
        warnings.append(f"Unsupported league '{division}'; neutral prior used")
        return _empty_prediction(date_str, division, home_team, away_team,
                                 bundle.model_version, cutoff_str)

    # Unknown team check (team never seen in this division).
    known_teams = set()
    if {"Div", "Team"} <= set(history_team_matches.columns):
        known_teams = set(history_team_matches.filter(
            pl.col("Div") == division
        ).select(pl.col("Team").unique()).to_series().to_list())
    unknown = []
    if home_team not in known_teams:
        unknown.append(home_team)
    if away_team not in known_teams:
        unknown.append(away_team)
    if unknown:
        warnings.append(f"Unknown team(s) in {division}: {unknown}; fallback prior used")

    # Build a synthetic target row so the feature engine can compute pre-match
    # features from the available history (history must end before the fixture).
    # Use history truncated to strictly before the fixture date to honor the
    # same-day safeguard (no same-date results leak into the target's features).
    from datetime import date as _date
    try:
        if "-" in date_str:
            # ISO format: 2026-11-04
            pred_date = _date.fromisoformat(date_str)
        else:
            # DD/MM/YYYY (4-digit year), e.g. 04/11/2026
            pred_date = datetime.strptime(date_str, "%d/%m/%Y").date()
    except (ValueError, AttributeError):
        # football-data.co.uk style DD/MM/YY (2-digit year), e.g. 04/11/26.
        # Maps 00-49 -> 2000-2049, 50-99 -> 1950-1999.
        try:
            pred_date = datetime.strptime(date_str, "%d/%m/%y").date()
        except (ValueError, AttributeError):
            pred_date = _date.fromisoformat(str(date_str)[:10])

    if bundle_cutoff is not None and pred_date <= bundle_cutoff:
        raise ValueError(
            "This bundle cannot provide an out-of-sample prediction "
            f"for a fixture at or before its training cutoff ({cutoff_str})."
        )

    # History strictly before the fixture date (same-day safeguard).
    select_cols = [c for c in CORE_COLS + ["Season"] if c in history_matches.columns]
    hist = history_matches.filter(pl.col("Date") < pl.lit(pred_date)).select(select_cols)
    # Use the fixture's actual season for the synthetic target row so that
    # team-season rolling features carry over from the correct season history
    # (never silently the latest season in history).
    target_season = _season_for_fixture(hist, pred_date, season)

    target_row = pl.DataFrame({
        "Div": [division], "Date": pl.Series([pred_date], dtype=pl.Date),
        "HomeTeam": [home_team], "AwayTeam": [away_team],
        "Season": [target_season],
        "FTHG": [None], "FTAG": [None], "FTR": [None],
        "Time": [None], "HTHG": [None], "HTAG": [None], "HTR": [None],
        "Referee": [None], "HS": [None], "AS": [None], "HST": [None], "AST": [None],
        "HF": [None], "AF": [None], "HC": [None], "AC": [None],
        "HY": [None], "AY": [None], "HR": [None], "AR": [None],
    })

    combined = pl.concat([hist, target_row], how="diagonal")

    # Convert to team-perspective, then features, then pair home/away.
    from data.load_data import build_team_match_dataset  # reuse existing loader

    model = bundle.model
    feature_names = bundle.feature_names

    # Fallback if we cannot build features.
    if unknown or division not in known_leagues:
        return _empty_prediction(date_str, division, home_team, away_team,
                                 bundle.model_version, cutoff_str)

    # Team-perspective frame (history + target), then pre-match features
    # (lags only), then pair home/away into a match-level row.
    team_matches_all = build_team_match_dataset(combined).sort(
        ["Div", "Team", "Season", "Date"]
    )

    feat = build_pre_match_features(team_matches_all)
    paired = pair_match_features(feat)

    m = paired.filter(
        (pl.col("Div") == division) &
        (pl.col("Date") == pl.lit(pred_date)) &
        (pl.col("HomeTeam") == home_team) &
        (pl.col("AwayTeam") == away_team)
    )

    if m.height == 0:
        return _empty_prediction(date_str, division, home_team, away_team,
                                 bundle.model_version, cutoff_str,
                                 extra_warning="No pre-match features computable")

    res = model.predict(m)
    probs = validate_outcome_distribution(res.outcome_probs[0])
    p_home, p_draw, p_away = float(probs[0]), float(probs[1]), float(probs[2])

    hl = float(res.home_lambda[0]) if res.home_lambda is not None else None
    al = float(res.away_lambda[0]) if res.away_lambda is not None else None

    if hl is None or al is None:
        # Model only gives outcome probs; derive lambdas by matching to 1X2 is
        # not identifiable, so use league-average lambdas.
        hl, al = 1.3, 1.1
        warnings.append("Model does not produce expected goals; league-average lambdas used")

    # Scoreline & derived markets.
    scoreline_probs = scoreline_probabilities(hl, al, max_goals=MAX_GOALS)
    derived = derived_market_probs(scoreline_probs)
    best_score, best_prob = most_probable_scoreline(hl, al, max_goals=MAX_GOALS)
    intervals = goal_prediction_intervals(hl, al, coverage=0.8)

    # Data quality status.
    dq = "complete"
    if warnings:
        dq = "partial_fallback"

    return FixturePrediction(
        date=date_str, league=division, home_team=home_team, away_team=away_team,
        home_win_probability=p_home, draw_probability=p_draw,
        away_win_probability=p_away,
        home_expected_goals=hl, away_expected_goals=al,
        total_expected_goals=(hl + al),
        most_probable_scoreline=(int(best_score[0]), int(best_score[1])),
        most_probable_scoreline_probability=float(best_prob),
        both_teams_to_score_probability=derived["btts"],
        over_1_5_probability=derived["over_1_5"],
        over_2_5_probability=derived["over_2_5"],
        over_3_5_probability=derived["over_3_5"],
        home_goals_interval=intervals["home_goals"],
        away_goals_interval=intervals["away_goals"],
        total_goals_interval=intervals["total_goals"],
        interval_coverage=intervals["coverage"],
        model_version=bundle.model_version,
        training_cutoff=cutoff_str,
        data_quality_status=dq,
        feature_warnings=warnings,
    )


def predict_from_csv(
    csv_path: Path,
    bundle: PredictionBundle,
    history_matches: pl.DataFrame,
    history_team_matches: pl.DataFrame,
    output_path: Optional[Path] = None,
) -> List[FixturePrediction]:
    """Batch-predict upcoming fixtures from a CSV with columns
    Date,Div,HomeTeam,AwayTeam. The CSV must NOT contain match results."""
    import csv

    csv_path = Path(csv_path)
    predictions: List[FixturePrediction] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"Date", "Div", "HomeTeam", "AwayTeam"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"CSV must contain columns {required}")
        for row in reader:
            pred = predict_fixture(
                division=row["Div"], date=row["Date"],
                home_team=row["HomeTeam"], away_team=row["AwayTeam"],
                bundle=bundle, history_matches=history_matches,
                history_team_matches=history_team_matches,
            )
            predictions.append(pred)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write("date,league,home_team,away_team,")
            f.write("home_win_probability,draw_probability,away_win_probability,")
            f.write("outcome_probs_sum,")
            f.write("home_expected_goals,away_expected_goals,total_expected_goals,")
            f.write("most_probable_scoreline,most_probable_scoreline_probability,")
            f.write("both_teams_to_score_probability,over_1_5_probability,")
            f.write("over_2_5_probability,over_3_5_probability,")
            f.write("home_goals_interval,away_goals_interval,total_goals_interval,")
            f.write("model_version,training_cutoff,data_quality_status\n")
            for p in predictions:
                f.write(",".join([
                    p.date, p.league, p.home_team, p.away_team,
                    f"{p.home_win_probability:.4f}", f"{p.draw_probability:.4f}",
                    f"{p.away_win_probability:.4f}", f"{p.outcome_probs_sum():.6f}",
                    str(p.home_expected_goals), str(p.away_expected_goals),
                    str(p.total_expected_goals),
                    f"{p.most_probable_scoreline[0]}-{p.most_probable_scoreline[1]}",
                    f"{p.most_probable_scoreline_probability:.4f}",
                    f"{p.both_teams_to_score_probability:.4f}",
                    f"{p.over_1_5_probability:.4f}",
                    f"{p.over_2_5_probability:.4f}",
                    f"{p.over_3_5_probability:.4f}",
                    f"{p.home_goals_interval[0]}-{p.home_goals_interval[1]}",
                    f"{p.away_goals_interval[0]}-{p.away_goals_interval[1]}",
                    f"{p.total_goals_interval[0]}-{p.total_goals_interval[1]}",
                    p.model_version, str(p.training_cutoff), p.data_quality_status,
                ]) + "\n")
    return predictions
