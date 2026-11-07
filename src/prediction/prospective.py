"""
Prospective evaluation infrastructure.

This module supports the prospective evaluation gate:
1. Freeze model, configuration, and evaluation rules
2. Define future fixture population in advance
3. Issue and append predictions before kickoff
4. Never overwrite issued predictions
5. Store baseline predictions alongside challenger
6. Attach outcomes later for evaluation

The prediction ledger is append-only and records all predictions with timestamps.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import uuid
import polars as pl
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prediction.inference import PredictionBundle, predict_fixture, FixturePrediction
from prediction.feature_schema import load_canonical_schema


@dataclass
class ProspectiveConfig:
    """Frozen configuration for prospective evaluation."""
    model_name: str
    model_version: str
    model_bundle_path: str
    feature_schema_version: str
    evaluation_rules: Dict[str, Any]
    cutoff_date: str  # Training cutoff
    freeze_timestamp: str  # When this config was frozen
    config_hash: str  # Hash of config for integrity
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "ProspectiveConfig":
        return cls(**d)
    
    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))
    
    @classmethod
    def load(cls, path: Path) -> "ProspectiveConfig":
        return cls.from_dict(json.loads(path.read_text()))


@dataclass
class PredictionRecord:
    """A single prediction record in the prospective ledger."""
    record_id: str  # UUID
    prediction_timestamp: str  # When prediction was issued (ISO format)
    fixture_date: str  # Match date
    fixture_time: Optional[str]  # Match time if known
    league: str
    season: str
    home_team: str
    away_team: str
    
    # Model identification
    model_name: str
    model_version: str
    model_bundle_hash: str
    
    # Cutoffs for reproducibility
    training_cutoff: str
    feature_cutoff: str
    
    # Probabilities
    p_home: float
    p_draw: float
    p_away: float
    
    # Expected goals (if available)
    home_lambda: Optional[float]
    away_lambda: Optional[float]
    
    # Derived markets
    btts_prob: Optional[float]
    over_1_5_prob: Optional[float]
    over_2_5_prob: Optional[float]
    over_3_5_prob: Optional[float]
    
    # Most probable scoreline
    most_probable_scoreline: Optional[str]
    most_probable_scoreline_prob: Optional[float]
    
    # Data quality
    fallback_status: str
    feature_warnings: List[str] = field(default_factory=list)
    
    # Outcome (filled later)
    actual_outcome: Optional[int] = None  # 0=H, 1=D, 2=A
    actual_home_goals: Optional[int] = None
    actual_away_goals: Optional[int] = None
    outcome_attached_timestamp: Optional[str] = None
    
    # Verification
    prediction_hash: str = ""  # Hash of prediction for tamper detection
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "PredictionRecord":
        return cls(**d)


class ProspectiveLedger:
    """Append-only ledger for prospective predictions.
    
    Records predictions before kickoff. Never overwrites existing records.
    Supports attaching outcomes after matches are played.
    """
    
    def __init__(self, ledger_path: Path, config: ProspectiveConfig):
        self.ledger_path = ledger_path
        self.config = config
        self.records: List[PredictionRecord] = []
        self._load_existing()
    
    def _load_existing(self) -> None:
        if self.ledger_path.exists():
            try:
                data = json.loads(self.ledger_path.read_text())
                self.records = [PredictionRecord.from_dict(r) for r in data]
            except Exception:
                self.records = []
    
    def _save(self) -> None:
        self.ledger_path.write_text(json.dumps([r.to_dict() for r in self.records], indent=2))
    
    def _compute_prediction_hash(self, record: PredictionRecord) -> str:
        """Compute hash of prediction for tamper detection."""
        import hashlib
        # Hash key fields only
        key_data = f"{record.record_id}|{record.fixture_date}|{record.league}|{record.home_team}|{record.away_team}|{record.p_home:.6f}|{record.p_draw:.6f}|{record.p_away:.6f}|{record.model_version}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]
    
    def _fixture_key(self, league: str, fixture_date: str, home: str, away: str) -> str:
        return f"{league}|{fixture_date}|{home}|{away}"
    
    def has_prediction(self, league: str, fixture_date: str, home: str, away: str) -> bool:
        """Check if prediction already exists for this fixture."""
        key = self._fixture_key(league, fixture_date, home, away)
        return any(self._fixture_key(r.league, r.fixture_date, r.home_team, r.away_team) == key for r in self.records)
    
    def add_prediction(
        self,
        fixture_date: str,
        league: str,
        season: str,
        home_team: str,
        away_team: str,
        prediction: FixturePrediction,
        fixture_time: Optional[str] = None,
        model_bundle_hash: Optional[str] = None,
    ) -> PredictionRecord:
        """Add a new prediction to the ledger (append-only)."""
        if self.has_prediction(league, fixture_date, home_team, away_team):
            raise ValueError(f"Prediction already exists for {league} {fixture_date} {home_team} vs {away_team}")
        
        record = PredictionRecord(
            record_id=str(uuid.uuid4())[:8],
            prediction_timestamp=datetime.now().isoformat(),
            fixture_date=fixture_date,
            fixture_time=fixture_time,
            league=league,
            season=season,
            home_team=home_team,
            away_team=away_team,
            model_name=prediction.model_version,
            model_version=prediction.model_version,
            model_bundle_hash=model_bundle_hash or "unknown",
            training_cutoff=prediction.training_cutoff or self.config.cutoff_date,
            feature_cutoff=prediction.training_cutoff or self.config.cutoff_date,
            p_home=prediction.home_win_probability,
            p_draw=prediction.draw_probability,
            p_away=prediction.away_win_probability,
            home_lambda=prediction.home_expected_goals,
            away_lambda=prediction.away_expected_goals,
            btts_prob=prediction.both_teams_to_score_probability,
            over_1_5_prob=prediction.over_1_5_probability,
            over_2_5_prob=prediction.over_2_5_probability,
            over_3_5_prob=prediction.over_3_5_probability,
            most_probable_scoreline=f"{prediction.most_probable_scoreline[0]}-{prediction.most_probable_scoreline[1]}",
            most_probable_scoreline_prob=prediction.most_probable_scoreline_probability,
            fallback_status=prediction.data_quality_status,
            feature_warnings=prediction.feature_warnings,
        )
        record.prediction_hash = self._compute_prediction_hash(record)
        
        self.records.append(record)
        self._save()
        return record
    
    def attach_outcome(
        self,
        league: str,
        fixture_date: str,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
    ) -> bool:
        """Attach actual outcome to a prediction record."""
        key = self._fixture_key(league, fixture_date, home_team, away_team)
        for record in self.records:
            if self._fixture_key(record.league, record.fixture_date, record.home_team, record.away_team) == key:
                if record.actual_outcome is not None:
                    raise ValueError(f"Outcome already attached for {key}")
                
                # Determine outcome: 0=H, 1=D, 2=A
                if home_goals > away_goals:
                    outcome = 0
                elif home_goals < away_goals:
                    outcome = 2
                else:
                    outcome = 1
                
                record.actual_outcome = outcome
                record.actual_home_goals = home_goals
                record.actual_away_goals = away_goals
                record.outcome_attached_timestamp = datetime.now().isoformat()
                self._save()
                return True
        return False
    
    def get_pending_predictions(self, before_date: Optional[str] = None) -> List[PredictionRecord]:
        """Get predictions that don't have outcomes attached yet."""
        pending = [r for r in self.records if r.actual_outcome is None]
        if before_date:
            pending = [r for r in pending if r.fixture_date < before_date]
        return pending
    
    def get_evaluable_predictions(self) -> List[PredictionRecord]:
        """Get predictions with outcomes attached."""
        return [r for r in self.records if r.actual_outcome is not None]
    
    def to_dataframe(self) -> pl.DataFrame:
        """Convert ledger to Polars DataFrame for analysis."""
        if not self.records:
            return pl.DataFrame()
        return pl.DataFrame([r.to_dict() for r in self.records])
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute evaluation metrics on predictions with outcomes."""
        from prediction.calibration import brier_score_multiclass, log_loss_multiclass
        
        eval_records = self.get_evaluable_predictions()
        if not eval_records:
            return {}
        
        probs = np.array([[r.p_home, r.p_draw, r.p_away] for r in eval_records])
        outcomes = np.array([r.actual_outcome for r in eval_records])
        
        return {
            "n": len(eval_records),
            "brier": float(brier_score_multiclass(probs, outcomes)),
            "log_loss": float(log_loss_multiclass(probs, outcomes)),
            "accuracy": float((probs.argmax(axis=1) == outcomes).mean()),
        }
    
    def compute_metrics_by_league(self) -> pl.DataFrame:
        """Compute metrics by league."""
        from prediction.calibration import brier_score_multiclass, log_loss_multiclass
        
        eval_records = self.get_evaluable_predictions()
        if not eval_records:
            return pl.DataFrame()
        
        df = pl.DataFrame([r.to_dict() for r in eval_records])
        results = []
        for league in df.select(pl.col("league").unique()).to_series().to_list():
            league_df = df.filter(pl.col("league") == league)
            n = league_df.height
            probs = np.column_stack([
                league_df["p_home"].to_numpy(),
                league_df["p_draw"].to_numpy(),
                league_df["p_away"].to_numpy(),
            ])
            outcomes = league_df["actual_outcome"].to_numpy()
            results.append({
                "league": league,
                "n": n,
                "brier": round(brier_score_multiclass(probs, outcomes), 5),
                "log_loss": round(log_loss_multiclass(probs, outcomes), 5),
                "accuracy": round(float((probs.argmax(axis=1) == outcomes).mean()), 5),
            })
        return pl.DataFrame(results)


def create_prospective_config(
    model_bundle_path: Path,
    output_dir: Path,
) -> ProspectiveConfig:
    """Create a frozen prospective configuration from a trained model bundle."""
    import hashlib
    
    bundle = PredictionBundle.load(model_bundle_path)
    metadata = bundle.metadata
    
    # Evaluation rules (frozen)
    evaluation_rules = {
        "primary_metric": "brier_score_multiclass",
        "secondary_metrics": ["log_loss_multiclass", "accuracy"],
        "min_fixtures_per_league": 10,
        "min_total_fixtures": 100,
        "evaluation_window_weeks": 4,
        "acceptance_criteria": {
            "brier_vs_baseline_threshold": -0.01,  # Must beat baseline by at least 0.01
            "majority_seasons_improve": True,
            "log_loss_not_worse": True,
        },
    }
    
    # Create config
    config = ProspectiveConfig(
        model_name=metadata.get("model_name", "multinomial_logistic"),
        model_version=metadata.get("model_version", "1.0.0"),
        model_bundle_path=str(model_bundle_path),
        feature_schema_version="1.0.0",
        evaluation_rules=evaluation_rules,
        cutoff_date=metadata.get("training_cutoff", ""),
        freeze_timestamp=datetime.now().isoformat(),
        config_hash="",  # Will be filled
    )
    
    # Compute config hash
    config_json = json.dumps(config.to_dict(), sort_keys=True)
    config.config_hash = hashlib.sha256(config_json.encode()).hexdigest()[:16]
    
    # Save
    config_path = output_dir / "prospective_config.json"
    config.save(config_path)
    
    return config


def run_prospective_batch(
    fixtures_csv: Path,
    model_bundle_path: Path,
    history_matches: pl.DataFrame,
    history_team_matches: pl.DataFrame,
    ledger: ProspectiveLedger,
) -> List[PredictionRecord]:
    """Run batch predictions for fixtures from CSV and add to ledger."""
    import csv
    
    bundle = PredictionBundle.load(model_bundle_path)
    model_bundle_hash = ""
    try:
        import joblib
        bundle_data = joblib.load(model_bundle_path)
        import hashlib
        model_bundle_hash = hashlib.sha256(str(bundle_data).encode()).hexdigest()[:16]
    except Exception:
        pass
    
    new_records = []
    with open(fixtures_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Parse date
                date_str = row["Date"]
                if "-" in date_str and len(date_str) == 10:
                    fixture_date = date_str
                elif "/" in date_str:
                    # Try DD/MM/YYYY or DD/MM/YY
                    parts = date_str.split("/")
                    if len(parts[2]) == 2:
                        fixture_date = f"20{parts[2]}-{parts[1]}-{parts[0]}"
                    else:
                        fixture_date = f"{parts[2]}-{parts[1]}-{parts[0]}"
                else:
                    fixture_date = date_str
                
                # Check if already predicted
                if ledger.has_prediction(row["Div"], fixture_date, row["HomeTeam"], row["AwayTeam"]):
                    print(f"  Skipping {row['Div']} {fixture_date} {row['HomeTeam']} vs {row['AwayTeam']} (already predicted)")
                    continue
                
                # Determine season
                pred_date = datetime.fromisoformat(fixture_date).date()
                from prediction.inference import _season_from_date
                season = _season_from_date(pred_date)
                
                # Make prediction
                prediction = predict_fixture(
                    division=row["Div"],
                    date=fixture_date,
                    home_team=row["HomeTeam"],
                    away_team=row["AwayTeam"],
                    bundle=bundle,
                    history_matches=history_matches,
                    history_team_matches=history_team_matches,
                    season=season,
                )
                
                # Add to ledger
                record = ledger.add_prediction(
                    fixture_date=fixture_date,
                    league=row["Div"],
                    season=season,
                    home_team=row["HomeTeam"],
                    away_team=row["AwayTeam"],
                    prediction=prediction,
                    model_bundle_hash=model_bundle_hash,
                )
                new_records.append(record)
                print(f"  Predicted {row['Div']} {fixture_date} {row['HomeTeam']} vs {row['AwayTeam']}: "
                      f"H={prediction.home_win_probability:.3f} D={prediction.draw_probability:.3f} A={prediction.away_win_probability:.3f}")
                
            except Exception as e:
                print(f"  ERROR predicting {row}: {e}")
    
    return new_records


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.load_data import build_master_dataset, build_team_match_dataset, CORE_COLS
    
    if len(sys.argv) < 3:
        print("Usage: python -m src.prediction.prospective <fixtures_csv> <model_bundle> [--data-dir DIR] [--output-dir DIR]")
        sys.exit(1)
    
    fixtures_csv = Path(sys.argv[1])
    model_bundle_path = Path(sys.argv[2])
    data_dir = Path(".")
    output_dir = Path("output/prospective")
    
    for i, arg in enumerate(sys.argv):
        if arg == "--data-dir" and i + 1 < len(sys.argv):
            data_dir = Path(sys.argv[i + 1])
        elif arg == "--output-dir" and i + 1 < len(sys.argv):
            output_dir = Path(sys.argv[i + 1])
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading historical data...")
    master = build_master_dataset(data_dir, ["E0", "E1", "E2", "E3", "SP1", "D1", "I1", "F1"], 
                                  ["1920", "2021", "2122", "2223", "2324", "2425", "2526", "2627"], usecols=CORE_COLS)
    team_matches = build_team_match_dataset(master)
    
    print("Creating prospective config...")
    config = create_prospective_config(model_bundle_path, output_dir)
    print(f"  Config hash: {config.config_hash}")
    print(f"  Model: {config.model_name} v{config.model_version}")
    print(f"  Training cutoff: {config.cutoff_date}")
    
    print("Initializing ledger...")
    ledger = ProspectiveLedger(output_dir / "prediction_ledger.json", config)
    
    print(f"Running predictions from {fixtures_csv}...")
    new_records = run_prospective_batch(fixtures_csv, model_bundle_path, master, team_matches, ledger)
    
    print(f"\nAdded {len(new_records)} new predictions to ledger.")
    print(f"Total records: {len(ledger.records)}")
    print(f"Ledger saved to: {output_dir / 'prediction_ledger.json'}")
    print(f"Config saved to: {output_dir / 'prospective_config.json'}")