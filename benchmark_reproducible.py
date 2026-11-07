#!/usr/bin/env python3
"""
Benchmark Reproducibility Script

Runs the full model ladder with fixed configuration, producing:
- Keyed predictions with fixture identifiers (Div, Season, Date, HomeTeam, AwayTeam)
- Data and code hashes for reproducibility
- Log loss as primary metric, Brier score as companion
- Calibration comparison (uncalibrated vs chronological calibration)
- Per-league and per-season breakdowns

Usage:
    python benchmark_reproducible.py --output-dir output/benchmark
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import polars as pl

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.load_data import build_master_dataset, build_team_match_dataset, CORE_COLS
from prediction.features import build_pre_match_features, pair_match_features
from prediction.models import (
    HistoricalBaseline, PoissonModel, MultinomialModel,
    SmallLogisticModel, EloLogisticModel, FittedPoissonModel,
    RestrictedHGBModel,
)
from prediction.validation_v2 import (
    run_chronological_validation, compare_models_validation,
    keyed_predictions_to_dataframe, compute_league_metrics,
)
from prediction.calibration import (
    brier_score_multiclass, log_loss_multiclass,
    expected_calibration_error,
)
from prediction.utils import encode_outcomes, set_seed, DEFAULT_SEED


# Fixed benchmark configuration
BENCHMARK_VERSION = "1.0.0"
MODEL_VERSION = "1.0.0"

# Standard leagues and seasons for benchmark
BENCHMARK_LEAGUES = ["E0", "E1", "E2", "E3", "SP1", "D1", "I1", "F1"]
BENCHMARK_SEASONS = ["2122", "2223", "2324", "2425", "2526"]

# Fixed model configurations (no tuning for reproducibility)
BENCHMARK_MODELS = {
    "historical_baseline": {
        "factory": "HistoricalBaseline",
        "params": {"seed": DEFAULT_SEED},
    },
    "poisson_independent": {
        "factory": "PoissonModel",
        "params": {"seed": DEFAULT_SEED, "use_dixon_coles": False},
    },
    "poisson_dixon_coles": {
        "factory": "PoissonModel",
        "params": {"seed": DEFAULT_SEED, "use_dixon_coles": True},
    },
    "small_logistic": {
        "factory": "SmallLogisticModel",
        "params": {"seed": DEFAULT_SEED, "C": 1.0, "calibrate": False},
    },
    "small_logistic_calibrated": {
        "factory": "SmallLogisticModel",
        "params": {"seed": DEFAULT_SEED, "C": 1.0, "calibrate": True},
    },
    "elo_logistic": {
        "factory": "EloLogisticModel",
        "params": {"seed": DEFAULT_SEED, "C": 1.0, "calibrate": True},
    },
    "elo_logistic_uncalibrated": {
        "factory": "EloLogisticModel",
        "params": {"seed": DEFAULT_SEED, "C": 1.0, "calibrate": False},
    },
    "fitted_poisson": {
        "factory": "FittedPoissonModel",
        "params": {"seed": DEFAULT_SEED, "alpha": 1.0, "use_recency": False},
    },
    "multinomial_logistic": {
        "factory": "MultinomialModel",
        "params": {"seed": DEFAULT_SEED, "C": 1.0, "calibrate": True, "class_weight": None},
    },
    "multinomial_logistic_uncalibrated": {
        "factory": "MultinomialModel",
        "params": {"seed": DEFAULT_SEED, "C": 1.0, "calibrate": False, "class_weight": None},
    },
    "restricted_hgb": {
        "factory": "RestrictedHGBModel",
        "params": {"seed": DEFAULT_SEED, "max_iter": 100, "max_leaf_nodes": 15,
                   "min_samples_leaf": 50, "l2_regularization": 1.0,
                   "max_depth": 3, "learning_rate": 0.05},
    },
}


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=Path(__file__).parent
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def get_code_hash() -> str:
    """Compute hash of all Python source files in src/ and tests/."""
    hasher = hashlib.sha256()
    for root in ["src", "tests"]:
        for path in sorted(Path(root).rglob("*.py")):
            hasher.update(path.read_bytes())
    return hasher.hexdigest()[:16]


def get_data_hash(master: pl.DataFrame) -> str:
    """Compute hash of the master dataset."""
    # Hash based on shape, columns, and a sample of data
    hasher = hashlib.sha256()
    hasher.update(str(master.shape).encode())
    hasher.update(",".join(sorted(master.columns)).encode())
    # Sample first 1000 rows for hashing
    sample = master.head(min(1000, master.height)).write_csv()
    hasher.update(sample.encode())
    return hasher.hexdigest()[:16]


@dataclass
class BenchmarkMetadata:
    """Metadata for benchmark reproducibility."""
    benchmark_version: str
    model_version: str
    git_commit: str
    code_hash: str
    data_hash: str
    timestamp: str
    seed: int
    leagues: List[str]
    seasons: List[str]
    config: Dict
    python_version: str
    numpy_version: str
    polars_version: str
    scipy_version: str
    sklearn_version: str


def create_model_factory(name: str, config: Dict):
    """Create a model factory function from config."""
    from prediction.models import (
        HistoricalBaseline, PoissonModel, MultinomialModel,
        SmallLogisticModel, EloLogisticModel, FittedPoissonModel,
        RestrictedHGBModel,
    )
    
    factory_map = {
        "HistoricalBaseline": HistoricalBaseline,
        "PoissonModel": PoissonModel,
        "MultinomialModel": MultinomialModel,
        "SmallLogisticModel": SmallLogisticModel,
        "EloLogisticModel": EloLogisticModel,
        "FittedPoissonModel": FittedPoissonModel,
        "RestrictedHGBModel": RestrictedHGBModel,
    }
    
    model_class = factory_map[config["factory"]]
    params = config["params"].copy()
    
    def factory():
        return model_class(**params)
    
    return factory


def run_benchmark(
    data_dir: Path,
    output_dir: Path,
    leagues: List[str],
    seasons: List[str],
    seed: int = DEFAULT_SEED,
    min_train_seasons: int = 3,
) -> Dict[str, Any]:
    """Run the full benchmark with fixed configuration."""
    
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ------------------------------------------------------------------ Load data
    print(f"\n[1/6] Loading data from {data_dir}...")
    master = build_master_dataset(data_dir, leagues, seasons, usecols=CORE_COLS)
    team_matches = build_team_match_dataset(master)
    print(f"  Master matches: {master.height}, Team-matches: {team_matches.height}")
    
    # ----------------------------------------------------------------- Features
    print("\n[2/6] Building strictly pre-match features...")
    features = build_pre_match_features(team_matches)
    paired = pair_match_features(features)
    print(f"  Paired match-feature rows: {paired.height}, Features: {paired.width}")
    
    # Compute hashes for reproducibility
    code_hash = get_code_hash()
    data_hash = get_data_hash(master)
    git_commit = get_git_commit()
    
    metadata = BenchmarkMetadata(
        benchmark_version=BENCHMARK_VERSION,
        model_version=MODEL_VERSION,
        git_commit=git_commit,
        code_hash=code_hash,
        data_hash=data_hash,
        timestamp=datetime.now().isoformat(),
        seed=seed,
        leagues=leagues,
        seasons=seasons,
        config=BENCHMARK_MODELS,
        python_version=sys.version.split()[0],
        numpy_version=np.__version__,
        polars_version=pl.__version__,
        scipy_version="unknown",
        sklearn_version="unknown",
    )
    
    try:
        import scipy
        metadata.scipy_version = scipy.__version__
    except:
        pass
    try:
        import sklearn
        metadata.sklearn_version = sklearn.__version__
    except:
        pass
    
    # ----------------------------------------------------------------- Validation
    print("\n[3/6] Running expanding-window chronological validation...")
    factories = {name: create_model_factory(name, cfg) for name, cfg in BENCHMARK_MODELS.items()}
    
    results = compare_models_validation(
        master, team_matches, factories, seasons=seasons, seed=seed,
        min_train_seasons=min_train_seasons
    )
    
    # ----------------------------------------------------------------- Collect results
    print("\n[4/6] Collecting and saving results...")
    
    # Per-fold metrics
    fold_rows = []
    for name, vr in results.items():
        for f in vr.folds:
            fold_rows.append({
                "model": name,
                "test_season": f.test_season,
                "train_seasons": "+".join(f.train_seasons),
                "n_train": f.n_train,
                "n_test": f.n_test,
                "brier": f.metrics.get("brier", float("nan")),
                "log_loss": f.metrics.get("log_loss", float("nan")),
                "accuracy": f.metrics.get("accuracy", float("nan")),
                "ece": f.metrics.get("ece", float("nan")),
                "home_mae": f.metrics.get("home_mae", float("nan")),
                "away_mae": f.metrics.get("away_mae", float("nan")),
            })
    
    # Model comparison table (primary: log_loss, secondary: brier)
    comp_rows = []
    for name, vr in results.items():
        comp_rows.append({
            "model": name,
            "log_loss": vr.aggregate.get("log_loss"),
            "brier": vr.aggregate.get("brier"),
            "accuracy": vr.aggregate.get("accuracy"),
            "ece": vr.aggregate.get("ece"),
            "n_folds": vr.n_folds,
            "total_test_matches": vr.aggregate.get("total_test_matches"),
        })
    
    # Save tables
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    fold_df = pl.DataFrame(fold_rows)
    fold_df.write_csv(output_dir / "tables" / "backtest_fold_metrics.csv")
    
    comp_df = pl.DataFrame(comp_rows).sort("log_loss")
    comp_df.write_csv(output_dir / "tables" / "model_comparison.csv")
    
    # Save keyed predictions for all models
    (output_dir / "predictions").mkdir(parents=True, exist_ok=True)
    all_keyed = {}
    for name, vr in results.items():
        if hasattr(vr, 'all_keyed_predictions') and vr.all_keyed_predictions:
            keyed_df = keyed_predictions_to_dataframe(vr.all_keyed_predictions)
            all_keyed[name] = keyed_df
            keyed_df.write_parquet(output_dir / "predictions" / f"keyed_{name}.parquet")
            keyed_df.write_csv(output_dir / "predictions" / f"keyed_{name}.csv")
            print(f"  Saved {name}: {keyed_df.height} keyed predictions")
    
    # ----------------------------------------------------------------- Calibration study
    print("\n[5/6] Running calibration comparison (uncalibrated vs calibrated)...")
    
    calibration_rows = []
    for name, vr in results.items():
        if not hasattr(vr, 'all_keyed_predictions') or not vr.all_keyed_predictions:
            continue
        keyed_df = keyed_predictions_to_dataframe(vr.all_keyed_predictions)
        if keyed_df.is_empty():
            continue
        
        probs = np.column_stack([
            keyed_df["p_home"].to_numpy(),
            keyed_df["p_draw"].to_numpy(),
            keyed_df["p_away"].to_numpy(),
        ])
        outcomes = keyed_df["actual_outcome"].to_numpy()
        
        # Overall metrics
        logloss = log_loss_multiclass(probs, outcomes)
        brier = brier_score_multiclass(probs, outcomes)
        ece_result = expected_calibration_error(probs, outcomes)
        ece = ece_result.get("ece", float("nan"))
        
        # Per-class reliability
        rdiag = []
        for class_idx, class_name in enumerate(["home", "draw", "away"]):
            class_data = __import__("prediction.calibration", fromlist=["ReliabilityDiagramData"]).ReliabilityDiagramData().per_class(probs, outcomes, class_idx)
            for bin_data in class_data:
                calibration_rows.append({
                    "model": name,
                    "class": class_name,
                    "bin": bin_data["bin"],
                    "lo": bin_data["lo"],
                    "hi": bin_data["hi"],
                    "count": bin_data["count"],
                    "mean_predicted": bin_data["mean_predicted"],
                    "observed_frequency": bin_data["observed_frequency"],
                })
        
        calibration_rows.append({
            "model": name,
            "log_loss": logloss,
            "brier": brier,
            "ece": ece,
            "n_samples": len(outcomes),
        })
    
    cal_df = pl.DataFrame(calibration_rows)
    cal_df.write_csv(output_dir / "tables" / "calibration_comparison.csv")
    
    # League breakdown
    league_rows = []
    for name, vr in results.items():
        if not hasattr(vr, 'all_keyed_predictions') or not vr.all_keyed_predictions:
            continue
        keyed_df = keyed_predictions_to_dataframe(vr.all_keyed_predictions)
        if keyed_df.is_empty():
            continue
        league_metrics = compute_league_metrics(keyed_df)
        for row in league_metrics.to_dicts():
            league_rows.append({"model": name, **row})
    
    if league_rows:
        league_df = pl.DataFrame(league_rows)
        league_df.write_csv(output_dir / "tables" / "league_breakdown.csv")
    
    # Season breakdown
    season_rows = []
    for name, vr in results.items():
        if not hasattr(vr, 'all_keyed_predictions') or not vr.all_keyed_predictions:
            continue
        keyed_df = keyed_predictions_to_dataframe(vr.all_keyed_predictions)
        if keyed_df.is_empty():
            continue
        # Add model name to keyed_df for segment computation
        keyed_df = keyed_df.with_columns(pl.lit(name).alias("model"))
        # Compute per season
        for season in keyed_df.select(pl.col("season").unique()).to_series().to_list():
            seg_df = keyed_df.filter(pl.col("season") == season)
            n = seg_df.height
            probs = np.column_stack([seg_df["p_home"].to_numpy(), seg_df["p_draw"].to_numpy(), seg_df["p_away"].to_numpy()])
            outcomes = seg_df["actual_outcome"].to_numpy()
            logloss = log_loss_multiclass(probs, outcomes)
            brier = brier_score_multiclass(probs, outcomes)
            accuracy = float((probs.argmax(axis=1) == outcomes).mean())
            season_rows.append({
                "model": name,
                "season": season,
                "n": n,
                "log_loss": logloss,
                "brier": brier,
                "accuracy": accuracy,
            })
    
    if season_rows:
        season_df = pl.DataFrame(season_rows)
        season_df.write_csv(output_dir / "tables" / "season_breakdown.csv")
    
    # ----------------------------------------------------------------- Save metadata
    print("\n[6/6] Saving metadata and manifest...")
    (output_dir / "metadata").mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "metadata" / "benchmark_metadata.json", "w") as f:
        json.dump(asdict(metadata), f, indent=2)
    
    # Manifest
    manifest = {
        "timestamp": metadata.timestamp,
        "seed": seed,
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "leagues": leagues,
        "seasons": seasons,
        "benchmark_version": BENCHMARK_VERSION,
        "model_version": MODEL_VERSION,
        "git_commit": git_commit,
        "code_hash": code_hash,
        "data_hash": data_hash,
        "n_matches": int(master.height),
        "n_team_matches": int(team_matches.height),
        "n_paired_features": int(paired.height),
        "models_compared": len(results),
        "best_model_by_log_loss": comp_df[0, "model"] if len(comp_df) > 0 else None,
        "best_log_loss": float(comp_df[0, "log_loss"]) if len(comp_df) > 0 else None,
    }
    
    with open(output_dir / "metadata" / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Models compared: {len(results)}")
    print(f"Best model (log_loss): {comp_df[0, 'model'] if len(comp_df) > 0 else 'N/A'}")
    print(f"Best log_loss: {comp_df[0, 'log_loss'] if len(comp_df) > 0 else 'N/A'}")
    print(f"\nModel ranking (by log_loss):")
    for i, row in enumerate(comp_df.to_dicts()):
        print(f"  {i+1}. {row['model']}: log_loss={row['log_loss']:.5f}, brier={row['brier']:.5f}, acc={row['accuracy']:.3f}")
    
    print(f"\nReproducibility:")
    print(f"  Git commit: {git_commit}")
    print(f"  Code hash: {code_hash}")
    print(f"  Data hash: {data_hash}")
    print(f"  Seed: {seed}")
    
    return {
        "metadata": metadata,
        "results": results,
        "comparison": comp_df,
        "calibration": cal_df,
        "league_breakdown": pl.DataFrame(league_rows) if league_rows else None,
        "season_breakdown": pl.DataFrame(season_rows) if season_rows else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Football Analytics Benchmark - Reproducible Model Evaluation")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Root directory containing season subdirectories with data.zip")
    parser.add_argument("--output-dir", type=str, default="output/benchmark",
                        help="Output directory for benchmark artifacts")
    parser.add_argument("--leagues", type=str, nargs="+", default=BENCHMARK_LEAGUES)
    parser.add_argument("--seasons", type=str, nargs="+", default=BENCHMARK_SEASONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--min-train-seasons", type=int, default=3)
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    
    try:
        run_benchmark(
            data_dir=data_dir,
            output_dir=output_dir,
            leagues=args.leagues,
            seasons=args.seasons,
            seed=args.seed,
            min_train_seasons=args.min_train_seasons,
        )
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()