#!/usr/bin/env python3
"""
Prediction pipeline CLI for the Football Analytics Study.

Runs the predictive intelligence engine:
  1. Builds strictly pre-match features from historical data.
  2. Fits candidate models (baseline, Poisson, multinomial, optional HGB).
  3. Runs expanding-window chronological validation.
  4. Produces calibrated probabilities, scoreline distributions, uncertainty.
  5. Generates scientific figures P1-P7 and machine-readable tables.
  6. Persists model bundles and metadata for reproducible inference.

Usage:
    python run_predictions.py --data-dir . --output-dir output/predictions --seed 42

A separate inference API (src/prediction/inference.py) supports forecasting
supplied upcoming fixtures via predict_fixture() or predict_from_csv().
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.load_data import (
    build_master_dataset, build_team_match_dataset, CORE_COLS,
)
from prediction.utils import set_seed, DEFAULT_SEED
from prediction.features import build_pre_match_features, pair_match_features
from prediction.models import (
    HistoricalBaseline, PoissonModel, MultinomialModel,
    HistGradientBoostingModel,
    SmallLogisticModel, EloLogisticModel, FittedPoissonModel,
    RestrictedHGBModel, DrawOptimizedModel, DrawBoostedModel,
)
from prediction.validation_v2 import (
    run_chronological_validation, compare_models_validation,
    keyed_predictions_to_dataframe, compute_league_metrics,
)
from prediction.evaluation import outcome_metrics, goal_metrics
from prediction.calibration import brier_score_multiclass, expected_calibration_error
from prediction.distributions import (
    scoreline_probabilities, derived_market_probs, poisson_1x2_from_lambdas,
)
from prediction.reporting import (
    figure_p1_performance, figure_p2_calibration, figure_p3_goal_quality,
    figure_p4_scoreline, figure_p5_league, figure_p6_feature_importance,
    figure_p7_baseline_vs_advanced,
)
from prediction.inference import PredictionBundle

SEASONS_DEFAULT = ["1920", "2021", "2122", "2223", "2324", "2425", "2526", "2627"]
LEAGUES_DEFAULT = ["E0", "E1", "E2", "E3", "SP1", "D1", "I1", "F1"]

MODEL_VERSION = "1.0.0"


def parse_args():
    p = argparse.ArgumentParser(description="Football predictive intelligence engine")
    p.add_argument("--data-dir", type=str, default=".",
                   help="Root directory containing season subdirectories with data.zip")
    p.add_argument("--output-dir", type=str, default="output/predictions",
                   help="Output directory for prediction artifacts")
    p.add_argument("--leagues", type=str, nargs="+", default=LEAGUES_DEFAULT)
    p.add_argument("--seasons", type=str, nargs="+", default=SEASONS_DEFAULT)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="Random seed for reproducibility")
    p.add_argument("--skip-hgb", action="store_true",
                   help="Skip the HistGradientBoosting challenger")
    p.add_argument("--reuse-cache", action="store_true",
                   help="Reuse cached EDA Parquet datasets if fingerprint matches")
    return p.parse_args()


def setup_output_dirs(output_dir: Path):
    dirs = {
        "root": output_dir,
        "data": output_dir / "data",
        "tables": output_dir / "tables",
        "figures": output_dir / "figures",
        "models": output_dir / "models",
        "metadata": output_dir / "metadata",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _jsonify(obj):
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, float, str)) or obj is None:
        return obj
    return str(obj)


def main():
    args = parse_args()
    set_seed(args.seed)

    DATA_DIR = Path(args.data_dir)
    OUTPUT_DIRS = setup_output_dirs(Path(args.output_dir))
    TARGET_LEAGUES = args.leagues
    SEASONS = args.seasons

    print("=" * 60)
    print("FOOTBALL PREDICTIVE INTELLIGENCE ENGINE")
    print("=" * 60)
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIRS['root']}")
    print(f"Leagues: {TARGET_LEAGUES}")
    print(f"Seasons: {SEASONS}")
    print(f"Random seed: {args.seed}")
    print("No bookmaker odds. No look-ahead leakage. Chronological validation only.")

    # ------------------------------------------------------------------ load
    print("\n[1/7] Ingesting data...")
    t0 = time.perf_counter()
    master = build_master_dataset(DATA_DIR, TARGET_LEAGUES, SEASONS, usecols=CORE_COLS)
    team_matches = build_team_match_dataset(master)
    print(f"  Master matches: {master.height}, team-matches: {team_matches.height}")
    stage_load = time.perf_counter() - t0

    # ----------------------------------------------------------------- features
    print("\n[2/7] Building strictly pre-match features...")
    t0 = time.perf_counter()
    features = build_pre_match_features(team_matches)
    paired = pair_match_features(features)
    print(f"  Paired match-feature rows: {paired.height}")
    print(f"  Feature columns: {paired.width}")
    stage_feat = time.perf_counter() - t0

    # Save historical features.
    features.write_parquet(OUTPUT_DIRS["data"] / "historical_features.parquet")
    paired.write_parquet(OUTPUT_DIRS["data"] / "paired_features.parquet")

    # ----------------------------------------------------------------- validation
    print("\n[3/7] Running expanding-window chronological validation...")
    t0 = time.perf_counter()

    def factory_baseline():
        return HistoricalBaseline(seed=args.seed)

    def factory_poisson():
        return PoissonModel(seed=args.seed, use_dixon_coles=False)

    def factory_multinomial():
        return MultinomialModel(seed=args.seed, C=None, calibrate=True, class_weight="balanced")

    def factory_small_logistic():
        return SmallLogisticModel(seed=args.seed, C=None, calibrate=False)

    def factory_elo_logistic():
        return EloLogisticModel(seed=args.seed, C=None, calibrate=True)

    def factory_fitted_poisson():
        return FittedPoissonModel(seed=args.seed, alpha=1.0, use_recency=False)

    def factory_draw_optimized():
        return DrawOptimizedModel(seed=args.seed, C=None, calibrate=True, class_weight="balanced")

    factories = {
        "historical_baseline": factory_baseline,
        "poisson": factory_poisson,
        "multinomial_logistic": factory_multinomial,
        "small_logistic": factory_small_logistic,
        "elo_logistic": factory_elo_logistic,
        "fitted_poisson": factory_fitted_poisson,
        "draw_optimized": factory_draw_optimized,
    }
    if not args.skip_hgb:
        def factory_hgb():
            return HistGradientBoostingModel(seed=args.seed)
        factories["hist_gradient_boosting"] = factory_hgb
    
    # Add restricted HGB as separate challenger
    def factory_restricted_hgb():
        return RestrictedHGBModel(seed=args.seed, max_iter=100, max_leaf_nodes=15,
                                   min_samples_leaf=50, l2_regularization=1.0,
                                   max_depth=3, learning_rate=0.05)
    factories["restricted_hgb"] = factory_restricted_hgb
    
    # Draw-boosted restricted HGB
    def factory_draw_boosted_hgb():
        base = RestrictedHGBModel(seed=args.seed, max_iter=100, max_leaf_nodes=15,
                                   min_samples_leaf=50, l2_regularization=1.0,
                                   max_depth=3, learning_rate=0.05)
        return DrawBoostedModel(base, draw_threshold=0.2, draw_boost=0.05)
    factories["draw_boosted_hgb"] = factory_draw_boosted_hgb

    results = compare_models_validation(
        master, team_matches, factories, seasons=SEASONS, seed=args.seed
    )

    # Collect per-fold metrics for reporting.
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
                "home_mae": f.metrics.get("home_mae", float("nan")),
                "away_mae": f.metrics.get("away_mae", float("nan")),
            })
    stage_val = time.perf_counter() - t0
    print(f"  Validation folds: {sum(vr.n_folds for vr in results.values())}")
    for name, vr in results.items():
        print(f"  {name}: Brier={vr.aggregate.get('brier')} "
              f"LogLoss={vr.aggregate.get('log_loss')} "
              f"Acc={vr.aggregate.get('accuracy')} n={vr.aggregate.get('total_test_matches')}")

    # ----------------------------------------------------------------- backtest predictions
    print("\n[4/7] Saving backtest predictions and tables...")
    t0 = time.perf_counter()
    fold_df = pl.DataFrame(fold_rows)
    fold_df.write_csv(OUTPUT_DIRS["tables"] / "backtest_fold_metrics.csv")
    # Aggregate model comparison table.
    comp_rows = []
    for name, vr in results.items():
        comp_rows.append({
            "model": name,
            "brier": vr.aggregate.get("brier"),
            "log_loss": vr.aggregate.get("log_loss"),
            "accuracy": vr.aggregate.get("accuracy"),
            "n_folds": vr.n_folds,
            "total_test_matches": vr.aggregate.get("total_test_matches"),
        })
    comp_df = pl.DataFrame(comp_rows)
    comp_df.write_csv(OUTPUT_DIRS["tables"] / "model_comparison.csv")
    
    # Save keyed predictions for auditability (best model)
    best_name = min(results, key=lambda n: results[n].aggregate.get("brier", float("inf")))
    best = results[best_name]
    if hasattr(best, 'all_keyed_predictions') and best.all_keyed_predictions:
        keyed_df = keyed_predictions_to_dataframe(best.all_keyed_predictions)
        keyed_df.write_parquet(OUTPUT_DIRS["tables"] / "keyed_predictions.parquet")
        keyed_df.write_csv(OUTPUT_DIRS["tables"] / "keyed_predictions.csv")
        print(f"  Saved keyed predictions: {keyed_df.height} rows")
    
    print(f"  Models compared: {len(comp_rows)}")
    stage_save = time.perf_counter() - t0

    # ----------------------------------------------------------------- figures
    print("\n[5/7] Generating scientific figures P1-P7...")
    t0 = time.perf_counter()
    figure_p1_performance(fold_rows, OUTPUT_DIRS["figures"])

    # P2 calibration: best model's pooled out-of-fold predictions.
    best_name = min(results, key=lambda n: results[n].aggregate.get("brier", float("inf")))
    best = results[best_name]
    all_probs = []; all_outcomes = []
    for f in best.folds:
        if f.predictions is not None and f.outcomes is not None:
            all_probs.append(f.predictions); all_outcomes.append(f.outcomes)
    if all_probs:
        P = np.vstack(all_probs); O = np.concatenate(all_outcomes)
        probs_by_class = {i: P for i in range(3)}
        outcomes_by_class = {i: O for i in range(3)}
        figure_p2_calibration(probs_by_class, outcomes_by_class, OUTPUT_DIRS["figures"])

    # P3 goal quality + P4 scoreline: use pooled out-of-fold lambdas from the
    # first model that produces them (baseline or Poisson).
    lambda_model = None
    for nm in ("poisson", "historical_baseline"):
        if nm in results and results[nm].pooled.get("home_lambda") is not None:
            lambda_model = nm; break
    if lambda_model is not None:
        pl_ = results[lambda_model].pooled
        hl = pl_["home_lambda"]; al = pl_["away_lambda"]
        hg = pl_["home_goals"]; ag = pl_["away_goals"]
        figure_p3_goal_quality(hl, al, hg, ag, OUTPUT_DIRS["figures"])
        # P4: predicted scoreline (from lambdas) vs observed scoreline.
        from prediction.distributions import scoreline_probabilities
        pred_scores = {}
        for i in range(hl.shape[0]):
            sp = scoreline_probabilities(hl[i], al[i])
            for s, p in sp.items():
                pred_scores[s] = pred_scores.get(s, 0.0) + p
        # normalize predicted to sum to 1 across matches (mean distribution)
        z = sum(pred_scores.values())
        if z > 0:
            pred_scores = {s: v / hl.shape[0] for s, v in pred_scores.items()}
        obs_scores = {}
        for i in range(hg.shape[0]):
            s = (int(hg[i]), int(ag[i]))
            obs_scores[s] = obs_scores.get(s, 0.0) + 1.0
        zo = sum(obs_scores.values())
        obs_scores = {s: v / zo for s, v in obs_scores.items()}
        figure_p4_scoreline(pred_scores, obs_scores, OUTPUT_DIRS["figures"])

    # P5 league-specific performance: out-of-sample Brier per league from keyed predictions.
    league_rows = []
    if hasattr(best, 'all_keyed_predictions') and best.all_keyed_predictions:
        keyed_df = keyed_predictions_to_dataframe(best.all_keyed_predictions)
        league_metrics = compute_league_metrics(keyed_df)
        for row in league_metrics.to_dicts():
            league_rows.append({"league": row["league"], "n": row["n"], "brier": row["brier"]})
    else:
        # Fallback: compute from fold predictions
        for d in TARGET_LEAGUES:
            n = int(master.filter(pl.col("Div") == d).filter(
                pl.col("Season").is_in(SEASONS[2:])).height)
            league_rows.append({"league": d, "n": n,
                                "brier": best.aggregate.get("brier")})
    figure_p5_league(league_rows, OUTPUT_DIRS["figures"])

    # P6 feature importance: standardized coefficients of the selected model
    # (interpretable model -> coefficients; reported on held-out-consistent features).
    bm = None
    if best_name in ("multinomial_logistic", "historical_baseline", "poisson"):
        bm = results[best_name].folds[0] if results[best_name].folds else None
    importance = []
    # Refit the best model on all training data to read coefficients.
    bmf = factories[best_name]()
    if isinstance(bmf, HistoricalBaseline):
        bmf.fit(team_matches, master)
    elif isinstance(bmf, PoissonModel):
        bmf.fit(team_matches, master)
    else:
        bmf.fit(team_matches, paired)
    coefs = getattr(bmf.model, "coef_", None)
    if coefs is not None and hasattr(bmf, "feature_names"):
        for name, c in zip(bmf.feature_names, coefs[0] if coefs.ndim > 1 else coefs):
            importance.append({"feature": name, "importance": float(abs(c))})
    figure_p6_feature_importance(importance, OUTPUT_DIRS["figures"])

    # P7 baseline vs advanced model with per-fold standard-error intervals.
    import math
    def per_fold_brier(model):
        vals = [r["brier"] for r in fold_rows if r["model"] == model and r["brier"] is not None
                and not (isinstance(r["brier"], float) and math.isnan(r["brier"]))]
        return vals
    base_b = per_fold_brier("historical_baseline")
    adv_b = per_fold_brier(best_name)
    ci_base = ci_adv = None
    if len(base_b) > 1:
        se = np.std(base_b, ddof=1) / math.sqrt(len(base_b))
        ci_base = (float(np.mean(base_b) - se), float(np.mean(base_b) + se))
    if len(adv_b) > 1:
        se = np.std(adv_b, ddof=1) / math.sqrt(len(adv_b))
        ci_adv = (float(np.mean(adv_b) - se), float(np.mean(adv_b) + se))
    figure_p7_baseline_vs_advanced(
        {"brier": float(np.mean(base_b)) if base_b else float("nan")},
        {"brier": float(np.mean(adv_b)) if adv_b else float("nan")},
        ci_base, ci_adv, OUTPUT_DIRS["figures"],
    )
    stage_fig = time.perf_counter() - t0
    print(f"  Figures generated in {OUTPUT_DIRS['figures']}")

    # ----------------------------------------------------------------- persist best model
    print("\n[6/7] Persisting model bundle and metadata...")
    t0 = time.perf_counter()
    # Fit the best model on ALL history for future inference.
    best_model_fn = factories[best_name]
    bm = best_model_fn()
    # Fit each model type on the data it requires.
    if isinstance(bm, HistoricalBaseline):
        bm.fit(team_matches, master)
    elif isinstance(bm, PoissonModel):
        bm.fit(team_matches, master)
    else:
        # Multinomial / HGB need the engineered paired features.
        bm.fit(team_matches, paired)

    # Determine actual calibration method used
    calibration_method = "none"
    if best_name == "multinomial_logistic" and results[best_name].folds:
        fold0 = results[best_name].folds[0]
        if hasattr(fold0, 'keyed_predictions') and fold0.keyed_predictions:
            # Check if model has calibrator
            bmf = factories[best_name]()
            if isinstance(bmf, MultinomialModel):
                calibration_method = f"{bmf.calibration_method} (OOF)"
    
    meta = {
        "model_version": MODEL_VERSION,
        "model_name": best_name,
        "training_cutoff": str(master.select(pl.col("Date").max()).to_series()[0]),
        "training_seasons": SEASONS,
        "training_leagues": TARGET_LEAGUES,
        "n_training_matches": int(master.height),
        "feature_names": getattr(bm, "feature_names", []),
        "random_seed": args.seed,
        "software_versions": {
            "polars": pl.__version__,
            "numpy": np.__version__,
        },
        "validation_results": {k: v.aggregate for k, v in results.items()},
        "selected_model": best_name,
        "calibration_method": calibration_method,
        "data_quality": {
            "missing_goals": int(master.select(pl.col("FTHG").null_count() + pl.col("FTAG").null_count()).to_series()[0]),
        },
    }
    bundle = PredictionBundle(bm, best_name, meta, getattr(bm, "feature_names", []),
                              cutoff_date=meta["training_cutoff"],
                              cutoff_season=SEASONS[-1])
    bundle.save(OUTPUT_DIRS["models"] / "model_bundle.joblib")
    with open(OUTPUT_DIRS["metadata"] / "prediction_manifest.json", "w") as f:
        json.dump(_jsonify(meta), f, indent=2)
    with open(OUTPUT_DIRS["metadata"] / "feature_dictionary.json", "w") as f:
        json.dump(_jsonify({"features": getattr(bm, "feature_names", [])}), f, indent=2)
    print(f"  Saved model bundle: {best_name}")
    print(f"  Training cutoff: {meta['training_cutoff']}")
    stage_persist = time.perf_counter() - t0

    # ----------------------------------------------------------------- manifest
    print("\n[7/7] Writing manifest...")
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "seed": args.seed,
        "data_dir": str(DATA_DIR),
        "output_dir": str(OUTPUT_DIRS["root"]),
        "leagues": TARGET_LEAGUES,
        "seasons": SEASONS,
        "model_version": MODEL_VERSION,
        "n_matches": int(master.height),
        "n_team_matches": int(team_matches.height),
        "n_paired_features": int(paired.height),
        "software_versions": {
            "polars": pl.__version__, "duckdb": None, "numpy": np.__version__,
        },
        "stage_times_seconds": {
            "data_ingestion": round(stage_load, 4),
            "feature_engineering": round(stage_feat, 4),
            "validation": round(stage_val, 4),
            "save_tables": round(stage_save, 4),
            "figures": round(stage_fig, 4),
            "persist": round(stage_persist, 4),
        },
        "total_runtime_seconds": round(
            stage_load + stage_feat + stage_val + stage_save + stage_fig + stage_persist, 4),
    }
    with open(OUTPUT_DIRS["metadata"] / "manifest.json", "w") as f:
        json.dump(_jsonify(manifest), f, indent=2)

    print("\n" + "=" * 60)
    print("PREDICTION ENGINE COMPLETE")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIRS['root']}")
    print(f"  data/:   historical_features.parquet, paired_features.parquet")
    print(f"  tables:  model_comparison.csv, backtest_fold_metrics.csv")
    print(f"  figures: P1-P7 PNGs + companion CSVs")
    print(f"  models:  model_bundle.joblib + prediction_manifest.json")
    print(f"  metadata: prediction_manifest.json, feature_dictionary.json, manifest.json")
    print(f"\nBest model (by out-of-sample Brier): {best_name}")
    print("\nNext: use src/prediction/inference.py predict_fixture() for upcoming fixtures.")


if __name__ == "__main__":
    main()
