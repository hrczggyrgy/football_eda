#!/usr/bin/env python3
"""
Validate prediction accuracy against the 2627 season results from
https://football-data.co.uk/mmz4281/2627/Latest_Results.csv

Uses the trained model bundle (output/predictions/models/model_bundle.joblib)
to predict each fixture in the CSV, then compares predicted outcome
probabilities and scorelines to actual results.
"""

from __future__ import annotations

import io
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent / "src"))

from prediction.inference import PredictionBundle, predict_fixture, _season_from_date
from data.load_data import build_master_dataset, build_team_match_dataset, CORE_COLS
from prediction.evaluation import outcome_metrics
from prediction.calibration import brier_score_multiclass

BUNDLE_PATH = Path("output/predictions/models/model_bundle.joblib")
HISTORY_SEASONS = ["1920", "2021", "2122", "2223", "2324", "2425", "2526", "2627"]
HISTORY_LEAGUES = ["E0", "E1", "E2", "E3", "SP1", "D1", "I1", "F1"]

DEFAULT_RESULTS_CSV = Path("latest_results_2627.csv")


def main(results_csv: Optional[Path] = None):
    # ---- Load trained model bundle ----
    if not BUNDLE_PATH.exists():
        print(f"ERROR: Model bundle not found at {BUNDLE_PATH}")
        print("Run: python run_predictions.py --data-dir . --output-dir output/predictions --seed 42")
        sys.exit(1)

    bundle = PredictionBundle.load(BUNDLE_PATH)
    print(f"Loaded model bundle: {bundle.model_name} (cutoff {bundle.cutoff_date})")

    bundle_cutoff = None
    if bundle.cutoff_date:
        try:
            bundle_cutoff = datetime.strptime(str(bundle.cutoff_date)[:10], "%Y-%m-%d").date()
        except ValueError:
            bundle_cutoff = None

    # ---- Load historical training data ----
    DATA_DIR = Path(".")
    master = build_master_dataset(DATA_DIR, HISTORY_LEAGUES, HISTORY_SEASONS, usecols=CORE_COLS)
    team_matches = build_team_match_dataset(master)
    print(f"History: {master.height} matches across {len(HISTORY_SEASONS)} seasons")

    # ---- Load 2627 results CSV ----
    csv_path = Path(results_csv) if results_csv is not None else DEFAULT_RESULTS_CSV
    if not csv_path.exists():
        print(f"ERROR: Results CSV not found at {csv_path}")
        print("Pass the 2627 results CSV path as an argument, e.g.:")
        print("  python validate_2627.py /path/to/latest_results_2627.csv")
        sys.exit(1)

    # utf-8-sig strips a leading BOM (common in football-data.co.uk exports)
    # so the first header column is "Div" rather than "\ufeffDiv".
    csv_text = csv_path.read_text(encoding="utf-8-sig")
    lines = csv_text.strip().split("\n")
    header = lines[0].split(",")
    rows = [l.split(",") for l in lines[1:] if l.strip()]

    # Map columns we need
    col_idx = {name: i for i, name in enumerate(header)}
    needed = ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    missing = [c for c in needed if c not in col_idx]
    if missing:
        print(f"WARNING: CSV missing columns: {missing}")

    print(f"\nLoaded {len(rows)} fixtures from {csv_path}")

    # ---- Predict each fixture and compare ----
    results = []
    n_rejected_cutoff = 0
    for row in rows:
        div = row[col_idx["Div"]]
        date_str = row[col_idx["Date"]]
        home = row[col_idx["HomeTeam"]]
        away = row[col_idx["AwayTeam"]]

        # Parse fixture date and resolve its actual season.
        try:
            pred_date = datetime.strptime(date_str.strip(), "%d/%m/%y").date()
        except ValueError:
            try:
                pred_date = datetime.strptime(date_str.strip(), "%d/%m/%Y").date()
            except ValueError:
                pred_date = date.fromisoformat(date_str.strip()[:10])
        season = _season_from_date(pred_date)

        # F05: never evaluate a fixture at or before the bundle's training
        # cutoff -- it is not out-of-sample for a future-trained bundle.
        if bundle_cutoff is not None and pred_date <= bundle_cutoff:
            n_rejected_cutoff += 1
            continue

        # Actual result
        fthg = row[col_idx["FTHG"]] if "FTHG" in col_idx else ""
        ftag = row[col_idx["FTAG"]] if "FTAG" in col_idx else ""
        ftr = row[col_idx["FTR"]] if "FTR" in col_idx else ""
        actual_home_goals = int(fthg) if str(fthg).strip() else None
        actual_away_goals = int(ftag) if str(ftag).strip() else None
        actual_outcome = str(ftr).strip() if str(ftr).strip() else None

        # Predict using only history before this fixture date.
        try:
            pred = predict_fixture(
                division=div,
                date=date_str.strip(),
                home_team=home,
                away_team=away,
                bundle=bundle,
                history_matches=master,
                history_team_matches=team_matches,
                cutoff_date=bundle.cutoff_date,
                season=season,
            )
        except Exception as e:
            results.append({
                "div": div, "date": date_str, "home": home, "away": away,
                "actual_outcome": actual_outcome,
                "actual_home_goals": actual_home_goals,
                "actual_away_goals": actual_away_goals,
                "error": str(e),
                "pred_outcome": None,
                "pred_confidence": None,
                "correct": None,
                "home_win_prob": None, "draw_prob": None, "away_win_prob": None,
                "pred_home_goals": None, "pred_away_goals": None,
                "correct_scoreline": None,
                "data_quality": None,
            })
            continue

        # Predicted outcome = argmax
        probs = [pred.home_win_probability, pred.draw_probability, pred.away_win_probability]
        pred_idx = int(np.argmax(probs))
        pred_outcome = ["H", "D", "A"][pred_idx]
        pred_confidence = float(probs[pred_idx])

        correct = (pred_outcome == actual_outcome) if actual_outcome else None
        correct_scoreline = (
            (actual_home_goals == round(pred.home_expected_goals or -1)
             and actual_away_goals == round(pred.away_expected_goals or -2))
            if actual_home_goals is not None else None
        )

        results.append({
            "div": div, "date": date_str, "home": home, "away": away,
            "actual_outcome": actual_outcome,
            "actual_home_goals": actual_home_goals,
            "actual_away_goals": actual_away_goals,
            "error": None,
            "pred_outcome": pred_outcome,
            "pred_confidence": pred_confidence,
            "correct": correct,
            "home_win_prob": pred.home_win_probability,
            "draw_prob": pred.draw_probability,
            "away_win_prob": pred.away_win_probability,
            "pred_home_goals": pred.home_expected_goals,
            "pred_away_goals": pred.away_expected_goals,
            "correct_scoreline": correct_scoreline,
            "data_quality": pred.data_quality_status,
        })

    # ---- Aggregate metrics ----
    valid = [r for r in results if r["error"] is None and r["actual_outcome"] is not None]
    n_valid = len(valid)
    n_correct = sum(1 for r in valid if r["correct"])
    accuracy = n_correct / n_valid if n_valid else 0.0

    # Brier score (multiclass) on the SAME scale as the main evaluation:
    # mean sum over classes of (p - y)^2, not divided by n*3.
    probs_arr = np.asarray([
        [r["home_win_prob"], r["draw_prob"], r["away_win_prob"]]
        for r in valid
    ], dtype=np.float64)
    outcomes_arr = np.asarray([
        ["H", "D", "A"].index(r["actual_outcome"])
        for r in valid
    ], dtype=np.int64)
    brier = brier_score_multiclass(probs_arr, outcomes_arr) if n_valid else float("nan")

    # Log loss (clipped)
    eps = 1e-15
    logloss_sum = 0.0
    for r in valid:
        y_true = ["H", "D", "A"].index(r["actual_outcome"])
        p = [r["home_win_prob"], r["draw_prob"], r["away_win_prob"]]
        p_clipped = [min(max(pi, eps), 1 - eps) for pi in p]
        logloss_sum -= np.log(p_clipped[y_true])
    logloss = logloss_sum / n_valid if n_valid else float("nan")

    # Per-class accuracy
    per_class = {}
    for cls in ["H", "D", "A"]:
        cls_rows = [r for r in valid if r["actual_outcome"] == cls]
        if cls_rows:
            cls_correct = sum(1 for r in cls_rows if r["correct"])
            per_class[cls] = {"n": len(cls_rows), "correct": cls_correct,
                              "accuracy": cls_correct / len(cls_rows)}

    # Scoreline accuracy (exact)
    scoreline_rows = [r for r in valid if r["correct_scoreline"] is not None]
    n_scoreline_correct = sum(1 for r in scoreline_rows if r["correct_scoreline"])
    scoreline_acc = n_scoreline_correct / len(scoreline_rows) if scoreline_rows else None

    # Print report
    print("=" * 70)
    print("PREDICTION VALIDATION vs 2627 RESULTS (football-data.co.uk)")
    print("=" * 70)
    print(f"Fixtures in CSV:     {len(results)}")
    print(f"Valid predictions:   {n_valid}")
    print(f"Errors / unknown:    {len(results) - n_valid}")
    print()
    print(f"Outcome accuracy:    {accuracy:.4f}  ({n_correct}/{n_valid})")
    print(f"Brier score:         {brier:.4f}")
    print(f"Log loss:            {logloss:.4f}")
    print(f"Exact scoreline acc: {scoreline_acc:.4f}  ({n_scoreline_correct}/{len(scoreline_rows)})" if scoreline_acc is not None else "Exact scoreline acc: N/A")
    print()
    print("Per-class accuracy:")
    for cls in ["H", "D", "A"]:
        c = per_class.get(cls)
        if c:
            print(f"  {cls}: {c['accuracy']:.4f}  ({c['correct']}/{c['n']})")
    print()

    # Confusion matrix
    print("Confusion matrix (rows=actual, cols=pred):")
    labels = ["H", "D", "A"]
    cm = np.zeros((3, 3), dtype=int)
    for r in valid:
        cm[labels.index(r["actual_outcome"]), labels.index(r["pred_outcome"])] += 1
    print("      pred H   pred D   pred A")
    for i, a in enumerate(labels):
        print(f"act {a}: {cm[i][0]:6d}  {cm[i][1]:6d}  {cm[i][2]:6d}")
    print()

    # Save detailed results
    out_dir = Path("output/predictions/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(results)
    df.write_csv(out_dir / "validation_2627_detailed.csv")
    print(f"Detailed results saved to {out_dir / 'validation_2627_detailed.csv'}")

    # Summary JSON
    summary = {
        "n_fixtures": len(results),
        "n_valid": n_valid,
        "accuracy": round(accuracy, 4),
        "brier": round(brier, 4),
        "log_loss": round(logloss, 4),
        "scoreline_accuracy": round(scoreline_acc, 4) if scoreline_acc else None,
        "per_class": per_class,
        "model": bundle.model_name,
        "cutoff": bundle.cutoff_date,
    }
    with open(out_dir / "validation_2627_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {out_dir / 'validation_2627_summary.json'}")


if __name__ == "__main__":
    csv_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    main(results_csv=csv_arg)
