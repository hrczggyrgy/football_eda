"""
Prediction visualization: publication-quality figures P1-P7.

All figures are generated with Matplotlib; Polars frames are converted to
pandas only at the plotting boundary. Each figure has a precise research
question, states population/sample size, includes uncertainty, and is
accompanied by a machine-readable CSV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prediction.utils import OUTCOME_LABELS, _jsonify


def _save_csv(rows: List[dict], path: Path) -> None:
    """Write a list of dicts to CSV (machine-readable companion)."""
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _style_ax(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def figure_p1_performance(fold_results: List[dict], output_dir: Path) -> Tuple[Path, Path]:
    """P1 — Predictive performance across chronological testing seasons.

    Research question: How does each model's out-of-sample Brier score / log loss
    evolve across successive test seasons?
    """
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    models = sorted({r["model"] for r in fold_results})
    seasons = sorted({r["test_season"] for r in fold_results})

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for mi, model in enumerate(models):
        briers = []
        losses = []
        for s in seasons:
            row = next((r for r in fold_results if r["model"] == model and r["test_season"] == s), None)
            briers.append(row["brier"] if row else np.nan)
            losses.append(row["log_loss"] if row else np.nan)
        axes[0].plot(seasons, briers, marker="o", label=model)
        axes[1].plot(seasons, losses, marker="o", label=model)

    _style_ax(axes[0], "P1a — Out-of-sample Multiclass Brier Score by Test Season",
              "Test season", "Brier score (lower is better)")
    axes[0].legend()
    _style_ax(axes[1], "P1b — Out-of-sample Log Loss by Test Season",
              "Test season", "Log loss (lower is better)")
    axes[1].legend()
    fig.tight_layout()

    png = output_dir / "prediction_performance.png"
    csv_ = output_dir / "prediction_performance.csv"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _save_csv([{"model": r["model"], "test_season": r["test_season"],
                "brier": r["brier"], "log_loss": r["log_loss"],
                "accuracy": r["accuracy"], "n_test": r["n_test"]}
               for r in fold_results], csv_)
    return png, csv_


def figure_p2_calibration(probs_by_class: Dict[int, np.ndarray],
                          outcomes_by_class: Dict[int, np.ndarray],
                          output_dir: Path) -> Tuple[Path, Path]:
    """P2 — Probability calibration: reliability diagrams for H, D, A.

    Research question: Do predicted probabilities match observed frequencies?
    """
    from prediction.calibration import ReliabilityDiagramData
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    class_names = {0: "Home win", 1: "Draw", 2: "Away win"}
    rd = ReliabilityDiagramData(n_bins=10)
    rows = []
    for ci, ax in enumerate(axes):
        bins = rd.per_class(probs_by_class[ci], outcomes_by_class[ci], ci)
        xs, ys, counts = [], [], []
        for b in bins:
            if b["count"] > 0 and b["mean_predicted"] is not None:
                xs.append(b["mean_predicted"]); ys.append(b["observed_frequency"])
                counts.append(b["count"])
                rows.append({"class": class_names[ci], "bin": b["bin"],
                             "lo": b["lo"], "hi": b["hi"], "count": b["count"],
                             "mean_predicted": b["mean_predicted"],
                             "observed_frequency": b["observed_frequency"]})
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        if xs:
            ax.plot(xs, ys, "o-", label=class_names[ci])
        _style_ax(ax, f"P2 — {class_names[ci]} reliability diagram",
                  "Mean predicted probability", "Observed frequency")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend()
    fig.tight_layout()
    png = output_dir / "probability_calibration.png"
    csv_ = output_dir / "calibration_metrics.csv"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _save_csv(rows, csv_)
    return png, csv_


def figure_p3_goal_quality(home_lambda: np.ndarray, away_lambda: np.ndarray,
                           home_goals: np.ndarray, away_goals: np.ndarray,
                           output_dir: Path) -> Tuple[Path, Path]:
    """P3 — Goal prediction quality: actual vs predicted goal distributions."""
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    maxg = 7
    home_pred = np.bincount(np.clip(np.round(home_lambda).astype(int), 0, maxg),
                            minlength=maxg + 1) / max(1, len(home_lambda))
    home_act = np.bincount(np.clip(home_goals.astype(int), 0, maxg),
                           minlength=maxg + 1) / max(1, len(home_goals))
    away_pred = np.bincount(np.clip(np.round(away_lambda).astype(int), 0, maxg),
                            minlength=maxg + 1) / max(1, len(away_lambda))
    away_act = np.bincount(np.clip(away_goals.astype(int), 0, maxg),
                           minlength=maxg + 1) / max(1, len(away_goals))
    x = np.arange(maxg + 1)
    width = 0.35
    axes[0].bar(x - width / 2, home_act, width, label="Actual", alpha=0.8)
    axes[0].bar(x + width / 2, home_pred, width, label="Predicted (mean lambda)", alpha=0.8)
    _style_ax(axes[0], "P3a — Home goals: actual vs predicted distribution",
              "Home goals", "Share of matches")
    axes[0].legend(); axes[0].set_xticks(x)
    axes[1].bar(x - width / 2, away_act, width, label="Actual", alpha=0.8)
    axes[1].bar(x + width / 2, away_pred, width, label="Predicted (mean lambda)", alpha=0.8)
    _style_ax(axes[1], "P3b — Away goals: actual vs predicted distribution",
              "Away goals", "Share of matches")
    axes[1].legend(); axes[1].set_xticks(x)
    fig.tight_layout()
    png = output_dir / "goal_prediction_quality.png"
    csv_ = output_dir / "goal_prediction_quality.csv"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _save_csv([{"goals": int(g), "home_actual": float(home_act[g]),
                "home_predicted": float(home_pred[g]),
                "away_actual": float(away_act[g]),
                "away_predicted": float(away_pred[g])} for g in x], csv_)
    return png, csv_


def figure_p4_scoreline(pred_scorelines: Dict[Tuple[int, int], float],
                        obs_scorelines: Dict[Tuple[int, int], float],
                        output_dir: Path) -> Tuple[Path, Path]:
    """P4 — Scoreline distributions: observed vs predicted frequencies."""
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    all_scores = sorted(set(pred_scorelines) | set(obs_scorelines),
                        key=lambda s: (1, 99, 99) if s == ("other", "other")
                        else (0, s[0], s[1]))
    p = [pred_scorelines.get(s, 0.0) for s in all_scores]
    o = [obs_scorelines.get(s, 0.0) for s in all_scores]
    labels = [f"{h}-{a}" for h, a in all_scores]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(x - width / 2, o, width, label="Observed", alpha=0.8)
    ax.bar(x + width / 2, p, width, label="Predicted", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right")
    _style_ax(ax, "P4 — Scoreline distribution: observed vs predicted",
              "Scoreline", "Probability / share")
    ax.legend()
    fig.tight_layout()
    png = output_dir / "scoreline_distribution.png"
    csv_ = output_dir / "scoreline_evaluation.csv"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _save_csv([{"scoreline": labels[i], "observed": o[i], "predicted": p[i]}
               for i in range(len(labels))], csv_)
    return png, csv_


def figure_p5_league(league_metrics: List[dict], output_dir: Path) -> Tuple[Path, Path]:
    """P5 — League-specific performance: Brier score by competition."""
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    leagues = [r["league"] for r in league_metrics]
    briers = [r["brier"] for r in league_metrics]
    ns = [r["n"] for r in league_metrics]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(leagues)))
    bars = ax.bar(leagues, briers, color=colors)
    ax.set_ylabel("Out-of-sample Brier score (lower is better)")
    _style_ax(ax, "P5 — Out-of-sample Brier score by league",
              "League", "Brier score")
    for b, n in zip(bars, ns):
        ax.annotate(f"n={n}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=8)
    fig.tight_layout()
    png = output_dir / "league_performance.png"
    csv_ = output_dir / "metrics_by_league.csv"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _save_csv(league_metrics, csv_)
    return png, csv_


def figure_p6_feature_importance(importance: List[dict], output_dir: Path) -> Tuple[Path, Path]:
    """P6 — Feature importance / interpretation.

    Research question: Which pre-match variables contain predictive information?
    """
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    imp = sorted(importance, key=lambda r: r["importance"], reverse=True)[:15]
    names = [r["feature"] for r in imp][::-1]
    vals = [r["importance"] for r in imp][::-1]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(names, vals, color="steelblue", alpha=0.8)
    ax.set_xlabel("Permutation importance (drop in Brier score)")
    _style_ax(ax, "P6 — Feature importance (permutation, held-out data)",
              "Importance", "Feature")
    fig.tight_layout()
    png = output_dir / "feature_importance.png"
    csv_ = output_dir / "feature_importance.csv"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _save_csv(imp, csv_)
    return png, csv_


def figure_p7_baseline_vs_advanced(baseline: dict, advanced: dict,
                                   ci_baseline: Optional[Tuple[float, float]] = None,
                                   ci_advanced: Optional[Tuple[float, float]] = None,
                                   output_dir: Path = None) -> Tuple[Path, Path]:
    """P7 — Baseline vs advanced model with uncertainty intervals."""
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    labels = ["Baseline", "Advanced"]
    means = [baseline["brier"], advanced["brier"]]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(2)
    # Error bars: half-width of the confidence interval (if available).
    yerr = []
    for ci in (ci_baseline, ci_advanced):
        if ci is not None and len(ci) == 2 and all(np.isfinite(ci)):
            yerr.append((ci[1] - ci[0]) / 2.0)
        else:
            yerr.append(0.0)
    ax.errorbar(x, means, yerr=yerr, fmt="o", capsize=6, color="darkred")
    ax.bar(x, means, color=["#6baed6", "#de2d26"], alpha=0.7, width=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Out-of-sample Brier score (lower is better)")
    _style_ax(ax, "P7 — Baseline vs advanced model (Brier score)",
              "Model", "Brier score")
    ax.axhline(baseline["brier"], color="#6baed6", linestyle="--", alpha=0.5)
    fig.tight_layout()
    png = output_dir / "baseline_vs_advanced.png"
    csv_ = output_dir / "model_comparison.csv"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _save_csv([{"model": "baseline", "brier": baseline["brier"],
                "ci_lo": ci_baseline[0] if ci_baseline else None,
                "ci_hi": ci_baseline[1] if ci_baseline else None},
               {"model": "advanced", "brier": advanced["brier"],
                "ci_lo": ci_advanced[0] if ci_advanced else None,
                "ci_hi": ci_advanced[1] if ci_advanced else None}], csv_)
    return png, csv_
