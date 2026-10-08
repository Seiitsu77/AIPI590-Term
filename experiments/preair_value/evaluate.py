"""Run the pre-registered pre-air value test (see PLAN.md).

Question: do AniList pre-air signals (A) rank a season's new TV anime better than
metadata alone (M)?

    H1 (primary)   ridge, label y_score80, all 394 test titles:
                   PASS iff delta > 0 and the 95% paired-bootstrap CI lower bound > 0
    H2 (secondary) same, on test titles that are not sequels (is_sequel == 0)

Everything else (boosted trees, MAL-members label, A-only, per-year splits) is
descriptive and has no pass/fail.

Metric: Spearman correlation between prediction and label inside each test season,
averaged over the 8 test seasons. Models are fit once on the 2021-2023 seasons;
the bootstrap resamples test titles within each season, the same draw for both
models, so the CI is for the paired difference.

--dry-run shuffles the labels within every season (train and test) before
fitting. It exercises the whole pipeline without revealing anything about the
real result; its deltas should hover around zero.

Usage:
    python experiments/preair_value/evaluate.py --dry-run
    python experiments/preair_value/evaluate.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import ALT_FEATURES, METADATA_FEATURES

HERE = Path(__file__).resolve().parent
FEATURES = HERE / "data" / "features.csv"
RESULTS = HERE / "results"
BOOTSTRAP_REPS = 2000
SEED = 20261008
RIDGE_ALPHAS = np.logspace(-2, 3, 11)


def load() -> list[dict[str, Any]]:
    with FEATURES.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        for column in (*METADATA_FEATURES, *ALT_FEATURES, "y_score80"):
            row[column] = float(row[column])
        row["y_mal_members"] = float(row["y_mal_members"]) if row["y_mal_members"] else None
    return rows


def make_model(kind: str):
    if kind == "ridge":
        return make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))
    return HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0, random_state=0
    )


def matrix(rows: list[dict], columns: list[str]) -> np.ndarray:
    return np.array([[row[column] for column in columns] for row in rows], dtype=float)


def fit_predict(kind: str, columns: list[str], train: list[dict], test: list[dict], label: str) -> np.ndarray:
    model = make_model(kind)
    model.fit(matrix(train, columns), np.array([row[label] for row in train]))
    return model.predict(matrix(test, columns))


def season_groups(test: list[dict]) -> list[np.ndarray]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(test):
        groups[row["listed_season"]].append(index)
    return [np.array(indices) for _, indices in sorted(groups.items())]


def mean_within_season(pred: np.ndarray, y: np.ndarray, groups: list[np.ndarray]) -> float:
    return float(np.mean([spearmanr(pred[g], y[g])[0] for g in groups]))


def paired_bootstrap(pred_base, pred_new, y, groups, rng) -> tuple[float, float, float]:
    point = mean_within_season(pred_new, y, groups) - mean_within_season(pred_base, y, groups)
    deltas = np.empty(BOOTSTRAP_REPS)
    for rep in range(BOOTSTRAP_REPS):
        draw = [rng.choice(g, size=len(g), replace=True) for g in groups]
        deltas[rep] = np.mean([spearmanr(pred_new[d], y[d])[0] - spearmanr(pred_base[d], y[d])[0] for d in draw])
    low, high = np.percentile(deltas, [2.5, 97.5])
    return point, float(low), float(high)


def compare(kind: str, train: list[dict], test: list[dict], label: str, rng) -> dict[str, Any]:
    """Fit M, M+A and A on train; score on test; paired bootstrap of M+A minus M."""
    y = np.array([row[label] for row in test])
    groups = season_groups(test)
    pred_m = fit_predict(kind, METADATA_FEATURES, train, test, label)
    pred_ma = fit_predict(kind, METADATA_FEATURES + ALT_FEATURES, train, test, label)
    pred_a = fit_predict(kind, ALT_FEATURES, train, test, label)
    delta, low, high = paired_bootstrap(pred_m, pred_ma, y, groups, rng)
    return {
        "model": kind,
        "label": label,
        "n_train": len(train),
        "n_test": len(test),
        "n_test_seasons": len(groups),
        "spearman_M": mean_within_season(pred_m, y, groups),
        "spearman_M+A": mean_within_season(pred_ma, y, groups),
        "spearman_A": mean_within_season(pred_a, y, groups),
        "delta": delta,
        "ci95": [low, high],
        "pass": bool(delta > 0 and low > 0),
    }


def shuffle_labels(rows: list[dict], rng) -> None:
    by_season: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_season[row["listed_season"]].append(row)
    for season_rows in by_season.values():
        for label in ("y_score80", "y_mal_members"):
            values = [row[label] for row in season_rows]
            rng.shuffle(values)
            for row, value in zip(season_rows, values):
                row[label] = value


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-registered pre-air value test.")
    parser.add_argument("--dry-run", action="store_true", help="Shuffle labels within seasons; do not save results.")
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    rows = load()
    if args.dry_run:
        shuffle_labels(rows, np.random.default_rng(SEED + 1))
    train = [row for row in rows if row["split"] == "train"]
    test = [row for row in rows if row["split"] == "test"]
    non_sequel = [row for row in test if row["is_sequel"] == 0]

    results: dict[str, Any] = {
        "H1": compare("ridge", train, test, "y_score80", rng),
        "H2": compare("ridge", train, non_sequel, "y_score80", rng),
        "descriptive": {
            "H1_boosted_trees": compare("hgb", train, test, "y_score80", rng),
            "H2_boosted_trees": compare("hgb", train, non_sequel, "y_score80", rng),
            "H1_mal_members": compare(
                "ridge",
                [r for r in train if r["y_mal_members"] is not None],
                [r for r in test if r["y_mal_members"] is not None],
                "y_mal_members",
                rng,
            ),
            "H2_mal_members": compare(
                "ridge",
                [r for r in train if r["y_mal_members"] is not None],
                [r for r in non_sequel if r["y_mal_members"] is not None],
                "y_mal_members",
                rng,
            ),
            **{
                f"H1_test_{year}": compare("ridge", train, [r for r in test if r["listed_season"].startswith(year)],
                                           "y_score80", rng)
                for year in ("2024", "2025")
            },
        },
    }  # fmt: skip

    header = "DRY RUN (labels shuffled within seasons; numbers are meaningless)" if args.dry_run else "RESULTS"
    print(header)
    print(f"{'analysis':<24}{'n_test':>7}{'M':>8}{'M+A':>8}{'A':>8}{'delta':>9}  {'95% CI':<18}{'pass':>5}")
    flat = {"H1": results["H1"], "H2": results["H2"], **results["descriptive"]}
    for name, r in flat.items():
        ci = f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]"
        verdict = ("PASS" if r["pass"] else "FAIL") if name in ("H1", "H2") else "-"
        print(f"{name:<24}{r['n_test']:>7}{r['spearman_M']:>8.3f}{r['spearman_M+A']:>8.3f}{r['spearman_A']:>8.3f}"
              f"{r['delta']:>+9.3f}  {ci:<18}{verdict:>5}")  # fmt: skip

    if not args.dry_run:
        RESULTS.mkdir(exist_ok=True)
        results["run_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        (RESULTS / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved {RESULTS / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
