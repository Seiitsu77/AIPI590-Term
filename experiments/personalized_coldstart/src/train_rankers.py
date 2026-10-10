"""Fit M0b, M1 and M2, choose C on validation, refit, and write MODEL_SELECTION.json.

Development only. The two roles named below are the only labels this script asks for,
and data_access would refuse the test roles if it asked.

All three models are the same learner on the same pairs with the same search:

  learner   L2 logistic regression (scikit-learn, lbfgs), features z-scored on the fit set
  fit set   ranker_train users x cold_train titles, every pair, label = rating >= 8
  search    C in {0.01, 0.1, 1, 10}; pick the highest mean user NDCG@10 on
            validation users x cold_val titles; a tie at four decimals goes to the smaller C
  refit     with C fixed, fit once more on the union of both sets; those are the frozen models

With millions of pairs and at most thirteen features the penalty barely moves the fit,
so the search is expected to be nearly flat. It is run anyway because it was registered.

Usage:
    python src/train_rankers.py
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow
import scipy
import sklearn
from sklearn.linear_model import LogisticRegression

from common import ARTIFACTS, MANIFEST, SEED, sha256_file
from data_access import MODEL_SELECTION, labels, manifest, repository_commit, users_of
from scoring import ALL_FEATURES, BASELINES, MODELS, Grid, coverage, model_scores, ndcg_and_recall, observed_auc

C_GRID = (0.01, 0.1, 1.0, 10.0)
FIT_ROLE, VALIDATION_ROLE = "cold_train", "cold_val"


def fit(design: np.ndarray, y: np.ndarray, features: list[str], c: float) -> dict:
    x = design[:, [ALL_FEATURES.index(name) for name in features]].astype(np.float64)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    model = LogisticRegression(C=c, solver="lbfgs", max_iter=1000, random_state=SEED).fit((x - mean) / scale, y)
    return {
        "features": features,
        "C": c,
        "scaler_mean": mean.tolist(),
        "scaler_scale": scale.tolist(),
        "coefficients": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "lbfgs_iterations": int(model.n_iter_[0]),
        "fit_pairs": int(len(y)),
        "fit_positives": int(y.sum()),
    }


def load(role: str) -> tuple[Grid, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grid = Grid(role)
    users = users_of(role)
    design = grid.design(Grid.user_vectors(users))
    liked, disliked = grid.label_matrices(labels(role), users)
    return grid, users, design, liked, disliked


def summarise(scores: np.ndarray, liked: np.ndarray, disliked: np.ndarray) -> dict:
    ndcg, recall = ndcg_and_recall(scores, liked)
    return {"ndcg_at_10": float(ndcg.mean()), "recall_at_10": float(recall.mean()), "observed_auc": float(np.nanmean(observed_auc(scores, liked, disliked))), "item_coverage_at_10": coverage(scores)}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if manifest()["status"] != "frozen":
        raise SystemExit("the split manifest is not frozen")
    if MODEL_SELECTION.exists():
        raise SystemExit("MODEL_SELECTION.json already exists; model selection is done once")

    _, _, fit_design, fit_liked, _ = load(FIT_ROLE)
    val_grid, val_users, val_design, val_liked, val_disliked = load(VALIDATION_ROLE)
    fit_y, val_y = fit_liked.reshape(-1).astype(np.int8), val_liked.reshape(-1).astype(np.int8)
    shape = val_liked.shape

    search, chosen = {}, {}
    for name, features in MODELS.items():
        search[name] = {}
        for c in C_GRID:
            candidate = fit(fit_design, fit_y, features, c)
            search[name][str(c)] = summarise(model_scores(candidate, val_design, shape), val_liked, val_disliked)
            print(f"{name}  C={c:<5}  validation NDCG@10 {search[name][str(c)]['ndcg_at_10']:.4f}", flush=True)
        # Highest validation NDCG@10 at four decimals; a tie goes to the smaller C.
        chosen[name] = min(C_GRID, key=lambda c, name=name: (-round(search[name][str(c)]["ndcg_at_10"], 4), c))

    union_design = np.concatenate([fit_design, val_design])
    union_y = np.concatenate([fit_y, val_y])
    models = {name: fit(union_design, union_y, features, chosen[name]) for name, features in MODELS.items()}

    validation = {name: search[name][str(chosen[name])] for name in MODELS}
    for name in BASELINES:
        validation[name] = summarise(val_grid.baseline(name, len(val_users)), val_liked, val_disliked)

    selection = {
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "models frozen; the final test has not been run",
        "selection_rule": "highest mean user NDCG@10 on validation at four decimals, ties to the smaller C",
        "c_grid": list(C_GRID),
        "chosen_c": chosen,
        "validation_search": search,
        "validation_at_chosen_c": validation,
        "validation_note": "scores of the models fitted on cold_train only; the frozen models below were refitted on cold_train + cold_val and have no held-out development score",
        "validation_set": {"users": int(len(val_users)), "titles": int(shape[1]), "pairs": int(val_y.size), "positives": int(val_y.sum())},
        "frozen_models": models,
        "baselines": {name: f"rank by {column}; titles without the AniList signal last" for name, column in BASELINES.items()},
        "inputs": {
            "manifest_sha256": sha256_file(MANIFEST / "split_manifest.json"),
            **{name: sha256_file(ARTIFACTS / name) for name in ("als_experiment.npz", "item_features.parquet", "item_vectors.npz", "user_vectors.npz", "day_zero.csv")},
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "code": repository_commit(),
    }
    MODEL_SELECTION.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(json.dumps({"chosen_c": chosen, "validation_at_chosen_c": validation, "code": selection["code"]["commit"]}, indent=1))


if __name__ == "__main__":
    main()
