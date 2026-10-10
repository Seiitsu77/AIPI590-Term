"""Train the experimental ALS on warm titles only.

This model exists for the experiment and nowhere else. It never sees a title that
premiered in 2019 or later, it is written under artifacts/ with the role "experiment",
and the production factors of Anime Compass are neither read nor touched.

Hyperparameters are Anime Compass's frozen production values and are not tuned here:
128 factors, 15 iterations, regularisation 0.05, alpha 5, 3 conjugate-gradient steps,
seed 42, binary confidence. The solver is that project's own (see als_solver.py).

Usage:
    python src/train_als.py
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

from als_solver import solve_factors
from common import ARTIFACTS, sha256_file
from data_access import item_roles, manifest, repository_commit, warm_positives

FACTORS, ITERATIONS, REGULARIZATION, ALPHA, CG_STEPS, ALS_SEED = 128, 15, 0.05, 5.0, 3, 42
SAMPLE_USERS = 2000  # trained user factors kept for the fold-in sanity check


def compressed_rows(row_index: np.ndarray, column_index: np.ndarray, rows: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(row_index, kind="stable")
    indptr = np.concatenate([[0], np.cumsum(np.bincount(row_index, minlength=rows))]).astype(np.int64)
    return indptr, column_index[order].astype(np.int64)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if manifest()["status"] != "frozen":
        raise SystemExit("the split manifest is not frozen; freeze and commit it before training")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    user, anime = warm_positives()
    users, user_index = np.unique(user, return_inverse=True)
    items, item_index = np.unique(anime, return_inverse=True)
    roles = item_roles().set_index("anime_id")["role"]
    if set(roles.loc[items].unique()) != {"warm"}:
        raise SystemExit("a non-warm title reached the ALS training set")

    user_indptr, user_indices = compressed_rows(user_index, item_index, len(users))
    item_indptr, item_indices = compressed_rows(item_index, user_index, len(items))

    generator = np.random.default_rng(ALS_SEED)
    user_factors = generator.normal(0.0, 0.01, size=(len(users), FACTORS)).astype(np.float32)
    item_factors = generator.normal(0.0, 0.01, size=(len(items), FACTORS)).astype(np.float32)
    for iteration in range(ITERATIONS):
        solve_factors(user_factors, item_factors, user_indptr, user_indices, REGULARIZATION, ALPHA, CG_STEPS)
        solve_factors(item_factors, user_factors, item_indptr, item_indices, REGULARIZATION, ALPHA, CG_STEPS)
        print(f"iteration {iteration + 1}/{ITERATIONS}  {time.perf_counter() - started:.0f}s", flush=True)
    if not np.isfinite(item_factors).all() or not np.isfinite(user_factors).all():
        raise SystemExit("ALS diverged: non-finite factors")

    sample = np.random.default_rng(ALS_SEED + 1).choice(len(users), size=SAMPLE_USERS, replace=False)
    metadata = {
        "artifact_role": "experiment",
        "method": "implicit-feedback ALS (conjugate gradient), solver copied from anime-compass@81e5979",
        "trained_on": "positives (rating >= 8) on warm titles only: premiere year known and before 2019",
        "factors": FACTORS,
        "iterations": ITERATIONS,
        "regularization": REGULARIZATION,
        "alpha": ALPHA,
        "cg_steps": CG_STEPS,
        "seed": ALS_SEED,
        "positives": int(len(user)),
        "users": int(len(users)),
        "titles": int(len(items)),
        "latest_premiere_year": int(item_roles().set_index("anime_id").loc[items, "year"].max()),
        "manifest_sha256": sha256_file(ARTIFACTS.parent / "manifest" / "split_manifest.json"),
        "training_seconds": round(time.perf_counter() - started, 1),
        "code": repository_commit(),
    }
    path = ARTIFACTS / "als_experiment.npz"
    np.savez_compressed(
        path,
        anime_ids=items.astype(np.int64),
        item_factors=item_factors,
        sample_user_ids=users[sample].astype(np.int64),
        sample_user_factors=user_factors[sample],
        metadata_json=json.dumps(metadata),
    )
    print(json.dumps({**metadata, "artifact": path.name, "sha256": sha256_file(path)}, indent=1))


if __name__ == "__main__":
    main()
