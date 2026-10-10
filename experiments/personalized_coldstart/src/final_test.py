"""The final test. Run once, after MODEL_SELECTION.json is committed.

Scores the frozen models on the test roles and computes everything the pre-registration
lists, in one pass, so the test labels are read a single time:

  primary        cold_test, the manifest's positive-eligible test users
  comparisons    H1 M2 vs M1 (the only primary hypothesis); secondary M2 vs M0a_raw,
                 M2 vs M0a_norm, M2 vs M0b
  uncertainty    paired user bootstrap; leave-one-title-out; title-resampling bootstrap
  sensitivity    ten most-liked titles removed; titles with an AniList signal only;
                 new titles vs sequels
  auxiliary      test-group users chosen by warm history alone (no cold label decides
                 who is in)
  supplementary  cold_test_late (2020 Q4)

`--dry-run-on-validation` runs the same code on cold_val with validation users. It
exists to exercise every code path before the real run. The frozen models were refitted
on data that includes cold_val, so its numbers are in-sample and are not results.

Usage:
    python src/final_test.py --dry-run-on-validation
    python src/final_test.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

import numpy as np

from common import ARTIFACTS, ROOT, SEED
from data_access import FINAL_TEST_LOCK, MODEL_SELECTION, final_test_permitted, item_roles, labels, repository_commit, users_of, warm_history_users
from scoring import BASELINES, K, MODELS, Grid, coverage, model_scores, ndcg_and_recall, observed_auc, top_k

COMPARISONS = {"H1_primary": ("M2", "M1"), "secondary_vs_M0a_raw": ("M2", "M0a_raw"), "secondary_vs_M0a_norm": ("M2", "M0a_norm"), "secondary_vs_M0b": ("M2", "M0b")}
USER_DRAWS, TITLE_DRAWS, TOP_REMOVED = 2000, 1000, 10
USER_BLOCK = 20_000  # users scored at a time, to bound memory
RESULTS = ROOT / "results"


def all_scores(grid: Grid, users: np.ndarray, frozen: dict) -> dict[str, np.ndarray]:
    """Score every baseline and frozen model for users x the grid's titles, a block of users at a time."""
    vectors = Grid.user_vectors(users)
    scores = {name: grid.baseline(name, len(users)) for name in BASELINES}
    blocks: dict[str, list[np.ndarray]] = {name: [] for name in MODELS}
    for start in range(0, len(users), USER_BLOCK):
        block = vectors[start : start + USER_BLOCK]
        design = grid.design(block)
        for name in MODELS:
            blocks[name].append(model_scores(frozen[name], design, (len(block), len(grid.anime_ids))))
    scores.update({name: np.concatenate(parts) for name, parts in blocks.items()})
    return scores


def mean_ndcg(scores: np.ndarray, liked: np.ndarray, columns: np.ndarray) -> float:
    """Mean NDCG@10 on a subset (or resample) of titles, over users who still have a liked title."""
    sub = liked[:, columns]
    keep = sub.any(axis=1)
    return float(ndcg_and_recall(scores[:, columns][keep], sub[keep])[0].mean()) if keep.any() else float("nan")


def evidence_level(delta: float, low: float, leave_one_out: np.ndarray) -> str:
    """Pre-defined evidence levels. They grade the evidence; they are not guarantees of generalisation."""
    if delta > 0 and low > 0:
        return "A" if (leave_one_out > 0).all() else "B"
    return "C"


def metrics_block(scores: dict[str, np.ndarray], liked: np.ndarray, disliked: np.ndarray) -> dict:
    out = {}
    for name, s in scores.items():
        ndcg, recall = ndcg_and_recall(s, liked)
        out[name] = {"ndcg_at_10": float(ndcg.mean()), "recall_at_10": float(recall.mean()), "observed_auc": float(np.nanmean(observed_auc(s, liked, disliked))), "item_coverage_at_10": coverage(s)}
    return out


def user_bootstrap(per_user: dict[str, np.ndarray], generator: np.random.Generator) -> dict:
    """Paired over users: every model is scored on the same resampled users."""
    n = len(next(iter(per_user.values())))
    draws = {name: np.empty(USER_DRAWS) for name in COMPARISONS}
    for b in range(USER_DRAWS):
        index = generator.integers(0, n, n)
        for name, (a, ref) in COMPARISONS.items():
            draws[name][b] = (per_user[a][index] - per_user[ref][index]).mean()
    out = {}
    for name, (a, ref) in COMPARISONS.items():
        low, high = np.percentile(draws[name], [2.5, 97.5])
        out[name] = {"delta": float((per_user[a] - per_user[ref]).mean()), "user_bootstrap_ci95": [float(low), float(high)]}
    return out


def evaluate(grid: Grid, users: np.ndarray, liked: np.ndarray, disliked: np.ndarray, frozen: dict, *, full: bool) -> dict:
    scores = all_scores(grid, users, frozen)
    generator = np.random.default_rng(SEED)
    m = liked.shape[1]
    result = {"users": int(len(users)), "titles": int(m), "positives": int(liked.sum()), "users_with_liked_and_disliked": int((liked.any(axis=1) & disliked.any(axis=1)).sum()), "metrics": metrics_block(scores, liked, disliked)}
    per_user = {name: ndcg_and_recall(s, liked)[0] for name, s in scores.items()}
    comparisons = user_bootstrap(per_user, generator)

    # Leave one title out: how much does each comparison lean on a single title?
    every = np.arange(m)
    without = {name: np.array([mean_ndcg(s, liked, every[every != j]) for j in range(m)]) for name, s in scores.items()}
    for name, (a, ref) in COMPARISONS.items():
        deltas = without[a] - without[ref]
        comparisons[name]["leave_one_title_out"] = {
            "min_delta": float(deltas.min()),
            "max_delta": float(deltas.max()),
            "deletions_with_delta_at_or_below_zero": int((deltas <= 0).sum()),
            "most_influential_title": int(grid.anime_ids[np.argmax(np.abs(deltas - comparisons[name]["delta"]))]),
        }
        comparisons[name]["evidence_level"] = evidence_level(comparisons[name]["delta"], comparisons[name]["user_bootstrap_ci95"][0], deltas)

    if full:
        # Resampling titles shows how much the answer depends on which titles happened to
        # be in the test season. It is descriptive: 67 titles from two seasons are not a
        # random sample of future titles.
        draws = {name: np.empty(TITLE_DRAWS) for name in scores}
        for b in range(TITLE_DRAWS):
            columns = np.sort(generator.integers(0, m, m))
            for name, s in scores.items():
                draws[name][b] = mean_ndcg(s, liked, columns)
        for name, (a, ref) in COMPARISONS.items():
            low, high = np.nanpercentile(draws[a] - draws[ref], [2.5, 97.5])
            comparisons[name]["title_bootstrap_interval95"] = [float(low), float(high)]
    result["comparisons"] = comparisons

    if full:
        popularity = liked.sum(axis=0)
        top = np.argsort(-popularity, kind="stable")[:TOP_REMOVED]
        roles = item_roles().set_index("anime_id").loc[grid.anime_ids]
        sequel = roles["is_sequel"].to_numpy() == 1
        has_signal = grid.items["al_missing"].to_numpy() == 0
        subsets = {
            "top_10_most_liked_removed": ~np.isin(every, top),
            "titles_with_anilist_signal": has_signal,
            "new_titles": ~sequel,
            "sequels": sequel,
        }
        result["sensitivity"] = {}
        for label, mask in subsets.items():
            columns = every[mask]
            sub_liked, sub_disliked = liked[:, columns], disliked[:, columns]
            keep = sub_liked.any(axis=1)
            sub_scores = {name: s[:, columns][keep] for name, s in scores.items()}
            sub_per_user = {name: ndcg_and_recall(s, sub_liked[keep])[0] for name, s in sub_scores.items()}
            result["sensitivity"][label] = {
                "titles": int(mask.sum()),
                "users": int(keep.sum()),
                "metrics": metrics_block(sub_scores, sub_liked[keep], sub_disliked[keep]),
                "comparisons": user_bootstrap(sub_per_user, np.random.default_rng(SEED)),
            }
        result["sensitivity"]["top_10_most_liked_removed"]["removed_titles"] = [int(a) for a in grid.anime_ids[top]]
        # Titles with no attention signal at all are too few to score as a group; show where each model puts them.
        silent = np.flatnonzero((grid.items["al_missing"].to_numpy() == 1) & (grid.items["wiki_missing"].to_numpy() == 1))
        ranks = {name: np.argsort(np.argsort(-s, axis=1, kind="stable"), axis=1, kind="stable") + 1 for name, s in scores.items()}
        result["titles_without_any_attention_signal"] = [
            {"anime_id": int(grid.anime_ids[j]), "positives": int(popularity[j]), "mean_rank": {name: float(r[:, j].mean()) for name, r in ranks.items()}} for j in silent
        ]
    return result


def auxiliary(grid: Grid, frozen: dict, *, group: str, role: str, final_test: bool) -> dict:
    """Users chosen by warm history alone. A user with no liked title scores 0 under every model."""
    users = warm_history_users(group)
    liked, disliked = grid.label_matrices(labels(role, users=users, final_test=final_test), users)
    scores = all_scores(grid, users, frozen)
    per_user = {name: ndcg_and_recall(s, liked)[0] for name, s in scores.items()}
    return {
        "membership": f"{group} group, at least five warm positives; no {role} label decides who is in",
        "users": int(len(users)),
        "share_with_a_liked_title": float(liked.any(axis=1).mean()),
        "metrics": {
            name: {
                "ndcg_at_10": float(per_user[name].mean()),
                "recall_at_10": float(ndcg_and_recall(s, liked)[1].mean()),
                "item_coverage_at_10": coverage(s),
                "disliked_titles_in_top_10_per_user": float(np.take_along_axis(disliked, top_k(s, K), axis=1).sum(axis=1).mean()),
            }
            for name, s in scores.items()
        },
        "comparisons": user_bootstrap(per_user, np.random.default_rng(SEED)),
        "note": "NDCG and recall here equal the conditional values times the share of users with a liked title, because the others score 0 under every model",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run-on-validation", action="store_true", help="exercise the code on cold_val; the numbers are in-sample and are not results")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    frozen = json.loads(MODEL_SELECTION.read_text(encoding="utf-8"))["frozen_models"]

    if args.dry_run_on_validation:
        grid, users = Grid("cold_val"), users_of("cold_val")
        liked, disliked = grid.label_matrices(labels("cold_val"), users)
        report = {
            "warning": "DRY RUN on validation. The frozen models were refitted on data that includes cold_val; these numbers are in-sample and are not results.",
            "primary": evaluate(grid, users, liked, disliked, frozen, full=True),
            "auxiliary_warm_history_users": auxiliary(grid, frozen, group="validation", role="cold_val", final_test=False),
        }
        (ARTIFACTS / "dry_run_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("dry run finished; every code path executed. Output: artifacts/dry_run_validation.json")
        return

    permitted, reason = final_test_permitted()
    if not permitted:
        raise SystemExit(f"final test refused: {reason}")
    code = repository_commit()
    if not code["experiment_folder_clean"]:
        raise SystemExit("final test refused: the experiment folder has uncommitted changes")
    started = ARTIFACTS / "FINAL_TEST_STARTED.json"
    earlier_starts = json.loads(started.read_text(encoding="utf-8")) if started.exists() else []
    started.write_text(json.dumps([*earlier_starts, {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "commit": code["commit"]}], indent=1), encoding="utf-8")

    report = {"run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "code": code, "starts_before_this_one": earlier_starts}
    grid, users = Grid("cold_test"), users_of("cold_test")
    liked, disliked = grid.label_matrices(labels("cold_test", final_test=True), users)
    report["primary"] = evaluate(grid, users, liked, disliked, frozen, full=True)
    report["auxiliary_warm_history_users"] = auxiliary(grid, frozen, group="test", role="cold_test", final_test=True)
    late_grid, late_users = Grid("cold_test_late"), users_of("cold_test_late")
    late_liked, late_disliked = late_grid.label_matrices(labels("cold_test_late", final_test=True), late_users)
    report["supplementary_cold_test_late"] = evaluate(late_grid, late_users, late_liked, late_disliked, frozen, full=False)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "final_test_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    FINAL_TEST_LOCK.write_text(report["run_utc"], encoding="utf-8")
    print(json.dumps({"H1": report["primary"]["comparisons"]["H1_primary"], "metrics": report["primary"]["metrics"]}, indent=1))


if __name__ == "__main__":
    main()
