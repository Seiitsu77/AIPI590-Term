"""Write the split manifest, check it for item leakage, freeze it, or verify it.

The manifest is the one place that says which role every title has and which group
every user is in. Every later script reads it and nothing else decides a role.

No model is trained here and no feature is built. The only use of ratings is to count
them and to decide who is eligible, which is part of the split's definition.

Roles (item-disjoint, by MAL premiere date):

  warm              premiere year known and before 2019, any format -> trains the ALS
  cold_train        TV, 2019-01-01..2019-09-30
  cold_val          TV, 2019-10-01..2020-03-31
  cold_test         TV, 2020-04-01..2020-09-30
  cold_test_late    TV, 2020-10-01..2020-12-31 (supplementary; never used for selection)
  excluded_recent   everything else from 2019 on (other formats, undated TV, 2021)
  excluded_undated  premiere year unknown
  Every role except `warm` is kept out of the experimental ALS.

Users: group = SplitMix64(user_id + seed) mod 3 -> ranker_train / validation / test.
A user is eligible for a cold role with at least 5 positives on warm titles and at least
one positive on a title of that role. Development groups are capped at 20,000 eligible
users each (smallest sampling keys); the test group is used in full.

The per-user lists (users_<role>.csv) are written next to the manifest but are not
committed. The manifest records their SHA-256, and `--verify` rebuilds them from the
rule, so a list that differs from the frozen one is detected.

Usage:
    python src/make_split_manifest.py            write a draft
    python src/make_split_manifest.py --freeze   write it frozen (refused if any check fails)
    python src/make_split_manifest.py --verify   rebuild and compare with the frozen manifest
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pv

from common import (
    ALT_DATA_INPUTS,
    ARTIFACTS,
    MAL_CATALOG,
    MAL_CATALOG_SHA256,
    MANIFEST,
    MAX_DEVELOPMENT_USERS,
    MIN_WARM_POSITIVES,
    NEGATIVE,
    POSITIVE,
    RATINGS,
    RATINGS_SHA256,
    ROLES,
    SEED,
    SERVING_CATALOG,
    SERVING_CATALOG_SHA256,
    USER_GROUPS,
    WARM_BEFORE_YEAR,
    approx_date,
    first_year,
    franchise_root,
    premiere_date,
    root_keys,
    sampling_key,
    sha256_file,
    user_group,
)

# Which user group evaluates which cold role. The late test reuses the test group.
ROLE_GROUP = {"cold_train": "ranker_train", "cold_val": "validation", "cold_test": "test", "cold_test_late": "test"}
EXPECTED_ITEMS = {"cold_train": 144, "cold_val": 99, "cold_test": 67, "cold_test_late": 48}
# Files produced before the freeze that the protocol depends on; pinned by digest.
DERIVED = {"day_zero.csv": ARTIFACTS / "day_zero.csv", "wikipedia_article_audit.csv": MANIFEST / "wikipedia_article_audit.csv", "wikipedia_article_audit.json": MANIFEST / "wikipedia_article_audit.json"}
COMPARED = ("inputs", "definitions", "titles_by_role", "als_training_set", "users", "warm_history_users", "leakage_checks", "franchise_overlap", "files", "alt_data_inputs", "derived_files")


def build(out_dir: Path, status: str) -> tuple[dict, bool]:
    """Compute the manifest and write item_roles.csv and the user lists into `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    digests = {"ratings": sha256_file(RATINGS), "mal_catalog": sha256_file(MAL_CATALOG), "serving_catalog": sha256_file(SERVING_CATALOG)}
    expected = {"ratings": RATINGS_SHA256, "mal_catalog": MAL_CATALOG_SHA256, "serving_catalog": SERVING_CATALOG_SHA256}
    for name in digests:
        if digests[name] != expected[name]:
            raise SystemExit(f"{name} does not match its pinned SHA-256; refusing to build a split from a different file")

    types = {"user_id": pa.int32(), "anime_id": pa.int32(), "rating": pa.int8()}
    table = pv.read_csv(RATINGS, convert_options=pv.ConvertOptions(column_types=types))
    user, anime, rating = (table[name].to_numpy() for name in ("user_id", "anime_id", "rating"))
    del table
    users, user_index = np.unique(user, return_inverse=True)
    items, item_index = np.unique(anime, return_inverse=True)
    del user, anime

    catalog_ids = np.array(sorted({int(item["id"]) for item in json.loads(SERVING_CATALOG.read_text(encoding="utf-8"))}))
    mal = pd.read_csv(MAL_CATALOG)
    mal["premiere"] = mal["Aired"].map(premiere_date)
    mal["year"] = mal["Aired"].map(first_year)
    frame = pd.DataFrame({"anime_id": items}).merge(mal[["MAL_ID", "Name", "Type", "Aired", "premiere", "year"]], left_on="anime_id", right_on="MAL_ID", how="left").drop(columns="MAL_ID")
    in_catalog = np.isin(items, catalog_ids)

    # ---------------------------------------------------------------- roles
    role = np.full(len(frame), "excluded_undated", dtype=object)
    year = frame["year"].to_numpy()
    role[year < WARM_BEFORE_YEAR] = "warm"
    role[year >= WARM_BEFORE_YEAR] = "excluded_recent"
    is_tv = (frame["Type"] == "TV").to_numpy()
    for name, (low, high) in ROLES.items():
        role[is_tv & frame["premiere"].between(pd.Timestamp(low), pd.Timestamp(high)).to_numpy()] = name
    role[~in_catalog] = "outside_catalog"
    frame["role"] = role

    row_ok = in_catalog[item_index]
    pos_row = row_ok & (rating >= POSITIVE)
    neg_row = row_ok & (rating <= NEGATIVE)
    role_of_row = role[item_index]

    def per_user(mask: np.ndarray) -> np.ndarray:
        return np.bincount(user_index, weights=mask, minlength=len(users)).astype(np.int64)

    frame["ratings"] = np.bincount(item_index, weights=row_ok, minlength=len(items)).astype(np.int64)
    frame["positives"] = np.bincount(item_index, weights=pos_row, minlength=len(items)).astype(np.int64)

    # ---------------------------------------------------------- franchises
    mal["when"] = mal["Aired"].map(approx_date)
    mal["root"] = mal["Name"].map(franchise_root)
    role_by_id = dict(zip(frame["anime_id"], frame["role"], strict=True))
    type_by_id = dict(zip(mal["MAL_ID"], mal["Type"], strict=True))
    by_root: dict[str, list[tuple[pd.Timestamp, int]]] = {}
    for mal_id, root, when in zip(mal["MAL_ID"], mal["root"], mal["when"], strict=True):
        if root and not pd.isna(when):
            by_root.setdefault(root, []).append((when, int(mal_id)))
    earlier_roles: list[str] = []
    promo_like: list[str] = []
    for anime_id, name, when, item_role in zip(frame["anime_id"], frame["Name"], frame["premiere"], frame["role"], strict=True):
        if item_role not in ROLES:
            earlier_roles.append("")
            promo_like.append("")
            continue
        found = sorted({(other_when, other_id) for key in root_keys(franchise_root(name)) for other_when, other_id in by_root.get(key, ()) if other_when < when and other_id != anime_id})
        earlier_roles.append("|".join(sorted({role_by_id.get(other_id, "not_rated") for _, other_id in found})))
        # A warm, non-TV entry of the same franchise released in the year before the
        # premiere is usually a trailer, prologue or advance special.
        promo_like.append("|".join(str(other_id) for other_when, other_id in found if role_by_id.get(other_id) == "warm" and type_by_id.get(other_id) != "TV" and (when - other_when).days <= 365))
    frame["earlier_franchise_roles"] = earlier_roles
    frame["warm_promo_like_ids"] = promo_like
    frame["is_sequel"] = [int(bool(value)) for value in earlier_roles]

    # ---------------------------------------------------------------- users
    group = user_group(users)
    warm_pos = per_user(pos_row & (role_of_row == "warm"))
    key = sampling_key(users)
    user_files, user_counts = {}, {}
    for role_name, group_name in ROLE_GROUP.items():
        in_role = role_of_row == role_name
        cold_pos = per_user(pos_row & in_role)
        cold_neg = per_user(neg_row & in_role)
        in_group = group == USER_GROUPS.index(group_name)
        eligible = in_group & (warm_pos >= MIN_WARM_POSITIVES) & (cold_pos >= 1)
        chosen = np.flatnonzero(eligible)
        if group_name != "test" and len(chosen) > MAX_DEVELOPMENT_USERS:
            chosen = chosen[np.argsort(key[chosen], kind="stable")[:MAX_DEVELOPMENT_USERS]]
        chosen = np.sort(chosen)
        path = out_dir / f"users_{role_name}.csv"
        pd.DataFrame({"user_id": users[chosen]}).to_csv(path, index=False, lineterminator="\n")
        user_files[role_name] = path.name
        user_counts[role_name] = {
            "user_group": group_name,
            "users_in_group": int(in_group.sum()),
            "eligible_in_group": int(eligible.sum()),
            "users_used": int(len(chosen)),
            "positives": int(cold_pos[chosen].sum()),
            "negatives": int(cold_neg[chosen].sum()),
            "users_with_positive_and_negative": int(((cold_pos[chosen] >= 1) & (cold_neg[chosen] >= 1)).sum()),
            "user_item_pairs": int(len(chosen) * int((role == role_name).sum())),
            "median_warm_positives": float(np.median(warm_pos[chosen])),
        }
    # The auxiliary evaluation takes every user of a group with enough warm history.
    # Membership uses warm ratings only; the cold counts below are recorded, not used.
    warm_history = {}
    for group_name, role_name in (("validation", "cold_val"), ("test", "cold_test")):
        members = (group == USER_GROUPS.index(group_name)) & (warm_pos >= MIN_WARM_POSITIVES)
        warm_history[group_name] = {"users": int(members.sum()), "of_those_with_a_positive_on": {role_name: int((members & (per_user(pos_row & (role_of_row == role_name)) >= 1)).sum())}}

    columns = ["anime_id", "Name", "Type", "Aired", "premiere", "year", "role", "ratings", "positives", "is_sequel", "earlier_franchise_roles", "warm_promo_like_ids"]
    items_path = out_dir / "item_roles.csv"
    frame[frame["role"] != "outside_catalog"][columns].to_csv(items_path, index=False, lineterminator="\n")

    # ------------------------------------------------------------- leakage
    als_items = frame[(frame["role"] == "warm") & (frame["positives"] > 0)]
    cold = frame[frame["role"].isin(ROLES)]
    kept_out = frame[frame["role"].isin([*ROLES, "excluded_recent", "excluded_undated"])]
    group_sets = {name: set(pd.read_csv(out_dir / file)["user_id"]) for name, file in user_files.items()}
    counts_ok = {name: int((role == name).sum()) == count for name, count in EXPECTED_ITEMS.items()}
    checks = {
        "every_rated_title_has_exactly_one_role": bool(frame["anime_id"].is_unique and frame["role"].notna().all()),
        "cold_roles_have_the_expected_number_of_titles": counts_ok,
        "latest_premiere_year_in_als_training": int(als_items["year"].max()),
        "no_title_from_2019_on_in_als_training": bool((als_items["year"] < WARM_BEFORE_YEAR).all()),
        "no_undated_title_in_als_training": bool(als_items["year"].notna().all()),
        "als_training_titles_overlapping_kept_out_titles": int(len(set(als_items["anime_id"]) & set(kept_out["anime_id"]))),
        "cold_roles_pairwise_disjoint": bool(cold["anime_id"].is_unique),
        "development_and_test_users_disjoint": bool(not (group_sets["cold_train"] & group_sets["cold_val"]) and not (group_sets["cold_train"] & group_sets["cold_test"]) and not (group_sets["cold_val"] & group_sets["cold_test"])),
        "late_test_users_are_from_the_test_group_only": bool(not (group_sets["cold_test_late"] & (group_sets["cold_train"] | group_sets["cold_val"]))),
        "warm_history_test_users_with_a_positive_equal_the_eligible_test_users": bool(warm_history["test"]["of_those_with_a_positive_on"]["cold_test"] == user_counts["cold_test"]["eligible_in_group"]),
    }
    passed = all(value for value in checks.values() if isinstance(value, bool)) and all(counts_ok.values()) and checks["als_training_titles_overlapping_kept_out_titles"] == 0

    def franchise_block(role_name: str) -> dict:
        rows = frame[frame["role"] == role_name]
        roles = rows["earlier_franchise_roles"]
        return {
            "titles": int(len(rows)),
            "with_an_earlier_franchise_title": int((roles != "").sum()),
            "earlier_title_is_warm": int(roles.str.contains("warm").sum()),
            "earlier_titles_only_kept_out_of_als": int(((roles != "") & ~roles.str.contains("warm")).sum()),
            "earlier_title_in_another_cold_role": int(roles.str.contains("cold_").sum()),
            "with_a_warm_promo_like_entry": int((rows["warm_promo_like_ids"] != "").sum()),
        }

    train_row = pos_row & (role_of_row == "warm")
    manifest = {
        "status": status,
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {
            "ratings": {"path": str(RATINGS), "sha256": digests["ratings"], "bytes": RATINGS.stat().st_size, "rows": int(len(rating)), "users": int(len(users)), "has_timestamps": False},
            "mal_catalog": {"path": MAL_CATALOG.relative_to(MANIFEST.parents[2]).as_posix(), "sha256": digests["mal_catalog"], "snapshot": "early 2021"},
            "serving_catalog": {"path": str(SERVING_CATALOG), "sha256": digests["serving_catalog"], "titles": int(len(catalog_ids))},
        },
        "definitions": {
            "positive": f"rating >= {POSITIVE}",
            "negative": f"rating <= {NEGATIVE}",
            "warm": f"premiere year known and < {WARM_BEFORE_YEAR}",
            "cold_roles": ROLES,
            "min_warm_positives": MIN_WARM_POSITIVES,
            "seed": SEED,
            "user_group": "SplitMix64(user_id + seed) mod 3 -> " + " / ".join(USER_GROUPS),
            "development_user_cap": MAX_DEVELOPMENT_USERS,
            "development_user_choice": "eligible users with the smallest SplitMix64(user_id + seed + 1)",
        },
        "titles_by_role": {name: {"titles": int((role == name).sum()), "positives": int((pos_row & (role_of_row == name)).sum())} for name in ("warm", *ROLES, "excluded_recent", "excluded_undated")},
        "als_training_set": {
            "titles_with_a_positive": int(len(als_items)),
            "positives": int(train_row.sum()),
            "users_with_a_positive": int((warm_pos >= 1).sum()),
            "share_of_catalog_positives": round(float(train_row.sum() / pos_row.sum()), 4),
        },
        "users": user_counts,
        "warm_history_users": warm_history,
        "leakage_checks": {"all_passed": bool(passed), **checks},
        "franchise_overlap": {name: franchise_block(name) for name in ROLES},
        "files": {"item_roles.csv": sha256_file(items_path), **{file: sha256_file(out_dir / file) for file in user_files.values()}},
        "files_not_committed": sorted(user_files.values()),
        "alt_data_inputs": {name: {"path": path.relative_to(MANIFEST.parents[2]).as_posix(), "sha256": sha256_file(path)} for name, path in ALT_DATA_INPUTS.items()},
        "derived_files": {name: sha256_file(path) for name, path in DERIVED.items() if path.exists()},
    }
    return manifest, passed


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--freeze", action="store_true", help="write the manifest frozen; refused if any leakage check fails")
    mode.add_argument("--verify", action="store_true", help="rebuild from the rule and compare with the manifest on disk")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    manifest_path = MANIFEST / "split_manifest.json"
    on_disk = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None

    if args.verify:
        if on_disk is None:
            raise SystemExit("there is no manifest to verify")
        scratch = ARTIFACTS / "manifest_verify"
        rebuilt, passed = build(scratch, on_disk["status"])
        # Compare what JSON would store, so a tuple in memory equals the list it was saved as.
        rebuilt = json.loads(json.dumps(rebuilt))
        differing = [key for key in COMPARED if rebuilt.get(key) != on_disk.get(key)]
        for name in rebuilt["files_not_committed"]:
            if not (MANIFEST / name).exists() and not differing:
                shutil.copyfile(scratch / name, MANIFEST / name)
        print(json.dumps({"status_on_disk": on_disk["status"], "leakage_checks_passed": passed, "sections_compared": list(COMPARED), "sections_differing": differing}, indent=1))
        if differing or not passed:
            raise SystemExit("the manifest on disk does not match what the rule produces")
        return

    if on_disk is not None and on_disk.get("status") == "frozen":
        raise SystemExit("manifest is frozen; it is not regenerated. Use --verify.")
    manifest, passed = build(MANIFEST, "frozen" if args.freeze else "draft")
    if args.freeze:
        missing = [name for name, path in DERIVED.items() if not path.exists()]
        if missing or not passed:
            raise SystemExit(f"not frozen: leakage checks passed = {passed}, missing derived files = {missing}")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("status", "titles_by_role", "als_training_set", "users", "warm_history_users", "leakage_checks", "derived_files")}, indent=1))
    if not passed:
        raise SystemExit("leakage checks failed; the manifest was written for inspection but must not be frozen")


if __name__ == "__main__":
    main()
