"""Is there enough user-level ground truth for a personalized item cold-start experiment?

Counting only. No model is trained, nothing in Anime Compass or AnimeRec is written to,
and the production ALS factors are never used as a stand-in for ratings: every number
here comes from the raw `rating_complete.csv`.

The question is narrower than coldstart_feasibility.py, which compared ways of choosing
cold items. Here the cold items are fixed by the brief -- TV anime that premiered in
2019-2020 and have real ratings -- and the script measures what an item-disjoint
train / validation / test design would have to work with.

Definitions (Anime Compass's, so counts are comparable with its own):

  positive       rating >= 8
  negative       rating <= 5
  catalog        ids in anime_catalog_serving.json; other ratings are dropped
  premiere       the full "Mon D, YYYY" date opening `Aired` in the MAL catalog
  candidate      MAL type TV, premiere 2019-01-01..2020-12-31, in catalog, >= 1 rating
  warm item      catalog item whose premiere year is known and before 2019
  warm history   a user's positives on warm items

Roles of the candidates, by premiere date (see SPLIT below), are item-disjoint. Every
item that premiered in 2019 or later -- candidate or not, any format -- is outside the
ALS training set in this design, so no cold item of any role reaches the factors.

The ratings carry no timestamps. A user's warm history therefore includes ratings they
gave after the cold titles premiered. This is a simulated cold start, not a backtest.

Usage:
    python src/personalized_feasibility.py --ratings <rating_complete.csv> --animerec <AnimeRec dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pv

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
MAL_CATALOG = REPO / "anime_coldstart" / "data" / "raw" / "anime.csv"
SEQUEL_FLAGS = REPO / "anime_coldstart" / "data" / "cohort.csv"
OUT = ROOT / "outputs" / "personalized_coldstart"

POSITIVE, NEGATIVE = 8, 5
WARM_BEFORE = pd.Timestamp("2019-01-01")
# Time-ordered and item-disjoint. Fall 2020 is kept apart: the ratings file holds only
# completed titles, and those shows were still airing or barely finished at the snapshot.
SPLIT = {
    "cold_train": ("2019-01-01", "2019-09-30"),
    "cold_val": ("2019-10-01", "2020-03-31"),
    "cold_test": ("2020-04-01", "2020-09-30"),
    "cold_censored": ("2020-10-01", "2020-12-31"),
}
WARM_THRESHOLDS = (5, 10, 20, 50)
FULL_DATE = re.compile(r"^([A-Z][a-z]{2}) (\d{1,2}), (\d{4})")
ANY_YEAR = re.compile(r"(19|20)\d{2}")


def premiere(aired: str) -> pd.Timestamp | None:
    match = FULL_DATE.match(aired or "")
    if not match:
        return None
    return pd.to_datetime(f"{match.group(1)} {match.group(2)} {match.group(3)}", format="%b %d %Y")


def first_year(aired: str) -> float:
    match = ANY_YEAR.search(aired or "")
    return float(match.group(0)) if match else np.nan


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe(values: np.ndarray) -> dict:
    if len(values) == 0:
        return {}
    q = np.percentile(values, [0, 10, 25, 50, 75, 90, 100])
    return {"min": int(q[0]), "p10": float(q[1]), "p25": float(q[2]), "median": float(q[3]), "p75": float(q[4]), "p90": float(q[5]), "max": int(q[6]), "mean": round(float(np.mean(values)), 2)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--animerec", type=Path, required=True, help="extracted AnimeRec project (the team repo's shenwei branch)")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUT.mkdir(parents=True, exist_ok=True)

    header = args.ratings.open(encoding="utf-8").readline().strip()
    types = {"user_id": pa.int32(), "anime_id": pa.int32(), "rating": pa.int8()}
    table = pv.read_csv(args.ratings, convert_options=pv.ConvertOptions(column_types=types))
    user, anime, rating = (table[name].to_numpy() for name in ("user_id", "anime_id", "rating"))
    del table

    users, user_index = np.unique(user, return_inverse=True)
    items, item_index = np.unique(anime, return_inverse=True)
    pair = user_index.astype(np.int64) * len(items) + item_index
    duplicate_pairs = int(len(pair) - len(np.unique(pair)))
    del pair, user, anime

    serving = json.loads((args.animerec / "data" / "anime_compass" / "anime_catalog_serving.json").read_text(encoding="utf-8"))
    catalog_ids = np.array(sorted({int(item["id"]) for item in serving}))
    in_catalog = np.isin(items, catalog_ids)
    row_ok = in_catalog[item_index]
    pos_row = row_ok & (rating >= POSITIVE)
    neg_row = row_ok & (rating <= NEGATIVE)

    def per_item(mask: np.ndarray) -> np.ndarray:
        return np.bincount(item_index, weights=mask, minlength=len(items)).astype(np.int64)

    def per_user(mask: np.ndarray) -> np.ndarray:
        return np.bincount(user_index, weights=mask, minlength=len(users)).astype(np.int64)

    mal = pd.read_csv(MAL_CATALOG)
    mal["premiere"] = mal["Aired"].map(premiere)
    mal["year"] = mal["Aired"].map(first_year)
    frame = pd.DataFrame({"anime_id": items, "in_catalog": in_catalog, "n_users": per_item(row_ok), "n_pos": per_item(pos_row), "n_neg": per_item(neg_row)})
    frame["n_mid"] = frame["n_users"] - frame["n_pos"] - frame["n_neg"]
    frame = frame.merge(mal[["MAL_ID", "Name", "Type", "Aired", "premiere", "year"]], left_on="anime_id", right_on="MAL_ID", how="left").drop(columns="MAL_ID")

    # AniList identity and dates, from the team's catalog of titles that have ALS factors.
    anilist = pd.read_csv(args.animerec / "data" / "ani_list" / "catalog.csv", usecols=["anilist_id", "mal_id", "format", "start_date", "source", "episodes"], low_memory=False)
    anilist = anilist.dropna(subset=["mal_id"]).astype({"mal_id": "int64"})
    entries_per_mal = anilist.groupby("mal_id")["anilist_id"].nunique().rename("anilist_entries")
    anilist = anilist.sort_values(["mal_id", "anilist_id"]).drop_duplicates("mal_id").merge(entries_per_mal, on="mal_id")
    anilist = anilist.rename(columns={"format": "anilist_format", "start_date": "anilist_start", "source": "anilist_source", "episodes": "anilist_episodes"})
    frame = frame.merge(anilist, left_on="anime_id", right_on="mal_id", how="left").drop(columns="mal_id")
    frame["anilist_start"] = pd.to_datetime(frame["anilist_start"], errors="coerce")

    # ------------------------------------------------------------ item roles
    role = np.full(len(frame), "undated", dtype=object)
    role[(frame["year"] < WARM_BEFORE.year).to_numpy()] = "warm"
    role[(frame["year"] >= WARM_BEFORE.year).to_numpy()] = "post_cutoff_other"
    is_candidate = (
        frame["in_catalog"] & (frame["Type"] == "TV") & frame["premiere"].between(WARM_BEFORE, pd.Timestamp("2020-12-31")) & (frame["n_users"] > 0)
    ).to_numpy()
    for name, (low, high) in SPLIT.items():
        role[is_candidate & frame["premiere"].between(pd.Timestamp(low), pd.Timestamp(high)).to_numpy()] = name
    role[~frame["in_catalog"].to_numpy()] = "outside_catalog"
    frame["role"] = role

    sequels = pd.read_csv(SEQUEL_FLAGS, usecols=["MAL_ID", "is_sequel"]).rename(columns={"MAL_ID": "anime_id"})
    frame = frame.merge(sequels, on="anime_id", how="left")
    frame["season"] = frame["premiere"].dt.year.astype("Int64").astype(str) + "Q" + frame["premiere"].dt.quarter.astype("Int64").astype(str)
    frame["date_gap_days"] = (frame["anilist_start"] - frame["premiere"]).dt.days

    candidates = frame[is_candidate].sort_values(["premiere", "anime_id"])
    columns = ["anime_id", "Name", "premiere", "season", "role", "is_sequel", "n_users", "n_pos", "n_neg", "n_mid",
               "anilist_id", "anilist_format", "anilist_start", "date_gap_days", "anilist_entries"]  # fmt: skip
    candidates[columns].to_csv(OUT / "candidate_items.csv", index=False)

    role_of_row = role[item_index]
    cold_roles = list(SPLIT)

    # ------------------------------------------------------------- summary
    summary: dict = {
        "ratings_file": {
            "path": str(args.ratings),
            "bytes": args.ratings.stat().st_size,
            "sha256": sha256(args.ratings),
            "header": header,
            "has_timestamp_column": False,
            "rows": int(len(rating)),
            "users": int(len(users)),
            "items": int(len(items)),
            "duplicate_user_item_pairs": duplicate_pairs,
            "rating_counts": {str(value): int(count) for value, count in zip(*np.unique(rating, return_counts=True), strict=True)},
            "rows_in_catalog": int(row_ok.sum()),
            "positives_in_catalog": int(pos_row.sum()),
            "negatives_in_catalog": int(neg_row.sum()),
        },
        "definitions": {"positive": f"rating >= {POSITIVE}", "negative": f"rating <= {NEGATIVE}", "warm_before": str(WARM_BEFORE.date()), "split": SPLIT},
    }

    by_role = frame[frame["in_catalog"]].groupby("role").agg(items=("anime_id", "size"), items_with_positive=("n_pos", lambda s: int((s > 0).sum())), ratings=("n_users", "sum"), positives=("n_pos", "sum"), negatives=("n_neg", "sum"))
    summary["items_by_role"] = json.loads(by_role.to_json(orient="index"))

    pos = candidates["n_pos"].to_numpy()
    summary["candidates"] = {
        "count": int(len(candidates)),
        "by_min_positives": {f">={k}": int((pos >= k).sum()) for k in (1, 10, 50, 100, 500, 1000, 5000)},
        "by_min_negatives": {f">={k}": int((candidates["n_neg"].to_numpy() >= k).sum()) for k in (1, 10, 50, 100)},
        "positives": describe(pos),
        "negatives": describe(candidates["n_neg"].to_numpy()),
        "unique_users_per_item": describe(candidates["n_users"].to_numpy()),
        "positive_share_of_ratings": round(float(pos.sum() / candidates["n_users"].sum()), 4),
        "top10_share_of_positives": round(float(np.sort(pos)[::-1][:10].sum() / pos.sum()), 4),
        "sequel_share_of_items": round(float(candidates["is_sequel"].mean()), 4),
        "sequel_share_of_positives": round(float(candidates.loc[candidates["is_sequel"] == 1, "n_pos"].sum() / pos.sum()), 4),
        "with_anilist_id": int(candidates["anilist_id"].notna().sum()),
        "anilist_format": candidates["anilist_format"].fillna("no AniList entry").value_counts().to_dict(),
        "mal_ids_split_across_anilist_entries": int((candidates["anilist_entries"] > 1).sum()),
        "premiere_date_gap_anilist_minus_mal": {
            "same_day": int((candidates["date_gap_days"] == 0).sum()),
            "within_1_day": int((candidates["date_gap_days"].abs() == 1).sum()),
            "2_to_7_days": int(candidates["date_gap_days"].abs().between(2, 7).sum()),
            "more_than_7_days": int((candidates["date_gap_days"].abs() > 7).sum()),
            "anilist_earlier": int((candidates["date_gap_days"] < 0).sum()),
            "anilist_later": int((candidates["date_gap_days"] > 0).sum()),
        },
    }
    by_season = candidates.groupby("season").agg(items=("anime_id", "size"), items_50_pos=("n_pos", lambda s: int((s >= 50).sum())), ratings=("n_users", "sum"), positives=("n_pos", "sum"), negatives=("n_neg", "sum"), median_users=("n_users", "median"))
    by_season.to_csv(OUT / "candidates_by_season.csv")
    summary["candidates_by_season"] = json.loads(by_season.to_json(orient="index"))

    # ----------------------------------------------------------------- users
    warm_pos = per_user(pos_row & (role_of_row == "warm"))
    warm_any = per_user(row_ok & (role_of_row == "warm"))
    all_cold_pos = per_user(pos_row & np.isin(role_of_row, cold_roles))
    summary["users"] = {
        "warm_positives_per_user": describe(warm_pos),
        "warm_ratings_per_user": describe(warm_any),
        "with_warm_history": {f">={k}": int((warm_pos >= k).sum()) for k in WARM_THRESHOLDS},
        "with_warm_history_and_1_candidate_positive": {f">={k}": int(((warm_pos >= k) & (all_cold_pos >= 1)).sum()) for k in WARM_THRESHOLDS},
        "with_warm_history_and_3_candidate_positives": {f">={k}": int(((warm_pos >= k) & (all_cold_pos >= 3)).sum()) for k in WARM_THRESHOLDS},
    }

    summary["splits"] = {}
    for name in cold_roles:
        in_role = role_of_row == name
        n_items = int((role == name).sum())
        cold_pos = per_user(pos_row & in_role)
        cold_neg = per_user(neg_row & in_role)
        cold_any = per_user(row_ok & in_role)
        block = {"items": n_items, "items_with_50_positives": int(((role == name) & (frame["n_pos"].to_numpy() >= 50)).sum()),
                 "ratings": int((row_ok & in_role).sum()), "positives": int((pos_row & in_role).sum()), "negatives": int((neg_row & in_role).sum())}  # fmt: skip
        for k in WARM_THRESHOLDS:
            eligible = (warm_pos >= k) & (cold_pos >= 1)
            n = int(eligible.sum())
            block[f"warm>={k}"] = {
                "eligible_users": n,
                "eligible_users_with_3_positives": int(((warm_pos >= k) & (cold_pos >= 3)).sum()),
                "eligible_users_with_pos_and_neg": int((eligible & (cold_neg >= 1)).sum()),
                "positives_from_eligible": int(cold_pos[eligible].sum()),
                "negatives_from_eligible": int(cold_neg[eligible].sum()),
                "positives_per_eligible_user": describe(cold_pos[eligible]),
                "rated_share_of_user_item_grid": round(float(cold_any[eligible].sum() / (n * n_items)), 5) if n and n_items else None,
                "positive_share_of_user_item_grid": round(float(cold_pos[eligible].sum() / (n * n_items)), 5) if n and n_items else None,
            }
        role_pos = frame.loc[role == name, "n_pos"].to_numpy()
        block["top10_share_of_positives"] = round(float(np.sort(role_pos)[::-1][:10].sum() / role_pos.sum()), 4) if role_pos.sum() else None
        summary["splits"][name] = block

    # What an ALS trained for this design would see: positives on warm items only.
    train_row = pos_row & (role_of_row == "warm")
    summary["als_training_set_for_this_design"] = {
        "positives": int(train_row.sum()),
        "users_with_1_positive": int((warm_pos >= 1).sum()),
        "items_with_1_positive": int(((role == "warm") & (frame["n_pos"].to_numpy() > 0)).sum()),
        "share_of_all_catalog_positives": round(float(train_row.sum() / pos_row.sum()), 4),
        "positives_removed": {r: int((pos_row & (role_of_row == r)).sum()) for r in (*cold_roles, "post_cutoff_other", "undated")},
    }

    (OUT / "ratings_feasibility.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("ratings_file", "items_by_role", "candidates", "users", "als_training_set_for_this_design")}, indent=1, default=str))
    print(by_season.to_string())
    for name, block in summary["splits"].items():
        print(name, json.dumps({k: v for k, v in block.items() if not k.startswith("warm>=")}), "\n   warm>=5:", json.dumps(block["warm>=5"]), "\n   warm>=20:", json.dumps(block["warm>=20"]))


if __name__ == "__main__":
    main()
