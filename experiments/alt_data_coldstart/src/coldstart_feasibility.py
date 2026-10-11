"""Can the existing MAL ratings support a simulated item cold-start evaluation?

No model is trained and nothing in Anime Compass is written to. The script reads the
raw ratings and counts, for several ways of choosing the cold items, how many items,
positives and users an evaluation would have.

The ratings carry no timestamps. Whatever split is chosen, a user's warm history holds
ratings they gave after the cold title premiered, so this can only ever be a *simulated*
item cold-start. It is not a historical pre-release backtest and must not be reported
as one.

Definitions follow Anime Compass so the counts are comparable with its own:

  positive      rating >= 8 (POSITIVE_THRESHOLD in scripts/build_production_als.py)
  catalog       the ids in data/processed/anime_catalog_serving.json; ratings of ids
                outside it are dropped, as its pipeline does
  warm history  at least 5 positives on warm items (the split's `minimum_positives`)
  premiere      the full "Mon D, YYYY" date at the start of `Aired` in the MAL catalog,
                the same rule as anime_coldstart/src/build_cohort.py

Usage:
    python src/coldstart_feasibility.py --ratings <rating_complete.csv> --serving-catalog <anime_catalog_serving.json>
"""

from __future__ import annotations

import argparse
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
OUT = ROOT / "outputs" / "coldstart_eval"

POSITIVE_THRESHOLD = 8
MIN_WARM_POSITIVES = 5
SEED = 590
FULL_DATE = re.compile(r"^([A-Z][a-z]{2}) (\d{1,2}), (\d{4})")


def premiere(aired: str) -> pd.Timestamp | None:
    match = FULL_DATE.match(aired or "")
    if not match:
        return None
    return pd.to_datetime(f"{match.group(1)} {match.group(2)} {match.group(3)}", format="%b %d %Y")


def quantiles(values: np.ndarray) -> dict:
    if len(values) == 0:
        return {"min": None, "p25": None, "median": None, "p75": None, "max": None}
    q = np.percentile(values, [0, 25, 50, 75, 100])
    return {"min": int(q[0]), "p25": float(q[1]), "median": float(q[2]), "p75": float(q[3]), "max": int(q[4])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--serving-catalog", type=Path, required=True)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUT.mkdir(parents=True, exist_ok=True)

    types = {"user_id": pa.int32(), "anime_id": pa.int32(), "rating": pa.int8()}
    table = pv.read_csv(args.ratings, convert_options=pv.ConvertOptions(column_types=types))
    user = table["user_id"].to_numpy()
    anime = table["anime_id"].to_numpy()
    rating = table["rating"].to_numpy()
    del table
    rows_scanned = len(rating)

    catalog_ids = np.array(sorted({int(item["id"]) for item in json.loads(args.serving_catalog.read_text(encoding="utf-8"))}))
    items, item_index = np.unique(anime, return_inverse=True)
    users, user_index = np.unique(user, return_inverse=True)
    del anime, user
    item_in_catalog = np.isin(items, catalog_ids)
    row_in_catalog = item_in_catalog[item_index]
    positive_row = (rating >= POSITIVE_THRESHOLD) & row_in_catalog

    n_ratings = np.bincount(item_index, weights=row_in_catalog, minlength=len(items)).astype(np.int64)
    n_pos = np.bincount(item_index, weights=positive_row, minlength=len(items)).astype(np.int64)
    pos_per_user = np.bincount(user_index, weights=positive_row, minlength=len(users)).astype(np.int64)

    mal = pd.read_csv(MAL_CATALOG)
    mal["premiere"] = mal["Aired"].map(premiere)
    table = pd.DataFrame({"anime_id": items, "in_serving_catalog": item_in_catalog, "n_ratings": n_ratings, "n_pos": n_pos})
    table = table.merge(mal[["MAL_ID", "Name", "Type", "premiere"]], left_on="anime_id", right_on="MAL_ID", how="left").drop(columns="MAL_ID")
    table.to_parquet(OUT / "item_rating_counts.parquet", index=False)
    year = table["premiere"].dt.year.to_numpy()
    kind = table["Type"].to_numpy()
    rated = item_in_catalog & (n_pos > 0)

    catalog_only = np.setdiff1d(catalog_ids, items)
    summary = {
        "ratings_file": str(args.ratings),
        "serving_catalog": str(args.serving_catalog),
        "has_timestamps": False,
        "rows_scanned": int(rows_scanned),
        "rows_in_catalog": int(row_in_catalog.sum()),
        "rows_outside_catalog": int((~row_in_catalog).sum()),
        "positive_threshold": POSITIVE_THRESHOLD,
        "positives_in_catalog": int(positive_row.sum()),
        "users": int(len(users)),
        "users_with_min_positives": int((pos_per_user >= MIN_WARM_POSITIVES).sum()),
        "catalog_items": int(len(catalog_ids)),
        "catalog_items_with_ratings": int((item_in_catalog & (n_ratings > 0)).sum()),
        "catalog_items_with_positives": int(rated.sum()),
        "catalog_items_without_any_rating": int(len(catalog_only)),
        "items_by_positive_count": {f">={k}": int((rated & (n_pos >= k)).sum()) for k in (1, 5, 20, 50, 100, 1000)},
        "catalog_items_with_full_premiere_date": int((item_in_catalog & ~np.isnan(year)).sum()),
    }

    rng = np.random.default_rng(SEED)
    era_tv = item_in_catalog & (year >= 2016) & (year <= 2020) & (kind == "TV")
    random_fifth = np.zeros(len(items), dtype=bool)
    random_fifth[rng.choice(np.flatnonzero(era_tv), size=int(round(era_tv.sum() * 0.2)), replace=False)] = True
    scenarios = {
        "premiere 2020, all types": item_in_catalog & (year == 2020),
        "premiere 2019-2020, all types": item_in_catalog & (year >= 2019) & (year <= 2020),
        "premiere 2019-2020, TV": item_in_catalog & (year >= 2019) & (year <= 2020) & (kind == "TV"),
        "premiere 2016-2020, TV": era_tv,
        f"random 20% of 2016-2020 TV (seed {SEED})": random_fifth,
    }

    rows = []
    for name, cold_item in scenarios.items():
        cold_row = positive_row & cold_item[item_index]
        warm_row = positive_row & ~cold_item[item_index]
        warm_per_user = np.bincount(user_index, weights=warm_row, minlength=len(users))
        cold_per_user = np.bincount(user_index, weights=cold_row, minlength=len(users))
        has_history = warm_per_user >= MIN_WARM_POSITIVES
        eligible = has_history & (cold_per_user >= 1)
        evaluable_row = cold_row & eligible[user_index]
        evaluable_per_item = np.bincount(item_index, weights=evaluable_row, minlength=len(items)).astype(np.int64)[cold_item]
        cold_pos = n_pos[cold_item]
        row = {
            "scenario": name,
            "cold_items": int(cold_item.sum()),
            "cold_items_with_1_positive": int((cold_pos >= 1).sum()),
            "cold_items_with_10_positives": int((cold_pos >= 10).sum()),
            "cold_items_with_50_positives": int((cold_pos >= 50).sum()),
            "cold_items_with_100_positives": int((cold_pos >= 100).sum()),
            "warm_items_with_1_positive": int((rated & ~cold_item).sum()),
            "warm_items_with_20_positives": int((rated & ~cold_item & (n_pos >= 20)).sum()),
            "warm_positives": int(warm_row.sum()),
            "cold_positives": int(cold_row.sum()),
            "share_of_positives_hidden": round(float(cold_row.sum() / positive_row.sum()), 4),
            "users_with_warm_history": int(has_history.sum()),
            "users_with_warm_history_and_1_cold_positive": int(eligible.sum()),
            "users_with_warm_history_and_3_cold_positives": int((has_history & (cold_per_user >= 3)).sum()),
            "evaluable_user_item_positives": int(evaluable_row.sum()),
            "cold_positives_per_eligible_user_median": float(np.median(cold_per_user[eligible])) if eligible.any() else None,
        }
        row.update({f"evaluable_positives_per_cold_item_{key}": value for key, value in quantiles(evaluable_per_item).items()})
        rows.append(row)

    pd.DataFrame(rows).to_csv(OUT / "cold_item_scenarios.csv", index=False)
    (OUT / "ratings_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print(pd.DataFrame(rows).set_index("scenario").T.to_string())


if __name__ == "__main__":
    main()
