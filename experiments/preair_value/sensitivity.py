"""Post-hoc sensitivity checks for the pre-air value test. NOT pre-registered.

Written after the pre-registered run, to test one leakage worry: AniList marks
131 cohort titles as already "releasing" (or airing an episode) on a trend row
dated before their recorded start date, almost always only on day -1. If the
start date is late by a day (late-night slots, early streaming), the day -1
features could contain premiere-day activity.

    cutoff_-8          rebuild the AniList features using only rows with day <= -8:
                       popularity at -8/-30/-90, growth from -30 to -8, trending over
                       days -14..-8. Every flagged row but two lies after day -8.
    drop_flagged       drop the 131 flagged titles from train and test, keep the
                       pre-registered features

Models, metric and bootstrap are the pre-registered ones (ridge, y_score80).

Usage:
    python experiments/preair_value/sensitivity.py
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluate import RESULTS, SEED, compare, load
from features import ANILIST, ALT_FEATURES, value_at

HERE = Path(__file__).resolve().parent


def trend_rows() -> dict[str, list[tuple[int, float, float, bool]]]:
    rows: dict[str, list[tuple[int, float, float, bool]]] = defaultdict(list)
    with (ANILIST / "trends.csv").open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            airing = row["releasing"] == "True" or bool(row["episode"])
            rows[row["anilist_id"]].append(
                (int(row["days_from_premiere"]), float(row["popularity"] or 0), float(row["trending"] or 0), airing)
            )
    return rows


def rebuild_alt_features(rows: list[dict], trends: dict, cutoff: int) -> None:
    """Overwrite the AniList features in place, using only trend rows with day <= cutoff."""
    for row in rows:
        pre_air = [(day, pop) for day, pop, _, _ in trends[row["anilist_id"]] if day <= cutoff]
        row["log_pop_m1"] = math.log1p(value_at(pre_air, cutoff, cutoff))  # "latest pre-air" now means day `cutoff`
        row["log_pop_m30"] = math.log1p(value_at(pre_air, -30, cutoff))
        row["log_pop_m90"] = math.log1p(value_at(pre_air, -90, cutoff))
        row["growth_30"] = row["log_pop_m1"] - row["log_pop_m30"]
        window = [trend for day, _, trend, _ in trends[row["anilist_id"]] if cutoff - 6 <= day <= cutoff]
        row["log_trending_m7"] = math.log1p(sum(window) / len(window)) if window else 0.0
        row["not_tracked"] = float(not pre_air)
        # log_days_tracked is unchanged: the first tracked day is either well before the premiere or the title is
        # not tracked; titles first tracked between -7 and -1 count as tracked here, which favours no one.


def run(name: str, rows: list[dict], rng) -> dict:
    train = [row for row in rows if row["split"] == "train"]
    test = [row for row in rows if row["split"] == "test"]
    return {
        "H1": compare("ridge", train, test, "y_score80", rng),
        "H2": compare("ridge", train, [row for row in test if row["is_sequel"] == 0], "y_score80", rng),
    }


def main() -> None:
    trends = trend_rows()
    flagged = {aid for aid, rows in trends.items() if any(day <= -1 and airing for day, _, _, airing in rows)}
    rng = np.random.default_rng(SEED)

    cutoff_rows = load()
    rebuild_alt_features(cutoff_rows, trends, cutoff=-8)
    results = {
        "n_flagged_titles": len(flagged),
        "cutoff_-8": run("cutoff_-8", cutoff_rows, rng),
        "drop_flagged": run("drop_flagged", [row for row in load() if row["anilist_id"] not in flagged], rng),
    }

    print(f"POST-HOC sensitivity (not pre-registered); {len(flagged)} flagged titles; features {ALT_FEATURES}")
    print(f"{'analysis':<20}{'n_test':>7}{'M':>8}{'M+A':>8}{'A':>8}{'delta':>9}  95% CI")
    for name in ("cutoff_-8", "drop_flagged"):
        for test_name, r in results[name].items():
            print(f"{name + ' ' + test_name:<20}{r['n_test']:>7}{r['spearman_M']:>8.3f}{r['spearman_M+A']:>8.3f}"
                  f"{r['spearman_A']:>8.3f}{r['delta']:>+9.3f}  [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]")  # fmt: skip
    (RESULTS / "sensitivity.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved {RESULTS / 'sensitivity.json'}")


if __name__ == "__main__":
    main()
