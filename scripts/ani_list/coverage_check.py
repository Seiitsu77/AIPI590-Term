"""Check whether AniList trends cover the cold tail before we build features on them.

The first feasibility round failed partly because Wikipedia coverage collapsed on
the least popular titles (7% in the bottom fifth) and missingness tracked the
outcome. This script asks the same question of AniList trends: within each
season, split the titles that had trends fetched into popularity quintiles and
report how many have pre-air rows at all, rows 30 and 90 days before the
premiere, and a MAL ID.

It also cross-checks AniList's idMal against anime-offline-database, the two
independent sources for the MAL <-> AniList ID map.

This is a data-quality check only. Quintiles use today's popularity, so do not
read feature/label relationships off this output.

Usage:
    python scripts/ani_list/coverage_check.py
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANILIST = PROJECT_ROOT / "data" / "ani_list"
DEFAULT_OFFLINE_DB = PROJECT_ROOT / "data" / "anime_offline_database" / "anime.csv"
QUINTILES = ("Q1 coldest", "Q2", "Q3", "Q4", "Q5 hottest")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def pct(count: int, total: int) -> str:
    return f"{100 * count / total:.0f}%" if total else "-"


def median(values: list[float]) -> str:
    return f"{statistics.median(values):.0f}" if values else "-"


def assign_quintiles(rows: list[dict[str, str]]) -> dict[str, str]:
    """anilist_id -> quintile label, ranking by today's popularity within each season."""
    by_season: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_season[row["listed_season"]].append(row)
    labels = {}
    for season_rows in by_season.values():
        ranked = sorted(season_rows, key=lambda row: int(row["popularity"] or 0))
        for rank, row in enumerate(ranked):
            labels[row["anilist_id"]] = QUINTILES[rank * len(QUINTILES) // len(ranked)]
    return labels


def coverage_rows(anime: list[dict[str, str]], earliest_day: dict[str, int]) -> list[dict[str, Any]]:
    quintile = assign_quintiles(anime)
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in anime:
        groups[(row["listed_season"], quintile[row["anilist_id"]])].append(row)
        groups[("all", quintile[row["anilist_id"]])].append(row)
        groups[("all", "all")].append(row)

    report = []
    for (season, label), rows in sorted(groups.items()):
        n = len(rows)
        report.append(
            {
                "season": season,
                "quintile": label,
                "n": n,
                "has_mal_id": pct(sum(1 for row in rows if row["mal_id"]), n),
                "any_pre_air": pct(sum(1 for row in rows if int(row["trend_days_pre_air"] or 0) > 0), n),
                "row_by_day_-30": pct(sum(1 for row in rows if earliest_day.get(row["anilist_id"], 1) <= -30), n),
                "row_by_day_-90": pct(sum(1 for row in rows if earliest_day.get(row["anilist_id"], 1) <= -90), n),
                "median_days_tracked_pre_air": median(
                    [int(row["days_tracked_before_premiere"]) for row in rows if row["days_tracked_before_premiere"]]
                ),
                "median_missing_days": median(
                    [int(row["trend_missing_days"]) for row in rows if row["trend_missing_days"]]
                ),
                "has_popularity_1y": pct(sum(1 for row in rows if row["popularity_1y"]), n),
            }
        )
    return report


def id_crosscheck(anime: list[dict[str, str]], offline_db_path: Path) -> None:
    if not offline_db_path.exists():
        print(f"\n(skipping ID cross-check: {offline_db_path} not found)")
        return
    offline_mal: dict[str, set[str]] = defaultdict(set)
    for entry in read_csv(offline_db_path):
        for anilist_id in filter(None, entry["anilist_id"].split("|")):
            offline_mal[anilist_id].update(filter(None, entry["mal_id"].split("|")))

    outcome: dict[str, list[str]] = defaultdict(list)
    for row in anime:
        anilist_mal, other = row["mal_id"], offline_mal.get(row["anilist_id"])
        if other is None:
            key = "AniList title missing from offline-db"
        elif not anilist_mal and not other:
            key = "no MAL ID in either source"
        elif not anilist_mal:
            key = "only offline-db has a MAL ID"
        elif not other:
            key = "only AniList has a MAL ID"
        elif anilist_mal in other:
            key = "agree"
        else:
            key = "DISAGREE"
        outcome[key].append(row["anilist_id"])

    print("\nMAL ID cross-check, AniList idMal vs anime-offline-database:")
    for key, ids in sorted(outcome.items(), key=lambda item: -len(item[1])):
        sample = f"  e.g. {', '.join(ids[:5])}" if key != "agree" else ""
        print(f"  {key:<40} {len(ids):>5}{sample}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check AniList trend coverage by popularity quintile.")
    parser.add_argument("--data", type=Path, default=DEFAULT_ANILIST, help="AniList data directory.")
    parser.add_argument("--offline-db", type=Path, default=DEFAULT_OFFLINE_DB, help="anime-offline-database anime.csv.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    anime_path, trends_path = args.data / "anime.csv", args.data / "trends.csv"
    if not anime_path.exists() or not trends_path.exists():
        print("Run scripts/ani_list/build_tables.py first.", file=sys.stderr)
        return 1

    anime = [row for row in read_csv(anime_path) if row["trends_status"] != "not fetched"]
    earliest_day: dict[str, int] = {}
    for trend in read_csv(trends_path):
        day = int(trend["days_from_premiere"])
        earliest_day[trend["anilist_id"]] = min(day, earliest_day.get(trend["anilist_id"], day))

    report = coverage_rows(anime, earliest_day)
    columns = list(report[0])
    with (args.data / "coverage.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(report)

    print(f"Trend coverage for {len(anime)} titles, quintiles by today's popularity within each season\n")
    print("| " + " | ".join(columns) + " |")
    print("|" + "---|" * len(columns))
    for row in report:
        print("| " + " | ".join(str(row[column]) for column in columns) + " |")
    print(f"\nSaved {args.data / 'coverage.csv'}")

    id_crosscheck(anime, args.offline_db)
    return 0


if __name__ == "__main__":
    sys.exit(main())
