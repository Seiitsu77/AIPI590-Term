"""Audit the two candidate labels before choosing one. Touches no predictive feature.

Candidates for "how many people ended up liking this title":

    mal_members   MAL list members, from the Anime Compass catalog. Its 2021+ rows
                  come from a third-party "Top 10K" snapshot of unknown date.
    score_80plus  AniList users who scored the title 80/100 or higher, snapshot
                  taken 2026-10-07 by scripts/ani_list/download.py.

Questions, all answered without looking at any pre-air feature:

    1. Coverage: which cohort titles have the label at all?
    2. Selection: is missingness concentrated among unpopular titles?
       (Popularity here is AniList's *current* list count, not a pre-air feature.)
    3. Snapshot timing: does the label look immature for the latest seasons?
       Measured as the label divided by AniList's current popularity, per season.
    4. Agreement: within-season Spearman between the label and AniList's current
       popularity.

Usage:
    python experiments/preair_value/label_audit.py
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANILIST_ANIME = PROJECT_ROOT / "data" / "ani_list" / "anime.csv"
CATALOG = PROJECT_ROOT / "data" / "anime_compass" / "anime_catalog_serving.json"
SEASONS = ("winter", "spring", "summer", "fall")


def season_key(listed_season: str) -> tuple[int, int]:
    year, _, season = listed_season.partition("-")
    return int(year), SEASONS.index(season)


def main() -> None:
    cohort = [row for row in csv.DictReader(ANILIST_ANIME.open(encoding="utf-8")) if row["trends_status"] == "ok"]
    catalog = {int(item["id"]): item for item in json.loads(CATALOG.read_text(encoding="utf-8"))}

    for row in cohort:
        row["popularity_now"] = int(row["popularity"] or 0)
        item = catalog.get(int(row["mal_id"])) if row["mal_id"] else None
        row["mal_members"] = int(item["members"]) if item and item.get("members") else None
        row["score_80plus"] = int(row["score_80plus"] or 0)

    print(f"Cohort: {len(cohort)} AniList TV titles, 2021-winter .. 2025-fall\n")

    print("1-2. Coverage by within-season quintile of AniList's current popularity")
    by_season: dict[str, list[dict]] = defaultdict(list)
    for row in cohort:
        by_season[row["listed_season"]].append(row)
    quintile_rows: dict[int, list[dict]] = defaultdict(list)
    for rows in by_season.values():
        ranked = sorted(rows, key=lambda row: row["popularity_now"])
        for rank, row in enumerate(ranked):
            quintile_rows[rank * 5 // len(ranked)].append(row)
    print(f"   {'quintile':<12}{'n':>5}{'mal_members':>14}{'score_80plus>0':>17}")
    for quintile in range(5):
        rows = quintile_rows[quintile]
        mal = sum(row["mal_members"] is not None for row in rows)
        anilist = sum(row["score_80plus"] > 0 for row in rows)
        label = ["Q1 coldest", "Q2", "Q3", "Q4", "Q5 hottest"][quintile]
        print(f"   {label:<12}{len(rows):>5}{mal / len(rows):>14.0%}{anilist / len(rows):>17.0%}")
    total_mal = sum(row["mal_members"] is not None for row in cohort)
    total_anilist = sum(row["score_80plus"] > 0 for row in cohort)
    print(f"   {'all':<12}{len(cohort):>5}{total_mal / len(cohort):>14.0%}{total_anilist / len(cohort):>17.0%}")

    print("\n3-4. Per season: label / AniList current popularity (median), and within-season Spearman with it")
    print(f"   {'season':<13}{'n':>4}{'n_mal':>7}{'mal/pop':>9}{'rho_mal':>9}{'80plus/pop':>12}{'rho_80plus':>12}")
    for season in sorted(by_season, key=season_key):
        rows = by_season[season]
        with_mal = [row for row in rows if row["mal_members"] is not None and row["popularity_now"] > 0]
        ratio_mal = statistics.median(row["mal_members"] / row["popularity_now"] for row in with_mal)
        rho_mal = spearmanr([r["mal_members"] for r in with_mal], [r["popularity_now"] for r in with_mal])[0]
        positive = [row for row in rows if row["popularity_now"] > 0]
        ratio_80 = statistics.median(row["score_80plus"] / row["popularity_now"] for row in positive)
        rho_80 = spearmanr([r["score_80plus"] for r in positive], [r["popularity_now"] for r in positive])[0]
        print(
            f"   {season:<13}{len(rows):>4}{len(with_mal):>7}{ratio_mal:>9.2f}{rho_mal:>9.3f}"
            f"{ratio_80:>12.3f}{rho_80:>12.3f}"
        )

    missing = [row for row in cohort if row["mal_members"] is None]
    print(f"\nTitles without mal_members: {len(missing)}")
    print("   median AniList current popularity, missing vs present:",
          statistics.median(r["popularity_now"] for r in missing),
          "vs", statistics.median(r["popularity_now"] for r in cohort if r["mal_members"] is not None))
    print("   by season:", dict(sorted(((s, sum(r["mal_members"] is None for r in rows)) for s, rows in by_season.items()),
                                      key=lambda item: season_key(item[0]))))


if __name__ == "__main__":
    main()
