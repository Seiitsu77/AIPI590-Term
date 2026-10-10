"""How well does the Wikipedia mapping cover the cohort, compared with the first feasibility round?

First round (2016-2020 cohort, own-article mapping): 67% of titles mapped; only 7%
of the coldest fifth had pre-air views; 36 of the 215 mapped titles in the hottest
fifth had zero pre-air views because they were mapped to articles split off after
the premiere.

For the 2021-2025 history cohort, split into within-season quintiles of AniList's
current popularity (a data-quality view only, not a feature/label analysis):

    item          share with a Wikidata item for their MAL id
    article       share mapped to at least one en/ja/zh article (any role)
    own_article   share mapped through the anime's own item or its series only
    pre_air       share with any views on days -30..-1, all mapped articles
    pre_air_own   the same, counting only own/series articles (the first round's mapping)

and, among mapped titles in the hottest fifth, how many have zero pre-air views.
The live seasons (2026-fall, 2027-winter) are summarised separately.

Usage:
    python scripts/wikipedia/coverage_check.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WIKIPEDIA = PROJECT_ROOT / "data" / "wikipedia"
ANILIST = {"history": PROJECT_ROOT / "data" / "ani_list" / "anime.csv",
           "live": PROJECT_ROOT / "data" / "ani_list_live" / "anime.csv"}  # fmt: skip
QUINTILES = ("Q1 coldest", "Q2", "Q3", "Q4", "Q5 hottest")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def main() -> None:
    titles = {row["anilist_id"]: row for row in read_csv(WIKIPEDIA / "titles.csv")}
    roles: dict[str, set[str]] = defaultdict(set)
    for row in read_csv(WIKIPEDIA / "mapping.csv"):
        roles[row["anilist_id"]].add(row["role"])
    own_views: dict[str, int] = defaultdict(int)
    for row in read_csv(WIKIPEDIA / "title_daily.csv"):
        if -30 <= int(row["days_from_premiere"]) <= -1:
            own_views[row["anilist_id"]] += sum(int(row[f"{wiki}_anime"]) for wiki in ("en", "ja", "zh"))

    def stats(rows: list[dict[str, str]]) -> dict[str, float]:
        n = len(rows)
        ids = [row["anilist_id"] for row in rows]
        has = lambda test: sum(1 for i in ids if test(i)) / n  # noqa: E731
        return {
            "n": n,
            "item": has(lambda i: i in titles and titles[i]["wikidata_item"]),
            "article": has(lambda i: i in titles and int(titles[i]["n_articles"] or 0) > 0),
            "own_article": has(lambda i: bool(roles[i] & {"self", "series"})),
            "pre_air": has(lambda i: i in titles and int(titles[i]["preair_views_m30"] or 0) > 0),
            "pre_air_own": has(lambda i: own_views[i] > 0),
        }

    def show(label: str, s: dict[str, float]) -> None:
        print(f"   {label:<12}{s['n']:>5}" + "".join(f"{s[k]:>13.0%}" for k in ("item", "article", "own_article",
                                                                               "pre_air", "pre_air_own")))  # fmt: skip

    header = f"   {'':<12}{'n':>5}" + "".join(f"{k:>13}" for k in ("item", "article", "own_article", "pre_air", "pre_air_own"))

    history = [row for row in read_csv(ANILIST["history"]) if row["trends_status"] == "ok" and row["format"] == "TV"]
    by_season: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in history:
        by_season[row["listed_season"]].append(row)
    quintile_rows: dict[int, list[dict[str, str]]] = defaultdict(list)
    for rows in by_season.values():
        ranked = sorted(rows, key=lambda row: int(row["popularity"] or 0))
        for rank, row in enumerate(ranked):
            quintile_rows[rank * 5 // len(ranked)].append(row)

    print("History cohort (2021-2025 TV), within-season quintiles of AniList's current popularity")
    print(header)
    for q in range(5):
        show(QUINTILES[q], stats(quintile_rows[q]))
    show("all", stats(history))

    hottest = [row for row in quintile_rows[4] if row["anilist_id"] in titles and int(titles[row["anilist_id"]]["n_articles"] or 0)]
    zero_all = [row for row in hottest if int(titles[row["anilist_id"]]["preair_views_m30"] or 0) == 0]
    zero_own = [row for row in hottest if own_views[row["anilist_id"]] == 0]
    print(f"\n   Hottest fifth, mapped titles with zero views on days -30..-1: {len(zero_all)}/{len(hottest)} "
          f"(own/series articles only: {len(zero_own)}/{len(hottest)}; first round: 36/215)")  # fmt: skip
    for row in zero_all[:8]:
        print(f"      {row['anilist_id']:>7} {row['title_romaji'][:50]}")

    live = [row for row in read_csv(ANILIST["live"]) if row["trends_status"] == "ok" and row["format"] == "TV"]
    print("\nLive seasons")
    print(header)
    for season in ("2026-fall", "2027-winter"):
        show(season, stats([row for row in live if row["listed_season"] == season]))


if __name__ == "__main__":
    main()
