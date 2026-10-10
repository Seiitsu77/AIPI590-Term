"""Turn the raw Wikipedia downloads into per-title tables.

Writes to data/wikipedia/:

    mapping.csv      one row per (title, wiki, article): which Wikipedia article a cohort
                     title is mapped to and why (role), plus its number of redirects
    titles.csv       one row per cohort title: Wikidata item, mapped articles per
                     language, and pre-air views summed over days -30..-1 and -90..-1
    title_daily.csv  one row per (title, day) from premiere - 365 to premiere + 120 (or
                     yesterday): views per language, split into the anime's own
                     articles (self, series) and its source/franchise articles

Roles, in priority order when one article is reached several ways:
    self       the Wikidata item that carries the title's MAL id
    series     its P179 "part of the series" (usually the TV series article)
    based_on   its or its series' P144 "based on" (usually the manga / light novel)
    franchise  its or its series' P8345 "media franchise"

Views of an article include the views logged under every redirect to it, which is
where views of a since-renamed title end up. Each page is counted once per title,
even when it is reached both as a mapped article and as a redirect to another one.
Days with no data count as 0.

For live titles announced only to the month, days_from_premiere counts from the 1st
of that month (titles.csv: premiere_provisional = 1).

Run without python -I: it imports the cohort loader from download.py next to it.

Usage:
    python scripts/wikipedia/build_tables.py
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

from download import DEFAULT_OUT, PAGEVIEWS_START, POST_DAYS, PRE_DAYS, ROLES, WIKIS, load_cohort, title_key

ROLE_PRIORITY = ("self", "series", "based_on", "franchise")
GROUP = {"self": "anime", "series": "anime", "based_on": "source", "franchise": "source"}
VIEW_COLUMNS = [f"{wiki}_{group}" for wiki in WIKIS for group in ("anime", "source")]


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Wikipedia tables from the raw downloads.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Wikipedia data directory.")
    out = parser.parse_args().out
    raw = out / "raw"

    cohort = load_cohort()
    sparql_rows = [
        {key: value["value"].rsplit("/", 1)[-1] for key, value in binding.items()}
        for path in sorted((raw / "sparql").glob("*.json"))
        for binding in json.loads(path.read_text(encoding="utf-8"))["response"]["results"]["bindings"]
    ]
    sitelinks: dict[str, dict[str, str]] = {}
    for path in (raw / "entities").glob("*.json"):
        for item, entity in json.loads(path.read_text(encoding="utf-8"))["response"]["entities"].items():
            links = entity.get("sitelinks", {})
            sitelinks[item] = {key[:-4]: value["title"] for key, value in links.items() if key[:-4] in WIKIS}
    redirects: dict[tuple[str, str], list[str]] = {}
    for path in (raw / "redirects").glob("*/*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        redirects[(record["wiki"], record["title"])] = record["redirects"]
    views: dict[tuple[str, str], dict[str, int]] = {}
    for path in (raw / "pageviews").glob("*/*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        views[(record["wiki"], record["title"])] = {item["date"]: item["views"] for item in record["items"]}

    # (mal id) -> {(wiki, article): (role, wikidata item)}, keeping the highest-priority role.
    articles_by_mal: dict[str, dict[tuple[str, str], tuple[str, str]]] = defaultdict(dict)
    self_item: dict[str, str] = {}
    for row in sparql_rows:
        self_item[row["mal"]] = row["item"]
        for var, role in ROLES.items():
            if var not in row:
                continue
            for wiki, article in sitelinks.get(row[var], {}).items():
                current = articles_by_mal[row["mal"]].get((wiki, article))
                if current is None or ROLE_PRIORITY.index(role) < ROLE_PRIORITY.index(current[0]):
                    articles_by_mal[row["mal"]][(wiki, article)] = (role, row[var])

    def pages_for(mapped: dict[tuple[str, str], tuple[str, str]]) -> dict[tuple[str, str], str]:
        """Every page whose views count for a title -> view column, each page once.

        A Wikidata sitelink can point at a page that is now a redirect to another mapped
        article (e.g. ja "鬼滅の刃 遊郭編" -> "鬼滅の刃 (アニメ)"), so the same page can be
        reached as an article and as a redirect; it is assigned to the higher-priority role.
        """
        pages: dict[tuple[str, str], str] = {}
        by_priority = sorted(mapped.items(), key=lambda item: ROLE_PRIORITY.index(item[1][0]))
        for (wiki, article), (role, _) in by_priority:
            for page in [article, *redirects.get((wiki, article), [])]:
                pages.setdefault((wiki, page), f"{wiki}_{GROUP[role]}")
        return pages

    yesterday = date.today() - timedelta(days=1)
    mapping_rows, title_rows, daily_rows = [], [], []
    for title in cohort:
        mapped = articles_by_mal.get(title["mal_id"], {})
        for (wiki, article), (role, item) in sorted(mapped.items()):
            mapping_rows.append({"anilist_id": title["anilist_id"], "mal_id": title["mal_id"], "cohort": title["cohort"],
                                 "role": role, "wikidata_item": item, "wiki": wiki, "article": article,
                                 "n_redirects": len(redirects.get((wiki, article), []))})  # fmt: skip

        premiere = title["premiere"]
        start = max(premiere - timedelta(days=PRE_DAYS), PAGEVIEWS_START)
        end = min(premiere + timedelta(days=POST_DAYS), yesterday)
        series = [(column, views.get(page, {})) for page, column in pages_for(mapped).items()]
        pre_air = {30: 0, 90: 0}
        day = start
        while mapped and day <= end:
            iso = day.isoformat()
            row = {"anilist_id": title["anilist_id"], "date": iso, "days_from_premiere": (day - premiere).days,
                   **{column: 0 for column in VIEW_COLUMNS}}  # fmt: skip
            for column, by_day in series:
                row[column] += by_day.get(iso, 0)
            row["total"] = sum(row[column] for column in VIEW_COLUMNS)
            for window in pre_air:
                if -window <= row["days_from_premiere"] <= -1:
                    pre_air[window] += row["total"]
            daily_rows.append(row)
            day += timedelta(days=1)

        title_rows.append({
            "anilist_id": title["anilist_id"], "mal_id": title["mal_id"], "cohort": title["cohort"],
            "premiere": premiere.isoformat(), "wikidata_item": self_item.get(title["mal_id"], ""),
            **{f"articles_{wiki}": "|".join(sorted(a for w, a in mapped if w == wiki)) for wiki in WIKIS},
            "n_articles": len(mapped), "preair_views_m30": pre_air[30] if mapped else "",
            "preair_views_m90": pre_air[90] if mapped else "",
        })  # fmt: skip

    write_csv(out / "mapping.csv", ["anilist_id", "mal_id", "cohort", "role", "wikidata_item", "wiki", "article",
                                    "n_redirects"], mapping_rows)  # fmt: skip
    write_csv(out / "titles.csv", list(title_rows[0]), title_rows)
    write_csv(out / "title_daily.csv", ["anilist_id", "date", "days_from_premiere", *VIEW_COLUMNS, "total"], daily_rows)
    print(f"mapping.csv      {len(mapping_rows)} rows")
    print(f"titles.csv       {len(title_rows)} rows ({sum(1 for r in title_rows if r['n_articles'])} with an article)")
    print(f"title_daily.csv  {len(daily_rows)} rows")


if __name__ == "__main__":
    main()
