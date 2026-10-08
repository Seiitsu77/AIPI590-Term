"""Turn the raw AniList responses from download.py into flat CSV tables.

Offline: reads data/ani_list/raw/ and writes to data/ani_list/:

    anime.csv           one row per listed title (every format): metadata, today's
                        popularity and score/status distributions, trend summary
    trends.csv          one row per (title, day): the daily MediaTrend history from
                        365 days before to 120 days after the premiere
    relations.csv       one row per related entry (SOURCE, PREQUEL, SEQUEL, ...)
    external_links.csv  one row per external link (official site, Twitter, streaming, ...)

Dates are Japan dates (AniList stamps trend rows at 00:00 JST). days_from_premiere
is negative before the start date; day 0 is the premiere day itself, so strictly
pre-air features should use days_from_premiere <= -1.

AniList occasionally returns a day twice (identical rows); duplicates are dropped
here. It also skips days, mostly for low-activity titles (median 49 skipped days
in the coldest fifth of the 2021-2025 TV cohort, 1 in the hottest; no stored row
ever has trending 0). Skipped days are left out and counted in
anime.csv:trend_missing_days. popularity usually still changes across a gap, so
interpolate between the neighbouring rows; forward-filling only gives a lower
bound.

A title with no rows before day 0 was not tracked by AniList before it aired
(13 of 926 cohort titles; kids' and niche shows in the pilot). That absence is informative: keep such
titles and flag them rather than dropping them.

Everything in anime.csv except the trend summary is the snapshot at
listing_fetched_at, i.e. post-air. relations.csv mixes ANIME and MANGA targets.

Usage:
    python scripts/ani_list/build_tables.py
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PROJECT_ROOT / "data" / "ani_list"
SEASONS = ("winter", "spring", "summer", "fall")
JST = timezone(timedelta(hours=9))
SCORE_BUCKETS = range(10, 101, 10)
STATUSES = ("CURRENT", "PLANNING", "COMPLETED", "DROPPED", "PAUSED")

ANIME_COLUMNS = [
    "anilist_id", "mal_id", "title_romaji", "title_english", "title_native",
    "format", "status", "source", "country", "episodes", "duration_min",
    "season", "season_year", "listed_season", "start_date", "end_date",
    "genres", "tags", "main_studios", "other_studios", "staff",
    "trailer_site", "trailer_id", "hashtag",
    "popularity", "favourites", "average_score", "mean_score",
    *[f"status_{status.lower()}" for status in STATUSES],
    *[f"score_{bucket}" for bucket in SCORE_BUCKETS], "score_80plus",
    "listing_fetched_at",
    "trends_status", "trend_days", "trend_days_pre_air", "trend_missing_days",
    "first_tracked_date", "days_tracked_before_premiere", "popularity_1y_date", "popularity_1y",
    "description",
]  # fmt: skip
TREND_COLUMNS = [
    "anilist_id", "date", "days_from_premiere", "popularity", "trending", "in_progress", "episode", "releasing",
    "average_score",
]  # fmt: skip
RELATION_COLUMNS = [
    "anilist_id", "relation_type", "target_anilist_id", "target_mal_id", "target_type", "target_format",
    "target_title", "target_start_date",
]  # fmt: skip
EXTERNAL_COLUMNS = ["anilist_id", "site", "type", "language", "url"]


def fuzzy_date(value: dict[str, Any] | None) -> str:
    """AniList FuzzyDate -> 'YYYY-MM-DD', 'YYYY-MM' or 'YYYY' depending on what is known."""
    value = value or {}
    parts = [value.get("year"), value.get("month"), value.get("day")]
    known = []
    for part, width in zip(parts, (4, 2, 2)):
        if not part:
            break
        known.append(f"{part:0{width}d}")
    return "-".join(known)


def full_date(value: dict[str, Any] | None) -> date | None:
    value = value or {}
    if value.get("year") and value.get("month") and value.get("day"):
        return date(value["year"], value["month"], value["day"])
    return None


def jst_date(timestamp: int) -> date:
    return datetime.fromtimestamp(timestamp, JST).date()


def clean_description(text: str | None) -> str:
    if not text:
        return ""
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text).strip()


def load_listings(raw_dir: Path) -> list[tuple[str, str, dict[str, Any]]]:
    """(listed_season, fetched_at, media) for every listed title, earliest season first, deduplicated."""
    folders = []
    for folder in (raw_dir / "seasons").glob("*-*"):
        year, _, season = folder.name.partition("-")
        if year.isdigit() and season in SEASONS:
            folders.append((int(year), SEASONS.index(season), folder))

    seen: set[int] = set()
    listed = []
    for _, _, folder in sorted(folders):
        for page in sorted(folder.glob("page-*.json")):
            record = json.loads(page.read_text(encoding="utf-8"))
            for media in record["response"]["data"]["Page"]["media"]:
                if media["id"] not in seen:
                    seen.add(media["id"])
                    listed.append((folder.name, record["fetched_at"], media))
    return listed


def media_row(media: dict[str, Any], listed_season: str, fetched_at: str) -> dict[str, Any]:
    title = media.get("title") or {}
    studios = (media.get("studios") or {}).get("edges") or []
    staff = (media.get("staff") or {}).get("edges") or []
    trailer = media.get("trailer") or {}
    stats = media.get("stats") or {}
    scores = {item["score"]: item["amount"] for item in stats.get("scoreDistribution") or []}
    statuses = {item["status"]: item["amount"] for item in stats.get("statusDistribution") or []}

    row = {
        "anilist_id": media["id"],
        "mal_id": media.get("idMal"),
        "title_romaji": title.get("romaji"),
        "title_english": title.get("english"),
        "title_native": title.get("native"),
        "format": media.get("format"),
        "status": media.get("status"),
        "source": media.get("source"),
        "country": media.get("countryOfOrigin"),
        "episodes": media.get("episodes"),
        "duration_min": media.get("duration"),
        "season": media.get("season"),
        "season_year": media.get("seasonYear"),
        "listed_season": listed_season,
        "start_date": fuzzy_date(media.get("startDate")),
        "end_date": fuzzy_date(media.get("endDate")),
        "genres": "|".join(media.get("genres") or []),
        "tags": "|".join(f"{tag['name']}:{tag['rank']}" for tag in media.get("tags") or []),
        "main_studios": "|".join(edge["node"]["name"] for edge in studios if edge.get("isMain")),
        "other_studios": "|".join(edge["node"]["name"] for edge in studios if not edge.get("isMain")),
        "staff": "|".join(f"{edge['role']}: {edge['node']['name']['full']}" for edge in staff),
        "trailer_site": trailer.get("site"),
        "trailer_id": trailer.get("id"),
        "hashtag": media.get("hashtag"),
        "popularity": media.get("popularity"),
        "favourites": media.get("favourites"),
        "average_score": media.get("averageScore"),
        "mean_score": media.get("meanScore"),
        "score_80plus": sum(scores.get(bucket, 0) for bucket in (80, 90, 100)),
        "listing_fetched_at": fetched_at,
        "trends_status": "not fetched",
        "description": clean_description(media.get("description")),
    }
    for status in STATUSES:
        row[f"status_{status.lower()}"] = statuses.get(status, 0)
    for bucket in SCORE_BUCKETS:
        row[f"score_{bucket}"] = scores.get(bucket, 0)
    return row


def trend_rows(anilist_id: int, premiere: date, data: dict[str, Any]) -> list[dict[str, Any]]:
    by_day: dict[date, dict[str, Any]] = {}
    for alias, page in data.items():
        if not re.fullmatch(r"p\d+", alias):
            continue
        for trend in page["mediaTrends"]:
            day = jst_date(trend["date"])
            by_day[day] = {
                "anilist_id": anilist_id,
                "date": day.isoformat(),
                "days_from_premiere": (day - premiere).days,
                "popularity": trend.get("popularity"),
                "trending": trend.get("trending"),
                "in_progress": trend.get("inProgress"),
                "episode": trend.get("episode"),
                "releasing": trend.get("releasing"),
                "average_score": trend.get("averageScore"),
            }
    return [by_day[day] for day in sorted(by_day)]


def add_trend_summary(row: dict[str, Any], premiere: date, record: dict[str, Any], rows: list[dict]) -> None:
    if record["status"] != 200:
        row["trends_status"] = f"HTTP {record['status']}"
        return
    row["trends_status"] = "ok"
    row["trend_days"] = len(rows)
    row["trend_days_pre_air"] = sum(1 for trend in rows if trend["days_from_premiere"] <= -1)
    if rows:
        span = rows[-1]["days_from_premiere"] - rows[0]["days_from_premiere"] + 1
        row["trend_missing_days"] = span - len(rows)

    data = record["response"]["data"]
    first = data["first"]["mediaTrends"]
    if first:
        first_day = jst_date(first[0]["date"])
        row["first_tracked_date"] = first_day.isoformat()
        row["days_tracked_before_premiere"] = (premiere - first_day).days
    year1 = data["year1"]["mediaTrends"]
    if year1:
        row["popularity_1y_date"] = jst_date(year1[0]["date"]).isoformat()
        row["popularity_1y"] = year1[0].get("popularity")


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CSV tables from raw AniList responses.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="AniList data directory (default: data/ani_list).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_dir = args.out / "raw"
    listings = load_listings(raw_dir)
    if not listings:
        print(f"No season listings under {raw_dir}. Run scripts/ani_list/download.py first.", file=sys.stderr)
        return 1

    anime_rows, all_trends, relation_rows, external_rows = [], [], [], []
    for listed_season, fetched_at, media in listings:
        row = media_row(media, listed_season, fetched_at)
        premiere = full_date(media.get("startDate"))
        trend_path = raw_dir / "trends" / f"{media['id']}.json"
        if premiere and trend_path.exists():
            record = json.loads(trend_path.read_text(encoding="utf-8"))
            rows = trend_rows(media["id"], premiere, record["response"]["data"]) if record["status"] == 200 else []
            add_trend_summary(row, premiere, record, rows)
            all_trends.extend(rows)
        anime_rows.append(row)

        for edge in (media.get("relations") or {}).get("edges") or []:
            node = edge["node"]
            relation_rows.append(
                {
                    "anilist_id": media["id"],
                    "relation_type": edge.get("relationType"),
                    "target_anilist_id": node.get("id"),
                    "target_mal_id": node.get("idMal"),
                    "target_type": node.get("type"),
                    "target_format": node.get("format"),
                    "target_title": (node.get("title") or {}).get("romaji"),
                    "target_start_date": fuzzy_date(node.get("startDate")),
                }
            )
        for link in media.get("externalLinks") or []:
            external_rows.append(
                {
                    "anilist_id": media["id"],
                    "site": link.get("site"),
                    "type": link.get("type"),
                    "language": link.get("language"),
                    "url": link.get("url"),
                }
            )

    write_csv(args.out / "anime.csv", ANIME_COLUMNS, anime_rows)
    write_csv(args.out / "trends.csv", TREND_COLUMNS, all_trends)
    write_csv(args.out / "relations.csv", RELATION_COLUMNS, relation_rows)
    write_csv(args.out / "external_links.csv", EXTERNAL_COLUMNS, external_rows)

    with_trends = sum(1 for row in anime_rows if row["trends_status"] == "ok")
    print(f"anime.csv           {len(anime_rows)} rows ({with_trends} with trends)")
    print(f"trends.csv          {len(all_trends)} rows")
    print(f"relations.csv       {len(relation_rows)} rows")
    print(f"external_links.csv  {len(external_rows)} rows")
    failed = [str(row["anilist_id"]) for row in anime_rows if row["trends_status"].startswith("HTTP")]
    if failed:
        print(f"trends failed for {len(failed)} titles: {', '.join(failed[:10])}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
