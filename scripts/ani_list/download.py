"""Download a seasonal anime cohort and its daily popularity history from AniList.

Step 1 lists every title AniList files under each season (all formats, adult
titles excluded) with full metadata: idMal, studios, staff, source, tags,
relations, external links, trailer, score and status distributions.

Step 2 fetches the daily trend history (MediaTrend) of every listed title whose
format is in --formats and whose start date is known. One request per title
covers 365 days before the premiere to 120 days after it, plus the earliest
tracked day and the first day one year after the premiere.

Trend rows are recorded by AniList on the day itself and are never rewritten, so
the pre-premiere rows are genuine pre-air data: `popularity` is how many users had
the title on a list that day. Everything in step 1 (popularity, scores,
statusDistribution) is today's snapshot, i.e. post-air for these cohorts.

Responses are saved under data/ani_list/raw/ as received. Files already on disk
are skipped, so an interrupted run resumes where it stopped. Run
scripts/ani_list/build_tables.py afterwards for CSV tables.

Usage:
    python scripts/ani_list/download.py --seasons 2021-winter 2024-fall   # pilot
    python scripts/ani_list/download.py                                   # 2021-winter .. 2025-fall
    python scripts/ani_list/download.py --formats TV ONA --limit 5
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PROJECT_ROOT / "data" / "ani_list"
GRAPHQL_URL = "https://graphql.anilist.co"
SEASONS = ("winter", "spring", "summer", "fall")
USER_AGENT = "AnimeRec/0.1 (Duke AIPI 590 course project)"
JST = timezone(timedelta(hours=9))  # AniList dates trend rows at 00:00 JST
DAY = 86400

# AniList currently allows 30 requests/min (see the X-RateLimit-Limit header).
MIN_INTERVAL_S = 2.1
MAX_ATTEMPTS = 5

PRE_DAYS = 365
POST_DAYS = 120
TREND_PAGES = 10  # 10 pages x 50 rows cover the 486-day window, all as aliases in one request

MEDIA_FIELDS = """
id idMal
title { romaji english native }
synonyms
format status episodes duration source(version: 3) countryOfOrigin isAdult
season seasonYear
startDate { year month day }
endDate { year month day }
genres
tags { name rank category isMediaSpoiler isGeneralSpoiler }
studios { edges { isMain node { id name isAnimationStudio } } }
staff(perPage: 10, sort: [RELEVANCE, ID]) { edges { role node { id name { full native } } } }
relations { edges { relationType(version: 2) node { id idMal type format title { romaji } startDate { year month day } } } }
externalLinks { site type language url }
trailer { site id }
hashtag
popularity favourites averageScore meanScore
stats { scoreDistribution { score amount } statusDistribution { status amount } }
description(asHtml: false)
"""

LISTING_QUERY = (
    """
query ($season: MediaSeason, $year: Int, $page: Int) {
  Page(page: $page, perPage: 50) {
    pageInfo { hasNextPage }
    media(season: $season, seasonYear: $year, type: ANIME, isAdult: false, sort: ID) { %s }
  }
}
"""
    % MEDIA_FIELDS
)

TREND_FIELDS = "date popularity trending inProgress episode releasing averageScore"
TRENDS_QUERY = (
    "query ($id: Int, $lo: Int, $hi: Int, $year1: Int) {\n"
    + "\n".join(
        f"  p{page}: Page(page: {page}, perPage: 50) {{ pageInfo {{ hasNextPage }} "
        f"mediaTrends(mediaId: $id, date_greater: $lo, date_lesser: $hi, sort: DATE) {{ {TREND_FIELDS} }} }}"
        for page in range(1, TREND_PAGES + 1)
    )
    + f"\n  first: Page(perPage: 1) {{ mediaTrends(mediaId: $id, sort: DATE) {{ {TREND_FIELDS} }} }}"
    + f"\n  year1: Page(perPage: 1) {{ mediaTrends(mediaId: $id, date_greater: $year1, sort: DATE) {{ {TREND_FIELDS} }} }}"
    + "\n}"
)


class AniListUnavailable(RuntimeError):
    pass


class AniListClient:
    def __init__(self, min_interval_s: float = MIN_INTERVAL_S) -> None:
        self.min_interval_s = min_interval_s
        self.requests_made = 0
        self._last_request = 0.0

    def query(self, query: str, variables: dict[str, Any]) -> tuple[int, Any]:
        """Return (status, body) for 200 and 404; retry 429/5xx/network errors; raise on anything else."""
        payload = json.dumps({"query": query, "variables": variables}).encode("utf-8")
        headers = {"User-Agent": USER_AGENT, "Content-Type": "application/json", "Accept": "application/json"}

        for attempt in range(1, MAX_ATTEMPTS + 1):
            self._throttle()
            retry_after = 0.0
            request = urllib.request.Request(GRAPHQL_URL, data=payload, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    body = json.loads(response.read().decode("utf-8"))
                if not body.get("errors"):
                    return 200, body
                reason = f"GraphQL errors: {body['errors']}"
            except urllib.error.HTTPError as exc:
                body = _error_body(exc)
                if exc.code == 404:
                    return 404, body
                if exc.code == 400:
                    raise AniListUnavailable(f"AniList rejected the query (HTTP 400): {body}") from None
                reason = f"HTTP {exc.code}"
                retry_after = _retry_after_seconds(exc)
            except (urllib.error.URLError, socket.timeout, ConnectionError) as exc:
                reason = f"{type(exc).__name__}: {getattr(exc, 'reason', exc)}"

            if attempt == MAX_ATTEMPTS:
                raise AniListUnavailable(f"request failed {MAX_ATTEMPTS} times, last error: {reason}")
            backoff = max(2.0**attempt, retry_after)
            print(f"    {reason}; retry {attempt}/{MAX_ATTEMPTS - 1} in {backoff:.0f}s", file=sys.stderr)
            time.sleep(backoff)
        raise AssertionError("unreachable")

    def _throttle(self) -> None:
        wait = self.min_interval_s - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()
        self.requests_made += 1


def _error_body(exc: urllib.error.HTTPError) -> Any:
    raw = exc.read().decode("utf-8", errors="replace")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"message": raw[:500]}


def _retry_after_seconds(exc: urllib.error.HTTPError) -> float:
    try:
        return float(exc.headers.get("Retry-After", 0))
    except (TypeError, ValueError):
        return 0.0


def save_record(path: Path, variables: dict[str, Any], status: int, body: Any) -> None:
    record = {
        "url": GRAPHQL_URL,
        "variables": variables,
        "status": status,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "response": body,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)  # atomic, so an interrupted run never leaves half a file


def parse_season(value: str) -> tuple[int, str]:
    year, _, season = value.partition("-")
    if not year.isdigit() or season not in SEASONS:
        raise argparse.ArgumentTypeError(f"expected YEAR-SEASON like 2021-winter, got {value!r}")
    return int(year), season


def season_range(start: tuple[int, str], end: tuple[int, str]) -> list[tuple[int, str]]:
    first = start[0] * 4 + SEASONS.index(start[1])
    last = end[0] * 4 + SEASONS.index(end[1])
    if first > last:
        raise SystemExit(f"--start {start} is after --end {end}")
    return [(index // 4, SEASONS[index % 4]) for index in range(first, last + 1)]


def premiere_timestamp(media: dict[str, Any]) -> int | None:
    """00:00 JST on the start date, the same clock AniList stamps trend rows with."""
    start = media.get("startDate") or {}
    if not (start.get("year") and start.get("month") and start.get("day")):
        return None
    return int(datetime(start["year"], start["month"], start["day"], tzinfo=JST).timestamp())


def fetch_season_listing(client: AniListClient, raw_dir: Path, year: int, season: str) -> list[dict[str, Any]]:
    media: list[dict[str, Any]] = []
    page = 1
    while True:
        path = raw_dir / "seasons" / f"{year}-{season}" / f"page-{page:02d}.json"
        if path.exists():
            body = json.loads(path.read_text(encoding="utf-8"))["response"]
        else:
            variables = {"season": season.upper(), "year": year, "page": page}
            status, body = client.query(LISTING_QUERY, variables)
            if status != 200:
                raise AniListUnavailable(f"season listing {year}-{season} page {page} returned HTTP {status}: {body}")
            save_record(path, variables, status, body)
        listing = body["data"]["Page"]
        media.extend(listing["media"])
        if not listing["pageInfo"]["hasNextPage"]:
            return media
        page += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a seasonal anime cohort and daily trends from AniList.")
    parser.add_argument("--seasons", nargs="+", type=parse_season, help="Explicit seasons, e.g. 2021-winter 2024-fall.")
    parser.add_argument("--start", type=parse_season, default=(2021, "winter"), help="First season (default: 2021-winter).")
    parser.add_argument("--end", type=parse_season, default=(2025, "fall"), help="Last season (default: 2025-fall).")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["TV"],
        help="AniList formats to fetch trends for, e.g. TV TV_SHORT ONA MOVIE (default: TV). Listings keep every format.",
    )
    parser.add_argument("--limit", type=int, help="Fetch trends for at most this many titles (for a smoke test).")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory (default: data/ani_list).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # keep progress and retry lines in order when piped to a log
    raw_dir = args.out / "raw"
    client = AniListClient()
    formats = set(args.formats)
    seasons = args.seasons or season_range(args.start, args.end)

    try:
        todo: list[dict[str, Any]] = []
        no_start_date = 0
        for year, season in seasons:
            media = fetch_season_listing(client, raw_dir, year, season)
            selected = [item for item in media if item.get("format") in formats]
            dated = [item for item in selected if premiere_timestamp(item) is not None]
            no_start_date += len(selected) - len(dated)
            todo.extend(dated)
            print(f"{year}-{season}: {len(media)} listed, {len(dated)} {'/'.join(args.formats)} with a start date")
        if no_start_date:
            print(f"Skipping {no_start_date} titles without a full start date (no premiere to anchor trends on)")

        todo = todo[: args.limit] if args.limit else todo
        missing = [item for item in todo if not (raw_dir / "trends" / f"{item['id']}.json").exists()]
        print(f"\n{len(todo)} titles, {len(missing)} trend requests to make (~{len(missing) * MIN_INTERVAL_S / 60:.0f} min)")

        for index, item in enumerate(missing, start=1):
            premiere = premiere_timestamp(item)
            variables = {
                "id": item["id"],
                "lo": premiere - PRE_DAYS * DAY - 1,
                "hi": premiere + POST_DAYS * DAY + 1,
                "year1": premiere + 364 * DAY,
            }
            status, body = client.query(TRENDS_QUERY, variables)
            save_record(raw_dir / "trends" / f"{item['id']}.json", variables, status, body)

            note = f"HTTP {status}"
            if status == 200:
                data = body["data"]
                rows = [row for page in range(1, TREND_PAGES + 1) for row in data[f"p{page}"]["mediaTrends"]]
                pre_air = sum(row["date"] < premiere for row in rows)
                note = f"{len(rows)} days, {pre_air} pre-air"
                if data[f"p{TREND_PAGES}"]["pageInfo"]["hasNextPage"]:
                    note += ", TRUNCATED"
            title = (item.get("title") or {}).get("romaji") or ""
            print(f"[{index}/{len(missing)}] {item['id']} {title[:55]}  ({note})")
    except AniListUnavailable as exc:
        print(f"\nAniList is unavailable: {exc}", file=sys.stderr)
        print("Everything fetched so far is saved; rerun the same command to resume.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted. Rerun the same command to resume.", file=sys.stderr)
        return 130

    print(f"\nDone. {client.requests_made} requests made. Raw responses are in {raw_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
