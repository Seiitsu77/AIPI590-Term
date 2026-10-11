"""Collect AniList MediaTrend history for the titles in a metadata table.

MediaTrend is AniList's dated per-title series: one row per recorded day with
`trending`, `popularity`, `inProgress`, `averageScore`, `releasing` and `episode`.
It is activity on AniList itself, by AniList users. It is not Wikipedia pageviews and
not search volume, and the two should not be read as the same signal.

Rows are stamped at 15:00 UTC, which is 00:00 in Japan, and the row stamped at the start
of Japan day D holds day D. All day arithmetic here is in Japan time for that reason; a
UTC calendar date is one day early and would count the premiere day as pre-release.

Three things are fetched per title, cheapest first so the headline answers arrive early:

  latest   the newest row (sort DATE_DESC, one row), a cross-check that paging the
           history upwards really reaches the end
  window   Japan days D-90 to D-1 for premiere day D, asked for with date_greater /
           date_lesser, which tests whether the API can be queried by date
  full     every row, oldest first, 50 a page, until hasNextPage is false

`window` must equal the same days cut out of `full`; audit_trends.py checks that.
With --window-only the full history is skipped, which is what a larger collection
that only needs pre-release features would run.

Usage:
    python src/collect_trends.py --metadata <csv or parquet with id, mal_id, start_date> --out outputs/<stage>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from anilist_client import query

PER_PAGE = 50
MAX_PAGES_PER_TITLE = 250
WINDOW_DAYS = 90
DAY = 86400
HOUR = 3600
JST_OFFSET = 9 * HOUR

FIELDS = "mediaId date trending averageScore popularity inProgress releasing episode"
FULL_QUERY = """
query ($mediaId: Int, $page: Int, $perPage: Int, $sort: [MediaTrendSort]) {
  Page(page: $page, perPage: $perPage) {
    pageInfo { total perPage currentPage lastPage hasNextPage }
    mediaTrends(mediaId: $mediaId, sort: $sort) { %s }
  }
}
""" % FIELDS
WINDOW_QUERY = """
query ($mediaId: Int, $page: Int, $perPage: Int, $after: Int, $before: Int) {
  Page(page: $page, perPage: $perPage) {
    pageInfo { total perPage currentPage lastPage hasNextPage }
    mediaTrends(mediaId: $mediaId, date_greater: $after, date_lesser: $before, sort: DATE) { %s }
  }
}
""" % FIELDS


def read_metadata(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    return frame[["id", "mal_id", "start_date"]].rename(columns={"id": "anilist_id"})


def pages(media_id: int, kind: str, graphql: str, variables: dict, first_page_only: bool = False) -> list[dict]:
    rows, page = [], 1
    while page <= MAX_PAGES_PER_TITLE:
        name = f"trends/{media_id}/{kind}_p{page:04d}"
        block = query(name, graphql, {"mediaId": media_id, "page": page, "perPage": PER_PAGE, **variables})["Page"]
        info = block["pageInfo"]
        rows += [{**row, "page": page, "reported_total": info["total"], "has_next_page": info["hasNextPage"]} for row in block["mediaTrends"]]
        if first_page_only or not info["hasNextPage"] or not block["mediaTrends"]:
            break
        page += 1
    return rows


def premiere_jst_midnight(start_date) -> int | None:
    """Unix time of 00:00 Japan time on the premiere day, or None without a full date."""
    if pd.isna(start_date) or len(str(start_date)) != 10:
        return None
    return int(pd.Timestamp(start_date, tz="UTC").timestamp()) - JST_OFFSET


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window-only", action="store_true", help="skip the full histories")
    parser.add_argument("--recheck-latest", action="store_true", help="fetch the newest row once more, to see whether it moved since the first look")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args.out.mkdir(parents=True, exist_ok=True)
    titles = read_metadata(args.metadata)

    latest, window = [], []
    for title in titles.itertuples():
        media_id = int(title.anilist_id)
        if not args.window_only:
            pages(media_id, "full", FULL_QUERY, {"sort": ["DATE"]}, first_page_only=True)
        latest += pages(media_id, "latest", FULL_QUERY, {"sort": ["DATE_DESC"], "perPage": 1}, first_page_only=True)
        premiere = premiere_jst_midnight(title.start_date)
        if premiere is not None:
            # Rows sit on Japan midnights. Each bound is placed an hour inside the gap between
            # two rows, so day D-91 and the premiere day stay out whether or not the API
            # treats its bounds as inclusive.
            bounds = {"after": premiere - (WINDOW_DAYS + 1) * DAY + HOUR, "before": premiere - HOUR}
            window += [{**row, **bounds} for row in pages(media_id, "window", WINDOW_QUERY, bounds)]
        print(f"{media_id}: newest row and pre-release window cached", flush=True)

    full = []
    if not args.window_only:
        for title in titles.itertuples():
            rows = pages(int(title.anilist_id), "full", FULL_QUERY, {"sort": ["DATE"]})
            full += rows
            print(f"{int(title.anilist_id)}: {len(rows)} rows in {rows[-1]['page'] if rows else 0} pages", flush=True)

    if args.recheck_latest:
        for title in titles.itertuples():
            pages(int(title.anilist_id), "recheck", FULL_QUERY, {"sort": ["DATE_DESC"], "perPage": 1}, first_page_only=True)

    ids = titles.set_index("anilist_id")["mal_id"]
    for name, rows in (("trends_full", full), ("trends_window", window), ("trends_latest", latest)):
        frame = pd.DataFrame(rows).rename(columns={"mediaId": "anilist_id", "date": "timestamp"})
        if len(frame):
            frame.insert(1, "mal_id", frame["anilist_id"].map(ids).astype("int64"))
            frame.insert(2, "day_jst", pd.to_datetime(frame["timestamp"] + JST_OFFSET, unit="s").dt.strftime("%Y-%m-%d"))
        frame.to_parquet(args.out / f"{name}.parquet", index=False)
        print(f"{name}: {len(frame)} rows", flush=True)


if __name__ == "__main__":
    main()
