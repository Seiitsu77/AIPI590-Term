"""Collect the pre-release MediaTrend window for the 2019-2020 cold-item candidates.

One request per title. It carries, as aliases:

  p1..p4   up to 200 daily rows from Japan day D_early-100 to D_late+30, where D_early
           and D_late are the earlier and later of the MAL and AniList premiere dates.
           The days after the premiere are not features. They are there so the audit
           can see when `releasing` and `episode` switch on relative to the date we
           call day 0, which is how a mis-dated premiere shows itself.
  first    the oldest row AniList holds for the title, to tell "not tracked yet" apart
           from "tracked, but nobody had listed it"
  media    format, start date, and the first two airing-schedule entries. `airingAt` is
           an exact UTC time, the only source here finer than a calendar date.

Rows are stamped at 15:00 UTC, which is 00:00 in Japan, so all day arithmetic is in
Japan time (see collect_trends.py).

Requests go through anilist_client: 20 a minute, cached on disk, resumable, logged.

Usage:
    python src/collect_candidate_windows.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from anilist_client import query

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "personalized_coldstart"

DAY, HOUR, JST_OFFSET = 86400, 3600, 9 * 3600
DAYS_BEFORE, DAYS_AFTER, PAGES = 100, 30, 4
FIELDS = "mediaId date trending averageScore popularity inProgress releasing episode"

GRAPHQL = (
    "query ($id: Int, $lo: Int, $hi: Int) {\n"
    + "\n".join(
        f"  p{page}: Page(page: {page}, perPage: 50) {{ pageInfo {{ hasNextPage }} "
        f"mediaTrends(mediaId: $id, date_greater: $lo, date_lesser: $hi, sort: DATE) {{ {FIELDS} }} }}"
        for page in range(1, PAGES + 1)
    )
    + f"\n  first: Page(perPage: 1) {{ mediaTrends(mediaId: $id, sort: DATE) {{ {FIELDS} }} }}"
    + "\n  media: Media(id: $id) { id idMal format status startDate { year month day } "
    "airingSchedule(perPage: 2) { nodes { episode airingAt } } }\n}"
)


def jst_midnight(day: pd.Timestamp) -> int:
    """Unix time of 00:00 Japan time on a calendar day."""
    return int(pd.Timestamp(day.date(), tz="UTC").timestamp()) - JST_OFFSET


def jst_day(timestamp: int) -> str:
    return pd.to_datetime(timestamp + JST_OFFSET, unit="s").strftime("%Y-%m-%d")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    candidates = pd.read_csv(OUT / "candidate_items.csv", parse_dates=["premiere", "anilist_start"])
    candidates = candidates.dropna(subset=["anilist_id"]).astype({"anilist_id": "int64"})

    trends, meta = [], []
    for title in candidates.itertuples():
        dates = [d for d in (title.premiere, title.anilist_start) if not pd.isna(d)]
        early, late = min(dates), max(dates)
        # Each bound sits an hour inside the gap between two daily rows, so the result
        # does not depend on whether the API treats its bounds as inclusive.
        variables = {"id": int(title.anilist_id), "lo": jst_midnight(early) - (DAYS_BEFORE + 1) * DAY + HOUR, "hi": jst_midnight(late) + (DAYS_AFTER + 1) * DAY - HOUR}
        data = query(f"candidates/{title.anilist_id}", GRAPHQL, variables)

        rows = [row for page in range(1, PAGES + 1) for row in data[f"p{page}"]["mediaTrends"]]
        trends += [{"anime_id": title.anime_id, **row} for row in rows]
        first = data["first"]["mediaTrends"]
        media = data["media"] or {}
        start = media.get("startDate") or {}
        schedule = sorted((media.get("airingSchedule") or {}).get("nodes") or [], key=lambda node: node["airingAt"])
        meta.append(
            {
                "anime_id": title.anime_id,
                "anilist_id": int(title.anilist_id),
                "window_rows": len(rows),
                "window_truncated": bool(data[f"p{PAGES}"]["pageInfo"]["hasNextPage"]),
                "first_row_timestamp": first[0]["date"] if first else None,
                "first_row_day_jst": jst_day(first[0]["date"]) if first else None,
                "api_format": media.get("format"),
                "api_status": media.get("status"),
                "api_start_date": "-".join(f"{start[k]:02d}" if k != "year" else str(start[k]) for k in ("year", "month", "day")) if all(start.get(k) for k in ("year", "month", "day")) else None,
                "first_scheduled_episode": schedule[0]["episode"] if schedule else None,
                "first_scheduled_airing_at": schedule[0]["airingAt"] if schedule else None,
            }
        )
        print(f"{title.anilist_id}: {len(rows)} rows", flush=True)

    frame = pd.DataFrame(trends).rename(columns={"mediaId": "anilist_id", "date": "timestamp"})
    frame.insert(3, "day_jst", pd.to_datetime(frame["timestamp"] + JST_OFFSET, unit="s").dt.strftime("%Y-%m-%d"))
    frame.to_parquet(OUT / "candidate_trends.parquet", index=False)
    pd.DataFrame(meta).to_csv(OUT / "candidate_anilist_meta.csv", index=False)
    print(f"{len(meta)} titles, {len(frame)} trend rows")


if __name__ == "__main__":
    main()
