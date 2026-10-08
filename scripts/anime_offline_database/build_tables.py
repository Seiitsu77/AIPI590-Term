"""Flatten a downloaded anime-offline-database release into CSV tables.

Writes to data/anime_offline_database/:

    anime.csv   one row per entry, with each site's ID as a column (mal_id, anilist_id, ...)
    id_map.csv  one row per (entry, site, site_id): the long form, for joins

An entry occasionally lists two IDs from the same site; anime.csv joins those with
"|" and id_map.csv keeps one row each. entry_id is the position in this release
and is not stable across releases: join on site IDs, not entry_id.

score_mean is the cross-site average score as of the release date, a post-air
snapshot like every other popularity number here.

Usage:
    python scripts/anime_offline_database/build_tables.py              # newest downloaded release
    python scripts/anime_offline_database/build_tables.py --tag 2026-27
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PROJECT_ROOT / "data" / "anime_offline_database"
ASSET_NAME = "anime-offline-database-minified.json"

# host -> short site name used in column names
SITES = {
    "myanimelist.net": "mal",
    "anilist.co": "anilist",
    "kitsu.app": "kitsu",
    "anidb.net": "anidb",
    "animenewsnetwork.com": "ann",
    "livechart.me": "livechart",
    "anime-planet.com": "animeplanet",
    "anisearch.com": "anisearch",
    "simkl.com": "simkl",
    "animecountdown.com": "animecountdown",
}
# The ID is the last path segment, or the value of ?id= for Anime News Network.
SOURCE_PATTERN = re.compile(r"^https?://(?:www\.)?(?P<host>[^/]+)/(?:.*[/=])?(?P<site_id>[^/=?]+)$")

ANIME_COLUMNS = [
    "entry_id", "title", "type", "episodes", "status", "season", "year", "duration_seconds", "score_mean",
    "studios", "producers", "tags", *[f"{site}_id" for site in SITES.values()], "n_sources",
]  # fmt: skip


def parse_source(url: str) -> tuple[str, str] | None:
    match = SOURCE_PATTERN.match(url)
    if not match or match["host"] not in SITES:
        return None
    return SITES[match["host"]], match["site_id"]


def entry_row(entry_id: int, entry: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ids: dict[str, list[str]] = {site: [] for site in SITES.values()}
    id_rows = []
    for url in entry.get("sources") or []:
        parsed = parse_source(url)
        if parsed is None:
            continue
        site, site_id = parsed
        ids[site].append(site_id)
        id_rows.append({"entry_id": entry_id, "site": site, "site_id": site_id, "url": url})

    season = entry.get("animeSeason") or {}
    duration = entry.get("duration") or {}
    row = {
        "entry_id": entry_id,
        "title": entry.get("title"),
        "type": entry.get("type"),
        "episodes": entry.get("episodes"),
        "status": entry.get("status"),
        "season": season.get("season"),
        "year": season.get("year"),
        "duration_seconds": duration.get("value") if duration.get("unit") == "SECONDS" else None,
        "score_mean": (entry.get("score") or {}).get("arithmeticMean"),
        "studios": "|".join(entry.get("studios") or []),
        "producers": "|".join(entry.get("producers") or []),
        "tags": "|".join(entry.get("tags") or []),
        "n_sources": len(entry.get("sources") or []),
    }
    for site, values in ids.items():
        row[f"{site}_id"] = "|".join(values)
    return row, id_rows


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CSV tables from anime-offline-database.")
    parser.add_argument("--tag", help="Release tag to read (default: newest downloaded).")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Data directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    releases = sorted(path.parent for path in (args.out / "raw").glob(f"*/{ASSET_NAME}"))
    if args.tag:
        releases = [path for path in releases if path.name == args.tag]
    if not releases:
        print("No downloaded release found. Run scripts/anime_offline_database/download.py first.", file=sys.stderr)
        return 1
    release_dir = releases[-1]
    payload = json.loads((release_dir / ASSET_NAME).read_text(encoding="utf-8"))

    anime_rows, id_rows = [], []
    for entry_id, entry in enumerate(payload["data"]):
        row, ids = entry_row(entry_id, entry)
        anime_rows.append(row)
        id_rows.extend(ids)

    write_csv(args.out / "anime.csv", ANIME_COLUMNS, anime_rows)
    write_csv(args.out / "id_map.csv", ["entry_id", "site", "site_id", "url"], id_rows)

    print(f"Release {release_dir.name} (lastUpdate {payload.get('lastUpdate')})")
    print(f"anime.csv   {len(anime_rows)} rows")
    print(f"id_map.csv  {len(id_rows)} rows")
    for site in SITES.values():
        print(f"  {site:<15} {sum(1 for row in anime_rows if row[f'{site}_id']):>6} entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
