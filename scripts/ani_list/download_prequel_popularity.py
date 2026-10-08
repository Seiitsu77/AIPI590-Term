"""Fetch each cohort title's prequel popularity as it stood the day before the title premiered.

For every PREQUEL or PARENT anime relation of a cohort title (one whose trends
were downloaded), ask AniList for the prequel's last MediaTrend row dated
strictly before 00:00 JST on the title's premiere day. That is the franchise
popularity a platform could have known on day 0, unlike the prequel's
popularity today.

Many (title, prequel) pairs go into one request as GraphQL aliases. Responses are
saved per title under data/ani_list/raw/prequel_popularity/<anilist_id>.json;
titles already saved are skipped. Run scripts/ani_list/build_tables.py afterwards
to get prequel_popularity.csv.

Run without python -I: it imports the client from download.py next to it.

Usage:
    python scripts/ani_list/download_prequel_popularity.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from download import DEFAULT_OUT, AniListClient, AniListUnavailable, premiere_timestamp, save_record

ALIASES_PER_REQUEST = 25


def load_pairs(out: Path) -> dict[int, dict]:
    """anilist_id -> {"premiere": timestamp, "prequels": [prequel anilist ids]} for cohort titles."""
    titles: dict[int, dict] = {}
    for page in sorted((out / "raw" / "seasons").glob("*/page-*.json")):
        for media in json.loads(page.read_text(encoding="utf-8"))["response"]["data"]["Page"]["media"]:
            if not (out / "raw" / "trends" / f"{media['id']}.json").exists():
                continue
            prequels = sorted(
                {
                    edge["node"]["id"]
                    for edge in (media.get("relations") or {}).get("edges") or []
                    if edge.get("relationType") in ("PREQUEL", "PARENT") and edge["node"].get("type") == "ANIME"
                }
            )
            if prequels:
                titles[media["id"]] = {"premiere": premiere_timestamp(media), "prequels": prequels}
    return titles


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    out = DEFAULT_OUT
    raw_dir = out / "raw" / "prequel_popularity"
    titles = load_pairs(out)
    todo = [anilist_id for anilist_id in titles if not (raw_dir / f"{anilist_id}.json").exists()]
    n_pairs = sum(len(titles[anilist_id]["prequels"]) for anilist_id in todo)
    print(f"{len(titles)} cohort titles have a prequel/parent; {len(todo)} to fetch ({n_pairs} pairs)")

    batches: list[list[int]] = [[]]
    for anilist_id in todo:
        if len(batches[-1]) and sum(len(titles[t]["prequels"]) for t in batches[-1]) + len(
            titles[anilist_id]["prequels"]
        ) > ALIASES_PER_REQUEST:
            batches.append([])
        batches[-1].append(anilist_id)

    client = AniListClient()
    try:
        for index, batch in enumerate(b for b in batches if b):
            fields = []
            for anilist_id in batch:
                for prequel in titles[anilist_id]["prequels"]:
                    fields.append(
                        f"t{anilist_id}_p{prequel}: Page(perPage: 1) {{ mediaTrends(mediaId: {prequel}, "
                        f"date_lesser: {titles[anilist_id]['premiere']}, sort: DATE_DESC) {{ date popularity }} }}"
                    )
            status, body = client.query("{ " + " ".join(fields) + " }", {})
            if status != 200:
                raise AniListUnavailable(f"HTTP {status}: {body}")
            for anilist_id in batch:
                result = {
                    str(prequel): body["data"][f"t{anilist_id}_p{prequel}"]["mediaTrends"]
                    for prequel in titles[anilist_id]["prequels"]
                }
                variables = {"premiere": titles[anilist_id]["premiere"], "prequels": titles[anilist_id]["prequels"]}
                save_record(raw_dir / f"{anilist_id}.json", variables, status, result)
            print(f"[{index + 1}] saved {len(batch)} titles")
    except AniListUnavailable as exc:
        print(f"\nAniList is unavailable: {exc}\nRerun to resume.", file=sys.stderr)
        return 1

    print(f"Done. {client.requests_made} requests made.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
