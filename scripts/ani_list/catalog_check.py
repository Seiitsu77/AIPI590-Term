"""Check that old-anime metadata from AniList lines up with the new-anime metadata.

Reads data/ani_list/catalog.csv (old anime, from download_catalog.py) and
data/ani_list/anime.csv (new anime) and reports:

    1. coverage   how many of the learned ALS items AniList returned, and what the
                  missing ones are (type and year from the Anime Compass catalog)
    2. IDs        AniList's idMal vs anime-offline-database for the same AniList id
    3. vocabulary for the 926 new TV titles, how much of their studio / genre /
                  tag / source / staff vocabulary also appears among old anime,
                  before (offline-db, Anime Compass catalog) and after (AniList)

Usage:
    python scripts/ani_list/catalog_check.py
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANILIST = PROJECT_ROOT / "data" / "ani_list"
OFFLINE_DB = PROJECT_ROOT / "data" / "anime_offline_database" / "anime.csv"
ALS = PROJECT_ROOT / "data" / "anime_compass" / "als_production_item_factors.npz"
COMPASS_CATALOG = PROJECT_ROOT / "data" / "anime_compass" / "anime_catalog_serving.json"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def split(value: str) -> list[str]:
    return [part for part in value.split("|") if part]


def names(rows: list[dict[str, str]], column: str, strip_rank: bool = False) -> Counter:
    counter: Counter = Counter()
    for row in rows:
        for part in split(row[column]):
            if strip_rank:
                part = part.rsplit(":", 1)[0]
            if column == "staff":
                part = part.split(": ", 1)[-1]
            counter[part] += 1
    return counter


def share_found(new: Counter, old: set[str]) -> str:
    """Share of new-anime vocabulary items found among old anime: distinct items, and weighted by use."""
    if not new:
        return "-"
    distinct = sum(1 for name in new if name in old) / len(new)
    weighted = sum(count for name, count in new.items() if name in old) / sum(new.values())
    return f"{distinct:.0%} of {len(new)} distinct, {weighted:.0%} of uses"


def main() -> None:
    with np.load(ALS, allow_pickle=False) as artifact:
        ids, norms = artifact["anime_ids"].astype(int), np.linalg.norm(artifact["item_factors"], axis=1)
    learned = {int(mal_id) for mal_id, norm in zip(ids, norms) if norm >= 1e-3}
    compass = {int(item["id"]): item for item in json.loads(COMPASS_CATALOG.read_text(encoding="utf-8"))}
    catalog = read_csv(ANILIST / "catalog.csv")
    new = [row for row in read_csv(ANILIST / "anime.csv") if row["trends_status"] == "ok" and row["format"] == "TV"]

    # 1. Coverage
    by_mal: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in catalog:
        if row["mal_id"]:
            by_mal[int(row["mal_id"])].append(row)
    found = learned & set(by_mal)
    missing = sorted(learned - found)
    print(f"1. Coverage: AniList returned {len(catalog)} entries for {len(found)}/{len(learned)} "
          f"learned ALS items ({len(found) / len(learned):.1%}); {len(missing)} missing")  # fmt: skip
    print(f"   MAL ids matched by more than one AniList entry: {sum(1 for rows in by_mal.values() if len(rows) > 1)}")
    print("   missing by type:", dict(Counter(compass[m].get("type") for m in missing).most_common()))
    decades = Counter((int(compass[m]["start_year"]) // 10 * 10) if compass[m].get("start_year") else None for m in missing)
    print("   missing by decade:", dict(sorted(decades.items(), key=lambda item: (item[0] is None, item[0] or 0))))
    members = sorted(int(compass[m].get("members") or 0) for m in missing)
    print(f"   missing titles' MAL members: median {members[len(members) // 2] if members else 0}, max {members[-1] if members else 0}")

    # 2. IDs vs anime-offline-database
    offline_mal: dict[str, set[str]] = defaultdict(set)
    for entry in read_csv(OFFLINE_DB):
        for anilist_id in split(entry["anilist_id"]):
            offline_mal[anilist_id].update(split(entry["mal_id"]))
    outcome: Counter = Counter()
    for row in catalog:
        other = offline_mal.get(row["anilist_id"])
        if other is None:
            outcome["AniList entry missing from offline-db"] += 1
        elif row["mal_id"] in other:
            outcome["agree"] += 1
        elif not other:
            outcome["offline-db has no MAL id"] += 1
        else:
            outcome["DISAGREE"] += 1
    print(f"\n2. AniList idMal vs offline-db: {dict(outcome.most_common())}")

    # 3. Vocabulary: what share of the new titles' vocabulary also occurs among old anime?
    old_offline = [entry for entry in read_csv(OFFLINE_DB) if any(int(m) in learned for m in split(entry["mal_id"]))]
    before = {
        "main_studios": {name for entry in old_offline for name in split(entry["studios"])},
        "genres": {genre for item in compass.values() for genre in item.get("genres") or []},
        "tags": {tag for entry in old_offline for tag in split(entry["tags"])},
    }
    print("\n3. New-anime vocabulary found among old anime")
    print(f"   {'field':<14}{'before (offline-db / Anime Compass catalog)':<48}after (AniList)")
    for column, strip in (("main_studios", False), ("genres", False), ("tags", True), ("source", False), ("staff", False)):
        new_vocab = names(new, column, strip) if column != "source" else Counter(row["source"] for row in new)
        old_vocab = set(names(catalog, column, strip)) if column != "source" else {row["source"] for row in catalog}
        old_before = before.get(column)
        print(f"   {column:<14}{share_found(new_vocab, old_before) if old_before else 'not available':<48}"
              f"{share_found(new_vocab, old_vocab)}")


if __name__ == "__main__":
    main()
