"""Build the point-in-time feature table for the pre-air value experiment (see PLAN.md).

Every feature is computed as of 00:00 JST on the title's premiere day:

    trend rows        only rows with days_from_premiere <= -1
    prequel           popularity on its last trend row before the premiere day
    studio history    earlier cohort titles whose day +120 fell before this premiere

Labels are carried along in separate y_* columns and never feed a feature.

Writes experiments/preair_value/data/features.csv.

Usage:
    python experiments/preair_value/features.py
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANILIST = PROJECT_ROOT / "data" / "ani_list"
CATALOG = PROJECT_ROOT / "data" / "anime_compass" / "anime_catalog_serving.json"
OUT = Path(__file__).resolve().parent / "data" / "features.csv"

TEST_FROM_YEAR = 2024  # listed seasons 2021-2023 train, 2024-2025 test
SOURCES = ("MANGA", "LIGHT_NOVEL", "ORIGINAL", "VIDEO_GAME", "MULTIMEDIA_PROJECT")  # >= 10 train titles; rest -> OTHER
GENRES = (
    "Action", "Adventure", "Comedy", "Drama", "Ecchi", "Fantasy", "Horror", "Mahou Shoujo", "Mecha", "Music",
    "Mystery", "Psychological", "Romance", "Sci-Fi", "Slice of Life", "Sports", "Supernatural", "Thriller",
)  # fmt: skip
STUDIO_LAG_DAYS = 120
PRE_AIR_DAYS = (1, 30, 90)

METADATA_FEATURES = [
    *[f"src_{source}" for source in SOURCES], "src_OTHER",
    *[f"genre_{genre}" for genre in GENRES],
    "log_duration",
    "is_sequel", "log_prequel_pop", "prequel_pop_missing",
    "studio_prior", "log_studio_n_prior", "studio_no_history",
]  # fmt: skip
ALT_FEATURES = [
    "log_pop_m1", "log_pop_m30", "log_pop_m90", "growth_30", "log_trending_m7", "not_tracked", "log_days_tracked",
]  # fmt: skip


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def strictly_before(fuzzy: str, day: date) -> bool:
    """True only if an AniList fuzzy date ('YYYY', 'YYYY-MM' or 'YYYY-MM-DD') is known to fall before day."""
    parts = [int(part) for part in fuzzy.split("-")] if fuzzy else []
    for known, reference in zip(parts, (day.year, day.month, day.day)):
        if known != reference:
            return known < reference
    return False


def value_at(rows: list[tuple[int, float]], day: int, horizon: int) -> float:
    """Value on `day` from rows (day, value) with day <= horizon: linear interpolation, carry forward, 0 before the first row."""
    usable = [row for row in rows if row[0] <= horizon]
    before = [row for row in usable if row[0] <= day]
    if not before:
        return 0.0
    day0, value0 = before[-1]
    after = [row for row in usable if row[0] > day]
    if day0 == day or not after:
        return value0
    day1, value1 = after[0]
    return value0 + (value1 - value0) * (day - day0) / (day1 - day0)


def main() -> None:
    anime = [row for row in read_csv(ANILIST / "anime.csv") if row["trends_status"] == "ok"]
    premiere = {row["anilist_id"]: date.fromisoformat(row["start_date"]) for row in anime}

    trends: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    for row in read_csv(ANILIST / "trends.csv"):
        trends[row["anilist_id"]].append(
            (int(row["days_from_premiere"]), float(row["popularity"] or 0), float(row["trending"] or 0))
        )

    earlier_prequels: dict[str, set[str]] = defaultdict(set)
    for relation in read_csv(ANILIST / "relations.csv"):
        title = relation["anilist_id"]
        if (
            title in premiere
            and relation["target_type"] == "ANIME"
            and relation["relation_type"] in ("PREQUEL", "PARENT")
            and strictly_before(relation["target_start_date"], premiere[title])
        ):
            earlier_prequels[title].add(relation["target_anilist_id"])

    prequel_pop: dict[tuple[str, str], float] = {}
    for row in read_csv(ANILIST / "prequel_popularity.csv"):
        if row["popularity"]:
            assert int(row["days_from_premiere"]) <= -1, row
            prequel_pop[(row["anilist_id"], row["prequel_anilist_id"])] = float(row["popularity"])

    # Day +120 popularity of every cohort title, for the studio track record.
    pop_120 = {
        title: value_at([(day, pop) for day, pop, _ in trends[title]], STUDIO_LAG_DAYS, STUDIO_LAG_DAYS)
        for title in premiere
    }
    studios = {row["anilist_id"]: set(filter(None, row["main_studios"].split("|"))) for row in anime}
    titles_by_studio: dict[str, list[str]] = defaultdict(list)
    for title, names in studios.items():
        for name in names:
            titles_by_studio[name].append(title)

    catalog_members = {
        int(item["id"]): item.get("members") for item in json.loads(CATALOG.read_text(encoding="utf-8"))
    }

    rows: list[dict[str, Any]] = []
    for item in anime:
        title, day0 = item["anilist_id"], premiere[item["anilist_id"]]
        source = item["source"] if item["source"] in SOURCES else "OTHER"
        genres = set(item["genres"].split("|"))
        row: dict[str, Any] = {
            "anilist_id": title,
            "mal_id": item["mal_id"],
            "title": item["title_romaji"],
            "listed_season": item["listed_season"],
            "split": "test" if int(item["listed_season"][:4]) >= TEST_FROM_YEAR else "train",
            **{f"src_{name}": int(source == name) for name in (*SOURCES, "OTHER")},
            **{f"genre_{genre}": int(genre in genres) for genre in GENRES},
            "log_duration": math.log(float(item["duration_min"])),
        }

        # Franchise: the most popular earlier prequel/parent, as of the day before premiere.
        prequels = earlier_prequels.get(title, set())
        known = [prequel_pop[(title, p)] for p in prequels if (title, p) in prequel_pop]
        row["is_sequel"] = int(bool(prequels))
        row["log_prequel_pop"] = math.log1p(max(known)) if known else 0.0
        row["prequel_pop_missing"] = int(bool(prequels) and not known)

        # Studio track record: earlier cohort titles by a shared main studio whose day +120 preceded this premiere.
        history = {
            other
            for name in studios[title]
            for other in titles_by_studio[name]
            if other != title and premiere[other] + timedelta(days=STUDIO_LAG_DAYS) < day0
        }
        row["studio_prior"] = (
            sum(math.log1p(pop_120[other]) for other in history) / len(history) if history else None
        )
        row["log_studio_n_prior"] = math.log1p(len(history))
        row["studio_no_history"] = int(not history)

        # Alternative data: AniList activity strictly before the premiere day.
        pre_air = [(day, pop) for day, pop, _ in trends[title] if day <= -1]
        for days in PRE_AIR_DAYS:
            row[f"log_pop_m{days}"] = math.log1p(value_at(pre_air, -days, -1))
        row["growth_30"] = row["log_pop_m1"] - row["log_pop_m30"]
        last_week = [trend for day, _, trend in trends[title] if -7 <= day <= -1]
        row["log_trending_m7"] = math.log1p(sum(last_week) / len(last_week)) if last_week else 0.0
        row["not_tracked"] = int(not pre_air)
        tracked = int(item["days_tracked_before_premiere"]) if item["days_tracked_before_premiere"] else 0
        row["log_days_tracked"] = math.log1p(max(0, tracked))

        # Labels (never features).
        row["y_score80"] = math.log1p(int(item["score_80plus"] or 0))
        members = catalog_members.get(int(item["mal_id"])) if item["mal_id"] else None
        row["y_mal_members"] = math.log1p(int(members)) if members else None
        rows.append(row)

    # Titles with no studio history get the train-only mean, flagged by studio_no_history.
    train_priors = [row["studio_prior"] for row in rows if row["split"] == "train" and row["studio_prior"] is not None]
    fill = sum(train_priors) / len(train_priors)
    for row in rows:
        if row["studio_prior"] is None:
            row["studio_prior"] = fill

    columns = ["anilist_id", "mal_id", "title", "listed_season", "split", *METADATA_FEATURES, *ALT_FEATURES,
               "y_score80", "y_mal_members"]  # fmt: skip
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)

    n_train = sum(row["split"] == "train" for row in rows)
    print(f"{OUT}: {len(rows)} titles ({n_train} train, {len(rows) - n_train} test)")
    print(f"  sequels {sum(r['is_sequel'] for r in rows)}, prequel popularity missing {sum(r['prequel_pop_missing'] for r in rows)}")
    print(f"  no studio history {sum(r['studio_no_history'] for r in rows)}, not tracked pre-air {sum(r['not_tracked'] for r in rows)}")
    print(f"  y_mal_members present {sum(r['y_mal_members'] is not None for r in rows)}")


if __name__ == "__main__":
    main()
