"""Fix day 0 for every cold title.

Day 0 is the earliest of three dates, all Japan calendar dates:

  MAL premiere date           the date roles were assigned by
  AniList startDate           when the two sources disagree
  first episode's airing day  from AniList's airing schedule, when its first entry is episode 1

Taking the earliest is the conservative choice: a premiere dated a day late would let
premiere-day activity into a "pre-air" feature, a premiere dated early only throws away
a little real pre-air data.

Reads no ratings. Writes artifacts/day_zero.csv, which every feature script uses and no
feature script recomputes.

Usage:
    python src/day_zero.py
"""

from __future__ import annotations

import sys

import pandas as pd

from common import ANILIST_ITEMS, ANILIST_META, ARTIFACTS, JST_OFFSET_SECONDS, MANIFEST, ROLES


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    roles = pd.read_csv(MANIFEST / "item_roles.csv", parse_dates=["premiere"])
    cold = roles[roles["role"].isin(ROLES)][["anime_id", "role", "premiere"]].rename(columns={"premiere": "mal_premiere"})

    anilist = pd.read_csv(ANILIST_ITEMS, usecols=["anime_id", "anilist_id", "anilist_start"], parse_dates=["anilist_start"])
    meta = pd.read_csv(ANILIST_META, usecols=["anime_id", "first_scheduled_episode", "first_scheduled_airing_at"])
    airing = pd.to_datetime(meta["first_scheduled_airing_at"] + JST_OFFSET_SECONDS, unit="s").dt.normalize()
    meta["episode_1_day_jst"] = airing.where(meta["first_scheduled_episode"] == 1)

    frame = cold.merge(anilist, on="anime_id", how="left").merge(meta[["anime_id", "episode_1_day_jst"]], on="anime_id", how="left")
    frame["day_zero"] = frame[["mal_premiere", "anilist_start", "episode_1_day_jst"]].min(axis=1)
    frame["days_earlier_than_mal"] = (frame["mal_premiere"] - frame["day_zero"]).dt.days
    frame["season"] = frame["mal_premiere"].dt.year.astype(str) + "Q" + frame["mal_premiere"].dt.quarter.astype(str)
    if frame["day_zero"].isna().any() or (frame["day_zero"] > frame["mal_premiere"]).any():
        raise SystemExit("day 0 is missing or later than the MAL premiere for some title")
    frame.sort_values("anime_id").to_csv(ARTIFACTS / "day_zero.csv", index=False, lineterminator="\n")

    moved = frame[frame["days_earlier_than_mal"] > 0]
    print(f"{len(frame)} cold titles; day 0 earlier than the MAL premiere for {len(moved)}")
    print(moved["days_earlier_than_mal"].value_counts().sort_index().rename("days earlier").to_string())
    print(frame.groupby("role")["season"].unique().map(sorted).to_string())


if __name__ == "__main__":
    main()
