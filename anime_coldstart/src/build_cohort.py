"""Build the 2016-2020 premiere cohort and its pre-air metadata features.

Input is the MAL catalog snapshot from the CC0 "Anime Recommendation Database 2020"
(data/raw/anime.csv). The snapshot was scraped in early 2021, so every count in it is
"as of early 2021", not "as of the premiere date". That matters twice below and both
are flagged where they occur.

Definitions were fixed before any feature was compared against the outcome:

  Cohort    titles whose `Aired` field starts with a full "Mon D, YYYY" date between
            2016-01-01 and 2020-12-31. Wikipedia pageviews begin 2015-07-01, so 2016 is
            the first premiere year with a complete 90-day pre-air window.
  Outcome   y = log1p(n_pos), n_pos = Score-8 + Score-9 + Score-10. This is the count of
            MAL users who rated the title 8 or higher, the same positive threshold
            Anime Compass trains its ALS model on. Ratings can only exist after airing.
  Baseline  metadata knowable before the premiere: type, source, age rating, genres,
            episode count, premiere quarter, a studio prior and a franchise prior.

The studio and franchise priors average the outcome of *earlier* titles. Those outcomes
are read from the 2021 snapshot rather than from the premiere date, which is a mild
look-ahead in the baseline's favour (older titles had kept accumulating ratings). It
biases against finding an alt-data increment, not for it.

Usage:
    python src/build_cohort.py
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data"

COHORT_START = pd.Timestamp("2016-01-01")
COHORT_END = pd.Timestamp("2020-12-31")
KEEP_TYPES = ("TV", "Movie", "ONA", "OVA", "Special")
# A studio's earlier title only counts once it has had time to air and be rated.
STUDIO_PRIOR_LAG_DAYS = 180

FULL_DATE = re.compile(r"^([A-Z][a-z]{2}) (\d{1,2}), (\d{4})")
ANY_YEAR = re.compile(r"(19|20)\d{2}")

# Season and format markers stripped before two titles are compared as one franchise.
_MARKERS = [
    r"\b(the )?final season\b.*$",
    r"\b\d+(st|nd|rd|th) season\b.*$",
    r"\bseason \d+\b.*$",
    r"\b(part|cour) \d+\b.*$",
    r"\b(movie|gekijouban|ova|oad|ona|specials?|recap|picture drama)\b.*$",
    r"\b(ii|iii|iv|vi|vii)$",
    r"\s\d+$",
]


def start_date(aired: str) -> pd.Timestamp | None:
    match = FULL_DATE.match(aired or "")
    if not match:
        return None
    return pd.to_datetime(f"{match.group(1)} {match.group(2)} {match.group(3)}", format="%b %d %Y")


def approx_date(aired: str) -> pd.Timestamp | None:
    """Start date, falling back to 1 January when only a year is given.

    Used only to order earlier catalog titles, never for a cohort title's window.
    """
    exact = start_date(aired)
    if exact is not None:
        return exact
    year = ANY_YEAR.search(aired or "")
    return pd.Timestamp(year=int(year.group(0)), month=1, day=1) if year else None


def franchise_root(name: str) -> str:
    """Normalised title with season/format markers removed.

    The catalog has no franchise or related-works column, so the title is the only
    evidence of a sequel. "Re:Zero" keeps its colon; only ": " splits off a subtitle.
    """
    text = unicodedata.normalize("NFKC", name or "").lower()
    text = text.split(": ")[0]
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    for pattern in _MARKERS:
        text = re.sub(pattern, "", text).strip()
    return text


def root_keys(root: str) -> list[str]:
    """The root itself plus every token-prefix of at least two words.

    Single-word prefixes are excluded: "K" must not match "K-On!".
    """
    tokens = root.split()
    keys = [root] if root else []
    keys += [" ".join(tokens[:n]) for n in range(2, len(tokens))]
    return keys


def main() -> None:
    catalog = pd.read_csv(RAW / "anime.csv")
    score_cols = [f"Score-{k}" for k in (8, 9, 10)]
    for column in score_cols:
        # "Unknown" marks an empty rating bin, not a missing title.
        catalog[column] = pd.to_numeric(catalog[column], errors="coerce").fillna(0.0)
    catalog["n_pos"] = catalog[score_cols].sum(axis=1)
    catalog["y"] = np.log1p(catalog["n_pos"])
    catalog["start"] = catalog["Aired"].map(start_date)
    catalog["start_approx"] = catalog["Aired"].map(approx_date)
    catalog["root"] = catalog["Name"].map(franchise_root)

    cohort = catalog[
        catalog["start"].between(COHORT_START, COHORT_END) & catalog["Type"].isin(KEEP_TYPES)
    ].copy()

    # Franchise prior: best outcome among earlier titles sharing a root.
    by_root: dict[str, list[tuple[pd.Timestamp, float, int]]] = {}
    dated = catalog.dropna(subset=["start_approx"])
    for mal_id, root, when, y in zip(dated["MAL_ID"], dated["root"], dated["start_approx"], dated["y"], strict=True):
        if root:
            by_root.setdefault(root, []).append((when, float(y), int(mal_id)))

    franchise_prior, franchise_n = [], []
    for mal_id, root, when in zip(cohort["MAL_ID"], cohort["root"], cohort["start"], strict=True):
        earlier = [
            y
            for key in root_keys(root)
            for (other_when, y, other_id) in by_root.get(key, ())
            if other_when < when and other_id != mal_id
        ]
        franchise_prior.append(max(earlier) if earlier else np.nan)
        franchise_n.append(len(earlier))
    cohort["franchise_prior"] = franchise_prior
    cohort["franchise_n"] = franchise_n
    cohort["is_sequel"] = (cohort["franchise_n"] > 0).astype(int)

    # Studio prior: mean outcome of the studio's titles that premiered long enough ago.
    studio_rows: dict[str, list[tuple[pd.Timestamp, float]]] = {}
    for studios, when, y in zip(dated["Studios"], dated["start_approx"], dated["y"], strict=True):
        for studio in str(studios).split(", "):
            if studio and studio != "Unknown":
                studio_rows.setdefault(studio, []).append((when, float(y)))

    studio_prior, studio_n = [], []
    lag = pd.Timedelta(days=STUDIO_PRIOR_LAG_DAYS)
    for studios, when in zip(cohort["Studios"], cohort["start"], strict=True):
        best, count = np.nan, 0
        for studio in str(studios).split(", "):
            past = [y for (other_when, y) in studio_rows.get(studio, ()) if other_when < when - lag]
            if past:
                count += len(past)
                mean = float(np.mean(past))
                best = mean if np.isnan(best) else max(best, mean)
        studio_prior.append(best)
        studio_n.append(count)
    cohort["studio_prior"] = studio_prior
    cohort["studio_n"] = studio_n

    cohort["episodes"] = pd.to_numeric(cohort["Episodes"], errors="coerce")
    cohort["season"] = cohort["start"].dt.year.astype(str) + "Q" + cohort["start"].dt.quarter.astype(str)

    columns = [
        "MAL_ID", "Name", "English name", "Type", "Source", "Rating", "Genres", "Studios",
        "episodes", "start", "season", "n_pos", "y", "Members", "Completed", "Plan to Watch",
        "root", "is_sequel", "franchise_n", "franchise_prior", "studio_n", "studio_prior",
    ]  # fmt: skip
    cohort = cohort[columns].sort_values(["start", "MAL_ID"]).reset_index(drop=True)
    cohort.to_csv(OUT / "cohort.csv", index=False)

    print(f"catalog rows: {len(catalog)}")
    print(f"cohort rows:  {len(cohort)}")
    print(cohort.groupby("Type").agg(n=("MAL_ID", "size"), sequel_share=("is_sequel", "mean"),
                                     median_n_pos=("n_pos", "median")).round(3))  # fmt: skip
    print(cohort.groupby(cohort["start"].dt.year)["MAL_ID"].size().rename("by year"))


if __name__ == "__main__":
    main()
