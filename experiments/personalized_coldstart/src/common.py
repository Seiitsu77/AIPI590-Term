"""Constants and helpers shared by every step of the personalized cold-start experiment.

Everything the pre-registration fixes lives here as a constant, so a later change to a
definition is a visible diff to this file rather than an edit buried in a script.

The three title helpers at the bottom are copied from the first feasibility round
(anime_coldstart/src/build_cohort.py) so this experiment does not import from it.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
MANIFEST = ROOT / "manifest"
ARTIFACTS = ROOT / "artifacts"  # git-ignored: the experimental ALS, features, predictions

# Inputs. Raw ratings and AniList responses stay out of git; they are pinned by digest.
RATINGS = Path(r"C:\Study\Toy project\archive\rating_complete.csv")
RATINGS_SHA256 = "b60519348a90bd5e02c25355b374f7ca055a0637a237f0e163447953b13ffaa0"
MAL_CATALOG = REPO / "anime_coldstart" / "data" / "raw" / "anime.csv"
MAL_CATALOG_SHA256 = "6e6d723f3e021e084a281d1ec557a0a10dfb9721d256b87596e02c27eb1f77e9"
SERVING_CATALOG = Path(r"C:\Study\Toy project\data\processed\anime_catalog_serving.json")
SERVING_CATALOG_SHA256 = "4b11cf3e5f3d015ed071232e569068248af709e7979e561194b451e3ac716bf3"

# Alternative-data inputs. The AniList files are API responses and stay local; the
# Wikipedia files are free to share but live with the first feasibility round. Their
# digests are recorded in the manifest when it is frozen.
ANILIST_DIR = REPO / "experiments" / "alt_data_coldstart" / "outputs" / "personalized_coldstart"
ANILIST_TRENDS = ANILIST_DIR / "candidate_trends.parquet"
ANILIST_META = ANILIST_DIR / "candidate_anilist_meta.csv"
ANILIST_ITEMS = ANILIST_DIR / "candidate_items.csv"
WIKI_DIR = REPO / "anime_coldstart" / "data"
WIKI_VIEWS = WIKI_DIR / "pageviews_daily.npz"
WIKI_INDEX = WIKI_DIR / "pageviews_index.csv"
WIKI_MAP = WIKI_DIR / "mal_wikipedia_map.csv"
WIKI_EXTRA = WIKI_DIR / "mal_wikipedia_extra_articles.csv"
ALT_DATA_INPUTS = {
    "anilist_trends": ANILIST_TRENDS,
    "anilist_meta": ANILIST_META,
    "anilist_items": ANILIST_ITEMS,
    "wiki_views": WIKI_VIEWS,
    "wiki_index": WIKI_INDEX,
    "wiki_map": WIKI_MAP,
    "wiki_extra": WIKI_EXTRA,
}

# Feature cutoffs, in days relative to day 0. Nothing later than CUTOFF_DAY is ever read.
CUTOFF_DAY = -2
ATTENTION_WINDOW = (-30, CUTOFF_DAY)  # AniList rows, Japan days
GROWTH_REFERENCE_WINDOW = (-37, -30)  # AniList rows, Japan days
WIKI_WINDOW = (-31, CUTOFF_DAY)  # Wikipedia days, UTC
JST_OFFSET_SECONDS = 9 * 3600
LANGS = ("en", "ja", "zh")

POSITIVE, NEGATIVE = 8, 5
MIN_WARM_POSITIVES = 5
WARM_BEFORE_YEAR = 2019
ROLES = {
    "cold_train": ("2019-01-01", "2019-09-30"),
    "cold_val": ("2019-10-01", "2020-03-31"),
    "cold_test": ("2020-04-01", "2020-09-30"),
    "cold_test_late": ("2020-10-01", "2020-12-31"),
}
SEED = 20261010
USER_GROUPS = ("ranker_train", "validation", "test")
MAX_DEVELOPMENT_USERS = 20_000  # per group, for ranker_train and validation only

FULL_DATE = re.compile(r"^([A-Z][a-z]{2}) (\d{1,2}), (\d{4})")
ANY_YEAR = re.compile(r"(19|20)\d{2}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def splitmix64(values: np.ndarray, salt: int) -> np.ndarray:
    """Deterministic 64-bit mix of integer ids (Steele, Lea & Flood's SplitMix64 finaliser).

    Used instead of a random generator so that a user's group depends only on the user
    id and the salt, never on library version, row order or which other users exist.
    """
    with np.errstate(over="ignore"):
        z = values.astype(np.uint64) + np.uint64(salt) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def user_group(user_ids: np.ndarray) -> np.ndarray:
    """Index into USER_GROUPS for each user id."""
    return (splitmix64(user_ids, SEED) % np.uint64(len(USER_GROUPS))).astype(np.int8)


def sampling_key(user_ids: np.ndarray) -> np.ndarray:
    """Second, independent mix. The development users are the eligible ones with the smallest keys."""
    return splitmix64(user_ids, SEED + 1)


# ------------------------------------------------------------------ titles

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


def premiere_date(aired: str) -> pd.Timestamp | None:
    """The full "Mon D, YYYY" date that opens MAL's `Aired`, or None."""
    match = FULL_DATE.match(aired or "")
    if not match:
        return None
    return pd.to_datetime(f"{match.group(1)} {match.group(2)} {match.group(3)}", format="%b %d %Y")


def first_year(aired: str) -> float:
    match = ANY_YEAR.search(aired or "")
    return float(match.group(0)) if match else np.nan


def approx_date(aired: str) -> pd.Timestamp | None:
    """Premiere date, falling back to 1 January when only a year is given."""
    exact = premiere_date(aired)
    if exact is not None:
        return exact
    year = first_year(aired)
    return None if np.isnan(year) else pd.Timestamp(year=int(year), month=1, day=1)


def franchise_root(name: str) -> str:
    """Normalised title with season and format markers removed.

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
    """The root itself plus every token-prefix of at least two words."""
    tokens = root.split()
    keys = [root] if root else []
    keys += [" ".join(tokens[:n]) for n in range(2, len(tokens))]
    return keys
