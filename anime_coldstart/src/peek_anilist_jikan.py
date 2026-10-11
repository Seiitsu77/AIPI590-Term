"""Pull a 10-title sample from the AniList and Jikan APIs and test whether they join.

Exploratory look at what the two catalogs expose. Nothing here feeds the analysis.

Sample: ten TV titles from data/cohort.csv, evenly spaced along the Members ranking, so
the look covers obscure titles as well as hits.

AniList (https://graphql.anilist.co, GraphQL, no key for public data). Every Media row
carries `idMal`, the MyAnimeList id, which is the join key. MAL numbers anime and manga
separately, so the query also pins `type: ANIME`; without it `idMal_in` returns the
manga that happens to share an id. The full list of Media fields is read from the schema
by introspection rather than typed in from memory.

Jikan (https://api.jikan.moe/v4, an unofficial REST scraper of MyAnimeList, no key).
One request per title to /anime/{id}/full.

Both APIs return today's values. Scores, member counts, tags and rankings are "as of the
fetch", not "as of the premiere" and not "as of the early-2021 snapshot" in
data/raw/anime.csv. The one exception is AniList's `trends`, a dated daily series, so
its first recorded day is pulled for each title.

Neither API asks for contact details, so none are sent. The run makes 2 AniList requests
and 10 Jikan requests, the latter one a second (Jikan allows 3 a second, 60 a minute).

Usage:
    python src/peek_anilist_jikan.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "api_sample"
OUT = ROOT / "data" / "api_sample"

ANILIST_API = "https://graphql.anilist.co"
JIKAN_API = "https://api.jikan.moe/v4/anime/{mal_id}/full"
HEADERS = {"User-Agent": "AnimeColdStartFeasibility/0.1 (Duke AIPI 590 student project) python-requests"}
JIKAN_PAUSE_SECONDS = 1.0
SAMPLE_SIZE = 10

FIELDS_QUERY = """
{
  __type(name: "Media") {
    fields {
      name
      description
      type { kind name ofType { kind name ofType { kind name ofType { kind name } } } }
    }
  }
}
"""

MEDIA_QUERY = """
query ($ids: [Int]) {
  Page(page: 1, perPage: 50) {
    media(type: ANIME, idMal_in: $ids) {
      id idMal siteUrl
      title { romaji english native }
      synonyms
      format status source countryOfOrigin isAdult isLicensed
      season seasonYear seasonInt
      startDate { year month day }
      endDate { year month day }
      episodes duration
      genres
      tags { name category rank isMediaSpoiler }
      averageScore meanScore popularity favourites trending
      hashtag description updatedAt bannerImage
      coverImage { large color }
      trailer { id site }
      studios { edges { isMain node { name isAnimationStudio } } }
      relations { edges { relationType node { idMal type format } } }
      externalLinks { site type language }
      rankings { rank type allTime year season context }
      stats { scoreDistribution { score amount } statusDistribution { status amount } }
      nextAiringEpisode { episode airingAt }
      trends(sort: DATE, perPage: 3) { nodes { date popularity trending averageScore inProgress releasing episode } }
      staff(sort: RELEVANCE, perPage: 5) { edges { role node { name { full } } } }
      characters(sort: ROLE, perPage: 3) { edges { role node { name { full } } } }
    }
  }
}
"""


def sample_titles() -> pd.DataFrame:
    cohort = pd.read_csv(ROOT / "data" / "cohort.csv")
    tv = cohort[cohort["Type"] == "TV"].sort_values(["Members", "MAL_ID"]).reset_index(drop=True)
    picks = np.linspace(0, len(tv) - 1, SAMPLE_SIZE).round().astype(int)
    sample = tv.iloc[picks][["MAL_ID", "Name", "start", "Members"]]
    return sample.rename(columns={"MAL_ID": "mal_id", "Name": "cohort_name", "start": "cohort_start", "Members": "cohort_members_2021"})


def anilist(query: str, variables: dict | None = None) -> tuple[dict, str | None]:
    """Run one GraphQL query. Returns the data and the rate limit the server reports."""
    for _ in range(6):
        response = requests.post(ANILIST_API, json={"query": query, "variables": variables or {}}, headers=HEADERS, timeout=60)
        if response.status_code == 429 or response.status_code >= 500:
            time.sleep(float(response.headers.get("Retry-After", 30)) + 1.0)
            continue
        body = response.json()
        if response.status_code != 200 or body.get("errors"):
            raise RuntimeError(f"AniList returned {response.status_code}: {json.dumps(body.get('errors'))[:600]}")
        return body["data"], response.headers.get("X-RateLimit-Limit")
    raise RuntimeError(f"AniList request kept failing (last status {response.status_code})")


def jikan(mal_id: int) -> dict | None:
    """One title from Jikan, or None when MyAnimeList has no such id."""
    for _ in range(6):
        response = requests.get(JIKAN_API.format(mal_id=mal_id), headers=HEADERS, timeout=(10, 60))
        if response.status_code == 200:
            return response.json()["data"]
        if response.status_code == 404:
            return None
        if response.status_code == 429 or response.status_code >= 500:
            time.sleep(float(response.headers.get("Retry-After", 4)) + 1.0)
            continue
        response.raise_for_status()
    raise RuntimeError(f"Jikan request kept failing for {mal_id} (last status {response.status_code})")


def render_type(node: dict) -> str:
    if node["kind"] == "NON_NULL":
        return render_type(node["ofType"]) + "!"
    if node["kind"] == "LIST":
        return "[" + render_type(node["ofType"]) + "]"
    return node["name"]


def joined(items) -> str | None:
    text = "; ".join(str(item) for item in items or [])
    return text or None


def fuzzy_date(parts: dict | None) -> str | None:
    """AniList dates are {year, month, day} and any part can be null."""
    if not parts or not parts.get("year"):
        return None
    text = f"{parts['year']:04d}"
    for key in ("month", "day"):
        if not parts.get(key):
            break
        text += f"-{parts[key]:02d}"
    return text


def unix_day(seconds: int | None) -> str | None:
    return datetime.fromtimestamp(seconds, timezone.utc).strftime("%Y-%m-%d") if seconds else None


def flatten_anilist(media: dict) -> dict:
    """One row per title. Nested objects become joined names or compact JSON."""
    row = {key: value for key, value in media.items() if not isinstance(value, (dict, list))}
    row["mal_id"] = row.pop("idMal")
    for key, value in media["title"].items():
        row[f"title_{key}"] = value
    row["synonyms"] = joined(media["synonyms"])
    row["start_date"] = fuzzy_date(media["startDate"])
    row["end_date"] = fuzzy_date(media["endDate"])
    row["genres"] = joined(media["genres"])
    tags = sorted(media["tags"], key=lambda tag: -tag["rank"])
    row["tags"] = joined(f"{tag['name']}:{tag['rank']}" for tag in tags)
    row["n_tags"] = len(tags)
    studios = media["studios"]["edges"]
    row["studios_main"] = joined(edge["node"]["name"] for edge in studios if edge["isMain"])
    row["studios_other"] = joined(edge["node"]["name"] for edge in studios if not edge["isMain"])
    row["relations"] = joined(
        f"{edge['relationType']}:{edge['node']['type']}:{edge['node']['idMal']}" for edge in media["relations"]["edges"]
    )
    row["external_sites"] = joined(sorted({link["site"] for link in media["externalLinks"]}))
    row["rankings"] = joined(
        f"#{rank['rank']} {rank['context']}" + ("" if rank["allTime"] else f" {rank['season'] or ''} {rank['year'] or ''}".rstrip())
        for rank in media["rankings"]
    )
    row["score_distribution"] = json.dumps({item["score"]: item["amount"] for item in media["stats"]["scoreDistribution"]})
    row["status_distribution"] = json.dumps({item["status"]: item["amount"] for item in media["stats"]["statusDistribution"]})
    trends = media["trends"]["nodes"]
    row["trends_first_day"] = unix_day(min(node["date"] for node in trends)) if trends else None
    row["trends_first_rows"] = json.dumps([{**node, "date": unix_day(node["date"])} for node in trends]) if trends else None
    row["cover_image"] = (media["coverImage"] or {}).get("large")
    row["cover_color"] = (media["coverImage"] or {}).get("color")
    row["trailer"] = f"{media['trailer']['site']}:{media['trailer']['id']}" if media["trailer"] else None
    row["next_airing_episode"] = json.dumps(media["nextAiringEpisode"]) if media["nextAiringEpisode"] else None
    row["staff_top"] = joined(f"{edge['role']}={edge['node']['name']['full']}" for edge in media["staff"]["edges"])
    row["characters_top"] = joined(f"{edge['role']}={edge['node']['name']['full']}" for edge in media["characters"]["edges"])
    return row


def flatten_jikan(data: dict) -> dict:
    row = {key: value for key, value in data.items() if not isinstance(value, (dict, list))}
    row["title_synonyms"] = joined(data["title_synonyms"])
    row["aired_from"] = (data["aired"]["from"] or "")[:10] or None
    row["aired_to"] = (data["aired"]["to"] or "")[:10] or None
    row["broadcast"] = data["broadcast"]["string"]
    for key in ("producers", "licensors", "studios", "genres", "explicit_genres", "themes", "demographics"):
        row[key] = joined(item["name"] for item in data[key])
    row["relations"] = joined(
        f"{group['relation']}:{entry['type']}:{entry['mal_id']}" for group in data["relations"] for entry in group["entry"]
    )
    row["external_sites"] = joined(sorted({link["name"] for link in data["external"]}))
    row["streaming_sites"] = joined(sorted({link["name"] for link in data["streaming"]}))
    row["n_openings"] = len(data["theme"]["openings"])
    row["n_endings"] = len(data["theme"]["endings"])
    row["trailer"] = f"youtube:{data['trailer']['youtube_id']}" if data["trailer"].get("youtube_id") else None
    row["image"] = data["images"]["jpg"]["large_image_url"]
    return row


def describe(frame: pd.DataFrame, title: str) -> None:
    print(f"\n{title}: {len(frame)} rows x {frame.shape[1]} columns")
    print(f"  {'column':<26} non-null  example")
    for column in frame.columns:
        values = frame[column].dropna()
        example = " ".join(str(values.iloc[0]).split()) if len(values) else ""
        print(f"  {column:<26} {len(values):>3}/{len(frame):<3}  {example[:78]}")


def norm(value) -> str | None:
    return None if pd.isna(value) else str(value).upper().replace(" ", "_")


def name_set(value) -> set[str]:
    return set() if pd.isna(value) else {part.strip().lower() for part in str(value).split(";")}


def agreement(merged: pd.DataFrame) -> pd.DataFrame:
    """For each concept both APIs carry: how many of the titles have it in both, and agree."""
    pairs = {
        "episodes": (pd.to_numeric(merged["al_episodes"]), pd.to_numeric(merged["jk_episodes"])),
        "start date": (merged["al_start_date"], merged["jk_aired_from"]),
        "format / type": (merged["al_format"].map(norm), merged["jk_type"].map(norm)),
        "source": (merged["al_source"].map(norm), merged["jk_source"].map(norm)),
        "season": (merged["al_season"].map(norm), merged["jk_season"].map(norm)),
        "season year": (pd.to_numeric(merged["al_seasonYear"]), pd.to_numeric(merged["jk_year"])),
        "main studio": (merged["al_studios_main"].map(norm), merged["jk_studios"].map(norm)),
    }
    rows = []
    for concept, (left, right) in pairs.items():
        both = left.notna() & right.notna()
        rows.append({"concept": concept, "in_both": int(both.sum()), "equal": int((left[both] == right[both]).sum())})
    genre_overlap = [
        len(name_set(a) & name_set(b)) / len(name_set(a) | name_set(b))
        for a, b in zip(merged["al_genres"], merged["jk_genres"], strict=True)
        if name_set(a) | name_set(b)
    ]
    rows.append({"concept": "genres (mean Jaccard)", "in_both": len(genre_overlap), "equal": round(float(np.mean(genre_overlap)), 2) if genre_overlap else None})
    return pd.DataFrame(rows)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 34)
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)

    sample = sample_titles()
    ids = [int(mal_id) for mal_id in sample["mal_id"]]

    schema, _ = anilist(FIELDS_QUERY)
    (RAW / "anilist_media_fields.json").write_text(json.dumps(schema, ensure_ascii=False, indent=1), encoding="utf-8")
    data, rate_limit = anilist(MEDIA_QUERY, {"ids": ids})
    (RAW / "anilist_media.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    media = data["Page"]["media"]

    requested = set(media[0]) if media else set()
    fields = pd.DataFrame(
        {
            "field": field["name"],
            "type": render_type(field["type"]),
            "requested": field["name"] in requested,
            "description": " ".join((field["description"] or "").split()),
        }
        for field in schema["__type"]["fields"]
    )
    fields.to_csv(OUT / "anilist_media_fields.csv", index=False)

    al = pd.DataFrame(flatten_anilist(item) for item in media)
    al.to_csv(OUT / "anilist.csv", index=False)

    print(f"AniList Media type: {len(fields)} fields in the schema, {int(fields['requested'].sum())} requested here")
    print(fields[["field", "type", "requested"]].to_string(index=False))
    describe(al, "AniList, flattened")
    back = sample.merge(al[["mal_id", "id", "title_romaji", "start_date"]], on="mal_id", how="left", validate="one_to_one")
    print(f"\nAniList idMal against the cohort's MAL_ID: {int(back['id'].notna().sum())} of {len(sample)} found")
    print(back.rename(columns={"id": "anilist_id"}).to_string(index=False), flush=True)

    # Jikan goes last. It is the less reliable host, and everything above stands without it.
    jikan_rows = {}
    for mal_id in ids:
        jikan_rows[mal_id] = jikan(mal_id)
        time.sleep(JIKAN_PAUSE_SECONDS)
    (RAW / "jikan_anime_full.json").write_text(json.dumps(jikan_rows, ensure_ascii=False, indent=1), encoding="utf-8")
    found = [row for row in jikan_rows.values() if row is not None]
    jk = pd.DataFrame(flatten_jikan(item) for item in found)
    jk.to_csv(OUT / "jikan.csv", index=False)

    merged = (
        sample.merge(al.add_prefix("al_").rename(columns={"al_mal_id": "mal_id"}), on="mal_id", how="left", validate="one_to_one", indicator="in_anilist")
        .merge(jk.add_prefix("jk_").rename(columns={"jk_mal_id": "mal_id"}), on="mal_id", how="left", validate="one_to_one", indicator="in_jikan")
    )  # fmt: skip
    for column in ("in_anilist", "in_jikan"):
        merged[column] = merged[column] == "both"
    merged.to_csv(OUT / "merged.csv", index=False)
    checks = agreement(merged)

    if found:
        print("\nJikan /anime/{id}/full top-level keys:")
        print("  " + ", ".join(f"{key} ({type(value).__name__})" for key, value in found[0].items()))
    describe(jk, "Jikan, flattened")

    print(f"\nJoin on AniList idMal = Jikan mal_id, {len(sample)} sampled titles")
    print(f"  found in AniList: {int(merged['in_anilist'].sum())}   found in Jikan: {int(merged['in_jikan'].sum())}   in both: {int((merged['in_anilist'] & merged['in_jikan']).sum())}")
    print(f"  merged table: {merged.shape[0]} rows x {merged.shape[1]} columns")
    side_by_side = [
        ["mal_id", "cohort_name", "al_title_romaji", "jk_title"],
        ["mal_id", "al_format", "jk_type", "al_episodes", "jk_episodes", "al_start_date", "jk_aired_from", "al_source", "jk_source"],
        ["mal_id", "al_season", "al_seasonYear", "jk_season", "jk_year", "al_studios_main", "jk_studios"],
        ["mal_id", "al_averageScore", "jk_score", "al_popularity", "jk_members", "cohort_members_2021", "al_favourites", "jk_favorites", "al_trends_first_day"],
        ["mal_id", "al_genres", "jk_genres", "jk_themes", "jk_demographics"],
    ]  # fmt: skip
    for columns in side_by_side:
        print()
        print(merged[columns].to_string(index=False))
    print("\nAgreement on shared concepts (after upper-casing and replacing spaces):")
    print(checks.to_string(index=False))

    manifest = {
        "anilist_endpoint": ANILIST_API,
        "jikan_endpoint": JIKAN_API,
        "sample_mal_ids": ids,
        "anilist_rows": len(media),
        "jikan_rows": len(found),
        "anilist_rate_limit_header": rate_limit,
        "python": sys.version.split()[0],
        "started_utc": started.isoformat(timespec="seconds"),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (RAW / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\n" + json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
