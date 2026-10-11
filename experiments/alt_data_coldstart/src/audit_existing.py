"""Task 1: audit the 10-title AniList sample that already exists.

Inputs are the two CSVs written by anime_coldstart/src/peek_anilist_jikan.py:

  anilist.csv               one row per title, 48 columns, nested values already flattened
  anilist_media_fields.csv  one row per field of AniList's Media type, from introspection

anilist.csv is a flattened view. Where the raw response is still on disk
(anime_coldstart/data/raw/api_sample/anilist_media.json, not in git) the nested shapes
and element counts are read from it. Three lists were cut short by that script's own
query, not by AniList: staff to 5, characters to 3 and trends to 3. A count that sits at
its cap says nothing about how many exist.

The leakage grade per column is a judgement, written down in LEAKAGE below with its
reason so it can be argued with. The API returns today's value for everything except
`trends`; it has no edit history, so "medium" means "probably stable, cannot be checked".

Outputs (outputs/stage1_10titles/): field_coverage.csv, id_match_audit.csv,
existing_data_audit.json.

Usage:
    python src/audit_existing.py
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
SAMPLE = REPO / "anime_coldstart" / "data" / "api_sample"
RAW_MEDIA = REPO / "anime_coldstart" / "data" / "raw" / "api_sample" / "anilist_media.json"
MAL_CATALOG = REPO / "anime_coldstart" / "data" / "raw" / "anime.csv"
RATING_COUNTS = ROOT / "outputs" / "coldstart_eval" / "item_rating_counts.parquet"
OUT = ROOT / "outputs" / "stage1_10titles"

FULL_DATE = re.compile(r"^([A-Z][a-z]{2}) (\d{1,2}), (\d{4})")
LIST_COLUMNS = ("synonyms", "genres", "tags", "studios_main", "studios_other", "relations", "external_sites", "rankings", "staff_top", "characters_top")
JSON_COLUMNS = ("score_distribution", "status_distribution", "trends_first_rows", "next_airing_episode")
CAPPED = {"staff_top": 5, "characters_top": 3, "trends_first_rows": 3}

# (grade, reason). "dated" marks the only field that carries its own observation date.
LEAKAGE = {
    "id": ("none", "identifier"),
    "mal_id": ("none", "identifier, the join key"),
    "siteUrl": ("none", "identifier"),
    "title_romaji": ("none", "identifier"),
    "title_native": ("none", "identifier"),
    "title_english": ("low", "an official English title often arrives with licensing, sometimes after the premiere"),
    "synonyms": ("low", "aliases accumulate over time"),
    "format": ("low", "announced before airing"),
    "source": ("low", "announced before airing"),
    "countryOfOrigin": ("low", "fixed"),
    "isAdult": ("low", "fixed"),
    "season": ("low", "announced before airing"),
    "seasonYear": ("low", "announced before airing"),
    "seasonInt": ("low", "announced before airing"),
    "start_date": ("low", "the premiere itself; defines the cutoff"),
    "genres": ("low", "editorial, usually set at announcement"),
    "studios_main": ("low", "announced before airing"),
    "studios_other": ("low", "producers are usually announced before airing; the list can grow"),
    "hashtag": ("low", "promotional, set before airing"),
    "trailer": ("low", "promotional; the link can be replaced by a later trailer"),
    "cover_image": ("low", "can be replaced after airing"),
    "cover_color": ("low", "derived from the cover image"),
    "bannerImage": ("low", "often added after airing"),
    "duration": ("low", "usually announced; today's value"),
    "episodes": ("medium", "final episode count; for a long or extended run it is only known afterwards"),
    "description": ("medium", "today's text; synopses are rewritten and can describe the whole run"),
    "staff_top": ("medium", "main staff are announced early, but credits keep being added"),
    "characters_top": ("medium", "cast list keeps being added"),
    "isLicensed": ("medium", "licensing can be decided after the premiere"),
    "external_sites": ("medium", "streaming links are added when a platform picks the title up"),
    "end_date": ("medium", "known in advance only when the run length was announced"),
    "relations": ("high", "includes SEQUEL and SIDE_STORY links to works made later; that a sequel exists is an outcome"),
    "tags": ("high", "community-voted; ranks and many tags come from people who watched it"),
    "n_tags": ("high", "grows with attention after airing"),
    "status": ("high", "FINISHED / RELEASING is today's state"),
    "averageScore": ("high", "ratings exist only after airing; this is the outcome"),
    "meanScore": ("high", "ratings exist only after airing; this is the outcome"),
    "popularity": ("high", "today's count of users with the title on a list; this is the outcome"),
    "favourites": ("high", "today's count; accumulates after airing"),
    "trending": ("high", "today's activity"),
    "rankings": ("high", "computed from score and popularity"),
    "score_distribution": ("high", "today's rating histogram"),
    "status_distribution": ("high", "today's watching-status counts"),
    "updatedAt": ("high", "time of the last edit to the entry; not a pre-release fact"),
    "next_airing_episode": ("high", "today's schedule state"),
    "nextAiringEpisode": ("high", "today's schedule state; an all-null duplicate of next_airing_episode left behind by the flattening"),
    "trends_first_day": ("dated", "first recorded day; a proxy for when the entry was created"),
    "trends_first_rows": ("dated", "dated rows; safe only when cut at the prediction cutoff"),
}


def premiere(aired: str) -> str | None:
    match = FULL_DATE.match(aired or "")
    if not match:
        return None
    return pd.to_datetime(f"{match.group(1)} {match.group(2)} {match.group(3)}", format="%b %d %Y").strftime("%Y-%m-%d")


def squash(text) -> str:
    """Lower-case letters and digits only, for a like-for-like title comparison."""
    return re.sub(r"[^a-z0-9]+", "", unicodedata.normalize("NFKC", str(text)).lower())


def value_kind(column: str, series: pd.Series) -> str:
    if column in JSON_COLUMNS:
        values = series.dropna()
        return "JSON array" if len(values) and str(values.iloc[0]).startswith("[") else "JSON object"
    if column in LIST_COLUMNS:
        return "joined list"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        values = series.dropna()
        return "integer" if len(values) and (values % 1 == 0).all() else "number"
    return "text"


def element_count(column: str, value) -> float:
    if pd.isna(value):
        return 0
    if column in JSON_COLUMNS:
        return len(json.loads(value))
    return len(str(value).split("; "))


def coverage(frame: pd.DataFrame, file: str) -> pd.DataFrame:
    rows = []
    for column in frame.columns:
        series = frame[column]
        present = series.dropna()
        row = {
            "file": file,
            "column": column,
            "pandas_dtype": str(series.dtype),
            "value_kind": value_kind(column, series),
            "n_rows": len(series),
            "n_non_null": len(present),
            "missing_rate": round(1 - len(present) / len(series), 3),
            "n_unique": int(present.astype(str).nunique()),
            "example": " ".join(str(present.iloc[0]).split())[:80] if len(present) else "",
        }
        if column in LIST_COLUMNS + JSON_COLUMNS:
            counts = series.map(lambda value, column=column: element_count(column, value))
            row.update(elements_min=int(counts.min()), elements_median=float(counts.median()), elements_max=int(counts.max()))
            row["capped_by_query_at"] = CAPPED.get(column)
        if file == "anilist.csv":
            row["leakage_risk"], row["leakage_reason"] = LEAKAGE.get(column, ("unclassified", ""))
        rows.append(row)
    return pd.DataFrame(rows)


def shape(value, depth: int = 0) -> str:
    """Compact description of a nested JSON value, e.g. {edges: [{isMain, node: {name}}]}."""
    if isinstance(value, dict):
        return "{" + ", ".join(key if not isinstance(inner, (dict, list)) else f"{key}: {shape(inner, depth + 1)}" for key, inner in value.items()) + "}"
    if isinstance(value, list):
        return "[" + (shape(value[0], depth + 1) if value else "") + "]"
    return type(value).__name__


def nested_audit(media: list[dict]) -> list[dict]:
    """Element counts per title for every nested Media field in the raw response."""
    counters = {
        "title": lambda m: sum(value is not None for value in m["title"].values()),
        "synonyms": lambda m: len(m["synonyms"]),
        "genres": lambda m: len(m["genres"]),
        "tags": lambda m: len(m["tags"]),
        "studios": lambda m: len(m["studios"]["edges"]),
        "relations": lambda m: len(m["relations"]["edges"]),
        "staff": lambda m: len(m["staff"]["edges"]),
        "characters": lambda m: len(m["characters"]["edges"]),
        "externalLinks": lambda m: len(m["externalLinks"]),
        "rankings": lambda m: len(m["rankings"]),
        "trends": lambda m: len(m["trends"]["nodes"]),
        "stats.scoreDistribution": lambda m: len(m["stats"]["scoreDistribution"]),
        "stats.statusDistribution": lambda m: len(m["stats"]["statusDistribution"]),
    }
    caps = {"staff": 5, "characters": 3, "trends": 3}
    rows = []
    for field, counter in counters.items():
        counts = np.array([counter(item) for item in media])
        example = next((item for item in media if counter(item)), media[0])
        value = example["stats"][field.split(".")[1]] if "." in field else example[field]
        rows.append(
            {
                "field": field,
                "shape": shape(value),
                "titles_with_at_least_one": int((counts > 0).sum()),
                "elements_min": int(counts.min()),
                "elements_median": float(np.median(counts)),
                "elements_max": int(counts.max()),
                "capped_by_query_at": caps.get(field),
                "titles_at_cap": int((counts == caps[field]).sum()) if field in caps else None,
            }
        )
    return rows


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 60)
    OUT.mkdir(parents=True, exist_ok=True)

    al = pd.read_csv(SAMPLE / "anilist.csv")
    fields = pd.read_csv(SAMPLE / "anilist_media_fields.csv")
    field_coverage = pd.concat([coverage(al, "anilist.csv"), coverage(fields, "anilist_media_fields.csv")], ignore_index=True)
    field_coverage.to_csv(OUT / "field_coverage.csv", index=False)

    # ID audit: exact idMal = MAL_ID only. Titles are compared, never used to match.
    mal = pd.read_csv(MAL_CATALOG)
    mal["premiere_mal"] = mal["Aired"].map(premiere)
    ids = al[["id", "mal_id", "title_romaji", "format", "episodes", "start_date"]].rename(columns={"id": "anilist_id", "start_date": "start_date_anilist"})
    ids["anilist_rows_with_this_mal_id"] = ids.groupby("mal_id")["anilist_id"].transform("size")
    ids = ids.merge(mal[["MAL_ID", "Name", "Type", "Episodes", "premiere_mal"]], left_on="mal_id", right_on="MAL_ID", how="left", validate="one_to_one")
    ids["in_mal_catalog"] = ids["MAL_ID"].notna()
    ids["title_equal_ignoring_case_and_punctuation"] = ids["title_romaji"].map(squash) == ids["Name"].map(squash)
    ids["start_date_equal"] = ids["start_date_anilist"] == ids["premiere_mal"]
    ids["episodes_equal"] = ids["episodes"].astype(str) == ids["Episodes"].astype(str)
    ids["format_equal"] = ids["format"].str.upper() == ids["Type"].str.upper()
    if RATING_COUNTS.exists():
        counts = pd.read_parquet(RATING_COUNTS)[["anime_id", "in_serving_catalog", "n_ratings", "n_pos"]]
        ids = ids.merge(counts, left_on="mal_id", right_on="anime_id", how="left").drop(columns="anime_id")
        ids["in_ratings"] = ids["n_ratings"].fillna(0) > 0
    ids["match_method"] = "idMal exact"
    ids = ids.drop(columns="MAL_ID").rename(columns={"Name": "title_mal", "Type": "type_mal", "Episodes": "episodes_mal", "format": "format_anilist", "episodes": "episodes_anilist"})
    ids.to_csv(OUT / "id_match_audit.csv", index=False)

    trend_rows = [dict(row, mal_id=int(mal_id)) for mal_id, cell in zip(al["mal_id"], al["trends_first_rows"], strict=True) if pd.notna(cell) for row in json.loads(cell)]
    trends = pd.DataFrame(trend_rows)
    per_title = trends.groupby("mal_id")["date"].agg(["size", "min", "max"])
    summary = {
        "anilist_csv": {
            "rows": len(al),
            "columns": al.shape[1],
            "unique_anilist_id": int(al["id"].nunique()),
            "unique_mal_id": int(al["mal_id"].nunique()),
            "null_mal_id": int(al["mal_id"].isna().sum()),
            "columns_fully_populated": int((al.notna().all()).sum()),
            "columns_with_missing": {column: int(al[column].isna().sum()) for column in al.columns if al[column].isna().any()},
            "value_kinds": field_coverage[field_coverage["file"] == "anilist.csv"]["value_kind"].value_counts().to_dict(),
            "leakage_grades": field_coverage[field_coverage["file"] == "anilist.csv"]["leakage_risk"].value_counts().to_dict(),
            "description_chars": {"min": int(al["description"].str.len().min()), "median": float(al["description"].str.len().median()), "max": int(al["description"].str.len().max())},
            "descriptions_containing_html": int(al["description"].str.contains(r"<[a-zA-Z/]", regex=True).sum()),
        },
        "media_fields_csv": {
            "rows": len(fields),
            "unique_field": int(fields["field"].nunique()),
            "requested": int(fields["requested"].sum()),
            "not_requested": sorted(fields.loc[~fields["requested"], "field"]),
            "missing_description": int(fields["description"].isna().sum()),
            "list_or_connection_fields": int(fields["type"].str.contains(r"\[|Connection", regex=True).sum()),
        },
        "id_match": {
            "titles": len(ids),
            "found_in_mal_catalog": int(ids["in_mal_catalog"].sum()),
            "one_to_one": bool((ids["anilist_rows_with_this_mal_id"] == 1).all()),
            "start_date_equal": int(ids["start_date_equal"].sum()),
            "episodes_equal": int(ids["episodes_equal"].sum()),
            "format_equal": int(ids["format_equal"].sum()),
            "title_equal_ignoring_case_and_punctuation": int(ids["title_equal_ignoring_case_and_punctuation"].sum()),
            "found_in_ratings": int(ids["in_ratings"].sum()) if "in_ratings" in ids else None,
            "found_in_serving_catalog": int(ids["in_serving_catalog"].fillna(False).sum()) if "in_serving_catalog" in ids else None,
        },
        "trends_first_rows": {
            "titles_with_rows": int(len(per_title)),
            "rows_total": int(len(trends)),
            "rows_per_title_min": int(per_title["size"].min()),
            "rows_per_title_max": int(per_title["size"].max()),
            "earliest_date_in_file": per_title["min"].min(),
            "latest_date_in_file": per_title["max"].max(),
            "non_null_by_field": {column: int(trends[column].notna().sum()) for column in ("trending", "popularity", "inProgress", "averageScore", "episode")},
            "limit_comes_from": "the query in peek_anilist_jikan.py: trends(sort: DATE, perPage: 3)",
            "date_note": "dates in this column are UTC calendar days; AniList stamps each row at 15:00 UTC, which is 00:00 in Japan on the next day",
        },
        "raw_response_available": RAW_MEDIA.exists(),
    }
    if RAW_MEDIA.exists():
        summary["nested_fields"] = nested_audit(json.loads(RAW_MEDIA.read_text(encoding="utf-8"))["Page"]["media"])
    (OUT / "existing_data_audit.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(field_coverage.drop(columns=["example", "leakage_reason"]).to_string(index=False))
    print()
    print(ids.to_string(index=False))
    print()
    print(json.dumps({key: value for key, value in summary.items() if key != "nested_fields"}, indent=1, ensure_ascii=False))
    if "nested_fields" in summary:
        print(pd.DataFrame(summary["nested_fields"]).to_string(index=False))


if __name__ == "__main__":
    main()
