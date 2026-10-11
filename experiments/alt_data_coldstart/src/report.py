"""Render data_quality_report.md and collection_log.json for one stage.

Every number in the report is read from the audit outputs, the request log or the
ratings summary. None is typed by hand, so the prose cannot drift from the files.

The stage recommendation follows a rule written here, in `recommend`, so it can be read
and disputed. It was written after the first trend pages had been looked at and before
the full audit ran, and it was changed once afterwards: the check "no day appears twice"
became "no day appears twice with different values" when the dry run turned up a handful
of days that AniList returns as two identical rows. Identical copies are dropped without
loss, so they do not block collection; the report states the change and the counts.

Usage:
    python src/report.py --stage stage1_10titles [--split-metadata <Anime Compass split json>]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from anilist_client import ENDPOINT, REQUEST_LOG, REQUESTS_PER_MINUTE

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
SCALES = (100, 1000, 3000, 5000)
METADATA_BATCH = 50
CONTENT_FIELDS = ("genres", "tags", "studios", "staff", "relations")


def table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join("" if cell is None or (isinstance(cell, float) and math.isnan(cell)) else str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def number(value) -> str:
    return "" if value is None or pd.isna(value) else f"{int(value):,}"


def collection_log() -> dict:
    entries = [json.loads(line) for line in REQUEST_LOG.read_text(encoding="utf-8").splitlines()]
    http = [entry for entry in entries if entry["outcome"] != "cache_hit"]
    outcomes = Counter(entry["outcome"] for entry in http)
    stamps = np.array(sorted(pd.Timestamp(entry["utc"]).timestamp() for entry in http))
    gaps = np.diff(stamps)
    active = float(gaps[gaps < 60].sum())  # pauses between script runs are not collection time
    remaining = [int(entry["remaining"]) for entry in http if entry.get("remaining") is not None]
    return {
        "endpoint": ENDPOINT,
        "configured_requests_per_minute": REQUESTS_PER_MINUTE,
        "http_requests": len(http),
        "outcomes": dict(outcomes),
        "succeeded": outcomes["ok"],
        "failed_or_retried": len(http) - outcomes["ok"],
        "success_rate": round(outcomes["ok"] / len(http), 4),
        "http_429": outcomes["rate_limited"],
        "requests_by_kind": dict(Counter(entry["name"].split("/")[-1].split("_")[0] for entry in http if entry["outcome"] == "ok")),
        "cache_hits": sum(entry["outcome"] == "cache_hit" for entry in entries),
        "first_request_utc": min(entry["utc"] for entry in http),
        "last_request_utc": max(entry["utc"] for entry in http),
        "active_minutes": round(active / 60, 1),
        "observed_requests_per_minute": round((len(http) - 1) / (active / 60), 1) if active else None,
        "rate_limit_header_values": sorted({entry["limit"] for entry in http if entry.get("limit")}),
        "lowest_remaining_header": min(remaining) if remaining else None,
        "median_response_seconds": float(np.median([entry["seconds"] for entry in http if entry.get("seconds") is not None])),
        "terms_of_use": "https://docs.anilist.co/guide/terms-of-use (read 2026-10-07): free for non-commercial use; hoarding or mass collection prohibited, lenient for purely educational projects",
    }


def recommend(trends: dict, log: dict) -> tuple[str, list[str]]:
    """GO / CONDITIONAL GO / NO-GO for collecting the next, larger stage."""
    titles, dated = trends["titles"], trends["titles_with_premiere_day"]
    mechanics = {
        "paging the history reaches the newest row for every title": trends["paging_reaches_newest_row"] == titles,
        "the date filter returns exactly the rows the full history has": trends["date_filter_matches_full_history"] == dated,
        "no day appears twice with different values": trends["conflicting_duplicate_days"] == 0,
        "every request succeeded without a 429": log["failed_or_retried"] == 0,
    }
    coverage = trends["titles_with_rows_in_window"]["30d"] / dated if dated else 0.0
    reasons = [f"{'holds' if passed else 'FAILS'}: {check}" for check, passed in mechanics.items()]
    reasons.append(f"{trends['titles_with_rows_in_window']['30d']} of {dated} titles have at least one record in the 30 days before the premiere")
    if not all(list(mechanics.values())[:3]):
        return "NO-GO", reasons
    if coverage < 0.5 or not mechanics["every request succeeded without a 429"]:
        return "CONDITIONAL GO", reasons
    return "GO", reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--split-metadata", type=Path, help="Anime Compass split json, for its train-positive count")
    args = parser.parse_args()
    stage = OUTPUTS / args.stage

    existing = json.loads((stage / "existing_data_audit.json").read_text(encoding="utf-8"))
    coverage = pd.read_csv(stage / "field_coverage.csv")
    ids = pd.read_csv(stage / "id_match_audit.csv")
    audit = pd.read_csv(stage / "historical_signal_audit.csv")
    features = pd.read_csv(stage / "prerelease_features.csv")
    trends = json.loads((stage / "trends_audit.json").read_text(encoding="utf-8"))
    window = pd.read_parquet(stage / "trends_window.parquet")
    ratings = json.loads((OUTPUTS / "coldstart_eval" / "ratings_summary.json").read_text(encoding="utf-8"))
    scenarios = pd.read_csv(OUTPUTS / "coldstart_eval" / "cold_item_scenarios.csv")
    log = collection_log()
    (stage / "collection_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    verdict, reasons = recommend(trends, log)

    sample = existing["anilist_csv"]
    schema = existing["media_fields_csv"]
    match = existing["id_match"]
    first_rows = existing["trends_first_rows"]
    al_cover = coverage[coverage["file"] == "anilist.csv"]
    dated = audit[audit["premiere_day_known"]]
    out: list[str] = []
    add = out.append

    add(f"# Data feasibility pilot, stage 1: the {sample['rows']}-title sample\n")
    add(f"Generated by `src/report.py` at {datetime.now(timezone.utc).isoformat(timespec='seconds')}. Every number is read from the files in `outputs/{args.stage}/` and `outputs/coldstart_eval/`; none is typed by hand.\n")
    add(f"**Recommendation for collecting the 100-title pilot: {verdict}.** The rule is in `recommend()` in `src/report.py`.\n")
    add("\n".join(f"- {reason}" for reason in reasons) + "\n")
    add(
        f"The duplicate check was narrowed after the first look at the data. As first written it failed on any repeated day; {trends['duplicate_days']} repeated days were found, "
        f"{trends['duplicate_days'] - trends['conflicting_duplicate_days']} of them exact copies, so it now fails only when the two rows disagree ({trends['conflicting_duplicate_days']} cases).\n"
    )

    add("## 1. Scope\n")
    add(
        f"- **Sample.** {sample['rows']} TV titles that premiered between 2016 and 2020, picked evenly along the MAL member ranking by `anime_coldstart/src/peek_anilist_jikan.py`. It is not a random sample and it has no films, OVAs or ONAs chosen as such.\n"
        f"- **Trend history.** Collected {log['first_request_utc']} to {log['last_request_utc']} (UTC) from `{ENDPOINT}`.\n"
        "- **Not done.** No model was trained. Nothing beyond these titles was downloaded. Jikan data is absent: `api.jikan.moe` was unreachable on 2026-10-07 and, according to the project owner, the service has shut down.\n"
        "- **What the trend data is.** Activity on AniList by AniList users. It is not Wikipedia pageviews and not search volume, and a result on one says nothing about the other.\n"
    )

    add("## 2. The existing files (Task 1)\n")
    add("### 2.1 Schema and identifiers\n")
    add(
        table(
            ["File", "Rows", "Columns", "Key", "Unique keys"],
            [
                ["`anilist.csv`", sample["rows"], sample["columns"], "`id` / `mal_id`", f"{sample['unique_anilist_id']} / {sample['unique_mal_id']}"],
                ["`anilist_media_fields.csv`", schema["rows"], 4, "`field`", schema["unique_field"]],
            ],
        )
        + "\n"
    )
    kinds = ", ".join(f"{count} {kind}" for kind, count in sample["value_kinds"].items())
    add(
        f"`anilist.csv` is a flattened view of the API response: {kinds}. `{sample['columns_fully_populated']}` of its {sample['columns']} columns have no missing value. "
        f"`anilist_media_fields.csv` lists the {schema['rows']} fields of AniList's `Media` type; {schema['requested']} were requested, {schema['rows'] - schema['requested']} were not "
        f"({', '.join('`' + name + '`' for name in schema['not_requested'])}). {schema['list_or_connection_fields']} of the {schema['rows']} are lists or paged connections. "
        "Per-column types, missing rates and examples are in `field_coverage.csv`.\n"
    )
    add("Columns with missing values:\n")
    add(table(["Column", "Missing", "Of"], [[f"`{column}`", count, sample["rows"]] for column, count in sample["columns_with_missing"].items()]) + "\n")
    add("`nextAiringEpisode` and `next_airing_episode` are the same field twice, both empty; the first is a leftover of the flattening.\n")

    add("### 2.2 AniList id against MAL id\n")
    add(
        f"Matching is `idMal` = `MAL_ID`, exact, with no title matching anywhere. {match['found_in_mal_catalog']} of {match['titles']} titles are in the MAL catalog, "
        f"{match['found_in_serving_catalog']} in the Anime Compass serving catalog and {match['found_in_ratings']} in `rating_complete.csv`. "
        f"The mapping is one-to-one: {'yes' if match['one_to_one'] else 'NO'}; `mal_id` is null in {sample['null_mal_id']} rows. "
        f"Premiere dates agree in {match['start_date_equal']}, episode counts in {match['episodes_equal']}, format in {match['format_equal']}, "
        f"and the title, ignoring case and punctuation, in {match['title_equal_ignoring_case_and_punctuation']}. Titles that differ are the reason not to match on them.\n"
    )
    add(
        table(
            ["AniList id", "MAL id", "AniList title", "MAL title", "Premiere equal", "Format (AniList / MAL)", "Ratings", "Positives (>= 8)"],
            [
                [row.anilist_id, row.mal_id, row.title_romaji, row.title_mal, "yes" if row.start_date_equal else "NO", f"{row.format_anilist} / {row.type_mal}", number(row.n_ratings), number(row.n_pos)]
                for row in ids.itertuples()
            ],
        )
        + "\n"
    )

    add("### 2.3 Completeness of the content fields\n")
    nested = pd.DataFrame(existing.get("nested_fields", []))
    rows = [["`description`", f"{int(al_cover.loc[al_cover['column'] == 'description', 'n_non_null'].iloc[0])} of {sample['rows']}", f"{sample['description_chars']['min']} to {sample['description_chars']['max']} characters, median {sample['description_chars']['median']:.0f}", ""]]
    for item in nested.itertuples():
        if item.field in CONTENT_FIELDS:
            cap = "" if pd.isna(item.capped_by_query_at) else f"cut at {int(item.capped_by_query_at)} by the query; {int(item.titles_at_cap)} titles sit at the cap"
            rows.append([f"`{item.field}`", f"{item.titles_with_at_least_one} of {sample['rows']}", f"{item.elements_min} to {item.elements_max} per title, median {item.elements_median:g}", cap])
    add(table(["Field", "Titles with a value", "Size", "Caveat"], rows) + "\n")
    add(
        f"{sample['descriptions_containing_html']} of {sample['rows']} descriptions contain HTML tags and need stripping before any text model. "
        "`staff` cannot be judged for completeness from this file: the query asked for five. The same holds for `characters` (three) and `trends` (three).\n"
    )
    if len(nested):
        add("Nested shapes in the raw response:\n")
        add(table(["Field", "Shape"], [[f"`{item.field}`", f"`{item.shape}`"] for item in nested.itertuples()]) + "\n")

    add("### 2.4 `trends_first_rows`\n")
    add(
        f"{first_rows['rows_total']} rows over {first_rows['titles_with_rows']} titles, {first_rows['rows_per_title_min']} to {first_rows['rows_per_title_max']} per title, dated {first_rows['earliest_date_in_file']} to {first_rows['latest_date_in_file']}. "
        f"It is the first three rows and only those: the limit comes from {first_rows['limit_comes_from']}. "
        + ", ".join(f"`{column}` is filled in {count}" for column, count in first_rows["non_null_by_field"].items())
        + f" of the {first_rows['rows_total']} rows. One correction: {first_rows['date_note']}, so each date in that column is one day early.\n"
    )

    add("### 2.5 Columns that would leak\n")
    add("Everything except `trends` is today's value. The API has no edit history, so a pre-release value cannot be recovered for any of them.\n")
    order = ["high", "medium", "dated", "low", "none"]
    add(table(["Risk", "Columns", "Count"], [[grade, ", ".join(f"`{column}`" for column in al_cover.loc[al_cover["leakage_risk"] == grade, "column"]), int((al_cover["leakage_risk"] == grade).sum())] for grade in order]) + "\n")
    add("Reasons per column are in `field_coverage.csv`. The grades are judgements: `high` is an outcome or accumulates after airing, `medium` is usually set before airing but editable with no way to check, `dated` carries its own observation day.\n")

    add("## 3. MediaTrend history (Task 2)\n")
    add("### 3.1 Can it be queried by date and paged to the end?\n")
    add(
        f"- **Paging.** {trends['rows']:,} rows in {trends['pages']} pages of 50 for {trends['titles']} titles. Walking the history upwards ends on the row the API returns as newest for {trends['paging_reaches_newest_row']} of {trends['titles']} titles.\n"
        f"- **Date filter.** `date_greater` / `date_lesser` returned exactly the rows the full history holds in the same range for {trends['date_filter_matches_full_history']} of {trends['titles_with_premiere_day']} titles.\n"
        f"- **`pageInfo.total` is not a count.** It reported {', '.join(str(value) for value in sorted(audit['reported_total_page1'].unique()))} on first pages while the real row counts ran from {audit['rows'].min():,} to {audit['rows'].max():,}. Only `hasNextPage` can be trusted.\n"
        f"- **Timestamps.** All rows sit on 15:00 UTC, which is 00:00 in Japan: {'yes' if trends['all_timestamps_on_jst_midnight'] else 'NO'}.\n"
    )

    add("### 3.2 First and last recorded day\n")
    add(
        table(
            ["AniList id", "Title", "Premiere", "First day", "Last day", "Rows", "Days recorded", "Longest gap (days)"],
            [[row.anilist_id, row.title, row.premiere, row.first_day, row.last_day, f"{row.rows:,}", f"{row.share_of_days_recorded:.0%}", row.longest_gap_days] for row in audit.itertuples()],
        )
        + "\n"
    )
    add("Days are Japan days. \"Days recorded\" is the share of calendar days between the first and last row that have a row.\n")

    add("### 3.3 Records before the premiere\n")
    add(
        table(
            ["AniList id", "First row to premiere (days)", "Rows in 90 d", "Rows in 30 d", "Rows in 7 d", "With popularity 90 / 30 / 7", "With inProgress 90 / 30 / 7"],
            [
                [
                    row.anilist_id,
                    int(row.days_from_first_row_to_premiere),
                    int(row.pre90_rows),
                    int(row.pre30_rows),
                    int(row.pre7_rows),
                    f"{int(row.pre90_rows_with_popularity)} / {int(row.pre30_rows_with_popularity)} / {int(row.pre7_rows_with_popularity)}",
                    f"{int(row.pre90_rows_with_inProgress)} / {int(row.pre30_rows_with_inProgress)} / {int(row.pre7_rows_with_inProgress)}",
                ]
                for row in dated.itertuples()
            ],
        )
        + "\n"
    )
    in_window = trends["titles_with_rows_in_window"]
    with_popularity = trends["titles_with_popularity_in_window"]
    add(
        f"Windows are Japan days D-90 to D-1, D-30 to D-1 and D-7 to D-1 for premiere day D. A negative first column means the AniList entry was created after the premiere. "
        f"Titles with at least one row: {in_window['90d']} (90 d), {in_window['30d']} (30 d), {in_window['7d']} (7 d) of {trends['titles_with_premiere_day']}. "
        f"Titles with at least one `popularity` value: {with_popularity['90d']}, {with_popularity['30d']}, {with_popularity['7d']}.\n"
    )

    add("### 3.4 Coverage and density of each signal\n")
    add(
        table(
            ["AniList id", "Rows", "trending", "popularity", "inProgress", "averageScore > 0", "First day with popularity", "First day with inProgress"],
            [
                [row.anilist_id, f"{row.rows:,}", f"{row.share_rows_with_trending:.0%}", f"{row.share_rows_with_popularity:.0%}", f"{row.share_rows_with_inProgress:.0%}", f"{row.share_rows_with_averageScore:.0%}", row.first_day_with_popularity, row.first_day_with_inProgress]
                for row in audit.itertuples()
            ],
        )
        + "\n"
    )
    add(
        f"`trending` is present in every row; it is 0 in {trends['rows_with_zero_trending']} of the {trends['rows']:,} rows and at least 1 in the rest. A day with no row is read here as a day on which nothing happened, not as a lost measurement. "
        f"That is an inference: AniList does not document when a row is written, and the {trends['rows_with_zero_trending']} zero rows show that a row can exist without any trending activity. "
        f"`popularity` first appears on {trends['earliest_day_with_popularity']} and `inProgress` on {trends['earliest_day_with_inProgress']} across these titles; earlier rows carry `trending` alone. "
        "Titles whose history starts before those days all switch on within a day of each other, which points to when AniList began recording the field rather than to anything about the title. "
        "`averageScore` is 0, not null, until a title has a score.\n"
    )

    add("### 3.5 Duplicates, gaps and odd movements\n")
    add(
        table(
            ["AniList id", "Duplicate days", "Missing days", "Popularity decreases", "Largest one-step drop", "Trending median", "Trending max"],
            [[row.anilist_id, row.duplicate_days, f"{row.missing_days:,}", row.popularity_day_over_day_decreases, number(row.popularity_largest_one_step_drop), f"{row.trending_median:g}", f"{row.trending_max:,}"] for row in audit.itertuples()],
        )
        + "\n"
    )
    add(
        f"Duplicate days: {trends['duplicate_days']} in {trends['rows']:,} rows, and {trends['conflicting_duplicate_days']} of them carry different values in the two rows. The copies arrive inside a single page, so they are in AniList's data and not an artefact of paging; the audit drops them before counting anything else. "
        "\"Missing days\" are calendar days with no row, which section 3.4 reads as days without activity. "
        "`popularity` is a running count of users with the title on a list, so a decrease is a list removal or a recount, not an error by itself.\n"
    )

    add("### 3.6 Leakage risks in the trend rows\n")
    spikes = dated.dropna(subset=["premiere_day_over_week_before"])
    seen_twice = trends["newest_row_seen_twice"]
    recheck_minutes = [(pd.Timestamp(item["seen_again_at"]) - pd.Timestamp(item["seen_at"])).total_seconds() / 60 for item in seen_twice]
    add(
        f"- **A UTC date is one day early.** The row for Japan day D is stamped at 15:00 UTC on D-1. On the premiere day itself `trending` was a median {spikes['premiere_day_over_week_before'].median():g} times the daily mean of the week before ({len(spikes)} titles with both). Dating rows in UTC puts that spike inside the pre-release window.\n"
        f"- **What a day's row covers is not documented.** The newest row any title had was dated {trends['fetched_utc']['newest_row_day']} in Japan, the day the collection ran, so a row exists before its day is over. "
        f"Fetched again a median {np.median(recheck_minutes):.0f} minutes later for {len(seen_twice)} titles, `trending`, `popularity` and `inProgress` had changed in {sum(item['changed'] for item in seen_twice)}. "
        "So the row is not a live counter, though a slower refresh during the day is not ruled out. The premiere-day jump above says the row dated D does hold day-D activity. Whichever it is, rows dated D-1 or earlier lie before the premiere, and that is the rule used here.\n"
        f"- **Markers inside the pre-release rows.** Rows before the premiere flagged `releasing`: {int(dated['pre_rows_flagged_releasing'].sum())}. With a positive `averageScore`: {int(dated['pre_rows_with_average_score'].sum())}. With an `episode` number: {int(dated['pre_rows_with_episode_number'].sum())}, in {trends['titles_with_episode_marker_before_premiere']} titles. "
        "An episode number on D-1 means the first episode was on the schedule for the day before the listed premiere, which happens with late-night slots written as 24:00 or later. Where it occurs, D-1 is not safely pre-release, and a one-day embargo (cutoff at D-2) is the conservative choice.\n"
        "- **Restatement cannot be ruled out from one snapshot.** Whether AniList ever rewrites old rows needs a second fetch on a later day and a row-by-row comparison.\n"
        "- **Coverage depends on when the entry was created.** A title added to AniList after it aired has no pre-release rows at all, and obscure titles are the ones added late.\n"
    )

    add("### 3.7 Pre-release features\n")
    add("Built from rows dated D-1 or earlier only (`prerelease_features.csv`). Calendar means count a day without a row as zero.\n")
    add(
        table(
            ["AniList id", "Cutoff", "Days recorded 90 / 30 / 7", "Mean trending 30 d", "Growth (last 7 vs prior 23)", "Volatility 30 d", "Popularity at cutoff", "Popularity growth 30 d", "inProgress at cutoff"],
            [
                [
                    row.anilist_id,
                    row.cutoff_day,
                    f"{int(row.trending_days_recorded_90d)} / {int(row.trending_days_recorded_30d)} / {int(row.trending_days_recorded_7d)}",
                    f"{row.trending_mean_calendar_30d:g}",
                    f"{row.trending_growth_last7_vs_prior23:+.2f}",
                    f"{row.trending_volatility_30d:g}",
                    number(row.popularity_at_cutoff),
                    "" if pd.isna(row.popularity_growth_30d) else f"{row.popularity_growth_30d:+.2f}",
                    number(row.in_progress_at_cutoff),
                ]
                for row in features.itertuples()
            ],
        )
        + "\n"
    )

    add("## 4. API log\n")
    add(
        f"{log['http_requests']} HTTP requests, {log['succeeded']} succeeded ({log['success_rate']:.1%}), {log['http_429']} were rate-limited, {log['failed_or_retried'] - log['http_429']} failed otherwise. "
        f"Pace: {log['observed_requests_per_minute']} a minute over {log['active_minutes']} active minutes against a configured ceiling of {log['configured_requests_per_minute']}. "
        f"The server reported a limit of {', '.join(log['rate_limit_header_values'])} a minute; the lowest `X-RateLimit-Remaining` seen was {log['lowest_remaining_header']}. "
        f"{log['cache_hits']} further lookups were served from the cache. Full detail is in `collection_log.json` and `cache/request_log.jsonl`.\n"
    )

    add("## 5. Cost of scaling\n")
    window_pages = window.groupby("anilist_id")["page"].max().reindex(dated["anilist_id"]).fillna(1)
    per_title_window = float(window_pages.mean())
    per_title_full = float(audit["pages"].mean())
    rows = []
    for size in SCALES:
        metadata = math.ceil(size / METADATA_BATCH)
        pre, full = metadata + size * per_title_window, metadata + size * per_title_full
        rows.append([f"{size:,}", metadata, f"{pre:,.0f}", f"{pre / REQUESTS_PER_MINUTE / 60:.1f} h", f"{full:,.0f}", f"{full / REQUESTS_PER_MINUTE / 60:.1f} h"])
    add(table(["Titles", "Metadata requests", "Requests, pre-release window only", "Runtime", "Requests, full history", "Runtime"], rows) + "\n")
    add(
        f"At {REQUESTS_PER_MINUTE} requests a minute. Trend cost is extrapolated from these {len(audit)} titles: {per_title_window:.2f} requests a title for the 90-day pre-release window, {per_title_full:.1f} for the full history. "
        f"The sample is all TV and reaches back to 2016, so full histories here are long; a different mix will move the full-history column a lot and the window column very little, since the window is capped at two pages. "
        f"Metadata is costed at {METADATA_BATCH} titles a request, which has not been run yet: the sample query fetched 10 in one request, and whether it stays under AniList's complexity limit at 50 is the first thing the next stage tests.\n"
    )

    add("## 6. Can the ratings support a simulated item cold-start? (Task 5)\n")
    add(
        f"`rating_complete.csv` has {ratings['rows_scanned']:,} ratings from {ratings['users']:,} users and no timestamps. Inside the Anime Compass catalog of {ratings['catalog_items']:,} items that leaves {ratings['rows_in_catalog']:,} ratings and {ratings['positives_in_catalog']:,} positives (rating >= {ratings['positive_threshold']}); "
        f"{ratings['users_with_min_positives']:,} users have at least five. These reproduce the counts in Anime Compass's own split metadata. "
        f"{ratings['catalog_items_with_positives']:,} catalog items have a positive at all and {ratings['catalog_items_without_any_rating']:,} have no rating of any kind, which are already cold in production and cannot be evaluated.\n"
    )
    by_count = ratings["items_by_positive_count"]
    add(table(["Positives per item", *by_count.keys()], [["Catalog items", *[f"{value:,}" for value in by_count.values()]]]) + "\n")
    add("Hiding every rating of a set of items and asking how much is left:\n")
    add(
        table(
            ["Cold set", "Cold items", "With >= 50 positives", "Warm items left (>= 1 positive)", "Positives hidden", "Users with warm history and a cold positive", "Evaluable positives", "Per cold item (median)"],
            [
                [row.scenario, f"{row.cold_items:,}", f"{row.cold_items_with_50_positives:,}", f"{row.warm_items_with_1_positive:,}", f"{row.share_of_positives_hidden:.1%}", f"{row.users_with_warm_history_and_1_cold_positive:,}", f"{row.evaluable_user_item_positives:,}", f"{row.evaluable_positives_per_cold_item_median:,.0f}"]
                for row in scenarios.itertuples()
            ],
        )
        + "\n"
    )
    reference = ""
    if args.split_metadata and args.split_metadata.exists():
        split = json.loads(args.split_metadata.read_text(encoding="utf-8"))
        reference = f" For scale, Anime Compass's evaluation ALS is trained on {split['train_positive_interactions']:,} of {split['positive_ratings']:,} positives, {1 - split['train_positive_interactions'] / split['positive_ratings']:.1%} fewer than production."
    add(
        "- **How many warm items are needed.** This cannot be answered without training, and nothing was trained. What the counts show is that none of these cold sets comes close to starving the warm side: the largest hides "
        f"{scenarios['share_of_positives_hidden'].max():.1%} of positives and leaves {scenarios['warm_items_with_1_positive'].min():,} warm items.{reference} Whether ALS quality on the warm side holds after the cut is an empirical check for the modelling stage.\n"
        "- **Which items can be cold test items.** Only items with ratings, and in practice only those with enough positives to give a stable per-item metric. A second filter comes later: the item must also have pre-release rows on AniList.\n"
        "- **It is a simulation.** With no timestamps, a user's warm history contains ratings given after the cold title aired. This measures whether features can place an unseen item among users whose tastes are otherwise fully known. It is not a pre-release backtest and should not be called one.\n"
    )
    add("Keeping cold-item ratings out of everything:\n")
    add(
        "1. Fix the cold item ids first, by premiere date or by seeded draw, and write them to a file with a checksum before touching ratings.\n"
        "2. Split items three ways: warm (in ALS, and the only items a feature-to-factor mapping is fitted on), cold validation (model selection) and cold test (opened once). ALS never sees a validation or test item.\n"
        "3. Build the ALS matrix from ratings filtered through that file, and assert in code that no cold id is in the matrix columns.\n"
        "4. Compute every rating-derived feature (studio prior, franchise prior, popularity counts, co-occurrence) from the warm matrix only. MAL catalog columns such as `Members`, `Score`, `Favorites` and the `Score-N` counts are outcomes and stay out.\n"
        "5. Use AniList fields graded `high` in section 2.5 for nothing, and `trends` only through the cutoff.\n"
        "6. Train a fresh evaluation artifact under this directory. The production ALS and the existing evaluation ALS have both seen these items' ratings and are invalid here.\n"
        "7. Draw evaluation users separately from the users Anime Compass has already scored, or state that they overlap.\n"
    )

    add("## 7. Selection bias and limits of this stage\n")
    add(
        f"- **{sample['rows']} titles.** Every share above is a count out of {sample['rows']}; none supports a rate for the catalog.\n"
        "- **The sample is conditioned on survival.** Titles were drawn from MAL's catalog as of early 2021 and looked up on AniList as it is today. A title's presence in both, and the day its AniList entry was created, are related to how much attention it got.\n"
        "- **AniList's audience is its own.** English-language, tracker-using viewers. The outcome in the ratings is MAL users. Activity on one predicting ratings on the other is plausible but is cross-platform, not general demand.\n"
        "- **Pre-2018 cohorts have `trending` only.** `popularity`, the closest thing to a plan-to-watch count, does not exist in the early rows.\n"
        "- **Terms of use.** AniList prohibits hoarding and mass collection and says it is lenient for purely educational projects. A course pilot fits that; shipping collected AniList data inside the public Anime Compass app is a different use and would need their answer first.\n"
    )
    (stage / "data_quality_report.md").write_text("\n".join(out), encoding="utf-8")
    print(f"{verdict}\n" + "\n".join(reasons))
    print(f"wrote {stage / 'data_quality_report.md'}")


if __name__ == "__main__":
    main()
