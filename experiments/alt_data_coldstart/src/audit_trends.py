"""Task 2: audit the collected MediaTrend history and build pre-release features.

Days are Japan days. AniList stamps every trend row at 15:00 UTC, which is 00:00 JST, and
the row stamped at the start of JST day D holds the activity of day D: the newest row a
title has is today's, and it is already there while today is still running. Converting
the timestamp to a UTC calendar date therefore lands one day early and files the
premiere-day spike under "the day before the premiere". Everything here adds nine hours
before taking the date.

Prediction cutoff: the end of JST day D-1, where D is the premiere date. Pre-release
windows are [D-90, D-1], [D-30, D-1] and [D-7, D-1]. No row dated D or later reaches a
feature, and the script asserts it.

A day without a row is read as a day without activity, not as a missing measurement:
no row in the data has trending below 1, and sparse titles have rows only now and then.
That reading is an inference from the data, so both "recorded days" and "calendar days"
versions of the means are written out.

MediaTrend is activity on AniList by AniList users. It is not Wikipedia pageviews and not
search volume.

Outputs (in --out): historical_signal_audit.csv, prerelease_features.csv, trends_audit.json.

Usage:
    python src/audit_trends.py --metadata <csv or parquet with id, mal_id, start_date> --out outputs/<stage>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from anilist_client import read_cached

DAY = 86400
JST_OFFSET = 9 * 3600
GRID_SECONDS = 15 * 3600  # 15:00 UTC = 00:00 JST
WINDOWS = (90, 30, 7)
SIGNALS = ("trending", "popularity", "inProgress")
VALUE_COLUMNS = ["trending", "popularity", "inProgress", "averageScore", "releasing", "episode"]


def jst_day(timestamp: pd.Series) -> pd.Series:
    """Days since 1970-01-01 in Japan time."""
    return (timestamp + JST_OFFSET) // DAY


def iso(day) -> str | None:
    return None if pd.isna(day) else (pd.Timestamp("1970-01-01") + pd.Timedelta(days=int(day))).strftime("%Y-%m-%d")


def last_known(frame: pd.DataFrame, column: str, through_day: int):
    """Latest non-null value on or before a day."""
    values = frame.loc[(frame["day"] <= through_day) & frame[column].notna(), column]
    return float(values.iloc[-1]) if len(values) else np.nan


def audit_title(meta, full: pd.DataFrame, window: pd.DataFrame, latest: pd.DataFrame) -> tuple[dict, dict | None]:
    fetched = full.sort_values("timestamp").reset_index(drop=True)
    row = {"anilist_id": int(meta.anilist_id), "mal_id": int(meta.mal_id), "title": meta.title_romaji, "premiere": meta.start_date}
    # A day can come back twice. Exact copies are dropped; a day whose two rows disagree
    # is counted separately, because then there is no telling which row is right.
    distinct = fetched.drop_duplicates(["timestamp", *VALUE_COLUMNS])
    full = distinct.drop_duplicates("timestamp").reset_index(drop=True)
    days = full["day"]
    gaps = days.diff().dropna()
    row.update(
        rows=len(fetched),
        pages=int(fetched["page"].max()),
        first_day=iso(days.min()),
        last_day=iso(days.max()),
        span_days=int(days.max() - days.min() + 1),
        days_recorded=int(days.nunique()),
        duplicate_days=int(len(fetched) - len(full)),
        conflicting_duplicate_days=int(len(distinct) - len(full)),
        timestamps_off_the_jst_midnight_grid=int((full["timestamp"] % DAY != GRID_SECONDS).sum()),
        longest_gap_days=int(gaps.max() - 1) if len(gaps) else 0,
        min_trending=int(full["trending"].min()),
        rows_with_zero_trending=int((full["trending"] == 0).sum()),
    )
    row["missing_days"] = row["span_days"] - row["days_recorded"]
    row["share_of_days_recorded"] = round(row["days_recorded"] / row["span_days"], 3)

    # Paging: does walking the history upwards end on the row the API calls newest?
    newest = int(latest["timestamp"].max())
    row["paging_reaches_newest_row"] = bool(full["timestamp"].max() == newest)
    row["reported_total_page1"] = int(full.loc[full["page"] == 1, "reported_total"].iloc[0])

    for column in SIGNALS + ("episode",):
        row[f"share_rows_with_{column}"] = round(float(full[column].notna().mean()), 3)
    for column in ("popularity", "inProgress"):
        row[f"first_day_with_{column}"] = iso(full.loc[full[column].notna(), "day"].min())
    # averageScore is 0, not null, while a title has no score yet.
    scored = full["averageScore"].fillna(0) > 0
    row["share_rows_with_averageScore"] = round(float(scored.mean()), 3)
    row["first_day_with_averageScore"] = iso(full.loc[scored, "day"].min())

    popularity = full.loc[full["popularity"].notna(), ["day", "popularity"]]
    change = popularity["popularity"].diff().dropna()
    row["popularity_day_over_day_decreases"] = int((change < 0).sum())
    row["popularity_largest_one_step_drop"] = int(-change.min()) if len(change) and change.min() < 0 else 0
    row["trending_max"] = int(full["trending"].max())
    row["trending_median"] = float(full["trending"].median())

    if pd.isna(meta.start_date) or len(str(meta.start_date)) != 10:
        row["premiere_day_known"] = False
        return row, None
    premiere = int(pd.Timestamp(meta.start_date).timestamp() // DAY)
    cutoff = premiere - 1
    before = full[full["day"] <= cutoff]
    row["premiere_day_known"] = True
    row["days_from_first_row_to_premiere"] = int(premiere - days.min())
    row["rows_before_premiere"] = len(before)
    for width in WINDOWS:
        inside = before[before["day"] >= premiere - width]
        row[f"pre{width}_rows"] = len(inside)
        for column in SIGNALS:
            row[f"pre{width}_rows_with_{column}"] = int(inside[column].notna().sum())

    # The date filter: the rows it returned must be the same rows the full history has in that range.
    if len(window):
        low, high = int(window["after"].iloc[0]), int(window["before"].iloc[0])
        expected = full[(full["timestamp"] > low) & (full["timestamp"] < high)]
        columns = ["timestamp", *VALUE_COLUMNS]
        window = window.drop_duplicates(columns).sort_values("timestamp")
        row["date_filter_rows"] = len(window)
        row["date_filter_matches_full_history"] = bool(
            len(window) == len(expected) and window[columns].reset_index(drop=True).equals(expected[columns].reset_index(drop=True))
        )
    else:
        row["date_filter_rows"] = 0
        row["date_filter_matches_full_history"] = bool(len(before[before["day"] >= premiere - 90]) == 0)

    # Leakage markers inside the pre-release rows.
    row["pre_rows_flagged_releasing"] = int(before["releasing"].fillna(False).sum())
    row["pre_rows_with_episode_number"] = int(before["episode"].notna().sum())
    row["pre_rows_with_average_score"] = int((before["averageScore"].fillna(0) > 0).sum())
    row["first_scored_day_minus_premiere"] = int(full.loc[scored, "day"].min() - premiere) if scored.any() else None
    marked = before.loc[before["episode"].notna(), "day"]
    row["earliest_episode_marker_minus_premiere"] = int(marked.min() - premiere) if len(marked) else None
    releasing = full.loc[full["releasing"].fillna(False), "day"]
    row["first_releasing_day_minus_premiere"] = int(releasing.min() - premiere) if len(releasing) else None
    on_premiere = full.loc[full["day"] == premiere, "trending"]
    week_before = before.loc[before["day"] >= premiere - 7, "trending"].sum() / 7
    row["premiere_day_trending"] = int(on_premiere.iloc[0]) if len(on_premiere) else None
    row["mean_daily_trending_week_before"] = round(float(week_before), 2)
    row["premiere_day_over_week_before"] = round(float(on_premiere.iloc[0] / week_before), 1) if len(on_premiere) and week_before > 0 else None

    # Features. Only `before` is used from here on.
    assert before["day"].max() < premiere if len(before) else True
    features = {"anilist_id": row["anilist_id"], "mal_id": row["mal_id"], "premiere": meta.start_date, "cutoff_day": iso(cutoff), "pre_release_rows": len(before)}
    series = {}
    for width in WINDOWS:
        calendar = pd.Series(0.0, index=range(premiere - width, premiere))
        inside = before[before["day"] >= premiere - width].drop_duplicates("day")
        calendar.loc[inside["day"].to_numpy()] = inside["trending"].to_numpy(dtype=float)
        series[width] = calendar
        features[f"trending_days_recorded_{width}d"] = len(inside)
        features[f"trending_sum_{width}d"] = float(calendar.sum())
        features[f"trending_mean_calendar_{width}d"] = round(float(calendar.mean()), 3)
        features[f"trending_mean_recorded_{width}d"] = round(float(inside["trending"].mean()), 3) if len(inside) else np.nan
    last_week, earlier = series[30].iloc[-7:].mean(), series[30].iloc[:-7].mean()
    features["trending_growth_last7_vs_prior23"] = round(float(np.log1p(last_week) - np.log1p(earlier)), 3)
    features["trending_volatility_30d"] = round(float(series[30].std()), 3)
    features["trending_cv_30d"] = round(float(series[30].std() / series[30].mean()), 3) if series[30].mean() > 0 else np.nan
    features["popularity_at_cutoff"] = last_known(before, "popularity", cutoff)
    features["popularity_30d_before_cutoff"] = last_known(before, "popularity", cutoff - 30)
    features["popularity_growth_30d"] = round(float(np.log1p(features["popularity_at_cutoff"]) - np.log1p(features["popularity_30d_before_cutoff"])), 3)
    features["in_progress_at_cutoff"] = last_known(before, "inProgress", cutoff)
    return row, features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 30)

    meta = pd.read_parquet(args.metadata) if args.metadata.suffix == ".parquet" else pd.read_csv(args.metadata)
    meta = meta.rename(columns={"id": "anilist_id"})
    frames = {name: pd.read_parquet(args.out / f"trends_{name}.parquet") for name in ("full", "window", "latest")}
    for frame in frames.values():
        if len(frame):
            frame["day"] = jst_day(frame["timestamp"])

    audits, features = [], []
    for title in meta.itertuples():
        selected = {name: frame[frame["anilist_id"] == title.anilist_id] if len(frame) else frame for name, frame in frames.items()}
        audit, feature = audit_title(title, selected["full"], selected["window"], selected["latest"])
        audits.append(audit)
        if feature is not None:
            features.append(feature)
    audit = pd.DataFrame(audits)
    audit.to_csv(args.out / "historical_signal_audit.csv", index=False)
    pd.DataFrame(features).to_csv(args.out / "prerelease_features.csv", index=False)

    # Is the newest row still moving? The same row was fetched at different times: by the
    # "latest" request, on the last "full" page, and by --recheck-latest if that was run.
    full = frames["full"]
    moved = []
    for title in meta.itertuples():
        media_id = int(title.anilist_id)
        first_look = read_cached(f"trends/{media_id}/latest_p0001")
        newest = first_look["data"]["Page"]["mediaTrends"][0]
        last_page = int(full.loc[full["anilist_id"] == media_id, "page"].max())
        looks = []
        for name in (f"trends/{media_id}/latest_p0001", f"trends/{media_id}/full_p{last_page:04d}", f"trends/{media_id}/recheck_p0001"):
            stored = read_cached(name)
            if stored is None:
                continue
            same_row = [row for row in stored["data"]["Page"]["mediaTrends"] if row["date"] == newest["date"]]
            if same_row:
                looks.append((stored["fetched_utc"], {column: same_row[0][column] for column in ("trending", "popularity", "inProgress")}))
        looks.sort(key=lambda look: look[0])
        if len(looks) > 1:
            moved.append({"anilist_id": media_id, "row_day": iso((newest["date"] + JST_OFFSET) // DAY), "seen_at": looks[0][0], "seen_again_at": looks[-1][0], "then": looks[0][1], "later": looks[-1][1], "changed": looks[0][1] != looks[-1][1]})

    known = audit[audit["premiere_day_known"]]
    summary = {
        "titles": len(audit),
        "rows": int(audit["rows"].sum()),
        "pages": int(audit["pages"].sum()),
        "all_timestamps_on_jst_midnight": bool((audit["timestamps_off_the_jst_midnight_grid"] == 0).all()),
        "duplicate_days": int(audit["duplicate_days"].sum()),
        "conflicting_duplicate_days": int(audit["conflicting_duplicate_days"].sum()),
        "titles_with_episode_marker_before_premiere": int((known["pre_rows_with_episode_number"] > 0).sum()),
        "fetched_utc": {"first": min(read_cached(f"trends/{int(i)}/latest_p0001")["fetched_utc"] for i in meta["anilist_id"]), "newest_row_day": iso(frames["latest"]["day"].max())},
        "min_trending_in_any_row": int(audit["min_trending"].min()),
        "rows_with_zero_trending": int(audit["rows_with_zero_trending"].sum()),
        "paging_reaches_newest_row": int(audit["paging_reaches_newest_row"].sum()),
        "date_filter_matches_full_history": int(known["date_filter_matches_full_history"].sum()),
        "titles_with_premiere_day": len(known),
        "titles_with_any_pre_release_row": int((known["rows_before_premiere"] > 0).sum()),
        "titles_with_rows_in_window": {f"{width}d": int((known[f"pre{width}_rows"] > 0).sum()) for width in WINDOWS},
        "titles_with_popularity_in_window": {f"{width}d": int((known[f"pre{width}_rows_with_popularity"] > 0).sum()) for width in WINDOWS},
        "earliest_day_with_popularity": audit["first_day_with_popularity"].dropna().min(),
        "earliest_day_with_inProgress": audit["first_day_with_inProgress"].dropna().min(),
        "newest_row_seen_twice": moved,
    }
    (args.out / "trends_audit.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(audit.set_index("anilist_id").T.to_string())
    print()
    print(pd.DataFrame(features).set_index("anilist_id").T.to_string())
    print()
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
