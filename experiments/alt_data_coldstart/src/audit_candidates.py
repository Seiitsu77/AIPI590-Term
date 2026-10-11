"""Audit the alternative-data and content coverage of the 2019-2020 cold-item candidates.

Three questions, all answered from data on disk (collect_candidate_windows.py fetched the
AniList part):

1. AniList MediaTrend: how many candidates have a signal 7, 30 and 90 days before the
   premiere, and how deep does the history go?
2. Time leakage: are trend rows really Japan days, is the date we call day 0 the day
   the first episode aired, and how wrong does a UTC reading get it?
3. What else would the three models stand on: Wikipedia pageviews (from the first
   feasibility round) and content metadata (from the team's AniList catalog).

Day 0 is the MAL premiere date, the date the candidates were selected by. A row's day is
its Japan calendar day. "Pre-air" means day <= -1.

Usage:
    python src/audit_candidates.py --animerec <AnimeRec dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
OUT = ROOT / "outputs" / "personalized_coldstart"
WIKIPEDIA_PANEL = REPO / "anime_coldstart" / "data" / "exploratory_analysis_panel.csv"

JST_OFFSET = 9 * 3600
WINDOWS = (7, 30, 90)
# A feature window ends this many days before the premiere. -1 is the nominal rule; -2
# and -8 leave a margin for a premiere dated a day late or an advance release.
WINDOW_ENDS = (-1, -2, -8)
ROLES = ("cold_train", "cold_val", "cold_test", "cold_censored")


def share(mask: pd.Series) -> float:
    return round(float(mask.mean()), 4) if len(mask) else float("nan")


def coverage_block(titles: pd.DataFrame) -> dict:
    block = {"titles": int(len(titles)), "with_anilist_entry": share(titles["anilist_id"].notna()), "any_pre_air_row": share(titles["pre_rows_90"] > 0)}
    for end in WINDOW_ENDS:
        for window in WINDOWS:
            block[f"signal_in_{window}d_ending_day{end}"] = share(titles[f"rows_{window}_end{end}"] > 0)
    for window in WINDOWS:
        block[f"tracked_since_day-{window}"] = share(titles["first_rel_day"] <= -window)
        block[f"popularity_known_at_day-{window}"] = share(titles[f"pop_at_-{window}"].notna())
    block["median_missing_days_in_90"] = float(titles.loc[titles["pre_rows_90"] > 0, "missing_days_90"].median()) if (titles["pre_rows_90"] > 0).any() else None
    block["median_popularity_day-1"] = float(titles["pop_at_-1"].median()) if titles["pop_at_-1"].notna().any() else None
    return block


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--animerec", type=Path, required=True)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    items = pd.read_csv(OUT / "candidate_items.csv", parse_dates=["premiere", "anilist_start"])
    trends = pd.read_parquet(OUT / "candidate_trends.parquet")
    meta = pd.read_csv(OUT / "candidate_anilist_meta.csv")
    trends = trends.drop_duplicates(["anime_id", "timestamp"])
    trends["day"] = pd.to_datetime(trends["day_jst"])
    trends = trends.merge(items[["anime_id", "premiere"]], on="anime_id")
    trends["rel_day"] = (trends["day"] - trends["premiere"]).dt.days

    report: dict = {"definitions": {"day_0": "MAL premiere date", "row_day": "Japan calendar day of the row", "pre_air": "rel_day <= -1"}}

    # ------------------------------------------------------- clock of the rows
    seconds_into_utc_day = trends["timestamp"] % 86400
    utc_day = pd.to_datetime(trends["timestamp"], unit="s").dt.normalize()
    report["row_clock"] = {
        "rows": int(len(trends)),
        "rows_stamped_15_00_utc": int((seconds_into_utc_day == 15 * 3600).sum()),
        "other_stamps": {str(int(k)): int(v) for k, v in seconds_into_utc_day[seconds_into_utc_day != 15 * 3600].value_counts().head(5).items()},
        "rows_whose_utc_date_is_one_day_before_their_japan_date": int((utc_day < trends["day"]).sum()),
    }

    # --------------------------------------------------------- per-title table
    pre = trends[trends["rel_day"] <= -1]
    rows = []
    for anime_id, group in trends.groupby("anime_id"):
        by_day = group.set_index("rel_day")
        row = {"anime_id": anime_id}
        for end in WINDOW_ENDS:
            for window in WINDOWS:
                row[f"rows_{window}_end{end}"] = int(group["rel_day"].between(end - window + 1, end).sum())
        row["pre_rows_90"] = int(group["rel_day"].between(-90, -1).sum())
        row["missing_days_90"] = 90 - row["pre_rows_90"]
        for day in (-1, -2, -8, *[-w for w in WINDOWS]):
            # The value on that day, or the latest one in the week before it.
            near = group[group["rel_day"].between(day - 6, day) & group["popularity"].notna()]
            row[f"pop_at_{day}"] = float(near.sort_values("rel_day")["popularity"].iloc[-1]) if len(near) else np.nan
        for day in (-1, 0, 1):
            row[f"pop_exact_{day}"] = float(by_day.loc[day, "popularity"]) if day in by_day.index else np.nan
            row[f"trending_exact_{day}"] = float(by_day.loc[day, "trending"]) if day in by_day.index else np.nan
        releasing = group[group["releasing"] == True]  # noqa: E712
        episode = group[group["episode"].notna()]
        row["first_releasing_rel_day"] = int(releasing["rel_day"].min()) if len(releasing) else np.nan
        row["first_episode_rel_day"] = int(episode["rel_day"].min()) if len(episode) else np.nan
        row["episode_on_day0_row"] = bool((episode["rel_day"] == 0).any())
        rows.append(row)
    per_title = items.merge(pd.DataFrame(rows), on="anime_id", how="left").merge(meta.drop(columns=["anilist_id"]), on="anime_id", how="left")
    count_columns = [c for c in per_title.columns if c.startswith("rows_")] + ["pre_rows_90"]
    per_title[count_columns] = per_title[count_columns].fillna(0)
    per_title["first_row_day"] = pd.to_datetime(per_title["first_row_day_jst"])
    per_title["first_rel_day"] = (per_title["first_row_day"] - per_title["premiere"]).dt.days

    airing = pd.to_datetime(per_title["first_scheduled_airing_at"] + JST_OFFSET, unit="s")
    per_title["ep1_jst"] = airing.where(per_title["first_scheduled_episode"] == 1)
    per_title["ep1_rel_day_mal"] = (per_title["ep1_jst"].dt.normalize() - per_title["premiere"]).dt.days
    per_title["ep1_rel_day_anilist"] = (per_title["ep1_jst"].dt.normalize() - per_title["anilist_start"]).dt.days
    per_title["ep1_hour_jst"] = per_title["ep1_jst"].dt.hour
    per_title["popularity_fifth"] = pd.qcut(per_title["n_pos"].rank(method="first"), 5, labels=["Q1_least", "Q2", "Q3", "Q4", "Q5_most"])
    per_title.to_csv(OUT / "candidate_trend_audit.csv", index=False)

    # ---------------------------------------------------------------- coverage
    report["anilist_coverage"] = {"all_candidates": coverage_block(per_title)}
    report["anilist_coverage"]["by_role"] = {role: coverage_block(per_title[per_title["role"] == role]) for role in ROLES}
    report["anilist_coverage"]["by_positives_fifth"] = {str(k): coverage_block(g) for k, g in per_title.groupby("popularity_fifth", observed=True)}
    report["anilist_coverage"]["by_anilist_format"] = {str(k): coverage_block(g) for k, g in per_title.groupby(per_title["anilist_format"].fillna("no AniList entry"))}
    report["anilist_coverage"]["at_least_50_positives"] = coverage_block(per_title[per_title["n_pos"] >= 50])
    report["anilist_coverage"]["window_truncated_titles"] = int(per_title["window_truncated"].fillna(False).sum())

    # --------------------------------------------------------------- leakage
    has_ep1 = per_title["ep1_jst"].notna()
    with_rows = per_title["pre_rows_90"] > 0
    report["time_leakage"] = {
        "titles_with_episode_1_airing_time": int(has_ep1.sum()),
        "episode_1_japan_date_minus_mal_date": {str(int(k)): int(v) for k, v in per_title.loc[has_ep1, "ep1_rel_day_mal"].value_counts().sort_index().items()},
        "episode_1_japan_date_minus_anilist_date": {str(int(k)): int(v) for k, v in per_title.loc[has_ep1, "ep1_rel_day_anilist"].dropna().value_counts().sort_index().items()},
        "episode_1_aired_before_mal_day_0": int((per_title["ep1_rel_day_mal"] < 0).sum()),
        "episode_1_airing_hour_jst": {
            "00-05 (after midnight)": int(per_title.loc[has_ep1, "ep1_hour_jst"].between(0, 5).sum()),
            "06-17": int(per_title.loc[has_ep1, "ep1_hour_jst"].between(6, 17).sum()),
            "18-23": int(per_title.loc[has_ep1, "ep1_hour_jst"].between(18, 23).sum()),
        },
        "releasing_flag_before_day_0": {
            "titles": int((per_title["first_releasing_rel_day"] <= -1).sum()),
            "only_on_day_-1": int((per_title["first_releasing_rel_day"] == -1).sum()),
            "earlier_than_day_-1": int((per_title["first_releasing_rel_day"] <= -2).sum()),
        },
        "episode_field_before_day_0": int((per_title["first_episode_rel_day"] <= -1).sum()),
        # The sharpest test of whether the day -1 row is really pre-air: does it already
        # carry episode 1, and for which premieres?
        "episode_1_on_day_-1_row": {
            "titles": int((per_title["first_episode_rel_day"] == -1).sum()),
            "of_those_with_an_airing_time": int(((per_title["first_episode_rel_day"] == -1) & has_ep1).sum()),
            "of_those_aired_00_to_05_jst_on_day_0": int(((per_title["first_episode_rel_day"] == -1) & per_title["ep1_hour_jst"].between(0, 5) & (per_title["ep1_rel_day_mal"] == 0)).sum()),
            "all_premieres_aired_00_to_05_jst_on_day_0": int((per_title["ep1_hour_jst"].between(0, 5) & (per_title["ep1_rel_day_mal"] == 0)).sum()),
        },
        "episode_1_more_than_a_day_before_day_0": per_title.loc[per_title["first_episode_rel_day"] <= -2, "Name"].tolist(),
        # If the episode is already on the day-0 row for shows that air in the evening,
        # a row is filled in through its day rather than frozen at midnight.
        "evening_premieres": {
            "titles": int(((per_title["ep1_hour_jst"] >= 18) & (per_title["ep1_rel_day_mal"] == 0)).sum()),
            "with_episode_on_day_0_row": int((per_title["episode_on_day0_row"] & (per_title["ep1_hour_jst"] >= 18) & (per_title["ep1_rel_day_mal"] == 0)).sum()),
        },
        "mal_and_anilist_dates_differ": int((per_title["date_gap_days"].abs() >= 1).sum()),
        "anilist_date_earlier_than_mal": int((per_title["date_gap_days"] <= -1).sum()),
    }
    both = per_title[per_title["pop_exact_-1"].notna() & per_title["pop_exact_0"].notna() & (per_title["pop_exact_-1"] > 0)]
    week = per_title[per_title["pop_at_-8"].notna() & per_title["pop_exact_-1"].notna() & (per_title["pop_at_-8"] > 0)]
    trend_both = per_title[per_title["trending_exact_-1"].notna() & per_title["trending_exact_0"].notna() & (per_title["trending_exact_-1"] > 0)]
    report["time_leakage"]["premiere_day_jump"] = {
        "titles_compared": int(len(both)),
        "median_popularity_growth_day_-1_to_0": round(float((both["pop_exact_0"] / both["pop_exact_-1"] - 1).median()), 4),
        "median_popularity_growth_per_day_over_days_-8_to_-1": round(float(((week["pop_exact_-1"] / week["pop_at_-8"]) ** (1 / 7) - 1).median()), 4),
        "median_trending_ratio_day_0_over_day_-1": round(float((trend_both["trending_exact_0"] / trend_both["trending_exact_-1"]).median()), 2),
        "note": "a UTC reading relabels the day-0 row as day -1, so this jump is what it would leak",
    }
    report["time_leakage"]["titles_with_rows"] = int(with_rows.sum())

    # Is the day -1 row contaminated when it carries the episode marker? Compare daily
    # `trending` around the premiere for three kinds of premiere. A contaminated day -1
    # would show the jump that day 0 shows.
    on_day0 = per_title["ep1_rel_day_mal"] == 0
    after_midnight = on_day0 & per_title["ep1_hour_jst"].between(0, 5)
    flagged = after_midnight & (per_title["first_episode_rel_day"] == -1)
    minutes = per_title["ep1_jst"].dt.hour * 60 + per_title["ep1_jst"].dt.minute
    groups = {
        "after_midnight_with_episode_on_day_-1_row": flagged,
        "after_midnight_other": after_midnight & ~flagged,
        "evening_18_to_23": on_day0 & (per_title["ep1_hour_jst"] >= 18),
    }
    near = trends[trends["rel_day"].between(-3, 1)]
    report["time_leakage"]["midnight_boundary"] = {
        "after_midnight_premieres": int(after_midnight.sum()),
        "of_those_with_episode_on_day_-1_row": int(flagged.sum()),
        "of_those_airing_00_00_to_00_29": int((flagged & (minutes < 30)).sum()),
        "of_those_airing_exactly_00_00": int((flagged & (minutes == 0)).sum()),
        "other_after_midnight_premieres_airing_00_00_to_00_29": int((after_midnight & ~flagged & (minutes < 30)).sum()),
        "median_trending_by_day": {
            name: {str(int(day)): float(value) for day, value in near[near["anime_id"].isin(per_title.loc[mask, "anime_id"])].groupby("rel_day")["trending"].median().items()}
            for name, mask in groups.items()
        },
        "titles_per_group": {name: int(mask.sum()) for name, mask in groups.items()},
    }

    # --------------------------------------------------------------- Wikipedia
    wiki = pd.read_csv(WIKIPEDIA_PANEL, usecols=["MAL_ID", "mapped_any", "views_pre30_en", "views_pre30_ja", "views_pre30_zh"]).rename(columns={"MAL_ID": "anime_id"})
    wiki["wiki_pre30"] = wiki[["views_pre30_en", "views_pre30_ja", "views_pre30_zh"]].sum(axis=1)
    joined = per_title.merge(wiki, on="anime_id", how="left")
    joined["mapped_any"] = joined["mapped_any"].fillna(False).astype(bool)
    joined["wiki_signal"] = joined["wiki_pre30"].fillna(0) > 0
    joined["anilist_signal"] = joined["rows_30_end-1"] > 0

    def wiki_block(g: pd.DataFrame) -> dict:
        return {"titles": int(len(g)), "mapped_to_article": share(g["mapped_any"]), "views_in_30d_before_premiere": share(g["wiki_signal"]),
                "anilist_or_wikipedia_signal": share(g["wiki_signal"] | g["anilist_signal"]), "both_signals": share(g["wiki_signal"] & g["anilist_signal"])}  # fmt: skip

    report["wikipedia_coverage"] = {"source": "anime_coldstart corrected mapping, days -30..-1, en+ja+zh", "all_candidates": wiki_block(joined), "by_role": {role: wiki_block(joined[joined["role"] == role]) for role in ROLES}}

    # ----------------------------------------------------------------- content
    catalog = pd.read_csv(args.animerec / "data" / "ani_list" / "catalog.csv", usecols=["mal_id", "format", "source", "genres", "tags", "main_studios", "staff", "start_date"], low_memory=False)
    catalog = catalog.dropna(subset=["mal_id"]).astype({"mal_id": "int64"}).drop_duplicates("mal_id")
    catalog["start"] = pd.to_datetime(catalog["start_date"], errors="coerce")
    fields = ["genres", "tags", "main_studios", "staff", "source"]
    cand = per_title[["anime_id"]].merge(catalog, left_on="anime_id", right_on="mal_id", how="left")
    warm = catalog[catalog["start"] < "2019-01-01"]
    warm_studios = {s.strip() for value in warm["main_studios"].dropna() for s in str(value).split("|") if s.strip()}
    studio_sets = cand["main_studios"].dropna().map(lambda value: {s.strip() for s in str(value).split("|") if s.strip()})
    report["content_coverage"] = {
        "source": "AnimeRec data/ani_list/catalog.csv (AniList metadata for titles that have ALS factors)",
        "candidates_in_catalog": share(cand["mal_id"].notna()),
        "candidate_field_filled": {field: share(cand[field].notna() & (cand[field].astype(str).str.len() > 0)) for field in fields},
        "warm_titles_in_catalog": int(len(warm)),
        "warm_field_filled": {field: share(warm[field].notna() & (warm[field].astype(str).str.len() > 0)) for field in fields},
        "studio_separator_assumed": "|",
        "candidates_with_a_studio_listed": int(len(studio_sets)),
        "of_those_with_a_studio_seen_among_warm_titles": int(studio_sets.map(lambda s: bool(s & warm_studios)).sum()),
        "sequel_share_by_title_heuristic": share(per_title["is_sequel"] == 1),
    }

    # ----------------------------------------------------------------- sequels
    # Where a candidate's earlier same-franchise titles sit decides what a model may know
    # about it: a prequel among the warm titles has ALS factors, a prequel that itself
    # premiered in 2019 or later is removed from training along with the candidate.
    sys.path.insert(0, str(REPO / "anime_coldstart" / "src"))
    from build_cohort import approx_date, franchise_root, root_keys

    mal = pd.read_csv(REPO / "anime_coldstart" / "data" / "raw" / "anime.csv", usecols=["MAL_ID", "Name", "Aired"])
    mal["when"] = mal["Aired"].map(approx_date)
    mal["root"] = mal["Name"].map(franchise_root)
    by_root: dict[str, list[tuple[pd.Timestamp, int]]] = {}
    for mal_id, root, when in zip(mal["MAL_ID"], mal["root"], mal["when"], strict=True):
        if root and not pd.isna(when):
            by_root.setdefault(root, []).append((when, int(mal_id)))
    warm_cut = pd.Timestamp("2019-01-01")
    in_warm, only_later = [], []
    for anime_id, name, when in zip(per_title["anime_id"], per_title["Name"], per_title["premiere"], strict=True):
        earlier = [other_when for key in root_keys(franchise_root(name)) for other_when, other_id in by_root.get(key, ()) if other_when < when and other_id != anime_id]
        in_warm.append(any(w < warm_cut for w in earlier))
        only_later.append(bool(earlier) and not any(w < warm_cut for w in earlier))
    per_title["prequel_in_warm"], per_title["prequel_only_after_cutoff"] = in_warm, only_later
    report["sequels"] = {
        "method": "title-root matching (anime_coldstart/src/build_cohort.py); the catalog has no relations column",
        "by_role": {
            role: {
                "titles": int((per_title["role"] == role).sum()),
                "earlier_title_among_warm": int(per_title.loc[per_title["role"] == role, "prequel_in_warm"].sum()),
                "earlier_titles_only_from_2019_on": int(per_title.loc[per_title["role"] == role, "prequel_only_after_cutoff"].sum()),
            }
            for role in ROLES
        },
    }
    per_title.to_csv(OUT / "candidate_trend_audit.csv", index=False)

    (OUT / "alt_data_coverage.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
