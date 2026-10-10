"""Audit the Wikipedia articles each cold title's pageviews are drawn from.

The article mapping is today's: it comes from Wikidata as it stood in October 2026. A
title's views are summed over its own article, its series article and the article of
its source work. Some of those pages did not exist when the show premiered. A page that
did not exist cannot have been read, so any views recorded under its title before it was
created belong to something else that once sat at that title (an earlier redirect, a
deleted page, a different subject).

This script asks Wikipedia when each mapped page was created (its first revision) and
applies one rule, fixed before any label is used:

    an article created after a title's cutoff day (day 0 minus 2) is excluded from that
    title's pageviews.

What it cannot see. A page's first revision travels with the page when it is moved, so a
page created early under another name passes the check although its present title did
not exist yet; its early views are then missing, which only understates the signal.
And an article that existed may still have been about less than it is today.

Reads no ratings. Responses are cached under artifacts/.

Usage:
    python src/check_wikipedia_articles.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

from common import ARTIFACTS, CUTOFF_DAY, LANGS, MANIFEST, ROLES, WIKI_EXTRA, WIKI_INDEX, WIKI_MAP, WIKI_VIEWS, WIKI_WINDOW

CACHE = ARTIFACTS / "wiki_first_revision"
CONTACT = os.environ.get("WIKIMEDIA_CONTACT", "").strip()
HEADERS = {"User-Agent": f"AnimeColdStartFeasibility/0.2 ({CONTACT or 'Duke AIPI 590 student project'}) python-requests"}
PAUSE_SECONDS = 0.35 if CONTACT else 6.5  # 200 a minute with contact details, 10 without


def first_revision(lang: str, title: str) -> dict:
    path = CACHE / lang / (hashlib.md5(title.encode("utf-8")).hexdigest() + ".json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    params = {"action": "query", "prop": "revisions", "titles": title, "rvlimit": 1, "rvdir": "newer", "rvprop": "timestamp", "format": "json", "formatversion": 2}
    for attempt in range(6):
        response = requests.get(f"https://{lang}.wikipedia.org/w/api.php", params=params, headers=HEADERS, timeout=60)
        if response.status_code == 200:
            page = response.json()["query"]["pages"][0]
            record = {"lang": lang, "title": title, "missing": bool(page.get("missing", False)), "created_utc": (page.get("revisions") or [{}])[0].get("timestamp")}
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
            time.sleep(PAUSE_SECONDS)
            return record
        if response.status_code == 429 or response.status_code >= 500:
            time.sleep(float(response.headers.get("Retry-After", 30)) + 1.0)
            continue
        response.raise_for_status()
    raise RuntimeError(f"first-revision lookup kept failing for {lang}:{title}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    days = pd.read_csv(ARTIFACTS / "day_zero.csv", parse_dates=["day_zero"])
    roles = pd.read_csv(MANIFEST / "item_roles.csv", usecols=["anime_id", "role"])
    cold_ids = set(roles.loc[roles["role"].isin(ROLES), "anime_id"])

    links = []
    mapping = pd.read_csv(WIKI_MAP)
    for row in mapping[mapping["MAL_ID"].isin(cold_ids)].itertuples():
        for lang in LANGS:
            for kind in ("own", "parent"):
                title = getattr(row, f"{kind}_{lang}_title")
                if isinstance(title, str):
                    links.append({"anime_id": row.MAL_ID, "lang": lang, "title": title, "relation": kind if kind == "own" else f"parent_{row.parent_prop}"})
    extra = pd.read_csv(WIKI_EXTRA)
    for row in extra[extra["MAL_ID"].isin(cold_ids)].itertuples():
        links.append({"anime_id": row.MAL_ID, "lang": row.lang, "title": row.title, "relation": row.relation})
    # One article can reach a title by two routes; keep one row and remember both routes.
    frame = pd.DataFrame(links).groupby(["anime_id", "lang", "title"], as_index=False)["relation"].agg(lambda s: "|".join(sorted(set(s))))

    pairs = sorted(set(zip(frame["lang"], frame["title"], strict=True)))
    print(f"{len(frame)} title-article links, {len(pairs)} distinct articles, one request every {PAUSE_SECONDS}s", flush=True)
    created = {}
    for position, (lang, title) in enumerate(pairs, 1):
        created[(lang, title)] = first_revision(lang, title)
        if position % 100 == 0:
            print(f"looked up {position} of {len(pairs)}", flush=True)

    index = pd.read_csv(WIKI_INDEX)
    row_of = {(lang, title): row for row, lang, title in zip(index["row"], index["lang"], index["title"], strict=True)}
    payload = np.load(WIKI_VIEWS)
    views, first_day = payload["views"], pd.Timestamp(str(payload["first_day"]))

    frame = frame.merge(days[["anime_id", "role", "day_zero"]], on="anime_id")
    cutoff = frame["day_zero"] + pd.Timedelta(days=CUTOFF_DAY)
    frame["created_utc"] = [created[(lang, title)]["created_utc"] for lang, title in zip(frame["lang"], frame["title"], strict=True)]
    frame["page_missing_today"] = [created[(lang, title)]["missing"] for lang, title in zip(frame["lang"], frame["title"], strict=True)]
    created_day = pd.to_datetime(frame["created_utc"], utc=True).dt.tz_localize(None).dt.normalize()
    frame["created_after_cutoff"] = created_day > cutoff
    window_views = []
    for lang, title, day_zero in zip(frame["lang"], frame["title"], frame["day_zero"], strict=True):
        offset = (day_zero - first_day).days
        window_views.append(int(views[row_of[(lang, title)], offset + WIKI_WINDOW[0] : offset + WIKI_WINDOW[1] + 1].sum()))
    frame["views_in_window"] = window_views
    # Excluded: created too late, or gone today so its creation cannot be established.
    frame["excluded"] = frame["created_after_cutoff"] | frame["page_missing_today"]
    frame["created_day_utc"] = created_day.dt.strftime("%Y-%m-%d")
    columns = ["anime_id", "role", "lang", "title", "relation", "created_day_utc", "page_missing_today", "created_after_cutoff", "views_in_window", "excluded"]
    frame.sort_values(["anime_id", "lang", "title"])[columns].to_csv(MANIFEST / "wikipedia_article_audit.csv", index=False, lineterminator="\n")

    per_title = frame.groupby("anime_id").agg(links=("title", "size"), kept=("excluded", lambda s: int((~s).sum())), views_all=("views_in_window", "sum"))
    kept_views = frame[~frame["excluded"]].groupby("anime_id")["views_in_window"].sum()
    per_title["views_kept"] = kept_views.reindex(per_title.index).fillna(0).astype(int)
    shared = frame[~frame["excluded"]].groupby(["lang", "title"])["anime_id"].nunique()
    late = frame[frame["created_after_cutoff"]]
    summary = {
        "rule": f"an article whose first revision is later than day {CUTOFF_DAY} of a title is excluded from that title's pageviews; so is a page that no longer exists",
        "cold_titles": int(len(cold_ids)),
        "cold_titles_with_a_mapped_article": int(len(per_title)),
        "title_article_links": int(len(frame)),
        "distinct_articles": int(len(pairs)),
        "links_created_after_cutoff": int(len(late)),
        "of_those_with_views_recorded_before_they_existed": int((late["views_in_window"] > 0).sum()),
        "views_removed_by_the_rule": int(frame.loc[frame["excluded"], "views_in_window"].sum()),
        "share_of_all_window_views_removed": round(float(frame.loc[frame["excluded"], "views_in_window"].sum() / max(frame["views_in_window"].sum(), 1)), 5),
        "links_to_pages_missing_today": int(frame["page_missing_today"].sum()),
        "links_excluded": int(frame["excluded"].sum()),
        "excluded_by_relation": frame.loc[frame["excluded"], "relation"].value_counts().to_dict(),
        "links_by_relation": frame["relation"].value_counts().to_dict(),
        "titles_losing_every_article": int((per_title["kept"] == 0).sum()),
        "titles_whose_signal_becomes_zero": int(((per_title["views_all"] > 0) & (per_title["views_kept"] == 0)).sum()),
        "titles_with_views_before_and_after_the_rule": {"before": int((per_title["views_all"] > 0).sum()), "after": int((per_title["views_kept"] > 0).sum())},
        "articles_shared_by_3_or_more_cold_titles": {f"{lang}:{title}": int(count) for (lang, title), count in shared[shared >= 3].sort_values(ascending=False).items()},
    }
    (MANIFEST / "wikipedia_article_audit.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
