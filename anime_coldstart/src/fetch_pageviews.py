"""Fetch daily Wikipedia pageviews for every mapped article.

Source: the Wikimedia Analytics pageviews API (per-article, daily, all-access, agent
"user", which excludes self-identified bots and spiders). One request per article
covers the whole study period, so pre-air windows are cut locally and the request count
does not depend on how many cohort titles share an article.

The series is point-in-time by construction: each day's count was logged on that day and
is not restated. It starts on 2015-07-01, which is why the cohort starts in 2016.

Rate limits (mediawiki.org/wiki/Wikimedia_APIs/Rate_limits, read 2026-10-04): 10
requests a minute for an unidentified client, 200 a minute for a client whose
User-Agent carries contact information. The contact is read from the WIKIMEDIA_CONTACT
environment variable so that it is never written into the repository. Without it the
script paces itself to the unidentified limit and simply takes longer.

A 404 from this endpoint means "no views recorded for this title in the range" and is
stored as an all-zero series with status "no_data". Any other failure is retried and
then raised. Nothing is ever filled in with placeholder values: an article that has not
been fetched yet is recorded as "not_fetched" and the analysis refuses to use it.

Responses are cached per article under data/raw/pageviews/, so an interrupted run
resumes where it stopped. TV titles are fetched first because they are the primary sample.

Usage:
    python src/fetch_pageviews.py [--minutes N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CACHE = RAW / "pageviews"
OUT = ROOT / "data"

API = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{lang}.wikipedia/all-access/user/{title}/daily/{start}/{end}"
FIRST_DAY = pd.Timestamp("2015-07-01")
LAST_DAY = pd.Timestamp("2021-03-31")
DAYS = pd.date_range(FIRST_DAY, LAST_DAY, freq="D")
LANGS = ("en", "ja", "zh")

CONTACT = os.environ.get("WIKIMEDIA_CONTACT", "").strip()
if CONTACT:
    HEADERS = {"User-Agent": f"AnimeColdStartFeasibility/0.1 ({CONTACT}) python-requests"}
    PAUSE_SECONDS = 0.4  # 150 a minute, under the 200 allowed
else:
    HEADERS = {"User-Agent": "AnimeColdStartFeasibility/0.1 (Duke AIPI 590 student project) python-requests"}
    PAUSE_SECONDS = 6.5  # just under 10 a minute


def cache_path(lang: str, title: str) -> Path:
    return CACHE / lang / (hashlib.md5(title.encode("utf-8")).hexdigest() + ".json")


def fetch(lang: str, title: str) -> str:
    url = API.format(lang=lang, title=quote(title, safe=""), start=FIRST_DAY.strftime("%Y%m%d"), end=LAST_DAY.strftime("%Y%m%d"))
    for _ in range(8):
        response = requests.get(url, headers=HEADERS, timeout=60)
        if response.status_code == 200:
            views = np.zeros(len(DAYS), dtype=np.int64)
            for item in response.json()["items"]:
                views[(pd.Timestamp(item["timestamp"][:8]) - FIRST_DAY).days] = int(item["views"])
            record = {"lang": lang, "title": title, "status": "ok", "views": views.tolist()}
            break
        if response.status_code == 404:
            record = {"lang": lang, "title": title, "status": "no_data", "views": []}
            break
        if response.status_code == 429 or response.status_code >= 500:
            time.sleep(float(response.headers.get("Retry-After", 30)) + 1.0)
            continue
        response.raise_for_status()
    else:
        raise RuntimeError(f"pageviews request kept failing for {lang}:{title} (last status {response.status_code})")
    path = cache_path(lang, title)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    return record["status"]


def ordered_pairs() -> list[tuple[str, str]]:
    """Every (language, article) to fetch, TV titles first, then en, ja, zh."""
    mapping = pd.read_csv(OUT / "mal_wikipedia_map.csv").merge(pd.read_csv(OUT / "cohort.csv")[["MAL_ID", "Type"]], on="MAL_ID")
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for rows in (mapping[mapping["Type"] == "TV"], mapping[mapping["Type"] != "TV"]):
        for lang in LANGS:
            titles = sorted({title for kind in ("own", "parent") for title in rows[f"{kind}_{lang}_title"].dropna()})
            for title in titles:
                if (lang, title) not in seen:
                    seen.add((lang, title))
                    pairs.append((lang, title))
    # Source-work articles added post hoc by map_wikipedia_exploratory.py. They go last
    # so the rows fetched for the pre-specified analysis keep their order.
    extra_path = OUT / "mal_wikipedia_extra_articles.csv"
    if extra_path.exists():
        extra = pd.read_csv(extra_path)
        for lang, title in sorted(set(zip(extra["lang"], extra["title"], strict=True))):
            if (lang, title) not in seen:
                seen.add((lang, title))
                pairs.append((lang, title))
    return pairs


def assemble(pairs: list[tuple[str, str]]) -> pd.DataFrame:
    matrix = np.zeros((len(pairs), len(DAYS)), dtype=np.int32)
    rows = []
    for row, (lang, title) in enumerate(pairs):
        path = cache_path(lang, title)
        status = "not_fetched"
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            status = record["status"]
            if record["views"]:
                matrix[row] = np.asarray(record["views"], dtype=np.int32)
        rows.append({"row": row, "lang": lang, "title": title, "status": status, "total_views": int(matrix[row].sum())})
    index = pd.DataFrame(rows)
    index.to_csv(OUT / "pageviews_index.csv", index=False)
    np.savez_compressed(OUT / "pageviews_daily.npz", views=matrix, first_day=str(FIRST_DAY.date()), last_day=str(LAST_DAY.date()))
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=None, help="stop fetching after this long and assemble what is cached")
    args = parser.parse_args()

    pairs = ordered_pairs()
    todo = [pair for pair in pairs if not cache_path(*pair).exists()]
    started = datetime.now(timezone.utc)
    deadline = time.monotonic() + args.minutes * 60 if args.minutes else None
    print(f"{len(pairs)} articles, {len(todo)} to fetch, one request every {PAUSE_SECONDS}s", flush=True)
    done = 0
    for lang, title in todo:
        if deadline is not None and time.monotonic() > deadline:
            break
        fetch(lang, title)
        done += 1
        if done % 50 == 0:
            print(f"fetched {done} of {len(todo)}", flush=True)
        time.sleep(PAUSE_SECONDS)

    index = assemble(pairs)
    manifest = {
        "endpoint": API,
        "access": "all-access",
        "agent": "user",
        "first_day": str(FIRST_DAY.date()),
        "last_day": str(LAST_DAY.date()),
        "articles": len(pairs),
        "status": index["status"].value_counts().to_dict(),
        "status_by_language": {lang: index[index["lang"] == lang]["status"].value_counts().to_dict() for lang in LANGS},
        "identified_client": bool(CONTACT),
        "last_run_started_utc": started.isoformat(timespec="seconds"),
        "last_run_finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (RAW / "pageviews_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=1), flush=True)


if __name__ == "__main__":
    main()
