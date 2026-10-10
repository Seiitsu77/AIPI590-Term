"""Download English, Japanese and Chinese Wikipedia pageviews for the AniList cohorts.

Cohort: every title with trends (TV, ONA, MOVIE, TV_SHORT) in data/ani_list/anime.csv (2021-2025) and
data/ani_list_live/anime.csv (2026-fall, 2027-winter), keyed by MAL ID.

Steps, each cached under data/wikipedia/raw/ so a rerun resumes:

    1. sparql     MAL ID (Wikidata P4086) -> the anime's Wikidata item, its series
                  (P179), what it is based on (P144), its media franchise (P8345),
                  and the series' own P144 / P8345.
    2. entities   en/ja/zh Wikipedia sitelinks of all those items (wbgetentities).
    3. redirects  every redirect to each article (MediaWiki API). A renamed article
                  keeps its old title as a redirect, and views are logged under the
                  title the reader hit, so pre-rename views live on the redirect.
    4. pageviews  daily user pageviews for each article and redirect (Wikimedia REST
                  API), over the union of [premiere - 365, premiere + 120] of the
                  titles that use it.

Why the series and source articles: an anime's own article (e.g. "Frieren (TV
series)") is often split off its manga article after the show airs, so before the
premiere it has no views at all; the attention sits on the source article
("Frieren"). The first feasibility round mapped only the anime's own article and
recorded zero pre-air views for 36 of its most popular titles.

The Wikimedia REST API answers 429 (with Retry-After) when a client goes too fast;
8 parallel workers tripped it within minutes. Requests are therefore paced by one
shared limiter (--rate, default 2 per second) and a 429 waits as told and retries
without giving up. Article series are fetched before redirect series, so
--articles-only gives usable data in a fraction of the time.

Usage:
    python scripts/wikipedia/download.py --articles-only   # ~2,500 series, ~20 min
    python scripts/wikipedia/download.py                   # adds ~7,000 redirect series
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COHORTS = {"history": PROJECT_ROOT / "data" / "ani_list" / "anime.csv",
           "live": PROJECT_ROOT / "data" / "ani_list_live" / "anime.csv"}  # fmt: skip
DEFAULT_OUT = PROJECT_ROOT / "data" / "wikipedia"
USER_AGENT = "AnimeRec/0.1 (https://github.com/Seiitsu77/AIPI590-Term; Duke AIPI 590 course project)"
WIKIS = ("en", "ja", "zh")
PRE_DAYS, POST_DAYS = 365, 120
PAGEVIEWS_START = date(2015, 7, 1)  # the REST API has no earlier data
SPARQL_BATCH, ENTITY_BATCH, TITLE_BATCH = 200, 50, 50
PAGEVIEW_WORKERS = 4
MAX_ATTEMPTS = 5  # for network errors and 5xx; 429 is retried until it succeeds
MAX_RATE_LIMIT_WAITS = 50

SPARQL = """SELECT ?mal ?item ?series ?basedOn ?franchise ?seriesBasedOn ?seriesFranchise WHERE {
  VALUES ?mal { %s }
  ?item wdt:P4086 ?mal .
  OPTIONAL { ?item wdt:P179 ?series .
             OPTIONAL { ?series wdt:P144 ?seriesBasedOn }
             OPTIONAL { ?series wdt:P8345 ?seriesFranchise } }
  OPTIONAL { ?item wdt:P144 ?basedOn }
  OPTIONAL { ?item wdt:P8345 ?franchise }
}"""
# Wikidata variable -> role; roles are deduplicated per title in this priority order.
ROLES = {"item": "self", "series": "series", "basedOn": "based_on", "seriesBasedOn": "based_on",
         "franchise": "franchise", "seriesFranchise": "franchise"}  # fmt: skip


class RateLimiter:
    """At most `rate` request starts per second, shared by all threads."""

    def __init__(self, rate: float) -> None:
        self.interval = 1.0 / rate
        self.next_start = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next_start)
            self.next_start = start + self.interval
        time.sleep(max(0.0, start - now))

    def back_off(self, seconds: float) -> None:
        """Push every thread's next start out after a 429."""
        with self.lock:
            self.next_start = max(self.next_start, time.monotonic() + seconds)


LIMITER = RateLimiter(2.0)


def get_json(url: str) -> tuple[int, Any]:
    """Paced GET; waits out 429s, retries 5xx/network errors; returns (status, body), 404 included."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    attempt = rate_limited = 0
    while True:
        LIMITER.wait()
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return 404, None
            if exc.code == 429 and rate_limited < MAX_RATE_LIMIT_WAITS:
                rate_limited += 1
                LIMITER.back_off(float(exc.headers.get("Retry-After") or 10))
                continue
            attempt += 1
            if exc.code not in (500, 502, 503, 504) or attempt >= MAX_ATTEMPTS:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            attempt += 1
            if attempt >= MAX_ATTEMPTS:
                raise
        time.sleep(2**attempt)


def save(path: Path, record: dict[str, Any]) -> None:
    record["fetched_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def title_key(wiki: str, title: str) -> str:
    return hashlib.sha1(f"{wiki}:{title}".encode("utf-8")).hexdigest()[:16]


def load_cohort() -> list[dict[str, Any]]:
    titles = []
    for cohort, path in COHORTS.items():
        with path.open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                if row["trends_status"] != "ok" or not row["mal_id"]:
                    continue
                parts = [int(part) for part in row["start_date"].split("-")]
                premiere = date(parts[0], parts[1], parts[2] if len(parts) == 3 else 1)
                titles.append({"anilist_id": row["anilist_id"], "mal_id": row["mal_id"], "cohort": cohort,
                               "premiere": premiere})  # fmt: skip
    return titles


def step_sparql(raw: Path, mal_ids: list[str]) -> list[dict[str, str]]:
    rows = []
    for index in range(0, len(mal_ids), SPARQL_BATCH):
        batch = mal_ids[index : index + SPARQL_BATCH]
        path = raw / "sparql" / f"batch-{index // SPARQL_BATCH:03d}.json"
        if not (path.exists() and json.loads(path.read_text(encoding="utf-8"))["mal_ids"] == batch):
            query = SPARQL % " ".join(f'"{mal_id}"' for mal_id in batch)
            _, body = get_json("https://query.wikidata.org/sparql?" + urllib.parse.urlencode({"query": query, "format": "json"}))
            save(path, {"mal_ids": batch, "response": body})
            time.sleep(1)
        for binding in json.loads(path.read_text(encoding="utf-8"))["response"]["results"]["bindings"]:
            rows.append({key: value["value"].rsplit("/", 1)[-1] for key, value in binding.items()})
    return rows


def step_entities(raw: Path, items: list[str]) -> dict[str, dict[str, str]]:
    """Wikidata item -> {wiki: article title}."""
    sitelinks: dict[str, dict[str, str]] = {}
    folder = raw / "entities"
    cached = {}
    for path in folder.glob("*.json"):
        cached.update(json.loads(path.read_text(encoding="utf-8"))["response"]["entities"])
    missing = [item for item in items if item not in cached]
    for index in range(0, len(missing), ENTITY_BATCH):
        batch = missing[index : index + ENTITY_BATCH]
        params = {"action": "wbgetentities", "ids": "|".join(batch), "props": "sitelinks",
                  "sitefilter": "|".join(f"{wiki}wiki" for wiki in WIKIS), "format": "json"}  # fmt: skip
        _, body = get_json("https://www.wikidata.org/w/api.php?" + urllib.parse.urlencode(params))
        save(folder / f"{hashlib.sha1('|'.join(batch).encode()).hexdigest()[:16]}.json", {"ids": batch, "response": body})
        cached.update(body["entities"])
    for item in items:
        links = cached.get(item, {}).get("sitelinks", {})
        sitelinks[item] = {key[:-4]: value["title"] for key, value in links.items() if key[:-4] in WIKIS}
    return sitelinks


def step_redirects(raw: Path, wiki: str, titles: list[str]) -> dict[str, list[str]]:
    """Article title -> titles of the redirects pointing at it (main namespace)."""
    folder = raw / "redirects" / wiki
    result: dict[str, list[str]] = {}
    missing = []
    for title in titles:
        path = folder / f"{title_key(wiki, title)}.json"
        if path.exists():
            result[title] = json.loads(path.read_text(encoding="utf-8"))["redirects"]
        else:
            missing.append(title)
    for index in range(0, len(missing), TITLE_BATCH):
        batch = missing[index : index + TITLE_BATCH]
        found: dict[str, list[str]] = {title: [] for title in batch}
        params: dict[str, str] = {"action": "query", "prop": "redirects", "titles": "|".join(batch), "rdlimit": "max",
                                  "rdnamespace": "0", "format": "json", "formatversion": "2"}  # fmt: skip
        while True:
            _, body = get_json(f"https://{wiki}.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params))
            for page in body["query"]["pages"]:
                if page["title"] in found:
                    found[page["title"]].extend(r["title"] for r in page.get("redirects", []))
            if "continue" not in body:
                break
            params.update(body["continue"])
        for title, redirects in found.items():
            save(folder / f"{title_key(wiki, title)}.json", {"wiki": wiki, "title": title, "redirects": redirects})
            result[title] = redirects
    return result


def fetch_pageviews(raw: Path, wiki: str, title: str, start: date, end: date) -> str:
    path = raw / "pageviews" / wiki / f"{title_key(wiki, title)}.json"
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["start"] <= start.isoformat() and record["end"] >= end.isoformat():
            return "cached"
    article = urllib.parse.quote(title.replace(" ", "_"), safe="")
    url = (f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{wiki}.wikipedia/all-access/user/"
           f"{article}/daily/{start:%Y%m%d}/{end:%Y%m%d}")  # fmt: skip
    status, body = get_json(url)
    items = [{"date": f"{i['timestamp'][:4]}-{i['timestamp'][4:6]}-{i['timestamp'][6:8]}", "views": i["views"]}
             for i in (body or {}).get("items", [])]  # fmt: skip
    save(path, {"wiki": wiki, "title": title, "start": start.isoformat(), "end": end.isoformat(), "status": status,
                "items": items})  # fmt: skip
    return "fetched"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Wikipedia pageviews for the AniList TV cohorts.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory (default: data/wikipedia).")
    parser.add_argument("--rate", type=float, default=2.0, help="Requests per second across all threads (default: 2).")
    parser.add_argument("--articles-only", action="store_true", help="Skip the redirect series in step 4.")
    args = parser.parse_args()
    raw = args.out / "raw"
    LIMITER.interval = 1.0 / args.rate
    sys.stdout.reconfigure(line_buffering=True)

    cohort = load_cohort()
    mal_ids = sorted({title["mal_id"] for title in cohort}, key=int)
    print(f"Cohort: {len(cohort)} titles, {len(mal_ids)} MAL ids")

    rows = step_sparql(raw, mal_ids)
    items = sorted({row[var] for row in rows for var in ROLES if var in row})
    print(f"1. Wikidata: {len({row['mal'] for row in rows})} MAL ids have an item; {len(items)} related items")

    sitelinks = step_entities(raw, items)
    articles = {(wiki, title) for links in sitelinks.values() for wiki, title in links.items()}
    print(f"2. Sitelinks: {sum(1 for links in sitelinks.values() if links)} items have en/ja/zh articles; "
          f"{len(articles)} distinct articles")  # fmt: skip

    redirects = {}
    for wiki in WIKIS:
        titles = sorted(title for w, title in articles if w == wiki)
        redirects[wiki] = step_redirects(raw, wiki, titles)
    n_redirects = sum(len(r) for by_title in redirects.values() for r in by_title.values())
    print(f"3. Redirects: {n_redirects} redirects to those articles")

    # Window per article: union of the windows of every title that maps to it.
    premieres_by_mal: dict[str, list[date]] = {}
    for title in cohort:
        premieres_by_mal.setdefault(title["mal_id"], []).append(title["premiere"])
    window: dict[tuple[str, str], tuple[date, date]] = {}
    for row in rows:
        for var in ROLES:
            if var not in row:
                continue
            for wiki, article in sitelinks[row[var]].items():
                for premiere in premieres_by_mal[row["mal"]]:
                    start, end = premiere - timedelta(days=PRE_DAYS), premiere + timedelta(days=POST_DAYS)
                    old = window.get((wiki, article))
                    window[(wiki, article)] = (min(start, old[0]), max(end, old[1])) if old else (start, end)
    yesterday = date.today() - timedelta(days=1)
    # One job per (wiki, title) over the union of every window it is needed for. A page can be
    # both a mapped article and a redirect to another mapped article; two jobs with different
    # windows would write the same file and the narrower one could overwrite the wider.
    article_windows: dict[tuple[str, str], tuple[date, date]] = {}
    redirect_windows: dict[tuple[str, str], tuple[date, date]] = {}

    def widen(target: dict, key: tuple[str, str], start: date, end: date) -> None:
        old = target.get(key)
        target[key] = (min(start, old[0]), max(end, old[1])) if old else (start, end)

    for (wiki, article), (start, end) in window.items():
        widen(article_windows, (wiki, article), start, end)
        for title in redirects[wiki].get(article, []):
            widen(redirect_windows, (wiki, title), start, end)
    for key, (start, end) in list(redirect_windows.items()):
        if key in article_windows:  # fold into the article job, keeping articles first
            widen(article_windows, key, start, end)
            del redirect_windows[key]

    def clipped(windows: dict) -> list[tuple[str, str, date, date]]:
        jobs = []
        for (wiki, title), (start, end) in sorted(windows.items()):
            start, end = max(start, PAGEVIEWS_START), min(end, yesterday)
            if start <= end:
                jobs.append((wiki, title, start, end))
        return jobs

    article_jobs, redirect_jobs = clipped(article_windows), clipped(redirect_windows)
    jobs = article_jobs if args.articles_only else article_jobs + redirect_jobs
    print(f"4. Pageviews: {len(article_jobs)} article series"
          + ("" if args.articles_only else f" + {len(redirect_jobs)} redirect series") + " to check")  # fmt: skip

    counts = {"cached": 0, "fetched": 0}
    with ThreadPoolExecutor(max_workers=PAGEVIEW_WORKERS) as pool:
        futures = [pool.submit(fetch_pageviews, raw, *job) for job in jobs]
        for done, future in enumerate(as_completed(futures), start=1):
            counts[future.result()] += 1
            if done % 1000 == 0 or done == len(futures):
                print(f"   [{done}/{len(futures)}] {counts}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
