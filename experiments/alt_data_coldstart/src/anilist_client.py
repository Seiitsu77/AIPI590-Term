"""AniList GraphQL client for the pilot: paced, cached, logged.

Terms and limits (docs.anilist.co/guide/terms-of-use and /guide/rate-limiting, read
2026-10-07). The API is free for non-commercial use. "Hoarding or mass collection of
data" is prohibited, with stated leniency "for purely educational projects, such as
school assignments". The documented limit is 90 requests a minute, 30 while the API is
"in a degraded state" (that notice was up on the day this was written), plus a burst
limiter with no published number. A 429 carries Retry-After and X-RateLimit-Reset.

This client stays at 20 requests a minute across script runs, waits out Retry-After on
a 429, pauses when X-RateLimit-Remaining runs low, and never repeats a request whose
response is already on disk. The cache is the checkpoint: an interrupted collection
resumes at the first page that is missing.

Every HTTP attempt is appended to cache/request_log.jsonl, and every cache hit is
counted there too. collection_log.json is built from that file, so its success and
failure counts are what happened, not what was planned.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
REQUEST_LOG = CACHE / "request_log.jsonl"
LAST_REQUEST = CACHE / ".last_request"

ENDPOINT = "https://graphql.anilist.co"
HEADERS = {
    "User-Agent": "AnimeColdStartFeasibility/0.1 (Duke AIPI 590 student project) python-requests",
    "Accept": "application/json",
}
REQUESTS_PER_MINUTE = 20
MIN_INTERVAL_SECONDS = 60.0 / REQUESTS_PER_MINUTE
LOW_REMAINING = 3
MAX_ATTEMPTS = 6


def _log(entry: dict) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    with REQUEST_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _pace() -> None:
    """Sleep until MIN_INTERVAL_SECONDS after the previous request, whichever script made it."""
    try:
        previous = float(LAST_REQUEST.read_text())
    except (FileNotFoundError, ValueError):
        previous = 0.0
    wait = previous + MIN_INTERVAL_SECONDS - time.time()
    if wait > 0:
        time.sleep(wait)
    CACHE.mkdir(parents=True, exist_ok=True)
    LAST_REQUEST.write_text(str(time.time()))


def read_cached(name: str) -> dict | None:
    path = CACHE / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def query(name: str, graphql: str, variables: dict) -> dict:
    """Return the `data` of one GraphQL query, from cache/<name>.json when it is there."""
    stored = read_cached(name)
    if stored is not None:
        if stored["variables"] != variables:
            raise RuntimeError(f"cache/{name}.json was fetched with different variables; delete it to refetch")
        _log({"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "name": name, "outcome": "cache_hit"})
        return stored["data"]

    for attempt in range(1, MAX_ATTEMPTS + 1):
        _pace()
        entry = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "name": name, "attempt": attempt}
        try:
            response = requests.post(ENDPOINT, json={"query": graphql, "variables": variables}, headers=HEADERS, timeout=(10, 60))
        except requests.RequestException as error:
            _log({**entry, "status": None, "outcome": "connection_error", "error": type(error).__name__})
            time.sleep(15 * attempt)
            continue

        remaining = response.headers.get("X-RateLimit-Remaining")
        entry.update(
            status=response.status_code,
            seconds=round(response.elapsed.total_seconds(), 3),
            limit=response.headers.get("X-RateLimit-Limit"),
            remaining=remaining,
            retry_after=response.headers.get("Retry-After"),
        )
        try:
            body = response.json()
        except ValueError:
            body = None
        errors = (body or {}).get("errors")

        if response.status_code == 200 and body and not errors:
            _log({**entry, "outcome": "ok"})
            path = CACHE / f"{name}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(".part")
            partial.write_text(
                json.dumps({"fetched_utc": entry["utc"], "variables": variables, "data": body["data"]}, ensure_ascii=False),
                encoding="utf-8",
            )
            os.replace(partial, path)
            if remaining is not None and int(remaining) <= LOW_REMAINING:
                time.sleep(30)
            return body["data"]
        if response.status_code == 429:
            _log({**entry, "outcome": "rate_limited"})
            time.sleep(float(response.headers.get("Retry-After", 60)) + 1.0)
            continue
        if response.status_code >= 500:
            _log({**entry, "outcome": "server_error"})
            time.sleep(15 * attempt)
            continue
        _log({**entry, "outcome": "rejected", "errors": json.dumps(errors)[:400]})
        raise RuntimeError(f"AniList rejected {name}: {response.status_code} {json.dumps(errors)[:400]}")
    raise RuntimeError(f"AniList request {name} failed {MAX_ATTEMPTS} times; rerun to resume from the cache")
