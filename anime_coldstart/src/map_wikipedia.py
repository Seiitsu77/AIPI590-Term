"""Map MAL IDs to Wikipedia articles through Wikidata.

Wikidata property P4086 is the "MyAnimeList anime ID". The property's label is checked
against Wikidata before the mapping is trusted, so a wrong property number fails loudly
instead of producing a plausible but meaningless join.

Each cohort title gets up to two articles per language:

  own     the article attached to the item that carries the MAL ID. Wikidata usually
          hangs the ID on a season-level item, so this is often a season article
          ("Dr. Stone season 1") or nothing at all.
  parent  the article of the nearest series, source work or franchise item, reached by
          walking: part of the series (P179), based on (P144), media franchise (P8345),
          part of (P361), in that order of preference, at most two hops.

Both are kept because season articles are typically created after the premiere. Before
a show airs, the attention is on the series or manga article, and that is the page whose
pre-air views are the signal under test. Which article to use was decided from the
mapping alone, before any pageview was compared against an outcome.

Limitations, reported downstream rather than hidden:

* It is today's mapping. Being in Wikidata at all correlates with how popular a title
  became, so coverage itself carries look-ahead. Views are point-in-time; coverage is not.
* Pageviews are logged under the title a page had on that day. A page moved since then
  shows too few early views. That understates the pre-air signal; it cannot inflate it.

If a request fails, this script raises. It never substitutes placeholder data.

Usage:
    python src/build_cohort.py && python src/map_wikipedia.py
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data"

MAL_PROPERTY = "P4086"
EXPECTED_LABEL = "MyAnimeList anime ID"
SPARQL_URL = "https://query.wikidata.org/sparql"
ENTITY_URL = "https://www.wikidata.org/w/api.php"
LANGS = ("en", "ja", "zh")
TITLE_COLS = [f"{lang}_title" for lang in LANGS]
# Wikimedia asks automated clients to identify themselves. No personal contact is sent.
HEADERS = {"User-Agent": "AnimeColdStartFeasibility/0.1 (Duke AIPI 590 student project; python-requests)"}

SITELINKS = """
  OPTIONAL { ?en schema:about ?target ; schema:isPartOf <https://en.wikipedia.org/> . }
  OPTIONAL { ?ja schema:about ?target ; schema:isPartOf <https://ja.wikipedia.org/> . }
  OPTIONAL { ?zh schema:about ?target ; schema:isPartOf <https://zh.wikipedia.org/> . }
"""
QUERY = "SELECT ?target ?mal ?en ?ja ?zh WHERE { ?target wdt:P4086 ?mal ." + SITELINKS + "}"

# Preference order: the anime series first, then the source work, then wider groupings.
PARENT_PROPERTIES = ("P179", "P144", "P8345", "P361")
PARENT_VALUES = " ".join(f"wdt:{prop}" for prop in PARENT_PROPERTIES)
HOP1 = (
    "SELECT ?item ?p ?target ?en ?ja ?zh WHERE { VALUES ?item { %s } VALUES ?p { %s } "
    "?item ?p ?target ." + SITELINKS + "}"
)
HOP2 = (
    "SELECT ?item ?p ?target ?en ?ja ?zh WHERE { VALUES ?item { %s } VALUES ?q { %s } VALUES ?p { %s } "
    "?item ?q ?mid . ?mid ?p ?target ." + SITELINKS + "}"
)
BATCH = 250


def verify_property() -> str:
    response = requests.get(
        ENTITY_URL,
        params={"action": "wbgetentities", "ids": MAL_PROPERTY, "props": "labels", "languages": "en", "format": "json"},
        headers=HEADERS,
        timeout=60,
    )
    response.raise_for_status()
    label = response.json()["entities"][MAL_PROPERTY]["labels"]["en"]["value"]
    if label != EXPECTED_LABEL:
        raise RuntimeError(f"{MAL_PROPERTY} is labelled {label!r}, expected {EXPECTED_LABEL!r}")
    return label


def sparql(query: str) -> tuple[list[dict], bytes]:
    response = requests.post(SPARQL_URL, data={"query": query, "format": "json"}, headers=HEADERS, timeout=180)
    response.raise_for_status()
    return response.json()["results"]["bindings"], response.content


def tail(binding: dict, key: str) -> str:
    return binding[key]["value"].rsplit("/", 1)[1]


def article_title(url: str | None) -> str | None:
    """Article title in the form the pageviews API expects (underscores, decoded)."""
    return unquote(url.rsplit("/wiki/", 1)[1]) if url else None


def titles_of(binding: dict) -> dict[str, str | None]:
    return {f"{lang}_title": article_title(binding.get(lang, {}).get("value")) for lang in LANGS}


def parent_articles(qids: list[str], template: str, hops: int) -> pd.DataFrame:
    """Preferred parent item with at least one article, for each item in `qids`."""
    rows = []
    for start in range(0, len(qids), BATCH):
        values = " ".join(f"wd:{qid}" for qid in qids[start : start + BATCH])
        args = (values, PARENT_VALUES) if hops == 1 else (values, PARENT_VALUES, PARENT_VALUES)
        for binding in sparql(template % args)[0]:
            rows.append(
                {
                    "qid": tail(binding, "item"),
                    "parent_qid": tail(binding, "target"),
                    "parent_prop": tail(binding, "p"),
                    **titles_of(binding),
                }
            )
    columns = ["qid", "parent_qid", "parent_prop", *TITLE_COLS]
    if not rows:
        return pd.DataFrame(columns=columns)
    found = pd.DataFrame(rows).drop_duplicates()
    found["n_sitelinks"] = found[TITLE_COLS].notna().sum(axis=1)
    found = found[(found["n_sitelinks"] > 0) & (found["qid"] != found["parent_qid"])]
    # Deterministic choice among several parents: property preference, then the item
    # with the most articles, then the lowest item number. Nothing here sees an outcome.
    found["priority"] = found["parent_prop"].map({prop: rank for rank, prop in enumerate(PARENT_PROPERTIES)})
    found["qnum"] = found["parent_qid"].str[1:].astype(int)
    found = found.sort_values(["qid", "priority", "n_sitelinks", "qnum"], ascending=[True, True, False, True])
    return found.drop_duplicates("qid")[columns]


def main() -> None:
    label = verify_property()
    print(f"{MAL_PROPERTY} label verified: {label}")

    bindings, payload = sparql(QUERY)
    retrieved = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (RAW / "wikidata_mal_sitelinks.json").write_bytes(payload)

    rows = [
        {"MAL_ID": int(binding["mal"]["value"]), "qid": tail(binding, "target"), **titles_of(binding)}
        for binding in bindings
        if binding["mal"]["value"].isdigit()
    ]
    links = pd.DataFrame(rows).drop_duplicates()
    links["n_sitelinks"] = links[TITLE_COLS].notna().sum(axis=1)
    # A MAL ID can sit on more than one item. Keep the item with the most articles.
    links = links.sort_values(["MAL_ID", "n_sitelinks", "qid"], ascending=[True, False, True])
    own = links.drop_duplicates("MAL_ID").drop(columns="n_sitelinks")
    own = own.rename(columns={column: f"own_{column}" for column in TITLE_COLS})

    cohort_ids = set(pd.read_csv(OUT / "cohort.csv")["MAL_ID"])
    own = own[own["MAL_ID"].isin(cohort_ids)].copy()

    qids = sorted(own["qid"].unique())
    hop1 = parent_articles(qids, HOP1, 1).assign(parent_hops=1)
    remaining = sorted(set(qids) - set(hop1["qid"]))
    hop2 = parent_articles(remaining, HOP2, 2).assign(parent_hops=2)
    parents = pd.concat([hop1, hop2], ignore_index=True)
    parents = parents.rename(columns={column: f"parent_{column}" for column in TITLE_COLS})

    mapping = own.merge(parents, on="qid", how="left")
    mapping.to_csv(OUT / "mal_wikipedia_map.csv", index=False)

    manifest = {
        "source": SPARQL_URL,
        "query": " ".join(QUERY.split()),
        "parent_properties_in_preference_order": list(PARENT_PROPERTIES),
        "property": MAL_PROPERTY,
        "property_label_verified": label,
        "retrieved_utc": retrieved,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "wikidata_rows_with_mal_id": len(rows),
        "cohort_titles_in_wikidata": int(len(mapping)),
        "with_parent_1_hop": int(len(hop1)),
        "with_parent_2_hops": int(len(hop2)),
    }
    (RAW / "wikidata_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("query", "sha256", "source")}, indent=1))
    print(mapping["parent_prop"].fillna("no parent").value_counts().rename("parent reached through"))


if __name__ == "__main__":
    main()
