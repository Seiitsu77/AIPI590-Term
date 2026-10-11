"""EXPLORATORY (post hoc): extend the article mapping to the source work.

Written after the pre-specified test in analyze.py had been run and read. It does not
change that test or its verdict; its output feeds only `analyze.py --exploratory`.

Why it exists. The pre-specified mapping took each title's own article plus its
*nearest* parent article. For the most popular shows that resolved to pages such as
"My Hero Academia season 1" and "My Hero Academia (TV series)". Both were split out of
the main article years after the premiere, so they have no pre-air views at all. The
attention before the premiere was on the main "My Hero Academia" article, which Wikidata
models as the source manga and which sits one step further up the chain. The count of
popular titles recorded with zero pre-air views for this reason is in the `diagnostics`
block of results/feasibility_report.json.

The rule here is the one the first mapping should have used, chosen on principle and
not by trying alternatives against the outcome: a title's attention is the sum over its
own article, its series article and the article of the work it is based on.

    item --P179--> series
    item --P144--> source work
    item --P179--> series --P144--> source work

Usage:
    python src/map_wikipedia_exploratory.py
"""

from __future__ import annotations

import pandas as pd

from map_wikipedia import BATCH, LANGS, OUT, SITELINKS, sparql, tail, titles_of

DIRECT = (
    "SELECT ?item ?p ?target ?en ?ja ?zh WHERE { VALUES ?item { %s } VALUES ?p { wdt:P179 wdt:P144 } "
    "?item ?p ?target ." + SITELINKS + "}"
)
VIA_SERIES = (
    "SELECT ?item ?target ?en ?ja ?zh WHERE { VALUES ?item { %s } "
    "?item wdt:P179 ?mid . ?mid wdt:P144 ?target ." + SITELINKS + "}"
)


def main() -> None:
    mapping = pd.read_csv(OUT / "mal_wikipedia_map.csv")
    qids = sorted(mapping["qid"].unique())
    rows = []
    for start in range(0, len(qids), BATCH):
        values = " ".join(f"wd:{qid}" for qid in qids[start : start + BATCH])
        for template, relation in ((DIRECT, None), (VIA_SERIES, "series_source")):
            for binding in sparql(template % values)[0]:
                kind = relation or {"P179": "series", "P144": "source"}[tail(binding, "p")]
                for lang in LANGS:
                    title = titles_of(binding)[f"{lang}_title"]
                    if title:
                        rows.append({"qid": tail(binding, "item"), "lang": lang, "title": title, "relation": kind})
    extra = pd.DataFrame(rows).drop_duplicates(["qid", "lang", "title"])
    extra = mapping[["MAL_ID", "qid"]].merge(extra, on="qid")
    extra.to_csv(OUT / "mal_wikipedia_extra_articles.csv", index=False)

    known = {(lang, title) for lang in LANGS for kind in ("own", "parent") for title in mapping[f"{kind}_{lang}_title"].dropna()}
    new = {(lang, title) for lang, title in zip(extra["lang"], extra["title"], strict=True)} - known
    print(f"extra article links: {len(extra)} across {extra['MAL_ID'].nunique()} titles")
    print(extra["relation"].value_counts().rename("by relation"))
    print(f"articles not fetched yet: {len(new)}")


if __name__ == "__main__":
    main()
