"""EXPLORATORY (post hoc): how much does attention in the first days after the premiere add?

Written after both runs of analyze.py had been read. The look-ahead arm there showed that
pageviews from the 28 days after the premiere carry roughly twice the signal of pre-air
pageviews. That prompted a different question, which is still a cold-start question:

    One week after the premiere a show has almost no ratings, so collaborative filtering
    still cannot score it. But its first episode has aired, and how many people looked
    it up that week is known. Does that predict the eventual audience beyond metadata
    and beyond what was already known before the premiere?

Using days 0..6 is not look-ahead for a decision taken on day 7. It would be look-ahead
for a decision taken on premiere day, which is why analyze.py labels its post-air arm
as unusable. The two analyses answer different questions about different decision dates.

Definitions, fixed before this script was first run:

  Mapping    the corrected (exploratory) article mapping.
  Horizons   3, 7, 14 and 28 days after the premiere. Day 7 is the headline.
  Features   per language, log1p(views on days 0..h-1), added to the six pre-air features.
  Compare    metadata + pre-air + first h days   against   metadata only,
             and against metadata + pre-air (what the extra days add).
  Everything else (sample, split, models, metric, bootstrap) is as in analyze.py.

Not tested here, and it matters: a system running on day 7 could also read MAL's own
first-week member counts. The snapshot has no time dimension, so that in-domain
comparison is missing.

Usage:
    python src/analyze_after_premiere.py
"""

from __future__ import annotations

import json

import pandas as pd

from analyze import (
    ALT,
    LANGS,
    METRICS,
    PRIMARY,
    RESULTS,
    TEST_START,
    baseline_design,
    boosted,
    bootstrap_deltas,
    build_frame,
    evaluate,
    ridge,
)

HORIZONS = (3, 7, 14, 28)
HEADLINE = 7


def score(train: pd.DataFrame, test: pd.DataFrame, arms: dict[str, list[str]], make_model) -> dict:
    """Fit every arm on `train`, score on `test`, and bootstrap the paired differences."""
    base_train, base_test = baseline_design(train, test, with_type=False)
    y, n_pos, season = test["y"].to_numpy(float), test["n_pos"].to_numpy(float), test["season"].to_numpy(str)
    preds = {}
    for arm, columns in arms.items():
        x_train = pd.concat([base_train, train[columns]], axis=1).to_numpy(float)
        x_test = pd.concat([base_test, test[columns]], axis=1).to_numpy(float)
        preds[arm] = make_model().fit(x_train, train["y"].to_numpy(float)).predict(x_test)
    return {
        "arms": {arm: evaluate(y, pred, n_pos, season) for arm, pred in preds.items()},
        "delta_vs_meta": bootstrap_deltas(y, preds, n_pos, season, "meta"),
        "delta_vs_meta_plus_pre": bootstrap_deltas(y, {k: v for k, v in preds.items() if k != "meta"}, n_pos, season, "meta+pre"),
    }


def main() -> None:
    windows = {f"first{h}": (0, h - 1) for h in HORIZONS}
    frame = build_frame(exploratory=True, windows=windows)
    sample = frame[(frame["Type"] == "TV") & frame["mapped_any"]]
    if not sample["pv_complete"].all():
        raise SystemExit("some primary-sample titles still have unfetched articles; run fetch_pageviews.py")
    train, test = sample[sample["start"] < TEST_START], sample[sample["start"] >= TEST_START]

    arms = {"meta": [], "meta+pre": ALT}
    for h in HORIZONS:
        arms[f"meta+pre+first{h}"] = [*ALT, *[f"pv_first{h}_{lang}" for lang in LANGS]]

    report: dict = {
        "definitions": {
            "status": "EXPLORATORY, post hoc",
            "article_mapping": "own + nearest parent + series + source-work articles",
            "horizons_days_after_premiere": list(HORIZONS),
            "headline_horizon": HEADLINE,
            "sample": "tv_mapped",
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "primary_metric": PRIMARY,
        },
        "models": {},
    }
    models = (("ridge", ridge), ("boosted", boosted))
    for model_name, make_model in models:
        report["models"][model_name] = score(train, test, arms, make_model)

    # Same consistency check analyze.py runs: one test year at a time, headline horizon.
    headline = f"meta+pre+first{HEADLINE}"
    headline_arms = {arm: arms[arm] for arm in ("meta", "meta+pre", headline)}
    report["by_test_year"] = {}
    for year in (2018, 2019, 2020):
        year_start, year_end = pd.Timestamp(year=year, month=1, day=1), pd.Timestamp(year=year + 1, month=1, day=1)
        year_train = sample[sample["start"] < year_start]
        year_test = sample[(sample["start"] >= year_start) & (sample["start"] < year_end)]
        report["by_test_year"][str(year)] = {"n_train": int(len(year_train)), "n_test": int(len(year_test))}
        for model_name, make_model in models:
            block = score(year_train, year_test, headline_arms, make_model)
            report["by_test_year"][str(year)][model_name] = {
                "meta": block["arms"]["meta"][PRIMARY],
                headline: block["arms"][headline][PRIMARY],
                "delta_vs_meta": block["delta_vs_meta"][headline][PRIMARY],
                "delta_vs_meta_plus_pre": block["delta_vs_meta_plus_pre"][headline][PRIMARY],
            }

    (RESULTS / "exploratory_after_premiere_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    def show(triple: list[float]) -> str:
        return f"{triple[0]:+.3f} [{triple[1]:+.3f}, {triple[2]:+.3f}]"

    for model_name, block in report["models"].items():
        print(f"\n{model_name}  (train {len(train)}, test {len(test)})")
        for arm in arms:
            line = f"  {arm:22s} " + "  ".join(f"{metric.split('_')[0]} {block['arms'][arm][metric]:.3f}" for metric in METRICS)
            if arm != "meta":
                line += f"  | vs meta {show(block['delta_vs_meta'][arm][PRIMARY])}"
            if arm not in ("meta", "meta+pre"):
                line += f"  | vs meta+pre {show(block['delta_vs_meta_plus_pre'][arm][PRIMARY])}"
            print(line)
    print(f"\none test year at a time, {headline}:")
    for year, block in report["by_test_year"].items():
        for model_name, _ in models:
            b = block[model_name]
            print(f"  {year} {model_name:8s} n={block['n_test']}  meta {b['meta']:.3f} -> {b[headline]:.3f}  vs meta {show(b['delta_vs_meta'])}  vs meta+pre {show(b['delta_vs_meta_plus_pre'])}")


if __name__ == "__main__":
    main()
