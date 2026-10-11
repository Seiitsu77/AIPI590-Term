"""Feasibility test: do pre-air Wikipedia pageviews add signal beyond metadata?

The claim under test is narrow. A new anime has no ratings on the day it premieres.
Metadata (studio, source, franchise) is known. Wikipedia attention in the weeks before
the premiere is also known. Does the attention predict how many people end up liking
the show, *over and above* what the metadata already predicts?

Every definition below was fixed before any feature was compared against the outcome.

  Outcome    y = log1p(MAL users who rated the title 8+), from the early-2021 snapshot.
  Alt data   per language (en, ja, zh), summed over the title's own article and its
             parent (series / source work) article:
               pv_pre30 = log1p(views on days -30..-1 before the premiere)
               pv_base  = log1p(views on days -90..-31)
             Days are UTC. A Japanese premiere on day 0 airs after UTC day -1 has ended,
             so the window is strictly pre-air.
  Baseline   type, source, age rating, genres, episode count, premiere quarter,
             studio prior, franchise prior (see build_cohort.py).
  Split      train on premieres 2016-2018, test on premieres 2019-2020. Temporal, never
             random: the question is whether the relationship holds going forward.
  Model      ridge regression, features standardised on the training set only, penalty
             chosen by cross-validation inside the training set only.
  Primary    within-season Spearman correlation on the test set. A cold-start prior
  metric     only has to rank the shows premiering in the same season, and ranking
             within a season also removes the fact that 2020 titles had less time to
             collect ratings before the snapshot.
  Primary    TV titles that map to at least one Wikipedia article.
  sample
  Decision   PASS if (baseline + alt) minus (baseline) is positive and the lower bound of
             its 95% paired bootstrap interval is above zero. FAIL otherwise, including
             a positive estimate whose interval contains zero.

Secondary metrics, samples, models and subgroups are reported but are not part of the
decision. The "post-air" arm is a deliberate look-ahead demonstration, not a result.

Usage:
    python src/analyze.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

LANGS = ("en", "ja", "zh")
TEST_START = pd.Timestamp("2019-01-01")
PRE_WINDOW = (-30, -1)
BASE_WINDOW = (-90, -31)
POST_WINDOW = (0, 27)
MIN_LEVEL_ROWS = 15
MIN_GENRE_SHARE = 0.03
MIN_SEASON_ROWS = 8
BOOTSTRAP_DRAWS = 2000
SEED = 12345

ALT = [f"pv_{kind}_{lang}" for lang in LANGS for kind in ("pre30", "base")]
POST = [f"pv_post_{lang}" for lang in LANGS]
METRICS = ("spearman_within_season", "r2_within_season", "top_quintile_hit_rate", "ndcg_at_10")
PRIMARY = "spearman_within_season"


# ------------------------------------------------------------------ features


def add_pageview_features(
    frame: pd.DataFrame,
    extra: dict[tuple[int, str], set[str]] | None = None,
    windows: dict[str, tuple[int, int]] | None = None,
) -> pd.DataFrame:
    """Window sums cut from the daily matrix, relative to each title's premiere.

    `extra` is used only by the post hoc exploratory run. It adds series and source-work
    articles to a title's views. It never changes which titles count as mapped, so the
    exploratory run scores exactly the same rows as the pre-specified one.

    `windows` adds further named windows as `pv_<name>_<lang>` columns, for analyses
    that observe the show for some days after its premiere.
    """
    extra = extra or {}
    windows = windows or {}
    index = pd.read_csv(DATA / "pageviews_index.csv")
    payload = np.load(DATA / "pageviews_daily.npz")
    views = payload["views"]
    first_day = pd.Timestamp(str(payload["first_day"]))
    row_of = {(lang, title): row for row, lang, title in zip(index["row"], index["lang"], index["title"], strict=True)}
    fetched = {(lang, title): status != "not_fetched" for lang, title, status in zip(index["lang"], index["title"], index["status"], strict=True)}
    n_days = views.shape[1]

    def window_sum(rows: list[int], start: pd.Timestamp, window: tuple[int, int]) -> float:
        lo = (start - first_day).days + window[0]
        hi = (start - first_day).days + window[1]
        if lo < 0 or hi >= n_days:
            raise ValueError(f"window {window} around {start.date()} falls outside the pageview range")
        return float(views[rows, lo : hi + 1].sum()) if rows else 0.0

    out = frame.copy()
    mapped_any = np.zeros(len(out), dtype=bool)
    # An article that has not been fetched is not the same as an article with no views.
    # Rows touching one are marked incomplete and no sample is allowed to include them.
    complete = np.ones(len(out), dtype=bool)
    for lang in LANGS:
        pre, base, post, mapped = [], [], [], []
        more: dict[str, list[float]] = {name: [] for name in windows}
        for position, (mal_id, own, parent, start) in enumerate(
            zip(out["MAL_ID"], out[f"own_{lang}_title"], out[f"parent_{lang}_title"], out["start"], strict=True)
        ):
            titles = {title for title in (own, parent) if isinstance(title, str)}
            mapped.append(bool(titles))
            titles |= extra.get((int(mal_id), lang), set())
            rows = sorted(row_of[(lang, title)] for title in titles)
            if not all(fetched[(lang, title)] for title in titles):
                complete[position] = False
            pre.append(window_sum(rows, start, PRE_WINDOW))
            base.append(window_sum(rows, start, BASE_WINDOW))
            post.append(window_sum(rows, start, POST_WINDOW))
            for name, window in windows.items():
                more[name].append(window_sum(rows, start, window))
        for name, values in more.items():
            out[f"pv_{name}_{lang}"] = np.log1p(values)
        out[f"mapped_{lang}"] = mapped
        out[f"views_pre30_{lang}"] = pre
        out[f"pv_pre30_{lang}"] = np.log1p(pre)
        out[f"pv_base_{lang}"] = np.log1p(base)
        out[f"pv_post_{lang}"] = np.log1p(post)
        mapped_any |= np.asarray(mapped)
    out["mapped_any"] = mapped_any
    out["pv_complete"] = complete
    preair = (out[[f"views_pre30_{lang}" for lang in LANGS]].sum(axis=1) > 0).astype(float)
    out["preair_views_any"] = preair.where(complete)
    return out


def one_hot(train: pd.Series, test: pd.Series, prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-hot with levels chosen on the training set; rare and unseen levels pool."""
    counts = train.value_counts()
    keep = set(counts[counts >= MIN_LEVEL_ROWS].index)
    frames = []
    for series in (train, test):
        pooled = series.where(series.isin(keep), "Other")
        frames.append(pd.get_dummies(pooled, prefix=prefix, dtype=float))
    columns = sorted(set(frames[0].columns))
    return frames[0].reindex(columns=columns, fill_value=0.0), frames[1].reindex(columns=columns, fill_value=0.0)


def baseline_design(train: pd.DataFrame, test: pd.DataFrame, *, with_type: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Metadata features. Every imputation value and level set comes from `train`."""
    parts_train, parts_test = [], []
    categoricals = ["Source", "Rating", "quarter"] + (["Type"] if with_type else [])
    for column in categoricals:
        a, b = one_hot(train[column].astype(str), test[column].astype(str), column)
        parts_train.append(a)
        parts_test.append(b)

    train_genres = train["Genres"].str.split(", ")
    share = train_genres.explode().value_counts() / len(train)
    genres = sorted(share[share >= MIN_GENRE_SHARE].index)
    for frame, parts in ((train, parts_train), (test, parts_test)):
        sets = frame["Genres"].str.split(", ").map(set)
        parts.append(pd.DataFrame({f"genre_{g}": sets.map(lambda s, g=g: float(g in s)) for g in genres}, index=frame.index))

    episode_median = float(np.log1p(train["episodes"]).median())
    studio_mean = float(train["studio_prior"].mean())
    for frame, parts in ((train, parts_train), (test, parts_test)):
        numeric = pd.DataFrame(index=frame.index)
        numeric["log_episodes"] = np.log1p(frame["episodes"]).fillna(episode_median)
        numeric["episodes_missing"] = frame["episodes"].isna().astype(float)
        numeric["is_sequel"] = frame["is_sequel"].astype(float)
        numeric["franchise_prior"] = frame["franchise_prior"].fillna(0.0)
        numeric["log_franchise_n"] = np.log1p(frame["franchise_n"])
        numeric["studio_prior"] = frame["studio_prior"].fillna(studio_mean)
        numeric["studio_missing"] = frame["studio_prior"].isna().astype(float)
        numeric["log_studio_n"] = np.log1p(frame["studio_n"])
        parts.append(numeric)
    return pd.concat(parts_train, axis=1), pd.concat(parts_test, axis=1)


# ------------------------------------------------------------------- metrics


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def evaluate(y: np.ndarray, pred: np.ndarray, n_pos: np.ndarray, season: np.ndarray) -> dict[str, float]:
    """All four metrics, each computed inside a premiere season and then combined."""
    rho_sum = rho_weight = 0.0
    hits = picks = 0.0
    ndcgs = []
    sse = sst = 0.0
    for label in np.unique(season):
        mask = season == label
        size = int(mask.sum())
        if size < MIN_SEASON_ROWS:
            continue
        ys, ps, gains = y[mask], pred[mask], n_pos[mask]
        rho = spearman(ps, ys)
        if not np.isnan(rho):
            rho_sum += rho * size
            rho_weight += size
        yd, pd_ = ys - ys.mean(), ps - ps.mean()
        sse += float(((yd - pd_) ** 2).sum())
        sst += float((yd**2).sum())
        k = max(1, round(0.2 * size))
        top_pred = set(np.argsort(-ps, kind="stable")[:k])
        top_true = set(np.argsort(-ys, kind="stable")[:k])
        hits += len(top_pred & top_true)
        picks += k
        depth = min(10, size)
        discount = 1.0 / np.log2(np.arange(2, depth + 2))
        dcg = float((gains[np.argsort(-ps, kind="stable")[:depth]] * discount).sum())
        ideal = float((np.sort(gains)[::-1][:depth] * discount).sum())
        if ideal > 0:
            ndcgs.append(dcg / ideal)
    return {
        "spearman_within_season": rho_sum / rho_weight if rho_weight else float("nan"),
        "r2_within_season": 1.0 - sse / sst if sst else float("nan"),
        "top_quintile_hit_rate": hits / picks if picks else float("nan"),
        "ndcg_at_10": float(np.mean(ndcgs)) if ndcgs else float("nan"),
    }


def bootstrap_deltas(
    y: np.ndarray, preds: dict[str, np.ndarray], n_pos: np.ndarray, season: np.ndarray, reference: str
) -> dict[str, dict[str, list[float]]]:
    """Paired bootstrap over test titles, resampled inside each season.

    Returns, per arm and metric, [point estimate, 2.5th, 97.5th percentile] of
    (arm minus reference). Pairing matters: both arms are scored on the same draw.
    """
    generator = np.random.default_rng(SEED)
    groups = [np.flatnonzero(season == label) for label in np.unique(season)]
    arms = [arm for arm in preds if arm != reference]
    draws = {arm: {metric: [] for metric in METRICS} for arm in arms}
    for _ in range(BOOTSTRAP_DRAWS):
        index = np.concatenate([generator.choice(group, size=len(group), replace=True) for group in groups])
        base = evaluate(y[index], preds[reference][index], n_pos[index], season[index])
        for arm in arms:
            scored = evaluate(y[index], preds[arm][index], n_pos[index], season[index])
            for metric in METRICS:
                draws[arm][metric].append(scored[metric] - base[metric])
    point_base = evaluate(y, preds[reference], n_pos, season)
    out: dict[str, dict[str, list[float]]] = {}
    for arm in arms:
        point = evaluate(y, preds[arm], n_pos, season)
        out[arm] = {}
        for metric in METRICS:
            values = np.asarray(draws[arm][metric])
            values = values[~np.isnan(values)]
            low, high = np.percentile(values, [2.5, 97.5])
            out[arm][metric] = [point[metric] - point_base[metric], float(low), float(high)]
    return out


# -------------------------------------------------------------------- models


def ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 26)))


def boosted():
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, random_state=SEED)


def fit_arms(train: pd.DataFrame, test: pd.DataFrame, *, with_type: bool, make_model) -> dict[str, np.ndarray]:
    """Predictions on `test` for each feature arm. Nothing is fitted on `test`."""
    base_train, base_test = baseline_design(train, test, with_type=with_type)
    arms = {
        "meta": [],
        "meta+alt": ALT,
        "alt_only": ALT,
        **{f"meta+{lang}": [f"pv_pre30_{lang}", f"pv_base_{lang}"] for lang in LANGS},
        "meta+postair_LOOKAHEAD": POST,
    }
    preds = {}
    for arm, extra in arms.items():
        if arm == "alt_only":
            x_train, x_test = train[extra], test[extra]
        else:
            x_train = pd.concat([base_train, train[extra]], axis=1)
            x_test = pd.concat([base_test, test[extra]], axis=1)
        model = make_model().fit(x_train.to_numpy(float), train["y"].to_numpy(float))
        preds[arm] = model.predict(x_test.to_numpy(float))
    return preds


def run(sample: pd.DataFrame, *, with_type: bool, make_model, test_from: pd.Timestamp, test_to: pd.Timestamp | None = None) -> dict:
    train = sample[sample["start"] < test_from]
    test = sample[sample["start"] >= test_from]
    if test_to is not None:
        test = test[test["start"] < test_to]
    preds = fit_arms(train, test, with_type=with_type, make_model=make_model)
    y, n_pos, season = test["y"].to_numpy(float), test["n_pos"].to_numpy(float), test["season"].to_numpy(str)
    result = {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "arms": {arm: evaluate(y, pred, n_pos, season) for arm, pred in preds.items()},
        "delta_vs_meta": bootstrap_deltas(y, preds, n_pos, season, "meta"),
    }
    groups = {"new_ip": test["is_sequel"].to_numpy() == 0, "sequel": test["is_sequel"].to_numpy() == 1}
    result["subgroups"] = {}
    for name, mask in groups.items():
        sub = {arm: pred[mask] for arm, pred in preds.items() if arm in ("meta", "meta+alt")}
        result["subgroups"][name] = {
            "n_test": int(mask.sum()),
            "arms": {arm: evaluate(y[mask], pred, n_pos[mask], season[mask]) for arm, pred in sub.items()},
            "delta_vs_meta": bootstrap_deltas(y[mask], sub, n_pos[mask], season[mask], "meta"),
        }
    result["_predictions"] = test[["MAL_ID", "Name", "season", "is_sequel", "n_pos", "y"]].assign(
        pred_meta=preds["meta"], pred_meta_alt=preds["meta+alt"]
    )
    return result


# ------------------------------------------------------------------ coverage


def coverage_tables(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Where the holes are. Popularity quintiles are computed inside each type."""
    flags = ["in_wikidata", "mapped_en", "mapped_ja", "mapped_zh", "mapped_any", "preair_views_any"]
    tables = {"by_type": frame.groupby("Type")[flags].mean().assign(n=frame.groupby("Type").size())}
    tv = frame[frame["Type"] == "TV"].copy()
    tv["popularity_quintile"] = pd.qcut(tv["n_pos"].rank(method="first"), 5, labels=["Q1_least", "Q2", "Q3", "Q4", "Q5_most"])
    tables["tv_by_popularity"] = tv.groupby("popularity_quintile", observed=True)[flags].mean().assign(
        n=tv.groupby("popularity_quintile", observed=True).size(),
        median_n_pos=tv.groupby("popularity_quintile", observed=True)["n_pos"].median(),
    )
    tv["kind"] = tv["is_sequel"].map({0: "new_ip", 1: "sequel"})
    tables["tv_by_sequel"] = tv.groupby("kind")[flags].mean().assign(n=tv.groupby("kind").size())
    tables["tv_by_year"] = tv.groupby(tv["start"].dt.year)[flags].mean().assign(n=tv.groupby(tv["start"].dt.year).size())
    return tables


def build_frame(*, exploratory: bool, windows: dict[str, tuple[int, int]] | None = None) -> pd.DataFrame:
    """Cohort joined to its article mapping, with pageview features attached."""
    cohort = pd.read_csv(DATA / "cohort.csv", parse_dates=["start"])
    mapping = pd.read_csv(DATA / "mal_wikipedia_map.csv")
    frame = cohort.merge(mapping, on="MAL_ID", how="left")
    frame["in_wikidata"] = frame["qid"].notna()
    frame["quarter"] = "Q" + frame["start"].dt.quarter.astype(str)
    extra: dict[tuple[int, str], set[str]] = {}
    if exploratory:
        links = pd.read_csv(DATA / "mal_wikipedia_extra_articles.csv")
        for mal_id, lang, title in zip(links["MAL_ID"], links["lang"], links["title"], strict=True):
            extra.setdefault((int(mal_id), lang), set()).add(title)
    return add_pageview_features(frame, extra, windows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exploratory",
        action="store_true",
        help="post hoc run: add series and source-work articles (map_wikipedia_exploratory.py); writes separate outputs",
    )
    args = parser.parse_args()
    # The exploratory run writes under its own names and never overwrites the
    # pre-specified outputs.
    tag = "exploratory_" if args.exploratory else ""

    RESULTS.mkdir(exist_ok=True)
    frame = build_frame(exploratory=args.exploratory)
    frame.drop(columns=["root"]).to_csv(DATA / f"{tag}analysis_panel.csv", index=False)

    tables = coverage_tables(frame)
    for name, table in tables.items():
        table.round(4).to_csv(RESULTS / f"{tag}coverage_{name}.csv")

    tv = frame[frame["Type"] == "TV"]
    samples = {
        "tv_mapped": (tv[tv["mapped_any"]], False),
        "tv_all": (tv, False),
        "all_types_mapped": (frame[frame["mapped_any"]], True),
    }

    report: dict = {
        "definitions": {
            "article_mapping": (
                "EXPLORATORY, post hoc: own + nearest parent + series + source-work articles"
                if args.exploratory
                else "pre-specified: own article + nearest parent article"
            ),
            "outcome": "log1p(Score-8 + Score-9 + Score-10), MAL snapshot of early 2021",
            "pre_air_window_days": list(PRE_WINDOW),
            "baseline_window_days": list(BASE_WINDOW),
            "train": "premieres 2016-01-01 to 2018-12-31",
            "test": "premieres 2019-01-01 to 2020-12-31",
            "primary_sample": "tv_mapped",
            "primary_model": "ridge",
            "primary_metric": PRIMARY,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "seed": SEED,
        },
        "coverage": {name: json.loads(table.round(4).to_json(orient="index")) for name, table in tables.items()},
        "results": {},
    }

    for name, (sample, _) in list(samples.items()):
        missing = int((~sample["pv_complete"]).sum())
        if missing and name == "tv_mapped":
            raise SystemExit(f"{missing} of {len(sample)} primary-sample titles still have unfetched articles; run fetch_pageviews.py")
        if missing:
            report["results"][name] = {"skipped": f"{missing} of {len(sample)} titles still have unfetched articles"}
            del samples[name]

    for name, (sample, with_type) in samples.items():
        report["results"][name] = {}
        for model_name, make_model in (("ridge", ridge), ("boosted", boosted)):
            result = run(sample, with_type=with_type, make_model=make_model, test_from=TEST_START)
            predictions = result.pop("_predictions")
            if name == "tv_mapped" and model_name == "ridge":
                predictions.round(4).to_csv(RESULTS / f"{tag}test_predictions_tv_mapped.csv", index=False)
            report["results"][name][model_name] = result

    # Robustness: one test year at a time, training on everything before it.
    report["by_test_year"] = {}
    for year in (2018, 2019, 2020):
        result = run(
            samples["tv_mapped"][0],
            with_type=False,
            make_model=ridge,
            test_from=pd.Timestamp(year=year, month=1, day=1),
            test_to=pd.Timestamp(year=year + 1, month=1, day=1),
        )
        result.pop("_predictions")
        report["by_test_year"][str(year)] = {
            "n_train": result["n_train"],
            "n_test": result["n_test"],
            "meta": result["arms"]["meta"][PRIMARY],
            "meta+alt": result["arms"]["meta+alt"][PRIMARY],
            "delta": result["delta_vs_meta"]["meta+alt"][PRIMARY],
        }

    # Univariate relationship on the primary sample, train and test separately.
    primary_sample = samples["tv_mapped"][0]
    report["univariate_spearman_with_outcome"] = {}
    for part, rows in (("train", primary_sample[primary_sample["start"] < TEST_START]), ("test", primary_sample[primary_sample["start"] >= TEST_START])):
        # The two priors are missing for titles with no earlier franchise or studio entry.
        report["univariate_spearman_with_outcome"][part] = {
            column: spearman(*rows[[column, "y"]].dropna().to_numpy(float).T) for column in [*ALT, "franchise_prior", "studio_prior"]
        }

    # Measurement check: popular titles that have an article yet show no pre-air views.
    # A real hit with zero recorded attention points at the mapping, not at the audience.
    ranked = tv.assign(fifth=pd.qcut(tv["n_pos"].rank(method="first"), 5, labels=False))
    top = ranked[ranked["fifth"] == 4]
    silent = top[top["mapped_any"] & (top["preair_views_any"] == 0)].sort_values("n_pos", ascending=False)
    report["diagnostics"] = {
        "tv_top_fifth": int(len(top)),
        "tv_top_fifth_mapped": int(top["mapped_any"].sum()),
        "tv_top_fifth_mapped_but_zero_preair_views": int(len(silent)),
        "largest_silent_titles": silent["Name"].head(8).tolist(),
    }

    delta, low, high = report["results"]["tv_mapped"]["ridge"]["delta_vs_meta"]["meta+alt"][PRIMARY]
    meets_rule = delta > 0 and low > 0
    report["decision"] = {
        "rule": "PASS if delta > 0 and the 95% bootstrap lower bound > 0",
        "delta": delta,
        "ci95": [low, high],
        # A post hoc run cannot pass or fail a test whose result was already known.
        "verdict": ("meets the rule (exploratory only)" if meets_rule else "does not meet the rule (exploratory only)")
        if args.exploratory
        else ("PASS" if meets_rule else "FAIL"),
    }
    (RESULTS / f"{tag}feasibility_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report["decision"], indent=1))
    for name in samples:
        for model_name in ("ridge", "boosted"):
            block = report["results"][name][model_name]
            print(f"\n{name} / {model_name}  (train {block['n_train']}, test {block['n_test']})")
            print(pd.DataFrame(block["arms"]).T.round(4).to_string())
    print("\nby test year:", json.dumps(report["by_test_year"], indent=1))


if __name__ == "__main__":
    main()
