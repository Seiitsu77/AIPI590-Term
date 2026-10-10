"""Build every feature the models use. No cold label is read here.

Three things are produced under artifacts/:

  item_features.parquet  one row per cold title: attention features (raw, standardised
                         within the premiere season, with missingness flags) and the
                         item-level content features
  item_vectors.npz       per cold title, the content-predicted ALS vector and the mean
                         ALS vector of its warm prequels
  user_vectors.npz       fold-in vectors of every user with at least five warm positives

Cold titles of all four roles are processed together, test roles included. That is safe
because nothing in this file can obtain a label: it does not import `labels`, and the
only interactions it reads are positives on warm titles.

Time discipline. AniList rows are used up to Japan day -2, Wikipedia days up to UTC
day -2. Each window is asserted, and the latest day actually read is written to
feature_checks.json so the claim can be checked rather than believed.

Usage:
    python src/build_features.py
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from common import (
    ANILIST_TRENDS,
    ARTIFACTS,
    ATTENTION_WINDOW,
    CUTOFF_DAY,
    GROWTH_REFERENCE_WINDOW,
    MAL_CATALOG,
    MANIFEST,
    MIN_WARM_POSITIVES,
    ROLES,
    WIKI_INDEX,
    WIKI_VIEWS,
    WIKI_WINDOW,
    approx_date,
    franchise_root,
    root_keys,
    sha256_file,
    splitmix64,
)
from data_access import item_roles, manifest, repository_commit, warm_positives

MAPPER_MIN_POSITIVES = 50
MAPPER_MIN_STUDIO_TITLES = 5
MAPPER_PENALTIES = (1.0, 10.0, 100.0, 1000.0)
MAPPER_FOLDS = 5
ALPHA, REGULARIZATION = 5.0, 0.05  # fold-in constants, as in Anime Compass serving
RAW_ATTENTION = ("al_pop", "al_trend", "al_growth", "wiki_views")


# ------------------------------------------------------------------ content


def split_field(value: object) -> list[str]:
    return [] if not isinstance(value, str) or value == "Unknown" else [part.strip() for part in value.split(",") if part.strip()]


class ContentEncoder:
    """Multi-hot encoding of the static MAL fields. The vocabulary comes from the titles it is fitted on."""

    def __init__(self, frame: pd.DataFrame):
        studios = pd.Series([s for value in frame["Studios"] for s in split_field(value)]).value_counts()
        self.genres = sorted({g for value in frame["Genres"] for g in split_field(value)})
        self.studios = sorted(studios[studios >= MAPPER_MIN_STUDIO_TITLES].index)
        self.sources = sorted(set(frame["Source"].dropna()) - {"Unknown"})
        self.ratings = sorted(set(frame["Rating"].dropna()) - {"Unknown"})
        self.types = sorted(set(frame["Type"].dropna()) - {"Unknown"})
        self.columns = (
            ["intercept"]
            + [f"genre={g}" for g in self.genres]
            + [f"studio={s}" for s in self.studios]
            + ["studio=other", "studio=none"]
            + [f"source={s}" for s in self.sources]
            + [f"rating={r}" for r in self.ratings]
            + [f"type={t}" for t in self.types]
        )
        self.position = {name: i for i, name in enumerate(self.columns)}

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = np.zeros((len(frame), len(self.columns)), dtype=np.float64)
        matrix[:, 0] = 1.0
        known = set(self.studios)
        for row, (genres, studios, source, rating, kind) in enumerate(zip(frame["Genres"], frame["Studios"], frame["Source"], frame["Rating"], frame["Type"], strict=True)):
            for genre in split_field(genres):
                if f"genre={genre}" in self.position:
                    matrix[row, self.position[f"genre={genre}"]] = 1.0
            listed = split_field(studios)
            for studio in listed:
                matrix[row, self.position[f"studio={studio}" if studio in known else "studio=other"]] = 1.0
            if not listed:
                matrix[row, self.position["studio=none"]] = 1.0
            for prefix, value in (("source", source), ("rating", rating), ("type", kind)):
                if f"{prefix}={value}" in self.position:
                    matrix[row, self.position[f"{prefix}={value}"]] = 1.0
        return matrix


def ridge(x: np.ndarray, y: np.ndarray, penalty: float) -> np.ndarray:
    """Closed-form multi-output ridge. The intercept (column 0) is not penalised."""
    gram = x.T @ x
    diagonal = np.full(x.shape[1], penalty)
    diagonal[0] = 0.0
    return np.linalg.solve(gram + np.diag(diagonal), x.T @ y)


def mean_cosine(a: np.ndarray, b: np.ndarray) -> float:
    norm = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return float(np.mean((a * b).sum(axis=1) / np.where(norm > 0, norm, 1.0)))


# ---------------------------------------------------------------- attention


def within_season_percentile(values: pd.Series, season: pd.Series) -> pd.Series:
    """Percentile rank among the season's titles that have the value; missing stays missing.

    Uses features only. A season belongs to one role, so no role sees another's scale.
    """
    return values.groupby(season).transform(lambda s: s.rank(method="average") / s.notna().sum())


def anilist_features(days: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    trends = pd.read_parquet(ANILIST_TRENDS).drop_duplicates(["anime_id", "timestamp"])
    trends["day"] = pd.to_datetime(trends["day_jst"])
    trends = trends.merge(days[["anime_id", "day_zero"]], on="anime_id")
    trends["rel_day"] = (trends["day"] - trends["day_zero"]).dt.days
    usable = trends[trends["rel_day"] <= CUTOFF_DAY]
    rows, latest_used = [], -10**9
    for anime_id, group in usable.groupby("anime_id"):
        window = group[group["rel_day"].between(*ATTENTION_WINDOW)].sort_values("rel_day")
        reference = group[group["rel_day"].between(*GROWTH_REFERENCE_WINDOW) & group["popularity"].notna()].sort_values("rel_day")
        with_pop = window[window["popularity"].notna()]
        row = {"anime_id": anime_id, "al_pop": np.nan, "al_trend": np.nan, "al_growth": np.nan}
        if len(window):
            latest_used = max(latest_used, int(window["rel_day"].max()))
            row["al_trend"] = float(window["trending"].mean())
        if len(with_pop):
            row["al_pop"] = float(with_pop["popularity"].iloc[-1])
            if len(reference) and row["al_pop"] > 0 and reference["popularity"].iloc[-1] > 0:
                row["al_growth"] = float(np.log(row["al_pop"]) - np.log(reference["popularity"].iloc[-1]))
        rows.append(row)
    assert latest_used <= CUTOFF_DAY, "an AniList row later than the cutoff was read"
    return pd.DataFrame(rows), latest_used


def wikipedia_features(days: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    audit = pd.read_csv(MANIFEST / "wikipedia_article_audit.csv")
    kept = audit[~audit["excluded"]]
    index = pd.read_csv(WIKI_INDEX)
    row_of = {(lang, title): row for row, lang, title in zip(index["row"], index["lang"], index["title"], strict=True)}
    payload = np.load(WIKI_VIEWS)
    views, first_day = payload["views"], pd.Timestamp(str(payload["first_day"]))
    day_zero = days.set_index("anime_id")["day_zero"]
    assert WIKI_WINDOW[1] <= CUTOFF_DAY, "the Wikipedia window ends after the cutoff"
    rows = []
    for anime_id, group in kept.groupby("anime_id"):
        offset = (day_zero[anime_id] - first_day).days
        article_rows = sorted({row_of[(lang, title)] for lang, title in zip(group["lang"], group["title"], strict=True)})
        total = float(views[article_rows, offset + WIKI_WINDOW[0] : offset + WIKI_WINDOW[1] + 1].sum())
        rows.append({"anime_id": anime_id, "wiki_views": total if total > 0 else np.nan})
    return pd.DataFrame(rows), WIKI_WINDOW[1]


# -------------------------------------------------------------------- users


def fold_in(item_factors: np.ndarray, liked_rows: list[np.ndarray]) -> np.ndarray:
    """x_u = (YtY + alpha Yu^T Yu + reg I)^-1 (alpha sum y_i), one user at a time."""
    gram = (item_factors.T @ item_factors).astype(np.float64) + REGULARIZATION * np.eye(item_factors.shape[1])
    out = np.zeros((len(liked_rows), item_factors.shape[1]), dtype=np.float32)
    for position, rows in enumerate(liked_rows):
        liked = item_factors[rows].astype(np.float64)
        out[position] = np.linalg.solve(gram + ALPHA * (liked.T @ liked), ALPHA * liked.sum(axis=0))
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if manifest()["status"] != "frozen":
        raise SystemExit("the split manifest is not frozen")
    roles = item_roles()
    days = pd.read_csv(ARTIFACTS / "day_zero.csv", parse_dates=["mal_premiere", "day_zero"])
    als_path = ARTIFACTS / "als_experiment.npz"
    als = np.load(als_path)
    als_ids, item_factors = als["anime_ids"], als["item_factors"]
    als_row = {int(anime_id): row for row, anime_id in enumerate(als_ids)}
    checks: dict = {"code": repository_commit(), "als_sha256": sha256_file(als_path)}

    mal = pd.read_csv(MAL_CATALOG, usecols=["MAL_ID", "Name", "Type", "Aired", "Genres", "Studios", "Source", "Rating"]).rename(columns={"MAL_ID": "anime_id"})
    catalog = roles[["anime_id", "role", "positives"]].merge(mal, on="anime_id", how="left")

    # ------------------------------------------------- content mapper (warm only)
    train = catalog[(catalog["role"] == "warm") & (catalog["positives"] >= MAPPER_MIN_POSITIVES) & catalog["anime_id"].isin(als_row)].reset_index(drop=True)
    assert set(train["role"]) == {"warm"}, "a non-warm title reached the content mapper"
    encoder = ContentEncoder(train)
    x = encoder.transform(train)
    y = item_factors[[als_row[int(a)] for a in train["anime_id"]]].astype(np.float64)
    fold = (splitmix64(train["anime_id"].to_numpy(), 7) % np.uint64(MAPPER_FOLDS)).astype(int)
    cv = {}
    for penalty in MAPPER_PENALTIES:
        predicted = np.zeros_like(y)
        for k in range(MAPPER_FOLDS):
            predicted[fold == k] = x[fold == k] @ ridge(x[fold != k], y[fold != k], penalty)
        cv[penalty] = mean_cosine(predicted, y)
    # Highest held-out cosine; a tie goes to the heavier penalty.
    penalty = max(MAPPER_PENALTIES, key=lambda p: (round(cv[p], 4), p))
    weights = ridge(x, y, penalty)
    checks["content_mapper"] = {
        "trained_on_roles": sorted(set(train["role"])),
        "training_titles": int(len(train)),
        "features": int(x.shape[1]),
        "vocabulary": {"genres": len(encoder.genres), "studios": len(encoder.studios), "sources": len(encoder.sources), "ratings": len(encoder.ratings), "types": len(encoder.types)},
        "held_out_mean_cosine_by_penalty": {str(p): round(v, 5) for p, v in cv.items()},
        "penalty": penalty,
    }

    cold = catalog[catalog["role"].isin(ROLES)].merge(days[["anime_id", "mal_premiere", "day_zero", "season"]], on="anime_id").sort_values("anime_id").reset_index(drop=True)
    content_vec = (encoder.transform(cold) @ weights).astype(np.float32)

    # ------------------------------------------------------------ warm prequels
    mal["when"] = mal["Aired"].map(approx_date)
    mal["root"] = mal["Name"].map(franchise_root)
    warm_positive_count = catalog.loc[catalog["role"] == "warm"].set_index("anime_id")["positives"]
    by_root: dict[str, list[tuple[pd.Timestamp, int]]] = {}
    for anime_id, root, when in zip(mal["anime_id"], mal["root"], mal["when"], strict=True):
        if root and not pd.isna(when) and int(anime_id) in als_row and anime_id in warm_positive_count.index:
            by_root.setdefault(root, []).append((when, int(anime_id)))
    prequel_vec = np.zeros_like(content_vec)
    has_prequel, prequel_log_pos = [], []
    for row, (name, when) in enumerate(zip(cold["Name"], cold["mal_premiere"], strict=True)):
        earlier = sorted({other_id for key in root_keys(franchise_root(name)) for other_when, other_id in by_root.get(key, ()) if other_when < when})
        has_prequel.append(float(bool(earlier)))
        prequel_log_pos.append(float(np.log1p(max(warm_positive_count[other] for other in earlier))) if earlier else 0.0)
        if earlier:
            prequel_vec[row] = item_factors[[als_row[other] for other in earlier]].mean(axis=0)

    items = cold[["anime_id", "role", "season", "day_zero"]].copy()
    items["content_norm"] = np.linalg.norm(content_vec, axis=1)
    items["has_warm_prequel"] = has_prequel
    items["prequel_log_pos"] = prequel_log_pos

    # --------------------------------------------------------------- attention
    anilist, latest_anilist = anilist_features(days)
    wikipedia, latest_wikipedia = wikipedia_features(days)
    items = items.merge(anilist, on="anime_id", how="left").merge(wikipedia, on="anime_id", how="left")
    items["al_missing"] = items["al_pop"].isna().astype(float)
    items["al_growth_missing"] = items["al_growth"].isna().astype(float)
    items["wiki_missing"] = items["wiki_views"].isna().astype(float)
    for name in RAW_ATTENTION:
        # Standardised within the season; a missing value sits at the middle and is flagged.
        items[f"{name}_pct"] = within_season_percentile(items[name], items["season"]).fillna(0.5)
    checks["time_discipline"] = {
        "cutoff_day": CUTOFF_DAY,
        "latest_anilist_japan_day_read": latest_anilist,
        "latest_wikipedia_utc_day_read": latest_wikipedia,
        "anilist_window": list(ATTENTION_WINDOW),
        "growth_reference_window": list(GROWTH_REFERENCE_WINDOW),
        "wikipedia_window": list(WIKI_WINDOW),
    }
    checks["attention_by_role"] = {
        role: {"titles": int(len(g)), "anilist_signal": int((g["al_missing"] == 0).sum()), "anilist_growth": int((g["al_growth_missing"] == 0).sum()),
               "wikipedia_signal": int((g["wiki_missing"] == 0).sum()), "no_attention_signal": int(((g["al_missing"] == 1) & (g["wiki_missing"] == 1)).sum()),
               "warm_prequel": int(g["has_warm_prequel"].sum())}
        for role, g in items.groupby("role")
    }  # fmt: skip

    items.to_parquet(ARTIFACTS / "item_features.parquet", index=False)
    np.savez_compressed(ARTIFACTS / "item_vectors.npz", anime_ids=items["anime_id"].to_numpy(np.int64), content_vec=content_vec, prequel_vec=prequel_vec)

    # ------------------------------------------------------------ user vectors
    user, anime = warm_positives()
    keep = np.isin(anime, als_ids)
    user, anime = user[keep], anime[keep]
    order = np.argsort(user, kind="stable")
    user, anime = user[order], anime[order]
    users, starts, counts = np.unique(user, return_index=True, return_counts=True)
    rows_of_item = np.searchsorted(als_ids, anime)
    enough = counts >= MIN_WARM_POSITIVES
    liked_rows = [rows_of_item[start : start + count] for start, count in zip(starts[enough], counts[enough], strict=True)]
    vectors = fold_in(item_factors, liked_rows)
    np.savez_compressed(ARTIFACTS / "user_vectors.npz", user_ids=users[enough].astype(np.int64), vectors=vectors)

    # Fold-in should reproduce the factors ALS itself learned for the same users.
    sample_ids, sample_trained = als["sample_user_ids"], als["sample_user_factors"]
    position = {int(u): i for i, u in enumerate(users[enough])}
    common = [(position[int(u)], i) for i, u in enumerate(sample_ids) if int(u) in position]
    folded = vectors[[a for a, _ in common]]
    trained = sample_trained[[b for _, b in common]]
    checks["user_vectors"] = {
        "built_from": "positives on warm titles only",
        "users": int(enough.sum()),
        "fold_in_vs_trained_mean_cosine": round(mean_cosine(folded.astype(np.float64), trained.astype(np.float64)), 4),
        "users_compared": len(common),
    }
    checks["artifacts"] = {name: sha256_file(ARTIFACTS / name) for name in ("item_features.parquet", "item_vectors.npz", "user_vectors.npz", "day_zero.csv")}
    (ARTIFACTS / "feature_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in checks.items() if k != "code"}, indent=1))


if __name__ == "__main__":
    main()
