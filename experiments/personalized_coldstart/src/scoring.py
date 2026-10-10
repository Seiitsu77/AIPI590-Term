"""Design matrices, model scores and ranking metrics, shared by development and the final test.

Everything here works on a full grid: a set of users times every title of one role.
Titles are always in ascending MAL id order, and ranking uses a stable sort, so equal
scores resolve to the smaller id first for every model alike.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import ARTIFACTS, NEGATIVE, POSITIVE

CONTENT = ["content_dot", "content_cos", "prequel_dot", "has_warm_prequel", "content_norm", "prequel_log_pos"]
ATTENTION = ["al_pop_pct", "al_trend_pct", "al_growth_pct", "wiki_views_pct", "al_missing", "al_growth_missing", "wiki_missing"]
ALL_FEATURES = CONTENT + ATTENTION
MODELS = {"M0b": ATTENTION, "M1": CONTENT, "M2": ALL_FEATURES}
# Fixed in advance, both reported: popularity as recorded, and its within-season percentile.
BASELINES = {"M0a_raw": "al_pop", "M0a_norm": "al_pop_pct"}
K = 10


class Grid:
    """The titles of one role with their features and vectors, plus user vectors on demand."""

    def __init__(self, role: str):
        items = pd.read_parquet(ARTIFACTS / "item_features.parquet")
        vectors = np.load(ARTIFACTS / "item_vectors.npz")
        keep = (items["role"] == role).to_numpy()
        assert np.array_equal(items["anime_id"].to_numpy(), vectors["anime_ids"]) and items["anime_id"].is_monotonic_increasing
        self.role = role
        self.items = items[keep].reset_index(drop=True)
        self.anime_ids = self.items["anime_id"].to_numpy()
        self.content_vec = vectors["content_vec"][keep].astype(np.float64)
        self.prequel_vec = vectors["prequel_vec"][keep].astype(np.float64)

    @staticmethod
    def user_vectors(user_ids: np.ndarray) -> np.ndarray:
        stored = np.load(ARTIFACTS / "user_vectors.npz")
        ids = stored["user_ids"]
        position = np.minimum(np.searchsorted(ids, user_ids), len(ids) - 1)
        if not np.array_equal(ids[position], user_ids):
            raise KeyError("some users have no fold-in vector")
        return stored["vectors"][position].astype(np.float64)

    def design(self, users: np.ndarray) -> np.ndarray:
        """Feature matrix for every (user, title) pair, user-major, columns as ALL_FEATURES."""
        n, m = len(users), len(self.items)
        dot = users @ self.content_vec.T
        norms = np.linalg.norm(users, axis=1)[:, None] * np.linalg.norm(self.content_vec, axis=1)[None, :]
        columns = {
            "content_dot": dot,
            "content_cos": dot / np.where(norms > 0, norms, 1.0),
            "prequel_dot": users @ self.prequel_vec.T,
        }
        for name in ALL_FEATURES:
            if name not in columns:
                columns[name] = np.broadcast_to(self.items[name].to_numpy(np.float64), (n, m))
        return np.stack([columns[name].reshape(-1) for name in ALL_FEATURES], axis=1).astype(np.float32)

    def baseline(self, name: str, n_users: int) -> np.ndarray:
        """Same ranking for every user. A title without the signal goes last."""
        values = self.items[BASELINES[name]].to_numpy(np.float64) if name == "M0a_raw" else np.where(self.items["al_missing"] == 1, np.nan, self.items["al_pop_pct"])
        return np.broadcast_to(np.where(np.isnan(values), -np.inf, values), (n_users, len(values))).copy()

    def label_matrices(self, labels: pd.DataFrame, user_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(liked, disliked) boolean matrices, users by titles. Unrated pairs are False in both."""
        rows = np.searchsorted(user_ids, labels["user_id"].to_numpy())
        columns = np.searchsorted(self.anime_ids, labels["anime_id"].to_numpy())
        liked = np.zeros((len(user_ids), len(self.anime_ids)), dtype=bool)
        disliked = np.zeros_like(liked)
        rating = labels["rating"].to_numpy()
        liked[rows[rating >= POSITIVE], columns[rating >= POSITIVE]] = True
        disliked[rows[rating <= NEGATIVE], columns[rating <= NEGATIVE]] = True
        return liked, disliked


def model_scores(model: dict, design: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Linear score of a frozen logistic model. Ranking by it equals ranking by probability."""
    columns = [ALL_FEATURES.index(name) for name in model["features"]]
    z = (design[:, columns].astype(np.float64) - np.asarray(model["scaler_mean"])) / np.asarray(model["scaler_scale"])
    return (z @ np.asarray(model["coefficients"]) + model["intercept"]).reshape(shape)


def top_k(scores: np.ndarray, k: int = K) -> np.ndarray:
    return np.argsort(-scores, axis=1, kind="stable")[:, :k]


_DISCOUNT = 1.0 / np.log2(np.arange(2, K + 2))
_IDEAL = np.concatenate([[0.0], np.cumsum(_DISCOUNT)])


def ndcg_and_recall(scores: np.ndarray, liked: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-user NDCG@10 and Recall@10. A user with no liked title scores 0 on both."""
    hits = np.take_along_axis(liked, top_k(scores), axis=1)
    positives = liked.sum(axis=1)
    ideal = _IDEAL[np.minimum(positives, K)]
    dcg = hits @ _DISCOUNT[: hits.shape[1]]
    safe = positives > 0
    ndcg = np.where(safe, dcg / np.where(safe, ideal, 1.0), 0.0)
    recall = np.where(safe, hits.sum(axis=1) / np.where(safe, positives, 1), 0.0)
    return ndcg, recall


def observed_auc(scores: np.ndarray, liked: np.ndarray, disliked: np.ndarray) -> np.ndarray:
    """Per-user AUC of liked over disliked titles, among titles the user rated. NaN without both kinds."""
    out = np.full(len(scores), np.nan)
    for user in np.flatnonzero(liked.any(axis=1) & disliked.any(axis=1)):
        up, down = scores[user, liked[user]], scores[user, disliked[user]]
        out[user] = ((up[:, None] > down[None, :]).sum() + 0.5 * (up[:, None] == down[None, :]).sum()) / (len(up) * len(down))
    return out


def coverage(scores: np.ndarray) -> float:
    """Share of the candidate titles that reach at least one user's top 10."""
    return float(len(np.unique(top_k(scores))) / scores.shape[1])
