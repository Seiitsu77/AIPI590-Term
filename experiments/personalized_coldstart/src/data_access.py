"""The only door to the ratings file once the split is frozen.

Every script after make_split_manifest.py gets interactions through the functions here
and never opens the ratings file itself (check_boundaries.py enforces that). Each
function returns one kind of data and nothing wider:

  warm_positives()        positives on warm titles: trains the ALS and builds user vectors
  warm_history_users(g)   users of a group with enough warm positives; uses no cold label
  users_of(role)          the manifest's user list for a cold role
  labels(role)            ratings of a role's users on that role's titles

`labels` refuses the two test roles unless the caller is the final test and the model
selection has been committed. That is what keeps test labels out of development: there
is no other way to obtain them.
"""

from __future__ import annotations

import json
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pv

from common import ARTIFACTS, MANIFEST, MIN_WARM_POSITIVES, POSITIVE, RATINGS, RATINGS_SHA256, ROOT, USER_GROUPS, sha256_file, user_group

DEVELOPMENT_ROLES = ("cold_train", "cold_val")
TEST_ROLES = ("cold_test", "cold_test_late")
MODEL_SELECTION = ROOT / "MODEL_SELECTION.json"
FINAL_TEST_LOCK = ARTIFACTS / "FINAL_TEST_DONE.lock"


@lru_cache(maxsize=1)
def manifest() -> dict:
    return json.loads((MANIFEST / "split_manifest.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def item_roles() -> pd.DataFrame:
    path = MANIFEST / "item_roles.csv"
    if sha256_file(path) != manifest()["files"]["item_roles.csv"]:
        raise RuntimeError("item_roles.csv does not match the digest in the manifest")
    return pd.read_csv(path, parse_dates=["premiere"])


@lru_cache(maxsize=1)
def _ratings() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """user ids, anime ids, ratings, and each row's role code (-1 outside every role)."""
    if sha256_file(RATINGS) != RATINGS_SHA256:
        raise RuntimeError("the ratings file does not match its pinned SHA-256")
    types = {"user_id": pa.int32(), "anime_id": pa.int32(), "rating": pa.int8()}
    table = pv.read_csv(RATINGS, convert_options=pv.ConvertOptions(column_types=types))
    user, anime, rating = (table[name].to_numpy() for name in ("user_id", "anime_id", "rating"))
    roles = item_roles()
    ids = roles["anime_id"].to_numpy()
    codes = pd.Categorical(roles["role"], categories=_role_names()).codes.astype(np.int8)
    order = np.argsort(ids)
    position = np.searchsorted(ids[order], anime)
    position[position == len(ids)] = 0
    found = ids[order][position] == anime
    row_role = np.where(found, codes[order][position], -1).astype(np.int8)
    return user, anime, rating, row_role


def _role_names() -> list[str]:
    return ["warm", "cold_train", "cold_val", "cold_test", "cold_test_late", "excluded_recent", "excluded_undated"]


def _rows_of(role: str) -> np.ndarray:
    return _ratings()[3] == _role_names().index(role)


def warm_positives() -> tuple[np.ndarray, np.ndarray]:
    """(user_id, anime_id) for every rating of 8 or more on a warm title."""
    user, anime, rating, _ = _ratings()
    keep = _rows_of("warm") & (rating >= POSITIVE)
    return user[keep], anime[keep]


def warm_history_users(group: str) -> np.ndarray:
    """Users of one group with at least MIN_WARM_POSITIVES warm positives.

    Membership depends only on warm ratings and the user id. No cold label is read.
    """
    users, counts = np.unique(warm_positives()[0], return_counts=True)
    keep = (counts >= MIN_WARM_POSITIVES) & (user_group(users) == USER_GROUPS.index(group))
    return users[keep]


def users_of(role: str) -> np.ndarray:
    """The manifest's user list for a cold role, checked against its recorded digest."""
    path = MANIFEST / f"users_{role}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path.name} is not on disk (user lists are not committed); rerun make_split_manifest.py --verify to rebuild it")
    if sha256_file(path) != manifest()["files"][path.name]:
        raise RuntimeError(f"{path.name} does not match the digest in the manifest")
    return pd.read_csv(path)["user_id"].to_numpy()


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=False)


def final_test_permitted() -> tuple[bool, str]:
    """Whether test labels may be read, and the reason when they may not."""
    if FINAL_TEST_LOCK.exists():
        return False, "the final test has already been run (lock file present)"
    if not MODEL_SELECTION.exists():
        return False, "MODEL_SELECTION.json does not exist"
    name = MODEL_SELECTION.name
    if _git("ls-files", "--error-unmatch", name).returncode != 0:
        return False, "MODEL_SELECTION.json is not committed"
    if _git("status", "--porcelain", "--", name).stdout.strip():
        return False, "MODEL_SELECTION.json has uncommitted changes"
    return True, "model selection is committed and the final test has not been run"


def labels(role: str, *, users: np.ndarray | None = None, final_test: bool = False) -> pd.DataFrame:
    """Ratings (user_id, anime_id, rating) of `users` on the titles of one cold role.

    `users` defaults to the manifest's list for the role. The test roles are released
    only to the final test, and only once the model selection is committed.
    """
    if role in TEST_ROLES:
        permitted, reason = final_test_permitted()
        if not (final_test and permitted):
            raise PermissionError(f"labels of {role} are withheld: {reason if final_test else 'only final_test.py may read them'}")
    elif role not in DEVELOPMENT_ROLES:
        raise ValueError(f"{role} is not a cold role")
    user, anime, rating, _ = _ratings()
    keep = _rows_of(role) & np.isin(user, users_of(role) if users is None else users)
    return pd.DataFrame({"user_id": user[keep], "anime_id": anime[keep], "rating": rating[keep]})


def repository_commit() -> dict:
    """The commit the code is running from, and whether the experiment folder is clean."""
    head = _git("rev-parse", "HEAD").stdout.strip()
    dirty = _git("status", "--porcelain", "--", ".").stdout.strip()
    return {"commit": head, "experiment_folder_clean": not dirty, "uncommitted": dirty.splitlines()}


def artifact_digest(path: Path) -> str:
    return sha256_file(path)
