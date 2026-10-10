"""Download AniList metadata for the old anime in Anime Compass's ALS model.

New titles come from AniList (download.py); old titles only have MAL-side metadata
in a different vocabulary (MAL genres/themes, lowercase studio names, no staff or
source). Fetching the old titles from AniList as well puts both sides in one
vocabulary, which is what "find old anime similar to this new one" needs.

The MAL IDs are the ALS items with a learned (non-zero) factor vector, read from
data/anime_compass/als_production_item_factors.npz. They are looked up 50 at a
time with idMal_in, using the same fields as the season listings. MAL IDs with no
AniList entry simply come back missing.

Responses are saved under data/ani_list/raw/catalog/; finished batches are
skipped on a rerun. Run scripts/ani_list/build_tables.py afterwards to get
catalog.csv.

Run without python -I: it imports the client from download.py next to it.

Usage:
    python scripts/ani_list/download_catalog.py
"""

from __future__ import annotations

import json
import sys

import numpy as np

from download import DEFAULT_OUT, MEDIA_FIELDS, PROJECT_ROOT, AniListClient, AniListUnavailable, save_record

ALS_ARTIFACT = PROJECT_ROOT / "data" / "anime_compass" / "als_production_item_factors.npz"
BATCH = 50
CATALOG_QUERY = (
    """
query ($ids: [Int]) {
  Page(page: 1, perPage: 50) {
    media(idMal_in: $ids, type: ANIME) { %s }
  }
}
"""
    % MEDIA_FIELDS
)


def learned_mal_ids() -> list[int]:
    with np.load(ALS_ARTIFACT, allow_pickle=False) as artifact:
        ids = artifact["anime_ids"].astype(int)
        norms = np.linalg.norm(artifact["item_factors"], axis=1)
    return sorted(int(mal_id) for mal_id, norm in zip(ids, norms) if norm >= 1e-3)


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    raw_dir = DEFAULT_OUT / "raw" / "catalog"
    mal_ids = learned_mal_ids()
    batches = [mal_ids[start : start + BATCH] for start in range(0, len(mal_ids), BATCH)]
    todo = [(index, batch) for index, batch in enumerate(batches) if not (raw_dir / f"batch-{index:04d}.json").exists()]
    print(f"{len(mal_ids)} learned ALS items, {len(batches)} batches, {len(todo)} to fetch")

    client = AniListClient()
    found = 0
    try:
        for done, (index, batch) in enumerate(todo, start=1):
            variables = {"ids": batch}
            status, body = client.query(CATALOG_QUERY, variables)
            if status != 200:
                raise AniListUnavailable(f"batch {index} returned HTTP {status}: {body}")
            save_record(raw_dir / f"batch-{index:04d}.json", variables, status, body)
            found += len(body["data"]["Page"]["media"])
            if done % 20 == 0 or done == len(todo):
                print(f"[{done}/{len(todo)}] {found} AniList entries so far")
    except AniListUnavailable as exc:
        print(f"\nAniList is unavailable: {exc}\nRerun to resume.", file=sys.stderr)
        return 1

    print(f"Done. {client.requests_made} requests made.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
