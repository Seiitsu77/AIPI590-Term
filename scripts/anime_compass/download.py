"""Download Anime Compass's trained ALS model and catalog from its public Hugging Face dataset.

Anime Compass (https://github.com/Seiitsu77/anime-compass) is the recommender this
project extends. Its production ALS artifact stores one factor vector per MAL
anime ID, trained on MAL ratings up to 2020. It has no user factors: a user vector
is folded in at request time from the titles the user liked,

    x_u = (YᵀY + alpha · Y_uᵀY_u + reg · I)⁻¹ (alpha · Σ_{i∈u} y_i)

with alpha and reg stored in the artifact metadata (see
anime-compass/backend/anime_agent/als_serving.py). That is what lets users the
model never saw, such as AniList users, be scored against the same item space.

Files land in data/anime_compass/ and are checked against the size and SHA-256
pinned in anime-compass/data/artifacts.manifest.json. Rerunning is a no-op once
both files verify.

Usage:
    python scripts/anime_compass/download.py
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PROJECT_ROOT / "data" / "anime_compass"
REPO_ID = "Seiitsu/anime-compass-data"
USER_AGENT = "AnimeRec/0.1 (Duke AIPI 590 course project)"

# Copied from anime-compass/data/artifacts.manifest.json.
ARTIFACTS = {
    "als_production_item_factors.npz": {
        "size": 7393985,
        "sha256": "95c079b1b8f4e0e509c8bab29e4357360f851e3adfd2abc261f358375ee13a10",
    },
    "anime_catalog_serving.json": {
        "size": 6633570,
        "sha256": "4b11cf3e5f3d015ed071232e569068248af709e7979e561194b451e3ac716bf3",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verifies(path: Path, expected: dict) -> bool:
    return path.is_file() and path.stat().st_size == expected["size"] and sha256_file(path) == expected["sha256"]


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as file:
        for chunk in iter(lambda: response.read(1024 * 1024), b""):
            file.write(chunk)
    tmp.replace(target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Anime Compass's ALS model and serving catalog.")
    parser.add_argument("--revision", default="main", help="Hugging Face dataset revision (default: main).")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory (default: data/anime_compass).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for filename, expected in ARTIFACTS.items():
        target = args.out / filename
        if verifies(target, expected):
            print(f"verified {target}")
            continue
        url = (
            f"https://huggingface.co/datasets/{REPO_ID}/resolve/"
            f"{urllib.parse.quote(args.revision, safe='')}/{urllib.parse.quote(filename)}?download=true"
        )
        print(f"downloading {filename} ({expected['size'] / 1e6:.1f} MB) from {REPO_ID}@{args.revision}")
        download(url, target)
        if not verifies(target, expected):
            target.unlink(missing_ok=True)
            print(f"{filename}: size or SHA-256 does not match the pinned manifest; file removed", file=sys.stderr)
            return 1
        print(f"verified {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
