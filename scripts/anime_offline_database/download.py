"""Download one release of manami-project/anime-offline-database.

The database merges ~40k anime entries from MyAnimeList, AniList, Kitsu, AniDB,
Anime News Network, LiveChart and others into one JSON file, with the source URLs
of every site per entry. We use it as the cross-site ID map and as an offline
metadata fallback that does not depend on any API being up.

The file lands in data/anime_offline_database/raw/<release tag>/ together with
release.json (tag, URL, size, SHA-256). Rerunning with the same tag is a no-op.

License: Open Database License 1.0 (ODbL). Attribute "anime-offline-database by
manami-project" and share derived databases under ODbL.

Usage:
    python scripts/anime_offline_database/download.py              # latest release
    python scripts/anime_offline_database/download.py --tag 2026-27
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PROJECT_ROOT / "data" / "anime_offline_database"
RELEASES_API = "https://api.github.com/repos/manami-project/anime-offline-database/releases"
ASSET_NAME = "anime-offline-database-minified.json"
USER_AGENT = "AnimeRec/0.1 (Duke AIPI 590 course project)"


def get_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def download(url: str, target: Path) -> str:
    """Stream url to target and return its SHA-256."""
    digest = hashlib.sha256()
    tmp = target.with_suffix(".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as file:
        for chunk in iter(lambda: response.read(1024 * 1024), b""):
            digest.update(chunk)
            file.write(chunk)
    tmp.replace(target)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a release of anime-offline-database.")
    parser.add_argument("--tag", help="Release tag such as 2026-27 (default: latest release).")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    release = get_json(f"{RELEASES_API}/tags/{args.tag}" if args.tag else f"{RELEASES_API}/latest")
    tag = release["tag_name"]
    asset = next((a for a in release.get("assets", []) if a["name"] == ASSET_NAME), None)
    if asset is None:
        print(f"Release {tag} has no {ASSET_NAME}", file=sys.stderr)
        return 1

    release_dir = args.out / "raw" / tag
    target = release_dir / ASSET_NAME
    if target.exists() and target.stat().st_size == asset["size"]:
        print(f"Release {tag} is already downloaded: {target}")
        return 0

    release_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading release {tag} ({asset['size'] / 1e6:.1f} MB) ...")
    sha256 = download(asset["browser_download_url"], target)
    if target.stat().st_size != asset["size"]:
        print(f"Size mismatch: expected {asset['size']}, got {target.stat().st_size}", file=sys.stderr)
        return 1

    metadata = {
        "tag": tag,
        "published_at": release.get("published_at"),
        "asset": ASSET_NAME,
        "url": asset["browser_download_url"],
        "size": asset["size"],
        "sha256": sha256,
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "license": "ODbL-1.0 (anime-offline-database by manami-project)",
    }
    (release_dir / "release.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {target}\nsha256 {sha256}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
