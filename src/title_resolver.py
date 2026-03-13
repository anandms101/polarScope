from __future__ import annotations

import gzip
import json
import shutil
import urllib.request
from pathlib import Path
from typing import Dict, Set

IMDB_BASICS_URL = "https://datasets.imdbws.com/title.basics.tsv.gz"
CACHE_PATH = Path("data/processed/title_cache.json")


def _download_imdb_basics(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading IMDb title.basics.tsv.gz (~150 MB) ...")
    urllib.request.urlretrieve(IMDB_BASICS_URL, str(dest))
    print("Download complete.")


def build_title_map(
    movie_ids: Set[str],
    *,
    cache_path: Path = CACHE_PATH,
    force: bool = False,
) -> Dict[str, str]:
    """
    Maps IMDb title IDs (e.g. 'tt0120616') to human-readable titles.

    Uses a local cache. If the cache doesn't have all IDs, downloads the IMDb
    title.basics.tsv.gz and extracts a mapping for the requested IDs.
    """
    title_map: Dict[str, str] = {}
    if cache_path.exists() and not force:
        title_map = json.loads(cache_path.read_text())
        missing = movie_ids - set(title_map.keys())
        if not missing:
            return {k: title_map[k] for k in movie_ids if k in title_map}
    else:
        missing = movie_ids

    gz_path = Path("data/raw/title.basics.tsv.gz")
    if not gz_path.exists():
        _download_imdb_basics(gz_path)

    print(f"Scanning title.basics.tsv.gz for {len(missing)} movie IDs ...")
    found: Dict[str, str] = {}
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.split("\t", 4)
            if len(parts) < 4:
                continue
            tconst = parts[0]
            if tconst in missing:
                primary_title = parts[2]
                found[tconst] = primary_title
                if len(found) == len(missing):
                    break

    title_map.update(found)
    # For IDs we couldn't resolve, keep the raw ID.
    for mid in movie_ids:
        if mid not in title_map:
            title_map[mid] = mid

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(title_map, indent=2))
    print(f"Title cache updated ({len(title_map)} entries) at {cache_path}")
    return {k: title_map[k] for k in movie_ids if k in title_map}


def load_title_cache(cache_path: Path = CACHE_PATH) -> Dict[str, str]:
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    return {}
