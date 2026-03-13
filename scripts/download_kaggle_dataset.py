from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a Kaggle dataset into data/raw using the Kaggle CLI."
    )
    parser.add_argument(
        "--dataset",
        default="mlopssss/imdb-movie-reviews-grouped-by-ratings",
        help="Kaggle dataset slug (owner/dataset-name).",
    )
    parser.add_argument(
        "--out",
        default="data/raw/kaggle",
        help="Output directory where the dataset will be downloaded/unzipped.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out) / args.dataset.replace("/", "__")
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        args.dataset,
        "-p",
        str(out_dir),
        "--unzip",
    ]
    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as e:
        raise SystemExit(
            "Kaggle CLI not found. Install dependencies and ensure `kaggle` is on PATH:\n"
            "  pip install -r requirements.txt\n\n"
            "Also configure Kaggle credentials (kaggle.json) at ~/.kaggle/kaggle.json.\n"
            "See: https://www.kaggle.com/docs/api"
        ) from e

    print(f"Downloaded and unzipped to: {out_dir}")


if __name__ == "__main__":
    main()

