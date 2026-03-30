from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

DATASET_SLUG = "mlopssss/imdb-movie-reviews-grouped-by-ratings"
DEFAULT_OUT = "data/raw/kaggle"
KAGGLE_URL = f"https://www.kaggle.com/datasets/{DATASET_SLUG}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the IMDb per-movie reviews dataset from Kaggle.",
    )
    parser.add_argument("--dataset", default=DATASET_SLUG, help="Kaggle dataset slug.")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output base directory.")
    args = parser.parse_args()

    out_dir = Path(args.out) / args.dataset.replace("/", "__")
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["kaggle", "datasets", "download", "-d", args.dataset, "-p", str(out_dir), "--unzip"]
    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print(
            "\n--- Kaggle CLI not found ---\n"
            "You can install it with:  pip install kaggle\n"
            "Then configure credentials: https://www.kaggle.com/docs/api\n\n"
            "--- Manual alternative (no CLI needed) ---\n"
            f"1. Open {KAGGLE_URL}\n"
            "2. Click 'Download' (free Kaggle account required).\n"
            f"3. Unzip the downloaded zip so CSVs are in:\n"
            f"   {out_dir}/\n"
        )
        raise SystemExit(1)

    print(f"\nDone! Dataset extracted to: {out_dir}")


if __name__ == "__main__":
    main()
