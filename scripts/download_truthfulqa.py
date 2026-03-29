"""
Download TruthfulQA Dataset
============================
Fetches the TruthfulQA benchmark CSV from the official GitHub repository and
saves it to data/TruthfulQA.csv under the project root.

Usage:
    python scripts/download_truthfulqa.py [--force]

Options:
    --force   Re-download even if the file already exists.

Dataset:
    Lin et al. (2022) "TruthfulQA: Measuring How Models Mimic Human Falsehoods"
    https://github.com/sylinrl/TruthfulQA
    https://arxiv.org/abs/2109.07958
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

CSV_URL = "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv"
DEFAULT_DEST = Path(__file__).parent.parent / "data" / "TruthfulQA.csv"


def download(dest: Path = DEFAULT_DEST, *, force: bool = False) -> Path:
    """Download TruthfulQA.csv to *dest*.

    Args:
        dest:  Destination file path.
        force: If True, overwrite an existing file.

    Returns:
        The path where the file was saved.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        print(f"Already cached: {dest}  (pass --force to re-download)")
        return dest

    print(f"Downloading TruthfulQA dataset from:\n  {CSV_URL}")
    print(f"Saving to: {dest}")

    try:
        urllib.request.urlretrieve(CSV_URL, str(dest))
    except Exception as exc:  # pylint: disable=broad-except
        print(f"ERROR: Download failed: {exc}", file=sys.stderr)
        sys.exit(1)

    size_kb = dest.stat().st_size / 1024
    print(f"Done. ({size_kb:.1f} KB)")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download TruthfulQA benchmark dataset.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the file already exists.",
    )
    parser.add_argument(
        "--dest",
        default=str(DEFAULT_DEST),
        help=f"Destination path (default: {DEFAULT_DEST})",
    )
    args = parser.parse_args()
    download(Path(args.dest), force=args.force)


if __name__ == "__main__":
    main()
