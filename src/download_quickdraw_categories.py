"""Opt-in downloader for selected QuickDraw simplified .ndjson files.

This intentionally downloads only named categories. It does not mirror the full
QuickDraw dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import quote

import requests


BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/{category}.ndjson"


def download_category(category: str, output_dir: Path, overwrite: bool = False) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{category}.ndjson"
    if output_path.exists() and not overwrite:
        print(f"exists: {output_path}")
        return output_path

    url = BASE_URL.format(category=quote(category, safe=""))
    print(f"downloading selected category '{category}' from {url}")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with output_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    print(f"wrote: {output_path}")
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download selected QuickDraw simplified categories.")
    parser.add_argument("--categories", nargs="+", required=True, help="Category names, e.g. cat flower bicycle")
    parser.add_argument("--output-dir", default="data/quickdraw/raw")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    for category in args.categories:
        download_category(category, Path(args.output_dir), overwrite=args.overwrite)


if __name__ == "__main__":
    main()
