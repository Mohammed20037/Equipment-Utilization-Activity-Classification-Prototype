#!/usr/bin/env python3
"""Download open-source sample video assets for local demos.

This script is intentionally simple and uses direct URLs listed in
`data/metadata/open_source_video_sources.csv`.
"""

from __future__ import annotations

import csv
import os
import pathlib
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "metadata" / "open_source_video_sources.csv"
OUT_DIR = ROOT / "data" / "raw_videos"


def download(url: str, out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response, out_path.open("wb") as fout:
        fout.write(response.read())


def main() -> int:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing source manifest: {CSV_PATH}")

    with CSV_PATH.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("No sources listed.")
        return 0

    for row in rows:
        name = row["name"].strip()
        url = row["source_url"].strip()
        ext = pathlib.Path(url).suffix or ".mp4"
        out_path = OUT_DIR / f"{name}{ext}"
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"skip existing: {out_path}")
            continue

        print(f"downloading {name} from {url}")
        try:
            download(url, out_path)
            print(f"saved: {out_path}")
        except Exception as exc:
            print(f"failed {name}: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
