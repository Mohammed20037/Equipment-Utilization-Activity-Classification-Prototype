#!/usr/bin/env python3
"""Export a GIF from processed video for submission artifact."""

from __future__ import annotations

import argparse
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/processed_output.mp4")
    parser.add_argument("--output", default="data/processed/demo.gif")
    parser.add_argument("--fps", default="8")
    args = parser.parse_args()

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        args.input,
        "-vf",
        f"fps={args.fps},scale=960:-1:flags=lanczos",
        args.output,
    ]
    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
