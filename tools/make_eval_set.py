#!/usr/bin/env python3
"""
Cut a competition/randori video into labeled eval clips from a timestamp file.

The eval set lives in eval_set/<technique>/ and is NEVER used for training —
it exists to measure real-world accuracy on footage that matters (randori,
single camera, your dojo/competitions).

Timestamp file format — one throw per line, time then technique slug:

    1:23  uchi-mata
    2:05  o-soto-gari
    134   tai-otoshi        # plain seconds also fine
    3:41  ?                 # visible throw, technique unclear -> skipped

Usage:
    python tools/make_eval_set.py competition1.mp4 competition1_throws.txt
    python -m pipeline.train_classifier --eval-dir eval_set   # after training
"""

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.clip_extractor import cut_clip, probe_duration  # noqa: E402

EVAL_DIR = REPO_ROOT / "eval_set"
PRE_S, POST_S = 4.0, 3.0


def parse_time(s: str) -> float:
    if ":" in s:
        parts = [float(p) for p in s.split(":")]
        t = 0.0
        for p in parts:
            t = t * 60 + p
        return t
    return float(s)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path)
    p.add_argument("timestamps", type=Path,
                   help="text file: '<m:ss or seconds> <technique-slug>' per line")
    p.add_argument("--pre", type=float, default=PRE_S)
    p.add_argument("--post", type=float, default=POST_S)
    args = p.parse_args()

    duration = probe_duration(args.video)
    n = 0
    for raw in args.timestamps.read_text().splitlines():
        line = re.sub(r"#.*", "", raw).strip()
        if not line:
            continue
        try:
            t_str, tech = line.split(None, 1)
            tech = tech.strip().lower()
            t = parse_time(t_str)
        except ValueError:
            print(f"  can't parse line, skipping: {raw!r}")
            continue
        if tech in ("?", "unknown"):
            print(f"  {t_str}: unlabeled, skipping")
            continue

        out_dir = EVAL_DIR / tech
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{args.video.stem}_{int(t):04d}s.mp4"
        cut_clip(args.video, max(0.0, t - args.pre),
                 min(duration, t + args.post), out, scale_height=1080)
        print(f"  {t_str} {tech} -> {out.relative_to(REPO_ROOT)}")
        n += 1

    print(f"\n{n} eval clips cut. Next:")
    print("  python -m pipeline.pose_features eval_set/")
    print("  python -m pipeline.train_classifier --eval-dir eval_set")


if __name__ == "__main__":
    main()
