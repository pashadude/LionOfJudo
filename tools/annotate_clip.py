#!/usr/bin/env python3
"""
Produce a shareable analysis video from a single clip (no IMU needed):
face blur -> skeleton + measurements overlay -> waza banner with the
classifier's verdict and nearest catalog matches.

    python tools/annotate_clip.py comp1_throw.mp4
    python tools/annotate_clip.py comp1_throw.mp4 --keep-athlete   # click son
    python tools/annotate_clip.py comp1_throw.mp4 --label uchi-mata  # override

Output: <clip>_showcase.mp4 next to the input (or --out).
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline import reference_bank as refbank                     # noqa: E402
from pipeline.face_blur import blur_clip, first_frame_boxes, \
    pick_person_frame                                              # noqa: E402
from pipeline.pose_features import features_from_poses             # noqa: E402
from pipeline.run_session import load_learned_classifier, \
    classify_learned                                               # noqa: E402


def banner(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    h, w = frame.shape[:2]
    pad = int(h * 0.055) * (len(lines) + 1)
    cv2.rectangle(frame, (0, h - pad), (w, h), (20, 20, 20), -1)
    for i, text in enumerate(lines):
        y = h - pad + int(h * 0.055) * (i + 1)
        cv2.putText(frame, text, (int(w * 0.02), y),
                    cv2.FONT_HERSHEY_SIMPLEX, h / 900, (60, 220, 255),
                    max(1, h // 500), cv2.LINE_AA)
    return frame


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("clip", type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--keep-athlete", action="store_true",
                   help="pick one person to keep unblurred (default: blur all)")
    p.add_argument("--label", default=None,
                   help="override the waza name shown in the banner")
    p.add_argument("--device", default="mps")
    args = p.parse_args()

    from ultralytics import YOLO
    from phase0_visual_analysis import VisualJudoAnalyzer

    out = args.out or args.clip.with_name(args.clip.stem + "_showcase.mp4")
    workdir = out.parent
    model = YOLO(str(REPO_ROOT / "yolo11x-pose.pt"))

    # 1. privacy blur
    son_id = None
    if args.keep_athlete:
        frame, boxes = first_frame_boxes(model, args.clip, args.device)
        if boxes:
            son_id = pick_person_frame(frame, boxes)
    blurred = workdir / (args.clip.stem + "_blurtmp.mp4")
    blur_clip(model, args.clip, blurred, son_id, args.device)

    # 2. skeleton + measurements (existing analyzer)
    analyzer = VisualJudoAnalyzer(output_dir=workdir)
    analysis = analyzer.process_video(blurred)
    annotated = workdir / f"{blurred.stem}_cam0_annotated.mp4"

    # 3. classification for the banner
    lines = []
    poses = analysis.get("poses") or []
    pf = features_from_poses(poses) if poses else None
    if args.label:
        lines.append(f"WAZA: {args.label}")
    elif pf is not None:
        bundle = load_learned_classifier()
        if bundle:
            res = classify_learned(bundle, pf["stats"])
            if res:
                lines.append(f"WAZA: {res['technique']}  "
                             f"({res['confidence']:.0%})")
        bank = refbank.load_bank()
        if bank:
            refs = refbank.match(bank, pf["stats"], top_k=3)
            lines.append("nearest: " + "  ".join(
                f"{r['waza']} {r['similarity']:.2f}" for r in refs))
    lines.append("LionOfJudo | pose + power analysis")

    # 4. burn the banner in
    cap = cv2.VideoCapture(str(annotated))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(banner(frame, lines))
    cap.release()
    writer.release()

    # tidy intermediates
    blurred.unlink(missing_ok=True)
    annotated.unlink(missing_ok=True)
    json_stray = workdir / f"{blurred.stem}_analysis.json"
    json_stray.unlink(missing_ok=True)

    print(f"\n✓ showcase video: {out}")
    for line in lines:
        print(f"  banner: {line}")


if __name__ == "__main__":
    main()
