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


ACCENT = (60, 220, 255)      # gold-ish (BGR)
BAR_BG = (60, 60, 60)


def draw_panel(frame: np.ndarray, top3: list[tuple[str, float]],
               refs: list[dict] | None, override: str | None) -> np.ndarray:
    """Recognition panel: top-3 waza with probability bars + footer."""
    h, w = frame.shape[:2]
    row = int(h * 0.052)
    n_rows = (1 if override else max(1, len(top3))) + (1 if refs else 0) + 2
    panel_h = row * n_rows + row // 2
    y0 = h - panel_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), (18, 14, 10), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.line(frame, (0, y0), (w, y0), ACCENT, max(1, h // 400))

    fs = h / 1100
    th = max(1, h // 550)
    x_text = int(w * 0.02)
    y = y0 + row

    cv2.putText(frame, "WAZA RECOGNITION", (x_text, y),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (200, 200, 200), th,
                cv2.LINE_AA)
    y += row

    if override:
        cv2.putText(frame, override.upper(), (x_text, y),
                    cv2.FONT_HERSHEY_TRIPLEX, fs * 1.5, ACCENT, th,
                    cv2.LINE_AA)
        y += row
    else:
        bar_x = int(w * 0.42)
        bar_w_max = int(w * 0.42)
        for rank, (waza, p) in enumerate(top3, 1):
            color = ACCENT if rank == 1 else (170, 170, 170)
            cv2.putText(frame, f"{rank}. {waza}", (x_text, y),
                        cv2.FONT_HERSHEY_SIMPLEX, fs * 1.1, color, th,
                        cv2.LINE_AA)
            bh = int(row * 0.45)
            by = y - bh + 2
            cv2.rectangle(frame, (bar_x, by), (bar_x + bar_w_max, by + bh),
                          BAR_BG, -1)
            cv2.rectangle(frame, (bar_x, by),
                          (bar_x + int(bar_w_max * p), by + bh), color, -1)
            cv2.putText(frame, f"{p:.0%}",
                        (bar_x + bar_w_max + int(w * 0.012), y),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, color, th, cv2.LINE_AA)
            y += row

    if refs:
        cv2.putText(frame, "catalog matches: " + "   ".join(
            f"{r['waza']} {r['similarity']:.2f}" for r in refs),
            (x_text, y), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.85,
            (150, 150, 150), max(1, th - 1), cv2.LINE_AA)
        y += row

    cv2.putText(frame, "LionOfJudo | pose + power analysis",
                (x_text, y), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.9,
                (120, 170, 190), max(1, th - 1), cv2.LINE_AA)
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

    # 3. classification: top-3 waza with probabilities
    top3: list[tuple[str, float]] = []
    refs = None
    poses = analysis.get("poses") or []
    pf = features_from_poses(poses) if poses else None
    if pf is not None and not args.label:
        bundle = load_learned_classifier()
        if bundle:
            clf, labels = bundle
            x = np.nan_to_num(pf["stats"], nan=0.0, posinf=0.0,
                              neginf=0.0).reshape(1, -1)
            proba = clf.predict_proba(x)[0]
            order = np.argsort(proba)[::-1][:3]
            top3 = [(labels[i], float(proba[i])) for i in order]
        bank = refbank.load_bank()
        if bank:
            refs = refbank.match(bank, pf["stats"], top_k=3)

    # 4. burn the recognition panel in
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
        writer.write(draw_panel(frame, top3, refs, args.label))
    cap.release()
    writer.release()

    # tidy intermediates
    blurred.unlink(missing_ok=True)
    annotated.unlink(missing_ok=True)
    json_stray = workdir / f"{blurred.stem}_analysis.json"
    json_stray.unlink(missing_ok=True)

    print(f"\n✓ showcase video: {out}")
    if args.label:
        print(f"  waza (override): {args.label}")
    for waza, p in top3:
        print(f"  {p:5.0%}  {waza}")


if __name__ == "__main__":
    main()
