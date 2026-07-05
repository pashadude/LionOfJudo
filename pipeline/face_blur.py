#!/usr/bin/env python3
"""
Privacy blur: blur every face except the designated athlete (the son).

Fail-safe direction: blur by default. Only a confirmed, tracked person is
exempt. If his track is lost, his face gets blurred too for those frames
(reported) — never the other way around.

Two detection layers per frame:
1. YOLO11-pose head keypoints (nose/eyes/ears) -> blur ellipse per person.
2. YuNet face detector as a safety net for anyone pose missed
   (partial bodies, spectators, referees).
"""

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
YUNET_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"

HEAD_KPTS = [0, 1, 2, 3, 4]  # nose, eyes, ears (COCO)
KPT_CONF = 0.3
ELLIPSE_SCALE = 1.7
MIN_HEAD_RADIUS = 12  # px; small/distant heads still get a meaningful blur


@dataclass
class BlurReport:
    total_frames: int = 0
    son_visible_frames: int = 0
    son_lost_ranges: list[tuple[int, int]] = field(default_factory=list)


def _head_region(kpts: np.ndarray) -> tuple[int, int, int] | None:
    """(cx, cy, radius) of the head from pose keypoints, or None."""
    pts = [(kpts[i][0], kpts[i][1]) for i in HEAD_KPTS if kpts[i][2] > KPT_CONF]
    if not pts:
        return None
    xs, ys = zip(*pts)
    cx, cy = int(np.mean(xs)), int(np.mean(ys))

    # size from ear-to-ear, else eye-to-eye, else fixed minimum
    size = None
    if kpts[3][2] > KPT_CONF and kpts[4][2] > KPT_CONF:
        size = np.hypot(kpts[3][0] - kpts[4][0], kpts[3][1] - kpts[4][1])
    elif kpts[1][2] > KPT_CONF and kpts[2][2] > KPT_CONF:
        size = np.hypot(kpts[1][0] - kpts[2][0], kpts[1][1] - kpts[2][1]) * 1.8
    radius = int(max(MIN_HEAD_RADIUS, (size or 0) * ELLIPSE_SCALE / 2 * 1.6))
    return cx, cy, radius


def _blur_ellipse(frame: np.ndarray, cx: int, cy: int, radius: int) -> None:
    h, w = frame.shape[:2]
    x0, y0 = max(0, cx - radius), max(0, cy - radius)
    x1, y1 = min(w, cx + radius), min(h, cy + radius)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    k = max(7, (radius // 2) * 2 + 1)
    blurred = cv2.GaussianBlur(roi, (k, k), 0)
    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.ellipse(mask, (cx - x0, cy - y0), (radius, int(radius * 1.2)),
                0, 0, 360, 255, -1)
    roi[mask > 0] = blurred[mask > 0]


def _load_yunet(input_size=(320, 320)):
    if not YUNET_PATH.exists():
        return None
    return cv2.FaceDetectorYN.create(str(YUNET_PATH), "", input_size,
                                     score_threshold=0.6)


def _yunet_faces(detector, frame: np.ndarray) -> list[tuple[int, int, int]]:
    """Return (cx, cy, radius) for each detected face."""
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    out = []
    if faces is not None:
        for f in faces:
            x, y, fw, fh = f[:4]
            out.append((int(x + fw / 2), int(y + fh / 2),
                        int(max(fw, fh) * 0.75)))
    return out


def pick_person_frame(frame: np.ndarray, boxes: list[tuple[int, list[float]]],
                      title: str = "Select your athlete") -> int | None:
    """Show numbered person boxes; return chosen track id or None (blur all).

    Keys: 1-9 select the numbered person, 0 or ESC = athlete not visible.
    """
    disp = frame.copy()
    for n, (tid, box) in enumerate(boxes, 1):
        x0, y0, x1, y1 = map(int, box)
        cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(disp, str(n), (x0 + 4, y0 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    cv2.putText(disp, "Press number of YOUR athlete, 0/ESC = not visible",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

    scale = min(1.0, 1400 / disp.shape[1])
    if scale < 1.0:
        disp = cv2.resize(disp, None, fx=scale, fy=scale)
    cv2.imshow(title, disp)
    try:
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in (27, ord("0")):
                return None
            n = key - ord("0")
            if 1 <= n <= len(boxes):
                return boxes[n - 1][0]
    finally:
        cv2.destroyWindow(title)
        cv2.waitKey(1)


def first_frame_boxes(model, video_path: Path, device: str = "mps"):
    """Run tracking on frame 0; return (frame, [(track_id, xyxy), ...])."""
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"cannot read first frame of {video_path}")

    results = model.track(frame, persist=False, verbose=False, device=device)
    boxes = []
    r = results[0]
    if r.boxes is not None and r.boxes.id is not None:
        for tid, xyxy in zip(r.boxes.id.tolist(), r.boxes.xyxy.tolist()):
            boxes.append((int(tid), xyxy))
    return frame, boxes


def blur_clip(model, in_path: Path, out_path: Path,
              son_track_id: int | None, device: str = "mps") -> BlurReport:
    """Blur all heads/faces except the tracked athlete's.

    `model` is a ultralytics YOLO pose model; tracking state is reset per clip.
    son_track_id None => blur everyone.
    """
    if hasattr(model, "predictor") and model.predictor is not None:
        # reset tracker state so track ids match first_frame_boxes' run
        if getattr(model.predictor, "trackers", None):
            for t in model.predictor.trackers:
                t.reset()

    yunet = _load_yunet()
    cap = cv2.VideoCapture(str(in_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                          fps, (w, h))

    report = BlurReport()
    lost_since: int | None = None
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.track(frame, persist=True, verbose=False, device=device)
        r = results[0]

        son_head: tuple[int, int, int] | None = None
        others: list[tuple[int, int, int]] = []

        if r.keypoints is not None and r.boxes is not None:
            ids = (r.boxes.id.tolist() if r.boxes.id is not None
                   else [None] * len(r.boxes))
            for tid, kpts in zip(ids, r.keypoints.data.cpu().numpy()):
                head = _head_region(kpts)
                if head is None:
                    continue
                if son_track_id is not None and tid == son_track_id:
                    son_head = head
                else:
                    others.append(head)

        for cx, cy, rad in others:
            _blur_ellipse(frame, cx, cy, rad)

        # Safety net: any YuNet face outside the son's protected region
        if yunet is not None:
            for cx, cy, rad in _yunet_faces(yunet, frame):
                if son_head is not None:
                    sx, sy, sr = son_head
                    if np.hypot(cx - sx, cy - sy) < max(sr, rad):
                        continue  # it's the son's face
                _blur_ellipse(frame, cx, cy, rad)

        # Privacy-safe failure: son requested but not found -> blur his
        # likely region too is impossible (unknown), so nothing to exempt;
        # just record the gap for the report.
        if son_track_id is not None:
            if son_head is not None:
                report.son_visible_frames += 1
                if lost_since is not None:
                    report.son_lost_ranges.append((lost_since, frame_idx - 1))
                    lost_since = None
            elif lost_since is None:
                lost_since = frame_idx

        out.write(frame)
        frame_idx += 1

    if lost_since is not None:
        report.son_lost_ranges.append((lost_since, frame_idx - 1))
    report.total_frames = frame_idx

    cap.release()
    out.release()
    return report


if __name__ == "__main__":
    import argparse
    from ultralytics import YOLO

    p = argparse.ArgumentParser(description="Blur all faces except one athlete")
    p.add_argument("--in", dest="inp", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--model", default="yolo11x-pose.pt")
    p.add_argument("--device", default="mps")
    p.add_argument("--blur-all", action="store_true",
                   help="skip the picker; blur everyone")
    args = p.parse_args()

    model = YOLO(args.model)
    son_id = None
    if not args.blur_all:
        frame, boxes = first_frame_boxes(model, args.inp, args.device)
        if boxes:
            son_id = pick_person_frame(frame, boxes)

    rep = blur_clip(model, args.inp, args.out, son_id, args.device)
    print(f"{rep.total_frames} frames; athlete visible "
          f"{rep.son_visible_frames}; lost ranges: {rep.son_lost_ranges}")
