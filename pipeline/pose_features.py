#!/usr/bin/env python3
"""
Turn a technique clip into a fixed-size feature vector for classification.

Pipeline: YOLO pose per frame -> pick the tori track (most hip-vertical
travel) -> normalize (hip-centered, torso-length scaled) -> resample to
SEQ_LEN timesteps -> per-timestep features + velocities.

Two variants are produced:
  "seq"   : (SEQ_LEN, C) full sequence (future temporal models / LLM export)
  "stats" : per-channel min/max/mean/std flattened (~4*C dims) — what the
            tree classifier trains on.

Features are cached as <clip>.npz next to the video (skipped when the cache
is newer than the clip).
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

SEQ_LEN = 32
KPT_CONF = 0.3
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# keypoint indices (COCO)
L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK = 11, 12, 13, 14, 15, 16

ANGLE_JOINTS = [
    (L_SHO, L_ELB, L_WRI),   # left elbow
    (R_SHO, R_ELB, R_WRI),   # right elbow
    (L_HIP, L_KNE, L_ANK),   # left knee
    (R_HIP, R_KNE, R_ANK),   # right knee
    (L_SHO, L_HIP, L_KNE),   # left hip
    (R_SHO, R_HIP, R_KNE),   # right hip
]


def _joint_angle(a, b, c) -> float:
    """Angle at b (radians); nan if degenerate."""
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return np.nan
    return float(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)))


def extract_tracks(video_path: Path, model, device: str = "mps",
                   max_frames: int = 128
                   ) -> dict[int, list[tuple[int, np.ndarray]]]:
    """Run tracking over the clip; return {track_id: [(frame_idx, kpts17x3)]}

    max_frames caps how many frames get YOLO'd (evenly strided): the feature
    sequence is resampled to SEQ_LEN=32 anyway, so processing every frame of
    a long clip is pure waste. 0 = no cap.
    """
    if hasattr(model, "predictor") and model.predictor is not None:
        if getattr(model.predictor, "trackers", None):
            for t in model.predictor.trackers:
                t.reset()

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    stride = max(1, total // max_frames) if max_frames and total else 1

    tracks: dict[int, list[tuple[int, np.ndarray]]] = {}
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            r = model.track(frame, persist=True, verbose=False,
                            device=device)[0]
            if r.keypoints is not None and r.boxes is not None \
                    and r.boxes.id is not None:
                for tid, kpts in zip(r.boxes.id.tolist(),
                                     r.keypoints.data.cpu().numpy()):
                    tracks.setdefault(int(tid), []).append((frame_idx, kpts))
        frame_idx += 1
    cap.release()
    return tracks


def pick_tori_track(tracks: dict[int, list[tuple[int, np.ndarray]]]
                    ) -> list[tuple[int, np.ndarray]] | None:
    """The thrower moves most: largest total |Δ hip-y|, requiring decent
    presence (≥40% of the longest track)."""
    if not tracks:
        return None
    max_len = max(len(t) for t in tracks.values())

    best, best_score = None, -1.0
    for t in tracks.values():
        if len(t) < max(8, 0.4 * max_len):
            continue
        hip_y = []
        for _, k in t:
            if k[L_HIP][2] > KPT_CONF and k[R_HIP][2] > KPT_CONF:
                hip_y.append((k[L_HIP][1] + k[R_HIP][1]) / 2)
        if len(hip_y) < 5:
            continue
        score = float(np.abs(np.diff(hip_y)).sum())
        if score > best_score:
            best, best_score = t, score
    return best


def _frame_features(kpts: np.ndarray) -> np.ndarray:
    """Normalized per-frame feature vector; nan where keypoints missing."""
    xy = kpts[:17, :2].astype(np.float64).copy()
    conf = kpts[:17, 2]
    xy[conf <= KPT_CONF] = np.nan

    hip_mid = np.nanmean(xy[[L_HIP, R_HIP]], axis=0)
    sho_mid = np.nanmean(xy[[L_SHO, R_SHO]], axis=0)
    torso = np.linalg.norm(sho_mid - hip_mid)
    if not np.isfinite(torso) or torso < 1e-3:
        return np.full(17 * 2 + len(ANGLE_JOINTS) + 1, np.nan)

    norm = (xy - hip_mid) / torso                    # (17,2) hip-centered

    angles = [_joint_angle(xy[a], xy[b], xy[c]) for a, b, c in ANGLE_JOINTS]
    # torso lean from vertical
    tv = sho_mid - hip_mid
    torso_angle = float(np.arctan2(abs(tv[0]), abs(tv[1]) + 1e-9))

    return np.concatenate([norm.ravel(), angles, [torso_angle]])


def _resample(seq: np.ndarray, n: int = SEQ_LEN) -> np.ndarray:
    """Linear-resample (T,C) to (n,C), interpolating over nans per channel."""
    t_src = np.linspace(0, 1, len(seq))
    t_dst = np.linspace(0, 1, n)
    out = np.empty((n, seq.shape[1]))
    for c in range(seq.shape[1]):
        col = seq[:, c]
        good = np.isfinite(col)
        if good.sum() < 2:
            out[:, c] = 0.0
            continue
        out[:, c] = np.interp(t_dst, t_src[good], col[good])
    return out


def features_from_poses(poses: list[dict]) -> dict[str, np.ndarray] | None:
    """Build features from an existing pose list (VisualJudoAnalyzer format:
    dicts with 'frame', 'person', 'keypoints'). Groups by person index per
    frame — used by run_session where tracking ids aren't available."""
    by_person: dict[int, list[tuple[int, np.ndarray]]] = {}
    for p in poses:
        by_person.setdefault(p["person"], []).append(
            (p["frame"], np.array(p["keypoints"])))
    track = pick_tori_track(by_person)
    return _finalize(track)


def extract_clip_features(video_path: Path, model, device: str = "mps",
                          use_cache: bool = True) -> dict[str, np.ndarray] | None:
    """Main entry: clip path -> {'seq': (32,C), 'stats': (4C,)} or None."""
    cache = video_path.with_suffix(".npz")
    if use_cache and cache.exists() \
            and cache.stat().st_mtime > video_path.stat().st_mtime:
        d = np.load(cache)
        return {"seq": d["seq"], "stats": d["stats"]}

    tracks = extract_tracks(video_path, model, device)
    feats = _finalize(pick_tori_track(tracks))
    if feats is not None and use_cache:
        np.savez_compressed(cache, **feats)
    return feats


# channel layout per timestep (41 base + 41 velocity = 82):
#   0..33  normalized keypoint coords x0,y0,x1,y1,... (17 kpts)
#   34..39 joint angles: L-elbow, R-elbow, L-knee, R-knee, L-hip, R-hip
#   40     torso lean (unsigned)
_KPT_LR = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12),
           (13, 14), (15, 16)]  # (left,right) keypoint index pairs


def _flip_maps() -> tuple[np.ndarray, np.ndarray]:
    """(permutation, sign) turning a 41-channel frame into its mirror."""
    perm = np.arange(41)
    for l, r in _KPT_LR:
        perm[[2 * l, 2 * r]] = perm[[2 * r, 2 * l]]          # x
        perm[[2 * l + 1, 2 * r + 1]] = perm[[2 * r + 1, 2 * l + 1]]  # y
    for l, r in [(34, 35), (36, 37), (38, 39)]:              # L/R angles
        perm[[l, r]] = perm[[r, l]]
    sign = np.ones(41)
    sign[0:34:2] = -1.0                                      # negate x
    return perm, sign


def flip_seq(seq: np.ndarray) -> np.ndarray:
    """Mirror a (SEQ_LEN, 82) feature sequence left<->right."""
    perm, sign = _flip_maps()
    full_perm = np.concatenate([perm, perm + 41])
    full_sign = np.concatenate([sign, sign])
    return seq[:, full_perm] * full_sign


def stats_from_seq(seq: np.ndarray) -> np.ndarray:
    return np.concatenate([seq.min(0), seq.max(0),
                           seq.mean(0), seq.std(0)]).astype(np.float32)


def _finalize(track) -> dict[str, np.ndarray] | None:
    if track is None or len(track) < 8:
        return None
    frames = sorted(track, key=lambda x: x[0])
    per_frame = np.stack([_frame_features(k) for _, k in frames])
    seq = _resample(per_frame)

    vel = np.diff(seq, axis=0, prepend=seq[:1])
    seq_full = np.concatenate([seq, vel], axis=1)      # (32, 2C)

    return {"seq": seq_full.astype(np.float32),
            "stats": stats_from_seq(seq_full)}


def main() -> None:
    from ultralytics import YOLO

    p = argparse.ArgumentParser(description="Extract features for a dataset dir")
    p.add_argument("dataset", type=Path,
                   help="dir with <technique>/<clip> layout, or a single clip")
    p.add_argument("--model", default="yolo11x-pose.pt")
    p.add_argument("--device", default="mps")
    p.add_argument("--force", action="store_true", help="ignore caches")
    p.add_argument("--exclude", nargs="*", default=["youtube_"],
                   help="skip clips whose path contains a dir starting "
                        "with any of these prefixes")
    args = p.parse_args()

    model = YOLO(args.model)
    clips = ([args.dataset] if args.dataset.is_file()
             else sorted(q for q in args.dataset.rglob("*")
                         if q.suffix.lower() in VIDEO_EXTS
                         and not any(part.startswith(x)
                                     for part in q.parts
                                     for x in args.exclude)))
    ok = failed = 0
    for clip in clips:
        f = extract_clip_features(clip, model, args.device,
                                  use_cache=not args.force)
        if f is None:
            print(f"  SKIP (no usable track): {clip}")
            failed += 1
        else:
            print(f"  ✓ {clip}  seq{f['seq'].shape} stats{f['stats'].shape}")
            ok += 1
    print(f"\n{ok} extracted, {failed} skipped")


if __name__ == "__main__":
    main()
