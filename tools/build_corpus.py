#!/usr/bin/env python3
"""
Organize training clips into the dataset/<technique>/ layout.

    # 1. auto-sort an inbox of files whose names contain a technique slug
    python tools/build_corpus.py --auto ~/JudoClips/inbox/

    # 2. keypress-label whatever --auto could not match
    python tools/build_corpus.py --label ~/JudoClips/inbox/

    # 3. optional: split multi-rep clips into one clip per repetition
    python tools/build_corpus.py --split-reps dataset/o-goshi/long_drill.mp4

    # 4. import confirmed throws from a processed session
    python tools/build_corpus.py --from-session sessions/2026-07-05/

Every run ends with the per-technique sample count table
(goal: >=20 per class, ideally >=50).
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DATASET_DIR = REPO_ROOT / "dataset"
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}

# Slugs matched inside filenames (loose: -, _, space all accepted).
# Aliases map spelling variants to the canonical folder name.
TECHNIQUES = [
    "o-soto-gari", "o-goshi", "uki-goshi", "ippon-seoi-nage",
    "seoi-nage", "uchi-mata", "harai-goshi", "tai-otoshi",
    "ko-uchi-gari", "o-uchi-gari", "tomoe-nage", "morote-seoi-nage",
]
ALIASES = {
    "ippon-seoi-nagi": "ippon-seoi-nage",   # spelling used in older repo files
    "osoto-gari": "o-soto-gari",
    "osotogari": "o-soto-gari",
    "ogoshi": "o-goshi",
    "ukigoshi": "uki-goshi",
}


def _norm(name: str) -> str:
    return re.sub(r"[\s_]+", "-", name.lower())


def match_technique(filename: str) -> str | None:
    """Return canonical technique slug found in the filename, or None.
    Longest match wins (o-soto-gari before seoi-nage etc.)."""
    n = _norm(filename)
    candidates = []
    for slug in TECHNIQUES:
        if slug in n:
            candidates.append(slug)
    for alias, canon in ALIASES.items():
        if alias in n:
            candidates.append(canon)
    return max(candidates, key=len) if candidates else None


def list_clips(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(q for q in path.rglob("*")
                  if q.suffix.lower() in VIDEO_EXTS)


def file_into(clip: Path, technique: str, move: bool) -> Path:
    dest_dir = DATASET_DIR / technique
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / clip.name
    i = 1
    while dest.exists():
        dest = dest_dir / f"{clip.stem}_{i}{clip.suffix}"
        i += 1
    (shutil.move if move else shutil.copy2)(str(clip), str(dest))
    return dest


def cmd_auto(inbox: Path, move: bool) -> None:
    unmatched = []
    for clip in list_clips(inbox):
        tech = match_technique(clip.name)
        if tech:
            dest = file_into(clip, tech, move)
            print(f"  {clip.name}  ->  {tech}/{dest.name}")
        else:
            unmatched.append(clip)
    if unmatched:
        print(f"\n{len(unmatched)} file(s) had no technique in the name — "
              f"label them with:\n  python tools/build_corpus.py --label "
              f"{inbox}")


def cmd_label(inbox: Path, move: bool) -> None:
    import cv2

    clips = [c for c in list_clips(inbox) if match_technique(c.name) is None]
    if not clips:
        print("nothing to label")
        return

    print("Keys: 1-9 = technique below, s = skip, d = discard file, q = quit")
    for i, t in enumerate(TECHNIQUES[:9], 1):
        print(f"  {i} = {t}")

    for clip in clips:
        cap = cv2.VideoCapture(str(clip))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n // 2))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            print(f"  unreadable, skipping: {clip}")
            continue

        h, w = frame.shape[:2]
        scale = min(1.0, 1200 / w)
        disp = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 \
            else frame.copy()
        cv2.putText(disp, clip.name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
        cv2.imshow("label clip", disp)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                cv2.destroyAllWindows()
                return
            if key == ord("s"):
                print(f"  skip {clip.name}")
                break
            if key == ord("d"):
                clip.unlink()
                print(f"  discarded {clip.name}")
                break
            idx = key - ord("1")
            if 0 <= idx < min(9, len(TECHNIQUES)):
                tech = TECHNIQUES[idx]
                dest = file_into(clip, tech, move)
                print(f"  {clip.name}  ->  {tech}/{dest.name}")
                break
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def cmd_split_reps(clip: Path, device: str) -> None:
    """Cut a multi-repetition drill clip into one sub-clip per rep using
    hip-drop events from the pose track."""
    import numpy as np
    from ultralytics import YOLO

    from pipeline.clip_extractor import cut_clip, probe_duration
    from pipeline.pose_features import extract_tracks, pick_tori_track, \
        L_HIP, R_HIP, KPT_CONF

    tech = clip.parent.name if clip.parent.parent == DATASET_DIR else None

    model = YOLO(str(REPO_ROOT / "yolo11x-pose.pt"))
    tracks = extract_tracks(clip, model, device)
    track = pick_tori_track(tracks)
    if track is None:
        print(f"  no usable person track in {clip}")
        return

    import cv2
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    hip = [(f, (k[L_HIP][1] + k[R_HIP][1]) / 2) for f, k in track
           if k[L_HIP][2] > KPT_CONF and k[R_HIP][2] > KPT_CONF]
    if len(hip) < 20:
        print(f"  too few hip detections in {clip}")
        return

    frames = np.array([h[0] for h in hip])
    ys = np.array([h[1] for h in hip])

    # smooth then find sharp downward hip movements (throw/entry events)
    k = max(3, int(fps // 6) | 1)
    ys_s = np.convolve(ys, np.ones(k) / k, mode="same")
    vel = np.gradient(ys_s)
    thresh = max(2.0, 2.5 * np.std(vel))

    events, last_f = [], -1e9
    for i, v in enumerate(vel):
        if v > thresh and frames[i] - last_f > fps * 1.5:
            events.append(frames[i])
            last_f = frames[i]

    if len(events) < 2:
        print(f"  only {len(events)} rep event(s) found — leaving clip as-is")
        return

    duration = probe_duration(clip)
    print(f"  {len(events)} repetitions detected in {clip.name}")
    for i, f in enumerate(events, 1):
        t = f / fps
        out = clip.parent / f"{clip.stem}_rep{i:02d}.mp4"
        cut_clip(clip, max(0.0, t - 2.5), min(duration, t + 2.0), out)
        print(f"    rep {i}: {t:.1f}s -> {out.name}")
    retired = clip.with_suffix(clip.suffix + ".original")
    clip.rename(retired)
    print(f"  original kept as {retired.name} (not counted in dataset)")


def cmd_from_session(session: Path, move: bool) -> None:
    """Import throws confirmed in a processed session's report. You are
    prompted per throw because run_session's technique guess may be wrong."""
    report_path = session / "session_report.json"
    if not report_path.exists():
        print(f"no session_report.json in {session}")
        return
    report = json.loads(report_path.read_text())

    print("For each throw enter the technique number, s to skip:")
    for i, t in enumerate(TECHNIQUES[:9], 1):
        print(f"  {i} = {t}")

    for throw in report.get("throws", []):
        tid = throw["throw_id"]
        guess = (throw.get("technique") or {}).get("technique", "?")
        clip = session / "throws" / f"throw_{tid:02d}" / "sony_raw.mp4"
        if not clip.exists():
            continue
        ans = input(f"throw {tid} @ {throw['t_peak_s']}s "
                    f"(pipeline guessed: {guess})> ").strip().lower()
        if ans.isdigit() and 1 <= int(ans) <= min(9, len(TECHNIQUES)):
            tech = TECHNIQUES[int(ans) - 1]
            dest = file_into(clip, tech, move=False)  # always copy sessions
            print(f"  -> {tech}/{dest.name}")


def print_counts() -> None:
    print("\n=== dataset counts (goal: >=20/class, ideally >=50) ===")
    if not DATASET_DIR.exists():
        print("  (empty — no dataset/ yet)")
        return
    total = 0
    for d in sorted(DATASET_DIR.iterdir()):
        if d.is_dir():
            n = len([c for c in d.iterdir()
                     if c.suffix.lower() in VIDEO_EXTS])
            total += n
            flag = "✓" if n >= 50 else ("~" if n >= 20 else "✗ need more")
            print(f"  {d.name:24s} {n:4d}  {flag}")
    print(f"  {'TOTAL':24s} {total:4d}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--auto", type=Path, metavar="INBOX",
                   help="sort files whose names contain a technique slug")
    p.add_argument("--label", type=Path, metavar="INBOX",
                   help="keypress-label files without a slug in the name")
    p.add_argument("--split-reps", type=Path, metavar="CLIP",
                   help="cut a multi-rep drill clip into per-rep clips")
    p.add_argument("--from-session", type=Path, metavar="SESSION_DIR",
                   help="import confirmed throws from a processed session")
    p.add_argument("--copy", action="store_true",
                   help="copy instead of move (originals stay in the inbox)")
    p.add_argument("--device", default="mps")
    args = p.parse_args()

    move = not args.copy
    if args.auto:
        cmd_auto(args.auto, move)
    if args.label:
        cmd_label(args.label, move)
    if args.split_reps:
        cmd_split_reps(args.split_reps, args.device)
    if args.from_session:
        cmd_from_session(args.from_session, move)

    print_counts()


if __name__ == "__main__":
    main()
