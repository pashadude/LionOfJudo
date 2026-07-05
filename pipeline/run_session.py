#!/usr/bin/env python3
"""
One-command session pipeline.

    python -m pipeline.run_session \
        --sony /path/C0012.MP4 --iphone /path/IMG_4411.MOV \
        --imu sessions/2026-07-05/imu/ --out sessions/2026-07-05/

Steps:
 1. Audio cross-correlation      -> master (Sony) timeline
 2. IMU ritual detection          -> per-unit clock maps
 3. IMU spike segmentation        -> throw windows
 4. ffmpeg clip cutting           -> 2 raw clips per throw
 5. Interactive athlete pick      -> one keypress per clip (only human step)
 6. Face blur (all except son)    -> blurred clips
 7. Existing VisualJudoAnalyzer   -> skeleton overlay + pose JSON
 8. Existing classify_technique   -> technique guess + reasoning
 9. Existing MovementAnalyzer     -> movement narrative
10. session_report.md / .json
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline import audio_sync, imu_ingest
from pipeline.clip_extractor import cut_clip, probe_duration
from pipeline.face_blur import blur_clip, first_frame_boxes, pick_person_frame
from pipeline.throw_segmenter import ThrowWindow, segment_throws

UNIT_FILES = {"chest": "chest", "hip": "hip"}  # filename prefixes in --imu dir
MIN_AUDIO_CONFIDENCE = 3.0
CLF_PATH = REPO_ROOT / "models" / "technique_clf.joblib"
CLF_LABELS_PATH = REPO_ROOT / "models" / "technique_labels.json"


def load_learned_classifier():
    """Return (clf, labels) if a trained model exists, else None.
    Trained by pipeline/train_classifier.py from the dataset/ corpus."""
    if not (CLF_PATH.exists() and CLF_LABELS_PATH.exists()):
        return None
    import joblib
    return joblib.load(CLF_PATH), json.loads(CLF_LABELS_PATH.read_text())


def classify_learned(clf_bundle, stats) -> dict | None:
    """Classify a throw's precomputed stats features with the trained model."""
    if stats is None:
        return None
    clf, labels = clf_bundle
    import numpy as np
    x = np.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0).reshape(1, -1)
    proba = clf.predict_proba(x)[0]
    best = int(proba.argmax())
    return {
        "technique": labels[best],
        "confidence": round(float(proba[best]), 3),
        "method": "learned",
        "alternatives": {labels[i]: round(float(p), 3)
                         for i, p in enumerate(proba) if i != best and p > 0.1},
    }


def log(msg: str) -> None:
    print(f"[run_session] {msg}", flush=True)


def find_imu_logs(imu_dir: Path) -> dict[str, Path]:
    logs = {}
    for unit, prefix in UNIT_FILES.items():
        candidates = sorted(imu_dir.glob(f"{prefix}*.bin"))
        if candidates:
            logs[unit] = candidates[-1]  # latest file per unit
    return logs


def build_clock_maps(logs: dict[str, imu_ingest.ImuLog],
                     sony: Path, threshold_g: float
                     ) -> tuple[dict[str, imu_ingest.ClockMap], list[float]]:
    """Align each unit's clock to the master (Sony) timeline via the ritual."""
    claps = audio_sync.detect_claps(sony, top_n=12)
    log(f"audio transient candidates: {[round(t, 2) for t in claps]}")

    maps: dict[str, imu_ingest.ClockMap] = {}
    ritual_master_all: list[float] = []

    for unit, imu_log in logs.items():
        ritual_imu = imu_ingest.detect_sync_ritual(imu_log,
                                                   threshold_g=threshold_g)
        if len(ritual_imu) != 3:
            raise SystemExit(
                f"ERROR: could not find the 3-jump ritual in the '{unit}' log "
                f"(threshold {threshold_g}g). Try --threshold-g lower, or "
                f"check tools/imu_plot.py")

        matched = imu_ingest.match_ritual_to_claps(ritual_imu, claps)
        if matched is None:
            raise SystemExit(
                f"ERROR: no 3 audio transients match the '{unit}' ritual "
                f"spacing. Was the ritual audible on the Sony?")

        pairs_imu, pairs_master = list(ritual_imu), list(matched)

        # Optional end ritual tightens drift correction
        ritual_end = imu_ingest.detect_sync_ritual(imu_log, from_end=True,
                                                   threshold_g=threshold_g)
        if len(ritual_end) == 3 and ritual_end != ritual_imu:
            matched_end = imu_ingest.match_ritual_to_claps(ritual_end, claps)
            if matched_end and matched_end != matched:
                pairs_imu += ritual_end
                pairs_master += matched_end

        cm = imu_ingest.align_to_master(pairs_imu, pairs_master)
        worst = max(abs(r) for r in cm.residuals_ms)
        log(f"{unit}: clock map a={cm.a:.6f} b={cm.b:+.3f}s "
            f"worst residual {worst:.1f}ms")
        if worst > 50:
            log(f"WARNING: {unit} residuals above 50ms — sync may be sloppy")
        maps[unit] = cm
        ritual_master_all += pairs_master

    return maps, ritual_master_all


def interactive_pick(model, clips: list[tuple[ThrowWindow, Path]]):
    """One pass of keypresses over all clips' first frames. Returns
    {throw_id: son_track_id or None}."""
    picks: dict[int, int | None] = {}
    for w, clip in clips:
        frame, boxes = first_frame_boxes(model, clip)
        if not boxes:
            picks[w.throw_id] = None
            continue
        picks[w.throw_id] = pick_person_frame(
            frame, boxes, title=f"Throw {w.throw_id} @ {w.t_peak:.0f}s")
    return picks


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sony", required=True, type=Path,
                   help="Sony FDR-X3000 file (defines the master timeline)")
    p.add_argument("--iphone", type=Path, default=None,
                   help="iPhone file (optional second angle)")
    p.add_argument("--imu", required=True, type=Path,
                   help="directory with chest*.bin / hip*.bin logs")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--threshold-g", type=float, default=3.0,
                   help="throw spike threshold; lower for lighter kids")
    p.add_argument("--pose-model", default=str(REPO_ROOT / "yolo11x-pose.pt"))
    p.add_argument("--device", default="mps")
    p.add_argument("--scale-height", type=int, default=1080,
                   help="downscale clips for inference (0 = keep 4K)")
    p.add_argument("--blur-all", action="store_true",
                   help="skip picking; blur every face including the athlete")
    p.add_argument("--no-blur", action="store_true",
                   help="skip face blur entirely (private review only!)")
    args = p.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    throws_dir = out_dir / "throws"

    # -- 1. audio sync ------------------------------------------------------
    offset_s = 0.0
    if args.iphone:
        res = audio_sync.find_audio_offset(args.sony, args.iphone)
        log(f"audio offset iPhone->Sony: {res.offset_s:+.3f}s "
            f"(confidence {res.confidence:.2f})")
        if res.confidence < MIN_AUDIO_CONFIDENCE:
            raise SystemExit(
                "ERROR: audio sync confidence too low. Check both files have "
                "usable audio of the same scene, or sync manually and pass "
                "pre-trimmed files.")
        offset_s = res.offset_s

    # -- 2. IMU clock maps --------------------------------------------------
    log_paths = find_imu_logs(args.imu)
    if not log_paths:
        raise SystemExit(f"ERROR: no chest*.bin / hip*.bin logs in {args.imu}")
    logs = {u: imu_ingest.load_imu_log(p) for u, p in log_paths.items()}
    for u, l in logs.items():
        log(f"{u}: {len(l.t_s)} samples @ {l.sample_rate_hz}Hz "
            f"({l.t_s[-1] - l.t_s[0]:.0f}s) from {log_paths[u].name}")

    clock_maps, ritual_master = build_clock_maps(logs, args.sony,
                                                 args.threshold_g)

    # -- 3. segmentation ----------------------------------------------------
    windows = segment_throws(logs, clock_maps,
                             exclude_master_times=ritual_master,
                             threshold_g=args.threshold_g)
    if not windows:
        raise SystemExit("No throws detected. Lower --threshold-g and retry "
                         "(re-run is cheap: nothing was rendered yet).")
    log(f"{len(windows)} throw(s) detected")

    sony_duration = probe_duration(args.sony)

    # -- 4. cut raw clips ---------------------------------------------------
    scale = args.scale_height or None
    clips: list[tuple[ThrowWindow, Path, Path | None]] = []
    for w in windows:
        if w.t_peak < 0 or w.t_start > sony_duration:
            log(f"throw {w.throw_id}: outside Sony video, skipped")
            continue
        tdir = throws_dir / f"throw_{w.throw_id:02d}"
        sony_clip = cut_clip(args.sony, w.t_start,
                             min(w.t_end, sony_duration),
                             tdir / "sony_raw.mp4", scale)
        iphone_clip = None
        if args.iphone:
            i0, i1 = w.t_start - offset_s, w.t_end - offset_s
            if i1 > 0:
                iphone_clip = cut_clip(args.iphone, max(0.0, i0), i1,
                                       tdir / "iphone_raw.mp4", scale)
        clips.append((w, sony_clip, iphone_clip))
        log(f"throw {w.throw_id}: clips cut "
            f"[{w.t_start:.1f}s..{w.t_end:.1f}s]")

    # -- 5..9: heavy phase --------------------------------------------------
    from ultralytics import YOLO

    from movement_analysis import MovementAnalyzer
    from phase0_judo_analysis import VideoAnalyzer
    from phase0_visual_analysis import VisualJudoAnalyzer

    model = YOLO(args.pose_model)

    picks: dict[int, int | None] = {}
    if not args.blur_all and not args.no_blur:
        log("interactive pass: pick your athlete in each clip "
            "(number key, 0 = not visible)")
        picks = interactive_pick(model, [(w, s) for w, s, _ in clips])

    # Reuse the rule-based classifier without triggering its model load:
    # classify_technique/extract_throw_features only need self.techniques.
    classifier = VideoAnalyzer.__new__(VideoAnalyzer)
    classifier.techniques = {
        'o-soto-gari': 'Major Outer Reap',
        'o-goshi': 'Major Hip Throw',
        'ippon-seoi-nagi': 'One-Arm Shoulder Throw',
        'uki-goshi': 'Floating Hip Throw',
    }
    movement = MovementAnalyzer()

    clf_bundle = load_learned_classifier()
    log("technique classifier: "
        + ("learned (models/technique_clf.joblib)" if clf_bundle
           else "rule-based fallback (train one with "
                "pipeline/train_classifier.py)"))

    from pipeline import reference_bank as refbank
    bank = refbank.load_bank()
    log("reference bank: "
        + (f"{len(bank['X'])} clips loaded" if bank
           else "none (build with pipeline/reference_bank.py --build)"))

    # One analyzer for all throws (loads the pose model once); output_dir is
    # retargeted per throw before each process_video call.
    visual = VisualJudoAnalyzer(output_dir=throws_dir)

    report: dict = {"session": str(out_dir), "sony": str(args.sony),
                    "iphone": str(args.iphone) if args.iphone else None,
                    "audio_offset_s": offset_s, "throws": []}

    for w, sony_clip, iphone_clip in clips:
        tdir = sony_clip.parent
        entry: dict = {
            "throw_id": w.throw_id,
            "t_peak_s": round(w.t_peak, 2),
            "window_s": [round(w.t_start, 2), round(w.t_end, 2)],
            "power": {u: asdict(m) for u, m in w.metrics.items()},
            "videos": {},
        }

        for name, clip in (("sony", sony_clip), ("iphone", iphone_clip)):
            if clip is None:
                continue
            # 6. blur
            if args.no_blur:
                work = clip
                blur_note = "NOT BLURRED (private)"
            else:
                work = tdir / f"{name}_blurred.mp4"
                rep = blur_clip(model, clip, work, picks.get(w.throw_id),
                                device=args.device)
                blur_note = (f"athlete visible {rep.son_visible_frames}/"
                             f"{rep.total_frames} frames"
                             + (f", lost {rep.son_lost_ranges}"
                                if rep.son_lost_ranges else ""))

            # 7. skeleton overlay + pose JSON (existing analyzer, unchanged)
            visual.output_dir = tdir
            analysis = visual.process_video(work)
            json_path = tdir / f"{work.stem}_analysis.json"
            json_path.write_text(json.dumps(analysis, indent=2))

            entry["videos"][name] = {
                "raw": clip.name,
                "annotated": f"{work.stem}_cam0_annotated.mp4",
                "blur": blur_note,
            }

            # 8+9 only need one angle; use the Sony pass
            if name == "sony" and analysis and analysis.get("poses"):
                poses = analysis["poses"]
                feats = classifier.extract_throw_features(
                    poses, 0, len(poses) - 1)
                rules = classifier.classify_technique(
                    {"features": feats,
                     "hip_drop_px": feats.get("max_hip_drop", 0)},
                    video_name="")
                rules["method"] = "rules"

                from pipeline.pose_features import features_from_poses
                pf = features_from_poses(poses)
                stats = pf["stats"] if pf else None

                learned = classify_learned(clf_bundle, stats) \
                    if clf_bundle else None
                if learned:
                    entry["technique"] = learned
                    if learned["technique"] not in rules["technique"]:
                        entry["technique_rules_disagree"] = rules
                else:
                    entry["technique"] = rules

                if bank and stats is not None:
                    entry["nearest_references"] = refbank.match(
                        bank, stats, top_k=3)
                entry["movement_phases"] = len(
                    movement.extract_movement_phases(poses))
                entry["analysis_json"] = json_path.name

        report["throws"].append(entry)
        log(f"throw {w.throw_id}: done")

    # -- 10. reports --------------------------------------------------------
    (out_dir / "session_report.json").write_text(
        json.dumps(report, indent=2, default=str))
    write_markdown_report(report, out_dir / "session_report.md")
    log(f"session report: {out_dir / 'session_report.md'}")


def write_markdown_report(report: dict, path: Path) -> None:
    lines = [f"# Training Session Report", "",
             f"- Sony: `{report['sony']}`"]
    if report["iphone"]:
        lines.append(f"- iPhone: `{report['iphone']}` "
                     f"(offset {report['audio_offset_s']:+.3f}s)")
    lines += ["", f"## {len(report['throws'])} Throws", ""]

    for t in report["throws"]:
        lines.append(f"### Throw {t['throw_id']} @ {t['t_peak_s']}s")
        tech = t.get("technique") or {}
        if tech:
            detail = tech.get("reasoning") or ", ".join(
                f"{k} {v}" for k, v in tech.get("alternatives", {}).items())
            lines.append(f"- **Technique:** {tech.get('technique', '?')} "
                         f"(confidence {tech.get('confidence', '?')}, "
                         f"{tech.get('method', 'rules')})"
                         + (f" — {detail}" if detail else ""))
            dis = t.get("technique_rules_disagree")
            if dis:
                lines.append(f"  - rules disagreed: {dis['technique']} "
                             f"({dis.get('reasoning', '')})")
        refs = t.get("nearest_references")
        if refs:
            lines.append("- **Nearest waza (full catalog):** " + ", ".join(
                f"{r['waza']} {r['similarity']:.2f} [{r['category']}]"
                for r in refs))
        for unit, pm in t.get("power", {}).items():
            lines.append(
                f"- **{unit}:** peak {pm['peak_g']:.1f}g, "
                f"rotation {pm['max_rotation_dps']:.0f}°/s, "
                f"duration {pm['duration_ms']:.0f}ms, "
                f"power index {pm['power_index']:.2f}")
        for name, v in t.get("videos", {}).items():
            lines.append(f"- {name}: [{v['annotated']}](throws/"
                         f"throw_{t['throw_id']:02d}/{v['annotated']}) "
                         f"({v['blur']})")
        lines.append("")

    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
