"""Local Sony/iPhone session import and synchronized media exports."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from pipeline.face_blur import BlurReport, build_privacy_processor
from pipeline.clip_extractor import (
    cut_clip,
    probe_duration,
    probe_fps,
    verify_media_export,
)
from pipeline.video_event_detection import (
    EventMetrics,
    motion_energy,
    recovery_to_stable_s,
    recover_blue_pose,
    select_blue_detection,
    suggest_event_metrics,
)
from pipeline.video_pose_metrics import json_safe
from pipeline.video_pose_metrics import compute_pose_metrics
from pipeline.video_review_contract import (
    AnchorPair,
    ReviewEvent,
    ReviewSession,
    validate_review_payload,
    validate_review_session,
)
from pipeline.video_review_reports import write_reports
from pipeline.video_review_metrics import (
    canonical_metric_schema,
    canonicalize_frame_metrics,
)
from pipeline.video_sync import fit_time_map
from pipeline.voice_labels import load_whisper_json, suggest_techniques


EXPORT_TOLERANCE_S = 0.75
PREVIEW_RADIUS_S = 2.0
ANNOTATION_KEYS = {
    "potvrdena_tehnika",
    "ocena",
    "napomena",
    "trainer_annotations",
    "coach_annotations",
    "annotations",
}


def _finite(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} mora biti JSON broj")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} mora biti konacan")
    return result


ANALYSIS_LIMITATION = (
    "Ograničenje: analiza video-pokreta koristi grubo uzorkovanje; "
    "preskočeni kadrovi nisu rekonstruisani."
)


def _analysis_profile(
    source_fps: float,
    requested_analysis_fps: float | None,
) -> dict[str, float | int | None | str]:
    source = _finite(source_fps, "source_fps")
    if source <= 0.0:
        raise ValueError("source_fps mora biti pozitivan")
    if requested_analysis_fps is None:
        return {
            "source_fps": source,
            "requested_analysis_fps": None,
            "effective_analysis_fps": source,
            "stride": 1,
            "analysis_limitation": ANALYSIS_LIMITATION,
        }
    requested = _finite(requested_analysis_fps, "analysis_fps")
    if requested <= 0.0 or requested > source:
        raise ValueError("analysis_fps mora biti konacan, pozitivan i ne veci od source_fps")
    stride = max(1, int(round(source / requested)))
    return {
        "source_fps": source,
        "requested_analysis_fps": requested,
        "effective_analysis_fps": source / stride,
        "stride": stride,
        "analysis_limitation": ANALYSIS_LIMITATION,
    }


def _probe_optional_fps(video: Path) -> float | None:
    """Return a finite source FPS, or None when the optional probe is unavailable."""
    try:
        value = probe_fps(Path(video))
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None
    try:
        value = _finite(value, "fps")
    except (TypeError, ValueError):
        return None
    return value if value > 0.0 else None


def _format_number(value: float) -> str:
    return format(float(value), ".12g")


def _signed_number(value: float) -> str:
    formatted = _format_number(abs(value))
    return f"+{formatted}" if value >= 0.0 else f"-{formatted}"


def make_side_by_side(
    sony: Path,
    iphone: Path,
    slope: float,
    intercept: float,
    end_s: float,
    output: Path,
    height: int = 720,
) -> Path:
    """Export Sony-master side-by-side video with Sony audio retained."""
    slope = _finite(slope, "slope")
    intercept = _finite(intercept, "intercept")
    end_s = _finite(end_s, "end_s")
    if slope <= 0.0:
        raise ValueError("slope vremenske mape mora biti pozitivan")
    if end_s <= 0.0:
        raise ValueError("kraj side-by-side prozora mora biti pozitivan")
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise ValueError("visina side-by-side izvoza mora biti pozitivan ceo broj")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sony_filter = (
        f"[0:v]trim=start=0:end={end_s:.3f},setpts=PTS-STARTPTS,"
        f"scale=-2:{height}[sony]"
    )
    iphone_filter = (
        f"[1:v]setpts={_format_number(slope)}*PTS"
        f"{_signed_number(intercept)}/TB,scale=-2:{height}[iphone]"
    )
    filter_complex = f"{sony_filter};{iphone_filter};[sony][iphone]hstack=inputs=2:shortest=1[v]"
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-i",
        str(sony),
        "-i",
        str(iphone),
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-shortest",
        "-movflags",
        "+faststart",
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True)
    verify_media_export(
        output,
        end_s,
        EXPORT_TOLERANCE_S,
        probe=probe_duration,
    )
    return output


def write_review_json(output_dir: Path, payload: Mapping[str, Any]) -> Path:
    """Atomically write strict JSON to the canonical review path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "review.json"
    temporary = output_dir / "review.json.tmp"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                json_safe(dict(payload)),
                handle,
                ensure_ascii=True,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _write_analysis_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                json_safe(dict(payload)),
                handle,
                ensure_ascii=True,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def run_pose_analysis(
    sony: Path,
    start_s: float,
    end_s: float,
    blue_seed: Sequence[float],
    *,
    model: Any | None = None,
    video_capture_factory: Any | None = None,
    fps: float | None = None,
    device: str = "mps",
    model_path: Path | str = "yolo11x-pose.pt",
    event_threshold: float = 0.5,
    analysis_fps: float | None = None,
) -> dict[str, Any]:
    """Track the coach-selected athlete and derive Task 3 video metrics.

    YOLO tracking and OpenCV are imported only when this production path is
    used. Tests inject both boundaries, so no model download or video decode
    is needed for unit coverage.
    """
    start_s = _finite(start_s, "start_s")
    end_s = _finite(end_s, "end_s")
    if start_s < 0.0 or end_s <= start_s:
        raise ValueError("pose prozor mora biti nenegativan i rastuci")
    seed = _blue_seed(blue_seed)

    if video_capture_factory is None:
        import cv2

        video_capture_factory = cv2.VideoCapture
        position_property = cv2.CAP_PROP_POS_FRAMES
        fps_property = cv2.CAP_PROP_FPS
    else:
        position_property = 1
        fps_property = 5

    capture = video_capture_factory(str(Path(sony)))
    fps_value = float(fps) if fps is not None else float(capture.get(fps_property) or 0.0)
    if not math.isfinite(fps_value) or fps_value <= 0.0:
        fps_value = probe_fps(Path(sony))
    profile = _analysis_profile(fps_value, analysis_fps)
    effective_fps = float(profile["effective_analysis_fps"])
    stride = int(profile["stride"])

    if model is None:
        model_path = Path(model_path).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(
                f"lokalna YOLO težina nije pronađena; automatsko preuzimanje nije dozvoljeno: {model_path}"
            )
        from ultralytics import YOLO

        model = YOLO(str(model_path))
    if hasattr(model, "predictor") and model.predictor is not None:
        for tracker in getattr(model.predictor, "trackers", []) or []:
            tracker.reset()

    start_frame = int(math.ceil(start_s * fps_value))
    capture.set(position_property, start_frame)
    source_frame_index = start_frame
    next_sample_frame = start_frame
    pose_frames: list[np.ndarray] = []
    timestamps: list[float] = []
    previous_bbox: Any | None = None
    selected_track_id: Any | None = None
    selected_seen = False

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        timestamp = source_frame_index / fps_value
        current_source_frame = source_frame_index
        source_frame_index += 1
        if timestamp < start_s:
            continue
        if timestamp >= end_s:
            break
        if current_source_frame < next_sample_frame:
            continue
        next_sample_frame += stride

        result = model.track(frame, persist=True, verbose=False, device=device)[0]
        candidates = _tracking_candidates(result)
        selected = None
        if selected_track_id is None:
            selected = select_blue_detection(candidates, seed)
            if selected is not None:
                selected_track_id = selected.get("track_id")
        else:
            selected = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate.get("track_id") == selected_track_id
                ),
                None,
            )
            if selected is None and previous_bbox is not None:
                selected = recover_blue_pose(
                    candidates,
                    previous_bbox,
                    frame=frame,
                    previous_track_id=selected_track_id,
                )
                if selected is not None:
                    selected_track_id = selected.get("track_id", selected_track_id)

        if selected is None:
            pose_frames.append(_missing_pose())
        else:
            selected_seen = True
            previous_bbox = selected["bbox"]
            pose_frames.append(selected["keypoints"])
        timestamps.append(float(timestamp))

    capture.release()
    metrics = compute_pose_metrics(pose_frames, effective_fps, timestamps)
    energy = motion_energy(metrics)
    suggested = suggest_event_metrics(
        energy,
        effective_fps,
        float(event_threshold),
        injury_cutoff_s=end_s,
        timestamps=timestamps,
    )
    event_metrics = [
        _enrich_event_metrics(event, metrics, energy)
        for event in suggested
        if event.status != "povreda"
    ]
    return {
        "fps": effective_fps,
        "source_fps": profile["source_fps"],
        "requested_analysis_fps": profile["requested_analysis_fps"],
        "effective_analysis_fps": profile["effective_analysis_fps"],
        "stride": profile["stride"],
        "analysis_start_s": start_s,
        "analysis_limitation": profile["analysis_limitation"],
        "selected_track_id": selected_track_id,
        "blue_seed_sony": seed,
        "athlete_seen": selected_seen,
        "frame_metrics": [metric.to_dict() for metric in metrics],
        "motion_energy": json_safe(energy),
        "events": [
            {"event_id": f"e-{index:03d}", **event.to_dict()}
            for index, event in enumerate(event_metrics, start=1)
        ],
    }


def _missing_pose() -> np.ndarray:
    pose = np.zeros((17, 3), dtype=float)
    return pose


def _tracking_candidates(result: Any) -> list[dict[str, Any]]:
    if result.boxes is None or result.keypoints is None:
        return []
    boxes = np.asarray(_tensor_to_numpy(result.boxes.xyxy), dtype=float)
    keypoints = np.asarray(
        _tensor_to_numpy(result.keypoints.data), dtype=float
    )
    ids_value = getattr(result.boxes, "id", None)
    ids = None if ids_value is None else np.asarray(_tensor_to_numpy(ids_value)).reshape(-1)
    candidates = []
    for index, (bbox, pose) in enumerate(zip(boxes, keypoints)):
        if len(bbox) != 4 or pose.shape[0] < 17 or pose.shape[1] < 3:
            continue
        track_id = None if ids is None or index >= len(ids) else int(ids[index])
        candidates.append({
            "bbox": tuple(float(value) for value in bbox),
            "track_id": track_id,
            "keypoints": np.asarray(pose[:17, :3], dtype=float),
        })
    return candidates


def _tensor_to_numpy(value: Any) -> Any:
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return value


def _enrich_event_metrics(
    event: EventMetrics,
    metrics: Sequence[Any],
    energy: Sequence[float | None],
) -> EventMetrics:
    samples = [
        (metric, energy[index])
        for index, metric in enumerate(metrics)
        if event.sony_start_s <= metric.timestamp_s <= event.sony_end_s
    ]
    speeds = [
        metric.brzina_ulaska_norm_s
        for metric, _ in samples
        if metric.brzina_ulaska_norm_s is not None
    ]
    rotations = [
        abs(metric.rotation_2d_dps)
        for metric, _ in samples
        if metric.rotation_2d_dps is not None
    ]
    hip_levels = [
        metric.hip_level_norm
        for metric, _ in samples
        if metric.hip_level_norm is not None
    ]
    stance_widths = [
        metric.stance_width_norm
        for metric, _ in samples
        if metric.stance_width_norm is not None
    ]
    energies = [value for _, value in samples if value is not None]
    return replace(
        event,
        brzina_ulaska_norm=max(speeds) if speeds else None,
        rotacija_trupa_2d_dps=max(rotations) if rotations else None,
        promena_visine_kukova_norm=(
            max(hip_levels) - min(hip_levels) if hip_levels else None
        ),
        sirina_stava_norm=max(stance_widths) if stance_widths else None,
        vreme_oporavka_s=recovery_to_stable_s(
            [metric.timestamp_s for metric in metrics],
            energy,
            event.sony_start_s,
            event.sony_end_s,
        ),
        intenzitet_pokreta_0_100=(max(energies) * 100.0 if energies else None),
    )


def _as_anchors(values: Iterable[AnchorPair | Mapping[str, Any]]) -> list[AnchorPair]:
    anchors = [
        value if isinstance(value, AnchorPair) else AnchorPair.from_dict(dict(value))
        for value in values
    ]
    return anchors


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_record(path: Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"izvorni video ne postoji: {path}")
    return {"path": str(path), "sha256": _sha256(path)}


def _blue_seed(value: Sequence[float]) -> list[float]:
    if isinstance(value, (str, bytes)) or len(value) != 4:
        raise ValueError("blue_seed mora imati cetiri koordinate")
    return [_finite(item, "blue_seed") for item in value]


def _has_trainer_annotations(payload: Mapping[str, Any]) -> bool:
    for key in ("trainer_annotations", "coach_annotations", "annotations"):
        value = payload.get(key)
        if value not in (None, "", {}, []):
            return True
    for event in payload.get("events", []):
        if not isinstance(event, Mapping):
            continue
        for key in ANNOTATION_KEYS:
            value = event.get(key)
            if value not in (None, "", {}, []):
                return True
    return False


def _merge_trainer_annotations(
    payload: dict[str, Any], previous: Mapping[str, Any]
) -> dict[str, Any]:
    for key in ("trainer_annotations", "coach_annotations", "annotations"):
        if key in previous and previous[key] not in (None, "", {}, []):
            payload[key] = json_safe(previous[key])

    previous_events = {
        event.get("event_id"): event
        for event in previous.get("events", [])
        if isinstance(event, Mapping) and event.get("event_id")
    }
    current_ids = {event.get("event_id") for event in payload.get("events", [])}
    for event in payload.get("events", []):
        old = previous_events.get(event.get("event_id"), {})
        for key in ANNOTATION_KEYS:
            if key in old and old[key] not in (None, "", {}, []):
                event[key] = json_safe(old[key])
    orphaned = [
        json_safe(dict(item))
        for item in previous.get("orphaned_annotations", [])
        if isinstance(item, Mapping)
    ]
    for event_id, old in previous_events.items():
        fields = {
            key: json_safe(old[key])
            for key in ANNOTATION_KEYS
            if key in old and old[key] not in (None, "", {}, [])
        }
        if event_id not in current_ids and fields:
            orphaned.append({
                "source_event_id": event_id,
                "reason": "neupareni_dogadjaj_pri_ponovnom_uvozu",
                **fields,
            })
    if orphaned:
        payload["orphaned_annotations"] = orphaned
    return payload


def _window_for_clip(
    start_s: float,
    end_s: float,
    duration_s: float,
) -> tuple[float, float] | None:
    start = max(0.0, float(start_s))
    end = min(float(end_s), duration_s)
    return (start, end) if end > start else None


def _injury_trace_window(
    injury_cutoff_s: float, sony_duration_s: float
) -> tuple[float, float] | None:
    """Return a one-second trace window, including an end-of-source cutoff."""
    end = min(sony_duration_s, injury_cutoff_s + 1.0)
    start = max(0.0, end - 1.0)
    return (start, end) if end > start else None


def _export_clip(
    source: Path,
    start_s: float,
    end_s: float,
    output: Path,
    source_duration_s: float,
) -> Path | None:
    window = _window_for_clip(start_s, end_s, source_duration_s)
    if window is None:
        return None
    start, end = window
    result = cut_clip(source, start, end, output)
    verify_media_export(
        result,
        end - start,
        EXPORT_TOLERANCE_S,
        probe=probe_duration,
    )
    return result


def _export_private_clip(
    source: Path,
    start_s: float,
    end_s: float,
    output: Path,
    source_duration_s: float,
    privacy_processor: Callable[[Path, Path], BlurReport],
) -> BlurReport | None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".raw-private-", dir=output.parent
    ) as raw:
        raw_path = Path(raw) / output.name
        exported = _export_clip(
            source, start_s, end_s, raw_path, source_duration_s
        )
        if exported is None:
            return None
        report = privacy_processor(Path(exported), output)
        if not isinstance(report, BlurReport) or not report.privacy_verified:
            reason = report.failure_reason if isinstance(report, BlurReport) else None
            raise ValueError(reason or "privatnost izvedenog videa nije potvrđena")
        return report


def _manifest_row(
    report: BlurReport, relative_path: str, media_type: str
) -> dict[str, Any]:
    return report.to_manifest(relative_path, media_type)


def _event_payload(
    raw: Mapping[str, Any],
    cutoff_s: float,
    time_map: Any,
) -> dict[str, Any] | None:
    event_id = str(raw.get("event_id", ""))
    start = _finite(raw.get("sony_start_s"), "sony_start_s")
    end = _finite(raw.get("sony_end_s"), "sony_end_s")
    if not event_id or start < 0.0 or end <= start or start >= cutoff_s:
        return None
    end = min(end, cutoff_s)
    if end <= start:
        return None
    injury_event = bool(
        raw.get("prijavljen_povredni_dogadjaj", False)
        or raw.get("status") == "povreda"
    )
    event = {
        "event_id": event_id,
        "sony_start_s": start,
        "sony_end_s": end,
        "iphone_start_s": (start - time_map.intercept) / time_map.slope,
        "iphone_end_s": (end - time_map.intercept) / time_map.slope,
        "predlog_tehnike": None,
        "potvrdena_tehnika": None,
        "glasovna_fraza": None,
        "pouzdanost_glasa": 0.0,
        "iskljuceno_iz_statistike": injury_event,
    }
    metrics = raw.get("metrics")
    if isinstance(metrics, Mapping):
        event.update(json_safe(dict(metrics)))
    for key in (
        "status",
        "brzina_ulaska_norm",
        "rotacija_trupa_2d_dps",
        "promena_visine_kukova_norm",
        "sirina_stava_norm",
        "vreme_oporavka_s",
        "intenzitet_pokreta_0_100",
        "predlog_tehnike",
        "potvrdena_tehnika",
        "glasovna_fraza",
        "pouzdanost_glasa",
        "iskljuceno_iz_statistike",
    ):
        if key in raw:
            event[key] = json_safe(raw[key])
    if injury_event:
        event["iskljuceno_iz_statistike"] = True
        event["prijavljen_povredni_dogadjaj"] = True
    return json_safe(event)


def _write_summary(output_dir: Path, payload: Mapping[str, Any]) -> None:
    write_review_json(output_dir / "analysis", payload | {"summary": True})
    summary_path = output_dir / "analysis" / "review.json"
    summary_path.replace(output_dir / "analysis" / "import_summary.json")


def import_session(
    sony: Path,
    iphone: Path,
    output_dir: Path,
    anchors: Sequence[AnchorPair | Mapping[str, Any]],
    injury_cutoff_s: float,
    blue_seed: Sequence[float],
    transcript_path: Path | None = None,
    *,
    force_reimport: bool = False,
    analysis_fps: float | None = None,
    model_path: Path | str = "yolo11x-pose.pt",
    device: str = "mps",
    event_threshold: float = 0.5,
    privacy_processor: Callable[[Path, Path], BlurReport] | None = None,
) -> Path:
    """Import one Sony-master session and return its canonical review path."""
    sony_input = Path(os.path.abspath(Path(sony).expanduser()))
    iphone_input = Path(os.path.abspath(Path(iphone).expanduser()))
    output_input = Path(os.path.abspath(Path(output_dir).expanduser()))
    sony = sony_input.resolve()
    iphone = iphone_input.resolve()
    output_dir = output_input.resolve()
    for camera, logical_source, resolved_source in (
        ("Sony", sony_input, sony),
        ("iPhone", iphone_input, iphone),
    ):
        if logical_source.is_relative_to(output_input) or resolved_source.is_relative_to(
            output_dir
        ):
            raise ValueError(
                f"izvorni {camera} video mora biti izvan direktorijuma sesije"
            )
    review_path = output_dir / "review.json"
    previous: dict[str, Any] | None = None
    if review_path.is_file():
        with review_path.open(encoding="utf-8") as handle:
            previous = json.load(handle)
        if _has_trainer_annotations(previous) and not force_reimport:
            raise ValueError(
                "postojeca trenerova zabelezba zahteva --force-reimport"
            )

    injury_cutoff_s = _finite(injury_cutoff_s, "injury_cutoff_s")
    if injury_cutoff_s <= 0.0:
        raise ValueError("injury cutoff mora biti pozitivan")
    blue_seed = _blue_seed(blue_seed)
    anchor_values = _as_anchors(anchors)
    time_map = fit_time_map(anchor_values)

    sony_record = _source_record(sony)
    iphone_record = _source_record(iphone)
    sony_fps = _probe_optional_fps(sony)
    iphone_fps = _probe_optional_fps(iphone)
    sony_duration_s = _finite(probe_duration(sony), "sony_duration_s")
    iphone_duration_s = _finite(probe_duration(iphone), "iphone_duration_s")
    if sony_duration_s <= 0.0 or iphone_duration_s <= 0.0:
        raise ValueError("trajanje izvornog videa mora biti pozitivno")
    if injury_cutoff_s > sony_duration_s:
        raise ValueError("injury cutoff je posle kraja Sony videa")

    analysis_start_s = min(anchor.sony_s for anchor in anchor_values)
    pose_result = run_pose_analysis(
        sony,
        analysis_start_s,
        injury_cutoff_s,
        blue_seed,
        analysis_fps=analysis_fps,
        model_path=model_path,
        device=device,
        event_threshold=event_threshold,
    )
    if isinstance(pose_result, Mapping):
        frame_metrics = canonicalize_frame_metrics(
            list(pose_result.get("frame_metrics", [])),
            trust_precomputed_acceleration=True,
        )
        pose_events = pose_result.get("events", [])
        pose_summary = {
            "fps": pose_result.get("fps"),
            "selected_track_id": pose_result.get("selected_track_id"),
            "athlete_seen": pose_result.get("athlete_seen", False),
        }
    else:
        frame_metrics = []
        pose_events = pose_result
        pose_summary = {}
    if sony_fps is None:
        fallback_fps = (
            pose_result.get("source_fps", pose_summary.get("fps"))
            if isinstance(pose_result, Mapping)
            else pose_summary.get("fps")
        )
        try:
            sony_fps = _finite(fallback_fps, "sony_fps")
        except (TypeError, ValueError):
            sony_fps = None
    if sony_fps is None or sony_fps <= 0.0:
        raise ValueError("Sony FPS nije dostupan; uvoz ne može da potvrdi kadriranje")
    profile = _analysis_profile(sony_fps, analysis_fps)
    pose_summary = {
        "fps": profile["effective_analysis_fps"],
        "source_fps": profile["source_fps"],
        "requested_analysis_fps": profile["requested_analysis_fps"],
        "effective_analysis_fps": profile["effective_analysis_fps"],
        "stride": profile["stride"],
        "analysis_start_s": analysis_start_s,
        "analysis_limitation": profile["analysis_limitation"],
        "selected_track_id": pose_summary.get("selected_track_id"),
        "athlete_seen": pose_summary.get("athlete_seen", False),
    }
    sony_record["fps"] = sony_fps
    iphone_record["fps"] = iphone_fps
    events = []
    for index, raw_event in enumerate(pose_events, start=1):
        if hasattr(raw_event, "to_dict"):
            raw_event = raw_event.to_dict()
        if isinstance(raw_event, Mapping) and not raw_event.get("event_id"):
            raw_event = dict(raw_event)
            raw_event["event_id"] = f"e-{index:03d}"
        event = _event_payload(raw_event, injury_cutoff_s, time_map)
        if event is not None:
            events.append(event)
    if not any(event.get("prijavljen_povredni_dogadjaj") for event in events):
        injury_window = _injury_trace_window(injury_cutoff_s, sony_duration_s)
        if injury_window is not None:
            injury_start, injury_end = injury_window
            events.append(
                {
                    "event_id": "povreda",
                    "status": "povreda",
                    "sony_start_s": injury_start,
                    "sony_end_s": injury_end,
                    "iphone_start_s": (
                        injury_start - time_map.intercept
                    ) / time_map.slope,
                    "iphone_end_s": (injury_end - time_map.intercept) / time_map.slope,
                    "predlog_tehnike": None,
                    "potvrdena_tehnika": None,
                    "glasovna_fraza": None,
                    "pouzdanost_glasa": 0.0,
                    "prijavljen_povredni_dogadjaj": True,
                    "iskljuceno_iz_statistike": True,
                }
            )
    events.sort(key=lambda item: (item["sony_start_s"], item["event_id"]))

    review_events = [
        ReviewEvent(
            event["event_id"],
            event["sony_start_s"],
            event["sony_end_s"],
            prijavljen_povredni_dogadjaj=event.get(
                "prijavljen_povredni_dogadjaj", False
            ),
            iskljuceno_iz_statistike=event["iskljuceno_iz_statistike"],
        )
        for event in events
    ]
    session = ReviewSession(
        session_id=output_dir.name,
        sony_video=str(sony),
        iphone_video=str(iphone),
        anchors=anchor_values,
        injury_cutoff_s=injury_cutoff_s,
        events=review_events,
    )
    validate_review_session(session, sony_duration_s, iphone_duration_s)

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("media", "events", "previews", "analysis"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    privacy_processor = privacy_processor or build_privacy_processor(
        model_path=model_path,
        device=device,
    )
    derived_media_manifest: list[dict[str, Any]] = []

    for index, anchor in enumerate(anchor_values, start=1):
        sony_window = _window_for_clip(
            anchor.sony_s - PREVIEW_RADIUS_S,
            anchor.sony_s + PREVIEW_RADIUS_S,
            sony_duration_s,
        )
        iphone_center = (anchor.sony_s - time_map.intercept) / time_map.slope
        iphone_radius = PREVIEW_RADIUS_S / time_map.slope
        iphone_window = _window_for_clip(
            iphone_center - iphone_radius,
            iphone_center + iphone_radius,
            iphone_duration_s,
        )
        if sony_window:
            relative = f"previews/anchor_{index:02d}_sony.mp4"
            report = _export_private_clip(
                sony,
                sony_window[0],
                sony_window[1],
                output_dir / relative,
                sony_duration_s,
                privacy_processor,
            )
            if report is not None:
                derived_media_manifest.append(
                    _manifest_row(report, relative, "anchor_preview")
                )
        if iphone_window:
            relative = f"previews/anchor_{index:02d}_iphone.mp4"
            report = _export_private_clip(
                iphone,
                iphone_window[0],
                iphone_window[1],
                output_dir / relative,
                iphone_duration_s,
                privacy_processor,
            )
            if report is not None:
                derived_media_manifest.append(
                    _manifest_row(report, relative, "anchor_preview")
                )

    for event in events:
        event_dir = output_dir / "events" / event["event_id"]
        sony_relative = f"events/{event['event_id']}/sony.mp4"
        sony_report = _export_private_clip(
            sony,
            event["sony_start_s"],
            event["sony_end_s"],
            event_dir / "sony.mp4",
            sony_duration_s,
            privacy_processor,
        )
        iphone_relative = f"events/{event['event_id']}/iphone.mp4"
        iphone_report = _export_private_clip(
            iphone,
            event["iphone_start_s"],
            event["iphone_end_s"],
            event_dir / "iphone.mp4",
            iphone_duration_s,
            privacy_processor,
        )
        if sony_report is not None:
            derived_media_manifest.append(
                _manifest_row(sony_report, sony_relative, "event_clip")
            )
        if iphone_report is not None:
            derived_media_manifest.append(
                _manifest_row(iphone_report, iphone_relative, "event_clip")
            )
        event["media"] = {
            "sony": f"/media/{sony_relative}",
            "iphone": f"/media/{iphone_relative}",
        }

    side_by_side = output_dir / "media" / "session_side_by_side.mp4"
    with tempfile.TemporaryDirectory(
        prefix=".raw-private-", dir=side_by_side.parent
    ) as raw:
        raw_side_by_side = Path(raw) / side_by_side.name
        make_side_by_side(
            sony,
            iphone,
            time_map.slope,
            time_map.intercept,
            injury_cutoff_s,
            raw_side_by_side,
        )
        side_report = privacy_processor(raw_side_by_side, side_by_side)
    if not isinstance(side_report, BlurReport) or not side_report.privacy_verified:
        reason = side_report.failure_reason if isinstance(side_report, BlurReport) else None
        raise ValueError(reason or "privatnost side-by-side videa nije potvrđena")
    verify_media_export(
        side_by_side,
        injury_cutoff_s,
        EXPORT_TOLERANCE_S,
        probe=probe_duration,
    )
    derived_media_manifest.append(
        _manifest_row(side_report, "session_side_by_side.mp4", "side_by_side")
    )

    if transcript_path is not None:
        words = load_whisper_json(Path(transcript_path))
        normal_events = [
            event for event in events
            if not event.get("iskljuceno_iz_statistike")
        ]
        suggestions = suggest_techniques(
            words,
            [
                (event["event_id"], event["sony_start_s"], event["sony_end_s"])
                for event in normal_events
            ],
        )
        for event in normal_events:
            suggestion = suggestions[event["event_id"]]
            event.update(json_safe(suggestion.to_review_fields()))
        transcript_evidence = [
            {"tekst": word.text, "pocetak_s": word.start_s, "kraj_s": word.end_s}
            for word in words
        ]
    else:
        transcript_evidence = []

    payload: dict[str, Any] = {
        "version": 2,
        "session_id": output_dir.name,
        "sony_video": str(sony),
        "iphone_video": str(iphone),
        "sources": {"sony": sony_record, "iphone": iphone_record},
        "sony_duration_s": sony_duration_s,
        "iphone_duration_s": iphone_duration_s,
        "sony_fps": sony_fps,
        "iphone_fps": iphone_fps,
        "source_fps": profile["source_fps"],
        "requested_analysis_fps": profile["requested_analysis_fps"],
        "effective_analysis_fps": profile["effective_analysis_fps"],
        "stride": profile["stride"],
        "analysis_start_s": analysis_start_s,
        "analysis_limitation": profile["analysis_limitation"],
        "anchors": [anchor.to_dict() for anchor in anchor_values],
        "time_map": time_map.to_dict(),
        "injury_cutoff_s": injury_cutoff_s,
        "blue_seed_sony": blue_seed,
        "pose_analysis": pose_summary,
        "metric_schema": canonical_metric_schema(profile["effective_analysis_fps"]),
        "frame_metrics": frame_metrics,
        "event_metrics": events,
        "events": events,
        "transkript": transcript_evidence,
        "sync_locked": True,
        "derived_media_manifest": derived_media_manifest,
        "status": "Uvoz zavrsen; normalna obrada zaustavljena na potvrdenom preseku povrede.",
    }
    if previous is not None and force_reimport:
        payload = _merge_trainer_annotations(payload, previous)

    validate_review_payload(payload)
    review_path = write_review_json(output_dir, payload)
    write_reports(review_path, payload)
    _write_analysis_json(
        output_dir / "analysis" / "frame_metrics.json",
        {
            "frame_metrics": frame_metrics,
            "pose_analysis": pose_summary,
            "metric_schema": payload["metric_schema"],
        },
    )
    _write_analysis_json(
        output_dir / "analysis" / "event_metrics.json",
        {"events": events},
    )
    _write_analysis_json(
        output_dir / "analysis" / "transcript.json",
        {"transkript": transcript_evidence},
    )
    _write_summary(
        output_dir,
        {
            "status": "Uvoz zavrsen",
            "session_id": output_dir.name,
            "broj_dogadjaja": len(events),
            "broj_potvrdjenih_ankera": len(anchor_values),
            "injury_cutoff_s": injury_cutoff_s,
            "media": str(side_by_side),
        },
    )
    return review_path


__all__ = [
    "import_session",
    "make_side_by_side",
    "run_pose_analysis",
    "verify_media_export",
    "write_review_json",
]
