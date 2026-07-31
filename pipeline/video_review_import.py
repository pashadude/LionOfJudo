"""Local Sony/iPhone session import and synchronized media exports."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pipeline.clip_extractor import cut_clip, probe_duration
from pipeline.video_pose_metrics import json_safe
from pipeline.video_review_contract import (
    AnchorPair,
    ReviewEvent,
    ReviewSession,
    validate_review_session,
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


def verify_media_export(
    video: Path,
    expected_duration_s: float | None = None,
    tolerance_s: float = EXPORT_TOLERANCE_S,
) -> float:
    """Verify an importer export using this module's probe boundary."""
    video = Path(video)
    if not video.is_file() or video.stat().st_size == 0:
        raise ValueError(f"izvoz medija je prazan: {video}")
    duration = _finite(probe_duration(video), "duration_s")
    if duration < 0.0:
        raise ValueError(f"trajanje izvoza nije nenegativno: {video}")
    if expected_duration_s is not None:
        expected = _finite(expected_duration_s, "expected_duration_s")
        tolerance = _finite(tolerance_s, "tolerance_s")
        if expected <= 0.0 or tolerance < 0.0:
            raise ValueError("provera izvoza zahteva valjano trajanje i toleranciju")
        if abs(duration - expected) > tolerance:
            raise ValueError(
                f"trajanje izvoza odstupa od prozora: {duration:.3f}s prema "
                f"{expected:.3f}s"
            )
    return duration


def _finite(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} mora biti JSON broj")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} mora biti konacan")
    return result


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
    # Real ffmpeg creates the output; the existence guard keeps command-only
    # callers and mocked subprocess tests independent of ffprobe.
    if output.exists():
        verify_media_export(output, end_s, EXPORT_TOLERANCE_S)
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


def run_pose_analysis(
    sony: Path,
    start_s: float,
    end_s: float,
    blue_seed: Sequence[float],
) -> list[Mapping[str, Any]]:
    """Hook for Task 5 pose inference, restricted to the usable Sony range.

    Task 4 keeps inference optional so importing a session remains local and
    deterministic on machines without a pose-model runtime.  The review app
    can replace this boundary with the Task 3 metric runner.
    """
    del sony, start_s, end_s, blue_seed
    return []


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


def _source_record(path: Path) -> dict[str, str]:
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
    for event_id, old in previous_events.items():
        if event_id not in current_ids and any(
            old.get(key) not in (None, "", {}, []) for key in ANNOTATION_KEYS
        ):
            payload.setdefault("events", []).append(json_safe(dict(old)))
    return payload


def _window_for_clip(
    start_s: float,
    end_s: float,
    duration_s: float,
) -> tuple[float, float] | None:
    start = max(0.0, float(start_s))
    end = min(float(end_s), duration_s)
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
    verify_media_export(result, end - start, EXPORT_TOLERANCE_S)
    return result


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
) -> Path:
    """Import one Sony-master session and return its canonical review path."""
    sony = Path(sony).expanduser().resolve()
    iphone = Path(iphone).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
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
    sony_duration_s = _finite(probe_duration(sony), "sony_duration_s")
    iphone_duration_s = _finite(probe_duration(iphone), "iphone_duration_s")
    if sony_duration_s <= 0.0 or iphone_duration_s <= 0.0:
        raise ValueError("trajanje izvornog videa mora biti pozitivno")
    if injury_cutoff_s > sony_duration_s:
        raise ValueError("injury cutoff je posle kraja Sony videa")

    raw_events = run_pose_analysis(sony, 0.0, injury_cutoff_s, blue_seed)
    events = []
    for raw_event in raw_events:
        event = _event_payload(raw_event, injury_cutoff_s, time_map)
        if event is not None:
            events.append(event)
    if not any(event.get("prijavljen_povredni_dogadjaj") for event in events):
        injury_end = min(sony_duration_s, injury_cutoff_s + 1.0)
        if injury_end > injury_cutoff_s:
            events.append(
                {
                    "event_id": "povreda",
                    "status": "povreda",
                    "sony_start_s": injury_cutoff_s,
                    "sony_end_s": injury_end,
                    "iphone_start_s": (
                        injury_cutoff_s - time_map.intercept
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
            _export_clip(
                sony,
                sony_window[0],
                sony_window[1],
                output_dir / "previews" / f"anchor_{index:02d}_sony.mp4",
                sony_duration_s,
            )
        if iphone_window:
            _export_clip(
                iphone,
                iphone_window[0],
                iphone_window[1],
                output_dir / "previews" / f"anchor_{index:02d}_iphone.mp4",
                iphone_duration_s,
            )

    for event in events:
        event_dir = output_dir / "events" / event["event_id"]
        _export_clip(
            sony,
            event["sony_start_s"],
            event["sony_end_s"],
            event_dir / "sony.mp4",
            sony_duration_s,
        )
        _export_clip(
            iphone,
            event["iphone_start_s"],
            event["iphone_end_s"],
            event_dir / "iphone.mp4",
            iphone_duration_s,
        )

    side_by_side = output_dir / "media" / "session_side_by_side.mp4"
    make_side_by_side(
        sony,
        iphone,
        time_map.slope,
        time_map.intercept,
        injury_cutoff_s,
        side_by_side,
    )
    verify_media_export(side_by_side, injury_cutoff_s, EXPORT_TOLERANCE_S)

    if transcript_path is not None:
        words = load_whisper_json(Path(transcript_path))
        suggestions = suggest_techniques(
            words,
            [
                (event["event_id"], event["sony_start_s"], event["sony_end_s"])
                for event in events
            ],
        )
        for event in events:
            if event.get("iskljuceno_iz_statistike"):
                continue
            suggestion = suggestions[event["event_id"]]
            event.update(json_safe(suggestion.__dict__))

    payload: dict[str, Any] = {
        "version": 1,
        "session_id": output_dir.name,
        "sony_video": str(sony),
        "iphone_video": str(iphone),
        "sources": {"sony": sony_record, "iphone": iphone_record},
        "sony_duration_s": sony_duration_s,
        "iphone_duration_s": iphone_duration_s,
        "anchors": [anchor.to_dict() for anchor in anchor_values],
        "time_map": time_map.to_dict(),
        "injury_cutoff_s": injury_cutoff_s,
        "blue_seed_sony": blue_seed,
        "events": events,
        "status": "Uvoz zavrsen; normalna obrada zaustavljena na potvrdenom preseku povrede.",
    }
    if previous is not None and force_reimport:
        payload = _merge_trainer_annotations(payload, previous)

    review_path = write_review_json(output_dir, payload)
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
