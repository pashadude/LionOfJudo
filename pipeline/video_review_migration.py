"""Lossless migration of imported review sessions to the canonical metric schema."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

from pipeline.video_review_contract import validate_review_payload
from pipeline.video_review_metrics import (
    canonical_metric_schema,
    canonicalize_frame_metrics,
    summarize_event_metrics,
)
from pipeline.video_review_reports import event_is_injury, write_reports
from pipeline.video_review_storage import (
    atomic_write_json,
    atomic_write_review,
    load_review_json,
)


def migrate_review_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    migrated = copy.deepcopy(dict(payload))
    raw_frames = migrated.get("frame_metrics", [])
    if not isinstance(raw_frames, list) or not all(isinstance(item, Mapping) for item in raw_frames):
        raise ValueError("frame_metrics mora biti lista JSON objekata")
    frames = canonicalize_frame_metrics(raw_frames)
    migrated["frame_metrics"] = frames
    migrated["version"] = max(2, int(migrated.get("version", 1)))
    migrated["sync_locked"] = bool(
        migrated.get("events")
        or migrated.get("event_metrics")
        or frames
        or migrated.get("pose_analysis")
    )
    effective_fps = migrated.get("effective_analysis_fps")
    if not isinstance(effective_fps, (int, float)) or isinstance(effective_fps, bool):
        pose_analysis = migrated.get("pose_analysis", {})
        effective_fps = (
            pose_analysis.get("effective_analysis_fps", pose_analysis.get("fps"))
            if isinstance(pose_analysis, Mapping)
            else None
        )
    migrated["metric_schema"] = canonical_metric_schema(effective_fps)

    events = migrated.get("events")
    if not isinstance(events, list):
        raise ValueError("events mora biti JSON lista")
    slope = float(migrated["time_map"]["slope"])
    intercept = float(migrated["time_map"]["intercept"])
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("events mora sadržati JSON objekte")
        if "source_phrase" in event and not event.get("glasovna_fraza"):
            event["glasovna_fraza"] = event.get("source_phrase")
        if "confidence" in event and event.get("pouzdanost_glasa") in (None, 0, 0.0):
            event["pouzdanost_glasa"] = event.get("confidence")
        event.pop("source_phrase", None)
        event.pop("confidence", None)
        event["iphone_start_s"] = (float(event["sony_start_s"]) - intercept) / slope
        event["iphone_end_s"] = (float(event["sony_end_s"]) - intercept) / slope
        if event_is_injury(event):
            event["predlog_tehnike"] = None
            event["glasovna_fraza"] = None
            event["pouzdanost_glasa"] = 0.0
            event.pop("glasovna_fraza_pocetak_s", None)
            event.pop("glasovna_fraza_kraj_s", None)
            for key in (
                "brzina_ulaska_norm",
                "rotacija_trupa_2d_dps",
                "promena_visine_kukova_norm",
                "sirina_stava_norm",
                "vreme_oporavka_s",
                "intenzitet_pokreta_0_100",
            ):
                event.pop(key, None)
        else:
            event.update(summarize_event_metrics(event, frames))
    events.sort(key=lambda item: (float(item["sony_start_s"]), str(item["event_id"])))
    migrated["event_metrics"] = copy.deepcopy(events)
    validate_review_payload(migrated)
    return migrated


def migrate_session(session_dir: Path) -> Path:
    session_dir = Path(session_dir).expanduser().resolve()
    review_path = session_dir / "review.json"
    migrated = migrate_review_payload(load_review_json(review_path))
    analysis_dir = session_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_review(review_path, migrated)
    atomic_write_json(
        analysis_dir / "frame_metrics.json",
        {
            "frame_metrics": migrated["frame_metrics"],
            "pose_analysis": migrated.get("pose_analysis", {}),
            "metric_schema": migrated["metric_schema"],
        },
    )
    atomic_write_json(
        analysis_dir / "event_metrics.json",
        {"events": migrated["event_metrics"]},
    )
    if "transkript" in migrated:
        atomic_write_json(
            analysis_dir / "transcript.json",
            {"transkript": migrated.get("transkript", [])},
        )
    write_reports(review_path, migrated)
    return review_path


__all__ = ["migrate_review_payload", "migrate_session"]
