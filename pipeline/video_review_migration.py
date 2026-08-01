"""Lossless migration of imported review sessions to the canonical metric schema."""

from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Mapping
import uuid

from pipeline.clip_extractor import probe_duration, verify_media_export
from pipeline.face_blur import BlurReport, build_privacy_processor
from pipeline.trainer_ai_state import migrate_trainer_ai_payload
from pipeline.video_review_contract import validate_review_payload
from pipeline.video_review_metrics import (
    canonical_metric_schema,
    canonicalize_frame_metrics,
    summarize_event_metrics,
)
from pipeline.video_review_reports import event_is_injury, write_reports
from pipeline.video_review_import import (
    EXPORT_TOLERANCE_S,
    PREVIEW_RADIUS_S,
    _export_private_clip,
    _window_for_clip,
    make_side_by_side,
)
from pipeline.video_review_storage import (
    atomic_write_json,
    atomic_write_review,
    load_review_json,
)
from pipeline.video_review_sync import iphone_media_bounds


def migrate_review_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    migrated = copy.deepcopy(dict(payload))
    version = migrated.get("version", 1)
    if isinstance(version, int) and not isinstance(version, bool) and version >= 3:
        validate_review_payload(migrated)
        return migrated

    raw_frames = migrated.get("frame_metrics", [])
    if not isinstance(raw_frames, list) or not all(isinstance(item, Mapping) for item in raw_frames):
        raise ValueError("frame_metrics mora biti lista JSON objekata")
    frames = canonicalize_frame_metrics(
        raw_frames,
        trust_precomputed_acceleration=False,
    )
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
        event.setdefault("iphone_sync_offset_s", 0.0)
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
            if isinstance(event.get("potvrdena_tehnika"), str) and event["potvrdena_tehnika"].strip():
                event["status"] = "trener"
            event.update(summarize_event_metrics(event, frames))
    events.sort(key=lambda item: (float(item["sony_start_s"]), str(item["event_id"])))
    migrated["event_metrics"] = copy.deepcopy(events)
    migrated = migrate_trainer_ai_payload(migrated)
    migrated["event_metrics"] = copy.deepcopy(migrated["events"])
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_source_signature(path: Path) -> tuple[str, int]:
    before = path.stat()
    digest = _sha256_file(path)
    after = path.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise ValueError(f"source je promenjen tokom hashiranja: {path}")
    return digest, after.st_size


def _prepare_ai_review_payload(
    payload: Mapping[str, Any], session_id: str
) -> dict[str, Any]:
    prepared = copy.deepcopy(dict(payload))
    prepared["version"] = 2
    prepared["session_id"] = session_id
    prepared["injury_cutoff_s"] = 135.0
    prepared.pop("derived_media_manifest", None)
    prepared.pop("session_ready", None)
    prepared["events"] = [
        {
            "event_id": "e-001",
            "sony_start_s": 128.5,
            "sony_end_s": 132.0,
            "iphone_start_s": 131.5,
            "iphone_end_s": 135.0,
            "iphone_sync_offset_s": 0.8,
            "predlog_tehnike": None,
            "potvrdena_tehnika": "Tai-otoshi",
            "glasovna_fraza": None,
            "pouzdanost_glasa": 0.0,
            "ocena": None,
            "napomena": "Naziv je potvrđen; ocena i razlog čekaju trenera.",
            "iskljuceno_iz_statistike": False,
            "status": "trener",
            "media": {
                "sony": "/media/events/e-001/sony.mp4",
                "iphone": "/media/events/e-001/iphone.mp4",
            },
        },
        {
            "event_id": "e-coach-001",
            "sony_start_s": 132.8,
            "sony_end_s": 135.0,
            "iphone_start_s": 135.8,
            "iphone_end_s": 138.0,
            "iphone_sync_offset_s": 0.0,
            "predlog_tehnike": None,
            "potvrdena_tehnika": "Morote-seoi-nage",
            "glasovna_fraza": None,
            "pouzdanost_glasa": 0.0,
            "ocena": None,
            "napomena": "Naziv je potvrđen; ocena i razlog čekaju trenera.",
            "iskljuceno_iz_statistike": False,
            "status": "trener",
            "media": {
                "sony": "/media/events/e-coach-001/sony.mp4",
                "iphone": "/media/events/e-coach-001/iphone.mp4",
            },
        },
        {
            "event_id": "povreda",
            "sony_start_s": 135.0,
            "sony_end_s": 136.0,
            "iphone_start_s": 138.0,
            "iphone_end_s": 139.0,
            "iphone_sync_offset_s": 0.0,
            "predlog_tehnike": None,
            "potvrdena_tehnika": None,
            "glasovna_fraza": None,
            "pouzdanost_glasa": 0.0,
            "prijavljen_povredni_dogadjaj": True,
            "iskljuceno_iz_statistike": True,
            "status": "povreda",
            "media": {
                "sony": "/media/events/povreda/sony.mp4",
                "iphone": "/media/events/povreda/iphone.mp4",
            },
        },
    ]
    prepared["event_metrics"] = copy.deepcopy(prepared["events"])
    migrated = migrate_review_payload(prepared)
    confirmed_anchor_times = [
        float(anchor["sony_s"])
        for anchor in migrated.get("anchors", [])
        if isinstance(anchor, Mapping) and anchor.get("user_confirmed") is True
    ]
    if not confirmed_anchor_times:
        raise ValueError("AI migracija zahteva potvrđen početni anchor")
    migrated["side_by_side_start_s"] = min(confirmed_anchor_times)
    migrated["side_by_side_end_s"] = max(
        float(event["sony_end_s"])
        for event in migrated["events"]
        if isinstance(event, Mapping)
    )
    migrated["derived_media_manifest"] = []
    migrated["session_ready"] = False
    migrated["event_metrics"] = copy.deepcopy(migrated["events"])
    validate_review_payload(migrated)
    return migrated


def _expected_private_media(review: Mapping[str, Any]) -> set[str]:
    expected = {"session_side_by_side.mp4"}
    anchors = review.get("anchors", [])
    events = review.get("events", [])
    if not isinstance(anchors, list) or not isinstance(events, list):
        raise ValueError("review nema validne anchors/events")
    for index, _anchor in enumerate(anchors, start=1):
        expected.add(f"previews/anchor_{index:02d}_sony.mp4")
        expected.add(f"previews/anchor_{index:02d}_iphone.mp4")
    for event in events:
        if not isinstance(event, Mapping) or not isinstance(event.get("event_id"), str):
            raise ValueError("review događaj nema validan event_id")
        for camera in ("sony", "iphone"):
            expected.add(f"events/{event['event_id']}/{camera}.mp4")
    return expected


def _regenerate_private_media(
    review: Mapping[str, Any],
    stage: Path,
    privacy_processor: Callable[[Path, Path], BlurReport],
) -> list[dict[str, Any]]:
    stage = Path(stage)
    for name in ("media", "events", "previews", "analysis"):
        (stage / name).mkdir(parents=True, exist_ok=True)
    sony = Path(str(review["sources"]["sony"]["path"])).expanduser().resolve()
    iphone = Path(str(review["sources"]["iphone"]["path"])).expanduser().resolve()
    sony_duration = float(review["sony_duration_s"])
    iphone_duration = float(review["iphone_duration_s"])
    slope = float(review["time_map"]["slope"])
    intercept = float(review["time_map"]["intercept"])
    manifest: list[dict[str, Any]] = []

    for index, anchor in enumerate(review["anchors"], start=1):
        sony_center = float(anchor["sony_s"])
        iphone_center = float(anchor["iphone_s"])
        windows = {
            "sony": (
                sony,
                _window_for_clip(
                    sony_center - PREVIEW_RADIUS_S,
                    sony_center + PREVIEW_RADIUS_S,
                    sony_duration,
                ),
                sony_duration,
            ),
            "iphone": (
                iphone,
                _window_for_clip(
                    iphone_center - PREVIEW_RADIUS_S / slope,
                    iphone_center + PREVIEW_RADIUS_S / slope,
                    iphone_duration,
                ),
                iphone_duration,
            ),
        }
        for camera, (source, window, duration) in windows.items():
            if window is None:
                raise ValueError(f"anchor {index} {camera} preview je van izvora")
            relative = f"previews/anchor_{index:02d}_{camera}.mp4"
            report = _export_private_clip(
                source,
                window[0],
                window[1],
                stage / relative,
                duration,
                privacy_processor,
            )
            if report is None or not report.privacy_verified:
                raise ValueError(f"privacy preview nije potvrđen: {relative}")
            manifest.append(report.to_manifest(relative, "anchor_preview"))

    for event in review["events"]:
        event_id = str(event["event_id"])
        iphone_start, iphone_end = iphone_media_bounds(event)
        for camera, source, duration, start, end in (
            (
                "sony",
                sony,
                sony_duration,
                float(event["sony_start_s"]),
                float(event["sony_end_s"]),
            ),
            ("iphone", iphone, iphone_duration, iphone_start, iphone_end),
        ):
            relative = f"events/{event_id}/{camera}.mp4"
            report = _export_private_clip(
                source,
                start,
                end,
                stage / relative,
                duration,
                privacy_processor,
            )
            if report is None or not report.privacy_verified:
                raise ValueError(f"privacy event klip nije potvrđen: {relative}")
            manifest.append(report.to_manifest(relative, "event_clip"))

    final_side = stage / "media" / "session_side_by_side.mp4"
    side_start_s = float(review["side_by_side_start_s"])
    side_end_s = float(review["side_by_side_end_s"])
    with tempfile.TemporaryDirectory(prefix=".raw-private-", dir=stage / "media") as raw:
        raw_side = Path(raw) / "session_side_by_side.mp4"
        make_side_by_side(
            sony,
            iphone,
            slope,
            intercept,
            side_end_s,
            raw_side,
            start_s=side_start_s,
        )
        side_report = privacy_processor(raw_side, final_side)
    if not isinstance(side_report, BlurReport) or not side_report.privacy_verified:
        reason = side_report.failure_reason if isinstance(side_report, BlurReport) else None
        raise ValueError(reason or "privacy side-by-side videa nije potvrđena")
    verify_media_export(
        final_side,
        side_end_s - side_start_s,
        EXPORT_TOLERANCE_S,
        probe=probe_duration,
    )
    side_manifest = side_report.to_manifest(
        "session_side_by_side.mp4", "side_by_side"
    )
    side_manifest["timeline_start_s"] = side_start_s
    side_manifest["timeline_end_s"] = side_end_s
    manifest.append(side_manifest)
    return manifest


def _write_ai_session_bundle(stage: Path, review: dict[str, Any]) -> None:
    analysis = Path(stage) / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    review_path = Path(stage) / "review.json"
    atomic_write_review(review_path, review)
    atomic_write_json(
        analysis / "frame_metrics.json",
        {
            "frame_metrics": review["frame_metrics"],
            "pose_analysis": review.get("pose_analysis", {}),
            "metric_schema": review["metric_schema"],
        },
    )
    atomic_write_json(
        analysis / "event_metrics.json",
        {"events": review["event_metrics"]},
    )
    if "transkript" in review:
        atomic_write_json(
            analysis / "transcript.json",
            {"transkript": review.get("transkript", [])},
        )
    write_reports(review_path, review)


def _activate_staged_directory(
    stage: Path, output_dir: Path, *, replace_derived: bool
) -> None:
    token = uuid.uuid4().hex
    generation = output_dir.with_name(
        f".{output_dir.name}.generation-{token}"
    )
    switch_link = output_dir.with_name(f".{output_dir.name}.switch-{token}")
    published = False
    try:
        if os.path.lexists(output_dir):
            if not output_dir.is_symlink():
                raise FileExistsError(
                    "postojeći output nije atomski versioniran; koristite novi output"
                )
            if not replace_derived:
                raise FileExistsError(
                    "output postoji; koristite --replace-derived"
                )
        os.replace(stage, generation)
        os.symlink(generation.name, switch_link)
        os.replace(switch_link, output_dir)
        published = True
    finally:
        switch_link.unlink(missing_ok=True)
        if not published and generation.exists():
            shutil.rmtree(generation, ignore_errors=True)


def migrate_ai_session(
    session_dir: Path,
    output_dir: Path,
    *,
    model_path: Path,
    device: str,
    replace_derived: bool = False,
    privacy_processor: Callable[[Path, Path], BlurReport] | None = None,
    media_regenerator: Callable[
        [Mapping[str, Any], Path, Callable[[Path, Path], BlurReport]],
        list[dict[str, Any]],
    ] | None = None,
) -> Path:
    """Build a separate v3 session after validating source immutability."""
    session_dir = Path(session_dir).expanduser().resolve()
    output_dir = Path(os.path.abspath(Path(output_dir).expanduser()))
    if session_dir == output_dir or (
        os.path.lexists(output_dir) and output_dir.resolve() == session_dir
    ):
        raise ValueError("source i output direktorijum moraju biti različiti")
    if os.path.lexists(output_dir):
        if output_dir.is_symlink():
            if not replace_derived:
                raise FileExistsError(
                    "output postoji; koristite --replace-derived"
                )
            if not output_dir.exists() or not output_dir.resolve().is_dir():
                raise FileExistsError("postojeći output symlink nije validan")
        elif output_dir.is_dir() and not any(output_dir.iterdir()):
            output_dir.rmdir()
        elif not replace_derived:
            raise FileExistsError(
                "output direktorijum nije prazan; koristite --replace-derived"
            )
        else:
            raise FileExistsError(
                "postojeći output nije atomski versioniran; koristite novi output"
            )
    payload = load_review_json(session_dir / "review.json")
    sources = payload.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("review nema validne source hash zapise")
    source_signatures: dict[str, tuple[Path, str, int]] = {}
    for camera in ("sony", "iphone"):
        record = sources.get(camera)
        if not isinstance(record, Mapping):
            raise ValueError(f"review nema {camera} source hash")
        path_value = record.get("path")
        expected = record.get("sha256")
        if not isinstance(path_value, str) or not isinstance(expected, str):
            raise ValueError(f"review nema validan {camera} source hash")
        source_path = Path(path_value).expanduser().resolve()
        if not source_path.is_file():
            raise ValueError(f"{camera} source hash se ne poklapa")
        observed_hash, observed_size = _stable_source_signature(source_path)
        if observed_hash != expected:
            raise ValueError(f"{camera} source hash se ne poklapa")
        source_signatures[camera] = (
            source_path,
            expected,
            observed_size,
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.stage-",
            dir=output_dir.parent,
        )
    )
    try:
        prepared = _prepare_ai_review_payload(payload, output_dir.name)
        processor = privacy_processor or build_privacy_processor(
            model_path=model_path,
            device=device,
        )
        regenerate = media_regenerator or _regenerate_private_media
        manifest = regenerate(prepared, stage, processor)
        expected = _expected_private_media(prepared)
        actual = {
            row.get("relative_path")
            for row in manifest
            if isinstance(row, Mapping) and row.get("privacy_verified") is True
        }
        if actual != expected or len(manifest) != len(expected):
            raise ValueError("verified manifest nije kompletan")
        prepared["derived_media_manifest"] = sorted(
            copy.deepcopy(manifest),
            key=lambda row: str(row.get("relative_path", "")),
        )
        prepared["session_ready"] = True
        prepared["event_metrics"] = copy.deepcopy(prepared["events"])
        validate_review_payload(prepared)
        _write_ai_session_bundle(stage, prepared)
        for camera, (source_path, source_hash, source_size) in source_signatures.items():
            final_hash, final_size = _stable_source_signature(source_path)
            if final_hash != source_hash or final_size != source_size:
                raise ValueError(f"{camera} source je promenjen tokom migracije")
        _activate_staged_directory(
            stage,
            output_dir,
            replace_derived=replace_derived,
        )
        stage = None
        return output_dir / "review.json"
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)


__all__ = ["migrate_ai_session", "migrate_review_payload", "migrate_session"]
