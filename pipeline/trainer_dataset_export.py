"""Build clean trainer examples and the separate assessment audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


MEDIA_CAMERAS = ("sony", "iphone")
VIDEO_METRICS = {
    "entry_speed_norm": "brzina_ulaska_norm",
    "torso_rotation_2d_dps": "rotacija_trupa_2d_dps",
    "hip_height_change_norm": "promena_visine_kukova_norm",
    "stance_width_norm": "sirina_stava_norm",
    "movement_intensity_0_100": "intenzitet_pokreta_0_100",
}


def _assessment_phase(row: Mapping[str, Any], local_revision: int) -> str:
    expected = "pre_ai" if local_revision == 1 else "post_ai_korekcija"
    if row.get("faza") != expected:
        raise ValueError("faza procene ne odgovara lokalnoj reviziji")
    return "pre_ai" if local_revision == 1 else "post_ai_correction"


def _is_injury(event: Mapping[str, Any]) -> bool:
    return bool(
        event.get("prijavljen_povredni_dogadjaj")
        or event.get("iskljuceno_iz_statistike")
        or event.get("status") == "povreda"
    )


def _safe_event_path(event_id: object, camera: str) -> tuple[str, PurePosixPath]:
    if not isinstance(event_id, str) or not event_id:
        raise ValueError("event_id nije validan za medijsku putanju")
    relative = f"events/{event_id}/{camera}.mp4"
    path = PurePosixPath(relative)
    if (
        path.is_absolute()
        or "\\" in relative
        or any(part in {"", ".", ".."} for part in path.parts)
        or len(path.parts) != 3
    ):
        raise ValueError("medijska putanja nije bezbedna")
    return relative, path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _media_reference(
    event_id: object,
    camera: str,
    manifest: list[Mapping[str, Any]],
    bundle_root: Path,
) -> dict[str, str] | None:
    relative, relative_path = _safe_event_path(event_id, camera)
    rows = [row for row in manifest if row.get("relative_path") == relative]
    if len(rows) != 1:
        return None
    row = rows[0]
    if (
        row.get("media_type") != "event_clip"
        or row.get("privacy_verified") is not True
        or row.get("failure_reason") is not None
    ):
        return None
    root = bundle_root.resolve()
    file_path = (root / Path(*relative_path.parts)).resolve()
    if root not in file_path.parents or not file_path.is_file():
        raise ValueError("potvrđeni medijski fajl nedostaje")
    review_url = f"/media/{relative}"
    if (
        not review_url.startswith("/media/")
        or review_url.startswith("//")
        or any(character in review_url for character in ("\\", "?", "#"))
        or ".." in PurePosixPath(review_url).parts
    ):
        raise ValueError("medijski URL nije bezbedan")
    return {
        "bundle_relative_path": relative,
        "review_url": review_url,
        "sha256": _sha256(file_path),
    }


def _identity(row: Mapping[str, Any]) -> dict[str, str] | None:
    trainer = row.get("trainer_name")
    wrestler = row.get("wrestler_name")
    if not isinstance(trainer, str) or not trainer.strip():
        return None
    if not isinstance(wrestler, str) or not wrestler.strip():
        return None
    return {"trainer_name": trainer, "wrestler_name": wrestler}


def _is_complete(row: Mapping[str, Any], event: Mapping[str, Any]) -> bool:
    technique = row.get("potvrdena_tehnika")
    reasoning = row.get("razlog")
    score = row.get("ocena")
    citations = row.get("citirani_sony_trenuci_s")
    if (
        not isinstance(technique, str)
        or not technique.strip()
        or not isinstance(reasoning, str)
        or not reasoning.strip()
        or isinstance(score, bool)
        or not isinstance(score, int)
        or not 1 <= score <= 5
        or not isinstance(citations, list)
        or not citations
    ):
        return False
    start = event.get("sony_start_s")
    end = event.get("sony_end_s")
    if isinstance(start, bool) or isinstance(end, bool):
        return False
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        return False
    return all(
        not isinstance(second, bool)
        and isinstance(second, (int, float))
        and float(start) <= float(second) <= float(end)
        for second in citations
    )


def _audit_row(
    event: Mapping[str, Any],
    assessment: Mapping[str, Any],
    *,
    local_revision: int,
    phase: str,
    training_eligible: bool,
    reasons: list[str],
) -> dict[str, Any]:
    return {
        "event_id": event.get("event_id"),
        "event_revision": assessment.get("event_revision"),
        "analysis_fingerprint": assessment.get("analysis_fingerprint"),
        "assessment_revision": local_revision,
        "assessment_phase": phase,
        "source_trainer_revision": assessment.get("revizija"),
        "trainer_name": assessment.get("trainer_name"),
        "wrestler_name": assessment.get("wrestler_name"),
        "visibility_status": assessment.get("status_vidljivosti"),
        "throw_name": assessment.get("potvrdena_tehnika"),
        "score_1_5": assessment.get("ocena"),
        "reasoning": assessment.get("razlog"),
        "cited_sony_seconds": assessment.get("citirani_sony_trenuci_s"),
        "locked_at": assessment.get("zakljucano_u"),
        "training_eligible": training_eligible,
        "ineligibility_reasons": reasons,
    }


def _training_example(
    event: Mapping[str, Any],
    assessment: Mapping[str, Any],
    *,
    local_revision: int,
    generation_id: str,
    media: dict[str, dict[str, str]],
) -> dict[str, Any]:
    event_id = event["event_id"]
    event_revision = assessment["event_revision"]
    fingerprint = assessment["analysis_fingerprint"]
    return {
        "example_id": (
            f"{event.get('session_id', '')}:{event_id}:{event_revision}:"
            f"{fingerprint.replace(':', '-')}:{local_revision}"
        ),
        "generation_id": generation_id,
        "trainer_name": assessment["trainer_name"],
        "wrestler_name": assessment["wrestler_name"],
        "throw_name": assessment["potvrdena_tehnika"],
        "score_1_5": assessment["ocena"],
        "reasoning": assessment["razlog"],
        "assessment_revision": local_revision,
        "assessment_phase": "pre_ai",
        "source_trainer_revision": assessment["revizija"],
        "event_id": event_id,
        "event_revision": event_revision,
        "analysis_fingerprint": fingerprint,
        "evidence": {
            "cited_sony_seconds": assessment["citirani_sony_trenuci_s"],
            "sony_bounds_s": [event.get("sony_start_s"), event.get("sony_end_s")],
            "iphone_bounds_s": [event.get("iphone_start_s"), event.get("iphone_end_s")],
            "sony_clip": media["sony"],
            "iphone_clip": media["iphone"],
        },
        "video_metrics": {
            output_key: event.get(input_key)
            for output_key, input_key in VIDEO_METRICS.items()
        },
        "locked_at": assessment["zakljucano_u"],
        "training_eligible": True,
    }


def build_trainer_exports(
    review: Mapping[str, Any],
    *,
    generation_id: str,
    bundle_root: Path,
    generated_at: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Project a validated review without consulting AI state fields."""
    manifest = [
        row
        for row in review.get("derived_media_manifest", [])
        if isinstance(row, Mapping)
    ]
    training_examples: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    session_id = review.get("session_id")
    for event in sorted(
        (item for item in review.get("events", []) if isinstance(item, Mapping)),
        key=lambda item: str(item.get("event_id", "")),
    ):
        if _is_injury(event):
            continue
        event_id = event.get("event_id")
        assessments = [
            row for row in event.get("trener_procene", []) if isinstance(row, Mapping)
        ]
        grouped: dict[tuple[Any, Any], list[Mapping[str, Any]]] = {}
        for assessment in assessments:
            key = (assessment.get("event_revision"), assessment.get("analysis_fingerprint"))
            grouped.setdefault(key, []).append(assessment)
        projected: list[tuple[Mapping[str, Any], int, str]] = []
        for rows in grouped.values():
            for position, assessment in enumerate(
                sorted(rows, key=lambda item: item.get("revizija")), start=1
            ):
                projected.append((assessment, position, _assessment_phase(assessment, position)))
        projected.sort(key=lambda item: item[0].get("revizija"))
        for assessment, local_revision, phase in projected:
            active_round = (
                assessment.get("event_revision") == event.get("event_revision")
                and assessment.get("analysis_fingerprint") == event.get("analysis_fingerprint")
            )
            identity = _identity(assessment)
            complete = _is_complete(assessment, event)
            media = {
                camera: _media_reference(event_id, camera, manifest, Path(bundle_root))
                for camera in MEDIA_CAMERAS
            }
            reasons: list[str] = []
            if not active_round:
                reasons.append("inactive_analysis_round")
            if phase != "pre_ai":
                reasons.append("post_ai_correction")
            if assessment.get("status_vidljivosti") != "dovoljno_vidljivo":
                reasons.append("insufficient_visibility")
            if identity is None:
                reasons.append("missing_identity_snapshot")
            if not complete:
                reasons.append("incomplete_assessment")
            if any(reference is None for reference in media.values()):
                reasons.append("missing_verified_media")
            eligible = not reasons
            audit_rows.append(
                _audit_row(
                    event,
                    assessment,
                    local_revision=local_revision,
                    phase=phase,
                    training_eligible=eligible,
                    reasons=reasons,
                )
            )
            if eligible:
                example_event = {**event, "session_id": session_id}
                training_examples.append(
                    _training_example(
                        example_event,
                        assessment,
                        local_revision=local_revision,
                        generation_id=generation_id,
                        media={camera: reference for camera, reference in media.items() if reference},
                    )
                )
    participants = review.get("participants")
    dataset: dict[str, Any] = {
        "schema_version": 1,
        "session_id": session_id,
        "generation_id": generation_id,
        "generated_at": generated_at,
        "participants": (
            {
                "trainer_name": participants.get("trainer_name"),
                "wrestler_name": participants.get("wrestler_name"),
            }
            if isinstance(participants, Mapping)
            else None
        ),
        "training_examples": training_examples,
    }
    audit = {
        "schema_version": 1,
        "session_id": session_id,
        "generation_id": generation_id,
        "generated_at": generated_at,
        "assessments": audit_rows,
    }
    return dataset, audit


def render_trainer_exports(
    review: Mapping[str, Any],
    *,
    generation_id: str,
    bundle_root: Path,
    generated_at: str,
) -> tuple[str, str]:
    dataset, audit = build_trainer_exports(
        review,
        generation_id=generation_id,
        bundle_root=bundle_root,
        generated_at=generated_at,
    )
    return (
        json.dumps(dataset, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
    )
