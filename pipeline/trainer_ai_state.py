"""Versioned trainer-versus-AI event state and validation."""

from __future__ import annotations

import copy
from datetime import datetime
import math
import re
from typing import Any, Mapping

from pipeline.trainer_ai_evaluator import (
    EVALUATOR_ID,
    compute_analysis_fingerprint,
    evaluate_event,
)


FINGERPRINT_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
ISO_TIME_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})$"
)
TRAINER_PHASES = {"pre_ai", "post_ai_korekcija"}
VISIBILITY_STATES = {"dovoljno_vidljivo", "nedovoljno_vidljivo"}
AI_RELATIONS = {"slazem_se", "delimicno", "ne_slazem_se"}
EVIDENCE_RELATIONS = {"prihvatam", "nepotpun", "osporavam"}
STATE_KEYS = {
    "event_revision",
    "analysis_fingerprint",
    "ai_procene",
    "imu_eksperimentalno",
    "trener_procene",
    "aktivna_trener_revizija",
    "procene_ai_predloga",
    "aktivni_duel",
}


def _is_injury(event: Mapping[str, Any]) -> bool:
    return bool(
        event.get("prijavljen_povredni_dogadjaj")
        or event.get("iskljuceno_iz_statistike")
        or event.get("status") == "povreda"
    )


def _fingerprint(value: Any, field: str) -> str:
    if not isinstance(value, str) or FINGERPRINT_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} fingerprint nije validan")
    return value


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} mora biti pozitivan ceo broj")
    if value <= 0:
        raise ValueError(f"{field} mora biti pozitivan ceo broj")
    return value


def _optional_score(value: Any, field: str) -> int | None:
    if value is None:
        return None
    score = _positive_int(value, field)
    if score > 5:
        raise ValueError(f"{field} mora biti u opsegu 1..5")
    return score


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} mora biti konačan broj")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{field} mora biti konačan broj")
    return numeric


def _iso_time(value: Any, field: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or ISO_TIME_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} mora biti strogo ISO-8601 vreme")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} mora biti ISO-8601 vreme") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} mora sadržati vremensku zonu")
    return value


def _list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{field} mora biti JSON lista")
    return value


def active_ai_evaluation(event: Mapping[str, Any]) -> dict[str, Any] | None:
    if _is_injury(event):
        return None
    revision = event.get("event_revision")
    fingerprint = event.get("analysis_fingerprint")
    matches = [
        item
        for item in event.get("ai_procene", [])
        if isinstance(item, dict)
        and item.get("event_revision") == revision
        and item.get("analysis_fingerprint") == fingerprint
        and item.get("evaluator_id") == EVALUATOR_ID
    ]
    if len(matches) > 1:
        raise ValueError("aktivna AI procena nije jedinstvena")
    return matches[0] if matches else None


def active_trainer_assessment(event: Mapping[str, Any]) -> dict[str, Any] | None:
    revision = event.get("aktivna_trener_revizija")
    if revision is None:
        return None
    matches = [
        item
        for item in event.get("trener_procene", [])
        if isinstance(item, dict) and item.get("revizija") == revision
    ]
    if len(matches) != 1:
        raise ValueError("aktivna trener revizija nije jedinstvena")
    active = matches[0]
    if (
        active.get("event_revision") != event.get("event_revision")
        or active.get("analysis_fingerprint") != event.get("analysis_fingerprint")
    ):
        raise ValueError("aktivna trener procena pripada drugoj rundi")
    return active


def _validate_ai_evaluations(event: Mapping[str, Any]) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    evaluations = _list(event.get("ai_procene"), "ai_procene")
    references: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    start = _finite(event.get("sony_start_s"), "sony_start_s")
    end = _finite(event.get("sony_end_s"), "sony_end_s")
    for index, evaluation in enumerate(evaluations):
        if not isinstance(evaluation, Mapping):
            raise TypeError("ai_procene moraju sadržati JSON objekte")
        revision = _positive_int(evaluation.get("event_revision"), "AI event_revision")
        fingerprint = _fingerprint(
            evaluation.get("analysis_fingerprint"), "AI analysis_fingerprint"
        )
        evaluator_id = evaluation.get("evaluator_id")
        if not isinstance(evaluator_id, str) or not evaluator_id:
            raise TypeError("AI evaluator_id mora biti string")
        reference = (revision, fingerprint, evaluator_id)
        if reference in references:
            raise ValueError("AI procena za istu reviziju mora biti jedinstvena")
        references[reference] = evaluation
        status = evaluation.get("status")
        if status not in {"dostupno", "niska_pouzdanost", "nedovoljno_podataka"}:
            raise ValueError("AI status nije validan")
        score = _optional_score(evaluation.get("predlozena_ocena"), "AI ocena")
        if (status == "dostupno") != (score is not None):
            raise ValueError("AI ocena mora postojati samo za dostupan rezultat")
        confidence = _finite(evaluation.get("pouzdanost_0_1"), "AI pouzdanost")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("AI pouzdanost mora biti u opsegu 0..1")
        if not isinstance(evaluation.get("razlog"), str) or not evaluation.get("razlog"):
            raise ValueError("AI razlog mora biti neprazan")
        _iso_time(evaluation.get("ai_otkriven_u"), "ai_otkriven_u", nullable=True)
        evidence_rows = _list(evaluation.get("dokazi"), "AI dokazi")
        for evidence in evidence_rows:
            if not isinstance(evidence, Mapping):
                raise TypeError("AI dokaz mora biti JSON objekat")
            if not isinstance(evidence.get("metrika"), str) or not evidence.get("metrika"):
                raise ValueError("AI dokaz mora imati metriku")
            if not isinstance(evidence.get("jedinica"), str) or not evidence.get("jedinica"):
                raise ValueError("AI dokaz: jedinica mora biti neprazna")
            _finite(evidence.get("vrednost"), "AI dokaz vrednost")
            _finite(evidence.get("sony_s"), "AI dokaz Sony sekunda")
            if reference == (
                event.get("event_revision"),
                event.get("analysis_fingerprint"),
                EVALUATOR_ID,
            ) and not start <= float(evidence["sony_s"]) <= end:
                raise ValueError("AI dokaz mora biti unutar aktivnog događaja")
    return references


def _validate_trainer_assessments(
    event: Mapping[str, Any],
    ai_references: Mapping[tuple[Any, ...], Mapping[str, Any]],
) -> dict[int, Mapping[str, Any]]:
    assessments = _list(event.get("trener_procene"), "trener_procene")
    references: dict[int, Mapping[str, Any]] = {}
    first_phase: dict[int, str] = {}
    start = _finite(event.get("sony_start_s"), "sony_start_s")
    end = _finite(event.get("sony_end_s"), "sony_end_s")
    for assessment in assessments:
        if not isinstance(assessment, Mapping):
            raise TypeError("trener_procene moraju sadržati JSON objekte")
        revision = _positive_int(assessment.get("revizija"), "trener revizija")
        if revision in references:
            raise ValueError("trener revizija mora biti jedinstvena")
        references[revision] = assessment
        phase = assessment.get("faza")
        if phase not in TRAINER_PHASES:
            raise ValueError("faza trener procene nije validna")
        event_revision = _positive_int(
            assessment.get("event_revision"), "trener event_revision"
        )
        fingerprint = _fingerprint(
            assessment.get("analysis_fingerprint"), "trener analysis_fingerprint"
        )
        if (event_revision, fingerprint, EVALUATOR_ID) not in ai_references:
            raise ValueError("trener procena nema odgovarajuću AI procenu")
        if event_revision not in first_phase:
            first_phase[event_revision] = phase
            if phase != "pre_ai":
                raise ValueError("prva trener procena mora biti pre_ai")
        elif phase != "post_ai_korekcija":
            raise ValueError("kasnija trener procena mora biti post_ai_korekcija")
        visibility = assessment.get("status_vidljivosti")
        if visibility not in VISIBILITY_STATES:
            raise ValueError("status vidljivosti nije validan")
        technique = assessment.get("potvrdena_tehnika")
        score = _optional_score(assessment.get("ocena"), "trener ocena")
        reason = assessment.get("razlog")
        citations = assessment.get("citirani_sony_trenuci_s")
        if visibility == "dovoljno_vidljivo":
            if not isinstance(technique, str) or not technique.strip():
                raise ValueError("trener tehnika je obavezna")
            if score is None:
                raise ValueError("trener ocena je obavezna")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("trener razlog je obavezan")
            if not isinstance(citations, list) or not citations:
                raise ValueError("potrebna je najmanje jedna Sony sekunda")
            for citation in citations:
                second = _finite(citation, "trener Sony sekunda")
                if event_revision == event.get("event_revision") and not start <= second <= end:
                    raise ValueError("trener Sony sekunda mora biti unutar događaja")
        else:
            if score is not None or reason not in (None, "") or citations not in (None, []):
                raise ValueError("nedovoljno vidljiv događaj ne sme imati ocenu ili razlog")
        _iso_time(assessment.get("zakljucano_u"), "zakljucano_u")
    return references


def _validate_feedback(
    event: Mapping[str, Any],
    ai_references: Mapping[tuple[Any, ...], Mapping[str, Any]],
    trainer_references: Mapping[int, Mapping[str, Any]],
) -> None:
    feedback_rows = _list(event.get("procene_ai_predloga"), "procene_ai_predloga")
    seen: set[tuple[Any, ...]] = set()
    for feedback in feedback_rows:
        if not isinstance(feedback, Mapping):
            raise TypeError("procena AI predloga mora biti JSON objekat")
        event_revision = _positive_int(feedback.get("event_revision"), "feedback event_revision")
        fingerprint = _fingerprint(
            feedback.get("analysis_fingerprint"), "feedback analysis_fingerprint"
        )
        trainer_revision = _positive_int(
            feedback.get("trener_revizija"), "feedback trener_revizija"
        )
        evaluator_id = feedback.get("evaluator_id")
        reference = (event_revision, fingerprint, trainer_revision, evaluator_id)
        if reference in seen:
            raise ValueError("procena AI predloga mora biti jedinstvena")
        seen.add(reference)
        if (event_revision, fingerprint, evaluator_id) not in ai_references:
            raise ValueError("feedback nema odgovarajuću AI procenu")
        if trainer_revision not in trainer_references:
            raise ValueError("feedback nema odgovarajuću trener reviziju")
        trainer = trainer_references[trainer_revision]
        if (
            trainer.get("event_revision") != event_revision
            or trainer.get("analysis_fingerprint") != fingerprint
        ):
            raise ValueError("feedback i trener procena nisu iz iste runde")
        if ai_references[(event_revision, fingerprint, evaluator_id)].get("ai_otkriven_u") is None:
            raise ValueError("feedback zahteva otkrivenu AI procenu iz iste runde")
        if feedback.get("odnos") not in AI_RELATIONS:
            raise ValueError("feedback odnos nije validan")
        reason = feedback.get("razlog")
        if reason is not None and not isinstance(reason, str):
            raise TypeError("feedback razlog mora biti string ili null")
        _iso_time(feedback.get("sacuvano_u"), "sacuvano_u")
        evidence_names = {
            row.get("metrika")
            for row in ai_references[(event_revision, fingerprint, evaluator_id)].get("dokazi", [])
            if isinstance(row, Mapping)
        }
        rated: set[str] = set()
        for rating in _list(feedback.get("procene_dokaza"), "procene_dokaza"):
            if not isinstance(rating, Mapping):
                raise TypeError("procena dokaza mora biti JSON objekat")
            metric = rating.get("metrika")
            if metric not in evidence_names or metric in rated:
                raise ValueError("procena dokaza mora referencirati jedinstven AI dokaz")
            rated.add(metric)
            if rating.get("odnos") not in EVIDENCE_RELATIONS:
                raise ValueError("odnos prema dokazu nije validan")


def validate_trainer_ai_event(event: Mapping[str, Any]) -> None:
    """Validate all immutable references for one event."""
    if not isinstance(event, Mapping):
        raise TypeError("event mora biti JSON objekat")
    if _is_injury(event):
        if any(key in event for key in STATE_KEYS):
            raise ValueError("povredni događaj ne sme imati AI/trener procene")
        return
    event_revision = _positive_int(event.get("event_revision"), "event_revision")
    fingerprint = _fingerprint(event.get("analysis_fingerprint"), "analysis_fingerprint")
    ai_references = _validate_ai_evaluations(event)
    active_reference = (event_revision, fingerprint, EVALUATOR_ID)
    if active_reference not in ai_references:
        raise ValueError("aktivna AI procena ne postoji")
    trainer_references = _validate_trainer_assessments(event, ai_references)
    assessments = list(trainer_references.values())
    for (revision, evaluation_fingerprint, _), evaluation in ai_references.items():
        revealed_at = evaluation.get("ai_otkriven_u")
        if revealed_at is None:
            continue
        pre_ai = [
            assessment
            for assessment in assessments
            if assessment.get("event_revision") == revision
            and assessment.get("analysis_fingerprint") == evaluation_fingerprint
            and assessment.get("faza") == "pre_ai"
        ]
        if len(pre_ai) != 1:
            raise ValueError(
                "otkrivena AI procena zahteva jednu pre_ai procenu trenera iz iste runde"
            )
        if datetime.fromisoformat(revealed_at) < datetime.fromisoformat(pre_ai[0]["zakljucano_u"]):
            raise ValueError("AI procena ne može biti otkrivena pre pre_ai procene trenera")
    active_revision = event.get("aktivna_trener_revizija")
    if active_revision is not None:
        active_revision = _positive_int(active_revision, "aktivna_trener_revizija")
        if active_revision not in trainer_references:
            raise ValueError("aktivna trener revizija ne postoji")
        active_trainer_assessment(event)
    _validate_feedback(event, ai_references, trainer_references)
    duel = event.get("aktivni_duel")
    if duel is not None:
        if not isinstance(duel, Mapping):
            raise TypeError("aktivni_duel mora biti JSON objekat ili null")
        duel_reference = (
            duel.get("event_revision"),
            duel.get("analysis_fingerprint"),
            duel.get("evaluator_id"),
        )
        if duel_reference not in ai_references:
            raise ValueError("aktivni duel nema odgovarajuću AI procenu")
        if duel_reference != active_reference:
            raise ValueError("aktivni duel pripada drugoj rundi")
        trainer_revision = duel.get("trener_revizija")
        if trainer_revision not in trainer_references:
            raise ValueError("aktivni duel nema odgovarajuću trener reviziju")
        if trainer_revision != active_revision:
            raise ValueError("aktivni duel i aktivna trener procena nisu iz iste runde")
        trainer = trainer_references[trainer_revision]
        if (
            trainer.get("event_revision") != event_revision
            or trainer.get("analysis_fingerprint") != fingerprint
        ):
            raise ValueError("aktivni duel i trener procena nisu iz iste runde")
        if ai_references[active_reference].get("ai_otkriven_u") is None:
            raise ValueError("aktivni duel zahteva otkrivenu AI procenu")


def _legacy_annotation(event: Mapping[str, Any]) -> dict[str, Any] | None:
    technique = event.get("potvrdena_tehnika")
    score = event.get("ocena")
    note = event.get("napomena")
    if technique is None and score is None and note in (None, ""):
        return None
    return {
        "potvrdena_tehnika": technique,
        "ocena": score,
        "napomena": note,
        "nije_pre_ai": True,
    }


def _effective_analysis_fps(review: Mapping[str, Any]) -> float:
    fps = review.get("effective_analysis_fps")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)):
        pose_analysis = review.get("pose_analysis")
        fps = (
            pose_analysis.get("effective_analysis_fps")
            if isinstance(pose_analysis, Mapping)
            else None
        )
    if isinstance(fps, bool) or not isinstance(fps, (int, float)):
        raise ValueError("effective_analysis_fps nedostaje")
    numeric = float(fps)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError("effective_analysis_fps mora biti pozitivan konačan broj")
    return numeric


def start_new_event_revision(
    review: Mapping[str, Any], event: dict[str, Any]
) -> dict[str, Any]:
    """Start an independent assessment round when evaluator inputs change."""
    if _is_injury(event):
        raise ValueError("povredni događaj nema AI/trener reviziju")
    frames = review.get("frame_metrics", [])
    if not isinstance(frames, list):
        raise TypeError("frame_metrics mora biti JSON lista")
    fps = _effective_analysis_fps(review)
    fingerprint = compute_analysis_fingerprint(review, event)
    previous_revision = event.get("event_revision")
    previous_fingerprint = event.get("analysis_fingerprint")
    if previous_revision is not None:
        previous_revision = _positive_int(previous_revision, "event_revision")
    if previous_fingerprint is not None:
        _fingerprint(previous_fingerprint, "analysis_fingerprint")

    for key in ("ai_procene", "trener_procene", "procene_ai_predloga"):
        value = event.setdefault(key, [])
        if not isinstance(value, list):
            raise TypeError(f"{key} mora biti JSON lista")
    event.setdefault("legacy_annotations", [])

    active_exists = any(
        isinstance(row, Mapping)
        and row.get("event_revision") == previous_revision
        and row.get("analysis_fingerprint") == fingerprint
        and row.get("evaluator_id") == EVALUATOR_ID
        for row in event["ai_procene"]
    )
    if previous_revision is not None and previous_fingerprint == fingerprint and active_exists:
        if "imu_eksperimentalno" not in event:
            evaluated = evaluate_event(
                event,
                frames,
                effective_analysis_fps=fps,
                analysis_fingerprint=fingerprint,
            )
            event["imu_eksperimentalno"] = evaluated["imu_eksperimentalno"]
        event.setdefault("aktivna_trener_revizija", None)
        event.setdefault("aktivni_duel", None)
        validate_trainer_ai_event(event)
        return event

    revision = 1 if previous_revision is None else previous_revision + 1
    event["event_revision"] = revision
    event["analysis_fingerprint"] = fingerprint
    evaluated = evaluate_event(
        event,
        frames,
        effective_analysis_fps=fps,
        analysis_fingerprint=fingerprint,
    )
    event["imu_eksperimentalno"] = evaluated.pop("imu_eksperimentalno")
    event["ai_procene"].append(
        {
            "event_revision": revision,
            **evaluated,
            "ai_otkriven_u": None,
        }
    )
    event["aktivna_trener_revizija"] = None
    event["aktivni_duel"] = None
    event["ocena"] = None
    validate_trainer_ai_event(event)
    return event


def migrate_trainer_ai_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Add v3 event histories without upgrading legacy labels to pre-AI truth."""
    migrated = copy.deepcopy(dict(payload))
    events = migrated.get("events")
    if not isinstance(events, list):
        raise TypeError("events mora biti JSON lista")
    _effective_analysis_fps(migrated)

    migrated["version"] = 3
    for event in events:
        if not isinstance(event, dict):
            raise TypeError("events moraju sadržati JSON objekte")
        if _is_injury(event):
            for key in STATE_KEYS:
                event.pop(key, None)
            continue
        existing_legacy = event.get("legacy_annotations")
        if existing_legacy is None:
            legacy = _legacy_annotation(event)
            event["legacy_annotations"] = [] if legacy is None else [legacy]
        elif not isinstance(existing_legacy, list):
            raise TypeError("legacy_annotations mora biti JSON lista")
        if "event_revision" not in event:
            event["ocena"] = None
        start_new_event_revision(migrated, event)
    return migrated


__all__ = [
    "active_ai_evaluation",
    "active_trainer_assessment",
    "migrate_trainer_ai_payload",
    "start_new_event_revision",
    "validate_trainer_ai_event",
]
