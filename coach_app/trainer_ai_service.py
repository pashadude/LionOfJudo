"""Atomic trainer-first assessment and deterministic AI reveal workflow."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import math
from pathlib import Path
import tempfile
import threading
from typing import Any, Callable, Mapping

from coach_app.review_bundle import GenerationStore
from pipeline.trainer_ai_state import (
    active_ai_evaluation,
    active_trainer_assessment,
    validate_participants,
    validate_trainer_ai_event,
)
from pipeline.video_review_reports import event_is_injury, write_reports
from pipeline.video_review_storage import load_review_json


ASSESSMENT_FIELDS = {
    "status_vidljivosti",
    "potvrdena_tehnika",
    "ocena",
    "razlog",
    "citirani_sony_trenuci_s",
}
FEEDBACK_FIELDS = {"odnos", "razlog", "procene_dokaza"}
DRAFT_FIELDS = {"potvrdena_tehnika", "ocena", "napomena"}
PARTICIPANT_INPUT_FIELDS = {"trainer_name", "wrestler_name"}
ATOMIC_LOCK_FIELDS = {"participants", "assessment"}
AI_RELATIONS = {"slazem_se", "delimicno", "ne_slazem_se"}
EVIDENCE_RELATIONS = {"prihvatam", "nepotpun", "osporavam"}


def _default_clock() -> datetime:
    return datetime.now(timezone.utc)


class TrainerAiService:
    def __init__(
        self,
        session_dir: Path,
        *,
        clock: Callable[[], datetime] = _default_clock,
        mutation_lock: threading.RLock | None = None,
        store: GenerationStore | None = None,
    ) -> None:
        self.session_dir = Path(session_dir).expanduser().resolve()
        self.clock = clock
        self.mutation_lock = mutation_lock or threading.RLock()
        self.store = store or GenerationStore(self.session_dir)

    def lock_assessment(self, event_id: str, payload: object) -> dict[str, Any]:
        with self.mutation_lock:
            review = self.load_review()
            if review.get("participants") is None:
                raise ValueError("ime trenera i ime rvača moraju biti sačuvani")
            participants = validate_participants(review.get("participants"), required=True)
            assert participants is not None
            return self._lock_assessment_in_review(
                review, event_id, payload, participants
            )

    def lock_assessment_with_participants(
        self, event_id: str, payload: object
    ) -> dict[str, Any]:
        with self.mutation_lock:
            if not isinstance(payload, Mapping) or set(payload) != ATOMIC_LOCK_FIELDS:
                raise ValueError("atomsko zaključavanje zahteva participants i assessment")
            raw_participants = payload["participants"]
            if (
                not isinstance(raw_participants, Mapping)
                or set(raw_participants) != PARTICIPANT_INPUT_FIELDS
            ):
                raise ValueError("podaci učesnika nemaju tačna obavezna polja")
            participants = validate_participants(
                {**raw_participants, "updated_at": self._now_iso()}, required=True
            )
            assert participants is not None
            review = self.load_review()
            review["participants"] = participants
            return self._lock_assessment_in_review(
                review, event_id, payload["assessment"], participants
            )

    def _lock_assessment_in_review(
        self,
        review: dict[str, Any],
        event_id: str,
        payload: object,
        participants: Mapping[str, str],
    ) -> dict[str, Any]:
        event = self._normal_event(review, event_id)
        values = self._validate_assessment_payload(event, payload)
        ai = active_ai_evaluation(event)
        if ai is None:
            raise ValueError("aktivna AI procena ne postoji")
        revealed = ai.get("ai_otkriven_u") is not None
        current_round = [
            row
            for row in event.get("trener_procene", [])
            if row.get("event_revision") == event["event_revision"]
            and row.get("analysis_fingerprint") == event["analysis_fingerprint"]
        ]
        if not revealed and current_round:
            raise ValueError("pre_ai procena je već zaključana")
        phase = "post_ai_korekcija" if revealed else "pre_ai"
        all_revisions = [
            row.get("revizija")
            for candidate in review.get("events", [])
            if isinstance(candidate, Mapping)
            for row in candidate.get("trener_procene", [])
            if isinstance(row, Mapping) and isinstance(row.get("revizija"), int)
        ]
        assessment = {
            "revizija": max(all_revisions, default=0) + 1,
            "faza": phase,
            "event_revision": event["event_revision"],
            "analysis_fingerprint": event["analysis_fingerprint"],
            "trainer_name": participants["trainer_name"],
            "wrestler_name": participants["wrestler_name"],
            **values,
            "zakljucano_u": self._now_iso(),
        }
        event["trener_procene"].append(assessment)
        event["aktivna_trener_revizija"] = assessment["revizija"]
        event["status"] = "trener"
        event["potvrdena_tehnika"] = assessment["potvrdena_tehnika"]
        event["ocena"] = assessment["ocena"]
        if revealed:
            event["aktivni_duel"] = {
                "event_revision": event["event_revision"],
                "analysis_fingerprint": event["analysis_fingerprint"],
                "trener_revizija": assessment["revizija"],
                "evaluator_id": ai["evaluator_id"],
            }
        if review.get("version", 1) >= 3:
            validate_trainer_ai_event(event)
        self._activate(review)
        return {
            "event": copy.deepcopy(event),
            "assessment": copy.deepcopy(assessment),
            "participants": copy.deepcopy(dict(participants)),
        }

    def save_participants(self, payload: object) -> dict[str, Any]:
        with self.mutation_lock:
            if not isinstance(payload, Mapping) or set(payload) != PARTICIPANT_INPUT_FIELDS:
                raise ValueError("podaci učesnika nemaju tačna obavezna polja")
            participants = validate_participants(
                {**payload, "updated_at": self._now_iso()}, required=True
            )
            assert participants is not None
            review = self.load_review()
            review["participants"] = participants
            self._activate(review)
            return {"participants": copy.deepcopy(participants)}

    def reveal_ai(self, event_id: str) -> dict[str, Any]:
        with self.mutation_lock:
            review = self.load_review()
            event = self._normal_event(review, event_id)
            ai = active_ai_evaluation(event)
            if ai is None:
                raise ValueError("aktivna AI procena ne postoji")
            if ai.get("ai_otkriven_u") is not None:
                raise ValueError("AI procena je već otkrivena")
            assessment = active_trainer_assessment(event)
            if assessment is None or not any(
                row.get("faza") == "pre_ai"
                and row.get("event_revision") == event["event_revision"]
                and row.get("analysis_fingerprint") == event["analysis_fingerprint"]
                for row in event.get("trener_procene", [])
            ):
                raise ValueError("AI se ne može otkriti bez zaključane pre_ai procene")
            ai["ai_otkriven_u"] = self._now_iso()
            event["aktivni_duel"] = {
                "event_revision": event["event_revision"],
                "analysis_fingerprint": event["analysis_fingerprint"],
                "trener_revizija": assessment["revizija"],
                "evaluator_id": ai["evaluator_id"],
            }
            if review.get("version", 1) >= 3:
                validate_trainer_ai_event(event)
            self._activate(review)
            return {"event": copy.deepcopy(event), "assessment": copy.deepcopy(assessment)}

    def save_ai_feedback(self, event_id: str, payload: object) -> dict[str, Any]:
        with self.mutation_lock:
            review = self.load_review()
            event = self._normal_event(review, event_id)
            if not isinstance(payload, dict) or set(payload) != FEEDBACK_FIELDS:
                raise ValueError("AI feedback nema tačna obavezna polja")
            duel = event.get("aktivni_duel")
            ai = active_ai_evaluation(event)
            if not isinstance(duel, Mapping) or ai is None or ai.get("ai_otkriven_u") is None:
                raise ValueError("AI feedback zahteva otkriven aktivni duel")
            relation = payload["odnos"]
            if relation not in AI_RELATIONS:
                raise ValueError("odnos prema AI nije validan")
            reason = payload["razlog"]
            if reason is not None and (not isinstance(reason, str) or len(reason) > 2000):
                raise ValueError("feedback razlog mora biti tekst ili null")
            ratings = payload["procene_dokaza"]
            if not isinstance(ratings, list):
                raise ValueError("procene_dokaza moraju biti lista")
            normalized_ratings = []
            for rating in ratings:
                if not isinstance(rating, dict) or set(rating) != {"metrika", "odnos"}:
                    raise ValueError("procena dokaza nema tačna polja")
                if not isinstance(rating["metrika"], str) or not rating["metrika"]:
                    raise ValueError("metrika dokaza nije validna")
                if rating["odnos"] not in EVIDENCE_RELATIONS:
                    raise ValueError("odnos prema dokazu nije validan")
                normalized_ratings.append(dict(rating))
            reference = (
                duel["event_revision"],
                duel["analysis_fingerprint"],
                duel["trener_revizija"],
                duel["evaluator_id"],
            )
            if any(
                (
                    row.get("event_revision"),
                    row.get("analysis_fingerprint"),
                    row.get("trener_revizija"),
                    row.get("evaluator_id"),
                ) == reference
                for row in event.get("procene_ai_predloga", [])
                if isinstance(row, Mapping)
            ):
                raise ValueError("feedback za aktivni duel je već sačuvan")
            feedback = {
                "event_revision": duel["event_revision"],
                "analysis_fingerprint": duel["analysis_fingerprint"],
                "trener_revizija": duel["trener_revizija"],
                "evaluator_id": duel["evaluator_id"],
                "odnos": relation,
                "razlog": reason,
                "procene_dokaza": normalized_ratings,
                "sacuvano_u": self._now_iso(),
            }
            event["procene_ai_predloga"].append(feedback)
            validate_trainer_ai_event(event)
            self._activate(review)
            return {"event": copy.deepcopy(event), "assessment": copy.deepcopy(feedback)}

    def save_draft_annotation(self, event_id: str, payload: object) -> dict[str, Any]:
        with self.mutation_lock:
            review = self.load_review()
            event = self._normal_event(review, event_id)
            if not isinstance(payload, dict) or set(payload) != DRAFT_FIELDS:
                raise ValueError("annotation nema tačna obavezna polja")
            technique = payload["potvrdena_tehnika"]
            score = payload["ocena"]
            note = payload["napomena"]
            if not isinstance(technique, str) or len(technique) > 120:
                raise ValueError("potvrđena tehnika mora biti tekst do 120 znakova")
            if score is not None and (
                isinstance(score, bool)
                or not isinstance(score, int)
                or not 1 <= score <= 5
            ):
                raise ValueError("ocena mora biti prazna ili ceo broj od 1 do 5")
            if not isinstance(note, str) or len(note) > 2000:
                raise ValueError("napomena mora biti tekst do 2000 znakova")
            event.update(
                {
                    "potvrdena_tehnika": technique,
                    "ocena": score,
                    "napomena": note,
                    "status": "trener",
                }
            )
            if review.get("version", 1) >= 3:
                validate_trainer_ai_event(event)
            self._activate(review)
            return copy.deepcopy(event)

    def public_review(self) -> dict[str, Any]:
        return self._public_projection(self.load_review())

    def project_review(self, review: Mapping[str, Any]) -> dict[str, Any]:
        return self._public_projection(review)

    def public_event(self, event_id: str) -> dict[str, Any]:
        review = self.public_review()
        event = next(
            (
                item
                for item in review.get("events", [])
                if isinstance(item, dict) and item.get("event_id") == event_id
            ),
            None,
        )
        if event is None:
            raise ValueError("događaj nije pronađen")
        return event

    def load_review(self) -> dict[str, Any]:
        return load_review_json(self.store.resolve_current().review_path)

    def activate_review(
        self,
        review: dict[str, Any],
        *,
        staged_media: Mapping[str, Path] | None = None,
        removed_media_prefixes: tuple[str, ...] = (),
    ) -> None:
        review["event_metrics"] = copy.deepcopy(review["events"])
        public = self._public_projection(review)
        csv_text, markdown_text = self._render_reports(public)
        self.store.stage_and_activate(
            review,
            review["event_metrics"],
            csv_text,
            markdown_text,
            generated_at=self._now_iso(),
            staged_media=staged_media,
            removed_media_prefixes=removed_media_prefixes,
        )

    def _activate(self, review: dict[str, Any]) -> None:
        self.activate_review(review)

    @staticmethod
    def _render_reports(review: Mapping[str, Any]) -> tuple[str, str]:
        with tempfile.TemporaryDirectory() as raw:
            review_path = Path(raw) / "review.json"
            write_reports(review_path, review)
            return (
                review_path.with_name("izvestaj.csv").read_text(encoding="utf-8"),
                review_path.with_name("izvestaj.md").read_text(encoding="utf-8"),
            )

    def _now_iso(self) -> str:
        value = self.clock()
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise ValueError("clock mora vratiti datetime sa vremenskom zonom")
        return value.isoformat()

    @staticmethod
    def _normal_event(review: Mapping[str, Any], event_id: str) -> dict[str, Any]:
        if not isinstance(event_id, str) or not event_id or "/" in event_id or "\\" in event_id:
            raise ValueError("event ID nije validan")
        event = next(
            (
                item
                for item in review.get("events", [])
                if isinstance(item, dict) and item.get("event_id") == event_id
            ),
            None,
        )
        if event is None:
            raise ValueError("događaj nije pronađen")
        if event_is_injury(event):
            raise ValueError("povredni događaj je samo za čitanje")
        return event

    @staticmethod
    def _validate_assessment_payload(
        event: Mapping[str, Any], payload: object
    ) -> dict[str, Any]:
        if not isinstance(payload, dict) or set(payload) != ASSESSMENT_FIELDS:
            raise ValueError("trener procena nema tačna obavezna polja")
        visibility = payload["status_vidljivosti"]
        if visibility not in {"dovoljno_vidljivo", "nedovoljno_vidljivo"}:
            raise ValueError("status vidljivosti nije validan")
        technique = payload["potvrdena_tehnika"]
        score = payload["ocena"]
        reason = payload["razlog"]
        citations = payload["citirani_sony_trenuci_s"]
        if visibility == "nedovoljno_vidljivo":
            if any(value is not None for value in (technique, score, reason, citations)):
                raise ValueError("nedovoljno vidljivo zahteva null procenu")
        else:
            if not isinstance(technique, str) or not technique.strip() or len(technique) > 120:
                raise ValueError("potvrđena tehnika je obavezna")
            if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
                raise ValueError("ocena mora biti ceo broj od 1 do 5")
            if not isinstance(reason, str) or not reason.strip() or len(reason) > 2000:
                raise ValueError("razlog je obavezan")
            if not isinstance(citations, list) or not citations:
                raise ValueError("potrebna je najmanje jedna citirana Sony sekunda")
            normalized = []
            for citation in citations:
                if isinstance(citation, bool) or not isinstance(citation, (int, float)):
                    raise ValueError("citirana Sony sekunda mora biti broj")
                second = float(citation)
                if not math.isfinite(second) or not float(event["sony_start_s"]) <= second <= float(event["sony_end_s"]):
                    raise ValueError("citirana Sony sekunda je van događaja")
                normalized.append(second)
            citations = normalized
        return {
            "status_vidljivosti": visibility,
            "potvrdena_tehnika": technique,
            "ocena": score,
            "razlog": reason,
            "citirani_sony_trenuci_s": citations,
        }

    @staticmethod
    def _public_projection(review: Mapping[str, Any]) -> dict[str, Any]:
        public = copy.deepcopy(dict(review))
        for frame in public.get("frame_metrics", []):
            if isinstance(frame, dict):
                frame.pop("proxy_ubrzanja_norm_s2", None)
        for event in public.get("events", []):
            if not isinstance(event, dict) or event_is_injury(event):
                continue
            active = active_ai_evaluation(event)
            if active is None or active.get("ai_otkriven_u") is None:
                event.pop("ai_procene", None)
                event.pop("imu_eksperimentalno", None)
                event.pop("procene_ai_predloga", None)
                event.pop("aktivni_duel", None)
            else:
                event["ai_procene"] = [copy.deepcopy(active)]
        public["event_metrics"] = copy.deepcopy(public.get("events", []))
        return public


__all__ = ["TrainerAiService"]
