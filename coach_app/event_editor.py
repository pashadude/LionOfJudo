"""Transactional correction of normal Sony-master review events."""

from __future__ import annotations

import copy
import math
import os
from pathlib import Path
import shutil
import tempfile
import threading
from typing import Any, Callable, Mapping
import uuid

from pipeline.clip_extractor import cut_clip, probe_duration, verify_media_export
from pipeline.face_blur import BlurReport
from pipeline.trainer_ai_state import start_new_event_revision
from pipeline.video_review_contract import validate_review_payload
from pipeline.video_review_reports import event_is_injury, write_reports
from pipeline.video_review_metrics import summarize_event_metrics
from pipeline.video_review_storage import atomic_write_review, load_review_json


ANNOTATION_FIELDS = ("potvrdena_tehnika", "ocena", "napomena")
AUTO_VOICE_FIELDS = (
    "predlog_tehnike",
    "glasovna_fraza",
    "pouzdanost_glasa",
    "glasovna_fraza_pocetak_s",
    "glasovna_fraza_kraj_s",
)
class EventConflictError(ValueError):
    pass


class MediaExportError(ValueError):
    pass


def _number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} mora biti broj")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} mora biti konačan broj")
    return result


def _annotation(event: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(event.get(key))
        for key in ANNOTATION_FIELDS
        if event.get(key) not in (None, "", [], {})
    }


class EventEditor:
    def __init__(
        self,
        session_dir: Path,
        *,
        lock: threading.RLock | None = None,
        clip_exporter: Callable[..., Path] = cut_clip,
        media_probe: Callable[[Path], float] = probe_duration,
        review_loader: Callable[[], dict[str, Any]] | None = None,
        generation_activator: Callable[..., None] | None = None,
        privacy_processor: Callable[[Path, Path], BlurReport] | None = None,
    ) -> None:
        self.session_dir = Path(session_dir).resolve()
        self.review_path = self.session_dir / "review.json"
        self.lock = lock or threading.RLock()
        self.clip_exporter = clip_exporter
        self.media_probe = media_probe
        self.review_loader = review_loader or (lambda: load_review_json(self.review_path))
        self.generation_activator = generation_activator
        self.privacy_processor = privacy_processor

    def create(self, payload: object) -> dict[str, Any]:
        return self._apply("create", payload)

    def update_bounds(self, event_id: str, payload: object) -> dict[str, Any]:
        return self._apply("bounds", payload, event_id=event_id)

    def split(self, event_id: str, payload: object) -> dict[str, Any]:
        return self._apply("split", payload, event_id=event_id)

    def merge(self, payload: object) -> dict[str, Any]:
        return self._apply("merge", payload)

    def delete(self, event_id: str) -> dict[str, Any]:
        return self._apply("delete", {}, event_id=event_id)

    def _apply(
        self, operation: str, payload: object, *, event_id: str | None = None
    ) -> dict[str, Any]:
        with self.lock:
            original = self.review_loader()
            self._reject_sources_in_managed_events(original)
            review = copy.deepcopy(original)
            generated: set[str] = set()
            deleted: set[str] = set()
            created_event_id: str | None = None

            if operation == "create":
                start, end = self._bounds_payload(payload)
                created_event_id = self._next_event_id(review)
                event = self._blank_event(created_event_id, start, end)
                review.setdefault("events", []).append(event)
                generated.add(created_event_id)
                selected_event_id = created_event_id
            elif operation == "bounds":
                selected = self._normal_event(review, event_id)
                start, end = self._bounds_payload(payload)
                selected["sony_start_s"] = start
                selected["sony_end_s"] = end
                generated.add(selected["event_id"])
                selected_event_id = selected["event_id"]
            elif operation == "split":
                if not isinstance(payload, dict) or set(payload) != {"sony_split_s"}:
                    raise ValueError("podela zahteva tačno Sony vreme podele")
                selected = self._normal_event(review, event_id)
                split_s = _number(payload["sony_split_s"], "sony_split_s")
                if not float(selected["sony_start_s"]) < split_s < float(selected["sony_end_s"]):
                    raise ValueError("vreme podele mora biti unutar događaja")
                created_event_id = self._next_event_id(review)
                original_end = float(selected["sony_end_s"])
                selected["sony_end_s"] = split_s
                if review.get("version", 1) >= 3:
                    right = self._blank_event(created_event_id, split_s, original_end)
                else:
                    right = copy.deepcopy(selected)
                    right["event_id"] = created_event_id
                    right["sony_start_s"] = split_s
                    right["sony_end_s"] = original_end
                    right["status"] = "trener"
                    for key in ANNOTATION_FIELDS:
                        right[key] = None
                    for key in AUTO_VOICE_FIELDS:
                        right[key] = 0.0 if key == "pouzdanost_glasa" else None
                review["events"].append(right)
                generated.update({selected["event_id"], created_event_id})
                selected_event_id = selected["event_id"]
            elif operation == "merge":
                if not isinstance(payload, dict) or set(payload) != {"event_ids"}:
                    raise ValueError("spajanje zahteva tačno dva ID događaja")
                ids = payload["event_ids"]
                if not isinstance(ids, list) or len(ids) != 2 or len(set(ids)) != 2:
                    raise ValueError("spajanje zahteva dva različita ID događaja")
                first = self._normal_event(review, ids[0])
                second = self._normal_event(review, ids[1])
                if review.get("version", 1) >= 3 and (
                    first.get("trener_procene") or second.get("trener_procene")
                ):
                    raise EventConflictError(
                        "događaji sa zaključanim trenerskim procenama ne mogu se spojiti"
                    )
                ordered = sorted((first, second), key=lambda item: float(item["sony_start_s"]))
                normal_order = [
                    item["event_id"]
                    for item in sorted(
                        (item for item in review["events"] if not event_is_injury(item)),
                        key=lambda item: float(item["sony_start_s"]),
                    )
                ]
                positions = sorted(normal_order.index(item["event_id"]) for item in ordered)
                if positions[1] != positions[0] + 1:
                    raise ValueError("mogu se spojiti samo susedni normalni događaji")
                survivor, removed = ordered
                first_annotation = _annotation(survivor)
                second_annotation = _annotation(removed)
                if first_annotation and second_annotation and first_annotation != second_annotation:
                    raise EventConflictError(
                        "događaji imaju različite trenerske anotacije; prvo ih uskladite"
                    )
                if not first_annotation and second_annotation:
                    for key in ANNOTATION_FIELDS:
                        survivor[key] = copy.deepcopy(removed.get(key))
                survivor["sony_start_s"] = min(
                    float(survivor["sony_start_s"]), float(removed["sony_start_s"])
                )
                survivor["sony_end_s"] = max(
                    float(survivor["sony_end_s"]), float(removed["sony_end_s"])
                )
                review["events"].remove(removed)
                generated.add(survivor["event_id"])
                deleted.add(removed["event_id"])
                selected_event_id = survivor["event_id"]
            elif operation == "delete":
                selected = self._normal_event(review, event_id)
                annotation = _annotation(selected)
                if annotation:
                    review.setdefault("orphaned_annotations", []).append({
                        "source_event_id": selected["event_id"],
                        "reason": "obrisan_dogadjaj",
                        **annotation,
                    })
                review["events"].remove(selected)
                deleted.add(selected["event_id"])
                selected_event_id = next(
                    (
                        item["event_id"]
                        for item in review["events"]
                        if not event_is_injury(item)
                    ),
                    review["events"][0]["event_id"] if review["events"] else None,
                )
            else:
                raise ValueError("nepoznata izmena događaja")

            review["events"].sort(
                key=lambda item: (float(item["sony_start_s"]), str(item["event_id"]))
            )
            for item in review["events"]:
                if item["event_id"] in generated:
                    self._refresh_event(review, item)
                    if review.get("version", 1) >= 3:
                        start_new_event_revision(review, item)
            if deleted:
                self._update_manifest(review, deleted, [])
            review["event_metrics"] = copy.deepcopy(review["events"])
            validate_review_payload(review)
            if review.get("version", 1) >= 3:
                self._persist_generation(review, generated, deleted)
            else:
                self._persist_transaction(original, review, generated, deleted)
            result = {"review": review, "selected_event_id": selected_event_id}
            if created_event_id is not None:
                result["created_event_id"] = created_event_id
            return result

    @staticmethod
    def _bounds_payload(payload: object) -> tuple[float, float]:
        if not isinstance(payload, dict) or set(payload) != {"sony_start_s", "sony_end_s"}:
            raise ValueError("granice zahtevaju početak i kraj na Sony vremenskoj osi")
        start = _number(payload["sony_start_s"], "sony_start_s")
        end = _number(payload["sony_end_s"], "sony_end_s")
        if end <= start:
            raise ValueError("kraj događaja mora biti posle početka")
        return start, end

    @staticmethod
    def _next_event_id(review: Mapping[str, Any]) -> str:
        existing = {
            event.get("event_id")
            for event in review.get("events", [])
            if isinstance(event, Mapping)
        }
        index = 1
        while f"e-coach-{index:03d}" in existing:
            index += 1
        return f"e-coach-{index:03d}"

    @staticmethod
    def _normal_event(review: Mapping[str, Any], event_id: str | None) -> dict[str, Any]:
        selected = next(
            (
                event for event in review.get("events", [])
                if isinstance(event, dict) and event.get("event_id") == event_id
            ),
            None,
        )
        if selected is None:
            raise ValueError("događaj nije pronađen")
        if event_is_injury(selected):
            raise ValueError("povredni događaj je samo za čitanje")
        return selected

    @staticmethod
    def _blank_event(event_id: str, start: float, end: float) -> dict[str, Any]:
        return {
            "event_id": event_id,
            "sony_start_s": start,
            "sony_end_s": end,
            "predlog_tehnike": None,
            "potvrdena_tehnika": None,
            "glasovna_fraza": None,
            "glasovna_fraza_pocetak_s": None,
            "glasovna_fraza_kraj_s": None,
            "pouzdanost_glasa": 0.0,
            "ocena": None,
            "napomena": None,
            "iskljuceno_iz_statistike": False,
            "status": "trener",
        }

    def _refresh_event(self, review: Mapping[str, Any], event: dict[str, Any]) -> None:
        anchors = review["anchors"]
        first_anchor = min(float(anchor["sony_s"]) for anchor in anchors)
        cutoff = float(review["injury_cutoff_s"])
        start = float(event["sony_start_s"])
        end = float(event["sony_end_s"])
        if start < first_anchor or end > cutoff:
            raise ValueError("događaj mora biti između prvog ankera i preseka povrede")
        slope = float(review["time_map"]["slope"])
        intercept = float(review["time_map"]["intercept"])
        iphone_start = (start - intercept) / slope
        iphone_end = (end - intercept) / slope
        iphone_duration = float(review["iphone_duration_s"])
        if not 0.0 <= iphone_start < iphone_end <= iphone_duration:
            raise ValueError("izvedene iPhone granice su van izvornog videa")
        event["iphone_start_s"] = iphone_start
        event["iphone_end_s"] = iphone_end
        frame_metrics = [
            item for item in review.get("frame_metrics", []) if isinstance(item, Mapping)
        ]
        event.update(summarize_event_metrics(event, frame_metrics))
        event["media"] = {
            "sony": f"/media/events/{event['event_id']}/sony.mp4",
            "iphone": f"/media/events/{event['event_id']}/iphone.mp4",
        }

    def _source_path(self, review: Mapping[str, Any], camera: str) -> Path:
        path = self._configured_source_path(review, camera).resolve()
        if not path.is_file():
            raise MediaExportError(f"izvorni {camera} video nije dostupan")
        return path

    def _configured_source_path(
        self, review: Mapping[str, Any], camera: str
    ) -> Path:
        source = review.get("sources", {}).get(camera, {})
        value = source.get("path") if isinstance(source, Mapping) else None
        if not value:
            value = review.get(f"{camera}_video")
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = self.session_dir / path
        return Path(os.path.abspath(path))

    def _reject_sources_in_managed_events(self, review: Mapping[str, Any]) -> None:
        events_root = (self.session_dir / "events").resolve()
        for camera in ("sony", "iphone"):
            logical_source = self._configured_source_path(review, camera)
            resolved_source = logical_source.resolve()
            if logical_source.is_relative_to(events_root) or resolved_source.is_relative_to(
                events_root
            ):
                raise MediaExportError(
                    f"izvorni {camera} video ne sme biti unutar upravljanog events direktorijuma"
                )

    def _export_event(
        self, review: Mapping[str, Any], event: Mapping[str, Any], output: Path
    ) -> list[dict[str, Any]]:
        if self.privacy_processor is None:
            raise MediaExportError("privacy processor nije dostupan")
        windows = {
            "sony": (float(event["sony_start_s"]), float(event["sony_end_s"])),
            "iphone": (float(event["iphone_start_s"]), float(event["iphone_end_s"])),
        }
        manifest_rows = []
        for camera, (start, end) in windows.items():
            destination = output / f"{camera}.mp4"
            destination.parent.mkdir(parents=True, exist_ok=True)
            raw = destination.with_name(f".{camera}.raw.mp4")
            try:
                result = self.clip_exporter(
                    self._source_path(review, camera), start, end, raw
                )
                verify_media_export(
                    Path(result),
                    end - start,
                    probe=self.media_probe,
                )
                report = self.privacy_processor(Path(result), destination)
                if not isinstance(report, BlurReport) or not report.privacy_verified:
                    reason = (
                        report.failure_reason if isinstance(report, BlurReport) else None
                    )
                    raise MediaExportError(
                        reason or "privatnost izvedenog videa nije potvrđena"
                    )
                verify_media_export(
                    destination,
                    end - start,
                    probe=self.media_probe,
                )
                relative = f"events/{event['event_id']}/{camera}.mp4"
                manifest_rows.append(
                    report.to_manifest(relative, "event_clip")
                )
            finally:
                raw.unlink(missing_ok=True)
        return manifest_rows

    @staticmethod
    def _update_manifest(
        review: dict[str, Any],
        touched: set[str],
        new_rows: list[dict[str, Any]],
    ) -> None:
        prefixes = tuple(f"events/{event_id}/" for event_id in touched)
        existing = review.get("derived_media_manifest", [])
        if not isinstance(existing, list):
            raise ValueError("derived_media_manifest mora biti lista")
        retained = [
            copy.deepcopy(row)
            for row in existing
            if isinstance(row, Mapping)
            and not any(
                str(row.get("relative_path", "")).startswith(prefix)
                for prefix in prefixes
            )
        ]
        review["derived_media_manifest"] = sorted(
            retained + copy.deepcopy(new_rows),
            key=lambda row: str(row.get("relative_path", "")),
        )

    def _persist_transaction(
        self,
        original: Mapping[str, Any],
        review: dict[str, Any],
        generated: set[str],
        deleted: set[str],
    ) -> None:
        events_root = self.session_dir / "events"
        events_root.mkdir(parents=True, exist_ok=True)
        transaction = events_root / f".txn-{uuid.uuid4().hex}"
        staged = transaction / "staged"
        backups = transaction / "backups"
        transaction.mkdir(parents=True)
        try:
            try:
                manifest_rows = []
                for event_id in generated:
                    event = next(
                        event for event in review["events"]
                        if event["event_id"] == event_id
                    )
                    manifest_rows.extend(
                        self._export_event(review, event, staged / event_id)
                    )
            except Exception as exc:
                raise MediaExportError(f"izvoz medija nije uspeo: {exc}") from exc

            self._update_manifest(
                review, generated | deleted, manifest_rows
            )
            validate_review_payload(review)

            touched = generated | deleted
            installed: set[str] = set()
            backed_up: set[str] = set()
            try:
                for event_id in touched:
                    current = events_root / event_id
                    if current.exists():
                        backup = backups / event_id
                        backup.parent.mkdir(parents=True, exist_ok=True)
                        os.replace(current, backup)
                        backed_up.add(event_id)
                for event_id in generated:
                    os.replace(staged / event_id, events_root / event_id)
                    installed.add(event_id)
                atomic_write_review(self.review_path, review)
                write_reports(self.review_path, review)
            except Exception:
                for event_id in installed:
                    shutil.rmtree(events_root / event_id, ignore_errors=True)
                for event_id in backed_up:
                    backup = backups / event_id
                    if backup.exists():
                        os.replace(backup, events_root / event_id)
                atomic_write_review(self.review_path, original)
                write_reports(self.review_path, original)
                raise
        finally:
            shutil.rmtree(transaction, ignore_errors=True)

    def _persist_generation(
        self,
        review: dict[str, Any],
        generated: set[str],
        deleted: set[str],
    ) -> None:
        if self.generation_activator is None:
            raise RuntimeError("v3 editor zahteva generation activator")
        with tempfile.TemporaryDirectory(prefix="lion-judo-events-") as raw:
            staging_root = Path(raw)
            staged_media: dict[str, Path] = {}
            try:
                manifest_rows = []
                for event_id in generated:
                    event = next(
                        event
                        for event in review["events"]
                        if event["event_id"] == event_id
                    )
                    event_root = staging_root / event_id
                    manifest_rows.extend(
                        self._export_event(review, event, event_root)
                    )
                    for camera in ("sony", "iphone"):
                        staged_media[f"events/{event_id}/{camera}.mp4"] = (
                            event_root / f"{camera}.mp4"
                        )
            except Exception as exc:
                raise MediaExportError(f"izvoz medija nije uspeo: {exc}") from exc
            self._update_manifest(
                review, generated | deleted, manifest_rows
            )
            validate_review_payload(review)
            self.generation_activator(
                review,
                staged_media=staged_media,
                removed_media_prefixes=tuple(
                    f"events/{event_id}" for event_id in sorted(deleted)
                ),
            )


__all__ = [
    "EventConflictError",
    "EventEditor",
    "MediaExportError",
    "summarize_event_metrics",
]
