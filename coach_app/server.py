"""Local-only HTTP server for reviewing one imported video session."""

from __future__ import annotations

import json
import mimetypes
import threading
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import unquote, urlsplit

from pipeline.video_review_contract import (
    AnchorPair,
    ReviewEvent,
    ReviewSession,
    validate_review_session,
)
from pipeline.video_sync import fit_time_map
from pipeline.clip_extractor import cut_clip, probe_duration
from pipeline.face_blur import BlurReport, build_privacy_processor
from pipeline.video_review_reports import event_is_injury, write_reports
from pipeline.video_review_storage import atomic_write_review, load_review_json

from coach_app.event_editor import EventConflictError, EventEditor, MediaExportError
from coach_app.trainer_ai_service import TrainerAiService, _default_clock


ANNOTATION_FIELDS = {"potvrdena_tehnika", "ocena", "napomena"}
MAX_EVENT_ID_LENGTH = 128
MAX_TECHNIQUE_LENGTH = 120
MAX_NOTE_LENGTH = 2000


def _strict_load(path: Path) -> dict[str, Any]:
    return load_review_json(path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_review(path, payload)


def _event_is_injury(event: Mapping[str, Any]) -> bool:
    return event_is_injury(event)


def _validate_annotation(event_id: object, payload: object, event: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(event_id, str) or not event_id or len(event_id) > MAX_EVENT_ID_LENGTH:
        raise ValueError("event ID nije validan")
    if "/" in event_id or "\\" in event_id or event_id in {".", ".."}:
        raise ValueError("event ID nije validan")
    if _event_is_injury(event):
        raise ValueError("povredni događaj je samo za čitanje")
    if not isinstance(payload, dict) or set(payload) != ANNOTATION_FIELDS:
        raise ValueError("annotation mora imati tačno polja potvrđena tehnika, ocena i napomena")
    technique = payload["potvrdena_tehnika"]
    note = payload["napomena"]
    score = payload["ocena"]
    if not isinstance(technique, str) or len(technique) > MAX_TECHNIQUE_LENGTH:
        raise ValueError("potvrđena tehnika mora biti tekst do 120 znakova")
    if not isinstance(note, str) or len(note) > MAX_NOTE_LENGTH:
        raise ValueError("napomena mora biti tekst do 2000 znakova")
    if score is not None and (
        isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5
    ):
        raise ValueError("ocena mora biti prazna ili ceo broj od 1 do 5")
    return {
        "potvrdena_tehnika": technique,
        "ocena": score,
        "napomena": note,
    }


def _write_report(path: Path, review: Mapping[str, Any]) -> None:
    write_reports(path, review)


def save_annotation(review_path: Path, event_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and atomically save one coach annotation, then regenerate reports."""
    review_path = Path(review_path)
    review = _strict_load(review_path)
    events = review.get("events")
    if not isinstance(events, list):
        raise ValueError("review.json nema events listu")
    selected = next(
        (event for event in events if isinstance(event, dict) and event.get("event_id") == event_id),
        None,
    )
    if selected is None:
        raise ValueError("događaj nije pronađen")
    annotation = _validate_annotation(event_id, payload, selected)
    selected.update(annotation)
    selected["status"] = "trener"
    event_metrics = review.get("event_metrics")
    if isinstance(event_metrics, list):
        metric_event = next(
            (
                event for event in event_metrics
                if isinstance(event, dict) and event.get("event_id") == event_id
            ),
            None,
        )
        if metric_event is not None:
            metric_event.update(annotation)
            metric_event["status"] = "trener"
    _atomic_json(review_path, review)
    _write_report(review_path, review)
    return dict(selected)


def _review_session_for_sync(review: Mapping[str, Any], anchors: list[AnchorPair], cutoff: float) -> ReviewSession:
    events_raw = review.get("events", [])
    if not isinstance(events_raw, list):
        raise ValueError("events mora biti JSON lista")
    events = [ReviewEvent.from_dict(event) for event in events_raw if isinstance(event, dict)]
    session = ReviewSession(
        session_id=review.get("session_id", "session"),
        sony_video=review.get("sony_video", ""),
        iphone_video=review.get("iphone_video", ""),
        anchors=anchors,
        injury_cutoff_s=cutoff,
        events=events,
    )
    sony_duration = review.get("sony_duration_s", float("inf"))
    iphone_duration = review.get("iphone_duration_s", float("inf"))
    if not isinstance(sony_duration, (int, float)) or not isinstance(iphone_duration, (int, float)):
        raise ValueError("izvorna trajanja nisu validna")
    validate_review_session(session, float(sony_duration), float(iphone_duration))
    return session


class SyncLockedError(ValueError):
    pass


def _sync_is_locked(review: Mapping[str, Any], session_dir: Path) -> bool:
    if review.get("sync_locked") is True:
        return True
    if any(review.get(key) for key in ("events", "event_metrics", "frame_metrics")):
        return True
    if review.get("pose_analysis") or review.get("sources"):
        return True
    if (session_dir / "media" / "session_side_by_side.mp4").is_file():
        return True
    events_dir = session_dir / "events"
    return events_dir.is_dir() and any(path.is_file() for path in events_dir.rglob("*"))


class ReviewServer:
    """Lifecycle wrapper around a loopback-only review HTTP server."""

    def __init__(
        self,
        session_dir: Path,
        port: int,
        *,
        clip_exporter: Callable[..., Path] = cut_clip,
        media_probe: Callable[[Path], float] = probe_duration,
        clock: Callable[[], datetime] = _default_clock,
        privacy_processor: Callable[[Path, Path], BlurReport] | None = None,
    ):
        self.session_dir = Path(session_dir).expanduser().resolve()
        if not self.session_dir.is_dir():
            raise ValueError("direktorijum sesije ne postoji")
        self.review_path = self.session_dir / "review.json"
        if not self.review_path.is_file():
            raise ValueError("sesija nema review.json")
        if not all(
            (self.session_dir / name).is_file()
            for name in ("izvestaj.csv", "izvestaj.md")
        ):
            _write_report(self.review_path, _strict_load(self.review_path))
        self.static_dir = Path(__file__).resolve().parent / "static"
        self.mutation_lock = threading.RLock()
        self.trainer_ai_service = TrainerAiService(
            self.session_dir,
            clock=clock,
            mutation_lock=self.mutation_lock,
        )
        privacy_processor = privacy_processor or build_privacy_processor()
        self.event_editor = EventEditor(
            self.session_dir,
            lock=self.mutation_lock,
            clip_exporter=clip_exporter,
            media_probe=media_probe,
            review_loader=self.trainer_ai_service.load_review,
            generation_activator=self.trainer_ai_service.activate_review,
            privacy_processor=privacy_processor,
        )
        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), _ReviewHandler)
        self.httpd.review_server = self  # type: ignore[attr-defined]
        bound_port = self.httpd.server_address[1]
        self.base_url = f"http://127.0.0.1:{bound_port}"

    def start_in_thread(self) -> threading.Thread:
        thread = threading.Thread(target=self.httpd.serve_forever, name="coach-review", daemon=True)
        thread.start()
        return thread

    def shutdown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def create_server(
    session_dir: Path,
    port: int = 8765,
    *,
    clip_exporter: Callable[..., Path] = cut_clip,
    media_probe: Callable[[Path], float] = probe_duration,
    clock: Callable[[], datetime] = _default_clock,
    privacy_processor: Callable[[Path, Path], BlurReport] | None = None,
) -> ReviewServer:
    return ReviewServer(
        Path(session_dir),
        port,
        clip_exporter=clip_exporter,
        media_probe=media_probe,
        clock=clock,
        privacy_processor=privacy_processor,
    )


class _ReviewHandler(BaseHTTPRequestHandler):
    server: ThreadingHTTPServer

    @property
    def app(self) -> ReviewServer:
        return self.server.review_server  # type: ignore[attr-defined]

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _send_json(self, status: int, payload: object) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        self._send_json(status, {"error": message})

    def _read_body(self) -> object:
        content_type = self.headers.get("Content-Type", "")
        if not content_type.lower().split(";", 1)[0].strip() == "application/json":
            raise ValueError("Content-Type mora biti application/json")
        content_length = self.headers.get("Content-Length")
        if content_length is None or not content_length.isdigit() or int(content_length) > 1_000_000:
            raise ValueError("zahtev mora imati validnu dužinu tela")
        raw = self.rfile.read(int(content_length))
        def reject_constant(value: str) -> None:
            raise ValueError(f"JSON konstanta nije dozvoljena: {value}")
        return json.loads(raw.decode("utf-8"), parse_constant=reject_constant)

    def do_GET(self) -> None:
        path = unquote(urlsplit(self.path).path)
        try:
            if self._is_internal_path(path):
                raise FileNotFoundError
            if path in {"/", "/index.html"}:
                self._serve_file(self.app.static_dir / "index.html", self.app.static_dir, "text/html; charset=utf-8")
            elif path.startswith("/static/"):
                self._serve_file(self.app.static_dir / path.removeprefix("/static/"), self.app.static_dir)
            elif path == "/api/session":
                review = self.app.trainer_ai_service.public_review()
                review["sync_locked"] = _sync_is_locked(review, self.app.session_dir)
                self._send_json(HTTPStatus.OK, review)
            elif path.startswith("/api/events/"):
                event_id = path.removeprefix("/api/events/")
                if not event_id or "/" in event_id:
                    raise FileNotFoundError
                review = self.app.trainer_ai_service.public_review()
                event = next(
                    (event for event in review.get("events", []) if isinstance(event, dict) and event.get("event_id") == event_id),
                    None,
                )
                if event is None:
                    raise FileNotFoundError
                self._send_json(HTTPStatus.OK, event)
            elif path in {"/izvestaj.csv", "/izvestaj.md"}:
                snapshot = self.app.trainer_ai_service.store.resolve_current()
                candidate = (
                    snapshot.csv_path if path.endswith(".csv") else snapshot.markdown_path
                )
                self._serve_file(candidate, snapshot.root)
            elif path in {"/trener_dataset.json", "/trener_assessment_audit.json"}:
                snapshot = self.app.trainer_ai_service.store.resolve_current()
                candidate = (
                    snapshot.dataset_path
                    if path == "/trener_dataset.json"
                    else snapshot.audit_path
                )
                if not candidate.is_file():
                    raise FileNotFoundError
                self._serve_file(candidate, snapshot.root)
            elif path.startswith("/media/"):
                relative = path.removeprefix("/media/")
                snapshot = self.app.trainer_ai_service.store.resolve_current()
                review = _strict_load(snapshot.review_path)
                relative = self._verified_media_relative(review, relative)
                if relative.startswith(("events/", "previews/", "analysis/")):
                    candidate = snapshot.root / relative
                else:
                    candidate = snapshot.root / "media" / relative
                self._serve_file(candidate, snapshot.root)
            else:
                relative = path.removeprefix("/")
                if relative.startswith(("events/", "previews/", "analysis/", "media/")):
                    raise FileNotFoundError
                self._serve_file(self.app.session_dir / relative, self.app.session_dir)
        except FileNotFoundError:
            self._error(HTTPStatus.NOT_FOUND, "resurs nije pronađen")
        except ValueError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except PermissionError:
            self._error(HTTPStatus.NOT_FOUND, "resurs nije pronađen")

    @staticmethod
    def _is_internal_path(path: str) -> bool:
        return bool(
            path == "/review.json"
            or path == "/current-generation.json"
            or path == "/analysis"
            or path.startswith("/analysis/")
            or path == "/.review-generations"
            or path.startswith("/.review-generations/")
        )

    @staticmethod
    def _verified_media_relative(
        review: Mapping[str, Any], relative_value: str
    ) -> str:
        relative = Path(relative_value)
        if (
            not relative_value
            or relative.is_absolute()
            or ".." in relative.parts
            or "." in relative.parts
            or "\\" in relative_value
        ):
            raise FileNotFoundError
        normalized = relative.as_posix()
        manifest = review.get("derived_media_manifest")
        if not isinstance(manifest, list):
            raise FileNotFoundError
        matches = [
            row
            for row in manifest
            if isinstance(row, Mapping)
            and row.get("relative_path") == normalized
            and row.get("privacy_verified") is True
            and (
                (
                    row.get("media_type") == "event_clip"
                    and normalized.startswith("events/")
                )
                or (
                    row.get("media_type") == "anchor_preview"
                    and normalized.startswith("previews/")
                )
                or (
                    row.get("media_type") == "side_by_side"
                    and normalized == "session_side_by_side.mp4"
                )
            )
        ]
        if len(matches) != 1:
            raise FileNotFoundError
        return normalized

    def do_PUT(self) -> None:
        path = unquote(urlsplit(self.path).path)
        if path == "/api/session/participants":
            try:
                result = self.app.trainer_ai_service.save_participants(self._read_body())
                self._send_json(HTTPStatus.OK, result)
            except (ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        prefix = "/api/events/"
        if not path.startswith(prefix):
            self._error(HTTPStatus.NOT_FOUND, "resurs nije pronađen")
            return
        suffix = next(
            (
                candidate
                for candidate in ("/annotation", "/bounds", "/ai-feedback")
                if path.endswith(candidate)
            ),
            None,
        )
        if suffix is None:
            self._error(HTTPStatus.NOT_FOUND, "resurs nije pronađen")
            return
        event_id = path[len(prefix):-len(suffix)]
        if not event_id or "/" in event_id:
            self._error(HTTPStatus.NOT_FOUND, "resurs nije pronađen")
            return
        try:
            payload = self._read_body()
            if suffix == "/annotation":
                snapshot = self.app.trainer_ai_service.store.resolve_current()
                current = _strict_load(snapshot.review_path)
                if current.get("version", 1) >= 3:
                    self.app.trainer_ai_service.save_draft_annotation(event_id, payload)
                else:
                    save_annotation(snapshot.review_path, event_id, payload)  # type: ignore[arg-type]
                result = self.app.trainer_ai_service.public_event(event_id)
            elif suffix == "/ai-feedback":
                internal = self.app.trainer_ai_service.save_ai_feedback(event_id, payload)
                result = {
                    "event": self.app.trainer_ai_service.public_event(event_id),
                    "assessment": internal["assessment"],
                }
            else:
                internal = self.app.event_editor.update_bounds(event_id, payload)
                result = dict(internal)
                result["review"] = self.app.trainer_ai_service.project_review(
                    internal["review"]
                )
            self._send_json(HTTPStatus.OK, result)
        except MediaExportError as exc:
            self._error(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))
        except EventConflictError as exc:
            self._error(HTTPStatus.CONFLICT, str(exc))
        except (ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))

    def do_POST(self) -> None:
        path = unquote(urlsplit(self.path).path)
        try:
            if path == "/api/session/sync":
                payload = self._read_body()
                if not isinstance(payload, dict) or set(payload) != {"anchors", "injury_cutoff_s"}:
                    raise ValueError("sinhronizacija zahteva ankere i presek povrede")
                raw_anchors = payload["anchors"]
                if not isinstance(raw_anchors, list) or len(raw_anchors) != 2:
                    raise ValueError("potrebna su tačno dva ankera")
                anchors = [AnchorPair.from_dict(value) for value in raw_anchors]
                time_map = fit_time_map(anchors)
                cutoff = payload["injury_cutoff_s"]
                if isinstance(cutoff, bool) or not isinstance(cutoff, (int, float)) or cutoff <= 0:
                    raise ValueError("presek povrede mora biti pozitivan broj")
                with self.app.mutation_lock:
                    review = _strict_load(self.app.review_path)
                    if _sync_is_locked(review, self.app.session_dir):
                        raise SyncLockedError(
                            "Sinhronizacija je zaključana za uvezenu sesiju sa izvedenim "
                            "medijima ili događajima; potreban je novi uvoz."
                        )
                    _review_session_for_sync(review, anchors, float(cutoff))
                    review["anchors"] = [anchor.to_dict() for anchor in anchors]
                    review["time_map"] = time_map.to_dict()
                    review["injury_cutoff_s"] = float(cutoff)
                    review["sync_locked"] = False
                    _atomic_json(self.app.review_path, review)
                    _write_report(self.app.review_path, review)
                self._send_json(
                    HTTPStatus.OK,
                    self.app.trainer_ai_service.project_review(review),
                )
            elif path == "/api/events":
                result = self.app.event_editor.create(self._read_body())
                result["review"] = self.app.trainer_ai_service.project_review(
                    result["review"]
                )
                self._send_json(HTTPStatus.CREATED, result)
            elif path == "/api/events/merge":
                result = self.app.event_editor.merge(self._read_body())
                result["review"] = self.app.trainer_ai_service.project_review(
                    result["review"]
                )
                self._send_json(HTTPStatus.OK, result)
            elif path.startswith("/api/events/") and path.endswith("/trainer-assessments"):
                event_id = path[len("/api/events/"):-len("/trainer-assessments")]
                if not event_id or "/" in event_id:
                    raise ValueError("event ID nije validan")
                internal = self.app.trainer_ai_service.lock_assessment(
                    event_id, self._read_body()
                )
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "event": self.app.trainer_ai_service.public_event(event_id),
                        "assessment": internal["assessment"],
                    },
                )
            elif path.startswith("/api/events/") and path.endswith("/ai-reveal"):
                event_id = path[len("/api/events/"):-len("/ai-reveal")]
                if not event_id or "/" in event_id:
                    raise ValueError("event ID nije validan")
                if self._read_body() != {}:
                    raise ValueError("AI reveal zahtev mora biti prazan JSON objekat")
                internal = self.app.trainer_ai_service.reveal_ai(event_id)
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "event": self.app.trainer_ai_service.public_event(event_id),
                        "assessment": internal["assessment"],
                    },
                )
            elif path.startswith("/api/events/") and path.endswith("/split"):
                event_id = path[len("/api/events/"):-len("/split")]
                if not event_id or "/" in event_id:
                    raise ValueError("event ID nije validan")
                result = self.app.event_editor.split(event_id, self._read_body())
                result["review"] = self.app.trainer_ai_service.project_review(
                    result["review"]
                )
                self._send_json(HTTPStatus.OK, result)
            else:
                self._error(HTTPStatus.NOT_FOUND, "resurs nije pronađen")
        except SyncLockedError as exc:
            self._error(HTTPStatus.CONFLICT, str(exc))
        except MediaExportError as exc:
            self._error(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))
        except EventConflictError as exc:
            self._error(HTTPStatus.CONFLICT, str(exc))
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))

    def do_DELETE(self) -> None:
        path = unquote(urlsplit(self.path).path)
        prefix = "/api/events/"
        event_id = path.removeprefix(prefix) if path.startswith(prefix) else ""
        if not event_id or "/" in event_id:
            self._error(HTTPStatus.NOT_FOUND, "resurs nije pronađen")
            return
        try:
            result = self.app.event_editor.delete(event_id)
            result["review"] = self.app.trainer_ai_service.project_review(
                result["review"]
            )
            self._send_json(HTTPStatus.OK, result)
        except MediaExportError as exc:
            self._error(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))
        except EventConflictError as exc:
            self._error(HTTPStatus.CONFLICT, str(exc))
        except (ValueError, TypeError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))

    def _serve_file(self, candidate: Path, root: Path, content_type: str | None = None) -> None:
        root = root.resolve()
        resolved = candidate.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise FileNotFoundError from exc
        if not resolved.is_file():
            raise FileNotFoundError
        size = resolved.stat().st_size
        start, end, partial = 0, max(0, size - 1), False
        range_header = self.headers.get("Range")
        if range_header:
            if not range_header.startswith("bytes=") or "," in range_header:
                self._send_range_error(size)
                return
            value = range_header.removeprefix("bytes=")
            try:
                first, last = value.split("-", 1)
                if first:
                    start = int(first)
                    end = int(last) if last else size - 1
                else:
                    length = int(last)
                    if length <= 0:
                        raise ValueError
                    start = max(0, size - length)
                    end = size - 1
                if start < 0 or start >= size or end < start:
                    raise ValueError
                end = min(end, size - 1)
            except ValueError:
                self._send_range_error(size)
                return
            partial = True
        mime = content_type or mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
        length = max(0, end - start + 1)
        self.send_response(HTTPStatus.PARTIAL_CONTENT if partial else HTTPStatus.OK)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with resolved.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining:
                chunk = handle.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _send_range_error(self, size: int) -> None:
        self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
        self.send_header("Content-Range", f"bytes */{size}")
        self.send_header("Content-Length", "0")
        self.end_headers()


__all__ = ["ReviewServer", "create_server", "save_annotation"]
