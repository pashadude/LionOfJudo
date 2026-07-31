"""Local-only HTTP server for reviewing one imported video session."""

from __future__ import annotations

import csv
import json
import mimetypes
import os
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import unquote, urlsplit

from pipeline.video_review_contract import (
    AnchorPair,
    ReviewEvent,
    ReviewSession,
    validate_review_session,
)
from pipeline.video_sync import fit_time_map


ANNOTATION_FIELDS = {"potvrdena_tehnika", "ocena", "napomena"}
MAX_EVENT_ID_LENGTH = 128
MAX_TECHNIQUE_LENGTH = 120
MAX_NOTE_LENGTH = 2000
REPORT_FIELDS = (
    "event_id",
    "Vreme početka (s)",
    "Vreme kraja (s)",
    "Predlog tehnike",
    "Potvrđena tehnika",
    "Glasovna fraza",
    "Pouzdanost glasa",
    "Brzina ulaska norm",
    "Rotacija trupa 2D (step/s)",
    "Promena visine kukova norm",
    "Vreme oporavka (s)",
    "Intenzitet pokreta (0-100)",
    "Ocena",
    "Napomena",
    "Status",
    "Isključeno iz statistike",
)
METRIC_KEYS = (
    "brzina_ulaska_norm",
    "rotacija_trupa_2d_dps",
    "promena_visine_kukova_norm",
    "vreme_oporavka_s",
    "intenzitet_pokreta_0_100",
)


def _strict_load(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"JSON konstanta nije dozvoljena: {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("review.json nije validan strogi JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("review.json mora sadržati JSON objekat")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                dict(payload),
                handle,
                ensure_ascii=False,
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


def _event_is_injury(event: Mapping[str, Any]) -> bool:
    return bool(
        event.get("prijavljen_povredni_dogadjaj")
        or event.get("iskljuceno_iz_statistike")
        or event.get("status") == "povreda"
    )


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
    if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
        raise ValueError("ocena mora biti ceo broj od 1 do 5")
    return {
        "potvrdena_tehnika": technique,
        "ocena": score,
        "napomena": note,
    }


def _report_row(event: Mapping[str, Any]) -> dict[str, Any]:
    status_parts = []
    if _event_is_injury(event):
        status_parts.append("Prijavljen povredni događaj")
    if str(event.get("vidljivost", "")).lower() == "nedovoljno vidljivo":
        status_parts.append("Nedovoljno vidljivo")
    return {
        "event_id": event.get("event_id", ""),
        "Vreme početka (s)": event.get("sony_start_s", ""),
        "Vreme kraja (s)": event.get("sony_end_s", ""),
        "Predlog tehnike": event.get("predlog_tehnike", ""),
        "Potvrđena tehnika": event.get("potvrdena_tehnika", ""),
        "Glasovna fraza": event.get("glasovna_fraza", ""),
        "Pouzdanost glasa": event.get("pouzdanost_glasa", ""),
        "Brzina ulaska norm": event.get("brzina_ulaska_norm", ""),
        "Rotacija trupa 2D (step/s)": event.get("rotacija_trupa_2d_dps", ""),
        "Promena visine kukova norm": event.get("promena_visine_kukova_norm", ""),
        "Vreme oporavka (s)": event.get("vreme_oporavka_s", ""),
        "Intenzitet pokreta (0-100)": event.get("intenzitet_pokreta_0_100", ""),
        "Ocena": event.get("ocena", ""),
        "Napomena": event.get("napomena", ""),
        "Status": "; ".join(status_parts) or event.get("status", ""),
        "Isključeno iz statistike": "da" if _event_is_injury(event) else "ne",
    }


def _write_report(path: Path, review: Mapping[str, Any]) -> None:
    events = review.get("events", [])
    if not isinstance(events, list):
        raise ValueError("events mora biti JSON lista")
    csv_path = path.with_name("izvestaj.csv")
    csv_tmp = csv_path.with_name("izvestaj.csv.tmp")
    try:
        with csv_tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS, extrasaction="raise")
            writer.writeheader()
            writer.writerows(_report_row(event) for event in events if isinstance(event, dict))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(csv_tmp, csv_path)
    finally:
        csv_tmp.unlink(missing_ok=True)

    markdown_path = path.with_name("izvestaj.md")
    markdown_tmp = markdown_path.with_name("izvestaj.md.tmp")
    def md(value: object) -> str:
        return str(value if value is not None else "").replace("|", "\\|").replace("\n", "<br>")

    try:
        with markdown_tmp.open("w", encoding="utf-8") as handle:
            normal_count = sum(
                1 for event in events if isinstance(event, dict) and not _event_is_injury(event)
            )
            handle.write("# Izveštaj trenerskog pregleda\n\n")
            handle.write(f"Normalni događaji u statistici: {normal_count}\n\n")
            handle.write("| " + " | ".join(REPORT_FIELDS) + " |\n")
            handle.write("| " + " | ".join("---" for _ in REPORT_FIELDS) + " |\n")
            for event in events:
                if isinstance(event, dict):
                    row = _report_row(event)
                    handle.write("| " + " | ".join(md(row[field]) for field in REPORT_FIELDS) + " |\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(markdown_tmp, markdown_path)
    finally:
        markdown_tmp.unlink(missing_ok=True)


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


class ReviewServer:
    """Lifecycle wrapper around a loopback-only review HTTP server."""

    def __init__(self, session_dir: Path, port: int):
        self.session_dir = Path(session_dir).expanduser().resolve()
        if not self.session_dir.is_dir():
            raise ValueError("direktorijum sesije ne postoji")
        self.review_path = self.session_dir / "review.json"
        if not self.review_path.is_file():
            raise ValueError("sesija nema review.json")
        _write_report(self.review_path, _strict_load(self.review_path))
        self.static_dir = Path(__file__).resolve().parent / "static"
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


def create_server(session_dir: Path, port: int = 8765) -> ReviewServer:
    return ReviewServer(Path(session_dir), port)


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
            if path in {"/", "/index.html"}:
                self._serve_file(self.app.static_dir / "index.html", self.app.static_dir, "text/html; charset=utf-8")
            elif path.startswith("/static/"):
                self._serve_file(self.app.static_dir / path.removeprefix("/static/"), self.app.static_dir)
            elif path == "/api/session":
                self._send_json(HTTPStatus.OK, _strict_load(self.app.review_path))
            elif path.startswith("/api/events/"):
                event_id = path.removeprefix("/api/events/")
                if not event_id or "/" in event_id:
                    raise FileNotFoundError
                review = _strict_load(self.app.review_path)
                event = next(
                    (event for event in review.get("events", []) if isinstance(event, dict) and event.get("event_id") == event_id),
                    None,
                )
                if event is None:
                    raise FileNotFoundError
                self._send_json(HTTPStatus.OK, event)
            elif path.startswith("/media/"):
                relative = path.removeprefix("/media/")
                if ".." in Path(relative).parts:
                    raise FileNotFoundError
                if relative.startswith(("events/", "previews/", "analysis/")):
                    candidate = self.app.session_dir / relative
                else:
                    candidate = self.app.session_dir / "media" / relative
                self._serve_file(candidate, self.app.session_dir)
            else:
                relative = path.removeprefix("/")
                self._serve_file(self.app.session_dir / relative, self.app.session_dir)
        except FileNotFoundError:
            self._error(HTTPStatus.NOT_FOUND, "resurs nije pronađen")
        except ValueError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except PermissionError:
            self._error(HTTPStatus.NOT_FOUND, "resurs nije pronađen")

    def do_PUT(self) -> None:
        path = unquote(urlsplit(self.path).path)
        prefix = "/api/events/"
        suffix = "/annotation"
        if not path.startswith(prefix) or not path.endswith(suffix):
            self._error(HTTPStatus.NOT_FOUND, "resurs nije pronađen")
            return
        event_id = path[len(prefix):-len(suffix)]
        if not event_id or "/" in event_id:
            self._error(HTTPStatus.NOT_FOUND, "resurs nije pronađen")
            return
        try:
            payload = self._read_body()
            self._send_json(HTTPStatus.OK, save_annotation(self.app.review_path, event_id, payload))  # type: ignore[arg-type]
        except (ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))

    def do_POST(self) -> None:
        path = unquote(urlsplit(self.path).path)
        if path != "/api/session/sync":
            self._error(HTTPStatus.NOT_FOUND, "resurs nije pronađen")
            return
        try:
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
            review = _strict_load(self.app.review_path)
            injury_events = [
                event for event in review.get("events", [])
                if isinstance(event, dict) and _event_is_injury(event)
            ]
            if any(float(cutoff) > float(event.get("sony_end_s", cutoff)) for event in injury_events):
                raise ValueError("presek povrede ne može biti posle povrednog događaja")
            _review_session_for_sync(review, anchors, float(cutoff))
            review["anchors"] = [anchor.to_dict() for anchor in anchors]
            review["time_map"] = time_map.to_dict()
            review["injury_cutoff_s"] = float(cutoff)
            _atomic_json(self.app.review_path, review)
            _write_report(self.app.review_path, review)
            self._send_json(HTTPStatus.OK, review)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
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
