"""Immutable review generations selected by one atomic pointer."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Mapping
import uuid
import warnings

from pipeline.video_review_storage import atomic_write_json, atomic_write_review


GENERATION_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
MEDIA_DIRECTORIES = ("media", "events", "previews")


@dataclass(frozen=True)
class GenerationSnapshot:
    generation_id: str | None
    root: Path

    @property
    def review_path(self) -> Path:
        return self.root / "review.json"

    @property
    def event_metrics_path(self) -> Path:
        return self.root / "analysis" / "event_metrics.json"

    @property
    def csv_path(self) -> Path:
        return self.root / "izvestaj.csv"

    @property
    def markdown_path(self) -> Path:
        return self.root / "izvestaj.md"


class GenerationStore:
    def __init__(self, session_dir: Path) -> None:
        self.session_dir = Path(session_dir).expanduser().resolve()
        self.generations_dir = self.session_dir / ".review-generations"
        self.pointer_path = self.session_dir / "current-generation.json"

    def resolve_current(self) -> GenerationSnapshot:
        if not self.pointer_path.exists():
            snapshot = GenerationSnapshot(None, self.session_dir)
            if not snapshot.review_path.is_file():
                raise ValueError("sesija nema review.json")
            return snapshot
        try:
            pointer = json.loads(self.pointer_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("current-generation.json nije validan") from exc
        generation_id = pointer.get("generation_id") if isinstance(pointer, dict) else None
        if not isinstance(generation_id, str) or GENERATION_ID_PATTERN.fullmatch(generation_id) is None:
            raise ValueError("current-generation.json nema validan generation_id")
        root = self.generations_dir / generation_id
        snapshot = GenerationSnapshot(generation_id, root)
        required = (
            snapshot.review_path,
            snapshot.event_metrics_path,
            snapshot.csv_path,
            snapshot.markdown_path,
        )
        if not root.is_dir() or not all(path.is_file() for path in required):
            raise ValueError("aktivna generacija nije kompletna")
        return snapshot

    def stage_and_activate(
        self,
        review: Mapping[str, Any],
        event_metrics: list[Mapping[str, Any]],
        csv_text: str,
        markdown_text: str,
        staged_media: Mapping[str, Path] | None = None,
        removed_media_prefixes: tuple[str, ...] = (),
    ) -> GenerationSnapshot:
        review_events = review.get("events")
        if not isinstance(review_events, list) or not all(
            isinstance(event, Mapping) for event in review_events
        ):
            raise ValueError("review events mora biti lista JSON objekata")
        normalized_metrics = [dict(event) for event in event_metrics]
        if normalized_metrics != review_events:
            raise ValueError("event_metrics mora tačno odgovarati review.events")
        removed_prefixes = tuple(
            self._safe_relative(value) for value in removed_media_prefixes
        )
        current = self.resolve_current()
        generation_id = uuid.uuid4().hex
        target = self.generations_dir / generation_id
        pointer_tmp = self.pointer_path.with_name("current-generation.json.tmp")
        pointer_switched = False
        self.generations_dir.mkdir(parents=True, exist_ok=True)
        target.mkdir()
        try:
            self._link_current_media(current.root, target, removed_prefixes)
            self._copy_staged_media(target, staged_media or {})
            atomic_write_review(target / "review.json", review)
            (target / "analysis").mkdir(parents=True, exist_ok=True)
            atomic_write_json(
                target / "analysis" / "event_metrics.json",
                {"events": normalized_metrics},
            )
            self._write_text(target / "izvestaj.csv", csv_text)
            self._write_text(target / "izvestaj.md", markdown_text)
            self._fsync_directory(target / "analysis")
            self._fsync_directory(target)
            self._fsync_directory(self.generations_dir)
            self._write_pointer(pointer_tmp, generation_id)
            os.replace(pointer_tmp, self.pointer_path)
            pointer_switched = True
            try:
                self._fsync_directory(self.session_dir)
            except OSError as exc:
                warnings.warn(
                    f"generacija je objavljena, ali fsync direktorijuma nije uspeo: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return GenerationSnapshot(generation_id, target)
        except Exception:
            pointer_tmp.unlink(missing_ok=True)
            if not pointer_switched:
                shutil.rmtree(target, ignore_errors=True)
            raise

    @staticmethod
    def _write_text(path: Path, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError("izveštaj mora biti tekst")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _write_pointer(path: Path, generation_id: str) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump({"generation_id": generation_id}, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _safe_relative(value: str) -> Path:
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("staged media putanja nije validna")
        return relative

    def _copy_staged_media(
        self, target: Path, staged_media: Mapping[str, Path]
    ) -> None:
        for relative_value, source_value in staged_media.items():
            relative = self._safe_relative(relative_value)
            source = Path(source_value)
            if not source.is_file():
                raise ValueError("staged media fajl ne postoji")
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.unlink(missing_ok=True)
            shutil.copy2(source, destination)
            self._fsync_file(destination)

    @staticmethod
    def _fsync_file(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _link_current_media(
        source_root: Path, target: Path, removed_prefixes: tuple[Path, ...] = ()
    ) -> None:
        for directory_name in MEDIA_DIRECTORIES:
            source_directory = source_root / directory_name
            if not source_directory.is_dir():
                continue
            for source in source_directory.rglob("*"):
                if not source.is_file():
                    continue
                relative = Path(directory_name) / source.relative_to(source_directory)
                if any(
                    relative == prefix or relative.is_relative_to(prefix)
                    for prefix in removed_prefixes
                ):
                    continue
                destination = target / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copy2(source, destination)
                    GenerationStore._fsync_file(destination)


__all__ = ["GenerationSnapshot", "GenerationStore"]
