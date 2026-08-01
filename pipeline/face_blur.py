#!/usr/bin/env python3
"""Fail-closed blur-all processing for derived review media."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
from typing import Any, Callable
import uuid

import cv2
import numpy as np


MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
YUNET_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
HEAD_KPTS = (0, 1, 2, 3, 4)
DEFAULT_KPT_CONFIDENCE = 0.30
MIN_HEAD_RADIUS = 12
SOURCE_DETECTION_BATCH = 8
MAX_REPAIR_PASSES = 4


@dataclass(frozen=True)
class BlurReport:
    total_frames: int = 0
    first_pass_candidates: int = 0
    second_pass_candidates: int = 0
    privacy_verified: bool = False
    failure_reason: str | None = None

    def to_manifest(self, relative_path: str, media_type: str) -> dict[str, Any]:
        return {
            "relative_path": relative_path,
            "media_type": media_type,
            "total_frames": self.total_frames,
            "first_pass_candidates": self.first_pass_candidates,
            "second_pass_candidates": self.second_pass_candidates,
            "privacy_verified": self.privacy_verified,
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True)
class SourceCandidateIndex:
    fps: float
    width: int
    height: int
    regions_by_frame: tuple[tuple[tuple[int, int, int], ...], ...]

    @property
    def total_frames(self) -> int:
        return len(self.regions_by_frame)

    @property
    def candidate_count(self) -> int:
        return sum(len(regions) for regions in self.regions_by_frame)


class PrivacyVerificationError(ValueError):
    pass


def _head_region(
    keypoints: np.ndarray,
    confidence: float = DEFAULT_KPT_CONFIDENCE,
    minimum_radius: int = MIN_HEAD_RADIUS,
) -> tuple[int, int, int] | None:
    points = [
        (keypoints[index][0], keypoints[index][1])
        for index in HEAD_KPTS
        if keypoints[index][2] >= confidence
    ]
    if not points:
        return None
    xs, ys = zip(*points)
    center_x = int(np.mean(xs))
    center_y = int(np.mean(ys))
    size = 0.0
    if keypoints[3][2] >= confidence and keypoints[4][2] >= confidence:
        size = float(np.hypot(
            keypoints[3][0] - keypoints[4][0],
            keypoints[3][1] - keypoints[4][1],
        ))
    elif keypoints[1][2] >= confidence and keypoints[2][2] >= confidence:
        size = 1.8 * float(np.hypot(
            keypoints[1][0] - keypoints[2][0],
            keypoints[1][1] - keypoints[2][1],
        ))
    radius = int(max(minimum_radius, size * 1.36))
    return center_x, center_y, radius


def _blur_ellipse(frame: np.ndarray, center_x: int, center_y: int, radius: int) -> None:
    height, width = frame.shape[:2]
    blur_radius = max(radius, int(np.ceil(radius * 1.6)))
    vertical_radius = int(np.ceil(blur_radius * 1.2))
    x0, y0 = max(0, center_x - blur_radius), max(0, center_y - vertical_radius)
    x1, y1 = min(width, center_x + blur_radius), min(height, center_y + vertical_radius)
    if x1 <= x0 or y1 <= y0:
        return
    region = frame[y0:y1, x0:x1]
    reduced_width = max(2, region.shape[1] // 24)
    reduced_height = max(2, region.shape[0] // 24)
    reduced = cv2.resize(
        region,
        (reduced_width, reduced_height),
        interpolation=cv2.INTER_AREA,
    )
    obscured = cv2.resize(
        reduced,
        (region.shape[1], region.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    maximum_kernel = min(31, min(region.shape[:2]))
    if maximum_kernel % 2 == 0:
        maximum_kernel -= 1
    if maximum_kernel >= 3:
        obscured = cv2.GaussianBlur(
            obscured,
            (maximum_kernel, maximum_kernel),
            0,
        )
    mask = np.zeros(region.shape[:2], dtype=np.uint8)
    cv2.ellipse(
        mask,
        (center_x - x0, center_y - y0),
        (blur_radius, vertical_radius),
        0,
        0,
        360,
        255,
        -1,
    )
    mean_color = np.rint(np.mean(region[mask > 0], axis=0)).astype(np.uint8)
    region[mask > 0] = obscured[mask > 0]
    core_mask = np.zeros(region.shape[:2], dtype=np.uint8)
    cv2.ellipse(
        core_mask,
        (center_x - x0, center_y - y0),
        (
            max(1, int(blur_radius * 0.80)),
            max(1, int(vertical_radius * 0.80)),
        ),
        0,
        0,
        360,
        255,
        -1,
    )
    region[core_mask > 0] = mean_color


def _region_is_obscured(
    frame: np.ndarray, center_x: int, center_y: int, radius: int
) -> bool:
    height, width = frame.shape[:2]
    half_width = max(4, int(radius * 0.55))
    half_height = max(4, int(radius * 0.65))
    x0, y0 = max(0, center_x - half_width), max(0, center_y - half_height)
    x1, y1 = min(width, center_x + half_width), min(height, center_y + half_height)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return False
    gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return sharpness <= 20.0


def _obscure_regions(
    frame: np.ndarray,
    regions: list[tuple[int, int, int]] | tuple[tuple[int, int, int], ...],
    *,
    only_unobscured: bool = False,
) -> None:
    targets = [
        region
        for region in regions
        if not only_unobscured or not _region_is_obscured(frame, *region)
    ]
    if not targets:
        return
    for center_x, center_y, radius in targets:
        _blur_ellipse(frame, center_x, center_y, radius)

    height, width = frame.shape[:2]
    core_union = np.zeros((height, width), dtype=np.uint8)
    for center_x, center_y, radius in targets:
        half_width = max(8, int(np.ceil(radius * 1.35)))
        half_height = max(8, int(np.ceil(radius * 1.50)))
        cv2.rectangle(
            core_union,
            (max(0, center_x - half_width), max(0, center_y - half_height)),
            (min(width - 1, center_x + half_width), min(height - 1, center_y + half_height)),
            255,
            -1,
        )
    pixels = core_union > 0
    if np.any(pixels):
        mean_color = np.rint(np.mean(frame[pixels], axis=0)).astype(np.uint8)
        frame[pixels] = mean_color


def _load_yunet(score_threshold: float = 0.60):
    if not YUNET_PATH.is_file():
        return None
    return cv2.FaceDetectorYN.create(
        str(YUNET_PATH),
        "",
        (320, 320),
        score_threshold=float(score_threshold),
    )


def _yunet_faces(detector, frame: np.ndarray) -> list[tuple[int, int, int]]:
    height, width = frame.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(frame)
    if faces is None:
        return []
    regions = []
    for x, y, face_width, face_height, *_rest in faces.tolist():
        face_extent = max(face_width, face_height)
        radius_scale = 0.85 if face_extent < height * 0.10 else 0.35
        regions.append(
            (
                int(x + face_width / 2),
                int(y + face_height / 2),
                int(max(MIN_HEAD_RADIUS, face_extent * radius_scale)),
            )
        )
    return regions


def _pose_heads(
    model,
    frame: np.ndarray,
    device: str,
    score_threshold: float,
) -> list[tuple[int, int, int]]:
    results = model.predict(
        frame,
        verbose=False,
        device=device,
        conf=float(score_threshold),
    )
    if not results:
        return []
    return _pose_heads_from_result(results[0], frame, score_threshold)


def _pose_heads_from_result(
    result,
    frame: np.ndarray,
    score_threshold: float,
) -> list[tuple[int, int, int]]:
    keypoints = getattr(result, "keypoints", None)
    if keypoints is None:
        return []
    raw = keypoints.data.cpu().numpy()
    minimum_radius = max(MIN_HEAD_RADIUS, int(round(frame.shape[0] / 100.0)))
    return [
        region
        for points in raw
        if (
            region := _head_region(
                points,
                score_threshold,
                minimum_radius=minimum_radius,
            )
        ) is not None
    ]


def _pose_heads_batch(
    model,
    frames: list[np.ndarray],
    device: str,
    score_threshold: float,
) -> list[list[tuple[int, int, int]]]:
    results = list(
        model.predict(
            frames,
            verbose=False,
            device=device,
            conf=float(score_threshold),
            batch=min(SOURCE_DETECTION_BATCH, len(frames)),
        )
    )
    if len(results) != len(frames):
        raise PrivacyVerificationError(
            "YOLO nije vratio rezultat za svaki kadar u batch-u"
        )
    return [
        _pose_heads_from_result(result, frame, score_threshold)
        for result, frame in zip(results, frames, strict=True)
    ]


def _deduplicate(
    regions: list[tuple[int, int, int]],
) -> list[tuple[int, int, int]]:
    selected: list[tuple[int, int, int]] = []
    for candidate in sorted(regions, key=lambda item: item[2], reverse=True):
        x, y, radius = candidate
        if any(
            np.hypot(x - other_x, y - other_y) <= 0.5 * max(radius, other_radius)
            for other_x, other_y, other_radius in selected
        ):
            continue
        selected.append(candidate)
    return selected


def _detect_candidates(
    model,
    yunet,
    frame: np.ndarray,
    device: str,
    score_threshold: float,
) -> list[tuple[int, int, int]]:
    return _deduplicate(
        _pose_heads(model, frame, device, score_threshold)
        + _yunet_faces(yunet, frame)
    )


def _default_writer(path: Path, fps: float, width: int, height: int):
    return cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )


def _capture_details(capture) -> tuple[float, int, int, int]:
    return (
        float(capture.get(cv2.CAP_PROP_FPS) or 30.0),
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    )


def build_source_candidate_index(
    model,
    input_path: Path,
    device: str,
    *,
    score_threshold: float = 0.30,
    capture_factory: Callable[[str], Any] = cv2.VideoCapture,
    yunet=None,
    candidate_detector: Callable[..., list[tuple[int, int, int]]] = _detect_candidates,
) -> SourceCandidateIndex:
    """Detect every source-frame candidate once for all privacy passes."""
    if model is None:
        raise PrivacyVerificationError("YOLO detektor nije dostupan")
    yunet = yunet if yunet is not None else _load_yunet(score_threshold)
    if yunet is None:
        raise PrivacyVerificationError("YuNet detektor nije dostupan")
    capture = capture_factory(str(input_path))
    if hasattr(capture, "isOpened") and not capture.isOpened():
        raise PrivacyVerificationError("ulazni video ne može da se dekodira")
    fps, width, height, expected_frames = _capture_details(capture)
    if width <= 0 or height <= 0:
        capture.release()
        raise PrivacyVerificationError("ulazni video nema validne dimenzije")
    regions_by_frame: list[tuple[tuple[int, int, int], ...]] = []

    def detect_batch(frames: list[np.ndarray]) -> None:
        if candidate_detector is _detect_candidates:
            pose_regions = _pose_heads_batch(
                model,
                frames,
                device,
                float(score_threshold),
            )
            for frame, pose in zip(frames, pose_regions, strict=True):
                regions_by_frame.append(
                    tuple(_deduplicate(pose + _yunet_faces(yunet, frame)))
                )
            return
        for frame in frames:
            regions_by_frame.append(
                tuple(
                    candidate_detector(
                        model,
                        yunet,
                        frame,
                        device,
                        float(score_threshold),
                    )
                )
            )

    frame_batch: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_batch.append(frame)
            if len(frame_batch) == SOURCE_DETECTION_BATCH:
                detect_batch(frame_batch)
                frame_batch = []
        if frame_batch:
            detect_batch(frame_batch)
    except Exception as exc:
        raise PrivacyVerificationError(
            f"detektor nije obradio svaki kadar: {exc}"
        ) from exc
    finally:
        capture.release()
    if not regions_by_frame or (
        expected_frames > 0 and len(regions_by_frame) != expected_frames
    ):
        raise PrivacyVerificationError("video nije dekodiran do kraja")
    return SourceCandidateIndex(
        fps=fps,
        width=width,
        height=height,
        regions_by_frame=tuple(regions_by_frame),
    )


def _candidate_index_failure(
    candidate_index: SourceCandidateIndex,
    width: int,
    height: int,
    expected_frames: int,
) -> str | None:
    if (candidate_index.width, candidate_index.height) != (width, height):
        return "indeks kandidata i video nemaju iste dimenzije"
    if expected_frames > 0 and candidate_index.total_frames != expected_frames:
        return "indeks kandidata i video nemaju isti broj kadrova"
    return None


def blur_all_faces(
    model,
    input_path: Path,
    output_path: Path,
    device: str,
    *,
    score_threshold: float = 0.60,
    only_unobscured: bool = False,
    capture_factory: Callable[[str], Any] = cv2.VideoCapture,
    writer_factory: Callable[..., Any] = _default_writer,
    yunet=None,
    candidate_detector: Callable[..., list[tuple[int, int, int]]] = _detect_candidates,
    candidate_index: SourceCandidateIndex | None = None,
) -> BlurReport:
    """Blur every candidate from both detectors; no person is exempt."""
    if candidate_index is None:
        if model is None:
            return BlurReport(failure_reason="YOLO detektor nije dostupan")
        yunet = yunet if yunet is not None else _load_yunet(score_threshold)
        if yunet is None:
            return BlurReport(failure_reason="YuNet detektor nije dostupan")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    capture = capture_factory(str(input_path))
    if hasattr(capture, "isOpened") and not capture.isOpened():
        return BlurReport(failure_reason="ulazni video ne može da se dekodira")
    fps, width, height, expected_frames = _capture_details(capture)
    if width <= 0 or height <= 0:
        capture.release()
        return BlurReport(failure_reason="ulazni video nema validne dimenzije")
    if candidate_index is not None and (
        index_failure := _candidate_index_failure(
            candidate_index, width, height, expected_frames
        )
    ):
        capture.release()
        return BlurReport(failure_reason=index_failure)
    writer = writer_factory(output_path, fps, width, height)
    if hasattr(writer, "isOpened") and not writer.isOpened():
        capture.release()
        return BlurReport(failure_reason="izlazni video ne može da se otvori")

    total_frames = 0
    candidates = 0
    failure_reason = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            regions = (
                candidate_index.regions_by_frame[total_frames]
                if candidate_index is not None
                else candidate_detector(
                    model, yunet, frame, device, float(score_threshold)
                )
            )
            candidates += len(regions)
            _obscure_regions(
                frame,
                regions,
                only_unobscured=only_unobscured,
            )
            writer.write(frame)
            total_frames += 1
    except Exception as exc:
        failure_reason = f"detektor nije obradio svaki kadar: {exc}"
    finally:
        capture.release()
        writer.release()
    if failure_reason is None and (
        total_frames == 0
        or (expected_frames > 0 and total_frames != expected_frames)
        or (candidate_index is not None and total_frames != candidate_index.total_frames)
    ):
        failure_reason = "video nije dekodiran do kraja"
    return BlurReport(
        total_frames=total_frames,
        first_pass_candidates=candidates,
        privacy_verified=False,
        failure_reason=failure_reason,
    )


def verify_blurred_clip(
    model,
    input_path: Path,
    device: str,
    *,
    score_threshold: float = 0.30,
    capture_factory: Callable[[str], Any] = cv2.VideoCapture,
    yunet=None,
    candidate_detector: Callable[..., list[tuple[int, int, int]]] = _detect_candidates,
) -> BlurReport:
    """Verify that both detectors process every frame and find no candidate."""
    if model is None:
        return BlurReport(failure_reason="YOLO detektor nije dostupan")
    yunet = yunet if yunet is not None else _load_yunet(score_threshold)
    if yunet is None:
        return BlurReport(failure_reason="YuNet detektor nije dostupan")
    capture = capture_factory(str(input_path))
    if hasattr(capture, "isOpened") and not capture.isOpened():
        return BlurReport(failure_reason="video ne može da se dekodira")
    _, _, _, expected_frames = _capture_details(capture)
    total_frames = 0
    candidates = 0
    failure_reason = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            regions = candidate_detector(
                model, yunet, frame, device, float(score_threshold)
            )
            candidates += sum(
                not _region_is_obscured(frame, center_x, center_y, radius)
                for center_x, center_y, radius in regions
            )
            total_frames += 1
    except Exception as exc:
        failure_reason = f"detektor nije obradio svaki kadar: {exc}"
    finally:
        capture.release()
    if failure_reason is None and (
        total_frames == 0
        or (expected_frames > 0 and total_frames != expected_frames)
    ):
        failure_reason = "video nije dekodiran do kraja"
    verified = failure_reason is None and candidates == 0
    return BlurReport(
        total_frames=total_frames,
        second_pass_candidates=candidates,
        privacy_verified=verified,
        failure_reason=failure_reason,
    )


def verify_blurred_against_source(
    model,
    source_path: Path,
    private_path: Path,
    device: str,
    *,
    score_threshold: float = 0.30,
    source_capture_factory: Callable[[str], Any] = cv2.VideoCapture,
    private_capture_factory: Callable[[str], Any] = cv2.VideoCapture,
    yunet=None,
    candidate_detector: Callable[..., list[tuple[int, int, int]]] = _detect_candidates,
    candidate_index: SourceCandidateIndex | None = None,
) -> BlurReport:
    """Require every low-threshold source candidate to be obscured in output."""
    if candidate_index is None:
        if model is None:
            return BlurReport(failure_reason="YOLO detektor nije dostupan")
        yunet = yunet if yunet is not None else _load_yunet(score_threshold)
        if yunet is None:
            return BlurReport(failure_reason="YuNet detektor nije dostupan")
    source = source_capture_factory(str(source_path))
    private = private_capture_factory(str(private_path))
    if (
        hasattr(source, "isOpened")
        and not source.isOpened()
        or hasattr(private, "isOpened")
        and not private.isOpened()
    ):
        source.release()
        private.release()
        return BlurReport(failure_reason="izvorni ili privatni video ne može da se dekodira")
    _, source_width, source_height, source_expected = _capture_details(source)
    _, private_width, private_height, private_expected = _capture_details(private)
    if (
        source_width <= 0
        or source_height <= 0
        or (source_width, source_height) != (private_width, private_height)
    ):
        source.release()
        private.release()
        return BlurReport(failure_reason="izvorni i privatni video nemaju iste dimenzije")
    if candidate_index is not None and (
        index_failure := _candidate_index_failure(
            candidate_index, source_width, source_height, source_expected
        )
    ):
        source.release()
        private.release()
        return BlurReport(failure_reason=index_failure)

    total_frames = 0
    candidates = 0
    failure_reason = None
    try:
        while True:
            source_ok, source_frame = source.read()
            private_ok, private_frame = private.read()
            if not source_ok or not private_ok:
                if source_ok != private_ok:
                    failure_reason = "izvorni i privatni video nemaju isti broj kadrova"
                break
            regions = (
                candidate_index.regions_by_frame[total_frames]
                if candidate_index is not None
                else candidate_detector(
                    model,
                    yunet,
                    source_frame,
                    device,
                    float(score_threshold),
                )
            )
            candidates += sum(
                not _region_is_obscured(private_frame, center_x, center_y, radius)
                for center_x, center_y, radius in regions
            )
            total_frames += 1
    except Exception as exc:
        failure_reason = f"detektor nije uporedio svaki kadar: {exc}"
    finally:
        source.release()
        private.release()
    if failure_reason is None and (
        total_frames == 0
        or (source_expected > 0 and total_frames != source_expected)
        or (private_expected > 0 and total_frames != private_expected)
        or (candidate_index is not None and total_frames != candidate_index.total_frames)
    ):
        failure_reason = "izvorni ili privatni video nije dekodiran do kraja"
    verified = failure_reason is None and candidates == 0
    return BlurReport(
        total_frames=total_frames,
        second_pass_candidates=candidates,
        privacy_verified=verified,
        failure_reason=failure_reason,
    )


def blur_from_reference(
    model,
    source_path: Path,
    private_input_path: Path,
    output_path: Path,
    device: str,
    *,
    score_threshold: float = 0.30,
    source_capture_factory: Callable[[str], Any] = cv2.VideoCapture,
    private_capture_factory: Callable[[str], Any] = cv2.VideoCapture,
    writer_factory: Callable[..., Any] = _default_writer,
    yunet=None,
    candidate_detector: Callable[..., list[tuple[int, int, int]]] = _detect_candidates,
    candidate_index: SourceCandidateIndex | None = None,
) -> BlurReport:
    """Repair output only at low-threshold regions detected in the source."""
    if candidate_index is None:
        if model is None:
            return BlurReport(failure_reason="YOLO detektor nije dostupan")
        yunet = yunet if yunet is not None else _load_yunet(score_threshold)
        if yunet is None:
            return BlurReport(failure_reason="YuNet detektor nije dostupan")
    source = source_capture_factory(str(source_path))
    private = private_capture_factory(str(private_input_path))
    if (
        hasattr(source, "isOpened")
        and not source.isOpened()
        or hasattr(private, "isOpened")
        and not private.isOpened()
    ):
        source.release()
        private.release()
        return BlurReport(failure_reason="izvorni ili privatni video ne može da se dekodira")
    fps, width, height, private_expected = _capture_details(private)
    _, source_width, source_height, source_expected = _capture_details(source)
    if width <= 0 or height <= 0 or (width, height) != (source_width, source_height):
        source.release()
        private.release()
        return BlurReport(failure_reason="izvorni i privatni video nemaju iste dimenzije")
    if candidate_index is not None and (
        index_failure := _candidate_index_failure(
            candidate_index, source_width, source_height, source_expected
        )
    ):
        source.release()
        private.release()
        return BlurReport(failure_reason=index_failure)
    writer = writer_factory(Path(output_path), fps, width, height)
    if hasattr(writer, "isOpened") and not writer.isOpened():
        source.release()
        private.release()
        return BlurReport(failure_reason="repair video ne može da se otvori")

    total_frames = 0
    candidates = 0
    failure_reason = None
    try:
        while True:
            source_ok, source_frame = source.read()
            private_ok, private_frame = private.read()
            if not source_ok or not private_ok:
                if source_ok != private_ok:
                    failure_reason = "izvorni i privatni video nemaju isti broj kadrova"
                break
            regions = (
                candidate_index.regions_by_frame[total_frames]
                if candidate_index is not None
                else candidate_detector(
                    model,
                    yunet,
                    source_frame,
                    device,
                    float(score_threshold),
                )
            )
            candidates += len(regions)
            _obscure_regions(
                private_frame,
                regions,
                only_unobscured=True,
            )
            writer.write(private_frame)
            total_frames += 1
    except Exception as exc:
        failure_reason = f"detektor nije popravio svaki kadar: {exc}"
    finally:
        source.release()
        private.release()
        writer.release()
    if failure_reason is None and (
        total_frames == 0
        or (source_expected > 0 and total_frames != source_expected)
        or (private_expected > 0 and total_frames != private_expected)
        or (candidate_index is not None and total_frames != candidate_index.total_frames)
    ):
        failure_reason = "repair video nije dekodiran do kraja"
    return BlurReport(
        total_frames=total_frames,
        first_pass_candidates=candidates,
        privacy_verified=False,
        failure_reason=failure_reason,
    )


def _mux_original_audio(video_path: Path, audio_source: Path, output_path: Path) -> Path:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_source),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(command, check=True, capture_output=True)
    return output_path


def privatize_media(
    model,
    raw_path: Path,
    output_path: Path,
    device: str,
    *,
    blur_fn: Callable[..., BlurReport] = blur_all_faces,
    repair_fn: Callable[..., BlurReport] = blur_from_reference,
    verify_fn: Callable[..., BlurReport] = verify_blurred_against_source,
    audio_muxer: Callable[[Path, Path, Path], Path] = _mux_original_audio,
) -> BlurReport:
    """Atomically publish media only after two-detector verification succeeds."""
    raw_path = Path(raw_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    first = output_path.with_name(f".{output_path.stem}.{token}.blur1.mp4")
    muxed = output_path.with_name(f".{output_path.stem}.{token}.private.mp4")
    repairs: list[Path] = []
    try:
        candidate_index = None
        use_candidate_index = (
            blur_fn is blur_all_faces
            and repair_fn is blur_from_reference
            and verify_fn is verify_blurred_against_source
        )
        if use_candidate_index:
            candidate_index = build_source_candidate_index(
                model,
                raw_path,
                device,
                score_threshold=0.30,
            )
        index_options = (
            {"candidate_index": candidate_index}
            if candidate_index is not None
            else {}
        )
        first_report = blur_fn(
            model,
            raw_path,
            first,
            device,
            score_threshold=0.30,
            **index_options,
        )
        if first_report.failure_reason is not None or not first.is_file():
            raise PrivacyVerificationError(
                first_report.failure_reason or "prvi blur prolaz nije napravio video"
            )
        residual = verify_fn(
            model,
            raw_path,
            first,
            device,
            score_threshold=0.30,
            **index_options,
        )
        candidate = first
        repaired_candidates = 0
        if residual.failure_reason is not None:
            raise PrivacyVerificationError(residual.failure_reason)
        for repair_index in range(1, MAX_REPAIR_PASSES + 1):
            if residual.second_pass_candidates == 0:
                break
            repaired_candidates += residual.second_pass_candidates
            repaired = output_path.with_name(
                f".{output_path.stem}.{token}.blur{repair_index + 1}.mp4"
            )
            repairs.append(repaired)
            repair_report = repair_fn(
                model,
                raw_path,
                candidate,
                repaired,
                device,
                score_threshold=0.30,
                **index_options,
            )
            if repair_report.failure_reason is not None or not repaired.is_file():
                raise PrivacyVerificationError(
                    repair_report.failure_reason or "drugi blur prolaz nije napravio video"
                )
            candidate = repaired
            residual = verify_fn(
                model,
                raw_path,
                candidate,
                device,
                score_threshold=0.30,
                **index_options,
            )
            if residual.failure_reason is not None:
                raise PrivacyVerificationError(residual.failure_reason)
        if residual.second_pass_candidates:
            raise PrivacyVerificationError(
                f"posle dodatnog zamagljivanja ostalo je "
                f"{residual.second_pass_candidates} oštrih kandidata"
            )
        audio_muxer(candidate, raw_path, muxed)
        if not muxed.is_file() or muxed.stat().st_size <= 0:
            raise PrivacyVerificationError("privatni video nije napravljen")
        final = verify_fn(
            model,
            raw_path,
            muxed,
            device,
            score_threshold=0.30,
            **index_options,
        )
        if not final.privacy_verified:
            raise PrivacyVerificationError(
                final.failure_reason
                or f"poslednja provera je našla {final.second_pass_candidates} kandidata"
            )
        os.replace(muxed, output_path)
        return BlurReport(
            total_frames=final.total_frames,
            first_pass_candidates=first_report.first_pass_candidates,
            second_pass_candidates=repaired_candidates,
            privacy_verified=True,
            failure_reason=None,
        )
    finally:
        for temporary in (first, *repairs, muxed):
            temporary.unlink(missing_ok=True)


def build_privacy_processor(
    model_path: Path | str = "yolo11x-pose.pt",
    device: str = "mps",
) -> Callable[[Path, Path], BlurReport]:
    """Create a lazy, reusable processor so one YOLO model serves all exports."""
    model = None

    def process(raw_path: Path, output_path: Path) -> BlurReport:
        nonlocal model
        if model is None:
            from ultralytics import YOLO

            model = YOLO(str(model_path))
        return privatize_media(model, raw_path, output_path, device)

    return process


__all__ = [
    "BlurReport",
    "PrivacyVerificationError",
    "SourceCandidateIndex",
    "blur_all_faces",
    "blur_from_reference",
    "build_source_candidate_index",
    "build_privacy_processor",
    "privatize_media",
    "verify_blurred_against_source",
    "verify_blurred_clip",
]
