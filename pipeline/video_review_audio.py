"""Audio candidates used by the video-only review workflow."""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


ENVELOPE_SR = 8000


@dataclass(frozen=True)
class TapTripletCandidate:
    """A review-only candidate; confirmation is always an explicit user action."""

    peaks_s: tuple[float, float, float]
    confidence: float
    user_confirmed: bool = False


def find_tap_triplet_candidates(
    peak_times_s: Sequence[float],
    min_gap_s: float = 0.12,
    max_gap_s: float = 0.85,
    max_span_s: float = 1.4,
) -> list[tuple[float, float, float]]:
    """Return overlapping ordered groups of three tap-like peaks."""
    ordered = sorted(float(value) for value in peak_times_s)
    candidates = []
    for first, second, third in zip(ordered, ordered[1:], ordered[2:]):
        if (
            min_gap_s <= second - first <= max_gap_s
            and min_gap_s <= third - second <= max_gap_s
            and third - first <= max_span_s
        ):
            candidates.append((first, second, third))
    return candidates


def detect_tap_triplet_candidates(
    video_path: Path,
    start_s: float = 0.0,
    end_s: float | None = None,
    sr: int = ENVELOPE_SR,
    min_gap_s: float = 0.12,
) -> list[tuple[float, float, float]]:
    """Extract transient peaks from a selected video range and group triplets."""
    from pipeline.audio_sync import extract_audio_envelope
    from scipy import signal

    if start_s < 0 or (end_s is not None and end_s < start_s):
        raise ValueError("selected audio range is invalid")

    envelope = extract_audio_envelope(video_path, sr)
    start_index = int(start_s * sr)
    end_index = len(envelope) if end_s is None else min(len(envelope), int(end_s * sr))
    selected = envelope[start_index:end_index]
    if selected.size == 0:
        return []

    distance = max(1, int(min_gap_s * sr))
    prominence = float(selected.std() * 2)
    peaks, properties = signal.find_peaks(selected, distance=distance, prominence=prominence)
    del properties
    peak_times = [start_s + float(index) / sr for index in peaks]
    return find_tap_triplet_candidates(peak_times)


def candidate_details(
    peak_times_s: Sequence[float],
    confidence: float = 0.0,
) -> list[TapTripletCandidate]:
    """Attach review metadata without turning candidates into confirmed anchors."""
    return [
        TapTripletCandidate(peaks_s=peaks, confidence=float(confidence))
        for peaks in find_tap_triplet_candidates(peak_times_s)
    ]
