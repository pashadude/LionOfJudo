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
    peak_times = [start_s + float(index) / sr for index in peaks]
    details = candidate_details(peak_times, properties["prominences"], min_gap_s=min_gap_s)
    return [candidate.peaks_s for candidate in details]


def detect_tap_triplet_candidate_details(
    video_path: Path,
    start_s: float = 0.0,
    end_s: float | None = None,
    sr: int = ENVELOPE_SR,
    min_gap_s: float = 0.12,
) -> list[TapTripletCandidate]:
    """Extract triplets with prominence confidence and unconfirmed state."""
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
    peak_times = [start_s + float(index) / sr for index in peaks]
    return candidate_details(peak_times, properties["prominences"], min_gap_s=min_gap_s)


def candidate_details(
    peak_times_s: Sequence[float],
    peak_prominences: Sequence[float] | None = None,
    min_gap_s: float = 0.12,
    max_gap_s: float = 0.85,
    max_span_s: float = 1.4,
) -> list[TapTripletCandidate]:
    """Attach normalized prominence confidence without confirming candidates."""
    ordered_times = sorted(float(value) for value in peak_times_s)
    if peak_prominences is None:
        ordered_prominences = [0.0] * len(ordered_times)
    else:
        if len(peak_prominences) != len(peak_times_s):
            raise ValueError("peak prominences must match peak times")
        ordered_prominences = [
            float(prominence)
            for _, prominence in sorted(zip(peak_times_s, peak_prominences))
        ]
    scale = max(ordered_prominences, default=0.0)
    candidates: list[TapTripletCandidate] = []
    for index, (first, second, third) in enumerate(
        zip(ordered_times, ordered_times[1:], ordered_times[2:])
    ):
        if (
            min_gap_s <= second - first <= max_gap_s
            and min_gap_s <= third - second <= max_gap_s
            and third - first <= max_span_s
        ):
            triplet_confidence = (
                sum(ordered_prominences[index : index + 3]) / 3 / scale
                if scale > 0
                else 0.0
            )
            candidates.append(
                TapTripletCandidate(
                    peaks_s=(first, second, third),
                    confidence=triplet_confidence,
                )
            )
    return candidates
