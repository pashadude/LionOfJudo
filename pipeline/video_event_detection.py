"""Blue-athlete track helpers and transparent movement-window proposals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .video_pose_metrics import FrameMetric, json_safe


DEFAULT_CONFIDENCE = 0.3
DEFAULT_WINDOW_EXPANSION_S = 1.0
DEFAULT_MERGE_GAP_S = 1.5
NEDOVOLJNO_VIDLJIVO = "nedovoljno_vidljivo"


def _bbox(value: Any) -> tuple[float, float, float, float]:
    if isinstance(value, Mapping):
        value = value.get("bbox", value.get("box", value.get("xyxy")))
        if value is None:
            raise ValueError("detection must contain bbox, box, or xyxy")
    elif hasattr(value, "bbox"):
        value = value.bbox
    values = tuple(float(item) for item in value)
    if len(values) != 4 or values[2] <= values[0] or values[3] <= values[1]:
        raise ValueError("bbox must be (x1, y1, x2, y2) with positive area")
    return values


def bbox_iou(first: Any, second: Any) -> float:
    """Return IoU for two ``(x1, y1, x2, y2)`` boxes."""
    ax1, ay1, ax2, ay2 = _bbox(first)
    bx1, by1, bx2, by2 = _bbox(second)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return float(intersection / union) if union else 0.0


iou = bbox_iou


def select_blue_detection(
    detections: Sequence[Any], seed_bbox: Any
) -> Any | None:
    """Select the detection with the largest positive seed-box IoU."""
    scored = [(bbox_iou(detection, seed_bbox), detection)
              for detection in detections]
    scored = [item for item in scored if item[0] > 0.0]
    return max(scored, key=lambda item: item[0])[1] if scored else None


def select_blue_track_id(detections: Sequence[Any], seed_bbox: Any) -> Any | None:
    selected = select_blue_detection(detections, seed_bbox)
    if selected is None:
        return None
    if isinstance(selected, Mapping):
        return selected.get("track_id")
    return getattr(selected, "track_id", None)


select_seed_track = select_blue_detection


def _candidate_value(candidate: Any, key: str, default: Any = None) -> Any:
    if isinstance(candidate, Mapping):
        return candidate.get(key, default)
    return getattr(candidate, key, default)


def _blue_fraction(frame: np.ndarray, box: Any) -> float:
    """Estimate the fraction of a central torso patch that is blue in HSV."""
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] < 3:
        return 0.0
    x1, y1, x2, y2 = _bbox(box)
    height, width = image.shape[:2]
    x1, x2 = max(0, int(x1)), min(width, int(x2))
    y1, y2 = max(0, int(y1)), min(height, int(y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    patch = image[y1 + (y2 - y1) // 5:y1 + 4 * (y2 - y1) // 5,
                  x1 + (x2 - x1) // 5:x1 + 4 * (x2 - x1) // 5, :3]
    if patch.size == 0:
        return 0.0
    bgr = patch.astype(float) / 255.0
    blue, green, red = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    maximum = np.max(bgr, axis=-1)
    minimum = np.min(bgr, axis=-1)
    delta = maximum - minimum
    saturation = np.divide(delta, maximum, out=np.zeros_like(delta), where=maximum > 0)
    hue = np.zeros_like(maximum)
    nonzero = delta > 1e-9
    blue_max = nonzero & (maximum == blue)
    hue[blue_max] = 60.0 * ((red[blue_max] - green[blue_max]) / delta[blue_max] + 4.0)
    green_max = nonzero & (maximum == green)
    hue[green_max] = 60.0 * ((blue[green_max] - red[green_max]) / delta[green_max] + 2.0)
    red_max = nonzero & (maximum == red)
    hue[red_max] = 60.0 * ((green[red_max] - blue[red_max]) / delta[red_max])
    hue %= 360.0
    blue_pixels = ((hue >= 190.0) & (hue <= 250.0)
                   & (saturation >= 0.35) & (maximum >= 0.15))
    return float(np.mean(blue_pixels))


def blue_dominant_torso(frame: np.ndarray, bbox: Any, minimum_fraction: float = 0.35) -> bool:
    return _blue_fraction(frame, bbox) >= float(minimum_fraction)


blue_dominant = blue_dominant_torso


def recover_blue_pose(
    candidates: Sequence[Any],
    previous_bbox: Any,
    frame: np.ndarray | None = None,
    previous_track_id: Any | None = None,
    minimum_blue_fraction: float = 0.35,
) -> Any | None:
    """Recover a post-occlusion pose using proximity and a blue torso cue.

    Recovery requires an image so the blue cue is measured from its HSV torso
    patch. Returning ``None`` is the ``nedovoljno_vidljivo`` outcome for the
    caller when evidence is unavailable or incompatible.
    """
    previous = _bbox(previous_bbox)
    previous_center = ((previous[0] + previous[2]) / 2.0,
                       (previous[1] + previous[3]) / 2.0)
    previous_size = max(previous[2] - previous[0], previous[3] - previous[1])
    compatible = []
    for candidate in candidates:
        try:
            box = _bbox(candidate)
        except (TypeError, ValueError):
            continue
        center = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
        distance = float(np.linalg.norm(np.asarray(center) - previous_center))
        candidate_size = max(box[2] - box[0], box[3] - box[1])
        if distance > 2.0 * max(previous_size, candidate_size):
            continue
        if frame is None:
            return None
        blue = blue_dominant_torso(frame, box, minimum_blue_fraction)
        if blue:
            same_id = _candidate_value(candidate, "track_id") == previous_track_id
            compatible.append((same_id, distance, candidate))
    if not compatible:
        return None
    return min(compatible, key=lambda item: (not item[0], item[1]))[2]


recover_track = recover_blue_pose


def motion_energy(metrics: Sequence[FrameMetric], smoothing: int = 3) -> list[float | None]:
    """Combine normalized entry and rotation proxies into a smoothed score."""
    speeds = np.asarray([
        metric.brzina_ulaska_norm_s if metric.brzina_ulaska_norm_s is not None else np.nan
        for metric in metrics
    ], dtype=float)
    rotations = np.asarray([
        abs(metric.rotation_2d_dps) if metric.rotation_2d_dps is not None else np.nan
        for metric in metrics
    ], dtype=float)

    def normalize(values: np.ndarray) -> np.ndarray:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return np.full(values.shape, np.nan)
        maximum = float(np.max(finite))
        return values / maximum if maximum > 0.0 else np.zeros(values.shape)

    normalized = np.vstack((normalize(speeds), normalize(rotations)))
    energy = np.full(speeds.shape, np.nan)
    usable = np.isfinite(normalized)
    counts = usable.sum(axis=0)
    sums = np.where(usable, normalized, 0.0).sum(axis=0)
    np.divide(sums, counts, out=energy, where=counts > 0)
    energy[~np.isfinite(energy)] = np.nan
    if smoothing <= 1:
        return [float(value) if np.isfinite(value) else None for value in energy]
    smoothed = energy.copy()
    radius = smoothing // 2
    for index, value in enumerate(energy):
        if not np.isfinite(value):
            continue
        neighbours = energy[max(0, index - radius):index + radius + 1]
        finite = neighbours[np.isfinite(neighbours)]
        smoothed[index] = np.mean(finite) if finite.size else np.nan
    return [float(value) if np.isfinite(value) else None for value in smoothed]


def suggest_event_windows(
    motion_samples: Sequence[float | None],
    fps: float,
    threshold: float,
    *,
    expansion_s: float = DEFAULT_WINDOW_EXPANSION_S,
    merge_gap_s: float = DEFAULT_MERGE_GAP_S,
    injury_cutoff_s: float | None = None,
    timestamps: Sequence[float] | None = None,
) -> list[tuple[float, float]]:
    """Suggest time windows from motion samples on the Sony timeline.

    The default applies the required one-second review padding. Callers that
    need the unexpanded sample interval can pass ``expansion_s=0.0``.
    """
    try:
        fps_value = float(fps)
    except (TypeError, ValueError) as exc:
        raise ValueError("fps must be a finite positive number") from exc
    if not isfinite(fps_value) or fps_value <= 0.0:
        raise ValueError("fps must be a finite positive number")
    expansion = float(expansion_s)
    merge_gap = float(merge_gap_s)
    if expansion < 0.0 or merge_gap < 0.0 or not isfinite(expansion) or not isfinite(merge_gap):
        raise ValueError("window expansion and merge gap must be finite and non-negative")
    if timestamps is not None and len(timestamps) != len(motion_samples):
        raise ValueError("timestamps must have one value per motion sample")
    if timestamps is None:
        times = [index / fps_value for index in range(len(motion_samples))]
    else:
        times = [float(value) for value in timestamps]
        if not all(isfinite(value) for value in times):
            raise ValueError("timestamps must be finite")

    active = []
    for index, value in enumerate(motion_samples):
        try:
            active.append(value is not None and isfinite(float(value))
                          and float(value) >= float(threshold))
        except (TypeError, ValueError):
            active.append(False)
    raw: list[tuple[float, float]] = []
    index = 0
    sample_width = 1.0 / fps_value
    while index < len(active):
        if not active[index]:
            index += 1
            continue
        start = index
        while index + 1 < len(active) and active[index + 1]:
            index += 1
        end = index
        raw.append((times[start] - sample_width,
                    times[end] + sample_width))
        index += 1

    expanded = []
    for start, end in raw:
        start -= expansion
        end += expansion
        if injury_cutoff_s is not None:
            cutoff = float(injury_cutoff_s)
            if not isfinite(cutoff):
                raise ValueError("injury cutoff must be finite")
            end = min(end, cutoff)
        start = max(0.0, start)
        if end > start:
            expanded.append((float(start), float(end)))

    merged: list[list[float]] = []
    for start, end in expanded:
        if merged and start - merged[-1][1] < merge_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


@dataclass(frozen=True)
class EventMetrics:
    sony_start_s: float
    sony_end_s: float
    status: str = "predlog"
    iskljuceno_iz_statistike: bool = False
    brzina_ulaska_norm: float | None = None
    rotacija_trupa_2d_dps: float | None = None
    promena_visine_kukova_norm: float | None = None
    vreme_oporavka_s: float | None = None
    intenzitet_pokreta_0_100: float | None = None

    @classmethod
    def from_windows(
        cls,
        windows: Sequence[tuple[float, float]],
        *,
        injury_cutoff_s: float | None = None,
        injury_window: tuple[float, float] | None = None,
    ) -> list["EventMetrics"]:
        events = [cls(float(start), float(end)) for start, end in windows]
        if injury_window is not None:
            start, end = injury_window
            if injury_cutoff_s is not None:
                start = max(float(start), float(injury_cutoff_s))
            if end > start:
                events.append(cls(float(start), float(end), "povreda", True))
        return events

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


def create_injury_event(
    start_s: float, end_s: float, injury_cutoff_s: float
) -> EventMetrics | None:
    start = max(float(start_s), float(injury_cutoff_s))
    end = float(end_s)
    return EventMetrics(start, end, "povreda", True) if end > start else None


def suggest_event_metrics(
    motion_samples: Sequence[float | None],
    fps: float,
    threshold: float,
    *,
    injury_cutoff_s: float | None = None,
    injury_window: tuple[float, float] | None = None,
    expansion_s: float = DEFAULT_WINDOW_EXPANSION_S,
    merge_gap_s: float = DEFAULT_MERGE_GAP_S,
    timestamps: Sequence[float] | None = None,
) -> list[EventMetrics]:
    """Return normal proposals plus an explicitly excluded injury event."""
    windows = suggest_event_windows(
        motion_samples,
        fps,
        threshold,
        expansion_s=expansion_s,
        merge_gap_s=merge_gap_s,
        injury_cutoff_s=injury_cutoff_s,
        timestamps=timestamps,
    )
    return EventMetrics.from_windows(
        windows,
        injury_cutoff_s=injury_cutoff_s,
        injury_window=injury_window,
    )


__all__ = [
    "EventMetrics",
    "bbox_iou",
    "blue_dominant_torso",
    "create_injury_event",
    "iou",
    "motion_energy",
    "NEDOVOLJNO_VIDLJIVO",
    "recover_blue_pose",
    "recover_track",
    "select_blue_detection",
    "select_blue_track_id",
    "select_seed_track",
    "suggest_event_metrics",
    "suggest_event_windows",
]
