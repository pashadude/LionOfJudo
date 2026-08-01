"""Transparent, video-only pose measures for the coach-confirmed athlete.

The values in this module are image-space proxies.  They are normalized by
the observed torso length and deliberately do not claim physical force,
power, acceleration, impact severity, or a medical outcome.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import atan2, degrees, isfinite
from typing import Any, Iterable, Sequence

import numpy as np


KPT_CONFIDENCE = 0.3
MAX_INTERPOLATION_GAP = 5
SPEED_CAP_NORM_S = 4.0
ROTATION_CAP_DPS = 540.0
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_ANKLE, RIGHT_ANKLE = 15, 16


def _pair_midpoint(keypoints: np.ndarray, left: int, right: int):
    if (keypoints[left, 2] < KPT_CONFIDENCE
            or keypoints[right, 2] < KPT_CONFIDENCE):
        return None
    return (keypoints[left, :2] + keypoints[right, :2]) / 2.0


def _keypoints_from_frame(frame: Any) -> np.ndarray:
    if isinstance(frame, dict):
        frame = frame["keypoints"]
    elif hasattr(frame, "keypoints"):
        frame = frame.keypoints
    array = np.asarray(frame, dtype=float)
    if array.shape == (17, 2):
        array = np.column_stack((array, np.ones(17, dtype=float)))
    if array.ndim != 2 or array.shape[0] < 17 or array.shape[1] < 3:
        raise ValueError("each pose must contain at least 17 keypoints with confidence")
    if not np.isfinite(array[:17, :3]).all():
        raise ValueError("pose keypoints and confidences must be finite")
    return array[:17, :3]


def _wrap_angle(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def wrap_angle(angle: float) -> float:
    """Wrap a degree angle to the interval [-180, 180)."""
    return float(_wrap_angle(float(angle)))


def json_safe(value: Any) -> Any:
    """Convert scalar containers to strict-JSON values without NaN/Infinity."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class FrameMetric:
    """One Sony-relative frame of descriptive pose measurements."""

    frame_index: int
    timestamp_s: float
    hip_midpoint: tuple[float, float] | None
    shoulder_midpoint: tuple[float, float] | None
    torso_length: float | None
    shoulder_angle_deg: float | None
    stance_width_norm: float | None
    vidljivo: bool
    interpolirano: bool
    brzina_ulaska_norm_s: float | None
    rotation_2d_dps: float | None
    hip_level_norm: float | None
    intensity_0_100: float | None
    acceleration_norm_s2: float | None

    @property
    def shoulder_line_angle_deg(self) -> float | None:
        return self.shoulder_angle_deg

    @property
    def shoulder_angle(self) -> float | None:
        return self.shoulder_angle_deg

    @property
    def visible(self) -> bool:
        return self.vidljivo

    @property
    def interpolated(self) -> bool:
        return self.interpolirano

    @property
    def rotacija_trupa_2d_dps(self) -> float | None:
        return self.rotation_2d_dps

    @property
    def promena_visine_kukova_norm(self) -> float | None:
        return self.hip_level_norm

    @property
    def osnova_stava_norm(self) -> float | None:
        return self.stance_width_norm

    @property
    def torso_length_px(self) -> float | None:
        return self.torso_length

    def to_dict(self) -> dict[str, Any]:
        """Return the rounded canonical per-frame review schema."""
        def rounded(value: float | None, digits: int = 6) -> float | None:
            return None if value is None else round(float(value), digits)

        def rounded_point(
            value: tuple[float, float] | None,
        ) -> list[float] | None:
            return None if value is None else [round(float(item), 6) for item in value]

        return json_safe({
            "frame_index": self.frame_index,
            "timestamp_s": rounded(self.timestamp_s, 3),
            "hip_midpoint": rounded_point(self.hip_midpoint),
            "shoulder_midpoint": rounded_point(self.shoulder_midpoint),
            "torso_length": rounded(self.torso_length),
            "shoulder_angle_deg": rounded(self.shoulder_angle_deg),
            "vidljivo": self.vidljivo,
            "interpolirano": self.interpolirano,
            "brzina_ulaska_norm": rounded(self.brzina_ulaska_norm_s),
            "rotacija_trupa_2d_dps": rounded(self.rotation_2d_dps),
            "promena_visine_kukova_norm": rounded(self.hip_level_norm),
            "sirina_stava_norm": rounded(self.stance_width_norm),
            "intenzitet_pokreta_0_100": rounded(self.intensity_0_100),
            "proxy_ubrzanja_norm_s2": rounded(self.acceleration_norm_s2),
        })


def canonical_intensity_series(
    speeds: Sequence[float | None],
    rotations: Sequence[float | None],
) -> list[float | None]:
    """Return centered three-sample intensity using fixed v1 caps."""
    if len(speeds) != len(rotations):
        raise ValueError("speeds and rotations must have equal length")
    raw_intensity: list[float | None] = []
    for speed, rotation in zip(speeds, rotations):
        if (speed is None or rotation is None
                or not isfinite(float(speed)) or not isfinite(float(rotation))):
            raw_intensity.append(None)
            continue
        speed_score = min(100.0, max(0.0, float(speed) / SPEED_CAP_NORM_S * 100.0))
        rotation_score = min(
            100.0, max(0.0, abs(float(rotation)) / ROTATION_CAP_DPS * 100.0)
        )
        raw_intensity.append(0.5 * speed_score + 0.5 * rotation_score)

    smoothed: list[float | None] = []
    for index, value in enumerate(raw_intensity):
        if value is None:
            smoothed.append(None)
            continue
        neighbours = raw_intensity[max(0, index - 1):index + 2]
        finite = [sample for sample in neighbours if sample is not None]
        smoothed.append(sum(finite) / len(finite))
    return smoothed


def _interpolate(values: list[Any]) -> tuple[list[Any], set[int]]:
    result = list(values)
    interpolated: set[int] = set()
    index = 0
    while index < len(result):
        if result[index] is not None:
            index += 1
            continue
        start = index
        while index < len(result) and result[index] is None:
            index += 1
        end = index
        if (start == 0 or end == len(result)
                or end - start > MAX_INTERPOLATION_GAP
                or result[start - 1] is None or result[end] is None):
            continue
        before = np.asarray(result[start - 1], dtype=float)
        after = np.asarray(result[end], dtype=float)
        for offset, position in enumerate(range(start, end), start=1):
            value = before + (after - before) * offset / (end - start + 1)
            result[position] = value
            interpolated.add(position)
    return result, interpolated


def _interpolate_scalar(values: list[float | None]) -> tuple[list[float | None], set[int]]:
    interpolated, positions = _interpolate(values)
    return [None if value is None else float(np.asarray(value))
            for value in interpolated], positions


def compute_pose_metrics(
    frames: Sequence[Any] | Iterable[Any],
    fps: float,
    timestamps: Sequence[float] | None = None,
) -> list[FrameMetric]:
    """Compute normalized pose measures from tracked 17-point COCO poses.

    Missing shoulder/hip pairs remain unavailable unless they form an interior
    gap of at most five frames, in which case their geometric values are
    linearly interpolated and the resulting frame is marked ``interpolirano``.
    """
    try:
        fps_value = float(fps)
    except (TypeError, ValueError) as exc:
        raise ValueError("fps must be a finite positive number") from exc
    if not isfinite(fps_value) or fps_value <= 0.0:
        raise ValueError("fps must be a finite positive number")

    poses = [_keypoints_from_frame(frame) for frame in frames]
    if timestamps is None:
        times = [index / fps_value for index in range(len(poses))]
    else:
        if len(timestamps) != len(poses):
            raise ValueError("timestamps must have one value per pose")
        times = [float(value) for value in timestamps]
        if not all(isfinite(value) for value in times):
            raise ValueError("timestamps must be finite")

    hips: list[np.ndarray | None] = []
    shoulders: list[np.ndarray | None] = []
    torsos: list[float | None] = []
    angles: list[float | None] = []
    stances: list[float | None] = []
    visibility: list[bool] = []

    for keypoints in poses:
        hip = _pair_midpoint(keypoints, LEFT_HIP, RIGHT_HIP)
        shoulder = _pair_midpoint(keypoints, LEFT_SHOULDER, RIGHT_SHOULDER)
        visible = hip is not None and shoulder is not None
        visibility.append(visible)
        # A partial torso is not a trustworthy normalized reference.  Keep
        # both centers unavailable until a bounded interior gap can be filled.
        hips.append(hip if visible else None)
        shoulders.append(shoulder if visible else None)

        if not visible:
            torsos.append(None)
            angles.append(None)
        else:
            torso = float(np.linalg.norm(shoulder - hip))
            if torso <= 1e-9 or not isfinite(torso):
                torsos.append(None)
                angles.append(None)
                visibility[-1] = False
            else:
                torsos.append(torso)
                shoulder_vector = (
                    keypoints[RIGHT_SHOULDER, :2]
                    - keypoints[LEFT_SHOULDER, :2]
                )
                if np.linalg.norm(shoulder_vector) <= 1e-9:
                    angles.append(None)
                else:
                    angles.append(float(degrees(atan2(
                        shoulder_vector[1], shoulder_vector[0]
                    ))))

        if (keypoints[LEFT_ANKLE, 2] >= KPT_CONFIDENCE
                and keypoints[RIGHT_ANKLE, 2] >= KPT_CONFIDENCE
                and torsos[-1] is not None):
            stance = np.linalg.norm(
                keypoints[LEFT_ANKLE, :2] - keypoints[RIGHT_ANKLE, :2]
            ) / torsos[-1]
            stances.append(float(stance))
        else:
            stances.append(None)

    hips, hip_interpolated = _interpolate(hips)
    shoulders, shoulder_interpolated = _interpolate(shoulders)
    torsos, torso_interpolated = _interpolate_scalar(torsos)
    angles, angle_interpolated = _interpolate_scalar(angles)
    stances, stance_interpolated = _interpolate_scalar(stances)
    interpolated = (hip_interpolated | shoulder_interpolated
                    | torso_interpolated | angle_interpolated
                    | stance_interpolated)

    baseline = next((hip for hip in hips if hip is not None), None)
    metrics: list[FrameMetric] = []
    for index in range(len(poses)):
        hip = hips[index]
        shoulder = shoulders[index]
        torso = torsos[index]
        angle = angles[index]
        speed = None
        rotation = None
        dt = times[index] - times[index - 1] if index > 0 else None
        if (dt is not None and dt > 0.0 and hip is not None
                and hips[index - 1] is not None and torso):
            speed = float(np.linalg.norm(hip - hips[index - 1]) / torso / dt)
        if (dt is not None and dt > 0.0 and angle is not None
                and angles[index - 1] is not None):
            rotation = float(_wrap_angle(angle - angles[index - 1]) / dt)
        hip_level = None
        if hip is not None and baseline is not None and torso:
            hip_level = float((hip[1] - baseline[1]) / torso)

        metrics.append(FrameMetric(
            frame_index=index,
            timestamp_s=float(times[index]),
            hip_midpoint=(float(hip[0]), float(hip[1])) if hip is not None else None,
            shoulder_midpoint=(float(shoulder[0]), float(shoulder[1]))
            if shoulder is not None else None,
            torso_length=torso,
            shoulder_angle_deg=angle,
            stance_width_norm=stances[index],
            vidljivo=visibility[index],
            interpolirano=index in interpolated,
            brzina_ulaska_norm_s=speed,
            rotation_2d_dps=rotation,
            hip_level_norm=hip_level,
            intensity_0_100=None,
            acceleration_norm_s2=None,
        ))

    smoothed = canonical_intensity_series(
        [metric.brzina_ulaska_norm_s for metric in metrics],
        [metric.rotation_2d_dps for metric in metrics],
    )
    accelerations: list[float | None] = []
    previous_speed: tuple[float, float] | None = None
    for metric in metrics:
        speed = metric.brzina_ulaska_norm_s
        if speed is None:
            accelerations.append(None)
            previous_speed = None
            continue
        acceleration = None
        if previous_speed is not None:
            speed_before, time_before = previous_speed
            dt = metric.timestamp_s - time_before
            if dt > 0.0:
                acceleration = abs(speed - speed_before) / dt
        accelerations.append(acceleration)
        previous_speed = (speed, metric.timestamp_s)

    return [
        replace(
            metric,
            intensity_0_100=smoothed[index],
            acceleration_norm_s2=accelerations[index],
        )
        for index, metric in enumerate(metrics)
    ]


__all__ = [
    "FrameMetric",
    "ROTATION_CAP_DPS",
    "SPEED_CAP_NORM_S",
    "canonical_intensity_series",
    "compute_pose_metrics",
    "json_safe",
    "wrap_angle",
]
