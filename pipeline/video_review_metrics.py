"""Canonical sampled metric schema and event summaries."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from pipeline.trainer_ai_evaluator import POSE_METRICS_ID
from pipeline.video_event_detection import recovery_to_stable_s
from pipeline.video_pose_metrics import canonical_intensity_series, json_safe


FRAME_SERIES = (
    ("brzina_ulaska_norm", "Brzina ulaska"),
    ("rotacija_trupa_2d_dps", "Rotacija trupa (2D)"),
    ("promena_visine_kukova_norm", "Visina kukova"),
    ("sirina_stava_norm", "Širina stava"),
    ("intenzitet_pokreta_0_100", "Intenzitet pokreta"),
)
RECOVERY_THRESHOLD = 0.20
RECOVERY_CONSECUTIVE_SAMPLES = 3


def canonical_metric_schema(effective_analysis_fps: float | None) -> dict[str, Any]:
    return {
        "version": 1,
        "pose_metrics_id": POSE_METRICS_ID,
        "frame_series": [
            {"key": key, "label": label} for key, label in FRAME_SERIES
        ],
        "effective_analysis_fps": effective_analysis_fps,
        "recovery_to_stable": {
            "source": "uzorkovana energija pokreta",
            "motion_energy_threshold": RECOVERY_THRESHOLD,
            "comparison": "manje ili jednako pragu",
            "consecutive_samples": RECOVERY_CONSECUTIVE_SAMPLES,
            "missing_sample_resets_run": True,
            "duration_endpoint": "treći uzastopni stabilni uzorak",
            "when_not_observable": None,
        },
        "limitation": (
            "Metrike su normalizovani opisi u ravni kamere iz uzorkovane video-analize; "
            "nisu sila, snaga, fizičko ubrzanje, težina udara niti medicinski zaključak."
        ),
    }


def canonicalize_frame_metrics(
    frame_metrics: Sequence[Mapping[str, Any]],
    *,
    trust_precomputed_acceleration: bool = False,
) -> list[dict[str, Any]]:
    """Map legacy FrameMetric keys and align derived 0..100 intensity by timestamp."""
    source = [dict(frame) for frame in frame_metrics]
    speeds = [
        frame.get("brzina_ulaska_norm", frame.get("brzina_ulaska_norm_s"))
        for frame in source
    ]
    rotations = [
        frame.get("rotacija_trupa_2d_dps", frame.get("rotation_2d_dps"))
        for frame in source
    ]
    intensity = canonical_intensity_series(speeds, rotations)
    accelerations: list[float | None] = []
    previous_speed: tuple[float, float] | None = None
    for frame, speed in zip(source, speeds):
        timestamp = frame.get("timestamp_s")
        has_precomputed = (
            trust_precomputed_acceleration
            and "proxy_ubrzanja_norm_s2" in frame
        )
        precomputed = frame.get("proxy_ubrzanja_norm_s2")
        acceleration = (
            float(precomputed)
            if has_precomputed
            and isinstance(precomputed, (int, float)) and not isinstance(precomputed, bool)
            and math.isfinite(float(precomputed))
            else None
        )
        if (
            not has_precomputed
            and isinstance(speed, (int, float)) and not isinstance(speed, bool)
            and isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool)
            and math.isfinite(float(speed)) and math.isfinite(float(timestamp))
            and previous_speed is not None
        ):
            old_speed, old_time = previous_speed
            dt = float(timestamp) - old_time
            if dt > 0.0:
                acceleration = abs(float(speed) - old_speed) / dt
        accelerations.append(acceleration)
        if (
            isinstance(speed, (int, float)) and not isinstance(speed, bool)
            and isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool)
            and math.isfinite(float(speed)) and math.isfinite(float(timestamp))
        ):
            previous_speed = (float(speed), float(timestamp))
        else:
            previous_speed = None
    canonical = []
    legacy_keys = {
        "brzina_ulaska_norm_s",
        "rotation_2d_dps",
        "hip_level_norm",
        "stance_width_norm",
    }
    for index, frame in enumerate(source):
        item = {
            key: json_safe(
                round(float(value), 3 if key == "timestamp_s" else 6)
                if isinstance(value, float) and math.isfinite(value)
                else value
            )
            for key, value in frame.items()
            if key not in legacy_keys
            and key not in {"intenzitet_pokreta_0_100", "proxy_ubrzanja_norm_s2"}
        }
        item["brzina_ulaska_norm"] = json_safe(
            None if speeds[index] is None else round(float(speeds[index]), 6)
        )
        item["rotacija_trupa_2d_dps"] = json_safe(
            None if rotations[index] is None else round(float(rotations[index]), 6)
        )
        hip_level = frame.get("promena_visine_kukova_norm", frame.get("hip_level_norm"))
        stance = frame.get("sirina_stava_norm", frame.get("stance_width_norm"))
        item["promena_visine_kukova_norm"] = json_safe(
            None if hip_level is None else round(float(hip_level), 6)
        )
        item["sirina_stava_norm"] = json_safe(
            None if stance is None else round(float(stance), 6)
        )
        item["intenzitet_pokreta_0_100"] = (
            None if intensity[index] is None else round(float(intensity[index]), 6)
        )
        item["proxy_ubrzanja_norm_s2"] = (
            None if accelerations[index] is None else round(float(accelerations[index]), 6)
        )
        canonical.append(json_safe(item))
    return canonical


def _values(samples: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    values = []
    for sample in samples:
        value = sample.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        numeric = float(value)
        if math.isfinite(numeric):
            values.append(numeric)
    return values


def summarize_event_metrics(
    event: Mapping[str, Any], frame_metrics: Sequence[Mapping[str, Any]]
) -> dict[str, float | None]:
    start = float(event["sony_start_s"])
    end = float(event["sony_end_s"])
    samples = [
        sample
        for sample in frame_metrics
        if isinstance(sample.get("timestamp_s"), (int, float))
        and not isinstance(sample.get("timestamp_s"), bool)
        and start <= float(sample["timestamp_s"]) <= end
    ]
    speeds = _values(samples, "brzina_ulaska_norm")
    rotations = [abs(value) for value in _values(samples, "rotacija_trupa_2d_dps")]
    hip_levels = _values(samples, "promena_visine_kukova_norm")
    stances = _values(samples, "sirina_stava_norm")
    intensities = _values(samples, "intenzitet_pokreta_0_100")
    recovery = recovery_to_stable_s(
        [float(sample["timestamp_s"]) for sample in samples],
        [
            None
            if sample.get("intenzitet_pokreta_0_100") is None
            else float(sample["intenzitet_pokreta_0_100"]) / 100.0
            for sample in samples
        ],
        start,
        end,
        stable_threshold=RECOVERY_THRESHOLD,
        consecutive_samples=RECOVERY_CONSECUTIVE_SAMPLES,
    ) if samples else None
    return {
        "brzina_ulaska_norm": max(speeds) if speeds else None,
        "rotacija_trupa_2d_dps": max(rotations) if rotations else None,
        "promena_visine_kukova_norm": (
            max(hip_levels) - min(hip_levels) if hip_levels else None
        ),
        "sirina_stava_norm": max(stances) if stances else None,
        "vreme_oporavka_s": recovery,
        "intenzitet_pokreta_0_100": max(intensities) if intensities else None,
    }


__all__ = [
    "FRAME_SERIES",
    "RECOVERY_CONSECUTIVE_SAMPLES",
    "RECOVERY_THRESHOLD",
    "canonical_metric_schema",
    "canonicalize_frame_metrics",
    "summarize_event_metrics",
]
