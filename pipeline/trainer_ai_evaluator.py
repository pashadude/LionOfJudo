"""Deterministic video-pose evaluator for trainer comparison."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from pipeline.video_pose_metrics import (
    ROTATION_CAP_DPS,
    SPEED_CAP_NORM_S,
    canonical_intensity_series,
    json_safe,
)


POSE_METRICS_ID = "video-pose-metrics-v1"
EVALUATOR_ID = "deterministicki-v1"
ACCELERATION_CAP_NORM_S2 = 24.0
IMPULSE_CAP_NORM = 12.0
WEIGHTS = {
    "speed": 0.20,
    "rotation": 0.25,
    "acceleration": 0.20,
    "impulse": 0.15,
    "intensity": 0.20,
}


def _value(sample: Any, *keys: str) -> Any:
    if isinstance(sample, Mapping):
        for key in keys:
            if key in sample:
                return sample[key]
        return None
    for key in keys:
        if hasattr(sample, key):
            return getattr(sample, key)
    return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _point(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    x = _number(value[0])
    y = _number(value[1])
    return None if x is None or y is None else (x, y)


def _round_number(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(float(value), digits)


def _bounds(event: Mapping[str, Any]) -> tuple[float, float]:
    start = _number(event.get("sony_start_s"))
    end = _number(event.get("sony_end_s"))
    if start is None or end is None or end <= start:
        raise ValueError("event must have finite increasing Sony bounds")
    return start, end


def _event_samples(
    event: Mapping[str, Any], frame_metrics: Sequence[Any]
) -> list[tuple[int, Any]]:
    start, end = _bounds(event)
    selected = []
    for position, sample in enumerate(frame_metrics):
        timestamp = _number(_value(sample, "timestamp_s"))
        if timestamp is not None and start <= timestamp <= end:
            selected.append((timestamp, position, sample))
    selected.sort(key=lambda item: (item[0], item[1]))
    return [(item[1], item[2]) for item in selected]


def _valid_sample(sample: Any) -> bool:
    return (
        _value(sample, "vidljivo", "visible") is True
        and _value(sample, "interpolirano", "interpolated") is not True
        and _point(_value(sample, "hip_midpoint")) is not None
        and _point(_value(sample, "shoulder_midpoint")) is not None
    )


def _longest_invalid_gap(
    valid_times: Sequence[float], start: float, end: float, fps: float
) -> float:
    duration = end - start
    if not valid_times:
        return duration
    times = sorted(valid_times)
    nominal_dt = 1.0 / fps
    gaps = [max(0.0, times[0] - start), max(0.0, end - times[-1])]
    gaps.extend(
        max(0.0, current - previous - nominal_dt)
        for previous, current in zip(times, times[1:])
    )
    return min(duration, max(gaps, default=0.0))


def _quality(
    event: Mapping[str, Any], samples: Sequence[Any], effective_analysis_fps: float
) -> dict[str, Any]:
    start, end = _bounds(event)
    duration = end - start
    expected = max(1, math.floor(duration * effective_analysis_fps + 1e-9) + 1)
    valid_times = [
        timestamp
        for sample in samples
        if _valid_sample(sample)
        and (timestamp := _number(_value(sample, "timestamp_s"))) is not None
    ]
    valid_count = len(valid_times)
    coverage = min(1.0, valid_count / expected)
    longest_gap = _longest_invalid_gap(
        valid_times, start, end, effective_analysis_fps
    )
    continuity = 1.0 - min(1.0, max(0.0, longest_gap / duration))
    confidence = 0.75 * coverage + 0.25 * continuity
    return {
        "ocekivani_uzorci": expected,
        "validni_uzorci": valid_count,
        "coverage": round(coverage, 6),
        "continuity": round(continuity, 6),
        "najduza_praznina_s": round(longest_gap, 6),
        "pouzdanost_0_1": round(confidence, 6),
        "_available": (
            valid_count >= 12
            and coverage >= 0.70
            and longest_gap <= 0.50 + 1e-9
            and confidence >= 0.70
        ),
        "_low": valid_count >= 6 and coverage >= 0.35,
    }


def _metric_points(
    samples: Sequence[Any], *keys: str, absolute: bool = False
) -> list[tuple[float, float]]:
    points = []
    for sample in samples:
        timestamp = _number(_value(sample, "timestamp_s"))
        value = _number(_value(sample, *keys))
        if timestamp is None or value is None:
            continue
        points.append((abs(value) if absolute else value, timestamp))
    return points


def _nearest_rank_90(points: Sequence[tuple[float, float]]) -> tuple[float, float] | None:
    if not points:
        return None
    ranked = sorted(points, key=lambda item: (item[0], item[1]))
    value = ranked[math.ceil(0.90 * len(ranked)) - 1][0]
    return value, min(timestamp for candidate, timestamp in ranked if candidate == value)


def _speed_change_points(samples: Sequence[Any]) -> tuple[list[tuple[float, float]], float]:
    changes: list[tuple[float, float]] = []
    impulse = 0.0
    previous: tuple[float, float] | None = None
    for sample in samples:
        timestamp = _number(_value(sample, "timestamp_s"))
        speed = _number(_value(sample, "brzina_ulaska_norm", "brzina_ulaska_norm_s"))
        if timestamp is None or speed is None:
            previous = None
            continue
        has_precomputed = (
            "proxy_ubrzanja_norm_s2" in sample
            if isinstance(sample, Mapping)
            else hasattr(sample, "acceleration_norm_s2")
        )
        precomputed = _number(_value(
            sample, "proxy_ubrzanja_norm_s2", "acceleration_norm_s2"
        ))
        if precomputed is not None:
            changes.append((abs(precomputed), timestamp))
        if previous is not None:
            previous_speed, previous_time = previous
            dt = timestamp - previous_time
            if dt > 0.0:
                delta = abs(speed - previous_speed)
                if not has_precomputed:
                    changes.append((delta / dt, timestamp))
                impulse += delta
        previous = (speed, timestamp)
    return changes, impulse


def _peak_time(points: Sequence[tuple[float, float]]) -> float | None:
    if not points:
        return None
    maximum = max(value for value, _ in points)
    return min(timestamp for value, timestamp in points if value == maximum)


def _clamp_score(value: float | None, cap: float) -> float | None:
    if value is None:
        return None
    return min(100.0, max(0.0, value / cap * 100.0))


def _evidence(
    metric: str,
    point: tuple[float, float] | None,
    unit: str,
) -> dict[str, Any] | None:
    if point is None:
        return None
    value, timestamp = point
    return {
        "metrika": metric,
        "vrednost": round(value, 6),
        "jedinica": unit,
        "sony_s": round(timestamp, 3),
    }


def _score(total: float) -> int:
    if total < 20.0:
        return 1
    if total < 40.0:
        return 2
    if total < 60.0:
        return 3
    if total < 80.0:
        return 4
    return 5


def evaluate_event(
    event: Mapping[str, Any],
    frame_metrics: Sequence[Any],
    *,
    effective_analysis_fps: float,
    analysis_fingerprint: str,
) -> dict[str, Any]:
    """Evaluate one event reproducibly from canonical Sony pose samples."""
    fps = _number(effective_analysis_fps)
    if fps is None or fps <= 0.0:
        raise ValueError("effective_analysis_fps must be finite and positive")
    if (
        not isinstance(analysis_fingerprint, str)
        or len(analysis_fingerprint) != 71
        or not analysis_fingerprint.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in analysis_fingerprint[7:])
    ):
        raise ValueError("analysis_fingerprint must be sha256 followed by 64 lowercase hex digits")

    all_samples = list(frame_metrics)
    indexed_samples = _event_samples(event, all_samples)
    samples = [sample for _, sample in indexed_samples]
    quality = _quality(event, samples, fps)
    speed = _nearest_rank_90(_metric_points(
        samples, "brzina_ulaska_norm", "brzina_ulaska_norm_s"
    ))
    rotation_points = _metric_points(
        samples, "rotacija_trupa_2d_dps", "rotation_2d_dps", absolute=True
    )
    rotation = _nearest_rank_90(rotation_points)
    acceleration_points, impulse_value = _speed_change_points(samples)
    acceleration = _nearest_rank_90(acceleration_points)
    canonical_intensity = canonical_intensity_series(
        [
            _number(_value(sample, "brzina_ulaska_norm", "brzina_ulaska_norm_s"))
            for sample in all_samples
        ],
        [
            _number(_value(sample, "rotacija_trupa_2d_dps", "rotation_2d_dps"))
            for sample in all_samples
        ],
    )
    intensity_points = [
        (canonical_intensity[position], timestamp)
        for position, sample in indexed_samples
        if canonical_intensity[position] is not None
        and (timestamp := _number(_value(sample, "timestamp_s"))) is not None
    ]
    intensity_points = [
        (float(value), timestamp)
        for value, timestamp in intensity_points
        if value is not None
    ]
    intensity = _nearest_rank_90(intensity_points)
    signed_rotations = _metric_points(
        samples, "rotacija_trupa_2d_dps", "rotation_2d_dps"
    )

    speed_value = None if speed is None else speed[0]
    rotation_value = None if rotation is None else rotation[0]
    acceleration_value = None if acceleration is None else acceleration[0]
    impulse = impulse_value if acceleration_points else None
    intensity_value = None if intensity is None else intensity[0]
    normalized = {
        "speed_0_100": _clamp_score(speed_value, SPEED_CAP_NORM_S),
        "rotation_0_100": _clamp_score(rotation_value, ROTATION_CAP_DPS),
        "acceleration_0_100": _clamp_score(
            acceleration_value, ACCELERATION_CAP_NORM_S2
        ),
        "impulse_0_100": _clamp_score(impulse, IMPULSE_CAP_NORM),
        "intensity_0_100": _clamp_score(intensity_value, 100.0),
    }
    score_inputs = list(normalized.values())
    missing_metrics = [
        name
        for name, value in (
            ("speed_peak", speed_value),
            ("rotation_peak", rotation_value),
            ("acceleration_peak", acceleration_value),
            ("impulse_proxy", impulse),
            ("intensity_peak", intensity_value),
        )
        if value is None
    ]
    total = None
    if all(value is not None for value in score_inputs):
        total = (
            WEIGHTS["speed"] * normalized["speed_0_100"]
            + WEIGHTS["rotation"] * normalized["rotation_0_100"]
            + WEIGHTS["acceleration"] * normalized["acceleration_0_100"]
            + WEIGHTS["impulse"] * normalized["impulse_0_100"]
            + WEIGHTS["intensity"] * normalized["intensity_0_100"]
        )

    if quality["_available"] and total is not None:
        status = "dostupno"
    elif quality["_low"]:
        status = "niska_pouzdanost"
    else:
        status = "nedovoljno_podataka"
    proposed_score = _score(total) if status == "dostupno" and total is not None else None

    impulse_point = None
    if acceleration_points and impulse is not None:
        impulse_point = (impulse, _peak_time(acceleration_points))
    evidence = [
        _evidence("brzina_ulaska_norm", speed, "duzina_trupa/s"),
        _evidence("ugaona_brzina_trupa_2d", rotation, "step/s"),
        _evidence("proxy_ubrzanja", acceleration, "norm/s2"),
        _evidence("proxy_impulsa", impulse_point, "norm"),
        _evidence("intenzitet_pokreta", intensity, "0-100"),
    ]
    evidence = [item for item in evidence if item is not None]

    if status == "dostupno":
        cited = evidence[1] if len(evidence) > 1 else evidence[0]
        second = evidence[-1]
        reason = (
            f"Na {cited['sony_s']:.3f} s metrika {cited['metrika']} je "
            f"{cited['vrednost']:.3f} {cited['jedinica']}; na "
            f"{second['sony_s']:.3f} s metrika {second['metrika']} je "
            f"{second['vrednost']:.3f} {second['jedinica']}. "
            f"Deterministički zbir v1 je {total:.3f}/100."
        )
    else:
        reason = (
            f"Praćenje ima {quality['validni_uzorci']} validnih uzoraka, "
            f"pokrivenost {quality['coverage']:.3f} i najdužu prazninu "
            f"{quality['najduza_praznina_s']:.3f} s; AI ocena nije dodeljena."
        )
        if missing_metrics:
            reason += " Nedostaju metrike: " + ", ".join(missing_metrics) + "."

    direction_sum = sum(value for value, _ in signed_rotations)
    direction = "desno" if direction_sum > 0.0 else "levo" if direction_sum < 0.0 else "neutralno"
    hip_values = [value for value, _ in _metric_points(
        samples, "promena_visine_kukova_norm", "hip_level_norm"
    )]
    stance_values = [value for value, _ in _metric_points(
        samples, "sirina_stava_norm", "stance_width_norm"
    )]
    imu_confidence = (
        "visoka" if status == "dostupno"
        else "srednja" if status == "niska_pouzdanost"
        else "niska"
    )
    public_quality = {key: value for key, value in quality.items() if not key.startswith("_")}
    result = {
        "analysis_fingerprint": analysis_fingerprint,
        "evaluator_id": EVALUATOR_ID,
        "pose_metrics_id": POSE_METRICS_ID,
        "status": status,
        "predlozena_ocena": proposed_score,
        "pouzdanost_0_1": quality["pouzdanost_0_1"],
        "razlog": reason,
        "dokazi": evidence,
        "nedostaju_metrike": missing_metrics,
        "kvalitet": public_quality,
        "pokazatelji": {
            "speed_peak": _round_number(speed_value),
            "rotation_peak": _round_number(rotation_value),
            "acceleration_peak": _round_number(acceleration_value),
            "impulse_proxy": _round_number(impulse),
            "intensity_peak": _round_number(intensity_value),
            "hip_height_change": _round_number(
                max(hip_values) - min(hip_values) if hip_values else None
            ),
            "stance_width_peak": _round_number(max(stance_values) if stance_values else None),
            "prototipski_zbir_0_100": _round_number(total),
        },
        "normalizacije": {
            key: _round_number(value) for key, value in normalized.items()
        },
        "konstante": {
            "speed_cap": SPEED_CAP_NORM_S,
            "rotation_cap": ROTATION_CAP_DPS,
            "acceleration_cap": ACCELERATION_CAP_NORM_S2,
            "impulse_cap": IMPULSE_CAP_NORM,
            "tezine": dict(WEIGHTS),
            "pragovi_ocene": [20.0, 40.0, 60.0, 80.0],
        },
        "imu_eksperimentalno": {
            "ugaona_brzina_trupa_dps": _round_number(rotation_value),
            "proxy_ubrzanja_0_100": _round_number(normalized["acceleration_0_100"]),
            "proxy_impulsa_0_100": _round_number(normalized["impulse_0_100"]),
            "intenzitet_0_100": _round_number(normalized["intensity_0_100"]),
            "dominantna_rotacija": direction,
            "vrh_sony_s": _round_number(_peak_time(intensity_points), 3),
            "pouzdanost": imu_confidence,
            "izvor": "video_pose_proxy_v1",
            "snaga_3d": None,
            "snaga_3d_status": "Biće kalibrisano u sledećoj verziji.",
        },
    }
    return json_safe(result)


def _source_signature(review: Mapping[str, Any], camera: str) -> dict[str, Any]:
    source = review.get("sources", {}).get(camera, {})
    if not isinstance(source, Mapping):
        raise ValueError(f"{camera} source metadata is missing")
    digest = source.get("sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{camera} source must have a lowercase SHA-256 digest")
    return {
        "sha256": digest,
        "size": source.get("size"),
    }


def compute_analysis_fingerprint(
    review: Mapping[str, Any],
    event: Mapping[str, Any],
    evaluator_id: str = EVALUATOR_ID,
) -> str:
    """Hash every input that defines one event evaluation."""
    if not isinstance(evaluator_id, str) or not evaluator_id:
        raise ValueError("evaluator_id must be non-empty")
    fps = _number(review.get("effective_analysis_fps"))
    if fps is None:
        pose_analysis = review.get("pose_analysis")
        if isinstance(pose_analysis, Mapping):
            fps = _number(pose_analysis.get("effective_analysis_fps"))
    track_id = event.get("selected_track_id", event.get("track_id"))
    if track_id is None:
        track_id = review.get("selected_track_id")
    if track_id is None:
        pose_analysis = review.get("pose_analysis")
        if isinstance(pose_analysis, Mapping):
            track_id = pose_analysis.get("selected_track_id")
    canonical = {
        "sources": {
            "sony": _source_signature(review, "sony"),
            "iphone": _source_signature(review, "iphone"),
        },
        "bounds": {
            "sony_start_s": event.get("sony_start_s"),
            "sony_end_s": event.get("sony_end_s"),
            "iphone_start_s": event.get("iphone_start_s"),
            "iphone_end_s": event.get("iphone_end_s"),
            "iphone_sync_offset_s": event.get("iphone_sync_offset_s", 0.0),
        },
        "selected_track_id": track_id,
        "effective_analysis_fps": fps,
        "pose_metrics_id": POSE_METRICS_ID,
        "evaluator_id": evaluator_id,
    }
    encoded = json.dumps(
        json_safe(canonical),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


__all__ = [
    "EVALUATOR_ID",
    "POSE_METRICS_ID",
    "compute_analysis_fingerprint",
    "evaluate_event",
]
