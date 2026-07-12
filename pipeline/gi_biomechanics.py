#!/usr/bin/env python3
"""Frame-aligned gi biomechanics from chest/hip IMU logs.

The IMU logs are sampled in each ESP32 unit's local clock.  After
``run_session`` fits ``t_master = a * t_imu + b`` for each unit, this module
resamples chest/hip acceleration and gyro signals onto the Sony master video
timeline.  The resulting CSV is the sensor-derived "gi biomechanics" layer:
one row per video frame, with video used only as the clock/reference surface.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from .imu_ingest import ClockMap, ImuLog

AXES = ("x", "y", "z")


def frame_master_times(t_start: float, t_end: float, fps: float) -> np.ndarray:
    """Return Sony-master timestamps for every frame intersecting a window."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    if t_end <= t_start:
        raise ValueError(f"bad time window: {t_start}..{t_end}")
    n = max(1, int(math.floor((t_end - t_start) * fps)) + 1)
    return t_start + np.arange(n, dtype=np.float64) / fps


def build_frame_rows(
    logs: dict[str, ImuLog],
    clock_maps: dict[str, ClockMap],
    t_start: float,
    t_end: float,
    fps: float,
) -> list[dict[str, float | int | str]]:
    """Build one gi-biomechanics row per Sony video frame.

    Columns are intentionally flat CSV-friendly names.  Missing units or
    out-of-range samples are emitted as blank cells by ``write_csv``.
    """
    times = frame_master_times(t_start, t_end, fps)
    sampled = {
        unit: _sample_unit(log, clock_maps[unit], times)
        for unit, log in logs.items()
        if unit in clock_maps
    }

    rows: list[dict[str, float | int | str]] = []
    for i, t in enumerate(times):
        row: dict[str, float | int | str] = {
            "clip_frame": i,
            "sony_frame": int(round(t * fps)),
            "t_master_s": float(t),
            "t_clip_s": float(t - t_start),
        }
        for unit, cols in sampled.items():
            for name, values in cols.items():
                row[f"{unit}_{name}"] = float(values[i])

        _add_combined_features(row)
        rows.append(row)
    return rows


def summarize_rows(rows: list[dict[str, float | int | str]]) -> dict[str, float]:
    """Summarize a frame-aligned throw window into compact sensor metrics."""
    if not rows:
        return {}

    t = _array(rows, "t_master_s")
    combined_g = _array(rows, "combined_total_g")
    hip_rot = _array(rows, "hip_gyro_mag_dps")
    chest_rot = _array(rows, "chest_gyro_mag_dps")
    coupling = _array(rows, "hip_chest_rotation_coupling")
    hip_g = _array(rows, "hip_total_g")
    chest_g = _array(rows, "chest_total_g")

    summary: dict[str, float] = {}
    summary["combined_peak_g"] = _nanmax(combined_g)
    summary["combined_peak_t_s"] = _time_at_max(t, combined_g)
    summary["hip_peak_rotation_dps"] = _nanmax(hip_rot)
    summary["chest_peak_rotation_dps"] = _nanmax(chest_rot)
    summary["mean_rotation_coupling"] = _nanmean(coupling)

    # Integrals are deliberately simple and interpretable:
    # - impact impulse: area above gravity baseline
    # - rotational impulse: accumulated hip angular speed
    summary["impact_impulse_g_s"] = _trapz_positive(t, combined_g - 1.0)
    summary["hip_rotational_impulse_deg"] = _trapz_positive(t, hip_rot)
    summary["chest_rotational_impulse_deg"] = _trapz_positive(t, chest_rot)

    # Positive means hip peak came before chest peak.
    summary["hip_rotation_lead_ms"] = (
        _time_at_max(t, chest_rot) - _time_at_max(t, hip_rot)
    ) * 1000.0
    summary["hip_impact_lead_ms"] = (
        _time_at_max(t, chest_g) - _time_at_max(t, hip_g)
    ) * 1000.0

    # A scalar for quick ranking, not a final coaching truth.
    max_rot = max(summary["hip_peak_rotation_dps"],
                  summary["chest_peak_rotation_dps"])
    summary["gi_load_index"] = summary["impact_impulse_g_s"] * max_rot
    summary["gi_power_index"] = summary["combined_peak_g"] * max_rot
    return {k: _clean_float(v) for k, v in summary.items()}


def write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    """Write frame rows with stable columns and blank cells for NaN."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames(rows)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                k: _format_cell(row.get(k, ""))
                for k in fieldnames
            })


def _sample_unit(log: ImuLog, clock_map: ClockMap,
                 target_master_t: np.ndarray) -> dict[str, np.ndarray]:
    unit_master_t = np.asarray(clock_map.to_master(log.t_s), dtype=np.float64)
    order = np.argsort(unit_master_t)
    unit_master_t = unit_master_t[order]
    accel = log.accel_g[order]
    gyro = log.gyro_dps[order]

    accel_i = _interp_matrix(unit_master_t, accel, target_master_t)
    gyro_i = _interp_matrix(unit_master_t, gyro, target_master_t)
    out: dict[str, np.ndarray] = {}
    for idx, axis in enumerate(AXES):
        out[f"a{axis}_g"] = accel_i[:, idx]
        out[f"g{axis}_dps"] = gyro_i[:, idx]
    out["total_g"] = np.linalg.norm(accel_i, axis=1)
    out["gyro_mag_dps"] = np.linalg.norm(gyro_i, axis=1)
    return out


def _interp_matrix(src_t: np.ndarray, values: np.ndarray,
                   dst_t: np.ndarray) -> np.ndarray:
    out = np.full((len(dst_t), values.shape[1]), np.nan, dtype=np.float64)
    if src_t.size == 0:
        return out
    valid = (dst_t >= src_t[0]) & (dst_t <= src_t[-1])
    if not valid.any():
        return out
    for i in range(values.shape[1]):
        out[valid, i] = np.interp(dst_t[valid], src_t, values[:, i])
    return out


def _add_combined_features(row: dict[str, float | int | str]) -> None:
    chest_g = _get(row, "chest_total_g")
    hip_g = _get(row, "hip_total_g")
    row["combined_total_g"] = _nanmax(np.asarray([chest_g, hip_g]))

    chest_gyro = np.asarray([_get(row, f"chest_g{a}_dps") for a in AXES])
    hip_gyro = np.asarray([_get(row, f"hip_g{a}_dps") for a in AXES])
    chest_mag = _get(row, "chest_gyro_mag_dps")
    hip_mag = _get(row, "hip_gyro_mag_dps")

    if np.isfinite(chest_mag) and np.isfinite(hip_mag) \
            and chest_mag > 1e-6 and hip_mag > 1e-6:
        row["hip_chest_rotation_coupling"] = float(
            np.dot(hip_gyro, chest_gyro) / (hip_mag * chest_mag)
        )
        row["hip_chest_gyro_delta_dps"] = float(hip_mag - chest_mag)
    else:
        row["hip_chest_rotation_coupling"] = np.nan
        row["hip_chest_gyro_delta_dps"] = np.nan

    if np.isfinite(chest_g) and np.isfinite(hip_g):
        row["hip_chest_total_g_delta"] = float(hip_g - chest_g)
    else:
        row["hip_chest_total_g_delta"] = np.nan

    max_rot = _nanmax(np.asarray([hip_mag, chest_mag]))
    combined_g = _get(row, "combined_total_g")
    row["instant_gi_power"] = combined_g * max_rot \
        if np.isfinite(combined_g) and np.isfinite(max_rot) else np.nan


def _fieldnames(rows: list[dict[str, float | int | str]]) -> list[str]:
    preferred = [
        "clip_frame", "sony_frame", "t_master_s", "t_clip_s",
        "chest_ax_g", "chest_ay_g", "chest_az_g", "chest_total_g",
        "chest_gx_dps", "chest_gy_dps", "chest_gz_dps",
        "chest_gyro_mag_dps",
        "hip_ax_g", "hip_ay_g", "hip_az_g", "hip_total_g",
        "hip_gx_dps", "hip_gy_dps", "hip_gz_dps", "hip_gyro_mag_dps",
        "combined_total_g", "hip_chest_total_g_delta",
        "hip_chest_gyro_delta_dps", "hip_chest_rotation_coupling",
        "instant_gi_power",
    ]
    present = {k for row in rows for k in row.keys()}
    return [k for k in preferred if k in present] + sorted(present - set(preferred))


def _array(rows: list[dict[str, float | int | str]], key: str) -> np.ndarray:
    return np.asarray([_get(row, key) for row in rows], dtype=np.float64)


def _get(row: dict[str, float | int | str], key: str) -> float:
    try:
        return float(row.get(key, np.nan))
    except (TypeError, ValueError):
        return np.nan


def _nanmax(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.max()) if finite.size else np.nan


def _nanmean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else np.nan


def _time_at_max(t: np.ndarray, values: np.ndarray) -> float:
    valid = np.isfinite(t) & np.isfinite(values)
    if not valid.any():
        return np.nan
    tv = t[valid]
    vv = values[valid]
    return float(tv[int(np.argmax(vv))])


def _trapz_positive(t: np.ndarray, values: np.ndarray) -> float:
    valid = np.isfinite(t) & np.isfinite(values)
    if valid.sum() < 2:
        return 0.0
    y = np.maximum(values[valid], 0.0)
    return float(np.trapezoid(y, t[valid]))


def _clean_float(value: float) -> float:
    if not np.isfinite(value):
        return float("nan")
    return float(value)


def _format_cell(value):
    if isinstance(value, float):
        if not np.isfinite(value):
            return ""
        return f"{value:.6f}"
    return value
