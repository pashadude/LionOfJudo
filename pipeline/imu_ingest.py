#!/usr/bin/env python3
"""
Ingest binary IMU logs from the lionimu ESP32 firmware.

Log format (little-endian):
  Header, 32 bytes:
    0-3   magic  b'LJIM'
    4     version (1)
    5     unit_id (0=chest, 1=hip)
    6-7   sample_rate_hz (uint16)
    8-11  accel_scale: LSB per g (float32)      e.g. 2048 for +/-16g
    12-15 gyro_scale:  LSB per deg/s (float32)  e.g. 16.4 for +/-2000dps
    16-31 reserved
  Records, 16 bytes each:
    uint32 millis, int16 ax, ay, az, gx, gy, gz
"""

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAGIC = b"LJIM"
HEADER_SIZE = 32
RECORD_SIZE = 16
UNIT_NAMES = {0: "chest", 1: "hip"}


@dataclass
class ImuLog:
    unit: str
    sample_rate_hz: int
    t_s: np.ndarray        # seconds since unit boot (from millis)
    accel_g: np.ndarray    # (N,3) in g
    gyro_dps: np.ndarray   # (N,3) in deg/s
    total_g: np.ndarray = field(init=False)
    gyro_mag: np.ndarray = field(init=False)

    def __post_init__(self):
        self.total_g = np.linalg.norm(self.accel_g, axis=1)
        self.gyro_mag = np.linalg.norm(self.gyro_dps, axis=1)


@dataclass
class Spike:
    t_s: float
    peak_g: float


@dataclass
class ClockMap:
    """t_master = a * t_imu + b"""
    a: float
    b: float
    residuals_ms: list[float]

    def to_master(self, t_imu):
        return self.a * np.asarray(t_imu) + self.b


@dataclass
class PowerMetrics:
    peak_g: float
    duration_ms: float
    max_rotation_dps: float
    power_index: float


def load_imu_log(path: Path) -> ImuLog:
    raw = Path(path).read_bytes()
    if len(raw) < HEADER_SIZE or raw[:4] != MAGIC:
        raise ValueError(f"{path}: not a LJIM log (bad magic or too short)")

    version, unit_id, rate = struct.unpack_from("<BBH", raw, 4)
    accel_scale, gyro_scale = struct.unpack_from("<ff", raw, 8)
    if version != 1:
        raise ValueError(f"{path}: unsupported LJIM version {version}")
    if accel_scale <= 0 or gyro_scale <= 0:
        raise ValueError(f"{path}: invalid scale factors in header")

    n = (len(raw) - HEADER_SIZE) // RECORD_SIZE
    body = raw[HEADER_SIZE:HEADER_SIZE + n * RECORD_SIZE]
    rec = np.frombuffer(body, dtype=np.dtype([
        ("millis", "<u4"),
        ("a", "<i2", (3,)),
        ("g", "<i2", (3,)),
    ]))

    return ImuLog(
        unit=UNIT_NAMES.get(unit_id, f"unit{unit_id}"),
        sample_rate_hz=rate,
        t_s=rec["millis"].astype(np.float64) / 1000.0,
        accel_g=rec["a"].astype(np.float64) / accel_scale,
        gyro_dps=rec["g"].astype(np.float64) / gyro_scale,
    )


def detect_spikes(log: ImuLog, threshold_g: float = 3.0,
                  min_gap_s: float = 1.5) -> list[Spike]:
    """Threshold crossings on |a|, deduped to per-event peaks.

    Port of detect_throws() from ACCELEROMETER_SYSTEM.md, on real timestamps.
    threshold_g is intentionally a knob: a child's throws may peak at 2-4 g,
    not the adult 4-8 g. Tune in Phase F.
    """
    spikes: list[Spike] = []
    above = log.total_g > threshold_g
    for i in np.flatnonzero(above):
        t, g = float(log.t_s[i]), float(log.total_g[i])
        if spikes and t - spikes[-1].t_s < min_gap_s:
            if g > spikes[-1].peak_g:
                spikes[-1] = Spike(t_s=t, peak_g=g)
        else:
            spikes.append(Spike(t_s=t, peak_g=g))
    return spikes


def detect_sync_ritual(log: ImuLog, search_s: float = 180.0,
                       window_s: float = 10.0, threshold_g: float = 3.0,
                       from_end: bool = False) -> list[float]:
    """Find the 3-jump/slam ritual: exactly 3 sharp spikes within window_s,
    inside the first (or last) search_s of the log. Returns the 3 spike times.
    """
    if from_end:
        t_max = float(log.t_s[-1])
        mask = log.t_s >= t_max - search_s
    else:
        mask = log.t_s <= float(log.t_s[0]) + search_s

    sub = ImuLog(unit=log.unit, sample_rate_hz=log.sample_rate_hz,
                 t_s=log.t_s[mask], accel_g=log.accel_g[mask],
                 gyro_dps=log.gyro_dps[mask])
    if sub.t_s.size == 0:
        return []
    # Ritual jumps are deliberate and spaced ~1s; use a tighter dedupe gap
    spikes = detect_spikes(sub, threshold_g=threshold_g, min_gap_s=0.5)

    # Slide over spike triplets, prefer the tightest cluster within window_s
    best: list[float] | None = None
    for i in range(len(spikes) - 2):
        t0, t2 = spikes[i].t_s, spikes[i + 2].t_s
        if t2 - t0 <= window_s:
            if best is None or (t2 - t0) < (best[2] - best[0]):
                best = [spikes[i].t_s, spikes[i + 1].t_s, spikes[i + 2].t_s]
    return best or []


def align_to_master(ritual_times_imu: list[float],
                    ritual_times_master: list[float]) -> ClockMap:
    """Least-squares fit t_master = a*t_imu + b over matched ritual events.

    With one ritual (3 points) 'a' stays ~1; with start+end rituals (6 points)
    it absorbs the ESP32 crystal drift.
    """
    if len(ritual_times_imu) != len(ritual_times_master):
        raise ValueError("ritual event lists must have equal length")
    if len(ritual_times_imu) < 2:
        raise ValueError("need at least 2 matched events to fit a clock map")

    x = np.asarray(ritual_times_imu)
    y = np.asarray(ritual_times_master)
    a, b = np.polyfit(x, y, 1)
    residuals_ms = list((y - (a * x + b)) * 1000.0)
    return ClockMap(a=float(a), b=float(b), residuals_ms=residuals_ms)


def match_ritual_to_claps(ritual_imu: list[float],
                          clap_candidates: list[float],
                          window_s: float = 10.0) -> list[float] | None:
    """Find 3 clap times whose spacing pattern matches the 3 IMU ritual spikes.

    The audio track has many transients; the ritual's inter-jump gaps are the
    fingerprint. Returns the matched clap times or None.
    """
    if len(ritual_imu) != 3 or len(clap_candidates) < 3:
        return None
    gaps_imu = np.diff(ritual_imu)

    best, best_err = None, 0.35  # max total gap mismatch (s)
    claps = sorted(clap_candidates)
    for i in range(len(claps) - 2):
        for j in range(i + 1, len(claps) - 1):
            for k in range(j + 1, len(claps)):
                trio = [claps[i], claps[j], claps[k]]
                if trio[2] - trio[0] > window_s:
                    continue
                err = float(np.abs(np.diff(trio) - gaps_imu).sum())
                if err < best_err:
                    best, best_err = trio, err
    return best


def measure_throw_power(log: ImuLog, t_peak: float,
                        window_s: float = 1.5) -> PowerMetrics:
    """Port of measure_throw_power() from ACCELEROMETER_SYSTEM.md on real time."""
    mask = (log.t_s >= t_peak - window_s) & (log.t_s <= t_peak + window_s)
    if not mask.any():
        return PowerMetrics(0.0, 0.0, 0.0, 0.0)

    tg = log.total_g[mask]
    ts = log.t_s[mask]
    gm = log.gyro_mag[mask]

    i_peak = int(np.argmax(tg))
    peak_g = float(tg[i_peak])
    duration_ms = float((ts[i_peak] - ts[0]) * 1000.0)
    max_rotation = float(gm.max())
    power_index = peak_g * max_rotation / max(duration_ms, 1.0)
    return PowerMetrics(peak_g=peak_g, duration_ms=duration_ms,
                        max_rotation_dps=max_rotation, power_index=power_index)


def write_log(path: Path, unit_id: int, sample_rate_hz: int,
              t_s: np.ndarray, accel_g: np.ndarray, gyro_dps: np.ndarray,
              accel_scale: float = 2048.0, gyro_scale: float = 16.4) -> None:
    """Write a LJIM log. Used by tests and simulators; firmware writes the
    same layout on-device."""
    header = struct.pack("<4sBBHff16x", MAGIC, 1, unit_id, sample_rate_hz,
                         accel_scale, gyro_scale)
    a = np.clip(np.asarray(accel_g) * accel_scale, -32768, 32767).astype("<i2")
    g = np.clip(np.asarray(gyro_dps) * gyro_scale, -32768, 32767).astype("<i2")
    ms = (np.asarray(t_s) * 1000.0).astype("<u4")

    rec = np.empty(len(ms), dtype=np.dtype([
        ("millis", "<u4"), ("a", "<i2", (3,)), ("g", "<i2", (3,)),
    ]))
    rec["millis"], rec["a"], rec["g"] = ms, a, g
    Path(path).write_bytes(header + rec.tobytes())
