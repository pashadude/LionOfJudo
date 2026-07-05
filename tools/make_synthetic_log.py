#!/usr/bin/env python3
"""Generate a synthetic LJIM log for pipeline testing (no hardware needed).

Simulates: quiet baseline at 1g, a 3-jump ritual near the start, several
throw spikes, and an optional 3-clap ritual near the end.
"""

import argparse
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.imu_ingest import write_log  # noqa: E402


def add_spike(accel: np.ndarray, gyro: np.ndarray, t: np.ndarray,
              t_event: float, peak_g: float, rot_dps: float,
              width_s: float = 0.15) -> None:
    pulse = np.exp(-0.5 * ((t - t_event) / (width_s / 3)) ** 2)
    accel[:, 2] += peak_g * pulse
    gyro[:, 1] += rot_dps * pulse


def build_log(out: Path, unit_id: int, rate: int, duration_s: float,
              ritual_start: list[float], throws: list[tuple[float, float]],
              ritual_end: list[float], seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    n = int(duration_s * rate)
    t = np.arange(n) / rate

    accel = rng.normal(0, 0.05, (n, 3))
    accel[:, 2] += 1.0  # gravity
    gyro = rng.normal(0, 3.0, (n, 3))

    for te in ritual_start:
        add_spike(accel, gyro, t, te, peak_g=4.5, rot_dps=150)
    for te, pg in throws:
        add_spike(accel, gyro, t, te, peak_g=pg, rot_dps=400, width_s=0.3)
    for te in ritual_end:
        add_spike(accel, gyro, t, te, peak_g=4.5, rot_dps=150)

    write_log(out, unit_id=unit_id, sample_rate_hz=rate,
              t_s=t, accel_g=accel, gyro_dps=gyro)
    print(f"Wrote {out} ({n} samples, {duration_s:.0f}s at {rate}Hz)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--unit", type=int, default=0, help="0=chest 1=hip")
    p.add_argument("--rate", type=int, default=200)
    p.add_argument("--duration", type=float, default=300.0)
    args = p.parse_args()

    build_log(
        args.out, args.unit, args.rate, args.duration,
        ritual_start=[10.0, 11.2, 12.1],
        throws=[(60.0, 3.5), (125.0, 5.2), (210.0, 2.8)],
        ritual_end=[290.0, 291.1, 292.3],
    )
