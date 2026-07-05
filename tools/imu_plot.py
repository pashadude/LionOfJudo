#!/usr/bin/env python3
"""
Plot an IMU log: total g-force and rotation with detected spikes marked.

The bench-test and threshold-tuning tool (Phases A, D, F):
    python tools/imu_plot.py sessions/2026-07-05/imu/chest_001.bin
    python tools/imu_plot.py chest_001.bin --threshold-g 2.5 --save plot.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.imu_ingest import (detect_spikes, detect_sync_ritual,  # noqa: E402
                                 load_imu_log)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("log", type=Path)
    p.add_argument("--threshold-g", type=float, default=3.0)
    p.add_argument("--save", type=Path, default=None,
                   help="save PNG instead of showing a window")
    args = p.parse_args()

    log = load_imu_log(args.log)
    spikes = detect_spikes(log, threshold_g=args.threshold_g)
    ritual = detect_sync_ritual(log, threshold_g=args.threshold_g)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(14, 7))
    fig.suptitle(f"{args.log.name}  ({log.unit}, {log.sample_rate_hz}Hz, "
                 f"{log.t_s[-1] - log.t_s[0]:.0f}s)")

    ax1.plot(log.t_s, log.total_g, lw=0.5, color="steelblue")
    ax1.axhline(args.threshold_g, color="orange", ls="--",
                label=f"threshold {args.threshold_g}g")
    for s in spikes:
        ax1.plot(s.t_s, s.peak_g, "rv")
        ax1.annotate(f"{s.peak_g:.1f}g", (s.t_s, s.peak_g),
                     textcoords="offset points", xytext=(0, 8), fontsize=8)
    for t in ritual:
        ax1.axvline(t, color="green", alpha=0.4)
    ax1.set_ylabel("total acceleration (g)")
    ax1.legend(loc="upper right")
    if ritual:
        ax1.set_title(f"green lines = sync ritual @ "
                      f"{[round(t, 2) for t in ritual]}", fontsize=9)

    ax2.plot(log.t_s, log.gyro_mag, lw=0.5, color="purple")
    ax2.set_ylabel("rotation (°/s)")
    ax2.set_xlabel("time since unit boot (s)")

    print(f"{len(spikes)} spikes above {args.threshold_g}g:")
    for s in spikes:
        print(f"  {s.t_s:8.2f}s  {s.peak_g:.1f}g")
    if ritual:
        print(f"sync ritual: {[round(t, 2) for t in ritual]}")
    else:
        print("sync ritual: NOT FOUND (need 3 spikes within 10s "
              "in the first 3 minutes)")

    if args.save:
        fig.savefig(args.save, dpi=110, bbox_inches="tight")
        print(f"saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
