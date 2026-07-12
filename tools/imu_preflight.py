#!/usr/bin/env python3
"""Check raw LionOfJudo logs before running the video pipeline.

Run after Pi/Mac collection and before ``pipeline.run_session``.  It confirms
the LJIM unit header, timing quality, range saturation, initial stillness, and
the required three-spike start ritual.  It does not modify a log.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.imu_ingest import (detect_sync_ritual, load_imu_log,
                                 log_quality)


def log_paths(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        if item.is_dir():
            paths.extend(sorted(item.glob("*.bin")))
        else:
            paths.append(item)
    return paths


def inspect(path: Path, threshold_g: float) -> dict:
    log = load_imu_log(path)
    quality = log_quality(log)
    ritual = detect_sync_ritual(log, threshold_g=threshold_g)
    expected = None
    stem = path.stem.lower()
    if stem.startswith("chest"):
        expected = "chest"
    elif stem.startswith("hip"):
        expected = "hip"
    return {
        "path": str(path),
        "unit": log.unit,
        "expected_unit": expected,
        "identity_ok": expected is None or expected == log.unit,
        "quality": quality,
        "start_ritual_s": ritual,
        "start_ritual_found": len(ritual) == 3,
    }


def render(result: dict) -> str:
    q = result["quality"]
    ritual = result["start_ritual_s"]
    lines = [
        f"{Path(result['path']).name} ({result['unit']})",
        "  identity: " + ("ok" if result["identity_ok"]
                            else f"MISMATCH (expected {result['expected_unit']})"),
        f"  {q['samples']} samples / {q['duration_s']:.1f}s @ "
        f"{q['sample_rate_hz']}Hz; median dt {q['median_dt_ms']:.2f}ms; "
        f"max dt {q['max_dt_ms']:.2f}ms; late intervals {q['late_intervals']}",
        f"  initial stillness: |a| {q['initial_total_g_mean']:.3f}g +/- "
        f"{q['initial_total_g_std']:.3f}g, gyro median "
        f"{q['initial_gyro_mag_median_dps']:.2f}dps",
        f"  range: accel {'CLIPPED' if q['accelerometer_saturated'] else 'ok'}, "
        f"gyro {'CLIPPED' if q['gyro_saturated'] else 'ok'}",
        "  start ritual: " + (", ".join(f"{t:.2f}s" for t in ritual)
                              if ritual else "NOT FOUND"),
    ]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", nargs="+", type=Path,
                   help="one or more .bin files or a directory containing them")
    p.add_argument("--threshold-g", type=float, default=3.0)
    p.add_argument("--json", type=Path, help="also write machine-readable report")
    args = p.parse_args()

    paths = log_paths(args.paths)
    if not paths:
        raise SystemExit("No .bin files found")
    reports = [inspect(path, args.threshold_g) for path in paths]
    for report in reports:
        print(render(report))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(reports, indent=2, allow_nan=False))

    failures = [r for r in reports if not r["identity_ok"]
                or not r["start_ritual_found"]
                or r["quality"]["accelerometer_saturated"]
                or r["quality"]["gyro_saturated"]]
    if failures:
        raise SystemExit("Preflight needs review: ritual missing or acceleration clipped")


if __name__ == "__main__":
    main()
