#!/usr/bin/env python3
"""Raspberry Pi collector for LionOfJudo IMU logs.

Run this on the Pi after training, once the MikroTik/dojo AP is on and the
wearables have rebooted into WiFi download mode.  The Pi pulls logs from the
ESP32 HTTP endpoints, stores them by session, and writes a small text/JSON
status file that can be shown on a terminal or tiny attached screen.
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.fetch_imu import fetch_units, parse_units, results_json  # noqa: E402


DEFAULT_ROOT = Path("/data/lionofjudo")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                   help="Pi storage root; use /data/lionofjudo on USB flash")
    p.add_argument("--session", default=datetime.now().strftime("%Y-%m-%d"),
                   help="session folder name")
    p.add_argument("--unit", action="append", default=None,
                   help="override unit endpoint, e.g. chest=http://host")
    p.add_argument("--wipe", action="store_true",
                   help="delete files from ESP32 after verified download")
    args = p.parse_args()

    session_dir = args.root / "sessions" / args.session
    imu_dir = session_dir / "imu"
    status_json = session_dir / "collector_status.json"
    status_txt = session_dir / "collector_status.txt"

    results = fetch_units(parse_units(args.unit), imu_dir, wipe=args.wipe)
    status_json.parent.mkdir(parents=True, exist_ok=True)
    status_json.write_text(results_json(results))
    status_txt.write_text(render_status(args.session, imu_dir, results))
    print(status_txt.read_text())


def render_status(session: str, imu_dir: Path, results) -> str:
    total = sum(r.bytes_downloaded for r in results)
    lines = [
        "LionOfJudo IMU Collector",
        f"Session: {session}",
        f"IMU dir: {imu_dir}",
        f"Updated: {datetime.now().isoformat(timespec='seconds')}",
        "",
    ]
    for r in results:
        if not r.reachable:
            lines.append(f"{r.unit}: UNREACHABLE ({r.error})")
            continue
        mb = r.bytes_downloaded / 1024 / 1024
        files = len([f for f in r.files if f.verified])
        battery = "?" if r.battery_v is None else f"{r.battery_v:.2f}V"
        free = "?" if r.free_kb is None else f"{r.free_kb}KB"
        lines.append(
            f"{r.unit}: OK  battery {battery}  free {free}  "
            f"files {files}  downloaded {mb:.2f}MB")
        for f in r.files:
            mark = "OK" if f.verified else "BAD"
            wipe = " wiped" if f.wiped else ""
            lines.append(f"  {mark}: {Path(f.dest).name} "
                         f"{f.size/1024/1024:.2f}MB{wipe}")
    lines += [
        "",
        f"TOTAL: {total/1024/1024:.2f}MB",
        "Next: copy this session folder to Mac and run pipeline.run_session.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
