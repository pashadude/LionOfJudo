#!/usr/bin/env python3
"""
Download IMU logs from the lionimu units over home WiFi.

Units appear as http://lion-chest.local and http://lion-hip.local when they
join the home network (plug into USB power at home; they boot into download
mode automatically).

Usage:
    python tools/fetch_imu.py --out sessions/2026-07-05/imu/
    python tools/fetch_imu.py --out sessions/2026-07-05/imu/ --wipe
"""

import argparse
import json
import urllib.request
from pathlib import Path

UNITS = {
    "chest": "http://lion-chest.local",
    "hip": "http://lion-hip.local",
}
TIMEOUT = 10


def get_json(url: str):
    with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
        return json.loads(r.read())


def download(url: str, dest: Path) -> int:
    with urllib.request.urlopen(url, timeout=60) as r:
        data = r.read()
    dest.write_bytes(data)
    return len(data)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--wipe", action="store_true",
                   help="delete files from the unit after verified download")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for unit, base in UNITS.items():
        print(f"\n=== {unit} ({base}) ===")
        try:
            status = get_json(f"{base}/status")
            print(f"battery: {status.get('battery_v', '?')}V  "
                  f"free flash: {status.get('free_kb', '?')}KB")
            files = get_json(f"{base}/list")
        except Exception as e:
            print(f"  UNREACHABLE: {e} (is the unit powered and on WiFi?)")
            continue

        for f in files:
            name, size = f["name"], f["size"]
            dest = args.out / f"{unit}_{name}"
            print(f"  {name} ({size/1024:.0f}KB) -> {dest}")
            got = download(f"{base}/download?f={name}", dest)
            if got != size:
                print(f"  SIZE MISMATCH ({got} != {size}) — keeping on device")
                continue
            print("  ✓ verified")
            if args.wipe:
                urllib.request.urlopen(
                    urllib.request.Request(f"{base}/delete?f={name}",
                                           method="POST"),
                    timeout=TIMEOUT)
                print("  ✓ wiped from device")


if __name__ == "__main__":
    main()
