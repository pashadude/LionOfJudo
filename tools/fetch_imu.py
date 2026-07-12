#!/usr/bin/env python3
"""
Download IMU logs from the lionimu units over WiFi.

Units appear as http://lion-chest.local and http://lion-hip.local when they
join the home network (plug into USB power at home; they boot into download
mode automatically).  The same script can run on a Raspberry Pi connected to
the dojo MikroTik AP; it still pulls logs from the ESP32 units, so no firmware
upload/push path is required.

Usage:
    python tools/fetch_imu.py --out sessions/2026-07-05/imu/
    python tools/fetch_imu.py --out sessions/2026-07-05/imu/ --wipe
    python tools/fetch_imu.py --out /data/lionofjudo/imu \
        --unit chest=http://lion-chest.local --unit hip=http://lion-hip.local
"""

import argparse
import json
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_UNITS = {
    "chest": "http://lion-chest.local",
    "hip": "http://lion-hip.local",
}
TIMEOUT = 10


@dataclass
class DownloadedFile:
    name: str
    size: int
    dest: str
    verified: bool
    wiped: bool = False


@dataclass
class UnitFetchResult:
    unit: str
    base_url: str
    reachable: bool
    battery_v: float | None = None
    free_kb: int | None = None
    files: list[DownloadedFile] = field(default_factory=list)
    error: str | None = None

    @property
    def bytes_downloaded(self) -> int:
        return sum(f.size for f in self.files if f.verified)


def get_json(url: str):
    with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
        return json.loads(r.read())


def download(url: str, dest: Path) -> int:
    with urllib.request.urlopen(url, timeout=60) as r:
        data = r.read()
    dest.write_bytes(data)
    return len(data)


def parse_units(values: list[str] | None) -> dict[str, str]:
    if not values:
        return dict(DEFAULT_UNITS)
    units: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise SystemExit(
                f"bad --unit '{item}', expected name=http://host")
        name, base = item.split("=", 1)
        units[name.strip()] = base.rstrip("/")
    return units


def fetch_unit(unit: str, base: str, out_dir: Path,
               wipe: bool = False) -> UnitFetchResult:
    result = UnitFetchResult(unit=unit, base_url=base, reachable=False)
    try:
        status = get_json(f"{base}/status")
        files = get_json(f"{base}/list")
        result.reachable = True
        result.battery_v = _maybe_float(status.get("battery_v"))
        result.free_kb = _maybe_int(status.get("free_kb"))
    except Exception as e:
        result.error = str(e)
        return result

    for f in files:
        name, size = f["name"], int(f["size"])
        dest = out_dir / f"{unit}_{name}"
        got = download(f"{base}/download?f={name}", dest)
        verified = got == size
        item = DownloadedFile(name=name, size=size, dest=str(dest),
                              verified=verified)
        if verified and wipe:
            urllib.request.urlopen(
                urllib.request.Request(f"{base}/delete?f={name}",
                                       method="POST"),
                timeout=TIMEOUT)
            item.wiped = True
        result.files.append(item)
    return result


def fetch_units(units: dict[str, str], out_dir: Path,
                wipe: bool = False) -> list[UnitFetchResult]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return [fetch_unit(unit, base, out_dir, wipe=wipe)
            for unit, base in units.items()]


def print_results(results: list[UnitFetchResult]) -> None:
    total = 0
    for r in results:
        print(f"\n=== {r.unit} ({r.base_url}) ===")
        if not r.reachable:
            print(f"  UNREACHABLE: {r.error} "
                  "(is the unit powered and on WiFi?)")
            continue
        total += r.bytes_downloaded
        print(f"battery: {_fmt(r.battery_v, 'V')}  "
              f"free flash: {_fmt(r.free_kb, 'KB')}")
        if not r.files:
            print("  no files on device")
        for f in r.files:
            print(f"  {f.name} ({f.size/1024:.0f}KB) -> {f.dest}")
            if f.verified:
                print("  ✓ verified" + (" and wiped" if f.wiped else ""))
            else:
                print("  SIZE MISMATCH — keeping on device")
    print(f"\nTOTAL downloaded: {total/1024/1024:.2f} MB")


def results_json(results: list[UnitFetchResult]) -> str:
    return json.dumps([asdict(r) for r in results], indent=2)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--unit", action="append", default=None,
                   help="override unit endpoint, e.g. chest=http://host")
    p.add_argument("--status-json", type=Path, default=None,
                   help="write machine-readable fetch status")
    p.add_argument("--wipe", action="store_true",
                   help="delete files from the unit after verified download")
    args = p.parse_args()

    results = fetch_units(parse_units(args.unit), args.out, wipe=args.wipe)
    print_results(results)
    if args.status_json:
        args.status_json.parent.mkdir(parents=True, exist_ok=True)
        args.status_json.write_text(results_json(results))


def _maybe_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _maybe_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _fmt(value, unit: str) -> str:
    if value is None:
        return f"?{unit}"
    if isinstance(value, float):
        return f"{value:.2f}{unit}"
    return f"{value}{unit}"


if __name__ == "__main__":
    main()
