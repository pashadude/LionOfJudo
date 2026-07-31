#!/usr/bin/env python3
"""
Frame-accurate clip extraction with ffmpeg.

Re-encodes (fast preset) instead of stream-copying: the Sony X3000's long-GOP
XAVC keyframes are seconds apart, so -c copy would snap cuts to keyframes and
ruin the tight throw windows.
"""

import subprocess
from math import isfinite
from pathlib import Path


def cut_clip(video: Path, t_start: float, t_end: float, out_path: Path,
             scale_height: int | None = None) -> Path:
    """Cut [t_start, t_end] seconds from video into out_path (H.264 + AAC).

    scale_height: optionally downscale (e.g. 1080) to speed up pose inference
    on 4K sources; aspect ratio preserved.
    """
    if t_end <= t_start:
        raise ValueError(f"bad clip window: {t_start}..{t_end}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-ss", f"{max(0.0, t_start):.3f}",
        "-i", str(video),
        "-t", f"{t_end - t_start:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac",
        "-movflags", "+faststart",
    ]
    if scale_height:
        cmd += ["-vf", f"scale=-2:{scale_height}"]
    cmd.append(str(out_path))

    subprocess.run(cmd, check=True, capture_output=True)
    return out_path


def probe_duration(video: Path) -> float:
    """Video duration in seconds."""
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "default=nw=1:nk=1", str(video)]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return float(out.stdout.strip())


def verify_media_export(
    video: Path,
    expected_duration_s: float | None = None,
    tolerance_s: float = 0.75,
) -> float:
    """Reject empty or unexpectedly short/long ffmpeg output."""
    video = Path(video)
    if not video.is_file() or video.stat().st_size == 0:
        raise ValueError(f"izvoz medija je prazan: {video}")

    duration = probe_duration(video)
    if not isfinite(duration) or duration < 0.0:
        raise ValueError(f"trajanje izvoza nije konacno: {video}")
    if expected_duration_s is not None:
        expected = float(expected_duration_s)
        tolerance = float(tolerance_s)
        if not isfinite(expected) or expected <= 0.0:
            raise ValueError("ocekivano trajanje mora biti pozitivno i konacno")
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("dozvoljeno odstupanje mora biti konacno i nenegativno")
        if abs(duration - expected) > tolerance:
            raise ValueError(
                f"trajanje izvoza odstupa od prozora: {duration:.3f}s prema "
                f"{expected:.3f}s"
            )
    return duration


def probe_fps(video: Path) -> float:
    """Primary video stream frame rate as frames per second."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,r_frame_rate",
        "-of", "default=nw=1:nk=1", str(video),
    ]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    for line in out.stdout.splitlines():
        fps = _parse_rate(line.strip())
        if fps > 0:
            return fps
    raise ValueError(f"could not determine FPS for {video}")


def _parse_rate(rate: str) -> float:
    if not rate or rate == "0/0":
        return 0.0
    if "/" in rate:
        num, den = rate.split("/", 1)
        den_f = float(den)
        return float(num) / den_f if den_f else 0.0
    return float(rate)
