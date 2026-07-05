#!/usr/bin/env python3
"""
Frame-accurate clip extraction with ffmpeg.

Re-encodes (fast preset) instead of stream-copying: the Sony X3000's long-GOP
XAVC keyframes are seconds apart, so -c copy would snap cuts to keyframes and
ruin the tight throw windows.
"""

import subprocess
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
