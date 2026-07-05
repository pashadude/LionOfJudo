#!/usr/bin/env python3
"""
Audio-based synchronization between the two cameras (Sony FDR-X3000 + iPhone).

Master timeline convention: the Sony file defines t=0. The iPhone timeline is
mapped onto it via cross-correlation of the two audio envelopes:
    t_master = t_iphone + offset_s
(offset_s > 0 means the iPhone started recording BEFORE the Sony).
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal

ENVELOPE_SR = 8000  # Hz; plenty for clap onsets, keeps correlation fast
SMOOTH_MS = 10


@dataclass
class SyncResult:
    offset_s: float       # add to iPhone time to get master (Sony) time
    confidence: float     # peak / second-peak ratio; > 3 is trustworthy
    search_window_s: float


def extract_audio_envelope(video_path: Path, sr: int = ENVELOPE_SR) -> np.ndarray:
    """Decode audio to mono PCM via ffmpeg and return an onset envelope.

    The two cameras have very different mics and AGC, so we correlate an
    onset-emphasized envelope (rectified first difference, smoothed) rather
    than raw waveforms.
    """
    cmd = [
        "ffmpeg", "-v", "error",
        "-i", str(video_path),
        "-map", "0:a:0", "-ac", "1", "-ar", str(sr),
        "-f", "s16le", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    audio = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32)
    if audio.size == 0:
        raise ValueError(f"No audio stream decoded from {video_path}")

    onset = np.abs(np.diff(audio, prepend=audio[0]))
    win = max(1, int(sr * SMOOTH_MS / 1000))
    kernel = np.ones(win, dtype=np.float32) / win
    envelope = np.convolve(onset, kernel, mode="same")

    std = envelope.std()
    if std > 0:
        envelope = (envelope - envelope.mean()) / std
    return envelope


def find_audio_offset(
    sony_path: Path,
    iphone_path: Path,
    max_offset_s: float = 120.0,
    sr: int = ENVELOPE_SR,
) -> SyncResult:
    """Cross-correlate the two envelopes and return the iPhone->Sony offset."""
    env_sony = extract_audio_envelope(sony_path, sr)
    env_iphone = extract_audio_envelope(iphone_path, sr)

    corr = signal.correlate(env_sony, env_iphone, mode="full", method="fft")
    lags = signal.correlation_lags(len(env_sony), len(env_iphone), mode="full")

    max_lag = int(max_offset_s * sr)
    in_window = np.abs(lags) <= max_lag
    corr_w = corr[in_window]
    lags_w = lags[in_window]

    best = int(np.argmax(corr_w))
    offset_s = lags_w[best] / sr

    # Confidence: best peak vs best peak at least 1s away from it
    away = np.abs(lags_w - lags_w[best]) > sr
    second = corr_w[away].max() if away.any() else 1e-9
    confidence = float(corr_w[best] / max(second, 1e-9))

    return SyncResult(offset_s=offset_s, confidence=confidence,
                      search_window_s=max_offset_s)


def detect_claps(
    video_path: Path,
    top_n: int = 8,
    min_gap_s: float = 0.25,
    sr: int = ENVELOPE_SR,
) -> list[float]:
    """Return times (s) of the sharpest audio transients (claps / slams).

    Used to align the IMU jump ritual to the master timeline.
    """
    env = extract_audio_envelope(video_path, sr)
    distance = max(1, int(min_gap_s * sr))
    peaks, props = signal.find_peaks(env, distance=distance,
                                     prominence=env.std() * 2)
    if peaks.size == 0:
        return []
    order = np.argsort(props["prominences"])[::-1][:top_n]
    times = sorted(peaks[order] / sr)
    return [float(t) for t in times]


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Find audio offset between two videos")
    p.add_argument("--sony", required=True, type=Path)
    p.add_argument("--iphone", required=True, type=Path)
    p.add_argument("--max-offset", type=float, default=120.0)
    args = p.parse_args()

    res = find_audio_offset(args.sony, args.iphone, args.max_offset)
    print(f"Offset (add to iPhone time to get Sony time): {res.offset_s:+.3f} s")
    print(f"Confidence ratio: {res.confidence:.2f} "
          f"({'OK' if res.confidence > 3 else 'LOW - verify manually!'})")
