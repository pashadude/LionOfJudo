#!/usr/bin/env python3
"""
Segment training video into throw windows using IMU spikes.

Replaces the pose-based motion filter (analyze_track_for_throws in
phase0_judo_analysis.py) as the segmenter: acceleration spikes on the athlete
are cheaper and far more reliable than pixel-space hip tracking with several
kids in frame.
"""

from dataclasses import dataclass, field

from .imu_ingest import (ClockMap, ImuLog, PowerMetrics, Spike,
                         detect_spikes, measure_throw_power)

PRE_S = 4.0    # window before the impact peak (entry + kuzushi)
POST_S = 3.0   # window after (landing + reset)


@dataclass
class ThrowWindow:
    throw_id: int
    t_start: float                 # master time (s)
    t_end: float                   # master time (s)
    t_peak: float                  # master time of the strongest spike
    metrics: dict[str, PowerMetrics] = field(default_factory=dict)  # per unit


def _master_spikes(log: ImuLog, clock_map: ClockMap,
                   threshold_g: float, min_gap_s: float) -> list[tuple[float, Spike]]:
    """Spikes with times mapped onto the master timeline."""
    return [(float(clock_map.to_master(s.t_s)), s)
            for s in detect_spikes(log, threshold_g, min_gap_s)]


def segment_throws(
    logs: dict[str, ImuLog],           # e.g. {"chest": ..., "hip": ...}
    clock_maps: dict[str, ClockMap],
    exclude_master_times: list[float] | None = None,  # ritual events etc.
    threshold_g: float = 3.0,
    min_gap_s: float = 1.5,
    merge_within_s: float = 1.0,
    exclude_margin_s: float = 5.0,
    pre_s: float = PRE_S,
    post_s: float = POST_S,
) -> list[ThrowWindow]:
    """Merge per-unit spikes into throw events and emit clip windows.

    - Spikes from different units within merge_within_s are one event.
    - Events near exclude_master_times (the sync rituals) are dropped.
    - Overlapping windows are merged (keeps the strongest peak's metrics).
    """
    exclude = exclude_master_times or []

    # Collect (master_time, unit, spike) across units, time-sorted
    events: list[tuple[float, str, Spike]] = []
    for unit, log in logs.items():
        if unit not in clock_maps:
            raise ValueError(f"no clock map for unit '{unit}'")
        for t_m, s in _master_spikes(log, clock_maps[unit], threshold_g, min_gap_s):
            events.append((t_m, unit, s))
    events.sort(key=lambda e: e[0])

    # Group into throw events
    groups: list[list[tuple[float, str, Spike]]] = []
    for ev in events:
        if groups and ev[0] - groups[-1][-1][0] <= merge_within_s:
            groups[-1].append(ev)
        else:
            groups.append([ev])

    windows: list[ThrowWindow] = []
    for group in groups:
        t_peak, peak_unit, peak_spike = max(group, key=lambda e: e[2].peak_g)
        if any(abs(t_peak - tx) <= exclude_margin_s for tx in exclude):
            continue

        metrics: dict[str, PowerMetrics] = {}
        for unit, log in logs.items():
            # measure in the unit's own timeline around the event
            t_imu = (t_peak - clock_maps[unit].b) / clock_maps[unit].a
            metrics[unit] = measure_throw_power(log, t_imu)

        windows.append(ThrowWindow(
            throw_id=0,  # numbered after overlap-merge
            t_start=max(0.0, t_peak - pre_s),
            t_end=t_peak + post_s,
            t_peak=t_peak,
            metrics=metrics,
        ))

    # Merge overlapping windows
    merged: list[ThrowWindow] = []
    for w in windows:
        if merged and w.t_start <= merged[-1].t_end:
            prev = merged[-1]
            keep = w if _peak_g(w) > _peak_g(prev) else prev
            merged[-1] = ThrowWindow(
                throw_id=0,
                t_start=min(prev.t_start, w.t_start),
                t_end=max(prev.t_end, w.t_end),
                t_peak=keep.t_peak,
                metrics=keep.metrics,
            )
        else:
            merged.append(w)

    for i, w in enumerate(merged, 1):
        w.throw_id = i
    return merged


def _peak_g(w: ThrowWindow) -> float:
    return max((m.peak_g for m in w.metrics.values()), default=0.0)
