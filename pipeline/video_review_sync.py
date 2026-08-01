"""Event-level source windows for synchronized review media."""

from __future__ import annotations

import math
from typing import Any, Mapping


def iphone_sync_offset(event: Mapping[str, Any]) -> float:
    value = event.get("iphone_sync_offset_s", 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("iphone_sync_offset_s must be a JSON number")
    offset = float(value)
    if not math.isfinite(offset):
        raise ValueError("iphone_sync_offset_s must be finite")
    return offset


def iphone_media_bounds(event: Mapping[str, Any]) -> tuple[float, float]:
    offset = iphone_sync_offset(event)
    return (
        float(event["iphone_start_s"]) + offset,
        float(event["iphone_end_s"]) + offset,
    )


__all__ = ["iphone_media_bounds", "iphone_sync_offset"]
