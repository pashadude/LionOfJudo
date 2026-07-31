from dataclasses import dataclass
import math
from typing import Any, Sequence

from pipeline.video_review_contract import AnchorPair


@dataclass(frozen=True)
class TimeMap:
    slope: float
    intercept: float

    def __post_init__(self) -> None:
        if isinstance(self.slope, bool) or not isinstance(self.slope, (int, float)):
            raise TypeError("slope must be a JSON number")
        if isinstance(self.intercept, bool) or not isinstance(self.intercept, (int, float)):
            raise TypeError("intercept must be a JSON number")
        normalized_slope = float(self.slope)
        normalized_intercept = float(self.intercept)
        if not math.isfinite(normalized_slope) or not math.isfinite(normalized_intercept):
            raise ValueError("time-map values must be finite")
        object.__setattr__(self, "slope", normalized_slope)
        object.__setattr__(self, "intercept", normalized_intercept)

    def to_dict(self) -> dict[str, Any]:
        self.__post_init__()
        return {"slope": self.slope, "intercept": self.intercept}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TimeMap":
        return cls(slope=value["slope"], intercept=value["intercept"])


def fit_time_map(anchors: Sequence[AnchorPair]) -> TimeMap:
    if len(anchors) != 2:
        raise ValueError("exactly two confirmed anchors are required")
    if not all(anchor.is_confirmed for anchor in anchors):
        raise ValueError("all anchors must have user-confirmed triple-tap evidence")
    first, second = sorted(anchors, key=lambda item: item.iphone_s)
    iphone_delta = second.iphone_s - first.iphone_s
    sony_delta = second.sony_s - first.sony_s
    if iphone_delta <= 0 or sony_delta <= 0:
        raise ValueError("anchors must advance in both videos")
    slope = sony_delta / iphone_delta
    return TimeMap(slope=slope, intercept=first.sony_s - slope * first.iphone_s)


def map_iphone_to_sony(iphone_s: float, time_map: TimeMap) -> float:
    return time_map.slope * iphone_s + time_map.intercept
