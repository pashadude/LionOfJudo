from dataclasses import dataclass
from typing import Any, Sequence

from pipeline.video_review_contract import AnchorPair


@dataclass(frozen=True)
class TimeMap:
    slope: float
    intercept: float

    def to_dict(self) -> dict[str, Any]:
        return {"slope": self.slope, "intercept": self.intercept}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TimeMap":
        return cls(slope=value["slope"], intercept=value["intercept"])


def fit_time_map(anchors: Sequence[AnchorPair]) -> TimeMap:
    if len(anchors) != 2:
        raise ValueError("exactly two confirmed anchors are required")
    first, second = sorted(anchors, key=lambda item: item.iphone_s)
    iphone_delta = second.iphone_s - first.iphone_s
    sony_delta = second.sony_s - first.sony_s
    if iphone_delta <= 0 or sony_delta <= 0:
        raise ValueError("anchors must advance in both videos")
    slope = sony_delta / iphone_delta
    return TimeMap(slope=slope, intercept=first.sony_s - slope * first.iphone_s)


def map_iphone_to_sony(iphone_s: float, time_map: TimeMap) -> float:
    return time_map.slope * iphone_s + time_map.intercept
