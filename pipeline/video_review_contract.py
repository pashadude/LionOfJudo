from dataclasses import dataclass
import math
from typing import Any


def _finite_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a JSON number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    return normalized


def _string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a JSON string")
    return value


def _boolean(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a JSON boolean")
    return value


@dataclass(frozen=True)
class AnchorPair:
    name: str
    sony_s: float
    iphone_s: float
    user_confirmed: bool = False
    triple_tap_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _string(self.name, "name"))
        object.__setattr__(self, "sony_s", _finite_float(self.sony_s, "sony_s"))
        object.__setattr__(self, "iphone_s", _finite_float(self.iphone_s, "iphone_s"))
        object.__setattr__(self, "user_confirmed", _boolean(self.user_confirmed, "user_confirmed"))
        if isinstance(self.triple_tap_count, bool) or not isinstance(self.triple_tap_count, int):
            raise TypeError("triple_tap_count must be a JSON integer")
        if self.triple_tap_count < 0:
            raise ValueError("triple_tap_count must not be negative")

    @property
    def is_confirmed(self) -> bool:
        return self.user_confirmed and self.triple_tap_count == 3

    def to_dict(self) -> dict[str, Any]:
        self.__post_init__()
        return {
            "name": self.name,
            "sony_s": self.sony_s,
            "iphone_s": self.iphone_s,
            "user_confirmed": self.user_confirmed,
            "triple_tap_count": self.triple_tap_count,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AnchorPair":
        return cls(
            name=value["name"],
            sony_s=value["sony_s"],
            iphone_s=value["iphone_s"],
            user_confirmed=value.get("user_confirmed", False),
            triple_tap_count=value.get("triple_tap_count", 0),
        )


@dataclass
class ReviewEvent:
    event_id: str
    sony_start_s: float
    sony_end_s: float
    prijavljen_povredni_dogadjaj: bool = False
    iskljuceno_iz_statistike: bool = False

    def __post_init__(self) -> None:
        self.event_id = _string(self.event_id, "event_id")
        self.sony_start_s = _finite_float(self.sony_start_s, "sony_start_s")
        self.sony_end_s = _finite_float(self.sony_end_s, "sony_end_s")
        self.prijavljen_povredni_dogadjaj = _boolean(
            self.prijavljen_povredni_dogadjaj, "prijavljen_povredni_dogadjaj"
        )
        self.iskljuceno_iz_statistike = _boolean(
            self.iskljuceno_iz_statistike, "iskljuceno_iz_statistike"
        )
        if self.sony_end_s <= self.sony_start_s:
            raise ValueError("event end must be after start")
        if self.prijavljen_povredni_dogadjaj:
            self.iskljuceno_iz_statistike = True

    def to_dict(self) -> dict[str, Any]:
        self.__post_init__()
        return {
            "event_id": self.event_id,
            "sony_start_s": self.sony_start_s,
            "sony_end_s": self.sony_end_s,
            "prijavljen_povredni_dogadjaj": self.prijavljen_povredni_dogadjaj,
            "iskljuceno_iz_statistike": self.iskljuceno_iz_statistike,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ReviewEvent":
        return cls(
            event_id=value["event_id"],
            sony_start_s=value["sony_start_s"],
            sony_end_s=value["sony_end_s"],
            prijavljen_povredni_dogadjaj=value.get("prijavljen_povredni_dogadjaj", False),
            iskljuceno_iz_statistike=value.get("iskljuceno_iz_statistike", False),
        )


@dataclass
class ReviewSession:
    session_id: str
    sony_video: str
    iphone_video: str
    anchors: list[AnchorPair]
    injury_cutoff_s: float
    events: list[ReviewEvent]

    def __post_init__(self) -> None:
        self.session_id = _string(self.session_id, "session_id")
        self.sony_video = _string(self.sony_video, "sony_video")
        self.iphone_video = _string(self.iphone_video, "iphone_video")
        if not isinstance(self.anchors, list) or not all(isinstance(anchor, AnchorPair) for anchor in self.anchors):
            raise TypeError("anchors must be a JSON list of AnchorPair values")
        self.injury_cutoff_s = _finite_float(self.injury_cutoff_s, "injury_cutoff_s")
        if not isinstance(self.events, list) or not all(isinstance(event, ReviewEvent) for event in self.events):
            raise TypeError("events must be a JSON list of ReviewEvent values")

    def normal_events(self) -> list[ReviewEvent]:
        return [event for event in self.events if not event.iskljuceno_iz_statistike]

    def to_dict(self) -> dict[str, Any]:
        self.__post_init__()
        return {
            "session_id": self.session_id,
            "sony_video": self.sony_video,
            "iphone_video": self.iphone_video,
            "anchors": [anchor.to_dict() for anchor in self.anchors],
            "injury_cutoff_s": self.injury_cutoff_s,
            "events": [event.to_dict() for event in self.events],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ReviewSession":
        return cls(
            session_id=value["session_id"],
            sony_video=value["sony_video"],
            iphone_video=value["iphone_video"],
            anchors=[AnchorPair.from_dict(anchor) for anchor in value["anchors"]],
            injury_cutoff_s=value["injury_cutoff_s"],
            events=[ReviewEvent.from_dict(event) for event in value["events"]],
        )


def validate_review_session(
    session: ReviewSession,
    sony_duration_s: float,
    iphone_duration_s: float,
) -> None:
    for anchor in session.anchors:
        if not 0.0 <= anchor.sony_s <= sony_duration_s:
            raise ValueError("anchor is outside Sony source duration")
        if not 0.0 <= anchor.iphone_s <= iphone_duration_s:
            raise ValueError("anchor is outside iPhone source duration")

    if session.anchors and session.injury_cutoff_s < min(anchor.sony_s for anchor in session.anchors):
        raise ValueError("injury cutoff must not precede the first anchor")

    for event in session.events:
        if event.prijavljen_povredni_dogadjaj and not event.iskljuceno_iz_statistike:
            raise ValueError("injury events must be excluded from statistics")
        if not event.iskljuceno_iz_statistike and event.sony_start_s < session.injury_cutoff_s < event.sony_end_s:
            raise ValueError("normal event must not cross the injury cutoff")
