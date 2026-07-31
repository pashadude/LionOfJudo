from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AnchorPair:
    name: str
    sony_s: float
    iphone_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sony_s": self.sony_s,
            "iphone_s": self.iphone_s,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AnchorPair":
        return cls(
            name=value["name"],
            sony_s=value["sony_s"],
            iphone_s=value["iphone_s"],
        )


@dataclass
class ReviewEvent:
    event_id: str
    sony_start_s: float
    sony_end_s: float
    prijavljen_povredni_dogadjaj: bool = False
    iskljuceno_iz_statistike: bool = False

    def __post_init__(self) -> None:
        if self.sony_end_s <= self.sony_start_s:
            raise ValueError("event end must be after start")
        if self.prijavljen_povredni_dogadjaj:
            self.iskljuceno_iz_statistike = True

    def to_dict(self) -> dict[str, Any]:
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

    def normal_events(self) -> list[ReviewEvent]:
        return [event for event in self.events if not event.iskljuceno_iz_statistike]

    def to_dict(self) -> dict[str, Any]:
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
