from dataclasses import dataclass
import math
from typing import Any, Mapping


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
        if not event.iskljuceno_iz_statistike and event.sony_end_s > session.injury_cutoff_s:
            raise ValueError("normal event must not end after the injury cutoff")


def validate_review_payload(payload: Mapping[str, Any]) -> None:
    """Validate the final persisted review payload across derived collections."""
    if not isinstance(payload, Mapping):
        raise TypeError("review payload must be a JSON object")
    raw_anchors = payload.get("anchors")
    raw_events = payload.get("events")
    if not isinstance(raw_anchors, list) or len(raw_anchors) != 2:
        raise ValueError("review payload requires exactly two anchors")
    if not isinstance(raw_events, list):
        raise ValueError("review payload events must be a JSON list")
    anchors = [AnchorPair.from_dict(anchor) for anchor in raw_anchors]
    if not all(anchor.is_confirmed for anchor in anchors):
        raise ValueError("review payload anchors must be confirmed triple taps")
    ordered_anchors = sorted(anchors, key=lambda anchor: anchor.iphone_s)
    iphone_delta = ordered_anchors[1].iphone_s - ordered_anchors[0].iphone_s
    sony_delta = ordered_anchors[1].sony_s - ordered_anchors[0].sony_s
    if iphone_delta <= 0.0 or sony_delta <= 0.0:
        raise ValueError("review payload anchors must advance in both videos")
    expected_slope = sony_delta / iphone_delta
    expected_intercept = ordered_anchors[0].sony_s - expected_slope * ordered_anchors[0].iphone_s

    raw_map = payload.get("time_map")
    if not isinstance(raw_map, Mapping):
        raise ValueError("review payload requires a time_map")
    slope = _finite_float(raw_map.get("slope"), "time_map.slope")
    intercept = _finite_float(raw_map.get("intercept"), "time_map.intercept")
    if slope <= 0.0:
        raise ValueError("time_map slope must be positive")
    if not math.isclose(slope, expected_slope, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("time_map slope does not match confirmed anchors")
    if not math.isclose(intercept, expected_intercept, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("time_map intercept does not match confirmed anchors")

    sony_duration = _finite_float(payload.get("sony_duration_s"), "sony_duration_s")
    iphone_duration = _finite_float(payload.get("iphone_duration_s"), "iphone_duration_s")
    if sony_duration <= 0.0 or iphone_duration <= 0.0:
        raise ValueError("source durations must be positive")
    events = [ReviewEvent.from_dict(event) for event in raw_events]
    session = ReviewSession(
        session_id=payload.get("session_id", "session"),
        sony_video=payload.get("sony_video", ""),
        iphone_video=payload.get("iphone_video", ""),
        anchors=anchors,
        injury_cutoff_s=payload.get("injury_cutoff_s"),
        events=events,
    )
    validate_review_session(session, sony_duration, iphone_duration)
    if session.injury_cutoff_s > sony_duration:
        raise ValueError("injury cutoff must be within the Sony source")

    event_ids = [event.event_id for event in events]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("event IDs must be unique")
    first_anchor = min(anchor.sony_s for anchor in anchors)
    normal_events = sorted(session.normal_events(), key=lambda event: event.sony_start_s)
    for event in events:
        if event.sony_start_s < 0.0 or event.sony_end_s > sony_duration:
            raise ValueError("event must be within the Sony source")
        if not event.iskljuceno_iz_statistike and event.sony_start_s < first_anchor:
            raise ValueError("normal event must not start before the first anchor")
    for previous, current in zip(normal_events, normal_events[1:]):
        if current.sony_start_s < previous.sony_end_s:
            raise ValueError("normal events must not overlap")

    metric_keys = (
        "brzina_ulaska_norm",
        "rotacija_trupa_2d_dps",
        "promena_visine_kukova_norm",
        "sirina_stava_norm",
        "vreme_oporavka_s",
        "intenzitet_pokreta_0_100",
    )
    for raw_event in raw_events:
        if not isinstance(raw_event, Mapping):
            raise TypeError("events must contain JSON objects")
        for key in metric_keys:
            value = raw_event.get(key)
            if value is not None:
                _finite_float(value, f"event.{key}")
        intensity = raw_event.get("intenzitet_pokreta_0_100")
        if intensity is not None and not 0.0 <= float(intensity) <= 100.0:
            raise ValueError("event intensity must be within 0..100")
        iphone_start = raw_event.get("iphone_start_s")
        iphone_end = raw_event.get("iphone_end_s")
        if (iphone_start is None) != (iphone_end is None):
            raise ValueError("event iPhone bounds must be present together")
        if iphone_start is not None:
            iphone_start_value = _finite_float(iphone_start, "iphone_start_s")
            iphone_end_value = _finite_float(iphone_end, "iphone_end_s")
            if not 0.0 <= iphone_start_value < iphone_end_value <= iphone_duration:
                raise ValueError("event must be within the iPhone source")
            expected_start = (float(raw_event["sony_start_s"]) - intercept) / slope
            expected_end = (float(raw_event["sony_end_s"]) - intercept) / slope
            if not math.isclose(iphone_start_value, expected_start, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError("event iPhone start does not match the inverse time map")
            if not math.isclose(iphone_end_value, expected_end, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError("event iPhone end does not match the inverse time map")

    frame_metrics = payload.get("frame_metrics", [])
    if not isinstance(frame_metrics, list):
        raise ValueError("frame_metrics must be a JSON list")
    previous_timestamp = -math.inf
    for frame in frame_metrics:
        if not isinstance(frame, Mapping):
            raise TypeError("frame_metrics must contain JSON objects")
        timestamp = _finite_float(frame.get("timestamp_s"), "frame.timestamp_s")
        if timestamp < previous_timestamp:
            raise ValueError("frame metric timestamps must be ordered")
        previous_timestamp = timestamp
        for key in metric_keys:
            value = frame.get(key)
            if value is not None:
                _finite_float(value, f"frame.{key}")
        intensity = frame.get("intenzitet_pokreta_0_100")
        if intensity is not None and not 0.0 <= float(intensity) <= 100.0:
            raise ValueError("frame intensity must be within 0..100")

    event_metrics = payload.get("event_metrics")
    if event_metrics is not None:
        if not isinstance(event_metrics, list):
            raise ValueError("event_metrics must be a JSON list")
        metric_ids = [
            event.get("event_id") for event in event_metrics
            if isinstance(event, Mapping)
        ]
        if len(metric_ids) != len(event_metrics) or metric_ids != event_ids:
            raise ValueError("event_metrics must match active events in order")
        for event, metrics in zip(raw_events, event_metrics):
            for key in ("sony_start_s", "sony_end_s", *metric_keys):
                if event.get(key) != metrics.get(key):
                    raise ValueError("event_metrics must match active event bounds and metrics")

    orphaned = payload.get("orphaned_annotations", [])
    if not isinstance(orphaned, list) or not all(isinstance(item, Mapping) for item in orphaned):
        raise ValueError("orphaned_annotations must be a JSON list of objects")
