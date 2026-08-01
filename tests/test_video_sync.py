import json
from decimal import Decimal
from pathlib import Path
import unittest

from pipeline.video_review_contract import (
    AnchorPair,
    ReviewEvent,
    ReviewSession,
    _validate_derived_media_manifest,
    validate_review_payload,
    validate_review_session,
)
from pipeline.video_sync import TimeMap, fit_time_map, map_iphone_to_sony


class VideoSyncTests(unittest.TestCase):
    def test_two_anchors_create_affine_iphone_to_sony_mapping(self):
        anchors = [
            AnchorPair("pocetak", sony_s=10.0, iphone_s=30.0, user_confirmed=True, triple_tap_count=3),
            AnchorPair("kontrola", sony_s=110.4, iphone_s=130.0, user_confirmed=True, triple_tap_count=3),
        ]
        time_map = fit_time_map(anchors)
        self.assertAlmostEqual(map_iphone_to_sony(80.0, time_map), 60.2, places=6)

    def test_rejects_duplicate_or_reversed_anchor_times(self):
        with self.assertRaises(ValueError):
            fit_time_map([
                AnchorPair("pocetak", 10.0, 30.0, user_confirmed=True, triple_tap_count=3),
                AnchorPair("kontrola", 10.0, 130.0, user_confirmed=True, triple_tap_count=3),
            ])

    def test_rejects_unconfirmed_anchor_before_mapping(self):
        with self.assertRaises(ValueError):
            fit_time_map([
                AnchorPair("pocetak", 10.0, 30.0, user_confirmed=True, triple_tap_count=3),
                AnchorPair("kontrola", 110.0, 130.0),
            ])

    def test_injury_event_is_excluded_from_normal_statistics(self):
        session = ReviewSession(
            session_id="demo",
            sony_video="sony.mp4",
            iphone_video="iphone.mov",
            anchors=[
                AnchorPair("pocetak", 10.0, 30.0, user_confirmed=True, triple_tap_count=3),
                AnchorPair("kontrola", 110.0, 130.0, user_confirmed=True, triple_tap_count=3),
            ],
            injury_cutoff_s=126.0,
            events=[ReviewEvent("e-1", 100.0, 105.0), ReviewEvent("e-2", 124.0, 126.0, prijavljen_povredni_dogadjaj=True)],
        )
        self.assertEqual([event.event_id for event in session.normal_events()], ["e-1"])

    def test_contract_round_trips_through_json_scalars_lists_and_dicts(self):
        session = ReviewSession(
            session_id="demo",
            sony_video="sony.mp4",
            iphone_video="iphone.mov",
            anchors=[
                AnchorPair("pocetak", 10.0, 30.0, user_confirmed=True, triple_tap_count=3),
                AnchorPair("kontrola", 110.0, 130.0, user_confirmed=True, triple_tap_count=3),
            ],
            injury_cutoff_s=126.0,
            events=[ReviewEvent("e-1", 100.0, 105.0)],
        )

        encoded = json.dumps(session.to_dict())
        restored = ReviewSession.from_dict(json.loads(encoded))

        self.assertEqual(restored, session)

    def test_serializers_reject_invalid_post_construction_mutation(self):
        event = ReviewEvent("e-1", 100.0, 105.0)
        event.sony_start_s = Decimal("100")
        with self.assertRaises(TypeError):
            event.to_dict()

        session = self._session()
        session.events[0].sony_end_s = float("inf")
        with self.assertRaises(ValueError):
            session.to_dict()

    def test_time_map_round_trips_through_json_scalars_lists_and_dicts(self):
        time_map = TimeMap(slope=1.004, intercept=-20.12)
        self.assertEqual(TimeMap.from_dict(json.loads(json.dumps(time_map.to_dict()))), time_map)

    def test_rejects_non_json_scalar_values_and_non_finite_timestamps(self):
        with self.assertRaises(TypeError):
            AnchorPair(Path("pocetak"), 10.0, 30.0)
        with self.assertRaises(TypeError):
            ReviewEvent("e-1", Decimal("1.0"), 2.0)
        with self.assertRaises(ValueError):
            ReviewEvent("e-1", float("nan"), 2.0)
        with self.assertRaises(ValueError):
            TimeMap(slope=float("inf"), intercept=0.0)

    def test_validation_rejects_anchors_outside_source_durations(self):
        session = self._session(anchors=[
            AnchorPair("pocetak", 10.0, 30.0, user_confirmed=True, triple_tap_count=3),
            AnchorPair("kontrola", 110.0, 130.0, user_confirmed=True, triple_tap_count=3),
        ])
        with self.assertRaises(ValueError):
            validate_review_session(session, sony_duration_s=100.0, iphone_duration_s=200.0)

    def test_validation_rejects_injury_cutoff_before_first_anchor(self):
        session = self._session(injury_cutoff_s=9.0)
        with self.assertRaises(ValueError):
            validate_review_session(session, sony_duration_s=200.0, iphone_duration_s=200.0)

    def test_validation_rejects_normal_event_crossing_cutoff(self):
        session = self._session(events=[ReviewEvent("e-1", 120.0, 130.0)])
        with self.assertRaises(ValueError):
            validate_review_session(session, sony_duration_s=200.0, iphone_duration_s=200.0)

    def test_validation_rejects_normal_event_wholly_after_cutoff(self):
        session = self._session(events=[ReviewEvent("e-1", 130.0, 135.0)])
        with self.assertRaisesRegex(ValueError, "injury cutoff"):
            validate_review_session(session, sony_duration_s=200.0, iphone_duration_s=200.0)

    def test_validation_rejects_injury_event_missing_exclusion_flag(self):
        event = ReviewEvent("e-1", 120.0, 125.0)
        event.prijavljen_povredni_dogadjaj = True
        event.iskljuceno_iz_statistike = False
        session = self._session(events=[event])
        with self.assertRaises(ValueError):
            validate_review_session(session, sony_duration_s=200.0, iphone_duration_s=200.0)

    def test_payload_validation_rejects_out_of_range_intensity_and_stale_event_metrics(self):
        payload = {
            **self._session().to_dict(),
            "sony_duration_s": 200.0,
            "iphone_duration_s": 200.0,
            "time_map": {"slope": 1.0, "intercept": -20.0},
            "frame_metrics": [
                {"timestamp_s": 100.0, "intenzitet_pokreta_0_100": 101.0}
            ],
            "event_metrics": [],
        }

        with self.assertRaisesRegex(ValueError, "0..100"):
            validate_review_payload(payload)

        payload["frame_metrics"][0]["intenzitet_pokreta_0_100"] = 50.0
        with self.assertRaisesRegex(ValueError, "event_metrics"):
            validate_review_payload(payload)

    def test_media_manifest_rejects_traversal_and_unrecognized_media_location(self):
        common = {
            "total_frames": 1,
            "first_pass_candidates": 1,
            "second_pass_candidates": 0,
            "privacy_verified": True,
            "failure_reason": None,
        }
        for relative_path, media_type in (
            ("events/../review.json", "event_clip"),
            ("analysis/fake.mp4", "side_by_side"),
        ):
            with self.subTest(relative_path=relative_path):
                with self.assertRaises(ValueError):
                    _validate_derived_media_manifest(
                        {
                            "derived_media_manifest": [
                                {
                                    **common,
                                    "relative_path": relative_path,
                                    "media_type": media_type,
                                }
                            ]
                        },
                        {"e-1"},
                    )

    def _session(self, **overrides):
        values = {
            "session_id": "demo",
            "sony_video": "sony.mp4",
            "iphone_video": "iphone.mov",
            "anchors": [
                AnchorPair("pocetak", 10.0, 30.0, user_confirmed=True, triple_tap_count=3),
                AnchorPair("kontrola", 110.0, 130.0, user_confirmed=True, triple_tap_count=3),
            ],
            "injury_cutoff_s": 126.0,
            "events": [ReviewEvent("e-1", 100.0, 105.0)],
        }
        values.update(overrides)
        return ReviewSession(**values)


if __name__ == "__main__":
    unittest.main()
