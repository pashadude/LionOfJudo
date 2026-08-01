import json
import unittest

import numpy as np

from pipeline.video_event_detection import (
    EventMetrics,
    bbox_iou,
    recovery_to_stable_s,
    recover_blue_pose,
    select_blue_detection,
    suggest_event_windows,
)


class EventDetectionTests(unittest.TestCase):
    def test_merges_adjacent_motion_samples_into_one_window(self):
        windows = suggest_event_windows(
            [0.0, 0.1, 0.9, 1.1, 0.0],
            fps=10.0,
            threshold=0.5,
            expansion_s=0.0,
        )

        self.assertEqual(windows, [(0.1, 0.4)])

    def test_default_expansion_adds_one_second_on_both_sides(self):
        windows = suggest_event_windows([0.0, 1.0, 0.0], fps=1.0, threshold=0.5)

        self.assertEqual(windows, [(0.0, 3.0)])

    def test_expands_merges_and_clips_windows_at_injury_cutoff(self):
        windows = suggest_event_windows(
            [0.8, 0.0, 0.0, 0.8, 0.8, 0.0, 0.8],
            fps=1.0,
            threshold=0.5,
            expansion_s=1.0,
            merge_gap_s=1.5,
            injury_cutoff_s=4.5,
        )

        self.assertEqual(windows, [(0.0, 4.5)])

    def test_expansion_adds_one_second_on_both_sides(self):
        windows = suggest_event_windows(
            [0.0, 1.0, 0.0], fps=1.0, threshold=0.5, expansion_s=1.0
        )

        self.assertEqual(windows, [(0.0, 3.0)])

    def test_injury_event_is_separate_and_excluded(self):
        events = EventMetrics.from_windows(
            [(0.0, 2.0)], injury_cutoff_s=2.0, injury_window=(2.0, 3.0)
        )

        self.assertEqual(len(events), 2)
        self.assertFalse(events[0].iskljuceno_iz_statistike)
        self.assertTrue(events[1].iskljuceno_iz_statistike)
        self.assertEqual(events[1].status, "povreda")

    def test_rejects_invalid_fps(self):
        with self.assertRaises(ValueError):
            suggest_event_windows([1.0], fps=-1.0, threshold=0.5)

    def test_seed_selects_largest_iou_and_preserves_track_id(self):
        detections = [
            {"bbox": (0, 0, 20, 20), "track_id": 7},
            {"bbox": (8, 8, 28, 28), "track_id": 11},
        ]

        selected = select_blue_detection(detections, (6, 6, 25, 25))

        self.assertEqual(selected["track_id"], 11)
        self.assertGreater(bbox_iou(selected["bbox"], (6, 6, 25, 25)), 0.5)

    def test_recovery_requires_blue_dominant_torso_patch(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[20:80, 20:60] = (255, 0, 0)
        candidates = [
            {"bbox": (20, 20, 60, 80), "track_id": 12},
            {"bbox": (65, 20, 95, 80), "track_id": 13},
        ]

        recovered = recover_blue_pose(
            candidates,
            previous_bbox=(20, 20, 60, 80),
            frame=frame,
            previous_track_id=7,
        )

        self.assertEqual(recovered["track_id"], 12)

    def test_recovery_without_compatible_blue_pose_is_not_visible(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[20:80, 20:60] = (0, 255, 0)

        recovered = recover_blue_pose(
            [{"bbox": (20, 20, 60, 80), "track_id": 12}],
            previous_bbox=(20, 20, 60, 80),
            frame=frame,
            previous_track_id=7,
        )

        self.assertIsNone(recovered)

    def test_recovery_cannot_bypass_hsv_evidence_with_boolean(self):
        recovered = recover_blue_pose(
            [{
                "bbox": (20, 20, 60, 80),
                "track_id": 12,
                "blue_dominant": True,
            }],
            previous_bbox=(20, 20, 60, 80),
            frame=None,
            previous_track_id=7,
        )

        self.assertIsNone(recovered)

    def test_event_metrics_convert_non_finite_numbers_to_json_null(self):
        event = EventMetrics(0.0, float("nan"), intenzitet_pokreta_0_100=float("inf"))

        payload = event.to_dict()

        self.assertIsNone(payload["sony_end_s"])
        self.assertIsNone(payload["intenzitet_pokreta_0_100"])
        json.dumps(payload, allow_nan=False)

    def test_recovery_requires_three_consecutive_samples_at_or_below_threshold(self):
        recovery = recovery_to_stable_s(
            timestamps=[0.0, 1.0, 2.0, 3.0, 4.0],
            motion_samples=[0.1, 1.0, 0.20, 0.15, 0.05],
            event_start_s=0.0,
            event_end_s=4.0,
            stable_threshold=0.20,
            consecutive_samples=3,
        )

        self.assertEqual(recovery, 3.0)

    def test_recovery_is_none_when_missing_or_active_sample_breaks_stability(self):
        recovery = recovery_to_stable_s(
            timestamps=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            motion_samples=[0.1, 1.0, 0.1, None, 0.1, 0.21],
            event_start_s=0.0,
            event_end_s=5.0,
            stable_threshold=0.20,
            consecutive_samples=3,
        )

        self.assertIsNone(recovery)


if __name__ == "__main__":
    unittest.main()
