import copy
import json
import unittest

from pipeline.trainer_ai_evaluator import (
    compute_analysis_fingerprint,
    evaluate_event,
)


def available_frame_fixture():
    rows = [
        (10.000000000, 0.000000000, 0.0, 4.166666667),
        (10.166666667, 0.333333333, 45.0, 8.333333333),
        (10.333333333, 0.666666667, 90.0, 16.666666667),
        (10.500000000, 1.000000000, 135.0, 25.000000000),
        (10.666666667, 1.333333333, 180.0, 33.333333333),
        (10.833333333, 1.666666667, 225.0, 41.666666667),
        (11.000000000, 2.000000000, 270.0, 50.000000000),
        (11.166666667, 2.333333333, 315.0, 58.333333333),
        (11.333333333, 2.666666667, 360.0, 66.666666667),
        (11.500000000, 3.000000000, 405.0, 75.000000000),
        (11.666666667, 3.333333333, 450.0, 83.333333333),
        (11.833333333, 3.666666667, 495.0, 91.666666667),
        (12.000000000, 4.000000000, 540.0, 95.833333333),
    ]
    return [
        {
            "frame_index": index,
            "timestamp_s": timestamp,
            "hip_midpoint": [100.0 + index, 200.0],
            "shoulder_midpoint": [100.0 + index, 180.0],
            "vidljivo": True,
            "interpolirano": False,
            "brzina_ulaska_norm": speed,
            "rotacija_trupa_2d_dps": rotation,
            "promena_visine_kukova_norm": index / 100.0,
            "sirina_stava_norm": 1.0 + index / 100.0,
            "intenzitet_pokreta_0_100": intensity,
        }
        for index, (timestamp, speed, rotation, intensity) in enumerate(rows)
    ]


def quality_fixture(valid_indices, total=20):
    valid = set(valid_indices)
    rows = []
    for index in range(total):
        is_valid = index in valid
        rows.append({
            "frame_index": index,
            "timestamp_s": index / 6.0,
            "hip_midpoint": [100.0 + index, 200.0] if is_valid else None,
            "shoulder_midpoint": [100.0 + index, 180.0] if is_valid else None,
            "vidljivo": is_valid,
            "interpolirano": False,
            "brzina_ulaska_norm": index / 10.0 if is_valid else None,
            "rotacija_trupa_2d_dps": index * 10.0 if is_valid else None,
            "promena_visine_kukova_norm": index / 100.0 if is_valid else None,
            "sirina_stava_norm": 1.0 if is_valid else None,
            "intenzitet_pokreta_0_100": index * 4.0 if is_valid else None,
        })
    return rows


class TrainerAiEvaluatorTests(unittest.TestCase):
    def test_available_evaluation_is_deterministic_and_cites_sony_times(self):
        event = {"event_id": "e-1", "sony_start_s": 10.0, "sony_end_s": 12.0}
        frames = available_frame_fixture()
        fingerprint = "sha256:" + "0" * 64

        result = evaluate_event(
            event,
            frames,
            effective_analysis_fps=6.0,
            analysis_fingerprint=fingerprint,
        )

        self.assertEqual(result["status"], "dostupno")
        self.assertEqual(result["evaluator_id"], "deterministicki-v1")
        self.assertEqual(result["predlozena_ocena"], 4)
        self.assertEqual(result["pouzdanost_0_1"], 1.0)
        self.assertEqual(result["pokazatelji"]["speed_peak"], 3.666667)
        self.assertEqual(result["pokazatelji"]["rotation_peak"], 495.0)
        self.assertEqual(result["pokazatelji"]["acceleration_peak"], 2.0)
        self.assertEqual(result["pokazatelji"]["impulse_proxy"], 4.0)
        self.assertEqual(result["pokazatelji"]["intensity_peak"], 91.666667)
        self.assertGreaterEqual(len(result["dokazi"]), 2)
        self.assertTrue(
            all(10.0 <= evidence["sony_s"] <= 12.0 for evidence in result["dokazi"])
        )
        self.assertIn("11.833 s", result["razlog"])
        self.assertEqual(
            result,
            evaluate_event(
                event,
                frames,
                effective_analysis_fps=6.0,
                analysis_fingerprint=fingerprint,
            ),
        )
        json.dumps(result, allow_nan=False)

    def test_exactly_twelve_valid_samples_are_available(self):
        result = evaluate_event(
            {"event_id": "e-1", "sony_start_s": 0.0, "sony_end_s": 11 / 6},
            quality_fixture(range(12), total=12),
            effective_analysis_fps=6.0,
            analysis_fingerprint="sha256:" + "1" * 64,
        )

        self.assertEqual(result["status"], "dostupno")
        self.assertIsNotNone(result["predlozena_ocena"])

    def test_eleven_valid_samples_have_low_confidence_and_no_score(self):
        result = evaluate_event(
            {"event_id": "e-1", "sony_start_s": 0.0, "sony_end_s": 11 / 6},
            quality_fixture(range(11), total=12),
            effective_analysis_fps=6.0,
            analysis_fingerprint="sha256:" + "2" * 64,
        )

        self.assertEqual(result["status"], "niska_pouzdanost")
        self.assertIsNone(result["predlozena_ocena"])
        self.assertIsNotNone(result["imu_eksperimentalno"]["intenzitet_0_100"])

    def test_coverage_thresholds_are_inclusive(self):
        available = evaluate_event(
            {"event_id": "e-1", "sony_start_s": 0.0, "sony_end_s": 19 / 6},
            quality_fixture({0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19}),
            effective_analysis_fps=6.0,
            analysis_fingerprint="sha256:" + "3" * 64,
        )
        low = evaluate_event(
            {"event_id": "e-2", "sony_start_s": 0.0, "sony_end_s": 19 / 6},
            quality_fixture({0, 3, 6, 9, 12, 15, 19}),
            effective_analysis_fps=6.0,
            analysis_fingerprint="sha256:" + "4" * 64,
        )

        self.assertEqual(available["kvalitet"]["coverage"], 0.7)
        self.assertEqual(available["status"], "dostupno")
        self.assertEqual(low["kvalitet"]["coverage"], 0.35)
        self.assertEqual(low["status"], "niska_pouzdanost")

    def test_half_second_gap_is_available_but_longer_gap_is_not(self):
        exact = evaluate_event(
            {"event_id": "e-1", "sony_start_s": 0.0, "sony_end_s": 19 / 6},
            quality_fixture(set(range(20)) - {8, 9, 10}),
            effective_analysis_fps=6.0,
            analysis_fingerprint="sha256:" + "5" * 64,
        )
        longer = evaluate_event(
            {"event_id": "e-2", "sony_start_s": 0.0, "sony_end_s": 19 / 6},
            quality_fixture(set(range(20)) - {8, 9, 10, 11}),
            effective_analysis_fps=6.0,
            analysis_fingerprint="sha256:" + "6" * 64,
        )

        self.assertEqual(exact["kvalitet"]["najduza_praznina_s"], 0.5)
        self.assertEqual(exact["status"], "dostupno")
        self.assertGreater(longer["kvalitet"]["najduza_praznina_s"], 0.5)
        self.assertEqual(longer["status"], "niska_pouzdanost")

    def test_insufficient_data_has_no_score_and_strict_json(self):
        frames = quality_fixture(range(5), total=12)
        frames[1]["timestamp_s"] = frames[0]["timestamp_s"]
        frames[2]["brzina_ulaska_norm"] = None
        frames[2]["rotacija_trupa_2d_dps"] = None

        result = evaluate_event(
            {"event_id": "e-1", "sony_start_s": 0.0, "sony_end_s": 11 / 6},
            frames,
            effective_analysis_fps=6.0,
            analysis_fingerprint="sha256:" + "7" * 64,
        )

        self.assertEqual(result["status"], "nedovoljno_podataka")
        self.assertIsNone(result["predlozena_ocena"])
        self.assertIn("5 validnih", result["razlog"])
        json.dumps(result, allow_nan=False)

    def test_rotation_direction_uses_signed_sum(self):
        frames = available_frame_fixture()
        for frame in frames:
            frame["rotacija_trupa_2d_dps"] *= -1

        result = evaluate_event(
            {"event_id": "e-1", "sony_start_s": 10.0, "sony_end_s": 12.0},
            frames,
            effective_analysis_fps=6.0,
            analysis_fingerprint="sha256:" + "8" * 64,
        )

        self.assertEqual(result["imu_eksperimentalno"]["dominantna_rotacija"], "levo")
        self.assertNotIn("snaga", result["razlog"].lower())
        self.assertNotIn("sila", result["razlog"].lower())

    def test_percentile_ties_cite_the_earliest_matching_time(self):
        frames = quality_fixture(range(12), total=12)
        for frame in frames:
            frame["brzina_ulaska_norm"] = 1.0
            frame["rotacija_trupa_2d_dps"] = 100.0
            frame["intenzitet_pokreta_0_100"] = 50.0

        result = evaluate_event(
            {"event_id": "e-1", "sony_start_s": 0.0, "sony_end_s": 11 / 6},
            frames,
            effective_analysis_fps=6.0,
            analysis_fingerprint="sha256:" + "9" * 64,
        )
        evidence = {item["metrika"]: item for item in result["dokazi"]}

        self.assertEqual(evidence["brzina_ulaska_norm"]["sony_s"], 0.0)
        self.assertEqual(evidence["ugaona_brzina_trupa_2d"]["sony_s"], 0.0)
        self.assertEqual(evidence["intenzitet_pokreta"]["sony_s"], 0.0)
        self.assertEqual(evidence["proxy_ubrzanja"]["sony_s"], 0.167)

    def test_evaluator_recomputes_canonical_intensity(self):
        frames = available_frame_fixture()
        tampered = [dict(frame, intenzitet_pokreta_0_100=0.0) for frame in frames]
        fingerprint = "sha256:" + "a" * 64

        canonical = evaluate_event(
            {"event_id": "e-1", "sony_start_s": 10.0, "sony_end_s": 12.0},
            frames,
            effective_analysis_fps=6.0,
            analysis_fingerprint=fingerprint,
        )
        recomputed = evaluate_event(
            {"event_id": "e-1", "sony_start_s": 10.0, "sony_end_s": 12.0},
            tampered,
            effective_analysis_fps=6.0,
            analysis_fingerprint=fingerprint,
        )

        self.assertEqual(recomputed, canonical)

    def test_saved_acceleration_uses_precomputed_precise_dt_value(self):
        frames = available_frame_fixture()
        for index, frame in enumerate(frames):
            frame["timestamp_s"] = round(frame["timestamp_s"], 3)
            frame["proxy_ubrzanja_norm_s2"] = None if index == 0 else 2.0

        result = evaluate_event(
            {"event_id": "e-1", "sony_start_s": 10.0, "sony_end_s": 12.0},
            frames,
            effective_analysis_fps=6.0,
            analysis_fingerprint="sha256:" + "c" * 64,
        )

        self.assertEqual(result["pokazatelji"]["acceleration_peak"], 2.0)

    def test_intensity_is_smoothed_before_event_slice(self):
        frames = quality_fixture(range(3), total=3)
        frames[0].update({
            "brzina_ulaska_norm": 0.0,
            "rotacija_trupa_2d_dps": 0.0,
        })
        frames[1].update({
            "brzina_ulaska_norm": 4.0,
            "rotacija_trupa_2d_dps": 540.0,
        })
        frames[2].update({
            "brzina_ulaska_norm": 0.0,
            "rotacija_trupa_2d_dps": 0.0,
        })

        result = evaluate_event(
            {"event_id": "e-1", "sony_start_s": 0.15, "sony_end_s": 0.18},
            frames,
            effective_analysis_fps=6.0,
            analysis_fingerprint="sha256:" + "d" * 64,
        )

        self.assertEqual(result["pokazatelji"]["intensity_peak"], 33.333333)

    def test_missing_required_metric_is_null_and_named_in_reason(self):
        frames = available_frame_fixture()
        for frame in frames:
            frame["rotacija_trupa_2d_dps"] = None

        result = evaluate_event(
            {"event_id": "e-1", "sony_start_s": 10.0, "sony_end_s": 12.0},
            frames,
            effective_analysis_fps=6.0,
            analysis_fingerprint="sha256:" + "b" * 64,
        )

        self.assertEqual(result["status"], "niska_pouzdanost")
        self.assertIsNone(result["predlozena_ocena"])
        self.assertEqual(
            result["nedostaju_metrike"],
            ["rotation_peak", "intensity_peak"],
        )
        self.assertIn("rotation_peak", result["razlog"])
        self.assertIn("intensity_peak", result["razlog"])

    def test_fingerprint_changes_with_each_analysis_input(self):
        review = {
            "sources": {
                "sony": {"sha256": "a" * 64},
                "iphone": {"sha256": "b" * 64},
            },
            "effective_analysis_fps": 6.0,
            "selected_track_id": 4,
        }
        event = {
            "event_id": "e-1",
            "sony_start_s": 10.0,
            "sony_end_s": 12.0,
            "iphone_start_s": 13.0,
            "iphone_end_s": 15.0,
        }

        baseline = compute_analysis_fingerprint(review, event)
        changed = dict(event, sony_end_s=12.1)

        self.assertRegex(baseline, r"^sha256:[0-9a-f]{64}$")
        self.assertNotEqual(baseline, compute_analysis_fingerprint(review, changed))
        self.assertEqual(baseline, compute_analysis_fingerprint(review, event))

        nested_track_review = dict(review)
        nested_track_review.pop("selected_track_id")
        nested_track_review["pose_analysis"] = {"selected_track_id": 4}
        changed_track_review = dict(nested_track_review)
        changed_track_review["pose_analysis"] = {"selected_track_id": 5}
        self.assertNotEqual(
            compute_analysis_fingerprint(nested_track_review, event),
            compute_analysis_fingerprint(changed_track_review, event),
        )

        for camera in ("sony", "iphone"):
            with self.subTest(camera=camera):
                unsigned = copy.deepcopy(review)
                unsigned["sources"][camera].pop("sha256")
                with self.assertRaisesRegex(ValueError, camera):
                    compute_analysis_fingerprint(unsigned, event)


if __name__ == "__main__":
    unittest.main()
