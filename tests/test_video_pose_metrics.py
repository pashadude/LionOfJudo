import json
import unittest

import numpy as np

from pipeline.video_pose_metrics import compute_pose_metrics


def pose(hip_x, hip_y, left_shoulder, right_shoulder):
    keypoints = np.zeros((17, 3), dtype=float)
    keypoints[:, 2] = 1.0
    keypoints[11, :2] = [hip_x - 5, hip_y]
    keypoints[12, :2] = [hip_x + 5, hip_y]
    keypoints[5, :2] = left_shoulder
    keypoints[6, :2] = right_shoulder
    keypoints[15, :2] = [hip_x - 20, hip_y + 40]
    keypoints[16, :2] = [hip_x + 20, hip_y + 40]
    return keypoints


class PoseMetricsTests(unittest.TestCase):
    def test_entry_speed_is_torso_normalized(self):
        frames = [
            pose(0, 100, (-5, 80), (5, 80)),
            pose(10, 100, (5, 80), (15, 80)),
        ]

        metrics = compute_pose_metrics(frames, fps=10.0)

        self.assertEqual(metrics[1].brzina_ulaska_norm_s, 5.0)

    def test_rotation_uses_the_shoulder_line_angle(self):
        frames = [
            pose(0, 100, (-10, 80), (10, 80)),
            pose(0, 100, (0, 90), (0, 70)),
        ]

        metric = compute_pose_metrics(frames, fps=10.0)[1]

        self.assertAlmostEqual(metric.shoulder_angle_deg, -90.0)
        self.assertAlmostEqual(metric.rotation_2d_dps, -900.0)

    def test_rejects_invalid_fps(self):
        with self.assertRaises(ValueError):
            compute_pose_metrics([pose(0, 100, (-5, 80), (5, 80))], fps=0)

    def test_rejects_non_finite_pose_values(self):
        invalid = pose(0, 100, (-5, 80), (5, 80))
        invalid[5, 0] = np.nan

        with self.assertRaises(ValueError):
            compute_pose_metrics([invalid], fps=10.0)

    def test_low_visibility_does_not_invent_an_edge_value(self):
        hidden = pose(10, 100, (5, 80), (15, 80))
        hidden[5, 2] = 0.2

        metric = compute_pose_metrics(
            [pose(0, 100, (-5, 80), (5, 80)), hidden], fps=10.0
        )[1]

        self.assertFalse(metric.vidljivo)
        self.assertIsNone(metric.hip_midpoint)
        self.assertIsNone(metric.brzina_ulaska_norm_s)

    def test_short_interior_gap_is_interpolated_and_marked(self):
        frames = [
            pose(0, 100, (-5, 80), (5, 80)),
            pose(10, 100, (5, 80), (15, 80)),
            pose(20, 100, (15, 80), (25, 80)),
        ]
        frames[1][5:7, 2] = 0.1
        frames[1][11:13, 2] = 0.1

        metric = compute_pose_metrics(frames, fps=10.0)[1]

        self.assertFalse(metric.vidljivo)
        self.assertTrue(metric.interpolirano)
        self.assertIsNotNone(metric.hip_midpoint)
        self.assertAlmostEqual(metric.hip_midpoint[0], 10.0)

    def test_gap_longer_than_five_frames_stays_missing(self):
        frames = [pose(i * 10, 100, (i * 10 - 5, 80), (i * 10 + 5, 80))
                  for i in range(8)]
        for frame in frames[1:7]:
            frame[5:7, 2] = 0.1
            frame[11:13, 2] = 0.1

        metrics = compute_pose_metrics(frames, fps=10.0)

        self.assertTrue(all(m.hip_midpoint is None for m in metrics[1:7]))
        self.assertTrue(all(not m.interpolirano for m in metrics[1:7]))

    def test_metrics_are_json_safe(self):
        metrics = compute_pose_metrics(
            [pose(0, 100, (-5, 80), (5, 80))],
            fps=10.0,
            timestamps=[12.5],
        )

        json.dumps(metrics[0].to_dict(), allow_nan=False)
        self.assertEqual(metrics[0].timestamp_s, 12.5)


if __name__ == "__main__":
    unittest.main()
