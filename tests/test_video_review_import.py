import json
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch

import tempfile

from pipeline.video_review_contract import AnchorPair
from pipeline.video_review_import import (
    import_session,
    make_side_by_side,
    verify_media_export,
    write_review_json,
)


class VideoReviewImportTests(TestCase):
    @patch("pipeline.video_review_import.subprocess.run")
    def test_side_by_side_maps_iphone_with_affine_setpts(self, mock_run):
        with self.subTest("command construction"):
            with self._temporary_directory() as output_dir:
                output_dir = Path(output_dir)
                make_side_by_side(
                    Path("sony.mp4"),
                    Path("iphone.mov"),
                    slope=1.002,
                    intercept=-19.4,
                    end_s=126.0,
                    output=output_dir / "out.mp4",
                )

        command = mock_run.call_args.args[0]
        command_text = " ".join(command)
        self.assertIn("setpts=1.002*PTS-19.4/TB", command_text)
        self.assertIn("trim=start=0:end=126.000", command_text)
        self.assertIn("-map", command)
        self.assertIn("0:a?", command)

    def test_side_by_side_rejects_invalid_range_before_ffmpeg(self):
        with patch("pipeline.video_review_import.subprocess.run") as mock_run:
            with self.assertRaises(ValueError):
                make_side_by_side(
                    Path("sony.mp4"), Path("iphone.mov"), 1.0, 0.0, 0.0,
                    Path("out.mp4"),
                )
            mock_run.assert_not_called()

    def test_side_by_side_rejects_nonpositive_slope_before_ffmpeg(self):
        with patch("pipeline.video_review_import.subprocess.run") as mock_run:
            with self.assertRaises(ValueError):
                make_side_by_side(
                    Path("sony.mp4"), Path("iphone.mov"), 0.0, 0.0, 1.0,
                    Path("out.mp4"),
                )
            mock_run.assert_not_called()

    def test_review_json_is_written_atomically_with_strict_json(self):
        with self._temporary_directory() as output_dir:
            output_dir = Path(output_dir)
            output = write_review_json(
                output_dir, {"session_id": "demo", "events": []}
            )

            self.assertEqual(output.name, "review.json")
            self.assertTrue(output.read_text(encoding="utf-8").startswith("{"))
            self.assertFalse((output_dir / "review.json.tmp").exists())
            json.loads(output.read_text(encoding="utf-8"))
            strict_output = write_review_json(output_dir, {"bad": float("nan")})
            self.assertIsNone(
                json.loads(strict_output.read_text(encoding="utf-8"))["bad"]
            )

    def test_import_requires_two_confirmed_anchors_before_ffmpeg(self):
        with self._temporary_directory() as output_dir:
            output_dir = Path(output_dir)
            sony, iphone = self._sources(output_dir)
            with patch(
                "pipeline.video_review_import.probe_duration",
                return_value=60.0,
            ), patch("pipeline.video_review_import.subprocess.run") as mock_run:
                with self.assertRaises(ValueError):
                    import_session(
                        sony,
                        iphone,
                        output_dir / "session",
                        anchors=[
                            {"name": "pocetak", "sony_s": 10.0, "iphone_s": 30.0},
                            {
                                "name": "kontrola",
                                "sony_s": 20.0,
                                "iphone_s": 40.0,
                                "user_confirmed": True,
                                "triple_tap_count": 3,
                            },
                        ],
                        injury_cutoff_s=25.0,
                        blue_seed=(1, 2, 3, 4),
                    )
                mock_run.assert_not_called()

    @patch("pipeline.video_review_import.run_pose_analysis", return_value=[])
    @patch("pipeline.video_review_import.make_side_by_side")
    @patch("pipeline.video_review_import.cut_clip")
    def test_import_has_deterministic_layout_and_source_hashes(
        self, mock_cut, mock_composite, _mock_pose
    ):
        with self._temporary_directory() as root:
            root = Path(root)
            sony, iphone = self._sources(root)
            mock_cut.side_effect = self._touch_result
            mock_composite.side_effect = self._touch_result
            with patch(
                "pipeline.video_review_import.probe_duration",
                side_effect=self._duration_for_path,
            ):
                review_path = import_session(
                    sony,
                    iphone,
                    root / "session",
                    anchors=self._anchors(),
                    injury_cutoff_s=20.0,
                    blue_seed=(1, 2, 3, 4),
                )

            review = json.loads(review_path.read_text(encoding="utf-8"))
            self.assertEqual(
                ["analysis", "events", "media", "previews", "review.json"],
                [
                    path.name
                    for path in sorted(
                        review_path.parent.iterdir(), key=lambda item: item.name
                    )
                    if path.name in {"media", "events", "previews", "analysis", "review.json"}
                ],
            )
            self.assertEqual(review["sony_video"], str(sony.resolve()))
            self.assertEqual(review["iphone_video"], str(iphone.resolve()))
            self.assertEqual(len(review["sources"]["sony"]["sha256"]), 64)
            self.assertEqual(review["time_map"]["slope"], 1.0)
            self.assertEqual(review["injury_cutoff_s"], 20.0)
            self.assertTrue(
                any(event.get("prijavljen_povredni_dogadjaj") for event in review["events"])
            )
            self.assertTrue((review_path.parent / "media" / "session_side_by_side.mp4").exists())
            self.assertTrue((review_path.parent / "analysis" / "import_summary.json").exists())

    @patch(
        "pipeline.video_review_import.run_pose_analysis",
        return_value=[
            {
                "event_id": "e-1",
                "sony_start_s": 11.0,
                "sony_end_s": 14.0,
                "metrics": {"brzina_ulaska_norm": 0.4},
            }
        ],
    )
    @patch("pipeline.video_review_import.make_side_by_side")
    @patch("pipeline.video_review_import.cut_clip")
    def test_event_exports_are_source_relative_and_pose_stops_at_cutoff(
        self, mock_cut, mock_composite, mock_pose
    ):
        with self._temporary_directory() as raw_root:
            root = Path(raw_root)
            sony, iphone = self._sources(root)
            mock_cut.side_effect = self._touch_result
            mock_composite.side_effect = self._touch_result
            with patch(
                "pipeline.video_review_import.probe_duration",
                side_effect=self._duration_for_path,
            ):
                review_path = import_session(
                    sony, iphone, root / "session", self._anchors(), 20.0, (1, 2, 3, 4)
                )
            review = json.loads(review_path.read_text(encoding="utf-8"))
            event = review["events"][0]
            self.assertEqual(event["sony_start_s"], 11.0)
            self.assertEqual(event["sony_end_s"], 14.0)
            self.assertTrue((root / "session" / "events" / "e-1" / "sony.mp4").exists())
            self.assertTrue((root / "session" / "events" / "e-1" / "iphone.mp4").exists())
            mock_pose.assert_called_once_with(
                sony.resolve(), 0.0, 20.0, [1.0, 2.0, 3.0, 4.0]
            )

    @patch("pipeline.video_review_import.run_pose_analysis", return_value=[])
    @patch("pipeline.video_review_import.make_side_by_side")
    @patch("pipeline.video_review_import.cut_clip")
    def test_reimport_preserves_annotations_only_with_force(
        self, mock_cut, mock_composite, _mock_pose
    ):
        with self._temporary_directory() as root:
            root = Path(root)
            sony, iphone = self._sources(root)
            mock_cut.side_effect = self._touch_result
            mock_composite.side_effect = self._touch_result
            with patch(
                "pipeline.video_review_import.probe_duration",
                side_effect=self._duration_for_path,
            ):
                review_path = import_session(
                    sony, iphone, root / "session", self._anchors(), 20.0, (1, 2, 3, 4)
                )
            payload = json.loads(review_path.read_text(encoding="utf-8"))
            payload["trainer_annotations"] = {"napomena": "Dobar ulaz."}
            write_review_json(review_path.parent, payload)

            with patch(
                "pipeline.video_review_import.probe_duration",
                side_effect=self._duration_for_path,
            ):
                with self.assertRaises(ValueError):
                    import_session(
                        sony, iphone, review_path.parent, self._anchors(), 20.0, (1, 2, 3, 4)
                    )

            with patch(
                "pipeline.video_review_import.probe_duration",
                side_effect=self._duration_for_path,
            ):
                import_session(
                    sony,
                    iphone,
                    review_path.parent,
                    self._anchors(),
                    20.0,
                    (1, 2, 3, 4),
                    force_reimport=True,
                )
            self.assertEqual(
                json.loads(review_path.read_text(encoding="utf-8"))["trainer_annotations"],
                {"napomena": "Dobar ulaz."},
            )

    def test_verify_media_export_rejects_zero_bytes_and_wrong_duration(self):
        with self._temporary_directory() as output_dir:
            output_dir = Path(output_dir)
            output = output_dir / "clip.mp4"
            with self.assertRaises(ValueError):
                verify_media_export(output, expected_duration_s=2.0)
            output.write_bytes(b"video")
            with patch("pipeline.video_review_import.probe_duration", return_value=4.0):
                with self.assertRaises(ValueError):
                    verify_media_export(output, expected_duration_s=2.0)

    @staticmethod
    def _temporary_directory():
        return tempfile.TemporaryDirectory()

    @staticmethod
    def _sources(root):
        sony = root / "sony.mp4"
        iphone = root / "iphone.mov"
        sony.write_bytes(b"sony source")
        iphone.write_bytes(b"iphone source")
        return sony, iphone

    @staticmethod
    def _anchors():
        return [
            AnchorPair("pocetak", 10.0, 30.0, True, 3),
            AnchorPair("kontrola", 19.0, 39.0, True, 3),
        ]

    @staticmethod
    def _touch_result(*args, **kwargs):
        output = kwargs.get("output") or args[-1]
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"export")
        return output

    @staticmethod
    def _duration_for_path(path):
        path = Path(path)
        if path.name in {"sony.mp4", "iphone.mov"} and path.parent.name not in {
            "e-1", "povreda", "previews"
        }:
            return 60.0
        if path.parent.name == "e-1":
            return 3.0
        if path.parent.name == "povreda":
            return 1.0
        if path.name == "session_side_by_side.mp4":
            return 20.0
        return 4.0


if __name__ == "__main__":
    main()
