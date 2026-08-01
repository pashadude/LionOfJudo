import json
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

import tempfile
import numpy as np

from pipeline.clip_extractor import verify_media_export as canonical_verify_media_export
from pipeline.face_blur import BlurReport
from pipeline.video_review_contract import AnchorPair
from pipeline.video_review_import import (
    _export_private_clip,
    _merge_trainer_annotations,
    import_session,
    make_side_by_side,
    run_pose_analysis,
    verify_media_export,
    write_review_json,
)


class VideoReviewImportTests(TestCase):
    def setUp(self):
        self._probe_fps_patcher = patch(
            "pipeline.video_review_import.probe_fps",
            side_effect=self._fps_for_path,
        )
        self.mock_probe_fps = self._probe_fps_patcher.start()
        self.addCleanup(self._probe_fps_patcher.stop)
        self._privacy_patcher = patch(
            "pipeline.video_review_import.build_privacy_processor",
            return_value=self._fake_privacy_processor,
        )
        self.mock_build_privacy = self._privacy_patcher.start()
        self.addCleanup(self._privacy_patcher.stop)

    @patch("pipeline.video_review_import.subprocess.run")
    def test_side_by_side_maps_iphone_with_affine_setpts(self, mock_run):
        def create_output(command, **_kwargs):
            Path(command[-1]).write_bytes(b"video")

        mock_run.side_effect = create_output
        with self.subTest("command construction"):
            with self._temporary_directory() as output_dir:
                output_dir = Path(output_dir)
                with patch(
                    "pipeline.video_review_import.probe_duration",
                    return_value=126.0,
                ):
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

    @patch("pipeline.video_review_import.subprocess.run")
    def test_side_by_side_can_export_only_the_confirmed_session_window(self, mock_run):
        def create_output(command, **_kwargs):
            Path(command[-1]).write_bytes(b"video")

        mock_run.side_effect = create_output
        with self._temporary_directory() as output_dir, patch(
            "pipeline.video_review_import.probe_duration",
            return_value=8.0,
        ):
            make_side_by_side(
                Path("sony.mp4"),
                Path("iphone.mov"),
                slope=1.0,
                intercept=-3.0,
                end_s=136.0,
                output=Path(output_dir) / "out.mp4",
                start_s=128.0,
            )

        command = mock_run.call_args.args[0]
        self.assertEqual(
            [command[index + 1] for index, value in enumerate(command) if value == "-ss"],
            ["128.000", "131.000"],
        )
        self.assertEqual(
            [command[index + 1] for index, value in enumerate(command) if value == "-t"],
            ["8.000", "8.000"],
        )
        self.assertIn("setpts=PTS-STARTPTS", " ".join(command))

    def test_side_by_side_rejects_missing_output_after_ffmpeg(self):
        with patch("pipeline.video_review_import.subprocess.run"):
            with self._temporary_directory() as raw_root:
                with self.assertRaises(ValueError):
                    make_side_by_side(
                        Path("sony.mp4"), Path("iphone.mov"), 1.0, 0.0, 2.0,
                        Path(raw_root) / "missing.mp4",
                    )

    def test_side_by_side_rejects_zero_byte_output_after_ffmpeg(self):
        def create_zero_byte(command, **_kwargs):
            Path(command[-1]).touch()

        with patch(
            "pipeline.video_review_import.subprocess.run",
            side_effect=create_zero_byte,
        ):
            with self._temporary_directory() as raw_root:
                with self.assertRaises(ValueError):
                    make_side_by_side(
                        Path("sony.mp4"), Path("iphone.mov"), 1.0, 0.0, 2.0,
                        Path(raw_root) / "empty.mp4",
                    )

    def test_side_by_side_rejects_duration_invalid_output_after_ffmpeg(self):
        def create_output(command, **_kwargs):
            Path(command[-1]).write_bytes(b"video")

        with patch(
            "pipeline.video_review_import.subprocess.run",
            side_effect=create_output,
        ), patch("pipeline.video_review_import.probe_duration", return_value=5.0):
            with self._temporary_directory() as raw_root:
                with self.assertRaises(ValueError):
                    make_side_by_side(
                        Path("sony.mp4"), Path("iphone.mov"), 1.0, 0.0, 2.0,
                        Path(raw_root) / "wrong-duration.mp4",
                    )

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

    @patch(
        "pipeline.video_review_import.run_pose_analysis",
        return_value=[
            {
                "event_id": "e-001",
                "sony_start_s": 11.0,
                "sony_end_s": 14.0,
            }
        ],
    )
    @patch("pipeline.video_review_import.make_side_by_side")
    @patch("pipeline.video_review_import.cut_clip")
    def test_import_rejects_source_inside_managed_session_directory(
        self, mock_cut, mock_composite, mock_pose
    ):
        with self._temporary_directory() as raw_root:
            root = Path(raw_root)
            output_dir = root / "session"
            sony = output_dir / "events" / "e-001" / "sony.mp4"
            sony.parent.mkdir(parents=True)
            sony.write_bytes(b"immutable sony source")
            iphone = root / "iphone.mov"
            iphone.write_bytes(b"immutable iphone source")
            mock_cut.side_effect = self._touch_result
            mock_composite.side_effect = self._touch_result

            with patch(
                "pipeline.video_review_import.probe_duration",
                side_effect=self._duration_for_path,
            ):
                with self.assertRaisesRegex(ValueError, "izvan direktorijuma sesije"):
                    import_session(
                        sony,
                        iphone,
                        output_dir,
                        self._anchors(),
                        20.0,
                        (1, 2, 3, 4),
                    )

            self.assertEqual(sony.read_bytes(), b"immutable sony source")
            mock_pose.assert_not_called()
            mock_cut.assert_not_called()
            mock_composite.assert_not_called()

    def test_run_pose_analysis_uses_seeded_track_and_stops_at_cutoff(self):
        frames = [self._pose_frame(0.0), self._pose_frame(20.0), self._pose_frame(40.0)]
        capture = _FakeCapture(frames, fps=10.0)
        model = _FakePoseModel(
            [
                _FakeResult(
                    [((100, 100, 120, 180), 2, self._keypoints(0.0)),
                     ((0, 0, 20, 80), 7, self._keypoints(0.0))]
                ),
                _FakeResult(
                    [((100, 100, 120, 180), 2, self._keypoints(0.0)),
                     ((0, 0, 20, 80), 7, self._keypoints(20.0))]
                ),
                _FakeResult(
                    [((100, 100, 120, 180), 2, self._keypoints(0.0)),
                     ((0, 0, 20, 80), 7, self._keypoints(40.0))]
                ),
            ]
        )

        result = run_pose_analysis(
            Path("sony.mp4"),
            0.0,
            0.3,
            (0, 0, 20, 80),
            model=model,
            video_capture_factory=lambda _path: capture,
            fps=10.0,
        )

        self.assertEqual(result["selected_track_id"], 7)
        self.assertEqual(len(result["frame_metrics"]), 3)
        self.assertTrue(result["events"])
        self.assertEqual(result["events"][0]["event_id"], "e-001")
        self.assertTrue(
            all(event["sony_end_s"] <= 0.3 for event in result["events"])
        )
        self.assertTrue(
            all(not event["iskljuceno_iz_statistike"] for event in result["events"])
        )
        self.assertIn("brzina_ulaska_norm", result["events"][0])
        self.assertIn("intenzitet_pokreta_0_100", result["events"][0])
        self.assertEqual(model.track_calls, 3)

    def test_run_pose_analysis_samples_source_stride_and_preserves_timestamps(self):
        frames = [self._pose_frame(float(index)) for index in range(10)]
        capture = _FakeCapture(frames, fps=10.0)
        model = _FakePoseModel(
            [
                _FakeResult([((0, 0, 20, 80), 7, self._keypoints(float(index)))])
                for index in (0, 3, 6)
            ]
        )

        result = run_pose_analysis(
            Path("sony.mp4"),
            0.0,
            0.9,
            (0, 0, 20, 80),
            model=model,
            video_capture_factory=lambda _path: capture,
            fps=10.0,
            analysis_fps=3.0,
        )

        self.assertEqual(result["source_fps"], 10.0)
        self.assertEqual(result["requested_analysis_fps"], 3.0)
        self.assertEqual(result["effective_analysis_fps"], 10.0 / 3.0)
        self.assertEqual(result["stride"], 3)
        self.assertEqual(
            [metric["timestamp_s"] for metric in result["frame_metrics"]],
            [0.0, 0.3, 0.6],
        )
        self.assertEqual(model.track_calls, 3)

    def test_run_pose_analysis_starts_at_anchor_frame_and_uses_local_model_path(self):
        frames = [self._pose_frame(float(index)) for index in range(8)]
        capture = _FakeCapture(frames, fps=10.0)
        model = _FakePoseModel(
            [_FakeResult([((0, 0, 20, 80), 7, self._keypoints(0.0))])]
        )

        result = run_pose_analysis(
            Path("sony.mp4"),
            0.52,
            0.7,
            (0, 0, 20, 80),
            model=model,
            video_capture_factory=lambda _path: capture,
            fps=10.0,
            analysis_fps=3.0,
            model_path=Path("/tmp/local-yolo.pt"),
        )

        self.assertEqual(capture.positioned_at, 6)
        self.assertEqual(result["analysis_start_s"], 0.52)
        self.assertEqual(result["frame_metrics"][0]["timestamp_s"], 0.6)
        self.assertEqual(model.track_calls, 1)

    def test_run_pose_analysis_rejects_analysis_rate_above_source_fps(self):
        with self.assertRaisesRegex(ValueError, "analysis_fps"):
            run_pose_analysis(
                Path("sony.mp4"),
                0.0,
                1.0,
                (0, 0, 20, 80),
                model=_FakePoseModel([]),
                video_capture_factory=lambda _path: _FakeCapture([], fps=10.0),
                fps=10.0,
                analysis_fps=10.1,
            )

    @patch("pipeline.video_review_import.run_pose_analysis")
    @patch("pipeline.video_review_import.make_side_by_side")
    @patch("pipeline.video_review_import.cut_clip")
    def test_import_persists_pose_frame_and_event_metrics(
        self, mock_cut, mock_composite, mock_pose
    ):
        with self._temporary_directory() as raw_root:
            root = Path(raw_root)
            sony, iphone = self._sources(root)
            mock_cut.side_effect = self._touch_result
            mock_composite.side_effect = self._touch_result
            mock_pose.return_value = {
                "selected_track_id": 7,
                "fps": 10.0,
                "frame_metrics": [{"timestamp_s": 1.0, "vidljivo": True}],
                "events": [{
                    "event_id": "e-1",
                    "sony_start_s": 11.0,
                    "sony_end_s": 14.0,
                    "metrics": {"brzina_ulaska_norm": 0.8},
                }],
            }
            with patch(
                "pipeline.video_review_import.probe_duration",
                side_effect=self._duration_for_path,
            ):
                review_path = import_session(
                    sony, iphone, root / "session", self._anchors(), 20.0, (1, 2, 3, 4)
                )
            review = json.loads(review_path.read_text(encoding="utf-8"))
            self.assertEqual(review["frame_metrics"][0]["timestamp_s"], 1.0)
            self.assertEqual(review["events"][0]["brzina_ulaska_norm"], 0.8)
            self.assertTrue(
                any(event.get("prijavljen_povredni_dogadjaj") for event in review["events"])
            )
            self.assertTrue((root / "session" / "analysis" / "frame_metrics.json").exists())
            self.assertTrue((root / "session" / "analysis" / "event_metrics.json").exists())

    @patch("pipeline.video_review_import.run_pose_analysis")
    @patch("pipeline.video_review_import.make_side_by_side")
    @patch("pipeline.video_review_import.cut_clip")
    def test_import_persists_probed_source_fps(
        self, mock_cut, mock_composite, mock_pose
    ):
        with self._temporary_directory() as raw_root:
            root = Path(raw_root)
            sony, iphone = self._sources(root)
            mock_cut.side_effect = self._touch_result
            mock_composite.side_effect = self._touch_result
            mock_pose.return_value = {
                "fps": 29.97,
                "frame_metrics": [],
                "events": [],
            }
            with patch(
                "pipeline.video_review_import.probe_duration",
                side_effect=self._duration_for_path,
            ):
                review_path = import_session(
                    sony, iphone, root / "session", self._anchors(), 20.0, (1, 2, 3, 4)
                )
            review = json.loads(review_path.read_text(encoding="utf-8"))
            self.assertEqual(review["sony_fps"], 29.97)
            self.assertEqual(review["iphone_fps"], 59.94)
            self.assertEqual(review["source_fps"], 29.97)
            self.assertIsNone(review["requested_analysis_fps"])
            self.assertEqual(review["effective_analysis_fps"], 29.97)
            self.assertEqual(review["stride"], 1)
            self.assertGreaterEqual(self.mock_probe_fps.call_count, 2)

    @patch("pipeline.video_review_import.run_pose_analysis")
    @patch("pipeline.video_review_import.make_side_by_side")
    @patch("pipeline.video_review_import.cut_clip")
    def test_import_persists_sampling_profile_and_starts_at_first_anchor(
        self, mock_cut, mock_composite, mock_pose
    ):
        with self._temporary_directory() as raw_root:
            root = Path(raw_root)
            sony, iphone = self._sources(root)
            mock_cut.side_effect = self._touch_result
            mock_composite.side_effect = self._touch_result
            mock_pose.return_value = {
                "source_fps": 29.97,
                "requested_analysis_fps": 3.0,
                "effective_analysis_fps": 29.97 / 10.0,
                "stride": 10,
                "analysis_start_s": 10.0,
                "fps": 29.97 / 10.0,
                "frame_metrics": [],
                "events": [],
            }
            with patch(
                "pipeline.video_review_import.probe_duration",
                side_effect=self._duration_for_path,
            ):
                review_path = import_session(
                    sony,
                    iphone,
                    root / "session",
                    self._anchors(),
                    20.0,
                    (1, 2, 3, 4),
                    analysis_fps=3.0,
                    model_path=Path("/tmp/local-yolo.pt"),
                    device="cpu",
                )

            review = json.loads(review_path.read_text(encoding="utf-8"))
            self.assertEqual(review["source_fps"], 29.97)
            self.assertEqual(review["requested_analysis_fps"], 3.0)
            self.assertEqual(review["effective_analysis_fps"], 29.97 / 10.0)
            self.assertEqual(review["stride"], 10)
            self.assertEqual(review["analysis_start_s"], 10.0)
            self.assertIn("grubo uzorkovanje", review["analysis_limitation"])
            mock_pose.assert_called_once_with(
                sony.resolve(),
                10.0,
                20.0,
                [1.0, 2.0, 3.0, 4.0],
                analysis_fps=3.0,
                model_path=Path("/tmp/local-yolo.pt"),
                device="cpu",
                event_threshold=0.5,
            )

    @patch("pipeline.video_review_import.run_pose_analysis", return_value=[])
    def test_import_fails_visibly_when_sony_fps_is_missing(
        self, _mock_pose
    ):
        with self._temporary_directory() as raw_root:
            root = Path(raw_root)
            sony, iphone = self._sources(root)
            with patch(
                "pipeline.video_review_import.probe_fps",
                side_effect=ValueError("fps nije dostupan"),
            ), patch(
                "pipeline.video_review_import.probe_duration",
                side_effect=self._duration_for_path,
            ):
                with self.assertRaisesRegex(ValueError, "Sony FPS"):
                    import_session(
                        sony, iphone, root / "session", self._anchors(), 20.0, (1, 2, 3, 4)
                    )

    @patch("pipeline.video_review_import.run_pose_analysis", return_value=[])
    @patch("pipeline.video_review_import.make_side_by_side")
    @patch("pipeline.video_review_import.cut_clip")
    def test_import_fails_closed_when_privacy_verification_fails(
        self, mock_cut, mock_composite, _mock_pose
    ):
        with self._temporary_directory() as raw_root:
            root = Path(raw_root)
            sony, iphone = self._sources(root)
            mock_cut.side_effect = self._touch_result
            mock_composite.side_effect = self._touch_result

            def reject(_raw_path, _output_path):
                return BlurReport(
                    total_frames=1,
                    first_pass_candidates=1,
                    second_pass_candidates=1,
                    privacy_verified=False,
                    failure_reason="lice je ostalo vidljivo",
                )

            with patch(
                "pipeline.video_review_import.probe_duration",
                side_effect=self._duration_for_path,
            ):
                with self.assertRaisesRegex(ValueError, "lice je ostalo vidljivo"):
                    import_session(
                        sony,
                        iphone,
                        root / "session",
                        self._anchors(),
                        20.0,
                        (1, 2, 3, 4),
                        privacy_processor=reject,
                    )

            self.assertFalse((root / "session" / "review.json").exists())
            self.assertEqual(
                list((root / "session" / "previews").glob("*.mp4")), []
            )

    def test_failed_private_clip_export_preserves_previous_verified_output(self):
        with self._temporary_directory() as raw_root:
            root = Path(raw_root)
            source = root / "source.mp4"
            output = root / "events" / "e-1" / "sony.mp4"
            source.write_bytes(b"source")
            output.parent.mkdir(parents=True)
            output.write_bytes(b"previous-verified")

            def reject(_raw_path, _output_path):
                return BlurReport(
                    total_frames=1,
                    first_pass_candidates=1,
                    second_pass_candidates=1,
                    privacy_verified=False,
                    failure_reason="lice je ostalo vidljivo",
                )

            with patch(
                "pipeline.video_review_import.cut_clip",
                side_effect=self._touch_result,
            ) as mock_cut, patch(
                "pipeline.video_review_import.probe_duration",
                return_value=1.0,
            ):
                with self.assertRaisesRegex(ValueError, "lice je ostalo vidljivo"):
                    _export_private_clip(
                        source,
                        0.0,
                        1.0,
                        output,
                        2.0,
                        reject,
                    )

            self.assertEqual(mock_cut.call_args.kwargs["scale_height"], 1080)
            self.assertEqual(output.read_bytes(), b"previous-verified")

    @patch("pipeline.video_review_import.run_pose_analysis", return_value=[])
    @patch("pipeline.video_review_import.make_side_by_side")
    @patch("pipeline.video_review_import.cut_clip")
    def test_end_of_source_cutoff_has_preceding_injury_window(
        self, mock_cut, mock_composite, _mock_pose
    ):
        with self._temporary_directory() as raw_root:
            root = Path(raw_root)
            sony, iphone = self._sources(root)
            mock_cut.side_effect = self._touch_result
            mock_composite.side_effect = self._touch_result

            def duration(path):
                path = Path(path)
                parent = path.parent.name
                if parent.startswith(".raw-private-"):
                    parent = path.parent.parent.name
                if path.name == "sony.mp4" and parent != "povreda":
                    return 10.0
                if path.name == "iphone.mov" and parent != "povreda":
                    return 60.0
                if path.name == "session_side_by_side.mp4":
                    return 10.0
                return self._duration_for_path(path)

            with patch(
                "pipeline.video_review_import.probe_duration",
                side_effect=duration,
            ):
                review_path = import_session(
                    sony,
                    iphone,
                    root / "session",
                    [
                        AnchorPair("pocetak", 2.0, 30.0, True, 3),
                        AnchorPair("kontrola", 4.0, 32.0, True, 3),
                    ],
                    10.0,
                    (1, 2, 3, 4),
                )
            review = json.loads(review_path.read_text(encoding="utf-8"))
            injury = next(
                event for event in review["events"]
                if event.get("prijavljen_povredni_dogadjaj")
            )
            self.assertEqual(injury["sony_end_s"], 10.0)
            self.assertEqual(injury["sony_start_s"], 9.0)
            self.assertTrue(injury["iskljuceno_iz_statistike"])
            self.assertTrue((root / "session" / "events" / "povreda" / "sony.mp4").exists())

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
            manifest = review["derived_media_manifest"]
            self.assertTrue(manifest)
            self.assertTrue(all(row["privacy_verified"] is True for row in manifest))
            self.assertEqual(
                {row["media_type"] for row in manifest},
                {"anchor_preview", "event_clip", "side_by_side"},
            )

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
                sony.resolve(),
                10.0,
                20.0,
                [1.0, 2.0, 3.0, 4.0],
                analysis_fps=None,
                model_path="yolo11x-pose.pt",
                device="mps",
                event_threshold=0.5,
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
            with patch("pipeline.clip_extractor.probe_duration", return_value=4.0):
                with self.assertRaises(ValueError):
                    verify_media_export(output, expected_duration_s=2.0)

    def test_importer_uses_canonical_media_verifier_with_injectable_probe(self):
        self.assertIs(verify_media_export, canonical_verify_media_export)
        with self._temporary_directory() as output_dir:
            output = Path(output_dir) / "clip.mp4"
            output.write_bytes(b"video")
            calls = []

            duration = verify_media_export(
                output,
                expected_duration_s=2.0,
                probe=lambda path: calls.append(Path(path)) or 2.0,
            )

        self.assertEqual(duration, 2.0)
        self.assertEqual(calls, [output])

    def test_force_reimport_preserves_matches_and_orphans_unmatched_annotations(self):
        fresh = {
            "injury_cutoff_s": 20.0,
            "events": [
                {
                    "event_id": "e-match",
                    "sony_start_s": 11.0,
                    "sony_end_s": 14.0,
                    "potvrdena_tehnika": None,
                    "ocena": None,
                    "napomena": None,
                    "iskljuceno_iz_statistike": False,
                },
                {
                    "event_id": "povreda",
                    "sony_start_s": 20.0,
                    "sony_end_s": 21.0,
                    "prijavljen_povredni_dogadjaj": True,
                    "iskljuceno_iz_statistike": True,
                },
            ],
        }
        previous = {
            "events": [
                {
                    "event_id": "e-match",
                    "sony_start_s": 11.0,
                    "sony_end_s": 14.0,
                    "potvrdena_tehnika": "O-soto-gari",
                    "ocena": 4,
                    "napomena": "Sačuvaj me.",
                },
                {
                    "event_id": "e-stale",
                    "sony_start_s": 25.0,
                    "sony_end_s": 30.0,
                    "media": {"sony": "/media/events/e-stale/sony.mp4"},
                    "potvrdena_tehnika": "Uki-goshi",
                    "ocena": 2,
                    "napomena": "Ne vraćaj kao aktivan događaj.",
                },
            ]
        }

        merged = _merge_trainer_annotations(fresh, previous)

        self.assertEqual([event["event_id"] for event in merged["events"]], ["e-match", "povreda"])
        self.assertEqual(merged["events"][0]["potvrdena_tehnika"], "O-soto-gari")
        self.assertEqual(merged["events"][0]["ocena"], 4)
        self.assertEqual(merged["events"][0]["napomena"], "Sačuvaj me.")
        self.assertEqual(len(merged["orphaned_annotations"]), 1)
        orphan = merged["orphaned_annotations"][0]
        self.assertEqual(orphan["source_event_id"], "e-stale")
        self.assertEqual(orphan["potvrdena_tehnika"], "Uki-goshi")
        self.assertNotIn("media", orphan)
        self.assertTrue(all(event["sony_end_s"] <= 21.0 for event in merged["events"]))

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
    def _fake_privacy_processor(raw_path, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(Path(raw_path).read_bytes())
        return BlurReport(
            total_frames=1,
            first_pass_candidates=1,
            second_pass_candidates=0,
            privacy_verified=True,
        )

    @staticmethod
    def _duration_for_path(path):
        path = Path(path)
        parent_name = path.parent.name
        if parent_name.startswith(".raw-private-"):
            parent_name = path.parent.parent.name
        if path.name in {"sony.mp4", "iphone.mov"} and parent_name not in {
            "e-1", "povreda", "previews"
        }:
            return 60.0
        if parent_name == "e-1":
            return 3.0
        if parent_name == "povreda":
            return 1.0
        if path.name == "session_side_by_side.mp4":
            return 20.0
        return 4.0

    @staticmethod
    def _fps_for_path(path):
        return 29.97 if Path(path).suffix.lower() == ".mp4" else 59.94

    @staticmethod
    def _pose_frame(offset):
        return np.zeros((32, 32, 3), dtype=np.uint8)

    @staticmethod
    def _keypoints(offset):
        keypoints = np.zeros((17, 3), dtype=float)
        keypoints[:, 2] = 1.0
        keypoints[11, :2] = [10.0 + offset, 20.0]
        keypoints[12, :2] = [20.0 + offset, 20.0]
        keypoints[5, :2] = [10.0 + offset, 0.0]
        keypoints[6, :2] = [20.0 + offset, 0.0]
        keypoints[15, :2] = [5.0 + offset, 30.0]
        keypoints[16, :2] = [25.0 + offset, 30.0]
        return keypoints


if __name__ == "__main__":
    main()


class _FakeCapture:
    def __init__(self, frames, fps):
        self.frames = iter(frames)
        self.fps = fps
        self.positioned_at = None

    def get(self, _property):
        return self.fps

    def read(self):
        try:
            return True, next(self.frames)
        except StopIteration:
            return False, None

    def set(self, _property, value):
        self.positioned_at = int(value)
        return True

    def release(self):
        return None


class _FakeTensor:
    def __init__(self, value):
        self.value = value

    def tolist(self):
        return self.value.tolist() if hasattr(self.value, "tolist") else self.value

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self.value)


class _FakeResult:
    def __init__(self, detections):
        self.boxes = SimpleNamespace(
            xyxy=_FakeTensor(np.asarray([item[0] for item in detections], dtype=float)),
            id=_FakeTensor(np.asarray([item[1] for item in detections], dtype=float)),
        )
        self.keypoints = SimpleNamespace(
            data=_FakeTensor(np.asarray([item[2] for item in detections], dtype=float))
        )


class _FakePoseModel:
    def __init__(self, results):
        self.results = iter(results)
        self.track_calls = 0

    def track(self, _frame, **_kwargs):
        self.track_calls += 1
        return [next(self.results)]
