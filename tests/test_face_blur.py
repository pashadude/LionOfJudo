import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import cv2

from pipeline.face_blur import (
    BlurReport,
    PrivacyVerificationError,
    _blur_ellipse,
    _mux_original_audio,
    _pose_heads,
    _region_is_obscured,
    _yunet_faces,
    blur_all_faces,
    privatize_media,
    verify_blurred_against_source,
    verify_blurred_clip,
)


class FakeCapture:
    def __init__(self, frames, *, frame_count=None):
        self.frames = [frame.copy() for frame in frames]
        self.index = 0
        self.frame_count = len(frames) if frame_count is None else frame_count

    def isOpened(self):
        return True

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame.copy()

    def get(self, property_id):
        values = {5: 30.0, 3: 200.0, 4: 100.0, 7: float(self.frame_count)}
        return values.get(property_id, 0.0)

    def release(self):
        return None


class FakeWriter:
    def __init__(self):
        self.frames = []

    def isOpened(self):
        return True

    def write(self, frame):
        self.frames.append(frame.copy())

    def release(self):
        return None


class FaceBlurTests(unittest.TestCase):
    @staticmethod
    def checkerboard():
        y, x = np.indices((100, 200))
        board = ((x + y) % 2 * 255).astype(np.uint8)
        return np.repeat(board[:, :, None], 3, axis=2)

    def test_blur_all_faces_changes_every_candidate_region(self):
        original = self.checkerboard()
        capture = FakeCapture([original])
        writer = FakeWriter()

        report = blur_all_faces(
            object(),
            Path("input.mp4"),
            Path("output.mp4"),
            "cpu",
            capture_factory=lambda _path: capture,
            writer_factory=lambda *_args: writer,
            yunet=object(),
            candidate_detector=lambda *_args, **_kwargs: [
                (45, 50, 24),
                (155, 50, 24),
            ],
        )

        self.assertEqual(report.total_frames, 1)
        self.assertEqual(report.first_pass_candidates, 2)
        self.assertEqual(len(writer.frames), 1)
        output = writer.frames[0]
        for center_x in (45, 155):
            before = original[35:65, center_x - 15:center_x + 15]
            after = output[35:65, center_x - 15:center_x + 15]
            self.assertFalse(np.array_equal(before, after))
            self.assertLess(float(np.var(after)), float(np.var(before)))

    def test_yunet_keeps_low_score_candidates_for_fail_closed_verification(self):
        detector = Mock()
        detector.detect.return_value = (
            None,
            np.array(
                [
                    [0, 0, 500, 600, 0.31],
                    [10, 10, 20, 25, 0.56],
                ],
                dtype=np.float32,
            ),
        )

        regions = _yunet_faces(detector, np.zeros((2160, 3840, 3), dtype=np.uint8))

        self.assertEqual(regions, [(250, 300, 210), (20, 22, 21)])

    def test_pose_head_minimum_radius_scales_for_4k_frame(self):
        keypoints = np.zeros((17, 3), dtype=np.float32)
        keypoints[0] = (100, 100, 0.9)
        tensor = Mock()
        tensor.cpu.return_value.numpy.return_value = np.array([keypoints])
        result = Mock()
        result.keypoints.data = tensor
        model = Mock()
        model.predict.return_value = [result]

        regions = _pose_heads(
            model,
            np.zeros((2160, 3840, 3), dtype=np.uint8),
            "cpu",
            0.30,
        )

        self.assertEqual(regions, [(100, 100, 22)])

    def test_blur_covers_half_radius_detector_jitter(self):
        frame = self.checkerboard()

        _blur_ellipse(frame, 50, 50, 20)

        self.assertTrue(_region_is_obscured(frame, 60, 50, 20))

    def test_repair_pass_does_not_reblur_already_obscured_candidate(self):
        frame = self.checkerboard()
        _blur_ellipse(frame, 50, 50, 20)
        left_before = frame[30:70, 30:70].copy()
        right_before = frame[30:70, 130:170].copy()
        writer = FakeWriter()

        blur_all_faces(
            object(),
            Path("input.mp4"),
            Path("output.mp4"),
            "cpu",
            only_unobscured=True,
            capture_factory=lambda _path: FakeCapture([frame]),
            writer_factory=lambda *_args: writer,
            yunet=object(),
            candidate_detector=lambda *_args, **_kwargs: [
                (50, 50, 20),
                (150, 50, 20),
            ],
        )

        self.assertTrue(np.array_equal(writer.frames[0][30:70, 30:70], left_before))
        self.assertFalse(np.array_equal(writer.frames[0][30:70, 130:170], right_before))

    def test_blur_has_flat_privacy_core_that_survives_detector_jitter(self):
        rng = np.random.default_rng(7)
        frame = rng.integers(0, 256, size=(100, 200, 3), dtype=np.uint8)

        _blur_ellipse(frame, 50, 50, 12)

        core = frame[45:55, 45:55]
        self.assertEqual(np.unique(core.reshape(-1, 3), axis=0).shape[0], 1)

    def test_overlapping_candidates_cannot_restore_each_others_visible_core(self):
        frame = np.zeros((120, 220, 3), dtype=np.uint8)
        frame[:, :110] = (20, 20, 240)
        frame[:, 110:] = (240, 20, 20)
        writer = FakeWriter()

        blur_all_faces(
            object(),
            Path("input.mp4"),
            Path("output.mp4"),
            "cpu",
            capture_factory=lambda _path: FakeCapture([frame]),
            writer_factory=lambda *_args: writer,
            yunet=object(),
            candidate_detector=lambda *_args, **_kwargs: [
                (90, 60, 30),
                (125, 60, 30),
            ],
        )

        output = writer.frames[0]
        self.assertTrue(_region_is_obscured(output, 90, 60, 30))
        self.assertTrue(_region_is_obscured(output, 125, 60, 30))

    def test_overlapping_privacy_cores_survive_video_encoding(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source.mp4"
            private = root / "private.mp4"
            frame = np.zeros((120, 220, 3), dtype=np.uint8)
            frame[:, :110] = (20, 20, 240)
            frame[:, 110:] = (240, 20, 20)
            writer = cv2.VideoWriter(
                str(source),
                cv2.VideoWriter_fourcc(*"mp4v"),
                10.0,
                (220, 120),
            )
            for _ in range(3):
                writer.write(frame)
            writer.release()
            detector = lambda *_args, **_kwargs: [
                (103, 60, 12),
                (117, 60, 12),
            ]

            blur_report = blur_all_faces(
                object(),
                source,
                private,
                "cpu",
                yunet=object(),
                candidate_detector=detector,
            )
            verify_report = verify_blurred_against_source(
                object(),
                source,
                private,
                "cpu",
                yunet=object(),
                candidate_detector=detector,
            )

            self.assertIsNone(blur_report.failure_reason)
            self.assertTrue(verify_report.privacy_verified)

    def test_minimum_radius_core_survives_hd_video_encoding(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source.mp4"
            private = root / "private.mp4"
            rng = np.random.default_rng(19)
            frame = rng.integers(0, 256, size=(720, 1686, 3), dtype=np.uint8)
            writer = cv2.VideoWriter(
                str(source),
                cv2.VideoWriter_fourcc(*"mp4v"),
                30.0,
                (1686, 720),
            )
            writer.write(frame)
            writer.write(np.roll(frame, 2, axis=1))
            writer.release()
            detector = lambda *_args, **_kwargs: [(636, 303, 12)]

            blur_all_faces(
                object(),
                source,
                private,
                "cpu",
                yunet=object(),
                candidate_detector=detector,
            )
            report = verify_blurred_against_source(
                object(),
                source,
                private,
                "cpu",
                yunet=object(),
                candidate_detector=detector,
            )

            self.assertTrue(report.privacy_verified)

    def test_verification_requires_zero_candidates_and_complete_decode(self):
        frame = self.checkerboard()
        clean = verify_blurred_clip(
            object(),
            Path("clean.mp4"),
            "cpu",
            capture_factory=lambda _path: FakeCapture([frame]),
            yunet=object(),
            candidate_detector=lambda *_args, **_kwargs: [],
        )
        residual = verify_blurred_clip(
            object(),
            Path("residual.mp4"),
            "cpu",
            capture_factory=lambda _path: FakeCapture([frame]),
            yunet=object(),
            candidate_detector=lambda *_args, **_kwargs: [(50, 50, 20)],
        )
        truncated = verify_blurred_clip(
            object(),
            Path("truncated.mp4"),
            "cpu",
            capture_factory=lambda _path: FakeCapture([frame], frame_count=2),
            yunet=object(),
            candidate_detector=lambda *_args, **_kwargs: [],
        )

        self.assertTrue(clean.privacy_verified)
        self.assertEqual(clean.second_pass_candidates, 0)
        self.assertFalse(residual.privacy_verified)
        self.assertEqual(residual.second_pass_candidates, 1)
        self.assertFalse(truncated.privacy_verified)
        self.assertIn("dekod", truncated.failure_reason)

    def test_verification_accepts_detected_head_when_region_is_strongly_blurred(self):
        frame = cv2.GaussianBlur(self.checkerboard(), (49, 49), 0)

        report = verify_blurred_clip(
            object(),
            Path("blurred.mp4"),
            "cpu",
            capture_factory=lambda _path: FakeCapture([frame]),
            yunet=object(),
            candidate_detector=lambda *_args, **_kwargs: [(50, 50, 20)],
        )

        self.assertTrue(report.privacy_verified)
        self.assertEqual(report.second_pass_candidates, 0)

    def test_reference_verification_checks_source_candidates_not_blur_artifacts(self):
        source = self.checkerboard()
        private = source.copy()
        _blur_ellipse(private, 50, 50, 20)

        clean = verify_blurred_against_source(
            object(),
            Path("source.mp4"),
            Path("private.mp4"),
            "cpu",
            source_capture_factory=lambda _path: FakeCapture([source]),
            private_capture_factory=lambda _path: FakeCapture([private]),
            yunet=object(),
            candidate_detector=lambda *_args, **_kwargs: [(50, 50, 20)],
        )
        visible = verify_blurred_against_source(
            object(),
            Path("source.mp4"),
            Path("private.mp4"),
            "cpu",
            source_capture_factory=lambda _path: FakeCapture([source]),
            private_capture_factory=lambda _path: FakeCapture([source]),
            yunet=object(),
            candidate_detector=lambda *_args, **_kwargs: [(50, 50, 20)],
        )

        self.assertTrue(clean.privacy_verified)
        self.assertEqual(clean.second_pass_candidates, 0)
        self.assertFalse(visible.privacy_verified)
        self.assertEqual(visible.second_pass_candidates, 1)

    def test_missing_detector_fails_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            report = verify_blurred_clip(
                None,
                Path(raw) / "clip.mp4",
                "cpu",
                capture_factory=lambda _path: FakeCapture([self.checkerboard()]),
                yunet=None,
            )

        self.assertFalse(report.privacy_verified)
        self.assertIn("detektor", report.failure_reason)

    @patch("pipeline.face_blur.subprocess.run")
    def test_audio_mux_outputs_browser_compatible_h264_without_trimming(self, mock_run):
        _mux_original_audio(
            Path("private-video.mp4"),
            Path("source-audio.mp4"),
            Path("muxed.mp4"),
        )

        command = mock_run.call_args.args[0]
        self.assertIn("libx264", command)
        self.assertIn("yuv420p", command)
        self.assertNotIn("copy", command)
        self.assertNotIn("-shortest", command)

    def test_private_publish_replaces_output_only_after_final_clean_verification(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "raw.mp4"
            output = root / "final.mp4"
            source.write_bytes(b"raw")
            output.write_bytes(b"old-private")

            blur_calls = []
            repair_calls = []

            def blur(_model, _source, destination, _device, **kwargs):
                blur_calls.append(kwargs)
                Path(destination).write_bytes(b"blurred")
                return BlurReport(total_frames=3, first_pass_candidates=2)

            def repair(
                _model,
                _reference,
                _current,
                destination,
                _device,
                **kwargs,
            ):
                repair_calls.append(kwargs)
                Path(destination).write_bytes(b"blurred")
                return BlurReport(total_frames=3, first_pass_candidates=1)

            def mux(video, _audio, destination):
                Path(destination).write_bytes(Path(video).read_bytes() + b"+audio")
                return Path(destination)

            verify_calls = []

            def verify(_model, _reference, _private, _device, **kwargs):
                verify_calls.append(kwargs)
                reports = [
                    BlurReport(total_frames=3, second_pass_candidates=1),
                    BlurReport(
                        total_frames=3,
                        second_pass_candidates=0,
                        privacy_verified=True,
                    ),
                    BlurReport(total_frames=3, privacy_verified=True),
                ]
                return reports[len(verify_calls) - 1]

            report = privatize_media(
                object(),
                source,
                output,
                "cpu",
                blur_fn=blur,
                repair_fn=repair,
                verify_fn=verify,
                audio_muxer=mux,
            )

            self.assertTrue(report.privacy_verified)
            self.assertEqual(report.second_pass_candidates, 1)
            self.assertEqual(output.read_bytes(), b"blurred+audio")
            self.assertEqual(
                [call["score_threshold"] for call in blur_calls],
                [0.30],
            )
            self.assertFalse(blur_calls[0].get("only_unobscured", False))
            self.assertEqual(
                [call["score_threshold"] for call in repair_calls],
                [0.30],
            )
            self.assertEqual(
                [call["score_threshold"] for call in verify_calls],
                [0.30, 0.30, 0.30],
            )
            self.assertFalse(any(path.name.startswith(".final.") for path in root.iterdir()))

    def test_private_publish_failure_keeps_previous_verified_output(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "raw.mp4"
            output = root / "final.mp4"
            source.write_bytes(b"raw")
            output.write_bytes(b"old-private")

            def blur(_model, _source, destination, _device, **_kwargs):
                Path(destination).write_bytes(b"blurred")
                return BlurReport(total_frames=3, first_pass_candidates=2)

            def mux(video, _audio, destination):
                Path(destination).write_bytes(Path(video).read_bytes())
                return Path(destination)

            verify = Mock(
                side_effect=[
                    BlurReport(total_frames=3, second_pass_candidates=0),
                    BlurReport(total_frames=3, second_pass_candidates=1),
                ]
            )

            with self.assertRaises(PrivacyVerificationError):
                privatize_media(
                    object(),
                    source,
                    output,
                    "cpu",
                    blur_fn=blur,
                    verify_fn=verify,
                    audio_muxer=mux,
                )

            self.assertEqual(output.read_bytes(), b"old-private")
            self.assertFalse(any(path.name.startswith(".final.") for path in root.iterdir()))

    def test_default_private_pipeline_detects_each_source_frame_once(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "raw.mp4"
            output = root / "final.mp4"
            writer = cv2.VideoWriter(
                str(source),
                cv2.VideoWriter_fourcc(*"mp4v"),
                10.0,
                (200, 100),
            )
            writer.write(self.checkerboard())
            writer.write(np.flipud(self.checkerboard()))
            writer.release()
            model = Mock()
            result = Mock()
            result.keypoints = None
            model.predict.return_value = [result, result]
            yunet = Mock()
            yunet.detect.return_value = (None, None)

            def copy_video(video, _audio, destination):
                Path(destination).write_bytes(Path(video).read_bytes())
                return Path(destination)

            with patch("pipeline.face_blur._load_yunet", return_value=yunet):
                report = privatize_media(
                    model,
                    source,
                    output,
                    "cpu",
                    audio_muxer=copy_video,
                )

            self.assertTrue(report.privacy_verified)
            self.assertEqual(report.total_frames, 2)
            self.assertEqual(model.predict.call_count, 1)
            batch = model.predict.call_args.args[0]
            self.assertEqual(len(batch), 2)


if __name__ == "__main__":
    unittest.main()
