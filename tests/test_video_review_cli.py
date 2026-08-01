import unittest
from pathlib import Path
from unittest.mock import patch

from tools.video_review import build_parser, main


class VideoReviewCliTests(unittest.TestCase):
    def test_import_parser_accepts_sampling_model_and_device_options(self):
        args = build_parser().parse_args(
            [
                "import",
                "--sony", "sony.mp4",
                "--iphone", "iphone.mov",
                "--session-dir", "session",
                "--anchors-json", "anchors.json",
                "--injury-cutoff-sony-s", "126.0",
                "--blue-seed-sony", "1897,887,2081,1486",
                "--analysis-fps", "3.0",
                "--model-path", "/tmp/yolo11x-pose.pt",
                "--device", "mps",
            ]
        )

        self.assertEqual(args.analysis_fps, 3.0)
        self.assertEqual(args.model_path, Path("/tmp/yolo11x-pose.pt"))
        self.assertEqual(args.device, "mps")
        self.assertEqual(args.event_threshold, 0.5)
        self.assertEqual(args.blue_seed_sony, (1897.0, 887.0, 2081.0, 1486.0))

    @patch("tools.video_review.import_session")
    def test_import_command_forwards_sampling_options(self, mock_import):
        mock_import.return_value = Path("session/review.json")
        anchors = Path("anchors.json")
        with patch.object(Path, "open") as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = (
                '{"anchors": []}'
            )
            with patch("json.load", return_value={"anchors": []}):
                status = main(
                    [
                        "import",
                        "--sony", "sony.mp4",
                        "--iphone", "iphone.mov",
                        "--session-dir", "session",
                        "--anchors-json", str(anchors),
                        "--injury-cutoff-sony-s", "126.0",
                        "--blue-seed-sony", "1897,887,2081,1486",
                        "--analysis-fps", "3.0",
                        "--model-path", "/tmp/yolo11x-pose.pt",
                        "--device", "mps",
                        "--event-threshold", "0.4",
                    ]
                )

        self.assertEqual(status, 0)
        self.assertEqual(mock_import.call_args.kwargs["analysis_fps"], 3.0)
        self.assertEqual(mock_import.call_args.kwargs["model_path"], Path("/tmp/yolo11x-pose.pt"))
        self.assertEqual(mock_import.call_args.kwargs["device"], "mps")
        self.assertEqual(mock_import.call_args.kwargs["event_threshold"], 0.4)


if __name__ == "__main__":
    unittest.main()
