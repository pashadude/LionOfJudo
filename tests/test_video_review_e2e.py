import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from coach_app.server import create_server
from pipeline.face_blur import BlurReport
from pipeline.video_review_contract import AnchorPair
from pipeline.video_review_import import import_session
from pipeline.voice_labels import TranscriptWord


class VideoReviewEndToEndTests(unittest.TestCase):
    def test_import_annotation_and_http_exports_form_one_review_contract(self):
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            sony = root / "sony.mp4"
            iphone = root / "iphone.mov"
            sony.write_bytes(b"synthetic sony source")
            iphone.write_bytes(b"synthetic iphone source")
            transcript = root / "transcript.json"
            transcript.write_text("{}", encoding="utf-8")

            def duration_for(path):
                path = Path(path).resolve()
                if path in {sony.resolve(), iphone.resolve()}:
                    return 60.0
                parent = path.parent.name
                if parent.startswith(".raw-private-"):
                    parent = path.parent.parent.name
                if parent == "e-normal":
                    return 3.0
                if parent == "povreda":
                    return 2.0
                if path.name == "session_side_by_side.mp4":
                    return 20.0
                return 4.0

            def touch_media(*args, **kwargs):
                output = Path(kwargs.get("output") or args[-1])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(b"synthetic media export")
                return output

            def privatize(raw_path, output_path):
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(Path(raw_path).read_bytes())
                return BlurReport(
                    total_frames=1,
                    first_pass_candidates=1,
                    second_pass_candidates=0,
                    privacy_verified=True,
                )

            anchors = [
                AnchorPair("pocetak", 10.0, 30.0, True, 3),
                AnchorPair("kontrola", 19.0, 39.0, True, 3),
            ]
            pose = {
                "fps": 29.97,
                "selected_track_id": 7,
                "athlete_seen": True,
                "frame_metrics": [{"timestamp_s": 11.0, "brzina_ulaska_norm": 0.8}],
                "events": [
                    {
                        "event_id": "e-normal",
                        "sony_start_s": 11.0,
                        "sony_end_s": 14.0,
                        "metrics": {"brzina_ulaska_norm": 0.8},
                    },
                    {
                        "event_id": "povreda",
                        "sony_start_s": 18.0,
                        "sony_end_s": 21.0,
                        "status": "povreda",
                        "prijavljen_povredni_dogadjaj": True,
                    },
                ],
            }
            whisper_words = [TranscriptWord("o soto gari", 11.5, 12.2)]

            # These are the three permitted seams: media subprocesses, YOLO,
            # and Whisper. Mapping, contract, persistence, HTTP, and reports stay real.
            with patch("pipeline.video_review_import.probe_duration", side_effect=duration_for), \
                patch("pipeline.video_review_import.probe_fps", side_effect=lambda path: 29.97 if Path(path) == sony else 59.94), \
                patch("pipeline.video_review_import.cut_clip", side_effect=touch_media), \
                patch("pipeline.video_review_import.make_side_by_side", side_effect=touch_media), \
                patch("pipeline.video_review_import.run_pose_analysis", return_value=pose) as mock_pose, \
                patch("pipeline.video_review_import.load_whisper_json", return_value=whisper_words):
                review_path = import_session(
                    sony=sony,
                    iphone=iphone,
                    output_dir=root / "session",
                    anchors=anchors,
                    injury_cutoff_s=20.0,
                    blue_seed=(100.0, 200.0, 180.0, 360.0),
                    transcript_path=transcript,
                    privacy_processor=privatize,
                )

            mock_pose.assert_called_once_with(
                sony.resolve(),
                10.0,
                20.0,
                [100.0, 200.0, 180.0, 360.0],
                analysis_fps=None,
                model_path="yolo11x-pose.pt",
                device="mps",
                event_threshold=0.5,
            )

            review = json.loads(review_path.read_text(encoding="utf-8"))
            self.assertEqual(review["time_map"], {"intercept": -20.0, "slope": 1.0})
            self.assertEqual(review["events"][0]["iphone_start_s"], 31.0)
            injury = next(event for event in review["events"] if event["event_id"] == "povreda")
            self.assertTrue(injury["iskljuceno_iz_statistike"])
            self.assertEqual(injury["status"], "povreda")
            self.assertEqual(review["events"][0]["predlog_tehnike"], "O-soto-gari")
            self.assertEqual(review["events"][0]["glasovna_fraza"], "o soto gari")
            self.assertGreater(review["events"][0]["pouzdanost_glasa"], 0.0)
            self.assertEqual(review["events"][0]["glasovna_fraza_pocetak_s"], 11.5)
            self.assertEqual(review["events"][0]["glasovna_fraza_kraj_s"], 12.2)
            self.assertEqual(
                review["transkript"],
                [{"tekst": "o soto gari", "pocetak_s": 11.5, "kraj_s": 12.2}],
            )
            self.assertIsNone(injury["predlog_tehnike"])
            self.assertIsNone(injury["glasovna_fraza"])
            self.assertNotIn("glasovna_fraza_pocetak_s", injury)
            self.assertTrue((root / "session" / "media" / "session_side_by_side.mp4").exists())

            server = create_server(root / "session", port=0)
            thread = server.start_in_thread()
            try:
                with urlopen(server.base_url + "/api/session") as response:
                    self.assertEqual(response.status, 200)
                    session = json.loads(response.read().decode("utf-8"))
                self.assertEqual(session["session_id"], "session")

                sync_body = json.dumps(
                    {"anchors": [anchor.to_dict() for anchor in anchors], "injury_cutoff_s": 20.0}
                ).encode("utf-8")
                with self.assertRaises(HTTPError) as raised:
                    urlopen(Request(server.base_url + "/api/session/sync", data=sync_body, method="POST", headers={"Content-Type": "application/json"}))
                self.assertEqual(raised.exception.code, 409)
                self.assertIn(
                    "novi uvoz",
                    json.loads(raised.exception.read().decode("utf-8"))["error"],
                )

                annotation_body = json.dumps(
                    {
                        "potvrdena_tehnika": "O-soto-gari",
                        "ocena": 4,
                        "napomena": "Dobar ulaz.",
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                with urlopen(Request(server.base_url + "/api/events/e-normal/annotation", data=annotation_body, method="PUT", headers={"Content-Type": "application/json"})) as response:
                    self.assertEqual(response.status, 200)
                    saved = json.loads(response.read().decode("utf-8"))
                self.assertEqual(saved["ocena"], 4)
            finally:
                server.shutdown()
                thread.join(timeout=2)

            final_review = json.loads(review_path.read_text(encoding="utf-8"))
            self.assertEqual(final_review["events"][0]["potvrdena_tehnika"], "O-soto-gari")
            self.assertEqual(final_review["events"][0]["ocena"], 4)
            with (root / "session" / "izvestaj.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["Potvrđena tehnika"], "O-soto-gari")
            self.assertEqual(rows[0]["Glasovna fraza"], "o soto gari")
            self.assertEqual(rows[0]["Početak glasovne fraze (s)"], "11.5")
            self.assertEqual(rows[0]["Kraj glasovne fraze (s)"], "12.2")
            self.assertGreater(float(rows[0]["Pouzdanost glasa"]), 0.0)
            self.assertEqual(rows[1]["Isključeno iz statistike"], "da")
            markdown = (root / "session" / "izvestaj.md").read_text(encoding="utf-8")
            self.assertIn("Dobar ulaz.", markdown)
            self.assertIn("o soto gari", markdown)
            self.assertIn("11.5", markdown)
            self.assertIn("12.2", markdown)
            self.assertIn("Normalni događaji u statistici: 1", markdown)


if __name__ == "__main__":
    unittest.main()
