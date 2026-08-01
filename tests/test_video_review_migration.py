import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from pipeline.video_review_contract import validate_review_payload
from pipeline.video_review_migration import migrate_review_payload, migrate_session


class VideoReviewMigrationTests(unittest.TestCase):
    def fixture(self, root: Path):
        sony = root / "sony.mp4"
        iphone = root / "iphone.mov"
        sony.write_bytes(b"sony source remains immutable")
        iphone.write_bytes(b"iphone source remains immutable")
        old_frames = []
        for index, energy in enumerate((0.1, 1.0, 0.1, 0.1, 0.1)):
            old_frames.append(
                {
                    "frame_index": index,
                    "timestamp_s": 10.0 + index,
                    "brzina_ulaska_norm_s": energy,
                    "rotation_2d_dps": energy * 100.0,
                    "hip_level_norm": -0.2 + index * 0.1,
                    "stance_width_norm": 0.5 + index * 0.05,
                    "hip_midpoint": [100.0 + index, 200.0],
                    "shoulder_midpoint": [100.0 + index, 180.0],
                    "torso_length": 20.0,
                    "shoulder_angle_deg": 0.0,
                    "vidljivo": True,
                    "interpolirano": False,
                }
            )
        normal = {
            "event_id": "e-001",
            "sony_start_s": 10.0,
            "sony_end_s": 14.0,
            "iphone_start_s": 30.0,
            "iphone_end_s": 34.0,
            "predlog_tehnike": None,
            "potvrdena_tehnika": "O-soto-gari",
            "glasovna_fraza": None,
            "pouzdanost_glasa": 0.0,
            "ocena": 4,
            "napomena": "Migracija čuva ovu napomenu.",
            "iskljuceno_iz_statistike": False,
            "status": "predlog",
        }
        injury = {
            "event_id": "povreda",
            "sony_start_s": 20.0,
            "sony_end_s": 21.0,
            "iphone_start_s": 40.0,
            "iphone_end_s": 41.0,
            "predlog_tehnike": None,
            "potvrdena_tehnika": None,
            "glasovna_fraza": None,
            "pouzdanost_glasa": 0.0,
            "prijavljen_povredni_dogadjaj": True,
            "iskljuceno_iz_statistike": True,
            "status": "povreda",
        }
        return {
            "version": 1,
            "session_id": "migration",
            "sony_video": str(sony),
            "iphone_video": str(iphone),
            "sources": {
                "sony": {"path": str(sony), "sha256": hashlib.sha256(sony.read_bytes()).hexdigest()},
                "iphone": {"path": str(iphone), "sha256": hashlib.sha256(iphone.read_bytes()).hexdigest()},
            },
            "sony_duration_s": 30.0,
            "iphone_duration_s": 60.0,
            "sony_fps": 30.0,
            "iphone_fps": 30.0,
            "effective_analysis_fps": 3.0,
            "anchors": [
                {
                    "name": "pocetak",
                    "sony_s": 10.0,
                    "iphone_s": 30.0,
                    "user_confirmed": True,
                    "triple_tap_count": 3,
                },
                {
                    "name": "kontrola",
                    "sony_s": 19.0,
                    "iphone_s": 39.0,
                    "user_confirmed": True,
                    "triple_tap_count": 3,
                },
            ],
            "time_map": {"slope": 1.0, "intercept": -20.0},
            "injury_cutoff_s": 20.0,
            "frame_metrics": old_frames,
            "events": [normal, injury],
            "event_metrics": [dict(normal), dict(injury)],
        }

    def test_payload_migration_creates_five_canonical_finite_series(self):
        with tempfile.TemporaryDirectory() as raw:
            payload = self.fixture(Path(raw))

            migrated = migrate_review_payload(payload)

        frame = migrated["frame_metrics"][0]
        for legacy_key in (
            "brzina_ulaska_norm_s",
            "rotation_2d_dps",
            "hip_level_norm",
            "stance_width_norm",
        ):
            self.assertNotIn(legacy_key, frame)
        series_keys = [
            item["key"] for item in migrated["metric_schema"]["frame_series"]
        ]
        self.assertEqual(
            series_keys,
            [
                "brzina_ulaska_norm",
                "rotacija_trupa_2d_dps",
                "promena_visine_kukova_norm",
                "sirina_stava_norm",
                "intenzitet_pokreta_0_100",
            ],
        )
        for key in series_keys:
            values = [sample[key] for sample in migrated["frame_metrics"]]
            self.assertTrue(all(isinstance(value, (int, float)) for value in values))
        self.assertTrue(all(0.0 <= sample["intenzitet_pokreta_0_100"] <= 100.0 for sample in migrated["frame_metrics"]))
        self.assertEqual(
            migrated["frame_metrics"][0]["intenzitet_pokreta_0_100"],
            11.967593,
        )
        self.assertEqual(
            migrated["frame_metrics"][1]["proxy_ubrzanja_norm_s2"],
            0.9,
        )
        event = migrated["events"][0]
        self.assertEqual(event["potvrdena_tehnika"], "O-soto-gari")
        self.assertEqual(event["status"], "trener")
        self.assertIsNone(event["ocena"])
        self.assertEqual(event["legacy_annotations"][0]["ocena"], 4)
        self.assertTrue(event["legacy_annotations"][0]["nije_pre_ai"])
        self.assertEqual(event["trener_procene"], [])
        self.assertIsNone(event["ai_procene"][0]["ai_otkriven_u"])
        self.assertEqual(event["napomena"], "Migracija čuva ovu napomenu.")
        self.assertEqual(migrated["event_metrics"], migrated["events"])
        self.assertEqual(migrated["version"], 3)
        self.assertEqual(migrated, migrate_review_payload(migrated))
        recovery = migrated["metric_schema"]["recovery_to_stable"]
        self.assertEqual(recovery["motion_energy_threshold"], 0.2)
        self.assertEqual(recovery["consecutive_samples"], 3)
        self.assertEqual(recovery["when_not_observable"], None)

    def test_migration_ignores_untrusted_legacy_acceleration(self):
        with tempfile.TemporaryDirectory() as raw:
            payload = self.fixture(Path(raw))
            payload["metric_schema"] = {"pose_metrics_id": "video-pose-metrics-v1"}
            payload["frame_metrics"][1]["proxy_ubrzanja_norm_s2"] = 99.0

            migrated = migrate_review_payload(payload)

        self.assertEqual(
            migrated["frame_metrics"][1]["proxy_ubrzanja_norm_s2"],
            0.9,
        )

    def test_fractional_migration_is_exactly_idempotent(self):
        with tempfile.TemporaryDirectory() as raw:
            payload = self.fixture(Path(raw))
            payload["frame_metrics"][0]["timestamp_s"] = 10.00049
            payload["frame_metrics"][1]["timestamp_s"] = 11.00051
            payload["frame_metrics"][0]["brzina_ulaska_norm_s"] = 0.123456789
            payload["frame_metrics"][1]["brzina_ulaska_norm_s"] = 0.987654321

            migrated = migrate_review_payload(payload)
            repeated = migrate_review_payload(migrated)

        self.assertEqual(migrated, repeated)

    def test_validation_rejects_stale_analysis_fingerprint(self):
        with tempfile.TemporaryDirectory() as raw:
            migrated = migrate_review_payload(self.fixture(Path(raw)))
        migrated["sources"]["sony"]["sha256"] = "c" * 64

        with self.assertRaisesRegex(ValueError, "fingerprint"):
            validate_review_payload(migrated)

    def test_validation_accepts_bounded_event_iphone_sync_offset(self):
        with tempfile.TemporaryDirectory() as raw:
            migrated = migrate_review_payload(self.fixture(Path(raw)))
        migrated["version"] = 2
        event = migrated["events"][0]
        event["iphone_sync_offset_s"] = 0.8
        migrated["event_metrics"] = copy.deepcopy(migrated["events"])

        validate_review_payload(migrated)

        event["iphone_sync_offset_s"] = 27.0
        migrated["event_metrics"] = copy.deepcopy(migrated["events"])
        with self.assertRaisesRegex(ValueError, "corrected iPhone"):
            validate_review_payload(migrated)

    def test_session_migration_writes_analysis_and_reports_without_touching_sources(self):
        with tempfile.TemporaryDirectory() as raw:
            session = Path(raw)
            payload = self.fixture(session)
            review_path = session / "review.json"
            review_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            before_hashes = {
                camera: hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest()
                for camera, record in payload["sources"].items()
            }

            migrated_path = migrate_session(session)

            migrated = json.loads(migrated_path.read_text(encoding="utf-8"))
            after_hashes = {
                camera: hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest()
                for camera, record in payload["sources"].items()
            }
            self.assertEqual(before_hashes, after_hashes)
            self.assertEqual(migrated["events"][0]["potvrdena_tehnika"], "O-soto-gari")
            self.assertIsNone(migrated["events"][0]["ocena"])
            self.assertTrue(migrated["sync_locked"])
            self.assertTrue((session / "analysis" / "frame_metrics.json").is_file())
            self.assertTrue((session / "analysis" / "event_metrics.json").is_file())
            self.assertIn("O-soto-gari", (session / "izvestaj.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
