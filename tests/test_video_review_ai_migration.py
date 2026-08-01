import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from pipeline.video_review_migration import (
    _prepare_ai_review_payload,
    migrate_ai_session,
)


class VideoReviewAiMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "corrected-session"
        self.source.mkdir()
        self.sony = self.root / "sony.mp4"
        self.iphone = self.root / "iphone.mov"
        self.sony.write_bytes(b"sony-source")
        self.iphone.write_bytes(b"iphone-source")
        review = {
            "sources": {
                "sony": {
                    "path": str(self.sony),
                    "sha256": self.digest(self.sony),
                },
                "iphone": {
                    "path": str(self.iphone),
                    "sha256": self.digest(self.iphone),
                },
            }
        }
        (self.source / "review.json").write_text(
            json.dumps(review), encoding="utf-8"
        )

    @staticmethod
    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def migrate(self, output: Path, *, replace_derived: bool = False):
        return migrate_ai_session(
            self.source,
            output,
            model_path=self.root / "model.pt",
            device="cpu",
            replace_derived=replace_derived,
        )

    def valid_review(self):
        frames = []
        for index in range(49):
            timestamp = 128.0 + index / 6.0
            frames.append(
                {
                    "frame_index": index,
                    "timestamp_s": timestamp,
                    "hip_midpoint": [100.0 + index, 200.0],
                    "shoulder_midpoint": [100.0 + index, 180.0],
                    "vidljivo": True,
                    "interpolirano": False,
                    "brzina_ulaska_norm": index / 12.0,
                    "rotacija_trupa_2d_dps": index * 20.0,
                    "promena_visine_kukova_norm": index / 100.0,
                    "sirina_stava_norm": 1.0,
                    "proxy_ubrzanja_norm_s2": None if index == 0 else 2.0,
                    "intenzitet_pokreta_0_100": min(100.0, index * 2.0),
                }
            )
        return {
            "version": 2,
            "session_id": "corrected-session",
            "sony_video": str(self.sony),
            "iphone_video": str(self.iphone),
            "sources": {
                "sony": {
                    "path": str(self.sony),
                    "sha256": self.digest(self.sony),
                    "fps": 30.0,
                },
                "iphone": {
                    "path": str(self.iphone),
                    "sha256": self.digest(self.iphone),
                    "fps": 30.0,
                },
            },
            "sony_duration_s": 222.0,
            "iphone_duration_s": 280.0,
            "source_fps": 30.0,
            "effective_analysis_fps": 6.0,
            "pose_analysis": {
                "selected_track_id": 1,
                "effective_analysis_fps": 6.0,
            },
            "anchors": [
                {
                    "name": "start",
                    "sony_s": 128.0,
                    "iphone_s": 131.0,
                    "user_confirmed": True,
                    "triple_tap_count": 3,
                },
                {
                    "name": "kontrola",
                    "sony_s": 129.0,
                    "iphone_s": 132.0,
                    "user_confirmed": True,
                    "triple_tap_count": 3,
                },
            ],
            "time_map": {"slope": 1.0, "intercept": -3.0},
            "injury_cutoff_s": 135.0,
            "frame_metrics": frames,
            "events": [],
            "event_metrics": [],
        }

    def test_rejects_same_source_and_output_directory(self):
        with self.assertRaisesRegex(ValueError, "različiti"):
            self.migrate(self.source)

    def test_rejects_nonempty_target_without_replace_derived(self):
        target = self.root / "trainer-ai-session"
        target.mkdir()
        (target / "keep.txt").write_text("do not overwrite", encoding="utf-8")

        with self.assertRaisesRegex(FileExistsError, "replace-derived"):
            self.migrate(target)

        self.assertEqual((target / "keep.txt").read_text(encoding="utf-8"), "do not overwrite")

    def test_rejects_nonversioned_directory_even_with_replace_derived(self):
        target = self.root / "trainer-ai-session"
        target.mkdir()
        marker = target / "keep.txt"
        marker.write_text("legacy-output", encoding="utf-8")

        with self.assertRaisesRegex(FileExistsError, "nije atomski versioniran"):
            self.migrate(target, replace_derived=True)

        self.assertEqual(marker.read_text(encoding="utf-8"), "legacy-output")

    def test_rejects_source_hash_mismatch_before_creating_target(self):
        self.sony.write_bytes(b"changed-source")
        target = self.root / "trainer-ai-session"

        with self.assertRaisesRegex(ValueError, "hash"):
            self.migrate(target)

        self.assertFalse(target.exists())

    def test_prepare_ai_payload_normalizes_three_events_and_hides_ai(self):
        prepared = _prepare_ai_review_payload(
            self.valid_review(), "trainer-ai-session"
        )

        self.assertEqual(prepared["version"], 3)
        self.assertEqual(prepared["session_id"], "trainer-ai-session")
        self.assertFalse(prepared["session_ready"])
        self.assertEqual(prepared["derived_media_manifest"], [])
        self.assertEqual(prepared["side_by_side_start_s"], 128.0)
        self.assertEqual(prepared["side_by_side_end_s"], 136.0)
        self.assertEqual(
            [event["event_id"] for event in prepared["events"]],
            ["e-001", "e-coach-001", "povreda"],
        )
        normal = prepared["events"][:2]
        self.assertEqual(
            [event["potvrdena_tehnika"] for event in normal],
            ["Tai-otoshi", "Morote-seoi-nage"],
        )
        self.assertEqual(
            [(event["sony_start_s"], event["sony_end_s"]) for event in normal],
            [(128.5, 132.0), (132.8, 135.0)],
        )
        for event in normal:
            self.assertEqual(event["status"], "trener")
            self.assertIsNone(event["ocena"])
            self.assertEqual(event["trener_procene"], [])
            self.assertEqual(len(event["ai_procene"]), 1)
            self.assertIsNone(event["ai_procene"][0]["ai_otkriven_u"])
        injury = prepared["events"][2]
        self.assertEqual((injury["sony_start_s"], injury["sony_end_s"]), (135.0, 136.0))
        self.assertNotIn("ai_procene", injury)

    @staticmethod
    def fake_media_regenerator(review, stage, _privacy_processor):
        rows = []

        def publish(relative_path, media_type):
            path = stage / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"verified-private")
            rows.append(
                {
                    "relative_path": relative_path,
                    "media_type": media_type,
                    "total_frames": 1,
                    "first_pass_candidates": 1,
                    "second_pass_candidates": 0,
                    "privacy_verified": True,
                    "failure_reason": None,
                }
            )

        for index, _anchor in enumerate(review["anchors"], start=1):
            publish(f"previews/anchor_{index:02d}_sony.mp4", "anchor_preview")
            publish(f"previews/anchor_{index:02d}_iphone.mp4", "anchor_preview")
        for event in review["events"]:
            for camera in ("sony", "iphone"):
                publish(
                    f"events/{event['event_id']}/{camera}.mp4",
                    "event_clip",
                )
        publish("session_side_by_side.mp4", "side_by_side")
        return rows

    def test_success_publishes_ready_session_after_complete_verified_manifest(self):
        (self.source / "review.json").write_text(
            json.dumps(self.valid_review()), encoding="utf-8"
        )
        target = self.root / "trainer-ai-session"

        review_path = migrate_ai_session(
            self.source,
            target,
            model_path=self.root / "model.pt",
            device="cpu",
            privacy_processor=object(),
            media_regenerator=self.fake_media_regenerator,
        )

        review = json.loads(review_path.read_text(encoding="utf-8"))
        self.assertTrue(target.is_symlink())
        self.assertTrue(review["session_ready"])
        self.assertEqual(len(review["derived_media_manifest"]), 11)
        self.assertTrue(
            all(row["privacy_verified"] for row in review["derived_media_manifest"])
        )
        self.assertEqual(
            {row["media_type"] for row in review["derived_media_manifest"]},
            {"event_clip", "anchor_preview", "side_by_side"},
        )
        self.assertFalse((target / self.sony.name).exists())
        self.assertFalse((target / self.iphone.name).exists())
        self.assertTrue((target / "analysis" / "event_metrics.json").is_file())
        self.assertTrue((target / "izvestaj.csv").is_file())

    def test_media_failure_does_not_publish_partial_target(self):
        (self.source / "review.json").write_text(
            json.dumps(self.valid_review()), encoding="utf-8"
        )
        target = self.root / "trainer-ai-session"

        def fail_media(_review, _stage, _privacy_processor):
            raise ValueError("privacy verification failed")

        with self.assertRaisesRegex(ValueError, "privacy verification failed"):
            migrate_ai_session(
                self.source,
                target,
                model_path=self.root / "model.pt",
                device="cpu",
                privacy_processor=object(),
                media_regenerator=fail_media,
            )

        self.assertFalse(target.exists())

    def test_incomplete_or_unverified_manifest_is_not_published(self):
        (self.source / "review.json").write_text(
            json.dumps(self.valid_review()), encoding="utf-8"
        )
        target = self.root / "trainer-ai-session"

        def incomplete_media(review, stage, privacy_processor):
            rows = self.fake_media_regenerator(review, stage, privacy_processor)
            rows[-1]["privacy_verified"] = False
            return rows

        with self.assertRaisesRegex(ValueError, "manifest nije kompletan"):
            migrate_ai_session(
                self.source,
                target,
                model_path=self.root / "model.pt",
                device="cpu",
                privacy_processor=object(),
                media_regenerator=incomplete_media,
            )

        self.assertFalse(target.exists())

    def test_failed_replacement_preserves_existing_derived_session(self):
        (self.source / "review.json").write_text(
            json.dumps(self.valid_review()), encoding="utf-8"
        )
        target = self.root / "trainer-ai-session"
        migrate_ai_session(
            self.source,
            target,
            model_path=self.root / "model.pt",
            device="cpu",
            privacy_processor=object(),
            media_regenerator=self.fake_media_regenerator,
        )
        marker = target / "review.json"
        previous = marker.read_bytes()
        previous_generation = target.resolve()

        def fail_media(_review, _stage, _privacy_processor):
            raise ValueError("privacy verification failed")

        with self.assertRaisesRegex(ValueError, "privacy verification failed"):
            migrate_ai_session(
                self.source,
                target,
                model_path=self.root / "model.pt",
                device="cpu",
                replace_derived=True,
                privacy_processor=object(),
                media_regenerator=fail_media,
            )

        self.assertEqual(marker.read_bytes(), previous)
        self.assertEqual(target.resolve(), previous_generation)

    def test_source_mutated_during_regeneration_is_not_published(self):
        (self.source / "review.json").write_text(
            json.dumps(self.valid_review()), encoding="utf-8"
        )
        target = self.root / "trainer-ai-session"

        def mutate_source(review, stage, privacy_processor):
            rows = self.fake_media_regenerator(review, stage, privacy_processor)
            self.sony.write_bytes(b"mutated-during-regeneration")
            return rows

        with self.assertRaisesRegex(ValueError, "promenjen tokom migracije"):
            migrate_ai_session(
                self.source,
                target,
                model_path=self.root / "model.pt",
                device="cpu",
                privacy_processor=object(),
                media_regenerator=mutate_source,
            )

        self.assertFalse(target.exists())

    def test_replacement_atomically_switches_symlink_and_keeps_old_generation(self):
        (self.source / "review.json").write_text(
            json.dumps(self.valid_review()), encoding="utf-8"
        )
        target = self.root / "trainer-ai-session"
        self.migrate_with_fake_media(target)
        previous_generation = target.resolve()
        previous_review = (previous_generation / "review.json").read_bytes()

        self.migrate_with_fake_media(target, replace_derived=True)

        self.assertTrue(target.is_symlink())
        self.assertNotEqual(target.resolve(), previous_generation)
        self.assertTrue(previous_generation.is_dir())
        self.assertEqual(
            (previous_generation / "review.json").read_bytes(),
            previous_review,
        )

    def migrate_with_fake_media(
        self, target: Path, *, replace_derived: bool = False
    ) -> Path:
        return migrate_ai_session(
            self.source,
            target,
            model_path=self.root / "model.pt",
            device="cpu",
            replace_derived=replace_derived,
            privacy_processor=object(),
            media_regenerator=self.fake_media_regenerator,
        )


if __name__ == "__main__":
    unittest.main()
