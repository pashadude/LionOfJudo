import json
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from coach_app.review_bundle import GenerationStore
from coach_app.server import create_server
from coach_app.trainer_ai_service import TrainerAiService
from pipeline.trainer_ai_state import active_ai_evaluation
from pipeline.video_review_migration import migrate_review_payload
from pipeline.video_review_reports import write_reports
import tests.test_video_review_migration as migration_fixtures


FIXED_TIME = datetime(2026, 8, 1, 12, 0, 0, tzinfo=timezone.utc)


class CoachTrainerAiTests(unittest.TestCase):
    def setUp(self):
        self.raw = tempfile.TemporaryDirectory()
        self.root = Path(self.raw.name)
        legacy = migration_fixtures.VideoReviewMigrationTests().fixture(self.root)
        self.review = migrate_review_payload(legacy)
        self.review["derived_media_manifest"] = [
            {
                "relative_path": "session_side_by_side.mp4",
                "media_type": "side_by_side",
                "total_frames": 1,
                "first_pass_candidates": 1,
                "second_pass_candidates": 0,
                "privacy_verified": True,
                "failure_reason": None,
            }
        ]
        self.review["participants"] = {
            "trainer_name": "Marko",
            "wrestler_name": "Dusan",
            "updated_at": "2026-08-01T12:00:00+00:00",
        }
        self.review_path = self.root / "review.json"
        self.review_path.write_text(
            json.dumps(self.review, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_reports(self.review_path, self.review)
        (self.root / "media").mkdir(exist_ok=True)
        (self.root / "media" / "session_side_by_side.mp4").write_bytes(b"derived-media")
        self.now = FIXED_TIME
        self.service = TrainerAiService(
            self.root,
            clock=lambda: self.now,
            mutation_lock=threading.RLock(),
        )

    def tearDown(self):
        self.raw.cleanup()

    def start_server(self):
        server = create_server(self.root, port=0, clock=lambda: self.now)
        thread = server.start_in_thread()
        self.addCleanup(lambda: self._stop_server(server, thread))
        return server

    @staticmethod
    def _stop_server(server, thread):
        server.shutdown()
        thread.join(timeout=2)

    @staticmethod
    def read_json(url, *, method="GET", payload=None):
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            url,
            data=body,
            method=method,
            headers={"Content-Type": "application/json"} if body is not None else {},
        )
        with urlopen(request) as response:
            return response.status, json.loads(response.read().decode("utf-8"))

    @staticmethod
    def visible_assessment(**overrides):
        payload = {
            "status_vidljivosti": "dovoljno_vidljivo",
            "potvrdena_tehnika": "Tai-otoshi",
            "ocena": 4,
            "razlog": "Na 11.000 s kukovi kasne za rotacijom.",
            "citirani_sony_trenuci_s": [11.0],
        }
        payload.update(overrides)
        return payload

    def test_lock_creates_immutable_pre_ai_revision(self):
        result = self.service.lock_assessment(
            "e-001",
            {
                "status_vidljivosti": "dovoljno_vidljivo",
                "potvrdena_tehnika": "Tai-otoshi",
                "ocena": 4,
                "razlog": "Na 11.000 s kukovi kasne za rotacijom.",
                "citirani_sony_trenuci_s": [11.0],
            },
        )

        assessment = result["assessment"]
        self.assertEqual(assessment["faza"], "pre_ai")
        self.assertEqual(assessment["zakljucano_u"], "2026-08-01T12:00:00+00:00")
        self.assertIsNone(active_ai_evaluation(result["event"])["ai_otkriven_u"])
        self.assertEqual(result["event"]["trener_procene"], [assessment])
        public_event = self.service.public_event("e-001")
        self.assertNotIn("ai_procene", public_event)
        self.assertNotIn("imu_eksperimentalno", public_event)
        snapshot = self.service.store.resolve_current()
        self.assertIsNotNone(snapshot.generation_id)
        self.assertNotEqual(snapshot.root, self.root)
        legacy = json.loads(self.review_path.read_text(encoding="utf-8"))
        self.assertEqual(legacy["events"][0]["trener_procene"], [])
        self.assertEqual(
            (self.root / "media" / "session_side_by_side.mp4").stat().st_ino,
            (snapshot.root / "media" / "session_side_by_side.mp4").stat().st_ino,
        )

    def test_participants_are_required_and_snapshotted_per_assessment(self):
        review = self.service.load_review()
        review.pop("participants")
        self.service.activate_review(review)
        with self.assertRaisesRegex(ValueError, "ime trenera"):
            self.service.lock_assessment("e-001", self.visible_assessment())
        saved = self.service.save_participants({
            "trainer_name": "  Marko Markovic  ",
            "wrestler_name": " Dusan ",
        })
        locked = self.service.lock_assessment("e-001", self.visible_assessment())

        self.assertEqual(saved["participants"]["trainer_name"], "Marko Markovic")
        self.assertEqual(locked["assessment"]["wrestler_name"], "Dusan")

    def test_later_participant_edit_does_not_rewrite_locked_identity(self):
        self.service.save_participants({"trainer_name": "Marko", "wrestler_name": "Dusan"})
        first = self.service.lock_assessment("e-001", self.visible_assessment())["assessment"]
        self.service.save_participants({"trainer_name": "Jovan", "wrestler_name": "Dusan"})

        self.assertEqual(first["trainer_name"], "Marko")

    def test_store_rejects_mismatched_event_metrics_without_switching(self):
        with self.assertRaisesRegex(ValueError, "event_metrics"):
            self.service.store.stage_and_activate(
                self.review,
                [],
                "csv",
                "markdown",
            )
        self.assertIsNone(self.service.store.resolve_current().generation_id)

    def test_media_response_uses_same_current_generation_as_review(self):
        replacement = self.root / "replacement.mp4"
        replacement.write_bytes(b"new-generation-media")
        self.service.store.stage_and_activate(
            self.review,
            self.review["events"],
            (self.root / "izvestaj.csv").read_text(encoding="utf-8"),
            (self.root / "izvestaj.md").read_text(encoding="utf-8"),
            staged_media={"media/session_side_by_side.mp4": replacement},
        )
        server = self.start_server()

        with urlopen(server.base_url + "/media/session_side_by_side.mp4") as response:
            body = response.read()

        self.assertEqual(body, b"new-generation-media")

    def test_copy_fallback_fsyncs_media_before_activation(self):
        store = GenerationStore(self.root)
        with patch("coach_app.review_bundle.os.link", side_effect=OSError("cross-device")), patch.object(
            GenerationStore, "_fsync_file", wraps=GenerationStore._fsync_file
        ) as fsync_file:
            snapshot = store.stage_and_activate(
                self.review,
                self.review["events"],
                "csv",
                "markdown",
            )

        copied = snapshot.root / "media" / "session_side_by_side.mp4"
        self.assertTrue(copied.is_file())
        fsync_file.assert_any_call(copied)

    def test_reveal_requires_lock_and_can_happen_only_once(self):
        with self.assertRaisesRegex(ValueError, "pre_ai"):
            self.service.reveal_ai("e-001")

        locked = self.service.lock_assessment("e-001", self.visible_assessment())
        self.now += timedelta(seconds=1)
        revealed = self.service.reveal_ai("e-001")

        self.assertEqual(
            revealed["event"]["aktivni_duel"]["trener_revizija"],
            locked["assessment"]["revizija"],
        )
        self.assertEqual(
            active_ai_evaluation(revealed["event"])["ai_otkriven_u"],
            "2026-08-01T12:00:01+00:00",
        )
        public = self.service.public_event("e-001")
        self.assertIn("ai_procene", public)
        self.assertIn("imu_eksperimentalno", public)
        with self.assertRaisesRegex(ValueError, "već otkrivena"):
            self.service.reveal_ai("e-001")

    def test_feedback_requires_reveal_and_is_bound_to_active_duel(self):
        self.service.lock_assessment("e-001", self.visible_assessment())
        feedback = {
            "odnos": "delimicno",
            "razlog": "Rotacija je vidljiva, ali ulaz nije potpun.",
            "procene_dokaza": [],
        }
        with self.assertRaisesRegex(ValueError, "otkriven"):
            self.service.save_ai_feedback("e-001", feedback)
        self.now += timedelta(seconds=1)
        self.service.reveal_ai("e-001")
        self.now += timedelta(seconds=1)

        result = self.service.save_ai_feedback("e-001", feedback)

        saved = result["assessment"]
        duel = result["event"]["aktivni_duel"]
        self.assertEqual(saved["trener_revizija"], duel["trener_revizija"])
        self.assertEqual(saved["sacuvano_u"], "2026-08-01T12:00:02+00:00")

    def test_post_ai_correction_preserves_first_assessment(self):
        first = self.service.lock_assessment("e-001", self.visible_assessment())[
            "assessment"
        ]
        first_bytes = json.dumps(first, ensure_ascii=False, sort_keys=True).encode()
        self.now += timedelta(seconds=1)
        self.service.reveal_ai("e-001")
        self.now += timedelta(seconds=1)

        corrected = self.service.lock_assessment(
            "e-001",
            self.visible_assessment(
                ocena=5,
                razlog="Na 12.000 s završetak je stabilniji nego u prvoj proceni.",
                citirani_sony_trenuci_s=[12.0],
            ),
        )

        self.assertEqual(corrected["assessment"]["faza"], "post_ai_korekcija")
        self.assertEqual(
            json.dumps(
                corrected["event"]["trener_procene"][0],
                ensure_ascii=False,
                sort_keys=True,
            ).encode(),
            first_bytes,
        )
        self.assertEqual(corrected["event"]["ocena"], 5)

    def test_lock_rejects_invalid_payloads_and_accepts_null_invisible_event(self):
        invalid = (
            self.visible_assessment(ocena=True),
            self.visible_assessment(citirani_sony_trenuci_s=[]),
            self.visible_assessment(citirani_sony_trenuci_s=[15.0]),
        )
        for payload in invalid:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    self.service.lock_assessment("e-001", payload)
        with self.assertRaisesRegex(ValueError, "povredni"):
            self.service.lock_assessment("povreda", self.visible_assessment())

        result = self.service.lock_assessment(
            "e-001",
            {
                "status_vidljivosti": "nedovoljno_vidljivo",
                "potvrdena_tehnika": None,
                "ocena": None,
                "razlog": None,
                "citirani_sony_trenuci_s": None,
            },
        )
        self.assertIsNone(result["assessment"]["ocena"])

    def test_pointer_switch_failure_keeps_legacy_snapshot_and_cleans_stage(self):
        original_replace = __import__("os").replace

        def fail_pointer(source, destination):
            if Path(destination).name == "current-generation.json":
                raise OSError("pointer failure")
            return original_replace(source, destination)

        with patch("coach_app.review_bundle.os.replace", side_effect=fail_pointer):
            with self.assertRaisesRegex(OSError, "pointer failure"):
                self.service.lock_assessment("e-001", self.visible_assessment())

        self.assertIsNone(GenerationStore(self.root).resolve_current().generation_id)
        generation_root = self.root / ".review-generations"
        self.assertEqual(list(generation_root.iterdir()), [])
        legacy = json.loads(self.review_path.read_text(encoding="utf-8"))
        self.assertEqual(legacy["events"][0]["trener_procene"], [])

    def test_post_switch_fsync_failure_keeps_published_snapshot_readable(self):
        original_fsync_directory = GenerationStore._fsync_directory

        def fail_after_switch(path):
            if (
                Path(path).resolve() == self.root.resolve()
                and (self.root / "current-generation.json").exists()
            ):
                raise OSError("directory fsync failure")
            return original_fsync_directory(path)

        with patch.object(
            GenerationStore, "_fsync_directory", side_effect=fail_after_switch
        ):
            with self.assertWarnsRegex(RuntimeWarning, "fsync"):
                result = self.service.lock_assessment(
                    "e-001", self.visible_assessment()
                )

        snapshot = GenerationStore(self.root).resolve_current()
        self.assertEqual(result["assessment"]["faza"], "pre_ai")
        self.assertIsNotNone(snapshot.generation_id)
        self.assertTrue(snapshot.review_path.is_file())
        self.assertTrue(snapshot.event_metrics_path.is_file())
        self.assertTrue(snapshot.csv_path.is_file())
        self.assertTrue(snapshot.markdown_path.is_file())
        self.assertEqual(
            json.loads(snapshot.review_path.read_text(encoding="utf-8"))["events"][0][
                "trener_procene"
            ],
            [
                {
                    "revizija": 1,
                    "faza": "pre_ai",
                    "event_revision": 1,
                    "analysis_fingerprint": self.review["events"][0][
                        "analysis_fingerprint"
                    ],
                    "trainer_name": "Marko",
                    "wrestler_name": "Dusan",
                    **self.visible_assessment(),
                    "zakljucano_u": "2026-08-01T12:00:00+00:00",
                }
            ],
        )

    def test_render_and_media_copy_failures_leave_current_snapshot_unchanged(self):
        with patch(
            "coach_app.trainer_ai_service.write_reports",
            side_effect=OSError("render failure"),
        ):
            with self.assertRaisesRegex(OSError, "render failure"):
                self.service.lock_assessment("e-001", self.visible_assessment())
        self.assertIsNone(self.service.store.resolve_current().generation_id)

        with patch("coach_app.review_bundle.os.link", side_effect=OSError("link")), patch(
            "coach_app.review_bundle.shutil.copy2", side_effect=OSError("copy failure")
        ):
            with self.assertRaisesRegex(OSError, "copy failure"):
                self.service.lock_assessment("e-001", self.visible_assessment())
        self.assertIsNone(self.service.store.resolve_current().generation_id)

    def test_reader_sees_complete_old_snapshot_until_pointer_switch(self):
        reached_switch = threading.Event()
        continue_switch = threading.Event()
        original_replace = __import__("os").replace

        def pause_pointer(source, destination):
            if Path(destination).name == "current-generation.json":
                reached_switch.set()
                self.assertTrue(continue_switch.wait(timeout=2))
            return original_replace(source, destination)

        errors = []

        def mutate():
            try:
                self.service.lock_assessment("e-001", self.visible_assessment())
            except Exception as exc:  # pragma: no cover - asserted below
                errors.append(exc)

        with patch("coach_app.review_bundle.os.replace", side_effect=pause_pointer):
            worker = threading.Thread(target=mutate)
            worker.start()
            self.assertTrue(reached_switch.wait(timeout=2))
            during = self.service.public_event("e-001")
            during_snapshot = self.service.store.resolve_current()
            self.assertEqual(during.get("trener_procene"), [])
            self.assertEqual(during_snapshot.root, self.root.resolve())
            self.assertTrue(during_snapshot.csv_path.is_file())
            self.assertTrue(during_snapshot.markdown_path.is_file())
            continue_switch.set()
            worker.join(timeout=2)

        self.assertEqual(errors, [])
        after = self.service.public_event("e-001")
        after_snapshot = self.service.store.resolve_current()
        self.assertEqual(len(after["trener_procene"]), 1)
        self.assertNotEqual(after_snapshot.root, self.root.resolve())
        self.assertTrue(after_snapshot.csv_path.is_file())
        self.assertTrue(after_snapshot.markdown_path.is_file())

    def test_http_trainer_first_flow_redacts_ai_until_reveal(self):
        server = self.start_server()

        _, session = self.read_json(server.base_url + "/api/session")
        _, event = self.read_json(server.base_url + "/api/events/e-001")
        for payload in (session, event):
            encoded = json.dumps(payload, ensure_ascii=False)
            self.assertNotIn("predlozena_ocena", encoded)
            self.assertNotIn("\"dokazi\"", encoded)
            self.assertNotIn("imu_eksperimentalno", encoded)

        status, locked = self.read_json(
            server.base_url + "/api/events/e-001/trainer-assessments",
            method="POST",
            payload=self.visible_assessment(),
        )
        self.assertEqual(status, 200)
        self.assertNotIn("ai_procene", locked["event"])

        self.now += timedelta(seconds=1)
        status, revealed = self.read_json(
            server.base_url + "/api/events/e-001/ai-reveal",
            method="POST",
            payload={},
        )
        self.assertEqual(status, 200)
        self.assertIn("ai_procene", revealed["event"])
        self.assertIn("imu_eksperimentalno", revealed["event"])

        self.now += timedelta(seconds=1)
        status, feedback = self.read_json(
            server.base_url + "/api/events/e-001/ai-feedback",
            method="PUT",
            payload={
                "odnos": "slazem_se",
                "razlog": None,
                "procene_dokaza": [],
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(feedback["assessment"]["odnos"], "slazem_se")

    def test_http_reports_hide_ai_until_reveal_and_then_include_trace_data(self):
        server = self.start_server()

        def reports():
            values = []
            for name in ("izvestaj.csv", "izvestaj.md"):
                with urlopen(server.base_url + "/" + name) as response:
                    values.append(response.read().decode("utf-8"))
            return values

        for report in reports():
            self.assertNotIn("deterministicki-v1", report)
            self.assertNotIn("video_pose_proxy_v1", report)

        self.read_json(
            server.base_url + "/api/events/e-001/trainer-assessments",
            method="POST",
            payload=self.visible_assessment(),
        )
        for report in reports():
            self.assertIn("kukovi kasne za rotacijom", report)
            self.assertNotIn("deterministicki-v1", report)
            self.assertNotIn("video_pose_proxy_v1", report)

        self.now += timedelta(seconds=1)
        self.read_json(
            server.base_url + "/api/events/e-001/ai-reveal",
            method="POST",
            payload={},
        )
        for report in reports():
            self.assertIn("deterministicki-v1", report)
            self.assertIn("video_pose_proxy_v1", report)
            self.assertIn("sony_s", report)

    def test_http_denies_internal_review_and_analysis_paths(self):
        server = self.start_server()
        for path in (
            "/review.json",
            "/analysis/",
            "/analysis/event_metrics.json",
            "/%61nalysis/event_metrics.json",
            "/current-generation.json",
            "/.review-generations/anything/review.json",
        ):
            with self.subTest(path=path):
                with self.assertRaises(HTTPError) as raised:
                    urlopen(server.base_url + path)
                self.assertEqual(raised.exception.code, 404)

    def test_legacy_annotation_route_returns_redacted_v3_event(self):
        server = self.start_server()

        status, event = self.read_json(
            server.base_url + "/api/events/e-001/annotation",
            method="PUT",
            payload={
                "potvrdena_tehnika": "Tai-otoshi",
                "ocena": None,
                "napomena": "Draft naziv bez zaključane ocene.",
            },
        )

        self.assertEqual(status, 200)
        self.assertNotIn("ai_procene", event)
        self.assertNotIn("imu_eksperimentalno", event)
        self.assertIsNone(event["ocena"])

    def test_legacy_annotation_updates_active_generation_after_lock(self):
        server = self.start_server()
        self.read_json(
            server.base_url + "/api/events/e-001/trainer-assessments",
            method="POST",
            payload=self.visible_assessment(),
        )

        _, event = self.read_json(
            server.base_url + "/api/events/e-001/annotation",
            method="PUT",
            payload={
                "potvrdena_tehnika": "Morote-seoi-nage",
                "ocena": None,
                "napomena": "Draft promenjen posle zaključavanja.",
            },
        )
        _, fetched = self.read_json(server.base_url + "/api/events/e-001")

        self.assertEqual(event["potvrdena_tehnika"], "Morote-seoi-nage")
        self.assertEqual(fetched["potvrdena_tehnika"], "Morote-seoi-nage")
        self.assertEqual(fetched["napomena"], "Draft promenjen posle zaključavanja.")
        self.assertNotIn("ai_procene", fetched)


if __name__ == "__main__":
    unittest.main()
