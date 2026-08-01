import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from pipeline.trainer_dataset_export import build_trainer_exports, render_trainer_exports


GENERATION_ID = "a" * 32
GENERATED_AT = "2026-08-02T10:00:00+02:00"
ACTIVE_FINGERPRINT = "sha256:" + "a" * 64
OLD_FINGERPRINT = "sha256:" + "b" * 64


class TrainerDatasetExportTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self._write_clip("e-001", "sony", b"sony-active")
        self._write_clip("e-001", "iphone", b"iphone-active")
        self._write_clip("e-insufficient", "sony", b"sony-insufficient")
        self._write_clip("e-insufficient", "iphone", b"iphone-insufficient")
        self._write_clip("e-unverified", "sony", b"sony-unverified")
        self._write_clip("e-unverified", "iphone", b"iphone-unverified")

    def tearDown(self):
        self.tempdir.cleanup()

    def _write_clip(self, event_id, camera, content):
        path = self.root / "events" / event_id / f"{camera}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    @staticmethod
    def _assessment(**overrides):
        assessment = {
            "revizija": 1,
            "faza": "pre_ai",
            "event_revision": 2,
            "analysis_fingerprint": ACTIVE_FINGERPRINT,
            "trainer_name": "Marko Markovic",
            "wrestler_name": "Dusan",
            "status_vidljivosti": "dovoljno_vidljivo",
            "potvrdena_tehnika": "Tai-otoshi",
            "ocena": 4,
            "razlog": "Kukovi ulaze pre rotacije.",
            "citirani_sony_trenuci_s": [12.5],
            "zakljucano_u": "2026-08-02T09:00:00+02:00",
        }
        assessment.update(overrides)
        return assessment

    def _review(self):
        active_pre = self._assessment()
        active_post = self._assessment(
            revizija=2,
            faza="post_ai_korekcija",
            ocena=5,
            razlog="Naknadna korekcija trenera.",
            zakljucano_u="2026-08-02T09:10:00+02:00",
        )
        old_pre = self._assessment(
            revizija=3,
            event_revision=1,
            analysis_fingerprint=OLD_FINGERPRINT,
            zakljucano_u="2026-08-01T09:00:00+02:00",
        )
        insufficient = self._assessment(
            revizija=4,
            event_revision=1,
            analysis_fingerprint="sha256:" + "c" * 64,
            status_vidljivosti="nedovoljno_vidljivo",
            potvrdena_tehnika=None,
            ocena=None,
            razlog=None,
            citirani_sony_trenuci_s=[],
        )
        unverified = self._assessment(
            revizija=5,
            event_revision=1,
            analysis_fingerprint="sha256:" + "d" * 64,
        )
        return {
            "session_id": "trainer-ai-session",
            "participants": {
                "trainer_name": "Current Trainer",
                "wrestler_name": "Current Wrestler",
                "updated_at": "2026-08-02T08:00:00+02:00",
            },
            "derived_media_manifest": [
                self._manifest("e-001", "sony", verified=True),
                self._manifest("e-001", "iphone", verified=True),
                self._manifest("e-insufficient", "sony", verified=True),
                self._manifest("e-insufficient", "iphone", verified=True),
                self._manifest("e-unverified", "sony", verified=False),
                self._manifest("e-unverified", "iphone", verified=True),
            ],
            "events": [
                {
                    "event_id": "e-001",
                    "event_revision": 2,
                    "analysis_fingerprint": ACTIVE_FINGERPRINT,
                    "sony_start_s": 10.0,
                    "sony_end_s": 15.0,
                    "iphone_start_s": 20.0,
                    "iphone_end_s": 25.0,
                    "trener_procene": [active_pre, active_post, old_pre],
                    "ai_procene": [{"ai_score": 5, "reason": "must not export"}],
                    "imu_eksperimentalno": {"ai_score": 3},
                    "procene_ai_predloga": [{"ai_score": 1}],
                    "aktivni_duel": {"ai_score": 2},
                },
                {
                    "event_id": "e-insufficient",
                    "event_revision": 1,
                    "analysis_fingerprint": insufficient["analysis_fingerprint"],
                    "sony_start_s": 30.0,
                    "sony_end_s": 35.0,
                    "iphone_start_s": 40.0,
                    "iphone_end_s": 45.0,
                    "trener_procene": [insufficient],
                },
                {
                    "event_id": "e-unverified",
                    "event_revision": 1,
                    "analysis_fingerprint": unverified["analysis_fingerprint"],
                    "sony_start_s": 50.0,
                    "sony_end_s": 55.0,
                    "iphone_start_s": 60.0,
                    "iphone_end_s": 65.0,
                    "trener_procene": [unverified],
                },
                {
                    "event_id": "injury",
                    "prijavljen_povredni_dogadjaj": True,
                    "iskljuceno_iz_statistike": True,
                    "status": "povreda",
                    "trener_procene": [],
                },
            ],
        }

    @staticmethod
    def _manifest(event_id, camera, *, verified):
        return {
            "relative_path": f"events/{event_id}/{camera}.mp4",
            "media_type": "event_clip",
            "privacy_verified": verified,
            "failure_reason": None if verified else "face blur failed",
        }

    def test_builds_clean_pre_ai_examples_and_audits_every_assessment(self):
        dataset, audit = build_trainer_exports(
            self._review(),
            generation_id=GENERATION_ID,
            bundle_root=self.root,
            generated_at=GENERATED_AT,
        )

        self.assertEqual(
            [row["assessment_phase"] for row in dataset["training_examples"]],
            ["pre_ai"],
        )
        self.assertNotIn("ai_score", json.dumps(dataset))
        self.assertEqual(
            [
                row["assessment_revision"]
                for row in audit["assessments"]
                if row["event_id"] == "e-001"
            ],
            [1, 2, 1],
        )
        self.assertIn(
            "post_ai_correction", audit["assessments"][1]["ineligibility_reasons"]
        )
        self.assertIn(
            "inactive_analysis_round", audit["assessments"][2]["ineligibility_reasons"]
        )

    def test_records_ineligibility_reasons_for_visibility_and_privacy(self):
        _, audit = build_trainer_exports(
            self._review(),
            generation_id=GENERATION_ID,
            bundle_root=self.root,
            generated_at=GENERATED_AT,
        )
        rows = {row["event_id"]: row for row in audit["assessments"]}

        self.assertIn("insufficient_visibility", rows["e-insufficient"]["ineligibility_reasons"])
        self.assertIn("missing_verified_media", rows["e-unverified"]["ineligibility_reasons"])

    def test_records_missing_identity_snapshot_as_ineligible(self):
        review = self._review()
        review["events"][0]["trener_procene"][0]["trainer_name"] = ""

        dataset, audit = build_trainer_exports(
            review,
            generation_id=GENERATION_ID,
            bundle_root=self.root,
            generated_at=GENERATED_AT,
        )

        self.assertEqual(dataset["training_examples"], [])
        self.assertIn(
            "missing_identity_snapshot",
            audit["assessments"][0]["ineligibility_reasons"],
        )

    def test_binds_clean_example_to_verified_media_bytes(self):
        dataset, _ = build_trainer_exports(
            self._review(),
            generation_id=GENERATION_ID,
            bundle_root=self.root,
            generated_at=GENERATED_AT,
        )
        example = dataset["training_examples"][0]

        self.assertEqual(example["generation_id"], GENERATION_ID)
        self.assertEqual(example["trainer_name"], "Marko Markovic")
        self.assertEqual(example["wrestler_name"], "Dusan")
        self.assertEqual(
            example["evidence"]["sony_clip"],
            {
                "bundle_relative_path": "events/e-001/sony.mp4",
                "review_url": "/media/events/e-001/sony.mp4",
                "sha256": hashlib.sha256(b"sony-active").hexdigest(),
            },
        )
        self.assertEqual(
            example["evidence"]["iphone_clip"]["sha256"],
            hashlib.sha256(b"iphone-active").hexdigest(),
        )

    def test_rendering_is_deterministic_finite_and_newline_terminated(self):
        dataset_text, audit_text = render_trainer_exports(
            self._review(),
            generation_id=GENERATION_ID,
            bundle_root=self.root,
            generated_at=GENERATED_AT,
        )

        self.assertTrue(dataset_text.endswith("\n"))
        self.assertTrue(audit_text.endswith("\n"))
        self.assertEqual(dataset_text, render_trainer_exports(
            self._review(),
            generation_id=GENERATION_ID,
            bundle_root=self.root,
            generated_at=GENERATED_AT,
        )[0])
        self.assertEqual(json.loads(dataset_text)["generation_id"], GENERATION_ID)

    def test_rejects_missing_or_unsafe_verified_media_paths(self):
        missing = self._review()
        (self.root / "events" / "e-001" / "sony.mp4").unlink()
        with self.assertRaisesRegex(ValueError, "nedostaje"):
            build_trainer_exports(
                missing,
                generation_id=GENERATION_ID,
                bundle_root=self.root,
                generated_at=GENERATED_AT,
            )

        unsafe = self._review()
        unsafe["events"][0]["event_id"] = "../escape"
        with self.assertRaisesRegex(ValueError, "putanja"):
            build_trainer_exports(
                unsafe,
                generation_id=GENERATION_ID,
                bundle_root=self.root,
                generated_at=GENERATED_AT,
            )

    def test_rendering_rejects_non_finite_metrics(self):
        review = self._review()
        review["events"][0]["brzina_ulaska_norm"] = float("nan")

        with self.assertRaises(ValueError):
            render_trainer_exports(
                review,
                generation_id=GENERATION_ID,
                bundle_root=self.root,
                generated_at=GENERATED_AT,
            )


if __name__ == "__main__":
    unittest.main()
