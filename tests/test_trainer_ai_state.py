import copy
import unittest
from unittest.mock import patch

from pipeline.trainer_ai_state import (
    active_ai_evaluation,
    active_trainer_assessment,
    migrate_trainer_ai_payload,
    start_new_event_revision,
    validate_participants,
    validate_trainer_ai_event,
)
from pipeline.video_review_contract import validate_review_payload


def legacy_review_fixture():
    frames = []
    for index in range(13):
        frames.append({
            "frame_index": index,
            "timestamp_s": index / 6.0,
            "hip_midpoint": [100.0 + index, 200.0],
            "shoulder_midpoint": [100.0 + index, 180.0],
            "vidljivo": True,
            "interpolirano": False,
            "brzina_ulaska_norm": index / 3.0,
            "rotacija_trupa_2d_dps": index * 45.0,
            "promena_visine_kukova_norm": index / 100.0,
            "sirina_stava_norm": 1.0,
            "proxy_ubrzanja_norm_s2": None if index == 0 else 2.0,
        })
    return {
        "version": 2,
        "effective_analysis_fps": 6.0,
        "pose_analysis": {"selected_track_id": 7},
        "sources": {
            "sony": {"sha256": "a" * 64},
            "iphone": {"sha256": "b" * 64},
        },
        "frame_metrics": frames,
        "events": [
            {
                "event_id": "e-1",
                "sony_start_s": 0.0,
                "sony_end_s": 2.0,
                "iphone_start_s": 3.0,
                "iphone_end_s": 5.0,
                "potvrdena_tehnika": "Tai-otoshi",
                "ocena": 4,
                "napomena": "Stara ocena nije nastala pre AI.",
                "status": "trener",
                "iskljuceno_iz_statistike": False,
            },
            {
                "event_id": "povreda",
                "sony_start_s": 2.0,
                "sony_end_s": 2.5,
                "iphone_start_s": 5.0,
                "iphone_end_s": 5.5,
                "prijavljen_povredni_dogadjaj": True,
                "iskljuceno_iz_statistike": True,
                "status": "povreda",
            },
        ],
    }


def valid_locked_event():
    event = migrate_trainer_ai_payload(legacy_review_fixture())["events"][0]
    fingerprint = event["analysis_fingerprint"]
    event["trener_procene"] = [{
        "revizija": 1,
        "faza": "pre_ai",
        "event_revision": 1,
        "analysis_fingerprint": fingerprint,
        "status_vidljivosti": "dovoljno_vidljivo",
        "potvrdena_tehnika": "Tai-otoshi",
        "ocena": 4,
        "razlog": "Na 1.000 s rotacija kasni.",
        "citirani_sony_trenuci_s": [1.0],
        "zakljucano_u": "2026-08-01T12:00:00+02:00",
    }]
    event["aktivna_trener_revizija"] = 1
    event["ai_procene"][0]["ai_otkriven_u"] = "2026-08-01T12:01:00+02:00"
    event["aktivni_duel"] = {
        "event_revision": 1,
        "analysis_fingerprint": fingerprint,
        "trener_revizija": 1,
        "evaluator_id": "deterministicki-v1",
    }
    return event


def valid_review_payload():
    review = migrate_trainer_ai_payload(legacy_review_fixture())
    review.update({
        "session_id": "trainer-state",
        "sony_video": "sony.mp4",
        "iphone_video": "iphone.mp4",
        "sony_duration_s": 6.0,
        "iphone_duration_s": 9.0,
        "sony_fps": 30.0,
        "iphone_fps": 30.0,
        "anchors": [
            {
                "name": "pocetak",
                "sony_s": 0.0,
                "iphone_s": 3.0,
                "user_confirmed": True,
                "triple_tap_count": 3,
            },
            {
                "name": "kontrola",
                "sony_s": 4.0,
                "iphone_s": 7.0,
                "user_confirmed": True,
                "triple_tap_count": 3,
            },
        ],
        "time_map": {"slope": 1.0, "intercept": -3.0},
        "injury_cutoff_s": 5.0,
    })
    injury = review["events"][1]
    injury.update({
        "sony_start_s": 5.0,
        "sony_end_s": 5.5,
        "iphone_start_s": 8.0,
        "iphone_end_s": 8.5,
    })
    review["event_metrics"] = copy.deepcopy(review["events"])
    return review


class TrainerAiStateTests(unittest.TestCase):
    def test_validation_groups_assessment_phase_by_revision_and_fingerprint(self):
        event = valid_locked_event()
        alternate_fingerprint = "sha256:" + "f" * 64
        alternate_ai = copy.deepcopy(event["ai_procene"][0])
        alternate_ai.update({
            "analysis_fingerprint": alternate_fingerprint,
            "ai_otkriven_u": None,
        })
        event["ai_procene"].append(alternate_ai)
        alternate_assessment = copy.deepcopy(event["trener_procene"][0])
        alternate_assessment.update({
            "revizija": 3,
            "analysis_fingerprint": alternate_fingerprint,
            "zakljucano_u": "2026-08-01T12:03:00+02:00",
        })
        post_ai = copy.deepcopy(event["trener_procene"][0])
        post_ai.update({
            "revizija": 2,
            "faza": "post_ai_korekcija",
            "zakljucano_u": "2026-08-01T12:02:00+02:00",
        })
        event["trener_procene"] = [post_ai, alternate_assessment, event["trener_procene"][0]]

        validate_trainer_ai_event(event)

    def test_validation_rejects_post_ai_assessment_before_reveal(self):
        event = valid_locked_event()
        event["trener_procene"].append({
            **event["trener_procene"][0],
            "revizija": 2,
            "faza": "post_ai_korekcija",
            "zakljucano_u": "2026-08-01T09:59:59+00:00",
        })
        with self.assertRaisesRegex(ValueError, "otkriv"):
            validate_trainer_ai_event(event)

    def test_validate_participants_requires_exact_fields_and_normalizes_names(self):
        with self.assertRaisesRegex(ValueError, "tačna obavezna polja"):
            validate_participants({"trainer_name": "Marko"})
        with self.assertRaisesRegex(ValueError, "ime trenera"):
            validate_participants({
                "trainer_name": " ",
                "wrestler_name": "Dusan",
                "updated_at": "2026-08-01T12:00:00+00:00",
            })

        self.assertEqual(
            validate_participants({
                "trainer_name": "  Marko Markovic  ",
                "wrestler_name": " Dusan ",
                "updated_at": "2026-08-01T12:00:00+00:00",
            }),
            {
                "trainer_name": "Marko Markovic",
                "wrestler_name": "Dusan",
                "updated_at": "2026-08-01T12:00:00+00:00",
            },
        )

    def test_review_rejects_duplicate_global_trainer_revision_across_events(self):
        review = valid_review_payload()
        first = valid_locked_event()["trener_procene"][0]
        first["analysis_fingerprint"] = review["events"][0]["analysis_fingerprint"]
        review["events"][0]["trener_procene"] = [copy.deepcopy(first)]
        review["events"][0]["aktivna_trener_revizija"] = 1
        review["events"][0]["ai_procene"][0]["ai_otkriven_u"] = (
            "2026-08-01T12:01:00+02:00"
        )
        review["events"][0]["aktivni_duel"] = {
            "event_revision": 1,
            "analysis_fingerprint": review["events"][0]["analysis_fingerprint"],
            "trener_revizija": 1,
            "evaluator_id": "deterministicki-v1",
        }
        second = copy.deepcopy(review["events"][0])
        second.update({
            "event_id": "e-2",
            "sony_start_s": 2.0,
            "sony_end_s": 4.0,
            "iphone_start_s": 5.0,
            "iphone_end_s": 7.0,
        })
        from pipeline.trainer_ai_evaluator import compute_analysis_fingerprint

        second_fingerprint = compute_analysis_fingerprint(review, second)
        second["analysis_fingerprint"] = second_fingerprint
        second["ai_procene"][0]["analysis_fingerprint"] = second_fingerprint
        second["ai_procene"][0]["dokazi"] = []
        second["trener_procene"][0]["analysis_fingerprint"] = second_fingerprint
        second["trener_procene"][0]["citirani_sony_trenuci_s"] = [3.0]
        second["aktivni_duel"]["analysis_fingerprint"] = second_fingerprint
        review["events"].append(second)
        review["event_metrics"] = copy.deepcopy(review["events"])

        with self.assertRaisesRegex(ValueError, "globalno jedinstvena"):
            validate_review_payload(review)

    def test_migration_adds_versioned_state_without_inventing_scores(self):
        migrated = migrate_trainer_ai_payload(legacy_review_fixture())
        normal, injury = migrated["events"]

        self.assertEqual(migrated["version"], 3)
        self.assertEqual(normal["event_revision"], 1)
        self.assertRegex(normal["analysis_fingerprint"], r"^sha256:[0-9a-f]{64}$")
        self.assertEqual(normal["trener_procene"], [])
        self.assertIsNone(normal["ocena"])
        self.assertEqual(normal["legacy_annotations"][0]["ocena"], 4)
        self.assertTrue(normal["legacy_annotations"][0]["nije_pre_ai"])
        self.assertEqual(len(normal["ai_procene"]), 1)
        self.assertIsNone(normal["ai_procene"][0]["ai_otkriven_u"])
        self.assertIn("imu_eksperimentalno", normal)
        self.assertNotIn("ai_procene", injury)
        self.assertEqual(migrated, migrate_trainer_ai_payload(migrated))

    def test_active_selectors_follow_current_pointers(self):
        event = valid_locked_event()

        self.assertEqual(active_ai_evaluation(event)["event_revision"], 1)
        self.assertEqual(active_trainer_assessment(event)["revizija"], 1)
        validate_trainer_ai_event(event)

    def test_validation_rejects_duplicate_trainer_revisions(self):
        event = valid_locked_event()
        event["trener_procene"].append(copy.deepcopy(event["trener_procene"][0]))

        with self.assertRaisesRegex(ValueError, "revizija"):
            validate_trainer_ai_event(event)

    def test_validation_rejects_bad_fingerprint(self):
        event = valid_locked_event()
        event["trener_procene"][0]["analysis_fingerprint"] = "sha256:bad"

        with self.assertRaisesRegex(ValueError, "fingerprint"):
            validate_trainer_ai_event(event)

    def test_validation_rejects_wrong_pre_ai_fingerprint(self):
        event = valid_locked_event()
        event["trener_procene"][0]["analysis_fingerprint"] = "sha256:" + "c" * 64

        with self.assertRaisesRegex(ValueError, "AI procen"):
            validate_trainer_ai_event(event)

    def test_validation_rejects_invalid_scores_and_missing_citation(self):
        for score in (True, 0, 6):
            with self.subTest(score=score):
                event = valid_locked_event()
                event["trener_procene"][0]["ocena"] = score
                with self.assertRaises((TypeError, ValueError)):
                    validate_trainer_ai_event(event)

        event = valid_locked_event()
        event["trener_procene"][0]["citirani_sony_trenuci_s"] = []
        with self.assertRaisesRegex(ValueError, "Sony"):
            validate_trainer_ai_event(event)

    def test_validation_rejects_post_ai_feedback_without_time(self):
        event = valid_locked_event()
        event["procene_ai_predloga"] = [{
            "event_revision": 1,
            "analysis_fingerprint": event["analysis_fingerprint"],
            "trener_revizija": 1,
            "evaluator_id": "deterministicki-v1",
            "odnos": "slazem_se",
            "razlog": None,
            "procene_dokaza": [],
            "sacuvano_u": None,
        }]

        with self.assertRaisesRegex((TypeError, ValueError), "sacuvano_u"):
            validate_trainer_ai_event(event)

    def test_validation_rejects_feedback_for_unknown_trainer_revision(self):
        event = valid_locked_event()
        event["procene_ai_predloga"] = [{
            "event_revision": 1,
            "analysis_fingerprint": event["analysis_fingerprint"],
            "trener_revizija": 99,
            "evaluator_id": "deterministicki-v1",
            "odnos": "ne_slazem_se",
            "razlog": "Ne odgovara snimku.",
            "procene_dokaza": [],
            "sacuvano_u": "2026-08-01T12:02:00+02:00",
        }]

        with self.assertRaisesRegex(ValueError, "trener"):
            validate_trainer_ai_event(event)

    def test_validation_rejects_trainer_revision_from_an_old_event_round(self):
        event = valid_locked_event()
        old_fingerprint = event["analysis_fingerprint"]
        new_fingerprint = "sha256:" + "d" * 64
        event["event_revision"] = 2
        event["analysis_fingerprint"] = new_fingerprint
        next_ai = copy.deepcopy(event["ai_procene"][0])
        next_ai.update({
            "event_revision": 2,
            "analysis_fingerprint": new_fingerprint,
            "ai_otkriven_u": "2026-08-01T12:03:00+02:00",
        })
        event["ai_procene"].append(next_ai)
        event["aktivni_duel"] = {
            "event_revision": 2,
            "analysis_fingerprint": new_fingerprint,
            "trener_revizija": 1,
            "evaluator_id": "deterministicki-v1",
        }
        self.assertEqual(event["trener_procene"][0]["analysis_fingerprint"], old_fingerprint)

        with self.assertRaisesRegex(ValueError, "rund"):
            validate_trainer_ai_event(event)

    def test_validation_rejects_feedback_mixed_across_rounds(self):
        event = valid_locked_event()
        new_fingerprint = "sha256:" + "e" * 64
        next_ai = copy.deepcopy(event["ai_procene"][0])
        next_ai.update({
            "event_revision": 2,
            "analysis_fingerprint": new_fingerprint,
            "ai_otkriven_u": "2026-08-01T12:03:00+02:00",
        })
        event["ai_procene"].append(next_ai)
        event["procene_ai_predloga"] = [{
            "event_revision": 2,
            "analysis_fingerprint": new_fingerprint,
            "trener_revizija": 1,
            "evaluator_id": "deterministicki-v1",
            "odnos": "ne_slazem_se",
            "razlog": None,
            "procene_dokaza": [],
            "sacuvano_u": "2026-08-01T12:04:00+02:00",
        }]

        with self.assertRaisesRegex(ValueError, "rund"):
            validate_trainer_ai_event(event)

    def test_validation_requires_strict_iso_time_and_evidence_unit(self):
        event = valid_locked_event()
        event["trener_procene"][0]["zakljucano_u"] = "20260801T120000+0200"
        with self.assertRaisesRegex(ValueError, "ISO-8601"):
            validate_trainer_ai_event(event)

        event = valid_locked_event()
        event["ai_procene"][0]["dokazi"][0]["jedinica"] = ""
        with self.assertRaisesRegex(ValueError, "jedinica"):
            validate_trainer_ai_event(event)

    def test_validation_rejects_revealed_ai_without_pre_ai_assessment(self):
        event = migrate_trainer_ai_payload(legacy_review_fixture())["events"][0]
        event["ai_procene"][0]["ai_otkriven_u"] = "2026-08-01T12:01:00+02:00"

        with self.assertRaisesRegex(ValueError, "pre_ai"):
            validate_trainer_ai_event(event)

    def test_validation_rejects_confidence_outside_zero_to_one(self):
        event = valid_locked_event()
        event["ai_procene"][0]["pouzdanost_0_1"] = 2.0

        with self.assertRaisesRegex(ValueError, "0..1"):
            validate_trainer_ai_event(event)

    def test_changed_source_starts_a_new_hidden_event_revision(self):
        migrated = migrate_trainer_ai_payload(legacy_review_fixture())
        migrated["sources"]["sony"]["sha256"] = "c" * 64

        revised = migrate_trainer_ai_payload(migrated)
        event = revised["events"][0]

        self.assertEqual(event["event_revision"], 2)
        self.assertEqual(len(event["ai_procene"]), 2)
        self.assertIsNone(event["ai_procene"][-1]["ai_otkriven_u"])
        self.assertIsNone(event["aktivna_trener_revizija"])
        self.assertIsNone(event["aktivni_duel"])

    def test_revision_helper_is_noop_until_a_fingerprinted_input_changes(self):
        review = migrate_trainer_ai_payload(legacy_review_fixture())
        event = review["events"][0]
        before = copy.deepcopy(event)

        start_new_event_revision(review, event)

        self.assertEqual(event, before)
        event["selected_track_id"] = 8
        start_new_event_revision(review, event)
        self.assertEqual(event["event_revision"], 2)
        self.assertEqual(len(event["ai_procene"]), 2)
        self.assertIsNone(event["ai_procene"][-1]["ai_otkriven_u"])

    def test_pose_and_evaluator_versions_each_start_a_new_round(self):
        review = migrate_trainer_ai_payload(legacy_review_fixture())
        event = review["events"][0]

        with patch("pipeline.trainer_ai_evaluator.POSE_METRICS_ID", "video-pose-metrics-v2"):
            start_new_event_revision(review, event)
        self.assertEqual(event["event_revision"], 2)

        with patch(
            "pipeline.trainer_ai_evaluator.EVALUATOR_ID", "deterministicki-v2"
        ), patch("pipeline.trainer_ai_state.EVALUATOR_ID", "deterministicki-v2"):
            start_new_event_revision(review, event)
            self.assertEqual(event["event_revision"], 3)
            self.assertEqual(event["ai_procene"][-1]["evaluator_id"], "deterministicki-v2")

    def test_injury_event_rejects_assessment_collections(self):
        injury = migrate_trainer_ai_payload(legacy_review_fixture())["events"][1]
        injury["trener_procene"] = []

        with self.assertRaisesRegex(ValueError, "povred"):
            validate_trainer_ai_event(injury)


if __name__ == "__main__":
    unittest.main()
