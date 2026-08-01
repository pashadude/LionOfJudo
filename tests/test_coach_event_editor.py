import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from coach_app.server import create_server
from pipeline.face_blur import BlurReport
from pipeline.trainer_ai_state import migrate_trainer_ai_payload


class CoachEventEditorTests(unittest.TestCase):
    def setUp(self):
        self.raw = tempfile.TemporaryDirectory()
        self.addCleanup(self.raw.cleanup)
        self.root = Path(self.raw.name)
        self.sony = self.root / "sony-source.mp4"
        self.iphone = self.root / "iphone-source.mov"
        self.sony.write_bytes(b"immutable sony")
        self.iphone.write_bytes(b"immutable iphone")
        frame_metrics = []
        for index in range(31):
            timestamp = 5.0 + index * 0.5
            frame_metrics.append(
                {
                    "frame_index": index,
                    "timestamp_s": timestamp,
                    "brzina_ulaska_norm": 0.1 + index * 0.01,
                    "rotacija_trupa_2d_dps": 5.0 + index,
                    "promena_visine_kukova_norm": -0.2 + index * 0.01,
                    "sirina_stava_norm": 0.5 + index * 0.01,
                    "intenzitet_pokreta_0_100": 10.0 + index,
                    "vidljivo": True,
                    "interpolirano": False,
                }
            )
        normal = {
            "event_id": "e-1",
            "sony_start_s": 8.0,
            "sony_end_s": 10.0,
            "iphone_start_s": 13.0,
            "iphone_end_s": 15.0,
            "predlog_tehnike": "O-soto-gari",
            "potvrdena_tehnika": "O-soto-gari",
            "glasovna_fraza": "o soto gari",
            "pouzdanost_glasa": 0.9,
            "ocena": 4,
            "napomena": "Sačuvana napomena.",
            "iskljuceno_iz_statistike": False,
            "status": "predlog",
        }
        injury = {
            "event_id": "povreda",
            "sony_start_s": 20.0,
            "sony_end_s": 21.0,
            "iphone_start_s": 25.0,
            "iphone_end_s": 26.0,
            "predlog_tehnike": None,
            "potvrdena_tehnika": None,
            "glasovna_fraza": None,
            "pouzdanost_glasa": 0.0,
            "prijavljen_povredni_dogadjaj": True,
            "iskljuceno_iz_statistike": True,
            "status": "povreda",
        }
        self.review_path = self.root / "review.json"
        self.review_path.write_text(
            json.dumps(
                {
                    "version": 2,
                    "session_id": "event-editor",
                    "sony_video": str(self.sony),
                    "iphone_video": str(self.iphone),
                    "sources": {
                        "sony": {"path": str(self.sony)},
                        "iphone": {"path": str(self.iphone)},
                    },
                    "sony_duration_s": 30.0,
                    "iphone_duration_s": 40.0,
                    "sony_fps": 30.0,
                    "iphone_fps": 30.0,
                    "anchors": [
                        {
                            "name": "pocetak",
                            "sony_s": 5.0,
                            "iphone_s": 10.0,
                            "user_confirmed": True,
                            "triple_tap_count": 3,
                        },
                        {
                            "name": "kontrola",
                            "sony_s": 15.0,
                            "iphone_s": 20.0,
                            "user_confirmed": True,
                            "triple_tap_count": 3,
                        },
                    ],
                    "time_map": {"slope": 1.0, "intercept": -5.0},
                    "injury_cutoff_s": 20.0,
                    "participants": {
                        "trainer_name": "Marko",
                        "wrestler_name": "Dusan",
                        "updated_at": "2026-08-01T12:00:00+00:00",
                    },
                    "sync_locked": True,
                    "frame_metrics": frame_metrics,
                    "events": [normal, injury],
                    "event_metrics": [copy.deepcopy(normal), copy.deepcopy(injury)],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def fake_cut(_source, start, end, output, **_kwargs):
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(str(float(end) - float(start)), encoding="ascii")
        return output

    @staticmethod
    def fake_probe(path):
        return float(Path(path).read_text(encoding="ascii"))

    def start_server(
        self, *, clip_exporter=None, media_probe=None, privacy_processor=None
    ):
        server = create_server(
            self.root,
            port=0,
            clip_exporter=clip_exporter or self.fake_cut,
            media_probe=media_probe or self.fake_probe,
            privacy_processor=privacy_processor or self.fake_privacy,
        )
        thread = server.start_in_thread()
        self.addCleanup(lambda: self.stop_server(server, thread))
        return server

    @staticmethod
    def fake_privacy(raw_path, output_path):
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(Path(raw_path).read_bytes())
        return BlurReport(
            total_frames=1,
            first_pass_candidates=1,
            second_pass_candidates=0,
            privacy_verified=True,
        )

    def upgrade_to_v3(self):
        review = json.loads(self.review_path.read_text(encoding="utf-8"))
        review["effective_analysis_fps"] = 2.0
        for camera, source in (("sony", self.sony), ("iphone", self.iphone)):
            review["sources"][camera].update(
                {
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "size": source.stat().st_size,
                }
            )
        review = migrate_trainer_ai_payload(review)
        review["event_metrics"] = copy.deepcopy(review["events"])
        self.review_path.write_text(
            json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return review

    @staticmethod
    def stop_server(server, thread):
        server.shutdown()
        thread.join(timeout=2)

    @staticmethod
    def request_json(server, path, *, method="GET", payload=None):
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            server.base_url + path,
            data=body,
            method=method,
            headers={"Content-Type": "application/json"} if body is not None else {},
        )
        with urlopen(request) as response:
            return response.status, json.loads(response.read().decode("utf-8"))

    def assert_http_error(self, code, server, path, *, method, payload=None):
        with self.assertRaises(HTTPError) as raised:
            self.request_json(server, path, method=method, payload=payload)
        self.assertEqual(raised.exception.code, code)
        return json.loads(raised.exception.read().decode("utf-8"))

    def test_create_normal_event_generates_both_clips_and_metrics_atomically(self):
        server = self.start_server()

        status, result = self.request_json(
            server,
            "/api/events",
            method="POST",
            payload={"sony_start_s": 6.0, "sony_end_s": 7.0},
        )

        self.assertEqual(status, 201)
        self.assertEqual(result["selected_event_id"], "e-coach-001")
        created = next(
            event for event in result["review"]["events"]
            if event["event_id"] == "e-coach-001"
        )
        self.assertEqual((created["iphone_start_s"], created["iphone_end_s"]), (11.0, 12.0))
        self.assertIsNotNone(created["brzina_ulaska_norm"])
        self.assertIsNotNone(created["intenzitet_pokreta_0_100"])
        self.assertTrue((self.root / "events" / "e-coach-001" / "sony.mp4").is_file())
        self.assertTrue((self.root / "events" / "e-coach-001" / "iphone.mp4").is_file())
        self.assertEqual(
            [event["event_id"] for event in result["review"]["event_metrics"]],
            [event["event_id"] for event in result["review"]["events"]],
        )
        rows = [
            row
            for row in result["review"]["derived_media_manifest"]
            if row["relative_path"].startswith("events/e-coach-001/")
        ]
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["privacy_verified"] is True for row in rows))

    def test_bounds_update_preserves_annotation_and_rebuilds_media(self):
        server = self.start_server()

        status, result = self.request_json(
            server,
            "/api/events/e-1/bounds",
            method="PUT",
            payload={"sony_start_s": 8.25, "sony_end_s": 9.75},
        )

        self.assertEqual(status, 200)
        event = next(item for item in result["review"]["events"] if item["event_id"] == "e-1")
        self.assertEqual((event["sony_start_s"], event["sony_end_s"]), (8.25, 9.75))
        self.assertEqual(event["potvrdena_tehnika"], "O-soto-gari")
        self.assertEqual(event["ocena"], 4)
        self.assertEqual(event["napomena"], "Sačuvana napomena.")
        self.assertEqual(self.fake_probe(self.root / "events" / "e-1" / "sony.mp4"), 1.5)

    def test_bounds_update_preserves_event_iphone_sync_offset_for_export(self):
        review = json.loads(self.review_path.read_text(encoding="utf-8"))
        review["events"][0]["iphone_sync_offset_s"] = 0.8
        review["event_metrics"] = copy.deepcopy(review["events"])
        self.review_path.write_text(json.dumps(review), encoding="utf-8")
        exports = []

        def recording_cut(source, start, end, output, **kwargs):
            exports.append((Path(source).name, float(start), float(end)))
            return self.fake_cut(source, start, end, output, **kwargs)

        server = self.start_server(clip_exporter=recording_cut)
        _, result = self.request_json(
            server,
            "/api/events/e-1/bounds",
            method="PUT",
            payload={"sony_start_s": 8.25, "sony_end_s": 9.75},
        )

        event = next(item for item in result["review"]["events"] if item["event_id"] == "e-1")
        self.assertEqual(event["iphone_sync_offset_s"], 0.8)
        self.assertIn((self.iphone.name, 14.05, 15.55), exports)

    def test_bounds_change_starts_new_assessment_round(self):
        legacy = self.upgrade_to_v3()
        legacy_bytes = self.review_path.read_bytes()
        server = self.start_server()
        assessment = {
            "status_vidljivosti": "dovoljno_vidljivo",
            "potvrdena_tehnika": "Tai-otoshi",
            "ocena": 4,
            "razlog": "Na 8.500 s rotacija počinje pre završetka ulaska.",
            "citirani_sony_trenuci_s": [8.5],
        }
        self.request_json(
            server,
            "/api/events/e-1/trainer-assessments",
            method="POST",
            payload=assessment,
        )
        self.request_json(
            server,
            "/api/events/e-1/ai-reveal",
            method="POST",
            payload={},
        )
        before_snapshot = server.trainer_ai_service.store.resolve_current()
        before_review = json.loads(before_snapshot.review_path.read_text(encoding="utf-8"))
        before_event = next(event for event in before_review["events"] if event["event_id"] == "e-1")
        first_assessment = json.dumps(
            before_event["trener_procene"][0], ensure_ascii=False, sort_keys=True
        )
        first_fingerprint = before_event["analysis_fingerprint"]

        status, result = self.request_json(
            server,
            "/api/events/e-1/bounds",
            method="PUT",
            payload={"sony_start_s": 8.25, "sony_end_s": 9.75},
        )

        self.assertEqual(status, 200)
        public_event = next(
            event for event in result["review"]["events"] if event["event_id"] == "e-1"
        )
        self.assertEqual(public_event["event_revision"], 2)
        self.assertNotEqual(public_event["analysis_fingerprint"], first_fingerprint)
        self.assertNotIn("ai_procene", public_event)
        self.assertNotIn("imu_eksperimentalno", public_event)
        after_snapshot = server.trainer_ai_service.store.resolve_current()
        self.assertNotEqual(after_snapshot.generation_id, before_snapshot.generation_id)
        after_review = json.loads(after_snapshot.review_path.read_text(encoding="utf-8"))
        after_event = next(event for event in after_review["events"] if event["event_id"] == "e-1")
        self.assertEqual(len(after_event["ai_procene"]), 2)
        self.assertEqual(len(after_event["trener_procene"]), 1)
        self.assertEqual(
            json.dumps(after_event["trener_procene"][0], ensure_ascii=False, sort_keys=True),
            first_assessment,
        )
        self.assertIsNone(after_event["ai_procene"][-1]["ai_otkriven_u"])
        self.assertIsNone(after_event["aktivna_trener_revizija"])
        self.assertIsNone(after_event["aktivni_duel"])
        self.assertEqual(self.fake_probe(after_snapshot.root / "events" / "e-1" / "sony.mp4"), 1.5)
        self.assertEqual(self.review_path.read_bytes(), legacy_bytes)
        self.assertEqual(legacy["events"][0]["event_revision"], 1)

    def test_split_then_merge_restores_bounds_and_preserves_left_annotation(self):
        server = self.start_server()
        _, split = self.request_json(
            server,
            "/api/events/e-1/split",
            method="POST",
            payload={"sony_split_s": 9.0},
        )
        right_id = split["created_event_id"]
        left = next(event for event in split["review"]["events"] if event["event_id"] == "e-1")
        right = next(event for event in split["review"]["events"] if event["event_id"] == right_id)
        self.assertEqual((left["sony_start_s"], left["sony_end_s"]), (8.0, 9.0))
        self.assertEqual((right["sony_start_s"], right["sony_end_s"]), (9.0, 10.0))
        self.assertEqual(left["potvrdena_tehnika"], "O-soto-gari")
        self.assertIsNone(right["potvrdena_tehnika"])
        self.assertIsNone(right["ocena"])
        self.assertIsNone(right["napomena"])

        status, merged = self.request_json(
            server,
            "/api/events/merge",
            method="POST",
            payload={"event_ids": ["e-1", right_id]},
        )

        self.assertEqual(status, 200)
        event = next(item for item in merged["review"]["events"] if item["event_id"] == "e-1")
        self.assertEqual((event["sony_start_s"], event["sony_end_s"]), (8.0, 10.0))
        self.assertEqual(event["potvrdena_tehnika"], "O-soto-gari")
        self.assertNotIn(right_id, [item["event_id"] for item in merged["review"]["events"]])

    def test_v3_split_keeps_history_only_on_left_and_resets_both_rounds(self):
        self.upgrade_to_v3()
        server = self.start_server()
        self.request_json(
            server,
            "/api/events/e-1/trainer-assessments",
            method="POST",
            payload={
                "status_vidljivosti": "dovoljno_vidljivo",
                "potvrdena_tehnika": "Tai-otoshi",
                "ocena": 4,
                "razlog": "Na 8.500 s ulazak prethodi rotaciji.",
                "citirani_sony_trenuci_s": [8.5],
            },
        )

        status, result = self.request_json(
            server,
            "/api/events/e-1/split",
            method="POST",
            payload={"sony_split_s": 9.0},
        )

        self.assertEqual(status, 200)
        snapshot = server.trainer_ai_service.store.resolve_current()
        internal = json.loads(snapshot.review_path.read_text(encoding="utf-8"))
        left = next(event for event in internal["events"] if event["event_id"] == "e-1")
        right = next(
            event
            for event in internal["events"]
            if event["event_id"] == result["created_event_id"]
        )
        self.assertEqual(left["event_revision"], 2)
        self.assertEqual(len(left["trener_procene"]), 1)
        self.assertEqual(len(left["ai_procene"]), 2)
        self.assertIsNone(left["aktivna_trener_revizija"])
        self.assertIsNone(left["aktivni_duel"])
        self.assertEqual(right["event_revision"], 1)
        self.assertEqual(right["trener_procene"], [])
        self.assertEqual(right["procene_ai_predloga"], [])
        self.assertEqual(len(right["ai_procene"]), 1)
        self.assertIsNone(right["potvrdena_tehnika"])
        self.assertIsNone(right["ocena"])

    def test_v3_merge_is_rejected_after_either_event_has_trainer_history(self):
        self.upgrade_to_v3()
        server = self.start_server()
        _, created = self.request_json(
            server,
            "/api/events",
            method="POST",
            payload={"sony_start_s": 10.5, "sony_end_s": 11.5},
        )
        created_id = created["created_event_id"]
        created_event = next(
            event for event in created["review"]["events"] if event["event_id"] == created_id
        )
        self.assertEqual(created_event["event_revision"], 1)
        self.assertEqual(created_event["trener_procene"], [])
        self.request_json(
            server,
            "/api/events/e-1/trainer-assessments",
            method="POST",
            payload={
                "status_vidljivosti": "dovoljno_vidljivo",
                "potvrdena_tehnika": "Tai-otoshi",
                "ocena": 4,
                "razlog": "Na 8.500 s ulazak prethodi rotaciji.",
                "citirani_sony_trenuci_s": [8.5],
            },
        )
        before = server.trainer_ai_service.store.resolve_current().generation_id

        error = self.assert_http_error(
            409,
            server,
            "/api/events/merge",
            method="POST",
            payload={"event_ids": ["e-1", created_id]},
        )

        self.assertIn("zaključanim", error["error"])
        self.assertEqual(
            server.trainer_ai_service.store.resolve_current().generation_id, before
        )

    def test_delete_removes_normal_event_and_ledgers_its_annotation(self):
        server = self.start_server()

        status, result = self.request_json(server, "/api/events/e-1", method="DELETE")

        self.assertEqual(status, 200)
        self.assertNotIn("e-1", [event["event_id"] for event in result["review"]["events"]])
        orphan = result["review"]["orphaned_annotations"][-1]
        self.assertEqual(orphan["source_event_id"], "e-1")
        self.assertEqual(orphan["reason"], "obrisan_dogadjaj")
        self.assertEqual(orphan["potvrdena_tehnika"], "O-soto-gari")
        self.assertFalse((self.root / "events" / "e-1").exists())

    def test_delete_rejects_event_with_trainer_history(self):
        self.upgrade_to_v3()
        server = self.start_server()
        review = server.trainer_ai_service.load_review()
        event = review["events"][0]
        event["trener_procene"] = [{
            "revizija": 1,
            "faza": "pre_ai",
            "event_revision": event["event_revision"],
            "analysis_fingerprint": event["analysis_fingerprint"],
            "status_vidljivosti": "dovoljno_vidljivo",
            "potvrdena_tehnika": "Tai-otoshi",
            "ocena": 4,
            "razlog": "Na 8.500 s ulazak je vidljiv.",
            "citirani_sony_trenuci_s": [8.5],
            "zakljucano_u": "2026-08-01T12:00:00+00:00",
        }]
        server.trainer_ai_service.activate_review(review)

        error = self.assert_http_error(409, server, "/api/events/e-1", method="DELETE")

        self.assertIn("zaključan", error["error"])

    def test_delete_rejects_source_video_inside_managed_events_directory(self):
        embedded_source = self.root / "events" / "e-1" / "sony.mp4"
        embedded_source.parent.mkdir(parents=True)
        embedded_source.write_bytes(b"embedded immutable sony")
        review = json.loads(self.review_path.read_text(encoding="utf-8"))
        review["sony_video"] = str(embedded_source)
        review["sources"]["sony"]["path"] = str(embedded_source)
        self.review_path.write_text(json.dumps(review), encoding="utf-8")
        before = self.review_path.read_bytes()
        server = self.start_server()

        error = self.assert_http_error(
            422,
            server,
            "/api/events/e-1",
            method="DELETE",
        )

        self.assertIn("izvorni", error["error"].lower())
        self.assertEqual(embedded_source.read_bytes(), b"embedded immutable sony")
        self.assertEqual(self.review_path.read_bytes(), before)

    def test_invalid_or_overlapping_ranges_leave_review_unchanged(self):
        server = self.start_server()
        before = self.review_path.read_bytes()
        for payload in (
            {"sony_start_s": 4.0, "sony_end_s": 6.0},
            {"sony_start_s": 19.5, "sony_end_s": 20.5},
            {"sony_start_s": 9.0, "sony_end_s": 11.0},
            {"sony_start_s": 7.0, "sony_end_s": 7.0},
        ):
            with self.subTest(payload=payload):
                self.assert_http_error(400, server, "/api/events", method="POST", payload=payload)
                self.assertEqual(self.review_path.read_bytes(), before)

    def test_injury_cannot_be_adjusted_split_merged_or_deleted(self):
        server = self.start_server()
        cases = (
            ("/api/events/povreda/bounds", "PUT", {"sony_start_s": 19.5, "sony_end_s": 20.5}),
            ("/api/events/povreda/split", "POST", {"sony_split_s": 20.5}),
            ("/api/events/merge", "POST", {"event_ids": ["e-1", "povreda"]}),
            ("/api/events/povreda", "DELETE", None),
        )
        for path, method, payload in cases:
            with self.subTest(path=path):
                error = self.assert_http_error(400, server, path, method=method, payload=payload)
                self.assertIn("povred", error["error"].lower())

    def test_media_duration_failure_rolls_back_json_and_staged_files(self):
        server = self.start_server(media_probe=lambda _path: 99.0)
        before = self.review_path.read_bytes()

        error = self.assert_http_error(
            422,
            server,
            "/api/events",
            method="POST",
            payload={"sony_start_s": 6.0, "sony_end_s": 7.0},
        )

        self.assertIn("medij", error["error"].lower())
        self.assertEqual(self.review_path.read_bytes(), before)
        self.assertFalse((self.root / "events" / "e-coach-001").exists())
        events_dir = self.root / "events"
        self.assertFalse(events_dir.exists() and any(path.name.startswith(".txn-") for path in events_dir.iterdir()))

    def test_v3_privacy_failure_keeps_current_generation_and_media(self):
        self.upgrade_to_v3()

        def rejected_privacy(_raw, _output):
            return BlurReport(
                total_frames=1,
                first_pass_candidates=1,
                second_pass_candidates=1,
                privacy_verified=False,
                failure_reason="lice je ostalo vidljivo",
            )

        server = self.start_server(privacy_processor=rejected_privacy)
        self.request_json(
            server,
            "/api/events/e-1/trainer-assessments",
            method="POST",
            payload={
                "status_vidljivosti": "dovoljno_vidljivo",
                "potvrdena_tehnika": "Tai-otoshi",
                "ocena": 4,
                "razlog": "Na 8.500 s ulazak prethodi rotaciji.",
                "citirani_sony_trenuci_s": [8.5],
            },
        )
        before = server.trainer_ai_service.store.resolve_current()
        before_review = before.review_path.read_bytes()

        error = self.assert_http_error(
            422,
            server,
            "/api/events/e-1/bounds",
            method="PUT",
            payload={"sony_start_s": 8.25, "sony_end_s": 9.75},
        )

        self.assertIn("lice je ostalo vidljivo", error["error"])
        after = server.trainer_ai_service.store.resolve_current()
        self.assertEqual(after.generation_id, before.generation_id)
        self.assertEqual(after.review_path.read_bytes(), before_review)


if __name__ == "__main__":
    unittest.main()
