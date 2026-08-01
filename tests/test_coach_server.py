import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from coach_app.server import create_server, save_annotation


class CoachServerTests(unittest.TestCase):
    def setUp(self):
        self.raw = tempfile.TemporaryDirectory()
        self.root = Path(self.raw.name)
        (self.root / "media").mkdir()
        (self.root / "media" / "clip.mp4").write_bytes(b"0123456789")
        (self.root / "media" / "session_side_by_side.mp4").write_bytes(b"0123456789")
        (self.root / "media" / "unlisted.mp4").write_bytes(b"unlisted")
        (self.root / "events" / "e-1").mkdir(parents=True)
        (self.root / "events" / "e-1" / "sony.mp4").write_bytes(b"raw-event")
        (self.root / "previews").mkdir()
        (self.root / "previews" / "anchor_01_sony.mp4").write_bytes(b"raw-preview")
        self.outside = self.root.parent / f"coach-outside-{os.getpid()}"
        self.outside.write_bytes(b"outside")
        self.review_path = self.root / "review.json"
        self.review_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "session_id": "demo",
                    "sony_video": "media/clip.mp4",
                    "iphone_video": "media/clip.mp4",
                    "sony_duration_s": 30.0,
                    "iphone_duration_s": 40.0,
                    "sony_fps": 30.0,
                    "iphone_fps": 60.0,
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
                    "sync_locked": True,
                    "derived_media_manifest": [
                        {
                            "relative_path": "session_side_by_side.mp4",
                            "media_type": "side_by_side",
                            "total_frames": 10,
                            "first_pass_candidates": 2,
                            "second_pass_candidates": 0,
                            "privacy_verified": True,
                            "failure_reason": None,
                        }
                    ],
                    "frame_metrics": [
                        {
                            "timestamp_s": 0.0,
                            "brzina_ulaska_norm": 0.2,
                            "rotacija_trupa_2d_dps": 4.0,
                            "promena_visine_kukova_norm": 0.3,
                            "vreme_oporavka_s": 0.4,
                            "intenzitet_pokreta_0_100": 10.0,
                        },
                        {
                            "timestamp_s": 30.0,
                            "brzina_ulaska_norm": 0.8,
                            "rotacija_trupa_2d_dps": 8.0,
                            "promena_visine_kukova_norm": 0.6,
                            "vreme_oporavka_s": 0.8,
                            "intenzitet_pokreta_0_100": 80.0,
                        },
                    ],
                    "events": [
                        {
                            "event_id": "e-1",
                            "sony_start_s": 8.0,
                            "sony_end_s": 10.0,
                            "iphone_start_s": 13.0,
                            "iphone_end_s": 15.0,
                            "predlog_tehnike": "O-soto-gari",
                            "potvrdena_tehnika": "",
                            "glasovna_fraza": "o soto gari",
                            "pouzdanost_glasa": 0.9,
                            "brzina_ulaska_norm": 0.5,
                            "rotacija_trupa_2d_dps": 7.0,
                            "promena_visine_kukova_norm": 0.4,
                            "vreme_oporavka_s": 0.7,
                            "intenzitet_pokreta_0_100": 55.0,
                            "ocena": 3,
                            "napomena": "",
                            "iskljuceno_iz_statistike": False,
                        },
                        {
                            "event_id": "povreda",
                            "sony_start_s": 19.0,
                            "sony_end_s": 20.0,
                            "prijavljen_povredni_dogadjaj": True,
                            "iskljuceno_iz_statistike": True,
                            "status": "povreda",
                            "vidljivost": "nedovoljno vidljivo",
                        },
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        try:
            (self.root / "media" / "symlink.mp4").symlink_to(self.outside)
            self.has_symlink = True
        except (OSError, NotImplementedError):
            self.has_symlink = False

    def tearDown(self):
        self.outside.unlink(missing_ok=True)
        self.raw.cleanup()

    def start_server(self):
        server = create_server(self.root, port=0)
        thread = server.start_in_thread()
        self.addCleanup(lambda: self._stop(server, thread))
        return server

    @staticmethod
    def _stop(server, thread):
        server.shutdown()
        thread.join(timeout=2)

    @staticmethod
    def read_json(url, *, method="GET", body=None):
        request = Request(
            url,
            data=body,
            method=method,
            headers={"Content-Type": "application/json"} if body else {},
        )
        with urlopen(request) as response:
            return response.status, json.loads(response.read().decode("utf-8"))

    def test_server_lifecycle_and_session_endpoints(self):
        server = self.start_server()
        self.assertEqual(server.httpd.server_address[0], "127.0.0.1")
        self.assertTrue((self.root / "izvestaj.csv").exists())
        self.assertTrue((self.root / "izvestaj.md").exists())
        status, session = self.read_json(server.base_url + "/api/session")
        self.assertEqual(status, 200)
        self.assertEqual(session["session_id"], "demo")
        status, event = self.read_json(server.base_url + "/api/events/e-1")
        self.assertEqual(status, 200)
        self.assertEqual(event["event_id"], "e-1")
        for report in ("/izvestaj.csv", "/izvestaj.md"):
            with urlopen(server.base_url + report) as response:
                self.assertEqual(response.status, 200)
                self.assertTrue(response.read())

    def test_annotation_persists_and_regenerates_csv_and_markdown(self):
        server = self.start_server()
        body = json.dumps(
            {
                "potvrdena_tehnika": "O-soto-gari",
                "ocena": 4,
                "napomena": "Stabilan, ulaz | kukovi mogu niže.",
            },
            ensure_ascii=False,
        ).encode("utf-8")
        status, saved = self.read_json(
            server.base_url + "/api/events/e-1/annotation",
            method="PUT",
            body=body,
        )
        self.assertEqual(status, 200)
        self.assertEqual(saved["potvrdena_tehnika"], "O-soto-gari")
        review = json.loads(self.review_path.read_text(encoding="utf-8"))
        self.assertEqual(review["events"][0]["ocena"], 4)
        self.assertFalse((self.root / "review.json.tmp").exists())
        csv_path = self.root / "izvestaj.csv"
        markdown_path = self.root / "izvestaj.md"
        self.assertTrue(csv_path.exists())
        self.assertTrue(markdown_path.exists())
        with csv_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["Potvrđena tehnika"], "O-soto-gari")
        self.assertIn('"Stabilan, ulaz | kukovi mogu niže."', csv_path.read_text(encoding="utf-8"))
        markdown = markdown_path.read_text(encoding="utf-8")
        self.assertIn("Potvrđena tehnika", markdown)
        self.assertIn("Stabilan, ulaz \\| kukovi", markdown)
        self.assertIn("Nedovoljno vidljivo", markdown)

    def test_annotation_allows_unscored_trainer_confirmation(self):
        server = self.start_server()
        body = json.dumps(
            {
                "potvrdena_tehnika": "Morote-seoi-nage",
                "ocena": None,
                "napomena": "Naziv potvrdio trener; ocena čeka trenera.",
            },
            ensure_ascii=False,
        ).encode("utf-8")

        status, saved = self.read_json(
            server.base_url + "/api/events/e-1/annotation",
            method="PUT",
            body=body,
        )

        self.assertEqual(status, 200)
        self.assertEqual(saved["potvrdena_tehnika"], "Morote-seoi-nage")
        self.assertIsNone(saved["ocena"])
        self.assertEqual(saved["status"], "trener")
        with (self.root / "izvestaj.csv").open(encoding="utf-8", newline="") as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual(row["Ocena"], "")

    def test_save_annotation_is_strict_and_atomic(self):
        payload = {
            "potvrdena_tehnika": "Uki-goshi",
            "ocena": 5,
            "napomena": "Čist ulaz.",
        }
        saved = save_annotation(self.review_path, "e-1", payload)
        self.assertEqual(saved["ocena"], 5)
        self.assertFalse((self.root / "review.json.tmp").exists())
        with self.assertRaises(ValueError):
            save_annotation(self.review_path, "missing", payload)
        with self.assertRaises(ValueError):
            save_annotation(self.review_path, "e-1", {**payload, "nepoznato": True})

    def test_annotation_rejects_bad_types_ranges_and_lengths(self):
        server = self.start_server()
        cases = [
            {"potvrdena_tehnika": "x", "ocena": True, "napomena": ""},
            {"potvrdena_tehnika": "x", "ocena": 0, "napomena": ""},
            {"potvrdena_tehnika": 4, "ocena": 3, "napomena": ""},
            {"potvrdena_tehnika": "x", "ocena": 3, "napomena": "x" * 2001},
            {"potvrdena_tehnika": "x", "ocena": 3},
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                request = Request(
                    server.base_url + "/api/events/e-1/annotation",
                    data=json.dumps(payload).encode("utf-8"),
                    method="PUT",
                    headers={"Content-Type": "application/json"},
                )
                with self.assertRaises(HTTPError) as raised:
                    urlopen(request)
                self.assertEqual(raised.exception.code, 400)

    def test_injury_annotation_rejects_normal_score_and_technique_controls(self):
        server = self.start_server()
        body = json.dumps(
            {
                "potvrdena_tehnika": "O-soto-gari",
                "ocena": 4,
                "napomena": "Prijavljen događaj.",
            }
        ).encode()
        request = Request(
            server.base_url + "/api/events/povreda/annotation",
            data=body,
            method="PUT",
            headers={"Content-Type": "application/json"},
        )
        with self.assertRaises(HTTPError) as raised:
            urlopen(request)
        self.assertEqual(raised.exception.code, 400)

    def test_injury_annotation_put_rejects_read_only_payload(self):
        server = self.start_server()
        body = json.dumps(
            {"potvrdena_tehnika": "", "ocena": None, "napomena": ""}
        ).encode()
        request = Request(
            server.base_url + "/api/events/povreda/annotation",
            data=body,
            method="PUT",
            headers={"Content-Type": "application/json"},
        )
        with self.assertRaises(HTTPError) as raised:
            urlopen(request)
        self.assertEqual(raised.exception.code, 400)

    def test_imported_session_sync_is_locked_with_serbian_conflict(self):
        server = self.start_server()
        anchors = [
            {
                "name": "pocetak",
                "sony_s": 6.0,
                "iphone_s": 11.0,
                "user_confirmed": True,
                "triple_tap_count": 3,
            },
            {
                "name": "kontrola",
                "sony_s": 16.0,
                "iphone_s": 21.0,
                "user_confirmed": True,
                "triple_tap_count": 3,
            },
        ]
        before = self.review_path.read_bytes()
        with self.assertRaises(HTTPError) as raised:
            self.read_json(
                server.base_url + "/api/session/sync",
                method="POST",
                body=json.dumps({"anchors": anchors, "injury_cutoff_s": 20.0}).encode(),
            )
        self.assertEqual(raised.exception.code, 409)
        error = json.loads(raised.exception.read().decode("utf-8"))["error"]
        self.assertIn("zaključana", error)
        self.assertIn("novi uvoz", error)
        self.assertEqual(self.review_path.read_bytes(), before)

    def test_minimal_preimport_session_can_update_sync(self):
        review = json.loads(self.review_path.read_text(encoding="utf-8"))
        review["events"] = []
        review["event_metrics"] = []
        review["frame_metrics"] = []
        review["sync_locked"] = False
        review["derived_media_manifest"] = []
        self.review_path.write_text(json.dumps(review), encoding="utf-8")
        (self.root / "media" / "session_side_by_side.mp4").unlink()
        (self.root / "events" / "e-1" / "sony.mp4").unlink()
        (self.root / "events" / "e-1").rmdir()
        (self.root / "events").rmdir()
        (self.root / "previews" / "anchor_01_sony.mp4").unlink()
        (self.root / "previews").rmdir()
        server = self.start_server()
        anchors = [
            {
                "name": "pocetak",
                "sony_s": 6.0,
                "iphone_s": 11.0,
                "user_confirmed": True,
                "triple_tap_count": 3,
            },
            {
                "name": "kontrola",
                "sony_s": 16.0,
                "iphone_s": 21.0,
                "user_confirmed": True,
                "triple_tap_count": 3,
            },
        ]

        status, synced = self.read_json(
            server.base_url + "/api/session/sync",
            method="POST",
            body=json.dumps({"anchors": anchors, "injury_cutoff_s": 20.0}).encode(),
        )

        self.assertEqual(status, 200)
        self.assertAlmostEqual(synced["time_map"]["intercept"], -5.0)
        unconfirmed = [dict(anchors[0]), dict(anchors[1], triple_tap_count=2)]
        with self.assertRaises(HTTPError) as raised:
            self.read_json(
                server.base_url + "/api/session/sync",
                method="POST",
                body=json.dumps({"anchors": unconfirmed, "injury_cutoff_s": 20.0}).encode(),
            )
        self.assertEqual(raised.exception.code, 400)

    def test_containment_rejects_encoded_traversal_and_symlink_escape(self):
        server = self.start_server()
        for path in (
            "/media/%2e%2e/%2e%2e/etc/passwd",
            "/media/..%2freview.json",
            "/static/%2e%2e/server.py",
        ):
            with self.subTest(path=path):
                with self.assertRaises(HTTPError) as raised:
                    urlopen(server.base_url + path)
            self.assertEqual(raised.exception.code, 404)
        if self.has_symlink:
            with self.assertRaises(HTTPError) as raised:
                urlopen(server.base_url + "/media/symlink.mp4")
            self.assertEqual(raised.exception.code, 404)

    def test_media_range_and_mime_behavior_support_html_video(self):
        server = self.start_server()
        request = Request(
            server.base_url + "/media/session_side_by_side.mp4",
            headers={"Range": "bytes=2-5"},
        )
        with urlopen(request) as response:
            self.assertEqual(response.status, 206)
            self.assertEqual(response.read(), b"2345")
            self.assertEqual(response.headers["Content-Type"], "video/mp4")
            self.assertEqual(response.headers["Content-Range"], "bytes 2-5/10")
            self.assertEqual(response.headers["Accept-Ranges"], "bytes")

    def test_only_manifest_verified_media_is_served(self):
        server = self.start_server()

        with urlopen(server.base_url + "/media/session_side_by_side.mp4") as response:
            self.assertEqual(response.read(), b"0123456789")
        for path in (
            "/events/e-1/sony.mp4",
            "/previews/anchor_01_sony.mp4",
            "/media/unlisted.mp4",
            "/media/events/e-1/sony.mp4",
            "/media/previews/anchor_01_sony.mp4",
        ):
            with self.subTest(path=path):
                with self.assertRaises(HTTPError) as raised:
                    urlopen(server.base_url + path)
                self.assertEqual(raised.exception.code, 404)

        review = json.loads(self.review_path.read_text(encoding="utf-8"))
        review["derived_media_manifest"][0]["privacy_verified"] = False
        self.review_path.write_text(json.dumps(review), encoding="utf-8")
        with self.assertRaises(HTTPError) as raised:
            urlopen(server.base_url + "/media/session_side_by_side.mp4")
        self.assertEqual(raised.exception.code, 404)

    def test_manifest_cannot_publish_arbitrary_analysis_file(self):
        analysis = self.root / "analysis"
        analysis.mkdir(exist_ok=True)
        (analysis / "fake.mp4").write_bytes(b"not-private")
        review = json.loads(self.review_path.read_text(encoding="utf-8"))
        review["derived_media_manifest"].append(
            {
                "relative_path": "analysis/fake.mp4",
                "media_type": "side_by_side",
                "total_frames": 1,
                "first_pass_candidates": 0,
                "second_pass_candidates": 0,
                "privacy_verified": True,
                "failure_reason": None,
            }
        )
        self.review_path.write_text(json.dumps(review), encoding="utf-8")
        server = self.start_server()

        with self.assertRaises(HTTPError) as raised:
            urlopen(server.base_url + "/media/analysis/fake.mp4")

        self.assertEqual(raised.exception.code, 404)

    def test_static_ui_contains_serbian_contract_and_chart_hooks(self):
        server = self.start_server()
        with urlopen(server.base_url + "/") as response:
            html = response.read().decode("utf-8")
        self.assertIn("Događaji", html)
        self.assertIn("Sinhronizacija", html)
        for label in (
            "Potvrđena tehnika",
            "Trenerova ocena 1–5",
            "Razlog trenera",
            "Dodaj trenutnu sekundu",
            "Zaključaj procenu",
            "Otkrij AI izazov",
            "AI činjenice",
            "IMU merenje (eksperimentalno)",
            "Prototip v1. Moguća velika greška.",
            "Slažem se",
            "Delimično",
            "Ne slažem se",
            "Podeli",
            "Spoji",
            "Obriši",
            "Izvezi izveštaj",
            "Prijavljen povredni događaj",
            "Nedovoljno vidljivo",
        ):
            self.assertIn(label, html)
        self.assertEqual(html.count("<canvas"), 5)
        self.assertIn('id="lock-assessment-button" class="button primary" type="submit" disabled', html)
        for asset in ("/static/app.js", "/static/styles.css"):
            with urlopen(server.base_url + asset) as response:
                self.assertEqual(response.status, 200)
                self.assertTrue(response.read())

    def test_static_ui_exposes_csv_and_markdown_report_downloads(self):
        server = self.start_server()
        with urlopen(server.base_url + "/") as response:
            html = response.read().decode("utf-8")

        self.assertIn('href="/izvestaj.csv"', html)
        self.assertIn('href="/izvestaj.md"', html)

    def test_static_ui_contract_uses_global_cursor_and_requires_persisted_fps(self):
        app_js = (Path(__file__).parents[1] / "coach_app" / "static" / "app.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("function localTimesForGlobal", app_js)
        self.assertIn("globalSonyTime - Number(event.sony_start_s || 0)", app_js)
        self.assertIn("iphoneTime(globalSonyTime) - Number(event.iphone_start_s || 0)", app_js)
        self.assertIn("function globalSonyTimeForLocal", app_js)
        self.assertIn("function sonyFps", app_js)
        self.assertNotIn("|| 30", app_js)
        self.assertIn("FPS Sony nije dostupan", app_js)
        self.assertIn("event.analysis_fingerprint || event.event_revision", app_js)
        self.assertIn('const MEDIA_CODEC_VERSION = "h264-v1"', app_js)
        self.assertIn("`${mediaVersionBase}:${MEDIA_CODEC_VERSION}`", app_js)
        self.assertIn("encodeURIComponent(String(mediaVersion))", app_js)

    def test_playback_sync_avoids_repeated_iphone_seeks(self):
        static = Path(__file__).parents[1] / "coach_app" / "static"
        app_js = (static / "app.js").read_text(
            encoding="utf-8"
        )
        html = (static / "index.html").read_text(encoding="utf-8")

        self.assertIn('src="/static/app.js?v=h264-media-1"', html)
        self.assertIn("function correctIphonePlayback", app_js)
        self.assertIn("Promise.all([sony.play(), iphone.play()])", app_js)
        self.assertIn("iphone.playbackRate = clamp", app_js)
        self.assertEqual(app_js.count("iphone.currentTime = times.iphoneLocalTime;"), 1)
        self.assertNotIn(
            'sony.addEventListener("play", () => { if (iphone.paused)',
            app_js,
        )

    def test_injury_editor_state_disables_trainer_controls_and_reenables_normal_events(self):
        app_js = (Path(__file__).parents[1] / "coach_app" / "static" / "app.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("function setEditorDisabled", app_js)
        self.assertIn('$("#trainer-reason").disabled = disabled;', app_js)
        self.assertIn('$("#lock-assessment-button").disabled = disabled;', app_js)
        self.assertIn("setEditorDisabled(true);", app_js)
        self.assertIn("setEditorDisabled(disabled);", app_js)

    def test_static_ui_exposes_trainer_first_ai_duel_contract(self):
        static = Path(__file__).parents[1] / "coach_app" / "static"
        html = (static / "index.html").read_text(encoding="utf-8")
        app_js = (static / "app.js").read_text(encoding="utf-8")

        for control_id in (
            "confirmed-technique",
            "trainer-reason",
            "add-current-second",
            "lock-assessment-button",
            "reveal-ai-button",
            "ai-duel",
            "duel-delta",
            "system-facts",
            "imu-panel",
            "feedback-reason",
            "save-feedback-button",
        ):
            self.assertIn(f'id="{control_id}"', html)
        self.assertEqual(html.count('name="trainer-score"'), 5)
        self.assertEqual(html.count('name="ai-relation"'), 3)
        self.assertEqual(html.count('class="imu-value"'), 8)
        self.assertIn("AI odstupa za ${delta} poena. Odbrani procenu.", app_js)
        self.assertIn("/trainer-assessments`", app_js)
        self.assertIn("/ai-reveal`", app_js)
        self.assertIn("/ai-feedback`", app_js)

    def test_duel_score_conversion_preserves_null_as_unscored(self):
        app_js = (Path(__file__).parents[1] / "coach_app" / "static" / "app.js").read_text(
            encoding="utf-8"
        )

        self.assertIn("function optionalScore", app_js)
        self.assertIn("if (value == null || value === \"\") return null;", app_js)
        self.assertIn("const trainerScore = optionalScore(trainer?.ocena);", app_js)
        self.assertIn("const aiScore = optionalScore(ai.predlozena_ocena);", app_js)
        self.assertIn('$("#duel-delta").textContent = "AI nema dovoljno podataka.";', app_js)

    def test_injury_selection_keeps_event_creation_and_draft_bounds_available(self):
        app_js = (Path(__file__).parents[1] / "coach_app" / "static" / "app.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("function normalEventDraftBounds", app_js)
        self.assertIn("const canEditSelectedNormal = Boolean(event && !injury(event));", app_js)
        self.assertIn("const canCreateEvent = Boolean(normalEventDraftBounds());", app_js)
        self.assertIn('$("#event-start").disabled = !(canEditSelectedNormal || canCreateEvent);', app_js)
        self.assertIn('$("#event-end").disabled = !(canEditSelectedNormal || canCreateEvent);', app_js)
        self.assertIn('$("#create-event-button").disabled = !canCreateEvent;', app_js)
        self.assertIn('$("#update-bounds-button").disabled = !canEditSelectedNormal;', app_js)
        self.assertIn('if (injury(event)) populateDraftEventBounds();', app_js)

    def test_normal_selection_keeps_bounds_editable_when_no_creation_gap_exists(self):
        app_js = (Path(__file__).parents[1] / "coach_app" / "static" / "app.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("const canEditSelectedNormal = Boolean(event && !injury(event));", app_js)
        self.assertIn('$("#event-start").disabled = !(canEditSelectedNormal || canCreateEvent);', app_js)
        self.assertIn('$("#event-end").disabled = !(canEditSelectedNormal || canCreateEvent);', app_js)
        self.assertIn('$("#update-bounds-button").disabled = !canEditSelectedNormal;', app_js)
        self.assertIn('$("#create-event-button").disabled = !canCreateEvent;', app_js)

    def test_event_creation_draft_is_clamped_to_confirmed_anchor_and_injury_cutoff(self):
        app_js = (Path(__file__).parents[1] / "coach_app" / "static" / "app.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("function firstConfirmedSonyAnchor", app_js)
        self.assertIn("Math.min(cutoff, cursor)", app_js)
        self.assertIn("Math.max(firstAnchor,", app_js)
        self.assertIn("end - start > MIN_EVENT_SPAN", app_js)
        self.assertIn("Nema dostupnog normalnog intervala", app_js)
        self.assertIn("if (!canCreateEvent && event && injury(event))", app_js)

    def test_confirmed_anchor_uses_minimum_valid_sony_timestamp(self):
        app_js = (Path(__file__).parents[1] / "coach_app" / "static" / "app.js").read_text(
            encoding="utf-8"
        )
        start = app_js.index("function firstConfirmedSonyAnchor")
        end = app_js.index("function normalEventDraftBounds", start)
        anchor_helper = app_js[start:end]
        self.assertIn(".filter((item) => item?.user_confirmed === true)", anchor_helper)
        self.assertIn(".map((item) => Number(item?.sony_s))", anchor_helper)
        self.assertIn(".filter((sonyTime) => Number.isFinite(sonyTime))", anchor_helper)
        self.assertIn("Math.min(...confirmedSonyTimes)", anchor_helper)
        self.assertIn("confirmedSonyTimes.length ?", anchor_helper)

    def test_ui_uses_canonical_event_series_and_complete_correction_controls(self):
        static = Path(__file__).parents[1] / "coach_app" / "static"
        html = (static / "index.html").read_text(encoding="utf-8")
        app_js = (static / "app.js").read_text(encoding="utf-8")

        self.assertIn('content="width=device-width, initial-scale=1"', html)
        expected_series = {
            "brzina_ulaska_norm": "Brzina ulaska",
            "rotacija_trupa_2d_dps": "Rotacija trupa (2D)",
            "promena_visine_kukova_norm": "Visina kukova",
            "sirina_stava_norm": "Širina stava",
            "intenzitet_pokreta_0_100": "Intenzitet pokreta",
        }
        for key, label in expected_series.items():
            self.assertIn(f'data-metric="{key}"', html)
            self.assertIn(label, html)
        self.assertNotIn('data-metric="vreme_oporavka_s"', html)
        self.assertNotIn("<h3>Stabilnost</h3>", html)
        for control_id, label in (
            ("event-start", "Početak"),
            ("event-end", "Kraj"),
            ("update-bounds-button", "Primeni granice"),
            ("create-event-button", "Novi događaj"),
            ("split-button", "Podeli"),
            ("merge-button", "Spoji"),
            ("delete-button", "Obriši"),
        ):
            self.assertIn(f'id="{control_id}"', html)
            self.assertIn(label, html)
        self.assertIn("function updateCorrectionControls", app_js)
        self.assertIn("function updateSyncLock", app_js)
        self.assertIn("review.sync_locked", app_js)
        self.assertIn('mutateReview("/api/events", "POST"', app_js)
        self.assertIn('`/api/events/${encodeURIComponent(event.event_id)}/bounds`', app_js)
        self.assertIn('`/api/events/${encodeURIComponent(event.event_id)}/split`', app_js)
        self.assertIn('fetch("/api/events/merge"', app_js)
        self.assertIn('"DELETE",', app_js)
        self.assertIn("selectedFrameSamples", app_js)

    def test_mobile_css_has_bounded_single_column_media_and_controls(self):
        css = (Path(__file__).parents[1] / "coach_app" / "static" / "styles.css").read_text(
            encoding="utf-8"
        )

        self.assertIn("overflow-x: hidden", css)
        self.assertIn(".event-correction", css)
        self.assertIn("@media (max-width: 620px)", css)
        self.assertIn(".video-grid, .charts-section, .form-grid", css)
        self.assertIn("grid-template-columns: 1fr", css)
        self.assertIn("max-width: 100%", css)


if __name__ == "__main__":
    unittest.main()
