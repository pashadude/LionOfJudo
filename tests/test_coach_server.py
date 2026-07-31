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
        status, session = self.read_json(server.base_url + "/api/session")
        self.assertEqual(status, 200)
        self.assertEqual(session["session_id"], "demo")
        status, event = self.read_json(server.base_url + "/api/events/e-1")
        self.assertEqual(status, 200)
        self.assertEqual(event["event_id"], "e-1")

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

    def test_sync_requires_two_confirmed_anchors_and_respects_injury_event(self):
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
        with self.assertRaises(HTTPError) as raised:
            self.read_json(
                server.base_url + "/api/session/sync",
                method="POST",
                body=json.dumps({"anchors": anchors, "injury_cutoff_s": 20.1}).encode(),
            )
        self.assertEqual(raised.exception.code, 400)
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
        request = Request(server.base_url + "/media/clip.mp4", headers={"Range": "bytes=2-5"})
        with urlopen(request) as response:
            self.assertEqual(response.status, 206)
            self.assertEqual(response.read(), b"2345")
            self.assertEqual(response.headers["Content-Type"], "video/mp4")
            self.assertEqual(response.headers["Content-Range"], "bytes 2-5/10")
            self.assertEqual(response.headers["Accept-Ranges"], "bytes")

    def test_static_ui_contains_serbian_contract_and_chart_hooks(self):
        server = self.start_server()
        with urlopen(server.base_url + "/") as response:
            html = response.read().decode("utf-8")
        self.assertIn("Događaji", html)
        self.assertIn("Sinhronizacija", html)
        for label in (
            "Predlog tehnike",
            "Potvrđena tehnika",
            "Ocena",
            "Napomena",
            "Sačuvaj",
            "Podeli",
            "Spoji",
            "Obriši",
            "Izvezi izveštaj",
            "Prijavljen povredni događaj",
            "Nedovoljno vidljivo",
        ):
            self.assertIn(label, html)
        self.assertEqual(html.count("<canvas"), 5)
        for asset in ("/static/app.js", "/static/styles.css"):
            with urlopen(server.base_url + asset) as response:
                self.assertEqual(response.status, 200)
                self.assertTrue(response.read())


if __name__ == "__main__":
    unittest.main()
