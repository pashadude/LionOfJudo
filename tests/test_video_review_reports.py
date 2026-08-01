import json
import tempfile
import unittest
from pathlib import Path

from pipeline.trainer_ai_state import migrate_trainer_ai_payload
from pipeline.video_review_reports import markdown_cell, report_rows, write_reports
from tests.test_trainer_ai_state import legacy_review_fixture


class VideoReviewReportTests(unittest.TestCase):
    @staticmethod
    def trainer_ai_review():
        review = migrate_trainer_ai_payload(legacy_review_fixture())
        event = review["events"][0]
        fingerprint = event["analysis_fingerprint"]
        event["trener_procene"] = [
            {
                "revizija": 1,
                "faza": "pre_ai",
                "event_revision": 1,
                "analysis_fingerprint": fingerprint,
                "status_vidljivosti": "dovoljno_vidljivo",
                "potvrdena_tehnika": "Tai-otoshi",
                "ocena": 4,
                "razlog": "Na 1.000 s kukovi kasne | ulaz je brz.\nDrugi red.",
                "citirani_sony_trenuci_s": [1.0],
                "zakljucano_u": "2026-08-01T12:00:00+02:00",
            }
        ]
        event["aktivna_trener_revizija"] = 1
        return review

    def test_markdown_cell_normalizes_mixed_newlines_and_escapes_backslashes_first(self):
        rendered = markdown_cell("putanja\\snimak | prvi\r\ndrugi\rtreci\n<kraj>")

        self.assertEqual(
            rendered,
            "putanja\\\\snimak \\| prvi<br>drugi<br>treci<br>&lt;kraj&gt;",
        )

    def test_markdown_report_keeps_mixed_content_inside_one_table_row(self):
        review = {
            "events": [
                {
                    "event_id": "e-1",
                    "sony_start_s": 10.0,
                    "sony_end_s": 12.0,
                    "napomena": "linija 1\r\nlinija 2 | C:\\video\rlinija 3",
                    "iskljuceno_iz_statistike": False,
                }
            ]
        }
        with tempfile.TemporaryDirectory() as raw:
            review_path = Path(raw) / "review.json"
            write_reports(review_path, review)
            markdown = review_path.with_name("izvestaj.md").read_text(encoding="utf-8")

        self.assertNotIn("\r", markdown)
        self.assertIn("linija 1<br>linija 2 \\| C:\\\\video<br>linija 3", markdown)
        self.assertEqual(sum(line.startswith("| e-1 |") for line in markdown.splitlines()), 1)

    def test_report_redacts_ai_and_imu_until_active_ai_is_revealed(self):
        review = self.trainer_ai_review()

        hidden = report_rows(review)[0]

        self.assertEqual(hidden["Trener pre-AI ocena"], 4)
        self.assertEqual(hidden["AI ocena"], "")
        self.assertEqual(hidden["AI razlog"], "")
        self.assertEqual(hidden["AI dokazi (JSON)"], "")
        self.assertEqual(hidden["IMU eksperimentalno (JSON)"], "")
        internal = report_rows(review, include_unrevealed=True)[0]
        self.assertNotEqual(internal["AI razlog"], "")
        self.assertNotEqual(internal["IMU eksperimentalno (JSON)"], "")

    def test_revealed_report_keeps_ai_evidence_and_all_trainer_revisions(self):
        review = self.trainer_ai_review()
        event = review["events"][0]
        fingerprint = event["analysis_fingerprint"]
        event["ai_procene"][0]["ai_otkriven_u"] = "2026-08-01T12:01:00+02:00"
        event["trener_procene"].append(
            {
                **event["trener_procene"][0],
                "revizija": 2,
                "faza": "post_ai_korekcija",
                "ocena": 5,
                "razlog": "Na 1.500 s završetak je stabilniji.",
                "citirani_sony_trenuci_s": [1.5],
                "zakljucano_u": "2026-08-01T12:02:00+02:00",
            }
        )
        event["aktivna_trener_revizija"] = 2
        event["aktivni_duel"] = {
            "event_revision": 1,
            "analysis_fingerprint": fingerprint,
            "trener_revizija": 2,
            "evaluator_id": "deterministicki-v1",
        }
        event["procene_ai_predloga"] = [
            {
                **event["aktivni_duel"],
                "odnos": "delimicno",
                "razlog": "Rotacija je jasna, trenutak ulaska nije.",
                "procene_dokaza": [
                    {"metrika": "ugaona_brzina_trupa_2d", "odnos": "prihvatam"}
                ],
                "sacuvano_u": "2026-08-01T12:03:00+02:00",
            }
        ]

        row = report_rows(review)[0]

        self.assertEqual(row["AI evaluator"], "deterministicki-v1")
        self.assertEqual(row["AI ocena"], event["ai_procene"][0]["predlozena_ocena"] or "")
        self.assertEqual(json.loads(row["AI dokazi (JSON)"]), event["ai_procene"][0]["dokazi"])
        self.assertEqual(json.loads(row["IMU eksperimentalno (JSON)"]), event["imu_eksperimentalno"])
        revisions = json.loads(row["Trener procene (JSON)"])
        self.assertEqual([item["faza"] for item in revisions], ["pre_ai", "post_ai_korekcija"])
        self.assertEqual(row["Odnos trenera prema AI"], "delimicno")
        self.assertEqual(row["Razlog odnosa prema AI"], "Rotacija je jasna, trenutak ulaska nije.")
        self.assertEqual(row["Odgovor sačuvan u"], "2026-08-01T12:03:00+02:00")
        feedback_rows = json.loads(row["Procene AI predloga (JSON)"])
        self.assertEqual(feedback_rows[0]["event_revision"], 1)
        self.assertEqual(feedback_rows[0]["analysis_fingerprint"], fingerprint)
        self.assertEqual(feedback_rows[0]["trener_revizija"], 2)
        self.assertEqual(feedback_rows[0]["evaluator_id"], "deterministicki-v1")

    def test_injury_row_never_exports_ai_or_trainer_values(self):
        injury = {
            "event_id": "povreda",
            "sony_start_s": 2.0,
            "sony_end_s": 2.5,
            "prijavljen_povredni_dogadjaj": True,
            "iskljuceno_iz_statistike": True,
            "status": "povreda",
            "potvrdena_tehnika": "ne sme izaći",
            "ocena": 5,
            "napomena": "ne sme izaći",
            "trener_procene": [{"razlog": "ne sme izaći"}],
            "procene_ai_predloga": [{"razlog": "ne sme izaći"}],
            "imu_eksperimentalno": {"izvor": "ne sme izaći"},
        }

        row = report_rows({"events": [injury]}, include_unrevealed=True)[0]

        for field in (
            "Potvrđena tehnika",
            "Ocena",
            "Napomena",
            "AI evaluator",
            "AI ocena",
            "AI razlog",
            "AI dokazi (JSON)",
            "IMU eksperimentalno (JSON)",
            "Trener pre-AI ocena",
            "Trener procene (JSON)",
            "Odnos trenera prema AI",
            "Procene AI predloga (JSON)",
        ):
            self.assertEqual(row[field], "", field)


if __name__ == "__main__":
    unittest.main()
